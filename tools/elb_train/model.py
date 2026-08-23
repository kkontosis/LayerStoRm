"""EPM-3 predictor v1 (Phase 29, spec/MoE-SpeQ_NOTES.md §3/§5) — PyTorch.

Standalone torch tool (out of the engine build).  INV-EPM-SIDE holds by
construction: nothing here touches the engine, the draft checkpoint, or any
tensor a token path consumes; the frozen target routers are non-trainable
REPLICAS (buffers with requires_grad=False).

Architecture (decisions (C)/(D), 2026-07-09):

  features [B, G, L, H]  (raw draft per-backbone-layer residual hiddens,
                          BF16-expanded to f32 by the caller)
    -> input normalization        none | rmsnorm (per-tap learnable gain) |
                                  standardize (train-set per-(tap, dim)
                                  mean/inv_std stored as buffers -> saved
                                  in the checkpoint)
    -> tap selection              fixed per-layer tap map (default = the
                                  aux-segment prior) OR Tier-2 5-tap FUSION
                                  (per-layer-group softmax mix over all L
                                  taps + optional per-tap low-rank residual
                                  projection, zero-init)
    -> per-layer input x_in [B, G, J, H]
    -> arm "router" (PRIMARY):    Tier-1 side adapters
         adapted = residual*x_in + base_g(j)(x_in) + delta_{k,g(j)}(x_in)
         logits  = adapted @ router_j.T        (frozen router replicas)
       arm "direct" (flagged):    trained low-rank head, no router
         logits  = base_head_g(j)(x_in) + delta_head_{k,g(j)}(x_in)
                   [+ learnable per-layer logit bias]
    -> logits [B, G, J, E]  (pre-top-k router-logit scale — the KL target)

Shared-base + per-position deltas (decision (C)): base adapters are
per-layer-group low-rank maps (rank r_base); position deltas are low-rank
maps whose rank schedule r[k, group] comes from the W[k][j] value map via
wmap.allocate_ranks (capacity follows W).  All low-rank pairs are
LoRA-init (down ~ N(0, 1/sqrt(H)) seeded, up = 0), so at init the model
IS Tier-0 (router arm with residual) / emits zero logits (direct arm),
and an optional bottleneck activation (adapter.activation) provides the
Tier-2 nonlinear arm.

Selection semantics stay noaux_tc: predict() ranks by sigmoid(logits) +
e_score_correction_bias via glm_router.rank_experts (router arm always;
direct arm when use_router_bias_for_selection), identical to the labels'
pipeline and the EPM-2 harness.

Checkpoint contract: safetensors weights (minimal writer/reader below —
no safetensors package dependency) + JSON sidecar carrying the full model
config, W spec + built matrix, resolved tap map, normalization kind/stats
provenance, seeds, and delta rank plan.  build_model(sidecar, ...) must
reconstruct the exact module tree from the sidecar alone (given the
router source), so load is config-file independent.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import glm_router, wmap as wmap_mod

# ── dims ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Dims:
    """Problem dimensions (from the shard index)."""
    hidden: int
    n_taps: int
    n_experts: int
    max_gamma: int
    moe_layers: np.ndarray            # int32 [J]

    @property
    def n_layers(self) -> int:
        return len(self.moe_layers)

    @classmethod
    def from_index(cls, index: dict) -> "Dims":
        return cls(hidden=int(index["hidden"]),
                   n_taps=int(index["n_draft_layers"]),
                   n_experts=int(index["n_experts"]),
                   max_gamma=int(index["max_gamma"]),
                   moe_layers=np.asarray(index["moe_layers"], np.int32))


# ── config resolution helpers ────────────────────────────────────────────────

def resolve_tap_map(taps_cfg: dict, dims: Dims) -> np.ndarray:
    """Per-layer tap indices int32 [J] from config: map = "prior" (the
    aux-segment prior — the default; EPM-2's real best-tap sweep is noise
    at n=25 and is NOT baked in), an int (same tap everywhere), or an
    explicit per-layer list."""
    m = taps_cfg.get("map", "prior")
    if m == "prior":
        taps = glm_router.aux_prior_tap(dims.moe_layers)
    elif isinstance(m, int):
        taps = np.full(dims.n_layers, m, np.int32)
    else:
        taps = np.asarray(m, np.int32)
        if taps.shape != (dims.n_layers,):
            raise ValueError(f"taps.map must have {dims.n_layers} entries")
    if np.any(taps < 0) or np.any(taps >= dims.n_taps):
        raise ValueError(f"tap indices out of range [0, {dims.n_taps})")
    return taps.astype(np.int32)


def resolve_layer_groups(groups_cfg, dims: Dims,
                         tap_map: np.ndarray) -> np.ndarray:
    """group_of_layer int64 [J].  "per_layer" (default: every layer its own
    adapter), "per_segment" (share within each aux-prior tap segment),
    "single", or an explicit list of absolute-layer-id lists."""
    j = dims.n_layers
    if groups_cfg in (None, "per_layer"):
        return np.arange(j, dtype=np.int64)
    if groups_cfg == "single":
        return np.zeros(j, np.int64)
    if groups_cfg == "per_segment":
        prior = glm_router.aux_prior_tap(dims.moe_layers)
        _, inv = np.unique(prior, return_inverse=True)
        return inv.astype(np.int64)
    # explicit: list of lists of absolute layer ids
    gol = np.full(j, -1, np.int64)
    layer_slot = {int(l): i for i, l in enumerate(dims.moe_layers)}
    for g, layers in enumerate(groups_cfg):
        for l in layers:
            if int(l) not in layer_slot:
                raise ValueError(f"layer_groups: unknown layer {l}")
            if gol[layer_slot[int(l)]] != -1:
                raise ValueError(f"layer_groups: layer {l} in two groups")
            gol[layer_slot[int(l)]] = g
    if np.any(gol < 0):
        missing = [int(dims.moe_layers[i]) for i in np.nonzero(gol < 0)[0]]
        raise ValueError(f"layer_groups: layers {missing} unassigned")
    return gol


_ACTIVATIONS = {"none": lambda x: x, "silu": torch.nn.functional.silu,
                "gelu": torch.nn.functional.gelu, "relu": torch.relu,
                "tanh": torch.tanh}


# ── modules ──────────────────────────────────────────────────────────────────

class InputNorm(nn.Module):
    """Configurable input normalization over H (applied per tap).

    kind "none":        identity.
    kind "rmsnorm":     x / rms(x) * gain; gain per-tap [L, H] (learnable
                        unless learnable=false, then fixed ones).
    kind "standardize": (x - mean) * inv_std with per-(tap, dim) train-set
                        stats; buffers default to identity (mean 0 /
                        inv_std 1) until set_standardize_stats() — the
                        trainer owns the stats pass; stats_set records
                        whether they were filled (asserted at forward).
    """

    def __init__(self, cfg: dict, dims: Dims) -> None:
        super().__init__()
        self.kind = cfg.get("kind", "none")
        self.eps = float(cfg.get("eps", 1e-6))
        if self.kind == "rmsnorm":
            gain = torch.ones(dims.n_taps, dims.hidden)
            if cfg.get("learnable", True):
                self.gain = nn.Parameter(gain)
            else:
                self.register_buffer("gain", gain)
        elif self.kind == "standardize":
            self.register_buffer("mean",
                                 torch.zeros(dims.n_taps, dims.hidden))
            self.register_buffer("inv_std",
                                 torch.ones(dims.n_taps, dims.hidden))
            self.register_buffer("stats_set",
                                 torch.zeros((), dtype=torch.int64))
        elif self.kind != "none":
            raise ValueError(f"unknown normalization kind {self.kind!r}")

    def set_standardize_stats(self, mean: np.ndarray,
                              std: np.ndarray) -> None:
        if self.kind != "standardize":
            raise ValueError("not a standardize norm")
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        inv = 1.0 / np.sqrt(np.asarray(std, np.float64) ** 2 + self.eps)
        self.inv_std.copy_(torch.as_tensor(inv, dtype=torch.float32))
        self.stats_set.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, G, L, H]
        if self.kind == "none":
            return x
        if self.kind == "rmsnorm":
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * rms * self.gain
        if not int(self.stats_set):
            raise RuntimeError("standardize stats not set — run the "
                               "trainer stats pass (or load a checkpoint "
                               "that carries them)")
        return (x - self.mean) * self.inv_std


class TapFusion(nn.Module):
    """Tier-2 5-tap fusion: per-layer-group softmax mix over ALL taps
    (+ optional per-tap low-rank residual projection, zero-init so the
    init is exactly the plain mix).  Mix logits init to +init_gain on the
    group's prior tap -> the init is a near-one-hot prior-tap selection
    (soft Tier-0-compatible starting point)."""

    def __init__(self, cfg: dict, dims: Dims, group_of_layer: np.ndarray,
                 tap_map: np.ndarray, gen: torch.Generator) -> None:
        super().__init__()
        n_groups = int(group_of_layer.max()) + 1
        init_gain = float(cfg.get("init_gain", 4.0))
        mix = torch.zeros(n_groups, dims.n_taps)
        for g in range(n_groups):
            layers = np.nonzero(group_of_layer == g)[0]
            # group's init tap = the tap map's entry of its first layer
            mix[g, int(tap_map[layers[0]])] = init_gain
        self.mix_logits = nn.Parameter(mix)
        self.rank = int(cfg.get("rank", 0))
        if self.rank > 0:
            down = torch.empty(dims.n_taps, self.rank, dims.hidden)
            nn.init.normal_(down, std=dims.hidden ** -0.5, generator=gen)
            self.proj_down = nn.Parameter(down)
            self.proj_up = nn.Parameter(
                torch.zeros(dims.n_taps, dims.hidden, self.rank))
        self.register_buffer(
            "group_of_layer",
            torch.as_tensor(group_of_layer, dtype=torch.int64),
            persistent=False)

    def forward(self, xn: torch.Tensor) -> torch.Tensor:
        # xn [B, G, L, H] -> x_in [B, G, J, H]
        if self.rank > 0:
            h = torch.einsum("bglh,lrh->bglr", xn, self.proj_down)
            xn = xn + torch.einsum("bglr,lhr->bglh", h, self.proj_up)
        mix = torch.softmax(self.mix_logits, dim=-1)     # [n_g, L]
        mix_l = mix.index_select(0, self.group_of_layer)  # [J, L]
        return torch.einsum("bglh,jl->bgjh", xn, mix_l)


class _DeltaBank(nn.Module):
    """Per-position low-rank deltas with a W-derived rank schedule.

    ranks int [G, n_groups]; parameters are stored per (position k,
    distinct rank r) bucket (down [m_g, r, in], up [m_g, out, r]) so
    same-rank groups batch into one einsum.  Forward adds each bucket's
    contribution into out[:, k] via index_add."""

    def __init__(self, ranks: np.ndarray, group_of_layer: np.ndarray,
                 in_dim: int, out_dim: int, activation: str,
                 gen: torch.Generator) -> None:
        super().__init__()
        self.act = _ACTIVATIONS[activation]
        self.plan: list[dict] = []
        self.down = nn.ParameterDict()
        self.up = nn.ParameterDict()
        g_pos, _ = ranks.shape
        for k in range(g_pos):
            for r in sorted(set(int(x) for x in ranks[k] if x > 0)):
                gs = np.nonzero(ranks[k] == r)[0]
                gset = {int(g): i for i, g in enumerate(gs)}
                layers = np.nonzero(np.isin(group_of_layer, gs))[0]
                pidx = np.array([gset[int(group_of_layer[l])]
                                 for l in layers], np.int64)
                key = f"k{k}_r{r}"
                down = torch.empty(len(gs), r, in_dim)
                nn.init.normal_(down, std=in_dim ** -0.5, generator=gen)
                self.down[key] = nn.Parameter(down)
                self.up[key] = nn.Parameter(torch.zeros(len(gs), out_dim,
                                                        r))
                self.plan.append({"k": k, "key": key,
                                  "layers": torch.as_tensor(layers),
                                  "pidx": torch.as_tensor(pidx)})

    def add_to(self, out: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
        """out, x_in: [B, G, J, ·]; returns out + delta contributions
        (out is mutated in place per bucket — it is a fresh intermediate,
        never an autograd leaf)."""
        g_avail = x_in.shape[1]
        dev = x_in.device
        for e in self.plan:
            k = e["k"]
            if k >= g_avail:
                continue
            layers = e["layers"].to(dev)
            pidx = e["pidx"].to(dev)
            xk = x_in[:, k].index_select(1, layers)      # [B, m, in]
            dn = self.down[e["key"]].index_select(0, pidx)
            up = self.up[e["key"]].index_select(0, pidx)
            h = self.act(torch.einsum("bmh,mrh->bmr", xk, dn))
            contrib = torch.einsum("bmr,mor->bmo", h, up)
            out[:, k] = out[:, k].index_add(1, layers, contrib)
        return out

    def rank_table(self) -> dict:
        return {e["key"]: int(self.down[e["key"]].shape[1])
                for e in self.plan}


class EpmPredictor(nn.Module):
    """The EPM-3 predictor (module docstring architecture).  Construct via
    build_model(); forward(features_f32 [B, G', L, H]) -> logits
    [B, G', J, E] with G' <= max_gamma."""

    def __init__(self, model_cfg: dict, dims: Dims, w: np.ndarray,
                 router_bank: glm_router.RouterBank | None,
                 seed: int) -> None:
        super().__init__()
        self.cfg = model_cfg
        self.dims = dims
        self.arm = model_cfg.get("arm", "router")
        if self.arm not in ("router", "direct"):
            raise ValueError(f"unknown arm {self.arm!r}")
        self.seed = int(seed)
        gen = torch.Generator().manual_seed(int(seed))

        # value map (buffer: single source of truth for the loss weights)
        if w.shape != (dims.max_gamma, dims.n_layers):
            raise ValueError(f"wmap shape {w.shape} != "
                             f"({dims.max_gamma}, {dims.n_layers})")
        self.register_buffer("wmap",
                             torch.as_tensor(w, dtype=torch.float32))
        self.register_buffer(
            "moe_layers",
            torch.as_tensor(dims.moe_layers.astype(np.int64)),
            persistent=False)

        # normalization + taps
        self.norm = InputNorm(model_cfg.get("normalization",
                                            {"kind": "none"}), dims)
        taps_cfg = model_cfg.get("taps", {"mode": "fixed"})
        self.tap_mode = taps_cfg.get("mode", "fixed")
        tap_map = resolve_tap_map(taps_cfg, dims)
        self._tap_map_np = tap_map
        gol = resolve_layer_groups(model_cfg.get("layer_groups",
                                                 "per_layer"), dims,
                                   tap_map)
        self._group_of_layer_np = gol
        self.n_groups = int(gol.max()) + 1
        self.register_buffer("group_of_layer",
                             torch.as_tensor(gol), persistent=False)
        if self.tap_mode == "fusion":
            self.fusion = TapFusion(taps_cfg, dims, gol, tap_map, gen)
        elif self.tap_mode == "fixed":
            self.fusion = None
            self.register_buffer(
                "tap_map", torch.as_tensor(tap_map.astype(np.int64)),
                persistent=False)
        else:
            raise ValueError(f"unknown taps.mode {self.tap_mode!r}")

        # frozen router replicas (non-trainable buffers, NOT persisted —
        # the bank lives in the run dir's routers.npz; INV-EPM-SIDE)
        need_bank = (self.arm == "router"
                     or model_cfg.get("direct_head", {}).get(
                         "use_router_bias_for_selection", True))
        if need_bank:
            if router_bank is None:
                raise ValueError("this configuration needs a RouterBank")
            if (router_bank.n_experts != dims.n_experts
                    or router_bank.hidden != dims.hidden
                    or not np.array_equal(router_bank.moe_layers,
                                          dims.moe_layers)):
                raise ValueError("RouterBank dims/layers != shard dims")
            self.register_buffer(
                "router_weight",
                torch.as_tensor(router_bank.weight, dtype=torch.float32),
                persistent=False)
            self.register_buffer(
                "router_bias",
                torch.as_tensor(router_bank.bias, dtype=torch.float32),
                persistent=False)
        else:
            self.router_weight = None
            self.router_bias = None

        # adapters / heads
        h, e = dims.hidden, dims.n_experts
        if self.arm == "router":
            acfg = model_cfg.get("adapter", {})
            out_dim = h
        else:
            acfg = model_cfg.get("direct_head", {})
            out_dim = e
        self.residual = bool(acfg.get("residual", True))
        self.activation = acfg.get("activation", "none")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}")
        self._act = _ACTIVATIONS[self.activation]
        r_base = int(acfg.get("r_base", 64))
        if r_base > 0:
            down = torch.empty(self.n_groups, r_base, h)
            nn.init.normal_(down, std=h ** -0.5, generator=gen)
            self.base_down = nn.Parameter(down)
            self.base_up = nn.Parameter(torch.zeros(self.n_groups, out_dim,
                                                    r_base))
        else:
            self.base_down = self.base_up = None
        self.r_base = r_base

        dcfg = acfg.get("delta", {})
        if dcfg.get("enabled", False):
            ranks = wmap_mod.allocate_ranks(
                w, gol, budget_params=int(dcfg["budget_params"]),
                in_dim=h, out_dim=out_dim,
                min_rank=int(dcfg.get("min_rank", 0)),
                max_rank=int(dcfg.get("max_rank", 256)))
            self.delta = _DeltaBank(ranks, gol, h, out_dim,
                                    self.activation, gen)
            self._delta_ranks = ranks
        else:
            self.delta = None
            self._delta_ranks = None

        if self.arm == "direct" and acfg.get("logit_bias", True):
            self.logit_bias = nn.Parameter(torch.zeros(dims.n_layers, e))
        else:
            self.logit_bias = None

        # b0_prev-HYBRID prior (EPM-5, DECISION.md 'EPM as a re-ranker ON
        # TOP of temporal locality'). When enabled, forward() adds
        # beta_j * prev_membership to the trained logits, where
        # prev_membership marks the previous decoded position's top-K per
        # (layer, expert). init_beta high ⇒ the head STARTS at b0_prev's
        # recall and learns discriminative residual logits that
        # re-rank/prune the temporal candidate set (b0_prev supplies the
        # union-cov 0.881 candidate set; precision is what the head adds).
        pcfg = model_cfg.get("prior", {})
        self.prior_enabled = bool(pcfg.get("enabled", False))
        if self.prior_enabled:
            init_beta = float(pcfg.get("init_beta", 3.0))
            per = pcfg.get("per", "layer")
            if per == "scalar":
                self.prior_beta = nn.Parameter(torch.full((1,), init_beta))
            elif per == "layer":
                self.prior_beta = nn.Parameter(
                    torch.full((dims.n_layers,), init_beta))
            else:
                raise ValueError(f"unknown prior.per {per!r}")
            self._prior_per = per
        else:
            self.prior_beta = None

    # ── forward ──────────────────────────────────────────────────────────

    def _layer_inputs(self, features: torch.Tensor) -> torch.Tensor:
        xn = self.norm(features)                          # [B, G, L, H]
        if self.fusion is not None:
            return self.fusion(xn)                        # [B, G, J, H]
        return xn.index_select(2, self.tap_map)

    def _low_rank(self, x_in: torch.Tensor) -> torch.Tensor:
        """base + position deltas: [B, G, J, H] -> [B, G, J, out]."""
        if self.base_down is not None:
            dn = self.base_down.index_select(0, self.group_of_layer)
            up = self.base_up.index_select(0, self.group_of_layer)
            hmid = self._act(torch.einsum("bgjh,jrh->bgjr", x_in, dn))
            out = torch.einsum("bgjr,jor->bgjo", hmid, up)
        else:
            b, g, j, _ = x_in.shape
            o = (self.dims.hidden if self.arm == "router"
                 else self.dims.n_experts)
            out = x_in.new_zeros(b, g, j, o)
        if self.delta is not None:
            out = self.delta.add_to(out, x_in)
        return out

    def _apply_prior(self, logits: torch.Tensor,
                     prior_membership: torch.Tensor | None) -> torch.Tensor:
        """Add beta_j * prev_membership to the trained logits (hybrid arm).

        prior_membership: float [B, G, J, E] (1 where the previous decoded
        position routed to expert e on layer j, else 0). No-op when the
        prior is disabled or no membership is supplied (e.g. a shard set
        without prev_top_ids sidecars — the arm degrades to the base head).
        """
        if not self.prior_enabled or prior_membership is None:
            return logits
        beta = self.prior_beta
        if self._prior_per == "layer":
            beta = beta[None, None, :, None]              # [1,1,J,1]
        else:
            beta = beta.view(1, 1, 1, 1)
        return logits + beta * prior_membership

    def forward(self, features: torch.Tensor,
                prior_membership: torch.Tensor | None = None) -> torch.Tensor:
        if features.dim() != 4 or features.shape[1] > self.dims.max_gamma:
            raise ValueError(f"features must be [B, G<= "
                             f"{self.dims.max_gamma}, L, H]")
        x_in = self._layer_inputs(features)
        lo = self._low_rank(x_in)
        if self.arm == "router":
            adapted = x_in + lo if self.residual else lo
            logits = torch.einsum("bgjh,jeh->bgje", adapted,
                                  self.router_weight)
        else:
            logits = lo
            if self.logit_bias is not None:
                logits = logits + self.logit_bias
        return self._apply_prior(logits, prior_membership)

    # ── prediction (EPM-2 harness semantics) ─────────────────────────────

    def selection_bias(self) -> np.ndarray | None:
        """The e_score_correction_bias used at selection (noaux_tc): the
        frozen router bias for the router arm (always) and for the direct
        arm when use_router_bias_for_selection (default true)."""
        if self.arm == "router":
            return self.router_bias.detach().cpu().numpy()
        if (self.cfg.get("direct_head", {}).get(
                "use_router_bias_for_selection", True)
                and self.router_bias is not None):
            return self.router_bias.detach().cpu().numpy()
        return None

    @torch.no_grad()
    def predict(self, features: torch.Tensor, m_rank: int = 32,
                prior_membership: torch.Tensor | None = None,
                ) -> tuple[np.ndarray, np.ndarray]:
        """(ids int32 [B, G, J, m_rank], scores f32 aligned) — ranked by
        the noaux_tc selection score via glm_router (identical semantics
        to the labels' engine pipeline and the EPM-2 baselines); scores
        are the UNBIASED sigmoid scores of the ranked experts."""
        logits = self(features, prior_membership).detach().cpu().numpy()
        ids = glm_router.rank_experts(logits, self.selection_bias(),
                                      m_rank)
        sc = np.take_along_axis(glm_router.stable_sigmoid(logits), ids,
                                axis=-1)
        return ids, sc

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def delta_rank_plan(self) -> list | None:
        return (None if self._delta_ranks is None
                else self._delta_ranks.tolist())


def build_model(model_cfg: dict, dims: Dims, wspec: dict | wmap_mod.WMapSpec,
                router_bank: glm_router.RouterBank | None,
                seed: int) -> EpmPredictor:
    w = wmap_mod.build_wmap(wspec, dims.max_gamma, dims.moe_layers)
    model = EpmPredictor(model_cfg, dims, w, router_bank, seed)
    for buf in model.buffers():          # explicit INV-EPM-SIDE guard:
        buf.requires_grad_(False)        # router replicas stay frozen
    return model


# ── minimal safetensors I/O (format-compatible; no safetensors package) ──────

_TORCH_TO_ST = {torch.float32: "F32", torch.float16: "F16",
                torch.bfloat16: "BF16", torch.int32: "I32",
                torch.int64: "I64", torch.uint8: "U8", torch.bool: "BOOL"}
_ST_TO_NP = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16,
             "I32": np.int32, "I64": np.int64, "U8": np.uint8,
             "BOOL": np.bool_}


def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.uint16)
    return t.numpy().tobytes()


def save_safetensors(tensors: dict[str, torch.Tensor],
                     path: str | Path) -> None:
    """Write a safetensors file (sorted keys, deterministic bytes)."""
    header: dict = {}
    blobs: list[bytes] = []
    off = 0
    for name in sorted(tensors):
        t = tensors[name]
        if t.dtype not in _TORCH_TO_ST:
            raise ValueError(f"{name}: unsupported dtype {t.dtype}")
        raw = _tensor_bytes(t)
        header[name] = {"dtype": _TORCH_TO_ST[t.dtype],
                        "shape": list(t.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hdr = json.dumps(header, sort_keys=True).encode()
    pad = (8 - len(hdr) % 8) % 8            # 8-byte alignment (spec-legal)
    hdr += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for raw in blobs:
            f.write(raw)


def load_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    data = Path(path).read_bytes()
    n = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8:8 + n])
    base = 8 + n
    out: dict[str, torch.Tensor] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        beg, end = meta["data_offsets"]
        arr = np.frombuffer(data, _ST_TO_NP[meta["dtype"]],
                            count=(end - beg)
                            // np.dtype(_ST_TO_NP[meta["dtype"]]).itemsize,
                            offset=base + beg).copy()
        t = torch.from_numpy(arr).reshape(meta["shape"])
        if meta["dtype"] == "BF16":
            t = t.view(torch.bfloat16)
        out[name] = t
    return out


# ── checkpoint save/load (weights + sidecar) ─────────────────────────────────

def save_model(model: EpmPredictor, path_prefix: str | Path,
               sidecar_extra: dict | None = None) -> None:
    """Write <prefix>.safetensors (persistent state dict: trainable params
    + norm-stat/wmap buffers; router replicas are NOT persisted — the run
    dir's routers.npz is their source) + <prefix>.sidecar.json (everything
    needed to rebuild the module tree)."""
    prefix = Path(path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(dict(model.state_dict()),
                     str(prefix) + ".safetensors")
    sidecar = {
        "model_cfg": model.cfg,
        "dims": {"hidden": model.dims.hidden, "n_taps": model.dims.n_taps,
                 "n_experts": model.dims.n_experts,
                 "max_gamma": model.dims.max_gamma,
                 "moe_layers": [int(l) for l in model.dims.moe_layers]},
        "wmap_matrix": model.wmap.detach().cpu().numpy().tolist(),
        "tap_map": [int(t) for t in model._tap_map_np],
        "tap_mode": model.tap_mode,
        "layer_groups": [int(g) for g in model._group_of_layer_np],
        "delta_ranks": model.delta_rank_plan(),
        "normalization": model.cfg.get("normalization", {"kind": "none"}),
        "arm": model.arm,
        "seed": model.seed,
        "num_trainable_params": model.num_trainable_params(),
        "torch_version": torch.__version__,
    }
    if sidecar_extra:
        sidecar.update(sidecar_extra)
    with open(str(prefix) + ".sidecar.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=1)


def load_model(path_prefix: str | Path,
               router_bank: glm_router.RouterBank | None,
               device: str = "cpu") -> tuple[EpmPredictor, dict]:
    """Rebuild the model from <prefix>.sidecar.json + load
    <prefix>.safetensors.  Returns (model, sidecar)."""
    prefix = Path(path_prefix)
    with open(str(prefix) + ".sidecar.json", encoding="utf-8") as f:
        sidecar = json.load(f)
    d = sidecar["dims"]
    dims = Dims(hidden=d["hidden"], n_taps=d["n_taps"],
                n_experts=d["n_experts"], max_gamma=d["max_gamma"],
                moe_layers=np.asarray(d["moe_layers"], np.int32))
    w = np.asarray(sidecar["wmap_matrix"], np.float32)
    model = EpmPredictor(sidecar["model_cfg"], dims, w, router_bank,
                         seed=int(sidecar.get("seed", 0)))
    state = load_safetensors(str(prefix) + ".safetensors")
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model, sidecar
