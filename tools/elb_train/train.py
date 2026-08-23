"""EPM-3 training tool (Phase 29) — config-driven, deterministic, on
CACHED features (no draft forward in the loop; INV-EPM-SIDE by
construction: this is a standalone torch tool).

Usage:
  python3 tools/elb_train/train.py --config tools/elb_train/configs/X.json
      [--out DIR]                # overrides cfg.out_dir
      [--resume CKPT_PREFIX|auto]
      [--set key.path=json ...]  # dotted config overrides (EPM-4 sweeps
                                 # stay config-file driven; --set is for
                                 # quick variants)
  python3 tools/elb_train/train.py --config X.json --evaluate \
      --checkpoint CKPT_PREFIX [--eval-split held_out|train|all] [--out D]

Data: assembled EPM-1 shards (dataset.assemble_shards).  Shard npz members
are memory-mapped when stored uncompressed (np.savez default) — parse the
zip local headers + npy headers and np.memmap the raw array bytes; falls
back to np.load per shard otherwise.  Splits are SEQUENCE-level
(dataset.sequence_split, INV-EPM-DATA); `data.in_sample: true` is the
explicit degenerate-corpus escape hatch (stamped on every artifact).

Losses (decision (A)):
  KL   (primary, weight loss.kl_weight, temperature loss.kl_temperature):
       KL(softmax(labels/T) || softmax(pred/T)) * T^2 over the E experts,
       computed on the RAW pre-bias router logits.  Semantics decision —
       KL is over the SOFTMAX of raw logits, NOT the sigmoid-score
       selection distribution: (i) sigmoid scores are per-expert
       INDEPENDENT Bernoullis, not a categorical distribution — a
       "sigmoid KL" is exactly a per-expert soft BCE (available as the
       BCE arm); softmax-KL adds the cross-expert competition/ordering
       signal that top-k selection depends on; (ii) noaux_tc selects by
       sigmoid(logit)+bias with the SAME frozen bias on the prediction
       and label paths, so ranking fidelity reduces to logit-ordering
       fidelity, which softmax-KL directly optimizes; (iii) caveat:
       softmax is shift-invariant while sigmoid is not, so KL alone pins
       logits only up to an additive constant (which can perturb
       biased-score near-ties and the unbiased gate WEIGHTS) — the BCE
       and/or MSE arms anchor the absolute scale when that matters.
  BCE  (auxiliary/fallback, loss.bce_weight): multi-label BCE-with-logits
       on top-8 MEMBERSHIP (targets from labels_top_ids), mean over E —
       matches the router's sigmoid score semantics.
  MSE  (optional arm, loss.mse_weight, default 0): plain MSE on the raw
       logits (the B2-skyline objective; the exact-scale anchor).
  Every per-cell loss is weighted by W[k][j] * label_mask
  * token-mismatch weight; the total is sum(w*loss)/sum(w).

Token-mismatch weighting (TD-EPM-REJECTED-LABELS semantics): a labeled
cell (block, k) is TOKEN-MATCHED iff the manifest joined (accepted_len >=
0), tokens_match, and k <= accepted_len — those labels were computed on
the exact draft tokens (verify feeds the draft prefix INCLUDING the first
rejected draft token, so k == accepted_len is matched).  For k >
accepted_len the label is the ACCEPTED stream's routing at that position
(a different token — though arguably the RIGHT prefetch target).  Default
trains on ALL labeled cells (data.token_mismatch_weight = 1.0); set it in
[0, 1) to down-weight or 0.0 to exclude mismatched cells.  Blocks without
a manifest join (accepted_len == -1, e.g. C++-golden smoke dumps) have
unknown provenance: data.unknown_provenance_matched (default true) decides
which side they fall on.

Checkpoints: <ckpt>/model.safetensors + model.sidecar.json (arch config,
W spec + matrix, tap map, normalization stats, seeds, data provenance,
rank plan) + optim.safetensors (AdamW state).  Resume is bitwise: the LR
schedule is a pure function of the step, data order is a pure function of
(seed, micro_step), and optimizer state round-trips exactly.

Metric-of-record: EPM-2's recall@8 via metrics.EvalAccumulator; periodic
and final evals emit the standard npz/json artifact contract
(metrics.save_result), so run_baselines-style comparison works directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):  # `python3 tools/elb_train/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from elb_train import dataset, glm_router, metrics, model as model_mod
    from elb_train import synth_corpus, wmap as wmap_mod
else:
    from . import dataset, glm_router, metrics, model as model_mod
    from . import synth_corpus, wmap as wmap_mod


# ── config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    "seed": 0,
    "device": "cpu",
    "out_dir": "build/epm3-train",
    "data": {
        "shards": None,
        "held_out_fraction": 0.1,
        "split_seed": 0,
        "in_sample": False,
        "batch_blocks": 8,
        "token_mismatch_weight": 1.0,
        "unknown_provenance_matched": True,
        # optional: "synthetic": {...} -> generate a synth corpus first
    },
    "router": {"checkpoint_dir": None, "npz": None},
    "model": {
        "arm": "router",
        "normalization": {"kind": "rmsnorm", "learnable": True,
                          "eps": 1e-6},
        "taps": {"mode": "fixed", "map": "prior"},
        "layer_groups": "per_layer",
        "adapter": {"r_base": 64, "residual": True, "activation": "none",
                    "delta": {"enabled": True, "budget_params": 60000000,
                              "min_rank": 0, "max_rank": 96}},
        "direct_head": {"r_base": 64, "logit_bias": True,
                        "use_router_bias_for_selection": True,
                        "residual": False, "activation": "none",
                        "delta": {"enabled": True,
                                  "budget_params": 30000000,
                                  "min_rank": 0, "max_rank": 96}},
    },
    "wmap": {"kind": "directive"},
    "loss": {"kl_weight": 1.0, "kl_temperature": 1.0, "bce_weight": 0.25,
             "mse_weight": 0.0},
    "optim": {"lr": 3e-4, "weight_decay": 0.01, "betas": [0.9, 0.999],
              "schedule": "cosine", "warmup_steps": 100,
              "min_lr_ratio": 0.0, "grad_accum": 1, "clip_grad_norm": 1.0,
              "max_steps": 20000},
    "eval": {"every_steps": 1000, "m_rank": 32},
    "checkpoint": {"every_steps": 2000, "keep": 3},
    "log_every": 25,
}


def _deep_update(base: dict, upd: dict) -> dict:
    """Recursive dict merge returning a FULLY fresh tree (nested dicts of
    `base` are copied even when untouched, so later in-place edits — e.g.
    --set overrides — can never mutate DEFAULT_CONFIG)."""
    out = {k: (_deep_update(v, {}) if isinstance(v, dict) else v)
           for k, v in base.items()}
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        elif isinstance(v, dict):
            out[k] = _deep_update(v, {})
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: list[str] | None = None,
                ) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = _deep_update(DEFAULT_CONFIG, json.load(f))
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        if not _:
            raise ValueError(f"--set needs key=value, got {ov!r}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            node[parts[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[parts[-1]] = val
    return cfg


# ── memory-mapped shard access ───────────────────────────────────────────────

def open_shard_mmap(path: str | Path,
                    keys: set[str] | None = None) -> dict[str, np.ndarray]:
    """Open an npz shard with its members memory-mapped (zero-copy reads
    into the page cache).  np.savez stores members uncompressed
    (ZIP_STORED): parse each local file header for the data offset, the
    npy header for shape/dtype, and np.memmap the raw bytes.  Falls back
    to a plain np.load (full read) for compressed members.

    ``keys`` (optional): load ONLY these members.  Load-bearing for
    COMPRESSED shards (savez_compressed) at 100k-block scale — decompressing
    every member (esp. the all-zero 4.1 GB/shard labels_logits when
    logits_stored=False) blows the process RAM to hundreds of GB
    (TD-EPM-LOADER-MEM). np.load(compressed) is lazy per member, so
    restricting the fallback access to the needed keys never touches the
    others."""
    path = Path(path)
    out: dict[str, np.ndarray] = {}
    fallback: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = info.filename[:-4] if info.filename.endswith(".npy") \
                else info.filename
            if keys is not None and name not in keys:
                continue
            if info.compress_type != zipfile.ZIP_STORED:
                fallback.append(name)
                continue
            with open(path, "rb") as f:
                f.seek(info.header_offset)
                hdr = f.read(30)
                if hdr[:4] != b"PK\x03\x04":
                    raise ValueError(f"{path}:{info.filename}: bad local "
                                     f"zip header")
                name_len, extra_len = struct.unpack("<HH", hdr[26:30])
                npy_off = info.header_offset + 30 + name_len + extra_len
                f.seek(npy_off)
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = \
                        np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran, dtype = \
                        np.lib.format.read_array_header_2_0(f)
                data_off = f.tell()
            if fortran:
                fallback.append(name)
                continue
            if int(np.prod(shape)) == 0:
                out[name] = np.zeros(shape, dtype)
            else:
                out[name] = np.memmap(path, dtype=dtype, mode="r",
                                      offset=data_off, shape=shape)
    if fallback:
        z = np.load(path)
        for name in fallback:
            out[name] = z[name]
    return out


class CachedShards:
    """Random-access cached-feature store over an assembled shard dir.

    rows(seq_keys) -> list of (shard_idx, row_idx) for one split;
    batch(rows) -> stacked numpy dict for those blocks (padded [b, G, ...]
    arrays exactly as stored)."""

    _KEYS = ("features_bf16", "labels_logits", "labels_top_ids",
             "label_mask", "feature_mask", "gamma", "accepted_len",
             "tokens_match", "seq_key")

    def __init__(self, shard_dir: str | Path) -> None:
        self.shard_dir = Path(shard_dir)
        self.index = dataset.load_index(shard_dir)
        self._shards: list[dict[str, np.ndarray] | None] = \
            [None] * len(self.index["shards"])
        # TD-EPM-LOADER-MEM: on a COMPRESSED logits-free corpus never load
        # labels_logits — it is an all-zero 4.1 GB/shard array whose
        # decompression (x ~110 train shards) blows RAM to hundreds of GB.
        # BCE reads labels_top_ids; KL is disabled on such a corpus
        # (Trainer fails closed). Load only the members the loss + batch use.
        self._logits_stored = bool(self.index.get("logits_stored", True))
        self._load_keys = set(self._KEYS)
        if not self._logits_stored:
            self._load_keys.discard("labels_logits")

    def shard(self, i: int) -> dict[str, np.ndarray]:
        if self._shards[i] is None:
            d = open_shard_mmap(
                self.shard_dir / self.index["shards"][i]["path"],
                keys=self._load_keys)
            # b0_prev-hybrid: attach the prev-membership sidecar when it
            # exists (dataset.build_prev_membership_sidecars). Absent ⇒
            # the prior arm degrades to the base head (forward no-ops).
            side = self.shard_dir / f"prev_top_ids_{i:05d}.npy"
            if side.is_file():
                d["prev_top_ids"] = np.load(side, mmap_mode="r")
            self._shards[i] = d
        return self._shards[i]

    @property
    def has_prev_membership(self) -> bool:
        return (self.shard_dir / "prev_top_ids_00000.npy").is_file()

    def _seq_keys(self, i: int) -> np.ndarray:
        """Just this shard's seq_key column — WITHOUT decompressing/caching
        the whole shard (rows() over the split would otherwise pull every
        train shard's features + labels into RAM up front)."""
        d = open_shard_mmap(self.shard_dir / self.index["shards"][i]["path"],
                            keys={"seq_key"})
        return d["seq_key"].astype(np.int64)

    def rows(self, seq_keys) -> list[tuple[int, int]]:
        keys = {int(s) for s in seq_keys}
        rows: list[tuple[int, int]] = []
        for i, sh in enumerate(self.index["shards"]):
            if not keys.intersection(sh["seq_keys"]):
                continue
            sk = self._seq_keys(i)
            for r in np.nonzero(np.isin(sk, list(keys)))[0]:
                rows.append((i, int(r)))
        return rows

    def batch(self, rows: list[tuple[int, int]]) -> dict[str, np.ndarray]:
        want = tuple(k for k in self._KEYS if k in self._load_keys) + (
            ("prev_top_ids",) if self.has_prev_membership else ())
        cols: dict[str, list] = {k: [] for k in want}
        for i, r in rows:
            sh = self.shard(i)
            for k in want:
                cols[k].append(np.asarray(sh[k][r]))
        return {k: np.stack(v) for k, v in cols.items()}


# ── per-cell weights + losses ────────────────────────────────────────────────

def token_match_mask(batch: dict, unknown_matched: bool) -> np.ndarray:
    """bool [B, G]: labeled cell's label tokens match the draft feature
    tokens (module docstring semantics)."""
    b, g = batch["label_mask"].shape
    acc = batch["accepted_len"].astype(np.int64)          # [B]
    tm = batch["tokens_match"].astype(bool)               # [B]
    ks = np.arange(g)[None, :]
    matched = tm[:, None] & (ks <= acc[:, None])
    unknown = acc < 0
    if unknown_matched:
        matched = matched | unknown[:, None]
    else:
        matched = matched & ~unknown[:, None]
    return matched


def cell_weights(batch: dict, wmap_t: torch.Tensor, data_cfg: dict,
                 device) -> torch.Tensor:
    """[B, G, J] = W[k][j] * label_mask * token-mismatch weight."""
    lm = torch.as_tensor(batch["label_mask"], dtype=torch.float32,
                         device=device)                   # [B, G]
    mw = float(data_cfg.get("token_mismatch_weight", 1.0))
    matched = token_match_mask(
        batch, bool(data_cfg.get("unknown_provenance_matched", True)))
    pos_w = torch.where(
        torch.as_tensor(matched, device=device),
        torch.ones((), device=device),
        torch.full((), mw, device=device))                # [B, G]
    g = lm.shape[1]
    return (lm * pos_w)[:, :, None] * wmap_t[None, :g]    # [B, G, J]


def compute_loss(pred: torch.Tensor, batch: dict, w_cell: torch.Tensor,
                 loss_cfg: dict) -> tuple[torch.Tensor, dict]:
    """Weighted (KL, BCE, MSE) combination; returns (total, components).
    Zero-weight cells contribute exactly zero gradient (they are
    multiplied out before the reduction)."""
    device = pred.device
    wsum = w_cell.sum()
    comps: dict[str, float] = {}
    total = pred.new_zeros(())
    if wsum <= 0:
        return total, {"loss": 0.0, "wsum": 0.0}
    kl_w = float(loss_cfg.get("kl_weight", 1.0))
    mse_w = float(loss_cfg.get("mse_weight", 0.0))
    # labels_logits is only materialized for the KL/MSE targets (logits-full
    # corpora); on a logits-free corpus it is not loaded and the Trainer
    # fails closed if kl_weight>0 (MSE on zero logits is meaningless too).
    labels = None
    if kl_w > 0 or mse_w > 0:
        labels = torch.as_tensor(np.ascontiguousarray(batch["labels_logits"]),
                                 dtype=torch.float32, device=device)
    if kl_w > 0:
        t = float(loss_cfg.get("kl_temperature", 1.0))
        logp = torch.log_softmax(labels / t, dim=-1)
        logq = torch.log_softmax(pred / t, dim=-1)
        kl_cell = (logp.exp() * (logp - logq)).sum(-1) * (t * t)
        kl = (w_cell * kl_cell).sum() / wsum
        total = total + kl_w * kl
        comps["kl"] = float(kl.detach())
    bce_w = float(loss_cfg.get("bce_weight", 0.0))
    if bce_w > 0:
        tids = torch.as_tensor(
            np.ascontiguousarray(batch["labels_top_ids"]),
            dtype=torch.int64, device=device)             # [B, G, J, K]
        tgt = torch.zeros_like(pred)
        tgt.scatter_(-1, tids.clamp(min=0), 1.0)
        # -1 pads scatter into expert 0; those rows are zero-weighted
        # (unlabeled) except top-1-truncated labels, which the shards
        # never produce (topk is uniform per corpus).
        bce_cell = torch.nn.functional.binary_cross_entropy_with_logits(
            pred, tgt, reduction="none").mean(-1)
        bce = (w_cell * bce_cell).sum() / wsum
        total = total + bce_w * bce
        comps["bce"] = float(bce.detach())
    if mse_w > 0:
        mse_cell = (pred - labels).pow(2).mean(-1)
        mse = (w_cell * mse_cell).sum() / wsum
        total = total + mse_w * mse
        comps["mse"] = float(mse.detach())
    comps["loss"] = float(total.detach())
    comps["wsum"] = float(wsum.detach())
    return total, comps


# ── LR schedule (pure function of step — resume-safe) ────────────────────────

def lr_at(step: int, optim_cfg: dict) -> float:
    base = float(optim_cfg["lr"])
    warm = int(optim_cfg.get("warmup_steps", 0))
    if warm > 0 and step < warm:
        return base * (step + 1) / warm
    if optim_cfg.get("schedule", "cosine") == "constant":
        return base
    total = max(1, int(optim_cfg["max_steps"]) - warm)
    prog = min(1.0, max(0.0, (step - warm) / total))
    min_r = float(optim_cfg.get("min_lr_ratio", 0.0))
    return base * (min_r + (1.0 - min_r)
                   * 0.5 * (1.0 + np.cos(np.pi * prog)))


# ── evaluation (EPM-2 harness) ───────────────────────────────────────────────

def evaluate_model(model: model_mod.EpmPredictor, shard_dir: str | Path,
                   seq_keys, m_rank: int = 32,
                   batch_blocks: int = 8) -> dict:
    """Run the model over one split through metrics.EvalAccumulator —
    the metric-of-record path (recall@8 primary).  Deterministic."""
    index = dataset.load_index(shard_dir)
    moe_layers = np.asarray(index["moe_layers"], np.int32)
    acc = metrics.EvalAccumulator(index["max_gamma"], moe_layers,
                                  topk=index["topk"])
    model.eval()
    device = next(model.parameters()).device
    pend: list[dict] = []

    def flush() -> None:
        if not pend:
            return
        g = max(b["gamma"] for b in pend)
        feats = np.zeros((len(pend), g, index["n_draft_layers"],
                          index["hidden"]), np.float32)
        for i, b in enumerate(pend):
            feats[i, :b["gamma"]] = glm_router.bf16_bits_to_f32(
                b["features_bf16"])
        x = torch.as_tensor(feats, device=device)
        prior = None
        if getattr(model, "prior_enabled", False) and "prev_top_ids" in pend[0]:
            j = len(index["moe_layers"])
            k = index["topk"]
            pt = np.full((len(pend), g, j, k), -1, np.int64)
            for i, b in enumerate(pend):
                pt[i, :b["gamma"]] = b["prev_top_ids"]
            tids = torch.as_tensor(pt, device=device)
            prior = torch.zeros((len(pend), g, j, index["n_experts"]),
                                device=device)
            prior.scatter_(-1, tids.clamp(min=0),
                           (tids >= 0).to(prior.dtype))
        ids, sc = model.predict(x, m_rank=m_rank, prior_membership=prior)
        for i, b in enumerate(pend):
            gb = b["gamma"]
            acc.add_block(ids[i, :gb], b["labels_top_ids"],
                          b["label_mask"], pred_scores=sc[i, :gb])
        pend.clear()

    with torch.no_grad():
        for b in dataset.iterate_blocks(shard_dir, seq_keys=seq_keys):
            pend.append(b)
            if len(pend) >= batch_blocks:
                flush()
        flush()
    return acc.result()


# ── standardize stats pass ───────────────────────────────────────────────────

def compute_standardize_stats(store: CachedShards,
                              rows: list[tuple[int, int]],
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Streaming per-(tap, dim) mean/std over the train split's features
    (feature_mask positions only)."""
    index = store.index
    l, h = index["n_draft_layers"], index["hidden"]
    s = np.zeros((l, h), np.float64)
    s2 = np.zeros((l, h), np.float64)
    n = 0
    for i, r in rows:
        sh = store.shard(i)
        fm = np.asarray(sh["feature_mask"][r], bool)
        f = glm_router.bf16_bits_to_f32(
            np.asarray(sh["features_bf16"][r]))[fm].astype(np.float64)
        s += f.sum(axis=0)
        s2 += (f * f).sum(axis=0)
        n += int(fm.sum())
    if n == 0:
        raise ValueError("no features in the train split")
    mean = s / n
    var = np.maximum(0.0, s2 / n - mean * mean)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


# ── trainer ──────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class TrainState:
    micro_step: int = 0     # micro-batches consumed (grad-accum units)
    step: int = 0           # optimizer steps taken


class Trainer:
    def __init__(self, cfg: dict, out_dir: str | Path | None = None,
                 router_bank: glm_router.RouterBank | None = None,
                 store: "CachedShards | None" = None) -> None:
        self.cfg = cfg
        self.out = Path(out_dir or cfg["out_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = cfg.get("device", "cpu")
        torch.manual_seed(int(cfg["seed"]))

        # ``store`` (optional): a pre-built CachedShards to REUSE across
        # many Trainers (the in-process EPM-5 arms grid — decompressing
        # ~110 compressed shards once instead of per-cell-subprocess,
        # TD-EPM-LOADER-MEM). When given, data.synthetic is ignored.
        if store is not None:
            self.store = store
        else:
            self._maybe_generate_synth()
            shards = cfg["data"]["shards"]
            if not shards:
                raise ValueError("data.shards not set (and no data.synthetic)")
            self.store = CachedShards(shards)
        self.dims = model_mod.Dims.from_index(self.store.index)

        # EPM-5 disk-bounded corpora may be assembled WITHOUT the raw
        # router logits (dataset.assemble_shards store_logits=False —
        # labels_logits is all-zero, index stamped). The KL loss would
        # silently distill against zeros: fail closed.
        if (not self.store.index.get("logits_stored", True)
                and float(cfg.get("loss", {}).get("kl_weight", 1.0)) > 0):
            raise ValueError(
                "corpus was assembled with store_logits=False "
                "(shards_index logits_stored=false) but loss.kl_weight > 0 "
                "— the KL target would be all-zero logits. Use a "
                "BCE-primary loss config (kl_weight=0) on this corpus.")

        # split (INV-EPM-DATA)
        all_keys = sorted({k for sh in self.store.index["shards"]
                           for k in sh["seq_keys"]})
        if cfg["data"].get("in_sample", False):
            self.train_keys = self.held_keys = list(all_keys)
        else:
            self.train_keys, self.held_keys = dataset.sequence_split(
                all_keys, cfg["data"].get("held_out_fraction", 0.1),
                seed=int(cfg["data"].get("split_seed", 0)))
            if not self.train_keys or not self.held_keys:
                raise ValueError(
                    f"degenerate split ({len(self.train_keys)} train / "
                    f"{len(self.held_keys)} held-out sequences) — set "
                    f"data.in_sample=true for smoke corpora (stamped)")
        self.train_rows = self.store.rows(self.train_keys)
        if not self.train_rows:
            raise ValueError("train split has no blocks")

        # router bank
        self.bank = router_bank if router_bank is not None \
            else self._load_bank()
        if self.bank is not None:
            self.bank.save_npz(self.out / "routers.npz")

        # model
        self.wspec = wmap_mod.WMapSpec.from_dict(cfg["wmap"])
        self.model = model_mod.build_model(
            cfg["model"], self.dims, self.wspec, self.bank,
            seed=int(cfg["seed"]))
        if (self.model.norm.kind == "standardize"
                and not int(self.model.norm.stats_set)):
            mean, std = compute_standardize_stats(self.store,
                                                  self.train_rows)
            self.model.norm.set_standardize_stats(mean, std)
        self.model.to(self.device)

        # optimizer
        ocfg = cfg["optim"]
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=float(ocfg["lr"]),
            betas=tuple(ocfg.get("betas", (0.9, 0.999))),
            weight_decay=float(ocfg.get("weight_decay", 0.0)))
        self.state = TrainState()
        self._log_f = open(self.out / "train_log.jsonl", "a",
                           encoding="utf-8")

    def _maybe_generate_synth(self) -> None:
        scfg = self.cfg["data"].get("synthetic")
        if not scfg:
            return
        work = self.out / "synth-corpus"
        if not (work / "shards" / "shards_index.json").is_file():
            truth = synth_corpus.generate_synthetic_run(
                work / "dump" / "run_1", **scfg)
            dataset.assemble_shards(work / "dump", work / "shards")
            truth["routers"].save_npz(work / "routers.npz")
            with open(work / "truth.json", "w", encoding="utf-8") as f:
                json.dump({"true_taps":
                           [int(t) for t in truth["true_taps"]],
                           "moe_layers": truth["moe_layers"],
                           "n_blocks": truth["n_blocks"]}, f, indent=1)
        self.cfg["data"]["shards"] = str(work / "shards")
        self.cfg["router"] = {"npz": str(work / "routers.npz"),
                              "checkpoint_dir": None}

    def _load_bank(self) -> glm_router.RouterBank | None:
        rcfg = self.cfg.get("router", {})
        if rcfg.get("npz"):
            return glm_router.RouterBank.from_npz(rcfg["npz"])
        if rcfg.get("checkpoint_dir"):
            return glm_router.RouterBank.from_checkpoint(
                rcfg["checkpoint_dir"], self.dims.moe_layers)
        return None

    # ── deterministic data order: micro-batch = f(seed, micro_step) ──────

    def _micro_batch_rows(self, micro_step: int) -> list[tuple[int, int]]:
        bs = int(self.cfg["data"].get("batch_blocks", 8))
        n = len(self.train_rows)
        per_epoch = max(1, (n + bs - 1) // bs)
        epoch = micro_step // per_epoch
        pos = micro_step % per_epoch
        rng = np.random.default_rng(
            [int(self.cfg["seed"]), 0x45504D33, epoch])
        perm = (rng.permutation(n)
                if self.cfg["data"].get("shuffle", True)
                else np.arange(n))
        sel = perm[pos * bs:(pos + 1) * bs]
        return [self.train_rows[i] for i in sel]

    def _batch_tensors(self, rows) -> tuple[torch.Tensor, dict]:
        batch = self.store.batch(rows)
        feats = glm_router.bf16_bits_to_f32(
            np.ascontiguousarray(batch["features_bf16"]))
        x = torch.as_tensor(feats, device=self.device)
        return x, batch

    def _prior_membership(self, batch: dict) -> torch.Tensor | None:
        """[B, G, J, E] float membership from batch['prev_top_ids'] (the
        b0_prev candidate set), or None when the model has no prior arm or
        the shards carry no sidecar. Built once per batch on device."""
        if not getattr(self.model, "prior_enabled", False):
            return None
        pt = batch.get("prev_top_ids")
        if pt is None:
            return None
        tids = torch.as_tensor(np.ascontiguousarray(pt),
                               dtype=torch.int64, device=self.device)
        mem = torch.zeros((*tids.shape[:-1], self.dims.n_experts),
                          device=self.device)
        mem.scatter_(-1, tids.clamp(min=0),
                     (tids >= 0).to(mem.dtype))          # -1 pad → no mark
        return mem

    # ── train loop ───────────────────────────────────────────────────────

    def train(self, max_steps: int | None = None) -> dict:
        ocfg = self.cfg["optim"]
        accum = int(ocfg.get("grad_accum", 1))
        max_steps = int(max_steps if max_steps is not None
                        else ocfg["max_steps"])
        last: dict = {}
        self.model.train()
        while self.state.step < max_steps:
            lr = lr_at(self.state.step, ocfg)
            for grp in self.optim.param_groups:
                grp["lr"] = lr
            self.optim.zero_grad(set_to_none=True)
            comps_acc: dict[str, float] = {}
            for _ in range(accum):
                rows = self._micro_batch_rows(self.state.micro_step)
                x, batch = self._batch_tensors(rows)
                pred = self.model(x, self._prior_membership(batch))
                w_cell = cell_weights(batch, self.model.wmap,
                                      self.cfg["data"], self.device)
                loss, comps = compute_loss(pred, batch, w_cell,
                                           self.cfg["loss"])
                (loss / accum).backward()
                self.state.micro_step += 1
                for k, v in comps.items():
                    comps_acc[k] = comps_acc.get(k, 0.0) + v / accum
            clip = float(ocfg.get("clip_grad_norm", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               clip)
            self.optim.step()
            self.state.step += 1
            last = {"step": self.state.step, "lr": lr, **comps_acc}
            if self.state.step % int(self.cfg.get("log_every", 25)) == 0 \
                    or self.state.step == max_steps:
                self._log(last)
            ev = int(self.cfg["eval"].get("every_steps", 0))
            if ev and self.state.step % ev == 0 \
                    and self.state.step < max_steps:
                self.eval_and_save(f"eval_step{self.state.step:07d}")
                self.model.train()
            ck = int(self.cfg["checkpoint"].get("every_steps", 0))
            if ck and self.state.step % ck == 0:
                self.save_checkpoint()
        self.save_checkpoint()
        return last

    def _log(self, rec: dict) -> None:
        rec = {**rec, "ts": time.time()}
        self._log_f.write(json.dumps(rec) + "\n")
        self._log_f.flush()
        msg = "  ".join(f"{k}={v:.6g}" if isinstance(v, float)
                        else f"{k}={v}" for k, v in rec.items()
                        if k != "ts")
        print(f"[epm3] {msg}", flush=True)

    # ── eval + artifacts ─────────────────────────────────────────────────

    def eval_and_save(self, name: str, split: str = "held_out") -> dict:
        keys = {"held_out": self.held_keys, "train": self.train_keys,
                "all": sorted(set(self.train_keys)
                              | set(self.held_keys))}[split]
        res = evaluate_model(self.model, self.cfg["data"]["shards"], keys,
                             m_rank=int(self.cfg["eval"].get("m_rank",
                                                             32)))
        meta = {"split": split, "step": self.state.step,
                "in_sample": bool(self.cfg["data"].get("in_sample",
                                                       False)),
                "provenance": f"EPM-3 {name}"}
        if meta["in_sample"]:
            meta["WARNING"] = ("IN-SAMPLE: train == eval sequences — "
                               "pipeline-proof numbers only")
        metrics.save_result(res, self.out / name, meta=meta)
        r8, n8 = metrics.overall_recall(res, 8)
        self._log({"step": self.state.step, "eval": name,
                   "recall8": r8, "n8": n8})
        return res

    # ── checkpointing ────────────────────────────────────────────────────

    def _ckpt_dir(self, step: int) -> Path:
        return self.out / f"ckpt_step{step:07d}"

    def save_checkpoint(self) -> Path:
        d = self._ckpt_dir(self.state.step)
        d.mkdir(parents=True, exist_ok=True)
        extra = {
            "train_state": dataclasses.asdict(self.state),
            "config": self.cfg,
            "wmap_spec": self.wspec.to_dict(),
            "data_provenance": {
                "shards": str(self.cfg["data"]["shards"]),
                "in_sample": bool(self.cfg["data"].get("in_sample",
                                                       False)),
                "split_seed": int(self.cfg["data"].get("split_seed", 0)),
                "held_out_fraction":
                    float(self.cfg["data"].get("held_out_fraction", 0.1)),
                "train_sequences": len(self.train_keys),
                "held_out_sequences": len(self.held_keys),
                "train_blocks": len(self.train_rows),
            },
            "router_source": str(self.out / "routers.npz")
            if self.bank is not None else None,
        }
        model_mod.save_model(self.model, d / "model", sidecar_extra=extra)
        # optimizer state (exact tensor round trip)
        sd = self.optim.state_dict()
        tensors: dict[str, torch.Tensor] = {}
        meta: dict = {"param_groups": sd["param_groups"], "scalars": {}}
        for pid, st in sd["state"].items():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    tensors[f"p{pid}.{k}"] = v
                else:
                    meta["scalars"][f"p{pid}.{k}"] = v
        model_mod.save_safetensors(tensors, d / "optim.safetensors")
        with open(d / "optim.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        (self.out / "latest").write_text(d.name, encoding="utf-8")
        self._prune_checkpoints()
        return d

    def _prune_checkpoints(self) -> None:
        keep = int(self.cfg["checkpoint"].get("keep", 0))
        if keep <= 0:
            return
        dirs = sorted(p for p in self.out.glob("ckpt_step*")
                      if p.is_dir())
        for p in dirs[:-keep]:
            shutil.rmtree(p)

    def resume(self, ckpt: str | Path) -> None:
        d = Path(ckpt)
        if str(ckpt) == "auto":
            latest = (self.out / "latest")
            if not latest.is_file():
                raise FileNotFoundError(f"{latest}: no checkpoint to "
                                        f"resume from")
            d = self.out / latest.read_text(encoding="utf-8").strip()
        state = model_mod.load_safetensors(d / "model.safetensors")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        with open(d / "model.sidecar.json", encoding="utf-8") as f:
            sidecar = json.load(f)
        ts = sidecar["train_state"]
        self.state = TrainState(micro_step=int(ts["micro_step"]),
                                step=int(ts["step"]))
        with open(d / "optim.json", encoding="utf-8") as f:
            ometa = json.load(f)
        tensors = model_mod.load_safetensors(d / "optim.safetensors")
        osd: dict = {"param_groups": ometa["param_groups"], "state": {}}
        for key, t in tensors.items():
            pid, _, field = key.partition(".")
            osd["state"].setdefault(int(pid[1:]), {})[field] = \
                t.to(self.device) if field != "step" else t
        for key, v in ometa["scalars"].items():
            pid, _, field = key.partition(".")
            osd["state"].setdefault(int(pid[1:]), {})[field] = v
        self.optim.load_state_dict(osd)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="EPM-3 predictor training tool (see module docstring)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None,
                    help="checkpoint dir (or 'auto' for <out>/latest)")
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    metavar="KEY.PATH=JSON")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--evaluate", action="store_true",
                    help="no training: evaluate --checkpoint on "
                         "--eval-split and write the EPM-2 artifacts")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint dir for --evaluate")
    ap.add_argument("--eval-split", default="held_out",
                    choices=("held_out", "train", "all"))
    ap.add_argument("--eval-name", default="eval_final")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    tr = Trainer(cfg, out_dir=args.out)

    if args.evaluate:
        ckpt = args.checkpoint or "auto"
        tr.resume(ckpt)
        res = tr.eval_and_save(args.eval_name, split=args.eval_split)
        r8, n8 = metrics.overall_recall(res, 8)
        print(f"[epm3] {args.eval_split} recall@8 = {r8:.4f} (n={n8}) -> "
              f"{tr.out / args.eval_name}.json")
        return 0

    if args.resume:
        tr.resume(args.resume)
    tr.train(max_steps=args.max_steps)
    tr.eval_and_save(args.eval_name, split=("all" if cfg["data"].get(
        "in_sample") else "held_out"))
    print(f"[epm3] done: step {tr.state.step}, artifacts in {tr.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
