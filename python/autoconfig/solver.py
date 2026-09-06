"""The autoconfig solver (AUTOCONFIG §5) — derivation ladder P1..M1.

Deterministic throughout: no randomness, no wall-clock, lowest-index
tie-breaks (the I8 house rule). Closed-form where the relation is closed
form, documented heuristics where it is judgement, and one small
deterministic degradation search (E1) where expert/KV/draft genuinely
couple. Engine-version-dependent rules come from engine_constraints (DATA —
deleting a row re-derives the config); measured speed comes from the
gpu_loader calibration artifact, never card names.
"""

from __future__ import annotations

import contextlib
import math
from typing import Callable
import os
from dataclasses import dataclass, field, replace

from . import combo_constraints as cc
from . import engine_constraints as ec
from . import sizing, templates
from .calibration import CalibrationView
from .explain import ExplanationSet, Infeasible
from .hwdetect import HardwareDescriptor, GpuInfo
from .modelshape import ModelShape
from .pins import Pin, PinSet

GIB = sizing.GIB


PREFERENCES = ("speed", "balanced", "capacity")

# TD-AUTOCONFIG-ACCURACY-LEVER — the 4th human-facing dial.  Tiers, not
# weights: tok/s and perplexity have no honest exchange rate (the same
# reason TD-AUTOCONFIG-SPEED-BUDGET rejected a scalar), so a tier FLOORS
# the numerics ladder (templates.ACCURACY_LADDER) instead of weighing it.
#   compact  — the most KV-byte-efficient serveable backend (all-TQ end)
#   standard — the family template's champion-proven choice (DEFAULT;
#              byte-identical to the pre-lever derivation)
#   high     — the most accurate serveable backend (all-FP8 end)
#   superior — DEFINED, NOT IMPLEMENTED: full-precision KV (bf16, codec
#              kFull), no TQ anywhere, everything maximal.  The engine
#              builds no kFull device arm (kv_codec.h:12-14), so this tier
#              REFUSES at derive time with that fact — it exists so the
#              day a kFull arm lands, the lever already has its name.
ACCURACY_TIERS = ("compact", "standard", "high", "superior")

# Headline of registry row `mla-tq-vs-snapmla-accuracy` — kept in ONE place
# so the explain text and the row cannot drift.  Filled from the 2026-09-02
# teacher-forced A/B (scratchpad/accuracy_lever/); see engine_constraints.py
# for the full statement, evidence bound and falsifiers.
_MLA_ACCURACY_HEADLINE = (
    "teacher-forced dNLL +0.0282 +- 0.0063 nats/token, TQ worse (ppl 6.542 "
    "vs 6.360, +2.9%); top-1 acc 56.57% vs 57.17%; argmax agreement 92.31% "
    "with flips at low margin (median 0.17 vs 1.35 nats); gap flat in "
    "context to 7k")


@dataclass(frozen=True)
class Levers:
    """AUTOCONFIG §2 — the four human-facing dials."""
    vram_expert_ratio: float | None = None       # global, in slots (§2.1)
    total_active_context_tokens: int | None = None  # aggregate tokens (§2.2)
    prefer: str = "balanced"                     # search ORDER only (§2.3)
    accuracy: str = "standard"                   # numerics FLOOR (§2.4)


@dataclass(frozen=True)
class Attempt:
    """One point of the E1 search lattice (AUTOCONFIG §5 E1/§2.3).

    Three independent axes, each already ordered "free first" in its own
    terms.  A `prefer` value is nothing but a choice of which axis is the
    OUTER loop, so every preference enumerates the SAME SET of points — that
    identity is what makes a preference unable to cause a refusal another
    preference would have avoided (§6).

      tp_idx    PRICED: position in tp_plan(); >0 = a TP escalation, priced
                by registry row `glm5next-tp1-default` (~0.5-2% of the
                decode wall) and taken for capacity, never for speed.
      idx_idx   PRICED: position in the dcp_modes() candidate list; >0 =
                `local`, priced by `local-indexer-prefill-cost` (~8-9% of
                served prefill) for half the per-rank indexer-K share.
      cap_idx   the USER'S ASK: 0 = whole; then concurrency down, then
                max_sequence_length halved (§2.2's documented order).
      draft_off the optional accelerator, shed for its VRAM.
      tier_idx  PRICED (P-31 gap 2 interlock): position in the tiering
                candidate list. On glm5_next the measured row
                `glm5next-tiering-default-off` keeps tiering OFF while the
                untiered pool can hold the ask; when it cannot (the
                conc-ask admission check), tiering is the CAPACITY
                escalation (INV-KVT-16 windowed admission) — tried before
                any other axis moves (last element of every order key).
    """
    tp_idx: int
    idx_idx: int
    cap_idx: int
    draft_off: bool
    tp: int
    idx_mode: str
    conc: int
    max_seq: int
    note: str
    tier_idx: int = 0
    tiering_on: bool = False


def _chain_key(prefer: str, a: Attempt) -> tuple:
    """Order of the (draft x capacity) plane — everything that is NOT a
    priced registry option.

    `speed` treats the draft as speed (it is): it walks the whole capacity
    ladder with the draft ON before dropping it and walking it again.
    `balanced`/`capacity` keep the shipped chain — the draft goes BEFORE any
    capacity is shed (the ticket's 63-vs-64-slot lesson), which leaves the
    draft-kept capacity rungs at the end of the order.  Those trailing points
    exist only to keep the attempt SET identical across preferences; in this
    order they are unreachable, because the same capacity rung without the
    draft is strictly cheaper in VRAM and is tried first."""
    if prefer == "speed":
        return (1 if a.draft_off else 0, a.cap_idx)
    if a.cap_idx == 0:
        return (0, 1 if a.draft_off else 0)
    return (1 if a.draft_off else 2, a.cap_idx)


def _order_key(prefer: str, a: Attempt) -> tuple:
    """The whole of the `prefer` lever: which axis is the outer loop.

    No exchange rate, no weight, no percentage — tok/s and MiB have no
    meaningful conversion (TD-AUTOCONFIG-SPEED-BUDGET).  Order only.

      speed     (tp, idx, chain)  — exhaust every free capacity lever before
                                    buying ANY priced slowdown
      balanced  (tp, chain, idx)  — the shipped ladder: spend the indexer
                                    price at each rung, escalate TP last
      capacity  (chain, tp, idx)  — spend every priced slowdown before
                                    touching the user's ask
    """
    if prefer == "speed":
        return (a.tp_idx, a.idx_idx, _chain_key(prefer, a), a.tier_idx)
    if prefer == "capacity":
        return (_chain_key(prefer, a), a.tp_idx, a.idx_idx, a.tier_idx)
    return (a.tp_idx, _chain_key(prefer, a), a.idx_idx, a.tier_idx)


@dataclass(frozen=True)
class AutoconfigKnobs:
    """Internal solver policy (mirrors _internal-autoconfig schema)."""
    vram_usable_fraction: float = 0.9375   # 15/16: driver/runtime reserve
    vram_safety_margin_gb: float = 2.25    # champion-proven; KvTiering device
                                           # pools allocate out of this margin
    non_tp_overhead_gib: float = 1.25      # CUDA ctx + NCCL + staging on a
                                           # GPU with no pinned weights
    host_pin_fraction_total: float = 0.9
    hbm_spill_fraction_free: float = 0.6   # registry row hbm-fraction-free-06
    prefill_scratch_gb: float = 0.5
    min_seq_for_pairing: int = 16384       # K1: split into 2 requests above 2x this
    device_pool_round_pages: int = 1024
    expert_gib_quantum: float = 0.5
    min_max_seq: int = 4096                # degradation floor before refusal
    page_size_tokens: int = 16
    indexer_k_page_size_tokens: int = 8192
    dcp_chunk_size: int = 16
    kv_tiering_host_to_device_ratio: float = 8.0
    prefix_cache_max_entries: int = 8
    # -- P-31: long-context runtime-scratch + stride fit (gap 1/2/4) --
    # Free-at-fit anchors for the MoE-big elastic fit model (measured
    # P-30 step 2 on this box's classes: 5090 free 1668 MiB / 5080 1685
    # MiB at the champion carve; fixed = physical - block - prefit - free).
    moe_fit_fixed_attn_mib: int = 2224
    moe_fit_fixed_expert_mib: int = 1627
    moe_fit_headroom_cushion: float = 1.15  # over the post-check demand
    moe_fit_free_floor_mib: int = 256       # residual floor after allocs
    runtime_margin_quantum_gb: float = 0.25
    stride_margin_cap_gb: float = 6.0       # never spend more expert VRAM
    block_table_budget_gb: float = 1.5      # caps derived max_batch_size


@dataclass(frozen=True)
class DraftCandidate:
    checkpoint_path: str          # as written in the recipe (repo-relative ok)
    weight_file: str              # resolved safetensors/gguf path, "" if unresolved
    is_gguf: bool = False
    block_size: int = 16
    speculative_tokens: int = 15
    draft_layers: int = 5
    draft_kv_heads: int = 64
    draft_head_dim: int = 64
    draft_hidden: int = 6144
    draft_vocab: int = 154880
    ctx_cap_tokens: int = 8192
    # identity (TD-AUTOCONFIG-DRAFT-IDENTITY): the target hidden size the
    # draft was trained against (0 = its own transformer hidden size), and
    # the TARGET layer indices its aux-hidden-state taps read. Checked by
    # draft_identity_errors() BEFORE the draft is ever costed.
    target_hidden: int = 0
    aux_layer_ids: tuple = ()


@dataclass
class GpuPlan:
    ordinal: int
    gpu_type: str
    usable_bytes: int
    pinned_bytes: int = 0
    draft_bytes: int = 0
    margin_bytes: int = 0
    kda_bytes: int = 0
    expert_bytes: int = 0
    expert_slots: int = 0
    kv_pool_bytes: int = 0
    idx_share_bytes: int = 0
    scratch_bytes: int = 0
    # TD-AUTOCONFIG-MAXSEQ-IGNORES-MAPPED-KDA: mapped KDA state rides IN the
    # shared pool (kda_bytes is then shared capacity, not a dead carve), and
    # the plan carries the single-request admissibility arithmetic so the
    # solver can refuse-or-reduce and the explain sheet can show the margin.
    kda_mapped: bool = False
    adm_pool_pages: int = 0        # modeled kMain pool (KV + idx + state shares)
    adm_demand_pages: int = 0      # one max-length request's whole-life demand
    adm_kv_pages: int = 0
    adm_idx_pages: int = 0
    adm_state_pages: int = 0
    admissible_max_seq: int = 0    # largest admissible single-request context

    def total_committed(self) -> int:
        return (self.pinned_bytes + self.draft_bytes + self.margin_bytes
                + self.kda_bytes + self.expert_bytes + self.kv_pool_bytes
                + self.idx_share_bytes + self.scratch_bytes)

    def slack(self) -> int:
        return self.usable_bytes - self.total_committed()


@dataclass(frozen=True)
class TpSetup:
    """Per-TP-degree constants the E1 lattice is built on (P4/P5 + P6)."""
    tp: int
    kv_mode: str
    idx_modes: tuple
    tiering_candidates: tuple   # ordered free-first (False before True)
    draft_ranks: list
    draft_bytes: int


@dataclass(frozen=True)
class LongCtxPlan:
    """P-31: the max_seq/stride-scaled runtime plan for one lattice point.

    Derived BEFORE the carve so the TP margin can fund it: the schedule
    width (gap 4), the prefill superchunk stride S (gap 1, speed axis),
    the attention-host MoE-big fit headroom that reserves room for the
    post-check allocations (block tables et al.), and the TP-rank margin
    that funds all of it (gap 2)."""
    max_batch: int
    superchunk: int          # 0 = family template has no stride to derive
    headroom_attn_mib: int   # 0 = engine default suffices
    headroom_exp_mib: int    # 0 = engine default suffices
    margin_gb: float         # per TP rank (>= knob base)
    scratch: "sizing.RuntimeScratch | None"
    spill_mib: int           # MoE-big transient spill at S on an attn host
    notes: tuple = ()


@dataclass
class TpFit:
    """One accepted fit at a fixed TP degree (solver stage P2/E1)."""
    tp: int
    kv_mode: str
    idx_mode: str
    tiering_on: bool
    conc: int
    max_seq: int
    draft_on: bool
    draft_ranks: list
    plans: list


@dataclass
class SolveResult:
    recipe: dict
    explanations: ExplanationSet
    warnings: list[str]
    gpu_plans: list[GpuPlan]
    degradations: list[str]      # what E1 had to give up, in order


def load_draft_candidate(checkpoint_path: str, repo_root: str = ".") -> DraftCandidate | None:
    """Resolve + read a .dspark checkpoint's own config (CPU-only)."""
    import json
    for root in (repo_root, "."):
        p = os.path.join(root, checkpoint_path)
        if os.path.isdir(p):
            cfg_p = os.path.join(p, "config.json")
            st = os.path.join(p, "model.safetensors")
            gguf = not os.path.exists(st)
            wf = st if not gguf else ""
            block, spec_toks = 16, 15
            layers, kvh, hd, hid, vocab = 5, 64, 64, 6144, 154880
            tgt_hidden, aux_ids = 0, ()
            if os.path.exists(cfg_p):
                with open(cfg_p) as f:
                    dc = json.load(f)
                block = int(dc.get("block_size", 16))
                pm = ((dc.get("speculators_config") or {}).get("proposal_methods") or [{}])[0]
                spec_toks = int(pm.get("speculative_tokens", 15))
                tl = dc.get("transformer_layer_config") or {}
                layers = int(tl.get("num_hidden_layers", 5))
                kvh = int(tl.get("num_key_value_heads", 64))
                hd = int(tl.get("head_dim", 64))
                hid = int(tl.get("hidden_size", 6144))
                # identity fields (TD-AUTOCONFIG-DRAFT-IDENTITY): the draft
                # proposes TARGET tokens, so its vocab is the top-level
                # draft_vocab_size when present; target_hidden_size is often
                # null (== the draft transformer's own hidden size)
                vocab = int(dc.get("draft_vocab_size")
                            or tl.get("vocab_size", 154880))
                tgt_hidden = int(dc.get("target_hidden_size") or 0)
                aux_ids = tuple(int(i) for i in
                                (dc.get("aux_hidden_state_layer_ids") or ()))
            return DraftCandidate(
                checkpoint_path=checkpoint_path, weight_file=wf, is_gguf=gguf,
                block_size=block, speculative_tokens=spec_toks,
                draft_layers=layers, draft_kv_heads=kvh, draft_head_dim=hd,
                draft_hidden=hid, draft_vocab=vocab,
                target_hidden=tgt_hidden, aux_layer_ids=aux_ids)
    return None


def draft_identity_errors(draft: DraftCandidate, shape: ModelShape) -> list:
    """TD-AUTOCONFIG-DRAFT-IDENTITY — is this .dspark checkpoint THIS
    model's speculator?  Checked BEFORE the draft is costed: a foreign
    draft must never be silently priced (it once cost 9.14 GiB/rank for a
    GLM-5.2 draft against GLM-5.3-Flash and was only shed by capacity
    pressure — on a roomier box it would have booted).

    Matched observables (all CPU-only, from the checkpoint's config.json):
      - hidden size: the draft reads the TARGET's hidden states, so the
        target hidden size it was trained against (target_hidden_size, or
        its own transformer hidden size when null) must equal the model's;
      - vocabulary: the draft proposes TARGET token ids, so its vocab must
        equal the model's — this is also the tokenizer-identity proxy
        observable without booting anything;
      - aux taps: every aux_hidden_state_layer_ids index must name a layer
        the target actually has."""
    errs = []
    want_hidden = draft.target_hidden or draft.draft_hidden
    if want_hidden != shape.hidden_size:
        errs.append(f"draft target hidden size {want_hidden} != model "
                    f"hidden size {shape.hidden_size}")
    if draft.draft_vocab != shape.vocab_size:
        errs.append(f"draft vocab {draft.draft_vocab} != model vocab "
                    f"{shape.vocab_size} (different tokenizer)")
    bad = [i for i in draft.aux_layer_ids if i >= shape.num_hidden_layers]
    if bad:
        errs.append(f"aux_hidden_state_layer_ids {list(draft.aux_layer_ids)} "
                    f"read layer(s) {bad} — the model has only "
                    f"{shape.num_hidden_layers} layers")
    return errs


def _round_gib(bytes_: int, quantum: float) -> float:
    return round(bytes_ / GIB / quantum) * quantum


def _floor_gib(bytes_: int, quantum: float) -> float:
    return math.floor(bytes_ / GIB / quantum) * quantum


class Solver:

    # memory.kv_cache.speculation_pool_fraction — ONE source for the recipe
    # field and the admissibility pool model (they must not drift).
    SPECULATION_POOL_FRACTION = 0.15
    def __init__(self, hw: HardwareDescriptor, shape: ModelShape, base: dict,
                 levers: Levers, knobs: AutoconfigKnobs = AutoconfigKnobs(),
                 cal: CalibrationView | None = None,
                 cal_reject_reasons: tuple[str, ...] = (),
                 expert_slot_bytes: int = 0, slot_provenance: str = "",
                 non_expert_pinned_bytes: "int | Callable[[int], int]" = 0,
                 pinned_provenance: str = "",
                 draft: DraftCandidate | None = None,
                 pins: "PinSet | None" = None) -> None:
        self.hw = hw
        self.shape = shape
        self.base = base
        self.levers = levers
        self.k = knobs
        self.cal = cal
        self.cal_reject = cal_reject_reasons
        self.slot_bytes = expert_slot_bytes
        self.slot_prov = slot_provenance
        # The pinned REGION is a per-RANK figure: the engine's LayerRegistry
        # divides the TP-sharded slots by tensor_parallelism, so a solver that
        # charges every lattice point the tp=1 region refuses tp=2 fits that
        # the engine would take.  Accept a callable so the region is re-derived
        # at each tp the search visits (an int stays legal — tests and callers
        # that already know their tp pass one).
        if callable(non_expert_pinned_bytes):
            self._pinned_for = non_expert_pinned_bytes
        else:
            self._pinned_for = lambda _tp, _b=int(non_expert_pinned_bytes): _b
        self.pinned_bytes = self._pinned_for(
            max(1, len(base.get("hardware", {}).get("tp_array", []) or [1])))
        self.pinned_prov = pinned_provenance
        self.draft = draft
        self.ex = ExplanationSet()
        self._tp_ceiling_why = ""      # set by tp_degree (P2)
        self.warnings: list[str] = []
        self.degradations: list[str] = []
        self.quant = dict(base.get("quantization") or {})
        self.kv_quant = self.quant.get("kv_cache", "fp8_e4m3")
        if levers.prefer not in PREFERENCES:
            raise ValueError(
                f"autoconfig: prefer={levers.prefer!r} is not one of "
                f"{'/'.join(PREFERENCES)} (AUTOCONFIG §2.3)")
        if levers.accuracy not in ACCURACY_TIERS:
            raise ValueError(
                f"autoconfig: accuracy={levers.accuracy!r} is not one of "
                f"{'/'.join(ACCURACY_TIERS)} (AUTOCONFIG §2.4)")
        self._quiet_warnings: list[str] | None = None
        # TD-AUTOCONFIG-PINNED-CONSTRAINTS: pins are HARD constraints — a
        # third input class beside the identity base and the levers. Empty
        # set == None so every pin branch below is dead on the default path
        # (the byte-identity obligation, design §8).
        self.pins = pins if (pins is not None and len(pins) > 0) else None
        self._relaxed: frozenset = frozenset()  # refusal-attribution probe

    # -------------------------------------------------------------- probing

    @contextlib.contextmanager
    def _quiet(self):
        """Run sizing arithmetic with the derivation record switched OFF.

        The E1 search evaluates many candidate fits; only the WINNING one's
        explanation lines and only the warnings of the attempts the chosen
        order actually visited belong in the output. Everything inside this
        block is a probe."""
        prev_mute, prev_sink = self.ex.muted, self._quiet_warnings
        self.ex.muted = True
        self._quiet_warnings = []
        try:
            yield self._quiet_warnings
        finally:
            self.ex.muted = prev_mute
            self._quiet_warnings = prev_sink

    def _warn(self, msg: str) -> None:
        sink = self.warnings if self._quiet_warnings is None else self._quiet_warnings
        if msg not in sink:
            sink.append(msg)

    # ------------------------------------------------------------------ pins

    def _pin(self, path: str) -> "Pin | None":
        """The user's pin for a path, or None — honouring the refusal-
        attribution probe's relaxed set (design §6). Every pin consumer
        reads through here, so relaxing one axis relaxes it everywhere."""
        if self.pins is None or path in self._relaxed:
            return None
        return self.pins.get(path)

    # ------------------------------------------------- combination registry

    def _combo_ctx(self, tp: int) -> dict:
        """The combination the model_constraints.json registry keys on
        (TD-AUTOCONFIG-COMBO-CONSTRAINTS). Every PREDICATE_KEYS entry is
        supplied so a row can never see an unevaluable predicate; a future
        GPU predicate (card class x feature) adds its ctx entry HERE, not a
        new mechanism."""
        return {"arch": self.shape.architecture, "tensor_parallelism": tp}

    # ---------------------------------------------------------------- P1-P3

    def order_gpus(self) -> list[GpuInfo]:
        """P1: best-first. Speed from the CALIBRATION when accepted (user
        directive 2026-08-30: never card names); VRAM from BAR1. Falls back
        to (vram, compute-weight table) marked provisional."""
        gpus = list(self.hw.gpus)
        if self.cal is not None and not self.cal_reject:
            # calibration positions are hardware.gpus emission order of the
            # RECIPE THAT CALIBRATED; map by UUID prefix to live ordinals
            speed_by_uuid = {}
            for d in self.cal.devices:
                speed_by_uuid[d.uuid[:36]] = d.tokens_per_us_at(d.compute_p)
            def speed(g: GpuInfo) -> float:
                return next((v for u, v in speed_by_uuid.items()
                             if g.uuid.startswith(u)), 0.0)
            # Speed BUCKET, not raw speed: within a class, sub-10% measured
            # deltas are batch-step noise — deterministic ordinal order wins
            # (I8 determinism rule). Across classes the measured gap decides.
            top = max((speed(g) for g in gpus), default=1.0) or 1.0
            def key(g: GpuInfo):
                bucket = int(math.log(max(speed(g), 1e-9) / top) / math.log(1.10))
                return (-g.vram_mib, -bucket, g.ordinal)
            meas = "gpu_loader calibration (%s)" % os.path.basename(self.cal.path)
        else:
            def key(g: GpuInfo):
                return (-g.vram_mib, -g.compute_weight, g.ordinal)
            meas = ""
            self.warnings.append(
                "no accepted gpu_loader calibration — GPU ordering falls back "
                "to VRAM + name table; PROVISIONAL until the engine's "
                "load_or_calibrate_with measures this box"
                + (": " + "; ".join(self.cal_reject) if self.cal_reject else ""))
        ordered = sorted(gpus, key=key)
        self.ex.add("hardware.gpus[].order", [g.ordinal for g in ordered],
                    "measured" if meas else "heuristic",
                    "best-first: TP group = first tensor_parallelism entries "
                    "(engine.cpp:585-592), expert-host prefix scan follows "
                    "(registry row expert-host-prefix)",
                    refs=("AUTOCONFIG §5 P1", "expert-host-prefix"),
                    measurement=meas)
        return ordered

    def tp_degree(self, ordered: list[GpuInfo]) -> int:
        """P2: largest power-of-2 run of the top VRAM class, capped by head
        divisibility (config_validator.cpp:637-680, :764-770)."""
        top_vram = ordered[0].vram_mib
        run = sum(1 for g in ordered if g.vram_mib == top_vram
                  and g.gpu_type == ordered[0].gpu_type)
        tp = 1
        while tp * 2 <= run:
            tp *= 2
        s = self.shape
        while tp > 1 and (s.num_attention_heads % tp != 0
                          or (s.linear_attn and s.linear_attn.num_heads % tp != 0)
                          or (s.o_groups > 1 and s.o_groups % tp != 0)):
            tp //= 2
        # per-rank head cap (validator :677-680) binds the other way (min tp);
        # refuse if even the max run cannot satisfy it
        if s.num_attention_heads // tp > 128:
            raise Infeasible("heads-per-rank-cap", "tp group",
                             f"{s.num_attention_heads} heads / tp {tp}",
                             "<=128 heads/rank",
                             "more top-class GPUs are required for this model")
        self._tp_ceiling_why = (
            f"largest power-of-2 run of the top class ({run} x "
            f"{ordered[0].gpu_type} {top_vram} MiB), heads "
            f"{s.num_attention_heads} % tp == 0, <=128 heads/rank")
        return tp

    def _backend_menu(self) -> list[str]:
        """The family's serveable backends, most-accurate-first
        (templates.ACCURACY_LADDER), minus anything an engine-gate registry
        row takes off the table TODAY.  The only such row ever used is
        `glm5next-tq-backend-unwired` (deleted 2026-09-01; the lookup stays
        so a re-introduced gate flips the derivation back by data alone)."""
        fam = templates.family_of(self.shape.architecture)
        menu = list(templates.ACCURACY_LADDER[fam])
        row = ec.get("glm5next-tq-backend-unwired")
        if (row is not None and row.scope == self.shape.architecture
                and "turboquant_mla" in menu):
            menu.remove("turboquant_mla")
        return menu

    def _backend_for_tier(self, tier: str) -> str:
        """§2.4: a tier is a FLOOR over the accuracy ladder, never a weight.

        `standard` is the family template's measured choice (byte-identical
        to the pre-lever derivation); `high` floors at the most accurate
        serveable backend, `compact` at the most KV-byte-efficient one;
        `superior` names the full-precision arm the engine does not build —
        it refuses rather than approximate (an approximation would spend
        the user's accuracy ask silently, the exact failure this lever
        exists to end)."""
        menu = self._backend_menu()
        if tier == "superior":
            raise Infeasible(
                "accuracy-superior-not-implemented", "numerics",
                "accuracy=superior (full-precision KV — bf16 rows, codec "
                "kFull — no TQ anywhere, everything maximal)",
                "no engine device arm stores full-precision KV: KvCodecKind"
                "::kFull is declared but 'NOT built as a device arm "
                "anywhere' (src/daemon/attention/kv_codec.h:12-14); the "
                "most accurate serveable arm is accuracy=high "
                f"({menu[0]}, all-FP8 KV path)",
                "use accuracy=high, or land a kFull codec arm (kv_codec.h "
                "names the registration seam) and put it on the ladder")
        if tier == "high":
            return menu[0]
        if tier == "compact":
            return menu[-1]
        template = templates.FAMILY[
            templates.family_of(self.shape.architecture)]["attention_backend"]
        return template if template in menu else menu[0]

    def attention_backend(self) -> str:
        """The attention backend this recipe can actually be SERVED with.

        The family template (templates.FAMILY) carries the measured KV-bytes
        choice; the `accuracy` lever (§2.4) may floor it up or down the
        family's ladder.  A registry row may say the engine cannot serve a
        backend for this architecture TODAY — an engine-version fact, not
        physics, so it lives in engine_constraints.py where deleting it
        re-derives the config with the rule gone.  Every sizer in this class
        reads the backend through here, so the KV / indexer / slab byte
        model is derived for the backend that will actually boot, never a
        paper one.  A PINNED backend removes the numerics axis entirely
        (design §5): the pin was validated against the family's serveable
        partition and the accuracy floor at solve() entry."""
        p = self._pin("compute.attention_backend")
        if p is not None:
            return p.value
        return self._backend_for_tier(self.levers.accuracy)

    # P-31: the tp1-default row's own evidence bound says its prefill leg
    # was measured at ONE 141-token shape; P-30 steps 1-4 measured that at
    # champion-scale prefill attention is 60.4-73% of the wall and
    # stride-invariant — head-sharding (TP) is the only lever that touches
    # it.  For deep-context asks (the tiering row's own >=256k re-decide
    # bound) the plan therefore orders the head-divisible ceiling FIRST;
    # decode-dominated/short asks keep the measured TP=1 default.
    TP_CEILING_FIRST_MAX_SEQ = 262144

    def tp_plan(self, ordered: list[GpuInfo],
                ask_max_seq: int = 0) -> list[int]:
        """P2 continued: WHICH TP degrees to try, in order.

        The head-divisible run is the CEILING, not automatically the choice.
        Registry row `glm5next-tp1-default` carries the measured glm5_next
        finding: the KDA decode kernel is latency-floored, so TP buys ~nothing
        on decode while 45 per-layer combines cost ~0.5-2% of a serving wall.
        TP there is CORRECT but not faster — so the ceiling stays as a
        CAPACITY fallback the E1 fit may escalate to. A throughput finding
        must never turn into a refusal (AUTOCONFIG §6)."""
        p = self._pin("parallelism.tensor_parallelism")
        if p is not None:
            # axis removed (design §3): the pinned degree is the ONLY
            # candidate; structural legality was checked at solve() entry
            # (incl. a combination gate contradicting a kv-mode pin)
            return [int(p.value)]
        ceiling = self.tp_degree(ordered)
        row = ec.get("glm5next-tp1-default")
        if self.shape.is_glm5_next and row is not None and ceiling > 1:
            if ask_max_seq >= self.TP_CEILING_FIRST_MAX_SEQ:
                plan = [ceiling, 1]
            else:
                plan = [1, ceiling]
        else:
            plan = [ceiling]
        return self._combo_filter_tp_plan(plan)

    def _combo_filter_tp_plan(self, plan: list[int]) -> list[int]:
        """Adapt around a kv-mode pin that a combination gate contradicts at
        SOME tp degrees (TD-AUTOCONFIG-COMBO-CONSTRAINTS): a tp candidate
        where a model_constraints.json row forces dcp_kv_mode to a different
        value than the pin cannot boot, so it leaves the plan — LOUDLY, the
        same adapt-around shape as `pin dcp_indexer_mode=local DISABLES KV
        tiering`. The combination is only consulted where it applies: a tp
        the gate does not cover stays. When every candidate falls, the
        refusal names the pin (never a quiet relax)."""
        kv_pin = self._pin("hardware.dcp_kv_mode")
        if kv_pin is None:
            return plan
        kept: list[int] = []
        for tp in plan:
            crow = cc.forcing("hardware.dcp_kv_mode", self._combo_ctx(tp))
            if (crow is not None
                    and crow.forced("hardware.dcp_kv_mode") != kv_pin.value):
                self._warn(
                    f"pin hardware.dcp_kv_mode={kv_pin.value!r} removes "
                    f"tp={tp} from the plan: combination gate '{crow.id}' "
                    f"({crow.when_text()}) forces "
                    f"{crow.forced('hardware.dcp_kv_mode')!r} there "
                    f"[{crow.site}] — the derivation adapts around the pin")
            else:
                kept.append(tp)
        if not kept:
            crow = cc.forcing("hardware.dcp_kv_mode",
                              self._combo_ctx(plan[0]))
            assert crow is not None
            raise Infeasible(
                "pinned-kv-mode-engine-gated", "dcp kv mode",
                f"pin hardware.dcp_kv_mode={kv_pin.value!r} "
                f"({kv_pin.source})",
                f"combination gate '{crow.id}' ({crow.when_text()}): "
                f"{crow.statement} [{crow.site}]",
                "an engine_gate registry row is not overridable by a pin — "
                f"pin {crow.forced('hardware.dcp_kv_mode')!r} or drop "
                "the pin")
        return kept

    # ---------------------------------------------------------------- P4-P5

    def dcp_modes(self, tp: int) -> tuple[str, tuple[str, ...], bool]:
        """P4/P5: (dcp_kv_mode, dcp_indexer_mode CANDIDATES, tiering_on).

        The indexer mode is returned as an ORDERED CANDIDATE LIST, fastest
        first, because it is a capacity decision with a measured price
        (registry row `local-indexer-prefill-cost`): `local` halves the
        per-rank indexer-K share and costs ~8-9% of served prefill. It is one
        of the two PRICED axes of the E1 lattice; WHERE the ladder spends it
        relative to the user's ask is the `prefer` lever's business (§2.3),
        and `_explain_indexer_mode` names the price whenever it is spent."""
        arch = self.shape.architecture
        kv_pin = self._pin("hardware.dcp_kv_mode")
        idx_pin = self._pin("hardware.dcp_indexer_mode")
        kt_pin = self._pin("memory.kv_tiering.enabled")
        if kv_pin is not None:
            # axis removed (design §3); v4+sharded was refused at solve()
            # entry (engine gate v4-no-sharded-kv is not overridable), a
            # combination gate contradicting the pin was refused there too
            # (both pinned) or its tp lane left the plan (tp unpinned)
            kv_mode = kv_pin.value
            if tp < 2:
                self._warn(f"pin hardware.dcp_kv_mode={kv_mode!r} is INERT "
                           f"at dcp={tp}: the engine ignores the field below "
                           "dcp 2; emitted as pinned anyway")
            else:
                crow = cc.forcing("hardware.dcp_kv_mode", self._combo_ctx(tp))
                if crow is not None:
                    # belt-and-braces: solve() entry / tp_plan already keep a
                    # contradicting pin out of this branch
                    forced = crow.forced("hardware.dcp_kv_mode")
                    if forced != kv_mode:
                        raise Infeasible(
                            "pinned-kv-mode-engine-gated", "dcp kv mode",
                            f"pin hardware.dcp_kv_mode={kv_mode!r} "
                            f"({kv_pin.source})",
                            f"combination gate '{crow.id}' "
                            f"({crow.when_text()}): {crow.statement} "
                            f"[{crow.site}]",
                            "an engine_gate registry row is not overridable "
                            f"by a pin — pin {forced!r} or drop the pin")
                    # the pin AGREES with the gate: nothing is overridden,
                    # so no measured-row override warning belongs here
                else:
                    srow = ec.get("sharded-kv-beats-replicated")
                    if (kv_mode == "replicated" and srow is not None
                            and not self.shape.is_v4
                            and (not srow.scope or srow.scope == arch)):
                        self._warn(
                            "pin hardware.dcp_kv_mode='replicated' OVERRIDES "
                            f"measured registry row '{srow.id}': {srow.statement}"
                            f" [{srow.site}] — deliberate override assumed, "
                            "never silent (TD-AUTOCONFIG-PINNED-CONSTRAINTS (a))")
        elif tp < 2:
            kv_mode = "sharded"  # ignored below tp 2 (schema note)
        elif ec.get("v4-no-sharded-kv") and self.shape.is_v4:
            kv_mode = "replicated"
            self.ex.add("hardware.dcp_kv_mode", kv_mode, "template",
                        "engine gate: deepseek_v4 rejects sharded KV",
                        refs=("v4-no-sharded-kv", "TD-V4-DCP-KV"))
        elif (crow := cc.forcing("hardware.dcp_kv_mode",
                                 self._combo_ctx(tp))) is not None:
            # a COMBINATION gate (model_constraints.json): engine-current
            # fact, consulted only because its combination applies at this
            # tp — its own explain kind, citing the enforcing engine site
            # (TD-AUTOCONFIG-COMBO-CONSTRAINTS (b))
            kv_mode = crow.forced("hardware.dcp_kv_mode")
            self.ex.add("hardware.dcp_kv_mode", kv_mode, "engine_gate",
                        f"combination gate ({crow.when_text()}): "
                        f"{crow.statement}",
                        refs=(crow.id, crow.site, crow.ticket))
        else:
            srow = ec.get("sharded-kv-beats-replicated")
            if srow is not None and (not srow.scope or srow.scope == arch):
                kv_mode = "sharded"
                self.ex.add("hardware.dcp_kv_mode", kv_mode, "measured",
                            "sharded KV measured +4.6% e2e over replicated at dcp=2",
                            refs=("sharded-kv-beats-replicated",),
                            measurement="arena placement campaign 2026-08")
            else:
                # the measured row is arch-scoped (a GLM-5.2 champion
                # number must never price an arch it was not measured on —
                # TD-AUTOCONFIG-GLM5NEXT-TP-SHARDED-KV (c)) or deleted:
                # sharded stays the structural default, honestly unpriced
                kv_mode = "sharded"
                self.ex.add("hardware.dcp_kv_mode", kv_mode, "heuristic",
                            "structural default: sharding splits KV bytes "
                            "across the dcp ranks; UNPRICED for this arch — "
                            "the +4.6% measurement (row "
                            "sharded-kv-beats-replicated) is scoped to the "
                            "arch it was measured on")
        tiering_want = self.shape.has_dsa
        tier_escalatable = False
        g5row = ec.get("glm5next-tiering-default-off")
        if kt_pin is not None:
            # K3 decision fixed by the pin (design §3); has_dsa and the KV
            # format gate were validated at solve() entry.  A pin collapses
            # the tiering axis to one point, like every other pinned axis.
            tiering_want = bool(kt_pin.value)
            if tiering_want and self.shape.is_glm5_next and g5row is not None:
                self._warn(
                    "pin memory.kv_tiering.enabled=true OVERRIDES measured "
                    f"registry row '{g5row.id}': {g5row.statement} "
                    f"[{g5row.site}] — deliberate override assumed, never "
                    "silent (TD-AUTOCONFIG-PINNED-CONSTRAINTS (a))")
        elif self.shape.is_glm5_next and g5row is not None:
            # P-31: OFF stays the measured default; ON becomes the CAPACITY
            # escalation the E1 lattice may take when the untiered pool
            # cannot hold the conc x max_seq ask (the row's own "re-decide
            # at >=256k" boundary).  The kv_tiering.enabled explain row is
            # emitted by _explain_tiering for the WINNING choice only.
            tiering_want = False
            tier_escalatable = self.shape.has_dsa
        # TD-KVT-LOCAL-INDEXER-UNBLOCK resolved 2026-08-30: the engine's
        # replicated-only tiering construction gate was a merge artifact and
        # is REMOVED — tiering composes with dcp_indexer_mode=local (exact
        # cross-rank merge; champion A/B token-identity-gated).  The registry
        # row is deleted; the lookup stays so a re-introduced gate (a new row
        # with this id) flips the derivation back by data alone.
        row = ec.get("local-indexer-disables-tiering")
        cost = ec.get("local-indexer-prefill-cost")
        if idx_pin is not None:
            # axis removed (design §3): the pinned mode is the ONLY candidate
            idx_modes: tuple[str, ...] = (idx_pin.value,)
            if tp < 2:
                self._warn(f"pin hardware.dcp_indexer_mode={idx_pin.value!r} "
                           f"is INERT at dcp={tp}: nothing to localise below "
                           "dcp 2; emitted as pinned anyway")
            else:
                if (idx_pin.value == "local" and tiering_want
                        and row is not None and not self.shape.is_v4):
                    # a re-introduced engine gate: tiering cannot be built on
                    # a local indexer. Adapt around the pin — unless tiering
                    # is ALSO pinned on, which is a jointly impossible ask.
                    if kt_pin is not None and bool(kt_pin.value):
                        raise Infeasible(
                            "pinned-constraints-conflict-engine-gate",
                            "kv tiering x indexer mode",
                            "pins memory.kv_tiering.enabled=true AND "
                            "hardware.dcp_indexer_mode='local'",
                            f"engine gate '{row.id}': {row.statement}",
                            "drop one of the two pins — the engine cannot "
                            "build tiering on a local indexer while the "
                            "gate row stands")
                    tiering_want = False
                    tier_escalatable = False
                    self._warn(
                        "pin hardware.dcp_indexer_mode='local' DISABLES KV "
                        f"tiering: engine gate '{row.id}' ({row.statement}) "
                        "— the derivation adapts around the pin")
                if (idx_pin.value == "local" and cost is not None
                        and self.shape.has_dsa and not self.shape.is_v4):
                    self._warn(
                        "pin hardware.dcp_indexer_mode='local' OVERRIDES "
                        f"measured registry row '{cost.id}': replicated is "
                        "the measured default at dcp>=2 — local costs ~8-9% "
                        "of served prefill (45.15/43.88/43.85/42.73 -> "
                        "41.56/39.95/40.04/38.82 tok/s at 8k/20k/20k/25k; "
                        "decode in-noise, token-identical) for half the "
                        f"per-rank indexer-K share [{cost.site}] — "
                        "deliberate override assumed, never silent "
                        "(TD-AUTOCONFIG-PINNED-CONSTRAINTS (a))")
        elif tp < 2:
            # dcp=1: no shard to localise, the field is inert
            idx_modes = ("replicated",)
        elif (tiering_want or tier_escalatable) and row is not None \
                and not self.shape.is_v4:
            # a RE-INTRODUCED engine gate takes `local` off the menu entirely
            idx_modes = ("replicated",)
        elif cost is not None and self.shape.has_dsa and not self.shape.is_v4:
            # the measured trade: replicated first, local as the capacity
            # escalation.  Scoped to the population the measurement and the
            # sizing model share — a DSA indexer whose per-rank share the
            # solver actually models (V4's side-tier indexer is neither
            # measured nor modelled here, so it keeps the old heuristic).
            idx_modes = ("replicated", "local")
        else:
            idx_modes = ("local",)
        fmt_row = ec.get("tiering-needs-row-self-contained-format")
        backend = self.attention_backend()
        fmt_ok = (fmt_row is None) or self.shape.is_v4 or backend in ("snapmla", "turboquant_mla")
        # tiering is indexer-mode-agnostic (INV-KVT-20); when a re-introduced
        # `local-indexer-disables-tiering` row says otherwise the candidate
        # list above has already been narrowed to replicated, so tiering
        # never has to be traded away for the mode.
        if kt_pin is not None:
            tier_cands: tuple = (tiering_want and fmt_ok,)
        elif tier_escalatable and fmt_ok:
            tier_cands = (False, True)   # OFF default, ON = escalation
        else:
            tier_cands = (tiering_want and fmt_ok,)
        return kv_mode, idx_modes, tier_cands

    def _indexer_share(self, mode: str, tp: int, max_seq: int) -> tuple[int, int]:
        """(slabs, bytes) of the per-rank indexer-K share for ONE max-length
        sequence — the unit the boot log and the champion A/B report (84
        slabs / 87.1 MiB replicated, 42 / 43.5 MiB local, at max_seq 25600).
        (0, 0) for models with no DSA indexer."""
        s, k = self.shape, self.k
        backend = self.attention_backend()
        geo = sizing.slab_geometry(s, self.kv_quant, backend, k.page_size_tokens,
                                   k.indexer_k_page_size_tokens)
        if geo is None:
            return (0, 0)
        pages = sizing.indexer_pages_per_seq(s, max_seq, k.indexer_k_page_size_tokens,
                                             mode, tp)
        slabs = pages * s.num_dsa_computing_layers
        return slabs, slabs * geo.slab_bytes

    def _explain_indexer_mode(self, chosen: str, candidates: tuple[str, ...],
                              tp: int, max_seq: int, slack_bytes: int,
                              escalated_from: "Infeasible | None") -> None:
        """P5: say WHY this mode, and NAME THE PRICE when `local` is taken.

        Same shape as the TP row's "capacity, not speed": a measured cost row
        makes the fast mode the default, and the fit escalates to the cheap
        mode only when the VRAM is genuinely contested."""
        cost = ec.get("local-indexer-prefill-cost")
        if cost is None or len(candidates) < 2:
            # no measured price (row deleted, or the mode was forced): the
            # pre-row rationale stands — local is the VRAM-cheaper mode and
            # nothing forbids it.
            row = ec.get("local-indexer-disables-tiering")
            if chosen == "replicated" and row is not None and tp >= 2:
                self.ex.add("hardware.dcp_indexer_mode", chosen, "template",
                            "KV tiering requires the replicated indexer at "
                            f"dcp>=2 (engine gate, registry row '{row.id}')",
                            refs=(row.id, row.ticket, row.site))
            elif chosen == "local":
                self.ex.add("hardware.dcp_indexer_mode", chosen, "heuristic",
                            "local halves indexer-K VRAM per rank (ceil-divide "
                            "by dcp; structurally equal to replicated "
                            "post-GF3.4/S4, and KV tiering composes with local "
                            "— TD-KVT-LOCAL-INDEXER-UNBLOCK resolved); note "
                            "the saving is arch-dependent (IndexPool archs "
                            "pool pages ~kpool x cheaper)",
                            refs=("vram_allocator.cpp:565-568",
                                  "TD-KVT-LOCAL-INDEXER-UNBLOCK"))
            return
        r_slabs, r_bytes = self._indexer_share("replicated", tp, max_seq)
        l_slabs, l_bytes = self._indexer_share("local", tp, max_seq)
        mib = 1 << 20
        ladder = ("45.15/43.88/43.85/42.73 -> 41.56/39.95/40.04/38.82 tok/s "
                  "at 8k/20k/20k/25k (-8.0/-9.0/-8.7/-9.2%)")
        refs = (cost.id, "TD-KVT-LOCAL-INDEXER-UNBLOCK", "INV-KVT-20",
                "AUTOCONFIG §5 P5")
        meas = ("champion A/B 2026-08-30, spec/measurements/glm_prefill.md; "
                "evidence scratchpad/kvtlocal_ab_{replicated,local}.json")
        if chosen == "replicated":
            self.ex.add(
                "hardware.dcp_indexer_mode", chosen, "measured",
                f"replicated is the MEASURED default at dcp={tp}: `local` "
                f"would halve the per-rank indexer-K share ({r_slabs} slabs / "
                f"{r_bytes / mib:.1f} MiB -> {l_slabs} slabs / "
                f"{l_bytes / mib:.1f} MiB per max-length sequence) but costs "
                f"~8-9% of served prefill ({ladder}; decode in-noise, "
                "token-identical, tiering composes) because the per-chunk-row "
                "cross-rank merge runs per layer over B=64 rows. Those "
                f"{(r_bytes - l_bytes) / mib:.1f} MiB/rank are AFFORDABLE here "
                f"({slack_bytes / GIB:.2f} GiB slack left on the tightest TP "
                "GPU), so the box buys prefill with VRAM it is not using; the "
                "E1 fit escalates to `local` on its own when they are "
                "contested",
                refs=refs, measurement=meas)
        else:
            why = (f"the fit at replicated was infeasible "
                   f"({escalated_from.constraint_id} on "
                   f"{escalated_from.binding})" if escalated_from is not None
                   else "replicated did not fit at this rung")
            self.ex.add(
                "hardware.dcp_indexer_mode", chosen, "searched",
                "escalated from the measured default `replicated` to `local` "
                f"for CAPACITY, NOT SPEED: {why}. local halves the per-rank "
                f"indexer-K share ({r_slabs} -> {l_slabs} slabs, "
                f"{r_bytes / mib:.1f} -> {l_bytes / mib:.1f} MiB per "
                f"max-length sequence — {(r_bytes - l_bytes) / mib:.1f} "
                "MiB/rank reclaimed) and the PRICE is ~8-9% of served prefill "
                f"({ladder}; decode in-noise, token-identical, tiering "
                "composes — the per-chunk-row cross-rank merge runs per layer "
                "over B=64 rows)",
                refs=refs, measurement=meas)

    def _explain_tiering(self, winner: Attempt, setup: TpSetup,
                         escalated_from: "Infeasible | None") -> None:
        """K3 for the WINNING attempt.  Emitted here (not in dcp_modes)
        because tiering became an E1 axis on glm5_next (P-31 gap-2
        interlock): the OFF default and the ON capacity escalation are the
        same registry row read two ways."""
        g5row = ec.get("glm5next-tiering-default-off")
        if len(setup.tiering_candidates) < 2:
            # axis collapsed (pin, or a non-escalatable arch): the pin row
            # or the historical silence owns the story
            if (self.shape.is_glm5_next and g5row is not None
                    and not winner.tiering_on
                    and self._pin("memory.kv_tiering.enabled") is None):
                self.ex.add("memory.kv_tiering.enabled", False, "measured",
                            g5row.statement, refs=(g5row.id, g5row.site))
            return
        if not winner.tiering_on:
            if g5row is not None:
                self.ex.add("memory.kv_tiering.enabled", False, "measured",
                            g5row.statement + " The untiered pool holds "
                            "the conc x max_seq ask here, so the capacity "
                            "escalation was not needed.",
                            refs=(g5row.id, g5row.site))
            return
        why = (f"the untiered fit was infeasible "
               f"({escalated_from.constraint_id} on "
               f"{escalated_from.binding})" if escalated_from is not None
               else "the untiered pool cannot hold the ask")
        self.ex.add(
            "memory.kv_tiering.enabled", True, "searched",
            "ESCALATED from the measured glm5_next OFF default for "
            f"CAPACITY: {why}. Tiering moves whole-life KV residency to "
            "the pinned host cold pool under windowed admission "
            "(INV-KVT-16, tiered_prefill), so the device pool holds only "
            "the hot window and the conc x max_seq ask becomes admissible"
            + (f"; registry row '{g5row.id}' keeps OFF the default below "
               "this bound" if g5row is not None else ""),
            refs=("glm5next-tiering-default-off", "INV-KVT-16",
                  "AUTOCONFIG §5 E1"))

    # ---------------------------------------------------------------- K1

    def active_split(self) -> tuple[int, int]:
        """K1: (max_concurrent_requests, max_sequence_length) from lever 2.

        Pinned halves of the split are HARD (design §3): a pinned
        max_sequence_length is not the base-recipe's carried starting point
        — the capacity ladder emits no halving rungs for it (`_lattice`) and
        the admissibility loop refuses instead of reducing it (solve())."""
        model_max = self.shape.max_position_embeddings
        seq_pin = self._pin("serving.max_sequence_length")
        conc_pin = self._pin("serving.max_concurrent_requests")
        if seq_pin is not None or conc_pin is not None:
            c_act = self.levers.total_active_context_tokens
            if seq_pin is not None and conc_pin is not None:
                conc, max_seq = int(conc_pin.value), int(seq_pin.value)
                if c_act is not None and conc * max_seq != c_act:
                    self._warn(
                        f"pins fix the active-context split to {conc} x "
                        f"{max_seq} = {conc * max_seq} tokens, overriding "
                        f"the --active-context lever's {c_act} (a pin is a "
                        "hard constraint; the lever is an ask)")
                return conc, max_seq
            if c_act is None:
                c_act = min(model_max, 2 * 65536)
                self.ex.add("autoconfig.total_active_context_tokens", c_act,
                            "heuristic",
                            "lever unset: default 2 x 64k active context "
                            "(override with --active-context)")
            if seq_pin is not None:
                max_seq = int(seq_pin.value)
                conc = max(1, round(c_act / max_seq))
                self.ex.add("serving.max_concurrent_requests", conc,
                            "closed_form",
                            f"total_active_context {c_act} / pinned "
                            f"max_sequence_length {max_seq}",
                            refs=("AUTOCONFIG §2.2/§2.5",))
                return conc, max_seq
            conc = int(conc_pin.value)
            gran = self.k.page_size_tokens * 2  # page x max dcp granularity
            max_seq = min(model_max,
                          max(gran, (c_act // conc) // gran * gran))
            self.ex.add("serving.max_sequence_length", max_seq, "closed_form",
                        f"total_active_context {c_act} split across the "
                        f"pinned concurrency {conc}, floored to {gran}-token "
                        "granularity",
                        refs=("AUTOCONFIG §2.2/§2.5",))
            return conc, max_seq
        c_act = self.levers.total_active_context_tokens
        if c_act is None:
            c_act = min(model_max, 2 * 65536)
            self.ex.add("autoconfig.total_active_context_tokens", c_act, "heuristic",
                        "lever unset: default 2 x 64k active context "
                        "(override with --active-context)")
        base_seq = (self.base.get("serving") or {}).get("max_sequence_length")
        if base_seq:
            max_seq = int(base_seq)
            conc = max(1, round(c_act / max_seq))
            kind, why = "carried", (f"serving.max_sequence_length {max_seq} carried "
                                    f"from the base recipe; concurrency covers the "
                                    f"lever: round({c_act}/{max_seq})")
        elif c_act >= 2 * self.k.min_seq_for_pairing:
            gran = self.k.page_size_tokens * 2  # page x max dcp granularity
            max_seq = (c_act // 2) // gran * gran
            conc = 2
            kind, why = "heuristic", (
                f"equal split of the {c_act}-token active-context lever into 2 "
                f"concurrent requests (headroom for a second sequence — the "
                f"lever's own rationale); set serving.max_sequence_length in "
                f"the base recipe to override the split")
        else:
            max_seq = min(c_act, model_max)
            conc = 1
            kind, why = "closed_form", f"lever {c_act} below 2x{self.k.min_seq_for_pairing}: single request"
        max_seq = min(max_seq, model_max)
        self.ex.add("serving.max_sequence_length", max_seq, kind, why,
                    refs=("AUTOCONFIG §2.2/§5 K1",))
        self.ex.add("serving.max_concurrent_requests", conc, kind,
                    f"total_active_context {c_act} / max_sequence_length {max_seq}",
                    refs=("AUTOCONFIG §2.2",))
        return conc, max_seq

    # ---------------------------------------------------------------- E1 fit

    def usable_bytes(self, g: GpuInfo) -> int:
        return int(g.vram_mib * (1 << 20) * self.k.vram_usable_fraction)

    def _tp_fit(self, g: GpuInfo, tp: int, conc: int, max_seq: int,
                draft_on: bool, draft_bytes: int, expert_bytes: int,
                idx_mode: str, tiering_on: bool, kv_mode: str,
                margin_gb: float | None = None) -> GpuPlan:
        s, k = self.shape, self.k
        plan = GpuPlan(ordinal=g.ordinal, gpu_type=g.gpu_type,
                       usable_bytes=self.usable_bytes(g))
        plan.margin_bytes = int((margin_gb if margin_gb is not None
                                 else k.vram_safety_margin_gb) * GIB)
        plan.pinned_bytes = self._pinned_for(tp)
        plan.draft_bytes = draft_bytes if draft_on else 0
        plan.scratch_bytes = int(k.prefill_scratch_gb * GIB)
        plan.expert_bytes = expert_bytes
        plan.expert_slots = sizing.expert_slots_in_zone(expert_bytes, self.slot_bytes)
        backend = self.attention_backend()
        geo = None
        if not s.is_v4 and s.has_dsa:
            geo = sizing.slab_geometry(s, self.kv_quant, backend,
                                       k.page_size_tokens,
                                       k.indexer_k_page_size_tokens)
        if s.num_linear_attention_layers > 0:
            slots = sizing.kda_policy_slots(conc, True, k.prefix_cache_max_entries)
            # Registry row kda-state-mapped-tenant (TD-KDA-STATE-MAPPED-SLABS
            # default since 2026-08-31; TD-AUTOCONFIG-STALE-KDA-CARVE-ROW):
            # the state is a MAPPED tenant of the shared slab pool — charge
            # the SLAB-PADDED per-request demand (150.2 MiB vs the 145.6 MiB
            # slot on GLM-5.3-Flash) and count it as SHARED pool capacity
            # below, exactly like the engine folds it into kv_main. Deleting
            # the row (or an unslabbed shape) reverts to the carve model.
            if ec.get("kda-state-mapped-tenant") is not None and geo is not None:
                plan.kda_mapped = True
                plan.kda_bytes = slots * sizing.kda_mapped_demand_bytes(
                    s, tp, geo.slab_bytes)
            else:
                plan.kda_bytes = slots * sizing.kda_slot_bytes(s, tp)
        if s.is_v4:
            # V4: side tiers auto-size with proportional scale-down; charge
            # the un-scaled demand for feasibility, engine scales gracefully.
            d = sizing.v4_tier_demand(s, backend, max_seq, conc,
                                      k.prefix_cache_max_entries, max_seq,
                                      k.indexer_k_page_size_tokens, 0.15)
            plan.kv_pool_bytes = (d.csa_pages * d.csa_bytes_per_page
                                  + d.hca_pages * d.hca_bytes_per_page
                                  + d.swa_pages * d.swa_bytes_per_page
                                  + d.lid_pages * d.lid_bytes_per_page
                                  + d.spec_pages * d.csa_bytes_per_page)
        else:
            shard = tp if (tp >= 2 and kv_mode == "sharded") else 1
            bpp = sizing.kv_bytes_per_page(s, self.kv_quant, backend, k.page_size_tokens)
            if s.has_dsa:
                idx_pages = sizing.indexer_pages_per_seq(
                    s, max_seq, k.indexer_k_page_size_tokens, idx_mode, tp)
                per_seq = idx_pages * s.num_dsa_computing_layers * geo.slab_bytes
                # engine affordability (quarter-residual, INV-KVT-14b): the
                # S4 share is sized to what VRAM affords, not to conc
                residual = (plan.usable_bytes - plan.pinned_bytes
                            - plan.draft_bytes - plan.margin_bytes
                            - plan.kda_bytes - plan.expert_bytes
                            - plan.scratch_bytes - per_seq)
                afford = 1 + max(residual, 0) // 4 // per_seq if per_seq else conc
                idx_seqs = min(conc, max(1, afford))
                if idx_seqs < conc:
                    msg = (f"indexer-K share: VRAM affords {idx_seqs} concurrent "
                           f"max-length sequence(s), below max_concurrent_requests="
                           f"{conc} ({per_seq / (1 << 20):.1f} MiB/seq) — S4 elastic: "
                           f"extra demand draws on free KV slabs; concurrent "
                           f"max-length requests churn holder evictions "
                           f"(retryable, TD-INDEXER-POOL-EVICT)")
                    self._warn(msg)
                plan.idx_share_bytes = per_seq * idx_seqs
            if tiering_on:
                hot = s.index_topk
                pages_per_layer = math.ceil(hot / k.page_size_tokens)
                pages = conc * pages_per_layer * s.engine_kv_pool_layers()
                pages = sizing.align_up(pages, k.device_pool_round_pages)
                plan.kv_pool_bytes = pages * bpp
            else:
                avail = (plan.usable_bytes - plan.pinned_bytes - plan.draft_bytes
                         - plan.margin_bytes - plan.kda_bytes - plan.expert_bytes
                         - plan.scratch_bytes - plan.idx_share_bytes)
                pages = sizing.auto_kv_pages(
                    max_seq, k.page_size_tokens, shard, k.dcp_chunk_size,
                    conc, s.engine_kv_pool_layers(), max(avail, 0), bpp)
                demand = (conc
                          * (math.ceil(max_seq / k.page_size_tokens) // max(shard, 1)
                             if shard > 1 else math.ceil(max_seq / k.page_size_tokens))
                          * s.engine_kv_pool_layers())
                if pages < demand:
                    raise Infeasible(
                        "kv-demand-exceeds-vram", f"gpu {g.ordinal}",
                        f"{demand} pages ({demand * bpp / GIB:.1f} GiB)",
                        f"{pages} pages ({pages * bpp / GIB:.1f} GiB)",
                        "reduce total_active_context_tokens, raise "
                        "vram_expert_ratio down, or enable KV tiering "
                        "(model lacks DSA?)")
                plan.kv_pool_bytes = pages * bpp
                # TD-AUTOCONFIG-MAXSEQ-IGNORES-MAPPED-KDA: single-request
                # admissibility over the ONE shared pool. Since the mapped
                # default, KV pages, the full-length indexer-K reservation
                # and the KDA state all draw on kv_main at admission — the
                # pool must hold their SUM for one max-length request or
                # the advertised context is a lie the user discovers at
                # request time (the GF3 1M arm: refused at 474,880 of an
                # advertised 500,000). Mirrors the engine's boot ceiling
                # (vram_allocator.cpp, TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA);
                # solve() reduces max_seq to the ceiling when it binds.
                pps = geo.pages_per_slab if geo is not None else 0
                state_pages = (sizing.kda_mapped_state_pages(
                                   s, tp, geo.slab_bytes, pps)
                               if plan.kda_mapped and geo is not None else 0)
                plan.adm_state_pages = state_pages
                # kMain pool = KV bytes MINUS the speculation split (the
                # engine carves spec_frac of the KV share into its own
                # pool: kv_without_scratch * frac, vram_allocator.cpp
                # kv_speculation_bytes) PLUS the indexer + mapped-state
                # shares, which ride in kv_main whole. Modelling the spec
                # split matters: without it the modeled pool (and so the
                # ceiling) runs ~15% hot vs the engine's own boot figure
                # (verified against the 2026-09-01 fixed-recipe boot:
                # modeled 1,117,188 vs engine 1,117,182 pages — within
                # slab quantization — once the split is subtracted).
                spec_bytes = int(plan.kv_pool_bytes
                                 * self.SPECULATION_POOL_FRACTION)
                plan.adm_pool_pages = (plan.kv_pool_bytes - spec_bytes
                                       + plan.idx_share_bytes
                                       + (plan.kda_bytes if plan.kda_mapped
                                          else 0)) // bpp
                idx_shard = tp if (idx_mode == "local" and tp > 1) else 1
                dsa_layers = (s.num_dsa_computing_layers
                              if (s.has_dsa and pps > 0) else 0)
                plan.adm_kv_pages = sizing.single_request_demand_pages(
                    max_seq, k.page_size_tokens, s.engine_kv_pool_layers(),
                    0, 0, 0, 0, shard, k.dcp_chunk_size)
                plan.adm_idx_pages = sizing.single_request_demand_pages(
                    max_seq, k.page_size_tokens, s.engine_kv_pool_layers(),
                    dsa_layers, k.indexer_k_page_size_tokens, pps, 0,
                    shard, k.dcp_chunk_size, idx_shard) - plan.adm_kv_pages
                plan.adm_demand_pages = (plan.adm_kv_pages
                                         + plan.adm_idx_pages + state_pages)
                plan.admissible_max_seq = sizing.admissible_context_tokens(
                    plan.adm_pool_pages, k.page_size_tokens,
                    s.engine_kv_pool_layers(), dsa_layers,
                    k.indexer_k_page_size_tokens, pps, state_pages,
                    shard, k.dcp_chunk_size, idx_shard)
                # P-31: the ask is CONC simultaneous max-length requests
                # (§2.2's own definition of total_active_context) — the
                # untiered pool must hold their SUM, or the second request
                # is refused at admission time.  The E1 lattice escalates
                # (tiering ON where a candidate exists, else sheds).
                if conc > 1 and plan.adm_pool_pages > 0 \
                        and plan.adm_pool_pages < conc * plan.adm_demand_pages:
                    raise Infeasible(
                        "conc-ask-exceeds-untiered-pool", f"gpu {g.ordinal}",
                        f"{conc} concurrent {max_seq}-token requests = "
                        f"{conc} x {plan.adm_demand_pages} = "
                        f"{conc * plan.adm_demand_pages} kMain pages "
                        f"(KV {plan.adm_kv_pages} + indexer "
                        f"{plan.adm_idx_pages} + mapped KDA state "
                        f"{plan.adm_state_pages} each)",
                        f"pool {plan.adm_pool_pages} pages",
                        "enable KV tiering (windowed admission, "
                        "INV-KVT-16), lower concurrency, or shorten "
                        "max_sequence_length")
        if plan.slack() < 0:
            raise Infeasible(
                "vram-carve-overflow", f"gpu {g.ordinal} ({g.gpu_type})",
                f"{plan.total_committed() / GIB:.2f} GiB "
                f"(pinned {plan.pinned_bytes / GIB:.1f} + draft "
                f"{plan.draft_bytes / GIB:.1f} + experts {plan.expert_bytes / GIB:.1f}"
                f" + kv {plan.kv_pool_bytes / GIB:.2f} + idx "
                f"{plan.idx_share_bytes / GIB:.2f} + kda {plan.kda_bytes / GIB:.2f}"
                f" + scratch/margin)",
                f"{plan.usable_bytes / GIB:.2f} GiB usable",
                "lower vram_expert_ratio, reduce total_active_context_tokens, "
                "or drop the dspark draft")
        return plan

    def expert_targets(self, ordered: list[GpuInfo], tp: int,
                       residual_tp_bytes: int) -> dict[int, int]:
        """Slot targets per ordinal from lever 1 (AUTOCONFIG §2.1)."""
        s, k = self.shape, self.k
        s_total = s.n_routed_experts * s.num_moe_layers
        floor_slots = s.num_experts_per_tok  # engine min_expert_cache
        tp_gpus = ordered[:tp]
        hosts = ordered[tp:]
        cap = {g.ordinal: int(_floor_gib(
            self.usable_bytes(g) - int(k.vram_safety_margin_gb * GIB)
            - int(k.non_tp_overhead_gib * GIB), k.expert_gib_quantum) * GIB)
            // self.slot_bytes for g in hosts}
        ratio = self.levers.vram_expert_ratio
        if ratio is None:
            targets = {g.ordinal: cap[g.ordinal] for g in hosts}
            tp_each = max(floor_slots, residual_tp_bytes // self.slot_bytes)
            for g in tp_gpus:
                targets[g.ordinal] = tp_each
            total = sum(targets.values())
            self.ex.add("autoconfig.vram_expert_ratio",
                        round(total / s_total, 4), "heuristic",
                        "lever unset: expert hosts filled to capacity, TP GPUs "
                        "take the residual after KV/indexer (the engine's own "
                        "residual carve)")
            return targets
        target_slots = round(ratio * s_total)
        targets = {g.ordinal: floor_slots for g in tp_gpus}
        remaining = target_slots - floor_slots * len(tp_gpus)
        if remaining < 0:
            raise Infeasible(
                "expert-ratio-below-engine-floor", "tp gpus",
                f"ratio {ratio} = {target_slots} slots",
                f">= {floor_slots * len(tp_gpus)} slots "
                f"(num_experts_per_tok x tp — vram_allocator.cpp:1259)",
                "raise vram_expert_ratio")
        # fill non-TP hosts equally up to capacity (deterministic, ordinal order)
        for rounds in range(2):
            live = [g for g in hosts if targets.get(g.ordinal, 0) < cap[g.ordinal]]
            if not live or remaining <= 0:
                break
            share = remaining // len(live)
            for g in live:
                take = min(share, cap[g.ordinal] - targets.get(g.ordinal, 0))
                targets[g.ordinal] = targets.get(g.ordinal, 0) + take
                remaining -= take
        # overflow to TP GPUs equally, bounded by their residual
        if remaining > 0:
            tp_cap_each = max(0, residual_tp_bytes // self.slot_bytes)
            share = remaining // len(tp_gpus)
            extra = 0
            for g in tp_gpus:
                take = min(share, tp_cap_each - (targets[g.ordinal] - floor_slots))
                targets[g.ordinal] += take
                extra += take
            remaining -= extra
        if remaining > 0:
            achievable = (target_slots - remaining) / s_total
            raise Infeasible(
                "expert-ratio-exceeds-vram", "all expert hosts",
                f"vram_expert_ratio {ratio} ({target_slots} slots)",
                f"~{achievable:.4f} ({target_slots - remaining} slots)",
                "lower vram_expert_ratio or total_active_context_tokens")
        return targets

    # ---------------------------------------------------------------- solve

    DRAFT_DROP_NOTE = ("dspark draft dropped (its VRAM is worth more as "
                       "experts/KV at the requested levers)")

    def _draft_charge(self, ordered: list[GpuInfo], tp: int,
                      explain: bool = True) -> tuple[list, int]:
        """P6: which ranks carry the dspark draft and what each is charged.
        ``explain=False`` sizes without emitting the draft_gpus row — the
        loud winner re-run passes it when E1 DROPPED the draft, so the
        explain sheet never carries a placement row for a draft that will
        not run (the drop itself is a degradation line)."""
        s = self.shape
        draft = self.draft
        if draft is None:
            return [], 0
        mp = self._pin("speculation.method")
        if mp is not None and mp.value == "none":
            # pinned OFF: never charged, never explained — the pin row is
            # the story (and no draft_gpus row for a draft that will not run)
            return [], 0
        if draft.is_gguf or tp < 2:
            draft_ranks = [tp] if len(ordered) > tp else []
            nr = 1
            quant = "bf16"
        else:
            draft_ranks = list(range(tp))
            nr = tp
            quant = "nvfp4"
        if draft.weight_file:
            w = sizing.dspark_weight_bytes_safetensors(draft.weight_file, quant, nr)
            prov = f"safetensors header of {os.path.basename(os.path.dirname(draft.weight_file))}"
        else:
            w = 8 * GIB // max(nr, 1)  # unverifiable checkpoint: conservative
            prov = "checkpoint unreadable — conservative 8 GiB estimate"
        sc = sizing.dspark_scratch_bytes(
            s, draft.draft_layers, draft.draft_kv_heads, draft.draft_head_dim,
            draft.draft_hidden, draft.draft_vocab, draft.ctx_cap_tokens,
            2048, draft.block_size, nr, 0)
        # Worst-rank headroom: the loader shards UNEVENLY (2026-08-30
        # boot: rank0 3.28 GiB vs rank1 1.03 at nr=2 — rank 0 carries
        # embed/head rows + aux/logits staging); charge every rank the
        # rank-0-shaped estimate x1.15 so the fit never under-reserves.
        draft_bytes = int((w + sc) * 1.15)
        if not explain:
            return draft_ranks, draft_bytes
        self.ex.add("speculation.dspark.draft_gpus",
                    [min(r, len(ordered) - 1) for r in draft_ranks], "measured"
                    if not draft.is_gguf else "template",
                    "nvfp4 draft sharded across the TP GPUs (measured: tax "
                    "10->7.7 ms/step, acceptance UP under quant); GGUF "
                    "dflash drafts are bf16 single-rank on the first "
                    "non-TP GPU (dspark_loader.cpp:449)",
                    refs=("dspark-sharded-nvfp4-draft",),
                    measurement=f"{draft_bytes / GIB:.2f} GiB/rank ({prov} "
                                f"+ scratch at ctx cap {draft.ctx_cap_tokens})")
        return draft_ranks, draft_bytes

    def _tp_setup(self, ordered: list[GpuInfo], tp: int,
                  explain_draft: bool = True) -> TpSetup:
        """Everything a fixed TP degree fixes before the E1 search: the DCP
        modes (P4/P5) and the draft charge (P6). Explains itself unless the
        solver is probing (`_quiet`), so the record carries the WINNING TP's
        rationale and never a rejected candidate's."""
        kv_mode, idx_modes, tier_cands = self.dcp_modes(tp)
        draft_ranks, draft_bytes = self._draft_charge(ordered, tp,
                                                      explain=explain_draft)
        return TpSetup(tp=tp, kv_mode=kv_mode, idx_modes=tuple(idx_modes),
                       tiering_candidates=tuple(tier_cands),
                       draft_ranks=draft_ranks, draft_bytes=draft_bytes)

    def _lattice(self, tp_candidates: list[int], setups: dict,
                 conc0: int, max_seq0: int) -> list[Attempt]:
        """The complete E1 search space (AUTOCONFIG §5 E1) — INDEPENDENT of
        `prefer`, which only sorts it. Building the set once, here, is what
        makes the never-refuse property (§6) a property of the code and not
        of three hand-written ladders that have to be kept in step."""
        k = self.k
        seq_pin = self._pin("serving.max_sequence_length")
        conc_pin = self._pin("serving.max_concurrent_requests")
        method_pin = self._pin("speculation.method")
        caps: list[tuple[int, int, str]] = [(conc0, max_seq0, "")]
        if conc_pin is None:
            c = conc0 - 1
            while c >= 1:
                caps.append((c, max_seq0, f"concurrency {conc0}->{c}"))
                c -= 1
        if seq_pin is None:
            # halving rungs keep a pinned concurrency whole (design §3):
            # the axis a pin removes never re-enters via a deeper rung
            hc = int(conc_pin.value) if conc_pin is not None else 1
            seq = max_seq0 // 2
            while seq >= k.min_max_seq:
                caps.append((hc, seq, f"max_sequence_length {max_seq0}->{seq}"))
                seq //= 2
        # draft_off values: with no draft there is only the "off" world;
        # a pinned speculation.method collapses the axis (design §3)
        if method_pin is not None:
            drafts = [True] if method_pin.value == "none" else [False]
        else:
            drafts = [False, True] if self.draft is not None else [True]
        out: list[Attempt] = []
        for ti, tp in enumerate(tp_candidates):
            for xi, mode in enumerate(setups[tp].idx_modes):
                for ci, (conc, ms, cnote) in enumerate(caps):
                    for off in drafts:
                        note = cnote if ci else (
                            self.DRAFT_DROP_NOTE
                            if (off and self.draft is not None
                                and method_pin is None) else "")
                        for zi, tier in enumerate(
                                setups[tp].tiering_candidates):
                            out.append(Attempt(
                                tp_idx=ti, idx_idx=xi, cap_idx=ci,
                                draft_off=off, tp=tp, idx_mode=mode,
                                conc=conc, max_seq=ms, note=note,
                                tier_idx=zi, tiering_on=tier))
        return out

    # -------------------------------------------- P-31 long-context plan

    def _declared_vram_gb(self, g: GpuInfo) -> int:
        return int(g.vram_mib * self.k.vram_usable_fraction) >> 10

    def _fit_free_mib(self, g: GpuInfo, margin_gb: float, *, prefit_mib: int,
                      expert_only: bool) -> int:
        """Modeled cudaMemGetInfo free at the MoE-big elastic fit check
        (command_dispatcher.cpp:539): physical BAR1 minus the VramAllocator
        block (vram_gb - margin), minus a per-class fixed overhead (CUDA
        ctx + arena page tables + lazy modules — measured anchors, P-30
        step 2: 5090 free 1668 / 5080 1685 MiB at the champion carve),
        minus the PRE-fit runtime allocations (sizing.runtime_scratch)."""
        k = self.k
        fixed = (k.moe_fit_fixed_expert_mib if expert_only
                 else k.moe_fit_fixed_attn_mib)
        block_mib = int((self._declared_vram_gb(g) - margin_gb) * 1024)
        return g.vram_mib - block_mib - fixed - prefit_mib

    def _longctx_plan(self, ordered: list[GpuInfo], tp: int, conc: int,
                      max_seq: int, tiering_on: bool,
                      kv_mode: str) -> LongCtxPlan:
        """Gaps 1/2/4 of TD-AUTOCONFIG-LONGCTX-RUNTIME-SCRATCH and
        TD-AUTOCONFIG-STRIDE-NOT-DERIVED, jointly (they fund each other):

          B  (gap 4)  largest halving of the family schedule-width cap
                      whose dev_block_tables fit the budget knob;
          S  (gap 1)  largest superchunk stride on the engine's own
                      per-device-class fit arithmetic — the expert-only
                      class binds first (no scratch tail), the attention
                      hosts are funded by raising the TP margin, capped;
          headroom    the attention-host `moe_big_fit_headroom_mb` must
                      cover the POST-check allocations (block tables,
                      logits, KVT union, MoE persistent set) or the boot
                      dies at command_dispatcher.cpp:1363;
          margin (2)  the TP-rank margin funds prefit + spill + headroom
                      beyond the physical slack — the block semantics make
                      in-block slack invisible to runtime cudaMalloc.
        """
        s, k = self.shape, self.k
        ftmpl = templates.FAMILY[templates.family_of(s.architecture)]
        backend = self.attention_backend()
        gguf = "gguf" in str(self.quant.get("weights", "")).lower()
        kv_sharded = (tp >= 2 and kv_mode == "sharded")

        b = b_cap = int(ftmpl.get(
            "max_batch_size", templates.COMMON["orchestrator"]["max_batch_size"]))
        budget = int(k.block_table_budget_gb * GIB)
        while b > 16 and sizing.block_tables_bytes(
                s, max_seq, b, k.page_size_tokens) > budget:
            b //= 2

        def scratch_at(sc: int) -> "sizing.RuntimeScratch":
            return sizing.runtime_scratch(
                s, self.kv_quant, backend, max_seq=max_seq, max_batch=b,
                superchunk=sc, tp=tp, rank0=True, tiering_on=tiering_on,
                kv_sharded=kv_sharded, page_size_tokens=k.page_size_tokens,
                indexer_k_page_size_tokens=k.indexer_k_page_size_tokens,
                dcp_chunk_size=k.dcp_chunk_size)

        tail_mib = int(k.prefill_scratch_gb * 1024)
        base_margin = k.vram_safety_margin_gb
        quantum = max(k.runtime_margin_quantum_gb, 0.01)

        def attn_margin_for(sc: int):
            """(margin_gb, headroom_mib, scratch, spill_mib) funding stride
            sc on the TP hosts.  Above the chunk floor the engine's fit
            gate (spill + headroom <= free, command_dispatcher.cpp:568)
            governs; at/below it the loop is skipped and the requirement
            is simply that the post-check allocations + a free floor fit."""
            rs = scratch_at(sc)
            trans = sizing.moe_big_transient_bytes(
                s, max(sc, 512), tp, gguf) // sizing.MIB
            spill = max(0, trans - tail_mib)
            head = max(1024, math.ceil(rs.postcheck_mib
                                       * k.moe_fit_headroom_cushion))
            need = spill + (head if sc > 512 else
                            rs.postcheck_mib + k.moe_fit_free_floor_mib)
            short = need - self._fit_free_mib(
                ordered[0], base_margin, prefit_mib=rs.prefit_mib,
                expert_only=False)
            margin = base_margin
            if short > 0:
                margin = base_margin + math.ceil(
                    short / 1024 / quantum) * quantum
            return margin, head, rs, spill

        superchunk = 0
        headroom_exp = 0
        if "prefill_superchunk_tokens" in ftmpl and s.num_moe_layers > 0 \
                and s.n_routed_experts > 0:
            hosts = ordered[tp:]
            superchunk = 512
            for cand in (512, 1024, 2048, 4096, 8192, 16384, 32768):
                if s.index_kpool > 1 and cand % s.index_kpool:
                    continue
                trans = sizing.moe_big_transient_bytes(
                    s, cand, tp, gguf) // sizing.MIB
                post_exp = sizing.moe_persistent_bytes(
                    s, max(cand, b)) // sizing.MIB + 2
                h_exp = max(256, math.ceil(post_exp
                                           * k.moe_fit_headroom_cushion))
                pf_exp = sizing.pair_buffer_bytes(
                    s, b, cand, expert_only=True) // sizing.MIB
                ok = all(trans + h_exp <= self._fit_free_mib(
                             gh, base_margin, prefit_mib=pf_exp,
                             expert_only=True) for gh in hosts)
                if ok:
                    m_need, _, _, _ = attn_margin_for(cand)
                    ok = m_need <= k.stride_margin_cap_gb
                if ok:
                    superchunk = cand
                    headroom_exp = h_exp if h_exp > 256 else 0
                elif cand >= 512:
                    break
        margin, head_attn, rs, spill = attn_margin_for(superchunk or 512)
        return LongCtxPlan(
            max_batch=b, superchunk=superchunk,
            headroom_attn_mib=(head_attn if superchunk > 512 else 0),
            headroom_exp_mib=headroom_exp,
            margin_gb=margin, scratch=rs, spill_mib=spill,
            notes=((f"max_batch_size {b_cap}->{b}: dev_block_tables budget",)
                   if b != b_cap else ()))

    def _fit_attempt(self, ordered: list[GpuInfo], a: Attempt,
                     setup: TpSetup) -> list[GpuPlan]:
        """The E1 carve at ONE lattice point; raises Infeasible when it does
        not fit. Pure sizing — the caller decides whether it is explained."""
        k = self.k
        draft_on = not a.draft_off
        lc = self._longctx_plan(ordered, a.tp, a.conc, a.max_seq,
                                a.tiering_on, setup.kv_mode)
        # TP residual (for ratio-None policy and overflow caps), probed
        # with a zero-expert plan first
        probe = self._tp_fit(ordered[0], a.tp, a.conc, a.max_seq, draft_on,
                             setup.draft_bytes, 0, a.idx_mode,
                             a.tiering_on, setup.kv_mode,
                             margin_gb=lc.margin_gb)
        targets = self.expert_targets(ordered, a.tp, probe.slack())
        plans: list[GpuPlan] = []
        for i, g in enumerate(ordered):
            eb = int(_round_gib(targets[g.ordinal] * self.slot_bytes,
                                k.expert_gib_quantum) * GIB)
            if i < a.tp:
                plans.append(self._tp_fit(g, a.tp, a.conc, a.max_seq, draft_on,
                                          setup.draft_bytes, eb, a.idx_mode,
                                          a.tiering_on, setup.kv_mode,
                                          margin_gb=lc.margin_gb))
                continue
            p = GpuPlan(ordinal=g.ordinal, gpu_type=g.gpu_type,
                        usable_bytes=self.usable_bytes(g))
            p.margin_bytes = int(k.vram_safety_margin_gb * GIB)
            p.pinned_bytes = int(k.non_tp_overhead_gib * GIB)
            if draft_on and i in setup.draft_ranks:
                p.draft_bytes = setup.draft_bytes
            p.expert_bytes = eb
            p.expert_slots = sizing.expert_slots_in_zone(eb, self.slot_bytes)
            if p.slack() < 0:
                raise Infeasible(
                    "vram-carve-overflow", f"gpu {g.ordinal}",
                    f"{p.total_committed() / GIB:.2f} GiB",
                    f"{p.usable_bytes / GIB:.2f} GiB",
                    "lower vram_expert_ratio")
            plans.append(p)
        return plans

    @staticmethod
    def _first_feasible(lattice: list[Attempt], prefer: str, evaluate):
        """Walk the lattice in `prefer` order and stop at the first fit.

        Returns (winner|None, visited, last_err) where `visited` is the
        prefix actually tried (warnings belong to those attempts alone) and
        `last_err` is the failure immediately preceding the winner — the one
        that forced whatever escalation the winner represents."""
        visited: list[Attempt] = []
        last_err: "Infeasible | None" = None
        for a in sorted(lattice, key=lambda x: _order_key(prefer, x)):
            visited.append(a)
            plans, err = evaluate(a)
            if plans is not None:
                return a, visited, last_err
            last_err = err
        return None, visited, last_err

    # ---------------------------------------------------------------- solve

    def _explain_preference(self, chosen: Attempt,
                            alt: "Attempt | None") -> None:
        """§2.3: NAME the preference exactly when it changed the outcome.

        Silent when it did not — a knob that narrates itself for doing
        nothing is noise, and one that silently reorders is worse than no
        knob at all. `alt` is what `balanced` would have derived from the
        same lattice (the counterfactual is free: the search memoises)."""
        prefer = self.levers.prefer
        if prefer == "balanced" or alt is None or chosen == alt:
            return
        idx_price = "~8-9% of served prefill"
        tp_price = "~0.5-2% of the decode wall"
        rule = {
            "speed": ("prefer=speed spends every free capacity lever before "
                      "it buys a priced slowdown"),
            "capacity": ("prefer=capacity spends every priced slowdown before "
                         "it touches the requested context/concurrency"),
        }[prefer]
        moves: list[str] = []
        refs = ["AUTOCONFIG §2.3", "AUTOCONFIG §5 E1", "TD-AUTOCONFIG-SPEED-BUDGET"]
        if chosen.idx_mode != alt.idx_mode:
            row = ec.get("local-indexer-prefill-cost")
            if row is not None:
                refs.append(row.id)
            if chosen.idx_idx > alt.idx_idx:
                moves.append(f"took `{chosen.idx_mode}` indexer mode early "
                             f"because prefer={prefer}; the price is "
                             f"{idx_price}")
            else:
                moves.append(f"declined `{alt.idx_mode}` indexer mode and its "
                             f"{idx_price}, keeping `{chosen.idx_mode}`")
        if chosen.tp != alt.tp:
            row = ec.get("glm5next-tp1-default")
            if row is not None:
                refs.append(row.id)
            if chosen.tp_idx > alt.tp_idx:
                moves.append(f"escalated to TP={chosen.tp} early because "
                             f"prefer={prefer}; the price is {tp_price}")
            else:
                moves.append(f"declined TP={alt.tp} and its {tp_price}, "
                             f"staying at TP={chosen.tp}")
        if chosen.draft_off != alt.draft_off:
            moves.append("kept the dspark draft" if not chosen.draft_off
                         else "dropped the dspark draft")
        if chosen.tiering_on != alt.tiering_on:
            moves.append("enabled KV tiering (windowed admission)"
                         if chosen.tiering_on else "kept KV tiering off")
        if (chosen.conc, chosen.max_seq) != (alt.conc, alt.max_seq):
            moves.append(f"serves {chosen.conc} x {chosen.max_seq} tokens "
                         f"where `balanced` serves {alt.conc} x {alt.max_seq}")
        self.ex.add(
            "autoconfig.prefer", prefer, "searched",
            f"prefer={prefer} CHANGED the E1 outcome — {rule}: "
            + "; ".join(moves)
            + ". The preference only ORDERS the search — every preference "
              "walks the same set of candidate fits, so it can never cause a "
              "refusal another preference would have avoided, and it buys "
              "nothing by arithmetic (there is no exchange rate between "
              "tok/s and MiB — TD-AUTOCONFIG-SPEED-BUDGET)",
            refs=tuple(refs))

    # ---------------------------------------------------- pins (design §3-§6)

    def _validate_pins(self, ordered: list[GpuInfo]) -> None:
        """Solve-time pin validation + the pinned explain rows.

        By registry-row KIND (design §4): `measured` rows are overridable —
        LOUDLY, quoting pin and row; `engine_gate`/`validator` rows and model
        geometry REFUSE, quoting both. Runs once, before the search."""
        assert self.pins is not None
        s = self.shape
        fam = templates.family_of(s.architecture)
        refs_base = ("TD-AUTOCONFIG-PINNED-CONSTRAINTS",
                     "spec/plans/AUTOCONFIG_PINNED_CONSTRAINTS.md",
                     "AUTOCONFIG §2.5")
        for path, pin in self.pins.items():
            self.ex.pin(
                path, pin.value,
                f"PINNED by the user ({pin.source}): a HARD CONSTRAINT the "
                "solver treated as fixed and solved around — told, not "
                "derived. It survives every E1 rung; an infeasible fit "
                "refuses naming this pin rather than relaxing it",
                refs=refs_base)

        p = self._pin("compute.attention_backend")
        if p is not None:
            menu = tuple(templates.ACCURACY_LADDER[fam])
            if p.value not in menu:
                raise Infeasible(
                    "pinned-backend-not-serveable", "numerics",
                    f"pin compute.attention_backend={p.value!r} ({p.source})",
                    f"the engine's serveable partition for {s.architecture} "
                    f"is {{{', '.join(menu)}}} "
                    "(config_validator.cpp:510-534)",
                    "pin one of the serveable backends, or drop the pin")
            gate = ec.get("glm5next-tq-backend-unwired")
            if (gate is not None and gate.scope == s.architecture
                    and p.value == "turboquant_mla"):
                raise Infeasible(
                    "pinned-backend-engine-gated", "numerics",
                    f"pin compute.attention_backend={p.value!r} ({p.source})",
                    f"engine gate '{gate.id}': {gate.statement}",
                    "an engine_gate registry row is not overridable by a "
                    "pin — delete the row when the gate falls, or pin a "
                    "serveable backend")
            tier = self.levers.accuracy
            if tier != "standard":
                floor = self._backend_for_tier(tier)   # superior raises here
                if floor != p.value:
                    raise Infeasible(
                        "pin-conflicts-accuracy-floor", "numerics",
                        f"pin compute.attention_backend={p.value!r} "
                        f"({p.source}) AND accuracy={tier} (floor: "
                        f"`{floor}`)",
                        "one numerics axis, two contradictory hard asks — "
                        "the accuracy tier refuses rather than approximates "
                        "(the `superior` precedent), and the E1 ladder "
                        "never degrades the backend",
                        f"drop the pin (accuracy={tier} then derives "
                        f"`{floor}`) or drop the tier (the pin then "
                        "removes the numerics axis)")
            std = self._backend_for_tier("standard")
            arow = ec.get("mla-tq-vs-snapmla-accuracy")
            if p.value != std and arow is not None and not s.is_v4:
                self._warn(
                    f"pin compute.attention_backend={p.value!r} moves the "
                    f"backend off the standard tier's `{std}` — measured "
                    f"registry row '{arow.id}' prices the step "
                    f"({_MLA_ACCURACY_HEADLINE}) [{arow.site}] — deliberate "
                    "override assumed, never silent "
                    "(TD-AUTOCONFIG-PINNED-CONSTRAINTS (a))")

        p = self._pin("hardware.dcp_kv_mode")
        if p is not None and p.value == "sharded" and s.is_v4:
            row = ec.get("v4-no-sharded-kv")
            if row is not None:
                raise Infeasible(
                    "pinned-kv-mode-engine-gated", "dcp kv mode",
                    f"pin hardware.dcp_kv_mode='sharded' ({p.source})",
                    f"engine gate '{row.id}': {row.statement} [{row.site}]",
                    "an engine_gate registry row is not overridable by a "
                    "pin — pin 'replicated' or drop the pin")
        tp_pin = self._pin("parallelism.tensor_parallelism")
        if p is not None and tp_pin is not None:
            # a COMBINATION gate (model_constraints.json) fires under the
            # PINNED combination and contradicts the kv-mode pin: refuse at
            # solve() entry quoting both pins and the row, exactly like
            # v4+sharded — never a silent override
            # (TD-AUTOCONFIG-COMBO-CONSTRAINTS (c)/(d)). With tp unpinned
            # the gated tp lanes instead leave the plan (adapt-around,
            # _combo_filter_tp_plan).
            crow = cc.forcing("hardware.dcp_kv_mode",
                              self._combo_ctx(int(tp_pin.value)))
            if crow is not None:
                forced = crow.forced("hardware.dcp_kv_mode")
                if forced != p.value:
                    raise Infeasible(
                        "pinned-kv-mode-engine-gated", "dcp kv mode",
                        f"pin hardware.dcp_kv_mode={p.value!r} ({p.source}) "
                        f"with pin parallelism.tensor_parallelism="
                        f"{tp_pin.value} ({tp_pin.source})",
                        f"combination gate '{crow.id}' "
                        f"({crow.when_text()}): {crow.statement} "
                        f"[{crow.site}]",
                        "an engine_gate registry row is not overridable by "
                        f"a pin — pin {forced!r}, lower the pinned tp, or "
                        "drop a pin")

        p = self._pin("parallelism.tensor_parallelism")
        if p is not None:
            tp = int(p.value)
            top = ordered[0]
            run = sum(1 for g in ordered if g.vram_mib == top.vram_mib
                      and g.gpu_type == top.gpu_type)
            if tp > run:
                vrow = ec.get("tp-gpus-top-class-only")
                raise Infeasible(
                    "pinned-tp-exceeds-top-class-run", "tp group",
                    f"pin parallelism.tensor_parallelism={tp} ({p.source})",
                    f"only {run} x {top.gpu_type} in the box's top attention "
                    "class"
                    + (f" — validator '{vrow.id}': {vrow.statement} "
                       f"[{vrow.site}]" if vrow is not None else ""),
                    "a validator rule is not overridable by a pin — lower "
                    "the pinned degree or add top-class GPUs")
            if (s.num_attention_heads % tp != 0
                    or (s.linear_attn and s.linear_attn.num_heads % tp != 0)
                    or (s.o_groups > 1 and s.o_groups % tp != 0)):
                raise Infeasible(
                    "pinned-tp-head-divisibility", "tp group",
                    f"pin parallelism.tensor_parallelism={tp} ({p.source})",
                    f"model geometry: {s.num_attention_heads} attention "
                    "heads (and KDA heads / o_groups where present) must "
                    "divide by tp (config_validator.cpp:637-680)",
                    "pin a divisor of the head counts, or drop the pin")
            if s.num_attention_heads // tp > 128:
                raise Infeasible(
                    "pinned-tp-heads-per-rank-cap", "tp group",
                    f"pin parallelism.tensor_parallelism={tp} ({p.source})",
                    f"{s.num_attention_heads} heads / tp {tp} > 128 "
                    "heads/rank (config_validator.cpp:677-680)",
                    "raise the pinned degree, or drop the pin")
            trow = ec.get("glm5next-tp1-default")
            if tp > 1 and s.is_glm5_next and trow is not None:
                self._warn(
                    f"pin parallelism.tensor_parallelism={tp} OVERRIDES "
                    f"measured registry row '{trow.id}': glm5_next serves at "
                    "TP=1 by default (KDA decode is latency-floored; "
                    "combines cost ~0.5-2% of the decode wall; TP is for "
                    f"CAPACITY, never speed) [{trow.site}] — deliberate "
                    "override assumed, never silent "
                    "(TD-AUTOCONFIG-PINNED-CONSTRAINTS (a))")

        p = self._pin("serving.max_sequence_length")
        if p is not None and int(p.value) > s.max_position_embeddings:
            raise Infeasible(
                "pinned-max-seq-exceeds-model", "model geometry",
                f"pin serving.max_sequence_length={p.value} ({p.source})",
                f"max_position_embeddings {s.max_position_embeddings}",
                "model geometry is not overridable by a pin — lower the "
                "pinned length")

        p = self._pin("speculation.method")
        if p is not None:
            if p.value not in ("none", "dspark"):
                raise Infeasible(
                    "pinned-method-not-modeled", "speculation",
                    f"pin speculation.method={p.value!r} ({p.source})",
                    "the solver sizes only 'none' and 'dspark' — it must "
                    "never price a method it cannot size (design §3/§9)",
                    "use a hand recipe for other methods")
            if p.value == "dspark" and self.draft is None:
                raise Infeasible(
                    "pinned-method-needs-draft", "speculation",
                    f"pin speculation.method='dspark' ({p.source})",
                    "no dspark checkpoint is discoverable for this model "
                    "(and none was passed with --draft)",
                    "point --draft at a matching .dspark checkpoint, or "
                    "drop the pin")

        p = self._pin("memory.kv_tiering.enabled")
        if p is not None and bool(p.value):
            if not s.has_dsa:
                raise Infeasible(
                    "pinned-tiering-needs-dsa", "kv tiering",
                    f"pin memory.kv_tiering.enabled=true ({p.source})",
                    "the model has no DSA indexer (index_topk == 0) — there "
                    "is no hot-window selection to tier on",
                    "drop the pin; tiering is undefined for this model")
            fmt_row = ec.get("tiering-needs-row-self-contained-format")
            backend = self.attention_backend()
            if (fmt_row is not None and not s.is_v4
                    and backend not in ("snapmla", "turboquant_mla")):
                raise Infeasible(
                    "pinned-tiering-format-gated", "kv tiering",
                    f"pin memory.kv_tiering.enabled=true ({p.source}) with "
                    f"attention_backend `{backend}`",
                    f"engine gate '{fmt_row.id}': {fmt_row.statement} "
                    f"[{fmt_row.site}]",
                    "an engine_gate registry row is not overridable by a "
                    "pin — use a row-self-contained KV backend or drop "
                    "the pin")

    def _any_fit(self, ordered: list[GpuInfo]) -> "Attempt | None":
        """Feasibility probe under the CURRENT self._relaxed set: quiet,
        fresh caches, fixed walk order (feasibility is order-independent).
        Used only by the refusal-attribution pass (design §6)."""
        with self._quiet():
            try:
                c0, m0 = self.active_split()
                tps = self.tp_plan(ordered, m0)
                setups = {tp: self._tp_setup(ordered, tp) for tp in tps}
                lat = self._lattice(tps, setups, c0, m0)
                for a in sorted(lat, key=lambda x: _order_key("balanced", x)):
                    try:
                        self._fit_attempt(ordered, a, setups[a.tp])
                        return a
                    except Infeasible:
                        continue
            except Infeasible:
                return None
        return None

    def _refuse_naming_binding_pins(self, ordered: list[GpuInfo]) -> None:
        """Design §6: WHICH pin binds, never just 'infeasible'. Relax each
        pin alone; the ones whose relaxation restores a fit are BINDING.
        Falls through (returns) when the box is infeasible regardless of
        pins, so the ordinary refusal keeps its exact text."""
        assert self.pins is not None
        binding: list[tuple[str, "Attempt"]] = []
        prev = self._relaxed
        try:
            for path in self.pins.paths():
                self._relaxed = frozenset({path})
                fit = self._any_fit(ordered)
                if fit is not None:
                    binding.append((path, fit))
            if not binding:
                self._relaxed = frozenset(self.pins.paths())
                joint = self._any_fit(ordered)
            else:
                joint = None
        finally:
            self._relaxed = prev
        if binding:
            names = "; ".join(
                f"{p}={self.pins.get(p).value!r}" for p, _ in binding)
            first_path, first_fit = binding[0]
            raise Infeasible(
                "pinned-constraint-binds", f"pin {names}",
                f"the pin set {self.pins.describe()}",
                f"a fit EXISTS without {first_path} (e.g. tp={first_fit.tp},"
                f" {first_fit.conc} x {first_fit.max_seq} tokens, "
                f"dcp_indexer_mode={first_fit.idx_mode}"
                + ("" if first_fit.draft_off else ", draft on") + ")",
                "drop or loosen the named pin(s), free VRAM (lower "
                "vram_expert_ratio / drop the draft), or lower the ask — "
                "a pin is never quietly relaxed")
        if joint is not None:
            raise Infeasible(
                "pinned-constraints-jointly-bind", "the pin set",
                f"the pin set {self.pins.describe()}",
                "no SINGLE pin is binding, but a fit exists with the whole "
                f"set relaxed (e.g. tp={joint.tp}, {joint.conc} x "
                f"{joint.max_seq} tokens)",
                "loosen the pin set — the pins are only jointly satisfiable "
                "by a bigger box")

    def solve(self) -> SolveResult:
        ordered = self.order_gpus()
        if self.pins is not None:
            self._validate_pins(ordered)
        conc0, max_seq0 = self.active_split()
        tp_candidates = self.tp_plan(ordered, max_seq0)
        with self._quiet():
            setups = {tp: self._tp_setup(ordered, tp) for tp in tp_candidates}
        lattice = self._lattice(tp_candidates, setups, conc0, max_seq0)

        cache: dict = {}

        def evaluate(a: Attempt):
            hit = cache.get(a)
            if hit is None:
                with self._quiet() as warns:
                    try:
                        hit = (self._fit_attempt(ordered, a, setups[a.tp]),
                               None, list(warns))
                    except Infeasible as e:
                        hit = (None, e, list(warns))
                cache[a] = hit
            return hit[0], hit[1]

        prefer = self.levers.prefer
        winner, visited, last_err = self._first_feasible(lattice, prefer, evaluate)
        if winner is None:
            if self.pins is not None:
                # design §6: name WHICH pin binds (returns when the box is
                # infeasible regardless of pins, keeping the refusal below)
                self._refuse_naming_binding_pins(ordered)
            # Nothing fits anywhere in the lattice, so the refusal is
            # preference-independent (§6) — and so is the number it reports:
            # the VRAM-CHEAPEST corner (every priced option spent, the ask
            # shed to the floor, no draft) is the point that got closest, so
            # its `affordable` is the largest the box can actually reach.
            # Taking "the last attempt tried" instead would make the reported
            # headroom depend on the search order, which is exactly the kind
            # of silent preference effect this lever must not have.
            cheapest = max(lattice, key=lambda a: (a.tp_idx, a.idx_idx,
                                                   a.cap_idx, a.draft_off))
            err = cache[cheapest][1] or last_err
            assert err is not None
            raise err
        # the counterfactual the explanation needs, at no extra sizing cost
        alt = winner if prefer == "balanced" else \
            self._first_feasible(lattice, "balanced", evaluate)[0]

        # warnings belong to the attempts the chosen order actually visited;
        # a TP the search moved away from left none of its own behind
        mark = len(self.warnings)
        prev_tp: "int | None" = None
        for a in visited:
            if prev_tp is not None and a.tp != prev_tp:
                del self.warnings[mark:]   # a rejected TP's warnings are not ours
            prev_tp = a.tp
            for w in cache[a][2]:
                if w not in self.warnings:
                    self.warnings.append(w)

        # re-run the WINNER out loud: this is the derivation, everything
        # above was search
        setup = self._tp_setup(ordered, winner.tp,
                               explain_draft=not winner.draft_off)
        plans = self._fit_attempt(ordered, winner, setup)
        if winner.note:
            self.degradations.append(winner.note)

        # TD-AUTOCONFIG-MAXSEQ-IGNORES-MAPPED-KDA: REFUSE OR REDUCE, never
        # advertise. A solver that emits a max_sequence_length its own
        # recipe's admission path refuses is worse than one that refuses to
        # emit it — the user learns at request time, hundreds of thousands
        # of tokens into a prefill. When the winning fit's single-request
        # ceiling binds below the ask, reduce max_seq to the exact ceiling
        # (page-floored) and say which term bound; below the degradation
        # floor, refuse outright.
        for _ in range(4):   # pool shares shrink with max_seq: fixed point
            adm = min((p.admissible_max_seq for p in plans[:winner.tp]
                       if p.admissible_max_seq > 0), default=0)
            if adm <= 0 or adm >= winner.max_seq:
                break
            asked = winner.max_seq
            p0 = plans[0]
            seq_pin = self._pin("serving.max_sequence_length")
            if seq_pin is not None:
                # the reduce loop must never shave a pinned value (design
                # §6): refuse with the full admissibility decomposition,
                # quoting the pin and the pool-model row
                krow = ec.get("kda-state-mapped-tenant")
                raise Infeasible(
                    "pinned-max-seq-not-admissible", "tp gpus",
                    f"pin serving.max_sequence_length={asked} "
                    f"({seq_pin.source})",
                    f"single-request admissibility ceiling {adm} tokens: "
                    f"the shared kMain pool holds {p0.adm_pool_pages} pages "
                    f"but ONE {asked}-token request's whole-life demand is "
                    f"{p0.adm_demand_pages} = KV {p0.adm_kv_pages} + "
                    f"indexer reservation {p0.adm_idx_pages} + mapped KDA "
                    f"state {p0.adm_state_pages}"
                    + (f" (registry row '{krow.id}': KV + indexer + mapped "
                       "state all draw on the ONE pool)" if krow is not None
                       else ""),
                    "free VRAM (lower vram_expert_ratio / drop the draft), "
                    "escalate TP, or lower the pinned length — the pin is "
                    "never quietly reduced")
            if adm < self.k.min_max_seq:
                raise Infeasible(
                    "max-seq-not-admissible-mapped-kda", "tp gpus",
                    f"max_sequence_length {asked}",
                    f"{adm} tokens (pool {p0.adm_pool_pages} kMain pages vs "
                    f"one request's {p0.adm_demand_pages} = KV "
                    f"{p0.adm_kv_pages} + indexer {p0.adm_idx_pages} + "
                    f"mapped KDA state {p0.adm_state_pages})",
                    "free VRAM (lower vram_expert_ratio / drop the draft) "
                    "or lower the ask")
            new_ms = max(self.k.min_max_seq,
                         adm // self.k.page_size_tokens
                         * self.k.page_size_tokens)
            winner = replace(winner, max_seq=new_ms)
            plans = self._fit_attempt(ordered, winner, setup)
            note = (f"max_sequence_length {asked}->{new_ms}: single-request "
                    f"admissibility ceiling")
            self.degradations.append(note)
            self.ex.add(
                "serving.max_sequence_length", new_ms, "closed_form",
                f"REDUCED from {asked}: the shared kMain pool holds "
                f"{p0.adm_pool_pages} pages but ONE {asked}-token request's "
                f"whole-life admission demand is {p0.adm_demand_pages} pages "
                f"= KV {p0.adm_kv_pages} + full-length indexer reservation "
                f"{p0.adm_idx_pages} + mapped KDA state "
                f"{p0.adm_state_pages} (all three are tenants of the ONE "
                f"pool since the mapped-state default) — admissible ceiling "
                f"{adm} tokens, floored to page granularity. Advertising "
                f"{asked} would refuse at request time, at full prefill "
                f"cost",
                refs=("kda-state-mapped-tenant",
                      "TD-AUTOCONFIG-MAXSEQ-IGNORES-MAPPED-KDA",
                      "TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA"))

        # The admissibility arithmetic is ALWAYS shown (a 0.66% miss must be
        # visible pre-boot, and so must the margin when it clears); the
        # engine prints the same figures at boot and enginecheck compares.
        p0 = plans[0]
        if p0.adm_pool_pages > 0:
            margin = p0.adm_pool_pages - p0.adm_demand_pages
            self.ex.add(
                "autoconfig.context_admissibility",
                f"ceiling {p0.admissible_max_seq} tokens "
                f"(margin {margin} pages)",
                "closed_form",
                f"one max-length ({winner.max_seq}-token) request's "
                f"whole-life admission demand is {p0.adm_demand_pages} kMain "
                f"pages = KV {p0.adm_kv_pages} + indexer reservation "
                f"{p0.adm_idx_pages} + mapped KDA state "
                f"{p0.adm_state_pages}, against a {p0.adm_pool_pages}-page "
                f"shared pool (margin {margin} pages) — single-request "
                f"ceiling {p0.admissible_max_seq} tokens; the engine "
                f"computes and logs the same ceiling at boot "
                f"('admissible context (single request)') and the "
                f"enginecheck row pins the two together",
                refs=("kda-state-mapped-tenant",
                      "TD-AUTOCONFIG-MAXSEQ-IGNORES-MAPPED-KDA",
                      "TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA"))

        if len(tp_candidates) > 1 and winner.tp_idx == 0 \
                and tp_candidates[0] != 1:
            # P-31: ceiling-first plan for a deep-context ask (see tp_plan)
            self.ex.add("parallelism.tensor_parallelism", winner.tp, "measured",
                        f"head-divisible ceiling ordered FIRST: the ask's "
                        f"max_sequence_length {winner.max_seq} >= "
                        f"{self.TP_CEILING_FIRST_MAX_SEQ} is prefill-"
                        "dominated, and attention is the measured majority "
                        "of the large-prefill wall (60.4% at S=2048, ~73% "
                        "at S=6144 — P-30 steps 1-4, stride-invariant); "
                        "head-sharding is the only lever that touches it. "
                        "The tp1-default row's flat-prefill evidence is one "
                        "141-token shape (its own evidence bound) and does "
                        "not govern this regime; TP=1 remains the fallback",
                        refs=("glm5next-tp1-default", "INV-KDA-TP",
                              "AUTOCONFIG §5 P2"),
                        measurement="P-30 steps 1-4, "
                                    "spec/measurements/glm53_flash.md")
        elif len(tp_candidates) > 1 and winner.tp_idx == 0:
            self.ex.add("parallelism.tensor_parallelism", winner.tp, "measured",
                        "TP=1 is the measured glm5_next default: the KDA "
                        "decode kernel is latency-floored (6.07 us at "
                        "H=64/TP1 vs 5.89 us at H=32/rank TP2 — halving "
                        "the heads buys ~nothing) while 45 per-layer bf16 "
                        "combines cost ~0.5-2 ms/token (~0.5-2% of a "
                        "10 tok/s wall). The head-divisible ceiling "
                        f"({tp_candidates[-1]}) stays available for CAPACITY "
                        "and is taken automatically when TP=1 does not fit",
                        refs=("glm5next-tp1-default", "INV-KDA-TP"),
                        measurement="GF3.10, spec/measurements/glm53_flash.md")
        elif winner.tp_idx > 0 and winner.tp == 1:
            self.ex.add("parallelism.tensor_parallelism", winner.tp,
                        "searched",
                        "fell back to TP=1: the ceiling-first plan's "
                        f"TP={tp_candidates[0]} did not fit the ask",
                        refs=("glm5next-tp1-default", "AUTOCONFIG §5 E1"))
        elif winner.tp_idx > 0:
            why = (f"infeasible ({last_err.constraint_id} on "
                   f"{last_err.binding})" if last_err is not None
                   else "infeasible")
            if prefer == "capacity":
                # under prefer=capacity the escalation happens at the FULL ask,
                # before the ladder would have shed anything — say so, or the
                # sentence overclaims that no rung at the lower TP could fit
                why += (" at the requested context/concurrency; prefer="
                        "capacity buys TP rather than shed the ask, so TP="
                        f"{tp_candidates[0]} was not re-tried at reduced "
                        "capacity first")
            self.ex.add("parallelism.tensor_parallelism", winner.tp, "searched",
                        f"escalated from the measured default TP="
                        f"{tp_candidates[0]} to {winner.tp} "
                        "for CAPACITY, NOT SPEED: the fit at TP="
                        f"{tp_candidates[0]} was "
                        f"{why}. TP is correct on this arch "
                        "and buys VRAM headroom; it is still expected to "
                        "cost ~0.5-2% of the decode wall in combines",
                        refs=("glm5next-tp1-default", "INV-KDA-TP",
                              "AUTOCONFIG §5 E1"),
                        measurement="GF3.10, spec/measurements/glm53_flash.md")
        else:
            self.ex.add("parallelism.tensor_parallelism", winner.tp, "closed_form",
                        self._tp_ceiling_why,
                        refs=("config_validator.cpp:655-680",
                              "tp-gpus-top-class-only"))

        self._explain_indexer_mode(
            winner.idx_mode, setup.idx_modes, winner.tp, winner.max_seq,
            min((p.slack() for p in plans[:winner.tp]), default=0),
            last_err if winner.idx_idx > 0 else None)
        if self.degradations:
            self.ex.add("serving.max_concurrent_requests", winner.conc, "searched",
                        "E1 degradation: " + "; ".join(self.degradations),
                        refs=("AUTOCONFIG §5 E1/§6",))
        self._explain_tiering(winner, setup,
                              last_err if winner.tier_idx > 0 else None)
        self._explain_preference(winner, alt)

        fit = TpFit(tp=winner.tp, kv_mode=setup.kv_mode,
                    idx_mode=winner.idx_mode, tiering_on=winner.tiering_on,
                    conc=winner.conc, max_seq=winner.max_seq,
                    draft_on=not winner.draft_off,
                    draft_ranks=setup.draft_ranks, plans=plans)

        lc = self._longctx_plan(ordered, winner.tp, winner.conc,
                                winner.max_seq, winner.tiering_on,
                                setup.kv_mode)
        recipe = self._assemble(ordered, fit.tp, fit.kv_mode, fit.idx_mode,
                                fit.tiering_on, fit.conc, fit.max_seq,
                                fit.draft_on, self.draft, fit.draft_ranks,
                                fit.plans, lc)
        self._host_arena_check(ordered, fit.tp)
        return SolveResult(recipe=recipe, explanations=self.ex,
                           warnings=self.warnings, gpu_plans=fit.plans,
                           degradations=self.degradations)

    # ------------------------------------------------------------ host arena

    def _host_arena_check(self, ordered, tp) -> None:
        s, k = self.shape, self.k
        need = sizing.host_arena_total_bytes(self.slot_bytes, s.n_routed_experts,
                                             s.num_moe_layers)
        gpu_nodes = sorted({g.numa_node for g in self.hw.gpus if g.numa_node >= 0})
        by_node = {n.node: n for n in self.hw.numa_nodes}
        cap = 0
        for n in gpu_nodes:
            node = by_node.get(n)
            if node:
                # plan_geometry: min(total/num_gpu_nodes, resolved fraction cap)
                cap += min(need // max(len(gpu_nodes), 1),
                           int(k.host_pin_fraction_total * node.mem_total_kib * 1024))
        tp_nodes = {ordered[i].numa_node for i in range(tp)}
        # spill set = same rule the emitted cross_node_spill uses (H1): every
        # memory node not local to a TP GPU; gpu-attached ones already counted
        for node in self.hw.numa_nodes:
            n = node.node
            if n in tp_nodes or n in gpu_nodes:
                continue
            if node.is_hbm_bank:
                # BOOT-TIME expectation, not current free: fraction_free
                # resolves against a mostly-empty bank at plan time (current
                # free is polluted by whatever runs now — the ±slots drift
                # lesson of INV-ARENA-CACHE-ORDER)
                cap += int(k.hbm_spill_fraction_free * 0.95
                           * node.mem_total_kib * 1024)
            else:
                cap += int(k.host_pin_fraction_total * node.mem_total_kib * 1024)
        if cap < need:
            short = 100.0 * (need - cap) / need
            if short < 5.0:
                self.warnings.append(
                    f"host expert arena: need {need / 1e9:.0f} GB, pinnable plan "
                    f"capacity ~{cap / 1e9:.0f} GB — warm store ~{short:.1f}% "
                    f"partial (cold misses fetch from disk); raise HBM "
                    f"fraction_free only with the node-local OOM risk in mind "
                    f"(registry row hbm-fraction-free-06)")
            else:
                self.warnings.append(
                    f"host expert arena: need {need / 1e9:.0f} GB, pinnable plan "
                    f"capacity ~{cap / 1e9:.0f} GB ({short:.0f}% short) — the "
                    f"engine may refuse (INV-4.12f zero-slot) or run a largely "
                    f"cold store; consider nvme_tier or a smaller-quant model")
        self.ex.add("memory.pin_host_expert_pool_sizing",
                    {"mode": "fraction_total", "value": k.host_pin_fraction_total},
                    "measured",
                    f"host arena {need / 1e9:.0f} GB across GPU nodes {gpu_nodes} "
                    f"+ spill (capacity ~{cap / 1e9:.0f} GB); per-NUMA-node "
                    "capacity is the real ceiling, not RLIMIT_MEMLOCK",
                    refs=("INV-4.12f", "pinned_expert_arena.cpp:315-388"))

    # ---------------------------------------------------------- accuracy §2.4

    def _explain_accuracy_tier(self, backend: str) -> None:
        """Name the `accuracy` tier EXACTLY when it changed the derived
        backend (the `prefer` silence discipline: a knob that narrates
        itself for doing nothing is noise).  Called only when
        backend != the standard tier's choice."""
        tier = self.levers.accuracy
        std = self._backend_for_tier("standard")
        fam = templates.family_of(self.shape.architecture)
        ladder = templates.ACCURACY_LADDER[fam]
        arow = ec.get("mla-tq-vs-snapmla-accuracy")
        refs: list[str] = ["AUTOCONFIG §2.4", "TD-AUTOCONFIG-ACCURACY-LEVER"]
        if not self.shape.is_v4:
            b_new = sizing.kv_bytes_per_token(self.shape, self.kv_quant, backend)
            b_std = sizing.kv_bytes_per_token(self.shape, self.kv_quant, std)
            price = (f"KV {b_new} B/row vs {b_std} B/row under `{std}` "
                     f"({b_new / b_std:.2f}x)")
            if arow is not None:
                kind = "measured"
                basis = ("the measured gap (registry row "
                         f"'{arow.id}': {_MLA_ACCURACY_HEADLINE})")
                refs.append(arow.id)
                meas = arow.site
            else:
                kind = "heuristic"
                basis = ("the structural precision ordering ONLY — the "
                         "measured row 'mla-tq-vs-snapmla-accuracy' is "
                         "deleted, so the fp8-rows-vs-4-bit-codec order is "
                         "asserted from per-element precision, unpriced")
                meas = ""
        else:
            fp8, tq = sizing.V4_FP8_ENTRY_BYTES, sizing.V4_TQ_ENTRY_BYTES
            price = (f"CSA/HCA tier entries {fp8} B (kFp8) vs {tq} B (kTq4) "
                     "per tier flipped (sizing.py:315-316, "
                     "vram_allocator.h:37-83)")
            kind = "heuristic"
            basis = ("the STRUCTURAL order only — each ladder step "
                     "quantizes strictly more V4 tiers to the 4-bit codec "
                     "(kFp8/kFp8 -> kTq4/kFp8 -> kTq4/kTq4, kv_codec.h), "
                     "so precision is monotone by nesting; the MAGNITUDE is "
                     "unmeasured on V4 (goldens 6/6 per arm are pass/fail, "
                     "not a gradient) — a V4 NLL A/B would price it")
            meas = "TECH_DEBT: DS4 FP8/TQ/mix goldens 6/6, 6/6, 2/2"
        refs += ["templates.ACCURACY_LADDER", "config_validator.cpp:510-534"]
        direction = ("floors the numerics ladder at its most accurate "
                     "serveable end" if tier == "high" else
                     "floors the numerics ladder at its most "
                     "KV-byte-efficient serveable end")
        self.ex.add(
            "compute.attention_backend", backend, kind,
            f"accuracy={tier} {direction}: `{backend}` instead of the "
            f"standard tier's `{std}` (ladder {'/'.join(ladder)}). "
            f"Basis: {basis}. Price: {price} — capacity consequences flow "
            "through the ordinary E1 fit and show up as degradations when "
            "the ask no longer fits",
            refs=tuple(refs), measurement=meas)
        self.ex.add(
            "autoconfig.accuracy", tier, kind,
            f"accuracy={tier} CHANGED the derived backend: `{std}` -> "
            f"`{backend}`. The tier is a FLOOR over the family's serveable "
            "ladder, never a weight — tok/s and perplexity have no honest "
            "exchange rate (the TD-AUTOCONFIG-SPEED-BUDGET argument), so "
            "the lever orders options and the fit pays whatever capacity "
            f"the floor costs. {price}",
            refs=tuple(refs), measurement=meas)

    # ------------------------------------------------------------- assemble

    def _assemble(self, ordered, tp, kv_mode, idx_mode, tiering_on,
                  conc, max_seq, draft_on, draft, draft_ranks, plans,
                  lc: "LongCtxPlan | None" = None) -> dict:
        s, k = self.shape, self.k
        fam = templates.family_of(s.architecture)
        ftmpl = templates.FAMILY[fam]
        T = templates.COMMON
        backend = self.attention_backend()
        base = self.base

        plan_by_ord = {p.ordinal: p for p in plans}
        gpus_out = []
        base_margin_bytes = int(k.vram_safety_margin_gb * GIB)
        for i, g in enumerate(ordered):
            p = plan_by_ord[g.ordinal]
            vram_gb = int(g.vram_mib * k.vram_usable_fraction) >> 10  # MiB->GiB
            entry = {
                "id": g.ordinal,
                "type": g.gpu_type,
                "vram_gb": vram_gb,
                "pcie_gen": g.pcie_gen_max,
                "pcie_width": g.pcie_width_max,
                "roles": (["attention", "resident", "expert_streaming"]
                          if i < tp else ["expert_streaming"]),
                "vram_allocation_gb": {
                    "expert_streaming": _round_gib(p.expert_bytes, k.expert_gib_quantum),
                    "stable_zone_fraction": ftmpl["stable_zone_fraction_per_gpu"],
                },
            }
            if i < tp and p.margin_bytes > base_margin_bytes:
                mg = round(p.margin_bytes / GIB, 2)
                entry["vram_allocation_gb"]["safety_margin_gb"] = mg
                sc = lc.scratch if lc is not None else None
                terms = ("; ".join(f"{n} {b // (1 << 20)} MiB"
                                   for n, b, _ph, _st in sc.terms)
                         if sc is not None else "")
                self.ex.add(
                    f"hardware.gpus[{i}].vram_allocation_gb.safety_margin_gb",
                    mg, "closed_form",
                    f"TP-rank margin RAISED from the global "
                    f"{k.vram_safety_margin_gb} GiB to fund the "
                    f"max_seq/stride-scaled runtime allocations that live "
                    f"OUTSIDE the VramAllocator block (block semantics: "
                    f"in-block slack is invisible to runtime cudaMalloc — "
                    f"only the margin + physical slack feed it): {terms}. "
                    f"Plus the MoE-big transient spill "
                    f"{lc.spill_mib if lc else 0} MiB at the derived "
                    f"stride and the fit-check headroom",
                    refs=("TD-AUTOCONFIG-LONGCTX-RUNTIME-SCRATCH",
                          "command_dispatcher.cpp:1359",
                          "engine.cpp:919", "AUTOCONFIG §5 E1"))
            gpus_out.append(entry)
            self.ex.add(f"hardware.gpus[{i}].vram_gb", vram_gb, "heuristic",
                        f"physical {g.vram_mib} MiB (BAR1) x {k.vram_usable_fraction} "
                        "usable fraction (1/16 driver+runtime reserve)",
                        measurement=f"BAR1 span, {g.pci_bus_id}")
            self.ex.add(f"hardware.gpus[{i}].vram_allocation_gb.expert_streaming",
                        entry["vram_allocation_gb"]["expert_streaming"],
                        "searched",
                        f"lever vram_expert_ratio: {p.expert_slots} slots x "
                        f"{self.slot_bytes} B ({self.slot_prov}); "
                        + ("TP GPU: overflow share after non-TP hosts filled"
                           if i < tp else
                           f"expert host: capacity after margin "
                           f"{k.vram_safety_margin_gb} + overhead {k.non_tp_overhead_gib} GiB"),
                        refs=("AUTOCONFIG §2.1/§5 E1",))

        # Array-shape row (TD-AUTOCONFIG-COMPARE-CLASSES): the gpus array's
        # SHAPE is a derivation — one entry per detected GPU — so a
        # reference recipe written for a different GPU count must not read
        # as an unexplained whole-array divergence.
        self.ex.add(
            "hardware.gpus", f"{len(ordered)} entries", "closed_form",
            f"one entry per detected GPU ({len(ordered)} on this box, "
            "hwdetect BAR1 scan), placement-ordered; the first "
            f"{tp} (tp_array) carry attention/resident roles, the rest are "
            "expert_streaming hosts",
            refs=("AUTOCONFIG §5 E1", "hwdetect.detect_hardware"))

        tp_nodes = {ordered[i].numa_node for i in range(tp)}
        spill_nodes = sorted(n.node for n in self.hw.numa_nodes
                             if n.node not in tp_nodes)
        per_node = [{"node": n.node, "mode": "fraction_free",
                     "value": k.hbm_spill_fraction_free}
                    for n in self.hw.numa_nodes if n.is_hbm_bank
                    and n.node in spill_nodes]
        self.ex.add("memory.cross_node_spill.nodes", spill_nodes, "heuristic",
                    f"all memory nodes except those local to a TP GPU {sorted(tp_nodes)} "
                    "(TP-local free memory serves KV-tiering host pools and "
                    "attention D2H staging — D2H must stay NUMA-local)",
                    refs=("AUTOCONFIG §5 H1", "feedback: D2H NUMA-aware"))
        self.ex.add("memory.cross_node_spill.per_node(hbm)", k.hbm_spill_fraction_free,
                    "measured",
                    "CPU-less HBM banks pin at fraction_free 0.6 — 0.8 measured "
                    "too tight (node-local OOM with 30 GB free system-wide)",
                    refs=("hbm-fraction-free-06",))

        bpp = None
        max_pages: object = "auto"
        if s.is_v4:
            max_pages = "auto"
            self.ex.add("memory.kv_cache.max_pages_per_gpu", "auto", "template",
                        "deepseek_v4 requires auto (proportional tier scale-down)",
                        refs=("v4-max-pages-auto-only",))
        elif tiering_on:
            bpp = sizing.kv_bytes_per_page(s, self.kv_quant, backend, k.page_size_tokens)
            hot = s.index_topk
            pages = sizing.align_up(
                conc * math.ceil(hot / k.page_size_tokens) * s.engine_kv_pool_layers(),
                k.device_pool_round_pages)
            max_pages = pages
            self.ex.add("memory.kv_cache.max_pages_per_gpu", pages, "heuristic",
                        f"tiered KV device pool = {conc} seq x ceil(hot {hot}/"
                        f"page {k.page_size_tokens}) x {s.engine_kv_pool_layers()} "
                        f"pool layers, rounded up to {k.device_pool_round_pages} "
                        f"({pages * bpp / (1 << 20):.0f} MiB at {bpp} B/page); the "
                        "host cold pool holds the rest (INV-KVT-16 windowed admission)",
                        refs=("kv_tiering_manager.cpp:120-137", "INV-KVT-16"))

        chunk = k.dcp_chunk_size
        growth = chunk if (tp >= 2 and kv_mode == "sharded") else 1024
        if growth != 1024:
            self.ex.add("memory.kv_cache.page_growth_chunk_tokens", growth,
                        "closed_form",
                        "sharded DCP KV: growth must follow dcp_chunk_size "
                        "granularity (round-robin-by-chunk rank ownership)",
                        refs=("kv_shard_math.h", "tiering-chunk-page-divisibility"))

        dense_layers = list(range(s.first_k_dense_replace))
        self.ex.add("memory.pinned_layers.dense_ffn_layers", dense_layers,
                    "closed_form", f"first_k_dense_replace={s.first_k_dense_replace}")

        prefix_tokens = sizing.align_up(conc * max_seq, 65536)
        speculation: dict = dict(T["speculation_scaffold"])
        if draft_on and draft is not None:
            speculation["method"] = "dspark"
            speculation["dspark"] = {
                "checkpoint_path": draft.checkpoint_path,
                "confidence_enabled": True,
                "draft_gpus": draft_ranks,
                "draft_weights_quant": "bf16" if draft.is_gguf or tp < 2 else "nvfp4",
                "block_size": draft.block_size,
                "speculative_tokens": draft.speculative_tokens,
            }
            self.ex.add("speculation.method", "dspark", "searched",
                        "draft checkpoint present and the E1 fit holds with its "
                        f"VRAM charge; block_size {draft.block_size} and "
                        f"speculative_tokens {draft.speculative_tokens} read from "
                        "the checkpoint's own config.json; note the draft stops "
                        f"helping above its {draft.ctx_cap_tokens}-token context cap",
                        refs=("dspark-ctx-cap", "TD-DSPARK-CTX-POLICY"))
        else:
            speculation["method"] = "none"
            why = ("no dspark checkpoint discoverable for this model"
                   if draft is None else
                   "E1 degradation dropped the draft — its VRAM is worth more "
                   "as experts/KV on this box (the 63-vs-64-slot lesson)")
            self.ex.add("speculation.method", "none", "searched", why,
                        refs=("AUTOCONFIG §5 P6",))
            # TD-AUTOCONFIG-COMPARE-CLASSES: the dspark section is omitted
            # WITH the method decision — a reference that carries one must
            # not read as an unexplained solver omission.
            self.ex.add("speculation.dspark", None, "searched",
                        f"omitted with the method decision — {why}",
                        refs=("AUTOCONFIG §5 P6",))
        # TD-AUTOCONFIG-NO-SERVING-SURFACE (baked constants): the scaffold's
        # template `enabled: true` beside method='none' was a constant
        # masquerading as a decision — the switch follows the P6 method.
        speculation["enabled"] = speculation["method"] != "none"
        self.ex.add("speculation.enabled", speculation["enabled"],
                    "closed_form",
                    "follows the P6 method decision — the speculation "
                    "scaffold arms exactly when a sized method is on "
                    "(the GF3 hand recipe's shape: enabled=false when no "
                    "method serves; enabled:true beside method:'none' was "
                    "a template constant masquerading as a decision)",
                    refs=("AUTOCONFIG §5 P6",
                          "TD-AUTOCONFIG-NO-SERVING-SURFACE"))

        mem = {
            "pinned_layers": {"attention": "all", "dense_ffn_layers": dense_layers,
                              "embedding": True, "output_head": True, "gating": "all"},
            "tp_mode_per_layer": {"default_mode": None, "gating": None,
                                  "pinned_dense_ffn": None, "attention": None,
                                  "shared_expert": None, "embedding": None,
                                  "output_head": None},
            "kv_cache": {
                "max_pages_per_gpu": max_pages,
                "page_size_tokens": k.page_size_tokens,
                "speculation_pool_fraction": self.SPECULATION_POOL_FRACTION,
                "indexer_k_page_size_tokens": k.indexer_k_page_size_tokens,
                "prefill_scratch_preallocated_gb": k.prefill_scratch_gb,
                "streaming_spill_fraction": 0.34,
                "naive_prefill_context_ceiling": None,
                "dcp_chunk_size": chunk,
                "page_growth_chunk_tokens": growth,
            },
            "expert_cache": dict(T["expert_cache"]),
            "nvme_tier": {"enabled": False, "io_engine": "io_uring",
                          "queue_depth": 64, "prefetch_ahead_layers": 3,
                          "host_ram_cache_gb": None, "direct_io": True},
            "numa": dict(T["numa"]),
            "preload_expert_buffers": True,
            "pin_host_expert_pool": True,
            "pin_host_expert_pool_direct_load": True,
            "pin_host_expert_pool_preload": True,
            "pin_host_expert_pool_direct_o_direct": True,
            "pin_host_expert_pool_sizing": {"mode": "fraction_total",
                                            "value": k.host_pin_fraction_total},
            "cross_node_spill": {
                "enabled": True,
                "nodes": [{"node": n, "weight": 1} for n in spill_nodes],
                "sizing_mode": "fraction_total",
                "sizing_value": k.host_pin_fraction_total,
                "per_node": per_node,
            },
            "kv_tiering": {
                "enabled": bool(tiering_on),
                "hot_buffer_slots": s.index_topk if tiering_on else 0,
                "host_to_device_ratio": k.kv_tiering_host_to_device_ratio,
                "tiered_prefill": bool(tiering_on),
            },
            "vram_safety_margin_gb": k.vram_safety_margin_gb,
            "arena_attach": {"enabled": True, "persist": True},
        }
        if (base.get("memory") or {}).get("arena_placement", {}).get("freq_table"):
            mem["arena_placement"] = {"freq_table": base["memory"]["arena_placement"]["freq_table"]}
            self.ex.add("memory.arena_placement", mem["arena_placement"], "carried",
                        "measured placement table carried into the derivation "
                        "(auto-run step 4 supplied/reused it, or a base recipe "
                        "carried it): a per-(layer,expert) demand-fetch fit, "
                        "trace-fit per model AND box — tools/loader_xray/"
                        "freq_table.py over an LS_PERF_TRACE dump. Its content "
                        "hash folds into the ArenaCache store identity, so "
                        "CHANGING it costs one cold store rebuild",
                        refs=("memory.arena_placement.freq_table",
                              "arena_placement.h", "spec/AUTO_RUN.md step 4"))
        else:
            self.ex.add("memory.arena_placement", None, "measured",
                        "no static freq table: it is a MEASURED artifact "
                        "(trace-fit per model+box) that no derivation can "
                        "invent — the auto-run FITS one in step 4 from the "
                        "training decode's own trace, so an absent table "
                        "means step 4 did not run (no trace, or "
                        "--skip-placement). The M3b online migrator is "
                        "schema-default ON and converges the layout from "
                        "live fetch traffic meanwhile; --refit-placement "
                        "captures a trace and fits one",
                        refs=("memory.arena_placement.online default true",
                              "spec/SPEC_UPDATES.md:516"),
                        measurement="online placement 10.459 tok/s vs "
                                    "10.23-10.42 static anchors (dsp52, "
                                    "2026-08-18 champion decision)")
        if tiering_on:
            kt = sizing.kv_tiering_sizes(s, self.kv_quant, backend,
                                         k.page_size_tokens, s.index_topk,
                                         k.kv_tiering_host_to_device_ratio,
                                         s.engine_kv_pool_layers(), tp,
                                         kv_sharded=(kv_mode == "sharded" and tp >= 2))
            self.ex.add("memory.kv_tiering.hot_buffer_slots", s.index_topk,
                        "measured",
                        f"hot window = index_topk (champion-measured; engine "
                        f"auto would be 2x); cold pool {kt.cold_pool_pages} "
                        f"pages/rank = {kt.host_pinned_bytes_per_rank / (1 << 20):.0f} "
                        "MiB pinned host per rank after replica dedup",
                        refs=("kv_tiering_manager.cpp:120-137",))

        nvme_note = ""
        if self.hw.nvmes:
            nv = self.hw.nvmes[0]
            nvme_note = (f"nvme gen{nv.pcie_gen} x{nv.pcie_width} ceiling "
                         f"~{nv.bandwidth_ceiling_gbps:.1f} GB/s")
        self.ex.add("memory.nvme_tier.enabled", False, "closed_form",
                    "host arena holds every expert slot in RAM — no NVMe tier "
                    f"needed ({nvme_note}; cold boot pays the disk ceiling once, "
                    "the arena holder makes warm boots ~25-75 s)",
                    refs=("disk-ceiling: hardware Gen3 x4",),
                    measurement=nvme_note)

        tok = (base.get("serving") or {}).get("tokenizer_path", "auto")
        if tok and tok != "auto":
            self.ex.add("serving.tokenizer_path", tok, "closed_form",
                        "a resolved tokenizer directory, not 'auto': serve's "
                        "auto-resolution only searches the WEIGHTS dir and "
                        "GGUF-embedded tokenizers are not extracted, so a "
                        "GGUF box needs the path in the recipe or every boot "
                        "needs --tokenizer-path",
                        refs=("python/cli/serve.py resolve_tokenizer_dir",
                              "TD-SERVE-GGUF-TOKENIZER"))
        else:
            self.ex.add("serving.tokenizer_path", "auto", "closed_form",
                        "no tokenizer directory found beside the weights — "
                        "serve resolves it at boot, or fails loudly asking "
                        "for --tokenizer-path",
                        refs=("python/cli/serve.py resolve_tokenizer_dir",))

        # TD-AUTOCONFIG-NO-SERVING-SURFACE (baked constants): every emitted
        # field is derived-and-explained, carried-and-labelled, or absent.
        # max_batch_size is a family serving-template value, not a universal
        # constant (the GF3 recipe serves glm5_next at 512; the champion
        # serves the MLA+DSA giants at 64) — so it carries its own row.
        orch = dict(T["orchestrator"])
        tmpl_b = int(ftmpl.get("max_batch_size", orch["max_batch_size"]))
        derived_b = lc.max_batch if lc is not None else tmpl_b
        orch["max_batch_size"] = derived_b
        if derived_b != tmpl_b:
            bt = sizing.block_tables_bytes(s, max_seq, tmpl_b,
                                           k.page_size_tokens)
            bt_d = sizing.block_tables_bytes(s, max_seq, derived_b,
                                             k.page_size_tokens)
            self.ex.add("orchestrator.max_batch_size", derived_b,
                        "closed_form",
                        f"REDUCED from the family template's {tmpl_b}: "
                        f"dev_block_tables are (layers+nextn) x B x "
                        f"ceil(max_seq/page) x 4 B per TP rank "
                        f"(command_dispatcher.cpp:1359) — {bt / GIB:.2f} "
                        f"GiB at B={tmpl_b} and max_sequence_length "
                        f"{max_seq}, vs the {k.block_table_budget_gb} GiB "
                        f"budget; largest halving that fits is B="
                        f"{derived_b} ({bt_d / GIB:.2f} GiB)",
                        refs=("TD-AUTOCONFIG-LONGCTX-RUNTIME-SCRATCH",
                              "command_dispatcher.cpp:1359",
                              "recipes/glm53flash_serve.json"))
        elif "max_batch_size" in ftmpl:
            self.ex.add("orchestrator.max_batch_size", orch["max_batch_size"],
                        "template",
                        "family serving template: the GF3 live-serving "
                        "recipe's schedule width for glm5_next — a copied "
                        "serving-policy knob (§5 M1), not a derivation; "
                        "its block tables fit the budget at this "
                        "max_sequence_length, so no reduction applies",
                        refs=("recipes/glm53flash_serve.json",
                              "TD-AUTOCONFIG-NO-SERVING-SURFACE"))
        else:
            self.ex.add("orchestrator.max_batch_size", orch["max_batch_size"],
                        "template",
                        "family serving template: the champion recipes' "
                        "schedule width (also the schema default) — a copied "
                        "serving-policy knob (§5 M1), not a derivation",
                        refs=("recipes/glm52_serve_champion.json",
                              "TD-AUTOCONFIG-NO-SERVING-SURFACE"))

        recipe = {
            "model": dict(base.get("model") or s.raw),
            "quantization": self.quant or {
                "weights": "gguf", "gguf_strategy": "int",
                "attention_compute": "fp8_e4m3", "kv_cache": "fp8_e4m3",
                "gating_compute": "fp16"},
            "hardware": {
                "gpus": gpus_out,
                "tp_array": list(range(tp)),
                "system_ram_gb": int(self.hw.mem_total_gib),
                "nvme_paths": [],
                "nvme_capacity_gb": None,
            },
            "memory": mem,
            "orchestrator": orch,
            "prefetch": dict(T["prefetch"]),
            "speculation": speculation,
            "parallelism": {"tensor_parallelism": tp, **T["parallelism"]},
            "transfer": dict(T["transfer"]),
            "compute": {
                "cuda_graphs": dict(T["compute_cuda_graphs"]),
                "gemm": dict(T["compute_gemm"]),
                "attention_backend": backend,
                "dsa_sparse_prefill": bool(ftmpl["dsa_sparse_prefill"] and s.has_dsa),
            },
            "serving": {
                "host": (base.get("serving") or {}).get("host", "0.0.0.0"),
                "port": (base.get("serving") or {}).get("port", 8000),
                "max_concurrent_requests": conc,
                "max_sequence_length": max_seq,
                "tokenizer_path": (base.get("serving") or {}).get("tokenizer_path", "auto"),
                "prefix_cache": {"enabled": True,
                                 "max_cached_tokens": prefix_tokens,
                                 "max_entries": k.prefix_cache_max_entries},
            },
            "_internal-prefix_cache": {"max_entry_tokens": max_seq},
        }
        # Inert-field hygiene (TD-AUTOCONFIG-NO-SERVING-SURFACE, baked
        # constants): below dcp=2 the engine ignores both DCP fields
        # (enginecheck reads them under a tp>=2 guard), so a tp=1 recipe
        # carrying dcp_kv_mode='sharded' claims a shard that does not
        # exist. Emitted iff meaningful (dcp>=2) or explicitly pinned —
        # the pin path warns INERT and emits what it was told.
        if tp >= 2 or self._pin("hardware.dcp_indexer_mode") is not None:
            recipe["hardware"]["dcp_indexer_mode"] = idx_mode
        else:
            # TD-AUTOCONFIG-COMPARE-CLASSES: the omission is itself a
            # decision — a reference recipe carrying the field must not
            # read as an unexplained solver omission.
            self.ex.add("hardware.dcp_indexer_mode", None, "closed_form",
                        "omitted below dcp=2 — the engine ignores both DCP "
                        "fields at tp=1 (enginecheck reads them under a "
                        "tp>=2 guard), so an emitted mode would claim a "
                        "shard that does not exist",
                        refs=("TD-AUTOCONFIG-NO-SERVING-SURFACE",))
        if tp >= 2 or self._pin("hardware.dcp_kv_mode") is not None:
            recipe["hardware"]["dcp_kv_mode"] = kv_mode
        else:
            self.ex.add("hardware.dcp_kv_mode", None, "closed_form",
                        "omitted below dcp=2 — the engine ignores both DCP "
                        "fields at tp=1 (enginecheck reads them under a "
                        "tp>=2 guard), so an emitted mode would claim a "
                        "shard that does not exist",
                        refs=("TD-AUTOCONFIG-NO-SERVING-SURFACE",))
        # Serving surface (TD-AUTOCONFIG-NO-SERVING-SURFACE): deployment
        # wiring the solver cannot derive — carried from the base recipe
        # when it says something, else the family template, else absent.
        surface = ftmpl.get("serving_surface") or {}
        base_serving = base.get("serving") or {}
        for key in ("tokenizer_mode", "tool_call_parser",
                    "reasoning_parser", "enable_auto_tool_choice"):
            if base_serving.get(key) is not None:
                recipe["serving"][key] = base_serving[key]
                self.ex.add(f"serving.{key}", base_serving[key], "carried",
                            "serving surface carried from the base recipe — "
                            "deployment wiring (parsers, tokenizer mode) the "
                            "solver must never silently drop",
                            refs=("TD-AUTOCONFIG-NO-SERVING-SURFACE",))
            elif key in surface:
                recipe["serving"][key] = surface[key]
                self.ex.add(f"serving.{key}", surface[key], "template",
                            "family serving surface (GF3.13): reasoning "
                            "split, tool-call wire format and tokenizer mode "
                            "(incl. reasoning_effort normalization + the "
                            "stop-token autodetect fix) — absent, an "
                            "autoconfigured boot silently loses them",
                            refs=("TD-AUTOCONFIG-NO-SERVING-SURFACE",
                                  "recipes/glm53flash_serve.json",
                                  "spec/measurements/glm53_flash.md (GF3.13)"))
        if "prefill_superchunk_tokens" in ftmpl and lc is not None \
                and lc.superchunk > 0:
            sc = lc.superchunk
            recipe["compute"]["prefill_superchunk_tokens"] = sc
            self.ex.add("compute.prefill_superchunk_tokens", sc,
                        "searched",
                        "DERIVED max-stride-that-fits (P-31, "
                        "TD-AUTOCONFIG-STRIDE-NOT-DERIVED): the largest "
                        "superchunk stride S whose MoE-big transient "
                        "(command_dispatcher.cpp:409-437, ~0.48 MiB/token "
                        "on this shape) + the per-class fit headroom fits "
                        "every expert device's modeled free-at-fit "
                        "(P-30 step 2 device-class arithmetic; the "
                        "expert-only class binds first — no scratch "
                        "tail). Measured basis: fresh prefill "
                        "+54-62% at S=2048 vs 512, decaying toward the "
                        "stride-invariant attention asymptote (P-30 "
                        "steps 1-4); decode is unaffected (dispatches "
                        "<=512 stay single-shot). SPEED-axis spend, "
                        "never the accuracy ladder; the engine's elastic "
                        "fit steps an over-ask down, never OOMs "
                        "(INV-MOE-BIG-ROWS)",
                        refs=("TD-AUTOCONFIG-STRIDE-NOT-DERIVED",
                              "TD-MOE-EP-XTP-WAVES",
                              "command_dispatcher.cpp:571-575",
                              "spec/measurements/glm53_flash.md P-30"),
                        measurement="P-30 steps 1-4: 24k fresh 118.5->189 "
                                    "(S 512->2048), 217.4 @4096 off-carve")
            if len(ordered) > tp:
                # EP-beyond-TP: the stride collapses onto
                # min(max(moe_big_chunk_tokens, max_batch), engine chunk
                # capacity) (orchestrator.py:1963-1977) — the diagonal
                # S = N must move together or the emitted stride is inert
                recipe["compute"]["moe_big_chunk_tokens"] = sc
                self.ex.add("compute.moe_big_chunk_tokens", sc,
                            "closed_form",
                            "= prefill_superchunk_tokens: on EP-beyond-TP "
                            "the orchestrator clamps the served stride to "
                            "min(max(moe_big_chunk_tokens, "
                            "max_batch_size), EngineInfo."
                            "moe_chunk_capacity) — the two knobs are ONE "
                            "knob on this topology (TD-MOE-EP-XTP-WAVES)",
                            refs=("TD-MOE-EP-XTP-WAVES",
                                  "python/orchestrator/orchestrator.py:"
                                  "1963-1977"))
            if lc.headroom_attn_mib and lc.headroom_attn_mib != 1024:
                recipe["compute"]["moe_big_fit_headroom_mb"] = \
                    lc.headroom_attn_mib
                post = lc.scratch.postcheck_mib if lc.scratch else 0
                self.ex.add(
                    "compute.moe_big_fit_headroom_mb",
                    lc.headroom_attn_mib, "closed_form",
                    f"attention-host fit headroom raised from the 1024 "
                    f"MiB default: the fit check runs BEFORE the "
                    f"post-check allocations (block tables, logits, KVT "
                    f"union staging, MoE persistent set = {post} MiB at "
                    f"this ask) and the headroom is what reserves room "
                    f"for them — the default is calibrated to ~578 MiB "
                    f"at the champion regime and under-reserves at long "
                    f"context (the boot would die at "
                    f"command_dispatcher.cpp:1363)",
                    refs=("TD-AUTOCONFIG-LONGCTX-RUNTIME-SCRATCH",
                          "command_dispatcher.cpp:497-521",
                          "config/schema.json compute.moe_big_fit_"
                          "headroom_mb"))
            if lc.headroom_exp_mib and lc.headroom_exp_mib != 256:
                recipe["compute"]["moe_big_fit_headroom_expert_only_mb"] = \
                    lc.headroom_exp_mib
                self.ex.add(
                    "compute.moe_big_fit_headroom_expert_only_mb",
                    lc.headroom_exp_mib, "closed_form",
                    "expert-only fit headroom raised over the 256 MiB "
                    "default to cover the post-check MoE persistent set "
                    "at the derived stride",
                    refs=("TD-AUTOCONFIG-LONGCTX-RUNTIME-SCRATCH",
                          "command_dispatcher.cpp:497-521"))
        elif "prefill_superchunk_tokens" in ftmpl:
            recipe["compute"]["prefill_superchunk_tokens"] = \
                ftmpl["prefill_superchunk_tokens"]
            self.ex.add("compute.prefill_superchunk_tokens",
                        ftmpl["prefill_superchunk_tokens"], "template",
                        "superchunk prefill REQUEST cap from the family "
                        "serving template: GF3.15 measured superchunk FASTER "
                        "than chunked on glm5_next (141-token arm: 35.9 vs "
                        "38.1 s mapped, 36.2 vs 41.3 s carve); the engine "
                        "derives the effective batch capacity K elastically "
                        "at init (prefill_moe_big), this caps the request",
                        refs=("recipes/glm53flash_serve.json",
                              "TD-PREFILL-SUPERCHUNK",
                              "TD-AUTOCONFIG-NO-SERVING-SURFACE"),
                        measurement="spec/measurements/glm53_flash.md (GF3.15)")
        self.ex.add("hardware.tp_array", list(range(tp)), "closed_form",
                    "positions of the first tensor_parallelism entries of "
                    "hardware.gpus (the engine's own TP-group rule; note the "
                    "id-vs-position validator ambiguity — both shipped recipes "
                    "use positions)",
                    refs=("engine.cpp:585-592",))
        self.ex.add("serving.prefix_cache.max_cached_tokens", prefix_tokens,
                    "heuristic", "active-context lever rounded up to a 64k "
                    "prefix budget granule")
        self.ex.add("_internal-prefix_cache.max_entry_tokens", max_seq,
                    "closed_form", "one full request is the largest reusable prefix")
        tqrow = ec.get("glm5next-tq-backend-unwired")
        if self._pin("compute.attention_backend") is not None:
            # pinned: the pin row is the whole story — in particular the
            # accuracy tier changed nothing (the axis is removed, design §5)
            # so no autoconfig.accuracy row may claim it did
            pass
        elif backend != self._backend_for_tier("standard"):
            self._explain_accuracy_tier(backend)
        elif backend != ftmpl["attention_backend"] and tqrow is not None:
            self.ex.add("compute.attention_backend", backend, "template",
                        "TEMPORARY substitution — the family template derives "
                        f"{ftmpl['attention_backend']} on KV bytes, but a "
                        "re-introduced engine-gate row says the engine "
                        "cannot serve it for this architecture today "
                        f"(registry row '{tqrow.id}'). The gate narrows the "
                        "accuracy lever's menu (§2.4) — every tier lands "
                        "here while the row exists. Delete the row and the "
                        "derivation moves back to the template.",
                        refs=(tqrow.id, tqrow.ticket, tqrow.site))
        elif backend == "snapmla" and s.is_glm5_next:
            tq_b = sizing.kv_bytes_per_token(s, self.kv_quant, "turboquant_mla")
            sn_b = sizing.kv_bytes_per_token(s, self.kv_quant, "snapmla")
            self.ex.add("compute.attention_backend", backend, "measured",
                        "family template: THE CHAMPION SWITCHED to snapmla "
                        "(P-29 step 14, pre-committed OQ-6/OQ-7 rule) — "
                        "snapmla+FP8-decode wins BOTH axes on the fixed "
                        "binary (8k repeat medians 24.10/24.21/24.04 vs TQ "
                        "23.67/23.80/23.74 tok/s; TF-NLL 1.7897 vs 1.8016, "
                        "-0.0118 +- 0.0060 nats). KV price: "
                        f"{sn_b} B/row vs TQ {tq_b} B ({sn_b / tq_b:.2f}x) "
                        "— `accuracy=compact` floors at TQ when KV bytes "
                        "matter more",
                        refs=("mla-tq-vs-snapmla-accuracy",
                              "recipes/glm53flash_serve.json",
                              "spec/measurements/glm53_flash.md "
                              "(P-29 step 14)"))
        elif backend == "turboquant_mla":
            # Byte argument stated with THIS geometry's numbers (glm5_next
            # NoPE: 258 vs 516 B, 2.0x; V3.2/GLM-5.2 rope: 386 vs 644 B,
            # 1.67x) — computed, never quoted from another model's row.
            tq_b = sizing.kv_bytes_per_token(s, self.kv_quant, "turboquant_mla")
            sn_b = sizing.kv_bytes_per_token(s, self.kv_quant, "snapmla")
            self.ex.add("compute.attention_backend", backend, "measured",
                        "family template: direct-TQ sparse is the only B=1 "
                        f"sparse decode path for MLA+DSA; TQ KV row {tq_b} B "
                        f"vs snapmla {sn_b} B for this geometry",
                        refs=("TQ campaign §12l-§12m",
                              "TD-GLM5-TQ-BACKEND-UNWIRED (glm5_next tp=1 "
                              "live verification 2026-09-01)"))
        else:
            self.ex.add("compute.attention_backend", backend, "measured",
                        "family template (v4: TQ-for-V4 codec composition, "
                        "goldens 6/6)",
                        refs=("TQ campaign §12l-§12m",))
        if base.get("preprocessing"):
            recipe["preprocessing"] = dict(base["preprocessing"])
        gl = dict(base.get("gpu_loader") or {})
        gl.setdefault("enabled", True)
        if self.cal is not None and not self.cal_reject:
            gl["calibration_mode"] = "loaded"
            self.ex.add("gpu_loader.calibration_mode", "loaded", "measured",
                        "an accepted calibration exists for this box+model, "
                        "so the engine loads it instead of re-measuring "
                        "(schema default; the champion omits the field). It "
                        "still self-heals: a missing file falls back to a "
                        "full calibration and writes it",
                        refs=("engine.cpp:519-568", "INV-LOADER-CAL-6"),
                        measurement=os.path.basename(self.cal.path))
        if self.cal is None or self.cal_reject:
            gl["calibration_mode"] = "full"
            self.ex.add("gpu_loader.calibration_mode", "full", "heuristic",
                        "no accepted calibration for this box/model — first "
                        "boot runs the engine's load_or_calibrate_with full "
                        "measurement and persists it; re-run autoconfig "
                        "after. The calibrating boot must be COLD: against a "
                        "warm arena-holder store the calibration footprint "
                        "node-OOMs (kill the holder first)",
                        refs=("engine.cpp:519-568", "INV-LOADER-CAL-6",
                              "full-calibration-needs-cold-boot"))
        recipe["gpu_loader"] = gl
        return recipe
