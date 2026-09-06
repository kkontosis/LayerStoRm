"""Determinism superflags — Python-side mirror of src/core/determinism.cpp.

TWO flags, one strict hierarchy:

* **Flag 1 — run-to-run determinism** (``compute.deterministic`` /
  env ``LS_DETERMINISTIC``): same binary+config+input => byte-identical
  output on every run.  Forces the enforcers ON (DET-REDUCE,
  DET-EP-COMBINE) and the genuinely nondeterministic knobs OFF
  (SpecRoundGovernor).  Trajectory-forking-but-STABLE optimizations
  survive it — ``LS_TQ_SPLITKV`` is the worked example (P-29 step 9: two full
  runs byte-identical, yet 327/6,999 TF argmax flips vs the reference).

* **Flag 2 — reference-trajectory identity**
  (``compute.reference_trajectory_identity`` /
  env ``LS_REFERENCE_TRAJECTORY_IDENTITY``): the STRONGER property — the
  output matches the canonical trajectory for this configuration, the one
  goldens/identity shas/A-B baselines pin.  Forces every trajectory-forking
  knob to its canonical state (split-KV included) on top of flag 1.

The hierarchy is strict and implemented as one: flag 2 IMPLIES flag 1
(matching a deterministic reference exactly means being deterministic),
never the reverse.  Requesting flag 2 while explicitly suppressing flag 1
is refused loudly.

WHAT "REFERENCE" MEANS: relative, not a global golden — what THIS
configuration produces with no flag-level shortcut forking it (the
canonical numerics path for the chosen backend, accuracy tier and
parallelism).  Not a fixed sha (the reference legitimately moves with a
deliberate re-golden), not "one true backend" (TQ vs snapMLA is a config
choice).  The guarantee is "no FLAG moved you off your config's canonical
trajectory".

Why a Python apply exists at all: CPython's ``os.environ`` is a snapshot —
C++ ``setenv`` at engine init is invisible to Python readers (governor,
subgrid, superchunk knobs).  serve.py therefore applies this mirror BEFORE
booting the engine; writing through ``os.environ`` putenvs down to C++, so
the engine-side apply becomes a conforming no-op.  The C++ apply remains
the backstop for entrypoints without Python (goldens, C++ tests).

Lockstep contract: the ACTIONABLE rows below must equal the same-class rows
of the C++ registry — ``tests/unit/test_determinism_superflag.py`` parses
src/core/determinism.cpp and asserts equality (env, axis, class, forced,
conflict kind, exact), and its tree scan makes an unregistered
LS_*/LAYERSTORM_* knob a test failure.

Design note: scratchpad/gf3_speed_saga/DETERMINISM_FLAG.md.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

log = logging.getLogger("layerstorm.determinism")

# Axes (which flag a row acts under) — spellings match the C++ enum.
RUN_TO_RUN = "kRunToRun"   # acts under flag 1 (and flag 2 via the hierarchy)
REFERENCE = "kReference"   # acts under flag 2 only

# Classes.
FORCE_ON = "kForceOn"
FORCE_OFF = "kForceOff"
PIN_DEFAULT = "kPinDefault"
KEEP_ON = "kKeepOn"
NEUTRAL = "kNeutral"

# Conflict kinds.
IF_TRUTHY = "kIfTruthy"   # set, non-empty, first char != '0'
IF_FALSY = "kIfFalsy"     # set, first char == '0'
IF_SET = "kIfSet"         # set non-empty at all
IF_EXACT = "kIfExact"     # value == exact
NO_CONFLICT = "kNone"

# Modes.
MODE_OFF = "off"
MODE_RUN_TO_RUN = "run_to_run"
MODE_REFERENCE = "reference"   # strictly includes run_to_run


@dataclass(frozen=True)
class Knob:
    env: str
    axis: str
    cls: str
    forced: Optional[str]          # value forced for FORCE_ON/FORCE_OFF
    conflict: str
    conflict_exact: Optional[str]
    evidence: str


# ── The actionable registry (mirror of the C++ table's actionable rows) ──────
# Evidence strings are abbreviated here; the C++ table carries the full
# citations.  (env, axis, cls, forced, conflict, exact) MUST match C++.
REGISTRY: tuple[Knob, ...] = (
    # Enforcers: forced ON.
    Knob("LAYERSTORM_DETERMINISTIC_REDUCE", RUN_TO_RUN, FORCE_ON, "1", IF_FALSY, None,
         "INV-DRIFT-DETREDUCE: legacy atomic path is not bit-reproducible run-to-run"),
    Knob("LAYERSTORM_DETERMINISTIC_EP_COMBINE", RUN_TO_RUN, FORCE_ON, "1", IF_FALSY, None,
         "INV-DRIFT-EPCOMBINE / 4b-1: mode-0 combine follows live residency"),
    Knob("LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION", REFERENCE, FORCE_ON, "bf16", IF_EXACT, "fp32",
         "bf16 is the canonical champion payload; fp32 is equally deterministic but forks the reference"),
    # Forking / nondeterministic: forced OFF.
    Knob("LS_TQ_SPLITKV", REFERENCE, FORCE_OFF, "0", IF_TRUTHY, None,
         "P-29 step 9: run-to-run deterministic but 327/6999 argmax flips vs reference — survives flag 1, forced off by flag 2; default ON since P-29 step 14 (OQ-6) — the canonical route stays unsplit"),
    Knob("LS_SNAPMLA_FP8_DECODE", REFERENCE, FORCE_OFF, "0", IF_TRUTHY, None,
         "P-29 step 10 stage (b): deterministic but +0.0063 nats vs snapMLA-exact, 310/6999 flips; default ON since P-29 step 14 (OQ-7) — the tier's canonical numerics stay exact"),
    Knob("LS_ORCH_SUBGRID_MIDEDGE", REFERENCE, FORCE_OFF, "0", IF_TRUTHY, None,
         "4b-3 / INV-PREFIX-CACHE-1 SUB-GRID: deterministically NON-identical hits"),
    Knob("LS_SPEC_GOVERNOR", RUN_TO_RUN, FORCE_OFF, "0", IF_TRUTHY, None,
         "4b-4 / TD-GLM52-CHAMPION-GREEDY-NONDET-RUN-TO-RUN: trajectory follows timing noise"),
    # Canonical pins (canonical == code default): refuse contrary pins.
    Knob("LS_PINNED_EXACT_WIDTHS", REFERENCE, PIN_DEFAULT, None, IF_FALSY, None,
         "TD-MOE-EP-XTP-PLACEMENT-DRIFT arm (3): =0 diverges at +16"),
    Knob("LS_MOE_RESIDENT_OVERLAP", REFERENCE, PIN_DEFAULT, None, IF_TRUTHY, None,
         "default OFF since P-29 step 18 (OQ-10); =1 re-arms the split whose "
         "INV-MOE-OVERLAP contract-(2) identity is falsified when it engages"),
    Knob("LS_INDEXER_REWIND", REFERENCE, PIN_DEFAULT, None, IF_FALSY, None,
         "=0 drops rewound rows to dense indexer steps — selection change"),
    Knob("LS_TP_COMBINE_FP32", REFERENCE, PIN_DEFAULT, None, IF_TRUTHY, None,
         "=1 changes TP-combine accumulation precision"),
    Knob("LS_CHUNK_SMALLM", REFERENCE, PIN_DEFAULT, None, IF_FALSY, None,
         "route flip explicitly NOT bit-identical (dcp_executor.cpp:1077)"),
    Knob("LS_NO_CHUNK_SMALLM", REFERENCE, PIN_DEFAULT, None, IF_TRUTHY, None,
         "kill-switch spelling of LS_CHUNK_SMALLM=0"),
    Knob("LS_GG_FORCE", REFERENCE, PIN_DEFAULT, None, IF_SET, None,
         "pins GGUF strategy: different-but-valid reduction order"),
    Knob("LS_NO_PERSLOT_ZEROFIX", REFERENCE, PIN_DEFAULT, None, IF_TRUTHY, None,
         "disables a correctness fix (stale per-slot rows)"),
    Knob("LS_ORCH_NO_SC", REFERENCE, PIN_DEFAULT, None, IF_EXACT, "1",
         "=1 returns prefill to the pre-superchunk trajectory (TD-SERVE-SC-TRAJECTORY)"),
    Knob("LS_ORCH_SC_STRIDE", REFERENCE, PIN_DEFAULT, None, IF_SET, None,
         "MoE batch width = grouped-GEMM shape class; recipe governs"),
    Knob("LS_ORCH_PREFILL_CHUNK", REFERENCE, PIN_DEFAULT, None, IF_SET, None,
         "prefill chunk rows = batch-width shape class; recipe governs"),
    Knob("LS_ORCH_FORCE_SPLIT_ACT", REFERENCE, PIN_DEFAULT, None, IF_EXACT, "1",
         "=1 restores legacy split/ACT arm (INV-LOADER-ACT-FPDRIFT)"),
    Knob("LAYERSTORM_MOE_BIG_VRAM_CAP_MB", REFERENCE, PIN_DEFAULT, None, IF_SET, None,
         "caps BIG-path VRAM -> prefill batch-width shape class"),
    Knob("LS_MOE_BIG_FIT_HEADROOM_MB", REFERENCE, PIN_DEFAULT, None, IF_SET, None,
         "chunk-fit headroom override -> realized chunk capacity = EP4 "
         "stride clamp -> prefill batch-width shape class; recipe governs"),
    Knob("LS_KVT_COHORT_ALWAYS", REFERENCE, PIN_DEFAULT, None, IF_TRUTHY, None,
         "TD-KVT-COHORT-PREFILL-NUMERICS open — pinned to default-auto"),
    Knob("LS_KVT_COHORT_ROWWISE", REFERENCE, PIN_DEFAULT, None, IF_TRUTHY, None,
         "same open numerics TD — pinned to default-auto"),
    Knob("LS_CPU_EXPERT", RUN_TO_RUN, PIN_DEFAULT, None, IF_TRUTHY, None,
         "CPU expert path: multithreaded reduction order run-to-run UNPROVEN — pinned under both flags"),
    Knob("LS_CPU_EXPERT_FORCE", RUN_TO_RUN, PIN_DEFAULT, None, IF_TRUTHY, None,
         "force-arm of the CPU expert path — same run-to-run-unproven class"),
    Knob("LS_V4_ROW_PREFILL", REFERENCE, PIN_DEFAULT, None, IF_TRUTHY, None,
         "bisect arm: bf16-lsb numerics differ (arch_deepseek_v4.cpp:1140)"),
    Knob("LS_V4_NSP", REFERENCE, PIN_DEFAULT, None, IF_SET, None,
         "V4 CSA split count — FP reassociation class"),
    Knob("LS_CSA_DECODE_LEGACY", REFERENCE, PIN_DEFAULT, None, IF_EXACT, "1",
         "CSA decode kernel-variant flip — unproven identity"),
    Knob("LS_TQ_ROTATE_LEGACY", REFERENCE, PIN_DEFAULT, None, IF_EXACT, "1",
         "TQ q-rotate variant flip — unproven identity"),
    Knob("LS_TQ_ROTATE_TILED", REFERENCE, PIN_DEFAULT, None, IF_EXACT, "1",
         "TQ v-rotate-back variant flip — unproven identity"),
    Knob("LS_MHC_FORCE_LEGACY", REFERENCE, PIN_DEFAULT, None, IF_EXACT, "1",
         "mHC mixes kernel variant flip — unproven identity"),
    Knob("LS_KDA_DECODE_PIPELINE", REFERENCE, PIN_DEFAULT, None, IF_FALSY, None,
         "KDA decode variant flip — default ON is the shipped champion state"),
    Knob("LS_GGUF_E1_KSPLIT", REFERENCE, PIN_DEFAULT, None, IF_FALSY, None,
         "E1 K-split GEMV reduction shape — default ON is the champion state"),
    Knob("LS_GGUF_GROUPED_KSPLIT", REFERENCE, PIN_DEFAULT, None, IF_SET, None,
         "grouped-INT variant selector — default kCompact is golden-gated"),
    Knob("LS_GGUF_GROUPED_KSPLIT_COMPACT", REFERENCE, PIN_DEFAULT, None, IF_SET, None,
         "tri-state override of the kCompact choice — pinned to auto"),
    Knob("LS_GGUF_GROUPED_PIPE", REFERENCE, PIN_DEFAULT, None, IF_FALSY, None,
         "grouped-INT pipeline variant — pinned to default ON"),
    Knob("LS_GGUF_GROUPED_CPASYNC", REFERENCE, PIN_DEFAULT, None, IF_FALSY, None,
         "grouped-INT cp.async variant — pinned to default ON"),
    Knob("LS_GGUF_GROUPED_KSC_SMEM", REFERENCE, PIN_DEFAULT, None, IF_FALSY, None,
         "grouped-INT smem variant — pinned to default ON"),
    Knob("LS_CPU_GEMV_SINGLE_ROW", REFERENCE, PIN_DEFAULT, None, IF_EXACT, "1",
         "CPU GEMV variant flip on the (pinned-off) CPU path"),
    # Proven bit/byte-identical optimizations: deliberately untouched.
    Knob("LS_KDA_PROJ_FUSED", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step 15: fused multi-segment KDA projection mmvq — bit-identical "
         "(unit byte-compare + negative control + identical shas)"),
    Knob("LS_TQ_SPARSE_SLIM", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step 6: bit-identical by construction + on-GPU proof (+26.9% @8k)"),
    Knob("LS_TQ_DECODE_GRAPH", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "same kernels/args replayed; fingerprinted re-capture"),
    Knob("LS_DECODE_CHAIN_GRAPH", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step 7: byte-identical (sha groups identical across arms)"),
    Knob("LS_MOE_FFN_GRAPH", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "INV-0.6(a): bit-identical replay, default ON since 2026-07-20"),
    Knob("LS_FFN_GRAPH_KCOPY", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step 2 KCOPY: same bytes, kernel-node emission only"),
    Knob("LS_MOE_META_CACHE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step 3 MPOKE: identical bytes cached behind fingerprints"),
    Knob("LS_NCCL_FUSE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "bit-identical to separate group calls (disjoint buffers)"),
    Knob("LS_NCCL_GRAPH", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "same collectives, same order (construction argument; default OFF)"),
    Knob("LS_MOE_NULL_SKIP_DECODE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "x+0 exact either way under the pre-zero/re-zero contract"),
    Knob("LS_MOE_DECODE_WAVE_GATE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "pass membership only; x+0 exact => bit-identical tokens"),
    Knob("LS_MOE_WAVE_MASK", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "bitwise-identical output (dispatch_detail.h:106)"),
    Knob("LS_FAR_GATED_FINAL", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "identical compute, only enqueue time changes"),
    Knob("LS_FAR_PROLOGUE_PREISSUE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step 16: MoE prologue pre-issued at FAR time — same kernels/"
         "inputs/stream order, finalize skips only the idempotent recompute"),
    Knob("LS_FAR_PROLOGUE_PRIME", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step-16 bisect sub-knob (prime half) — bit-identical either way"),
    Knob("LS_FAR_PROLOGUE_BCAST", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step-16 bisect sub-knob (broadcast half); since P-29 step 17 default ON "
         "iff LS_MOE_XTP_BOUNCE on (OQ-9 resolved) — bit-identical either way"),
    Knob("LS_FAR_ISSUE_SLIM", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step 17: fused single-pass b_ptr staging fill — staged bytes equal "
         "the legacy per-projection walks; host-only bookkeeping"),
    Knob("LS_MOE_XTP_BOUNCE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step 17: EP-XTP broadcast via explicit pinned-bounce — identical "
         "bytes to identical destinations, equivalent ordering"),
    Knob("LS_SPEC_VERIFY_FETCH_HIDE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-32 stage 1: B=1 fetch-hiding fast paths extended to spec_verify "
         "R<=8 MoE commands; enqueue/detection time only; consulted only on "
         "spec arms (speculation.method=mtp); =0 restores phase-B exposed-H2D "
         "verify byte-identically"),
    Knob("LS_SPEC_VERIFY_BATCHED", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-32 stage 1: sparse-MLA verify layers run R rows as ONE batched "
         "dispatch (device s_q=R arm, exact per-row kernel bodies, "
         "INV-DSA-BATCH discipline; per-row sub-dispatch fallback); "
         "consulted only on spec_verify FAR commands; =0 restores the "
         "phase-B per-row command loop"),
    Knob("LS_FAR_GATE_DISPATCH", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step 19: force-dispatch the layer's own staged demand copies past "
         "the (measured 100%-stale) inflight cap inside the gated-final commit — "
         "same copies, same dispatch order, enqueue time only"),
    Knob("LS_EVENT_POOL", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step 19: DeviceBackend event free-list — pooled handles query "
         "cudaSuccess so reuse equals a fresh event; no stream content change"),
    Knob("LS_FAR_STREAM_GATE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "device gating of the same kernels; timing only"),
    Knob("LS_MOE_FOLD_VIA_HOST", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "identical fold bytes via a pinned-host hop"),
    Knob("LS_MOE_SKIP_MISS_PROBE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "telemetry source only, compute unchanged"),
    Knob("LS_SNAPMLA_BUDGET_STAGING", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step 10 stage (a): bit-identical, three independent proofs"),
    Knob("LS_KV_EXPERT_REBALANCE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "44z: byte-identical text across all six boots / 66 runs"),
    Knob("LS_MTP_PROBE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "P-29 step 11: zero-perturbation proven (0/98 then 0/5000 byte-match)"),
    Knob("LS_MTP_FORCE_ARM", REFERENCE, PIN_DEFAULT, None, IF_SET, None,
         "P-29 step 13 bisect knob on the MTP spec arm — forced-arm identity unproven, safe-side pin"),
    Knob("LS_MTP_ROW0_PROBE", REFERENCE, PIN_DEFAULT, None, IF_SET, None,
         "P-29 step 13 anchor+restore probe — zero-perturbation unproven in serving, safe-side pin"),
    Knob("LS_V4_META_CACHE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "V4 twin of LS_MOE_META_CACHE — identical meta bytes"),
    Knob("LS_ORCH_NO_MIDEDGE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "identity-safe both ways with sub-grid forced OFF (INV-PREFIX-CACHE-1)"),
    Knob("LS_DSPARK_CTX_ROTATE", REFERENCE, KEEP_ON, None, NO_CONFLICT, None,
         "arm choice only; both arms commit target argmax at B=1 (INV-DSPARK-LOSSLESS)"),
)


class DeterminismConflictError(RuntimeError):
    """Raised when a determinism mode is requested but user pins conflict."""


def _env_bool_or(env: Mapping[str, str], name: str, config_value: bool) -> bool:
    e = env.get(name)
    if e:
        return e[0] != "0"
    return bool(config_value)


def mode_requested(cfg: Optional[Mapping] = None,
                   environ: Optional[Mapping[str, str]] = None) -> str:
    """Resolve the mode: reference > run_to_run > off.

    Per flag, env overrides config either way.  Does not check the
    hierarchy-suppression conflict — apply_deterministic_mode does, loudly.
    """
    env = os.environ if environ is None else environ
    comp = (cfg or {}).get("compute") or {}
    ref = _env_bool_or(env, "LS_REFERENCE_TRAJECTORY_IDENTITY",
                       comp.get("reference_trajectory_identity", False))
    det = _env_bool_or(env, "LS_DETERMINISTIC",
                       comp.get("deterministic", False))
    if ref:
        return MODE_REFERENCE
    if det:
        return MODE_RUN_TO_RUN
    return MODE_OFF


def _env_conflicts(k: Knob, v: Optional[str]) -> bool:
    if v is None:
        return False
    if k.conflict == IF_TRUTHY:
        return v != "" and v[0] != "0"
    if k.conflict == IF_FALSY:
        return v != "" and v[0] == "0"
    if k.conflict == IF_SET:
        return v != ""
    if k.conflict == IF_EXACT:
        return v == (k.conflict_exact or "")
    return False


def _config_conflicts(cfg: Mapping, mode: str) -> list[str]:
    """EXPLICIT config values that contradict the requested mode.

    Only key-present-and-contrary counts (a schema default is not a pin —
    forcing past defaults is the whole point).  The C++ apply cannot see
    key-presence post-parse; this check is the serve-path tightening.
    """
    out: list[str] = []
    comp = cfg.get("compute") or {}
    # Axis kRunToRun contradictions — both modes.
    if comp.get("deterministic_ep_combine") is False:
        out.append("config compute.deterministic_ep_combine explicitly false "
                   "contradicts run-to-run determinism (INV-DRIFT-EPCOMBINE)")
    if comp.get("deterministic_reduce") is False:
        out.append("config compute.deterministic_reduce explicitly false "
                   "contradicts run-to-run determinism (INV-DRIFT-DETREDUCE)")
    # Hierarchy: flag 2 requested while flag 1 explicitly disabled in config.
    if mode == MODE_REFERENCE and comp.get("deterministic") is False:
        out.append("config requests reference_trajectory_identity while "
                   "explicitly setting deterministic: false — reference "
                   "identity IMPLIES run-to-run determinism; drop one")
    # Axis kReference contradictions — reference mode only.
    if mode == MODE_REFERENCE:
        if comp.get("deterministic_ep_combine_precision") == "fp32":
            out.append("config compute.deterministic_ep_combine_precision="
                       "fp32 forks the canonical bf16 reference trajectory")
        pci = cfg.get("_internal-prefix_cache") or {}
        if pci.get("subgrid_mid_edge") is True:
            out.append("config _internal-prefix_cache.subgrid_mid_edge=true "
                       "is deterministically NON-identity-safe "
                       "(INV-PREFIX-CACHE-1 SUB-GRID clause)")
    return out


def apply_deterministic_mode(
    cfg: Optional[Mapping] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    setter: Optional[Callable[[str, str], None]] = None,
) -> list[str]:
    """Apply the requested mode (if any).  Returns the audit-log lines.

    ``environ``/``setter`` are injectable for tests; production callers pass
    neither (os.environ is both read and written, which putenvs through to
    the C++ getenv sites in this same process).

    Raises DeterminismConflictError listing EVERY conflicting pin — including
    the hierarchy-suppression conflict (flag 2 requested while
    LS_DETERMINISTIC=0 pins flag 1 off) — rather than silently overriding.
    """
    env = os.environ if environ is None else environ
    mode = mode_requested(cfg, env)
    if mode == MODE_OFF:
        return []
    if setter is None:
        def setter(k: str, v: str) -> None:  # noqa: F811
            os.environ[k] = v

    reference = mode == MODE_REFERENCE

    conflicts: list[str] = []
    # Hierarchy suppression via env: flag 2 on, flag 1 explicitly "0".
    det_env = env.get("LS_DETERMINISTIC")
    if reference and det_env and det_env[0] == "0":
        conflicts.append(
            "LS_REFERENCE_TRAJECTORY_IDENTITY is requested but "
            "LS_DETERMINISTIC=0 explicitly suppresses run-to-run "
            "determinism — reference identity IMPLIES run-to-run "
            "determinism; drop one of the two pins")
    if cfg is not None:
        conflicts.extend(_config_conflicts(cfg, mode))
    active = [k for k in REGISTRY if reference or k.axis == RUN_TO_RUN]
    for k in active:
        v = env.get(k.env)
        if _env_conflicts(k, v):
            flag = ("run-to-run determinism (LS_DETERMINISTIC)"
                    if k.axis == RUN_TO_RUN else
                    "reference-trajectory identity "
                    "(LS_REFERENCE_TRAJECTORY_IDENTITY)")
            conflicts.append(f"{k.env}={v} pinned by env conflicts with "
                             f"{flag} — {k.evidence}")
    if conflicts:
        msg = ("[determinism] REFUSING to boot: the requested mode is pinned "
               "against. Unset the pin(s) or drop the flag:\n  - "
               + "\n  - ".join(conflicts))
        for c in conflicts:
            log.error("[determinism] CONFLICT: %s", c)
        raise DeterminismConflictError(msg)

    lines: list[str] = [
        "[determinism] mode: %s (LS_DETERMINISTIC=%s, "
        "LS_REFERENCE_TRAJECTORY_IDENTITY=%s)%s" % (
            "REFERENCE-TRAJECTORY IDENTITY (implies run-to-run determinism)"
            if reference else "RUN-TO-RUN DETERMINISM",
            env.get("LS_DETERMINISTIC", "<unset>"),
            env.get("LS_REFERENCE_TRAJECTORY_IDENTITY", "<unset>"),
            "" if reference else
            " — trajectory-forking-but-stable optimizations (e.g. "
            "LS_TQ_SPLITKV) are permitted under this flag")
    ]
    forced = 0
    kept = 0
    for k in active:
        if k.cls in (FORCE_ON, FORCE_OFF):
            prior = env.get(k.env)
            assert k.forced is not None
            tag = "run-to-run" if k.axis == RUN_TO_RUN else "reference"
            if prior == k.forced:
                lines.append(f"[determinism] {k.env}={k.forced} already "
                             f"pinned (conforming, {tag})")
            else:
                lines.append(
                    f"[determinism] FORCED ({tag}) {k.env}={k.forced} "
                    f"(was {prior if prior is not None else '<unset>'})")
                forced += 1
            setter(k.env, k.forced)
        elif k.cls == KEEP_ON:
            kept += 1
    lines.append(
        f"[determinism] forced {forced} knob(s); kept {kept} "
        "proven-bit-identical optimization(s) enabled (make-it-reproducible, "
        "not make-it-slow)")
    lines.append(
        "[determinism] note: DET-TOPK-TIES (INV-TOPK-TIE-DET) is a build-time "
        "precondition (deps/LayerStoRmKernels >= 8228e71) — verified by the "
        "gate suite, not at runtime")
    lines.append(
        "[determinism] note: cross-PLACEMENT bit-identity at depth on EP "
        "shapes is NOT guaranteed (TD-MOE-EP-XTP-PLACEMENT-DRIFT, open)")
    lines.append(
        "[determinism] note: sampled decoding (temperature > 0) is "
        "reproducible only per decode arm with a fixed seed; the reference "
        "discipline is greedy")
    if reference:
        lines.append(
            "[determinism] note: 'reference' is RELATIVE to this "
            "configuration — the canonical numerics path for the chosen "
            "backend/tier/parallelism with no flag-level shortcut forking "
            "it, not a global golden sha")
    for line in lines:
        log.info("%s", line)
    return lines
