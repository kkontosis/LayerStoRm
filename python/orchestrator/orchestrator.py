"""The bridge-based production orchestrator (successor core).

Rewrite of the orchestrator around ``python/bridge`` — the ring layer
proven bit-identical to the C++ champion by
tests/integration/test_dsp52_bridge.py — replacing the 6-phase async
event loop (``orchestrator.loop.orchestrator_loop``) with the measured
champion decode shape driven synchronously:

  * fused ``E_CMD_FAR_FORWARD_LAYER`` burst sweeps (REEF service placement
    + victim maps daemon-side),
  * DSpark speculation in the champion arm — batched verify + overlap
    (draft hidden under an always-committing plain step) + DSP-9
    confidence truncation — for greedy requests,
  * DISTRIBUTION-LOSSLESS sampled speculation (TD-ORCH-SAMPLED-SPEC
    resolved) for temperature>0 (and logprobs) requests: the same
    champion round shape with every committing argmax replaced by an
    exact host-side sample from the target distribution, read from the
    engine's multi-row full-logits readback (kMaxLogitsReadbackRows).
    DSpark drafts deterministically (per-slot argmax), so the exact
    rejection rule for a point-mass proposal is the equality coupling:
    draw s_k ~ p_k, accept the draft iff s_k equals it — see
    _decode_speculative_sampled for the math,
  * the plain sampled AR path (CMD_SAMPLE_TOKENS temperature/top_p/top_k/
    seed) as fallback when the multi-row logits readback is unavailable
    (engine builds predating it) or speculation is off/non-adoptable.

ORIENTATION (user direction 2026-08-18): the next step is the OpenAI
endpoint (python/server/http_server.py) consuming THIS class in place of
OrchestratorLoop.  The submission surface mirrors the old loop's
(`submit_request` / `cancel_request` / callbacks / EngineMetadata) so the
rewiring is an import swap plus loop-boot changes in cli/serve.py.  B=1
serially today: the engine's sideband command-input slots are
single-owner (TD-ORCH-SIDEBAND-INPUT-MULTI), so one generation is active
at a time and extra requests queue.  B>1 is an additive lever: sequences
are already first-class (seq_id + batch descriptors), the verify chunk
already batches rows, and admission lives in one place (`_serve_next`).

Threading model: ``run()`` blocks its caller (serve's main thread) and
executes requests serially; HTTP handlers submit from other threads into
a thread-safe deque and wait on their callbacks — same contract as the
old loop.  The engine's C++ daemon thread never takes the GIL; the
bridge's Cython waits release it while spinning.
"""

from __future__ import annotations

import collections
import ctypes
import inspect
import math
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from bridge.ring_bridge import (BridgeError, DsparkDraftError, EngineBridge,
                                is_pool_exhaustion)
from orchestrator.types import EngineMetadata, StepLogprobs, TokenLogprob

__all__ = [
    "InferenceRequest",
    "Orchestrator",
    "PrefixCacheConfig",
    "RequestStats",
    "SamplingParams",
    "SpeculationConfig",
    "assert_identity_preconditions",
]


def assert_identity_preconditions(orch: "Orchestrator") -> None:
    """Dossier 4b, ENFORCED: every token-identity gate, golden and A/B
    harness calls this once after boot, before serving its first leg.
    A harness that silently runs with a divergence source ON produces
    the false-regression class that burned this campaign twice (the S1
    B4 leg; the r3 cache_off drift) — so the preconditions are asserted
    here, in ONE place, not remembered per harness.

      1. deterministic_ep_combine must be ON
         (TD-SERVE-PREFILL-NONDET-RUN-TO-RUN: with the legacy mode-0
         combine, routed partial sums follow live expert residency and
         greedy output is not reproducible run-to-run).
      2. subgrid_mid_edge must be OFF (INV-PREFIX-CACHE-1 SUB-GRID
         clause: with it ON a mid-edge prefix hit reuses the exact LCP
         off the prefill grid and is deterministically NON-identical to
         an uncached run — an accepted class for serving, poison for a
         gate).

      4. the SpecRoundGovernor must be OFF (LS_SPEC_GOVERNOR=0) whenever
         speculation is armed (TD-GLM52-CHAMPION-GREEDY-NONDET-RUN-TO-RUN:
         the governor picks each round's shape — width-1 plain step vs
         batched verify chunk — from wall-clock EMAs, and near-tie argmax
         follows the shape (INV-DSPARK-LOSSLESS B>1 clause), so the
         committed trajectory follows timing noise; runs realize one of a
         small discrete set of equally-valid greedy trajectories).

    The remaining 4b precondition — the DET-TOPK-TIES kernel
    (deps/LayerStoRmKernels >= 8228e71) — is a build-time property the
    runner must verify; it is not observable from the orchestrator.
    """
    assert getattr(orch, "deterministic_ep_combine", True), (
        "dossier 4b: identity legs need deterministic_ep_combine=ON "
        "(compute.deterministic_ep_combine or "
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE=1)")
    assert not getattr(orch, "subgrid_mid_edge", False), (
        "dossier 4b: identity legs need subgrid_mid_edge=OFF "
        "(unset LS_ORCH_SUBGRID_MIDEDGE / "
        "_internal-prefix_cache.subgrid_mid_edge — sub-grid mid-edge "
        "hits are deterministically NON-token-identical, "
        "INV-PREFIX-CACHE-1 SUB-GRID clause)")
    spec = getattr(orch, "spec", None)
    if spec is not None and getattr(spec, "enabled", False):
        assert os.environ.get("LS_SPEC_GOVERNOR", "") == "0", (
            "dossier 4b: identity legs on a speculative arm need "
            "LS_SPEC_GOVERNOR=0 — the round-shape governor converts "
            "wall-clock noise into trajectory choice "
            "(TD-GLM52-CHAMPION-GREEDY-NONDET-RUN-TO-RUN)")

_PREFILL_CHUNK = 64          # test-proven prefill chunk (spec_decode parity)
# Boot-time prefill chunk rows come from config
# (orchestrator.prefill_chunk_tokens, schema default 64): expert-union-
# saturating architectures (DeepSeek-V4-class — the routed union approaches
# the full expert set per layer) carry 512 in their recipes so a maximal
# chunk streams each layer's expert set ~once per chunk (512 = ipc
# kMaxBatchDescriptors / kMaxSidebandTokenIds = the engine's
# elastic-superchunk floor).  Clamped to EngineInfo.moe_batch_capacity.
_DECODE_TIMEOUT_US = 5_000_000
_PREFILL_TIMEOUT_US = 120_000_000

# on_complete error-detail dispatch (TD-SERVE-ERROR-MASKING).  Cached per
# CODE OBJECT: parameter names and the **kwargs flag live in __code__, so
# the answer is identical for every function sharing it (and the cache
# cannot grow with per-request lambdas).
_ERROR_ARG_CACHE: dict[Any, bool] = {}


def _accepts_error_arg(cb: Callable[..., None]) -> bool:
    """True when an ``on_complete`` consumer declares the optional
    ``error`` keyword (or absorbs it via ``**kwargs``).

    Threading the failure DETAIL to the caller must not break the legacy
    four-argument contract, so the orchestrator asks the callback what it
    can take (see InferenceRequest)."""
    fn = getattr(cb, "__func__", cb)
    key = getattr(fn, "__code__", None)
    if key is not None and key in _ERROR_ARG_CACHE:
        return _ERROR_ARG_CACHE[key]
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):              # C callables, builtins
        ok = False
    else:
        p = params.get("error")
        ok = ((p is not None
               and p.kind is not inspect.Parameter.POSITIONAL_ONLY)
              or any(q.kind is inspect.Parameter.VAR_KEYWORD
                     for q in params.values()))
    if key is not None:
        _ERROR_ARG_CACHE[key] = ok
    return ok


# ---------------------------------------------------------------------------
# Request / config surface
# ---------------------------------------------------------------------------

@dataclass
class SamplingParams:
    """OpenAI-shaped sampling controls.  temperature==0.0 → greedy argmax
    (the byte-identical champion speculative arm); temperature>0 →
    distribution-lossless sampled speculation (host-side sampling from
    the engine's per-row logits readback) with the plain sampled AR path
    as fallback.  ``seed`` semantics: a fixed seed reproduces the token
    stream exactly WITHIN a decode arm (each arm owns its RNG stream —
    host np.default_rng(seed) on the speculative/guided arms, per-step
    Philox keys derived from (seed, position) on the plain engine-side
    arm), not across arms."""
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 42

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    def as_tuple(self) -> tuple[float, float, int, int]:
        return (self.temperature, self.top_p, self.top_k, self.seed)


@dataclass
class InferenceRequest:
    """External request (contract-compatible with the old loop's).

    on_token(request_id, token_id, step_logprobs) fires per committed
    token IN ORDER from the orchestrator thread; on_complete(request_id,
    tokens, finish_reason, logprobs) fires exactly once.  finish_reason ∈
    {"stop", "length", "cancelled", "error"}.

    ERROR DETAIL (TD-SERVE-ERROR-MASKING): a consumer that declares a
    fifth ``error`` parameter (or absorbs ``**kwargs``) additionally
    receives the failure detail as the ``error=`` KEYWORD — "" on every
    non-error completion.  The parameter is OPTIONAL: legacy four-argument
    consumers are called exactly as before.  A serving front end MUST use
    it — a failed request has to surface as an HTTP error, never as a
    200-with-empty-body (INV-SERVE-ERROR).

    ``logprobs`` (OpenAI semantics): None = off (byte-identical decode,
    zero readback cost); K >= 0 = serve per-token logprobs — each
    committed token carries a StepLogprobs (chosen-token logprob + top-K,
    K may be 0) computed HOST-SIDE by log-softmax over the engine's
    full-logits readback row; on_complete additionally carries the full
    per-token list.  Logprobs requests keep the speculative speedup when
    the engine exposes the multi-row logits readback (every verify row's
    raw logits are read back — TD-ORCH-LOGPROBS-SPEC resolved); on
    single-row engine builds they fall back to the PLAIN arm.  Token
    output is unchanged either way (INV-DSPARK-LOSSLESS / the equality
    coupling)."""
    request_id: int
    prompt_token_ids: list[int] = field(default_factory=list)
    max_tokens: int = 0                      # 0 = unlimited (EOS only)
    sampling: SamplingParams = field(default_factory=SamplingParams)
    logprobs: int | None = None              # None=off; K>=0 = top-K per token
    on_token: Callable[..., None] | None = None
    on_complete: Callable[..., None] | None = None
    # TD-SERVE-NAMED-TOOL-CHOICE (guided decoding): a single-use grammar
    # state implementing the server.guided.GuidedState protocol
    # (pick_and_accept / try_accept / rollback / completed).  Non-None
    # routes the request through grammar-constrained decoding; emission
    # stops with finish_reason "tool_calls" once the grammar completes.
    guided: Any = None
    # Ops knob: force the plain (non-speculative) decode path — used to
    # A/B the guided speculative arm against plain constrained decode.
    force_plain: bool = False


@dataclass
class SpeculationConfig:
    """DSpark arming derived from the engine config (the engine already
    loaded the drafter; this only shapes the Python loop)."""
    enabled: bool = False
    gamma: int = 0                           # speculative_tokens
    method: str = "dspark"                   # P-29 step 13: "dspark" | "mtp"
    index_kpool: int = 1                     # P-29 step 13: pool-boundary anchors
    conf_thresh: float = 0.0                 # 0 = truncation off
    # TD-DSPARK-CTX-CAP mirror (TD-PREFIX-DSPARK-FORK-CTX residual): the
    # draft's context-KV arena capacity in tokens.  With ctx_rotate FALSE a
    # request whose context would overflow it (prompt + generation budget)
    # is routed to the PLAIN arm upfront — the engine invalidates the
    # drafting context on arena overflow and a later RUN_DSPARK_STEP
    # against the invalid context CMP_ERRORs the whole request (surfaced by
    # long-prompt serving once TD-KVT-ADMISSION-UPFRONT made >8k-token
    # prompts admissible).
    ctx_cap_tokens: int = 8192
    # TD-DSPARK-CTX-POLICY: the engine rotates the draft context arena as a
    # sliding window (LS_DSPARK_CTX_ROTATE, default ON; classic dspark
    # backbone only — the V4 dflash arm keeps the legacy cap).  When TRUE,
    # over-cap prompts stay on the speculative arm: the draft attends the
    # newest <= ctx_cap_tokens positions and drafting survives any length.
    ctx_rotate: bool = True


class SpecRoundGovernor:
    """Content-adaptive speculation governor (TD-DSPARK-CTX-POLICY,
    acceptance axis).

    S10 (spec/reports/SERVING_REMEASURE_S10.md) measured 4 of 6 sub-8k
    SPECULATIVE turns at or below the plain band on agentic tool-result
    content (acc 0.07/0.12/0.33/0.57 -> 3.92/4.64/4.90/4.69 tok/s vs plain
    4.8-5.0): at low acceptance a round still pays the verify chunk and the
    draft's GPU contention (the draft shares the target's GPUs) for almost
    no accepted tokens.  Restoring drafting past the 8k window (ctx
    rotation) without an acceptance guard would extend that loss regime to
    arbitrary depth.

    Mechanism: the governor measures the running speculative GAIN per round
    -- gain = committed_tokens * plain_step_ms - round_wall_ms -- as an EMA,
    against a plain-step baseline learned from CLEAN plain steps (the seed
    step and suspended rounds; steps overlapped with a draft are contended
    and never update the baseline).  While the gain EMA is negative the
    round loop decodes PLAIN (no draft issued, no verify chunk); every
    probe_period-th suspended round runs one full speculative PROBE round
    so a content shift re-raises the EMA and drafting resumes.

    Token-lossless PER SHAPE, not per prompt (corrected 2026-09-02,
    TD-GLM52-CHAMPION-GREEDY-NONDET-RUN-TO-RUN): both round shapes commit
    target-argmax (or target-sampled) tokens only, but a verify chunk's
    argmax is the BATCHED forward's own output (INV-DSPARK-LOSSLESS B>1
    clause) — a legitimately different FP-reduction shape from the width-1
    plain step, so near-tie positions flip between the two shapes.  The
    committed trajectory is therefore a deterministic function of the
    ROUND-SHAPE SEQUENCE (proven byte-identical given equal sequences,
    across boots and arena states), and the governor picks that sequence
    from wall-clock EMAs — i.e. with the governor enabled, greedy decode
    run-to-run reproducibility is NOT guaranteed; runs realize one of a
    small discrete set of equally-valid greedy trajectories.  Identity
    gates and champion A/Bs must run LS_SPEC_GOVERNOR=0 (deterministic,
    dossier §4b precondition 4) or group runs by the [orch-stats]
    rounds/acc/gov_plain/gov_probes trajectory tuple.

    Env: LS_SPEC_GOVERNOR=0 disables (always draft — pre-slice behavior);
    LS_SPEC_GOV_PROBE overrides the probe period (default 16)."""

    def __init__(self, enabled: bool = True, probe_period: int = 16,
                 alpha: float = 0.25, warmup_rounds: int = 4) -> None:
        self.enabled = enabled
        self.probe_period = max(2, probe_period)
        self.alpha = alpha
        self.warmup_rounds = warmup_rounds
        self.plain_ms = 0.0          # EMA of clean plain-step wall
        self.gain_ms = 0.0           # EMA of per-round speculative gain
        self.suspended = False
        self.probes = 0              # probe rounds issued while suspended
        self._have_plain = False
        self._have_gain = False
        self._rounds = 0             # speculative rounds observed
        self._since_probe = 0

    @classmethod
    def from_env(cls) -> "SpecRoundGovernor":
        return cls(
            enabled=os.environ.get("LS_SPEC_GOVERNOR", "") != "0",
            probe_period=int(os.environ.get("LS_SPEC_GOV_PROBE", "16")))

    def note_plain(self, ms: float) -> None:
        """One CLEAN plain step's wall (seed step / suspended round)."""
        if ms <= 0.0:
            return
        if self._have_plain:
            self.plain_ms += self.alpha * (ms - self.plain_ms)
        else:
            self.plain_ms, self._have_plain = ms, True

    def draft_this_round(self) -> bool:
        """Round-shape decision: True = speculative round (draft+verify),
        False = plain round.  Handles the probe cadence internally."""
        if not self.enabled or not self.suspended:
            return True
        self._since_probe += 1
        if self._since_probe >= self.probe_period:
            self._since_probe = 0
            self.probes += 1
            return True
        return False

    def note_round(self, tokens: int, ms: float) -> None:
        """One speculative round's outcome: tokens committed, wall ms."""
        if not self.enabled or not self._have_plain or ms <= 0.0:
            return
        gain = tokens * self.plain_ms - ms
        if self._have_gain:
            self.gain_ms += self.alpha * (gain - self.gain_ms)
        else:
            self.gain_ms, self._have_gain = gain, True
        self._rounds += 1
        if self._rounds < self.warmup_rounds:
            return
        if self.gain_ms < 0.0:
            self.suspended = True
        elif self.gain_ms > 0.0:
            self.suspended = False


@dataclass
class PrefixCacheConfig:
    """serving.prefix_cache + _internal-prefix_cache (config/schema.json)
    — basic prompt-prefix KV caching via retained SEQ_FORK holder
    sequences.

    max_cached_tokens is the TOTAL unique-token budget (chain-aware, see
    PrefixCache.total_unique_tokens); max_entry_tokens is the independent
    PER-ENTRY registration cap (0 = bounded only by the total budget).
    They were ONE number (8192) from the 2026-08-18 landing until
    2026-08-26 — a leftover from before TD-KVT-ADMISSION-UPFRONT made
    >8k prompts servable — which silently killed prefix caching for every
    long prompt AND let a handful of sub-8192 prompts evict each other.
    Budgets are soft working-set bounds, not reservations: page-pool /
    indexer-K exhaustion evicts holders on demand (evict_for_admission +
    fork evict-retry, TD-INDEXER-POOL-EVICT), so a generous budget
    degrades by LRU churn, never by wedging admission."""
    enabled: bool = True
    max_cached_tokens: int = 131072
    max_entries: int = 8
    max_entry_tokens: int = 0
    # R3 (TD-PREFIX-POOL-PRESSURE-EVICTS-THE-PRIZE): demote a freshly
    # registered holder's hot kMain pages to the tiering cold pool
    # (CMD_SEQ_HIBERNATE).  Without it a frozen holder never steps, so
    # window demotion never reaches it and it pins ~the retention window
    # of VRAM (measured ~5,100 pages/rank per deep holder on the GLM
    # champion — the churn mechanism).  No-op on arms without KV tiering.
    hibernate_holders: bool = True
    # SUB-GRID mid-edge reuse (USER DECISION 2026-08-28;
    # TD-PREFIX-MIDEDGE-SUBGRID-CLASS resolved 2026-08-29: the repeat
    # gate proved page-aligned sub-grid diverges too — NO sub-grid
    # subclass is identity-safe): opt-in, DEFAULT OFF.  OFF = the
    # shipped R4c grid-clamped policy (INV-PREFIX-CACHE-1 holds
    # unconditionally).  ON = mid-edge reuse takes the EXACT token LCP
    # (gain: LCP mod grid extra tokens per hit, up to grid-1, avg ~256
    # on the champion ~= 6 s of TTFT at champion prefill rates) and
    # INV-PREFIX-CACHE-1 RELAXES for those hits to the R4b-measured
    # accepted-divergence class: a DETERMINISTIC, shape-induced,
    # different-but-valid greedy trajectory (~1e-3 logit deltas at the
    # flipping steps; first divergence within ~15 decode tokens in the
    # measured legs) -- the same accepted class as
    # TD-SERVE-SC-TRAJECTORY.  Identity gates / goldens / A/B harnesses
    # MUST assert this is OFF (assert_identity_preconditions below --
    # dossier 4b precondition 3).  Env LS_ORCH_SUBGRID_MIDEDGE
    # overrides EITHER WAY when set (first char != "0" = ON), mirroring
    # LAYERSTORM_DETERMINISTIC_EP_COMBINE precedence; the
    # LS_ORCH_NO_MIDEDGE=1 kill switch disables ALL mid-edge reuse and
    # therefore sub-grid too.  Inert on non-truncatable archs (V4
    # in-place rings): the engine cannot fork at an interior length
    # there (INV-SEQ-FORK-TRUNC), so the legacy filter stays regardless.
    subgrid_mid_edge: bool = False
    # P-29 step 24 (OQ-13, LS_KDA_PREFIX_CKPT): position-keyed UNCOMPRESSED
    # host-RAM KDA state checkpoints captured during prefill, so a DIVERGED
    # prompt on a recurrent-state arch (glm5_next) replays from the nearest
    # checkpoint <= the divergence instead of re-prefilling from zero.
    # Restore is a whole-slot fp32 byte round-trip; replay from a
    # 64-aligned uniform frontier is BIT-IDENTICAL (INV-KDA-CARRY) — the
    # feature changes admission latency, never bytes. DEFAULT ON since
    # 2026-09-06 (user decision) — the three byte-identity gates are green
    # (golden fork-replay bit-exact, TF byte-compare 1023/1023 zero
    # mismatches, in-vivo sha == control); env LS_KDA_PREFIX_CKPT
    # overrides enabled EITHER WAY when set.
    kda_ckpt_enabled: bool = True
    kda_ckpt_interval_tokens: int = 2048
    kda_ckpt_budget_bytes: int = 32 << 30

    @classmethod
    def from_config(cls, cfg: dict) -> "PrefixCacheConfig":
        pc = (cfg.get("serving") or {}).get("prefix_cache") or {}
        pci = cfg.get("_internal-prefix_cache") or {}
        kc = pc.get("kda_checkpoints") or {}
        return cls(
            enabled=bool(pc.get("enabled", True)),
            max_cached_tokens=int(pc.get("max_cached_tokens", 131072)),
            max_entries=int(pc.get("max_entries", 8)),
            max_entry_tokens=int(pci.get("max_entry_tokens", 0)),
            hibernate_holders=bool(pci.get("hibernate_holders", True)),
            subgrid_mid_edge=bool(pci.get("subgrid_mid_edge", False)),
            kda_ckpt_enabled=bool(kc.get("enabled", True)),
            kda_ckpt_interval_tokens=int(kc.get("interval_tokens", 2048)),
            # Schema carries MiB (the config parser's int is 32-bit);
            # internally everything is bytes.
            kda_ckpt_budget_bytes=int(kc.get("budget_mib", 32768)) << 20,
        )


@dataclass
class PrefixSpillConfig:
    """_internal-prefix_spill (TD-PREFIX-TIDY-COLD-SPILL): age idle
    hibernated prefix holders' cold KV out to disk — the SECOND tiering
    hop.  The orchestrator owns the POLICY (which holder, when: leaf
    holders only — an interior node's cold slots are refcount-shared with
    its descendants, so spilling one frees no RAM — idle >= idle_seconds
    since the last hit, LRU-first, at most one per idle sweep, never on
    the request path).  The engine owns the byte moves and enforces the
    byte cap exactly, BEFORE each write (CMD_SEQ_HIBERNATE spill=1;
    status 2 = cap refusal -> evict the LRU spilled holder, deleting its
    file, and retry once).  The directory is a CACHE, never a store:
    ls-spill-*.kvspill files are reclaimed wholesale at startup and a
    lost/unreadable file degrades to a cache MISS (the holder is evicted;
    the request re-prefills — never fails).  An unwritable directory
    DISABLES the feature loudly at boot."""
    enabled: bool = True
    path: str = "~/.layerstorm/prefix-cache"
    max_mib: int = 32768
    idle_seconds: float = 300.0

    @classmethod
    def from_config(cls, cfg: dict) -> "PrefixSpillConfig":
        ps = cfg.get("_internal-prefix_spill") or {}
        return cls(
            enabled=bool(ps.get("enabled", True)),
            path=str(ps.get("path", "~/.layerstorm/prefix-cache")),
            max_mib=int(ps.get("max_mib", 32768)),
            idle_seconds=float(ps.get("idle_seconds", 300.0)),
        )


@dataclass
class RequestStats:
    tokens: int = 0
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    prefill_ms: float = 0.0
    decode_wall_ms: float = 0.0
    prefix_hit_tokens: int = 0               # prompt tokens NOT re-prefilled
    # Guided decoding observability (constrained requests only):
    grammar_trunc_slots: int = 0             # draft slots cut by the grammar
    grammar_refeeds: int = 0                 # violation re-feed masked steps
    # Spec→plain fallback (INV-SERVE-SPEC-FALLBACK): the 1-based round at
    # which a draft-side dspark failure switched this request onto the
    # plain arm for its remainder; 0 = never fell back.
    spec_fallback_round: int = 0
    # SpecRoundGovernor (TD-DSPARK-CTX-POLICY acceptance axis): rounds this
    # request decoded PLAIN because the measured speculative gain was
    # negative (suspension), and speculative PROBE rounds issued while
    # suspended.  0/0 = the governor never engaged (healthy acceptance).
    gov_plain_rounds: int = 0
    gov_probe_rounds: int = 0
    # TD-MOE-PROGRESSIVE-DEGRADED-SILENT: layers whose progressive MoE
    # finalized DEGRADED during this request (router-selected experts left
    # out for a capacity/deadline reason — the engine's graceful degradation
    # under INV-FAR-WAVE).  0 = every layer computed its full routed set.
    # Nonzero means this request's output was computed DIFFERENTLY from a
    # healthy run: serving may retry it, and identity/golden harnesses must
    # discard it rather than gate on its tokens.
    moe_degraded_layers: int = 0
    # TD-V4-FIRSTREQ-COLD-SHARE-OVER-CAPACITY option (d): times this
    # request was RE-RUN by the serving-level degraded-retry policy before
    # the delivered attempt (0 = served first try).  The delivered
    # attempt's own degradation is moe_degraded_layers above -- nonzero
    # there DESPITE retries here means the box degrades repeatedly, not
    # just on the post-boot cold share.
    degraded_retries: int = 0
    # TD-INDEXER-NO-DENSE-FALLBACK witness: attention steps of THIS request
    # that ran (some row) DSA-DENSE because of dead indexer coverage
    # (IndexerSeqMode::kDead).  With reserve-at-admission live this MUST be
    # 0 — any nonzero is a BUG (a silently ~10x-slower request), surfaced on
    # [orch-stats] and by a loud WARNING, never accepted as degradation.
    indexer_dense_steps: int = 0
    # P-29 step 24 (LS_KDA_PREFIX_CKPT): checkpoint position this request
    # restored from (0 = no divergence restore — full miss or ordinary
    # extension hit), checkpoints captured during this request's prefill,
    # and host bytes they consumed (measured from capture completions).
    ckpt_restore_pos: int = 0
    ckpt_captured: int = 0
    ckpt_bytes: int = 0

    @property
    def tok_per_s(self) -> float:
        return (1000.0 * self.tokens / self.decode_wall_ms
                if self.decode_wall_ms > 0 else 0.0)


class _Cancelled(Exception):
    pass


class _DegradedRetry(Exception):
    """Internal (TD-V4-FIRSTREQ-COLD-SHARE-OVER-CAPACITY option (d)): a
    retry-eligible request saw degraded MoE finalizes during PREFILL --
    abandon the attempt before decode emits a single token, so even
    STREAMING requests are retried without double-emitting.  Raised only
    while _serve_next holds retry budget; the attempt's sequence is freed
    by _generate's ``finally`` and no prefix holder was registered (the
    degraded-registration guard precedes this check)."""

    def __init__(self, layers: int) -> None:
        super().__init__(f"{layers} degraded MoE layer(s) during prefill")
        self.layers = layers


class _PrefixEntry:
    """Radix-tree node: one frozen prefix-holder sequence.

    ``parent`` is the LONGEST registered proper prefix of ``tokens`` (None
    for a root); ``children`` are the registered extensions with no
    intermediate registered node.  Siblings are mutually prefix-
    incompatible by construction, so a prompt descends through at most one
    child per level.  The tree edge ``parent → self`` spans the unique
    token tail ``tokens[len(parent.tokens):]`` — the only real memory this
    node adds (CoW page sharing, chain policy 2026-08-18)."""

    __slots__ = ("tokens", "seq_id", "last_used", "parent", "children",
                 "wall_touch", "spilled", "spill_attempted",
                 "kda_ckpts", "kda_ckpt_bytes")

    def __init__(self, tokens: tuple[int, ...], seq_id: int,
                 last_used: int = 0) -> None:
        self.tokens = tokens
        self.seq_id = seq_id
        self.last_used = last_used               # LRU stamp (monotonic)
        self.parent: "_PrefixEntry | None" = None
        self.children: list["_PrefixEntry"] = []
        # TD-PREFIX-TIDY-COLD-SPILL: wall-clock hit stamp (age policy),
        # whether the holder's cold pages live on disk, and whether a
        # spill was already attempted and declined (V4 arm / nothing
        # cold / engine cap) — never re-issued every sweep.
        self.wall_touch = time.monotonic()
        self.spilled = False
        self.spill_attempted = False
        # P-29 step 24 (LS_KDA_PREFIX_CKPT): sorted positions of the KDA
        # state checkpoints this holder carries (the engine MOVED the
        # blobs here at the frozen registration fork), and their total
        # host bytes (budget bookkeeping — freed with the holder).
        self.kda_ckpts: tuple[int, ...] = ()
        self.kda_ckpt_bytes = 0


class PrefixCache:
    """Radix tree of retained PREFIX HOLDER sequences (LRU-evicted).

    Each node is a frozen sequence (created by SEQ_FORK from a working
    sequence right after its prompt prefill) whose refcounted CoW pages
    keep the prefix KV resident — "unused but still accessible".  A later
    request whose prompt starts with a registered prefix forks FROM the
    holder and prefills only the delta.  Correctness invariant
    (INV-PREFIX-CACHE-1): cache ON vs OFF is token-identical — the reused
    pages ARE the recomputed bytes; only prefill wall changes.  Nodes
    exist ONLY at lengths the registration grid blesses (grid-aligned or
    exact prompt body — the shape-identity contract), so a descent can
    never land on a reuse point an uncached run would not reproduce
    (spec/plans/RADIX_SLAB_DESIGN.md §3).

    Holder cost (R3, TD-PREFIX-POOL-PRESSURE-EVICTS-THE-PRIZE): on
    GLM/DSA a registration fork is FROZEN — the engine skips both CoW
    frontier splits (kMain group + indexer-K group; a holder never
    appends, so the NEXT fork from it performs the CoW) — and the holder
    is then HIBERNATED (its hot kMain pages demoted to the tiering cold
    pool; without that a frozen holder pins ~the retention window of
    VRAM forever, the measured churn mechanism: ~5,100 pages/rank per
    deep holder on the champion).  On V4 every node still owns a full
    copy-on-fork side-tier set (INV-PREFIX-CACHE-3) and hibernation is
    an engine no-op, so the live node count stays bounded by
    max_entries — the boot-time V4 side-pool sizing contract.

    TREE POLICY (TD-PREFIX-RADIX-BLOCKS; supersedes the 2026-08-18 list
    shape, semantics preserved): nested prompts are ancestor paths and
    physically SHARE their common-prefix pages (each was forked over the
    previous one's pages, so a node's real memory is its edge — the
    unique tail beyond its parent):
      1. TOUCH PROPAGATES DOWN: a hit stamps the node and its whole
         ancestor path — they are all in use as prefixes of the used
         state.
      2. BYTE/POOL-pressure EVICTION is leaf-only (a leaf IS a
         chain-maximal entry: an interior node's pages are pinned by its
         descendants' refs, so evicting it frees nothing and forfeits a
         full re-prefill), LRU-major (oldest stamp first), deepest-first
         minor (the tail is cheap to rebuild by delta prefill from its
         surviving parent).
      3. The token budget counts UNIQUE tokens: a node contributes its
         edge length (len − parent len), so a chain costs its longest
         member — matching the CoW physical reality; a naive per-entry
         sum overestimates chains and evicts prematurely.
    """

    def __init__(self, cfg: PrefixCacheConfig, bridge: EngineBridge) -> None:
        self.cfg = cfg
        self._bridge = bridge
        self._entries: list[_PrefixEntry] = []   # all live nodes
        self._roots: list[_PrefixEntry] = []     # nodes without a parent
        self._clock = 0                          # LRU stamp source
        # SC lookup grid (0 = chunked path / every length grid-valid).
        # Set by the orchestrator at boot; used to classify NON-GRID
        # interior nodes (an extended exact-body holder) as low-value:
        # the SC validity filter admits them only for an EXACT repeat of
        # the same prompt, so pool-pressure eviction may retire them
        # (churn ticket must-hold 3 — no immortal unhittable nodes).
        self.sc_grid = 0
        self._hibernate_warned = False
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        # P-29 step 24: host bytes held by REGISTERED entries' KDA
        # checkpoints (the orchestrator's budget ledger; a live request's
        # not-yet-registered captures ride the request until registration
        # moves them here or seq_free drops them). Diverged-hit stats.
        self.kda_ckpt_bytes = 0
        self.kda_ckpt_hits = 0
        # Token LCP of the most recent lookup_kda_ckpt call (hit or miss)
        # — the OQ-13 falsifier instrument: a miss-LCP distribution that
        # sits below the first checkpoint means the cadence (or the
        # prompt template) must change, not the mechanism.
        self.last_kda_lcp = 0
        # Why the most recent register() returned False (None after a
        # success) — surfaced in the serve log by the orchestrator so a
        # skipped registration is never silent (the pre-2026-08-26 silent
        # over-sized skip read as a wall-clock regression).
        self.last_skip: str | None = None

    # ── radix helpers ────────────────────────────────────────────────────

    @staticmethod
    def _is_prefix_of(p: tuple[int, ...], q: tuple[int, ...]) -> bool:
        return len(p) <= len(q) and q[:len(p)] == p

    def _descend(self, tokens: tuple[int, ...],
                 max_len: int) -> "_PrefixEntry | None":
        """Deepest registered node whose tokens are a prefix of ``tokens``
        with length <= ``max_len`` (one radix descent).  All such nodes lie
        on a single root-to-leaf path: registered prefixes of one string
        are totally ordered, and every registered prefix of a node is one
        of its tree ancestors by construction."""
        best: _PrefixEntry | None = None
        candidates = self._roots
        while True:
            nxt = None
            for c in candidates:                 # ≤ 1 sibling can match
                if (len(c.tokens) <= max_len
                        and self._is_prefix_of(c.tokens, tokens)):
                    nxt = c
                    break
            if nxt is None:
                return best
            best, candidates = nxt, nxt.children

    def _attach(self, entry: _PrefixEntry) -> None:
        """Insert a node: parent = longest registered proper prefix; any
        sibling the new node subsumes (a registered extension previously
        attached higher) is reparented under it."""
        parent = self._descend(entry.tokens, len(entry.tokens) - 1)
        siblings = parent.children if parent is not None else self._roots
        moved = [c for c in siblings
                 if self._is_prefix_of(entry.tokens, c.tokens)]
        for c in moved:
            siblings.remove(c)
            c.parent = entry
            entry.children.append(c)
        entry.parent = parent
        siblings.append(entry)
        self._entries.append(entry)

    def _detach(self, entry: _PrefixEntry) -> None:
        """Remove a node; children reparent to its parent (radix node
        deletion — their edges lengthen by the removed edge, so
        total_unique_tokens is invariant)."""
        siblings = (entry.parent.children if entry.parent is not None
                    else self._roots)
        siblings.remove(entry)
        for c in entry.children:
            c.parent = entry.parent
            siblings.append(c)
        entry.children = []
        entry.parent = None
        self._entries.remove(entry)

    def total_unique_tokens(self) -> int:
        """Budget accounting matching the CoW page reality: each node
        contributes its EDGE length — the unique tail beyond its parent
        (a chain costs the length of its longest member)."""
        return sum(len(e.tokens)
                   - (len(e.parent.tokens) if e.parent is not None else 0)
                   for e in self._entries)

    def _victim_order(self) -> list[_PrefixEntry]:
        """BYTE-budget candidates: LEAVES only (a leaf is exactly a
        chain-maximal entry — an interior node's pages are pinned by its
        descendants' refs, so evicting it frees nothing and forfeits a
        full re-prefill; retiring a zero-byte interior under the byte
        budget would spin).  LRU-major (oldest stamp first),
        deepest-first minor."""
        return sorted((e for e in self._entries if not e.children),
                      key=lambda e: (e.last_used, -len(e.tokens)))

    def _non_grid_interior(self, e: _PrefixEntry) -> bool:
        """An EXTENDED non-grid-aligned node (exact-body holder that a
        later registration subsumed): on the SC path the validity filter
        admits it only for an exact repeat of the same prompt, so its
        expected reuse is near zero — retirable under POOL pressure
        (churn ticket must-hold 3), where freeing its holder returns
        engine resources without spinning a byte-budget loop."""
        return (self.sc_grid > 0 and bool(e.children)
                and len(e.tokens) % self.sc_grid != 0)

    def _pool_victim_order(self) -> list[_PrefixEntry]:
        """POOL-exhaustion candidates (engine said a pool is exhausted;
        caller retries the failed op after one eviction): leaves PLUS
        non-grid interiors (see _non_grid_interior).  LRU-major; at equal
        stamps a leaf goes before an interior (an interior retirement
        frees no shared KV), deepest leaf first."""
        return sorted(
            (e for e in self._entries
             if not e.children or self._non_grid_interior(e)),
            key=lambda e: (e.last_used,
                           1 if e.children else 0,
                           -len(e.tokens)))

    def _slot_victim_order(self) -> list[_PrefixEntry]:
        """SLOT-pressure candidates (TD-PREFIX-SUPERSEDE realized on the
        radix tree): leaves PLUS single-child interior nodes.  A
        single-child interior is a SUBSUMED ancestor whose registered
        extensions form one collinear chain; retiring it merges its edge
        into the child — zero KV-byte loss (the child's refs pin every
        shared page: the same argument that keeps interiors out of the
        byte-pressure order) — while freeing the entry SLOT, the holder
        sequence id, its indexer-K frontier page group and, on V4, a full
        copy-on-fork side-tier set.  A BRANCH POINT (>= 2 children) is
        never retirable: two conversations share it as their root, and
        collinearity — not extension count — is the guard (in t1 ⊂ t2 ⊂ t3
        the node t1 has ONE child, t2, however many registered extensions
        exist down the chain).

        Selection is LAZY and crosses lineages: LRU-major (a stale sibling
        conversation goes before the active chain — touch propagation
        keeps a live lineage uniformly fresh); among equal stamps a
        SHALLOWEST single-child interior wins (the slot win with zero byte
        loss — need room for t4, retire t1), then the deepest leaf."""
        return sorted((e for e in self._entries if len(e.children) <= 1),
                      key=lambda e: (
                          e.last_used,
                          0 if e.children else 1,   # interior before leaf
                          len(e.tokens) if e.children else -len(e.tokens)))

    # ── lookup / registration ────────────────────────────────────────────

    def lookup(self, prompt: list[int],
               valid=None) -> _PrefixEntry | None:
        """Longest registered prefix usable for `prompt` (needs at least
        one prompt token beyond the prefix for the seed feed) — one radix
        descent.  ``valid`` (optional ``len -> bool``) filters candidate
        node LENGTHS — the superchunk path accepts only grid-aligned or
        exact-prompt-body nodes (shape-identity, TD-V4-SERVE-PREFIX); the
        descent tracks the deepest VALID node on the match path."""
        key = tuple(prompt)
        best: _PrefixEntry | None = None
        candidates = self._roots
        while True:
            nxt = None
            for c in candidates:                 # ≤ 1 sibling can match
                if (len(c.tokens) < len(key)
                        and self._is_prefix_of(c.tokens, key)):
                    nxt = c
                    break
            if nxt is None:
                break
            if valid is None or valid(len(nxt.tokens)):
                best = nxt
            candidates = nxt.children
        if best is not None:
            # Touch propagation: the hit node AND its whole ancestor path
            # (every registered prefix of it) share one fresh stamp.
            self._clock += 1
            node: _PrefixEntry | None = best
            while node is not None:
                node.last_used = self._clock
                node.wall_touch = time.monotonic()
                node = node.parent
            self.hits += 1
        else:
            self.misses += 1
        return best

    def lookup_mid_edge(
            self, prompt: list[int], grid: int,
            subgrid: bool = False) -> "tuple[_PrefixEntry, int] | None":
        """R4c (GLM-only; caller gates on EngineInfo.seq_fork_truncatable):
        deepest GRID-ALIGNED reuse point derivable from ANY registered
        node, returned as ``(node, reuse_len)`` — the reuse point no
        longer has to be a registered length, because the engine can fork
        a node TRUNCATED to reuse_len (CMD_SEQ_FORK prefix_len,
        INV-SEQ-FORK-TRUNC).  One radix descent: descend while a child is
        a full prefix of the prompt; at the stopping level the best
        MID-EDGE point is the max token-LCP against the candidates there
        (a deeper descendant of a partially-matched child cannot match
        further: the prompt already diverged inside that child's tokens).
        Siblings are mutually prefix-incompatible but may still share
        tokens beyond their parent (ABCD vs ABCE), so all candidates at
        the stopping level are scanned.

        REUSE POLICY (R4b evidence 2026-08-28, spec/measurements/
        glm_serving.md): a legacy-valid FULL node (grid-aligned length or
        exact prompt body) is reused whole; any other LCP is CLAMPED DOWN
        to the last ``grid`` boundary — design RADIX_SLAB_DESIGN §3(b):
        an edge may split at any GRID boundary inside it, NEVER FINER.
        Sub-grid reuse is FORBIDDEN: the R4b offset sweep (det-combine ON,
        DET-TOPK-TIES, degraded legs discarded) measured mid-page delta
        starts producing a DIVERGENT-but-plausible greedy trajectory
        (offsets 777/1535/1537/1599 — including a prefix_len=0 FULL fork
        of a non-grid node, exonerating the truncating-fork primitive),
        while grid-aligned starts are token-identical AND identical by
        construction (the delta realigns to the absolute grid, so its
        command stream equals an uncached run's from that position on).
        ``grid`` must be the ACTIVE prefill stride (superchunk stride on
        the sc path, the chunk size otherwise) — the same absolute grid
        the prefill loops below realign to.

        ``subgrid=True`` (USER DECISION 2026-08-28, opt-in, default OFF
        — PrefixCacheConfig.subgrid_mid_edge / LS_ORCH_SUBGRID_MIDEDGE)
        skips the grid clamp: the reuse point is the EXACT token LCP.
        The caller accepts the R4b-measured divergence class for such
        hits (deterministic, shape-induced, ~1e-3 logit deltas, first
        divergence within ~15 decode tokens — INV-PREFIX-CACHE-1
        SUB-GRID clause); identity harnesses must assert it is OFF
        (assert_identity_preconditions).  The seed-feed clamp and the
        committed-length bound below apply unchanged, so the
        INV-SEQ-FORK-TRUNC caller contract is preserved either way.

        ``reuse_len`` is additionally clamped to len(prompt) - 1 (at
        least one prompt token must remain for the seed feed) and never
        exceeds the source node's registered length — the holder's
        committed, frozen frontier — so the INV-SEQ-FORK-TRUNC caller
        contract (N <= committed length, outside any live rewind depth)
        holds by construction: a holder never steps and never rewinds.
        The source node prefers an EXACT-length ancestor when one is
        registered at reuse_len (legacy full fork, prefix_len 0);
        otherwise the fork truncates (prefix_len = reuse_len).
        """
        assert grid > 0, "grid must be the active prefill stride"
        key = tuple(prompt)
        cap = len(key) - 1                       # seed-feed clamp
        if cap <= 0:
            return None
        best: _PrefixEntry | None = None         # deepest full-match node
        candidates = self._roots
        while True:
            nxt = None
            for c in candidates:                 # <= 1 full match
                if (len(c.tokens) <= cap
                        and self._is_prefix_of(c.tokens, key)):
                    nxt = c
                    break
            if nxt is None:
                break
            best, candidates = nxt, nxt.children
        node: _PrefixEntry | None = best
        lcp = len(best.tokens) if best is not None else 0
        # Mid-edge scan at the stopping level: max LCP wins; ties keep the
        # full-match node (cheaper legacy full fork, proven CoW shape).
        floor = lcp
        for c in candidates:
            lo, hi = floor, min(len(c.tokens), cap)
            if hi <= lo or c.tokens[:lo] != key[:lo]:
                continue
            n = lo
            while n < hi and c.tokens[n] == key[n]:
                n += 1
            if n > lcp:
                node, lcp = c, n
        if node is None or lcp <= 0:
            self.misses += 1
            return None
        # GRID CLAMP (see policy above): full legacy-valid nodes pass
        # through; everything else floors to the grid — unless the
        # sub-grid opt-in is ON, in which case the exact LCP is the
        # reuse point (accepted-divergence class, see docstring).
        if subgrid:
            reuse = lcp
        elif lcp == len(node.tokens) and (lcp % grid == 0 or lcp == cap):
            reuse = lcp
        else:
            reuse = (lcp // grid) * grid
        if reuse <= 0:
            self.misses += 1
            return None
        # Source selection: the deepest full-match node covers reuse
        # whenever reuse <= its length; prefer an EXACT-length node on
        # the ancestor path (legacy full fork beats a truncating one).
        if best is not None and reuse <= len(best.tokens):
            node = best
        e: _PrefixEntry | None = node
        while e is not None and len(e.tokens) >= reuse:
            if len(e.tokens) == reuse:
                node = e
                break
            e = e.parent
        # Touch propagation: the source node and its whole ancestor path
        # are in use as prefixes of the reused state.
        self._clock += 1
        e = node
        while e is not None:
            e.last_used = self._clock
            e.wall_touch = time.monotonic()
            e = e.parent
        self.hits += 1
        return node, reuse

    def has_exact(self, tokens: tuple[int, ...]) -> bool:
        deepest = self._descend(tokens, len(tokens))
        return deepest is not None and deepest.tokens == tokens

    def lookup_kda_ckpt(
            self, prompt: list[int]) -> "tuple[_PrefixEntry, int] | None":
        """P-29 step 24 (LS_KDA_PREFIX_CKPT; caller gates on the arch
        being NON-truncatable and the feature enabled): divergence lookup
        for recurrent-state (KDA) architectures. Called only after the
        legacy whole-node ``lookup`` missed, so the prompt DIVERGES from
        every registered prefix mid-edge — the class that costs a full
        re-prefill today. One radix descent (the ``lookup_mid_edge``
        loop): per candidate at the stopping level, the token LCP with
        the prompt; the reuse point is the LARGEST checkpoint position
        C <= min(LCP, len(entry) - 1) over each candidate AND its
        ancestor path (ancestors are prefixes of the candidate, so their
        checkpoints are valid reuse points for the same prompt). Returns
        ``(entry, C)`` — the caller forks the entry's holder TRUNCATED to
        C (the engine admits it because the holder physically carries a
        checkpoint at C) and replays ``[C, len(prompt))`` through the
        NORMAL chunked prefill: bit-identical over the still-shared span
        (identical tokens from an identical 64-aligned state,
        INV-KDA-CARRY), plain new content past the divergence.

        No checkpoint at or below the divergence => None (full
        re-prefill — never approximate, the INV-KDA-ANCHOR refusal
        discipline). ``last_kda_lcp`` records the token LCP either way
        (the OQ-13 cadence falsifier needs the miss distribution).
        C is at most len(prompt) - 1 by the LCP caps (seed feed keeps at
        least one prompt token to replay).

        PURE QUERY: no hit/touch side effects (it runs on EVERY request
        beside the whole-node lookup and only sometimes wins) — the caller
        does the touch + accounting via note_kda_ckpt_hit when it actually
        selects this result. Records last_kda_lcp for the falsifier."""
        key = tuple(prompt)
        cap = len(key) - 1                       # seed-feed clamp
        self.last_kda_lcp = 0
        if cap <= 0:
            return None
        best: _PrefixEntry | None = None         # deepest full-match node
        candidates = self._roots
        while True:
            nxt = None
            for c in candidates:                 # <= 1 full match
                if (len(c.tokens) <= cap
                        and self._is_prefix_of(c.tokens, key)):
                    nxt = c
                    break
            if nxt is None:
                break
            best, candidates = nxt, nxt.children
        # Candidate set: the deepest full-match node (LCP = its length)
        # plus every stopping-level child scanned for a mid-edge LCP.
        floor = len(best.tokens) if best is not None else 0
        cands: list[tuple[_PrefixEntry, int]] = []
        if best is not None:
            cands.append((best, floor))
        for c in candidates:
            hi = min(len(c.tokens), cap)
            if hi <= floor or c.tokens[:floor] != key[:floor]:
                continue
            n = floor
            while n < hi and c.tokens[n] == key[n]:
                n += 1
            if n > floor:
                cands.append((c, n))
        node: _PrefixEntry | None = None
        best_c = 0
        for cand, lcp in cands:
            self.last_kda_lcp = max(self.last_kda_lcp, lcp)
            e: _PrefixEntry | None = cand
            while e is not None:
                limit = min(lcp, len(e.tokens) - 1)
                for p in reversed(e.kda_ckpts):
                    if p <= limit:
                        if p > best_c:
                            node, best_c = e, p
                        break
                e = e.parent
        if node is None or best_c <= 0:
            return None
        return node, best_c

    def note_kda_ckpt_hit(self, node: "_PrefixEntry") -> None:
        """Account a chosen checkpoint-divergence hit (called by the
        orchestrator only when lookup_kda_ckpt's result actually wins over
        the whole-node lookup — see the pure-query note there): touch the
        node + ancestor path and bump the hit counters."""
        self._clock += 1
        e: _PrefixEntry | None = node
        while e is not None:
            e.last_used = self._clock
            e.wall_touch = time.monotonic()
            e = e.parent
        self.hits += 1
        self.kda_ckpt_hits += 1

    def register(self, tokens: tuple[int, ...], src_seq_id: int,
                 holder_seq_id: int, kda_ckpts: tuple[int, ...] = (),
                 kda_ckpt_bytes: int = 0) -> bool:
        """Fork src → a frozen holder and register it. Returns False when
        registration is skipped (duplicate/empty/over-sized prefix) or the
        fork fails even after eviction (caller proceeds uncached);
        ``last_skip`` then carries the reason for the serve log.

        Per-entry cap and total budget are INDEPENDENT knobs: the cap
        (max_entry_tokens, 0 = no separate cap) bounds one entry; the
        budget (max_cached_tokens) bounds the chain-aware unique-token
        SUM via LRU eviction. An entry longer than the total budget can
        never fit and is refused up front."""
        self.last_skip = None
        if not tokens:
            self.last_skip = "empty prefix"
            return False
        if self.has_exact(tokens):
            self.last_skip = (f"duplicate ({len(tokens)}-token prefix "
                              f"already registered)")
            return False
        cap = self.cfg.max_entry_tokens
        if cap > 0 and len(tokens) > cap:
            self.last_skip = (f"over-sized ({len(tokens)} tokens > "
                              f"max_entry_tokens {cap})")
            return False
        if len(tokens) > self.cfg.max_cached_tokens:
            self.last_skip = (f"over-sized ({len(tokens)} tokens > total "
                              f"budget max_cached_tokens "
                              f"{self.cfg.max_cached_tokens})")
            return False
        if self._fork_with_evict_retry(src_seq_id, holder_seq_id,
                                       protect=None, frozen=True) is None:
            self.last_skip = ("fork-failed (page/indexer pool exhausted "
                              "after eviction)")
            return False
        # R3 holder hibernation: a frozen holder never steps, so window
        # demotion never reaches it — demote its hot kMain pages now (all
        # but the append-frontier group) so it does not pin ~a retention
        # window of VRAM for its whole life.  Fail-safe: on error the
        # holder simply stays hot (logged once).
        if self.cfg.hibernate_holders:
            try:
                self._bridge.hibernate_sequence(holder_seq_id, len(tokens))
            except BridgeError as err:
                if not self._hibernate_warned:
                    self._hibernate_warned = True
                    print(f"  [orch] prefix-holder hibernate failed "
                          f"(holder stays hot): {err}", flush=True)
        self._clock += 1
        entry = _PrefixEntry(tokens, holder_seq_id, self._clock)
        # P-29 step 24: the frozen fork above MOVED the source's KDA
        # checkpoint blobs (all of them — every capture position is <= the
        # fork-time frontier == the registered length) into the holder;
        # mirror the ownership here for lookup + budget bookkeeping.
        if kda_ckpts:
            entry.kda_ckpts = tuple(sorted(kda_ckpts))
            entry.kda_ckpt_bytes = int(kda_ckpt_bytes)
            self.kda_ckpt_bytes += entry.kda_ckpt_bytes
        self._attach(entry)
        self._evict_over_budget(protect=entry)
        return True

    def fork_from(self, entry: _PrefixEntry, dst_seq_id: int,
                  reuse_len: int | None = None,
                  reserve_tokens: int = 0) -> int:
        """Fork a working sequence from a holder (cache hit). Raises
        BridgeError when the pool stays exhausted after evicting every
        other entry.

        ``reuse_len`` (R4c mid-edge reuse): take only the holder's first
        reuse_len tokens via a TRUNCATING fork (CMD_SEQ_FORK prefix_len,
        INV-SEQ-FORK-TRUNC).  None or the full registered length = the
        legacy full fork (prefix_len 0 — byte-identical pre-R4 path).
        Caller contract (the engine cannot verify it): reuse_len <= the
        holder's registered length — its committed, frozen frontier; a
        frozen holder never steps and never rewinds, so any interior
        length is outside live rewind depth by construction."""
        prefix_len = 0
        if reuse_len is not None and reuse_len < len(entry.tokens):
            if not 0 < reuse_len:
                raise ValueError(f"reuse_len {reuse_len} not in "
                                 f"(0, {len(entry.tokens)}]")
            prefix_len = reuse_len
        granted = self._fork_with_evict_retry(entry.seq_id, dst_seq_id,
                                              protect=entry,
                                              prefix_len=prefix_len,
                                              reserve_tokens=reserve_tokens)
        if granted is None:
            raise BridgeError("prefix-cache fork: pool exhausted after "
                              "eviction")
        return granted

    # ── eviction ─────────────────────────────────────────────────────────

    def _evict_over_budget(self, protect: _PrefixEntry | None) -> None:
        """SLOT pressure (entry count) and BYTE pressure (unique-token
        budget) use DIFFERENT victim orders: slots may be reclaimed by
        retiring a subsumed single-child ancestor (zero KV loss,
        TD-PREFIX-SUPERSEDE), bytes only by evicting a leaf (the only
        member whose eviction frees pages).  One victim per iteration;
        retirement is lazy — it happens here, under pressure, never
        eagerly at registration."""
        while self._entries:
            over_slots = len(self._entries) > self.cfg.max_entries
            over_bytes = (self.total_unique_tokens()
                          > self.cfg.max_cached_tokens)
            if not (over_slots or over_bytes):
                return
            order = (self._slot_victim_order() if over_slots
                     else self._victim_order())
            victim = next((e for e in order if e is not protect), None)
            if victim is None:
                return                           # only `protect` remains
            self._detach(victim)
            self._free_holder(victim)

    def _evict_one(self, protect: _PrefixEntry | None) -> bool:
        # POOL pressure (not byte budget): candidates include non-grid
        # interiors — no immortal unhittable nodes (must-hold 3).
        for e in self._pool_victim_order():      # LRU-major
            if e is protect:
                continue
            self._detach(e)
            self._free_holder(e)
            return True
        # FALLBACK (R3 gate finding): the ordered candidates freed nothing
        # the engine needed — e.g. frozen chain holders share one indexer-K
        # frontier group per GENERATION, so freeing leaves only decrements
        # refcounts until a whole generation is gone.  Forward progress of
        # the LIVE request beats cache retention (pre-R3 worst-case
        # parity: wipe the cache rather than fail the request), so under
        # sustained pool pressure grid interiors become evictable too,
        # LRU-major (stale lineages before the active chain).
        for e in sorted(self._entries,
                        key=lambda x: (x.last_used, -len(x.tokens))):
            if e is protect:
                continue
            self._detach(e)
            self._free_holder(e)
            return True
        return False

    def evict_for_admission(self) -> bool:
        """Free ONE holder to make room for a NEW request's seq_create
        (page-pool admission pressure — the 2026-08-23 regression-hunt
        finding: holder pages pinned the pool, a second distinct large
        request died at seq_create). Same LRU-major deepest-first policy
        as every other eviction. Returns False when nothing is evictable."""
        return self._evict_one(protect=None)

    def drop_leaf(self, e: _PrefixEntry) -> bool:
        """Evict one SPECIFIC leaf node (TD-PREFIX-TIDY-COLD-SPILL: the
        spill-directory eviction — freeing a SPILLED holder deletes its
        file engine-side — and the lost-spill-file miss fallback).  Same
        bookkeeping as _evict_one; interior nodes are refused (their
        prefix pages are pinned by descendants)."""
        if e.children or e not in self._entries:
            return False
        self._detach(e)
        self._free_holder(e)
        return True

    def _free_holder(self, e: _PrefixEntry) -> None:
        self.evictions += 1
        # P-29 step 24: the engine releases the holder's KDA checkpoint
        # blobs inside seq_free; settle the orchestrator's budget ledger.
        if e.kda_ckpt_bytes:
            self.kda_ckpt_bytes = max(
                0, self.kda_ckpt_bytes - e.kda_ckpt_bytes)
            e.kda_ckpts = ()
            e.kda_ckpt_bytes = 0
        self._bridge.free_sequence(e.seq_id)     # best-effort inside

    def clear(self) -> None:
        while self._entries:
            e = self._entries[-1]
            self._detach(e)
            self._free_holder(e)

    def _fork_with_evict_retry(self, src: int, dst: int,
                               protect: _PrefixEntry | None,
                               frozen: bool = False,
                               prefix_len: int = 0,
                               reserve_tokens: int = 0) -> int | None:
        """Fork with the pool-eviction retry seam.  Returns the engine's
        GRANTED indexer-K reservation (0 = none — legacy bridge, frozen
        holder, or no reservation requested); None = exhausted with
        nothing left to evict.  ``reserve_tokens`` is passed through only
        when nonzero, so pre-reservation mock bridges keep working."""
        while True:
            try:
                if reserve_tokens > 0:
                    granted = self._bridge.fork_sequence(
                        src, dst, frozen=frozen, prefix_len=prefix_len,
                        reserve_tokens=reserve_tokens)
                else:
                    granted = self._bridge.fork_sequence(
                        src, dst, frozen=frozen, prefix_len=prefix_len)
                return int(granted or 0)
            except BridgeError as err:
                if not is_pool_exhaustion(err):
                    raise
                if not self._evict_one(protect):
                    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """B=1 serving orchestrator over EngineBridge (see module docstring).

    Production boot: ``Orchestrator.boot(config_path)``.  Tests inject a
    ready bridge + metadata directly through ``__init__``.
    """

    # TD-SERVE-PREFILL-NONDET-RUN-TO-RUN: effective routed-EP combine
    # determinism, resolved by boot() from config compute
    # .deterministic_ep_combine with env LAYERSTORM_DETERMINISTIC_EP_COMBINE
    # overriding EITHER way (the engine ctor precedence, mirrored).  False =>
    # the engine sums the routed expert partials in the legacy mode-0
    # placement-DEPENDENT grouping (per-rank sums follow live expert
    # residency), so greedy output is NOT reproducible run-to-run even at
    # equal shapes -- token-identity harnesses must consult this flag and
    # refuse identity verdicts when it is False.  Direct-constructed test
    # orchestrators (null engines) default True.
    deterministic_ep_combine: bool = True

    def __init__(self, bridge: EngineBridge, *,
                 metadata: EngineMetadata,
                 speculation: SpeculationConfig | None = None,
                 prefix_cache: PrefixCacheConfig | None = None,
                 prefix_spill: "PrefixSpillConfig | None" = None,
                 engine_module: Any = None,
                 teacher_forced_prefill: bool = False,
                 prefill_chunk: int = _PREFILL_CHUNK,
                 chunk_prefill_arms_draft: bool = True,
                 prefill_superchunk: bool = False,
                 prefill_superchunk_stride: int = 0,
                 prefill_sc_min_tokens: int = 256,
                 prefill_sc_small_chunk: int = 64,
                 degraded_retry_max: int = 0,
                 kvxp_large_prefill_tokens: int = 8192,
                 kvxp_large_prefill_wait_ms: int = 2000,
                 kv_expert_rebalance_enabled: bool = False) -> None:
        self.bridge = bridge
        self.metadata = metadata
        self.spec = speculation or SpeculationConfig()
        # TD-INDEXER-NO-DENSE-FALLBACK (Route 1) capability probe: the real
        # EngineBridge takes reserve_tokens on create/fork; unit-test mock
        # bridges predate it and opt in by adding the parameter.  When the
        # bridge cannot reserve, admission behaves exactly as before (legacy
        # lazy indexer growth) — no kwargs are passed.
        try:
            import inspect as _inspect
            self._bridge_reserves = (
                "reserve_tokens"
                in _inspect.signature(bridge.create_sequence).parameters
                and "reserve_tokens"
                in _inspect.signature(bridge.fork_sequence).parameters)
        except (TypeError, ValueError):
            self._bridge_reserves = False
        # TD-V4-CHUNK-PREFILL RESOLVED (2026-08-21): the V4 executor now
        # serves single-sequence prefill chunks (per-row loop, monotone
        # windows), so V4 boots take the chunked path like GLM.  True keeps
        # the per-token teacher-forced feed (the ticket-H golden lock-step
        # shape) — retained as a debug/measurement arm only.
        self._teacher_forced_prefill = bool(teacher_forced_prefill)
        # Prefill chunk size.  GLM champion: 64 (test-proven, spec_decode
        # parity — and the prefix-cache registration grid).  V4: 512 — the
        # expert-fetch UNION saturates toward the full 256-expert set per
        # layer, so bigger chunks stream each layer's experts ~once
        # (512 = kMaxBatchDescriptors = the elastic-superchunk floor).
        self._prefill_chunk = max(1, int(prefill_chunk))
        # SC (superchunk port): prefill the prompt body in SUPERCHUNKS up
        # to EngineInfo.moe_batch_capacity — sub-chunked attention
        # (row_offset + superchunk flag, exports unioned) + ONE
        # FETCH_AND_RUN_MOE_BIG per layer, so each layer's expert union
        # streams once per superchunk instead of once per 512-row chunk.
        # Every serving boot enables it (with prefill_moe_big; default ON
        # since 2026-08-23).
        self._prefill_superchunk = bool(prefill_superchunk)
        # Superchunk STRIDE: tokens per superchunk (also the prefix-cache
        # registration grid on the sc path).  0 = the engine's elastic
        # moe_batch_capacity (valid on TP-only topologies).  Topologies
        # with expert-only ranks (the EP4 champions) must bound it at the
        # engine's single-shot MoE chunk capacity —
        # max(compute.moe_big_chunk_tokens, orchestrator.max_batch_size)
        # — because CHUNKED MoE batches are rejected with expert-only
        # ranks resident (TD-MOE-EP-XTP-WAVES) and the layer's output
        # would be WRONG; at or below the capacity MOE_BIG runs the
        # byte-identical single-shot pipeline (INV-MOE-BIG-1).
        self._prefill_sc_stride = max(0, int(prefill_superchunk_stride))
        # SC SMALL-PREFILL threshold (_internal-orchestrator.
        # prefill_sc_min_tokens): a superchunk-path prefill whose DELTA
        # (tokens actually prefilled after any prefix-cache hit) is
        # STRICTLY BELOW this runs the ordinary chunked path at
        # _prefill_sc_small_chunk rows instead — a whole-delta MOE_BIG
        # union sweep on a small delta evicts the decode-warmed expert
        # cache for no amortization win (user report 2026-08-26).
        # 0 disables the downgrade.  Default 256: below one superchunk
        # stride, and small enough that per-64-row unions stay a fraction
        # of the routed set; the champion 8k/20k/25k prefill numbers
        # (glm_prefill.md) route identically (delta >= 256 on a cold
        # cache).  Adaptive successor: TD-SC-SMALL-PREFILL-ADAPTIVE.
        self._prefill_sc_min = max(0, int(prefill_sc_min_tokens))
        self._prefill_sc_small_chunk = max(1, int(prefill_sc_small_chunk))
        # TD-V4-SPEC-PREFILL-CTX RESOLVED (2026-08-22): the engine now
        # fires the V4 dflash draft's LAST aux tap (id == num_layers) at
        # the final layer's FETCH_AND_RUN finalize whenever the capture
        # epoch awaits only the final slot (maybe_dspark_capture_moe_final;
        # heads dedupe via the MoE-final mark) — so HEADLESS prefill
        # chunks/superchunks arm the draft context exactly like a
        # decode-step head would, and V4 boots pass True like GLM.  False
        # is retained as an escape hatch (mirrors an engine whose chunks
        # cannot arm the draft — plain arm, token-identical).
        self._chunk_prefill_arms_draft = bool(chunk_prefill_arms_draft)
        pc_cfg = prefix_cache or PrefixCacheConfig()
        # SUB-GRID mid-edge opt-in (USER DECISION 2026-08-28; default
        # OFF): effective value = config (_internal-prefix_cache.
        # subgrid_mid_edge) with env LS_ORCH_SUBGRID_MIDEDGE overriding
        # EITHER WAY when set (deterministic_ep_combine precedence).
        # Identity harnesses assert this attribute is False
        # (assert_identity_preconditions, dossier 4b precondition 3).
        subgrid = pc_cfg.subgrid_mid_edge
        env_sg = os.environ.get("LS_ORCH_SUBGRID_MIDEDGE")
        if env_sg:
            subgrid = env_sg[0] != "0"
        self.subgrid_mid_edge = bool(subgrid)
        if self.subgrid_mid_edge:
            # Same boot-notice pattern as deterministic_ep_combine: name
            # the consequence once, loudly, where the serve log starts.
            inert = ("" if metadata.seq_fork_truncatable else
                     " (INERT on this arch: forks are not truncatable, "
                     "INV-SEQ-FORK-TRUNC — the legacy filter stays)")
            print("  [orch] NOTICE: subgrid_mid_edge is ON -- mid-edge "
                  "prefix hits reuse the EXACT token LCP off the prefill "
                  "grid, so a hit is NOT token-identical to an uncached "
                  "run (deterministic shape-induced greedy divergence, "
                  "~1e-3 logit deltas, first divergence typically within "
                  "~15 decode tokens; INV-PREFIX-CACHE-1 SUB-GRID "
                  "clause). Identity/golden/A-B harnesses MUST NOT run "
                  "with this on" + inert, flush=True)
        self.prefix_cache: PrefixCache | None = (
            PrefixCache(pc_cfg, bridge) if pc_cfg.enabled else None)
        # P-29 step 24 (LS_KDA_PREFIX_CKPT, OQ-13): KDA prefix checkpoints.
        # Effective value = config (serving.prefix_cache.kda_checkpoints.
        # enabled) with the env overriding EITHER WAY when set. Meaningful
        # only where forks are NOT truncatable (the recurrent-state arch —
        # exactly where mid-edge reuse is otherwise impossible); the
        # capture cadence rounds UP to a multiple of 512 so every capture
        # lands between prefill commands, where the per-layer KDA frontier
        # is uniform (the engine refuses anything else, loudly).
        kckpt = pc_cfg.kda_ckpt_enabled
        env_kc = os.environ.get("LS_KDA_PREFIX_CKPT")
        if env_kc:
            kckpt = env_kc[0] != "0"
        iv = max(0, int(pc_cfg.kda_ckpt_interval_tokens))
        if iv % 512:
            iv = ((iv + 511) // 512) * 512
        # kda_state_slot_bytes > 0 is the KDA-arch witness (TD-GLM5-KDA-
        # SLOTS-EXPORT): V4 is also non-truncatable but has NO recurrent
        # state slot — captures there would only trip the engine refusal.
        self._kda_ckpt_on = (bool(kckpt) and iv > 0
                             and self.prefix_cache is not None
                             and not metadata.seq_fork_truncatable
                             and getattr(metadata, "kda_state_slot_bytes",
                                         0) > 0)
        self._kda_ckpt_iv = iv
        self._kda_ckpt_budget = max(0, int(pc_cfg.kda_ckpt_budget_bytes))
        # Capture TRIPWIRE (design §3.1): engine-side refusals of
        # D_CMD_KDA_CKPT mean the capture POINT is wrong (mid-sweep /
        # off-grid), not that refusals are acceptable — this counter must
        # read 0 in every run.
        self._kda_ckpt_refusals = 0
        # Host bytes of one checkpoint — seeded with the per-rank slot
        # size (a lower bound; TD-GLM5-KDA-SLOTS-EXPORT), refined to the
        # measured all-rank total by the first capture completion; used
        # for the pre-capture budget check.
        self._kda_ckpt_unit_bytes = int(
            getattr(metadata, "kda_state_slot_bytes", 0) or 0)
        if self._kda_ckpt_on:
            print(f"  [orch] KDA prefix checkpoints ON "
                  f"(interval {iv} tok, budget "
                  f"{self._kda_ckpt_budget >> 20} MiB host) — diverged "
                  f"prompts replay from the nearest checkpoint "
                  f"(bit-identical restore, P-29 step 24)", flush=True)
        # TD-PREFIX-TIDY-COLD-SPILL: resolved spill config (None =
        # disabled — flag off, no cache, or unwritable directory).
        self._spill: PrefixSpillConfig | None = None
        self._spill_dir: str = ""
        self._spill_last_sweep = 0.0
        if (prefix_spill is not None and prefix_spill.enabled
                and self.prefix_cache is not None):
            self._spill_dir = self._init_spill_dir(prefix_spill)
            if self._spill_dir:
                self._spill = prefix_spill
        if self.prefix_cache is not None:
            # R3: hand the SC lookup grid to the cache (classifies extended
            # non-grid exact-body nodes as pool-pressure retirable) and
            # state the holder-cost model once at boot (churn ticket
            # must-hold 4 — sizing must be honest).
            sc_stride0 = (min(self._prefill_sc_stride,
                              bridge.moe_batch_capacity)
                          if self._prefill_sc_stride > 0
                          else bridge.moe_batch_capacity)
            sc0 = self._prefill_superchunk and sc_stride0 > 0
            self.prefix_cache.sc_grid = sc_stride0 if sc0 else 0
            # GF3.2: attention_types codes — 0/1/2 are V4 tiers, 3/4 are
            # glm5_next (linear/sparse).  V4 is "populated and no glm5_next
            # code present"; glm5_next holders get their own cost line when
            # GF3.12 lands the state-copy holder model.
            att = tuple(getattr(metadata, "attention_types", ()) or ())
            g5n = any(t in (3, 4) for t in att)
            v4 = bool(att) and not g5n
            cost = ("copy-on-fork side-tier set per holder "
                    "(~45.3 MiB/holder/GPU at 32k, INV-PREFIX-CACHE-3)"
                    if v4 else
                    (("full per-holder KDA state copy (~146 MiB fp32/TP "
                      "GPU, INV-PREFIX-CACHE-3 third cost class) — "
                      "TRANSIENT in VRAM: hibernate spills it to "
                      "NUMA-local host RAM and returns the pool slot "
                      "(GF3.12; kill switch LS_KDA_HOLDER_SPILL=0)"
                      if pc_cfg.hibernate_holders else
                      "full per-holder KDA state copy (~146 MiB fp32/TP "
                      "GPU) HELD IN VRAM — hibernate_holders is OFF, so "
                      "every holder pins a KDA state-pool slot for its "
                      "whole life and the slot pool clamps concurrency "
                      "(INV-PREFIX-CACHE-3 third cost class)")
                     if g5n else
                     ("frozen fork (0 pages) + hibernated kMain "
                      "(~1 frontier logical group hot; prefix in the "
                      "tiering cold pool)"
                      if pc_cfg.hibernate_holders else
                      "frozen fork (0 pages) but NOT hibernated — each "
                      "holder pins ~the retention window of VRAM")))
            print(f"  [orch] prefix-cache sizing: max_entries="
                  f"{pc_cfg.max_entries}, budget "
                  f"{pc_cfg.max_cached_tokens} tok, holder cost: {cost}",
                  flush=True)
            if not v4 and not g5n and pc_cfg.hibernate_holders:
                # Must-hold 4 (churn ticket): hibernated holders retain
                # their prefixes in the KV-tiering COLD pool — the budget
                # above is also a cold-slot bound, and the pool should
                # cover it BESIDE max_concurrent live requests
                # (kv_tiering.host_to_device_ratio).  A full cold pool is
                # no longer a wedge (TD-KVT-COLD-FULL-HOT-WEDGE /
                # INV-KVT-18): the engine skips demotions in-step, drains
                # the backlog out-of-step on kMain pressure, and surfaces
                # retryable kKvPoolExhausted so THIS cache's eviction seam
                # frees slots — an undersized ratio costs holder churn +
                # stalls, not failures.
                print(f"  [orch] prefix-cache sizing: hibernated-holder "
                      f"cold usage is bounded by the "
                      f"{pc_cfg.max_cached_tokens}-token budget; size "
                      f"kv_tiering.host_to_device_ratio for it beside "
                      f"live-request demotion (cold-full = in-step skip; "
                      f"recovered by the engine pressure sweep + holder "
                      f"eviction)", flush=True)
        # TD-GLM5-KDA-SLOTS-EXPORT: state the state-pool concurrency
        # surface once at boot (independent of the prefix cache) — the
        # state pool is the resource that actually clamps concurrency on
        # this architecture, and an oversubscribed box otherwise degrades
        # into evict-retry churn with no number anyone can size against.
        if metadata.kda_state_slot_bytes > 0:
            mib = metadata.kda_state_slot_bytes / (1 << 20)
            if metadata.kda_state_mapped:
                pool = metadata.kda_state_pool_pages
                per = max(1, metadata.kda_state_pages_per_seq)
                print(
                    f"  [orch] KDA state pool: MAPPED over the shared "
                    f"kMain pool ({pool} pages); one admission claims "
                    f"{per} pages of state ({mib:.1f} MiB/rank) beside "
                    f"its KV+indexer pages — state-only ceiling "
                    f"{pool // per} concurrent sequences; live pressure = "
                    f"kv_main_free_pages vs the per-seq demand "
                    f"(exhaustion is the retryable kKvPoolExhausted "
                    f"evict-retry seam)", flush=True)
            else:
                print(
                    f"  [orch] KDA state pool: DEDICATED CARVE — "
                    f"{metadata.kda_state_slots} whole-request slots x "
                    f"{mib:.1f} MiB; the HARD concurrency cap = in-flight "
                    f"requests + non-hibernated holders (hibernated "
                    f"holders spill and return their slot, INV-KDA-STATE "
                    f"(g))", flush=True)
        self._engine = engine_module          # owned iff boot() created it
        self._queue: collections.deque[InferenceRequest] = collections.deque()
        self._queue_lock = threading.Lock()
        self._wake = threading.Event()
        self._cancelled: set[int] = set()
        self._shutdown = False
        self._next_seq_id = 1
        # TD-INDEXER-POOL-EVICT: prefill steps answered by evicting a
        # prefix holder and re-issuing (pool pressure, not an error).
        self._pool_evict_retries = 0
        # TD-SPEC-ROUND-POOL-EVICT observability: evict-retries taken
        # on MID-speculative-round steps (overlap/verify/bonus-refeed)
        # — counted apart from the prefill/plain/seed retries above so
        # a box that constantly recovers mid-round is visible.
        self._spec_round_pool_evict_retries = 0
        # ── TD-KVXP-PER-STEP-FLOOR: bounded LARGE-PREFILL wait ─────────
        # Under the 44z per-step reserve the rare BULK upfront claim (a
        # large prompt's admission KV/state/indexer) is deliberately not
        # covered by a standing reserve: when every evictable holder is
        # gone and the pool refusal is the retryable exhaustion class, a
        # LARGE prefill may wait — bounded — for the engine's eager
        # background drain (armed by the very refusal, 44z
        # note_pool_pressure_refusal; measured ~400 ms) and re-issue.
        # Short prefills and decode steps NEVER wait (user requirement):
        # the wait engages only while the request's prefill phase is
        # active AND len(prompt) >= the threshold. Each engine-side claim
        # stays all-or-nothing with full rollback — the wait sits ABOVE
        # the admission protocol, so INV-KDA-STATE (a) is untouched.
        # Gated on the 44z rebalancer switch itself: without the rebalancer
        # no drain is coming and the wait would only delay the 500.
        # Schema-backed knobs (_internal-orchestrator.kvxp_large_prefill_*,
        # TD-KVXP-SCHEMA-KNOBS resolved) with the env vars overriding
        # EITHER WAY when set (kda_state.mapped precedence):
        #   LS_KVXP_LARGE_PREFILL_TOKENS  threshold, default 8192
        #     (derived: drains are tick-bound 200-600 ms vs measured
        #     prefill rates 88.8-140 tok/s — a >= ~6,600-token prompt
        #     keeps one drain under ~1% of TTFT; 0 disables the wait)
        #   LS_KVXP_LARGE_PREFILL_WAIT_MS total budget/request, default
        #     2000 (>= 3 drain periods; the measured miss was ~200 ms —
        #     the budget is a cap on futile waiting, the typical cost is
        #     one drain)
        def _env_int(name: str, dflt: int) -> int:
            try:
                return int(os.environ.get(name, "") or dflt)
            except ValueError:
                print(f"  [orch] {name} is not an integer — keeping "
                      f"{dflt}", flush=True)
                return dflt
        self._kvxp_wait_threshold = max(
            0, _env_int("LS_KVXP_LARGE_PREFILL_TOKENS",
                        int(kvxp_large_prefill_tokens)))
        self._kvxp_wait_budget_ms = max(
            0, _env_int("LS_KVXP_LARGE_PREFILL_WAIT_MS",
                        int(kvxp_large_prefill_wait_ms)))
        # Master switch mirror: config (_internal-kv_expert_rebalance.
        # enabled, via from_config) with LS_KV_EXPERT_REBALANCE overriding
        # EITHER WAY when SET ('1' enables, anything else disables) — the
        # same precedence the engine's from_config_env applies, so the
        # orchestrator's wait and the engine's rebalancer cannot disagree.
        _xz_env = os.environ.get("LS_KV_EXPERT_REBALANCE")
        _xz_on = (_xz_env == "1") if _xz_env is not None \
            else bool(kv_expert_rebalance_enabled)
        self._kvxp_wait_enabled = (
            _xz_on
            and self._kvxp_wait_threshold > 0
            and self._kvxp_wait_budget_ms > 0)
        self._kvxp_wait_poll_s = 0.1
        # Per-request wait state: None = not eligible / prefill done;
        # otherwise the monotonic deadline (set on first engagement).
        self._kvxp_wait_deadline: float | None = None
        self._kvxp_wait_armed = False
        self._kvxp_wait_used = False
        self._kvxp_wait_req = 0
        # Boot-lifetime observability.
        self._kvxp_wait_engagements = 0   # sleeps taken
        self._kvxp_wait_rescues = 0       # requests saved by a wait
        # TD-V4-FIRSTREQ-COLD-SHARE-OVER-CAPACITY option (d): serving-level
        # retry of a request whose completion carries degraded MoE layers
        # (incomplete expert set -- a known-wrong answer).  Bounded per
        # request by degraded_retry_max (0 = OFF; production boots default
        # to 1 via _internal-orchestrator.degraded_retry_{enabled,max}).
        # Direct-constructed (test) orchestrators default OFF so scripted
        # degraded-finalize tests observe the raw completion.
        self._degraded_retry_max = max(0, int(degraded_retry_max))
        # Boot-lifetime observability: a retry loop that HIDES a real
        # capacity problem is worse than the bug, so every action is
        # counted -- retries performed, degraded results delivered anyway
        # (retries off/exhausted/unretryable), and degraded streaming
        # completions that could not be retried because tokens had
        # already left the server.
        self.degraded_retries_total = 0
        self.degraded_delivered_total = 0
        self.degraded_unretryable_total = 0
        # DSpark drafting-context mirror (TD-PREFIX-DSPARK-FORK-CTX): the
        # engine's DsparkRuntime tracks ONE ingested drafting context; a
        # full prefill re-arms it at position 0, and a prefix-cache-forked
        # sequence (which never feeds position 0) can only ADOPT it when
        # its fork point lands inside the tracked frontier — otherwise the
        # runtime invalidates fail-closed and every dspark run_step for
        # that sequence CMP_ERRORs, killing the request.  B=1 means this
        # orchestrator drives every capture, so adoption eligibility is
        # mirrored here and non-adoptable forked requests take the PLAIN
        # greedy path (token-identical — INV-DSPARK-LOSSLESS: drafts only
        # ever change acceptance).  _draft_ctx_len is the conservative
        # (prefill-end) frontier; decode extends it engine-side, so
        # understating only costs speculation on some adoptable forks,
        # never a false speculative arm.
        self._draft_ctx_valid = False
        self._draft_ctx_len = 0
        self.last_stats: RequestStats | None = None
        self.info: Any = None                 # EngineInfo when boot()ed

        if self.spec.enabled:
            r = self.spec.gamma + 1
            if r > bridge.moe_batch_capacity:
                raise BridgeError(
                    f"gamma {self.spec.gamma} needs {r} verify rows > "
                    f"engine MoE batch capacity {bridge.moe_batch_capacity}")

    # ── boot ─────────────────────────────────────────────────────────────

    @classmethod
    def boot(cls, config_path: str, *, engine_module: Any = None,
             eos_token_ids: tuple[int, ...] = (),
             conf_thresh: float = 0.1,
             speculation_depth: int | None = None,
             test_engine: bool = False) -> "Orchestrator":
        """Start the engine from a config json and wire the bridge.

        ``conf_thresh`` arms DSP-9 truncation when the config enables
        dspark confidence (champion value 0.1). ``eos_token_ids`` come
        from the tokenizer (the serve layer owns vocabulary knowledge).
        ``speculation_depth``: None = config-derived gamma; 0 = force
        plain decode; >0 = clamp gamma.  Placement arm: the REEF service
        needs the engine's gpu_loader calibration — configs without
        ``gpu_loader.enabled`` (and null-backend test engines) fall back
        to the static ACT arm.

        ONE FLOW (INV-ORCH-ONE-FLOW, 2026-08-25): every architecture
        boots the SAME champion flow — REEF routing (E_CMD_REEF_ROUTE +
        epoch-latched banks), fused FAR + pipelined burst, superchunk
        prefill, the spec arms, prefix cache.  Where behavior must
        genuinely differ it is driven by CONFIG (schema fields the
        recipes carry: prefill_chunk_tokens,
        decode_expert_fetch_timeout_s, gpu roles, gpu_loader) or by
        ENGINE-REPORTED capability (EngineInfo.moe_batch_capacity,
        vocab_size, attention_types) — never by model-name checks
        (the orchestrator mirror of INV-ATTN-ARCH / INV-MOE-ARCH).
        Diagnostic kill switch: LS_ORCH_FORCE_SPLIT_ACT=1 restores the
        legacy split/ACT command arm (no FAR, no REEF, static e%N
        placement)."""
        import json

        if engine_module is None:
            import layerstorm_engine as engine_module
        with open(config_path) as f:
            cfg = json.load(f)

        # TD-ORCH-EP-GPU-INDICES-DEAD: this orchestrator derives the
        # expert-owner set from hardware.gpus roles and the daemon's REEF
        # placement (route_moe_by_loader) picks each expert's owner — a
        # config-supplied _internal-orchestrator.ep_gpu_indices is NOT
        # consulted here.  It used to be silently discarded, which is how
        # it masqueraded as a live bisect lever during the EP4
        # investigation.  Refuse LOUDLY before the engine starts rather
        # than pretend to honor it.  (The field stays live for the legacy
        # OrchestratorLoop drivers — tools/elb_train — via
        # engine_glue.ep_gpu_indices_from_config.)
        _ep_idx = (cfg.get("_internal-orchestrator") or {}) \
            .get("ep_gpu_indices")
        if _ep_idx:
            raise ValueError(
                "_internal-orchestrator.ep_gpu_indices is set "
                f"({list(_ep_idx)}) but the production orchestrator does "
                "not honor it: expert-hosting GPUs come from hardware.gpus "
                "roles (expert_streaming/resident prefix) and the daemon's "
                "REEF placement picks per-expert owners. Remove the field "
                "or express the owner set via hardware.gpus roles "
                "(TD-ORCH-EP-GPU-INDICES-DEAD).")

        info = (engine_module.start_engine_test(config_path) if test_engine
                else engine_module.start_engine(config_path))
        try:
            model = cfg.get("model") or {}
            # TD-VOCAB-AUTODETECT: prefer the engine's resolved vocab width
            # (EngineInfo.vocab_size — weights-derived when the config field
            # is 0/absent, cross-checked otherwise). The config dict read is
            # the fallback for engine builds predating the field.
            vocab = int(getattr(info, "vocab_size", 0) or 0) \
                or int(model.get("vocab_size", 0))
            # Dense-prefix depth is the config field the DAEMON itself
            # reads for the FAR dense/moe split (far_forward_layer's
            # is_moe test) — config is the single source of truth on both
            # sides.  All-MoE architectures (V4-Flash) carry 0 in their
            # recipes: the bridge's dense branch never fires.
            first_moe = int(model.get("first_k_dense_replace", 3))

            buffer_ids = engine_module.query_buffer_ids()
            hidden_buf = logits_buf = 0
            for name, bid in buffer_ids.items():
                if name.startswith("hidden_state.attn.rank0"):
                    hidden_buf = int(bid)
                elif name.startswith("logits_scratch.pos0"):
                    logits_buf = int(bid)
            if not (hidden_buf and logits_buf) and not test_engine:
                raise BridgeError("hidden/logits buffers not found")

            # Route arm: REEF whenever the config arms the calibrated
            # gpu_loader service (the engine self-calibrates FULL and
            # writes the weights-adjacent calibration file on the first
            # absent run — engine.cpp kLoaded fallback); gpu_loader-less
            # configs and null-backend test engines fall back to the
            # static ACT arm.  LS_ORCH_FORCE_SPLIT_ACT=1 (diagnostic)
            # restores the legacy split/ACT arm end-to-end (no FAR).
            force_split = \
                os.environ.get("LS_ORCH_FORCE_SPLIT_ACT") == "1"
            reef_ok = bool((cfg.get("gpu_loader") or {}).get("enabled")) \
                and not test_engine and not force_split
            bridge = EngineBridge(info, vocab_size=vocab,
                                  first_moe_layer=first_moe,
                                  hidden_buf_id=hidden_buf,
                                  logits_buf_id=logits_buf,
                                  route_arm="reef" if reef_ok else "act",
                                  use_far=not force_split)
            orch_cfg = cfg.get("orchestrator") or {}
            # Daemon-side expert-fetch deadline: config
            # (orchestrator.decode_expert_fetch_timeout_s, schema default
            # 5 s = the historical champion value).  Streaming-wall
            # recipes (V4 routed experts from the GGUF page cache — a
            # cold layer can exceed 5 s) carry 200, the golden-harness
            # precedent.
            bridge.decode_timeout_us = max(1, int(
                orch_cfg.get("decode_expert_fetch_timeout_s")
                or 5)) * 1_000_000
            # Expert-hosting GPUs = the expert-role PREFIX of
            # hardware.gpus (schema default: a role-less GPU entry has
            # ALL roles).  A GPU whose declared roles carry neither
            # `resident` nor `expert_streaming` (e.g. a dedicated dspark
            # draft host) stops the scan — the static e%N placement
            # addresses gpu indices 0..N-1, so later non-expert gpus get
            # no expert work (correct, narrower EP; the ticket-J
            # draft-host pin, now carried BY CONFIG roles).  Configs
            # without a hardware section keep every engine GPU (the
            # champion default).  The daemon's FAR act placement mirrors
            # this rule exactly (dispatch_reef.cpp).
            gpus_cfg = (cfg.get("hardware") or {}).get("gpus") or []
            if gpus_cfg:
                expert_roles = {"expert_streaming", "resident"}
                default_roles = ("attention", "resident",
                                 "expert_streaming")
                n_moe = 0
                for g in gpus_cfg:
                    if not expert_roles & set(g.get("roles")
                                              or default_roles):
                        break
                    n_moe += 1
                bridge.moe_gpus = max(1, n_moe)
            # Guided-decoding logits readback row (0 when the engine build
            # predates it or vocab is unknown).
            if hasattr(engine_module, "logits_readback_addr"):
                addr, nbytes = engine_module.logits_readback_addr()
                if int(addr) and int(nbytes) >= vocab * 4:
                    bridge.logits_host_addr = int(addr)
                    # Multi-row region (kMaxLogitsReadbackRows) → the
                    # sampled/logprobs speculative arms; 1 on old builds.
                    bridge.logits_host_rows = max(
                        1, int(nbytes) // (vocab * 4))

            spec_cfg = cfg.get("speculation") or {}
            spec = SpeculationConfig()
            # Ticket J: dspark speculation is arch-agnostic — for V4 the
            # engine runs the dflash draft on a second GPU, verify chunks
            # ride the V4 micro-chunk arm (executor per-row loop + rewind
            # snapshots), and the aux capture uses the stream-MEAN
            # representation. Configs without a dspark method keep spec OFF.
            # P-29 step 13 phase B: in-model MTP speculation (glm5_next).
            # gamma <= 2: depth-1 drafts from the target hidden, depth-2
            # from the recycled MTP hidden; deeper chains unwired.
            _arch_name = (cfg.get("model") or {}).get("architecture") or ""
            if (spec_cfg.get("enabled")
                    and spec_cfg.get("method") == "mtp"
                    and (spec_cfg.get("mtp") or {}).get("enabled")
                    and _arch_name == "glm5_next"
                    and speculation_depth != 0):
                _md = int((spec_cfg.get("mtp") or {}).get("max_depth", 2))
                _g = min(2, max(1, _md))
                if speculation_depth is not None:
                    _g = min(_g, max(1, speculation_depth))
                spec = SpeculationConfig(
                    enabled=True, gamma=_g, method="mtp",
                    index_kpool=max(1, int((cfg.get("model") or {})
                                           .get("index_kpool", 1) or 1)))
            elif (spec_cfg.get("enabled")
                    and spec_cfg.get("method") == "dspark"
                    and speculation_depth != 0):
                ds = spec_cfg.get("dspark") or {}
                gamma = int(ds.get("speculative_tokens",
                                   max(1, int(ds.get("block_size", 8)) - 1)))
                if speculation_depth is not None:
                    gamma = min(gamma, speculation_depth)
                # TD-DSPARK-CTX-POLICY: mirror the engine's rotation flag
                # (schema field speculation.dspark.ctx_rotate, default ON;
                # env LS_DSPARK_CTX_ROTATE overrides EITHER WAY when set —
                # '0' disables, anything else enables — matching the
                # engine's DsparkRuntime::create precedence).  The V4
                # dflash draft keeps the legacy cap engine-side, so its
                # mirror must too.
                arch = (cfg.get("model") or {}).get("architecture") or ""
                _rot_env = os.environ.get("LS_DSPARK_CTX_ROTATE")
                _rot = bool(ds.get("ctx_rotate", True)) \
                    if _rot_env is None else (_rot_env != "0")
                spec = SpeculationConfig(
                    enabled=gamma >= 1, gamma=gamma,
                    conf_thresh=(conf_thresh
                                 if ds.get("confidence_enabled") else 0.0),
                    ctx_cap_tokens=int(
                        ds.get("draft_context_capacity_tokens", 8192)),
                    ctx_rotate=(_rot and arch != "deepseek_v4"))

            meta = EngineMetadata(
                num_gpus=info.num_gpus,
                num_moe_layers=info.num_moe_layers,
                num_experts=info.num_experts,
                num_layers=info.num_layers,
                expert_bytes=getattr(info, "expert_bytes", 0),
                kv_bytes_per_page=getattr(info, "kv_bytes_per_page", 0),
                # P-29 step 13 / TD-MTP-PROBE-DEFERRED-CONSUMERS: the dense
                # prefix depth comes from CONFIG (same `first_moe` the
                # bridge's dense/MoE split uses), never from
                # `num_layers - num_moe_layers` — the MTP-armed expert
                # census counts the NextN block, so the subtraction
                # would put the boundary one layer low.
                first_moe_layer=first_moe,
                eos_token_ids=tuple(eos_token_ids),
                hidden_buf_id=hidden_buf,
                logits_buf_id=logits_buf,
                vocab_size=vocab,
                moe_batch_capacity=info.moe_batch_capacity,
                attention_types=tuple(
                    getattr(info, "attention_types", ()) or ()),
                # R4b: arch capability, property-derived engine-side
                # (AttentionArch::lossy_position_indexed_state).
                # LS_ORCH_NO_MIDEDGE=1 is the ops kill switch back to
                # grid/exact-node hits (mirrors LS_ORCH_NO_SC).
                seq_fork_truncatable=(
                    bool(getattr(info, "seq_fork_truncatable", 0))
                    and os.environ.get("LS_ORCH_NO_MIDEDGE") != "1"),
                # TD-GLM5-KDA-SLOTS-EXPORT: state-pool geometry — the
                # admission-pressure surface (boot sizing print below;
                # live pressure = kv_main_free_pages vs pages_per_seq).
                kda_state_mapped=bool(getattr(info, "kda_state_mapped", 0)),
                kda_state_slots=int(getattr(info, "kda_state_slots", 0)),
                kda_state_slot_bytes=int(
                    getattr(info, "kda_state_slot_bytes", 0)),
                kda_state_pages_per_seq=int(
                    getattr(info, "kda_state_pages_per_seq", 0)),
                kda_state_pool_pages=int(
                    getattr(info, "kda_state_pool_pages", 0)),
            )
            pc_cfg = PrefixCacheConfig.from_config(cfg)
            spill_cfg = PrefixSpillConfig.from_config(cfg)
            # TD-V4-SERVE-PREFIX RESOLVED (2026-08-22): CMD_SEQ_FORK
            # clones complete per-seq state for every arch (kMain CoW +
            # side-tier copy-on-fork + executor state rings), so the
            # prefix cache is config-only.  On the superchunk path the
            # registration grid is the SUPERCHUNK stride — see _generate.
            moe_big = bool((cfg.get("compute") or {})
                           .get("prefill_moe_big", True))
            # Served-prefill levers (2026-08-23 serving-prefill campaign):
            # (1) GLM prefill CHUNK SIZE — the historical 64 streams every
            #     layer's routed union once per 64 tokens; bigger FAR
            #     chunks amortize the expert H2D per union (measured A/B in
            #     spec/measurements/glm_prefill.md).  LS_ORCH_PREFILL_CHUNK
            #     overrides (<= 512 = kMaxBatchDescriptors).
            # (2) GLM MINI-SUPERCHUNKS (single-shot MOE_BIG): superchunks
            #     BOUNDED at the engine's single-shot MoE chunk capacity —
            #     max(compute.moe_big_chunk_tokens [default 512],
            #     orchestrator.max_batch_size) — with 64-row attention
            #     sub-chunks: ONE REEF-routed MOE_BIG per layer per
            #     stride instead of one FETCH_AND_RUN per layer per
            #     64-row chunk (~8x fewer expert-union streams).  The
            #     stride bound is CORRECTNESS-CRITICAL on the EP4 champion
            #     topology: MoE batches ABOVE the capacity run the CHUNKED
            #     grouped-GEMM path, which is REJECTED with expert-only
            #     ranks resident (TD-MOE-EP-XTP-WAVES, "EP-beyond-TP is
            #     single-shot only") and the layer output is WRONG
            #     (measured: decode acceptance 0.0000, 150 dispatch
            #     errors at stride 3069).  At or below it, MOE_BIG is the
            #     byte-identical single-shot pipeline (INV-MOE-BIG-1).
            #     LS_ORCH_NO_SC=1 restores the chunked 64-token path.
            #     TP-only topologies (no expert-only ranks) keep the full
            #     elastic-capacity stride (0 = moe_batch_capacity).
            #     GREEN-LIT DEFAULT ON (user verdict 2026-08-23; was
            #     opt-in): the stride-512 served-prefill trajectory
            #     change (bf16 grouped-GEMM batch-width shape class —
            #     stride-64 probe token-identical, both trajectories
            #     valid greedy continuations) is ACCEPTED for the
            #     2.1-2.3x prefill win.  TD-SERVE-SC-TRAJECTORY records
            #     the situation; LS_ORCH_NO_SC=1 is the kill switch back
            #     to the chunk-64 path (and its pre-flip trajectory).
            sc_arm = moe_big and os.environ.get("LS_ORCH_NO_SC") != "1"
            comp = cfg.get("compute") or {}
            # Superchunk STRIDE — ONE topology rule (arch-free, the
            # TD-V4-TP-DSPARK predicate generalized): the full elastic
            # stride (0 = moe_batch_capacity) is valid ONLY while every
            # expert host is a TP rank.  With expert-only ranks resident
            # (bridge.moe_gpus beyond the tp_array — the ep4 champion
            # topologies), MoE batches above the single-shot capacity run
            # the CHUNKED grouped-GEMM path, which is REJECTED with
            # extra-rank residents (TD-MOE-EP-XTP-WAVES) — clamp to the
            # single-shot bound.
            ep_extra = bridge.moe_gpus > len(
                (cfg.get("hardware") or {}).get("tp_array") or [0])
            sc_stride = max(
                int(comp.get("moe_big_chunk_tokens") or 512),
                int(orch_cfg.get("max_batch_size") or 64)) \
                if ep_extra else 0
            # P-30 step 1: the config value is a REQUEST — the engine's
            # elastic chunk fail-safe may have stepped the realized
            # single-shot bound down (EngineInfo.moe_chunk_capacity). A
            # stride above the realized bound would drive the CHUNKED
            # grouped-GEMM path, which is REJECTED with expert-only ranks
            # resident (TD-MOE-EP-XTP-WAVES) — clamp to the engine truth.
            eng_chunk = int(getattr(bridge, "moe_chunk_capacity", 0) or 0)
            if ep_extra and eng_chunk > 0:
                sc_stride = min(sc_stride, eng_chunk)
            se = os.environ.get("LS_ORCH_SC_STRIDE")  # diagnostic override
            if se and se.isdigit() and int(se) > 0:
                sc_stride = int(se)
            # Prefill chunk rows: config (orchestrator.
            # prefill_chunk_tokens, schema default 64 — the GLM
            # test-proven stride; expert-union-saturating recipes carry
            # 512), clamped to the engine-reported MoE batch capacity
            # (covers prefill_moe_big-off builds whose capacity shrinks).
            chunk = max(1, min(
                int(orch_cfg.get("prefill_chunk_tokens") or 64), 512,
                int(getattr(info, "moe_batch_capacity", 0) or 512)))
            ce = os.environ.get("LS_ORCH_PREFILL_CHUNK")
            if ce and ce.isdigit() and int(ce) > 0:
                chunk = min(int(ce), 512)
            # SC small-prefill downgrade knobs (_internal-orchestrator):
            # threshold (delta tokens; 0 = off) + the small chunk size the
            # downgraded delta runs at (clamped like `chunk` above).
            iorch = cfg.get("_internal-orchestrator") or {}
            sc_min = max(0, int(iorch.get("prefill_sc_min_tokens", 256)))
            sc_small_chunk = max(1, min(
                int(iorch.get("prefill_sc_small_chunk_tokens", 64)), 512,
                int(getattr(info, "moe_batch_capacity", 0) or 512)))
            # Serving-level degraded-request retry
            # (TD-V4-FIRSTREQ-COLD-SHARE-OVER-CAPACITY option (d),
            # _internal-orchestrator): default ON with ONE retry -- the V4
            # first-request degrade self-corrects once residency exists,
            # so a single re-run lands clean (measured 2026-08-27).
            deg_retry = (max(0, int(iorch.get("degraded_retry_max", 1)))
                         if bool(iorch.get("degraded_retry_enabled", True))
                         else 0)
            # 44z bounded large-prefill wait knobs (_internal-orchestrator,
            # TD-KVXP-SCHEMA-KNOBS) + the rebalancer master switch mirror
            # (_internal-kv_expert_rebalance.enabled).  Env overrides for
            # all three are applied in __init__, either way.
            kvxp_tokens = max(
                0, int(iorch.get("kvxp_large_prefill_tokens", 8192)))
            kvxp_wait_ms = max(
                0, int(iorch.get("kvxp_large_prefill_wait_ms", 2000)))
            ikvxp = cfg.get("_internal-kv_expert_rebalance") or {}
            kvxp_enabled = bool(ikvxp.get("enabled", False))
            orch = cls(bridge, metadata=meta, speculation=spec,
                       prefix_cache=pc_cfg,
                       prefix_spill=spill_cfg,
                       engine_module=engine_module,
                       prefill_chunk=chunk,
                       chunk_prefill_arms_draft=True,
                       prefill_superchunk=sc_arm,
                       prefill_superchunk_stride=sc_stride,
                       prefill_sc_min_tokens=sc_min,
                       prefill_sc_small_chunk=sc_small_chunk,
                       degraded_retry_max=deg_retry)
            orch.info = info                  # EngineInfo (reporting)
            # TD-SERVE-PREFILL-NONDET-RUN-TO-RUN: resolve the effective
            # routed-EP combine determinism (config, env overrides EITHER
            # way -- the command_dispatcher ctor precedence) and say so
            # once at boot: with the legacy mode-0 combine the greedy
            # output of this boot is NOT reproducible run-to-run, and no
            # token-identity gate may compare two requests served by it.
            det = bool(comp.get("deterministic_ep_combine", False))
            env_det = os.environ.get("LAYERSTORM_DETERMINISTIC_EP_COMBINE")
            if env_det:
                det = env_det[0] != "0"
            orch.deterministic_ep_combine = det
            if not det:
                print("  [orch] NOTICE: deterministic_ep_combine is OFF -- "
                      "the routed EP combine follows live expert residency, "
                      "so greedy output is NOT reproducible run-to-run "
                      "(TD-SERVE-PREFILL-NONDET-RUN-TO-RUN); token-identity "
                      "gates must boot with it ON", flush=True)
            return orch
        except Exception:
            engine_module.stop_engine()
            raise

    # ── submission surface (thread-safe; old-loop contract) ──────────────

    def submit_request(self, req: InferenceRequest) -> None:
        with self._queue_lock:
            self._queue.append(req)
        self._wake.set()

    def cancel_request(self, request_id: int) -> None:
        self._cancelled.add(request_id)
        self._wake.set()

    def shutdown(self) -> None:
        self._shutdown = True
        self._wake.set()

    def stop_engine(self) -> None:
        """Stop the engine iff boot() created it."""
        if self._engine is not None:
            self._engine.stop_engine()
            self._engine = None

    # ── loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Serve queued requests serially until shutdown()."""
        while not self._shutdown:
            if not self._serve_next():
                # TD-PREFIX-TIDY-COLD-SPILL: idle-time only, by
                # construction — the sweep runs strictly between requests
                # (never on the request path) and does at most one spill
                # per pass.
                self._spill_sweep()
                self._wake.wait(timeout=0.05)
                self._wake.clear()

    # ── TD-PREFIX-TIDY-COLD-SPILL: idle holder spill (2nd tiering hop) ──

    def _init_spill_dir(self, cfg: "PrefixSpillConfig") -> str:
        """Expand/create/probe the spill directory and reclaim stale
        files.  Returns "" (feature disabled, loudly) when unwritable —
        requests are never failed over spill I/O."""
        path = os.path.expanduser(cfg.path)
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".ls-spill-probe")
            with open(probe, "w"):
                pass
            os.unlink(probe)
        except OSError as err:
            print(f"  [orch] WARNING: prefix-spill dir '{path}' is not "
                  f"writable ({err}) — holder cold-spill DISABLED for "
                  f"this boot (TD-PREFIX-TIDY-COLD-SPILL)", flush=True)
            return ""
        # Stale-file reclamation: the directory is a CACHE, never a
        # store — a spill file is only meaningful to the engine boot that
        # wrote it (boot-nonced names), so EVERYTHING from earlier boots
        # is dead weight on the user's disk.
        reclaimed = 0
        try:
            for name in os.listdir(path):
                if name.startswith("ls-spill-") \
                        and name.endswith(".kvspill"):
                    os.unlink(os.path.join(path, name))
                    reclaimed += 1
        except OSError:
            pass
        print(f"  [orch] prefix-spill ON — dir '{path}', cap "
              f"{cfg.max_mib} MiB, idle {cfg.idle_seconds:.0f} s"
              + (f"; reclaimed {reclaimed} stale file(s)" if reclaimed
                 else ""), flush=True)
        return path

    def _spill_sweep(self) -> None:
        sp = self._spill
        pc = self.prefix_cache
        if sp is None or pc is None:
            return
        if not hasattr(self.bridge, "spill_sequence"):
            return  # test bridges without the seam
        now = time.monotonic()
        if now - self._spill_last_sweep < 5.0:
            return
        self._spill_last_sweep = now
        with self._queue_lock:
            if self._queue:
                return  # background only — a request just arrived
        # LEAF holders only: an interior node's cold slots are refcount-
        # shared with its descendants (children inherit them at fork), so
        # spilling one frees no pinned RAM — and dropping it later would
        # need reparenting.  Age + hit history, never size: idle >=
        # idle_seconds since the last hit, LRU-first.
        cands = [e for e in pc._entries
                 if not e.children and not e.spilled
                 and not e.spill_attempted
                 and now - e.wall_touch >= sp.idle_seconds]
        if not cands:
            return
        e = min(cands, key=lambda x: (x.last_used, -len(x.tokens)))
        self._spill_holder(e, retry_on_cap=True)

    def _spill_holder(self, e, retry_on_cap: bool) -> None:
        pc = self.prefix_cache
        assert pc is not None
        try:
            status, pages = self.bridge.spill_sequence(
                e.seq_id, len(e.tokens))
        except BridgeError as err:
            e.spill_attempted = True
            print(f"  [orch] prefix-spill of holder seq {e.seq_id} "
                  f"failed ({err}) — holder stays resident", flush=True)
            return
        if status == 2:
            # Byte-cap refusal: the eviction policy over the spill
            # directory itself — drop the LRU SPILLED leaf (freeing its
            # holder deletes its file engine-side) and retry ONCE.
            victims = [x for x in pc._entries
                       if x.spilled and not x.children and x is not e]
            if retry_on_cap and victims:
                v = min(victims, key=lambda x: x.last_used)
                print(f"  [orch] prefix-spill cap reached — evicting LRU "
                      f"spilled holder seq {v.seq_id} "
                      f"({len(v.tokens)} tok) and retrying", flush=True)
                pc.drop_leaf(v)
                self._spill_holder(e, retry_on_cap=False)
            else:
                e.spill_attempted = True
            return
        if pages > 0:
            e.spilled = True
            print(f"  [orch] prefix-spill: holder seq {e.seq_id} "
                  f"({len(e.tokens)} tok) spilled {pages} cold pages to "
                  f"disk", flush=True)
        else:
            # Nothing cold to spill (V4 arm, hibernation off, or an
            # all-shared holder) — do not re-issue every sweep.
            e.spill_attempted = True

    def _serve_next(self) -> bool:
        """Serve one queued request; False when the queue is empty."""
        with self._queue_lock:
            req = self._queue.popleft() if self._queue else None
        if req is None:
            return False
        if req.request_id in self._cancelled:
            print(f"  [orch] request {req.request_id} cancelled while "
                  f"queued", flush=True)
            self._finish(req, [], "cancelled")
            return True
        retries = 0
        try:
            while True:
                try:
                    tokens, reason, stats, logprobs = self._generate(
                        req, degraded_retry_ok=(
                            retries < self._degraded_retry_max))
                except _DegradedRetry as dr:
                    # PREFILL degraded with budget left: re-run before any
                    # token was emitted (streaming-safe by construction).
                    # The abandoned attempt's sequence is freed and it
                    # registered no prefix holder; the retry lands on the
                    # residency the failed attempt established -- measured
                    # clean (TD-V4-FIRSTREQ-COLD-SHARE-OVER-CAPACITY).
                    retries += 1
                    self.degraded_retries_total += 1
                    print(f"  [orch] request {req.request_id}: {dr.layers} "
                          f"MoE layer(s) degraded during prefill -- "
                          f"retrying before first token (retry {retries}/"
                          f"{self._degraded_retry_max}; "
                          f"{self.degraded_retries_total} degraded "
                          f"retries this boot)", flush=True)
                    continue
                if (stats.moe_degraded_layers
                        and retries < self._degraded_retry_max):
                    streamed = req.on_token is not None and stats.tokens > 0
                    if streamed or req.guided is not None:
                        # Unretryable decode-time degradation.  STREAMED:
                        # the tokens already left the server, so a silent
                        # re-run would emit a second stream.  GUIDED: the
                        # grammar state is SINGLE-USE (InferenceRequest.
                        # guided) and was consumed by this attempt's
                        # decode -- a re-run would re-drive it from a
                        # mid-request state.  (Guided requests still get
                        # the PREFILL-degraded retry above: the grammar is
                        # untouched until decode.)  Honest behaviour:
                        # deliver the flagged result (the WARNING below +
                        # per-request stats carry the degradation) and
                        # count the miss.
                        self.degraded_unretryable_total += 1
                        why = ("tokens were streamed" if streamed
                               else "grammar state is single-use")
                        print(f"  [orch] request {req.request_id}: MoE "
                              f"layers degraded during decode but "
                              f"{why} -- NOT retrying "
                              f"({self.degraded_unretryable_total} "
                              f"unretryable this boot)", flush=True)
                    else:
                        # Degraded during decode but nothing has reached
                        # the client (non-streaming, delivery happens in
                        # _finish below): re-run the whole request.
                        retries += 1
                        self.degraded_retries_total += 1
                        print(f"  [orch] request {req.request_id}: "
                              f"{stats.moe_degraded_layers} MoE layer(s) "
                              f"degraded -- retrying (retry {retries}/"
                              f"{self._degraded_retry_max}; "
                              f"{self.degraded_retries_total} degraded "
                              f"retries this boot)", flush=True)
                        continue
                break
        except _Cancelled:
            # Observability (INV-SERVE-CANCEL e2e evidence): a cancelled
            # generation must be visible server-side — the [orch-stats]
            # line only prints for completed requests.
            print(f"  [orch] request {req.request_id} cancelled "
                  f"(client gone / response closed)", flush=True)
            self._finish(req, [], "cancelled")
            return True
        except BridgeError as e:
            print(f"  [orch] request {req.request_id} failed: {e}",
                  flush=True)
            self._finish(req, [], "error", str(e))
            return True
        except Exception as e:                       # noqa: BLE001
            # Any other request-scoped failure is the SAME class of event
            # (TD-SERVE-ERROR-MASKING): report it to the caller instead of
            # killing the serving loop (which would hang the HTTP request
            # until the client gives up).
            print(f"  [orch] request {req.request_id} failed: "
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                  flush=True)
            self._finish(req, [], "error", f"{type(e).__name__}: {e}")
            return True
        stats.degraded_retries = retries
        self.last_stats = stats
        # Server-side per-request performance line (serving-gap ledger):
        # SUSTAINED decode tok/s = committed tokens / decode wall — prefill
        # and queueing excluded by construction (RequestStats.tok_per_s).
        acc = (stats.accepted / stats.proposed) if stats.proposed else 0.0
        print(f"  [orch-stats] req={req.request_id} tokens={stats.tokens} "
              f"finish={reason} rounds={stats.rounds} "
              f"acc={acc:.4f} ({stats.accepted}/{stats.proposed}) "
              f"gov_plain={stats.gov_plain_rounds} "
              f"gov_probes={stats.gov_probe_rounds} "
              f"prefix_hit={stats.prefix_hit_tokens} "
              f"prefill_ms={stats.prefill_ms:.1f} "
              f"decode_ms={stats.decode_wall_ms:.1f} "
              f"decode_tok_s={stats.tok_per_s:.3f} "
              f"moe_degraded_layers={stats.moe_degraded_layers} "
              f"degraded_retries={stats.degraded_retries} "
              f"indexer_dense_steps={stats.indexer_dense_steps}"
              # P-29 step 24 observability (appended LAST — harnesses
              # regex this line positionally): checkpoint restore point,
              # captures this request, and the cache-wide hit/miss export
              # (PrefixCache counted these since 2026-08 but never
              # exported them — the step-23 finding that entry eviction
              # was indistinguishable from divergence in the field).
              + (f" ckpt_restore={stats.ckpt_restore_pos}"
                 f" ckpt_captured={stats.ckpt_captured}"
                 if self._kda_ckpt_on else "")
              + (f" pc_hits={self.prefix_cache.hits}"
                 f" pc_misses={self.prefix_cache.misses}"
                 f" pc_evictions={self.prefix_cache.evictions}"
                 if self.prefix_cache is not None else ""), flush=True)
        if stats.indexer_dense_steps:
            # TD-INDEXER-NO-DENSE-FALLBACK: this must NEVER happen with
            # reserve-at-admission live — a nonzero count means some
            # attention steps ran DSA-DENSE (dead indexer coverage): the
            # request was served ~10x slower AND its trajectory is not
            # comparable to a sparse run.  Treat as a BUG (file/inspect the
            # engine log for "INDEXER DENSE DOWNGRADE"), not as load.
            print(f"  [orch] WARNING req={req.request_id}: "
                  f"{stats.indexer_dense_steps} attention step(s) ran with "
                  f"DENSE (kDead) indexer coverage — this is a BUG witness "
                  f"(TD-INDEXER-NO-DENSE-FALLBACK); no prefix holder was "
                  f"registered from this request", flush=True)
        if stats.moe_degraded_layers:
            # TD-MOE-PROGRESSIVE-DEGRADED-SILENT: this answer was computed
            # with an incomplete expert set on N layers — flag it loudly so
            # callers/harnesses can retry or discard instead of comparing
            # it.  With the serving-level retry ON, reaching here means the
            # retry budget was exhausted, the off-switch is set, or tokens
            # had already streamed — a GROWING delivered-degraded total is
            # a real capacity problem (the post-boot cold share degrades
            # once and self-corrects), not something retries should hide.
            self.degraded_delivered_total += 1
            print(f"  [orch] WARNING req={req.request_id}: "
                  f"{stats.moe_degraded_layers} MoE layer(s) finalized "
                  f"DEGRADED (incomplete expert set) — output is not "
                  f"identity-comparable "
                  f"(retries {stats.degraded_retries}/"
                  f"{self._degraded_retry_max}; "
                  f"{self.degraded_delivered_total} degraded request(s) "
                  f"delivered this boot)", flush=True)
        self._finish(req, tokens, reason, logprobs=logprobs)
        return True

    def _finish(self, req: InferenceRequest, tokens: list[int],
                reason: str, err: str = "",
                logprobs: list[StepLogprobs | None] | None = None) -> None:
        self._cancelled.discard(req.request_id)
        cb = req.on_complete
        if cb is None:
            return
        lp = None if err else logprobs
        if _accepts_error_arg(cb):
            cb(req.request_id, tokens, reason, lp, error=err)
        else:
            cb(req.request_id, tokens, reason, lp)

    # ── generation ───────────────────────────────────────────────────────

    # ── TD-KVXP-PER-STEP-FLOOR: bounded large-prefill wait helpers ───────
    # (knob rationale + the INV-KDA-STATE (a) argument at the __init__
    # block that reads the env knobs.)

    def _kvxp_wait_arm(self, prompt_len: int, request_id: int) -> None:
        """Arm the bounded wait for THIS request iff it is a large prefill
        (len(prompt) >= LS_KVXP_LARGE_PREFILL_TOKENS) and the 44z
        rebalancer is on. Called at the top of _generate; disarmed the
        moment the prefill phase completes (decode steps never wait)."""
        self._kvxp_wait_deadline = None
        self._kvxp_wait_used = False
        self._kvxp_wait_req = request_id
        self._kvxp_wait_armed = (self._kvxp_wait_enabled
                                 and prompt_len >= self._kvxp_wait_threshold)

    def _kvxp_wait_disarm(self, rescued_counts: bool = False) -> None:
        """Disarm at prefill end (rescued_counts=True: a completed prefill
        that used the wait was rescued by it) and in the request finally."""
        if rescued_counts and getattr(self, "_kvxp_wait_used", False):
            self._kvxp_wait_rescues += 1
            print(f"  [orch] request {self._kvxp_wait_req}: large-prefill "
                  f"bounded wait RESCUED this prefill "
                  f"({self._kvxp_wait_rescues} rescued total)", flush=True)
        self._kvxp_wait_armed = False
        self._kvxp_wait_deadline = None
        self._kvxp_wait_used = False

    def _kvxp_try_wait(self, err: BaseException) -> bool:
        """One bounded-wait step at a retryable pool-exhaustion refusal
        AFTER holder eviction has nothing left to give. True = slept
        (caller re-issues the identical claim); False = not eligible or
        budget exhausted (caller raises — today's behavior, verbatim).

        The refusal itself armed the engine's eager drain
        (note_pool_pressure_refusal), so the sleep is spent on a drain
        already in flight. Each re-issued claim is a fresh all-or-nothing
        admission attempt engine-side (INV-KDA-STATE (a) untouched)."""
        if not self._kvxp_wait_armed or not is_pool_exhaustion(err):
            return False
        now = time.monotonic()
        if self._kvxp_wait_deadline is None:
            self._kvxp_wait_deadline = (now
                                        + self._kvxp_wait_budget_ms / 1e3)
            print(f"  [orch] request {self._kvxp_wait_req}: pool exhausted "
                  f"on a LARGE prefill with nothing left to evict — "
                  f"waiting up to {self._kvxp_wait_budget_ms} ms for the "
                  f"44z background drain (TD-KVXP-PER-STEP-FLOOR): {err}",
                  flush=True)
        if now >= self._kvxp_wait_deadline:
            print(f"  [orch] request {self._kvxp_wait_req}: large-prefill "
                  f"wait budget exhausted "
                  f"({self._kvxp_wait_budget_ms} ms) — surfacing the "
                  f"refusal", flush=True)
            return False
        self._kvxp_wait_engagements += 1
        self._kvxp_wait_used = True
        time.sleep(min(self._kvxp_wait_poll_s,
                       max(0.0, self._kvxp_wait_deadline - now)))
        return True

    def _with_pool_evict_retry(self, fn, *, spec_round: bool = False):
        """Run one engine step, answering POOL EXHAUSTION with a prefix-holder
        eviction + retry (TD-INDEXER-POOL-EVICT).

        Retained holders are live sequences: they pin KV pages AND a CoW
        indexer-K frontier page group each.  A step that exhausts a pool is
        therefore usually blocked by CACHE, not by work — the same reading
        that already drives the seq_create (`evict_for_admission`) and
        seq_fork (`_fork_with_evict_retry`) retries, now extended to the
        PREFILL step itself.  This is what keeps a KV-demoted sequence
        SPARSE: the engine fail-closes such a step instead of downgrading it
        to dense (a dense step would punch a permanent hole in the indexer
        coverage and force a full cold-page re-promotion), so the only way
        forward is to free capacity and re-issue the identical step.  The
        failing step mutated no engine state (the guard runs before any
        attention work), so the retry is exact.  Re-raises when the error is
        not exhaustion or nothing is left to evict.

        ``spec_round=True`` marks a MID-SPECULATIVE-ROUND step (resolves
        TD-SPEC-ROUND-POOL-EVICT: the overlap plain/masked step, the
        batched-verify chunk, the guided bonus re-feed).  The retry is
        exact there too, for the same reasons plus three round-specific
        facts:
          1. ENGINE — the mid-round shapes are the fail-closed step shapes
             already wrapped elsewhere: the overlap step IS a plain decode
             step, and the verify chunk provisions every pool through its
             LAST row's position before any KV/tier mutation
             (ensure_v4_tier_pages Ticket-J block / ensure_indexer_pages);
             re-issuing the identical rows at the identical positions is
             the blessed position-addressed re-feed (INV-DSA-REWIND /
             INV-DSPARK-REFEED), deterministic, so identical values land.
          2. BRIDGE — when the dspark draft is still IN FLIGHT (the
             overlap step only; the verify chunk runs after collect), the
             eviction's SEQ_FREE wait() rides the existing async-dspark
             stash: a draft completion (or draft CMP_ERROR) arriving
             during ANY wait is stashed and surfaced at
             dspark_collect_async, on both bridge poll paths — freeing a
             holder under a pending draft is therefore the already-built
             interleave, unit-validated against the scripted daemon.
          3. DRAFT STATE — eviction frees a HOLDER sequence, disjoint from
             the live sequence and from the draft's separate context
             arena; the in-flight proposal depends only on (anchor, fed),
             which the retry does not change, so the collected draft still
             verifies against identical target rows and the round's
             acceptance statistics are counted exactly once.
        Mid-round retries are counted SEPARATELY
        (``_spec_round_pool_evict_retries``) so a box riding this seam
        constantly is visible in the serve log rather than silently
        slower.
        """
        while True:
            try:
                return fn()
            except BridgeError as err:
                if (not is_pool_exhaustion(err)
                        or self.prefix_cache is None
                        or not self.prefix_cache.evict_for_admission()):
                    # TD-KVXP-PER-STEP-FLOOR: with every evictable holder
                    # gone, a LARGE prefill's step may wait — bounded — on
                    # the 44z background drain instead of failing the
                    # request (windowed admission defers the bulk KV claim
                    # to exactly these mid-prefill growth steps). Decode
                    # and mid-spec-round steps never reach here armed: the
                    # wait is disarmed the moment the prefill phase ends.
                    if (not spec_round and is_pool_exhaustion(err)
                            and self._kvxp_try_wait(err)):
                        continue
                    raise
                if spec_round:
                    self._spec_round_pool_evict_retries += 1
                    print(f"  [orch] pool exhausted mid-spec-round — "
                          f"evicted a prefix holder, retrying "
                          f"({self._spec_round_pool_evict_retries} "
                          f"mid-round total)", flush=True)
                else:
                    self._pool_evict_retries += 1
                    print(f"  [orch] pool exhausted mid-prefill — "
                          f"evicted a prefix holder, retrying "
                          f"({self._pool_evict_retries} "
                          f"total)", flush=True)

    def _generate(self, req: InferenceRequest,
                  degraded_retry_ok: bool = False
                  ) -> tuple[list[int], str, RequestStats,
                             list[StepLogprobs | None] | None]:
        prompt = req.prompt_token_ids
        if not prompt:
            raise BridgeError("empty prompt")
        vocab = self.bridge.vocab_size
        for t in prompt:
            if not (0 <= t < vocab):
                raise BridgeError(f"prompt token out of vocab: {t}")

        stats = RequestStats()
        # TD-KVXP-PER-STEP-FLOOR: arm the bounded large-prefill wait for
        # this request (no-op unless 44z is on and the prompt clears the
        # threshold); disarmed at prefill end and in the finally below.
        self._kvxp_wait_arm(len(prompt), req.request_id)
        # TD-MOE-PROGRESSIVE-DEGRADED-SILENT: per-request delta of the
        # bridge's monotonic degraded-finalize counter.
        degraded_base = self.bridge.moe_degraded_layers
        # TD-INDEXER-NO-DENSE-FALLBACK: per-request delta of the dense-step
        # witness counter (0 in mock bridges without the attribute).
        idense_base = getattr(self.bridge, "indexer_dense_steps", 0)
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        eos = set(self.metadata.eos_token_ids)
        limit = req.max_tokens if req.max_tokens > 0 else None

        out: list[int] = []
        finish = "length"
        want_lp = req.logprobs is not None
        lp_out: list[StepLogprobs | None] | None = [] if want_lp else None

        def emit(tok: int, lp: StepLogprobs | None = None) -> bool:
            """Commit one token toward the caller. False = stop now
            (EOS reached or limit hit); tokens past the stop point in the
            same speculative round are DROPPED (standard spec-dec serving
            semantics — KV overshoot is freed with the sequence).
            ``lp`` = this token's StepLogprobs (logprobs requests only)."""
            nonlocal finish
            out.append(tok)
            if lp_out is not None:
                lp_out.append(lp)
            if req.on_token is not None:
                req.on_token(req.request_id, tok, lp)
            if tok in eos:
                finish = "stop"
                return False
            if limit is not None and len(out) >= limit:
                finish = "length"
                return False
            return True

        def check_cancel() -> None:
            if self._shutdown or req.request_id in self._cancelled:
                raise _Cancelled()

        pc_skip_logged = False

        def pc_register(toks: tuple[int, ...]) -> None:
            """Register a frozen prefix holder for this request.  A
            skipped registration is LOGGED once per request with the
            reason (over-sized / duplicate / fork-failed) — a silent skip
            reads as a wall-clock regression on the next shared-prefix
            request (user report 2026-08-26: prefix caching 'dead' above
            the old 8192-token default)."""
            nonlocal pc_skip_logged, kda_ckpt_pending_bytes
            pc = self.prefix_cache
            # TD-MOE-PROGRESSIVE-DEGRADED-SILENT: a degraded MoE finalize
            # anywhere in this request so far means its hidden states — and
            # therefore the KV this holder would freeze — were computed with
            # an incomplete expert set.  Registering it would propagate the
            # wrong KV to every later prefix HIT (an INV-PREFIX-CACHE-1
            # violation that outlives the degraded request), so skip.
            if self.bridge.moe_degraded_layers - degraded_base > 0:
                if not pc_skip_logged:
                    pc_skip_logged = True
                    print(f"  [orch] request {req.request_id}: prefix-cache "
                          f"registration skipped — degraded MoE layers in "
                          f"this request (KV not identity-trustworthy)",
                          flush=True)
                return
            # TD-INDEXER-NO-DENSE-FALLBACK: a request that served any
            # DENSE attention step has a HOLE in its indexer coverage — a
            # holder frozen from it would hand every future hit-child a
            # kDead (permanently dense) start AND break INV-PREFIX-CACHE-1
            # (an uncached run would serve sparse; dense flips near-tie
            # tokens).  Never register such a holder.
            if (getattr(self.bridge, "indexer_dense_steps", 0)
                    - idense_base > 0):
                if not pc_skip_logged:
                    pc_skip_logged = True
                    print(f"  [orch] request {req.request_id}: prefix-cache "
                          f"registration skipped — request served DENSE "
                          f"indexer steps (coverage hole; "
                          f"TD-INDEXER-NO-DENSE-FALLBACK)", flush=True)
                return
            holder_id = self._next_seq_id
            # P-29 step 24: a successful registration is a FROZEN fork —
            # the engine MOVES this request's captured KDA checkpoint
            # blobs to the holder; hand the position/byte bookkeeping to
            # the entry and stop carrying it on the request. A skipped or
            # failed registration leaves the blobs on the request seq
            # (they die at its seq_free) and the pending list intact for
            # the NEXT registration attempt (grid then exact-body).
            if pc.register(toks, seq_id, holder_id,
                           kda_ckpts=tuple(kda_ckpts_pending),
                           kda_ckpt_bytes=kda_ckpt_pending_bytes):
                self._next_seq_id += 1
                if kda_ckpts_pending:
                    kda_ckpts_pending.clear()
                    kda_ckpt_pending_bytes = 0
            elif not pc_skip_logged:
                pc_skip_logged = True
                print(f"  [orch] request {req.request_id}: prefix-cache "
                      f"registration skipped — {pc.last_skip}")

        kda_ckpts_pending: list[int] = []
        kda_ckpt_pending_bytes = 0

        def kda_ckpt_capture(cpos: int) -> None:
            """P-29 step 24: capture one KDA prefix checkpoint at cpos
            (a between-command prefill boundary — uniform frontier by
            construction; multiple of 512 by the cadence arithmetic).
            Failure NEVER fails the request: budget/capacity skips are
            silent-but-counted; engine refusals are the capture TRIPWIRE
            (wrong capture point — must read 0) and are logged loudly."""
            nonlocal kda_ckpt_pending_bytes
            pc = self.prefix_cache
            if pc is None:
                return
            est = self._kda_ckpt_unit_bytes
            if (self._kda_ckpt_budget > 0
                    and (pc.kda_ckpt_bytes + kda_ckpt_pending_bytes + est
                         > self._kda_ckpt_budget)):
                return  # budget: skip capture, keep serving
            try:
                st, nbytes = self.bridge.kda_ckpt(seq_id, cpos)
            except BridgeError as err:
                self._kda_ckpt_refusals += 1
                print(f"  [orch] request {req.request_id}: KDA checkpoint "
                      f"capture @{cpos} REFUSED ({err}) — capture-point "
                      f"tripwire (refusals={self._kda_ckpt_refusals}, "
                      f"must be 0)", flush=True)
                return
            if st == 0 and nbytes > 0:
                kda_ckpts_pending.append(cpos)
                kda_ckpt_pending_bytes += int(nbytes)
                self._kda_ckpt_unit_bytes = int(nbytes)
                stats.ckpt_captured += 1
                stats.ckpt_bytes += int(nbytes)

        # ── prefix-cache lookup: fork from a retained holder when the
        # prompt starts with a registered prefix.  INV-PREFIX-CACHE-1
        # (token identity ON vs OFF) holds BY CONSTRUCTION because both
        # the registry and the prefill run on the ABSOLUTE 64-token chunk
        # grid: prefill chunk boundaries are position 0 + k*64, and
        # holders are registered only at grid-aligned lengths — so a hit's
        # reused KV was produced by exactly the chunk shapes the uncached
        # run would use, and its delta continues on the same grid
        # (chunk-boundary bf16 numerics are shape-sensitive; identical
        # boundaries ⇒ bit-identical KV, TD-PREFILL-NONDET).
        pre = len(prompt) - 1
        # Registration grid: the ACTIVE prefill stride.  Chunked path: the
        # chunk size (GLM 64).  Superchunk path (V4): the SUPERCHUNK stride
        # (moe_batch_capacity, boot-constant) — holders must sit on
        # superchunk boundaries so a hit's delta reproduces the exact
        # absolute superchunk/sub-chunk shapes of an uncached run
        # (TD-V4-SERVE-PREFIX; chunk-boundary bf16 numerics are
        # shape-sensitive).
        sc_stride = (min(self._prefill_sc_stride,
                         self.bridge.moe_batch_capacity)
                     if self._prefill_sc_stride > 0
                     else self.bridge.moe_batch_capacity)
        grid = (sc_stride
                if (self._prefill_superchunk and sc_stride > 0)
                else self._prefill_chunk)
        grid_len = (pre // grid) * grid
        sc_path = self._prefill_superchunk and sc_stride > 0
        # Superchunk hit validity: grid-aligned entries reproduce uncached
        # superchunk boundaries; an EXACT-prompt-body entry (n == pre) is
        # by-construction identical too (the holder's shapes ARE this
        # body's uncached shapes). Chunked path: every entry is
        # grid-aligned already.
        # R4c mid-edge reuse — gated on the ARCH CAPABILITY
        # (EngineInfo.seq_fork_truncatable, INV-SEQ-FORK-TRUNC; a property
        # of the architecture, never a model name): when the engine can
        # fork a holder TRUNCATED to an interior length, the reuse point
        # is the GRID-CLAMPED longest common token prefix with ANY
        # registered node — no registered node is needed at the reuse
        # length (the win over the legacy full-node filter), but the
        # reuse point itself stays ON the absolute prefill grid
        # (RADIX_SLAB_DESIGN §3(b): edge splits at grid boundaries, never
        # finer).  Identity therefore holds BY CONSTRUCTION (the delta's
        # command stream from a grid point equals an uncached run's — the
        # realignment in the prefill loops is a no-op there) AND was
        # re-established by the R4b offset gates; SUB-GRID reuse is
        # forbidden — the R4b sweep measured mid-page delta starts
        # diverging (deterministic_ep_combine=ON, degraded discarded;
        # spec/measurements/glm_serving.md 2026-08-28).  Without the
        # capability (V4 in-place rings) the legacy grid / exact-body
        # filter stays — the engine would reject prefix_len there.
        # TD-INDEXER-NO-DENSE-FALLBACK (Route 1) admission reservation:
        # the context this request may EVER reach — prompt + generation
        # budget + a speculative-overshoot margin (a verify round computes
        # attention up to gamma+1 positions past the last committed token;
        # overshoot tokens are dropped but their steps still append indexer
        # keys).  Unbounded requests (max_tokens unset) reserve the full
        # window — the engine clamps to serving.max_sequence_length and
        # returns the GRANT, which then caps generation below so no step
        # can ever run past the reservation.  reserve=0 (incapable bridge)
        # keeps legacy lazy provisioning end to end.
        margin = (self.spec.gamma + 2) if self.spec.enabled else 0
        reserve = 0
        if self._bridge_reserves:
            want = (len(prompt) + margin
                    + (limit if limit is not None else (1 << 31)))
            reserve = min(want, 0xFFFFFFFF)
        granted = 0
        entry = None
        reuse = 0
        if self.prefix_cache is not None:
            if self.metadata.seq_fork_truncatable:
                hit = self.prefix_cache.lookup_mid_edge(
                    prompt, grid, subgrid=self.subgrid_mid_edge)
                if hit is not None:
                    entry, reuse = hit
            else:
                entry = self.prefix_cache.lookup(
                    prompt,
                    valid=((lambda n: n % grid == 0 or n == pre)
                           if sc_path else None))
                if entry is not None:
                    reuse = len(entry.tokens)
                if self._kda_ckpt_on:
                    # P-29 step 24 (LS_KDA_PREFIX_CKPT): the checkpoint
                    # floor COMPETES with the whole-node hit and wins when
                    # it is DEEPER. A whole-node hit lands on the deepest
                    # registered node that is a full PREFIX of the prompt
                    # — which, in a deep diverging session, is often a
                    # SHALLOW ancestor (e.g. an 8k system-prompt holder
                    # that prefixes a 97k conversation): taking it would
                    # re-prefill everything past it. The checkpoint floor
                    # (largest checkpoint <= the divergence over every
                    # candidate and its ancestors) reaches far deeper, and
                    # the engine admits the truncating fork because the
                    # holder physically carries a blob there; the replay
                    # of [C, len(prompt)) is bit-identical over the shared
                    # span (INV-KDA-CARRY). No checkpoint beats the
                    # whole-node hit => keep it (or full re-prefill on a
                    # total miss); never approximate.
                    got = self.prefix_cache.lookup_kda_ckpt(prompt)
                    if got is not None and got[1] > reuse:
                        entry, reuse = got
                        stats.ckpt_restore_pos = reuse
                        self.prefix_cache.note_kda_ckpt_hit(entry)
                    elif entry is None and self.prefix_cache.last_kda_lcp > 0:
                        # OQ-13 falsifier instrument: the miss-LCP
                        # distribution decides whether the cadence (or
                        # the prompt template) needs to change.
                        print(f"  [orch] request {req.request_id}: KDA "
                              f"divergence miss — token LCP "
                              f"{self.prefix_cache.last_kda_lcp}, no "
                              f"checkpoint at or below it (full "
                              f"re-prefill)", flush=True)
        if entry is not None:
            was_spilled = bool(getattr(entry, "spilled", False))
            while True:
                try:
                    granted = self.prefix_cache.fork_from(
                        entry, seq_id, reuse_len=reuse,
                        reserve_tokens=reserve)
                    # TD-PREFIX-TIDY-COLD-SPILL: a hit on a spilled holder
                    # reloaded its cold pages engine-side (the fork handler
                    # unspills first; the file is unlinked on success).
                    entry.spilled = False
                    break
                except BridgeError as err:
                    # TD-KVXP-PER-STEP-FLOOR: a cache-hit fork claims the
                    # same non-deferrable upfront footprint as a fresh
                    # admission (full KDA state copy + child reservation);
                    # a LARGE prefill may wait — bounded — on the 44z
                    # drain and re-issue the fork (all-or-nothing with
                    # rollback engine-side, so the retry is exact and the
                    # holder is protected from its own eviction seam).
                    if (not was_spilled and is_pool_exhaustion(err)
                            and self._kvxp_try_wait(err)):
                        continue
                    # A spilled holder whose reload cannot complete (lost
                    # or unreadable file; cold slots exhausted even after
                    # the eviction seam) degrades to a cache MISS: evict
                    # the holder, re-prefill from scratch — a spill NEVER
                    # fails a request (TD-PREFIX-TIDY-COLD-SPILL).
                    if not was_spilled:
                        raise
                    print(f"  [orch] request {req.request_id}: spilled "
                          f"prefix holder seq {entry.seq_id} could not "
                          f"reload ({err}) — evicting it and serving as "
                          f"a MISS", flush=True)
                    self.prefix_cache.drop_leaf(entry)
                    entry = None
                    break
        if entry is not None:
            pos = reuse
            sub_hit = reuse % grid != 0 and reuse != pre
            if reuse != len(entry.tokens) or sub_hit:
                # A mid-edge hit is never silent: the serve log must show
                # mid-edge reuse engaging (R4b gate greps for this line).
                # A SUB-GRID hit (opt-in, accepted-divergence class) is
                # additionally marked — including the full-fork-of-a-
                # non-grid-node shape (reuse == registered length), which
                # diverges the same way (R4b offset 1599).
                kind = ("truncating fork" if reuse != len(entry.tokens)
                        else "full fork")
                mark = ("SUB-GRID (opt-in, divergence-accepted), "
                        if sub_hit else "")
                print(f"  [orch] request {req.request_id}: prefix mid-edge "
                      f"hit — reusing {reuse}/{len(entry.tokens)} tokens "
                      f"of a registered node ({mark}{kind})", flush=True)
            stats.prefix_hit_tokens = pos
            # Forked sequences never capture position 0: drafting works
            # only if the engine ADOPTS the tracked context (fork point
            # inside its frontier) — see _draft_ctx_valid in __init__.
            # Under rotation (TD-DSPARK-CTX-POLICY) the engine also
            # requires the fork point at or above its window base; the
            # post-rotation window is always >= cap/2, so this bound is
            # conservative — an engine base above it (post-oversized-epoch
            # re-arm) surfaces as the sticky per-request fallback
            # (INV-SERVE-SPEC-FALLBACK), never an error.
            draft_adoptable = (self._draft_ctx_valid
                               and pos <= self._draft_ctx_len
                               and (not self.spec.ctx_rotate
                                    or pos >= max(
                                        0, self._draft_ctx_len
                                        - self.spec.ctx_cap_tokens // 2)))
        else:
            # Evict-at-admission (regression-hunt 2026-08-23 finding (b)):
            # retained prefix holders pin their KV pages, so a NEW request's
            # seq_create can hit page-pool exhaustion while the pool is
            # merely full of cache, not of work. Mirror the fork-time
            # evict-retry: on an engine "exhausted" error, free one holder
            # (LRU-major, deepest-first) and retry; re-raise when nothing
            # is left to evict or the error is not exhaustion.
            while True:
                try:
                    if reserve > 0:
                        granted = self.bridge.create_sequence(
                            seq_id, len(prompt),
                            reserve_tokens=reserve) or 0
                    else:
                        granted = self.bridge.create_sequence(
                            seq_id, len(prompt)) or 0
                    break
                except BridgeError as err:
                    # Retryable exhaustion now includes the ADMISSION
                    # RESERVATION failure (TD-INDEXER-NO-DENSE-FALLBACK):
                    # the pool is full of holder cache, not work — evict
                    # one and re-issue, same seam as before.
                    if (not is_pool_exhaustion(err)
                            or self.prefix_cache is None
                            or not self.prefix_cache.evict_for_admission()):
                        # TD-KVXP-PER-STEP-FLOOR: a LARGE prefill's
                        # admission (KDA state + windowed KV + indexer
                        # reservation, all-or-nothing engine-side) may
                        # wait — bounded — on the 44z background drain
                        # the refusal itself armed, then re-issue the
                        # identical seq_create. Measured repro
                        # (kvxp_repub.log 17:12:18): the retry 500'd
                        # ~200 ms before the eager drain landed.
                        if (is_pool_exhaustion(err)
                                and self._kvxp_try_wait(err)):
                            continue
                        raise
            pos = 0
            draft_adoptable = True  # position-0 capture re-arms the context
        # TD-INDEXER-NO-DENSE-FALLBACK: the engine committed indexer-K pages
        # for positions [0, granted) — cap generation so NO step (verify
        # overshoot included, via margin) can compute a position past the
        # reservation.  What used to be a silent mid-request dense downgrade
        # at the window edge is now an admission-time refusal / clamp.
        if granted > 0:
            cap = granted - len(prompt) - margin
            if cap <= 0:
                raise BridgeError(
                    f"prompt ({len(prompt)} tokens) leaves no generation "
                    f"room inside the reserved context window ({granted} "
                    f"tokens, speculation margin {margin})")
            if limit is None or limit > cap:
                if limit is not None:
                    print(f"  [orch] request {req.request_id}: max_tokens "
                          f"{limit} clamped to {cap} (reserved context "
                          f"window {granted}, prompt {len(prompt)}, "
                          f"margin {margin})", flush=True)
                limit = cap
        # ── SC small-prefill downgrade (user report 2026-08-26): a
        # superchunk sweeps the UNION of experts for the whole chunk in
        # one MOE_BIG per layer — on a SMALL prefill (a short follow-up
        # turn on a cached prefix) that sweep evicts the decode-warmed
        # expert cache for no amortization win.  Below the threshold the
        # DELTA (pre - pos: the tokens actually swept — the decision input
        # is eviction pressure, not prompt length) runs the ordinary
        # chunked path at a SMALL chunk size, touching experts in
        # increments the resident cache can absorb.  0 disables (route
        # everything superchunk — pre-2026-08-26 behavior).  Lookup above
        # already used the SC validity filter, so a downgraded delta still
        # starts sc-grid-aligned; registration below stays on the SC grid
        # (holders remain hittable by later superchunk lookups).  A
        # downgraded delta's chunk shapes differ from an uncached sc run's
        # — same accepted bf16 batch-width shape class as
        # TD-SERVE-SC-TRAJECTORY (INV-PREFIX-CACHE-1 relaxation recorded
        # there and in SPEC_UPDATES); the adaptive successor is
        # TD-SC-SMALL-PREFILL-ADAPTIVE.
        sc_small = (sc_path and self._prefill_sc_min > 0
                    and (pre - pos) < self._prefill_sc_min)
        # Pessimise the mirror while this request's captures are in flight:
        # a cancel mid-prefill leaves the engine context partially fed, so
        # only a COMPLETED prefill may re-validate it (below).
        self._draft_ctx_valid = False
        try:
            # ── chunked prefill of prompt[pos:-1] on the absolute grid;
            # register a frozen holder at THIS prompt's grid boundary
            # (fork before the partial tail chunk / decode mutate the
            # frontier — CoW gives the holder its own frontier copy). ────
            t0 = time.monotonic()
            if self._teacher_forced_prefill:
                # deepseek_v4 (TD-V4-CHUNK-PREFILL): one B=1 decode-shaped
                # step per prompt token — the exact ticket-H golden
                # harness shape (head+sample included; their cost is noise
                # against the per-token expert streaming wall).  The
                # sampled token is discarded (teacher forcing).
                while pos < pre:
                    check_cancel()
                    r = self._with_pool_evict_retry(
                        lambda: self.bridge.decode_step_fetch_and_run(
                            prompt[pos], seq_id, pos, None))
                    self._guard(r)
                    pos += 1
            elif sc_path and not sc_small:
                # SC (superchunk port): superchunks of `sc_stride` tokens
                # (V4: the engine's elastic MoE batch capacity; GLM: the
                # single-shot MoE chunk capacity — TD-MOE-EP-XTP-WAVES),
                # sub-chunked at _prefill_chunk rows — ONE
                # FETCH_AND_RUN_MOE_BIG per layer per superchunk.
                # TD-V4-SERVE-PREFIX: holders are registered on the
                # SUPERCHUNK grid (grid_len above), so both the holder's
                # prefill and a hit's delta run the exact absolute
                # superchunk boundaries of an uncached run.
                while pos < pre:
                    if (self.prefix_cache is not None and pos == grid_len
                            and grid_len > 0):
                        pc_register(tuple(prompt[:grid_len]))
                    check_cancel()
                    # R4c: a mid-edge fork can start OFF-GRID — the
                    # first superchunk ends at the next ABSOLUTE stride
                    # boundary so every subsequent superchunk reproduces
                    # an uncached run's shapes (identity argument, R4b).
                    n = min(sc_stride - pos % sc_stride, pre - pos)
                    self._with_pool_evict_retry(
                        lambda: self.bridge.prefill_superchunk_fetch_and_run(
                            prompt[pos:pos + n], seq_id, pos,
                            self._prefill_chunk))
                    self._mtp_fill_chunk(prompt, seq_id, pos, n)
                    pos += n
                    # P-29 step 24: checkpoint cadence — capture BETWEEN
                    # prefill commands (uniform per-layer KDA frontier by
                    # construction), on multiples of the interval, never
                    # at grid_len (the holder's own registered-length
                    # spill blob already serves that position) and never
                    # at the prompt end (nothing to replay past it).
                    if (self._kda_ckpt_on and pos < pre
                            and pos != grid_len
                            and pos % self._kda_ckpt_iv == 0):
                        kda_ckpt_capture(pos)
                if (self.prefix_cache is not None and pos == grid_len
                        and grid_len > 0):
                    # pre was exactly grid-aligned — register at the end.
                    pc_register(tuple(prompt[:grid_len]))
                if (self.prefix_cache is not None and pre > 0
                        and pre != grid_len):
                    # EXACT-prompt-body holder: a repeated identical body
                    # (the common shared-prompt case) reuses shapes by
                    # construction even when pre is not superchunk-aligned
                    # — a hit skips the whole prefill.
                    pc_register(tuple(prompt[:pre]))
            elif sc_path:
                # SC-SMALL downgrade (see sc_small above): the
                # below-threshold DELTA runs ordinary chunked FETCH_AND_RUN
                # at the SMALL chunk size — per-chunk expert unions the
                # decode-warmed cache can absorb, instead of one whole-
                # delta MOE_BIG union sweep.  Registration MIRRORS the sc
                # branch (SC grid + exact body): holders must stay on the
                # sc lookup grid to be hittable by later superchunk-path
                # requests (a small-chunk grid entry would be filtered
                # out by the SC validity check and only waste budget).
                step = self._prefill_sc_small_chunk
                while pos < pre:
                    if (self.prefix_cache is not None and pos == grid_len
                            and grid_len > 0):
                        pc_register(tuple(prompt[:grid_len]))
                    check_cancel()
                    # R4c realignment: first chunk ends at the next
                    # absolute `step` boundary (mid-edge forks start
                    # off-grid).
                    n = min(step - pos % step, pre - pos)
                    if pos < grid_len:       # never stride past the SC
                        n = min(n, grid_len - pos)   # registration point
                    self._with_pool_evict_retry(
                        lambda: self.bridge.prefill_chunk_fetch_and_run(
                            prompt[pos:pos + n], seq_id, pos, None))
                    self._mtp_fill_chunk(prompt, seq_id, pos, n)
                    pos += n
                    # P-29 step 24: checkpoint cadence — capture BETWEEN
                    # prefill commands (uniform per-layer KDA frontier by
                    # construction), on multiples of the interval, never
                    # at grid_len (the holder's own registered-length
                    # spill blob already serves that position) and never
                    # at the prompt end (nothing to replay past it).
                    if (self._kda_ckpt_on and pos < pre
                            and pos != grid_len
                            and pos % self._kda_ckpt_iv == 0):
                        kda_ckpt_capture(pos)
                if (self.prefix_cache is not None and pos == grid_len
                        and grid_len > 0):
                    # pre was exactly grid-aligned — register at the end.
                    pc_register(tuple(prompt[:grid_len]))
                if (self.prefix_cache is not None and pre > 0
                        and pre != grid_len):
                    # EXACT-prompt-body holder (same rationale as the sc
                    # branch — serves exact repeats of this prompt).
                    pc_register(tuple(prompt[:pre]))
            else:
                while pos < pre:
                    if (self.prefix_cache is not None and pos == grid_len
                            and grid_len > 0):
                        pc_register(tuple(prompt[:grid_len]))
                    check_cancel()
                    # R4c realignment (chunked path): same absolute-grid
                    # argument as the sc branch.
                    n = min(self._prefill_chunk - pos % self._prefill_chunk,
                            pre - pos)
                    self._with_pool_evict_retry(
                        lambda: self.bridge.prefill_chunk_fetch_and_run(
                            prompt[pos:pos + n], seq_id, pos, None))
                    self._mtp_fill_chunk(prompt, seq_id, pos, n)
                    pos += n
                    # P-29 step 24: checkpoint cadence — capture BETWEEN
                    # prefill commands (uniform per-layer KDA frontier by
                    # construction), on multiples of the interval, never
                    # at grid_len (the holder's own registered-length
                    # spill blob already serves that position) and never
                    # at the prompt end (nothing to replay past it).
                    if (self._kda_ckpt_on and pos < pre
                            and pos != grid_len
                            and pos % self._kda_ckpt_iv == 0):
                        kda_ckpt_capture(pos)
                if (self.prefix_cache is not None and pos == grid_len
                        and grid_len > 0):
                    # pre was exactly grid-aligned — register at the end.
                    pc_register(tuple(prompt[:grid_len]))
            stats.prefill_ms = (time.monotonic() - t0) * 1e3
            # TD-KVXP-PER-STEP-FLOOR: prefill done — decode steps must
            # NEVER wait (user requirement), so the wait disarms HERE, not
            # in the finally alone. A completed prefill that engaged the
            # wait was rescued by it (counted).
            self._kvxp_wait_disarm(rescued_counts=True)

            # TD-V4-FIRSTREQ-COLD-SHARE-OVER-CAPACITY option (d): a MoE
            # layer finalized DEGRADED during prefill and the caller holds
            # retry budget -- abandon the attempt NOW, before decode emits
            # a single token, so even streaming requests are retried
            # without double-emitting.  The ``finally`` frees this
            # attempt's sequence; pc_register already refused its holders.
            if (degraded_retry_ok
                    and self.bridge.moe_degraded_layers - degraded_base > 0):
                raise _DegradedRetry(
                    self.bridge.moe_degraded_layers - degraded_base)

            # TD-V4-SPEC-PREFILL-CTX: headless V4 prefill chunks never fire
            # the head-sited aux tap — the engine's draft context ends up
            # INVALID for this sequence.  Mirror that (plain arm; a dspark
            # step against an invalid context would CMP_ERROR the request).
            if (pre > 0 and not self._teacher_forced_prefill
                    and not self._chunk_prefill_arms_draft):
                draft_adoptable = False

            # Mirror the engine's context tracking (conservative frontier =
            # prefill end; engine ctx_len only grows beyond it in decode).
            if draft_adoptable:
                self._draft_ctx_valid = True
                self._draft_ctx_len = pre
            else:
                # The engine invalidates on the first non-adoptable capture
                # of this sequence (sequence switch beyond the frontier).
                self._draft_ctx_valid = False

            t0 = time.monotonic()
            # Arm routing.  The champion greedy arm serves greedy
            # logprobs-off requests byte-identically.  Sampled (T>0)
            # and/or logprobs requests keep the speculative speedup via
            # the host-sampled arm — it needs the engine's MULTI-ROW
            # full-logits readback (>= 2 rows for a verify chunk); on
            # single-row builds they fall back to the plain arm
            # (token-lossless, just slower).  Guided sampled speculation
            # is deferred (TD-GUIDED-SAMPLED-SPEC): guided non-greedy
            # requests take the guided plain arm.
            # TD-DSPARK-CTX-CAP mirror: a PROMPT that overflows the draft's
            # context-KV arena (plus one drafting round of headroom)
            # invalidates the drafting context engine-side during prefill
            # ingest — route it to the PLAIN arm upfront instead of
            # CMP_ERRORing the request at the first RUN_DSPARK_STEP
            # (TD-PREFIX-DSPARK-FORK-CTX; surfaced by long-prompt serving
            # once TD-KVT-ADMISSION-UPFRONT made >8k-token prompts
            # admissible).  Lossless: plain decode is token-identical, just
            # undrafted.  Requests whose GENERATION later crosses the cap
            # keep today's behavior (the residual TD).
            if (self.spec.enabled and draft_adoptable
                    and not self.spec.ctx_rotate
                    and len(prompt) + self.spec.gamma + 1
                        > self.spec.ctx_cap_tokens):
                # Legacy cap only (TD-DSPARK-CTX-POLICY): with rotation the
                # engine windows the draft context and over-cap prompts
                # keep the speculative arm.
                draft_adoptable = False
                print(f"  [orch] request {req.request_id}: prompt "
                      f"{len(prompt)} tokens overflows the draft context-KV "
                      f"arena cap {self.spec.ctx_cap_tokens} — plain decode "
                      f"arm (TD-DSPARK-CTX-CAP)", flush=True)
            # P-29 step 13: the MTP arm needs no draft context (in-model draft).
            mtp_arm = (self.spec.enabled and self.spec.method == "mtp"
                       and not req.force_plain and req.sampling.greedy
                       and not want_lp and req.guided is None)
            spec_ok = (self.spec.enabled and self.spec.method == "dspark"
                       and draft_adoptable
                       and not req.force_plain)
            speculate = spec_ok and req.sampling.greedy and not want_lp
            spec_sampled = (spec_ok and not speculate
                            and req.guided is None
                            and self.bridge.logits_host_addr != 0
                            and self.bridge.logits_host_rows >= 2)
            if req.guided is not None:
                # Guided decoding (TD-SERVE-NAMED-TOOL-CHOICE): emission
                # only after the grammar accepts; completion → tool_calls.
                guided = req.guided

                def emit_guided(tok: int,
                                lp: StepLogprobs | None = None) -> bool:
                    nonlocal finish
                    cont = emit(tok, lp)
                    if guided.completed:
                        finish = "tool_calls"
                        return False
                    return cont

                if speculate:
                    self._decode_speculative_guided(
                        req, seq_id, prompt, emit_guided, check_cancel,
                        stats, guided)
                else:
                    self._decode_guided_plain(
                        req, seq_id, prompt, emit_guided, check_cancel,
                        stats, guided)
            elif mtp_arm:
                self._decode_speculative_mtp(req, seq_id, prompt, pre,
                                             emit, check_cancel, stats)
            elif speculate:
                self._decode_speculative(req, seq_id, prompt, emit,
                                         check_cancel, stats)
            elif spec_sampled:
                self._decode_speculative_sampled(req, seq_id, prompt, emit,
                                                 check_cancel, stats)
            else:
                self._decode_plain(req, seq_id, prompt, emit, check_cancel,
                                   stats)
            stats.decode_wall_ms = (time.monotonic() - t0) * 1e3
            stats.tokens = len(out)
        finally:
            # TD-KVXP-PER-STEP-FLOOR: belt-and-braces disarm — a request
            # that failed mid-prefill must not leave the wait armed for
            # the next request's decode steps.
            self._kvxp_wait_disarm()
            stats.moe_degraded_layers = (self.bridge.moe_degraded_layers
                                         - degraded_base)
            stats.indexer_dense_steps = (
                getattr(self.bridge, "indexer_dense_steps", 0) - idense_base)
            self.bridge.free_sequence(seq_id)
        return out, finish, stats, lp_out

    def _decode_plain(self, req: InferenceRequest, seq_id: int,
                      prompt: list[int], emit, check_cancel,
                      stats: RequestStats) -> None:
        """Sampled (or spec-off greedy) AR decode.

        Logprobs requests add ``logprobs_readback`` to every step: the
        token pick stays engine-side (byte-identical trajectory) while
        the pinned full-logits row feeds the host-side StepLogprobs.

        Per-step seeds: the engine's Philox sampler keys its stream on
        (random_seed, row_idx) and B=1 decode always samples row 0, so a
        constant per-request seed would reuse ONE uniform draw at every
        step (perfectly correlated sampling — the same CDF quantile
        forever).  Each step therefore derives an independent,
        reproducible key from (request seed, position) — _step_seed."""
        self._decode_plain_from(req, seq_id, prompt[-1], len(prompt) - 1,
                                emit, check_cancel, stats)

    def _decode_plain_from(self, req: InferenceRequest, seq_id: int,
                           token: int, pos: int, emit, check_cancel,
                           stats: RequestStats) -> None:
        """The plain AR loop body from an arbitrary (token, pos) resume
        point — feeds ``token`` at ``pos`` first.  ``_decode_plain`` is
        the whole-request entry; the spec→plain FALLBACK
        (INV-SERVE-SPEC-FALLBACK) resumes here mid-request with
        (last committed token, next feed position)."""
        base = None if req.sampling.greedy else req.sampling.as_tuple()
        want_lp = req.logprobs is not None
        while True:
            check_cancel()
            sampling = None
            if base is not None:
                sampling = (base[0], base[1], base[2],
                            self._step_seed(base[3], pos))
            # Evict-retry on pool exhaustion: V4 side tiers (HCA every 256
            # tokens, LID every 8192) and DSA indexer pages also GROW during
            # decode; the engine's provisioning guard fail-closes BEFORE any
            # KV/tier mutation, so the retry is exact — same seam as the
            # prefill steps (TD-INDEXER-POOL-EVICT).
            r = self._with_pool_evict_retry(
                lambda: self.bridge.decode_step_fetch_and_run(
                    token, seq_id, pos, None, sampling=sampling,
                    logprobs_readback=want_lp))
            self._guard(r)
            stats.rounds += 1
            pos += 1
            token = r.sampled_token
            lp = (self._step_logprobs(self._read_logits(), token,
                                      req.logprobs) if want_lp else None)
            if not emit(token, lp):
                return

    def _mtp_fill_chunk(self, prompt: list[int], seq_id: int, pos: int,
                        n: int) -> None:
        """P-29 step 13 phase B: MTP-layer prompt fill for one prefill chunk.

        Covers positions [pos, pos+n): per <=64-row slice, a norm_only
        head produces the slice's post-final-norm hiddens, then serial
        MTP_PROJECTs + one prefill-shaped ATTN(45)+MoE append the MTP
        layer's KV/indexer rows (probe semantics: row @p consumes
        token@p+1 + hidden@p). No-op unless the MTP arm is configured.
        The MTP layer's coverage is its own (engine mtp_indexer_cov), so
        fills advance it independently of the trunk frontier; prefix-
        cache forks clone + truncate it, keeping hit children gap-free."""
        if not (self.spec.enabled and self.spec.method == "mtp"):
            return
        b = self.bridge
        step = 64
        for s0 in range(0, n, step):
            k = min(step, n - s0)
            b.norm_only_head(k, input_row=s0)
            toks = prompt[pos + s0 + 1: pos + s0 + k + 1]
            self._with_pool_evict_retry(
                lambda: b.mtp_prefill_fill(toks, seq_id, pos + s0))

    def _decode_speculative_mtp(self, req: InferenceRequest, seq_id: int,
                                prompt: list[int], pre: int, emit,
                                check_cancel,
                                stats: RequestStats) -> None:
        """P-29 step 13 phase B: gamma<=2 MTP speculation for glm5_next.

        Round = ONE batched verify pass through the target's own kernels
        (bridge.mtp_verify_pass: daemon-side per-row decode-shaped
        attention + one cross-row MoE per layer) over
        [committed-unabsorbed rows] + [drafts], engine-side argmax per
        row; host acceptance walk; KDA anchor-and-replay rewind on
        partial rejection (INV-KDA-REWIND: anchors are taken at pool-
        boundary crossings inside the pass, restore rolls the recurrent
        state + frontier back and the next pass replays the committed
        prefix). MTP rows (probe semantics) back-fill the draft layer's
        KV per round and produce the next drafts: depth-1 from the last
        committed position's target hidden, depth-2 from the recycled
        MTP hidden. Token-identity: every committed token is an argmax
        of the SAME kernels plain decode runs (per-row attention =
        exact B=1 decode path; M=R MoE bitwise per-row-independent on
        the mmvq route) — rejected rows never influence committed ones.

        Draftless rounds (no anchor yet / governor suspension / no
        drafts) are plain-equivalent: 1-row passes committing 1 token.
        Anchor-claim failure degrades to draftless — never an error."""
        b = self.bridge
        gamma = self.spec.gamma
        kp = max(1, self.spec.index_kpool)
        committed = list(prompt)
        fed = len(prompt)           # committed frontier (positions < fed)
        state = pre                 # KDA/state frontier (prefill absorbed)
        mtp_next = pre              # first position lacking a correct MTP row
        drafts: list[int] = []
        anchors: list[int] = []     # live anchor positions (host mirror)
        hw = pre                    # indexer high-water mirror (pool safety)
        dbg = os.environ.get("LS_MTP_ROUND_DEBUG") == "1"
        # Debug bisect knob: "1row" = draftless batch passes (R=1 through
        # the verify machinery, fast path disabled) — isolates the per-row
        # FAR/head shape from the multi-row batch.
        force_arm = os.environ.get("LS_MTP_FORCE_ARM", "")
        gov = SpecRoundGovernor.from_env()

        while True:
            check_cancel()
            replay = fed - state
            # A draft round must be rewindable to an anchor that ALSO
            # keeps the MTP catch-up hiddens reachable (restore target
            # <= mtp_next = fed-1, so next round's pass rows cover every
            # position the catch-up needs).
            hw_if_draft = max(hw, fed + len(drafts))
            # A draft round needs a legal restore target CLOSE enough that
            # the post-rejection replay keeps the next pass within the
            # 8-row spec_verify cap: a >= fed-4 bounds replay at 6.
            want_draft = bool(drafts) and any(
                fed - 4 <= a <= fed
                and (a % kp == 0 or a // kp == hw_if_draft // kp)
                for a in anchors)
            if want_draft and not gov.draft_this_round():
                want_draft = False
                stats.gov_plain_rounds += 1
            if force_arm in ("1row", "always_rewind"):
                want_draft = False
                drafts = []
            if (not force_arm and not want_draft and fed - state == 1
                    and state == hw):
                # Draftless single-row round AT the high-water (a plain
                # step below it would be a rewind re-feed, whose pool-
                # safety exemption only the spec-verify shape carries) —
                # the PLAIN fast path is
                # command-identical (same decode step, graphs engaged) and
                # its head norms scratch row 0 = position `state` — exactly
                # the prev_src=1 row the MTP catch-up below consumes.
                t_round0 = time.monotonic()
                rp = self._with_pool_evict_retry(
                    lambda: b.decode_step_fetch_and_run(
                        committed[state], seq_id, state, None))
                self._guard(rp)
                pos0 = state
                R = 1
                argmax = [rp.sampled_token]
                try:
                    b.kda_snapshot(seq_id, state + 1)
                    a0 = state + 1
                    slot = 0 if a0 % kp == 0 else 1
                    anchors = [x for x in anchors
                               if (0 if x % kp == 0 else 1) != slot]
                    anchors.append(a0)
                except BridgeError:
                    pass  # no anchor -> draftless rounds (graceful)
                replay = 1
                j = 0
                new_toks = [rp.sampled_token]
                committed.extend(new_toks)
                fed_new = fed + 1
                state = state + 1
                stats.rounds += 1
                gov.note_plain(rp.timings.total_ms)
                if not emit(new_toks[0]):
                    return
                d1 = -1
                try:
                    for pp in range(mtp_next, fed_new - 1):
                        prev_row = pp - pos0
                        if prev_row < 0 or prev_row >= R:
                            raise BridgeError(
                                f"mtp catch-up hidden unavailable: pos "
                                f"{pp} outside [{pos0}, {pos0 + R})")
                        r = b.mtp_row(committed[pp + 1], seq_id, pp,
                                      prev_row, head=(pp == fed_new - 2))
                        if pp == fed_new - 2:
                            d1 = r
                    mtp_next = fed_new - 1
                    if d1 >= 0 and gamma >= 2:
                        d2 = b.mtp_chain_row(d1, seq_id, fed_new - 1)
                        drafts = [d1, d2]
                    elif d1 >= 0:
                        drafts = [d1]
                    else:
                        drafts = []
                except BridgeError as e:
                    self._note_spec_fallback(req, stats, fed_new, e)
                    drafts = []
                hw = max(hw, state)
                fed = fed_new
                continue
            rows = committed[state:fed] + (drafts if want_draft else [])
            drafts_in = list(drafts)
            pos0 = state
            R = len(rows)
            # Anchors carry TWO roles (engine slot 0/1 keyed by pool
            # alignment): a rolling POOL-BOUNDARY anchor (always a legal
            # rewind start — a completed pool recomposes only from its
            # first member) and a per-round anchor after the last replay
            # row (legal only while its pool is still the frontier pool).
            # Restore picks the deepest LEGAL candidate; boundary anchors
            # bound the replay at kp+gamma rows.
            mask = 1 << (replay - 1)          # per-round anchor at fed
            # Boundary anchors from COMMITTED (replay) rows only — a
            # boundary snapshot on a draft row would sit in rejected
            # territory and evict the always-legal slot-0 anchor.
            for i in range(min(replay, 8)):
                if (pos0 + i + 1) % kp == 0:  # boundary anchor row(s)
                    mask |= 1 << i
            t_round0 = time.monotonic()
            # Debug probe: at the named position, capture the PLAIN step's
            # argmax at the same state (anchor+restore makes the double
            # run legal), then run the pass and compare row 0.
            probe_at = os.environ.get("LS_MTP_ROW0_PROBE")
            if probe_at and int(probe_at) == pos0:
                b.kda_snapshot(seq_id, pos0)
                rp0 = b.decode_step_fetch_and_run(rows[0], seq_id, pos0,
                                                  None)
                print(f"  [row0-probe] pos0={pos0} plain argmax "
                      f"{rp0.sampled_token}", flush=True)
                b.kda_restore(seq_id, pos0)
            argmax, _ = b.mtp_verify_pass(rows, seq_id, pos0, mask)
            if probe_at and int(probe_at) == pos0:
                print(f"  [row0-probe] verify R={R} row0 argmax "
                      f"{argmax[0]}", flush=True)
            t_verify = time.monotonic()
            # Mirror the engine's two named slots: slot 0 = the newest
            # boundary anchor covered by this pass; slot 1 = the per-round
            # anchor (fed) when it is NOT itself a boundary.
            new_anchors = [pos0 + i + 1 for i in range(min(replay, 8))
                           if (mask >> i) & 1]
            for a0 in new_anchors:
                slot = 0 if a0 % kp == 0 else 1
                anchors = [x for x in anchors
                           if (0 if x % kp == 0 else 1) != slot]
                anchors.append(a0)
            # Acceptance walk (greedy): draft k at row replay+k-1's
            # position+1; its check is the PREVIOUS row's argmax.
            j = 0
            if want_draft:
                while j < len(drafts) and argmax[replay - 1 + j] == drafts[j]:
                    j += 1
            new_toks = (drafts[:j] if want_draft else [])                 + [argmax[replay - 1 + j]]
            committed.extend(new_toks)
            fed_new = fed + len(new_toks)
            rejected = want_draft and j < len(drafts)
            hw_new = max(hw, pos0 + R)
            if rejected:
                # Legal rewind starts only: a pool boundary, or a position
                # still inside the FRONTIER pool (raw tail intact). The
                # catch-up below runs before the next pass and uses THIS
                # pass's scratch rows, so the only other bound is that the
                # next pass keeps >= 1 committed row: a <= fed_new - 1.
                cands = [a for a in anchors
                         if a <= fed_new - 1
                         and (a % kp == 0 or a // kp == hw_new // kp)]
                a = max(cands)  # exists: want_draft gated on one below
                b.kda_restore(seq_id, a)
                state = a
                anchors = [x for x in anchors if x <= a]
            else:
                state = pos0 + R
            hw = hw_new
            wall_ms = (time.monotonic() - t_round0) * 1e3
            stats.rounds += 1
            if want_draft:
                stats.proposed += len(drafts)
                stats.accepted += j
                gov.note_round(len(new_toks), wall_ms)
            elif R == 1:
                gov.note_plain(wall_ms)
            stop = False
            for tok in new_toks:
                if not emit(tok):
                    stop = True
                    break
            if stop:
                return
            # ── MTP rows: catch-up (correct-token back-fill, target
            # hiddens from THIS pass) + depth-2 chain (recycled hidden).
            d1 = -1
            try:
                for pp in range(mtp_next, fed_new - 1):
                    prev_row = pp - pos0
                    if prev_row < 0 or prev_row >= R:
                        raise BridgeError(
                            f"mtp catch-up hidden unavailable: pos {pp} "
                            f"outside pass rows [{pos0}, {pos0 + R})")
                    r = b.mtp_row(committed[pp + 1], seq_id, pp, prev_row,
                                  head=(pp == fed_new - 2))
                    if pp == fed_new - 2:
                        d1 = r
                mtp_next = fed_new - 1
                if d1 >= 0 and gamma >= 2:
                    d2 = b.mtp_chain_row(d1, seq_id, fed_new - 1)
                    if os.environ.get("LS_MTP_FORCE_ARM") == "chain_run_d1":
                        drafts = [d1]  # bisect: chain side-effects, R=2
                    elif os.environ.get("LS_MTP_FORCE_ARM") == "d2_wrong":
                        drafts = [d1, 0]  # bisect: R=3, d2 never accepted
                    else:
                        drafts = [d1, d2]
                elif d1 >= 0:
                    drafts = [d1]
                else:
                    drafts = []
            except BridgeError as e:
                # Draft-side failure only — committed state intact.
                # Degrade to draftless rounds (plain-equivalent).
                self._note_spec_fallback(req, stats, fed_new, e)
                drafts = []
            if dbg:
                t_end = time.monotonic()
                print(f"  [mtp-round] R={R} replay={replay} pos0={pos0} "
                      f"draft={int(want_draft)} j={j} "
                      f"commit={len(new_toks)} toks={new_toks} "
                      f"drafts_in={drafts_in} argmax={argmax} "
                      f"rej={int(rejected)} "
                      f"verify={1e3 * (t_verify - t_round0):.1f}ms "
                      f"mtp={1e3 * (t_end - t_verify):.1f}ms "
                      f"wall={1e3 * (t_end - t_round0):.1f}ms", flush=True)
            fed = fed_new

    def _decode_speculative(self, req: InferenceRequest, seq_id: int,
                            prompt: list[int], emit, check_cancel,
                            stats: RequestStats) -> None:
        """The champion greedy arm: DSpark overlap rounds (draft hidden
        under an always-committing plain step) + batched verify + DSP-9
        truncation.  Round structure mirrors
        bridge.spec_decode.run_speculative_loop (the C++-parity harness);
        serving deltas: EOS/limit stop via emit(), cancel checks, no
        forced-trajectory machinery."""
        b = self.bridge
        gamma = self.spec.gamma
        with_conf = self.spec.conf_thresh > 0.0
        conf_thresh = self.spec.conf_thresh

        # ── LS_ORCH_ROUND_CSV (serving-gap ledger): one row per seed step /
        # overlap macro-round with the per-section walls — OFF unless the
        # env var names a file (zero champion overhead: the timings written
        # already exist).  Columns mirror bridge.spec_decode's
        # LS_BRIDGE_ROUND_CSV so the pairing analyzer applies unchanged.
        round_csv = None
        rc = os.environ.get("LS_ORCH_ROUND_CSV")
        if rc:
            round_csv = open(rc, "a")
            round_csv.write(
                "kind,round,wall_ms,plain_total,plain_embed,plain_layers,"
                "plain_head,plain_sample,draft_exposed,verify_total,"
                "verify_embed,verify_layers,verify_head,gap_ms,j,g_use\n")

        # Seed feed: one plain step arms the draft context (aux export).
        # Evict-retry: no draft is in flight yet, so a pool-exhausted step
        # (side-tier/indexer growth) is safely re-issued after freeing a
        # holder.  Mid-round steps are wrapped too (spec_round=True,
        # TD-SPEC-ROUND-POOL-EVICT resolved): the eviction's SEQ_FREE under
        # a pending draft rides the bridge wait() dspark stash — see the
        # seam docstring for the exactness argument.
        check_cancel()
        r0 = self._with_pool_evict_retry(
            lambda: b.decode_step_fetch_and_run(prompt[-1], seq_id,
                                                len(prompt) - 1, None))
        self._guard(r0)
        anchor = r0.sampled_token
        fed = len(prompt)
        # SpecRoundGovernor (TD-DSPARK-CTX-POLICY acceptance axis): the
        # seed step is the one guaranteed-clean plain step of the request
        # — it bootstraps the plain baseline.
        gov = SpecRoundGovernor.from_env()
        gov.note_plain(r0.timings.total_ms)
        if round_csv is not None:
            t0 = r0.timings
            round_csv.write(
                f"seed,0,{t0.total_ms:.6f},{t0.total_ms:.6f},"
                f"{t0.embedding_ms:.6f},"
                f"{sum(t0.attention_ms) + sum(t0.moe_ms):.6f},"
                f"{t0.output_head_ms:.6f},{t0.sample_ms:.6f},"
                f"0,0,0,0,0,0,0,0\n")
        if not emit(anchor):
            if round_csv is not None:
                round_csv.close()
            return

        try:
            while True:
                check_cancel()
                if not gov.draft_this_round():
                    # Suspension: decode this round PLAIN — identical
                    # committed token (INV-DSPARK-LOSSLESS), no draft
                    # contention, no verify tax.  A clean step: refresh
                    # the plain baseline.
                    rp = self._with_pool_evict_retry(
                        lambda: b.decode_step_fetch_and_run(
                            anchor, seq_id, fed, None))
                    self._guard(rp)
                    anchor = rp.sampled_token
                    fed += 1
                    gov.note_plain(rp.timings.total_ms)
                    stats.rounds += 1
                    stats.gov_plain_rounds += 1
                    if not emit(anchor):
                        return
                    continue
                t_round0 = time.monotonic()
                # Draft async UNDER the always-committing plain step.
                b.dspark_send_async(seq_id, anchor, fed, gamma)
                # Mid-round evict-retry (TD-SPEC-ROUND-POOL-EVICT): the
                # draft is in flight — see the seam docstring.
                rr = self._with_pool_evict_retry(
                    lambda: b.decode_step_fetch_and_run(anchor, seq_id, fed,
                                                        None),
                    spec_round=True)
                self._guard(rr)
                t = rr.sampled_token
                fed += 1
                committed = 1
                cont = emit(t)
                tw = time.monotonic() if round_csv is not None else 0.0
                try:
                    draft, confs = b.dspark_collect_async(gamma, with_conf)
                except DsparkDraftError as e:
                    # Draft-side failure only — target state (and the
                    # committed token t) intact.  Fall back to the plain
                    # arm for the remainder (INV-SERVE-SPEC-FALLBACK).
                    self._note_spec_fallback(req, stats, fed, e)
                    if not cont:
                        return
                    self._decode_plain_from(req, seq_id, t, fed, emit,
                                            check_cancel, stats)
                    return
                draft_exposed = ((time.monotonic() - tw) * 1e3
                                 if round_csv is not None else 0.0)
                if not cont:
                    return
                # DSP-9 truncation over the REMAINING slots (slot 0 is
                # free against the plain result).
                g_use = gamma
                if with_conf:
                    cum, g_use = 1.0, 1
                    while g_use < gamma:
                        cum *= confs[g_use]
                        if cum < conf_thresh:
                            break
                        g_use += 1
                j = 0
                round_verify_ms = v_embed = v_layers = v_head = 0.0
                if t == draft[0] and g_use >= 2:
                    row_toks = [t] + draft[1:g_use]
                    # Draft already collected — no async dspark pending;
                    # the verify chunk fail-closes at provisioning, so the
                    # evict-retry re-issues identical rows (exact).
                    vr = self._with_pool_evict_retry(
                        lambda: b.verify_step_fetch_and_run(row_toks, seq_id,
                                                            fed, None),
                        spec_round=True)
                    self._guard(vr)
                    if round_csv is not None:
                        round_verify_ms = vr.timings.total_ms
                        v_embed = vr.timings.embedding_ms
                        v_layers = (sum(vr.timings.attention_ms)
                                    + sum(vr.timings.moe_ms))
                        v_head = vr.timings.output_head_ms
                    j2 = 0
                    while (j2 + 1 < g_use
                           and vr.argmax[j2] == draft[j2 + 1]):
                        j2 += 1
                    fed += j2 + 1
                    committed += j2 + 1
                    anchor = vr.argmax[j2]       # bonus
                    j = 1 + j2
                    stop = False
                    for tok in draft[1:j2 + 1]:
                        if not emit(tok):
                            stop = True
                            break
                    if not stop and not emit(anchor):
                        stop = True
                    if stop:
                        stats.rounds += 1
                        stats.proposed += g_use
                        stats.accepted += j
                        return
                else:
                    if t == draft[0]:
                        j = 1
                    anchor = t
                stats.rounds += 1
                stats.proposed += g_use
                stats.accepted += j
                gov.note_round(committed,
                               (time.monotonic() - t_round0) * 1e3)
                if round_csv is not None:
                    wall = (time.monotonic() - t_round0) * 1e3
                    tp = rr.timings
                    round_csv.write(
                        f"overlap,{stats.rounds},{wall:.6f},"
                        f"{tp.total_ms:.6f},{tp.embedding_ms:.6f},"
                        f"{sum(tp.attention_ms) + sum(tp.moe_ms):.6f},"
                        f"{tp.output_head_ms:.6f},{tp.sample_ms:.6f},"
                        f"{draft_exposed:.6f},{round_verify_ms:.6f},"
                        f"{v_embed:.6f},{v_layers:.6f},{v_head:.6f},"
                        f"{wall - tp.total_ms - draft_exposed - round_verify_ms:.6f},"
                        f"{j},{g_use}\n")
        finally:
            stats.gov_probe_rounds = gov.probes
            b.drain_pending_dspark(gamma)
            if round_csv is not None:
                round_csv.close()

    # ── sampled speculation (TD-ORCH-SAMPLED-SPEC) ──────────────────────

    @staticmethod
    def _step_seed(seed: int, pos: int) -> int:
        """Independent, reproducible engine Philox key for one plain
        sampled step: splitmix64 finalizer over (request seed, position).
        u64-wrapped (Command.random_seed is uint64)."""
        z = (seed + pos * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 30
        z = (z * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 27
        z = (z * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
        return z

    @staticmethod
    def _target_probs(logits: np.ndarray, temperature: float,
                      top_p: float, top_k: int,
                      mask: np.ndarray | None = None) -> np.ndarray:
        """The TARGET distribution p (float64 [vocab]) that defines
        "distribution-lossless": standard temperature/top-k/top-p over
        one raw logits row — logits/T → softmax → keep the top_k largest
        (ties at the k-th value all kept) → nucleus: smallest
        descending-order prefix with cumulative mass >= top_p (crossing
        token INCLUDED) → renormalize.  temperature <= 0 → one-hot
        argmax (lowest index on ties — matching the engine argmax
        kernel's packed-score tie-break), so the T→0 limit of every
        consumer recovers the exact greedy arm.

        ``mask`` (guided composition, INV-GUIDED-1): boolean allow-vector
        applied to the logits BEFORE the transform — the accept rule then
        runs against the masked target distribution.  The transform
        mirrors server.guided.GuidedState._sample so guided and
        speculative sampling agree on the target."""
        x = logits.astype(np.float64)
        if mask is not None:
            x = np.where(mask, x, -np.inf)
        if temperature <= 0.0:
            p = np.zeros(x.size)
            p[int(np.argmax(x))] = 1.0
            return p
        x = x / temperature
        if top_k and 0 < top_k < x.size:
            kth = np.partition(x, -top_k)[-top_k]
            x = np.where(x >= kth, x, -np.inf)
        x = x - x.max()
        p = np.exp(x)
        p /= p.sum()
        if 0.0 < top_p < 1.0:
            order = np.argsort(-p, kind="stable")
            csum = np.cumsum(p[order])
            cut = int(np.searchsorted(csum, top_p) + 1)
            keep = np.zeros(p.size, dtype=bool)
            keep[order[:cut]] = True
            p = np.where(keep, p, 0.0)
            p /= p.sum()
        return p

    def _read_logits_rows(self, n: int) -> np.ndarray:
        """Copy the first n rows of the engine's pinned full-logits
        readback region (written by an OUTPUT_HEAD readback_logits step;
        host-visible once its completion fired — B=1 serialization makes
        the read race-free).  Returns float32 [n, vocab]."""
        addr = self.bridge.logits_host_addr
        if not addr:
            raise BridgeError("logits readback region unavailable")
        if n > self.bridge.logits_host_rows:
            raise BridgeError(
                f"logits readback rows {n} > region capacity "
                f"{self.bridge.logits_host_rows}")
        vocab = self.bridge.vocab_size
        buf = (ctypes.c_float * (n * vocab)).from_address(addr)
        return np.frombuffer(buf, dtype=np.float32).copy().reshape(n, vocab)

    def _decode_speculative_sampled(self, req: InferenceRequest,
                                    seq_id: int, prompt: list[int], emit,
                                    check_cancel,
                                    stats: RequestStats) -> None:
        """DISTRIBUTION-LOSSLESS sampled speculation — the champion
        overlap round shape with every committing argmax replaced by an
        exact host-side sample from the target distribution (engine
        multi-row full-logits readback + one np RNG stream seeded per
        request).

        THE MATH.  DSpark proposes DETERMINISTICALLY: each draft slot is
        the draft head's argmax (dspark_runtime.cpp sample chain); the
        DSP-6 confidence c_k is a trained SURVIVAL estimator, not a
        proposal probability, and the full draft distribution is not
        read back.  The proposal q_k is therefore a point mass at d_k,
        and the exact leftover/rejection rule (vLLM RejectionSampler,
        NO_DRAFT_PROBS arm — accept d_k iff u < p_k(d_k)/1, else sample
        the residual p_k with d_k masked out, renormalized; port after
        ref/vllm vllm/v1/sample/rejection_sampler.py, Apache-2.0)
        degenerates to the EQUALITY COUPLING implemented here: draw
        s_k ~ p_k and accept d_k iff s_k == d_k.  P(accept) = p_k(d_k),
        and conditioned on rejection s_k is distributed exactly as the
        residual p_k(x)/(1-p_k(d_k)) over x != d_k — the same joint law,
        one draw, no separate residual sampling.  Every committed token
        (anchor step, accepted slots, mismatch replacement, bonus) is an
        exact sample from its target conditional, so the output
        distribution equals plain sampled AR at every position — for ANY
        draft quality and ANY truncation (DSP-9 / row-capacity clamps
        only shorten the proposal).  T→0 turns p into the one-hot argmax
        and this arm degenerates to the greedy champion arm's logic.

        The RNG advances exactly once per committed token, in order, so
        a fixed seed yields the SAME trajectory as sequential host-
        sampled AR decode — speculation changes wall time, never tokens
        (the sampled analog of INV-DSPARK-LOSSLESS; INV-SAMPLED-SPEC).

        Logprobs ride the same readback rows for free (raw pre-transform
        distribution, matching the plain arm's semantics)."""
        b = self.bridge
        gamma = self.spec.gamma
        with_conf = self.spec.conf_thresh > 0.0
        conf_thresh = self.spec.conf_thresh
        smp = req.sampling
        want_lp = req.logprobs is not None
        rng = np.random.default_rng(smp.seed)
        # Every verify row's logits must fit the pinned readback region
        # (the overlap chunk has g_use <= gamma rows).
        rows_cap = max(1, min(gamma, b.logits_host_rows))

        def pick(logits: np.ndarray) -> tuple[int, StepLogprobs | None]:
            """One exact target-distribution sample (+ logprobs)."""
            p = self._target_probs(logits, smp.temperature, smp.top_p,
                                   smp.top_k)
            tok = int(rng.choice(p.size, p=p))
            lp = (self._step_logprobs(logits, tok, req.logprobs)
                  if want_lp else None)
            return tok, lp

        # Seed feed: one plain step (host-side pick) arms the draft
        # context.  logits_readback=1 skips CMD_SAMPLE_TOKENS entirely.
        # Evict-retry: no draft in flight yet; mid-round steps below are
        # wrapped with spec_round=True (TD-SPEC-ROUND-POOL-EVICT resolved
        # — see the seam docstring).
        check_cancel()
        r0 = self._with_pool_evict_retry(
            lambda: b.decode_step_fetch_and_run(prompt[-1], seq_id,
                                                len(prompt) - 1, None,
                                                logits_readback=True))
        self._guard(r0)
        anchor, lp0 = pick(self._read_logits())
        fed = len(prompt)
        # SpecRoundGovernor (TD-DSPARK-CTX-POLICY acceptance axis).
        gov = SpecRoundGovernor.from_env()
        gov.note_plain(r0.timings.total_ms)
        if not emit(anchor, lp0):
            return

        try:
            while True:
                check_cancel()
                if not gov.draft_this_round():
                    # Suspension: one plain host-sampled step — trajectory
                    # unchanged (INV-SAMPLED-SPEC: one RNG draw per
                    # committed token, in order).
                    rp = self._with_pool_evict_retry(
                        lambda: b.decode_step_fetch_and_run(
                            anchor, seq_id, fed, None,
                            logits_readback=True))
                    self._guard(rp)
                    fed += 1
                    anchor, lp_p = pick(self._read_logits())
                    gov.note_plain(rp.timings.total_ms)
                    stats.rounds += 1
                    stats.gov_plain_rounds += 1
                    if not emit(anchor, lp_p):
                        return
                    continue
                t_round0 = time.monotonic()
                # Draft async UNDER the always-committing sampled step.
                b.dspark_send_async(seq_id, anchor, fed, gamma)
                # Mid-round evict-retry: draft in flight (seam docstring).
                # The RNG draw happens only after the step succeeds, so a
                # retry never advances the sample stream (INV-SAMPLED-SPEC).
                rr = self._with_pool_evict_retry(
                    lambda: b.decode_step_fetch_and_run(anchor, seq_id, fed,
                                                        None,
                                                        logits_readback=True),
                    spec_round=True)
                self._guard(rr)
                t, lp_t = pick(self._read_logits())
                fed += 1
                committed = 1
                cont = emit(t, lp_t)
                try:
                    draft, confs = b.dspark_collect_async(gamma, with_conf)
                except DsparkDraftError as e:
                    # Draft-side failure — continue HOST-SAMPLED plain AR
                    # with the SAME RNG stream: by INV-SAMPLED-SPEC the
                    # remainder is trajectory-identical to what the
                    # speculative arm would have committed (one draw per
                    # committed token, in order).
                    self._note_spec_fallback(req, stats, fed, e)
                    if not cont:
                        return
                    token = t
                    while True:
                        check_cancel()
                        # Plain remainder (draft dead, none pending) — the
                        # ordinary plain-decode evict-retry applies.
                        r = self._with_pool_evict_retry(
                            lambda: b.decode_step_fetch_and_run(
                                token, seq_id, fed, None,
                                logits_readback=True))
                        self._guard(r)
                        stats.rounds += 1
                        fed += 1
                        token, lp = pick(self._read_logits())
                        if not emit(token, lp):
                            return
                if not cont:
                    return
                # DSP-9 truncation over the REMAINING slots (slot 0 is
                # free against the committed step), then the readback-
                # capacity clamp — both only shorten the proposal, which
                # the accept rule is lossless under.
                g_use = gamma
                if with_conf:
                    cum, g_use = 1.0, 1
                    while g_use < gamma:
                        cum *= confs[g_use]
                        if cum < conf_thresh:
                            break
                        g_use += 1
                g_use = min(g_use, rows_cap)
                j = 0
                if t == draft[0] and g_use >= 2:
                    row_toks = [t] + draft[1:g_use]
                    # No draft pending (collected above); retry is the exact
                    # identical-rows re-issue, before any RNG draw.
                    vr = self._with_pool_evict_retry(
                        lambda: b.verify_step_fetch_and_run(
                            row_toks, seq_id, fed, None,
                            logits_readback=True),
                        spec_round=True)
                    self._guard(vr)
                    rows = self._read_logits_rows(g_use)
                    # Equality-coupled accept walk: draw s from row j2's
                    # target; accept draft[j2+1] iff s == draft[j2+1];
                    # the first mismatch's s IS the residual sample; the
                    # draw after full acceptance is the bonus.  One RNG
                    # draw per committed token, in order.
                    j2 = 0
                    picks: list[tuple[int, StepLogprobs | None]] = []
                    while True:
                        s, lp_s = pick(rows[j2])
                        if j2 + 1 < g_use and s == draft[j2 + 1]:
                            picks.append((s, lp_s))
                            j2 += 1
                            continue
                        break
                    fed += j2 + 1
                    committed += j2 + 1
                    anchor = s               # residual sample or bonus
                    j = 1 + j2
                    stop = False
                    for tok, lp in picks:
                        if not emit(tok, lp):
                            stop = True
                            break
                    if not stop and not emit(s, lp_s):
                        stop = True
                    if stop:
                        return
                else:
                    if t == draft[0]:
                        j = 1
                    anchor = t
                stats.rounds += 1
                stats.proposed += g_use
                stats.accepted += j
                gov.note_round(committed,
                               (time.monotonic() - t_round0) * 1e3)
        finally:
            stats.gov_probe_rounds = gov.probes
            b.drain_pending_dspark(gamma)

    # ── logits readback consumers (guided decoding + logprobs) ──────────

    def _read_logits(self) -> np.ndarray:
        """Copy the engine's pinned full-logits readback row (written by
        the OUTPUT_HEAD readback_logits step; host-visible once its
        completion fired — B=1 serialization makes the read race-free)."""
        addr = self.bridge.logits_host_addr
        if not addr:
            raise BridgeError("logits readback row unavailable "
                              "(engine build without logits_readback_addr?)")
        vocab = self.bridge.vocab_size
        buf = (ctypes.c_float * vocab).from_address(addr)
        return np.frombuffer(buf, dtype=np.float32).copy()

    @staticmethod
    def _step_logprobs(logits: np.ndarray, chosen: int,
                       k: int) -> StepLogprobs:
        """StepLogprobs from one full-logits row: log-softmax over the
        RAW model distribution (float64 accumulation; guided requests
        report pre-mask logprobs — a grammar-repaired token honestly
        shows its unconstrained probability) + top-K sorted descending.
        K = 0 serves the chosen-token logprob only (OpenAI semantics)."""
        x = logits.astype(np.float64)
        m = float(x.max())
        lse = m + math.log(float(np.exp(x - m).sum()))
        tops: tuple[TokenLogprob, ...] = ()
        if k > 0:
            kk = min(k, x.size)
            idx = np.argpartition(x, -kk)[-kk:]
            idx = idx[np.argsort(-x[idx], kind="stable")]
            tops = tuple(TokenLogprob(int(i), float(x[i] - lse))
                         for i in idx)
        return StepLogprobs(
            token=TokenLogprob(int(chosen), float(x[chosen] - lse)),
            top_logprobs=tops)

    # ── guided decoding (TD-SERVE-NAMED-TOOL-CHOICE) ─────────────────────

    def _masked_step(self, guided, tok: int, pos: int, seq_id: int,
                     sampling: SamplingParams, *,
                     spec_round: bool = False) -> tuple[int, np.ndarray]:
        """One plain step with full-logits readback + host-side masked
        pick (grammar-advancing); returns (picked token, RAW logits row —
        pick_and_accept masks a copy, so the row is pre-mask and feeds
        logprobs).  Feeding at pos < the engine high-water mark is the
        standard spec-dec re-feed (INV-DSA-REWIND / INV-DSPARK-REFEED
        position-addressed overwrite).

        The step rides the pool-exhaustion evict-retry seam (guided decode
        grows the same pools as unguided decode); the grammar advance
        (pick_and_accept) runs only after the step SUCCEEDS, so a retry
        never double-advances the matcher.  ``spec_round`` marks the
        mid-speculative-round call sites (TD-SPEC-ROUND-POOL-EVICT)."""
        r = self._with_pool_evict_retry(
            lambda: self.bridge.decode_step_fetch_and_run(
                tok, seq_id, pos, None, logits_readback=True),
            spec_round=spec_round)
        self._guard(r)
        logits = self._read_logits()
        return guided.pick_and_accept(logits, *sampling.as_tuple()), logits

    def _decode_guided_plain(self, req: InferenceRequest, seq_id: int,
                             prompt: list[int], emit, check_cancel,
                             stats: RequestStats, guided) -> None:
        """Grammar-constrained plain AR decode: every step reads the full
        logits back, applies the grammar's token bitmask host-side, and
        picks greedy (temperature 0) or sampled per SamplingParams —
        every emitted token is grammar-valid by construction.  Logprobs
        ride the SAME readback row for free (raw pre-mask distribution)."""
        self._decode_guided_plain_from(req, seq_id, prompt[-1],
                                       len(prompt) - 1, emit, check_cancel,
                                       stats, guided)

    def _decode_guided_plain_from(self, req: InferenceRequest, seq_id: int,
                                  token: int, pos: int, emit, check_cancel,
                                  stats: RequestStats, guided) -> None:
        """The guided plain loop body from an arbitrary (token, pos)
        resume point (spec→plain fallback entry — the grammar matcher
        state already reflects every emitted token, so it carries over
        unchanged; INV-SERVE-SPEC-FALLBACK)."""
        want_lp = req.logprobs is not None
        while True:
            check_cancel()
            token, logits = self._masked_step(guided, token, pos, seq_id,
                                              req.sampling)
            stats.rounds += 1
            pos += 1
            lp = (self._step_logprobs(logits, token, req.logprobs)
                  if want_lp else None)
            if not emit(token, lp):
                return

    def _decode_speculative_guided(self, req: InferenceRequest, seq_id: int,
                                   prompt: list[int], emit, check_cancel,
                                   stats: RequestStats, guided) -> None:
        """Optimistic constrained speculation (the champion greedy arm
        under a grammar).  Round shape mirrors _decode_speculative with
        three guided deltas:

          1. The always-committing plain step (and the seed step) becomes
             a MASKED step — logits readback + host-side masked argmax —
             so the committed anchor token is grammar-valid by
             construction (identical to the unconstrained argmax whenever
             that token satisfies the grammar, i.e. almost always inside
             the free-form think section).
          2. STAGE B violation suppressor: remaining draft slots are
             grammar-truncated (tentative matcher advance + rollback) as a
             second predicate on g_use beside DSP-9 confidence truncation
             — accepted verify rows are therefore always grammar-valid.
          3. The BONUS token (verify argmax at the first mismatch row) is
             the only unconstrained commit candidate left; if the grammar
             rejects it, it is NOT emitted — ONE masked plain step re-feeds
             the same (token, position) and picks the masked replacement.
             Engine-side this is exactly the champion's per-round re-feed
             shape (positions < high-water are position-addressed
             overwrites; INV-DSA-REWIND / INV-DSPARK-REFEED), so no
             rollback IPC is needed.
        """
        b = self.bridge
        gamma = self.spec.gamma
        with_conf = self.spec.conf_thresh > 0.0
        conf_thresh = self.spec.conf_thresh
        smp = req.sampling

        # Seed feed: one masked plain step arms the draft context.
        check_cancel()
        # SpecRoundGovernor (TD-DSPARK-CTX-POLICY acceptance axis): masked
        # steps carry no engine timings — wall-clock the clean steps.
        gov = SpecRoundGovernor.from_env()
        t_seed = time.monotonic()
        anchor, _ = self._masked_step(guided, prompt[-1], seq_id=seq_id,
                                      pos=len(prompt) - 1, sampling=smp)
        gov.note_plain((time.monotonic() - t_seed) * 1e3)
        fed = len(prompt)
        if not emit(anchor):
            return

        try:
            while True:
                check_cancel()
                if not gov.draft_this_round():
                    # Suspension: one plain masked step — the matcher
                    # advances identically (token-lossless either way).
                    t_p = time.monotonic()
                    anchor, _ = self._masked_step(guided, anchor, fed,
                                                  seq_id, smp)
                    gov.note_plain((time.monotonic() - t_p) * 1e3)
                    fed += 1
                    stats.rounds += 1
                    stats.gov_plain_rounds += 1
                    if not emit(anchor):
                        return
                    continue
                t_round0 = time.monotonic()
                # Draft async UNDER the always-committing masked step.
                b.dspark_send_async(seq_id, anchor, fed, gamma)
                # Mid-round evict-retry: draft in flight (seam docstring).
                t, _ = self._masked_step(guided, anchor, fed, seq_id, smp,
                                         spec_round=True)
                fed += 1
                committed = 1
                cont = emit(t)
                try:
                    draft, confs = b.dspark_collect_async(gamma, with_conf)
                except DsparkDraftError as e:
                    # Draft-side failure — continue grammar-constrained
                    # plain decode; the matcher state already reflects
                    # every emitted token (INV-SERVE-SPEC-FALLBACK).
                    self._note_spec_fallback(req, stats, fed, e)
                    if not cont:
                        return
                    self._decode_guided_plain_from(req, seq_id, t, fed,
                                                   emit, check_cancel,
                                                   stats, guided)
                    return
                if not cont:
                    return
                # DSP-9 truncation over the REMAINING slots.
                g_use = gamma
                if with_conf:
                    cum, g_use = 1.0, 1
                    while g_use < gamma:
                        cum *= confs[g_use]
                        if cum < conf_thresh:
                            break
                        g_use += 1
                # STAGE B: grammar truncation — tentatively advance the
                # matcher over draft[1:], keep the valid prefix, rollback.
                if t == draft[0] and g_use >= 2:
                    ok = 0
                    while 1 + ok < g_use and guided.try_accept(draft[1 + ok]):
                        ok += 1
                    guided.rollback(ok)
                    stats.grammar_trunc_slots += g_use - (1 + ok)
                    g_use = 1 + ok
                j = 0
                if t == draft[0] and g_use >= 2:
                    row_toks = [t] + draft[1:g_use]
                    # No draft pending (collected above); exact re-issue.
                    vr = self._with_pool_evict_retry(
                        lambda: b.verify_step_fetch_and_run(row_toks, seq_id,
                                                            fed, None),
                        spec_round=True)
                    self._guard(vr)
                    j2 = 0
                    while (j2 + 1 < g_use
                           and vr.argmax[j2] == draft[j2 + 1]):
                        j2 += 1
                    fed += j2 + 1
                    committed += j2 + 1
                    bonus = vr.argmax[j2]
                    j = 1 + j2
                    stop = False
                    for tok in draft[1:j2 + 1]:
                        if not guided.try_accept(tok):
                            raise BridgeError(
                                "guided: pre-validated draft slot rejected")
                        if not emit(tok):
                            stop = True
                            break
                    if not stop:
                        if guided.try_accept(bonus):
                            if not emit(bonus):
                                stop = True
                            anchor = bonus
                        else:
                            # Violation at the bonus position: masked
                            # re-feed of (row_toks[j2], fed-1) — the
                            # champion re-feed shape, one plain step.
                            stats.grammar_refeeds += 1
                            t2, _ = self._masked_step(guided, row_toks[j2],
                                                      fed - 1, seq_id, smp,
                                                      spec_round=True)
                            if not emit(t2):
                                stop = True
                            anchor = t2
                    if stop:
                        return
                else:
                    if t == draft[0]:
                        j = 1
                    anchor = t
                stats.rounds += 1
                stats.proposed += g_use
                stats.accepted += j
                gov.note_round(committed,
                               (time.monotonic() - t_round0) * 1e3)
        finally:
            stats.gov_probe_rounds = gov.probes
            b.drain_pending_dspark(gamma)

    def _note_spec_fallback(self, req: InferenceRequest,
                            stats: RequestStats, fed: int,
                            err: Exception) -> None:
        """Bookkeeping for a draft-side dspark failure mid-request
        (INV-SERVE-SPEC-FALLBACK): the engine's drafting context is dead
        for this request (it re-arms only on the next position-0 capture,
        i.e. a later request's full prefill) — pessimise the adoption
        mirror so forked successors route plain, record the cut round,
        and log the event server-side (the request itself CONTINUES on
        the plain arm, so no error ever reaches the client)."""
        self._draft_ctx_valid = False
        stats.spec_fallback_round = stats.rounds + 1
        print(f"  [orch] request {req.request_id}: draft failed ({err}) — "
              f"speculative→plain fallback at round {stats.rounds + 1} "
              f"(position {fed}); remainder decodes plain", flush=True)

    @staticmethod
    def _guard(r) -> None:
        if not (math.isfinite(r.top1_prob) and math.isfinite(r.entropy)):
            raise BridgeError("NaN top1_prob/entropy from engine")
