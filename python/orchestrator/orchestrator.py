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

from bridge.ring_bridge import BridgeError, EngineBridge
from orchestrator.types import EngineMetadata, StepLogprobs, TokenLogprob

__all__ = [
    "InferenceRequest",
    "Orchestrator",
    "PrefixCacheConfig",
    "RequestStats",
    "SamplingParams",
    "SpeculationConfig",
]

_PREFILL_CHUNK = 64          # test-proven prefill chunk (spec_decode parity)
# deepseek_v4 prefill chunk (TD-V4-CHUNK-PREFILL): the routed-expert UNION
# saturates toward all 256 experts/layer, so a maximal chunk streams each
# layer's expert set ~once per chunk instead of 6 experts × tokens times.
# 512 = ipc kMaxBatchDescriptors / kMaxSidebandTokenIds = the engine's
# elastic-superchunk floor (prefill_moe_big) = the executor's V4 row bound.
_V4_PREFILL_CHUNK = 512
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
    conf_thresh: float = 0.0                 # 0 = truncation off


@dataclass
class PrefixCacheConfig:
    """serving.prefix_cache (config/schema.json) — basic prompt-prefix KV
    caching via retained SEQ_FORK holder sequences."""
    enabled: bool = True
    max_cached_tokens: int = 8192
    max_entries: int = 8

    @classmethod
    def from_config(cls, cfg: dict) -> "PrefixCacheConfig":
        pc = (cfg.get("serving") or {}).get("prefix_cache") or {}
        return cls(
            enabled=bool(pc.get("enabled", True)),
            max_cached_tokens=int(pc.get("max_cached_tokens", 8192)),
            max_entries=int(pc.get("max_entries", 8)),
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

    @property
    def tok_per_s(self) -> float:
        return (1000.0 * self.tokens / self.decode_wall_ms
                if self.decode_wall_ms > 0 else 0.0)


class _Cancelled(Exception):
    pass


class _PrefixEntry:
    __slots__ = ("tokens", "seq_id", "last_used")

    def __init__(self, tokens: tuple[int, ...], seq_id: int,
                 last_used: int = 0) -> None:
        self.tokens = tokens
        self.seq_id = seq_id
        self.last_used = last_used               # LRU stamp (monotonic)


class PrefixCache:
    """LRU registry of retained PREFIX HOLDER sequences.

    Each entry is a frozen sequence (created by SEQ_FORK from a working
    sequence right after its prompt prefill) whose refcounted CoW pages
    keep the prefix KV resident — "unused but still accessible".  A later
    request whose prompt starts with a registered prefix forks FROM the
    holder and prefills only the delta.  Correctness invariant
    (INV-PREFIX-CACHE-1): cache ON vs OFF is token-identical — the reused
    pages ARE the recomputed bytes; only prefill wall changes.

    Holders are cheap in pages: full pages are shared by refcount; each
    fork materializes only one CoW append-frontier logical page group (KV
    + DSA indexer-K).  Entries are freed (free_sequence) on token/entry
    budget pressure or on fork/pool exhaustion (evict-retry).

    CHAIN-AWARE POLICY (user directive 2026-08-18): entries whose prompts
    nest (kv1 ⊂ kv1..kv2 ⊂ … ⊂ kv1..kvn) form a CHAIN and physically
    SHARE their common-prefix pages (each was forked over the previous
    one's pages, so only an entry's unique tail beyond its longest
    registered ancestor is real memory):
      1. TOUCH PROPAGATES DOWN: a hit on any entry stamps recently-used
         every registered prefix of it — they are all in use as prefixes
         of the used state.
      2. EVICTION is LRU-major (oldest stamp first), DEEPEST-FIRST minor
         (longest prompt first among equal stamps): the tail is cheap to
         reconstruct (delta prefill from the surviving shorter entry) and
         is the only member whose eviction releases real pages — a short
         member's pages stay pinned by its extensions' refs.
      3. The token budget counts UNIQUE tokens (each entry contributes
         len − its longest registered ancestor's len, so a chain costs
         its longest member), matching the CoW physical reality; a naive
         per-entry sum overestimates chains and evicts prematurely.
    """

    def __init__(self, cfg: PrefixCacheConfig, bridge: EngineBridge) -> None:
        self.cfg = cfg
        self._bridge = bridge
        self._entries: list[_PrefixEntry] = []
        self._clock = 0                          # LRU stamp source
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ── chain helpers ────────────────────────────────────────────────────

    @staticmethod
    def _is_prefix_of(p: tuple[int, ...], q: tuple[int, ...]) -> bool:
        return len(p) <= len(q) and q[:len(p)] == p

    def total_unique_tokens(self) -> int:
        """Budget accounting matching the CoW page reality: each entry
        contributes only its unique tail beyond its longest registered
        ancestor (a chain costs the length of its longest member)."""
        total = 0
        for e in self._entries:
            anc = 0
            for p in self._entries:
                if (p is not e and len(p.tokens) < len(e.tokens)
                        and self._is_prefix_of(p.tokens, e.tokens)):
                    anc = max(anc, len(p.tokens))
            total += len(e.tokens) - anc
        return total

    def _victim_order(self) -> list[_PrefixEntry]:
        """Deepest-first within a chain is ABSOLUTE: only chain-MAXIMAL
        entries (no registered extension still present) are eviction
        candidates — a proper prefix outlives all its extensions (its
        pages are pinned by their refs, so evicting it frees nothing and
        forfeits a full re-prefill).  Across candidates: LRU-major
        (oldest stamp first), deepest-first minor."""
        maximal = [e for e in self._entries
                   if not any(p is not e
                              and len(p.tokens) > len(e.tokens)
                              and self._is_prefix_of(e.tokens, p.tokens)
                              for p in self._entries)]
        return sorted(maximal,
                      key=lambda e: (e.last_used, -len(e.tokens)))

    # ── lookup / registration ────────────────────────────────────────────

    def lookup(self, prompt: list[int],
               valid=None) -> _PrefixEntry | None:
        """Longest registered prefix usable for `prompt` (needs at least
        one prompt token beyond the prefix for the seed feed).  ``valid``
        (optional ``len -> bool``) filters candidate entry LENGTHS — the
        superchunk path accepts only grid-aligned or exact-prompt-body
        entries (shape-identity, TD-V4-SERVE-PREFIX)."""
        best: _PrefixEntry | None = None
        for e in self._entries:
            n = len(e.tokens)
            if (n <= len(prompt) - 1 and (valid is None or valid(n))
                    and (best is None or n > len(best.tokens))
                    and tuple(prompt[:n]) == e.tokens):
                best = e
        if best is not None:
            # Chain touch propagation: the used entry AND every registered
            # prefix of it share one fresh stamp.
            self._clock += 1
            for e in self._entries:
                if self._is_prefix_of(e.tokens, best.tokens):
                    e.last_used = self._clock
            self.hits += 1
        else:
            self.misses += 1
        return best

    def has_exact(self, tokens: tuple[int, ...]) -> bool:
        return any(e.tokens == tokens for e in self._entries)

    def register(self, tokens: tuple[int, ...], src_seq_id: int,
                 holder_seq_id: int) -> bool:
        """Fork src → a frozen holder and register it. Returns False when
        registration is skipped (duplicate/empty/over-sized prefix) or the
        fork fails even after eviction (caller proceeds uncached)."""
        if (not tokens or self.has_exact(tokens)
                or len(tokens) > self.cfg.max_cached_tokens):
            return False
        if not self._fork_with_evict_retry(src_seq_id, holder_seq_id,
                                           protect=None):
            return False
        self._clock += 1
        entry = _PrefixEntry(tokens, holder_seq_id, self._clock)
        self._entries.append(entry)
        self._evict_over_budget(protect=entry)
        return True

    def fork_from(self, entry: _PrefixEntry, dst_seq_id: int) -> None:
        """Fork a working sequence from a holder (cache hit). Raises
        BridgeError when the pool stays exhausted after evicting every
        other entry."""
        if not self._fork_with_evict_retry(entry.seq_id, dst_seq_id,
                                           protect=entry):
            raise BridgeError("prefix-cache fork: pool exhausted after "
                              "eviction")

    # ── eviction ─────────────────────────────────────────────────────────

    def _evict_over_budget(self, protect: _PrefixEntry | None) -> None:
        while self._entries and (len(self._entries) > self.cfg.max_entries
                                 or (self.total_unique_tokens()
                                     > self.cfg.max_cached_tokens)):
            if not self._evict_one(protect=protect):
                return                           # only `protect` remains

    def _evict_one(self, protect: _PrefixEntry | None) -> bool:
        for e in self._victim_order():           # LRU-major, deepest-first
            if e is protect:
                continue
            self._entries.remove(e)
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

    def _free_holder(self, e: _PrefixEntry) -> None:
        self.evictions += 1
        self._bridge.free_sequence(e.seq_id)     # best-effort inside

    def clear(self) -> None:
        while self._entries:
            self._free_holder(self._entries.pop())

    def _fork_with_evict_retry(self, src: int, dst: int,
                               protect: _PrefixEntry | None) -> bool:
        while True:
            try:
                self._bridge.fork_sequence(src, dst)
                return True
            except BridgeError as err:
                if "exhausted" not in str(err):
                    raise
                if not self._evict_one(protect):
                    return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """B=1 serving orchestrator over EngineBridge (see module docstring).

    Production boot: ``Orchestrator.boot(config_path)``.  Tests inject a
    ready bridge + metadata directly through ``__init__``.
    """

    def __init__(self, bridge: EngineBridge, *,
                 metadata: EngineMetadata,
                 speculation: SpeculationConfig | None = None,
                 prefix_cache: PrefixCacheConfig | None = None,
                 engine_module: Any = None,
                 teacher_forced_prefill: bool = False,
                 prefill_chunk: int = _PREFILL_CHUNK,
                 chunk_prefill_arms_draft: bool = True,
                 prefill_superchunk: bool = False,
                 prefill_superchunk_stride: int = 0) -> None:
        self.bridge = bridge
        self.metadata = metadata
        self.spec = speculation or SpeculationConfig()
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
        # V4 boots enable it (with prefill_moe_big); GLM serving uses it
        # with a bounded STRIDE (below) since 2026-08-23.
        self._prefill_superchunk = bool(prefill_superchunk)
        # Superchunk STRIDE: tokens per superchunk (also the prefix-cache
        # registration grid on the sc path).  0 = the engine's elastic
        # moe_batch_capacity (V4).  GLM serving on the EP4 champion
        # topology must bound it at the engine's single-shot MoE chunk
        # capacity — max(compute.moe_big_chunk_tokens, orchestrator.
        # max_batch_size) — because CHUNKED MoE batches are rejected with
        # expert-only ranks resident (TD-MOE-EP-XTP-WAVES) and the layer's
        # output would be WRONG; at or below the capacity MOE_BIG runs the
        # byte-identical single-shot pipeline (INV-MOE-BIG-1).
        self._prefill_sc_stride = max(0, int(prefill_superchunk_stride))
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
        self.prefix_cache: PrefixCache | None = (
            PrefixCache(pc_cfg, bridge) if pc_cfg.enabled else None)
        self._engine = engine_module          # owned iff boot() created it
        self._queue: collections.deque[InferenceRequest] = collections.deque()
        self._queue_lock = threading.Lock()
        self._wake = threading.Event()
        self._cancelled: set[int] = set()
        self._shutdown = False
        self._next_seq_id = 1
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

        deepseek_v4 configs boot ARCHITECTURE-AWARE: split/ACT command
        flow (no FAR, no REEF), prefix cache OFF, first_moe_layer 0
        (all-MoE), CHUNKED prefill at 512-token chunks (TD-V4-CHUNK-
        PREFILL resolved 2026-08-21 — the expert-fetch union amortizes
        the streaming wall), 200 s expert-fetch deadline."""
        import json

        if engine_module is None:
            import layerstorm_engine as engine_module
        with open(config_path) as f:
            cfg = json.load(f)

        info = (engine_module.start_engine_test(config_path) if test_engine
                else engine_module.start_engine(config_path))
        try:
            model = cfg.get("model") or {}
            # Architecture-aware boot: deepseek_v4's landed engine path is
            # the SPLIT command sequence (EMBEDDING → per-layer
            # RUN_ATTENTION + FETCH_AND_RUN_MOE with static e%tp entries →
            # OUTPUT_HEAD → SAMPLE — the ticket-H golden harness shape,
            # which the bridge's split/ACT arm speaks byte-identically).
            # The FAR fused command and the REEF service are MLA/GLM-shaped
            # seams; V4 forces use_far=False + route_arm "act".
            is_v4 = str(model.get("architecture") or "") == "deepseek_v4"
            # TD-VOCAB-AUTODETECT: prefer the engine's resolved vocab width
            # (EngineInfo.vocab_size — weights-derived when the config field
            # is 0/absent, cross-checked otherwise). The config dict read is
            # the fallback for engine builds predating the field.
            vocab = int(getattr(info, "vocab_size", 0) or 0) \
                or int(model.get("vocab_size", 0))
            # V4-Flash is all-MoE (first_k_dense_replace 0): the bridge's
            # dense branch must never fire.
            first_moe = 0 if is_v4 \
                else int(model.get("first_k_dense_replace", 3))

            buffer_ids = engine_module.query_buffer_ids()
            hidden_buf = logits_buf = 0
            for name, bid in buffer_ids.items():
                if name.startswith("hidden_state.attn.rank0"):
                    hidden_buf = int(bid)
                elif name.startswith("logits_scratch.pos0"):
                    logits_buf = int(bid)
            if not (hidden_buf and logits_buf) and not test_engine:
                raise BridgeError("hidden/logits buffers not found")

            reef_ok = bool((cfg.get("gpu_loader") or {}).get("enabled")) \
                and not test_engine and not is_v4
            bridge = EngineBridge(info, vocab_size=vocab,
                                  first_moe_layer=first_moe,
                                  hidden_buf_id=hidden_buf,
                                  logits_buf_id=logits_buf,
                                  route_arm="reef" if reef_ok else "act",
                                  use_far=not is_v4)
            if is_v4:
                # Golden-harness fetch deadline: V4 streams ~3.4 GB of
                # routed experts per token from the GGUF page cache; a
                # cold layer can exceed the 5 s champion default.
                bridge.decode_timeout_us = 200_000_000
                # Ticket J: with the dspark draft on a second GPU, routed
                # experts stay on the tp GPUs (position 0) — an e%num_gpus
                # spread would fork the ticket-H golden trajectory (EP
                # combine order) and target the draft GPU.
                tp_arr = (cfg.get("hardware") or {}).get("tp_array") or [0]
                bridge.moe_gpus = max(1, len(tp_arr))
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
            if (spec_cfg.get("enabled")
                    and spec_cfg.get("method") == "dspark"
                    and speculation_depth != 0):
                ds = spec_cfg.get("dspark") or {}
                gamma = int(ds.get("speculative_tokens",
                                   max(1, int(ds.get("block_size", 8)) - 1)))
                if speculation_depth is not None:
                    gamma = min(gamma, speculation_depth)
                spec = SpeculationConfig(
                    enabled=gamma >= 1, gamma=gamma,
                    conf_thresh=(conf_thresh
                                 if ds.get("confidence_enabled") else 0.0))

            meta = EngineMetadata(
                num_gpus=info.num_gpus,
                num_moe_layers=info.num_moe_layers,
                num_experts=info.num_experts,
                num_layers=info.num_layers,
                expert_bytes=getattr(info, "expert_bytes", 0),
                kv_bytes_per_page=getattr(info, "kv_bytes_per_page", 0),
                eos_token_ids=tuple(eos_token_ids),
                hidden_buf_id=hidden_buf,
                logits_buf_id=logits_buf,
                vocab_size=vocab,
                moe_batch_capacity=info.moe_batch_capacity,
                v4_attention_types=tuple(
                    getattr(info, "v4_attention_types", ()) or ()),
            )
            pc_cfg = PrefixCacheConfig.from_config(cfg)
            # TD-V4-SERVE-PREFIX RESOLVED (2026-08-22): CMD_SEQ_FORK now
            # clones complete V4 per-seq state (kMain CoW + side-tier
            # copy-on-fork + executor state rings), so the prefix cache is
            # enabled for V4 like GLM.  The V4 registration grid is the
            # SUPERCHUNK stride (moe_batch_capacity) — see _generate.
            # V4 chunk = 512 rides the engine's elastic-superchunk row
            # sizing (prefill_moe_big, schema default ON — floor 512); a
            # config that turns it off shrinks the executor's V4 row bound
            # to orchestrator.max_batch_size, so fall back to 64 there.
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
            #     V4 (TP-only, no expert-only ranks) keeps the full
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
            sc_stride = 0 if is_v4 else max(
                int(comp.get("moe_big_chunk_tokens") or 512),
                int((cfg.get("orchestrator") or {})
                    .get("max_batch_size") or 64))
            se = os.environ.get("LS_ORCH_SC_STRIDE")  # diagnostic override
            if se and se.isdigit() and int(se) > 0:
                sc_stride = int(se)
            chunk = _V4_PREFILL_CHUNK if is_v4 and moe_big \
                else _PREFILL_CHUNK
            ce = os.environ.get("LS_ORCH_PREFILL_CHUNK")
            if ce and ce.isdigit() and int(ce) > 0:
                chunk = min(int(ce), 512)
            orch = cls(bridge, metadata=meta, speculation=spec,
                       prefix_cache=pc_cfg,
                       engine_module=engine_module,
                       prefill_chunk=chunk,
                       chunk_prefill_arms_draft=True,
                       prefill_superchunk=sc_arm,
                       prefill_superchunk_stride=sc_stride)
            orch.info = info                  # EngineInfo (reporting)
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
                self._wake.wait(timeout=0.05)
                self._wake.clear()

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
        try:
            tokens, reason, stats, logprobs = self._generate(req)
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
        self.last_stats = stats
        # Server-side per-request performance line (serving-gap ledger):
        # SUSTAINED decode tok/s = committed tokens / decode wall — prefill
        # and queueing excluded by construction (RequestStats.tok_per_s).
        acc = (stats.accepted / stats.proposed) if stats.proposed else 0.0
        print(f"  [orch-stats] req={req.request_id} tokens={stats.tokens} "
              f"finish={reason} rounds={stats.rounds} "
              f"acc={acc:.4f} ({stats.accepted}/{stats.proposed}) "
              f"prefix_hit={stats.prefix_hit_tokens} "
              f"prefill_ms={stats.prefill_ms:.1f} "
              f"decode_ms={stats.decode_wall_ms:.1f} "
              f"decode_tok_s={stats.tok_per_s:.3f}", flush=True)
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

    def _generate(self, req: InferenceRequest
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
        entry = (self.prefix_cache.lookup(
                     prompt,
                     valid=((lambda n: n % grid == 0 or n == pre)
                            if sc_path else None))
                 if self.prefix_cache is not None else None)
        if entry is not None:
            self.prefix_cache.fork_from(entry, seq_id)
            pos = len(entry.tokens)
            stats.prefix_hit_tokens = pos
            # Forked sequences never capture position 0: drafting works
            # only if the engine ADOPTS the tracked context (fork point
            # inside its frontier) — see _draft_ctx_valid in __init__.
            draft_adoptable = (self._draft_ctx_valid
                               and pos <= self._draft_ctx_len)
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
                    self.bridge.create_sequence(seq_id, len(prompt))
                    break
                except BridgeError as err:
                    if ("exhausted" not in str(err)
                            or self.prefix_cache is None
                            or not self.prefix_cache.evict_for_admission()):
                        raise
            pos = 0
            draft_adoptable = True  # position-0 capture re-arms the context
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
                    r = self.bridge.decode_step_fetch_and_run(
                        prompt[pos], seq_id, pos, None)
                    self._guard(r)
                    pos += 1
            elif sc_path:
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
                        holder_id = self._next_seq_id
                        if self.prefix_cache.register(
                                tuple(prompt[:grid_len]), seq_id,
                                holder_id):
                            self._next_seq_id += 1
                    check_cancel()
                    n = min(sc_stride, pre - pos)
                    self.bridge.prefill_superchunk_fetch_and_run(
                        prompt[pos:pos + n], seq_id, pos,
                        self._prefill_chunk)
                    pos += n
                if (self.prefix_cache is not None and pos == grid_len
                        and grid_len > 0):
                    # pre was exactly grid-aligned — register at the end.
                    holder_id = self._next_seq_id
                    if self.prefix_cache.register(tuple(prompt[:grid_len]),
                                                  seq_id, holder_id):
                        self._next_seq_id += 1
                if (self.prefix_cache is not None and pre > 0
                        and pre != grid_len):
                    # EXACT-prompt-body holder: a repeated identical body
                    # (the common shared-prompt case) reuses shapes by
                    # construction even when pre is not superchunk-aligned
                    # — a hit skips the whole prefill.
                    holder_id = self._next_seq_id
                    if self.prefix_cache.register(tuple(prompt[:pre]),
                                                  seq_id, holder_id):
                        self._next_seq_id += 1
            else:
                while pos < pre:
                    if (self.prefix_cache is not None and pos == grid_len
                            and grid_len > 0):
                        holder_id = self._next_seq_id
                        if self.prefix_cache.register(
                                tuple(prompt[:grid_len]), seq_id,
                                holder_id):
                            self._next_seq_id += 1
                    check_cancel()
                    n = min(self._prefill_chunk, pre - pos)
                    self.bridge.prefill_chunk_fetch_and_run(
                        prompt[pos:pos + n], seq_id, pos, None)
                    pos += n
                if (self.prefix_cache is not None and pos == grid_len
                        and grid_len > 0):
                    # pre was exactly grid-aligned — register at the end.
                    holder_id = self._next_seq_id
                    if self.prefix_cache.register(tuple(prompt[:grid_len]),
                                                  seq_id, holder_id):
                        self._next_seq_id += 1
            stats.prefill_ms = (time.monotonic() - t0) * 1e3

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
            spec_ok = (self.spec.enabled and draft_adoptable
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
        base = None if req.sampling.greedy else req.sampling.as_tuple()
        want_lp = req.logprobs is not None
        token = prompt[-1]
        pos = len(prompt) - 1
        while True:
            check_cancel()
            sampling = None
            if base is not None:
                sampling = (base[0], base[1], base[2],
                            self._step_seed(base[3], pos))
            r = self.bridge.decode_step_fetch_and_run(
                token, seq_id, pos, None, sampling=sampling,
                logprobs_readback=want_lp)
            self._guard(r)
            stats.rounds += 1
            pos += 1
            token = r.sampled_token
            lp = (self._step_logprobs(self._read_logits(), token,
                                      req.logprobs) if want_lp else None)
            if not emit(token, lp):
                return

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
        check_cancel()
        r0 = b.decode_step_fetch_and_run(prompt[-1], seq_id,
                                         len(prompt) - 1, None)
        self._guard(r0)
        anchor = r0.sampled_token
        fed = len(prompt)
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
                t_round0 = time.monotonic() if round_csv is not None else 0.0
                # Draft async UNDER the always-committing plain step.
                b.dspark_send_async(seq_id, anchor, fed, gamma)
                rr = b.decode_step_fetch_and_run(anchor, seq_id, fed, None)
                self._guard(rr)
                t = rr.sampled_token
                fed += 1
                cont = emit(t)
                tw = time.monotonic() if round_csv is not None else 0.0
                draft, confs = b.dspark_collect_async(gamma, with_conf)
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
                    vr = b.verify_step_fetch_and_run(row_toks, seq_id, fed,
                                                     None)
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
        check_cancel()
        r0 = b.decode_step_fetch_and_run(prompt[-1], seq_id,
                                         len(prompt) - 1, None,
                                         logits_readback=True)
        self._guard(r0)
        anchor, lp0 = pick(self._read_logits())
        fed = len(prompt)
        if not emit(anchor, lp0):
            return

        try:
            while True:
                check_cancel()
                # Draft async UNDER the always-committing sampled step.
                b.dspark_send_async(seq_id, anchor, fed, gamma)
                rr = b.decode_step_fetch_and_run(anchor, seq_id, fed, None,
                                                 logits_readback=True)
                self._guard(rr)
                t, lp_t = pick(self._read_logits())
                fed += 1
                cont = emit(t, lp_t)
                draft, confs = b.dspark_collect_async(gamma, with_conf)
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
                    vr = b.verify_step_fetch_and_run(
                        row_toks, seq_id, fed, None, logits_readback=True)
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
        finally:
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
                     sampling: SamplingParams) -> tuple[int, np.ndarray]:
        """One plain step with full-logits readback + host-side masked
        pick (grammar-advancing); returns (picked token, RAW logits row —
        pick_and_accept masks a copy, so the row is pre-mask and feeds
        logprobs).  Feeding at pos < the engine high-water mark is the
        standard spec-dec re-feed (INV-DSA-REWIND / INV-DSPARK-REFEED
        position-addressed overwrite)."""
        r = self.bridge.decode_step_fetch_and_run(tok, seq_id, pos, None,
                                                  logits_readback=True)
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
        want_lp = req.logprobs is not None
        token = prompt[-1]
        pos = len(prompt) - 1
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
        anchor, _ = self._masked_step(guided, prompt[-1], seq_id=seq_id,
                                      pos=len(prompt) - 1, sampling=smp)
        fed = len(prompt)
        if not emit(anchor):
            return

        try:
            while True:
                check_cancel()
                # Draft async UNDER the always-committing masked step.
                b.dspark_send_async(seq_id, anchor, fed, gamma)
                t, _ = self._masked_step(guided, anchor, fed, seq_id, smp)
                fed += 1
                cont = emit(t)
                draft, confs = b.dspark_collect_async(gamma, with_conf)
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
                    vr = b.verify_step_fetch_and_run(row_toks, seq_id, fed,
                                                     None)
                    self._guard(vr)
                    j2 = 0
                    while (j2 + 1 < g_use
                           and vr.argmax[j2] == draft[j2 + 1]):
                        j2 += 1
                    fed += j2 + 1
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
                                                      fed - 1, seq_id, smp)
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
        finally:
            b.drain_pending_dspark(gamma)

    @staticmethod
    def _guard(r) -> None:
        if not (math.isfinite(r.top1_prob) and math.isfinite(r.entropy)):
            raise BridgeError("NaN top1_prob/entropy from engine")
