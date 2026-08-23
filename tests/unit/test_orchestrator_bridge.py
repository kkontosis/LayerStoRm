"""Unit suite for the bridge-based production Orchestrator (successor
core, python/orchestrator/orchestrator.py) against the scripted FakeDaemon
from test_bridge_ring — no CUDA, no weights.

Covers the serving semantics the OpenAI-endpoint step will rely on:
greedy-speculative decode (champion arm) with EOS/max_tokens stop and
in-order on_token streaming, the plain sampled path routing, prompt
prefill chunking, cancellation, and the submit/complete callback
contract."""

from __future__ import annotations

import pytest

from test_bridge_ring import (  # the scripted-daemon harness
    FIRST_MOE,
    VOCAB,
    FakeDaemon,
    _finish,
    _make,
    chain,
    f,
)

from orchestrator.orchestrator import (
    InferenceRequest,
    Orchestrator,
    SamplingParams,
    SpeculationConfig,
)
from orchestrator.types import EngineMetadata


def _meta(eos: tuple[int, ...] = ()) -> EngineMetadata:
    return EngineMetadata(
        num_gpus=4, num_moe_layers=2, num_experts=64, num_layers=4,
        expert_bytes=0, kv_bytes_per_page=0, eos_token_ids=eos,
        vocab_size=VOCAB, moe_batch_capacity=512)


def _orch(daemon_gamma: int = 5, *, eos: tuple[int, ...] = (),
          spec: bool = True):
    bridge, daemon, _ = _make(gamma=daemon_gamma, use_far=True)
    orch = Orchestrator(
        bridge, metadata=_meta(eos),
        speculation=SpeculationConfig(enabled=spec, gamma=daemon_gamma,
                                      conf_thresh=0.0))
    return orch, daemon


class _Sink:
    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.lps: list = []                  # per-token StepLogprobs | None
        self.done: tuple | None = None
        self.done_lp = None                  # on_complete logprobs list

    def on_token(self, rid: int, tok: int, logp) -> None:
        self.tokens.append(tok)
        self.lps.append(logp)

    def on_complete(self, rid: int, tokens: list[int], reason: str,
                    logp) -> None:
        assert self.done is None, "on_complete fired twice"
        self.done = (rid, list(tokens), reason)
        self.done_lp = logp


def _serve(orch: Orchestrator, req: InferenceRequest) -> None:
    orch.submit_request(req)
    assert orch._serve_next() is True


def test_greedy_speculative_stream_and_length_stop():
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=7, prompt_token_ids=[4321], max_tokens=24,
            on_token=sink.on_token, on_complete=sink.on_complete))
        rid, tokens, reason = sink.done
        assert rid == 7 and reason == "length"
        assert tokens == chain(4321, 24), "speculative path lost losslessness"
        assert sink.tokens == tokens, "on_token stream != final tokens"
        st = orch.last_stats
        assert st.tokens == 24 and st.proposed > 0 and st.accepted > 0
        assert daemon.far_layers > 0
    finally:
        _finish(daemon)


def test_eos_stop_truncates_speculative_overshoot():
    ref = chain(4321, 24)
    eos_tok = ref[9]
    orch, daemon = _orch(eos=(eos_tok,))
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=[4321], max_tokens=0,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "stop"
        assert tokens == ref[:10], "must stop AT the EOS token"
        assert sink.tokens == tokens
    finally:
        _finish(daemon)


def test_prompt_prefill_chunking():
    prompt = chain(11, 130)          # > 2 prefill chunks + seed
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=8,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length"
        assert tokens == chain(prompt[-1], 8)
    finally:
        _finish(daemon)


def test_sampled_request_routes_to_plain_path():
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=3, prompt_token_ids=[555], max_tokens=6,
            sampling=SamplingParams(temperature=0.7, top_p=0.9, seed=123),
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length"
        # FakeDaemon samples argmax regardless of params — the routing is
        # what's under test: no speculation stats accumulate on this path.
        assert tokens == chain(555, 6)
        assert orch.last_stats.proposed == 0
        assert daemon.dspark_calls == 0
    finally:
        _finish(daemon)


def test_spec_disabled_greedy_uses_plain_path():
    orch, daemon = _orch(spec=False)
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=4, prompt_token_ids=[999], max_tokens=5,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert tokens == chain(999, 5) and reason == "length"
        assert daemon.dspark_calls == 0
    finally:
        _finish(daemon)


def test_cancel_before_serve():
    orch, daemon = _orch()
    try:
        sink = _Sink()
        orch.submit_request(InferenceRequest(
            request_id=5, prompt_token_ids=[42], max_tokens=100,
            on_complete=sink.on_complete))
        orch.cancel_request(5)
        assert orch._serve_next() is True
        assert sink.done == (5, [], "cancelled")
    finally:
        _finish(daemon)


def test_cancel_mid_generation_via_on_token():
    orch, daemon = _orch()
    try:
        sink = _Sink()

        def cancelling_on_token(rid: int, tok: int, logp) -> None:
            sink.tokens.append(tok)
            if len(sink.tokens) == 4:
                orch.cancel_request(rid)

        _serve(orch, InferenceRequest(
            request_id=6, prompt_token_ids=[4321], max_tokens=1000,
            on_token=cancelling_on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "cancelled" and tokens == []
        assert len(sink.tokens) >= 4          # streamed before the cancel
        assert sink.tokens == chain(4321, len(sink.tokens))
    finally:
        _finish(daemon)


def test_empty_prompt_is_an_error():
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=8, prompt_token_ids=[],
            on_complete=sink.on_complete))
        assert sink.done[2] == "error"
    finally:
        _finish(daemon)


# ── prefix cache (serving.prefix_cache): registry + SEQ_FORK reuse ──────────


def _orch_pc(daemon_gamma: int = 5, **pc_kw):
    from orchestrator.orchestrator import PrefixCacheConfig
    bridge, daemon, _ = _make(gamma=daemon_gamma, use_far=True)
    orch = Orchestrator(
        bridge, metadata=_meta(),
        speculation=SpeculationConfig(enabled=True, gamma=daemon_gamma,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(**pc_kw))
    return orch, daemon


def test_prefix_hit_skips_prefill_token_identical():
    prompt = chain(11, 130)                  # 2×64-chunk prefill + seed
    orch, daemon = _orch_pc()
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_token=s1.on_token, on_complete=s1.on_complete))
        assert daemon.forks == 1             # holder registered
        assert orch.last_stats.prefix_hit_tokens == 0
        embeds_before = len(daemon.embed_calls)

        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=8,
            on_token=s2.on_token, on_complete=s2.on_complete))
        # INV-PREFIX-CACHE-1: token-identical to the uncached run.
        assert s2.done[1] == s1.done[1] == chain(prompt[-1], 8)
        # Grid-aligned registry: prompt 130 -> pre 129 -> holder at 128.
        assert orch.last_stats.prefix_hit_tokens == 128
        # Exact-duplicate prefix → no second holder; hit forked once.
        assert daemon.forks == 2
        assert orch.prefix_cache.hits == 1
        # No 64-token prefill chunk was embedded in request 2's window.
        assert 64 not in daemon.embed_calls[embeds_before:]
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_hit_shorter_entry_extends_registry():
    base = chain(21, 100)
    longer = base + chain(999, 40)
    orch, daemon = _orch_pc()
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=base, max_tokens=4,
            on_complete=s1.on_complete))
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=longer, max_tokens=4,
            on_complete=s2.on_complete))
        assert s2.done[1] == chain(longer[-1], 4)
        # Hit on the shorter entry (grid-aligned at 64 tokens) ...
        assert orch.last_stats.prefix_hit_tokens == 64
        # ... and the longer prefix registered as a second entry.
        assert len(orch.prefix_cache._entries) == 2
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_superchunk_grid_registration_and_hit():
    """TD-V4-SERVE-PREFIX: the superchunk prefill path (V4 serving shape)
    registers holders on the SUPERCHUNK grid (moe_batch_capacity) and
    serves hits token-identically — holder and delta reproduce the exact
    absolute superchunk boundaries of an uncached run."""
    from orchestrator.orchestrator import PrefixCacheConfig
    prompt = chain(31, 600)                  # pre 599 -> holder at 512
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    orch = Orchestrator(
        bridge, metadata=_meta(),
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(),
        prefill_superchunk=True)
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        # Two holders: the grid entry (512, mid-prefill) + the exact-body
        # entry (599, post-prefill).
        assert daemon.forks == 2
        assert orch.last_stats.prefix_hit_tokens == 0
        assert len(orch.prefix_cache._entries) == 2

        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s2.on_complete))
        # INV-PREFIX-CACHE-1: token-identical to the uncached run; the
        # longest valid entry (exact body, 599) wins over the grid entry.
        assert s2.done[1] == s1.done[1] == chain(prompt[-1], 8)
        assert orch.last_stats.prefix_hit_tokens == 599
        assert daemon.forks == 3             # the hit fork
        assert orch.prefix_cache.hits == 1

        # A prompt EXTENDING the shared 512-aligned prefix hits the GRID
        # entry (the 599 exact-body entry is invalid for it).
        s3 = _Sink()
        longer = prompt[:512] + chain(777, 100)
        _serve(orch, InferenceRequest(
            request_id=3, prompt_token_ids=longer, max_tokens=4,
            on_complete=s3.on_complete))
        assert s3.done[1] == chain(longer[-1], 4)
        assert orch.last_stats.prefix_hit_tokens == 512
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_superchunk_exact_body_hit_short_prompt():
    """TD-V4-SERVE-PREFIX: a prompt shorter than the superchunk grid still
    registers an EXACT-prompt-body holder — a repeated identical body hits
    and skips the whole prefill (by-construction identical: the holder's
    shapes ARE this body's uncached shapes)."""
    from orchestrator.orchestrator import PrefixCacheConfig
    prompt = chain(37, 130)                  # pre 129 < grid 512
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    orch = Orchestrator(
        bridge, metadata=_meta(),
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(),
        prefill_superchunk=True)
    try:
        s1, s2, s3 = _Sink(), _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=6,
            on_complete=s1.on_complete))
        assert daemon.forks == 1             # exact-body holder at 129
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=6,
            on_complete=s2.on_complete))
        assert s2.done[1] == s1.done[1] == chain(prompt[-1], 6)
        assert orch.last_stats.prefix_hit_tokens == 129
        # A LONGER prompt sharing the 129-token body must NOT use the
        # non-aligned entry (shape identity) — it prefills uncached.
        longer = prompt[:-1] + chain(555, 40)
        _serve(orch, InferenceRequest(
            request_id=3, prompt_token_ids=longer, max_tokens=4,
            on_complete=s3.on_complete))
        assert s3.done[1] == chain(longer[-1], 4)
        assert orch.last_stats.prefix_hit_tokens == 0
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_cache_lru_eviction_budget():
    orch, daemon = _orch_pc(max_entries=2)
    try:
        for i, seed in enumerate((31, 41, 51)):
            sink = _Sink()
            _serve(orch, InferenceRequest(
                request_id=10 + i, prompt_token_ids=chain(seed, 70),
                max_tokens=3, on_complete=sink.on_complete))
            assert sink.done[2] == "length"
        pc = orch.prefix_cache
        assert len(pc._entries) == 2
        assert pc.evictions == 1
        assert daemon.seq_frees >= 4         # 3 working seqs + 1 holder
        # Oldest prefix (seed 31) evicted; newest two remain.
        kept = {e.tokens[0] for e in pc._entries}
        assert kept == {chain(41, 1)[0], chain(51, 1)[0]}
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_cache_token_budget_eviction():
    orch, daemon = _orch_pc(max_cached_tokens=100)
    try:
        for i, seed in enumerate((61, 71)):
            sink = _Sink()
            _serve(orch, InferenceRequest(
                request_id=20 + i, prompt_token_ids=chain(seed, 100),
                max_tokens=3, on_complete=sink.on_complete))
        pc = orch.prefix_cache
        assert len(pc._entries) == 1         # 64+64 > 100 → LRU evicted
        assert pc.evictions == 1
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_cache_disabled_parity():
    from orchestrator.orchestrator import PrefixCacheConfig
    prompt = chain(77, 90)
    ref = None
    for enabled in (False, True):
        orch, daemon = _orch_pc(enabled=enabled)
        try:
            sink = _Sink()
            _serve(orch, InferenceRequest(
                request_id=1, prompt_token_ids=prompt, max_tokens=6,
                on_complete=sink.on_complete))
            if ref is None:
                ref = sink.done[1]
                assert orch.prefix_cache is None
                assert daemon.forks == 0     # disabled = pre-cache flow
            else:
                assert sink.done[1] == ref   # ON vs OFF token-identical
            assert not daemon.errors, daemon.errors
        finally:
            _finish(daemon)


def test_seq_create_evicts_holder_at_admission():
    """Evict-at-admission (regression-hunt 2026-08-23 finding (b)): a
    retained prefix holder pins its pages; a NEW (different-prompt)
    request whose seq_create hits page-pool exhaustion must evict
    holders and retry instead of failing the request."""
    orch, daemon = _orch_pc()
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_token=s1.on_token, on_complete=s1.on_complete))
        assert len(orch.prefix_cache._entries) == 1      # holder retained
        assert daemon.known_seqs, "holder must outlive the request"
        # Freeze the pool: the live holder occupies ALL capacity, exactly
        # the hunt's shape (pool full of cache, not of work).
        daemon.seq_capacity = len(daemon.known_seqs)

        prompt_b = chain(500, 130)                       # different prefix
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt_b, max_tokens=8,
            on_token=s2.on_token, on_complete=s2.on_complete))
        # The request SUCCEEDED via eviction, token-identical to an
        # uncached run (the evicted holder never contributed compute).
        assert s2.done[2] == "length"
        assert s2.done[1] == chain(prompt_b[-1], 8)
        assert daemon.seq_admission_rejects >= 1         # it WAS rejected
        assert orch.prefix_cache.evictions >= 1          # then evicted
        assert orch.last_stats.prefix_hit_tokens == 0    # miss path
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_seq_create_exhausted_no_holders_is_an_error():
    """No holders to evict → the admission failure still surfaces as a
    request error (never an infinite retry)."""
    orch, daemon = _orch_pc()
    try:
        daemon.seq_capacity = 0                          # nothing admits
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_complete=sink.on_complete))
        assert sink.done[2] == "error"
        assert daemon.seq_admission_rejects == 1         # exactly one try
    finally:
        _finish(daemon)


# ── error-detail threading (TD-SERVE-ERROR-MASKING) ────────────────────────
# A failed request must hand the CALLER the engine's failure detail: the
# serving layer turns it into a typed HTTP error instead of a 200 with an
# empty body.  The detail rides an OPTIONAL fifth ``error`` argument, so
# every four-argument consumer above keeps working untouched.


class _ErrSink(_Sink):
    """on_complete consumer that opts into the error detail."""

    def __init__(self) -> None:
        super().__init__()
        self.error: str | None = None

    def on_complete(self, rid: int, tokens: list[int], reason: str,
                    logp, error: str = "") -> None:
        super().on_complete(rid, tokens, reason, logp)
        self.error = error


def test_error_detail_threaded_for_seq_create_failure():
    """The pool-exhaustion detail (pool identity + page counts) reaches
    the caller verbatim — the HTTP layer classifies on it."""
    orch, daemon = _orch_pc()
    try:
        daemon.seq_capacity = 0
        sink = _ErrSink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_complete=sink.on_complete))
        assert sink.done[2] == "error"
        assert "pool exhausted" in sink.error
        assert "seq_create" in sink.error
    finally:
        _finish(daemon)


def test_error_detail_threaded_for_rejected_prompt():
    orch, daemon = _orch()
    try:
        sink = _ErrSink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=[], on_complete=sink.on_complete))
        assert sink.done[2] == "error"
        assert sink.error == "empty prompt"

        sink2 = _ErrSink()
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=[VOCAB + 7],
            on_complete=sink2.on_complete))
        assert sink2.done[2] == "error"
        assert "out of vocab" in sink2.error
    finally:
        _finish(daemon)


def test_error_detail_empty_on_successful_completions():
    orch, daemon = _orch()
    try:
        sink = _ErrSink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(3, 4), max_tokens=3,
            on_complete=sink.on_complete))
        assert sink.done[2] == "length"
        assert sink.error == ""                # never None: always passed

        cancelled = _ErrSink()
        req = InferenceRequest(request_id=2, prompt_token_ids=chain(3, 4),
                               max_tokens=3, on_complete=cancelled.on_complete)
        orch.submit_request(req)
        orch.cancel_request(2)
        assert orch._serve_next() is True
        assert cancelled.done[2] == "cancelled"
        assert cancelled.error == ""           # cancel is NOT a failure
    finally:
        _finish(daemon)


def test_legacy_four_arg_consumer_still_gets_errors():
    """Backward compatibility: a consumer without the ``error`` parameter
    is called with exactly four arguments (no TypeError)."""
    orch, daemon = _orch()
    try:
        seen: list[tuple] = []
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=[],
            on_complete=lambda rid, toks, reason, lp: seen.append(
                (rid, toks, reason, lp))))
        assert seen == [(1, [], "error", None)]
    finally:
        _finish(daemon)


def test_kwargs_consumer_receives_error_detail():
    from orchestrator.orchestrator import _accepts_error_arg
    orch, daemon = _orch()
    try:
        seen: list[dict] = []

        def sink(rid, toks, reason, lp, **kw):
            seen.append(kw)

        assert _accepts_error_arg(sink) is True
        assert _accepts_error_arg(lambda a, b, c, d: None) is False
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=[], on_complete=sink))
        assert seen == [{"error": "empty prompt"}]
    finally:
        _finish(daemon)


def test_unexpected_exception_fails_the_request_not_the_loop():
    """A non-BridgeError escape inside generation is reported as a request
    error (typed detail) instead of killing the serving loop — a dead loop
    would hang the HTTP request forever."""
    orch, daemon = _orch()
    try:
        def boom(req):
            raise ValueError("scripted explosion")

        orch._generate = boom
        sink = _ErrSink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(3, 4), max_tokens=2,
            on_complete=sink.on_complete))
        assert sink.done[2] == "error"
        assert sink.error == "ValueError: scripted explosion"
    finally:
        _finish(daemon)


def test_prefix_cache_cancel_keeps_registry_consistent():
    prompt = chain(88, 80)
    orch, daemon = _orch_pc()
    try:
        sink = _Sink()

        def cancelling(rid: int, tok: int, lp) -> None:
            sink.tokens.append(tok)
            if len(sink.tokens) == 2:
                orch.cancel_request(rid)

        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=100,
            on_token=cancelling, on_complete=sink.on_complete))
        assert sink.done[2] == "cancelled"
        # The holder registered before decode survives the cancel ...
        assert len(orch.prefix_cache._entries) == 1
        s2 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=5,
            on_complete=s2.on_complete))
        # ... and the follow-up request hits it and decodes correctly.
        assert orch.last_stats.prefix_hit_tokens == 64
        assert s2.done[1] == chain(prompt[-1], 5)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


# ── chain-aware eviction (user directive 2026-08-18): nested prefixes share
# CoW pages — touch propagates down the chain, eviction is LRU-major +
# deepest-first minor, and the token budget counts UNIQUE tokens. ───────────


class _StubBridge:
    """fork/free recorder for direct PrefixCache policy tests (no rings)."""

    def __init__(self) -> None:
        self.forks: list[tuple[int, int]] = []
        self.freed: list[int] = []

    def fork_sequence(self, src: int, dst: int) -> None:
        self.forks.append((src, dst))

    def free_sequence(self, seq_id: int) -> None:
        self.freed.append(seq_id)


def _pc(**cfg_kw):
    from orchestrator.orchestrator import PrefixCache, PrefixCacheConfig
    return PrefixCache(PrefixCacheConfig(**cfg_kw), _StubBridge())


def _reg(pc, tokens, seq_id):
    assert pc.register(tuple(tokens), src_seq_id=1000 + seq_id,
                       holder_seq_id=seq_id)


def test_prefix_chain_touch_propagates_down():
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    A = list(range(64))
    _reg(pc, A, 1)                       # chain head
    _reg(pc, A + list(range(100, 164)), 2)   # chain tail (A ⊂ A+B)
    _reg(pc, list(range(500, 564)), 3)       # unrelated C (newest stamp)
    # Hit the chain TAIL: head A must be co-touched past C.
    hit = pc.lookup(A + list(range(100, 164)) + [7])
    assert hit is not None and hit.seq_id == 2
    by_id = {e.seq_id: e for e in pc._entries}
    assert by_id[1].last_used == by_id[2].last_used, "head not co-touched"
    assert by_id[3].last_used < by_id[1].last_used
    # Under entry pressure the eviction victim is C (oldest stamp), NOT the
    # chain head registered before it.
    assert pc._evict_one(protect=None)
    assert {e.seq_id for e in pc._entries} == {1, 2}


def test_prefix_chain_deepest_first_eviction():
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    A = list(range(64))
    _reg(pc, A, 1)
    _reg(pc, A + list(range(100, 164)), 2)
    _reg(pc, A + list(range(100, 164)) + list(range(200, 264)), 3)
    pc.lookup(A + list(range(100, 164)) + list(range(200, 264)) + [7])
    # Whole chain shares one stamp → deepest-first is the within-chain order.
    assert pc._evict_one(protect=None)
    assert {e.seq_id for e in pc._entries} == {1, 2}, "deepest must go first"
    assert pc._evict_one(protect=None)
    assert {e.seq_id for e in pc._entries} == {1}
    # The tail eviction is the memory-effective one: unique total drops by
    # exactly the evicted tails.
    assert pc.total_unique_tokens() == 64


def test_prefix_chain_unique_token_budget():
    # Chain 64 ⊂ 128 ⊂ 192 under a 200-token budget: naive sum (384) would
    # evict; unique accounting (192 = longest member) must keep all three.
    pc = _pc(max_entries=8, max_cached_tokens=200)
    A = list(range(64))
    _reg(pc, A, 1)
    _reg(pc, A + list(range(100, 164)), 2)
    _reg(pc, A + list(range(100, 164)) + list(range(200, 264)), 3)
    assert pc.total_unique_tokens() == 192
    assert len(pc._entries) == 3 and pc.evictions == 0
    # An UNRELATED 64-token entry pushes unique total to 256 > 200. Only
    # chain-MAXIMAL entries are evictable (a prefix outlives its
    # extensions), so the chain TAIL (192) goes — freeing real pages —
    # and the head/mid survive with the protected newcomer:
    # unique 64+128 shared-chain + 64 = 192 <= 200.
    _reg(pc, list(range(500, 564)), 4)
    assert pc.total_unique_tokens() == 192
    assert {e.seq_id for e in pc._entries} == {1, 2, 4}
    assert pc.evictions == 1


def test_prefix_chain_non_chain_entries_unaffected():
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    _reg(pc, list(range(64)), 1)
    _reg(pc, list(range(500, 564)), 2)
    pc.lookup(list(range(64)) + [7])         # touch only entry 1
    by_id = {e.seq_id: e for e in pc._entries}
    assert by_id[2].last_used < by_id[1].last_used
    assert pc.total_unique_tokens() == 128   # unrelated = plain sum
    assert pc._evict_one(protect=None)
    assert {e.seq_id for e in pc._entries} == {1}


# ── TD-PREFIX-DSPARK-FORK-CTX: forked sequences must not issue dspark when
# the drafting context cannot be adopted (engine fails closed with
# CMP_ERROR otherwise — the plain greedy path is token-identical).


def test_prefix_fork_adoptable_keeps_speculative_path():
    prompt = chain(11, 130)                  # holder at 128
    orch, daemon = _orch_pc()
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        calls_after_a = daemon.dspark_calls
        assert calls_after_a > 0
        # Immediate same-prefix repeat: fork point 128 <= tracked frontier
        # (129) — the engine ADOPTS, so speculation stays armed.
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s2.on_complete))
        assert orch.last_stats.prefix_hit_tokens == 128
        assert daemon.dspark_calls > calls_after_a
        assert s2.done[1] == s1.done[1]
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_fork_non_adoptable_routes_to_plain_path():
    prompt = chain(11, 130)                  # holder at 128
    orch, daemon = _orch_pc()
    try:
        s1, s2, s3 = _Sink(), _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        # Intervening short full-prefill request re-arms the drafting
        # context at a frontier BELOW the fork point (128).
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=chain(500, 10), max_tokens=4,
            on_complete=s2.on_complete))
        calls_before = daemon.dspark_calls
        # Same-prefix repeat now forks at 128 > tracked frontier: the
        # engine would invalidate + CMP_ERROR every run_step — the
        # orchestrator must route to PLAIN decode (lossless) instead.
        _serve(orch, InferenceRequest(
            request_id=3, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s3.on_complete))
        assert orch.last_stats.prefix_hit_tokens == 128
        assert daemon.dspark_calls == calls_before   # no dspark issued
        assert orch.last_stats.proposed == 0
        assert s3.done[1] == s1.done[1]              # token-identical
        assert s3.done[2] == "length"
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


# ── guided decoding (TD-SERVE-NAMED-TOOL-CHOICE): grammar-constrained
# flows against the scripted daemon.  The FakeDaemon's readback_logits row
# peaks at the chain token f(prev) with runner-up (f+1) % VOCAB, so a
# scripted "grammar" that bans a chain token forces the deterministic
# repair token — reference simulation in guided_chain(). ─────────────────


class FakeGuided:
    """Scripted GuidedState: a token is a violation iff in `banned`;
    grammar completes after `complete_after` accepted tokens."""

    def __init__(self, banned=(), complete_after=None) -> None:
        self.banned = set(banned)
        self.complete_after = complete_after
        self.accepted: list[int] = []
        self.picks = 0

    @property
    def completed(self) -> bool:
        return (self.complete_after is not None
                and len(self.accepted) >= self.complete_after)

    def try_accept(self, tok: int) -> bool:
        if tok in self.banned:
            return False
        self.accepted.append(tok)
        return True

    def rollback(self, n: int) -> None:
        if n:
            del self.accepted[-n:]

    def pick_and_accept(self, logits, temperature=0.0, top_p=1.0,
                        top_k=0, seed=42) -> int:
        import numpy as np
        self.picks += 1
        for t in np.argsort(-logits)[:8]:
            if int(t) not in self.banned:
                self.accepted.append(int(t))
                return int(t)
        raise AssertionError("no allowed token in the top-8 logits")


def guided_chain(seed: int, n: int, banned=()) -> list[int]:
    """Reference: greedy chain with banned tokens repaired to the
    runner-up (f+1) % VOCAB (the FakeDaemon logits shape)."""
    banned = set(banned)
    out, t = [], seed
    for _ in range(n):
        nxt = f(t)
        if nxt in banned:
            nxt = (nxt + 1) % VOCAB
        out.append(nxt)
        t = nxt
    return out


def _serve_guided(orch, guided, *, prompt=(4321,), max_tokens=24,
                  force_plain=False):
    sink = _Sink()
    _serve(orch, InferenceRequest(
        request_id=99, prompt_token_ids=list(prompt), max_tokens=max_tokens,
        guided=guided, force_plain=force_plain,
        on_token=sink.on_token, on_complete=sink.on_complete))
    return sink


def test_guided_plain_unconstrained_grammar_matches_chain():
    orch, daemon = _orch()
    try:
        sink = _serve_guided(orch, FakeGuided(), force_plain=True)
        _, tokens, reason = sink.done
        assert reason == "length"
        assert tokens == chain(4321, 24)
        assert daemon.dspark_calls == 0          # plain path
    finally:
        _finish(daemon)


def test_guided_plain_mask_forces_repair_tokens():
    ref = chain(4321, 24)
    banned = {ref[3], ref[10]}
    orch, daemon = _orch()
    try:
        sink = _serve_guided(orch, FakeGuided(banned=banned),
                             force_plain=True)
        _, tokens, _ = sink.done
        assert tokens == guided_chain(4321, 24, banned)
        assert tokens != ref                     # the mask actually bit
    finally:
        _finish(daemon)


def test_guided_completion_stops_with_tool_calls():
    orch, daemon = _orch()
    try:
        sink = _serve_guided(orch, FakeGuided(complete_after=7),
                             force_plain=True, max_tokens=100)
        _, tokens, reason = sink.done
        assert reason == "tool_calls"
        assert tokens == chain(4321, 7)
    finally:
        _finish(daemon)


def test_guided_speculative_lossless_and_stats():
    orch, daemon = _orch()
    try:
        sink = _serve_guided(orch, FakeGuided())
        _, tokens, reason = sink.done
        assert reason == "length"
        assert tokens == chain(4321, 24), "guided spec lost losslessness"
        st = orch.last_stats
        assert st.proposed > 0 and st.accepted > 0   # speculation engaged
        assert daemon.dspark_calls > 0
        assert st.grammar_refeeds == 0               # nothing violated
    finally:
        _finish(daemon)


def test_guided_speculative_violation_refeed_matches_plain():
    ref = chain(4321, 40)
    banned = {ref[5], ref[17], ref[29]}
    results = []
    for force_plain in (True, False):
        orch, daemon = _orch()
        try:
            sink = _serve_guided(orch, FakeGuided(banned=banned),
                                 force_plain=force_plain, max_tokens=40)
            _, tokens, reason = sink.done
            assert reason == "length"
            results.append(tokens)
            if not force_plain:
                st = orch.last_stats
                # The grammar predicate engaged: truncated draft slots
                # and/or bonus re-feeds must have fired.
                assert st.grammar_trunc_slots + st.grammar_refeeds > 0
        finally:
            _finish(daemon)
    assert results[0] == results[1] == guided_chain(4321, 40, banned), \
        "guided plain and guided speculative paths must be token-identical"


def test_guided_speculative_completion_mid_round():
    orch, daemon = _orch()
    try:
        sink = _serve_guided(orch, FakeGuided(complete_after=9),
                             max_tokens=100)
        _, tokens, reason = sink.done
        assert reason == "tool_calls"
        assert tokens == chain(4321, 9), \
            "tokens past the grammar-complete point must be dropped"
    finally:
        _finish(daemon)


def test_prefix_fork_recovers_speculation_after_full_prefill():
    prompt = chain(11, 130)
    orch, daemon = _orch_pc()
    try:
        sinks = [_Sink() for _ in range(4)]
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=sinks[0].on_complete))
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=chain(500, 10), max_tokens=4,
            on_complete=sinks[1].on_complete))
        _serve(orch, InferenceRequest(          # plain-path fork
            request_id=3, prompt_token_ids=prompt, max_tokens=8,
            on_complete=sinks[2].on_complete))
        # A fresh LONG full prefill re-arms the context past the fork
        # point; the next fork is adoptable and speculation returns.
        long2 = chain(77, 200)
        _serve(orch, InferenceRequest(
            request_id=4, prompt_token_ids=long2, max_tokens=4,
            on_complete=sinks[3].on_complete))
        calls_before = daemon.dspark_calls
        s5 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=5, prompt_token_ids=long2, max_tokens=4,
            on_complete=s5.on_complete))
        assert orch.last_stats.prefix_hit_tokens > 0
        assert daemon.dspark_calls > calls_before
        assert s5.done[1] == sinks[3].done[1]
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


# ── logprobs serving (TD-ORCH-LOGPROBS): host-side log-softmax over the
# engine's full-logits readback row.  The FakeDaemon logits row is 0.0
# everywhere except peak 1.0 at the chain token and 0.5 at the runner-up,
# so every expected logprob is closed-form. ─────────────────────────────────


def _lse() -> float:
    """log-sum-exp of the FakeDaemon logits row."""
    import math
    return math.log((VOCAB - 2) * 1.0 + math.exp(1.0) + math.exp(0.5))


def test_logprobs_plain_fallback_serves_topk():
    orch, daemon = _orch()                   # speculation ARMED
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=30, prompt_token_ids=[4321], max_tokens=6,
            logprobs=3, on_token=sink.on_token,
            on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length"
        # TD-ORCH-LOGPROBS-SPEC: logprobs requests fall back to the plain
        # arm (last-row-only readback) — token output unchanged.
        assert tokens == chain(4321, 6)
        assert daemon.dspark_calls == 0
        # One full-logits readback per emitted token; the token pick
        # stayed ENGINE-side (CMD_SAMPLE_TOKENS still ran every step).
        assert daemon.logits_readbacks == 6
        assert daemon.samples == 6
        lse = _lse()
        for tok, lp in zip(tokens, sink.lps):
            assert lp is not None
            assert lp.token.token_id == tok
            assert lp.token.logprob == pytest.approx(1.0 - lse)
            assert lp.token.logprob <= 0.0
            assert len(lp.top_logprobs) == 3
            vals = [t.logprob for t in lp.top_logprobs]
            assert vals == sorted(vals, reverse=True)
            assert all(v <= 0.0 for v in vals)
            # Greedy chain: the chosen token IS the distribution peak.
            assert lp.top_logprobs[0].token_id == tok
            assert lp.top_logprobs[1].token_id == (tok + 1) % VOCAB
            assert lp.top_logprobs[1].logprob == pytest.approx(0.5 - lse)
        # on_complete carries the full per-token list, emit-aligned.
        assert sink.done_lp == sink.lps and len(sink.done_lp) == 6
    finally:
        _finish(daemon)


def test_logprobs_off_zero_readback_cost():
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=31, prompt_token_ids=[4321], max_tokens=12,
            on_token=sink.on_token, on_complete=sink.on_complete))
        assert sink.done[1] == chain(4321, 12)
        # Byte-identical chains: no readback_logits head ever issued.
        assert daemon.logits_readbacks == 0
        assert all(lp is None for lp in sink.lps)
        assert sink.done_lp is None
    finally:
        _finish(daemon)


def test_logprobs_k0_chosen_token_only():
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=32, prompt_token_ids=[555], max_tokens=4,
            logprobs=0, on_token=sink.on_token,
            on_complete=sink.on_complete))
        assert sink.done[1] == chain(555, 4)
        assert daemon.dspark_calls == 0      # still the plain fallback
        for lp in sink.lps:
            assert lp is not None and lp.top_logprobs == ()
            assert lp.token.logprob == pytest.approx(1.0 - _lse())
    finally:
        _finish(daemon)


def test_logprobs_sampled_path():
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=33, prompt_token_ids=[555], max_tokens=5,
            sampling=SamplingParams(temperature=0.7, top_p=0.9, seed=1),
            logprobs=2, on_token=sink.on_token,
            on_complete=sink.on_complete))
        # FakeDaemon samples argmax regardless — routing + lp shape are
        # under test; the engine-side sampler still picked every token.
        assert sink.done[1] == chain(555, 5)
        assert daemon.samples == 5 and daemon.logits_readbacks == 5
        assert all(lp is not None and len(lp.top_logprobs) == 2
                   for lp in sink.lps)
    finally:
        _finish(daemon)


def test_guided_logprobs_report_raw_distribution():
    ref = chain(4321, 8)
    banned = {ref[2]}
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=34, prompt_token_ids=[4321], max_tokens=8,
            logprobs=2, guided=FakeGuided(banned=banned),
            on_token=sink.on_token, on_complete=sink.on_complete))
        expect = guided_chain(4321, 8, banned)
        assert sink.done[1] == expect
        # logprobs + guided → guided PLAIN fallback (no drafts).
        assert daemon.dspark_calls == 0
        lse = _lse()
        lp = sink.lps[2]                     # the grammar-repaired step
        assert lp.token.token_id == expect[2] == (ref[2] + 1) % VOCAB
        # Raw (pre-mask) distribution: the repair token honestly reports
        # its runner-up probability; the banned peak stays top-1.
        assert lp.token.logprob == pytest.approx(0.5 - lse)
        assert lp.top_logprobs[0].token_id == ref[2]
        assert lp.top_logprobs[0].logprob == pytest.approx(1.0 - lse)
        assert len(sink.done_lp) == 8
    finally:
        _finish(daemon)
