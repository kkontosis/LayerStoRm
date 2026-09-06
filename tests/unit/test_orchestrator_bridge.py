"""Unit suite for the bridge-based production Orchestrator (successor
core, python/orchestrator/orchestrator.py) against the scripted FakeDaemon
from test_bridge_ring — no CUDA, no weights.

Covers the serving semantics the OpenAI-endpoint step will rely on:
greedy-speculative decode (champion arm) with EOS/max_tokens stop and
in-order on_token streaming, the plain sampled path routing, prompt
prefill chunking, cancellation, and the submit/complete callback
contract."""

from __future__ import annotations

import contextlib
import os
import threading
import time

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


def test_moe_degraded_layers_surfaced_per_request():
    """TD-MOE-PROGRESSIVE-DEGRADED-SILENT: a degraded progressive-MoE
    finalize must reach RequestStats (per-request delta), and a healthy
    follow-up request in the same boot must read ZERO — the exact shape
    of the measured V4 first-request-after-boot incident."""
    orch, daemon = _orch()
    try:
        daemon.degrade_moe_remaining = 5      # 5 layers degrade, then healthy
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=41, prompt_token_ids=[4321], max_tokens=12,
            on_token=sink.on_token, on_complete=sink.on_complete))
        assert sink.done is not None and sink.done[2] == "length"
        assert orch.last_stats.moe_degraded_layers == 5, (
            f"degraded layers not surfaced: "
            f"{orch.last_stats.moe_degraded_layers}")
        # Same boot, identical follow-up request: counter must be a
        # PER-REQUEST delta, not the bridge's monotonic total.
        sink2 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=42, prompt_token_ids=[4321], max_tokens=12,
            on_token=sink2.on_token, on_complete=sink2.on_complete))
        assert sink2.done is not None and sink2.done[2] == "length"
        assert orch.last_stats.moe_degraded_layers == 0, (
            "healthy request reported stale degraded layers")
        assert sink.done[1] == sink2.done[1]  # FakeDaemon is deterministic
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


def test_degraded_request_does_not_register_prefix_holder():
    """TD-MOE-PROGRESSIVE-DEGRADED-SILENT x INV-PREFIX-CACHE-1: a request
    that saw degraded MoE finalizes must NOT freeze its (wrongly-computed)
    KV into a prefix holder — a later hit would silently reuse it."""
    prompt = chain(11, 130)                  # 2x64-chunk prefill + seed
    orch, daemon = _orch_pc()
    try:
        daemon.degrade_moe_remaining = 2     # degrade inside request 1
        s1 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_token=s1.on_token, on_complete=s1.on_complete))
        assert orch.last_stats.moe_degraded_layers == 2
        assert daemon.forks == 0, "degraded request registered a holder"
        # Healthy identical follow-up: registers normally, and its stats
        # read zero.
        s2 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=8,
            on_token=s2.on_token, on_complete=s2.on_complete))
        assert orch.last_stats.moe_degraded_layers == 0
        assert daemon.forks == 1             # holder registered now
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


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
        prefill_superchunk=True,
        prefill_sc_min_tokens=0)   # mechanism test: downgrade OFF
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
        prefill_superchunk=True,
        prefill_sc_min_tokens=0)   # mechanism test: downgrade OFF
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


def test_prefix_midedge_e2e_truncating_fork_and_realignment():
    """R4c e2e (GLM shape): a prompt diverging INSIDE a registered node's
    edge reuses the GRID-CLAMPED common prefix (here: the 512 boundary,
    served as a legacy full fork of the registered 512 node), the delta
    prefill stays on the absolute superchunk grid, the deeper grid holder
    still registers, and decode continues token-correctly."""
    import dataclasses
    from orchestrator.orchestrator import PrefixCacheConfig
    prompt = chain(31, 600)                  # pre 599: holders at 512+599
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    orch = Orchestrator(
        bridge,
        metadata=dataclasses.replace(_meta(), seq_fork_truncatable=True),
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(),
        prefill_superchunk=True,
        prefill_sc_min_tokens=0)   # mechanism test: downgrade OFF
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        assert len(orch.prefix_cache._entries) == 2      # 512 + 599

        # Divergence at 585: mid-page (585 % 16 != 0), mid-chunk
        # (585 % 64 != 0), sub-grid (585 % 512 != 0) — inside the
        # 512->599 edge.
        div = (prompt[585] + 1) % VOCAB
        prompt2 = prompt[:585] + [div] + chain(div, 599)  # pre2 = 1184
        calls: list[tuple[int, int]] = []
        forks: list[tuple[int, int, bool, int]] = []
        orig_sc = orch.bridge.prefill_superchunk_fetch_and_run
        orig_fork = orch.bridge.fork_sequence

        def spy_sc(toks, sid, pos0, sub):
            calls.append((pos0, len(toks)))
            return orig_sc(toks, sid, pos0, sub)

        def spy_fork(src, dst, frozen=False, prefix_len=0,
                     reserve_tokens=0):
            forks.append((src, dst, frozen, prefix_len))
            return orig_fork(src, dst, frozen=frozen,
                             prefix_len=prefix_len,
                             reserve_tokens=reserve_tokens)

        orch.bridge.prefill_superchunk_fetch_and_run = spy_sc
        orch.bridge.fork_sequence = spy_fork
        try:
            _serve(orch, InferenceRequest(
                request_id=2, prompt_token_ids=prompt2, max_tokens=8,
                on_complete=s2.on_complete))
        finally:
            orch.bridge.prefill_superchunk_fetch_and_run = orig_sc
            orch.bridge.fork_sequence = orig_fork
        # First fork of the request = the HIT fork.  The LCP is 585 —
        # sub-grid — so the reuse point CLAMPS to 512 and the registered
        # 512 node serves it as a legacy FULL fork (prefix_len 0); the
        # later registration forks are frozen, prefix 0 too.
        assert forks[0][2] is False and forks[0][3] == 0, forks
        assert all(fr and pl == 0 for _, _, fr, pl in forks[1:]), forks
        assert orch.last_stats.prefix_hit_tokens == 512
        # Delta runs on the ABSOLUTE stride grid from 512.
        assert calls == [(512, 512), (1024, 160)], calls
        # The deeper grid holder (1024) + prompt2's exact body (1184)
        # registered on top of the two originals.
        lens = sorted(len(e.tokens) for e in orch.prefix_cache._entries)
        assert lens == [512, 599, 1024, 1184], lens
        # Token-correct decode continuation.
        assert s2.done[1] == chain(prompt2[-1], 8)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_midedge_e2e_truncating_after_interior_retirement():
    """R4c e2e: when no node exists at the clamped grid boundary (the R2
    slot-pressure case — the grid interior was retired, only the deeper
    exact-body node survives), the hit is a TRUNCATING fork of the
    surviving node at the boundary, and the delta runs on the absolute
    grid from there."""
    import dataclasses
    from orchestrator.orchestrator import PrefixCacheConfig
    prompt = chain(31, 1300)                 # pre 1299: holders 1024+1299
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    orch = Orchestrator(
        bridge,
        metadata=dataclasses.replace(_meta(), seq_fork_truncatable=True),
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(),
        prefill_superchunk=True,
        prefill_sc_min_tokens=0)
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        pc = orch.prefix_cache
        by_len = {len(e.tokens): e for e in pc._entries}
        assert sorted(by_len) == [1024, 1299]
        pc._detach(by_len[1024])             # R2-style interior retirement
        pc._free_holder(by_len[1024])

        div = (prompt[1150] + 1) % VOCAB
        prompt2 = prompt[:1150] + [div] + chain(div, 300)   # pre2 = 1450
        forks: list[tuple[int, int, bool, int]] = []
        orig_fork = orch.bridge.fork_sequence

        def spy_fork(src, dst, frozen=False, prefix_len=0,
                     reserve_tokens=0):
            forks.append((src, dst, frozen, prefix_len))
            return orig_fork(src, dst, frozen=frozen,
                             prefix_len=prefix_len,
                             reserve_tokens=reserve_tokens)

        orch.bridge.fork_sequence = spy_fork
        try:
            _serve(orch, InferenceRequest(
                request_id=2, prompt_token_ids=prompt2, max_tokens=8,
                on_complete=s2.on_complete))
        finally:
            orch.bridge.fork_sequence = orig_fork
        # LCP 1150 clamps to 1024; no node there => truncating fork of
        # the surviving 1299 node.
        assert forks[0][2] is False and forks[0][3] == 1024, forks
        assert orch.last_stats.prefix_hit_tokens == 1024
        assert s2.done[1] == chain(prompt2[-1], 8)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_midedge_gated_off_without_arch_capability():
    """Without seq_fork_truncatable (V4 in-place rings / older engine),
    the legacy grid+exact-body validity filter stays: the same divergent
    prompt hits the GRID node with a FULL fork, never a prefix_len."""
    from orchestrator.orchestrator import PrefixCacheConfig
    prompt = chain(31, 600)
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    orch = Orchestrator(
        bridge, metadata=_meta(),        # seq_fork_truncatable=False
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(),
        prefill_superchunk=True,
        prefill_sc_min_tokens=0)
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        div = (prompt[585] + 1) % VOCAB
        prompt2 = prompt[:585] + [div] + chain(div, 599)
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt2, max_tokens=8,
            on_complete=s2.on_complete))
        assert orch.last_stats.prefix_hit_tokens == 512   # grid hit only
        assert daemon.last_fork_prefix == 0               # full fork
        assert s2.done[1] == chain(prompt2[-1], 8)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_midedge_subgrid_optin_e2e_exact_lcp():
    """SUB-GRID opt-in (USER DECISION 2026-08-28, default OFF): with
    PrefixCacheConfig.subgrid_mid_edge=True the same divergent prompt
    that clamps to 512 by default reuses the EXACT 585-token LCP via a
    truncating fork, the delta realigns to the next absolute stride
    boundary, and the serve log marks the hit SUB-GRID."""
    import dataclasses
    from orchestrator.orchestrator import PrefixCacheConfig
    prompt = chain(31, 600)                  # pre 599: holders at 512+599
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    orch = Orchestrator(
        bridge,
        metadata=dataclasses.replace(_meta(), seq_fork_truncatable=True),
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(subgrid_mid_edge=True),
        prefill_superchunk=True,
        prefill_sc_min_tokens=0)
    assert orch.subgrid_mid_edge is True
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        div = (prompt[585] + 1) % VOCAB
        prompt2 = prompt[:585] + [div] + chain(div, 599)  # pre2 = 1184
        calls: list[tuple[int, int]] = []
        forks: list[tuple[int, int, bool, int]] = []
        orig_sc = orch.bridge.prefill_superchunk_fetch_and_run
        orig_fork = orch.bridge.fork_sequence

        def spy_sc(toks, sid, pos0, sub):
            calls.append((pos0, len(toks)))
            return orig_sc(toks, sid, pos0, sub)

        def spy_fork(src, dst, frozen=False, prefix_len=0,
                     reserve_tokens=0):
            forks.append((src, dst, frozen, prefix_len))
            return orig_fork(src, dst, frozen=frozen,
                             prefix_len=prefix_len)

        orch.bridge.prefill_superchunk_fetch_and_run = spy_sc
        orch.bridge.fork_sequence = spy_fork
        try:
            _serve(orch, InferenceRequest(
                request_id=2, prompt_token_ids=prompt2, max_tokens=8,
                on_complete=s2.on_complete))
        finally:
            orch.bridge.prefill_superchunk_fetch_and_run = orig_sc
            orch.bridge.fork_sequence = orig_fork
        # First fork = the HIT: a TRUNCATING fork at the exact LCP 585
        # (no clamp) of the 599 node; later registration forks frozen.
        assert forks[0][2] is False and forks[0][3] == 585, forks
        assert all(fr and pl == 0 for _, _, fr, pl in forks[1:]), forks
        assert orch.last_stats.prefix_hit_tokens == 585
        # Delta starts OFF-GRID at 585; the first superchunk ends at the
        # next ABSOLUTE stride boundary (the R4b-measured shape change
        # the user accepted).
        assert calls == [(585, 439), (1024, 160)], calls
        lens = sorted(len(e.tokens) for e in orch.prefix_cache._entries)
        assert lens == [512, 599, 1024, 1184], lens
        assert s2.done[1] == chain(prompt2[-1], 8)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_midedge_subgrid_optin_inert_without_arch_capability():
    """V4 refuses REGARDLESS of the switch: without seq_fork_truncatable
    the engine cannot fork at an interior length (INV-SEQ-FORK-TRUNC),
    so even with subgrid_mid_edge=True the legacy grid/exact-body filter
    stays — grid hit, full fork, never a prefix_len."""
    from orchestrator.orchestrator import PrefixCacheConfig
    prompt = chain(31, 600)
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    orch = Orchestrator(
        bridge, metadata=_meta(),        # seq_fork_truncatable=False
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(subgrid_mid_edge=True),
        prefill_superchunk=True,
        prefill_sc_min_tokens=0)
    assert orch.subgrid_mid_edge is True     # ON but inert on this arch
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        div = (prompt[585] + 1) % VOCAB
        prompt2 = prompt[:585] + [div] + chain(div, 599)
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt2, max_tokens=8,
            on_complete=s2.on_complete))
        assert orch.last_stats.prefix_hit_tokens == 512   # grid hit only
        assert daemon.last_fork_prefix == 0               # full fork
        assert s2.done[1] == chain(prompt2[-1], 8)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_glm5next_r4_exclusion_full_fork_only(capsys):
    """GF3.12, the R4 exclusion stated as a test: glm5_next carries a KDA
    recurrent state (lossy in the purest form — no position axis, so
    state@N is algebraically unrecoverable, INV-KDA-REWIND), and exports
    seq_fork_truncatable=False exactly like V4's rings. The orchestrator
    must therefore (a) never call the mid-edge path — the reuse point is
    the deepest REGISTERED length (reuse points are state checkpoints:
    the holder's full-slot copy IS the checkpoint at its length), served
    by a FULL fork (prefix_len 0, state copied); (b) keep the sub-grid
    opt-in INERT even when switched on; (c) announce the third
    holder-cost class (full KDA state copy, spilled to host at
    hibernate) in the boot sizing line."""
    import dataclasses
    from orchestrator.orchestrator import PrefixCacheConfig
    prompt = chain(31, 600)
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    # glm5_next attention-type map: 3 = KDA linear, 4 = sparse NoPE MLA
    # (codes from EngineInfo.attention_types, GF3.2). Truncation stays
    # capability-gated — the property, never the model name.
    att = tuple(3 if i % 4 != 3 else 4 for i in range(45))
    orch = Orchestrator(
        bridge,
        metadata=dataclasses.replace(
            _meta(), seq_fork_truncatable=False, attention_types=att),
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(subgrid_mid_edge=True),  # inert
        prefill_superchunk=True,
        prefill_sc_min_tokens=0)
    boot_out = capsys.readouterr().out
    assert "full per-holder KDA state copy" in boot_out, boot_out
    assert "hibernate spills it" in boot_out, boot_out
    assert orch.subgrid_mid_edge is True      # ON but inert on this arch
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        div = (prompt[585] + 1) % VOCAB
        prompt2 = prompt[:585] + [div] + chain(div, 599)
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt2, max_tokens=8,
            on_complete=s2.on_complete))
        # Divergence at 585: the only LEGAL reuse shape is the full fork
        # of the deepest registered node at 512 (the grid holder). The
        # 585-token LCP is UNREACHABLE: truncation is rejected by the
        # arch (engine belt-and-braces: kSeqFork error on a
        # state-carrying seq — CommandDispatcherKdaState covers it), and
        # no state checkpoint exists at 585.
        assert orch.last_stats.prefix_hit_tokens == 512   # grid hit only
        assert daemon.last_fork_prefix == 0               # full fork
        assert s2.done[1] == chain(prompt2[-1], 8)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_glm5next_holder_cost_line_no_hibernate(capsys):
    """GF3.12: with hibernate_holders OFF on glm5_next the spill never
    runs — the boot line must say the slot is HELD IN VRAM (each holder
    pins a state-pool slot; the pool clamps concurrency)."""
    import dataclasses
    from orchestrator.orchestrator import PrefixCacheConfig
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    att = tuple(3 if i % 4 != 3 else 4 for i in range(45))
    Orchestrator(
        bridge,
        metadata=dataclasses.replace(
            _meta(), seq_fork_truncatable=False, attention_types=att),
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(hibernate_holders=False),
        prefill_superchunk=True,
        prefill_sc_min_tokens=0)
    boot_out = capsys.readouterr().out
    assert "HELD IN VRAM" in boot_out, boot_out
    _finish(daemon)


def test_prefix_midedge_subgrid_env_overrides_config(monkeypatch):
    """LS_ORCH_SUBGRID_MIDEDGE overrides the config field EITHER WAY when
    set (the LAYERSTORM_DETERMINISTIC_EP_COMBINE precedence), and the
    boot NOTICE names the consequence when the effective value is ON."""
    import dataclasses
    from orchestrator.orchestrator import PrefixCacheConfig

    def mk(cfg_on: bool, env: str | None, capsys=None):
        if env is None:
            monkeypatch.delenv("LS_ORCH_SUBGRID_MIDEDGE", raising=False)
        else:
            monkeypatch.setenv("LS_ORCH_SUBGRID_MIDEDGE", env)
        bridge, daemon, _ = _make(gamma=5, use_far=True)
        try:
            orch = Orchestrator(
                bridge,
                metadata=dataclasses.replace(
                    _meta(), seq_fork_truncatable=True),
                prefix_cache=PrefixCacheConfig(subgrid_mid_edge=cfg_on))
            return orch.subgrid_mid_edge
        finally:
            _finish(daemon)

    assert mk(False, None) is False          # default OFF
    assert mk(True, None) is True            # config opts in
    assert mk(False, "1") is True            # env forces ON
    assert mk(True, "0") is False            # env forces OFF (wins)
    assert mk(False, "0") is False


def test_prefix_midedge_subgrid_boot_notice(monkeypatch, capsys):
    import dataclasses
    from orchestrator.orchestrator import PrefixCacheConfig
    monkeypatch.delenv("LS_ORCH_SUBGRID_MIDEDGE", raising=False)
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    try:
        Orchestrator(
            bridge,
            metadata=dataclasses.replace(
                _meta(), seq_fork_truncatable=True),
            prefix_cache=PrefixCacheConfig(subgrid_mid_edge=True))
        out = capsys.readouterr().out
        assert "NOTICE: subgrid_mid_edge is ON" in out
        assert "NOT token-identical" in out
        # OFF boots stay silent.
        Orchestrator(
            bridge,
            metadata=dataclasses.replace(
                _meta(), seq_fork_truncatable=True),
            prefix_cache=PrefixCacheConfig())
        assert "subgrid_mid_edge" not in capsys.readouterr().out
    finally:
        _finish(daemon)


def test_prefix_cache_config_from_config_subgrid_field():
    from orchestrator.orchestrator import PrefixCacheConfig
    assert PrefixCacheConfig.from_config({}).subgrid_mid_edge is False
    cfg = {"_internal-prefix_cache": {"subgrid_mid_edge": True}}
    assert PrefixCacheConfig.from_config(cfg).subgrid_mid_edge is True


# ═══ P-29 step 24 (LS_KDA_PREFIX_CKPT): KDA prefix checkpoints ═════════════

def _kda_orch(daemon_gamma: int = 5, *, kda_kw: dict | None = None,
              env: str | None = None, monkeypatch=None):
    """A glm5_next-shaped orchestrator: non-truncatable forks + a KDA
    state slot exported (TD-GLM5-KDA-SLOTS-EXPORT — the arch witness the
    checkpoint path gates on), superchunk prefill, FakeDaemon in
    kda_arch mode (truncating forks admitted ONLY at checkpoints)."""
    import dataclasses
    from orchestrator.orchestrator import PrefixCacheConfig
    if monkeypatch is not None:
        if env is None:
            monkeypatch.delenv("LS_KDA_PREFIX_CKPT", raising=False)
        else:
            monkeypatch.setenv("LS_KDA_PREFIX_CKPT", env)
    bridge, daemon, _ = _make(gamma=daemon_gamma, use_far=True)
    daemon.kda_arch = True
    att = tuple(3 if i % 4 != 3 else 4 for i in range(45))
    orch = Orchestrator(
        bridge,
        metadata=dataclasses.replace(
            _meta(), seq_fork_truncatable=False, attention_types=att,
            kda_state_slot_bytes=1 << 20),
        speculation=SpeculationConfig(enabled=True, gamma=daemon_gamma,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(**(kda_kw or {})),
        prefill_superchunk=True,
        prefill_sc_min_tokens=0)
    return orch, daemon


def test_kda_ckpt_default_on_and_config_env_gates(monkeypatch):
    """Default ON since 2026-09-06 (user decision; the three byte-identity
    gates are green — golden fork-replay bit-exact, TF byte-compare
    1023/1023 zero mismatches, in-vivo sha == control). Config can turn it
    OFF; env LS_KDA_PREFIX_CKPT overrides EITHER WAY; interval rounds UP to
    a multiple of 512; from_config parses the schema block."""
    from orchestrator.orchestrator import PrefixCacheConfig
    # from_config parsing (schema serving.prefix_cache.kda_checkpoints).
    assert PrefixCacheConfig.from_config({}).kda_ckpt_enabled is True
    cfg = {"serving": {"prefix_cache": {"kda_checkpoints": {
        "enabled": True, "interval_tokens": 1000,
        "budget_mib": 12}}}}
    pcc = PrefixCacheConfig.from_config(cfg)
    assert pcc.kda_ckpt_enabled is True
    # An explicit config false still disables (the flip changed the DEFAULT,
    # not the precedence).
    assert PrefixCacheConfig.from_config(
        {"serving": {"prefix_cache": {"kda_checkpoints": {
            "enabled": False}}}}).kda_ckpt_enabled is False
    assert pcc.kda_ckpt_interval_tokens == 1000
    assert pcc.kda_ckpt_budget_bytes == 12 << 20
    for kda_kw, env, want_on in (
            ({}, None, True),                                    # default ON
            ({"kda_ckpt_enabled": False}, None, False),          # config off
            ({"kda_ckpt_enabled": True}, "0", False),            # env off
            ({"kda_ckpt_enabled": False}, "1", True),            # env on
    ):
        orch, daemon = _kda_orch(kda_kw=kda_kw, env=env,
                                 monkeypatch=monkeypatch)
        try:
            assert orch._kda_ckpt_on is want_on, (kda_kw, env)
            if want_on:
                assert orch._kda_ckpt_iv % 512 == 0
        finally:
            _finish(daemon)
    # Interval rounding: 1000 -> 1024.
    orch, daemon = _kda_orch(
        kda_kw={"kda_ckpt_enabled": True,
                "kda_ckpt_interval_tokens": 1000},
        monkeypatch=monkeypatch)
    try:
        assert orch._kda_ckpt_iv == 1024
    finally:
        _finish(daemon)


def test_kda_ckpt_capture_cadence_and_divergence_restore(monkeypatch):
    """The headline path e2e on the scripted daemon: captures land at
    interval multiples (never at grid_len, never at the prompt end),
    registration MOVES them to the grid holder, and a DIVERGED prompt —
    a whole-node miss today — forks TRUNCATED at the nearest checkpoint
    <= the divergence and replays only the tail. Token stream identical
    to the uncached run of the same prompt (FakeDaemon chain semantics).
    Negative control: with the feature OFF the same divergence is a full
    re-prefill (prefix_hit 0, full fork only)."""
    prompt = chain(41, 1200)                 # pre 1199, grid_len 1024
    # ── control arm first: flag OFF (default) ──
    orch, daemon = _kda_orch(monkeypatch=monkeypatch)
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        assert daemon.kda_ckpt_captures == []          # no captures
        div = (prompt[800] + 1) % VOCAB
        prompt2 = prompt[:800] + [div] + chain(div, 398)
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt2, max_tokens=8,
            on_complete=s2.on_complete))
        # Whole-node lookup misses (every entry is longer than the LCP)
        # and no checkpoint machinery exists: the FULL re-prefill.
        assert orch.last_stats.prefix_hit_tokens == 0
        assert all(p == 0 for p in daemon.fork_prefixes)  # no truncation
        control_tokens = s2.done[1]
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)

    # ── feature arm: flag ON, interval 512 ──
    orch, daemon = _kda_orch(
        kda_kw={"kda_ckpt_enabled": True,
                "kda_ckpt_interval_tokens": 512},
        monkeypatch=monkeypatch)
    try:
        s1, s2, s3 = _Sink(), _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        # Cadence: multiples of 512 in (0, pre) minus grid_len 1024
        # => exactly one capture, at 512, on the request seq — then the
        # grid registration MOVED it to the holder.
        assert [p for (_s, p) in daemon.kda_ckpt_captures] == [512]
        assert orch.last_stats.ckpt_captured == 1
        assert orch.last_stats.ckpt_bytes == daemon.kda_ckpt_bytes_each
        grid_entry = next(e for e in orch.prefix_cache._entries
                          if len(e.tokens) == 1024)
        assert grid_entry.kda_ckpts == (512,)
        assert orch.prefix_cache.kda_ckpt_bytes == \
            daemon.kda_ckpt_bytes_each
        assert daemon.kda_ckpts.get(grid_entry.seq_id) == {512}

        # DIVERGENCE at 800: whole-node lookup misses (both entries are
        # longer than the LCP); the checkpoint path floors 800 -> 512 and
        # forks TRUNCATED at 512.
        div = (prompt[800] + 1) % VOCAB
        prompt2 = prompt[:800] + [div] + chain(div, 398)
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt2, max_tokens=8,
            on_complete=s2.on_complete))
        assert 512 in daemon.fork_prefixes             # truncating fork
        assert orch.last_stats.prefix_hit_tokens == 512
        assert orch.last_stats.ckpt_restore_pos == 512
        assert orch.prefix_cache.kda_ckpt_hits == 1
        # INV-PREFIX-CACHE-1 for the restored class: token-identical to
        # the flag-OFF run of the same diverged prompt.
        assert s2.done[1] == control_tokens == chain(prompt2[-1], 8)
        # The tripwire read 0 the whole way.
        assert orch._kda_ckpt_refusals == 0

        # Deeper divergence in a THIRD prompt sharing [0, 512): the
        # holder keeps its blob — a second checkpoint fork succeeds.
        div3 = (prompt[600] + 3) % VOCAB
        prompt3 = prompt[:600] + [div3] + chain(div3, 300)
        n_trunc = sum(1 for p in daemon.fork_prefixes if p == 512)
        _serve(orch, InferenceRequest(
            request_id=3, prompt_token_ids=prompt3, max_tokens=4,
            on_complete=s3.on_complete))
        assert sum(1 for p in daemon.fork_prefixes
                   if p == 512) == n_trunc + 1
        assert s3.done[1] == chain(prompt3[-1], 4)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_kda_ckpt_divergence_below_first_checkpoint_is_full_miss(
        monkeypatch, capsys):
    """Refusal discipline (INV-KDA-ANCHOR): divergence BELOW the first
    checkpoint => full re-prefill, never approximate — and the miss-LCP
    is logged (the OQ-13 cadence falsifier instrument)."""
    prompt = chain(43, 1200)
    orch, daemon = _kda_orch(
        kda_kw={"kda_ckpt_enabled": True,
                "kda_ckpt_interval_tokens": 512},
        monkeypatch=monkeypatch)
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        capsys.readouterr()
        div = (prompt[100] + 1) % VOCAB      # below the 512 checkpoint
        prompt2 = prompt[:100] + [div] + chain(div, 1098)
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt2, max_tokens=8,
            on_complete=s2.on_complete))
        assert orch.last_stats.prefix_hit_tokens == 0    # full miss
        assert orch.last_stats.ckpt_restore_pos == 0
        assert orch.prefix_cache.last_kda_lcp == 100
        out = capsys.readouterr().out
        assert "KDA divergence miss" in out and "token LCP 100" in out
        assert s2.done[1] == chain(prompt2[-1], 8)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_kda_ckpt_budget_skips_capture_and_eviction_frees(monkeypatch):
    """Budget is enforced at CAPTURE time (skip, keep serving — capacity
    never correctness) and checkpoint bytes are released when their
    entry is evicted (clear() -> _free_holder ledger)."""
    prompt = chain(47, 1200)
    # Budget below one checkpoint's seeded estimate => zero captures.
    orch, daemon = _kda_orch(
        kda_kw={"kda_ckpt_enabled": True,
                "kda_ckpt_interval_tokens": 512,
                "kda_ckpt_budget_bytes": 1024},
        monkeypatch=monkeypatch)
    try:
        s1 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        assert daemon.kda_ckpt_captures == []
        assert orch.last_stats.ckpt_captured == 0
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)
    # Generous budget: capture happens; eviction settles the ledger.
    orch, daemon = _kda_orch(
        kda_kw={"kda_ckpt_enabled": True,
                "kda_ckpt_interval_tokens": 512},
        monkeypatch=monkeypatch)
    try:
        s1 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        assert orch.prefix_cache.kda_ckpt_bytes == \
            daemon.kda_ckpt_bytes_each
        orch.prefix_cache.clear()
        assert orch.prefix_cache.kda_ckpt_bytes == 0
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_lookup_kda_ckpt_ancestor_walk_and_flooring():
    """PrefixCache.lookup_kda_ckpt unit semantics on a hand-built radix:
    largest checkpoint <= min(LCP, len(entry)-1) over the candidate AND
    its ancestors; entries without checkpoints contribute nothing
    (negative control); seed-feed clamp respected."""
    from orchestrator.orchestrator import (PrefixCache, PrefixCacheConfig,
                                           _PrefixEntry)

    class _NullBridge:
        def free_sequence(self, *_a, **_k):
            pass
    pc = PrefixCache(PrefixCacheConfig(), _NullBridge())
    base = list(range(100, 100 + 2048))
    parent = _PrefixEntry(tuple(base[:1024]), seq_id=11)
    parent.kda_ckpts = (512,)
    child = _PrefixEntry(tuple(base[:2048]), seq_id=12)
    child.kda_ckpts = (1536,)
    pc._attach(parent)
    pc._attach(child)

    # Divergence at 1800: child's own 1536 wins.
    p = base[:1800] + [7] + base[1801:2048]
    got = pc.lookup_kda_ckpt(p)
    assert got is not None and got[0] is child and got[1] == 1536
    # Divergence at 1200: child's 1536 is above the LCP — the ANCESTOR's
    # 512 serves it.
    p = base[:1200] + [7] + base[1201:2048]
    got = pc.lookup_kda_ckpt(p)
    assert got is not None and got[0] is parent and got[1] == 512
    # Divergence at 300: below every checkpoint — miss, LCP recorded.
    p = base[:300] + [7] + base[301:2048]
    assert pc.lookup_kda_ckpt(p) is None
    assert pc.last_kda_lcp == 300
    # NEGATIVE CONTROL: strip the checkpoints — the same deep divergence
    # is a miss (deep LCP alone earns nothing on this arch).
    parent.kda_ckpts = ()
    child.kda_ckpts = ()
    p = base[:1800] + [7] + base[1801:2048]
    assert pc.lookup_kda_ckpt(p) is None
    assert pc.last_kda_lcp == 1800
    # Seed-feed clamp: a 2-token prompt can never reuse its full length.
    assert pc.lookup_kda_ckpt(base[:2]) is None


def test_kda_ckpt_deep_checkpoint_beats_shallow_whole_node_hit(monkeypatch):
    """The 97k-rung bug (found in measurement): a SHALLOW registered
    holder that is a proper prefix of a deep diverging prompt must NOT
    short-circuit a far deeper checkpoint fork. Serve an 8k-ish prompt
    (registers a grid holder), then a MUCH longer prompt that shares the
    whole 8k body AND extends far past it with its own checkpoints, then
    a late divergence: the whole-node lookup would land on the shallow
    8k holder, but the checkpoint floor near the divergence must win."""
    short = chain(51, 1100)                   # grid holder at 1024
    orch, daemon = _kda_orch(
        kda_kw={"kda_ckpt_enabled": True,
                "kda_ckpt_interval_tokens": 512},
        monkeypatch=monkeypatch)
    try:
        s1, s2, s3 = _Sink(), _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=short, max_tokens=4,
            on_complete=s1.on_complete))
        # A long prompt that CONTINUES the short one (shares [0,1099]) and
        # runs to 3200 — captures checkpoints at 1536/2048/2560/3072.
        longp = short[:1099] + chain(short[-1], 2101)
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=longp, max_tokens=4,
            on_complete=s2.on_complete))
        assert orch.prefix_cache.kda_ckpt_hits == 0   # none yet
        # Divergence at 2600: the whole-node lookup finds the SHALLOW
        # short-prompt grid holder (1024, a proper prefix) — but the deep
        # holder carries a checkpoint at 2560, which must win.
        div = (longp[2600] + 1) % VOCAB
        prompt3 = longp[:2600] + [div] + chain(div, 300)
        _serve(orch, InferenceRequest(
            request_id=3, prompt_token_ids=prompt3, max_tokens=4,
            on_complete=s3.on_complete))
        assert orch.last_stats.ckpt_restore_pos == 2560, \
            "deep checkpoint must beat the shallow whole-node ancestor"
        assert orch.last_stats.prefix_hit_tokens == 2560
        assert 2560 in daemon.fork_prefixes
        assert s3.done[1] == chain(prompt3[-1], 4)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_assert_identity_preconditions_enforces_dossier_4b(monkeypatch):
    """The shared harness precondition check (dossier 4b, enforced not
    remembered): det-combine must be ON, sub-grid mid-edge OFF, and — on a
    speculative arm — the round-shape governor OFF
    (TD-GLM52-CHAMPION-GREEDY-NONDET-RUN-TO-RUN)."""
    import types
    from orchestrator.orchestrator import assert_identity_preconditions

    ok = types.SimpleNamespace(deterministic_ep_combine=True,
                               subgrid_mid_edge=False)
    assert_identity_preconditions(ok)        # passes silently

    with pytest.raises(AssertionError, match="subgrid_mid_edge=OFF"):
        assert_identity_preconditions(types.SimpleNamespace(
            deterministic_ep_combine=True, subgrid_mid_edge=True))
    with pytest.raises(AssertionError, match="deterministic_ep_combine=ON"):
        assert_identity_preconditions(types.SimpleNamespace(
            deterministic_ep_combine=False, subgrid_mid_edge=False))

    # Precondition 4: speculation armed + governor not explicitly OFF.
    spec_on = types.SimpleNamespace(
        deterministic_ep_combine=True, subgrid_mid_edge=False,
        spec=types.SimpleNamespace(enabled=True))
    monkeypatch.delenv("LS_SPEC_GOVERNOR", raising=False)
    with pytest.raises(AssertionError, match="LS_SPEC_GOVERNOR=0"):
        assert_identity_preconditions(spec_on)
    monkeypatch.setenv("LS_SPEC_GOVERNOR", "1")
    with pytest.raises(AssertionError, match="LS_SPEC_GOVERNOR=0"):
        assert_identity_preconditions(spec_on)
    monkeypatch.setenv("LS_SPEC_GOVERNOR", "0")
    assert_identity_preconditions(spec_on)   # governor OFF → passes
    # Speculation disabled → the governor is irrelevant.
    monkeypatch.delenv("LS_SPEC_GOVERNOR", raising=False)
    assert_identity_preconditions(types.SimpleNamespace(
        deterministic_ep_combine=True, subgrid_mid_edge=False,
        spec=types.SimpleNamespace(enabled=False)))


def test_assert_identity_preconditions_fires_on_real_subgrid_orch(
        monkeypatch):
    """A REAL orchestrator booted with the switch ON trips the harness
    assertion — the enforcement path the gates rely on."""
    import dataclasses
    from orchestrator.orchestrator import (PrefixCacheConfig,
                                           assert_identity_preconditions)
    monkeypatch.setenv("LS_ORCH_SUBGRID_MIDEDGE", "1")
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    try:
        orch = Orchestrator(
            bridge,
            metadata=dataclasses.replace(
                _meta(), seq_fork_truncatable=True),
            prefix_cache=PrefixCacheConfig())     # config default OFF
        with pytest.raises(AssertionError, match="dossier 4b"):
            assert_identity_preconditions(orch)
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


def test_prefix_cache_long_prompt_registers_and_hits_default_config():
    """User report 2026-08-26: prefix caching went DEAD above ~10k ctx —
    PrefixCacheConfig.max_cached_tokens (8192) silently refused every
    long-prompt registration AND doubled as the total eviction budget.
    Under the DEFAULT config a >8192-token prompt must now register and a
    repeat request must hit it."""
    prompt = chain(91, 8260)             # pre 8259 → grid holder at 8256
    orch, daemon = _orch_pc()
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=3,
            on_complete=s1.on_complete))
        assert daemon.forks == 1         # 8256 > old 8192 cap: registered
        assert orch.prefix_cache._entries[0].tokens == tuple(prompt[:8256])
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=3,
            on_complete=s2.on_complete))
        assert s2.done[1] == s1.done[1] == chain(prompt[-1], 3)
        assert orch.last_stats.prefix_hit_tokens == 8256
        assert orch.prefix_cache.hits == 1
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefix_cache_entry_cap_and_total_budget_independent():
    """The PER-ENTRY cap (_internal-prefix_cache.max_entry_tokens) and the
    TOTAL budget (serving.prefix_cache.max_cached_tokens) are independent
    knobs, independently enforced."""
    # Per-entry cap refuses ONLY over-cap entries; the budget never fires.
    pc = _pc(max_entry_tokens=100, max_cached_tokens=10_000)
    assert not pc.register(tuple(range(150)), src_seq_id=1, holder_seq_id=2)
    assert "over-sized" in pc.last_skip
    assert "max_entry_tokens" in pc.last_skip
    _reg(pc, list(range(90)), 1)             # under the cap: registers
    assert pc.last_skip is None
    assert pc.evictions == 0
    # No separate cap (0): an entry longer than the WHOLE budget can never
    # fit and is refused up front...
    pc2 = _pc(max_entry_tokens=0, max_cached_tokens=200)
    assert not pc2.register(tuple(range(250)), src_seq_id=1, holder_seq_id=2)
    assert "over-sized" in pc2.last_skip
    assert "max_cached_tokens" in pc2.last_skip
    # ...while under-budget entries register and BUDGET pressure (not the
    # cap) evicts LRU.
    _reg(pc2, list(range(150)), 1)
    _reg(pc2, list(range(500, 650)), 2)      # unique 300 > 200 → evict #1
    assert {e.seq_id for e in pc2._entries} == {2}
    assert pc2.evictions == 1


def test_prefix_cache_skipped_registration_is_logged(capsys):
    """A skipped registration must not be silent (the pre-2026-08-26
    silent over-sized skip read as a wall-clock regression): the serve
    log carries one line per request with the reason."""
    # over-sized: per-entry cap below the grid holder length.
    from orchestrator.orchestrator import PrefixCacheConfig
    orch, daemon = _orch_pc(max_entry_tokens=64)
    try:
        prompt = chain(31, 130)              # grid holder at 128 > cap 64
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=3,
            on_complete=sink.on_complete))
        out = capsys.readouterr().out
        assert "prefix-cache registration skipped" in out
        assert "over-sized" in out and "max_entry_tokens 64" in out
        assert daemon.forks == 0             # nothing registered
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)
    # duplicate: a repeat request re-attempts the same grid holder.
    orch, daemon = _orch_pc()
    try:
        prompt = chain(37, 130)
        for rid in (1, 2):
            sink = _Sink()
            _serve(orch, InferenceRequest(
                request_id=rid, prompt_token_ids=prompt, max_tokens=3,
                on_complete=sink.on_complete))
        out = capsys.readouterr().out
        assert "request 2: prefix-cache registration skipped" in out
        assert "duplicate" in out
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


# ── SC small-prefill downgrade (user report 2026-08-26): a superchunk
# sweeps the whole delta's expert UNION in one MOE_BIG per layer, evicting
# the decode-warmed expert cache — below prefill_sc_min_tokens the delta
# runs the ordinary chunked path in small increments instead.  Adaptive
# successor: TD-SC-SMALL-PREFILL-ADAPTIVE. ─────────────────────────────────


def _orch_sc(sc_min: int, **kw):
    from orchestrator.orchestrator import PrefixCacheConfig
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    orch = Orchestrator(
        bridge, metadata=_meta(),
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(),
        prefill_superchunk=True,
        prefill_sc_min_tokens=sc_min, **kw)
    return orch, daemon


def test_sc_small_prefill_routes_chunked_below_threshold():
    prompt = chain(43, 130)                  # pre 129 < 256
    orch, daemon = _orch_sc(256)
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=4,
            on_complete=s1.on_complete))
        # Chunked path: NO whole-delta MOE_BIG union sweep; small chunks.
        assert daemon.fetch_moe_bigs == 0
        assert daemon.embed_calls[:2] == [64, 64]
        # Registration mirrors the sc branch: the EXACT-body holder is
        # still registered (SC-grid policy) and a repeat request hits it.
        assert daemon.forks == 1
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=4,
            on_complete=s2.on_complete))
        assert s2.done[1] == s1.done[1] == chain(prompt[-1], 4)
        assert orch.last_stats.prefix_hit_tokens == 129
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_sc_small_prefill_threshold_boundary_is_strictly_below():
    prompt = chain(47, 131)                  # pre (delta) = 130
    # delta == threshold → SUPERCHUNK (threshold is strictly-below).
    orch, daemon = _orch_sc(130)
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=2,
            on_complete=sink.on_complete))
        assert daemon.fetch_moe_bigs > 0
    finally:
        _finish(daemon)
    # delta == threshold - 1 → chunked.
    orch, daemon = _orch_sc(131)
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=2,
            on_complete=sink.on_complete))
        assert daemon.fetch_moe_bigs == 0
    finally:
        _finish(daemon)


def test_sc_above_threshold_byte_identical_to_no_threshold():
    """The threshold must be invisible above itself: an above-threshold
    prefill issues the exact same command stream as a threshold-0 boot
    (the champion 8k/20k/25k profiles route identically)."""
    prompt = chain(53, 600)                  # pre 599 >= 256
    traces = []
    for sc_min in (0, 256):
        orch, daemon = _orch_sc(sc_min)
        try:
            sink = _Sink()
            _serve(orch, InferenceRequest(
                request_id=1, prompt_token_ids=prompt, max_tokens=5,
                on_complete=sink.on_complete))
            traces.append((daemon.embed_calls, daemon.fetch_moe_bigs,
                           daemon.moe_big_rows, daemon.forks,
                           sink.done[1]))
            assert not daemon.errors, daemon.errors
        finally:
            _finish(daemon)
    assert traces[0] == traces[1]


def test_sc_small_delta_on_cache_hit_routes_chunked():
    """The downgrade decision input is the DELTA actually prefilled (the
    eviction pressure), not the prompt length: a short follow-up on a
    cached long prefix takes the chunked path."""
    prompt = chain(59, 600)                  # pre 599: sc path, registers
    orch, daemon = _orch_sc(256)             # grid 512 + exact 599 holders
    try:
        s1, s2, s3 = _Sink(), _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=4,
            on_complete=s1.on_complete))
        bigs_after_1 = daemon.fetch_moe_bigs
        assert bigs_after_1 > 0 and daemon.forks == 2
        # Follow-up: shares the 512-grid prefix, delta 99 < 256.
        follow = prompt[:512] + chain(888, 100)
        mark = len(daemon.embed_calls)
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=follow, max_tokens=4,
            on_complete=s2.on_complete))
        assert s2.done[1] == chain(follow[-1], 4)
        assert orch.last_stats.prefix_hit_tokens == 512
        assert daemon.fetch_moe_bigs == bigs_after_1   # no new MOE_BIG
        # 99-delta prefill = small chunks [64, 35] (decode embeds follow).
        assert daemon.embed_calls[mark:mark + 2] == [64, 35]
        # Downgraded registration stays on the SC lookup grid: the new
        # EXACT-body holder (611) serves a full repeat.
        _serve(orch, InferenceRequest(
            request_id=3, prompt_token_ids=follow, max_tokens=4,
            on_complete=s3.on_complete))
        assert s3.done[1] == s2.done[1]
        assert orch.last_stats.prefix_hit_tokens == 611
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


def test_prefill_evicts_holder_on_indexer_pool_exhaustion():
    """TD-INDEXER-POOL-EVICT: kIndexerK exhaustion DURING PREFILL is pool
    pressure, not a request failure. The engine fail-closes the step (a
    dense downgrade would punch a permanent hole in the sequence's indexer
    coverage and, on a KV-demoted sequence, force a full cold-page
    re-promotion — the 2026-08-24 serving incident); the orchestrator
    answers by freeing a prefix holder, whose indexer pages die with its
    sequence, and re-issuing the identical step."""
    orch, daemon = _orch_pc()
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_token=s1.on_token, on_complete=s1.on_complete))
        assert len(orch.prefix_cache._entries) == 1      # holder retained

        # The pool stays exhausted until ONE more sequence is freed — i.e.
        # until a holder is evicted (the working sequence is still alive).
        daemon.indexer_exhaust_until_frees = daemon.seq_frees + 1

        prompt_b = chain(500, 130)                       # different prefix
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt_b, max_tokens=8,
            on_token=s2.on_token, on_complete=s2.on_complete))

        # Rejected, then served — token-identical to an uncached run.
        assert daemon.indexer_exhaust_rejects >= 1
        assert orch.prefix_cache.evictions >= 1
        assert orch._pool_evict_retries >= 1
        assert s2.done[2] == "length"
        assert s2.done[1] == chain(prompt_b[-1], 8)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_prefill_pool_exhaustion_reraises_when_nothing_evictable():
    """The retry is bounded by what is actually reclaimable: with no holder
    left to free, the exhaustion surfaces as the request error it is —
    never an infinite evict/retry spin."""
    orch, daemon = _orch_pc()
    try:
        s = _Sink()
        daemon.indexer_exhaust_until_frees = 10**6      # never satisfiable
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_token=s.on_token, on_complete=s.on_complete))
        assert s.done[2] == "error", s.done
        assert daemon.indexer_exhaust_rejects >= 1
        # No holder existed yet (registration happens mid-prefill, after the
        # first chunk), so nothing was evictable and the retry did not spin.
        assert orch._pool_evict_retries == 0
    finally:
        _finish(daemon)


# ── TD-KVXP-PER-STEP-FLOOR: bounded LARGE-prefill wait ─────────────────────


@contextlib.contextmanager
def _kvxp_env(**kw: str):
    """Set (or clear, on None) the wait's env knobs for the duration of the
    block and restore the prior process environment afterwards — the knobs
    are read once, in Orchestrator.__init__, so the orchestrator must be
    built INSIDE the block, and no leakage may reach the neighbours."""
    keys = ("LS_KV_EXPERT_REBALANCE", "LS_KVXP_LARGE_PREFILL_TOKENS",
            "LS_KVXP_LARGE_PREFILL_WAIT_MS")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            v = kw.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_large_prefill_bounded_wait_rescues_request():
    """TD-KVXP-PER-STEP-FLOOR: a LARGE prefill that exhausts a pool with
    NOTHING left to evict must not fail the request while the 44z eager
    background drain (armed by that very refusal) is still in flight — it
    waits, bounded, and re-issues the identical step. The wait sits ABOVE
    the engine's all-or-nothing admission claim, so INV-KDA-STATE (a) is
    untouched: every re-issue is a fresh, fully-rolled-back attempt."""
    with _kvxp_env(LS_KV_EXPERT_REBALANCE="1",
                   LS_KVXP_LARGE_PREFILL_TOKENS="64",
                   LS_KVXP_LARGE_PREFILL_WAIT_MS="3000"):
        orch, daemon = _orch_pc()           # env read here, in __init__
    prompt = chain(11, 130)                 # >= threshold → armed
    # Nothing is evictable (no holder exists yet mid-prefill), so the
    # refusal reaches the wait — and capacity only comes back the way it
    # does on the box: asynchronously, from the background drain.
    daemon.indexer_exhaust_until_frees = 10**6
    drain = threading.Timer(
        0.4, lambda: setattr(daemon, "indexer_exhaust_until_frees", 0))
    try:
        drain.start()
        s = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_token=s.on_token, on_complete=s.on_complete))
        assert s.done[2] == "length", s.done
        assert s.done[1] == chain(prompt[-1], 8)
        assert daemon.indexer_exhaust_rejects >= 1
        assert orch._pool_evict_retries == 0        # nothing was evictable
        assert orch._kvxp_wait_engagements >= 1
        assert orch._kvxp_wait_rescues == 1
        assert not daemon.errors, daemon.errors
    finally:
        drain.cancel()
        drain.join()
        _finish(daemon)


def test_short_prefill_never_waits():
    """The wait is for LARGE prefills only: a prompt under the threshold
    surfaces the refusal at once (a short TTFT must never be inflated by a
    drain wait), exactly as before TD-KVXP-PER-STEP-FLOOR."""
    with _kvxp_env(LS_KV_EXPERT_REBALANCE="1",
                   LS_KVXP_LARGE_PREFILL_TOKENS="100000",
                   LS_KVXP_LARGE_PREFILL_WAIT_MS="3000"):
        orch, daemon = _orch_pc()
    try:
        daemon.indexer_exhaust_until_frees = 10**6      # never satisfiable
        s = _Sink()
        t0 = time.monotonic()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_token=s.on_token, on_complete=s.on_complete))
        elapsed = time.monotonic() - t0
        assert s.done[2] == "error", s.done
        assert daemon.indexer_exhaust_rejects >= 1
        assert orch._kvxp_wait_engagements == 0
        assert orch._kvxp_wait_rescues == 0
        assert elapsed < 1.5, f"short prefill waited {elapsed:.2f}s"
    finally:
        _finish(daemon)


def test_wait_disabled_without_rebalancer_env():
    """Without the 44z rebalancer no drain is coming, so waiting could only
    delay the 500: the knob pair alone must not arm the wait."""
    with _kvxp_env(LS_KV_EXPERT_REBALANCE=None,
                   LS_KVXP_LARGE_PREFILL_TOKENS="64",
                   LS_KVXP_LARGE_PREFILL_WAIT_MS="3000"):
        orch, daemon = _orch_pc()
    try:
        daemon.indexer_exhaust_until_frees = 10**6      # never satisfiable
        s = _Sink()
        t0 = time.monotonic()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_token=s.on_token, on_complete=s.on_complete))
        elapsed = time.monotonic() - t0
        assert s.done[2] == "error", s.done
        assert orch._kvxp_wait_engagements == 0
        assert elapsed < 1.5, f"disabled wait still slept {elapsed:.2f}s"
    finally:
        _finish(daemon)


def test_wait_budget_bounded():
    """The wait is a CAP on futile waiting, never an unbounded stall: when
    capacity never returns, the request burns its per-request budget and
    then surfaces the same refusal it would have surfaced immediately."""
    with _kvxp_env(LS_KV_EXPERT_REBALANCE="1",
                   LS_KVXP_LARGE_PREFILL_TOKENS="64",
                   LS_KVXP_LARGE_PREFILL_WAIT_MS="300"):
        orch, daemon = _orch_pc()
    try:
        daemon.indexer_exhaust_until_frees = 10**6      # never satisfiable
        s = _Sink()
        t0 = time.monotonic()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_token=s.on_token, on_complete=s.on_complete))
        elapsed = time.monotonic() - t0
        assert s.done[2] == "error", s.done
        assert orch._kvxp_wait_engagements >= 1
        assert orch._kvxp_wait_rescues == 0             # prefill never ran
        assert elapsed >= 0.3, f"budget under-spent ({elapsed:.2f}s)"
        assert elapsed < 2.0, f"budget over-spent ({elapsed:.2f}s)"
    finally:
        _finish(daemon)


def test_kvxp_wait_knobs_config_and_env_precedence():
    """TD-KVXP-SCHEMA-KNOBS: the wait knobs are schema-backed
    (_internal-orchestrator.kvxp_large_prefill_tokens / _wait_ms via the
    __init__ params from_config passes) and the master switch mirrors
    _internal-kv_expert_rebalance.enabled; each LS_* env var overrides
    EITHER WAY when set and leaves the config value in force when unset."""
    from orchestrator.orchestrator import PrefixCacheConfig

    def build(**kw):
        bridge, daemon, _ = _make(gamma=5, use_far=True)
        orch = Orchestrator(
            bridge, metadata=_meta(),
            speculation=SpeculationConfig(enabled=True, gamma=5,
                                          conf_thresh=0.0),
            prefix_cache=PrefixCacheConfig(), **kw)
        _finish(daemon)
        return orch

    # config alone (env unset) IS the policy
    with _kvxp_env():
        orch = build(kvxp_large_prefill_tokens=64,
                     kvxp_large_prefill_wait_ms=500,
                     kv_expert_rebalance_enabled=True)
        assert orch._kvxp_wait_threshold == 64
        assert orch._kvxp_wait_budget_ms == 500
        assert orch._kvxp_wait_enabled
        # config-off master switch disarms the wait even with live knobs
        orch = build(kvxp_large_prefill_tokens=64,
                     kvxp_large_prefill_wait_ms=500,
                     kv_expert_rebalance_enabled=False)
        assert not orch._kvxp_wait_enabled
    # a SET env var wins either way
    with _kvxp_env(LS_KV_EXPERT_REBALANCE="0",
                   LS_KVXP_LARGE_PREFILL_TOKENS="128"):
        orch = build(kvxp_large_prefill_tokens=64,
                     kvxp_large_prefill_wait_ms=500,
                     kv_expert_rebalance_enabled=True)
        assert orch._kvxp_wait_threshold == 128
        assert not orch._kvxp_wait_enabled, "env off must beat config on"
    with _kvxp_env(LS_KV_EXPERT_REBALANCE="1"):
        orch = build(kv_expert_rebalance_enabled=False)
        assert orch._kvxp_wait_enabled, "env on must beat config off"
        # and the defaults match the schema defaults when nothing is set
        assert orch._kvxp_wait_threshold == 8192
        assert orch._kvxp_wait_budget_ms == 2000


def test_prefill_evicts_holder_on_v4_side_tier_exhaustion():
    """INV-PREFIX-CACHE-3 seam (2026-08-26 incident): V4 side-tier
    exhaustion (kSwa/kHca/kIndexerK — COPY-ON-FORK holder cost) arrives as
    a CMP_ERROR whose 80-byte message field TRUNCATED the word "exhausted"
    to "...pool exha".  The evict-retry seam must recognize retryability
    from error_category=kKvPoolExhausted and answer exactly like indexer-K
    exhaustion: free one holder, re-issue the identical step."""
    orch, daemon = _orch_pc()
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_token=s1.on_token, on_complete=s1.on_complete))
        assert len(orch.prefix_cache._entries) == 1      # holder retained

        # The regression shape: the delivered message must NOT contain
        # "exhausted" (it truncates at "exha") — the category alone must
        # carry the retry decision.
        wire_msg = FakeDaemon.V4_TIER_EXHAUST_MSG.encode()[:79].decode()
        assert "exhausted" not in wire_msg and "pool exha" in wire_msg

        daemon.v4_tier_exhaust_until_frees = daemon.seq_frees + 1
        prompt_b = chain(500, 130)                       # different prefix
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt_b, max_tokens=8,
            on_token=s2.on_token, on_complete=s2.on_complete))

        assert daemon.v4_tier_exhaust_rejects >= 1
        assert orch.prefix_cache.evictions >= 1
        assert orch._pool_evict_retries >= 1
        assert s2.done[2] == "length"
        assert s2.done[1] == chain(prompt_b[-1], 8)      # token-identical
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


# ── TD-INDEXER-NO-DENSE-FALLBACK (Route 1): reserve-at-admission ───────────


def test_admission_reservation_wired_and_grant_returned():
    """Every admission carries an indexer-K reservation for the context the
    request may reach — prompt + max_tokens + the speculative-overshoot
    margin (gamma + 2) — a HIT child's fork carries the same target, and
    frozen holder registrations reserve nothing (pure refcount share)."""
    prompt = chain(11, 130)
    orch, daemon = _orch_pc()
    try:
        s1 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_token=s1.on_token, on_complete=s1.on_complete))
        margin = orch.spec.gamma + 2
        assert daemon.last_create_reserve == len(prompt) + 8 + margin
        assert daemon.frozen_forks >= 1           # holder registered
        assert daemon.last_fork_reserve == 0      # frozen: reserve zeroed
        # Exact-prompt repeat: the hit CHILD fork reserves for ITS request.
        s2 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=8,
            on_token=s2.on_token, on_complete=s2.on_complete))
        assert orch.last_stats.prefix_hit_tokens > 0
        assert daemon.last_fork_reserve == len(prompt) + 8 + margin
        assert s2.done[1] == s1.done[1]           # hit stays token-identical
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_create_reservation_exhaustion_evicts_and_retries():
    """A create whose RESERVATION cannot be satisfied is a RETRYABLE
    refusal (category kKvPoolExhausted): the admission seam evicts a
    prefix holder and re-issues — what used to become a silent mid-prefill
    dense downgrade now never admits under-capacity at all."""
    orch, daemon = _orch_pc()
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_token=s1.on_token, on_complete=s1.on_complete))
        assert len(orch.prefix_cache._entries) == 1      # holder retained

        # Reservation refused until ONE more sequence is freed (= a holder
        # eviction; the live holder pins the pool).
        daemon.reserve_exhaust_until_frees = daemon.seq_frees + 1
        prompt_b = chain(500, 130)                       # different prefix
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt_b, max_tokens=8,
            on_token=s2.on_token, on_complete=s2.on_complete))

        assert daemon.reserve_exhaust_rejects >= 1
        assert orch.prefix_cache.evictions >= 1
        assert s2.done[2] == "length"
        assert s2.done[1] == chain(prompt_b[-1], 8)      # served normally
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_reservation_grant_caps_generation():
    """The engine's GRANT (clamped to its serving window) caps generation:
    a request that would decode past the reserved context finishes with
    reason "length" AT the cap — the old behavior was a mid-request dense
    downgrade (or failure) at the window edge."""
    prompt = chain(11, 130)
    orch, daemon = _orch_pc()
    try:
        margin = orch.spec.gamma + 2
        daemon.max_seq_tokens = len(prompt) + margin + 3  # room for 3 tokens
        s = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_token=s.on_token, on_complete=s.on_complete))
        assert s.done[2] == "length"
        assert s.done[1] == chain(prompt[-1], 3)          # capped at 3
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_indexer_dense_step_is_loud_and_blocks_registration():
    """The kDead witness: an attention completion flagged indexer_dense
    surfaces on RequestStats (the [orch-stats] counter) and the request
    registers NO prefix holder (its indexer coverage has a hole — a frozen
    holder would hand every hit-child a permanently dense start and break
    INV-PREFIX-CACHE-1)."""
    prompt = chain(11, 130)
    orch, daemon = _orch_pc()
    try:
        daemon.dense_flag_remaining = 2      # two dense steps in request 1
        s1 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_token=s1.on_token, on_complete=s1.on_complete))
        assert orch.last_stats.indexer_dense_steps == 2
        assert daemon.forks == 0, "dense request registered a holder"
        # Healthy identical follow-up: registers normally, stats read zero.
        s2 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=8,
            on_token=s2.on_token, on_complete=s2.on_complete))
        assert orch.last_stats.indexer_dense_steps == 0
        assert daemon.forks >= 1
        assert not daemon.errors, daemon.errors
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


def test_v4_side_tier_exhaustion_truncated_error_when_nothing_evictable():
    """With no holder to evict, the truncated side-tier CMP_ERROR surfaces
    verbatim ("...pool exha") as the request error — proving the retry in
    test_prefill_evicts_holder_on_v4_side_tier_exhaustion was carried by
    the error CATEGORY, not by a message substring."""
    orch, daemon = _orch_pc()
    try:
        daemon.v4_tier_exhaust_until_frees = 10**6      # never satisfiable
        sink = _ErrSink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_complete=sink.on_complete))
        assert sink.done[2] == "error"
        assert "pool exha" in sink.error
        assert "exhausted" not in sink.error             # truncation fidelity
        assert daemon.v4_tier_exhaust_rejects >= 1
        assert orch._pool_evict_retries == 0             # nothing evictable
    finally:
        _finish(daemon)


# ── mid-spec-round pool exhaustion (TD-SPEC-ROUND-POOL-EVICT) ──────────────
# The overlap plain step runs with a dspark draft ASYNCHRONOUSLY in flight;
# the FakeDaemon holds that draft's completion, declines attention with the
# 2026-08-26 CMP replayed byte-for-byte (80-byte truncation, category 29),
# and releases the held completion only INSIDE the eviction's SEQ_FREE —
# the exact engine interleave the seam must survive.


def test_mid_spec_round_exhaustion_evicts_holder_and_retries():
    """Pool exhaustion on a MID-round step evicts a holder and re-issues
    the identical step: token-identical output, acceptance stats counted
    once, draft completion stashed across the free, and the retry counted
    DISTINCTLY from the prefill/plain/seed retries."""
    orch, daemon = _orch_pc()
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_token=s1.on_token, on_complete=s1.on_complete))
        assert len(orch.prefix_cache._entries) == 1      # holder retained

        # Request 2's FIRST overlap round: its draft is held "in flight";
        # while held, every attention command is declined with the
        # truncated V4 CMP (category kKvPoolExhausted, no "exhausted").
        daemon.spec_round_exhaust_at_call = daemon.dspark_calls
        prompt_b = chain(500, 130)                       # different prefix
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt_b, max_tokens=8,
            on_token=s2.on_token, on_complete=s2.on_complete))

        assert daemon.spec_round_exhaust_rejects >= 1    # it WAS declined
        assert daemon.held_dspark_releases == 1          # freed UNDER draft
        assert orch.prefix_cache.evictions >= 1
        assert orch._spec_round_pool_evict_retries >= 1  # distinct counter
        assert orch._pool_evict_retries == 0             # not conflated
        assert s2.done[2] == "length"
        assert s2.done[1] == chain(prompt_b[-1], 8)      # token-identical
        st = orch.last_stats
        assert st.spec_fallback_round == 0               # round survived
        assert st.accepted <= st.proposed                # single-counted
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_mid_spec_round_exhaustion_nothing_evictable_fails_cleanly():
    """Mid-round exhaustion with NO holder to evict: the truncated CMP
    surfaces as the request error (truncation fidelity intact — proving
    the CATEGORY carried the mid-round retry decision, INV-IPC-ERRMSG-80),
    the pending draft is drained, the abandoned attempt frees its
    sequence, and no prefix holder is registered."""
    orch, daemon = _orch_pc()
    try:
        # Short prompt: below the 64-token registration grid, so no holder
        # exists and nothing is evictable.
        daemon.spec_round_exhaust_at_call = 0
        daemon.spec_round_exhaust_max_rejects = 1  # draft completes anyway
        sink = _ErrSink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 40), max_tokens=8,
            on_complete=sink.on_complete))
        assert sink.done[2] == "error"
        assert "pool exha" in sink.error
        assert "exhausted" not in sink.error             # truncation fidelity
        assert daemon.spec_round_exhaust_rejects == 1
        assert orch._spec_round_pool_evict_retries == 0  # no retry happened
        assert daemon.held_dspark_releases == 1          # drained, not leaked
        assert daemon.known_seqs == set()                # sequence freed
        assert daemon.forks == 0                         # no holder existed
    finally:
        _finish(daemon)


def test_mid_spec_round_exhaustion_draft_error_during_eviction_falls_back():
    """The draft FAILS while its completion is in flight across the
    eviction: the CMP_ERROR lands inside the SEQ_FREE wait, must be
    stashed (not kill the free), the retried step must succeed, and the
    round's collect surfaces DsparkDraftError -> spec->plain fallback
    (INV-SERVE-SPEC-FALLBACK) — still token-identical."""
    orch, daemon = _orch_pc()
    try:
        s1, s2 = _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_token=s1.on_token, on_complete=s1.on_complete))
        assert len(orch.prefix_cache._entries) == 1

        daemon.spec_round_exhaust_at_call = daemon.dspark_calls
        daemon.spec_round_draft_error = True
        prompt_b = chain(500, 130)
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt_b, max_tokens=8,
            on_token=s2.on_token, on_complete=s2.on_complete))

        assert daemon.spec_round_exhaust_rejects >= 1
        assert daemon.held_dspark_releases == 1
        assert orch._spec_round_pool_evict_retries >= 1
        assert s2.done[2] == "length"
        assert s2.done[1] == chain(prompt_b[-1], 8)      # token-identical
        assert orch.last_stats.spec_fallback_round >= 1  # fell back plainly
        assert not daemon.errors, daemon.errors
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
        def boom(req, **kw):
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
        self.frozen: list[bool] = []          # per-fork frozen flag (R3)
        self.prefix_lens: list[int] = []      # per-fork prefix_len (R4c)
        self.freed: list[int] = []
        self.hibernated: list[int] = []       # CMD_SEQ_HIBERNATE (R3)
        self.hibernate_error: Exception | None = None

    def fork_sequence(self, src: int, dst: int,
                      frozen: bool = False, prefix_len: int = 0) -> None:
        self.forks.append((src, dst))
        self.frozen.append(frozen)
        self.prefix_lens.append(prefix_len)

    def free_sequence(self, seq_id: int) -> None:
        self.freed.append(seq_id)

    def hibernate_sequence(self, seq_id: int, kv_len: int) -> None:
        if self.hibernate_error is not None:
            raise self.hibernate_error
        self.hibernated.append((seq_id, kv_len))


def _pc(**cfg_kw):
    from orchestrator.orchestrator import PrefixCache, PrefixCacheConfig
    return PrefixCache(PrefixCacheConfig(**cfg_kw), _StubBridge())


def _reg(pc, tokens, seq_id):
    assert pc.register(tuple(tokens), src_seq_id=1000 + seq_id,
                       holder_seq_id=seq_id)


# ── R3 registration-point density riders (TD-PREFIX-POOL-PRESSURE-EVICTS-
# THE-PRIZE): frozen registration forks, holder hibernation, and
# pool-pressure retirement of extended non-grid (exact-body) nodes. ────────


def test_prefix_register_frozen_fork_then_hibernates():
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    _reg(pc, list(range(64)), 1)
    # Registration fork is FROZEN (engine-side CoW-free)...
    assert pc._bridge.forks == [(1001, 1)]
    assert pc._bridge.frozen == [True]
    # ...and the holder is hibernated right after, with its KV coverage
    # (the engine demotes only pages strictly below kv_len/page_size).
    assert pc._bridge.hibernated == [(1, 64)]
    # A HIT fork is a normal CoW fork (the child appends).
    entry = pc.lookup(list(range(64)) + [999])
    pc.fork_from(entry, 77)
    assert pc._bridge.forks[-1] == (1, 77)
    assert pc._bridge.frozen[-1] is False
    assert pc._bridge.hibernated == [(1, 64)]   # hits never hibernate


def test_prefix_hibernate_failure_is_failsafe():
    from bridge.ring_bridge import BridgeError
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    pc._bridge.hibernate_error = BridgeError("seq_hibernate 1 status 1")
    _reg(pc, list(range(64)), 1)             # still registers
    assert len(pc._entries) == 1
    assert pc._bridge.hibernated == []
    assert pc._hibernate_warned


def test_prefix_hibernate_disabled_by_config():
    pc = _pc(max_entries=8, max_cached_tokens=10_000,
             hibernate_holders=False)
    _reg(pc, list(range(64)), 1)
    assert pc._bridge.frozen == [True]       # frozen fork regardless
    assert pc._bridge.hibernated == []


def test_prefix_config_parses_hibernate_holders():
    from orchestrator.orchestrator import PrefixCacheConfig
    cfg = PrefixCacheConfig.from_config(
        {"_internal-prefix_cache": {"hibernate_holders": False}})
    assert cfg.hibernate_holders is False
    assert PrefixCacheConfig.from_config({}).hibernate_holders is True


def test_prefix_non_grid_interior_retirable_under_pool_pressure():
    # Churn ticket must-hold 3: an extended EXACT-BODY node (non-grid
    # length, has children) is admissible only for an exact repeat — it
    # must be RETIRABLE under pool pressure, never immortal.  The byte-
    # budget order stays leaf-only (retiring a zero-byte interior there
    # would spin).
    pc = _pc(max_entries=8, max_cached_tokens=100_000)
    pc.sc_grid = 512
    A = list(range(1000))                    # non-grid exact body
    _reg(pc, A, 1)
    _reg(pc, A + list(range(2000, 2560)), 2)     # extension → A is interior
    node_a = next(e for e in pc._entries if len(e.tokens) == 1000)
    assert node_a.children, "A must be an interior node"
    # Byte-budget order: leaf only.
    assert node_a not in pc._victim_order()
    # Pool-pressure order: A included (oldest stamp → first victim).
    assert node_a in pc._pool_victim_order()
    assert pc._evict_one(protect=None)
    assert node_a not in pc._entries
    assert 1 in pc._bridge.freed             # holder sequence freed
    # The extension survives and reparents to a root.
    assert [len(e.tokens) for e in pc._entries] == [1560]


def test_prefix_pool_eviction_falls_back_to_grid_interiors():
    # R3 gate finding: frozen chain holders share one indexer-K group per
    # generation, so freeing leaves may free nothing the engine needs.
    # Under sustained pool pressure _evict_one must keep making progress
    # (LRU-major) until the cache is empty — never leave the live request
    # stuck while any entry remains.
    pc = _pc(max_entries=8, max_cached_tokens=100_000)
    pc.sc_grid = 512
    A = list(range(512))
    _reg(pc, A, 1)
    _reg(pc, A + list(range(2000, 2512)), 2)     # 512 becomes grid interior
    assert pc._evict_one(protect=None)           # evicts the leaf (1024)
    assert pc._evict_one(protect=None)           # FALLBACK: grid interior
    assert pc._entries == []
    assert not pc._evict_one(protect=None)
    assert sorted(pc._bridge.freed) == [1, 2]


def test_prefix_grid_interior_not_pool_retirable():
    # A GRID-ALIGNED interior stays protected under pool pressure (its
    # descendants extend it and future prompts hit it).
    pc = _pc(max_entries=8, max_cached_tokens=100_000)
    pc.sc_grid = 512
    A = list(range(512))
    _reg(pc, A, 1)
    _reg(pc, A + list(range(2000, 2512)), 2)
    node_a = next(e for e in pc._entries if len(e.tokens) == 512)
    assert node_a.children
    assert node_a not in pc._pool_victim_order()


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


# ── radix structure (TD-PREFIX-RADIX-BLOCKS R1): the registry is a prefix
# tree — parent = longest registered proper prefix, children back-links —
# and every list-era policy (touch path, leaf-only eviction, edge-length
# budget) is a tree walk.  Design: spec/plans/RADIX_SLAB_DESIGN.md. ────────


def test_prefix_radix_links_and_out_of_order_reparent():
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    A = list(range(64))
    AB = A + list(range(100, 164))
    AC = A + list(range(300, 364))
    _reg(pc, AB, 2)                          # extension registered FIRST
    _reg(pc, A, 1)                           # ancestor arrives later ...
    by_id = {e.seq_id: e for e in pc._entries}
    # ... and subsumes the earlier root: AB reparents under A.
    assert by_id[2].parent is by_id[1]
    assert by_id[1].parent is None
    assert pc._roots == [by_id[1]]
    _reg(pc, AC, 3)                          # sibling branch under A
    by_id = {e.seq_id: e for e in pc._entries}
    assert by_id[3].parent is by_id[1]
    assert {c.seq_id for c in by_id[1].children} == {2, 3}
    # Edge-length budget: A(64) + two 64-token tails = 192.
    assert pc.total_unique_tokens() == 192


def test_prefix_radix_insert_between_splices_edge():
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    A = list(range(64))
    AB = A + list(range(100, 164))
    ABC = AB + list(range(200, 264))
    _reg(pc, A, 1)
    _reg(pc, ABC, 3)
    _reg(pc, AB, 2)                          # lands BETWEEN 1 and 3
    by_id = {e.seq_id: e for e in pc._entries}
    assert by_id[2].parent is by_id[1]
    assert by_id[3].parent is by_id[2], "insert-between must splice the edge"
    assert [c.seq_id for c in by_id[2].children] == [3]
    assert pc.total_unique_tokens() == 192   # 64 + 64 + 64 edges


def test_prefix_radix_detach_interior_merges_edges():
    """Radix node deletion: removing an interior node reparents its
    children to its parent — their edges lengthen by the removed edge, so
    the unique-token total is invariant (memory-neutral by construction:
    the child's refs pin every shared page)."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    A = list(range(64))
    AB = A + list(range(100, 164))
    ABC = AB + list(range(200, 264))
    _reg(pc, A, 1)
    _reg(pc, AB, 2)
    _reg(pc, ABC, 3)
    assert pc.total_unique_tokens() == 192
    by_id = {e.seq_id: e for e in pc._entries}
    pc._detach(by_id[2])                     # interior
    assert by_id[3].parent is by_id[1]
    assert [c.seq_id for c in by_id[1].children] == [3]
    assert {e.seq_id for e in pc._entries} == {1, 3}
    assert pc.total_unique_tokens() == 192   # invariant under merge


def test_prefix_radix_lookup_tracks_deepest_valid_on_path():
    """The validity filter (superchunk shape-identity) applies per node on
    the ONE descent path: a deeper matching-but-invalid node must not
    shadow a shallower valid one."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    A = list(range(64))
    AB = A + list(range(100, 164))
    _reg(pc, A, 1)
    _reg(pc, AB, 2)
    hit = pc.lookup(AB + [7], valid=lambda n: n == 64)
    assert hit is not None and hit.seq_id == 1
    # Both nodes on the path share the fresh touch stamp.
    by_id = {e.seq_id: e for e in pc._entries}
    assert by_id[1].last_used > 0
    hit2 = pc.lookup(AB + [7])               # unfiltered: deepest wins
    assert hit2 is not None and hit2.seq_id == 2
    assert by_id[1].last_used == by_id[2].last_used  # ancestor co-touched


# ── R4c mid-edge reuse (INV-SEQ-FORK-TRUNC): the reuse point is the
# GRID-CLAMPED longest common token prefix with ANY registered node —
# design §3(b): edge splits at grid boundaries, NEVER finer (the R4b sweep
# measured sub-grid/mid-page delta starts diverging) — driven by a
# truncating fork, gated on the arch capability seq_fork_truncatable. ─────


def test_prefix_midedge_lookup_lcp_mechanics_unit_grid():
    """LCP mechanics at the degenerate grid=1 (every length on-grid —
    the 1-token-chunk stride, where every position IS an absolute chunk
    boundary): a prompt diverging INSIDE a node's edge reuses the full
    common prefix via (node, reuse_len); fork_from drives prefix_len."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    base = list(range(700))
    _reg(pc, base[:512], 1)                  # grid node
    _reg(pc, base[:599], 2)                  # exact-body node (child edge)
    prompt = base[:585] + [9001, 9002, 9003]
    hit = pc.lookup_mid_edge(prompt, grid=1)
    assert hit is not None
    node, reuse = hit
    assert node.seq_id == 2 and reuse == 585
    assert pc.hits == 1
    # Touch propagated to the node and its ancestor (the grid node).
    by_id = {e.seq_id: e for e in pc._entries}
    assert by_id[1].last_used == by_id[2].last_used > 0
    # Truncating fork: prefix_len == reuse_len, live (not frozen).
    pc.fork_from(node, 77, reuse_len=reuse)
    assert pc._bridge.forks[-1] == (2, 77)
    assert pc._bridge.prefix_lens[-1] == 585
    assert pc._bridge.frozen[-1] is False


def test_prefix_midedge_subgrid_lookup_exact_lcp_vs_clamp():
    """The subgrid flag (opt-in, default False) is the ONLY difference
    between clamped and exact-LCP policy: same tree, same prompt —
    OFF floors 585 to the 512 boundary, ON returns the exact 585 LCP
    (truncating the 599 node); both honour the seed-feed cap and never
    exceed the source node's registered (committed) length."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    base = list(range(700))
    _reg(pc, base[:512], 1)                  # grid node
    _reg(pc, base[:599], 2)                  # exact-body node (child edge)
    prompt = base[:585] + [9001, 9002, 9003]

    node, reuse = pc.lookup_mid_edge(prompt, grid=512)   # default OFF
    assert (node.seq_id, reuse) == (1, 512)              # clamped

    node, reuse = pc.lookup_mid_edge(prompt, grid=512, subgrid=True)
    assert (node.seq_id, reuse) == (2, 585)              # exact LCP
    assert reuse <= len(node.tokens)         # committed-length bound
    assert reuse <= len(prompt) - 1          # seed-feed cap


def test_prefix_midedge_subgrid_full_offgrid_node_and_seed_cap():
    """Sub-grid ON reuses a full OFF-GRID node whole (the R4b offset-1599
    class — a prefix_len=0 full fork of a non-grid node, opted into),
    and the seed-feed clamp still truncates an exact-prompt overlap."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    base = list(range(700))
    _reg(pc, base[:599], 2)                  # off-grid node only
    prompt = base[:650]
    node, reuse = pc.lookup_mid_edge(prompt, grid=512, subgrid=True)
    assert (node.seq_id, reuse) == (2, 599)  # whole node, off-grid
    # Prompt equal to the node body: cap = len(prompt)-1 truncates.
    node, reuse = pc.lookup_mid_edge(base[:599], grid=512, subgrid=True)
    assert (node.seq_id, reuse) == (2, 598)
    assert reuse == len(base[:599]) - 1


def test_prefix_midedge_grid_clamp_prefers_exact_length_ancestor():
    """PRODUCTION grid (512): a divergence inside the 512→599 edge clamps
    to the 512 boundary, and the registered 512 node is preferred as the
    source (legacy full fork, prefix_len 0) over truncating the deeper
    node."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    base = list(range(700))
    _reg(pc, base[:512], 1)
    _reg(pc, base[:599], 2)
    prompt = base[:585] + [9001, 9002, 9003]
    node, reuse = pc.lookup_mid_edge(prompt, grid=512)
    assert node.seq_id == 1 and reuse == 512
    pc.fork_from(node, 77, reuse_len=reuse)
    assert pc._bridge.prefix_lens[-1] == 0   # full fork of the 512 node


def test_prefix_midedge_grid_clamp_truncates_without_exact_node():
    """When no node exists at the clamped grid boundary (e.g. the grid
    interior was retired under slot pressure, R2), the fork TRUNCATES the
    covering node at the boundary — reuse deeper than any registered
    full-prefix node."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    base = list(range(1400))
    _reg(pc, base[:1024], 1)
    _reg(pc, base[:1299], 2)
    # Simulate R2 retirement of the 1024 grid interior.
    by_id = {e.seq_id: e for e in pc._entries}
    pc._detach(by_id[1])
    prompt = base[:1150] + [9001, 9002]
    node, reuse = pc.lookup_mid_edge(prompt, grid=512)
    assert node.seq_id == 2 and reuse == 1024
    pc.fork_from(node, 79, reuse_len=reuse)
    assert pc._bridge.prefix_lens[-1] == 1024   # truncating, grid-aligned


def test_prefix_midedge_grid_clamp_floor_zero_is_miss():
    """A common prefix entirely below the first grid boundary clamps to
    zero — a MISS (sub-grid reuse is forbidden, R4b evidence)."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    base = list(range(500))
    _reg(pc, base[:399], 1)                  # exact body only (pre < grid)
    prompt = base[:350] + [9001, 9002]
    assert pc.lookup_mid_edge(prompt, grid=512) is None
    assert pc.misses == 1 and pc.hits == 0


def test_prefix_midedge_exact_body_full_node_passes_unclamped():
    """A legacy-valid full node (exact prompt body, n == pre) is reused
    whole even at a non-grid length — the by-construction-identical
    delta-empty hit shape."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    base = list(range(700))
    _reg(pc, base[:599], 2)
    node, reuse = pc.lookup_mid_edge(base[:600], grid=512)  # pre == 599
    assert node.seq_id == 2 and reuse == 599
    pc.fork_from(node, 80, reuse_len=reuse)
    assert pc._bridge.prefix_lens[-1] == 0   # full fork


def test_prefix_midedge_full_node_hit_uses_legacy_full_fork():
    """reuse_len == the node's registered length is the legacy hit shape:
    prefix_len 0 (byte-identical full fork), and a divergence exactly AT a
    node length prefers the full node over a partial edge match."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    base = list(range(700))
    _reg(pc, base[:512], 1)
    _reg(pc, base[:599], 2)
    prompt = base[:512] + [8001, 8002]       # diverges exactly at 512
    node, reuse = pc.lookup_mid_edge(prompt, grid=512)
    assert node.seq_id == 1 and reuse == 512
    pc.fork_from(node, 78, reuse_len=reuse)
    assert pc._bridge.prefix_lens[-1] == 0   # full fork, legacy wire shape


def test_prefix_midedge_reuse_clamped_to_seed_feed():
    """At least one prompt token must remain for the seed feed: a prompt
    that is entirely a prefix of a node clamps reuse to len(prompt)-1
    (grid=1 mechanics check)."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    base = list(range(700))
    _reg(pc, base[:599], 2)
    node, reuse = pc.lookup_mid_edge(base[:590], grid=1)
    assert node.seq_id == 2 and reuse == 589


def test_prefix_midedge_sibling_max_lcp_wins():
    """Siblings are mutually prefix-incompatible but may share tokens
    beyond their parent — the scan picks the max LCP across the level
    (grid=1 mechanics check)."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    common = list(range(100))
    b = [1000 + i for i in range(60)]
    c = b[:30] + [2000 + i for i in range(30)]
    _reg(pc, common, 1)
    _reg(pc, common + b, 2)                  # diverges from c at 130
    _reg(pc, common + c, 3)
    prompt = common + c[:45] + [7, 8, 9]
    node, reuse = pc.lookup_mid_edge(prompt, grid=1)
    assert node.seq_id == 3 and reuse == 145


def test_prefix_midedge_no_common_prefix_is_miss():
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    _reg(pc, list(range(100, 200)), 1)
    assert pc.lookup_mid_edge([1, 2, 3, 4], grid=64) is None
    assert pc.misses == 1 and pc.hits == 0


def test_prefix_midedge_fork_reuse_len_validation():
    import pytest
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    _reg(pc, list(range(64)), 1)
    (node,) = pc._entries
    with pytest.raises(ValueError):
        pc.fork_from(node, 79, reuse_len=0)
    pc.fork_from(node, 79, reuse_len=None)   # legacy default: full fork
    assert pc._bridge.prefix_lens[-1] == 0


# ── slot-pressure ancestor retirement (TD-PREFIX-SUPERSEDE via radix, R2):
# under ENTRY-COUNT pressure a subsumed single-child ancestor is retirable
# (zero KV loss — the child's refs pin every shared page); byte pressure and
# pool-exhaustion eviction stay leaf-only.  Lazy: retirement happens only
# under pressure, never eagerly at registration. ────────────────────────────


def test_prefix_slot_pressure_retires_shallow_ancestor_not_tail():
    """The motivating multi-turn shape: a conversation t1 ⊂ t2 ⊂ t3 fills
    slots; registering t4 must retire t1 (the shallowest subsumed
    ancestor) — never tn, the longest and most useful entry."""
    pc = _pc(max_entries=3, max_cached_tokens=10_000)
    t = list(range(64))
    chain_lens = {}
    for i, seq in enumerate((1, 2, 3), start=1):
        _reg(pc, t, seq)
        chain_lens[seq] = len(t)
        t = t + list(range(i * 100, i * 100 + 64))
    pc.lookup(t + [7])                       # whole chain co-touched
    total_before = pc.total_unique_tokens()
    _reg(pc, t, 4)                           # t4: slot pressure at 4 > 3
    assert {e.seq_id for e in pc._entries} == {2, 3, 4}
    assert pc.evictions == 1
    assert pc._bridge.freed == [1]           # t1's holder sequence freed
    # Memory-neutral: t2's edge absorbed t1's — unique total grew only by
    # t4's new tail.
    assert pc.total_unique_tokens() == total_before + 64
    by_id = {e.seq_id: e for e in pc._entries}
    assert by_id[2].parent is None           # t2 is the new lineage root


def test_prefix_slot_pressure_retires_one_victim_per_turn():
    """Lazy, one victim at a time: need room for t5, retire t2 — the
    surviving ancestors are never kicked wholesale."""
    pc = _pc(max_entries=3, max_cached_tokens=10_000)
    t = list(range(64))
    for i, seq in enumerate((1, 2, 3), start=1):
        _reg(pc, t, seq)
        t = t + list(range(i * 100, i * 100 + 64))
    _reg(pc, t, 4)                           # retires t1
    t = t + list(range(900, 964))
    _reg(pc, t, 5)                           # retires t2
    assert {e.seq_id for e in pc._entries} == {3, 4, 5}
    assert pc._bridge.freed == [1, 2]
    assert pc.evictions == 2


def test_prefix_slot_pressure_branch_point_is_protected():
    """Collinearity is the guard, not extension count: a node with two
    sibling extensions (two conversations sharing a system prompt) is a
    BRANCH POINT and must never be retired — slot pressure falls back to
    the LRU leaf."""
    pc = _pc(max_entries=3, max_cached_tokens=10_000)
    A = list(range(64))
    _reg(pc, A, 1)                           # shared system prompt
    _reg(pc, A + list(range(100, 164)), 2)   # conversation 1
    _reg(pc, A + list(range(300, 364)), 3)   # conversation 2
    _reg(pc, list(range(500, 564)), 4)       # unrelated: slot pressure
    by_id = {e.seq_id: e for e in pc._entries}
    assert 1 in by_id, "branch point must survive"
    # Victim was the LRU leaf (conversation 1's tail), not the shared root.
    assert {e.seq_id for e in pc._entries} == {1, 3, 4}
    assert pc._bridge.freed == [2]


def test_prefix_slot_pressure_selects_across_lineages():
    """A stale sibling conversation goes before the active chain's
    shallowest member (LRU-major across lineages)."""
    pc = _pc(max_entries=3, max_cached_tokens=10_000)
    _reg(pc, list(range(500, 564)), 9)       # stale unrelated lineage
    A = list(range(64))
    _reg(pc, A, 1)
    _reg(pc, A + list(range(100, 164)), 2)
    pc.lookup(A + list(range(100, 164)) + [7])   # chain co-touched, fresh
    _reg(pc, A + list(range(100, 164)) + list(range(200, 264)), 3)
    assert {e.seq_id for e in pc._entries} == {1, 2, 3}
    assert pc._bridge.freed == [9], "stale lineage retired, active kept"


def test_prefix_byte_pressure_never_retires_interiors():
    """Byte pressure must keep the leaf-only deepest-first rule — only a
    leaf eviction frees KV bytes; retiring an interior would spin (the
    unique total is invariant under edge merge)."""
    pc = _pc(max_entries=8, max_cached_tokens=200)
    A = list(range(64))
    _reg(pc, A, 1)
    _reg(pc, A + list(range(100, 164)), 2)
    _reg(pc, A + list(range(100, 164)) + list(range(200, 264)), 3)
    pc.lookup(A + list(range(100, 164)) + list(range(200, 264)) + [7])
    _reg(pc, list(range(500, 564)), 4)       # byte pressure (256 > 200)
    # The chain TAIL went (frees 64 unique) — interiors 1 and 2 survive.
    assert {e.seq_id for e in pc._entries} == {1, 2, 4}
    assert pc._bridge.freed == [3]


def test_prefix_radix_unregistered_branch_point_counts_per_lineage():
    """Two entries sharing an UNREGISTERED common prefix are separate
    lineages: no synthetic branch node exists, and each was physically
    prefilled on its own pages (sharing arises from fork lineage, which
    the registered-prefix relation is the proxy for) — so the budget
    counts the common tokens once per lineage, as the list era did."""
    pc = _pc(max_entries=8, max_cached_tokens=10_000)
    A = list(range(64))
    _reg(pc, A + list(range(100, 164)), 1)
    _reg(pc, A + list(range(300, 364)), 2)
    by_id = {e.seq_id: e for e in pc._entries}
    assert by_id[1].parent is None and by_id[2].parent is None
    assert len(pc._roots) == 2
    assert pc.total_unique_tokens() == 256


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


# ── spec→plain fallback (INV-SERVE-SPEC-FALLBACK): a DRAFT-side dspark
# failure mid-request (drafting-context invalidation, TD-DSPARK-CTX-CAP
# class) must never fail the request — the orchestrator switches the live
# request onto the plain arm at the round boundary and the stream
# continues, token-lossless.  The FakeDaemon scripts the engine's sticky
# invalidation via dspark_fail_from. ───────────────────────────────────────


def test_spec_fallback_mid_request_lossless_stream():
    orch, daemon = _orch()
    daemon.dspark_fail_from = 2              # rounds 1-2 draft, round 3 dies
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=41, prompt_token_ids=[4321], max_tokens=24,
            on_token=sink.on_token, on_complete=sink.on_complete))
        rid, tokens, reason = sink.done
        assert reason == "length", "fallback must not error the request"
        assert tokens == chain(4321, 24), "fallback lost token identity"
        assert sink.tokens == tokens, "on_token stream != final tokens"
        st = orch.last_stats
        assert st.spec_fallback_round == 3
        assert st.proposed > 0                    # rounds 1-2 speculated
        # The failing send is the LAST dspark ever issued for the request:
        # the remainder decodes plain (sticky, no per-round retry storm).
        assert daemon.dspark_calls == 3
        assert not orch._draft_ctx_valid          # mirror pessimised
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_spec_fallback_from_first_round():
    """Context already invalid at request start (unmirrored — e.g. a
    stale adoption mirror): round 1's draft dies, the whole request
    decodes plain, lossless."""
    orch, daemon = _orch()
    daemon.dspark_fail_from = 0
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=42, prompt_token_ids=[4321], max_tokens=16,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length"
        assert tokens == chain(4321, 16)
        st = orch.last_stats
        assert st.spec_fallback_round == 1 and st.proposed == 0
        assert daemon.dspark_calls == 1
    finally:
        _finish(daemon)


def test_spec_fallback_honors_eos_stop():
    ref = chain(4321, 24)
    eos_tok = ref[9]
    orch, daemon = _orch(eos=(eos_tok,))
    daemon.dspark_fail_from = 0              # fall back at round 1
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=43, prompt_token_ids=[4321], max_tokens=0,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "stop"
        assert tokens == ref[:10], "must stop AT the EOS token"
        assert sink.tokens == tokens
        assert orch.last_stats.spec_fallback_round == 1
    finally:
        _finish(daemon)


def test_spec_fallback_never_surfaces_error_to_caller():
    """The TD-SERVE-ERROR-MASKING error channel stays SILENT on a
    draft-side failure: error == '' and finish_reason is normal."""
    orch, daemon = _orch()
    daemon.dspark_fail_from = 1
    try:
        sink = _ErrSink()
        _serve(orch, InferenceRequest(
            request_id=44, prompt_token_ids=[4321], max_tokens=12,
            on_complete=sink.on_complete))
        assert sink.done[2] == "length"
        assert sink.error == ""
        assert sink.done[1] == chain(4321, 12)
    finally:
        _finish(daemon)


def test_spec_fallback_pessimises_mirror_then_full_prefill_rearms():
    """Cross-request semantics: after a fallback, a prefix-FORKED
    successor must route plain upfront (the engine context is dead and a
    fork never feeds position 0), while a later FULL-prefill request
    re-arms drafting (position-0 capture) and speculates again."""
    prompt = chain(11, 130)                  # holder registers at 128
    orch, daemon = _orch_pc()
    daemon.dspark_fail_from = 0
    try:
        s1, s2, s3 = _Sink(), _Sink(), _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        assert orch.last_stats.spec_fallback_round == 1
        assert daemon.dspark_calls == 1
        assert s1.done[1] == chain(prompt[-1], 8)
        # Forked successor: prefix hit at 128, mirror invalid → plain
        # upfront, ZERO dspark sends (no round-1 error round).
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s2.on_complete))
        assert orch.last_stats.prefix_hit_tokens == 128
        assert orch.last_stats.proposed == 0
        assert orch.last_stats.spec_fallback_round == 0
        assert daemon.dspark_calls == 1              # unchanged
        assert s2.done[1] == s1.done[1]
        # Full-prefill request: position-0 capture re-arms the engine
        # context (scripted: the daemon accepts drafts again) and the
        # mirror re-validates — speculation is BACK.
        daemon.dspark_fail_from = None
        _serve(orch, InferenceRequest(
            request_id=3, prompt_token_ids=chain(500, 10), max_tokens=8,
            on_complete=s3.on_complete))
        assert daemon.dspark_calls > 1
        assert orch.last_stats.proposed > 0
        assert orch._draft_ctx_valid
        assert s3.done[1] == chain(chain(500, 10)[-1], 8)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_ctx_cap_guard_routes_long_prompt_plain():
    """TD-DSPARK-CTX-CAP start-time mirror (landed 8aa4c39b, kept as the
    LEGACY contract — ctx_rotate=False): a prompt whose context (+ one
    drafting round) overflows the draft arena cap routes to the PLAIN arm
    upfront — zero dspark sends, token-identical output."""
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    orch = Orchestrator(
        bridge, metadata=_meta(),
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0, ctx_cap_tokens=64,
                                      ctx_rotate=False))
    try:
        sink = _Sink()
        prompt = chain(11, 130)              # 130 + 5 + 1 > 64
        _serve(orch, InferenceRequest(
            request_id=45, prompt_token_ids=prompt, max_tokens=8,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length"
        assert tokens == chain(prompt[-1], 8)
        assert daemon.dspark_calls == 0          # plain from the start
        assert orch.last_stats.proposed == 0
        assert orch.last_stats.spec_fallback_round == 0
    finally:
        _finish(daemon)


def test_ctx_rotate_keeps_long_prompt_speculative():
    """TD-DSPARK-CTX-POLICY: with rotation (the default) the engine
    windows the draft context, so an over-cap prompt STAYS on the
    speculative arm — dspark engages and the output is token-identical
    to the plain chain (INV-DSPARK-LOSSLESS)."""
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    orch = Orchestrator(
        bridge, metadata=_meta(),
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0, ctx_cap_tokens=64,
                                      ctx_rotate=True))
    try:
        sink = _Sink()
        prompt = chain(11, 130)              # over-cap: 130 + 5 + 1 > 64
        _serve(orch, InferenceRequest(
            request_id=46, prompt_token_ids=prompt, max_tokens=8,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length"
        assert tokens == chain(prompt[-1], 8)    # token-identical
        assert daemon.dspark_calls > 0           # drafting engaged
        assert orch.last_stats.spec_fallback_round == 0
    finally:
        _finish(daemon)


def test_guided_spec_fallback_matches_guided_plain():
    """Guided arm: a mid-request draft death continues the constrained
    decode (matcher state carries over) — output identical to the guided
    PLAIN arm, including grammar repairs past the cut."""
    ref = chain(4321, 40)
    banned = {ref[5], ref[17], ref[29]}
    results = []
    for fail_from in (None, 1):
        orch, daemon = _orch()
        daemon.dspark_fail_from = fail_from
        try:
            sink = _serve_guided(orch, FakeGuided(banned=banned),
                                 max_tokens=40,
                                 force_plain=(fail_from is None))
            results.append(sink.done[1])
            if fail_from is not None:
                assert orch.last_stats.spec_fallback_round >= 1
                assert daemon.dspark_calls == fail_from + 1
            assert not daemon.errors, daemon.errors
        finally:
            _finish(daemon)
    assert results[0] == results[1] == guided_chain(4321, 40, banned)


def test_guided_spec_fallback_completion_stops_with_tool_calls():
    orch, daemon = _orch()
    daemon.dspark_fail_from = 0
    try:
        sink = _serve_guided(orch, FakeGuided(complete_after=7),
                             max_tokens=100)
        _, tokens, reason = sink.done
        assert reason == "tool_calls"
        assert tokens == chain(4321, 7)
        assert orch.last_stats.spec_fallback_round == 1
    finally:
        _finish(daemon)


# ── TD-V4-FIRSTREQ-COLD-SHARE-OVER-CAPACITY option (d): serving-level
# retry of a request that completed with degraded MoE layers.  Policy:
# retry ONLY while nothing has reached the client — a prefill-degraded
# attempt is abandoned BEFORE its first token (streaming-safe), a
# non-streaming request may retry after decode too, and a streaming
# request that degraded mid-decode is delivered flagged, never re-run. ──


def _orch_retry(retry_max: int, *, pc: bool = False, daemon_gamma: int = 5):
    from orchestrator.orchestrator import PrefixCacheConfig
    bridge, daemon, _ = _make(gamma=daemon_gamma, use_far=True)
    orch = Orchestrator(
        bridge, metadata=_meta(),
        speculation=SpeculationConfig(enabled=True, gamma=daemon_gamma,
                                      conf_thresh=0.0),
        prefix_cache=(PrefixCacheConfig() if pc
                      else PrefixCacheConfig(enabled=False)),
        degraded_retry_max=retry_max)
    return orch, daemon


def test_degraded_prefill_retries_once_clean_single_stream():
    """A degraded completion triggers exactly ONE retry; the retry (which
    lands on the residency the failed attempt established) is clean; the
    STREAM is emitted exactly once — the abandoned attempt emitted no
    token; the abandoned attempt's sequence is freed."""
    prompt = chain(11, 130)                  # 2x64-chunk prefill + seed
    orch, daemon = _orch_retry(1)
    try:
        daemon.degrade_moe_remaining = 2     # attempt 1 degrades, then clean
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_token=sink.on_token, on_complete=sink.on_complete))
        assert sink.done[2] == "length"
        assert sink.done[1] == chain(prompt[-1], 8)   # healthy-run tokens
        assert sink.tokens == sink.done[1], "stream emitted twice or lost"
        st = orch.last_stats
        assert st.moe_degraded_layers == 0, "delivered attempt not clean"
        assert st.degraded_retries == 1
        assert orch.degraded_retries_total == 1
        assert orch.degraded_delivered_total == 0
        assert orch.degraded_unretryable_total == 0
        assert daemon.seq_frees == 2, "abandoned attempt's seq not freed"
        assert not daemon.known_seqs, "leaked sequence"
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_clean_request_never_retries():
    orch, daemon = _orch_retry(1)
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_complete=sink.on_complete))
        assert sink.done[2] == "length"
        st = orch.last_stats
        assert st.degraded_retries == 0 and st.moe_degraded_layers == 0
        assert orch.degraded_retries_total == 0
        assert daemon.seq_frees == 1
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_degraded_retry_cap_honoured_and_delivery_flagged():
    """A box that degrades CONSTANTLY must not loop: exactly `max` retries,
    then the degraded result is DELIVERED, flagged and counted — the
    failure mode the counters exist for."""
    prompt = chain(11, 130)
    orch, daemon = _orch_retry(2)
    try:
        daemon.degrade_moe_remaining = 10**9   # every attempt degrades
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=sink.on_complete))     # non-streaming
        assert sink.done[2] == "length"
        assert sink.done[1] == chain(prompt[-1], 8)   # flag never mutates
        st = orch.last_stats
        assert st.degraded_retries == 2, "cap not honoured"
        assert st.moe_degraded_layers > 0, "delivery lost the flag"
        assert orch.degraded_retries_total == 2
        assert orch.degraded_delivered_total == 1
        assert daemon.seq_frees == 3           # 2 abandoned + delivered
        assert not daemon.known_seqs
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_degraded_retry_off_switch():
    """retry_max == 0 (the boot-level off-switch maps here): the first
    attempt is delivered — degraded, flagged, zero retries."""
    orch, daemon = _orch_retry(0)
    try:
        daemon.degrade_moe_remaining = 2
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=chain(11, 130), max_tokens=8,
            on_complete=sink.on_complete))
        assert sink.done[2] == "length"
        st = orch.last_stats
        assert st.degraded_retries == 0
        assert st.moe_degraded_layers == 2
        assert orch.degraded_retries_total == 0
        assert orch.degraded_delivered_total == 1
        assert daemon.seq_frees == 1, "off-switch still re-ran the request"
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_streamed_decode_degradation_is_not_retried():
    """STREAMING decision: a request that degraded only during DECODE has
    already streamed tokens to the client — a silent re-run would emit a
    second stream, so it is delivered flagged and counted unretryable."""
    orch, daemon = _orch_retry(1)
    try:
        daemon.degrade_moe_remaining = 3     # no prefill (1-token prompt):
        sink = _Sink()                       # degradation lands in decode
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=[4321], max_tokens=12,
            on_token=sink.on_token, on_complete=sink.on_complete))
        assert sink.done[2] == "length"
        assert sink.tokens == sink.done[1] == chain(4321, 12)
        st = orch.last_stats
        assert st.degraded_retries == 0, "streamed request was re-run"
        assert st.moe_degraded_layers == 3
        assert orch.degraded_retries_total == 0
        assert orch.degraded_unretryable_total == 1
        assert orch.degraded_delivered_total == 1
        assert daemon.seq_frees == 1
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_decode_degradation_without_streaming_retries():
    """The same decode-time degradation IS retried when nothing was
    streamed (no on_token): delivery happens only in on_complete, after
    the retry decision."""
    orch, daemon = _orch_retry(1)
    try:
        daemon.degrade_moe_remaining = 3
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=[4321], max_tokens=12,
            on_complete=sink.on_complete))
        assert sink.done[2] == "length"
        assert sink.done[1] == chain(4321, 12)
        st = orch.last_stats
        assert st.degraded_retries == 1
        assert st.moe_degraded_layers == 0, "retry not clean"
        assert orch.degraded_retries_total == 1
        assert orch.degraded_unretryable_total == 0
        assert orch.degraded_delivered_total == 0
        assert daemon.seq_frees == 2
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_degraded_retry_holder_registered_only_by_clean_attempt():
    """Interlock with the degraded-registration guard: the abandoned
    degraded attempt registers NO prefix holder, the clean retry does,
    and a follow-up request hits it token-identically."""
    prompt = chain(11, 130)
    orch, daemon = _orch_retry(1, pc=True)
    try:
        daemon.degrade_moe_remaining = 2
        s1 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s1.on_complete))
        assert orch.last_stats.degraded_retries == 1
        assert orch.last_stats.moe_degraded_layers == 0
        assert daemon.forks == 1, "clean retry did not register a holder"
        s2 = _Sink()
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=prompt, max_tokens=8,
            on_complete=s2.on_complete))
        assert orch.last_stats.prefix_hit_tokens == 128
        assert s2.done[1] == s1.done[1] == chain(prompt[-1], 8)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_guided_decode_degradation_not_retried_single_use_grammar():
    """A GUIDED request that degraded during decode is never re-run even
    un-streamed: its grammar state (InferenceRequest.guided) is single-use
    and was consumed by the attempt's decode."""
    orch, daemon = _orch_retry(1)
    try:
        daemon.degrade_moe_remaining = 3
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=[4321], max_tokens=12,
            guided=FakeGuided(), on_complete=sink.on_complete))
        assert sink.done[2] == "length"
        assert sink.done[1] == chain(4321, 12)
        st = orch.last_stats
        assert st.degraded_retries == 0, "single-use grammar was re-driven"
        assert st.moe_degraded_layers == 3
        assert orch.degraded_unretryable_total == 1
        assert daemon.seq_frees == 1
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_guided_prefill_degradation_still_retries():
    """PREFILL-degraded guided attempts abort before decode ever touches
    the grammar, so the retry stays available to guided requests."""
    prompt = chain(11, 130)
    orch, daemon = _orch_retry(1)
    try:
        daemon.degrade_moe_remaining = 2
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=8,
            guided=FakeGuided(), on_complete=sink.on_complete))
        assert sink.done[2] == "length"
        assert sink.done[1] == chain(prompt[-1], 8)
        st = orch.last_stats
        assert st.degraded_retries == 1
        assert st.moe_degraded_layers == 0
        assert orch.degraded_unretryable_total == 0
        assert daemon.seq_frees == 2
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


# ── TD-ORCH-EP-GPU-INDICES-DEAD: the production boot REFUSES a config that
# sets _internal-orchestrator.ep_gpu_indices (it would otherwise be silently
# discarded — the orchestrator derives expert hosts from hardware.gpus roles
# and the daemon's REEF placement picks per-expert owners). The field stays
# live only for the legacy OrchestratorLoop drivers (engine_glue.
# ep_gpu_indices_from_config, tools/elb_train). ────────────────────────────


class _EngineMustNotStart:
    """Sentinel engine module: proves boot fails BEFORE the engine starts."""

    class Started(RuntimeError):
        pass

    def start_engine(self, path):
        raise self.Started("engine started")

    def start_engine_test(self, path):
        raise self.Started("engine started")


def _write_cfg(tmp_path, internal_orch):
    import json

    cfg = {"model": {}, "hardware": {"tp_array": [0]}}
    if internal_orch is not None:
        cfg["_internal-orchestrator"] = internal_orch
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_boot_refuses_nonempty_ep_gpu_indices(tmp_path):
    path = _write_cfg(tmp_path, {"ep_gpu_indices": [0, 1]})
    with pytest.raises(ValueError) as ei:
        Orchestrator.boot(path, engine_module=_EngineMustNotStart(),
                          test_engine=True)
    msg = str(ei.value)
    assert "ep_gpu_indices" in msg
    assert "hardware.gpus" in msg  # points at the supported owner-set knob


def test_boot_passes_empty_or_absent_ep_gpu_indices(tmp_path):
    # Empty list and absent section both fall through to the engine start
    # (the sentinel fires — proving the refusal is scoped to non-empty).
    for internal in ({"ep_gpu_indices": []}, None):
        path = _write_cfg(tmp_path, internal)
        with pytest.raises(_EngineMustNotStart.Started):
            Orchestrator.boot(path, engine_module=_EngineMustNotStart(),
                              test_engine=True)
