"""Sampled speculation unit suite (TD-ORCH-SAMPLED-SPEC) — the
distribution-lossless host-sampled arm on the scripted FakeDaemon, plus
pure-math gates for the target-distribution transform and the
equality-coupled accept rule.

Load-bearing invariants:

  * INV-SAMPLED-SPEC: with a fixed seed, the speculative sampled arm
    commits EXACTLY the trajectory of sequential host-sampled AR decode
    (the RNG advances once per committed token, in order) — speculation
    changes wall, never tokens.
  * T→0 limit: the sampled arm degenerates to the greedy champion chain.
  * The accept rule (equality coupling for DSpark's deterministic
    point-mass proposal) leaves the per-position marginal EXACTLY the
    target distribution — checked in closed form and statistically.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_bridge_ring import (  # the scripted-daemon harness
    LOGITS_ROWS,
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

# Concentrated 3-token support: softmax leakage to the 99 997 zero-logit
# tokens is ~1e5 * e^-30 ≈ 9e-9 — negligible against any tolerance here.
VALS = (30.0, 29.0, 28.0)


def _meta(eos: tuple[int, ...] = ()) -> EngineMetadata:
    return EngineMetadata(
        num_gpus=4, num_moe_layers=2, num_experts=64, num_layers=4,
        expert_bytes=0, kv_bytes_per_page=0, eos_token_ids=eos,
        vocab_size=VOCAB, moe_batch_capacity=512)


def _orch(daemon_gamma: int = 5, *, eos: tuple[int, ...] = (),
          logit_vals: tuple = VALS, logits_rows: int = LOGITS_ROWS,
          conf_enabled: bool = False, conf_thresh: float = 0.0):
    bridge, daemon, _ = _make(gamma=daemon_gamma, use_far=True,
                              logit_vals=logit_vals,
                              logits_rows=logits_rows,
                              conf_enabled=conf_enabled)
    orch = Orchestrator(
        bridge, metadata=_meta(eos),
        speculation=SpeculationConfig(enabled=True, gamma=daemon_gamma,
                                      conf_thresh=conf_thresh))
    return orch, daemon


class _Sink:
    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.lps: list = []
        self.done: tuple | None = None
        self.done_lp = None

    def on_token(self, rid: int, tok: int, logp) -> None:
        self.tokens.append(tok)
        self.lps.append(logp)

    def on_complete(self, rid: int, tokens: list[int], reason: str,
                    logp) -> None:
        assert self.done is None
        self.done = (rid, list(tokens), reason)
        self.done_lp = logp


def _serve(orch: Orchestrator, req: InferenceRequest) -> None:
    orch.submit_request(req)
    assert orch._serve_next() is True


def _scripted_logits(tok: int, vals: tuple = VALS) -> np.ndarray:
    """The FakeDaemon's logits row for previous token ``tok``."""
    row = np.zeros(VOCAB, dtype=np.float32)
    peak = f(tok)
    for i, v in enumerate(vals):
        row[(peak + i) % VOCAB] = v
    return row


def _host_reference(seed_token: int, n: int, smp: SamplingParams,
                    vals: tuple = VALS) -> list[int]:
    """Sequential host-sampled AR decode over the scripted model — the
    trajectory the speculative sampled arm must reproduce EXACTLY."""
    rng = np.random.default_rng(smp.seed)
    out, tok = [], seed_token
    for _ in range(n):
        p = Orchestrator._target_probs(_scripted_logits(tok, vals),
                                       smp.temperature, smp.top_p,
                                       smp.top_k)
        tok = int(rng.choice(p.size, p=p))
        out.append(tok)
    return out


# ── _target_probs closed-form gates ─────────────────────────────────────────


def test_target_probs_temperature_softmax():
    logits = np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float32)
    p = Orchestrator._target_probs(logits, 1.0, 1.0, 0)
    e = np.exp(np.array([2.0, 1.0, 0.0, -1.0]))
    assert np.allclose(p, e / e.sum(), atol=1e-12)
    # T=2 halves the gaps.
    p2 = Orchestrator._target_probs(logits, 2.0, 1.0, 0)
    e2 = np.exp(np.array([1.0, 0.5, 0.0, -0.5]))
    assert np.allclose(p2, e2 / e2.sum(), atol=1e-12)


def test_target_probs_greedy_limit_one_hot():
    logits = np.array([0.5, 2.0, 2.0, -1.0], dtype=np.float32)
    p = Orchestrator._target_probs(logits, 0.0, 0.7, 3)
    # One-hot argmax, LOWEST index on ties (engine packed-score parity).
    assert p.tolist() == [0.0, 1.0, 0.0, 0.0]
    # The continuous limit agrees: T=1e-9 concentrates all mass.
    p_lim = Orchestrator._target_probs(logits, 1e-9, 1.0, 0)
    assert p_lim[1] == pytest.approx(0.5) and p_lim[2] == pytest.approx(0.5)


def test_target_probs_top_k():
    logits = np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float32)
    p = Orchestrator._target_probs(logits, 1.0, 1.0, 2)
    e2, e1 = np.exp(2.0), np.exp(1.0)
    assert p[0] == pytest.approx(e2 / (e2 + e1), abs=1e-12)
    assert p[1] == pytest.approx(e1 / (e2 + e1), abs=1e-12)
    assert p[2] == 0.0 and p[3] == 0.0


def test_target_probs_top_p_includes_crossing_token():
    # p = [0.6, 0.3, 0.1] exactly.
    logits = np.log(np.array([6.0, 3.0, 1.0])).astype(np.float32)
    # top_p=0.7: 0.6 < 0.7 → the crossing token 0.3 is INCLUDED.
    p = Orchestrator._target_probs(logits, 1.0, 0.7, 0)
    assert np.allclose(p, [2 / 3, 1 / 3, 0.0], atol=1e-9)
    # top_p=0.5: the top token alone crosses → one-hot.
    p = Orchestrator._target_probs(logits, 1.0, 0.5, 0)
    assert np.allclose(p, [1.0, 0.0, 0.0], atol=1e-9)


def test_target_probs_guided_mask_composes():
    # INV-GUIDED-1 composition seam: the mask hits the logits BEFORE the
    # transform — the accept rule sees the masked target distribution.
    logits = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    mask = np.array([False, True, True])
    p = Orchestrator._target_probs(logits, 1.0, 1.0, 0, mask=mask)
    e2, e1 = np.exp(2.0), np.exp(1.0)
    assert p[0] == 0.0
    assert p[1] == pytest.approx(e2 / (e2 + e1), abs=1e-12)
    # Masked greedy falls to the allowed runner-up.
    p0 = Orchestrator._target_probs(logits, 0.0, 1.0, 0, mask=mask)
    assert p0.tolist() == [0.0, 1.0, 0.0]


def test_step_seed_distinct_stable_u64():
    seeds = [Orchestrator._step_seed(42, pos) for pos in range(1000)]
    assert len(set(seeds)) == 1000, "per-position keys must not collide"
    assert all(0 <= s < 2 ** 64 for s in seeds)
    assert seeds == [Orchestrator._step_seed(42, pos)
                     for pos in range(1000)], "must be reproducible"
    assert Orchestrator._step_seed(43, 0) != seeds[0]


# ── accept-rule law (equality coupling ≡ vLLM NO_DRAFT_PROBS rule) ─────────


def test_equality_coupling_marginal_and_residual():
    """Draw s ~ p, accept the deterministic draft d iff s == d: the
    committed marginal is exactly p, the acceptance rate is p[d], and
    the rejection-conditional matches the residual p/(1-p[d]) over
    x != d — the vLLM rejection rule for a point-mass proposal."""
    logits = np.array([1.5, 0.7, 0.0, -0.4, -1.0], dtype=np.float32)
    p = Orchestrator._target_probs(logits, 0.9, 1.0, 0)
    d = 1                                    # a non-argmax draft slot
    rng = np.random.default_rng(7)
    n = 40_000
    s = rng.choice(p.size, p=p, size=n)
    accepted = s == d
    acc_rate = accepted.mean()
    assert acc_rate == pytest.approx(p[d], abs=0.01)
    counts = np.bincount(s, minlength=p.size) / n
    assert np.abs(counts - p).max() < 0.01, "committed marginal != target"
    resid = np.where(np.arange(p.size) == d, 0.0, p)
    resid /= resid.sum()
    rej = s[~accepted]
    rc = np.bincount(rej, minlength=p.size) / rej.size
    assert np.abs(rc - resid).max() < 0.01, "rejection law != residual"


# ── the speculative sampled arm on the scripted daemon ─────────────────────


def test_sampled_spec_equals_sequential_reference():
    """INV-SAMPLED-SPEC: the speculative arm's committed trajectory is
    IDENTICAL to sequential host-sampled AR decode with the same seed —
    the sampled analog of the lossless invariant."""
    smp = SamplingParams(temperature=0.8, top_p=1.0, top_k=0, seed=1234)
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=[4321], max_tokens=32,
            sampling=smp, on_token=sink.on_token,
            on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length" and len(tokens) == 32
        assert tokens == _host_reference(4321, 32, smp)
        st = orch.last_stats
        assert daemon.dspark_calls > 0, "speculation must be ACTIVE"
        assert st.proposed > 0 and st.accepted > 0, (
            "concentrated target + chain drafts must accept slots")
        assert daemon.samples == 0, (
            "the sampled arm owns sampling host-side (no CMD_SAMPLE_TOKENS)")
    finally:
        _finish(daemon)


def test_sampled_spec_fixed_seed_determinism():
    smp = SamplingParams(temperature=0.9, top_p=0.95, seed=99)
    outs = []
    for _ in range(2):
        orch, daemon = _orch()
        try:
            sink = _Sink()
            _serve(orch, InferenceRequest(
                request_id=2, prompt_token_ids=[555], max_tokens=24,
                sampling=smp, on_token=sink.on_token,
                on_complete=sink.on_complete))
            outs.append(sink.done[1])
        finally:
            _finish(daemon)
    assert outs[0] == outs[1], "fixed seed must reproduce the trajectory"
    # A different seed forks the stream (deterministic given both seeds).
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=3, prompt_token_ids=[555], max_tokens=24,
            sampling=SamplingParams(temperature=0.9, top_p=0.95, seed=100),
            on_token=sink.on_token, on_complete=sink.on_complete))
        assert sink.done[1] != outs[0]
    finally:
        _finish(daemon)


def test_sampled_spec_t_to_zero_equals_greedy_arm():
    """T→0 identity: the sampled arm at a vanishing temperature commits
    the exact greedy champion chain."""
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=4, prompt_token_ids=[4321], max_tokens=20,
            sampling=SamplingParams(temperature=1e-9, seed=5),
            on_token=sink.on_token, on_complete=sink.on_complete))
        assert sink.done[1] == chain(4321, 20)
        assert daemon.dspark_calls > 0
    finally:
        _finish(daemon)


def test_sampled_spec_top_p_reference_parity():
    """Nucleus + top-k engaged end-to-end (the transform runs on every
    verify row) still matches the sequential reference exactly."""
    smp = SamplingParams(temperature=1.1, top_p=0.8, top_k=2, seed=77)
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=5, prompt_token_ids=[999], max_tokens=24,
            sampling=smp, on_token=sink.on_token,
            on_complete=sink.on_complete))
        assert sink.done[1] == _host_reference(999, 24, smp)
    finally:
        _finish(daemon)


def test_sampled_spec_conf_truncation_lossless():
    """DSP-9 truncation only shortens proposals — the committed
    trajectory must be unchanged (the accept rule is lossless under any
    proposal length)."""
    smp = SamplingParams(temperature=0.8, seed=1234)
    orch, daemon = _orch(conf_enabled=True, conf_thresh=0.5)
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=6, prompt_token_ids=[4321], max_tokens=32,
            sampling=smp, on_token=sink.on_token,
            on_complete=sink.on_complete))
        assert sink.done[1] == _host_reference(4321, 32, smp)
    finally:
        _finish(daemon)


def test_sampled_spec_eos_stops_and_drops_overshoot():
    smp = SamplingParams(temperature=0.8, seed=1234)
    ref = _host_reference(4321, 32, smp)
    eos_tok = ref[7]
    orch, daemon = _orch(eos=(eos_tok,))
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=7, prompt_token_ids=[4321], max_tokens=0,
            sampling=smp, on_token=sink.on_token,
            on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "stop"
        idx = ref.index(eos_tok)
        assert tokens == ref[:idx + 1], "must stop AT the EOS token"
    finally:
        _finish(daemon)


def test_sampled_logprobs_ride_the_speculative_arm():
    """T>0 + logprobs: served WITH speculation from the per-row readback;
    each committed token carries logprobs of the RAW distribution."""
    smp = SamplingParams(temperature=0.8, seed=42)
    orch, daemon = _orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=8, prompt_token_ids=[4321], max_tokens=16,
            sampling=smp, logprobs=2, on_token=sink.on_token,
            on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert tokens == _host_reference(4321, 16, smp)
        assert daemon.dspark_calls > 0
        assert len(sink.lps) == 16 and all(lp is not None
                                           for lp in sink.lps)
        for tok, lp in zip(tokens, sink.lps):
            assert lp.token.token_id == tok
            assert lp.token.logprob <= 0.0
            assert len(lp.top_logprobs) == 2
        assert sink.done_lp == sink.lps
    finally:
        _finish(daemon)


def test_greedy_logprobs_keep_speculation():
    """TD-ORCH-LOGPROBS-SPEC resolved: greedy + logprobs no longer falls
    to the plain arm when the engine has the multi-row readback — the
    trajectory is the exact greedy chain WITH per-token logprobs."""
    import math
    orch, daemon = _orch(logit_vals=(1.0, 0.5))
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=9, prompt_token_ids=[4321], max_tokens=12,
            logprobs=2, on_token=sink.on_token,
            on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert tokens == chain(4321, 12), "greedy chain must be exact"
        assert daemon.dspark_calls > 0, "speculation must stay ON"
        lse = math.log((VOCAB - 2) * 1.0 + math.exp(1.0) + math.exp(0.5))
        for tok, lp in zip(tokens, sink.lps):
            assert lp is not None and lp.token.token_id == tok
            assert lp.token.logprob == pytest.approx(1.0 - lse)
            assert lp.top_logprobs[0].token_id == tok
            assert lp.top_logprobs[1].token_id == (tok + 1) % VOCAB
    finally:
        _finish(daemon)


def test_single_row_engine_falls_back_to_plain():
    """logits_host_rows == 1 (engine builds predating the multi-row
    readback): sampled requests keep the plain AR path — and its
    engine-side sampler gets an INDEPENDENT per-step Philox key (the
    correlated-draw fix)."""
    orch, daemon = _orch(logits_rows=1, logit_vals=(1.0, 0.5))
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=10, prompt_token_ids=[555], max_tokens=6,
            sampling=SamplingParams(temperature=0.7, top_p=0.9, seed=123),
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert tokens == chain(555, 6)       # FakeDaemon samples argmax
        assert daemon.dspark_calls == 0
        assert daemon.samples == 6
        assert len(daemon.sample_seeds) == 6
        assert len(set(daemon.sample_seeds)) == 6, (
            "per-step seeds must differ (Philox row-0 correlation fix)")
        # Reproducible: the same request derives the same per-step keys
        # (positions start at len(prompt) - 1 = 0).
        expect = [Orchestrator._step_seed(123, i) for i in range(6)]
        assert daemon.sample_seeds == expect
    finally:
        _finish(daemon)


def test_sampled_spec_fallback_trajectory_identical():
    """INV-SERVE-SPEC-FALLBACK on the sampled arm: a mid-request draft
    death continues HOST-SAMPLED plain AR on the SAME RNG stream — the
    full trajectory (across the cut) is EXACTLY the sequential
    host-sampled reference, i.e. exactly what the speculative arm would
    have committed (INV-SAMPLED-SPEC continuity)."""
    smp = SamplingParams(temperature=0.8, top_p=1.0, top_k=0, seed=1234)
    orch, daemon = _orch()
    daemon.dspark_fail_from = 2              # rounds 1-2 draft, round 3 dies
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=61, prompt_token_ids=[4321], max_tokens=32,
            sampling=smp, on_token=sink.on_token,
            on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length" and len(tokens) == 32
        assert tokens == _host_reference(4321, 32, smp), (
            "sampled fallback broke trajectory identity across the cut")
        assert sink.tokens == tokens
        st = orch.last_stats
        assert st.spec_fallback_round == 3
        assert daemon.dspark_calls == 3      # sticky: no re-issue after
        assert daemon.samples == 0           # host-side sampling throughout
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_sampled_spec_fallback_logprobs_continue():
    """Logprobs requests ride the sampled-spec arm; after the fallback
    every remaining step still serves a StepLogprobs from the readback
    row (no gap in the per-token list)."""
    smp = SamplingParams(temperature=0.7, seed=5)
    orch, daemon = _orch()
    daemon.dspark_fail_from = 1
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=62, prompt_token_ids=[4321], max_tokens=12,
            sampling=smp, logprobs=0, on_token=sink.on_token,
            on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length" and len(tokens) == 12
        assert tokens == _host_reference(4321, 12, smp)
        assert orch.last_stats.spec_fallback_round >= 1
        assert len(sink.done_lp) == 12
        assert all(lp is not None for lp in sink.done_lp), (
            "logprobs must not go dark after the fallback")
    finally:
        _finish(daemon)
