"""SpecRoundGovernor unit suite (TD-DSPARK-CTX-POLICY, acceptance axis).

The governor suspends drafting when the measured speculative round gain
is negative (S10: 4 of 6 sub-8k speculative turns ran at or below the
plain band on agentic content) and probes periodically so a content
shift re-enables it.  Load-bearing property tested here: on a
SHAPE-INVARIANT engine (this scripted daemon), suspended (plain) rounds
and speculative rounds commit the identical trajectory.  On the REAL
engine that holds only per round-shape sequence: verify-chunk argmax is
the batched forward's own output (INV-DSPARK-LOSSLESS B>1 clause), so
near-tie tokens flip between shapes and the governor's wall-clock-driven
shape choice makes served greedy decode non-reproducible run-to-run
(TD-GLM52-CHAMPION-GREEDY-NONDET-RUN-TO-RUN; identity gates need
LS_SPEC_GOVERNOR=0 — dossier §4b precondition 4).
"""

from __future__ import annotations

import pytest

from test_bridge_ring import _finish, _make, chain

from orchestrator.orchestrator import (
    InferenceRequest,
    Orchestrator,
    SpecRoundGovernor,
    SpeculationConfig,
)
from test_orchestrator_bridge import _Sink, _meta, _serve


# ── pure state-machine tests ────────────────────────────────────────────


def test_governor_stays_active_on_positive_gain():
    gov = SpecRoundGovernor(warmup_rounds=2)
    gov.note_plain(200.0)
    for _ in range(10):
        assert gov.draft_this_round()
        gov.note_round(tokens=4, ms=500.0)   # 4*200 - 500 = +300
    assert not gov.suspended
    assert gov.probes == 0


def test_governor_suspends_on_negative_gain_after_warmup():
    gov = SpecRoundGovernor(warmup_rounds=4)
    gov.note_plain(200.0)
    for i in range(3):
        gov.note_round(tokens=1, ms=350.0)   # 1*200 - 350 = -150
        assert not gov.suspended, f"suspended during warmup round {i}"
    gov.note_round(tokens=1, ms=350.0)
    assert gov.suspended


def test_governor_probe_cadence_and_resume():
    gov = SpecRoundGovernor(probe_period=4, warmup_rounds=1, alpha=0.5)
    gov.note_plain(200.0)
    gov.note_round(tokens=1, ms=400.0)       # gain -200 -> suspend
    assert gov.suspended
    # Suspended rounds decode plain; every 4th is a speculative probe.
    draft_flags = [gov.draft_this_round() for _ in range(8)]
    assert draft_flags == [False, False, False, True,
                           False, False, False, True]
    assert gov.probes == 2
    # Positive probes pull the EMA back over zero -> resume.
    gov.note_round(tokens=5, ms=300.0)       # gain +700
    assert not gov.suspended
    assert gov.draft_this_round()


def test_governor_no_baseline_never_suspends():
    gov = SpecRoundGovernor(warmup_rounds=1)
    for _ in range(10):
        assert gov.draft_this_round()
        gov.note_round(tokens=1, ms=1e9)     # ignored: no plain baseline
    assert not gov.suspended


def test_governor_disabled_always_drafts():
    gov = SpecRoundGovernor(enabled=False, warmup_rounds=1)
    gov.note_plain(200.0)
    for _ in range(10):
        assert gov.draft_this_round()
        gov.note_round(tokens=1, ms=1e9)
    assert not gov.suspended


def test_governor_plain_baseline_tracks_ema():
    gov = SpecRoundGovernor(alpha=0.5)
    gov.note_plain(100.0)
    assert gov.plain_ms == pytest.approx(100.0)
    gov.note_plain(300.0)
    assert gov.plain_ms == pytest.approx(200.0)
    gov.note_plain(0.0)                       # ignored (non-positive)
    assert gov.plain_ms == pytest.approx(200.0)


# ── loop integration: suspension is token-lossless ──────────────────────


def test_suspended_rounds_commit_identical_tokens(monkeypatch):
    """A pre-suspended governor (pinned hugely negative gain) decodes
    almost every round PLAIN with periodic probes — and the committed
    trajectory is EXACTLY the plain chain."""

    def _suspended_gov():
        gov = SpecRoundGovernor(probe_period=3, warmup_rounds=1)
        gov.suspended = True
        gov._have_gain = True
        gov.gain_ms = -1e12     # probes cannot flip it back
        return gov

    monkeypatch.setattr(SpecRoundGovernor, "from_env",
                        classmethod(lambda cls: _suspended_gov()))
    bridge, daemon, _ = _make(gamma=5, use_far=True)
    orch = Orchestrator(
        bridge, metadata=_meta(),
        speculation=SpeculationConfig(enabled=True, gamma=5,
                                      conf_thresh=0.0))
    try:
        sink = _Sink()
        prompt = chain(21, 6)
        _serve(orch, InferenceRequest(
            request_id=61, prompt_token_ids=prompt, max_tokens=12,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length"
        assert tokens == chain(prompt[-1], 12)          # token-lossless
        st = orch.last_stats
        assert st.gov_plain_rounds > 0                  # suspension ran
        assert st.gov_probe_rounds >= 1                 # probes issued
        # Every dspark send was a probe round.
        assert daemon.dspark_calls == st.gov_probe_rounds
    finally:
        _finish(daemon)


def test_governor_defaults_from_env(monkeypatch):
    monkeypatch.delenv("LS_SPEC_GOVERNOR", raising=False)
    monkeypatch.delenv("LS_SPEC_GOV_PROBE", raising=False)
    gov = SpecRoundGovernor.from_env()
    assert gov.enabled and gov.probe_period == 16
    monkeypatch.setenv("LS_SPEC_GOVERNOR", "0")
    monkeypatch.setenv("LS_SPEC_GOV_PROBE", "5")
    gov = SpecRoundGovernor.from_env()
    assert not gov.enabled and gov.probe_period == 5
