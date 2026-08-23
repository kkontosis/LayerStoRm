"""Unit tests for DSpark STS calibration (DSP-7).

The fit is validated on a SYNTHETIC miscalibrated trace with known
per-position miscalibration temperatures: the head reports
c_k = sigma(T_true[k] * logit(p_k)) for true conditional survival p_k, so
the exact inverse is the STS temperature T_true[k]
(sigma(logit(c)/T_true) == p).  The fit must recover T_true within grid
resolution, drive the cumulative-survival ECE well below the raw value,
and be provably order-preserving (rankings within a position unchanged).
"""

from __future__ import annotations

import math
import random

from orchestrator.dspark_calibration import (
    ConfidenceRound,
    apply_temperature,
    cumulative_survival_ece,
    expected_calibration_error,
    fit_sts_temperatures,
    logit,
)
from orchestrator.dspark_draft import (
    DsparkDraft,
    DsparkDraftConfig,
    DsparkDraftResult,
)

GAMMA = 4
T_TRUE = [2.0, 1.0, 0.5, 3.0]  # over-, well-, under-, over-confident


def _synthetic_trace(num_rounds: int, seed: int = 7) -> list[ConfidenceRound]:
    """Rounds with per-position TRUE conditional survival p_k drawn per
    round (variation is required for binned ECE to be meaningful) and
    sequentially-sampled prefix labels; reported c_k miscalibrated by
    T_TRUE on the logit."""
    rng = random.Random(seed)
    rounds: list[ConfidenceRound] = []
    for _ in range(num_rounds):
        p = [rng.uniform(0.2, 0.95) for _ in range(GAMMA)]
        accepted = 0
        for k in range(GAMMA):
            if rng.random() < p[k]:
                accepted += 1
            else:
                break
        c = tuple(1.0 / (1.0 + math.exp(-T_TRUE[k] * logit(p[k])))
                  for k in range(GAMMA))
        rounds.append(ConfidenceRound(confidences=c, accepted=accepted))
    return rounds


class TestStsFit:
    def test_recovers_known_temperatures_and_reduces_ece(self):
        trace = _synthetic_trace(6000)
        raw_ece = cumulative_survival_ece(trace)
        temps = fit_sts_temperatures(trace, GAMMA)
        assert len(temps) == GAMMA
        cal_ece = cumulative_survival_ece(trace, temps)

        # Grid step is 2^(1/8) ~ 9%; sampling noise on 6000 rounds adds a
        # little — 2 grid steps (~19%) is a tight recovery bound.
        for k, (t, t_true) in enumerate(zip(temps, T_TRUE)):
            assert abs(math.log(t / t_true)) <= 2.1 * math.log(2) / 8, (
                f"position {k}: fitted {t} vs true {t_true}")

        # The paper's point: ECE drops to the ~1% regime (measured on this
        # seeded trace: raw 0.037 -> calibrated 0.006).
        assert raw_ece > 0.025, f"synthetic trace not miscalibrated: {raw_ece}"
        assert cal_ece < 0.01, f"calibrated ECE {cal_ece} (raw {raw_ece})"
        assert cal_ece < raw_ece / 3

    def test_calibrated_trace_fits_identity(self):
        # A perfectly calibrated head (T_true = 1 everywhere) must fit
        # ~identity temperatures (grid contains exactly 1.0; ties prefer
        # the T closest to 1).
        global T_TRUE
        saved = T_TRUE
        T_TRUE = [1.0] * GAMMA
        try:
            trace = _synthetic_trace(6000, seed=11)
        finally:
            T_TRUE = saved
        temps = fit_sts_temperatures(trace, GAMMA)
        for t in temps:
            assert abs(math.log(t)) <= 2.1 * math.log(2) / 8, temps

    def test_order_preserving_for_any_positive_temperature(self):
        # sigma(logit(c)/T) is strictly increasing in c for T > 0 — the
        # STS contract: rankings within a position never change.
        rng = random.Random(3)
        for t in (0.125, 0.5, 1.0, 2.0, 8.0):
            cs = sorted(rng.uniform(0.001, 0.999) for _ in range(50))
            cal = [apply_temperature(c, t) for c in cs]
            assert cal == sorted(cal), f"ordering broken at T={t}"
            # Strictly (0,1) mathematically; float sigmoid may saturate to
            # 1.0 at extreme logit/T ratios — bounds stay [0, 1].
            assert all(0.0 <= x <= 1.0 for x in cal)

    def test_unobserved_positions_keep_identity(self):
        # No round drafted deeper than 2 -> positions 2..3 stay T=1.
        trace = [ConfidenceRound(confidences=(0.9, 0.8), accepted=1)] * 50
        temps = fit_sts_temperatures(trace, GAMMA)
        assert len(temps) == GAMMA
        assert temps[2] == 1.0 and temps[3] == 1.0

    def test_empty_trace_identity(self):
        assert fit_sts_temperatures([], GAMMA) == [1.0] * GAMMA

    def test_ece_basic_properties(self):
        # Perfect predictions -> 0; constant-0.5 on balanced labels -> 0;
        # confident-wrong -> large.
        assert expected_calibration_error([1.0 - 1e-9, 1e-9], [1, 0]) < 1e-6
        assert expected_calibration_error([0.5, 0.5], [1, 0]) < 1e-9
        assert expected_calibration_error([0.99, 0.99], [0, 0]) > 0.9


class TestStsApplication:
    def _draft(self, temps=()):
        return DsparkDraft(DsparkDraftConfig(
            enabled=True, block_size=8, speculative_tokens=7,
            confidence_enabled=True, sts_temperatures=temps))

    def test_survival_confidences_applies_temperatures_before_cumprod(self):
        d = self._draft(temps=(2.0, 1.0))
        res = DsparkDraftResult(tokens=[1, 2], confidences=[0.8, 0.6])
        a = d.survival_confidences(res)
        c0 = apply_temperature(0.8, 2.0)   # calibrated position 0
        assert a is not None
        assert abs(a[0] - c0) < 1e-12
        assert abs(a[1] - c0 * 0.6) < 1e-12  # T=1 -> raw c_1

    def test_identity_default_is_raw_cumprod(self):
        d = self._draft()
        res = DsparkDraftResult(tokens=[1, 2], confidences=[0.8, 0.6])
        assert d.survival_confidences(res) == [0.8, 0.8 * 0.6]

    def test_non_positive_temperature_rejected(self):
        try:
            self._draft(temps=(1.0, 0.0))
        except ValueError:
            pass
        else:
            raise AssertionError("T=0 must be rejected")

    def test_trace_recorded_for_fit(self):
        d = self._draft()
        d.record_result(DsparkDraftResult(
            tokens=[1, 2, 3], confidences=[0.9, 0.8, 0.7]), num_accepted=2)
        d.record_result(DsparkDraftResult(
            tokens=[4], confidences=[0.4]), num_accepted=0)
        # No confidences (head off / DSP-5 round) -> not recorded.
        d.record_result(DsparkDraftResult(tokens=[5]), num_accepted=1)
        trace = d.confidence_trace
        assert len(trace) == 2
        assert trace[0] == ConfidenceRound((0.9, 0.8, 0.7), 2)
        assert trace[1] == ConfidenceRound((0.4,), 0)
        # The trace feeds the fit directly.
        temps = fit_sts_temperatures(trace, gamma=3)
        assert len(temps) == 3

    def test_trace_capacity_bounded(self):
        d = DsparkDraft(DsparkDraftConfig(
            enabled=True, confidence_enabled=True,
            confidence_trace_capacity=3))
        for i in range(5):
            d.record_result(DsparkDraftResult(
                tokens=[i], confidences=[0.5]), num_accepted=1)
        assert len(d.confidence_trace) == 3
