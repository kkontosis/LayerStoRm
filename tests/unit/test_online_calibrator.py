"""Tests for orchestrator.online_calibrator — acceptance rate feedback loop."""

import pytest

from orchestrator.online_calibrator import (
    CalibrationConfig,
    DraftParams,
    OnlineCalibrator,
)


def _cal(min_rate: float = 0.3, target: float = 0.6,
         interval: int = 100, alpha: float = 1.0) -> CalibrationConfig:
    return CalibrationConfig(
        min_acceptance_rate=min_rate,
        target_acceptance_rate=target,
        adjustment_interval_tokens=interval,
        acceptance_ema_alpha=alpha,
    )


class TestLowAcceptance:
    def test_disables_layer_skip_first(self):
        params = DraftParams(draft_expert_count=1, layer_skip_enabled=True)
        cal = OnlineCalibrator(_cal(alpha=1.0), params)
        adjusted = cal.update(0.1, 100)
        assert adjusted is True
        assert params.layer_skip_enabled is False
        assert params.draft_expert_count == 1

    def test_increases_expert_count_after_skip_disabled(self):
        params = DraftParams(draft_expert_count=1, layer_skip_enabled=False)
        cal = OnlineCalibrator(_cal(alpha=1.0), params)
        adjusted = cal.update(0.1, 100)
        assert adjusted is True
        assert params.draft_expert_count == 2

    def test_expert_count_clamped_at_max(self):
        params = DraftParams(draft_expert_count=8, layer_skip_enabled=False,
                             max_draft_expert_count=8)
        cal = OnlineCalibrator(_cal(alpha=1.0), params)
        adjusted = cal.update(0.1, 100)
        assert adjusted is False
        assert params.draft_expert_count == 8

    def test_two_step_reduction(self):
        params = DraftParams(draft_expert_count=1, layer_skip_enabled=True)
        cal = OnlineCalibrator(_cal(alpha=1.0, interval=50), params)

        cal.update(0.1, 50)
        assert params.layer_skip_enabled is False
        assert params.draft_expert_count == 1

        cal.update(0.1, 50)
        assert params.draft_expert_count == 2


class TestHighAcceptance:
    def test_decreases_expert_count(self):
        params = DraftParams(draft_expert_count=3, layer_skip_enabled=False)
        cal = OnlineCalibrator(_cal(alpha=1.0), params)
        adjusted = cal.update(0.8, 100)
        assert adjusted is True
        assert params.draft_expert_count == 2

    def test_enables_layer_skip_at_min_count(self):
        params = DraftParams(draft_expert_count=1, layer_skip_enabled=False)
        cal = OnlineCalibrator(_cal(alpha=1.0), params)
        adjusted = cal.update(0.8, 100)
        assert adjusted is True
        assert params.layer_skip_enabled is True

    def test_no_adjustment_if_already_max_aggressive(self):
        params = DraftParams(draft_expert_count=1, layer_skip_enabled=True)
        cal = OnlineCalibrator(_cal(alpha=1.0), params)
        adjusted = cal.update(0.8, 100)
        assert adjusted is False


class TestInterval:
    def test_no_adjustment_before_interval(self):
        params = DraftParams(draft_expert_count=1, layer_skip_enabled=False)
        cal = OnlineCalibrator(_cal(alpha=1.0, interval=200), params)
        assert cal.update(0.1, 50) is False
        assert cal.update(0.1, 50) is False
        assert cal.update(0.1, 50) is False
        assert params.draft_expert_count == 1

    def test_adjusts_at_interval(self):
        params = DraftParams(draft_expert_count=1, layer_skip_enabled=False)
        cal = OnlineCalibrator(_cal(alpha=1.0, interval=200), params)
        cal.update(0.1, 100)
        adjusted = cal.update(0.1, 100)
        assert adjusted is True
        assert params.draft_expert_count == 2


class TestEma:
    def test_smoothing(self):
        cal = OnlineCalibrator(_cal(alpha=0.5, interval=1000))
        cal.update(1.0, 0)
        assert cal.smoothed_acceptance_rate == pytest.approx(1.0)
        cal.update(0.0, 0)
        assert cal.smoothed_acceptance_rate == pytest.approx(0.5)
        cal.update(0.0, 0)
        assert cal.smoothed_acceptance_rate == pytest.approx(0.25)

    def test_alpha_one_uses_latest(self):
        cal = OnlineCalibrator(_cal(alpha=1.0, interval=1000))
        cal.update(0.9, 0)
        cal.update(0.1, 0)
        assert cal.smoothed_acceptance_rate == pytest.approx(0.1)


class TestNeutralZone:
    def test_no_adjustment_between_min_and_target(self):
        params = DraftParams(draft_expert_count=2, layer_skip_enabled=False)
        cal = OnlineCalibrator(_cal(min_rate=0.3, target=0.6, alpha=1.0), params)
        adjusted = cal.update(0.45, 100)
        assert adjusted is False
        assert params.draft_expert_count == 2


class TestAdjustmentCount:
    def test_tracks_adjustments(self):
        params = DraftParams(draft_expert_count=1, layer_skip_enabled=True)
        cal = OnlineCalibrator(_cal(alpha=1.0, interval=50), params)
        assert cal.adjustments_made == 0

        cal.update(0.1, 50)
        assert cal.adjustments_made == 1

        cal.update(0.1, 50)
        assert cal.adjustments_made == 2
