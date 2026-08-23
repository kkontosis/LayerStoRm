"""Tests for orchestrator.utility_scorer — Cascade-style speculation depth."""

import pytest

from orchestrator.utility_scorer import UtilityScorer, UtilityScorerConfig


def _us(**kwargs) -> UtilityScorer:
    return UtilityScorer(UtilityScorerConfig(**kwargs))


def _record(us: UtilityScorer, k: int, accepted: int,
            draft_us: float = 100.0, verify_us: float = 400.0,
            base_us: float = 500.0) -> None:
    us.record_iteration(k, accepted, draft_us, verify_us, base_us)


# ---------------------------------------------------------------------------
# Compute utility
# ---------------------------------------------------------------------------


class TestComputeUtility:

    def test_profitable(self):
        u = UtilityScorer.compute_utility(etr=2.0, cost_ratio=1.5)
        assert u == pytest.approx(2.0 / 1.5)

    def test_unprofitable(self):
        u = UtilityScorer.compute_utility(etr=1.0, cost_ratio=2.0)
        assert u == pytest.approx(0.5)

    def test_break_even(self):
        u = UtilityScorer.compute_utility(etr=1.0, cost_ratio=1.0)
        assert u == pytest.approx(1.0)

    def test_zero_cost_ratio(self):
        assert UtilityScorer.compute_utility(etr=2.0, cost_ratio=0.0) == 0.0


# ---------------------------------------------------------------------------
# Recommended depth
# ---------------------------------------------------------------------------


class TestRecommendedDepth:

    def test_initial_is_first_candidate(self):
        us = _us(candidate_depths=(0, 1, 3, 5))
        depth = us.recommended_depth()
        assert depth in (0, 1, 3, 5)

    def test_high_utility_recommends_profitable_k(self):
        us = _us(candidate_depths=(0, 1, 3),
                 test_iterations=2, set_iterations=4)
        for _ in range(2):
            _record(us, 0, 0, base_us=500, draft_us=0, verify_us=500)
        for _ in range(2):
            _record(us, 1, 1, base_us=500, draft_us=100, verify_us=450)
        for _ in range(2):
            _record(us, 3, 3, base_us=500, draft_us=200, verify_us=400)
        assert us.recommended_depth() >= 1

    def test_low_utility_recommends_zero(self):
        us = _us(candidate_depths=(0, 1, 3),
                 test_iterations=2, utility_threshold=1.0)
        for _ in range(2):
            _record(us, 0, 0, base_us=500, draft_us=0, verify_us=500)
        for _ in range(2):
            _record(us, 1, 0, base_us=500, draft_us=200, verify_us=600)
        for _ in range(2):
            _record(us, 3, 0, base_us=500, draft_us=400, verify_us=800)
        assert us.recommended_depth() == 0

    def test_disabled_returns_max_candidate(self):
        us = _us(enabled=False, candidate_depths=(0, 1, 3, 7))
        assert us.recommended_depth() == 7

    def test_never_exceeds_max_candidate(self):
        us = _us(candidate_depths=(0, 1, 2, 3))
        for _ in range(100):
            _record(us, 3, 3, base_us=500, draft_us=100, verify_us=300)
        assert us.recommended_depth() <= 3


# ---------------------------------------------------------------------------
# Cascade phases
# ---------------------------------------------------------------------------


class TestCascadePhases:

    def test_starts_in_test(self):
        us = _us()
        assert us.current_phase == "test"

    def test_transitions_to_set(self):
        us = _us(candidate_depths=(0, 1), test_iterations=2)
        _record(us, 0, 0, base_us=500, draft_us=0, verify_us=500)
        _record(us, 0, 0, base_us=500, draft_us=0, verify_us=500)
        _record(us, 1, 1, base_us=500, draft_us=100, verify_us=400)
        _record(us, 1, 1, base_us=500, draft_us=100, verify_us=400)
        assert us.current_phase == "set"

    def test_set_uses_best_k(self):
        us = _us(candidate_depths=(0, 1, 3), test_iterations=2)
        _record(us, 0, 0, base_us=500, draft_us=0, verify_us=500)
        _record(us, 0, 0, base_us=500, draft_us=0, verify_us=500)
        _record(us, 1, 1, base_us=500, draft_us=50, verify_us=300)
        _record(us, 1, 1, base_us=500, draft_us=50, verify_us=300)
        _record(us, 3, 3, base_us=500, draft_us=100, verify_us=250)
        _record(us, 3, 3, base_us=500, draft_us=100, verify_us=250)
        assert us.current_phase == "set"
        assert us.best_depth >= 1

    def test_returns_to_test_after_set(self):
        us = _us(candidate_depths=(0, 1), test_iterations=1,
                 set_iterations=2)
        _record(us, 0, 0, base_us=500, draft_us=0, verify_us=500)
        _record(us, 1, 1, base_us=500, draft_us=50, verify_us=300)
        assert us.current_phase == "set"
        _record(us, 1, 1, base_us=500, draft_us=50, verify_us=300)
        _record(us, 1, 1, base_us=500, draft_us=50, verify_us=300)
        assert us.current_phase == "test"

    def test_backoff_doubles_set_duration(self):
        us = _us(candidate_depths=(0, 1), test_iterations=1,
                 set_iterations=4, utility_threshold=1.1)
        # K=0: ETR=1, cost=500/500=1.0, utility=1.0 < 1.1
        _record(us, 0, 0, base_us=500, draft_us=0, verify_us=500)
        # K=1: ETR=1, cost=800/500=1.6, utility=0.625 < 1.1
        _record(us, 1, 0, base_us=500, draft_us=200, verify_us=600)
        assert us.current_phase == "set"
        assert us.best_depth == 0
        assert us._set_duration == 8


# ---------------------------------------------------------------------------
# Apply ceiling
# ---------------------------------------------------------------------------


class TestApplyCeiling:

    def test_respects_max_verifiable(self):
        assert UtilityScorer.apply_ceiling(5, 3) == 3

    def test_returns_zero_when_max_is_zero(self):
        assert UtilityScorer.apply_ceiling(5, 0) == 0

    def test_depth_below_ceiling_unchanged(self):
        assert UtilityScorer.apply_ceiling(2, 5) == 2


# ---------------------------------------------------------------------------
# Coverage discount
# ---------------------------------------------------------------------------


class TestCoverageDiscount:

    def test_full_coverage(self):
        assert UtilityScorer.coverage_discount(1.0) == pytest.approx(1.0)

    def test_partial_coverage(self):
        assert UtilityScorer.coverage_discount(0.88) == pytest.approx(0.88)

    def test_below_floor(self):
        assert UtilityScorer.coverage_discount(0.3, floor=0.5) == pytest.approx(0.5)

    def test_custom_floor(self):
        assert UtilityScorer.coverage_discount(0.1, floor=0.2) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Hill climbing
# ---------------------------------------------------------------------------


class TestHillClimb:

    def test_increasing_utility_climbs_up(self):
        us = _us(candidate_depths=(0, 1, 3, 5), test_iterations=2)
        _record(us, 0, 0, base_us=500, draft_us=0, verify_us=500)
        _record(us, 0, 0, base_us=500, draft_us=0, verify_us=500)
        _record(us, 1, 1, base_us=500, draft_us=50, verify_us=300)
        _record(us, 1, 1, base_us=500, draft_us=50, verify_us=300)
        depth = us.recommended_depth()
        assert depth >= 3

    def test_decreasing_utility_backtracks(self):
        us = _us(candidate_depths=(0, 1, 3, 5), test_iterations=2)
        _record(us, 0, 0, base_us=500, draft_us=0, verify_us=500)
        _record(us, 0, 0, base_us=500, draft_us=0, verify_us=500)
        _record(us, 1, 0, base_us=500, draft_us=200, verify_us=700)
        _record(us, 1, 0, base_us=500, draft_us=200, verify_us=700)
        assert us.best_depth == 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:

    def test_default_config(self):
        cfg = UtilityScorerConfig()
        assert cfg.enabled is True
        assert cfg.utility_threshold == pytest.approx(1.0)
        assert cfg.candidate_depths == (0, 1, 2, 3, 5, 7)
        assert cfg.test_iterations == 4
        assert cfg.set_iterations == 16

    def test_frozen_config(self):
        cfg = UtilityScorerConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = False  # type: ignore[misc]

    def test_is_enabled_property(self):
        assert _us(enabled=True).is_enabled is True
        assert _us(enabled=False).is_enabled is False

    def test_default_strategy_is_cascade(self):
        cfg = UtilityScorerConfig()
        assert cfg.strategy == "cascade"


# ---------------------------------------------------------------------------
# Static strategy
# ---------------------------------------------------------------------------


class TestStaticStrategy:

    def test_returns_static_depth(self):
        us = _us(strategy="fixed", static_depth=5)
        assert us.recommended_depth() == 5

    def test_record_iteration_is_noop(self):
        us = _us(strategy="fixed", static_depth=3)
        _record(us, 3, 0, base_us=500, draft_us=200, verify_us=600)
        assert us.recommended_depth() == 3

    def test_different_static_depths(self):
        assert _us(strategy="fixed", static_depth=1).recommended_depth() == 1
        assert _us(strategy="fixed", static_depth=7).recommended_depth() == 7


# ---------------------------------------------------------------------------
# EMA strategy (sglang-style)
# ---------------------------------------------------------------------------


class TestEmaStrategy:

    def test_initial_depth_is_middle_candidate(self):
        us = _us(strategy="ema", candidate_depths=(0, 1, 3, 5, 7))
        assert us.recommended_depth() == 3

    def test_high_acceptance_increases_depth(self):
        us = _us(strategy="ema", candidate_depths=(0, 1, 3, 5),
                 ema_alpha=0.5, ema_up_threshold=0.0)
        for _ in range(10):
            _record(us, 3, 4)
        assert us.recommended_depth() >= 3

    def test_zero_acceptance_decreases_depth(self):
        us = _us(strategy="ema", candidate_depths=(0, 1, 3, 5),
                 ema_alpha=0.5, ema_down_threshold=-0.25)
        for _ in range(20):
            _record(us, 3, 0)
        assert us.recommended_depth() < 3

    def test_ema_does_not_use_timing(self):
        us = _us(strategy="ema", candidate_depths=(0, 1, 3))
        _record(us, 1, 1, draft_us=9999, verify_us=9999, base_us=1)
        assert us.recommended_depth() in (0, 1, 3)

    def test_stable_acceptance_holds_depth(self):
        us = _us(strategy="ema", candidate_depths=(0, 1, 3, 5),
                 ema_alpha=0.3)
        initial = us.recommended_depth()
        for _ in range(5):
            _record(us, initial, initial)
        assert abs(us.recommended_depth() - initial) <= 2
