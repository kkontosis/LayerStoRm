"""Tests for draft_combiner — multi-source draft merging with utility selection."""

import pytest

from orchestrator.draft_combiner import (
    CombinedDraft,
    CombinedDraftStep,
    DraftCombiner,
    DraftCombinerConfig,
)
from orchestrator.mtp_draft import DraftStep, MtpDraftResult
from orchestrator.self_speculative import SelfSpecDraftResult


def _mtp(tokens_and_confidences: list[tuple[int, float]]) -> MtpDraftResult:
    return MtpDraftResult(
        steps=[DraftStep(token_id=t, confidence=c, mtp_layer_idx=61 + i)
               for i, (t, c) in enumerate(tokens_and_confidences)],
    )


def _ss(tokens_and_confidences: list[tuple[int, float]]) -> SelfSpecDraftResult:
    return SelfSpecDraftResult(
        steps=[DraftStep(token_id=t, confidence=c, mtp_layer_idx=0)
               for t, c in tokens_and_confidences],
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:

    def test_defaults(self):
        c = DraftCombinerConfig()
        assert c.enabled is True
        assert c.prompt_lookup_priority_boost == 1.0
        assert c.mtp_compute_cost == 0.1
        assert c.self_spec_compute_cost == 0.5
        assert c.acceptance_ema_alpha == 0.3
        assert c.latency_per_position_us == 920.0
        assert c.min_utility_threshold == 0.0

    def test_frozen(self):
        c = DraftCombinerConfig()
        with pytest.raises(AttributeError):
            c.enabled = False  # type: ignore[misc]

    def test_is_enabled(self):
        dc = DraftCombiner()
        assert dc.is_enabled is True
        dc2 = DraftCombiner(DraftCombinerConfig(enabled=False))
        assert dc2.is_enabled is False

    def test_custom_values(self):
        c = DraftCombinerConfig(
            prompt_lookup_priority_boost=2.0,
            mtp_compute_cost=0.2,
            self_spec_compute_cost=0.8,
        )
        assert c.prompt_lookup_priority_boost == 2.0
        assert c.mtp_compute_cost == 0.2
        assert c.self_spec_compute_cost == 0.8


# ---------------------------------------------------------------------------
# CombinedDraft type
# ---------------------------------------------------------------------------

class TestCombinedDraft:

    def test_tokens(self):
        cd = CombinedDraft(steps=[
            CombinedDraftStep(10, 0, "mtp", 0.8, 100.0),
            CombinedDraftStep(20, 1, "prompt_lookup", 0.6, 90.0),
        ])
        assert cd.tokens == [10, 20]

    def test_depth(self):
        cd = CombinedDraft(steps=[
            CombinedDraftStep(1, 0, "mtp", 0.9, 50.0),
            CombinedDraftStep(2, 1, "mtp", 0.8, 40.0),
            CombinedDraftStep(3, 2, "mtp", 0.7, 30.0),
        ])
        assert cd.depth == 3

    def test_empty(self):
        cd = CombinedDraft()
        assert cd.tokens == []
        assert cd.depth == 0
        assert cd.source_breakdown == {}

    def test_source_breakdown(self):
        cd = CombinedDraft(steps=[
            CombinedDraftStep(1, 0, "prompt_lookup", 0.5, 100.0),
            CombinedDraftStep(2, 1, "prompt_lookup", 0.5, 90.0),
            CombinedDraftStep(3, 2, "mtp", 0.8, 80.0),
        ])
        assert cd.source_breakdown == {"prompt_lookup": 2, "mtp": 1}


# ---------------------------------------------------------------------------
# Prompt-lookup priority (INV-0.8d)
# ---------------------------------------------------------------------------

class TestPromptLookupPriority:

    def test_wins_over_mtp(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[100, 200],
            mtp_result=_mtp([(300, 0.95), (400, 0.95)]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.5,
            max_depth=5,
            max_verifiable_depth=5,
        )
        assert result.steps[0].source == "prompt_lookup"
        assert result.steps[0].token_id == 100
        assert result.steps[1].source == "prompt_lookup"
        assert result.steps[1].token_id == 200

    def test_fills_covered_positions(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[10, 20, 30],
            mtp_result=_mtp([(40, 0.9), (50, 0.9), (60, 0.9), (70, 0.9)]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.6,
            max_depth=5,
            max_verifiable_depth=5,
        )
        assert result.steps[0].source == "prompt_lookup"
        assert result.steps[1].source == "prompt_lookup"
        assert result.steps[2].source == "prompt_lookup"
        assert result.steps[3].source == "mtp"

    def test_zero_compute_cost(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[10],
            mtp_result=None,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.3,
            max_depth=1,
            max_verifiable_depth=1,
        )
        assert result.depth == 1
        step = result.steps[0]
        assert step.utility > 0

    def test_empty_lookup_falls_through(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[],
            mtp_result=_mtp([(42, 0.8)]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.5,
            max_depth=3,
            max_verifiable_depth=3,
        )
        assert result.steps[0].source == "mtp"
        assert result.steps[0].token_id == 42

    def test_boost_zero_allows_mtp_win(self):
        cfg = DraftCombinerConfig(prompt_lookup_priority_boost=0.0)
        dc = DraftCombiner(cfg)
        result = dc.combine(
            prompt_lookup_tokens=[10],
            mtp_result=_mtp([(20, 0.99)]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.01,
            max_depth=1,
            max_verifiable_depth=1,
        )
        assert result.steps[0].source == "mtp"


# ---------------------------------------------------------------------------
# Utility ranking
# ---------------------------------------------------------------------------

class TestUtilityRanking:

    def test_higher_confidence_preferred(self):
        dc = DraftCombiner(DraftCombinerConfig(prompt_lookup_priority_boost=0.0))
        result = dc.combine(
            prompt_lookup_tokens=[],
            mtp_result=_mtp([(10, 0.9)]),
            self_spec_result=_ss([(20, 0.3)]),
            prompt_lookup_acceptance_rate=0.0,
            max_depth=1,
            max_verifiable_depth=1,
        )
        assert result.steps[0].source == "mtp"

    def test_deeper_positions_more_latency(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[],
            mtp_result=_mtp([(10, 0.5), (20, 0.5), (30, 0.5)]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.0,
            max_depth=3,
            max_verifiable_depth=3,
        )
        assert result.steps[0].utility < result.steps[1].utility
        assert result.steps[1].utility < result.steps[2].utility

    def test_high_cost_penalized(self):
        dc = DraftCombiner(DraftCombinerConfig(
            prompt_lookup_priority_boost=0.0,
            mtp_compute_cost=0.05,
            self_spec_compute_cost=0.95,
        ))
        result = dc.combine(
            prompt_lookup_tokens=[],
            mtp_result=_mtp([(10, 0.5)]),
            self_spec_result=_ss([(20, 0.5)]),
            prompt_lookup_acceptance_rate=0.0,
            max_depth=1,
            max_verifiable_depth=1,
        )
        assert result.steps[0].source == "mtp"

    def test_below_threshold_excluded(self):
        dc = DraftCombiner(DraftCombinerConfig(min_utility_threshold=99999.0))
        result = dc.combine(
            prompt_lookup_tokens=[10],
            mtp_result=_mtp([(20, 0.5)]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.5,
            max_depth=3,
            max_verifiable_depth=3,
        )
        assert result.depth == 0


# ---------------------------------------------------------------------------
# Depth ceiling
# ---------------------------------------------------------------------------

class TestDepthCeiling:

    def test_max_depth_caps(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[1, 2, 3, 4, 5],
            mtp_result=None,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.8,
            max_depth=2,
            max_verifiable_depth=10,
        )
        assert result.depth == 2

    def test_max_verifiable_depth_caps(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[1, 2, 3, 4, 5],
            mtp_result=None,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.8,
            max_depth=10,
            max_verifiable_depth=3,
        )
        assert result.depth == 3

    def test_ceiling_is_min(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[1, 2, 3, 4, 5, 6, 7],
            mtp_result=None,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.8,
            max_depth=4,
            max_verifiable_depth=3,
        )
        assert result.depth == 3


# ---------------------------------------------------------------------------
# Coverage discount (INV-A)
# ---------------------------------------------------------------------------

class TestCoverageDiscount:

    def test_low_coverage_reduces_utility(self):
        dc = DraftCombiner(DraftCombinerConfig(prompt_lookup_priority_boost=0.0))
        full = dc.combine(
            prompt_lookup_tokens=[],
            mtp_result=_mtp([(10, 0.6)]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.0,
            max_depth=1,
            max_verifiable_depth=1,
            expert_coverage_fraction=1.0,
        )

        dc2 = DraftCombiner(DraftCombinerConfig(prompt_lookup_priority_boost=0.0))
        partial = dc2.combine(
            prompt_lookup_tokens=[],
            mtp_result=_mtp([(10, 0.6)]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.0,
            max_depth=1,
            max_verifiable_depth=1,
            expert_coverage_fraction=0.6,
        )
        assert partial.steps[0].p_accept < full.steps[0].p_accept
        assert partial.steps[0].utility < full.steps[0].utility

    def test_full_coverage_no_discount(self):
        dc = DraftCombiner(DraftCombinerConfig(prompt_lookup_priority_boost=0.0))
        result = dc.combine(
            prompt_lookup_tokens=[],
            mtp_result=_mtp([(10, 0.8)]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.0,
            max_depth=1,
            max_verifiable_depth=1,
            expert_coverage_fraction=1.0,
        )
        assert result.steps[0].p_accept == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_all_empty(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[],
            mtp_result=None,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.0,
            max_depth=5,
            max_verifiable_depth=5,
        )
        assert result.depth == 0

    def test_only_prompt_lookup(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[10, 20],
            mtp_result=None,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.7,
            max_depth=5,
            max_verifiable_depth=5,
        )
        assert result.depth == 2
        assert all(s.source == "prompt_lookup" for s in result.steps)

    def test_only_mtp(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[],
            mtp_result=_mtp([(10, 0.8), (20, 0.7)]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.0,
            max_depth=5,
            max_verifiable_depth=5,
        )
        assert result.depth == 2
        assert all(s.source == "mtp" for s in result.steps)

    def test_disabled_returns_empty(self):
        dc = DraftCombiner(DraftCombinerConfig(enabled=False))
        result = dc.combine(
            prompt_lookup_tokens=[1, 2, 3],
            mtp_result=_mtp([(10, 0.9)]),
            self_spec_result=_ss([(20, 0.9)]),
            prompt_lookup_acceptance_rate=0.9,
            max_depth=5,
            max_verifiable_depth=5,
        )
        assert result.depth == 0


# ---------------------------------------------------------------------------
# Contiguity
# ---------------------------------------------------------------------------

class TestContiguity:

    def test_gap_truncates(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[],
            mtp_result=MtpDraftResult(steps=[
                DraftStep(token_id=10, confidence=0.8, mtp_layer_idx=61),
            ]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.0,
            max_depth=5,
            max_verifiable_depth=5,
        )
        assert result.depth == 1

    def test_mixed_lengths(self):
        dc = DraftCombiner()
        result = dc.combine(
            prompt_lookup_tokens=[10, 20],
            mtp_result=_mtp([(30, 0.7), (40, 0.7), (50, 0.7), (60, 0.7)]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.5,
            max_depth=4,
            max_verifiable_depth=4,
        )
        assert result.depth == 4
        assert result.steps[0].source == "prompt_lookup"
        assert result.steps[1].source == "prompt_lookup"
        assert result.steps[2].source == "mtp"
        assert result.steps[3].source == "mtp"


# ---------------------------------------------------------------------------
# Acceptance tracking
# ---------------------------------------------------------------------------

class TestAcceptanceTracking:

    def test_initial_rate(self):
        dc = DraftCombiner()
        assert dc.acceptance_rate == 0.0

    def test_first_record(self):
        dc = DraftCombiner()
        cd = CombinedDraft(steps=[
            CombinedDraftStep(1, 0, "mtp", 0.8, 100.0),
            CombinedDraftStep(2, 1, "mtp", 0.7, 90.0),
        ])
        dc.record_result(cd, num_accepted=1)
        assert dc.acceptance_rate == pytest.approx(0.5)

    def test_ema_smoothing(self):
        cfg = DraftCombinerConfig(acceptance_ema_alpha=0.5)
        dc = DraftCombiner(cfg)
        cd = CombinedDraft(steps=[
            CombinedDraftStep(1, 0, "mtp", 0.8, 100.0),
        ])
        dc.record_result(cd, num_accepted=1)
        assert dc.acceptance_rate == pytest.approx(1.0)

        dc.record_result(cd, num_accepted=0)
        assert dc.acceptance_rate == pytest.approx(0.5)

        dc.record_result(cd, num_accepted=0)
        assert dc.acceptance_rate == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:

    def test_total_combines(self):
        dc = DraftCombiner()
        assert dc.total_combines == 0
        dc.combine([], None, None, 0.0, 3, 3)
        assert dc.total_combines == 1
        dc.combine([], None, None, 0.0, 3, 3)
        assert dc.total_combines == 2

    def test_source_selection_rates(self):
        dc = DraftCombiner()
        dc.combine(
            prompt_lookup_tokens=[10, 20],
            mtp_result=_mtp([(30, 0.8), (40, 0.8), (50, 0.8), (60, 0.8)]),
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.6,
            max_depth=4,
            max_verifiable_depth=4,
        )
        rates = dc.source_selection_rates
        assert rates["prompt_lookup"] == pytest.approx(2 / 4)
        assert rates["mtp"] == pytest.approx(2 / 4)
        assert rates["self_speculative"] == pytest.approx(0.0)
