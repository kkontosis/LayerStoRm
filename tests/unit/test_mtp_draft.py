"""Tests for orchestrator.mtp_draft — DeepSeek MTP head draft generation."""

import pytest

from orchestrator.mtp_draft import (
    DraftStep,
    MtpDraft,
    MtpDraftConfig,
    MtpDraftResult,
    MtpLayerPlan,
)


# ---------------------------------------------------------------------------
# MTP layer index (cyclic reuse)
# ---------------------------------------------------------------------------


class TestMtpLayerIdx:

    def test_single_mtp_layer_step_0(self):
        mtp = MtpDraft(num_mtp_layers=1, num_layers=61)
        assert mtp.mtp_layer_idx(0) == 61

    def test_single_mtp_layer_reuses(self):
        mtp = MtpDraft(num_mtp_layers=1, num_layers=61)
        assert mtp.mtp_layer_idx(0) == 61
        assert mtp.mtp_layer_idx(1) == 61
        assert mtp.mtp_layer_idx(2) == 61

    def test_two_mtp_layers_alternate(self):
        mtp = MtpDraft(num_mtp_layers=2, num_layers=61)
        assert mtp.mtp_layer_idx(0) == 61
        assert mtp.mtp_layer_idx(1) == 62
        assert mtp.mtp_layer_idx(2) == 61
        assert mtp.mtp_layer_idx(3) == 62

    def test_glm5_num_layers(self):
        mtp = MtpDraft(num_mtp_layers=1, num_layers=78)
        assert mtp.mtp_layer_idx(0) == 78


# ---------------------------------------------------------------------------
# Plan draft steps
# ---------------------------------------------------------------------------


class TestPlanDraftSteps:

    def test_plan_three_steps(self):
        mtp = MtpDraft(num_mtp_layers=1, num_layers=61)
        plan = mtp.plan_draft_steps(3)
        assert len(plan) == 3
        assert all(isinstance(p, MtpLayerPlan) for p in plan)
        assert plan[0].step_idx == 0
        assert plan[0].mtp_layer_idx == 61
        assert plan[1].step_idx == 1
        assert plan[1].mtp_layer_idx == 61
        assert plan[2].step_idx == 2
        assert plan[2].mtp_layer_idx == 61

    def test_plan_depth_zero(self):
        mtp = MtpDraft()
        assert mtp.plan_draft_steps(0) == []

    def test_plan_depth_exceeds_max_is_capped(self):
        mtp = MtpDraft(MtpDraftConfig(max_depth=2))
        plan = mtp.plan_draft_steps(5)
        assert len(plan) == 2

    def test_plan_two_layers_cyclic(self):
        mtp = MtpDraft(MtpDraftConfig(max_depth=4),
                       num_mtp_layers=2, num_layers=61)
        plan = mtp.plan_draft_steps(4)
        assert [p.mtp_layer_idx for p in plan] == [61, 62, 61, 62]


# ---------------------------------------------------------------------------
# Dynamic depth gating
# ---------------------------------------------------------------------------


class TestShouldContinue:

    def test_first_step_always_continues(self):
        mtp = MtpDraft(MtpDraftConfig(dynamic_depth=True,
                                       confidence_threshold=0.9))
        assert mtp.should_continue(0, confidence=0.0) is True

    def test_high_confidence_continues(self):
        mtp = MtpDraft(MtpDraftConfig(confidence_threshold=0.4))
        assert mtp.should_continue(1, confidence=0.8) is True

    def test_low_confidence_stops(self):
        mtp = MtpDraft(MtpDraftConfig(confidence_threshold=0.4))
        assert mtp.should_continue(1, confidence=0.2) is False

    def test_at_threshold_continues(self):
        mtp = MtpDraft(MtpDraftConfig(confidence_threshold=0.4))
        assert mtp.should_continue(1, confidence=0.4) is True

    def test_dynamic_depth_disabled_ignores_confidence(self):
        mtp = MtpDraft(MtpDraftConfig(dynamic_depth=False,
                                       confidence_threshold=0.99))
        assert mtp.should_continue(1, confidence=0.01) is True
        assert mtp.should_continue(2, confidence=0.0) is True

    def test_at_max_depth_stops(self):
        mtp = MtpDraft(MtpDraftConfig(max_depth=3))
        assert mtp.should_continue(3, confidence=1.0) is False

    def test_beyond_max_depth_stops(self):
        mtp = MtpDraft(MtpDraftConfig(max_depth=2))
        assert mtp.should_continue(5, confidence=1.0) is False


# ---------------------------------------------------------------------------
# Draft result
# ---------------------------------------------------------------------------


class TestDraftResult:

    def test_tokens_property(self):
        result = MtpDraftResult(steps=[
            DraftStep(token_id=100, confidence=0.9, mtp_layer_idx=61),
            DraftStep(token_id=200, confidence=0.8, mtp_layer_idx=61),
        ])
        assert result.tokens == [100, 200]

    def test_depth_property(self):
        result = MtpDraftResult(steps=[
            DraftStep(token_id=1, confidence=0.5, mtp_layer_idx=61),
            DraftStep(token_id=2, confidence=0.4, mtp_layer_idx=61),
            DraftStep(token_id=3, confidence=0.3, mtp_layer_idx=61),
        ])
        assert result.depth == 3

    def test_empty_result(self):
        result = MtpDraftResult()
        assert result.tokens == []
        assert result.depth == 0

    def test_source_field(self):
        result = MtpDraftResult()
        assert result.source == "mtp"


# ---------------------------------------------------------------------------
# Acceptance rate tracking
# ---------------------------------------------------------------------------


class TestAcceptanceRate:

    def test_initial_acceptance_rate_zero(self):
        mtp = MtpDraft()
        assert mtp.acceptance_rate == 0.0

    def test_first_record_sets_rate(self):
        mtp = MtpDraft()
        result = MtpDraftResult(steps=[
            DraftStep(token_id=1, confidence=0.9, mtp_layer_idx=61),
            DraftStep(token_id=2, confidence=0.8, mtp_layer_idx=61),
        ])
        mtp.record_result(result, num_accepted=2)
        assert mtp.acceptance_rate == pytest.approx(1.0)

    def test_ema_smoothing(self):
        mtp = MtpDraft(MtpDraftConfig(acceptance_ema_alpha=0.5))
        step = DraftStep(token_id=1, confidence=0.9, mtp_layer_idx=61)

        mtp.record_result(MtpDraftResult(steps=[step, step]), num_accepted=2)
        assert mtp.acceptance_rate == pytest.approx(1.0)

        mtp.record_result(MtpDraftResult(steps=[step, step]), num_accepted=0)
        assert mtp.acceptance_rate == pytest.approx(0.5)

        mtp.record_result(MtpDraftResult(steps=[step, step]), num_accepted=0)
        assert mtp.acceptance_rate == pytest.approx(0.25)

    def test_empty_draft_no_ema_update(self):
        mtp = MtpDraft()
        step = DraftStep(token_id=1, confidence=0.9, mtp_layer_idx=61)
        mtp.record_result(MtpDraftResult(steps=[step]), num_accepted=1)
        rate_before = mtp.acceptance_rate

        mtp.record_result(MtpDraftResult(), num_accepted=0)
        assert mtp.acceptance_rate == rate_before

    def test_partial_acceptance(self):
        mtp = MtpDraft()
        steps = [DraftStep(token_id=i, confidence=0.5, mtp_layer_idx=61)
                 for i in range(3)]
        mtp.record_result(MtpDraftResult(steps=steps), num_accepted=2)
        assert mtp.acceptance_rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


class TestCounters:

    def test_total_drafts_incremented(self):
        mtp = MtpDraft()
        step = DraftStep(token_id=1, confidence=0.9, mtp_layer_idx=61)
        mtp.record_result(MtpDraftResult(steps=[step]), num_accepted=1)
        mtp.record_result(MtpDraftResult(steps=[step, step]), num_accepted=1)
        assert mtp.total_drafts == 2

    def test_total_steps_accumulates(self):
        mtp = MtpDraft()
        step = DraftStep(token_id=1, confidence=0.9, mtp_layer_idx=61)
        mtp.record_result(MtpDraftResult(steps=[step]), num_accepted=1)
        mtp.record_result(MtpDraftResult(steps=[step, step, step]),
                          num_accepted=2)
        assert mtp.total_steps == 4

    def test_total_accepted_accumulates(self):
        mtp = MtpDraft()
        step = DraftStep(token_id=1, confidence=0.9, mtp_layer_idx=61)
        mtp.record_result(MtpDraftResult(steps=[step, step]), num_accepted=1)
        mtp.record_result(MtpDraftResult(steps=[step, step]), num_accepted=2)
        assert mtp.total_accepted == 3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:

    def test_default_config(self):
        cfg = MtpDraftConfig()
        assert cfg.enabled is True
        assert cfg.max_depth == 3
        assert cfg.dynamic_depth is True
        assert cfg.confidence_threshold == pytest.approx(0.4)
        assert cfg.acceptance_ema_alpha == pytest.approx(0.3)

    def test_frozen_config(self):
        cfg = MtpDraftConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = False  # type: ignore[misc]

    def test_is_enabled_property(self):
        assert MtpDraft(MtpDraftConfig(enabled=True)).is_enabled is True
        assert MtpDraft(MtpDraftConfig(enabled=False)).is_enabled is False

    def test_max_depth_property(self):
        mtp = MtpDraft(MtpDraftConfig(max_depth=5))
        assert mtp.max_depth == 5

    def test_num_mtp_layers_property(self):
        mtp = MtpDraft(num_mtp_layers=2)
        assert mtp.num_mtp_layers == 2
