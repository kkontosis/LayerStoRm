"""Tests for orchestrator.self_speculative — reduced-expert draft generation."""

import pytest

from orchestrator.mtp_draft import DraftStep
from orchestrator.self_speculative import (
    SelfSpecDraftResult,
    SelfSpecLayerPlan,
    SelfSpeculative,
    SelfSpeculativeConfig,
)

N_LAYERS = 10
N_MOE = 7
FIRST_MOE = 3


def _ss(**kwargs) -> SelfSpeculative:
    cfg_args = {}
    other_args = {}
    cfg_fields = {f.name for f in SelfSpeculativeConfig.__dataclass_fields__.values()}
    for k, v in kwargs.items():
        if k in cfg_fields:
            cfg_args[k] = v
        else:
            other_args[k] = v
    cfg = SelfSpeculativeConfig(**cfg_args)
    other_args.setdefault("num_layers", N_LAYERS)
    other_args.setdefault("num_moe_layers", N_MOE)
    other_args.setdefault("first_moe_layer", FIRST_MOE)
    return SelfSpeculative(cfg, **other_args)


def _step(token_id: int = 1, confidence: float = 0.9) -> DraftStep:
    return DraftStep(token_id=token_id, confidence=confidence, mtp_layer_idx=0)


# ---------------------------------------------------------------------------
# Plan forward pass
# ---------------------------------------------------------------------------


class TestPlanForwardPass:

    def test_returns_all_layers(self):
        ss = _ss()
        plan = ss.plan_forward_pass()
        assert len(plan) == N_LAYERS
        assert all(isinstance(p, SelfSpecLayerPlan) for p in plan)

    def test_moe_layers_marked(self):
        ss = _ss()
        plan = ss.plan_forward_pass()
        for p in plan:
            if FIRST_MOE <= p.layer_idx < FIRST_MOE + N_MOE:
                assert p.is_moe is True
            else:
                assert p.is_moe is False

    def test_non_moe_no_residual_correction(self):
        ss = _ss(residual_correction_enabled=True)
        plan = ss.plan_forward_pass()
        for p in plan:
            if not p.is_moe:
                assert p.apply_residual_correction is False
                assert p.store_gating_output is False

    def test_residual_correction_enabled_on_moe(self):
        ss = _ss(residual_correction_enabled=True)
        plan = ss.plan_forward_pass()
        moe_plans = [p for p in plan if p.is_moe]
        assert all(p.apply_residual_correction for p in moe_plans)

    def test_residual_correction_disabled(self):
        ss = _ss(residual_correction_enabled=False)
        plan = ss.plan_forward_pass()
        assert all(not p.apply_residual_correction for p in plan)

    def test_store_gating_output_on_moe(self):
        ss = _ss()
        plan = ss.plan_forward_pass()
        moe_plans = [p for p in plan if p.is_moe]
        assert all(p.store_gating_output for p in moe_plans)

    def test_layer_skip_set(self):
        ss = _ss()
        skip_set = {4, 6, 8}
        plan = ss.plan_forward_pass(layer_skip_set=skip_set)
        for p in plan:
            if p.layer_idx in skip_set:
                assert p.skip is True
            else:
                assert p.skip is False

    def test_no_skip_by_default(self):
        ss = _ss()
        plan = ss.plan_forward_pass()
        assert all(not p.skip for p in plan)


# ---------------------------------------------------------------------------
# Should continue
# ---------------------------------------------------------------------------


class TestShouldContinue:

    def test_first_step_always_continues(self):
        ss = _ss(adaptive_exit_enabled=True, draft_confidence_threshold=0.99)
        assert ss.should_continue(0, confidence=0.0) is True

    def test_high_confidence_continues(self):
        ss = _ss(draft_confidence_threshold=0.4)
        assert ss.should_continue(1, confidence=0.8) is True

    def test_low_confidence_stops(self):
        ss = _ss(draft_confidence_threshold=0.4)
        assert ss.should_continue(1, confidence=0.2) is False

    def test_at_threshold_continues(self):
        ss = _ss(draft_confidence_threshold=0.4)
        assert ss.should_continue(1, confidence=0.4) is True

    def test_adaptive_exit_disabled(self):
        ss = _ss(adaptive_exit_enabled=False, draft_confidence_threshold=0.99)
        assert ss.should_continue(1, confidence=0.01) is True
        assert ss.should_continue(4, confidence=0.0) is True

    def test_at_max_depth_stops(self):
        ss = _ss(max_depth=3)
        assert ss.should_continue(3, confidence=1.0) is False

    def test_beyond_max_depth_stops(self):
        ss = _ss(max_depth=2)
        assert ss.should_continue(5, confidence=1.0) is False


# ---------------------------------------------------------------------------
# Draft result
# ---------------------------------------------------------------------------


class TestDraftResult:

    def test_tokens_property(self):
        result = SelfSpecDraftResult(steps=[_step(100), _step(200)])
        assert result.tokens == [100, 200]

    def test_depth_property(self):
        result = SelfSpecDraftResult(steps=[_step(), _step(), _step()])
        assert result.depth == 3

    def test_empty_result(self):
        result = SelfSpecDraftResult()
        assert result.tokens == []
        assert result.depth == 0

    def test_source_field(self):
        assert SelfSpecDraftResult().source == "self_speculative"


# ---------------------------------------------------------------------------
# Acceptance rate tracking
# ---------------------------------------------------------------------------


class TestAcceptanceRate:

    def test_initial_rate_zero(self):
        ss = _ss()
        assert ss.acceptance_rate == 0.0

    def test_first_record_sets_rate(self):
        ss = _ss()
        result = SelfSpecDraftResult(steps=[_step(), _step()])
        ss.record_result(result, num_accepted=2)
        assert ss.acceptance_rate == pytest.approx(1.0)

    def test_ema_smoothing(self):
        ss = _ss(acceptance_ema_alpha=0.5)
        result = SelfSpecDraftResult(steps=[_step(), _step()])
        ss.record_result(result, num_accepted=2)
        assert ss.acceptance_rate == pytest.approx(1.0)
        ss.record_result(result, num_accepted=0)
        assert ss.acceptance_rate == pytest.approx(0.5)

    def test_empty_draft_no_update(self):
        ss = _ss()
        ss.record_result(SelfSpecDraftResult(steps=[_step()]), num_accepted=1)
        rate_before = ss.acceptance_rate
        ss.record_result(SelfSpecDraftResult(), num_accepted=0)
        assert ss.acceptance_rate == rate_before


# ---------------------------------------------------------------------------
# Is MoE layer
# ---------------------------------------------------------------------------


class TestIsMoeLayer:

    def test_moe_layer_true(self):
        ss = _ss()
        for layer in range(FIRST_MOE, FIRST_MOE + N_MOE):
            assert ss.is_moe_layer(layer) is True

    def test_non_moe_layer_false(self):
        ss = _ss()
        for layer in range(FIRST_MOE):
            assert ss.is_moe_layer(layer) is False

    def test_boundary(self):
        ss = _ss()
        assert ss.is_moe_layer(FIRST_MOE - 1) is False
        assert ss.is_moe_layer(FIRST_MOE) is True
        assert ss.is_moe_layer(FIRST_MOE + N_MOE - 1) is True
        assert ss.is_moe_layer(FIRST_MOE + N_MOE) is False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:

    def test_default_config(self):
        cfg = SelfSpeculativeConfig()
        assert cfg.enabled is True
        assert cfg.draft_expert_count == 1
        assert cfg.adaptive_exit_enabled is True
        assert cfg.draft_confidence_threshold == pytest.approx(0.4)
        assert cfg.max_depth == 5

    def test_frozen_config(self):
        cfg = SelfSpeculativeConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = False  # type: ignore[misc]

    def test_properties(self):
        ss = _ss(enabled=True, draft_expert_count=3, max_depth=7)
        assert ss.is_enabled is True
        assert ss.draft_expert_count == 3
        assert ss.max_depth == 7


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


class TestCounters:

    def test_total_drafts(self):
        ss = _ss()
        ss.record_result(SelfSpecDraftResult(steps=[_step()]), 1)
        ss.record_result(SelfSpecDraftResult(steps=[_step(), _step()]), 1)
        assert ss.total_drafts == 2

    def test_total_steps(self):
        ss = _ss()
        ss.record_result(SelfSpecDraftResult(steps=[_step()]), 1)
        ss.record_result(SelfSpecDraftResult(steps=[_step(), _step(), _step()]), 2)
        assert ss.total_steps == 4

    def test_total_accepted(self):
        ss = _ss()
        ss.record_result(SelfSpecDraftResult(steps=[_step(), _step()]), 1)
        ss.record_result(SelfSpecDraftResult(steps=[_step(), _step()]), 2)
        assert ss.total_accepted == 3
