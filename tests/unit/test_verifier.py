"""Tests for verifier — unified MoE-Spec masked verification."""

import pytest
import numpy as np

from orchestrator.draft_combiner import CombinedDraft, CombinedDraftStep
from orchestrator.types import ExpertKey
from orchestrator.verifier import (
    CommandDescriptor,
    MoeAnalysis,
    VerificationCommandPlan,
    VerificationResult,
    Verifier,
    VerifierConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verifier(
    num_layers: int = 7,
    num_moe_layers: int = 4,
    first_moe_layer: int = 3,
    num_experts: int = 8,
    **config_kw,
) -> Verifier:
    return Verifier(
        config=VerifierConfig(**config_kw) if config_kw else None,
        num_layers=num_layers,
        num_moe_layers=num_moe_layers,
        first_moe_layer=first_moe_layer,
        num_experts=num_experts,
    )


def _gating(
    depth: int,
    num_moe_layers: int,
    num_experts: int,
    dominant: int = 0,
    dominant_weight: float = 0.95,
) -> np.ndarray:
    g = np.full((depth, num_moe_layers, num_experts), 0.01, dtype=np.float32)
    remainder = (1.0 - dominant_weight) / max(num_experts - 1, 1)
    g[:, :, :] = remainder
    g[:, :, dominant] = dominant_weight
    return g


def _gating_spread(
    depth: int,
    num_moe_layers: int,
    num_experts: int,
    top_k: int = 4,
) -> np.ndarray:
    g = np.zeros((depth, num_moe_layers, num_experts), dtype=np.float32)
    per_expert = 1.0 / top_k
    for e in range(top_k):
        g[:, :, e] = per_expert
    return g


def _resident(*keys: ExpertKey, gpu: int = 0) -> dict[int, set[ExpertKey]]:
    return {gpu: set(keys)}


def _resident_multi(
    mapping: dict[int, list[ExpertKey]],
) -> dict[int, set[ExpertKey]]:
    return {gpu: set(keys) for gpu, keys in mapping.items()}


def _draft(tokens: list[int]) -> CombinedDraft:
    return CombinedDraft(steps=[
        CombinedDraftStep(
            token_id=t, position=i, source="mtp", p_accept=0.8, utility=1.0,
        )
        for i, t in enumerate(tokens)
    ])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:

    def test_defaults(self):
        c = VerifierConfig()
        assert c.adaptive_topk_threshold == 0.92
        assert c.verification_quality_floor == 0.85
        assert c.in_flight_transfer_mode == "conservative"
        assert c.substitution_policy == "substitution"
        assert c.max_draft_tree_size == 64
        assert c.acceptance_ema_alpha == 0.3

    def test_frozen(self):
        c = VerifierConfig()
        with pytest.raises(AttributeError):
            c.adaptive_topk_threshold = 0.5  # type: ignore[misc]

    def test_custom_values(self):
        c = VerifierConfig(
            adaptive_topk_threshold=0.80,
            verification_quality_floor=0.70,
            max_draft_tree_size=32,
        )
        assert c.adaptive_topk_threshold == 0.80
        assert c.verification_quality_floor == 0.70
        assert c.max_draft_tree_size == 32

    def test_modes(self):
        c = VerifierConfig(in_flight_transfer_mode="optimistic")
        assert c.in_flight_transfer_mode == "optimistic"
        v = Verifier(config=c)
        assert v.config.in_flight_transfer_mode == "optimistic"

    def test_substitution_policy(self):
        c = VerifierConfig(substitution_policy="discard")
        assert c.substitution_policy == "discard"


# ---------------------------------------------------------------------------
# MoE Coverage Analysis
# ---------------------------------------------------------------------------

class TestMoeCoverageAnalysis:

    def test_all_resident_full_coverage(self):
        v = _verifier()
        draft = _draft([10, 20])
        g = _gating(2, 4, 8, dominant=0, dominant_weight=0.95)
        res = _resident(
            *[ExpertKey(layer, 0) for layer in range(3, 7)],
        )
        analyses = v.analyze_moe_coverage(draft, g, res)
        assert len(analyses) == 2
        for a in analyses:
            assert a.min_layer_coverage == pytest.approx(1.0)
            assert all(m == 0 for m in a.per_layer_moe_mode)
            assert not a.quality_floor_violated
            assert not a.truncated

    def test_none_resident_zero_coverage(self):
        v = _verifier()
        draft = _draft([10])
        g = _gating(1, 4, 8, dominant=0, dominant_weight=0.95)
        analyses = v.analyze_moe_coverage(draft, g, {})
        assert len(analyses) == 1
        a = analyses[0]
        assert a.min_layer_coverage == pytest.approx(0.0)
        assert a.quality_floor_violated
        assert a.truncated

    def test_partial_coverage_above_threshold(self):
        v = _verifier(adaptive_topk_threshold=0.90)
        draft = _draft([10])
        g = np.zeros((1, 4, 8), dtype=np.float32)
        g[0, :, 0] = 0.70
        g[0, :, 1] = 0.25
        g[0, :, 2] = 0.05
        res = _resident(
            *[ExpertKey(layer, 0) for layer in range(3, 7)],
            *[ExpertKey(layer, 1) for layer in range(3, 7)],
        )
        analyses = v.analyze_moe_coverage(draft, g, res)
        a = analyses[0]
        assert a.min_layer_coverage == pytest.approx(0.95 / 0.95, abs=0.02)
        assert all(m == 0 for m in a.per_layer_moe_mode)
        assert not a.quality_floor_violated

    def test_below_quality_floor(self):
        v = _verifier(verification_quality_floor=0.85)
        draft = _draft([10])
        g = np.zeros((1, 4, 8), dtype=np.float32)
        g[0, :, 0] = 0.50
        g[0, :, 1] = 0.30
        g[0, :, 2] = 0.20
        res = _resident(
            *[ExpertKey(layer, 0) for layer in range(3, 7)],
        )
        analyses = v.analyze_moe_coverage(draft, g, res)
        a = analyses[0]
        assert a.min_layer_coverage < 0.85
        assert a.quality_floor_violated
        assert a.truncated
        assert "quality_floor_violated" in a.truncation_reason

    def test_per_layer_different_coverage(self):
        v = _verifier()
        draft = _draft([10])
        g = np.zeros((1, 4, 8), dtype=np.float32)
        g[0, :, 0] = 0.95
        g[0, 2, 0] = 0.50
        g[0, 2, 1] = 0.45
        g[0, 2, 2] = 0.05
        res = _resident(
            *[ExpertKey(layer, 0) for layer in range(3, 7)],
        )
        analyses = v.analyze_moe_coverage(draft, g, res)
        a = analyses[0]
        assert a.per_layer_coverage[0] == pytest.approx(1.0)
        assert a.per_layer_coverage[2] < 1.0
        assert a.min_layer_coverage == min(a.per_layer_coverage)

    def test_discard_policy_sets_truncation(self):
        v = _verifier(substitution_policy="discard",
                      verification_quality_floor=0.85)
        draft = _draft([10])
        g = np.zeros((1, 4, 8), dtype=np.float32)
        g[0, :, 0] = 0.90
        g[0, :, 1] = 0.10
        res = _resident(
            *[ExpertKey(layer, 0) for layer in range(3, 7)],
        )
        analyses = v.analyze_moe_coverage(draft, g, res)
        a = analyses[0]
        assert a.min_layer_coverage == pytest.approx(0.9)
        assert any(m == 1 for m in a.per_layer_moe_mode)
        assert not a.quality_floor_violated
        assert a.truncated
        assert "discard_policy" in a.truncation_reason

    def test_multiple_positions(self):
        v = _verifier()
        draft = _draft([10, 20, 30])
        g = _gating(3, 4, 8, dominant=0, dominant_weight=0.95)
        res = _resident(
            *[ExpertKey(layer, 0) for layer in range(3, 7)],
        )
        analyses = v.analyze_moe_coverage(draft, g, res)
        assert len(analyses) == 3
        for a in analyses:
            assert not a.truncated

    def test_empty_draft(self):
        v = _verifier()
        draft = CombinedDraft()
        g = np.array([], dtype=np.float32)
        analyses = v.analyze_moe_coverage(draft, g, {})
        assert analyses == []


# ---------------------------------------------------------------------------
# Verifiable Depth
# ---------------------------------------------------------------------------

class TestVerifiableDepth:

    def _make_analyses(self, n: int, truncated_at: int | None = None,
                       violated_at: int | None = None) -> list[MoeAnalysis]:
        analyses = []
        for i in range(n):
            trunc = (truncated_at is not None and i >= truncated_at)
            viol = (violated_at is not None and i >= violated_at)
            analyses.append(MoeAnalysis(
                position=i,
                min_layer_coverage=0.5 if (trunc or viol) else 1.0,
                quality_floor_violated=viol,
                truncated=trunc or viol,
                truncation_reason="test" if (trunc or viol) else None,
            ))
        return analyses

    def test_full_depth_when_all_pass(self):
        v = _verifier()
        analyses = self._make_analyses(5)
        assert v.compute_verifiable_depth(analyses, max_depth=10) == 5

    def test_truncates_at_first_violation(self):
        v = _verifier()
        analyses = self._make_analyses(5, violated_at=2)
        assert v.compute_verifiable_depth(analyses, max_depth=10) == 2

    def test_sequential_truncation(self):
        v = _verifier()
        analyses = self._make_analyses(5, truncated_at=3)
        assert v.compute_verifiable_depth(analyses, max_depth=10) == 3

    def test_max_depth_ceiling(self):
        v = _verifier()
        analyses = self._make_analyses(10)
        assert v.compute_verifiable_depth(analyses, max_depth=4) == 4

    def test_max_draft_tree_size_cap(self):
        v = _verifier(max_draft_tree_size=3)
        analyses = self._make_analyses(10)
        assert v.compute_verifiable_depth(analyses, max_depth=100) == 3

    def test_depth_zero_for_immediate_failure(self):
        v = _verifier()
        analyses = self._make_analyses(5, violated_at=0)
        assert v.compute_verifiable_depth(analyses, max_depth=10) == 0

    def test_empty_analyses(self):
        v = _verifier()
        assert v.compute_verifiable_depth([], max_depth=10) == 0


# ---------------------------------------------------------------------------
# Command Plan
# ---------------------------------------------------------------------------

class TestCommandPlan:

    def test_plan_structure(self):
        v = _verifier(num_layers=7, num_moe_layers=4, first_moe_layer=3)
        analyses = [MoeAnalysis(position=i, min_layer_coverage=1.0,
                                per_layer_moe_mode=[0, 0, 0, 0])
                    for i in range(3)]
        plan = v.plan_verification_pass(3, analyses)
        assert plan.depth == 3
        assert plan.num_layers == 7
        assert plan.commands[0].cmd_type == "EMBEDDING_LOOKUP"
        assert plan.commands[-1].cmd_type == "OUTPUT_HEAD"
        attn_count = sum(1 for c in plan.commands if c.cmd_type == "RUN_ATTENTION")
        moe_count = sum(1 for c in plan.commands if c.cmd_type == "RUN_MOE")
        assert attn_count == 7
        assert moe_count == 7  # TD-59b: RUN_MOE emitted for all layers (dense + MoE)

    def test_moe_mode_per_layer(self):
        v = _verifier(num_layers=7, num_moe_layers=4, first_moe_layer=3)
        analyses = [MoeAnalysis(position=0, min_layer_coverage=0.9,
                                per_layer_moe_mode=[0, 2, 0, 2])]
        plan = v.plan_verification_pass(1, analyses)
        moe_cmds = [c for c in plan.commands if c.cmd_type == "RUN_MOE"]
        assert len(moe_cmds) == 7  # TD-59b: all layers
        # Dense layers 0-2: moe_mode=0
        for i in range(3):
            assert moe_cmds[i].moe_mode == 0
        # MoE layers 3-6: per-layer modes from analysis
        assert moe_cmds[3].moe_mode == 0
        assert moe_cmds[4].moe_mode == 2
        assert moe_cmds[5].moe_mode == 0
        assert moe_cmds[6].moe_mode == 2

    def test_dense_layers_get_moe_mode_zero(self):
        """TD-59b: dense layers emit RUN_MOE with moe_mode=0, no checkpoint."""
        v = _verifier(num_layers=7, num_moe_layers=4, first_moe_layer=3)
        analyses = [MoeAnalysis(position=0, min_layer_coverage=1.0,
                                per_layer_moe_mode=[0, 0, 0, 0])]
        plan = v.plan_verification_pass(1, analyses)
        # Dense layers 0-2 should have RUN_MOE with moe_mode=0
        for layer in range(3):
            moe_cmds = [c for c in plan.commands
                        if c.layer_idx == layer and c.cmd_type == "RUN_MOE"]
            assert len(moe_cmds) == 1
            assert moe_cmds[0].moe_mode == 0
            assert moe_cmds[0].emit_checkpoint == 0

    def test_depth_propagated(self):
        v = _verifier()
        analyses = [MoeAnalysis(position=0, min_layer_coverage=1.0,
                                per_layer_moe_mode=[0, 0, 0, 0])]
        plan = v.plan_verification_pass(5, analyses)
        for c in plan.commands:
            assert c.num_seqs == 5

    def test_readback_flags(self):
        v = _verifier()
        analyses = [MoeAnalysis(position=0, min_layer_coverage=1.0,
                                per_layer_moe_mode=[0, 0, 0, 0])]
        plan = v.plan_verification_pass(1, analyses)
        out = plan.commands[-1]
        assert out.cmd_type == "OUTPUT_HEAD"
        assert out.readback_to_host == 1
        assert out.compute_confidence == 1

    def test_worst_case_moe_mode_across_positions(self):
        v = _verifier(num_layers=7, num_moe_layers=4, first_moe_layer=3)
        a0 = MoeAnalysis(position=0, min_layer_coverage=1.0,
                         per_layer_moe_mode=[0, 0, 0, 0])
        a1 = MoeAnalysis(position=1, min_layer_coverage=0.9,
                         per_layer_moe_mode=[0, 2, 0, 1])
        plan = v.plan_verification_pass(2, [a0, a1])
        moe_cmds = [c for c in plan.commands if c.cmd_type == "RUN_MOE"]
        assert len(moe_cmds) == 7  # TD-59b: all layers
        # Dense layers 0-2: moe_mode=0
        for i in range(3):
            assert moe_cmds[i].moe_mode == 0
        # MoE layers 3-6: worst-case across positions
        assert moe_cmds[3].moe_mode == 0
        assert moe_cmds[4].moe_mode == 2
        assert moe_cmds[5].moe_mode == 0
        assert moe_cmds[6].moe_mode == 1

    def test_zero_depth_empty_plan(self):
        v = _verifier()
        plan = v.plan_verification_pass(0, [])
        assert plan.commands == []
        assert plan.depth == 0


# ---------------------------------------------------------------------------
# Logit Comparison
# ---------------------------------------------------------------------------

class TestLogitComparison:

    def test_perfect_match(self):
        assert Verifier.compare_logits([1, 2, 3], [1, 2, 3]) == 3

    def test_first_mismatch(self):
        assert Verifier.compare_logits([1, 2, 3], [9, 2, 3]) == 0

    def test_partial_match(self):
        assert Verifier.compare_logits([1, 2, 3, 4, 5], [1, 2, 3, 9, 5]) == 3

    def test_empty_sequences(self):
        assert Verifier.compare_logits([], []) == 0
        assert Verifier.compare_logits([1], []) == 0
        assert Verifier.compare_logits([], [1]) == 0

    def test_different_lengths(self):
        assert Verifier.compare_logits([1, 2], [1, 2, 3, 4]) == 2
        assert Verifier.compare_logits([1, 2, 3, 4], [1, 2]) == 2


# ---------------------------------------------------------------------------
# Build Result
# ---------------------------------------------------------------------------

class TestBuildResult:

    def test_full_acceptance(self):
        v = _verifier()
        draft = _draft([10, 20, 30])
        analyses = [MoeAnalysis(position=i, min_layer_coverage=1.0,
                                per_layer_moe_mode=[0, 0, 0, 0])
                    for i in range(3)]
        result = v.build_result(draft, analyses, verified_depth=3,
                                accepted_length=3, seq_id=42)
        assert result.accepted_length == 3
        assert result.attempted_length == 3
        assert result.accepted_tokens == [10, 20, 30]
        assert result.rejected_positions == []
        assert result.pages_to_promote == [(42, 0), (42, 1), (42, 2)]
        assert result.pages_to_free == []

    def test_partial_acceptance(self):
        v = _verifier()
        draft = _draft([10, 20, 30, 40, 50])
        analyses = [MoeAnalysis(position=i, min_layer_coverage=1.0,
                                per_layer_moe_mode=[0, 0, 0, 0])
                    for i in range(5)]
        result = v.build_result(draft, analyses, verified_depth=5,
                                accepted_length=3, seq_id=7)
        assert result.accepted_length == 3
        assert result.accepted_tokens == [10, 20, 30]
        assert result.rejected_positions == [3, 4]
        assert result.pages_to_promote == [(7, 0), (7, 1), (7, 2)]
        assert result.pages_to_free == [(7, 3), (7, 4)]

    def test_zero_acceptance(self):
        v = _verifier()
        draft = _draft([10, 20])
        analyses = [MoeAnalysis(position=i, min_layer_coverage=1.0,
                                per_layer_moe_mode=[0, 0, 0, 0])
                    for i in range(2)]
        result = v.build_result(draft, analyses, verified_depth=2,
                                accepted_length=0, seq_id=1)
        assert result.accepted_tokens == []
        assert result.rejected_positions == [0, 1]
        assert result.pages_to_promote == []
        assert result.pages_to_free == [(1, 0), (1, 1)]

    def test_quality_floor_violations_reported(self):
        v = _verifier()
        draft = _draft([10, 20, 30])
        analyses = [
            MoeAnalysis(position=0, min_layer_coverage=1.0),
            MoeAnalysis(position=1, min_layer_coverage=0.5,
                        quality_floor_violated=True, truncated=True,
                        truncation_reason="quality_floor_violated_at_layer_4"),
            MoeAnalysis(position=2, min_layer_coverage=0.3,
                        quality_floor_violated=True, truncated=True),
        ]
        result = v.build_result(draft, analyses, verified_depth=1,
                                accepted_length=1, seq_id=5)
        assert result.quality_floor_violations == [1, 2]

    def test_truncation_reason(self):
        v = _verifier()
        draft = _draft([10, 20, 30])
        analyses = [
            MoeAnalysis(position=0, min_layer_coverage=1.0),
            MoeAnalysis(position=1, min_layer_coverage=0.5,
                        truncated=True,
                        truncation_reason="quality_floor_violated_at_layer_5"),
            MoeAnalysis(position=2, min_layer_coverage=0.3, truncated=True),
        ]
        result = v.build_result(draft, analyses, verified_depth=1,
                                accepted_length=1, seq_id=5)
        assert result.truncation_reason == "quality_floor_violated_at_layer_5"

    def test_per_position_coverage_and_mode(self):
        v = _verifier()
        draft = _draft([10, 20])
        analyses = [
            MoeAnalysis(position=0, min_layer_coverage=1.0,
                        per_layer_moe_mode=[0, 0, 2, 0]),
            MoeAnalysis(position=1, min_layer_coverage=0.88,
                        per_layer_moe_mode=[0, 2, 2, 0]),
        ]
        result = v.build_result(draft, analyses, verified_depth=2,
                                accepted_length=2, seq_id=1)
        assert result.per_position_moe_mode == [2, 2]
        assert result.per_position_coverage == [1.0, 0.88]


# ---------------------------------------------------------------------------
# Verify (convenience method)
# ---------------------------------------------------------------------------

class TestVerifyConvenience:

    def test_end_to_end(self):
        v = _verifier()
        draft = _draft([10, 20, 30])
        g = _gating(3, 4, 8, dominant=0, dominant_weight=0.95)
        res = _resident(
            *[ExpertKey(layer, 0) for layer in range(3, 7)],
        )
        analyses, depth, plan = v.verify(draft, g, res, max_verifiable_depth=10)
        assert len(analyses) == 3
        assert depth == 3
        assert plan.depth == 3
        assert len(plan.commands) > 0

    def test_realistic_deepseek_v3(self):
        v = Verifier(
            num_layers=61, num_moe_layers=58,
            first_moe_layer=3, num_experts=256,
        )
        draft = _draft([100, 200, 300])
        g = _gating(3, 58, 256, dominant=0, dominant_weight=0.95)
        res = _resident(
            *[ExpertKey(layer, 0) for layer in range(3, 61)],
        )
        analyses, depth, plan = v.verify(draft, g, res, max_verifiable_depth=5)
        assert depth == 3
        attn_count = sum(1 for c in plan.commands
                         if c.cmd_type == "RUN_ATTENTION")
        moe_count = sum(1 for c in plan.commands if c.cmd_type == "RUN_MOE")
        assert attn_count == 61
        assert moe_count == 61  # TD-59b: all layers get RUN_MOE

    def test_empty_gating(self):
        v = _verifier()
        draft = _draft([10])
        g = np.array([], dtype=np.float32)
        analyses, depth, plan = v.verify(draft, g, {}, max_verifiable_depth=10)
        assert analyses == []
        assert depth == 0
        assert plan.depth == 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:

    def test_initial_values(self):
        v = _verifier()
        assert v.total_verifications == 0
        assert v.total_accepted == 0
        assert v.total_attempted == 0
        assert v.acceptance_rate == 0.0
        assert v.quality_floor_violation_rate == 0.0
        assert v.truncation_count == 0

    def test_record_updates_counters(self):
        v = _verifier()
        result = VerificationResult(
            accepted_length=3, attempted_length=5,
            accepted_tokens=[10, 20, 30],
            rejected_positions=[3, 4],
            quality_floor_violations=[4],
            truncation_reason="test_reason",
        )
        v.record_result(result)
        assert v.total_verifications == 1
        assert v.total_accepted == 3
        assert v.total_attempted == 5
        assert v.truncation_count == 1

    def test_ema_acceptance_rate(self):
        v = _verifier(acceptance_ema_alpha=0.5)
        r1 = VerificationResult(accepted_length=4, attempted_length=4)
        v.record_result(r1)
        assert v.acceptance_rate == pytest.approx(1.0)

        r2 = VerificationResult(accepted_length=0, attempted_length=4)
        v.record_result(r2)
        assert v.acceptance_rate == pytest.approx(0.5)

    def test_quality_floor_violation_rate(self):
        v = _verifier()
        r1 = VerificationResult(
            accepted_length=2, attempted_length=5,
            quality_floor_violations=[2, 3, 4],
        )
        v.record_result(r1)
        assert v.quality_floor_violation_rate == pytest.approx(3 / 5)

        r2 = VerificationResult(
            accepted_length=5, attempted_length=5,
            quality_floor_violations=[],
        )
        v.record_result(r2)
        assert v.quality_floor_violation_rate == pytest.approx(3 / 10)
