"""Tests for orchestrator.prefetch_fuser — signal fusion into PrefetchPriority."""

import pytest

from orchestrator.types import (
    ExpertKey,
    PrefetchConfidence,
    PrefetchHint,
    PrefetchPriority,
    PrefetchSource,
)
from orchestrator.prefetch_fuser import FuserConfig, PrefetchFuser


def _hint(layer: int, expert: int, source: PrefetchSource,
          score: float, confidence: PrefetchConfidence = PrefetchConfidence.MEDIUM) -> PrefetchHint:
    return PrefetchHint(
        key=ExpertKey(layer, expert),
        target_layer=layer,
        confidence=confidence,
        source=source,
        score=score,
    )


# ── Fusion formula ────────────────────────────────────────────────────────


class TestFusionFormula:
    def test_single_prescope(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.0))
        hints = [_hint(5, 0, PrefetchSource.PRESCOPE, 0.8)]
        result = f.fuse(hints, [], [], [])
        assert len(result) == 1
        assert result[0].priority_score == pytest.approx(0.5 * 0.8)
        assert result[0].source_scores == {"prescope": 0.8}

    def test_two_sources_combine(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.0))
        ps = [_hint(5, 0, PrefetchSource.PRESCOPE, 0.8)]
        pr = [_hint(5, 0, PrefetchSource.PROBE, 0.6)]
        result = f.fuse(ps, pr, [], [])
        assert len(result) == 1
        expected = 0.5 * 0.8 + 0.3 * 0.6
        assert result[0].priority_score == pytest.approx(expected)

    def test_all_four_sources(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.0))
        ps = [_hint(5, 0, PrefetchSource.PRESCOPE, 1.0)]
        pr = [_hint(5, 0, PrefetchSource.PROBE, 1.0)]
        sq = [_hint(5, 0, PrefetchSource.MOE_SPEQ, 1.0)]
        sp = [_hint(5, 0, PrefetchSource.SP_MOE, 1.0)]
        result = f.fuse(ps, pr, sq, sp)
        expected = 0.5 + 0.3 + 0.2 + 0.15
        assert result[0].priority_score == pytest.approx(min(expected, 1.0))

    def test_missing_sources_contribute_zero(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.0))
        pr = [_hint(5, 0, PrefetchSource.PROBE, 0.9)]
        result = f.fuse([], pr, [], [])
        assert result[0].priority_score == pytest.approx(0.3 * 0.9)

    def test_max_score_per_source_when_duplicates(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.0))
        ps = [
            _hint(5, 0, PrefetchSource.PRESCOPE, 0.3),
            _hint(5, 0, PrefetchSource.PRESCOPE, 0.9),
        ]
        result = f.fuse(ps, [], [], [])
        assert result[0].priority_score == pytest.approx(0.5 * 0.9)

    def test_zero_weight_disables_source(self):
        f = PrefetchFuser(FuserConfig(prescope_alpha=0.0, time_decay=0.0))
        ps = [_hint(5, 0, PrefetchSource.PRESCOPE, 1.0)]
        result = f.fuse(ps, [], [], [])
        assert result[0].priority_score == pytest.approx(0.0)


# ── Time-aware priority ──────────────────────────────────────────────────


class TestTimeAwarePriority:
    def test_nearby_higher_than_distant(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.1))
        near = [_hint(6, 0, PrefetchSource.PROBE, 0.9)]
        far = [_hint(15, 1, PrefetchSource.PROBE, 0.9)]
        result = f.fuse([], near + far, [], [], current_layer=5)
        by_expert = {r.key.expert_idx: r.priority_score for r in result}
        assert by_expert[0] > by_expert[1]

    def test_zero_decay_disables(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.0))
        near = [_hint(6, 0, PrefetchSource.PROBE, 0.9)]
        far = [_hint(50, 1, PrefetchSource.PROBE, 0.9)]
        result = f.fuse([], near + far, [], [], current_layer=5)
        by_expert = {r.key.expert_idx: r.priority_score for r in result}
        assert by_expert[0] == pytest.approx(by_expert[1])

    def test_estimated_time_populated(self):
        f = PrefetchFuser(FuserConfig(per_layer_us=1000, time_decay=0.0))
        hints = [_hint(10, 0, PrefetchSource.PRESCOPE, 0.8)]
        result = f.fuse(hints, [], [], [], current_layer=5)
        assert result[0].estimated_time_until_needed_us == 5 * 1000


# ── Dominance patterns ────────────────────────────────────────────────────


class TestDominancePatterns:
    def test_prescope_dominates_at_1_layer(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.1))
        ps = [_hint(6, 0, PrefetchSource.PRESCOPE, 0.9)]
        pr = [_hint(6, 0, PrefetchSource.PROBE, 0.9)]
        result = f.fuse(ps, pr, [], [], current_layer=5)
        assert result[0].source_scores["prescope"] >= result[0].source_scores["probe"]

    def test_probe_dominates_at_distant_layers(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.0))
        pr = [_hint(50, 0, PrefetchSource.PROBE, 0.8)]
        result = f.fuse([], pr, [], [], current_layer=5)
        assert result[0].source_scores == {"probe": 0.8}
        assert result[0].priority_score == pytest.approx(0.3 * 0.8)

    def test_sp_moe_boosts_existing(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.0))
        pr_only = [_hint(5, 0, PrefetchSource.PROBE, 0.8)]
        result_without = f.fuse([], pr_only, [], [])
        sp = [_hint(5, 0, PrefetchSource.SP_MOE, 0.7)]
        result_with = f.fuse([], pr_only, [], sp)
        assert result_with[0].priority_score > result_without[0].priority_score


# ── Output properties ─────────────────────────────────────────────────────


class TestOutputProperties:
    def test_sorted_descending(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.0))
        hints = [
            _hint(5, 0, PrefetchSource.PRESCOPE, 0.3),
            _hint(5, 1, PrefetchSource.PRESCOPE, 0.9),
            _hint(5, 2, PrefetchSource.PRESCOPE, 0.6),
        ]
        result = f.fuse(hints, [], [], [])
        scores = [r.priority_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_scores_clamped(self):
        f = PrefetchFuser(FuserConfig(
            prescope_alpha=1.0, probe_beta=1.0,
            speq_gamma=1.0, sp_moe_delta=1.0, time_decay=0.0))
        ps = [_hint(5, 0, PrefetchSource.PRESCOPE, 1.0)]
        pr = [_hint(5, 0, PrefetchSource.PROBE, 1.0)]
        sq = [_hint(5, 0, PrefetchSource.MOE_SPEQ, 1.0)]
        sp = [_hint(5, 0, PrefetchSource.SP_MOE, 1.0)]
        result = f.fuse(ps, pr, sq, sp)
        assert result[0].priority_score <= 1.0

    def test_source_scores_populated(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.0))
        ps = [_hint(5, 0, PrefetchSource.PRESCOPE, 0.8)]
        pr = [_hint(5, 0, PrefetchSource.PROBE, 0.6)]
        result = f.fuse(ps, pr, [], [])
        assert "prescope" in result[0].source_scores
        assert "probe" in result[0].source_scores
        assert result[0].source_scores["prescope"] == pytest.approx(0.8)
        assert result[0].source_scores["probe"] == pytest.approx(0.6)

    def test_empty_inputs_empty_output(self):
        f = PrefetchFuser()
        assert f.fuse([], [], [], []) == []

    def test_different_experts_separate_entries(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.0))
        hints = [
            _hint(5, 0, PrefetchSource.PRESCOPE, 0.9),
            _hint(5, 1, PrefetchSource.PRESCOPE, 0.7),
        ]
        result = f.fuse(hints, [], [], [])
        assert len(result) == 2
        keys = {r.key for r in result}
        assert keys == {ExpertKey(5, 0), ExpertKey(5, 1)}


# ── Config ────────────────────────────────────────────────────────────────


class TestConfig:
    def test_default_matches_schema(self):
        f = PrefetchFuser()
        assert f.config.prescope_alpha == pytest.approx(0.5)
        assert f.config.probe_beta == pytest.approx(0.3)
        assert f.config.speq_gamma == pytest.approx(0.2)
        assert f.config.sp_moe_delta == pytest.approx(0.15)

    def test_custom_weights(self):
        cfg = FuserConfig(prescope_alpha=0.8, probe_beta=0.1, time_decay=0.0)
        f = PrefetchFuser(cfg)
        ps = [_hint(5, 0, PrefetchSource.PRESCOPE, 1.0)]
        pr = [_hint(5, 0, PrefetchSource.PROBE, 1.0)]
        result = f.fuse(ps, pr, [], [])
        expected = 0.8 * 1.0 + 0.1 * 1.0
        assert result[0].priority_score == pytest.approx(expected)

    def test_prefetch_priority_fields(self):
        f = PrefetchFuser(FuserConfig(time_decay=0.0))
        ps = [_hint(5, 3, PrefetchSource.PRESCOPE, 0.7)]
        result = f.fuse(ps, [], [], [])
        p = result[0]
        assert p.key == ExpertKey(5, 3)
        assert p.target_layer == 5
        assert isinstance(p.priority_score, float)
        assert isinstance(p.estimated_time_until_needed_us, int)
        assert isinstance(p.source_scores, dict)
