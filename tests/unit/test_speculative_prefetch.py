"""Tests for SP-MoE speculative prefetch — verification transfer planning."""

from __future__ import annotations

import numpy as np
import pytest

from orchestrator.speculative_prefetch import (
    SpeculativePrefetch,
    SpeculativePrefetchConfig,
    _adaptive_topk,
)
from orchestrator.types import (
    CacheZone,
    ExpertKey,
    PrefetchConfidence,
    PrefetchSource,
    TransferPlanEntry,
    VerificationPlan,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_EXPERT_BYTES = 2_359_296
_DEFAULT_NUM_GPUS = 4


def _sp(
    config: SpeculativePrefetchConfig | None = None,
    num_gpus: int = _DEFAULT_NUM_GPUS,
    expert_bytes: int = _DEFAULT_EXPERT_BYTES,
) -> SpeculativePrefetch:
    return SpeculativePrefetch(config=config, num_gpus=num_gpus,
                               expert_bytes=expert_bytes)


def _gating(
    depth: int,
    num_moe_layers: int,
    num_experts: int,
    dominant: int | None = 0,
    dominant_weight: float = 0.95,
) -> np.ndarray:
    g = np.full((depth, num_moe_layers, num_experts), 0.001, dtype=np.float32)
    if dominant is not None:
        g[:, :, dominant] = dominant_weight
    return g


def _uniform_gating(depth: int, num_moe_layers: int, num_experts: int) -> np.ndarray:
    w = 1.0 / num_experts
    return np.full((depth, num_moe_layers, num_experts), w, dtype=np.float32)


def _resident(*keys: ExpertKey, gpu: int = 0) -> dict[int, set[ExpertKey]]:
    return {gpu: set(keys)}


def _resident_multi(mapping: dict[int, list[ExpertKey]]) -> dict[int, set[ExpertKey]]:
    return {gpu: set(keys) for gpu, keys in mapping.items()}


# ---------------------------------------------------------------------------
# TestAdaptiveTopK
# ---------------------------------------------------------------------------

class TestAdaptiveTopK:

    def test_single_expert_above_threshold(self):
        w = np.array([0.95, 0.03, 0.02], dtype=np.float32)
        assert _adaptive_topk(w, 0.92) == [0]

    def test_multiple_experts_needed(self):
        w = np.array([0.50, 0.30, 0.15, 0.05], dtype=np.float32)
        result = _adaptive_topk(w, 0.92)
        assert result == [0, 1, 2]

    def test_all_zeros_returns_all(self):
        w = np.zeros(8, dtype=np.float32)
        result = _adaptive_topk(w, 0.92)
        assert len(result) == 8  # cumulative stays 0, never reaches threshold

    def test_exact_threshold_boundary(self):
        w = np.array([0.50, 0.42, 0.08], dtype=np.float32)
        result = _adaptive_topk(w, 0.92)
        assert 0 in result and 1 in result
        cumulative = sum(float(w[i]) for i in result)
        assert cumulative >= 0.92

    def test_always_at_least_one(self):
        w = np.array([0.01, 0.01], dtype=np.float32)
        result = _adaptive_topk(w, 0.92)
        assert len(result) >= 1

    def test_uniform_weights(self):
        w = np.full(256, 1.0 / 256, dtype=np.float32)
        result = _adaptive_topk(w, 0.92)
        cumulative = sum(float(w[i]) for i in result)
        assert cumulative >= 0.92
        assert len(result) >= 236

    def test_single_nonzero(self):
        w = np.zeros(10, dtype=np.float32)
        w[7] = 1.0
        result = _adaptive_topk(w, 0.92)
        assert result == [7]


# ---------------------------------------------------------------------------
# TestExpertUnion
# ---------------------------------------------------------------------------

class TestExpertUnion:

    def test_deduplication_across_positions(self):
        g = np.zeros((2, 1, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        g[1, 0, 0] = 0.95
        sp = _sp()
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        keys = [e.key for e in plan.transfers]
        assert keys.count(ExpertKey(3, 0)) == 1

    def test_layer_grouping(self):
        g = np.zeros((1, 2, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        g[0, 1, 1] = 0.95
        sp = _sp()
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        keys = {e.key for e in plan.transfers}
        assert ExpertKey(3, 0) in keys
        assert ExpertKey(4, 1) in keys

    def test_union_grows_with_depth(self):
        g1 = np.zeros((1, 1, 4), dtype=np.float32)
        g1[0, 0, 0] = 0.95
        g2 = np.zeros((2, 1, 4), dtype=np.float32)
        g2[0, 0, 0] = 0.95
        g2[1, 0, 1] = 0.95
        sp = _sp()
        plan1 = sp.compute_verification_plan(g1, {}, first_moe_layer=3)
        plan2 = sp.compute_verification_plan(g2, {}, first_moe_layer=3)
        assert len(plan2.transfers) >= len(plan1.transfers)

    def test_overlapping_positions_deduplicate(self):
        g = np.zeros((2, 1, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        g[0, 0, 1] = 0.00
        g[1, 0, 0] = 0.95
        g[1, 0, 2] = 0.95
        sp = _sp(config=SpeculativePrefetchConfig(adaptive_topk_threshold=0.50))
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        keys = [e.key for e in plan.transfers]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# TestTransferSchedule
# ---------------------------------------------------------------------------

class TestTransferSchedule:

    def test_ordered_by_layer_then_position(self):
        g = np.zeros((2, 2, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95  # layer 3, pos 0
        g[0, 1, 1] = 0.95  # layer 4, pos 0
        g[1, 0, 2] = 0.95  # layer 3, pos 1
        g[1, 1, 3] = 0.95  # layer 4, pos 1
        sp = _sp()
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        layers = [e.key.layer_idx for e in plan.transfers]
        assert layers == sorted(layers)

    def test_resident_experts_filtered(self):
        g = np.zeros((1, 1, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        resident = _resident(ExpertKey(3, 0))
        sp = _sp()
        plan = sp.compute_verification_plan(g, resident, first_moe_layer=3)
        assert len(plan.transfers) == 0

    def test_all_resident_empty_transfers(self):
        g = _gating(2, 2, 4, dominant=0)
        resident = _resident(ExpertKey(3, 0), ExpertKey(4, 0))
        sp = _sp()
        plan = sp.compute_verification_plan(g, resident, first_moe_layer=3)
        assert len(plan.transfers) == 0

    def test_affinity_respected(self):
        g = np.zeros((1, 1, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        affinity = {ExpertKey(3, 0): 2}
        sp = _sp()
        plan = sp.compute_verification_plan(g, {}, affinity=affinity,
                                            first_moe_layer=3)
        assert plan.transfers[0].target_gpu == 2

    def test_round_robin_without_affinity(self):
        g = np.zeros((1, 8, 4), dtype=np.float32)
        for m in range(8):
            g[0, m, m % 4] = 0.95
        sp = _sp(num_gpus=4)
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        gpus = [e.target_gpu for e in plan.transfers]
        assert set(gpus) == {0, 1, 2, 3}

    def test_priority_descending(self):
        g = np.zeros((1, 3, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        g[0, 1, 1] = 0.95
        g[0, 2, 2] = 0.95
        sp = _sp()
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        priorities = [e.priority for e in plan.transfers]
        assert priorities == sorted(priorities, reverse=True)

    def test_max_transfers_cap(self):
        cfg = SpeculativePrefetchConfig(
            adaptive_topk_threshold=0.01,
            max_transfers_per_plan=5,
        )
        g = np.zeros((1, 1, 256), dtype=np.float32)
        for i in range(256):
            g[0, 0, i] = 0.01
        sp = _sp(config=cfg)
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        assert len(plan.transfers) <= 5

    def test_expert_bytes_populated(self):
        g = np.zeros((1, 1, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        sp = _sp(expert_bytes=12345)
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        assert plan.transfers[0].expert_bytes == 12345

    def test_zone_is_streaming(self):
        g = np.zeros((1, 1, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        sp = _sp()
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        for entry in plan.transfers:
            assert entry.zone == CacheZone.STREAMING


# ---------------------------------------------------------------------------
# TestMaxDepthCalculation
# ---------------------------------------------------------------------------

class TestMaxDepthCalculation:

    def test_all_resident_full_depth(self):
        g = _gating(3, 2, 4, dominant=0)
        resident = _resident(ExpertKey(3, 0), ExpertKey(4, 0))
        sp = _sp()
        plan = sp.compute_verification_plan(g, resident, first_moe_layer=3)
        assert plan.max_depth == 2  # depth - 1

    def test_nothing_resident_max_depth_zero(self):
        g = _gating(3, 2, 4, dominant=0)
        sp = _sp()
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        assert plan.max_depth == 0

    def test_partial_coverage(self):
        g = np.zeros((3, 1, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        g[1, 0, 0] = 0.95
        g[2, 0, 1] = 0.95
        resident = _resident(ExpertKey(3, 0))
        sp = _sp()
        plan = sp.compute_verification_plan(g, resident, first_moe_layer=3)
        assert plan.max_depth >= 1

    def test_single_missing_at_position0(self):
        g = np.zeros((2, 1, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        g[1, 0, 0] = 0.95
        sp = _sp()
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        assert plan.max_depth == 0

    def test_budget_calculation_matches_formula(self):
        cfg = SpeculativePrefetchConfig(
            per_layer_us=1000,
            pcie_bw_bytes_per_us=10_000.0,
        )
        expert_bytes = 100_000
        g = np.zeros((3, 2, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        g[0, 1, 0] = 0.95
        g[1, 0, 0] = 0.95
        g[1, 1, 1] = 0.95
        g[2, 0, 2] = 0.95
        g[2, 1, 2] = 0.95
        resident = _resident(ExpertKey(3, 0), ExpertKey(4, 0))
        sp = _sp(config=cfg, expert_bytes=expert_bytes)
        plan = sp.compute_verification_plan(g, resident, first_moe_layer=3)
        # pos 0: all resident → 0 bytes, budget=0 → ok (max_depth>=0)
        # pos 1: expert (4,1) non-resident → 100KB, transfer=10us, budget=1*2*1000=2000us → ok
        # pos 2: expert (3,2),(4,2) → +200KB cum=300KB, transfer=30us, budget=2*2*1000=4000us → ok
        assert plan.max_depth == 2

    def test_depth_one_gets_budget(self):
        cfg = SpeculativePrefetchConfig(
            per_layer_us=1000,
            pcie_bw_bytes_per_us=1.0,
        )
        g = np.zeros((2, 1, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        g[1, 0, 1] = 0.95
        resident = _resident(ExpertKey(3, 0))
        sp = _sp(config=cfg, expert_bytes=500)
        plan = sp.compute_verification_plan(g, resident, first_moe_layer=3)
        # pos 0: resident → ok
        # pos 1: 500 bytes, transfer=500us, budget=1*1*1000=1000us → ok
        assert plan.max_depth == 1

    def test_zero_depth_gating(self):
        g = np.zeros((0, 2, 4), dtype=np.float32)
        sp = _sp()
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        assert plan.max_depth == 0


# ---------------------------------------------------------------------------
# TestVerificationPlan (end-to-end)
# ---------------------------------------------------------------------------

class TestVerificationPlan:

    def test_full_plan_all_resident(self):
        g = _gating(3, 2, 4, dominant=0)
        resident = _resident(ExpertKey(3, 0), ExpertKey(4, 0))
        sp = _sp()
        plan = sp.compute_verification_plan(g, resident, first_moe_layer=3)
        assert plan.transfers == []
        assert plan.max_depth == 2

    def test_full_plan_none_resident(self):
        g = _gating(2, 2, 4, dominant=0)
        sp = _sp()
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        assert len(plan.transfers) == 2  # ExpertKey(3,0) and ExpertKey(4,0)
        assert plan.max_depth >= 0

    def test_plan_with_affinity(self):
        g = np.zeros((1, 2, 4), dtype=np.float32)
        g[0, 0, 0] = 0.95
        g[0, 1, 1] = 0.95
        affinity = {ExpertKey(3, 0): 1, ExpertKey(4, 1): 3}
        sp = _sp()
        plan = sp.compute_verification_plan(g, {}, affinity=affinity,
                                            first_moe_layer=3)
        by_key = {e.key: e for e in plan.transfers}
        assert by_key[ExpertKey(3, 0)].target_gpu == 1
        assert by_key[ExpertKey(4, 1)].target_gpu == 3

    def test_plan_with_mixed_residency(self):
        g = np.zeros((2, 2, 4), dtype=np.float32)
        g[:, 0, 0] = 0.95
        g[:, 1, 1] = 0.95
        resident = _resident(ExpertKey(3, 0))
        sp = _sp()
        plan = sp.compute_verification_plan(g, resident, first_moe_layer=3)
        transfer_keys = {e.key for e in plan.transfers}
        assert ExpertKey(3, 0) not in transfer_keys
        assert ExpertKey(4, 1) in transfer_keys

    def test_realistic_deepseek_v3(self):
        rng = np.random.default_rng(42)
        depth, num_moe_layers, num_experts = 3, 58, 256
        g = rng.exponential(0.01, (depth, num_moe_layers, num_experts)).astype(np.float32)
        for d in range(depth):
            for m in range(num_moe_layers):
                top = rng.choice(num_experts, size=8, replace=False)
                g[d, m, top] = rng.uniform(0.05, 0.20, size=8).astype(np.float32)
                g[d, m] /= g[d, m].sum()

        resident_keys = set()
        for m in range(num_moe_layers):
            for eidx in rng.choice(num_experts, size=20, replace=False):
                resident_keys.add(ExpertKey(3 + m, int(eidx)))

        sp = _sp()
        plan = sp.compute_verification_plan(
            g, {0: resident_keys}, first_moe_layer=3,
        )
        assert isinstance(plan, VerificationPlan)
        assert plan.max_depth >= 0
        assert plan.max_depth <= depth - 1
        assert len(plan.transfers) <= 256

    def test_empty_gating(self):
        g = np.zeros((0, 0, 0), dtype=np.float32)
        sp = _sp()
        plan = sp.compute_verification_plan(g, {}, first_moe_layer=3)
        assert plan.transfers == []
        assert plan.max_depth == 0


# ---------------------------------------------------------------------------
# TestGatingToHints
# ---------------------------------------------------------------------------

class TestGatingToHints:

    def test_produces_sp_moe_hints(self):
        g = _gating(1, 1, 4, dominant=0)
        sp = _sp()
        hints = sp.gating_to_hints(g, first_moe_layer=3)
        assert all(h.source == PrefetchSource.SP_MOE for h in hints)

    def test_confidence_is_high(self):
        g = _gating(1, 1, 4, dominant=0)
        sp = _sp()
        hints = sp.gating_to_hints(g, first_moe_layer=3)
        assert all(h.confidence == PrefetchConfidence.HIGH for h in hints)

    def test_deduplication_keeps_max_score(self):
        g = np.zeros((2, 1, 4), dtype=np.float32)
        g[0, 0, 0] = 0.80
        g[1, 0, 0] = 0.95
        sp = _sp()
        hints = sp.gating_to_hints(g, first_moe_layer=3)
        key_hints = {h.key: h for h in hints}
        assert key_hints[ExpertKey(3, 0)].score == pytest.approx(0.95)

    def test_sorted_by_score_descending(self):
        g = np.zeros((1, 2, 4), dtype=np.float32)
        g[0, 0, 0] = 0.50
        g[0, 1, 1] = 0.95
        sp = _sp(config=SpeculativePrefetchConfig(adaptive_topk_threshold=0.01))
        hints = sp.gating_to_hints(g, first_moe_layer=3)
        scores = [h.score for h in hints]
        assert scores == sorted(scores, reverse=True)

    def test_empty_gating_empty_hints(self):
        g = np.zeros((0, 0, 0), dtype=np.float32)
        sp = _sp()
        hints = sp.gating_to_hints(g, first_moe_layer=3)
        assert hints == []


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------

class TestConfig:

    def test_default_values(self):
        cfg = SpeculativePrefetchConfig()
        assert cfg.adaptive_topk_threshold == 0.92
        assert cfg.verification_quality_floor == 0.85
        assert cfg.per_layer_us == 920
        assert cfg.pcie_bw_bytes_per_us == 64_000.0
        assert cfg.max_transfers_per_plan == 256

    def test_custom_threshold_changes_behavior(self):
        g = np.array([[[0.60, 0.30, 0.10]]], dtype=np.float32)
        sp_strict = _sp(config=SpeculativePrefetchConfig(
            adaptive_topk_threshold=0.92))
        sp_loose = _sp(config=SpeculativePrefetchConfig(
            adaptive_topk_threshold=0.50))
        plan_strict = sp_strict.compute_verification_plan(g, {},
                                                          first_moe_layer=3)
        plan_loose = sp_loose.compute_verification_plan(g, {},
                                                        first_moe_layer=3)
        assert len(plan_strict.transfers) >= len(plan_loose.transfers)

    def test_frozen_config(self):
        cfg = SpeculativePrefetchConfig()
        with pytest.raises(AttributeError):
            cfg.adaptive_topk_threshold = 0.5  # type: ignore[misc]
