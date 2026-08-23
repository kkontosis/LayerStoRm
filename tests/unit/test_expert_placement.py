"""Tests for orchestrator.expert_placement — affinity map, zone recommendation, duplication."""

import pytest

from orchestrator.types import (
    AffinityHint,
    CacheZone,
    DuplicationCandidate,
    ExpertKey,
    InitialAssignment,
)
from orchestrator.expert_placement import (
    ExpertPlacement,
    ExpertPlacementConfig,
)


def _small_cfg(**overrides):
    defaults = dict(
        num_moe_layers=3,
        num_experts=8,
        first_moe_layer=1,
        cache_gpu_indices=(0, 1, 2, 3),
        initial_assignment=InitialAssignment.ROUND_ROBIN,
        duplication_enabled=True,
        max_duplicated_fraction=0.05,
        duplication_frequency_threshold_percentile=95,
    )
    defaults.update(overrides)
    return ExpertPlacementConfig(**defaults)


def _v32_cfg(**overrides):
    defaults = dict(
        num_moe_layers=58,
        num_experts=256,
        first_moe_layer=3,
        cache_gpu_indices=(0, 1, 2, 3),
    )
    defaults.update(overrides)
    return ExpertPlacementConfig(**defaults)


# ── Construction ────────────────────────────────────────────────────────────


class TestConstruction:
    def test_round_robin(self):
        ep = ExpertPlacement(_small_cfg())
        assert ep.total_experts() == 3 * 8
        for layer in range(1, 4):
            for exp in range(8):
                assert ep.affinity_gpu(ExpertKey(layer, exp)) != -1

        counts = ep.experts_per_gpu()
        assert len(counts) == 4
        assert all(c == 6 for c in counts)

    def test_empty_gpu_list_raises(self):
        with pytest.raises(ValueError):
            ExpertPlacement(_small_cfg(cache_gpu_indices=()))

    def test_zero_experts_raises(self):
        with pytest.raises(ValueError):
            ExpertPlacement(_small_cfg(num_experts=0))

    def test_zero_layers_raises(self):
        with pytest.raises(ValueError):
            ExpertPlacement(_small_cfg(num_moe_layers=0))


# ── Round-Robin Distribution ────────────────────────────────────────────────


class TestRoundRobin:
    def test_even_distribution(self):
        ep = ExpertPlacement(_v32_cfg())
        counts = ep.experts_per_gpu()
        assert sum(counts) == 58 * 256
        assert all(c == 3712 for c in counts)

    def test_single_gpu(self):
        ep = ExpertPlacement(_small_cfg(cache_gpu_indices=(7,)))
        for layer in range(1, 4):
            for exp in range(8):
                assert ep.affinity_gpu(ExpertKey(layer, exp)) == 7

    def test_non_contiguous_gpu_ids(self):
        ep = ExpertPlacement(_small_cfg(cache_gpu_indices=(0, 2, 5)))
        assert ep.affinity_gpu(ExpertKey(1, 0)) == 0
        assert ep.affinity_gpu(ExpertKey(1, 1)) == 2
        assert ep.affinity_gpu(ExpertKey(1, 2)) == 5
        assert ep.affinity_gpu(ExpertKey(1, 3)) == 0

    def test_deterministic(self):
        ep1 = ExpertPlacement(_v32_cfg())
        ep2 = ExpertPlacement(_v32_cfg())
        for layer in range(3, 61):
            for exp in range(256):
                assert ep1.affinity_gpu(ExpertKey(layer, exp)) == \
                       ep2.affinity_gpu(ExpertKey(layer, exp))


# ── Affinity Queries ────────────────────────────────────────────────────────


class TestAffinityQueries:
    def test_invalid_key(self):
        ep = ExpertPlacement(_small_cfg())
        assert ep.affinity_gpu(ExpertKey(0, 0)) == -1
        assert ep.affinity_gpu(ExpertKey(4, 0)) == -1
        assert ep.affinity_gpu(ExpertKey(1, 8)) == -1

    def test_experts_on_gpu(self):
        ep = ExpertPlacement(_small_cfg())
        seen: set[ExpertKey] = set()
        for gpu in (0, 1, 2, 3):
            experts = ep.experts_on_gpu(gpu)
            for k in experts:
                assert k not in seen, f"Duplicate: {k}"
                seen.add(k)
        assert len(seen) == ep.total_experts()

    def test_experts_per_gpu_matches_experts_on_gpu(self):
        ep = ExpertPlacement(_small_cfg())
        counts = ep.experts_per_gpu()
        for i, gpu in enumerate((0, 1, 2, 3)):
            assert len(ep.experts_on_gpu(gpu)) == counts[i]


# ── Affinity Updates ────────────────────────────────────────────────────────


class TestAffinityUpdates:
    def test_apply_hints_changes_map(self):
        ep = ExpertPlacement(_small_cfg())
        old_gpu = ep.affinity_gpu(ExpertKey(1, 0))
        new_gpu = 1 if old_gpu != 1 else 0
        hints = [AffinityHint(key=ExpertKey(1, 0), preferred_gpu=new_gpu, score=1.0)]
        changed = ep.apply_affinity_hints(hints)
        assert changed == 1
        assert ep.affinity_gpu(ExpertKey(1, 0)) == new_gpu

    def test_apply_hints_no_change_returns_zero(self):
        ep = ExpertPlacement(_small_cfg())
        current = ep.affinity_gpu(ExpertKey(1, 0))
        hints = [AffinityHint(key=ExpertKey(1, 0), preferred_gpu=current, score=1.0)]
        assert ep.apply_affinity_hints(hints) == 0

    def test_apply_hints_skips_invalid_keys(self):
        ep = ExpertPlacement(_small_cfg())
        hints = [
            AffinityHint(key=ExpertKey(0, 0), preferred_gpu=0, score=1.0),
            AffinityHint(key=ExpertKey(99, 0), preferred_gpu=0, score=1.0),
        ]
        assert ep.apply_affinity_hints(hints) == 0

    def test_set_affinity_override(self):
        ep = ExpertPlacement(_small_cfg())
        old = ep.affinity_gpu(ExpertKey(2, 3))
        ret = ep.set_affinity(ExpertKey(2, 3), 99)
        assert ret == old
        assert ep.affinity_gpu(ExpertKey(2, 3)) == 99

    def test_set_affinity_invalid_key(self):
        ep = ExpertPlacement(_small_cfg())
        assert ep.set_affinity(ExpertKey(0, 0), 0) == -1


# ── Reset ───────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_to_round_robin(self):
        ep = ExpertPlacement(_small_cfg())
        original = [ep.affinity_gpu(ExpertKey(l, e))
                     for l in range(1, 4) for e in range(8)]
        ep.set_affinity(ExpertKey(1, 0), 99)
        ep.set_affinity(ExpertKey(2, 5), 99)
        assert ep.affinity_gpu(ExpertKey(1, 0)) == 99

        ep.reset_to_round_robin()
        restored = [ep.affinity_gpu(ExpertKey(l, e))
                     for l in range(1, 4) for e in range(8)]
        assert restored == original


# ── Zone Recommendation ─────────────────────────────────────────────────────


class TestZoneRecommendation:
    def test_affinity_match_stable(self):
        ep = ExpertPlacement(_small_cfg())
        gpu = ep.affinity_gpu(ExpertKey(1, 0))
        assert ep.recommended_zone(ExpertKey(1, 0), gpu) == CacheZone.STABLE

    def test_affinity_mismatch_streaming(self):
        ep = ExpertPlacement(_small_cfg())
        gpu = ep.affinity_gpu(ExpertKey(1, 0))
        other = 1 if gpu != 1 else 0
        assert ep.recommended_zone(ExpertKey(1, 0), other) == CacheZone.STREAMING

    def test_invalid_key_streaming(self):
        ep = ExpertPlacement(_small_cfg())
        assert ep.recommended_zone(ExpertKey(0, 0), 0) == CacheZone.STREAMING


# ── Duplication Candidates ──────────────────────────────────────────────────


class TestDuplication:
    def test_disabled_returns_empty(self):
        ep = ExpertPlacement(_small_cfg(duplication_enabled=False))
        result = ep.find_duplication_candidates(
            frequencies={}, frequency_percentiles={},
            resident_set={}, can_duplicate_fn=lambda _: True,
            slot_info={}, avg_cross_gpu_latency_s=0.001)
        assert result == []

    def test_candidates_sorted_by_benefit(self):
        cfg = _small_cfg(duplication_frequency_threshold_percentile=50)
        ep = ExpertPlacement(cfg)

        freqs: dict[ExpertKey, float] = {}
        pcts: dict[ExpertKey, float] = {}
        for layer in range(1, 4):
            for exp in range(8):
                k = ExpertKey(layer, exp)
                if layer == 1 and exp in (0, 1):
                    freqs[k] = 0.9 if exp == 0 else 0.5
                    pcts[k] = 99.0 if exp == 0 else 80.0
                else:
                    freqs[k] = 0.01
                    pcts[k] = 10.0

        slot_info = {g: (100, 10) for g in (0, 1, 2, 3)}
        result = ep.find_duplication_candidates(
            frequencies=freqs, frequency_percentiles=pcts,
            resident_set={}, can_duplicate_fn=lambda _: True,
            slot_info=slot_info, avg_cross_gpu_latency_s=0.001)

        for i in range(1, len(result)):
            assert result[i - 1].benefit >= result[i].benefit

    def test_already_resident_excluded(self):
        cfg = _small_cfg(
            cache_gpu_indices=(0, 1),
            duplication_frequency_threshold_percentile=50,
        )
        ep = ExpertPlacement(cfg)

        key = ExpertKey(1, 0)
        affinity = ep.affinity_gpu(key)
        other = 1 if affinity == 0 else 0

        freqs = {key: 0.9}
        pcts = {key: 99.0}
        resident_set = {other: {key}}
        slot_info = {0: (100, 10), 1: (100, 10)}

        result = ep.find_duplication_candidates(
            frequencies=freqs, frequency_percentiles=pcts,
            resident_set=resident_set, can_duplicate_fn=lambda _: True,
            slot_info=slot_info, avg_cross_gpu_latency_s=0.001)

        for c in result:
            if c.key == key:
                assert c.target_gpu != other


# ── Coactivation Init ───────────────────────────────────────────────────────


class TestCoactivationInit:
    def test_coactivation_init_unassigned(self):
        ep = ExpertPlacement(_small_cfg(
            initial_assignment=InitialAssignment.COACTIVATION))
        for layer in range(1, 4):
            for exp in range(8):
                assert ep.affinity_gpu(ExpertKey(layer, exp)) == -1

    def test_coactivation_then_apply_hints(self):
        cfg = _small_cfg(initial_assignment=InitialAssignment.COACTIVATION)
        ep = ExpertPlacement(cfg)

        hints = []
        gpus = cfg.cache_gpu_indices
        for layer in range(1, 4):
            for exp in range(8):
                gpu = gpus[exp % len(gpus)]
                hints.append(AffinityHint(
                    key=ExpertKey(layer, exp), preferred_gpu=gpu, score=1.0))

        changed = ep.apply_affinity_hints(hints)
        assert changed == 24

        for layer in range(1, 4):
            for exp in range(8):
                assert ep.affinity_gpu(ExpertKey(layer, exp)) != -1
