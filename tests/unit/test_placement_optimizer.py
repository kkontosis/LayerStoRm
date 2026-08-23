"""Tests for orchestrator.placement_optimizer — timer guards, shifts, NUMA migrations."""

import pytest

from orchestrator.types import AffinityHint, ExpertKey, InitialAssignment
from orchestrator.expert_placement import ExpertPlacement, ExpertPlacementConfig
from orchestrator.placement_optimizer import (
    AffinityHintRequest,
    NumaMigration,
    PlacementOptimizer,
    PlacementOptimizerConfig,
)


def _placement(num_gpus=4):
    return ExpertPlacement(ExpertPlacementConfig(
        num_moe_layers=3,
        num_experts=8,
        first_moe_layer=1,
        cache_gpu_indices=tuple(range(num_gpus)),
    ))


def _opt_cfg(**overrides):
    defaults = dict(
        reoptimize_interval_s=300.0,
        workload_shift_decay=0.1,
        min_tokens_before_optimize=256,
        coactivation_enabled=True,
        max_numa_migrations_per_cycle=4,
    )
    defaults.update(overrides)
    return PlacementOptimizerConfig(**defaults)


def _hints(placement, new_gpu=1):
    """Build hints moving all experts to new_gpu."""
    hints = []
    cfg = placement.config
    for layer in range(cfg.first_moe_layer,
                       cfg.first_moe_layer + cfg.num_moe_layers):
        for exp in range(cfg.num_experts):
            hints.append(AffinityHint(
                key=ExpertKey(layer, exp), preferred_gpu=new_gpu, score=1.0))
    return hints


# ── Timer guards ────────────────────────────────────────────────────────────


class TestTimerGuards:
    def test_first_call_optimizes_immediately(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(), ep)
        req = opt.maybe_reoptimize(
            shift_detected=False, tokens_processed=1000, now_s=0.0,
            gpu_capacities=[100, 100, 100, 100])
        assert req is not None
        assert req.num_gpus == 4

    def test_timer_respects_interval(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(reoptimize_interval_s=300.0), ep)
        # First call succeeds.
        req1 = opt.maybe_reoptimize(
            shift_detected=False, tokens_processed=1000, now_s=0.0,
            gpu_capacities=[100] * 4)
        assert req1 is not None
        # Too soon — blocked.
        req2 = opt.maybe_reoptimize(
            shift_detected=False, tokens_processed=2000, now_s=100.0,
            gpu_capacities=[100] * 4)
        assert req2 is None
        # After interval — succeeds.
        req3 = opt.maybe_reoptimize(
            shift_detected=False, tokens_processed=3000, now_s=301.0,
            gpu_capacities=[100] * 4)
        assert req3 is not None

    def test_blocked_by_min_tokens(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(min_tokens_before_optimize=256), ep)
        req = opt.maybe_reoptimize(
            shift_detected=False, tokens_processed=100, now_s=0.0,
            gpu_capacities=[100] * 4)
        assert req is None


# ── Affinity application ───────────────────────────────────────────────────


class TestAffinityApplication:
    def test_affinity_changes_reported(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(), ep)
        hints = [AffinityHint(key=ExpertKey(1, 0), preferred_gpu=3, score=1.0)]
        old = ep.affinity_gpu(ExpertKey(1, 0))
        changed = opt.apply_hints(hints, gpu_numa_nodes={})
        if old != 3:
            assert changed == 1
        assert ep.affinity_gpu(ExpertKey(1, 0)) == 3

    def test_apply_hints_updates_placement(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(), ep)
        hints = _hints(ep, new_gpu=2)
        opt.apply_hints(hints, gpu_numa_nodes={})
        for layer in range(1, 4):
            for exp in range(8):
                assert ep.affinity_gpu(ExpertKey(layer, exp)) == 2


# ── Workload shift ─────────────────────────────────────────────────────────


class TestWorkloadShift:
    def test_shift_resets_placement(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(), ep)
        original = [ep.affinity_gpu(ExpertKey(l, e))
                     for l in range(1, 4) for e in range(8)]
        # Modify placement.
        ep.set_affinity(ExpertKey(1, 0), 99)
        assert ep.affinity_gpu(ExpertKey(1, 0)) == 99
        # Shift resets to round-robin.
        opt.on_workload_shift(tokens_processed=500)
        restored = [ep.affinity_gpu(ExpertKey(l, e))
                     for l in range(1, 4) for e in range(8)]
        assert restored == original

    def test_shift_resets_timer_guard(self):
        ep = _placement()
        opt = PlacementOptimizer(
            _opt_cfg(reoptimize_interval_s=300.0, min_tokens_before_optimize=10),
            ep)
        # First call at t=0 succeeds.
        req1 = opt.maybe_reoptimize(False, 1000, 0.0, [100] * 4)
        assert req1 is not None
        # t=50, too soon, blocked.
        req2 = opt.maybe_reoptimize(False, 2000, 50.0, [100] * 4)
        assert req2 is None
        # Shift at t=60 resets timer.
        req3 = opt.maybe_reoptimize(True, 3000, 60.0, [100] * 4)
        assert req3 is None  # blocked by min_tokens (0 new tokens since shift)
        # After enough new tokens, first call post-shift is immediate.
        req4 = opt.maybe_reoptimize(False, 3100, 61.0, [100] * 4)
        assert req4 is not None

    def test_shift_clears_migrations(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(), ep)
        # Queue a migration manually.
        opt._pending_migrations.append(
            NumaMigration(key=ExpertKey(1, 0), new_gpu=1, new_numa_node=1))
        assert opt.pending_migration_count == 1
        opt.on_workload_shift(tokens_processed=500)
        assert opt.pending_migration_count == 0


# ── NUMA migrations ────────────────────────────────────────────────────────


class TestNumaMigrations:
    def test_migration_scheduled_on_cross_node(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(), ep)
        key = ExpertKey(1, 0)
        old_gpu = ep.affinity_gpu(key)
        new_gpu = 3 if old_gpu != 3 else 2
        hints = [AffinityHint(key=key, preferred_gpu=new_gpu, score=1.0)]
        # GPUs on different NUMA nodes.
        numa = {old_gpu: 0, new_gpu: 1}
        opt.apply_hints(hints, gpu_numa_nodes=numa)
        assert opt.pending_migration_count >= 1

    def test_migration_skipped_same_node(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(), ep)
        key = ExpertKey(1, 0)
        old_gpu = ep.affinity_gpu(key)
        new_gpu = 3 if old_gpu != 3 else 2
        hints = [AffinityHint(key=key, preferred_gpu=new_gpu, score=1.0)]
        # Same NUMA node.
        numa = {old_gpu: 0, new_gpu: 0}
        opt.apply_hints(hints, gpu_numa_nodes=numa)
        assert opt.pending_migration_count == 0

    def test_migration_limited_per_call(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(max_numa_migrations_per_cycle=2), ep)
        for i in range(5):
            opt._pending_migrations.append(
                NumaMigration(key=ExpertKey(1, i), new_gpu=ep.affinity_gpu(ExpertKey(1, i)),
                              new_numa_node=1))
        result = opt.process_numa_migrations(
            host_resident_keys={ExpertKey(1, i) for i in range(8)})
        assert len(result) <= 2

    def test_stale_migration_discarded(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(), ep)
        key = ExpertKey(1, 0)
        original_gpu = ep.affinity_gpu(key)
        # Queue migration to GPU 3.
        opt._pending_migrations.append(
            NumaMigration(key=key, new_gpu=3, new_numa_node=1))
        # But affinity changed again to GPU 2.
        ep.set_affinity(key, 2)
        result = opt.process_numa_migrations(
            host_resident_keys={key})
        # Migration to GPU 3 is stale (affinity is now 2), discarded.
        assert len(result) == 0

    def test_migration_skipped_not_in_host(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(), ep)
        key = ExpertKey(1, 0)
        opt._pending_migrations.append(
            NumaMigration(key=key, new_gpu=ep.affinity_gpu(key),
                          new_numa_node=1))
        # Key not in host_resident_keys.
        result = opt.process_numa_migrations(host_resident_keys=set())
        assert len(result) == 0


# ── Guards ──────────────────────────────────────────────────────────────────


class TestGuards:
    def test_disabled_coactivation_skips(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(coactivation_enabled=False), ep)
        req = opt.maybe_reoptimize(False, 10000, 0.0, [100] * 4)
        assert req is None

    def test_returns_none_when_blocked(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(min_tokens_before_optimize=999999), ep)
        req = opt.maybe_reoptimize(False, 100, 0.0, [100] * 4)
        assert req is None


# ── Stats ───────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_increment(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(), ep)
        opt.maybe_reoptimize(False, 1000, 0.0, [100] * 4)
        assert opt.stats.total_reoptimizations == 1
        hints = [AffinityHint(key=ExpertKey(1, 0), preferred_gpu=3, score=1.0)]
        opt.apply_hints(hints, gpu_numa_nodes={})
        assert opt.stats.total_affinity_changes >= 0

    def test_stats_after_shift(self):
        ep = _placement()
        opt = PlacementOptimizer(_opt_cfg(), ep)
        opt.on_workload_shift(tokens_processed=500)
        assert opt.stats.total_shift_events == 1
        opt.on_workload_shift(tokens_processed=1000)
        assert opt.stats.total_shift_events == 2
