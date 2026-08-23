"""Tests for orchestrator.transfer_scheduler — transfer planning, eviction delegation."""

import pytest

from orchestrator.types import (
    CacheZone,
    EvictionPlanEntry,
    ExpertEvictionInput,
    ExpertKey,
    PrefetchPriority,
    TransferPlan,
    TransferPlanEntry,
)
from orchestrator.eviction_policy import EvictionPolicy, EvictionPolicyConfig
from orchestrator.transfer_scheduler import (
    DEFAULT_PCIE_GEN5_X16_GBPS,
    STABLE_ZONE_THRESHOLD,
    TransferScheduler,
    TransferSchedulerConfig,
)
from orchestrator.scheduler import TransferPlanner


def _pp(layer=0, expert=0, gpu=0, priority=0.5, time_us=0):
    return PrefetchPriority(
        key=ExpertKey(layer, expert),
        target_layer=layer,
        target_gpu=gpu,
        priority_score=priority,
        estimated_time_until_needed_us=time_us,
    )


def _cfg(**overrides):
    defaults = dict(
        num_gpus=4,
        expert_bytes=1_000_000,
        pcie_bandwidth_gbps=(63.0, 63.0, 63.0, 63.0),
        max_inflight_per_gpu=4,
        min_priority_threshold=0.01,
        cycle_budget_us=1000.0,
    )
    defaults.update(overrides)
    return TransferSchedulerConfig(**defaults)


def _empty_state():
    return dict(resident_set={}, inflight_set={}, inflight_counts={},
                transfer_latency_us={})


# ── Construction ────────────────────────────────────────────────────────────


class TestConstruction:
    def test_valid(self):
        ts = TransferScheduler(_cfg())
        assert ts.config.num_gpus == 4

    def test_zero_gpus_raises(self):
        with pytest.raises(ValueError):
            TransferScheduler(_cfg(num_gpus=0))

    def test_zero_expert_bytes_raises(self):
        with pytest.raises(ValueError):
            TransferScheduler(_cfg(expert_bytes=0))


# ── plan_transfers filtering ────────────────────────────────────────────────


class TestFiltering:
    def test_higher_priority_first(self):
        ts = TransferScheduler(_cfg())
        pps = [_pp(expert=0, priority=0.3), _pp(expert=1, priority=0.9)]
        plan = ts.plan_transfers(pps, **_empty_state())
        assert len(plan.entries) == 2
        assert plan.entries[0].key.expert_idx == 1
        assert plan.entries[1].key.expert_idx == 0

    def test_below_threshold_filtered(self):
        ts = TransferScheduler(_cfg(min_priority_threshold=0.5))
        pps = [_pp(expert=0, priority=0.3), _pp(expert=1, priority=0.9)]
        plan = ts.plan_transfers(pps, **_empty_state())
        assert len(plan.entries) == 1
        assert plan.entries[0].key.expert_idx == 1

    def test_already_resident_skipped(self):
        ts = TransferScheduler(_cfg())
        key = ExpertKey(0, 0)
        pps = [_pp(expert=0, gpu=0, priority=0.9)]
        plan = ts.plan_transfers(pps, resident_set={0: {key}},
                                 inflight_set={}, inflight_counts={},
                                 transfer_latency_us={})
        assert len(plan.entries) == 0

    def test_already_inflight_skipped(self):
        ts = TransferScheduler(_cfg())
        key = ExpertKey(0, 0)
        pps = [_pp(expert=0, gpu=0, priority=0.9)]
        plan = ts.plan_transfers(pps, resident_set={},
                                 inflight_set={0: {key}},
                                 inflight_counts={},
                                 transfer_latency_us={})
        assert len(plan.entries) == 0

    def test_inflight_limit_per_gpu(self):
        ts = TransferScheduler(_cfg(max_inflight_per_gpu=2))
        pps = [_pp(expert=i, gpu=0, priority=0.9 - i * 0.1) for i in range(5)]
        plan = ts.plan_transfers(pps, resident_set={},
                                 inflight_set={},
                                 inflight_counts={0: 2},
                                 transfer_latency_us={})
        assert len(plan.entries) == 0


# ── Bandwidth budget ────────────────────────────────────────────────────────


class TestBandwidthBudget:
    def test_single_gpu_budget_respected(self):
        ts = TransferScheduler(_cfg(
            num_gpus=1,
            expert_bytes=1_000_000,
            pcie_bandwidth_gbps=(1.0,),
            cycle_budget_us=1000.0,
        ))
        # budget = 1.0 * 1000.0 * 1e3 = 1_000_000 bytes → fits 1 expert
        pps = [_pp(expert=0, gpu=0, priority=0.9),
               _pp(expert=1, gpu=0, priority=0.8)]
        plan = ts.plan_transfers(pps, **_empty_state())
        assert len(plan.entries) == 1

    def test_multi_gpu_independent_budgets(self):
        ts = TransferScheduler(_cfg(
            num_gpus=2,
            expert_bytes=1_000_000,
            pcie_bandwidth_gbps=(1.0, 1.0),
            cycle_budget_us=1000.0,
        ))
        pps = [_pp(expert=0, gpu=0, priority=0.9),
               _pp(expert=1, gpu=0, priority=0.8),
               _pp(expert=2, gpu=1, priority=0.7)]
        plan = ts.plan_transfers(pps, **_empty_state())
        gpu0 = [e for e in plan.entries if e.target_gpu == 0]
        gpu1 = [e for e in plan.entries if e.target_gpu == 1]
        assert len(gpu0) == 1
        assert len(gpu1) == 1

    def test_asymmetric_bandwidth(self):
        ts = TransferScheduler(_cfg(
            num_gpus=2,
            expert_bytes=1_000_000,
            pcie_bandwidth_gbps=(2.0, 1.0),
            cycle_budget_us=1000.0,
        ))
        # GPU 0: 2M budget (2 experts), GPU 1: 1M budget (1 expert)
        pps = [_pp(expert=0, gpu=0, priority=0.9),
               _pp(expert=1, gpu=0, priority=0.8),
               _pp(expert=2, gpu=0, priority=0.7),
               _pp(expert=3, gpu=1, priority=0.6),
               _pp(expert=4, gpu=1, priority=0.5)]
        plan = ts.plan_transfers(pps, **_empty_state())
        gpu0 = [e for e in plan.entries if e.target_gpu == 0]
        gpu1 = [e for e in plan.entries if e.target_gpu == 1]
        assert len(gpu0) == 2
        assert len(gpu1) == 1


# ── JIT delay ───────────────────────────────────────────────────────────────


class TestJitDelay:
    def test_zero_time_no_delay(self):
        ts = TransferScheduler(_cfg())
        pps = [_pp(expert=0, gpu=0, priority=0.9, time_us=0)]
        plan = ts.plan_transfers(pps, **_empty_state())
        assert plan.entries[0].start_delay_us == 0

    def test_distant_prediction_gets_delay(self):
        ts = TransferScheduler(_cfg())
        pps = [_pp(expert=0, gpu=0, priority=0.9, time_us=5000)]
        plan = ts.plan_transfers(pps, resident_set={}, inflight_set={},
                                 inflight_counts={},
                                 transfer_latency_us={0: 1000.0})
        assert plan.entries[0].start_delay_us == 4000

    def test_jit_clamped_to_zero(self):
        ts = TransferScheduler(_cfg())
        pps = [_pp(expert=0, gpu=0, priority=0.9, time_us=500)]
        plan = ts.plan_transfers(pps, resident_set={}, inflight_set={},
                                 inflight_counts={},
                                 transfer_latency_us={0: 2000.0})
        assert plan.entries[0].start_delay_us == 0


# ── Zone routing ────────────────────────────────────────────────────────────


class TestZoneRouting:
    def test_high_priority_routes_to_stable(self):
        ts = TransferScheduler(_cfg())
        pps = [_pp(expert=0, gpu=0, priority=0.9)]
        plan = ts.plan_transfers(pps, **_empty_state())
        assert plan.entries[0].zone == CacheZone.STABLE

    def test_low_priority_routes_to_streaming(self):
        ts = TransferScheduler(_cfg())
        pps = [_pp(expert=0, gpu=0, priority=0.3)]
        plan = ts.plan_transfers(pps, **_empty_state())
        assert plan.entries[0].zone == CacheZone.STREAMING

    def test_boundary_priority_is_stable(self):
        ts = TransferScheduler(_cfg())
        pps = [_pp(expert=0, gpu=0, priority=0.7)]
        plan = ts.plan_transfers(pps, **_empty_state())
        assert plan.entries[0].zone == CacheZone.STABLE


# ── plan_evictions ──────────────────────────────────────────────────────────


class TestPlanEvictions:
    def test_no_eviction_when_free_slots(self):
        ts = TransferScheduler(_cfg())
        ep = EvictionPolicy()
        plan = ts.plan_evictions(
            required_slots={0: 1},
            eviction_inputs=[],
            eviction_policy=ep,
            free_slots={0: (5, 5)},
        )
        assert len(plan.entries) == 0

    def test_eviction_count_matches_deficit(self):
        ts = TransferScheduler(_cfg())
        ep = EvictionPolicy()
        inputs = [
            ExpertEvictionInput(key=ExpertKey(0, i), gpu_idx=0, recency=0.5)
            for i in range(10)
        ]
        plan = ts.plan_evictions(
            required_slots={0: 3},
            eviction_inputs=inputs,
            eviction_policy=ep,
            free_slots={0: (0, 0)},
        )
        assert len(plan.entries) == 3

    def test_eviction_delegates_to_policy(self):
        ts = TransferScheduler(_cfg())
        ep = EvictionPolicy()
        inputs = [
            ExpertEvictionInput(key=ExpertKey(0, 0), gpu_idx=0,
                                recency=0.1, frequency=0.9),
            ExpertEvictionInput(key=ExpertKey(0, 1), gpu_idx=0,
                                recency=0.9, frequency=0.1),
        ]
        plan = ts.plan_evictions(
            required_slots={0: 1},
            eviction_inputs=inputs,
            eviction_policy=ep,
            free_slots={0: (0, 0)},
        )
        assert len(plan.entries) == 1
        assert plan.entries[0].key.expert_idx == 1  # higher recency = more evictable


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_prefetch_empty_plan(self):
        ts = TransferScheduler(_cfg())
        plan = ts.plan_transfers([], **_empty_state())
        assert len(plan.entries) == 0

    def test_invalid_gpu_skipped(self):
        ts = TransferScheduler(_cfg(num_gpus=2))
        pps = [_pp(expert=0, gpu=99, priority=0.9),
               _pp(expert=1, gpu=-1, priority=0.8),
               _pp(expert=2, gpu=0, priority=0.7)]
        plan = ts.plan_transfers(pps, **_empty_state())
        assert len(plan.entries) == 1
        assert plan.entries[0].key.expert_idx == 2

    def test_expert_bytes_in_entries(self):
        ts = TransferScheduler(_cfg(expert_bytes=42_000))
        pps = [_pp(expert=0, gpu=0, priority=0.9)]
        plan = ts.plan_transfers(pps, **_empty_state())
        assert plan.entries[0].expert_bytes == 42_000

    def test_required_slots_per_gpu(self):
        plan = TransferPlan(entries=[
            TransferPlanEntry(key=ExpertKey(0, 0), target_gpu=0, expert_bytes=100),
            TransferPlanEntry(key=ExpertKey(0, 1), target_gpu=0, expert_bytes=200),
            TransferPlanEntry(key=ExpertKey(0, 2), target_gpu=1, expert_bytes=150),
        ])
        assert plan.required_slots_per_gpu() == {0: 2, 1: 1}


# ── Protocol conformance ───────────────────────────────────────────────────


class TestProtocol:
    def test_implements_transfer_planner(self):
        ts = TransferScheduler(_cfg())
        assert isinstance(ts, TransferPlanner)
