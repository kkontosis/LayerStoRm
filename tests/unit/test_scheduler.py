"""Tests for orchestrator.scheduler — scoring + planning interfaces."""

import pytest

from orchestrator.types import (
    ComputeBatch,
    EvictionPlan,
    EvictionPlanEntry,
    ExpertKey,
    PrefetchConfidence,
    PrefetchHint,
    PrefetchSource,
    TransferPlan,
    TransferPlanEntry,
    WorkItem,
    WorkOperation,
    WorkStatus,
)
from orchestrator.scheduler import Scheduler, SchedulerConfig


def _item(request_id: int = 1, layer: int = 0,
          op: WorkOperation = WorkOperation.EXPERT_FFN,
          target_gpu: int = 0,
          experts: list[ExpertKey] | None = None,
          is_spec: bool = False, spec_pos: int = 0,
          created_ns: int = 0) -> WorkItem:
    return WorkItem(
        request_id=request_id, layer_idx=layer, operation=op,
        target_gpu=target_gpu, status=WorkStatus.READY,
        required_experts=experts or [],
        is_speculative=is_spec, speculation_position=spec_pos,
        timestamp_created_ns=created_ns,
    )


class TestScoring:
    def test_higher_wait_higher_urgency(self):
        s = Scheduler(SchedulerConfig(urgency_weight=1.0, readiness_weight=0.0))
        old = _item(created_ns=0)
        new = _item(request_id=2, created_ns=500_000_000)
        now = 1_000_000_000
        assert s.score(old, set(), now) > s.score(new, set(), now)

    def test_speculation_penalty(self):
        s = Scheduler(SchedulerConfig(urgency_weight=1.0, readiness_weight=0.0,
                                      speculation_penalty=0.2))
        real = _item(created_ns=0)
        spec = _item(request_id=2, created_ns=0, is_spec=True, spec_pos=3)
        now = 1_000_000_000
        assert s.score(real, set(), now) > s.score(spec, set(), now)

    def test_full_readiness(self):
        s = Scheduler(SchedulerConfig(urgency_weight=0.0, readiness_weight=1.0))
        experts = [ExpertKey(1, 0), ExpertKey(1, 1)]
        item = _item(experts=experts)
        resident = {ExpertKey(1, 0), ExpertKey(1, 1)}
        assert s.score(item, resident, 0) == pytest.approx(1.0)

    def test_partial_readiness(self):
        s = Scheduler(SchedulerConfig(urgency_weight=0.0, readiness_weight=1.0))
        experts = [ExpertKey(1, 0), ExpertKey(1, 1), ExpertKey(1, 2), ExpertKey(1, 3)]
        item = _item(experts=experts)
        resident = {ExpertKey(1, 0)}
        assert s.score(item, resident, 0) == pytest.approx(0.25)

    def test_no_experts_full_readiness(self):
        s = Scheduler(SchedulerConfig(urgency_weight=0.0, readiness_weight=1.0))
        item = _item(experts=[])
        assert s.score(item, set(), 0) == pytest.approx(1.0)

    def test_combined_weights(self):
        s = Scheduler(SchedulerConfig(urgency_weight=0.5, readiness_weight=0.5))
        experts = [ExpertKey(1, 0), ExpertKey(1, 1)]
        item = _item(experts=experts, created_ns=0)
        resident = {ExpertKey(1, 0), ExpertKey(1, 1)}
        now = 1_000_000_000
        score = s.score(item, resident, now)
        assert score == pytest.approx(0.5 * 1.0 + 0.5 * 1.0)

    def test_zero_wait_zero_urgency(self):
        s = Scheduler(SchedulerConfig(urgency_weight=1.0, readiness_weight=0.0))
        item = _item(created_ns=1000)
        assert s.score(item, set(), 1000) == pytest.approx(0.0)


class TestPlanCompute:
    def test_groups_by_gpu(self):
        s = Scheduler()
        items = [
            _item(request_id=1, target_gpu=0),
            _item(request_id=2, target_gpu=1),
            _item(request_id=3, target_gpu=0),
        ]
        batches = s.plan_compute(items)
        assert len(batches) == 2
        gpu_ids = {b.gpu_idx for b in batches}
        assert gpu_ids == {0, 1}

    def test_respects_max_batch_size(self):
        s = Scheduler(SchedulerConfig(max_batch_size=2))
        items = [_item(request_id=i, target_gpu=0) for i in range(5)]
        batches = s.plan_compute(items)
        gpu0_batches = [b for b in batches if b.gpu_idx == 0]
        assert len(gpu0_batches) == 3
        assert len(gpu0_batches[0].items) == 2
        assert len(gpu0_batches[1].items) == 2
        assert len(gpu0_batches[2].items) == 1

    def test_empty_input(self):
        s = Scheduler()
        assert s.plan_compute([]) == []


class TestPlanTransfers:
    def test_no_delegate_returns_empty(self):
        s = Scheduler()
        plan = s.plan_transfers([], [])
        assert len(plan.entries) == 0

    def test_with_delegate(self):
        class MockPlanner:
            def plan(self, prefetch_priorities, resident_inputs):
                return TransferPlan(entries=[
                    TransferPlanEntry(key=ExpertKey(1, 0), target_gpu=0,
                                     expert_bytes=1000),
                ])

        s = Scheduler()
        s.transfer_planner = MockPlanner()
        plan = s.plan_transfers([], [])
        assert len(plan.entries) == 1
        assert plan.entries[0].key == ExpertKey(1, 0)


class TestPlanEvictions:
    def test_no_delegate_returns_empty(self):
        s = Scheduler()
        plan = s.plan_evictions({0: 2}, [])
        assert len(plan.entries) == 0

    def test_with_delegate(self):
        class MockScorer:
            def score_evictions(self, inputs, gpu_idx, zone, count):
                return [
                    EvictionPlanEntry(key=ExpertKey(1, i), gpu_idx=gpu_idx)
                    for i in range(count)
                ]

        s = Scheduler()
        s.eviction_scorer = MockScorer()
        plan = s.plan_evictions({0: 2, 1: 1}, [])
        assert len(plan.entries) == 3


class TestPlanTypes:
    def test_transfer_plan_slots_per_gpu(self):
        plan = TransferPlan(entries=[
            TransferPlanEntry(key=ExpertKey(1, 0), target_gpu=0, expert_bytes=100),
            TransferPlanEntry(key=ExpertKey(1, 1), target_gpu=0, expert_bytes=200),
            TransferPlanEntry(key=ExpertKey(1, 2), target_gpu=1, expert_bytes=150),
        ])
        assert plan.required_slots_per_gpu() == {0: 2, 1: 1}
        assert plan.bytes_per_gpu() == {0: 300, 1: 150}

    def test_empty_plans(self):
        assert TransferPlan().required_slots_per_gpu() == {}
        assert EvictionPlan().entries == []
