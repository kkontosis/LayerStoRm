"""Scheduler interface — scoring, transfer/compute/eviction planning.

Implements the heuristic scoring function from spec §3.3. Planning methods
delegate to pluggable Protocol implementations (Phase 13a) or return empty
plans when no delegate is set.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from orchestrator.types import (
    CacheZone,
    ComputeBatch,
    EvictionPlan,
    EvictionPlanEntry,
    ExpertEvictionInput,
    ExpertKey,
    PrefetchPriority,
    TransferPlan,
    WorkItem,
    WorkStatus,
)


@runtime_checkable
class EvictionScorer(Protocol):
    def score_evictions(
        self,
        inputs: Sequence[ExpertEvictionInput],
        gpu_idx: int,
        zone: CacheZone,
        count: int,
    ) -> list[EvictionPlanEntry]: ...


@runtime_checkable
class TransferPlanner(Protocol):
    def plan(
        self,
        prefetch_priorities: list[PrefetchPriority],
        resident_inputs: Sequence[ExpertEvictionInput],
    ) -> TransferPlan: ...


@runtime_checkable
class GpuAssigner(Protocol):
    def assign(self, item: WorkItem) -> int: ...


@dataclass(frozen=True)
class SchedulerConfig:
    urgency_weight: float = 0.4
    readiness_weight: float = 0.4
    batching_weight: float = 0.2
    max_batch_size: int = 64
    speculation_penalty: float = 0.1


class Scheduler:

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self._config = config or SchedulerConfig()
        self.eviction_scorer: EvictionScorer | None = None
        self.transfer_planner: TransferPlanner | None = None
        self.gpu_assigner: GpuAssigner | None = None

    @property
    def config(self) -> SchedulerConfig:
        return self._config

    def score(self, item: WorkItem, resident_keys: set[ExpertKey],
              now_ns: int) -> float:
        urgency = self._compute_urgency(item, now_ns)
        readiness = self._compute_readiness(item, resident_keys)
        return (self._config.urgency_weight * urgency
                + self._config.readiness_weight * readiness)

    def plan_transfers(self, prefetch_priorities: list[PrefetchPriority],
                       resident_inputs: Sequence[ExpertEvictionInput]) -> TransferPlan:
        if self.transfer_planner is not None:
            return self.transfer_planner.plan(prefetch_priorities, resident_inputs)
        return TransferPlan()

    def plan_compute(self, ready_items: list[WorkItem]) -> list[ComputeBatch]:
        if not ready_items:
            return []
        by_gpu: dict[int, list[WorkItem]] = defaultdict(list)
        for item in ready_items:
            by_gpu[item.target_gpu].append(item)

        batches: list[ComputeBatch] = []
        for gpu_idx, items in by_gpu.items():
            for i in range(0, len(items), self._config.max_batch_size):
                chunk = items[i:i + self._config.max_batch_size]
                batches.append(ComputeBatch(items=chunk, gpu_idx=gpu_idx))
        return batches

    def plan_evictions(self, required_slots: dict[int, int],
                       inputs: Sequence[ExpertEvictionInput]) -> EvictionPlan:
        if self.eviction_scorer is None:
            return EvictionPlan()
        entries: list[EvictionPlanEntry] = []
        for gpu_idx, count in required_slots.items():
            entries.extend(
                self.eviction_scorer.score_evictions(
                    inputs, gpu_idx, CacheZone.STABLE, count))
        return EvictionPlan(entries=entries)

    def _compute_urgency(self, item: WorkItem, now_ns: int) -> float:
        if now_ns > item.timestamp_created_ns:
            wait_s = (now_ns - item.timestamp_created_ns) / 1e9
            urgency = min(wait_s, 1.0)
        else:
            urgency = 0.0
        if item.is_speculative:
            urgency -= self._config.speculation_penalty * item.speculation_position
            urgency = max(urgency, 0.0)
        return urgency

    def _compute_readiness(self, item: WorkItem,
                           resident_keys: set[ExpertKey]) -> float:
        if not item.required_experts:
            return 1.0
        count = sum(1 for k in item.required_experts if k in resident_keys)
        return count / len(item.required_experts)
