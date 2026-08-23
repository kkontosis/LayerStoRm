"""Transfer scheduler — stateless transfer and eviction planning.

Plans which expert H2D transfers to issue and which experts to evict to make
room. Does NOT issue DMA itself (INV-4.5.1) — the orchestrator's DISPATCH
phase does that via command ring.

Respects: PCIe bandwidth budget per GPU, VRAM capacity, in-flight dedup,
just-in-time scheduling for distant predictions.

Reimplements C++ src/core/transfer/transfer_scheduler.{h,cpp}.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from orchestrator.eviction_policy import EvictionPolicy
from orchestrator.types import (
    CacheZone,
    EvictionPlan,
    EvictionPlanEntry,
    ExpertEvictionInput,
    ExpertKey,
    PrefetchPriority,
    TransferPlan,
    TransferPlanEntry,
)

STABLE_ZONE_THRESHOLD: float = 0.7
DEFAULT_PCIE_GEN5_X16_GBPS: float = 63.008


@dataclass(frozen=True)
class TransferSchedulerConfig:
    num_gpus: int = 4
    expert_bytes: int = 0
    pcie_bandwidth_gbps: tuple[float, ...] = ()
    max_inflight_per_gpu: int = 4
    min_priority_threshold: float = 0.01
    cycle_budget_us: float = 1000.0
    stable_zone_threshold: float = STABLE_ZONE_THRESHOLD


class TransferScheduler:
    """Priority-based transfer queue planner for the orchestrator's PLAN phase.

    Not thread-safe. Single-threaded orchestrator (INV-3.4.2).
    """

    def __init__(self, config: TransferSchedulerConfig) -> None:
        if config.num_gpus <= 0:
            raise ValueError("num_gpus must be > 0")
        if config.expert_bytes <= 0:
            raise ValueError("expert_bytes must be > 0")

        self._config = config
        self._budget_bytes: list[int] = []
        bw_list = config.pcie_bandwidth_gbps
        for i in range(config.num_gpus):
            bw = bw_list[i] if i < len(bw_list) else DEFAULT_PCIE_GEN5_X16_GBPS
            self._budget_bytes.append(int(bw * config.cycle_budget_us * 1e3))

    @property
    def config(self) -> TransferSchedulerConfig:
        return self._config

    def plan_transfers(
        self,
        prefetch_priorities: list[PrefetchPriority],
        resident_set: dict[int, set[ExpertKey]],
        inflight_set: dict[int, set[ExpertKey]],
        inflight_counts: dict[int, int],
        transfer_latency_us: dict[int, float],
    ) -> TransferPlan:
        if not prefetch_priorities:
            return TransferPlan()

        cfg = self._config
        expert_bytes = cfg.expert_bytes

        candidates: list[tuple[ExpertKey, int, float, int]] = []
        for pp in prefetch_priorities:
            gpu = pp.target_gpu
            if gpu < 0 or gpu >= cfg.num_gpus:
                continue
            if pp.priority_score < cfg.min_priority_threshold:
                continue
            if pp.key in resident_set.get(gpu, set()):
                continue
            if pp.key in inflight_set.get(gpu, set()):
                continue
            if inflight_counts.get(gpu, 0) >= cfg.max_inflight_per_gpu:
                continue
            candidates.append((
                pp.key, gpu, pp.priority_score,
                pp.estimated_time_until_needed_us))

        candidates.sort(key=lambda c: c[2], reverse=True)

        budget = list(self._budget_bytes)
        entries: list[TransferPlanEntry] = []

        for key, gpu, priority, time_us in candidates:
            if budget[gpu] < expert_bytes:
                continue

            delay = _compute_jit_delay(time_us, transfer_latency_us.get(gpu, 0.0))
            zone = _route_to_zone(priority, cfg.stable_zone_threshold)

            entries.append(TransferPlanEntry(
                key=key,
                target_gpu=gpu,
                zone=zone,
                priority=priority,
                start_delay_us=delay,
                expert_bytes=expert_bytes,
            ))
            budget[gpu] -= expert_bytes

        return TransferPlan(entries=entries)

    def plan_evictions(
        self,
        required_slots: dict[int, int],
        eviction_inputs: Sequence[ExpertEvictionInput],
        eviction_policy: EvictionPolicy,
        free_slots: dict[int, tuple[int, int]],
    ) -> EvictionPlan:
        entries: list[EvictionPlanEntry] = []

        for gpu_idx, needed in required_slots.items():
            if gpu_idx < 0 or gpu_idx >= self._config.num_gpus:
                continue

            stable_free, streaming_free = free_slots.get(gpu_idx, (0, 0))
            total_free = streaming_free + stable_free

            if needed <= total_free:
                continue

            deficit = needed - total_free
            candidates = eviction_policy.select_evictions(
                eviction_inputs, gpu_idx, CacheZone.STABLE, deficit)
            entries.extend(candidates)

        return EvictionPlan(entries=entries)

    def plan(
        self,
        prefetch_priorities: list[PrefetchPriority],
        resident_inputs: Sequence[ExpertEvictionInput],
    ) -> TransferPlan:
        resident_set: dict[int, set[ExpertKey]] = {}
        for inp in resident_inputs:
            resident_set.setdefault(inp.gpu_idx, set()).add(inp.key)
        return self.plan_transfers(
            prefetch_priorities,
            resident_set=resident_set,
            inflight_set={},
            inflight_counts={},
            transfer_latency_us={},
        )


def _compute_jit_delay(time_until_needed_us: int, transfer_latency_us: float) -> int:
    transfer_int = int(transfer_latency_us)
    if time_until_needed_us <= transfer_int:
        return 0
    return time_until_needed_us - transfer_int


def _route_to_zone(priority_score: float, threshold: float) -> CacheZone:
    return CacheZone.STABLE if priority_score >= threshold else CacheZone.STREAMING
