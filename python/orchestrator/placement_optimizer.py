"""Placement optimizer — timer-gated expert affinity re-optimization.

Periodically triggers CoactivationGraph partitioning (via IPC) and applies
results to ExpertPlacement. Handles workload shifts (aggressive decay + reset)
and queues NUMA buffer migrations for cross-node affinity changes.

Reimplements C++ src/parallelism/placement_optimizer.{h,cpp}.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from orchestrator.expert_placement import ExpertPlacement
from orchestrator.types import AffinityHint, ExpertKey


@dataclass(frozen=True)
class PlacementOptimizerConfig:
    reoptimize_interval_s: float = 300.0
    workload_shift_decay: float = 0.1
    min_tokens_before_optimize: int = 256
    coactivation_enabled: bool = True
    max_numa_migrations_per_cycle: int = 4


@dataclass(frozen=True)
class AffinityHintRequest:
    num_gpus: int
    gpu_capacities: list[int]


@dataclass
class NumaMigration:
    key: ExpertKey
    new_gpu: int
    new_numa_node: int


@dataclass
class OptimizerStats:
    total_reoptimizations: int = 0
    total_affinity_changes: int = 0
    total_shift_events: int = 0
    total_numa_migrations: int = 0


class PlacementOptimizer:
    """Timer-gated coordinator for expert placement re-optimization.

    Called by the orchestrator every cycle. Stateful — tracks timer,
    pending NUMA migrations, and statistics.

    Not thread-safe. Single-threaded orchestrator (INV-3.4.2).
    """

    def __init__(
        self,
        config: PlacementOptimizerConfig,
        placement: ExpertPlacement,
    ) -> None:
        self._config = config
        self._placement = placement
        self._last_reoptimize_time: float = 0.0
        self._has_ever_optimized: bool = False
        self._tokens_at_last_shift: int = 0
        self._pending_migrations: list[NumaMigration] = []
        self.stats = OptimizerStats()

    @property
    def config(self) -> PlacementOptimizerConfig:
        return self._config

    @property
    def pending_migration_count(self) -> int:
        return len(self._pending_migrations)

    def maybe_reoptimize(
        self,
        shift_detected: bool,
        tokens_processed: int,
        now_s: float,
        gpu_capacities: list[int],
    ) -> AffinityHintRequest | None:
        if shift_detected:
            self.on_workload_shift(tokens_processed)

        cfg = self._config
        if not cfg.coactivation_enabled:
            return None

        tokens_since_shift = tokens_processed - self._tokens_at_last_shift
        if tokens_since_shift < cfg.min_tokens_before_optimize:
            return None

        if self._has_ever_optimized:
            if (now_s - self._last_reoptimize_time) < cfg.reoptimize_interval_s:
                return None

        self._last_reoptimize_time = now_s
        self._has_ever_optimized = True
        self.stats.total_reoptimizations += 1

        return AffinityHintRequest(
            num_gpus=len(gpu_capacities),
            gpu_capacities=gpu_capacities,
        )

    def apply_hints(
        self,
        hints: Sequence[AffinityHint],
        gpu_numa_nodes: dict[int, int],
    ) -> int:
        old_gpus = [self._placement.affinity_gpu(h.key) for h in hints]

        changed = self._placement.apply_affinity_hints(hints)

        for i, h in enumerate(hints):
            new_pos = h.preferred_gpu
            old_pos = old_gpus[i]
            if old_pos == new_pos or old_pos < 0:
                continue

            old_node = gpu_numa_nodes.get(old_pos, -1)
            new_node = gpu_numa_nodes.get(new_pos, -1)
            if old_node != new_node and old_node >= 0 and new_node >= 0:
                self._pending_migrations.append(
                    NumaMigration(key=h.key, new_gpu=new_pos,
                                  new_numa_node=new_node))

        self.stats.total_affinity_changes += changed
        return changed

    def on_workload_shift(self, tokens_processed: int) -> None:
        self._placement.reset_to_round_robin()
        self._pending_migrations.clear()
        self._has_ever_optimized = False
        self._tokens_at_last_shift = tokens_processed
        self.stats.total_shift_events += 1

    def process_numa_migrations(
        self,
        host_resident_keys: set[ExpertKey],
        max_count: int | None = None,
    ) -> list[NumaMigration]:
        if max_count is None:
            max_count = self._config.max_numa_migrations_per_cycle

        result: list[NumaMigration] = []
        while self._pending_migrations and len(result) < max_count:
            m = self._pending_migrations.pop()  # LIFO
            if m.key not in host_resident_keys:
                continue
            if self._placement.affinity_gpu(m.key) != m.new_gpu:
                continue
            result.append(m)
            self.stats.total_numa_migrations += 1

        return result
