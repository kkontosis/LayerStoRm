"""Expert placement — soft affinity tracking for routed expert placement.

Maintains a dense per-expert affinity map (ExpertKey -> gpu_idx) that tells the
orchestrator's transfer scheduler which GPU an expert should preferentially
reside on. Advisory only — ExpertCache handles actual slot allocation.

Initial placement: round-robin across all GPUs with expert cache capacity.
Affinity updates: consumed from CoactivationGraph AffinityHint via
apply_affinity_hints(). Periodic re-optimization handled by PlacementOptimizer.

Reimplements C++ src/parallelism/expert_placement.{h,cpp}.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from orchestrator.types import (
    AffinityHint,
    CacheZone,
    DuplicationCandidate,
    ExpertKey,
    InitialAssignment,
)


@dataclass(frozen=True)
class ExpertPlacementConfig:
    num_moe_layers: int = 58
    num_experts: int = 256
    first_moe_layer: int = 3
    cache_gpu_indices: tuple[int, ...] = (0, 1, 2, 3)
    initial_assignment: InitialAssignment = InitialAssignment.ROUND_ROBIN
    duplication_enabled: bool = True
    max_duplicated_fraction: float = 0.05
    duplication_frequency_threshold_percentile: int = 95


class ExpertPlacement:
    """Dense affinity map for expert-to-GPU placement.

    Not thread-safe. Single-threaded orchestrator (INV-3.4.2).
    """

    def __init__(self, config: ExpertPlacementConfig | None = None) -> None:
        self._config = config or ExpertPlacementConfig()
        cfg = self._config

        if not cfg.cache_gpu_indices:
            raise ValueError("cache_gpu_indices must not be empty")
        if cfg.num_moe_layers == 0:
            raise ValueError("num_moe_layers must be > 0")
        if cfg.num_experts == 0:
            raise ValueError("num_experts must be > 0")

        total = cfg.num_moe_layers * cfg.num_experts
        self._affinity_map = np.full(total, -1, dtype=np.int32)

        if cfg.initial_assignment == InitialAssignment.ROUND_ROBIN:
            self._assign_round_robin()

    @property
    def config(self) -> ExpertPlacementConfig:
        return self._config

    def affinity_gpu(self, key: ExpertKey) -> int:
        if not self._valid_key(key):
            return -1
        return int(self._affinity_map[self._flat_index(key)])

    def affinity_map_batch(
        self, keys: Sequence[ExpertKey],
    ) -> dict[ExpertKey, int]:
        """Vectorized {key -> affinity gpu} for many keys at once.

        TD-ORCH-PLAN-CRAWL: Phase-4 PLAN builds an affinity map over EVERY
        resident expert each cycle (~1k keys in the full-fit deployment);
        per-key affinity_gpu() calls (validity check + flat index + numpy
        scalar read) dominated the cycle. One fancy-index does all keys;
        invalid keys map to -1 (same as affinity_gpu).
        """
        cfg = self._config
        n = len(keys)
        if n == 0:
            return {}
        layers = np.fromiter((k.layer_idx for k in keys), np.int64, count=n)
        experts = np.fromiter((k.expert_idx for k in keys), np.int64, count=n)
        rel = layers - cfg.first_moe_layer
        valid = ((rel >= 0) & (rel < cfg.num_moe_layers)
                 & (experts >= 0) & (experts < cfg.num_experts))
        flat = np.where(valid, rel * cfg.num_experts + experts, 0)
        gpus = np.where(valid, self._affinity_map[flat], -1)
        return dict(zip(keys, gpus.tolist()))

    def experts_on_gpu(self, gpu_idx: int) -> list[ExpertKey]:
        indices = np.where(self._affinity_map == gpu_idx)[0]
        cfg = self._config
        result: list[ExpertKey] = []
        for idx in indices:
            layer = cfg.first_moe_layer + int(idx) // cfg.num_experts
            expert = int(idx) % cfg.num_experts
            result.append(ExpertKey(layer, expert))
        return result

    def experts_per_gpu(self) -> list[int]:
        gpus = self._config.cache_gpu_indices
        return [int(np.count_nonzero(self._affinity_map == g)) for g in gpus]

    def total_experts(self) -> int:
        return len(self._affinity_map)

    def apply_affinity_hints(self, hints: Sequence[AffinityHint]) -> int:
        changed = 0
        for h in hints:
            if not self._valid_key(h.key):
                continue
            idx = self._flat_index(h.key)
            if self._affinity_map[idx] != h.preferred_gpu:
                self._affinity_map[idx] = h.preferred_gpu
                changed += 1
        return changed

    def set_affinity(self, key: ExpertKey, gpu_idx: int) -> int:
        if not self._valid_key(key):
            return -1
        idx = self._flat_index(key)
        old = int(self._affinity_map[idx])
        self._affinity_map[idx] = gpu_idx
        return old

    def reset_to_round_robin(self) -> None:
        self._assign_round_robin()

    def recommended_zone(self, key: ExpertKey, target_gpu: int) -> CacheZone:
        if not self._valid_key(key):
            return CacheZone.STREAMING
        if self._affinity_map[self._flat_index(key)] == target_gpu:
            return CacheZone.STABLE
        return CacheZone.STREAMING

    def find_duplication_candidates(
        self,
        frequencies: dict[ExpertKey, float],
        frequency_percentiles: dict[ExpertKey, float],
        resident_set: dict[int, set[ExpertKey]],
        can_duplicate_fn: Callable[[int], bool],
        slot_info: dict[int, tuple[int, int]],
        avg_cross_gpu_latency_s: float,
    ) -> list[DuplicationCandidate]:
        if not self._config.duplication_enabled:
            return []

        cfg = self._config
        threshold = cfg.duplication_frequency_threshold_percentile
        candidates: list[DuplicationCandidate] = []

        for layer in range(cfg.first_moe_layer,
                           cfg.first_moe_layer + cfg.num_moe_layers):
            for exp in range(cfg.num_experts):
                key = ExpertKey(layer, exp)
                freq_pct = frequency_percentiles.get(key, 0.0)
                if freq_pct < threshold:
                    continue

                idx = self._flat_index(key)
                affinity = int(self._affinity_map[idx])
                freq = frequencies.get(key, 0.0)

                for gpu_idx in cfg.cache_gpu_indices:
                    if gpu_idx == affinity:
                        continue
                    if key in resident_set.get(gpu_idx, set()):
                        continue
                    if not can_duplicate_fn(gpu_idx):
                        continue

                    total, used = slot_info.get(gpu_idx, (0, 0))
                    pressure = used / total if total > 0 else 1.0
                    vram_cost = 1.0 / total if total > 0 else 1.0

                    benefit = (freq * avg_cross_gpu_latency_s
                               - vram_cost * pressure)
                    if benefit > 0.0:
                        candidates.append(DuplicationCandidate(
                            key=key,
                            source_gpu=affinity,
                            target_gpu=gpu_idx,
                            benefit=benefit,
                            frequency_percentile=freq_pct,
                        ))

        candidates.sort(key=lambda c: c.benefit, reverse=True)
        return candidates

    def _flat_index(self, key: ExpertKey) -> int:
        return ((key.layer_idx - self._config.first_moe_layer)
                * self._config.num_experts + key.expert_idx)

    def _valid_key(self, key: ExpertKey) -> bool:
        cfg = self._config
        return (cfg.first_moe_layer <= key.layer_idx
                < cfg.first_moe_layer + cfg.num_moe_layers
                and 0 <= key.expert_idx < cfg.num_experts)

    def _assign_round_robin(self) -> None:
        gpus = self._config.cache_gpu_indices
        n = len(gpus)
        for i in range(len(self._affinity_map)):
            self._affinity_map[i] = gpus[i % n]
