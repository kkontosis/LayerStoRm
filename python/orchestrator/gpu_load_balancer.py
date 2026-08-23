"""GPU load balancer — cross-GPU work distribution with 7-term scoring.

Implements the GpuAssigner protocol from scheduler.py. Scores each (work_item,
gpu) pair using 7 terms: residency overlap, compute capacity, VRAM headroom,
soft affinity, transfer cost (bin-aware, urgency-modulated), stable-zone cache
quality, streaming-zone cache quality.

Supports asymmetric GPU configurations (2x RTX5090 TP + 2x RTX5080).
ATTENTION/EMBEDDING/OUTPUT_HEAD ops pinned to TP GPUs; EXPERT_FFN/GATING ops
distributed across all GPUs via scoring.

New module — no C++ predecessor.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from orchestrator.types import (
    ExpertKey,
    GpuConfig,
    WorkItem,
    WorkOperation,
)

_NUM_FEATURES = 7

_TP_PINNED_OPS: frozenset[WorkOperation] = frozenset({
    WorkOperation.ATTENTION,
    WorkOperation.EMBEDDING,
    WorkOperation.OUTPUT_HEAD,
})


@dataclass(frozen=True)
class GpuLoadBalancerConfig:
    gpus: tuple[GpuConfig, ...] = ()

    weight_residency: float = 0.25
    weight_capacity: float = 0.20
    weight_headroom: float = 0.10
    weight_affinity: float = 0.15
    weight_transfer_cost: float = 0.15
    weight_cache_quality_stable: float = 0.10
    weight_cache_quality_streaming: float = 0.05

    priority_bin_thresholds: tuple[float, ...] = (0.3, 0.7)


class GpuLoadBalancer:
    """7-term scoring GPU work distributor.

    Satisfies GpuAssigner Protocol (scheduler.py).
    Not thread-safe. Single-threaded orchestrator (INV-3.4.2).
    """

    def __init__(self, config: GpuLoadBalancerConfig) -> None:
        if not config.gpus:
            raise ValueError("gpus must not be empty")

        self._config = config
        self._gpu_indices: list[int] = [g.position for g in config.gpus]
        self._tp_gpu_indices: list[int] = [
            g.position for g in config.gpus if g.is_tp
        ]
        self._gpu_by_position: dict[int, GpuConfig] = {
            g.position: g for g in config.gpus
        }
        self._bin_thresholds: list[float] = sorted(config.priority_bin_thresholds)
        self._num_bins: int = len(self._bin_thresholds) + 1

        self._resident_keys: dict[int, set[ExpertKey]] = {}
        self._queue_depths: dict[int, int] = {}
        self._vram_free: dict[int, int] = {}
        self._affinity_map: dict[ExpertKey, int] | None = None
        self._pending_transfers: dict[int, list[int]] = {}
        self._cache_quality_stable: dict[int, float] = {}
        self._cache_quality_streaming: dict[int, float] = {}

    @property
    def config(self) -> GpuLoadBalancerConfig:
        return self._config

    def update_state(
        self,
        resident_keys_per_gpu: dict[int, set[ExpertKey]],
        queue_depths: dict[int, int],
        vram_free: dict[int, int],
        affinity_map: dict[ExpertKey, int] | None = None,
        pending_bin_counts_per_gpu: dict[int, list[int]] | None = None,
        cache_quality_stable: dict[int, float] | None = None,
        cache_quality_streaming: dict[int, float] | None = None,
    ) -> None:
        self._resident_keys = resident_keys_per_gpu
        self._queue_depths = dict(queue_depths)
        self._vram_free = vram_free
        self._affinity_map = affinity_map
        self._pending_transfers = (
            {gpu: list(bins) for gpu, bins in pending_bin_counts_per_gpu.items()}
            if pending_bin_counts_per_gpu is not None
            else {}
        )
        self._cache_quality_stable = cache_quality_stable or {}
        self._cache_quality_streaming = cache_quality_streaming or {}

    def assign(self, item: WorkItem) -> int:
        return self._assign_impl(
            item,
            self._resident_keys,
            self._queue_depths,
            self._vram_free,
            self._affinity_map,
            self._pending_transfers,
            self._cache_quality_stable,
            self._cache_quality_streaming,
        )

    def assign_target_gpu(
        self,
        item: WorkItem,
        resident_keys_per_gpu: dict[int, set[ExpertKey]],
        queue_depths: dict[int, int],
        vram_free: dict[int, int],
        affinity_map: dict[ExpertKey, int] | None = None,
        pending_bin_counts_per_gpu: dict[int, list[int]] | None = None,
        cache_quality_stable: dict[int, float] | None = None,
        cache_quality_streaming: dict[int, float] | None = None,
    ) -> int:
        pending = (
            {gpu: list(bins) for gpu, bins in pending_bin_counts_per_gpu.items()}
            if pending_bin_counts_per_gpu is not None
            else {}
        )
        return self._assign_impl(
            item,
            resident_keys_per_gpu,
            queue_depths,
            vram_free,
            affinity_map,
            pending,
            cache_quality_stable or {},
            cache_quality_streaming or {},
        )

    def recommend_transfer_target(
        self,
        key: ExpertKey,
        resident_keys_per_gpu: dict[int, set[ExpertKey]],
        queue_depths: dict[int, int],
        vram_free: dict[int, int],
        affinity_map: dict[ExpertKey, int] | None = None,
        cache_quality_stable: dict[int, float] | None = None,
        cache_quality_streaming: dict[int, float] | None = None,
    ) -> int:
        cqs = cache_quality_stable or {}
        cqr = cache_quality_streaming or {}
        cfg = self._config
        best_gpu = self._gpu_indices[0]
        best_score = -float("inf")

        for gpu_idx in self._gpu_indices:
            gpu = self._gpu_by_position[gpu_idx]
            is_resident = key in resident_keys_per_gpu.get(gpu_idx, set())
            residency_bonus = 1.0 if is_resident else 0.0
            affinity_score = (
                1.0
                if affinity_map is not None and affinity_map.get(key) == gpu_idx
                else 0.0
            )
            headroom = self._compute_headroom(gpu_idx, vram_free)
            capacity = _compute_capacity(gpu, queue_depths.get(gpu_idx, 0))

            score = (
                cfg.weight_residency * residency_bonus
                + cfg.weight_affinity * affinity_score
                + cfg.weight_headroom * headroom
                + cfg.weight_capacity * capacity
                + cfg.weight_cache_quality_stable * cqs.get(gpu_idx, 0.0)
                + cfg.weight_cache_quality_streaming * cqr.get(gpu_idx, 0.0)
            )
            if score > best_score:
                best_score = score
                best_gpu = gpu_idx

        return best_gpu

    def assign_batch(
        self,
        items: list[WorkItem],
        resident_keys_per_gpu: dict[int, set[ExpertKey]],
        queue_depths: dict[int, int],
        vram_free: dict[int, int],
        affinity_map: dict[ExpertKey, int] | None = None,
        pending_bin_counts_per_gpu: dict[int, list[int]] | None = None,
        cache_quality_stable: dict[int, float] | None = None,
        cache_quality_streaming: dict[int, float] | None = None,
    ) -> list[WorkItem]:
        depths = dict(queue_depths)
        pending = (
            {gpu: list(bins) for gpu, bins in pending_bin_counts_per_gpu.items()}
            if pending_bin_counts_per_gpu is not None
            else {}
        )
        cqs = cache_quality_stable or {}
        cqr = cache_quality_streaming or {}

        for item in items:
            gpu = self._assign_impl(
                item, resident_keys_per_gpu, depths, vram_free,
                affinity_map, pending, cqs, cqr,
            )
            item.target_gpu = gpu
            depths[gpu] = depths.get(gpu, 0) + 1
            self._increment_pending(
                pending, gpu, item, resident_keys_per_gpu,
            )

        return items

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assign_impl(
        self,
        item: WorkItem,
        resident_keys_per_gpu: dict[int, set[ExpertKey]],
        queue_depths: dict[int, int],
        vram_free: dict[int, int],
        affinity_map: dict[ExpertKey, int] | None,
        pending: dict[int, list[int]],
        cache_quality_stable: dict[int, float],
        cache_quality_streaming: dict[int, float],
    ) -> int:
        if item.operation in _TP_PINNED_OPS:
            return self._assign_tp(queue_depths)

        best_gpu = self._gpu_indices[0]
        best_score = -float("inf")

        for gpu_idx in self._gpu_indices:
            score = self._score_gpu(
                item, gpu_idx, resident_keys_per_gpu, queue_depths,
                vram_free, affinity_map, pending,
                cache_quality_stable, cache_quality_streaming,
            )
            if score > best_score:
                best_score = score
                best_gpu = gpu_idx

        return best_gpu

    def _assign_tp(self, queue_depths: dict[int, int]) -> int:
        if not self._tp_gpu_indices:
            return self._gpu_indices[0]

        best_gpu = self._tp_gpu_indices[0]
        best_load = float("inf")
        for gpu_idx in self._tp_gpu_indices:
            cfg = self._gpu_by_position[gpu_idx]
            load = queue_depths.get(gpu_idx, 0) / max(cfg.compute_weight, 1e-9)
            if load < best_load:
                best_load = load
                best_gpu = gpu_idx
        return best_gpu

    def _score_gpu(
        self,
        item: WorkItem,
        gpu_idx: int,
        resident_keys_per_gpu: dict[int, set[ExpertKey]],
        queue_depths: dict[int, int],
        vram_free: dict[int, int],
        affinity_map: dict[ExpertKey, int] | None,
        pending: dict[int, list[int]],
        cache_quality_stable: dict[int, float],
        cache_quality_streaming: dict[int, float],
    ) -> float:
        cfg = self._config
        gpu = self._gpu_by_position[gpu_idx]
        resident = resident_keys_per_gpu.get(gpu_idx, set())

        residency = _compute_residency(item.required_experts, resident)
        capacity = _compute_capacity(gpu, queue_depths.get(gpu_idx, 0))
        headroom = self._compute_headroom(gpu_idx, vram_free)
        affinity = _compute_affinity(item.required_experts, gpu_idx, affinity_map)
        transfer_cost = self._compute_transfer_cost(
            item, resident, pending.get(gpu_idx),
        )

        return (
            cfg.weight_residency * residency
            + cfg.weight_capacity * capacity
            + cfg.weight_headroom * headroom
            + cfg.weight_affinity * affinity
            - cfg.weight_transfer_cost * transfer_cost * (1.0 + item.priority)
            + cfg.weight_cache_quality_stable * cache_quality_stable.get(gpu_idx, 0.0)
            + cfg.weight_cache_quality_streaming * cache_quality_streaming.get(gpu_idx, 0.0)
        )

    def _compute_headroom(self, gpu_idx: int, vram_free: dict[int, int]) -> float:
        gpu = self._gpu_by_position.get(gpu_idx)
        if gpu is None or gpu.vram_bytes <= 0:
            return 0.0
        free = vram_free.get(gpu_idx, 0)
        return max(0.0, min(1.0, free / gpu.vram_bytes))

    def _compute_transfer_cost(
        self,
        item: WorkItem,
        resident_on_gpu: set[ExpertKey],
        gpu_bins: list[int] | None,
    ) -> float:
        if not item.required_experts:
            return 0.0

        non_resident = sum(
            1 for k in item.required_experts if k not in resident_on_gpu
        )
        if non_resident == 0:
            return 0.0

        non_resident_frac = non_resident / len(item.required_experts)

        congestion = 0.0
        if gpu_bins is not None:
            item_bin = self._item_priority_bin(item)
            higher_bin_total = sum(gpu_bins[item_bin:])
            all_total = sum(gpu_bins)
            if all_total > 0:
                congestion = higher_bin_total / all_total

        return non_resident_frac * (1.0 + congestion)

    def _item_priority_bin(self, item: WorkItem) -> int:
        return bisect.bisect_right(self._bin_thresholds, item.priority)

    def _increment_pending(
        self,
        pending: dict[int, list[int]],
        gpu_idx: int,
        item: WorkItem,
        resident_keys_per_gpu: dict[int, set[ExpertKey]],
    ) -> None:
        if not item.required_experts:
            return
        resident = resident_keys_per_gpu.get(gpu_idx, set())
        non_resident = sum(
            1 for k in item.required_experts if k not in resident
        )
        if non_resident == 0:
            return
        if gpu_idx not in pending:
            pending[gpu_idx] = [0] * self._num_bins
        bin_idx = self._item_priority_bin(item)
        pending[gpu_idx][bin_idx] += non_resident


def _compute_residency(
    required_experts: list[ExpertKey],
    resident_on_gpu: set[ExpertKey],
) -> float:
    if not required_experts:
        return 1.0
    count = sum(1 for k in required_experts if k in resident_on_gpu)
    return count / len(required_experts)


def _compute_capacity(gpu: GpuConfig, queue_depth: int) -> float:
    cw = gpu.compute_weight
    if cw <= 0:
        return 0.0
    return 1.0 / (1.0 + queue_depth / cw)


def _compute_affinity(
    required_experts: list[ExpertKey],
    gpu_idx: int,
    affinity_map: dict[ExpertKey, int] | None,
) -> float:
    if not required_experts or affinity_map is None:
        return 0.0
    count = sum(1 for k in required_experts if affinity_map.get(k) == gpu_idx)
    return count / len(required_experts)


# ======================================================================
# Learned GPU Scorer — online-trained MLP alternative to heuristic
# ======================================================================


@dataclass(frozen=True)
class LearnedGpuScorerConfig:
    enabled: bool = False
    hidden_size: int = 32
    learning_rate: float = 1e-4
    training_buffer_size: int = 10000
    training_batch_size: int = 32
    min_samples_before_predict: int = 128
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


class _RingBuffer:
    __slots__ = ("capacity", "inputs", "labels", "count", "_idx")

    def __init__(self, capacity: int, input_dim: int, label_dim: int) -> None:
        self.capacity = capacity
        self.inputs = np.zeros((capacity, input_dim), dtype=np.float32)
        self.labels = np.zeros((capacity, label_dim), dtype=np.float32)
        self.count = 0
        self._idx = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        self.inputs[self._idx] = x
        self.labels[self._idx] = y
        self._idx = (self._idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        high = min(self.count, self.capacity)
        idx = rng.integers(0, high, size=min(n, high))
        return self.inputs[idx], self.labels[idx]


class LearnedGpuScorer:
    """Online-trained MLP latency predictor implementing GpuAssigner.

    Learns to predict per-GPU latency from the same 7 normalized features
    the heuristic computes. Picks the GPU with lowest predicted latency.
    Falls back to heuristic scorer when cold (not enough training samples)
    or for TP-pinned ops.

    Not thread-safe. Single-threaded orchestrator (INV-3.4.2).
    """

    def __init__(
        self,
        config: LearnedGpuScorerConfig,
        fallback: GpuLoadBalancer,
    ) -> None:
        self._config = config
        self._fallback = fallback
        self._rng = np.random.default_rng(44)

        h = config.hidden_size
        self._weights: list[dict[str, np.ndarray]] = [
            self._init_weights(h), self._init_weights(h),
        ]
        self._adam_m: dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self._weights[0].items()
        }
        self._adam_v: dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self._weights[0].items()
        }
        self._adam_step: int = 0
        self._active_buf: int = 0
        self._train_count: int = 0
        self._buffer = _RingBuffer(config.training_buffer_size, _NUM_FEATURES, 1)
        self._pending_outcomes: dict[tuple[int, int], np.ndarray] = {}

    @property
    def config(self) -> LearnedGpuScorerConfig:
        return self._config

    @property
    def fallback(self) -> GpuLoadBalancer:
        return self._fallback

    def update_state(
        self,
        resident_keys_per_gpu: dict[int, set[ExpertKey]],
        queue_depths: dict[int, int],
        vram_free: dict[int, int],
        affinity_map: dict[ExpertKey, int] | None = None,
        pending_bin_counts_per_gpu: dict[int, list[int]] | None = None,
        cache_quality_stable: dict[int, float] | None = None,
        cache_quality_streaming: dict[int, float] | None = None,
    ) -> None:
        self._fallback.update_state(
            resident_keys_per_gpu, queue_depths, vram_free,
            affinity_map, pending_bin_counts_per_gpu,
            cache_quality_stable, cache_quality_streaming,
        )

    def assign(self, item: WorkItem) -> int:
        if item.operation in _TP_PINNED_OPS:
            return self._fallback._assign_tp(self._fallback._queue_depths)

        if not self._config.enabled or not self.is_warm():
            return self._fallback.assign(item)

        fb = self._fallback
        best_gpu = fb._gpu_indices[0]
        best_latency = float("inf")

        for gpu_idx in fb._gpu_indices:
            features = self._extract_features(item, gpu_idx)
            latency = self._predict_latency(features)
            if latency < best_latency:
                best_latency = latency
                best_gpu = gpu_idx

        self._pending_outcomes[(item.request_id, best_gpu)] = (
            self._extract_features(item, best_gpu)
        )
        return best_gpu

    def record_outcome(
        self,
        request_id: int,
        gpu_idx: int,
        observed_latency_us: float,
    ) -> None:
        key = (request_id, gpu_idx)
        features = self._pending_outcomes.pop(key, None)
        if features is None:
            return
        self._buffer.add(features, np.array([observed_latency_us],
                                            dtype=np.float32))

    def maybe_train(self) -> bool:
        cfg = self._config
        if self._buffer.count < cfg.training_batch_size:
            return False

        active = self._active_buf
        inactive = 1 - active
        w = self._weights[inactive]
        for k, v in self._weights[active].items():
            w[k] = v.copy()

        X, Y = self._buffer.sample(cfg.training_batch_size, self._rng)

        hidden = np.maximum(0.0, X @ w["w1"] + w["b1"])
        pred = hidden @ w["w2"] + w["b2"]

        diff = pred - Y
        d_out = diff / X.shape[0]

        d_hidden = d_out @ w["w2"].T
        d_hidden = d_hidden * (hidden > 0).astype(np.float32)

        grads = {
            "w2": hidden.T @ d_out,
            "b2": d_out.sum(axis=0),
            "w1": X.T @ d_hidden,
            "b1": d_hidden.sum(axis=0),
        }

        self._adam_step += 1
        t = self._adam_step
        for k in grads:
            self._adam_m[k] = cfg.beta1 * self._adam_m[k] + (1 - cfg.beta1) * grads[k]
            self._adam_v[k] = cfg.beta2 * self._adam_v[k] + (1 - cfg.beta2) * grads[k] ** 2
            m_hat = self._adam_m[k] / (1 - cfg.beta1 ** t)
            v_hat = self._adam_v[k] / (1 - cfg.beta2 ** t)
            w[k] -= cfg.learning_rate * m_hat / (np.sqrt(v_hat) + cfg.epsilon)

        self._active_buf = inactive
        self._train_count += 1
        return True

    def is_warm(self) -> bool:
        return self._buffer.count >= self._config.min_samples_before_predict

    def sample_count(self) -> int:
        return self._buffer.count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_features(self, item: WorkItem, gpu_idx: int) -> np.ndarray:
        fb = self._fallback
        gpu = fb._gpu_by_position[gpu_idx]
        resident = fb._resident_keys.get(gpu_idx, set())

        residency = _compute_residency(item.required_experts, resident)
        capacity = _compute_capacity(gpu, fb._queue_depths.get(gpu_idx, 0))
        headroom = fb._compute_headroom(gpu_idx, fb._vram_free)
        affinity = _compute_affinity(
            item.required_experts, gpu_idx, fb._affinity_map,
        )
        transfer_cost = fb._compute_transfer_cost(
            item, resident, fb._pending_transfers.get(gpu_idx),
        )
        cq_stable = fb._cache_quality_stable.get(gpu_idx, 0.0)
        cq_streaming = fb._cache_quality_streaming.get(gpu_idx, 0.0)

        return np.array(
            [residency, capacity, headroom, affinity,
             transfer_cost, cq_stable, cq_streaming],
            dtype=np.float32,
        )

    def _predict_latency(self, features: np.ndarray) -> float:
        w = self._weights[self._active_buf]
        hidden = np.maximum(0.0, features @ w["w1"] + w["b1"])
        return float((hidden @ w["w2"] + w["b2"])[0])

    def _init_weights(self, hidden: int) -> dict[str, np.ndarray]:
        limit1 = np.sqrt(2.0 / _NUM_FEATURES)
        limit2 = np.sqrt(2.0 / hidden)
        return {
            "w1": (self._rng.standard_normal((_NUM_FEATURES, hidden))
                   * limit1).astype(np.float32),
            "b1": np.zeros(hidden, dtype=np.float32),
            "w2": (self._rng.standard_normal((hidden, 1))
                   * limit2).astype(np.float32),
            "b2": np.zeros(1, dtype=np.float32),
        }
