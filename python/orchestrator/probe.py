"""PROBE — multi-layer predictive prefetching from probe points.

Places lightweight predictors at configurable probe points (default 25%, 50%,
75% through the network). At each probe point, the current hidden state is
projected through a fixed random feature extractor + SiLU, then per-target-layer
output heads predict expert activation probabilities for ALL remaining MoE
layers. Pure signal producer — no GPU commands, no expert transfers.

Reference: Zhu et al., "PROBE: Co-Balancing Computation and Communication in
MoE Inference via Real-Time Predictive Prefetching", arXiv:2602.00509.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orchestrator.types import (
    EngineMetadata,
    ExpertKey,
    PrefetchConfidence,
    PrefetchHint,
    PrefetchSource,
)


@dataclass(frozen=True)
class ProbeConfig:
    """Inline config (prefetch.probe in main config)."""
    enabled: bool = True
    probe_points: tuple[float, ...] = (0.25, 0.5, 0.75)
    confidence_threshold: float = 0.6
    top_k: int = 8
    score_threshold: float = 0.01


@dataclass(frozen=True)
class ProbePredictorConfig:
    """Internal config (_internal-probe, loaded from probe.json)."""
    hidden_size: int = 7168
    feature_dim: int = 256
    learning_rate: float = 1e-3
    momentum: float = 0.9
    training_batch_size: int = 32
    training_buffer_size: int = 2048
    min_samples_before_predict: int = 64


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -88, 88)
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


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

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        high = min(self.count, self.capacity)
        idx = rng.integers(0, high, size=min(n, high))
        return self.inputs[idx], self.labels[idx]


class ProbePredictor:
    """Multi-layer expert activation predictor from probe points.

    Architecture per probe point:
      - Fixed random orthogonal projection (hidden_size -> feature_dim) + SiLU
      - Trainable output heads: (feature_dim -> R * num_experts) where R is
        the number of MoE layers remaining after the probe point
      - Per (probe_point, target_layer) ring buffers for training data
      - Double-buffered weights with SGD + momentum training
    """

    def __init__(self, metadata: EngineMetadata,
                 config: ProbePredictorConfig,
                 probe_points: tuple[float, ...] = (0.25, 0.5, 0.75)) -> None:
        self._cfg = config
        self._ne = metadata.num_experts
        self._n_moe = metadata.num_moe_layers
        self._n_layers = metadata.num_layers
        self._first_moe = metadata.num_layers - metadata.num_moe_layers
        self._rng = np.random.default_rng(42)

        self._probe_layers = tuple(
            round(frac * (metadata.num_layers - 1)) for frac in probe_points
        )

        self._target_layers: dict[int, list[int]] = {}
        for pp in self._probe_layers:
            self._target_layers[pp] = [
                layer for layer in range(self._first_moe,
                                         self._first_moe + self._n_moe)
                if layer > pp
            ]

        self._projections: dict[int, np.ndarray] = {}
        for pp in self._probe_layers:
            raw = self._rng.standard_normal(
                (config.hidden_size, config.feature_dim)).astype(np.float32)
            q, _ = np.linalg.qr(raw)
            self._projections[pp] = q[:, :config.feature_dim].astype(np.float32)

        self._weights: list[dict[int, dict[str, np.ndarray]]] = [{}, {}]
        self._velocity: dict[int, dict[str, np.ndarray]] = {}
        self._active_buf: dict[int, int] = {}

        self._buffers: dict[tuple[int, int], _RingBuffer] = {}

        for pp in self._probe_layers:
            R = len(self._target_layers[pp])
            for b in range(2):
                self._weights[b][pp] = self._init_weights(R)
            self._velocity[pp] = {
                k: np.zeros_like(v)
                for k, v in self._weights[0][pp].items()
            }
            self._active_buf[pp] = 0
            for target in self._target_layers[pp]:
                self._buffers[(pp, target)] = _RingBuffer(
                    config.training_buffer_size, config.feature_dim, self._ne)

    @property
    def probe_layer_indices(self) -> tuple[int, ...]:
        return self._probe_layers

    def target_layers_for(self, pp_layer: int) -> list[int]:
        return list(self._target_layers.get(pp_layer, []))

    def sample_count(self, pp_layer: int) -> int:
        targets = self._target_layers.get(pp_layer, [])
        if not targets:
            return 0
        return min(self._buffers[(pp_layer, t)].count for t in targets)

    def is_warm(self, pp_layer: int) -> bool:
        return self.sample_count(pp_layer) >= self._cfg.min_samples_before_predict

    def predict(self, pp_layer: int,
                hidden_state: np.ndarray) -> dict[int, np.ndarray]:
        if not self.is_warm(pp_layer):
            return {}
        targets = self._target_layers.get(pp_layer)
        if not targets:
            return {}
        feat = self._extract_features(pp_layer, hidden_state)
        w = self._weights[self._active_buf[pp_layer]][pp_layer]
        logits = feat @ w['w_out'] + w['b_out']
        scores = _sigmoid(logits)
        R = len(targets)
        scores_matrix = scores.reshape(R, self._ne)
        return {target: scores_matrix[i] for i, target in enumerate(targets)}

    def record_sample(self, pp_layer: int, hidden_state: np.ndarray,
                      target_layer: int,
                      actual_expert_mask: np.ndarray) -> None:
        buf = self._buffers.get((pp_layer, target_layer))
        if buf is None:
            return
        feat = self._extract_features(pp_layer, hidden_state)
        buf.add(feat, actual_expert_mask.astype(np.float32))

    def record_samples_batch(
            self, pp_layer: int, hidden_state: np.ndarray,
            samples: list[tuple[int, np.ndarray]],
    ) -> None:
        if not samples:
            return
        feat = self._extract_features(pp_layer, hidden_state)
        for target_layer, expert_mask in samples:
            buf = self._buffers.get((pp_layer, target_layer))
            if buf is not None:
                buf.add(feat, expert_mask.astype(np.float32))

    def maybe_train(self, pp_layer: int) -> bool:
        targets = self._target_layers.get(pp_layer, [])
        if not targets:
            return False
        bs = self._cfg.training_batch_size
        if self.sample_count(pp_layer) < bs:
            return False

        active = self._active_buf[pp_layer]
        inactive = 1 - active
        w = self._weights[inactive][pp_layer]
        for k, v in self._weights[active][pp_layer].items():
            w[k] = v.copy()

        ne = self._ne
        grads: dict[str, np.ndarray] = {
            'w_out': np.zeros_like(w['w_out']),
            'b_out': np.zeros_like(w['b_out']),
        }

        for i, target in enumerate(targets):
            buf = self._buffers[(pp_layer, target)]
            X, Y = buf.sample(bs, self._rng)

            w_slice = w['w_out'][:, i * ne:(i + 1) * ne]
            b_slice = w['b_out'][i * ne:(i + 1) * ne]

            logits = X @ w_slice + b_slice
            P = _sigmoid(logits)
            dlogits = (P - Y) / bs

            grads['w_out'][:, i * ne:(i + 1) * ne] = X.T @ dlogits
            grads['b_out'][i * ne:(i + 1) * ne] = dlogits.sum(axis=0)

        vel = self._velocity[pp_layer]
        lr = self._cfg.learning_rate
        mu = self._cfg.momentum
        for k in grads:
            vel[k] = mu * vel[k] + grads[k]
            w[k] = w[k] - lr * vel[k]

        self._active_buf[pp_layer] = inactive
        return True

    def _extract_features(self, pp_layer: int,
                          hidden_state: np.ndarray) -> np.ndarray:
        proj = self._projections[pp_layer]
        z = hidden_state.astype(np.float32) @ proj
        return _silu(z)

    def _init_weights(self, num_targets: int) -> dict[str, np.ndarray]:
        fd = self._cfg.feature_dim
        ne = self._ne

        def _he(fan_in: int, fan_out: int) -> np.ndarray:
            return (self._rng.standard_normal((fan_in, fan_out)).astype(np.float32)
                    * np.float32(np.sqrt(2.0 / fan_in)))

        return {
            'w_out': _he(fd, num_targets * ne),
            'b_out': np.zeros(num_targets * ne, dtype=np.float32),
        }


class Probe:
    """Multi-layer predictive prefetching from probe points."""

    def __init__(self, metadata: EngineMetadata,
                 config: ProbeConfig | None = None,
                 predictor: ProbePredictor | None = None) -> None:
        self._config = config or ProbeConfig()
        self._predictor = predictor
        self._num_experts = metadata.num_experts
        self._num_layers = metadata.num_layers
        self._first_moe = metadata.num_layers - metadata.num_moe_layers

        self._probe_layers = tuple(
            round(frac * (metadata.num_layers - 1))
            for frac in self._config.probe_points
        )

    @property
    def config(self) -> ProbeConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def probe_layers(self) -> tuple[int, ...]:
        return self._probe_layers

    def is_probe_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._probe_layers

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self._first_moe <= layer_idx < self._num_layers

    def predict(self, probe_point_layer: int,
                hidden_state: np.ndarray) -> list[PrefetchHint]:
        if self._predictor is None:
            return []
        if probe_point_layer not in self._probe_layers:
            return []
        predictions = self._predictor.predict(probe_point_layer, hidden_state)
        if not predictions:
            return []
        hints: list[PrefetchHint] = []
        for target_layer, scores in predictions.items():
            hints.extend(self._scores_to_hints(target_layer, scores))
        hints.sort(key=lambda h: h.score, reverse=True)
        return hints

    def record_training_sample(
            self, probe_point_layer: int, hidden_state: np.ndarray,
            target_layer: int, actual_expert_mask: np.ndarray,
    ) -> None:
        if self._predictor is not None:
            self._predictor.record_sample(
                probe_point_layer, hidden_state,
                target_layer, actual_expert_mask)

    def record_training_samples(
            self, probe_point_layer: int, hidden_state: np.ndarray,
            samples: list[tuple[int, np.ndarray]],
    ) -> None:
        if self._predictor is not None:
            self._predictor.record_samples_batch(
                probe_point_layer, hidden_state, samples)

    def _scores_to_hints(self, target_layer: int,
                         scores: np.ndarray) -> list[PrefetchHint]:
        if scores.ndim == 1:
            scores = scores[np.newaxis, :]
        top_k = min(self._config.top_k, scores.shape[1])
        agg: dict[int, float] = {}
        for row in scores:
            if top_k >= len(row):
                indices = np.arange(len(row))
            else:
                indices = np.argpartition(row, -top_k)[-top_k:]
            for idx in indices:
                val = float(row[idx])
                if val >= self._config.score_threshold:
                    agg[int(idx)] = max(agg.get(int(idx), 0.0), val)
        return [
            PrefetchHint(
                key=ExpertKey(target_layer, eidx),
                target_layer=target_layer,
                confidence=PrefetchConfidence.MEDIUM,
                source=PrefetchSource.PROBE,
                score=sc,
            )
            for eidx, sc in agg.items()
        ]
