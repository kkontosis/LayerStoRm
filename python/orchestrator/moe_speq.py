"""MoE-SpeQ — online-trained per-(source,target) MoE layer pair predictor.

Trains a separate predictor for each (source_layer, target_layer) pair of MoE
layers. Each predictor uses a fixed random projection (hidden_size -> feature_dim)
with ReLU, then a trainable output head (feature_dim -> num_experts) with sigmoid.
Runs entirely on CPU. Produces PrefetchHints with variable confidence mapped from
sigmoid probabilities. Pure signal producer — no GPU commands, no expert transfers.

Reference: "MoE-SpeQ: Proactive Expert Prefetching for Resource-Constrained
MoE Inference".
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
class MoeSpeqConfig:
    """Inline config (prefetch.moe_speq in main config)."""
    enabled: bool = True
    top_k: int = 8
    score_threshold: float = 0.01
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.5


@dataclass(frozen=True)
class MoeSpeqPredictorConfig:
    """Internal config (_internal-moe-speq)."""
    hidden_size: int = 7168
    feature_dim: int = 256
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    training_batch_size: int = 32
    training_buffer_size: int = 512
    min_samples_before_predict: int = 64


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -88, 88)
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


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


class MoeSpeqPredictor:
    """Per-(source,target) MoE layer pair predictor with Adam training.

    Architecture per source layer:
      - Fixed random orthogonal projection (hidden_size -> feature_dim) + ReLU
      - Trainable output heads: (feature_dim -> R * num_experts) where R is
        the number of MoE layers after the source layer
      - Per (source, target) ring buffers for training data
      - Double-buffered weights with Adam optimizer
    """

    def __init__(self, metadata: EngineMetadata,
                 config: MoeSpeqPredictorConfig) -> None:
        self._cfg = config
        self._ne = metadata.num_experts
        self._n_moe = metadata.num_moe_layers
        self._n_layers = metadata.num_layers
        self._first_moe = metadata.num_layers - metadata.num_moe_layers
        self._rng = np.random.default_rng(43)

        self._source_layers = tuple(
            range(self._first_moe, self._first_moe + self._n_moe)
        )

        self._target_layers: dict[int, list[int]] = {}
        for src in self._source_layers:
            self._target_layers[src] = [
                layer for layer in self._source_layers if layer > src
            ]

        self._projections: dict[int, np.ndarray] = {}
        for src in self._source_layers:
            raw = self._rng.standard_normal(
                (config.hidden_size, config.feature_dim)).astype(np.float32)
            q, _ = np.linalg.qr(raw)
            self._projections[src] = q[:, :config.feature_dim].astype(np.float32)

        self._weights: list[dict[int, dict[str, np.ndarray]]] = [{}, {}]
        self._adam_m: dict[int, dict[str, np.ndarray]] = {}
        self._adam_v: dict[int, dict[str, np.ndarray]] = {}
        self._adam_step: dict[int, int] = {}
        self._active_buf: dict[int, int] = {}

        self._buffers: dict[tuple[int, int], _RingBuffer] = {}

        for src in self._source_layers:
            R = len(self._target_layers[src])
            if R == 0:
                for b in range(2):
                    self._weights[b][src] = {'w_out': np.empty(0), 'b_out': np.empty(0)}
                self._adam_m[src] = {}
                self._adam_v[src] = {}
            else:
                for b in range(2):
                    self._weights[b][src] = self._init_weights(R)
                self._adam_m[src] = {
                    k: np.zeros_like(v)
                    for k, v in self._weights[0][src].items()
                }
                self._adam_v[src] = {
                    k: np.zeros_like(v)
                    for k, v in self._weights[0][src].items()
                }
            self._adam_step[src] = 0
            self._active_buf[src] = 0

    @property
    def source_layers(self) -> tuple[int, ...]:
        return self._source_layers

    def target_layers_for(self, source_layer: int) -> list[int]:
        return list(self._target_layers.get(source_layer, []))

    def _get_buffer(self, source_layer: int, target_layer: int) -> _RingBuffer:
        key = (source_layer, target_layer)
        buf = self._buffers.get(key)
        if buf is None:
            buf = _RingBuffer(
                self._cfg.training_buffer_size, self._cfg.feature_dim, self._ne)
            self._buffers[key] = buf
        return buf

    def sample_count(self, source_layer: int) -> int:
        targets = self._target_layers.get(source_layer, [])
        if not targets:
            return 0
        counts = []
        for t in targets:
            buf = self._buffers.get((source_layer, t))
            counts.append(buf.count if buf is not None else 0)
        return min(counts)

    def is_warm(self, source_layer: int) -> bool:
        return self.sample_count(source_layer) >= self._cfg.min_samples_before_predict

    def predict(self, source_layer: int,
                hidden_state: np.ndarray) -> dict[int, np.ndarray]:
        if not self.is_warm(source_layer):
            return {}
        targets = self._target_layers.get(source_layer)
        if not targets:
            return {}
        feat = self._extract_features(source_layer, hidden_state)
        w = self._weights[self._active_buf[source_layer]][source_layer]
        logits = feat @ w['w_out'] + w['b_out']
        scores = _sigmoid(logits)
        R = len(targets)
        scores_matrix = scores.reshape(R, self._ne)
        return {tgt: scores_matrix[i] for i, tgt in enumerate(targets)}

    def record_sample(self, source_layer: int, hidden_state: np.ndarray,
                      target_layer: int,
                      actual_expert_mask: np.ndarray) -> None:
        if target_layer not in self._target_layers.get(source_layer, []):
            return
        feat = self._extract_features(source_layer, hidden_state)
        self._get_buffer(source_layer, target_layer).add(
            feat, actual_expert_mask.astype(np.float32))

    def record_samples_batch(
            self, source_layer: int, hidden_state: np.ndarray,
            samples: list[tuple[int, np.ndarray]],
    ) -> None:
        if not samples:
            return
        valid_targets = set(self._target_layers.get(source_layer, []))
        feat = self._extract_features(source_layer, hidden_state)
        for target_layer, expert_mask in samples:
            if target_layer in valid_targets:
                self._get_buffer(source_layer, target_layer).add(
                    feat, expert_mask.astype(np.float32))

    def maybe_train(self, source_layer: int) -> bool:
        targets = self._target_layers.get(source_layer, [])
        if not targets:
            return False
        bs = self._cfg.training_batch_size
        if self.sample_count(source_layer) < bs:
            return False

        active = self._active_buf[source_layer]
        inactive = 1 - active
        w = self._weights[inactive][source_layer]
        for k, v in self._weights[active][source_layer].items():
            w[k] = v.copy()

        ne = self._ne
        grads: dict[str, np.ndarray] = {
            'w_out': np.zeros_like(w['w_out']),
            'b_out': np.zeros_like(w['b_out']),
        }

        for i, tgt in enumerate(targets):
            buf = self._get_buffer(source_layer, tgt)
            X, Y = buf.sample(bs, self._rng)

            w_slice = w['w_out'][:, i * ne:(i + 1) * ne]
            b_slice = w['b_out'][i * ne:(i + 1) * ne]

            logits = X @ w_slice + b_slice
            P = _sigmoid(logits)
            dlogits = (P - Y) / bs

            grads['w_out'][:, i * ne:(i + 1) * ne] = X.T @ dlogits
            grads['b_out'][i * ne:(i + 1) * ne] = dlogits.sum(axis=0)

        cfg = self._cfg
        self._adam_step[source_layer] += 1
        t = self._adam_step[source_layer]
        m = self._adam_m[source_layer]
        v = self._adam_v[source_layer]
        for k in grads:
            m[k] = cfg.beta1 * m[k] + (1 - cfg.beta1) * grads[k]
            v[k] = cfg.beta2 * v[k] + (1 - cfg.beta2) * grads[k] ** 2
            m_hat = m[k] / (1 - cfg.beta1 ** t)
            v_hat = v[k] / (1 - cfg.beta2 ** t)
            w[k] -= cfg.learning_rate * m_hat / (np.sqrt(v_hat) + cfg.epsilon)

        self._active_buf[source_layer] = inactive
        return True

    def _extract_features(self, source_layer: int,
                          hidden_state: np.ndarray) -> np.ndarray:
        proj = self._projections[source_layer]
        z = hidden_state.astype(np.float32) @ proj
        return _relu(z)

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


class MoeSpeq:
    """Online-trained per-(source,target) MoE layer pair predictor."""

    def __init__(self, metadata: EngineMetadata,
                 config: MoeSpeqConfig | None = None,
                 predictor: MoeSpeqPredictor | None = None) -> None:
        self._config = config or MoeSpeqConfig()
        self._predictor = predictor
        self._num_experts = metadata.num_experts
        self._num_layers = metadata.num_layers
        self._first_moe = metadata.num_layers - metadata.num_moe_layers

    @property
    def config(self) -> MoeSpeqConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self._first_moe <= layer_idx < self._num_layers

    def predict(self, source_layer: int,
                hidden_state: np.ndarray) -> list[PrefetchHint]:
        if self._predictor is None:
            return []
        if not self.is_moe_layer(source_layer):
            return []
        predictions = self._predictor.predict(source_layer, hidden_state)
        if not predictions:
            return []
        hints: list[PrefetchHint] = []
        for target_layer, scores in predictions.items():
            hints.extend(self._scores_to_hints(target_layer, scores))
        hints.sort(key=lambda h: h.score, reverse=True)
        return hints

    def record_training_sample(
            self, source_layer: int, hidden_state: np.ndarray,
            target_layer: int, actual_expert_mask: np.ndarray,
    ) -> None:
        if self._predictor is not None:
            self._predictor.record_sample(
                source_layer, hidden_state,
                target_layer, actual_expert_mask)

    def record_training_samples(
            self, source_layer: int, hidden_state: np.ndarray,
            samples: list[tuple[int, np.ndarray]],
    ) -> None:
        if self._predictor is not None:
            self._predictor.record_samples_batch(
                source_layer, hidden_state, samples)

    def _score_to_confidence(self, score: float) -> PrefetchConfidence:
        if score >= self._config.high_confidence_threshold:
            return PrefetchConfidence.HIGH
        if score >= self._config.medium_confidence_threshold:
            return PrefetchConfidence.MEDIUM
        return PrefetchConfidence.LOW

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
                confidence=self._score_to_confidence(sc),
                source=PrefetchSource.MOE_SPEQ,
                score=sc,
            )
            for eidx, sc in agg.items()
        ]
