"""PreScope — exact gating lookahead with optional LLaPor learned predictor.

Two modes: (1) exact gating processes daemon-computed gating output into
PrefetchHints (HIGH confidence), (2) LLaPor-style learned predictor runs
CPU-side MLP inference for zero-GPU-cost prediction (MEDIUM confidence).
Both are pure signal producers — no GPU commands, no expert transfers.

Reference: Yu et al., "PreScope: Unleashing the Power of Prefetching for
Resource-Constrained MoE Inference", arXiv:2509.23638.
"""

from __future__ import annotations

import enum
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
class PrescopeConfig:
    """Inline config (prefetch.prescope in main config)."""
    enabled: bool = True
    top_k: int = 8
    score_threshold: float = 0.01
    predictor_enabled: bool = False


@dataclass(frozen=True)
class PredictorConfig:
    """Internal config (_internal-prescope, loaded from prescope.json)."""
    hidden_size: int = 7168
    pca_dim: int = 256
    hidden_dim: int = 256
    learning_rate: float = 1e-3
    momentum: float = 0.9
    training_batch_size: int = 32
    training_buffer_size: int = 2048
    focal_loss_gamma: float = 2.0
    balance_loss_lambda: float = 0.5
    input_group_frac: float = 0.25
    output_group_frac: float = 0.25
    min_samples_before_predict: int = 64
    freq_ema_alpha: float = 0.01


class LayerGroup(enum.IntEnum):
    INPUT = 0
    MIDDLE = 1
    OUTPUT = 2


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -88, 88)
    return 1.0 / (1.0 + np.exp(-x))


def _gelu(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(1.702 * x)
    return x * s


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(1.702 * x)
    return s + x * 1.702 * s * (1.0 - s)


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


class PrescopePredictor:
    """LLaPor-style learned predictor for 1-layer expert lookahead."""

    def __init__(self, metadata: EngineMetadata,
                 config: PredictorConfig) -> None:
        self._cfg = config
        self._ne = metadata.num_experts
        self._n_moe = metadata.num_moe_layers
        self._first_moe = metadata.num_layers - metadata.num_moe_layers
        self._input_dim = config.pca_dim + self._ne
        self._rng = np.random.default_rng(42)

        n_input = max(1, int(self._n_moe * config.input_group_frac))
        n_output = max(1, int(self._n_moe * config.output_group_frac))
        self._input_end = self._first_moe + n_input
        self._output_start = self._first_moe + self._n_moe - n_output

        self._projections: dict[LayerGroup, np.ndarray] = {}
        for g in LayerGroup:
            raw = self._rng.standard_normal(
                (config.hidden_size, config.pca_dim)).astype(np.float32)
            q, _ = np.linalg.qr(raw)
            self._projections[g] = q[:, :config.pca_dim].astype(np.float32)

        self._weights: list[dict[int, dict[str, np.ndarray]]] = [{}, {}]
        self._velocity: dict[int, dict[str, np.ndarray]] = {}
        self._active_buf = 0
        self._buffers: dict[int, _RingBuffer] = {}
        self._expert_freq: dict[int, np.ndarray] = {}

        for layer in range(self._first_moe, self._first_moe + self._n_moe):
            group = self.layer_group(layer)
            for b in range(2):
                self._weights[b][layer] = self._init_weights(group)
            self._velocity[layer] = {
                k: np.zeros_like(v)
                for k, v in self._weights[0][layer].items()
            }
            self._buffers[layer] = _RingBuffer(
                config.training_buffer_size, self._input_dim, self._ne)
            self._expert_freq[layer] = np.full(self._ne, 1.0 / self._ne,
                                               dtype=np.float32)

    def layer_group(self, layer_idx: int) -> LayerGroup:
        if layer_idx < self._input_end:
            return LayerGroup.INPUT
        if layer_idx >= self._output_start:
            return LayerGroup.OUTPUT
        return LayerGroup.MIDDLE

    def sample_count(self, layer_idx: int) -> int:
        buf = self._buffers.get(layer_idx)
        return buf.count if buf is not None else 0

    def is_warm(self, layer_idx: int) -> bool:
        return self.sample_count(layer_idx) >= self._cfg.min_samples_before_predict

    def predict(self, layer_idx: int, hidden_state: np.ndarray,
                prev_routing_weights: np.ndarray | None = None) -> np.ndarray:
        if not self.is_warm(layer_idx):
            return np.zeros(self._ne, dtype=np.float32)
        x = self._build_input(layer_idx, hidden_state, prev_routing_weights)
        w = self._weights[self._active_buf][layer_idx]
        return self._forward(w, x, self.layer_group(layer_idx))

    def record_sample(self, layer_idx: int, hidden_state: np.ndarray,
                      actual_expert_mask: np.ndarray,
                      prev_routing_weights: np.ndarray | None = None) -> None:
        buf = self._buffers.get(layer_idx)
        if buf is None:
            return
        x = self._build_input(layer_idx, hidden_state, prev_routing_weights)
        buf.add(x, actual_expert_mask.astype(np.float32))
        alpha = self._cfg.freq_ema_alpha
        self._expert_freq[layer_idx] = (
            (1 - alpha) * self._expert_freq[layer_idx]
            + alpha * actual_expert_mask.astype(np.float32)
        )

    def maybe_train(self, layer_idx: int) -> bool:
        buf = self._buffers.get(layer_idx)
        if buf is None or buf.count < self._cfg.training_batch_size:
            return False
        X, Y = buf.sample(self._cfg.training_batch_size, self._rng)
        inactive = 1 - self._active_buf
        w = self._weights[inactive][layer_idx]
        for k, v in self._weights[self._active_buf][layer_idx].items():
            w[k] = v.copy()

        group = self.layer_group(layer_idx)
        grads = self._train_step(w, X, Y, group, layer_idx)

        vel = self._velocity[layer_idx]
        lr = self._cfg.learning_rate
        mu = self._cfg.momentum
        for k in grads:
            vel[k] = mu * vel[k] + grads[k]
            w[k] = w[k] - lr * vel[k]

        self._active_buf = inactive
        return True

    def _build_input(self, layer_idx: int, hidden_state: np.ndarray,
                     prev_routing_weights: np.ndarray | None) -> np.ndarray:
        group = self.layer_group(layer_idx)
        proj = self._projections[group]
        x_pca = hidden_state.astype(np.float32) @ proj
        if prev_routing_weights is not None:
            rv = prev_routing_weights.astype(np.float32)
        else:
            rv = np.zeros(self._ne, dtype=np.float32)
        return np.concatenate([x_pca, rv])

    def _forward(self, w: dict[str, np.ndarray], x: np.ndarray,
                 group: LayerGroup) -> np.ndarray:
        is_batch = x.ndim == 2
        if not is_batch:
            x = x[np.newaxis, :]
        z1 = x @ w['w1'] + w['b1']
        h = _gelu(z1)
        if group == LayerGroup.MIDDLE:
            rz1 = h @ w['res_w1'] + w['res_b1']
            rh1 = _gelu(rz1)
            rz2 = rh1 @ w['res_w2'] + w['res_b2']
            h_res = h + rz2
            x_pca = x[:, :self._cfg.pca_dim]
            gate = _sigmoid(x_pca @ w['gate_w'] + w['gate_b'])
            h = gate * h_res
        logits = h @ w['w2'] + w['b2']
        out = _sigmoid(logits)
        if not is_batch:
            return out[0]
        return out

    def _train_step(self, w: dict[str, np.ndarray],
                    X: np.ndarray, Y: np.ndarray,
                    group: LayerGroup,
                    layer_idx: int) -> dict[str, np.ndarray]:
        B = X.shape[0]
        pca_dim = self._cfg.pca_dim

        z1 = X @ w['w1'] + w['b1']
        h1 = _gelu(z1)
        is_mid = group == LayerGroup.MIDDLE

        if is_mid:
            rz1 = h1 @ w['res_w1'] + w['res_b1']
            rh1 = _gelu(rz1)
            rz2 = rh1 @ w['res_w2'] + w['res_b2']
            h_res = h1 + rz2
            x_pca = X[:, :pca_dim]
            gz = x_pca @ w['gate_w'] + w['gate_b']
            gate = _sigmoid(gz)
            h_out = gate * h_res
        else:
            h_out = h1

        logits = h_out @ w['w2'] + w['b2']
        P = _sigmoid(logits)

        P_t = Y * P + (1 - Y) * (1 - P)
        focal_w = np.power(np.clip(1 - P_t, 0, 1), self._cfg.focal_loss_gamma)
        freq = self._expert_freq[layer_idx]
        balance_w = 1.0 / (freq + 1e-8)
        cw = balance_w + self._cfg.balance_loss_lambda * focal_w
        dlogits = cw * (P - Y) / B

        g: dict[str, np.ndarray] = {}
        g['w2'] = h_out.T @ dlogits
        g['b2'] = dlogits.sum(axis=0)
        dh = dlogits @ w['w2'].T

        if is_mid:
            dgate = dh * h_res
            dh_res = dh * gate
            dgz = dgate * gate * (1 - gate)
            g['gate_w'] = x_pca.T @ dgz
            g['gate_b'] = dgz.sum(axis=0)

            drz2 = dh_res
            drh1 = drz2 @ w['res_w2'].T
            g['res_w2'] = rh1.T @ drz2
            g['res_b2'] = drz2.sum(axis=0)
            drz1 = drh1 * _gelu_grad(rz1)
            g['res_w1'] = h1.T @ drz1
            g['res_b1'] = drz1.sum(axis=0)
            dh1 = dh_res + drz1 @ w['res_w1'].T
        else:
            dh1 = dh

        dz1 = dh1 * _gelu_grad(z1)
        g['w1'] = X.T @ dz1
        g['b1'] = dz1.sum(axis=0)
        return g

    def _init_weights(self, group: LayerGroup) -> dict[str, np.ndarray]:
        hd = self._cfg.hidden_dim
        idim = self._input_dim
        ne = self._ne
        pca = self._cfg.pca_dim

        def _he(fan_in: int, fan_out: int) -> np.ndarray:
            return (self._rng.standard_normal((fan_in, fan_out)).astype(np.float32)
                    * np.float32(np.sqrt(2.0 / fan_in)))

        w: dict[str, np.ndarray] = {
            'w1': _he(idim, hd),
            'b1': np.zeros(hd, dtype=np.float32),
            'w2': _he(hd, ne),
            'b2': np.zeros(ne, dtype=np.float32),
        }
        if group == LayerGroup.MIDDLE:
            w['res_w1'] = _he(hd, hd)
            w['res_b1'] = np.zeros(hd, dtype=np.float32)
            w['res_w2'] = _he(hd, hd)
            w['res_b2'] = np.zeros(hd, dtype=np.float32)
            w['gate_w'] = _he(pca, hd)
            w['gate_b'] = np.zeros(hd, dtype=np.float32)
        return w


class PreScope:
    """Exact gating lookahead with optional LLaPor learned predictor."""

    def __init__(self, metadata: EngineMetadata,
                 config: PrescopeConfig | None = None,
                 predictor: PrescopePredictor | None = None) -> None:
        self._config = config or PrescopeConfig()
        self._predictor = predictor
        self._num_experts = metadata.num_experts
        self._num_layers = metadata.num_layers
        self._first_moe = metadata.num_layers - metadata.num_moe_layers

    @property
    def config(self) -> PrescopeConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def first_moe_layer(self) -> int:
        return self._first_moe

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self._first_moe <= layer_idx < self._num_layers

    def target_layer(self, completed_attention_layer: int) -> int | None:
        nxt = completed_attention_layer + 1
        if self.is_moe_layer(nxt):
            return nxt
        return None

    def process_gating_output(self, target_layer: int,
                              gating_scores: np.ndarray) -> list[PrefetchHint]:
        return self._scores_to_hints(target_layer, gating_scores,
                                     PrefetchConfidence.HIGH)

    def predict_from_hidden_state(
            self, layer_idx: int, hidden_state: np.ndarray,
            prev_routing_weights: np.ndarray | None = None,
    ) -> list[PrefetchHint]:
        if self._predictor is None:
            return []
        target = layer_idx + 1
        if not self.is_moe_layer(target):
            return []
        scores = self._predictor.predict(layer_idx, hidden_state,
                                         prev_routing_weights)
        if not np.any(scores > 0):
            return []
        return self._scores_to_hints(target, scores, PrefetchConfidence.MEDIUM)

    def record_training_sample(
            self, layer_idx: int, hidden_state: np.ndarray,
            actual_expert_mask: np.ndarray,
            prev_routing_weights: np.ndarray | None = None,
    ) -> None:
        if self._predictor is not None:
            self._predictor.record_sample(layer_idx, hidden_state,
                                          actual_expert_mask,
                                          prev_routing_weights)

    def _scores_to_hints(self, target_layer: int, scores: np.ndarray,
                         confidence: PrefetchConfidence) -> list[PrefetchHint]:
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
        hints = [
            PrefetchHint(
                key=ExpertKey(target_layer, eidx),
                target_layer=target_layer,
                confidence=confidence,
                source=PrefetchSource.PRESCOPE,
                score=sc,
            )
            for eidx, sc in agg.items()
        ]
        hints.sort(key=lambda h: h.score, reverse=True)
        return hints
