"""Residual correction — online-trained MLP per MoE layer for self-speculative draft.

Predicts `full_output - top_1_output` from the top-1 expert output and gating
scores. Applied during self-speculative draft (#52) to improve draft quality
without loading additional experts.

At draft time:
  draft_output = top_1_expert_output + residual_correction_mlp(top_1_output, gating_scores)

Training data collected during full verification:
  input  = concat(top_1_output, gating_scores)
  target = full_output - top_1_output

Double-buffered weights (INV-4.17a): train shadow copy, atomically swap pointer.
No locks on inference hot path.

Reference: IMPLEMENTATION_GUIDE.md §4.7.2.
Paper: Kangaroo — Lossless Self-Speculative Decoding with Double Early Exiting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualCorrectionConfig:
    enabled: bool = True
    hidden_size: int = 128
    learning_rate: float = 5e-5
    training_buffer_size: int = 500
    training_batch_size: int = 32
    min_samples_before_predict: int = 64
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

    def sample(self, n: int,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        high = min(self.count, self.capacity)
        idx = rng.integers(0, high, size=min(n, high))
        return self.inputs[idx], self.labels[idx]


class ResidualCorrection:

    def __init__(self, config: ResidualCorrectionConfig | None = None,
                 num_moe_layers: int = 58,
                 output_dim: int = 2048,
                 num_experts: int = 256) -> None:
        self._config = config or ResidualCorrectionConfig()
        self._num_moe_layers = num_moe_layers
        self._output_dim = output_dim
        self._num_experts = num_experts
        self._input_dim = output_dim + num_experts
        self._hidden = self._config.hidden_size
        self._rng = np.random.default_rng(42)

        self._weights: list[dict[int, dict[str, np.ndarray]]] = [{}, {}]
        self._adam_m: dict[int, dict[str, np.ndarray]] = {}
        self._adam_v: dict[int, dict[str, np.ndarray]] = {}
        self._adam_step: dict[int, int] = {}
        self._active_buf: dict[int, int] = {}
        self._buffers: dict[int, _RingBuffer] = {}
        self._train_count: dict[int, int] = {}

        for layer in range(num_moe_layers):
            w_init = self._init_weights()
            for b in range(2):
                self._weights[b][layer] = {
                    k: v.copy() for k, v in w_init.items()
                }
            self._adam_m[layer] = {
                k: np.zeros_like(v) for k, v in w_init.items()
            }
            self._adam_v[layer] = {
                k: np.zeros_like(v) for k, v in w_init.items()
            }
            self._adam_step[layer] = 0
            self._active_buf[layer] = 0
            self._train_count[layer] = 0
            self._buffers[layer] = _RingBuffer(
                self._config.training_buffer_size,
                self._input_dim,
                output_dim,
            )

    def _init_weights(self) -> dict[str, np.ndarray]:
        d_in = self._input_dim
        h = self._hidden
        d_out = self._output_dim
        limit1 = np.sqrt(6.0 / (d_in + h))
        limit2 = np.sqrt(6.0 / (h + d_out))
        return {
            'w1': self._rng.uniform(-limit1, limit1,
                                     (d_in, h)).astype(np.float32),
            'b1': np.zeros(h, dtype=np.float32),
            'w2': self._rng.uniform(-limit2, limit2,
                                     (h, d_out)).astype(np.float32),
            'b2': np.zeros(d_out, dtype=np.float32),
        }

    def predict(self, layer_idx: int, top1_output: np.ndarray,
                gating_scores: np.ndarray) -> np.ndarray:
        if not self._config.enabled or not self.is_warm(layer_idx):
            return np.zeros(self._output_dim, dtype=np.float32)

        x = np.concatenate([top1_output, gating_scores]).astype(np.float32)
        w = self._weights[self._active_buf[layer_idx]][layer_idx]
        hidden = np.maximum(0.0, x @ w['w1'] + w['b1'])
        return (hidden @ w['w2'] + w['b2']).astype(np.float32)

    def record_sample(self, layer_idx: int, top1_output: np.ndarray,
                      gating_scores: np.ndarray,
                      full_output: np.ndarray) -> None:
        x = np.concatenate([top1_output, gating_scores]).astype(np.float32)
        target = (full_output - top1_output).astype(np.float32)
        self._buffers[layer_idx].add(x, target)

    def maybe_train(self, layer_idx: int) -> bool:
        bs = self._config.training_batch_size
        if self.sample_count(layer_idx) < bs:
            return False

        active = self._active_buf[layer_idx]
        inactive = 1 - active
        w = self._weights[inactive][layer_idx]
        for k, v in self._weights[active][layer_idx].items():
            w[k] = v.copy()

        buf = self._buffers[layer_idx]
        X, Y = buf.sample(bs, self._rng)

        hidden = np.maximum(0.0, X @ w['w1'] + w['b1'])
        pred = hidden @ w['w2'] + w['b2']

        diff = pred - Y
        d_out = diff / bs

        d_hidden = d_out @ w['w2'].T
        d_hidden = d_hidden * (hidden > 0).astype(np.float32)

        grads = {
            'w2': hidden.T @ d_out,
            'b2': d_out.sum(axis=0),
            'w1': X.T @ d_hidden,
            'b1': d_hidden.sum(axis=0),
        }

        cfg = self._config
        self._adam_step[layer_idx] += 1
        t = self._adam_step[layer_idx]
        m = self._adam_m[layer_idx]
        v = self._adam_v[layer_idx]
        for k in grads:
            m[k] = cfg.beta1 * m[k] + (1 - cfg.beta1) * grads[k]
            v[k] = cfg.beta2 * v[k] + (1 - cfg.beta2) * grads[k] ** 2
            m_hat = m[k] / (1 - cfg.beta1 ** t)
            v_hat = v[k] / (1 - cfg.beta2 ** t)
            w[k] -= cfg.learning_rate * m_hat / (np.sqrt(v_hat) + cfg.epsilon)

        self._active_buf[layer_idx] = inactive
        self._train_count[layer_idx] += 1
        return True

    def is_warm(self, layer_idx: int) -> bool:
        return (self._buffers[layer_idx].count
                >= self._config.min_samples_before_predict)

    def sample_count(self, layer_idx: int) -> int:
        return self._buffers[layer_idx].count

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    @property
    def num_layers_trained(self) -> int:
        return sum(1 for c in self._train_count.values() if c > 0)
