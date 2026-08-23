"""Layer skip — cosine similarity-based layer skipping during self-speculative draft.

When cos_sim(hidden_before, hidden_after) > threshold for a layer, the layer
barely changes the representation. If enough evidence accumulates (contiguous
window and/or product gate), the next layer is skipped.

No pretrained model or classifier needed — pure runtime heuristic exploiting
transformer layer redundancy (Deja Vu, LayerSkip, CLaSp papers).

Only active during self-speculative draft (INV-4.15a). Skip decisions are
communicated to the prefetch pipeline to suppress expert prefetch for skipped
layers (INV-4.15b).

Reference: LayerSkip (Elhoushi et al., Meta FAIR 2024),
           CLaSp (Chen et al., 2025).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayerSkipConfig:
    enabled: bool = False
    threshold: float = 0.995
    no_skip_first: int = 3
    no_skip_last: int = 3
    min_contiguous: int = 1
    product_window: int = 1
    product_threshold: float = 0.985
    min_acceptance_rate: float = 0.5


class LayerSkip:

    def __init__(self, config: LayerSkipConfig | None = None,
                 num_layers: int = 61) -> None:
        self._config = config or LayerSkipConfig()
        self._num_layers = num_layers

    def is_skippable(self, layer_idx: int) -> bool:
        if layer_idx < self._config.no_skip_first:
            return False
        if layer_idx >= self._num_layers - self._config.no_skip_last:
            return False
        return True

    def compute_skip_set(self, similarities: dict[int, float]) -> set[int]:
        if not similarities:
            return set()

        cfg = self._config
        skip_set: set[int] = set()

        for layer in range(self._num_layers - 1):
            candidate = layer + 1
            if not self.is_skippable(candidate):
                continue

            if not self._passes_contiguous_gate(layer, similarities, cfg):
                continue

            if cfg.product_window > 1:
                if not self._passes_product_gate(layer, similarities, cfg):
                    continue

            skip_set.add(candidate)

        return skip_set

    def _passes_contiguous_gate(self, layer: int,
                                similarities: dict[int, float],
                                cfg: LayerSkipConfig) -> bool:
        for offset in range(cfg.min_contiguous):
            check_layer = layer - offset
            if check_layer < 0:
                return False
            if not self.is_skippable(check_layer):
                return False
            sim = similarities.get(check_layer)
            if sim is None or sim < cfg.threshold:
                return False
        return True

    def _passes_product_gate(self, layer: int,
                             similarities: dict[int, float],
                             cfg: LayerSkipConfig) -> bool:
        product = 1.0
        for offset in range(cfg.product_window):
            check_layer = layer - offset
            if check_layer < 0:
                return False
            sim = similarities.get(check_layer)
            if sim is None:
                return False
            product *= sim
        return product >= cfg.product_threshold

    def should_enable(self, acceptance_rate: float) -> bool:
        if not self._config.enabled:
            return False
        return acceptance_rate >= self._config.min_acceptance_rate

    @staticmethod
    def cosine_similarity(before: np.ndarray, after: np.ndarray) -> float:
        norm_b = np.linalg.norm(before)
        norm_a = np.linalg.norm(after)
        if norm_b == 0.0 or norm_a == 0.0:
            return 0.0
        return float(np.dot(before, after) / (norm_b * norm_a))

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    @property
    def threshold(self) -> float:
        return self._config.threshold

    @property
    def no_skip_set(self) -> set[int]:
        result: set[int] = set()
        for i in range(self._config.no_skip_first):
            result.add(i)
        for i in range(self._num_layers - self._config.no_skip_last,
                       self._num_layers):
            result.add(i)
        return result
