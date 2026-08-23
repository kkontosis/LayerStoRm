"""Reasoning mode — parameter overrides inside <think>/<​/think> blocks.

Reads is_thinking from ContextAnnotation.reasoning and returns adjusted
speculation/verification parameters. Stateless: the think state tracking
lives in ReasoningModeTracker (context_annotation.py). This module maps
that state to parameter values.

On </think> detection, the next compute_overrides() call returns inactive
overrides — instant revert per INV-4.8.

Spec: IMPLEMENTATION_GUIDE.md §4.8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from orchestrator.context_annotation import ContextAnnotation


@dataclass(frozen=True)
class ReasoningModeConfig:
    enabled: bool = False
    think_token_detection: bool = True
    aggressive_speculation_depth_multiplier: float = 2.0
    relaxed_verification_threshold: float = 0.85


_INACTIVE = None


@dataclass(frozen=True)
class ReasoningOverrides:
    active: bool = False
    depth_multiplier: float = 1.0
    adaptive_topk_threshold: float | None = None
    sparse_verification: bool | None = None


class ReasoningMode:

    __slots__ = ("_config", "_base_topk", "_base_sparse")

    def __init__(
        self,
        config: ReasoningModeConfig | None = None,
        base_adaptive_topk_threshold: float = 0.92,
        base_sparse_verification: bool = False,
    ) -> None:
        self._config = config or ReasoningModeConfig()
        self._base_topk = base_adaptive_topk_threshold
        self._base_sparse = base_sparse_verification

    @property
    def config(self) -> ReasoningModeConfig:
        return self._config

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    def compute_overrides(
        self, annotation: ContextAnnotation,
    ) -> ReasoningOverrides:
        if not self._config.enabled:
            return ReasoningOverrides()

        if not annotation.reasoning.is_thinking:
            return ReasoningOverrides()

        return ReasoningOverrides(
            active=True,
            depth_multiplier=self._config.aggressive_speculation_depth_multiplier,
            adaptive_topk_threshold=self._config.relaxed_verification_threshold,
            sparse_verification=True,
        )

    @staticmethod
    def apply_depth(base_depth: int, overrides: ReasoningOverrides) -> int:
        return math.floor(base_depth * overrides.depth_multiplier)
