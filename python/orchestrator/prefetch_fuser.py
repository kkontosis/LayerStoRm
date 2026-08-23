"""Prefetch fuser — combines PreScope, PROBE, MoE-SpeQ, and SP-MoE signals.

Stateless weighted fusion of prediction signals into unified PrefetchPriority
per (expert, layer). Time-aware decay discounts distant-layer predictions.
Formula: fused = α×prescope + β×probe + γ×speq + δ×sp_moe, then
priority = fused / (1 + layers_ahead × time_decay).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from orchestrator.types import (
    ExpertKey,
    PrefetchHint,
    PrefetchPriority,
    PrefetchSource,
)


_SOURCE_KEY = {
    PrefetchSource.PRESCOPE: "prescope",
    PrefetchSource.PROBE: "probe",
    PrefetchSource.MOE_SPEQ: "speq",
    PrefetchSource.SP_MOE: "sp_moe",
}


@dataclass(frozen=True)
class FuserConfig:
    """Fusion weights and time-aware decay parameters."""
    prescope_alpha: float = 0.5
    probe_beta: float = 0.3
    speq_gamma: float = 0.2
    sp_moe_delta: float = 0.15
    time_decay: float = 0.1
    per_layer_us: int = 920


class PrefetchFuser:
    """Fuses prefetch signals into unified PrefetchPriority list."""

    def __init__(self, config: FuserConfig | None = None) -> None:
        self._config = config or FuserConfig()

    @property
    def config(self) -> FuserConfig:
        return self._config

    def fuse(
            self,
            prescope_hints: list[PrefetchHint],
            probe_hints: list[PrefetchHint],
            speq_hints: list[PrefetchHint],
            sp_moe_hints: list[PrefetchHint],
            current_layer: int = 0,
    ) -> list[PrefetchPriority]:
        by_key: dict[ExpertKey, dict[str, float]] = defaultdict(dict)

        for hint in prescope_hints:
            sk = _SOURCE_KEY.get(hint.source, "prescope")
            prev = by_key[hint.key].get(sk, 0.0)
            by_key[hint.key][sk] = max(prev, hint.score)

        for hint in probe_hints:
            sk = _SOURCE_KEY.get(hint.source, "probe")
            prev = by_key[hint.key].get(sk, 0.0)
            by_key[hint.key][sk] = max(prev, hint.score)

        for hint in speq_hints:
            sk = _SOURCE_KEY.get(hint.source, "speq")
            prev = by_key[hint.key].get(sk, 0.0)
            by_key[hint.key][sk] = max(prev, hint.score)

        for hint in sp_moe_hints:
            sk = _SOURCE_KEY.get(hint.source, "sp_moe")
            prev = by_key[hint.key].get(sk, 0.0)
            by_key[hint.key][sk] = max(prev, hint.score)

        cfg = self._config
        weights = {
            "prescope": cfg.prescope_alpha,
            "probe": cfg.probe_beta,
            "speq": cfg.speq_gamma,
            "sp_moe": cfg.sp_moe_delta,
        }

        results: list[PrefetchPriority] = []
        for key, scores in by_key.items():
            raw = sum(weights.get(sk, 0.0) * sc for sk, sc in scores.items())
            layers_ahead = max(0, key.layer_idx - current_layer)
            decay = 1.0 / (1.0 + layers_ahead * cfg.time_decay)
            priority = min(max(raw * decay, 0.0), 1.0)
            results.append(PrefetchPriority(
                key=key,
                target_layer=key.layer_idx,
                priority_score=priority,
                estimated_time_until_needed_us=layers_ahead * cfg.per_layer_us,
                source_scores=dict(scores),
            ))

        results.sort(key=lambda p: p.priority_score, reverse=True)
        return results
