"""Eviction policy — stateless scoring engine for expert cache management.

Three togglable scoring methods:
  impact_weighted_lru (default): 7-term weighted sum (spec §4.6)
  lru: recency only
  lfu: 1 - frequency

Optional MoE-SpAc hysteresis: per-expert state machine with asymmetric
promotion/demotion thresholds prevents cache thrashing.
Reference: MoE-SpAc (arXiv 2603.09983).

Reimplements C++ src/core/memory/eviction_policy.{h,cpp}.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from orchestrator.types import (
    CacheZone,
    EvictionPlanEntry,
    EvictionPolicyType,
    ExpertEvictionInput,
    ExpertKey,
)

DUPLICATE_PENALTY: float = 1000.0
STREAMING_ZONE_BONUS: float = 500.0


@dataclass(frozen=True)
class EvictionPolicyConfig:
    policy: EvictionPolicyType = EvictionPolicyType.IMPACT_WEIGHTED_LRU
    alpha_recency: float = 0.4
    beta_frequency: float = 0.35
    gamma_routing_weight: float = 0.25
    delta_temporal_autocorr: float = 0.0
    epsilon_coactivation: float = 0.0
    zeta_prefetch_score: float = 0.0
    eta_hysteresis: float = 0.0
    hysteresis_enabled: bool = False
    hysteresis_stages: int = 5
    hysteresis_up_threshold: int = 5
    hysteresis_down_threshold: int = 1


class EvictionPolicy:
    """Stateless eviction scoring engine.

    Constructed from config once. Called by the orchestrator during PLAN phase
    to score experts and select eviction candidates. Pure scoring function
    (INV-4.6a). Not thread-safe. Single-threaded orchestrator (INV-3.4.2).
    """

    def __init__(self, config: EvictionPolicyConfig | None = None) -> None:
        self._config = config or EvictionPolicyConfig()
        self._score_fn = {
            EvictionPolicyType.IMPACT_WEIGHTED_LRU: self._score_impact_weighted,
            EvictionPolicyType.LRU: self._score_lru,
            EvictionPolicyType.LFU: self._score_lfu,
        }[self._config.policy]

    @property
    def config(self) -> EvictionPolicyConfig:
        return self._config

    def score(self, inp: ExpertEvictionInput) -> float:
        s = self._score_fn(inp)
        if inp.is_duplicate:
            s += DUPLICATE_PENALTY
        return s

    def rank_candidates(
        self, inputs: Sequence[ExpertEvictionInput],
    ) -> list[EvictionPlanEntry]:
        entries = [
            EvictionPlanEntry(
                key=inp.key,
                gpu_idx=inp.gpu_idx,
                zone=inp.zone,
                eviction_score=self.score(inp),
            )
            for inp in inputs
        ]
        entries.sort(key=lambda e: e.eviction_score, reverse=True)
        return entries

    def select_evictions(
        self,
        inputs: Sequence[ExpertEvictionInput],
        gpu_idx: int,
        zone: CacheZone,
        count: int,
    ) -> list[EvictionPlanEntry]:
        if count == 0:
            return []

        candidates: list[EvictionPlanEntry] = []
        for inp in inputs:
            if inp.gpu_idx != gpu_idx:
                continue
            if zone == CacheZone.STREAMING and inp.zone != CacheZone.STREAMING:
                continue

            s = self.score(inp)

            if zone == CacheZone.STABLE and inp.zone == CacheZone.STREAMING:
                s += STREAMING_ZONE_BONUS

            candidates.append(EvictionPlanEntry(
                key=inp.key,
                gpu_idx=inp.gpu_idx,
                zone=inp.zone,
                eviction_score=s,
            ))

        candidates.sort(key=lambda e: e.eviction_score, reverse=True)
        return candidates[:count]

    def score_evictions(
        self,
        inputs: Sequence[ExpertEvictionInput],
        gpu_idx: int,
        zone: CacheZone,
        count: int,
    ) -> list[EvictionPlanEntry]:
        return self.select_evictions(inputs, gpu_idx, zone, count)

    def _score_impact_weighted(self, inp: ExpertEvictionInput) -> float:
        cfg = self._config
        return (cfg.alpha_recency * inp.recency
                - cfg.beta_frequency * inp.frequency
                - cfg.gamma_routing_weight * inp.routing_weight
                - cfg.delta_temporal_autocorr * inp.temporal_autocorr
                - cfg.epsilon_coactivation * inp.coactivation
                - cfg.zeta_prefetch_score * inp.prefetch_score
                - cfg.eta_hysteresis * inp.hysteresis_state)

    @staticmethod
    def _score_lru(inp: ExpertEvictionInput) -> float:
        return inp.recency

    @staticmethod
    def _score_lfu(inp: ExpertEvictionInput) -> float:
        return 1.0 - inp.frequency


class HysteresisTracker:
    """Per-expert state machine for anti-thrash cache priority.

    Each expert has an integer state in [0, stages]. State transitions
    use asymmetric thresholds to prevent oscillation:
      - Promote: activation_count - state >= up_threshold -> state += 1
      - Demote:  state - activation_count >= down_threshold -> state -= 1
      - Otherwise: no change (hysteresis band)

    Reference: MoE-SpAc (arXiv 2603.09983), StateMachine class in
    ref/MoE-SpAc/ggml/src/ggml-backend.cpp.
    """

    def __init__(self, config: EvictionPolicyConfig) -> None:
        self._stages = config.hysteresis_stages
        self._up = config.hysteresis_up_threshold
        self._down = config.hysteresis_down_threshold
        self._states: dict[ExpertKey, int] = {}

    @property
    def stages(self) -> int:
        return self._stages

    def update(self, key: ExpertKey, activation_count: int) -> int:
        state = self._states.get(key, 0)
        diff = activation_count - state
        if diff >= self._up:
            state = min(self._stages, state + 1)
        elif -diff >= self._down:
            state = max(0, state - 1)
        self._states[key] = state
        return state

    def state(self, key: ExpertKey) -> int:
        return self._states.get(key, 0)

    def normalized_state(self, key: ExpertKey) -> float:
        if self._stages <= 0:
            return 0.0
        return self._states.get(key, 0) / self._stages

    def reset(self) -> None:
        self._states.clear()
