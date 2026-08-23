"""Configurable latency/throughput objective function for scheduler decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveConfig:
    latency_weight: float = 0.6
    throughput_weight: float = 0.4


class PerformanceObjective:

    def __init__(self, config: ObjectiveConfig | None = None) -> None:
        if config is None:
            config = ObjectiveConfig()
        total = config.latency_weight + config.throughput_weight
        if total == 0.0:
            self._lw = 0.5
            self._tw = 0.5
        else:
            self._lw = config.latency_weight / total
            self._tw = config.throughput_weight / total

    def score(self, latency_score: float, throughput_score: float) -> float:
        return self._lw * latency_score + self._tw * throughput_score

    @property
    def latency_weight(self) -> float:
        return self._lw

    @property
    def throughput_weight(self) -> float:
        return self._tw

    @property
    def is_latency_dominant(self) -> bool:
        return self._lw > self._tw

    @property
    def is_throughput_dominant(self) -> bool:
        return self._tw > self._lw
