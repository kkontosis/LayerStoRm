"""Online calibrator — auto-adjusts draft speculation parameters based on acceptance rates.

Adjustment priority on low acceptance (below min_acceptance_rate):
  1. Disable layer_skip_enabled (larger quality impact per unit speed)
  2. Increase draft_expert_count (up to max)

Adjustment priority on high acceptance (above target_acceptance_rate):
  1. Decrease draft_expert_count (down to 1)
  2. Enable layer_skip_enabled (if count already at minimum)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationConfig:
    min_acceptance_rate: float = 0.3
    target_acceptance_rate: float = 0.6
    adjustment_interval_tokens: int = 500
    acceptance_ema_alpha: float = 0.5


@dataclass
class DraftParams:
    draft_expert_count: int = 1
    layer_skip_enabled: bool = False
    max_draft_expert_count: int = 8
    min_draft_expert_count: int = 1


class OnlineCalibrator:

    def __init__(self, config: CalibrationConfig | None = None,
                 params: DraftParams | None = None) -> None:
        self._config = config or CalibrationConfig()
        self._params = params or DraftParams()
        self._ema_rate: float = self._config.target_acceptance_rate
        self._tokens_since_adjust: int = 0
        self._adjustments_made: int = 0
        self._initialized: bool = False

    def update(self, acceptance_rate: float, tokens_generated: int) -> bool:
        if not self._initialized:
            self._ema_rate = acceptance_rate
            self._initialized = True
        else:
            a = self._config.acceptance_ema_alpha
            self._ema_rate = a * acceptance_rate + (1.0 - a) * self._ema_rate

        self._tokens_since_adjust += tokens_generated
        if self._tokens_since_adjust < self._config.adjustment_interval_tokens:
            return False
        self._tokens_since_adjust = 0

        adjusted = False
        if self._ema_rate < self._config.min_acceptance_rate:
            adjusted = self._reduce_aggressiveness()
        elif self._ema_rate > self._config.target_acceptance_rate:
            adjusted = self._increase_aggressiveness()

        if adjusted:
            self._adjustments_made += 1
        return adjusted

    def _reduce_aggressiveness(self) -> bool:
        if self._params.layer_skip_enabled:
            self._params.layer_skip_enabled = False
            return True
        if self._params.draft_expert_count < self._params.max_draft_expert_count:
            self._params.draft_expert_count += 1
            return True
        return False

    def _increase_aggressiveness(self) -> bool:
        if self._params.draft_expert_count > self._params.min_draft_expert_count:
            self._params.draft_expert_count -= 1
            return True
        if not self._params.layer_skip_enabled:
            self._params.layer_skip_enabled = True
            return True
        return False

    @property
    def smoothed_acceptance_rate(self) -> float:
        return self._ema_rate

    @property
    def params(self) -> DraftParams:
        return self._params

    @property
    def adjustments_made(self) -> int:
        return self._adjustments_made
