"""Utility scorer — adaptive speculation depth control.

Three strategies selectable via config:

  cascade (default): Three-phase test-and-set from the Lookahead paper
    (Yang et al.). Hill-climbing K selection with adaptive back-off.
    Utility = ETR / cost_ratio (Eq. 2-4). R²=99.4% TPOT prediction.

  ema: sglang-style EMA of acceptance length. Adjusts K from candidate
    set based on smoothed acceptance. Simpler, not cost-aware.
    Ref: sglang adaptive_spec_params.py lines 148-179.

  static: Fixed K from config. No adaptation. Useful for benchmarking.

Reference: Lookahead Utility-Driven Speculative Decoding for MoE
           (lines 867-946, 1223-1228, 1293-1306, 1310-1318).
Spec: IMPLEMENTATION_GUIDE.md §4.7, §4.7.4 INV-A.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


@dataclass(frozen=True)
class UtilityScorerConfig:
    enabled: bool = True
    strategy: str = "cascade"  # "cascade", "ema", "fixed"
    static_depth: int = 3
    utility_threshold: float = 1.0
    vram_headroom_fraction: float = 0.1
    candidate_depths: tuple[int, ...] = (0, 1, 2, 3, 5, 7)
    test_iterations: int = 4
    set_iterations: int = 16
    convergence_tolerance: float = 0.1
    backoff_multiplier: int = 2
    ema_alpha: float = 0.3
    ema_up_threshold: float = 0.0
    ema_down_threshold: float = -0.25
    coverage_discount_floor: float = 0.5


class _Phase(enum.Enum):
    TEST = "test"
    SET = "set"


class UtilityScorer:

    def __init__(self, config: UtilityScorerConfig | None = None) -> None:
        self._config = config or UtilityScorerConfig()
        candidates = self._config.candidate_depths
        self._candidates = sorted(candidates)

        # Cascade state
        self._phase = _Phase.TEST
        self._phase_iter: int = 0
        self._test_idx: int = 0
        self._test_results: dict[int, float] = {}
        self._test_counts: dict[int, int] = {}
        self._best_k: int = self._candidates[0] if self._candidates else 0
        self._set_duration: int = self._config.set_iterations
        self._hill_direction: int = 1
        self._prev_test_k: int | None = None
        self._prev_test_utility: float | None = None
        self._per_k_ema: dict[int, float] = {}

        # EMA state (sglang-style)
        self._ema_accept_len: float = 0.0
        self._ema_initialized: bool = False
        self._ema_k: int = self._candidates[len(self._candidates) // 2] if self._candidates else 1

    @staticmethod
    def compute_utility(etr: float, cost_ratio: float) -> float:
        if cost_ratio <= 0.0:
            return 0.0
        return etr / cost_ratio

    def record_iteration(self, depth_used: int, tokens_accepted: int,
                         draft_time_us: float, verify_time_us: float,
                         base_time_us: float) -> None:
        strategy = self._config.strategy

        if strategy == "fixed":
            return

        if strategy == "ema":
            self._record_ema(tokens_accepted)
            return

        # cascade (default)
        etr = 1.0 + tokens_accepted
        spec_time = draft_time_us + verify_time_us
        cost_ratio = spec_time / base_time_us if base_time_us > 0 else 1.0
        utility = self.compute_utility(etr, cost_ratio)

        alpha = self._config.ema_alpha
        if depth_used in self._per_k_ema:
            self._per_k_ema[depth_used] = (
                alpha * utility + (1.0 - alpha) * self._per_k_ema[depth_used]
            )
        else:
            self._per_k_ema[depth_used] = utility

        self._phase_iter += 1

        if self._phase == _Phase.TEST:
            self._advance_test(depth_used, utility)
        elif self._phase == _Phase.SET:
            if self._phase_iter >= self._set_duration:
                self._start_test()

    def _record_ema(self, tokens_accepted: int) -> None:
        alpha = self._config.ema_alpha
        if not self._ema_initialized:
            self._ema_accept_len = float(tokens_accepted)
            self._ema_initialized = True
        else:
            self._ema_accept_len = (
                alpha * tokens_accepted
                + (1.0 - alpha) * self._ema_accept_len
            )

        target = round(self._ema_accept_len) + 1
        try:
            idx = self._candidates.index(self._ema_k)
        except ValueError:
            idx = 0

        if target > self._ema_k + self._config.ema_up_threshold and idx + 1 < len(self._candidates):
            self._ema_k = self._candidates[idx + 1]
        elif target < self._ema_k + self._config.ema_down_threshold and idx > 0:
            self._ema_k = self._candidates[idx - 1]

    def _advance_test(self, k: int, utility: float) -> None:
        self._test_results[k] = self._per_k_ema.get(k, utility)
        self._test_counts[k] = self._test_counts.get(k, 0) + 1

        if self._test_counts.get(k, 0) < self._config.test_iterations:
            return

        if self._should_early_exit():
            self._finish_test()
            return

        next_k = self._next_test_k(k, utility)
        if next_k is None:
            self._finish_test()
            return

        self._prev_test_k = k
        self._prev_test_utility = utility

    def _next_test_k(self, current_k: int, current_utility: float) -> int | None:
        if self._prev_test_k is not None and self._prev_test_utility is not None:
            if current_utility > self._prev_test_utility:
                self._hill_direction = (1 if current_k > self._prev_test_k
                                        else -1)
            else:
                self._hill_direction = (-1 if current_k > self._prev_test_k
                                        else 1)

        try:
            idx = self._candidates.index(current_k)
        except ValueError:
            return None

        next_idx = idx + self._hill_direction
        if next_idx < 0 or next_idx >= len(self._candidates):
            return None

        next_k = self._candidates[next_idx]
        if next_k in self._test_counts:
            return None

        return next_k

    def _should_early_exit(self) -> bool:
        if len(self._test_results) < 2:
            return False

        values = [self._test_results[k] for k in sorted(self._test_results)]
        if all(values[i] >= values[i + 1] for i in range(len(values) - 1)):
            if values[-1] < self._config.utility_threshold and len(values) >= 2:
                return True

        if len(values) >= 2:
            max_v = max(values)
            min_v = min(values)
            if max_v > 0 and (max_v - min_v) / max_v < self._config.convergence_tolerance:
                return True

        return False

    def _finish_test(self) -> None:
        if self._test_results:
            self._best_k = max(self._test_results,
                               key=lambda k: self._test_results[k])
            best_utility = self._test_results[self._best_k]

            if best_utility < self._config.utility_threshold:
                self._best_k = 0
                self._set_duration = min(
                    self._set_duration * self._config.backoff_multiplier,
                    1024,
                )
            else:
                self._set_duration = self._config.set_iterations

        self._phase = _Phase.SET
        self._phase_iter = 0

    def _start_test(self) -> None:
        self._phase = _Phase.TEST
        self._phase_iter = 0
        self._test_results.clear()
        self._test_counts.clear()
        self._prev_test_k = None
        self._prev_test_utility = None
        self._hill_direction = 1

    def recommended_depth(self) -> int:
        if not self._config.enabled:
            return max(self._candidates) if self._candidates else 0

        strategy = self._config.strategy

        if strategy == "fixed":
            return self._config.static_depth

        if strategy == "ema":
            return self._ema_k

        # cascade
        if self._phase == _Phase.TEST:
            tested = set(self._test_counts.keys())
            for k in self._candidates:
                if k not in tested:
                    return k
            return self._best_k

        return self._best_k

    @staticmethod
    def apply_ceiling(depth: int, max_verifiable_depth: int) -> int:
        return min(depth, max_verifiable_depth)

    @staticmethod
    def coverage_discount(expert_coverage_fraction: float,
                          floor: float = 0.5) -> float:
        return max(expert_coverage_fraction, floor)

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    @property
    def current_phase(self) -> str:
        return self._phase.value

    @property
    def best_depth(self) -> int:
        return self._best_k

    @property
    def iterations_in_phase(self) -> int:
        return self._phase_iter
