"""Self-speculative draft — reduced-expert full-model forward pass for draft generation.

Runs the entire transformer with top-K experts per MoE layer (default K=1)
plus optional residual correction (#51) and optional layer skipping (#53).
Third-priority draft source after prompt lookup and MTP.

At draft time each MoE layer uses:
  draft_output = top_K_expert_output + residual_correction_mlp(top_1_output, gating_scores)

The orchestrator loop (#62) dispatches GPU commands based on this module's
layer plan. This module does not dispatch IPC commands directly.

Reference: IMPLEMENTATION_GUIDE.md §4.7.
Paper: Kangaroo — Lossless Self-Speculative Decoding with Double Early Exiting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orchestrator.mtp_draft import DraftStep


@dataclass(frozen=True)
class SelfSpeculativeConfig:
    enabled: bool = True
    draft_expert_count: int = 1
    adaptive_exit_enabled: bool = True
    draft_confidence_threshold: float = 0.4
    max_depth: int = 5
    acceptance_ema_alpha: float = 0.3


@dataclass
class SelfSpecLayerPlan:
    layer_idx: int
    is_moe: bool
    apply_residual_correction: bool
    store_gating_output: bool
    skip: bool = False


@dataclass
class SelfSpecDraftResult:
    steps: list[DraftStep] = field(default_factory=list)
    source: str = "self_speculative"

    @property
    def tokens(self) -> list[int]:
        return [s.token_id for s in self.steps]

    @property
    def depth(self) -> int:
        return len(self.steps)


class SelfSpeculative:

    def __init__(self, config: SelfSpeculativeConfig | None = None,
                 num_layers: int = 61,
                 num_moe_layers: int = 58,
                 first_moe_layer: int = 3,
                 residual_correction_enabled: bool = True) -> None:
        self._config = config or SelfSpeculativeConfig()
        self._num_layers = num_layers
        self._num_moe_layers = num_moe_layers
        self._first_moe_layer = first_moe_layer
        self._residual_correction_enabled = residual_correction_enabled
        self._total_drafts: int = 0
        self._total_steps: int = 0
        self._total_accepted: int = 0
        self._ema_acceptance_rate: float = 0.0
        self._ema_initialized: bool = False

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (self._first_moe_layer
                <= layer_idx
                < self._first_moe_layer + self._num_moe_layers)

    def plan_forward_pass(self, layer_skip_set: set[int] | None = None
                          ) -> list[SelfSpecLayerPlan]:
        skips = layer_skip_set or set()
        plan: list[SelfSpecLayerPlan] = []
        for layer in range(self._num_layers):
            moe = self.is_moe_layer(layer)
            plan.append(SelfSpecLayerPlan(
                layer_idx=layer,
                is_moe=moe,
                apply_residual_correction=moe and self._residual_correction_enabled,
                store_gating_output=moe,
                skip=layer in skips,
            ))
        return plan

    def should_continue(self, step_idx: int, confidence: float) -> bool:
        if step_idx >= self._config.max_depth:
            return False
        if not self._config.adaptive_exit_enabled:
            return True
        if step_idx == 0:
            return True
        return confidence >= self._config.draft_confidence_threshold

    def record_result(self, result: SelfSpecDraftResult,
                      num_accepted: int) -> None:
        depth = result.depth
        self._total_drafts += 1
        self._total_steps += depth
        self._total_accepted += num_accepted

        if depth > 0:
            batch_rate = num_accepted / depth
            if not self._ema_initialized:
                self._ema_acceptance_rate = batch_rate
                self._ema_initialized = True
            else:
                a = self._config.acceptance_ema_alpha
                self._ema_acceptance_rate = (
                    a * batch_rate + (1.0 - a) * self._ema_acceptance_rate
                )

    @property
    def acceptance_rate(self) -> float:
        return self._ema_acceptance_rate

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    @property
    def max_depth(self) -> int:
        return self._config.max_depth

    @property
    def draft_expert_count(self) -> int:
        return self._config.draft_expert_count

    @property
    def total_drafts(self) -> int:
        return self._total_drafts

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def total_accepted(self) -> int:
        return self._total_accepted
