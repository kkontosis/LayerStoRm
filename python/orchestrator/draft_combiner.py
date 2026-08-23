"""Draft combiner — merge candidates from prompt-lookup + MTP + self-spec
+ DSpark.

Per-position utility-based selection. Prompt-lookup checked first (zero
compute cost, INV-0.8d priority). For each future position, selects the
candidate with highest expected utility:

  utility = P(accept) * latency_saved - (1 - P(accept)) * compute_wasted

Where P(accept) is discounted by expert coverage fraction (INV-A).

DSpark (DSP-5/DSP-6): whole-γ-block drafts from the dedicated draft GPU.
With the trained confidence head on (confidence_enabled, INV-DSPARK-CONF)
the per-position p_accept is the cumulative survival a_j = Π_{i≤j} c_i
(``dspark_confidences``); otherwise the DsparkDraft EMA-acceptance proxy
(``dspark_confidence``) stands in.

Spec: IMPLEMENTATION_GUIDE.md §4.7.
Papers: Lookahead Utility-Driven Speculative Decoding (Yang et al.),
        Kangaroo (Li et al.), Prompt Lookup Decoding (Saxena),
        DSpark (see spec/DSpark_NOTES.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orchestrator.dspark_draft import DsparkDraftResult
from orchestrator.mtp_draft import DraftStep, MtpDraftResult
from orchestrator.self_speculative import SelfSpecDraftResult


@dataclass(frozen=True)
class DraftCombinerConfig:
    enabled: bool = True
    prompt_lookup_priority_boost: float = 1.0
    mtp_compute_cost: float = 0.1
    self_spec_compute_cost: float = 0.5
    dspark_compute_cost: float = 0.05
    acceptance_ema_alpha: float = 0.3
    latency_per_position_us: float = 920.0
    min_utility_threshold: float = 0.0
    coverage_discount_floor: float = 0.5


@dataclass
class CombinedDraftStep:
    token_id: int
    position: int
    source: str
    p_accept: float
    utility: float


@dataclass
class CombinedDraft:
    steps: list[CombinedDraftStep] = field(default_factory=list)

    @property
    def tokens(self) -> list[int]:
        return [s.token_id for s in self.steps]

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def source_breakdown(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.steps:
            counts[s.source] = counts.get(s.source, 0) + 1
        return counts


_SOURCES = ("prompt_lookup", "mtp", "self_speculative", "dspark")


class DraftCombiner:

    def __init__(self, config: DraftCombinerConfig | None = None) -> None:
        self._config = config or DraftCombinerConfig()
        self._total_combines: int = 0
        self._per_source_selected: dict[str, int] = {s: 0 for s in _SOURCES}
        self._per_source_positions: dict[str, int] = {s: 0 for s in _SOURCES}
        self._ema_acceptance_rate: float = 0.0
        self._ema_initialized: bool = False

    def combine(
        self,
        prompt_lookup_tokens: list[int],
        mtp_result: MtpDraftResult | None,
        self_spec_result: SelfSpecDraftResult | None,
        prompt_lookup_acceptance_rate: float,
        max_depth: int,
        max_verifiable_depth: int,
        expert_coverage_fraction: float = 1.0,
        dspark_result: DsparkDraftResult | None = None,
        dspark_confidence: float = 0.5,
        dspark_confidences: list[float] | None = None,
    ) -> CombinedDraft:
        self._total_combines += 1

        if not self._config.enabled:
            return CombinedDraft()

        effective_ceiling = min(max_depth, max_verifiable_depth)
        if effective_ceiling <= 0:
            return CombinedDraft()

        coverage_discount = max(expert_coverage_fraction,
                                self._config.coverage_discount_floor)
        lat = self._config.latency_per_position_us
        boost = self._config.prompt_lookup_priority_boost
        mtp_cost = self._config.mtp_compute_cost
        ss_cost = self._config.self_spec_compute_cost
        ds_cost = self._config.dspark_compute_cost
        threshold = self._config.min_utility_threshold

        steps: list[CombinedDraftStep] = []

        for pos in range(effective_ceiling):
            best_step: CombinedDraftStep | None = None
            best_utility = -float("inf")
            latency_saved = (pos + 1) * lat

            if pos < len(prompt_lookup_tokens):
                self._per_source_positions["prompt_lookup"] += 1
                p_accept = prompt_lookup_acceptance_rate * coverage_discount
                utility = (p_accept * latency_saved
                           + boost * lat)
                if utility > best_utility:
                    best_utility = utility
                    best_step = CombinedDraftStep(
                        token_id=prompt_lookup_tokens[pos],
                        position=pos,
                        source="prompt_lookup",
                        p_accept=p_accept,
                        utility=utility,
                    )

            if mtp_result is not None and pos < mtp_result.depth:
                self._per_source_positions["mtp"] += 1
                step = mtp_result.steps[pos]
                p_accept = step.confidence * coverage_discount
                utility = (p_accept * latency_saved
                           - (1.0 - p_accept) * mtp_cost * lat)
                if utility > best_utility:
                    best_utility = utility
                    best_step = CombinedDraftStep(
                        token_id=step.token_id,
                        position=pos,
                        source="mtp",
                        p_accept=p_accept,
                        utility=utility,
                    )

            if self_spec_result is not None and pos < self_spec_result.depth:
                self._per_source_positions["self_speculative"] += 1
                step = self_spec_result.steps[pos]
                p_accept = step.confidence * coverage_discount
                utility = (p_accept * latency_saved
                           - (1.0 - p_accept) * ss_cost * lat)
                if utility > best_utility:
                    best_utility = utility
                    best_step = CombinedDraftStep(
                        token_id=step.token_id,
                        position=pos,
                        source="self_speculative",
                        p_accept=p_accept,
                        utility=utility,
                    )

            # DSpark (DSP-5/DSP-6): whole-block ids.  p_accept per position
            # = the trained head's cumulative survival a_j when provided
            # (dspark_confidences, INV-DSPARK-CONF), else the per-round
            # EMA-acceptance proxy (dspark_confidence).
            if dspark_result is not None and pos < dspark_result.depth:
                self._per_source_positions["dspark"] += 1
                if dspark_confidences is not None \
                        and pos < len(dspark_confidences):
                    p_accept = dspark_confidences[pos] * coverage_discount
                else:
                    p_accept = dspark_confidence * coverage_discount
                utility = (p_accept * latency_saved
                           - (1.0 - p_accept) * ds_cost * lat)
                if utility > best_utility:
                    best_utility = utility
                    best_step = CombinedDraftStep(
                        token_id=dspark_result.tokens[pos],
                        position=pos,
                        source="dspark",
                        p_accept=p_accept,
                        utility=utility,
                    )

            if best_step is None or best_utility < threshold:
                break

            steps.append(best_step)
            self._per_source_selected[best_step.source] += 1

        return CombinedDraft(steps=steps)

    def record_result(self, result: CombinedDraft,
                      num_accepted: int) -> None:
        depth = result.depth
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
    def total_combines(self) -> int:
        return self._total_combines

    @property
    def source_selection_rates(self) -> dict[str, float]:
        total = sum(self._per_source_selected.values())
        if total == 0:
            return {s: 0.0 for s in _SOURCES}
        return {s: self._per_source_selected[s] / total for s in _SOURCES}
