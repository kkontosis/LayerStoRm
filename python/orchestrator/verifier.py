"""Unified verifier — MoE-Spec masked verification for speculative decoding.

Takes a combined draft (from draft_combiner), checks expert residency against
MoE-Spec masking rules, determines the verifiable prefix depth, plans the
verification forward pass, and after execution compares logits to accept the
longest matching prefix.

Two-phase architecture:
  Pre-forward:  analyze_moe_coverage → compute_verifiable_depth → plan_verification_pass
  Post-forward: compare_logits → build_result → record_result

Invariants enforced:
  INV-B: verification_quality_floor (hard reject if any layer < floor)
  INV-C: conservative/optimistic transfer mode (orchestrator concern; verifier
         works with whatever resident_experts snapshot it receives)
  INV-D: final_output_quality >= verification_expert_coverage * full_model_quality
  INV-4.9b: pages_to_promote (accepted) / pages_to_free (rejected)
  INV-3.4.5: rejected → cancel all downstream, free KV pages

Spec: IMPLEMENTATION_GUIDE.md §4.7, §4.7.4.
Papers: MoE-Spec (expert budgeting), SP-MoE (speculative prefetch),
        Lookahead (utility-driven depth), Draft & Verify (self-speculative).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from orchestrator.draft_combiner import CombinedDraft
from orchestrator.speculative_prefetch import _adaptive_topk
from orchestrator.types import ExpertKey


@dataclass(frozen=True)
class VerifierConfig:
    adaptive_topk_threshold: float = 0.92
    verification_quality_floor: float = 0.85
    in_flight_transfer_mode: str = "conservative"
    substitution_policy: str = "substitution"
    max_draft_tree_size: int = 64
    acceptance_ema_alpha: float = 0.3


@dataclass
class MoeAnalysis:
    position: int
    min_layer_coverage: float
    per_layer_coverage: list[float] = field(default_factory=list)
    per_layer_moe_mode: list[int] = field(default_factory=list)
    quality_floor_violated: bool = False
    truncated: bool = False
    truncation_reason: str | None = None


@dataclass
class VerificationResult:
    accepted_length: int
    attempted_length: int
    accepted_tokens: list[int] = field(default_factory=list)
    rejected_positions: list[int] = field(default_factory=list)
    per_position_moe_mode: list[int] = field(default_factory=list)
    per_position_coverage: list[float] = field(default_factory=list)
    truncation_reason: str | None = None
    quality_floor_violations: list[int] = field(default_factory=list)
    pages_to_free: list[tuple[int, int]] = field(default_factory=list)
    pages_to_promote: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class CommandDescriptor:
    cmd_type: str
    layer_idx: int = 0
    num_seqs: int = 0
    moe_mode: int = 0
    is_draft: int = 0
    emit_checkpoint: int = 0
    readback_to_host: int = 0
    compute_confidence: int = 0


@dataclass
class VerificationCommandPlan:
    commands: list[CommandDescriptor] = field(default_factory=list)
    depth: int = 0
    num_layers: int = 0


class Verifier:

    def __init__(
        self,
        config: VerifierConfig | None = None,
        num_layers: int = 61,
        num_moe_layers: int = 58,
        first_moe_layer: int = 3,
        num_experts: int = 256,
    ) -> None:
        self._config = config or VerifierConfig()
        self._num_layers = num_layers
        self._num_moe_layers = num_moe_layers
        self._first_moe_layer = first_moe_layer
        self._num_experts = num_experts

        self._total_verifications: int = 0
        self._total_accepted: int = 0
        self._total_attempted: int = 0
        self._total_quality_floor_violations: int = 0
        self._total_truncations: int = 0
        self._ema_acceptance_rate: float = 0.0
        self._ema_initialized: bool = False

    @property
    def config(self) -> VerifierConfig:
        return self._config

    # ------------------------------------------------------------------
    # Pre-forward phase
    # ------------------------------------------------------------------

    def analyze_moe_coverage(
        self,
        draft: CombinedDraft,
        draft_gating: np.ndarray,
        resident_experts: dict[int, set[ExpertKey]],
        first_moe_layer: int | None = None,
    ) -> list[MoeAnalysis]:
        if draft.depth == 0 or draft_gating.size == 0:
            return []

        fml = first_moe_layer if first_moe_layer is not None else self._first_moe_layer
        depth, num_moe_layers, _num_experts = draft_gating.shape
        depth = min(depth, draft.depth)
        cfg = self._config

        all_resident: set[ExpertKey] = set()
        for keys in resident_experts.values():
            all_resident |= keys

        use_substitution = cfg.substitution_policy == "substitution"

        analyses: list[MoeAnalysis] = []
        for d in range(depth):
            per_layer_cov: list[float] = []
            per_layer_mode: list[int] = []
            floor_violated = False
            truncated = False
            reason: str | None = None

            for m in range(num_moe_layers):
                weights = draft_gating[d, m]
                selected = _adaptive_topk(weights, cfg.adaptive_topk_threshold)
                abs_layer = fml + m

                total_weight = 0.0
                resident_weight = 0.0
                all_present = True
                for eidx in selected:
                    w = float(weights[eidx])
                    total_weight += w
                    key = ExpertKey(abs_layer, eidx)
                    if key in all_resident:
                        resident_weight += w
                    else:
                        all_present = False

                if total_weight > 0.0:
                    coverage = resident_weight / total_weight
                else:
                    coverage = 0.0

                if all_present:
                    moe_mode = 0
                elif use_substitution:
                    moe_mode = 2
                else:
                    moe_mode = 1

                per_layer_cov.append(coverage)
                per_layer_mode.append(moe_mode)

                if coverage < cfg.verification_quality_floor:
                    floor_violated = True
                    if reason is None:
                        reason = f"quality_floor_violated_at_layer_{abs_layer}"

            min_cov = min(per_layer_cov) if per_layer_cov else 0.0

            if floor_violated:
                truncated = True
            elif not use_substitution and any(mode == 1 for mode in per_layer_mode):
                truncated = True
                if reason is None:
                    first_trunc = next(
                        fml + i for i, mode in enumerate(per_layer_mode) if mode == 1
                    )
                    reason = f"discard_policy_at_layer_{first_trunc}"

            analyses.append(MoeAnalysis(
                position=d,
                min_layer_coverage=min_cov,
                per_layer_coverage=per_layer_cov,
                per_layer_moe_mode=per_layer_mode,
                quality_floor_violated=floor_violated,
                truncated=truncated,
                truncation_reason=reason,
            ))

        return analyses

    def compute_verifiable_depth(
        self,
        analyses: list[MoeAnalysis],
        max_depth: int | None = None,
    ) -> int:
        if not analyses:
            return 0

        ceiling = self._config.max_draft_tree_size
        if max_depth is not None:
            ceiling = min(ceiling, max_depth)
        ceiling = min(ceiling, len(analyses))

        for i in range(ceiling):
            a = analyses[i]
            if a.truncated or a.quality_floor_violated:
                return i
        return ceiling

    def plan_verification_pass(
        self,
        depth: int,
        analyses: list[MoeAnalysis],
    ) -> VerificationCommandPlan:
        if depth <= 0:
            return VerificationCommandPlan(
                depth=0, num_layers=self._num_layers,
            )

        commands: list[CommandDescriptor] = []

        commands.append(CommandDescriptor(
            cmd_type="EMBEDDING_LOOKUP",
            num_seqs=depth,
        ))

        fml = self._first_moe_layer
        for layer in range(self._num_layers):
            commands.append(CommandDescriptor(
                cmd_type="RUN_ATTENTION",
                layer_idx=layer,
                num_seqs=depth,
                is_draft=0,
                emit_checkpoint=1,
            ))

            # TD-59b: emit RUN_MOE for ALL layers. Dense layers use
            # moe_mode=0 (C++ dispatch_moe_internal handles dense FFN).
            # TD-VERIFY-FETCHSEAM (production path decision, 2026-07-05):
            # RUN_MOE is deprecated for ROUTED layers — the verification pass
            # must be rebased on FETCH_AND_RUN_MOE (attention emit_gating
            # routing export -> routed list -> fetch+run), adapted for the
            # multi-token draft batch, when speculation work resumes. Dense
            # layers legitimately stay on RUN_MOE.
            moe_idx = layer - fml
            is_moe = 0 <= moe_idx < self._num_moe_layers
            worst_mode = 0
            if is_moe:
                for a in analyses[:depth]:
                    if moe_idx < len(a.per_layer_moe_mode):
                        worst_mode = max(worst_mode, a.per_layer_moe_mode[moe_idx])
            commands.append(CommandDescriptor(
                cmd_type="RUN_MOE",
                layer_idx=layer,
                num_seqs=depth,
                moe_mode=worst_mode,
                emit_checkpoint=1 if is_moe else 0,
            ))

        commands.append(CommandDescriptor(
            cmd_type="OUTPUT_HEAD",
            num_seqs=depth,
            readback_to_host=1,
            compute_confidence=1,
        ))

        return VerificationCommandPlan(
            commands=commands,
            depth=depth,
            num_layers=self._num_layers,
        )

    def verify(
        self,
        draft: CombinedDraft,
        draft_gating: np.ndarray,
        resident_experts: dict[int, set[ExpertKey]],
        max_verifiable_depth: int,
        first_moe_layer: int | None = None,
    ) -> tuple[list[MoeAnalysis], int, VerificationCommandPlan]:
        analyses = self.analyze_moe_coverage(
            draft, draft_gating, resident_experts, first_moe_layer,
        )
        depth = self.compute_verifiable_depth(analyses, max_verifiable_depth)
        plan = self.plan_verification_pass(depth, analyses)
        return analyses, depth, plan

    # ------------------------------------------------------------------
    # Post-forward phase
    # ------------------------------------------------------------------

    @staticmethod
    def compare_logits(
        draft_tokens: list[int],
        target_tokens: list[int],
    ) -> int:
        length = min(len(draft_tokens), len(target_tokens))
        for i in range(length):
            if draft_tokens[i] != target_tokens[i]:
                return i
        return length

    def build_result(
        self,
        draft: CombinedDraft,
        analyses: list[MoeAnalysis],
        verified_depth: int,
        accepted_length: int,
        seq_id: int,
    ) -> VerificationResult:
        accepted_tokens = draft.tokens[:accepted_length]
        rejected_positions = list(range(accepted_length, verified_depth))

        per_pos_mode: list[int] = []
        per_pos_cov: list[float] = []
        for i in range(verified_depth):
            if i < len(analyses):
                a = analyses[i]
                per_pos_mode.append(
                    max(a.per_layer_moe_mode) if a.per_layer_moe_mode else 0,
                )
                per_pos_cov.append(a.min_layer_coverage)
            else:
                per_pos_mode.append(0)
                per_pos_cov.append(1.0)

        truncation_reason: str | None = None
        if verified_depth < draft.depth:
            for a in analyses[verified_depth:]:
                if a.truncation_reason is not None:
                    truncation_reason = a.truncation_reason
                    break
            if truncation_reason is None and verified_depth < len(analyses):
                a = analyses[verified_depth]
                if a.truncation_reason is not None:
                    truncation_reason = a.truncation_reason

        qf_violations = [
            a.position for a in analyses if a.quality_floor_violated
        ]

        pages_to_promote = [(seq_id, pos) for pos in range(accepted_length)]
        pages_to_free = [(seq_id, pos) for pos in rejected_positions]

        return VerificationResult(
            accepted_length=accepted_length,
            attempted_length=verified_depth,
            accepted_tokens=accepted_tokens,
            rejected_positions=rejected_positions,
            per_position_moe_mode=per_pos_mode,
            per_position_coverage=per_pos_cov,
            truncation_reason=truncation_reason,
            quality_floor_violations=qf_violations,
            pages_to_free=pages_to_free,
            pages_to_promote=pages_to_promote,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def record_result(self, result: VerificationResult) -> None:
        self._total_verifications += 1
        self._total_accepted += result.accepted_length
        self._total_attempted += result.attempted_length
        self._total_quality_floor_violations += len(result.quality_floor_violations)
        if result.truncation_reason is not None:
            self._total_truncations += 1

        if result.attempted_length > 0:
            batch_rate = result.accepted_length / result.attempted_length
            alpha = self._config.acceptance_ema_alpha
            if not self._ema_initialized:
                self._ema_acceptance_rate = batch_rate
                self._ema_initialized = True
            else:
                self._ema_acceptance_rate = (
                    alpha * batch_rate + (1.0 - alpha) * self._ema_acceptance_rate
                )

    @property
    def acceptance_rate(self) -> float:
        return self._ema_acceptance_rate

    @property
    def total_verifications(self) -> int:
        return self._total_verifications

    @property
    def total_accepted(self) -> int:
        return self._total_accepted

    @property
    def total_attempted(self) -> int:
        return self._total_attempted

    @property
    def quality_floor_violation_rate(self) -> float:
        if self._total_attempted == 0:
            return 0.0
        return self._total_quality_floor_violations / self._total_attempted

    @property
    def truncation_count(self) -> int:
        return self._total_truncations
