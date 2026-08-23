"""SP-MoE speculative prefetch — verification transfer planning from draft gating.

Processes DraftGatingCache (full gating weight vectors collected during speculative
draft) to compute which experts verification will need, an ordered transfer schedule,
and the maximum verifiable depth.  See spec §4.7.3.

Algorithm:
  1. Adaptive top-K per (position, layer): keep experts until cumulative weight >= threshold.
  2. Union of required experts across positions, deduplicated.
  3. Filter already-resident experts, order transfers by (layer, first-position-needed).
  4. Compute max_depth: deepest position where all transfers can complete before
     verification reaches it, using conservative single-stream bandwidth model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orchestrator.types import (
    CacheZone,
    ExpertKey,
    PrefetchConfidence,
    PrefetchHint,
    PrefetchSource,
    TransferPlanEntry,
    VerificationPlan,
)


@dataclass(frozen=True)
class SpeculativePrefetchConfig:
    adaptive_topk_threshold: float = 0.92
    verification_quality_floor: float = 0.85
    per_layer_us: int = 920
    pcie_bw_bytes_per_us: float = 64_000.0
    max_transfers_per_plan: int = 256


class SpeculativePrefetch:

    def __init__(
        self,
        config: SpeculativePrefetchConfig | None = None,
        num_gpus: int = 4,
        expert_bytes: int = 2_359_296,
    ) -> None:
        self._config = config or SpeculativePrefetchConfig()
        self._num_gpus = num_gpus
        self._expert_bytes = expert_bytes

    @property
    def config(self) -> SpeculativePrefetchConfig:
        return self._config

    def compute_verification_plan(
        self,
        draft_gating: np.ndarray,
        resident_experts: dict[int, set[ExpertKey]],
        affinity: dict[ExpertKey, int] | None = None,
        first_moe_layer: int = 3,
    ) -> VerificationPlan:
        if draft_gating.size == 0:
            return VerificationPlan()

        depth, num_moe_layers, _num_experts = draft_gating.shape
        cfg = self._config

        required = self._required_experts_by_position(
            draft_gating, cfg.adaptive_topk_threshold, first_moe_layer,
        )

        all_resident: set[ExpertKey] = set()
        for keys in resident_experts.values():
            all_resident |= keys

        transfers = self._build_transfer_schedule(
            required, all_resident, affinity, first_moe_layer,
        )
        max_depth = self._compute_max_depth(
            required, all_resident, num_moe_layers,
        )

        return VerificationPlan(transfers=transfers, max_depth=max_depth)

    def gating_to_hints(
        self,
        draft_gating: np.ndarray,
        first_moe_layer: int = 3,
    ) -> list[PrefetchHint]:
        if draft_gating.size == 0:
            return []

        depth, num_moe_layers, _num_experts = draft_gating.shape
        cfg = self._config

        by_key: dict[ExpertKey, float] = {}
        for d in range(depth):
            for m in range(num_moe_layers):
                weights = draft_gating[d, m]
                selected = _adaptive_topk(weights, cfg.adaptive_topk_threshold)
                abs_layer = first_moe_layer + m
                for eidx in selected:
                    key = ExpertKey(abs_layer, eidx)
                    score = float(weights[eidx])
                    if key not in by_key or score > by_key[key]:
                        by_key[key] = score

        hints = [
            PrefetchHint(
                key=key,
                target_layer=key.layer_idx,
                confidence=PrefetchConfidence.HIGH,
                source=PrefetchSource.SP_MOE,
                score=score,
            )
            for key, score in by_key.items()
        ]
        hints.sort(key=lambda h: h.score, reverse=True)
        return hints

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _required_experts_by_position(
        draft_gating: np.ndarray,
        threshold: float,
        first_moe_layer: int,
    ) -> list[list[set[ExpertKey]]]:
        depth, num_moe_layers, _ = draft_gating.shape
        result: list[list[set[ExpertKey]]] = []
        for d in range(depth):
            layers: list[set[ExpertKey]] = []
            for m in range(num_moe_layers):
                selected = _adaptive_topk(draft_gating[d, m], threshold)
                abs_layer = first_moe_layer + m
                layers.append({ExpertKey(abs_layer, eidx) for eidx in selected})
            result.append(layers)
        return result

    def _build_transfer_schedule(
        self,
        required: list[list[set[ExpertKey]]],
        all_resident: set[ExpertKey],
        affinity: dict[ExpertKey, int] | None,
        first_moe_layer: int,
    ) -> list[TransferPlanEntry]:
        seen: dict[ExpertKey, tuple[int, int]] = {}
        for d, layers in enumerate(required):
            for keys in layers:
                for key in keys:
                    if key not in all_resident and key not in seen:
                        seen[key] = (key.layer_idx, d)

        ordered = sorted(seen.keys(), key=lambda k: seen[k])

        entries: list[TransferPlanEntry] = []
        rr_counter = 0
        total = max(len(ordered), 1)
        for rank, key in enumerate(ordered):
            if len(entries) >= self._config.max_transfers_per_plan:
                break
            if affinity is not None and key in affinity:
                target_gpu = affinity[key]
            else:
                target_gpu = rr_counter % self._num_gpus
                rr_counter += 1
            entries.append(TransferPlanEntry(
                key=key,
                target_gpu=target_gpu,
                zone=CacheZone.STREAMING,
                priority=1.0 - rank / total,
                start_delay_us=0,
                expert_bytes=self._expert_bytes,
            ))
        return entries

    def _compute_max_depth(
        self,
        required: list[list[set[ExpertKey]]],
        all_resident: set[ExpertKey],
        num_moe_layers: int,
    ) -> int:
        if not required:
            return 0

        cfg = self._config
        verification_time_per_pos = num_moe_layers * cfg.per_layer_us

        seen_non_resident: set[ExpertKey] = set()
        cumulative_bytes = 0
        max_depth = -1

        for d, layers in enumerate(required):
            for keys in layers:
                for key in keys:
                    if key not in all_resident and key not in seen_non_resident:
                        seen_non_resident.add(key)
                        cumulative_bytes += self._expert_bytes

            transfer_time_us = cumulative_bytes / cfg.pcie_bw_bytes_per_us
            budget_us = d * verification_time_per_pos

            if transfer_time_us <= budget_us:
                max_depth = d
            else:
                if d == 0 and cumulative_bytes == 0:
                    max_depth = 0
                break
        else:
            max_depth = len(required) - 1

        return max(max_depth, 0)


def _adaptive_topk(weights: np.ndarray, threshold: float) -> list[int]:
    indices = np.argsort(weights)[::-1]
    cumulative = 0.0
    selected: list[int] = []
    for idx in indices:
        w = float(weights[idx])
        selected.append(int(idx))
        cumulative += w
        if cumulative >= threshold:
            break
    if not selected:
        selected.append(int(np.argmax(weights)))
    return selected
