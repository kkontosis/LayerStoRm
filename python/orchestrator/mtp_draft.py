"""MTP draft — DeepSeek Multi-Token Prediction head draft generation.

Orchestrator-side planner for MTP speculative drafting. Computes MTP layer
indices (cyclic reuse), dynamic depth gating, and acceptance tracking.
Does not dispatch GPU commands directly — the orchestrator loop (#62) uses
this module's planning output to build IPC command sequences.

Architecture per draft step k (DeepSeek V3 §2.2, eq 21-23):
  h'_k = M_k [RMSNorm(h_{k-1}); RMSNorm(Emb(t_{k}))]   -- projection
  h_k  = TRM_k(h'_k)                                      -- decoder block
  P_k  = OutHead(h_k)                                      -- shared LM head

MTP layers are indexed after main model layers:
  mtp_layer_idx = num_layers + (step_idx % num_mtp_layers)

Reference: DeepSeek V3 Technical Report, §2.2 Multi-Token Prediction.
vLLM: ref/vllm/vllm/model_executor/models/deepseek_mtp.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MtpDraftConfig:
    enabled: bool = True
    max_depth: int = 3
    dynamic_depth: bool = True
    confidence_threshold: float = 0.4
    acceptance_ema_alpha: float = 0.3
    # Batched verification (the throughput-bearing mode): the round's K
    # draft tokens are verified in ONE teacher-forced K+1-token forward on
    # the MAIN sequence (EMBEDDING -> per-layer chunked attention +
    # FETCH_AND_RUN_MOE -> multi-token OUTPUT_HEAD readback) instead of K+1
    # sequential full-model feeds.  Lossless either way (INV-MTP-LOSSLESS);
    # False falls back to the sequential early-stop schedule (the
    # MtpLossless golden's reference shape).
    batched_verify: bool = True


@dataclass
class DraftStep:
    token_id: int
    confidence: float
    mtp_layer_idx: int


@dataclass
class MtpDraftResult:
    steps: list[DraftStep] = field(default_factory=list)
    source: str = "mtp"

    @property
    def tokens(self) -> list[int]:
        return [s.token_id for s in self.steps]

    @property
    def depth(self) -> int:
        return len(self.steps)


@dataclass
class MtpLayerPlan:
    step_idx: int
    mtp_layer_idx: int


class MtpDraft:

    def __init__(self, config: MtpDraftConfig | None = None,
                 num_mtp_layers: int = 1,
                 num_layers: int = 61) -> None:
        self._config = config or MtpDraftConfig()
        self._num_mtp_layers = num_mtp_layers
        self._num_layers = num_layers
        self._total_drafts: int = 0
        self._total_steps: int = 0
        self._total_accepted: int = 0
        self._ema_acceptance_rate: float = 0.0
        self._ema_initialized: bool = False

    def mtp_layer_idx(self, step_idx: int) -> int:
        return self._num_layers + (step_idx % self._num_mtp_layers)

    def plan_draft_steps(self, depth: int) -> list[MtpLayerPlan]:
        effective_depth = min(depth, self._config.max_depth)
        return [
            MtpLayerPlan(step_idx=d, mtp_layer_idx=self.mtp_layer_idx(d))
            for d in range(effective_depth)
        ]

    def should_continue(self, step_idx: int, confidence: float) -> bool:
        if step_idx >= self._config.max_depth:
            return False
        if not self._config.dynamic_depth:
            return True
        if step_idx == 0:
            return True
        return confidence >= self._config.confidence_threshold

    def record_result(self, result: MtpDraftResult,
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
    def batched_verify(self) -> bool:
        return self._config.batched_verify

    @property
    def max_depth(self) -> int:
        return self._config.max_depth

    @property
    def num_mtp_layers(self) -> int:
        return self._num_mtp_layers

    @property
    def total_drafts(self) -> int:
        return self._total_drafts

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def total_accepted(self) -> int:
        return self._total_accepted
