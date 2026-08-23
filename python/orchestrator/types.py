"""Shared domain types for the Python orchestrator.

High-level Pythonic types (dataclasses, enums, NamedTuples) used across all
orchestrator modules. No ctypes dependency — wire-format structs live in
shm_protocol.py; these are the semantic layer above them.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import NamedTuple


class ExpertKey(NamedTuple):
    layer_idx: int
    expert_idx: int


class CacheZone(enum.IntEnum):
    STABLE = 0
    STREAMING = 1


class GpuTier(enum.IntEnum):
    ABSENT = 0
    RESERVED = 1
    TRANSFERRING = 2
    DRAINING = 3
    PARTIAL_1 = 4
    PARTIAL_1_2 = 5
    HOT = 6


class HostTier(enum.IntEnum):
    COLD = 0
    LOADING_TO_RAM = 1
    WARM = 2


class EvictionPolicyType(enum.Enum):
    IMPACT_WEIGHTED_LRU = "impact_weighted_lru"
    LRU = "lru"
    LFU = "lfu"


class InitialAssignment(enum.Enum):
    ROUND_ROBIN = "round_robin"
    COACTIVATION = "coactivation"


class PrefetchSource(enum.Enum):
    PRESCOPE = "prescope"
    PROBE = "probe"
    MOE_SPEQ = "moe_speq"
    SP_MOE = "sp_moe"


class PrefetchConfidence(enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


DEFAULT_COMPUTE_WEIGHTS: dict[str, float] = {
    "rtx5090": 1.0,
    "rtx5080": 0.5,
}


@dataclass(frozen=True)
class GpuConfig:
    position: int
    gpu_type: str
    is_tp: bool
    vram_bytes: int
    compute_weight: float = 1.0


@dataclass(frozen=True)
class TokenLogprob:
    token_id: int
    logprob: float


@dataclass(frozen=True)
class StepLogprobs:
    token: TokenLogprob
    top_logprobs: tuple[TokenLogprob, ...]


@dataclass(frozen=True)
class EngineMetadata:
    num_gpus: int
    num_moe_layers: int
    num_experts: int
    num_layers: int
    expert_bytes: int
    kv_bytes_per_page: int
    num_expert_devices: int = 0
    gpus: tuple[GpuConfig, ...] = ()
    think_start_token_id: int = -1
    think_end_token_id: int = -2
    eos_token_ids: tuple[int, ...] = ()
    # Engine buffer-registry ids for the forward-pass seams (#91).  Filled
    # by the engine glue from query_buffer_ids(); 0 in mock/unit contexts
    # (the scripted daemons ignore buf ids).
    hidden_buf_id: int = 0     # "hidden_state.attn.rank0" — EMBEDDING out / head in
    logits_buf_id: int = 0     # "logits_scratch.pos0"     — OUTPUT_HEAD logits out
    vocab_size: int = 0        # for CMD_SAMPLE_TOKENS (non-greedy sampling)
    # TD-PREFILL-MOE-BIG: the engine's effective MoE token-batch capacity
    # (EngineInfo.moe_batch_capacity) — the max num_seqs a single
    # FETCH_AND_RUN_MOE[_BIG] / RUN_MOE accepts, and therefore the
    # superchunk-prefill token bound.  0 in mock/unit contexts
    # (superchunk prefill stays off; per-token prefill unchanged).
    moe_batch_capacity: int = 0
    # DeepSeek-V4 (V4-8): per hidden layer attention type from
    # EngineInfo.v4_attention_types — 0 = SWA-only, 1 = CSA (ratio 4),
    # 2 = HCA (ratio 128).  Empty for non-V4 models.
    v4_attention_types: tuple[int, ...] = ()

    def attention_type_for_layer(self, layer: int) -> int:
        """V4 per-layer attention type (V4-8); 0/SWA-equivalent default
        when the model is not V4 or the layer is out of range."""
        if 0 <= layer < len(self.v4_attention_types):
            return self.v4_attention_types[layer]
        return 0


@dataclass
class PrefetchHint:
    key: ExpertKey
    target_layer: int
    confidence: PrefetchConfidence
    source: PrefetchSource
    score: float = 0.0


@dataclass
class PrefetchPriority:
    key: ExpertKey
    target_layer: int
    target_gpu: int = 0
    priority_score: float = 0.0
    estimated_time_until_needed_us: int = 0
    source_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AffinityHint:
    key: ExpertKey
    preferred_gpu: int
    score: float


@dataclass
class WorkItemExperts:
    layer_idx: int
    expert_keys: list[ExpertKey] = field(default_factory=list)
    routing_weights: list[float] = field(default_factory=list)


class WorkOperation(enum.IntEnum):
    ATTENTION = 0
    GATING = 1
    EXPERT_FFN = 2
    EMBEDDING = 3
    OUTPUT_HEAD = 4


class WorkStatus(enum.IntEnum):
    PENDING = 0
    WAITING_TRANSFER = 1
    READY = 2
    DISPATCHED = 3
    COMPLETED = 4


class SpeculationState(enum.Enum):
    AUTOREGRESSIVE = "autoregressive"
    DRAFTING = "drafting"
    PREFETCHING_VERIFY = "prefetching"
    VERIFYING = "verifying"
    ACCEPTING = "accepting"


@dataclass
class WorkItem:
    request_id: int
    layer_idx: int
    operation: WorkOperation
    target_gpu: int = 0
    priority: float = 0.0
    status: WorkStatus = WorkStatus.PENDING
    is_speculative: bool = False
    speculation_position: int = 0
    required_experts: list[ExpertKey] = field(default_factory=list)
    routing_weights: list[float] = field(default_factory=list)
    timestamp_created_ns: int = 0
    cmd_seq: int = 0


@dataclass
class TransferPlanEntry:
    key: ExpertKey
    target_gpu: int
    zone: CacheZone = CacheZone.STREAMING
    priority: float = 0.0
    start_delay_us: int = 0
    expert_bytes: int = 0


@dataclass
class TransferPlan:
    entries: list[TransferPlanEntry] = field(default_factory=list)

    def required_slots_per_gpu(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for e in self.entries:
            counts[e.target_gpu] = counts.get(e.target_gpu, 0) + 1
        return counts

    def bytes_per_gpu(self) -> dict[int, int]:
        totals: dict[int, int] = {}
        for e in self.entries:
            totals[e.target_gpu] = totals.get(e.target_gpu, 0) + e.expert_bytes
        return totals


@dataclass
class VerificationPlan:
    transfers: list[TransferPlanEntry] = field(default_factory=list)
    max_depth: int = 0


@dataclass
class EvictionPlanEntry:
    key: ExpertKey
    gpu_idx: int
    zone: CacheZone = CacheZone.STABLE
    eviction_score: float = 0.0


@dataclass
class EvictionPlan:
    entries: list[EvictionPlanEntry] = field(default_factory=list)


@dataclass(frozen=True)
class ExpertEvictionInput:
    key: ExpertKey
    zone: CacheZone = CacheZone.STREAMING
    is_duplicate: bool = False
    gpu_idx: int = 0
    recency: float = 0.0
    frequency: float = 0.0
    routing_weight: float = 0.0
    temporal_autocorr: float = 0.0
    coactivation: float = 0.0
    prefetch_score: float = 0.0
    hysteresis_state: float = 0.0


@dataclass(frozen=True)
class DuplicationCandidate:
    key: ExpertKey
    source_gpu: int
    target_gpu: int
    benefit: float
    frequency_percentile: float


@dataclass
class ComputeBatch:
    items: list[WorkItem] = field(default_factory=list)
    gpu_idx: int = 0
