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
    # P-29 step 13 / TD-MTP-PROBE-DEFERRED-CONSUMERS: index of the FIRST MoE
    # layer = the model config's `first_k_dense_replace` (layers below it
    # are dense-FFN).  Pass it from config at every real construction
    # site: the historical `num_layers - num_moe_layers` subtraction is
    # only correct while the MoE census stops at num_hidden_layers.  With
    # the MTP expert census armed (glm5_next + speculation.method mtp +
    # speculation.mtp.enabled) the census counts the NextN block too —
    # 43 MoE layers on a 45-layer model — and the subtraction yields 2,
    # silently making dense layer 2 look like an MoE layer.
    # -1 (default) = not supplied: __post_init__ falls back to the legacy
    # subtraction so mock/unit callers keep their exact prior geometry.
    first_moe_layer: int = -1
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
    # P-30 step 1: the engine's REALIZED single-shot MoE chunk bound
    # (EngineInfo.moe_chunk_capacity) after the elastic chunk fail-safe.
    # Batches above it run the chunked grouped-GEMM path — rejected with
    # expert-only ranks resident (TD-MOE-EP-XTP-WAVES), so EP-beyond-TP
    # superchunk strides must clamp to this. 0 in mock/unit contexts.
    moe_chunk_capacity: int = 0
    # Per hidden layer attention type from EngineInfo.attention_types
    # (GF3.2 / TD-ATTN-TYPES-V4-NAMING: renamed from v4_attention_types).
    # V4 codes: 0 = SWA-only, 1 = CSA (ratio 4), 2 = HCA (ratio 128).
    # glm5_next codes: 3 = linear (KDA — no KV, per-request recurrent
    # state), 4 = sparse MLA (NoPE DSA — KV-bearing).
    # Empty for homogeneous-attention models (GLM-5.2 / V3.2).
    attention_types: tuple[int, ...] = ()
    # R4b arch capability (EngineInfo.seq_fork_truncatable,
    # INV-SEQ-FORK-TRUNC): True iff CMD_SEQ_FORK honours prefix_len
    # truncation on this boot's architecture — i.e. the arch has NO lossy
    # position-indexed per-sequence state (V4 in-place rings have it, so
    # V4 is False).  Gates the orchestrator's mid-edge prefix reuse
    # (PrefixCache.lookup_mid_edge); False = grid/exact-node hits only.
    # Default False: mock/unit contexts keep the legacy lookup unless a
    # test opts in.
    seq_fork_truncatable: bool = False
    # ── TD-GLM5-KDA-SLOTS-EXPORT (EngineInfo.kda_state_*): KDA state-pool
    # geometry — all zero for models without linear-attention per-request
    # state.  kda_state_mapped: the state units are whole-slab runs claimed
    # from the SHARED kMain pool (the default; kda_state_pool_pages is that
    # pool, kda_state_pages_per_seq is ONE admission's state demand beside
    # its KV+indexer pages — live pressure reads
    # StateSnapshot.kv_main_free_pages against it).  Not mapped: a
    # dedicated carve of kda_state_slots whole-request slots is the hard
    # concurrency cap (in-flight requests + non-hibernated prefix holders;
    # hibernated holders spill and return their slot, INV-KDA-STATE (g)).
    kda_state_mapped: bool = False
    kda_state_slots: int = 0
    kda_state_slot_bytes: int = 0
    kda_state_pages_per_seq: int = 0
    kda_state_pool_pages: int = 0

    def __post_init__(self) -> None:
        # Legacy fallback for callers that do not (yet) carry the model
        # config's first_k_dense_replace: the pre-MTP-census identity
        # `num_layers - num_moe_layers`.  Materialized once so every
        # consumer can read `metadata.first_moe_layer` unconditionally
        # (never the sentinel).
        if self.first_moe_layer < 0:
            object.__setattr__(self, "first_moe_layer",
                               max(0, self.num_layers - self.num_moe_layers))

    def attention_type_for_layer(self, layer: int) -> int:
        """Per-layer attention type (V4-8, extended by GF3.2);
        0/SWA-equivalent default when the model has no per-layer
        heterogeneity or the layer is out of range."""
        if 0 <= layer < len(self.attention_types):
            return self.attention_types[layer]
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
