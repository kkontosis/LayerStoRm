"""IPC protocol definitions — Python ctypes mirrors of C++ structs in ipc_protocol.h.

All structs, enums, and constants match the C++ wire format byte-for-byte.
Module-level asserts verify struct sizes at import time.
"""

import ctypes

# ── Constants ────────────────────────────────────────────────────────────────

PROTOCOL_VERSION = 1
MAX_GPUS = 8
MAX_MOE_LAYERS = 128  # TD-IPC-MOE-LAYER-CAP: mirror kMaxMoeLayers (>= GLM-5.2's 75 MoE + MTP)
MAX_EXPERTS = 256
MAX_TRACKED_REQUESTS = 64

DEFAULT_CMD_RING_SLOTS = 8192
DEFAULT_CMP_RING_SLOTS = 8192
CMD_SLOT_BYTES = 256
CMP_SLOT_BYTES = 128

# Sideband sub-region max entry counts (IPC-8b).
MAX_BATCH_DESCRIPTORS  = 512
MAX_EXPERT_PREFETCH    = 256
MAX_EXPERT_EVICTION    = 256
MAX_TRANSFER_BATCH     = 256
MAX_RESERVE_BATCH      = 256
MAX_NVME_READ_BATCH    = 256
MAX_SIDEBAND_TOKEN_IDS = 512

# ── CmdType ──────────────────────────────────────────────────────────────────

CMD_TRANSFER_H2D           = 0x0001
CMD_TRANSFER_D2H           = 0x0002

CMD_CACHE_RESERVE          = 0x0010
CMD_CACHE_EVICT            = 0x0011
CMD_CACHE_PROMOTE          = 0x0012
CMD_CACHE_DEMOTE           = 0x0013

CMD_ATTENTION_DECODE       = 0x0100
CMD_ATTENTION_PREFILL      = 0x0101
CMD_GATING                 = 0x0102
CMD_EXPERT_FFN             = 0x0103
CMD_EMBEDDING_LOOKUP       = 0x0104
CMD_OUTPUT_HEAD            = 0x0105
CMD_RMSNORM                = 0x0106
CMD_SWIGLU                 = 0x0107
CMD_MOE_PERMUTE            = 0x0108
CMD_MOE_UNPERMUTE          = 0x0109
CMD_DCP_CORRECTION         = 0x010A
CMD_NCCL_ALLREDUCE         = 0x010B
CMD_DYNAMIC_FP8_QUANT      = 0x010C
CMD_SAMPLE_TOKENS          = 0x010D  # GPU-side sampling (IPC-8f)

CMD_PRESCOPE_GATING        = 0x0200
CMD_PROBE_MLP              = 0x0201

CMD_GRAPH_REPLAY           = 0x0300

CMD_RECORD_EVENT           = 0x0400
CMD_STREAM_WAIT_EVENT      = 0x0401

CMD_COMPUTE_AFFINITY_HINTS = 0x0500
CMD_NUMA_MIGRATE           = 0x0501

CMD_SEQ_CREATE             = 0x0600
CMD_SEQ_FREE               = 0x0601
CMD_SEQ_FORK               = 0x0602

CMD_NVME_READ              = 0x0700
CMD_NVME_WRITE             = 0x0701
CMD_NVME_EVICT_HOST        = 0x0702
CMD_CANCEL_TRANSFER        = 0x0703

# Fused compute commands (IPC-8d)
D_B_CMD_RUN_ATTENTION      = 0x0800
D_B_CMD_RUN_MOE            = 0x0801

# Fused batch + sequential commands (IPC-8e)
D_B_CMD_PREFETCH_BATCH     = 0x0802
D_B_CMD_EVICT_BATCH        = 0x0803

# Batch commands (IPC-8e)
B_CMD_NVME_BATCH_READ      = 0x0900

# Sequential fusion commands (IPC-8e)
D_CMD_PREFETCH_EXPERT      = 0x0A00
D_CMD_EVICT_TO_HOST        = 0x0A01
D_CMD_STAGE_EXPERT         = 0x0A02
D_CMD_RUN_PREFETCH_PROBE   = 0x0A03
D_CMD_RUN_ADAPTER_FORWARD  = 0x0A04
D_CMD_SLOW_EVICT_TO_HOST   = 0x0A05  # D2H + VRAM evict (IPC-8i)
D_CMD_RUN_MTP_STEP         = 0x0A06  # fused MTP draft step (#62d)
D_CMD_RUN_SELF_SPEC_FORWARD = 0x0A07  # fused self-spec forward pass (#62e)
D_CMD_MTP_PROJECT          = 0x0A08  # MTP projection: enorm(Emb)+hnorm(h) -> eh_proj (#16)
D_CMD_RUN_DSPARK_STEP      = 0x0A09  # fused DSpark DFlash-backbone draft step (DSP-3/DSP-5)

# Config update (9.8-1b)
CMD_CONFIG_UPDATE          = 0x0B00

# Extended commands (IPC-8e)
E_CMD_SEQ_CREATE           = 0x0C00
E_CMD_SEQ_FREE             = 0x0C01
E_CMD_FETCH_AND_RUN_MOE    = 0x0C02  # progressive fetch + MoE execution (#90)
# TD-PREFILL-MOE-BIG: big-batch progressive fetch + CHUNKED MoE execution
# (chunked grouped GEMMs, elastic capacity, double-buffered waves).
E_CMD_FETCH_AND_RUN_MOE_BIG = 0x0C04
E_CMD_REEF_ROUTE           = 0x0C05  # REEF placement/eviction solve over the sideband
E_CMD_FAR_FORWARD_LAYER    = 0x0C06  # fused attention + routed FETCH_AND_RUN layer

CMD_SHUTDOWN               = 0xFF00
CMD_NOOP                   = 0xFFFF

# ── CompletionType ───────────────────────────────────────────────────────────

CMP_TRANSFER_DONE = 0x0001
CMP_CACHE_OP_DONE = 0x0010
CMP_COMPUTE_DONE  = 0x0100
CMP_CHECKPOINT    = 0x0200
CMP_EVENT_STATUS  = 0x0400
CMP_SEQ_OP_DONE   = 0x0600
CMP_NVME_DONE     = 0x0700
CMP_CANCEL_DONE   = 0x0703
CMP_CONFIG_UPDATE_DONE = 0x0B00
CMP_ELM_EXPERT_READY    = 0x0800
CMP_ELM_EXPERT_EVICTED  = 0x0801
CMP_ELM_EXPERT_PROGRESS = 0x0802
CMP_ERROR         = 0xEE00
CMP_GPU_FATAL     = 0xEF00

# ── CheckpointType ──────────────────────────────────────────────────────────

CHECKPOINT_HIDDEN_STATE      = 0
CHECKPOINT_GATING_OUTPUT     = 1
CHECKPOINT_LAYER_SIMILARITY  = 2  # daemon-computed cos_sim per layer (#62g)
CHECKPOINT_SEAM_ROUTING      = 3  # F-7: attention↔MoE seam — routed top-K
                                  #   (topk_weights f32 + topk_indices i32) at
                                  #   IpcLayout.kSeamCheckpointOff. host_buf_offset
                                  #   + data_bytes on the CMP_CHECKPOINT locate it.

# ── StreamId ─────────────────────────────────────────────────────────────────

STREAM_ATTENTION       = 0
STREAM_EXPERT_FFN      = 1
STREAM_GATING          = 2
STREAM_H2D_TRANSFER    = 3
STREAM_D2H_TRANSFER    = 4
STREAM_PREFETCH_COMPUTE = 5

# ── CacheZone ────────────────────────────────────────────────────────────────

ZONE_STABLE    = 0
ZONE_STREAMING = 1

# ── SubComponent (bitfield) ──────────────────────────────────────────────────

SUB_GATE = 0x01
SUB_UP   = 0x02
SUB_DOWN = 0x04
SUB_ALL  = 0x07

# ── GpuTier (matches C++ GpuTier enum, ELM-8) ──────────────────────────────

GPU_TIER_ABSENT        = 0
GPU_TIER_RESERVED      = 1
GPU_TIER_TRANSFERRING  = 2
GPU_TIER_DRAINING      = 3
GPU_TIER_PARTIAL_1     = 4  # gate ready (forward-compat)
GPU_TIER_PARTIAL_1_2   = 5  # gate + up ready (forward-compat)
GPU_TIER_HOT           = 6  # all sub-components ready

# ── HostTier (matches C++ HostTier enum, ELM-8) ────────────────────────────

HOST_TIER_COLD    = 0  # on NVMe or mmap only
HOST_TIER_LOADING = 1  # NVMe→RAM read inflight
HOST_TIER_WARM    = 2  # in host RAM warm cache (or mmap)

# ── IpcHeader (256 bytes) ────────────────────────────────────────────────────

class IpcHeader(ctypes.Structure):
    _fields_ = [
        ("version",            ctypes.c_uint32),
        ("_pad_v",             ctypes.c_uint8 * 60),
        ("shutdown_requested", ctypes.c_uint32),
        ("_pad_s",             ctypes.c_uint8 * 60),
        ("error_code",         ctypes.c_uint32),
        ("error_message",      ctypes.c_char * 124),
    ]

assert ctypes.sizeof(IpcHeader) == 256, f"IpcHeader size {ctypes.sizeof(IpcHeader)} != 256"

# ── RingHeader (128 bytes) ───────────────────────────────────────────────────

class RingHeader(ctypes.Structure):
    _fields_ = [
        ("producer_seq", ctypes.c_uint64),
        ("_pad_p",       ctypes.c_uint8 * 56),
        ("consumer_seq", ctypes.c_uint64),
        ("_pad_c",       ctypes.c_uint8 * 48),
        ("slot_count",   ctypes.c_uint32),
        ("slot_size",    ctypes.c_uint32),
    ]

assert ctypes.sizeof(RingHeader) == 128, f"RingHeader size {ctypes.sizeof(RingHeader)} != 128"

# ── Command payload structs ──────────────────────────────────────────────────

class TransferPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",     ctypes.c_uint32),
        ("expert_idx",    ctypes.c_uint16),
        ("sub_component", ctypes.c_uint8),
        ("zone",          ctypes.c_uint8),
        ("bytes",         ctypes.c_int64),
        ("priority",      ctypes.c_float),
        ("delay_us",      ctypes.c_uint32),
    ]

class CacheReservePayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",    ctypes.c_uint32),
        ("expert_idx",   ctypes.c_uint16),
        ("zone",         ctypes.c_uint8),
        ("is_duplicate", ctypes.c_uint8),
    ]

class CacheOpPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("_pad",       ctypes.c_uint16),
    ]

class AttentionPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",           ctypes.c_uint32),
        ("batch_size",          ctypes.c_uint32),
        ("use_graph",           ctypes.c_uint8),
        ("is_sparse",           ctypes.c_uint8),
        ("_pad",                ctypes.c_uint16),
        ("hidden_state_buf_id", ctypes.c_uint32),
        ("kv_cache_buf_id",     ctypes.c_uint32),
        ("seqlens_buf_id",      ctypes.c_uint32),   # KD-2
        ("block_table_buf_id",  ctypes.c_uint32),   # KD-2
    ]

class GatingPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",              ctypes.c_uint32),
        ("num_tokens",             ctypes.c_uint32),
        ("input_buf_id",           ctypes.c_uint32),
        ("output_weights_buf_id",  ctypes.c_uint32),
        ("output_indices_buf_id",  ctypes.c_uint32),
    ]

class ExpertFfnPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",            ctypes.c_uint32),
        ("num_experts",          ctypes.c_uint32),
        ("total_tokens",         ctypes.c_uint32),
        ("permuted_input_buf_id", ctypes.c_uint32),
        ("output_buf_id",        ctypes.c_uint32),
        ("expert_offsets_buf_id", ctypes.c_uint32),
        ("hidden_dim",           ctypes.c_uint32),    # KD-2: output dim N
        ("weights_buf_id",       ctypes.c_uint32),    # KD-2
        ("workspace_buf_id",     ctypes.c_uint32),    # KD-2
        ("k_dim",                ctypes.c_uint32),    # GG-5: input dim K (GGUF path)
        ("quant_mode",           ctypes.c_uint8),     # KD-2: 0=NVFP4,1=FP8,2=GGUF
        ("gguf_type",            ctypes.c_uint8),     # GG-5: GgufKQuantType (Q2_K=0..Q8_0=5)
        ("_pad_eff",             ctypes.c_uint8 * 2),
    ]

class PrescopePayload(ctypes.Structure):
    _fields_ = [
        ("target_layer_idx",    ctypes.c_uint32),
        ("num_tokens",          ctypes.c_uint32),
        ("hidden_state_buf_id", ctypes.c_uint32),
        ("output_buf_id",       ctypes.c_uint32),
    ]

class ProbePayload(ctypes.Structure):
    _fields_ = [
        ("probe_point_idx",     ctypes.c_uint32),
        ("num_tokens",          ctypes.c_uint32),
        ("hidden_state_buf_id", ctypes.c_uint32),
        ("output_buf_id",       ctypes.c_uint32),
    ]

class EmbeddingLookupPayload(ctypes.Structure):
    _fields_ = [
        ("num_tokens",   ctypes.c_uint32),
        ("output_buf_id", ctypes.c_uint32),
        # TD-PREFILL-SUPERCHUNK: destination ROW offset into the hidden buffer.
        ("row_offset",   ctypes.c_uint32),
    ]

class OutputHeadPayload(ctypes.Structure):
    _fields_ = [
        ("num_tokens",          ctypes.c_uint32),
        ("input_buf_id",        ctypes.c_uint32),
        ("output_buf_id",       ctypes.c_uint32),
        ("readback_to_host",    ctypes.c_uint8),
        ("compute_confidence",  ctypes.c_uint8),   # IPC-8g
        ("num_logprobs",        ctypes.c_uint8),    # 0=disabled, 1-20=top-K requested
        ("mtp_head",            ctypes.c_uint8),    # #16: 0=main head, else mtp_idx+1
        ("readback_logits",     ctypes.c_uint8),    # guided decoding: D2H full
                                                    # last-row logits to the pinned
                                                    # readback row (TD-SERVE-NAMED-
                                                    # TOOL-CHOICE)
    ]

class SampleTokensPayload(ctypes.Structure):
    _fields_ = [
        ("num_tokens",    ctypes.c_uint32),
        ("logits_buf_id", ctypes.c_uint32),
        ("vocab_size",    ctypes.c_uint32),
        ("top_k",         ctypes.c_uint32),
        ("temperature",   ctypes.c_float),
        ("top_p",         ctypes.c_float),
        ("random_seed",   ctypes.c_uint64),
    ]

class RunAttentionPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",       ctypes.c_uint32),
        ("num_seqs",        ctypes.c_uint32),
        ("is_prefill",      ctypes.c_uint8),
        ("use_graph",       ctypes.c_uint8),
        ("is_draft",        ctypes.c_uint8),
        ("emit_checkpoint", ctypes.c_uint8),
        ("chunk_start",     ctypes.c_uint32),
        ("chunk_len",       ctypes.c_uint32),
        ("emit_gating",     ctypes.c_uint8),   # F-1: run router+topk gating at end of attn
        ("store_gating",    ctypes.c_uint8),   # F-3: also export routed top-K to sideband
        # TD-PREFILL-SUPERCHUNK: sub-chunk launch of a superchunk (coverage
        # window relaxation) + hidden-state row offset for its rows.
        ("superchunk",      ctypes.c_uint8),
        ("_pad_ra",         ctypes.c_uint8),
        ("row_offset",      ctypes.c_uint32),
    ]

class RunMoePayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",                 ctypes.c_uint32),
        ("num_seqs",                  ctypes.c_uint32),
        ("moe_mode",                  ctypes.c_uint8),
        ("apply_residual_correction", ctypes.c_uint8),
        ("store_gating_output",       ctypes.c_uint8),
        ("emit_checkpoint",           ctypes.c_uint8),
        ("use_precomputed_gating",    ctypes.c_uint8),  # F-2: consume topk in moe_scratch_
    ]

class PrefetchBatchPayload(ctypes.Structure):
    _fields_ = [
        ("count",    ctypes.c_uint32),
        ("priority", ctypes.c_float),
        ("delay_us", ctypes.c_uint32),
        ("_pad",     ctypes.c_uint32),
    ]

class EvictBatchPayload(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("_pad",  ctypes.c_uint32),
    ]

class NvmeBatchReadPayload(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("_pad",  ctypes.c_uint32),
    ]

class PrefetchExpertPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("zone",       ctypes.c_uint8),
        ("gpu_idx",    ctypes.c_uint8),
        ("priority",   ctypes.c_float),
        ("delay_us",   ctypes.c_uint32),
    ]

class EvictToHostPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("gpu_idx",    ctypes.c_uint16),
    ]

class SlowEvictToHostPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("gpu_idx",    ctypes.c_uint16),
    ]

class StageExpertPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("zone",       ctypes.c_uint8),
        ("gpu_idx",    ctypes.c_uint8),
        ("priority",   ctypes.c_float),
        ("delay_us",   ctypes.c_uint32),
    ]

class RunPrefetchProbePayload(ctypes.Structure):
    _fields_ = [
        ("target_layer", ctypes.c_uint32),
        ("num_tokens",   ctypes.c_uint32),
        ("probe_points", ctypes.c_uint8),
        ("_pad",         ctypes.c_uint8 * 3),
    ]

class RunAdapterForwardPayload(ctypes.Structure):
    _fields_ = [
        ("num_tokens",             ctypes.c_uint32),
        ("adapter_weights_buf_id", ctypes.c_uint32),
    ]

class MtpStepPayload(ctypes.Structure):
    _fields_ = [
        ("mtp_layer_idx",  ctypes.c_uint32),
        ("seq_id",         ctypes.c_uint64),
        ("input_token_id", ctypes.c_uint32),
        ("step_idx",       ctypes.c_uint8),
        ("_pad",           ctypes.c_uint8 * 3),
    ]

class MtpProjectPayload(ctypes.Structure):
    """#16: MTP projection — eh_proj(enorm(Emb(token)) || hnorm(prev_hidden)).

    Composes the production-seam MTP draft step: MTP_PROJECT ->
    RUN_ATTENTION(mtp_layer, emit/store gating) -> FETCH_AND_RUN_MOE ->
    OUTPUT_HEAD(mtp_head) -> SAMPLE_TOKENS.

    hidden_row selects the attn_buf ROW holding prev_hidden (a batched MTP
    verify leaves the pass's trunk hiddens in rows [0..V); MTP steps write
    only row 0, so higher rows survive a sequential catch-up chain).
    0 = historical single-row behavior.
    """
    _fields_ = [
        ("mtp_layer_idx",  ctypes.c_uint32),
        ("input_token_id", ctypes.c_uint32),
        ("step_idx",       ctypes.c_uint8),
        ("hidden_row",     ctypes.c_uint8),
        ("_pad",           ctypes.c_uint8 * 2),
    ]

class SelfSpecForwardPayload(ctypes.Structure):
    _fields_ = [
        ("seq_id",              ctypes.c_uint64),
        ("input_token_id",      ctypes.c_uint32),
        ("draft_expert_count",  ctypes.c_uint8),
        ("apply_residual_corr", ctypes.c_uint8),
        ("store_gating",        ctypes.c_uint8),
        ("step_idx",            ctypes.c_uint8),
        ("skip_mask_lo",        ctypes.c_uint64),
        ("skip_mask_hi",        ctypes.c_uint64),
    ]

class DsparkStepPayload(ctypes.Structure):
    """D_CMD_RUN_DSPARK_STEP — fused DSpark draft step (DSP-3/DSP-5).

    ONE DFlash-backbone forward over the whole gamma block off the
    aux-hidden-ingested context KV (positions < anchor_pos of seq_id),
    chained with the sequential Markov head on the draft stream.  The
    completion (CMP_COMPUTE_DONE) carries host_buf_offset/data_bytes of the
    gamma sampled draft ids (i32), host-visible in the sideband readback
    scratch (SIDEBAND_SPEC_CHECKPOINT_OFF + 2560) when it fires.  Executes
    on the DRAFT GPU regardless of the header gpu_idx.  Mirrors
    run_dspark_step in C++ ipc_protocol.h.
    """
    _fields_ = [
        ("seq_id",          ctypes.c_uint64),
        ("anchor_token_id", ctypes.c_uint32),
        ("anchor_pos",      ctypes.c_uint32),
        ("num_query",       ctypes.c_uint8),   # 0 = config speculative_tokens
        ("step_idx",        ctypes.c_uint8),
        ("_pad",            ctypes.c_uint8 * 2),
    ]

# F-6: max inline per-entry gating weights in the fetch_and_run_moe decider
# (mirrors kMaxFetchDeciderWeights in C++ ipc_protocol.h).
MAX_FETCH_DECIDER_WEIGHTS = 8

class FetchAndRunMoePayload(ctypes.Structure):
    """E_CMD_FETCH_AND_RUN_MOE — progressive fetch + MoE execution (#90).

    Expert list in sideband at SIDEBAND_EXPERT_PREFETCH_OFF
    (ExpertPrefetchEntry[expert_count]); batch descriptor at
    SIDEBAND_BATCH_DESCRIPTOR_OFF (same contract as RUN_ATTENTION).
    This is the PRODUCTION routed-MoE seam: the daemon fetches only the
    missing routed experts (H2D) and runs the MoE. All F-6 decider fields
    default to 0 → fetch every entry. Mirrors fetch_and_run_moe in C++.
    """
    _fields_ = [
        ("layer_idx",               ctypes.c_uint32),
        ("num_seqs",                ctypes.c_uint32),
        ("expert_count",            ctypes.c_uint32),
        ("timeout_us",              ctypes.c_uint32),
        ("moe_mode",                ctypes.c_uint8),
        ("weight_count",            ctypes.c_uint8),
        ("min_experts",             ctypes.c_uint16),
        ("max_new_fetches",         ctypes.c_uint16),
        ("have_evict_map",          ctypes.c_uint8),
        ("_pad",                    ctypes.c_uint8),
        ("gating_weight_threshold", ctypes.c_float),
        ("weights",                 ctypes.c_float * MAX_FETCH_DECIDER_WEIGHTS),
    ]

assert ctypes.sizeof(FetchAndRunMoePayload) == 60

class ReefRoutePayload(ctypes.Structure):
    """E_CMD_REEF_ROUTE — REEF placement/eviction solve over IPC.

    Routed union at SIDEBAND_EXPERT_PREFETCH_OFF (layer_idx + expert_idx
    per entry; gpu_idx/zone IGNORED on input). The daemon's ReefOrch
    service rewrites entries[i].gpu_idx in place and fills the
    index-aligned ExpertEvictionEntry[] at SIDEBAND_EXPERT_EVICTION_OFF
    (0xFFFF sentinel = no victim), then completes CMP_COMPUTE_DONE. The
    next E_CMD_FETCH_AND_RUN_MOE goes out with have_evict_map=1 over the
    SAME sideband (ring order = race-free handoff).
    """
    _fields_ = [
        ("layer_idx",    ctypes.c_uint32),
        ("expert_count", ctypes.c_uint32),
    ]

class FarForwardLayerPayload(ctypes.Structure):
    """E_CMD_FAR_FORWARD_LAYER — fused attention + routed FETCH layer.

    ONE command per layer through the PRODUCTION seam (distinct from the
    deprecated-path E_FORWARD_ONE_LAYER): attention (+fused gate on MoE
    layers) → daemon-side routing dedup → placement per route_mode
    (0 = static e%num_gpus, 1 = ReefOrch service + victim map) →
    FETCH_AND_RUN execution. ONE CMP_COMPUTE_DONE whose data_bytes = the
    deduped entry count (the `lookups` accumulator; no readback here).
    Dense layers run attention + the dense-FFN path, data_bytes=0.
    """
    _fields_ = [
        ("layer_idx",   ctypes.c_uint32),
        ("num_seqs",    ctypes.c_uint32),
        ("chunk_start", ctypes.c_uint32),
        ("chunk_len",   ctypes.c_uint32),
        ("timeout_us",  ctypes.c_uint32),
        ("is_prefill",  ctypes.c_uint8),
        ("route_mode",  ctypes.c_uint8),
        ("_pad",        ctypes.c_uint8 * 2),
    ]

assert ctypes.sizeof(ReefRoutePayload) == 8
assert ctypes.sizeof(FarForwardLayerPayload) == 24

class FetchAndRunMoeBigPayload(ctypes.Structure):
    """E_CMD_FETCH_AND_RUN_MOE_BIG — big-batch fetch + CHUNKED MoE
    (TD-PREFILL-MOE-BIG).

    Layout-identical PREFIX to FetchAndRunMoePayload (the C++ handler reads
    the common fields through cmd.fetch_and_run_moe) plus chunk_tokens —
    the grouped-GEMM chunk-size override (0 = engine default,
    compute.moe_big_chunk_tokens).  num_seqs may exceed the legacy
    kMaxBatchDescriptors up to EngineInfo.moe_batch_capacity (the elastic
    superchunk capacity).  Same sideband contract as fetch_and_run_moe.
    """
    _fields_ = [
        ("layer_idx",               ctypes.c_uint32),
        ("num_seqs",                ctypes.c_uint32),
        ("expert_count",            ctypes.c_uint32),
        ("timeout_us",              ctypes.c_uint32),
        ("moe_mode",                ctypes.c_uint8),
        ("weight_count",            ctypes.c_uint8),
        ("min_experts",             ctypes.c_uint16),
        ("max_new_fetches",         ctypes.c_uint16),
        ("have_evict_map",          ctypes.c_uint8),
        ("_pad",                    ctypes.c_uint8),
        ("gating_weight_threshold", ctypes.c_float),
        ("weights",                 ctypes.c_float * MAX_FETCH_DECIDER_WEIGHTS),
        ("chunk_tokens",            ctypes.c_uint32),
    ]

assert ctypes.sizeof(FetchAndRunMoeBigPayload) == 64

class RmsnormPayload(ctypes.Structure):
    _fields_ = [
        ("num_tokens",    ctypes.c_uint32),
        ("input_buf_id",  ctypes.c_uint32),
        ("output_buf_id", ctypes.c_uint32),
        ("weight_buf_id", ctypes.c_uint32),
        ("eps",           ctypes.c_float),
        ("hidden_size",   ctypes.c_uint32),   # KD-2
    ]

class SwigluPayload(ctypes.Structure):
    _fields_ = [
        ("num_tokens",    ctypes.c_uint32),
        ("hidden_dim",    ctypes.c_uint32),
        ("input_buf_id",  ctypes.c_uint32),
        ("output_buf_id", ctypes.c_uint32),
    ]

class MoePermutePayload(ctypes.Structure):
    _fields_ = [
        ("num_tokens",       ctypes.c_uint32),
        ("num_experts",      ctypes.c_uint32),
        ("input_buf_id",     ctypes.c_uint32),
        ("output_buf_id",    ctypes.c_uint32),
        ("indices_buf_id",   ctypes.c_uint32),
        ("offsets_buf_id",   ctypes.c_uint32),
        ("topk",             ctypes.c_uint32),    # KD-2
        ("hidden_dim",       ctypes.c_uint32),    # KD-2
        ("workspace_buf_id", ctypes.c_uint32),    # KD-2
    ]

class MoeUnpermutePayload(ctypes.Structure):
    _fields_ = [
        ("num_tokens",     ctypes.c_uint32),
        ("num_experts",    ctypes.c_uint32),
        ("input_buf_id",   ctypes.c_uint32),
        ("output_buf_id",  ctypes.c_uint32),
        ("indices_buf_id", ctypes.c_uint32),
        ("topk",           ctypes.c_uint32),    # KD-2
        ("hidden_dim",     ctypes.c_uint32),    # KD-2
        ("weights_buf_id", ctypes.c_uint32),    # KD-2
    ]

class DcpCorrectionPayload(ctypes.Structure):
    _fields_ = [
        ("batch_size",    ctypes.c_uint32),
        ("input_buf_id",  ctypes.c_uint32),
        ("output_buf_id", ctypes.c_uint32),
        ("lse_buf_id",    ctypes.c_uint32),
    ]

class NcclAllreducePayload(ctypes.Structure):
    _fields_ = [
        ("count",  ctypes.c_uint32),
        ("buf_id", ctypes.c_uint32),
        ("dtype",  ctypes.c_uint8),
        ("_pad",   ctypes.c_uint8 * 3),
    ]

class DynamicFp8QuantPayload(ctypes.Structure):
    _fields_ = [
        ("num_tokens",     ctypes.c_uint32),
        ("hidden_dim",     ctypes.c_uint32),
        ("input_buf_id",   ctypes.c_uint32),
        ("output_buf_id",  ctypes.c_uint32),
        ("scales_buf_id",  ctypes.c_uint32),
    ]

class GraphPayload(ctypes.Structure):
    _fields_ = [
        ("graph_type",  ctypes.c_uint8),
        ("_pad",        ctypes.c_uint8 * 3),
        ("batch_size",  ctypes.c_uint32),
    ]

class EventPayload(ctypes.Structure):
    _fields_ = [
        ("event_id", ctypes.c_uint32),
    ]

class AffinityHintsPayload(ctypes.Structure):
    _fields_ = [
        ("num_gpus",           ctypes.c_uint32),
        ("gpu_capacity_slots", ctypes.c_uint32 * MAX_GPUS),
    ]

class NumaMigratePayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",        ctypes.c_uint32),
        ("expert_idx",       ctypes.c_uint16),
        ("target_numa_node", ctypes.c_uint8),
        ("_pad",             ctypes.c_uint8),
    ]

class SeqCreatePayload(ctypes.Structure):
    _fields_ = [
        ("seq_id",     ctypes.c_uint64),
        ("prompt_len", ctypes.c_uint32),
        ("pool",       ctypes.c_uint8),
        ("_pad",       ctypes.c_uint8 * 3),
    ]

class SeqFreePayload(ctypes.Structure):
    _fields_ = [
        ("seq_id", ctypes.c_uint64),
    ]

class SeqForkPayload(ctypes.Structure):
    _fields_ = [
        ("src_seq_id", ctypes.c_uint64),
        ("dst_seq_id", ctypes.c_uint64),
    ]

class NvmeReadPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("gpu_hint",   ctypes.c_uint8),
        ("_pad",       ctypes.c_uint8),
    ]

class NvmeWritePayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("_pad",       ctypes.c_uint16),
    ]

class NvmeEvictHostPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("_pad",       ctypes.c_uint16),
    ]

class CancelTransferPayload(ctypes.Structure):
    _fields_ = [
        ("target_cmd_seq", ctypes.c_uint32),
        ("_pad",           ctypes.c_uint32),
    ]

# ── Sideband entry structs (IPC-8b) ─────────────────────────────────────────

class BatchDescriptorEntry(ctypes.Structure):
    _fields_ = [
        ("seq_id",    ctypes.c_uint64),
        ("token_pos", ctypes.c_uint32),
        ("_pad",      ctypes.c_uint32),
    ]

assert ctypes.sizeof(BatchDescriptorEntry) == 16

class ExpertPrefetchEntry(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("zone",       ctypes.c_uint8),
        ("gpu_idx",    ctypes.c_uint8),
    ]

assert ctypes.sizeof(ExpertPrefetchEntry) == 8

class ExpertEvictionEntry(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("gpu_idx",    ctypes.c_uint8),
        ("_pad",       ctypes.c_uint8),
    ]

assert ctypes.sizeof(ExpertEvictionEntry) == 8

class TransferBatchEntry(ctypes.Structure):
    _fields_ = [
        ("layer_idx",     ctypes.c_uint32),
        ("expert_idx",    ctypes.c_uint16),
        ("sub_component", ctypes.c_uint8),
        ("zone",          ctypes.c_uint8),
        ("bytes",         ctypes.c_int64),
    ]

assert ctypes.sizeof(TransferBatchEntry) == 16

class ReserveBatchEntry(ctypes.Structure):
    _fields_ = [
        ("layer_idx",    ctypes.c_uint32),
        ("expert_idx",   ctypes.c_uint16),
        ("zone",         ctypes.c_uint8),
        ("is_duplicate", ctypes.c_uint8),
    ]

assert ctypes.sizeof(ReserveBatchEntry) == 8

class NvmeReadEntry(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("gpu_hint",   ctypes.c_uint8),
        ("_pad",       ctypes.c_uint8),
    ]

assert ctypes.sizeof(NvmeReadEntry) == 8

# ── Sideband sub-region layout (mirrors IpcLayout in C++) ────────────────────

SIDEBAND_BATCH_DESCRIPTOR_OFF  = 0
SIDEBAND_BATCH_DESCRIPTOR_SIZE = MAX_BATCH_DESCRIPTORS * ctypes.sizeof(BatchDescriptorEntry)

SIDEBAND_EXPERT_PREFETCH_OFF   = SIDEBAND_BATCH_DESCRIPTOR_OFF + SIDEBAND_BATCH_DESCRIPTOR_SIZE
SIDEBAND_EXPERT_PREFETCH_SIZE  = MAX_EXPERT_PREFETCH * ctypes.sizeof(ExpertPrefetchEntry)

SIDEBAND_EXPERT_EVICTION_OFF   = SIDEBAND_EXPERT_PREFETCH_OFF + SIDEBAND_EXPERT_PREFETCH_SIZE
SIDEBAND_EXPERT_EVICTION_SIZE  = MAX_EXPERT_EVICTION * ctypes.sizeof(ExpertEvictionEntry)

SIDEBAND_TRANSFER_BATCH_OFF    = SIDEBAND_EXPERT_EVICTION_OFF + SIDEBAND_EXPERT_EVICTION_SIZE
SIDEBAND_TRANSFER_BATCH_SIZE   = MAX_TRANSFER_BATCH * ctypes.sizeof(TransferBatchEntry)

SIDEBAND_RESERVE_BATCH_OFF     = SIDEBAND_TRANSFER_BATCH_OFF + SIDEBAND_TRANSFER_BATCH_SIZE
SIDEBAND_RESERVE_BATCH_SIZE    = MAX_RESERVE_BATCH * ctypes.sizeof(ReserveBatchEntry)

SIDEBAND_NVME_READ_OFF         = SIDEBAND_RESERVE_BATCH_OFF + SIDEBAND_RESERVE_BATCH_SIZE
SIDEBAND_NVME_READ_SIZE        = MAX_NVME_READ_BATCH * ctypes.sizeof(NvmeReadEntry)

SIDEBAND_TOKEN_IDS_OFF         = SIDEBAND_NVME_READ_OFF + SIDEBAND_NVME_READ_SIZE
SIDEBAND_TOKEN_IDS_SIZE        = MAX_SIDEBAND_TOKEN_IDS * ctypes.sizeof(ctypes.c_uint32)

SIDEBAND_SPEC_CHECKPOINT_OFF   = SIDEBAND_TOKEN_IDS_OFF + SIDEBAND_TOKEN_IDS_SIZE
SIDEBAND_SPEC_CHECKPOINT_SIZE  = 4096

# F-4 routing-export slot (polled): RoutingExportHeader (16B) + weights[] + indices[],
# sized for MAX_BATCH_DESCRIPTORS tokens at ROUTING_EXPORT_MAX_TOPK each (float weight
# + int32 index per slot). Mirrors IpcLayout::kRoutingExport* in C++.
ROUTING_EXPORT_HEADER_SIZE     = 16
ROUTING_EXPORT_MAX_TOPK        = 16
SIDEBAND_ROUTING_EXPORT_OFF    = SIDEBAND_SPEC_CHECKPOINT_OFF + SIDEBAND_SPEC_CHECKPOINT_SIZE
SIDEBAND_ROUTING_EXPORT_SIZE   = (
    ROUTING_EXPORT_HEADER_SIZE
    + MAX_BATCH_DESCRIPTORS * ROUTING_EXPORT_MAX_TOPK
      * (ctypes.sizeof(ctypes.c_float) + ctypes.sizeof(ctypes.c_int32))
)

class RoutingExportHeader(ctypes.Structure):
    """F-4 routed top-K export header (mirrors RoutingExportHeader in C++).

    Written by the daemon after RUN_ATTENTION with emit_gating+store_gating:
    weights[] at SIDEBAND_ROUTING_EXPORT_WEIGHTS_OFF, indices[] at
    SIDEBAND_ROUTING_EXPORT_INDICES_OFF (num_tokens * topk entries each).
    """
    _fields_ = [
        ("num_tokens", ctypes.c_uint32),   # tokens whose routing was published
        ("topk",       ctypes.c_uint32),   # routed experts per token
        ("layer_idx",  ctypes.c_uint32),   # layer the routing belongs to
        ("_pad",       ctypes.c_uint32),
    ]

assert ctypes.sizeof(RoutingExportHeader) == ROUTING_EXPORT_HEADER_SIZE

# weights[] immediately follows the header; indices[] follow the full
# weights capacity (both fixed offsets — mirrors kRoutingExportWeightsOff /
# kRoutingExportIndicesOff in C++).
SIDEBAND_ROUTING_EXPORT_WEIGHTS_OFF = (
    SIDEBAND_ROUTING_EXPORT_OFF + ROUTING_EXPORT_HEADER_SIZE
)
SIDEBAND_ROUTING_EXPORT_INDICES_OFF = (
    SIDEBAND_ROUTING_EXPORT_WEIGHTS_OFF
    + MAX_BATCH_DESCRIPTORS * ROUTING_EXPORT_MAX_TOPK
      * ctypes.sizeof(ctypes.c_float)
)

# F-7 attention↔MoE seam checkpoint: expanded routed top-K (float weight + int32 index)
# for up to MAX_SEAM_ROUTING_EXPANDED entries. Mirrors IpcLayout::kSeamCheckpoint* in C++.
MAX_SEAM_ROUTING_EXPANDED      = 512
SIDEBAND_SEAM_CHECKPOINT_OFF   = SIDEBAND_ROUTING_EXPORT_OFF + SIDEBAND_ROUTING_EXPORT_SIZE
SIDEBAND_SEAM_CHECKPOINT_SIZE  = (
    MAX_SEAM_ROUTING_EXPANDED
    * (ctypes.sizeof(ctypes.c_float) + ctypes.sizeof(ctypes.c_int32))
)

SIDEBAND_TOTAL_SIZE            = SIDEBAND_SEAM_CHECKPOINT_OFF + SIDEBAND_SEAM_CHECKPOINT_SIZE

class ConfigUpdateEntry(ctypes.Structure):
    _fields_ = [
        ("field_id",   ctypes.c_uint16),
        ("value_type", ctypes.c_uint8),
        ("_pad",       ctypes.c_uint8),
        ("raw_value",  ctypes.c_uint32),
    ]

class ConfigUpdatePayload(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("_pad",  ctypes.c_uint32),
        ("entries", ConfigUpdateEntry * 29),
    ]

class CommandPayload(ctypes.Union):
    _fields_ = [
        ("transfer",         TransferPayload),
        ("cache_reserve",    CacheReservePayload),
        ("cache_op",         CacheOpPayload),
        ("attention",        AttentionPayload),
        ("gating",           GatingPayload),
        ("expert_ffn",       ExpertFfnPayload),
        ("embedding_lookup", EmbeddingLookupPayload),
        ("output_head",      OutputHeadPayload),
        ("rmsnorm",          RmsnormPayload),
        ("swiglu",           SwigluPayload),
        ("moe_permute",      MoePermutePayload),
        ("moe_unpermute",    MoeUnpermutePayload),
        ("dcp_correction",   DcpCorrectionPayload),
        ("nccl_allreduce",   NcclAllreducePayload),
        ("dynamic_fp8_quant", DynamicFp8QuantPayload),
        ("sample_tokens",    SampleTokensPayload),
        ("prescope",         PrescopePayload),
        ("probe",            ProbePayload),
        ("graph",            GraphPayload),
        ("event",            EventPayload),
        ("affinity_hints",   AffinityHintsPayload),
        ("numa_migrate",     NumaMigratePayload),
        ("seq_create",       SeqCreatePayload),
        ("seq_free",         SeqFreePayload),
        ("seq_fork",         SeqForkPayload),
        ("nvme_read",        NvmeReadPayload),
        ("nvme_write",       NvmeWritePayload),
        ("nvme_evict_host",  NvmeEvictHostPayload),
        ("cancel_transfer",  CancelTransferPayload),
        ("run_attention",    RunAttentionPayload),
        ("run_moe",          RunMoePayload),
        ("prefetch_batch",       PrefetchBatchPayload),
        ("evict_batch",          EvictBatchPayload),
        ("nvme_batch_read",      NvmeBatchReadPayload),
        ("prefetch_expert",      PrefetchExpertPayload),
        ("evict_to_host",        EvictToHostPayload),
        ("slow_evict_to_host",   SlowEvictToHostPayload),
        ("stage_expert",         StageExpertPayload),
        ("run_prefetch_probe",   RunPrefetchProbePayload),
        ("run_adapter_forward",  RunAdapterForwardPayload),
        ("run_mtp_step",         MtpStepPayload),
        ("mtp_project",          MtpProjectPayload),
        ("self_spec_forward",    SelfSpecForwardPayload),
        ("run_dspark_step",      DsparkStepPayload),
        ("fetch_and_run_moe",    FetchAndRunMoePayload),
        ("fetch_and_run_moe_big", FetchAndRunMoeBigPayload),
        ("reef_route",           ReefRoutePayload),
        ("far_forward_layer",    FarForwardLayerPayload),
        ("config_update",    ConfigUpdatePayload),
        ("raw",              ctypes.c_uint8 * 240),
    ]

# ── Command (256 bytes) ─────────────────────────────────────────────────────

class Command(ctypes.Structure):
    _fields_ = [
        ("cmd_type",  ctypes.c_uint32),
        ("cmd_seq",   ctypes.c_uint32),
        ("gpu_idx",   ctypes.c_uint32),
        ("stream_id", ctypes.c_uint32),
        ("payload",   CommandPayload),
    ]

assert ctypes.sizeof(Command) == 256, f"Command size {ctypes.sizeof(Command)} != 256"

# ── Completion payload structs ───────────────────────────────────────────────

class TransferCompletionPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",    ctypes.c_uint32),
        ("expert_idx",   ctypes.c_uint16),
        ("direction",    ctypes.c_uint8),
        ("_pad",         ctypes.c_uint8),
        ("vram_address", ctypes.c_uint64),
    ]

class ComputeCompletionPayload(ctypes.Structure):
    _fields_ = [
        ("cmd_type",           ctypes.c_uint32),
        ("layer_idx",          ctypes.c_uint32),
        ("host_buf_offset",    ctypes.c_uint32),
        ("data_bytes",         ctypes.c_uint32),
        ("top1_prob",          ctypes.c_float),     # IPC-8g
        ("entropy",            ctypes.c_float),     # IPC-8g
        ("routed_miss_count",  ctypes.c_uint8),     # TD-89n: top-K experts not resident
        ("_pad_rmc",           ctypes.c_uint8 * 3),
    ]

class SeqOpCompletionPayload(ctypes.Structure):
    _fields_ = [
        ("seq_id",     ctypes.c_uint64),
        ("page_count", ctypes.c_uint32),
        ("_pad",       ctypes.c_uint32),
    ]

class NvmeCompletionPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("op",         ctypes.c_uint8),
        ("_pad",       ctypes.c_uint8),
    ]

class CancelCompletionPayload(ctypes.Structure):
    _fields_ = [
        ("target_cmd_seq", ctypes.c_uint32),
        ("cancelled",      ctypes.c_uint8),
        ("_pad",           ctypes.c_uint8 * 3),
    ]

class ElmExpertCompletionPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("_pad",       ctypes.c_uint8 * 2),
        ("nvme_us",    ctypes.c_uint32),
        ("dma_us",     ctypes.c_uint32),
        ("total_us",   ctypes.c_uint32),
    ]

class ElmProgressCompletionPayload(ctypes.Structure):
    _fields_ = [
        ("layer_idx",  ctypes.c_uint32),
        ("expert_idx", ctypes.c_uint16),
        ("phase",      ctypes.c_uint8),
        ("_pad",       ctypes.c_uint8),
    ]

class CheckpointCompletionPayload(ctypes.Structure):
    _fields_ = [
        ("cmd_type",        ctypes.c_uint32),
        ("layer_idx",       ctypes.c_uint32),
        ("checkpoint_type", ctypes.c_uint8),
        ("_pad",            ctypes.c_uint8 * 3),
        ("host_buf_offset", ctypes.c_uint32),
        ("data_bytes",      ctypes.c_uint32),
    ]

class ErrorCompletionPayload(ctypes.Structure):
    _fields_ = [
        ("error_category", ctypes.c_uint32),
        ("message",        ctypes.c_char * 80),
    ]

class GpuFatalPayload(ctypes.Structure):
    _fields_ = [
        ("vendor_error_code", ctypes.c_uint32),
        ("message",           ctypes.c_char * 108),
    ]

class CompletionPayload(ctypes.Union):
    _fields_ = [
        ("transfer",      TransferCompletionPayload),
        ("compute",       ComputeCompletionPayload),
        ("checkpoint",    CheckpointCompletionPayload),
        ("seq_op",        SeqOpCompletionPayload),
        ("nvme_op",       NvmeCompletionPayload),
        ("cancel_result", CancelCompletionPayload),
        ("elm_expert",    ElmExpertCompletionPayload),
        ("elm_progress",  ElmProgressCompletionPayload),
        ("error",         ErrorCompletionPayload),
        ("gpu_fatal",     GpuFatalPayload),
        ("raw",           ctypes.c_uint8 * 112),
    ]

# ── Completion (128 bytes) ───────────────────────────────────────────────────

class Completion(ctypes.Structure):
    _fields_ = [
        ("cmp_type", ctypes.c_uint32),
        ("cmd_seq",  ctypes.c_uint32),
        ("gpu_idx",  ctypes.c_uint32),
        ("status",   ctypes.c_uint32),
        ("payload",  CompletionPayload),
    ]

assert ctypes.sizeof(Completion) == 128, f"Completion size {ctypes.sizeof(Completion)} != 128"

# ── GpuSnapshot (56 bytes) ───────────────────────────────────────────────────

class GpuSnapshot(ctypes.Structure):
    _fields_ = [
        ("vram_used_bytes",       ctypes.c_uint64),
        ("vram_total_bytes",      ctypes.c_uint64),
        ("expert_stable_used",    ctypes.c_uint32),
        ("expert_stable_total",   ctypes.c_uint32),
        ("expert_streaming_used", ctypes.c_uint32),
        ("expert_streaming_total", ctypes.c_uint32),
        ("kv_main_free_pages",    ctypes.c_uint32),
        ("kv_spec_free_pages",    ctypes.c_uint32),
        ("inflight_h2d_count",    ctypes.c_uint32),
        ("inflight_d2h_count",    ctypes.c_uint32),
        ("compute_queue_depth",   ctypes.c_uint32),
        ("prefill_mode",          ctypes.c_uint8),
        ("_pad",                  ctypes.c_uint8 * 3),
    ]

assert ctypes.sizeof(GpuSnapshot) == 56, f"GpuSnapshot size {ctypes.sizeof(GpuSnapshot)} != 56"

# ── RequestAcceptance (16 bytes) ─────────────────────────────────────────────

class RequestAcceptance(ctypes.Structure):
    _fields_ = [
        ("request_id",       ctypes.c_uint64),
        ("acceptance_rate",  ctypes.c_double),
    ]

assert ctypes.sizeof(RequestAcceptance) == 16

# ── StateSnapshot ────────────────────────────────────────────────────────────
# Matches C++ layout including alignas(64) on seqlock via explicit padding.

_RESIDENCY_BITMAP_SIZE = MAX_MOE_LAYERS * MAX_EXPERTS * MAX_GPUS // 8  # 16384
_GPU_TRIPLE_SIZE       = MAX_MOE_LAYERS * MAX_EXPERTS * MAX_GPUS       # 131072
_EXPERT_ARRAY_SIZE     = MAX_MOE_LAYERS * MAX_EXPERTS                  # 16384
_HOST_BITMAP_SIZE      = MAX_MOE_LAYERS * MAX_EXPERTS // 8             # 2048

MAX_NUMA = 8
_NUMA_TIER_SIZE = MAX_MOE_LAYERS * MAX_EXPERTS * MAX_NUMA  # 131072

class StateSnapshot(ctypes.Structure):
    _fields_ = [
        # Seqlock (alignas(64) in C++ — seqlock at offset 0, then 56B padding)
        ("seqlock",                  ctypes.c_uint64),
        ("_pad_seq",                 ctypes.c_uint8 * 56),

        # Timestamps
        ("daemon_cycle_count",       ctypes.c_uint64),
        ("timestamp_ns",             ctypes.c_uint64),

        # Per-GPU state
        ("gpus",                     GpuSnapshot * MAX_GPUS),
        ("num_gpus",                 ctypes.c_uint32),
        ("_pad_ng",                  ctypes.c_uint8 * 4),

        # Expert residency bitmap
        ("residency_bitmap",         ctypes.c_uint8 * _RESIDENCY_BITMAP_SIZE),

        # ELM GPU-tier state (ELM-8, replaces sub_components)
        ("expert_gpu_tier",          ctypes.c_uint8 * _GPU_TRIPLE_SIZE),

        # ELM interest counts (ELM-8)
        ("expert_interest_count",    ctypes.c_uint8 * _GPU_TRIPLE_SIZE),

        # Host-tier bitmap: fast "is warm?" check (ELM-8)
        ("host_resident_bitmap",     ctypes.c_uint8 * _HOST_BITMAP_SIZE),

        # Per-expert host tier (ELM-8)
        ("host_tier",                ctypes.c_uint8 * _EXPERT_ARRAY_SIZE),

        # Expert statistics (float32)
        ("expert_frequency",         ctypes.c_float * _EXPERT_ARRAY_SIZE),
        ("expert_recency",           ctypes.c_float * _EXPERT_ARRAY_SIZE),
        ("expert_routing_weight",    ctypes.c_float * _EXPERT_ARRAY_SIZE),
        ("expert_temporal_autocorr", ctypes.c_float * _EXPERT_ARRAY_SIZE),

        # Acceptance tracking
        ("global_acceptance_rate",   ctypes.c_double),
        ("windowed_acceptance_rate", ctypes.c_double),
        ("layer_skip_acceptance_rate", ctypes.c_double),
        ("total_verifications",      ctypes.c_uint64),
        ("total_accepted_tokens",    ctypes.c_uint64),
        ("total_attempted_tokens",   ctypes.c_uint64),

        # Workload detector
        ("shift_detected",           ctypes.c_uint8),
        ("_pad1",                    ctypes.c_uint8 * 7),

        # Transfer engine
        ("total_inflight_transfers", ctypes.c_uint32),
        ("_pad_tit",                 ctypes.c_uint8 * 4),

        # Per-request acceptance ring
        ("per_request_acceptance",   RequestAcceptance * MAX_TRACKED_REQUESTS),
        ("num_tracked_requests",     ctypes.c_uint32),
        ("_pad_ntr",                 ctypes.c_uint8 * 4),

        # Host NUMA placement (int8, -1 = not warm)
        ("expert_host_numa",         ctypes.c_int8 * _EXPERT_ARRAY_SIZE),

        # Per-NUMA host tier (ELM-8b): HostTier per (layer, expert, NUMA node)
        # Indexed: layer * MAX_EXPERTS * MAX_NUMA + expert * MAX_NUMA + numa
        ("host_numa_tier",           ctypes.c_uint8 * _NUMA_TIER_SIZE),

        # Last ELM state change timestamp (ELM-8)
        ("expert_last_change_ns",    ctypes.c_uint64 * _EXPERT_ARRAY_SIZE),

        # Tail padding: C++ alignas(64) on seqlock forces sizeof to be
        # a multiple of 64. Without this, ctypes computes a smaller value.
        ("_pad_tail",                ctypes.c_uint8 * 32),
    ]

# ── EngineInfo ───────────────────────────────────────────────────────────────

class EngineInfo(ctypes.Structure):
    _fields_ = [
        ("ipc_base",        ctypes.c_uint64),
        ("ipc_total_bytes", ctypes.c_uint64),

        ("cmd_ring_offset", ctypes.c_uint64),
        ("cmd_ring_slots",  ctypes.c_uint32),
        ("cmd_slot_bytes",  ctypes.c_uint32),

        ("cmp_ring_offset", ctypes.c_uint64),
        ("cmp_ring_slots",  ctypes.c_uint32),
        ("cmp_slot_bytes",  ctypes.c_uint32),

        ("state_offset",    ctypes.c_uint64),
        ("state_bytes",     ctypes.c_uint64),

        ("sideband_offset", ctypes.c_uint64),
        ("sideband_bytes",  ctypes.c_uint64),

        ("num_gpus",        ctypes.c_int32),
        ("num_moe_layers",  ctypes.c_int32),
        ("num_experts",     ctypes.c_int32),
        # TD-PREFILL-SUPERCHUNK: effective MoE token-batch capacity (max
        # num_seqs for RUN_MOE / FETCH_AND_RUN_MOE) after the VRAM fail-safe.
        ("moe_batch_capacity", ctypes.c_int32),
        ("expert_bytes",    ctypes.c_int64),
        ("num_layers",      ctypes.c_int32),
        ("num_expert_devices", ctypes.c_int32),
        ("kv_bytes_per_page", ctypes.c_int64),
        # TD-VOCAB-AUTODETECT: the engine's resolved vocab width
        # (weights-derived or cross-checked); prefer over config/tokenizer.
        ("vocab_size",      ctypes.c_int32),

        # ── DeepSeek-V4 metadata (V4-7a) — all zero for non-V4 models ──
        ("v4_hc_mult",         ctypes.c_int32),   # mHC streams (0 = non-V4)
        ("v4_num_hash_layers", ctypes.c_int32),   # tid2eid-routed layers
        # Per hidden layer attention type from compress_ratios: 0 = SWA,
        # 1 = CSA (ratio 4), 2 = HCA (ratio 128).  First num_layers
        # entries valid when v4_hc_mult > 0 (kV4MaxLayers = 96).
        ("v4_attention_types", ctypes.c_uint8 * 96),
    ]

# ── IPC layout calculator ───────────────────────────────────────────────────

HEADER_SIZE = ctypes.sizeof(IpcHeader)
RING_HEADER_SIZE = ctypes.sizeof(RingHeader)


def cmd_ring_offset() -> int:
    return HEADER_SIZE


def cmd_ring_size(slots: int) -> int:
    return RING_HEADER_SIZE + slots * CMD_SLOT_BYTES


def cmp_ring_offset(cmd_slots: int) -> int:
    return cmd_ring_offset() + cmd_ring_size(cmd_slots)


def cmp_ring_size(slots: int) -> int:
    return RING_HEADER_SIZE + slots * CMP_SLOT_BYTES


def state_offset(cmd_slots: int, cmp_slots: int) -> int:
    return cmp_ring_offset(cmd_slots) + cmp_ring_size(cmp_slots)


def sideband_offset(cmd_slots: int, cmp_slots: int) -> int:
    return state_offset(cmd_slots, cmp_slots) + ctypes.sizeof(StateSnapshot)


def total_ipc_size(cmd_slots: int, cmp_slots: int) -> int:
    return sideband_offset(cmd_slots, cmp_slots) + SIDEBAND_TOTAL_SIZE
