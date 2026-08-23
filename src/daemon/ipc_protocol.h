#pragma once

// IPC protocol definitions for Python orchestrator <-> C++ daemon communication.
//
// This header is DEPENDENCY-FREE: it uses only primitive types so Python ctypes
// can match layouts exactly. No C++ domain types (ExpertKey, CacheZone, etc.).
//
// Type correspondence (for command dispatcher in IPC-4):
//   layer_idx (uint32_t) + expert_idx (uint16_t)  <->  ExpertKey
//   zone (uint8_t: 0=stable, 1=streaming)         <->  CacheZone
//   sub_component (uint8_t bitfield)               <->  SubComponent (kGate=0x01, kUp=0x02, kDown=0x04)
//   stream_id (uint32_t: 0-5)                      <->  StreamId enum
//   graph_type (uint8_t)                           <->  GraphType enum

#include <atomic>
#include <cstdint>
#include <cstring>

namespace layerstorm::ipc {

// ── Constants ───────────────────────────────────────────────────────────────

static constexpr uint32_t kProtocolVersion    = 1;
static constexpr int      kMaxGpus            = 8;
static constexpr int      kMaxMoeLayers       = 128;  // TD-IPC-MOE-LAYER-CAP: ≥ GLM-5.2's 75 MoE + MTP (was 64 → last 11 planning-blind)
static constexpr int      kMaxExperts         = 256;
static constexpr int      kMaxTrackedRequests = 64;
static constexpr int      kMaxNuma            = 8;

static constexpr uint32_t kDefaultCmdRingSlots = 8192;
static constexpr uint32_t kDefaultCmpRingSlots = 8192;
static constexpr uint32_t kCmdSlotBytes        = 256;
static constexpr uint32_t kCmpSlotBytes        = 128;

// Sideband sub-region max entry counts (IPC-8b).
static constexpr uint32_t kMaxBatchDescriptors = 512;
static constexpr uint32_t kMaxExpertPrefetch   = 256;
static constexpr uint32_t kMaxExpertEviction   = 256;
static constexpr uint32_t kMaxTransferBatch    = 256;
static constexpr uint32_t kMaxReserveBatch     = 256;
static constexpr uint32_t kMaxNvmeReadBatch    = 256;
static constexpr uint32_t kMaxSidebandTokenIds = 512;

// F-6: max inline per-entry gating weights carried in the fetch_and_run_moe
// command for the selective-fetch decider. Sized to the decode top-K
// (model.num_experts_per_tok = 8). When a fetch list exceeds this (e.g. EP
// broadcast of all routed experts), per-entry weight gating is disabled and the
// count-based deciders (min_experts/max_new_fetches) still apply.
static constexpr uint32_t kMaxFetchDeciderWeights = 8;
// F-4: routing-export sideband sub-region capacity. One (weight,index) pair per
// (token, top-K slot). Sized for the full decode batch (kMaxBatchDescriptors
// tokens) at a generous top-K cap (DeepSeek V3.2 uses 8; we allow up to 16).
static constexpr uint32_t kMaxRoutingExportTokens = kMaxBatchDescriptors;  // 512
static constexpr uint32_t kRoutingExportMaxTopk   = 16;

// Batched-verify OUTPUT_HEAD readback (MTP batched verification): when
// readback_to_host is set with num_tokens > 1, the daemon D2H-copies ALL
// num_tokens sampled (argmax) token ids into the sideband spec readback
// scratch — tokens u32[n] at [0..4n), top1_prob f32 at [4n..4n+4).  Capped
// so the readback always fits the spec readback slice
// (kSpecCheckpointSize - 2560 = 1536 bytes → 256*4 + 4 <= 1536 holds with
// margin).  num_tokens == 1 reproduces the historical layout byte-for-byte
// (token at [0..3], top1_prob at [4..7]).
static constexpr uint32_t kMaxOutputHeadReadbackTokens = 256;

// CMD_OUTPUT_HEAD readback_logits=1: number of full-logits rows the pinned
// host readback region (Engine::logits_readback_addr()) holds.  The daemon
// D2H-copies the FIRST min(num_tokens, kMaxLogitsReadbackRows) head-batch
// rows contiguously (row i of the batch → host row i) — num_tokens == 1
// keeps the historical single-row guided-decoding layout byte-identical.
// Multi-row consumers: the speculative sampled/logprobs verify chunk
// (TD-ORCH-SAMPLED-SPEC / TD-ORCH-LOGPROBS-SPEC) with R = gamma+1 <= 16
// rows.  Pinned cost: 16 rows x vocab x 4 B (~9.9 MB at GLM's 154880).
static constexpr uint32_t kMaxLogitsReadbackRows = 16;

// ── IpcHeader (256 bytes) ───────────────────────────────────────────────────
// Cache-line isolation: shutdown_requested and error_code on separate lines
// to prevent false sharing between daemon and orchestrator threads.

struct IpcHeader {
    uint32_t version;                            // Protocol version (kProtocolVersion)
    uint8_t  _pad_v[60];

    alignas(64) uint32_t shutdown_requested;     // 0 -> 1 triggers graceful shutdown
    uint8_t  _pad_s[60];

    alignas(64) uint32_t error_code;             // 0 = ok, nonzero = error
    char     error_message[124];                 // Null-terminated error string
};
static_assert(sizeof(IpcHeader) == 256);

// ── RingHeader (128 bytes) ──────────────────────────────────────────────────
// producer_seq and consumer_seq on separate cache lines.
// slot_count and slot_size are immutable after init (safe in consumer's line).

struct RingHeader {
    alignas(64) uint64_t producer_seq;           // Written by producer only
    uint8_t  _pad_p[56];

    alignas(64) uint64_t consumer_seq;           // Written by consumer only
    uint8_t  _pad_c[48];
    uint32_t slot_count;                         // Power of 2
    uint32_t slot_size;                          // Bytes per slot (256 cmd, 128 cmp)
};
static_assert(sizeof(RingHeader) == 128);

// ── CmdType ─────────────────────────────────────────────────────────────────

enum CmdType : uint32_t {
    // Transfer commands
    CMD_TRANSFER_H2D           = 0x0001,
    CMD_TRANSFER_D2H           = 0x0002,

    // Expert cache commands
    CMD_CACHE_RESERVE          = 0x0010,
    CMD_CACHE_EVICT            = 0x0011,
    CMD_CACHE_PROMOTE          = 0x0012,  // streaming -> stable
    CMD_CACHE_DEMOTE           = 0x0013,  // stable -> streaming

    // Compute kernel commands
    CMD_ATTENTION_DECODE       = 0x0100,
    CMD_ATTENTION_PREFILL      = 0x0101,
    CMD_GATING                 = 0x0102,
    CMD_EXPERT_FFN             = 0x0103,
    CMD_EMBEDDING_LOOKUP       = 0x0104,
    CMD_OUTPUT_HEAD            = 0x0105,
    CMD_RMSNORM                = 0x0106,
    CMD_SWIGLU                 = 0x0107,
    CMD_MOE_PERMUTE            = 0x0108,
    CMD_MOE_UNPERMUTE          = 0x0109,
    CMD_DCP_CORRECTION         = 0x010A,
    CMD_NCCL_ALLREDUCE         = 0x010B,
    CMD_DYNAMIC_FP8_QUANT      = 0x010C,
    CMD_SAMPLE_TOKENS          = 0x010D,  // GPU-side argmax/top-p/top-k sampling (IPC-8f)

    // Prefetch compute (Stream 5)
    CMD_PRESCOPE_GATING        = 0x0200,
    CMD_PROBE_MLP              = 0x0201,

    // CUDA graph
    CMD_GRAPH_REPLAY           = 0x0300,

    // Stream synchronization
    CMD_RECORD_EVENT           = 0x0400,
    CMD_STREAM_WAIT_EVENT      = 0x0401,

    // Placement optimizer (low-frequency, ~1 per 300s)
    CMD_COMPUTE_AFFINITY_HINTS = 0x0500,
    CMD_NUMA_MIGRATE           = 0x0501,

    // Sequence lifecycle (KV cache)
    CMD_SEQ_CREATE             = 0x0600,
    CMD_SEQ_FREE               = 0x0601,
    CMD_SEQ_FORK               = 0x0602,
    // Debug/test-only sequence-state checkpoint (long-context test restarts):
    // SNAPSHOT writes a sequence's cached state (main-KV pages + paged
    // indexer-K + coverage) to the file named by env LS_SEQ_CKPT_PATH;
    // RESTORE loads it back into a freshly created sequence. NOT a production
    // feature — no versioned stability guarantees beyond the build.
    CMD_SEQ_SNAPSHOT           = 0x0603,
    CMD_SEQ_RESTORE            = 0x0604,

    // NVMe tier (host RAM <-> NVMe) + transfer cancellation
    CMD_NVME_READ              = 0x0700,
    CMD_NVME_WRITE             = 0x0701,
    CMD_NVME_EVICT_HOST        = 0x0702,
    CMD_CANCEL_TRANSFER        = 0x0703,

    // Fused compute commands (IPC-8d)
    D_B_CMD_RUN_ATTENTION      = 0x0800,
    // DEPRECATED for routed MoE: production routed-MoE execution is
    // E_CMD_FETCH_AND_RUN_MOE (fused-gate routing export -> routed expert
    // list -> progressive fetch + run). RUN_MOE remains the path for
    // DENSE-FFN layers (no experts to fetch) and unit-test scaffolding.
    D_B_CMD_RUN_MOE            = 0x0801,

    // Fused batch + sequential commands (IPC-8e)
    D_B_CMD_PREFETCH_BATCH     = 0x0802,  // batch reserve + H2D via sideband
    D_B_CMD_EVICT_BATCH        = 0x0803,  // batch evict + optional D2H via sideband

    // Batch commands (IPC-8e)
    B_CMD_NVME_BATCH_READ      = 0x0900,  // batch NVMe→host via sideband

    // Sequential fusion commands (IPC-8e)
    D_CMD_PREFETCH_EXPERT      = 0x0A00,  // reserve + H2D single expert
    D_CMD_EVICT_TO_HOST        = 0x0A01,  // evict + optional D2H single expert
    D_CMD_STAGE_EXPERT         = 0x0A02,  // NVMe→host→VRAM single expert
    D_CMD_RUN_PREFETCH_PROBE   = 0x0A03,  // prescope + probe on Stream 5
    D_CMD_RUN_ADAPTER_FORWARD  = 0x0A04,  // Kangaroo adapter (norm→attn→norm)
    D_CMD_SLOW_EVICT_TO_HOST   = 0x0A05,  // D2H + VRAM evict (IPC-8i)
    D_CMD_RUN_MTP_STEP         = 0x0A06,  // fused MTP draft step (#62d)
    D_CMD_RUN_SELF_SPEC_FORWARD = 0x0A07, // fused self-spec forward pass (#62e)
    D_CMD_MTP_PROJECT          = 0x0A08,  // MTP projection: enorm(Emb)‖hnorm(h) → eh_proj (#16)
    D_CMD_RUN_DSPARK_STEP      = 0x0A09,  // fused DSpark DFlash-backbone draft step (DSP-3)

    // Config update (9.8-1b)
    CMD_CONFIG_UPDATE          = 0x0B00,  // live config parameter update

    // Extended commands — push decisions to C++ (IPC-8e)
    E_CMD_SEQ_CREATE           = 0x0C00,  // C++ handles page alloc + KV init + DCP routing
    E_CMD_SEQ_FREE             = 0x0C01,  // C++ handles full teardown + spill decisions
    E_CMD_FETCH_AND_RUN_MOE    = 0x0C02,  // progressive fetch + MoE execution (#90)
    E_FORWARD_ONE_LAYER        = 0x0C03,  // autonomous one-layer forward (attn+gate+MoE) (F-5)
    // TD-PREFILL-MOE-BIG: big-batch progressive fetch + CHUNKED MoE execution.
    // Same sideband contract as E_CMD_FETCH_AND_RUN_MOE (expert list at
    // kExpertPrefetchOff, optional eviction map, F-6 decider fields), extended
    // two ways: (1) the grouped GEMMs run in ~moe_big_chunk_tokens chunks with
    // the transient scratch REUSED per chunk (never-OOM at any num_seqs up to
    // EngineInfo.moe_batch_capacity); (2) the rolling expert waves are DOUBLE-
    // BUFFERED — wave i+1's H2D fetches are issued while wave i's chunk-GGEMMs
    // compute, keeping the transfer engine saturated.
    E_CMD_FETCH_AND_RUN_MOE_BIG = 0x0C04,
    // REEF placement/eviction solve over IPC (the Python orchestrator's REEF
    // arm — P-18 line). Orchestrator writes the routed union at
    // kExpertPrefetchOff (layer_idx + expert_idx per entry; gpu_idx/zone
    // IGNORED on input), sends this; the daemon runs the engine-hosted
    // ReefOrch service (core/gpu_loader/reef_orch.h — the SAME decision
    // stack the KEEPER52_REEF_ORCH test arm drives: calibrated LoaderSolver
    // /LoaderSolver256 + EvictScoreBoard, self-consistent residency model)
    // and REWRITES entries[i].gpu_idx in place + fills the index-aligned
    // ExpertEvictionEntry[] at kExpertEvictionOff (0xFFFF sentinel = no
    // victim), then completes CMP_COMPUTE_DONE (cmd_type/layer_idx echoed).
    // The orchestrator then issues E_CMD_FETCH_AND_RUN_MOE with
    // have_evict_map=1 over the SAME sideband — ring order makes the
    // handoff race-free. Service is built lazily on the first command
    // (calibration from gpu_loader.calibration_path resolved against the
    // weights dir; caps from the live ExpertCache stable totals; policy
    // from LS_LOADER_POLICY + env overrides; bank lookup from the live
    // PinnedExpertArena) — construction failure is CMP_ERROR, fail loud.
    // CPU-only: never touches CUDA.
    E_CMD_REEF_ROUTE           = 0x0C05,
    // Fused per-layer forward through the PRODUCTION routed-MoE seam (named
    // to not collide with E_FORWARD_ONE_LAYER, which runs the deprecated
    // RUN_MOE path). ONE command = D_B_CMD_RUN_ATTENTION (fused gate +
    // routing export on MoE layers; batch descriptors from the sideband as
    // usual) + daemon-side wait for the attention stream + the deduped
    // first-occurrence routed union + placement per route_mode (0 = static
    // e%num_gpus, no victim map; 1 = the ReefOrch service incl. the
    // eviction map) + the E_CMD_FETCH_AND_RUN_MOE execution path — ONE
    // CMP_COMPUTE_DONE at the end whose data_bytes carries the deduped
    // entry count (the orchestrator's `lookups` accumulator; documented
    // reuse of the readback-size field — there is no readback here).
    // Dense layers (< first_k_dense_replace): attention + the dense-FFN
    // RUN_MOE path, same single completion, data_bytes=0. Byte-equivalence
    // contract: identical kernels in identical order to the two/three
    // command sequence it replaces.
    E_CMD_FAR_FORWARD_LAYER    = 0x0C06,

    // Lifecycle
    CMD_SHUTDOWN               = 0xFF00,
    CMD_NOOP                   = 0xFFFF,
};

// ── Command (256 bytes) ─────────────────────────────────────────────────────
// Python orchestrator -> C++ daemon via command ring.

struct Command {
    // ── Common header (16 bytes) ──
    uint32_t cmd_type;     // CmdType enum value
    uint32_t cmd_seq;      // Monotonic sequence for completion matching
    uint32_t gpu_idx;      // Target GPU position index
    uint32_t stream_id;    // StreamId (0-5)

    // ── Payload (240 bytes) ──
    union {
        // CMD_TRANSFER_H2D / CMD_TRANSFER_D2H
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint8_t  sub_component;     // SubComponent bitfield
            uint8_t  zone;              // 0=stable, 1=streaming
            int64_t  bytes;             // Transfer size
            float    priority;          // Staging queue ordering (0.0 = default)
            uint32_t delay_us;          // JIT delay passthrough (0 = immediate)
        } transfer;

        // CMD_CACHE_RESERVE
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint8_t  zone;              // 0=stable, 1=streaming
            uint8_t  is_duplicate;      // Nonzero = duplicate copy
        } cache_reserve;

        // CMD_CACHE_EVICT / CMD_CACHE_PROMOTE / CMD_CACHE_DEMOTE
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint16_t _pad;
        } cache_op;

        // CMD_ATTENTION_DECODE / CMD_ATTENTION_PREFILL
        // TD-ATTN-LEGACY (resolved): routed through the same production path
        // as D_B_CMD_RUN_ATTENTION (DcpExecutor: absorption + RoPE + scale;
        // KV metadata from the batch-descriptor sideband). The buf_id fields
        // are retained for layout stability but IGNORED — hidden state and
        // KV cache come from the engine-wired buffers. Prefer
        // D_B_CMD_RUN_ATTENTION for new code.
        struct {
            uint32_t layer_idx;
            uint32_t batch_size;
            uint8_t  use_graph;         // Nonzero = replay captured CUDA graph
            uint8_t  is_sparse;         // Nonzero = DSA sparse attention
            uint16_t _pad;
            uint32_t hidden_state_buf_id;  // ignored (legacy)
            uint32_t kv_cache_buf_id;      // ignored (legacy)
            uint32_t seqlens_buf_id;       // ignored (legacy)
            uint32_t block_table_buf_id;   // ignored (legacy)
        } attention;

        // CMD_GATING
        struct {
            uint32_t layer_idx;
            uint32_t num_tokens;
            uint32_t input_buf_id;
            uint32_t output_weights_buf_id;
            uint32_t output_indices_buf_id;
        } gating;

        // CMD_EXPERT_FFN
        struct {
            uint32_t layer_idx;
            uint32_t num_experts;
            uint32_t total_tokens;
            uint32_t permuted_input_buf_id;
            uint32_t output_buf_id;
            uint32_t expert_offsets_buf_id;
            uint32_t hidden_dim;           // KD-2: output dim N (gate/up: I; down: H)
            uint32_t weights_buf_id;       // KD-2: device pointer table [num_experts × 3]
            uint32_t workspace_buf_id;     // KD-2: GEMM workspace
            uint32_t k_dim;                // GG-5: input dim K (gate/up: H; down: I); GGUF path
            uint8_t  quant_mode;           // KD-2: 0=NVFP4, 1=FP8, 2=GGUF
            uint8_t  gguf_type;            // GG-5: model::GgufKQuantType (Q2_K=0..Q8_0=5); quant_mode==2 only
            uint8_t  _pad_eff[2];
        } expert_ffn;

        // CMD_PRESCOPE_GATING
        struct {
            uint32_t target_layer_idx;
            uint32_t num_tokens;
            uint32_t hidden_state_buf_id;
            uint32_t output_buf_id;
        } prescope;

        // CMD_PROBE_MLP
        struct {
            uint32_t probe_point_idx;   // 0=25%, 1=50%, 2=75%
            uint32_t num_tokens;
            uint32_t hidden_state_buf_id;
            uint32_t output_buf_id;
        } probe;

        // CMD_EMBEDDING_LOOKUP (token IDs from sideband at kTokenIdsOff)
        struct {
            uint32_t num_tokens;
            uint32_t output_buf_id;
            uint32_t row_offset;     // TD-PREFILL-SUPERCHUNK: destination ROW
                                     // offset into the output hidden buffer
                                     // (rows [row_offset, row_offset+num_tokens)).
                                     // 0 = legacy behavior.
        } embedding_lookup;

        // CMD_OUTPUT_HEAD
        struct {
            uint32_t num_tokens;
            uint32_t input_buf_id;
            uint32_t output_buf_id;
            uint8_t  readback_to_host;     // Nonzero = async D2H copy of logits
            uint8_t  compute_confidence;   // IPC-8g: nonzero = return {top1_prob, entropy}
            uint8_t  num_logprobs;         // 0=disabled, 1-20=return top-K logprobs in readback
            uint8_t  mtp_head;             // #16: 0 = main head; else MTP module (mtp_idx+1):
                                           //   shared_head.norm replaces final_norm; MTP
                                           //   shared_head.head replaces lm_head when present
                                           //   (GGUF dedups it into the main head → fallback).
            uint8_t  readback_logits;      // TD-SERVE-NAMED-TOOL-CHOICE / TD-ORCH-SAMPLED-SPEC:
                                           //   nonzero = async D2H copy of the FIRST
                                           //   min(num_tokens, kMaxLogitsReadbackRows) rows'
                                           //   FULL logits (vocab_size f32 each, contiguous)
                                           //   into the pinned host buffer at
                                           //   Engine::logits_readback_addr() — host-visible
                                           //   when CMP_COMPUTE_DONE fires.  num_tokens == 1 is
                                           //   the historical guided-decoding single row; the
                                           //   speculative sampled/logprobs verify chunk reads
                                           //   all R rows.  Host-side masking/sampling (B=1).
        } output_head;

        // D_B_CMD_RUN_ATTENTION — full attention block (sideband batch descriptor)
        struct {
            uint32_t layer_idx;
            uint32_t num_seqs;       // Sequence count in sideband batch descriptor
            uint8_t  is_prefill;     // 0=decode, 1=prefill
            uint8_t  use_graph;      // 1=replay CUDA graph if available
            uint8_t  is_draft;           // IPC-8d: 1=checkpoint hidden state to pinned host
            uint8_t  emit_checkpoint;    // 1=emit CMP_CHECKPOINT at midpoint
            uint32_t chunk_start;    // Chunked prefill: token offset (0 for decode)
            uint32_t chunk_len;      // Chunked prefill: token count (0 for decode)
            uint8_t  emit_gating;    // F-1: 1=run router+topk gating at end of attention,
                                     //      writing topk_weights/topk_indices to moe_scratch_
            uint8_t  store_gating;   // F-3: 1=also export routed top-K to the
                                     //      kRoutingExportOff sideband slot (needs emit_gating)
            uint8_t  superchunk;     // TD-PREFILL-SUPERCHUNK: 1 = this launch is a
                                     //      SUB-CHUNK of a superchunk (layer-wise
                                     //      replay of K sub-chunks). Relaxes the
                                     //      indexer coverage `repeat` guard to the
                                     //      superchunk window (end <= next_pos).
            uint8_t  _pad_ra;
            uint32_t row_offset;     // TD-PREFILL-SUPERCHUNK: hidden-state ROW
                                     //      offset for this sub-chunk — attention
                                     //      reads/writes attn_buf/moe_buf rows
                                     //      [row_offset, row_offset+num_seqs) and the
                                     //      fused gate stores topk at the same row
                                     //      offset in moe_scratch_. Batch descriptors
                                     //      stay at sideband rows [0, num_seqs).
                                     //      0 = legacy behavior.
        } run_attention;  // 28B

        // D_B_CMD_RUN_MOE — full MoE block (sideband batch descriptor).
        // DEPRECATED for routed MoE (use E_CMD_FETCH_AND_RUN_MOE);
        // retained for dense-FFN layers + tests.
        struct {
            uint32_t layer_idx;
            uint32_t num_seqs;                    // Sequence count in sideband batch descriptor
            uint8_t  moe_mode;                    // 0=normal, 1=truncation, 2=substitution
            uint8_t  apply_residual_correction;   // 1=run correction MLP after FFN
            uint8_t  store_gating_output;         // 1=write gating weights to buffer
            uint8_t  emit_checkpoint;             // 1=emit CMP_CHECKPOINT at midpoint
            uint8_t  use_precomputed_gating;      // F-2: 1=consume topk already in moe_scratch_, skip internal gate (default 0=self-gate)
        } run_moe;  // 13B

        // D_B_CMD_PREFETCH_BATCH — batch reserve + H2D (sideband: ExpertPrefetchEntry[])
        struct {
            uint32_t count;     // entries at IpcLayout::kExpertPrefetchOff
            float    priority;  // shared priority for all entries in batch
            uint32_t delay_us;  // shared JIT delay for all entries in batch
            uint32_t _pad;
        } prefetch_batch;  // 16B

        // D_B_CMD_EVICT_BATCH — batch evict + optional D2H (sideband: ExpertEvictionEntry[])
        struct {
            uint32_t count;     // entries at IpcLayout::kExpertEvictionOff
            uint32_t _pad;
        } evict_batch;  // 8B

        // B_CMD_NVME_BATCH_READ — batch NVMe→host (sideband: NvmeReadEntry[])
        struct {
            uint32_t count;     // entries at IpcLayout::kNvmeReadOff
            uint32_t _pad;
        } nvme_batch_read;  // 8B

        // D_CMD_PREFETCH_EXPERT — reserve VRAM slot + enqueue H2D
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint8_t  zone;       // CacheZone: 0=streaming, 1=stable
            uint8_t  gpu_idx;
            float    priority;   // Staging queue ordering (0.0 = default)
            uint32_t delay_us;   // JIT delay passthrough (0 = immediate)
        } prefetch_expert;  // 16B

        // D_CMD_EVICT_TO_HOST — evict from VRAM, optional D2H spill to host
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint16_t gpu_idx;
        } evict_to_host;  // 8B

        // D_CMD_SLOW_EVICT_TO_HOST — D2H + VRAM evict (IPC-8i)
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint16_t gpu_idx;
        } slow_evict_to_host;  // 8B

        // D_CMD_STAGE_EXPERT — full cold path: NVMe→host RAM→VRAM
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint8_t  zone;
            uint8_t  gpu_idx;
            float    priority;   // Staging queue ordering (0.0 = default)
            uint32_t delay_us;   // JIT delay passthrough (0 = immediate)
        } stage_expert;  // 16B

        // D_CMD_RUN_PREFETCH_PROBE — prescope gating + probe MLP on Stream 5
        struct {
            uint32_t target_layer;
            uint32_t num_tokens;
            uint8_t  probe_points;  // bitmask: bit0=25%, bit1=50%, bit2=75%
            uint8_t  _pad[3];
        } run_prefetch_probe;  // 12B

        // D_CMD_RUN_ADAPTER_FORWARD — Kangaroo adapter (norm→attn→norm)
        struct {
            uint32_t num_tokens;
            uint32_t adapter_weights_buf_id;
        } run_adapter_forward;  // 8B

        // D_CMD_RUN_MTP_STEP — fused MTP draft step (#62d)
        // Daemon pipeline: embed → MTP projection → attention → MoE → output head
        // Caller contract: the sideband batch descriptor at
        // kBatchDescriptorOff must carry (seq_id, token_pos) for the draft
        // position BEFORE this command is issued (same as RUN_ATTENTION) —
        // the MTP layer's attention/KV append reads it.
        struct {
            uint32_t mtp_layer_idx;    // MTP layer index (num_layers + step % num_mtp_layers)
            uint32_t _pad0;            // alignment: ctypes pads uint32 before uint64
            uint64_t seq_id;           // draft fork sequence ID
            uint32_t input_token_id;   // token to embed for this step
            uint8_t  step_idx;         // MTP step index (0-based, for logging)
            uint8_t  _pad1[3];
        } run_mtp_step;  // 24B

        // D_CMD_MTP_PROJECT — MTP projection only (#16 / GLM-25g).
        // eh_proj(concat(enorm(Emb(input_token)), hnorm(prev_hidden))) →
        // hidden-state pair attn_buf on every TP rank.  prev_hidden is the
        // CURRENT attn_buf content: the trunk hidden of the last main-model
        // forward (draft step 0) or the MTP layer output of the previous
        // draft step (chained steps) — the caller sequences commands so that
        // the right hidden is resident.  Composes the production-seam MTP
        // draft step with existing commands:
        //   D_CMD_MTP_PROJECT → D_B_CMD_RUN_ATTENTION(mtp_layer, emit/store
        //   gating) → E_CMD_FETCH_AND_RUN_MOE(mtp_layer) →
        //   CMD_OUTPUT_HEAD(mtp_head) → CMD_SAMPLE_TOKENS.
        struct {
            uint32_t mtp_layer_idx;    // absolute MTP layer index (>= num_hidden_layers)
            uint32_t input_token_id;   // token to embed for this step
            uint8_t  step_idx;         // 0-based chain index (logging)
            uint8_t  hidden_row;       // prev-hidden ROW of attn_buf to hnorm
                                       // (batched-verify catch-ups: the trunk
                                       //  hiddens of the K-token verify pass
                                       //  sit in attn_buf rows [0..K); MTP
                                       //  steps write only row 0, so higher
                                       //  rows survive a sequential chain).
                                       // 0 = historical single-row behavior.
            uint8_t  _pad[2];
        } mtp_project;  // 12B

        // D_CMD_RUN_DSPARK_STEP — fused DSpark draft step (DSP-3).
        // ONE DFlash-backbone forward over the whole gamma block: query
        // slots [anchor_token, mask x (num_query-1)] at positions
        // anchor_pos + [0..num_query), non-causal inside the block,
        // attending to the ingested context KV at positions < anchor_pos.
        // Emits base logits [num_query, V] FP32 + hidden states
        // [num_query, H] BF16 in draft-GPU scratch (slot k predicts
        // position query_pos + 1 — INV-DSPARK-ANCHOR).  DSP-4 (Markov
        // head) and DSP-6 (confidence head) consume both on-device.
        // Caller contract: the target forward must have run with the
        // dspark aux-hidden export armed (speculation.method=dspark) so the
        // draft context KV covers positions [0, anchor_pos) of seq_id; the
        // command fails with CMP_ERROR otherwise (fail closed).  The
        // command executes on the DRAFT GPU regardless of header gpu_idx.
        // DSP-5 readback: the completion carries host_buf_offset/data_bytes
        // of the gamma sampled draft ids (i32), copied D2H into the
        // sideband readback scratch (kSpecCheckpointOff + 2560) before the
        // completion event — host-visible when CMP_COMPUTE_DONE fires.
        // DSP-6: when speculation.dspark.confidence_enabled, the gamma raw
        // survival probabilities c_k (f32, trained confidence head —
        // INV-DSPARK-CONF) follow the ids in the same slice; data_bytes =
        // gamma * 4 * 2 then (both sides know confidence_enabled from
        // config).
        struct {
            uint64_t seq_id;           // context identity (must match the
                                       // runtime's tracked/ingested context)
            uint32_t anchor_token_id;  // last accepted token (query slot 0)
            uint32_t anchor_pos;       // slot-0 position == accepted length
            uint8_t  num_query;        // gamma (0 = config speculative_tokens;
                                       // <= block_size)
            uint8_t  step_idx;         // 0-based draft step (logging)
            uint8_t  _pad[2];
        } run_dspark_step;  // 20B

        // D_CMD_RUN_SELF_SPEC_FORWARD — fused self-spec forward pass (#62e)
        // Daemon pipeline: embed → for each non-skipped layer: attention + MoE(top-K) → output head
        struct {
            uint64_t seq_id;                // draft fork sequence ID
            uint32_t input_token_id;        // token to embed for this position
            uint8_t  draft_expert_count;    // experts per MoE layer (default 1)
            uint8_t  apply_residual_corr;   // 1 = apply residual correction MLP
            uint8_t  store_gating;          // 1 = checkpoint gating output at every MoE layer
            uint8_t  step_idx;              // draft position index (for logging)
            uint64_t skip_mask_lo;          // layer skip bitmask bits 0-63
            uint64_t skip_mask_hi;          // layer skip bitmask bits 64-127
        } self_spec_forward;  // 32B

        // E_CMD_FETCH_AND_RUN_MOE — progressive fetch + MoE execution (#90)
        // Expert list in sideband at kExpertPrefetchOff (ExpertPrefetchEntry[]).
        //
        // F-6: selective-fetch decider. The orchestrator writes prefetch entries
        // in gating SELECTION-RANK order (the top-K order the gating kernel emits —
        // by biased score). NOTE: this is NOT weight-sorted — V3.2 selects by
        // sigmoid+e_score_correction_bias but the per-entry `weight` is the unbiased
        // sigmoid score (INV-10c-2), so weights are not monotonic across entries.
        // min_experts/max_new_fetches honor position (= selection rank = model
        // priority); the weight threshold is per-entry (position-independent). The
        // daemon then fetches only the selected missing experts (H2D), running the
        // resident rest —
        // generalizing the progressive+timeout partial-MoE path. Missing experts
        // that are deliberately not fetched degrade gracefully exactly like a
        // timeout (excluded from the bitset, counted in routed_miss_count).
        // All decider fields default to 0 → today's behavior (fetch every entry).
        struct {
            uint32_t layer_idx;
            uint32_t num_seqs;       // Sequence count in sideband batch descriptor
            uint32_t expert_count;   // Entries at kExpertPrefetchOff
            uint32_t timeout_us;     // Deadline for missing expert arrivals (0 = no timeout)
            uint8_t  moe_mode;       // 0=normal, 1=truncation, 2=substitution
            // ── F-6 decider params ──
            uint8_t  weight_count;   // # valid entries in weights[] (0 = threshold inactive)
            uint16_t min_experts;    // always include >= this many leading entries (0 = none)
            uint16_t max_new_fetches;// cap H2D issues for missing experts (0 = unlimited)
            // 13c-2.0 Option A: when 1, an index-aligned ExpertEvictionEntry[] at
            // kExpertEvictionOff supplies, per fetched (expert→gpu) transfer, the
            // resident victim to evict on that GPU (expert_idx==0xFFFF = none →
            // local fallback). Honored subject to lock/needed-this-layer guards.
            uint8_t  have_evict_map; // 0 = local (unranked) eviction only
            uint8_t  _pad;
            float    gating_weight_threshold;  // skip fetching missing experts below this (0 = no cutoff)
            // Optional per-entry gating weights (entry i ↔ weights[i]), used only
            // when weight_count > 0. Sized to the decode top-K (kMaxFetchDeciderWeights).
            float    weights[kMaxFetchDeciderWeights];
        } fetch_and_run_moe;

        // E_CMD_FETCH_AND_RUN_MOE_BIG — big-batch fetch + CHUNKED MoE
        // (TD-PREFILL-MOE-BIG). Layout-identical PREFIX to fetch_and_run_moe
        // (the handler reads the common fields through cmd.fetch_and_run_moe;
        // C++ common-initial-sequence guarantees the aliasing) plus:
        //   chunk_tokens — grouped-GEMM chunk size override, clamped to the
        //                  engine's chunk scratch capacity
        //                  (compute.moe_big_chunk_tokens). 0 = engine default.
        // num_seqs may exceed kMaxBatchDescriptors up to
        // EngineInfo.moe_batch_capacity (the elastic superchunk capacity).
        struct {
            uint32_t layer_idx;
            uint32_t num_seqs;
            uint32_t expert_count;
            uint32_t timeout_us;
            uint8_t  moe_mode;
            uint8_t  weight_count;
            uint16_t min_experts;
            uint16_t max_new_fetches;
            uint8_t  have_evict_map;
            uint8_t  _pad;
            float    gating_weight_threshold;
            float    weights[kMaxFetchDeciderWeights];
            uint32_t chunk_tokens;   // 0 = compute.moe_big_chunk_tokens default
        } fetch_and_run_moe_big;

        // E_CMD_REEF_ROUTE — REEF placement/eviction solve (see CmdType doc)
        struct {
            uint32_t layer_idx;
            uint32_t expert_count;   // Entries at kExpertPrefetchOff
        } reef_route;  // 8B

        // E_CMD_FAR_FORWARD_LAYER — fused attention + routed FETCH_AND_RUN
        // layer (see CmdType doc). route_mode: 0 = static e%num_gpus (no
        // victim map), 1 = ReefOrch service (entries + eviction map).
        struct {
            uint32_t layer_idx;
            uint32_t num_seqs;       // Rows in the sideband batch descriptor
            uint32_t chunk_start;    // Chunked prefill: token offset (0 decode)
            uint32_t chunk_len;      // Chunked prefill: token count (0 decode)
            uint32_t timeout_us;     // FETCH deadline passthrough (0 = none)
            uint8_t  is_prefill;     // 0=decode, 1=prefill/chunk shape
            uint8_t  route_mode;     // 0=static e%num_gpus, 1=reef service
            uint8_t  _pad[2];
        } far_forward_layer;  // 24B

        // E_FORWARD_ONE_LAYER — autonomous one-layer forward (F-5)
        // Daemon delegates to forward_one_layer(): attention + gate + MoE for a
        // single decode layer. Output is byte-equivalent to issuing
        // D_B_CMD_RUN_ATTENTION followed by D_B_CMD_RUN_MOE for the same layer.
        // Batch descriptor read from sideband at kBatchDescriptorOff, like the
        // two-op form. Completes once with CMP_COMPUTE_DONE on the MoE stream.
        struct {
            uint32_t layer_idx;
            uint32_t num_seqs;       // Sequence count in sideband batch descriptor
            int32_t  topk_override;  // 0 = use model num_experts_per_tok default
            uint8_t  is_prefill;     // 0=decode, 1=prefill
            uint8_t  use_graph;      // 1=replay CUDA graph if available
            uint8_t  is_draft;       // 1=draft (speculation) pass
            uint8_t  store_gating;   // 1=export gating output to sideband
        } forward_one_layer;  // 16B

        // CMD_RMSNORM
        struct {
            uint32_t num_tokens;
            uint32_t input_buf_id;
            uint32_t output_buf_id;
            uint32_t weight_buf_id;
            float    eps;
            uint32_t hidden_size;          // KD-2: dimension (model.hidden_size, kv_lora_rank, etc.)
        } rmsnorm;

        // CMD_SWIGLU
        struct {
            uint32_t num_tokens;
            uint32_t hidden_dim;        // Per-expert hidden_dim
            uint32_t input_buf_id;
            uint32_t output_buf_id;
        } swiglu;

        // CMD_MOE_PERMUTE
        struct {
            uint32_t num_tokens;
            uint32_t num_experts;
            uint32_t input_buf_id;
            uint32_t output_buf_id;
            uint32_t indices_buf_id;
            uint32_t offsets_buf_id;
            uint32_t topk;                 // KD-2: model.num_experts_per_tok
            uint32_t hidden_dim;           // KD-2: model.hidden_size
            uint32_t workspace_buf_id;     // KD-2: scratch for src_to_dest_map + permuted_idx
        } moe_permute;

        // CMD_MOE_UNPERMUTE
        struct {
            uint32_t num_tokens;
            uint32_t num_experts;
            uint32_t input_buf_id;
            uint32_t output_buf_id;
            uint32_t indices_buf_id;       // src_to_dest_map from permute
            uint32_t topk;                 // KD-2: model.num_experts_per_tok
            uint32_t hidden_dim;           // KD-2: model.hidden_size
            uint32_t weights_buf_id;       // KD-2: topk_weights [num_tokens, topk] FP32
        } moe_unpermute;

        // CMD_DCP_CORRECTION
        struct {
            uint32_t batch_size;
            uint32_t input_buf_id;
            uint32_t output_buf_id;
            uint32_t lse_buf_id;
        } dcp_correction;

        // CMD_NCCL_ALLREDUCE
        struct {
            uint32_t count;             // Number of elements
            uint32_t buf_id;            // In-place reduce buffer
            uint8_t  dtype;             // 0=BF16, 1=FP32
            uint8_t  _pad[3];
        } nccl_allreduce;

        // CMD_DYNAMIC_FP8_QUANT
        struct {
            uint32_t num_tokens;
            uint32_t hidden_dim;
            uint32_t input_buf_id;
            uint32_t output_buf_id;
            uint32_t scales_buf_id;
        } dynamic_fp8_quant;

        // CMD_SAMPLE_TOKENS — GPU-side token sampling (IPC-8f)
        // Input logits: [num_tokens, vocab_size] FP32 in logits_buf_id.
        // Output: sampled token IDs written to sideband at kTokenIdsOff.
        // Modes: temperature<=0 -> argmax; top_k>0 -> top-K; top_p<1.0 -> nucleus.
        // Mutates logits in-place (temperature scaling). Caller must not reuse.
        struct {
            uint32_t num_tokens;       // Tokens to sample (one logit row each)
            uint32_t logits_buf_id;    // Input: [num_tokens, vocab_size] FP32 logits
            uint32_t vocab_size;       // Logit dimension (e.g. 129280)
            uint32_t top_k;            // Top-K cutoff (0 = disabled)
            float    temperature;      // Temperature (<=0.0 = argmax)
            float    top_p;            // Nucleus threshold (1.0 = disabled)
            uint64_t random_seed;      // RNG seed for reproducibility
        } sample_tokens;  // 32B

        // CMD_GRAPH_REPLAY
        struct {
            uint8_t  graph_type;        // GraphType enum value
            uint8_t  _pad[3];
            uint32_t batch_size;
        } graph;

        // CMD_RECORD_EVENT / CMD_STREAM_WAIT_EVENT
        struct {
            uint32_t event_id;
        } event;

        // CMD_COMPUTE_AFFINITY_HINTS
        struct {
            uint32_t num_gpus;
            uint32_t gpu_capacity_slots[kMaxGpus];
        } affinity_hints;

        // CMD_NUMA_MIGRATE
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint8_t  target_numa_node;
            uint8_t  _pad;
        } numa_migrate;

        // CMD_SEQ_CREATE — gpu_idx header field is the default GPU
        struct {
            uint64_t seq_id;           // Orchestrator-assigned sequence ID
            uint32_t prompt_len;       // Number of prompt tokens to pre-allocate
            uint8_t  pool;             // 0=kMain, 1=kSpeculation
            uint8_t  _pad[3];
        } seq_create;

        // CMD_SEQ_FREE
        struct {
            uint64_t seq_id;           // Sequence to free
        } seq_free;

        // CMD_SEQ_SNAPSHOT / CMD_SEQ_RESTORE — debug/test-only checkpoint.
        // File path comes from env LS_SEQ_CKPT_PATH (in-process test only).
        struct {
            uint64_t seq_id;
            uint32_t token_count;   // positions [0, token_count) are cached
        } seq_ckpt;

        // CMD_SEQ_FORK
        struct {
            uint64_t src_seq_id;       // Source sequence to fork from
            uint64_t dst_seq_id;       // New forked sequence ID (CoW)
        } seq_fork;

        // CMD_NVME_READ — read expert from NVMe into host RAM warm cache
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint8_t  gpu_hint;         // NUMA node hint for host buffer allocation
            uint8_t  _pad;
        } nvme_read;

        // CMD_NVME_WRITE — write expert from host RAM to NVMe
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint16_t _pad;
        } nvme_write;

        // CMD_NVME_EVICT_HOST — discard expert from host RAM (INV-4.12e)
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint16_t _pad;
        } nvme_evict_host;

        // CMD_CANCEL_TRANSFER — cancel in-flight PCIe or NVMe transfer
        struct {
            uint32_t target_cmd_seq;   // cmd_seq of the originating transfer
            uint32_t _pad;
        } cancel_transfer;

        // CMD_CONFIG_UPDATE — live config parameter update (9.8-1b)
        struct {
            uint32_t count;            // Number of field updates (max 29)
            uint32_t _pad;
            struct {
                uint16_t field_id;     // FieldId enum value
                uint8_t  value_type;   // 0=bool, 1=int32, 2=float
                uint8_t  _pad;
                uint32_t raw_value;    // Reinterpreted per value_type
            } entries[29];             // 29 × 8 = 232 bytes + 8 header = 240
        } config_update;

        uint8_t raw[240];
    };
};
static_assert(sizeof(Command) == 256);

// ── CompletionType ──────────────────────────────────────────────────────────

enum CompletionType : uint32_t {
    CMP_TRANSFER_DONE = 0x0001,
    CMP_CACHE_OP_DONE = 0x0010,
    CMP_COMPUTE_DONE  = 0x0100,
    CMP_CHECKPOINT    = 0x0200,  // Mid-execution data available in pinned buffer
    CMP_EVENT_STATUS  = 0x0400,
    CMP_SEQ_OP_DONE   = 0x0600,
    CMP_NVME_DONE     = 0x0700,
    CMP_CANCEL_DONE   = 0x0703,

    CMP_CONFIG_UPDATE_DONE = 0x0B00,

    // ELM lifecycle completions
    CMP_ELM_EXPERT_READY    = 0x0800,  // Expert is HOT on target GPU
    CMP_ELM_EXPERT_EVICTED  = 0x0801,  // Expert async eviction complete (ELM-6)
    CMP_ELM_EXPERT_PROGRESS = 0x0802,  // Mid-lifecycle FSM transition (stats)

    CMP_ERROR         = 0xEE00,
    CMP_GPU_FATAL     = 0xEF00,  // GPU device-fatal (INV-5c), at most once per GPU per poll path
};

// ── CheckpointType ─────────────────────────────────────────────────────────

enum class CheckpointType : uint8_t {
    kHiddenState      = 0,  // D_B_CMD_RUN_ATTENTION: hidden state written
    kGatingOutput     = 1,  // D_B_CMD_RUN_MOE: gating output written
    kLayerSimilarity  = 2,  // D_CMD_RUN_SELF_SPEC_FORWARD: daemon-computed cos_sim (#62g)
    kSeamRouting      = 3,  // F-7: attention↔MoE seam — routed top-K (weights+indices)
                            //      published to kSeamCheckpointOff. Layout:
                            //        topk_weights[expanded] f32, then
                            //        topk_indices[expanded] i32  (expanded = num_seqs*topk)
};

// ── CMP_ERROR category codes ────────────────────────────────────────────────

enum class CmpErrorCategory : uint32_t {
    kUnknownCommand       = 1,   // Unrecognized cmd_type
    kException            = 2,   // std::exception during dispatch
    kCacheSlotMissing     = 3,   // ExpertCache lookup failed (no VRAM slot)
    kHostSourceMissing    = 4,   // Host-side expert weight not found
    kNvmeTierMissing      = 5,   // NvmeTier not configured
    kCacheReserveFailed   = 6,   // ExpertCache zone full
    kGraphError           = 7,   // Invalid graph type or graph not captured
    kEventPoolError       = 8,   // Event ID not found in pool
    kNumaMissing          = 9,   // NvmeTier not configured for NUMA migrate
    kComputeValidation    = 10,  // Compute dispatch validation (buffers, devices, params)
    kSeqCreate            = 11,  // Sequence create/append errors
    kSeqFree              = 12,  // Sequence free errors
    kSeqFork              = 13,  // Sequence fork errors
    kNvmeRead             = 14,  // NVMe read errors
    kNvmeWrite            = 15,  // NVMe write errors
    kNvmeEvict            = 16,  // NVMe evict/host-tier errors
    kReserveBatch         = 17,  // Batch reserve errors
    kNvmeBatchRead        = 18,  // NVMe batch read errors (legacy path)
    kPrefetchBatch        = 19,  // Prefetch batch errors
    kEvictBatch           = 20,  // Evict batch errors
    kNvmeBatchReadNew     = 21,  // NVMe batch read errors (new path)
    kElmExpertOp          = 22,  // ELM expert operation errors
    kAdapterForward       = 23,  // Adapter forward errors
    kConfigUpdate         = 24,  // Config update errors
    kGpuFatal             = 25,  // GPU device-fatal during compute poll
    kPrefetchExpert       = 26,  // Single expert prefetch errors (legacy path)
    kFetchAndRunMoe       = 27,  // Progressive fetch-and-run MoE errors (#90)
    kForwardOneLayer      = 28,  // Autonomous one-layer forward errors (F-5)
    kKvPoolExhausted      = 29,  // KV page pool exhausted (or unknown seq) during
                                 // attention KV metadata build (TD-GOLDEN-KV-EXHAUST)
};

/// Default human-readable message for each error category.
inline const char* default_message(CmpErrorCategory cat) {
    switch (cat) {
        case CmpErrorCategory::kUnknownCommand:    return "unknown command type";
        case CmpErrorCategory::kException:         return "dispatch exception";
        case CmpErrorCategory::kCacheSlotMissing:  return "expert cache slot not found";
        case CmpErrorCategory::kHostSourceMissing: return "host expert weight not found";
        case CmpErrorCategory::kNvmeTierMissing:   return "NVMe tier not configured";
        case CmpErrorCategory::kCacheReserveFailed: return "expert cache reserve failed";
        case CmpErrorCategory::kGraphError:        return "graph error";
        case CmpErrorCategory::kEventPoolError:    return "event pool error";
        case CmpErrorCategory::kNumaMissing:       return "NUMA manager not configured";
        case CmpErrorCategory::kComputeValidation: return "compute dispatch validation error";
        case CmpErrorCategory::kSeqCreate:         return "sequence create error";
        case CmpErrorCategory::kSeqFree:           return "sequence free error";
        case CmpErrorCategory::kSeqFork:           return "sequence fork error";
        case CmpErrorCategory::kNvmeRead:          return "NVMe read error";
        case CmpErrorCategory::kNvmeWrite:         return "NVMe write error";
        case CmpErrorCategory::kNvmeEvict:         return "NVMe evict error";
        case CmpErrorCategory::kReserveBatch:      return "batch reserve error";
        case CmpErrorCategory::kNvmeBatchRead:     return "NVMe batch read error";
        case CmpErrorCategory::kPrefetchBatch:     return "prefetch batch error";
        case CmpErrorCategory::kEvictBatch:        return "evict batch error";
        case CmpErrorCategory::kNvmeBatchReadNew:  return "NVMe batch read error";
        case CmpErrorCategory::kElmExpertOp:       return "ELM expert operation error";
        case CmpErrorCategory::kAdapterForward:    return "adapter forward error";
        case CmpErrorCategory::kConfigUpdate:      return "config update error";
        case CmpErrorCategory::kGpuFatal:          return "GPU device-fatal error";
        case CmpErrorCategory::kPrefetchExpert:    return "expert prefetch error";
        case CmpErrorCategory::kFetchAndRunMoe:    return "progressive fetch-and-run MoE error";
        case CmpErrorCategory::kForwardOneLayer:   return "forward-one-layer error";
        case CmpErrorCategory::kKvPoolExhausted:   return "KV page pool exhausted";
    }
    return "unknown error";
}

// ── Completion (128 bytes) ──────────────────────────────────────────────────
// C++ daemon -> Python orchestrator via completion ring.

struct Completion {
    // ── Common header (16 bytes) ──
    uint32_t cmp_type;     // CompletionType enum value
    uint32_t cmd_seq;      // Matches Command.cmd_seq for correlation
    uint32_t gpu_idx;      // GPU that completed the work
    uint32_t status;       // 0 = success, nonzero = error code

    // ── Payload (112 bytes) ──
    union {
        // CMP_TRANSFER_DONE
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint8_t  direction;     // 0=H2D, 1=D2H
            uint8_t  _pad;
            uint64_t vram_address;  // Final VRAM address
        } transfer;

        // CMP_COMPUTE_DONE
        // F-4 routing contract: when an op published routed top-K, this same
        // CMP_COMPUTE_DONE is the signal — the routing itself lives in the
        // sideband routing-export slot (IpcLayout::kRoutingExportOff), NOT in
        // this struct. Do not add a routing payload here; the Completion stays a
        // fixed 128 B and the consumer reads the slot after CMP_COMPUTE_DONE.
        struct {
            uint32_t cmd_type;        // Original CmdType from Command
            uint32_t layer_idx;
            uint32_t host_buf_offset; // CMD_OUTPUT_HEAD readback: offset into pinned host buffer
            uint32_t data_bytes;      // CMD_OUTPUT_HEAD readback: logit data size (0 if no readback)
            float    top1_prob;       // IPC-8g: max softmax probability (0.0 if not requested)
            float    entropy;         // IPC-8g: normalized Shannon entropy [0,1] (0.0 if not requested)
            uint8_t  routed_miss_count; // TD-89n: top-K experts not resident (0 = all hit)
            uint8_t  _pad_rmc[3];
        } compute;

        // CMP_CHECKPOINT — mid-execution data available
        struct {
            uint32_t cmd_type;           // Original CmdType
            uint32_t layer_idx;
            uint8_t  checkpoint_type;    // CheckpointType: 0=hidden_state, 1=gating_output,
                                         //   2=layer_similarity, 3=seam_routing (F-7)
            uint8_t  _pad[3];
            uint32_t host_buf_offset;    // Offset into pinned checkpoint/sideband buffer
                                         //   (seam_routing: kSeamCheckpointOff)
            uint32_t data_bytes;         // Size of checkpointed data
        } checkpoint;  // 20B

        // CMP_SEQ_OP_DONE
        struct {
            uint64_t seq_id;        // Sequence that was operated on
            uint32_t page_count;    // Pages allocated/freed/forked
            uint32_t _pad;
        } seq_op;

        // CMP_NVME_DONE — NVMe read/write/evict completed
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint8_t  op;            // 0=read, 1=write, 2=evict_host
            uint8_t  _pad;
        } nvme_op;

        // CMP_CANCEL_DONE — cancellation result
        struct {
            uint32_t target_cmd_seq;
            uint8_t  cancelled;     // 1=successfully cancelled, 0=not found
            uint8_t  _pad[3];
        } cancel_result;

        // CMP_ELM_EXPERT_READY / CMP_ELM_EXPERT_EVICTED
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint8_t  _pad[2];
            uint32_t nvme_us;       // NVMe I/O latency (read on load, write on evict; 0 if warm/mmap)
            uint32_t dma_us;        // PCIe DMA latency (H2D on load, D2H on evict)
            uint32_t total_us;      // Total lifecycle latency
        } elm_expert;

        // CMP_ELM_EXPERT_PROGRESS — mid-lifecycle FSM transition
        struct {
            uint32_t layer_idx;
            uint16_t expert_idx;
            uint8_t  phase;         // 0=nvme_started, 1=nvme_done, 2=reserved, 3=h2d_started
            uint8_t  _pad;
        } elm_progress;

        // CMP_ERROR
        struct {
            uint32_t error_category;
            char     message[80];   // Null-terminated
        } error;

        // CMP_GPU_FATAL — at most once per GPU per poll path on device-fatal error
        struct {
            uint32_t vendor_error_code;  // e.g. cudaError_t value
            char     message[108];       // Null-terminated
        } gpu_fatal;

        uint8_t raw[112];
    };
};
static_assert(sizeof(Completion) == 128);

// ── GpuSnapshot (per-GPU state within StateSnapshot) ────────────────────────

struct GpuSnapshot {
    uint64_t vram_used_bytes;
    uint64_t vram_total_bytes;
    uint32_t expert_stable_used;
    uint32_t expert_stable_total;
    uint32_t expert_streaming_used;
    uint32_t expert_streaming_total;
    uint32_t kv_main_free_pages;
    uint32_t kv_spec_free_pages;
    uint32_t inflight_h2d_count;
    uint32_t inflight_d2h_count;
    uint32_t compute_queue_depth;
    uint8_t  prefill_mode;          // 0=normal, 1=spill_active
    uint8_t  _pad[3];
};
static_assert(sizeof(GpuSnapshot) == 56);

// ── RequestAcceptance (per-request tracking entry) ──────────────────────────

struct RequestAcceptance {
    uint64_t request_id;
    double   acceptance_rate;
};
static_assert(sizeof(RequestAcceptance) == 16);

// ── StateSnapshot (~820 KB) ─────────────────────────────────────────────────
// C++ daemon writes under seqlock. Python reader retries on odd or mismatch.
// All expert arrays indexed as [layer * kMaxExperts + expert].
// GPU-triple arrays indexed as [layer * kMaxExperts * kMaxGpus + expert * kMaxGpus + gpu].

struct StateSnapshot {
    // ── Seqlock (cache-line isolated) ──
    alignas(64) uint64_t seqlock;   // Even=readable, odd=writing
    uint8_t _pad_seq[56];

    // ── Timestamps ──
    uint64_t daemon_cycle_count;
    uint64_t timestamp_ns;

    // ── Per-GPU state ──
    GpuSnapshot gpus[kMaxGpus];
    uint32_t num_gpus;
    uint8_t  _pad_ng[4];

    // ── Expert residency bitmap ──
    // 1 bit per (layer, expert, gpu) triple.
    // Bit index = layer * kMaxExperts * kMaxGpus + expert * kMaxGpus + gpu.
    uint8_t residency_bitmap[kMaxMoeLayers * kMaxExperts * kMaxGpus / 8];

    // ── ELM GPU-tier state (replaces sub_components, ELM-8) ──
    // GpuTier enum per (layer, expert, gpu).  Published event-driven by ELM.
    uint8_t expert_gpu_tier[kMaxMoeLayers * kMaxExperts * kMaxGpus];

    // ── ELM interest counts (ELM-8) ──
    // Active interest (reference) count per (layer, expert, gpu).
    uint8_t expert_interest_count[kMaxMoeLayers * kMaxExperts * kMaxGpus];

    // ── Host-tier bitmap (fast "is warm?" check, ELM-8) ──
    // Bit index = layer * kMaxExperts + expert.
    uint8_t host_resident_bitmap[kMaxMoeLayers * kMaxExperts / 8];

    // ── Per-expert host tier (ELM-8) ──
    // HostTier enum per (layer, expert).  Published event-driven by ELM.
    uint8_t host_tier[kMaxMoeLayers * kMaxExperts];

    // ── Expert statistics (normalized [0,1]) ──
    float expert_frequency[kMaxMoeLayers * kMaxExperts];
    float expert_recency[kMaxMoeLayers * kMaxExperts];
    float expert_routing_weight[kMaxMoeLayers * kMaxExperts];
    float expert_temporal_autocorr[kMaxMoeLayers * kMaxExperts];

    // ── Acceptance tracking ──
    double   global_acceptance_rate;
    double   windowed_acceptance_rate;
    double   layer_skip_acceptance_rate;
    uint64_t total_verifications;
    uint64_t total_accepted_tokens;
    uint64_t total_attempted_tokens;

    // ── Workload detector ──
    uint8_t  shift_detected;        // Nonzero = workload shift detected
    uint8_t  _pad1[7];

    // ── Transfer engine ──
    uint32_t total_inflight_transfers;
    uint8_t  _pad_tit[4];

    // ── Per-request acceptance ring ──
    RequestAcceptance per_request_acceptance[kMaxTrackedRequests];
    uint32_t num_tracked_requests;
    uint8_t  _pad_ntr[4];

    // ── Host NUMA placement (per-expert, -1 = not warm) ──
    // Indexed as [layer * kMaxExperts + expert].
    int8_t expert_host_numa[kMaxMoeLayers * kMaxExperts];

    // ── Per-NUMA host tier (ELM-8b) ──
    // HostTier enum per (layer, expert, NUMA node).  NUMA-inner indexing:
    // host_numa_tier[layer * kMaxExperts * kMaxNuma + expert * kMaxNuma + numa].
    uint8_t host_numa_tier[kMaxMoeLayers * kMaxExperts * kMaxNuma];

    // ── Last ELM state change timestamp (ELM-8) ──
    // Nanosecond timestamp per (layer, expert).  Published event-driven by ELM.
    uint64_t expert_last_change_ns[kMaxMoeLayers * kMaxExperts];
};

static_assert(sizeof(StateSnapshot) == 1676928);  // TD-IPC-MOE-LAYER-CAP: kMaxMoeLayers 64→128 doubles the per-(layer,expert) arrays

// ── Seqlock helpers ─────────────────────────────────────────────────────────
// Used by daemon (writer) and state publisher. Python has its own protocol.

inline void seqlock_write_begin(StateSnapshot& snap) {
    auto* seq = reinterpret_cast<volatile uint64_t*>(&snap.seqlock);
    *seq = *seq + 1;  // Now odd (write in progress)
    std::atomic_thread_fence(std::memory_order_release);
}

inline void seqlock_write_end(StateSnapshot& snap) {
    std::atomic_thread_fence(std::memory_order_release);
    auto* seq = reinterpret_cast<volatile uint64_t*>(&snap.seqlock);
    *seq = *seq + 1;  // Now even (readable)
}

/// Returns the sequence number. Spins while odd (write in progress).
inline uint64_t seqlock_read_begin(const StateSnapshot& snap) {
    uint64_t seq;
    do {
        seq = *reinterpret_cast<const volatile uint64_t*>(&snap.seqlock);
        std::atomic_thread_fence(std::memory_order_acquire);
    } while (seq & 1);
    return seq;
}

/// Returns true if snapshot was not modified since seqlock_read_begin.
inline bool seqlock_read_validate(const StateSnapshot& snap, uint64_t seq) {
    std::atomic_thread_fence(std::memory_order_acquire);
    return *reinterpret_cast<const volatile uint64_t*>(&snap.seqlock) == seq;
}

// ── Sideband entry structs (IPC-8b) ────────────────────────────────────────
// Variable-length data passed from Python to C++ via sideband region.
// See spec/IPC_FUSIONS.md §"Sideband region layout" for field semantics.

/// Batch descriptor: one entry per active sequence in a forward pass.
struct BatchDescriptorEntry {
    uint64_t seq_id;
    uint32_t token_pos;
    uint32_t _pad;
};
static_assert(sizeof(BatchDescriptorEntry) == 16);

/// Expert prefetch list entry.
struct ExpertPrefetchEntry {
    uint32_t layer_idx;
    uint16_t expert_idx;
    uint8_t  zone;       // 0=stable, 1=streaming
    uint8_t  gpu_idx;
};
static_assert(sizeof(ExpertPrefetchEntry) == 8);

/// Expert eviction list entry.
/// Note: spec lists uint16_t gpu_idx (10B) but entry size is 8B.
/// gpu_idx < kMaxGpus=8, so uint8_t suffices.
struct ExpertEvictionEntry {
    uint32_t layer_idx;
    uint16_t expert_idx;
    uint8_t  gpu_idx;
    uint8_t  _pad;
};
static_assert(sizeof(ExpertEvictionEntry) == 8);

/// Transfer batch list entry: one expert transfer.
struct TransferBatchEntry {
    uint32_t layer_idx;
    uint16_t expert_idx;
    uint8_t  sub_component;  // SubComponent bitfield
    uint8_t  zone;           // 0=stable, 1=streaming
    int64_t  bytes;
};
static_assert(sizeof(TransferBatchEntry) == 16);

/// Cache reserve batch list entry.
struct ReserveBatchEntry {
    uint32_t layer_idx;
    uint16_t expert_idx;
    uint8_t  zone;           // 0=stable, 1=streaming
    uint8_t  is_duplicate;
};
static_assert(sizeof(ReserveBatchEntry) == 8);

/// NVMe read batch list entry.
struct NvmeReadEntry {
    uint32_t layer_idx;
    uint16_t expert_idx;
    uint8_t  gpu_hint;       // NUMA node hint for host buffer allocation
    uint8_t  _pad;
};
static_assert(sizeof(NvmeReadEntry) == 8);

// Token IDs sub-region: raw uint32_t[], no struct needed.

// ── F-4: Routing export contract (RUN_ATTENTION [emit_gating] / MoE gating) ──
//
// Defines the documented sideband slot where the daemon publishes routed top-K
// (weights + expert indices) so an orchestrator/test can read routing WITHOUT a
// second gate and WITHOUT packing it into the Completion struct. The routing is
// signalled by the op's normal CMP_COMPUTE_DONE (zero-copy sideband is the
// established LayerStoRm pattern; the Completion struct stays a fixed 128 B and
// carries no routing payload — see the `compute` completion comment below).
//
// Producer (current code): dispatch_moe_internal, gated by the run_moe
// `store_gating_output` flag — after Step 1 gating it D2H-copies the top-K it is
// about to consume into this slot. Forward-compatible producer (F-1/F-3):
// RUN_ATTENTION [emit_gating] will populate the SAME slot at attention-end, so
// the contract (offset + layout below) is the stable seam and only the producer
// moves. Default (flag off) writes nothing — byte-for-byte unchanged.
//
// Layout at kRoutingExportOff:
//   RoutingExportHeader header;                 // num_tokens, topk
//   float   weights[num_tokens * topk];         // top-K routing weights, row-major
//   int32_t indices[num_tokens * topk];         // top-K expert indices, row-major
// weights[] starts at kRoutingExportWeightsOff, indices[] at
// kRoutingExportIndicesOff (both fixed; header occupies the first 16 bytes).
struct RoutingExportHeader {
    uint32_t num_tokens;   // Tokens whose routing was published (0 = none)
    uint32_t topk;         // Routed experts per token (model.num_experts_per_tok)
    uint32_t layer_idx;    // Layer the routing belongs to
    uint32_t _pad;
};
static_assert(sizeof(RoutingExportHeader) == 16);

// ── IPC region layout calculator ────────────────────────────────────────────

struct IpcLayout {
    static constexpr uint64_t kHeaderOffset = 0;
    static constexpr uint64_t kHeaderSize   = sizeof(IpcHeader);  // 256

    static constexpr uint64_t cmd_ring_offset() {
        return kHeaderSize;
    }

    static constexpr uint64_t cmd_ring_size(uint32_t slots) {
        return sizeof(RingHeader) + static_cast<uint64_t>(slots) * kCmdSlotBytes;
    }

    static constexpr uint64_t cmp_ring_offset(uint32_t cmd_slots) {
        return cmd_ring_offset() + cmd_ring_size(cmd_slots);
    }

    static constexpr uint64_t cmp_ring_size(uint32_t slots) {
        return sizeof(RingHeader) + static_cast<uint64_t>(slots) * kCmpSlotBytes;
    }

    static constexpr uint64_t state_offset(uint32_t cmd_slots, uint32_t cmp_slots) {
        return cmp_ring_offset(cmd_slots) + cmp_ring_size(cmp_slots);
    }

    // ── Sideband region (IPC-8b) ────────────────────────────────────────

    static constexpr uint64_t sideband_offset(uint32_t cmd_slots, uint32_t cmp_slots) {
        return state_offset(cmd_slots, cmp_slots) + sizeof(StateSnapshot);
    }

    // Sub-region offsets relative to sideband start.
    // Fixed at compile time — command type determines which sub-region to read.
    static constexpr uint64_t kBatchDescriptorOff  = 0;
    static constexpr uint64_t kBatchDescriptorSize =
        kMaxBatchDescriptors * sizeof(BatchDescriptorEntry);                // 8192

    static constexpr uint64_t kExpertPrefetchOff   =
        kBatchDescriptorOff + kBatchDescriptorSize;                         // 8192
    static constexpr uint64_t kExpertPrefetchSize  =
        kMaxExpertPrefetch * sizeof(ExpertPrefetchEntry);                   // 2048

    static constexpr uint64_t kExpertEvictionOff   =
        kExpertPrefetchOff + kExpertPrefetchSize;                           // 10240
    static constexpr uint64_t kExpertEvictionSize  =
        kMaxExpertEviction * sizeof(ExpertEvictionEntry);                   // 2048

    static constexpr uint64_t kTransferBatchOff    =
        kExpertEvictionOff + kExpertEvictionSize;                           // 12288
    static constexpr uint64_t kTransferBatchSize   =
        kMaxTransferBatch * sizeof(TransferBatchEntry);                     // 4096

    static constexpr uint64_t kReserveBatchOff     =
        kTransferBatchOff + kTransferBatchSize;                             // 16384
    static constexpr uint64_t kReserveBatchSize    =
        kMaxReserveBatch * sizeof(ReserveBatchEntry);                       // 2048

    static constexpr uint64_t kNvmeReadOff         =
        kReserveBatchOff + kReserveBatchSize;                               // 18432
    static constexpr uint64_t kNvmeReadSize        =
        kMaxNvmeReadBatch * sizeof(NvmeReadEntry);                          // 2048

    static constexpr uint64_t kTokenIdsOff         =
        kNvmeReadOff + kNvmeReadSize;                                       // 20480
    static constexpr uint64_t kTokenIdsSize        =
        kMaxSidebandTokenIds * sizeof(uint32_t);                            // 2048

    // KD-3c: speculation checkpoint data sub-region
    static constexpr uint64_t kSpecCheckpointOff   =
        kTokenIdsOff + kTokenIdsSize;                                       // 22528
    static constexpr uint64_t kSpecCheckpointSize  = 4096;                  // 4KB
    // Layout within kSpecCheckpointOff:
    //   [0..511]:    per-layer cos_sim floats (128 layers * 4 bytes)
    //   [512..2559]: per-layer gating weights (128 * topk(8) * sizeof(float))
    //   [2560..4095]: readback scratch (token_id u32, confidence f32, ...;
    //                 DSP-5: D_CMD_RUN_DSPARK_STEP writes the gamma sampled
    //                 draft ids [gamma <= 16] i32 here — one readback
    //                 consumer at a time per ring ordering; DSP-6: with
    //                 confidence_enabled the gamma raw survival c_k f32
    //                 follow the ids, gamma*8 <= 128 bytes total)

    // F-4: routing-export sub-region. Documented slot for routed top-K published
    // by RUN_ATTENTION [emit_gating] (or, today, MoE gating with
    // store_gating_output=1). See RoutingExportHeader above for the layout.
    static constexpr uint64_t kRoutingExportOff        =
        kSpecCheckpointOff + kSpecCheckpointSize;                           // 26624
    static constexpr uint64_t kRoutingExportWeightsOff =
        kRoutingExportOff + sizeof(RoutingExportHeader);                    // 26640
    static constexpr uint64_t kRoutingExportIndicesOff =
        kRoutingExportWeightsOff
        + static_cast<uint64_t>(kMaxRoutingExportTokens)
              * kRoutingExportMaxTopk * sizeof(float);                      // 26640 + 32768
    static constexpr uint64_t kRoutingExportSize       =
        sizeof(RoutingExportHeader)
        + static_cast<uint64_t>(kMaxRoutingExportTokens)
              * kRoutingExportMaxTopk * (sizeof(float) + sizeof(int32_t));  // 16 + 65536

    // F-7: attention↔MoE seam-routing checkpoint sub-region — appended AFTER the
    // F-4 routing-export slot (cluster reconciliation). Distinct mechanism from
    // F-4: this is the completion-signalled CMP_CHECKPOINT{kSeamRouting} payload,
    // whereas kRoutingExportOff is the polled slot. (Both publish routed top-K;
    // dedup tracked as TD-F-cluster-rt.) Layout within kSeamCheckpointOff:
    //   [0 .. 4*expanded):            topk_weights[expanded] f32
    //   [4*expanded .. 8*expanded):   topk_indices[expanded] i32
    static constexpr uint64_t kSeamCheckpointOff   =
        kRoutingExportOff + kRoutingExportSize;                            // after F-4 slot
    static constexpr uint64_t kMaxSeamRoutingExpanded = 512;
    static constexpr uint64_t kSeamCheckpointSize  =
        kMaxSeamRoutingExpanded * (sizeof(float) + sizeof(int32_t));        // 4096

    static constexpr uint64_t kSidebandTotalSize   =
        kSeamCheckpointOff + kSeamCheckpointSize;

    static constexpr uint64_t total_size(uint32_t cmd_slots, uint32_t cmp_slots) {
        return sideband_offset(cmd_slots, cmp_slots) + kSidebandTotalSize;
    }
};

// 481-3: H2D source-path snapshot surfaced by Engine::h2d_path_stats() for
// benchmarks. Mirrors ExpertLifecycleManager::H2dPathStats without pulling the
// ELM header into IPC-only consumers (tests). `direct` = pinned-pool source
// (full bandwidth); `staged` = unpinned source routed through staging.
struct EngineH2dPathStats {
    uint64_t direct = 0;
    uint64_t staged = 0;
    // 481-3 sub-phase x-ray (from TransferEngine::h2d_phase_stats):
    // memcpy_ns = host→pinned-staging CPU copy (0 for direct sources);
    // dma_wait_ns = DMA-enqueue → completion-detected; phase_count = H2D xfers.
    uint64_t memcpy_ns   = 0;
    uint64_t dma_wait_ns = 0;
    uint64_t phase_count = 0;
    // perf_trace pure-DMA x-ray (0 unless perf_trace enabled): dma_gpu_ns =
    // Σ GPU-measured memcpy time; dma_gpu_count = transfers that contributed.
    // dma_wait_ns − dma_gpu_ns ≈ host detection/queue overhead.
    uint64_t dma_gpu_ns    = 0;
    uint64_t dma_gpu_count = 0;
};

// ── EngineInfo (returned by start_engine to Python) ─────────────────────────

struct EngineInfo {
    uintptr_t ipc_base;
    uint64_t  ipc_total_bytes;

    uint64_t  cmd_ring_offset;
    uint32_t  cmd_ring_slots;
    uint32_t  cmd_slot_bytes;

    uint64_t  cmp_ring_offset;
    uint32_t  cmp_ring_slots;
    uint32_t  cmp_slot_bytes;

    uint64_t  state_offset;
    uint64_t  state_bytes;

    uint64_t  sideband_offset;    // Offset from ipc_base to sideband region (IPC-8b)
    uint64_t  sideband_bytes;     // Total sideband region size (kSidebandTotalSize)

    // Model metadata
    int32_t   num_gpus;
    int32_t   num_moe_layers;
    int32_t   num_experts;
    int32_t   moe_batch_capacity;      // TD-PREFILL-SUPERCHUNK: effective MoE token
                                       // batch capacity (max num_seqs for RUN_MOE /
                                       // FETCH_AND_RUN_MOE). Equals
                                       // max(kMaxBatchDescriptors,
                                       //     compute.prefill_superchunk_tokens)
                                       // after the VRAM fail-safe step-down.
    int64_t   expert_bytes;
    int32_t   num_layers;
    int32_t   num_expert_devices;
    int64_t   kv_bytes_per_page;       // Data bytes per KV page (for Python VRAM headroom)
    int32_t   vocab_size;              // TD-VOCAB-AUTODETECT: the engine's resolved
                                       // vocab width (weights-derived when the config
                                       // field was 0/absent; cross-checked otherwise).
                                       // Preferred over any config/tokenizer read.

    // ── DeepSeek-V4 metadata (V4-7a) — all zero for non-V4 models ─────────
    static constexpr int kV4MaxLayers = 96;  // covers Flash (43) and Pro (61)
    int32_t   v4_hc_mult;              // mHC stream count (0 = non-V4)
    int32_t   v4_num_hash_layers;      // hash-gated MoE layers (tid2eid routing)
    // Per hidden layer attention type from compress_ratios:
    // 0 = SWA-only, 1 = CSA (ratio 4), 2 = HCA (ratio 128). Valid for the
    // first num_layers entries when v4_hc_mult > 0.
    uint8_t   v4_attention_types[kV4MaxLayers];
};

}  // namespace layerstorm::ipc
