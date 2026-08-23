#pragma once

// Command dispatcher: maps CmdType enum values from the IPC command ring
// to C++ module method calls.
//
// Converts IPC primitive types to domain types, manages async completion
// tracking (transfer token → cmd_seq, compute event → cmd_seq), and writes
// completions to the completion ring.
//
// Thread safety: NOT thread-safe. Called exclusively from daemon thread
// (INV-3.4.2).

#include <array>
#include <chrono>   // LS_ATTN_CHUNK_PROF: chunk-attention dispatch x-ray
#include <cstddef>
#include <cstdint>
#include <cstdio>   // FILE* (LS_LOADER_SHADOW_DUMP JSONL sink)
#include <functional>  // lifecycle_pump_ (queued FAR/FETCH drain)
#include <memory>
#include <string>
#include <optional>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "compute/graphs/nccl_group_graph.h"  // INV-NCCL-GRAPH: fused-combine runner
#include "compute/graphs/routed_ffn_graph.h"
#include "config/config_parser.h"
#include "core/gpu_loader/loader_solver.h"   // I8: persistent LoaderSolver member (by value)
#include "core/gpu_loader/reef_orch.h"       // E_CMD_REEF_ROUTE service (P-18 REEF arm)
#include "core/gpu_loader/loader_evict_scores.h"  // I8: per-GPU evict-score board (hot path)
#include "core/gpu_loader/loader_place_cons.h"    // I8: place_cons table (hot path)
#include "core/gpu_loader/loader_place_sum.h"     // residency-reframe weighted-sum place_cons
#include "core/memory/page_allocator.h"
#include "core/transfer/host_source.h"
#include "daemon/ipc_protocol.h"
#include "model/quantization/gguf_kquant.h"      // GG-5c: per-projection GgufKQuantType on dense/shared weights
#include "model/quantization/quant_interface.h"  // ExpertShape (used by value in Deps)
#include "model/weight_loader/weight_loader.h"   // GG-9: GgufModelExpertTypes (per-layer routed types)
#include "parallelism/dcp_executor.h"   // AttentionLayerWeights (used by value)

// Forward declarations — full headers only in .cpp
namespace layerstorm::ipc {
template <uint32_t> class SpscRing;
using CompletionRing = SpscRing<128>;
}  // namespace layerstorm::ipc

namespace layerstorm::transfer {
class TransferEngine;
}

namespace layerstorm::memory {
class ExpertCache;
class NvmeTier;
class NumaManager;
class PinnedExpertArena;
}  // namespace layerstorm::memory

namespace layerstorm::compute {
class DeviceBackend;
class StreamManager;
class GraphRegistry;
class AttentionDevice;
class ExpertDevice;
enum class StreamId : int;  // KD-R5: forward decl for OutputHeadOpts
}  // namespace layerstorm::compute

namespace layerstorm::statistics {
class CoactivationGraph;
class ExpertStats;
}

namespace layerstorm::parallelism {
class DcpCommunicator;
}  // namespace layerstorm::parallelism

namespace layerstorm::model {
struct LoadedModel;
class PrepackedSource;
class PackedBufferCache;
class GgufQuantInterface;  // GG-5b: per-projection GGUF type for fused MoE GEMMs
}

namespace layerstorm::speculation {
class DsparkRuntime;  // DSP-3: DFlash backbone + aux-hidden export runtime
class EpmRoutingDumper;  // EPM-1: routing-label dump writer
}

namespace layerstorm::daemon {

class BufferRegistry;
class ExpertLifecycleManager;
class KvTieringManager;
class V4KvTiering;  // GLM-25k (kv_tiering_manager.h)

/// KD-R2: Typed wrapper for the rank↔gpu hidden-state buffer pair.
/// Bundles the attention buffer (indexed by rank), MoE buffer (indexed by
/// GPU position), the rank↔gpu mapping, and per-pair sync events.
struct HiddenStatePair {
    void* attn_buf       = nullptr;  ///< attention reads/writes here (rank-indexed)
    void* moe_buf        = nullptr;  ///< MoE reads/writes here (gpu-position-indexed)
    int rank             = -1;       ///< DCP rank
    int gpu_position     = -1;       ///< GPU position index
    int gpu_id           = -1;       ///< CUDA device ordinal
    void* attn_moe_event = nullptr;  ///< attention-done → MoE sync
    void* moe_attn_event = nullptr;  ///< MoE-done → next-layer attention sync

    /// Copy moe_buf → attn_buf (MoE output → attention input for next layer).
    /// Enqueues async D2D memcpy on the given stream via DeviceBackend.
    void commit(size_t bytes, void* stream, compute::DeviceBackend* dev) const;
};

/// Tracks a compute command dispatched to a GPU stream but not yet completed.
struct PendingCompute {
    uint32_t cmd_seq;
    uint32_t gpu_idx;
    uint32_t cmd_type;
    uint32_t layer_idx;
    void*    cuda_event;
    // perf_trace x-ray: timing-event pair bracketing the finalize MoE GEMM chain on
    // the kExpertFfn stream (start recorded before dispatch, end after). When the
    // completion is reaped, event_elapsed_ms(start,end) → kComputeGpu (GPU compute
    // time). Both null unless perf_trace tracing is on. (FETCH_AND_RUN_MOE only.)
    void*    compute_t_start = nullptr;
    void*    compute_t_end   = nullptr;

    struct CheckpointData {
        uint8_t  checkpoint_type = 0;
        uint32_t host_buf_offset = 0;
        uint32_t data_bytes = 0;
    };
    std::optional<CheckpointData> checkpoint_data;

    // KD-3c: multi-layer pipeline checkpoints (emitted in order before final event).
    struct PipelineCheckpoint {
        void*    cuda_event;
        uint32_t layer_idx;
        uint8_t  checkpoint_type;
        uint32_t host_buf_offset;
        uint32_t data_bytes;
        bool     emitted = false;
    };
    std::vector<PipelineCheckpoint> pipeline_checkpoints;

    uint32_t host_buf_offset = 0;
    uint32_t data_bytes = 0;
    float    top1_prob = 0.0f;
    float    entropy = 0.0f;
    bool     has_confidence = false;
    uint8_t  routed_miss_count = 0;  // TD-89m/89n: top-K experts not resident

    // Union-aware cache partitioning: GPUs whose streaming-zone residents are
    // released when this completion is reaped (transient union fetches,
    // FETCH_AND_RUN_MOE only). 0 = no sweep (every zone=0-only command).
    uint32_t transient_sweep_mask = 0;

    // LS_MOE_BIG_XRAY: clean per-command phase decomposition for the BIG MoE
    // path (perf_trace pairing is broken there — cmd_seq collisions). Filled
    // at finalize by advance_progressive_moe, logged once at completion reap
    // (which also has the finalize GPU elapsed from compute_t_start/end).
    struct MoeBigXray {
        uint32_t num_seqs = 0;
        int      experts = 0;
        uint32_t waves = 0;        // wave-partial passes enqueued
        uint32_t fetches = 0;      // ensure_resident issues (all waves)
        float issue_ms = 0.0f;     // host time inside issue_moe_wave
        float enq_ms = 0.0f;       // host time inside run_moe_wave_pass
        float wait_ms = 0.0f;      // wall attributed to kWaitingExperts
        float drain_ms = 0.0f;     // wall attributed to kWaveDrain
        float wave_gpu_ms = 0.0f;  // Σ per-wave kExpertFfn event elapsed
        float wall_ms = 0.0f;      // command enter → finalize enqueue
    };
    std::optional<MoeBigXray> moe_big_xray;

    // LS_ATTN_CHUNK_PROF (opt-in diag): per-command dispatch decomposition
    // for chunk-shaped (is_prefill) attention commands. host_* = daemon host
    // wall inside dispatch_attention_internal (pre-executor kv-meta+indexer,
    // DcpExecutor::execute_attention, post-executor residual+gate+export);
    // GPU stream-chain time comes from the generic compute_t_start/end pair
    // (recorded on the primary kAttention stream around the dispatch);
    // completion-detect tail = reap time − dispatch_end. Logged once at reap.
    struct AttnChunkProf {
        float host_pre_ms  = 0.0f;
        float host_exec_ms = 0.0f;
        float host_post_ms = 0.0f;
        std::chrono::steady_clock::time_point dispatch_end{};
    };
    std::optional<AttnChunkProf> attn_prof;
};

// Attention refactor V2 (arch_base.h): per-model attention phase classes.
// Defined in daemon/attention/; friends of CommandDispatcher (they are
// stateless facades over dispatcher state — the P1 phase-class skeleton).
class AttentionArch;
class ArchMla;
class ArchDeepseekV4;

// MoE by-model split (moe/arch_base.h): per-model MoE arch participants.
// Defined in daemon/moe/; friends of CommandDispatcher (stateless facades
// over dispatcher state — INV-MOE-ARCH).
class MoeArch;
class ArchMlaMoe;
class ArchDeepseekV4Moe;

class CommandDispatcher {
public:
    struct Deps {
        ipc::CompletionRing*             cmp_ring           = nullptr;
        transfer::TransferEngine*        transfer_engine    = nullptr;
        memory::ExpertCache*             expert_cache       = nullptr;
        // TODO:DEBT TD-37: stream_manager/graph_registry nullable but dereferenced without null check
        compute::StreamManager*          stream_manager     = nullptr;
        compute::GraphRegistry*          graph_registry     = nullptr;
        statistics::CoactivationGraph*   coactivation_graph = nullptr;
        statistics::ExpertStats*         expert_stats       = nullptr;  // nullable
        memory::NvmeTier*                nvme_tier          = nullptr;  // nullable
        memory::NumaManager*             numa_manager       = nullptr;
        memory::PinnedExpertArena*       pinned_arena       = nullptr;  // nullable (P-24)
        model::LoadedModel*              loaded_model       = nullptr;  // nullable
        model::PrepackedSource*          prepacked_source   = nullptr;  // nullable (WP-3)
        model::PackedBufferCache*        packed_cache       = nullptr;  // nullable (WP-4)
        model::ExpertShape               expert_shape{};              // For lazy NVFP4 packing
        parallelism::DcpExecutor*        dcp_executor       = nullptr;  // nullable
        parallelism::DcpCommunicator*    dcp_communicator   = nullptr;  // nullable
        BufferRegistry*                  buffer_registry    = nullptr;
        memory::PageAllocator*           page_allocator     = nullptr;  // nullable until IPC-8a
        uint8_t*                         sideband_base      = nullptr;  // IPC-8c: sideband region base
        ExpertLifecycleManager*          elm                = nullptr;  // ELM-3: nullable
        config::Config*                  live_config        = nullptr;  // 9.8-1b: nullable
        // GG-5b: per-projection GGUF k-quant type for the fused MoE GEMM path
        // (dense FFN, routed experts, shared experts). nullptr for non-GGUF
        // weights. Points at the Engine's owned/registry GgufQuantInterface
        // (same instance the ExpertCache + atomic dispatch path use). Lifetime is
        // the Engine's; the dispatcher only reads projection_type(gate/up/down).
        const model::GgufQuantInterface* gguf_quant         = nullptr;  // GG-5b: nullable
        // GG-9: per-layer routed-expert GGUF k-quant types. A mixed "XL" GGUF uses
        // DIFFERENT routed k-quants per layer (e.g. gate Q4_K/Q5_K/Q6_K across
        // layers), but `gguf_quant` carries only ONE type per projection — decoding
        // a layer with the wrong type = garbage. Indexed by layer_idx; empty →
        // fall back to the uniform `gguf_quant` types (uniform GGUF / non-GGUF).
        std::vector<model::GgufModelExpertTypes> routed_layer_gguf_types;
        const gpu_loader::LoaderConstants* loader_constants = nullptr;  // I8: calibrated cost constants (nullable)
        uint32_t                         max_inflight_compute = 32;    // KD-1: backpressure cap

        // KD-2: device pointers for compute dispatch (indexed by gpu position)
        std::vector<compute::AttentionDevice*> attention_devices;  // nullptr for non-TP GPUs
        std::vector<compute::ExpertDevice*>    expert_devices;
        std::vector<compute::DeviceBackend*>   device_backends;    // #86b: one per GPU position
        bool cuda_kernels_enabled = true;  // false for null-backend tests

        // KD-2: pinned weight device pointers (populated after weight upload)
        std::vector<const void*>   embedding_table_ptrs;    // [vocab_size, hidden_size] per TP GPU
        std::vector<const void*>   output_head_weight_ptrs; // [vocab_size, hidden_size] per TP GPU
        std::vector<const float*>  output_head_bias_ptrs;   // [vocab_size] per TP GPU, or empty
        std::vector<std::vector<const float*>>  gating_bias_ptrs;  // [num_layers][num_gpus] e_score_correction_bias, position-indexed
        // V4-4: hash-layer routing. Layers < moe_hash_layers route by token
        // id via the tid2eid table ([num_experts_per_tok, vocab] I32);
        // exp_probs_b bias never applies on those layers. moe_hash_layers is
        // 0 for every non-V4 architecture (hash branch fully inert).
        std::vector<std::vector<const int32_t*>> hash_gating_table_ptrs;  // [num_layers][num_gpus]
        int moe_hash_layers = 0;

        // KD-3a: KV cache layout for fused attention (resolves TD-14c)
        int64_t kv_cache_stride_block = 0;
        int     kv_cache_stride_row = 0;
        int     kv_page_size = 64;

        // KD-R2: per-TP-GPU paired hidden state buffers (one per DCP rank).
        // Each pair bundles the attention buffer (rank-indexed) and MoE buffer
        // (gpu-position-indexed) with their mapping and sync events.
        std::vector<HiddenStatePair> hidden_state_pairs;

        // Per-GPU MoE hidden state for non-TP GPUs (standalone MoE dispatch).
        // Indexed by GPU position. nullptr for TP GPUs (use pairs instead).
        std::vector<void*> fused_moe_hidden_states;

        // KD-3d: per-layer attention weights [num_layers][dcp_size].
        // Each inner vector holds one AttentionLayerWeights per DCP rank.
        std::vector<std::vector<parallelism::AttentionLayerWeights>> per_layer_attn_weights;
        int max_batch_size = 64;  // DcpExecutor buffer cap for batch_size validation

        // TD-PREFILL-SUPERCHUNK: requested superchunk token capacity
        // (compute.prefill_superchunk_tokens; 0 = off). The dispatcher runs a
        // VRAM fail-safe at construction and may step the EFFECTIVE MoE batch
        // capacity down (see moe_batch_capacity()).
        int superchunk_tokens = 0;

        // TD-PREFILL-MOE-BIG: per-GPU-position prefill-scratch tail spans —
        // the RESERVED tail of the VramAllocator kv_main region
        // (prefill_scratch_preallocated_bytes, ~512 MB on TP GPUs; {nullptr,0}
        // elsewhere). Verified idle in production (the attention devices
        // self-alloc their prefill scratch; the only other consumer is the
        // test-only ExpertCache spill mode, which uses the ADJACENT streaming
        // spill zone, not this tail). Under prefill_moe_big the dispatcher
        // HOMES the chunk-bounded TRANSIENT MoE buffers here (bump-allocated,
        // 256-aligned; per-buffer device_alloc fallback when the tail is
        // absent/full) so the chunked path adds ZERO new VRAM pressure on the
        // post-region free space. Empty/null → device_alloc everything
        // (null-backend tests, prefill_moe_big off — byte-identical legacy).
        std::vector<std::pair<void*, size_t>> prefill_scratch_tails;

        // KD-3e: per-layer router projection weight pointers [layer_idx][gpu_idx].
        // Points into pinned region. nullptr until Engine populates.
        std::vector<std::vector<const void*>> router_weight_ptrs;

        // KD-3e: per-layer shared expert weight pointers [layer_idx][gpu_idx].
        // TODO:DEBT TD-53s: SharedExpertWeights stores single gate_up/down pair (n_shared_experts > 1 breaks)
        struct SharedExpertWeights {
            const void* gate_up = nullptr;  // [2*intermediate, hidden] BF16/quantized
            const void* down    = nullptr;  // [hidden, intermediate] BF16/quantized
            const void* gate_up_scales = nullptr;  // NVFP4/FP8 scales, or nullptr
            const void* down_scales    = nullptr;  // NVFP4/FP8 scales, or nullptr
            // GG-5c: this shared expert's OWN per-projection GGUF k-quant types
            // (the GGUF `ffn_*_shexp` tensors can be a different family than the
            // routed `ffn_*_exps`). Used by the dense/shared GGUF GEMMs instead of
            // the routed `Deps::gguf_quant` types. gate==up → fused [2*I,H] GEMM;
            // gate!=up → split into two single-type [I,H] GEMMs. Sentinel default
            // (Q4_K) for non-GGUF weights, where these are never read.
            model::GgufKQuantType gate_gguf_type = model::GgufKQuantType::Q4_K;
            model::GgufKQuantType up_gguf_type   = model::GgufKQuantType::Q4_K;
            model::GgufKQuantType down_gguf_type = model::GgufKQuantType::Q4_K;
            // V4-7b (ticket H): true only when the source tensors really are
            // packed k-quants. A GGUF checkpoint can carry RAW BF16 FFN
            // tensors (DeepSeek-V4 shared experts are BF16-native) — those
            // must run the plain BF16 GEMM route, never the packed-block
            // decoder (which read BF16 bytes as Q4_K → 1e6-scale garbage).
            bool gate_is_gguf = false;
            bool up_is_gguf   = false;
            bool down_is_gguf = false;
            float weight_scale_2 = 1.f;     // NVFP4 global scale (gate+up)
            float weight_scale_2_down = 1.f; // NVFP4 global scale (down)
            // FP4-ACT-SCALE: calibrated activation input_scales (gate+up merged
            // via max — one quantized activation feeds both halves) and the
            // precomputed GEMM alphas (= ws2 * is; stable storage for async H2D).
            float input_scale = 1.f;
            float input_scale_down = 1.f;
            float alpha = 1.f;
            float alpha_down = 1.f;
        };
        std::vector<std::vector<SharedExpertWeights>> shared_expert_weight_ptrs;

        // KD-3d-plan-dense: per-layer dense FFN weight pointers [layer_idx][gpu_idx].
        struct DenseFFNWeights {
            const void* gate_up = nullptr;
            const void* down    = nullptr;
            const void* gate_up_scales = nullptr;
            const void* down_scales    = nullptr;
            // GG-5c: this dense FFN's OWN per-projection GGUF k-quant types
            // (the GGUF `ffn_*` tensors on dense layers can be a different family
            // than the routed `ffn_*_exps`). See SharedExpertWeights above.
            model::GgufKQuantType gate_gguf_type = model::GgufKQuantType::Q4_K;
            model::GgufKQuantType up_gguf_type   = model::GgufKQuantType::Q4_K;
            model::GgufKQuantType down_gguf_type = model::GgufKQuantType::Q4_K;
            // V4-7b (ticket H): true only when the source tensors really are
            // packed k-quants. A GGUF checkpoint can carry RAW BF16 FFN
            // tensors (DeepSeek-V4 shared experts are BF16-native) — those
            // must run the plain BF16 GEMM route, never the packed-block
            // decoder (which read BF16 bytes as Q4_K → 1e6-scale garbage).
            bool gate_is_gguf = false;
            bool up_is_gguf   = false;
            bool down_is_gguf = false;
            float weight_scale_2 = 1.f;     // NVFP4 global scale (gate+up)
            float weight_scale_2_down = 1.f; // NVFP4 global scale (down)
            // FP4-ACT-SCALE: see SharedExpertWeights.
            float input_scale = 1.f;
            float input_scale_down = 1.f;
            float alpha = 1.f;
            float alpha_down = 1.f;
        };
        std::vector<std::vector<DenseFFNWeights>> dense_ffn_weight_ptrs;

        // KD-4a: final norm weight pointers [num_gpus], position-indexed. Managed by Engine.
        std::vector<const void*> final_norm_ptrs;

        // V4-5b mHC: number of residual streams (hc_mult when the model has
        // mHC, else 1). When > 1 the hidden-state pair buffers carry
        // [rows, hc_streams, hidden] and every attention/FFN module is
        // wrapped in hc_pre/hc_post; all branches are inert at 1.
        int hc_streams = 1;
        // V4-5b mHC: output collapse weights [num_gpus], position-indexed
        // (F32; fn logical [hc, hc*hidden], base [hc], scale [1]).
        std::vector<const void*> output_hc_fn_ptrs;
        std::vector<const void*> output_hc_base_ptrs;
        std::vector<const void*> output_hc_scale_ptrs;
        // V4-5b mHC: attention-stage scratch, rank-indexed (engine-allocated).
        // hc_attn_x [rows, hidden] BF16 (post-collapse module input);
        // hc_attn_post [rows, hc] F32; hc_attn_comb [rows, hc*hc] F32.
        std::vector<void*> hc_attn_x;
        std::vector<void*> hc_attn_post;
        std::vector<void*> hc_attn_comb;

        // KD-4f-b: MTP-specific weight pointers [num_mtp][num_gpus], position-indexed.
        // Consumed by the MTP projection (#16: dispatch_mtp_projection) and
        // the MTP shared-head output path (OutputHeadOpts.mtp_head_idx).
        // eh_proj is TP row-sharded ([H/tp, 2H] per rank); norms replicated.
        // embed_tokens / shared_head_weight may be null when the checkpoint
        // dedups them into the main embedding / lm_head (GLM-5.2 GGUF).
        std::vector<std::vector<const void*>> mtp_embed_tokens_ptrs;
        std::vector<std::vector<const void*>> mtp_shared_head_weight_ptrs;
        std::vector<std::vector<const void*>> mtp_shared_head_norm_ptrs;
        std::vector<std::vector<const void*>> mtp_eh_proj_ptrs;
        std::vector<std::vector<const void*>> mtp_enorm_ptrs;
        std::vector<std::vector<const void*>> mtp_hnorm_ptrs;

        // #16: eh_proj quantization metadata [num_mtp][num_gpus], parallel to
        // mtp_eh_proj_ptrs.  GGUF checkpoints upload eh_proj RAW (packed
        // k-quant blocks, e.g. Q8_0 on GLM-5.2) — the projection GEMM routes
        // through the GGUF kernels (DcpExecutor::route_gguf_gemm).  Non-GGUF
        // (is_gguf false) means BF16 rows → plain BF16 GEMM.
        std::vector<std::vector<uint8_t>> mtp_eh_proj_is_gguf;
        std::vector<std::vector<model::GgufKQuantType>> mtp_eh_proj_gguf_type;

        // DSP-3: DSpark DFlash backbone runtime (nullable — armed only when
        // speculation.method=dspark).  Owns the draft weights + scratch on
        // the draft GPU.  When set, dispatch_attention_internal exports the
        // target's aux hidden states (aux_hidden_state_layer_ids) to the
        // draft GPU, and D_CMD_RUN_DSPARK_STEP drives the backbone forward.
        // nullptr keeps every target path byte-identical.
        speculation::DsparkRuntime* dspark = nullptr;
    };

    explicit CommandDispatcher(Deps deps);
    ~CommandDispatcher();

    CommandDispatcher(const CommandDispatcher&) = delete;
    CommandDispatcher& operator=(const CommandDispatcher&) = delete;
    CommandDispatcher(CommandDispatcher&&) = delete;
    CommandDispatcher& operator=(CommandDispatcher&&) = delete;

    /// Dispatch a single command.  Called by DaemonLoop for every command
    /// that is not CMD_NOOP or CMD_SHUTDOWN.
    void dispatch(const ipc::Command& cmd);

    /// TD-SERVE-NAMED-TOOL-CHOICE: pinned host buffer receiving the
    /// guided-decoding full-logits readback (CMD_OUTPUT_HEAD with
    /// readback_logits=1).  nullptr when unavailable (no CUDA / vocab 0).
    void* logits_readback_host() const { return logits_readback_host_; }
    size_t logits_readback_bytes() const { return logits_readback_bytes_; }

    /// Resolve a transfer token to the originating cmd_seq.
    /// Returns 0 if the token is not tracked (e.g. dedup or already cleaned up).
    uint32_t resolve_cmd_seq(uint64_t transfer_token) const;

    /// Remove a token→cmd_seq mapping after the completion has been written.
    void remove_token_mapping(uint64_t transfer_token);

    /// Resolve an NVMe IoToken to the originating cmd_seq.
    /// Returns 0 if the token is not tracked.
    uint32_t resolve_nvme_cmd_seq(uint64_t nvme_token) const;

    /// Remove an NVMe token mapping after the completion has been written.
    void remove_nvme_token_mapping(uint64_t nvme_token);

    /// Poll pending compute events.  For each completed event, writes
    /// CMP_COMPUTE_DONE (and optionally CMP_CHECKPOINT) to the completion ring.
    /// Returns number of completions written.  Called once per daemon cycle.
    uint32_t poll_compute_completions();

    /// Current number of inflight compute commands across all GPUs.
    uint32_t pending_compute_count() const;

    /// Current pending compute count for a specific GPU.
    uint32_t pending_compute_count(uint32_t gpu_idx) const;

    /// #90: advance the progressive MoE state machine. Called each daemon cycle.
    /// Returns true if work was done (expert arrived or finalized).
    bool advance_progressive_moe();

    /// Queued-command backpressure (FAR/FETCH bursts): the completion pump
    /// the blocking drain runs alongside advance_progressive_moe() so expert
    /// arrivals keep materializing (DaemonLoop::pump_completions — arrivals
    /// are produced NOWHERE else). Injected post-construction by the engine;
    /// without it the drain refuses (commands keep the legacy reject).
    void set_lifecycle_pump(std::function<void()> pump) {
        lifecycle_pump_ = std::move(pump);
    }

    /// TD-GLM-INDEXER-COV/-PREFILL: per-sequence indexer-K coverage
    /// introspection (tests + diagnostics). Returns {mode, next_pos} with
    /// mode 0=unset, 1=paged, 2=arena, 3=dead; {-1, 0} for an untracked
    /// sequence.
    std::pair<int, uint32_t> indexer_coverage(uint64_t seq_id) const;

    /// TD-PREFILL-SUPERCHUNK: effective MoE token-batch capacity — the max
    /// num_seqs accepted by RUN_MOE / FETCH_AND_RUN_MOE[_BIG] and the row bound
    /// for RUN_ATTENTION row_offset + num_seqs. Equals
    /// max(kMaxBatchDescriptors, Deps::superchunk_tokens) after the
    /// construction-time VRAM fail-safe (never OOM). Under prefill_moe_big the
    /// fail-safe is ELASTIC: transients are chunk-bounded and the capacity is
    /// derived from the per-token PERSISTENT cost vs free VRAM
    /// (TD-PREFILL-MOE-BIG).
    int moe_batch_capacity() const { return moe_batch_capacity_; }

    /// TD-PREFILL-MOE-BIG: single-shot token bound. MoE dispatches with
    /// num_tokens <= this run the legacy byte-identical single-shot pipeline;
    /// larger batches run the chunked grouped-GEMM path (transient scratch is
    /// sized for this many tokens and reused per chunk). Equals
    /// moe_batch_capacity() when prefill_moe_big is off (chunking never
    /// engages).
    int moe_chunk_capacity() const { return moe_chunk_capacity_; }

    /// KVS-2 (tests + diagnostics): read-only view of one rank's HOST KV
    /// metadata staging as built by the last build_kv_metadata call. Layer l,
    /// batch row b: block_tables[(l*B + b)*max_blocks + j], slot_mappings
    /// [l*B + b]. global_seqlens is non-null only under sharded KV mode.
    /// Returns all-null view for an out-of-range rank.
    struct KvMetaHostView {
        const int* seqlens_k      = nullptr;  ///< [B] (LOCAL lengths if sharded)
        const int* block_tables   = nullptr;  ///< [kv_layers * B * max_blocks]
        const int* slot_mappings  = nullptr;  ///< [kv_layers * B]
        const int* global_seqlens = nullptr;  ///< [B] global, sharded only
        int max_blocks_per_seq = 0;
        int kv_layers = 0;
        int trash_slot_base = -1;             ///< rank's trash page base, sharded only
    };
    KvMetaHostView kv_meta_host_view(int rank) const;

    /// GG-S1 Phase 4 (TD-GG5C-SPLIT-BUFFER-ALWAYS-ALLOC): load-time gate≠up
    /// scan. Returns true iff ANY dense-FFN or shared-expert unit has its gate
    /// projection's GGUF k-quant type different from its up projection's type —
    /// the ONLY condition under which the dense/shared GGUF gate_up SPLIT path
    /// (launch_gguf_dense_gate_up, gate_type != up_type) fires and the
    /// MoeScratch::gguf_gate_up_split buffer is read. When false (the common
    /// case: every dense/shared unit has gate==up, e.g. GLM-4.7-Flash), the
    /// split buffer is dead VRAM and is gated off at sizing time.
    ///
    /// CUDA-free pure logic over the already-populated Deps weight vectors
    /// (filled by engine.cpp upload_ffn BEFORE this dispatcher is constructed),
    /// so it is safe to call at MoeScratch sizing time. Non-GGUF models leave
    /// the sentinel Q4_K==Q4_K defaults ⇒ returns false (and the alloc is in any
    /// case gated by is_gguf_weight_quant). Exposed static for unit testing.
    static bool any_dense_shared_gate_ne_up(
        const std::vector<std::vector<Deps::DenseFFNWeights>>&
            dense_ffn_weight_ptrs,
        const std::vector<std::vector<Deps::SharedExpertWeights>>&
            shared_expert_weight_ptrs);

private:
    // ── Per-command handlers ────────────────────────────────────────────

    void handle_transfer_h2d(const ipc::Command& cmd);
    void handle_transfer_d2h(const ipc::Command& cmd);
    void handle_cache_reserve(const ipc::Command& cmd);
    void handle_cache_evict(const ipc::Command& cmd);
    void handle_cache_promote(const ipc::Command& cmd);
    void handle_cache_demote(const ipc::Command& cmd);
    void handle_graph_replay(const ipc::Command& cmd);
    void handle_record_event(const ipc::Command& cmd);
    void handle_stream_wait_event(const ipc::Command& cmd);
    void handle_compute_affinity_hints(const ipc::Command& cmd);
    void handle_numa_migrate(const ipc::Command& cmd);
    void handle_compute_command(const ipc::Command& cmd);
    void handle_seq_create(const ipc::Command& cmd);
    void handle_seq_free(const ipc::Command& cmd);
    void handle_seq_fork(const ipc::Command& cmd);
    void handle_nvme_read(const ipc::Command& cmd);
    void handle_nvme_write(const ipc::Command& cmd);
    void handle_nvme_evict_host(const ipc::Command& cmd);
    void handle_cancel_transfer(const ipc::Command& cmd);
    void handle_fused_compute_command(const ipc::Command& cmd);
    bool dispatch_fused_attention(const ipc::Command& cmd);
    bool dispatch_fused_moe(const ipc::Command& cmd);

    // KD-3c: Internal dispatch param structs for multi-layer pipelines
    struct InternalAttentionParams {
        uint32_t layer_idx;
        uint32_t num_seqs;
        uint32_t gpu_idx;
        uint8_t  is_prefill = 0;
        uint8_t  use_graph = 0;
        uint8_t  is_draft = 0;
        uint32_t chunk_start = 0;
        uint32_t chunk_len = 0;
        // F-1: when true, run router projection + topk gating at the end of
        // attention (post-attn norm of the residual-added hidden) and write
        // topk_weights/topk_indices into moe_scratch_[gpu]. Default off =
        // unchanged. Reuses launch_router_projection + launch_topk_gating.
        bool     emit_gating = false;
        // F-3: when true (and emit_gating ran), publish the routed top-K from
        // moe_scratch_ into the F-4 routing-export sideband slot at attention-end,
        // so the orchestrator gets routing BEFORE the MoE op (prefetch seam).
        bool     store_gating = false;
        // TD-PREFILL-SUPERCHUNK: hidden-state ROW offset of this sub-chunk —
        // attention reads/writes attn_buf/moe_buf rows
        // [row_offset, row_offset+num_seqs) and the fused gate stores topk at
        // the same row offset in moe_scratch_. 0 = legacy.
        uint32_t row_offset = 0;
        // TD-PREFILL-SUPERCHUNK: true on every sub-launch of a superchunk —
        // relaxes the indexer coverage `repeat` guard to the superchunk window
        // (a later layer replays sub-chunks BEHIND the advanced frontier).
        bool     superchunk = false;
    };

    // KD-4g: phase control for TP>1 MoE dispatch.
    // kFull: TP=1 path — full pipeline (current behavior).
    // kPreAllreduce: TP>1 — run up to and including down GEMM, then return.
    // kPostAllreduce: TP>1 — run only residual add + commit.
    enum class MoeDispatchPhase : uint8_t {
        kFull = 0,
        kPreAllreduce = 1,
        kPostAllreduce = 2,
    };

    // TD-PREFILL-FETCH-SEAM-SCALING: rolling-wave routed-MoE execution for
    // FETCH_AND_RUN commands whose routed union EXCEEDS the free stable-zone
    // capacity (B>1 prefill chunks). The union is fetched in capacity-bounded
    // WAVES; each wave runs Steps 2..5 (permute → down GEMM) over ONLY the
    // wave's resident experts and ACCUMULATES the per-row expert outputs into
    // moe_wave_accum (each permuted row belongs to exactly one expert, so it is
    // written by exactly one wave and the accumulation is bit-exact: x + 0 = x).
    // The FINAL pass adds its own rows, then runs the unchanged Step 6+ chain
    // (unpermute → shared expert → allreduce → residual → commit) ONCE from the
    // accumulator — bit-identical to a hypothetical all-resident single pass.
    //   kNone:    legacy single-pass (in-capacity unions, decode, RUN_MOE).
    //   kPartial: Steps 2..5 + accumulate, then return (no unpermute/shared/
    //             residual/commit/allreduce).
    //   kFinal:   Steps 2..5 + accumulate, then Step 6+ reads the accumulator.
    enum class MoeWavePass : uint8_t {
        kNone = 0,
        kPartial = 1,
        kFinal = 2,
    };

    struct InternalMoeParams {
        uint32_t layer_idx;
        uint32_t num_seqs;
        uint32_t gpu_idx;
        int      topk_override = 0;  // 0 = use model default
        bool     store_gating = false;
        uint8_t  moe_mode = 0;       // 0=normal, 1=truncation, 2=substitution
        MoeDispatchPhase phase = MoeDispatchPhase::kFull;
        bool     bitset_precomputed = false;  // TD-89p: TP intersection already in moe_scratch_
        // §12h: skip the TD-89m miss-count D2H probe (a per-rank per-layer
        // event spin-sync on the kExpertFfn stream — measured ~75 µs on rank0,
        // handoff §12h). Set by the FETCH_AND_RUN finalize, whose completion
        // reports miss counts from ProgressiveMoeState instead; the probe's
        // last_moe_miss_count_ feeds only RUN_MOE/fused-forward completions.
        bool     skip_miss_probe = false;
        // F-2: when true, Step 0 (router projection) and Step 1 (TopK gating) are
        // skipped — topk_weights/topk_indices are assumed already present in
        // moe_scratch_[gpu] (e.g. written by RUN_ATTENTION's emit_gating in F-1,
        // or any prior gating pass). Kills the double-gate in the prefetch flow.
        // Default false preserves the self-gating path (standalone D_B_CMD_RUN_MOE).
        bool     use_precomputed_gating = false;
        // F-7: emit a seam-routing checkpoint (top-K weights+indices D2H to
        // kSeamCheckpointOff) right after Step 1 gating, before the expert GEMM.
        bool     emit_seam_checkpoint = false;
        // DET-REDUCE Phase 1b: selects how Step 6 unpermute writes the routed
        // result for the cross-GPU EP combine (mirrors MoeCombineMode; kept as a
        // POD uint8_t so this header stays free of expert_device.h):
        //   0 = kReducedBf16 (legacy bf16 K-way sum → moe_output);
        //   1 = kPerSlotFp32 (per-slot fp32 → moe_output_fp32);
        //   2 = kPerSlotBf16 (per-slot bf16 → moe_output_bf16_perslot).
        // Set non-zero only for the EP-within-TP routed path when
        // deterministic_ep_combine is on.
        uint8_t  ep_combine_mode = 0;
        // TD-PREFILL-FETCH-SEAM-SCALING: rolling-wave pass selector (see
        // MoeWavePass). kNone keeps the legacy single-pass path byte-identical.
        MoeWavePass wave_pass = MoeWavePass::kNone;
        // TD-PREFILL-MOE-BIG: grouped-GEMM chunk-size override for the chunked
        // path (tokens; clamped to moe_chunk_capacity_). 0 = engine default.
        // Ignored by the single-shot path (num_tokens <= moe_chunk_capacity_).
        int chunk_tokens = 0;
        // C-6 early-kick (LS_CPU_EXPERT_EARLY_KICK): CPU-INPUT PRIME pass. Run
        // ONLY the pre-MoE RMSNorm (producing normalized_hidden) + record the
        // input-ready event, then return before the expert pipeline. The routed
        // top-K already sits in moe_scratch_ (attention emit_gating), so the host
        // CPU-expert FFN has all its inputs. Lets the caller kick the host FFN at
        // fetch-launch time so it overlaps the missing-expert H2D window. Touches
        // no accumulator / moe_output / bitset — pure input prime.
        bool prime_cpu_input_only = false;
    };

    // KD-R1: shared per-layer forward pass opts for speculation pipelines
    struct ForwardLayerOpts {
        uint32_t layer_idx;
        uint32_t num_seqs;
        uint32_t gpu_idx;
        bool is_prefill = false;
        bool use_graph = false;
        bool is_draft = false;
        int topk_override = 0;      // 0 = use model default
        bool store_gating = false;
    };

    // KD-R5: unified output head dispatch (final_norm → GEMM → confidence → sampling)
    struct OutputHeadOpts {
        uint32_t gpu_idx;
        int      num_tokens = 1;
        const void* input = nullptr;       // device ptr: hidden states (pre-norm)
        float* logits_out = nullptr;       // device ptr: [num_tokens, vocab_size]
        void* stream = nullptr;            // CUDA stream handle
        compute::StreamId stream_id;  // caller must set (no default — avoids header dep on enum values)

        bool apply_final_norm = true;
        bool compute_confidence = false;
        bool do_sample = false;

        // Sampling params (do_sample=true only)
        float temperature = 0.0f;
        float top_p = 1.0f;
        int   top_k = 1;
        uint64_t seed = 0;

        // Speculation readback destination (nullptr = CMD_OUTPUT_HEAD path).
        // When non-null, D2H copies go here, with
        // n = min(num_tokens, ipc::kMaxOutputHeadReadbackTokens):
        //   sampled token ids → readback_host_dst[0..4n)      (if do_sample)
        //   top1_prob (last)  → readback_host_dst[4n..4n+4)   (if compute_confidence)
        // num_tokens == 1 reproduces the historical single-token layout
        // (token at [0..3], top1_prob at [4..7]) byte-for-byte — all fused
        // speculation pipelines pass num_tokens=1 and are unaffected.
        uint8_t* readback_host_dst = nullptr;

        // #16: MTP shared-head selection.  >= 0 selects MTP module
        // mtp_head_idx: the pre-head RMSNorm uses mtp_shared_head_norm_ptrs
        // (falls back to final_norm when absent) and the head GEMM uses
        // mtp_shared_head_weight_ptrs when present (GGUF checkpoints dedup
        // MTP shared_head.head into the main lm_head → fallback is exact).
        int mtp_head_idx = -1;

        // TD-SERVE-NAMED-TOOL-CHOICE / TD-ORCH-SAMPLED-SPEC: when
        // non-null, the FIRST min(num_tokens, ipc::kMaxLogitsReadbackRows)
        // rows' FULL logits ([n, vocab_size] f32, contiguous) are
        // D2H-copied here on opts.stream after the head GEMM (and, for TP,
        // after the allgather/transpose) — host-visible when the
        // completion event fires.  num_tokens == 1 keeps the historical
        // guided-decoding single-row layout byte-identical.  Points at
        // logits_readback_host_ for the ring path.
        uint8_t* logits_host_dst = nullptr;
    };

    bool dispatch_attention_internal(const InternalAttentionParams& p);

    // Attention refactor V2 P1: the per-model phase classes (arch_base.h)
    // carry the model-special driver phases (shape gate / step staging /
    // execution). They are friends — stateless facades over this
    // dispatcher's state, constructed lazily on the first attention
    // dispatch and selected per call by the model architecture.
    friend class AttentionArch;
    friend class ArchMla;
    friend class ArchDeepseekV4;
    std::unique_ptr<AttentionArch> arch_mla_;
    std::unique_ptr<AttentionArch> arch_v4_;

    // MoE by-model split (moe/arch_base.h): same pattern for the MoE driver
    // (INV-MOE-ARCH) — constructed lazily on the first MoE dispatch and
    // selected per call by the model architecture; hook bodies keep their
    // original data-gated conditions verbatim.
    friend class MoeArch;
    friend class ArchMlaMoe;
    friend class ArchDeepseekV4Moe;
    std::unique_ptr<MoeArch> moe_arch_mla_;
    std::unique_ptr<MoeArch> moe_arch_v4_;

    /// TD-PREFILL-NONDET diagnostic (LS_SEAM_DUMP=<path>, off by default):
    /// D2H + append one binary record of a hidden-state buffer at a named
    /// pipeline stage — used to localize the exact first diverging stage of
    /// the cross-chunk prefill fork by diffing two runs. Layers limited by
    /// LS_SEAM_DUMP_MAXLAYER (default 4). Full device sync when enabled
    /// (diagnosis-only, like LS_DRIFT_DUMP). Record (LE): int32
    /// hdr[5]={tag4cc,layer,gpu,rows,hidden}; bf16 payload rows*hidden.
    void seam_dump_hidden(uint32_t tag4cc, int layer, int gpu,
                          const void* dev_buf, int rows, int hidden);

    /// DSP-3 aux-hidden export hook (INV-DSPARK-AUX): called at the start of
    /// dispatch_attention_internal (after the moe_attn_event waits) when the
    /// dspark runtime is armed and p.layer_idx is one of the draft's
    /// aux_hidden_state_layer_ids — the primary rank's attn_buf then holds
    /// the post-residual output of layer_idx-1 == the input of layer_idx
    /// (the vLLM aux capture convention the checkpoint was trained against).
    /// Enqueues the cross-GPU copy to the draft GPU on the primary rank's
    /// kAttention stream; NEVER mutates target state (read-only source,
    /// fail-closed inside the runtime on unsupported step shapes).
    void maybe_dspark_capture(const InternalAttentionParams& p);

    /// Ticket J: the V4 dflash draft's FINAL-residual aux tap (aux id ==
    /// model.num_hidden_layers = "input of the head"). Called at the start
    /// of dispatch_output_head (before the output_hc collapse) when the
    /// runtime wants the stream-MEAN representation: reduces the hc-wide
    /// head input rows to [rows, hidden] on the target GPU
    /// (launch_hc_stream_mean into dspark_mean_scratch_) and captures them
    /// as the LAST aux slot, chaining the draft-context ingest.
    void maybe_dspark_capture_final(uint32_t gpu, const void* head_input,
                                    int num_tokens, int mtp_head_idx,
                                    void* stream);

    /// TD-V4-SPEC-PREFILL-CTX: chunk-final aux capture. Called at the
    /// terminal finalize of E_CMD_FETCH_AND_RUN_MOE[_BIG] — when the
    /// completed layer is the LAST layer and the dspark runtime's capture
    /// epoch awaits only the final slot (pending_final_window), the pair
    /// attn_buf holds the committed hc-wide final residual for the whole
    /// window (row r of the buffer = absolute position start+r — the
    /// superchunk row-offset layout). Fires the final tap from there in
    /// kDsparkMeanScratchRows pieces on the primary rank's kAttention
    /// stream (ordered after the MoE commit via moe_attn_event), so
    /// HEADLESS prefill chunks arm the draft context exactly like a
    /// decode-step head would. Records dspark_moe_final_mark_ so a head
    /// that DOES follow (verify chunks, decode steps) skips its duplicate
    /// head-time capture.
    void maybe_dspark_capture_moe_final(uint32_t gpu, uint32_t layer_idx,
                                        int num_tokens);

    /// Window covered by the last MoE-final aux capture (head-time dedupe).
    struct DsparkMoeFinalMark {
        uint64_t seq_id = 0;
        uint32_t start = 0;
        uint32_t end = 0;
        bool valid = false;
    };
    DsparkMoeFinalMark dspark_moe_final_mark_;

    /// Per-GPU stream-mean staging for the V4 dflash aux capture
    /// ([kDsparkMeanScratchRows, hidden] BF16, lazily device_alloc'd).
    /// TD-V4-CHUNK-PREFILL: must cover a whole prefill chunk (512 =
    /// kMaxBatchDescriptors) — a smaller cap would invalidate the draft
    /// context on every chunked prefill (4 MB at hidden 4096).
    static constexpr int kDsparkMeanScratchRows =
        static_cast<int>(ipc::kMaxBatchDescriptors);
    std::vector<void*> dspark_mean_scratch_;

    // ── EPM-1 (Phase 29): routing-label dump ────────────────────────────
    // Label-side tap at the F-1 attention-end gate (the production routing
    // producer for decode under FETCH_AND_RUN): after launch_topk_gating on
    // the PRIMARY rank, D2H the full pre-top-k router logits (FP32 -> FP16
    // on host) + the top-8 ids/weights — the EXACT device buffers the F-3
    // routing export / F-7 seam / FETCH_AND_RUN expert list are built from —
    // and append an EPMR row keyed (seq_id, token_pos). Decode-only
    // (batch 1, non-draft, non-prefill), routed target layers only.
    // Gated on speculation.dspark.epm_dump_dir / LS_EPM_DUMP; the hot path
    // pays ONE int compare when off (epm_dump_state_, resolved lazily).
    // Implemented in dispatch_attention.cpp beside its only call site.
    void epm_capture_routing(const InternalAttentionParams& p, int layer,
                             int batch_size, uint32_t gpu, void* stream);
    int epm_dump_state_ = -1;  ///< -1 unresolved, 0 off, 1 armed
    std::unique_ptr<speculation::EpmRoutingDumper> epm_routing_dump_;
    void* epm_route_host_ = nullptr;   ///< NUMA-local pinned D2H staging
    size_t epm_route_host_bytes_ = 0;
    int epm_route_host_node_ = -1;
    bool epm_route_host_registered_ = false;
    bool epm_route_host_from_numa_ = false;
    int epm_route_host_gpu_ = -1;      ///< staging's source GPU position

    bool dispatch_moe_internal(const InternalMoeParams& p);
    bool dispatch_moe_all_ranks(const InternalMoeParams& mp_template);

    /// INV-MOE-EP-XTP: EP-beyond-TP — dispatch the routed expert subsets that
    /// live on EXPERT-ONLY (non-DCP) GPUs and fold their partial outputs into
    /// a TP rank's routed buffer BEFORE the cross-rank EP combine. Sequence
    /// per extra rank g: (1) D2D-broadcast rank0's normalized hidden + routed
    /// top-K (weights+indices) onto g (stream-ordered after rank0's Phase-1
    /// enqueue), (2) dispatch_moe_internal on g (kPreAllreduce, precomputed
    /// bitset+gating — routed pipeline only: no shared expert / residual /
    /// commit on expert-only ranks), (3) D2D-stage g's routed partial
    /// (mode-0 [B,H] bf16 or per-slot [B,K,H]) onto tp_rank(i % dcp_size)'s
    /// ep_xtp_staging and add it into that rank's same-mode routed buffer
    /// (event-chained per destination rank). Disjoint expert ownership
    /// (dedup_ep_residency spans TP + extra ranks) makes the per-slot fold
    /// bit-exact (x + 0 = x). Called between Phase 1 and Phase 2 of
    /// dispatch_moe_all_ranks; no-op when `extras` is empty. Chunked batches
    /// are unsupported (fail loud — TD-MOE-EP-XTP-WAVES).
    bool dispatch_moe_ep_extras(const InternalMoeParams& mp_template,
                                const std::vector<int>& extras,
                                uint8_t ep_combine_mode);

    /// INV-MOE-EP-XTP broadcast half: enqueue rank0's normalized hidden +
    /// routed top-K (weights/indices) D2D copies to every extra rank, then
    /// make each extra rank's kExpertFfn stream wait on the copies.
    /// after_rank0_dispatch=false (precomputed gating): enqueued BEFORE the
    /// TP Phase-1 loop — waits rank0's attn→moe event, launches the
    /// post-attention rmsnorm into rank0's normalized_hidden itself (rank0's
    /// own dispatch re-runs it idempotently), so the extra ranks' compute
    /// overlaps the TP ranks'. after_rank0_dispatch=true (legacy self-gating):
    /// enqueued after rank0's Phase-1 dispatch produced the top-K + norm.
    bool ep_xtp_broadcast(const InternalMoeParams& mp_template,
                          const std::vector<int>& extras,
                          bool after_rank0_dispatch);

    /// TD-PREFILL-MOE-BIG: chunked MoE pipeline for token batches larger than
    /// moe_chunk_capacity_ (defined in dispatch_moe_big.cpp). Runs the same
    /// step sequence as dispatch_moe_internal but loops the permute → grouped
    /// GEMMs → unpermute (and the dense/shared-expert GEMMs) over
    /// ~moe_big_chunk_tokens chunks, REUSING the chunk-sized transient scratch;
    /// only the [B, H] persistent buffers (moe_output / normalized_hidden /
    /// shared_expert_output / topk) span the whole batch. Wave passes
    /// (kPartial/kFinal) accumulate each wave's chunk-unpermuted partials into
    /// moe_output ([B, H] bf16; single-wave commands are bit-identical to the
    /// single-shot result since 0 + x = x — INV-MOE-BIG-2). Dispatched from
    /// dispatch_moe_internal when num_tokens > moe_chunk_capacity_ and CUDA is
    /// live; never entered otherwise.
    bool dispatch_moe_chunked_internal(const InternalMoeParams& p);

    // F-7: publish routed top-K (weights f32 + indices i32) to the sideband seam
    // sub-region (kSeamCheckpointOff) and record last_seam_checkpoint_. No-op when
    // p.emit_seam_checkpoint is false or scratch routing is unavailable.
    void publish_seam_routing(const InternalMoeParams& p, uint32_t gpu,
                              int expanded_tokens, void* stream);
    // F-4/F-3: publish routed top-K from moe_scratch_[gpu] into the canonical
    // routing-export slot (kRoutingExportOff). Shared by the MoE producer (F-4)
    // and the attention producer (F-3); no-op if sideband/scratch unavailable.
    void publish_routing_export(uint32_t gpu, int num_tokens, int topk,
                                uint32_t layer_idx, void* stream,
                                int src_row_offset = 0);
    bool forward_one_layer(const ForwardLayerOpts& opts);
    bool dispatch_output_head(const OutputHeadOpts& opts);
    bool dispatch_output_head_tp(const OutputHeadOpts& opts);

    /// KD-4e: Build KV cache metadata from sideband batch descriptors + seq_pages_.
    /// TD-GOLDEN-KVMETA-PER-LAYER: builds ALL kv_layers_ layers' block tables and
    /// slot mappings once per (batch, seq, pos) and uploads them in a single H2D
    /// per array per rank; per-layer attention reads layer-offset pointers.
    /// kUnavailable = infrastructure not wired (page allocator/sideband/scratch);
    /// kFailed = unknown seq_id or page-pool exhaustion — the attention command
    /// must fail with CMP_ERROR (TD-GOLDEN-KV-EXHAUST), never write to slot 0.
    enum class KvMetaResult { kOk, kUnavailable, kFailed };
    KvMetaResult build_kv_metadata(int batch_size, int dcp_size);

    /// KD-4e1: Grow seq_pages_ if token_pos exceeds current allocation.
    /// Returns false when the page pool is exhausted (partial logical page
    /// rolled back); true otherwise (including unknown seq_id — caller checks).
    bool ensure_pages(uint64_t seq_id, uint32_t token_pos);

    void run_mtp_pipeline(const ipc::Command& cmd);
    void run_self_spec_pipeline(const ipc::Command& cmd);

    /// #16 / GLM-25g (resolves TD-50m): MTP projection —
    /// eh_proj(concat(enorm(Emb(token_id)), hnorm(prev_hidden))) written to
    /// every TP rank's hidden-state pair attn_buf (the MTP layer's attention
    /// input) and the primary rank's SpecScratch.hidden_a.  prev_hidden is
    /// each rank's CURRENT attn_buf content (trunk hidden after a main-model
    /// step; MTP-layer output after a chained draft step).  eh_proj is TP
    /// row-sharded: each rank GEMMs its output-row shard into a zeroed
    /// full-H buffer and an allreduce-sum reconstructs the full projection
    /// on all ranks (same collective pattern as the sharded embedding).
    /// GGUF checkpoints route the GEMM through DcpExecutor::route_gguf_gemm
    /// (eh_proj kept packed, e.g. Q8_0 on GLM-5.2); BF16 checkpoints use the
    /// plain BF16 GEMM.  On failure writes CMP_ERROR for cmd_seq and returns
    /// false.  stream_out/pair_idx_out mirror setup_spec_pipeline.
    /// hidden_row selects the attn_buf ROW holding prev_hidden (batched
    /// verify leaves the K-token trunk hiddens in rows [0..K); MTP steps
    /// write only row 0, so a sequential catch-up chain can consume rows
    /// in ascending order).  0 = historical single-row behavior.
    bool dispatch_mtp_projection(uint32_t cmd_seq, uint32_t gpu,
                                 uint32_t token_id, int mtp_layer_idx,
                                 int hidden_row,
                                 void*& stream_out, int& pair_idx_out);

    /// TD-GOLDEN-EMB-OOB: embedding TP degree (config override or dcp_size).
    /// > 1 means the embedding table is vocab-sharded across TP ranks.
    int embedding_tp_degree() const;

    /// TD-GOLDEN-EMB-OOB: masked local-shard embedding lookup on EVERY TP
    /// rank + allreduce-sum, leaving the full embedding replicated in all
    /// ranks' attn_bufs (kAttention streams). Token ids must already be in
    /// the sideband token-id slot. Returns false when a prerequisite is
    /// missing (caller reports the error — never fall back to a full-vocab
    /// lookup on a sharded table).
    bool dispatch_embedding_lookup_sharded(int num_tokens,
                                           uint32_t row_offset = 0);

    // Fused command handlers (IPC-8e)
    void handle_prefetch_batch(const ipc::Command& cmd);
    void handle_evict_batch(const ipc::Command& cmd);
    void handle_nvme_batch_read(const ipc::Command& cmd);
    void handle_prefetch_expert(const ipc::Command& cmd);
    void handle_evict_to_host(const ipc::Command& cmd);
    void handle_slow_evict_to_host(const ipc::Command& cmd);
    void handle_stage_expert(const ipc::Command& cmd);
    void handle_run_prefetch_probe(const ipc::Command& cmd);
    void handle_run_adapter_forward(const ipc::Command& cmd);
    void handle_run_mtp_step(const ipc::Command& cmd);
    void handle_mtp_project(const ipc::Command& cmd);
    void handle_run_dspark_step(const ipc::Command& cmd);  // DSP-3
    void handle_run_self_spec_forward(const ipc::Command& cmd);
    void handle_forward_one_layer(const ipc::Command& cmd);
    void handle_e_seq_create(const ipc::Command& cmd);
    void handle_e_seq_free(const ipc::Command& cmd);
    void handle_fetch_and_run_moe(const ipc::Command& cmd);
    /// TD-PREFILL-MOE-BIG: E_CMD_FETCH_AND_RUN_MOE_BIG — same progressive
    /// machine as handle_fetch_and_run_moe (shared body) with (1) the chunked
    /// grouped-GEMM execution for the big batch and (2) DOUBLE-BUFFERED waves:
    /// a wave's issue budget reserves min(free/2, 8) stable slots when the
    /// routed union exceeds capacity, and the next wave's H2D fetches are
    /// issued into the reserve the moment a wave-partial compute pass is
    /// enqueued — the transfer engine streams wave i+1 while the GPU
    /// computes wave i (bounded reserve: extra waves cost full-batch GEMM
    /// sweeps, INV-MOE-BIG-PIPE).
    void handle_fetch_and_run_moe_big(const ipc::Command& cmd);
    /// Shared body of the two FETCH_AND_RUN handlers (`big` selects the
    /// TD-PREFILL-MOE-BIG extensions; the payloads are layout-compatible).
    void handle_fetch_and_run_moe_impl(const ipc::Command& cmd, bool big);
    // E_CMD_REEF_ROUTE — REEF placement/eviction solve over the sideband
    // (dispatch_reef.cpp; CPU-only). E_CMD_FAR_FORWARD_LAYER — fused
    // attention + routed FETCH layer (same TU).
    void handle_reef_route(const ipc::Command& cmd);
    void handle_far_forward_layer(const ipc::Command& cmd);
    // Lazy REEF service construction (first REEF command). Returns null +
    // remembers failure (subsequent commands fail fast) on a bad
    // calibration/config. errmsg receives the failure reason.
    gpu_loader::ReefOrch* ensure_reef_service(std::string& errmsg);
    void handle_config_update(const ipc::Command& cmd);

    // I8 P1 shadow-mode: solve expert→device with LoaderSolver and LOG j[·]
    // beside the orchestrator's gpu_idx — NO behavior change. Env-gated
    // (LS_LOADER_SHADOW); requires loader_constants matching the live device set.
    // When LS_LOADER_SHADOW_DUMP=<path> is also set, appends one structured JSONL
    // record per solve (model input→output) keyed by (cmd_seq, layer_idx) — the
    // join key with the perf_trace CSV for the I8b model-vs-reality x-ray.
    // out_pos (when non-null) is filled [n] with the solver's chosen GPU *position*
    // per expert (-1 if none) — used by P2 (LS_LOADER_ACT) to route by j[·].
    // chunk_width = the MoE row/token count of THIS solve (fetch_and_run_moe.
    // num_seqs). 1 for plain decode / draft steps; 1+γ (=R) for a dsp52 batched
    // verify chunk. Drives the TASK-1 B-aware never-lose CPU-offload post-pass
    // (host FFN amortizes over the chunk; fetch/fold amortize; overlap grows).
    void shadow_solve_and_log(const ipc::ExpertPrefetchEntry* entries,
                              uint32_t n, uint32_t layer_idx, uint32_t cmd_seq,
                              std::vector<int>* out_pos = nullptr,
                              uint32_t chunk_width = 1);

    struct ProgressiveMoeState;  // fwd decl (full definition below) for the param type
    // I8 GPU-loader shadow/act integration (all env-gated, inert in production;
    // defined in dispatch_loader.cpp):
    //   init_loader_from_env  — read the LS_LOADER_* gates once (ctor).
    //   close_loader_dump     — flush+close the JSONL x-ray sink (dtor).
    //   route_moe_by_loader   — P1 shadow solve+log + P2 (LS_LOADER_ACT) reroute.
    //   feed_expert_stats     — recency feed (only consumed by the loader evict_cum).
    void init_loader_from_env();
    void close_loader_dump();
    void route_moe_by_loader(const ipc::ExpertPrefetchEntry* entries,
                             const ipc::Command& cmd, ProgressiveMoeState& state);
    void feed_expert_stats(const ipc::Command& cmd,
                           const ipc::ExpertPrefetchEntry* entries,
                           bool have_weights);

    // LOADER_STATS_LOCALITY: lazily build the per-GPU evict-score board +
    // place_cons table the first time the loader path runs (sized from
    // loader_constants / live_config). No-op once built or when the loader is
    // inert. Keeps the boards out of the production decode path entirely.
    void ensure_loader_stats_boards();
    // TD-EVICT-BOARD-DESYNC: the EvictScoreBoard is registered as the
    // ExpertCache's ResidencyListener (in ensure_loader_stats_boards), so board
    // residency is fed by the cache's own add/evict choke-points — the former
    // manual loader_on_place/loader_on_evict triggers are retired.
#ifndef NDEBUG
    // Debug-only parity assert: board residency == ExpertCache stable residency.
    void assert_board_cache_parity() const;
#endif
    // Advance the daemon's per-token bookkeeping once per decode token (a non-
    // increasing MoE layer index starts a token). Idempotent within a layer;
    // updates stats_token_id_/last_stats_layer_ for the off-hot ExpertStats feed.
    // (The EvictScoreBoard no longer keeps an internal clock; scores are external.)
    // Returns true iff this call started a NEW token (the per-token decay hook).
    bool advance_loader_token_if_new(uint32_t layer_idx);
    // Flatten (layer_idx, expert_idx) → the dense expert id used by the
    // PlaceConsTable / ExpertStats id space. -1 if out of the MoE-layer range.
    int loader_flat_expert_id(uint32_t layer_idx, uint16_t expert_idx) const;

    // ── Host data resolution for transfers ──────────────────────────────

    using HostSourceResult = transfer::HostSourceResult;
    HostSourceResult resolve_host_source(uint32_t layer_idx, uint16_t expert_idx);

    // ── KD-2: device lookup helpers ────────────────────────────────────

    compute::AttentionDevice* attn_dev(uint32_t gpu_idx) const;
    compute::ExpertDevice*    expert_dev(uint32_t gpu_idx) const;

    // ── C-6 CPU expert offload (Milestone A) ────────────────────────────
    // LS_CPU_EXPERT (kill-switch) + LS_CPU_EXPERT_FORCE=<layer>:<e0,e1,..>[;..]
    // force a SMALL set of (layer, expert) onto the host CPU expert device,
    // bypassing the solver for the first correctness boot. The forced experts
    // are EXCLUDED from every GPU's resident bitset (never fetched/computed on
    // a GPU), computed on host from the pinned arena, and their reduced-bf16
    // moe_output contribution is H2D-folded onto every TP rank's moe_output
    // AFTER the EP combine and BEFORE the Phase-3 residual add. Default OFF ⇒
    // the champion path is byte-identical.
    bool cpu_expert_enabled();
    bool is_cpu_forced(uint32_t layer_idx, uint16_t expert_idx);
    bool cpu_layer_has_forced(uint32_t layer_idx);
    compute::ExpertDevice* cpu_expert_device();
    /// Compute every forced CPU expert of `layer_idx` on host and fold their
    /// contribution into each GPU position in `gpu_positions`. Rank0
    /// (gpu_positions[0]) supplies the normalized hidden + routing (D2H).
    ///
    /// Two fold modes (returns false on a hard failure):
    ///   * perslot=false (legacy): a host-reduced bf16 [num_tokens, hidden]
    ///     contribution is added into moe_output AFTER the EP combine. Non-
    ///     bit-exact (double bf16 rounding vs the canonical fp32 per-slot
    ///     reduce) — used only for the mode-0 / TP=1 fallback.
    ///   * perslot=true (canonical, BIT-EXACT): per-slot [num_tokens, topk,
    ///     hidden] rows (fp32 when bf16_payload=false, else bf16) are added into
    ///     each rank's moe_output_fp32 / moe_output_bf16_perslot BEFORE the
    ///     fixed-order K-slot combine reduce — the forced experts join the SAME
    ///     canonical reduce as the GPU experts (their slots are 0 on every GPU,
    ///     0 + c_k = c_k is exact), so ON == all-GPU OFF bit-for-bit under
    ///     LS_CPU_EXPERT_LOSSLESS.
    bool fold_cpu_forced_experts(uint32_t layer_idx, int num_tokens,
                                 const std::vector<int>& gpu_positions,
                                 bool perslot = false,
                                 bool bf16_payload = false);

    /// C-6 Milestone C (true host↔GPU overlap): KICK the host FFN on a separate
    /// worker thread right after the GPU FFN GEMMs are enqueued (Phase 1), so the
    /// host compute (`cpu_forced_produce`) runs CONCURRENTLY with the in-flight
    /// GPU GEMMs + EP allreduce instead of serializing behind them at finalize.
    /// `fold_cpu_forced_experts` then JOINS the worker and does only the (fast)
    /// H2D fold on the daemon stream, ordered before the combine reduce. No-op
    /// unless overlap is ON and the layer has forced experts; the destructor and
    /// each new kick join any pending worker.
    void start_cpu_forced_experts(uint32_t layer_idx, int num_tokens,
                                  const std::vector<int>& gpu_positions,
                                  bool perslot = false,
                                  bool bf16_payload = false);
    /// Produce the host-FFN per-slot/reduced contribution into `cpu_fold_moe_o_`
    /// (D2H input on a side stream + the GGUF FFN chain). Runs on the worker
    /// thread (overlap) or synchronously (serial / non-kicked paths). Stores the
    /// fold metadata (`cpu_fold_active_/contrib_/fp32_add_`) for the H2D pass.
    bool cpu_forced_produce(uint32_t layer_idx, int num_tokens, int rank0,
                            bool perslot, bool bf16_payload);
    /// H2D-fold the produced `cpu_fold_moe_o_` into every rank's combine buffer
    /// (daemon thread, kExpertFfn stream) — the GPU-ordered tail of the fold.
    bool cpu_forced_fold_h2d(uint32_t layer_idx, int num_tokens,
                             const std::vector<int>& gpu_positions,
                             bool perslot, bool bf16_payload);
    /// Join a pending overlap worker (if any). Called before each new kick, at
    /// the fold, and in the destructor so a std::thread is never left joinable.
    void join_cpu_fold_worker();

    /// C-6 Task A (graph-hoist overlap): record the per-gpu input-ready event on
    /// `stream` so the fold's input D2H can ride a side stream keyed to it (host
    /// FFN overlaps the GPU replay). Called after the RMSNorm (protects the norm
    /// even on the zero-resident skip path) and again after the router (covers
    /// norm+router on the normal path). No-op unless forced-CPU decode.
    void record_cpu_input_event(uint32_t gpu, int num_tokens,
                                uint32_t layer_idx, void* stream);

    /// C-6 Milestone C (graph-hoist OVERLAP gate, LS_CPU_EXPERT_OVERLAP).
    /// DEFAULT ON (1): the fold's input D2H rides a side stream keyed to the
    /// input-ready event and the host FFN runs CONCURRENTLY with the captured
    /// GPU FFN replay (B=1) / eager FFN emit (B>1 verify chunk); moe_o is
    /// deferred-freed via per-rank consume events (no post-fold device drain).
    /// =0 forces the SERIAL/EXPOSED path (synchronize_device before the host
    /// FFN + a device drain after the fold) — the A/B "no-overlap" arm.
    /// CHAMPION-SAFE: only ever consulted inside the forced-CPU fold, so with no
    /// CPU device the pure champion is byte-identical regardless of this flag.
    bool cpu_expert_overlap_enabled();

    /// C-6 early-kick (LS_CPU_EXPERT_EARLY_KICK). DEFAULT ON (1): the host
    /// CPU-expert FFN is kicked at FETCH-LAUNCH time (handle_fetch_and_run, right
    /// after route_moe_by_loader resolves the CPU set + issue_moe_wave launches
    /// the missing-expert H2D) via a light rank0 norm PRIME + start_cpu_forced_
    /// experts, so the host compute overlaps the (big) fetch window instead of only
    /// the post-Phase-1 GEMM/allreduce window. =0 falls back to the Milestone-C
    /// late kick inside dispatch_moe_all_ranks (Phase-1 enqueue). CHAMPION-SAFE:
    /// only consulted when the layer has forced CPU experts + overlap is on.
    bool cpu_expert_early_kick_enabled();

    /// Predict the CPU-fold payload format (perslot / bf16) the finalize
    /// (dispatch_moe_all_ranks) will use, so the early-kick produce and the
    /// finalize fold agree on moe_o's layout. A mismatch (rare replicated-EP case)
    /// is reconciled at finalize by discarding the early worker and re-producing.
    void predict_cpu_fold_format(uint32_t layer_idx, int num_tokens,
                                 bool& perslot, bool& bf16) const;

    int  cpu_expert_state_ = -1;          ///< -1 unresolved, 0 off, 1 on
    int  cpu_expert_overlap_state_ = -1;  ///< -1 unresolved, 0 serial, 1 overlap
    int  cpu_expert_early_kick_state_ = -1; ///< -1 unresolved, 0 late, 1 early
    // C-6 early-kick reconciliation: the layer whose host FFN was early-kicked at
    // fetch time (UINT32_MAX = none), plus the fold format the kick predicted. The
    // finalize checks these to skip the redundant late kick and to fall back to a
    // synchronous re-produce when the prediction disagrees with the real format.
    uint32_t cpu_early_kick_layer_ = 0xffffffffu;
    bool cpu_kick_perslot_ = false;
    bool cpu_kick_bf16_ = false;
    bool cpu_forced_parsed_ = false;      ///< LS_CPU_EXPERT_FORCE parsed once
    std::unordered_map<uint32_t, std::vector<uint16_t>> cpu_forced_experts_;

    // ── TASK-2: never-lose I8 CPU-solver bridge (LS_LOADER_CPU_SOLVER=1) ─────
    // Wires append_cpu_expert_device into the LIVE solver K: the CPU becomes an
    // assignable device COLUMN with a calibrated per-expert ComputeCurve (host
    // FFN) and ZERO transfer (reads host RAM). The solver then offloads a routed
    // MISS to CPU ONLY when the CPU FFN cost it adds is CHEAPER than the GPU
    // (PCIe fetch + GPU compute + reuse-victim place) it displaces — else it
    // keeps the expert on GPU. Fallback (no beneficial offload) ⇒ ZERO CPU
    // assignment ⇒ champion == OFF == neutral: the "never lose" guarantee, now
    // EMPIRICALLY VALIDATED not argued. shadow_solve_and_log DYNAMICALLY
    // populates cpu_forced_experts_[layer] from the solver's CPU-column
    // assignments each solve (the fold path — cpu_layer_has_forced /
    // fold_cpu_forced_experts — then executes them); route_moe_by_loader marks
    // state.experts[i].cpu_forced. Requires LS_CPU_EXPERT=1 (fold machinery) and
    // LS_CPU_EXPERT_FORCE UNSET (the solver is the sole authority). Default OFF.
    // Sanity/mechanism-live probe: LS_LOADER_CPU_COST_MULT<1 scales the CPU
    // ComputeCurve DOWN → the solver MUST then offload (proving the bridge is a
    // live decision, not a dead wire).
    bool   loader_cpu_solver_        = false;   ///< LS_LOADER_CPU_SOLVER
    int    loader_cpu_solver_node_   = 3;       ///< LS_LOADER_CPU_SOLVER_NODE (host bank)
    bool   loader_cpu_skip_b1_       = true;    ///< LS_LOADER_CPU_SKIP_B1: skip the B=1 solve (k=0)
    double loader_cpu_a_us_          = 1160.0;  ///< host FFN per-expert (measured threaded node-3)
    double loader_cpu_b_us_          = 440.0;   ///< per-engaged-layer fold overhead (batch-step)
    int    loader_cpu_p_             = 256;     ///< batch width (fold charged once for c≤P)
    double loader_cpu_cost_mult_     = 1.0;     ///< LS_LOADER_CPU_COST_MULT (sanity: <1 forces offload)
    // ── B-AWARE host-FFN ComputeCurve (TASK-1, 2026-08-09) ──────────────────
    // The B=1 present-wall (compute_us(cpu,1)=a+b) vetoes on the verify chunk
    // EXACTLY as at plain decode because it is B-agnostic — but the dsp52
    // SPECULATIVE verify chunk processes 1+γ (=16) rows in ONE FETCH_AND_RUN_MOE
    // (num_seqs=R). At batch B the host FFN is an M=B GEMM: its per-TOKEN cost
    // DROPS (fixed weight-read+coord amortize over B, only the compute slope is
    // ∝B), and the per-engaged-layer FOLD + the per-expert H2D FETCH are
    // B-INVARIANT fixed costs that amortize over the B chunk tokens. So the CPU
    // offload flips from the B=1 veto to a WIN in the verify band. The host FFN
    // per expert at batch B (µs) = a_fixed + a_pertok·B  (calibrated: 921 µs/tok
    // at B=1 → 71 µs/tok at B=16, matching the offline B-sweep sim). The
    // never-lose decision is LAYER-LEVEL (fetch-floor makespan across the M GPUs
    // + host FFN overlapped by the ~B× GPU window); k=0 (all-GPU champion) is
    // always in the search ⇒ never worse. At B=1 the overlap is near-serial
    // (η≈floor) and the host FFN is huge ⇒ k*=0 (zero offload, matches the
    // Task-2 engine boot). All knobs env-overridable for the B-sweep sensitivity.
    double loader_cpu_a_fixed_us_    = 907.0;   ///< LS_LOADER_CPU_A_FIXED_US: B-invariant host FFN (weight-read+coord)
    double loader_cpu_a_pertok_us_   = 14.0;    ///< LS_LOADER_CPU_A_PERTOK_US: host FFN per-token compute slope
    double loader_cpu_overlap_floor_ = 0.15;    ///< LS_LOADER_CPU_OVERLAP_FLOOR: η(1) near-serial overlap
    double loader_cpu_overlap_tau_   = 4.0;     ///< LS_LOADER_CPU_OVERLAP_TAU: η saturation batch scale
    uint64_t loader_cpu_bmax_seen_   = 0;       ///< max verify-chunk width B observed (diagnostic)
    bool   loader_cpu_k_built_       = false;   ///< loader_k_cpu_ built lazily on first solve
    int    loader_cpu_dev_pos_       = -1;      ///< CPU device solver position (== n_backends)
    gpu_loader::LoaderConstants loader_k_cpu_;  ///< live K + appended CPU device
    uint64_t loader_cpu_assign_total_ = 0;      ///< cumulative experts offloaded to CPU (all solves)
    uint64_t loader_cpu_recur_off_total_ = 0;   ///< of those, how many were in the prev-round union (would-recur/HIT)
    uint64_t loader_cpu_veto_total_   = 0;      ///< solver-chose-CPU but present-wall veto kept it on GPU
    uint64_t loader_cpu_target_rare_cand_total_ = 0;  ///< (e) cumulative target-node RARE candidates seen
    uint64_t loader_cpu_target_off_total_ = 0;  ///< (e) cumulative offloads residing on the target node (LOCAL read)
    uint64_t loader_cpu_solve_count_  = 0;      ///< solves run through the CPU-solver path
    // Build loader_k_cpu_ from *deps_.loader_constants on first use. Returns the
    // CPU device's solver position (== the pre-append num_devices), or -1 if the
    // live K is unavailable.
    int ensure_cpu_solver_k();
    // TASK-1 REEF path: the B-aware never-lose CPU-offload decision run DIRECTLY
    // on the REAL orchestrator placement (state.experts residency + target GPU),
    // used when the engine loader's shadow solve is OFF (KEEPER52_REEF_ORCH forces
    // LS_LOADER_SHADOW=0 — the TEST is the orchestrator, so route_moe_by_loader's
    // shadow post-pass never fires). Same greedy/curve as the shadow post-pass but
    // no B&B solve (zero shadow tax on the ON arm ⇒ clean A/B); populates
    // cpu_forced_experts_[layer] and marks state.experts[i].cpu_forced +
    // cpu_forced_count. Gated by loader_cpu_solver_ && !loader_shadow_ (so exactly
    // ONE of the two drivers runs). chunk_width = fetch_and_run_moe.num_seqs (=1+γ
    // on a batched verify chunk) ⇒ the offload flips on the verify band, stays
    // k*=0 (zero offload, never-lose) on plain-decode/draft B=1 steps.
    void apply_cpu_offload_never_lose(const ipc::ExpertPrefetchEntry* entries,
                                      uint32_t n, ProgressiveMoeState& state,
                                      uint32_t layer_idx, uint32_t chunk_width);

    // ── Residency-reframe weighted-sum place_cons (loader_place_sum.h) ───────
    // The CPU-column cost the never-lose greedy pays PER OFFLOADED EXPERT, on top
    // of the exposed host FFN: a weighted sum of the residency/future-miss
    // baseline (dominant), NUMA home-locality, and freq/EMA hotness. DEFAULT ON;
    // engaged only when LS_LOADER_CPU_SOLVER is on ⇒ champion-safe. The residency
    // baseline's recurrence estimate is the FREE b0_prev prev-round-union signal:
    // prev_round_experts_[layer] = the routed expert set from the PREVIOUS visit
    // to this layer; an expert in it is predicted to recur (a would-be HIT ⇒
    // EXPENSIVE to offload), an expert NOT in it is residency-safe (would-miss-
    // anyway ⇒ free). Updated at the end of each apply_cpu_offload_never_lose.
    bool                       loader_place_sum_ = true;   ///< LS_LOADER_PLACE_SUM (default ON)
    gpu_loader::PlaceSumWeights loader_place_sum_w_{};      ///< place_cons factor weights
    // Only w_numa is trained (residency+hotness are heuristics): env flag = LS_LOADER_
    // PLACE_W_NUMA was set ⇒ it wins over the trained-JSON value (env > JSON > default).
    bool loader_place_w_numa_env_ = false;
    // Latch: adopt the trained w_numa from the loaded LoaderConstants (place_sum_
    // weights block) once, at first solve (deps_.loader_constants may be set after
    // construction) unless env-overridden.
    bool loader_place_sum_calib_adopted_ = false;
    std::unordered_map<uint32_t, std::vector<uint16_t>> prev_round_experts_;
    std::unordered_set<uint16_t> loader_prev_round_set_;    ///< scratch: this layer's prev set
    bool cpu_expert_dev_resolved_ = false;
    compute::ExpertDevice* cpu_expert_dev_ = nullptr;
    std::vector<void*> cpu_fold_stage_;   ///< per-GPU-position device staging (grown)
    std::vector<size_t> cpu_fold_stage_bytes_;

    // ── C-6 Task A (graph-hoist overlap) ────────────────────────────────────
    // The host CPU-expert FFN must OVERLAP the captured GPU FFN graph replay,
    // not serialize behind it. norm+router are enqueued on rank0's kExpertFfn
    // stream BEFORE the FFN replay and an input-ready event is recorded there
    // (dispatch_moe_internal). The fold's input D2H then rides the kD2hTransfer
    // stream keyed to that event (NOT a full synchronize_device), so the host
    // compute runs while the GPU replay executes. The host fold source (moe_o)
    // is PERSISTED and guarded by per-rank consume events (deferred free) so the
    // fold never full-syncs the device — the tail wait is deferred to the next
    // layer's reuse (a no-op in steady state).
    std::vector<void*> cpu_input_ready_event_;   ///< per-gpu, recorded pre-replay
    void*  cpu_fold_moe_o_ = nullptr;            ///< persisted host fold source (CPU dev)
    size_t cpu_fold_moe_o_bytes_ = 0;
    std::vector<void*> cpu_fold_consume_event_;  ///< per-gpu, guards moe_o reuse
    void*  cpu_fold_d2h_done_event_ = nullptr;   ///< input D2H completion (rank0 dev)
    int    cpu_fold_d2h_event_dev_ = -1;
    bool   cpu_fold_moe_o_inflight_ = false;

    // ── C-6 Milestone C (async host-FFN worker) ─────────────────────────────
    // The host FFN (cpu_forced_produce → cpu_fold_moe_o_) runs on this worker,
    // kicked after Phase-1 GEMM enqueue and joined at the fold, so it overlaps
    // the GPU FFN + EP allreduce window. Produce hands the fold metadata to the
    // daemon H2D pass via the members below.
    std::thread cpu_fold_worker_;
    bool     cpu_fold_worker_kicked_ = false;
    bool     cpu_fold_worker_ok_ = false;
    uint32_t cpu_fold_worker_layer_ = 0xffffffffu;
    int      cpu_fold_active_ = 0;             ///< #forced experts actually computed
    size_t   cpu_fold_contrib_elems_ = 0;      ///< elems in moe_o (perslot P·H or B·H)
    size_t   cpu_fold_contrib_bytes_ = 0;      ///< bytes in moe_o
    bool     cpu_fold_fp32_add_ = false;       ///< fold uses fp32 add-inplace (vs bf16)

    // ── C-6 early-kick η instrumentation (host-FFN ∥ fetch overlap fraction) ──
    // Steady-clock ns. The fetch window opens at issue_moe_wave (handle_fetch_and_
    // run, forced layer) and closes when the finalize fold runs (experts arrived).
    // The host FFN interval [hf_start,hf_end] is stamped by the produce worker.
    // η = overlap(host-FFN, fetch-window) / host-FFN-duration, reported at the fold.
    uint64_t cpu_fetch_win_start_ns_ = 0;  ///< fetch-launch stamp (issue_moe_wave)
    uint64_t cpu_fetch_win_end_ns_ = 0;    ///< finalize entry = fetch done (daemon)
    uint64_t cpu_hf_start_ns_ = 0;         ///< host FFN compute begin (worker)
    uint64_t cpu_hf_end_ns_ = 0;           ///< host FFN compute end (worker)
    double   cpu_eta_sum_ = 0.0;           ///< Σ η over engaged folds
    uint64_t cpu_eta_n_ = 0;               ///< #engaged folds (active>0)

    // ── Shared micro-helpers (refactor: deduplicate repeated patterns) ──

    /// GPU position → hidden_state_pairs index (-1 = non-TP).
    int resolve_pair_idx(uint32_t gpu) const;

    /// Create a CUDA event on `gpu` and record it on `sid`. Returns the event.
    void* create_and_record_event(int gpu, compute::StreamId sid);

    /// Register a PCIe transfer token → cmd_seq mapping (bidirectional).
    void register_pcie_token(uint64_t token, uint32_t cmd_seq);

    /// Register an NVMe IO token → cmd_seq mapping (bidirectional).
    void register_nvme_token(uint64_t token, uint32_t cmd_seq);

    /// Set up device, stream, embedding, and attn_buf sync for speculation
    /// pipelines (MTP + self-spec shared preamble). Returns false if device
    /// lookup fails (error written to completion ring). On success, stream_out
    /// and pair_idx_out are populated.
    struct SpecScratch;  // defined below (near spec_scratch_ member)
    bool setup_spec_pipeline(uint32_t cmd_seq, uint32_t gpu,
                             uint32_t token_id, const char* pipeline_name,
                             SpecScratch& ss, void*& stream_out,
                             int& pair_idx_out);

    // ── Completion helpers ──────────────────────────────────────────────

    void write_cache_completion(uint32_t cmd_seq, uint32_t gpu_idx, uint32_t status);
    void write_compute_completion(uint32_t orig_cmd_type, uint32_t cmd_seq,
                                  uint32_t gpu_idx, uint32_t layer_idx,
                                  uint32_t status,
                                  uint32_t host_buf_offset = 0,
                                  uint32_t data_bytes = 0,
                                  float top1_prob = 0.0f,
                                  float entropy = 0.0f,
                                  uint8_t routed_miss_count = 0);
    void write_checkpoint_completion(uint32_t orig_cmd_type, uint32_t cmd_seq,
                                     uint32_t gpu_idx, uint32_t layer_idx,
                                     uint8_t checkpoint_type,
                                     uint32_t host_buf_offset = 0,
                                     uint32_t data_bytes = 0);
    void write_event_completion(uint32_t cmd_seq, uint32_t gpu_idx, uint32_t status);
    void write_error(uint32_t cmd_seq, uint32_t gpu_idx,
                     ipc::CmpErrorCategory category,
                     const char* msg = nullptr);
    void write_gpu_fatal(uint32_t gpu_idx, int vendor_error_code, const char* msg);
    void write_seq_completion(uint32_t cmd_seq, uint32_t gpu_idx,
                              uint64_t seq_id, uint32_t page_count,
                              uint32_t status);
    void write_nvme_completion(uint32_t cmd_seq, uint32_t gpu_idx,
                               uint32_t layer_idx, uint16_t expert_idx,
                               uint8_t op, uint32_t status);
    void write_cancel_completion(uint32_t cmd_seq, uint32_t gpu_idx,
                                 uint32_t target_cmd_seq, uint8_t cancelled,
                                 uint32_t status);

    // ── State ───────────────────────────────────────────────────────────

    Deps deps_;
    transfer::HostSourceDeps host_source_deps_;  // extracted from deps_ at construction

    /// Transfer token → originating cmd_seq.
    /// Populated on enqueue, consumed by DaemonLoop's poll_transfer_completions
    /// via resolve_cmd_seq(), cleaned up via remove_token_mapping().
    std::unordered_map<uint64_t, uint32_t> token_to_cmd_seq_;

    /// Pooled CUDA event with the GPU it was created on (TD-70m).
    struct PooledEvent { void* handle = nullptr; int gpu_idx = 0; };

    /// event_id (Python-assigned) → pooled CUDA event.
    /// Events are created lazily on first CMD_RECORD_EVENT and destroyed
    /// in the destructor via StreamManager::destroy_event().
    std::unordered_map<uint32_t, PooledEvent> event_pool_;

    // Per-sequence engine state (KV pages, fork flag, indexer pages,
    // indexer coverage, V4 side tiers) lives in ONE aggregate —
    // SequenceState / sequences_, declared further down next to its member
    // types (INV-SEQ-FORK-STATE single enumeration point).

    /// NVMe IoToken → originating cmd_seq (parallel to token_to_cmd_seq_
    /// which tracks PCIe TransferTokens).
    std::unordered_map<uint64_t, uint32_t> nvme_token_to_cmd_seq_;

    /// cmd_seq → (internal_token, is_nvme).  Reverse map for
    /// CMD_CANCEL_TRANSFER: given a cmd_seq, find which system (PCIe vs
    /// NVMe) owns the token and what the internal token value is.
    std::unordered_map<uint32_t, std::pair<uint64_t, bool>> cmd_seq_to_token_;

    /// Pending compute commands awaiting GPU event completion (KD-1).
    std::vector<PendingCompute> pending_compute_;

    /// Per-GPU flag: CMP_GPU_FATAL already emitted (INV-5c). Once per GPU lifetime.
    std::vector<bool> gpu_fatal_emitted_;

    /// KD-2: per-GPU device scratch for sampling kernel output.
    std::vector<void*> sampling_scratch_;

    /// KD-4b: per-rank device scratch for embedding token ID H2D copy.
    std::vector<void*> embedding_token_scratch_;

    /// KD-4b: per-rank device scratch for final RMSNorm before output head.
    std::vector<void*> output_norm_scratch_;

    /// KD-4f: per-rank device scratch for logits output [max_batch, vocab] FP32.
    std::vector<void*> logits_scratch_;

    /// TD-SERVE-NAMED-TOOL-CHOICE: pinned host buffer for the guided-decoding
    /// full-logits readback (ONE row: vocab_size f32).  Allocated with the
    /// logits scratch; address exposed via Engine::logits_readback_addr() —
    /// the single-process Python orchestrator reads it directly after the
    /// OUTPUT_HEAD completion (B=1 serialization makes this race-free).
    void* logits_readback_host_ = nullptr;
    size_t logits_readback_bytes_ = 0;

    /// KD-4g: per-GPU partial logits scratch for TP>1 output head [max_batch, vocab/tp] FP32.
    std::vector<void*> partial_logits_scratch_;

    /// KD-4g TD-72a: allgather recv scratch for TP>1 output head when num_tokens>1.
    /// [max_batch, vocab_size] FP32, allocated only on primary TP GPU.
    /// Needed because allgather output layout is [tp, batch, local_vocab] (rank-major)
    /// which must be transposed to [batch, vocab_size] before confidence/sampling.
    void* logits_gather_scratch_ = nullptr;
    uint32_t logits_gather_scratch_gpu_ = 0;  // GPU position owning the scratch

    /// KD-2a: per-GPU device scratch for confidence kernel output.
    std::vector<void*> confidence_top1_scratch_;
    std::vector<void*> confidence_entropy_scratch_;

    struct ConfidenceHostStaging {
        float top1_prob = 0.0f;
        float entropy = 0.0f;
    };
    std::vector<ConfidenceHostStaging> confidence_host_staging_;

    /// KD-3b: per-GPU MoE pipeline scratch buffers.
    struct MoeScratch {
        void* router_logits      = nullptr;
        void* topk_weights       = nullptr;
        void* topk_indices       = nullptr;
        // V4-4 hash gating: persistent [B] int32 layer-input token ids for
        // this step, written by the embedding handler AT row_offset (the
        // per-launch embedding_token_scratch_ holds only the CURRENT
        // sub-chunk at base row 0 — not superchunk-safe for later gating).
        // Only allocated when Deps::moe_hash_layers > 0.
        void* moe_token_ids      = nullptr;
        void* permuted_input     = nullptr;
        void* expert_offsets     = nullptr;
        void* src_to_dest_map    = nullptr;
        void* permuted_idx       = nullptr;
        void* gate_up_output     = nullptr;
        void* activation_output  = nullptr;
        void* expert_output      = nullptr;
        void* moe_output         = nullptr;
        // TD-PREFILL-FETCH-SEAM-SCALING: BF16 [expanded, H] rolling-wave
        // accumulator for over-capacity routed unions. Each wave's Step-5
        // expert_output rows are added here (rows of experts outside the wave
        // are zeros, so the add is exact); the final pass unpermutes from this
        // buffer. Zeroed lazily at the first wave pass of each FETCH command.
        void* moe_wave_accum     = nullptr;
        // DET-REDUCE Phase 1b: FP32 [B, H] routed partial-sum buffer for the
        // placement-invariant EP combine. Only allocated when
        // deterministic_ep_combine_ is on (default off ⇒ nullptr, zero VRAM cost).
        void* moe_output_fp32    = nullptr;
        // DET-REDUCE Phase 1b BF16-PAYLOAD: BF16 [B, topk, H] per-slot buffer for
        // the placement-invariant EP combine (half the bytes of moe_output_fp32).
        // Only allocated when deterministic_ep_combine_ is on AND the precision is
        // bf16 (default off / fp32 ⇒ nullptr, zero VRAM cost).
        void* moe_output_bf16_perslot = nullptr;
        void* normalized_hidden  = nullptr;
        // V4-5b mHC FFN-stage scratch (allocated only when Deps::hc_streams>1):
        // hc_x [B, H] BF16 collapsed module input; hc_post [B, hc] F32;
        // hc_comb [B, hc*hc] F32 — produced by the FFN hc_pre, consumed by the
        // FFN hc_post that replaces the Step-8 residual add.
        void* hc_x               = nullptr;
        void* hc_post            = nullptr;
        void* hc_comb            = nullptr;
        void* permute_workspace  = nullptr;
        void* gemm_workspace     = nullptr;
        size_t gemm_workspace_bytes = 0;

        // KD-3f: grouped GEMM metadata buffers (routed experts).
        void* problem_sizes      = nullptr;  // [E * 3] INT32 — {M_e, N, K} per expert
        void* sf_offsets         = nullptr;  // [E + 1] INT32 — NVFP4 scale offsets

        // KD-3e: shared expert scratch (sized for num_tokens, not expanded_tokens).
        void* shared_gate_up_output  = nullptr;  // [B, 2*I] BF16
        void* shared_activation      = nullptr;  // [B, I] BF16
        void* shared_expert_output   = nullptr;  // [B, H] BF16
        void* shared_expert_offsets  = nullptr;  // [2] INT32 = {0, num_tokens}

        // KD-3f: grouped GEMM metadata buffers (shared expert).
        void* shared_problem_sizes   = nullptr;  // [3] INT32
        void* shared_sf_offsets      = nullptr;  // [2] INT32

        // KD-3g: activation quantization scratch (reused across all 4 quant points).
        void* quant_act          = nullptr;  // FP8: [exp, max_K] E4M3 | FP4: [exp, max_K/2] packed
        void* quant_scale        = nullptr;  // FP8: [exp, ceil(max_K/128)] f32 | FP4: [sf_max, K/16] UE4M3
        size_t quant_act_bytes   = 0;
        size_t quant_scale_bytes = 0;

        // KD-4f-d: NVFP4 alpha device buffer (one float, written per-dispatch).
        void* nvfp4_alpha        = nullptr;  // [1] float32

        // FP4-ACT-SCALE: per-expert calibrated activation input_scales (TRT-LLM
        // scheme), gathered from cache slots (routed) or H2D'd (shared/dense)
        // before each activation quant; the alphas multiply the same values.
        void* moe_input_scales   = nullptr;  // [E] float32

        // KD-4f-c2: per-expert B pointer arrays for routed GEMM dispatch.
        void* routed_b_ptrs      = nullptr;  // [E] void* — device array
        void* routed_sb_ptrs     = nullptr;  // [E] void* — device array
        std::vector<const void*> routed_b_ptrs_host;   // [E] host staging
        std::vector<const void*> routed_sb_ptrs_host;  // [E] host staging

        // TD-DECODE-FFN-GRAPH (experiment): separate per-projection B/scale-B
        // device arrays + PINNED host staging so the gate/up/down b_ptrs coexist
        // and their H2D feeds can be captured into one replayable graph. The
        // shared routed_b_ptrs/host above cannot be captured: 3 interleaved
        // H2D copies into one destination from one host buffer would, on replay,
        // all read the last-written (down) pointers. Indexed [0]=gate,[1]=up,
        // [2]=down. Host buffers are cudaHostAlloc'd (pageable H2D is illegal
        // during capture). Only allocated when the FFN graph is enabled.
        void* g_b_ptrs[3]  = {nullptr, nullptr, nullptr};  // device [E] void*
        void* g_sb_ptrs[3] = {nullptr, nullptr, nullptr};  // device [E] void*
        const void** g_b_ptrs_host[3]  = {nullptr, nullptr, nullptr};  // pinned [E]
        const void** g_sb_ptrs_host[3] = {nullptr, nullptr, nullptr};  // pinned [E]

        // INV-MOE-OVERLAP: SEPARATE b_ptrs sets for the captured kPartial
        // (resident-overlap) pass. The overlap pass's captured H2D nodes read
        // their pinned host staging ASYNCHRONOUSLY (whenever the graph
        // executes on the GPU) while the host may already be refilling the
        // staging for the same layer's kFinal pass — sharing one set would be
        // a host-write-during-device-read race. Only allocated when the FFN
        // graph is enabled (the eager overlap path uses the shared
        // routed_b_ptrs whose pageable H2D is host-synchronous for the src).
        void* g_b_ptrs_w[3]  = {nullptr, nullptr, nullptr};  // device [E] void*
        void* g_sb_ptrs_w[3] = {nullptr, nullptr, nullptr};  // device [E] void*
        const void** g_b_ptrs_host_w[3]  = {nullptr, nullptr, nullptr};  // pinned
        const void** g_sb_ptrs_host_w[3] = {nullptr, nullptr, nullptr};  // pinned

        // GG-5b: 1-element device B_ptrs array for dense/shared GGUF GEMMs
        // (num_experts==1). The GGUF grouped kernel has NO B_base — only a device
        // B_ptrs array — so the single dense/shared weight pointer is H2D'd here.
        // Only allocated for GGUF weights.
        void* gguf_single_b_ptr      = nullptr;  // [1] void* — device array

        // GG-5c: dense/shared GGUF gate_up SPLIT path scratch (gate_gguf_type !=
        // up_gguf_type). A dedicated [B, I_dense] BF16 buffer holds the up GEMM's
        // output (expert_output is only sized [expanded, H] — too small when the
        // dense intermediate exceeds hidden), interleaved with the gate output
        // (activation_output / shared_activation) into the [B, 2*I] gate_up buffer
        // for the BF16 SwiGLU. Only allocated for GGUF weights.
        void* gguf_gate_up_split     = nullptr;  // [B, I_dense] BF16 — split up output

        // TD-PREFILL-MOE-BIG: chunk-partial unpermute staging for the chunked
        // WAVE passes ([moe_chunk_capacity_, H] BF16). A wave's chunk is
        // unpermuted here, then residual-added into the moe_output accumulator
        // rows (single-pass chunked dispatches unpermute straight into
        // moe_output at the chunk row offset and never touch this). Only
        // allocated when prefill_moe_big is on.
        void* big_unperm_tmp         = nullptr;

        // TD-89m: partial expert execution support.
        void* zero_weight_buf        = nullptr;  // [max_proj_bytes] zeroed — B_ptr for missing experts
        size_t zero_weight_buf_bytes = 0;
        std::vector<int32_t> topk_indices_host;      // [max_batch * topk] host staging for D2H
        std::vector<uint8_t> expert_resident_bitset; // [ceil(E/8)] — 1 bit per expert

        // TD-MOE-BIG-GEMM-SWEEP (wave-masked permute): per-chunk masked top-K
        // staging ([expanded_t] int32) + device per-expert byte mask ([E]).
        // Chunked WAVE passes mask non-wave experts' top-K entries to the -1
        // permute sentinel so their rows sort past expert_offsets[E] and every
        // grouped GEMM only covers the wave's own experts (the full-batch
        // zero-weight sweep — waves × chunks × 3 GEMMs over all E groups — was
        // the dominant prefill MoE cost). Only allocated when prefill_moe_big.
        void* wave_masked_topk       = nullptr;  // [expanded_t] int32
        void* wave_expert_mask       = nullptr;  // [E] uint8 device mask
        std::vector<uint8_t> wave_expert_mask_host;  // [E] host staging

        // INV-MOE-EP-XTP: incoming-partial staging for the EP-beyond-TP routed
        // gather. Allocated ONLY on DCP(TP)-rank GPUs and ONLY when expert-only
        // (non-DCP) GPUs exist. An extra rank's routed partial is D2D-copied
        // here, then folded into this rank's same-mode routed buffer
        // (moe_output / moe_output_fp32 / moe_output_bf16_perslot) before the
        // cross-rank EP combine. Sized max(mode-0 [B,H] bf16, per-slot
        // [Bt,K,H] at the active payload precision when the canonical combine
        // is enabled).
        void* ep_xtp_staging         = nullptr;
    };
    std::vector<MoeScratch> moe_scratch_;

    // TD-PREFILL-SUPERCHUNK: effective MoE token-batch capacity (see the
    // public moe_batch_capacity() accessor). Set once in the constructor:
    // max(kMaxBatchDescriptors, Deps::superchunk_tokens) stepped down by the
    // VRAM fail-safe (projected per-GPU MoE scratch bytes vs device_mem_info
    // free VRAM with headroom; halves the superchunk part, floor at
    // kMaxBatchDescriptors — never OOM, never below legacy capacity).
    int moe_batch_capacity_ = static_cast<int>(ipc::kMaxBatchDescriptors);
    // TD-PREFILL-MOE-BIG: single-shot token bound (see moe_chunk_capacity()).
    // With prefill_moe_big: max(compute.moe_big_chunk_tokens, max_batch_size)
    // — the TRANSIENT scratch row basis. Without: == moe_batch_capacity_
    // (chunking never engages; legacy sizing byte-identical).
    int moe_chunk_capacity_ = static_cast<int>(ipc::kMaxBatchDescriptors);
    // TD-PREFILL-MOE-BIG: mode flag (compute.prefill_moe_big), read once at
    // construction. Gates the chunk-bounded transient sizing, the elastic
    // capacity derivation, and the big_unperm_tmp allocation.
    bool moe_big_enabled_ = false;
    // TD-PREFILL-MOE-BIG scratch homing: per-GPU prefill-scratch tail span
    // (from Deps::prefill_scratch_tails; only populated when moe_big_enabled_)
    // + bump cursor. The chunk-bounded TRANSIENT MoE buffers are carved from
    // here; moe_scratch_alloc falls back to device_alloc when absent/full and
    // the destructor skips device_free for tail-carved pointers.
    std::vector<std::pair<char*, size_t>> moe_tail_span_;
    std::vector<size_t> moe_tail_used_;
    void* moe_scratch_alloc(size_t gpu_idx, size_t bytes);
    bool moe_ptr_in_tail(size_t gpu_idx, const void* p) const;
    // TD-PREFILL-SUPERCHUNK: usable HIDDEN-STATE rows for row_offset targets —
    // max(max_batch_size, min(superchunk_tokens, moe_batch_capacity_)). The
    // pair buffers are sized max(max_batch, REQUESTED superchunk) by the
    // engine, and the MoE/topk scratch covers moe_batch_capacity_ rows, so
    // this is the tight bound for embedding row_offset+num_tokens and
    // attention row_offset+num_seqs (a stepped-down capacity shrinks it).
    int superchunk_rows_ = 0;

    // TD-DECODE-FFN-GRAPH (experiment): per-GPU routed-FFN graph runners + state.
    // Enabled via LS_MOE_FFN_GRAPH env (read once at first decode dispatch).
    // Captures Step 2..6 (permute → gate/up GEMM → SwiGLU → down GEMM →
    // unpermute) on the first eligible decode dispatch per GPU, replays after.
    // INV-0.6 is intentionally violated under this flag for measurement only.
    //
    // TD-DECODE-FFN-GRAPH-GGUF-CAPTURE-BROKEN fix: ONE graph per GPU is wrong
    // on a mixed-precision GGUF — the per-layer k-quant type triple
    // (gate,up,down) selects DIFFERENT kernel instantiations, and a graph
    // captured on layer L bakes L's kernels and replays them for every layer
    // (GLM-5.2 Q4_K_XL: layer 3 is (Q5K,Q5K,Q6K), 71 layers are
    // (Q4K,Q4K,Q5K) → replay decodes Q4K blocks as Q5K → NaN logits). The
    // runners are therefore keyed per GPU by a VARIANT KEY (k-quant triple +
    // ep_combine_mode; a single key covers NVFP4/FP8). Bounded by
    // kMaxFfnGraphVariants — beyond that, dispatch falls back to eager.
    // INV-MOE-OVERLAP raises the variant space: key now also carries the
    // wave-pass kind (kNone/kPartial/kFinal) + the wave-first bit, so a mixed
    // GGUF can need (triples × combine-mode × 4 pass variants).
    static constexpr size_t kMaxFfnGraphVariants = 32;
    std::vector<std::unordered_map<
        uint32_t, std::unique_ptr<compute::RoutedFfnGraphRunner>>>
        routed_ffn_graphs_;
    int   moe_ffn_graph_enabled_ = -1;  // -1 unread, 0 off, 1 on
    bool moe_ffn_graph_enabled();

    // INV-NCCL-GRAPH (env LS_NCCL_GRAPH, default OFF): replay the per-layer
    // fused MoE-combine collectives (shared-expert reduce + routed EP-combine
    // reduce, one nccl group) through captured per-rank CUDA graphs instead of
    // eager NCCL enqueues. Decode (B==1) only; buffers/counts are fixed
    // scratch addresses, validated against the baked signature per replay.
    int  nccl_graph_enabled_ = -1;      // -1 unread, 0 off, 1 on
    bool nccl_graph_enabled();
    std::unique_ptr<compute::NcclGroupGraphRunner> moe_combine_graph_;
    std::vector<void*> moe_combine_graph_bufs_a_;   // baked shared buffers
    std::vector<void*> moe_combine_graph_bufs_b_;   // baked combine buffers
    bool moe_combine_graph_fp32_ = false;
    int  moe_combine_graph_rows_ = 1;
    bool moe_combine_graph_failed_ = false;         // capture failed → eager

    // DET-REDUCE Phase 1b: placement-invariant fp32 EP combine for the routed
    // MoE partial sums. Read once in the ctor from config.compute
    // .deterministic_ep_combine, overridable EITHER way by env
    // LAYERSTORM_DETERMINISTIC_EP_COMBINE (set & !="0" → on, "0" → off).
    // Default off ⇒ byte-identical to the legacy bf16 EP combine.
    bool deterministic_ep_combine_ = false;

    // DET-REDUCE Phase 1b BF16-PAYLOAD: when deterministic_ep_combine_ is on,
    // selects the canonical per-slot payload precision. false (default) = fp32
    // payload (today's canonical, [t,K,H] fp32 gather); true = bf16 payload
    // ([t,K,H] bf16 gather — half the bytes, fp32-accumulate reduce, matches
    // vLLM/llama.cpp). Read once in the ctor from config.compute
    // .deterministic_ep_combine_precision, overridable by env
    // LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION (bf16|fp32). Ignored when the
    // combine is off (default off ⇒ byte-identical).
    bool ep_combine_bf16_payload_ = false;

    // INV-MOE-EP-XTP: expert-only GPU positions — GPUs that host an
    // ExpertDevice + expert cache but are NOT in the DCP(TP) group (EP degree
    // beyond TP). Filled once at construction when CUDA is live, a DCP
    // executor with dcp_size>=2 exists, and the model has routed experts;
    // empty otherwise (⇒ every EP-XTP hook is a structural no-op and the
    // TP-only paths stay byte-identical).
    std::vector<int> ep_xtp_gpus_;

    // I8 P1: persistent solver (holds ~16 KiB scratch; one per dispatcher) +
    // shadow gate (LS_LOADER_SHADOW, read once in ctor). Shadow-only today.
    gpu_loader::LoaderSolver loader_solver_;
    // LOADER_STATS_LOCALITY: top-layer daemon-state stats with O(N) hot-path
    // footprint (separate from ExpertStats' dense states_ — never pollute it).
    // Built lazily (ensure_loader_stats_boards) only when the loader is active;
    // nullopt = inert (zero footprint, zero work) on the production path.
    std::optional<gpu_loader::EvictScoreBoard> evict_board_;  // per-GPU evict scores + min-heap + recent ring
    std::optional<gpu_loader::PlaceConsTable>  place_table_;  // [expert][gpu] contiguous place_cons
    bool loader_stats_boards_built_ = false;
    // MoE-layer id-space dims for the flat expert id (from live_config /
    // loader_constants), resolved once when the boards are built.
    uint32_t loader_first_moe_layer_ = 0;
    uint32_t loader_num_moe_layers_ = 0;
    uint32_t loader_experts_per_layer_ = 0;
    bool loader_shadow_ = false;
    bool loader_act_ = false;  // LS_LOADER_ACT (P2): route miss experts by the solver's j[·]
    // Hot-path solve cost levers (TD-LOADER-SHADOW-HOTPATH-COST):
    // pin cache hits to their resident device in the solve (the assignment ACT
    // executes anyway) so the exact B&B enumerates only the misses — the M=4,N=8
    // decode shape drops 4^8 -> 4^misses candidates (~145 us -> ~us). Env
    // LS_LOADER_PIN_HITS=0 restores the full-domain solve (x-ray studies).
    bool loader_pin_hits_ = true;
    // The per-layer solver-vs-orchestrator spdlog line + its per-expert string
    // building ran unconditionally under shadow; it is diagnostics, not function.
    // Env LS_LOADER_SHADOW_LOG=1 re-enables. The JSONL dump is independent.
    bool loader_shadow_log_ = false;
    // Persistent per-solve scratch (vector capacities survive across layers —
    // no per-layer heap churn on the daemon hot path).
    gpu_loader::SolveRequest loader_req_;
    std::vector<int>    loader_globals_;     // flat expert ids (place_cons gather)
    std::vector<double> loader_vcost_;       // cheapest-victim scores scratch
    std::vector<double> loader_hotness_cold_; // per-expert coldness scratch (place-hotness)
    std::vector<int>    loader_solver_pos_;  // solver j[.] -> GPU position out
    std::string arena_map_dump_path_;  // LS_ARENA_MAP_DUMP (§12h NUMA-route bridge)
    bool arena_map_dumped_ = false;
    // I8 Stage-4 mechanism-vs-choice partition: when set, run the FULL acting path
    // (per-layer solve on the hot path, FETCH-by-j reroute, eviction plumbing) but
    // FORCE the assignment to the e%tp identity (j[i]=expert_idx % tp). Isolates the
    // mechanical overhead of *acting differently* from the cost of placement choices.
    bool loader_force_identity_ = false;  // LS_LOADER_FORCE_IDENTITY
    // I8b model-vs-reality x-ray: when LS_LOADER_SHADOW_DUMP=<path> is set,
    // shadow_solve_and_log appends one JSONL model-record per solve to this file
    // (opened lazily on first record, owned for the dispatcher's lifetime).
    std::string loader_shadow_dump_path_;   // empty = disabled
    FILE*       loader_shadow_dump_fp_ = nullptr;
    // I8 P2: per-victim eviction unit cost (µs) used to build the solver's convex
    // evict_cum[j][n] term from the live cache LRU tail. Env LS_LOADER_EVICT_UNIT_US
    // (<=0 → fall back to the mean bank egress_us). Resolved once in the ctor.
    double      loader_evict_unit_us_ = 0.0;  // 0 = use mean bank egress_us default
    // place_cons cross-token reuse reward (TD-LOADER-ROUTING-CROSSTOKEN): per-device
    // miss-placement cost w/(1 + age/tau) protecting recently-used victims; the
    // bounded-age replacement for the inflation-poisoned evict term. Sim-developed
    // (loader_offline_sim/trajectory_sim.py). Env LS_LOADER_REUSE_W / _TAU.
    double      loader_reuse_w_   = 2000.0;  // µs at age 0; <=0 disables
    double      loader_reuse_tau_ = 300.0;   // age half-scale, board layer-visits (~4 tokens)
    // Decayed-frequency eviction protection (policy lab; LS_LOADER_FREQ_W/_DECAY):
    // per-access f = f·e^{-d}+1, board raw stamped clock + w·f on every resident
    // copy (touch_existing_all — also the RESIDENCY-AWARE touch that fixes
    // TD-LOADER-REUSE-ENGINE-FIDELITY's stale-replica recency). w<=0 = plain touch.
    // M2v2 exposed-wall placement objective (LS_LOADER_M2=<params json>;
    // pinned-tier-only in the solver). Loaded lazily at first solve (needs M).
    std::string loader_m2_path_;
    bool        loader_m2_tried_ = false;
    gpu_loader::M2Params loader_m2_;
    double      loader_freq_w_     = 60.0;
    double      loader_freq_decay_ = 0.1;
    double      loader_freq_mult_  = 0.9048374180359595;  // exp(-decay), hoisted in ctor
    std::unordered_map<memory::ExpertKey, float> loader_freq_;
    // I8: horizon discount γ on the evict_cons term (model §3.6/§4.1). Eviction is a
    // FUTURE-token consequence, not this-layer critical-path time — the x-ray showed
    // the undiscounted term was ~59% of predicted-T, and the 2026-07-17 trajectory
    // x-ray showed even γ=0.1 left it at 99.7% (recency-raw clock inflation, no
    // trainer coefficient). DEFAULT 0 — the bounded-age reuse place term above
    // carries victim protection now. Env LS_LOADER_EVICT_WEIGHT restores.
    double      loader_evict_weight_ = 0.0;
    // P-25 pattern chase — solver-objective term ablation (LS_LOADER_ABLATE):
    // hand the solver a copy of LoaderConstants with the named cost INPUTS
    // zeroed (xfer: matrix rate/lat + device xfer_lat; compute: per-device
    // curves; bank: egress_us; recon: overhead/added). Zeroed inputs kill the
    // corresponding objective terms with the B&B bounds trivially valid — no
    // solver change, and mask==0 leaves the original constants untouched.
    enum : uint32_t {
        kAblateXfer    = 1u << 0,
        kAblateCompute = 1u << 1,
        kAblateBank    = 1u << 2,
        kAblateRecon   = 1u << 3,
    };
    uint32_t loader_ablate_mask_ = 0;
    gpu_loader::LoaderConstants loader_k_ablated_;
    const gpu_loader::LoaderConstants* loader_k_ablated_src_ = nullptr;
    const gpu_loader::LoaderConstants& ablated_loader_constants();
    // P-25 existence proof (LS_LOADER_PLACE_AFFINITY): replace the reuse-reward
    // + evict-curve builds with the synthetic affinity-expressing objective
    // (loader_affinity_place.h) and canonicalize co-optimal miss pairings to
    // the router's routed-order round-robin. Implies full legacy ablation
    // unless LS_LOADER_ABLATE is set explicitly.
    bool loader_place_affinity_ = false;
    // TASK-A2 phase 1 (LS_LOADER_PLACE_HOTNESS): activate the dormant place_cons
    // term from the EvictScoreBoard hotness `eff`. Adds w·coldness(i)·victim_hot(j)
    // onto the miss-placement cost (loader_place_hotness.h) — steers HOT residents
    // to STAY on GPU and COLD experts toward a flat-cost CPU column. DEFAULT OFF
    // (w==0 / flag unset ⇒ champion byte-identical). Env LS_LOADER_PLACE_HOTNESS
    // (enable), _W (scale µs, may be negative), _TAU (reuse:inv age half-scale),
    // _CLAMP (0 ⇒ req.clamp_place=false so a negative w becomes a true reward).
    bool   loader_place_hotness_       = false;
    double loader_place_hotness_w_     = 2000.0;  // µs; may be negative
    double loader_place_hotness_tau_   = 300.0;   // board layer-visit age scale
    bool   loader_place_hotness_clamp_ = true;    // false ⇒ allow negative place
    // TASK-1 freq-prior coldness (LS_LOADER_PLACE_HOTNESS_FREQ=<M3 CSV>): the
    // board `eff` cannot score a NON-resident expert (coldness≈1 for ~all B=1
    // misses ⇒ the term collapses to a scaled reuse penalty — the −2.4% Phase-2
    // loss). Replace the coldness source with the SAME per-(layer,expert) M3
    // frequency prior that drives arena placement (arena_placement.h): a routed
    // MISS now carries a real coldness from its fetch-frequency rank —
    //   coldness(i) = clamp(1 − freq(i)/fhot, 0, 1)
    // (freq=0 ⇒ maximally cold 1; freq≥fhot ⇒ hot 0). fhot defaults to the
    // table's max count (parameter-free); _FHOT overrides. Loaded once; empty
    // path ⇒ legacy board-eff coldness (byte-identical to Phase-1).
    bool   loader_place_hotness_use_freq_ = false;
    double loader_place_hotness_fhot_     = 0.0;  // 0 ⇒ table max count
    std::unordered_map<memory::ExpertKey, float> loader_place_hotness_freq_;
    // (e) TARGET-NODE RARE offload steering (loader_place_sum.h factor e). The
    // GPU-FREE DDR node whose experts are the most expensive to GPU-fetch
    // (LS_LOADER_PLACE_OFFLOAD_NODE; -1 ⇒ off) and the normalized-M3 rarity cutoff
    // (LS_LOADER_PLACE_RARITY_THRESH). Together they resolve PlaceSumFactors::
    // target_rare (1 iff location_node==node AND normM3<thresh).
    int    loader_place_offload_node_   = -1;   ///< LS_LOADER_PLACE_OFFLOAD_NODE
    double loader_place_rarity_thresh_  = 0.0;  ///< LS_LOADER_PLACE_RARITY_THRESH
    // P-25 place-affinity fine-tick pooled-LRU age signal (dispatcher-owned,
    // arm-only). The board's per-layer grouped clock cannot order same-layer
    // victims ACROSS devices (all of a layer-visit's touches share one clock),
    // but the keeper router's tie-break compares exactly such victims at the
    // eviction frontier (witnessed: first E0 divergence was a same-layer
    // cross-device age tie). This mirror stamps per-TOUCH ticks in the keeper
    // model's application order — per-GPU grouped (g ascending), hits before
    // misses, routed order within — and answers "oldest stamped resident on g,
    // excluding this layer's hits" via a lazy min-heap (stale entries: map tick
    // mismatch or no longer board-resident). Untouched (empty) outside the
    // place-affinity arm; the board's clock/eviction semantics are unchanged.
    uint64_t aff_fine_tick_ = 0;
    std::vector<std::unordered_map<memory::ExpertKey, uint64_t>> aff_fine_;  // [gpu pos]
    std::vector<std::vector<std::pair<uint64_t, memory::ExpertKey>>> aff_fine_heap_;
    void   aff_fine_stamp(const ipc::ExpertPrefetchEntry* entries, uint32_t n,
                          const int* pos_of_expert);
    double aff_fine_oldest(int pos, const memory::ExpertKey* excl, int nexcl);
    // EXPERIMENT (env LS_EVICT_DECAY, 0 = off): drive the EvictScoreBoard with a
    // recency-decay reuse score. Each new token multiplies every board raw score
    // by this factor (0<f<1); every routed expert is re-touched back to base
    // (0.9) on use → recently-used ≈ keep, long-idle → 0 = evict. Feeds evict_cum
    // (placement) AND the cheapest-victim order. Default off (the parked external
    // score SOURCE for EVICTBOARD_EXTERNAL_SCORES, wired as an A/B knob).
    double      loader_evict_decay_ = 0.0;
    // LS_LOADER_MACH_PROF=1: per-block wall accumulators around the loader
    // ACT-path per-layer machinery (route_moe_by_loader / shadow_solve_and_log /
    // issue_moe_wave_ensure_residents). Default OFF = one predicted branch per
    // block, zero clock reads. One summary table via spdlog at destruction.
    struct LoaderMachProf {
        enum Block : int {
            kTouch = 0,        // per-access freq update + touch_existing_all loop
            kReqBuild,         // SolveRequest build (cache lookups, pins, banks)
            kPlaceGather,      // place_cons row gather
            kReuse,            // reuse place-cost block (age/free-slot scan)
            kEvictCurve,       // per-device evict_cum curve build (heap walks)
            kSolve,            // LoaderSolver::solve
            kResultOut,        // solver j[.] -> out_pos + counts (+log/dump)
            kReroute,          // ACT reroute loop (+force-identity overwrite)
            kEnsureVictims,    // ensure_residents wave: grouping + victim gather
            kEnsureResidents,  // ensure_residents wave: ELM calls + issued scan
            kNumBlocks
        };
        bool enabled = false;
        std::array<uint64_t, kNumBlocks> ns{};
        std::array<uint64_t, kNumBlocks> calls{};
        // Solve cost bucketed by the number of FREE (unpinned) experts — the
        // B&B branch depth. Bucket 9+ clamps (decode top-K is 8).
        std::array<uint64_t, 10> solve_ns_by_free{};
        std::array<uint64_t, 10> solve_calls_by_free{};
        void add(int b, uint64_t d) { ns[b] += d; ++calls[b]; }
        void add_solve(int nfree, uint64_t d) {
            if (nfree < 0) nfree = 0;
            if (nfree > 9) nfree = 9;
            solve_ns_by_free[nfree] += d;
            ++solve_calls_by_free[nfree];
        }
    };
    LoaderMachProf mach_prof_;
    void report_loader_mach_prof() const;  // dtor summary (enabled only)
    // NOTE (rejected optimization): hoisting issue_moe_wave_ensure_residents'
    // per-layer grouping containers into persistent ascending-GPU scratch was
    // REVERTED — the per-GPU ELM calls are NOT order-independent under REEF:
    // evicting a replicated key on one GPU reranks its duplicate copies on
    // OTHER GPUs (EvictScoreBoard cross-GPU cascade), so a different GPU
    // processing order changes later victim picks (hit-rate fingerprint
    // drift). Any future hoist must preserve the exact container iteration
    // order it replaces.

    // TD-89m: set by dispatch_moe_internal, read by handle_fused_compute_command.
    uint8_t last_moe_miss_count_ = 0;

    // F-7: seam-routing checkpoint recorded by dispatch_moe_internal when
    // emit_seam_checkpoint is set, read by handle_fused_compute_command to
    // populate the CMP_CHECKPOINT (mirrors last_moe_miss_count_). Carries the
    // real {kSeamRouting, offset, bytes} of the routing published to sideband.
    struct SeamCheckpoint {
        uint8_t  checkpoint_type = 0;  // CheckpointType::kSeamRouting when valid
        uint32_t host_buf_offset = 0;  // kSeamCheckpointOff
        uint32_t data_bytes      = 0;  // 0 when no routing was published
    };
    SeamCheckpoint last_seam_checkpoint_{};

    // ── #90: Progressive fetch-and-run MoE state machine ────────────────

    enum class ProgressiveMoePhase : uint8_t {
        kIdle = 0,
        kWaitingExperts,   // first compute done, waiting for H2D completions
        kFinalize,         // all experts arrived (or timeout) — final pass
        // TD-PREFILL-FETCH-SEAM-SCALING: a wave-partial compute pass has been
        // enqueued; waiting on its kExpertFfn events before evicting the wave's
        // experts (the GPU is still reading their VRAM slots) and issuing the
        // next capacity-bounded wave.
        kWaveDrain,
    };

    struct ExpertRequest {
        memory::ExpertKey key;
        int target_gpu = -1;
        bool was_cached = false;    // resident at INIT time
        bool is_arrived = false;    // H2D complete (or was_cached)
        bool is_locked = false;     // holding eviction lock
        // F-6: selective-fetch decider state.
        float weight = 0.0f;        // gating weight (entry order; 0 if not provided)
        bool fetch_requested = false; // missing expert selected for H2D (vs. skipped)
        // TD-FAR-GATING: cache zone from the sideband entry, kept so the
        // progressive poll can RE-ISSUE ensure_resident for a requested expert the
        // ELM dropped (transient arena slot-pressure reverted it to ABSENT with no
        // interest). Without a retry the expert never arrives despite warm host
        // data, and the MoE finalizes on a partial routed set (≠ RUN_MOE).
        uint8_t zone = 0;           // 0 = stable, 1 = streaming
        // TD-PREFILL-FETCH-SEAM-SCALING: capacity-bounded wave scheduling.
        // issued: ensure_resident has been sent (fetches are issued in waves
        // bounded by the target GPU's free stable slots — never over-issued, so
        // the reserve-fail→drop→re-issue livelock is structurally impossible).
        // computed: this expert's rows were accumulated by a wave-partial pass
        // (it may be evicted afterwards; the final pass must exclude it).
        bool issued = false;
        bool computed = false;
        // C-6 Milestone A: this (layer,expert) is forced onto the host CPU
        // expert device — it is NEVER made resident/fetched/locked on any GPU
        // and is excluded from every GPU's routed bitset; the CPU computes it
        // and folds the contribution post-EP-combine (fold_cpu_forced_experts).
        bool cpu_forced = false;
    };

    struct ProgressiveMoeState {
        // Command identity
        uint32_t cmd_seq = 0;
        uint32_t gpu_idx = 0;      // primary GPU (command header gpu_idx)
        uint32_t layer_idx = 0;
        uint32_t num_seqs = 0;
        uint8_t  moe_mode = 0;
        // E_CMD_FAR_FORWARD_LAYER delegation: the fused handler starts this
        // progressive MoE under ITS cmd_seq — the finalize completion echoes
        // cmp_cmd_type_override (0 = the plain FETCH type) and carries
        // cmp_data_bytes (the deduped entry count) in data_bytes.
        uint32_t cmp_cmd_type_override = 0;
        uint32_t cmp_data_bytes = 0;

        // TD-PREFILL-MOE-BIG: E_CMD_FETCH_AND_RUN_MOE_BIG — enables the
        // double-buffered wave pipeline (half-budget first wave + next-wave
        // issue at compute-enqueue time). chunk_tokens = per-command chunk
        // override for the chunked GGEMM path (0 = engine default).
        bool     big = false;
        int      chunk_tokens = 0;

        // Expert tracking
        std::vector<ExpertRequest> experts;

        // Union-aware cache partitioning: bitmask of GPU positions that hold
        // TRANSIENT (zone=1, streaming-zone) union entries for this command.
        // Propagated to PendingCompute::transient_sweep_mask at finalize; the
        // completion reap then releases every evictable streaming resident on
        // these GPUs (sweep-on-reap — after the kExpertFfn event, so the GPU
        // is done reading the weights). 0 for every zone=0-only command
        // (all current production/test paths) — byte-identical when unused.
        uint32_t transient_gpu_mask = 0;

        // Phase + counters
        ProgressiveMoePhase phase = ProgressiveMoePhase::kIdle;
        int total_experts = 0;
        int arrived_count = 0;         // cached + H2D completed
        int last_computed_count = 0;   // experts included in last compute pass

        // Timeout
        uint64_t deadline_ns = 0;      // steady_clock deadline (0 = no timeout)
        bool timed_out = false;

        // F-6: selective-fetch decider — count of missing experts deliberately
        // skipped (not fetched) by the decider. These degrade gracefully exactly
        // like timed-out experts and are reported in routed_miss_count.
        int skipped_count = 0;

        // C-6 Milestone A: count of experts forced onto the host CPU device.
        // They are neither fetched (arrived) nor skipped (missed) — they are
        // computed on host and folded at finalize. Counted toward the "all
        // accounted for" phase check so the command does not wait on them, but
        // NOT reported as misses.
        int cpu_forced_count = 0;

        // TD-PREFILL-FETCH-SEAM-SCALING: rolling-wave state.
        // wave_ran: at least one wave-partial pass accumulated → the finalize
        // must run with MoeWavePass::kFinal (consume the accumulator).
        bool wave_ran = false;
        // INV-MOE-OVERLAP: the decode resident-overlap pass already broadcast
        // rank0's normalized hidden + top-K to the extra (EP-XTP) ranks for
        // this layer — the finalize must NOT re-broadcast (the D2D writes
        // would race the extras' already-enqueued kPartial reads; the data is
        // identical anyway).
        bool xtp_broadcast_done = false;
        // Per-rank kExpertFfn events of the in-flight wave-partial pass
        // (gpu position, event). Polled in kWaveDrain; the wave's experts are
        // evicted only after ALL events fire (GPU done reading their slots).
        std::vector<std::pair<int, void*>> wave_events;
        // Indices (into experts) enqueued in the current wave-partial pass —
        // unlocked + evicted at drain to free slots for the next wave.
        std::vector<size_t> wave_batch;

        // LS_MOE_BIG_XRAY: clean per-command phase decomposition (the
        // perf_trace markers pair garbage for the BIG command — cmd_seq
        // collisions). Only written when moe_big_xray_enabled(); zero cost
        // otherwise. Wall gaps between advance_progressive_moe calls are
        // attributed to the phase in effect at gap start (fetch-WAIT vs
        // GPU-drain); issue/enqueue are timed inside their functions.
        uint64_t xray_enter_ns = 0;
        uint64_t xray_last_ns = 0;
        uint64_t xray_wait_experts_ns = 0;   // waiting on H2D arrivals
        uint64_t xray_wave_drain_ns = 0;     // waiting on wave GPU events
        uint64_t xray_issue_ns = 0;          // host: issue_moe_wave
        uint64_t xray_wave_enqueue_ns = 0;   // host: run_moe_wave_pass
        uint32_t xray_waves = 0;
        uint32_t xray_fetches = 0;
        float    xray_wave_gpu_ms = 0.0f;    // Σ wave kExpertFfn elapsed
        // Per-wave GPU timing-event pairs (gpu, start, end); reaped (elapsed →
        // xray_wave_gpu_ms) at wave drain, when the wave's kExpertFfn event —
        // recorded AFTER the end timing event on the same stream — has fired.
        std::vector<std::tuple<int, void*, void*>> wave_timing_events;

        bool is_active() const { return phase != ProgressiveMoePhase::kIdle; }

        /// Unlock all locked experts via the cache. Called on cleanup/error.
        void release_locks(memory::ExpertCache* cache);
    };

    std::optional<ProgressiveMoeState> progressive_moe_;

    // E_CMD_REEF_ROUTE service (dispatch_reef.cpp): lazily-built REEF
    // decision stack (calibrated solver + evict board, self-consistent
    // residency model). reef_service_failed_ latches a construction failure
    // so every later command fails fast with the remembered reason.
    std::unique_ptr<gpu_loader::ReefOrch> reef_service_;
    bool reef_service_failed_ = false;
    // LS_REEF_RELOC_TRACE=1: arena location-change sink installed (writes
    // "M <solve> <kind> <layer> <expert> <old> <new>" into the reef
    // decision dump); cleared in the destructor (the arena outlives us).
    bool reef_reloc_trace_installed_ = false;
    std::string reef_service_error_;

    // TD-PREFILL-FETCH-SEAM-SCALING: per-GPU "wave accumulator touched" flags
    // for the ACTIVE progressive MoE command (indexed by gpu position, sized
    // like moe_scratch_). The first wave/final pass on a GPU zeroes
    // moe_wave_accum before adding; reset by handle_fetch_and_run_moe.
    std::vector<uint8_t> moe_wave_accum_used_;

    // §12h Variant-A diagnostic (LS_MOE_FOLD_VIA_HOST): per-extra-GPU pinned
    // bounce buffer {ptr, capacity} for the fold-via-host-RAM hop, plus the
    // per-GPU event guarding buffer reuse {event, owner gpu = fold dst}.
    // Lazily grown; freed in the destructor.
    std::vector<std::pair<void*, size_t>> fold_host_staging_;
    std::vector<std::pair<void*, int>> fold_host_ev_;

    /// TD-PREFILL-FETCH-SEAM-SCALING: issue ensure_resident for un-issued
    /// requested experts, bounded per target GPU by the free stable-zone slot
    /// count (zone-0 entries; streaming-zone entries are not budgeted). Returns
    /// the number of fetches issued. Budgeting is skipped when CUDA/the expert
    /// cache are absent (test/null-backend parity with the legacy issue-all).
    /// issue_moe_wave is a thin timing shim (LS_MOE_BIG_XRAY accumulates the
    /// host cost + issue count into st) around issue_moe_wave_inner.
    int issue_moe_wave(ProgressiveMoeState& st);
    int issue_moe_wave_inner(ProgressiveMoeState& st);
    // Union-aware cache partitioning: release every evictable streaming-zone
    // resident on the GPUs in gpu_mask (transient union entries + any spent
    // prefetched streaming copies). Locked / interested / mid-transfer
    // entries are skipped (request_evict refuses them). Called at completion
    // reap of a FETCH_AND_RUN_MOE command that carried zone=1 entries.
    void sweep_transient_streaming(uint32_t gpu_mask);

    /// TD-FAR-SLOT-RESERVE-STALL: decode-path replacement for the budgeted
    /// per-expert issue + one-shot apply_far_evictions. One ExpertLifecycleManager
    /// ::ensure_residents call per target GPU with a cheapest-first, pre-filtered
    /// (not-needed-this-command, not-locked) victim list from the evict board, so a
    /// full stable zone evicts inline instead of parking. Returns fetches issued.
    /// Gated to num_seqs==1 (env LS_FAR_ENSURE_RESIDENTS, default ON).
    int issue_moe_wave_ensure_residents(ProgressiveMoeState& st);

    /// TD-PREFILL-FETCH-SEAM-SCALING: enqueue a wave-partial compute pass
    /// (Steps 2..5 + accumulate) over every arrived-and-uncomputed expert,
    /// per target GPU, and record per-rank kExpertFfn events into
    /// st.wave_events. Marks the passed experts computed and fills
    /// st.wave_batch. Returns true if a pass was enqueued.
    bool run_moe_wave_pass(ProgressiveMoeState& st);

    /// INV-MOE-OVERLAP (decode fetch-overlap split, env LS_MOE_RESIDENT_OVERLAP,
    /// default ON): right after the missing-expert H2Ds are ISSUED, enqueue a
    /// wave-partial pass (Steps 2..5 + accumulate) over the experts that are
    /// ALREADY resident — on the TP ranks AND the EP-XTP extra ranks (after
    /// the rank0 hidden/top-K broadcast) — so resident-expert compute overlaps
    /// the in-flight H2D instead of waiting behind it. Unlike the prefill
    /// rolling waves this pass records NO wave_events and fills NO wave_batch
    /// (the computed experts stay locked + resident — no drain, no evict);
    /// the finalize then runs MoeWavePass::kFinal over only the just-fetched
    /// remainder (arrived ∧ !computed) and unpermutes from the accumulator —
    /// bit-identical to the single-pass result (each permuted row is written
    /// by exactly one pass; x + 0 = x). Returns true if a pass was enqueued.
    bool run_moe_overlap_pass(ProgressiveMoeState& st);
    int moe_resident_overlap_enabled_ = -1;  // -1 unread, 0 off, 1 on
    bool moe_resident_overlap_enabled();

    /// TD-FAR-STREAM-GATE (probe, env LS_FAR_STREAM_GATE, default OFF): the
    /// low-hanging HALF of the device-side fetch→finalize gate. For a DECODE
    /// (B==1) layer whose entire routed fetch set is already DISPATCHED onto
    /// the per-GPU H2D streams, record one barrier event per involved GPU on
    /// its h2d stream, make that GPU's kExpertFfn stream wait on it
    /// device-side, and mark the fetched experts arrived — so the finalize
    /// GEMM can be ENQUEUED immediately without the host busy-polling
    /// per-expert `is_arrived` (the ~957 µs/fetch detection wall,
    /// spec/TECH_DEBT.md TD-FAR-STREAM-GATE). Returns true iff the gate
    /// engaged (caller then finalizes via the existing arrived+skipped==total
    /// branch). Returns false — leaving the byte-identical host-poll path in
    /// charge — whenever: flag off, B>1, waves would apply (some requested
    /// expert un-issued), any expert not yet stream-dispatched, or any
    /// null-backend dependency. Full fix (up-front GEMM enqueue, no
    /// progressive machine for decode, hybrid deadline/partial) stays in the
    /// TD entry.
    bool try_far_stream_gate(ProgressiveMoeState& st);

    /// Shared body of the h2d-barrier device gate (no env check): verify every
    /// pending fetch is stream-ordered on its GPU's h2d stream, arm one barrier
    /// event per involved GPU, make that GPU's kExpertFfn stream wait on it,
    /// and mark the gated experts arrived+locked. Used by try_far_stream_gate
    /// (LS_FAR_STREAM_GATE — pre-overlap, single kFull finalize) and by the
    /// gated-final hybrid (LS_FAR_GATED_FINAL — post-overlap-pass, kFinal
    /// finalize queued behind the barrier). `tag` names the engaging flag in
    /// the one-time engagement log. Fallback reasons are counted in
    /// gated_final_fb_* (teardown summary) so an A/B can verify the gate
    /// actually ENGAGED per layer, not just once.
    bool far_stream_gate_commit(ProgressiveMoeState& st, const char* tag);

    /// LS_FAR_GATED_FINAL (hybrid of INV-MOE-OVERLAP + TD-FAR-STREAM-GATE):
    /// after the resident-overlap kPartial pass is enqueued on the kExpertFfn
    /// streams, device-gate the fetched remainder on per-GPU h2d barriers and
    /// finalize IMMEDIATELY — kFinal's kernels queue behind the barrier, so
    /// host arrival-detection + the finalize host dispatch leave the fetch
    /// critical path (they run DURING the DMA window). Bit-identical compute
    /// (same passes, same kernels — only enqueue time changes).
    int moe_gated_final_enabled_ = -1;  // -1 unread, 0 off, 1 on
    bool moe_gated_final_enabled();
    uint64_t gated_final_engaged_ = 0;        // fetch layers device-gated
    uint64_t gated_final_fb_unissued_ = 0;    // wave split → host path
    uint64_t gated_final_fb_not_transferring_ = 0;  // ELM not kTransferring
    uint64_t gated_final_fb_not_dispatched_ = 0;    // DMA staged, not on-stream
    uint64_t gated_final_fb_other_ = 0;       // null deps / no vram / barrier fail

    // ── ExpertStats recency feed (FETCH_AND_RUN_MOE path) ───────────────
    // Monotonic per-TOKEN id handed to ExpertStats::update() so last_used_token /
    // recency self-advances. A token's MoE layer sequence restarts at the lowest
    // MoE layer, so we bump the id whenever handle_fetch_and_run_moe sees a
    // layer_idx that is not strictly greater than the previous one (i.e. a new
    // token began). last_stats_layer_ tracks the previous FETCH layer for that
    // boundary detection; UINT32_MAX = no FETCH seen yet (first FETCH = token 1).
    uint64_t stats_token_id_  = 0;
    uint32_t last_stats_layer_ = UINT32_MAX;

    // 13c-2.0 Option A: orchestrator-eviction-map accounting (logged at dtor).
    // honored = provided victims actually evicted; rejected = provided victims
    // refused (needed-this-layer / locked / not evictable); fallback = victims
    // chosen by the local unranked scan (sentinels + rejections + no-map path).
    uint64_t far_evict_honored_ = 0;
    uint64_t far_evict_rejected_ = 0;
    uint64_t far_evict_fallback_ = 0;

    /// Queued-command backpressure: block-drain the active progressive MoE to
    /// finalization (advance + lifecycle pump each iteration) so a burst-
    /// published FAR/FETCH behaves like serial pacing instead of erroring.
    /// Returns false if no pump is injected or the safety cap expires (caller
    /// falls back to the legacy reject). Daemon-thread only.
    bool drain_progressive_moe(const char* ctx);
    std::function<void()> lifecycle_pump_;

    /// TD-far: true iff at least one requested-but-not-arrived expert is still
    /// genuinely progressing toward residency (slot reserved, H2D in flight, or
    /// host read pending). When false, advance_progressive_moe() finalizes the
    /// command immediately instead of waiting out the per-command timeout — the
    /// common FETCH_AND_RUN case lists the full expert set, of which only the
    /// routed top-K ever become resident, so the rest would otherwise stall the
    /// deadline every layer.
    bool any_fetch_in_flight(const ProgressiveMoeState& st) const;

    /// KD-3c: per-GPU speculation pipeline scratch buffers.
    struct SpecScratch {
        void* hidden_a     = nullptr;  // [1, hidden_size] BF16
        void* hidden_b     = nullptr;  // [1, hidden_size] BF16 (prev layer snapshot)
        void* logits       = nullptr;  // [1, vocab_size] FP32
        void* cos_sim_out  = nullptr;  // [1] FP32
        void* readback     = nullptr;  // 8 bytes: token_id(u32) + confidence(f32)
        void* mtp_concat   = nullptr;  // [1, 2*hidden_size] BF16
    };
    std::vector<SpecScratch> spec_scratch_;

    /// KD-4e: per-rank KV cache metadata scratch (host staging + device scratch).
    /// TD-GOLDEN-KVMETA-PER-LAYER: block tables and slot mappings hold ALL
    /// kv_layers_ layers, layer-major with layer stride = the batch size of the
    /// last build (block_tables[(l*B + b)*max_blocks + j], slot_mappings[l*B + b]);
    /// seqlens_k is layer-invariant.
    struct KvMetaScratch {
        std::vector<int> host_seqlens_k;      // [max_batch_size]
        std::vector<int> host_block_tables;   // [kv_layers * max_batch * max_blocks]
        std::vector<int> host_slot_mappings;  // [kv_layers * max_batch]
        void* dev_seqlens_k    = nullptr;     // device: [max_batch_size] int32
        void* dev_block_tables = nullptr;     // device: [kv_layers * max_batch * max_blocks] int32
        void* dev_slot_mappings = nullptr;    // device: [kv_layers * max_batch] int32
        // KVS-2 (sharded KV only): device GLOBAL seqlens (token_pos+1; identical
        // values on every rank, each rank holds its own GPU's copy). Under
        // sharding, host_seqlens_k / dev_seqlens_k above hold rank-LOCAL KV
        // shard lengths; RoPE / position math must use the global array.
        // nullptr under replicated mode (seqlens_k IS global there).
        void* dev_global_seqlens = nullptr;   // device: [max_batch_size] int32
    };
    std::vector<KvMetaScratch> kv_meta_scratch_;     // [dcp_size]
    std::vector<void*>         kv_cache_base_ptrs_;  // [dcp_size] — kv_main base per rank
    int max_blocks_per_seq_ = 0;
    int chunk_size_pages_ = 0;  ///< KD-4e1: pages per auto-growth chunk (from config)

    /// KVS-2: sequence-sharded KV mode (hardware.dcp_kv_mode == sharded),
    /// effective only at dcp_size >= 2; resolved once at construction.
    /// Replicated mode keeps every legacy path byte-identical (INV-DCP-KVREP).
    bool kv_sharded_ = false;
    /// KVS-2: dcp_chunk_size (tokens per round-robin ownership chunk) mirrored
    /// from config — MUST equal PageAllocator::dcp_config().dcp_chunk_size.
    int kv_dcp_chunk_tokens_ = 16;
    /// KVS-2: host staging for the GLOBAL seqlens array (sharded mode only).
    std::vector<int> host_global_seqlens_;
    /// KVS-2: per-rank TRASH slot base (page_idx * page_size of a dedicated
    /// kMain page on each rank's GPU, never referenced by any block table).
    /// k_append rows NOT owned by a rank are routed here, so the kernels need
    /// no negative-slot support and CUDA-graph capture stays unconditional.
    /// -1 = unavailable (sharded metadata build then fails closed). Sharded
    /// mode only; empty under replication.
    std::vector<int> kv_trash_slot_base_;
    std::vector<memory::PageHandle> kv_trash_pages_;  ///< owned; freed in dtor

    /// TD-51cb: dirty guard — skip rebuild when sideband hasn't changed.
    /// No layer term: the build covers all layers (TD-GOLDEN-KVMETA-PER-LAYER).
    int      kv_meta_last_batch_size_ = -1;
    uint64_t kv_meta_last_seq_id_     = ~0ULL;
    uint32_t kv_meta_last_token_pos_  = ~0U;

    /// TD-PREFILL-KVMETA-BT-SPLIT: block-table split of the dirty guard.
    /// The block tables ([kv_layers × batch × max_blocks] — 165.7 MiB/rank at
    /// max_batch 256 / 32k window) depend ONLY on the batch row→seq mapping
    /// and each seq's allocated page list, NOT on token_pos: per-step
    /// fingerprint changes (chunked prefill advances token_pos every
    /// sub-chunk) must re-stage only seqlens + slot_mappings (~82 KiB), and
    /// the block tables rebuild + re-upload ONLY when `kv_pages_epoch_`
    /// advanced (any seq page-topology mutation: create/free/fork/restore/
    /// re-promotion — every invalidate_kv_meta() site), the batch size
    /// changed, or the row→seq composition changed. This removed a
    /// 165.7 MiB pageable H2D per attention sub-chunk (~30 s of a 103 s
    /// 2876-token prefill). Demotion-neutralized handles (page_idx = -1) do
    /// NOT bump the epoch — stale device block-table entries for cold pages
    /// are never read (INV-KVT-2; same contract the per-token fingerprint
    /// skip already relied on within a decode step).
    uint64_t kv_pages_epoch_    = 0;
    uint64_t kv_bt_built_epoch_ = ~0ULL;   ///< epoch of last block-table build
    int      kv_bt_batch_size_  = -1;      ///< batch size of last bt build
    std::vector<uint64_t> kv_bt_seq_ids_;  ///< row→seq of last bt build

    /// Invalidate the KV-metadata dirty guard AND the block-table epoch —
    /// every call site that previously poisoned kv_meta_last_batch_size_
    /// mutates page topology (or must force a re-upload), so both levels
    /// must rebuild on the next build_kv_metadata.
    void invalidate_kv_meta() {
        kv_meta_last_batch_size_ = -1;
        ++kv_pages_epoch_;
    }

    /// GLM-25k: DSA-guided KV tiering manager (memory.kv_tiering.enabled).
    /// Constructed at dispatcher init when the model has DSA, KV is
    /// replicated, the cache format is SnapMLA FP8 and CUDA is live; null
    /// otherwise (non-tiered path byte-identical). Engaged per attention
    /// dispatch on sparse-blessed B==1 decode steps AND (TD-KVT-PREFILL)
    /// blessed B==1 sparse prefill chunks (compute.dsa_sparse_prefill,
    /// replicated KV); demotion runs after each such layer's attention.
    /// Fork of a demoted parent REFCOUNT-shares its cold slots
    /// (on_seq_fork); rewind-into-demoted / restore-onto-demoted re-promote
    /// cold pages back to VRAM (repromote_for_rewind / repromote_seq via
    /// the alloc_page seam, before the kv-meta build).  Fail-closed guards
    /// still reject drafts/graph-replay (Phase-12, TD-KVT-SPEC-FORK
    /// residual), dense prefill (TD-KVT-PREFILL-REPROMOTE) and B>1 cohorts
    /// (TD-KVT-BATCH-COHORT) on a demoted sequence.
    std::unique_ptr<KvTieringManager> kv_tiering_;
    /// TD-V4-KVT (P3): V4 CSA-bucket tiering (page-granular demote +
    /// selection-driven repromote); constructed instead of kv_tiering_
    /// for deepseek_v4 configs with memory.kv_tiering.enabled.
    std::unique_ptr<V4KvTiering> v4_kv_tiering_;
    /// Per-rank host block-table row pointers staged for begin_layer.
    std::vector<const int*> tier_bt_scratch_;

    /// TD-GLM-INDEXER-PAGED/-BATCH: paged indexer-K provisioning (dcp==1
    /// producer scope; per-entry rows support B>1). Pages come from
    /// Pool::kIndexerK per computing layer (IndexShare full ∪ layer 0); the
    /// HOST table of device page base pointers is handed to DcpExecutor via
    /// AttentionExecParams ([batch * batch_stride + layer * page_stride +
    /// logical_page]). Freed with the sequence. Returns false on pool
    /// exhaustion (producer falls back: B==1 arena, B>1 dense — never an
    /// error).
    // Debug/test-only checkpoint (CMD_SEQ_SNAPSHOT/RESTORE) — see
    // dispatch_lifecycle.cpp for the file format.
    void handle_seq_snapshot(const ipc::Command& cmd);
    void handle_seq_restore(const ipc::Command& cmd);

    bool ensure_indexer_pages(uint64_t seq_id, uint32_t token_pos,
                              int batch_slot, int dcp_size);

    /// ── V4-7b (ticket H): per-seq V4 side-tier pages ─────────────────────
    /// kSwa: ONE ring page per layer (window == page_tokens ⇒ decode-exact);
    /// kHca: pages per HCA layer grown at 128-boundaries; LID: kIndexerK
    /// pages per CSA layer (entry = CSA block). Freed at seq_free (explicit
    /// handle frees + DcpExecutor::v4_free_sequence for the state rings).
    /// V4-2c TP: tiers are PER RANK (each TP GPU keeps its replicated
    /// side-tier pages; page ids are pool-relative per GPU).
    struct V4SeqTiers {
        // [rank][n_layers] / [rank][n_layers][pg]
        std::vector<std::vector<memory::PageHandle>> swa;
        std::vector<std::vector<std::vector<memory::PageHandle>>> hca;
        std::vector<std::vector<std::vector<memory::PageHandle>>> lid;
    };
    /// Grow seq's side-tier pages to cover token_pos. Fail-closed (false) on
    /// pool exhaustion — a V4 step must never run with missing tier slots.
    bool ensure_v4_tier_pages(uint64_t seq_id, uint32_t token_pos);

    /// Shared side-tier page claim (P2 dedup): allocate one `pool` page on
    /// `gpu` for (seq, layer), stamp meta, zero the first `zero_bytes`
    /// (ticket-J determinism — reused pages must not leak residue).
    /// nullopt on exhaustion; caller logs + fails closed.
    std::optional<memory::PageHandle> claim_zeroed_tier_page(
        int gpu, memory::Pool pool, uint64_t seq_id, int layer,
        int64_t zero_bytes);
    // Per-call staging for AttentionExecParams::v4 (host arrays).
    parallelism::AttentionExecParams::V4Step v4_step_{};
    std::vector<parallelism::AttentionExecParams::V4StepRank> v4_step_ranks_;
    std::vector<std::vector<int>> v4_hca_ids_;          // [rank][pg]
    std::vector<std::vector<int>> v4_lid_ids_;          // [rank][pg]
    std::vector<std::vector<const void*>> v4_lid_ptrs_; // [rank][pg]
    /// Per-RANK host tables ([dcp][kMaxBatch * batch_stride]) + the array of
    /// their base pointers handed to AttentionExecParams.indexer_k_pages
    /// (TD-GLM-INDEXER-DCP replicated: rank r's table holds rank r's GPU's
    /// replica pages at GLOBAL page slots; TD-GLM-INDEXER-LOCAL-MERGE local:
    /// rank r's table holds ONLY its OWNED pages — round-robin by indexer
    /// page — at LOCAL-compacted slots pg / dcp).
    std::vector<std::vector<const void*>> indexer_page_table_;
    std::vector<const void* const*> indexer_table_bases_;
    std::vector<uint8_t> indexer_computes_;        ///< layer → computes-indexer mask
    int indexer_page_stride_ = 0;                  ///< logical pages per layer row
    int indexer_batch_stride_ = 0;                 ///< n_layers * page_stride

    /// TD-GLM-INDEXER-COV: per-seq indexer-K coverage guard. Sparse scoring is
    /// only valid when EVERY position [0, len) was appended to the SAME
    /// storage by consecutive qualifying steps. Qualifying steps: decode
    /// (producer appends one key per entry) and prefill/chunked prefill
    /// (TD-GLM-INDEXER-PREFILL: the executor's chunk appender appends every
    /// chunk position — coverage advances by chunk_len). A non-appending
    /// step (graph replay, draft, pool exhaustion at B>1, position jump,
    /// unsupported prefill shape) leaves garbage at the skipped positions,
    /// and switching arena→paged mid-sequence would score never-written
    /// pages. The guard pins each sequence to one storage mode on first use
    /// and permanently downgrades it to DENSE (kDead) on any gap. Erased at
    /// seq_free.
    enum class IndexerSeqMode : uint8_t { kUnset, kPaged, kArena, kDead };
    struct IndexerCov {
        uint32_t next_pos = 0;              ///< next position that must append
        IndexerSeqMode mode = IndexerSeqMode::kUnset;
    };

    /// ── SequenceState: THE per-sequence aggregate (INV-SEQ-FORK-STATE) ───
    /// Every piece of dispatcher-owned per-sequence engine state lives HERE,
    /// in one struct keyed by seq_id — never in a parallel side map. The
    /// lifecycle handlers (dispatch_lifecycle.cpp: handle_seq_create /
    /// handle_seq_fork / handle_seq_free, plus snapshot/restore) enumerate
    /// the members through STRUCTURED BINDINGS — adding a member without
    /// updating them is a COMPILE ERROR, forcing an explicit
    /// clone-vs-exclude decision at fork and a release decision at free
    /// (this replaces the old five-parallel-maps scheme whose hand
    /// enumeration silently missed indexer_cov at fork —
    /// TD-PREFIX-FORK-COV).
    ///
    /// External per-seq state NOT owned here (wired by the same lifecycle
    /// handlers; extend the checklist when adding a family):
    ///   - kv_tiering_ (GLM cold pools): on_seq_fork / on_seq_free.
    ///   - v4_kv_tiering_ (V4 CSA cold pools): free_sequence.
    ///   - DcpExecutor V4 state rings/snapshots: v4_free_sequence.
    ///   - DSpark draft context: adoption semantics (engine-side, advisory).
    struct SequenceState {
        /// KV page handles, layer-major per logical page: handle for
        /// (logical page j, layer l) at [j * kv_layers_ + l] (TD-GOLDEN).
        /// Tracks logical sequence→page ownership for CoW-safe fork/free.
        /// V4 non-CSA layers hold SENTINEL handles (page_idx == -1,
        /// TD-V4-KMAIN-SIZING); tiering-demoted pages are neutralized the
        /// same way.
        std::vector<memory::PageHandle> kv_pages;
        /// Created via fork (PageBudgetTracker accounting).
        bool forked = false;
        /// Indexer-K pool pages, (page, computing-layer, rank)-ordered
        /// (TD-GLM-INDEXER-PAGED). Empty ⇔ never provisioned.
        std::vector<memory::PageHandle> indexer_pages;
        /// DSA indexer coverage state machine (TD-GLM-INDEXER-COV).
        /// mode == kUnset ⇔ untracked (indexer_coverage() reports -1); any
        /// dispatched step leaves it in a non-kUnset mode.
        IndexerCov indexer_cov;
        /// V4 side-tier pages (kSwa/kHca/LID; V4-7b). swa.empty() ⇔ never
        /// provisioned (ensure_v4_tier_pages resizes swa first).
        V4SeqTiers v4_tiers;
    };
    std::unordered_map<uint64_t, SequenceState> sequences_;
    SequenceState* find_seq(uint64_t seq_id) {
        auto it = sequences_.find(seq_id);
        return it == sequences_.end() ? nullptr : &it->second;
    }
    const SequenceState* find_seq(uint64_t seq_id) const {
        auto it = sequences_.find(seq_id);
        return it == sequences_.end() ? nullptr : &it->second;
    }

    /// INV-DSA-REWIND (config compute.dsa_indexer_rewind, env
    /// LS_INDEXER_REWIND overrides either way; read once at construction;
    /// default OFF = legacy guard byte-identical): bless SAME-SEQUENCE
    /// contiguous OVERWRITE-REWINDS (pos0 <= next_pos — the speculative
    /// batched-verify partial-acceptance re-feed shape) instead of kDead.
    /// Sound because every indexer-K append is POSITION-keyed (slot =
    /// seqlens_k[t] − 1 + page bias — the KV slot-mapping convention), so a
    /// re-feed overwrites its rows in place exactly as it overwrites the KV
    /// rows themselves; next_pos becomes the HIGH-WATER mark of contiguously
    /// written rows and sparse validity still comes from each step's own
    /// kv-meta seqlens bound (stale rows past the committed length are never
    /// scored). Gaps, mixed-sequence chunks, foreign sequences, and
    /// superchunk shapes stay fail-closed unchanged.
    bool indexer_rewind_ok_ = false;

    /// LS_ATTN_CHUNK_PROF=1 (read once at construction; opt-in DIAGNOSTIC,
    /// default OFF = zero overhead): per-command dispatch x-ray for
    /// chunk-shaped (is_prefill) attention commands. dispatch_attention_
    /// internal stamps enter/pre-executor/post-executor into the members
    /// below; handle_fused_compute_command wraps the dispatch in a timing-
    /// event pair on the primary kAttention stream (reaped through the
    /// generic compute_t_start/end path) and attaches a
    /// PendingCompute::AttnChunkProf, logged once at completion reap.
    bool attn_chunk_prof_ = false;
    std::chrono::steady_clock::time_point attn_prof_enter_{};
    std::chrono::steady_clock::time_point attn_prof_pre_exec_{};
    std::chrono::steady_clock::time_point attn_prof_post_exec_{};
    bool attn_prof_have_ = false;

    /// TD-GLM-INDEXER-B1CASCADE resolved (INV-DSA-ROWMIX): per-row dense
    /// mask scratch for a MIXED B>1 decode cohort (1 = coverage-dead row
    /// runs DENSE; 0 = sparse-eligible). Handed to the executor via
    /// AttentionExecParams::indexer_row_dense only when the cohort mixes
    /// both kinds; must outlive the execute_attention call (member, not a
    /// stack local).
    std::vector<uint8_t> indexer_row_dense_;

    /// TD-GOLDEN-KV-EXHAUST: specific error from the innermost dispatch helper,
    /// consumed by the command-level handlers when an internal dispatch returns
    /// false. msg == nullptr means "no specific error" (use the generic one).
    ipc::CmpErrorCategory last_internal_error_cat_ =
        ipc::CmpErrorCategory::kComputeValidation;
    const char* last_internal_error_msg_ = nullptr;

    /// TD-GOLDEN: KV cache layer count. Every attention layer needs its OWN
    /// cache rows per token — seq_pages_ holds kv_layers_ physical pages per
    /// logical page, layer-major: handle for (logical page j, layer l) lives
    /// at [j * kv_layers_ + l]. Without this, all 61 layers shared one row
    /// per token: each layer read back the LAST layer's KV for every cached
    /// position (pos 0 accidentally worked — read right after own write).
    int      kv_layers_ = 1;

    /// TD-V4-KMAIN-SIZING (INV-KV-LAYER refined for V4): per-layer kMain-page
    /// mask. Non-empty only for deepseek_v4: v4_kmain_layer_[l] != 0 iff
    /// layer l's main tier lives in the kMain (CSA) bucket. HCA/SWA layers
    /// (and the SWA-only MTP layers) keep their KV in the kv_hca/kv_swa side
    /// pools (ensure_v4_tier_pages) — their layer-major seq_pages_ slots hold
    /// SENTINEL handles (page_idx == -1, gpu_ptr == nullptr; the tiering
    /// neutralization convention) so the [j * kv_layers_ + l] indexing and
    /// block-table layout stay intact without consuming CSA-bucket pages.
    /// The V4 auto-sizer funds kMain for num_csa_layers only — allocating
    /// real pages for all layers exhausts the pool (the 759-vs-4000 prefill
    /// boundary).
    std::vector<uint8_t> v4_kmain_layer_;
    bool layer_takes_kmain_page(int l) const {
        return v4_kmain_layer_.empty()
            || (l >= 0 && l < static_cast<int>(v4_kmain_layer_.size())
                && v4_kmain_layer_[l] != 0);
    }

    /// Runtime page budget tracker — counts active sequences/forks for
    /// introspection.  Headroom policy lives in PageAllocator.
    struct PageBudgetTracker {
        int active_sequences = 0;   ///< sequences_ entries from create
        int active_forks     = 0;   ///< sequences_ entries from fork
    };
    PageBudgetTracker page_budget_;

    /// Backpressure cap from Deps (copied at construction).
    uint32_t max_inflight_compute_ = 32;

    /// KD-R2: gpu_position → index in deps_.hidden_state_pairs (-1 = non-TP).
    std::vector<int> gpu_pos_to_pair_idx_;

    /// KD-R2: contiguous attn_buf array for DcpExecutor::execute_attention().
    /// Populated from hidden_state_pairs at construction (read-only projection).
    std::vector<void*> attn_bufs_;
    // TD-PREFILL-SUPERCHUNK: per-dispatch scratch of ROW-OFFSET attn_buf
    // pointers (attn_bufs_[r] + row_offset·H·2) handed to the executor when a
    // sub-chunk targets rows past 0. Member (not a local) to avoid a per-call
    // allocation on the per-layer hot path.
    std::vector<void*> attn_bufs_offset_;
    // V4-5b mHC: per-rank collapsed embedding staging pointers (aliases of
    // moe_scratch_[pos].hc_x) for the embedding allreduce + repeat-expansion.
    std::vector<void*> hc_embed_stage_;
};

}  // namespace layerstorm::daemon
