#pragma once

// Engine module: single entry point for the Python orchestrator.
//
// start_engine(config_path) initializes all Phase 1-9 C++ modules, allocates
// the IPC region (SPSC rings + StateSnapshot), spawns a daemon thread, and
// returns EngineInfo with pointers + model metadata.
//
// stop_engine() signals shutdown, joins the daemon thread, and frees resources.
//
// Singleton: only one Engine per process.

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "model/quantization/gguf_kquant.h"  // GgufQuantInterface (owned member)

#include "daemon/buffer_registry.h"
#include "daemon/command_dispatcher.h"
#include "daemon/daemon_loop.h"
#include "daemon/ipc_protocol.h"
#include "daemon/spsc_ring.h"
#include "daemon/state_publisher.h"

// Forward declarations for DCP
#include "parallelism/collective_backend.h"
#include "parallelism/dcp_executor.h"
#include "core/gpu_loader/loader_constants.h"

// Forward declarations for types only used via unique_ptr.
namespace layerstorm {

namespace config { struct Config; }

namespace model {
class ModelConfig;
class QuantInterface;
class LayerRegistry;
class PrepackedSource;
class PackedBufferCache;
struct LoadedModel;
struct PrepackResult;
}  // namespace model

namespace memory {
class ExpertCache;
class NumaManager;
class NvmeTier;
class PinnedExpertArena;
class ArenaLoader;
class ArenaMigrator;
class ArenaIpcClient;
class ArenaCache;
class ArenaMetaSegment;
}  // namespace memory

namespace statistics {
class ExpertStats;
class CoactivationGraph;
class WorkloadDetector;
class AcceptanceTracker;
}  // namespace statistics

namespace compute {
class DeviceBackend;
class DcpAttentionWrapper;
class ExpertDevice;
class StreamManager;
struct TqResources;
}  // namespace compute

namespace transfer {
class TransferEngine;
}  // namespace transfer

namespace speculation {
class SpeculationMethod;
class DsparkRuntime;  // DSP-3
}  // namespace speculation

}  // namespace layerstorm

namespace layerstorm::daemon {

// ── Backend overrides for testability ───────────────────────────────────────

/// Groups all pluggable backends.  Production code uses default_backends();
/// unit tests use null_backends() (heap memory, no-op CUDA, no weight loading).
struct EngineBackends {
    bool skip_hardware_detection = false;  // true: skip resolve_config()
    bool skip_weight_loading     = false;  // true: skip load_weights()
    bool skip_cuda_graphs        = false;  // true: skip CUDA graph capture
};

/// Real CUDA backends (default for production).
EngineBackends default_backends();

/// All null/heap backends with all skip flags true (for unit tests without CUDA).
EngineBackends null_backends();

// ── Engine ──────────────────────────────────────────────────────────────────

class Engine {
public:
    /// Construct and fully initialize the engine.
    /// Throws std::runtime_error on any init failure.
    Engine(const std::string& config_path, EngineBackends backends);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    /// Return the EngineInfo struct (populated at init).
    const ipc::EngineInfo& info() const { return engine_info_; }

    /// Return the parsed config.
    const config::Config& config() const { return *cfg_; }

    /// Return the buffer registry (IPC-6).
    const BufferRegistry* buffer_registry() const { return buffer_registry_.get(); }

    /// Return the expert cache (per-GPU two-zone slot allocator). Used by tests
    /// to query per-GPU/zone slot capacity for a test-side LRU residency model.
    const memory::ExpertCache* expert_cache() const { return expert_cache_.get(); }

    /// Return the VRAM allocator (per-GPU region byte budgets + layout). Used by
    /// tests/reports to break down TP VRAM consumption per region.
    const memory::VramAllocator* vram_allocator() const { return vram_allocator_.get(); }

    /// Return the pinned host expert arena (nullptr when disabled). Used by
    /// the test-side REEF stack for the shared epoch-latched bank seam
    /// (INV-REEF-BANK, gl::install_arena_bank_seam).
    const memory::PinnedExpertArena* pinned_arena() const {
        return pinned_arena_.get();
    }

    /// Return the NUMA manager (nullptr before init). Used by the test-side
    /// REEF stack for HBM→CPU-affinity bank pairing (INV-REEF-BANK).
    const memory::NumaManager* numa_manager() const {
        return numa_manager_.get();
    }

    /// TD-SERVE-NAMED-TOOL-CHOICE / TD-ORCH-SAMPLED-SPEC: host address
    /// (+ bytes) of the pinned full-logits readback region (CMD_OUTPUT_HEAD
    /// readback_logits=1 target; ipc::kMaxLogitsReadbackRows rows — row 0
    /// serves the B=1 guided single-row use, the speculative sampled /
    /// logprobs verify chunk reads the first R rows).  0 when unavailable.
    /// Single-process: the Python orchestrator reads it directly after the
    /// completion; rows = bytes / (vocab_size * 4).
    uint64_t logits_readback_addr() const;
    uint64_t logits_readback_bytes() const;

    /// Return the configured SpeculationMethod (SPEC-SCAFFOLD seam), or
    /// nullptr when speculation.method == none (DEFAULT). Future dispatch
    /// (#16 MTP draft + verify passes) hangs off this.
    speculation::SpeculationMethod* speculation_method() const {
        return speculation_method_.get();
    }

    /// Return the DSpark DFlash backbone runtime (DSP-3), or nullptr when
    /// speculation.method != dspark.  Tests read the tracked-context state
    /// and the base_logits/hidden_out device pointers through this.
    speculation::DsparkRuntime* dspark_runtime() const {
        return dspark_runtime_.get();
    }

    /// H2D source-path counters (481-2): {direct, staged}. `direct` counts
    /// expert H2D copies sourced from a NUMA-pinned host pool (full bandwidth);
    /// `staged` counts copies routed through the pinned staging pool. Exposed for
    /// benchmarks comparing memory.pin_host_expert_pool on/off. Returns {0,0} when
    /// the ELM is absent. NOTE: read from the controlling thread is a benign race
    /// against the daemon's monotonic increments — intended as an end-of-run snapshot.
    ipc::EngineH2dPathStats h2d_path_stats() const;

    /// Run offline expert pre-processing and return.
    /// Lightweight init: config + model config + quant + weight loading only.
    /// No GPU/CUDA/IPC initialization, no daemon thread.
    /// @param config_path  Path to config JSON file.
    /// @param output_dir   Target directory for pre-processed files.
    /// @return PrepackResult (from expert_prepacker.h).
    static model::PrepackResult run_prepack(
        const std::string& config_path,
        const std::filesystem::path& output_dir);

    /// Signal shutdown and join daemon thread.
    void shutdown();

    bool is_running() const { return running_.load(std::memory_order_acquire); }

private:
    void init_modules();
    void allocate_ipc_region();
    void spawn_daemon_thread();
    void upload_pinned_weights();
    void quantize_attention_weights();
    void wire_kv_bv_dequant();

    // ── Init args ──
    std::string config_path_;
    EngineBackends backends_;

    // ── Per-GPU DeviceBackend instances (#86a) ──
    // Must be declared before subsystem members that hold raw pointers
    // to these (reverse destruction order keeps pointers valid).
    std::vector<std::unique_ptr<compute::DeviceBackend>> device_backends_;

    // ── Config (step 1) ──
    std::unique_ptr<config::Config> cfg_;

    // ── Model metadata (steps 2-4) ──
    std::unique_ptr<model::ModelConfig>  model_cfg_;
    const model::QuantInterface*         quant_ = nullptr;  // non-owning (registry, or owned_gguf_quant_)
    // Owned expert QuantInterface for the generic `gguf` weight type: GGUF is
    // per-tensor mixed/data-dependent, so the registry sentinel cannot size it.
    // Built from the file's per-projection k-quant types (pre-scanned at init);
    // quant_ points at this when weights == gguf (generic). nullopt otherwise.
    std::optional<model::GgufQuantInterface> owned_gguf_quant_;
    std::unique_ptr<model::LayerRegistry> layer_registry_;

    // ── Memory (steps 5-10) ──
    std::unique_ptr<memory::VramAllocator>  vram_allocator_;
    std::unique_ptr<memory::PageAllocator>  page_allocator_;
    std::unique_ptr<memory::ExpertCache>    expert_cache_;
    std::unique_ptr<memory::NumaManager>    numa_manager_;
    gpu_loader::LoaderConstants             loader_constants_;  // I8b (filled iff gpu_loader.enabled)
    std::unique_ptr<memory::NvmeTier>       nvme_tier_;         // optional
    // P-24b: persistent-arena attachment (memory.arena_attach). The client's
    // open socket IS the attachment — declared BEFORE pinned_arena_ so it is
    // destroyed AFTER the arena (unregister/unmap first, then detach; the
    // holder keeps the memfds either way). arena_meta_/arena_cache_ likewise
    // precede pinned_arena_ (the arena's cache hooks die before the cache).
    std::unique_ptr<memory::ArenaIpcClient>   arena_ipc_;       // optional
    std::unique_ptr<memory::ArenaMetaSegment> arena_meta_;      // optional
    std::unique_ptr<memory::ArenaCache>       arena_cache_;     // optional
    // P-24b lever-1: early arena-attach worker — overlaps the holder attach,
    // warm-adopt + (warm-only) registration with rank init / GGUF load.
    // Launched before attention-device init; joined at the step-18d arena
    // block (and defensively in shutdown()). The worker writes arena_ipc_/
    // arena_meta_/arena_cache_/pinned_arena_; the main thread reads them only
    // after join. Cold boots do NOT register on the worker (registration
    // stays overlapped with the NVMe preload, TD-INIT-OVERLAP).
    struct ArenaEarlyState {
        bool launched = false;
        bool warm_ready = false;     ///< worker built+registered a kAdopt arena
        bool cold_attached = false;  ///< attached; empty/mismatch → late cold build
        uint64_t geom_hash = 0;
        uint64_t source_id = 0;
        uint64_t placement_id = 0;   ///< arena host placement identity (0 = off);
                                     ///< read from env on the MAIN thread
                                     ///< (fail-loud) before the worker spawns
        size_t slot_bytes = 0;       ///< manifest stride (must match PrepackedSource)
        std::vector<std::pair<int, int>> nodeids;  ///< (node, share_degree) sorted
        std::unordered_map<int, int> spill_weight;
        std::unordered_map<int, int> node_share;
        std::exception_ptr fatal;    ///< BUSY — rethrown at join (engine-fatal)
        // Round 2b: cudaHostRegister holds driver locks that STALL the main
        // thread's CUDA calls — so the warm-path registration must overlap
        // CPU-only work (the GGUF load), not rank init. The worker finishes
        // adopt/build, then waits for this flag (set right before
        // load_weights) before registering.
        std::atomic<bool> start_register{false};
        std::thread thread;
    };
    ArenaEarlyState arena_early_;
    void launch_arena_attach_early_();
    void arena_attach_early_worker_();
    // P-24: per-NUMA-node pinned expert arena (slab). Built only when
    // memory.pin_host_expert_pool is on; it is the warm DMA tier for host_source.
    std::unique_ptr<memory::PinnedExpertArena> pinned_arena_;   // optional
    // J-1: async cold-load worker pool. Built alongside pinned_arena_; fills
    // arena slots off the daemon thread so run_one_cycle never blocks on the
    // file→slot copy (INV-3.4.2). Null when the arena is off (sync fallback).
    std::unique_ptr<memory::ArenaLoader> arena_loader_;         // optional

    // ── Transfer + compute (steps 11-14) ──
    std::unique_ptr<transfer::TransferEngine> transfer_engine_;
    std::unique_ptr<compute::StreamManager>   stream_manager_;
    std::unique_ptr<compute::GraphRegistry>   graph_registry_;
    std::unique_ptr<daemon::BufferRegistry>   buffer_registry_;

    // ── Expert devices (all GPUs, INV-BH-5) ──
    std::vector<std::unique_ptr<compute::ExpertDevice>> expert_devices_;

    // ── DCP (steps 14-15, optional: TP >= 2) ──
    std::unique_ptr<parallelism::CollectiveBackend> collective_backend_;
    std::unique_ptr<parallelism::DcpCommunicator> dcp_communicator_;
    std::vector<std::unique_ptr<compute::AttentionDevice>> attention_devices_;
    // INV-TQ-PERRANK (TD-KVT-TQ-GOLDEN root-cause): TQ codebook + Pi device
    // pointers are consumed by EVERY TP rank's kernels — one TqResources per
    // attention device, allocated on THAT rank's GPU.  A single shared
    // instance put rank-0 pointers under rank-1 kernels (illegal access on
    // the first TQ k_append).  Contents are identical across ranks (INV-TQ-4
    // deterministic seeds; same codebook JSON).
    std::vector<std::unique_ptr<compute::TqResources>> tq_resources_;
    std::unique_ptr<compute::DcpAttentionWrapper> dcp_attention_wrapper_;  // TD-74i
    std::unique_ptr<parallelism::DcpExecutor>     dcp_executor_;

    // ── Statistics (step 16) ──
    std::unique_ptr<statistics::ExpertStats>       expert_stats_;
    std::unique_ptr<statistics::CoactivationGraph> coactivation_graph_;
    std::unique_ptr<statistics::WorkloadDetector>  workload_detector_;
    std::unique_ptr<statistics::AcceptanceTracker> acceptance_tracker_;

    // ── Weights (step 18, optional) ──
    std::unique_ptr<model::LoadedModel> loaded_model_;

    // ── Speculation method seam (step 19d, SPEC-SCAFFOLD) ──
    // Constructed only when speculation.method != none (DEFAULT none → null,
    // zero behavior change).  #16 / GLM-25g drops the concrete MTP method in
    // here.  Declared AFTER attention_devices_ / expert_devices_ /
    // loaded_model_: the method holds non-owning pointers into them
    // (reverse destruction order keeps them valid); also reset explicitly
    // in shutdown() before device teardown.
    std::unique_ptr<speculation::SpeculationMethod> speculation_method_;

    // ── DSpark DFlash backbone runtime (DSP-3, optional) ──
    // Constructed only when speculation.method == dspark: loads + uploads the
    // draft checkpoint onto the draft GPU and arms the aux-hidden export in
    // the CommandDispatcher (Deps.dspark).  Declared after device_backends_ /
    // stream_manager_ (holds non-owning pointers into both) and reset in
    // shutdown() after command_dispatcher_ (which references it).
    std::unique_ptr<speculation::DsparkRuntime> dspark_runtime_;
    // Dedicated draft streams when a draft rank hosts on a TP GPU
    // (TD-DSPARK-DRAFT-QUANT 5090 placement; one per sharded rank under
    // TD-DSPARK-DRAFT-SHARD): the kAttention stream carries target
    // attention there, so the draft pipeline gets its own stream(s).
    // Owned; destroyed in shutdown() after dspark_runtime_ is reset.
    std::vector<std::pair<compute::DeviceBackend*, void*>>
        dspark_draft_streams_;

    // ── Pre-processed source (WP-3, optional) ──
    std::unique_ptr<model::PrepackedSource> prepacked_source_;

    // M3b: online self-tuning arena placement (LS_ARENA_PLACE_ONLINE=1,
    // default OFF). Declared AFTER pinned_arena_/prepacked_source_ so it is
    // destroyed FIRST (it holds references to both; its private loader
    // worker joins in its dtor). Ticked from the daemon loop background
    // hook; fed by the ELM fresh-H2D-enqueue observer.
    std::unique_ptr<memory::ArenaMigrator> arena_migrator_;

    // TD-ARENA-MIGRATE-EMA-PERSIST: writes the migrator's fetch EMA into the
    // ArenaCache meta trailer. Wired only when BOTH the online migrator and
    // the holder-backed cache exist; invoked periodically from the migrator
    // tick (daemon thread) and once more in shutdown() after daemon join.
    std::function<void()> ema_persist_fn_;

    // ── Packed buffer cache (WP-4, optional) ──
    std::unique_ptr<model::PackedBufferCache> packed_cache_;

    // ── Weight device pointers (KD-3d: populated by upload_pinned_weights) ──
    std::vector<const void*>  embedding_table_ptrs_;
    std::vector<const void*>  output_head_weight_ptrs_;
    std::vector<const float*> output_head_bias_ptrs_;
    std::vector<std::vector<const float*>> gating_bias_ptrs_;  // [layer][num_gpus]
    // V4-4: hash-layer token-id→expert tables (ffn_gate_tid2eid I32,
    // [num_experts_per_tok, vocab] per hash layer). Non-hash layers nullptr.
    std::vector<std::vector<const int32_t*>> hash_gating_table_ptrs_;  // [layer][num_gpus]
    std::vector<std::vector<parallelism::AttentionLayerWeights>> per_layer_attn_weights_;

    // KD-3e: router projection + shared expert weight device pointers.
    std::vector<std::vector<const void*>> router_weight_ptrs_;  // [layer][gpu]
    std::vector<std::vector<CommandDispatcher::Deps::SharedExpertWeights>>
        shared_expert_weight_ptrs_;  // [layer][gpu]
    std::vector<std::vector<CommandDispatcher::Deps::DenseFFNWeights>>
        dense_ffn_weight_ptrs_;  // [layer][gpu]

    // KD-3d-fix: final norm + post_attention_layernorm weight pointers.
    std::vector<const void*> final_norm_ptrs_;  // [num_gpus] per position
    // V4-5b mHC output collapse weights (F32, replicated per position).
    std::vector<const void*> output_hc_fn_ptrs_;     // [num_gpus]
    std::vector<const void*> output_hc_base_ptrs_;   // [num_gpus]
    std::vector<const void*> output_hc_scale_ptrs_;  // [num_gpus]
    // V4-5b mHC attention-stage scratch, rank-indexed (see Deps::hc_attn_*).
    std::vector<void*> hc_attn_x_;
    std::vector<void*> hc_attn_post_;
    std::vector<void*> hc_attn_comb_;

    // KD-4f-b: MTP-specific weight pointers [num_mtp][num_gpus]
    std::vector<std::vector<const void*>> mtp_embed_tokens_ptrs_;
    std::vector<std::vector<const void*>> mtp_shared_head_weight_ptrs_;
    std::vector<std::vector<const void*>> mtp_shared_head_norm_ptrs_;
    std::vector<std::vector<const void*>> mtp_eh_proj_ptrs_;
    std::vector<std::vector<const void*>> mtp_enorm_ptrs_;
    std::vector<std::vector<const void*>> mtp_hnorm_ptrs_;
    // #16: eh_proj GGUF metadata (parallel to mtp_eh_proj_ptrs_) — GGUF
    // checkpoints keep eh_proj packed (Q8_0 on GLM-5.2); the dispatcher
    // routes the projection GEMM by these.
    std::vector<std::vector<uint8_t>> mtp_eh_proj_is_gguf_;
    std::vector<std::vector<model::GgufKQuantType>> mtp_eh_proj_gguf_type_;

    // KD-4f-c3: FP8 weight + scale buffers for online-quantized attention projections.
    struct QuantizedAttnAlloc {
        void* fp8_weight = nullptr;
        void* scales = nullptr;
        int rank = -1;  // for device_free via attention_devices_[rank]
    };
    std::vector<QuantizedAttnAlloc> quantized_attn_allocs_;

    // KD-R2: paired hidden state buffers (owned by Engine, freed in shutdown).
    std::vector<HiddenStatePair> hidden_state_pairs_;           // [tp] per rank
    std::vector<void*>           fused_moe_hidden_state_bufs_;  // non-TP GPUs only

    // ── IPC region ──
    std::unique_ptr<uint8_t, void(*)(void*)> ipc_region_{nullptr, nullptr};
    uint64_t                ipc_total_bytes_ = 0;
    bool                    ipc_region_registered_ = false;  ///< compute.ipc_pin
    ipc::IpcHeader*         ipc_header_      = nullptr;
    ipc::StateSnapshot*     state_snapshot_  = nullptr;
    ipc::EngineInfo         engine_info_{};

    // ── Expert lifecycle (ELM-8) ──
    std::unique_ptr<ExpertLifecycleManager> elm_;

    // ── Daemon thread ──
    std::unique_ptr<ipc::CommandRing>    cmd_ring_;
    std::unique_ptr<ipc::CompletionRing> cmp_ring_;
    std::unique_ptr<CommandDispatcher>   command_dispatcher_;
    std::unique_ptr<StatePublisher>     state_publisher_;
    std::unique_ptr<DaemonLoop>          daemon_loop_impl_;
    std::thread        daemon_thread_;
    std::atomic<bool>  running_{false};
};

// ── Free functions (singleton) ──────────────────────────────────────────────

ipc::EngineInfo start_engine(const std::string& config_path);
ipc::EngineInfo start_engine(const std::string& config_path, EngineBackends backends);
void stop_engine();

/// Access the singleton engine (for pybind11 bindings). Returns nullptr if not running.
Engine* get_engine();

}  // namespace layerstorm::daemon
