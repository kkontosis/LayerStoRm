#pragma once

// ExpertLifecycleManager: unified per-expert state machine for cross-tier
// coordination.  Composes ExpertCache, NvmeTier, and TransferEngine behind
// a single ensure_resident() API.  See spec/IPC_EXPERT_LIFECYCLE.md.
//
// Thread safety: NOT thread-safe. Called exclusively from daemon thread
// (INV-ELM-1).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "core/memory/expert_cache.h"
#include "core/memory/numa_manager.h"
#include "core/memory/nvme_tier.h"
#include "core/transfer/host_source.h"
#include "core/transfer/transfer_engine.h"
#include "daemon/ipc_protocol.h"
#include "model/quantization/quant_interface.h"

namespace layerstorm::model {
struct LoadedModel;
class PrepackedSource;
class PackedBufferCache;
}

namespace layerstorm::daemon {

class StateTransaction;  // forward-declare (defined in state_transaction.h)

// ── Tier enums ──────────────────────────────────────────────────────────────

enum class HostTier : uint8_t {
    kCold,           // On NVMe or mmap only
    kLoadingToRam,   // NVMe→RAM read inflight
    kWarm,           // In host RAM warm cache (or mmap)
};

enum class GpuTier : uint8_t {
    kAbsent       = 0,  // Not on this GPU
    kReserved     = 1,  // VRAM slot allocated, no data
    kTransferring = 2,  // H2D inflight
    kDraining     = 3,  // D2H inflight before eviction (not usable for compute)
    kPartial1     = 4,  // Gate sub-component ready (forward-compat, dead per INV-ELM-6)
    kPartial1_2   = 5,  // Gate + Up ready (forward-compat)
    kHot          = 6,  // All sub-components ready
};

// ── Result types ────────────────────────────────────────────────────────────

using LifecycleToken = uint64_t;

struct ExpertState {
    HostTier host_tier;
    GpuTier  gpu_tier;
    uint32_t interest_count;
};

struct LifecycleCompletion {
    LifecycleToken    token;
    uint32_t          cmd_seq;
    memory::ExpertKey key;
    int               gpu_idx;
    bool              success;
    bool              is_eviction = false;  // ELM-6: true for kDraining→ABSENT completions
    uint32_t          nvme_us  = 0;   // NVMe I/O latency (read on load, write on evict; 0 if warm/mmap)
    uint32_t          dma_us   = 0;   // PCIe DMA latency (H2D on load, D2H on evict)
    uint32_t          total_us = 0;   // Total lifecycle latency
};

struct ProgressEvent {
    memory::ExpertKey key;
    int               gpu_idx;
    uint8_t           phase;     // 3 = h2d_started (only phase currently emitted)
    uint32_t          cmd_seq;
};

struct PollResult {
    std::vector<LifecycleCompletion> lifecycle;
    std::vector<ProgressEvent> progress;
    std::vector<transfer::TransferCompletion> unhandled_transfers;
    std::vector<memory::IoCompletion> unhandled_nvme;
};

// ── ExpertLifecycleManager ──────────────────────────────────────────────────

class ExpertLifecycleManager {
public:
    struct Deps {
        memory::ExpertCache*             expert_cache     = nullptr;
        memory::NvmeTier*                nvme_tier        = nullptr;  // nullable
        transfer::TransferEngine*        transfer_engine  = nullptr;
        memory::NumaManager*             numa_manager     = nullptr;  // nullable (ELM-6: D2H drain buffers)
        memory::PinnedExpertArena*       pinned_arena     = nullptr;  // nullable (P-24 warm DMA tier)
        memory::ArenaLoader*             arena_loader     = nullptr;  // nullable (J-1 async cold load)
        model::LoadedModel*              loaded_model     = nullptr;  // nullable
        model::PrepackedSource*          prepacked_source = nullptr;  // nullable (WP-3)
        model::PackedBufferCache*        packed_cache     = nullptr;  // nullable (WP-4)
        ipc::StateSnapshot*              snapshot         = nullptr;  // nullable (null in unit tests)
        uint32_t                         first_moe_layer  = 0;
        uint32_t                         num_moe_layers   = 0;
        uint32_t                         num_experts      = 0;
        int                              num_gpus         = 0;
        bool                             emit_progress    = false; // ELM-13: emit CMP_ELM_EXPERT_PROGRESS
        model::ExpertShape               expert_shape{};  // For lazy NVFP4 packing
    };

    explicit ExpertLifecycleManager(Deps deps);
    ~ExpertLifecycleManager();

    ExpertLifecycleManager(const ExpertLifecycleManager&) = delete;
    ExpertLifecycleManager& operator=(const ExpertLifecycleManager&) = delete;

    // ── Primary command ─────────────────────────────────────────────────

    /// Ensure expert is HOT on gpu.  Auto-chains from any tier.
    /// Returns a unique token for completion/cancel tracking.
    /// If already HOT, queues an immediate completion (returned by next poll()).
    LifecycleToken ensure_resident(memory::ExpertKey key, int gpu,
                                   memory::CacheZone zone,
                                   uint32_t cmd_seq,
                                   float priority = 0.0f,
                                   int delay_us = 0);

    /// Batched residency for a single GPU with inline make-room eviction
    /// (TD-FAR-SLOT-RESERVE-STALL). For each req: if the expert is ABSENT and its
    /// stable zone is full, evict cheapest victims from `victims` (caller-supplied,
    /// cheapest-first, already filtered to exclude this command's needed keys +
    /// locked entries) via request_evict until one frees a slot, THEN reserve+start
    /// through the normal ensure_resident path — so a full zone never drops the
    /// interest and parks ~30 ms. `victims` is consumed in order as a shared cursor
    /// across the batch (request_evict skips any still-un-evictable kTransferring /
    /// interested victim). Returns the count that secured a slot.
    struct ResidentReq {
        memory::ExpertKey key;
        memory::CacheZone zone;
    };
    int ensure_residents(int gpu, std::span<const ResidentReq> reqs,
                         std::vector<memory::ExpertKey>& victims,
                         uint32_t cmd_seq, float priority = 0.0f);

    // ── Eviction ────────────────────────────────────────────────────────

    /// Request eviction from VRAM.  Returns false if interests pending.
    bool request_evict(memory::ExpertKey key, int gpu);

    /// Initiate async D2H eviction (HOT → kDraining → ABSENT).
    /// Expert must be HOT with no pending interests.
    /// Returns true if drain was initiated, false on precondition failure.
    /// Completion arrives via poll() as LifecycleCompletion with is_eviction=true.
    bool request_drain(memory::ExpertKey key, int gpu, uint32_t cmd_seq);

    // ── Cancellation ────────────────────────────────────────────────────

    /// Cancel one interest by token.  Decrements refcount.
    /// If refcount reaches 0, cancels underlying operations.
    void cancel(LifecycleToken token);

    /// Find lifecycle token by originating cmd_seq.  Returns 0 if not found.
    /// For batch commands (shared cmd_seq), returns the first match only.
    LifecycleToken find_by_cmd_seq(uint32_t cmd_seq) const;

    /// Find all lifecycle tokens for a cmd_seq (batch-aware).
    /// Returns empty vector if not found.
    std::vector<LifecycleToken> find_all_by_cmd_seq(uint32_t cmd_seq) const;

    // ── State queries ───────────────────────────────────────────────────

    ExpertState state(memory::ExpertKey key, int gpu) const;
    HostTier host_state(memory::ExpertKey key) const;

    /// H2D source-path counters (481-2): direct = pinned source (full
    /// bandwidth, no staging copy); staged = unpinned source routed through the
    /// pinned staging pool. Used to verify the NUMA-pinned pool is effective.
    struct H2dPathStats { uint64_t direct = 0; uint64_t staged = 0; };
    H2dPathStats h2d_path_stats() const { return h2d_path_stats_; }

    /// M3b: observer invoked once per FRESH H2D enqueue (a demand fetch put
    /// expert bytes on the wire — the online-placement frequency signal;
    /// dedup joins to an already-inflight H2D do NOT fire). Daemon thread
    /// only, must be cheap. Nullable.
    void set_h2d_enqueue_observer(
        std::function<void(memory::ExpertKey)> fn) {
        h2d_enqueue_observer_ = std::move(fn);
    }

    /// M3b phase gate: the dispatcher flips this at MoE-wave issue so BIG
    /// (prefill-class) commands' fetches are NOT counted — matching the
    /// static table's freq_table.py semantics (decode+chunk counted,
    /// prefill excluded: prefill's differently-skewed hot set pollutes the
    /// EMA and the online layout plateaus below the decode optimum).
    void set_fetch_count_enabled(bool b) { fetch_count_enabled_ = b; }

    // TEMP DEBUG (TD-100j deep x-ray): per-H2D micro-timing. resolve = the
    // resolve_host_source call (incl. cold arena load_into pread); cold = count
    // with resolve > 5 ms; predma = start_ns→h2d_start_ns (queue+resolve+load);
    // dma = h2d_start_ns→completion-detected. Logged via log_h2d_micro().
    struct H2dMicro {
        uint64_t n = 0, resolve_sum = 0, resolve_max = 0, cold = 0,
                 predma_sum = 0, dma_sum = 0, dma_max = 0;
    };
    void log_h2d_micro() const;

    // ── Poll ────────────────────────────────────────────────────────────

    /// Advance state machines from pre-polled subsystem completions.
    /// ELM processes completions for experts it manages; unhandled ones
    /// are returned for legacy processing by DaemonLoop.
    PollResult poll(std::span<const transfer::TransferCompletion> transfer_completions,
                    std::span<const memory::IoCompletion> nvme_completions);

    // ── Zone management ─────────────────────────────────────────────────

    bool promote(memory::ExpertKey key, int gpu);
    bool demote(memory::ExpertKey key, int gpu);

private:
    H2dPathStats h2d_path_stats_{};
    H2dMicro h2d_micro_{};   // TEMP DEBUG (TD-100j)
    std::function<void(memory::ExpertKey)> h2d_enqueue_observer_;  // M3b
    bool fetch_count_enabled_ = true;                              // M3b

    // ── Internal types ──────────────────────────────────────────────────

    struct Interest {
        LifecycleToken    token;
        uint32_t          cmd_seq;
        memory::CacheZone zone;
    };

    struct GpuEntryKey {
        memory::ExpertKey key;
        int               gpu_idx;
        bool operator==(const GpuEntryKey&) const = default;
    };

    struct GpuEntryKeyHash {
        size_t operator()(const GpuEntryKey& k) const noexcept {
            auto h1 = std::hash<memory::ExpertKey>{}(k.key);
            auto h2 = std::hash<int>{}(k.gpu_idx);
            return h1 ^ (h2 << 32 | h2 >> 32);
        }
    };

    struct PendingGpuChain {
        int               gpu_idx;
        memory::CacheZone zone;
    };

    struct HostEntry {
        HostTier tier = HostTier::kCold;
        uint64_t nvme_token = 0;
        std::vector<PendingGpuChain> gpu_waiters;
    };

    struct GpuEntry {
        GpuTier  tier = GpuTier::kAbsent;
        uint64_t transfer_token = 0;
        std::vector<Interest> interests;
        uint64_t start_ns = 0;       // when ensure_resident first called (ABSENT→chain)
        uint64_t nvme_done_ns = 0;   // when host tier became WARM
        uint64_t h2d_start_ns = 0;   // when H2D was enqueued
        uint64_t resolve_ns = 0;     // TEMP DEBUG (TD-100j): resolve_host_source time
        float    priority = 0.0f;    // staging queue priority (#43b)
        int      delay_us = 0;       // JIT delay passthrough (#43b)
        // ELM-6: drain (D2H eviction) state
        uint32_t          drain_cmd_seq = 0;
        memory::NumaBuffer drain_host_buf{};
        uint64_t          drain_start_ns = 0;
        // TD-82a: keeps lazily-packed host buffer alive during H2D DMA.
        std::shared_ptr<std::vector<std::byte>> host_buf_ref;
    };

    // ── Internal methods ────────────────────────────────────────────────

    using HostSourceResult = transfer::HostSourceResult;

    /// Resolve host source for an expert (NvmeTier warm cache or LoadedModel mmap).
    /// May lazily pack NVFP4 expert bundles on first access.
    HostSourceResult resolve_host_source(memory::ExpertKey key,
                                         int target_gpu = -1);

    /// Release lazily-packed NVFP4 host buffer for an expert (TD-82a).
    /// No-op for FP8 experts or if owned_buf is already null.
    void release_packed_host_buffer(memory::ExpertKey key);

    /// Determine current host tier for an expert (lazy initialization).
    HostTier compute_host_tier(memory::ExpertKey key) const;

    /// Try to advance a GPU entry from RESERVED to TRANSFERRING.
    /// Returns true if H2D was enqueued. J-1: if the host source is still
    /// loading asynchronously (arena pending), the entry stays RESERVED, the
    /// (key,gpu) pair is recorded in arena_load_waiters_, and false is returned;
    /// poll() retries once the load completes.
    bool try_start_h2d(const GpuEntryKey& gk, GpuEntry& ge, HostEntry& he);

    /// J-1: reap completed async arena loads, mark slots ready, and re-drive
    /// try_start_h2d for the GPU entries waiting on each load. On load failure,
    /// fail-complete the waiting interests. Called at the top of poll().
    void process_arena_loads(std::vector<LifecycleCompletion>& out);

    /// Notify all interests on a GPU entry with a completion.
    void complete_interests(const GpuEntryKey& gk, GpuEntry& ge,
                            bool success,
                            std::vector<LifecycleCompletion>& out);

    /// Cancel all pending operations for a GPU entry and revert to ABSENT.
    void revert_gpu_entry(const GpuEntryKey& gk, GpuEntry& ge);

    // ── Snapshot publishing (ELM-8) ────────────────────────────────────

    /// Write GPU-tier + interest count to StateSnapshot for one (expert, gpu).
    void publish_gpu_state(const GpuEntryKey& gk, GpuTier tier, uint32_t interest_count);

    /// Write host-tier + bitmap bit to StateSnapshot for one expert.
    void publish_host_state(memory::ExpertKey key, HostTier tier);

    /// Initialize snapshot arrays from pre-existing ExpertCache state.
    void bootstrap_snapshot();

    // ── State ───────────────────────────────────────────────────────────

    Deps deps_;
    transfer::HostSourceDeps host_source_deps_;  // extracted from deps_ at construction

    // Snapshot (ELM-8): event-driven publication.  Nullable.
    ipc::StateSnapshot*                snapshot_ = nullptr;
    std::unique_ptr<StateTransaction>  snapshot_tx_;
    uint32_t first_moe_layer_ = 0;
    uint32_t num_moe_layers_  = 0;
    uint32_t num_experts_     = 0;
    int      num_gpus_        = 0;

    std::unordered_map<memory::ExpertKey, HostEntry> host_entries_;
    std::unordered_map<GpuEntryKey, GpuEntry, GpuEntryKeyHash> gpu_entries_;

    // J-1: GPU entries whose H2D is deferred pending an async arena load. When
    // the ArenaLoader signals the slot is filled, poll() re-drives try_start_h2d
    // for these (key,gpu) pairs. Keyed by ExpertKey because the arena slot may be
    // shared across GPUs on one NUMA node (one load serves all waiters).
    std::unordered_map<memory::ExpertKey,
                       std::vector<GpuEntryKey>> arena_load_waiters_;
    // Stage 2 (direct/mmap-free mode): GPU entries deferred because the arena was
    // momentarily full (every slot pinned mid-H2D, no mmap fallback). Unlike
    // arena_load_waiters_ these submitted NO load, so they are re-driven when a
    // slot frees (an H2D completion), not on an arena-load completion. Without
    // this they would never wake. Effectively only populated when the arena is
    // sized smaller than the in-flight waterline.
    std::vector<GpuEntryKey> slot_pressure_waiters_;
    std::unordered_map<LifecycleToken, GpuEntryKey> token_index_;
    std::unordered_multimap<uint32_t, LifecycleToken> cmd_seq_to_token_;

    // Immediate completions (for already-HOT experts).
    std::vector<LifecycleCompletion> pending_completions_;
    std::vector<ProgressEvent> pending_progress_;

    LifecycleToken next_token_ = 1;
};

}  // namespace layerstorm::daemon
