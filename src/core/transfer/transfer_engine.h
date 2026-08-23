#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

#include "core/gpu_ref.h"
#include "core/memory/eviction_policy.h"  // ExpertKey, std::hash<ExpertKey>

namespace layerstorm::compute {
class DeviceBackend;
}

namespace layerstorm::transfer {

// ── Types ───────────────────────────────────────────────────────────────────

using TransferToken = uint64_t;

enum class TransferDirection : uint8_t { kH2D, kD2H };

/// Completion callback. Must be fast (<1us) per INV-3.4.1.
using TransferCallback = std::function<void(TransferToken token, bool success)>;

struct TransferCompletion {
    TransferToken token{};
    memory::ExpertKey key;
    int gpu_idx{};
    TransferDirection direction{};
    int64_t bytes{};
    bool success{};
};

/// Info returned when a transfer is cancelled (for cleanup by caller).
struct CancelledTransfer {
    memory::ExpertKey key;
    int gpu_idx{};
    TransferDirection direction{};
};

/// Key for dedup map: (expert_key, gpu_idx, direction).
struct InFlightKey {
    memory::ExpertKey key;
    int gpu_idx{};
    TransferDirection dir{};

    bool operator==(const InFlightKey&) const = default;
};

}  // namespace layerstorm::transfer

// ── Hash specialization ─────────────────────────────────────────────────────

template <>
struct std::hash<layerstorm::transfer::InFlightKey> {
    size_t operator()(const layerstorm::transfer::InFlightKey& k) const noexcept {
        auto h1 = std::hash<layerstorm::memory::ExpertKey>{}(k.key);
        auto h2 = std::hash<int>{}(k.gpu_idx);
        auto h3 = std::hash<uint8_t>{}(static_cast<uint8_t>(k.dir));
        // FNV-style combine.
        h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        h1 ^= h3 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        return h1;
    }
};

namespace layerstorm::transfer {

// ── TransferEngine ──────────────────────────────────────────────────────────

/// Low-level async DMA engine for moving expert weights between host RAM and
/// GPU VRAM. Provides dedup, completion polling, and bandwidth estimation.
///
/// All expert fetching goes through this module (INV-4.5.1). The orchestrator
/// wires callbacks to ExpertCache::mark_ready(). The transfer engine is
/// format-agnostic: it takes raw (dst, src, bytes) and a key for dedup.
///
/// Owned and driven by the single-threaded orchestrator (INV-3.4.2).
class TransferEngine {
public:
    struct Options {
        std::vector<compute::DeviceBackend*> device_backends;  ///< One per GPU (INV-BH-1)
        /// Per-GPU PCIe config for bandwidth estimation. Index = gpu position.
        /// If empty, defaults to gen5 x16 for all GPUs.
        struct PcieInfo {
            int pcie_gen = 5;
            int pcie_width = 16;
        };
        std::vector<PcieInfo> pcie_info;
        int min_dispatch_per_gpu = 2;   ///< Immediate-dispatch watermark
        int max_inflight_per_gpu = 8;   ///< Hard cap on concurrent DMA per GPU
        /// N thresholds → N+1 bins. Must be sorted ascending.
        std::vector<float> priority_bin_thresholds;
    };

    explicit TransferEngine(Options opts);
    ~TransferEngine();

    TransferEngine(const TransferEngine&) = delete;
    TransferEngine& operator=(const TransferEngine&) = delete;

    // ── Enqueue a transfer ──────────────────────────────────────────────
    // Returns token if enqueued/staged, nullopt if already in-flight (dedup).
    // If inflight < min_dispatch and delay_us == 0: immediate DMA dispatch.
    // Otherwise: staged in per-GPU priority queue for flush_staged().
    // Callback fires during poll_completions() when event signals done.

    std::optional<TransferToken> enqueue_h2d(
        memory::ExpertKey key, int gpu_idx,
        void* dst, const void* src, int64_t bytes,
        float priority = 0.0f, int delay_us = 0,
        TransferCallback callback = {},
        bool needs_pinned_staging = false);

    std::optional<TransferToken> enqueue_d2h(
        memory::ExpertKey key, int gpu_idx,
        void* dst, const void* src, int64_t bytes,
        float priority = 0.0f, int delay_us = 0,
        TransferCallback callback = {});

    // ── Staging queue flush ────────────────────────────────────────────
    // Submits staged transfers up to max_inflight_per_gpu, skipping items
    // whose start_delay_us hasn't elapsed. Called each daemon cycle.
    void flush_staged();

    // ── Completion polling (non-blocking, INV-3.4.1) ────────────────────
    // Polls events, fires callbacks, removes from in-flight map.
    // After completions, auto-promotes from staging queue to maintain
    // the min_dispatch watermark (completion-driven replenishment).
    std::vector<TransferCompletion> poll_completions();

    // ── Dedup queries ───────────────────────────────────────────────────

    bool is_inflight_h2d(memory::ExpertKey key, int gpu_idx) const;
    bool is_inflight_d2h(memory::ExpertKey key, int gpu_idx) const;

    /// TD-FAR-STREAM-GATE: true iff an H2D for (key, gpu) has actually been
    /// DISPATCHED onto the GPU's h2d stream (in pending_, i.e. the DMA is
    /// stream-ordered), as opposed to merely staged in the host-side priority
    /// queue. A barrier event recorded on the h2d stream (record_h2d_barrier)
    /// only fences dispatched copies — a still-staged transfer would NOT be
    /// covered, so device-gating consumers must check this per key first.
    bool is_dispatched_h2d(memory::ExpertKey key, int gpu_idx) const;
    int inflight_count(int gpu_idx) const;
    int inflight_count() const;
    int inflight_h2d_count(int gpu_idx) const;
    int inflight_d2h_count(int gpu_idx) const;

    // ── Bandwidth estimation ────────────────────────────────────────────

    /// PCIe throughput in GB/s for given generation and width.
    static double pcie_bandwidth_gbps(int pcie_gen, int pcie_width);

    /// Estimated transfer time in microseconds for given byte count on a GPU.
    double estimate_transfer_us(int gpu_idx, int64_t bytes) const;

    // ── Staging queries ─────────────────────────────────────────────────

    int staged_count(int gpu_idx) const;
    int staged_count() const;

    // ── H2D sub-phase timing (481-3 x-ray) ──────────────────────────────
    // Aggregate cost split of H2D expert transfers, to localize the staging
    // cost: `memcpy_ns` = total CPU time in the host→pinned-staging `std::memcpy`
    // (0 for direct/pinned sources that skip staging); `dma_wait_ns` = total time
    // from DMA enqueue to completion-detected in poll (DMA execution + queue +
    // poll latency); `count` = H2D transfers completed. Monotonic; snapshot-diff
    // around a run. Benign cross-thread read (controlling thread vs daemon).
    struct H2dPhaseStats {
        uint64_t memcpy_ns   = 0;
        uint64_t dma_wait_ns = 0;
        uint64_t count       = 0;
        // perf_trace pure-DMA x-ray: Σ cudaEventElapsedTime(start,end) over H2D
        // transfers = GPU-measured memcpy time only (excludes host poll/detect
        // lag and pre-start stream backlog). dma_wait_ns − dma_gpu_ns ≈ host
        // detection/queue overhead. Accumulated only when perf_trace is enabled;
        // gpu_count tracks how many transfers contributed.
        uint64_t dma_gpu_ns  = 0;
        uint64_t gpu_count   = 0;
    };
    H2dPhaseStats h2d_phase_stats() const { return h2d_phase_stats_; }

    // ── Priority bin counts ─────────────────────────────────────────────

    /// Counts all pending transfers (staged + dispatched) on a GPU,
    /// classified into bins by priority using the configured thresholds.
    /// Returns vector of size priority_bin_thresholds.size() + 1.
    std::vector<int> pending_bin_counts(int gpu_idx) const;

    // ── H2D stream barrier (TD-FAR-STREAM-GATE) ────────────────────────

    /// Record a "barrier" event on a GPU's H2D stream and return it. The event
    /// is the stream-ordered no-op/doorbell marker: it fires only after every
    /// H2D copy ALREADY dispatched onto that stream has completed, so a
    /// consumer stream can `stream_wait_event` on it to order compute after
    /// the copies entirely device-side (zero host completion polling for that
    /// edge — the TRT-LLM / llama.cpp copy→compute stream-gate pattern).
    /// Covers only DISPATCHED copies (see is_dispatched_h2d); staged
    /// transfers flushed after this call are NOT fenced.
    /// Caller owns the event and must destroy it via the SAME GPU's
    /// DeviceBackend::destroy_event (safe immediately after the wait is
    /// enqueued — CUDA/HIP retain the dependency). Returns nullptr on a bad
    /// gpu_idx. Uses only DeviceBackend methods (INV-GPU-1).
    void* record_h2d_barrier(int gpu_idx);

    // ── Cancellation ───────────────────────────────────────────────────

    /// Cancel a transfer by token.  Checks staged queue first (O(1)
    /// metadata removal, no orphaned DMA), then in-flight pending map.
    /// Returns cancelled transfer info for cleanup, or nullopt if not found.
    std::optional<CancelledTransfer> cancel(TransferToken token);

    // ── Priority re-assertion (ELB: demand joins a staged prefetch) ─────
    /// RAISE the priority of a STILL-STAGED transfer (never lowers). When a
    /// demand fetch joins an in-flight interest whose H2D was enqueued at
    /// speculative-prefetch priority, the staged entry must not keep riding
    /// the low priority behind other prefetches (kDemandFetchPriority intent:
    /// demand outranks any speculative prefetch in the staged admission
    /// queue). Rebuilds the owning GPU's staged priority queue (bounded by
    /// its staged depth). Returns true if a staged entry was updated; false
    /// if the transfer is already dispatched (DMA is stream-ordered and
    /// cannot be reordered) or the token is unknown.
    bool reprioritize(TransferToken token, float priority);

    // ── Pinned staging pool (WP-7b) ──────────────────────────────────

    /// Initialize the pinned staging buffer pool.  Called after
    /// PrepackedSource is created (slot_size comes from the manifest).
    /// Uses `backend` for host_alloc_pinned.  No-op if already initialized
    /// or if count == 0.
    void init_pinned_pool(size_t buffer_size, int count,
                          compute::DeviceBackend* backend);

    /// Number of free pinned staging slots.
    int pinned_pool_free() const;

    /// Total pinned staging slots.
    int pinned_pool_size() const;

    // ── Shutdown ────────────────────────────────────────────────────────

    /// Drain all in-flight transfers (blocking). Called during shutdown.
    void drain();

private:
    struct PerGpuState {
        void* h2d_stream = nullptr;
        void* d2h_stream = nullptr;
        int pcie_gen = 5;
        int pcie_width = 16;
    };

    struct PendingTransfer {
        TransferToken token{};
        memory::ExpertKey key;
        int gpu_idx{};
        TransferDirection direction{};
        int64_t bytes{};
        float priority{};
        void* event = nullptr;
        TransferCallback callback;
        int pinned_slot_idx = -1;  ///< Index into pinned_pool_ (-1 = none). WP-7b.
        bool cancelled = false;    ///< Deferred cleanup: drain in poll_completions.
        uint64_t dma_enqueue_ns = 0;  ///< 481-3: steady_clock at DMA enqueue (H2D x-ray).
        // perf_trace pure-DMA x-ray (profiling-only; null unless tracing): a pair
        // of timing-enabled events bracketing the memcpy on the stream, so
        // cudaEventElapsedTime(t_start,t_end) yields the GPU memcpy time alone.
        void* t_start = nullptr;
        void* t_end   = nullptr;
        int   src_numa = -1;  ///< perf_trace: NUMA node of the DMA source (local
                              ///< vs cross-NUMA classification). -1 unless tracing.
    };

    struct StagedTransfer {
        TransferToken token{};
        memory::ExpertKey key;
        int gpu_idx{};
        TransferDirection direction{};
        void* dst = nullptr;
        const void* src = nullptr;
        int64_t bytes{};
        float priority{};
        int delay_us{};
        int64_t enqueue_time_us{};
        TransferCallback callback;
        bool needs_pinned_staging = false;  ///< WP-7b: acquire pinned slot at dispatch.

        bool operator<(const StagedTransfer& o) const {
            return priority < o.priority;
        }
    };

    std::optional<TransferToken> enqueue_impl(
        TransferDirection dir, memory::ExpertKey key, int gpu_idx,
        void* dst, const void* src, int64_t bytes,
        float priority, int delay_us,
        TransferCallback callback,
        bool needs_pinned_staging = false);

    void dispatch_staged(StagedTransfer& st);
    void replenish_gpu(int gpu_idx);
    int bin_for(float priority) const;

    // ── Pinned staging pool (WP-7b) ────────────────────────────────
    struct PinnedStagingSlot {
        void* data = nullptr;
        size_t size = 0;
        bool in_use = false;
    };
    int acquire_pinned_slot();
    void release_pinned_slot(int idx);

    std::vector<PerGpuState> gpus_;
    std::unordered_map<InFlightKey, TransferToken> inflight_;
    std::unordered_map<TransferToken, PendingTransfer> pending_;
    std::vector<std::priority_queue<StagedTransfer>> staged_;
    std::unordered_map<TransferToken, int> staged_token_gpu_;
    TransferToken next_token_ = 1;
    std::vector<compute::DeviceBackend*> device_backends_;
    int min_dispatch_per_gpu_{};
    int max_inflight_per_gpu_{};
    std::vector<float> priority_bin_thresholds_;
    std::vector<std::vector<int>> staged_bin_counts_;  // [gpu_idx][bin]

    // WP-7b: pinned staging pool
    std::vector<PinnedStagingSlot> pinned_pool_;
    compute::DeviceBackend* pinned_pool_backend_ = nullptr;

    // 481-3: H2D sub-phase timing accumulators (see h2d_phase_stats()).
    H2dPhaseStats h2d_phase_stats_{};
};

}  // namespace layerstorm::transfer
