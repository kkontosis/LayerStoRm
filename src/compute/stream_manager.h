#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/device_backend.h"

namespace layerstorm::compute {

// ── Stream IDs (INV-5a) ─────────────────────────────────────────────────────

/// Fixed stream assignment per GPU. Values match spec §5.
enum class StreamId : int {
    kAttention       = 0,  // Stream 0: Attention compute
    kExpertFfn       = 1,  // Stream 1: Expert FFN compute (dynamic dispatch)
    kGating          = 2,  // Stream 2: Gating compute
    kH2dTransfer     = 3,  // Stream 3: Host-to-device (expert weights)
    kD2hTransfer     = 4,  // Stream 4: Device-to-host (evicted experts)
    kPrefetchCompute = 5,  // Stream 5: Prefetch compute (PreScope, PROBE)
    kAsyncDequant    = 6,  // Stream 6: Async kv_bv FP8→BF16 dequant (predictive)
    kCount           = 7
};

// ── StreamManager ───────────────────────────────────────────────────────────

/// Centralized owner of all per-GPU CUDA streams and provider of event-based
/// inter-stream synchronization. Per INV-5a, manages 6 streams per GPU.
///
/// Synchronization is via CUDA events only (INV-5b). The orchestrator records
/// events when dispatching operations and polls them in the COLLECT phase.
///
/// Spec §5 synchronization patterns:
///   Event A: Attention complete → triggers gating (Stream 2) + PreScope (Stream 5)
///   Event B: Expert transfer complete → enables expert FFN (Stream 1)
///   Event C: Expert FFN complete → triggers next attention (Stream 0)
///
/// Owned and driven by the single-threaded orchestrator (INV-3.4.2).
class StreamManager {
public:
    struct Options {
        std::vector<DeviceBackend*> device_backends;  ///< One per GPU (INV-BH-1)
    };

    explicit StreamManager(Options opts);
    ~StreamManager();

    StreamManager(const StreamManager&) = delete;
    StreamManager& operator=(const StreamManager&) = delete;
    StreamManager(StreamManager&&) = delete;
    StreamManager& operator=(StreamManager&&) = delete;

    // ── Stream access ────────────────────────────────────────────────────

    /// Returns the raw stream handle for (gpu_idx, stream_id).
    /// @throws std::out_of_range on invalid gpu_idx or StreamId.
    void* stream(int gpu_idx, StreamId id) const;

    // ── Event lifecycle ──────────────────────────────────────────────────

    /// Create a new event on the specified GPU's backend. Caller owns the
    /// returned handle and must call destroy_event() when done.
    /// @throws std::out_of_range on invalid gpu_idx.
    void* create_event(int gpu_idx);

    /// Destroy an event previously created by create_event().
    /// gpu_idx must match the GPU used at creation time.
    void destroy_event(void* event, int gpu_idx);

    // ── Event operations (INV-5b: events only, no cudaStreamSynchronize) ─

    /// Record an event on a specific stream.
    /// @throws std::out_of_range on invalid gpu_idx or StreamId.
    void record_event(void* event, int gpu_idx, StreamId stream_id);

    /// Non-blocking event query on the specified GPU's backend.
    /// Returns {kReady} if complete, {kNotReady} if pending,
    /// {kError, vendor_code} on device-fatal.
    EventQueryResult query_event(void* event, int gpu_idx) const;

    /// Make a stream wait on an event. The specified stream will not execute
    /// further work until the event has completed. This is the core inter-stream
    /// dependency mechanism (maps to cudaStreamWaitEvent).
    /// @throws std::out_of_range on invalid gpu_idx or StreamId.
    void wait_event(int gpu_idx, StreamId stream_id, void* event);

    // ── Data transfer ───────────────────────────────────────────────────

    /// Async device-to-host memcpy on a specific stream (KD-2).
    /// @throws std::out_of_range on invalid gpu_idx or StreamId.
    void memcpy_d2h_async(void* host_dst, const void* device_src,
                          size_t bytes, int gpu_idx, StreamId stream_id);

    // ── Queries ──────────────────────────────────────────────────────────

    int num_gpus() const { return static_cast<int>(gpus_.size()); }

private:
    static constexpr int kStreamsPerGpu = static_cast<int>(StreamId::kCount);

    struct PerGpuStreams {
        void* streams[kStreamsPerGpu] = {};
    };

    void* get_stream(int gpu_idx, StreamId id) const;

    std::vector<PerGpuStreams> gpus_;
    std::vector<DeviceBackend*> device_backends_;
};

}  // namespace layerstorm::compute
