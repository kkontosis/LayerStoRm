#pragma once

// IPC-5: State publisher.
//
// Copies module state into the flat StateSnapshot struct.  Called once per
// daemon cycle.  Each sub-publish method opens its own StateTransaction
// so the seqlock is released between logical update groups.
// See spec/IPC_TX.md.

#include <cstdint>
#include <functional>

#include "daemon/ipc_protocol.h"

// Forward declarations — only used via pointer, no header includes needed.
namespace layerstorm::statistics {
class ExpertStats;
class WorkloadDetector;
class AcceptanceTracker;
}  // namespace layerstorm::statistics

namespace layerstorm::memory {
class ExpertCache;
class VramAllocator;
class PageAllocator;
}  // namespace layerstorm::memory

namespace layerstorm::transfer {
class TransferEngine;
}  // namespace layerstorm::transfer

namespace layerstorm::daemon {

class StateTransaction;

class StatePublisher {
public:
    /// Non-owning pointers to source modules.  All nullable — null modules
    /// produce zeroed snapshot sections.
    struct Deps {
        const statistics::ExpertStats*       expert_stats       = nullptr;
        statistics::WorkloadDetector*        workload_detector  = nullptr;  // non-const: shift_detected() mutates
        const statistics::AcceptanceTracker* acceptance_tracker = nullptr;
        const memory::ExpertCache*           expert_cache       = nullptr;
        const memory::VramAllocator*         vram_allocator     = nullptr;
        const memory::PageAllocator*         page_allocator     = nullptr;
        const transfer::TransferEngine*      transfer_engine    = nullptr;

        /// KD-1: Query pending compute count per GPU.  Nullable.
        using PendingComputePerGpuFn = std::function<uint32_t(uint32_t gpu_idx)>;
        PendingComputePerGpuFn pending_compute_per_gpu;
    };

    explicit StatePublisher(Deps deps);

    /// Fill all snapshot fields (except seqlock, daemon_cycle_count,
    /// timestamp_ns which are written by DaemonLoop::publish_state).
    /// Each sub-publish opens its own transaction on tx.
    void publish(ipc::StateSnapshot& snap, StateTransaction& tx);

private:
    void publish_gpu_snapshots(ipc::StateSnapshot& snap, StateTransaction& tx);
    void publish_expert_stats(ipc::StateSnapshot& snap, StateTransaction& tx);
    void publish_acceptance(ipc::StateSnapshot& snap, StateTransaction& tx);
    void publish_workload(ipc::StateSnapshot& snap, StateTransaction& tx);
    void publish_transfers(ipc::StateSnapshot& snap, StateTransaction& tx);

    Deps deps_;
    uint32_t num_moe_layers_   = 0;
    uint32_t num_experts_      = 0;
    uint32_t first_moe_layer_  = 0;
    int      num_gpus_         = 0;
};

}  // namespace layerstorm::daemon
