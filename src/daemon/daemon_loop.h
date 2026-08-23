#pragma once

// Daemon thread main loop.
//
// Extracted from Engine::daemon_loop() to enable unit testing and define
// pluggable interfaces for command dispatch (IPC-4) and state publication
// (IPC-5).  DaemonLoop is non-owning: all module pointers belong to Engine.
//
// Loop structure (per cycle):
//   1. check_shutdown()          — IpcHeader.shutdown_requested
//   2. drain_and_dispatch()      — command ring → NOOP/SHUTDOWN handled inline,
//                                  everything else forwarded to dispatch_fn
//   3. poll_transfer_completions() — TransferEngine::poll_completions() →
//                                    Completion structs → completion ring
//   4. publish_state()           — seqlock write: cycle_count + timestamp +
//                                  optional publish_fn for full snapshot
//   5. yield

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace layerstorm::ipc {
struct IpcHeader;
struct StateSnapshot;
struct Command;
template <uint32_t> class SpscRing;
using CommandRing    = SpscRing<256>;
using CompletionRing = SpscRing<128>;
}  // namespace layerstorm::ipc

#include "core/memory/nvme_tier.h"
#include "core/transfer/transfer_engine.h"
#include "daemon/expert_lifecycle_manager.h"

namespace layerstorm::daemon {

class StateTransaction;  // Forward-declare (defined in state_transaction.h)

/// Maximum commands drained per cycle.  Keeps per-cycle latency bounded
/// while still batching efficiently (~1 µs for 64 SPSC reads).
static constexpr uint32_t kMaxCommandsPerCycle = 64;

class DaemonLoop {
public:
    // ── Pluggable hooks (set by IPC-4 / IPC-5) ────────────────────────────

    /// Called for every command that is NOT CMD_NOOP or CMD_SHUTDOWN.
    /// IPC-4 will provide the real implementation.
    /// When null, unhandled commands log a warning.
    using CommandDispatchFn = std::function<void(const ipc::Command& cmd)>;

    /// Called once per cycle to populate the full StateSnapshot beyond basic
    /// fields (cycle_count, timestamp_ns).  IPC-5 provides the implementation.
    /// Callback receives a StateTransaction& and must use StateTransactionGuard
    /// for each logical update group.  See spec/IPC_TX.md.
    using StatePublishFn = std::function<void(ipc::StateSnapshot& snap,
                                              StateTransaction& tx)>;

    // ── Non-owning dependencies ────────────────────────────────────────────

    struct Deps {
        ipc::CommandRing*     cmd_ring        = nullptr;
        ipc::CompletionRing*  cmp_ring        = nullptr;
        ipc::IpcHeader*       ipc_header      = nullptr;
        ipc::StateSnapshot*   state_snapshot  = nullptr;
        std::atomic<bool>*    running         = nullptr;

        /// Nullable.  Null when running with null_backends() (no CUDA).
        transfer::TransferEngine* transfer_engine = nullptr;

        /// Nullable.  Null when NVMe tier is not configured.
        memory::NvmeTier* nvme_tier = nullptr;

        /// Nullable.  IPC-4 sets this after daemon loop construction.
        CommandDispatchFn  dispatch_fn;

        /// Nullable.  IPC-5 sets this after daemon loop construction.
        StatePublishFn     publish_fn;

        /// Resolve transfer token → originating cmd_seq (IPC-4 provides).
        /// Returns 0 for unknown tokens.
        using CmdSeqResolverFn = std::function<uint32_t(uint64_t transfer_token)>;
        CmdSeqResolverFn cmd_seq_resolver;

        /// Cleanup token mapping after completion is written (IPC-4 provides).
        using TokenCleanupFn = std::function<void(uint64_t transfer_token)>;
        TokenCleanupFn token_cleanup;

        /// NVMe IoToken → cmd_seq resolution (parallel to transfer token).
        CmdSeqResolverFn nvme_cmd_seq_resolver;
        TokenCleanupFn   nvme_token_cleanup;

        /// ELM-5: ExpertLifecycleManager for single-poll routing.  Nullable.
        ExpertLifecycleManager* elm = nullptr;

        /// Writer yield interval for StateTransaction (from config).
        int writer_yield_interval = 16;

        /// #91: Minimum interval between StateSnapshot publishes (ns).
        /// The daemon loop cycles at ~µs when spinning; publishing every
        /// cycle keeps the seqlock churning so fast that a Python reader's
        /// µs-scale transaction window can essentially never validate
        /// (INV-IPC-PUBLISH-THROTTLE — observed as a multi-second retry
        /// storm per StateReader field read, #91).  Throttling to a few ms
        /// keeps the snapshot fresh for the orchestrator's PLAN phase
        /// while leaving the seqlock quiet >90% of the time.
        /// Default 0 = publish every cycle (legacy per-cycle semantics —
        /// what the DaemonLoop unit tests assert); the ENGINE always wires
        /// this from config `orchestrator.ipc_transaction.publish_interval_us`
        /// (schema default 5000 µs), so every real engine is throttled.
        uint64_t publish_interval_ns = 0;

        /// KD-1: Poll pending compute events.  Returns count written.  Nullable.
        using PollComputeFn = std::function<uint32_t()>;
        PollComputeFn poll_compute_fn;

        /// KD-1: Query current pending compute count for backpressure.  Nullable.
        using PendingComputeCountFn = std::function<uint32_t()>;
        PendingComputeCountFn pending_compute_count_fn;

        /// KD-1: Backpressure threshold.
        uint32_t max_inflight_compute = 32;

        /// #90: Advance progressive fetch-and-run MoE state machine.
        /// Returns true if work was done.  Nullable.
        using AdvanceProgressiveFn = std::function<bool()>;
        AdvanceProgressiveFn advance_progressive_fn;

        /// M3b: low-rate background maintenance hook (arena migrator tick).
        /// Called once per cycle after the progressive-MoE advance; the
        /// callee must be near-free when it has nothing to do (the daemon
        /// cycles at ~µs, INV-3.4.2).  Nullable.
        using BackgroundFn = std::function<void()>;
        BackgroundFn background_fn;
    };

    explicit DaemonLoop(Deps deps);
    ~DaemonLoop();  // Defined in .cpp (StateTransaction is incomplete here)

    DaemonLoop(const DaemonLoop&)            = delete;
    DaemonLoop& operator=(const DaemonLoop&) = delete;
    DaemonLoop(DaemonLoop&&)                 = delete;
    DaemonLoop& operator=(DaemonLoop&&)      = delete;

    /// Blocking loop: runs until shutdown.  Called from the daemon thread.
    void run();

    /// Single iteration.  Returns false when shutdown was requested.
    /// Exposed for unit testing.
    bool run_one_cycle();

    /// ELM-5 completion pump (transfer/NVMe → ELM → completion writes +
    /// staged-transfer flush) — the sole producer of expert arrivals.
    /// Public so the dispatcher's progressive-MoE drain (queued FAR/FETCH
    /// backpressure) can make arrival progress from inside a handler.
    /// Daemon-thread only.
    void pump_completions();

    /// Monotonically increasing cycle counter.
    uint64_t cycle_count() const { return cycle_count_; }

private:
    bool     check_shutdown();
    uint32_t drain_and_dispatch();
    uint32_t poll_transfer_completions();
    uint32_t poll_nvme_completions();
    void     write_transfer_completions(std::span<const transfer::TransferCompletion> completions);
    void     write_nvme_completions(std::span<const memory::IoCompletion> completions);
    void     write_elm_completions(std::span<const LifecycleCompletion> completions);
    void     write_elm_progress(std::span<const ProgressEvent> events);
    void     publish_state();

    Deps     deps_;
    uint64_t cycle_count_ = 0;
    uint64_t last_publish_end_ns_ = 0;  // #91: publish-throttle bookkeeping
    // TD-ORCH-ELM-COMPLETION-LIVELOCK: ring-full ELM drop log throttles —
    // a producer-side livelock once wrote ~1 GB of identical error lines in
    // ~35 s. Log the first drop, then every 100000th, with a running count.
    uint64_t elm_drop_count_ = 0;
    uint64_t elm_progress_drop_count_ = 0;
    std::unique_ptr<StateTransaction> state_tx_;
    std::vector<bool> gpu_fatal_emitted_;  // CMP_GPU_FATAL once per GPU

    // Adaptive spin (fetch-latency experiment): run_one_cycle() sets this true
    // when it observed any transfer/NVMe completion this cycle. run() keeps
    // spinning (no yield) for a budget of cycles after each completion so H2D
    // completions are detected with minimal latency during active decode, then
    // degrades to yield() when quiet so an idle daemon doesn't burn a core.
    bool cycle_had_completion_ = false;
};

}  // namespace layerstorm::daemon
