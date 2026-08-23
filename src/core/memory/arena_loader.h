#pragma once

// ArenaLoader (J-1) — small async worker pool that fills pinned arena slots off
// the daemon thread, so a cold expert read (PrepackedSource mmap → pinned slot
// copy, ~150 ms when page-fault-bound, TD-100j) does NOT block run_one_cycle()
// (INV-3.4.2). On an arena miss the daemon reserves a slot, marks it `loading`,
// and submits the file→slot read here; the worker runs the copy and pushes a
// completion the daemon reaps on a later poll (then mark_ready + enqueue H2D).
//
// Two backends:
//   kWorkerPool (J-1, default): a thread pool runs the synchronous
//     PrepackedSource::load_into()/pread_into() off the daemon thread. Concurrency
//     = num_workers. Works for mmap-memcpy and buffered-pread reads.
//   kIoUring (Stage 3, TD-J-1-b): submit an io_uring_prep_read straight from the
//     prepacked file (PrepackedSource direct-mode fd) into the pinned slot; reap
//     CQEs on poll_completed(). NO worker threads — the kernel does the async I/O,
//     so concurrency = queue depth (≫ a thread pool) and a fetch call with many
//     misses overlaps them all. Requires PrepackedSource direct mode (open fds)
//     and a liburing build (LAYERSTORM_HAS_URING); falls back to kWorkerPool
//     otherwise.
//
// Threading: submit() / poll_completed() / busy() / pending_count() are called
// ONLY from the daemon thread (INV-ELM-1). In kWorkerPool the job/done queues are
// mutex-guarded so worker threads can hand results back; in kIoUring the ring is
// daemon-thread-only. No CUDA here (the slot is plain pinned host memory).

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/memory/eviction_policy.h"  // ExpertKey

struct io_uring;  // forward decl (liburing C struct); ring lives in the .cpp

namespace layerstorm::model { class PrepackedSource; }

namespace layerstorm::memory {

/// Result of one async arena slot fill.
struct ArenaLoadCompletion {
    ExpertKey key;
    int       gpu = -1;     ///< computing GPU (consumer-local arena, INV-MoE-NUMA)
    void*     dst = nullptr; ///< the slot that was filled (for diagnostics)
    bool      success = false;
};

class ArenaLoader {
public:
    enum class Backend { kWorkerPool, kIoUring };

    /// kWorkerPool: spawn `num_workers` threads (clamped to ≥1, default min(4,hw)).
    /// kIoUring: init an io_uring of `queue_depth` (default 64), no worker threads;
    /// falls back to kWorkerPool if built without liburing (LAYERSTORM_HAS_URING).
    explicit ArenaLoader(int num_workers = 0,
                         Backend backend = Backend::kWorkerPool,
                         int queue_depth = 64);

    /// Joins workers / tears down the ring (shutdown only — never on the hot path).
    ~ArenaLoader();

    ArenaLoader(const ArenaLoader&) = delete;
    ArenaLoader& operator=(const ArenaLoader&) = delete;
    ArenaLoader(ArenaLoader&&) = delete;
    ArenaLoader& operator=(ArenaLoader&&) = delete;

    /// Submit an async copy of `key`'s packed bytes from `src` into `dst` (a
    /// pre-pinned, pre-reserved arena slot). The slot MUST stay reserved +
    /// `loading` until the matching completion is reaped (the arena's pick_slot
    /// skips loading slots, so it won't be reused). `gpu` is echoed back in the
    /// completion. No-op (returns false) if `key` is already in flight here
    /// (dedup) — a slot is filled at most once per outstanding load.
    bool submit(ExpertKey key, int gpu, void* dst, const model::PrepackedSource* src);

    /// Drain finished loads into `out` (appends; daemon thread only). Non-blocking.
    void poll_completed(std::vector<ArenaLoadCompletion>& out);

    /// Like poll_completed but, in io_uring mode, BLOCKS until at least one CQE is
    /// available, then drains all ready completions. Used only by bulk preload
    /// draining, where the runtime's non-blocking poll would busy-spin a core
    /// while the NVMe reads are in flight. The caller must guarantee at least one
    /// read is outstanding (else io_uring would block forever). In worker-pool /
    /// non-URING builds it degrades to a non-blocking poll_completed (preload only
    /// takes the async path when backend()==kIoUring).
    void wait_completed(std::vector<ArenaLoadCompletion>& out);

    /// True if a load for `key` is currently queued or running.
    bool busy(ExpertKey key) const;

    /// Number of loads queued or running (for diagnostics / tests).
    size_t pending_count() const;

    /// Number of worker threads (0 in io_uring mode).
    int num_workers() const { return static_cast<int>(workers_.size()); }

    /// The active backend (may differ from requested if liburing is unavailable).
    Backend backend() const { return backend_; }

private:
    struct Job {
        ExpertKey key;
        int       gpu = -1;
        void*     dst = nullptr;
        const model::PrepackedSource* src = nullptr;
    };

    void worker_main();

    Backend backend_ = Backend::kWorkerPool;

    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::queue<Job>         jobs_;
    std::vector<ArenaLoadCompletion> done_;
    std::unordered_set<ExpertKey> inflight_;   ///< queued|running (dedup)
    bool                    stop_ = false;

    std::vector<std::thread> workers_;

    // io_uring backend (Stage 3). ring_ is null in worker-pool mode / non-URING
    // builds. Daemon-thread-only (still guarded by mu_ for uniformity with
    // busy()/pending_count()). uring_inflight_ maps the SQE user_data token to the
    // pending completion (key/gpu/dst) awaiting its CQE.
    struct UringPending {
        ExpertKey key;
        int       gpu = -1;
        void*     dst = nullptr;
        int64_t   len = 0;   ///< expected bytes (CQE res must equal this)
    };
    ::io_uring* ring_ = nullptr;
    uint64_t    next_token_ = 1;
    std::unordered_map<uint64_t, UringPending> uring_inflight_;
};

}  // namespace layerstorm::memory
