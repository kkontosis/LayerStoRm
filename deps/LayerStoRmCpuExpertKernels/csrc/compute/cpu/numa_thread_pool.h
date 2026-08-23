#pragma once

// NumaThreadPool — fork-join thread pool pinned to the PHYSICAL cores of one
// NUMA node, for the CPU expert FFN (C-6).
//
// Threading model (locked by the C-6 design):
//   * One parallel region per FFN call. The pool is created once (per device)
//     and reused; each parallel_for() call is a single fork-join with an
//     ik-barrier rendezvous (src/compute/cpu/ik_barrier.h).
//   * Thread 0 == the CALLING daemon thread. It participates in the work and
//     holds the result on return — parallel_for() does NOT return until every
//     worker has finished its slice (the calling thread runs slice 0 inline).
//   * Workers (threads 1..n_threads-1) are persistent std::threads spun up at
//     construction, each pinned to one physical core of this node and parked on
//     a job handoff until parallel_for() dispatches.
//   * Threads are pinned to PHYSICAL cores: SMT siblings are dropped so at most
//     one logical CPU per physical core is used (avoids two FFN threads sharing
//     one core's vector units). Pinning via pthread_setaffinity_np.
//
// Core enumeration (no such code exists in the repo yet — this builds it):
//   * node_physical_cpus(node): read /sys/devices/system/node/node<N>/cpulist,
//     expand the ranges, then for each logical CPU read
//     /sys/devices/system/cpu/cpu<n>/topology/thread_siblings_list and keep only
//     the lowest-numbered sibling (one logical CPU per physical core). Returns
//     the kept logical CPU ids, ascending.
//
// CPU-only: no CUDA / GPU SDK include. Goes through layerstorm_no_cuda_check.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "compute/cpu/ik_barrier.h"

namespace layerstorm::compute::cpu {

// ── Core enumeration (free functions) ────────────────────────────────────────

/// Logical CPU ids on `numa_node`, with SMT siblings dropped (one logical CPU
/// per physical core), ascending. Reads sysfs cpulist + thread_siblings_list.
/// Empty if the node has no cpulist (e.g. sysfs unavailable) — the caller then
/// falls back to an unpinned single-thread run.
/// IMPL: threadpool agent.
std::vector<int> node_physical_cpus(int numa_node);

/// Parse a Linux sysfs "cpulist" string (e.g. "0-3,8,10-11") into the explicit
/// ascending list of cpu ids. Exposed for unit testing the range expansion.
/// IMPL: threadpool agent.
std::vector<int> parse_cpulist(const std::string& cpulist);

/// Pin the CALLING thread to a single logical CPU via pthread_setaffinity_np.
/// Returns true on success, false if the affinity call failed (caller may
/// continue unpinned). IMPL: threadpool agent.
bool pin_thread_to_cpu(int cpu);

// ── Fork-join pool ───────────────────────────────────────────────────────────

/// The worker body signature for parallel_for. Each participating thread is
/// invoked once per parallel_for() call with:
///   thread_id : 0..n_threads-1 (0 == calling daemon thread)
///   n_threads : total participants (== num_threads())
///   barrier   : the shared barrier for THIS parallel region — call
///               cpu::barrier(barrier) at each gate/up -> SwiGLU -> down stage
///               boundary so all threads sync before the next stage reads the
///               previous stage's output.
/// The body itself partitions work (e.g. output rows N) across thread_id.
using ParallelBody =
    std::function<void(int thread_id, int n_threads, CpuBarrierState& barrier)>;

/// Fork-join thread pool pinned to one NUMA node's physical cores.
///
/// Construction pins the worker threads (thread_id 1..) to this node's physical
/// cores and parks them. Thread 0 is whatever thread calls parallel_for (the
/// daemon thread); it is pinned to one of the node's cores for the duration of
/// the call. n_threads = min(requested, node_physical_cpus().size()); if the
/// node has no enumerable cores, n_threads collapses to 1 (serial, unpinned).
class NumaThreadPool {
public:
    /// Build a pool for `numa_node`. `max_threads` <= 0 means "use all physical
    /// cores on the node". The pool reserves one core slot for thread 0 (the
    /// caller) so total participants == n cores used. IMPL: threadpool agent.
    ///
    /// `reserve_leading_cores` (default 0): skip this many of the node's LEADING
    /// physical cores when selecting the pool's cores — they are held out of the
    /// FFN pool (neither leader nor worker touches them) so the orchestrator /
    /// daemon (which this node also hosts, e.g. gpu1==node3) has an undisturbed
    /// core. With reserve=1 on a 14-core node, workers+leader use the SECOND core
    /// onward (13 participants); the reserved leading core id(s) are exposed via
    /// reserved_cpus() so the caller can pin its daemon thread there (C-6 QA).
    /// Clamped so at least ONE core always remains for the pool.
    NumaThreadPool(int numa_node, int max_threads, int reserve_leading_cores = 0);
    ~NumaThreadPool();

    NumaThreadPool(const NumaThreadPool&) = delete;
    NumaThreadPool& operator=(const NumaThreadPool&) = delete;
    NumaThreadPool(NumaThreadPool&&) = delete;
    NumaThreadPool& operator=(NumaThreadPool&&) = delete;

    /// Total participants in a parallel region (worker threads + calling thread).
    int num_threads() const { return n_threads_; }

    /// NUMA node this pool is pinned to.
    int numa_node() const { return numa_node_; }

    /// The node's LEADING physical core id(s) held OUT of the pool (empty unless
    /// reserve_leading_cores > 0). The caller pins its daemon/orchestrator thread
    /// to reserved_cpus().front() so the FFN pool never contends with it (C-6 QA).
    const std::vector<int>& reserved_cpus() const { return reserved_cpus_; }

    /// Run `body` as a single fork-join parallel region. The CALLING thread is
    /// thread 0 and runs its slice inline; this returns only after every worker
    /// has returned from `body`. A fresh CpuBarrierState (n_threads set to
    /// num_threads()) is handed to every participant for THIS region. Re-entrant
    /// calls are NOT supported (one parallel region at a time per pool).
    /// IMPL: threadpool agent.
    void parallel_for(const ParallelBody& body);

private:
    int numa_node_;
    int n_threads_ = 1;
    std::vector<int> core_cpus_;          ///< physical CPU ids, [0]=caller core
    std::vector<int> reserved_cpus_;      ///< leading cores held out (reserve>0)
    std::vector<std::thread> workers_;    ///< workers_.size() == n_threads_-1

    // Job handoff + lifecycle state. IMPL: threadpool agent fleshes out the
    // exact synchronization (the design permits a generation-counter handoff:
    // workers spin/wait on a job generation, run body, then signal done).
    const ParallelBody* job_ = nullptr;
    CpuBarrierState* job_barrier_ = nullptr;
    std::atomic<uint64_t> job_generation_{0};
    std::atomic<int> jobs_remaining_{0};
    std::atomic<bool> shutdown_{false};
    // Idle-park BLOCKING wait (C-6 GPU-overlap fix): CPU experts run OVERLAPPED
    // with the GPU-driving daemon + CUDA host threads + H2D staging, so idle
    // workers MUST sleep (not spin) between the ~2.3 ms/layer job bursts — a
    // sched_yield spin never sleeps and pins the node's cores at 100%, stealing
    // cycles/DDR bandwidth from the GPU path. A short bounded spin precedes the
    // block (hybrid) so back-to-back layers still wake with ~pause latency.
    std::mutex job_mu_;
    std::condition_variable job_cv_;

    void worker_main(int thread_id);
};

}  // namespace layerstorm::compute::cpu
