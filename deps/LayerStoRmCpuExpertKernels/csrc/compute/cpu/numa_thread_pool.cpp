// IMPL: threadpool agent.
//
// Implements everything declared in numa_thread_pool.h:
//   * parse_cpulist / node_physical_cpus  — sysfs cpulist + thread_siblings_list
//     enumeration, dropping SMT siblings (keep lowest-numbered sibling per
//     physical core).
//   * pin_thread_to_cpu                    — pthread_setaffinity_np on the
//     calling thread.
//   * NumaThreadPool ctor/dtor             — spawn + pin worker threads to this
//     node's physical cores, park them on the generation handoff; dtor sets
//     shutdown_ and joins.
//   * NumaThreadPool::parallel_for         — fork-join: hand `body` + a fresh
//     CpuBarrierState (n_threads = num_threads()) to all participants, run
//     thread 0 (caller) inline, wait until jobs_remaining_ drains to 0.
//   * worker_main                          — park on job_generation_, run body
//     with its thread_id, signal done.
//
// CPU-only TU. Does NOT include any CUDA/GPU SDK header (INV-GPU-1) — this file
// is covered by layerstorm_no_cuda_check.

#include "compute/cpu/numa_thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <thread>
#include <set>
#include <sstream>
#include <string>

#include <pthread.h>
#include <sched.h>

#if defined(__SSE3__)
#include <immintrin.h>
#endif

namespace layerstorm::compute::cpu {

// ── sysfs helpers ────────────────────────────────────────────────────────────

namespace {

// Read the whole contents of a sysfs file into a string. Returns false (and
// leaves `out` empty) if the file cannot be opened / read.
bool read_sysfs_file(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    // Trim a trailing newline (sysfs entries are newline-terminated).
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return f.good() || f.eof();
}

}  // namespace

// ── parse_cpulist ────────────────────────────────────────────────────────────

// Parse a Linux sysfs "cpulist" string ("0-3,8,10-11") into the explicit
// ascending list of distinct cpu ids. Tolerant of surrounding whitespace and
// empty fields. Ranges with hi < lo are skipped.
std::vector<int> parse_cpulist(const std::string& cpulist) {
    std::set<int> ids;
    std::stringstream ss(cpulist);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Strip whitespace from both ends of the token.
        const auto first = token.find_first_not_of(" \t\n\r");
        if (first == std::string::npos) {
            continue;  // empty field
        }
        const auto last = token.find_last_not_of(" \t\n\r");
        token = token.substr(first, last - first + 1);

        const auto dash = token.find('-');
        if (dash == std::string::npos) {
            // Single cpu id.
            try {
                ids.insert(std::stoi(token));
            } catch (const std::exception&) {
                // Ignore unparsable field.
            }
        } else {
            // Inclusive range "lo-hi".
            try {
                const int lo = std::stoi(token.substr(0, dash));
                const int hi = std::stoi(token.substr(dash + 1));
                for (int c = lo; c <= hi; ++c) {
                    ids.insert(c);
                }
            } catch (const std::exception&) {
                // Ignore unparsable field.
            }
        }
    }
    return std::vector<int>(ids.begin(), ids.end());
}

// ── node_physical_cpus ───────────────────────────────────────────────────────

// Logical CPU ids on `numa_node` with SMT siblings dropped (one logical CPU per
// physical core), ascending. Empty if the node has no readable cpulist.
std::vector<int> node_physical_cpus(int numa_node) {
    if (numa_node < 0) {
        return {};
    }

    std::string cpulist;
    const std::string node_path =
        "/sys/devices/system/node/node" + std::to_string(numa_node) + "/cpulist";
    if (!read_sysfs_file(node_path, cpulist) || cpulist.empty()) {
        return {};
    }

    const std::vector<int> logical = parse_cpulist(cpulist);

    // Reduce to physical cores: for each logical CPU, read its
    // thread_siblings_list and keep only the lowest-numbered sibling. Using a
    // "representative" set avoids depending on enumeration order.
    std::set<int> kept;
    for (int cpu : logical) {
        const std::string sib_path = "/sys/devices/system/cpu/cpu" +
                                     std::to_string(cpu) +
                                     "/topology/thread_siblings_list";
        std::string sib_str;
        int representative = cpu;  // fallback if topology is unreadable
        if (read_sysfs_file(sib_path, sib_str) && !sib_str.empty()) {
            const std::vector<int> sibs = parse_cpulist(sib_str);
            if (!sibs.empty()) {
                representative = sibs.front();  // parse_cpulist returns ascending
            }
        }
        // Only keep the representative if it is actually one of this node's
        // logical CPUs (it should be — siblings share a node). If the lowest
        // sibling somehow lives off-node, fall back to the on-node cpu itself.
        if (std::find(logical.begin(), logical.end(), representative) !=
            logical.end()) {
            kept.insert(representative);
        } else {
            kept.insert(cpu);
        }
    }

    return std::vector<int>(kept.begin(), kept.end());
}

// ── pin_thread_to_cpu ────────────────────────────────────────────────────────

// Pin the CALLING thread to a single logical CPU via pthread_setaffinity_np.
bool pin_thread_to_cpu(int cpu) {
    if (cpu < 0) {
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
    return rc == 0;
}

// ── NumaThreadPool ───────────────────────────────────────────────────────────

NumaThreadPool::NumaThreadPool(int numa_node, int max_threads,
                               int reserve_leading_cores)
    : numa_node_(numa_node) {
    // Enumerate this node's physical cores. If sysfs is unavailable (or the node
    // has no cores), collapse to a single unpinned serial thread (thread 0 =
    // caller). core_cpus_ stays empty in that degenerate case so parallel_for
    // skips pinning entirely.
    std::vector<int> phys = node_physical_cpus(numa_node);

    int requested = (max_threads <= 0) ? static_cast<int>(phys.size())
                                       : max_threads;
    if (requested < 1) {
        requested = 1;
    }

    if (phys.empty()) {
        // No enumerable cores → serial fallback (unpinned). One participant.
        n_threads_ = 1;
        return;
    }

    // Reserve the node's LEADING physical core(s) for the orchestrator/daemon
    // (C-6 QA): hold them out of the pool so neither the leader nor the workers
    // ever pin onto them. Clamp so at least ONE core remains usable (a 1-core
    // node cannot reserve). The reserved ids are exposed via reserved_cpus() so
    // the engine can pin its daemon thread to reserved_cpus().front().
    int reserve = reserve_leading_cores;
    if (reserve < 0) reserve = 0;
    if (reserve > static_cast<int>(phys.size()) - 1)
        reserve = static_cast<int>(phys.size()) - 1;  // keep >= 1 usable
    if (reserve > 0) {
        reserved_cpus_.assign(phys.begin(), phys.begin() + reserve);
        phys.erase(phys.begin(), phys.begin() + reserve);
    }

    n_threads_ = std::min<int>(requested, static_cast<int>(phys.size()));
    if (n_threads_ < 1) {
        n_threads_ = 1;
    }

    // core_cpus_[i] is the physical CPU pinned to participant i. core_cpus_[0]
    // is the caller's core (pinned on entry to parallel_for, restored on exit).
    core_cpus_.assign(phys.begin(), phys.begin() + n_threads_);

    // Spawn persistent worker threads 1..n_threads_-1. Each parks on the job
    // generation handoff (worker_main) and pins itself to its physical core.
    workers_.reserve(static_cast<size_t>(n_threads_ - 1));
    for (int tid = 1; tid < n_threads_; ++tid) {
        workers_.emplace_back([this, tid]() { worker_main(tid); });
    }
}

NumaThreadPool::~NumaThreadPool() {
    // Signal shutdown and wake parked workers by bumping the generation, then
    // notify any that are blocked in the cv wait so they observe the shutdown.
    {
        std::lock_guard<std::mutex> lk(job_mu_);
        shutdown_.store(true, std::memory_order_release);
        job_generation_.fetch_add(1, std::memory_order_acq_rel);
    }
    job_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void NumaThreadPool::worker_main(int thread_id) {
    // Pin this worker to its physical core for its entire lifetime. Best-effort:
    // a failed pin (e.g. restricted cgroup affinity) just runs unpinned.
    if (thread_id < static_cast<int>(core_cpus_.size())) {
        pin_thread_to_cpu(core_cpus_[thread_id]);
    }

    uint64_t seen_generation = 0;

    for (;;) {
        // Park until a new job generation is published (or shutdown). HYBRID wait:
        // a short bounded spin (cheap sub-µs wakeup for back-to-back layer jobs),
        // then a REAL BLOCKING cv wait so the core is RELEASED (sleeps) during the
        // idle stretches between the ~2.3 ms/layer bursts. The old sched_yield spin
        // never slept → the node's cores sat at 100% for the whole run, starving
        // the GPU-driving daemon/CUDA-host threads this device overlaps with.
        constexpr int kSpinBudget = 4096;  // ~a few µs of pause before blocking
        bool got = false;
        for (int spin = 0; spin < kSpinBudget; ++spin) {
            if (job_generation_.load(std::memory_order_acquire) != seen_generation) {
                got = true;
                break;
            }
#if defined(__SSE3__)
            _mm_pause();
#elif defined(__ARM_NEON)
            __asm__ __volatile__("isb\n");
#endif
        }
        if (!got) {
            std::unique_lock<std::mutex> lk(job_mu_);
            job_cv_.wait(lk, [&] {
                return job_generation_.load(std::memory_order_acquire) !=
                           seen_generation ||
                       shutdown_.load(std::memory_order_acquire);
            });
        }
        seen_generation = job_generation_.load(std::memory_order_acquire);

        if (shutdown_.load(std::memory_order_acquire)) {
            return;
        }

        // Run this region's body. job_ and job_barrier_ were published before
        // the generation bump (release in parallel_for), so they are visible.
        const ParallelBody* body = job_;
        CpuBarrierState* bar = job_barrier_;
        if (body != nullptr && bar != nullptr) {
            (*body)(thread_id, n_threads_, *bar);
        }

        // Signal completion. The leader waits on jobs_remaining_ draining to 0.
        jobs_remaining_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void NumaThreadPool::parallel_for(const ParallelBody& body) {
    // Serial / unpinned fast path: no workers, run body inline as the sole
    // participant. A fresh barrier with n_threads=1 makes cpu::barrier() a
    // no-op, so the assemble layer's gate/up -> barrier -> swiglu -> barrier ->
    // down sequence still works correctly.
    if (n_threads_ <= 1) {
        CpuBarrierState bar;
        bar.n_threads = 1;
        body(0, 1, bar);
        return;
    }

    // Pin the calling thread (thread 0, the leader) to its physical core for the
    // duration of the region; restore its prior affinity afterwards so the
    // daemon thread is not left bound to one core between FFN calls.
    cpu_set_t prev_set;
    CPU_ZERO(&prev_set);
    const bool had_prev =
        (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &prev_set) == 0);
    const bool pinned_self = pin_thread_to_cpu(core_cpus_[0]);

    // Publish the job: one fresh barrier for THIS region (counters start at 0),
    // n_threads = all participants. Workers will decrement jobs_remaining_ as
    // they finish; the leader runs slice 0 inline and then drains the counter.
    CpuBarrierState bar;
    bar.n_threads = n_threads_;

    // Publish the job + bump the generation UNDER the mutex, then wake any
    // workers blocked in the cv wait. The job_/barrier_/jobs_remaining_ stores
    // happen-before the generation bump; a worker that observes the new
    // generation (lock-free in its spin, or via the predicate under the lock) is
    // guaranteed to see the new pointers. notify_all is issued AFTER releasing
    // the lock so woken workers don't immediately contend on it.
    {
        std::lock_guard<std::mutex> lk(job_mu_);
        job_ = &body;
        job_barrier_ = &bar;
        jobs_remaining_.store(n_threads_ - 1, std::memory_order_release);
        job_generation_.fetch_add(1, std::memory_order_acq_rel);
    }
    job_cv_.notify_all();

    // Leader runs its own slice inline (thread 0). It participates in the same
    // barriers as the workers, so the gate/up -> SwiGLU -> down rendezvous holds.
    body(0, n_threads_, bar);

    // Wait for all workers to finish this region. This is the leader (the CALLING
    // daemon thread) completing its OWN synchronous fold — bounded by the job's
    // slowest slice (~ms), not idle theft. Spin a bounded budget (workers almost
    // always finish within it), then fall back to a short real sleep so a
    // descheduled straggler cannot pin the daemon core at 100% (which would steal
    // from the GPU path). The ~50 µs sleep granularity is negligible vs the
    // ms-scale job and is only reached off the fast path.
    constexpr int kSpinBudget = 100000;
    for (int spin = 0;
         jobs_remaining_.load(std::memory_order_acquire) != 0;
         ++spin) {
        if (spin >= kSpinBudget) {
            spin = 0;
            std::this_thread::sleep_for(std::chrono::microseconds(20));
            continue;
        }
#if defined(__SSE3__)
        _mm_pause();
#elif defined(__ARM_NEON)
        __asm__ __volatile__("isb\n");
#endif
    }

    // Clear the published job so a stale pointer can't be re-run on the next
    // (unrelated) generation observation, then restore the leader's affinity.
    job_ = nullptr;
    job_barrier_ = nullptr;

    if (pinned_self && had_prev) {
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &prev_set);
    }
}

}  // namespace layerstorm::compute::cpu
