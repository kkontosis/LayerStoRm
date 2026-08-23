#pragma once

// parallel_for — tiny std::thread fan-out for embarrassingly-parallel CPU
// loops (weight-load transforms, dequants, packs). Deliberately NOT OpenMP:
// no new build/runtime dependency, and the codebase's existing parallel
// sections (arena prefault, register_all, TQ rotation precompute) are all
// std::thread. Thread count is capped LOW by default (min(hw, 15)) — these run
// during engine boot next to CUDA init / NVMe preload and must not starve
// them (user decision: "a safe non-extreme factor").
//
// Semantics: fn(i) is invoked exactly once for every i in [0, n), from up to
// max_threads threads (dynamic chunking via an atomic counter — good load
// balance for skewed per-item costs like mixed-size tensors). fn must write
// only item-disjoint state. Exceptions: the FIRST thrown exception (by
// completion order) is rethrown on the caller after all threads join.
// n <= 1 or max_threads <= 1 runs inline (zero overhead, same code path).

#include <atomic>
#include <cstddef>
#include <exception>
#include <thread>
#include <vector>

namespace layerstorm::core {

inline unsigned parallel_for_default_threads() {
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1u : (hw < 15u ? hw : 15u);  // min(hw, 15) — user: "<16"
}

namespace detail {
/// Nesting guard: a parallel_for called from inside a parallel_for worker
/// runs INLINE — keeps the global thread count at the outer fan-out's cap
/// (≤8) instead of multiplying (outer × inner).
inline thread_local bool in_parallel_for_worker = false;
}  // namespace detail

template <typename Fn>
void parallel_for(size_t n, Fn&& fn,
                  unsigned max_threads = parallel_for_default_threads()) {
    if (n == 0) return;
    const unsigned nthreads =
        max_threads < 2 || n < 2 || detail::in_parallel_for_worker
            ? 1u
            : (static_cast<size_t>(max_threads) < n
                   ? max_threads
                   : static_cast<unsigned>(n));
    if (nthreads == 1) {
        for (size_t i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::exception_ptr err;
    std::atomic<bool> have_err{false};
    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) {
        workers.emplace_back([&] {
            detail::in_parallel_for_worker = true;
            try {
                for (size_t i = next.fetch_add(1); i < n;
                     i = next.fetch_add(1)) {
                    if (have_err.load(std::memory_order_relaxed)) return;
                    fn(i);
                }
            } catch (...) {
                if (!have_err.exchange(true)) err = std::current_exception();
            }
        });
    }
    for (auto& w : workers) w.join();
    if (err) std::rethrow_exception(err);
}

}  // namespace layerstorm::core
