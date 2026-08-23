#pragma once

// CPU fork-join thread barrier — verbatim adaptation of ggml_barrier_impl.
//
// ─────────────────────────────────────────────────────────────────────────────
// Adapted (copied) from ik_llama.cpp ggml/src/ggml.c (function
// ggml_barrier_impl, ~line 4524). Two-counter sense-reversing barrier with a
// spin-then-yield wait; no mutex / condvar. The only change is mechanical: the
// C11 `atomic_int` / `atomic_*` calls become C++ `std::atomic<int>` member
// access, and the shared state lives in CpuBarrierState instead of
// ggml_compute_state_shared. The algorithm (counter increment, last-thread
// reset + passed bump, spin count, _mm_pause()/isb + sched_yield) is unchanged.
//
// MIT License
// Copyright (c) 2023-2024 The ggml authors
//   (https://github.com/ggml-org/ggml/blob/master/AUTHORS)
// Copyright (c) 2023-2024 The llama.cpp authors
//   (https://github.com/ggml-org/llama.cpp/blob/master/AUTHORS)
// Copyright (c) 2024-2025 The ik_llama.cpp authors
//   (https://github.com/ikawrakow/ik_llama.cpp/blob/main/AUTHORS)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ─────────────────────────────────────────────────────────────────────────────
//
// This header is COMPLETE (no body to fill in by a downstream agent): it is the
// verbatim copy + the std::atomic adaptation. CPU-only, no GPU SDK.

#include <atomic>
#include <sched.h>

#if defined(__SSE3__)
#include <immintrin.h>
#endif

namespace layerstorm::compute::cpu {

/// Shared state for a single fork-join barrier instance.
///
/// `n_threads` is the number of threads that will rendezvous (the calling
/// daemon thread counts as thread 0, so this == pool worker count + 1). The two
/// counters implement the sense-reversing barrier and must be reset to 0 before
/// the barrier is reused across an independent parallel region (a fresh
/// CpuBarrierState per parallel region is the simplest correct usage).
struct CpuBarrierState {
    int n_threads = 1;
    std::atomic<int> n_barrier{0};
    std::atomic<int> n_barrier_passed{0};
};

/// Rendezvous all `state.n_threads` threads. Verbatim ggml_barrier_impl logic.
///
/// Every participating thread must call barrier() the same number of times with
/// the same CpuBarrierState. The last thread to arrive (== n_threads - 1) resets
/// n_barrier and bumps n_barrier_passed; the others spin n_spin_before_sleep
/// iterations with a CPU pause hint then sched_yield() until passed advances.
inline void barrier(CpuBarrierState& state) {
    if (state.n_threads == 1) {
        return;
    }

    std::atomic<int>* n_barrier = &state.n_barrier;
    std::atomic<int>* n_barrier_passed = &state.n_barrier_passed;

    int n_threads = state.n_threads;
    int passed_old = n_barrier_passed->load();

    if (n_barrier->fetch_add(1) == n_threads - 1) {
        // last thread
        n_barrier->store(0);
        n_barrier_passed->fetch_add(1);
    } else {
        // wait for other threads
        const int n_spin_before_sleep = 100000;
        while (true) {
            for (int i = 0; i < n_spin_before_sleep; i++) {
                if (n_barrier_passed->load() != passed_old) {
                    return;
                }
#if defined(__SSE3__)
                _mm_pause();
#elif defined(__ARM_NEON)
                __asm__ __volatile__("isb\n");
#endif
            }
            sched_yield();
        }
    }
}

}  // namespace layerstorm::compute::cpu
