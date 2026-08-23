#pragma once
// EPM-0 keeppred: prev-round-union eviction-RETENTION bias (spec/reports/
// DSP52_BOOST.md "EPM-0"; INV-KEEPPRED). NOT a prefetch — it never fetches and
// never changes routing/placement; it only reorders the eviction VICTIM
// candidate list so experts predicted to recur soon (members of their layer's
// previous-round routed union — the same free temporal-locality signal the
// DSP52_PREFETCH lever uses, union-cov ~0.88) become more costly to evict and
// are preferentially KEPT if already resident. Fewer demand fetches / higher
// hit rate at ZERO extra bandwidth (the bandwidth-bound bus, FETCH_XRAY.md).
//
// Pure, ALLOCATION-FREE (operates on caller-owned fixed buffers — no heap, no
// std::vector, no std::sort temp), CUDA-free (layerstorm_core / INV-GPU-1).
// Header-only so both the daemon eviction path and the REEF test harnesses
// share ONE mechanism over a small bounded victim window.
#include <cstddef>
#include <cstdint>

namespace layerstorm::gpu_loader {

// Max victim-window candidates the keeppred reorder considers (a bounded cheap
// tail — see reef_orch_apply). Sized so the per-GPU miss count of the largest
// routed union (kMaxExpertsLarge=256 over ≥4 devices ⇒ ≤64/GPU) plus headroom
// fits; the caller stack-allocates buffers of this size (small: ~1 KiB).
inline constexpr int kKeeppredWindow = 128;

// Reorder a board cheapest-first victim candidate window under the retention
// bias. Fixed-buffer / allocation-free.
//
//   eff[i]      : keys[i]'s board EFFECTIVE score (the value cheapest_keys /
//                 cheapest_scores_sorted rank by), ascending (cheapest first).
//   retained[i] : 1 iff keys[i] is a predicted-reuse member (a member of its
//                 layer's previous-round routed union) → protect it.
//   n           : candidate count (0 <= n <= caller buffer capacity).
//   bias        : additive keep-cost, >= 0. adjusted[i] = eff[i] +
//                 (retained[i] ? bias : 0). BOUNDED: a protected expert is still
//                 evicted when nothing cheaper is left (never fabricates budget
//                 → no livelock, INV-FAR-WAVE discipline).
//   order[]     : OUT — candidate indices sorted by adjusted score ASCENDING,
//                 STABLE (equal-adjusted ties preserve the input cheapest-first
//                 order → deterministic, lowest-slot tie-break). Caller-owned,
//                 capacity >= n.
//
// At bias <= 0, or when no candidate is retained, order is the identity
// 0..n-1 — the caller's victim sequence is then BYTE-IDENTICAL to the un-biased
// cheapest-first path (the OFF/weight-0 contract). Stable insertion sort (n is a
// small bounded window; no heap, no std::sort scratch).
inline void keeppred_victim_order(const double* eff, const uint8_t* retained,
                                  int n, double bias, int* order) {
  for (int i = 0; i < n; ++i) order[i] = i;
  if (bias <= 0.0 || n <= 1) return;
  bool any_retained = false;
  for (int i = 0; i < n; ++i)
    if (retained[i]) { any_retained = true; break; }
  if (!any_retained) return;  // identity — nothing to reorder
  // Stable insertion sort by adjusted score ascending (strict >, so equal
  // adjusted keys keep their input cheapest-first order).
  for (int i = 1; i < n; ++i) {
    const int cur = order[i];
    const double ck = eff[cur] + (retained[cur] ? bias : 0.0);
    int j = i - 1;
    for (; j >= 0; --j) {
      const int oj = order[j];
      const double kj = eff[oj] + (retained[oj] ? bias : 0.0);
      if (kj > ck) order[j + 1] = order[j];
      else break;
    }
    order[j + 1] = cur;
  }
}

}  // namespace layerstorm::gpu_loader
