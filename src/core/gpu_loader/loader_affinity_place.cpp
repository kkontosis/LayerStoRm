#include "core/gpu_loader/loader_affinity_place.h"

#include <algorithm>
#include <numeric>

namespace layerstorm::gpu_loader {

namespace {

// Device visit order under the router's tie chain: free slot (tick 0 = oldest)
// first, then older victim (smaller board score), then lower index. Stable ⇒
// the strict-< scan's keep-first-seen semantics on exact ties.
std::array<int, kMaxDevices> age_order(const AffinityPlaceSignals& sig, int M) {
  std::array<int, kMaxDevices> order{};
  std::iota(order.begin(), order.begin() + M, 0);
  std::stable_sort(order.begin(), order.begin() + M, [&](int a, int b) {
    if (sig.free_slot[a] != sig.free_slot[b])
      return sig.free_slot[a] > sig.free_slot[b];
    return sig.victim_score[a] < sig.victim_score[b];
  });
  return order;
}

bool cached_any(const SolveRequest& req, int i, int M) {
  for (int j = 0; j < M; ++j)
    if (req.cached_at(i, j)) return true;
  return false;
}

}  // namespace

void build_affinity_place(const AffinityPlaceSignals& sig, SolveRequest& req) {
  const int M = req.num_devices;
  const int n = req.num_experts;

  const auto order = age_order(sig, M);
  std::array<int, kMaxDevices> agerank{};
  for (int r = 0; r < M; ++r) agerank[order[static_cast<size_t>(r)]] = r;

  const size_t nm = static_cast<size_t>(n) * M;
  if (req.place.size() != nm) req.place.assign(nm, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < M; ++j)
      // agerank+1, NOT agerank: a fetch onto the oldest device must still cost
      // strictly more than riding an existing resident copy (0), or the solver
      // ties "fetch at agerank 0" with "hit at the replica" and the gate order
      // — not the policy — would pick. Ordering across devices is unchanged.
      req.place[static_cast<size_t>(i) * M + j] =
          sig.valid[j] ? kAffinityAgeUs * (static_cast<double>(agerank[j]) + 1.0)
                       : kAffinityInvalidUs;

  // Convex per-device count term: cum[c] = kCount·c(c−1)/2. NO free-slot
  // folding — the router's miss_cnt counts every assigned miss regardless of
  // slot headroom (unlike the capacity-priced evict term this scratch reuses).
  if (static_cast<int>(req.evict_cum.size()) != M) req.evict_cum.assign(M, {});
  const size_t curve_len = static_cast<size_t>(n) + 1;
  for (int j = 0; j < M; ++j) {
    auto& cum = req.evict_cum[static_cast<size_t>(j)];
    cum.assign(curve_len, 0.0);
    for (size_t c = 2; c < curve_len; ++c)
      cum[c] = kAffinityCountUs * 0.5 * static_cast<double>(c)
               * static_cast<double>(c - 1);
  }
}

void canonicalize_affinity_pairing(const AffinityPlaceSignals& sig,
                                   const SolveRequest& req, int* assignment) {
  const int M = req.num_devices;
  const int n = req.num_experts;

  // True misses in routed order, and the solver's per-device miss counts.
  std::array<int, kMaxExperts> miss_idx{};
  std::array<int, kMaxDevices> count{};
  int nmiss = 0;
  for (int i = 0; i < n; ++i) {
    if (!req.pinned.empty() && req.pinned[i] >= 0) continue;
    if (cached_any(req, i, M)) continue;  // resident-replica placement: keep
    const int j = assignment[i];
    if (j < 0 || j >= M) continue;  // defensive: leave malformed entries alone
    miss_idx[static_cast<size_t>(nmiss++)] = i;
    ++count[static_cast<size_t>(j)];
  }
  if (nmiss <= 1) return;

  // Greedy routed-order round-robin over the solver-chosen multiset: level l
  // takes every device with count > l, in agerank order.
  const auto order = age_order(sig, M);
  int next = 0;
  for (int level = 0; next < nmiss; ++level)
    for (int r = 0; r < M && next < nmiss; ++r) {
      const int j = order[static_cast<size_t>(r)];
      if (count[static_cast<size_t>(j)] > level)
        assignment[miss_idx[static_cast<size_t>(next++)]] = j;
    }
}

}  // namespace layerstorm::gpu_loader
