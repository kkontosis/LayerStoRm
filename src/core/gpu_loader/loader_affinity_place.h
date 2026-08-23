#pragma once
// P-25 work item 1 — the "REEF expresses affinity" synthetic solver objective.
//
// build_affinity_place() fills SolveRequest::place / ::evict_cum so that, with
// every legacy cost input ablated (LS_LOADER_ABLATE: zeroed transfer matrix,
// compute curves, bank egress, recon), the exact solver's minimum reproduces
// the keeper affinity router's placement (keeper52_test.cpp): hits ride
// residency (pinned, or a free 0-cost resident-replica placement), and each
// miss goes to argmin_g (misses_assigned_this_layer[g], oldest-victim age, g),
// greedily in routed order. The greedy's two cost criteria map onto the
// objective as a strict scale hierarchy:
//   count — evict_cum[j][n] = kCount·n(n−1)/2: convex in nunc_[j], so the
//           joint minimum balances per-device miss counts exactly like the
//           greedy (Σ C(n_j,2) is minimized by the most-equal partition);
//   age   — place[i][j] = kAge·agerank_j (agerank: free-slot devices first —
//           the router reads a free slot as tick 0, the oldest possible — then
//           ascending victim score = ascending last-touch clock, then j;
//           stable, so exact-tie ages keep the lower device index like the
//           router's strict-< scan). kCount ≫ N·M·kAge keeps them lexicographic.
//
// The remaining greedy artifact — WHICH miss lands on which chosen device when
// a device takes ≥2 misses in one layer (warmup shapes) — is round-robin by
// routed order, which no per-(i,j) linear place cost can express (pairing
// permutations are exact co-optima of this objective: same counts, same device
// multiset, place cost depends on j only). canonicalize_affinity_pairing()
// therefore re-pairs the solved miss set to the greedy's routed-order
// round-robin AMONG THOSE CO-OPTIMA — a deterministic tie normalization, not a
// placement override: counts and the device set stay exactly the solver's.
//
// Pure and CUDA-free: all live-state reads (board victim scores, cache free
// slots) happen in the caller (dispatch_loader.cpp); this unit is testable
// against a reference greedy router with no daemon state.
#include <array>

#include "core/gpu_loader/loader_solver.h"

namespace layerstorm::gpu_loader {

// Scale hierarchy (µs units, only ratios matter). kInvalid poisons devices the
// dispatcher cannot route to (position out of backend range) so the solver
// never selects them for a miss.
inline constexpr double kAffinityCountUs   = 1e6;
inline constexpr double kAffinityAgeUs     = 1e3;
inline constexpr double kAffinityInvalidUs = 1e12;

// Per-device pooled-LRU signals, indexed by solver device j. Frozen for the
// whole layer solve — matching the router, whose LRU state does not change
// while it assigns one layer's misses.
struct AffinityPlaceSignals {
  int num_devices = 0;
  std::array<uint8_t, kMaxDevices> valid{};        // routable device
  std::array<uint8_t, kMaxDevices> free_slot{};    // stable-zone free slots > 0
  std::array<double, kMaxDevices>  victim_score{}; // board cheapest effective
                                                   // score (smaller = older)
};

// Overwrites req.place (N·M) and req.evict_cum (M × N+1). Reads req.num_*,
// req.cached (a cached-anywhere row is never charged place — apply() skips
// hits — so only true misses feel these costs).
void build_affinity_place(const AffinityPlaceSignals& sig, SolveRequest& req);

// Re-pair the TRUE-MISS entries of `assignment` (n = req.num_experts, values =
// solver device j) to the greedy router's routed-order round-robin over the
// solver-chosen device multiset, devices visited in agerank order. Entries
// with any cached[] bit (hits / resident replicas) and pinned entries are
// untouched. Objective-neutral by construction (see header note).
void canonicalize_affinity_pairing(const AffinityPlaceSignals& sig,
                                   const SolveRequest& req, int* assignment);

}  // namespace layerstorm::gpu_loader
