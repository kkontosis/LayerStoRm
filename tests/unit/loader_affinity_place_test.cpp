// P-25 existence-proof property test: the exact solver, given the synthetic
// affinity-expressing objective (loader_affinity_place) over ABLATED constants
// (all legacy cost inputs zero — what LS_LOADER_ABLATE hands it), must
// reproduce the keeper affinity router's placement decisions exactly:
//   hits ride residency (pinned, or free 0-cost resident-replica placement);
//   each miss -> argmin_g (misses_assigned_this_layer[g], free-slot/oldest-
//   victim key, g), greedily in routed order (strict-< scan, keep-first-seen);
//   >=2 misses on one device pair round-robin in routed order.
// The reference below transcribes keeper52_test.cpp's router with frozen
// signals (its LRU state does not change while one layer is assigned). Rows
// resident on >1 device are not generated: the affinity policy never creates
// duplicates (hits run where resident; preload is a disjoint e%tp partition).
#include "core/gpu_loader/loader_affinity_place.h"

#include <array>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace gl = layerstorm::gpu_loader;

namespace {

// All-zero cost constants: what the solver sees under full LS_LOADER_ABLATE.
gl::LoaderConstants ablated_constants(int M) {
  gl::LoaderConstants k;
  k.source      = "test";
  k.num_devices = M;
  k.num_banks   = 1;
  k.ncf         = {0.0, 1.0};
  for (int d = 0; d < M; ++d) {
    gl::DeviceConstants dc;
    dc.position  = d;
    dc.numa_node = 0;
    k.devices.push_back(dc);  // xfer_lat/compute/recon all zero
  }
  gl::BankConstants bc;
  bc.node = 0;  // egress 0
  k.banks.push_back(bc);
  k.matrix.assign(1, std::vector<gl::TransferCell>(static_cast<size_t>(M)));
  return k;
}

// Keeper router transcription (keeper52_test.cpp miss loop, NUMA route off,
// frozen signals). key(g) precedence: free slot (tick 0 = oldest possible)
// beats any full device; among full devices smaller victim score = older wins;
// exact ties keep the first-seen (lower g). Returns -1 for "no valid device".
struct RefRouter {
  const gl::AffinityPlaceSignals& sig;
  std::array<int, gl::kMaxDevices> miss_cnt{};

  explicit RefRouter(const gl::AffinityPlaceSignals& s) : sig(s) {}

  bool key_less(int a, int b) const {  // strictly-older tie key
    if (sig.free_slot[a] != sig.free_slot[b])
      return sig.free_slot[a] > sig.free_slot[b];
    return sig.victim_score[a] < sig.victim_score[b];
  }
  int place_miss() {
    int best = -1;
    for (int g = 0; g < sig.num_devices; ++g) {
      if (!sig.valid[g]) continue;
      if (best < 0 || miss_cnt[g] < miss_cnt[best] ||
          (miss_cnt[g] == miss_cnt[best] && key_less(g, best)))
        best = g;
    }
    if (best >= 0) ++miss_cnt[best];
    return best;
  }
};

struct Config {
  int M = 4;
  int n = 8;
  gl::AffinityPlaceSignals sig;
  std::vector<int> resident;  // per expert: device with the only copy, -1 = miss
  std::vector<int> target;    // orchestrator target (e%tp analog)
};

Config random_config(std::mt19937& rng) {
  Config c;
  c.M = (rng() % 2) ? 4 : 2;
  c.n = 1 + static_cast<int>(rng() % 8);
  c.sig.num_devices = c.M;
  std::uniform_real_distribution<double> score(0.0, 1000.0);
  for (int j = 0; j < c.M; ++j) {
    c.sig.valid[j]     = 1;
    c.sig.free_slot[j] = (rng() % 4) == 0;
    // Duplicate scores across devices with some probability (tie exercise).
    c.sig.victim_score[j] =
        (j > 0 && (rng() % 5) == 0) ? c.sig.victim_score[j - 1] : score(rng);
  }
  c.resident.resize(static_cast<size_t>(c.n));
  c.target.resize(static_cast<size_t>(c.n));
  for (int i = 0; i < c.n; ++i) {
    c.target[static_cast<size_t>(i)] = static_cast<int>(rng() % c.M);
    const int roll = static_cast<int>(rng() % 3);
    c.resident[static_cast<size_t>(i)] =
        roll == 0 ? -1                                  // miss
        : roll == 1 ? c.target[static_cast<size_t>(i)]  // hit at target (pinned)
                    : static_cast<int>(rng() % c.M);    // resident anywhere
  }
  return c;
}

// Build the SolveRequest the dispatcher would (pin-hits rule: pinned iff
// cached at the orchestrator target), run the ablated solve + synthetic
// objective + pairing canonicalization, and return the assignment.
std::vector<int> solve_arm(const Config& c) {
  gl::SolveRequest req;
  req.num_devices = c.M;
  req.num_experts = c.n;
  req.bank_of.assign(static_cast<size_t>(c.n), 0);
  req.cached.assign(static_cast<size_t>(c.n) * c.M, 0);
  req.pinned.assign(static_cast<size_t>(c.n), -1);
  for (int i = 0; i < c.n; ++i) {
    const int r = c.resident[static_cast<size_t>(i)];
    if (r >= 0) req.cached[static_cast<size_t>(i) * c.M + r] = 1;
    if (r >= 0 && r == c.target[static_cast<size_t>(i)]) req.pinned[i] = r;
  }
  gl::build_affinity_place(c.sig, req);
  const gl::LoaderConstants k = ablated_constants(c.M);
  gl::LoaderSolver solver;
  auto res = solver.solve(k, req);
  gl::canonicalize_affinity_pairing(c.sig, req, res.assignment.data());
  return std::vector<int>(res.assignment.begin(),
                          res.assignment.begin() + c.n);
}

std::vector<int> reference(const Config& c) {
  std::vector<int> out(static_cast<size_t>(c.n), -1);
  RefRouter ref(c.sig);
  for (int i = 0; i < c.n; ++i) {  // hits first: residency decided before misses
    const int r = c.resident[static_cast<size_t>(i)];
    if (r >= 0) out[static_cast<size_t>(i)] = r;
  }
  for (int i = 0; i < c.n; ++i)
    if (c.resident[static_cast<size_t>(i)] < 0)
      out[static_cast<size_t>(i)] = ref.place_miss();
  return out;
}

}  // namespace

TEST(LoaderAffinityPlace, SolverReproducesGreedyRouter) {
  std::mt19937 rng(20260718);
  for (int trial = 0; trial < 2000; ++trial) {
    const Config c = random_config(rng);
    const auto got = solve_arm(c);
    const auto want = reference(c);
    ASSERT_EQ(got, want)
        << "trial " << trial << " M=" << c.M << " n=" << c.n;
  }
}

TEST(LoaderAffinityPlace, AllMissWarmupRoundRobins) {
  // Token-0 shape: 8 misses on 4 full devices with distinct ages -> 2 per
  // device, paired round-robin in routed order over the age order.
  Config c;
  c.M = 4;
  c.n = 8;
  c.sig.num_devices = 4;
  for (int j = 0; j < 4; ++j) {
    c.sig.valid[j]     = 1;
    c.sig.free_slot[j] = 0;
  }
  // Age order (oldest first): device 2, 0, 3, 1.
  c.sig.victim_score = {20.0, 40.0, 10.0, 30.0};
  c.resident.assign(8, -1);
  c.target.assign(8, 0);
  const auto got = solve_arm(c);
  const std::vector<int> want = {2, 0, 3, 1, 2, 0, 3, 1};
  EXPECT_EQ(got, want);
}

TEST(LoaderAffinityPlace, FreeSlotBeatsOldVictim) {
  Config c;
  c.M = 2;
  c.n = 1;
  c.sig.num_devices = 2;
  c.sig.valid = {1, 1};
  c.sig.free_slot = {0, 1};           // device 1 has headroom
  c.sig.victim_score = {5.0, 999.0};  // device 0's victim is far older
  c.resident.assign(1, -1);
  c.target.assign(1, 0);
  EXPECT_EQ(solve_arm(c), (std::vector<int>{1}));
}

TEST(LoaderAffinityPlace, CachedElsewhereRidesResidency) {
  // Expert resident away from its orchestrator target stays FREE for the
  // solver, which must still send it to the resident replica (0-cost) — the
  // "hits ride residency" half of the affinity policy.
  Config c;
  c.M = 4;
  c.n = 3;
  c.sig.num_devices = 4;
  for (int j = 0; j < 4; ++j) c.sig.valid[j] = 1;
  c.sig.victim_score = {1.0, 2.0, 3.0, 4.0};
  c.resident = {2, -1, 3};  // experts 0/2 resident off-target
  c.target   = {0, 0, 1};
  const auto got = solve_arm(c);
  EXPECT_EQ(got[0], 2);
  EXPECT_EQ(got[2], 3);
  EXPECT_EQ(got[1], 0);  // sole miss -> oldest victim (device 0)
}
