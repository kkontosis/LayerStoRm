#pragma once
// GPU loader solver (spec/GPU_LOADER_MODEL.md §3/§5, ticket I8).
//
// Given the calibrated LoaderConstants + a per-(token,layer) request (the N
// routed experts, their source banks, per-device cache state), choose the
// expert->device assignment j[.] minimizing the predicted token-layer time
//
//   T = prep + max(device_makespan, bank_egress) + recon + place + evict
//
// CUDA-free, pure, and **preallocated**: LoaderSolver holds fixed-size scratch
// (bounded by kMax* below) and reuses it across solves — zero hot-path heap
// allocation. Runs on the daemon thread every MoE layer; deterministic
// (lowest-index tie-break).
//
// The solver is a template on the max-expert bound (BasicLoaderSolver<NMax>)
// with exactly TWO instantiations:
//   * LoaderSolver    = BasicLoaderSolver<kMaxExperts (64)> — the calibrated
//     production solver. Its code path is FROZEN: the template parameter is the
//     only textual change vs the historical non-template class, so the 64
//     instantiation is expression-for-expression identical (REEF fingerprints
//     are calibrated against it; guarded by Solver64GoldenRegression).
//   * LoaderSolver256 = BasicLoaderSolver<kMaxExpertsLarge (256)> — large
//     batched-verify unions (dsp52 γ-chunk deduped unions > 64).
//
// Tiers (solve()):
//   (1) M^N <= kMaxEnumerations  -> enumerate (full-objective exact)
//   (2) N   <= kMaxC             -> subset-partition DP O(M*3^N) (makespan+floor
//                                   exact; consequence post-eval corrective)
//   (3) else, NMax == 64         -> greedy-prefix (route 3: LPT freezes the
//                                   costliest N-C, exact DP on the cheapest C)
//   (3') else, NMax > 64         -> bounded greedy (greedy_pinned): pinned hits
//                                   ride their resident device verbatim, then
//                                   the misses are admitted greedily in index
//                                   order on the full incremental objective
//                                   (same terms as the exact tiers). The exact
//                                   tiers (1)/(2) and the pinned-hits domain
//                                   reduction keep their EXISTING small-M/N
//                                   budgets in both instantiations.
#include <array>
#include <cstdint>
#include <vector>

#include "core/gpu_loader/loader_constants.h"

namespace layerstorm::gpu_loader {

// Compile-time bounds (sets the preallocated buffer sizes; see ticket I8b).
inline constexpr int      kMaxExperts = 64;       // Nmax routed experts per solve (default instantiation)
inline constexpr int      kMaxExpertsLarge = 256; // Nmax for LoaderSolver256 (batched-verify unions)
inline constexpr int      kMaxC       = 5;        // Cmax experts solved exactly by the DP (2^C states)
inline constexpr int      kMaxDevices = 16;       // Mmax routable devices
inline constexpr int      kMaxBanks   = 16;       // Bmax NUMA banks
inline constexpr uint64_t kMaxEnumerations = 1u << 22;  // full-objective enumeration budget (~4.2M)
inline constexpr int      kDpStates   = 1 << kMaxC;     // 64

// compute_j(c) = a*c + b*ceil(c/P)  (batch-step, §2.3). c>=0; 0 at c=0.
// inline (header) so it folds into the hot dp_core/greedy loops (loop-invariant
// a/b/P per device); P==1 skips the integer divide.
inline double compute_us(const ComputeCurve& cc, int c) {
  if (c <= 0) return 0.0;
  const int P = cc.P > 0 ? cc.P : 1;
  const int batches = (P == 1) ? c : 1 + (c - 1) / P;  // ceil(c/P)
  return cc.a_us * static_cast<double>(c) + cc.b_us * static_cast<double>(batches);
}

// Contention-aware bank-egress factor (§3.3). raw_sum is the strict-serial
// per-bank draw (Σ egress over the uncached experts assigned from this bank);
// g is the parallelism degree = number of DISTINCT devices holding ≥1 uncached
// expert from this bank in the candidate assignment (g≥1 whenever raw_sum>0). c
// is the calibrated contention (1 = serial, 0 = parallel; clamped to [0,1]):
//   eff = raw_sum · ( c + (1−c)/g )
// c=1 → factor 1 → EXACTLY the strict-serial raw_sum (no-op vs. the old floor);
// c=0 → factor 1/g → spreading across g devices divides the draw by g. The term
// thus REWARDS spreading only when the bank channel is PARALLEL (c→0), and is
// inert when SERIAL (c→1). Assignment-dependent via g (the old floor was not).
inline double bank_egress_factor(double c, int g) {
  if (c < 0.0) c = 0.0;        // defensive clamp to [0,1]
  if (c > 1.0) c = 1.0;
  if (g < 1)   g = 1;
  return c + (1.0 - c) / static_cast<double>(g);
}

// Per-solve inputs (extracted from live daemon state for the N experts of
// interest). Device index j in [0,M) aligns with LoaderConstants.devices[j];
// bank index in [0,num_banks) aligns with LoaderConstants.banks[b]/matrix[b].
struct SolveRequest {
  int num_devices = 0;                  // M (<= constants.num_devices, <= kMaxDevices)
  int num_experts = 0;                  // N (<= the solver instantiation's NMax bound)

  std::vector<int>     bank_of;         // [N] source-bank slot of each expert
  std::vector<uint8_t> cached;          // [N*M] row-major cached_gpu[i][j]; 1 = resident on j
  std::vector<double>  subprep_us;      // [N] NVMe->RAM stage (0 if in RAM); j-independent

  std::vector<double>              place;      // [N*M] place_cons[i][j] (empty -> 0)
  std::vector<std::vector<double>> evict_cum;  // [M][k] prefix sum: evict_cum[j][n] = cost of n evictions on j
  bool clamp_place = true;              // clamp0: place_cons -> max(0, .) (default; §5.4)

  // Optional M2v2 EXPOSED-WALL makespan (config-scoped fit; exposed_model.py
  // --form v2; DETHRONE_AFFINITY_IDEAS.md B>1-forward reqs). When armed, the
  // device makespan term becomes
  //   cpw[d] * ( m2_gc*compute_us(d, cnt_d) + max(0, s[d]*sub_d - credit_d) )
  //   credit_d = o0[d] + oc[d]*hsat[d]*(1 - exp(-hits_d/hsat[d]))
  // i.e. the batch-step compute stays a first-class term (never absorbed by the
  // credit — wrong at B>1 otherwise) and the overlap credit SATURATES in the
  // resident-hit count. credit_d is FROZEN from the request's pinned hits
  // (m2_arm): constant during any DFS, so the objective is monotone in
  // sub_/cnt_ and the prefix bound stays valid at arbitrary miss counts.
  // ARMING: solve() arms m2 ONLY on the pinned tier; the makespan-specific
  // suffix residual floors (res_ms_/avg-load) are zeroed under m2 — they price
  // the LEGACY makespan and a saturating credit can absorb a future miss's
  // transfer, so they could over-prune. Bank/place/prep residuals stay (terms
  // unchanged). Standalone evaluate() honors req.m2 verbatim (credit from the
  // pins when present, else from the given assignment's cached placements).
  // All vectors sized num_devices; malformed sizes silently disable m2.
  bool m2 = false;
  std::vector<double> m2_s, m2_o0, m2_oc, m2_hsat, m2_cpw;
  double m2_gc = 1.0;

  // Optional domain restriction (hot-path decode): pinned[i] = device j to FIX
  // expert i on (-1 = free). Empty (default) = all free, identical to the
  // unrestricted solve. Intended use: pin cache HITS to their resident device —
  // the assignment ACT executes anyway (hits are never rerouted, cache-safe) —
  // so the exact B&B enumerates only the misses: M^misses candidates instead of
  // M^N (the M=4,N=8 decode shape drops 4^8=65536 -> 4^miss). The result is
  // EXACT over the restricted domain. When the free subset is too large for the
  // enumeration budget (prefill shapes), the pins are IGNORED and the solve
  // falls through to the unrestricted tiers (dp_full/dp_prefix) unchanged.
  std::vector<int> pinned;              // [N] or empty; entries in [-1, M)

  uint8_t cached_at(int i, int j) const { return cached[static_cast<size_t>(i) * num_devices + j]; }
  double  place_at(int i, int j) const {
    return place.empty() ? 0.0 : place[static_cast<size_t>(i) * num_devices + j];
  }
};

template <int NMax>
struct BasicSolveResult {
  std::array<int, NMax> assignment{};   // valid entries [0, n); j[i] in [0,M)
  int    n = 0;                         // number of valid assignment entries (== req.num_experts)
  double predicted_us = 0.0;            // T
  double device_makespan_us = 0.0;
  double bank_egress_us      = 0.0;
  double recon_us            = 0.0;
  double place_us            = 0.0;
  double evict_us            = 0.0;
  double prep_us             = 0.0;
  bool   exact               = true;    // false for the greedy-prefix / greedy paths
};
using SolveResult    = BasicSolveResult<kMaxExperts>;       // frozen 64 shape
using SolveResult256 = BasicSolveResult<kMaxExpertsLarge>;

// Preallocated, reusable solver. Hold one per daemon thread; methods reset their
// scratch each call. All buffers are bounded by kMax*/NMax — N>NMax,
// M>kMaxDevices, banks>kMaxBanks, or C>kMaxC trip an assertion.
// ONLY the two explicit instantiations (64/256, see the header note) exist;
// member definitions live in loader_solver.cpp.
template <int NMax>
class BasicLoaderSolver {
 public:
  using Result = BasicSolveResult<NMax>;
  Result solve(const LoaderConstants& k, const SolveRequest& req);
  Result solve_greedy(const LoaderConstants& k, const SolveRequest& req);
  // Full-objective evaluation of a fixed assignment. Fills *out when non-null.
  // Reads req.num_experts entries from the assignment. The vector overload
  // forwards to the pointer core (no length param: the core uses N_ = num_experts).
  double evaluate(const LoaderConstants& k, const SolveRequest& req,
                  const std::vector<int>& assignment, Result* out = nullptr);
  double evaluate(const LoaderConstants& k, const SolveRequest& req,
                  const int* assignment, Result* out = nullptr);

 private:
  Result bnb(const LoaderConstants& k, const SolveRequest& req);     // exact: DFS branch-and-bound
  // Pinned variant: req.pinned experts pre-applied as fixed assignments, DFS
  // levels over the free subset only (free_idx_[0..nfree_)). Exact over the
  // restricted domain; same lowest-index tie-break determinism as bnb().
  Result bnb_pinned(const LoaderConstants& k, const SolveRequest& req);
  void dfs_free(const LoaderConstants& k, const SolveRequest& req, int fi);
  // Residual (suffix) admissible lower-bound tables over the free experts
  // [fi..nfree_), precomputed once per bnb/bnb_pinned solve; lb_free(fi) folds
  // them into the node bound. SOUNDNESS (leaf-gate validity — the historical
  // leaf update is NOT strict-improving, see the note in loader_solver.cpp):
  // every residual term lower-bounds each descendant LEAF's GATE value, so a
  // pruned subtree contains only leaves that would have failed their own gate
  // — the gate-passer sequence, hence the returned assignment, is bit-exact.
  // See fill_free_residuals() for the per-term FP-admissibility slack/guards.
  void   fill_free_residuals(const LoaderConstants& k, const SolveRequest& req);
  double lb_free(const LoaderConstants& k, const SolveRequest& req, int fi) const;
  // Greedy warm-start incumbent VALUE (threshold-only, never the returned
  // assignment); sound because dfs_free's leaf update is strict-improving.
  double seed_incumbent(const LoaderConstants& k, const SolveRequest& req);
  // NOTE: the historical dfs(i) (naive-bound unpinned DFS) is retired — bnb()
  // now drives dfs_free with an identity free list (same order, tighter bound).
  Result dp_full(const LoaderConstants& k, const SolveRequest& req);
  Result dp_prefix(const LoaderConstants& k, const SolveRequest& req);
  // Large-union bounded greedy tier (tier 3' — reached only when NMax >
  // kMaxExperts): pinned hits are applied verbatim (resident device), then the
  // free experts (misses) are admitted greedily in index order, each on the
  // device minimizing the FULL incremental objective (makespan/fetch-load
  // balance, §3.3 bank floor, place = the reuse/victim-age reward, evict_cum,
  // prep) — the same signals the exact tiers rank by. Deterministic: index
  // order + strict < (lowest device index wins ties). O(nfree·M·(M+B)).
  Result greedy_pinned(const LoaderConstants& k, const SolveRequest& req);
  void dp_core(const LoaderConstants& k, const SolveRequest& req, const int* ex, int C,
               int* a);

  // shared incremental primitives (operate on the maintained per-device/per-bank sums)
  void   fill_sx(const LoaderConstants& k, const SolveRequest& req);  // precompute the N×M subxfer table
  void   reset_sums();
  void   apply(const LoaderConstants& k, const SolveRequest& req, int i, int d);  // place expert i on d
  void   undo (const LoaderConstants& k, const SolveRequest& req, int i, int d);  // reverse apply
  // lower_bound: use the optimistic (g_b=M_) §3.3 bank floor instead of the true
  // g_b=bdcnt_[b] — a valid lower bound for DFS pruning (the contention floor is
  // non-monotone in #assigned when c_b<1). Default false = true objective.
  double objective_from_sums(const LoaderConstants& k, const SolveRequest& req, Result* out,
                             bool lower_bound = false);
  bool   any_parallel_bank(const LoaderConstants& k) const;  // any bank contention<1

  // ── preallocated scratch (compile-time bounded; ~16 KiB at NMax=64) ──
  std::array<double, NMax * kMaxDevices>        sx_{};    // subxfer[i][j] (stride kMaxDevices)
  std::array<double, kMaxDevices * kDpStates>   rd_{};    // Rd[d][T]
  std::array<int,    kMaxDevices * kDpStates>   choice_{};// DP backpointers
  std::array<double, kDpStates>                 dp_{}, ndp_{}, sacc_{};
  std::array<double, kMaxDevices>               sub_{}, init_sub_{};
  std::array<int,    kMaxDevices>               cnt_{}, init_cnt_{}, nunc_{};
  std::array<double, kMaxBanks>                 egress_{};   // raw_sum_b (strict-serial draw)
  // Per-(bank,device) uncached-expert presence counts (stride kMaxDevices) and
  // the derived distinct-device count per bank (g_b, §3.3). bdcnt_[b]==g_b: the
  // number of devices with ≥1 uncached expert from bank b in the current state.
  std::array<int,    kMaxBanks * kMaxDevices>   bnd_{};
  std::array<int,    kMaxBanks>                 bdcnt_{};
  std::array<double, NMax>                      solo_{};
  std::array<int,    NMax>                      order_{}, cur_a_{}, best_a_{};
  std::array<int,    NMax>                      free_idx_{};  // pinned solve: free-expert indices
  // Symmetry chains (P-25 LASTPASSER re-baseline): prev_same_[f] = latest
  // earlier free slot whose expert is bitwise-indistinguishable under every
  // objective input; dfs_free explores device-sorted tuples across each chain.
  std::array<int,    NMax>                      prev_same_{};
  // Suffix residual lower-bound tables for the pinned DFS (see lb_free):
  //   res_bank_[fi*kMaxBanks+b] = guaranteed extra bank-b draw from free
  //     experts >= fi that are uncached on EVERY device (they must pull from
  //     their source bank wherever placed);
  //   res_ms_[fi]    = max static solo floor min_d(sx[i][d]+compute(d,1));
  //   res_place_[fi] = sum of per-expert min-device place contributions;
  //   res_prep_[fi]  = sum of the (device-independent) subprep terms.
  std::array<double, (NMax + 1) * kMaxBanks>        res_bank_{};
  std::array<double, NMax + 1>                      res_ms_{};
  std::array<double, NMax + 1>                      res_place_{};
  std::array<double, NMax + 1>                      res_prep_{};
  // Average-load makespan floor: res_minsx_[fi] = sum of min_d sx over the
  // remaining free experts; max_d R_d >= (sum_d sub_d + res_minsx_)/M. Valid
  // only under the same nonneg guards as res_ms_ (res_avg_ok_).
  std::array<double, NMax + 1>                      res_minsx_{};
  bool res_avg_ok_ = false;
  double res_min_a_ = 0.0;  // min per-expert linear compute (avg-floor addend)
  // Per-solve constants hoisted out of the per-node lb_free evaluation —
  // BITWISE-identical values (same pure-function inputs), just precomputed:
  //   fbank_[b]  = bank_egress_factor(contention_b, M_)   (lb-mode g = M_)
  //   comp_[d*(NMax+1)+c] = compute_us(devices[d].compute, c), c<=N_
  std::array<double, kMaxBanks>                     fbank_{};
  std::array<double, kMaxDevices * (NMax + 1)>      comp_{};
  // (A symmetry-breaking chain across indistinguishable free experts was tried
  // and REVERTED: swapped assignments are only equal up to FP re-association,
  // so the sorted-representative walk flips near-tie argmins — see dfs_free.)
  // M2 gate for THIS solve/evaluate (see SolveRequest::m2 docs): armed by
  // solve() on the pinned tier and by the public evaluate(); internal
  // evaluate_armed() calls never re-arm, so fall-through tiers stay legacy.
  bool m2_active_ = false;
  std::array<double, kMaxDevices> m2_credit_{};
  bool m2_valid(const SolveRequest& req) const;
  void m2_arm(const SolveRequest& req, const int* assignment);
  double evaluate_armed(const LoaderConstants& k, const SolveRequest& req,
                        const int* assignment, Result* out);

  double best_T_ = 0.0, place_acc_ = 0.0, prep_acc_ = 0.0;
  int    M_ = 0, N_ = 0, B_ = 0, nfree_ = 0;
};

// The ONLY two instantiations (defined in loader_solver.cpp):
//   LoaderSolver    — the frozen production 64-expert solver (REEF-calibrated).
//   LoaderSolver256 — batched-verify unions (>64, up to kMaxExpertsLarge).
extern template class BasicLoaderSolver<kMaxExperts>;
extern template class BasicLoaderSolver<kMaxExpertsLarge>;
using LoaderSolver    = BasicLoaderSolver<kMaxExperts>;
using LoaderSolver256 = BasicLoaderSolver<kMaxExpertsLarge>;

// Convenience free functions: construct a local LoaderSolver (stack, no heap).
// The daemon should hold a persistent LoaderSolver instead.
SolveResult solve(const LoaderConstants& k, const SolveRequest& req);
SolveResult solve_greedy(const LoaderConstants& k, const SolveRequest& req);
double      evaluate(const LoaderConstants& k, const SolveRequest& req,
                     const std::vector<int>& assignment, SolveResult* out = nullptr);

}  // namespace layerstorm::gpu_loader
