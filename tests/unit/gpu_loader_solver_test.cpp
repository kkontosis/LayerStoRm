// CPU unit tests for the GPU loader solver (spec/GPU_LOADER_MODEL.md §3/§5,
// ticket I8). Pure: synthetic LoaderConstants + SolveRequest -> assignment.
#include "core/gpu_loader/loader_solver.h"
#include "core/gpu_loader/loader_constants.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace gl = layerstorm::gpu_loader;

namespace {

// M devices (device d local to bank d), B banks. Linear compute (a=150,b=0),
// uniform transfer rate=500 + lat=5, egress=egress_us. Tests override cells.
gl::LoaderConstants make_constants(int M, int B, double egress_us = 50.0) {
  gl::LoaderConstants k;
  k.source       = "test";
  k.expert_bytes = 24772992.0;
  k.num_devices  = M;
  k.num_banks    = B;
  k.ncf          = {0.0, 1.0, 1.2, 1.5};
  for (int d = 0; d < M; ++d) {
    gl::DeviceConstants dc;
    dc.position    = d;
    dc.numa_node   = d;
    dc.xfer_lat_us = 5.0;
    dc.compute     = {150.0, 0.0, 1};  // seed: pure linear 150 us/expert
    k.devices.push_back(dc);
  }
  for (int b = 0; b < B; ++b) {
    gl::BankConstants bc;
    bc.node       = b;
    bc.egress_us  = egress_us;
    bc.contention = 1.0;
    k.banks.push_back(bc);
  }
  k.matrix.assign(B, std::vector<gl::TransferCell>(M));
  for (int b = 0; b < B; ++b)
    for (int d = 0; d < M; ++d) k.matrix[b][d] = gl::TransferCell{500.0, (b == d ? 1 : 2), 5.0};
  return k;
}

// N experts, all from bank `bank`, all uncached by default.
gl::SolveRequest make_request(int M, int N, int bank = 0) {
  gl::SolveRequest req;
  req.num_devices = M;
  req.num_experts = N;
  req.bank_of.assign(N, bank);
  req.cached.assign(static_cast<size_t>(N) * M, 0);
  return req;
}

std::vector<int> counts(const std::vector<int>& a, int M) {
  std::vector<int> c(M, 0);
  for (int j : a) ++c[j];
  return c;
}

// Valid-prefix view of a SolveResult's fixed assignment array as a vector
// (the assignment is std::array<int, kMaxExperts>; only [0, n) is valid).
std::vector<int> av(const gl::SolveResult& r) {
  return {r.assignment.begin(), r.assignment.begin() + r.n};
}

}  // namespace

TEST(GpuLoaderSolver, ComputeCurveBatchStep) {
  EXPECT_DOUBLE_EQ(gl::compute_us({150.0, 0.0, 1}, 0), 0.0);
  EXPECT_DOUBLE_EQ(gl::compute_us({150.0, 0.0, 1}, 3), 450.0);          // pure linear
  EXPECT_DOUBLE_EQ(gl::compute_us({100.0, 40.0, 2}, 3), 100.0 * 3 + 40.0 * 2);  // ceil(3/2)=2 batches
  EXPECT_DOUBLE_EQ(gl::compute_us({100.0, 40.0, 2}, 4), 100.0 * 4 + 40.0 * 2);  // ceil(4/2)=2
  EXPECT_DOUBLE_EQ(gl::compute_us({100.0, 40.0, 2}, 5), 100.0 * 5 + 40.0 * 3);  // ceil(5/2)=3
}

TEST(GpuLoaderSolver, EvaluateMatchesManual) {
  const auto k = make_constants(/*M=*/1, /*B=*/1, /*egress=*/50.0);
  auto req = make_request(1, 1);
  const double T = gl::evaluate(k, req, {0});
  // 1 uncached expert on dev0: subxfer = 500+5, compute(1)=150, egress floor = 50.
  // max(505+150, 50) = 655.
  EXPECT_DOUBLE_EQ(T, 655.0);
}

TEST(GpuLoaderSolver, AllCachedNoTransfers) {
  auto k = make_constants(2, 2, /*egress=*/400.0);
  auto req = make_request(2, 2);
  for (auto& c : req.cached) c = 1;  // both experts cached on both devices
  const auto r = gl::solve(k, req);
  EXPECT_DOUBLE_EQ(r.bank_egress_us, 0.0);  // nothing fetched
  EXPECT_EQ(counts(av(r), 2), (std::vector<int>{1, 1}));  // balance compute -> split
  EXPECT_DOUBLE_EQ(r.predicted_us, 150.0);  // compute(1) per device, no transfer/floor
  EXPECT_TRUE(r.exact);
}

TEST(GpuLoaderSolver, StaggerLoadBalance) {
  // 4 uncached experts, 2 devices, symmetric -> split 2/2 (device makespan is the
  // binder; egress small). This is the cross-GPU stagger fix in miniature.
  auto k = make_constants(2, 2, /*egress=*/10.0);
  auto req = make_request(2, 4, /*bank=*/0);
  const auto r = gl::solve(k, req);
  EXPECT_EQ(counts(av(r), 2), (std::vector<int>{2, 2}));
}

TEST(GpuLoaderSolver, PrefersCheaperDevice) {
  auto k = make_constants(2, 1, /*egress=*/10.0);
  k.matrix[0][0].rate_us = 300.0;  // bank0 cheap to dev0
  k.matrix[0][1].rate_us = 900.0;  // expensive to dev1
  auto req = make_request(2, 1, /*bank=*/0);
  const auto r = gl::solve(k, req);
  EXPECT_EQ(r.assignment[0], 0);
}

TEST(GpuLoaderSolver, BankEgressFloorDominates) {
  // 4 experts all from bank0, large egress -> bank_egress = 4*400 = 1600 floors T,
  // independent of device split; solver still balances devices (2/2).
  auto k = make_constants(2, 2, /*egress=*/400.0);
  auto req = make_request(2, 4, /*bank=*/0);
  const auto r = gl::solve(k, req);
  EXPECT_DOUBLE_EQ(r.bank_egress_us, 1600.0);
  EXPECT_DOUBLE_EQ(r.predicted_us, 1600.0);              // floored by the bank
  EXPECT_EQ(counts(av(r), 2), (std::vector<int>{2, 2}));
}

TEST(GpuLoaderSolver, BankEgressIsMaxOverBanks) {
  // 3 experts on bank0, 1 on bank1 -> floor = max(3*400, 1*400) = 1200.
  auto k = make_constants(2, 2, /*egress=*/400.0);
  auto req = make_request(2, 4);
  req.bank_of = {0, 0, 0, 1};
  const auto r = gl::solve(k, req);
  EXPECT_DOUBLE_EQ(r.bank_egress_us, 1200.0);
}

TEST(GpuLoaderSolver, DeterministicLowestIndexTieBreak) {
  auto k = make_constants(2, 1, /*egress=*/10.0);  // dev0/dev1 identical for bank0
  auto req = make_request(2, 1, /*bank=*/0);
  const auto r1 = gl::solve(k, req);
  const auto r2 = gl::solve(k, req);
  EXPECT_EQ(av(r1), av(r2));
  EXPECT_EQ(r1.assignment[0], 0) << "symmetric tie must pick the lowest device index";
}

TEST(GpuLoaderSolver, ExactFindsKnownOptimum) {
  // Construct an instance whose optimum balances compute despite a cheaper-but-
  // overloading option, and verify solve() <= every other assignment.
  auto k = make_constants(2, 1, /*egress=*/10.0);
  auto req = make_request(2, 4, /*bank=*/0);
  const auto r = gl::solve(k, req);
  // brute reference over all 2^4 assignments
  double best = 1e300;
  for (int m = 0; m < 16; ++m) {
    std::vector<int> a(4);
    for (int i = 0; i < 4; ++i) a[i] = (m >> i) & 1;
    best = std::min(best, gl::evaluate(k, req, a));
  }
  EXPECT_DOUBLE_EQ(r.predicted_us, best);
  EXPECT_TRUE(r.exact);
}

TEST(GpuLoaderSolver, EvictAndPlaceConsequence) {
  auto k = make_constants(2, 1, /*egress=*/10.0);
  auto req = make_request(2, 2, /*bank=*/0);
  // place_cons: routing expert 0 to dev1 carries +1000 penalty -> avoid.
  req.place.assign(2 * 2, 0.0);
  req.place[0 * 2 + 1] = 1000.0;
  // eviction: each device's first eviction costs 7 (prefix sum [0,7,...]).
  req.evict_cum = {{0.0, 7.0, 7.0}, {0.0, 7.0, 7.0}};
  const auto r = gl::solve(k, req);
  EXPECT_NE(r.assignment[0], 1) << "place_cons penalty must steer expert 0 off dev1";
  EXPECT_GT(r.evict_us, 0.0);  // two uncached fetches -> evictions charged
}

// I8 P2 (shadow_solve_and_log evict_cum wiring): on a bank-/makespan-tied instance
// a populated CONVEX evict_cum[j][n] must break the tie toward a BALANCED split
// instead of overloading one GPU. Without evict_cum the deterministic lowest-index
// tie-break piles all misses onto dev0 (8/0); a per-victim eviction cost that rises
// with the number of evictions (convex) makes 4/4 strictly cheaper than 8/0.
TEST(GpuLoaderSolver, ConvexEvictCumBalancesOverOverload) {
  // 2 devices, 8 uncached experts, all from bank0. Tiny transfer/compute, huge
  // egress -> bank_egress (8*1000=8000) floors T for EVERY split (makespan <= 48 in
  // the worst 8/0 case), so makespan/bank are assignment-invariant. evict_cum is the
  // ONLY term that distinguishes splits.
  auto k = make_constants(2, 1, /*egress=*/1000.0);
  for (auto& dc : k.devices) dc.compute = {1.0, 0.0, 1};  // ~flat, well under the floor
  for (auto& row : k.matrix) for (auto& c : row) { c.rate_us = 5.0; c.lat_us = 0.0; }
  auto req = make_request(2, 8, /*bank=*/0);

  // Baseline: no evict_cum -> tie -> lowest-index overload (8/0).
  {
    const auto r0 = gl::solve(k, req);
    EXPECT_EQ(counts(av(r0), 2), (std::vector<int>{8, 0}))
        << "without evict_cum the tie collapses onto dev0";
  }

  // Convex per-device curve: fs=0; each successive eviction costs MORE than the last
  // (rising per-victim cost = sorted-cheapest-first). N=8 -> length N+1=9.
  // step_n = 10*n  -> cum[n] = 10*(1+..+n) (strictly convex). Both devices identical.
  std::vector<double> cum(9, 0.0);
  double acc = 0.0;
  for (int n = 1; n <= 8; ++n) { acc += 10.0 * n; cum[n] = acc; }
  req.evict_cum = {cum, cum};
  // Sanity: the curve is non-decreasing and convex (increments rise).
  for (int n = 1; n <= 8; ++n) EXPECT_GE(cum[n], cum[n - 1]);
  for (int n = 2; n <= 8; ++n)
    EXPECT_GE(cum[n] - cum[n - 1], cum[n - 1] - cum[n - 2] - 1e-9);

  const auto r = gl::solve(k, req);
  EXPECT_EQ(counts(av(r), 2), (std::vector<int>{4, 4}))
      << "convex evict_cum must steer the solver to a balanced 4/4 split";
  EXPECT_GT(r.evict_us, 0.0);
  // 4/4 evict (2*cum[4]) is strictly below the 8/0 evict (cum[8]).
  EXPECT_LT(2.0 * cum[4], cum[8]);
}

namespace {

// Realistic heterogeneous instance: N=8 routed experts, M=10 devices, 4 NUMA
// banks. Devices have varied compute; one device (dev0) is PCIe-capped (slow on
// every bank); each device is NUMA-local to bank d%4 (cheap) and remote elsewhere
// (expensive). Two experts are already cached. M^N = 10^8 > enumeration budget
// AND N=8 > kMaxC(6), so solve() takes the greedy-prefix path (LPT freezes the
// 2 costliest, exact DP on the cheapest 6, take the better of that and greedy).
gl::LoaderConstants make_n8_m10_constants() {
  gl::LoaderConstants k;
  k.source       = "test-n8m10";
  k.expert_bytes = 24772992.0;
  k.num_devices  = 10;
  k.num_banks    = 4;
  k.ncf          = {0.0, 1.0, 1.22, 1.5};
  for (int d = 0; d < 10; ++d) {
    gl::DeviceConstants dc;
    dc.position    = d;
    dc.numa_node   = d % 4;
    dc.xfer_lat_us = 5.0;
    dc.compute     = {120.0 + 8.0 * d, 0.0, 1};  // mild per-device variation
    k.devices.push_back(dc);
  }
  const double egress[4] = {180.0, 190.0, 200.0, 210.0};
  for (int b = 0; b < 4; ++b) {
    gl::BankConstants bc;
    bc.node = b;
    bc.egress_us = egress[b];
    k.banks.push_back(bc);
  }
  k.matrix.assign(4, std::vector<gl::TransferCell>(10));
  for (int b = 0; b < 4; ++b)
    for (int d = 0; d < 10; ++d) {
      const bool local = (d % 4 == b);
      double rate = local ? 480.0 : 660.0;
      if (d == 0) rate = 820.0;  // dev0 PCIe-capped on every bank
      k.matrix[b][d] = gl::TransferCell{rate, local ? 1 : 2, 5.0};
    }
  return k;
}

}  // namespace

// Full worked example + greedy comparison + timing. Prints both solutions and
// the per-call solve time (the hot-path budget); asserts the exact solver is
// never worse than the forward greedy.
TEST(GpuLoaderSolver, N8M10ExampleVsGreedyAndBenchmark) {
  const auto k = make_n8_m10_constants();
  gl::SolveRequest req;
  req.num_devices = 10;
  req.num_experts = 8;
  req.bank_of.resize(8);
  for (int i = 0; i < 8; ++i) req.bank_of[i] = i % 4;  // 2 experts per bank
  req.cached.assign(8 * 10, 0);
  req.cached[0 * 10 + 0] = 1;  // expert 0 already resident on dev0
  req.cached[1 * 10 + 1] = 1;  // expert 1 already resident on dev1

  const gl::SolveResult sv = gl::solve(k, req);
  const gl::SolveResult gr = gl::solve_greedy(k, req);

  auto print = [](const char* tag, const gl::SolveResult& r) {
    fprintf(stderr, "[N8M10 %-6s] T=%.1f us  (makespan=%.1f floor=%.1f recon=%.1f place=%.1f evict=%.1f)  exact=%d\n  j[.] =",
            tag, r.predicted_us, r.device_makespan_us, r.bank_egress_us, r.recon_us, r.place_us, r.evict_us,
            static_cast<int>(r.exact));
    for (int j : av(r)) fprintf(stderr, " %d", j);
    fprintf(stderr, "\n");
  };
  print("solver", sv);
  print("greedy", gr);
  fprintf(stderr, "[N8M10] solver/greedy T ratio = %.4f  (lower solver is better/equal)\n",
          gr.predicted_us > 0 ? sv.predicted_us / gr.predicted_us : 1.0);

  EXPECT_FALSE(sv.exact) << "N=8 > kMaxC=6 -> greedy-prefix (heuristic) path";
  for (int j : av(sv)) { EXPECT_GE(j, 0); EXPECT_LT(j, 10); }
  EXPECT_LE(sv.predicted_us, gr.predicted_us + 1e-6) << "prefix path must not be worse than greedy";

  // Benchmark: 10 iterations each (the solver runs every token x layer). NOTE:
  // single-digit-µs timings are noisy at this count — a one-off 5000-iter run
  // measured the stable cost at ~12 µs/solve (prefix) / ~4 µs (greedy).
  constexpr int kIters = 10;
  auto bench = [&](gl::SolveResult (*fn)(const gl::LoaderConstants&, const gl::SolveRequest&)) {
    volatile double sink = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < kIters; ++it) sink += fn(k, req).predicted_us;
    const auto t1 = std::chrono::steady_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
  };
  const double solver_us = bench(gl::solve);          // free fn: constructs a solver per call
  const double greedy_us = bench(gl::solve_greedy);
  // Persistent solver (how the daemon uses it): construct once, reuse — isolates
  // the per-call construction (16 KiB std::array zero-init) from the algorithm.
  gl::LoaderSolver persistent;
  volatile double sink = 0.0;
  const auto p0 = std::chrono::steady_clock::now();
  for (int it = 0; it < kIters; ++it) sink += persistent.solve(k, req).predicted_us;
  const auto p1 = std::chrono::steady_clock::now();
  (void)sink;
  const double persist_us = std::chrono::duration<double, std::micro>(p1 - p0).count();
  fprintf(stderr, "[N8M10 bench, %d iters] solver(free, per-call ctor) %.2f us/solve\n", kIters, solver_us / kIters);
  fprintf(stderr, "[N8M10 bench, %d iters] solver(persistent, reused)  %.2f us/solve\n", kIters, persist_us / kIters);
  fprintf(stderr, "[N8M10 bench, %d iters] greedy                      %.2f us/solve\n", kIters, greedy_us / kIters);

  // The representative daemon path: persistent solver + populated convex evict_cum,
  // with the §3.3 contention term — at c=1 (serial, factor=1 no-op) and c=0
  // (parallel, /g_b active). The g_b tracking + O(B) floor recompute run in both;
  // a 200k-iter one-off measured ~5.45 us/solve for both (c-invariant). kIters=10
  // here is noisy — bump locally for a stable number.
  gl::SolveRequest req_ev = req;
  {
    std::vector<double> cum(9, 0.0);
    const double inc[8] = {100, 150, 200, 250, 300, 350, 400, 450};
    for (int n = 1; n <= 8; ++n) cum[n] = cum[n - 1] + inc[n - 1];
    req_ev.evict_cum.assign(10, cum);
  }
  auto bench_p = [&](const gl::LoaderConstants& kk, const gl::SolveRequest& rr) {
    volatile double s = 0.0;
    const auto a = std::chrono::steady_clock::now();
    for (int it = 0; it < kIters; ++it) s += persistent.solve(kk, rr).predicted_us;
    const auto b = std::chrono::steady_clock::now();
    (void)s;
    return std::chrono::duration<double, std::micro>(b - a).count() / kIters;
  };
  gl::LoaderConstants k_par = k;
  for (auto& bk : k_par.banks) bk.contention = 0.0;
  fprintf(stderr, "[N8M10 bench, %d iters] persistent +evict_cum (c=1 serial)   %.2f us/solve\n",
          kIters, bench_p(k, req_ev));
  fprintf(stderr, "[N8M10 bench, %d iters] persistent +evict_cum (c=0 parallel) %.2f us/solve\n",
          kIters, bench_p(k_par, req_ev));
}

// Greedy-prefix path (N=20 > kMaxC): valid full assignment, heuristic, never
// worse than greedy.
TEST(GpuLoaderSolver, PrefixPathN20M10Valid) {
  const auto k = make_constants(10, 4, /*egress=*/60.0);
  gl::SolveRequest req;
  req.num_devices = 10;
  req.num_experts = 20;
  req.bank_of.resize(20);
  for (int i = 0; i < 20; ++i) req.bank_of[i] = i % 4;
  req.cached.assign(20 * 10, 0);
  const auto sv = gl::solve(k, req);
  const auto gr = gl::solve_greedy(k, req);
  EXPECT_FALSE(sv.exact);
  ASSERT_EQ(sv.n, 20);
  for (int j : av(sv)) { EXPECT_GE(j, 0); EXPECT_LT(j, 10); }
  EXPECT_GT(sv.predicted_us, 0.0);
  EXPECT_LE(sv.predicted_us, gr.predicted_us + 1e-6);
}

// Incremental greedy cross-check: on a separable instance (each expert uniquely
// cheapest on its own device, N<M) the forward greedy is optimal, so it must
// match the exact solver. Guards the opt-2 incremental rewrite.
TEST(GpuLoaderSolver, GreedyMatchesExactWhenSeparable) {
  auto k = make_constants(5, 4, /*egress=*/10.0);  // local rate 500, cross 500 -> make local cheaper:
  for (int b = 0; b < 4; ++b)
    for (int d = 0; d < 5; ++d) k.matrix[b][d].rate_us = (b == d) ? 300.0 : 900.0;
  gl::SolveRequest req;
  req.num_devices = 5;
  req.num_experts = 3;
  req.bank_of = {0, 1, 2};  // expert i uniquely cheapest on device i
  req.cached.assign(3 * 5, 0);
  const auto sv = gl::solve(k, req);        // M^N=125 <= budget -> exact B&B
  const auto gr = gl::solve_greedy(k, req);
  EXPECT_TRUE(sv.exact);
  EXPECT_NEAR(gr.predicted_us, sv.predicted_us, 1e-6) << "greedy optimal on separable instance";
  EXPECT_EQ(av(gr), (std::vector<int>{0, 1, 2}));
}

namespace {

// ── Reference forward greedy: a literal copy of the ORIGINAL solve_greedy.
// It maintains the per-device/per-bank sums incrementally and scores each
// candidate via apply / objective-from-sums / undo — exactly the arithmetic
// (including FP add/undo residue) the original solver used. This is the
// independent oracle the incremental max-tracking rewrite must reproduce
// bit-identically.
//
// State holder mirrors LoaderSolver's maintained sums.
struct RefState {
  std::vector<double> sub, egress;
  std::vector<int>    cnt, nunc;
  std::vector<std::vector<int>> bdev;  // bdev[b][d] = #uncached experts from bank b on dev d
  double prep_acc = 0.0, place_acc = 0.0;
};

double ref_compute_us(const gl::ComputeCurve& cc, int c) {
  if (c <= 0) return 0.0;
  const int P = cc.P > 0 ? cc.P : 1;
  const int batches = (P == 1) ? c : 1 + (c - 1) / P;
  return cc.a_us * static_cast<double>(c) + cc.b_us * static_cast<double>(batches);
}

double ref_subxfer(const gl::LoaderConstants& k, const gl::SolveRequest& req, int i, int d) {
  if (req.cached_at(i, d)) return 0.0;
  const gl::TransferCell& cell = k.matrix[req.bank_of[i]][d];
  return cell.rate_us + cell.lat_us;
}

void ref_apply(const gl::LoaderConstants& k, const gl::SolveRequest& req, RefState& s, int i, int d) {
  s.cnt[d] += 1;
  if (!req.subprep_us.empty()) s.prep_acc += req.subprep_us[i];
  if (!req.cached_at(i, d)) {
    s.sub[d] += ref_subxfer(k, req, i, d);
    s.egress[req.bank_of[i]] += k.banks[req.bank_of[i]].egress_us;
    s.bdev[req.bank_of[i]][d] += 1;
    s.nunc[d] += 1;
    double p = req.place_at(i, d);
    if (req.clamp_place) p = std::max(0.0, p);
    s.place_acc += p;
  }
}

void ref_undo(const gl::LoaderConstants& k, const gl::SolveRequest& req, RefState& s, int i, int d) {
  s.cnt[d] -= 1;
  if (!req.subprep_us.empty()) s.prep_acc -= req.subprep_us[i];
  if (!req.cached_at(i, d)) {
    s.sub[d] -= ref_subxfer(k, req, i, d);
    s.egress[req.bank_of[i]] -= k.banks[req.bank_of[i]].egress_us;
    s.bdev[req.bank_of[i]][d] -= 1;
    s.nunc[d] -= 1;
    double p = req.place_at(i, d);
    if (req.clamp_place) p = std::max(0.0, p);
    s.place_acc -= p;
  }
}

double ref_objective(const gl::LoaderConstants& k, const gl::SolveRequest& req, const RefState& s) {
  const int M = req.num_devices;
  const int B = k.num_banks;
  double makespan = 0.0, ov = 0.0, ad = 0.0;
  int participants = 0;
  for (int d = 0; d < M; ++d) {
    if (s.cnt[d] == 0) continue;
    makespan = std::max(makespan, s.sub[d] + ref_compute_us(k.devices[d].compute, s.cnt[d]));
    ov = std::max(ov, k.devices[d].recon_overhead_us);
    ad += k.devices[d].recon_added_us;
    ++participants;
  }
  // §3.3 contention-aware floor: max_b raw_sum_b·(c_b + (1−c_b)/g_b), g_b = distinct
  // devices holding an uncached expert from bank b. c=1 -> max_b egress[b] (old).
  double bank_egress = 0.0;
  for (int b = 0; b < B; ++b) {
    if (s.egress[b] <= 0.0) continue;
    int g = 0;
    for (int d = 0; d < M; ++d) if (s.bdev[b][d] > 0) ++g;
    double c = k.banks[b].contention;
    if (c < 0.0) c = 0.0;
    if (c > 1.0) c = 1.0;
    if (g < 1) g = 1;
    bank_egress = std::max(bank_egress, s.egress[b] * (c + (1.0 - c) / static_cast<double>(g)));
  }
  const double recon = (participants > 0) ? ov + ad : 0.0;
  double evict = 0.0;
  if (!req.evict_cum.empty()) {
    for (int d = 0; d < M; ++d) {
      const std::vector<double>& cum = req.evict_cum[d];
      if (cum.empty()) continue;
      const int en = std::min(s.nunc[d], static_cast<int>(cum.size()) - 1);
      if (en > 0) evict += cum[en];
    }
  }
  return s.prep_acc + std::max(makespan, bank_egress) + recon + s.place_acc + evict;
}

// Reference forward list-scheduling greedy (literal copy of original solve_greedy).
std::pair<std::vector<int>, double> ref_greedy(const gl::LoaderConstants& k,
                                               const gl::SolveRequest& req) {
  const int M = req.num_devices;
  const int N = req.num_experts;
  RefState s;
  s.sub.assign(M, 0.0);
  s.egress.assign(k.num_banks, 0.0);
  s.cnt.assign(M, 0);
  s.nunc.assign(M, 0);
  s.bdev.assign(k.num_banks, std::vector<int>(M, 0));
  std::vector<int> a(N, 0);
  for (int i = 0; i < N; ++i) {
    int best_d = 0;
    double best = 0.0;
    for (int d = 0; d < M; ++d) {
      ref_apply(k, req, s, i, d);
      const double t = ref_objective(k, req, s);
      ref_undo(k, req, s, i, d);
      if (d == 0 || t < best) { best = t; best_d = d; }
    }
    a[i] = best_d;
    ref_apply(k, req, s, i, best_d);  // commit
  }
  // predicted_us via the standalone evaluate (matches solve_greedy's final eval).
  return {a, gl::evaluate(k, req, a, nullptr)};
}

}  // namespace

// Randomized cross-check: the incremental greedy must produce a BIT-IDENTICAL
// assignment and predicted_us to the reference full-recompute greedy across many
// random instances — both with and without the consequence terms (place/evict/
// recon). This is the correctness oracle for the incremental max-tracking rewrite.
TEST(GpuLoaderSolver, GreedyMatchesReferenceRandomized) {
  std::mt19937_64 rng(0xC0FFEEULL);
  auto uni  = [&](double lo, double hi) {
    return std::uniform_real_distribution<double>(lo, hi)(rng);
  };
  auto uniI = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };

  // Many instances: the small integer-ish rates intentionally produce frequent
  // ULP-level ties, exercising the FP-residue / prep-rounding tie paths the
  // incremental rewrite must reproduce bit-for-bit.
  constexpr int kInstances = 20000;
  for (int inst = 0; inst < kInstances; ++inst) {
    const int N = uniI(1, 14);
    const int M = uniI(1, 10);
    const int B = uniI(1, 4);
    const bool with_consequence = (inst % 2 == 1);
    const bool clamp_place      = (uniI(0, 1) == 1);
    // A third of instances use small integer-valued constants so exact ties (and
    // the residue/prep rounding around them) occur very frequently.
    const bool tie_heavy        = (inst % 3 == 0);
    auto rate = [&](double lo, double hi) {
      return tie_heavy ? static_cast<double>(uniI(static_cast<int>(lo), static_cast<int>(hi)))
                       : uni(lo, hi);
    };

    gl::LoaderConstants k;
    k.source       = "rand";
    k.expert_bytes = 1.0;
    k.num_devices  = M;
    k.num_banks    = B;
    k.ncf          = {0.0, 1.0, 1.2, 1.5};
    for (int d = 0; d < M; ++d) {
      gl::DeviceConstants dc;
      dc.position    = d;
      dc.numa_node   = d % B;
      dc.xfer_lat_us = rate(0.0, 10.0);
      dc.compute     = {rate(50.0, 200.0), rate(0.0, 60.0), uniI(1, 4)};
      // recon terms: nonzero only in the with_consequence half (and randomly).
      if (with_consequence) {
        dc.recon_overhead_us = (uniI(0, 1) ? rate(0.0, 30.0) : 0.0);
        dc.recon_added_us    = (uniI(0, 1) ? rate(0.0, 20.0) : 0.0);
      }
      k.devices.push_back(dc);
    }
    // Exercise the §3.3 contention factor: half the instances keep the serial
    // c=1 (no-op) baseline; the other half draw a random c∈[0,1] per bank, so the
    // greedy fast path's per-candidate O(B) floor recompute is cross-checked
    // against the independent reference at every parallelism degree.
    const bool with_contention = (inst % 2 == 0);
    for (int b = 0; b < B; ++b) {
      gl::BankConstants bc;
      bc.node       = b;
      bc.egress_us  = rate(0.0, 400.0);
      bc.contention = with_contention ? uni(0.0, 1.0) : 1.0;
      k.banks.push_back(bc);
    }
    k.matrix.assign(B, std::vector<gl::TransferCell>(M));
    for (int b = 0; b < B; ++b)
      for (int d = 0; d < M; ++d)
        k.matrix[b][d] = gl::TransferCell{rate(100.0, 1000.0), (d % B == b ? 1 : 2), rate(0.0, 10.0)};

    gl::SolveRequest req;
    req.num_devices = M;
    req.num_experts = N;
    req.clamp_place = clamp_place;
    req.bank_of.resize(N);
    for (int i = 0; i < N; ++i) req.bank_of[i] = uniI(0, B - 1);
    req.cached.assign(static_cast<size_t>(N) * M, 0);
    for (auto& c : req.cached) c = (uniI(0, 3) == 0) ? 1 : 0;  // ~25% cached
    // random subprep (j-independent NVMe stage), sometimes empty.
    if (uniI(0, 1)) {
      req.subprep_us.resize(N);
      for (int i = 0; i < N; ++i) req.subprep_us[i] = rate(0.0, 50.0);
    }
    if (with_consequence) {
      req.place.assign(static_cast<size_t>(N) * M, 0.0);
      for (auto& p : req.place) p = uni(-30.0, 200.0);  // signed -> exercises clamp
      req.evict_cum.resize(M);
      for (int d = 0; d < M; ++d) {
        if (uniI(0, 1)) { req.evict_cum[d].clear(); continue; }  // some devices have none
        const int len = uniI(1, N + 1);
        req.evict_cum[d].assign(len + 1, 0.0);
        double acc = 0.0;
        for (int n = 1; n <= len; ++n) { acc += uni(0.0, 15.0); req.evict_cum[d][n] = acc; }
      }
    }

    const auto [ref_a, ref_T] = ref_greedy(k, req);
    const auto got = gl::solve_greedy(k, req);

    ASSERT_EQ(av(got), ref_a)
        << "instance " << inst << " (N=" << N << " M=" << M << " B=" << B
        << " consequence=" << with_consequence << ")";
    ASSERT_DOUBLE_EQ(got.predicted_us, ref_T)
        << "instance " << inst << " (N=" << N << " M=" << M << " B=" << B
        << " consequence=" << with_consequence << ")";
  }
}

// Exact path at N == kMaxC, M=14: solved exactly (full DP when M^N exceeds the
// enumeration budget, else enumeration — both set exact=true). Robust to kMaxC.
TEST(GpuLoaderSolver, ExactPathAtKMaxC) {
  const int N = gl::kMaxC;
  const auto k = make_constants(14, 4, /*egress=*/60.0);
  gl::SolveRequest req;
  req.num_devices = 14;
  req.num_experts = N;
  req.bank_of.resize(N);
  for (int i = 0; i < N; ++i) req.bank_of[i] = i % 4;
  req.cached.assign(static_cast<size_t>(N) * 14, 0);
  const auto sv = gl::solve(k, req);
  EXPECT_TRUE(sv.exact) << "N == kMaxC must be solved exactly";
  const auto gr = gl::solve_greedy(k, req);
  EXPECT_LE(sv.predicted_us, gr.predicted_us + 1e-6);
}

// ── §3.3 contention-aware bank-egress term ───────────────────────────────────
//
// Independent ground-truth objective: recomputes T = prep + max(makespan, floor)
// + recon + place + evict + fixed_overhead from an assignment using the §3.3
// contention factor max_b raw_sum_b·(c_b + (1−c_b)/g_b) DIRECTLY (no solver
// internals). This oracle verifies BOTH gl::evaluate AND the solved assignment.
namespace {
double brute_objective(const gl::LoaderConstants& k, const gl::SolveRequest& req,
                       const std::vector<int>& a) {
  const int M = req.num_devices, N = req.num_experts, B = k.num_banks;
  std::vector<double> sub(M, 0.0), egress(B, 0.0);
  std::vector<int>    cnt(M, 0), nunc(M, 0);
  std::vector<std::vector<int>> bdev(B, std::vector<int>(M, 0));  // bdev[b][d] presence
  double prep = 0.0, place = 0.0;
  for (int i = 0; i < N; ++i) {
    const int d = a[i], b = req.bank_of[i];
    cnt[d] += 1;
    if (!req.subprep_us.empty()) prep += req.subprep_us[i];
    if (!req.cached_at(i, d)) {
      sub[d] += ref_subxfer(k, req, i, d);
      egress[b] += k.banks[b].egress_us;
      bdev[b][d] += 1;
      nunc[d] += 1;
      double p = req.place_at(i, d);
      if (req.clamp_place) p = std::max(0.0, p);
      place += p;
    }
  }
  double makespan = 0.0, ov = 0.0, ad = 0.0;
  int participants = 0;
  for (int d = 0; d < M; ++d) {
    if (cnt[d] == 0) continue;
    makespan = std::max(makespan, sub[d] + ref_compute_us(k.devices[d].compute, cnt[d]));
    ov = std::max(ov, k.devices[d].recon_overhead_us);
    ad += k.devices[d].recon_added_us;
    ++participants;
  }
  double floor = 0.0;
  for (int b = 0; b < B; ++b) {
    if (egress[b] <= 0.0) continue;
    int g = 0;
    for (int d = 0; d < M; ++d) if (bdev[b][d] > 0) ++g;
    double c = k.banks[b].contention;
    if (c < 0.0) c = 0.0;
    if (c > 1.0) c = 1.0;
    if (g < 1) g = 1;
    floor = std::max(floor, egress[b] * (c + (1.0 - c) / static_cast<double>(g)));
  }
  const double recon = (participants > 0) ? ov + ad : 0.0;
  double evict = 0.0;
  if (!req.evict_cum.empty()) {
    for (int d = 0; d < M; ++d) {
      const std::vector<double>& cum = req.evict_cum[d];
      if (cum.empty()) continue;
      const int en = std::min(nunc[d], static_cast<int>(cum.size()) - 1);
      if (en > 0) evict += cum[en];
    }
  }
  return prep + std::max(makespan, floor) + recon + place + evict + k.fixed_overhead_us;
}
}  // namespace

// Test 1: at c_b=0 (perfectly parallel bank channel) the §3.3 term REWARDS using
// more devices — the bank's draw is divided by g_b (distinct target devices). 4
// experts, one bank, 2 devices, egress so large the bank floor dominates: the
// solver must SPREAD across BOTH devices (g=2 -> floor = raw/2), instead of the
// c=1 single-device pile (g=1 -> full raw_sum floor). The factor depends on the
// device COUNT g (not the balance), so 2/2 and 3/1 are floor-equivalent at c=0;
// the load-balancing comes from makespan/ties — here we assert the floor halves
// and both devices are used.
TEST(GpuLoaderSolver, ContentionParallelBankRewardsSpread) {
  auto k = make_constants(2, 1, /*egress=*/1000.0);  // huge per-expert egress
  // tiny transfer + compute so the bank floor (not makespan) binds for every split.
  for (auto& dc : k.devices) dc.compute = {1.0, 0.0, 1};
  for (auto& row : k.matrix) for (auto& c : row) { c.rate_us = 5.0; c.lat_us = 0.0; }
  auto req = make_request(2, 4, /*bank=*/0);

  // c=0: parallel -> spreading to both devices halves the bank term. raw_sum =
  // 4*1000 = 4000; g=2 -> floor = 4000/2 = 2000.
  k.banks[0].contention = 0.0;
  const auto rp = gl::solve(k, req);
  const auto cp = counts(av(rp), 2);
  EXPECT_GT(cp[0], 0); EXPECT_GT(cp[1], 0)
      << "parallel bank (c=0): solver must use BOTH devices to halve the bank term";
  EXPECT_DOUBLE_EQ(rp.bank_egress_us, 2000.0);  // raw_sum / g = 4000 / 2
  EXPECT_DOUBLE_EQ(rp.predicted_us, brute_objective(k, req, av(rp)));
  // and it must NOT pile onto one device (g=1 -> floor 4000 would be strictly worse).
  EXPECT_LT(rp.predicted_us, 4000.0);

  // c=1: serial -> full raw_sum floor regardless of split (assignment-invariant
  // bank term, the old behavior). Floor stays 4000; the deterministic lowest-index
  // tie-break piles all 4 onto dev0 (makespan negligible under the floor).
  k.banks[0].contention = 1.0;
  const auto rs = gl::solve(k, req);
  EXPECT_DOUBLE_EQ(rs.bank_egress_us, 4000.0)
      << "serial bank (c=1): full raw_sum floor, no spread benefit";
  EXPECT_EQ(counts(av(rs), 2), (std::vector<int>{4, 0}))
      << "serial bank (c=1): no bank-term incentive to spread -> tie-break piles onto dev0";
  EXPECT_DOUBLE_EQ(rs.predicted_us, brute_objective(k, req, av(rs)));
}

// Test 2: brute-force exhaustive reference vs solve()/solve_greedy() at c∈{0,0.5}
// across the reachable solver tiers — the exact enumeration/B&B tier (M^N within
// budget) and the greedy-prefix path (N>kMaxC). (The subset-partition DP tier is
// unreachable for these bounds: N≤kMaxC=5 with M≤kMaxDevices=16 gives M^N≤16^5≈
// 1.05M < the 4.2M enumeration budget, so B&B always wins for N≤kMaxC.) The
// solver result is checked to equal the global optimum (B&B) / be no worse than
// greedy (prefix), and predicted_us is checked against the §3.3 brute objective.
TEST(GpuLoaderSolver, ContentionBruteForceReferenceAllTiers) {
  for (double c : {0.0, 0.5}) {
    // Tier 1 (exact B&B): M=2, N=4 (2^4 enumerable). One bank, large egress so the
    // contention floor is the dominant, assignment-dependent term.
    {
      auto k = make_constants(2, 1, /*egress=*/300.0);
      k.banks[0].contention = c;
      auto req = make_request(2, 4, /*bank=*/0);
      const auto sv = gl::solve(k, req);
      EXPECT_TRUE(sv.exact) << "M^N=16 within budget -> exact B&B (c=" << c << ")";
      // exhaustive optimum over all 2^4 assignments (independent §3.3 objective).
      double best = 1e300;
      for (int m = 0; m < 16; ++m) {
        std::vector<int> a(4);
        for (int i = 0; i < 4; ++i) a[i] = (m >> i) & 1;
        best = std::min(best, brute_objective(k, req, a));
      }
      EXPECT_DOUBLE_EQ(sv.predicted_us, best) << "B&B must hit the §3.3 optimum (c=" << c << ")";
      EXPECT_DOUBLE_EQ(sv.predicted_us, brute_objective(k, req, av(sv)));
      // gl::evaluate must agree with the independent brute objective on the result.
      EXPECT_DOUBLE_EQ(gl::evaluate(k, req, av(sv)), best);
    }

    // Tier 3 (greedy-prefix): M=3, N=8 (> kMaxC and 3^8=6561 > budget? no — clamp
    // M so M^N exceeds the enumeration budget). Use M=10, N=8 like the worked
    // example so solve() takes the prefix path. Verify the prefix assignment's T
    // matches the independent brute objective and is no worse than greedy.
    {
      auto k = make_constants(10, 2, /*egress=*/250.0);
      for (auto& b : k.banks) b.contention = c;
      gl::SolveRequest req;
      req.num_devices = 10;
      req.num_experts = 8;
      req.bank_of.resize(8);
      for (int i = 0; i < 8; ++i) req.bank_of[i] = i % 2;
      req.cached.assign(8 * 10, 0);
      const auto sv = gl::solve(k, req);
      EXPECT_FALSE(sv.exact) << "N=8>kMaxC, M^N>budget -> greedy-prefix (c=" << c << ")";
      EXPECT_DOUBLE_EQ(sv.predicted_us, brute_objective(k, req, av(sv)))
          << "prefix predicted_us must equal the §3.3 objective (c=" << c << ")";
      const auto gr = gl::solve_greedy(k, req);
      EXPECT_DOUBLE_EQ(gr.predicted_us, brute_objective(k, req, av(gr)))
          << "greedy predicted_us must equal the §3.3 objective (c=" << c << ")";
      EXPECT_LE(sv.predicted_us, gr.predicted_us + 1e-6);
    }
  }
}

// ── Pinned-domain solve (TD-LOADER-SHADOW-HOTPATH-COST) ────────────────────────

// Pins are respected verbatim; the free (miss) subset is solved exactly.
TEST(GpuLoaderSolver, PinnedRespectedAndExact) {
  auto k   = make_constants(4, 4);
  auto req = make_request(4, 8, /*bank=*/0);
  // Experts 0..5 cached on device i%4 (hits); 6,7 uncached (misses).
  for (int i = 0; i < 6; ++i) req.cached[static_cast<size_t>(i) * 4 + (i % 4)] = 1;
  req.pinned.assign(8, -1);
  for (int i = 0; i < 6; ++i) req.pinned[i] = i % 4;

  gl::LoaderSolver s;
  const auto r = s.solve(k, req);
  ASSERT_EQ(r.n, 8);
  EXPECT_TRUE(r.exact);
  for (int i = 0; i < 6; ++i) EXPECT_EQ(r.assignment[i], i % 4) << "pin violated at " << i;
  // The two misses land on distinct least-loaded devices (makespan-optimal):
  // devices 0,1 carry 2 pinned hits each? No: counts pinned = {2,2,1,1} (0,4->0;
  // 1,5->1; 2->2; 3->3). Misses must go to devices 2 and 3 (compute-balance).
  const auto c = counts(av(r), 4);
  EXPECT_EQ(c[2] + c[3], 4) << "misses should balance onto the lighter devices";

  // Cross-check exactness: brute-force the two free experts over all 4x4 choices
  // with evaluate() and compare the optimum.
  double best = 1e300;
  auto a = av(r);
  for (int d6 = 0; d6 < 4; ++d6)
    for (int d7 = 0; d7 < 4; ++d7) {
      a[6] = d6; a[7] = d7;
      best = std::min(best, s.evaluate(k, req, a));
    }
  EXPECT_NEAR(r.predicted_us, best, 1e-9) << "pinned solve must equal the restricted optimum";
}

// Empty pinned vector: bit-identical to the historical unrestricted solve.
TEST(GpuLoaderSolver, PinnedEmptyIdenticalToUnpinned) {
  auto k   = make_constants(3, 3);
  auto req = make_request(3, 6, /*bank=*/1);
  req.cached[0 * 3 + 2] = 1;  // one hit on device 2
  gl::LoaderSolver s;
  const auto r0 = s.solve(k, req);          // no pinned field at all
  req.pinned.assign(6, -1);                 // present but all-free -> fall-through
  const auto r1 = s.solve(k, req);
  EXPECT_EQ(av(r0), av(r1));
  EXPECT_DOUBLE_EQ(r0.predicted_us, r1.predicted_us);
}

// All experts pinned: the solve is a pure evaluation of the pinned assignment.
TEST(GpuLoaderSolver, PinnedAllFixed) {
  auto k   = make_constants(4, 4);
  auto req = make_request(4, 4, /*bank=*/2);
  req.pinned = {3, 2, 1, 0};
  gl::LoaderSolver s;
  const auto r = s.solve(k, req);
  EXPECT_EQ(av(r), (std::vector<int>{3, 2, 1, 0}));
  EXPECT_TRUE(r.exact);
  const double manual = s.evaluate(k, req, std::vector<int>{3, 2, 1, 0});
  EXPECT_DOUBLE_EQ(r.predicted_us, manual);
}

// A restricted optimum can never beat the unrestricted one; with no binding
// pins they coincide.
TEST(GpuLoaderSolver, PinnedNeverBeatsUnpinned) {
  auto k   = make_constants(4, 4);
  std::mt19937 rng(7);
  for (int trial = 0; trial < 20; ++trial) {
    auto req = make_request(4, 7, /*bank=*/static_cast<int>(rng() % 4));
    // Random hits.
    for (int i = 0; i < 7; ++i)
      if (rng() % 2) req.cached[static_cast<size_t>(i) * 4 + (rng() % 4)] = 1;
    gl::LoaderSolver s;
    const auto full = s.solve(k, req);
    // Pin every hit to its resident device (the dispatcher policy).
    req.pinned.assign(7, -1);
    for (int i = 0; i < 7; ++i)
      for (int j = 0; j < 4; ++j)
        if (req.cached[static_cast<size_t>(i) * 4 + j]) { req.pinned[i] = j; break; }
    const auto pin = s.solve(k, req);
    EXPECT_GE(pin.predicted_us + 1e-9, full.predicted_us)
        << "restricted optimum beat the unrestricted one (trial " << trial << ")";
  }
}

// The dispatcher's free-replica case: an expert cached ONLY away from its
// orchestrator target is left free, and the solver reroutes it to the resident
// copy (zero transfer) when that is optimal.
TEST(GpuLoaderSolver, PinnedFreeReplicaReroutes) {
  auto k   = make_constants(4, 4);
  auto req = make_request(4, 5, /*bank=*/0);
  req.cached[0 * 4 + 3] = 1;   // expert 0 resident on device 3 only
  req.pinned.assign(5, -1);    // orchestrator targeted device 0 (a miss there) -> free
  gl::LoaderSolver s;
  const auto r = s.solve(k, req);
  EXPECT_EQ(r.assignment[0], 3) << "free expert should ride the resident replica";
  EXPECT_TRUE(r.exact);
}

// Pinned hits + per-device place cost (the reuse reward shape): the place term
// must steer ONLY the miss subspace — pinned hits are charged nothing (cached
// placements skip place in apply()), and a transfer-tied miss goes to the
// cheap-victim (low place) device.
TEST(GpuLoaderSolver, PinnedWithReusePlaceSteersMisses) {
  auto k   = make_constants(4, 4);
  auto req = make_request(4, 5, /*bank=*/0);
  // Experts 0..3 cached on device i (hits, pinned); expert 4 uncached (miss).
  for (int i = 0; i < 4; ++i) req.cached[static_cast<size_t>(i) * 4 + i] = 1;
  req.pinned = {0, 1, 2, 3, -1};
  // Uniform transfer (bank0 same rate to all devices) -> the miss is transfer-
  // tied across devices; per-device place cost makes device 2 the cheap victim.
  req.place.assign(5 * 4, 0.0);
  const double pd[4] = {900.0, 900.0, 10.0, 900.0};
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 4; ++j) req.place[static_cast<size_t>(i) * 4 + j] = pd[j];
  gl::LoaderSolver s;
  const auto r = s.solve(k, req);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(r.assignment[i], i);
  EXPECT_EQ(r.assignment[4], 2) << "miss must go to the cheap-victim device";
  EXPECT_DOUBLE_EQ(r.place_us, 10.0) << "only the miss is charged place";
}

// ── M2v2 exposed-wall objective (SolveRequest::m2; DETHRONE_AFFINITY_IDEAS) ───

namespace {
void arm_m2(gl::SolveRequest& req, int M) {
  req.m2 = true;
  req.m2_s.assign(M, 1.0);
  req.m2_o0.assign(M, 0.0);
  req.m2_oc.assign(M, 0.0);
  req.m2_hsat.assign(M, 8.0);
  req.m2_cpw.assign(M, 1.0);
  req.m2_gc = 1.0;
}
}  // namespace

// Pinned m2 solve equals the brute-forced restricted optimum under the SAME
// (public-evaluate) m2 objective — end-to-end soundness of the m2 bounds
// (prefix objective + zeroed legacy makespan floors never over-prune).
TEST(GpuLoaderSolver, M2PinnedExactMatchesBruteEvaluate) {
  auto k   = make_constants(4, 4);
  auto req = make_request(4, 8, /*bank=*/0);
  for (int i = 0; i < 6; ++i) req.cached[static_cast<size_t>(i) * 4 + (i % 4)] = 1;
  req.pinned.assign(8, -1);
  for (int i = 0; i < 6; ++i) req.pinned[i] = i % 4;
  arm_m2(req, 4);
  req.m2_oc = {40.0, 60.0, 25.0, 50.0};   // credit varies per device
  req.m2_cpw = {2.0, 2.3, 1.7, 1.4};      // 5090-style critical-path weights
  req.m2_gc = 0.4;

  gl::LoaderSolver s;
  const auto r = s.solve(k, req);
  ASSERT_EQ(r.n, 8);
  EXPECT_TRUE(r.exact);
  for (int i = 0; i < 6; ++i) EXPECT_EQ(r.assignment[i], i % 4);
  double best = 1e300;
  auto a = av(r);
  for (int d6 = 0; d6 < 4; ++d6)
    for (int d7 = 0; d7 < 4; ++d7) {
      a[6] = d6; a[7] = d7;
      best = std::min(best, s.evaluate(k, req, a));
    }
  EXPECT_NEAR(r.predicted_us, best, 1e-9)
      << "m2 pinned solve must equal the restricted m2 optimum";
}

// A device whose credit absorbs its whole exposure attracts the miss under m2
// even when its raw transfer is the slowest — the legacy objective avoids it.
TEST(GpuLoaderSolver, M2CreditSteersMissTowardCreditedDevice) {
  auto k = make_constants(4, 1);
  for (int d = 0; d < 4; ++d) k.matrix[0][d] = gl::TransferCell{500.0, 2, 5.0};
  k.matrix[0][1] = gl::TransferCell{900.0, 2, 5.0};  // device 1: slowest raw xfer
  auto req = make_request(4, 5, /*bank=*/0);
  for (int i = 0; i < 4; ++i) req.cached[static_cast<size_t>(i) * 4 + i] = 1;
  req.pinned.assign(5, -1);
  for (int i = 0; i < 4; ++i) req.pinned[i] = i;  // one pinned hit per device

  gl::LoaderSolver s;
  const auto legacy = s.solve(k, req);
  EXPECT_NE(legacy.assignment[4], 1) << "legacy avoids the slowest-transfer device";

  arm_m2(req, 4);
  req.m2_o0 = {0.0, 1e6, 0.0, 0.0};  // device 1's exposure fully credited
  req.m2_gc = 0.0;                    // isolate the transfer/credit effect
  const auto m2 = s.solve(k, req);
  EXPECT_EQ(m2.assignment[4], 1)
      << "m2 places the miss where the credit hides its transfer";
  for (int i = 0; i < 4; ++i) EXPECT_EQ(m2.assignment[i], i);
}

// Malformed m2 vectors (wrong size) silently disable m2: result bit-identical
// to the m2=false solve.
TEST(GpuLoaderSolver, M2InvalidSizesDisabled) {
  auto k   = make_constants(4, 2);
  auto req = make_request(4, 6, /*bank=*/1);
  for (int i = 0; i < 3; ++i) req.cached[static_cast<size_t>(i) * 4 + i] = 1;
  req.pinned.assign(6, -1);
  for (int i = 0; i < 3; ++i) req.pinned[i] = i;
  gl::LoaderSolver s;
  const auto base = s.solve(k, req);
  arm_m2(req, 4);
  req.m2_hsat.resize(3);  // wrong size -> m2_valid false
  const auto r = s.solve(k, req);
  EXPECT_EQ(av(r), av(base));
  EXPECT_DOUBLE_EQ(r.predicted_us, base.predicted_us);
}

// m2 arms ONLY on the pinned tier: with no pins the solve is bit-identical to
// the legacy solve even when valid m2 params are supplied.
TEST(GpuLoaderSolver, M2WithoutPinsFallsThroughLegacy) {
  auto k   = make_constants(3, 3);
  auto req = make_request(3, 6, /*bank=*/1);
  req.cached[0 * 3 + 2] = 1;
  gl::LoaderSolver s;
  const auto base = s.solve(k, req);
  arm_m2(req, 3);
  req.m2_oc.assign(3, 500.0);
  const auto r = s.solve(k, req);
  EXPECT_EQ(av(r), av(base));
  EXPECT_DOUBLE_EQ(r.predicted_us, base.predicted_us);
}

// ── CPU expert offload: CPU as an assignable solver target (Stage 1) ─────────
// append_cpu_expert_device adds a device that reads weights from host RAM (no
// PCIe transfer — transfer cells 0) with a per-expert CPU-FFN ComputeCurve; its
// read pressure is still priced as egress on the expert's source bank. These
// verify the solver (A) offloads the fetch-bound tail to CPU when it lowers
// makespan, (B) stays inert when CPU compute is too slow (assignment + T
// identical to GPU-only — the frozen-golden safety property), and (C) cannot
// cheat a bottleneck DDR bank's contention floor.

TEST(GpuLoaderSolverCpuOffload, AbsorbsFetchBoundTail) {
  // 2 fetch-bound GPUs (rate 500 + compute 150/expert), 6 uncached experts,
  // bank egress small (not binding). GPU-only makespan is fetch-dominated.
  auto k = make_constants(/*M=*/2, /*B=*/1, /*egress=*/50.0);
  auto req0 = make_request(2, 6, /*bank=*/0);
  const auto gpu_only = gl::solve(k, req0);

  // A CPU node that computes an expert in 120 us (P=1) with NO fetch.
  const int cpu = gl::append_cpu_expert_device(k, /*numa_node=*/3, {0.0, 120.0, 1});
  auto req1 = make_request(3, 6, /*bank=*/0);
  const auto with_cpu = gl::solve(k, req1);

  const auto c = counts(av(with_cpu), 3);
  EXPECT_GT(c[cpu], 0) << "CPU should absorb part of the fetch-bound tail";
  EXPECT_LT(with_cpu.predicted_us, gpu_only.predicted_us)
      << "offloading off the binding PCIe bus must lower predicted makespan";
}

TEST(GpuLoaderSolverCpuOffload, InertWhenCpuTooSlow) {
  auto k = make_constants(2, 1, /*egress=*/50.0);
  auto req0 = make_request(2, 6, /*bank=*/0);
  const auto gpu_only = gl::solve(k, req0);

  const int cpu = gl::append_cpu_expert_device(k, 3, {0.0, 100000.0, 1});  // absurdly slow
  auto req1 = make_request(3, 6, /*bank=*/0);
  const auto with_cpu = gl::solve(k, req1);

  EXPECT_EQ(counts(av(with_cpu), 3)[cpu], 0) << "a too-slow CPU must never be assigned";
  EXPECT_EQ(av(with_cpu), av(gpu_only)) << "assignment must be identical to GPU-only";
  EXPECT_DOUBLE_EQ(with_cpu.predicted_us, gpu_only.predicted_us);
}

TEST(GpuLoaderSolverCpuOffload, RespectsBankContentionFloor) {
  // bank1 = the H2D-bottleneck DDR node (egress 600, strict serial). A CPU expert
  // reading it is booked as egress on bank1 just like a GPU fetch, so the CPU can
  // never drop predicted below bank1's egress floor — it still reads those bytes.
  std::vector<int> banks = {0, 0, 0, 0, 1, 1, 1, 1};
  auto k0 = make_constants(2, 2, /*egress=*/50.0);
  k0.banks[1].egress_us = 600.0;
  auto req0 = make_request(2, 8, /*bank=*/0);
  req0.bank_of = banks;
  const auto gpu_only = gl::solve(k0, req0);

  auto k = make_constants(2, 2, /*egress=*/50.0);
  k.banks[1].egress_us = 600.0;
  gl::append_cpu_expert_device(k, 3, {0.0, 80.0, 1});  // fast CPU
  auto req = make_request(3, 8, /*bank=*/0);
  req.bank_of = banks;
  const auto r = gl::solve(k, req);

  const double bank1_floor = 4 * 600.0;  // contention 1 → strict serial sum
  EXPECT_GE(r.predicted_us, bank1_floor - 1e-6)
      << "the CPU still reads the bottleneck DDR node → cannot beat its egress floor";
  EXPECT_LE(r.predicted_us, gpu_only.predicted_us + 1e-6)
      << "adding the CPU option must never increase makespan (it can offload GPU-bound work)";
}

// ── 64-instantiation golden regression + LoaderSolver256 (large unions) ──────
//
// The solver became a template on the max-expert bound (BasicLoaderSolver<NMax>,
// instantiations 64 + 256) for the dsp52 batched-verify unions. The 64 path is
// FROZEN (REEF fingerprints are calibrated against it): the goldens below were
// captured from the PRE-TEMPLATE solver (commit 7398d777) by running the
// golden64 generator (gpu_loader_solver_golden.inc) against it on the dev box
// (x86-64, g++ RelWithDebInfo + FAST_CPU -O3 -march=native, libstdc++), and the
// templated 64 instantiation must reproduce every assignment AND every
// predicted_us BIT PATTERN. NOTE: the goldens encode this box's libstdc++
// uniform_*_distribution and -march=native FP codegen — recapture (see the
// harness note in the .inc) if either ever changes, which re-baselines the
// frozen identity guarantee.

#include <cstdint>
#include <cstring>

#include "gpu_loader_solver_golden.inc"

namespace {

uint64_t dbits(double x) {
  uint64_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

struct Golden64 {
  std::vector<int> a;   // solve() assignment
  uint64_t t;           // solve() predicted_us bit pattern
  int exact;            // solve() exact flag
  std::vector<int> ga;  // solve_greedy() assignment
  uint64_t gt;          // solve_greedy() predicted_us bit pattern
};

static const Golden64 kGolden64[golden64::kNumInstances] = {
    {{0,1,2,3,0,1,0,1}, 0x4081ebfe3bd67227ULL, 1, {0,1,2,3,0,1,1,3}, 0x4089680000000000ULL},  // inst 0 N=8 M=4
    {{0,3,3,4,4,4,1,1}, 0x408861ccdc782b41ULL, 1, {0,7,3,4,5,6,1,1}, 0x407969afe941d532ULL},  // inst 1 N=8 M=10
    {{6,1,8,3,4,7,7,8,8,9,5,3,9,8,0,2,9,4,2,6}, 0x407b17d101c65f89ULL, 0, {6,1,8,7,4,4,3,5,0,9,2,9,4,8,8,3,5,7,0,1}, 0x407cbaadc706f3efULL},  // inst 2 N=20 M=10
    {{2,2,1,4,5}, 0x407b300000000000ULL, 1, {4,8,1,13,5}, 0x4071000000000000ULL},  // inst 3 N=5 M=16
    {{0,2,2,5,0,3,1,0,0,2,0,1,4,2,2,2,3}, 0x408ec057666ef9a6ULL, 1, {3,2,0,4,3,2,1,1,2,2,0,1,4,2,2,3,3}, 0x4093309a35c49905ULL},  // inst 4 N=17 M=6
    {{6,7,1,5,1,4,0,0,5,2}, 0x4089a7277d36116bULL, 0, {6,7,1,5,1,4,0,0,5,2}, 0x4089a7277d36116bULL},  // inst 5 N=10 M=8
    {{0,0,0,0,1,1,1,0,1,1,0,0,0,1,1}, 0x409eb40000000000ULL, 1, {0,0,1,0,1,1,1,0,0,1,0,0,1,1,1}, 0x40a0ac0000000000ULL},  // inst 6 N=15 M=2
    {{2,2,1,2,0}, 0x4087a454f1387298ULL, 1, {2,2,1,2,0}, 0x4087a454f1387298ULL},  // inst 7 N=5 M=4
    {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 0x40c400cc755e824dULL, 1, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 0x40c400cc755e824dULL},  // inst 8 N=20 M=1
    {{1,1,1,0,0,0}, 0x408e95f172b58394ULL, 1, {1,0,1,0,0,1}, 0x40943b50bcf39733ULL},  // inst 9 N=6 M=2
    {{0,0,0,0,0,0}, 0x40a8479d40e422dcULL, 1, {0,0,0,0,0,0}, 0x40a8479d40e422dcULL},  // inst 10 N=6 M=1
    {{2,5,1,0,2}, 0x40828941e344ad3eULL, 1, {2,5,1,0,2}, 0x40828941e344ad3eULL},  // inst 11 N=5 M=7
    {{0,2,1,2,1,0,2,2,0,0,2,0,2,0,1,1}, 0x4095440000000000ULL, 1, {0,2,1,2,1,0,0,2,0,1,0,2,2,2,1,0}, 0x4098100000000000ULL},  // inst 12 N=16 M=3
    {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 0x40c98e32600a7f57ULL, 1, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 0x40c98e32600a7f57ULL},  // inst 13 N=19 M=1
    {{2,1,8,8,8,7,0,6}, 0x4074a8373f3cded1ULL, 0, {2,1,6,0,4,7,0,1}, 0x4074a8373f3cded1ULL},  // inst 14 N=8 M=10
    {{0,3,0,2}, 0x4083c4e7c625c1cfULL, 1, {0,3,0,2}, 0x4083c4e7c625c1cfULL},  // inst 15 N=4 M=4
    {{5,0,0,2,2,1,4,0}, 0x408a43b6119ac8d8ULL, 1, {5,4,6,2,2,1,7,0}, 0x408440dd1e2ba571ULL},  // inst 16 N=8 M=8
    {{0,0,0,0,0,0}, 0x40b934d24d5b0ca4ULL, 1, {0,0,0,0,0,0}, 0x40b934d24d5b0ca4ULL},  // inst 17 N=6 M=1
    {{1,3,3,0,4,0,0}, 0x407f500000000000ULL, 1, {4,3,3,0,4,0,0}, 0x407f500000000000ULL},  // inst 18 N=7 M=7
    {{1,0,0,2,1,0,0,1,0,2,1,0,1}, 0x40a620ee5bc7cccfULL, 1, {1,0,0,2,1,0,2,1,0,2,1,0,1}, 0x40a483422d961a63ULL},  // inst 19 N=13 M=3
    {{0,5,2,5,4,8,0,3,7,2,6}, 0x407c2d585d631777ULL, 0, {0,5,2,5,4,6,0,3,3,2,5}, 0x4087994d031aaaf9ULL},  // inst 20 N=11 M=9
    {{2,1,4,1}, 0x4075c00000000000ULL, 1, {3,4,6,7}, 0x4067a00000000000ULL},  // inst 21 N=4 M=9
    {{0,1,1,2,0,2,1,2,1,1,1,2,0,2}, 0x409afc38eb798561ULL, 1, {2,1,1,2,1,2,1,0,1,1,0,2,0,2}, 0x409afc38eb798561ULL},  // inst 22 N=14 M=3
    {{5,0,6,0,0,4,1,3,7,6,4,2,1,0,5,2}, 0x408de960dd52d77aULL, 0, {5,0,6,0,3,1,1,3,7,6,4,0,1,0,5,2}, 0x409213a7f41d6095ULL},  // inst 23 N=16 M=8
    {{1,0,0,0,0,0,2}, 0x407ea00000000000ULL, 1, {1,0,3,0,2,3,3}, 0x4080f00000000000ULL},  // inst 24 N=7 M=4
    {{0,0,1,1,1,1,1,0,0,1}, 0x40ab3efa75b5c202ULL, 1, {0,1,1,0,1,1,1,0,0,1}, 0x40ad1ff02796af83ULL},  // inst 25 N=10 M=2
    {{1,0,3,4,2,1,3}, 0x406d77c68b3e4781ULL, 1, {1,1,3,2,0,1,3}, 0x4073a4d961dad32cULL},  // inst 26 N=7 M=5
    {{4,2,0,6,0,7,0,0}, 0x408751c0871e2bc5ULL, 1, {4,2,2,6,2,1,5,4}, 0x4083f9c0871e2bc5ULL},  // inst 27 N=8 M=8
    {{1,1,0,2,0,2,2,0,0,0,1,1,0,2,1,0,2}, 0x40a15e80ba68e82dULL, 1, {1,2,0,2,0,2,2,0,0,2,1,1,0,2,0,0,2}, 0x40a251717d937ce9ULL},  // inst 28 N=17 M=3
    {{5,0,2,1,3,1,1,4}, 0x40730b13d4aff411ULL, 1, {5,0,2,1,3,1,4,4}, 0x407ca97a3832320eULL},  // inst 29 N=8 M=6
    {{0,1,0,1,1,0,0,0,1,0,0,1,1,1,1,0,0,1}, 0x40aad00000000000ULL, 1, {1,0,0,1,0,0,1,1,0,0,1,0,1,1,0,0,0,1}, 0x40afd80000000000ULL},  // inst 30 N=18 M=2
    {{2,3,0,2,0,6,5}, 0x4084799867590795ULL, 1, {2,0,3,2,1,6,5}, 0x40893749b3abb569ULL},  // inst 31 N=7 M=7
    {{1}, 0x406af8283d272c43ULL, 1, {1}, 0x406af8283d272c43ULL},  // inst 32 N=1 M=4
    {{0,3,2}, 0x40874f1ebae9b187ULL, 1, {0,3,2}, 0x40874f1ebae9b187ULL},  // inst 33 N=3 M=4
    {{4,2,0,5,6,1,1,6,2,1,0}, 0x407a73f38f800560ULL, 1, {8,2,0,5,6,1,3,6,4,5,0}, 0x4077fd654f26497eULL},  // inst 34 N=11 M=9
    {{0,0,0,0,0}, 0x40ac0afa2f6c2427ULL, 1, {0,0,0,0,0}, 0x40ac0afa2f6c2427ULL},  // inst 35 N=5 M=1
    {{5}, 0x406a200000000000ULL, 1, {5}, 0x406a200000000000ULL},  // inst 36 N=1 M=6
    {{0,0,0,0,0,0,0,0,0,0,0,0}, 0x40c1952d0a4f3e21ULL, 1, {0,0,0,0,0,0,0,0,0,0,0,0}, 0x40c1952d0a4f3e21ULL},  // inst 37 N=12 M=1
    {{4,4,0,1}, 0x407b6420f083b1fcULL, 1, {4,5,0,1}, 0x407b6420f083b1fcULL},  // inst 38 N=4 M=6
    {{7,7,1,2,3,2,3,3,0,4,2,0,5,3,3,2}, 0x408aeac098f7e58dULL, 1, {7,8,1,2,3,2,8,3,6,5,2,5,7,4,3,4}, 0x408c5e99e1d19703ULL},  // inst 39 N=16 M=10
    {{1,1,0,2,3,3,0,2,1,3,0,0,2,1,2}, 0x4096500c2ed7dbceULL, 1, {1,1,0,2,3,3,0,2,0,1,2,0,3,1,2}, 0x40971c3710c7a9beULL},  // inst 40 N=15 M=4
    {{0,1,0,1,1,0,0}, 0x40a1ffb7fc0c2ffaULL, 1, {0,1,0,1,1,0,0}, 0x40a1ffb7fc0c2ffaULL},  // inst 41 N=7 M=2
    {{5,0,0,2,3,1,1}, 0x408ea00000000000ULL, 1, {5,4,0,2,0,1,1}, 0x408e380000000000ULL},  // inst 42 N=7 M=6
    {{8,5,5,0,1,5,1,1}, 0x408146836372ebcaULL, 1, {8,5,5,0,4,5,3,3}, 0x407b16b624245509ULL},  // inst 43 N=8 M=9
    {{0,0,0,1,0,0,1,0,0,1,1,0,0,0}, 0x409c059c7b281e93ULL, 1, {0,0,0,1,1,0,1,0,0,0,0,1,0,0}, 0x40a365e52dae228dULL},  // inst 44 N=14 M=2
    {{0,1,0,1,1,0,0,1}, 0x40a4d3917477cc31ULL, 1, {1,1,0,1,0,1,1,0}, 0x40a5548643e0d2adULL},  // inst 45 N=8 M=2
    {{1,1,1,1,2,1,0}, 0x4075073bd4dd85dcULL, 1, {1,1,1,5,2,5,0}, 0x406c93d4aef691e5ULL},  // inst 46 N=7 M=8
    {{1,5,8,0,1,9,7,2,6,9,3,1,3,6,3,5,8,3}, 0x409241169ee69f7bULL, 0, {1,5,8,0,1,9,7,2,6,9,3,1,3,6,3,5,8,3}, 0x409241169ee69f7bULL},  // inst 47 N=18 M=10
};

// Valid-prefix view working for BOTH instantiations' result types.
template <typename R>
std::vector<int> avn(const R& r) {
  return {r.assignment.begin(), r.assignment.begin() + r.n};
}

}  // namespace

// (a) 64-path identity: the templated LoaderSolver (= BasicLoaderSolver<64>)
// must reproduce the pre-template solver's outputs BIT-EXACTLY on all 48
// golden instances (every reachable tier: pinned B&B / B&B / prefix / greedy,
// consequence terms, contention, m2, fixed overhead).
TEST(GpuLoaderSolver, Solver64GoldenRegression) {
  for (int inst = 0; inst < golden64::kNumInstances; ++inst) {
    const auto in = golden64::make_instance(inst);
    gl::LoaderSolver s1;
    const auto r = s1.solve(in.k, in.req);
    gl::LoaderSolver s2;
    const auto g = s2.solve_greedy(in.k, in.req);
    const Golden64& G = kGolden64[inst];
    ASSERT_EQ(avn(r), G.a) << "solve assignment drifted, inst " << inst;
    ASSERT_EQ(dbits(r.predicted_us), G.t)
        << "solve predicted_us bits drifted, inst " << inst;
    ASSERT_EQ(r.exact ? 1 : 0, G.exact) << "exact flag drifted, inst " << inst;
    ASSERT_EQ(avn(g), G.ga) << "greedy assignment drifted, inst " << inst;
    ASSERT_EQ(dbits(g.predicted_us), G.gt)
        << "greedy predicted_us bits drifted, inst " << inst;
  }
}

// The 256 instantiation runs the IDENTICAL tier code wherever the exact tiers
// fire (pinned B&B / B&B / dp_full keep their existing budgets) — cross-check
// it bit-exactly against the frozen 64 solver on every exact golden instance.
// (Heuristic-tier instances differ by design: dp_prefix vs the greedy tier.)
TEST(GpuLoaderSolver256, ExactTiersMatchFrozen64) {
  int checked = 0;
  for (int inst = 0; inst < golden64::kNumInstances; ++inst) {
    const auto in = golden64::make_instance(inst);
    gl::LoaderSolver s64;
    const auto r64 = s64.solve(in.k, in.req);
    if (!r64.exact) continue;
    gl::LoaderSolver256 s256;
    const auto r256 = s256.solve(in.k, in.req);
    ASSERT_TRUE(r256.exact) << "inst " << inst;
    ASSERT_EQ(avn(r256), avn(r64)) << "inst " << inst;
    ASSERT_EQ(dbits(r256.predicted_us), dbits(r64.predicted_us)) << "inst " << inst;
    ++checked;
  }
  EXPECT_GE(checked, 30) << "golden set lost its exact-tier coverage";
}

namespace {

// Live dsp52 batched-verify shape: union of N experts over M devices, the
// first `hits` resident (round-robin device i%M) and PINNED there (dispatcher
// policy), the rest misses. Reuse-style place rewards optional.
gl::SolveRequest make_union_request(int M, int N, int hits, bool with_place) {
  gl::SolveRequest req = make_request(M, N, /*bank=*/0);
  req.bank_of.resize(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) req.bank_of[static_cast<size_t>(i)] = i % 4;
  req.pinned.assign(static_cast<size_t>(N), -1);
  for (int i = 0; i < hits; ++i) {
    req.cached[static_cast<size_t>(i) * M + (i % M)] = 1;
    req.pinned[static_cast<size_t>(i)] = i % M;
  }
  if (with_place) {
    // Per-device victim-age reward shape (reef_orch_route's pd[j]).
    req.place.assign(static_cast<size_t>(N) * M, 0.0);
    const double pd[4] = {120.0, 60.0, 200.0, 30.0};
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        req.place[static_cast<size_t>(i) * M + j] = pd[j % 4];
  }
  return req;
}

}  // namespace

// (b) 256 greedy tier: pinned hits respected verbatim, every miss placed on a
// valid device, result flagged heuristic.
TEST(GpuLoaderSolver256, GreedyTierPinsRespectedAllMissesPlaced) {
  const auto k = make_constants(4, 4, /*egress=*/50.0);
  const int N = 128, hits = 98;
  const auto req = make_union_request(4, N, hits, /*with_place=*/true);
  gl::LoaderSolver256 s;
  const auto r = s.solve(k, req);
  ASSERT_EQ(r.n, N);
  EXPECT_FALSE(r.exact) << "union 128 with 30 misses must take the greedy tier";
  for (int i = 0; i < hits; ++i)
    ASSERT_EQ(r.assignment[static_cast<size_t>(i)], i % 4) << "pin violated at " << i;
  for (int i = 0; i < N; ++i) {
    ASSERT_GE(r.assignment[static_cast<size_t>(i)], 0);
    ASSERT_LT(r.assignment[static_cast<size_t>(i)], 4);
  }
  EXPECT_GT(r.predicted_us, 0.0);
}

// Small miss counts inside a large union still fire the EXACT pinned B&B tier
// (existing budgets untouched in the 256 instantiation).
TEST(GpuLoaderSolver256, SmallMissCountStaysExact) {
  const auto k = make_constants(4, 4, /*egress=*/50.0);
  const auto req = make_union_request(4, /*N=*/100, /*hits=*/96, /*with_place=*/false);
  gl::LoaderSolver256 s;
  const auto r = s.solve(k, req);  // 4 misses free: 4^4 = 256 within budget
  ASSERT_EQ(r.n, 100);
  EXPECT_TRUE(r.exact) << "4 free misses fit the pinned enumeration budget";
  for (int i = 0; i < 96; ++i)
    ASSERT_EQ(r.assignment[static_cast<size_t>(i)], i % 4);
}

// Deterministic across runs and across solver objects (stable ordering +
// explicit lowest-index tie-breaks, no RNG).
TEST(GpuLoaderSolver256, GreedyTierDeterministic) {
  const auto k = make_constants(4, 4, /*egress=*/50.0);
  const auto req = make_union_request(4, 128, 98, /*with_place=*/true);
  gl::LoaderSolver256 s;
  const auto r1 = s.solve(k, req);
  const auto r2 = s.solve(k, req);   // same (reused) solver
  gl::LoaderSolver256 fresh;
  const auto r3 = fresh.solve(k, req);  // fresh scratch
  EXPECT_EQ(avn(r1), avn(r2));
  EXPECT_EQ(avn(r1), avn(r3));
  EXPECT_EQ(dbits(r1.predicted_us), dbits(r2.predicted_us));
  EXPECT_EQ(dbits(r1.predicted_us), dbits(r3.predicted_us));
}

// Load-balance sanity 1: fully symmetric all-miss union splits exactly evenly
// (the greedy ranks by the same makespan term the exact tiers use).
TEST(GpuLoaderSolver256, GreedyTierBalancesSymmetricMisses) {
  const auto k = make_constants(4, 1, /*egress=*/10.0);
  auto req = make_request(4, 128, /*bank=*/0);  // all miss, no pins
  gl::LoaderSolver256 s;
  const auto r = s.solve(k, req);
  const auto c = counts(avn(r), 4);
  EXPECT_EQ(c, (std::vector<int>{32, 32, 32, 32}))
      << "symmetric 128-miss union must split 32/32/32/32";
}

// Load-balance sanity 2: a device already loaded with pinned hits (its compute
// occupancy counts toward the makespan) receives the FEWEST misses.
TEST(GpuLoaderSolver256, GreedyTierAvoidsHitLoadedDevice) {
  const auto k = make_constants(4, 1, /*egress=*/10.0);
  auto req = make_request(4, 128, /*bank=*/0);
  req.pinned.assign(128, -1);
  for (int i = 0; i < 40; ++i) {           // 40 hits, ALL resident on device 0
    req.cached[static_cast<size_t>(i) * 4 + 0] = 1;
    req.pinned[static_cast<size_t>(i)] = 0;
  }
  gl::LoaderSolver256 s;
  const auto r = s.solve(k, req);
  std::vector<int> miss_cnt(4, 0);
  for (int i = 40; i < 128; ++i) ++miss_cnt[static_cast<size_t>(r.assignment[static_cast<size_t>(i)])];
  for (int j = 1; j < 4; ++j)
    EXPECT_LT(miss_cnt[0], miss_cnt[static_cast<size_t>(j)])
        << "hit-loaded device 0 must receive the fewest misses (got "
        << miss_cnt[0] << " vs dev" << j << "=" << miss_cnt[static_cast<size_t>(j)] << ")";
}

// (c) Solve-time microbench at the LIVE dsp52 shape (union≈128, ~30 misses,
// M=4 devices, place + convex evict populated — the reef_orch_route request
// shape). Repo lesson: bench the LIVE shape. Budget: order-of-µs; hard-fail
// above 200 µs/solve.
TEST(GpuLoaderSolver256, LiveShapeMicrobench) {
  const auto k = make_constants(4, 4, /*egress=*/50.0);
  auto req = make_union_request(4, 128, 98, /*with_place=*/true);
  req.evict_cum.resize(4);
  for (int d = 0; d < 4; ++d) {
    req.evict_cum[static_cast<size_t>(d)].assign(129, 0.0);
    double acc = 0.0;
    for (int n = 1; n <= 128; ++n) {
      acc += 10.0 + n;
      req.evict_cum[static_cast<size_t>(d)][static_cast<size_t>(n)] = acc;
    }
  }
  gl::LoaderSolver256 s;   // persistent, like the orchestrator holds it
  volatile double sink = 0.0;
  for (int it = 0; it < 50; ++it) sink += s.solve(k, req).predicted_us;  // warm
  constexpr int kIters = 1000;
  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < kIters; ++it) sink += s.solve(k, req).predicted_us;
  const auto t1 = std::chrono::steady_clock::now();
  (void)sink;
  const double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
  fprintf(stderr,
          "[Solver256 bench, %d iters] live shape (N=128, 30 misses, M=4, "
          "place+evict): %.2f us/solve\n",
          kIters, us);
  EXPECT_LT(us, 200.0) << "live-shape solve must stay well under the µs budget";
}
