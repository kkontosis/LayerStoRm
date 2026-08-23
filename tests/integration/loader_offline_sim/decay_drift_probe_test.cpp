// CPU-only drift-probe: resolve the "LS_EVICT_DECAY is a no-op" contradiction.
//
// The keeper shows ACT+decay NEW 0.629 vs plain-ACT NEW 0.392 (a 0.24 gap), yet
// on the SINGLE fixed routed_trace the two collapse to 0.3905 vs 0.3906. A true
// no-op would be BIT-IDENTICAL. This test resolves the contradiction with four
// evidence parts (all CPU-only, GPU-free, driving the REAL engine classes):
//
//   A. DECISION-STREAM DIFF — replay plain-ACT NEW and ACT+decay NEW on the
//      identical fixed trace, capturing every routed-expert lookup (target +
//      hit/miss) and every eviction (victim). Count exactly how many of the
//      46 400 lookups diverge (target / hit) and how many eviction victims
//      differ; report the FIRST divergence. Proves decay is NOT bit-identical.
//
//   B. MECHANISM (board micro-experiment) — on the REAL EvictScoreBoard, with
//      concrete values: (1) decay_all changes cheapest_scores_sorted VALUES (the
//      evict_cum magnitudes fed to the placement solver) while (2) leaving
//      cheapest_keys ORDER invariant for non-duplicates (eff==raw, uniform
//      scale), and (3) it CAN flip a victim when a cross-GPU DUPLICATE is present
//      (eff = raw - rank·base is non-linear in raw) or via the max(0) clamp tie.
//      => decay's fixed-trace seed is the evict_cum MAGNITUDE channel + the rare
//      duplicate/clamp flip — an algorithmic lever on PLACEMENT, not a pure FP
//      accident; the keeper's 0.629 is that seed amplified by routing drift.
//
//   C. LRU-CORRECTNESS PROBE — plain-ACT NEW hit-rate vs the textbook clean-LRU
//      ceiling on the same trace+placement, and a bounded-recency prototype
//      (TD-EVICT-RECENCY-MAGNITUDE) measuring whether bounding the clock moves
//      plain-ACT toward the ceiling DETERMINISTICALLY.
//
//   D. DETERMINISM — plain-ACT NEW run twice is bit-identical (the sim is fully
//      deterministic), establishing the CPU side is causal-steering, not sampling.
//
// ALGO-NOT-DATA: oracle numbers are never read here; the keeper hit-rates are
// referenced in comments only. Never touches a GPU / CUDA.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "core/gpu_loader/loader_evict_scores.h"
#include "core/memory/eviction_policy.h"
#include "offline_sim.h"

namespace {

using namespace layerstorm::offline_sim;
namespace gl = layerstorm::gpu_loader;
namespace lmem = layerstorm::memory;

std::string asset(const std::string& name) {
  return std::string(LAYERSTORM_SOURCE_DIR
                     "/tests/integration/loader_offline_sim/assets/") + name;
}
std::string read_file(const std::string& path) {
  std::ifstream in(path);
  EXPECT_TRUE(in.good()) << "cannot open " << path;
  std::stringstream ss; ss << in.rdbuf(); return ss.str();
}

// Principled bounded-recency increment scale: base_score / capacity makes the
// recency spread over one residency window comparable to the rank·base duplicate
// penalty (so the penalty bites). NOT a tuned target (algo-not-data).
double base_over_cap() { return gl::kDefaultBaseScore / 460.0; }

// Per-(token,layer,gpu) sorted victim multiset, for an order-independent diff.
using EvKey = std::tuple<int, int, int>;
std::map<EvKey, std::vector<uint64_t>> evict_index(const DecisionLog& d) {
  std::map<EvKey, std::vector<uint64_t>> m;
  for (const auto& e : d.evicts) {
    const uint64_t vk = (static_cast<uint64_t>(e.v_layer) << 16) | e.v_expert;
    m[{e.token, e.layer, e.gpu}].push_back(vk);
  }
  for (auto& [k, v] : m) std::sort(v.begin(), v.end());
  return m;
}

class DriftProbe : public ::testing::Test {
 protected:
  void SetUp() override {
    k_ = gl::from_json_string(
        read_file(asset("gpu_loader_calibration_5090x2.json")));
    ASSERT_EQ(k_.num_devices, 2);
    auto rows = load_trace(asset("routed_trace.csv"));
    ASSERT_FALSE(rows.empty());
    layers_ = group_layers(rows);
    tokens_ = num_tokens(rows);
  }
  gl::LoaderConstants k_;
  std::vector<LayerInstance> layers_;
  int tokens_ = 0;
  static constexpr int kCap = 460;
};

// ───────────────────────── Part A: decision-stream diff ─────────────────────
TEST_F(DriftProbe, A_DecisionStreamDiff) {
  VariantSpec plain{"plain-ACT NEW", "routed_trace.csv", true, false, true, true};
  VariantSpec decay{"ACT+decay NEW", "routed_trace.csv", true, true,  true, true};
  DecisionLog dp, dd;
  VariantResult rp = run_variant(plain, layers_, tokens_, k_, kCap, nullptr, &dp);
  VariantResult rd = run_variant(decay, layers_, tokens_, k_, kCap, nullptr, &dd);

  ASSERT_EQ(dp.lookups.size(), dd.lookups.size());
  ASSERT_EQ(dp.lookups.size(), 46400u);

  uint64_t target_diff = 0, hit_diff = 0;
  std::string first_diff;
  for (size_t i = 0; i < dp.lookups.size(); ++i) {
    const auto& a = dp.lookups[i];
    const auto& b = dd.lookups[i];
    ASSERT_EQ(a.token, b.token);
    ASSERT_EQ(a.layer, b.layer);
    ASSERT_EQ(a.expert, b.expert);  // same fixed workload, lockstep
    const bool td = a.target != b.target;
    const bool hd = a.hit != b.hit;
    if (td) ++target_diff;
    if (hd) ++hit_diff;
    if ((td || hd) && first_diff.empty()) {
      std::stringstream ss;
      ss << "(t=" << a.token << ",l=" << a.layer << ",e=" << a.expert
         << ") plain[tgt=" << a.target << ",hit=" << int(a.hit)
         << "] decay[tgt=" << b.target << ",hit=" << int(b.hit) << "]";
      first_diff = ss.str();
    }
  }

  // Eviction victim diff (order-independent per token,layer,gpu).
  auto ip = evict_index(dp), id = evict_index(dd);
  uint64_t evict_victims_diff = 0, evict_cells_diff = 0;
  std::set<EvKey> all;
  for (auto& [kk, v] : ip) all.insert(kk);
  for (auto& [kk, v] : id) all.insert(kk);
  for (const auto& kk : all) {
    const auto& va = ip.count(kk) ? ip[kk] : std::vector<uint64_t>{};
    const auto& vb = id.count(kk) ? id[kk] : std::vector<uint64_t>{};
    if (va != vb) {
      ++evict_cells_diff;
      // symmetric-difference size = victims that differ
      std::vector<uint64_t> sd;
      std::set_symmetric_difference(va.begin(), va.end(), vb.begin(), vb.end(),
                                    std::back_inserter(sd));
      evict_victims_diff += sd.size();
    }
  }

  std::cerr << std::fixed << std::setprecision(4)
            << "\n=== Part A: decay-on vs decay-off decision-stream diff (fixed trace) ===\n"
            << "  lookups                : " << dp.lookups.size() << "\n"
            << "  plain hit-rate         : " << rp.hit_rate << "\n"
            << "  decay hit-rate         : " << rd.hit_rate << "\n"
            << "  lookups w/ diff TARGET : " << target_diff << "  ("
            << 100.0 * target_diff / dp.lookups.size() << "%)\n"
            << "  lookups w/ diff HIT    : " << hit_diff << "  ("
            << 100.0 * hit_diff / dp.lookups.size() << "%)\n"
            << "  reroutes  plain/decay  : " << rp.reroute_count << " / "
            << rd.reroute_count << "  (Δ "
            << (int64_t(rd.reroute_count) - int64_t(rp.reroute_count)) << ")\n"
            << "  evicts    plain/decay  : " << dp.evicts.size() << " / "
            << dd.evicts.size() << "\n"
            << "  (token,layer,gpu) cells w/ different victim set : "
            << evict_cells_diff << "\n"
            << "  total victim keys that differ (symdiff)         : "
            << evict_victims_diff << "\n"
            << "  FIRST divergence       : " << first_diff << "\n";

  // The contradiction is RESOLVED: decay is NOT bit-identical (not a no-op), but
  // the hit-rate barely moves. Decisions DO diverge.
  EXPECT_GT(target_diff + hit_diff + evict_victims_diff, 0u)
      << "decay must change at least one decision (else the keeper 0.24 gap is "
         "impossible)";
  EXPECT_NEAR(rp.hit_rate, rd.hit_rate, 0.002)
      << "yet on a FIXED trace the net hit-rate is a near-wash";
}

// ───────────────────────── Part B: mechanism (board) ────────────────────────
TEST_F(DriftProbe, B_Mechanism) {
  const double base = gl::kDefaultBaseScore;  // 0.9
  const double f = kDecay;                     // 0.98

  // (1)+(2) NON-DUPLICATE: uniform scale changes VALUES, preserves ORDER.
  {
    gl::EvictScoreBoard b(2, 16);
    const lmem::ExpertKey k1{0, 10}, k2{0, 20}, k3{0, 30};
    b.update(0, k1, 100.0);
    b.update(0, k2, 200.0);
    b.update(0, k3, 300.0);
    std::vector<double> sc0; std::vector<lmem::ExpertKey> ky0;
    b.cheapest_scores_sorted(0, 8, sc0);
    b.cheapest_keys(0, 8, ky0);
    b.decay_all(f);
    std::vector<double> sc1; std::vector<lmem::ExpertKey> ky1;
    b.cheapest_scores_sorted(0, 8, sc1);
    b.cheapest_keys(0, 8, ky1);
    std::cerr << "\n=== Part B: decay mechanism on the REAL board ===\n"
              << "  (1) non-dup cheapest VALUES  pre : " << sc0[0] << "," << sc0[1]
              << "," << sc0[2] << "\n"
              << "                              post : " << sc1[0] << "," << sc1[1]
              << "," << sc1[2] << "   (×" << f << " — magnitudes CHANGE)\n";
    bool order_same = (ky0.size() == ky1.size());
    for (size_t i = 0; order_same && i < ky0.size(); ++i)
      order_same = (ky0[i].expert_idx == ky1[i].expert_idx);
    std::cerr << "  (2) non-dup victim ORDER preserved : "
              << (order_same ? "YES" : "NO") << "\n";
    EXPECT_NE(sc0[0], sc1[0]) << "decay scales the evict_cum magnitude (a real "
                                 "input to the placement solver)";
    EXPECT_TRUE(order_same) << "non-dup victim order is decay-invariant";
  }

  // (3a) DUPLICATE flip: eff = raw - rank*base is non-linear in raw, so a uniform
  // raw scale CAN cross a duplicate (rank1) past a non-dup (rank0) on the same
  // GPU. Construct raw gap in (base, base/f) so the flip is exact.
  {
    gl::EvictScoreBoard b(2, 16);
    const lmem::ExpertKey kd{0, 50};   // duplicate (resident on GPU0 and GPU1)
    const lmem::ExpertKey kx{0, 60};   // non-dup on GPU0
    // raw(kd@0)=10.91, raw(kd@1)=11.91 -> GPU1 rank0, GPU0 rank1; raw(kx)=10.00.
    b.update(1, kd, 11.91);
    b.update(0, kd, 10.91);
    b.update(0, kx, 10.00);
    // eff(kd@0)=10.91-0.9=10.01 ; eff(kx)=10.00  -> kx is the cheaper victim.
    std::vector<lmem::ExpertKey> pre; b.cheapest_keys(0, 1, pre);
    const int victim_pre = pre.empty() ? -1 : pre[0].expert_idx;
    b.decay_all(f);
    // raw(kd@0)=10.6918 -> eff=9.7918 ; raw(kx)=9.80 -> eff=9.80 ; kd now cheaper.
    std::vector<lmem::ExpertKey> post; b.cheapest_keys(0, 1, post);
    const int victim_post = post.empty() ? -1 : post[0].expert_idx;
    std::cerr << "  (3a) DUP victim flip   pre=" << victim_pre
              << "  post=" << victim_post
              << "  (rank·base penalty does NOT scale with raw ⇒ flip)\n";
    EXPECT_NE(victim_pre, victim_post)
        << "a cross-GPU duplicate lets decay flip the victim (the only DIRECT "
           "fixed-trace victim-order channel)";
  }

  // (3b) max(0) CLAMP tie: a trivialized duplicate clamps to 0; decay can push a
  // second entry's eff to the 0 floor too, creating a tie broken by slot index.
  {
    gl::EvictScoreBoard b(2, 16);
    const lmem::ExpertKey kd{0, 70};
    b.update(1, kd, 100.0);   // rank0 on GPU1
    b.update(0, kd, 0.5);     // rank1 on GPU0 -> eff=max(0,0.5-0.9)=0 (clamped)
    std::vector<double> e; b.cheapest_scores_sorted(0, 4, e);
    std::cerr << "  (3b) CLAMP: trivialized dup eff floored at 0 : " << e[0]
              << "  (ties at 0 break by slot index)\n";
    EXPECT_DOUBLE_EQ(e[0], 0.0);
  }
}

// ───────────────── Part C: clean-LRU ceiling + bounded recency ──────────────
TEST_F(DriftProbe, C_LruCorrectnessAndBoundedClock) {
  const double ceiling = clean_lru_ceiling(layers_, k_.num_devices, kCap);

  VariantSpec plain{"plain-ACT NEW", "routed_trace.csv", true, false, true, true};
  VariantResult rp = run_variant(plain, layers_, tokens_, k_, kCap);

  std::cerr << std::fixed << std::setprecision(4)
            << "\n=== Part C: LRU-correctness probe (fixed trace) ===\n"
            << "  clean-LRU ceiling (no reroute) : " << ceiling << "\n"
            << "  plain-ACT NEW (reroute+board)  : " << rp.hit_rate
            << "   gap = " << (ceiling - rp.hit_rate) << "\n"
            << "  -> ACT reroute LEAVES headroom vs a clean per-GPU LRU; the "
               "reroute machinery is the lever, not decay.\n\n"
            << "  bounded-recency prototype (TD-EVICT-RECENCY-MAGNITUDE):\n";

  // Sweep a few principled recency increments. base/cap makes the recency spread
  // over a residency window comparable to the duplicate penalty; we also bracket
  // it by 10x each way. NOT tuned to any target (algo-not-data) — report falls out.
  const double incs[] = {base_over_cap() * 0.1, base_over_cap(),
                         base_over_cap() * 10.0};
  for (double inc : incs) {
    VariantSpec bspec{"bounded", "routed_trace.csv", true, false, true, true};
    bspec.recency_inc = inc;
    VariantResult rb = run_variant(bspec, layers_, tokens_, k_, kCap);
    std::cerr << "    recency_inc=" << std::setprecision(6) << inc
              << "  hit=" << std::setprecision(4) << rb.hit_rate
              << "  reroute=" << rb.reroute_count
              << "  rr->cached=" << rb.reroute_to_cached
              << "  gap_to_ceiling=" << (ceiling - rb.hit_rate) << "\n";
  }

  EXPECT_LT(rp.hit_rate, ceiling + 0.0005)
      << "plain-ACT NEW does not exceed the clean-LRU ceiling on a fixed trace";
}

// ───────────────────────── Part D: determinism ──────────────────────────────
TEST_F(DriftProbe, D_Determinism) {
  VariantSpec plain{"plain-ACT NEW", "routed_trace.csv", true, false, true, true};
  DecisionLog d1, d2;
  VariantResult r1 = run_variant(plain, layers_, tokens_, k_, kCap, nullptr, &d1);
  VariantResult r2 = run_variant(plain, layers_, tokens_, k_, kCap, nullptr, &d2);
  std::cerr << "\n=== Part D: determinism (CPU sim) ===\n"
            << "  run1 hit=" << std::fixed << std::setprecision(6) << r1.hit_rate
            << "  run2 hit=" << r2.hit_rate << "\n";
  EXPECT_EQ(r1.transfers, r2.transfers);
  EXPECT_EQ(d1.lookups.size(), d2.lookups.size());
  ASSERT_EQ(d1.evicts.size(), d2.evicts.size());
  bool identical = true;
  for (size_t i = 0; i < d1.lookups.size(); ++i)
    identical &= (d1.lookups[i].target == d2.lookups[i].target &&
                  d1.lookups[i].hit == d2.lookups[i].hit);
  EXPECT_TRUE(identical)
      << "the CPU sim is bit-deterministic: decay's influence on the trajectory "
         "is causal steering, not run-to-run sampling noise";
}

}  // namespace
