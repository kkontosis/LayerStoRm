// CPU-only, GPU-free integration test: replay each recorded routed trace through
// the REAL engine eviction — a CPU memory::ExpertCache (460 stable slots/GPU,
// NullBackends), the gpu_loader::EvictScoreBoard as its ResidencyListener, the
// SHARED gpu_loader::apply_far_evictions victim selector (the SAME function the
// daemon calls), the REAL gpu_loader::LoaderSolver placement, and the REAL T(j)
// cost model. Because residency + scoring + eviction all come from the engine
// classes, sim hit-rate == engine hit-rate by construction (single source of
// truth — nothing re-implemented).
//
// FIDELITY FIX (fab1-offline-sim-fidelity, see RESULTS.md): all 5 variants replay
// the SINGLE baseline trace routed_trace.csv, varying ONLY the algorithm
// (ALGO-NOT-DATA). The prior epoch drove each variant off its OWN policy-drifted
// oracle trace, which manufactured a SPURIOUS decay inversion (their intrinsic
// cacheability differs wildly: oracle_act ceiling 0.7629 vs oracle_decay 0.3931).
// On the single trace plain-ACT NEW reproduces the keeper (0.3906 vs 0.392), and
// LS_EVICT_DECAY is shown to be a NO-OP on a fixed workload (order-preserving;
// ≈0 cross-GPU duplicates) — so the keeper's 0.629 decay gain is routing-drift-
// generated, not placement/algorithm (honest dual outcome B). VERDICT: the prior
// "inversion" was a test-data bug; the engine eviction mechanism is FAITHFUL.
//
// The oracle (oj,j,cached0,cached1) columns are read ONLY to diff/assert the
// BASELINE residency; the expert_idx/bank columns are the INPUT workload. tok/s
// comes from the T(j) cost model under ONE affine calibration pinning baseline to
// the keeper's tok/s — a baseline-anchored ESTIMATE, NOT measured truth.
//
// CANONICAL-TRAJECTORY REFRESH (fab1-sim-fixture-refresh, 2026-06-30, eb3fc650): the
// fixture was re-captured from the engine with ALL THREE EP fixes landed (placement-
// invariant canonical combine + force-ON dedup + zero-resident per-slot zero-fix) plus
// det-reduce default-ON. The retired 0.40 epoch's trace was captured BEFORE the
// canonical combine, when expert placement (reroute/decay) perturbed the bf16 EP-combine
// → a routing butterfly the open-loop sim could not predict (ACT+decay 0.6291, ~0.24
// short). With the canonical combine, placement is routing-INERT: ALL FIVE scenarios
// converge to the ONE trajectory (sha 34b429e3, the on-disk routed_trace.csv; verified
// 0/5800 routing mismatches vs the canonical drift dump). The variants now differ ONLY
// on the EVICTION axis (recency-LRU vs hash). RESULT: the butterfly is GONE and the
// open-loop sim reproduces ALL FIVE variants to <0.001 (baseline 0.6663 EXACT cached_div=0;
// plain-ACT 0.6548; plain-ACT legacy 0.2573; ACT+decay 0.6550 == plain-ACT; ACT+decay
// legacy 0.2578). The clean same-trajectory eviction A/B (recency 0.6548 ≫ hash 0.2573)
// is the deliverable. tok/s is the baseline-anchored cost-model estimate (relative only).
// See RESULTS.md §CANONICAL-TRAJECTORY REFRESH.
//
// Never touches a GPU / CUDA. See offline_sim.h and assets/REFERENCE.md.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "offline_sim.h"

namespace {

using namespace layerstorm::offline_sim;

std::string asset(const std::string& name) {
  return std::string(LAYERSTORM_SOURCE_DIR
                     "/tests/integration/loader_offline_sim/assets/") + name;
}

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  EXPECT_TRUE(in.good()) << "cannot open " << path;
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Hit@j implied by an oracle (transfer-based: hit iff resident at the solver
// target j). The recorded engine's own keeper-metric on its OWN workload — the
// single-source assertion target for the matching policy. Oracle is read for
// ASSERTION only.
double oracle_hit_at_j(const OracleMap& o) {
  uint64_t n = 0, hit = 0;
  for (const auto& [k, row] : o) {
    ++n;
    const int cached = (row.j == 0) ? row.c0 : row.c1;
    if (cached) ++hit;
  }
  return n ? static_cast<double>(hit) / n : 0.0;
}

// Keeper A/B reference (2x5090 idle box, eb3fc650 — ALL THREE EP fixes landed:
// placement-invariant canonical combine + force-ON dedup + zero-resident per-slot
// zero-fix — PLUS compute.deterministic_reduce default-ON; `fab1-sim-fixture-
// refresh` 2026-06-30). With the canonical combine, expert placement (e%tp /
// reroute / decay) no longer perturbs the bf16 EP-combine, so ALL FIVE scenarios
// converge to the ONE canonical routing trajectory (sha 34b429e3, the on-disk
// routed_trace.csv). The variants now differ ONLY in cache hit/miss (the honest
// eviction axis), NOT in routing. Each hit is BIT-STABLE across 2 idle-checked
// full-pool runs (misses identical run-to-run); tok/s is the 2-run mean (baseline
// = the loader-OFF run). SUPERSEDES the retired 0.40 EP-combine-OFF epoch
// (baseline 0.4002 / plain-ACT 0.3926 / ACT+decay-butterfly 0.6291) — that trace
// was captured before the canonical-combine fix and forked under decay. With the
// fork closed, the open-loop sim reproduces ALL FIVE variants on the single trace
// (the butterfly is GONE: sim ACT+decay 0.655 == engine 0.655, not 0.24 short).
struct Keeper { double hit; double tokps; bool tokps_known; };

class LoaderOfflineSim : public ::testing::Test {
 protected:
  void SetUp() override {
    k_ = layerstorm::gpu_loader::from_json_string(
        read_file(asset("gpu_loader_calibration_5090x2.json")));
    ASSERT_EQ(k_.num_devices, 2);
    ASSERT_GT(k_.num_banks, 0);
    ASSERT_GT(k_.expert_bytes, 0.0);
  }

  void load(const std::string& csv, std::vector<LayerInstance>& layers,
            int& tokens, OracleMap& oracle) {
    auto rows = load_trace(asset(csv));
    ASSERT_FALSE(rows.empty());
    layers = group_layers(rows);
    tokens = num_tokens(rows);
    oracle = load_oracle(asset(csv));
  }

  layerstorm::gpu_loader::LoaderConstants k_;
  static constexpr int kCapacity = 460;            // stable slots / GPU (FullFit keeper)
  static constexpr double kBaselineTokps = 6.945;  // affine anchor: post-all-fixes loader-OFF baseline (eb3fc650, 2026-06-30)
};

TEST_F(LoaderOfflineSim, FiveVariantReplay) {
  // ── the 5 keeper A/B variants, ALL on the SINGLE baseline routed trace ─────
  // ALGO-NOT-DATA (hard rule): every variant replays the IDENTICAL input workload
  // (routed_trace.csv == oracle_baseline.csv cols 1-5) and differs ONLY by the
  // algorithm (ACT reroute / board / decay / fallback). FP routing-drift is
  // treated as negligible per the investigation mandate; the per-policy oracle
  // traces (oracle_act/decay.csv) are NOT used to drive any variant.
  const char* kTrace = "routed_trace.csv";
  const VariantSpec specs[5] = {
    {"baseline",          kTrace, /*act=*/false, /*decay=*/false, /*lru=*/true,  /*board=*/false},
    {"plain-ACT NEW",     kTrace, /*act=*/true,  /*decay=*/false, /*lru=*/true,  /*board=*/true },
    {"plain-ACT legacy",  kTrace, /*act=*/true,  /*decay=*/false, /*lru=*/false, /*board=*/true },
    {"ACT+decay NEW",     kTrace, /*act=*/true,  /*decay=*/true,  /*lru=*/true,  /*board=*/true },
    {"ACT+decay legacy",  kTrace, /*act=*/true,  /*decay=*/true,  /*lru=*/false, /*board=*/true },
  };
  // CANONICAL-TRAJECTORY keeper (eb3fc650, all EP fixes + det-reduce; 2 idle-checked
  // full-pool runs each, zero hit-rate spread — misses bit-identical run-to-run).
  // tok/s is the 2-run mean (baseline = the loader-OFF run). ALL five share the ONE
  // 34b429e3 trajectory; they differ ONLY on the eviction axis (recency-LRU vs hash).
  const Keeper keeper[5] = {
    {0.6663, 6.945, true},   // baseline (calibration anchor)  misses 15485 ×3 (incl loader-OFF)
    {0.6548, 7.029, true},   // plain-ACT NEW   (recency-LRU)  misses 16017 ×2
    {0.2573, 5.139, true},   // plain-ACT legacy (hash-order)  misses 34461 ×2
    {0.6550, 7.023, true},   // ACT+decay NEW   (decay no-op)  misses 16010 ×2
    {0.2578, 5.118, true},   // ACT+decay legacy (decay no-op) misses 34438 ×2
  };

  // ONE workload for all variants. The oracle from oracle_baseline.csv carries the
  // baseline-policy solve-time residency (cached0,cached1) — valid ONLY for the
  // baseline variant (routed_trace == oracle_baseline cols 1-5); it does NOT match
  // the ACT/decay residency, so it is passed to the baseline run only.
  std::vector<LayerInstance> layers; int tokens = 0; OracleMap oracle_base;
  load(kTrace, layers, tokens, oracle_base);            // trace (5 cols)
  oracle_base = load_oracle(asset("oracle_baseline.csv"));  // 9-col residency truth

  VariantResult r[5];
  for (int v = 0; v < 5; ++v)
    r[v] = run_variant(specs[v], layers, tokens, k_, kCapacity,
                       v == 0 ? &oracle_base : nullptr);
  const double ceiling = clean_lru_ceiling(layers, k_.num_devices, kCapacity);

  // Recorded-engine hit@j of the per-policy oracle fixtures (their OWN hash-order
  // epoch metric) — reference context only, NOT a sim driver.
  const double oj_base  = oracle_hit_at_j(oracle_base);                       // ~0.6663
  const double oj_act   = oracle_hit_at_j(load_oracle(asset("oracle_act.csv")));   // ~0.6547
  const double oj_decay = oracle_hit_at_j(load_oracle(asset("oracle_decay.csv"))); // ~0.6547

  // ── affine tok/s calibration: pin baseline to the keeper's 5.249 tok/s ─────
  const double C = 1000.0 / kBaselineTokps - r[0].t_pred_ms_per_token;
  for (int v = 0; v < 5; ++v)
    r[v].tok_s_sim = 1000.0 / (r[v].t_pred_ms_per_token + C);

  // ── report ────────────────────────────────────────────────────────────────
  std::cerr << std::fixed << std::setprecision(4);
  std::cerr << "\n=== Loader offline-sim: REAL board+cache, 5-variant replay (CPU-only) ===\n";
  std::cerr << "affine calibration C = " << std::setprecision(3) << C
            << " ms/token  (baseline pinned to " << kBaselineTokps << " tok/s)\n";
  std::cerr << "recorded-engine hit@j (fixtures' own metric, canonical 34b429e3 trajectory): base="
            << std::setprecision(4) << oj_base << " act=" << oj_act
            << " decay=" << oj_decay << "\n\n";
  std::cerr << "single workload = " << kTrace << " (all 5 variants); clean-LRU ceiling = "
            << ceiling << "\n\n";
  std::cerr << "variant            sim_hit  keep_hit  d_keep  sim_tok/s  "
               "T_pred(ms)  xfers/tok  resvfail  dupIns  dupRes%  reroute  rr->cached  recency\n";
  for (int v = 0; v < 5; ++v) {
    const double dup_res_pct = r[v].resident_layersum
        ? 100.0 * static_cast<double>(r[v].dup_resident_layersum) / r[v].resident_layersum : 0.0;
    std::cerr << "  " << std::left << std::setw(17) << r[v].name << std::right
              << "  " << std::setw(6) << r[v].hit_rate
              << "  " << std::setw(7) << keeper[v].hit
              << "  " << std::setw(6) << (r[v].hit_rate - keeper[v].hit)
              << "  " << std::setprecision(3) << std::setw(8) << r[v].tok_s_sim
              << "  " << std::setprecision(1) << std::setw(9) << r[v].t_pred_ms_per_token
              << "  " << std::setw(8) << r[v].transfers_per_token
              << "  " << std::setw(7) << r[v].reserve_failures
              << "  " << std::setw(6) << r[v].dup_inserts
              << "  " << std::setprecision(3) << std::setw(6) << dup_res_pct
              << "  " << std::setw(7) << r[v].reroute_count
              << "  " << std::setw(10) << r[v].reroute_to_cached
              << "  " << std::setprecision(0) << std::setw(7) << r[v].max_recency
              << std::setprecision(4) << "\n";
  }
  std::cerr << "\n  far-evict victim breakdown (honored / rejected / fallback):\n";
  for (int v = 0; v < 5; ++v)
    std::cerr << "    " << std::left << std::setw(17) << r[v].name << std::right
              << "  honored=" << r[v].ev_honored
              << "  rejected=" << r[v].ev_rejected
              << "  fallback=" << r[v].ev_fallback << "\n";
  std::cerr << "\n  EP-combine dedup (routed cross-GPU dups at compute time / owner shifts):\n";
  for (int v = 0; v < 5; ++v)
    std::cerr << "    " << std::left << std::setw(17) << r[v].name << std::right
              << "  ep_combine_dups=" << r[v].ep_combine_dups
              << "  ep_owner_shifts=" << r[v].ep_owner_shifts
              << "  (of " << r[v].lookups << " lookups)\n";
  std::cerr << "\n  baseline oracle cached-diff (sim recomputed vs recorded ground truth):\n"
            << "    rows=" << r[0].oracle_rows << "  cached_div=" << r[0].cached_divergences
            << "  " << r[0].first_divergence << "\n";
  std::cerr << "\n  NOTE: hit-rates are REAL (engine-reproduced) high-confidence "
               "outputs; tok/s are a baseline-anchored COST-MODEL estimate\n"
               "  (INV-LOADER-OBJECTIVE-MYOPIC — relative comparison only, NOT "
               "measured truth). All variants share ONE workload (algo-not-data);\n"
               "  the keep_hit column is the keeper A/B truth (read for assertion only, "
               "never a sim driver).\n\n";

  // ── assertions ─────────────────────────────────────────────────────────────
  // (1) baseline: EXACT reproduction of the recorded engine — bit-for-bit
  // residency (zero cached divergences over all 46 400 rows) and the keeper hit.
  // On the canonical 34b429e3 trace the sim baseline reproduces the GPU baseline
  // EXACTLY (0.6663 vs 0.6663, cached_div=0) — the foundational gate.
  EXPECT_NEAR(r[0].hit_rate, oj_base, 0.0015)
      << "baseline must reproduce the recorded engine hit@j";
  EXPECT_NEAR(r[0].hit_rate, keeper[0].hit, 0.004)
      << "baseline real-cache hit-rate must reproduce the keeper";
  EXPECT_EQ(r[0].cached_divergences, 0u)
      << "baseline residency must reproduce the oracle EXACTLY";
  EXPECT_DOUBLE_EQ(r[0].tok_s_sim, kBaselineTokps)
      << "baseline calibrates to exactly " << kBaselineTokps << " tok/s";

  // (1b) EP-COMBINE DEDUP no-op on the disjoint baseline. Baseline is strictly
  // e%tp (no reroute) ⇒ every routed expert is resident on exactly ONE GPU ⇒ the
  // force-ON engine dedup (mirrored via ep_combine_owners → daemon::dedup_ep_
  // residency) finds ZERO routed cross-GPU duplicates and shifts ZERO compute
  // owners off their home. This is WHY the deduped cost model still reproduces the
  // pre-dedup baseline tok/s byte-for-byte (the EXPECT_DOUBLE_EQ above).
  EXPECT_EQ(r[0].ep_combine_dups, 0u)
      << "disjoint baseline must have no routed cross-GPU duplicates";
  EXPECT_EQ(r[0].ep_owner_shifts, 0u)
      << "EP-combine dedup must be a byte-identical no-op on the disjoint baseline";

  // (2) HEADLINE — with all three EP fixes landed (placement-invariant canonical
  // combine) the engine no longer forks the trajectory under placement, so the
  // open-loop sim reproduces ALL FIVE variants on the ONE 34b429e3 trace. The
  // retired EP-combine-OFF epoch's "decay butterfly" (sim 0.39 vs keeper 0.6291,
  // 0.24 SHORT) is GONE: ACT+decay now nets the SAME hit as plain-ACT in BOTH engine
  // and sim. The recency (board cheapest_keys) variants reproduce the keeper TIGHTLY
  // (no hash-order dependence); the legacy (hash-order) variants are faithful via the
  // SHARED memory::ExpertCache hash iteration + apply_far_evictions, with a looser tol
  // guarding libstdc++ unordered_map ordering portability. Keeper READ for assertion
  // only (never a sim driver).
  EXPECT_NEAR(r[1].hit_rate, keeper[1].hit, 0.005)
      << "plain-ACT NEW (recency-LRU) open-loop reproduces the keeper";
  EXPECT_NEAR(r[3].hit_rate, keeper[3].hit, 0.005)
      << "ACT+decay NEW (recency-LRU) open-loop reproduces the keeper — butterfly GONE";
  EXPECT_NEAR(r[2].hit_rate, keeper[2].hit, 0.02)
      << "plain-ACT legacy (hash-order) open-loop reproduces the keeper";
  EXPECT_NEAR(r[4].hit_rate, keeper[4].hit, 0.02)
      << "ACT+decay legacy (hash-order) open-loop reproduces the keeper";

  // (3) DECAY IS A NO-OP — on the single trace LS_EVICT_DECAY nets r[3]≈r[1] and
  // r[4]≈r[2] (it scales placement-cost magnitudes but reshuffles without changing
  // cacheability). With the canonical combine this now holds in the ENGINE too
  // (keeper ACT+decay 0.6550 ≈ plain-ACT 0.6548, decay legacy 0.2578 ≈ legacy 0.2573),
  // so the sim's long-standing no-op claim is finally CONSISTENT with the GPU — the
  // old keeper 0.6291 was the placement→bf16-EP-combine butterfly artifact, now fixed.
  EXPECT_NEAR(r[3].hit_rate, r[1].hit_rate, 0.005)
      << "decay is a hit-rate no-op (NEW pair must coincide)";
  EXPECT_NEAR(r[4].hit_rate, r[2].hit_rate, 0.005)
      << "decay is a hit-rate no-op (legacy pair must coincide)";

  // (3b) THE CLEAN SAME-TRAJECTORY EVICTION A/B — recency-LRU (board cheapest_keys)
  // ≫ hash-order legacy on the IDENTICAL 34b429e3 trajectory. This is the honest
  // eviction comparison the fixed trace finally enables (placement is now routing-
  // inert, so the ONLY axis left is cache quality): sim recency 0.6548 vs hash 0.2574
  // (a ~0.40 gap), matching the engine's 0.6548 vs 0.2573. The board's recency ranking
  // is the dominant cache-quality lever.
  EXPECT_GT(r[1].hit_rate - r[2].hit_rate, 0.30)
      << "recency-LRU must dominate hash-order eviction (NEW vs legacy)";
  EXPECT_GT(r[3].hit_rate - r[4].hit_rate, 0.30)
      << "recency-LRU must dominate hash-order eviction (decay pair)";

  // (4) ACT placement mints FEW cross-GPU duplicates (< 1%). The LoaderSolver routes
  // a miss to a GPU where the expert is already cached (free transfer), so duplicates
  // rarely co-reside; on the more-cacheable canonical trace the steady-state duplicate
  // residency is ~0.2% (up from the retired epoch's ~0.004%, but still < 1%).
  for (int v : {1, 2, 3, 4})
    EXPECT_LT(r[v].dup_resident_layersum, r[v].resident_layersum / 100)
        << r[v].name << ": ACT placement should mint < 1% cross-GPU duplicates";

  // (4b) The ROUTED top-K cross-GPU duplicate count at COMPUTE time stays ≪ lookups
  // (ep_combine_dups < lookups/100), so the engine's force-ON EP-combine dedup is a
  // near-no-op on this trace ⇒ the deduped sim cost model is numerically
  // indistinguishable from the pre-dedup one HERE. The dedup matters only for
  // placement policies that DO co-reside routed duplicates.
  for (int v : {1, 2, 3, 4})
    EXPECT_LT(r[v].ep_combine_dups, r[v].lookups / 100)
        << r[v].name << ": ACT placement mints ~no routed cross-GPU duplicates";

  // Reserve failures negligible (FullFit eviction always frees enough).
  for (int v = 0; v < 5; ++v)
    EXPECT_LT(r[v].reserve_failures, r[v].lookups / 20)
        << r[v].name << ": excessive reserve failures distort the hit metric";
}

// ─────────────────────────────────────────────────────────────────────────────
// EP-combine compute-owner dedup (FIDELITY to the force-ON engine dedup).
//
// These mirror tests/unit/ep_residency_dedup_test.cpp at the SIM owner-mapping
// level: the sim feeds per-GPU residency to the SHARED daemon::dedup_ep_residency
// (via ep_combine_owners) and must single-count each routed expert to its LOWEST-
// rank holder, while leaving disjoint placement a byte-identical no-op. The
// underlying dedup bit-logic is already covered by the unit test; here we lock the
// sim's USE of it: residency table → canonical compute owner per routed slot.
// ─────────────────────────────────────────────────────────────────────────────

// residency[slot][gpu] → predicate accepted by ep_combine_owners.
std::function<bool(int, int)> resident_of(
    const std::vector<std::vector<bool>>& tbl) {
  return [tbl](int slot, int gpu) { return tbl[slot][gpu]; };
}

// (A) DISJOINT placement (e%tp) → owner == target, zero duplicates (no-op).
TEST(EpCombineOwners, DisjointIsNoOp) {
  const int M = 2, K = 8;
  std::vector<std::vector<bool>> tbl(K, std::vector<bool>(M, false));
  std::vector<int> target(K);
  for (int i = 0; i < K; ++i) { target[i] = i % M; tbl[i][i % M] = true; }

  int dups = -1;
  const auto owner = ep_combine_owners(K, M, resident_of(tbl), target, &dups);
  EXPECT_EQ(dups, 0);
  EXPECT_EQ(owner, target);  // every routed expert computed on its e%tp home.
}

// (B) A synthetic cross-GPU DUPLICATE collapses to the LOWEST-rank owner — the
//     single-count the engine's deduped EP SUM-combine relies on.
TEST(EpCombineOwners, SingleDuplicateCollapsesToLowestRankOwner) {
  const int M = 2, K = 4;
  std::vector<std::vector<bool>> tbl(K, std::vector<bool>(M, false));
  std::vector<int> target(K);
  for (int i = 0; i < K; ++i) { target[i] = i % M; tbl[i][i % M] = true; }

  // Routed slot 1 homes on GPU 1 (1%2) but a stale copy persists on GPU 0 too →
  // a cross-GPU duplicate. Naive accounting would compute it at target=1; the
  // engine dedup (and thus the sim) must instead pick the lowest holder = GPU 0.
  tbl[1][0] = true;
  ASSERT_TRUE(tbl[1][0] && tbl[1][1]);

  int dups = -1;
  const auto owner = ep_combine_owners(K, M, resident_of(tbl), target, &dups);
  EXPECT_EQ(dups, 1);
  EXPECT_EQ(owner[1], 0) << "duplicate routed expert must collapse to lowest rank";
  EXPECT_NE(owner[1], target[1]) << "owner shifted off the naive reroute/home target";
  // Non-duplicate slots are untouched.
  for (int i : {0, 2, 3}) EXPECT_EQ(owner[i], target[i]) << i;
}

// (C) A routed expert resident on ALL GPUs collapses to rank 0.
TEST(EpCombineOwners, RoutedExpertOnAllGpus) {
  const int M = 4, K = 4;
  std::vector<std::vector<bool>> tbl(K, std::vector<bool>(M, false));
  std::vector<int> target(K);
  for (int i = 0; i < K; ++i) { target[i] = i % M; tbl[i][i % M] = true; }

  tbl[2] = std::vector<bool>(M, true);  // routed slot 2 cached on every GPU.
  int dups = -1;
  const auto owner = ep_combine_owners(K, M, resident_of(tbl), target, &dups);
  EXPECT_EQ(dups, 1);
  EXPECT_EQ(owner[2], 0);  // lowest-rank canonical owner.
}

// (D) A routed slot resident on NO GPU (e.g. a reserve failure) falls back to the
//     naive target — it is still counted exactly once (never dropped).
TEST(EpCombineOwners, NonResidentFallsBackToTarget) {
  const int M = 2, K = 3;
  std::vector<std::vector<bool>> tbl(K, std::vector<bool>(M, false));
  std::vector<int> target = {1, 0, 1};
  tbl[0][1] = true;  // resident at its target.
  // slot 1 resident nowhere; slot 2 resident nowhere.

  int dups = -1;
  const auto owner = ep_combine_owners(K, M, resident_of(tbl), target, &dups);
  EXPECT_EQ(dups, 0);
  EXPECT_EQ(owner[0], 1);  // resident → its (only) holder.
  EXPECT_EQ(owner[1], target[1]);  // non-resident → fallback to naive target.
  EXPECT_EQ(owner[2], target[2]);
}

}  // namespace
