// CPU unit tests for the loader stats-locality structures: EvictScoreBoard
// (spec/tickets/EVICTBOARD_EXTERNAL_SCORES.md — externally score-driven,
// duplicate-aware ranking) and PlaceConsTable
// (spec/tickets/LOADER_STATS_LOCALITY.md §B). Pure CPU, no CUDA, no expert
// cache. Covers the never-out-of-sync invariant (INV-LOADER-EVICTBOARD-1), the
// duplicate-aware effective-score formula (rank trivialization), every one of
// the 6 mutator API paths, a brute-force-cross-checked randomized stress, the
// place_cons contiguous layout + O(N) gather, and the solver local-index
// correctness (INV-LOADER-LOCALIDX-1 / -PLACECONS-1).
#include "core/gpu_loader/loader_evict_scores.h"
#include "core/gpu_loader/loader_place_cons.h"
#include "core/gpu_loader/loader_solver.h"
#include "core/gpu_loader/loader_constants.h"
#include "core/memory/eviction_policy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace gl = layerstorm::gpu_loader;
namespace mem = layerstorm::memory;

namespace {

mem::ExpertKey key(uint32_t layer, uint16_t expert) { return mem::ExpertKey{layer, expert}; }

}  // namespace

// ── API path 2 (insert_default) + the duplicate-aware accessor ──────────────
// EVICTBOARD_EXTERNAL_SCORES: a freshly-placed slot gets the baseScoreValue raw
// score (the "valuable" side). With M==1 the effective score == raw == base.
TEST(EvictScoreBoard, InsertDefaultAndCheapestScore) {
  gl::EvictScoreBoard b(2, 16);
  EXPECT_EQ(b.resident_count(0), 0);
  EXPECT_DOUBLE_EQ(b.cheapest_score(0), gl::kEmptyScore);  // empty GPU → 0
  EXPECT_DOUBLE_EQ(b.base_score(), gl::kDefaultBaseScore);

  b.insert_default(0, key(3, 10));
  b.insert_default(0, key(3, 11));
  EXPECT_EQ(b.resident_count(0), 2);
  EXPECT_TRUE(b.is_resident(0, key(3, 10)));
  EXPECT_FALSE(b.is_resident(1, key(3, 10)));  // per-GPU
  // single copy (M=1) → rank 0 → effective == raw == base (0.9).
  EXPECT_DOUBLE_EQ(b.score(0, key(3, 10)), gl::kDefaultBaseScore);
  EXPECT_DOUBLE_EQ(b.cheapest_score(0), gl::kDefaultBaseScore);
  EXPECT_DOUBLE_EQ(b.raw_of(0, key(3, 10)), gl::kDefaultBaseScore);
  EXPECT_EQ(b.rank_of(0, key(3, 10)), 0);
  EXPECT_EQ(b.dup_count(key(3, 10)), 1);
  // absent reads return the sentinel.
  EXPECT_DOUBLE_EQ(b.score(0, key(3, 99)), gl::kAbsentScore);
  EXPECT_EQ(b.rank_of(0, key(3, 99)), -1);
  EXPECT_TRUE(b.check_invariant(0));
  EXPECT_TRUE(b.check_invariant(1));
}

// ── decay_all: recency-decay reuse score (per-token raw *= factor) ──────────
// Newest-used stays at base (keep); long-idle decays toward 0 (evict first).
TEST(EvictScoreBoard, DecayAllRecencySignal) {
  gl::EvictScoreBoard b(1, 16);
  b.insert_default(0, key(0, 1));  // "hot" — re-touched every token
  b.insert_default(0, key(0, 2));  // "cold" — never re-touched
  EXPECT_DOUBLE_EQ(b.raw_of(0, key(0, 1)), 0.9);
  EXPECT_DOUBLE_EQ(b.raw_of(0, key(0, 2)), 0.9);

  // 10 tokens: decay everyone by 0.9, then re-touch the hot expert back to base.
  for (int t = 0; t < 10; ++t) {
    b.decay_all(0.9);
    b.insert_default(0, key(0, 1));  // reset-on-use
  }
  // Hot stays at base; cold decayed to 0.9·0.9^10.
  EXPECT_DOUBLE_EQ(b.raw_of(0, key(0, 1)), 0.9);
  EXPECT_NEAR(b.raw_of(0, key(0, 2)), 0.9 * std::pow(0.9, 10), 1e-9);
  // The cold expert is now the cheapest victim (M==1 → eff==raw).
  EXPECT_NEAR(b.cheapest_score(0), 0.9 * std::pow(0.9, 10), 1e-9);
  std::vector<double> v;
  b.cheapest_scores_sorted(0, 2, v, nullptr);
  ASSERT_EQ(v.size(), 2u);
  EXPECT_LT(v[0], v[1]);  // cold (lowest) ranked before hot
  const double hot_before = b.raw_of(0, key(0, 1));
  b.decay_all(1.0);  // factor 1 = no-op
  EXPECT_DOUBLE_EQ(b.raw_of(0, key(0, 1)), hot_before);
  EXPECT_TRUE(b.check_invariant(0));
}

// ── API path 1 (update upsert): insert-absent honors value; present overwrites ─
TEST(EvictScoreBoard, UpdateUpsertSemantics) {
  gl::EvictScoreBoard b(1, 16);
  // absent → INSERT with the supplied value (NOT the default 0.9).
  b.update(0, key(3, 1), 0.42);
  EXPECT_TRUE(b.is_resident(0, key(3, 1)));
  EXPECT_DOUBLE_EQ(b.raw_of(0, key(3, 1)), 0.42);
  EXPECT_DOUBLE_EQ(b.score(0, key(3, 1)), 0.42);  // M=1 → eff == raw
  EXPECT_TRUE(b.check_invariant(0));
  // present → overwrite raw.
  b.update(0, key(3, 1), 0.77);
  EXPECT_DOUBLE_EQ(b.raw_of(0, key(3, 1)), 0.77);
  EXPECT_DOUBLE_EQ(b.score(0, key(3, 1)), 0.77);
  EXPECT_EQ(b.resident_count(0), 1);  // not duplicated
  EXPECT_TRUE(b.check_invariant(0));
}

// ── score() rank formula: M=1/2/3 trivialization + max(0,…) floor + tie ──────
TEST(EvictScoreBoard, EffectiveScoreRankFormula) {
  gl::EvictScoreBoard b(4, 16);
  b.set_base_score(0.9);
  const auto k = key(3, 7);
  // M=1: one copy on g0 raw 1.0 → r0, eff 1.0.
  b.update(0, k, 1.0);
  EXPECT_DOUBLE_EQ(b.score(0, k), 1.0);
  EXPECT_EQ(b.rank_of(0, k), 0);

  // M=2: add a copy on g1 raw 0.8. g0(1.0) → r0 (eff 1.0); g1(0.8) → r1
  // (eff = max(0, 0.8 − 0.9) = 0.0, trivialized).
  b.update(1, k, 0.8);
  EXPECT_EQ(b.dup_count(k), 2);
  EXPECT_DOUBLE_EQ(b.score(0, k), 1.0);
  EXPECT_EQ(b.rank_of(0, k), 0);
  EXPECT_DOUBLE_EQ(b.score(1, k), 0.0);  // max(0, 0.8-0.9) floored
  EXPECT_EQ(b.rank_of(1, k), 1);

  // M=3: add a copy on g2 raw 0.95. Order by raw DESC: g0(1.0) r0, g2(0.95) r1,
  // g1(0.8) r2. eff: g0=1.0; g2 = max(0, 0.95-0.9)=0.05; g1 = max(0,0.8-1.8)=0.0.
  b.update(2, k, 0.95);
  EXPECT_EQ(b.dup_count(k), 3);
  EXPECT_DOUBLE_EQ(b.score(0, k), 1.0);
  EXPECT_NEAR(b.score(2, k), 0.05, 1e-12);
  EXPECT_DOUBLE_EQ(b.score(1, k), 0.0);  // 2 ranks down, floored at 0
  EXPECT_EQ(b.rank_of(0, k), 0);
  EXPECT_EQ(b.rank_of(2, k), 1);
  EXPECT_EQ(b.rank_of(1, k), 2);
  for (int g = 0; g < 4; ++g) EXPECT_TRUE(b.check_invariant(g));

  // tie on raw: two copies with identical raw → LOWER gpu index keeps r0.
  gl::EvictScoreBoard t(2, 16);
  t.set_base_score(0.5);
  t.update(0, key(3, 9), 0.6);
  t.update(1, key(3, 9), 0.6);  // tie
  EXPECT_EQ(t.rank_of(0, key(3, 9)), 0);   // lower gpu kept
  EXPECT_EQ(t.rank_of(1, key(3, 9)), 1);   // higher gpu trivialized
  EXPECT_DOUBLE_EQ(t.score(0, key(3, 9)), 0.6);
  EXPECT_NEAR(t.score(1, key(3, 9)), 0.1, 1e-12);  // max(0, 0.6-0.5)
  EXPECT_TRUE(t.check_invariant(0));
  EXPECT_TRUE(t.check_invariant(1));
}

// ── API path 3 (evict): removes one entry; survivor un-trivializes; no-op ────
TEST(EvictScoreBoard, EvictPromotesSurvivor) {
  gl::EvictScoreBoard b(2, 16);
  b.set_base_score(0.9);
  const auto k = key(3, 1);
  b.update(0, k, 0.9);   // g0 r0 eff 0.9
  b.update(1, k, 0.9);   // g1 r1 eff 0.0 (trivialized duplicate)
  EXPECT_DOUBLE_EQ(b.score(1, k), 0.0);
  EXPECT_DOUBLE_EQ(b.cheapest_score(1), 0.0);  // the duplicate is cheapest

  // evict the PRIMARY copy on g0; g1's copy is now M=1, r0, eff 0.9 (promoted).
  b.evict(0, k);
  EXPECT_FALSE(b.is_resident(0, k));
  EXPECT_EQ(b.dup_count(k), 1);
  EXPECT_DOUBLE_EQ(b.score(1, k), 0.9);  // un-trivialized
  EXPECT_DOUBLE_EQ(b.cheapest_score(1), 0.9);
  EXPECT_TRUE(b.check_invariant(0));
  EXPECT_TRUE(b.check_invariant(1));

  // evict the last copy → expert disappears from the index.
  b.evict(1, k);
  EXPECT_EQ(b.dup_count(k), 0);
  EXPECT_EQ(b.resident_count(1), 0);
  // evict of a non-resident is a no-op (cannot desync).
  b.evict(0, key(3, 77));
  EXPECT_TRUE(b.check_invariant(0));
  EXPECT_TRUE(b.check_invariant(1));
}

// ── API path 4 (batch_upsert): mixed insert/update + multi-GPU same expert ───
TEST(EvictScoreBoard, BatchUpsertMixedAndDedupRerank) {
  gl::EvictScoreBoard b(3, 16);
  b.set_base_score(0.9);
  b.update(0, key(3, 5), 0.3);  // pre-existing on g0
  // a batch that: updates g0/key5; inserts key5 on g1 AND g2 (same expert, 3
  // GPUs in ONE batch → must re-rank against the FINAL entry-set, not partial).
  std::vector<gl::ScoreUpdate> ups = {
      {0, key(3, 5), 1.0},   // raise g0's raw to 1.0
      {1, key(3, 5), 0.95},  // new copy on g1
      {2, key(3, 5), 0.5},   // new copy on g2
      {0, key(3, 6), 0.7},   // unrelated insert
  };
  b.batch_upsert(ups);
  EXPECT_EQ(b.dup_count(key(3, 5)), 3);
  // ranks by raw DESC: g0(1.0) r0, g1(0.95) r1, g2(0.5) r2.
  EXPECT_DOUBLE_EQ(b.score(0, key(3, 5)), 1.0);
  EXPECT_NEAR(b.score(1, key(3, 5)), std::max(0.0, 0.95 - 0.9), 1e-12);
  EXPECT_DOUBLE_EQ(b.score(2, key(3, 5)), 0.0);  // max(0, 0.5 - 1.8)
  EXPECT_DOUBLE_EQ(b.score(0, key(3, 6)), 0.7);
  for (int g = 0; g < 3; ++g) EXPECT_TRUE(b.check_invariant(g));
}

// ── API path 5 (batch_evict): removes many; survivors re-rank ───────────────
TEST(EvictScoreBoard, BatchEvictSurvivorsRerank) {
  gl::EvictScoreBoard b(3, 16);
  b.set_base_score(0.9);
  const auto k = key(3, 2);
  b.update(0, k, 1.0); b.update(1, k, 0.9); b.update(2, k, 0.8);  // 3 copies
  b.update(0, key(3, 3), 0.5);
  EXPECT_EQ(b.dup_count(k), 3);
  // batch-evict the two highest-ranked copies of k + the unrelated key3.
  std::vector<gl::KeyOnGpu> evs = {{0, k}, {1, k}, {0, key(3, 3)}};
  b.batch_evict(evs);
  EXPECT_EQ(b.dup_count(k), 1);
  EXPECT_FALSE(b.is_resident(0, key(3, 3)));
  // the lone survivor (g2, raw 0.8) is now M=1, r0 → eff 0.8 (un-trivialized).
  EXPECT_DOUBLE_EQ(b.score(2, k), 0.8);
  for (int g = 0; g < 3; ++g) EXPECT_TRUE(b.check_invariant(g));
}

// ── API path 6 (update_existing_only): no insert, silently ignore absent (T9) ─
TEST(EvictScoreBoard, UpdateExistingOnlyNeverInserts) {
  gl::EvictScoreBoard b(2, 16);
  b.update(0, key(3, 1), 0.5);  // resident
  std::vector<gl::ScoreUpdate> ups = {
      {0, key(3, 1), 0.9},   // present → updated
      {0, key(3, 99), 0.1},  // absent → IGNORED (no insert)
      {1, key(3, 1), 0.2},   // absent on g1 → IGNORED
  };
  b.update_existing_only(ups);
  EXPECT_DOUBLE_EQ(b.score(0, key(3, 1)), 0.9);  // updated
  EXPECT_FALSE(b.is_resident(0, key(3, 99)));     // NOT inserted
  EXPECT_FALSE(b.is_resident(1, key(3, 1)));      // NOT inserted
  EXPECT_EQ(b.resident_count(0), 1);
  EXPECT_EQ(b.resident_count(1), 0);
  EXPECT_TRUE(b.check_invariant(0));
  EXPECT_TRUE(b.check_invariant(1));
}

// ── insert_default re-touch + M transitions stay synced ─────────────────────
TEST(EvictScoreBoard, InsertDefaultRetouchAndTransitions) {
  gl::EvictScoreBoard b(2, 16);
  b.set_base_score(0.9);
  b.update(0, key(3, 5), 0.2);
  b.insert_default(0, key(3, 5));  // re-touch → raw reset to base 0.9
  EXPECT_DOUBLE_EQ(b.raw_of(0, key(3, 5)), gl::kDefaultBaseScore);
  EXPECT_EQ(b.resident_count(0), 1);  // not duplicated
  // 1→2→1 transitions.
  b.insert_default(1, key(3, 5));  // M:1→2
  EXPECT_EQ(b.dup_count(key(3, 5)), 2);
  b.evict(1, key(3, 5));           // M:2→1
  EXPECT_EQ(b.dup_count(key(3, 5)), 1);
  EXPECT_DOUBLE_EQ(b.score(0, key(3, 5)), gl::kDefaultBaseScore);  // un-trivialized
  EXPECT_TRUE(b.check_invariant(0));
  EXPECT_TRUE(b.check_invariant(1));
}

// ── cheapest_scores_sorted: the k LOWEST effective scores, ascending, O(N) ───
TEST(EvictScoreBoard, CheapestScoresSortedReturnsLowestK) {
  gl::EvictScoreBoard b(1, 256);
  // 50 experts with raw score e/100 → effective == raw (all M=1).
  for (int e = 0; e < 50; ++e)
    b.update(0, key(3, static_cast<uint16_t>(e)), e / 100.0);
  EXPECT_TRUE(b.check_invariant(0));
  std::vector<double> out;
  int rc = 0;
  const int k = 8;
  int got = b.cheapest_scores_sorted(0, k, out, &rc);
  EXPECT_EQ(got, k);
  EXPECT_EQ(rc, 50);
  ASSERT_EQ(out.size(), static_cast<size_t>(k));
  for (int i = 1; i < got; ++i) EXPECT_LE(out[i - 1], out[i]);  // ascending
  // the k lowest == experts 0..7 (lowest raw == lowest eff).
  for (int i = 0; i < k; ++i) EXPECT_NEAR(out[i], i / 100.0, 1e-12);

  int got_all = b.cheapest_scores_sorted(0, 1000, out, &rc);
  EXPECT_EQ(got_all, 50);
  EXPECT_EQ(rc, 50);
  for (int i = 1; i < got_all; ++i) EXPECT_LE(out[i - 1], out[i]);
  EXPECT_EQ(b.cheapest_scores_sorted(0, 0, out, &rc), 0);  // k=0 → empty
}

// ── cheapest_keys: the k LOWEST-effective-score KEYS, ascending, O(k log N) ──
// Key-returning twin of cheapest_scores_sorted (TD-FAR-EVICT-REROUTE step 2).
TEST(EvictScoreBoard, CheapestKeysReturnsLowestK) {
  gl::EvictScoreBoard b(1, 256);
  // 50 experts with raw score e/100 → effective == raw (all M=1), so expert e
  // is the e-th cheapest victim.
  for (int e = 0; e < 50; ++e)
    b.update(0, key(3, static_cast<uint16_t>(e)), e / 100.0);
  EXPECT_TRUE(b.check_invariant(0));

  std::vector<mem::ExpertKey> out;
  int rc = 0;
  const int k = 8;
  int got = b.cheapest_keys(0, k, out, &rc);
  EXPECT_EQ(got, k);
  EXPECT_EQ(rc, 50);
  ASSERT_EQ(out.size(), static_cast<size_t>(k));
  // the k cheapest keys == experts 0..7 ascending (lowest raw == lowest eff).
  for (int i = 0; i < k; ++i) EXPECT_EQ(out[i], key(3, static_cast<uint16_t>(i)));
  // scores along the returned keys are non-decreasing (ascending eff order).
  for (int i = 1; i < got; ++i)
    EXPECT_LE(b.score(0, out[i - 1]), b.score(0, out[i]));

  // k > resident count → clamps to all residents, still ascending.
  int got_all = b.cheapest_keys(0, 1000, out, &rc);
  EXPECT_EQ(got_all, 50);
  EXPECT_EQ(rc, 50);
  for (int i = 0; i < got_all; ++i)
    EXPECT_EQ(out[i], key(3, static_cast<uint16_t>(i)));

  EXPECT_EQ(b.cheapest_keys(0, 0, out, &rc), 0);  // k=0 → empty
  EXPECT_TRUE(out.empty());
}

// Empty GPU → no keys, resident_count 0.
TEST(EvictScoreBoard, CheapestKeysEmptyGpu) {
  gl::EvictScoreBoard b(2, 16);
  std::vector<mem::ExpertKey> out;
  int rc = -1;
  EXPECT_EQ(b.cheapest_keys(0, 4, out, &rc), 0);
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(rc, 0);
  // out-of-range GPU is safe.
  EXPECT_EQ(b.cheapest_keys(5, 4, out, &rc), 0);
  EXPECT_EQ(rc, 0);
}

// cheapest_keys and cheapest_scores_sorted walk the SAME frontier → identical
// slots → key[i] has score == scores[i], including the duplicate-aware ranks
// (a cross-GPU duplicate is trivialized toward 0 = cheapest victim).
TEST(EvictScoreBoard, CheapestKeysConsistentWithScores) {
  const int M = 3;
  const double base = 0.9;
  gl::EvictScoreBoard b(M, 32);
  b.set_base_score(base);
  // Spread of scores on GPU 0, plus a replicated expert across all GPUs so its
  // secondary copies are rank-trivialized (rank>0 → low eff → cheap).
  for (int e = 0; e < 12; ++e)
    b.update(0, key(3, static_cast<uint16_t>(e)), 0.2 + 0.05 * e);
  const auto dup = key(3, 100);
  b.update(0, dup, 0.5);   // r0 on gpu0 (others lower)
  b.update(1, dup, 0.4);
  b.update(2, dup, 0.3);
  EXPECT_TRUE(b.check_invariant(0));

  std::vector<mem::ExpertKey> keys;
  std::vector<double> scores;
  int rck = 0, rcs = 0;
  const int got_k = b.cheapest_keys(0, 100, keys, &rck);
  const int got_s = b.cheapest_scores_sorted(0, 100, scores, &rcs);
  ASSERT_EQ(got_k, got_s);
  ASSERT_EQ(rck, rcs);
  ASSERT_EQ(keys.size(), scores.size());
  for (int i = 0; i < got_k; ++i) {
    // each emitted key's effective score equals the score emitted at the same
    // rank position (same slot frontier order).
    EXPECT_NEAR(b.score(0, keys[i]), scores[i], 1e-12) << "i=" << i;
    if (i) EXPECT_LE(scores[i - 1], scores[i]);  // ascending
  }
  // brute-force cross-check: the returned key set == ALL residents sorted by
  // score() ascending (cheapest victim first).
  std::vector<mem::ExpertKey> all;
  for (int e = 0; e < 12; ++e) all.push_back(key(3, static_cast<uint16_t>(e)));
  all.push_back(dup);
  std::stable_sort(all.begin(), all.end(),
                   [&](mem::ExpertKey a, mem::ExpertKey c) {
                     return b.score(0, a) < b.score(0, c);
                   });
  ASSERT_EQ(all.size(), keys.size());
  for (size_t i = 0; i < all.size(); ++i)
    EXPECT_NEAR(b.score(0, all[i]), b.score(0, keys[i]), 1e-12) << "i=" << i;
  // the rank-trivialized duplicate is the cheapest victim on gpu0? Its eff is
  // 0.5 (r0, primary) so NOT cheapest here; on gpu2 (r2) it is trivialized.
  std::vector<mem::ExpertKey> k2;
  b.cheapest_keys(2, 1, k2, nullptr);
  ASSERT_EQ(k2.size(), 1u);
  EXPECT_EQ(k2[0], dup);  // only resident on gpu2, trivially cheapest
}

// ── set_base_score re-ranks globally (heap order changes for duplicates) ─────
TEST(EvictScoreBoard, SetBaseScoreGlobalRerank) {
  gl::EvictScoreBoard b(2, 16);
  b.set_base_score(0.9);
  const auto k = key(3, 1);
  b.update(0, k, 1.0);  // r0
  b.update(1, k, 0.95); // r1 → eff = max(0, 0.95-0.9) = 0.05
  EXPECT_NEAR(b.score(1, k), 0.05, 1e-12);
  // shrink the penalty: now r1 eff = max(0, 0.95-0.1) = 0.85.
  b.set_base_score(0.1);
  EXPECT_NEAR(b.score(1, k), 0.85, 1e-12);
  EXPECT_DOUBLE_EQ(b.score(0, k), 1.0);  // primary unchanged
  EXPECT_TRUE(b.check_invariant(0));
  EXPECT_TRUE(b.check_invariant(1));
}

// ── Brute-force reference: recompute ranks + effective scores from raw alone ──
namespace {
struct BruteForce {
  int M;
  double base;
  // (gpu -> (key -> raw)).
  std::vector<std::map<std::pair<uint32_t, uint16_t>, double>> raw;
  explicit BruteForce(int m, double b) : M(m), base(b), raw(m) {}

  static std::pair<uint32_t, uint16_t> kk(mem::ExpertKey k) {
    return {k.layer_idx, k.expert_idx};
  }
  void upsert(int g, mem::ExpertKey k, double s) { raw[g][kk(k)] = s; }
  void update_existing(int g, mem::ExpertKey k, double s) {
    auto& m = raw[g];
    auto it = m.find(kk(k));
    if (it != m.end()) it->second = s;
  }
  void erase(int g, mem::ExpertKey k) { raw[g].erase(kk(k)); }
  bool resident(int g, mem::ExpertKey k) const { return raw[g].count(kk(k)) > 0; }
  double raw_of(int g, mem::ExpertKey k) const {
    auto it = raw[g].find(kk(k));
    return it == raw[g].end() ? gl::kAbsentScore : it->second;
  }
  // rank = # of OTHER copies strictly ahead under (raw DESC, gpu ASC).
  int rank(int g, mem::ExpertKey k) const {
    const double my = raw_of(g, k);
    if (my == gl::kAbsentScore) return -1;
    int r = 0;
    for (int o = 0; o < M; ++o) {
      if (o == g) continue;
      auto it = raw[o].find(kk(k));
      if (it == raw[o].end()) continue;
      const double er = it->second;
      if (er > my || (er == my && o < g)) ++r;
    }
    return r;
  }
  double eff(int g, mem::ExpertKey k) const {
    const double my = raw_of(g, k);
    if (my == gl::kAbsentScore) return gl::kAbsentScore;
    return std::max(0.0, my - static_cast<double>(rank(g, k)) * base);
  }
  double cheapest(int g) const {
    double mn = gl::kEmptyScore;
    bool any = false;
    for (const auto& [key_pair, _] : raw[g]) {
      mem::ExpertKey k{key_pair.first, key_pair.second};
      const double e = eff(g, k);
      if (!any || e < mn) { mn = e; any = true; }
    }
    return any ? mn : gl::kEmptyScore;
  }
  std::vector<double> sorted_effs(int g) const {
    std::vector<double> v;
    for (const auto& [key_pair, _] : raw[g]) {
      mem::ExpertKey k{key_pair.first, key_pair.second};
      v.push_back(eff(g, k));
    }
    std::sort(v.begin(), v.end());
    return v;
  }
  int count(int g) const { return static_cast<int>(raw[g].size()); }
};
}  // namespace

// ── randomized stress: 6-method mix, cross-checked vs the brute-force ref ────
TEST(EvictScoreBoard, RandomizedStressVsBruteForce) {
  const int M = 4;
  const double base = 0.9;
  gl::EvictScoreBoard b(M, 64);
  b.set_base_score(base);
  BruteForce ref(M, base);

  std::mt19937 rng(0xC0FFEE);
  std::uniform_int_distribution<int> gpu_d(0, M - 1);
  std::uniform_int_distribution<int> exp_d(0, 24);  // small space → many dups
  std::uniform_int_distribution<int> op_d(0, 5);
  std::uniform_int_distribution<int> batch_d(1, 5);
  std::uniform_real_distribution<double> sc_d(0.0, 2.0);

  for (int it = 0; it < 25000; ++it) {
    switch (op_d(rng)) {
      case 0: {  // update (upsert)
        const int g = gpu_d(rng);
        const auto k = key(3, static_cast<uint16_t>(exp_d(rng)));
        const double s = sc_d(rng);
        b.update(g, k, s);
        ref.upsert(g, k, s);
        break;
      }
      case 1: {  // insert_default
        const int g = gpu_d(rng);
        const auto k = key(3, static_cast<uint16_t>(exp_d(rng)));
        b.insert_default(g, k);
        ref.upsert(g, k, base);
        break;
      }
      case 2: {  // evict
        const int g = gpu_d(rng);
        const auto k = key(3, static_cast<uint16_t>(exp_d(rng)));
        b.evict(g, k);
        ref.erase(g, k);
        break;
      }
      case 3: {  // batch_upsert
        const int n = batch_d(rng);
        std::vector<gl::ScoreUpdate> ups;
        for (int i = 0; i < n; ++i) {
          gl::ScoreUpdate u{gpu_d(rng), key(3, static_cast<uint16_t>(exp_d(rng))),
                            sc_d(rng)};
          ups.push_back(u);
        }
        b.batch_upsert(ups);
        for (const auto& u : ups) ref.upsert(u.gpu, u.key, u.score);
        break;
      }
      case 4: {  // batch_evict
        const int n = batch_d(rng);
        std::vector<gl::KeyOnGpu> evs;
        for (int i = 0; i < n; ++i)
          evs.push_back({gpu_d(rng), key(3, static_cast<uint16_t>(exp_d(rng)))});
        b.batch_evict(evs);
        for (const auto& e : evs) ref.erase(e.gpu, e.key);
        break;
      }
      case 5: {  // update_existing_only
        const int n = batch_d(rng);
        std::vector<gl::ScoreUpdate> ups;
        for (int i = 0; i < n; ++i)
          ups.push_back({gpu_d(rng), key(3, static_cast<uint16_t>(exp_d(rng))),
                         sc_d(rng)});
        b.update_existing_only(ups);
        for (const auto& u : ups) ref.update_existing(u.gpu, u.key, u.score);
        break;
      }
    }

    if (it % 53 == 0) {
      for (int g = 0; g < M; ++g) {
        ASSERT_TRUE(b.check_invariant(g)) << "it=" << it << " g=" << g;
        ASSERT_EQ(b.resident_count(g), ref.count(g)) << "it=" << it << " g=" << g;
        // cheapest effective score matches the brute force.
        ASSERT_NEAR(b.cheapest_score(g), ref.cheapest(g), 1e-9)
            << "it=" << it << " g=" << g;
        // per-resident effective score + rank match.
        for (int e = 0; e <= 24; ++e) {
          const auto k = key(3, static_cast<uint16_t>(e));
          ASSERT_EQ(b.is_resident(g, k), ref.resident(g, k))
              << "it=" << it << " g=" << g << " e=" << e;
          if (ref.resident(g, k)) {
            ASSERT_NEAR(b.score(g, k), ref.eff(g, k), 1e-9)
                << "it=" << it << " g=" << g << " e=" << e;
            ASSERT_EQ(b.rank_of(g, k), ref.rank(g, k))
                << "it=" << it << " g=" << g << " e=" << e;
            ASSERT_NEAR(b.raw_of(g, k), ref.raw_of(g, k), 1e-12);
          }
        }
        // the k-cheapest list matches the brute-force sorted effective scores.
        std::vector<double> ks; int rc = 0;
        int got = b.cheapest_scores_sorted(g, 64, ks, &rc);
        const auto ref_sorted = ref.sorted_effs(g);
        ASSERT_EQ(rc, ref.count(g)) << "it=" << it << " g=" << g;
        ASSERT_EQ(got, static_cast<int>(ref_sorted.size()));
        for (int i = 0; i < got; ++i)
          ASSERT_NEAR(ks[i], ref_sorted[i], 1e-9) << "it=" << it << " i=" << i;
        if (got > 0) ASSERT_NEAR(ks[0], b.cheapest_score(g), 1e-12);
        // cheapest_keys walks the SAME frontier as cheapest_scores_sorted →
        // same count, same per-position effective score (key[i] ⟷ score[i]).
        std::vector<mem::ExpertKey> kk_keys; int rck = 0;
        int got_k = b.cheapest_keys(g, 64, kk_keys, &rck);
        ASSERT_EQ(rck, rc) << "it=" << it << " g=" << g;
        ASSERT_EQ(got_k, got) << "it=" << it << " g=" << g;
        for (int i = 0; i < got_k; ++i)
          ASSERT_NEAR(b.score(g, kk_keys[i]), ks[i], 1e-9)
              << "it=" << it << " i=" << i;
      }
    }
  }
}

// ── PlaceConsTable: contiguous [expert][gpu] layout, set/at/row ─────────────
TEST(PlaceConsTable, LayoutAndAccess) {
  gl::PlaceConsTable t(10, 4);
  EXPECT_EQ(t.num_experts(), 10);
  EXPECT_EQ(t.num_gpus(), 4);
  // default neutral 0.
  EXPECT_DOUBLE_EQ(t.at(3, 2), 0.0);
  t.set(3, 2, -1.5);
  t.set(3, 0, 2.0);
  EXPECT_DOUBLE_EQ(t.at(3, 2), -1.5);
  EXPECT_DOUBLE_EQ(t.at(3, 0), 2.0);
  // INNER stride = gpu: the row is M contiguous doubles.
  const double* r = t.row(3);
  ASSERT_NE(r, nullptr);
  EXPECT_DOUBLE_EQ(r[0], 2.0);
  EXPECT_DOUBLE_EQ(r[2], -1.5);
  // add / set_row.
  t.add(3, 2, 0.5);
  EXPECT_DOUBLE_EQ(t.at(3, 2), -1.0);
  std::array<double, 4> row_vals = {1, 2, 3, 4};
  t.set_row(5, row_vals);
  for (int j = 0; j < 4; ++j) EXPECT_DOUBLE_EQ(t.at(5, j), row_vals[j]);
  // out-of-range is safe (no-op / neutral).
  t.set(-1, 0, 9.0);
  t.set(100, 0, 9.0);
  EXPECT_DOUBLE_EQ(t.at(100, 0), 0.0);
  EXPECT_EQ(t.row(100), nullptr);
}

// ── gather: only the N routed rows, local-index ordered ─────────────────────
TEST(PlaceConsTable, GatherRoutedRowsOnly) {
  const int X = 256, M = 4;
  gl::PlaceConsTable t(X, M);
  // fill a known pattern: at(e,j) = e*100 + j.
  for (int e = 0; e < X; ++e)
    for (int j = 0; j < M; ++j) t.set(e, j, e * 100.0 + j);

  std::vector<int> globals = {7, 200, 1, 255};  // N=4 routed experts (global ids)
  std::vector<double> out;
  t.gather(globals, out);
  ASSERT_EQ(out.size(), globals.size() * M);
  for (int i = 0; i < static_cast<int>(globals.size()); ++i)
    for (int j = 0; j < M; ++j)
      EXPECT_DOUBLE_EQ(out[i * M + j], globals[i] * 100.0 + j)
          << "i=" << i << " j=" << j;

  // OOB / unseen global → neutral zero row.
  std::vector<int> g2 = {7, -1, 9999};
  t.gather(g2, out);
  ASSERT_EQ(out.size(), g2.size() * M);
  for (int j = 0; j < M; ++j) EXPECT_DOUBLE_EQ(out[0 * M + j], 7 * 100.0 + j);
  for (int j = 0; j < M; ++j) EXPECT_DOUBLE_EQ(out[1 * M + j], 0.0);
  for (int j = 0; j < M; ++j) EXPECT_DOUBLE_EQ(out[2 * M + j], 0.0);
}

// ── solver local-index correctness: gathered place == global place value ────
//    (INV-LOADER-LOCALIDX-1 / -PLACECONS-1)
TEST(PlaceConsTable, SolverLocalIndexMatchesGlobal) {
  const int X = 256, M = 2, B = 2, N = 5;
  gl::PlaceConsTable t(X, M);
  std::mt19937 rng(999);
  std::uniform_real_distribution<double> pd(-3.0, 3.0);
  for (int e = 0; e < X; ++e)
    for (int j = 0; j < M; ++j) t.set(e, j, pd(rng));

  // Build a solver constants + request for N routed experts with given globals.
  std::vector<int> globals = {17, 3, 200, 99, 42};
  gl::LoaderConstants k;
  k.source = "test"; k.num_devices = M; k.num_banks = B;
  k.ncf = {0.0, 1.0, 1.2};
  for (int d = 0; d < M; ++d) {
    gl::DeviceConstants dc; dc.position = d; dc.numa_node = d;
    dc.compute = {150.0, 0.0, 1}; k.devices.push_back(dc);
  }
  for (int b = 0; b < B; ++b) {
    gl::BankConstants bc; bc.node = b; bc.egress_us = 50.0; bc.contention = 1.0;
    k.banks.push_back(bc);
  }
  k.matrix.assign(B, std::vector<gl::TransferCell>(M));
  for (int b = 0; b < B; ++b)
    for (int d = 0; d < M; ++d) k.matrix[b][d] = gl::TransferCell{500.0, (b == d) ? 1 : 2, 5.0};

  gl::SolveRequest req;
  req.num_devices = M; req.num_experts = N;
  req.bank_of.assign(N, 0);
  for (int i = 0; i < N; ++i) req.bank_of[i] = i % B;
  req.cached.assign(static_cast<size_t>(N) * M, 0);  // all uncached → place counts
  req.clamp_place = false;  // verify the raw (signed) gathered values flow through

  // gather the N routed rows into req.place (local index order).
  t.gather(globals, req.place);
  ASSERT_EQ(req.place.size(), static_cast<size_t>(N) * M);
  // every place_at(i,j) must equal the GLOBAL table value for globals[i].
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j)
      EXPECT_DOUBLE_EQ(req.place_at(i, j), t.at(globals[i], j));

  // sanity: the solver runs and returns a valid assignment in [0,M).
  gl::LoaderSolver solver;
  auto r = solver.solve(k, req);
  EXPECT_EQ(r.n, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_GE(r.assignment[i], 0);
    EXPECT_LT(r.assignment[i], M);
  }
}
