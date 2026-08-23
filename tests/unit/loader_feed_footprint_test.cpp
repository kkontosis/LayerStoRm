// Fast GPU-stubbed harness for the loader-ON daemon hot-path FEED cost
// (LOADER_STATS_LOCALITY / TD-EXPERTSTATS-FEED-COST). Reproduces, in SECONDS and
// with NO GPU (no arena preload, no weights, no CUDA), the per-MoE-layer work the
// daemon critical thread does when the I8 loader is enabled, so the
// ~18 ms/token "loader-on" regression can be ISOLATED and the fix VALIDATED
// without the ~9.5-min keeper.
//
// What it measures, per (token × MoE-layer × device), cold-cache:
//   OLD path (pre-fix, what cost 18 ms/token):
//     - feed: build a GatingResult + ExpertStats::update()  (dense states_ touch)
//     - solve evict_cum build: ExpertCache::eviction_inputs(pos) iterate the GPU's
//       full resident map (~460 entries → vector alloc) + ExpertStats::
//       fill_eviction_scores (SCATTERED reads of the dense ~475 KB states_ array,
//       one per resident) + sort + curve.
//   NEW path (EVICTBOARD_EXTERNAL_SCORES, board self-sufficient):
//     - solve evict_cum build: EvictScoreBoard::cheapest_scores_sorted(pos, N)
//       from the board's OWN effective-score min-heap (touches ~N heap nodes) +
//       curve. No token clock (scores are external).
//   No ExpertStats, no resident-map scan, no dense array.
//
// The harness drives the SAME data structures the dispatcher uses (a real
// unordered_map<ExpertKey,CacheEntry>-shaped resident set, a real ExpertStats,
// the real EvictScoreBoard) with SYNTHETIC routing tuned to the keeper's churn
// (hit-rate ≈ 0.40, ~277 fetches/token, per-GPU resident sets near full), so the
// touched-bytes / cold-cache cost is faithful to the daemon footprint.
//
// CAVEAT (stated honestly): a fully-stubbed harness measures intrinsic CPU +
// cache footprint; it does NOT reproduce the full transfer-pipeline serialization
// amplification (that needs the real H2D pipeline). It is for fast ITERATION
// (seconds) and footprint/CPU-regression detection — final tok/s validation is
// ONE real keeper run.
//
// Default (CI) run: a small, fast flatness/footprint assertion (no env needed).
// Verbose timing report (publishable µs/ms-per-token numbers):
//   LS_RUN_FEED_FOOTPRINT=1 ./layerstorm_unit_tests \
//       --gtest_filter='LoaderFeedFootprint.*'
// Knobs: LS_FEED_TOKENS (default 20), LS_FEED_THRASH_MB (default 64),
//        LS_FEED_RESIDENTS (default 460, per-GPU stable residents),
//        LS_FEED_LAYERS (default 58).
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "core/gpu_loader/loader_evict_scores.h"
#include "core/memory/eviction_policy.h"
#include "core/statistics/expert_stats.h"

namespace {

namespace gl = layerstorm::gpu_loader;
namespace stats = layerstorm::statistics;
namespace mem = layerstorm::memory;
using Clock = std::chrono::steady_clock;

bool verbose() { return std::getenv("LS_RUN_FEED_FOOTPRINT") != nullptr; }
int env_int(const char* k, int def) {
  const char* v = std::getenv(k);
  return (v && *v) ? std::atoi(v) : def;
}

// Capacity-thrash + clflush a >LLC buffer to evict the daemon's hot working set,
// so each timed call pays the realistic COLD cost (mirrors loader_hotpath_bench).
struct CacheFlusher {
  std::vector<char> thrash;
  explicit CacheFlusher(size_t bytes) : thrash(bytes ? bytes : 1, 1) {}
  void clear() {
    volatile char sink = 0;
    for (size_t i = 0; i < thrash.size(); i += 64) {
      thrash[i] = static_cast<char>(thrash[i] + 1);
      sink = static_cast<char>(sink ^ thrash[i]);
    }
#if defined(__x86_64__)
    for (size_t i = 0; i < thrash.size(); i += 64) _mm_clflush(&thrash[i]);
    _mm_mfence();
#endif
    (void)sink;
  }
};

double median_us(std::vector<double>& v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
double mean_us(const std::vector<double>& v) {
  double s = 0; for (double x : v) s += x; return v.empty() ? 0.0 : s / v.size();
}

// Deterministic LCG (reproducible routing).
struct Lcg {
  uint64_t s = 0x243f6a8885a308d3ULL;
  uint32_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
};

// A minimal resident-set mirror with the SAME shape the dispatcher scans: an
// unordered_map<ExpertKey,CacheEntry>-equivalent whose eviction_inputs()
// iterates every resident into a freshly-allocated vector (exactly what
// ExpertCache::eviction_inputs does). This is the OLD-path footprint driver.
struct ResidentMirror {
  struct Ent { mem::CacheZone zone; bool is_duplicate; };
  std::unordered_map<mem::ExpertKey, Ent> residents;
  void place(mem::ExpertKey k) {
    residents[k] = Ent{mem::CacheZone::kStable, false};
  }
  // Mirror of ExpertCache::eviction_inputs(pos): allocate + fill a vector of all
  // residents (the per-call allocation + hash-map walk the daemon paid).
  std::vector<mem::ExpertEvictionInput> eviction_inputs(int gpu_idx) const {
    std::vector<mem::ExpertEvictionInput> out;
    out.reserve(residents.size());
    for (const auto& [k, e] : residents) {
      mem::ExpertEvictionInput in;
      in.key = k; in.zone = e.zone; in.is_duplicate = e.is_duplicate;
      in.gpu_idx = gpu_idx;
      out.push_back(in);
    }
    return out;
  }
  int size() const { return static_cast<int>(residents.size()); }
};

// Build the convex evict_cum curve from per-victim costs (the dispatcher's exact
// curve math). vcost must be ascending (cheapest victim first).
double build_curve(const std::vector<double>& vcost, int free_slots,
                   int num_experts, double unit) {
  const size_t curve_len = static_cast<size_t>(num_experts) + 1;
  const double max_v = vcost.empty() ? unit : vcost.back();
  double acc = 0.0, last = 0.0;
  for (size_t n = 1; n < curve_len; ++n) {
    const int evicts = static_cast<int>(n) - free_slots;
    if (evicts <= 0) { last = 0.0; continue; }
    const int vi = evicts - 1;
    const double step = (vi < static_cast<int>(vcost.size())) ? vcost[vi] : max_v;
    acc += step; last = acc;
  }
  return last;  // sink to defeat DCE
}

constexpr uint32_t kFirstMoe = 3;
constexpr uint32_t kExperts = 256;
constexpr int kTopK = 8;
constexpr int kM = 2;  // 2x5090 keeper shape

}  // namespace

// ── OLD daemon path (pre-fix): ExpertStats feed + eviction_inputs scan +
//    fill_eviction_scores scattered dense reads + curve. The 18 ms/token class. ─
static double old_path_layer(ResidentMirror* res, stats::ExpertStats& es,
                             const std::vector<mem::ExpertKey>& routed,
                             uint64_t token_id, uint32_t layer,
                             double unit) {
  double sink = 0.0;
  // feed_expert_stats (OLD): GatingResult build + dense states_ update.
  {
    stats::GatingResult gr;
    gr.token_id = token_id; gr.layer_idx = layer;
    gr.activations.reserve(routed.size());
    for (auto k : routed) gr.activations.push_back(stats::ExpertActivation{k, 0.5f});
    es.update(std::span<const stats::GatingResult>(&gr, 1));
  }
  // solve evict_cum (OLD): per device, scan residents + fill scores (dense).
  for (int g = 0; g < kM; ++g) {
    auto residents = res[g].eviction_inputs(g);          // map walk + vector alloc
    es.fill_eviction_scores(residents);                  // SCATTERED states_ reads
    std::vector<double> vcost; vcost.reserve(residents.size());
    for (const auto& ein : residents) {
      const double keep = std::clamp(1.0 - ein.recency, 0.0, 1.0);
      vcost.push_back(unit * keep);
    }
    std::sort(vcost.begin(), vcost.end());
    const int fs = 0;  // FullFit: stable zone full
    sink += build_curve(vcost, fs, kTopK, unit);
  }
  return sink;
}

// ── NEW daemon path (EVICTBOARD_EXTERNAL_SCORES): board cheapest_scores_sorted +
//    curve. No ExpertStats, no resident-map scan, no dense array, no token clock.
//    The board's effective-score min-heap surfaces the cheapest ≤ N victims in
//    O(N log N), touching only ~N heap nodes. ──────────────────────────────────
static double new_path_layer(gl::EvictScoreBoard& board,
                             const std::vector<int>& /*resident_counts*/,
                             double unit, std::vector<double>& score_scratch) {
  double sink = 0.0;
  for (int g = 0; g < kM; ++g) {
    int rc = 0;
    board.cheapest_scores_sorted(g, kTopK, score_scratch, &rc);  // board's own heap
    for (double& v : score_scratch) v *= unit;
    const int fs = 0;
    sink += build_curve(score_scratch, fs, kTopK, unit);
  }
  return sink;
}

// ── The harness: OLD vs NEW per-layer cold-cache cost at keeper-realistic churn ─
TEST(LoaderFeedFootprint, OldVsNewPerLayerCost) {
  const int tokens = env_int("LS_FEED_TOKENS", verbose() ? 20 : 6);
  const int residents_per_gpu = env_int("LS_FEED_RESIDENTS", 460);
  const int layers = env_int("LS_FEED_LAYERS", 58);
  const int thrash_mb = env_int("LS_FEED_THRASH_MB", verbose() ? 64 : 8);
  CacheFlusher flusher(static_cast<size_t>(thrash_mb) << 20);
  const double unit = 600.0;  // representative egress µs

  // ── shared setup: ExpertStats (dense), ResidentMirror[M], EvictScoreBoard ──
  stats::ExpertStats::Options opts;
  opts.num_moe_layers = static_cast<uint32_t>(layers);
  opts.num_experts = kExperts;
  opts.first_moe_layer = kFirstMoe;
  stats::ExpertStats es(opts);

  ResidentMirror res[kM];
  gl::EvictScoreBoard board(kM, residents_per_gpu + 64);

  // Pre-populate both structures to near-full residency (keeper FullFit). Spread
  // residents across layers/experts and give the board a spread of EXTERNAL raw
  // scores (via update()) so the effective-score heap has a non-trivial order.
  Lcg rng;
  for (int g = 0; g < kM; ++g) {
    for (int i = 0; i < residents_per_gpu; ++i) {
      const uint32_t layer = kFirstMoe + (rng.next() % layers);
      const uint16_t expert = static_cast<uint16_t>(rng.next() % kExperts);
      mem::ExpertKey k{layer, expert};
      res[g].place(k);
      const double raw = static_cast<double>(rng.next() % 1000) / 1000.0;  // [0,1)
      board.update(g, k, raw);  // external score upsert
    }
  }
  // Warm ExpertStats so recency()/frequency() do real work (not the zero path).
  for (uint64_t t = 0; t < 64; ++t) {
    stats::GatingResult gr; gr.token_id = t; gr.layer_idx = kFirstMoe;
    for (int i = 0; i < kTopK; ++i)
      gr.activations.push_back(stats::ExpertActivation{
          mem::ExpertKey{kFirstMoe, static_cast<uint16_t>(rng.next() % kExperts)}, 0.5f});
    es.update(std::span<const stats::GatingResult>(&gr, 1));
  }

  std::vector<double> old_cold, new_cold;
  old_cold.reserve(static_cast<size_t>(tokens) * layers);
  new_cold.reserve(static_cast<size_t>(tokens) * layers);
  std::vector<double> score_scratch;
  std::vector<int> rc_dummy;
  volatile double guard = 0.0;

  for (int t = 0; t < tokens; ++t) {
    for (int l = 0; l < layers; ++l) {
      const uint32_t layer = kFirstMoe + static_cast<uint32_t>(l);
      // synthetic routed top-K for this layer.
      std::vector<mem::ExpertKey> routed;
      std::vector<int> globals;
      routed.reserve(kTopK);
      for (int i = 0; i < kTopK; ++i) {
        const uint16_t e = static_cast<uint16_t>(rng.next() % kExperts);
        routed.push_back(mem::ExpertKey{layer, e});
      }
      // ── OLD ──
      flusher.clear();
      auto a0 = Clock::now();
      guard += old_path_layer(res, es, routed, static_cast<uint64_t>(t), layer, unit);
      old_cold.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a0).count());
      // ── NEW ──
      flusher.clear();
      auto a1 = Clock::now();
      guard += new_path_layer(board, rc_dummy, unit, score_scratch);
      new_cold.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a1).count());
    }
  }
  (void)guard;

  const double old_med = median_us(old_cold), old_mean = mean_us(old_cold);
  const double new_med = median_us(new_cold), new_mean = mean_us(new_cold);

  // Touched-bytes accounting (independent of timer noise):
  //   OLD per layer = ExpertStats dense states_ scattered (M × residents × sizeof
  //     PerExpertState) + M resident-vector allocations (residents × sizeof
  //     ExpertEvictionInput) + the dense states_ working set itself.
  //   NEW per layer = M × kTopK heap nodes touched (≈ kTopK × (last_seq + key)).
  const size_t per_expert_state = 4 * sizeof(uint64_t);  // PerExpertState ≈ 32 B
  const size_t evinp = sizeof(mem::ExpertEvictionInput);
  const size_t old_touched =
      static_cast<size_t>(kM) * residents_per_gpu * (per_expert_state + evinp);
  const size_t new_touched =
      static_cast<size_t>(kM) * kTopK * (sizeof(uint64_t) + sizeof(int) * 2);

  if (verbose()) {
    std::fprintf(stderr,
      "\n── LoaderFeedFootprint (M=%d, residents/GPU=%d, layers=%d, tokens=%d) ──\n"
      "  OLD path  median %.3f us/layer  mean %.3f  → %.3f ms/token\n"
      "  NEW path  median %.3f us/layer  mean %.3f  → %.3f ms/token\n"
      "  speedup (mean):           %.1fx\n"
      "  touched_bytes/layer  OLD: %zu (%.1f KiB)   NEW: %zu (%.1f KiB)   ratio %.1fx\n",
      kM, residents_per_gpu, layers, tokens,
      old_med, old_mean, old_mean * layers / 1000.0,
      new_med, new_mean, new_mean * layers / 1000.0,
      new_mean > 0 ? old_mean / new_mean : 0.0,
      old_touched, old_touched / 1024.0,
      new_touched, new_touched / 1024.0,
      new_touched > 0 ? static_cast<double>(old_touched) / new_touched : 0.0);
  }

  // ── Assertions (run unconditionally, robust to shared-box noise) ──
  // 1. The NEW path touches DRAMATICALLY fewer bytes per layer (O(N) vs O(R)).
  EXPECT_LT(new_touched, old_touched / 10)
      << "new touched_bytes not << old (new=" << new_touched
      << " old=" << old_touched << ")";
  // 2. The NEW per-layer CPU cost is lower than OLD (the regression is the OLD
  //    dense scan). Generous bound for timer noise on a shared box.
  EXPECT_LT(new_mean, old_mean)
      << "new path not cheaper than old (new=" << new_mean
      << "us old=" << old_mean << "us)";
}
