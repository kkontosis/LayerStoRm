// Cold-cache microbenchmarks for the two I8 loader hot-path functions
// (route_moe_by_loader / feed_expert_stats), measuring ONLY the work each does:
//   - feed_expert_stats  → GatingResult build + ExpertStats::update()  [BenchFeed]
//   - route_moe_by_loader→ LoaderSolver::solve() (= shadow_solve_and_log core) [BenchRoute]
// Both functions are CommandDispatcher members buried under heavy deps; their cost
// is entirely the standalone work above, so we drive that work directly.
//
// Each timed call is preceded by a FULL cache clear (capacity-thrash a >LLC buffer
// + clflush sweep + fence) so we measure the realistic COLD-cache cost the daemon
// pays — its caches are polluted by transfer/poll work between calls — not a warm
// micro-loop. A warm pass is reported alongside for contrast.
//
// Env-gated (loops + per-call cache flush are slow); run on demand:
//   LS_RUN_HOTPATH_BENCH=1 ./layerstorm_unit_tests \
//       --gtest_filter='LoaderHotpathBench.*'
// Knobs: LS_BENCH_TOKENS (default 40), LS_BENCH_THRASH_MB (default 96).
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "core/gpu_loader/loader_constants.h"
#include "core/gpu_loader/loader_evict_scores.h"
#include "core/gpu_loader/loader_place_cons.h"
#include "core/gpu_loader/loader_solver.h"
#include "core/memory/eviction_policy.h"
#include "core/statistics/expert_stats.h"

namespace {

namespace gl = layerstorm::gpu_loader;
namespace stats = layerstorm::statistics;
namespace mem = layerstorm::memory;
using Clock = std::chrono::steady_clock;

bool bench_enabled() { return std::getenv("LS_RUN_HOTPATH_BENCH") != nullptr; }
int env_int(const char* k, int def) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::atoi(v) : def;
}

// ── Cache clear "in all possible ways" we can from userspace ────────────────
// (1) capacity-thrash a buffer larger than the LLC (write+read every line), which
// evicts ALL prior cached data by capacity; (2) clflush every line of that buffer
// so nothing of it lingers either; (3) a full fence. Anonymous RAM (states_/solver
// scratch) is evicted by (1). drop_caches is N/A here — these structures are
// anonymous memory, never page-cache-backed.
struct CacheFlusher {
    std::vector<char> thrash;
    explicit CacheFlusher(size_t bytes) : thrash(bytes, 1) {}
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
    double s = 0; for (double x : v) s += x; return v.empty() ? 0 : s / v.size();
}

// Deterministic LCG so the routed top-K is reproducible run to run.
struct Lcg { uint64_t s = 0x9e3779b97f4a7c15ULL;
    uint32_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t)(s >> 33); } };

void report(const char* name, std::vector<double>& cold, std::vector<double>& warm,
            int calls_per_token) {
    double cmed = median_us(cold), cmean = mean_us(cold);
    double wmed = median_us(warm), wmean = mean_us(warm);
    std::fprintf(stderr,
        "\n── %s (%zu timed calls; %d calls/token) ──\n"
        "  COLD  median %.3f us/call  mean %.3f us/call  → %.4f ms/token  (%.3f ms/100-tok)\n"
        "  WARM  median %.3f us/call  mean %.3f us/call  → %.4f ms/token\n"
        "  cold/warm ratio (median): %.1fx\n",
        name, cold.size(), calls_per_token,
        cmed, cmean, cmean * calls_per_token / 1000.0, cmean * calls_per_token / 1000.0 * 100.0,
        wmed, wmean, wmean * calls_per_token / 1000.0,
        wmed > 0 ? cmed / wmed : 0.0);
}

}  // namespace

// ── feed_expert_stats work: GatingResult build + ExpertStats::update() ──────
TEST(LoaderHotpathBench, FeedExpertStats) {
    if (!bench_enabled()) GTEST_SKIP() << "set LS_RUN_HOTPATH_BENCH=1";
    const int tokens = env_int("LS_BENCH_TOKENS", 40);
    CacheFlusher flusher(static_cast<size_t>(env_int("LS_BENCH_THRASH_MB", 96)) << 20);

    stats::ExpertStats::Options opts;  // 58 MoE layers, 256 experts, first=3
    stats::ExpertStats es(opts);
    const uint32_t L0 = opts.first_moe_layer, nL = opts.num_moe_layers;
    const int topk = 8;
    Lcg rng;

    auto one_call = [&](uint64_t token_id, uint32_t layer) {
        // EXACT work of feed_expert_stats: build a 1-element GatingResult with the
        // routed top-K, then update().
        stats::GatingResult gr;
        gr.token_id = token_id;
        gr.layer_idx = layer;
        gr.activations.reserve(topk);
        for (int i = 0; i < topk; ++i) {
            mem::ExpertKey key{layer, static_cast<uint16_t>(rng.next() % opts.num_experts)};
            gr.activations.push_back(stats::ExpertActivation{key, 0.5f});
        }
        es.update(std::span<const stats::GatingResult>(&gr, 1));
    };

    std::vector<double> cold, warm;
    cold.reserve(static_cast<size_t>(tokens) * nL);
    warm.reserve(static_cast<size_t>(tokens) * nL);
    for (int t = 0; t < tokens; ++t)
        for (uint32_t l = 0; l < nL; ++l) {
            flusher.clear();                       // COLD: evict caches first
            auto a = Clock::now();
            one_call(static_cast<uint64_t>(t), L0 + l);
            cold.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
        }
    for (int t = 0; t < tokens; ++t)               // WARM: no flush, caches hot
        for (uint32_t l = 0; l < nL; ++l) {
            auto a = Clock::now();
            one_call(static_cast<uint64_t>(tokens + t), L0 + l);
            warm.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
        }
    report("feed_expert_stats work (GatingResult build + ExpertStats::update)",
           cold, warm, static_cast<int>(nL));
}

// ── route_moe_by_loader work: LoaderSolver::solve() (shadow_solve core) ─────
TEST(LoaderHotpathBench, RouteLoaderSolve) {
    if (!bench_enabled()) GTEST_SKIP() << "set LS_RUN_HOTPATH_BENCH=1";
    const int tokens = env_int("LS_BENCH_TOKENS", 40);
    CacheFlusher flusher(static_cast<size_t>(env_int("LS_BENCH_THRASH_MB", 96)) << 20);

    // Realistic keeper shape: 2 devices, 4 banks, calibrated compute curve.
    const int M = 2, B = 4, N = 8;
    gl::LoaderConstants k;
    k.num_devices = M;
    k.num_banks = B;
    for (int d = 0; d < M; ++d) {
        gl::DeviceConstants dc;
        dc.position = d;
        dc.compute = {16.6, 64.0, 64};  // calibrated a/b/P
        k.devices.push_back(dc);
    }
    for (int b = 0; b < B; ++b) {
        gl::BankConstants bc;
        bc.node = b;
        bc.egress_us = 440.0 + 100.0 * b;
        bc.contention = 1.0;
        k.banks.push_back(bc);
    }
    k.matrix.assign(B, std::vector<gl::TransferCell>(M));
    for (int b = 0; b < B; ++b)
        for (int d = 0; d < M; ++d)
            k.matrix[b][d] = gl::TransferCell{440.0 + 30.0 * ((b + d) % 4), (b == d) ? 1 : 2, 2.0};

    gl::SolveRequest req;
    req.num_devices = M;
    req.num_experts = N;
    req.bank_of.assign(N, 0);
    for (int i = 0; i < N; ++i) req.bank_of[i] = i % B;
    req.cached.assign(static_cast<size_t>(N) * M, 0);
    for (int i = 0; i < N; ++i) req.cached[static_cast<size_t>(i) * M + (i % M)] = 1;  // ~half cached

    gl::LoaderSolver solver;
    std::vector<double> cold, warm;
    const int iters = tokens * 58;  // match the feed's per-token call budget
    cold.reserve(iters); warm.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        flusher.clear();
        auto a = Clock::now();
        solver.solve(k, req);
        cold.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
    }
    for (int i = 0; i < iters; ++i) {
        auto a = Clock::now();
        solver.solve(k, req);
        warm.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
    }
    report("route_moe_by_loader work (LoaderSolver::solve)", cold, warm, 58);
}

// ── O(N)-footprint proof for the new stats-locality path ────────────────────
// The whole point of EvictScoreBoard + PlaceConsTable is that the daemon hot-
// path feed/read touches ONLY O(N) contiguous bytes (N = routed top-K), so its
// cold-cache cost does NOT scale with the total expert count X — unlike the old
// recency feed, which scatter-writes/recomputes over the dense X-wide array.
//
// This test sweeps X (experts/layer) and:
//   (1) measures the NEW per-solve hot-path footprint work:
//         place_table.gather(N routed)  +  N × evict_board.cheapest_scores_sorted
//       at small X and large X, asserting the cold-cache cost is FLAT in X.
//   (2) measures a CONTROL that scans the full X-wide dense array per call,
//       to show that footprint DOES scale with X (the regression class).
// The flatness assertion runs unconditionally (cheap); the verbose timing
// report is printed when LS_RUN_HOTPATH_BENCH=1.
namespace {

// One hot-path stats touch for the routed top-K on M GPUs, EXACTLY mirroring the
// loader-on daemon path after EVICTBOARD_EXTERNAL_SCORES: gather place rows (N×M
// contiguous) + per-GPU cheapest-N effective scores for the convex evict_cum
// curve (the board's own effective-score heap — NO eviction_inputs scan, NO
// ExpertStats). Returns a sink to defeat dead-code elimination.
double new_path_touch(gl::PlaceConsTable& place, gl::EvictScoreBoard& board,
                      const std::vector<int>& globals, int M,
                      std::vector<double>& place_scratch,
                      std::vector<double>& score_scratch) {
  place.gather(globals, place_scratch);
  double sink = 0.0;
  for (double v : place_scratch) sink += v;
  const int N = static_cast<int>(globals.size());
  for (int g = 0; g < M; ++g) {
    int rc = 0;
    board.cheapest_scores_sorted(g, N, score_scratch, &rc);  // the real hot-path read
    for (double v : score_scratch) sink += v;
    sink += board.cheapest_score(g);
  }
  return sink;
}

// CONTROL: the old class of work — scan the entire X-wide dense array (the
// ~475 KB states_-style footprint). Cost scales with X.
double control_dense_scan(const std::vector<double>& dense) {
  double sink = 0.0;
  for (double v : dense) sink += v;  // touches every X*M byte
  return sink;
}

// median cold-cache µs over `iters` of `fn`, flushing the cache before each.
template <typename Fn>
double cold_median_us(CacheFlusher& flusher, int iters, Fn&& fn) {
  std::vector<double> t;
  t.reserve(iters);
  volatile double guard = 0.0;
  for (int i = 0; i < iters; ++i) {
    flusher.clear();
    auto a = Clock::now();
    guard += fn();
    t.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
  }
  (void)guard;
  return median_us(t);
}

void build_board_and_place(int X, int M, int N, gl::PlaceConsTable& place,
                           gl::EvictScoreBoard& board, std::vector<int>& globals) {
  Lcg rng;
  // Fill the place table (so the gathered rows have real data to read).
  for (int e = 0; e < X; ++e)
    for (int j = 0; j < M; ++j)
      place.set(e, j, static_cast<double>((rng.next() % 1000)) * 0.001);
  // Populate the board with many residents across GPUs (X total) with a spread of
  // EXTERNAL raw scores; the effective-score heap is realistic. Only the
  // cheapest-N-per-GPU are read on the hot path (board's own state).
  for (int e = 0; e < X; ++e) {
    int g = e % M;
    layerstorm::memory::ExpertKey k{3, static_cast<uint16_t>(e % 65535)};
    board.update(g, k, static_cast<double>((rng.next() % 1000)) * 0.001);
  }
  // The routed top-K globals (N experts).
  globals.clear();
  for (int i = 0; i < N; ++i) globals.push_back(static_cast<int>(rng.next() % X));
}

}  // namespace

TEST(LoaderHotpathBench, EvictPlaceLocalityIsONotX) {
  const int M = 2, N = 8;
  const int X_small = 256;    // ~DeepSeek per-layer experts
  const int X_big   = 256 * 58 * 4;  // ~59k flattened experts (58 layers × 4)
  const bool verbose = bench_enabled();
  // Non-verbose (default CI) path: a light flush + few iters keeps the assertion
  // fast (the flatness gap is ~100×, robust to a small thrash). Verbose runs use
  // the full LS_BENCH_THRASH_MB / LS_BENCH_TOKENS knobs for a publishable number.
  const int iters = verbose ? env_int("LS_BENCH_TOKENS", 40) * 8 : 16;
  const int thrash_mb = verbose ? env_int("LS_BENCH_THRASH_MB", 96) : 8;
  CacheFlusher flusher(static_cast<size_t>(thrash_mb) << 20);

  // ── small X ──
  gl::PlaceConsTable place_s(X_small, M);
  gl::EvictScoreBoard board_s(M, X_small);
  std::vector<int> globals_s;
  build_board_and_place(X_small, M, N, place_s, board_s, globals_s);
  std::vector<double> scratch_s, keep_s;
  double new_small = cold_median_us(flusher, iters, [&] {
    return new_path_touch(place_s, board_s, globals_s, M, scratch_s, keep_s);
  });

  // ── big X ──
  gl::PlaceConsTable place_b(X_big, M);
  gl::EvictScoreBoard board_b(M, X_big);
  std::vector<int> globals_b;
  build_board_and_place(X_big, M, N, place_b, board_b, globals_b);
  std::vector<double> scratch_b, keep_b;
  double new_big = cold_median_us(flusher, iters, [&] {
    return new_path_touch(place_b, board_b, globals_b, M, scratch_b, keep_b);
  });

  // ── control: dense X-wide scan, small vs big ──
  std::vector<double> dense_s(static_cast<size_t>(X_small) * M, 0.5);
  std::vector<double> dense_b(static_cast<size_t>(X_big) * M, 0.5);
  double ctl_small = cold_median_us(flusher, iters, [&] { return control_dense_scan(dense_s); });
  double ctl_big   = cold_median_us(flusher, iters, [&] { return control_dense_scan(dense_b); });

  if (verbose) {
    std::fprintf(stderr,
        "\n── EvictPlaceLocality footprint (N=%d, M=%d) ──\n"
        "  NEW path  gather+recent : X=%-6d %.3f us   X=%-6d %.3f us  (ratio %.2fx)\n"
        "  CONTROL   dense X-scan  : X=%-6d %.3f us   X=%-6d %.3f us  (ratio %.2fx)\n"
        "  touched_bytes(NEW) = O(N*M + N) independent of X; CONTROL = O(X*M)\n",
        N, M,
        X_small, new_small, X_big, new_big, new_small > 0 ? new_big / new_small : 0.0,
        X_small, ctl_small, X_big, ctl_big, ctl_small > 0 ? ctl_big / ctl_small : 0.0);
  }

  // The NEW path's cold-cache cost must NOT scale with X (O(N), N fixed): big/small
  // ratio stays small (≈1, generous bound for timer noise on a shared box).
  EXPECT_LT(new_big, new_small * 4.0 + 5.0)
      << "new-path footprint scaled with X (new_small=" << new_small
      << " new_big=" << new_big << ")";
  // The CONTROL, which scans the dense X-wide array, MUST scale ~linearly with X
  // (X_big/X_small = 232×) — this is the regression class the new path avoids.
  // Use a conservative lower bound (≥4×) so the test is robust to noise/prefetch.
  EXPECT_GT(ctl_big, ctl_small * 4.0)
      << "control did not scale with X — bench environment unreliable (ctl_small="
      << ctl_small << " ctl_big=" << ctl_big << ")";
}
