#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/statistics/workload_detector.h"

namespace lmem = layerstorm::memory;
namespace lstats = layerstorm::statistics;

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace {

// Window size 4, threshold 3 sigma.
constexpr int kWindowSize = 4;
// Enough steady-state windows for EWMA to converge and Welford variance to
// settle.  With alpha=0.3 the EWMA converges by window ~5; 30 windows gives
// ~25 near-zero distance samples that dominate the variance.
constexpr int kSteadyWindows = 30;

lstats::WorkloadDetector::Options default_opts() {
    return {
        .num_moe_layers = 4,
        .num_experts = 8,
        .first_moe_layer = 3,
        .token_window_size = kWindowSize,
        .shift_threshold_std_devs = 3.0,
    };
}

lstats::ExpertStats::Options default_stats_opts() {
    return {
        .ewma_alpha = 0.3,
        .num_moe_layers = 4,
        .num_experts = 8,
        .first_moe_layer = 3,
    };
}

/// Build a GatingResult for a single token at one layer.
lstats::GatingResult make_result(
    uint64_t token_id, uint32_t layer_idx,
    std::vector<std::pair<uint16_t, float>> experts) {
    lstats::GatingResult r;
    r.token_id = token_id;
    r.layer_idx = layer_idx;
    for (auto [eid, w] : experts) {
        r.activations.push_back({.key = {layer_idx, eid}, .routing_weight = w});
    }
    return r;
}

/// Activate a fixed set of experts across all MoE layers for one token.
void activate_token(lstats::ExpertStats& stats, uint64_t token_id,
                    const std::vector<std::pair<uint16_t, float>>& experts,
                    uint32_t first_moe_layer, uint32_t num_moe_layers) {
    std::vector<lstats::GatingResult> results;
    for (uint32_t l = 0; l < num_moe_layers; ++l) {
        results.push_back(make_result(token_id, first_moe_layer + l, experts));
    }
    stats.update(results);
}

/// Run a steady workload for N full windows, activating the same experts.
void run_steady_workload(lstats::ExpertStats& stats,
                         lstats::WorkloadDetector& detector,
                         uint64_t start_token, int num_windows,
                         const std::vector<std::pair<uint16_t, float>>& experts,
                         uint32_t first_moe_layer, uint32_t num_moe_layers,
                         int window_size) {
    for (int w = 0; w < num_windows; ++w) {
        for (int t = 0; t < window_size; ++t) {
            uint64_t tid = start_token +
                           static_cast<uint64_t>(w) * window_size + t;
            activate_token(stats, tid, experts, first_moe_layer, num_moe_layers);
        }
        detector.update(stats);
    }
}

/// Token ID after running N windows starting at start_token.
uint64_t after_windows(uint64_t start_token, int num_windows) {
    return start_token + static_cast<uint64_t>(num_windows) * kWindowSize;
}

}  // namespace

// ── Tests ────────────────────────────────────────────────────────────────────

TEST(WorkloadDetector, ConstructionValid) {
    auto opts = default_opts();
    lstats::WorkloadDetector detector(opts);
    EXPECT_EQ(detector.options().num_moe_layers, 4);
    EXPECT_EQ(detector.options().num_experts, 8);
    EXPECT_EQ(detector.options().first_moe_layer, 3);
    EXPECT_EQ(detector.options().token_window_size, kWindowSize);
    EXPECT_DOUBLE_EQ(detector.options().shift_threshold_std_devs, 3.0);
    EXPECT_EQ(detector.windows_processed(), 0u);
}

TEST(WorkloadDetector, NoShiftInitially) {
    lstats::WorkloadDetector detector(default_opts());
    EXPECT_FALSE(detector.shift_detected());
}

TEST(WorkloadDetector, NoActionBeforeWindowBoundary) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    // Process fewer tokens than one window.
    for (uint64_t t = 1; t <= 3; ++t) {
        activate_token(stats, t, {{0, 0.5f}, {1, 0.3f}}, 3, 4);
    }
    detector.update(stats);
    EXPECT_EQ(detector.windows_processed(), 0u);
}

TEST(WorkloadDetector, WindowBoundaryAdvances) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    // Process exactly one window worth of tokens.
    for (uint64_t t = 1; t <= 4; ++t) {
        activate_token(stats, t, {{0, 0.5f}, {1, 0.3f}}, 3, 4);
    }
    detector.update(stats);
    EXPECT_EQ(detector.windows_processed(), 1u);

    // Process second window.
    for (uint64_t t = 5; t <= 8; ++t) {
        activate_token(stats, t, {{0, 0.5f}, {1, 0.3f}}, 3, 4);
    }
    detector.update(stats);
    EXPECT_EQ(detector.windows_processed(), 2u);
}

TEST(WorkloadDetector, SteadyWorkloadNoShift) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    run_steady_workload(stats, detector, 1, kSteadyWindows,
                        {{0, 0.5f}, {1, 0.3f}}, 3, 4, kWindowSize);

    EXPECT_FALSE(detector.shift_detected());
    EXPECT_GE(detector.windows_processed(),
              static_cast<uint64_t>(kSteadyWindows));
}

TEST(WorkloadDetector, AbruptShiftDetected) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    run_steady_workload(stats, detector, 1, kSteadyWindows,
                        {{0, 0.9f}, {1, 0.9f}}, 3, 4, kWindowSize);
    EXPECT_FALSE(detector.shift_detected());

    uint64_t tid = after_windows(1, kSteadyWindows);
    run_steady_workload(stats, detector, tid, 1,
                        {{6, 0.9f}, {7, 0.9f}}, 3, 4, kWindowSize);

    EXPECT_TRUE(detector.shift_detected());
}

TEST(WorkloadDetector, ShiftDetectedOneShot) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    run_steady_workload(stats, detector, 1, kSteadyWindows,
                        {{0, 0.9f}, {1, 0.9f}}, 3, 4, kWindowSize);

    uint64_t tid = after_windows(1, kSteadyWindows);
    run_steady_workload(stats, detector, tid, 1,
                        {{6, 0.9f}, {7, 0.9f}}, 3, 4, kWindowSize);

    // First call should return true.
    EXPECT_TRUE(detector.shift_detected());
    // Second call should return false (INV-4.14c).
    EXPECT_FALSE(detector.shift_detected());
}

TEST(WorkloadDetector, GradualDriftNoShift) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    // Start with 3 experts already active so the drift doesn't introduce a
    // discontinuity from zero→non-zero frequency.
    run_steady_workload(stats, detector, 1, kSteadyWindows,
                        {{0, 0.5f}, {1, 0.5f}, {2, 0.1f}}, 3, 4, kWindowSize);

    uint64_t tid = after_windows(1, kSteadyWindows);

    // Gradually shift weights among already-active experts.
    // Each step changes by ~0.01 — well within normal variance.
    for (int w = 0; w < 20; ++w) {
        float w2 = 0.1f + 0.01f * (w + 1);  // 0.11 → 0.30
        for (int t = 0; t < kWindowSize; ++t) {
            activate_token(stats, tid++,
                           {{0, 0.5f}, {1, 0.5f}, {2, w2}}, 3, 4);
        }
        detector.update(stats);
    }

    EXPECT_FALSE(detector.shift_detected());
}

TEST(WorkloadDetector, NoDetectionBeforeMinHistory) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    // Window 1: experts {0,1}
    run_steady_workload(stats, detector, 1, 1,
                        {{0, 0.9f}, {1, 0.9f}}, 3, 4, kWindowSize);

    // Window 2-3: dramatically different experts, but < kMinHistorySamples.
    uint64_t tid = after_windows(1, 1);
    for (int w = 0; w < 2; ++w) {
        for (int t = 0; t < kWindowSize; ++t) {
            activate_token(stats, tid++,
                           {{6, 0.9f}, {7, 0.9f}}, 3, 4);
        }
        detector.update(stats);
    }

    // Not enough history yet — should not detect.
    EXPECT_FALSE(detector.shift_detected());
}

TEST(WorkloadDetector, DetectionAfterMinHistory) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    run_steady_workload(stats, detector, 1, kSteadyWindows,
                        {{0, 0.9f}, {1, 0.9f}}, 3, 4, kWindowSize);

    EXPECT_GE(detector.windows_processed(),
              static_cast<uint64_t>(kSteadyWindows));
    EXPECT_FALSE(detector.shift_detected());

    uint64_t tid = after_windows(1, kSteadyWindows);
    run_steady_workload(stats, detector, tid, 1,
                        {{6, 0.9f}, {7, 0.9f}}, 3, 4, kWindowSize);

    EXPECT_TRUE(detector.shift_detected());
}

TEST(WorkloadDetector, CooldownPreventsRepeatTrigger) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    run_steady_workload(stats, detector, 1, kSteadyWindows,
                        {{0, 0.9f}, {1, 0.9f}}, 3, 4, kWindowSize);

    uint64_t tid = after_windows(1, kSteadyWindows);

    // Trigger first shift.
    run_steady_workload(stats, detector, tid, 1,
                        {{6, 0.9f}, {7, 0.9f}}, 3, 4, kWindowSize);
    EXPECT_TRUE(detector.shift_detected());
    tid += kWindowSize;

    // Another dramatic shift during cooldown (within 4 windows).
    run_steady_workload(stats, detector, tid, 1,
                        {{0, 0.9f}, {1, 0.9f}}, 3, 4, kWindowSize);
    // Should NOT trigger during cooldown (INV-4.14d).
    EXPECT_FALSE(detector.shift_detected());
}

TEST(WorkloadDetector, ShiftAfterCooldown) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    run_steady_workload(stats, detector, 1, kSteadyWindows,
                        {{0, 0.9f}, {1, 0.9f}}, 3, 4, kWindowSize);

    uint64_t tid = after_windows(1, kSteadyWindows);

    // Trigger first shift.
    run_steady_workload(stats, detector, tid, 1,
                        {{6, 0.9f}, {7, 0.9f}}, 3, 4, kWindowSize);
    EXPECT_TRUE(detector.shift_detected());
    tid += kWindowSize;

    // Wait through cooldown (4 windows) with steady experts.
    run_steady_workload(stats, detector, tid, 5,
                        {{6, 0.9f}, {7, 0.9f}}, 3, 4, kWindowSize);
    tid += 5 * kWindowSize;

    // New shift after cooldown.
    run_steady_workload(stats, detector, tid, 1,
                        {{0, 0.9f}, {1, 0.9f}}, 3, 4, kWindowSize);
    EXPECT_TRUE(detector.shift_detected());
}

TEST(WorkloadDetector, EmptyUpdatesNoEffect) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    // Update with zero tokens processed — should not crash.
    detector.update(stats);
    EXPECT_EQ(detector.windows_processed(), 0u);
    EXPECT_FALSE(detector.shift_detected());
}

TEST(WorkloadDetector, AllZeroFrequencies) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    // Feed tokens with empty activations (no experts activated).
    for (int w = 0; w < 8; ++w) {
        for (int t = 0; t < kWindowSize; ++t) {
            uint64_t tid = 1 + static_cast<uint64_t>(w) * kWindowSize + t;
            std::vector<lstats::GatingResult> results;
            for (uint32_t l = 0; l < 4; ++l) {
                results.push_back(make_result(tid, 3 + l, {}));
            }
            stats.update(results);
        }
        detector.update(stats);
    }

    EXPECT_DOUBLE_EQ(detector.last_distance(), 0.0);
    EXPECT_FALSE(detector.shift_detected());
}

TEST(WorkloadDetector, SingleExpertShift) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    run_steady_workload(stats, detector, 1, kSteadyWindows,
                        {{0, 0.9f}}, 3, 4, kWindowSize);

    uint64_t tid = after_windows(1, kSteadyWindows);
    run_steady_workload(stats, detector, tid, 1,
                        {{7, 0.9f}}, 3, 4, kWindowSize);

    EXPECT_TRUE(detector.shift_detected());
}

TEST(WorkloadDetector, MultipleLayerShift) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    run_steady_workload(stats, detector, 1, kSteadyWindows,
                        {{0, 0.9f}, {1, 0.9f}}, 3, 4, kWindowSize);

    uint64_t tid = after_windows(1, kSteadyWindows);
    run_steady_workload(stats, detector, tid, 1,
                        {{5, 0.9f}, {6, 0.9f}, {7, 0.9f}}, 3, 4, kWindowSize);

    EXPECT_TRUE(detector.shift_detected());
}

TEST(WorkloadDetector, DistanceMeanAndStdDev) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    run_steady_workload(stats, detector, 1, kSteadyWindows,
                        {{0, 0.5f}, {1, 0.3f}}, 3, 4, kWindowSize);

    EXPECT_GE(detector.distance_mean(), 0.0);
    EXPECT_GE(detector.distance_std_dev(), 0.0);
    EXPECT_TRUE(std::isfinite(detector.distance_mean()));
    EXPECT_TRUE(std::isfinite(detector.distance_std_dev()));
}

TEST(WorkloadDetector, LastDistanceTracked) {
    auto opts = default_opts();
    lstats::ExpertStats stats(default_stats_opts());
    lstats::WorkloadDetector detector(opts);

    EXPECT_DOUBLE_EQ(detector.last_distance(), 0.0);

    // First window: snapshot only.
    run_steady_workload(stats, detector, 1, 1,
                        {{0, 0.5f}, {1, 0.3f}}, 3, 4, kWindowSize);

    // Second window: distance computed.
    run_steady_workload(stats, detector, 1 + kWindowSize, 1,
                        {{0, 0.5f}, {1, 0.3f}}, 3, 4, kWindowSize);

    EXPECT_GE(detector.last_distance(), 0.0);
    EXPECT_TRUE(std::isfinite(detector.last_distance()));
}
