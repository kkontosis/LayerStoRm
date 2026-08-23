#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/statistics/acceptance_tracker.h"

namespace lstats = layerstorm::statistics;

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace {

lstats::AcceptanceTracker::Options default_opts() {
    return {
        .ema_alpha = 0.5,
        .per_request_ema_alpha = 0.5,
        .window_size = 4,
        .calibration_buffer_size = 8,
    };
}

/// Build a VerificationResult with no confidence samples.
lstats::VerificationResult make_result(uint64_t request_id,
                                       uint32_t accepted, uint32_t attempted,
                                       bool layer_skip = false) {
    return {
        .request_id = request_id,
        .accepted_tokens = accepted,
        .attempted_tokens = attempted,
        .used_layer_skip = layer_skip,
        .confidence_samples = {},
    };
}

/// Build a VerificationResult with confidence samples.
lstats::VerificationResult make_result_with_samples(
    uint64_t request_id, uint32_t accepted, uint32_t attempted,
    std::vector<lstats::ConfidenceSample> samples,
    bool layer_skip = false) {
    return {
        .request_id = request_id,
        .accepted_tokens = accepted,
        .attempted_tokens = attempted,
        .used_layer_skip = layer_skip,
        .confidence_samples = std::move(samples),
    };
}

}  // namespace

// ── Construction ─────────────────────────────────────────────────────────────

TEST(AcceptanceTracker, ConstructionValid) {
    auto opts = default_opts();
    lstats::AcceptanceTracker tracker(opts);
    EXPECT_DOUBLE_EQ(tracker.options().ema_alpha, 0.5);
    EXPECT_DOUBLE_EQ(tracker.options().per_request_ema_alpha, 0.5);
    EXPECT_EQ(tracker.options().window_size, 4u);
    EXPECT_EQ(tracker.options().calibration_buffer_size, 8u);
}

TEST(AcceptanceTracker, AllZerosInitially) {
    lstats::AcceptanceTracker tracker(default_opts());
    EXPECT_DOUBLE_EQ(tracker.global_rate(), 0.0);
    EXPECT_DOUBLE_EQ(tracker.windowed_rate(), 0.0);
    EXPECT_DOUBLE_EQ(tracker.cumulative_rate(), 0.0);
    EXPECT_DOUBLE_EQ(tracker.layer_skip_rate(), -1.0);
    EXPECT_EQ(tracker.total_verifications(), 0u);
    EXPECT_EQ(tracker.total_accepted_tokens(), 0u);
    EXPECT_EQ(tracker.total_attempted_tokens(), 0u);
    EXPECT_EQ(tracker.calibration_samples_count(), 0u);
}

// ── Global EMA ───────────────────────────────────────────────────────────────

TEST(AcceptanceTracker, GlobalEmaSingleUpdateSeeds) {
    lstats::AcceptanceTracker tracker(default_opts());
    auto r = make_result(1, 3, 4);  // ratio = 0.75
    std::vector<lstats::VerificationResult> batch = {r};
    tracker.update(batch);
    EXPECT_DOUBLE_EQ(tracker.global_rate(), 0.75);
}

TEST(AcceptanceTracker, GlobalEmaBlending) {
    auto opts = default_opts();
    opts.ema_alpha = 0.5;
    lstats::AcceptanceTracker tracker(opts);

    // First: seeds to 0.75
    auto r1 = make_result(1, 3, 4);
    std::vector<lstats::VerificationResult> b1 = {r1};
    tracker.update(b1);
    EXPECT_DOUBLE_EQ(tracker.global_rate(), 0.75);

    // Second: ratio = 0.25, EMA = 0.5 * 0.25 + 0.5 * 0.75 = 0.5
    auto r2 = make_result(1, 1, 4);
    std::vector<lstats::VerificationResult> b2 = {r2};
    tracker.update(b2);
    EXPECT_DOUBLE_EQ(tracker.global_rate(), 0.5);
}

TEST(AcceptanceTracker, GlobalEmaConvergesToOne) {
    lstats::AcceptanceTracker tracker(default_opts());
    for (int i = 0; i < 50; ++i) {
        auto r = make_result(1, 4, 4);  // ratio = 1.0
        std::vector<lstats::VerificationResult> batch = {r};
        tracker.update(batch);
    }
    EXPECT_NEAR(tracker.global_rate(), 1.0, 1e-9);
}

TEST(AcceptanceTracker, GlobalEmaDrivesToZero) {
    auto opts = default_opts();
    opts.ema_alpha = 0.5;
    lstats::AcceptanceTracker tracker(opts);

    // Seed with 1.0
    auto r1 = make_result(1, 4, 4);
    std::vector<lstats::VerificationResult> b1 = {r1};
    tracker.update(b1);

    // Drive towards 0.0
    for (int i = 0; i < 50; ++i) {
        auto r = make_result(1, 0, 4);
        std::vector<lstats::VerificationResult> batch = {r};
        tracker.update(batch);
    }
    EXPECT_NEAR(tracker.global_rate(), 0.0, 1e-9);
}

// ── Cumulative ───────────────────────────────────────────────────────────────

TEST(AcceptanceTracker, CumulativeCorrectRatio) {
    lstats::AcceptanceTracker tracker(default_opts());

    auto r1 = make_result(1, 3, 4);
    auto r2 = make_result(2, 2, 8);
    std::vector<lstats::VerificationResult> batch = {r1, r2};
    tracker.update(batch);

    EXPECT_EQ(tracker.total_accepted_tokens(), 5u);
    EXPECT_EQ(tracker.total_attempted_tokens(), 12u);
    EXPECT_EQ(tracker.total_verifications(), 2u);
    EXPECT_DOUBLE_EQ(tracker.cumulative_rate(), 5.0 / 12.0);
}

TEST(AcceptanceTracker, CumulativeZeroAttempted) {
    lstats::AcceptanceTracker tracker(default_opts());

    auto r = make_result(1, 0, 0);
    std::vector<lstats::VerificationResult> batch = {r};
    tracker.update(batch);

    EXPECT_EQ(tracker.total_verifications(), 1u);
    EXPECT_DOUBLE_EQ(tracker.cumulative_rate(), 0.0);
}

// ── Windowed ─────────────────────────────────────────────────────────────────

TEST(AcceptanceTracker, WindowedPartialFill) {
    auto opts = default_opts();
    opts.window_size = 4;
    lstats::AcceptanceTracker tracker(opts);

    auto r1 = make_result(1, 2, 4);  // 0.5
    auto r2 = make_result(2, 3, 4);  // 0.75
    std::vector<lstats::VerificationResult> batch = {r1, r2};
    tracker.update(batch);

    // 5 accepted / 8 attempted = 0.625
    EXPECT_DOUBLE_EQ(tracker.windowed_rate(), 5.0 / 8.0);
}

TEST(AcceptanceTracker, WindowedFullWindow) {
    auto opts = default_opts();
    opts.window_size = 4;
    lstats::AcceptanceTracker tracker(opts);

    for (uint64_t i = 1; i <= 4; ++i) {
        auto r = make_result(i, 2, 4);
        std::vector<lstats::VerificationResult> batch = {r};
        tracker.update(batch);
    }

    // 4 * 2 / 4 * 4 = 8/16 = 0.5
    EXPECT_DOUBLE_EQ(tracker.windowed_rate(), 0.5);
}

TEST(AcceptanceTracker, WindowedEviction) {
    auto opts = default_opts();
    opts.window_size = 4;
    lstats::AcceptanceTracker tracker(opts);

    // Fill window with 4 entries: ratio 0.5 each
    for (uint64_t i = 1; i <= 4; ++i) {
        auto r = make_result(i, 2, 4);
        std::vector<lstats::VerificationResult> batch = {r};
        tracker.update(batch);
    }
    EXPECT_DOUBLE_EQ(tracker.windowed_rate(), 0.5);

    // Add 5th entry: ratio 1.0 — evicts first (2/4)
    auto r5 = make_result(5, 4, 4);
    std::vector<lstats::VerificationResult> batch5 = {r5};
    tracker.update(batch5);

    // Window: (2/4, 2/4, 2/4, 4/4) = 10/16 = 0.625
    EXPECT_DOUBLE_EQ(tracker.windowed_rate(), 10.0 / 16.0);
}

TEST(AcceptanceTracker, WindowedNumericCorrectness) {
    auto opts = default_opts();
    opts.window_size = 3;
    lstats::AcceptanceTracker tracker(opts);

    // Insert 3 entries: 1/2, 3/4, 5/8
    auto r1 = make_result(1, 1, 2);
    auto r2 = make_result(2, 3, 4);
    auto r3 = make_result(3, 5, 8);
    std::vector<lstats::VerificationResult> batch = {r1, r2, r3};
    tracker.update(batch);

    // 9/14
    EXPECT_DOUBLE_EQ(tracker.windowed_rate(), 9.0 / 14.0);

    // Evict 1/2, insert 2/2
    auto r4 = make_result(4, 2, 2);
    std::vector<lstats::VerificationResult> batch2 = {r4};
    tracker.update(batch2);

    // Window: (3/4, 5/8, 2/2) = 10/14
    EXPECT_DOUBLE_EQ(tracker.windowed_rate(), 10.0 / 14.0);
}

TEST(AcceptanceTracker, WindowedEmptyReturnsZero) {
    lstats::AcceptanceTracker tracker(default_opts());
    EXPECT_DOUBLE_EQ(tracker.windowed_rate(), 0.0);
}

// ── Per-request ──────────────────────────────────────────────────────────────

TEST(AcceptanceTracker, PerRequestUnknownReturnsZero) {
    lstats::AcceptanceTracker tracker(default_opts());
    EXPECT_DOUBLE_EQ(tracker.rate(42), 0.0);
}

TEST(AcceptanceTracker, PerRequestSingleUpdate) {
    lstats::AcceptanceTracker tracker(default_opts());

    auto r = make_result(42, 3, 4);  // ratio = 0.75
    std::vector<lstats::VerificationResult> batch = {r};
    tracker.update(batch);

    EXPECT_DOUBLE_EQ(tracker.rate(42), 0.75);
}

TEST(AcceptanceTracker, PerRequestEmaBlending) {
    auto opts = default_opts();
    opts.per_request_ema_alpha = 0.5;
    lstats::AcceptanceTracker tracker(opts);

    // First: seeds to 0.75
    auto r1 = make_result(42, 3, 4);
    std::vector<lstats::VerificationResult> b1 = {r1};
    tracker.update(b1);
    EXPECT_DOUBLE_EQ(tracker.rate(42), 0.75);

    // Second: ratio = 0.25, EMA = 0.5 * 0.25 + 0.5 * 0.75 = 0.5
    auto r2 = make_result(42, 1, 4);
    std::vector<lstats::VerificationResult> b2 = {r2};
    tracker.update(b2);
    EXPECT_DOUBLE_EQ(tracker.rate(42), 0.5);
}

TEST(AcceptanceTracker, PerRequestMultipleIndependent) {
    lstats::AcceptanceTracker tracker(default_opts());

    auto r1 = make_result(1, 1, 4);  // 0.25
    auto r2 = make_result(2, 3, 4);  // 0.75
    std::vector<lstats::VerificationResult> batch = {r1, r2};
    tracker.update(batch);

    EXPECT_DOUBLE_EQ(tracker.rate(1), 0.25);
    EXPECT_DOUBLE_EQ(tracker.rate(2), 0.75);
}

TEST(AcceptanceTracker, PerRequestRemoveCleanup) {
    lstats::AcceptanceTracker tracker(default_opts());

    auto r = make_result(42, 3, 4);
    std::vector<lstats::VerificationResult> batch = {r};
    tracker.update(batch);
    EXPECT_DOUBLE_EQ(tracker.rate(42), 0.75);

    tracker.remove_request(42);
    EXPECT_DOUBLE_EQ(tracker.rate(42), 0.0);
}

// ── Layer-skip ───────────────────────────────────────────────────────────────

TEST(AcceptanceTracker, LayerSkipNoDataReturnsNegOne) {
    lstats::AcceptanceTracker tracker(default_opts());
    EXPECT_DOUBLE_EQ(tracker.layer_skip_rate(), -1.0);
}

TEST(AcceptanceTracker, LayerSkipTrackedCorrectly) {
    lstats::AcceptanceTracker tracker(default_opts());

    auto r = make_result(1, 3, 4, /*layer_skip=*/true);
    std::vector<lstats::VerificationResult> batch = {r};
    tracker.update(batch);

    EXPECT_DOUBLE_EQ(tracker.layer_skip_rate(), 0.75);
}

TEST(AcceptanceTracker, LayerSkipIgnoresNonSkipResults) {
    lstats::AcceptanceTracker tracker(default_opts());

    // Non-skip result should not affect layer_skip_rate.
    auto r1 = make_result(1, 0, 4, /*layer_skip=*/false);
    std::vector<lstats::VerificationResult> b1 = {r1};
    tracker.update(b1);
    EXPECT_DOUBLE_EQ(tracker.layer_skip_rate(), -1.0);

    // Now a layer-skip result.
    auto r2 = make_result(2, 3, 4, /*layer_skip=*/true);
    std::vector<lstats::VerificationResult> b2 = {r2};
    tracker.update(b2);
    EXPECT_DOUBLE_EQ(tracker.layer_skip_rate(), 0.75);
}

TEST(AcceptanceTracker, LayerSkipEmaBlending) {
    auto opts = default_opts();
    opts.ema_alpha = 0.5;
    lstats::AcceptanceTracker tracker(opts);

    // Seed: ratio = 1.0
    auto r1 = make_result(1, 4, 4, true);
    std::vector<lstats::VerificationResult> b1 = {r1};
    tracker.update(b1);
    EXPECT_DOUBLE_EQ(tracker.layer_skip_rate(), 1.0);

    // Blend: ratio = 0.0, EMA = 0.5 * 0 + 0.5 * 1.0 = 0.5
    auto r2 = make_result(2, 0, 4, true);
    std::vector<lstats::VerificationResult> b2 = {r2};
    tracker.update(b2);
    EXPECT_DOUBLE_EQ(tracker.layer_skip_rate(), 0.5);
}

// ── Calibration ──────────────────────────────────────────────────────────────

TEST(AcceptanceTracker, CalibrationAccumulation) {
    lstats::AcceptanceTracker tracker(default_opts());

    std::vector<lstats::ConfidenceSample> samples = {
        {0.9f, true}, {0.3f, false}, {0.7f, true},
    };
    auto r = make_result_with_samples(1, 2, 3, samples);
    std::vector<lstats::VerificationResult> batch = {r};
    tracker.update(batch);

    EXPECT_EQ(tracker.calibration_samples_count(), 3u);
    EXPECT_FLOAT_EQ(tracker.calibration_sample(0).raw_confidence, 0.9f);
    EXPECT_TRUE(tracker.calibration_sample(0).was_accepted);
    EXPECT_FLOAT_EQ(tracker.calibration_sample(1).raw_confidence, 0.3f);
    EXPECT_FALSE(tracker.calibration_sample(1).was_accepted);
    EXPECT_FLOAT_EQ(tracker.calibration_sample(2).raw_confidence, 0.7f);
    EXPECT_TRUE(tracker.calibration_sample(2).was_accepted);
}

TEST(AcceptanceTracker, CalibrationRingWrap) {
    auto opts = default_opts();
    opts.calibration_buffer_size = 4;
    lstats::AcceptanceTracker tracker(opts);

    // Insert 6 samples into buffer of size 4 — oldest 2 should be evicted.
    for (int i = 0; i < 6; ++i) {
        std::vector<lstats::ConfidenceSample> samples = {
            {static_cast<float>(i) * 0.1f, i % 2 == 0},
        };
        auto r = make_result_with_samples(static_cast<uint64_t>(i), 1, 1, samples);
        std::vector<lstats::VerificationResult> batch = {r};
        tracker.update(batch);
    }

    EXPECT_EQ(tracker.calibration_samples_count(), 4u);

    // Oldest surviving = index 2 (confidence 0.2), newest = index 5 (confidence 0.5)
    EXPECT_FLOAT_EQ(tracker.calibration_sample(0).raw_confidence, 0.2f);
    EXPECT_FLOAT_EQ(tracker.calibration_sample(3).raw_confidence, 0.5f);
}

TEST(AcceptanceTracker, CalibrationDrainClears) {
    lstats::AcceptanceTracker tracker(default_opts());

    std::vector<lstats::ConfidenceSample> samples = {{0.5f, true}};
    auto r = make_result_with_samples(1, 1, 1, samples);
    std::vector<lstats::VerificationResult> batch = {r};
    tracker.update(batch);

    EXPECT_EQ(tracker.calibration_samples_count(), 1u);
    tracker.drain_calibration_data();
    EXPECT_EQ(tracker.calibration_samples_count(), 0u);
}

TEST(AcceptanceTracker, CalibrationIndexedAccessCorrect) {
    auto opts = default_opts();
    opts.calibration_buffer_size = 8;
    lstats::AcceptanceTracker tracker(opts);

    // Insert 5 samples: 0.1, 0.2, 0.3, 0.4, 0.5
    std::vector<lstats::ConfidenceSample> samples;
    for (int i = 1; i <= 5; ++i) {
        samples.push_back({static_cast<float>(i) * 0.1f, true});
    }
    auto r = make_result_with_samples(1, 5, 5, samples);
    std::vector<lstats::VerificationResult> batch = {r};
    tracker.update(batch);

    EXPECT_EQ(tracker.calibration_samples_count(), 5u);
    for (uint32_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(tracker.calibration_sample(i).raw_confidence,
                         static_cast<float>(i + 1) * 0.1f);
    }
}

// ── Edge cases ───────────────────────────────────────────────────────────────

TEST(AcceptanceTracker, EmptySpanNoOp) {
    lstats::AcceptanceTracker tracker(default_opts());
    std::vector<lstats::VerificationResult> empty;
    tracker.update(empty);

    EXPECT_DOUBLE_EQ(tracker.global_rate(), 0.0);
    EXPECT_EQ(tracker.total_verifications(), 0u);
}

TEST(AcceptanceTracker, ZeroAttemptedTokens) {
    lstats::AcceptanceTracker tracker(default_opts());

    auto r = make_result(1, 0, 0);
    std::vector<lstats::VerificationResult> batch = {r};
    tracker.update(batch);

    // ratio = 0.0, seeds global EMA to 0.0
    EXPECT_DOUBLE_EQ(tracker.global_rate(), 0.0);
    EXPECT_EQ(tracker.total_verifications(), 1u);
    EXPECT_DOUBLE_EQ(tracker.cumulative_rate(), 0.0);
}
