#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/statistics/expert_stats.h"

namespace lmem = layerstorm::memory;
namespace lstats = layerstorm::statistics;

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace {

lstats::ExpertStats::Options default_opts() {
    return {
        .ewma_alpha = 0.01,
        .num_moe_layers = 4,
        .num_experts = 8,
        .first_moe_layer = 3,
        .max_recency_tokens = 100,
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

/// Activate one expert at one layer for N consecutive tokens.
void activate_expert_n_times(
    lstats::ExpertStats& stats, uint32_t layer, uint16_t expert,
    float weight, uint64_t start_token, uint64_t count) {
    for (uint64_t t = 0; t < count; ++t) {
        auto r = make_result(start_token + t, layer, {{expert, weight}});
        std::vector<lstats::GatingResult> v = {std::move(r)};
        stats.update(v);
    }
}

/// Advance the token counter by N tokens without any activations.
void advance_empty_tokens(
    lstats::ExpertStats& stats, uint64_t start_token, uint64_t count) {
    for (uint64_t t = 0; t < count; ++t) {
        auto r = make_result(start_token + t, 3, {});  // empty activations
        std::vector<lstats::GatingResult> v = {std::move(r)};
        stats.update(v);
    }
}

}  // namespace

// ── Construction ─────────────────────────────────────────────────────────────

TEST(ExpertStats, ConstructionValid) {
    lstats::ExpertStats stats(default_opts());
    EXPECT_EQ(stats.total_tokens_processed(), 0u);
    EXPECT_DOUBLE_EQ(stats.options().ewma_alpha, 0.01);
}

TEST(ExpertStats, AllZerosInitially) {
    lstats::ExpertStats stats(default_opts());
    lmem::ExpertKey key{3, 0};
    EXPECT_DOUBLE_EQ(stats.frequency(key), 0.0);
    EXPECT_DOUBLE_EQ(stats.recency(key), 0.0);  // no tokens processed
    EXPECT_DOUBLE_EQ(stats.routing_weight(key), 0.0);
    EXPECT_DOUBLE_EQ(stats.temporal_autocorr(key), 0.0);
}

// ── Frequency tracking ──────────────────────────────────────────────────────

TEST(ExpertStats, SingleActivationIncreasesFrequency) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 1);

    // Only one expert activated → it is the max → frequency = 1.0
    EXPECT_DOUBLE_EQ(stats.frequency({3, 0}), 1.0);
}

TEST(ExpertStats, RepeatedActivationIncreasesFrequency) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 50);

    // After 50 consecutive activations, frequency should be high.
    double freq = stats.frequency({3, 0});
    EXPECT_GT(freq, 0.5);
    EXPECT_LE(freq, 1.0);
}

TEST(ExpertStats, InactiveExpertFrequencyDecays) {
    lstats::ExpertStats stats(default_opts());

    // Activate expert 0 for 10 tokens.
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 10);
    double freq_after_active = stats.raw_frequency({3, 0});

    // Now activate a DIFFERENT expert for 50 tokens.
    activate_expert_n_times(stats, 3, 1, 0.5f, 11, 50);
    double freq_after_inactive = stats.raw_frequency({3, 0});

    EXPECT_LT(freq_after_inactive, freq_after_active);
}

TEST(ExpertStats, FrequencyNormalized01) {
    lstats::ExpertStats stats(default_opts());

    // Activate two experts with different frequencies.
    for (uint64_t t = 1; t <= 100; ++t) {
        auto r = make_result(t, 3, {{0, 0.5f}});
        std::vector<lstats::GatingResult> v = {std::move(r)};
        stats.update(v);

        if (t % 5 == 0) {
            auto r2 = make_result(t, 4, {{1, 0.3f}});  // different layer
            std::vector<lstats::GatingResult> v2 = {std::move(r2)};
            stats.update(v2);
        }
    }

    double f0 = stats.frequency({3, 0});
    double f1 = stats.frequency({4, 1});
    EXPECT_GE(f0, 0.0);
    EXPECT_LE(f0, 1.0);
    EXPECT_GE(f1, 0.0);
    EXPECT_LE(f1, 1.0);
    EXPECT_GT(f0, f1);  // Expert 0 activated more often.
}

TEST(ExpertStats, LazyDecayCorrectness) {
    // Compare lazy-decayed value against brute-force EWMA.
    const double alpha = 0.05;
    auto opts = default_opts();
    opts.ewma_alpha = alpha;
    lstats::ExpertStats stats(opts);

    // Activate expert 0 at tokens 1,2,3, then nothing for 10, then at token 14.
    activate_expert_n_times(stats, 3, 0, 1.0f, 1, 3);
    advance_empty_tokens(stats, 4, 10);
    activate_expert_n_times(stats, 3, 0, 1.0f, 14, 1);

    // Brute-force compute expected EWMA.
    double expected = 0.0;
    for (uint64_t t = 1; t <= 14; ++t) {
        bool activated = (t <= 3) || (t == 14);
        expected = (1.0 - alpha) * expected + (activated ? alpha : 0.0);
    }

    double actual = stats.raw_frequency({3, 0});
    EXPECT_NEAR(actual, expected, 1e-12);
}

// ── Recency tracking ─────────────────────────────────────────────────────────

TEST(ExpertStats, JustUsedExpertRecencyZero) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 1);
    EXPECT_DOUBLE_EQ(stats.recency({3, 0}), 0.0);
}

TEST(ExpertStats, OldExpertRecencyHigher) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 1);

    // Advance 50 tokens without activating expert 0.
    activate_expert_n_times(stats, 3, 1, 0.5f, 2, 50);

    double rec = stats.recency({3, 0});
    EXPECT_GT(rec, 0.0);
    EXPECT_DOUBLE_EQ(rec, 50.0 / 100.0);  // max_recency_tokens = 100
}

TEST(ExpertStats, RecencyClampedToOne) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 1);

    // Advance beyond max_recency_tokens.
    activate_expert_n_times(stats, 3, 1, 0.5f, 2, 200);

    EXPECT_DOUBLE_EQ(stats.recency({3, 0}), 1.0);
}

TEST(ExpertStats, NeverUsedExpertMaxRecency) {
    lstats::ExpertStats stats(default_opts());
    // Activate some other expert to advance the counter.
    activate_expert_n_times(stats, 3, 1, 0.5f, 1, 200);

    // Expert 0 was never used → recency = 1.0 (clamped).
    EXPECT_DOUBLE_EQ(stats.recency({3, 0}), 1.0);
}

// ── Routing weight tracking ──────────────────────────────────────────────────

TEST(ExpertStats, RoutingWeightUpdatedOnActivation) {
    lstats::ExpertStats stats(default_opts());

    // Activate with weight 0.8 repeatedly.
    activate_expert_n_times(stats, 3, 0, 0.8f, 1, 50);

    double rw = stats.routing_weight({3, 0});
    EXPECT_GT(rw, 0.0);
    EXPECT_LE(rw, 1.0);
}

TEST(ExpertStats, HighWeightExpertNormalizesToOne) {
    lstats::ExpertStats stats(default_opts());

    // Activate expert 0 with weight 0.9, expert 1 with weight 0.1.
    for (uint64_t t = 1; t <= 50; ++t) {
        auto r = make_result(t, 3, {{0, 0.9f}, {1, 0.1f}});
        std::vector<lstats::GatingResult> v = {std::move(r)};
        stats.update(v);
    }

    // Expert 0 has the highest routing weight → should normalize to 1.0.
    EXPECT_DOUBLE_EQ(stats.routing_weight({3, 0}), 1.0);
    EXPECT_LT(stats.routing_weight({3, 1}), 1.0);
    EXPECT_GT(stats.routing_weight({3, 1}), 0.0);
}

TEST(ExpertStats, RoutingWeightNormalized01) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 10);

    double rw = stats.routing_weight({3, 0});
    EXPECT_GE(rw, 0.0);
    EXPECT_LE(rw, 1.0);
}

// ── Temporal autocorrelation ─────────────────────────────────────────────────

TEST(ExpertStats, ActivatedLastTokenAutocorrOne) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 1);
    // Just activated at the current token → autocorr = 1.0.
    EXPECT_DOUBLE_EQ(stats.temporal_autocorr({3, 0}), 1.0);
}

TEST(ExpertStats, ActivatedTwoTokensAgoAutocorrZero) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 1);

    // Advance 2 tokens without activating expert 0.
    activate_expert_n_times(stats, 3, 1, 0.5f, 2, 2);

    // Expert 0 last used at token 1, current token = 3. Distance = 2 → 0.0.
    EXPECT_DOUBLE_EQ(stats.temporal_autocorr({3, 0}), 0.0);
}

TEST(ExpertStats, ConsecutiveTokensAutocorrOne) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 1);
    activate_expert_n_times(stats, 3, 0, 0.5f, 2, 1);

    // Used at token 1 and 2, current = 2. Distance = 0 → 1.0.
    EXPECT_DOUBLE_EQ(stats.temporal_autocorr({3, 0}), 1.0);
}

TEST(ExpertStats, PreviousTokenAutocorrOne) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 1);

    // Advance one token without expert 0.
    activate_expert_n_times(stats, 3, 1, 0.5f, 2, 1);

    // Expert 0 last used at token 1, current = 2. Distance = 1 → 1.0.
    EXPECT_DOUBLE_EQ(stats.temporal_autocorr({3, 0}), 1.0);
}

TEST(ExpertStats, NotActivatedAutocorrZero) {
    lstats::ExpertStats stats(default_opts());
    // Advance a few tokens without ever activating expert 0.
    activate_expert_n_times(stats, 3, 1, 0.5f, 1, 5);
    EXPECT_DOUBLE_EQ(stats.temporal_autocorr({3, 0}), 0.0);
}

// ── fill_eviction_scores ─────────────────────────────────────────────────────

TEST(ExpertStats, FillsAllFourTerms) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.7f, 1, 10);

    // Advance a few tokens so recency > 0.
    activate_expert_n_times(stats, 3, 1, 0.5f, 11, 5);

    lmem::ExpertEvictionInput input;
    input.key = {3, 0};
    input.zone = lmem::CacheZone::kStable;
    input.is_duplicate = true;
    input.gpu_idx = 2;
    input.coactivation = 0.42;
    input.prefetch_score = 0.77;

    std::vector<lmem::ExpertEvictionInput> inputs = {input};
    stats.fill_eviction_scores(inputs);

    // Scoring terms should be set.
    EXPECT_GT(inputs[0].frequency, 0.0);
    EXPECT_GT(inputs[0].recency, 0.0);
    EXPECT_GT(inputs[0].routing_weight, 0.0);
    // temporal_autocorr: expert 0 last used at token 10, current=15, dist=5 → 0.0
    EXPECT_DOUBLE_EQ(inputs[0].temporal_autocorr, 0.0);

    // Other fields must be untouched.
    EXPECT_EQ(inputs[0].key.layer_idx, 3u);
    EXPECT_EQ(inputs[0].key.expert_idx, 0u);
    EXPECT_EQ(inputs[0].zone, lmem::CacheZone::kStable);
    EXPECT_TRUE(inputs[0].is_duplicate);
    EXPECT_EQ(inputs[0].gpu_idx, 2);
    EXPECT_DOUBLE_EQ(inputs[0].coactivation, 0.42);
    EXPECT_DOUBLE_EQ(inputs[0].prefetch_score, 0.77);
}

TEST(ExpertStats, FillEmptySpanNoOp) {
    lstats::ExpertStats stats(default_opts());
    std::span<lmem::ExpertEvictionInput> empty;
    stats.fill_eviction_scores(empty);  // Should not crash.
}

// ── frequency_vector ─────────────────────────────────────────────────────────

TEST(ExpertStats, FrequencyVectorCorrectSize) {
    lstats::ExpertStats stats(default_opts());
    auto fv = stats.frequency_vector(3);  // first MoE layer
    EXPECT_EQ(fv.size(), 8u);  // num_experts = 8
}

TEST(ExpertStats, FrequencyVectorReflectsActivations) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 2, 0.5f, 1, 10);

    auto fv = stats.frequency_vector(3);
    EXPECT_GT(fv[2], 0.0);  // Expert 2 was activated.
    EXPECT_DOUBLE_EQ(fv[0], 0.0);  // Expert 0 was not.
}

TEST(ExpertStats, FrequencyVectorMatchesIndividualQueries) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 4, 3, 0.5f, 1, 20);
    activate_expert_n_times(stats, 4, 5, 0.3f, 21, 10);

    auto fv = stats.frequency_vector(4);
    EXPECT_NEAR(fv[3], stats.raw_frequency({4, 3}), 1e-12);
    EXPECT_NEAR(fv[5], stats.raw_frequency({4, 5}), 1e-12);
}

TEST(ExpertStats, FrequencyVectorOutOfRangeEmpty) {
    lstats::ExpertStats stats(default_opts());
    auto fv = stats.frequency_vector(0);  // Below first_moe_layer.
    EXPECT_TRUE(fv.empty());
    auto fv2 = stats.frequency_vector(100);  // Beyond last MoE layer.
    EXPECT_TRUE(fv2.empty());
}

// ── frequency_percentile ─────────────────────────────────────────────────────

TEST(ExpertStats, SingleExpertHighPercentile) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 50);

    // All other experts have frequency 0. Expert 0's count_le includes
    // only itself among non-zero → percentile near 100.
    double pct = stats.frequency_percentile({3, 0});
    EXPECT_GT(pct, 99.0);
}

TEST(ExpertStats, NeverActivatedLowPercentile) {
    lstats::ExpertStats stats(default_opts());
    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 50);

    // Never-activated expert has frequency 0 → tied with many others.
    double pct = stats.frequency_percentile({3, 7});
    EXPECT_LT(pct, 100.0);
}

// ── Multi-token update ───────────────────────────────────────────────────────

TEST(ExpertStats, BatchUpdateProcessesAllTokens) {
    lstats::ExpertStats stats(default_opts());

    // Build 5 gating results for 5 different tokens.
    std::vector<lstats::GatingResult> batch;
    for (uint64_t t = 1; t <= 5; ++t) {
        batch.push_back(make_result(t, 3, {{0, 0.5f}}));
    }
    stats.update(batch);

    EXPECT_EQ(stats.total_tokens_processed(), 5u);
    EXPECT_GT(stats.raw_frequency({3, 0}), 0.0);
}

TEST(ExpertStats, MultipleLayersSameToken) {
    lstats::ExpertStats stats(default_opts());

    // Two layers for the same token.
    std::vector<lstats::GatingResult> batch = {
        make_result(1, 3, {{0, 0.5f}}),
        make_result(1, 4, {{2, 0.3f}}),
    };
    stats.update(batch);

    // Only one token should be counted.
    EXPECT_EQ(stats.total_tokens_processed(), 1u);

    // Both experts should be tracked.
    EXPECT_GT(stats.raw_frequency({3, 0}), 0.0);
    EXPECT_GT(stats.raw_frequency({4, 2}), 0.0);
}

TEST(ExpertStats, VariableExpertCountPerToken) {
    lstats::ExpertStats stats(default_opts());

    // Token 1: 1 expert. Token 2: 4 experts. Token 3: 0 experts.
    std::vector<lstats::GatingResult> batch = {
        make_result(1, 3, {{0, 0.5f}}),
        make_result(2, 3, {{0, 0.2f}, {1, 0.3f}, {2, 0.25f}, {3, 0.25f}}),
        make_result(3, 3, {}),
    };
    stats.update(batch);

    EXPECT_EQ(stats.total_tokens_processed(), 3u);
}

// ── Edge cases ───────────────────────────────────────────────────────────────

TEST(ExpertStats, AlphaZeroNoLearning) {
    auto opts = default_opts();
    opts.ewma_alpha = 0.0;
    lstats::ExpertStats stats(opts);

    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 10);

    // With alpha=0, EWMA never updates from 0.
    EXPECT_DOUBLE_EQ(stats.raw_frequency({3, 0}), 0.0);
}

TEST(ExpertStats, AlphaOneInstantLearning) {
    auto opts = default_opts();
    opts.ewma_alpha = 1.0;
    lstats::ExpertStats stats(opts);

    // Token 1: activate.
    activate_expert_n_times(stats, 3, 0, 0.8f, 1, 1);

    // With alpha=1, frequency = alpha on activation.
    // (1-alpha)^gap = 0^gap = 0 for gap > 0, so only the current activation matters.
    EXPECT_DOUBLE_EQ(stats.raw_frequency({3, 0}), 1.0);

    // Token 2: don't activate → frequency decays to 0 (decay_factor = 0^1 = 0).
    activate_expert_n_times(stats, 3, 1, 0.5f, 2, 1);
    EXPECT_DOUBLE_EQ(stats.raw_frequency({3, 0}), 0.0);
}

TEST(ExpertStats, EmptyCompletedNoOp) {
    lstats::ExpertStats stats(default_opts());
    std::vector<lstats::GatingResult> empty;
    stats.update(empty);
    EXPECT_EQ(stats.total_tokens_processed(), 0u);
}

TEST(ExpertStats, LargeGapLazyDecay) {
    auto opts = default_opts();
    opts.ewma_alpha = 0.01;
    lstats::ExpertStats stats(opts);

    activate_expert_n_times(stats, 3, 0, 0.5f, 1, 1);
    double freq_initial = stats.raw_frequency({3, 0});

    // Advance 10,000 tokens without activating expert 0.
    activate_expert_n_times(stats, 3, 1, 0.5f, 2, 10000);

    double freq_after = stats.raw_frequency({3, 0});
    EXPECT_LT(freq_after, freq_initial * 0.001);  // Heavily decayed.
}

TEST(ExpertStats, InvalidKeyReturnsZero) {
    lstats::ExpertStats stats(default_opts());
    // Layer 0 is below first_moe_layer (3).
    EXPECT_DOUBLE_EQ(stats.frequency({0, 0}), 0.0);
    EXPECT_DOUBLE_EQ(stats.recency({0, 0}), 0.0);
    EXPECT_DOUBLE_EQ(stats.routing_weight({0, 0}), 0.0);
    EXPECT_DOUBLE_EQ(stats.temporal_autocorr({0, 0}), 0.0);
    EXPECT_DOUBLE_EQ(stats.raw_frequency({0, 0}), 0.0);

    // Expert index out of range.
    EXPECT_DOUBLE_EQ(stats.frequency({3, 100}), 0.0);
}

TEST(ExpertStats, MaxExpertsPerToken) {
    lstats::ExpertStats stats(default_opts());

    // Activate all 8 experts in one token.
    std::vector<std::pair<uint16_t, float>> all_experts;
    for (uint16_t e = 0; e < 8; ++e) {
        all_experts.push_back({e, 1.0f / 8.0f});
    }
    auto r = make_result(1, 3, all_experts);
    std::vector<lstats::GatingResult> v = {std::move(r)};
    stats.update(v);

    // All experts should have equal frequency → all normalize to 1.0.
    for (uint16_t e = 0; e < 8; ++e) {
        EXPECT_DOUBLE_EQ(stats.frequency({3, e}), 1.0);
    }
}
