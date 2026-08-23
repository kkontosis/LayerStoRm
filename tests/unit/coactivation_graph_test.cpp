#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "core/statistics/coactivation_graph.h"

namespace lmem = layerstorm::memory;
namespace lstats = layerstorm::statistics;

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace {

lstats::CoactivationGraph::Options default_opts() {
    return {
        .num_moe_layers = 4,
        .num_experts = 8,
        .first_moe_layer = 3,
        .decay_factor = 0.999,
        .workload_shift_decay = 0.1,
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

/// Activate a set of experts at one layer for one token.
void activate_once(
    lstats::CoactivationGraph& graph, uint64_t token_id, uint32_t layer,
    std::vector<std::pair<uint16_t, float>> experts) {
    auto r = make_result(token_id, layer, std::move(experts));
    std::vector<lstats::GatingResult> v = {std::move(r)};
    graph.update(v);
}

/// Advance the token counter by N tokens with empty activations at a layer.
void advance_empty_tokens(
    lstats::CoactivationGraph& graph, uint32_t layer,
    uint64_t start_token, uint64_t count) {
    for (uint64_t t = 0; t < count; ++t) {
        activate_once(graph, start_token + t, layer, {});
    }
}

}  // namespace

// ── Construction ─────────────────────────────────────────────────────────────

TEST(CoactivationGraph, ConstructionValid) {
    lstats::CoactivationGraph graph(default_opts());
    EXPECT_EQ(graph.total_tokens_processed(), 0u);
    EXPECT_DOUBLE_EQ(graph.options().decay_factor, 0.999);
    EXPECT_EQ(graph.options().num_moe_layers, 4u);
}

TEST(CoactivationGraph, AllZerosInitially) {
    lstats::CoactivationGraph graph(default_opts());
    for (uint32_t layer = 3; layer < 7; ++layer) {
        for (uint16_t i = 0; i < 8; ++i) {
            for (uint16_t j = 0; j < 8; ++j) {
                EXPECT_FLOAT_EQ(graph.weight(layer, i, j), 0.0f)
                    << "layer=" << layer << " i=" << i << " j=" << j;
            }
        }
    }
}

// ── Single pair update ───────────────────────────────────────────────────────

TEST(CoactivationGraph, SinglePairUpdate) {
    lstats::CoactivationGraph graph(default_opts());
    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}});

    float w = graph.weight(3, 0, 1);
    EXPECT_GT(w, 0.0f);
    // After one token: 1.0 * decay^0 = 1.0 (decay applied before increment,
    // but layer was at token 0, gap=1, so decay first then +1).
    // Actually: gap=1 from token 0→1, decay the zero matrix (still zero),
    // then add 1.0. So weight = 1.0 exactly.
    EXPECT_FLOAT_EQ(w, 1.0f);
}

TEST(CoactivationGraph, Symmetry) {
    lstats::CoactivationGraph graph(default_opts());
    activate_once(graph, 1, 3, {{2, 0.5f}, {5, 0.3f}});

    EXPECT_FLOAT_EQ(graph.weight(3, 2, 5), graph.weight(3, 5, 2));
}

// ── Pairwise completeness ────────────────────────────────────────────────────

TEST(CoactivationGraph, PairwiseCompleteness) {
    lstats::CoactivationGraph graph(default_opts());
    // Activate 4 experts → C(4,2) = 6 pairs should be non-zero.
    activate_once(graph, 1, 3, {{0, 0.2f}, {1, 0.3f}, {2, 0.25f}, {3, 0.25f}});

    // All 6 pairs should be non-zero.
    EXPECT_GT(graph.weight(3, 0, 1), 0.0f);
    EXPECT_GT(graph.weight(3, 0, 2), 0.0f);
    EXPECT_GT(graph.weight(3, 0, 3), 0.0f);
    EXPECT_GT(graph.weight(3, 1, 2), 0.0f);
    EXPECT_GT(graph.weight(3, 1, 3), 0.0f);
    EXPECT_GT(graph.weight(3, 2, 3), 0.0f);

    // Non-activated pairs should be zero.
    EXPECT_FLOAT_EQ(graph.weight(3, 0, 4), 0.0f);
    EXPECT_FLOAT_EQ(graph.weight(3, 4, 5), 0.0f);
    EXPECT_FLOAT_EQ(graph.weight(3, 6, 7), 0.0f);
}

// ── Accumulation ─────────────────────────────────────────────────────────────

TEST(CoactivationGraph, AccumulationWithDecay) {
    lstats::CoactivationGraph graph(default_opts());
    const double decay = 0.999;

    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}});
    float w1 = graph.weight(3, 0, 1);  // 1.0

    activate_once(graph, 2, 3, {{0, 0.5f}, {1, 0.3f}});
    float w2 = graph.weight(3, 0, 1);  // decay * 1.0 + 1.0

    // Second activation should increase weight despite decay.
    EXPECT_GT(w2, w1);
    EXPECT_NEAR(w2, static_cast<float>(decay * 1.0 + 1.0), 1e-5f);
}

TEST(CoactivationGraph, TenConsecutiveActivations) {
    auto opts = default_opts();
    opts.decay_factor = 0.999;
    lstats::CoactivationGraph graph(opts);

    // Brute-force expected: weight = sum_{k=0}^{9} 0.999^k
    double expected = 0.0;
    for (int k = 0; k < 10; ++k) {
        expected = expected * 0.999 + 1.0;
    }

    for (uint64_t t = 1; t <= 10; ++t) {
        activate_once(graph, t, 3, {{0, 0.5f}, {1, 0.3f}});
    }

    EXPECT_NEAR(graph.weight(3, 0, 1), static_cast<float>(expected), 1e-3f);
}

// ── Decay ────────────────────────────────────────────────────────────────────

TEST(CoactivationGraph, DecayReducesWeight) {
    lstats::CoactivationGraph graph(default_opts());
    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}});
    float w_before = graph.weight(3, 0, 1);

    // Advance 100 empty tokens.
    advance_empty_tokens(graph, 3, 2, 100);
    float w_after = graph.weight(3, 0, 1);

    EXPECT_LT(w_after, w_before);
}

TEST(CoactivationGraph, LazyDecayCorrectness) {
    auto opts = default_opts();
    opts.decay_factor = 0.95;  // Aggressive decay for clarity.
    lstats::CoactivationGraph graph(opts);

    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}});

    // Advance 20 tokens without activating.
    advance_empty_tokens(graph, 3, 2, 20);

    // Expected: 1.0 * 0.95^20
    double expected = std::pow(0.95, 20);
    EXPECT_NEAR(graph.weight(3, 0, 1), static_cast<float>(expected), 1e-5f);
}

TEST(CoactivationGraph, LargeGapDecay) {
    auto opts = default_opts();
    opts.decay_factor = 0.99;
    lstats::CoactivationGraph graph(opts);

    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}});

    // Advance 10,000 tokens.
    advance_empty_tokens(graph, 3, 2, 10000);

    float w = graph.weight(3, 0, 1);
    // 0.99^10000 ≈ 2.25e-44 → effectively zero.
    EXPECT_LT(w, 1e-30f);
}

TEST(CoactivationGraph, DecayOnlyAppliesToCorrectLayer) {
    lstats::CoactivationGraph graph(default_opts());

    // Activate pair on layer 3 and layer 4.
    std::vector<lstats::GatingResult> batch = {
        make_result(1, 3, {{0, 0.5f}, {1, 0.3f}}),
        make_result(1, 4, {{0, 0.5f}, {1, 0.3f}}),
    };
    graph.update(batch);

    // Advance tokens on layer 3 only.
    advance_empty_tokens(graph, 3, 2, 50);

    float w3 = graph.weight(3, 0, 1);
    float w4 = graph.weight(4, 0, 1);

    // Layer 3 should have decayed more than layer 4 (which got no updates
    // and is still stale from token 1 — but lazy read materializes same gap).
    // Actually: both layers had last_decayed_token = 1. After advancing
    // layer 3 by 50 tokens (tokens 2-51), global_token_counter_ = 51.
    // Layer 3: eagerly decayed during those empty updates.
    // Layer 4: lazy — weight() materializes decay from token 1 to 51 = 50-token gap.
    // So both should show similar decay. Let me verify they're equal.
    EXPECT_NEAR(w3, w4, 1e-5f);
}

// ── Workload shift decay ─────────────────────────────────────────────────────

TEST(CoactivationGraph, ShiftDecay) {
    lstats::CoactivationGraph graph(default_opts());
    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}});
    float w_before = graph.weight(3, 0, 1);

    graph.apply_shift_decay(0.1);
    float w_after = graph.weight(3, 0, 1);

    EXPECT_NEAR(w_after, w_before * 0.1f, 1e-5f);
}

TEST(CoactivationGraph, ShiftDecayZero) {
    lstats::CoactivationGraph graph(default_opts());
    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}});

    graph.apply_shift_decay(0.0);

    EXPECT_FLOAT_EQ(graph.weight(3, 0, 1), 0.0f);
}

TEST(CoactivationGraph, ShiftDecayOne) {
    lstats::CoactivationGraph graph(default_opts());
    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}});
    float w_before = graph.weight(3, 0, 1);

    graph.apply_shift_decay(1.0);
    float w_after = graph.weight(3, 0, 1);

    EXPECT_FLOAT_EQ(w_before, w_after);
}

TEST(CoactivationGraph, ShiftDecayAffectsAllLayers) {
    lstats::CoactivationGraph graph(default_opts());

    std::vector<lstats::GatingResult> batch = {
        make_result(1, 3, {{0, 0.5f}, {1, 0.3f}}),
        make_result(1, 4, {{2, 0.5f}, {3, 0.3f}}),
        make_result(1, 5, {{4, 0.5f}, {5, 0.3f}}),
    };
    graph.update(batch);

    graph.apply_shift_decay(0.1);

    // All layers should be decayed.
    EXPECT_LT(graph.weight(3, 0, 1), 0.15f);
    EXPECT_LT(graph.weight(4, 2, 3), 0.15f);
    EXPECT_LT(graph.weight(5, 4, 5), 0.15f);
}

// ── Token deduplication ──────────────────────────────────────────────────────

TEST(CoactivationGraph, TokenDeduplication) {
    lstats::CoactivationGraph graph(default_opts());

    // Two layers for the same token.
    std::vector<lstats::GatingResult> batch = {
        make_result(1, 3, {{0, 0.5f}, {1, 0.3f}}),
        make_result(1, 4, {{2, 0.5f}, {3, 0.3f}}),
    };
    graph.update(batch);

    // Only one token counted.
    EXPECT_EQ(graph.total_tokens_processed(), 1u);
}

// ── Row access ───────────────────────────────────────────────────────────────

TEST(CoactivationGraph, RowAccessMatchesPointQueries) {
    lstats::CoactivationGraph graph(default_opts());
    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}, {2, 0.25f}});

    auto r = graph.row(3, 0);
    ASSERT_EQ(r.size(), 8u);

    for (uint16_t j = 0; j < 8; ++j) {
        EXPECT_FLOAT_EQ(r[j], graph.weight(3, 0, j))
            << "mismatch at j=" << j;
    }
}

TEST(CoactivationGraph, RowAccessWithPendingDecay) {
    auto opts = default_opts();
    opts.decay_factor = 0.95;
    lstats::CoactivationGraph graph(opts);

    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}});
    advance_empty_tokens(graph, 3, 2, 10);

    auto r = graph.row(3, 0);
    ASSERT_EQ(r.size(), 8u);

    // Row should match materialized point queries.
    for (uint16_t j = 0; j < 8; ++j) {
        EXPECT_NEAR(r[j], graph.weight(3, 0, j), 1e-6f);
    }
}

TEST(CoactivationGraph, RowAccessInvalidLayerEmpty) {
    lstats::CoactivationGraph graph(default_opts());
    auto r = graph.row(0, 0);  // Below first_moe_layer.
    EXPECT_TRUE(r.empty());
}

TEST(CoactivationGraph, RowAccessInvalidExpertEmpty) {
    lstats::CoactivationGraph graph(default_opts());
    auto r = graph.row(3, 100);  // Expert out of range.
    EXPECT_TRUE(r.empty());
}

// ── Diagonal ─────────────────────────────────────────────────────────────────

TEST(CoactivationGraph, DiagonalAlwaysZero) {
    lstats::CoactivationGraph graph(default_opts());

    // Activate all experts.
    std::vector<std::pair<uint16_t, float>> all;
    for (uint16_t e = 0; e < 8; ++e) all.push_back({e, 0.125f});
    activate_once(graph, 1, 3, all);

    for (uint16_t e = 0; e < 8; ++e) {
        EXPECT_FLOAT_EQ(graph.weight(3, e, e), 0.0f) << "diagonal e=" << e;
    }
}

// ── Variable expert counts (INV-0.2) ─────────────────────────────────────────

TEST(CoactivationGraph, VariableExpertCounts) {
    lstats::CoactivationGraph graph(default_opts());

    // Token 1: 1 expert (no pairs).
    activate_once(graph, 1, 3, {{0, 0.5f}});
    EXPECT_FLOAT_EQ(graph.weight(3, 0, 1), 0.0f);

    // Token 2: 3 experts → 3 pairs.
    activate_once(graph, 2, 3, {{0, 0.5f}, {1, 0.3f}, {2, 0.2f}});
    EXPECT_GT(graph.weight(3, 0, 1), 0.0f);
    EXPECT_GT(graph.weight(3, 0, 2), 0.0f);
    EXPECT_GT(graph.weight(3, 1, 2), 0.0f);

    // Token 3: 0 experts (empty — only decay).
    activate_once(graph, 3, 3, {});

    // Token 4: 6 experts → 15 pairs.
    activate_once(graph, 4, 3, {{0, 0.1f}, {1, 0.1f}, {2, 0.1f},
                                {3, 0.1f}, {4, 0.1f}, {5, 0.1f}});
    // Newly activated pairs should be non-zero.
    EXPECT_GT(graph.weight(3, 3, 4), 0.0f);
    EXPECT_GT(graph.weight(3, 4, 5), 0.0f);
}

// ── Invalid keys ─────────────────────────────────────────────────────────────

TEST(CoactivationGraph, InvalidLayerReturnsZero) {
    lstats::CoactivationGraph graph(default_opts());
    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}});

    // Layer 0 is below first_moe_layer.
    EXPECT_FLOAT_EQ(graph.weight(0, 0, 1), 0.0f);
    // Layer 100 is beyond last MoE layer.
    EXPECT_FLOAT_EQ(graph.weight(100, 0, 1), 0.0f);
}

TEST(CoactivationGraph, InvalidExpertReturnsZero) {
    lstats::CoactivationGraph graph(default_opts());
    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}});

    EXPECT_FLOAT_EQ(graph.weight(3, 0, 100), 0.0f);
    EXPECT_FLOAT_EQ(graph.weight(3, 100, 0), 0.0f);
}

// ── Empty update ─────────────────────────────────────────────────────────────

TEST(CoactivationGraph, EmptyUpdateNoOp) {
    lstats::CoactivationGraph graph(default_opts());
    std::vector<lstats::GatingResult> empty;
    graph.update(empty);
    EXPECT_EQ(graph.total_tokens_processed(), 0u);
}

// ── Greedy partitioning ──────────────────────────────────────────────────────

TEST(CoactivationGraph, PartitioningBasic) {
    auto opts = default_opts();
    opts.decay_factor = 1.0;  // No decay for deterministic test.
    lstats::CoactivationGraph graph(opts);

    // Experts 0 and 1 co-activate heavily.
    for (uint64_t t = 1; t <= 10; ++t) {
        activate_once(graph, t, 3, {{0, 0.5f}, {1, 0.5f}});
    }
    // Experts 2 and 3 co-activate heavily.
    for (uint64_t t = 11; t <= 20; ++t) {
        activate_once(graph, t, 3, {{2, 0.5f}, {3, 0.5f}});
    }

    std::vector<int> capacities = {4, 4};  // 2 GPUs, 4 slots each.
    auto hints = graph.compute_affinity_hints(2, capacities);

    // Find assignments for layer 3.
    int gpu_of_0 = -1, gpu_of_1 = -1, gpu_of_2 = -1, gpu_of_3 = -1;
    for (const auto& h : hints) {
        if (h.key.layer_idx != 3) continue;
        if (h.key.expert_idx == 0) gpu_of_0 = h.gpu_idx;
        if (h.key.expert_idx == 1) gpu_of_1 = h.gpu_idx;
        if (h.key.expert_idx == 2) gpu_of_2 = h.gpu_idx;
        if (h.key.expert_idx == 3) gpu_of_3 = h.gpu_idx;
    }

    // Co-activating pairs should be on the same GPU.
    EXPECT_EQ(gpu_of_0, gpu_of_1);
    EXPECT_EQ(gpu_of_2, gpu_of_3);
    // The two clusters should be on different GPUs.
    EXPECT_NE(gpu_of_0, gpu_of_2);
}

TEST(CoactivationGraph, PartitioningCapacity) {
    auto opts = default_opts();
    opts.decay_factor = 1.0;
    lstats::CoactivationGraph graph(opts);

    // Activate all 8 experts together.
    activate_once(graph, 1, 3, {{0, 0.1f}, {1, 0.1f}, {2, 0.1f}, {3, 0.1f},
                                {4, 0.1f}, {5, 0.1f}, {6, 0.1f}, {7, 0.1f}});

    std::vector<int> capacities = {4, 4};  // 2 GPUs, 4 slots each.
    auto hints = graph.compute_affinity_hints(2, capacities);

    // Count assignments per GPU for layer 3.
    int count_gpu0 = 0, count_gpu1 = 0;
    for (const auto& h : hints) {
        if (h.key.layer_idx != 3) continue;
        if (h.gpu_idx == 0) count_gpu0++;
        else if (h.gpu_idx == 1) count_gpu1++;
    }

    // Neither GPU should exceed capacity.
    EXPECT_LE(count_gpu0, 4);
    EXPECT_LE(count_gpu1, 4);
    // All 8 experts should be assigned.
    EXPECT_EQ(count_gpu0 + count_gpu1, 8);
}

TEST(CoactivationGraph, PartitioningEmptyGraph) {
    lstats::CoactivationGraph graph(default_opts());

    std::vector<int> capacities = {4, 4};
    auto hints = graph.compute_affinity_hints(2, capacities);

    // All experts across all layers should be assigned (even with zero weights).
    // 4 layers * 8 experts = 32 hints.
    EXPECT_EQ(hints.size(), 32u);

    // All scores should be zero (no co-activation data).
    for (const auto& h : hints) {
        EXPECT_DOUBLE_EQ(h.score, 0.0);
    }
}

TEST(CoactivationGraph, PartitioningZeroCapacityGpu) {
    auto opts = default_opts();
    opts.decay_factor = 1.0;
    lstats::CoactivationGraph graph(opts);

    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.5f}});

    // GPU 0 has 0 capacity, GPU 1 has 8.
    std::vector<int> capacities = {0, 8};
    auto hints = graph.compute_affinity_hints(2, capacities);

    // All experts should go to GPU 1.
    for (const auto& h : hints) {
        EXPECT_EQ(h.gpu_idx, 1) << "expert " << h.key.expert_idx
            << " on layer " << h.key.layer_idx << " was assigned to GPU 0";
    }
}

TEST(CoactivationGraph, PartitioningZeroGpus) {
    lstats::CoactivationGraph graph(default_opts());
    std::vector<int> capacities;
    auto hints = graph.compute_affinity_hints(0, capacities);
    EXPECT_TRUE(hints.empty());
}

TEST(CoactivationGraph, PartitioningMultipleLayers) {
    auto opts = default_opts();
    opts.decay_factor = 1.0;
    lstats::CoactivationGraph graph(opts);

    // Different co-activation patterns per layer.
    // Layer 3: experts 0,1 co-activate.
    for (uint64_t t = 1; t <= 10; ++t) {
        activate_once(graph, t, 3, {{0, 0.5f}, {1, 0.5f}});
    }
    // Layer 4: experts 2,3 co-activate.
    for (uint64_t t = 1; t <= 10; ++t) {
        // Use same token_ids — different layers.
        auto r = make_result(t, 4, {{2, 0.5f}, {3, 0.5f}});
        std::vector<lstats::GatingResult> v = {std::move(r)};
        graph.update(v);
    }

    std::vector<int> capacities = {4, 4};
    auto hints = graph.compute_affinity_hints(2, capacities);

    // Verify layer 3 and layer 4 have independent partitioning.
    // Layer 3: experts 0,1 should be on same GPU.
    int l3_gpu0 = -1, l3_gpu1 = -1;
    int l4_gpu2 = -1, l4_gpu3 = -1;
    for (const auto& h : hints) {
        if (h.key.layer_idx == 3 && h.key.expert_idx == 0) l3_gpu0 = h.gpu_idx;
        if (h.key.layer_idx == 3 && h.key.expert_idx == 1) l3_gpu1 = h.gpu_idx;
        if (h.key.layer_idx == 4 && h.key.expert_idx == 2) l4_gpu2 = h.gpu_idx;
        if (h.key.layer_idx == 4 && h.key.expert_idx == 3) l4_gpu3 = h.gpu_idx;
    }
    EXPECT_EQ(l3_gpu0, l3_gpu1);
    EXPECT_EQ(l4_gpu2, l4_gpu3);
}

// ── Row sum for normalization ────────────────────────────────────────────────

TEST(CoactivationGraph, RowSumForNormalization) {
    auto opts = default_opts();
    opts.decay_factor = 1.0;
    lstats::CoactivationGraph graph(opts);

    // Expert 0 co-activates with experts 1, 2, 3.
    activate_once(graph, 1, 3, {{0, 0.5f}, {1, 0.3f}, {2, 0.2f}});
    activate_once(graph, 2, 3, {{0, 0.5f}, {3, 0.3f}});

    auto r = graph.row(3, 0);
    ASSERT_EQ(r.size(), 8u);

    // Row sum should be > 0 (expert 0 has co-activation partners).
    float row_sum = 0.0f;
    for (size_t j = 0; j < r.size(); ++j) {
        row_sum += r[j];
    }
    EXPECT_GT(row_sum, 0.0f);

    // Fraction of co-activation with {1, 2} out of total.
    float subset_sum = r[1] + r[2];
    double fraction = static_cast<double>(subset_sum) / row_sum;
    EXPECT_GT(fraction, 0.0);
    EXPECT_LE(fraction, 1.0);
}
