// Unit tests for adaptive gating CUDA kernel.
// Tests threshold-based expert pruning on top-K gating output.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "smxx/gating/adaptive_gating.h"
#include "sm120/gating/topk_gating.h"
#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;

// ── CUDA error checking ─────────────────────────────────────────────────────

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: "                         \
                                     << cudaGetErrorString(_err);              \
    } while (0)

// ── CPU reference ───────────────────────────────────────────────────────────

/// CPU reference implementation of adaptive gating.
static void adaptive_gating_ref(float* out_weights, int32_t* out_indices,
                                 int32_t* expert_counts,
                                 const float* in_weights,
                                 const int32_t* in_indices,
                                 int num_tokens, int topk, float threshold) {
    for (int t = 0; t < num_tokens; ++t) {
        const int offset = t * topk;

        // Copy and sort by weight descending.
        std::vector<std::pair<float, int32_t>> experts(topk);
        for (int k = 0; k < topk; ++k) {
            experts[k] = {in_weights[offset + k], in_indices[offset + k]};
        }
        std::sort(experts.begin(), experts.end(),
                  [](const auto& a, const auto& b) {
                      return a.first > b.first;
                  });

        // Compute total weight.
        float total_weight = 0.0f;
        for (int k = 0; k < topk; ++k) {
            total_weight += experts[k].first;
        }

        // Cumulative scan.
        float target = threshold * total_weight;
        float cumsum = 0.0f;
        int count = topk;
        for (int k = 0; k < topk; ++k) {
            cumsum += experts[k].first;
            if (cumsum >= target) {
                count = k + 1;
                break;
            }
        }
        if (count < 1) count = 1;

        // Write kept experts.
        for (int k = 0; k < count; ++k) {
            out_weights[offset + k] = experts[k].first;
            out_indices[offset + k] = experts[k].second;
        }

        // Sentinel padding.
        for (int k = count; k < topk; ++k) {
            out_weights[offset + k] = 0.0f;
            out_indices[offset + k] = -1;
        }

        expert_counts[t] = count;
    }
}

// ── Helper: run kernel and compare against CPU reference ────────────────────

struct AdaptiveTestCase {
    int num_tokens;
    int topk;
    float threshold;
};

static void run_adaptive_test(const AdaptiveTestCase& tc,
                               const std::vector<float>& h_in_weights,
                               const std::vector<int32_t>& h_in_indices,
                               float weight_tol = 1e-6f) {
    const int T = tc.num_tokens;
    const int K = tc.topk;

    // CPU reference.
    std::vector<float> ref_weights(T * K);
    std::vector<int32_t> ref_indices(T * K);
    std::vector<int32_t> ref_counts(T);
    adaptive_gating_ref(ref_weights.data(), ref_indices.data(),
                         ref_counts.data(), h_in_weights.data(),
                         h_in_indices.data(), T, K, tc.threshold);

    // GPU.
    float *d_in_w, *d_out_w;
    int32_t *d_in_i, *d_out_i, *d_counts;
    size_t weight_bytes = static_cast<size_t>(T) * K * sizeof(float);
    size_t index_bytes = static_cast<size_t>(T) * K * sizeof(int32_t);
    size_t count_bytes = static_cast<size_t>(T) * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_in_w, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_in_i, index_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_w, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_i, index_bytes));
    CUDA_CHECK(cudaMalloc(&d_counts, count_bytes));
    CUDA_CHECK(
        cudaMemcpy(d_in_w, h_in_weights.data(), weight_bytes,
                   cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_in_i, h_in_indices.data(), index_bytes,
                   cudaMemcpyHostToDevice));

    lc::AdaptiveGatingParams params{};
    params.num_tokens = T;
    params.topk = K;
    params.threshold = tc.threshold;

    lc::launch_adaptive_gating(d_out_w, d_out_i, d_counts, d_in_w, d_in_i,
                                params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_weights(T * K);
    std::vector<int32_t> gpu_indices(T * K);
    std::vector<int32_t> gpu_counts(T);
    CUDA_CHECK(cudaMemcpy(gpu_weights.data(), d_out_w, weight_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_indices.data(), d_out_i, index_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_counts.data(), d_counts, count_bytes,
                          cudaMemcpyDeviceToHost));

    // Compare counts.
    for (int t = 0; t < T; ++t) {
        ASSERT_EQ(gpu_counts[t], ref_counts[t])
            << "Token " << t << ": expert count mismatch";
    }

    // Compare kept experts (order and weights).
    for (int t = 0; t < T; ++t) {
        int count = ref_counts[t];
        for (int k = 0; k < count; ++k) {
            ASSERT_EQ(gpu_indices[t * K + k], ref_indices[t * K + k])
                << "Token " << t << " slot " << k << ": index mismatch";
            ASSERT_NEAR(gpu_weights[t * K + k], ref_weights[t * K + k],
                        weight_tol)
                << "Token " << t << " slot " << k << ": weight mismatch";
        }
        // Check sentinel padding.
        for (int k = count; k < K; ++k) {
            ASSERT_EQ(gpu_indices[t * K + k], -1)
                << "Token " << t << " slot " << k << ": sentinel index";
            ASSERT_EQ(gpu_weights[t * K + k], 0.0f)
                << "Token " << t << " slot " << k << ": sentinel weight";
        }
    }

    cudaFree(d_in_w);
    cudaFree(d_in_i);
    cudaFree(d_out_w);
    cudaFree(d_out_i);
    cudaFree(d_counts);
}

// ── Fixture ─────────────────────────────────────────────────────────────────

class AdaptiveGatingTest : public ::testing::Test {
protected:
    void SetUp() override { gen_.seed(42); }

    /// Fill weights with well-separated values so sort order is deterministic.
    /// Produces K weights per token summing to roughly 1.0 (normalized-ish).
    void fill_separated(std::vector<float>& weights,
                        std::vector<int32_t>& indices,
                        int num_tokens, int topk) {
        weights.resize(static_cast<size_t>(num_tokens) * topk);
        indices.resize(static_cast<size_t>(num_tokens) * topk);
        std::uniform_int_distribution<int32_t> expert_dist(0, 255);
        for (int t = 0; t < num_tokens; ++t) {
            // Generate K distinct expert indices.
            std::vector<int32_t> eidxs;
            while (static_cast<int>(eidxs.size()) < topk) {
                int32_t e = expert_dist(gen_);
                bool dup = false;
                for (auto x : eidxs)
                    if (x == e) { dup = true; break; }
                if (!dup) eidxs.push_back(e);
            }
            // Generate separated weights: largest first with gaps.
            float base = 1.0f / topk;
            for (int k = 0; k < topk; ++k) {
                // Weight decreases: 0.5, 0.25, 0.125, ... (geometric-ish)
                weights[t * topk + k] = base * std::pow(0.5f, k);
                indices[t * topk + k] = eidxs[k];
            }
            // Shuffle to simulate top-K output sorted by biased score (not weight).
            for (int k = topk - 1; k > 0; --k) {
                std::uniform_int_distribution<int> sd(0, k);
                int j = sd(gen_);
                std::swap(weights[t * topk + k], weights[t * topk + j]);
                std::swap(indices[t * topk + k], indices[t * topk + j]);
            }
        }
    }

    /// Fill with random weights and indices.
    void fill_random(std::vector<float>& weights,
                     std::vector<int32_t>& indices,
                     int num_tokens, int topk,
                     float lo = 0.01f, float hi = 1.0f) {
        weights.resize(static_cast<size_t>(num_tokens) * topk);
        indices.resize(static_cast<size_t>(num_tokens) * topk);
        std::uniform_real_distribution<float> wdist(lo, hi);
        std::uniform_int_distribution<int32_t> edist(0, 255);
        for (int t = 0; t < num_tokens; ++t) {
            std::vector<int32_t> eidxs;
            while (static_cast<int>(eidxs.size()) < topk) {
                int32_t e = edist(gen_);
                bool dup = false;
                for (auto x : eidxs)
                    if (x == e) { dup = true; break; }
                if (!dup) eidxs.push_back(e);
            }
            for (int k = 0; k < topk; ++k) {
                weights[t * topk + k] = wdist(gen_);
                indices[t * topk + k] = eidxs[k];
            }
        }
    }

    std::mt19937 gen_;
};

// ═════════════════════════════════════════════════════════════════════════════
// Core Adaptive Gating Tests
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(AdaptiveGatingTest, Threshold1_0_KeepsAll) {
    REQUIRES_GPU();

    // threshold=1.0 requires all cumulative weight, so keeps all K experts.
    constexpr int T = 8, K = 8;
    std::vector<float> weights;
    std::vector<int32_t> indices;
    fill_separated(weights, indices, T, K);

    AdaptiveTestCase tc{T, K, 1.0f};
    run_adaptive_test(tc, weights, indices);
}

TEST_F(AdaptiveGatingTest, Threshold0_92_DropsExperts) {
    REQUIRES_GPU();

    // With geometric weights (0.5, 0.25, 0.125, ...), threshold=0.92 should
    // drop the lowest-weight experts since top few cover most of the mass.
    constexpr int T = 4, K = 8;
    std::vector<float> weights;
    std::vector<int32_t> indices;
    fill_separated(weights, indices, T, K);

    AdaptiveTestCase tc{T, K, 0.92f};
    run_adaptive_test(tc, weights, indices);
}

TEST_F(AdaptiveGatingTest, VariableCountsAcrossTokens) {
    REQUIRES_GPU();

    // Construct tokens with different weight distributions so they keep
    // different numbers of experts at the same threshold.
    constexpr int T = 3, K = 4;
    std::vector<float> weights(T * K);
    std::vector<int32_t> indices(T * K);

    // Token 0: one dominant expert (0.9, 0.05, 0.03, 0.02) → keeps 1
    weights[0] = 0.9f; weights[1] = 0.05f; weights[2] = 0.03f; weights[3] = 0.02f;
    indices[0] = 10; indices[1] = 20; indices[2] = 30; indices[3] = 40;

    // Token 1: two nearly equal (0.45, 0.40, 0.10, 0.05) → keeps 2
    weights[4] = 0.45f; weights[5] = 0.40f; weights[6] = 0.10f; weights[7] = 0.05f;
    indices[4] = 11; indices[5] = 21; indices[6] = 31; indices[7] = 41;

    // Token 2: uniform-ish (0.28, 0.26, 0.24, 0.22) → keeps all 4
    weights[8] = 0.28f; weights[9] = 0.26f; weights[10] = 0.24f; weights[11] = 0.22f;
    indices[8] = 12; indices[9] = 22; indices[10] = 32; indices[11] = 42;

    AdaptiveTestCase tc{T, K, 0.90f};
    run_adaptive_test(tc, weights, indices);
}

TEST_F(AdaptiveGatingTest, SentinelPadding) {
    REQUIRES_GPU();

    // Verify that positions >= count have sentinel values.
    constexpr int T = 4, K = 6;
    std::vector<float> weights;
    std::vector<int32_t> indices;
    fill_separated(weights, indices, T, K);

    AdaptiveTestCase tc{T, K, 0.80f};
    run_adaptive_test(tc, weights, indices);
    // run_adaptive_test already checks sentinel padding.
}

TEST_F(AdaptiveGatingTest, SingleToken) {
    REQUIRES_GPU();

    constexpr int T = 1, K = 8;
    std::vector<float> weights;
    std::vector<int32_t> indices;
    fill_random(weights, indices, T, K);

    AdaptiveTestCase tc{T, K, 0.92f};
    run_adaptive_test(tc, weights, indices);
}

TEST_F(AdaptiveGatingTest, LargeBatch) {
    REQUIRES_GPU();

    constexpr int T = 1024, K = 8;
    std::vector<float> weights;
    std::vector<int32_t> indices;
    fill_random(weights, indices, T, K);

    AdaptiveTestCase tc{T, K, 0.92f};
    run_adaptive_test(tc, weights, indices);
}

TEST_F(AdaptiveGatingTest, Topk1) {
    REQUIRES_GPU();

    // K=1 always keeps exactly 1 expert regardless of threshold.
    constexpr int T = 8, K = 1;
    std::vector<float> weights;
    std::vector<int32_t> indices;
    fill_random(weights, indices, T, K);

    AdaptiveTestCase tc{T, K, 0.5f};
    run_adaptive_test(tc, weights, indices);
}

TEST_F(AdaptiveGatingTest, ZeroTokens) {
    REQUIRES_GPU();

    // Should not crash.
    lc::AdaptiveGatingParams params{};
    params.num_tokens = 0;
    params.topk = 8;
    params.threshold = 0.92f;

    lc::launch_adaptive_gating(nullptr, nullptr, nullptr, nullptr, nullptr,
                                params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
}

TEST_F(AdaptiveGatingTest, ExpertCountCorrectness) {
    REQUIRES_GPU();

    // Verify expert_counts match the number of non-sentinel entries.
    constexpr int T = 16, K = 8;
    std::vector<float> weights;
    std::vector<int32_t> indices;
    fill_random(weights, indices, T, K);

    float *d_in_w, *d_out_w;
    int32_t *d_in_i, *d_out_i, *d_counts;
    size_t w_bytes = T * K * sizeof(float);
    size_t i_bytes = T * K * sizeof(int32_t);
    CUDA_CHECK(cudaMalloc(&d_in_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_in_i, i_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_i, i_bytes));
    CUDA_CHECK(cudaMalloc(&d_counts, T * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_in_w, weights.data(), w_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_i, indices.data(), i_bytes,
                          cudaMemcpyHostToDevice));

    lc::AdaptiveGatingParams params{T, K, 0.85f};
    lc::launch_adaptive_gating(d_out_w, d_out_i, d_counts, d_in_w, d_in_i,
                                params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_w(T * K);
    std::vector<int32_t> h_i(T * K);
    std::vector<int32_t> h_counts(T);
    CUDA_CHECK(cudaMemcpy(h_w.data(), d_out_w, w_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_i.data(), d_out_i, i_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, T * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    for (int t = 0; t < T; ++t) {
        int count = h_counts[t];
        ASSERT_GE(count, 1) << "Token " << t << ": must keep at least 1";
        ASSERT_LE(count, K) << "Token " << t << ": can't exceed K";

        // Non-sentinel entries.
        int actual_count = 0;
        for (int k = 0; k < K; ++k) {
            if (h_i[t * K + k] != -1) ++actual_count;
        }
        ASSERT_EQ(actual_count, count)
            << "Token " << t << ": count vs non-sentinel mismatch";
    }

    cudaFree(d_in_w);
    cudaFree(d_in_i);
    cudaFree(d_out_w);
    cudaFree(d_out_i);
    cudaFree(d_counts);
}

TEST_F(AdaptiveGatingTest, CumulativeWeightVerification) {
    REQUIRES_GPU();

    // Verify that kept weight >= threshold × total for all tokens.
    constexpr int T = 32, K = 8;
    constexpr float threshold = 0.92f;
    std::vector<float> weights;
    std::vector<int32_t> indices;
    fill_random(weights, indices, T, K);

    float *d_in_w, *d_out_w;
    int32_t *d_in_i, *d_out_i, *d_counts;
    size_t w_bytes = T * K * sizeof(float);
    size_t i_bytes = T * K * sizeof(int32_t);
    CUDA_CHECK(cudaMalloc(&d_in_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_in_i, i_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_i, i_bytes));
    CUDA_CHECK(cudaMalloc(&d_counts, T * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_in_w, weights.data(), w_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_i, indices.data(), i_bytes,
                          cudaMemcpyHostToDevice));

    lc::AdaptiveGatingParams params{T, K, threshold};
    lc::launch_adaptive_gating(d_out_w, d_out_i, d_counts, d_in_w, d_in_i,
                                params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out_w(T * K);
    std::vector<int32_t> h_counts(T);
    CUDA_CHECK(cudaMemcpy(h_out_w.data(), d_out_w, w_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, T * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    for (int t = 0; t < T; ++t) {
        // Total weight from input.
        float total = 0.0f;
        for (int k = 0; k < K; ++k) {
            total += weights[t * K + k];
        }

        // Kept weight from output.
        float kept = 0.0f;
        for (int k = 0; k < h_counts[t]; ++k) {
            kept += h_out_w[t * K + k];
        }

        ASSERT_GE(kept, threshold * total - 1e-6f)
            << "Token " << t << ": kept weight " << kept
            << " < threshold * total " << (threshold * total);
    }

    cudaFree(d_in_w);
    cudaFree(d_in_i);
    cudaFree(d_out_w);
    cudaFree(d_out_i);
    cudaFree(d_counts);
}

TEST_F(AdaptiveGatingTest, SortOrderWithBias) {
    REQUIRES_GPU();

    // Simulate bias reordering: top-K output sorted by biased score (not weight).
    // Expert 0 has lowest weight but was selected first due to high bias.
    // Adaptive gating should re-sort by weight correctly.
    constexpr int T = 1, K = 4;
    std::vector<float> weights = {0.05f, 0.40f, 0.30f, 0.25f};
    std::vector<int32_t> indices = {100, 200, 300, 400};

    // After re-sort by weight: 200(0.40), 300(0.30), 400(0.25), 100(0.05)

    float *d_in_w, *d_out_w;
    int32_t *d_in_i, *d_out_i, *d_counts;
    CUDA_CHECK(cudaMalloc(&d_in_w, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in_i, K * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_out_w, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_i, K * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_counts, sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_in_w, weights.data(), K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_i, indices.data(), K * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    // threshold=1.0 to keep all, just verify sort order.
    lc::AdaptiveGatingParams params{T, K, 1.0f};
    lc::launch_adaptive_gating(d_out_w, d_out_i, d_counts, d_in_w, d_in_i,
                                params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_w(K);
    std::vector<int32_t> h_i(K);
    CUDA_CHECK(cudaMemcpy(h_w.data(), d_out_w, K * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_i.data(), d_out_i, K * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // Verify descending weight order.
    ASSERT_EQ(h_i[0], 200); ASSERT_NEAR(h_w[0], 0.40f, 1e-6f);
    ASSERT_EQ(h_i[1], 300); ASSERT_NEAR(h_w[1], 0.30f, 1e-6f);
    ASSERT_EQ(h_i[2], 400); ASSERT_NEAR(h_w[2], 0.25f, 1e-6f);
    ASSERT_EQ(h_i[3], 100); ASSERT_NEAR(h_w[3], 0.05f, 1e-6f);

    cudaFree(d_in_w);
    cudaFree(d_in_i);
    cudaFree(d_out_w);
    cudaFree(d_out_i);
    cudaFree(d_counts);
}

// ═════════════════════════════════════════════════════════════════════════════
// Validation Tests
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(AdaptiveGatingTest, InvalidParams) {
    REQUIRES_GPU();

    lc::AdaptiveGatingParams params{};
    params.num_tokens = 1;

    // topk = 0
    params.topk = 0;
    params.threshold = 0.92f;
    EXPECT_THROW(lc::launch_adaptive_gating(nullptr, nullptr, nullptr, nullptr,
                                             nullptr, params, nullptr),
                 std::invalid_argument);

    // topk = 9
    params.topk = 9;
    EXPECT_THROW(lc::launch_adaptive_gating(nullptr, nullptr, nullptr, nullptr,
                                             nullptr, params, nullptr),
                 std::invalid_argument);

    // threshold < 0
    params.topk = 8;
    params.threshold = -0.1f;
    EXPECT_THROW(lc::launch_adaptive_gating(nullptr, nullptr, nullptr, nullptr,
                                             nullptr, params, nullptr),
                 std::invalid_argument);

    // threshold > 1
    params.threshold = 1.1f;
    EXPECT_THROW(lc::launch_adaptive_gating(nullptr, nullptr, nullptr, nullptr,
                                             nullptr, params, nullptr),
                 std::invalid_argument);
}

// ═════════════════════════════════════════════════════════════════════════════
// Renormalization pipeline tests
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(AdaptiveGatingTest, RenormalizedInputs) {
    REQUIRES_GPU();

    // Simulate renormalized top-K output (weights sum to routed_scaling_factor).
    // Adaptive gating should still work correctly.
    constexpr int T = 8, K = 6;
    constexpr float scaling = 2.5f;

    std::vector<float> weights;
    std::vector<int32_t> indices;
    fill_random(weights, indices, T, K, 0.1f, 1.0f);

    // Renormalize each token's weights to sum to scaling factor.
    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += weights[t * K + k];
        float scale = scaling / sum;
        for (int k = 0; k < K; ++k) weights[t * K + k] *= scale;
    }

    AdaptiveTestCase tc{T, K, 0.92f};
    run_adaptive_test(tc, weights, indices);
}

TEST_F(AdaptiveGatingTest, NonRenormalizedInputs) {
    REQUIRES_GPU();

    // Raw sigmoid weights (not renormalized). Values in ~[0.1, 0.7] range.
    constexpr int T = 8, K = 6;
    std::vector<float> weights;
    std::vector<int32_t> indices;
    fill_random(weights, indices, T, K, 0.1f, 0.7f);

    AdaptiveTestCase tc{T, K, 0.92f};
    run_adaptive_test(tc, weights, indices);
}

// ═════════════════════════════════════════════════════════════════════════════
// End-to-end pipeline tests (top-K → adaptive)
// ═════════════════════════════════════════════════════════════════════════════

/// CPU sigmoid reference (same as topk_gating_test.cpp).
static float sigmoid_ref(float x) {
    return 0.5f * std::tanh(0.5f * x) + 0.5f;
}

TEST_F(AdaptiveGatingTest, V32Config) {
    REQUIRES_GPU();

    // End-to-end: topk_gating(grouped) → adaptive_gating.
    // V3.2: 256 experts, 8 groups, top-4 groups, top-8, renormalize, scaling=2.5
    constexpr int T = 8, E = 256, K = 8;

    // Generate logits.
    std::vector<float> logits(T * E);
    {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (auto& x : logits) x = dist(gen_);
    }
    std::vector<float> bias(E);
    {
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        for (auto& x : bias) x = dist(gen_);
    }

    // GPU: run top-K gating.
    float *d_logits, *d_bias, *d_topk_w, *d_adapt_w;
    int32_t *d_topk_i, *d_adapt_i, *d_counts;
    CUDA_CHECK(cudaMalloc(&d_logits, T * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_w, T * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_i, T * K * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_adapt_w, T * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_adapt_i, T * K * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_counts, T * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), T * E * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), E * sizeof(float),
                          cudaMemcpyHostToDevice));

    lc::TopkGatingParams topk_params{};
    topk_params.num_tokens = T;
    topk_params.num_experts = E;
    topk_params.topk = K;
    topk_params.n_group = 8;
    topk_params.topk_group = 4;
    topk_params.routed_scaling_factor = 2.5f;
    topk_params.renormalize = true;

    lc::launch_topk_gating(d_topk_w, d_topk_i, d_logits, d_bias, topk_params,
                           nullptr);

    lc::AdaptiveGatingParams adapt_params{T, K, 0.92f};
    lc::launch_adaptive_gating(d_adapt_w, d_adapt_i, d_counts, d_topk_w,
                                d_topk_i, adapt_params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read results.
    std::vector<float> h_adapt_w(T * K);
    std::vector<int32_t> h_adapt_i(T * K);
    std::vector<int32_t> h_counts(T);
    CUDA_CHECK(cudaMemcpy(h_adapt_w.data(), d_adapt_w, T * K * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_adapt_i.data(), d_adapt_i, T * K * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, T * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // Also read top-K output for CPU reference.
    std::vector<float> h_topk_w(T * K);
    std::vector<int32_t> h_topk_i(T * K);
    CUDA_CHECK(cudaMemcpy(h_topk_w.data(), d_topk_w, T * K * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_topk_i.data(), d_topk_i, T * K * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // CPU reference on top-K output.
    std::vector<float> ref_w(T * K);
    std::vector<int32_t> ref_i(T * K);
    std::vector<int32_t> ref_counts(T);
    adaptive_gating_ref(ref_w.data(), ref_i.data(), ref_counts.data(),
                         h_topk_w.data(), h_topk_i.data(), T, K, 0.92f);

    for (int t = 0; t < T; ++t) {
        ASSERT_EQ(h_counts[t], ref_counts[t])
            << "V3.2 token " << t << ": count mismatch";
        ASSERT_GE(h_counts[t], 1);
        ASSERT_LE(h_counts[t], K);

        for (int k = 0; k < h_counts[t]; ++k) {
            ASSERT_EQ(h_adapt_i[t * K + k], ref_i[t * K + k])
                << "V3.2 token " << t << " slot " << k;
            ASSERT_NEAR(h_adapt_w[t * K + k], ref_w[t * K + k], 1e-5f)
                << "V3.2 token " << t << " slot " << k;
        }
    }

    cudaFree(d_logits);
    cudaFree(d_bias);
    cudaFree(d_topk_w);
    cudaFree(d_topk_i);
    cudaFree(d_adapt_w);
    cudaFree(d_adapt_i);
    cudaFree(d_counts);
}

TEST_F(AdaptiveGatingTest, GLM5Config) {
    REQUIRES_GPU();

    // End-to-end: topk_gating(simple) → adaptive_gating.
    // GLM-5: 256 experts, n_group=1, topk=8, scaling=2.5
    constexpr int T = 8, E = 256, K = 8;

    std::vector<float> logits(T * E);
    {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (auto& x : logits) x = dist(gen_);
    }
    std::vector<float> bias(E);
    {
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        for (auto& x : bias) x = dist(gen_);
    }

    float *d_logits, *d_bias, *d_topk_w, *d_adapt_w;
    int32_t *d_topk_i, *d_adapt_i, *d_counts;
    CUDA_CHECK(cudaMalloc(&d_logits, T * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_w, T * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_i, T * K * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_adapt_w, T * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_adapt_i, T * K * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_counts, T * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), T * E * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), E * sizeof(float),
                          cudaMemcpyHostToDevice));

    lc::TopkGatingParams topk_params{};
    topk_params.num_tokens = T;
    topk_params.num_experts = E;
    topk_params.topk = K;
    topk_params.n_group = 1;
    topk_params.topk_group = 1;
    topk_params.routed_scaling_factor = 2.5f;
    topk_params.renormalize = true;

    lc::launch_topk_gating(d_topk_w, d_topk_i, d_logits, d_bias, topk_params,
                           nullptr);

    lc::AdaptiveGatingParams adapt_params{T, K, 0.92f};
    lc::launch_adaptive_gating(d_adapt_w, d_adapt_i, d_counts, d_topk_w,
                                d_topk_i, adapt_params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_adapt_w(T * K);
    std::vector<int32_t> h_adapt_i(T * K);
    std::vector<int32_t> h_counts(T);
    CUDA_CHECK(cudaMemcpy(h_adapt_w.data(), d_adapt_w, T * K * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_adapt_i.data(), d_adapt_i, T * K * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, T * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // Read top-K for CPU ref.
    std::vector<float> h_topk_w(T * K);
    std::vector<int32_t> h_topk_i(T * K);
    CUDA_CHECK(cudaMemcpy(h_topk_w.data(), d_topk_w, T * K * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_topk_i.data(), d_topk_i, T * K * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    std::vector<float> ref_w(T * K);
    std::vector<int32_t> ref_i(T * K);
    std::vector<int32_t> ref_counts(T);
    adaptive_gating_ref(ref_w.data(), ref_i.data(), ref_counts.data(),
                         h_topk_w.data(), h_topk_i.data(), T, K, 0.92f);

    for (int t = 0; t < T; ++t) {
        ASSERT_EQ(h_counts[t], ref_counts[t])
            << "GLM-5 token " << t << ": count mismatch";
        ASSERT_GE(h_counts[t], 1);
        ASSERT_LE(h_counts[t], K);

        for (int k = 0; k < h_counts[t]; ++k) {
            ASSERT_EQ(h_adapt_i[t * K + k], ref_i[t * K + k])
                << "GLM-5 token " << t << " slot " << k;
            ASSERT_NEAR(h_adapt_w[t * K + k], ref_w[t * K + k], 1e-5f)
                << "GLM-5 token " << t << " slot " << k;
        }
    }

    cudaFree(d_logits);
    cudaFree(d_bias);
    cudaFree(d_topk_w);
    cudaFree(d_topk_i);
    cudaFree(d_adapt_w);
    cudaFree(d_adapt_i);
    cudaFree(d_counts);
}
