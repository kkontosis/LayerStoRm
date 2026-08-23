// Unit tests for top-K gating CUDA kernels.
// Tests both simple (n_group=1) and grouped (n_group>1) routing variants.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <vector>

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

static float sigmoid_ref(float x) {
    return 0.5f * std::tanh(0.5f * x) + 0.5f;
}

/// CPU reference implementation of top-K gating (grouped or simple).
static void topk_gating_ref(
    float* weights, int32_t* indices, const float* logits, const float* bias,
    int num_tokens, int num_experts, int n_group, int topk_group, int topk,
    float routed_scaling_factor, bool renormalize) {

    const int experts_per_group = num_experts / n_group;

    for (int t = 0; t < num_tokens; ++t) {
        const float* tok_logits = logits + t * num_experts;

        // Compute sigmoid scores.
        std::vector<float> scores(num_experts);
        std::vector<float> sel(num_experts);
        for (int e = 0; e < num_experts; ++e) {
            scores[e] = sigmoid_ref(tok_logits[e]);
            sel[e] = bias ? scores[e] + bias[e] : scores[e];
        }

        // Group selection (grouped routing only).
        if (n_group > 1) {
            // Compute group scores: sum of top-2 selection scores per group.
            std::vector<float> group_scores(n_group);
            for (int g = 0; g < n_group; ++g) {
                int start = g * experts_per_group;
                float top1 = -FLT_MAX, top2 = -FLT_MAX;
                for (int i = 0; i < experts_per_group; ++i) {
                    float v = sel[start + i];
                    if (v > top1) {
                        top2 = top1;
                        top1 = v;
                    } else if (v > top2) {
                        top2 = v;
                    }
                }
                group_scores[g] = top1 + top2;
            }

            // Select top-topk_group groups.
            std::vector<bool> group_selected(n_group, false);
            for (int g = 0; g < topk_group; ++g) {
                float best = -FLT_MAX;
                int best_g = 0;
                for (int i = 0; i < n_group; ++i) {
                    if (!group_selected[i] && group_scores[i] > best) {
                        best = group_scores[i];
                        best_g = i;
                    }
                }
                group_selected[best_g] = true;
            }

            // Mask non-candidates.
            for (int e = 0; e < num_experts; ++e) {
                if (!group_selected[e / experts_per_group])
                    sel[e] = -FLT_MAX;
            }
        }

        // Select top-K experts by selection score.
        float* out_w = weights + t * topk;
        int32_t* out_i = indices + t * topk;

        for (int k = 0; k < topk; ++k) {
            float best = -FLT_MAX;
            int best_e = 0;
            for (int e = 0; e < num_experts; ++e) {
                if (sel[e] > best ||
                    (sel[e] == best && e < best_e)) {
                    best = sel[e];
                    best_e = e;
                }
            }
            out_i[k] = best_e;
            out_w[k] = scores[best_e];  // unbiased
            sel[best_e] = -FLT_MAX;
        }

        // Renormalize.
        if (renormalize) {
            float sum = 0.0f;
            for (int k = 0; k < topk; ++k) sum += out_w[k];
            if (sum > 0.0f) {
                float scale = routed_scaling_factor / sum;
                for (int k = 0; k < topk; ++k) out_w[k] *= scale;
            }
        }
    }
}

// ── Helper: run kernel and compare against CPU reference ────────────────────

struct GatingTestCase {
    int num_tokens;
    int num_experts;
    int topk;
    int n_group;
    int topk_group;
    float routed_scaling_factor;
    bool renormalize;
    bool use_bias;
};

static void run_gating_test(const GatingTestCase& tc,
                            const std::vector<float>& h_logits,
                            const std::vector<float>& h_bias,
                            float weight_tol = 1e-5f) {
    const int E = tc.num_experts;
    const int T = tc.num_tokens;
    const int K = tc.topk;

    // CPU reference.
    std::vector<float> ref_weights(T * K);
    std::vector<int32_t> ref_indices(T * K);
    topk_gating_ref(ref_weights.data(), ref_indices.data(), h_logits.data(),
                    tc.use_bias ? h_bias.data() : nullptr, T, E, tc.n_group,
                    tc.topk_group, K, tc.routed_scaling_factor,
                    tc.renormalize);

    // GPU.
    float *d_logits, *d_bias = nullptr, *d_weights;
    int32_t* d_indices;
    size_t logit_bytes = static_cast<size_t>(T) * E * sizeof(float);
    size_t weight_bytes = static_cast<size_t>(T) * K * sizeof(float);
    size_t index_bytes = static_cast<size_t>(T) * K * sizeof(int32_t);
    size_t bias_bytes = static_cast<size_t>(E) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_logits, logit_bytes));
    CUDA_CHECK(cudaMalloc(&d_weights, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_indices, index_bytes));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), logit_bytes,
                          cudaMemcpyHostToDevice));
    if (tc.use_bias) {
        CUDA_CHECK(cudaMalloc(&d_bias, bias_bytes));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_bytes,
                              cudaMemcpyHostToDevice));
    }

    lc::TopkGatingParams params{};
    params.num_tokens = T;
    params.num_experts = E;
    params.topk = K;
    params.n_group = tc.n_group;
    params.topk_group = tc.topk_group;
    params.routed_scaling_factor = tc.routed_scaling_factor;
    params.renormalize = tc.renormalize;

    lc::launch_topk_gating(d_weights, d_indices, d_logits, d_bias, params,
                           nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_weights(T * K);
    std::vector<int32_t> gpu_indices(T * K);
    CUDA_CHECK(cudaMemcpy(gpu_weights.data(), d_weights, weight_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_indices.data(), d_indices, index_bytes,
                          cudaMemcpyDeviceToHost));

    // Compare: same set of selected experts per token, matching weights.
    for (int t = 0; t < T; ++t) {
        std::set<int> ref_set, gpu_set;
        for (int k = 0; k < K; ++k) {
            ref_set.insert(ref_indices[t * K + k]);
            gpu_set.insert(gpu_indices[t * K + k]);
        }
        ASSERT_EQ(ref_set, gpu_set)
            << "Token " << t << ": selected expert sets differ";

        // Match weights by expert id.
        for (int k = 0; k < K; ++k) {
            int gpu_eid = gpu_indices[t * K + k];
            // Find the same expert in ref.
            float ref_w = 0.0f;
            for (int j = 0; j < K; ++j) {
                if (ref_indices[t * K + j] == gpu_eid) {
                    ref_w = ref_weights[t * K + j];
                    break;
                }
            }
            ASSERT_NEAR(gpu_weights[t * K + k], ref_w, weight_tol)
                << "Token " << t << " expert " << gpu_eid
                << " weight mismatch";
        }
    }

    cudaFree(d_logits);
    cudaFree(d_weights);
    cudaFree(d_indices);
    if (d_bias) cudaFree(d_bias);
}

// ── Fixture ─────────────────────────────────────────────────────────────────

class TopkGatingTest : public ::testing::Test {
protected:
    void SetUp() override { gen_.seed(42); }

    void fill_random(std::vector<float>& v, float lo = -2.0f,
                     float hi = 2.0f) {
        std::uniform_real_distribution<float> dist(lo, hi);
        for (auto& x : v) x = dist(gen_);
    }

    /// Generate logits with well-separated scores so top-K is deterministic.
    /// Logits are centered at 0 within [-range/2, range/2] (default range=4.0)
    /// to stay in the sigmoid-sensitive zone and avoid saturation.
    /// Noise amplitude is 1% of the step to preserve ordering.
    void fill_separated(std::vector<float>& logits, int num_tokens,
                        int num_experts, float range = 4.0f) {
        logits.resize(static_cast<size_t>(num_tokens) * num_experts);
        const float step = range / num_experts;
        const float noise_amp = step * 0.01f;
        std::uniform_real_distribution<float> noise(-noise_amp, noise_amp);
        for (int t = 0; t < num_tokens; ++t) {
            // Permute expert ordering per token for variety.
            std::vector<int> perm(num_experts);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), gen_);
            for (int e = 0; e < num_experts; ++e) {
                float rank = static_cast<float>(perm[e]);
                logits[t * num_experts + e] =
                    (rank - num_experts * 0.5f) * step + noise(gen_);
            }
        }
    }

    std::mt19937 gen_;
};

// ═════════════════════════════════════════════════════════════════════════════
// Simple Top-K Tests (n_group = 1)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(TopkGatingTest, Simple_Small_NoBias) {
    REQUIRES_GPU();

    constexpr int T = 4, E = 16, K = 4;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias;  // unused

    GatingTestCase tc{T, E, K, 1, 1, 2.5f, true, false};
    run_gating_test(tc, logits, bias);
}

TEST_F(TopkGatingTest, Simple_Small_WithBias) {
    REQUIRES_GPU();

    constexpr int T = 4, E = 16, K = 4;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias(E);
    fill_random(bias, -0.1f, 0.1f);

    GatingTestCase tc{T, E, K, 1, 1, 2.5f, true, true};
    run_gating_test(tc, logits, bias);
}

TEST_F(TopkGatingTest, Simple_GLM5_256Experts) {
    REQUIRES_GPU();

    // GLM-5: 256 experts, n_group=1, topk=8, scaling=2.5
    constexpr int T = 8, E = 256, K = 8;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias(E);
    fill_random(bias, -0.05f, 0.05f);

    GatingTestCase tc{T, E, K, 1, 1, 2.5f, true, true};
    run_gating_test(tc, logits, bias);
}

TEST_F(TopkGatingTest, Simple_K25_384Experts) {
    REQUIRES_GPU();

    // Kimi K2.5: 384 experts, n_group=1, topk=8, scaling=2.827
    constexpr int T = 8, E = 384, K = 8;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias(E);
    fill_random(bias, -0.05f, 0.05f);

    GatingTestCase tc{T, E, K, 1, 1, 2.827f, true, true};
    run_gating_test(tc, logits, bias);
}

TEST_F(TopkGatingTest, Simple_NoRenormalize) {
    REQUIRES_GPU();

    constexpr int T = 4, E = 32, K = 4;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias;

    GatingTestCase tc{T, E, K, 1, 1, 2.5f, false, false};
    run_gating_test(tc, logits, bias);
}

TEST_F(TopkGatingTest, Simple_Topk1) {
    REQUIRES_GPU();

    constexpr int T = 4, E = 32, K = 1;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias;

    GatingTestCase tc{T, E, K, 1, 1, 2.5f, true, false};
    run_gating_test(tc, logits, bias);
}

TEST_F(TopkGatingTest, Simple_SingleToken) {
    REQUIRES_GPU();

    constexpr int T = 1, E = 256, K = 8;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias;

    GatingTestCase tc{T, E, K, 1, 1, 2.5f, true, false};
    run_gating_test(tc, logits, bias);
}

TEST_F(TopkGatingTest, Simple_LargeBatch) {
    REQUIRES_GPU();

    constexpr int T = 512, E = 256, K = 8;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias(E);
    fill_random(bias, -0.05f, 0.05f);

    GatingTestCase tc{T, E, K, 1, 1, 2.5f, true, true};
    run_gating_test(tc, logits, bias);
}

// ═════════════════════════════════════════════════════════════════════════════
// Grouped Top-K Tests (n_group > 1)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(TopkGatingTest, Grouped_V32_NoBias) {
    REQUIRES_GPU();

    // V3.2: 256 experts, 8 groups, top-4 groups, top-8 experts
    constexpr int T = 8, E = 256, K = 8;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias;

    GatingTestCase tc{T, E, K, 8, 4, 2.5f, true, false};
    run_gating_test(tc, logits, bias);
}

TEST_F(TopkGatingTest, Grouped_V32_WithBias) {
    REQUIRES_GPU();

    constexpr int T = 8, E = 256, K = 8;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias(E);
    fill_random(bias, -0.05f, 0.05f);

    GatingTestCase tc{T, E, K, 8, 4, 2.5f, true, true};
    run_gating_test(tc, logits, bias);
}

TEST_F(TopkGatingTest, Grouped_Small) {
    REQUIRES_GPU();

    // Small grouped: 32 experts, 4 groups of 8, select top-2 groups, top-4
    constexpr int T = 4, E = 32, K = 4;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias(E);
    fill_random(bias, -0.05f, 0.05f);

    GatingTestCase tc{T, E, K, 4, 2, 2.5f, true, true};
    run_gating_test(tc, logits, bias);
}

TEST_F(TopkGatingTest, Grouped_NoRenormalize) {
    REQUIRES_GPU();

    constexpr int T = 4, E = 256, K = 8;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias;

    GatingTestCase tc{T, E, K, 8, 4, 2.5f, false, false};
    run_gating_test(tc, logits, bias);
}

TEST_F(TopkGatingTest, Grouped_AllGroups) {
    REQUIRES_GPU();

    // Select all groups (topk_group == n_group): degenerates to simple top-K
    // but through the grouped code path. Results must match simple kernel.
    constexpr int T = 4, E = 64, K = 4;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias;

    GatingTestCase tc{T, E, K, 4, 4, 2.5f, true, false};
    run_gating_test(tc, logits, bias);
}

TEST_F(TopkGatingTest, Grouped_LargeBatch) {
    REQUIRES_GPU();

    constexpr int T = 256, E = 256, K = 8;
    std::vector<float> logits;
    fill_separated(logits, T, E);
    std::vector<float> bias(E);
    fill_random(bias, -0.05f, 0.05f);

    GatingTestCase tc{T, E, K, 8, 4, 2.5f, true, true};
    run_gating_test(tc, logits, bias);
}

// ═════════════════════════════════════════════════════════════════════════════
// Validation / edge case tests
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(TopkGatingTest, ZeroTokens) {
    REQUIRES_GPU();

    // Should not crash.
    lc::TopkGatingParams params{};
    params.num_tokens = 0;
    params.num_experts = 256;
    params.topk = 8;
    params.n_group = 1;
    params.topk_group = 1;
    params.routed_scaling_factor = 2.5f;
    params.renormalize = true;

    lc::launch_topk_gating(nullptr, nullptr, nullptr, nullptr, params,
                           nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
}

TEST_F(TopkGatingTest, InvalidParams) {
    REQUIRES_GPU();

    lc::TopkGatingParams params{};
    params.num_tokens = 1;
    params.num_experts = 0;
    params.topk = 1;
    params.n_group = 1;
    params.topk_group = 1;

    EXPECT_THROW(lc::launch_topk_gating(nullptr, nullptr, nullptr, nullptr,
                                         params, nullptr),
                 std::invalid_argument);

    params.num_experts = 256;
    params.topk = 0;
    EXPECT_THROW(lc::launch_topk_gating(nullptr, nullptr, nullptr, nullptr,
                                         params, nullptr),
                 std::invalid_argument);

    params.topk = 8;
    params.n_group = 3;  // 256 % 3 != 0
    EXPECT_THROW(lc::launch_topk_gating(nullptr, nullptr, nullptr, nullptr,
                                         params, nullptr),
                 std::invalid_argument);
}

TEST_F(TopkGatingTest, RenormalizeWeightsSum) {
    REQUIRES_GPU();

    // Verify that renormalized weights sum to routed_scaling_factor.
    constexpr int T = 16, E = 64, K = 4;
    constexpr float scaling = 3.14f;
    std::vector<float> logits;
    fill_separated(logits, T, E);

    float *d_logits, *d_weights;
    int32_t* d_indices;
    CUDA_CHECK(cudaMalloc(&d_logits, T * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, T * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, T * K * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), T * E * sizeof(float),
                          cudaMemcpyHostToDevice));

    lc::TopkGatingParams params{T, E, K, 1, 1, scaling, true};
    lc::launch_topk_gating(d_weights, d_indices, d_logits, nullptr, params,
                           nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_weights(T * K);
    CUDA_CHECK(cudaMemcpy(h_weights.data(), d_weights, T * K * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += h_weights[t * K + k];
        ASSERT_NEAR(sum, scaling, 1e-4f) << "Token " << t << " weight sum";
    }

    cudaFree(d_logits);
    cudaFree(d_weights);
    cudaFree(d_indices);
}

TEST_F(TopkGatingTest, BiasOnlyAffectsSelection) {
    REQUIRES_GPU();

    // With large bias on expert 0, it should always be selected.
    // But its weight should still be sigmoid(logit[0]), not sigmoid+bias.
    constexpr int T = 1, E = 32, K = 1;

    std::vector<float> logits(E, 0.0f);  // all equal
    logits[0] = -1.0f;                   // expert 0 has lowest logit

    std::vector<float> bias(E, 0.0f);
    bias[0] = 100.0f;  // huge bias forces selection of expert 0

    float *d_logits, *d_bias, *d_weights;
    int32_t* d_indices;
    CUDA_CHECK(cudaMalloc(&d_logits, E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, K * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), E * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), E * sizeof(float),
                          cudaMemcpyHostToDevice));

    lc::TopkGatingParams params{T, E, K, 1, 1, 1.0f, false};
    lc::launch_topk_gating(d_weights, d_indices, d_logits, d_bias, params,
                           nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    int32_t h_idx;
    float h_weight;
    CUDA_CHECK(
        cudaMemcpy(&h_idx, d_indices, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(&h_weight, d_weights, sizeof(float), cudaMemcpyDeviceToHost));

    // Expert 0 must be selected (bias forces it).
    ASSERT_EQ(h_idx, 0);

    // Weight must be sigmoid(-1.0), NOT sigmoid(-1.0) + 100.
    float expected = sigmoid_ref(-1.0f);
    ASSERT_NEAR(h_weight, expected, 1e-5f)
        << "Weight should be unbiased sigmoid";

    cudaFree(d_logits);
    cudaFree(d_bias);
    cudaFree(d_weights);
    cudaFree(d_indices);
}

// ── V4-4a: sqrtsoftplus scoring (DeepSeek V4) ───────────────────────────────
//
// Reference: llama.cpp build_moe_ffn LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS
// — probs = sqrt(softplus(logits)); selection uses probs + exp_probs_b bias;
// routing weights use UNBIASED probs, renormalized to routed_scaling (1.5).

static float sqrtsoftplus_ref(float x) {
    const float sp = (x > 20.0f) ? x + std::log(1.0f + std::exp(-x))
                                 : std::log(1.0f + std::exp(x));
    return std::sqrt(sp);
}

TEST_F(TopkGatingTest, Simple_V4_Sqrtsoftplus_256E_Top6_Bias) {
    REQUIRES_GPU();

    const int T = 33, E = 256, K = 6;
    const float rsf = 1.5f;

    std::mt19937 rng(4242);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::vector<float> h_logits(static_cast<size_t>(T) * E);
    for (auto& v : h_logits) v = dist(rng);
    std::vector<float> h_bias(E);
    for (auto& v : h_bias) v = dist(rng) * 0.1f;

    // CPU reference: top-6 by biased sqrtsoftplus; weights unbiased + renorm.
    std::vector<float> ref_weights(static_cast<size_t>(T) * K);
    std::vector<int32_t> ref_indices(static_cast<size_t>(T) * K);
    for (int t = 0; t < T; ++t) {
        const float* lg = h_logits.data() + static_cast<size_t>(t) * E;
        std::vector<float> score(E), selv(E);
        for (int e = 0; e < E; ++e) {
            score[e] = sqrtsoftplus_ref(lg[e]);
            selv[e] = score[e] + h_bias[e];
        }
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + K, order.end(),
                          [&](int a, int b) { return selv[a] > selv[b]; });
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += score[order[k]];
        for (int k = 0; k < K; ++k) {
            ref_indices[static_cast<size_t>(t) * K + k] = order[k];
            ref_weights[static_cast<size_t>(t) * K + k] =
                score[order[k]] * (sum > 0.0f ? rsf / sum : 1.0f);
        }
    }

    float *d_logits, *d_bias, *d_weights;
    int32_t* d_indices;
    CUDA_CHECK(cudaMalloc(&d_logits, h_logits.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights,
                          static_cast<size_t>(T) * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices,
                          static_cast<size_t>(T) * K * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(),
                          h_logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(),
                          h_bias.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    lc::TopkGatingParams params{};
    params.num_tokens = T;
    params.num_experts = E;
    params.topk = K;
    params.n_group = 1;
    params.topk_group = 1;
    params.routed_scaling_factor = rsf;
    params.renormalize = true;
    params.scoring_func = lc::ScoringFunc::kSqrtSoftplus;

    lc::launch_topk_gating(d_weights, d_indices, d_logits, d_bias, params,
                           nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_weights(static_cast<size_t>(T) * K);
    std::vector<int32_t> gpu_indices(static_cast<size_t>(T) * K);
    CUDA_CHECK(cudaMemcpy(gpu_weights.data(), d_weights,
                          gpu_weights.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_indices.data(), d_indices,
                          gpu_indices.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    for (int t = 0; t < T; ++t) {
        std::set<int> ref_set, gpu_set;
        for (int k = 0; k < K; ++k) {
            ref_set.insert(ref_indices[static_cast<size_t>(t) * K + k]);
            gpu_set.insert(gpu_indices[static_cast<size_t>(t) * K + k]);
        }
        ASSERT_EQ(ref_set, gpu_set)
            << "Token " << t << ": sqrtsoftplus expert sets differ";
        float wsum = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int gpu_eid = gpu_indices[static_cast<size_t>(t) * K + k];
            float ref_w = -1.0f;
            for (int j = 0; j < K; ++j)
                if (ref_indices[static_cast<size_t>(t) * K + j] == gpu_eid)
                    ref_w = ref_weights[static_cast<size_t>(t) * K + j];
            ASSERT_NEAR(gpu_weights[static_cast<size_t>(t) * K + k], ref_w,
                        2e-4f)
                << "Token " << t << " expert " << gpu_eid;
            wsum += gpu_weights[static_cast<size_t>(t) * K + k];
        }
        // noaux_tc composition: renormalized weights sum to routed_scaling 1.5.
        ASSERT_NEAR(wsum, rsf, 1e-3f) << "Token " << t;
    }

    cudaFree(d_logits);
    cudaFree(d_bias);
    cudaFree(d_weights);
    cudaFree(d_indices);
}
