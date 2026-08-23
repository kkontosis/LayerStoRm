// Unit tests for MoE token permutation and unpermutation kernels.
//
// Tests run on GPU (REQUIRES_GPU). Uses BF16 data type as the primary path.
// Verifies: correct grouping by expert, round-trip identity, sentinel handling,
// empty experts, and routing weight scaling.

#include "smxx/permute/moe_permute.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: "                         \
                                     << cudaGetErrorString(_err);              \
    } while (0)

static float bf16_to_float(uint16_t bits) {
    uint32_t f = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

static uint16_t float_to_bf16(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

// ── BasicPermutation ───────────────────────────────────────────────────────
// 4 tokens, 4 experts, topk=2. Verify tokens are grouped by expert.

TEST(MoePermuteTest, BasicPermutation) {
    REQUIRES_GPU();

    const int num_tokens = 4;
    const int topk = 2;
    const int hidden_dim = 16;
    const int num_experts = 4;
    const int expanded = num_tokens * topk;

    // Expert assignments: token 0→{0,1}, token 1→{1,2}, token 2→{2,3}, token 3→{0,3}
    std::vector<int32_t> h_topk_indices = {0, 1, 1, 2, 2, 3, 0, 3};

    // Hidden states: each token has hidden_dim values = token_id * 1.0
    std::vector<uint16_t> h_hidden(num_tokens * hidden_dim);
    for (int t = 0; t < num_tokens; t++) {
        for (int d = 0; d < hidden_dim; d++) {
            h_hidden[t * hidden_dim + d] = float_to_bf16(static_cast<float>(t + 1));
        }
    }

    // Allocate device memory
    int32_t *d_topk_indices, *d_expert_offsets, *d_src_to_dest, *d_permuted_idx;
    uint16_t *d_hidden, *d_permuted;
    void* d_workspace;

    size_t ws_size = lc::query_moe_permute_workspace_size(num_tokens, topk, num_experts);

    CUDA_CHECK(cudaMalloc(&d_topk_indices, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_expert_offsets, (num_experts + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_src_to_dest, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_permuted_idx, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_hidden, num_tokens * hidden_dim * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_permuted, expanded * hidden_dim * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_workspace, ws_size));

    CUDA_CHECK(cudaMemcpy(d_topk_indices, h_topk_indices.data(),
                            expanded * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(),
                            num_tokens * hidden_dim * sizeof(uint16_t),
                            cudaMemcpyHostToDevice));

    lc::launch_moe_permute(
        d_permuted, d_expert_offsets, d_src_to_dest, d_permuted_idx,
        d_hidden, d_topk_indices,
        num_tokens, topk, hidden_dim, num_experts,
        2 /*elem_size_bytes*/, d_workspace, nullptr /*default stream*/);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back expert_offsets
    std::vector<int32_t> h_offsets(num_experts + 1);
    CUDA_CHECK(cudaMemcpy(h_offsets.data(), d_expert_offsets,
                            (num_experts + 1) * sizeof(int32_t),
                            cudaMemcpyDeviceToHost));

    // Expert 0: tokens 0 and 3 → count = 2
    // Expert 1: tokens 0 and 1 → count = 2
    // Expert 2: tokens 1 and 2 → count = 2
    // Expert 3: tokens 2 and 3 → count = 2
    EXPECT_EQ(h_offsets[0], 0);
    EXPECT_EQ(h_offsets[num_experts], expanded);  // Total = 8

    // Verify offsets are monotonically non-decreasing
    for (int i = 0; i < num_experts; i++) {
        EXPECT_LE(h_offsets[i], h_offsets[i + 1]);
    }

    // Verify permuted data: read back and check grouping
    std::vector<int32_t> h_permuted_idx(expanded);
    CUDA_CHECK(cudaMemcpy(h_permuted_idx.data(), d_permuted_idx,
                            expanded * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // For each expert range, verify the source rows are correct expert assignments
    for (int e = 0; e < num_experts; e++) {
        for (int i = h_offsets[e]; i < h_offsets[e + 1]; i++) {
            int src_expanded = h_permuted_idx[i];
            int src_token = src_expanded / topk;
            int src_k = src_expanded % topk;
            EXPECT_EQ(h_topk_indices[src_expanded], e)
                << "Expert " << e << " got wrong source at position " << i;
        }
    }

    // Cleanup
    cudaFree(d_topk_indices);
    cudaFree(d_expert_offsets);
    cudaFree(d_src_to_dest);
    cudaFree(d_permuted_idx);
    cudaFree(d_hidden);
    cudaFree(d_permuted);
    cudaFree(d_workspace);
}

// ── RoundTrip ──────────────────────────────────────────────────────────────
// Permute then unpermute with uniform weights should recover original data.

TEST(MoePermuteTest, RoundTrip) {
    REQUIRES_GPU();

    const int num_tokens = 8;
    const int topk = 2;
    const int hidden_dim = 32;
    const int num_experts = 4;
    const int expanded = num_tokens * topk;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> expert_dist(0, num_experts - 1);

    // Random expert assignments
    std::vector<int32_t> h_topk_indices(expanded);
    for (int i = 0; i < expanded; i++) {
        h_topk_indices[i] = expert_dist(rng);
    }

    // Random hidden states
    std::vector<uint16_t> h_hidden(num_tokens * hidden_dim);
    for (auto& v : h_hidden) {
        v = float_to_bf16(static_cast<float>(rng() % 100) / 100.0f);
    }

    // Uniform routing weights: 1/topk for each
    std::vector<float> h_weights(expanded, 1.0f / topk);

    // Allocate
    int32_t *d_topk_indices, *d_expert_offsets, *d_src_to_dest, *d_permuted_idx;
    uint16_t *d_hidden, *d_permuted, *d_output;
    float *d_weights;
    void* d_workspace;

    size_t ws_size = lc::query_moe_permute_workspace_size(num_tokens, topk, num_experts);

    CUDA_CHECK(cudaMalloc(&d_topk_indices, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_expert_offsets, (num_experts + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_src_to_dest, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_permuted_idx, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_hidden, num_tokens * hidden_dim * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_permuted, expanded * hidden_dim * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * hidden_dim * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_weights, expanded * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_workspace, ws_size));

    CUDA_CHECK(cudaMemcpy(d_topk_indices, h_topk_indices.data(),
                            expanded * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(),
                            num_tokens * hidden_dim * sizeof(uint16_t),
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(),
                            expanded * sizeof(float), cudaMemcpyHostToDevice));

    // Permute
    lc::launch_moe_permute(
        d_permuted, d_expert_offsets, d_src_to_dest, d_permuted_idx,
        d_hidden, d_topk_indices,
        num_tokens, topk, hidden_dim, num_experts,
        2, d_workspace, nullptr);

    // For round-trip, the "expert FFN" is identity — just copy permuted as-is.
    // Unpermute with uniform weights: output[i] = sum_k(permuted[dest[i*topk+k]] * 1/topk)
    // Since each token is duplicated topk times, sum = token_value * topk * (1/topk) = token_value
    lc::launch_moe_unpermute(
        d_output, d_permuted, d_weights, d_src_to_dest,
        num_tokens, topk, hidden_dim, 2, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back and compare
    std::vector<uint16_t> h_output(num_tokens * hidden_dim);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                            num_tokens * hidden_dim * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < num_tokens * hidden_dim; i++) {
        float expected = bf16_to_float(h_hidden[i]);
        float actual = bf16_to_float(h_output[i]);
        float err = std::fabs(expected - actual);
        max_err = std::max(max_err, err);
    }
    // BF16 round-trip: expect very small error from float→bf16→float conversions
    EXPECT_LT(max_err, 0.02f) << "Round-trip max error too large";

    cudaFree(d_topk_indices);
    cudaFree(d_expert_offsets);
    cudaFree(d_src_to_dest);
    cudaFree(d_permuted_idx);
    cudaFree(d_hidden);
    cudaFree(d_permuted);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_workspace);
}

// ── SentinelHandling ───────────────────────────────────────────────────────
// Tokens with index=-1 (adaptive gating sentinel) should be excluded.

TEST(MoePermuteTest, SentinelHandling) {
    REQUIRES_GPU();

    const int num_tokens = 4;
    const int topk = 3;
    const int hidden_dim = 8;
    const int num_experts = 4;
    const int expanded = num_tokens * topk;

    // Some entries are -1 (sentinel)
    std::vector<int32_t> h_topk_indices = {
        0, 1, -1,   // token 0: experts 0, 1, sentinel
        2, -1, -1,  // token 1: expert 2, two sentinels
        0, 1, 2,    // token 2: experts 0, 1, 2
        3, -1, -1   // token 3: expert 3, two sentinels
    };

    std::vector<uint16_t> h_hidden(num_tokens * hidden_dim);
    for (int i = 0; i < num_tokens * hidden_dim; i++) {
        h_hidden[i] = float_to_bf16(1.0f);
    }

    int32_t *d_topk_indices, *d_expert_offsets, *d_src_to_dest, *d_permuted_idx;
    uint16_t *d_hidden, *d_permuted;
    void* d_workspace;

    size_t ws_size = lc::query_moe_permute_workspace_size(num_tokens, topk, num_experts);

    CUDA_CHECK(cudaMalloc(&d_topk_indices, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_expert_offsets, (num_experts + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_src_to_dest, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_permuted_idx, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_hidden, num_tokens * hidden_dim * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_permuted, expanded * hidden_dim * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_workspace, ws_size));

    CUDA_CHECK(cudaMemcpy(d_topk_indices, h_topk_indices.data(),
                            expanded * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(),
                            num_tokens * hidden_dim * sizeof(uint16_t),
                            cudaMemcpyHostToDevice));

    lc::launch_moe_permute(
        d_permuted, d_expert_offsets, d_src_to_dest, d_permuted_idx,
        d_hidden, d_topk_indices,
        num_tokens, topk, hidden_dim, num_experts,
        2, d_workspace, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> h_offsets(num_experts + 1);
    CUDA_CHECK(cudaMemcpy(h_offsets.data(), d_expert_offsets,
                            (num_experts + 1) * sizeof(int32_t),
                            cudaMemcpyDeviceToHost));

    // Expected counts: expert 0=2, expert 1=2, expert 2=2, expert 3=1 → total=7
    // Sentinels (4 of them) are remapped to num_experts and sorted to end
    int total_valid = h_offsets[num_experts];
    EXPECT_EQ(total_valid, 7) << "Expected 7 valid expert assignments, got " << total_valid;

    // Verify monotonic
    for (int i = 0; i < num_experts; i++) {
        EXPECT_LE(h_offsets[i], h_offsets[i + 1]);
    }

    cudaFree(d_topk_indices);
    cudaFree(d_expert_offsets);
    cudaFree(d_src_to_dest);
    cudaFree(d_permuted_idx);
    cudaFree(d_hidden);
    cudaFree(d_permuted);
    cudaFree(d_workspace);
}

// ── EmptyExperts ───────────────────────────────────────────────────────────
// Some experts receive 0 tokens.

TEST(MoePermuteTest, EmptyExperts) {
    REQUIRES_GPU();

    const int num_tokens = 4;
    const int topk = 1;
    const int hidden_dim = 8;
    const int num_experts = 8;
    const int expanded = num_tokens * topk;

    // Only use experts 0, 2, 5, 7 — experts 1, 3, 4, 6 are empty
    std::vector<int32_t> h_topk_indices = {0, 2, 5, 7};

    std::vector<uint16_t> h_hidden(num_tokens * hidden_dim);
    for (int i = 0; i < num_tokens * hidden_dim; i++) {
        h_hidden[i] = float_to_bf16(1.0f);
    }

    int32_t *d_topk_indices, *d_expert_offsets, *d_src_to_dest, *d_permuted_idx;
    uint16_t *d_hidden, *d_permuted;
    void* d_workspace;

    size_t ws_size = lc::query_moe_permute_workspace_size(num_tokens, topk, num_experts);

    CUDA_CHECK(cudaMalloc(&d_topk_indices, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_expert_offsets, (num_experts + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_src_to_dest, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_permuted_idx, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_hidden, num_tokens * hidden_dim * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_permuted, expanded * hidden_dim * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_workspace, ws_size));

    CUDA_CHECK(cudaMemcpy(d_topk_indices, h_topk_indices.data(),
                            expanded * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(),
                            num_tokens * hidden_dim * sizeof(uint16_t),
                            cudaMemcpyHostToDevice));

    lc::launch_moe_permute(
        d_permuted, d_expert_offsets, d_src_to_dest, d_permuted_idx,
        d_hidden, d_topk_indices,
        num_tokens, topk, hidden_dim, num_experts,
        2, d_workspace, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> h_offsets(num_experts + 1);
    CUDA_CHECK(cudaMemcpy(h_offsets.data(), d_expert_offsets,
                            (num_experts + 1) * sizeof(int32_t),
                            cudaMemcpyDeviceToHost));

    // Total should be 4
    EXPECT_EQ(h_offsets[num_experts], expanded);

    // Empty experts: offset[i] == offset[i+1]
    EXPECT_EQ(h_offsets[1], h_offsets[1]);  // expert 1 empty
    EXPECT_EQ(h_offsets[3], h_offsets[4]);  // expert 3, 4 empty
    EXPECT_EQ(h_offsets[4], h_offsets[4]);
    EXPECT_EQ(h_offsets[6], h_offsets[6]);  // expert 6 empty

    cudaFree(d_topk_indices);
    cudaFree(d_expert_offsets);
    cudaFree(d_src_to_dest);
    cudaFree(d_permuted_idx);
    cudaFree(d_hidden);
    cudaFree(d_permuted);
    cudaFree(d_workspace);
}

// ── WeightScaling ──────────────────────────────────────────────────────────
// Non-uniform routing weights are correctly applied in unpermute.

TEST(MoePermuteTest, WeightScaling) {
    REQUIRES_GPU();

    const int num_tokens = 2;
    const int topk = 2;
    const int hidden_dim = 8;
    const int num_experts = 4;
    const int expanded = num_tokens * topk;

    // Token 0 → experts {0, 1}, Token 1 → experts {2, 3}
    std::vector<int32_t> h_topk_indices = {0, 1, 2, 3};

    // Hidden states: all 1.0
    std::vector<uint16_t> h_hidden(num_tokens * hidden_dim);
    for (auto& v : h_hidden) v = float_to_bf16(1.0f);

    // Weights: token 0 gets {0.7, 0.3}, token 1 gets {0.6, 0.4}
    std::vector<float> h_weights = {0.7f, 0.3f, 0.6f, 0.4f};

    int32_t *d_topk_indices, *d_expert_offsets, *d_src_to_dest, *d_permuted_idx;
    uint16_t *d_hidden, *d_permuted, *d_output;
    float *d_weights;
    void* d_workspace;

    size_t ws_size = lc::query_moe_permute_workspace_size(num_tokens, topk, num_experts);

    CUDA_CHECK(cudaMalloc(&d_topk_indices, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_expert_offsets, (num_experts + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_src_to_dest, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_permuted_idx, expanded * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_hidden, num_tokens * hidden_dim * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_permuted, expanded * hidden_dim * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * hidden_dim * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_weights, expanded * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_workspace, ws_size));

    CUDA_CHECK(cudaMemcpy(d_topk_indices, h_topk_indices.data(),
                            expanded * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(),
                            num_tokens * hidden_dim * sizeof(uint16_t),
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(),
                            expanded * sizeof(float), cudaMemcpyHostToDevice));

    lc::launch_moe_permute(
        d_permuted, d_expert_offsets, d_src_to_dest, d_permuted_idx,
        d_hidden, d_topk_indices,
        num_tokens, topk, hidden_dim, num_experts,
        2, d_workspace, nullptr);

    // Identity FFN: permuted output = permuted input
    lc::launch_moe_unpermute(
        d_output, d_permuted, d_weights, d_src_to_dest,
        num_tokens, topk, hidden_dim, 2, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_output(num_tokens * hidden_dim);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                            num_tokens * hidden_dim * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost));

    // Token 0: output = 1.0 * 0.7 + 1.0 * 0.3 = 1.0
    // Token 1: output = 1.0 * 0.6 + 1.0 * 0.4 = 1.0
    for (int t = 0; t < num_tokens; t++) {
        float expected = (t == 0) ? (0.7f + 0.3f) : (0.6f + 0.4f);
        for (int d = 0; d < hidden_dim; d++) {
            float actual = bf16_to_float(h_output[t * hidden_dim + d]);
            EXPECT_NEAR(actual, expected, 0.02f)
                << "Token " << t << ", dim " << d;
        }
    }

    cudaFree(d_topk_indices);
    cudaFree(d_expert_offsets);
    cudaFree(d_src_to_dest);
    cudaFree(d_permuted_idx);
    cudaFree(d_hidden);
    cudaFree(d_permuted);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_workspace);
}
