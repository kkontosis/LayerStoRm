// Unit tests for the V4 hash-layer gating kernel (V4-4).
//
// Reference semantics (ref/llama.cpp deepseek4.cpp:1127-1132 +
// llama-graph.cpp build_moe_ffn with selected_experts_in):
//   probs      = scoring_fn(router_logits) over ALL experts
//   selected_k = tid2eid[token_id * topk + k]  (TABLE order)
//   weights_k  = probs[selected_k], renormalized to routed_scaling (1.5)
//   exp_probs_b bias NEVER applies on hash layers.

#include "compute/kernels/moe/hash_gating.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
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

namespace {

float sigmoid_ref(float x) { return 0.5f * std::tanh(0.5f * x) + 0.5f; }

float sqrtsoftplus_ref(float x) {
    const float sp = (x > 20.0f) ? x + std::log(1.0f + std::exp(-x))
                                 : std::log(1.0f + std::exp(x));
    return std::sqrt(sp);
}

struct HashCase {
    int T = 17;
    int E = 256;
    int K = 6;
    int vocab = 1000;
    float rsf = 1.5f;
    bool renormalize = true;
    lc::ScoringFunc fn = lc::ScoringFunc::kSqrtSoftplus;
    uint32_t seed = 99;
};

void run_hash_case(const HashCase& tc) {
    const int T = tc.T, E = tc.E, K = tc.K;

    std::mt19937 rng(tc.seed);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::uniform_int_distribution<int32_t> vdist(0, tc.vocab - 1);
    std::uniform_int_distribution<int32_t> edist(0, E - 1);

    std::vector<float> h_logits(static_cast<size_t>(T) * E);
    for (auto& v : h_logits) v = dist(rng);
    std::vector<int32_t> h_table(static_cast<size_t>(tc.vocab) * K);
    for (auto& v : h_table) v = edist(rng);
    std::vector<int32_t> h_tokens(T);
    for (auto& v : h_tokens) v = vdist(rng);

    // CPU reference.
    std::vector<float> ref_w(static_cast<size_t>(T) * K);
    std::vector<int32_t> ref_i(static_cast<size_t>(T) * K);
    for (int t = 0; t < T; ++t) {
        const float* lg = h_logits.data() + static_cast<size_t>(t) * E;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int32_t e =
                h_table[static_cast<size_t>(h_tokens[t]) * K + k];
            ref_i[static_cast<size_t>(t) * K + k] = e;
            const float s = (tc.fn == lc::ScoringFunc::kSigmoid)
                                ? sigmoid_ref(lg[e])
                                : sqrtsoftplus_ref(lg[e]);
            ref_w[static_cast<size_t>(t) * K + k] = s;
            sum += s;
        }
        if (tc.renormalize && sum > 0.0f) {
            for (int k = 0; k < K; ++k)
                ref_w[static_cast<size_t>(t) * K + k] *= tc.rsf / sum;
        }
    }

    float *d_logits, *d_weights;
    int32_t *d_table, *d_tokens, *d_indices;
    CUDA_CHECK(cudaMalloc(&d_logits, h_logits.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_table, h_table.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_tokens, h_tokens.size() * sizeof(int32_t)));
    CUDA_CHECK(
        cudaMalloc(&d_weights, static_cast<size_t>(T) * K * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_indices, static_cast<size_t>(T) * K * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(),
                          h_logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(),
                          h_table.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(),
                          h_tokens.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    lc::HashGatingParams p{};
    p.num_tokens = T;
    p.num_experts = E;
    p.topk = K;
    p.vocab_size = tc.vocab;
    p.routed_scaling_factor = tc.rsf;
    p.renormalize = tc.renormalize;
    p.scoring_func = tc.fn;

    lc::launch_hash_gating(d_weights, d_indices, d_logits, d_table, d_tokens,
                           p, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_w(static_cast<size_t>(T) * K);
    std::vector<int32_t> gpu_i(static_cast<size_t>(T) * K);
    CUDA_CHECK(cudaMemcpy(gpu_w.data(), d_weights,
                          gpu_w.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_i.data(), d_indices,
                          gpu_i.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    for (int t = 0; t < T; ++t) {
        float wsum = 0.0f;
        for (int k = 0; k < K; ++k) {
            const size_t idx = static_cast<size_t>(t) * K + k;
            // Determinism: expert ids come from the table VERBATIM, in table
            // (selection-rank) order — position-exact, not set-equal.
            ASSERT_EQ(gpu_i[idx], ref_i[idx])
                << "Token " << t << " slot " << k << " expert id mismatch";
            ASSERT_NEAR(gpu_w[idx], ref_w[idx], 2e-4f)
                << "Token " << t << " slot " << k << " weight mismatch";
            wsum += gpu_w[idx];
        }
        if (tc.renormalize)
            ASSERT_NEAR(wsum, tc.rsf, 1e-3f) << "Token " << t;
    }

    cudaFree(d_logits);
    cudaFree(d_table);
    cudaFree(d_tokens);
    cudaFree(d_weights);
    cudaFree(d_indices);
}

}  // namespace

class HashGatingTest : public ::testing::Test {};

TEST_F(HashGatingTest, V4_Sqrtsoftplus_Renorm) {
    REQUIRES_GPU();
    run_hash_case({});
}

TEST_F(HashGatingTest, Sigmoid_Renorm) {
    REQUIRES_GPU();
    HashCase tc;
    tc.fn = lc::ScoringFunc::kSigmoid;
    tc.seed = 5;
    run_hash_case(tc);
}

TEST_F(HashGatingTest, NoRenormalize_RawScores) {
    REQUIRES_GPU();
    HashCase tc;
    tc.renormalize = false;
    tc.seed = 6;
    run_hash_case(tc);
}

TEST_F(HashGatingTest, Deterministic_RepeatedTokenIdsIdenticalRouting) {
    REQUIRES_GPU();
    // Same token id in every row ⇒ identical expert ids across all tokens
    // (hash routing depends only on the token id).
    const int T = 8, E = 64, K = 6, vocab = 50;

    std::vector<float> h_logits(static_cast<size_t>(T) * E);
    std::mt19937 rng(3);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_logits) v = dist(rng);
    std::vector<int32_t> h_table(static_cast<size_t>(vocab) * K);
    std::uniform_int_distribution<int32_t> edist(0, E - 1);
    for (auto& v : h_table) v = edist(rng);
    std::vector<int32_t> h_tokens(T, 42);

    float *d_logits, *d_weights;
    int32_t *d_table, *d_tokens, *d_indices;
    CUDA_CHECK(cudaMalloc(&d_logits, h_logits.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_table, h_table.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_tokens, h_tokens.size() * sizeof(int32_t)));
    CUDA_CHECK(
        cudaMalloc(&d_weights, static_cast<size_t>(T) * K * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_indices, static_cast<size_t>(T) * K * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(),
                          h_logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(),
                          h_table.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(),
                          h_tokens.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    lc::HashGatingParams p{};
    p.num_tokens = T;
    p.num_experts = E;
    p.topk = K;
    p.vocab_size = vocab;
    p.routed_scaling_factor = 1.5f;
    p.renormalize = true;
    p.scoring_func = lc::ScoringFunc::kSqrtSoftplus;
    lc::launch_hash_gating(d_weights, d_indices, d_logits, d_table, d_tokens,
                           p, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> gpu_i(static_cast<size_t>(T) * K);
    CUDA_CHECK(cudaMemcpy(gpu_i.data(), d_indices,
                          gpu_i.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k)
            ASSERT_EQ(gpu_i[static_cast<size_t>(t) * K + k],
                      h_table[static_cast<size_t>(42) * K + k])
                << "Token " << t << " slot " << k;

    cudaFree(d_logits);
    cudaFree(d_table);
    cudaFree(d_tokens);
    cudaFree(d_weights);
    cudaFree(d_indices);
}

TEST_F(HashGatingTest, OutOfRangeTokenIdDropsSlots) {
    REQUIRES_GPU();
    // Token id >= vocab ⇒ every slot -1 (permute drop sentinel), weight 0.
    const int T = 2, E = 16, K = 4, vocab = 10;
    std::vector<float> h_logits(static_cast<size_t>(T) * E, 1.0f);
    std::vector<int32_t> h_table(static_cast<size_t>(vocab) * K, 3);
    std::vector<int32_t> h_tokens = {5, 10};  // second is out of range

    float *d_logits, *d_weights;
    int32_t *d_table, *d_tokens, *d_indices;
    CUDA_CHECK(cudaMalloc(&d_logits, h_logits.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_table, h_table.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_tokens, h_tokens.size() * sizeof(int32_t)));
    CUDA_CHECK(
        cudaMalloc(&d_weights, static_cast<size_t>(T) * K * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_indices, static_cast<size_t>(T) * K * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(),
                          h_logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(),
                          h_table.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(),
                          h_tokens.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    lc::HashGatingParams p{};
    p.num_tokens = T;
    p.num_experts = E;
    p.topk = K;
    p.vocab_size = vocab;
    p.routed_scaling_factor = 1.5f;
    p.renormalize = true;
    p.scoring_func = lc::ScoringFunc::kSigmoid;
    lc::launch_hash_gating(d_weights, d_indices, d_logits, d_table, d_tokens,
                           p, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_w(static_cast<size_t>(T) * K);
    std::vector<int32_t> gpu_i(static_cast<size_t>(T) * K);
    CUDA_CHECK(cudaMemcpy(gpu_w.data(), d_weights,
                          gpu_w.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_i.data(), d_indices,
                          gpu_i.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    for (int k = 0; k < K; ++k) {
        EXPECT_EQ(gpu_i[k], 3) << "in-range token slot " << k;
        EXPECT_EQ(gpu_i[K + k], -1) << "OOB token slot " << k;
        EXPECT_EQ(gpu_w[K + k], 0.0f) << "OOB token weight " << k;
    }

    cudaFree(d_logits);
    cudaFree(d_table);
    cudaFree(d_tokens);
    cudaFree(d_weights);
    cudaFree(d_indices);
}
