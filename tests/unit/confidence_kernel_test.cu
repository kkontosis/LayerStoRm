// GPU tests for the confidence estimation kernel (IPC-8g).
//
// Validates top-1 probability and normalized entropy against CPU
// reference implementations.  Requires a CUDA GPU.

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "compute/kernels/confidence/confidence.h"

namespace lcomp = layerstorm::compute;

// ── Helpers ────────────────────────────────────────────────────────────────

/// RAII wrapper for device memory.
struct DeviceBuf {
    void* ptr = nullptr;
    ~DeviceBuf() { if (ptr) cudaFree(ptr); }
};

static DeviceBuf alloc_device(size_t bytes) {
    DeviceBuf buf;
    EXPECT_EQ(cudaSuccess, cudaMalloc(&buf.ptr, bytes));
    return buf;
}

/// CPU reference: compute softmax top-1 probability and normalized entropy.
static void cpu_confidence(const float* logits, int vocab_size,
                           float* top1_prob, float* entropy) {
    // Find max for numerical stability.
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        max_val = std::max(max_val, logits[i]);
    }

    // Compute exp and sum.
    float sum = 0.0f;
    float max_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        float e = std::exp(logits[i] - max_val);
        sum += e;
        max_exp = std::max(max_exp, e);
    }

    // Compute entropy.
    float inv_sum = 1.0f / sum;
    float ent = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        float p = std::exp(logits[i] - max_val) * inv_sum;
        if (p > 0.0f) {
            ent -= p * std::log(p);
        }
    }

    *top1_prob = max_exp / sum;
    *entropy = ent / std::log(static_cast<float>(vocab_size));
}

// ── Test fixture ──────────────────────────────────────────────────────────

class ConfidenceKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        REQUIRES_GPU();
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_));
    }

    void TearDown() override {
        if (stream_) cudaStreamDestroy(stream_);
    }

    cudaStream_t stream_ = nullptr;
};

// ── Uniform logits: max entropy ───────────────────────────────────────────

TEST_F(ConfidenceKernelTest, UniformLogits) {
    constexpr int V = 1024;
    std::vector<float> logits(V, 0.0f);  // All equal.

    auto d_logits = alloc_device(V * sizeof(float));
    auto d_top1   = alloc_device(sizeof(float));
    auto d_ent    = alloc_device(sizeof(float));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                       V * sizeof(float), cudaMemcpyHostToDevice));

    lcomp::launch_compute_confidence(
        static_cast<const float*>(d_logits.ptr),
        static_cast<float*>(d_top1.ptr),
        static_cast<float*>(d_ent.ptr),
        /*num_tokens=*/1, /*vocab_size=*/V, stream_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    float top1 = -1.0f, ent = -1.0f;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&top1, d_top1.ptr, sizeof(float),
                                       cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&ent, d_ent.ptr, sizeof(float),
                                       cudaMemcpyDeviceToHost));

    // Uniform distribution: top1 = 1/V, entropy = 1.0 (max entropy).
    EXPECT_NEAR(top1, 1.0f / V, 1e-5f);
    EXPECT_NEAR(ent, 1.0f, 1e-4f);
}

// ── One-hot logits: zero entropy ──────────────────────────────────────────

TEST_F(ConfidenceKernelTest, OneHot) {
    constexpr int V = 1024;
    std::vector<float> logits(V, 0.0f);
    logits[42] = 100.0f;  // Dominant logit.

    auto d_logits = alloc_device(V * sizeof(float));
    auto d_top1   = alloc_device(sizeof(float));
    auto d_ent    = alloc_device(sizeof(float));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                       V * sizeof(float), cudaMemcpyHostToDevice));

    lcomp::launch_compute_confidence(
        static_cast<const float*>(d_logits.ptr),
        static_cast<float*>(d_top1.ptr),
        static_cast<float*>(d_ent.ptr),
        1, V, stream_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    float top1 = -1.0f, ent = -1.0f;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&top1, d_top1.ptr, sizeof(float),
                                       cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&ent, d_ent.ptr, sizeof(float),
                                       cudaMemcpyDeviceToHost));

    // Near-one-hot: top1 ~1.0, entropy ~0.0.
    EXPECT_GT(top1, 0.99f);
    EXPECT_LT(ent, 0.01f);
}

// ── Batch tokens: each row matches CPU reference ──────────────────────────

TEST_F(ConfidenceKernelTest, BatchTokens) {
    constexpr int N = 8;
    constexpr int V = 512;
    std::vector<float> logits(N * V);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (auto& v : logits) v = dist(rng);

    auto d_logits = alloc_device(N * V * sizeof(float));
    auto d_top1   = alloc_device(N * sizeof(float));
    auto d_ent    = alloc_device(N * sizeof(float));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                       N * V * sizeof(float), cudaMemcpyHostToDevice));

    lcomp::launch_compute_confidence(
        static_cast<const float*>(d_logits.ptr),
        static_cast<float*>(d_top1.ptr),
        static_cast<float*>(d_ent.ptr),
        N, V, stream_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    std::vector<float> top1(N), ent(N);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(top1.data(), d_top1.ptr,
                                       N * sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(ent.data(), d_ent.ptr,
                                       N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int t = 0; t < N; ++t) {
        float ref_top1, ref_ent;
        cpu_confidence(logits.data() + t * V, V, &ref_top1, &ref_ent);
        EXPECT_NEAR(top1[t], ref_top1, 1e-4f) << "token " << t << " top1_prob";
        EXPECT_NEAR(ent[t], ref_ent, 1e-4f) << "token " << t << " entropy";
    }
}

// ── Large vocab (DeepSeek V3.2 size) ──────────────────────────────────────

TEST_F(ConfidenceKernelTest, LargeVocab) {
    constexpr int V = 129280;
    std::vector<float> logits(V);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (auto& v : logits) v = dist(rng);

    auto d_logits = alloc_device(V * sizeof(float));
    auto d_top1   = alloc_device(sizeof(float));
    auto d_ent    = alloc_device(sizeof(float));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                       V * sizeof(float), cudaMemcpyHostToDevice));

    lcomp::launch_compute_confidence(
        static_cast<const float*>(d_logits.ptr),
        static_cast<float*>(d_top1.ptr),
        static_cast<float*>(d_ent.ptr),
        1, V, stream_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    float top1 = -1.0f, ent = -1.0f;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&top1, d_top1.ptr, sizeof(float),
                                       cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&ent, d_ent.ptr, sizeof(float),
                                       cudaMemcpyDeviceToHost));

    float ref_top1, ref_ent;
    cpu_confidence(logits.data(), V, &ref_top1, &ref_ent);

    // Wider tolerance for large vocab due to float32 reduction order differences.
    EXPECT_NEAR(top1, ref_top1, 5e-4f) << "top1_prob";
    EXPECT_NEAR(ent, ref_ent, 5e-4f) << "entropy";
}

// ── Logits not modified ───────────────────────────────────────────────────

TEST_F(ConfidenceKernelTest, LogitsNotModified) {
    constexpr int V = 1024;
    std::vector<float> logits(V);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : logits) v = dist(rng);

    std::vector<float> logits_copy = logits;

    auto d_logits = alloc_device(V * sizeof(float));
    auto d_top1   = alloc_device(sizeof(float));
    auto d_ent    = alloc_device(sizeof(float));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                       V * sizeof(float), cudaMemcpyHostToDevice));

    lcomp::launch_compute_confidence(
        static_cast<const float*>(d_logits.ptr),
        static_cast<float*>(d_top1.ptr),
        static_cast<float*>(d_ent.ptr),
        1, V, stream_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    // Read back logits and verify they are unchanged.
    std::vector<float> logits_after(V);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(logits_after.data(), d_logits.ptr,
                                       V * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < V; ++i) {
        EXPECT_EQ(logits_after[i], logits_copy[i]) << "logit[" << i << "] modified";
    }
}
