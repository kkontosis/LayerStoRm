// GPU tests for the sampling kernel (IPC-8f).
//
// Validates argmax, top-K, top-P, and combined sampling correctness
// against CPU reference implementations.  Requires a CUDA GPU.

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "compute/kernels/sampling/sampling.h"

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

/// CPU argmax reference.
static int32_t cpu_argmax(const float* logits, int vocab_size) {
    int32_t best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

// ── Test fixture ───────────────────────────────────────────────────────────

class SamplingKernelTest : public ::testing::Test {
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

// ── Argmax tests ───────────────────────────────────────────────────────────

TEST_F(SamplingKernelTest, ArgmaxSingleToken) {
    constexpr int V = 1024;
    std::vector<float> logits(V, -1.0f);
    logits[42] = 10.0f;  // Known max.

    auto d_logits = alloc_device(V * sizeof(float));
    auto d_ids    = alloc_device(sizeof(int32_t));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                       V * sizeof(float), cudaMemcpyHostToDevice));

    lcomp::launch_sample_tokens(
        static_cast<int32_t*>(d_ids.ptr),
        static_cast<float*>(d_logits.ptr),
        /*num_tokens=*/1, /*vocab_size=*/V,
        /*temperature=*/0.0f, /*top_p=*/1.0f, /*top_k=*/0,
        /*seed=*/0, stream_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    int32_t result = -1;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&result, d_ids.ptr, sizeof(int32_t),
                                       cudaMemcpyDeviceToHost));
    EXPECT_EQ(result, 42);
}

TEST_F(SamplingKernelTest, ArgmaxBatch) {
    constexpr int N = 16;
    constexpr int V = 512;
    std::vector<float> logits(N * V, -1.0f);
    // Each token has a unique argmax.
    for (int t = 0; t < N; ++t) {
        int target = (t * 37 + 7) % V;  // Pseudorandom positions.
        logits[t * V + target] = 100.0f;
    }

    auto d_logits = alloc_device(N * V * sizeof(float));
    auto d_ids    = alloc_device(N * sizeof(int32_t));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                       N * V * sizeof(float), cudaMemcpyHostToDevice));

    lcomp::launch_sample_tokens(
        static_cast<int32_t*>(d_ids.ptr),
        static_cast<float*>(d_logits.ptr),
        N, V, /*temperature=*/0.0f, /*top_p=*/1.0f, /*top_k=*/0,
        /*seed=*/0, stream_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    std::vector<int32_t> results(N);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(results.data(), d_ids.ptr,
                                       N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    for (int t = 0; t < N; ++t) {
        int expected = (t * 37 + 7) % V;
        EXPECT_EQ(results[t], expected) << "token " << t;
    }
}

TEST_F(SamplingKernelTest, ArgmaxLargeVocab) {
    constexpr int V = 129280;  // DeepSeek V3.2 vocab size.
    std::vector<float> logits(V);

    // Fill with a gradient so max is at a known position.
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (int i = 0; i < V; ++i) logits[i] = dist(rng);

    // Plant the max at a specific index.
    constexpr int target = 100000;
    logits[target] = 999.0f;

    auto d_logits = alloc_device(V * sizeof(float));
    auto d_ids    = alloc_device(sizeof(int32_t));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                       V * sizeof(float), cudaMemcpyHostToDevice));

    lcomp::launch_sample_tokens(
        static_cast<int32_t*>(d_ids.ptr),
        static_cast<float*>(d_logits.ptr),
        1, V, /*temperature=*/0.0f, /*top_p=*/1.0f, /*top_k=*/0,
        /*seed=*/0, stream_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    int32_t result = -1;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&result, d_ids.ptr, sizeof(int32_t),
                                       cudaMemcpyDeviceToHost));
    EXPECT_EQ(result, target);
}

// ── Top-K tests ────────────────────────────────────────────────────────────

TEST_F(SamplingKernelTest, TopKCorrectness) {
    // Only K=5 elements are non-negative-infinity; rest are -inf.
    // Sampled token must be among the 5.
    constexpr int V = 1024;
    constexpr int K = 5;
    std::vector<float> logits(V, -1e30f);
    int valid_indices[K] = {10, 200, 500, 700, 999};
    for (int i = 0; i < K; ++i) {
        logits[valid_indices[i]] = static_cast<float>(i + 1);
    }

    auto d_logits = alloc_device(V * sizeof(float));
    auto d_ids    = alloc_device(sizeof(int32_t));

    // Run 20 trials with different seeds to check sampling variety.
    for (uint64_t seed = 0; seed < 20; ++seed) {
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                           V * sizeof(float), cudaMemcpyHostToDevice));

        lcomp::launch_sample_tokens(
            static_cast<int32_t*>(d_ids.ptr),
            static_cast<float*>(d_logits.ptr),
            1, V, /*temperature=*/1.0f, /*top_p=*/1.0f, /*top_k=*/K,
            seed, stream_);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

        int32_t result = -1;
        ASSERT_EQ(cudaSuccess, cudaMemcpy(&result, d_ids.ptr, sizeof(int32_t),
                                           cudaMemcpyDeviceToHost));

        bool found = false;
        for (int i = 0; i < K; ++i) {
            if (result == valid_indices[i]) { found = true; break; }
        }
        EXPECT_TRUE(found) << "seed=" << seed << " result=" << result
                           << " not in valid set";
    }
}

// ── Top-P tests ────────────────────────────────────────────────────────────

TEST_F(SamplingKernelTest, TopPCorrectness) {
    // Create logits where one element dominates (prob > 0.99 after softmax).
    constexpr int V = 256;
    std::vector<float> logits(V, 0.0f);
    logits[77] = 20.0f;  // After softmax with temperature=1, this dominates.

    auto d_logits = alloc_device(V * sizeof(float));
    auto d_ids    = alloc_device(sizeof(int32_t));

    // With top_p=0.5, the dominant element should be selected almost always.
    int dominant_count = 0;
    constexpr int trials = 50;
    for (uint64_t seed = 0; seed < trials; ++seed) {
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                           V * sizeof(float), cudaMemcpyHostToDevice));

        lcomp::launch_sample_tokens(
            static_cast<int32_t*>(d_ids.ptr),
            static_cast<float*>(d_logits.ptr),
            1, V, /*temperature=*/1.0f, /*top_p=*/0.5f, /*top_k=*/0,
            seed, stream_);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

        int32_t result = -1;
        ASSERT_EQ(cudaSuccess, cudaMemcpy(&result, d_ids.ptr, sizeof(int32_t),
                                           cudaMemcpyDeviceToHost));
        if (result == 77) ++dominant_count;
    }

    // With a dominant element at logit=20, it should be selected > 90% of the time.
    EXPECT_GE(dominant_count, static_cast<int>(trials * 0.9))
        << "dominant_count=" << dominant_count << " out of " << trials;
}

// ── Combined top-K + top-P test ────────────────────────────────────────────

TEST_F(SamplingKernelTest, TopKTopPCombined) {
    // 10 elements with positive logits, rest are -inf.
    // top_k=10, top_p=0.3 should further restrict to the top few.
    constexpr int V = 512;
    constexpr int K = 10;
    std::vector<float> logits(V, -1e30f);

    // Create a clear ranking among the K candidates.
    for (int i = 0; i < K; ++i) {
        logits[i * 50] = static_cast<float>(K - i) * 5.0f;  // 50, 45, 40, ...
    }

    auto d_logits = alloc_device(V * sizeof(float));
    auto d_ids    = alloc_device(sizeof(int32_t));

    // With top_p=0.3, only the top ~1-2 candidates should survive.
    int top1_count = 0;
    constexpr int trials = 30;
    for (uint64_t seed = 100; seed < 100 + trials; ++seed) {
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                           V * sizeof(float), cudaMemcpyHostToDevice));

        lcomp::launch_sample_tokens(
            static_cast<int32_t*>(d_ids.ptr),
            static_cast<float*>(d_logits.ptr),
            1, V, /*temperature=*/1.0f, /*top_p=*/0.3f, /*top_k=*/K,
            seed, stream_);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

        int32_t result = -1;
        ASSERT_EQ(cudaSuccess, cudaMemcpy(&result, d_ids.ptr, sizeof(int32_t),
                                           cudaMemcpyDeviceToHost));

        // Result must be one of the K valid candidates.
        bool found = false;
        for (int i = 0; i < K; ++i) {
            if (result == i * 50) { found = true; break; }
        }
        EXPECT_TRUE(found) << "seed=" << seed << " result=" << result;
        if (result == 0) ++top1_count;  // Index 0 has the highest logit.
    }

    // The top candidate should dominate with top_p=0.3.
    EXPECT_GE(top1_count, static_cast<int>(trials * 0.5))
        << "top1_count=" << top1_count;
}

// ── Temperature=0 equals argmax ────────────────────────────────────────────

TEST_F(SamplingKernelTest, TemperatureZeroEqualsArgmax) {
    constexpr int V = 2048;
    std::vector<float> logits(V);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (int i = 0; i < V; ++i) logits[i] = dist(rng);

    int32_t expected = cpu_argmax(logits.data(), V);

    auto d_logits = alloc_device(V * sizeof(float));
    auto d_ids    = alloc_device(sizeof(int32_t));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                       V * sizeof(float), cudaMemcpyHostToDevice));

    lcomp::launch_sample_tokens(
        static_cast<int32_t*>(d_ids.ptr),
        static_cast<float*>(d_logits.ptr),
        1, V, /*temperature=*/0.0f, /*top_p=*/1.0f, /*top_k=*/0,
        /*seed=*/0, stream_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    int32_t result = -1;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&result, d_ids.ptr, sizeof(int32_t),
                                       cudaMemcpyDeviceToHost));
    EXPECT_EQ(result, expected);
}

// ── Philox reproducibility ─────────────────────────────────────────────────

TEST_F(SamplingKernelTest, PhiloxReproducibility) {
    constexpr int V = 256;
    // Uniform logits so result depends entirely on RNG.
    std::vector<float> logits(V, 0.0f);

    auto d_logits = alloc_device(V * sizeof(float));
    auto d_ids    = alloc_device(sizeof(int32_t));

    // Run twice with same seed — must produce same result.
    int32_t result_a = -1, result_b = -1;
    for (int run = 0; run < 2; ++run) {
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                           V * sizeof(float), cudaMemcpyHostToDevice));
        lcomp::launch_sample_tokens(
            static_cast<int32_t*>(d_ids.ptr),
            static_cast<float*>(d_logits.ptr),
            1, V, /*temperature=*/1.0f, /*top_p=*/1.0f, /*top_k=*/0,
            /*seed=*/999, stream_);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

        int32_t result = -1;
        ASSERT_EQ(cudaSuccess, cudaMemcpy(&result, d_ids.ptr, sizeof(int32_t),
                                           cudaMemcpyDeviceToHost));
        if (run == 0) result_a = result;
        else result_b = result;
    }
    EXPECT_EQ(result_a, result_b) << "Same seed should produce same result";

    // Run with different seed — should produce different result (probabilistic,
    // but with 256 vocab and uniform logits the chance of collision is ~0.4%).
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_logits.ptr, logits.data(),
                                       V * sizeof(float), cudaMemcpyHostToDevice));
    lcomp::launch_sample_tokens(
        static_cast<int32_t*>(d_ids.ptr),
        static_cast<float*>(d_logits.ptr),
        1, V, /*temperature=*/1.0f, /*top_p=*/1.0f, /*top_k=*/0,
        /*seed=*/12345, stream_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    int32_t result_c = -1;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&result_c, d_ids.ptr, sizeof(int32_t),
                                       cudaMemcpyDeviceToHost));
    // Not a hard requirement, but very likely with different seeds.
    // If this fails, it's a statistical fluke — rerun.
    EXPECT_NE(result_a, result_c)
        << "Different seeds should (almost certainly) produce different results";
}
