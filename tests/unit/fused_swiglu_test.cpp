// Unit tests for fused SwiGLU activation kernel.
//
// Tests run on GPU (REQUIRES_GPU). Uses BF16 data type.
// CPU reference: SiLU(gate) * up = gate / (1 + exp(-gate)) * up

#include "smxx/activation/fused_swiglu.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
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

static float silu_ref(float x) {
    return x / (1.0f + std::exp(-x));
}

// ── BasicCorrectness ───────────────────────────────────────────────────────

TEST(FusedSwigluTest, BasicCorrectness) {
    REQUIRES_GPU();

    const int num_tokens = 8;
    const int d = 64;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    // Input: [num_tokens, 2*d] — gate in first d, up in last d
    std::vector<uint16_t> h_input(num_tokens * 2 * d);
    for (auto& v : h_input) v = float_to_bf16(dist(rng));

    // CPU reference
    std::vector<float> h_expected(num_tokens * d);
    for (int t = 0; t < num_tokens; t++) {
        for (int j = 0; j < d; j++) {
            float gate = bf16_to_float(h_input[t * 2 * d + j]);
            float up = bf16_to_float(h_input[t * 2 * d + d + j]);
            h_expected[t * d + j] = silu_ref(gate) * up;
        }
    }

    // Allocate device memory
    uint16_t *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, num_tokens * 2 * d * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * d * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                            h_input.size() * sizeof(uint16_t),
                            cudaMemcpyHostToDevice));

    lc::FusedSwigluParams params;
    params.num_tokens = num_tokens;
    params.d = d;

    lc::launch_fused_swiglu(d_output, d_input, params, 2, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_output(num_tokens * d);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                            h_output.size() * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < num_tokens * d; i++) {
        float actual = bf16_to_float(h_output[i]);
        float expected = h_expected[i];
        float err = std::fabs(actual - expected);
        max_err = std::max(max_err, err);
    }
    EXPECT_LT(max_err, 0.05f) << "SwiGLU max error too large";

    cudaFree(d_input);
    cudaFree(d_output);
}

// ── LargeD (V3.2 intermediate_size = 2048) ─────────────────────────────────

TEST(FusedSwigluTest, LargeD) {
    REQUIRES_GPU();

    const int num_tokens = 4;
    const int d = 2048;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> h_input(num_tokens * 2 * d);
    for (auto& v : h_input) v = float_to_bf16(dist(rng));

    std::vector<float> h_expected(num_tokens * d);
    for (int t = 0; t < num_tokens; t++) {
        for (int j = 0; j < d; j++) {
            float gate = bf16_to_float(h_input[t * 2 * d + j]);
            float up = bf16_to_float(h_input[t * 2 * d + d + j]);
            h_expected[t * d + j] = silu_ref(gate) * up;
        }
    }

    uint16_t *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, num_tokens * 2 * d * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * d * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                            h_input.size() * sizeof(uint16_t),
                            cudaMemcpyHostToDevice));

    lc::FusedSwigluParams params{num_tokens, d};
    lc::launch_fused_swiglu(d_output, d_input, params, 2, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_output(num_tokens * d);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                            h_output.size() * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < num_tokens * d; i++) {
        float actual = bf16_to_float(h_output[i]);
        float expected = h_expected[i];
        max_err = std::max(max_err, std::fabs(actual - expected));
    }
    EXPECT_LT(max_err, 0.05f) << "Large-d SwiGLU max error too large";

    cudaFree(d_input);
    cudaFree(d_output);
}

// ── ZeroInput ──────────────────────────────────────────────────────────────

TEST(FusedSwigluTest, ZeroInput) {
    REQUIRES_GPU();

    const int num_tokens = 4;
    const int d = 32;

    std::vector<uint16_t> h_input(num_tokens * 2 * d, float_to_bf16(0.0f));

    uint16_t *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * d * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                            h_input.size() * sizeof(uint16_t),
                            cudaMemcpyHostToDevice));

    lc::FusedSwigluParams params{num_tokens, d};
    lc::launch_fused_swiglu(d_output, d_input, params, 2, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_output(num_tokens * d);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                            h_output.size() * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost));

    // SiLU(0) = 0, so output should be all zeros
    for (int i = 0; i < num_tokens * d; i++) {
        EXPECT_EQ(bf16_to_float(h_output[i]), 0.0f) << "Zero input produced non-zero at " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

// ── NegativeGate ───────────────────────────────────────────────────────────
// SiLU(-5) ≈ -0.033, so output should be small magnitude.

TEST(FusedSwigluTest, NegativeGate) {
    REQUIRES_GPU();

    const int num_tokens = 2;
    const int d = 16;

    std::vector<uint16_t> h_input(num_tokens * 2 * d);
    // Gate = -5.0, Up = 1.0
    for (int t = 0; t < num_tokens; t++) {
        for (int j = 0; j < d; j++) {
            h_input[t * 2 * d + j] = float_to_bf16(-5.0f);
            h_input[t * 2 * d + d + j] = float_to_bf16(1.0f);
        }
    }

    uint16_t *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * d * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                            h_input.size() * sizeof(uint16_t),
                            cudaMemcpyHostToDevice));

    lc::FusedSwigluParams params{num_tokens, d};
    lc::launch_fused_swiglu(d_output, d_input, params, 2, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_output(num_tokens * d);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                            h_output.size() * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost));

    float expected = silu_ref(-5.0f) * 1.0f;  // ≈ -0.033
    for (int i = 0; i < num_tokens * d; i++) {
        float actual = bf16_to_float(h_output[i]);
        EXPECT_NEAR(actual, expected, 0.02f)
            << "Negative gate mismatch at " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

// ── V3.2 Dimensions ────────────────────────────────────────────────────────

TEST(FusedSwigluTest, V3_2Dimensions) {
    REQUIRES_GPU();

    const int num_tokens = 64;
    const int d = 2048;  // V3.2 moe_intermediate_size

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    std::vector<uint16_t> h_input(num_tokens * 2 * d);
    for (auto& v : h_input) v = float_to_bf16(dist(rng));

    std::vector<float> h_expected(num_tokens * d);
    for (int t = 0; t < num_tokens; t++) {
        for (int j = 0; j < d; j++) {
            float gate = bf16_to_float(h_input[t * 2 * d + j]);
            float up = bf16_to_float(h_input[t * 2 * d + d + j]);
            h_expected[t * d + j] = silu_ref(gate) * up;
        }
    }

    uint16_t *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * d * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                            h_input.size() * sizeof(uint16_t),
                            cudaMemcpyHostToDevice));

    lc::FusedSwigluParams params{num_tokens, d};
    lc::launch_fused_swiglu(d_output, d_input, params, 2, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_output(num_tokens * d);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                            h_output.size() * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < num_tokens * d && errors < 10; i++) {
        float actual = bf16_to_float(h_output[i]);
        float expected = h_expected[i];
        float err = std::fabs(actual - expected);
        if (err > 0.05f) {
            errors++;
            if (errors <= 3) {
                EXPECT_NEAR(actual, expected, 0.05f) << "V3.2 mismatch at " << i;
            }
        }
    }
    EXPECT_EQ(errors, 0) << "V3.2 SwiGLU: " << errors << " elements with >0.05 error";

    cudaFree(d_input);
    cudaFree(d_output);
}

// ── V4-4b: swiglu_limit clamp (llama.cpp LLM_ARCH_DEEPSEEK4 semantics) ─────
//
// Reference (ref/llama.cpp/src/llama-graph.cpp build_ffn/build_moe_ffn,
// arch == DEEPSEEK4, ggml_swiglu_split):
//   out = SiLU(min(gate, L)) * clamp(up, -L, +L)
// gate has NO lower clamp; up is clamped two-sided; no post-SiLU cap.

TEST(FusedSwigluTest, SwigluLimitClampGolden) {
    REQUIRES_GPU();

    const int num_tokens = 16;
    const int d = 512;
    const float limit = 10.0f;

    std::mt19937 rng(7);
    // Scale well past the limit so gate max-clamp, gate-lower-NOT-clamped,
    // and the two-sided up clamp are all exercised.
    std::uniform_real_distribution<float> dist(-30.0f, 30.0f);

    std::vector<uint16_t> h_input(num_tokens * 2 * d);
    for (auto& v : h_input) v = float_to_bf16(dist(rng));

    std::vector<float> h_expected(num_tokens * d);
    for (int t = 0; t < num_tokens; t++) {
        for (int j = 0; j < d; j++) {
            float gate = bf16_to_float(h_input[t * 2 * d + j]);
            float up = bf16_to_float(h_input[t * 2 * d + d + j]);
            gate = std::fmin(gate, limit);                          // max only
            up = std::fmin(std::fmax(up, -limit), limit);           // two-sided
            h_expected[t * d + j] = silu_ref(gate) * up;
        }
    }

    uint16_t *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, num_tokens * 2 * d * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * d * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          h_input.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));

    lc::FusedSwigluParams params{num_tokens, d, limit};
    lc::launch_fused_swiglu(d_output, d_input, params, 2, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_output(num_tokens * d);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                          h_output.size() * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));

    // Products reach ~|silu(10)*10| ~ 100; bf16 mantissa at that magnitude is
    // ~0.5, so allow a relative tolerance.
    for (int i = 0; i < num_tokens * d; i++) {
        const float actual = bf16_to_float(h_output[i]);
        const float expected = h_expected[i];
        const float tol = 0.02f + 0.01f * std::fabs(expected);
        ASSERT_NEAR(actual, expected, tol) << "clamp mismatch at " << i;
    }

    // Sanity: some gates below -limit must exist AND produce non-clamped SiLU
    // (i.e. essentially 0 for gate ~ -30, NOT silu(-10) ~ -4.5e-4 scaled).
    // Covered implicitly by the golden above (reference has no lower clamp).

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST(FusedSwigluTest, SwigluLimitZeroRegressionBitExact) {
    REQUIRES_GPU();

    const int num_tokens = 8;
    const int d = 2048;

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<uint16_t> h_input(num_tokens * 2 * d);
    for (auto& v : h_input) v = float_to_bf16(dist(rng));

    uint16_t *d_input, *d_out_a, *d_out_b;
    CUDA_CHECK(cudaMalloc(&d_input, num_tokens * 2 * d * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_out_a, num_tokens * d * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_out_b, num_tokens * d * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          h_input.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));

    lc::FusedSwigluParams p_default{num_tokens, d};       // limit 0 (off)
    lc::FusedSwigluParams p_zero{num_tokens, d, 0.0f};    // explicit 0
    lc::launch_fused_swiglu(d_out_a, d_input, p_default, 2, nullptr);
    lc::launch_fused_swiglu(d_out_b, d_input, p_zero, 2, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_a(num_tokens * d), h_b(num_tokens * d);
    CUDA_CHECK(cudaMemcpy(h_a.data(), d_out_a, h_a.size() * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_out_b, h_b.size() * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(0, std::memcmp(h_a.data(), h_b.data(),
                             h_a.size() * sizeof(uint16_t)))
        << "limit=0 must be bit-identical to the unclamped legacy path";

    cudaFree(d_input);
    cudaFree(d_out_a);
    cudaFree(d_out_b);
}
