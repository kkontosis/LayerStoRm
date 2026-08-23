// Unit tests for RMSNorm CUDA kernels.
// Tests correctness against CPU reference for FP32, BF16, FP16.

#include "compute/kernels/norm/rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;

// ── CUDA error checking ─────────────────────────────────────────────────────

#define CUDA_CHECK(expr)                                                 \
    do {                                                                 \
        cudaError_t _err = (expr);                                       \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_err); \
    } while (0)

// ── CPU reference implementation ────────────────────────────────────────────

static void rmsnorm_ref(float* out, const float* input, const float* weight,
                        float epsilon, int num_tokens, int hidden_size) {
    for (int t = 0; t < num_tokens; ++t) {
        const float* row = input + t * hidden_size;
        float* out_row = out + t * hidden_size;
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            sum_sq += row[i] * row[i];
        }
        float rms_inv = 1.0f / std::sqrt(sum_sq / hidden_size + epsilon);
        for (int i = 0; i < hidden_size; ++i) {
            out_row[i] = row[i] * rms_inv * weight[i];
        }
    }
}

static void fused_add_rmsnorm_ref(float* out, const float* input,
                                  float* residual, const float* weight,
                                  float epsilon, int num_tokens,
                                  int hidden_size) {
    for (int t = 0; t < num_tokens; ++t) {
        float* res_row = residual + t * hidden_size;
        const float* in_row = input + t * hidden_size;
        float* out_row = out + t * hidden_size;
        // Add input to residual
        for (int i = 0; i < hidden_size; ++i) {
            res_row[i] += in_row[i];
        }
        // RMSNorm on updated residual
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            sum_sq += res_row[i] * res_row[i];
        }
        float rms_inv = 1.0f / std::sqrt(sum_sq / hidden_size + epsilon);
        for (int i = 0; i < hidden_size; ++i) {
            out_row[i] = res_row[i] * rms_inv * weight[i];
        }
    }
}

// ── Type conversion helpers (host-side, bit manipulation) ───────────────────

static uint16_t float_to_half_bits(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exponent = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = f & 0x7FFFFF;
    if (exponent <= 0) {
        // Subnormal or zero
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000;
        int shift = 14 - exponent;
        uint32_t round_bit = (mantissa >> (shift - 1)) & 1;
        uint16_t result = static_cast<uint16_t>(sign | (mantissa >> shift));
        return static_cast<uint16_t>(result + round_bit);
    }
    if (exponent >= 31) {
        // Overflow to inf
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    uint32_t round_bit = (mantissa >> 12) & 1;
    uint16_t result = static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
    return static_cast<uint16_t>(result + round_bit);
}

static float half_bits_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) {
            f = sign;
        } else {
            exponent = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        f = sign | 0x7F800000 | (mantissa << 13);
    } else {
        f = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

static __half float_to_half(float v) {
    uint16_t bits = float_to_half_bits(v);
    __half h;
    std::memcpy(&h, &bits, sizeof(h));
    return h;
}

static float half_to_float(__half h) {
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return half_bits_to_float(bits);
}

static __nv_bfloat16 float_to_bf16(float v) {
    // BF16 = upper 16 bits of float32 with round-to-nearest-even
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint32_t round_bit = (f >> 15) & 1;
    uint32_t lsb = (f >> 16) & 1;
    // Round to nearest even: add 0x7FFF + lsb
    f += 0x7FFF + lsb;
    uint16_t bits = static_cast<uint16_t>(f >> 16);
    __nv_bfloat16 b;
    std::memcpy(&b, &bits, sizeof(b));
    return b;
}

static float bf16_to_float(__nv_bfloat16 b) {
    uint16_t bits;
    std::memcpy(&bits, &b, sizeof(bits));
    uint32_t f = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

// ── Test fixture ────────────────────────────────────────────────────────────

class RMSNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Deterministic random data
        gen_.seed(42);
    }

    // Fill vector with uniform random in [-1, 1]
    void fill_random(std::vector<float>& v) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : v) x = dist(gen_);
    }

    // Fill weight vector with uniform random in [0.5, 1.5] (positive weights)
    void fill_weight(std::vector<float>& v) {
        std::uniform_real_distribution<float> dist(0.5f, 1.5f);
        for (auto& x : v) x = dist(gen_);
    }

    std::mt19937 gen_;
};

// ── FP32 plain RMSNorm ─────────────────────────────────────────────────────

TEST_F(RMSNormTest, PlainFP32_Small) {
    REQUIRES_GPU();
    constexpr int num_tokens = 4;
    constexpr int hidden_size = 128;
    constexpr float eps = 1e-6f;

    std::vector<float> h_input(num_tokens * hidden_size);
    std::vector<float> h_weight(hidden_size);
    std::vector<float> h_out_ref(num_tokens * hidden_size);
    std::vector<float> h_out_gpu(num_tokens * hidden_size);

    fill_random(h_input);
    fill_weight(h_weight);
    rmsnorm_ref(h_out_ref.data(), h_input.data(), h_weight.data(), eps,
                num_tokens, hidden_size);

    float *d_input, *d_weight, *d_out;
    CUDA_CHECK(cudaMalloc(&d_input, num_tokens * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, num_tokens * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          num_tokens * hidden_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(),
                          hidden_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    lc::launch_rmsnorm(d_out, d_input, d_weight, eps, num_tokens, hidden_size,
                       lc::NormDtype::kFloat32, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out,
                          num_tokens * hidden_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        EXPECT_NEAR(h_out_gpu[i], h_out_ref[i], 1e-5f)
            << "Mismatch at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_out);
}

TEST_F(RMSNormTest, PlainFP32_ModelSize) {
    REQUIRES_GPU();
    constexpr int num_tokens = 64;
    constexpr int hidden_size = 7168;  // V3.2
    constexpr float eps = 1e-6f;

    std::vector<float> h_input(num_tokens * hidden_size);
    std::vector<float> h_weight(hidden_size);
    std::vector<float> h_out_ref(num_tokens * hidden_size);
    std::vector<float> h_out_gpu(num_tokens * hidden_size);

    fill_random(h_input);
    fill_weight(h_weight);
    rmsnorm_ref(h_out_ref.data(), h_input.data(), h_weight.data(), eps,
                num_tokens, hidden_size);

    float *d_input, *d_weight, *d_out;
    size_t data_bytes = num_tokens * hidden_size * sizeof(float);
    size_t weight_bytes = hidden_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
    CUDA_CHECK(
        cudaMemcpy(d_input, h_input.data(), data_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice));

    lc::launch_rmsnorm(d_out, d_input, d_weight, eps, num_tokens, hidden_size,
                       lc::NormDtype::kFloat32, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(h_out_gpu.data(), d_out, data_bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        EXPECT_NEAR(h_out_gpu[i], h_out_ref[i], 1e-4f)
            << "Mismatch at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_out);
}

// ── BF16 plain RMSNorm ─────────────────────────────────────────────────────

TEST_F(RMSNormTest, PlainBF16) {
    REQUIRES_GPU();
    constexpr int num_tokens = 16;
    constexpr int hidden_size = 128;
    constexpr float eps = 1e-6f;

    std::vector<float> h_input_f(num_tokens * hidden_size);
    std::vector<float> h_weight_f(hidden_size);
    fill_random(h_input_f);
    fill_weight(h_weight_f);

    // Convert to bf16 then back to float for reference (captures quantization)
    std::vector<__nv_bfloat16> h_input_bf(num_tokens * hidden_size);
    std::vector<__nv_bfloat16> h_weight_bf(hidden_size);
    std::vector<float> h_input_roundtrip(num_tokens * hidden_size);
    std::vector<float> h_weight_roundtrip(hidden_size);

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        h_input_bf[i] = float_to_bf16(h_input_f[i]);
        h_input_roundtrip[i] = bf16_to_float(h_input_bf[i]);
    }
    for (int i = 0; i < hidden_size; ++i) {
        h_weight_bf[i] = float_to_bf16(h_weight_f[i]);
        h_weight_roundtrip[i] = bf16_to_float(h_weight_bf[i]);
    }

    // CPU reference on roundtripped values
    std::vector<float> h_out_ref(num_tokens * hidden_size);
    rmsnorm_ref(h_out_ref.data(), h_input_roundtrip.data(),
                h_weight_roundtrip.data(), eps, num_tokens, hidden_size);

    // GPU
    __nv_bfloat16 *d_input, *d_weight, *d_out;
    size_t data_bytes = num_tokens * hidden_size * sizeof(__nv_bfloat16);
    size_t weight_bytes = hidden_size * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input_bf.data(), data_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight_bf.data(), weight_bytes,
                          cudaMemcpyHostToDevice));

    lc::launch_rmsnorm(d_out, d_input, d_weight, eps, num_tokens, hidden_size,
                       lc::NormDtype::kBFloat16, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out_bf(num_tokens * hidden_size);
    CUDA_CHECK(cudaMemcpy(h_out_bf.data(), d_out, data_bytes,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        float gpu_val = bf16_to_float(h_out_bf[i]);
        EXPECT_NEAR(gpu_val, h_out_ref[i], 2e-2f)
            << "BF16 mismatch at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_out);
}

// ── FP16 plain RMSNorm ─────────────────────────────────────────────────────

TEST_F(RMSNormTest, PlainFP16) {
    REQUIRES_GPU();
    constexpr int num_tokens = 16;
    constexpr int hidden_size = 128;
    constexpr float eps = 1e-6f;

    std::vector<float> h_input_f(num_tokens * hidden_size);
    std::vector<float> h_weight_f(hidden_size);
    fill_random(h_input_f);
    fill_weight(h_weight_f);

    std::vector<__half> h_input_h(num_tokens * hidden_size);
    std::vector<__half> h_weight_h(hidden_size);
    std::vector<float> h_input_roundtrip(num_tokens * hidden_size);
    std::vector<float> h_weight_roundtrip(hidden_size);

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        h_input_h[i] = float_to_half(h_input_f[i]);
        h_input_roundtrip[i] = half_to_float(h_input_h[i]);
    }
    for (int i = 0; i < hidden_size; ++i) {
        h_weight_h[i] = float_to_half(h_weight_f[i]);
        h_weight_roundtrip[i] = half_to_float(h_weight_h[i]);
    }

    std::vector<float> h_out_ref(num_tokens * hidden_size);
    rmsnorm_ref(h_out_ref.data(), h_input_roundtrip.data(),
                h_weight_roundtrip.data(), eps, num_tokens, hidden_size);

    __half *d_input, *d_weight, *d_out;
    size_t data_bytes = num_tokens * hidden_size * sizeof(__half);
    size_t weight_bytes = hidden_size * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input_h.data(), data_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight_h.data(), weight_bytes,
                          cudaMemcpyHostToDevice));

    lc::launch_rmsnorm(d_out, d_input, d_weight, eps, num_tokens, hidden_size,
                       lc::NormDtype::kFloat16, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> h_out_h(num_tokens * hidden_size);
    CUDA_CHECK(cudaMemcpy(h_out_h.data(), d_out, data_bytes,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        float gpu_val = half_to_float(h_out_h[i]);
        EXPECT_NEAR(gpu_val, h_out_ref[i], 1e-2f)
            << "FP16 mismatch at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_out);
}

// ── Fused add+RMSNorm FP32 ─────────────────────────────────────────────────

TEST_F(RMSNormTest, FusedAddFP32) {
    REQUIRES_GPU();
    constexpr int num_tokens = 8;
    constexpr int hidden_size = 128;
    constexpr float eps = 1e-6f;

    std::vector<float> h_input(num_tokens * hidden_size);
    std::vector<float> h_residual(num_tokens * hidden_size);
    std::vector<float> h_weight(hidden_size);
    fill_random(h_input);
    fill_random(h_residual);
    fill_weight(h_weight);

    // CPU reference (copy residual since it's modified in-place)
    std::vector<float> h_residual_ref = h_residual;
    std::vector<float> h_out_ref(num_tokens * hidden_size);
    fused_add_rmsnorm_ref(h_out_ref.data(), h_input.data(),
                          h_residual_ref.data(), h_weight.data(), eps,
                          num_tokens, hidden_size);

    float *d_input, *d_residual, *d_weight, *d_out;
    size_t data_bytes = num_tokens * hidden_size * sizeof(float);
    size_t weight_bytes = hidden_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_residual, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
    CUDA_CHECK(
        cudaMemcpy(d_input, h_input.data(), data_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual.data(), data_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes,
                          cudaMemcpyHostToDevice));

    lc::launch_fused_add_rmsnorm(d_out, d_input, d_residual, d_weight, eps,
                                 num_tokens, hidden_size,
                                 lc::NormDtype::kFloat32, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check output
    std::vector<float> h_out_gpu(num_tokens * hidden_size);
    CUDA_CHECK(
        cudaMemcpy(h_out_gpu.data(), d_out, data_bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        EXPECT_NEAR(h_out_gpu[i], h_out_ref[i], 1e-5f)
            << "Output mismatch at " << i;
    }

    // Check residual was updated in-place
    std::vector<float> h_residual_gpu(num_tokens * hidden_size);
    CUDA_CHECK(cudaMemcpy(h_residual_gpu.data(), d_residual, data_bytes,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        EXPECT_NEAR(h_residual_gpu[i], h_residual_ref[i], 1e-6f)
            << "Residual mismatch at " << i;
    }

    cudaFree(d_input);
    cudaFree(d_residual);
    cudaFree(d_weight);
    cudaFree(d_out);
}

// ── Fused add+RMSNorm BF16 ─────────────────────────────────────────────────

TEST_F(RMSNormTest, FusedAddBF16) {
    REQUIRES_GPU();
    constexpr int num_tokens = 8;
    constexpr int hidden_size = 128;
    constexpr float eps = 1e-6f;

    std::vector<float> h_input_f(num_tokens * hidden_size);
    std::vector<float> h_residual_f(num_tokens * hidden_size);
    std::vector<float> h_weight_f(hidden_size);
    fill_random(h_input_f);
    fill_random(h_residual_f);
    fill_weight(h_weight_f);

    std::vector<__nv_bfloat16> h_input_bf(num_tokens * hidden_size);
    std::vector<__nv_bfloat16> h_residual_bf(num_tokens * hidden_size);
    std::vector<__nv_bfloat16> h_weight_bf(hidden_size);
    std::vector<float> h_input_rt(num_tokens * hidden_size);
    std::vector<float> h_residual_rt(num_tokens * hidden_size);
    std::vector<float> h_weight_rt(hidden_size);

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        h_input_bf[i] = float_to_bf16(h_input_f[i]);
        h_input_rt[i] = bf16_to_float(h_input_bf[i]);
        h_residual_bf[i] = float_to_bf16(h_residual_f[i]);
        h_residual_rt[i] = bf16_to_float(h_residual_bf[i]);
    }
    for (int i = 0; i < hidden_size; ++i) {
        h_weight_bf[i] = float_to_bf16(h_weight_f[i]);
        h_weight_rt[i] = bf16_to_float(h_weight_bf[i]);
    }

    std::vector<float> h_out_ref(num_tokens * hidden_size);
    fused_add_rmsnorm_ref(h_out_ref.data(), h_input_rt.data(),
                          h_residual_rt.data(), h_weight_rt.data(), eps,
                          num_tokens, hidden_size);

    __nv_bfloat16 *d_input, *d_residual, *d_weight, *d_out;
    size_t data_bytes = num_tokens * hidden_size * sizeof(__nv_bfloat16);
    size_t weight_bytes = hidden_size * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_residual, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input_bf.data(), data_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual_bf.data(), data_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight_bf.data(), weight_bytes,
                          cudaMemcpyHostToDevice));

    lc::launch_fused_add_rmsnorm(d_out, d_input, d_residual, d_weight, eps,
                                 num_tokens, hidden_size,
                                 lc::NormDtype::kBFloat16, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out_bf(num_tokens * hidden_size);
    CUDA_CHECK(cudaMemcpy(h_out_bf.data(), d_out, data_bytes,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        float gpu_val = bf16_to_float(h_out_bf[i]);
        EXPECT_NEAR(gpu_val, h_out_ref[i], 2e-2f)
            << "BF16 fused mismatch at " << i;
    }

    cudaFree(d_input);
    cudaFree(d_residual);
    cudaFree(d_weight);
    cudaFree(d_out);
}

// ── Epsilon sensitivity ─────────────────────────────────────────────────────

TEST_F(RMSNormTest, EpsilonSensitivity) {
    REQUIRES_GPU();
    constexpr int num_tokens = 1;
    constexpr int hidden_size = 128;

    std::vector<float> h_input(hidden_size);
    std::vector<float> h_weight(hidden_size);
    fill_random(h_input);
    fill_weight(h_weight);

    float *d_input, *d_weight, *d_out1, *d_out2;
    size_t bytes = hidden_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, bytes));
    CUDA_CHECK(cudaMalloc(&d_out1, bytes));
    CUDA_CHECK(cudaMalloc(&d_out2, bytes));
    CUDA_CHECK(
        cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_weight, h_weight.data(), bytes, cudaMemcpyHostToDevice));

    lc::launch_rmsnorm(d_out1, d_input, d_weight, 1e-6f, num_tokens,
                       hidden_size, lc::NormDtype::kFloat32, nullptr);
    lc::launch_rmsnorm(d_out2, d_input, d_weight, 1e-5f, num_tokens,
                       hidden_size, lc::NormDtype::kFloat32, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out1(hidden_size), h_out2(hidden_size);
    CUDA_CHECK(
        cudaMemcpy(h_out1.data(), d_out1, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(h_out2.data(), d_out2, bytes, cudaMemcpyDeviceToHost));

    // Outputs should differ (epsilon affects the normalization)
    bool any_differ = false;
    for (int i = 0; i < hidden_size; ++i) {
        if (std::abs(h_out1[i] - h_out2[i]) > 1e-8f) {
            any_differ = true;
            break;
        }
    }
    EXPECT_TRUE(any_differ) << "Different epsilons should produce different outputs";

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_out1);
    cudaFree(d_out2);
}

// ── Single token edge case ──────────────────────────────────────────────────

TEST_F(RMSNormTest, SingleToken) {
    REQUIRES_GPU();
    constexpr int num_tokens = 1;
    constexpr int hidden_size = 7168;
    constexpr float eps = 1e-6f;

    std::vector<float> h_input(hidden_size);
    std::vector<float> h_weight(hidden_size);
    std::vector<float> h_out_ref(hidden_size);
    fill_random(h_input);
    fill_weight(h_weight);
    rmsnorm_ref(h_out_ref.data(), h_input.data(), h_weight.data(), eps, 1,
                hidden_size);

    float *d_input, *d_weight, *d_out;
    size_t bytes = hidden_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(
        cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_weight, h_weight.data(), bytes, cudaMemcpyHostToDevice));

    lc::launch_rmsnorm(d_out, d_input, d_weight, eps, num_tokens, hidden_size,
                       lc::NormDtype::kFloat32, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out_gpu(hidden_size);
    CUDA_CHECK(
        cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < hidden_size; ++i) {
        EXPECT_NEAR(h_out_gpu[i], h_out_ref[i], 1e-4f)
            << "Single token mismatch at " << i;
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_out);
}

// ── Large batch (exercises max_block_size=256 path) ─────────────────────────

TEST_F(RMSNormTest, LargeBatch) {
    REQUIRES_GPU();
    constexpr int num_tokens = 1024;
    constexpr int hidden_size = 128;
    constexpr float eps = 1e-6f;

    std::vector<float> h_input(num_tokens * hidden_size);
    std::vector<float> h_weight(hidden_size);
    std::vector<float> h_out_ref(num_tokens * hidden_size);
    fill_random(h_input);
    fill_weight(h_weight);
    rmsnorm_ref(h_out_ref.data(), h_input.data(), h_weight.data(), eps,
                num_tokens, hidden_size);

    float *d_input, *d_weight, *d_out;
    size_t data_bytes = num_tokens * hidden_size * sizeof(float);
    size_t weight_bytes = hidden_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
    CUDA_CHECK(
        cudaMemcpy(d_input, h_input.data(), data_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice));

    lc::launch_rmsnorm(d_out, d_input, d_weight, eps, num_tokens, hidden_size,
                       lc::NormDtype::kFloat32, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out_gpu(num_tokens * hidden_size);
    CUDA_CHECK(
        cudaMemcpy(h_out_gpu.data(), d_out, data_bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        EXPECT_NEAR(h_out_gpu[i], h_out_ref[i], 1e-5f)
            << "Large batch mismatch at " << i;
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_out);
}

// ── Fused add+RMSNorm FP16 ─────────────────────────────────────────────────

TEST_F(RMSNormTest, FusedAddFP16) {
    REQUIRES_GPU();
    constexpr int num_tokens = 8;
    constexpr int hidden_size = 128;
    constexpr float eps = 1e-6f;

    std::vector<float> h_input_f(num_tokens * hidden_size);
    std::vector<float> h_residual_f(num_tokens * hidden_size);
    std::vector<float> h_weight_f(hidden_size);
    fill_random(h_input_f);
    fill_random(h_residual_f);
    fill_weight(h_weight_f);

    std::vector<__half> h_input_h(num_tokens * hidden_size);
    std::vector<__half> h_residual_h(num_tokens * hidden_size);
    std::vector<__half> h_weight_h(hidden_size);
    std::vector<float> h_input_rt(num_tokens * hidden_size);
    std::vector<float> h_residual_rt(num_tokens * hidden_size);
    std::vector<float> h_weight_rt(hidden_size);

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        h_input_h[i] = float_to_half(h_input_f[i]);
        h_input_rt[i] = half_to_float(h_input_h[i]);
        h_residual_h[i] = float_to_half(h_residual_f[i]);
        h_residual_rt[i] = half_to_float(h_residual_h[i]);
    }
    for (int i = 0; i < hidden_size; ++i) {
        h_weight_h[i] = float_to_half(h_weight_f[i]);
        h_weight_rt[i] = half_to_float(h_weight_h[i]);
    }

    std::vector<float> h_out_ref(num_tokens * hidden_size);
    fused_add_rmsnorm_ref(h_out_ref.data(), h_input_rt.data(),
                          h_residual_rt.data(), h_weight_rt.data(), eps,
                          num_tokens, hidden_size);

    __half *d_input, *d_residual, *d_weight, *d_out;
    size_t data_bytes = num_tokens * hidden_size * sizeof(__half);
    size_t weight_bytes = hidden_size * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_residual, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input_h.data(), data_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual_h.data(), data_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight_h.data(), weight_bytes,
                          cudaMemcpyHostToDevice));

    lc::launch_fused_add_rmsnorm(d_out, d_input, d_residual, d_weight, eps,
                                 num_tokens, hidden_size,
                                 lc::NormDtype::kFloat16, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> h_out_h(num_tokens * hidden_size);
    CUDA_CHECK(cudaMemcpy(h_out_h.data(), d_out, data_bytes,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        float gpu_val = half_to_float(h_out_h[i]);
        EXPECT_NEAR(gpu_val, h_out_ref[i], 2e-2f)
            << "FP16 fused mismatch at " << i;
    }

    cudaFree(d_input);
    cudaFree(d_residual);
    cudaFree(d_weight);
    cudaFree(d_out);
}

// ── Strided RMSNorm (TD-PREFILL-CHUNK-ATTN) ─────────────────────────────────
//
// kv_a_layernorm normalizes the c_kv half (hidden = kv_lora_rank = 512) of the
// INTERLEAVED kv_a output rows [c_kv | k_pe] (row stride = 512 + 64 = 576).
// Root-cause lock-in for the B>1 multi-token prefill defect: the tight-row
// launch was exact only at num_tokens == 1 — row 1 read row 0's rope half into
// its norm and the output clobbered every row's rope half. The strided launch
// must (a) be BIT-EQUAL per row to a tight single-row rmsnorm and (b) leave the
// rope tail [hidden, stride) of every row untouched.
TEST_F(RMSNormTest, StridedBF16_BatchBitEqualsPerRow_TailUntouched) {
    REQUIRES_GPU();
    constexpr int num_tokens = 4;
    constexpr int hidden_size = 512;   // kv_lora_rank
    constexpr int row_stride = 576;    // kv_lora_rank + qk_rope_head_dim
    constexpr float eps = 1e-6f;

    std::vector<float> h_input_f(num_tokens * row_stride);
    std::vector<float> h_weight_f(hidden_size);
    fill_random(h_input_f);
    fill_weight(h_weight_f);

    std::vector<__nv_bfloat16> h_input_bf(num_tokens * row_stride);
    std::vector<__nv_bfloat16> h_weight_bf(hidden_size);
    for (size_t i = 0; i < h_input_bf.size(); ++i)
        h_input_bf[i] = float_to_bf16(h_input_f[i]);
    for (int i = 0; i < hidden_size; ++i)
        h_weight_bf[i] = float_to_bf16(h_weight_f[i]);

    const size_t buf_bytes = h_input_bf.size() * sizeof(__nv_bfloat16);
    const size_t weight_bytes = hidden_size * sizeof(__nv_bfloat16);
    const size_t row_bytes = hidden_size * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_strided, *d_weight, *d_row_in, *d_row_out;
    CUDA_CHECK(cudaMalloc(&d_strided, buf_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_row_in, row_bytes));
    CUDA_CHECK(cudaMalloc(&d_row_out, row_bytes));
    CUDA_CHECK(cudaMemcpy(d_strided, h_input_bf.data(), buf_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight_bf.data(), weight_bytes,
                          cudaMemcpyHostToDevice));

    // Batched strided launch, IN-PLACE (exactly the kv_a_norm call shape).
    lc::launch_rmsnorm_strided(d_strided, d_strided, d_weight, eps, num_tokens,
                               hidden_size, row_stride,
                               lc::NormDtype::kBFloat16, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_strided_out(h_input_bf.size());
    CUDA_CHECK(cudaMemcpy(h_strided_out.data(), d_strided, buf_bytes,
                          cudaMemcpyDeviceToHost));

    for (int t = 0; t < num_tokens; ++t) {
        // Reference: tight single-row rmsnorm on the SAME row data.
        CUDA_CHECK(cudaMemcpy(d_row_in,
                              h_input_bf.data() + t * row_stride,
                              row_bytes, cudaMemcpyHostToDevice));
        lc::launch_rmsnorm(d_row_out, d_row_in, d_weight, eps,
                           /*num_tokens=*/1, hidden_size,
                           lc::NormDtype::kBFloat16, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> h_row(hidden_size);
        CUDA_CHECK(cudaMemcpy(h_row.data(), d_row_out, row_bytes,
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < hidden_size; ++i) {
            const uint16_t a = *reinterpret_cast<const uint16_t*>(
                &h_strided_out[t * row_stride + i]);
            const uint16_t b = *reinterpret_cast<const uint16_t*>(&h_row[i]);
            ASSERT_EQ(a, b) << "row " << t << " elem " << i
                            << ": strided batch != per-row (bit mismatch)";
        }
        // Rope tail [hidden, stride) must be byte-identical to the input.
        for (int i = hidden_size; i < row_stride; ++i) {
            const uint16_t a = *reinterpret_cast<const uint16_t*>(
                &h_strided_out[t * row_stride + i]);
            const uint16_t b = *reinterpret_cast<const uint16_t*>(
                &h_input_bf[t * row_stride + i]);
            ASSERT_EQ(a, b) << "row " << t << " tail elem " << i
                            << ": rope half clobbered";
        }
    }

    cudaFree(d_strided);
    cudaFree(d_weight);
    cudaFree(d_row_in);
    cudaFree(d_row_out);
}
