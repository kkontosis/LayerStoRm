// Unit tests for grouped GEMM (NVFP4 and FP8) on SM120.
//
// Tests run on GPU (REQUIRES_GPU). Uses uniform/identity scales to isolate
// the grouped dispatch logic from scale-factor complexity.

#include "sm120/gemm/grouped_gemm.h"
#include "model/quantization/nvfp4.h"
#include "model/quantization/fp8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
namespace lm = layerstorm::model;

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: "                         \
                                     << cudaGetErrorString(_err);              \
    } while (0)

static float bf16_to_float(uint16_t bits) {
    uint32_t f = static_cast<uint32_t>(bits) << 16;
    float r; std::memcpy(&r, &f, sizeof(r)); return r;
}

// ── FP8 E4M3 encoding ─────────────────────────────────────────────────────

static uint8_t encode_fp8_e4m3(float val) {
    if (std::isnan(val)) return 0x7F;
    uint8_t sign = val < 0 ? 1 : 0;
    float abs_val = std::fabs(val);
    if (abs_val == 0.0f) return 0;
    if (abs_val > 448.0f) abs_val = 448.0f;

    float log2_val = std::log2(abs_val);
    int exp_unbiased = static_cast<int>(std::floor(log2_val));
    exp_unbiased = std::max(exp_unbiased, -6);
    exp_unbiased = std::min(exp_unbiased, 8);

    float mantissa_f = abs_val / std::ldexp(1.0f, exp_unbiased) - 1.0f;
    int mant = static_cast<int>(std::round(mantissa_f * 8.0f));
    mant = std::min(mant, 7);

    int biased_exp = exp_unbiased + 7;
    biased_exp = std::max(biased_exp, 0);
    biased_exp = std::min(biased_exp, 15);

    return (sign << 7) | (biased_exp << 3) | mant;
}

static float decode_fp8_e4m3(uint8_t bits) {
    return lm::fp8_e4m3::decode(bits);
}

// ═══════════════════════════════════════════════════════════════════════════
// FP8 Grouped GEMM Tests
// ═══════════════════════════════════════════════════════════════════════════

// ── FP8 Basic: 4 experts, uniform M ─────────────────────────────────────

TEST(Fp8GroupedGemmTest, BasicUniformGroups) {
    REQUIRES_GPU();

    const int num_experts = 4;
    const int M_per = 128;   // SM120 requires M >= 128 for FP8
    const int N = 128;
    const int K = 128;
    const int total_M = num_experts * M_per;
    const int scale_k_blocks = (K + 127) / 128;  // = 1
    const int scale_n_blocks = (N + 127) / 128;  // = 1

    std::mt19937 rng(42);

    // Generate small FP8 values
    std::vector<uint8_t> h_A(total_M * K);
    std::vector<uint8_t> h_B(num_experts * N * K);
    for (auto& v : h_A) v = encode_fp8_e4m3(0.1f);
    for (auto& v : h_B) v = encode_fp8_e4m3(0.1f);

    // Uniform scales = 1.0
    std::vector<float> h_scale_A(total_M * scale_k_blocks, 1.0f);
    std::vector<float> h_scale_B(num_experts * scale_k_blocks * scale_n_blocks, 1.0f);

    // Expert offsets: [0, 128, 256, 384, 512]
    std::vector<int32_t> h_offsets(num_experts + 1);
    for (int i = 0; i <= num_experts; i++) h_offsets[i] = i * M_per;

    // Problem sizes: {M_per, N, K} per expert
    std::vector<int32_t> h_problems(num_experts * 3);
    for (int i = 0; i < num_experts; i++) {
        h_problems[i * 3 + 0] = M_per;
        h_problems[i * 3 + 1] = N;
        h_problems[i * 3 + 2] = K;
    }

    // Allocate device memory
    void *d_A, *d_B, *d_D, *d_scale_A, *d_scale_B;
    int32_t *d_offsets, *d_problems;
    void* d_workspace;

    size_t out_bytes = total_M * N * sizeof(uint16_t);
    size_t ws_size = lc::query_fp8_grouped_gemm_workspace_size(
        num_experts, N, K, lc::GemmOutputDtype::kBFloat16);

    CUDA_CHECK(cudaMalloc(&d_A, total_M * K));
    CUDA_CHECK(cudaMalloc(&d_B, num_experts * N * K));
    CUDA_CHECK(cudaMalloc(&d_D, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_scale_A, h_scale_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale_B, h_scale_B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_offsets, h_offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_problems, h_problems.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_workspace, ws_size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_A, h_scale_A.data(), h_scale_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_B, h_scale_B.data(), h_scale_B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_problems, h_problems.data(), h_problems.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D, 0, out_bytes));

    lc::Fp8GroupedGemmParams params;
    params.num_experts = num_experts;
    params.N = N;
    params.K = K;
    params.A_base = d_A;
    params.B_base = d_B;
    params.D_base = d_D;
    params.scale_A_base = d_scale_A;
    params.scale_B_base = d_scale_B;
    params.expert_offsets = d_offsets;
    params.problem_sizes = d_problems;
    params.output_dtype = lc::GemmOutputDtype::kBFloat16;

    lc::launch_fp8_grouped_gemm(params, d_workspace, ws_size, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read output
    std::vector<uint16_t> h_D(total_M * N);
    CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, out_bytes, cudaMemcpyDeviceToHost));

    // CPU reference: each element = K * decode(0.1) * decode(0.1) * 1.0 * 1.0
    float a_val = decode_fp8_e4m3(encode_fp8_e4m3(0.1f));
    float b_val = decode_fp8_e4m3(encode_fp8_e4m3(0.1f));
    float expected = K * a_val * b_val;

    int errors = 0;
    for (int i = 0; i < total_M * N && errors < 10; i++) {
        float actual = bf16_to_float(h_D[i]);
        float rel_err = std::fabs(actual - expected) / std::max(std::fabs(expected), 1e-6f);
        if (rel_err > 0.1f) {
            errors++;
            if (errors <= 3) {
                EXPECT_NEAR(actual, expected, std::fabs(expected) * 0.1f)
                    << "FP8 grouped GEMM mismatch at index " << i;
            }
        }
    }
    EXPECT_EQ(errors, 0) << "FP8 grouped GEMM: " << errors << " elements with >10% error";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    cudaFree(d_scale_A); cudaFree(d_scale_B);
    cudaFree(d_offsets); cudaFree(d_problems);
    cudaFree(d_workspace);
}

// ── FP8 Variable Groups: non-uniform M with one empty expert ────────────

TEST(Fp8GroupedGemmTest, VariableGroups) {
    REQUIRES_GPU();

    const int num_experts = 4;
    const int M_vals[] = {256, 128, 0, 128};  // Expert 2 is empty
    const int N = 128;
    const int K = 128;

    int total_M = 0;
    for (int i = 0; i < num_experts; i++) total_M += M_vals[i];

    std::vector<int32_t> h_offsets(num_experts + 1);
    std::vector<int32_t> h_problems(num_experts * 3);
    h_offsets[0] = 0;
    for (int i = 0; i < num_experts; i++) {
        h_offsets[i + 1] = h_offsets[i] + M_vals[i];
        h_problems[i * 3 + 0] = M_vals[i];
        h_problems[i * 3 + 1] = N;
        h_problems[i * 3 + 2] = K;
    }

    // Small FP8 values
    std::vector<uint8_t> h_A(total_M * K, encode_fp8_e4m3(0.05f));
    std::vector<uint8_t> h_B(num_experts * N * K, encode_fp8_e4m3(0.05f));

    int scale_k = (K + 127) / 128;
    int scale_n = (N + 127) / 128;
    std::vector<float> h_scale_A(total_M * scale_k, 1.0f);
    std::vector<float> h_scale_B(num_experts * scale_k * scale_n, 1.0f);

    void *d_A, *d_B, *d_D, *d_sA, *d_sB;
    int32_t *d_off, *d_prob;
    void* d_ws;

    size_t out_bytes = total_M * N * sizeof(uint16_t);
    size_t ws_size = lc::query_fp8_grouped_gemm_workspace_size(
        num_experts, N, K, lc::GemmOutputDtype::kBFloat16);

    CUDA_CHECK(cudaMalloc(&d_A, total_M * K));
    CUDA_CHECK(cudaMalloc(&d_B, num_experts * N * K));
    CUDA_CHECK(cudaMalloc(&d_D, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_sA, h_scale_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sB, h_scale_B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_off, h_offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_prob, h_problems.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ws, ws_size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sA, h_scale_A.data(), h_scale_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sB, h_scale_B.data(), h_scale_B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_off, h_offsets.data(), h_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prob, h_problems.data(), h_problems.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D, 0, out_bytes));

    lc::Fp8GroupedGemmParams params;
    params.num_experts = num_experts;
    params.N = N;
    params.K = K;
    params.A_base = d_A;
    params.B_base = d_B;
    params.D_base = d_D;
    params.scale_A_base = d_sA;
    params.scale_B_base = d_sB;
    params.expert_offsets = d_off;
    params.problem_sizes = d_prob;
    params.output_dtype = lc::GemmOutputDtype::kBFloat16;

    // Should not crash with empty expert (M=0)
    lc::launch_fp8_grouped_gemm(params, d_ws, ws_size, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify output is non-zero for non-empty experts
    std::vector<uint16_t> h_D(total_M * N);
    CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, out_bytes, cudaMemcpyDeviceToHost));

    bool any_nonzero = false;
    for (auto v : h_D) {
        if (bf16_to_float(v) != 0.0f) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero) << "FP8 grouped GEMM output is all zeros";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    cudaFree(d_sA); cudaFree(d_sB);
    cudaFree(d_off); cudaFree(d_prob); cudaFree(d_ws);
}

// ═══════════════════════════════════════════════════════════════════════════
// NVFP4 Grouped GEMM Tests
// ═══════════════════════════════════════════════════════════════════════════

// Helper: encode float to nearest FP4 E2M1 nibble and pack pairs into bytes
static uint8_t quantize_fp4_pair(float a, float b) {
    auto nearest_nibble = [](float val) -> uint8_t {
        float abs_val = std::fabs(val);
        uint8_t best = 0;
        float best_dist = abs_val;
        for (uint8_t n = 0; n < 8; n++) {
            float dist = std::fabs(abs_val - lm::nvfp4::kE2M1Magnitudes[n]);
            if (dist < best_dist) { best_dist = dist; best = n; }
        }
        return (val < 0) ? (best | 0x8) : best;
    };
    uint8_t lo = nearest_nibble(a);
    uint8_t hi = nearest_nibble(b);
    return (hi << 4) | lo;
}

TEST(Nvfp4GroupedGemmTest, BasicUniformGroups) {
    REQUIRES_GPU();

    const int num_experts = 4;
    const int M_per = 128;
    const int N = 128;
    const int K = 128;
    const int total_M = num_experts * M_per;
    const int group_size = 16;

    // Fill A and B with uniform small FP4 values (0.5 = nibble 0x4)
    // Packed: 2 nibbles per byte, K/2 bytes per row
    int half_K = K / 2;
    std::vector<uint8_t> h_A(total_M * half_K);
    std::vector<uint8_t> h_B(num_experts * N * half_K);
    for (auto& v : h_A) v = quantize_fp4_pair(0.5f, 0.5f);
    for (auto& v : h_B) v = quantize_fp4_pair(0.5f, 0.5f);

    // Uniform UE4M3 scales = 1.0 (0x38)
    // Scale layout: padded per CUTLASS Sm1xxBlkScaledConfig
    int sf_M = ((total_M + 127) / 128) * 128;
    int sf_K = ((K / group_size + 3) / 4) * 4;
    int sf_N = ((N + 127) / 128) * 128;
    std::vector<uint8_t> h_scale_A(sf_M * sf_K, 0x38);
    std::vector<uint8_t> h_scale_B(num_experts * sf_N * sf_K, 0x38);

    // Expert offsets and SF offsets
    std::vector<int32_t> h_offsets(num_experts + 1);
    std::vector<int32_t> h_sf_offsets(num_experts + 1);
    // SF offsets must be 128-byte aligned per expert
    int sf_rows_per_expert = ((M_per + 127) / 128) * 128;
    for (int i = 0; i <= num_experts; i++) {
        h_offsets[i] = i * M_per;
        h_sf_offsets[i] = i * sf_rows_per_expert;
    }

    // Problem sizes
    std::vector<int32_t> h_problems(num_experts * 3);
    for (int i = 0; i < num_experts; i++) {
        h_problems[i * 3 + 0] = M_per;
        h_problems[i * 3 + 1] = N;
        h_problems[i * 3 + 2] = K;
    }

    // Alphas = 1.0
    std::vector<float> h_alphas(num_experts, 1.0f);

    // Allocate device memory
    void *d_A, *d_B, *d_D, *d_sA, *d_sB;
    float *d_alphas;
    int32_t *d_off, *d_sf_off, *d_prob;
    void* d_ws;

    size_t out_bytes = total_M * N * sizeof(uint16_t);
    size_t ws_size = lc::query_nvfp4_grouped_gemm_workspace_size(
        num_experts, N, K, lc::GemmOutputDtype::kBFloat16);

    CUDA_CHECK(cudaMalloc(&d_A, h_A.size()));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size()));
    CUDA_CHECK(cudaMalloc(&d_D, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_sA, h_scale_A.size()));
    CUDA_CHECK(cudaMalloc(&d_sB, h_scale_B.size()));
    CUDA_CHECK(cudaMalloc(&d_alphas, h_alphas.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_off, h_offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_sf_off, h_sf_offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_prob, h_problems.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ws, ws_size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sA, h_scale_A.data(), h_scale_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sB, h_scale_B.data(), h_scale_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alphas, h_alphas.data(), h_alphas.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_off, h_offsets.data(), h_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sf_off, h_sf_offsets.data(), h_sf_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prob, h_problems.data(), h_problems.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D, 0, out_bytes));

    lc::Nvfp4GroupedGemmParams params;
    params.num_experts = num_experts;
    params.N = N;
    params.K = K;
    params.A_base = d_A;
    params.B_base = d_B;
    params.D_base = d_D;
    params.scale_A_base = d_sA;
    params.scale_B_base = d_sB;
    params.alphas = d_alphas;
    params.expert_offsets = d_off;
    params.sf_offsets = d_sf_off;
    params.problem_sizes = d_prob;
    params.output_dtype = lc::GemmOutputDtype::kBFloat16;

    lc::launch_nvfp4_grouped_gemm(params, d_ws, ws_size, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read output
    std::vector<uint16_t> h_D(total_M * N);
    CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, out_bytes, cudaMemcpyDeviceToHost));

    // CPU reference: each element = K * 0.5 * 0.5 * 1.0 * 1.0 * 1.0 = K * 0.25
    float expected = K * 0.25f;

    int errors = 0;
    for (int i = 0; i < total_M * N && errors < 10; i++) {
        float actual = bf16_to_float(h_D[i]);
        float rel_err = std::fabs(actual - expected) / std::max(std::fabs(expected), 1e-6f);
        if (rel_err > 0.25f) {  // FP4 tolerance is wider
            errors++;
            if (errors <= 3) {
                EXPECT_NEAR(actual, expected, std::fabs(expected) * 0.25f)
                    << "NVFP4 grouped GEMM mismatch at index " << i;
            }
        }
    }
    EXPECT_EQ(errors, 0) << "NVFP4 grouped GEMM: " << errors << " elements with >25% error";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    cudaFree(d_sA); cudaFree(d_sB); cudaFree(d_alphas);
    cudaFree(d_off); cudaFree(d_sf_off); cudaFree(d_prob);
    cudaFree(d_ws);
}

// ── NVFP4 output is non-zero (basic sanity) ────────────────────────────────

TEST(Nvfp4GroupedGemmTest, OutputNonZero) {
    REQUIRES_GPU();

    const int num_experts = 2;
    const int M_per = 128;
    const int N = 128;
    const int K = 128;
    const int total_M = num_experts * M_per;
    const int group_size = 16;
    int half_K = K / 2;

    std::vector<uint8_t> h_A(total_M * half_K, quantize_fp4_pair(1.0f, 1.0f));
    std::vector<uint8_t> h_B(num_experts * N * half_K, quantize_fp4_pair(1.0f, 1.0f));

    int sf_M = ((total_M + 127) / 128) * 128;
    int sf_K = ((K / group_size + 3) / 4) * 4;
    int sf_N = ((N + 127) / 128) * 128;
    std::vector<uint8_t> h_scale_A(sf_M * sf_K, 0x38);
    std::vector<uint8_t> h_scale_B(num_experts * sf_N * sf_K, 0x38);

    int sf_rows_per_expert = ((M_per + 127) / 128) * 128;
    std::vector<int32_t> h_offsets(num_experts + 1);
    std::vector<int32_t> h_sf_offsets(num_experts + 1);
    std::vector<int32_t> h_problems(num_experts * 3);
    std::vector<float> h_alphas(num_experts, 1.0f);

    for (int i = 0; i <= num_experts; i++) {
        h_offsets[i] = i * M_per;
        h_sf_offsets[i] = i * sf_rows_per_expert;
    }
    for (int i = 0; i < num_experts; i++) {
        h_problems[i * 3] = M_per;
        h_problems[i * 3 + 1] = N;
        h_problems[i * 3 + 2] = K;
    }

    void *d_A, *d_B, *d_D, *d_sA, *d_sB;
    float *d_alphas;
    int32_t *d_off, *d_sf_off, *d_prob;
    void* d_ws;

    size_t out_bytes = total_M * N * sizeof(uint16_t);
    size_t ws_size = lc::query_nvfp4_grouped_gemm_workspace_size(
        num_experts, N, K, lc::GemmOutputDtype::kBFloat16);

    CUDA_CHECK(cudaMalloc(&d_A, h_A.size()));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size()));
    CUDA_CHECK(cudaMalloc(&d_D, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_sA, h_scale_A.size()));
    CUDA_CHECK(cudaMalloc(&d_sB, h_scale_B.size()));
    CUDA_CHECK(cudaMalloc(&d_alphas, num_experts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_off, h_offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_sf_off, h_sf_offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_prob, h_problems.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ws, ws_size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sA, h_scale_A.data(), h_scale_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sB, h_scale_B.data(), h_scale_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alphas, h_alphas.data(), num_experts * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_off, h_offsets.data(), h_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sf_off, h_sf_offsets.data(), h_sf_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prob, h_problems.data(), h_problems.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D, 0, out_bytes));

    lc::Nvfp4GroupedGemmParams params;
    params.num_experts = num_experts;
    params.N = N;
    params.K = K;
    params.A_base = d_A;
    params.B_base = d_B;
    params.D_base = d_D;
    params.scale_A_base = d_sA;
    params.scale_B_base = d_sB;
    params.alphas = d_alphas;
    params.expert_offsets = d_off;
    params.sf_offsets = d_sf_off;
    params.problem_sizes = d_prob;
    params.output_dtype = lc::GemmOutputDtype::kBFloat16;

    lc::launch_nvfp4_grouped_gemm(params, d_ws, ws_size, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_D(total_M * N);
    CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, out_bytes, cudaMemcpyDeviceToHost));

    bool any_nonzero = false;
    for (auto v : h_D) {
        if (bf16_to_float(v) != 0.0f) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero) << "NVFP4 grouped GEMM output is all zeros";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    cudaFree(d_sA); cudaFree(d_sB); cudaFree(d_alphas);
    cudaFree(d_off); cudaFree(d_sf_off); cudaFree(d_prob);
    cudaFree(d_ws);
}
