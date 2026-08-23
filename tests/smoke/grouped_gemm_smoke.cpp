// Smoke test: Grouped GEMM (NVFP4 + FP8) on real GPU hardware.
//
// Expected hardware (dev machine):
//   GPU 0: RTX 5090, 31 GiB VRAM, SM120
//   GPU 1: RTX 5090, 31 GiB VRAM, SM120
//   GPU 2: RTX 5080, 15 GiB VRAM, SM120
//   GPU 3: RTX 5080, 15 GiB VRAM, SM120
//
// Exercises V3.2 MoE grouped GEMM with multiple experts per GPU.
// Verifies output is non-zero on every available GPU.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "sm120/gemm/grouped_gemm.h"
#include "model/quantization/nvfp4.h"
#include "model/quantization/fp8.h"

namespace lc = layerstorm::compute;
namespace lm = layerstorm::model;

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        EXPECT_EQ(_err, cudaSuccess) << "CUDA error: "                         \
                                     << cudaGetErrorString(_err);              \
    } while (0)

static float bf16_to_float(uint16_t bits) {
    uint32_t f = static_cast<uint32_t>(bits) << 16;
    float r; std::memcpy(&r, &f, sizeof(r)); return r;
}

static uint8_t encode_fp8_e4m3_simple(float val) {
    if (val == 0.0f) return 0;
    uint8_t sign = val < 0 ? 1 : 0;
    float abs_val = std::fabs(val);
    if (abs_val > 448.0f) abs_val = 448.0f;
    float log2_val = std::log2(abs_val);
    int exp_unbiased = std::max(-6, std::min(8, static_cast<int>(std::floor(log2_val))));
    float mantissa_f = abs_val / std::ldexp(1.0f, exp_unbiased) - 1.0f;
    int mant = std::min(7, static_cast<int>(std::round(mantissa_f * 8.0f)));
    int biased_exp = std::max(0, std::min(15, exp_unbiased + 7));
    return (sign << 7) | (biased_exp << 3) | mant;
}

// ── FP8 Grouped GEMM smoke: 8 experts, V3.2 down-proj dims ─────────────

TEST(GroupedGemmSmoke, Fp8_DownProj_AllGPUs) {
    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    if (gpu_count == 0) GTEST_SKIP() << "No CUDA GPUs available";

    std::cout << "Testing FP8 grouped GEMM (down proj) on " << gpu_count << " GPUs\n";

    const int num_experts = 8;
    const int M_per = 128;  // SM120 FP8 needs M >= 128
    const int N = 7168;     // V3.2 hidden_size
    const int K = 2048;     // V3.2 moe_intermediate_size
    const int total_M = num_experts * M_per;

    for (int gpu = 0; gpu < gpu_count; gpu++) {
        SCOPED_TRACE("GPU " + std::to_string(gpu));
        cudaSetDevice(gpu);

        int scale_k = (K + 127) / 128;
        int scale_n = (N + 127) / 128;

        std::vector<uint8_t> h_A(total_M * K, encode_fp8_e4m3_simple(0.01f));
        std::vector<uint8_t> h_B(num_experts * N * K, encode_fp8_e4m3_simple(0.01f));
        std::vector<float> h_sA(total_M * scale_k, 1.0f);
        std::vector<float> h_sB(num_experts * scale_k * scale_n, 1.0f);

        std::vector<int32_t> h_off(num_experts + 1);
        std::vector<int32_t> h_prob(num_experts * 3);
        for (int i = 0; i <= num_experts; i++) h_off[i] = i * M_per;
        for (int i = 0; i < num_experts; i++) {
            h_prob[i*3] = M_per; h_prob[i*3+1] = N; h_prob[i*3+2] = K;
        }

        void *d_A, *d_B, *d_D, *d_sA, *d_sB;
        int32_t *d_off, *d_prob;
        void* d_ws;

        size_t out_bytes = total_M * N * sizeof(uint16_t);
        size_t ws_size = lc::query_fp8_grouped_gemm_workspace_size(
            num_experts, N, K, lc::GemmOutputDtype::kBFloat16);

        CUDA_CHECK(cudaMalloc(&d_A, h_A.size()));
        CUDA_CHECK(cudaMalloc(&d_B, h_B.size()));
        CUDA_CHECK(cudaMalloc(&d_D, out_bytes));
        CUDA_CHECK(cudaMalloc(&d_sA, h_sA.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sB, h_sB.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_off, h_off.size() * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_prob, h_prob.size() * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_ws, ws_size));

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sA, h_sA.data(), h_sA.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sB, h_sB.data(), h_sB.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_off, h_off.data(), h_off.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_prob, h_prob.data(), h_prob.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_D, 0, out_bytes));

        lc::Fp8GroupedGemmParams params;
        params.num_experts = num_experts;
        params.N = N; params.K = K;
        params.A_base = d_A; params.B_base = d_B; params.D_base = d_D;
        params.scale_A_base = d_sA; params.scale_B_base = d_sB;
        params.expert_offsets = d_off; params.problem_sizes = d_prob;
        params.output_dtype = lc::GemmOutputDtype::kBFloat16;

        lc::launch_fp8_grouped_gemm(params, d_ws, ws_size, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check output non-zero
        std::vector<uint16_t> h_D(total_M * N);
        CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, out_bytes, cudaMemcpyDeviceToHost));

        bool any_nonzero = false;
        for (size_t i = 0; i < h_D.size(); i++) {
            if (bf16_to_float(h_D[i]) != 0.0f) { any_nonzero = true; break; }
        }
        EXPECT_TRUE(any_nonzero) << "GPU " << gpu << ": FP8 grouped GEMM output all zeros";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
        cudaFree(d_sA); cudaFree(d_sB);
        cudaFree(d_off); cudaFree(d_prob); cudaFree(d_ws);
    }
}

// ── NVFP4 Grouped GEMM smoke: 8 experts, V3.2 gate+up dims ─────────────

TEST(GroupedGemmSmoke, Nvfp4_GateUp_AllGPUs) {
    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    if (gpu_count == 0) GTEST_SKIP() << "No CUDA GPUs available";

    std::cout << "Testing NVFP4 grouped GEMM (gate+up) on " << gpu_count << " GPUs\n";

    const int num_experts = 8;
    const int M_per = 128;
    const int N = 4096;     // 2 * moe_intermediate_size (gate+up fused)
    const int K = 7168;     // V3.2 hidden_size
    const int total_M = num_experts * M_per;
    const int group_size = 16;
    int half_K = K / 2;

    for (int gpu = 0; gpu < gpu_count; gpu++) {
        SCOPED_TRACE("GPU " + std::to_string(gpu));
        cudaSetDevice(gpu);

        // FP4 packed data
        std::vector<uint8_t> h_A(total_M * half_K, 0x44);  // 0.5, 0.5
        std::vector<uint8_t> h_B(num_experts * N * half_K, 0x44);

        // UE4M3 scales = 1.0 (0x38)
        int sf_M = ((total_M + 127) / 128) * 128;
        int sf_K = ((K / group_size + 3) / 4) * 4;
        int sf_N = ((N + 127) / 128) * 128;
        std::vector<uint8_t> h_sA(sf_M * sf_K, 0x38);
        std::vector<uint8_t> h_sB(num_experts * sf_N * sf_K, 0x38);

        int sf_rows_per = ((M_per + 127) / 128) * 128;
        std::vector<int32_t> h_off(num_experts + 1);
        std::vector<int32_t> h_sf_off(num_experts + 1);
        std::vector<int32_t> h_prob(num_experts * 3);
        std::vector<float> h_alphas(num_experts, 1.0f);

        for (int i = 0; i <= num_experts; i++) {
            h_off[i] = i * M_per;
            h_sf_off[i] = i * sf_rows_per;
        }
        for (int i = 0; i < num_experts; i++) {
            h_prob[i*3] = M_per; h_prob[i*3+1] = N; h_prob[i*3+2] = K;
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
        CUDA_CHECK(cudaMalloc(&d_sA, h_sA.size()));
        CUDA_CHECK(cudaMalloc(&d_sB, h_sB.size()));
        CUDA_CHECK(cudaMalloc(&d_alphas, h_alphas.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_off, h_off.size() * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_sf_off, h_sf_off.size() * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_prob, h_prob.size() * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_ws, ws_size));

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sA, h_sA.data(), h_sA.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sB, h_sB.data(), h_sB.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_alphas, h_alphas.data(), h_alphas.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_off, h_off.data(), h_off.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sf_off, h_sf_off.data(), h_sf_off.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_prob, h_prob.data(), h_prob.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_D, 0, out_bytes));

        lc::Nvfp4GroupedGemmParams params;
        params.num_experts = num_experts;
        params.N = N; params.K = K;
        params.A_base = d_A; params.B_base = d_B; params.D_base = d_D;
        params.scale_A_base = d_sA; params.scale_B_base = d_sB;
        params.alphas = d_alphas;
        params.expert_offsets = d_off; params.sf_offsets = d_sf_off;
        params.problem_sizes = d_prob;
        params.output_dtype = lc::GemmOutputDtype::kBFloat16;

        lc::launch_nvfp4_grouped_gemm(params, d_ws, ws_size, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint16_t> h_D(total_M * N);
        CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, out_bytes, cudaMemcpyDeviceToHost));

        bool any_nonzero = false;
        for (size_t i = 0; i < h_D.size(); i++) {
            if (bf16_to_float(h_D[i]) != 0.0f) { any_nonzero = true; break; }
        }
        EXPECT_TRUE(any_nonzero) << "GPU " << gpu << ": NVFP4 grouped GEMM output all zeros";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
        cudaFree(d_sA); cudaFree(d_sB); cudaFree(d_alphas);
        cudaFree(d_off); cudaFree(d_sf_off); cudaFree(d_prob);
        cudaFree(d_ws);
    }
}
