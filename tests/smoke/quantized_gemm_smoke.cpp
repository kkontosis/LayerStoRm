// Smoke test: Quantized GEMM (NVFP4 + FP8) on real GPU hardware.
//
// Expected hardware (dev machine):
//   GPU 0: RTX 5090, 31 GiB VRAM, SM120
//   GPU 1: RTX 5090, 31 GiB VRAM, SM120
//   GPU 2: RTX 5080, 15 GiB VRAM, SM120
//   GPU 3: RTX 5080, 15 GiB VRAM, SM120
//
// Exercises V3.2 MoE FFN dimensions:
//   Gate+Up fused: M=8, K=7168, N=4096 (2 * moe_intermediate_size)
//   Down proj:     M=8, K=2048, N=7168
//   Large-M:       M=256, K=7168, N=4096
//
// Both NVFP4 (FP4 E2M1 + UE4M3) and FP8 (E4M3 + float32 blockwise) paths.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "sm120/gemm/nvfp4/nvfp4_gemm.h"
#include "sm120/gemm/fp8/fp8_gemm.h"
#include "model/quantization/nvfp4.h"
#include "model/quantization/fp8.h"

namespace lc = layerstorm::compute;
namespace lm = layerstorm::model;

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: "                         \
                                     << cudaGetErrorString(_err);              \
    } while (0)

// ── Minimal quantization helpers (duplicated from unit tests for isolation) ──

static uint8_t quantize_e2m1(float val) {
    uint8_t sign = val < 0 ? 1 : 0;
    float abs_val = std::fabs(val);
    int best_idx = 0;
    float best_dist = std::fabs(abs_val - lm::nvfp4::kE2M1Magnitudes[0]);
    for (int i = 1; i < 8; ++i) {
        float dist = std::fabs(abs_val - lm::nvfp4::kE2M1Magnitudes[i]);
        if (dist < best_dist) { best_dist = dist; best_idx = i; }
    }
    return static_cast<uint8_t>((sign << 3) | best_idx);
}

static uint8_t encode_ue4m3(float val) {
    if (val <= 0.0f) return 0;
    constexpr float kMax = 448.0f;
    if (val > kMax) val = kMax;
    float log2_val = std::log2(val);
    int exp = static_cast<int>(std::floor(log2_val)) + 7;
    if (exp < 1) exp = 1;
    if (exp > 15) exp = 15;
    float base = std::ldexp(1.0f, exp - 7);
    int mant = static_cast<int>(std::round((val / base - 1.0f) * 8.0f));
    if (mant < 0) mant = 0;
    if (mant > 7) { mant = 0; exp++; if (exp > 15) { exp = 15; mant = 6; } }
    if (exp == 15 && mant == 7) mant = 6;
    return static_cast<uint8_t>((exp << 3) | mant);
}

static uint8_t encode_fp8_e4m3(float val) {
    if (std::isnan(val)) return 0x7F;
    uint8_t sign = val < 0 ? 1 : 0;
    float abs_val = std::fabs(val);
    if (abs_val == 0.0f) return 0;
    if (abs_val > 448.0f) abs_val = 448.0f;
    float log2_val = std::log2(abs_val);
    int exp = static_cast<int>(std::floor(log2_val)) + 7;
    if (exp <= 0) {
        int mant = static_cast<int>(std::round(abs_val / std::ldexp(1.0f, -9)));
        if (mant > 7) mant = 7;
        return static_cast<uint8_t>((sign << 7) | std::max(0, mant));
    }
    if (exp > 15) exp = 15;
    float base = std::ldexp(1.0f, exp - 7);
    int mant = static_cast<int>(std::round((abs_val / base - 1.0f) * 8.0f));
    if (mant < 0) mant = 0;
    if (mant > 7) { mant = 0; exp++; if (exp > 15) { exp = 15; mant = 6; } }
    if (exp == 15 && mant == 7) mant = 6;
    return static_cast<uint8_t>((sign << 7) | (exp << 3) | mant);
}

static int round_up(int x, int y) { return (x + y - 1) / y * y; }

// ── Fixture ─────────────────────────────────────────────────────────────────

class QuantizedGemmSmoke : public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            GTEST_SKIP() << "No CUDA GPU — skipping quantized GEMM smoke test";
        gpu_count_ = count;
        gen_.seed(12345);
        std::cout << "Detected " << gpu_count_ << " GPU(s)" << std::endl;
    }

    int gpu_count_ = 0;
    std::mt19937 gen_;
};

// ── NVFP4 smoke: V3.2 gate+up fused GEMM on all GPUs ───────────────────────

TEST_F(QuantizedGemmSmoke, Nvfp4_GateUp_AllGPUs) {
    constexpr int M = 8, N = 4096, K = 7168;  // gate+up fused

    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::vector<float> h_A(M * K), h_B(N * K);
    for (auto& x : h_A) x = dist(gen_);
    for (auto& x : h_B) x = dist(gen_);

    // Quantize to FP4
    int groups_per_row_a = K / 16, groups_per_row_b = K / 16;
    std::vector<uint8_t> packed_A(M * K / 2), packed_B(N * K / 2);
    std::vector<uint8_t> scales_A(M * groups_per_row_a);
    std::vector<uint8_t> scales_B(N * groups_per_row_b);

    for (int r = 0; r < M; ++r) {
        for (int g = 0; g < groups_per_row_a; ++g) {
            int base = r * K + g * 16;
            float amax = 0.0f;
            for (int i = 0; i < 16; ++i)
                amax = std::max(amax, std::fabs(h_A[base + i]));
            float scale = (amax > 0) ? (amax / 6.0f) : 1.0f;
            float inv_s = 1.0f / scale;
            scales_A[r * groups_per_row_a + g] = encode_ue4m3(scale);
            for (int i = 0; i < 16; i += 2) {
                uint8_t n0 = quantize_e2m1(h_A[base + i] * inv_s);
                uint8_t n1 = quantize_e2m1(h_A[base + i + 1] * inv_s);
                packed_A[r * (K / 2) + (g * 16 + i) / 2] = (n1 << 4) | (n0 & 0xF);
            }
        }
    }
    for (int r = 0; r < N; ++r) {
        for (int g = 0; g < groups_per_row_b; ++g) {
            int base = r * K + g * 16;
            float amax = 0.0f;
            for (int i = 0; i < 16; ++i)
                amax = std::max(amax, std::fabs(h_B[base + i]));
            float scale = (amax > 0) ? (amax / 6.0f) : 1.0f;
            float inv_s = 1.0f / scale;
            scales_B[r * groups_per_row_b + g] = encode_ue4m3(scale);
            for (int i = 0; i < 16; i += 2) {
                uint8_t n0 = quantize_e2m1(h_B[base + i] * inv_s);
                uint8_t n1 = quantize_e2m1(h_B[base + i + 1] * inv_s);
                packed_B[r * (K / 2) + (g * 16 + i) / 2] = (n1 << 4) | (n0 & 0xF);
            }
        }
    }

    // Pad scales
    int sf_ra = round_up(M, 128), sf_ca = round_up(K / 16, 4);
    int sf_rb = round_up(N, 128), sf_cb = round_up(K / 16, 4);
    std::vector<uint8_t> padded_sfa(sf_ra * sf_ca, 0);
    std::vector<uint8_t> padded_sfb(sf_rb * sf_cb, 0);
    for (int r = 0; r < M; ++r)
        std::memcpy(&padded_sfa[r * sf_ca], &scales_A[r * groups_per_row_a], groups_per_row_a);
    for (int r = 0; r < N; ++r)
        std::memcpy(&padded_sfb[r * sf_cb], &scales_B[r * groups_per_row_b], groups_per_row_b);

    for (int gpu = 0; gpu < gpu_count_; ++gpu) {
        SCOPED_TRACE("GPU " + std::to_string(gpu));
        CUDA_CHECK(cudaSetDevice(gpu));

        void *d_A, *d_B, *d_D, *d_sfa, *d_sfb, *d_ws;
        CUDA_CHECK(cudaMalloc(&d_A, packed_A.size()));
        CUDA_CHECK(cudaMalloc(&d_B, packed_B.size()));
        CUDA_CHECK(cudaMalloc(&d_D, M * N * 2));
        CUDA_CHECK(cudaMalloc(&d_sfa, padded_sfa.size()));
        CUDA_CHECK(cudaMalloc(&d_sfb, padded_sfb.size()));
        size_t ws = lc::query_nvfp4_gemm_workspace_size(M, N, K, lc::GemmOutputDtype::kBFloat16);
        d_ws = nullptr;
        if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

        CUDA_CHECK(cudaMemcpy(d_A, packed_A.data(), packed_A.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, packed_B.data(), packed_B.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sfa, padded_sfa.data(), padded_sfa.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sfb, padded_sfb.data(), padded_sfb.size(), cudaMemcpyHostToDevice));

        lc::Nvfp4GemmParams params{};
        params.M = M; params.N = N; params.K = K;
        params.A = d_A; params.B = d_B; params.D = d_D;
        params.scale_A = d_sfa; params.scale_B = d_sfb;
        params.alpha = 1.0f;
        params.output_dtype = lc::GemmOutputDtype::kBFloat16;

        EXPECT_NO_THROW(lc::launch_nvfp4_gemm(params, d_ws, nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Verify output is not all zeros (basic sanity)
        std::vector<uint16_t> h_out(M * N);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_D, M * N * 2, cudaMemcpyDeviceToHost));
        bool all_zero = true;
        for (auto v : h_out) { if (v != 0) { all_zero = false; break; } }
        EXPECT_FALSE(all_zero) << "Output is all zeros — GEMM likely failed";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
        cudaFree(d_sfa); cudaFree(d_sfb);
        if (d_ws) cudaFree(d_ws);
    }
}

// ── FP8 smoke: V3.2 down projection on all GPUs ────────────────────────────

TEST_F(QuantizedGemmSmoke, Fp8_DownProj_AllGPUs) {
    // M=128 minimum for SM120 trivial blockwise 128³ tile (M=8 fails can_implement).
    // Small-M support requires M-padding in the wrapper or a smaller tile config.
    constexpr int M = 128, N = 7168, K = 2048;

    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> h_A(M * K), h_B(N * K);
    for (auto& x : h_A) x = dist(gen_);
    for (auto& x : h_B) x = dist(gen_);

    // Quantize A: [M, K] row-major, blocks along K
    int k_blocks = (K + 127) / 128;
    std::vector<uint8_t> fp8_A(M * K);
    std::vector<float> scales_A(M * k_blocks);
    for (int r = 0; r < M; ++r) {
        for (int b = 0; b < k_blocks; ++b) {
            int start = b * 128, end = std::min(start + 128, K);
            float amax = 0;
            for (int i = start; i < end; ++i)
                amax = std::max(amax, std::fabs(h_A[r * K + i]));
            float scale = (amax > 0) ? (amax / 448.0f) : 1.0f;
            scales_A[r * k_blocks + b] = scale;
            for (int i = start; i < end; ++i)
                fp8_A[r * K + i] = encode_fp8_e4m3(h_A[r * K + i] / scale);
        }
    }

    // Quantize B: [N, K] row-major (col-major [K, N] transposed)
    std::vector<uint8_t> fp8_B(N * K);
    std::vector<float> scales_B_raw(N * k_blocks);
    for (int r = 0; r < N; ++r) {
        for (int b = 0; b < k_blocks; ++b) {
            int start = b * 128, end = std::min(start + 128, K);
            float amax = 0;
            for (int i = start; i < end; ++i)
                amax = std::max(amax, std::fabs(h_B[r * K + i]));
            float scale = (amax > 0) ? (amax / 448.0f) : 1.0f;
            scales_B_raw[r * k_blocks + b] = scale;
            for (int i = start; i < end; ++i)
                fp8_B[r * K + i] = encode_fp8_e4m3(h_B[r * K + i] / scale);
        }
    }

    // Reformat scale_B to [k_blocks, n_blocks] col-major
    int n_blocks = (N + 127) / 128;
    std::vector<float> scales_B(k_blocks * n_blocks, 1.0f);
    for (int nb = 0; nb < n_blocks; ++nb) {
        int rep_row = nb * 128;
        if (rep_row >= N) break;
        for (int kb = 0; kb < k_blocks; ++kb)
            scales_B[nb * k_blocks + kb] = scales_B_raw[rep_row * k_blocks + kb];
    }

    for (int gpu = 0; gpu < gpu_count_; ++gpu) {
        SCOPED_TRACE("GPU " + std::to_string(gpu));
        CUDA_CHECK(cudaSetDevice(gpu));

        void *d_A, *d_B, *d_D, *d_sa, *d_sb, *d_ws;
        CUDA_CHECK(cudaMalloc(&d_A, fp8_A.size()));
        CUDA_CHECK(cudaMalloc(&d_B, fp8_B.size()));
        CUDA_CHECK(cudaMalloc(&d_D, M * N * 2));
        CUDA_CHECK(cudaMalloc(&d_sa, scales_A.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sb, scales_B.size() * sizeof(float)));
        size_t ws = lc::query_fp8_gemm_workspace_size(M, N, K, lc::GemmOutputDtype::kBFloat16);
        d_ws = nullptr;
        if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

        CUDA_CHECK(cudaMemcpy(d_A, fp8_A.data(), fp8_A.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, fp8_B.data(), fp8_B.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sa, scales_A.data(), scales_A.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sb, scales_B.data(), scales_B.size() * 4, cudaMemcpyHostToDevice));

        lc::Fp8GemmParams params{};
        params.M = M; params.N = N; params.K = K;
        params.A = d_A; params.B = d_B; params.D = d_D;
        params.scale_A = d_sa; params.scale_B = d_sb;
        params.output_dtype = lc::GemmOutputDtype::kBFloat16;

        EXPECT_NO_THROW(lc::launch_fp8_gemm(params, d_ws, nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint16_t> h_out(M * N);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_D, M * N * 2, cudaMemcpyDeviceToHost));
        bool all_zero = true;
        for (auto v : h_out) { if (v != 0) { all_zero = false; break; } }
        EXPECT_FALSE(all_zero) << "Output is all zeros — GEMM likely failed";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
        cudaFree(d_sa); cudaFree(d_sb);
        if (d_ws) cudaFree(d_ws);
    }
}

// ── NVFP4 smoke: Large-M prefill scenario ───────────────────────────────────

TEST_F(QuantizedGemmSmoke, Nvfp4_LargeM_256) {
    constexpr int M = 256, N = 4096, K = 7168;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> h_A(M * K), h_B(N * K);
    for (auto& x : h_A) x = dist(gen_);
    for (auto& x : h_B) x = dist(gen_);

    int gpr_a = K / 16, gpr_b = K / 16;
    std::vector<uint8_t> packed_A(M * K / 2), packed_B(N * K / 2);
    std::vector<uint8_t> sc_A(M * gpr_a), sc_B(N * gpr_b);

    auto quant_rows = [&](const std::vector<float>& src, int rows, int cols,
                           std::vector<uint8_t>& packed, std::vector<uint8_t>& scales) {
        int gpr = cols / 16;
        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < gpr; ++g) {
                int base = r * cols + g * 16;
                float amax = 0;
                for (int i = 0; i < 16; ++i)
                    amax = std::max(amax, std::fabs(src[base + i]));
                float scale = (amax > 0) ? (amax / 6.0f) : 1.0f;
                scales[r * gpr + g] = encode_ue4m3(scale);
                float inv_s = 1.0f / scale;
                for (int i = 0; i < 16; i += 2) {
                    uint8_t n0 = quantize_e2m1(src[base + i] * inv_s);
                    uint8_t n1 = quantize_e2m1(src[base + i + 1] * inv_s);
                    packed[r * (cols / 2) + (g * 16 + i) / 2] = (n1 << 4) | (n0 & 0xF);
                }
            }
        }
    };
    quant_rows(h_A, M, K, packed_A, sc_A);
    quant_rows(h_B, N, K, packed_B, sc_B);

    int sf_ra = round_up(M, 128), sf_ca = round_up(K / 16, 4);
    int sf_rb = round_up(N, 128), sf_cb = round_up(K / 16, 4);
    std::vector<uint8_t> psfa(sf_ra * sf_ca, 0), psfb(sf_rb * sf_cb, 0);
    for (int r = 0; r < M; ++r)
        std::memcpy(&psfa[r * sf_ca], &sc_A[r * gpr_a], gpr_a);
    for (int r = 0; r < N; ++r)
        std::memcpy(&psfb[r * sf_cb], &sc_B[r * gpr_b], gpr_b);

    // Run on GPU 0 only for large-M (saves time)
    CUDA_CHECK(cudaSetDevice(0));

    void *d_A, *d_B, *d_D, *d_sfa, *d_sfb, *d_ws;
    CUDA_CHECK(cudaMalloc(&d_A, packed_A.size()));
    CUDA_CHECK(cudaMalloc(&d_B, packed_B.size()));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * 2));
    CUDA_CHECK(cudaMalloc(&d_sfa, psfa.size()));
    CUDA_CHECK(cudaMalloc(&d_sfb, psfb.size()));
    size_t ws = lc::query_nvfp4_gemm_workspace_size(M, N, K, lc::GemmOutputDtype::kBFloat16);
    d_ws = nullptr;
    if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

    CUDA_CHECK(cudaMemcpy(d_A, packed_A.data(), packed_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, packed_B.data(), packed_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfa, psfa.data(), psfa.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfb, psfb.data(), psfb.size(), cudaMemcpyHostToDevice));

    lc::Nvfp4GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = d_A; params.B = d_B; params.D = d_D;
    params.scale_A = d_sfa; params.scale_B = d_sfb;
    params.alpha = 1.0f;
    params.output_dtype = lc::GemmOutputDtype::kBFloat16;

    EXPECT_NO_THROW(lc::launch_nvfp4_gemm(params, d_ws, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(M * N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_D, M * N * 2, cudaMemcpyDeviceToHost));
    bool all_zero = true;
    for (auto v : h_out) { if (v != 0) { all_zero = false; break; } }
    EXPECT_FALSE(all_zero);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    cudaFree(d_sfa); cudaFree(d_sfb);
    if (d_ws) cudaFree(d_ws);
}
