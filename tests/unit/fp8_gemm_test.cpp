// Unit tests for FP8 E4M3 blockwise-scaled GEMM on SM120.
//
// Strategy: M >= 128 (tile-aligned) to satisfy SM120 tile constraints.
// Per-block float32 scales with block_size=128 along K dimension.
// CPU reference: dequant FP8 → float, matmul, compare with GPU output.

#include "sm120/gemm/fp8/fp8_gemm.h"
#include "model/quantization/fp8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
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

static float fp16_to_float(uint16_t bits) {
    uint32_t sign = (bits >> 15) & 1;
    uint32_t exp  = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FF;
    if (exp == 0) return sign ? -std::ldexp(float(mant), -24)
                              :  std::ldexp(float(mant), -24);
    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    float val = std::ldexp(1.0f + mant / 1024.0f, int(exp) - 15);
    return sign ? -val : val;
}

// ── FP8 E4M3 quantization (uniform scale = 1.0) ────────────────────────────
// With scale=1.0, the FP8 values directly represent the numbers.
// Clamp input to [-448, 448] (FP8 E4M3 max finite).

static uint8_t encode_fp8_e4m3(float val) {
    if (std::isnan(val)) return 0x7F;
    uint8_t sign = val < 0 ? 1 : 0;
    float abs_val = std::fabs(val);
    if (abs_val == 0.0f) return 0;
    if (abs_val > 448.0f) abs_val = 448.0f;

    float log2_val = std::log2(abs_val);
    int exp = int(std::floor(log2_val)) + 7;
    if (exp <= 0) {
        int mant = int(std::round(abs_val / std::ldexp(1.0f, -9)));
        return uint8_t((sign << 7) | std::clamp(mant, 0, 7));
    }
    if (exp > 15) exp = 15;
    float base = std::ldexp(1.0f, exp - 7);
    int mant = int(std::round((abs_val / base - 1.0f) * 8.0f));
    if (mant < 0) mant = 0;
    if (mant > 7) { mant = 0; exp++; if (exp > 15) { exp = 15; mant = 6; } }
    if (exp == 15 && mant == 7) mant = 6;
    return uint8_t((sign << 7) | (exp << 3) | mant);
}

// CPU reference matmul: C[M,N] = A[M,K] @ B[N,K]^T
static void matmul_ref(float* C, const float* A, const float* B,
                        int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += double(A[m * K + k]) * double(B[n * K + k]);
            C[m * N + n] = float(acc);
        }
}

// ── Test fixture ────────────────────────────────────────────────────────────

class Fp8GemmTest : public ::testing::Test {
protected:
    void SetUp() override { gen_.seed(42); }

    void fill_random(std::vector<float>& v, float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        for (auto& x : v) x = dist(gen_);
    }

    std::mt19937 gen_;
};

// ── FP8 GEMM BF16: M=128, N=128, K=128, uniform scale ─────────────────────

TEST_F(Fp8GemmTest, CorrectnessBf16) {
    REQUIRES_GPU();

    constexpr int M = 128, N = 128, K = 128;

    // Random data in FP8 E4M3 range
    std::vector<float> h_A(M * K), h_B(N * K);
    fill_random(h_A, -2.0f, 2.0f);
    fill_random(h_B, -2.0f, 2.0f);

    // Quantize to FP8 E4M3 (with uniform scale=1.0)
    std::vector<uint8_t> fp8_A(M * K), fp8_B(N * K);
    for (int i = 0; i < M * K; ++i) fp8_A[i] = encode_fp8_e4m3(h_A[i]);
    for (int i = 0; i < N * K; ++i) fp8_B[i] = encode_fp8_e4m3(h_B[i]);

    // Dequant for CPU reference
    std::vector<float> deq_A(M * K), deq_B(N * K);
    for (int i = 0; i < M * K; ++i) deq_A[i] = lm::fp8_e4m3::decode(fp8_A[i]);
    for (int i = 0; i < N * K; ++i) deq_B[i] = lm::fp8_e4m3::decode(fp8_B[i]);
    std::vector<float> h_ref(M * N);
    matmul_ref(h_ref.data(), deq_A.data(), deq_B.data(), M, N, K);

    // Scale tensors: all 1.0f (uniform)
    // scale_A: [M, ceil(K/128)] = [128, 1]
    // scale_B: [ceil(K/128), ceil(N/128)] = [1, 1]
    // Allocate generous buffers to handle any CUTLASS padding
    int k_blocks = (K + 127) / 128;
    int n_blocks = (N + 127) / 128;
    std::vector<float> scale_A(M * k_blocks, 1.0f);
    std::vector<float> scale_B(k_blocks * n_blocks, 1.0f);

    void *d_A, *d_B, *d_D, *d_sa, *d_sb, *d_ws;
    CUDA_CHECK(cudaMalloc(&d_A, fp8_A.size()));
    CUDA_CHECK(cudaMalloc(&d_B, fp8_B.size()));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * 2));
    CUDA_CHECK(cudaMalloc(&d_sa, scale_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sb, scale_B.size() * sizeof(float)));
    size_t ws = lc::query_fp8_gemm_workspace_size(M, N, K, lc::GemmOutputDtype::kBFloat16);
    d_ws = nullptr;
    if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

    CUDA_CHECK(cudaMemcpy(d_A, fp8_A.data(), fp8_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, fp8_B.data(), fp8_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sa, scale_A.data(), scale_A.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sb, scale_B.data(), scale_B.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D, 0, M * N * 2));

    lc::Fp8GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = d_A; params.B = d_B; params.D = d_D;
    params.scale_A = d_sa; params.scale_B = d_sb;
    params.output_dtype = lc::GemmOutputDtype::kBFloat16;

    ASSERT_NO_THROW(lc::launch_fp8_gemm(params, d_ws, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(M * N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_D, M * N * 2, cudaMemcpyDeviceToHost));

    float max_rel_err = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < M * N; ++i) {
        float gpu = bf16_to_float(h_out[i]);
        float ref = h_ref[i];
        float err = std::fabs(gpu - ref);
        float denom = std::max(1.0f, std::fabs(ref));
        float rel = err / denom;
        max_rel_err = std::max(max_rel_err, rel);
        if (rel > 0.05f) ++mismatches;
    }
    EXPECT_LT(mismatches, M * N / 10)
        << "Too many mismatches. Max relative error: " << max_rel_err;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    cudaFree(d_sa); cudaFree(d_sb);
    if (d_ws) cudaFree(d_ws);
}

// ── FP8 GEMM BF16: non-square (N ≠ K) — verifies layout with asymmetric dims ─

TEST_F(Fp8GemmTest, CorrectnessNonSquareBf16) {
    REQUIRES_GPU();

    constexpr int M = 128, N = 256, K = 128;

    std::vector<float> h_A(M * K), h_B(N * K);
    fill_random(h_A, -2.0f, 2.0f);
    fill_random(h_B, -2.0f, 2.0f);

    std::vector<uint8_t> fp8_A(M * K), fp8_B(N * K);
    for (int i = 0; i < M * K; ++i) fp8_A[i] = encode_fp8_e4m3(h_A[i]);
    for (int i = 0; i < N * K; ++i) fp8_B[i] = encode_fp8_e4m3(h_B[i]);

    // CPU reference uses row-major B
    std::vector<float> deq_A(M * K), deq_B(N * K);
    for (int i = 0; i < M * K; ++i) deq_A[i] = lm::fp8_e4m3::decode(fp8_A[i]);
    for (int i = 0; i < N * K; ++i) deq_B[i] = lm::fp8_e4m3::decode(fp8_B[i]);
    std::vector<float> h_ref(M * N);
    matmul_ref(h_ref.data(), deq_A.data(), deq_B.data(), M, N, K);

    int k_blocks = (K + 127) / 128;
    int n_blocks = (N + 127) / 128;
    std::vector<float> scale_A(M * k_blocks, 1.0f);
    std::vector<float> scale_B(k_blocks * n_blocks, 1.0f);

    void *d_A, *d_B, *d_D, *d_sa, *d_sb, *d_ws;
    CUDA_CHECK(cudaMalloc(&d_A, fp8_A.size()));
    CUDA_CHECK(cudaMalloc(&d_B, fp8_B.size()));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * 2));
    CUDA_CHECK(cudaMalloc(&d_sa, scale_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sb, scale_B.size() * sizeof(float)));
    size_t ws = lc::query_fp8_gemm_workspace_size(M, N, K, lc::GemmOutputDtype::kBFloat16);
    d_ws = nullptr;
    if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

    CUDA_CHECK(cudaMemcpy(d_A, fp8_A.data(), fp8_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, fp8_B.data(), fp8_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sa, scale_A.data(), scale_A.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sb, scale_B.data(), scale_B.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D, 0, M * N * 2));

    lc::Fp8GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = d_A; params.B = d_B; params.D = d_D;
    params.scale_A = d_sa; params.scale_B = d_sb;
    params.output_dtype = lc::GemmOutputDtype::kBFloat16;

    ASSERT_NO_THROW(lc::launch_fp8_gemm(params, d_ws, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(M * N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_D, M * N * 2, cudaMemcpyDeviceToHost));

    float max_rel_err = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < M * N; ++i) {
        float gpu = bf16_to_float(h_out[i]);
        float ref = h_ref[i];
        float err = std::fabs(gpu - ref);
        float denom = std::max(1.0f, std::fabs(ref));
        float rel = err / denom;
        max_rel_err = std::max(max_rel_err, rel);
        if (rel > 0.05f) ++mismatches;
    }
    EXPECT_LT(mismatches, M * N / 10)
        << "Non-square col-major B. Max relative error: " << max_rel_err;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    cudaFree(d_sa); cudaFree(d_sb);
    if (d_ws) cudaFree(d_ws);
}

// ── FP8 GEMM FP16 output ───────────────────────────────────────────────────

TEST_F(Fp8GemmTest, CorrectnessFp16) {
    REQUIRES_GPU();

    constexpr int M = 128, N = 128, K = 128;

    std::vector<float> h_A(M * K), h_B(N * K);
    fill_random(h_A, -2.0f, 2.0f);
    fill_random(h_B, -2.0f, 2.0f);

    std::vector<uint8_t> fp8_A(M * K), fp8_B(N * K);
    for (int i = 0; i < M * K; ++i) fp8_A[i] = encode_fp8_e4m3(h_A[i]);
    for (int i = 0; i < N * K; ++i) fp8_B[i] = encode_fp8_e4m3(h_B[i]);

    std::vector<float> deq_A(M * K), deq_B(N * K);
    for (int i = 0; i < M * K; ++i) deq_A[i] = lm::fp8_e4m3::decode(fp8_A[i]);
    for (int i = 0; i < N * K; ++i) deq_B[i] = lm::fp8_e4m3::decode(fp8_B[i]);
    std::vector<float> h_ref(M * N);
    matmul_ref(h_ref.data(), deq_A.data(), deq_B.data(), M, N, K);

    int k_blocks = (K + 127) / 128;
    int n_blocks = (N + 127) / 128;
    std::vector<float> scale_A(M * k_blocks, 1.0f);
    std::vector<float> scale_B(k_blocks * n_blocks, 1.0f);

    void *d_A, *d_B, *d_D, *d_sa, *d_sb, *d_ws;
    CUDA_CHECK(cudaMalloc(&d_A, fp8_A.size()));
    CUDA_CHECK(cudaMalloc(&d_B, fp8_B.size()));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * 2));
    CUDA_CHECK(cudaMalloc(&d_sa, scale_A.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_sb, scale_B.size() * 4));
    size_t ws = lc::query_fp8_gemm_workspace_size(M, N, K, lc::GemmOutputDtype::kFloat16);
    d_ws = nullptr;
    if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

    CUDA_CHECK(cudaMemcpy(d_A, fp8_A.data(), fp8_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, fp8_B.data(), fp8_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sa, scale_A.data(), scale_A.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sb, scale_B.data(), scale_B.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D, 0, M * N * 2));

    lc::Fp8GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = d_A; params.B = d_B; params.D = d_D;
    params.scale_A = d_sa; params.scale_B = d_sb;
    params.output_dtype = lc::GemmOutputDtype::kFloat16;

    ASSERT_NO_THROW(lc::launch_fp8_gemm(params, d_ws, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(M * N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_D, M * N * 2, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < M * N; ++i) {
        float gpu = fp16_to_float(h_out[i]);
        float ref = h_ref[i];
        float err = std::fabs(gpu - ref);
        float denom = std::max(1.0f, std::fabs(ref));
        if (err / denom > 0.05f) ++mismatches;
    }
    EXPECT_LT(mismatches, M * N / 10);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    cudaFree(d_sa); cudaFree(d_sb);
    if (d_ws) cudaFree(d_ws);
}

// ── Workspace query ─────────────────────────────────────────────────────────

TEST_F(Fp8GemmTest, WorkspaceQuery) {
    REQUIRES_GPU();
    size_t ws1 = lc::query_fp8_gemm_workspace_size(128, 128, 128, lc::GemmOutputDtype::kBFloat16);
    size_t ws2 = lc::query_fp8_gemm_workspace_size(128, 128, 128, lc::GemmOutputDtype::kBFloat16);
    EXPECT_EQ(ws1, ws2);
    lc::query_fp8_gemm_workspace_size(128, 128, 128, lc::GemmOutputDtype::kFloat16);
}

// ── Alignment violation ─────────────────────────────────────────────────────

TEST_F(Fp8GemmTest, AlignmentViolation) {
    REQUIRES_GPU();
    lc::Fp8GemmParams params{};
    params.M = 128; params.N = 100; params.K = 128;
    params.output_dtype = lc::GemmOutputDtype::kBFloat16;
    EXPECT_THROW(lc::launch_fp8_gemm(params, nullptr, nullptr), std::runtime_error);
}
