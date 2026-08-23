// Unit tests for NVFP4 (FP4 E2M1) GEMM on SM120.
//
// Strategy: use UNIFORM scales (all bytes = UE4M3(1.0) = 0x38) so the
// CUTLASS swizzled scale layout doesn't affect results (every position reads
// the same value). This lets us verify the FP4 data path and matmul logic.
// The dequantized value = decode_e2m1(nibble) * 1.0 = decode_e2m1(nibble).
// CPU reference: decode each FP4 nibble → float, matmul, compare.

#include "sm120/gemm/nvfp4/nvfp4_gemm.h"
#include "model/quantization/nvfp4.h"

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
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
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

// ── FP4 quantization with UNIFORM scale = 1.0 ──────────────────────────────
// All scale bytes set to UE4M3(1.0) = 0x38 (E=7, M=0 → 2^(7-7) * 1.0 = 1.0).
// Input values clamped to E2M1 range [-6.0, 6.0] before quantization.
// Two nibbles per byte: low nibble = even element, high nibble = odd element.

static constexpr uint8_t kUE4M3_One = 0x38;  // UE4M3 encoding of 1.0

static uint8_t quantize_e2m1(float val) {
    uint8_t sign = val < 0 ? 1 : 0;
    float abs_val = std::fabs(val);
    int best = 0;
    float best_d = std::fabs(abs_val - lm::nvfp4::kE2M1Magnitudes[0]);
    for (int i = 1; i < 8; ++i) {
        float d = std::fabs(abs_val - lm::nvfp4::kE2M1Magnitudes[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return static_cast<uint8_t>((sign << 3) | best);
}

// Pack row-major float data into FP4 packed bytes with scale=1.0.
// Returns packed bytes [rows, cols/2] — two nibbles per byte.
static std::vector<uint8_t> pack_fp4_unit_scale(const std::vector<float>& data,
                                                  int rows, int cols) {
    std::vector<uint8_t> packed(static_cast<size_t>(rows) * (cols / 2));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; c += 2) {
            uint8_t n0 = quantize_e2m1(data[r * cols + c]);
            uint8_t n1 = quantize_e2m1(data[r * cols + c + 1]);
            packed[r * (cols / 2) + c / 2] = (n1 << 4) | (n0 & 0xF);
        }
    }
    return packed;
}

// Dequantize FP4 packed bytes (scale=1.0) → float for CPU reference.
static std::vector<float> unpack_fp4_unit_scale(const std::vector<uint8_t>& packed,
                                                  int rows, int cols) {
    std::vector<float> out(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; c += 2) {
            uint8_t byte = packed[r * (cols / 2) + c / 2];
            out[r * cols + c]     = lm::nvfp4::decode_e2m1(byte & 0xF);
            out[r * cols + c + 1] = lm::nvfp4::decode_e2m1((byte >> 4) & 0xF);
        }
    }
    return out;
}

// CPU reference: C[M,N] = alpha * A[M,K] @ B[N,K]^T
static void matmul_ref(float* C, const float* A, const float* B,
                        int M, int N, int K, float alpha) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += double(A[m * K + k]) * double(B[n * K + k]);
            C[m * N + n] = float(acc * alpha);
        }
    }
}

static int round_up(int x, int y) { return (x + y - 1) / y * y; }

// ── Test fixture ────────────────────────────────────────────────────────────

class Nvfp4GemmTest : public ::testing::Test {
protected:
    void SetUp() override { gen_.seed(42); }

    void fill_random(std::vector<float>& v, float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        for (auto& x : v) x = dist(gen_);
    }

    std::mt19937 gen_;
};

// ── NVFP4 GEMM BF16: M=128, N=128, K=128, uniform scale ───────────────────

TEST_F(Nvfp4GemmTest, CorrectnessBf16) {
    REQUIRES_GPU();

    constexpr int M = 128, N = 128, K = 128;
    constexpr float alpha = 1.0f;

    // Random data in E2M1 range [-6, 6]
    std::vector<float> h_A(M * K), h_B(N * K);
    fill_random(h_A, -5.0f, 5.0f);
    fill_random(h_B, -5.0f, 5.0f);

    // Pack to FP4 with scale=1.0
    auto packed_A = pack_fp4_unit_scale(h_A, M, K);
    auto packed_B = pack_fp4_unit_scale(h_B, N, K);

    // CPU reference from decoded FP4
    auto deq_A = unpack_fp4_unit_scale(packed_A, M, K);
    auto deq_B = unpack_fp4_unit_scale(packed_B, N, K);
    std::vector<float> h_ref(M * N);
    matmul_ref(h_ref.data(), deq_A.data(), deq_B.data(), M, N, K, alpha);

    // Uniform scale buffers (all bytes = UE4M3(1.0))
    // Generous size: round_up(dim, 128) * round_up(K/16, 4)
    int sfa_size = round_up(M, 128) * round_up(K / 16, 4);
    int sfb_size = round_up(N, 128) * round_up(K / 16, 4);
    std::vector<uint8_t> scale_A(sfa_size, kUE4M3_One);
    std::vector<uint8_t> scale_B(sfb_size, kUE4M3_One);

    // GPU alloc + copy
    void *d_A, *d_B, *d_D, *d_sfa, *d_sfb, *d_ws;
    CUDA_CHECK(cudaMalloc(&d_A, packed_A.size()));
    CUDA_CHECK(cudaMalloc(&d_B, packed_B.size()));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * 2));
    CUDA_CHECK(cudaMalloc(&d_sfa, scale_A.size()));
    CUDA_CHECK(cudaMalloc(&d_sfb, scale_B.size()));
    size_t ws = lc::query_nvfp4_gemm_workspace_size(M, N, K, lc::GemmOutputDtype::kBFloat16);
    d_ws = nullptr;
    if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

    CUDA_CHECK(cudaMemcpy(d_A, packed_A.data(), packed_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, packed_B.data(), packed_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfa, scale_A.data(), scale_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfb, scale_B.data(), scale_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D, 0, M * N * 2));

    lc::Nvfp4GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = d_A; params.B = d_B; params.D = d_D;
    params.scale_A = d_sfa; params.scale_B = d_sfb;
    params.alpha = alpha;
    params.output_dtype = lc::GemmOutputDtype::kBFloat16;

    ASSERT_NO_THROW(lc::launch_nvfp4_gemm(params, d_ws, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(M * N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_D, M * N * 2, cudaMemcpyDeviceToHost));

    // Verify: FP4 quantization error + BF16 output truncation
    float max_rel_err = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < M * N; ++i) {
        float gpu = bf16_to_float(h_out[i]);
        float ref = h_ref[i];
        float err = std::fabs(gpu - ref);
        float denom = std::max(1.0f, std::fabs(ref));
        float rel = err / denom;
        max_rel_err = std::max(max_rel_err, rel);
        if (rel > 0.2f) ++mismatches;
    }
    EXPECT_LT(mismatches, M * N / 10)
        << "Too many mismatches. Max relative error: " << max_rel_err;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    cudaFree(d_sfa); cudaFree(d_sfb);
    if (d_ws) cudaFree(d_ws);
}

// ── NVFP4 GEMM BF16: non-square (N ≠ K) — verifies layout with asymmetric dims

TEST_F(Nvfp4GemmTest, CorrectnessNonSquareBf16) {
    REQUIRES_GPU();

    constexpr int M = 128, N = 256, K = 128;
    constexpr float alpha = 1.0f;

    std::vector<float> h_A(M * K), h_B(N * K);
    fill_random(h_A, -5.0f, 5.0f);
    fill_random(h_B, -5.0f, 5.0f);

    auto packed_A = pack_fp4_unit_scale(h_A, M, K);
    auto packed_B = pack_fp4_unit_scale(h_B, N, K);

    auto deq_A = unpack_fp4_unit_scale(packed_A, M, K);
    auto deq_B = unpack_fp4_unit_scale(packed_B, N, K);
    std::vector<float> h_ref(M * N);
    matmul_ref(h_ref.data(), deq_A.data(), deq_B.data(), M, N, K, alpha);

    int sfa_size = round_up(M, 128) * round_up(K / 16, 4);
    int sfb_size = round_up(N, 128) * round_up(K / 16, 4);
    std::vector<uint8_t> scale_A(sfa_size, kUE4M3_One);
    std::vector<uint8_t> scale_B(sfb_size, kUE4M3_One);

    void *d_A, *d_B, *d_D, *d_sfa, *d_sfb, *d_ws;
    CUDA_CHECK(cudaMalloc(&d_A, packed_A.size()));
    CUDA_CHECK(cudaMalloc(&d_B, packed_B.size()));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * 2));
    CUDA_CHECK(cudaMalloc(&d_sfa, scale_A.size()));
    CUDA_CHECK(cudaMalloc(&d_sfb, scale_B.size()));
    size_t ws = lc::query_nvfp4_gemm_workspace_size(M, N, K, lc::GemmOutputDtype::kBFloat16);
    d_ws = nullptr;
    if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

    CUDA_CHECK(cudaMemcpy(d_A, packed_A.data(), packed_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, packed_B.data(), packed_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfa, scale_A.data(), scale_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfb, scale_B.data(), scale_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D, 0, M * N * 2));

    lc::Nvfp4GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = d_A; params.B = d_B; params.D = d_D;
    params.scale_A = d_sfa; params.scale_B = d_sfb;
    params.alpha = alpha;
    params.output_dtype = lc::GemmOutputDtype::kBFloat16;

    ASSERT_NO_THROW(lc::launch_nvfp4_gemm(params, d_ws, nullptr));
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
        if (rel > 0.2f) ++mismatches;
    }
    EXPECT_LT(mismatches, M * N / 10)
        << "Non-square col-major B. Max relative error: " << max_rel_err;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    cudaFree(d_sfa); cudaFree(d_sfb);
    if (d_ws) cudaFree(d_ws);
}

// ── NVFP4 GEMM FP16 output ─────────────────────────────────────────────────

TEST_F(Nvfp4GemmTest, CorrectnessFp16) {
    REQUIRES_GPU();

    constexpr int M = 128, N = 128, K = 128;
    constexpr float alpha = 1.0f;

    std::vector<float> h_A(M * K), h_B(N * K);
    fill_random(h_A, -5.0f, 5.0f);
    fill_random(h_B, -5.0f, 5.0f);

    auto packed_A = pack_fp4_unit_scale(h_A, M, K);
    auto packed_B = pack_fp4_unit_scale(h_B, N, K);
    auto deq_A = unpack_fp4_unit_scale(packed_A, M, K);
    auto deq_B = unpack_fp4_unit_scale(packed_B, N, K);
    std::vector<float> h_ref(M * N);
    matmul_ref(h_ref.data(), deq_A.data(), deq_B.data(), M, N, K, alpha);

    int sfa_size = round_up(M, 128) * round_up(K / 16, 4);
    int sfb_size = round_up(N, 128) * round_up(K / 16, 4);
    std::vector<uint8_t> scale_A(sfa_size, kUE4M3_One);
    std::vector<uint8_t> scale_B(sfb_size, kUE4M3_One);

    void *d_A, *d_B, *d_D, *d_sfa, *d_sfb, *d_ws;
    CUDA_CHECK(cudaMalloc(&d_A, packed_A.size()));
    CUDA_CHECK(cudaMalloc(&d_B, packed_B.size()));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * 2));
    CUDA_CHECK(cudaMalloc(&d_sfa, scale_A.size()));
    CUDA_CHECK(cudaMalloc(&d_sfb, scale_B.size()));
    size_t ws = lc::query_nvfp4_gemm_workspace_size(M, N, K, lc::GemmOutputDtype::kFloat16);
    d_ws = nullptr;
    if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

    CUDA_CHECK(cudaMemcpy(d_A, packed_A.data(), packed_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, packed_B.data(), packed_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfa, scale_A.data(), scale_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfb, scale_B.data(), scale_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D, 0, M * N * 2));

    lc::Nvfp4GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = d_A; params.B = d_B; params.D = d_D;
    params.scale_A = d_sfa; params.scale_B = d_sfb;
    params.alpha = alpha;
    params.output_dtype = lc::GemmOutputDtype::kFloat16;

    ASSERT_NO_THROW(lc::launch_nvfp4_gemm(params, d_ws, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(M * N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_D, M * N * 2, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < M * N; ++i) {
        float gpu = fp16_to_float(h_out[i]);
        float ref = h_ref[i];
        float err = std::fabs(gpu - ref);
        float denom = std::max(1.0f, std::fabs(ref));
        if (err / denom > 0.2f) ++mismatches;
    }
    EXPECT_LT(mismatches, M * N / 10);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    cudaFree(d_sfa); cudaFree(d_sfb);
    if (d_ws) cudaFree(d_ws);
}

// ── Workspace query ─────────────────────────────────────────────────────────

TEST_F(Nvfp4GemmTest, WorkspaceQuery) {
    REQUIRES_GPU();

    size_t ws1 = lc::query_nvfp4_gemm_workspace_size(128, 128, 128,
                                                       lc::GemmOutputDtype::kBFloat16);
    size_t ws2 = lc::query_nvfp4_gemm_workspace_size(128, 128, 128,
                                                       lc::GemmOutputDtype::kBFloat16);
    EXPECT_EQ(ws1, ws2) << "Must be deterministic";
    // FP16 should also work
    lc::query_nvfp4_gemm_workspace_size(128, 128, 128, lc::GemmOutputDtype::kFloat16);
}

// ── Alignment violation ─────────────────────────────────────────────────────

TEST_F(Nvfp4GemmTest, AlignmentViolation) {
    REQUIRES_GPU();

    lc::Nvfp4GemmParams params{};
    params.M = 128; params.N = 100; params.K = 128;  // N not div by 32
    params.output_dtype = lc::GemmOutputDtype::kBFloat16;
    params.alpha = 1.0f;

    EXPECT_THROW(lc::launch_nvfp4_gemm(params, nullptr, nullptr),
                 std::runtime_error);
}
