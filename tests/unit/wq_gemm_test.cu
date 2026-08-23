// Fused weight-dequant GEMM unit tests (TD-DSPARK-DRAFT-QUANT).
//
// GPU-required. For both storage formats (FP8-E4M3 group-128, NVFP4
// group-16) and all three kernel shapes (M == 1 GEMV, 1 < M <= 32 multi-row
// GEMV, M > 32 tiled), launch_wq_gemm_nt must match a CPU float reference
// built from the SAME kgroup packing (dequantize_rows_*): kernel and CPU
// dequant are bit-matched, so residual error is FP32 accumulation order
// only. Also exercises the strided/column-window form (k_off > 0 over a
// wider stored row — the DSpark chunked-fc slot GEMM shape) with a k_off
// that is NOT a multiple of the scale group (ragged window).

#include "compute/kernels/sm120/gemm/wq_gemm.h"

#include "model/quantization/kgroup_quant.h"

#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace lc = layerstorm::compute;
namespace kg = layerstorm::model::kgroup;

namespace {

bool has_gpu() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + rounding) >> 16);
}

float bf2f(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

template <typename T>
T* to_dev(const std::vector<T>& h) {
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    return d;
}

struct Problem {
    int M, N, K;
    int64_t ldw, k_off;
};

/// Run one (kind, problem) case: quantize a random BF16 W [N, ldw], GEMM the
/// window [k_off, k_off+K) against random BF16 A, compare FP32 output vs the
/// CPU reference over the SAME dequantized weights.
void run_case(lc::WqWeightKind kind, const Problem& p, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    std::vector<uint16_t> a_bf(static_cast<size_t>(p.M) * p.K);
    for (auto& x : a_bf) x = f2bf(u(rng));
    // Weights over the full stored row; the GEMM consumes the [k_off,
    // k_off + K) column window.
    std::vector<uint16_t> w_bf(static_cast<size_t>(p.N) * p.ldw);
    for (auto& x : w_bf) x = f2bf(u(rng));

    // Pack + CPU dequant of the FULL rows.
    std::vector<uint8_t> q;
    std::vector<float> s32;
    std::vector<uint8_t> s8;
    std::vector<float> w_deq(static_cast<size_t>(p.N) * p.ldw);
    int64_t lds;
    if (kind == lc::WqWeightKind::kFp8E4M3) {
        q.resize(static_cast<size_t>(kg::fp8_weight_bytes(p.N, p.ldw)));
        s32.resize(static_cast<size_t>(kg::fp8_scale_bytes(p.N, p.ldw)) /
                   sizeof(float));
        kg::quantize_rows_fp8_e4m3(w_bf.data(), p.N, p.ldw, q.data(),
                                   s32.data());
        kg::dequantize_rows_fp8_e4m3(q.data(), s32.data(), p.N, p.ldw,
                                     w_deq.data());
        lds = (p.ldw + kg::kFp8GroupSize - 1) / kg::kFp8GroupSize;
    } else {
        q.resize(static_cast<size_t>(kg::nvfp4_weight_bytes(p.N, p.ldw)));
        s8.resize(static_cast<size_t>(kg::nvfp4_scale_bytes(p.N, p.ldw)));
        kg::quantize_rows_nvfp4(w_bf.data(), p.N, p.ldw, q.data(), s8.data());
        kg::dequantize_rows_nvfp4(q.data(), s8.data(), p.N, p.ldw,
                                  w_deq.data());
        lds = (p.ldw + kg::kNvfp4GroupSize - 1) / kg::kNvfp4GroupSize;
    }

    // CPU reference (FP32 accumulate over exact dequant values).
    std::vector<float> ref(static_cast<size_t>(p.M) * p.N);
    for (int m = 0; m < p.M; ++m)
        for (int n = 0; n < p.N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < p.K; ++k)
                acc += bf2f(a_bf[static_cast<size_t>(m) * p.K + k]) *
                       w_deq[static_cast<size_t>(n) * p.ldw + p.k_off + k];
            ref[static_cast<size_t>(m) * p.N + n] = acc;
        }

    // Device buffers.
    std::vector<__nv_bfloat16> a_dev_h(a_bf.size());
    std::memcpy(a_dev_h.data(), a_bf.data(), a_bf.size() * 2);
    auto* dA = to_dev(a_dev_h);
    auto* dQ = to_dev(q);
    float* dC = nullptr;
    ASSERT_EQ(cudaMalloc(&dC, ref.size() * sizeof(float)), cudaSuccess);
    void* dS = nullptr;
    if (kind == lc::WqWeightKind::kFp8E4M3) {
        dS = to_dev(s32);
    } else {
        dS = to_dev(s8);
    }

    lc::launch_wq_gemm_nt(dC, dA, dQ, dS, kind, p.M, p.N, p.K,
                          /*lda=*/p.K, p.ldw, p.k_off, lds,
                          lc::GemmAccOutDtype::kFloat32, /*stream=*/nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> got(ref.size());
    ASSERT_EQ(cudaMemcpy(got.data(), dC, got.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        ASSERT_TRUE(std::isfinite(got[i])) << i;
        const float e = std::fabs(got[i] - ref[i]);
        max_err = std::max(max_err, e);
        // Same dequant values on both sides; only FP32 reduction order
        // differs -> tight band.
        ASSERT_LE(e, 1e-3f + 1e-4f * std::fabs(ref[i]))
            << "elem " << i << " got " << got[i] << " ref " << ref[i]
            << " (M=" << p.M << " N=" << p.N << " K=" << p.K
            << " k_off=" << p.k_off << ")";
    }

    cudaFree(dA);
    cudaFree(dQ);
    cudaFree(dS);
    cudaFree(dC);
}

}  // namespace

TEST(WqGemm, Fp8AllKernelShapes) {
    if (!has_gpu()) GTEST_SKIP() << "No CUDA GPU";
    run_case(lc::WqWeightKind::kFp8E4M3, {1, 96, 256, 256, 0}, 1);    // GEMV
    run_case(lc::WqWeightKind::kFp8E4M3, {9, 96, 256, 256, 0}, 2);    // mrows
    run_case(lc::WqWeightKind::kFp8E4M3, {33, 70, 192, 192, 0}, 22);  // mrows edge -> tiled? (33 > 32: tiled)
    run_case(lc::WqWeightKind::kFp8E4M3, {100, 96, 256, 256, 0}, 3);  // tiled
}

TEST(WqGemm, Nvfp4AllKernelShapes) {
    if (!has_gpu()) GTEST_SKIP() << "No CUDA GPU";
    run_case(lc::WqWeightKind::kNvfp4, {1, 96, 256, 256, 0}, 4);
    run_case(lc::WqWeightKind::kNvfp4, {9, 96, 256, 256, 0}, 5);
    run_case(lc::WqWeightKind::kNvfp4, {100, 96, 256, 256, 0}, 6);
}

TEST(WqGemm, ColumnWindowStridedForm) {
    if (!has_gpu()) GTEST_SKIP() << "No CUDA GPU";
    // The DSpark chunked-fc slot shape: window K=64 of a stored row ldw=320,
    // k_off NOT a multiple of the scale group (ragged: 96 % 128 != 0,
    // 96 % 16 == 0 for nvfp4 -> also test 40 for a mid-group nvfp4 window...
    // 40 is even (NVFP4 packing needs no byte split at even offsets).
    run_case(lc::WqWeightKind::kFp8E4M3, {5, 48, 64, 320, 96}, 7);
    run_case(lc::WqWeightKind::kNvfp4, {5, 48, 64, 320, 40}, 8);
    // Odd k_off (nibble-straddling window) for NVFP4.
    run_case(lc::WqWeightKind::kNvfp4, {3, 32, 33, 320, 41}, 9);
    // Odd k_off for FP8 too (scale group straddles mid-window).
    run_case(lc::WqWeightKind::kFp8E4M3, {3, 32, 100, 320, 41}, 10);
}

TEST(WqGemm, Bf16OutputRounds) {
    if (!has_gpu()) GTEST_SKIP() << "No CUDA GPU";
    // BF16 out: identical math, stored with __float2bfloat16_rn. Just check
    // finiteness + closeness to the FP32 result at BF16 precision.
    const Problem p{9, 64, 128, 128, 0};
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<uint16_t> a_bf(static_cast<size_t>(p.M) * p.K);
    for (auto& x : a_bf) x = f2bf(u(rng));
    std::vector<uint16_t> w_bf(static_cast<size_t>(p.N) * p.ldw);
    for (auto& x : w_bf) x = f2bf(u(rng));

    std::vector<uint8_t> q(static_cast<size_t>(
        kg::fp8_weight_bytes(p.N, p.ldw)));
    std::vector<float> s(static_cast<size_t>(
        kg::fp8_scale_bytes(p.N, p.ldw)) / sizeof(float));
    kg::quantize_rows_fp8_e4m3(w_bf.data(), p.N, p.ldw, q.data(), s.data());
    std::vector<float> w_deq(static_cast<size_t>(p.N) * p.ldw);
    kg::dequantize_rows_fp8_e4m3(q.data(), s.data(), p.N, p.ldw,
                                 w_deq.data());

    std::vector<__nv_bfloat16> a_dev_h(a_bf.size());
    std::memcpy(a_dev_h.data(), a_bf.data(), a_bf.size() * 2);
    auto* dA = to_dev(a_dev_h);
    auto* dQ = to_dev(q);
    auto* dS = to_dev(s);
    __nv_bfloat16* dC = nullptr;
    ASSERT_EQ(cudaMalloc(&dC, static_cast<size_t>(p.M) * p.N * 2),
              cudaSuccess);
    const int64_t lds = (p.ldw + kg::kFp8GroupSize - 1) / kg::kFp8GroupSize;
    lc::launch_wq_gemm_nt(dC, dA, dQ, dS, lc::WqWeightKind::kFp8E4M3, p.M,
                          p.N, p.K, p.K, p.ldw, 0, lds,
                          lc::GemmAccOutDtype::kBFloat16, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint16_t> got(static_cast<size_t>(p.M) * p.N);
    ASSERT_EQ(cudaMemcpy(got.data(), dC, got.size() * 2,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int m = 0; m < p.M; ++m)
        for (int n = 0; n < p.N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < p.K; ++k)
                acc += bf2f(a_bf[static_cast<size_t>(m) * p.K + k]) *
                       w_deq[static_cast<size_t>(n) * p.ldw + k];
            const float g = bf2f(got[static_cast<size_t>(m) * p.N + n]);
            ASSERT_TRUE(std::isfinite(g));
            ASSERT_LE(std::fabs(g - acc), 0.02f + 0.01f * std::fabs(acc));
        }

    cudaFree(dA);
    cudaFree(dQ);
    cudaFree(dS);
    cudaFree(dC);
}
