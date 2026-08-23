// Small-M GEMV route (bf16_gemm.cu) — GPU numerical test.
//
// The speculative-verify chunk fix routes 1 < M <= 32 launch_bf16_gemm_nt
// calls (fused-gate router projection shape and friends) through
// bf16_gemv_nt_mrows_kernel instead of the tiled bf16_gemm_nt_kernel. It is
// the DEFAULT since 2026-08-17 (TD-CHUNK-SMALLM-DEFAULT resolved); the test
// still pins the gate explicitly (clearing the LS_NO_CHUNK_SMALLM escape
// hatch) so it asserts the route itself, not the current default. Two
// contracts are asserted here, with the route pinned ON before the first
// launch (the gate is read once):
//
//   1. BIT-IDENTITY to the M == 1 GEMV: per output row the multi-row kernel
//      keeps the exact lane-strided K + shfl_down reduction order of
//      bf16_gemv_nt_kernel, so slicing A row-by-row through M == 1 launches
//      must produce byte-identical outputs (both FP32-out and BF16-out).
//   2. CPU float reference within tolerance (sanity against both kernels
//      simply being wrong the same way).
//
// Shapes cover the measured hot case (router: M=16, N=256, K=6144), an
// MT-partial tail (M=12), the crossover edges (M=2, M=32), and a non-multiple
// N (odd warp tail). GPU-required; SKIPPED headless.

#include "compute/kernels/sm120/gemm/bf16_gemm.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace layerstorm::compute;

namespace {

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

void run_case(int M, int N, int K, uint32_t seed) {
    SCOPED_TRACE("M=" + std::to_string(M) + " N=" + std::to_string(N)
                 + " K=" + std::to_string(K));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    std::vector<__nv_bfloat16> A(static_cast<size_t>(M) * K);
    std::vector<__nv_bfloat16> W(static_cast<size_t>(N) * K);
    for (auto& v : A) v = __float2bfloat16(d(rng));
    for (auto& v : W) v = __float2bfloat16(d(rng));

    auto* dA = upload(A);
    auto* dW = upload(W);
    float* dC = nullptr;
    cudaMalloc(&dC, static_cast<size_t>(M) * N * sizeof(float));
    float* dC_row = nullptr;  // per-row M==1 reference launches
    cudaMalloc(&dC_row, static_cast<size_t>(N) * sizeof(float));

    // Multi-row route (flag forced ON in SetUp → M in (1,32] takes the
    // mrows kernel).
    launch_bf16_gemm_nt(dC, dA, dW, M, N, K, GemmInDtype::kBFloat16,
                        GemmAccOutDtype::kFloat32, /*stream=*/nullptr);
    std::vector<float> C(static_cast<size_t>(M) * N);
    cudaMemcpy(C.data(), dC, C.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Contract 1: bit-identical to M == 1 GEMV launches per row.
    for (int m = 0; m < M; ++m) {
        launch_bf16_gemm_nt(dC_row, dA + static_cast<size_t>(m) * K, dW,
                            /*M=*/1, N, K, GemmInDtype::kBFloat16,
                            GemmAccOutDtype::kFloat32, nullptr);
        std::vector<float> row(N);
        cudaMemcpy(row.data(), dC_row, N * sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (int n = 0; n < N; ++n) {
            ASSERT_EQ(std::memcmp(&row[n], &C[static_cast<size_t>(m) * N + n],
                                  sizeof(float)), 0)
                << "row " << m << " col " << n << ": mrows="
                << C[static_cast<size_t>(m) * N + n] << " gemv=" << row[n];
        }
    }

    // Contract 2: CPU float reference within tolerance.
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += double(__bfloat162float(A[static_cast<size_t>(m) * K + k]))
                     * double(__bfloat162float(W[static_cast<size_t>(n) * K + k]));
            const float got = C[static_cast<size_t>(m) * N + n];
            EXPECT_NEAR(got, static_cast<float>(acc),
                        1e-2 + 2e-3 * std::abs(acc))
                << "row " << m << " col " << n;
        }
    }

    cudaFree(dA); cudaFree(dW); cudaFree(dC); cudaFree(dC_row);
}

class Bf16GemmSmallM : public ::testing::Test {
protected:
    void SetUp() override {
        REQUIRES_GPU();
        // Pin the small-M route ON before the launcher's one-time env read
        // (default ON since 2026-08-17; clear the inverse escape hatch too so
        // an ambient LS_NO_CHUNK_SMALLM cannot silently skip the route).
        ::unsetenv("LS_NO_CHUNK_SMALLM");
        ::setenv("LS_CHUNK_SMALLM", "1", /*overwrite=*/1);
    }
};

TEST_F(Bf16GemmSmallM, RouterShapeAndEdges) {
    run_case(/*M=*/16, /*N=*/256, /*K=*/6144, 1);   // measured hot case
    run_case(/*M=*/12, /*N=*/160, /*K=*/1024, 2);   // MT-partial tail
    run_case(/*M=*/2,  /*N=*/96,  /*K=*/512, 3);    // lower crossover edge
    run_case(/*M=*/32, /*N=*/61,  /*K=*/768, 4);    // upper edge, odd N
}

}  // namespace
