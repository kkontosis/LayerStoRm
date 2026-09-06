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
//
// P-29 step 13 addendum (M-independence of the speculative-verify batch): the
// verify pass pushes M = 3 draft rows through the BF16 output head
// (launch_output_head -> launch_bf16_gemm_nt, FP32 out) and through the BF16
// eh_proj (BF16 out) in ONE call and requires each row to be byte-identical to
// its own M == 1 launch. M = 3 was not on the shape ladder above and the
// BF16-out route the header claims was never actually exercised, so both are
// added here, together with a NEGATIVE CONTROL proving the byte comparison can
// fail and that the rows are genuinely independent.

#include "compute/kernels/sm120/gemm/bf16_gemm.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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

// Random A [M, K] / W [N, K] for a case. Shared by the identity cases and the
// negative control so both see the same generator.
void make_inputs(int M, int N, int K, uint32_t seed,
                 std::vector<__nv_bfloat16>& A, std::vector<__nv_bfloat16>& W) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    A.resize(static_cast<size_t>(M) * K);
    W.resize(static_cast<size_t>(N) * K);
    for (auto& v : A) v = __float2bfloat16(d(rng));
    for (auto& v : W) v = __float2bfloat16(d(rng));
}

// One M > 1 launch through the small-M route; returns C as raw output bytes.
// OutT selects the FP32 (output head / router) or BF16 (eh_proj) store path.
template <typename OutT>
std::vector<OutT> launch_rows(const __nv_bfloat16* dA, const __nv_bfloat16* dW,
                              int M, int N, int K, GemmAccOutDtype out_dtype) {
    OutT* dC = nullptr;
    cudaMalloc(&dC, static_cast<size_t>(M) * N * sizeof(OutT));
    launch_bf16_gemm_nt(dC, dA, dW, M, N, K, GemmInDtype::kBFloat16, out_dtype,
                        /*stream=*/nullptr);
    std::vector<OutT> C(static_cast<size_t>(M) * N);
    cudaMemcpy(C.data(), dC, C.size() * sizeof(OutT), cudaMemcpyDeviceToHost);
    cudaFree(dC);
    return C;
}

// Contract 1 for one output dtype: multi-row launch == per-row M == 1 launches,
// BYTE for byte.
template <typename OutT>
void check_row_identity(const __nv_bfloat16* dA, const __nv_bfloat16* dW,
                        int M, int N, int K, GemmAccOutDtype out_dtype,
                        const char* out_name) {
    SCOPED_TRACE(std::string("out=") + out_name);
    const std::vector<OutT> C = launch_rows<OutT>(dA, dW, M, N, K, out_dtype);
    for (int m = 0; m < M; ++m) {
        const std::vector<OutT> row = launch_rows<OutT>(
            dA + static_cast<size_t>(m) * K, dW, /*M=*/1, N, K, out_dtype);
        for (int n = 0; n < N; ++n) {
            ASSERT_EQ(std::memcmp(&row[n], &C[static_cast<size_t>(m) * N + n],
                                  sizeof(OutT)), 0)
                << "row " << m << " col " << n;
        }
    }
}

void run_case(int M, int N, int K, uint32_t seed) {
    SCOPED_TRACE("M=" + std::to_string(M) + " N=" + std::to_string(N)
                 + " K=" + std::to_string(K));
    std::vector<__nv_bfloat16> A, W;
    make_inputs(M, N, K, seed, A, W);

    auto* dA = upload(A);
    auto* dW = upload(W);

    // Contract 1: bit-identical to M == 1 GEMV launches per row, on BOTH the
    // FP32-out (output head / router) and BF16-out (eh_proj) store paths.
    check_row_identity<float>(dA, dW, M, N, K, GemmAccOutDtype::kFloat32,
                              "fp32");
    check_row_identity<__nv_bfloat16>(dA, dW, M, N, K,
                                      GemmAccOutDtype::kBFloat16, "bf16");

    // Contract 2: CPU float reference within tolerance.
    const std::vector<float> C =
        launch_rows<float>(dA, dW, M, N, K, GemmAccOutDtype::kFloat32);
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

    cudaFree(dA); cudaFree(dW);
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

// P-29 step 13: the speculative-verify batch shape (M = 3) on the output-head
// geometry (N = vocab slice, K = hidden) and on a small eh_proj-like shape.
TEST_F(Bf16GemmSmallM, VerifyBatchM3OutputHead) {
    run_case(/*M=*/3, /*N=*/1024, /*K=*/2048, 11);  // output head slice
    run_case(/*M=*/3, /*N=*/129,  /*K=*/512, 12);   // eh_proj-ish, odd N tail
}

// P-29 step 13 NEGATIVE CONTROL. Perturb ONE element of the middle row's
// activation and re-launch at M = 3: that row's output bytes MUST change
// (so the byte comparison above is not vacuous) and the two neighbour rows
// MUST stay byte-identical (so the rows really are independent, which is the
// property the verify pass relies on).
TEST_F(Bf16GemmSmallM, VerifyBatchM3NegativeControl) {
    constexpr int M = 3, N = 256, K = 1024;
    std::vector<__nv_bfloat16> A, W;
    make_inputs(M, N, K, 21, A, W);

    auto* dW = upload(W);
    auto* dA = upload(A);
    const std::vector<float> C0 =
        launch_rows<float>(dA, dW, M, N, K, GemmAccOutDtype::kFloat32);
    cudaFree(dA);

    // A real BF16 step, not a rounding no-op.
    std::vector<__nv_bfloat16> A1 = A;
    const size_t idx = static_cast<size_t>(1) * K + 7;
    A1[idx] = __float2bfloat16(__bfloat162float(A[idx]) + 0.5f);
    ASSERT_NE(std::memcmp(&A1[idx], &A[idx], sizeof(__nv_bfloat16)), 0);

    auto* dA1 = upload(A1);
    const std::vector<float> C1 =
        launch_rows<float>(dA1, dW, M, N, K, GemmAccOutDtype::kFloat32);
    cudaFree(dA1);
    cudaFree(dW);

    bool differs = false;
    for (int n = 0; n < N && !differs; ++n)
        differs = std::memcmp(&C0[static_cast<size_t>(1) * N + n],
                              &C1[static_cast<size_t>(1) * N + n],
                              sizeof(float)) != 0;
    EXPECT_TRUE(differs) << "negative control vacuous: perturbing row 1's "
                            "input left row 1's output unchanged";

    for (int m : {0, 2}) {
        for (int n = 0; n < N; ++n) {
            ASSERT_EQ(std::memcmp(&C0[static_cast<size_t>(m) * N + n],
                                  &C1[static_cast<size_t>(m) * N + n],
                                  sizeof(float)), 0)
                << "row " << m << " changed when row 1's input was perturbed";
        }
    }
}

}  // namespace
