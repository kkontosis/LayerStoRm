// Self-contained BF16/FP16/FP32 dense GEMM kernels for SM120 — cuBLAS-free.
//
// Replaces the cublasGemmEx call sites (output head, router projection) and the
// cublasGemmStridedBatchedEx site (kv_b_v projection). All paths use FP32
// accumulation to match CUBLAS_COMPUTE_32F numerics. No host-side allocation
// and no lazy handle creation: every launch is CUDA-graph-capturable.
//
// All variants compute the row-major form:
//   C[M, N] = A[M, K] @ W[N, K]^T   (i.e. W is stored [N, K] row-major)
// which is exactly what the lm_head / router / kv_b_v GEMMs need.

#pragma once

#include <cstdint>

namespace layerstorm::compute {

// Input element type for the BF16-class GEMMs.
enum class GemmInDtype { kFloat32, kBFloat16, kFloat16 };

// Output element type.
enum class GemmAccOutDtype { kFloat32, kBFloat16, kFloat16 };

// Dense GEMM: C[M,N] = A[M,K] @ W[N,K]^T, FP32 accumulation.
//   A: [M, K] row-major, dtype `in_dtype`
//   W: [N, K] row-major, dtype `in_dtype`
//   C: [M, N] row-major, dtype `out_dtype`
// An M==1 warp-per-row GEMV fast path is used for the decode case; M>1 uses a
// register-tiled GEMM. Graph-capturable (no host allocation).
void launch_bf16_gemm_nt(void* C, const void* A, const void* W,
                         int M, int N, int K,
                         GemmInDtype in_dtype, GemmAccOutDtype out_dtype,
                         void* stream /*cudaStream_t*/);

// Strided-operand variant: identical contract to launch_bf16_gemm_nt except
// A and W carry explicit leading dimensions (in ELEMENTS, >= K): A row m
// starts at A + m*lda, W row n at W + n*ldw. C stays tight [M, N] row-major.
// Lets a caller GEMM a column block out of a wider row-major matrix (e.g. one
// slot of an interleaved [rows, n_aux*H] staging against the matching fc
// column block) without materializing a tight copy. lda == ldw == K is
// bit-identical to launch_bf16_gemm_nt (same kernels, same FP32 reduction
// order). Graph-capturable (no host allocation).
void launch_bf16_gemm_nt_strided(void* C, const void* A, const void* W,
                                 int M, int N, int K, int64_t lda, int64_t ldw,
                                 GemmInDtype in_dtype,
                                 GemmAccOutDtype out_dtype,
                                 void* stream /*cudaStream_t*/);

// Strided batched BF16 GEMM mirroring the old cublasGemmStridedBatchedEx site.
// Per batch b: C_b[m, n] = A_b[m, k] @ B_b[n, k]^T with the interleaved-head
// leading dims / strides supplied by the caller (see StridedBatchedGemmBf16Params
// in attention_device.h). BF16 in, FP32 accumulate, BF16 out.
//
//   A_b stored [m, k] row-major, leading dim lda (= k), batch stride strideA
//   B_b stored [n, k] with leading dim ldb (head-interleaved), batch stride strideB
//   C_b stored [n, m] with leading dim ldc (head-interleaved), batch stride strideC
void launch_bf16_strided_batched_gemm_nt(
    void* C, const void* A, const void* B,
    int m, int n, int k,
    int lda, int64_t strideA,
    int ldb, int64_t strideB,
    int ldc, int64_t strideC,
    int batch_count,
    void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
