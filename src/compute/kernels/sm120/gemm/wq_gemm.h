// Fused weight-dequant dense GEMM for SM120 (TD-DSPARK-DRAFT-QUANT).
//
// Sibling of launch_bf16_gemm_nt for QUANTIZED weight operands: the DSpark
// draft's GEMM projections are stored FP8-E4M3 (per-128-column-group FP32
// scales) or NVFP4 (E2M1 g=16 + UE8M0 scales) — the model/quantization/
// kgroup_quant.h packing — and dequantized IN THE K-LOOP, so no BF16
// weight scratch ever materializes (the bandwidth win is the point).
//
// Row-major contract, matching launch_bf16_gemm_nt:
//   C[M, N] = A[M, K] @ W[N, K]^T
//     A: [M, K] BF16, leading dim lda (elements, >= K)
//     W: quantized rows of full length ldw (elements); this GEMM consumes
//        the column window [k_off, k_off + K) of each row (the DSpark
//        chunked-fc slot GEMMs pass k_off = slot*H over a wider fc row;
//        plain calls pass k_off = 0, ldw = K). Any k_off is legal: scale
//        groups are anchored at the row START (global column / group).
//        NVFP4 rows are packed independently at ceil(ldw/2) bytes/row
//        (ldw must be even).
//     scales: [N, lds] — FP32 (FP8 path) or UE8M0 bytes (NVFP4 path),
//        lds = ceil(ldw / group) groups per row.
//     C: [M, N] tight row-major, BF16 or FP32.
//
// FP32 accumulation. Deterministic per shape (fixed reduction order); NOT
// bit-comparable to the BF16 kernels (different weights by construction).
// Graph-capturable: no host allocation, no lazy state.

#pragma once

#include <cstdint>

#include "compute/kernels/sm120/gemm/bf16_gemm.h"  // GemmAccOutDtype

namespace layerstorm::compute {

// Storage format of the quantized weight operand.
enum class WqWeightKind {
    kFp8E4M3,  // q [N, ldw] u8, scales [N, lds] FP32, group 128
    kNvfp4,    // q [N, ldw/2] packed u8 (low nibble = even column),
               // scales [N, lds] UE8M0 u8, group 16
};

void launch_wq_gemm_nt(void* C, const void* A /*BF16*/, const void* Wq,
                       const void* scales, WqWeightKind kind,
                       int M, int N, int K,
                       int64_t lda, int64_t ldw, int64_t k_off, int64_t lds,
                       GemmAccOutDtype out_dtype, void* stream);

}  // namespace layerstorm::compute
