#pragma once

// Shared CPU GGUF grouped-GEMM driver (C-6). Used by both NumaCpuExpertDevice and
// MultiNumaCpuExpertDevice's gguf_grouped_gemm overrides so the quantize +
// per-expert verbatim-ik GEMM + BF16 round logic lives in ONE place.
//
// Pipeline (one projection: gate, up, or down):
//   BF16 permuted activations -> FP32 -> ik activation format (Q8_2_X4 / Q8_K)
//   then per expert e: ls_iqk_mul_mat(W_e rows, A_e tokens) -> FP32 -> BF16 out.
// Output rows (N) are partitioned across the supplied pool's threads.
//
// CPU-only; no CUDA (INV-GPU-1). The ik kernels are vendored verbatim under
// compute/cpu/ik_vendor; this is just the LayerStoRm grouped/threaded driver.

#include "compute/cpu/ik_vendor/ik_gguf_gemm.h"

#include <cstdint>

namespace layerstorm::compute::cpu {

class NumaThreadPool;  // fwd (numa_thread_pool.h)

/// Run a GGUF grouped GEMM for ONE projection across `num_experts`.
///   D            : [total_tokens, N] BF16 output (row-major)
///   A            : [total_tokens, K] BF16 activations (row-major, permuted)
///   B_ptrs[e]    : packed GGUF weight block for expert e ([N, K] @ `type`)
///   expert_offsets : [num_experts + 1] cumulative permuted-token counts
///   pool         : partitions N output rows across threads (nullptr => serial)
/// Throws std::runtime_error if `type` is unsupported on this build/host.
void cpu_gguf_grouped_gemm(uint16_t* D, const uint16_t* A,
                           ik::GgufType type, const void* const* B_ptrs,
                           const int32_t* expert_offsets, int num_experts,
                           int N, int K, NumaThreadPool* pool);

// ── Multi-NUMA split-N building blocks (TD-IK-MULTINODE) ─────────────────────
// The multi-node device partitions the N output rows across M NUMA nodes; each
// GGUF output row is a self-contained packed weight row, so a node's N-slice is a
// contiguous range of packed weight rows (no sub-row alignment constraint, unlike
// NVFP4's 128-row Sm1xx scale atom). These two helpers let the multi-node device
// (1) quantize the activations ONCE and broadcast the small Q8_x buffer to all
// nodes, then (2) run each node's row-slice from its LOCAL staged weight slab.

/// Quantize the BF16 permuted activations to the ik activation format ONCE for the
/// whole call. `quant_a_out` must be sized total_tokens * activation_row_bytes(type,K).
/// Shared (broadcast) across nodes by the multi-node device. Throws if `type` is
/// unsupported on this build/host.
void cpu_gguf_quantize_activations(const uint16_t* A, ik::GgufType type,
                                   int total_tokens, int K, uint8_t* quant_a_out);

/// Run the row-slice [n0, n0+nslice) of a GGUF grouped GEMM, given PRE-QUANTIZED
/// activations (cpu_gguf_quantize_activations) and per-expert LOCAL weight slabs
/// holding ONLY rows [n0, n0+nslice) (stride weight_row_bytes(type,K), `nslice`
/// rows each). Writes columns [n0, n0+nslice) of the shared D [total_tokens, N].
///   D              : [total_tokens, N] BF16 output (full width; this writes the slice's cols)
///   quant_a        : pre-quantized activations [total_tokens, act_rb]
///   B_slabs[e]     : expert e's LOCAL packed-weight-row slab (nslice rows @ n0..)
///   expert_offsets : [num_experts + 1] cumulative permuted-token counts
///   N              : FULL output width (D row stride); n0/nslice index into it
///   pool           : partitions the nslice rows across this node's threads
/// Bit-exact to the single-node path for the same row-slice (each output element
/// is a complete dot computed on ONE node; no cross-node accumulation).
void cpu_gguf_grouped_gemm_slice(uint16_t* D, const uint8_t* quant_a,
                                 ik::GgufType type, const void* const* B_slabs,
                                 const int32_t* expert_offsets, int num_experts,
                                 int N, int K, int n0, int nslice,
                                 NumaThreadPool* pool);

}  // namespace layerstorm::compute::cpu
