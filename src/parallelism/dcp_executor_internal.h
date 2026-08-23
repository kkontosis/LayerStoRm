#pragma once
// Shared internal constants/helpers for the dcp_executor TU family
// (dcp_executor.cpp + daemon/attention/arch_mla.cpp +
// daemon/attention/arch_deepseek_v4.cpp), extracted in the attention
// refactor V2 P1 code motion.

#include <cstddef>

namespace layerstorm::parallelism {

// ── Helper: ceil division ───────────────────────────────────────────────────

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

// ── FP8 GEMM scale block size (matches Fp8GemmParams / dynamic_fp8_quant) ──

inline constexpr int kFp8ScaleBlockSize = 128;

// ── GGUF Q8_1 activation workspace sizing (GG-4) ────────────────────────────
// gguf_mmvq/gguf_mmq quantize the BF16 activation [M,K] into M*(K/32) Q8_1
// blocks of 36 bytes (sizeof(block_q8_1): 32 int8 quants + 2 fp16 d/sum).
// Mirrors compute::gguf_mmvq_workspace_bytes(M,K) without including the CUDA
// header (this TU is CUDA-free, INV-GPU-1). QK8_1=32 must divide K (true for
// the attention projection K dims). The kernel guards on this; we size for the
// worst case so a single workspace serves every projection.
inline constexpr int kGgufQ8_1BlockBytes = 36;
inline constexpr int kGgufQ8_1QK = 32;

inline size_t gguf_q8_1_workspace_bytes(int M, int K) {
    return static_cast<size_t>(M) * (static_cast<size_t>(K) / kGgufQ8_1QK)
         * kGgufQ8_1BlockBytes;
}

}  // namespace layerstorm::parallelism
