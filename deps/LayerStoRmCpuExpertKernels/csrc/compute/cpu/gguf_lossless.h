#pragma once

// Bit-compatible (with the GPU GGUF mmvq path) CPU Q4_K expert grouped GEMM.
//
// The production TARGET model is GGUF Q4_K (gate/up); the GPU routed-expert GEMM
// quantizes activations to **Q8_1** on-device (block 32, d = amax/127 kept full-
// float for the quant, stored + applied as FP16; qs = round_nearest(x/d)) and
// does the Q4_K×Q8_1 integer dot
//   acc += d_act · ( (d·sc)·Σ(q4·q8) − (dmin·m)·Σq8 )   per 4-value group,
// exactly vec_dot_q4_K_q8_1 (deps/LayerStoRmGemmKernels smxx/formats/q8_1_format.h,
// sm120/gemm/gguf/{gguf_mmvq_impl.cuh,gguf_int_policy.h}; ggml/llama.cpp
// lineage, MIT License, Copyright (c) 2023-2026 The ggml authors — see
// THIRD_PARTY_NOTICES.md and csrc/compute/cpu/ik_vendor/LICENSE.ik).
//
// The default CPU expert GGUF path (cpu_gguf_grouped_gemm) uses the VERBATIM ik
// kernels with **Q8_2_X4** activations — d re-rounded to BF16 before quantizing —
// which is MORE accurate but DIFFERENT from the GPU, so a CPU-computed Q4_K
// expert would drift the TARGET's MoE hidden vs a GPU-computed one and could flip
// the TQ golden (12089). This kernel closes that gap: it reproduces the GPU's
// Q8_1 quant + mmvq integer dot so a CPU-computed Q4_K expert matches the GPU
// GGUF expert to within FP32 accumulation-ORDER round-off (the integer dot and
// the quantization are bit-identical; only the final reduction order differs —
// GPU warp-tree vs CPU sequential — exactly as the GPU's own mmvq vs mmq differ).
//
// Q4_K only (the dominant gate/up target quant). Q5_K/Q6_K (the down projection
// in Q4_K_XL) use the identical technique (the GPU policy has Q5K/Q6K group()
// math) and are the remaining piece for a fully-lossless expert FFN.
//
// CPU-only: no CUDA (INV-GPU-1).

#include <cstddef>
#include <cstdint>

namespace layerstorm::compute::cpu {

class NumaThreadPool;  // fwd (numa_thread_pool.h)

/// Quantize ONE activation row [K] BF16 → Q8_1 blocks, matching the GPU
/// gguf_quantize_q8_1: per 32-group d = amax/127 (full float) for the quant,
/// qs[i] = round_nearest(x[i]/d) int8; the per-block scale used by the dot is
/// FP16(d). `out_d` receives K/32 FP16-rounded block scales (as float); `out_qs`
/// receives K int8 quants (row-major, 32 per block). K must be a multiple of 32.
void q8_1_quantize_row(const uint16_t* a_bf16, int K, float* out_d, int8_t* out_qs);

/// K-quant families with a bit-compatible CPU lossless path (super-block = 256).
/// Q4_K/Q5_K carry a super-scale + super-min (dm) and 8 6-bit (scale,min) pairs;
/// Q6_K carries a super-scale d and 16 int8 sub-scales (no min, centred quants).
enum class KQuantLossless { Q4_K, Q5_K, Q6_K };

/// Bit-compatible K-quant × Q8_1 grouped GEMM (one projection across experts),
/// matching the GPU mmvq (gguf_int_policy.h Q4K/Q5K/Q6K::group). Covers the whole
/// expert FFN losslessly: gate/up = Q4_K, down = Q5_K or Q6_K.
///   D            : [total_tokens, N] BF16 out (row-major)
///   A            : [total_tokens, K] BF16 activations (row-major)
///   B_ptrs[e]    : expert e's packed weight rows ([N, K]; 144/176/210 B per
///                  super-block for Q4_K/Q5_K/Q6_K respectively)
///   expert_offsets : [num_experts + 1] cumulative permuted-token counts
///   pool         : partitions N output rows across threads (nullptr => serial)
/// K must be a multiple of 256; N is the output width.
void cpu_gguf_grouped_gemm_kq_lossless(
    KQuantLossless type, uint16_t* D, const uint16_t* A, const void* const* B_ptrs,
    const int32_t* expert_offsets, int num_experts, int N, int K,
    NumaThreadPool* pool);

/// Q4_K convenience wrapper (== cpu_gguf_grouped_gemm_kq_lossless(Q4_K, ...)).
void cpu_gguf_grouped_gemm_q4k_lossless(
    uint16_t* D, const uint16_t* A, const void* const* B_ptrs,
    const int32_t* expert_offsets, int num_experts, int N, int K,
    NumaThreadPool* pool);

/// SCALAR reference for the same GPU-mmvq-order K-quant × Q8_1 grouped GEMM: the
/// per-group integer dot is a plain scalar loop (NOT AVX512-VNNI) but the FP32
/// accumulation reproduces the SAME GPU 32-lane warp-tree reduction order as the
/// vectorized production kernel. Exists so a unit test can prove the VNNI integer
/// dot is BIT-IDENTICAL (0/N differ) to the scalar dot under the identical
/// reduction. Not used on the hot path.
void cpu_gguf_grouped_gemm_kq_lossless_scalar_ref(
    KQuantLossless type, uint16_t* D, const uint16_t* A, const void* const* B_ptrs,
    const int32_t* expert_offsets, int num_experts, int N, int K,
    NumaThreadPool* pool);

}  // namespace layerstorm::compute::cpu
