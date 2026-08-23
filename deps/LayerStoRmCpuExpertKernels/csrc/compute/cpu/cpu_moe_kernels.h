#pragma once

// CPU MoE non-GEMM kernels for the expert FFN (C-8 SwiGLU, C-9 permute/unpermute).
//
// These mirror the device-side semantics of deps/LayerStoRmExpertKernels
// (smxx/activation/fused_swiglu, smxx/permute/moe_permute) but run on host
// cores. They are dtype-generic over `elem_size_bytes` (2 == BF16, the only
// value used by the expert FFN today).
//
// CPU-only: no CUDA / GPU SDK include. Covered by layerstorm_no_cuda_check.

#include <cstddef>
#include <cstdint>

namespace layerstorm::compute::cpu {

class NumaThreadPool;  // fwd (numa_thread_pool.h)

// ── C-8: SwiGLU ──────────────────────────────────────────────────────────────

/// Fused SwiGLU: output[t, j] = SiLU(gate[t, j]) * up[t, j], where
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
///
/// Two input layouts are supported via `gate_up_interleaved`:
///   * true  (matches GPU fused gate+up GEMM): `input` is [num_tokens, 2*d];
///     gate = input[t, 0:d], up = input[t, d:2d]; `up` is ignored.
///   * false (separate gate/up GEMMs, the C-6 CPU chain): `input` is the gate
///     buffer [num_tokens, d] and `up` is a separate [num_tokens, d] buffer.
/// `output` is [num_tokens, d]. All buffers same dtype (`elem_size_bytes`).
/// Element-wise + embarrassingly parallel; `pool` partitions tokens (null =>
/// serial). IMPL: moe kernels agent.
///
/// Optional clamping (`swiglu_limit` > 0), llama.cpp LLM_ARCH_DEEPSEEK4
/// semantics (MIT License, Copyright (c) 2023-2026 The ggml authors — see
/// THIRD_PARTY_NOTICES.md) — mirrors the GPU kernel (V4-4b):
///   gate = min(gate, limit)            (upper bound only)
///   up   = clamp(up, -limit, limit)    (two-sided)
/// 0.0 (default) = off, V3.2/GLM bit-identical legacy behavior.
void cpu_swiglu(
    void* output,                 ///< [num_tokens, d]
    const void* input,            ///< gate (or gate+up if interleaved)
    const void* up,               ///< [num_tokens, d] or nullptr when interleaved
    int num_tokens, int d,
    bool gate_up_interleaved,
    int elem_size_bytes,          ///< 2 (BF16)
    NumaThreadPool* pool,         ///< nullptr => serial
    float swiglu_limit = 0.0f);   ///< V4 clamp bound (0 = off)

// ── C-9: permute ─────────────────────────────────────────────────────────────

/// Sort the (token, expert) routing pairs by expert and scatter token rows into
/// expert-contiguous order, so the grouped GEMM can run per-expert contiguous
/// slices (no per-expert pointer indirection on the activation side).
///
/// Inputs:
///   hidden_states : [num_tokens, hidden_dim] BF16 row-major
///   topk_indices  : [num_tokens, topk] int32 expert id per (token, slot)
/// Outputs:
///   permuted_input  : [num_tokens * topk, hidden_dim] BF16, expert-contiguous
///   expert_offsets  : [num_experts + 1] cumulative permuted-row counts
///   src_to_dest_map : [num_tokens * topk] maps original (t*topk+k) -> permuted
///                     row index (used by unpermute to gather back)
///   permuted_idx    : [num_tokens * topk] permuted row -> source token index t
///                     (used by the GEMM/debug to know which token each row is)
///
/// Single-threaded sort is fine at decode token counts (num_tokens typically 1);
/// `pool` is accepted for symmetry / large-batch scatter and may be ignored by
/// the impl. IMPL: moe kernels agent.
void cpu_moe_permute(
    void* permuted_input,
    int32_t* expert_offsets,
    int32_t* src_to_dest_map,
    int32_t* permuted_idx,
    const void* hidden_states,
    const int32_t* topk_indices,
    int num_tokens, int topk, int hidden_dim, int num_experts,
    int elem_size_bytes,
    NumaThreadPool* pool);        ///< nullptr => serial

// ── C-9: unpermute ───────────────────────────────────────────────────────────

/// Reduce expert outputs back to original token order, weighted by the routing
/// weights:  output[t] = sum_k topk_weights[t,k] * permuted_output[src_to_dest_map[t*topk+k]].
///
///   permuted_output : [num_tokens * topk, hidden_dim] BF16 (down-proj output)
///   topk_weights    : [num_tokens, topk] f32 routing weights
///   src_to_dest_map : [num_tokens * topk] from cpu_moe_permute
///   output          : [num_tokens, hidden_dim] BF16 (zeroed then accumulated)
///
/// Accumulate in FP32, store BF16. `pool` partitions tokens (null => serial).
/// IMPL: moe kernels agent.
void cpu_moe_unpermute(
    void* output,
    const void* permuted_output,
    const float* topk_weights,
    const int32_t* src_to_dest_map,
    int num_tokens, int topk, int hidden_dim,
    int elem_size_bytes,
    NumaThreadPool* pool);        ///< nullptr => serial

/// DET-REDUCE Phase 1b (canonical placement-INVARIANT EP combine) — PER-SLOT
/// host variant of cpu_moe_unpermute. Instead of reducing the K expert
/// contributions per token, it writes each contribution into its OWN slot row
/// (dropped/padded slots write 0), mirroring the GPU kernels
/// finalize_moe_routing_bf16_to_{fp32,bf16}_perslot
/// (deps/LayerStoRmExpertKernels/csrc/smxx/permute/moe_permute.cu):
///   perslot_output[(t*topk + k)*hidden_dim + j]
///       = w_k * bf16_to_f32(permuted_output[dest[t*topk+k], j])
/// with the SAME single-multiply arithmetic (no FMA/accumulation) so the value
/// is bit-identical to the GPU per-slot combine when the down-proj rows match
/// (LS_CPU_EXPERT_LOSSLESS). `fp32_output` selects the payload dtype:
///   * true  → perslot_output is FP32 [num_tokens, topk, hidden_dim]
///   * false → perslot_output is BF16 [num_tokens, topk, hidden_dim]
///             (the product is rounded to BF16 once, matching __float2bfloat16).
///   permuted_output : [num_tokens * topk, hidden_dim] BF16 (down-proj output)
///   topk_weights    : [num_tokens, topk] f32 routing weights
///   src_to_dest_map : [num_tokens * topk] from cpu_moe_permute (-1 => slot 0)
void cpu_moe_unpermute_perslot(
    void* perslot_output,
    const void* permuted_output,
    const float* topk_weights,
    const int32_t* src_to_dest_map,
    int num_tokens, int topk, int hidden_dim,
    bool fp32_output,
    NumaThreadPool* pool);        ///< nullptr => serial

}  // namespace layerstorm::compute::cpu
