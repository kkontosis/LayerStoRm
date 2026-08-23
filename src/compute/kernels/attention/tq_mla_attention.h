#pragma once

//==============================================================================
// TQ MLA Attention Kernel Wrappers
//
// Thin dispatch layer for TurboQuant MLA kernels.  The TQ attention device
// (tq_sm120_attention_device.cpp) includes ONLY this header — never
// mla_attention.h — to prevent accidental cross-backend kernel calls
// (see spec/TURBOQUANT_DETAILS.md §17.4).
//
// TQ-only prep:   tq_k_append, tq_dequant_ckv_indexed, tq_q_rotate,
//                 tq_v_rotate_back
// TQ decode:      dense + sparse (scalar codebook, rotated-space PV)
// TQ prefill:     composite — dequant TQ cache → BF16, then shared prefill
// Shared:         mla_combine (float instantiation), get_mla_metadata
//==============================================================================

#include <cuda_runtime.h>

// Forward-declare TQ param structs (submodule headers).
namespace sm120::decode::tq_dense { struct TqDenseDecodeParams; }
namespace sm120::decode::tq_sparse { struct TqSparseDecodeParams; }
namespace sm120::prefill::dense::head64 { struct DenseAttnFwdParams; }
struct SparseAttnFwdParams;
struct MlaCombineParams;
struct GetMlaMetadataParams;
namespace sm120::prep {
struct TqFusedKAppendParams;
struct TqDequantCKVIndexedParams;
struct TqQRotateParams;
struct TqVRotateBackParams;
}

namespace layerstorm::compute {

// ── TQ Decode ──────────────────────────────────────────────────────────────

// Dense TQ decode (paged cache, split-KV). Output is FP32 in rotated space.
void launch_decode_dense_tq(const sm120::decode::tq_dense::TqDenseDecodeParams& params);

// Sparse TQ decode (topk indices). Output is FP32 in rotated space.
void launch_decode_sparse_tq(const sm120::decode::tq_sparse::TqSparseDecodeParams& params);

// TD-TQ-SPARSE-DECODE-UNWIRED: DSA selection (sequence positions) × block-table
// linearization (position → pool slot) → pool-slot indices for the TQ sparse
// decode kernel; -1 outside topk_length[0] / causal bound (mirrors phase1).
// seq_len_dev (optional): device int holding the causal bound — read at
// EXECUTION time so the launch is CUDA-graph-replayable across tokens
// (LS_TQ_DECODE_GRAPH); when null the host seq_len_kv value is baked in.
void launch_tq_sparse_translate_indices(
    const int* sparse_indices, const int* lin_slots, const int* topk_length,
    int topk, int seq_len_kv, const int* seq_len_dev, int* out,
    cudaStream_t stream);

// ── TQ Prefill (composite: dequant + shared BF16 prefill) ──────────────────

// Dense prefill: dequant TQ cache → BF16, then standard absorbed dense prefill.
void launch_prefill_dense_tq(
    const sm120::prep::TqDequantCKVIndexedParams& dequant_params,
    const sm120::prefill::dense::head64::DenseAttnFwdParams& prefill_params,
    cudaStream_t stream);

// Sparse prefill: dequant TQ cache → BF16, then standard absorbed sparse prefill.
void launch_prefill_sparse_tq(
    const sm120::prep::TqDequantCKVIndexedParams& dequant_params,
    const SparseAttnFwdParams& prefill_params,
    cudaStream_t stream);

// ── TQ Prep kernels ────────────────────────────────────────────────────────

// K append: BF16 c_kv → TQ_MSE 4-bit packed + BF16 rope into paged cache.
void launch_tq_k_append(const sm120::prep::TqFusedKAppendParams& params,
                         cudaStream_t stream);

// CKV dequant: gather from TQ paged cache → contiguous BF16.
void launch_tq_dequant_ckv_indexed(const sm120::prep::TqDequantCKVIndexedParams& params,
                                    cudaStream_t stream);

// Q rotate: pre-rotate q_nope by Pi^T (BF16 → FP32). Once per layer per decode step.
void launch_tq_q_rotate(const sm120::prep::TqQRotateParams& params,
                         cudaStream_t stream);

// V rotate back: inverse rotation on combine output (FP32 → BF16). TQ-only epilogue.
void launch_tq_v_rotate_back(const sm120::prep::TqVRotateBackParams& params,
                              cudaStream_t stream);

// ── Split-KV utilities (shared, re-declared for TQ self-containment) ───────

// Merge split-KV partial results (FP32 output for TQ rotated space).
void launch_mla_combine_f32(MlaCombineParams& params, cudaStream_t stream);

// Compute split-KV tile scheduler metadata (format-agnostic).
void launch_get_mla_metadata_tq(GetMlaMetadataParams& params, cudaStream_t stream);

}  // namespace layerstorm::compute
