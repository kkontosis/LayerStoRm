#pragma once

//==============================================================================
// MLA Attention Kernel Wrappers
//
// Thin dispatch layer in layerstorm::compute that forwards to SnapMLA kernel
// launch functions from deps/LayerStoRmKernels. Handles model-type dispatch
// (V32 vs MODEL1 vs GLM5N) based on runtime parameters.
//
// GEOMETRIES (d_qk = d_nope + d_rope, d_v = the V width of a staged KV row):
//   V32    576 = 512 nope + 64 rope, d_v 512  (GLM-5.2 / DeepSeek-V3.2)
//   MODEL1 512 = 448 nope + 64 rope, d_v 448  (FlashMLA-inherited)
//   GLM5N  512 = 512 nope +  0 rope, d_v 512  (GLM-5.3-Flash NoPE)
// d_nope alone does NOT identify the model — V32 and GLM5N share d_nope 512 —
// so every dispatch below keys on BOTH dims.
//==============================================================================

#include <cuda_runtime.h>

// Forward-declare param structs to avoid pulling heavy headers into every TU.
// Callers that need to populate params include the full headers directly.
namespace sm120::decode::dense_fp8 { struct DenseAttnDecodeParams; }
namespace sm120::decode::sparse_fp8 { struct SparseAttnDecodeParams; }
namespace sm120::prefill::dense::head64 { struct DenseAttnFwdParams; }
struct SparseAttnFwdParams;
struct MlaCombineParams;
struct GetMlaMetadataParams;
namespace sm120::prep {
struct FusedQQuantParams;
struct FusedKAppendParams;
struct DequantCKVIndexedParams;
struct QAbsorbParams;
struct RopeRotateParams;
}

namespace layerstorm::compute {

// ── Decode ──────────────────────────────────────────────────────────────────

// Dense FP8 decode (SnapMLA, paged KV). Dispatches by (d_qk, d_nope):
// (576,512)→V32, (512,512)→GLM5N, (512,448)→MODEL1; anything else throws.
void launch_decode_dense_fp8(const sm120::decode::dense_fp8::DenseAttnDecodeParams& params);

// Sparse FP8 decode (SnapMLA with topk indices). Dispatches by params.model_type.
void launch_decode_sparse_fp8(const sm120::decode::sparse_fp8::SparseAttnDecodeParams& params);

// ── Prefill ─────────────────────────────────────────────────────────────────

// Dense absorbed BF16 prefill. Dispatches by (d_qk, d_v): 576→<576> (V32),
// (512,512)→<512> (GLM5N NoPE); anything else throws.
// NOTE: MLA decode routes through the prefill kernels at batch_size == 1 on the
// SnapMLA backend, so this dispatch is DECODE-correctness-critical.
void launch_prefill_dense(const sm120::prefill::dense::head64::DenseAttnFwdParams& params);

// Sparse absorbed BF16 prefill. Dispatches by d_qk: 576→<576> (V32),
// 512→<512> (GLM5N NoPE, requires d_v == 512); anything else throws.
// NOTE: DECODE-correctness-critical for the same reason as the dense twin.
void launch_prefill_sparse(const SparseAttnFwdParams& params);

// ── Prep kernels ────────────────────────────────────────────────────────────

// Q quantization: BF16 → FP8 NOPE + BF16 ROPE (pre-scaled) + per-head scales.
void launch_fused_q_quant(const sm120::prep::FusedQQuantParams& params,
                          cudaStream_t stream);

// Q absorption: ql_nope = q_nope · W_UK (K-half of kv_b_proj) + rope concat → [s_q, h_q, d_c+d_rope].
void launch_q_absorb(const sm120::prep::QAbsorbParams& params,
                     cudaStream_t stream);

// In-place strided RoPE rotation (interleaved adjacent-pair, pos = seqlens_k[t]−1).
void launch_rope_rotate(const sm120::prep::RopeRotateParams& params,
                        cudaStream_t stream);

// K append: quantize c_KV + pre-scale ROPE, write to paged KV cache.
void launch_fused_k_append(const sm120::prep::FusedKAppendParams& params,
                           cudaStream_t stream);

// CKV dequant: gather from paged FP8 cache → contiguous BF16 (for chunked prefill).
void launch_dequant_ckv_indexed(const sm120::prep::DequantCKVIndexedParams& params,
                                cudaStream_t stream);

// ── Split-KV utilities ──────────────────────────────────────────────────────

// Merge split-KV partial results into final output (BF16).
void launch_mla_combine(MlaCombineParams& params, cudaStream_t stream);

// Compute split-KV tile scheduler metadata.
void launch_get_mla_metadata(GetMlaMetadataParams& params, cudaStream_t stream);

}  // namespace layerstorm::compute
