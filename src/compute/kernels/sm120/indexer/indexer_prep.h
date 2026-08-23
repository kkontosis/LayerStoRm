#pragma once

// GLM-25a: DSA indexer producer prep kernels (host-callable launch wrappers).
//
// These are the indexer-specific primitives the produce_sparse_indices()
// orchestration needs beyond the existing gemm / rope_rotate / score / topk:
//   - indexer_layernorm_bias : full LayerNorm (mean-center) with weight + bias,
//     applied to the single-head indexer key. NOTE: the indexer k_norm is a
//     LayerNorm, NOT RMSNorm (llama.cpp deepseek32.cpp:254 LLM_NORM), so the
//     engine's rmsnorm cannot be reused here.
//   - indexer_k_quant_append : quantize the (rotated, normed) single-head key to
//     FP8 E4M3 with one per-token scale and scatter it into the MQA indexer-K
//     cache ([block, index_head_dim]) at each token's block slot.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace layerstorm::compute {

// LayerNorm (subtract mean, divide by sqrt(var+eps)) then `*weight + bias`, over
// the last dim `dim`, for `num_rows` BF16 rows in place. weight/bias are BF16
// [dim]. Matches ggml LLM_NORM used by the DSA indexer k_norm.
struct IndexerLayerNormParams {
    __nv_bfloat16* x;              // [num_rows, dim] in place
    const __nv_bfloat16* weight;   // [dim]
    const __nv_bfloat16* bias;     // [dim]
    int num_rows;
    int dim;
    float eps;
};
void launch_indexer_layernorm_bias(const IndexerLayerNormParams& p, cudaStream_t stream);

// Quantize `num_tokens` single-head keys [num_tokens, index_head_dim] BF16 to
// FP8 E4M3 with one absmax-derived scale per token, writing each token's FP8 key
// into `k_cache[(slot_mapping[t] + slot_bias) * index_head_dim ..]` and its scale
// into `k_scales[slot_mapping[t] + slot_bias]`. slot_bias = −1 lets the decode
// path pass the seqlens_k device array directly (slot = seqlen − 1 = position).
// Dequant convention matches lightning_score_mqa: value = fp8 * scale
// (scale = absmax / 448).
struct IndexerKQuantAppendParams {
    const __nv_bfloat16* k_in;     // [num_tokens, index_head_dim]
    const int* slot_mapping;       // [num_tokens] → block slot in the cache
    __nv_fp8_e4m3* k_cache;        // [num_blocks, index_head_dim]
    float* k_scales;               // [num_blocks]
    int num_tokens;
    int index_head_dim;
    int slot_bias = 0;             // added to slot_mapping[t] (−1 for seqlens input)
};
void launch_indexer_k_quant_append(const IndexerKQuantAppendParams& p, cudaStream_t stream);

// Convert BF16 indexer weights [num_rows, n] → F32 score_proj [num_rows, n],
// multiplied by `scale` (= 1/sqrt(index_head_dim * index_n_heads)). This scaled
// per-token vector is the lightning-score `score_proj` input.
struct IndexerScaleWeightsParams {
    const __nv_bfloat16* in;   // [num_rows, n]
    float* out;                // [num_rows, n]
    int num_rows;
    int n;
    float scale;
};
void launch_indexer_scale_weights(const IndexerScaleWeightsParams& p, cudaStream_t stream);

}  // namespace layerstorm::compute
