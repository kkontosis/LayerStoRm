// Shared prefill param population helpers for absorbed MLA.
//
// Both SnapMLA and TQ attention devices populate identical
// DenseAttnFwdParams / SparseAttnFwdParams fields for the shared
// BF16 prefill kernels.  These inline helpers eliminate duplication.
//
// Included only from CUDA-aware .cpp files (snapmla_sm120_attention_device.cpp,
// tq_sm120_attention_device.cpp).

#pragma once

#include <sm120/prefill/dense/fwd/head64/params.h>
#include <sm120/prefill/sparse/params.h>
#include <cmath>

namespace layerstorm::compute {

/// Model dimensions needed by prefill param population.
struct PrefillDims {
    int d_c;       // kv_lora_rank
    int d_rope;    // qk_rope_head_dim
    int h_q;       // num_heads_local (after TP split)
    int num_sm;    // GPU SM count
    // Model softmax scale: (qk_nope + qk_rope)^-0.5 (× yarn mscale²) — the absorbed
    // dot product equals the non-absorbed per-head dot product, so the NON-absorbed
    // head dim sets the scale (DeepSeek model.py:434). 0 → legacy 1/√d_qk fallback
    // (kernel-lib self-consistent default, kept for device unit tests).
    float sm_scale = 0.0f;
};

/// Compute max_blocks from max_blocks_per_seq with fallback.
inline int prefill_max_blocks(int max_blocks_per_seq, int seq_len_kv, int page_size) {
    return max_blocks_per_seq > 0
        ? max_blocks_per_seq
        : (seq_len_kv + page_size - 1) / page_size;
}

/// Populate all DenseAttnFwdParams fields for absorbed MLA prefill.
inline void populate_dense_prefill_params(
    sm120::prefill::dense::head64::DenseAttnFwdParams& p,
    const PrefillDims& dims,
    const void* q, void* kv_staging,
    int batch_size, int seq_len_kv,
    void* out, float* lse, cudaStream_t stream)
{
    const int d_qk = dims.d_c + dims.d_rope;
    const float sm_scale = dims.sm_scale > 0.0f
        ? dims.sm_scale : 1.0f / sqrtf(static_cast<float>(d_qk));

    p.s_q = batch_size;
    p.s_kv = seq_len_kv;
    p.h_q = dims.h_q;
    p.h_kv = 1;                          // absorbed MLA
    p.d_qk = d_qk;
    p.d_v = dims.d_c;
    p.sm_scale = sm_scale;
    p.sm_scale_div_log2 = sm_scale / logf(2.0f);
    p.q = const_cast<cutlass::bfloat16_t*>(
        static_cast<const cutlass::bfloat16_t*>(q));
    p.kv = static_cast<cutlass::bfloat16_t*>(kv_staging);
    p.attn_sink = nullptr;
    p.stride_q_s_q = dims.h_q * d_qk;
    p.stride_q_h_q = d_qk;
    p.stride_kv_s_kv = d_qk;            // h_kv=1
    p.stride_kv_h_kv = d_qk;
    p.out = static_cast<cutlass::bfloat16_t*>(out);
    p.max_logits = nullptr;
    p.lse = lse;
    p.num_sm = dims.num_sm;
    p.stream = stream;
}

/// Populate all SparseAttnFwdParams fields for absorbed MLA sparse prefill.
inline void populate_sparse_prefill_params(
    SparseAttnFwdParams& p,
    const PrefillDims& dims,
    const void* q, void* kv_staging,
    const int* sparse_indices, const int* topk_lengths, int topk,
    int batch_size, int seq_len_kv,
    void* out, float* lse, cudaStream_t stream)
{
    const int d_qk = dims.d_c + dims.d_rope;
    const float sm_scale = dims.sm_scale > 0.0f
        ? dims.sm_scale : 1.0f / sqrtf(static_cast<float>(d_qk));

    p.s_q = batch_size;
    p.s_kv = seq_len_kv;
    p.h_q = dims.h_q;
    p.h_kv = 1;
    p.d_qk = d_qk;
    p.d_v = dims.d_c;
    p.topk = topk;
    p.sm_scale = sm_scale;
    p.sm_scale_div_log2 = sm_scale / logf(2.0f);
    p.q = const_cast<cutlass::bfloat16_t*>(
        static_cast<const cutlass::bfloat16_t*>(q));
    p.kv = static_cast<cutlass::bfloat16_t*>(kv_staging);
    p.indices = const_cast<int*>(sparse_indices);
    p.attn_sink = nullptr;
    p.topk_length = const_cast<int*>(topk_lengths);
    p.stride_q_s_q = dims.h_q * d_qk;
    p.stride_q_h_q = d_qk;
    p.stride_kv_s_kv = d_qk;
    p.stride_kv_h_kv = d_qk;
    p.stride_indices_s_q = 1 * topk;     // h_kv=1
    p.stride_indices_h_kv = topk;
    p.out = static_cast<cutlass::bfloat16_t*>(out);
    p.max_logits = nullptr;
    p.lse = lse;
    p.num_sm = dims.num_sm;
    p.stream = stream;
}

}  // namespace layerstorm::compute
