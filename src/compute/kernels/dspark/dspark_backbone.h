// DSpark DFlash-backbone CUDA kernel interfaces (DSP-3).
//
// The draft backbone is a dense Qwen3 transformer (per-head QK-RMSNorm,
// NEOX RoPE, standard MHA, SwiGLU MLP) running a NON-CAUSAL query block of
// gamma tokens over a precomputed context KV (projected from the target's
// fc-fused aux hidden states — vLLM qwen3_dflash.py semantics).  These three
// kernels cover the pieces the existing kernel library lacks; everything else
// (GEMMs, RMSNorm, embedding lookup, output head, fused add+RMSNorm) reuses
// launch_bf16_gemm_nt / launch_rmsnorm / launch_embedding_lookup /
// launch_output_head / launch_fused_add_rmsnorm.
//
// All kernels are BF16 in / BF16 out with FP32 math, deterministic (fixed
// reduction order), graph-capturable (no host allocation).

#pragma once

namespace layerstorm::compute {

// In-place NEOX-style rotary embedding on [num_tokens, n_heads, head_dim]
// BF16.  Row t uses position (base_pos + t); rotation pairs are
// (i, i + head_dim/2), angle = pos * theta^(-2i/head_dim).  The cos/sin are
// computed in FP32 on the fly (matches vLLM's FP32 cos_sin_cache class of
// precision) — no position table, so 1M-token positions cost no VRAM.
void launch_dspark_rope(void* x, int num_tokens, int n_heads, int head_dim,
                        int base_pos, float rope_theta,
                        void* stream /*cudaStream_t*/);

// Elementwise SwiGLU from SEPARATE gate/up buffers (the backbone runs two
// plain GEMMs, not the fused-MoE interleaved layout):
//   out[i] = silu(gate[i]) * up[i], FP32 math, BF16 in/out.  n = tokens * I.
void launch_dspark_silu_mul(void* out, const void* gate, const void* up,
                            long long n, void* stream /*cudaStream_t*/);

// DFlash non-causal block attention (DSP-3, INV-DSPARK-ANCHOR layout):
// every query row attends to ALL ctx_len context positions (the draft KV
// arena rows projected from target hiddens, already K-normed + RoPE'd) plus
// ALL num_query block positions (bidirectional inside the block — DSpark is
// non-causal, vLLM dspark/speculator.py dflash_causal=False).
//
//   q:      [num_query, n_heads, head_dim] BF16 (QK-normed + RoPE'd)
//   ctx_k:  [ctx_len,  n_heads, head_dim] BF16 (may be nullptr iff ctx_len==0)
//   ctx_v:  [ctx_len,  n_heads, head_dim] BF16
//   blk_k:  [num_query, n_heads, head_dim] BF16 (QK-normed + RoPE'd)
//   blk_v:  [num_query, n_heads, head_dim] BF16
//   out:    [num_query, n_heads * head_dim] BF16
//
// scale = head_dim^-0.5.  Online-softmax with a fixed intra-block merge
// order (deterministic).  head_dim <= 128.
void launch_dspark_block_attention(void* out, const void* q,
                                   const void* ctx_k, const void* ctx_v,
                                   const void* blk_k, const void* blk_v,
                                   int num_query, int ctx_len,
                                   int n_heads, int head_dim, float scale,
                                   void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
