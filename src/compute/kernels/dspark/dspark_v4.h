// DSpark V4 dflash draft kernels (ticket J) — CUDA-free launchers
// (INV-GPU-1: callable from non-designated TUs).
//
// The DeepSeek-V4-Flash dspark draft runs 3 V4-shaped SWA-only blocks on the
// draft GPU over a BF16 context-KV arena (single 512 latent per position,
// K == V, roped tail). These kernels cover what the existing library lacks;
// everything else reuses launch_v4_q_prep / launch_v4_attn_sinks /
// launch_v4_out_inverse_rope (v4_prep.h), the mHC launchers (mhc.h),
// launch_bf16_gemm_nt(+strided_batched), launch_rmsnorm,
// launch_embedding_lookup / launch_output_head, launch_topk_gating /
// launch_router_projection, and the ExpertDevice MoE virtuals.
//
// RoPE convention: interleaved consecutive pairs with the ticket-D cos|sin
// half-row table (row = [cos_0..cos_{r/2-1} | sin_0..sin_{r/2-1}], stride =
// rope_dim floats, frequency index == pair index) — identical to v4_prep.cu.
// The draft uses the BASE table (theta 10000, no yarn): its blocks are
// trained as extra ratio-0 V4 layers (dual-rope rule, ticket D).

#pragma once

namespace layerstorm::compute {

// Rope the LAST rope_dim dims of each [head_dim] latent row at position
// (base_pos + row); the nope prefix is copied through. out may alias in.
//   in/out: [rows, head_dim] BF16
//   cos_sin: [max_pos, rope_dim] f32 cos|sin half rows (base table)
void launch_dspark_v4_kv_rope(void* out, const void* in, const void* cos_sin,
                              int base_pos, int rows, int head_dim,
                              int rope_dim, void* stream /*cudaStream_t*/);

// V4 dflash draft attention: latent MQA (h_q heads share ONE [head_dim]
// K == V vector per key), non-causal inside the query block, sliding-window
// context visibility, BF16 arena.
//
//   out:    [nq, h_q, head_dim] BF16
//   lse:    [nq, h_q] f32 — NATURAL-log-units logsumexp (for the
//           launch_v4_attn_sinks post-epilogue)
//   q_nope: [nq, h_q, head_dim] BF16 ([normed 448 | 0 x rope_dim] per head —
//           launch_v4_q_prep output)
//   q_rope: [nq, h_q, rope_dim] BF16 (roped pe)
//   ctx_kv: [>= n_ctx, head_dim] BF16 roped latent rows; ROW INDEX ==
//           ABSOLUTE POSITION (no ring). Nullable iff n_ctx == 0.
//   blk_kv: [nq, head_dim] BF16 roped latent rows of the query block itself
//           (row t at position base_pos + t)
//
// Query t (position p = base_pos + t) attends context positions
// j in [max(0, p - window + 1), n_ctx) — the llama.cpp SWA_STANDARD window —
// plus ALL nq block rows (non-causal; intra-block distances < window).
// score = (q_nope . k) + (q_rope . k[head_dim - rope_dim :]), x scale
// (the duplicated-rope score identity; q_nope's zero tail keeps the 512-dot
// exact). Deterministic fixed-order reductions. Sinks are NOT applied here —
// chain launch_v4_attn_sinks on (out, lse).
void launch_dspark_v4_attention(void* out, float* lse, const void* q_nope,
                                const void* q_rope, const void* ctx_kv,
                                const void* blk_kv, int nq, int n_ctx,
                                int base_pos, int window, int h_q,
                                int head_dim, int rope_dim, float scale,
                                void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
