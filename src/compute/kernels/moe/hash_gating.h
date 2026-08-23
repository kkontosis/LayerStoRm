// Hash-layer MoE gating (DeepSeek V4): expert selection by TOKEN ID via the
// ffn_gate_tid2eid I32 table instead of learned top-K, with routing weights
// computed from the router logits restricted to the selected experts.
//
// Reference semantics (ref/llama.cpp src/models/deepseek4.cpp:1127-1132 +
// src/llama-graph.cpp build_moe_ffn with selected_experts_in):
//   probs      = scoring_fn(router_logits)          (over ALL experts)
//   selected_k = tid2eid[token_id * topk + k]       (TABLE order, k = 0..topk-1)
//   weights_k  = probs[selected_k]                  (no exp_probs_b bias EVER)
//   if renormalize: weights *= routed_scaling / sum(weights)
//
// The scoring math mirrors deps/LayerStoRmExpertKernels topk_gating.cu
// bit-for-bit (stable sigmoid via tanh identity; sqrt(softplus) with the
// x > 20 overflow guard) so hash-layer weights match what learned gating
// would produce for the same logits.

#pragma once

#include <cstdint>

#include "sm120/gating/topk_gating.h"  // ScoringFunc (deps/LayerStoRmExpertKernels)

namespace layerstorm::compute {

/// Parameters for the hash-gating kernel launch.
struct HashGatingParams {
    int num_tokens;
    int num_experts;                ///< routed expert count (table ids must be < this)
    int topk;                       ///< experts per token (= table row width)
    int vocab_size;                 ///< tid2eid row count (token ids must be < this)
    float routed_scaling_factor;    ///< V4: 1.5 (expert_weights_scale)
    bool renormalize;               ///< norm_topk_prob
    ScoringFunc scoring_func = ScoringFunc::kSigmoid;
};

/// Hash-layer gating: expert ids from tid2eid[token_id], weights from logits.
///
///   topk_weights: [num_tokens, topk] FP32 out (table order — selection rank)
///   topk_indices: [num_tokens, topk] INT32 out
///   logits:       [num_tokens, num_experts] FP32 (router projection output)
///   tid2eid:      [vocab_size, topk] INT32 device table (GGUF ne0 = topk ⇒
///                 topk consecutive ids per vocab row)
///   token_ids:    [num_tokens] INT32 device (the layer-input token ids)
///
/// Out-of-range token ids or table entries produce expert id -1 (the permute
/// drop sentinel) with weight 0 — the slot is dropped, never misrouted.
void launch_hash_gating(float* topk_weights, int32_t* topk_indices,
                        const float* logits, const int32_t* tid2eid,
                        const int32_t* token_ids,
                        const HashGatingParams& params,
                        void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
