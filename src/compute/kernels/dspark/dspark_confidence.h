// DSpark trained confidence-head CUDA kernel interface (DSP-6).
//
// Per-position survival estimator (INV-DSPARK-CONF — authoritative refs:
// ref/DeepSpec deepspec/modeling/dspark/common.py `AcceptRatePredictor` +
// qwen3/modeling.py `predict_confidence_step` / eval/dspark/draft_ops.py
// `_predict_confidence_logits`; paper §3.2.1):
//
//   feat_k = [ hidden_k (H) ; markov_w1[x_{k-1}] (r) ]   concat, hidden FIRST
//   c_k    = sigmoid( proj_w · feat_k + proj_b )          scalar in (0,1)
//
// c_k is the TRAINED conditional probability that draft token k survives
// verification GIVEN all predecessors accepted (DeepSpec trains the logit
// with BCE against c*_k = 1 − ½‖p_draft − p_target‖₁) — cumprod-composable:
// the orchestrator forms the cumulative survival a_j = Π_{i≤j} c_i.
// x_{k−1} is the ACTUALLY-SAMPLED previous draft token (the anchor for
// k = 0) — exactly the DSP-4 Markov e-chain (markov_w1 row), reused as-is.
//
// INV-DSPARK-CONF: this trained head is DISTINCT from the output-head
// {top1_prob, entropy} confidence heuristic in compute/kernels/confidence/
// (IPC-8g).  Separate entry point, separate semantics — never overload.
//
// checkpoint `confidence_head_with_markov` = false ⇒ feat_k = hidden_k only:
// pass prev_e = nullptr / rank = 0.

#pragma once

namespace layerstorm::compute {

/// ONE kernel over the gamma positions: conf_out[k] = sigmoid(logit_k).
///   conf_out: [num_query] FP32 out, each in (0,1)
///   hidden:   [num_query, hidden_dim] BF16 (post-final-norm backbone out)
///   prev_e:   [num_query, rank] BF16 — row k = markov_w1[x_{k-1}] (the
///             DSP-4 e-chain stash); nullptr when the head is hidden-only
///   proj_w:   [hidden_dim + rank] BF16 (confidence_head.proj.weight [1, .]
///             row-major: first hidden_dim elems multiply hidden, next rank
///             multiply prev_e — DeepSpec concat order, hidden FIRST)
///   proj_b:   [1] BF16 (confidence_head.proj.bias) — nullptr when the
///             checkpoint ships no bias (V4 dflash GGUF): contributes 0
/// FP32 dot accumulation, fixed reduction order (bit-deterministic).
void launch_dspark_confidence(float* conf_out, const void* hidden,
                              const void* prev_e, const void* proj_w,
                              const void* proj_b, int num_query,
                              int hidden_dim, int rank,
                              void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
