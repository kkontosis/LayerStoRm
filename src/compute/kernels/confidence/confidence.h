// GPU-side confidence estimation kernel for LayerStoRm.
//
// Computes top-1 softmax probability and normalized Shannon entropy
// from raw logits.  Paper motivation: Kangaroo (confidence-based draft
// termination: max(softmax) <= eta), FLy (entropy gate: h < theta).
//
// Operates on raw logits (not pre-softmaxed).  Numerically stable via
// online softmax (max subtraction).  Entropy normalized to [0, 1] by
// dividing by log(vocab_size), following the FLy paper convention.
//
// Does NOT modify the input logits buffer (unlike sampling, which
// applies temperature scaling in-place).

#pragma once

#include <cstdint>

namespace layerstorm::compute {

/// GPU-side confidence estimation: computes top-1 probability and
/// normalized entropy for each row of logits.
///
///   logits:       [num_tokens, vocab_size], FP32, on device. NOT modified.
///   top1_probs:   [num_tokens], FP32, on device. Output: max softmax prob.
///   entropies:    [num_tokens], FP32, on device. Output: normalized entropy in [0,1].
///   num_tokens:   Number of tokens (rows in logits matrix).
///   vocab_size:   Vocabulary dimension (e.g. 129280).
///   stream:       CUDA stream (cast to cudaStream_t internally).
void launch_compute_confidence(const float* logits,
                               float* top1_probs,
                               float* entropies,
                               int num_tokens, int vocab_size,
                               void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
