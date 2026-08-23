// GPU-side token sampling kernels for LayerStoRm.
//
// Supports argmax, top-K, top-P (nucleus), and combined top-K + top-P.
// One block per token; each token is sampled independently.
//
// Mode selection (runtime):
//   temperature <= 0 OR top_k == 1  ->  argmax (deterministic)
//   top_k > 0 AND top_p == 1.0      ->  top-K only
//   top_k == 0 AND top_p < 1.0      ->  top-P only (pre-filtered to 1024 candidates)
//   top_k > 0 AND top_p < 1.0       ->  top-K then top-P within K candidates
//
// RNG: Inline Philox-4x32-10 (no curand dependency).
// Each (seed, token_index) pair gives a deterministic, independent random stream.
//
// WARNING: Temperature scaling modifies logits in-place.
// Caller must not reuse the logits buffer after calling this function.

#pragma once

#include <cstdint>

namespace layerstorm::compute {

/// GPU-side token sampling: picks one token ID per row of logits.
///
///   logits:      [num_tokens, vocab_size], FP32, on device.
///                Modified in-place (temperature scaling).
///   output_ids:  [num_tokens], int32_t, on device.
///   num_tokens:  Number of tokens to sample (one per logit row).
///   vocab_size:  Vocabulary dimension (e.g. 129280).
///   temperature: > 0 for stochastic sampling, <= 0 for argmax.
///   top_p:       Nucleus threshold in (0, 1]. 1.0 disables.
///   top_k:       Top-K cutoff. 0 disables.
///   seed:        Random seed (combined with token index for per-token state).
///   stream:      CUDA stream (cast to cudaStream_t internally).
void launch_sample_tokens(int32_t* output_ids, float* logits,
                          int num_tokens, int vocab_size,
                          float temperature, float top_p, int top_k,
                          uint64_t seed, void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
