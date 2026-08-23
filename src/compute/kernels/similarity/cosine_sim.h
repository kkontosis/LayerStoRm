#pragma once

#include <cstdint>

namespace layerstorm::compute {

/// Computes cosine similarity between two BF16 vectors on device.
/// Output: single float written to out_cos_sim.
///   out_cos_sim: [1] FP32, device pointer.
///   vec_a, vec_b: [dim] BF16, device pointers.
///   dim: vector dimension (e.g. 7168).
///   stream: CUDA stream (cast to cudaStream_t internally).
void launch_cosine_similarity(float* out_cos_sim,
                              const void* vec_a, const void* vec_b,
                              int dim, void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
