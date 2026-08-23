// RMSNorm CUDA kernel interface for LayerStoRm.
// Adapted from TRT-LLM rmsnormKernels.cu and vLLM layernorm_kernels.cu
// (both Apache-2.0 — see THIRD_PARTY_NOTICES.md).

#pragma once

namespace layerstorm::compute {

// Runtime data type selection (no templates in public API).
enum class NormDtype { kFloat32, kBFloat16, kFloat16 };

// Plain RMSNorm: out = (input / rms(input)) * weight
//   input:  [num_tokens, hidden_size]
//   weight: [hidden_size]
//   out:    [num_tokens, hidden_size], same dtype as input
void launch_rmsnorm(void* out, const void* input, const void* weight,
                    float epsilon, int num_tokens, int hidden_size,
                    NormDtype dtype, void* stream /*cudaStream_t*/);

// Strided RMSNorm: same math, but consecutive token rows of BOTH input and out
// are `row_stride` ELEMENTS apart (row_stride >= hidden_size). Only the first
// hidden_size elements of each row are read/written — the tail
// [hidden_size, row_stride) is untouched. Needed when normalizing a sub-slice
// of interleaved rows (e.g. the c_kv half of the kv_a projection output
// [num_tokens, kv_lora_rank + qk_rope]): treating that as tight rows is exact
// only at num_tokens == 1 (TD-PREFILL-CHUNK-ATTN). launch_rmsnorm ==
// launch_rmsnorm_strided with row_stride == hidden_size.
void launch_rmsnorm_strided(void* out, const void* input, const void* weight,
                            float epsilon, int num_tokens, int hidden_size,
                            int row_stride, NormDtype dtype,
                            void* stream /*cudaStream_t*/);

// Fused residual-add + RMSNorm:
//   residual += input; out = rmsnorm(residual) * weight
//   input:    [num_tokens, hidden_size]
//   residual: [num_tokens, hidden_size], updated in-place
//   weight:   [hidden_size]
//   out:      [num_tokens, hidden_size], same dtype as input
void launch_fused_add_rmsnorm(void* out, const void* input, void* residual,
                              const void* weight, float epsilon,
                              int num_tokens, int hidden_size, NormDtype dtype,
                              void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
