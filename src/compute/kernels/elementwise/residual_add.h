// In-place BF16 residual addition: residual[i] += input[i].

#pragma once

#include <cstdint>

namespace layerstorm::compute {

// Vectorized BF16 in-place add: residual += input.
//   residual: [num_elements] BF16, updated in-place
//   input:    [num_elements] BF16
void launch_residual_add(
    void* residual,
    const void* input,
    int num_elements,
    void* stream /*cudaStream_t*/);

// FP32 in-place add: dst[i] += src[i]. Used by the EP-beyond-TP routed
// partial gather (INV-MOE-EP-XTP) when the canonical per-slot combine runs
// with the fp32 payload — the expert-only rank's [B, topk, H] fp32 per-slot
// partial is D2D-staged onto a TP rank and folded in with this add (disjoint
// slots ⇒ x + 0 = x, bit-exact).
void launch_add_inplace_f32(
    void* dst,
    const void* src,
    int num_elements,
    void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
