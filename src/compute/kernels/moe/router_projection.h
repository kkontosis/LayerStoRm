// Router projection GEMM: hidden_states × W_router^T → router_logits.
// Self-contained BF16 GEMM (bf16_gemm.cu) — works on all GPUs (no
// AttentionDevice dependency), graph-capturable, no cuBLAS.

#pragma once

#include <cstdint>

namespace layerstorm::compute {

// Compute router logits for MoE gating.
//   router_logits: [num_tokens, n_experts] FP32 output
//   hidden_states: [num_tokens, hidden_size] BF16 input
//   weight:        [n_experts, hidden_size] BF16 row-major (pinned region)
void launch_router_projection(
    float* router_logits,
    const void* hidden_states,
    const void* weight,
    int num_tokens, int n_experts, int hidden_size,
    void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
