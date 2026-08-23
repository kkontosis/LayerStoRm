// Router projection GEMM for MoE gating (SM120 / GeForce Blackwell).
// Self-contained BF16 GEMM (bf16_gemm.cu) with FP32 accumulation — replaces the
// former cuBLAS GEMM (Item 5: drop libcublas).
//
//   logits[num_tokens, n_experts] = hidden[num_tokens, hidden_size] × W[n_experts, hidden_size]^T
//
// This is the row-major NT GEMM C[M,N] = A[M,K] @ W[N,K]^T with
// M = num_tokens, N = n_experts, K = hidden_size. FP32 accumulation matches the
// prior CUBLAS_COMPUTE_32F numerics. Graph-capturable (no cuBLAS handle).

#include "compute/kernels/moe/router_projection.h"
#include "compute/kernels/sm120/gemm/bf16_gemm.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace layerstorm::compute {

void launch_router_projection(
    float* router_logits,
    const void* hidden_states,
    const void* weight,
    int num_tokens, int n_experts, int hidden_size,
    void* stream) {

    if (num_tokens <= 0) return;
    if (n_experts <= 0)
        throw std::invalid_argument(
            "launch_router_projection: n_experts must be > 0");
    if (hidden_size <= 0)
        throw std::invalid_argument(
            "launch_router_projection: hidden_size must be > 0");

    auto cuda_stream = static_cast<cudaStream_t>(stream);

    launch_bf16_gemm_nt(
        router_logits, hidden_states, weight,
        num_tokens, n_experts, hidden_size,
        GemmInDtype::kBFloat16, GemmAccOutDtype::kFloat32,
        cuda_stream);
}

}  // namespace layerstorm::compute
