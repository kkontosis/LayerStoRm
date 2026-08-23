// DSpark trained confidence-head kernel (DSP-6).
// See compute/kernels/dspark/dspark_confidence.h for the contract
// (INV-DSPARK-CONF: c_k = sigmoid(proj·[hidden_k ; markov_w1[x_{k-1}]] + b),
// hidden-first concat, FP32 accumulate, deterministic reduction order).

#include "compute/kernels/dspark/dspark_confidence.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace layerstorm::compute {

namespace {

constexpr int kThreads = 256;  // 8 warps
constexpr int kWarps = kThreads / 32;

// One block per draft position k.  The [H + r] dot is thread-strided in
// FP32 (fixed stride order), reduced warp-butterfly then across warps by
// thread 0 in fixed warp order — bit-deterministic run to run.
__global__ void dspark_confidence_kernel(
        float* __restrict__ conf_out, const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ prev_e,
        const __nv_bfloat16* __restrict__ proj_w,
        const __nv_bfloat16* __restrict__ proj_b, int hidden_dim, int rank) {
    const int k = blockIdx.x;
    const __nv_bfloat16* h_row =
        hidden + static_cast<long long>(k) * hidden_dim;
    const __nv_bfloat16* e_row =
        prev_e ? prev_e + static_cast<long long>(k) * rank : nullptr;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += kThreads)
        acc += __bfloat162float(h_row[i]) * __bfloat162float(proj_w[i]);
    if (e_row)
        for (int i = threadIdx.x; i < rank; i += kThreads)
            acc += __bfloat162float(e_row[i]) *
                   __bfloat162float(proj_w[hidden_dim + i]);

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);

    __shared__ float s_acc[kWarps];
    if ((threadIdx.x & 31) == 0) s_acc[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        // Ticket J: the V4 dflash GGUF conf_proj ships NO bias — a null
        // proj_b contributes 0 (GLM checkpoints always pass one).
        float logit = proj_b ? __bfloat162float(proj_b[0]) : 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) logit += s_acc[w];
        conf_out[k] = 1.0f / (1.0f + expf(-logit));
    }
}

}  // namespace

void launch_dspark_confidence(float* conf_out, const void* hidden,
                              const void* prev_e, const void* proj_w,
                              const void* proj_b, int num_query,
                              int hidden_dim, int rank, void* stream) {
    if (num_query <= 0 || hidden_dim <= 0) return;
    dspark_confidence_kernel<<<num_query, kThreads, 0,
                               static_cast<cudaStream_t>(stream)>>>(
        conf_out, static_cast<const __nv_bfloat16*>(hidden),
        static_cast<const __nv_bfloat16*>(prev_e),
        static_cast<const __nv_bfloat16*>(proj_w),
        static_cast<const __nv_bfloat16*>(proj_b), hidden_dim,
        prev_e ? rank : 0);
}

}  // namespace layerstorm::compute
