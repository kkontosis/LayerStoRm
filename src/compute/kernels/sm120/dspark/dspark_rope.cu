// DSpark backbone: in-place NEOX RoPE (DSP-3).  See dspark_backbone.h.

#include "compute/kernels/dspark/dspark_backbone.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace layerstorm::compute {

namespace {

// One thread per rotation pair: (token, head, i) with i in [0, head_dim/2).
__global__ void dspark_rope_kernel(__nv_bfloat16* x, int num_tokens,
                                   int n_heads, int head_dim, int base_pos,
                                   float rope_theta) {
    const int half = head_dim / 2;
    const long long total =
        static_cast<long long>(num_tokens) * n_heads * half;
    const long long idx =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int i = static_cast<int>(idx % half);
    const int th = static_cast<int>(idx / half);   // token * n_heads + head
    const int t = th / n_heads;

    // angle = pos * theta^(-2i/d)  (FP32, matching vLLM's FP32 cache class).
    const float pos = static_cast<float>(base_pos + t);
    const float inv_freq =
        powf(rope_theta, -2.0f * static_cast<float>(i) /
                             static_cast<float>(head_dim));
    float s, c;
    sincosf(pos * inv_freq, &s, &c);

    __nv_bfloat16* row = x + static_cast<long long>(th) * head_dim;
    const float x0 = __bfloat162float(row[i]);
    const float x1 = __bfloat162float(row[i + half]);
    row[i]        = __float2bfloat16(x0 * c - x1 * s);
    row[i + half] = __float2bfloat16(x0 * s + x1 * c);
}

}  // namespace

void launch_dspark_rope(void* x, int num_tokens, int n_heads, int head_dim,
                        int base_pos, float rope_theta, void* stream) {
    if (num_tokens <= 0) return;
    const long long total =
        static_cast<long long>(num_tokens) * n_heads * (head_dim / 2);
    const int threads = 256;
    const int blocks =
        static_cast<int>((total + threads - 1) / threads);
    dspark_rope_kernel<<<blocks, threads, 0,
                         static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(x), num_tokens, n_heads, head_dim,
        base_pos, rope_theta);
}

}  // namespace layerstorm::compute
