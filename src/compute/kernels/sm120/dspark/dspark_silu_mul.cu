// DSpark backbone: elementwise SwiGLU from separate gate/up buffers (DSP-3).

#include "compute/kernels/dspark/dspark_backbone.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace layerstorm::compute {

namespace {

__global__ void dspark_silu_mul_kernel(__nv_bfloat16* out,
                                       const __nv_bfloat16* gate,
                                       const __nv_bfloat16* up,
                                       long long n) {
    const long long i =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __bfloat162float(gate[i]);
    const float u = __bfloat162float(up[i]);
    const float silu = g / (1.0f + expf(-g));
    out[i] = __float2bfloat16(silu * u);
}

}  // namespace

void launch_dspark_silu_mul(void* out, const void* gate, const void* up,
                            long long n, void* stream) {
    if (n <= 0) return;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    dspark_silu_mul_kernel<<<blocks, threads, 0,
                             static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<const __nv_bfloat16*>(up), n);
}

}  // namespace layerstorm::compute
