// hc-stream MEAN reduction (ticket J). See mhc/hc_stream_mean.h.

#include "compute/kernels/mhc/hc_stream_mean.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace layerstorm::compute {

namespace {

constexpr int kThreads = 256;

__global__ void hc_stream_mean_kernel(__nv_bfloat16* __restrict__ out,
                                      const __nv_bfloat16* __restrict__ in,
                                      int rows, int hc, int hidden) {
    const long long total = (long long)rows * hidden;
    const long long idx = (long long)blockIdx.x * kThreads + threadIdx.x;
    if (idx >= total) return;
    const long long r = idx / hidden;
    const long long d = idx % hidden;
    const __nv_bfloat16* row = in + r * (long long)hc * hidden + d;
    float acc = 0.0f;
    for (int s = 0; s < hc; ++s)
        acc += __bfloat162float(row[(long long)s * hidden]);
    out[idx] = __float2bfloat16_rn(acc / (float)hc);
}

}  // namespace

void launch_hc_stream_mean(void* out, const void* in, int rows, int hc,
                           int hidden, void* stream) {
    if (rows <= 0 || hidden <= 0 || hc <= 0) return;
    const long long total = (long long)rows * hidden;
    const unsigned blocks =
        static_cast<unsigned>((total + kThreads - 1) / kThreads);
    hc_stream_mean_kernel<<<blocks, kThreads, 0,
                            static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(in), rows, hc, hidden);
}

}  // namespace layerstorm::compute
