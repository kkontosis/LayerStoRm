// FP32 -> BF16 elementwise round — see cast_f32_bf16.h
// (TD-GLM5-TP-COMBINE-PRECISION: the single post-allreduce bf16 rounding of
// the TP fp32 partial-combine path).

#include "compute/kernels/sm120/gemm/cast_f32_bf16.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace layerstorm::compute {

namespace {

__global__ void cast_f32_to_bf16_kernel(__nv_bfloat16* __restrict__ dst,
                                        const float* __restrict__ src,
                                        int64_t count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x
                    + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t j = i; j < count; j += stride)
        dst[j] = __float2bfloat16_rn(src[j]);
}

}  // namespace

void launch_cast_f32_to_bf16(void* dst_bf16, const void* src_f32,
                             int64_t count, void* stream) {
    if (count <= 0) return;
    constexpr int kBlock = 256;
    // Grid-stride: cap the grid so tiny decode counts stay one-wave and huge
    // prefill counts do not overflow gridDim.
    const int64_t blocks_needed = (count + kBlock - 1) / kBlock;
    const int grid = static_cast<int>(
        blocks_needed < 1024 ? blocks_needed : int64_t{1024});
    cast_f32_to_bf16_kernel<<<grid, kBlock, 0,
                              static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(dst_bf16),
        static_cast<const float*>(src_f32), count);
}

}  // namespace layerstorm::compute
