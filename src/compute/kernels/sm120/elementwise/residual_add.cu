// In-place BF16 residual addition for SM120 (GeForce Blackwell).
// residual[i] += input[i], vectorized via __nv_bfloat162.

#include "compute/kernels/elementwise/residual_add.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

__global__ void residual_add_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ input,
    int num_elements) {

    // Process 4 BF16 elements per thread (2 × bfloat162).
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec2_count = num_elements / 2;

    auto* res2 = reinterpret_cast<__nv_bfloat162*>(residual);
    const auto* inp2 = reinterpret_cast<const __nv_bfloat162*>(input);

    for (int i = tid; i < vec2_count; i += gridDim.x * blockDim.x) {
        __nv_bfloat162 r = res2[i];
        __nv_bfloat162 a = inp2[i];
        res2[i] = __hadd2(r, a);
    }

    // Handle odd trailing element.
    if (num_elements & 1) {
        const int last = num_elements - 1;
        if (tid == 0) {
            residual[last] = __hadd(residual[last], input[last]);
        }
    }
}

}  // namespace

void launch_residual_add(
    void* residual,
    const void* input,
    int num_elements,
    void* stream) {

    if (num_elements <= 0) return;

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    auto* res_bf16 = static_cast<__nv_bfloat16*>(residual);
    const auto* inp_bf16 = static_cast<const __nv_bfloat16*>(input);

    const int vec2_count = num_elements / 2;
    constexpr int kBlockSize = 256;
    const int grid_size = std::min((vec2_count + kBlockSize - 1) / kBlockSize, 65535);

    residual_add_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
        res_bf16, inp_bf16, num_elements);
}

namespace {

__global__ void add_inplace_f32_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int num_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < num_elements; i += gridDim.x * blockDim.x) {
        dst[i] += src[i];
    }
}

}  // namespace

void launch_add_inplace_f32(
    void* dst,
    const void* src,
    int num_elements,
    void* stream) {

    if (num_elements <= 0) return;

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    constexpr int kBlockSize = 256;
    const int grid_size =
        std::min((num_elements + kBlockSize - 1) / kBlockSize, 65535);

    add_inplace_f32_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
        static_cast<float*>(dst), static_cast<const float*>(src), num_elements);
}

}  // namespace layerstorm::compute
