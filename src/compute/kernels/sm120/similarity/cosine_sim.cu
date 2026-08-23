// GPU-side cosine similarity kernel for LayerStoRm.
//
// Computes cos(a, b) = dot(a, b) / (||a|| * ||b||) for two BF16 vectors.
// Single block, 256 threads.  Three parallel reductions: dot, norm_a, norm_b.
// Output: single FP32 value.

#include "compute/kernels/similarity/cosine_sim.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace layerstorm::compute {

static constexpr int kCosBlockSize = 256;

__device__ __forceinline__ float warp_reduce_sum_cos(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ float block_reduce_sum_cos(float val, float* smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    val = warp_reduce_sum_cos(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    const int num_warps = blockDim.x / 32;
    val = (threadIdx.x < static_cast<unsigned>(num_warps)) ? smem[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum_cos(val);

    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

__global__ void cosine_similarity_kernel(float* __restrict__ out_cos_sim,
                                         const __nv_bfloat16* __restrict__ a,
                                         const __nv_bfloat16* __restrict__ b,
                                         int dim) {
    __shared__ float smem[8];

    float dot_acc   = 0.0f;
    float norm_a_sq = 0.0f;
    float norm_b_sq = 0.0f;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float fa = __bfloat162float(a[i]);
        float fb = __bfloat162float(b[i]);
        dot_acc   += fa * fb;
        norm_a_sq += fa * fa;
        norm_b_sq += fb * fb;
    }

    dot_acc   = block_reduce_sum_cos(dot_acc, smem);
    norm_a_sq = block_reduce_sum_cos(norm_a_sq, smem);
    norm_b_sq = block_reduce_sum_cos(norm_b_sq, smem);

    if (threadIdx.x == 0) {
        float denom = sqrtf(norm_a_sq) * sqrtf(norm_b_sq);
        *out_cos_sim = (denom > 0.0f) ? (dot_acc / denom) : 0.0f;
    }
}

void launch_cosine_similarity(float* out_cos_sim,
                              const void* vec_a, const void* vec_b,
                              int dim, void* stream) {
    cosine_similarity_kernel<<<1, kCosBlockSize, 0,
                               static_cast<cudaStream_t>(stream)>>>(
        out_cos_sim,
        static_cast<const __nv_bfloat16*>(vec_a),
        static_cast<const __nv_bfloat16*>(vec_b),
        dim);
}

}  // namespace layerstorm::compute
