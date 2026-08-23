// GLM-25a: DSA indexer producer prep kernels. See indexer_prep.h.

#include "compute/kernels/sm120/indexer/indexer_prep.h"

#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

constexpr float kFp8E4M3Max = 448.0f;

// One CTA per row; blockDim.x threads cooperate over `dim`. Two-pass welford-free
// LayerNorm (mean then variance) using a shared-memory reduction.
__global__ void indexer_layernorm_bias_kernel(IndexerLayerNormParams p) {
    const int row = blockIdx.x;
    if (row >= p.num_rows) return;
    __nv_bfloat16* x = p.x + static_cast<int64_t>(row) * p.dim;

    extern __shared__ float smem[];  // blockDim.x floats
    const int tid = threadIdx.x;

    // Pass 1: mean.
    float local = 0.0f;
    for (int i = tid; i < p.dim; i += blockDim.x) local += __bfloat162float(x[i]);
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    const float mean = smem[0] / p.dim;
    __syncthreads();

    // Pass 2: variance.
    float lvar = 0.0f;
    for (int i = tid; i < p.dim; i += blockDim.x) {
        float d = __bfloat162float(x[i]) - mean;
        lvar += d * d;
    }
    smem[tid] = lvar;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    const float inv_std = rsqrtf(smem[0] / p.dim + p.eps);

    // Normalize + affine.
    for (int i = tid; i < p.dim; i += blockDim.x) {
        float v = (__bfloat162float(x[i]) - mean) * inv_std;
        v = v * __bfloat162float(p.weight[i]) + __bfloat162float(p.bias[i]);
        x[i] = __float2bfloat16(v);
    }
}

// One CTA per token. absmax reduction → per-token scale, then FP8-quantize and
// scatter into the token's block slot.
__global__ void indexer_k_quant_append_kernel(IndexerKQuantAppendParams p) {
    const int t = blockIdx.x;
    if (t >= p.num_tokens) return;
    const __nv_bfloat16* k = p.k_in + static_cast<int64_t>(t) * p.index_head_dim;

    extern __shared__ float smem[];
    const int tid = threadIdx.x;

    float local = 0.0f;
    for (int i = tid; i < p.index_head_dim; i += blockDim.x)
        local = fmaxf(local, fabsf(__bfloat162float(k[i])));
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    const float absmax = smem[0];
    const float scale = absmax > 0.0f ? absmax / kFp8E4M3Max : 1.0f;
    const float inv_scale = absmax > 0.0f ? kFp8E4M3Max / absmax : 0.0f;

    const int slot = p.slot_mapping[t] + p.slot_bias;
    __nv_fp8_e4m3* dst = p.k_cache + static_cast<int64_t>(slot) * p.index_head_dim;
    for (int i = tid; i < p.index_head_dim; i += blockDim.x) {
        dst[i] = __nv_fp8_e4m3(__bfloat162float(k[i]) * inv_scale);
    }
    if (tid == 0) p.k_scales[slot] = scale;
}

__global__ void indexer_scale_weights_kernel(IndexerScaleWeightsParams p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = p.num_rows * p.n;
    if (i < total) p.out[i] = __bfloat162float(p.in[i]) * p.scale;
}

int block_dim_for(int dim) {
    int b = 32;
    while (b < dim && b < 1024) b <<= 1;
    return b;  // power of two ≤ 1024, ≥ 32 (reduction assumes power of two)
}

}  // namespace

void launch_indexer_layernorm_bias(const IndexerLayerNormParams& p, cudaStream_t stream) {
    if (p.num_rows == 0) return;
    if (p.dim <= 0 || !p.x || !p.weight || !p.bias)
        throw std::runtime_error("launch_indexer_layernorm_bias: invalid params");
    int bdim = block_dim_for(p.dim);
    indexer_layernorm_bias_kernel<<<p.num_rows, bdim, bdim * sizeof(float), stream>>>(p);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_indexer_layernorm_bias: ")
                                 + cudaGetErrorString(err));
}

void launch_indexer_k_quant_append(const IndexerKQuantAppendParams& p, cudaStream_t stream) {
    if (p.num_tokens == 0) return;
    if (p.index_head_dim <= 0 || !p.k_in || !p.k_cache || !p.k_scales || !p.slot_mapping)
        throw std::runtime_error("launch_indexer_k_quant_append: invalid params");
    int bdim = block_dim_for(p.index_head_dim);
    indexer_k_quant_append_kernel<<<p.num_tokens, bdim, bdim * sizeof(float), stream>>>(p);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_indexer_k_quant_append: ")
                                 + cudaGetErrorString(err));
}

void launch_indexer_scale_weights(const IndexerScaleWeightsParams& p, cudaStream_t stream) {
    const int total = p.num_rows * p.n;
    if (total <= 0) return;
    if (!p.in || !p.out) throw std::runtime_error("launch_indexer_scale_weights: null");
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    indexer_scale_weights_kernel<<<blocks, threads, 0, stream>>>(p);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_indexer_scale_weights: ")
                                 + cudaGetErrorString(err));
}

}  // namespace layerstorm::compute
