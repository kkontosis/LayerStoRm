// RMSNorm CUDA kernels for LayerStoRm.
// Adapted from TRT-LLM rmsnormKernels.cu and vLLM layernorm_kernels.cu
// (both Apache-2.0).
//
// SM120 (GeForce Blackwell) improvement over previous SM100-era implementation:
//   · Warp-shuffle block reduce (from TRT-LLM) instead of CUB BlockReduce<1024>
//     — eliminates fixed 1024-thread shared memory allocation, reduces shmem
//     footprint, increases CTA occupancy on SM120

#include "compute/kernels/norm/rmsnorm.h"
#include "compute/kernels/smxx/norm/vec_types.cuh"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace layerstorm::compute {

// ── Warp / block reduce helpers (TRT-LLM rmsnormKernels.cu pattern) ─────────

/// Intra-warp sum reduction. All 32 lanes receive the result.
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

/// Block-level sum-of-squares → rsqrtf → broadcast to all threads.
/// Uses only 32 floats of shared memory (one per warp) for inter-warp
/// communication — no fixed-size CUB TempStorage. Result is written to
/// s_warp[0] and returned. Two __syncthreads() inside.
// NVCC SM120 bug: __forceinline__ + __restrict__ on the shared memory pointer
// lets the compiler cache s_warp[0] in a register across __syncthreads(),
// causing non-thread-0 lanes to read stale values instead of the rsqrtf result.
// __noinline__ prevents this miscompilation.
__device__ __noinline__ float block_rms_reduce(float sos, int hidden_size,
                                                float epsilon,
                                                float* s_warp) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;

    // Intra-warp sum; all lanes hold the warp partial sum.
    sos = warp_reduce_sum(sos);
    if (lane_id == 0)
        s_warp[warp_id] = sos;
    __syncthreads();  // barrier 1: all warp sums visible

    // Thread 0 reduces warp sums and computes rsqrtf.
    if (threadIdx.x == 0) {
        float total = 0.f;
        // blockDim.x is always a multiple of 32 (enforced by dispatcher).
        const int nwarps = blockDim.x / 32;
        for (int w = 0; w < nwarps; ++w)
            total += s_warp[w];
        s_warp[0] = rsqrtf(total / static_cast<float>(hidden_size) + epsilon);
    }
    __syncthreads();  // barrier 2: inv_rms in s_warp[0] visible to all

    return s_warp[0];
}

// ── Plain RMSNorm kernel ─────────────────────────────────────────────────────
// One block per token. Two passes over global memory:
//   Pass 1: vectorized read → accumulate sum of squares
//   Pass 2: vectorized read → normalize × weight → write
//
// Static shared: s_warp[32] — warp reduce scratch / inv_rms broadcast

template <typename scalar_t, int VEC_SIZE>
__global__ void rms_norm_kernel(scalar_t* __restrict__ out,
                                const scalar_t* __restrict__ input,
                                const scalar_t* __restrict__ weight,
                                const float epsilon, const int num_tokens,
                                const int hidden_size, const int row_stride) {
    __shared__ float s_warp[32];

    const scalar_t* input_row =
        input + static_cast<int64_t>(blockIdx.x) * row_stride;

    // Pass 1: accumulate sum of squares via vectorized global reads.
    using vec_t = vec_n_t<scalar_t, VEC_SIZE>;
    const int num_vec = hidden_size / VEC_SIZE;
    const auto* v_in = reinterpret_cast<const vec_t*>(input_row);

    float variance = 0.f;
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        vec_t v = v_in[i];
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            float x = static_cast<float>(v.val[j]);
            variance += x * x;
        }
    }

    // Block-level reduce → rsqrtf → broadcast (2 __syncthreads inside).
    const float inv_rms = block_rms_reduce(variance, hidden_size, epsilon, s_warp);

    // Pass 2: normalize and scale (second vectorized global read).
    const auto* v_w  = reinterpret_cast<const vec_t*>(weight);
    auto*       v_out = reinterpret_cast<vec_t*>(
        out + static_cast<int64_t>(blockIdx.x) * row_stride);

    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        vec_t src = v_in[i];
        vec_t wt  = v_w[i];
        vec_t dst;
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            float x = static_cast<float>(src.val[j]);
            dst.val[j] = static_cast<scalar_t>(x * inv_rms) * wt.val[j];
        }
        v_out[i] = dst;
    }
}

// ── Fused add+RMSNorm: optimized width-8 path for half/bfloat16 ────────────
// Adapted from TRT-LLM and vLLM. Warp-shuffle block reduce replaces CUB.
// Static shared: s_warp[32] for warp reduce scratch (no dynamic smem needed).

template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && TypeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(scalar_t* __restrict__ out,
                          const scalar_t* __restrict__ input,
                          scalar_t* __restrict__ residual,
                          const scalar_t* __restrict__ weight,
                          const float epsilon, const int num_tokens,
                          const int hidden_size) {
    static_assert(sizeof(f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);
    __shared__ float s_warp[32];

    const int vec_hidden_size = hidden_size / width;

    auto* __restrict__ out_v      = reinterpret_cast<f16Vec<scalar_t, width>*>(out);
    const auto* __restrict__ in_v = reinterpret_cast<const f16Vec<scalar_t, width>*>(input);
    auto* __restrict__ res_v      = reinterpret_cast<f16Vec<scalar_t, width>*>(residual);
    const auto* __restrict__ wt_v = reinterpret_cast<const f16Vec<scalar_t, width>*>(weight);

    // Pass 1: residual += input, accumulate variance.
    float variance = 0.f;
    for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
        const int id = blockIdx.x * vec_hidden_size + idx;
        f16Vec<scalar_t, width> temp = in_v[id];
        temp += res_v[id];
        variance += temp.sum_squares();
        res_v[id] = temp;
    }

    const float inv_rms = block_rms_reduce(variance, hidden_size, epsilon, s_warp);

    // Pass 2: normalize residual, scale by weight, write output.
    for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
        const int id = blockIdx.x * vec_hidden_size + idx;
        f16Vec<scalar_t, width> temp = res_v[id];
        temp *= inv_rms;
        temp *= wt_v[idx];
        out_v[id] = temp;
    }
}

// ── Fused add+RMSNorm: generic scalar fallback ─────────────────────────────

template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !TypeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(scalar_t* __restrict__ out,
                          const scalar_t* __restrict__ input,
                          scalar_t* __restrict__ residual,
                          const scalar_t* __restrict__ weight,
                          const float epsilon, const int num_tokens,
                          const int hidden_size) {
    __shared__ float s_warp[32];

    // Pass 1: residual += input, accumulate variance.
    float variance = 0.f;
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        scalar_t z = input[blockIdx.x * hidden_size + idx];
        z += residual[blockIdx.x * hidden_size + idx];
        const float x = static_cast<float>(z);
        variance += x * x;
        residual[blockIdx.x * hidden_size + idx] = z;
    }

    const float inv_rms = block_rms_reduce(variance, hidden_size, epsilon, s_warp);

    // Pass 2: normalize residual, scale by weight, write output.
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        const float x = static_cast<float>(
            residual[blockIdx.x * hidden_size + idx]);
        out[blockIdx.x * hidden_size + idx] =
            static_cast<scalar_t>(x * inv_rms) * weight[idx];
    }
}

// ── Launch helpers ──────────────────────────────────────────────────────────

namespace {

template <typename scalar_t>
void dispatch_rmsnorm(void* out, const void* input, const void* weight,
                      float epsilon, int num_tokens, int hidden_size,
                      int row_stride, cudaStream_t stream) {
    constexpr int max_vec =
        static_cast<int>(16 / sizeof(scalar_t));  // 4 for fp32, 8 for fp16/bf16
    const int vec_size = std::gcd(max_vec, hidden_size);
    const int max_block_size = (num_tokens < 256) ? 1024 : 256;
    // Always use a multiple of 32 (full warps only) so the warp-shuffle block
    // reduce operates correctly with 0xffffffff masks. Extra threads contribute
    // variance = 0 and do not affect correctness.
    const int raw_block = std::min(hidden_size / vec_size, max_block_size);
    const int block_size = std::max(32, (raw_block + 31) & ~31);

    dim3 grid(num_tokens);
    dim3 block(block_size);

    auto* o = static_cast<scalar_t*>(out);
    auto* i = static_cast<const scalar_t*>(input);
    auto* w = static_cast<const scalar_t*>(weight);

    // Check actual pointer alignment — pinned-region offsets may not satisfy
    // vectorized access requirements even when hidden_size is divisible.
    // A strided layout must also keep every ROW aligned: row_stride bytes
    // must be a multiple of the vector width or later rows fall off alignment.
    const int req_align = vec_size * static_cast<int>(sizeof(scalar_t));
    const bool aligned =
        (reinterpret_cast<uintptr_t>(o) % req_align == 0) &&
        (reinterpret_cast<uintptr_t>(i) % req_align == 0) &&
        (reinterpret_cast<uintptr_t>(w) % req_align == 0) &&
        ((static_cast<size_t>(row_stride) * sizeof(scalar_t)) % req_align == 0);
    const int effective_vec = aligned ? vec_size : 1;

    switch (effective_vec) {
        case 8:
            rms_norm_kernel<scalar_t, 8>
                <<<grid, block, 0, stream>>>(
                    o, i, w, epsilon, num_tokens, hidden_size, row_stride);
            break;
        case 4:
            rms_norm_kernel<scalar_t, 4>
                <<<grid, block, 0, stream>>>(
                    o, i, w, epsilon, num_tokens, hidden_size, row_stride);
            break;
        case 2:
            rms_norm_kernel<scalar_t, 2>
                <<<grid, block, 0, stream>>>(
                    o, i, w, epsilon, num_tokens, hidden_size, row_stride);
            break;
        default:
            rms_norm_kernel<scalar_t, 1>
                <<<grid, block, 0, stream>>>(
                    o, i, w, epsilon, num_tokens, hidden_size, row_stride);
            break;
    }
}

template <typename scalar_t>
void dispatch_fused_add_rmsnorm(void* out, const void* input, void* residual,
                                const void* weight, float epsilon,
                                int num_tokens, int hidden_size,
                                cudaStream_t stream) {
    const int max_block_size = (num_tokens < 256) ? 1024 : 256;
    dim3 grid(num_tokens);
    // Always a multiple of 32 (full warps) for warp-shuffle reduce correctness.
    const int raw_fused_block = std::min(hidden_size, max_block_size);
    dim3 block(std::max(32, (raw_fused_block + 31) & ~31));

    auto* o = static_cast<scalar_t*>(out);
    auto* i = static_cast<const scalar_t*>(input);
    auto* r = static_cast<scalar_t*>(residual);
    auto* w = static_cast<const scalar_t*>(weight);

    // Try width-8 vectorized path for fp16/bf16 with proper alignment.
    constexpr int vector_width = 8;
    if constexpr (TypeConvert<scalar_t>::exists &&
                  !std::is_same_v<scalar_t, float>) {
        constexpr int req_alignment =
            vector_width * static_cast<int>(sizeof(scalar_t));
        const bool aligned =
            (reinterpret_cast<uintptr_t>(o) % req_alignment == 0) &&
            (reinterpret_cast<uintptr_t>(i) % req_alignment == 0) &&
            (reinterpret_cast<uintptr_t>(r) % req_alignment == 0) &&
            (reinterpret_cast<uintptr_t>(w) % req_alignment == 0);
        const bool divisible = (hidden_size % vector_width == 0);

        if (aligned && divisible) {
            fused_add_rms_norm_kernel<scalar_t, vector_width>
                <<<grid, block, 0, stream>>>(
                    o, i, r, w, epsilon, num_tokens, hidden_size);
            return;
        }
    }

    // Scalar fallback (no dynamic smem needed for fused kernel).
    fused_add_rms_norm_kernel<scalar_t, 0>
        <<<grid, block, 0, stream>>>(o, i, r, w, epsilon, num_tokens,
                                     hidden_size);
}

}  // anonymous namespace

// ── Public API ──────────────────────────────────────────────────────────────

void launch_rmsnorm(void* out, const void* input, const void* weight,
                    float epsilon, int num_tokens, int hidden_size,
                    NormDtype dtype, void* stream) {
    launch_rmsnorm_strided(out, input, weight, epsilon, num_tokens,
                           hidden_size, /*row_stride=*/hidden_size, dtype,
                           stream);
}

void launch_rmsnorm_strided(void* out, const void* input, const void* weight,
                            float epsilon, int num_tokens, int hidden_size,
                            int row_stride, NormDtype dtype, void* stream) {
    if (row_stride < hidden_size) {
        throw std::invalid_argument(
            "launch_rmsnorm_strided: row_stride < hidden_size");
    }
    // Sticky-error discriminator (see spec/DEBUG.md, cudaHostRegister
    // partial-overlap entry): cudaGetLastError is per-thread and STICKY, so a
    // failure from an earlier UNCHECKED async call would otherwise be blamed
    // on this launch. Separate the two cases explicitly.
    {
        cudaError_t pre = cudaGetLastError();
        if (pre != cudaSuccess) {
            char buf[512];
            snprintf(buf, sizeof(buf),
                     "launch_rmsnorm: PRE-EXISTING sticky error '%s' "
                     "(out=%p in=%p w=%p tokens=%d hidden=%d stride=%d "
                     "stream=%p)",
                     cudaGetErrorString(pre), out, input, weight, num_tokens,
                     hidden_size, row_stride, stream);
            throw std::runtime_error(buf);
        }
    }
    auto s = static_cast<cudaStream_t>(stream);
    switch (dtype) {
        case NormDtype::kFloat32:
            dispatch_rmsnorm<float>(out, input, weight, epsilon, num_tokens,
                                    hidden_size, row_stride, s);
            break;
        case NormDtype::kBFloat16:
            dispatch_rmsnorm<__nv_bfloat16>(out, input, weight, epsilon,
                                            num_tokens, hidden_size,
                                            row_stride, s);
            break;
        case NormDtype::kFloat16:
            dispatch_rmsnorm<__half>(out, input, weight, epsilon, num_tokens,
                                     hidden_size, row_stride, s);
            break;
        default:
            throw std::invalid_argument("launch_rmsnorm: unsupported dtype");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "launch_rmsnorm failed: %s (out=%p in=%p w=%p tokens=%d "
                 "hidden=%d stride=%d stream=%p)",
                 cudaGetErrorString(err), out, input, weight, num_tokens,
                 hidden_size, row_stride, stream);
        throw std::runtime_error(buf);
    }
}

void launch_fused_add_rmsnorm(void* out, const void* input, void* residual,
                              const void* weight, float epsilon,
                              int num_tokens, int hidden_size, NormDtype dtype,
                              void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    switch (dtype) {
        case NormDtype::kFloat32:
            dispatch_fused_add_rmsnorm<float>(out, input, residual, weight,
                                              epsilon, num_tokens, hidden_size,
                                              s);
            break;
        case NormDtype::kBFloat16:
            dispatch_fused_add_rmsnorm<__nv_bfloat16>(
                out, input, residual, weight, epsilon, num_tokens, hidden_size,
                s);
            break;
        case NormDtype::kFloat16:
            dispatch_fused_add_rmsnorm<__half>(out, input, residual, weight,
                                               epsilon, num_tokens, hidden_size,
                                               s);
            break;
        default:
            throw std::invalid_argument(
                "launch_fused_add_rmsnorm: unsupported dtype");
    }
}

}  // namespace layerstorm::compute
