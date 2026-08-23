// GPU-side confidence estimation kernel for LayerStoRm.
//
// Computes top-1 softmax probability and normalized Shannon entropy
// from raw logits.  One block per token, 256 threads.  Each token is
// processed independently.
//
// Algorithm (3-pass over logits per token):
//   Pass 1: Find max logit (numerical stability).
//   Pass 2: Compute exp(x - max), accumulate sum, track max_exp.
//   Pass 3: Compute p * log(p) entropy sum using known normalizer.
//
// Does NOT modify the input logits buffer.
//
// Paper motivation:
//   Kangaroo — confidence-based draft termination: max(softmax) <= eta
//   FLy      — entropy gate: h < theta (normalized to [0,1] via H/log(V))

#include "compute/kernels/confidence/confidence.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>

namespace layerstorm::compute {

// ── Constants ──────────────────────────────────────────────────────────────

static constexpr int kBlockSize = 256;

// ── Block reduce helpers ───────────────────────────────────────────────────
// Same warp/block reduce pattern as sampling kernel.

__device__ __forceinline__ float warp_reduce_max_conf(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_conf(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ float block_reduce_max_conf(float val, float* smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    val = warp_reduce_max_conf(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    const int num_warps = blockDim.x / 32;
    val = (threadIdx.x < static_cast<unsigned>(num_warps)) ? smem[threadIdx.x] : -FLT_MAX;
    if (warp_id == 0) val = warp_reduce_max_conf(val);

    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

__device__ float block_reduce_sum_conf(float val, float* smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    val = warp_reduce_sum_conf(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    const int num_warps = blockDim.x / 32;
    val = (threadIdx.x < static_cast<unsigned>(num_warps)) ? smem[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum_conf(val);

    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

// ── Confidence kernel ──────────────────────────────────────────────────────
// One block per token.  Three passes over the logit row.

__global__ void confidence_kernel(const float* __restrict__ logits,
                                  float* __restrict__ top1_probs,
                                  float* __restrict__ entropies,
                                  int vocab_size) {
    const int token_idx = blockIdx.x;
    const float* row = logits + static_cast<int64_t>(token_idx) * vocab_size;

    __shared__ float smem[8];  // up to 8 warps (256 threads / 32)

    // ── Pass 1: find max logit for numerical stability ─────────────────
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row[i]);
    }
    float block_max = block_reduce_max_conf(thread_max, smem);

    // ── Pass 2: compute exp(x - max), accumulate sum, track max_exp ───
    float thread_sum = 0.0f;
    float thread_max_exp = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float e = expf(row[i] - block_max);
        thread_sum += e;
        thread_max_exp = fmaxf(thread_max_exp, e);
    }
    float block_sum = block_reduce_sum_conf(thread_sum, smem);
    float block_max_exp = block_reduce_max_conf(thread_max_exp, smem);

    // ── Pass 3: compute entropy sum using known normalizer ─────────────
    float inv_sum = 1.0f / block_sum;
    float thread_entropy = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float e = expf(row[i] - block_max);
        float p = e * inv_sum;
        if (p > 0.0f) {
            thread_entropy -= p * logf(p);
        }
    }
    float block_entropy = block_reduce_sum_conf(thread_entropy, smem);

    // ── Write results ──────────────────────────────────────────────────
    // TODO:DEBT TD-56d: logf(vocab_size)==0 when vocab_size==1 → NaN entropy
    if (threadIdx.x == 0) {
        top1_probs[token_idx] = block_max_exp * inv_sum;
        entropies[token_idx]  = block_entropy / logf(static_cast<float>(vocab_size));
    }
}

// ── Public API ─────────────────────────────────────────────────────────────

void launch_compute_confidence(const float* logits,
                               float* top1_probs,
                               float* entropies,
                               int num_tokens, int vocab_size,
                               void* stream) {
    if (num_tokens <= 0) return;
    if (vocab_size <= 0) {
        throw std::invalid_argument(
            "launch_compute_confidence: vocab_size must be > 0");
    }

    auto cuda_stream = static_cast<cudaStream_t>(stream);

    // Shared memory: 8 floats for block reduction (8 warps max).
    confidence_kernel<<<num_tokens, kBlockSize, 0, cuda_stream>>>(
        logits, top1_probs, entropies, vocab_size);
}

}  // namespace layerstorm::compute
