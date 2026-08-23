// DSpark sequential Markov head kernels (DSP-4).
// See compute/kernels/dspark/dspark_markov.h for the contract
// (INV-DSPARK-MARKOV: bias = markov_w2 @ markov_w1[x_{k-1}], greedy argmax,
// on-device sequential chain, deterministic reduction order).

#include "compute/kernels/dspark/dspark_markov.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace layerstorm::compute {

namespace {

constexpr int kThreads = 256;  // 8 warps
constexpr int kWarps = kThreads / 32;

/// Argmax ordering: higher value wins; on EXACT float equality the LOWER
/// index wins (torch.argmax first-occurrence semantics).
__device__ __forceinline__ bool better(float v, int i, float best_v,
                                       int best_i) {
    return v > best_v || (v == best_v && i < best_i);
}

// One block covers kDsparkMarkovRowsPerBlock vocab rows.  Warp w handles
// rows row0 + w, row0 + w + kWarps, ... (increasing order per warp).  Each
// row's bias is a lane-strided BF16 dot against e (shared, FP32) followed by
// a butterfly reduce — fixed order, bit-deterministic run to run.
__global__ void dspark_markov_bias_argmax_kernel(
        float* __restrict__ corrected, const float* __restrict__ base,
        const __nv_bfloat16* __restrict__ w2,
        const __nv_bfloat16* __restrict__ e,
        DsparkMarkovPartial* __restrict__ partials, int vocab, int rank) {
    extern __shared__ float e_f[];  // [rank]
    for (int i = threadIdx.x; i < rank; i += blockDim.x)
        e_f[i] = __bfloat162float(e[i]);
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * kDsparkMarkovRowsPerBlock;
    const int row_end = min(row0 + kDsparkMarkovRowsPerBlock, vocab);

    float best = -INFINITY;
    int best_idx = INT_MAX;
    for (int row = row0 + warp; row < row_end; row += kWarps) {
        const __nv_bfloat16* wrow =
            w2 + static_cast<int64_t>(row) * rank;
        float acc = 0.0f;
        for (int i = lane; i < rank; i += 32)
            acc += __bfloat162float(wrow[i]) * e_f[i];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        const float logit = base[row] + acc;
        if (lane == 0) {
            corrected[row] = logit;
            if (better(logit, row, best, best_idx)) {
                best = logit;
                best_idx = row;
            }
        }
    }

    // Per-warp bests (lane 0 of each warp) -> block best.  Warps interleave
    // rows, so the explicit index tie-break keeps first-occurrence exact.
    __shared__ float s_val[kWarps];
    __shared__ int s_idx[kWarps];
    if (lane == 0) {
        s_val[warp] = best;
        s_idx[warp] = best_idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bv = s_val[0];
        int bi = s_idx[0];
#pragma unroll
        for (int w = 1; w < kWarps; ++w)
            if (better(s_val[w], s_idx[w], bv, bi)) {
                bv = s_val[w];
                bi = s_idx[w];
            }
        partials[blockIdx.x] = DsparkMarkovPartial{bv, bi};
    }
}

// Single-block final reduce: global argmax over the per-block partials,
// the draft->target id map (d2t, identity when nullptr — TD-DSPARK-VOCAB-
// REMAP), then (unless last step) the markov_w1 gather for the next step —
// the sampled token id never leaves the device.
__global__ void dspark_markov_finalize_kernel(
        int32_t* __restrict__ token_out, __nv_bfloat16* __restrict__ e_next,
        const DsparkMarkovPartial* __restrict__ partials, int num_blocks,
        const __nv_bfloat16* __restrict__ w1, int rank,
        const int64_t* __restrict__ d2t) {
    __shared__ float s_val[kThreads];
    __shared__ int s_idx[kThreads];

    float bv = -INFINITY;
    int bi = INT_MAX;
    for (int b = threadIdx.x; b < num_blocks; b += kThreads) {
        const DsparkMarkovPartial p = partials[b];
        if (better(p.val, p.idx, bv, bi)) {
            bv = p.val;
            bi = p.idx;
        }
    }
    s_val[threadIdx.x] = bv;
    s_idx[threadIdx.x] = bi;
    __syncthreads();
#pragma unroll
    for (int off = kThreads / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off &&
            better(s_val[threadIdx.x + off], s_idx[threadIdx.x + off],
                   s_val[threadIdx.x], s_idx[threadIdx.x])) {
            s_val[threadIdx.x] = s_val[threadIdx.x + off];
            s_idx[threadIdx.x] = s_idx[threadIdx.x + off];
        }
        __syncthreads();
    }

    // Draft-vocab argmax -> target-vocab id (identity when d2t is null).
    // Every thread computes the same mapped id (the redundant d2t load hits
    // L1/L2); markov_w1 is target-vocab, so the e-gather uses the mapped id.
    const int draft_token = s_idx[0];
    const int64_t token =
        static_cast<int64_t>(draft_token) +
        (d2t != nullptr ? d2t[draft_token] : int64_t{0});
    if (threadIdx.x == 0) *token_out = static_cast<int32_t>(token);
    if (e_next != nullptr)
        for (int i = threadIdx.x; i < rank; i += kThreads)
            e_next[i] = w1[token * rank + i];
}

}  // namespace

void launch_dspark_markov_bias_argmax(float* corrected, const float* base,
                                      const void* w2, const void* e,
                                      void* partials, int vocab, int rank,
                                      void* stream) {
    if (vocab <= 0 || rank <= 0) return;
    const int blocks = dspark_markov_num_blocks(vocab);
    const size_t smem = static_cast<size_t>(rank) * sizeof(float);
    dspark_markov_bias_argmax_kernel<<<blocks, kThreads, smem,
                                       static_cast<cudaStream_t>(stream)>>>(
        corrected, base, static_cast<const __nv_bfloat16*>(w2),
        static_cast<const __nv_bfloat16*>(e),
        static_cast<DsparkMarkovPartial*>(partials), vocab, rank);
}

void launch_dspark_markov_finalize(int32_t* token_out, void* e_next,
                                   const void* partials, int num_blocks,
                                   const void* w1, int rank,
                                   const void* d2t, void* stream) {
    if (num_blocks <= 0 || rank <= 0) return;
    dspark_markov_finalize_kernel<<<1, kThreads, 0,
                                    static_cast<cudaStream_t>(stream)>>>(
        token_out, static_cast<__nv_bfloat16*>(e_next),
        static_cast<const DsparkMarkovPartial*>(partials), num_blocks,
        static_cast<const __nv_bfloat16*>(w1), rank,
        static_cast<const int64_t*>(d2t));
}

}  // namespace layerstorm::compute
