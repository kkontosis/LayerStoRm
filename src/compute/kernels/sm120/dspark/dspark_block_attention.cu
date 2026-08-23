// DSpark DFlash non-causal block attention (DSP-3).  See dspark_backbone.h.
//
// Shape regime: num_query = gamma (<= 16) tiny, ctx_len up to ~100k, 64 heads
// x head_dim 64.  One block per (query, head); 64 threads split the key range
// with per-thread online softmax (m, l, acc[head_dim]) and a deterministic
// shared-memory tree merge — a flash-decoding split at toy scale.  Total
// work is a few GFLOP; bandwidth (one pass over ctx K/V) dominates.

#include "compute/kernels/dspark/dspark_backbone.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace layerstorm::compute {

namespace {

constexpr int kThreads = 64;
constexpr int kMaxHeadDim = 128;

__global__ void dspark_block_attention_kernel(
    __nv_bfloat16* out, const __nv_bfloat16* q, const __nv_bfloat16* ctx_k,
    const __nv_bfloat16* ctx_v, const __nv_bfloat16* blk_k,
    const __nv_bfloat16* blk_v, int num_query, int ctx_len, int n_heads,
    int head_dim, float scale) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    // Layout: sq[head_dim] | sm[kThreads] | sl[kThreads]
    //         | sacc[kThreads][head_dim]
    float* sq = smem;
    float* sm = sq + head_dim;
    float* sl = sm + kThreads;
    float* sacc = sl + kThreads;

    // Load this (query, head) row of q into shared FP32.
    const __nv_bfloat16* qrow =
        q + (static_cast<long long>(t) * n_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += kThreads)
        sq[d] = __bfloat162float(qrow[d]);
    __syncthreads();

    // Per-thread online softmax over keys tid, tid+kThreads, ...
    const int total = ctx_len + num_query;
    float m = -INFINITY;
    float l = 0.0f;
    float acc[kMaxHeadDim];
#pragma unroll 4
    for (int d = 0; d < head_dim; ++d) acc[d] = 0.0f;

    for (int j = tid; j < total; j += kThreads) {
        const bool in_ctx = j < ctx_len;
        const int r = in_ctx ? j : j - ctx_len;
        const __nv_bfloat16* krow =
            (in_ctx ? ctx_k : blk_k) +
            (static_cast<long long>(r) * n_heads + h) * head_dim;
        const __nv_bfloat16* vrow =
            (in_ctx ? ctx_v : blk_v) +
            (static_cast<long long>(r) * n_heads + h) * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += sq[d] * __bfloat162float(krow[d]);
        const float s = dot * scale;

        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new);   // 0 when m == -inf (exp(-inf))
        const float w = expf(s - m_new);
        l = l * corr + w;
        for (int d = 0; d < head_dim; ++d)
            acc[d] = acc[d] * corr + w * __bfloat162float(vrow[d]);
        m = m_new;
    }

    // Publish per-thread partials.
    sm[tid] = m;
    sl[tid] = l;
    for (int d = 0; d < head_dim; ++d)
        sacc[tid * head_dim + d] = acc[d];
    __syncthreads();

    // Deterministic pairwise tree merge (fixed order).
    for (int off = kThreads / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float m1 = sm[tid], m2 = sm[tid + off];
            const float mm = fmaxf(m1, m2);
            // exp(-inf - -inf) guards: when both are -inf the weights are
            // exp(-inf)=0 and the merged l stays 0.
            const float c1 = (m1 == -INFINITY) ? 0.0f : expf(m1 - mm);
            const float c2 = (m2 == -INFINITY) ? 0.0f : expf(m2 - mm);
            sm[tid] = mm;
            sl[tid] = sl[tid] * c1 + sl[tid + off] * c2;
            float* a1 = sacc + tid * head_dim;
            const float* a2 = sacc + (tid + off) * head_dim;
            for (int d = 0; d < head_dim; ++d)
                a1[d] = a1[d] * c1 + a2[d] * c2;
        }
        __syncthreads();
    }

    // Write out[t, h*head_dim + d] = acc / l.
    const float inv_l = (sl[0] > 0.0f) ? (1.0f / sl[0]) : 0.0f;
    __nv_bfloat16* orow =
        out + (static_cast<long long>(t) * n_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += kThreads)
        orow[d] = __float2bfloat16(sacc[d] * inv_l);
}

}  // namespace

void launch_dspark_block_attention(void* out, const void* q,
                                   const void* ctx_k, const void* ctx_v,
                                   const void* blk_k, const void* blk_v,
                                   int num_query, int ctx_len, int n_heads,
                                   int head_dim, float scale, void* stream) {
    if (num_query <= 0 || head_dim > kMaxHeadDim) return;
    const dim3 grid(static_cast<unsigned>(n_heads),
                    static_cast<unsigned>(num_query));
    const size_t smem_bytes =
        static_cast<size_t>(head_dim + 2 * kThreads +
                            kThreads * head_dim) * sizeof(float);
    dspark_block_attention_kernel<<<grid, kThreads, smem_bytes,
                                    static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(q),
        static_cast<const __nv_bfloat16*>(ctx_k),
        static_cast<const __nv_bfloat16*>(ctx_v),
        static_cast<const __nv_bfloat16*>(blk_k),
        static_cast<const __nv_bfloat16*>(blk_v), num_query, ctx_len, n_heads,
        head_dim, scale);
}

}  // namespace layerstorm::compute
