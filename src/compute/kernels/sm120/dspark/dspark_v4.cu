// DSpark V4 dflash draft kernels (ticket J). See dspark/dspark_v4.h.
//
// Shape regime: nq <= 16 query rows, key count <= window(128) + nq — tiny;
// head_dim 512 rules out per-thread accumulator rows (dspark_block_attention
// style), so the attention kernel stages ALL key scores in shared memory
// (two-pass softmax, fixed reduction order — deterministic) and each thread
// then accumulates strided output dims over the same fixed key order.

#include "compute/kernels/dspark/dspark_v4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>

namespace layerstorm::compute {

namespace {

constexpr int kRopeThreads = 64;

__global__ void dspark_v4_kv_rope_kernel(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ in,
    const float* __restrict__ cos_sin, int base_pos, int rows, int head_dim,
    int rope_dim) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int nope_dim = head_dim - rope_dim;
    const int half = rope_dim / 2;
    const __nv_bfloat16* src = in + (long long)row * head_dim;
    __nv_bfloat16* dst = out + (long long)row * head_dim;
    // Nope prefix copy (skipped when out aliases in).
    if (dst != src) {
        for (int i = threadIdx.x; i < nope_dim; i += kRopeThreads)
            dst[i] = src[i];
    }
    const float* cs = cos_sin + (long long)(base_pos + row) * rope_dim;
    for (int i = threadIdx.x; i < half; i += kRopeThreads) {
        const float e = __bfloat162float(src[nope_dim + 2 * i]);
        const float o = __bfloat162float(src[nope_dim + 2 * i + 1]);
        const float c = __ldg(cs + i);
        const float s = __ldg(cs + half + i);
        dst[nope_dim + 2 * i] = __float2bfloat16_rn(e * c - o * s);
        dst[nope_dim + 2 * i + 1] = __float2bfloat16_rn(e * s + o * c);
    }
}

constexpr int kAttnThreads = 128;
constexpr int kMaxKeys = 160;  // window(128) + block(16) + slack

__global__ void dspark_v4_attention_kernel(
    __nv_bfloat16* __restrict__ out, float* __restrict__ lse,
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_rope,
    const __nv_bfloat16* __restrict__ ctx_kv,
    const __nv_bfloat16* __restrict__ blk_kv, int nq, int n_ctx, int base_pos,
    int window, int h_q, int head_dim, int rope_dim, float scale) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    // Layout: sq[head_dim] | sqr[rope_dim] | sscore[kMaxKeys] | sred[2]
    float* sq = smem;
    float* sqr = sq + head_dim;
    float* sscore = sqr + rope_dim;
    float* sred = sscore + kMaxKeys;

    const int p = base_pos + t;
    const int c0 = max(0, p - window + 1);
    const int nc = max(0, n_ctx - c0);
    const int total = nc + nq;

    // Stage this (query, head)'s q into shared FP32.
    const __nv_bfloat16* qn = q_nope + ((long long)t * h_q + h) * head_dim;
    const __nv_bfloat16* qr = q_rope + ((long long)t * h_q + h) * rope_dim;
    for (int d = tid; d < head_dim; d += kAttnThreads)
        sq[d] = __bfloat162float(qn[d]);
    for (int d = tid; d < rope_dim; d += kAttnThreads)
        sqr[d] = __bfloat162float(qr[d]);
    __syncthreads();

    // Pass 1: scaled scores into shared (strided keys per thread).
    const int nope_off = head_dim - rope_dim;
    for (int j = tid; j < total; j += kAttnThreads) {
        const bool in_ctx = j < nc;
        const __nv_bfloat16* krow =
            in_ctx ? ctx_kv + (long long)(c0 + j) * head_dim
                   : blk_kv + (long long)(j - nc) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += sq[d] * __bfloat162float(krow[d]);
        for (int d = 0; d < rope_dim; ++d)
            dot += sqr[d] * __bfloat162float(krow[nope_off + d]);
        sscore[j] = dot * scale;
    }
    __syncthreads();

    // Softmax over the FIXED key order (thread 0 serial over <= kMaxKeys
    // entries — deterministic and cheap at this scale).
    if (tid == 0) {
        float m = -INFINITY;
        for (int j = 0; j < total; ++j) m = fmaxf(m, sscore[j]);
        float l = 0.0f;
        for (int j = 0; j < total; ++j) {
            const float w = expf(sscore[j] - m);
            sscore[j] = w;
            l += w;
        }
        sred[0] = m;
        sred[1] = l;
    }
    __syncthreads();
    const float m = sred[0];
    const float l = sred[1];
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;

    // Pass 2: each thread owns strided output dims; V == K (latent rows).
    __nv_bfloat16* orow = out + ((long long)t * h_q + h) * head_dim;
    for (int d = tid; d < head_dim; d += kAttnThreads) {
        float acc = 0.0f;
        for (int j = 0; j < total; ++j) {
            const __nv_bfloat16* vrow =
                (j < nc) ? ctx_kv + (long long)(c0 + j) * head_dim
                         : blk_kv + (long long)(j - nc) * head_dim;
            acc += sscore[j] * __bfloat162float(vrow[d]);
        }
        orow[d] = __float2bfloat16_rn(acc * inv_l);
    }
    if (tid == 0) lse[(long long)t * h_q + h] = m + logf(l);
}

}  // namespace

void launch_dspark_v4_kv_rope(void* out, const void* in, const void* cos_sin,
                              int base_pos, int rows, int head_dim,
                              int rope_dim, void* stream) {
    if (rows <= 0) return;
    dspark_v4_kv_rope_kernel<<<static_cast<unsigned>(rows), kRopeThreads, 0,
                               static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(in),
        static_cast<const float*>(cos_sin), base_pos, rows, head_dim,
        rope_dim);
}

void launch_dspark_v4_attention(void* out, float* lse, const void* q_nope,
                                const void* q_rope, const void* ctx_kv,
                                const void* blk_kv, int nq, int n_ctx,
                                int base_pos, int window, int h_q,
                                int head_dim, int rope_dim, float scale,
                                void* stream) {
    if (nq <= 0) return;
    const dim3 grid(static_cast<unsigned>(h_q), static_cast<unsigned>(nq));
    const size_t smem_bytes =
        static_cast<size_t>(head_dim + rope_dim + kMaxKeys + 2) *
        sizeof(float);
    dspark_v4_attention_kernel<<<grid, kAttnThreads, smem_bytes,
                                 static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(out), lse,
        static_cast<const __nv_bfloat16*>(q_nope),
        static_cast<const __nv_bfloat16*>(q_rope),
        static_cast<const __nv_bfloat16*>(ctx_kv),
        static_cast<const __nv_bfloat16*>(blk_kv), nq, n_ctx, base_pos,
        window, h_q, head_dim, rope_dim, scale);
}

}  // namespace layerstorm::compute
