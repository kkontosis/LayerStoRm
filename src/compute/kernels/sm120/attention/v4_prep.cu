// DeepSeek-V4 attention prep/compress kernels (V4-5a) — SM120 implementation.
// Contracts + model references in compute/kernels/attention/v4_prep.h.
//
// Math ported from ref/llama.cpp/src/models/deepseek4.cpp (MIT, llama.cpp
// authors): build_attention q/kv prep (:826-868),
// build_overlap_compressed_kv_from_state (:440), build_hca_compressed_kv_
// from_state (:382), ggml_rope_ext_back inverse rotation (:1061), attention
// sinks per build_attn_mha (softmax-denominator extra logit). FP8 entry
// layout + scale convention match deps/LayerStoRmKernels
// csrc/sm120/decode/csa_fp8/params.h V4CacheLayout and
// csrc/sm120/prep/v4_fp8_k_append.cu (scale = amax/448).

#include "compute/kernels/attention/v4_prep.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

constexpr int kThreads = 256;
constexpr float kFp8Max = 448.0f;

void check_launch(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " launch failed: " +
                                 cudaGetErrorString(err));
    }
}

// Block-wide sum reduction of one value; result valid on all threads.
__device__ float block_reduce_sum(float v, float* s_warp) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) s_warp[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float x = (lane < kThreads / 32) ? s_warp[lane] : 0.0f;
#pragma unroll
        for (int off = (kThreads / 32) / 2; off > 0; off >>= 1)
            x += __shfl_down_sync(0xffffffffu, x, off);
        if (lane == 0) s_warp[0] = x;
    }
    __syncthreads();
    const float r = s_warp[0];
    __syncthreads();
    return r;
}

__device__ float block_reduce_max(float v, float* s_warp) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    if (lane == 0) s_warp[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float x = (lane < kThreads / 32) ? s_warp[lane] : -INFINITY;
#pragma unroll
        for (int off = (kThreads / 32) / 2; off > 0; off >>= 1)
            x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, off));
        if (lane == 0) s_warp[0] = x;
    }
    __syncthreads();
    const float r = s_warp[0];
    __syncthreads();
    return r;
}

// ── Q prep ───────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(kThreads) v4_q_prep_kernel(
    __nv_bfloat16* __restrict__ q_nope_out,
    __nv_bfloat16* __restrict__ q_rope_out,
    const __nv_bfloat16* __restrict__ q_in,
    const int* __restrict__ positions,
    const float* __restrict__ cos_sin,
    float rms_eps, int num_tokens, int h_q, int head_dim, int rope_dim) {
    const int row = blockIdx.x;  // token*h_q + head
    if (row >= num_tokens * h_q) return;
    const int token = row / h_q;
    const int nope_dim = head_dim - rope_dim;

    const __nv_bfloat16* src = q_in + (int64_t)row * head_dim;
    __shared__ float s_warp[kThreads / 32];

    // Parameterless RMS over the full head_dim vector (pre-rope).
    float sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += kThreads) {
        const float v = __bfloat162float(src[i]);
        sq += v * v;
    }
    const float sqrsum = block_reduce_sum(sq, s_warp);
    const float inv_rms = rsqrtf(sqrsum / (float)head_dim + rms_eps);

    __nv_bfloat16* qn = q_nope_out + (int64_t)row * head_dim;
    __nv_bfloat16* qr = q_rope_out + (int64_t)row * rope_dim;

    // Nope part + zero pad.
    for (int i = threadIdx.x; i < head_dim; i += kThreads) {
        if (i < nope_dim) {
            qn[i] = __float2bfloat16_rn(__bfloat162float(src[i]) * inv_rms);
        } else {
            qn[i] = __float2bfloat16_rn(0.0f);
        }
    }
    // Rope part: interleaved pairs at positions[token].
    const int half = rope_dim / 2;
    const int pos = __ldg(positions + token);
    const float* cs = cos_sin + (int64_t)pos * rope_dim;
    for (int i = threadIdx.x; i < half; i += kThreads) {
        const float e = __bfloat162float(src[nope_dim + 2 * i]) * inv_rms;
        const float o = __bfloat162float(src[nope_dim + 2 * i + 1]) * inv_rms;
        const float c = __ldg(cs + i);
        const float s = __ldg(cs + half + i);
        qr[2 * i] = __float2bfloat16_rn(e * c - o * s);
        qr[2 * i + 1] = __float2bfloat16_rn(e * s + o * c);
    }
}

// ── FP8 entry write helper (duplicated-rope layout) ──────────────────────
// vals[] = per-thread channels of the roped 512 vector (2 per thread).
// Writes k_nope/k_scale/k_rope/v_nope/v_scale of one 1160-B entry.
__device__ void v4_write_fp8_entry(uint8_t* __restrict__ entry,
                                   float v0, float v1, int d0, int head_dim,
                                   int rope_dim, float* s_warp) {
    const float amax_local = fmaxf(fabsf(v0), fabsf(v1));
    const float amax = block_reduce_max(amax_local, s_warp);
    const float scale = amax / kFp8Max;
    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    const int k_scale_off = head_dim;
    const int k_rope_off = head_dim + 4;
    const int v_nope_off = k_rope_off + rope_dim * 2;
    const int v_scale_off = v_nope_off + head_dim;

    auto* k8 = reinterpret_cast<__nv_fp8_e4m3*>(entry);
    auto* v8 = reinterpret_cast<__nv_fp8_e4m3*>(entry + v_nope_off);
    auto* kr = reinterpret_cast<__nv_bfloat16*>(entry + k_rope_off);
    if (d0 < head_dim) {
        const float q0 = fmaxf(-kFp8Max, fminf(kFp8Max, v0 * inv_scale));
        const float q1 = fmaxf(-kFp8Max, fminf(kFp8Max, v1 * inv_scale));
        k8[d0] = __nv_fp8_e4m3(q0);
        k8[d0 + 1] = __nv_fp8_e4m3(q1);
        v8[d0] = __nv_fp8_e4m3(q0);
        v8[d0 + 1] = __nv_fp8_e4m3(q1);
        // Duplicate the roped pe as BF16 k_rope.
        const int nope_dim = head_dim - rope_dim;
        if (d0 >= nope_dim) {
            kr[d0 - nope_dim] = __float2bfloat16_rn(v0);
            kr[d0 + 1 - nope_dim] = __float2bfloat16_rn(v1);
        }
    }
    if (threadIdx.x == 0) {
        *reinterpret_cast<float*>(entry + k_scale_off) = scale;
        *reinterpret_cast<float*>(entry + v_scale_off) = scale;
    }
}

// ── Raw KV append ────────────────────────────────────────────────────────

__global__ void __launch_bounds__(kThreads) v4_raw_kv_append_kernel(
    const __nv_bfloat16* __restrict__ kv_in,
    const int* __restrict__ positions,
    const int* __restrict__ slots,
    uint8_t* __restrict__ kv_cache,
    const float* __restrict__ cos_sin,
    int num_tokens, int head_dim, int rope_dim) {
    const int t = blockIdx.x;
    if (t >= num_tokens) return;
    const int nope_dim = head_dim - rope_dim;
    const int entry_bytes = head_dim + 4 + rope_dim * 2 + head_dim + 4;
    const int d0 = threadIdx.x * 2;

    const __nv_bfloat16* src = kv_in + (int64_t)t * head_dim;
    const int pos = __ldg(positions + t);
    const float* cs = cos_sin + (int64_t)pos * rope_dim;
    const int half = rope_dim / 2;

    // Assemble the roped 512 vector: [448 nope | rope(pe)].
    float v0 = 0.0f, v1 = 0.0f;
    if (d0 < head_dim) {
        if (d0 < nope_dim) {
            v0 = __bfloat162float(src[d0]);
            v1 = __bfloat162float(src[d0 + 1]);
        } else {
            // Pair (d0, d0+1) within the pe: pair index i = (d0-nope_dim)/2.
            const int i = (d0 - nope_dim) / 2;
            const float e = __bfloat162float(src[nope_dim + 2 * i]);
            const float o = __bfloat162float(src[nope_dim + 2 * i + 1]);
            const float c = __ldg(cs + i);
            const float s = __ldg(cs + half + i);
            v0 = e * c - o * s;
            v1 = e * s + o * c;
        }
    }
    __shared__ float s_warp[kThreads / 32];
    uint8_t* entry = kv_cache + (int64_t)__ldg(slots + t) * entry_bytes;
    v4_write_fp8_entry(entry, v0, v1, d0, head_dim, rope_dim, s_warp);
}

// ── State ring write ─────────────────────────────────────────────────────

__global__ void v4_state_ring_write_kernel(
    __nv_bfloat16* __restrict__ ring, const __nv_bfloat16* __restrict__ src,
    const int* __restrict__ positions, int ring_capacity, int state_dim,
    int num_tokens) {
    const int t = blockIdx.x;
    if (t >= num_tokens) return;
    const int row = __ldg(positions + t) % ring_capacity;
    const __nv_bfloat16* s = src + (int64_t)t * state_dim;
    __nv_bfloat16* d = ring + (int64_t)row * state_dim;
    for (int i = threadIdx.x; i < state_dim; i += blockDim.x) d[i] = s[i];
}

// ── Overlap / HCA compression + insert ───────────────────────────────────

struct V4CompressDev {
    const __nv_bfloat16* kv_state;
    const __nv_bfloat16* score_state;
    int ring_capacity;
    int state_dim;
    int overlap;
    int stride;
    const float* ape;
    const float* norm_w;
    const float* cos_sin;
    float rms_eps;
    int D;
    int rope_dim;
    int first_block;
    int num_blocks;
    const int* slots;
    int out_mode;   // 0 = fp8 entry, 1 = indexer paged, 2 = bf16 rows (TQ)
    uint8_t* kv_cache;
    uint8_t* idx_pages;
    int idx_page_tokens;
    int64_t idx_page_bytes;
    __nv_bfloat16* bf16_rows;       // out_mode 2: [num_blocks, D]
    __nv_bfloat16* bf16_rope_rows;  // out_mode 2: [num_blocks, rope_dim]
};

__global__ void __launch_bounds__(kThreads) v4_compress_insert_kernel(
    const V4CompressDev p) {
    const int bi = blockIdx.x;
    if (bi >= p.num_blocks) return;
    const int j = p.first_block + bi;
    const int W = p.overlap ? 2 * p.stride : p.stride;
    const int nope_dim = p.D - p.rope_dim;

    __shared__ float s_warp[kThreads / 32];
    __shared__ float s_pooled[512];  // D <= 512

    // Each thread owns channels c = threadIdx.x*2, +1 when D >= 2*kThreads,
    // else a strided loop; D is 512 or 128, both handled by the strided loop.
    // Window-entry addressing shared by both passes: entry w → global token t
    // (t < 0 excluded) + which state half to read.
    auto entry_at = [&](int w, int& t, int& half) {
        if (p.overlap) {
            if (w < p.stride) {
                t = (j - 1) * p.stride + w;
                half = 0;
            } else {
                t = j * p.stride + (w - p.stride);
                half = 1;
            }
        } else {
            t = j * p.stride + w;
            half = 0;
        }
    };
    for (int c = threadIdx.x; c < p.D; c += kThreads) {
        // Per-channel stable softmax over the window; two passes (max, then
        // exp/sum/pool) to avoid a per-thread window-sized local array.
        float m = -INFINITY;
        for (int w = 0; w < W; ++w) {
            int t, half;
            entry_at(w, t, half);
            if (t < 0) continue;
            const int row = t % p.ring_capacity;
            const int ch = half * p.D + c;
            const float sc =
                __bfloat162float(
                    p.score_state[(int64_t)row * p.state_dim + ch]) +
                __ldg(p.ape + (int64_t)(t % p.stride) * p.state_dim + ch);
            m = fmaxf(m, sc);
        }
        float sum = 0.0f;
        float pooled = 0.0f;
        for (int w = 0; w < W; ++w) {
            int t, half;
            entry_at(w, t, half);
            if (t < 0) continue;
            const int row = t % p.ring_capacity;
            const int ch = half * p.D + c;
            const float sc =
                __bfloat162float(
                    p.score_state[(int64_t)row * p.state_dim + ch]) +
                __ldg(p.ape + (int64_t)(t % p.stride) * p.state_dim + ch);
            const float e = expf(sc - m);
            sum += e;
            pooled += e * __bfloat162float(
                              p.kv_state[(int64_t)row * p.state_dim + ch]);
        }
        s_pooled[c] = (sum > 0.0f) ? pooled / sum : 0.0f;
    }
    __syncthreads();

    // Weighted RMS norm over D.
    float sq = 0.0f;
    for (int c = threadIdx.x; c < p.D; c += kThreads) {
        sq += s_pooled[c] * s_pooled[c];
    }
    const float sqrsum = block_reduce_sum(sq, s_warp);
    const float inv_rms = rsqrtf(sqrsum / (float)p.D + p.rms_eps);
    for (int c = threadIdx.x; c < p.D; c += kThreads) {
        s_pooled[c] = s_pooled[c] * inv_rms * __ldg(p.norm_w + c);
    }
    __syncthreads();

    // Rope the pe part at the block-endpoint position.
    const int pos = (j + 1) * p.stride - 1;
    const float* cs = p.cos_sin + (int64_t)pos * p.rope_dim;
    const int half_r = p.rope_dim / 2;
    for (int i = threadIdx.x; i < half_r; i += kThreads) {
        const float e = s_pooled[nope_dim + 2 * i];
        const float o = s_pooled[nope_dim + 2 * i + 1];
        const float c = __ldg(cs + i);
        const float s = __ldg(cs + half_r + i);
        s_pooled[nope_dim + 2 * i] = e * c - o * s;
        s_pooled[nope_dim + 2 * i + 1] = e * s + o * c;
    }
    __syncthreads();

    const int slot = __ldg(p.slots + bi);
    if (p.out_mode == 2) {
        // TQ codec staging (V4-5T): the roped compressed vector as BF16
        // rows + the duplicated roped tail — quantize/pack happens in
        // launch_v4_tq_entry_append (deps v4_tq_k_append).
        __nv_bfloat16* row = p.bf16_rows + (int64_t)bi * p.D;
        for (int c = threadIdx.x; c < p.D; c += kThreads)
            row[c] = __float2bfloat16_rn(s_pooled[c]);
        __nv_bfloat16* rr = p.bf16_rope_rows + (int64_t)bi * p.rope_dim;
        for (int i = threadIdx.x; i < p.rope_dim; i += kThreads)
            rr[i] = __float2bfloat16_rn(s_pooled[nope_dim + i]);
        return;
    }
    if (p.out_mode == 0) {
        const int entry_bytes = p.D + 4 + p.rope_dim * 2 + p.D + 4;
        uint8_t* entry = p.kv_cache + (int64_t)slot * entry_bytes;
        const int d0 = threadIdx.x * 2;
        float v0 = 0.0f, v1 = 0.0f;
        if (d0 < p.D) {
            v0 = s_pooled[d0];
            v1 = s_pooled[d0 + 1];
        }
        v4_write_fp8_entry(entry, v0, v1, d0, p.D, p.rope_dim, s_warp);
    } else {
        // Indexer paged: [page_tokens × D FP8 | page_tokens × f32 scales].
        const int page = slot / p.idx_page_tokens;
        const int row = slot % p.idx_page_tokens;
        uint8_t* pg = p.idx_pages + (int64_t)page * p.idx_page_bytes;
        auto* k8 = reinterpret_cast<__nv_fp8_e4m3*>(pg + (int64_t)row * p.D);
        auto* sc = reinterpret_cast<float*>(
            pg + (int64_t)p.idx_page_tokens * p.D + (int64_t)row * 4);
        float amax_local = 0.0f;
        for (int c = threadIdx.x; c < p.D; c += kThreads)
            amax_local = fmaxf(amax_local, fabsf(s_pooled[c]));
        const float amax = block_reduce_max(amax_local, s_warp);
        const float scale = amax / kFp8Max;
        const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        for (int c = threadIdx.x; c < p.D; c += kThreads) {
            const float q = fmaxf(-kFp8Max,
                                  fminf(kFp8Max, s_pooled[c] * inv_scale));
            k8[c] = __nv_fp8_e4m3(q);
        }
        if (threadIdx.x == 0) *sc = scale;
    }
}

// ── Entry gather ─────────────────────────────────────────────────────────

__global__ void v4_entry_gather_kernel(
    uint8_t* __restrict__ dst, const uint8_t* __restrict__ src_cache,
    const int* __restrict__ slots, int count, int entry_bytes,
    int dst_row_offset) {
    const int i = blockIdx.x;
    if (i >= count) return;
    const uint8_t* s = src_cache + (int64_t)__ldg(slots + i) * entry_bytes;
    uint8_t* d = dst + (int64_t)(dst_row_offset + i) * entry_bytes;
    // 1160 B per entry; copy as 4-byte words (1160 % 4 == 0).
    const int words = entry_bytes / 4;
    const auto* s4 = reinterpret_cast<const uint32_t*>(s);
    auto* d4 = reinterpret_cast<uint32_t*>(d);
    for (int w = threadIdx.x; w < words; w += blockDim.x) d4[w] = s4[w];
}

// ── Logical→physical slot translation (V4-7b) ───────────────────────────

__global__ void v4_slot_translate_kernel(
    int* __restrict__ out, const int* __restrict__ logical_in,
    const int* __restrict__ page_table, int entries_per_page, int n_valid,
    int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int in = logical_in ? __ldg(logical_in + i) : i;
    if (in < 0 || in >= n_valid) {
        out[i] = -1;
        return;
    }
    const int page = __ldg(page_table + in / entries_per_page);
    out[i] = page * entries_per_page + in % entries_per_page;
}

// ── Batched-prefill per-row logical index build (superchunk port) ────────
// out[i, j] (j < topk) =
//   row_num_blocks[i] <= topk : j < row_num_blocks[i] ? j : -1   (IOTA arm —
//     deterministic all-visible selection, the ticket-J rule per row)
//   else                      : lightning_in[i, j]               (top-k arm;
//     -1 padding passes through; in-place merge legal — same-slot rewrite)
__global__ void v4_prefill_indices_kernel(
    int* __restrict__ out, const int* __restrict__ lightning_in,
    const int* __restrict__ row_num_blocks, int rows, int topk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * topk) return;
    const int row = i / topk;
    const int j = i % topk;
    const int nb = __ldg(row_num_blocks + row);
    if (nb <= topk || !lightning_in) {
        out[i] = (j < nb) ? j : -1;
    } else {
        out[i] = __ldg(lightning_in + i);
    }
}

// ── Batched-prefill per-row SWA block table (superchunk port) ────────────
// Staging layout: rows [0, w_pref) hold the ring prefix (positions
// [p0-w_pref, p0)), rows [w_pref, w_pref+R) the chunk's own entries
// (position p0+i at row w_pref+i). With swa_page_block_size == 1 the block
// table IS a per-token index list: row i's window (ascending chronological)
//   bt[i, j] = i - swa_len[i] + 1 + j + w_pref   for j < swa_len[i]
// (== staging row of position pos_i - swa_len_i + 1 + j), -1 beyond.
__global__ void v4_prefill_swa_bt_kernel(
    int* __restrict__ bt, const int* __restrict__ swa_len, int rows,
    int window, int w_pref) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * window) return;
    const int row = i / window;
    const int j = i % window;
    const int len = __ldg(swa_len + row);
    bt[i] = (j < len) ? (row - len + 1 + j + w_pref) : -1;
}

// ── Attention sinks post-epilogue ────────────────────────────────────────

__global__ void v4_attn_sinks_kernel(
    __nv_bfloat16* __restrict__ out, float* __restrict__ lse,
    const float* __restrict__ sinks, int head_offset, int num_tokens, int h_q,
    int d_v) {
    const int row = blockIdx.x;  // token*h_q + head
    if (row >= num_tokens * h_q) return;
    const int h = row % h_q;
    const float l = lse[row];
    const float s = __ldg(sinks + head_offset + h);
    // factor = L/(L+e^s) = sigmoid(lse − s); logaddexp for the new lse.
    const float factor = 1.0f / (1.0f + expf(s - l));
    __nv_bfloat16* o = out + (int64_t)row * d_v;
    for (int i = threadIdx.x; i < d_v; i += blockDim.x) {
        o[i] = __float2bfloat16_rn(__bfloat162float(o[i]) * factor);
    }
    if (threadIdx.x == 0) {
        const float mx = fmaxf(l, s);
        const float mn = fminf(l, s);
        lse[row] = mx + log1pf(expf(mn - mx));
    }
}

// ── Output inverse rope ──────────────────────────────────────────────────

__global__ void v4_out_inverse_rope_kernel(
    __nv_bfloat16* __restrict__ out, const int* __restrict__ positions,
    const float* __restrict__ cos_sin, int num_tokens, int h_q, int head_dim,
    int rope_dim) {
    const int row = blockIdx.x;  // token*h_q + head
    if (row >= num_tokens * h_q) return;
    const int token = row / h_q;
    const int nope_dim = head_dim - rope_dim;
    const int half = rope_dim / 2;
    const int pos = __ldg(positions + token);
    const float* cs = cos_sin + (int64_t)pos * rope_dim;
    __nv_bfloat16* pe = out + (int64_t)row * head_dim + nope_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float e = __bfloat162float(pe[2 * i]);
        const float o = __bfloat162float(pe[2 * i + 1]);
        const float c = __ldg(cs + i);
        const float s = __ldg(cs + half + i);
        // Inverse rotation (transpose): e' = e·c + o·s; o' = −e·s + o·c.
        pe[2 * i] = __float2bfloat16_rn(e * c + o * s);
        pe[2 * i + 1] = __float2bfloat16_rn(-e * s + o * c);
    }
}

}  // namespace

void launch_v4_q_prep(void* q_nope_out, void* q_rope_out, const void* q_in,
                      const int* positions, const void* cos_sin, float rms_eps,
                      int num_tokens, int h_q, int head_dim, int rope_dim,
                      void* stream) {
    if (num_tokens <= 0) return;
    v4_q_prep_kernel<<<num_tokens * h_q, kThreads, 0,
                       static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(q_nope_out),
        static_cast<__nv_bfloat16*>(q_rope_out),
        static_cast<const __nv_bfloat16*>(q_in), positions,
        static_cast<const float*>(cos_sin), rms_eps, num_tokens, h_q, head_dim,
        rope_dim);
    check_launch("v4_q_prep");
}

void launch_v4_raw_kv_append(const void* kv_in, const int* positions,
                             const int* slots, void* kv_cache,
                             const void* cos_sin, int num_tokens, int head_dim,
                             int rope_dim, void* stream) {
    if (num_tokens <= 0) return;
    v4_raw_kv_append_kernel<<<num_tokens, kThreads, 0,
                              static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(kv_in), positions, slots,
        static_cast<uint8_t*>(kv_cache), static_cast<const float*>(cos_sin),
        num_tokens, head_dim, rope_dim);
    check_launch("v4_raw_kv_append");
}

void launch_v4_state_ring_write(void* ring, const void* src,
                                const int* positions, int ring_capacity,
                                int state_dim, int num_tokens, void* stream) {
    if (num_tokens <= 0) return;
    v4_state_ring_write_kernel<<<num_tokens, 256, 0,
                                 static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(ring),
        static_cast<const __nv_bfloat16*>(src), positions, ring_capacity,
        state_dim, num_tokens);
    check_launch("v4_state_ring_write");
}

void launch_v4_compress_insert(const V4CompressArgs& args, void* stream) {
    if (args.num_blocks <= 0) return;
    if (args.D > 512 || (args.overlap ? 2 * args.stride : args.stride) > 256) {
        throw std::runtime_error("v4_compress_insert: unsupported dims");
    }
    V4CompressDev p{};
    p.kv_state = static_cast<const __nv_bfloat16*>(args.kv_state);
    p.score_state = static_cast<const __nv_bfloat16*>(args.score_state);
    p.ring_capacity = args.ring_capacity;
    p.state_dim = args.state_dim;
    p.overlap = args.overlap ? 1 : 0;
    p.stride = args.stride;
    p.ape = static_cast<const float*>(args.ape);
    p.norm_w = static_cast<const float*>(args.norm_w);
    p.cos_sin = static_cast<const float*>(args.cos_sin);
    p.rms_eps = args.rms_eps;
    p.D = args.D;
    p.rope_dim = args.rope_dim;
    p.first_block = args.first_block;
    p.num_blocks = args.num_blocks;
    p.slots = args.slots;
    p.out_mode = args.out_mode == V4CompressArgs::Out::kFp8Entry   ? 0
               : args.out_mode == V4CompressArgs::Out::kIndexerPaged ? 1
                                                                     : 2;
    p.kv_cache = static_cast<uint8_t*>(args.kv_cache);
    p.idx_pages = static_cast<uint8_t*>(args.idx_pages);
    p.idx_page_tokens = args.idx_page_tokens;
    p.idx_page_bytes = args.idx_page_bytes;
    p.bf16_rows = static_cast<__nv_bfloat16*>(args.bf16_rows);
    p.bf16_rope_rows = static_cast<__nv_bfloat16*>(args.bf16_rope_rows);
    v4_compress_insert_kernel<<<args.num_blocks, kThreads, 0,
                                static_cast<cudaStream_t>(stream)>>>(p);
    check_launch("v4_compress_insert");
}

void launch_v4_entry_gather(void* dst, const void* src_cache,
                            const int* slots, int count, int entry_bytes,
                            int dst_row_offset, void* stream) {
    if (count <= 0) return;
    if (entry_bytes % 4 != 0) {
        throw std::runtime_error("v4_entry_gather: entry_bytes % 4 != 0");
    }
    v4_entry_gather_kernel<<<count, 128, 0,
                             static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(dst), static_cast<const uint8_t*>(src_cache),
        slots, count, entry_bytes, dst_row_offset);
    check_launch("v4_entry_gather");
}

void launch_v4_slot_translate(int* out, const int* logical_in,
                              const int* page_table, int entries_per_page,
                              int n_valid, int count, void* stream) {
    if (count <= 0) return;
    if (entries_per_page <= 0) {
        throw std::runtime_error("v4_slot_translate: entries_per_page <= 0");
    }
    const int threads = 128;
    const int blocks = (count + threads - 1) / threads;
    v4_slot_translate_kernel<<<blocks, threads, 0,
                               static_cast<cudaStream_t>(stream)>>>(
        out, logical_in, page_table, entries_per_page, n_valid, count);
    check_launch("v4_slot_translate");
}

void launch_v4_prefill_indices(int* out, const int* lightning_in,
                               const int* row_num_blocks, int rows, int topk,
                               void* stream) {
    if (rows <= 0 || topk <= 0) return;
    const int threads = 128;
    const int total = rows * topk;
    const int blocks = (total + threads - 1) / threads;
    v4_prefill_indices_kernel<<<blocks, threads, 0,
                                static_cast<cudaStream_t>(stream)>>>(
        out, lightning_in, row_num_blocks, rows, topk);
    check_launch("v4_prefill_indices");
}

void launch_v4_prefill_swa_bt(int* bt, const int* swa_len, int rows,
                              int window, int w_pref, void* stream) {
    if (rows <= 0 || window <= 0) return;
    const int threads = 128;
    const int total = rows * window;
    const int blocks = (total + threads - 1) / threads;
    v4_prefill_swa_bt_kernel<<<blocks, threads, 0,
                               static_cast<cudaStream_t>(stream)>>>(
        bt, swa_len, rows, window, w_pref);
    check_launch("v4_prefill_swa_bt");
}

void launch_v4_attn_sinks(void* out, void* lse, const void* sinks,
                          int head_offset, int num_tokens, int h_q, int d_v,
                          void* stream) {
    if (num_tokens <= 0) return;
    v4_attn_sinks_kernel<<<num_tokens * h_q, 256, 0,
                           static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(out), static_cast<float*>(lse),
        static_cast<const float*>(sinks), head_offset, num_tokens, h_q, d_v);
    check_launch("v4_attn_sinks");
}

void launch_v4_out_inverse_rope(void* out, const int* positions,
                                const void* cos_sin, int num_tokens, int h_q,
                                int head_dim, int rope_dim, void* stream) {
    if (num_tokens <= 0) return;
    v4_out_inverse_rope_kernel<<<num_tokens * h_q, 64, 0,
                                 static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(out), positions,
        static_cast<const float*>(cos_sin), num_tokens, h_q, head_dim,
        rope_dim);
    check_launch("v4_out_inverse_rope");
}


// ── Two-way LSE-weighted partial merge (V4-5T; see v4_prep.h) ────────────

__global__ void v4_lse_merge2_kernel(
    __nv_bfloat16* __restrict__ out_a, float* __restrict__ lse_a,
    const __nv_bfloat16* __restrict__ out_b,
    const float* __restrict__ lse_b, int rows, int heads, int head_dim) {
    const int rh = blockIdx.x;
    if (rh >= rows * heads) return;
    const float la = __ldg(lse_a + rh);
    const float lb = __ldg(lse_b + rh);
    // A side with no keys carries lse <= -1e30 (the kernels' empty
    // sentinel) — weight 0 by construction of the stable merge below.
    const float m = fmaxf(la, lb);
    if (!(m > -1e30f)) {
        // Both empty: leave out_a as-is (never consumed downstream at
        // zero visibility), keep the sentinel lse.
        return;
    }
    const float wa = __expf(la - m);
    const float wb = __expf(lb - m);
    const float inv = 1.0f / (wa + wb);
    __nv_bfloat16* oa = out_a + (int64_t)rh * head_dim;
    const __nv_bfloat16* ob = out_b + (int64_t)rh * head_dim;
    for (int c = threadIdx.x; c < head_dim; c += blockDim.x) {
        const float v = (wa * __bfloat162float(oa[c]) +
                         wb * __bfloat162float(ob[c])) * inv;
        oa[c] = __float2bfloat16_rn(v);
    }
    if (threadIdx.x == 0) lse_a[rh] = m + logf(wa + wb);
}

void launch_v4_lse_merge2(const V4LseMerge2Args& args, void* stream) {
    if (args.rows <= 0 || args.heads <= 0) return;
    v4_lse_merge2_kernel<<<args.rows * args.heads, 256, 0,
                           static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(args.out_a), args.lse_a,
        static_cast<const __nv_bfloat16*>(args.out_b), args.lse_b,
        args.rows, args.heads, args.head_dim);
    check_launch("v4_lse_merge2");
}

}  // namespace layerstorm::compute
