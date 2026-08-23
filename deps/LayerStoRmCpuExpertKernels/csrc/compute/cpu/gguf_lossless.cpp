// Bit-compatible CPU K-quant × Q8_1 grouped GEMM — see gguf_lossless.h.
//
// Reproduces the GPU GGUF mmvq numerics EXACTLY, both the arithmetic AND the
// FP32 accumulation ORDER (deps/LayerStoRmGemmKernels; ggml/llama.cpp
// lineage, MIT License, Copyright (c) 2023-2026 The ggml authors — see
// THIRD_PARTY_NOTICES.md):
//   * Q8_1 activation quant  (formats/q8_1_format.h, sm120/gemm/gguf/gguf_mmvq.cu):
//       d = amax/127 (full float), stored FP16; qs[i] = round_nearest(x[i] / d).
//       NOTE the DIVISION x/d — the GPU does __float2int_rn(xi / d), NOT a
//       reciprocal-multiply x*(1/d); the two round differently ~1/1e4 elements
//       and that sub-ULP mismatch was one of the two token-22 drift sources.
//   * K-quant block layout + get_scale_min_k4 + per-4-value-group int dot
//     (gguf_int_policy.h {Q4K,Q5K,Q6K}::group, gguf_mmvq_impl.cuh accumulate()).
//   * The PRODUCTION decode path is the WPB=8 grouped mmvq compact kernel
//     (gguf_grouped_int.cu gguf_mmvq_grouped_compact*_kernel): ONE warp (32 lanes)
//     per output channel. Lane l accumulates groups g = l, l+32 within EACH
//     superblock, superblocks ascending; then a butterfly __shfl_xor reduction
//     (o = 16,8,4,2,1) across the 32 lanes; then __float2bfloat16_rn. This kernel
//     REPLICATES that exact 32-lane tree in FP32 — the other token-22 drift source
//     was the old code's flat gg=0..N sequential sum (wrong reduction ORDER).
//
// The per-group integer dot (dotQ = Σ Q·q8, sumqx = Σ q8) is quant-agnostic and
// order-free, so it is computed with AVX512-VNNI (_mm512_dpbusd_epi32) for speed
// while the (order-sensitive) FP32 tree above is replayed bit-for-bit. On a build
// without VNNI a scalar integer dot is used (same reduction → same result).
//
// CPU-only, no CUDA (INV-GPU-1).

#include "compute/cpu/gguf_lossless.h"

#include "compute/cpu/numa_thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
#include <immintrin.h>
#define LS_KQ_HAVE_VNNI 1
#else
#define LS_KQ_HAVE_VNNI 0
#endif

namespace layerstorm::compute::cpu {

namespace {

inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}
inline uint16_t f32_to_bf16(float f) {
    uint32_t bits; std::memcpy(&bits, &f, sizeof(bits));
    bits += 0x7FFF + ((bits >> 16) & 1);  // round-to-nearest-even
    return static_cast<uint16_t>(bits >> 16);
}
inline float f16_to_f32(uint16_t h) {
    return static_cast<float>(*reinterpret_cast<const _Float16*>(&h));
}
// FP16(x) as a float, round-to-nearest-even — matches CUDA __float2half_rn.
inline float round_to_f16(float x) {
    _Float16 h = static_cast<_Float16>(x);
    return static_cast<float>(h);
}

constexpr int kQK_K = 256;         // K-quant super-block values
constexpr int kQ4K_BYTES = 144;    // block_q4_K size
constexpr int kQ5K_BYTES = 176;    // block_q5_K size
constexpr int kQ6K_BYTES = 210;    // block_q6_K size
constexpr int kQK8 = 32;           // Q8_1 block values
constexpr int kGPS = kQK_K / 4;    // 64 four-value groups per super-block

inline size_t kq_block_bytes(KQuantLossless t) {
    switch (t) {
        case KQuantLossless::Q4_K: return kQ4K_BYTES;
        case KQuantLossless::Q5_K: return kQ5K_BYTES;
        case KQuantLossless::Q6_K: return kQ6K_BYTES;
    }
    return kQ4K_BYTES;
}

// get_scale_min_k4 — verbatim ggml (ggml-cuda/convert.cu), sub-block j in [0,7].
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

// ── Per-super-block decode to raw UNSIGNED weight bytes + per-group A/B ───────
// Fills, for one 256-value super-block:
//   wbytes[4*g + i]  = RAW UNSIGNED quant for logical position (g,i), i=0..3,
//                      laid out to pair with activation byte (sb*256 + 4*g + i);
//                      Q4_K nibble 0..15, Q5_K 5-bit 0..31, Q6_K 6-bit 0..63.
//   A[g], B[g]       = the FP group scale / min term used by the dot
//                      (Q4K/Q5K: A=d·sc, B=−dmin·m; Q6K: A=d·sc, B=0, centring is
//                       done on the integer dot, see below).
// Mirrors gguf_int_policy.h {Q4K,Q5K,Q6K}::group but produces all 64 groups.
// The scale/min (A,B) span 8 consecutive groups per sub-block (sub = g>>3), so the
// FP16→FP32 super-scales and the 8 get_scale_min_k4 pairs are hoisted out of the
// per-group loop (was recomputed 64× — the decode bottleneck). The nibble unpack
// stays per-group (cheap integer shifts).
// A[g]/B[g] (super-scales × 6-bit sub-scales) span 8 groups each (sub = g>>3);
// hoisted here for both Q4_K and Q5_K.
inline void decode_kq_scalemin(const uint8_t* scales, float d, float dmin,
                               float* A, float* B) {
    for (int s = 0; s < 8; ++s) {
        uint8_t sc, m; get_scale_min_k4(s, scales, sc, m);
        const float a = d * sc, b = -dmin * m;
        for (int g = s * 8; g < s * 8 + 8; ++g) { A[g] = a; B[g] = b; }
    }
}

// The weight-byte layout is per 32-value pair chunk (pair p = qs[32p..32p+31]):
//   wbytes[64p + 0..31]  = lo nibble of qs[32p + 0..31]   (groups 16p..16p+7)
//   wbytes[64p + 32..63] = hi nibble of qs[32p + 0..31]   (groups 16p+8..16p+15)
// so the whole super-block unpacks as 4 pairs × {mask, shift} — vectorized below.
inline void decode_sb_q4k(const uint8_t* blk, uint8_t* wbytes, float* A, float* B) {
    uint16_t dh, dmh; std::memcpy(&dh, blk, 2); std::memcpy(&dmh, blk + 2, 2);
    decode_kq_scalemin(blk + 4, f16_to_f32(dh), f16_to_f32(dmh), A, B);
    const uint8_t* qs = blk + 16;
#if LS_KQ_HAVE_VNNI
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    for (int p = 0; p < 4; ++p) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32 * p));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(wbytes + 64 * p),
                            _mm256_and_si256(v, m0f));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(wbytes + 64 * p + 32),
                            _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f));
    }
#else
    for (int p = 0; p < 4; ++p) for (int j = 0; j < 32; ++j) {
        wbytes[64 * p + j]      = qs[32 * p + j] & 0x0F;
        wbytes[64 * p + 32 + j] = qs[32 * p + j] >> 4;
    }
#endif
}
inline void decode_sb_q5k(const uint8_t* blk, uint8_t* wbytes, float* A, float* B) {
    uint16_t dh, dmh; std::memcpy(&dh, blk, 2); std::memcpy(&dmh, blk + 2, 2);
    decode_kq_scalemin(blk + 4, f16_to_f32(dh), f16_to_f32(dmh), A, B);
    const uint8_t* qh = blk + 16; const uint8_t* qs = blk + 48;
    // Same pair layout as Q4_K plus the 5th bit from qh[0..31]: for pair p, the lo
    // half uses qh bit (2p), the hi half bit (2p+1); shifted to bit 4 and OR'd.
#if LS_KQ_HAVE_VNNI
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i m01 = _mm256_set1_epi8(0x01);
    const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
    for (int p = 0; p < 4; ++p) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32 * p));
        // lo/hi bit of qh per byte, moved to bit position 4 (value 16).
        const __m256i blo = _mm256_slli_epi16(
            _mm256_and_si256(_mm256_srli_epi16(qhv, 2 * p), m01), 4);
        const __m256i bhi = _mm256_slli_epi16(
            _mm256_and_si256(_mm256_srli_epi16(qhv, 2 * p + 1), m01), 4);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(wbytes + 64 * p),
                            _mm256_or_si256(_mm256_and_si256(v, m0f), blo));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(wbytes + 64 * p + 32),
                            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v, 4), m0f), bhi));
    }
#else
    for (int p = 0; p < 4; ++p) for (int j = 0; j < 32; ++j) {
        const int blo = ((qh[j] >> (2 * p)) & 1) << 4;
        const int bhi = ((qh[j] >> (2 * p + 1)) & 1) << 4;
        wbytes[64 * p + j]      = (qs[32 * p + j] & 0x0F) | blo;
        wbytes[64 * p + 32 + j] = (qs[32 * p + j] >> 4)   | bhi;
    }
#endif
}
// Q6_K super-block = 2 halves (ip) × 4 quads of 32 raw 6-bit values. Per half ip
// the 32-value quads are (ql = low 4 bits, qh[32ip..] = high 2 bits, reused):
//   quad0 wbytes[128ip+  0.. 31] = (ql[64ip+ 0..31] & 0xF) | ((qh>>0 & 3)<<4)
//   quad1 wbytes[128ip+ 32.. 63] = (ql[64ip+32..63] & 0xF) | ((qh>>2 & 3)<<4)
//   quad2 wbytes[128ip+ 64.. 95] = (ql[64ip+ 0..31] >> 4)  | ((qh>>4 & 3)<<4)
//   quad3 wbytes[128ip+ 96..127] = (ql[64ip+32..63] >> 4)  | ((qh>>6 & 3)<<4)
// A[g] = d·scales[8ip + (groups 0..3 of quad → +0, 4..7 → +1) + 2quad]; B[g] = 0
// (centring q-32 is folded into the integer dot). RAW 0..63 bytes (unsigned).
inline void decode_sb_q6k(const uint8_t* blk, uint8_t* wbytes, float* A, float* B) {
    const uint8_t* ql = blk; const uint8_t* qh = blk + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);
    uint16_t dh; std::memcpy(&dh, blk + 208, 2); const float d = f16_to_f32(dh);
    for (int ip = 0; ip < 2; ++ip) for (int quad = 0; quad < 4; ++quad) {
        const float a0 = d * static_cast<float>(scales[8 * ip + 2 * quad]);
        const float a1 = d * static_cast<float>(scales[8 * ip + 2 * quad + 1]);
        const int gbase = ip * 32 + quad * 8;         // first group of this quad
        for (int k = 0; k < 4; ++k) A[gbase + k] = a0;
        for (int k = 4; k < 8; ++k) A[gbase + k] = a1;
        for (int k = 0; k < 8; ++k) B[gbase + k] = 0.0f;
    }
#if LS_KQ_HAVE_VNNI
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i m03 = _mm256_set1_epi8(0x03);
    for (int ip = 0; ip < 2; ++ip) {
        const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh + 32 * ip));
        const __m256i qlA = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 64 * ip));
        const __m256i qlB = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 64 * ip + 32));
        auto h = [&](int sh) {                         // (qh >> sh & 3) << 4, per byte
            return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, sh), m03), 4);
        };
        uint8_t* w = wbytes + 128 * ip;
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(w + 0),
                            _mm256_or_si256(_mm256_and_si256(qlA, m0f), h(0)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(w + 32),
                            _mm256_or_si256(_mm256_and_si256(qlB, m0f), h(2)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(w + 64),
                            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlA, 4), m0f), h(4)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(w + 96),
                            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlB, 4), m0f), h(6)));
    }
#else
    for (int ip = 0; ip < 2; ++ip) for (int j = 0; j < 32; ++j) {
        const uint8_t qhj = qh[32 * ip + j];
        const uint8_t la = ql[64 * ip + j], lb = ql[64 * ip + 32 + j];
        uint8_t* w = wbytes + 128 * ip;
        w[0  + j] = (la & 0x0F) | (((qhj >> 0) & 3) << 4);
        w[32 + j] = (lb & 0x0F) | (((qhj >> 2) & 3) << 4);
        w[64 + j] = (la >> 4)   | (((qhj >> 4) & 3) << 4);
        w[96 + j] = (lb >> 4)   | (((qhj >> 6) & 3) << 4);
    }
#endif
}

inline void decode_superblock(KQuantLossless type, const uint8_t* blk,
                              uint8_t* wbytes, float* A, float* B) {
    switch (type) {
        case KQuantLossless::Q4_K: decode_sb_q4k(blk, wbytes, A, B); break;
        case KQuantLossless::Q5_K: decode_sb_q5k(blk, wbytes, A, B); break;
        case KQuantLossless::Q6_K: decode_sb_q6k(blk, wbytes, A, B); break;
    }
}

// ── Per-group integer dots (dotQ = Σ Q·q8, sumqx = Σ q8) over ONE super-block ──
// wbytes[256] raw unsigned weight bytes, aq_sb[256] the super-block's activation
// int8 (contiguous K-order). Writes dotQ_g[64], sumqx_g[64]. Integer, order-free
// -> the scalar and VNNI variants are bit-identical by construction.
inline void group_dots_scalar(const uint8_t* wbytes, const int8_t* aq_sb,
                              int32_t* dotQ_g, int32_t* sumqx_g) {
    for (int g = 0; g < kGPS; ++g) {
        const uint8_t* w = wbytes + 4 * g; const int8_t* a = aq_sb + 4 * g;
        int dq = 0, sq = 0;
        for (int i = 0; i < 4; ++i) { dq += static_cast<int>(w[i]) * static_cast<int>(a[i]); sq += a[i]; }
        dotQ_g[g] = dq; sumqx_g[g] = sq;
    }
}

#if LS_KQ_HAVE_VNNI
// AVX512-VNNI: 64 raw weight bytes (u8) · 64 activation bytes (s8) -> 16 group
// dots via _mm512_dpbusd_epi32; four passes cover the 64 groups of a super-block.
// sumqx via dpbusd against an all-ones u8 vector. Bit-identical INTEGER result to
// group_dots_scalar (the accumulation is per-32-bit-lane over 4 bytes).
inline void group_dots_vnni(const uint8_t* wbytes, const int8_t* aq_sb,
                            int32_t* dotQ_g, int32_t* sumqx_g) {
    const __m512i ones = _mm512_set1_epi8(1);
    for (int c = 0; c < 4; ++c) {                 // 4 chunks of 64 values = 16 groups
        const __m512i w = _mm512_loadu_si512(wbytes + c * 64);
        const __m512i a = _mm512_loadu_si512(aq_sb + c * 64);
        const __m512i dq = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, a);
        const __m512i sq = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, a);
        _mm512_storeu_si512(dotQ_g + c * 16, dq);
        _mm512_storeu_si512(sumqx_g + c * 16, sq);
    }
}
#endif

// ── GPU 32-lane warp-tree FP32 reduction (bit-exact vs the compact mmvq) ──────
// Given, for ONE (output channel n, token m), the per-group integers dotQ_g[gg]
// (already CENTRED for Q6_K), sumqx_g[gg], the per-group FP scale/min A/B (n only)
// and the activation block scales dxrow[b32] (m only), replays:
//   part[l] = Σ over (sb, gi) in order of  dx·(A·dotQ + B·sumqx),  gg = sb*64+gi*32+l
//   butterfly:  for o=16,8,4,2,1:  part[l] += part[l^o]
//   out = bf16_rne(part[0])
// EXACTLY the gguf_mmvq_grouped_compact_kernel per-row order.
inline uint16_t reduce_gpuorder(const int32_t* dotQ_g, const int32_t* sumqx_g,
                                const float* A, const float* B, const float* dxrow,
                                int nblk) {
    float part[32];
    for (int l = 0; l < 32; ++l) part[l] = 0.0f;
    for (int sb = 0; sb < nblk; ++sb) {
        const int gg0 = sb * kGPS;             // first global group of this super-block
        for (int gi = 0; gi < 2; ++gi) {       // groups-per-lane = 64/32 = 2
            const int base = gg0 + gi * 32;    // 32 consecutive groups -> the 32 lanes
            for (int l = 0; l < 32; ++l) {
                const int gg = base + l;
                const float dx = dxrow[gg >> 3];
                // Match nvcc's --fmad=true contraction of the GPU expression
                //   acc += dx * (A*dotQ + B*sumqx):
                //   inner = fma(A, dotQ, B*sumqx);  acc = fma(dx, inner, acc).
                const float inner = std::fma(A[gg], static_cast<float>(dotQ_g[gg]),
                                             B[gg] * static_cast<float>(sumqx_g[gg]));
                part[l] = std::fma(dx, inner, part[l]);
            }
        }
    }
    // Butterfly (o = 16,8,4,2,1), simultaneous update — matches __shfl_xor_sync.
    for (int o = 16; o > 0; o >>= 1) {
        float nx[32];
        for (int l = 0; l < 32; ++l) nx[l] = part[l] + part[l ^ o];
        std::memcpy(part, nx, sizeof(part));
    }
    return f32_to_bf16(part[0]);
}

#if LS_KQ_HAVE_VNNI
// Vectorized variant of reduce_gpuorder: the 32-lane accumulation is a per-lane
// running FP32 sum (lanes independent), so two __m512 (lanes 0..15 / 16..31)
// carry the partials and each (sb,gi) 32-group chunk is added as a vector.
// Per-lane FP ops are IEEE-identical to the scalar loop -> BIT-IDENTICAL result.
// The butterfly is done scalar (32*5 ops) on the materialised partials.
inline uint16_t reduce_gpuorder_vec(const int32_t* dotQ_g, const int32_t* sumqx_g,
                                    const float* A, const float* B, const float* dxrow,
                                    int nblk) {
    __m512 plo = _mm512_setzero_ps(), phi = _mm512_setzero_ps();
    for (int sb = 0; sb < nblk; ++sb) {
        const int gg0 = sb * kGPS;
        const int b8 = gg0 >> 3;               // dxrow base for this super-block (8 b32 each)
        for (int gi = 0; gi < 2; ++gi) {
            const int base = gg0 + gi * 32;
            const int db = b8 + gi * 4;         // 4 consecutive b32 span these 32 groups
            // dx per lane: lanes [0..7]->dxrow[db], [8..15]->db+1, [16..23]->db+2, [24..31]->db+3.
            const __m512 dxlo = _mm512_mask_blend_ps(
                0xFF00, _mm512_set1_ps(dxrow[db]), _mm512_set1_ps(dxrow[db + 1]));
            const __m512 dxhi = _mm512_mask_blend_ps(
                0xFF00, _mm512_set1_ps(dxrow[db + 2]), _mm512_set1_ps(dxrow[db + 3]));
            const __m512 dqlo = _mm512_cvtepi32_ps(_mm512_loadu_si512(dotQ_g + base));
            const __m512 dqhi = _mm512_cvtepi32_ps(_mm512_loadu_si512(dotQ_g + base + 16));
            const __m512 sqlo = _mm512_cvtepi32_ps(_mm512_loadu_si512(sumqx_g + base));
            const __m512 sqhi = _mm512_cvtepi32_ps(_mm512_loadu_si512(sumqx_g + base + 16));
            const __m512 Alo = _mm512_loadu_ps(A + base), Ahi = _mm512_loadu_ps(A + base + 16);
            const __m512 Blo = _mm512_loadu_ps(B + base), Bhi = _mm512_loadu_ps(B + base + 16);
            // Match nvcc's contraction of  acc += dx*(A*dotQ + B*sumqx):
            //   inner = fma(A, dotQ, B*sumqx);  acc = fma(dx, inner, acc).
            // Bit-for-bit identical to reduce_gpuorder's std::fma form.
            const __m512 innerlo = _mm512_fmadd_ps(Alo, dqlo, _mm512_mul_ps(Blo, sqlo));
            const __m512 innerhi = _mm512_fmadd_ps(Ahi, dqhi, _mm512_mul_ps(Bhi, sqhi));
            plo = _mm512_fmadd_ps(dxlo, innerlo, plo);
            phi = _mm512_fmadd_ps(dxhi, innerhi, phi);
        }
    }
    alignas(64) float part[32];
    _mm512_store_ps(part, plo);
    _mm512_store_ps(part + 16, phi);
    for (int o = 16; o > 0; o >>= 1) {
        float nx[32];
        for (int l = 0; l < 32; ++l) nx[l] = part[l] + part[l ^ o];
        std::memcpy(part, nx, sizeof(part));
    }
    return f32_to_bf16(part[0]);
}
#endif

// ── Per output-row-slice GEMM (VNNI or scalar) ───────────────────────────────
// use_vnni selects the vectorized integer dot + reduction (production) vs the
// scalar reference (unit-test parity). `act_q` [total,K] int8, `act_d` [total,K/32]
// FP16-rounded block scales, `sumqx_all` [total, total_groups] precomputed per row.
void gemm_rows(KQuantLossless type, bool use_vnni, uint16_t* D,
               const void* const* B_ptrs, const float* act_d, const int8_t* act_q,
               const int32_t* sumqx_all, const int32_t* expert_offsets,
               int num_experts, int N, int K, int n0, int n1) {
    const int nblk = K / kQK_K;
    const int total_groups = nblk * kGPS;
    const size_t wrow_bytes = static_cast<size_t>(nblk) * kq_block_bytes(type);
    const size_t nb8 = static_cast<size_t>(K / kQK8);
    const bool is_q6k = (type == KQuantLossless::Q6_K);

    std::vector<uint8_t> wbytes(static_cast<size_t>(nblk) * kQK_K);
    std::vector<float>   A(static_cast<size_t>(total_groups));
    std::vector<float>   B(static_cast<size_t>(total_groups));
    std::vector<int32_t> dotQ(static_cast<size_t>(total_groups));

    for (int e = 0; e < num_experts; ++e) {
        const int m0 = expert_offsets[e], m1 = expert_offsets[e + 1];
        if (m1 <= m0) continue;
        const auto* W = static_cast<const uint8_t*>(B_ptrs[e]);
        for (int n = n0; n < n1; ++n) {
            const uint8_t* wrow = W + static_cast<size_t>(n) * wrow_bytes;
            // Decode the whole weight row ONCE (reused across this expert's tokens).
            for (int sb = 0; sb < nblk; ++sb)
                decode_superblock(type, wrow + static_cast<size_t>(sb) * kq_block_bytes(type),
                                  wbytes.data() + static_cast<size_t>(sb) * kQK_K,
                                  A.data() + static_cast<size_t>(sb) * kGPS,
                                  B.data() + static_cast<size_t>(sb) * kGPS);
            for (int m = m0; m < m1; ++m) {
                const int8_t* aq = act_q + static_cast<size_t>(m) * K;
                const float*  ad = act_d + static_cast<size_t>(m) * nb8;
                const int32_t* sumqx = sumqx_all + static_cast<size_t>(m) * total_groups;
                for (int sb = 0; sb < nblk; ++sb) {
                    const uint8_t* wb = wbytes.data() + static_cast<size_t>(sb) * kQK_K;
                    const int8_t*  ab = aq + static_cast<size_t>(sb) * kQK_K;
                    int32_t* dq = dotQ.data() + static_cast<size_t>(sb) * kGPS;
                    int32_t sq_tmp[kGPS];
#if LS_KQ_HAVE_VNNI
                    if (use_vnni) group_dots_vnni(wb, ab, dq, sq_tmp);
                    else          group_dots_scalar(wb, ab, dq, sq_tmp);
#else
                    (void)use_vnni; group_dots_scalar(wb, ab, dq, sq_tmp);
#endif
                    if (is_q6k) {   // centre: dotQ_centred = dotQ_raw - 32*sumqx (int, exact)
                        const int32_t* sqg = sumqx + static_cast<size_t>(sb) * kGPS;
                        for (int g = 0; g < kGPS; ++g) dq[g] -= 32 * sqg[g];
                    }
                }
                uint16_t out;
#if LS_KQ_HAVE_VNNI
                out = use_vnni
                    ? reduce_gpuorder_vec(dotQ.data(), sumqx, A.data(), B.data(), ad, nblk)
                    : reduce_gpuorder(dotQ.data(), sumqx, A.data(), B.data(), ad, nblk);
#else
                out = reduce_gpuorder(dotQ.data(), sumqx, A.data(), B.data(), ad, nblk);
#endif
                D[static_cast<size_t>(m) * N + n] = out;
            }
        }
    }
}

// Full-driver body shared by the VNNI (production) and scalar-reference entries.
void run_grouped(KQuantLossless type, bool use_vnni, uint16_t* D, const uint16_t* A_in,
                 const void* const* B_ptrs, const int32_t* expert_offsets,
                 int num_experts, int N, int K, NumaThreadPool* pool) {
    if (num_experts == 0 || N == 0) return;
    const int total_tokens = expert_offsets[num_experts];
    if (total_tokens <= 0) return;

    const int nblk = K / kQK_K;
    const int total_groups = nblk * kGPS;
    const size_t nb8 = static_cast<size_t>(K / kQK8);

    // Quantize every activation row to Q8_1 once (n-independent), + per-group sumqx.
    std::vector<float>   act_d(static_cast<size_t>(total_tokens) * nb8);
    std::vector<int8_t>  act_q(static_cast<size_t>(total_tokens) * K);
    std::vector<int32_t> sumqx_all(static_cast<size_t>(total_tokens) * total_groups);
    for (int m = 0; m < total_tokens; ++m) {
        int8_t* aq = act_q.data() + static_cast<size_t>(m) * K;
        q8_1_quantize_row(A_in + static_cast<size_t>(m) * K, K,
                          act_d.data() + static_cast<size_t>(m) * nb8, aq);
        int32_t* sq = sumqx_all.data() + static_cast<size_t>(m) * total_groups;
        for (int g = 0; g < total_groups; ++g) {
            const int8_t* a = aq + 4 * g;
            sq[g] = static_cast<int>(a[0]) + a[1] + a[2] + a[3];
        }
    }

    if (pool == nullptr || pool->num_threads() <= 1) {
        gemm_rows(type, use_vnni, D, B_ptrs, act_d.data(), act_q.data(),
                  sumqx_all.data(), expert_offsets, num_experts, N, K, 0, N);
        return;
    }
    pool->parallel_for([&](int ith, int nth, CpuBarrierState& /*b*/) {
        const int base = N / nth, rem = N % nth;
        const int n0 = ith * base + std::min(ith, rem);
        const int n1 = n0 + base + (ith < rem ? 1 : 0);
        if (n0 < n1)
            gemm_rows(type, use_vnni, D, B_ptrs, act_d.data(), act_q.data(),
                      sumqx_all.data(), expert_offsets, num_experts, N, K, n0, n1);
    });
}

}  // namespace

void q8_1_quantize_row(const uint16_t* a_bf16, int K, float* out_d, int8_t* out_qs) {
    const int nblk = K / kQK8;
    for (int b = 0; b < nblk; ++b) {
        const uint16_t* x = a_bf16 + static_cast<size_t>(b) * kQK8;
        float amax = 0.0f;
        for (int i = 0; i < kQK8; ++i) amax = std::max(amax, std::fabs(bf16_to_f32(x[i])));
        const float d = amax / 127.0f;                 // full-float d for the quant
        out_d[b] = round_to_f16(d);                     // FP16(d) is what the dot uses
        int8_t* q = out_qs + static_cast<size_t>(b) * kQK8;
        if (amax > 0.0f) {
            for (int i = 0; i < kQK8; ++i) {
                // GPU: __float2int_rn(xi / d) — DIVISION (not xi*(1/d)); RNE, no clamp
                // needed since |xi| <= amax => |xi/d| <= 127.
                int v = static_cast<int>(std::lrint(bf16_to_f32(x[i]) / d));
                q[i] = static_cast<int8_t>(std::max(-127, std::min(127, v)));
            }
        } else {
            for (int i = 0; i < kQK8; ++i) q[i] = 0;
        }
    }
}

void cpu_gguf_grouped_gemm_kq_lossless(
    KQuantLossless type, uint16_t* D, const uint16_t* A, const void* const* B_ptrs,
    const int32_t* expert_offsets, int num_experts, int N, int K,
    NumaThreadPool* pool) {
    run_grouped(type, /*use_vnni=*/true, D, A, B_ptrs, expert_offsets,
                num_experts, N, K, pool);
}

void cpu_gguf_grouped_gemm_kq_lossless_scalar_ref(
    KQuantLossless type, uint16_t* D, const uint16_t* A, const void* const* B_ptrs,
    const int32_t* expert_offsets, int num_experts, int N, int K,
    NumaThreadPool* pool) {
    run_grouped(type, /*use_vnni=*/false, D, A, B_ptrs, expert_offsets,
                num_experts, N, K, pool);
}

void cpu_gguf_grouped_gemm_q4k_lossless(
    uint16_t* D, const uint16_t* A, const void* const* B_ptrs,
    const int32_t* expert_offsets, int num_experts, int N, int K,
    NumaThreadPool* pool) {
    cpu_gguf_grouped_gemm_kq_lossless(KQuantLossless::Q4_K, D, A, B_ptrs,
                                      expert_offsets, num_experts, N, K, pool);
}

}  // namespace layerstorm::compute::cpu
