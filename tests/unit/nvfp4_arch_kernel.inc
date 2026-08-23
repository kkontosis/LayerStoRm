// Multi-arch NVFP4 decode + GEMV core, parameterized by namespace + arch.
//
// This file is the EXACT arch-tiered decode/GEMV logic from
// src/compute/cpu/nvfp4_cpu_kernel.cpp, factored into an includable unit so the
// PART-1 test/bench can compile and compare ALL THREE arch tiers (AVX-512 / AVX2
// / scalar) on the same host. The production kernel selects the tier with
// #if __AVX512F__ / #elif __AVX2__ / #else; here the SAME #if/#elif/#else is
// driven by the per-TU compile flags (-march=native vs -mavx2 -mno-avx512f vs
// scalar), so each of the three TUs that #include this picks a different tier.
//
// The includer must define:
//   LS_NVFP4_ARCH_NS   — namespace name (avx512 / avx2 / scalarref)
// and is compiled with the matching arch flags.
//
// CPU-only. No CUDA. (Test-side mirror of the shipped kernel; kept in lockstep.)

#include "model/quantization/fp8.h"

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace LS_NVFP4_ARCH_NS {

inline const float kE2M1Table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

inline int64_t scale_byte_index(int64_t n, int64_t g, int64_t groups) {
    const int64_t n_tiles_g = (groups + 3) / 4;
    const int64_t tile = (n / 128) * n_tiles_g + (g / 4);
    return tile * 512 + (n % 32) * 16 + ((n % 128) / 32) * 4 + (g % 4);
}

inline float bf16_to_f32(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

#if defined(__AVX512F__)
inline __m512 e2m1_table_vec() {
    return _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                          -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
}
inline __m512 decode_group16(const uint8_t* gp, __m512 table, __m512 gscale) {
    const __m128i bytes8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(gp));
    const __m256i b32 = _mm256_cvtepu8_epi32(bytes8);
    const __m512i b32z = _mm512_zextsi256_si512(b32);
    const __m512i lo = _mm512_and_si512(b32z, _mm512_set1_epi32(0x0F));
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi32(b32z, 4),
                                        _mm512_set1_epi32(0x0F));
    const __m512i il_lo = _mm512_unpacklo_epi32(lo, hi);
    const __m512i il_hi = _mm512_unpackhi_epi32(lo, hi);
    const __m512i idx = _mm512_permutex2var_epi32(
        il_lo,
        _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19,
                          4, 5, 6, 7, 20, 21, 22, 23),
        il_hi);
    const __m512 w = _mm512_permutexvar_ps(idx, table);
    return _mm512_mul_ps(w, gscale);
}
#elif defined(__AVX2__)
inline __m256 e2m1_table_lo() {
    return _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f);
}
inline __m256 e2m1_table_hi() {
    return _mm256_setr_ps(-0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
}
inline __m256 e2m1_decode8(__m256i idx16, __m256 tlo, __m256 thi) {
    const __m256i lo3 = _mm256_and_si256(idx16, _mm256_set1_epi32(0x07));
    const __m256 wlo = _mm256_permutevar8x32_ps(tlo, lo3);
    const __m256 whi = _mm256_permutevar8x32_ps(thi, lo3);
    const __m256i sign = _mm256_and_si256(idx16, _mm256_set1_epi32(0x08));
    const __m256 selhi =
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(sign, _mm256_set1_epi32(0x08)));
    return _mm256_blendv_ps(wlo, whi, selhi);
}
inline void decode_group16(const uint8_t* gp, __m256 tlo, __m256 thi,
                           __m256 gscale, __m256& w0, __m256& w1) {
    const __m128i bytes8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(gp));
    const __m256i b32 = _mm256_cvtepu8_epi32(bytes8);
    const __m256i lo = _mm256_and_si256(b32, _mm256_set1_epi32(0x0F));
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi32(b32, 4),
                                        _mm256_set1_epi32(0x0F));
    const __m256i il_lo = _mm256_unpacklo_epi32(lo, hi);
    const __m256i il_hi = _mm256_unpackhi_epi32(lo, hi);
    const __m256i idx0 = _mm256_permute2x128_si256(il_lo, il_hi, 0x20);
    const __m256i idx1 = _mm256_permute2x128_si256(il_lo, il_hi, 0x31);
    w0 = _mm256_mul_ps(e2m1_decode8(idx0, tlo, thi), gscale);
    w1 = _mm256_mul_ps(e2m1_decode8(idx1, tlo, thi), gscale);
}
inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(lo, lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_shuffle_ps(lo, lo, 0x55);
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}
#endif

// Projection geometry (matches PackedProjection in nvfp4_cpu_kernel.h).
struct Proj {
    const uint8_t* base;
    int N;
    int K;
};
inline const uint8_t* fp4_base(const Proj& p) { return p.base; }
inline const uint8_t* scale_base(const Proj& p) {
    const int64_t P = static_cast<int64_t>(p.N) * p.K;
    return p.base + (P + 1) / 2;
}

// Decode one full output row (K weights) into out (FP32). Arch-tiered.
inline void dequant_weight_row(const Proj& proj, int64_t n, float ws2,
                               float* out) {
    const int64_t K = proj.K;
    const int64_t groups = K / 16;
    const uint8_t* fp4_row = fp4_base(proj) + (n * K) / 2;
    const uint8_t* scl = scale_base(proj);
#if defined(__AVX512F__)
    const __m512 table = e2m1_table_vec();
#elif defined(__AVX2__)
    const __m256 tlo = e2m1_table_lo();
    const __m256 thi = e2m1_table_hi();
#endif
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t sb = scale_byte_index(n, g, groups);
        const float gscale =
            layerstorm::model::fp8_e4m3::decode(scl[sb]) * ws2;
        const int64_t k0 = g * 16;
        const uint8_t* gp = fp4_row + (k0 >> 1);
        float* op = out + k0;
#if defined(__AVX512F__)
        _mm512_storeu_ps(op,
                         decode_group16(gp, table, _mm512_set1_ps(gscale)));
#elif defined(__AVX2__)
        __m256 w0, w1;
        decode_group16(gp, tlo, thi, _mm256_set1_ps(gscale), w0, w1);
        _mm256_storeu_ps(op, w0);
        _mm256_storeu_ps(op + 8, w1);
#else
        for (int b = 0; b < 8; ++b) {
            const uint8_t byte = gp[b];
            op[2 * b]     = kE2M1Table[byte & 0x0F] * gscale;
            op[2 * b + 1] = kE2M1Table[byte >> 4]   * gscale;
        }
#endif
    }
}

// FP32 dot of BF16 activation row vs FP32 weight row. Arch-tiered.
inline float dot_bf16_f32(const uint16_t* a, const float* w, int K) {
#if defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 32 <= K; k += 32) {
        __m256i a16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k));
        __m256i a16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k + 16));
        __m512i wa0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a16_0), 16);
        __m512i wa1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a16_1), 16);
        acc0 = _mm512_fmadd_ps(_mm512_castsi512_ps(wa0), _mm512_loadu_ps(w + k), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_castsi512_ps(wa1), _mm512_loadu_ps(w + k + 16), acc1);
    }
    for (; k + 16 <= K; k += 16) {
        __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k));
        __m512i wa = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a16), 16);
        acc0 = _mm512_fmadd_ps(_mm512_castsi512_ps(wa), _mm512_loadu_ps(w + k), acc0);
    }
    float acc = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; k < K; ++k) acc += bf16_to_f32(a[k]) * w[k];
    return acc;
#elif defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        __m128i a8_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k));
        __m128i a8_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k + 8));
        __m256i wa0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(a8_0), 16);
        __m256i wa1 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(a8_1), 16);
        acc0 = _mm256_fmadd_ps(_mm256_castsi256_ps(wa0), _mm256_loadu_ps(w + k), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_castsi256_ps(wa1), _mm256_loadu_ps(w + k + 8), acc1);
    }
    for (; k + 8 <= K; k += 8) {
        __m128i a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k));
        __m256i wa = _mm256_slli_epi32(_mm256_cvtepu16_epi32(a8), 16);
        acc0 = _mm256_fmadd_ps(_mm256_castsi256_ps(wa), _mm256_loadu_ps(w + k), acc0);
    }
    float acc = hsum256_ps(_mm256_add_ps(acc0, acc1));
    for (; k < K; ++k) acc += bf16_to_f32(a[k]) * w[k];
    return acc;
#else
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += bf16_to_f32(a[k]) * w[k];
    return acc;
#endif
}

// Fused decode->FMA single-row M=1 GEMV. Arch-tiered.
inline float fused_decode_dot_row(const Proj& proj, int64_t n, float ws2,
                                  const uint16_t* a) {
    const int64_t K = proj.K;
    const int64_t groups = K / 16;
    const uint8_t* fp4_row = fp4_base(proj) + (n * K) / 2;
    const uint8_t* scl = scale_base(proj);
#if defined(__AVX512F__)
    const __m512 table = e2m1_table_vec();
    __m512 acc = _mm512_setzero_ps();
    for (int64_t g = 0; g < groups; ++g) {
        const float gscale =
            layerstorm::model::fp8_e4m3::decode(scl[scale_byte_index(n, g, groups)]) * ws2;
        const int64_t k0 = g * 16;
        const __m512 wv = decode_group16(fp4_row + (k0 >> 1), table, _mm512_set1_ps(gscale));
        const __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k0));
        const __m512i wa = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a16), 16);
        acc = _mm512_fmadd_ps(_mm512_castsi512_ps(wa), wv, acc);
    }
    return _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
    const __m256 tlo = e2m1_table_lo();
    const __m256 thi = e2m1_table_hi();
    __m256 acc = _mm256_setzero_ps();
    for (int64_t g = 0; g < groups; ++g) {
        const float gscale =
            layerstorm::model::fp8_e4m3::decode(scl[scale_byte_index(n, g, groups)]) * ws2;
        const int64_t k0 = g * 16;
        __m256 wv0, wv1;
        decode_group16(fp4_row + (k0 >> 1), tlo, thi, _mm256_set1_ps(gscale), wv0, wv1);
        const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k0));
        const __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k0 + 8));
        const __m256 av0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a0), 16));
        const __m256 av1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a1), 16));
        acc = _mm256_fmadd_ps(av0, wv0, acc);
        acc = _mm256_fmadd_ps(av1, wv1, acc);
    }
    return hsum256_ps(acc);
#else
    float acc = 0.0f;
    for (int64_t g = 0; g < groups; ++g) {
        const float gscale =
            layerstorm::model::fp8_e4m3::decode(scl[scale_byte_index(n, g, groups)]) * ws2;
        const int64_t k0 = g * 16;
        const uint8_t* gp = fp4_row + (k0 >> 1);
        for (int b = 0; b < 8; ++b) {
            const uint8_t byte = gp[b];
            acc += bf16_to_f32(a[k0 + 2 * b])     * (kE2M1Table[byte & 0x0F] * gscale);
            acc += bf16_to_f32(a[k0 + 2 * b + 1]) * (kE2M1Table[byte >> 4]   * gscale);
        }
    }
    return acc;
#endif
}

// ── Public arch-tagged entrypoints ──────────────────────────────────────────
// Decode a full expert projection (all N*K weights) into `out` (FP32).
void decode_full(const uint8_t* base, int N, int K, float ws2, float* out);
// M=1 grouped GEMV: D[n] = sum_k A[k] * W[n,k] for n in [0,N) using the
// fused decode->FMA single-row path (the steady-state decode hot path).
void gemv_m1_fused(const uint8_t* base, int N, int K, float ws2,
                   const uint16_t* a, float* out);
// Materialize-then-dot path (one decoded row reused — the M>1 path primitive).
void gemv_m1_materialize(const uint8_t* base, int N, int K, float ws2,
                         const uint16_t* a, float* out, float* row_scratch);

inline void decode_full(const uint8_t* base, int N, int K, float ws2,
                        float* out) {
    Proj p{base, N, K};
    for (int n = 0; n < N; ++n)
        dequant_weight_row(p, n, ws2, out + static_cast<size_t>(n) * K);
}

inline void gemv_m1_fused(const uint8_t* base, int N, int K, float ws2,
                          const uint16_t* a, float* out) {
    Proj p{base, N, K};
    for (int n = 0; n < N; ++n)
        out[n] = fused_decode_dot_row(p, n, ws2, a);
}

inline void gemv_m1_materialize(const uint8_t* base, int N, int K, float ws2,
                                const uint16_t* a, float* out,
                                float* row_scratch) {
    Proj p{base, N, K};
    for (int n = 0; n < N; ++n) {
        dequant_weight_row(p, n, ws2, row_scratch);
        out[n] = dot_bf16_f32(a, row_scratch, K);
    }
}

}  // namespace LS_NVFP4_ARCH_NS
