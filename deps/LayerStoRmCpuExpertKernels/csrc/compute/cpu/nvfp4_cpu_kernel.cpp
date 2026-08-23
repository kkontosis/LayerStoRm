// CPU NVFP4 grouped GEMM — dequant-weights-on-the-fly (C-6 nvfp4 kernel agent).
//
// Implements nvfp4_cpu_kernel.h:
//   * kE2M1Table[16]            — {0,0.5,1,1.5,2,3,4,6,-0,-0.5,-1,-1.5,-2,-3,-4,-6}
//   * scale_byte_index          — Sm1xx de-interleave (inverse of
//                                 model/weight_loader/nvfp4_sfb_reformat.h)
//   * read_weight_scale_2       — f32 at proj_base[proj_bytes-8]
//   * dequant_weight            — kE2M1Table[nibble] * fp8_e4m3::decode(group_scale)
//                                 * weight_scale_2
//   * cpu_nvfp4_grouped_gemm    — per-expert FP32-accumulate matmul over BF16
//                                 activations, dequant weights on the fly,
//                                 partition N output rows across pool threads.
//
// Numerics contract (matches tools/golden_ref/nvfp4.py dequant()):
//   w[n,k] = FP4_E2M1_TABLE[nibble(n,k)]
//          * fp8_e4m3::decode(group_scale_byte(n, k/16))    // E4M3, 0x38 == 1.0
//          * weight_scale_2
// Activations stay BF16, accumulate FP32, output BF16. input_scale is IGNORED.
//
// CPU-only TU — NO CUDA/GPU SDK header (INV-GPU-1, layerstorm_no_cuda_check).

#include "compute/cpu/nvfp4_cpu_kernel.h"

#include "compute/cpu/numa_thread_pool.h"
// Self-contained FP8 E4M3 decode declaration (definition: engine fp8.cpp).
// Avoids pulling the engine's quant_interface.h into this dependency.
#include "compute/cpu/fp8_e4m3_decode.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace layerstorm::compute::cpu {

// ── E2M1 (FP4) dequant table ────────────────────────────────────────────────
// bit3 = sign, bits2:0 = magnitude index. Matches FP4_E2M1_TABLE in
// tools/golden_ref/nvfp4.py and model/quantization/nvfp4 kE2M1Magnitudes.
const float kE2M1Table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// ── Sm1xx scale de-interleave ───────────────────────────────────────────────
// Byte offset within a projection's Sm1xx-INTERLEAVED E4M3 group-scale region of
// the scale for logical (row n, group g), G = groups = K/16.
// Inverse-companion of model/weight_loader/nvfp4_sfb_reformat.h reformat_nvfp4_sfb:
//   atom  = 128 rows x 4 groups = 512 bytes
//   tile  = (n / 128) * ceil(G / 4) + (g / 4)
//   byte  = tile * 512 + (n % 32) * 16 + ((n % 128) / 32) * 4 + (g % 4)
int64_t scale_byte_index(int64_t n, int64_t g, int64_t groups) {
    const int64_t n_tiles_g = (groups + 3) / 4;          // ceil(G / 4)
    const int64_t tile = (n / 128) * n_tiles_g + (g / 4);
    return tile * 512 + (n % 32) * 16 + ((n % 128) / 32) * 4 + (g % 4);
}

// ── weight_scale_2 ──────────────────────────────────────────────────────────
// f32 at proj_end - 8 (input_scale at proj_end - 4 is IGNORED on this path).
float read_weight_scale_2(const uint8_t* proj_base, size_t proj_bytes) {
    assert(proj_bytes >= 8 && "projection block too small for tail scalars");
    float ws2;
    std::memcpy(&ws2, proj_base + proj_bytes - 8, sizeof(float));
    return ws2;
}

// ── Packed-projection geometry helpers (internal) ───────────────────────────
namespace {

// FP4 weight region: (P+1)/2 bytes, 2 vals/byte, P = N*K.
inline const uint8_t* fp4_base(const PackedProjection& proj) {
    return proj.base;
}

// E4M3 group-scale region follows the FP4 region. P = N*K, group_size = 16.
inline const uint8_t* scale_base(const PackedProjection& proj) {
    const int64_t P = static_cast<int64_t>(proj.N) * proj.K;
    const int64_t fp4_bytes = (P + 1) / 2;
    return proj.base + fp4_bytes;
}

}  // namespace

// ── Single weight-element dequant (reference / scalar path) ─────────────────
float dequant_weight(const PackedProjection& proj, int64_t n, int64_t k,
                     float weight_scale_2) {
    const int64_t K = proj.K;
    const int64_t groups = K / 16;
    const int64_t P = static_cast<int64_t>(proj.N) * K;

    // FP4 nibble: row-major [N, K] packed 2 vals/byte; low nibble = even k.
    const int64_t flat = n * K + k;
    const uint8_t byte = fp4_base(proj)[flat >> 1];
    const uint8_t nibble = (flat & 1) ? (byte >> 4) : (byte & 0x0F);

    // E4M3 group scale (de-interleaved Sm1xx index).
    const int64_t sb = scale_byte_index(n, k / 16, groups);
    const float gs = model::fp8_e4m3::decode(scale_base(proj)[sb]);

    (void)P;
    return kE2M1Table[nibble] * gs * weight_scale_2;
}

// ── Vectorized E2M1 (FP4) decode ────────────────────────────────────────────
namespace {

#if defined(__AVX512F__)
// The 16-entry signed E2M1 table is a perfect 16-lane FP32 shuffle-LUT. Holding
// it in one __m512 lets vpermps (_mm512_permutexvar_ps, AVX-512F) decode all 16
// nibbles of a group in a single instruction.
inline __m512 e2m1_table_vec() {
    return _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                          -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
}

// Decode 16 packed FP4 elements (8 bytes at `gp`) into 16 FP32 weights, scaled
// by the broadcast group scale. Nibble order matches the scalar reference:
//   out[2i]   = table[gp[i] & 0x0F]   (even k = low nibble)
//   out[2i+1] = table[gp[i] >> 4]     (odd  k = high nibble)
inline __m512 decode_group16(const uint8_t* gp, __m512 table, __m512 gscale) {
    // Widen the 8 packed bytes into 8 int32 lanes (lanes 0..7).
    const __m128i bytes8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(gp));
    const __m256i b32 = _mm256_cvtepu8_epi32(bytes8);          // 8 x int32
    const __m512i b32z = _mm512_zextsi256_si512(b32);          // lanes 0..7 set
    const __m512i lo = _mm512_and_si512(b32z, _mm512_set1_epi32(0x0F));
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi32(b32z, 4),
                                        _mm512_set1_epi32(0x0F));
    // Interleave low/high into element order: idx[2i]=lo[i], idx[2i+1]=hi[i].
    // unpacklo/hi operate per 128-bit lane (4 int32 each), so the result lands
    // in the right 2i / 2i+1 positions across the low 8 source lanes.
    const __m512i il_lo = _mm512_unpacklo_epi32(lo, hi);  // src lanes 0,1 & 4,5
    const __m512i il_hi = _mm512_unpackhi_epi32(lo, hi);  // src lanes 2,3 & 6,7
    // Reassemble into linear element order across the 4 source 128-bit halves.
    // Source 256-bit b32 occupies lanes 0..7; unpack produced:
    //   il_lo lane-groups: [l0 h0 l1 h1 | l4 h4 l5 h5]  -> elems 0..3, 8..11
    //   il_hi lane-groups: [l2 h2 l3 h3 | l6 h6 l7 h7]  -> elems 4..7, 12..15
    const __m512i idx = _mm512_permutex2var_epi32(
        il_lo,
        _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19,
                          4, 5, 6, 7, 20, 21, 22, 23),
        il_hi);
    const __m512 w = _mm512_permutexvar_ps(idx, table);
    return _mm512_mul_ps(w, gscale);
}

#elif defined(__AVX2__)
// AVX2 has only an 8-wide cross-lane FP32 permute (_mm256_permutevar8x32_ps), so
// the 16-entry E2M1 table is split into two 8-lane halves. Each nibble decode is
// two 8-wide permutes (low half = table[0..7], high half = table[8..15]) blended
// by the index's bit3 (sign bit). 16 weights = two __m256 (lo8 / hi8 elements).
inline __m256 e2m1_table_lo() {
    return _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f);
}
inline __m256 e2m1_table_hi() {
    return _mm256_setr_ps(-0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
}

// Decode 8 nibble indices (int32 lanes, values 0..15) into 8 FP32 weights via the
// split LUT. Low 3 bits index both halves; bit3 selects hi vs lo.
inline __m256 e2m1_decode8(__m256i idx16, __m256 tlo, __m256 thi) {
    const __m256i lo3 = _mm256_and_si256(idx16, _mm256_set1_epi32(0x07));
    const __m256 wlo = _mm256_permutevar8x32_ps(tlo, lo3);
    const __m256 whi = _mm256_permutevar8x32_ps(thi, lo3);
    // sign bit set (bit3) => take hi-half value. Build a mask from bit3.
    const __m256i sign = _mm256_and_si256(idx16, _mm256_set1_epi32(0x08));
    const __m256 selhi =
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(sign, _mm256_set1_epi32(0x08)));
    return _mm256_blendv_ps(wlo, whi, selhi);
}

// Decode 16 packed FP4 elements (8 bytes at gp) -> two __m256 (elems 0..7, 8..15)
// scaled by the broadcast group scale. Nibble order matches the scalar reference:
//   out[2i] = table[gp[i] & 0x0F]  (even k), out[2i+1] = table[gp[i] >> 4] (odd k).
inline void decode_group16(const uint8_t* gp, __m256 tlo, __m256 thi,
                           __m256 gscale, __m256& w0, __m256& w1) {
    const __m128i bytes8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(gp));
    const __m256i b32 = _mm256_cvtepu8_epi32(bytes8);              // 8 x int32
    const __m256i lo = _mm256_and_si256(b32, _mm256_set1_epi32(0x0F));
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi32(b32, 4),
                                        _mm256_set1_epi32(0x0F));
    // Interleave lo/hi into element order idx[2i]=lo[i], idx[2i+1]=hi[i].
    // _mm256_unpacklo/hi_epi32 operate per 128-bit lane (4 int32 each):
    //   il_lo lanes: [l0 h0 l1 h1 | l4 h4 l5 h5]  (src lanes 0,1 & 4,5)
    //   il_hi lanes: [l2 h2 l3 h3 | l6 h6 l7 h7]  (src lanes 2,3 & 6,7)
    const __m256i il_lo = _mm256_unpacklo_epi32(lo, hi);
    const __m256i il_hi = _mm256_unpackhi_epi32(lo, hi);
    // Reassemble linear element order:
    //   elems 0..7  = [il_lo.lo128 (0..3) | il_hi.lo128 (4..7)]
    //   elems 8..15 = [il_lo.hi128 (8..11)| il_hi.hi128 (12..15)]
    const __m256i idx0 = _mm256_permute2x128_si256(il_lo, il_hi, 0x20);
    const __m256i idx1 = _mm256_permute2x128_si256(il_lo, il_hi, 0x31);
    w0 = _mm256_mul_ps(e2m1_decode8(idx0, tlo, thi), gscale);
    w1 = _mm256_mul_ps(e2m1_decode8(idx1, tlo, thi), gscale);
}

// Horizontal sum of a __m256.
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
#endif  // arch select

}  // namespace

// ── BF16 helpers ────────────────────────────────────────────────────────────
namespace {

inline float bf16_to_f32(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Round-to-nearest-even FP32 -> BF16 (matches the GPU __float2bfloat16 default).
inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) {  // NaN: keep it quiet
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    }
    const uint32_t rounding = 0x7fffu + ((u >> 16) & 1u);
    u += rounding;
    return static_cast<uint16_t>(u >> 16);
}

// Dequantize one full output row's K weights (W[n, 0..K)) into `out` (FP32).
// Walks the row in 16-element groups so the E4M3 group scale (* weight_scale_2)
// is decoded once per group and broadcast across its 16 elements.
inline void dequant_weight_row(const PackedProjection& proj, int64_t n,
                               float weight_scale_2, float* out) {
    const int64_t K = proj.K;
    const int64_t groups = K / 16;
    const uint8_t* fp4_row = fp4_base(proj) + (n * K) / 2;  // K even => byte-aligned
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
            model::fp8_e4m3::decode(scl[sb]) * weight_scale_2;
        const int64_t k0 = g * 16;
        // 16 elements = 8 packed bytes; low nibble = even k, high = odd k.
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

// Scalar (non-vpermps) variant of dequant_weight_row. Decodes one full output
// row's K weights with the per-nibble scalar table lookup — the pre-Change-1
// decode. Used by the V0 attribution measurement only.
inline void dequant_weight_row_scalar(const PackedProjection& proj, int64_t n,
                                      float weight_scale_2, float* out) {
    const int64_t K = proj.K;
    const int64_t groups = K / 16;
    const uint8_t* fp4_row = fp4_base(proj) + (n * K) / 2;
    const uint8_t* scl = scale_base(proj);
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t sb = scale_byte_index(n, g, groups);
        const float gscale =
            model::fp8_e4m3::decode(scl[sb]) * weight_scale_2;
        const int64_t k0 = g * 16;
        const uint8_t* gp = fp4_row + (k0 >> 1);
        float* op = out + k0;
        for (int b = 0; b < 8; ++b) {
            const uint8_t byte = gp[b];
            op[2 * b]     = kE2M1Table[byte & 0x0F] * gscale;
            op[2 * b + 1] = kE2M1Table[byte >> 4]   * gscale;
        }
    }
}

// FP32 dot of a BF16 activation row (length K) with an FP32 weight row.
inline float dot_bf16_f32(const uint16_t* a, const float* w, int K) {
#if defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 32 <= K; k += 32) {
        // Widen 16 BF16 -> 16 FP32 by zero-extending then shifting left 16 bits.
        __m256i a16_0 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a + k));
        __m256i a16_1 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a + k + 16));
        __m512i wa0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a16_0), 16);
        __m512i wa1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a16_1), 16);
        __m512 av0 = _mm512_castsi512_ps(wa0);
        __m512 av1 = _mm512_castsi512_ps(wa1);
        acc0 = _mm512_fmadd_ps(av0, _mm512_loadu_ps(w + k), acc0);
        acc1 = _mm512_fmadd_ps(av1, _mm512_loadu_ps(w + k + 16), acc1);
    }
    for (; k + 16 <= K; k += 16) {
        __m256i a16 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a + k));
        __m512i wa = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a16), 16);
        acc0 = _mm512_fmadd_ps(_mm512_castsi512_ps(wa),
                               _mm512_loadu_ps(w + k), acc0);
    }
    float acc = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; k < K; ++k) acc += bf16_to_f32(a[k]) * w[k];
    return acc;
#elif defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        // Widen 8 BF16 -> 8 FP32 by zero-extending then shifting left 16 bits.
        __m128i a8_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k));
        __m128i a8_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k + 8));
        __m256i wa0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(a8_0), 16);
        __m256i wa1 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(a8_1), 16);
        acc0 = _mm256_fmadd_ps(_mm256_castsi256_ps(wa0),
                               _mm256_loadu_ps(w + k), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_castsi256_ps(wa1),
                               _mm256_loadu_ps(w + k + 8), acc1);
    }
    for (; k + 8 <= K; k += 8) {
        __m128i a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k));
        __m256i wa = _mm256_slli_epi32(_mm256_cvtepu16_epi32(a8), 16);
        acc0 = _mm256_fmadd_ps(_mm256_castsi256_ps(wa),
                               _mm256_loadu_ps(w + k), acc0);
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

// Fused decode->FMA GEMV for M=1: dot one BF16 activation row (length K) with
// output weight row n WITHOUT materializing the FP32 weight row. Decodes 16
// weights per group on the fly (Change-1 vectorized), widens 16 BF16 acts to
// FP32, FMAs into a running accumulator, then horizontal-reduces. Matches the
// materialize-then-dot path bit-for-bit (same group scale, same nibble order).
inline float fused_decode_dot_row(const PackedProjection& proj, int64_t n,
                                  float weight_scale_2, const uint16_t* a) {
    const int64_t K = proj.K;
    const int64_t groups = K / 16;
    const uint8_t* fp4_row = fp4_base(proj) + (n * K) / 2;
    const uint8_t* scl = scale_base(proj);
#if defined(__AVX512F__)
    const __m512 table = e2m1_table_vec();
    __m512 acc = _mm512_setzero_ps();
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t sb = scale_byte_index(n, g, groups);
        const float gscale =
            model::fp8_e4m3::decode(scl[sb]) * weight_scale_2;
        const int64_t k0 = g * 16;
        const __m512 wv =
            decode_group16(fp4_row + (k0 >> 1), table, _mm512_set1_ps(gscale));
        // Widen 16 BF16 acts -> 16 FP32 (zero-extend + shift left 16).
        const __m256i a16 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k0));
        const __m512i wa = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a16), 16);
        acc = _mm512_fmadd_ps(_mm512_castsi512_ps(wa), wv, acc);
    }
    return _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
    const __m256 tlo = e2m1_table_lo();
    const __m256 thi = e2m1_table_hi();
    __m256 acc = _mm256_setzero_ps();
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t sb = scale_byte_index(n, g, groups);
        const float gscale =
            model::fp8_e4m3::decode(scl[sb]) * weight_scale_2;
        const int64_t k0 = g * 16;
        __m256 wv0, wv1;
        decode_group16(fp4_row + (k0 >> 1), tlo, thi, _mm256_set1_ps(gscale),
                       wv0, wv1);
        // Widen 16 BF16 acts -> two __m256 (elems 0..7, 8..15).
        const __m128i a0 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k0));
        const __m128i a1 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k0 + 8));
        const __m256 av0 = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(a0), 16));
        const __m256 av1 = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(a1), 16));
        acc = _mm256_fmadd_ps(av0, wv0, acc);
        acc = _mm256_fmadd_ps(av1, wv1, acc);
    }
    return hsum256_ps(acc);
#else
    float acc = 0.0f;
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t sb = scale_byte_index(n, g, groups);
        const float gscale =
            model::fp8_e4m3::decode(scl[sb]) * weight_scale_2;
        const int64_t k0 = g * 16;
        const uint8_t* gp = fp4_row + (k0 >> 1);
        for (int b = 0; b < 8; ++b) {
            const uint8_t byte = gp[b];
            acc += bf16_to_f32(a[k0 + 2 * b]) *
                   (kE2M1Table[byte & 0x0F] * gscale);
            acc += bf16_to_f32(a[k0 + 2 * b + 1]) *
                   (kE2M1Table[byte >> 4] * gscale);
        }
    }
    return acc;
#endif
}

// ── ik_llama multi-row GEMV technique, ported to NVFP4 decode ────────────────
//
// PORTED TECHNIQUE (NOT a code copy — ik's kernels consume Q4_K/IQ4_K/Q8_0 and
// CANNOT read nvfp4-sm1xx). Borrowed structure from ik_llama.cpp
// ggml/src/iqk/iqk_gemm_floats.cpp:
//   * mul_mat_Qx_Qy_MxN  (lines 183-223): for the GEMV case it holds ONE
//     activation vector `yv` per K-step and FMAs it into Qx::nrc INDEPENDENT
//     accumulators acc[ix] — one per output row — so the per-row FMA dependency
//     chains run in parallel (ILP hides the FMA latency).
//   * mul_mat_fX_fY_T    (lines 278-329): the GEMV dispatcher picks a register
//     row-block of k_nx = 5 (AVX-512) / 4 (AVX2) output rows, then handles the
//     1..4 row remainder with smaller specializations.
// Here `yv` is the 16 widened BF16 activations of one K-group (loaded ONCE per
// group), and each output row gets its own decode_group16 (vpermps) + FMA into
// its own __m512 accumulator. Reusing the activation load across the row block
// is the load-amortization half; the independent accumulators are the
// latency-hiding half. We process kRowBlock = 4 rows (4 independent vpermps +
// FMA chains comfortably fit the SKX/ICX register file and the FMA issue width)
// and fall back to fused_decode_dot_row for the row remainder.
//
// MIT License — technique borrowed from ik_llama.cpp.
//   Copyright (c) 2024-2025 The ik_llama.cpp authors
//   (https://github.com/ikawrakow/ik_llama.cpp/blob/main/AUTHORS)
// (Same license/attribution model as ik_barrier.h.)
inline void fused_decode_dot_block4(const PackedProjection& proj, int n0,
                                    float weight_scale_2, const uint16_t* a,
                                    float* out4) {
    const int64_t K = proj.K;
    const int64_t groups = K / 16;
    const uint8_t* scl = scale_base(proj);
#if defined(__AVX512F__)
    const __m512 table = e2m1_table_vec();
    const uint8_t* fp4_r0 = fp4_base(proj) + (static_cast<int64_t>(n0) * K) / 2;
    const int64_t row_bytes = K / 2;  // packed FP4 bytes per output row
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t k0 = g * 16;
        // Activation group loaded ONCE, widened BF16 -> FP32, reused for 4 rows.
        const __m256i a16 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k0));
        const __m512 av =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a16), 16));
        // Per-row group scale (de-interleaved index depends on the row n).
        const float gs0 =
            model::fp8_e4m3::decode(scl[scale_byte_index(n0, g, groups)]) *
            weight_scale_2;
        const float gs1 =
            model::fp8_e4m3::decode(scl[scale_byte_index(n0 + 1, g, groups)]) *
            weight_scale_2;
        const float gs2 =
            model::fp8_e4m3::decode(scl[scale_byte_index(n0 + 2, g, groups)]) *
            weight_scale_2;
        const float gs3 =
            model::fp8_e4m3::decode(scl[scale_byte_index(n0 + 3, g, groups)]) *
            weight_scale_2;
        const uint8_t* gp = fp4_r0 + (k0 >> 1);
        // 4 independent decode -> FMA chains. The compiler is free to interleave
        // the vpermps/cvt/FMA across rows because acc0..acc3 are independent.
        acc0 = _mm512_fmadd_ps(
            av, decode_group16(gp, table, _mm512_set1_ps(gs0)), acc0);
        acc1 = _mm512_fmadd_ps(
            av, decode_group16(gp + row_bytes, table, _mm512_set1_ps(gs1)), acc1);
        acc2 = _mm512_fmadd_ps(
            av, decode_group16(gp + 2 * row_bytes, table, _mm512_set1_ps(gs2)),
            acc2);
        acc3 = _mm512_fmadd_ps(
            av, decode_group16(gp + 3 * row_bytes, table, _mm512_set1_ps(gs3)),
            acc3);
    }
    out4[0] = _mm512_reduce_add_ps(acc0);
    out4[1] = _mm512_reduce_add_ps(acc1);
    out4[2] = _mm512_reduce_add_ps(acc2);
    out4[3] = _mm512_reduce_add_ps(acc3);
#elif defined(__AVX2__)
    const __m256 tlo = e2m1_table_lo();
    const __m256 thi = e2m1_table_hi();
    const uint8_t* fp4_r0 = fp4_base(proj) + (static_cast<int64_t>(n0) * K) / 2;
    const int64_t row_bytes = K / 2;  // packed FP4 bytes per output row
    // 4 rows x 2 halves = 8 independent accumulator chains (fit AVX2's 16 ymm).
    __m256 a0lo = _mm256_setzero_ps(), a0hi = _mm256_setzero_ps();
    __m256 a1lo = _mm256_setzero_ps(), a1hi = _mm256_setzero_ps();
    __m256 a2lo = _mm256_setzero_ps(), a2hi = _mm256_setzero_ps();
    __m256 a3lo = _mm256_setzero_ps(), a3hi = _mm256_setzero_ps();
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t k0 = g * 16;
        // Activation group loaded ONCE, widened BF16 -> FP32, reused for 4 rows.
        const __m128i av0i =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k0));
        const __m128i av1i =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k0 + 8));
        const __m256 av0 = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(av0i), 16));
        const __m256 av1 = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(av1i), 16));
        const float gs0 = model::fp8_e4m3::decode(
                              scl[scale_byte_index(n0, g, groups)]) * weight_scale_2;
        const float gs1 = model::fp8_e4m3::decode(
                              scl[scale_byte_index(n0 + 1, g, groups)]) * weight_scale_2;
        const float gs2 = model::fp8_e4m3::decode(
                              scl[scale_byte_index(n0 + 2, g, groups)]) * weight_scale_2;
        const float gs3 = model::fp8_e4m3::decode(
                              scl[scale_byte_index(n0 + 3, g, groups)]) * weight_scale_2;
        const uint8_t* gp = fp4_r0 + (k0 >> 1);
        __m256 w0lo, w0hi, w1lo, w1hi, w2lo, w2hi, w3lo, w3hi;
        decode_group16(gp,                 tlo, thi, _mm256_set1_ps(gs0), w0lo, w0hi);
        decode_group16(gp + row_bytes,     tlo, thi, _mm256_set1_ps(gs1), w1lo, w1hi);
        decode_group16(gp + 2 * row_bytes, tlo, thi, _mm256_set1_ps(gs2), w2lo, w2hi);
        decode_group16(gp + 3 * row_bytes, tlo, thi, _mm256_set1_ps(gs3), w3lo, w3hi);
        a0lo = _mm256_fmadd_ps(av0, w0lo, a0lo); a0hi = _mm256_fmadd_ps(av1, w0hi, a0hi);
        a1lo = _mm256_fmadd_ps(av0, w1lo, a1lo); a1hi = _mm256_fmadd_ps(av1, w1hi, a1hi);
        a2lo = _mm256_fmadd_ps(av0, w2lo, a2lo); a2hi = _mm256_fmadd_ps(av1, w2hi, a2hi);
        a3lo = _mm256_fmadd_ps(av0, w3lo, a3lo); a3hi = _mm256_fmadd_ps(av1, w3hi, a3hi);
    }
    out4[0] = hsum256_ps(_mm256_add_ps(a0lo, a0hi));
    out4[1] = hsum256_ps(_mm256_add_ps(a1lo, a1hi));
    out4[2] = hsum256_ps(_mm256_add_ps(a2lo, a2hi));
    out4[3] = hsum256_ps(_mm256_add_ps(a3lo, a3hi));
#else
    // Scalar fallback: just do four single-row dots (still independent chains;
    // the scalar compiler can interleave them).
    for (int r = 0; r < 4; ++r)
        out4[r] = fused_decode_dot_row(proj, n0 + r, weight_scale_2, a);
#endif
}

// Core grouped GEMM for the N-row slice [n_begin, n_end). Dequants each weight
// row once and applies it across all permuted tokens of the owning expert.
//
// The FP32 row-materialize scratch (K floats) is allocated LAZILY — only when an
// expert with >1 routed token is hit. The M=1 hot path (one token/expert, the
// decode-bound steady-state decode case) fuses decode->FMA and never materializes
// a row, so on that path no scratch is touched at all: this drops a per-thread
// std::vector<float>(K) (~28 KB) malloc+first-touch+free from EVERY projection
// GEMM, which on a SMALL per-node slice (MultiNuma M=4) was a measurable fraction
// of the call. row_scratch_buf is the caller's reusable storage (resized once).
void grouped_gemm_rows(uint16_t* D, const uint16_t* A,
                       const CpuNvfp4ExpertWeights& weights,
                       const int32_t* expert_offsets, int N, int K,
                       int n_begin, int n_end, std::vector<float>& row_scratch_buf,
                       M1GemvPath m1_path) {
    for (int e = 0; e < weights.num_experts; ++e) {
        const int m_begin = expert_offsets[e];
        const int m_end = expert_offsets[e + 1];
        if (m_end <= m_begin) continue;  // expert routed no tokens this batch

        const PackedProjection& proj = weights.projections[e];
        const float ws2 = read_weight_scale_2(proj.base, proj.proj_bytes);

        // M=1 fast path: fuse decode->FMA so we never materialize the K-length
        // FP32 weight row (pure waste at one token — the gate GEMM alone would
        // round-trip ~114 MB of scratch per call). Decode reuse across tokens
        // only pays off at M>1, so keep materialize-then-dot there.
        if (m_end - m_begin == 1) {
            const uint16_t* a_row = A + static_cast<size_t>(m_begin) * K;
            const size_t d_base = static_cast<size_t>(m_begin) * N;
            if (m1_path == M1GemvPath::kFusedRowBlock) {
                // ik-technique: process 4 output rows at a time with independent
                // decode->FMA accumulator chains (ILP hides the vpermps->FMA
                // latency), reusing each activation group load across the block.
                int n = n_begin;
                for (; n + 4 <= n_end; n += 4) {
                    float r4[4];
                    fused_decode_dot_block4(proj, n, ws2, a_row, r4);
                    D[d_base + n]     = f32_to_bf16(r4[0]);
                    D[d_base + n + 1] = f32_to_bf16(r4[1]);
                    D[d_base + n + 2] = f32_to_bf16(r4[2]);
                    D[d_base + n + 3] = f32_to_bf16(r4[3]);
                }
                for (; n < n_end; ++n) {
                    const float acc = fused_decode_dot_row(proj, n, ws2, a_row);
                    D[d_base + n] = f32_to_bf16(acc);
                }
            } else {
                for (int n = n_begin; n < n_end; ++n) {
                    const float acc = fused_decode_dot_row(proj, n, ws2, a_row);
                    D[d_base + n] = f32_to_bf16(acc);
                }
            }
            continue;
        }

        // Multi-token expert: materialize each decoded weight row once and reuse
        // it across this expert's tokens. Allocate the scratch on first need.
        if (row_scratch_buf.size() < static_cast<size_t>(K))
            row_scratch_buf.resize(static_cast<size_t>(K));
        float* row_scratch = row_scratch_buf.data();
        for (int n = n_begin; n < n_end; ++n) {
            dequant_weight_row(proj, n, ws2, row_scratch);
            for (int m = m_begin; m < m_end; ++m) {
                const uint16_t* a_row = A + static_cast<size_t>(m) * K;
                const float acc = dot_bf16_f32(a_row, row_scratch, K);
                D[static_cast<size_t>(m) * N + n] = f32_to_bf16(acc);
            }
        }
    }
}

// Partition N output rows across [ith, nth) in 4-row blocks, spreading the block
// remainder ONE-per-thread over the leading threads (balanced; not ceil-packed).
//
// Rationale (per-node-slice latency, multi-NUMA): a coarse ceil partition
// (e.g. 16-row blocks, all remainder loaded onto the leading threads) leaves the
// trailing threads idle on a SMALL slice — exactly the per-node case of
// MultiNumaCpuExpertDevice, where N is already split M ways before reaching this
// kernel. With M=4 a 512-row gate slice over 14 threads previously used only ~11
// threads with a lopsided tail; the fine, balanced split keeps all 14 busy and
// equal-length, recovering the per-row throughput the single-node (full-N) path
// gets. 4-row blocks match the ik row-block GEMV granularity (process-4-rows) so
// the M=1 fast path's block never splits mid-iteration; the materialize path is
// per-row, so any block size is correct. Group alignment is NOT required — each
// output row's scale de-interleave is self-contained per row (scale_byte_index
// takes the absolute row n), so a thread boundary between rows never straddles a
// scale group.
inline void row_partition(int N, int ith, int nth, int& n_begin, int& n_end) {
    constexpr int kBlock = 4;
    const int n_blocks = (N + kBlock - 1) / kBlock;
    const int base = n_blocks / nth;          // whole blocks each thread gets
    const int rem = n_blocks % nth;           // first `rem` threads get one extra
    // Block range [b0, b1) for this thread: leading `rem` threads carry base+1.
    const int b0 = ith * base + std::min(ith, rem);
    const int b1 = b0 + base + (ith < rem ? 1 : 0);
    n_begin = std::min(b0 * kBlock, N);
    n_end = std::min(b1 * kBlock, N);
}

// ── Bit-compatible activation-quantization helpers ──────────────────────────
//
// These reproduce the GPU activation quantizer (LayerStoRmGemmKernels
// smxx/quant/bf16_to_nvfp4_grouped_detail.cuh) on the CPU so a CPU-computed
// expert consumes the SAME lossy NVFP4 activations the GPU CUTLASS path does.

// Quantize a scalar to the nearest E2M1 nibble (0..15). Sign in bit3. Thresholds
// are the E2M1 magnitude midpoints — IDENTICAL to grouped_quantize_to_fp4.
inline uint8_t quantize_to_fp4_nibble(float v) {
    const uint8_t sign = (v < 0.0f) ? 8 : 0;
    const float a = std::fabs(v);
    const uint8_t idx = (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) +
                        (a >= 1.75f) + (a >= 2.5f) + (a >= 3.5f) + (a >= 5.0f);
    return static_cast<uint8_t>(idx | sign);
}

// Encode a NON-NEGATIVE float to a UE4M3 (E4M3, sign always 0) byte, round-to-
// nearest-even, saturating to the max finite 448 (== 0x7e) — matching CUDA
// __nv_fp8_e4m3(clamp(v,0,480)) used by grouped_float_to_ue4m3. NaN/<=0 → 0x00.
// Decoded with the SAME model::fp8_e4m3::decode used for the weight scales, so
// the round-trip scale_repr is self-consistent with the weight path.
inline uint8_t float_to_ue4m3(float v) {
    if (!(v > 0.0f)) return 0x00;          // v<=0 or NaN → 0
    if (v >= 448.0f) return 0x7e;          // saturate to max finite
    int e;
    const float m = std::frexp(v, &e);     // v = m·2^e, m ∈ [0.5,1)
    float mant = 2.0f * m;                 // v = mant·2^E, mant ∈ [1,2)
    int E = e - 1;
    if (E < -6) {                          // subnormal (exp field 0)
        const float sv = v * 512.0f;       // v / 2^-9  (subnormal LSB = 2^-9)
        int q = static_cast<int>(std::lrint(sv));
        if (q <= 0) return 0x00;
        if (q > 7) return static_cast<uint8_t>(1 << 3);  // rounded up to 2^-6 (smallest normal)
        return static_cast<uint8_t>(q);
    }
    int q = static_cast<int>(std::lrint((mant - 1.0f) * 8.0f));  // mantissa, RNE
    if (q == 8) { q = 0; ++E; }            // mantissa carry into exponent
    if (E > 8) return 0x7e;                // saturate
    const int exp_field = E + 7;           // bias 7
    return static_cast<uint8_t>((exp_field << 3) | q);
}

// Core bit-compatible grouped GEMM for the N-row slice [n_begin, n_end): FP32
// pre-quantized activations `Adeq` [total_tokens, K] dotted against the
// dequantized NVFP4 weight rows (same weight decode as the lossy path). FP32
// accumulate, BF16 store — matching the GPU accumulate dtype (order differs).
void grouped_gemm_rows_actquant(uint16_t* D, const float* Adeq,
                                const CpuNvfp4ExpertWeights& weights,
                                const int32_t* expert_offsets, int N, int K,
                                int n_begin, int n_end,
                                std::vector<float>& row_scratch_buf) {
    if (row_scratch_buf.size() < static_cast<size_t>(K))
        row_scratch_buf.resize(static_cast<size_t>(K));
    float* row = row_scratch_buf.data();
    for (int e = 0; e < weights.num_experts; ++e) {
        const int m_begin = expert_offsets[e];
        const int m_end = expert_offsets[e + 1];
        if (m_end <= m_begin) continue;
        const PackedProjection& proj = weights.projections[e];
        const float ws2 = read_weight_scale_2(proj.base, proj.proj_bytes);
        for (int n = n_begin; n < n_end; ++n) {
            dequant_weight_row(proj, n, ws2, row);   // FP32 weight row [K]
            for (int m = m_begin; m < m_end; ++m) {
                const float* a = Adeq + static_cast<size_t>(m) * K;
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += a[k] * row[k];
                D[static_cast<size_t>(m) * N + n] = f32_to_bf16(acc);
            }
        }
    }
}

}  // namespace

// ── Grouped GEMM (public) ───────────────────────────────────────────────────
void cpu_nvfp4_grouped_gemm(void* D, const void* A,
                            const CpuNvfp4ExpertWeights& weights,
                            const int32_t* expert_offsets, int N, int K,
                            int elem_size_bytes, NumaThreadPool* pool) {
    assert(elem_size_bytes == 2 && "CPU NVFP4 GEMM only supports BF16 (2 bytes)");
    assert((K % 16) == 0 && "K must be a multiple of the NVFP4 group size (16)");
    (void)elem_size_bytes;

    auto* Du = static_cast<uint16_t*>(D);
    const auto* Au = static_cast<const uint16_t*>(A);

    const M1GemvPath m1_path = select_m1_gemv_path();

    if (pool == nullptr || pool->num_threads() <= 1) {
        // Lazy scratch: empty until a multi-token expert needs row materialize.
        std::vector<float> scratch;
        grouped_gemm_rows(Du, Au, weights, expert_offsets, N, K, 0, N,
                          scratch, m1_path);
        return;
    }

    pool->parallel_for([&](int ith, int nth, CpuBarrierState& /*barrier*/) {
        // Thread-local scratch first-touched on the running thread's core (so
        // the dequant row buffer lands in this NUMA node's local memory). Empty
        // until a multi-token expert is hit — the M=1 hot path never allocates.
        std::vector<float> scratch;
        int n_begin = 0, n_end = 0;
        row_partition(N, ith, nth, n_begin, n_end);
        if (n_begin < n_end) {
            grouped_gemm_rows(Du, Au, weights, expert_offsets, N, K, n_begin,
                              n_end, scratch, m1_path);
        }
        // No cross-stage barrier here: this is ONE projection GEMM. The
        // assemble layer rendezvouses between gate/up -> SwiGLU -> down.
    });
}

// ── Bit-compatible activation quantization (public) ─────────────────────────
void nvfp4_quantize_activation_row(const uint16_t* a_bf16, int K,
                                   float input_scale, float* a_deq_out) {
    assert((K % 16) == 0 && "K must be a multiple of the NVFP4 group size (16)");
    const float is = (input_scale > 0.0f) ? input_scale : 1.0f;
    for (int g0 = 0; g0 < K; g0 += 16) {
        float amax = 0.0f;
        for (int k = 0; k < 16; ++k)
            amax = std::max(amax, std::fabs(bf16_to_f32(a_bf16[g0 + k])));
        const float raw = amax / (6.0f * is);
        const float scale_repr =
            model::fp8_e4m3::decode(float_to_ue4m3(raw));   // E4M3-rounded group scale
        const float denom = scale_repr * is;
        for (int k = 0; k < 16; ++k) {
            const float x = bf16_to_f32(a_bf16[g0 + k]);
            if (denom > 0.0f) {
                const uint8_t nib = quantize_to_fp4_nibble(x / denom);
                a_deq_out[g0 + k] = kE2M1Table[nib] * scale_repr * is;
            } else {
                a_deq_out[g0 + k] = 0.0f;
            }
        }
    }
}

// ── Bit-compatible NVFP4 grouped GEMM (public) ──────────────────────────────
void cpu_nvfp4_grouped_gemm_actquant(void* D, const void* A,
                                     const CpuNvfp4ExpertWeights& weights,
                                     const float* input_scales,
                                     const int32_t* expert_offsets, int N, int K,
                                     int elem_size_bytes, NumaThreadPool* pool) {
    assert(elem_size_bytes == 2 && "CPU NVFP4 GEMM only supports BF16 (2 bytes)");
    assert((K % 16) == 0 && "K must be a multiple of the NVFP4 group size (16)");
    (void)elem_size_bytes;

    auto* Du = static_cast<uint16_t*>(D);
    const auto* Au = static_cast<const uint16_t*>(A);
    const int total_tokens = expert_offsets[weights.num_experts];
    if (total_tokens <= 0) return;

    // Pre-quantize every activation row ONCE (n-independent, reused across all N
    // output rows). Per-expert input_scale; the small [total_tokens, K] FP32
    // buffer is dwarfed by the weight decode.
    std::vector<float> a_deq(static_cast<size_t>(total_tokens) * K);
    for (int e = 0; e < weights.num_experts; ++e) {
        const float is = input_scales ? input_scales[e] : 1.0f;
        for (int m = expert_offsets[e]; m < expert_offsets[e + 1]; ++m)
            nvfp4_quantize_activation_row(Au + static_cast<size_t>(m) * K, K, is,
                                          a_deq.data() + static_cast<size_t>(m) * K);
    }

    if (pool == nullptr || pool->num_threads() <= 1) {
        std::vector<float> scratch;
        grouped_gemm_rows_actquant(Du, a_deq.data(), weights, expert_offsets, N, K,
                                   0, N, scratch);
        return;
    }
    pool->parallel_for([&](int ith, int nth, CpuBarrierState& /*barrier*/) {
        std::vector<float> scratch;
        int n_begin = 0, n_end = 0;
        row_partition(N, ith, nth, n_begin, n_end);
        if (n_begin < n_end)
            grouped_gemm_rows_actquant(Du, a_deq.data(), weights, expert_offsets, N,
                                       K, n_begin, n_end, scratch);
    });
}

// ── M=1 GEMV path selection ──────────────────────────────────────────────────
namespace {
// -1 = uninitialised, 0 = single-row, 1 = row-block. Defaults to row-block
// (the ik-technique) unless LS_CPU_GEMV_SINGLE_ROW is set in the environment.
std::atomic<int> g_m1_path{-1};

int read_m1_path_env() {
    const char* e = std::getenv("LS_CPU_GEMV_SINGLE_ROW");
    if (e && e[0] == '1') return 0;  // force original single-row path
    return 1;                        // default: ik-technique row-block
}
}  // namespace

M1GemvPath select_m1_gemv_path() {
    int v = g_m1_path.load(std::memory_order_relaxed);
    if (v < 0) {
        v = read_m1_path_env();
        g_m1_path.store(v, std::memory_order_relaxed);
    }
    return v == 0 ? M1GemvPath::kFusedSingleRow : M1GemvPath::kFusedRowBlock;
}

void set_m1_gemv_path_for_testing(M1GemvPath path) {
    g_m1_path.store(path == M1GemvPath::kFusedSingleRow ? 0 : 1,
                    std::memory_order_relaxed);
}

// ── Microbenchmark / attribution hooks ───────────────────────────────────────
void bench_decode_full_expert(const PackedProjection& proj, float* out) {
    const int64_t K = proj.K;
    const float ws2 = read_weight_scale_2(proj.base, proj.proj_bytes);
    for (int n = 0; n < proj.N; ++n)
        dequant_weight_row(proj, n, ws2, out + static_cast<size_t>(n) * K);
}

void bench_gemv_predecoded(const float* w_pre, const uint16_t* a, int N, int K,
                           float* out) {
    for (int n = 0; n < N; ++n)
        out[n] = dot_bf16_f32(a, w_pre + static_cast<size_t>(n) * K, K);
}

float bench_m1_v0_scalar_materialize(const PackedProjection& proj, int64_t n,
                                     float weight_scale_2, const uint16_t* a,
                                     float* row_scratch) {
    dequant_weight_row_scalar(proj, n, weight_scale_2, row_scratch);
    return dot_bf16_f32(a, row_scratch, static_cast<int>(proj.K));
}

float bench_m1_v1_vec_materialize(const PackedProjection& proj, int64_t n,
                                  float weight_scale_2, const uint16_t* a,
                                  float* row_scratch) {
    dequant_weight_row(proj, n, weight_scale_2, row_scratch);
    return dot_bf16_f32(a, row_scratch, static_cast<int>(proj.K));
}

float bench_m1_v2_fused(const PackedProjection& proj, int64_t n,
                        float weight_scale_2, const uint16_t* a) {
    return fused_decode_dot_row(proj, n, weight_scale_2, a);
}

}  // namespace layerstorm::compute::cpu
