// Copied verbatim from ik_llama.cpp (commit d47f484), MIT License.
// Source: https://github.com/ikawrakow/ik_llama.cpp  (see LICENSE.ik)
// Minimal extraction: the Q8_0/Q8_1/Q8_2 activation-quantization template
// (quantize_row_q8_1_x4_T) + its q8_1_x4 / q8_2_x4 wrappers, lifted byte-for-byte
// from ggml/src/iqk/iqk_quantize.cpp (lines ~1003-1178). We extract ONLY these
// functions rather than vendoring the whole 434 KB iqk_quantize.cpp (which pulls
// in every gemm-quant header) — that is the documented stub (see TECH_DEBT).
// The activation format the AVX-512-VNNI legacy Q8_0 GEMM consumes is Q8_2_X4
// (HAVE_FANCY_SIMD path in iqk_gemm_legacy_quants.cpp).
//
// iqk_common.h supplies hsum_i32_8 + the block_q8_* types + GGML_FP32_TO_*.
#include "iqk_common.h"

#ifdef IQK_IMPLEMENT

namespace {
template <typename Block, typename Block_x4>
void quantize_row_q8_1_x4_T(const float * x, Block * y, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    const int nb4 = 4*(nb/4);
    Block_x4 * y4 = (Block_x4 *)y;
#if defined(__aarch64__)
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }

        int32x4_t accv = vdupq_n_s32(0);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            if (i < nb4) {
                y4[i4].qs[QK8_1*ir + 4*j + 0] = vgetq_lane_s32(vi, 0);
                y4[i4].qs[QK8_1*ir + 4*j + 1] = vgetq_lane_s32(vi, 1);
                y4[i4].qs[QK8_1*ir + 4*j + 2] = vgetq_lane_s32(vi, 2);
                y4[i4].qs[QK8_1*ir + 4*j + 3] = vgetq_lane_s32(vi, 3);
            } else {
                y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
                y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
                y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
                y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
            }

            accv = vaddq_s32(accv, vi);
        }

        if constexpr (std::is_same_v<Block, block_q8_1>) {
            if (i < nb4) {
                y4[i4].d[ir+4] = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
            } else {
                y[i].s = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
            }
        } else {
            if (i < nb4) {
                y4[i4].d[ir+4] = GGML_FP32_TO_BF16(d * vaddvq_s32(accv)).bits;
            } else {
                y[i].s = GGML_FP32_TO_BF16(d * vaddvq_s32(accv)).bits;
            }
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float max_scalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        float d = max_scalar / 127.f;
        if constexpr (std::is_same_v<Block, block_q8_1>) {
            if (i < nb4) {
                y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
            } else {
                y[i].d = GGML_FP32_TO_FP16(d);
            }
        } else {
            auto t = GGML_FP32_TO_BF16(d);
            d = ggml_bf16_to_fp32(t);
            if (i < nb4) {
                y4[i4].d[ir] = t.bits;
            } else {
                y[i].d = t.bits;
            }
        }
        const float id = d > 0 ? 1/d : 0.f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Compute the sum of the quants and set y[i].s
        int isum = hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
        if constexpr (std::is_same_v<Block, block_q8_1>) {
            if (i < nb4) {
                y4[i4].d[ir+4] = GGML_FP32_TO_FP16(d * isum);
            } else {
                y[i].s = GGML_FP32_TO_FP16(d * isum);
            }
        } else {
            if (i < nb4) {
                auto i16 = (int16_t *)y4[i4].d;
                i16[ir+4] = isum;
            } else {
                auto i16 = (int16_t *)&y[i].s;
                i16[0] = isum;
            }
        }

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *)y4[i4].qs + ir, i0);
        } else {
            _mm256_storeu_si256((__m256i *)y[i].qs, i0);
        }
    }
#endif
}
}

void quantize_row_q8_1_x4(const float * x, void * vy, int64_t k) {
    quantize_row_q8_1_x4_T<block_q8_1, block_q8_1_x4>(x, (block_q8_1 *)vy, k);
}

void quantize_row_q8_2_x4(const float * x, void * vy, int64_t k) {
    quantize_row_q8_1_x4_T<block_q8_2, block_q8_2_x4>(x, (block_q8_2 *)vy, k);
}


#endif  // IQK_IMPLEMENT
