// K-grouped weight requantization (TD-DSPARK-DRAFT-QUANT). See header.

#include "model/quantization/kgroup_quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#include "model/quantization/fp8.h"
#include "model/quantization/nvfp4.h"

namespace layerstorm::model::kgroup {

namespace {

float bf16_to_float(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

/// Run fn(row_begin, row_end) across hardware threads (init-path only).
template <typename Fn>
void parallel_rows(int64_t n, Fn&& fn) {
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int64_t threads =
        std::min<int64_t>(n, static_cast<int64_t>(std::min(hw, 32u)));
    if (threads <= 1) {
        fn(int64_t{0}, n);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads));
    const int64_t chunk = (n + threads - 1) / threads;
    for (int64_t t = 0; t < threads; ++t) {
        const int64_t b = t * chunk;
        const int64_t e = std::min(n, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&fn, b, e] { fn(b, e); });
    }
    for (auto& th : pool) th.join();
}

}  // namespace

// ── Scalar encoders ─────────────────────────────────────────────────────────

uint8_t encode_fp8_e4m3(float x) {
    if (std::isnan(x)) return 0x7F;
    const uint8_t sign = std::signbit(x) ? 0x80 : 0x00;
    float a = std::fabs(x);
    if (a == 0.0f) return sign;
    if (a > fp8_e4m3::kMaxFinite) a = fp8_e4m3::kMaxFinite;  // saturate

    int e = 0;
    (void)std::frexp(a, &e);  // a = m * 2^e, m in [0.5, 1)
    int e2 = e - 1;           // a = (2m) * 2^e2, 2m in [1, 2)

    if (e2 < -6) {
        // Subnormal grid: value = f/8 * 2^-6, f in [0, 7].
        const float f = a * std::ldexp(1.0f, 9);  // a / 2^-9
        const int fi = static_cast<int>(std::nearbyint(f));  // RNE
        if (fi >= 8) return sign | 0x08;  // rounds up into first normal
        return sign | static_cast<uint8_t>(fi);
    }
    const float frac = std::ldexp(a, -e2) - 1.0f;  // [0, 1)
    int fi = static_cast<int>(std::nearbyint(frac * 8.0f));  // RNE
    if (fi == 8) {
        fi = 0;
        ++e2;
    }
    if (e2 > 8) return sign | 0x7E;  // beyond max finite -> saturate to 448
    return sign |
           static_cast<uint8_t>(((e2 + fp8_e4m3::kBias) << 3) | fi);
}

uint8_t encode_e2m1(float x) {
    const uint8_t sign = std::signbit(x) ? 0x08 : 0x00;
    const float a = std::fabs(x);
    // Grid {0, .5, 1, 1.5, 2, 3, 4, 6}; midpoints with ties-to-even-mantissa
    // (indices 0,2,4,6 have mantissa bit 0): the strict/non-strict choice at
    // each midpoint sends the tie to the even neighbor.
    uint8_t idx;
    if (a <= 0.25f) idx = 0;        // tie 0.25 -> 0 (even)
    else if (a < 0.75f) idx = 1;    // tie 0.75 -> 2 (even)
    else if (a <= 1.25f) idx = 2;   // tie 1.25 -> 2 (even)
    else if (a < 1.75f) idx = 3;    // tie 1.75 -> 4 (even)
    else if (a <= 2.5f) idx = 4;    // tie 2.5 -> 4 (even)
    else if (a < 3.5f) idx = 5;     // tie 3.5 -> 6 (even)
    else if (a <= 5.0f) idx = 6;    // tie 5.0 -> 6 (even)
    else idx = 7;                   // saturates at 6
    return sign | idx;
}

// ── Row packers ─────────────────────────────────────────────────────────────

void quantize_rows_fp8_e4m3(const uint16_t* bf16, int64_t n, int64_t k,
                            uint8_t* q, float* scales) {
    const int64_t groups = (k + kFp8GroupSize - 1) / kFp8GroupSize;
    parallel_rows(n, [&](int64_t rb, int64_t re) {
        for (int64_t r = rb; r < re; ++r) {
            const uint16_t* src = bf16 + r * k;
            uint8_t* dst = q + r * k;
            float* srow = scales + r * groups;
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t k0 = g * kFp8GroupSize;
                const int64_t k1 = std::min(k, k0 + kFp8GroupSize);
                float amax = 0.0f;
                for (int64_t j = k0; j < k1; ++j)
                    amax = std::max(amax, std::fabs(bf16_to_float(src[j])));
                const float scale =
                    amax > 0.0f ? amax / fp8_e4m3::kMaxFinite : 1.0f;
                srow[g] = scale;
                const float inv = 1.0f / scale;
                for (int64_t j = k0; j < k1; ++j)
                    dst[j] = encode_fp8_e4m3(bf16_to_float(src[j]) * inv);
            }
        }
    });
}

void quantize_rows_nvfp4(const uint16_t* bf16, int64_t n, int64_t k,
                         uint8_t* q, uint8_t* scales) {
    const int64_t groups = (k + kNvfp4GroupSize - 1) / kNvfp4GroupSize;
    const int64_t row_bytes = (k + 1) / 2;
    parallel_rows(n, [&](int64_t rb, int64_t re) {
        for (int64_t r = rb; r < re; ++r) {
            const uint16_t* src = bf16 + r * k;
            uint8_t* dst = q + r * row_bytes;
            uint8_t* srow = scales + r * groups;
            std::memset(dst, 0, static_cast<size_t>(row_bytes));
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t k0 = g * kNvfp4GroupSize;
                const int64_t k1 = std::min(k, k0 + kNvfp4GroupSize);
                float amax = 0.0f;
                for (int64_t j = k0; j < k1; ++j)
                    amax = std::max(amax, std::fabs(bf16_to_float(src[j])));
                // Power-of-two UE8M0 scale: smallest 2^e with amax/2^e <= 6.
                int e = 0;
                if (amax > 0.0f) {
                    e = static_cast<int>(
                        std::ceil(std::log2(amax / nvfp4::kE2M1Max)));
                    // Guard fp rounding at exact powers: nudge until it fits.
                    while (amax * std::ldexp(1.0f, -e) > nvfp4::kE2M1Max) ++e;
                }
                // Upper clamp 252 (not 254): 6 * 2^(sb-127) must stay finite
                // in float (sb 253 -> 6 * 2^126 > FLT_MAX). Near-FLT_MAX
                // groups then saturate on the grid; the error stays inside
                // the amax/3 format band.
                const int sb = std::clamp(e + 127, 1, 252);
                srow[g] = static_cast<uint8_t>(sb);
                const float inv = std::ldexp(1.0f, -(sb - 127));
                for (int64_t j = k0; j < k1; ++j) {
                    const uint8_t nib =
                        encode_e2m1(bf16_to_float(src[j]) * inv);
                    dst[j >> 1] |= (j & 1) ? static_cast<uint8_t>(nib << 4)
                                           : nib;
                }
            }
        }
    });
}

// ── CPU dequant references ──────────────────────────────────────────────────

void dequantize_rows_fp8_e4m3(const uint8_t* q, const float* scales,
                              int64_t n, int64_t k, float* out) {
    const int64_t groups = (k + kFp8GroupSize - 1) / kFp8GroupSize;
    for (int64_t r = 0; r < n; ++r)
        for (int64_t j = 0; j < k; ++j)
            out[r * k + j] = fp8_e4m3::decode(q[r * k + j]) *
                             scales[r * groups + j / kFp8GroupSize];
}

void dequantize_rows_nvfp4(const uint8_t* q, const uint8_t* scales,
                           int64_t n, int64_t k, float* out) {
    const int64_t groups = (k + kNvfp4GroupSize - 1) / kNvfp4GroupSize;
    const int64_t row_bytes = (k + 1) / 2;
    for (int64_t r = 0; r < n; ++r)
        for (int64_t j = 0; j < k; ++j) {
            const uint8_t byte = q[r * row_bytes + (j >> 1)];
            const uint8_t nib = (j & 1) ? (byte >> 4) : (byte & 0x0F);
            out[r * k + j] =
                nvfp4::decode_e2m1(nib) *
                nvfp4::decode_ue8m0(scales[r * groups + j / kNvfp4GroupSize]);
        }
}

}  // namespace layerstorm::model::kgroup
