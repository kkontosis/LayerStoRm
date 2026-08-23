// K-grouped weight requantization unit tests (TD-DSPARK-DRAFT-QUANT).
// CPU-only: scalar encoder contracts, packer round-trip error bounds vs the
// BF16 source, ragged-group/tail handling, determinism.

#include "model/quantization/kgroup_quant.h"

#include "model/quantization/fp8.h"
#include "model/quantization/nvfp4.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace kg = layerstorm::model::kgroup;
namespace fp8 = layerstorm::model::fp8_e4m3;
namespace nvfp4 = layerstorm::model::nvfp4;

namespace {

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + rounding) >> 16);
}

float bf2f(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::vector<uint16_t> random_bf16(int64_t n, uint32_t seed, float lo,
                                  float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<uint16_t> v(static_cast<size_t>(n));
    for (auto& x : v) x = f2bf(u(rng));
    return v;
}

}  // namespace

// ── Scalar encoder contracts ─────────────────────────────────────────────────

TEST(KgroupQuant, Fp8EncodeGridExactAndSaturating) {
    // Every finite E4M3 code must round-trip encode(decode(b)) == b
    // (canonical zero: -0 decodes to -0.0f which encodes back to 0x80).
    for (int b = 0; b < 256; ++b) {
        const float v = fp8::decode(static_cast<uint8_t>(b));
        if (std::isnan(v)) continue;  // 0x7F / 0xFF
        EXPECT_EQ(kg::encode_fp8_e4m3(v), static_cast<uint8_t>(b))
            << "code " << b << " value " << v;
    }
    // Saturation + specials.
    EXPECT_EQ(kg::encode_fp8_e4m3(1e6f), 0x7E);    // +448
    EXPECT_EQ(kg::encode_fp8_e4m3(-1e6f), 0xFE);   // -448
    EXPECT_EQ(kg::encode_fp8_e4m3(0.0f), 0x00);
    EXPECT_EQ(kg::encode_fp8_e4m3(NAN), 0x7F);
    // Round-to-nearest-even at a midpoint: between 1.0 (0x38) and 1.125
    // (0x39) the midpoint 1.0625 goes to the even mantissa (0x38).
    EXPECT_EQ(kg::encode_fp8_e4m3(1.0625f), 0x38);
    EXPECT_EQ(kg::encode_fp8_e4m3(1.07f), 0x39);
}

TEST(KgroupQuant, E2m1EncodeGridExactAndTies) {
    for (int nib = 0; nib < 16; ++nib) {
        const float v = nvfp4::decode_e2m1(static_cast<uint8_t>(nib));
        if (v == 0.0f && (nib & 8)) continue;  // -0 encodes as +0
        EXPECT_EQ(kg::encode_e2m1(v), static_cast<uint8_t>(nib))
            << "nibble " << nib;
    }
    EXPECT_EQ(kg::encode_e2m1(100.0f), 0x07);   // saturate to 6
    EXPECT_EQ(kg::encode_e2m1(-100.0f), 0x0F);  // saturate to -6
    // Ties to the even-mantissa neighbor.
    EXPECT_EQ(kg::encode_e2m1(0.25f), 0x00);  // 0 vs 0.5 -> 0
    EXPECT_EQ(kg::encode_e2m1(0.75f), 0x02);  // 0.5 vs 1 -> 1.0
    EXPECT_EQ(kg::encode_e2m1(2.5f), 0x04);   // 2 vs 3 -> 2
    EXPECT_EQ(kg::encode_e2m1(3.5f), 0x06);   // 3 vs 4 -> 4
    EXPECT_EQ(kg::encode_e2m1(5.0f), 0x06);   // 4 vs 6 -> 4
}

// ── Packer round-trip error bounds ──────────────────────────────────────────

TEST(KgroupQuant, Fp8RoundTripWithinFormatBand) {
    const int64_t n = 7, k = 300;  // ragged: groups 128/128/44
    auto src = random_bf16(n * k, 42, -2.0f, 2.0f);
    std::vector<uint8_t> q(static_cast<size_t>(kg::fp8_weight_bytes(n, k)));
    std::vector<float> s(
        static_cast<size_t>(kg::fp8_scale_bytes(n, k)) / sizeof(float));
    kg::quantize_rows_fp8_e4m3(src.data(), n, k, q.data(), s.data());
    std::vector<float> out(static_cast<size_t>(n * k));
    kg::dequantize_rows_fp8_e4m3(q.data(), s.data(), n, k, out.data());

    const int64_t groups = (k + kg::kFp8GroupSize - 1) / kg::kFp8GroupSize;
    for (int64_t r = 0; r < n; ++r)
        for (int64_t g = 0; g < groups; ++g) {
            const int64_t k0 = g * kg::kFp8GroupSize;
            const int64_t k1 = std::min(k, k0 + kg::kFp8GroupSize);
            float amax = 0.0f;
            for (int64_t j = k0; j < k1; ++j)
                amax = std::max(amax,
                                std::fabs(bf2f(src[static_cast<size_t>(
                                    r * k + j)])));
            for (int64_t j = k0; j < k1; ++j) {
                const float x = bf2f(src[static_cast<size_t>(r * k + j)]);
                const float e = std::fabs(out[static_cast<size_t>(r * k + j)]
                                          - x);
                // E4M3 with amax/448 scaling: worst grid gap near the top is
                // 32 scaled units -> |err| <= amax * 16/448 = amax/28.
                EXPECT_LE(e, amax / 28.0f + 1e-7f)
                    << "row " << r << " col " << j << " x=" << x;
            }
        }
}

TEST(KgroupQuant, Nvfp4RoundTripWithinFormatBand) {
    const int64_t n = 5, k = 50;  // ragged: groups 16/16/16/2, odd tail byte
    auto src = random_bf16(n * k, 7, -1.5f, 1.5f);
    std::vector<uint8_t> q(static_cast<size_t>(kg::nvfp4_weight_bytes(n, k)));
    std::vector<uint8_t> s(static_cast<size_t>(kg::nvfp4_scale_bytes(n, k)));
    kg::quantize_rows_nvfp4(src.data(), n, k, q.data(), s.data());
    std::vector<float> out(static_cast<size_t>(n * k));
    kg::dequantize_rows_nvfp4(q.data(), s.data(), n, k, out.data());

    const int64_t groups =
        (k + kg::kNvfp4GroupSize - 1) / kg::kNvfp4GroupSize;
    for (int64_t r = 0; r < n; ++r)
        for (int64_t g = 0; g < groups; ++g) {
            const int64_t k0 = g * kg::kNvfp4GroupSize;
            const int64_t k1 = std::min(k, k0 + kg::kNvfp4GroupSize);
            float amax = 0.0f;
            for (int64_t j = k0; j < k1; ++j)
                amax = std::max(amax,
                                std::fabs(bf2f(src[static_cast<size_t>(
                                    r * k + j)])));
            // Power-of-two scale with amax/scale <= 6: scale <= amax/3
            // (ceil(log2) overshoot < 2x), worst half-gap on the E2M1 grid
            // is 1.0 scaled unit -> |err| <= amax/3.
            for (int64_t j = k0; j < k1; ++j) {
                const float x = bf2f(src[static_cast<size_t>(r * k + j)]);
                const float e = std::fabs(out[static_cast<size_t>(r * k + j)]
                                          - x);
                EXPECT_LE(e, amax / 3.0f + 1e-7f)
                    << "row " << r << " col " << j << " x=" << x;
            }
        }
}

TEST(KgroupQuant, AllZeroGroupsAndDeterminism) {
    const int64_t n = 3, k = 160;
    std::vector<uint16_t> src(static_cast<size_t>(n * k), 0);  // all +0
    std::vector<uint8_t> q(static_cast<size_t>(kg::fp8_weight_bytes(n, k)));
    std::vector<float> s(
        static_cast<size_t>(kg::fp8_scale_bytes(n, k)) / sizeof(float));
    kg::quantize_rows_fp8_e4m3(src.data(), n, k, q.data(), s.data());
    for (float sc : s) EXPECT_EQ(sc, 1.0f);   // all-zero group -> scale 1
    for (uint8_t b : q) EXPECT_EQ(b, 0x00);

    std::vector<uint8_t> q4(static_cast<size_t>(kg::nvfp4_weight_bytes(n, k)));
    std::vector<uint8_t> s4(static_cast<size_t>(kg::nvfp4_scale_bytes(n, k)));
    kg::quantize_rows_nvfp4(src.data(), n, k, q4.data(), s4.data());
    for (uint8_t sc : s4) EXPECT_EQ(sc, 127);  // UE8M0 1.0
    for (uint8_t b : q4) EXPECT_EQ(b, 0x00);

    // Determinism across runs (multi-threaded packers must partition, not
    // race).
    auto rnd = random_bf16(n * k, 3, -1.0f, 1.0f);
    std::vector<uint8_t> qa(q.size()), qb(q.size());
    std::vector<float> sa(s.size()), sb(s.size());
    kg::quantize_rows_fp8_e4m3(rnd.data(), n, k, qa.data(), sa.data());
    kg::quantize_rows_fp8_e4m3(rnd.data(), n, k, qb.data(), sb.data());
    EXPECT_EQ(qa, qb);
    EXPECT_EQ(sa, sb);
}

// ── UE8M0 fit guarantee: every quantized element must be on the grid ────────

TEST(KgroupQuant, Nvfp4ScaleAlwaysCoversGroupMax) {
    // Exact powers of two and values straddling 6*2^e exercise the
    // ceil(log2) fit guard.
    std::vector<float> vals = {6.0f, 6.0001f, 12.0f, 0.75f, 3.0f, 1e-4f,
                               96.0f, 0.046875f};
    for (float v : vals) {
        const int64_t k = 16;
        std::vector<uint16_t> src(16, f2bf(v));
        std::vector<uint8_t> q(8), s(1);
        kg::quantize_rows_nvfp4(src.data(), 1, k, q.data(), s.data());
        const float scale = nvfp4::decode_ue8m0(s[0]);
        ASSERT_GT(scale, 0.0f);
        EXPECT_LE(bf2f(f2bf(v)) / scale, nvfp4::kE2M1Max * 1.0001f)
            << "v=" << v;
    }
}
