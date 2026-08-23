#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "model/quantization/fp8.h"
#include "model/quantization/registry.h"

using namespace layerstorm::model;
using namespace layerstorm;

// ── Fp8E4M3: QuantInterface metadata ────────────────────────────────────────

TEST(Fp8E4M3, Methods) {
    Fp8E4M3 q;
    EXPECT_EQ(q.name(), "fp8_e4m3");
    EXPECT_EQ(q.weight_quant(), config::WeightQuant::fp8_e4m3);
    EXPECT_TRUE(q.tensor_core_eligible());
    EXPECT_DOUBLE_EQ(q.dequant_flops_per_element(), 1.0);
    EXPECT_DOUBLE_EQ(q.bytes_per_element(), 1.0 + 4.0 / (128.0 * 128.0));
}

TEST(Fp8E4M3, MemoryLayout) {
    Fp8E4M3 q;
    auto layout = q.memory_layout();
    EXPECT_EQ(layout.bits_per_weight, 8);
    EXPECT_EQ(layout.group_size, 0);
    EXPECT_FALSE(layout.has_zero_point);
    EXPECT_EQ(layout.scale_dtype, ScaleDtype::none);
    EXPECT_EQ(layout.alignment_bytes, 64);
}

// ── Fp8E5M2: QuantInterface metadata ────────────────────────────────────────

TEST(Fp8E5M2, Methods) {
    Fp8E5M2 q;
    EXPECT_EQ(q.name(), "fp8_e5m2");
    EXPECT_EQ(q.weight_quant(), config::WeightQuant::fp8_e5m2);
    EXPECT_TRUE(q.tensor_core_eligible());
    EXPECT_DOUBLE_EQ(q.dequant_flops_per_element(), 1.0);
    EXPECT_DOUBLE_EQ(q.bytes_per_element(), 1.0 + 4.0 / (128.0 * 128.0));
}

TEST(Fp8E5M2, MemoryLayout) {
    Fp8E5M2 q;
    auto layout = q.memory_layout();
    EXPECT_EQ(layout.bits_per_weight, 8);
    EXPECT_EQ(layout.group_size, 0);
    EXPECT_FALSE(layout.has_zero_point);
    EXPECT_EQ(layout.scale_dtype, ScaleDtype::none);
    EXPECT_EQ(layout.alignment_bytes, 64);
}

// ── Size calculations ───────────────────────────────────────────────────────

TEST(Fp8E4M3, BytesPerProjectionV32) {
    Fp8E4M3 q;
    ExpertShape shape{7168, 2048};
    // weight: 7168*2048 = 14,680,064. scale: ceil(2048/128)*ceil(7168/128)*4 = 16*56*4 = 3,584
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::gate), 14'683'648);
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::up),   14'683'648);
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::down), 14'683'648);
}

TEST(Fp8E4M3, WeightBytesPerProjectionV32) {
    Fp8E4M3 q;
    ExpertShape shape{7168, 2048};
    // weight_bytes = params only (no scales)
    EXPECT_EQ(q.weight_bytes_per_projection(shape, Projection::gate), 14'680'064);
    EXPECT_EQ(q.weight_bytes_per_projection(shape, Projection::up),   14'680'064);
    EXPECT_EQ(q.weight_bytes_per_projection(shape, Projection::down), 14'680'064);
}

TEST(Fp8E4M3, BlockwiseScaleBytesV32) {
    Fp8E4M3 q;
    ExpertShape shape{7168, 2048};
    // Scale = bytes_per_projection - weight_bytes_per_projection
    int64_t scale = q.bytes_per_projection(shape, Projection::gate)
                  - q.weight_bytes_per_projection(shape, Projection::gate);
    EXPECT_EQ(scale, 3584);  // 16 * 56 * 4
}

TEST(Fp8E4M3, BytesPerExpertV32) {
    Fp8E4M3 q;
    ExpertShape shape{7168, 2048};
    EXPECT_EQ(q.bytes_per_expert(shape), 3 * 14'683'648);  // 44,050,944
}

TEST(Fp8E4M3, BytesPerExpertConsistency) {
    Fp8E4M3 q;
    ExpertShape shape{7168, 2048};
    int64_t sum = q.bytes_per_projection(shape, Projection::gate)
                + q.bytes_per_projection(shape, Projection::up)
                + q.bytes_per_projection(shape, Projection::down);
    EXPECT_EQ(q.bytes_per_expert(shape), sum);
}

TEST(Fp8E4M3, BytesPerElementConsistency) {
    Fp8E4M3 q;
    ExpertShape shape{7168, 2048};
    double ratio = static_cast<double>(q.bytes_per_expert(shape)) / shape.total_params();
    EXPECT_NEAR(ratio, q.bytes_per_element(), 1e-6);
}

TEST(Fp8E4M3, BytesPerProjectionGlm5) {
    Fp8E4M3 q;
    ExpertShape shape{6144, 2048};
    // weight: 6144*2048 = 12,582,912. scale: 16*48*4 = 3,072
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::gate), 12'585'984);
    EXPECT_EQ(q.bytes_per_expert(shape), 3 * 12'585'984);
}

TEST(Fp8E4M3, BytesPerProjectionNonAligned) {
    Fp8E4M3 q;
    ExpertShape shape{100, 17};
    // weight: 100*17 = 1700. scale: ceil(17/128)*ceil(100/128)*4 = 1*1*4 = 4
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::gate), 1704);
    EXPECT_EQ(q.bytes_per_expert(shape), 3 * 1704);
}

TEST(Fp8E5M2, BytesPerProjectionV32) {
    Fp8E5M2 q;
    ExpertShape shape{7168, 2048};
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::gate), 14'683'648);
}

TEST(Fp8E5M2, BytesPerExpertV32) {
    Fp8E5M2 q;
    ExpertShape shape{7168, 2048};
    EXPECT_EQ(q.bytes_per_expert(shape), 3 * 14'683'648);
}

// ── E4M3 decode: zero ───────────────────────────────────────────────────────

TEST(Fp8E4M3, DecodeZero) {
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x00), 0.0f);
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x80), 0.0f);
    // Negative zero
    EXPECT_TRUE(std::signbit(fp8_e4m3::decode(0x80)));
    EXPECT_FALSE(std::signbit(fp8_e4m3::decode(0x00)));
}

// ── E4M3 decode: NaN ────────────────────────────────────────────────────────

TEST(Fp8E4M3, DecodeNaN) {
    // Only 0x7F (exp=15, mant=7) and 0xFF are NaN in E4M3fn
    EXPECT_TRUE(std::isnan(fp8_e4m3::decode(0x7F)));
    EXPECT_TRUE(std::isnan(fp8_e4m3::decode(0xFF)));

    // 0x78 (exp=15, mant=0) is a valid normal, NOT inf or NaN
    EXPECT_FALSE(std::isnan(fp8_e4m3::decode(0x78)));
    EXPECT_TRUE(std::isfinite(fp8_e4m3::decode(0x78)));
}

// ── E4M3 decode: normals ────────────────────────────────────────────────────

TEST(Fp8E4M3, DecodeNormals) {
    // 0x38: sign=0, exp=0111(7), mant=000 → 2^(7-7) * 1.0 = 1.0
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x38), 1.0f);

    // 0x39: sign=0, exp=7, mant=001 → 2^0 * (1 + 1/8) = 1.125
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x39), 1.125f);

    // 0x3C: sign=0, exp=7, mant=100 → 2^0 * (1 + 4/8) = 1.5
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x3C), 1.5f);

    // 0x40: sign=0, exp=1000(8), mant=000 → 2^(8-7) * 1.0 = 2.0
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x40), 2.0f);

    // 0x7E: sign=0, exp=1111(15), mant=110(6) → 2^8 * (1 + 6/8) = 256 * 1.75 = 448.0
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x7E), 448.0f);
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x7E), fp8_e4m3::kMaxFinite);

    // 0xB8: sign=1, exp=7, mant=0 → -1.0
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0xB8), -1.0f);

    // 0xFE: sign=1, exp=15, mant=6 → -448.0
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0xFE), -448.0f);

    // Min positive normal: 0x08 (exp=1, mant=0) → 2^(1-7) = 2^(-6) = 0.015625
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x08), 0.015625f);
}

// ── E4M3 decode: subnormals ─────────────────────────────────────────────────

TEST(Fp8E4M3, DecodeSubnormals) {
    // 0x01: exp=0, mant=1 → 2^(-6) * (1/8) = 2^(-9) = 0.001953125
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x01), std::ldexp(1.0f, -9));

    // 0x03: exp=0, mant=3 → 3 * 2^(-9)
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x03), 3.0f * std::ldexp(1.0f, -9));

    // 0x07: exp=0, mant=7 → 7 * 2^(-9) (max subnormal)
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x07), 7.0f * std::ldexp(1.0f, -9));

    // 0x87: negative subnormal → -7 * 2^(-9)
    EXPECT_FLOAT_EQ(fp8_e4m3::decode(0x87), -7.0f * std::ldexp(1.0f, -9));
}

// ── E4M3 decode: exhaustive special value check ─────────────────────────────

TEST(Fp8E4M3, DecodeExhaustiveSpecials) {
    int nan_count = 0;
    int inf_count = 0;
    for (int b = 0; b < 256; ++b) {
        float v = fp8_e4m3::decode(static_cast<uint8_t>(b));
        if (std::isnan(v)) ++nan_count;
        if (std::isinf(v)) ++inf_count;
    }
    EXPECT_EQ(nan_count, 2);   // only 0x7F and 0xFF
    EXPECT_EQ(inf_count, 0);   // E4M3fn has no infinity
}

// ── E5M2 decode: zero ───────────────────────────────────────────────────────

TEST(Fp8E5M2, DecodeZero) {
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x00), 0.0f);
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x80), 0.0f);
    EXPECT_TRUE(std::signbit(fp8_e5m2::decode(0x80)));
    EXPECT_FALSE(std::signbit(fp8_e5m2::decode(0x00)));
}

// ── E5M2 decode: infinity ───────────────────────────────────────────────────

TEST(Fp8E5M2, DecodeInfinity) {
    // 0x7C: exp=31, mant=0 → +inf
    EXPECT_TRUE(std::isinf(fp8_e5m2::decode(0x7C)));
    EXPECT_GT(fp8_e5m2::decode(0x7C), 0.0f);

    // 0xFC: sign=1, exp=31, mant=0 → -inf
    EXPECT_TRUE(std::isinf(fp8_e5m2::decode(0xFC)));
    EXPECT_LT(fp8_e5m2::decode(0xFC), 0.0f);
}

// ── E5M2 decode: NaN ────────────────────────────────────────────────────────

TEST(Fp8E5M2, DecodeNaN) {
    // exp=31, mant!=0 → NaN
    EXPECT_TRUE(std::isnan(fp8_e5m2::decode(0x7D)));  // mant=1
    EXPECT_TRUE(std::isnan(fp8_e5m2::decode(0x7E)));  // mant=2
    EXPECT_TRUE(std::isnan(fp8_e5m2::decode(0x7F)));  // mant=3
    EXPECT_TRUE(std::isnan(fp8_e5m2::decode(0xFD)));  // negative NaN
    EXPECT_TRUE(std::isnan(fp8_e5m2::decode(0xFE)));
    EXPECT_TRUE(std::isnan(fp8_e5m2::decode(0xFF)));
}

// ── E5M2 decode: normals ────────────────────────────────────────────────────

TEST(Fp8E5M2, DecodeNormals) {
    // 0x3C: sign=0, exp=01111(15), mant=00 → 2^(15-15) * 1.0 = 1.0
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x3C), 1.0f);

    // 0x3D: exp=15, mant=01 → 2^0 * (1 + 1/4) = 1.25
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x3D), 1.25f);

    // 0x3E: exp=15, mant=10 → 2^0 * (1 + 2/4) = 1.5
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x3E), 1.5f);

    // 0x3F: exp=15, mant=11 → 2^0 * (1 + 3/4) = 1.75
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x3F), 1.75f);

    // 0x40: sign=0, exp=10000(16), mant=00 → 2^(16-15) * 1.0 = 2.0
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x40), 2.0f);

    // 0x7B: exp=11110(30), mant=11(3) → 2^15 * (1 + 3/4) = 32768 * 1.75 = 57344.0
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x7B), 57344.0f);
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x7B), fp8_e5m2::kMaxFinite);

    // 0xBC: sign=1, exp=15, mant=0 → -1.0
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0xBC), -1.0f);

    // Min positive normal: 0x04 (exp=1, mant=0) → 2^(1-15) = 2^(-14)
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x04), std::ldexp(1.0f, -14));
}

// ── E5M2 decode: subnormals ─────────────────────────────────────────────────

TEST(Fp8E5M2, DecodeSubnormals) {
    // 0x01: exp=0, mant=1 → 2^(-14) * (1/4) = 2^(-16)
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x01), std::ldexp(1.0f, -16));

    // 0x02: exp=0, mant=2 → 2^(-14) * (2/4) = 2^(-15)
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x02), std::ldexp(1.0f, -15));

    // 0x03: exp=0, mant=3 → 3 * 2^(-16) (max subnormal)
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x03), 3.0f * std::ldexp(1.0f, -16));

    // 0x83: negative subnormal
    EXPECT_FLOAT_EQ(fp8_e5m2::decode(0x83), -3.0f * std::ldexp(1.0f, -16));
}

// ── E5M2 decode: exhaustive special value check ─────────────────────────────

TEST(Fp8E5M2, DecodeExhaustiveSpecials) {
    int nan_count = 0;
    int inf_count = 0;
    for (int b = 0; b < 256; ++b) {
        float v = fp8_e5m2::decode(static_cast<uint8_t>(b));
        if (std::isnan(v)) ++nan_count;
        if (std::isinf(v)) ++inf_count;
    }
    EXPECT_EQ(nan_count, 6);   // 0x7D,0x7E,0x7F,0xFD,0xFE,0xFF
    EXPECT_EQ(inf_count, 2);   // 0x7C,0xFC
}

// ── Dequantize ──────────────────────────────────────────────────────────────

TEST(Fp8E4M3, DequantizeBasic) {
    std::vector<uint8_t> packed = {0x38, 0x40, 0x00};  // 1.0, 2.0, 0.0
    std::vector<float> out(3);

    fp8_e4m3::dequantize(packed.data(), out.data(), 3);

    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
}

TEST(Fp8E4M3, DequantizeAllValues) {
    // One of each byte value; verify dequantize matches decode
    std::vector<uint8_t> packed(256);
    for (int i = 0; i < 256; ++i) packed[i] = static_cast<uint8_t>(i);

    std::vector<float> out(256);
    fp8_e4m3::dequantize(packed.data(), out.data(), 256);

    for (int i = 0; i < 256; ++i) {
        float expected = fp8_e4m3::decode(static_cast<uint8_t>(i));
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(out[i])) << "byte " << i;
        } else {
            EXPECT_FLOAT_EQ(out[i], expected) << "byte " << i;
        }
    }
}

TEST(Fp8E5M2, DequantizeBasic) {
    std::vector<uint8_t> packed = {0x3C, 0x40, 0x00};  // 1.0, 2.0, 0.0
    std::vector<float> out(3);

    fp8_e5m2::dequantize(packed.data(), out.data(), 3);

    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
}

TEST(Fp8E5M2, DequantizeAllValues) {
    std::vector<uint8_t> packed(256);
    for (int i = 0; i < 256; ++i) packed[i] = static_cast<uint8_t>(i);

    std::vector<float> out(256);
    fp8_e5m2::dequantize(packed.data(), out.data(), 256);

    for (int i = 0; i < 256; ++i) {
        float expected = fp8_e5m2::decode(static_cast<uint8_t>(i));
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(out[i])) << "byte " << i;
        } else if (std::isinf(expected)) {
            EXPECT_EQ(out[i], expected) << "byte " << i;
        } else {
            EXPECT_FLOAT_EQ(out[i], expected) << "byte " << i;
        }
    }
}

// ── Registry integration ────────────────────────────────────────────────────

TEST(Fp8E4M3, ManualRegistration) {
    clear_registry();
    Fp8E4M3 instance;
    register_format(&instance);

    EXPECT_EQ(find_format("fp8_e4m3"), &instance);
    EXPECT_EQ(find_format(config::WeightQuant::fp8_e4m3), &instance);

    auto& ref = get_format("fp8_e4m3");
    EXPECT_EQ(ref.name(), "fp8_e4m3");
    EXPECT_DOUBLE_EQ(ref.bytes_per_element(), 1.0 + 4.0 / (128.0 * 128.0));

    clear_registry();
}

TEST(Fp8E5M2, ManualRegistration) {
    clear_registry();
    Fp8E5M2 instance;
    register_format(&instance);

    EXPECT_EQ(find_format("fp8_e5m2"), &instance);
    EXPECT_EQ(find_format(config::WeightQuant::fp8_e5m2), &instance);

    auto& ref = get_format("fp8_e5m2");
    EXPECT_EQ(ref.name(), "fp8_e5m2");
    EXPECT_DOUBLE_EQ(ref.bytes_per_element(), 1.0 + 4.0 / (128.0 * 128.0));

    clear_registry();
}

TEST(Fp8, BothFormatsCoexist) {
    clear_registry();
    Fp8E4M3 e4m3;
    Fp8E5M2 e5m2;
    register_format(&e4m3);
    register_format(&e5m2);

    EXPECT_EQ(find_format("fp8_e4m3"), &e4m3);
    EXPECT_EQ(find_format("fp8_e5m2"), &e5m2);
    EXPECT_NE(find_format("fp8_e4m3"), find_format("fp8_e5m2"));

    auto names = registered_names();
    EXPECT_EQ(names.size(), 2u);

    clear_registry();
}
