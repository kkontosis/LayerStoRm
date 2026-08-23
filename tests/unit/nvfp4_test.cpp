#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "model/quantization/nvfp4.h"
#include "model/quantization/registry.h"

using namespace layerstorm::model;
using namespace layerstorm;

// ── QuantInterface metadata ─────────────────────────────────────────────────

TEST(Nvfp4, Methods) {
    Nvfp4 q;
    EXPECT_EQ(q.name(), "nvfp4");
    EXPECT_EQ(q.weight_quant(), config::WeightQuant::nvfp4);
    EXPECT_TRUE(q.tensor_core_eligible());
    EXPECT_DOUBLE_EQ(q.dequant_flops_per_element(), 2.0);
    EXPECT_DOUBLE_EQ(q.bytes_per_element(), 0.5625);
}

TEST(Nvfp4, MemoryLayout) {
    Nvfp4 q;
    auto layout = q.memory_layout();
    EXPECT_EQ(layout.bits_per_weight, 4);
    EXPECT_EQ(layout.group_size, 16);
    EXPECT_FALSE(layout.has_zero_point);
    EXPECT_EQ(layout.scale_dtype, ScaleDtype::ue8m0);
    EXPECT_EQ(layout.alignment_bytes, 32);
}

// ── Size calculations ───────────────────────────────────────────────────────

TEST(Nvfp4, BytesPerProjectionV32) {
    Nvfp4 q;
    ExpertShape shape{7168, 2048};
    // Each projection: 7168 * 2048 = 14,680,064 params
    // weight_bytes = 14,680,064 / 2 = 7,340,032
    // scale_bytes  = 14,680,064 / 16 = 917,504
    // scalar_scales = 8 (weight_scale_2 + input_scale)
    // raw = 8,257,544, aligned to 128 = 8,257,664
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::gate), 8'257'664);
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::up),   8'257'664);
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::down), 8'257'664);
}

TEST(Nvfp4, BytesPerExpertV32) {
    Nvfp4 q;
    ExpertShape shape{7168, 2048};
    EXPECT_EQ(q.bytes_per_expert(shape), 3 * 8'257'664);  // 24,772,992
}

TEST(Nvfp4, BytesPerExpertConsistency) {
    Nvfp4 q;
    ExpertShape shape{7168, 2048};
    int64_t sum = q.bytes_per_projection(shape, Projection::gate)
                + q.bytes_per_projection(shape, Projection::up)
                + q.bytes_per_projection(shape, Projection::down);
    EXPECT_EQ(q.bytes_per_expert(shape), sum);
}

TEST(Nvfp4, BytesPerElementConsistency) {
    Nvfp4 q;
    ExpertShape shape{7168, 2048};
    double ratio = static_cast<double>(q.bytes_per_expert(shape)) / shape.total_params();
    // Ratio exceeds raw bytes_per_element slightly due to 128-byte alignment padding.
    EXPECT_GE(ratio, q.bytes_per_element());
    EXPECT_NEAR(ratio, q.bytes_per_element(), 0.001);  // <0.1% overhead
}

TEST(Nvfp4, BytesPerProjectionNonAligned) {
    Nvfp4 q;
    // 100 * 17 = 1700 params (not divisible by 16)
    ExpertShape shape{100, 17};
    int64_t gate = q.bytes_per_projection(shape, Projection::gate);
    int64_t expected_weight = (1700 + 1) / 2;    // 850
    int64_t expected_scale  = (1700 + 15) / 16;  // 108
    int64_t scalar_scales   = 8;                  // weight_scale_2 + input_scale
    int64_t raw = expected_weight + expected_scale + scalar_scales;  // 966
    int64_t aligned = (raw + 127) & ~int64_t{127};  // 1024
    EXPECT_EQ(gate, aligned);
}

TEST(Nvfp4, BytesPerProjectionGlm5) {
    Nvfp4 q;
    ExpertShape shape{6144, 2048};
    // 6144 * 2048 = 12,582,912 params
    // weight_bytes = 12,582,912 / 2 = 6,291,456
    // scale_bytes  = 12,582,912 / 16 = 786,432
    // scalar_scales = 8, raw = 7,077,896, aligned to 128 = 7,078,016
    int64_t raw = 6'291'456 + 786'432 + 8;
    int64_t expected = (raw + 127) & ~int64_t{127};  // 7,078,016
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::gate), expected);
    EXPECT_EQ(q.bytes_per_expert(shape), 3 * expected);
}

// ── E2M1 decode ─────────────────────────────────────────────────────────────

TEST(Nvfp4, DecodeE2M1AllValues) {
    // Positive values (sign bit = 0): nibbles 0x0..0x7
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0x0),  0.0f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0x1),  0.5f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0x2),  1.0f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0x3),  1.5f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0x4),  2.0f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0x5),  3.0f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0x6),  4.0f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0x7),  6.0f);

    // Negative values (sign bit = 1): nibbles 0x8..0xF
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0x8), -0.0f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0x9), -0.5f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0xA), -1.0f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0xB), -1.5f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0xC), -2.0f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0xD), -3.0f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0xE), -4.0f);
    EXPECT_FLOAT_EQ(nvfp4::decode_e2m1(0xF), -6.0f);
}

// ── UE8M0 decode ────────────────────────────────────────────────────────────

TEST(Nvfp4, DecodeUe8m0KnownValues) {
    EXPECT_FLOAT_EQ(nvfp4::decode_ue8m0(0),   0.0f);     // zero
    EXPECT_FLOAT_EQ(nvfp4::decode_ue8m0(127), 1.0f);      // 2^0
    EXPECT_FLOAT_EQ(nvfp4::decode_ue8m0(128), 2.0f);      // 2^1
    EXPECT_FLOAT_EQ(nvfp4::decode_ue8m0(126), 0.5f);      // 2^(-1)
    EXPECT_FLOAT_EQ(nvfp4::decode_ue8m0(129), 4.0f);      // 2^2
    EXPECT_FLOAT_EQ(nvfp4::decode_ue8m0(1),   std::ldexp(1.0f, -126)); // 2^(-126)
}

// ── Dequantize ──────────────────────────────────────────────────────────────

TEST(Nvfp4, DequantizeSimpleGroup) {
    // 16 elements, scale = 1.0 (UE8M0 byte 127)
    // Pack: element[0]=0x2 (1.0), element[1]=0x5 (3.0), rest zeros
    std::vector<uint8_t> packed(8, 0);  // 16 elements / 2
    packed[0] = 0x52;  // low nibble=0x2 (1.0), high nibble=0x5 (3.0)

    uint8_t scales[1] = {127};  // scale = 1.0
    float out[16] = {};

    nvfp4::dequantize(packed.data(), scales, out, 16);

    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 3.0f);
    for (int i = 2; i < 16; ++i) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }

    // Same data with scale = 2.0 (UE8M0 byte 128): values should double
    scales[0] = 128;
    nvfp4::dequantize(packed.data(), scales, out, 16);

    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[1], 6.0f);
}

TEST(Nvfp4, DequantizeMultipleGroups) {
    // 32 elements = 2 groups of 16, different scales
    std::vector<uint8_t> packed(16, 0);  // 32 / 2
    // Group 0: element[0] = 0x3 (1.5), scale = 128 (2.0) → 3.0
    packed[0] = 0x03;  // low=0x3, high=0x0
    // Group 1: element[16] = 0x4 (2.0), scale = 126 (0.5) → 1.0
    packed[8] = 0x04;  // low=0x4, high=0x0

    uint8_t scales[2] = {128, 126};
    float out[32] = {};

    nvfp4::dequantize(packed.data(), scales, out, 32);

    EXPECT_FLOAT_EQ(out[0], 3.0f);   // 1.5 * 2.0
    EXPECT_FLOAT_EQ(out[16], 1.0f);  // 2.0 * 0.5
}

TEST(Nvfp4, DequantizePackingOrder) {
    // Verify low nibble is element 0, high nibble is element 1
    uint8_t packed[1] = {0x71};  // low=0x1 (0.5), high=0x7 (6.0)
    uint8_t scales[1] = {127};   // scale = 1.0
    float out[2] = {};

    nvfp4::dequantize(packed, scales, out, 2);

    EXPECT_FLOAT_EQ(out[0], 0.5f);  // low nibble
    EXPECT_FLOAT_EQ(out[1], 6.0f);  // high nibble
}

TEST(Nvfp4, DequantizeZeroScale) {
    // Scale byte 0 → scale = 0.0 → all outputs zero regardless of weights
    uint8_t packed[1] = {0x77};  // both nibbles = 0x7 (6.0)
    uint8_t scales[1] = {0};
    float out[2] = {};

    nvfp4::dequantize(packed, scales, out, 2);

    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
}

TEST(Nvfp4, DequantizeNonAlignedCount) {
    // 17 elements (odd): last element is low nibble of byte[8]
    std::vector<uint8_t> packed(9, 0);  // ceil(17/2) = 9 bytes
    packed[8] = 0xF3;  // low=0x3 (1.5), high=0xF (unused for element 17)

    // 2 scale groups: group 0 covers [0..15], group 1 covers [16]
    uint8_t scales[2] = {127, 128};  // group 0 scale=1.0, group 1 scale=2.0
    std::vector<float> out(17, -1.0f);

    nvfp4::dequantize(packed.data(), scales, out.data(), 17);

    // Elements 0-15 are zero (packed bytes 0-7 are 0x00), scale 1.0
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }
    // Element 16: low nibble of byte[8] = 0x3 → 1.5, scale = 2.0 → 3.0
    EXPECT_FLOAT_EQ(out[16], 3.0f);
}

// ── Registry integration ────────────────────────────────────────────────────

TEST(Nvfp4, ManualRegistration) {
    clear_registry();
    Nvfp4 instance;
    register_format(&instance);

    EXPECT_EQ(find_format("nvfp4"), &instance);
    EXPECT_EQ(find_format(config::WeightQuant::nvfp4), &instance);

    auto& ref = get_format("nvfp4");
    EXPECT_EQ(ref.name(), "nvfp4");
    EXPECT_DOUBLE_EQ(ref.bytes_per_element(), 0.5625);

    clear_registry();
}
