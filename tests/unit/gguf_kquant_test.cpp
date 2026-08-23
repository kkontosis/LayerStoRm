#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "model/quantization/gguf_kquant.h"
#include "model/quantization/registry.h"

using namespace layerstorm::model;
using namespace layerstorm;

namespace {

// Hand-computed reference: out * (in/QK) * block_bytes.
int64_t expected_packed(int64_t out, int64_t in, int block_bytes, int qk) {
    return out * (in / qk) * block_bytes;
}

}  // namespace

// ── Reusable primitive: gguf_packed_bytes ───────────────────────────────────

TEST(GgufKQuant, PackedBytesPerTypeMatchesFormula) {
    struct Case { GgufKQuantType type; int block_bytes; int qk; };
    const Case cases[] = {
        {GgufKQuantType::Q2_K, 84, 256},
        {GgufKQuantType::Q3_K, 110, 256},
        {GgufKQuantType::Q4_K, 144, 256},
        {GgufKQuantType::Q5_K, 176, 256},
        {GgufKQuantType::Q6_K, 210, 256},
        {GgufKQuantType::Q8_0, 34, 32},
    };
    // Two shapes per type; in must be a multiple of QK (256 and 32 both divide).
    const std::pair<int64_t, int64_t> shapes[] = {{2048, 7168}, {7168, 2048}};
    for (const auto& c : cases) {
        EXPECT_EQ(gguf::block_bytes(c.type), c.block_bytes);
        EXPECT_EQ(gguf::block_values(c.type), c.qk);
        for (auto [out, in] : shapes) {
            EXPECT_EQ(gguf::gguf_packed_bytes(out, in, c.type),
                      expected_packed(out, in, c.block_bytes, c.qk))
                << "type=" << gguf::type_name(c.type) << " out=" << out << " in=" << in;
        }
    }
}

TEST(GgufKQuant, PackedBytesThrowsOnNonMultipleOfQk) {
    // 100 is not a multiple of 256 (k-quant QK).
    EXPECT_THROW(gguf::gguf_packed_bytes(2048, 100, GgufKQuantType::Q4_K),
                 std::runtime_error);
    // 33 is not a multiple of 32 (Q8_0 QK).
    EXPECT_THROW(gguf::gguf_packed_bytes(2048, 33, GgufKQuantType::Q8_0),
                 std::runtime_error);
}

TEST(GgufKQuant, BytesPerElementFractional) {
    EXPECT_DOUBLE_EQ(gguf::bytes_per_element(GgufKQuantType::Q4_K), 144.0 / 256.0);
    EXPECT_DOUBLE_EQ(gguf::bytes_per_element(GgufKQuantType::Q4_K), 0.5625);
    EXPECT_DOUBLE_EQ(gguf::bytes_per_element(GgufKQuantType::Q6_K), 210.0 / 256.0);
    EXPECT_DOUBLE_EQ(gguf::bytes_per_element(GgufKQuantType::Q8_0), 34.0 / 32.0);
}

// ── Uniform QuantInterface per type ─────────────────────────────────────────

TEST(GgufKQuant, UniformProjectionAndExpertBytes) {
    const ExpertShape shape{7168, 2048};  // hidden, intermediate
    // gate/up: [intermediate, hidden] = [2048, 7168]; down: [7168, 2048].
    struct Case { GgufKQuantType type; int block_bytes; int qk; };
    const Case cases[] = {
        {GgufKQuantType::Q2_K, 84, 256},
        {GgufKQuantType::Q3_K, 110, 256},
        {GgufKQuantType::Q4_K, 144, 256},
        {GgufKQuantType::Q5_K, 176, 256},
        {GgufKQuantType::Q6_K, 210, 256},
        {GgufKQuantType::Q8_0, 34, 32},
    };
    for (const auto& c : cases) {
        GgufQuantInterface q{c.type};

        int64_t gate = expected_packed(2048, 7168, c.block_bytes, c.qk);
        int64_t up   = expected_packed(2048, 7168, c.block_bytes, c.qk);
        int64_t down = expected_packed(7168, 2048, c.block_bytes, c.qk);

        EXPECT_EQ(q.bytes_per_projection(shape, Projection::gate), gate);
        EXPECT_EQ(q.bytes_per_projection(shape, Projection::up), up);
        EXPECT_EQ(q.bytes_per_projection(shape, Projection::down), down);
        EXPECT_EQ(q.bytes_per_expert(shape), gate + up + down);

        // K-quants store scales inside blocks: weight == total, replicated == 0.
        EXPECT_EQ(q.weight_bytes_per_projection(shape, Projection::gate), gate);
        EXPECT_EQ(q.weight_bytes_per_projection(shape, Projection::down), down);
        EXPECT_EQ(q.replicated_bytes_per_projection(), 0);

        EXPECT_TRUE(q.tensor_core_eligible());
        EXPECT_EQ(q.name(), gguf::type_name(c.type));
    }
}

TEST(GgufKQuant, UniformMemoryLayout) {
    GgufQuantInterface q4{GgufKQuantType::Q4_K};
    auto l = q4.memory_layout();
    EXPECT_EQ(l.bits_per_weight, 5);  // round(8*0.5625) = round(4.5) = 5 (round-half-up via lround)
    EXPECT_EQ(l.group_size, 256);
    EXPECT_FALSE(l.has_zero_point);
    EXPECT_EQ(l.scale_dtype, ScaleDtype::fp16);
    EXPECT_EQ(l.alignment_bytes, 144);

    GgufQuantInterface q8{GgufKQuantType::Q8_0};
    auto l8 = q8.memory_layout();
    EXPECT_EQ(l8.group_size, 32);
    EXPECT_EQ(l8.alignment_bytes, 34);
}

// ── Mixed construction ──────────────────────────────────────────────────────

TEST(GgufKQuant, MixedPerProjectionSizing) {
    const ExpertShape shape{7168, 2048};
    // gate=Q4_K, up=Q4_K, down=Q6_K.
    GgufQuantInterface q = make_gguf_quant(GgufKQuantType::Q4_K,
                                           GgufKQuantType::Q4_K,
                                           GgufKQuantType::Q6_K);
    EXPECT_EQ(q.name(), "gguf");
    EXPECT_EQ(q.weight_quant(), config::WeightQuant::gguf);
    EXPECT_EQ(q.projection_type(Projection::gate), GgufKQuantType::Q4_K);
    EXPECT_EQ(q.projection_type(Projection::down), GgufKQuantType::Q6_K);

    int64_t gate = expected_packed(2048, 7168, 144, 256);
    int64_t up   = expected_packed(2048, 7168, 144, 256);
    int64_t down = expected_packed(7168, 2048, 210, 256);  // Q6_K

    EXPECT_EQ(q.bytes_per_projection(shape, Projection::gate), gate);
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::up), up);
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::down), down);
    EXPECT_EQ(q.bytes_per_expert(shape), gate + up + down);
    EXPECT_NE(q.bytes_per_projection(shape, Projection::up),
              q.bytes_per_projection(shape, Projection::down));
}

TEST(GgufKQuant, MixedAllSameIsUniform) {
    GgufQuantInterface q = make_gguf_quant(GgufKQuantType::Q5_K,
                                           GgufKQuantType::Q5_K,
                                           GgufKQuantType::Q5_K);
    EXPECT_EQ(q.name(), "gguf_q5_k");
    EXPECT_EQ(q.weight_quant(), config::WeightQuant::gguf_q5_k);
}

// ── Registry integration ────────────────────────────────────────────────────
// The GGUF singletons live in the process-wide registry. Other suites
// (Fp8/Nvfp4 ManualRegistration) call clear_registry() and do NOT restore the
// GGUF globals, so in-suite these reads would find an empty registry. The
// fixture re-registers the GGUF formats (idempotent) in SetUp so these tests are
// order-independent. They must NOT clear_registry() themselves.

class GgufKQuantRegistry : public ::testing::Test {
protected:
    void SetUp() override { register_gguf_formats(); }
};

TEST_F(GgufKQuantRegistry, RegistryUniformLookup) {
    const config::WeightQuant variants[] = {
        config::WeightQuant::gguf_q2_k, config::WeightQuant::gguf_q3_k,
        config::WeightQuant::gguf_q4_k, config::WeightQuant::gguf_q5_k,
        config::WeightQuant::gguf_q6_k, config::WeightQuant::gguf_q8_0,
    };
    for (auto wq : variants) {
        const QuantInterface* p = find_format(wq);
        ASSERT_NE(p, nullptr) << static_cast<int>(wq);
        EXPECT_EQ(p->weight_quant(), wq);
        // Name round-trip: by-enum instance is the same as by-name instance.
        EXPECT_EQ(find_format(p->name()), p);
        // Working instance: sizes a real expert > 0.
        EXPECT_GT(p->bytes_per_expert(ExpertShape{7168, 2048}), 0);
    }
}

TEST_F(GgufKQuantRegistry, RegistryQ4kMatchesFormula) {
    const QuantInterface& q = get_format(config::WeightQuant::gguf_q4_k);
    const ExpertShape shape{7168, 2048};
    int64_t gate = expected_packed(2048, 7168, 144, 256);
    int64_t down = expected_packed(7168, 2048, 144, 256);
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::gate), gate);
    EXPECT_EQ(q.bytes_per_projection(shape, Projection::down), down);
}

TEST_F(GgufKQuantRegistry, RegistryGenericGgufResolvesButSizingThrows) {
    // TD-GGUF-GENERIC-DEFAULT-MISSIZE (resolved by GG-6): get_format(gguf)
    // resolves to the sentinel so registry lookups + config validation work, but
    // its sizing methods THROW — real mixed sizing MUST go through
    // make_gguf_quant() built from the file's per-projection types.
    const QuantInterface& q = get_format(config::WeightQuant::gguf);
    EXPECT_EQ(q.name(), "gguf");
    EXPECT_EQ(q.weight_quant(), config::WeightQuant::gguf);
    EXPECT_TRUE(q.tensor_core_eligible());

    const ExpertShape shape{7168, 2048};
    EXPECT_THROW(q.bytes_per_expert(shape), std::runtime_error);
    EXPECT_THROW(q.bytes_per_projection(shape, Projection::gate), std::runtime_error);
    EXPECT_THROW(q.weight_bytes_per_projection(shape, Projection::down), std::runtime_error);
    EXPECT_THROW(q.replicated_bytes_per_projection(), std::runtime_error);
    EXPECT_THROW(q.bytes_per_element(), std::runtime_error);
    EXPECT_THROW(q.dequant_flops_per_element(), std::runtime_error);
    EXPECT_THROW(q.memory_layout(), std::runtime_error);

    // find_format round-trips by name "gguf".
    EXPECT_EQ(find_format("gguf"), &q);
}

TEST(GgufKQuant, TypeFromWeightQuantGenericThrows) {
    EXPECT_THROW(gguf::type_from_weight_quant(config::WeightQuant::gguf),
                 std::runtime_error);
    EXPECT_THROW(gguf::type_from_weight_quant(config::WeightQuant::nvfp4),
                 std::runtime_error);
    EXPECT_EQ(gguf::type_from_weight_quant(config::WeightQuant::gguf_q6_k),
              GgufKQuantType::Q6_K);
}

// ═══════════════════════════════════════════════════════════════════════════
// MXFP4 (V4 QAT routed experts) — dequant-correctness golden vs the
// ref/llama.cpp type-39 reference math (ggml dequantize_row_mxfp4 +
// ggml_e8m0_to_fp32_half), CPU-only.
// ═══════════════════════════════════════════════════════════════════════════

TEST(GgufKQuantMxfp4, BlockSpecAndSizing) {
    EXPECT_EQ(gguf::block_bytes(GgufKQuantType::MXFP4), 17);
    EXPECT_EQ(gguf::block_values(GgufKQuantType::MXFP4), 32);
    EXPECT_DOUBLE_EQ(gguf::bytes_per_element(GgufKQuantType::MXFP4), 17.0 / 32.0);
    // A V4-Flash routed expert projection: [2048, 4096] → 2048*(4096/32)*17.
    EXPECT_EQ(gguf::gguf_packed_bytes(2048, 4096, GgufKQuantType::MXFP4),
              2048LL * 128 * 17);
    // K not a multiple of 32 throws.
    EXPECT_THROW(gguf::gguf_packed_bytes(8, 40, GgufKQuantType::MXFP4),
                 std::runtime_error);
}

TEST(GgufKQuantMxfp4, NamesAndRegistry) {
    EXPECT_EQ(gguf::type_name(GgufKQuantType::MXFP4), "gguf_mxfp4");
    EXPECT_EQ(gguf::type_from_name("gguf_mxfp4"), GgufKQuantType::MXFP4);
    EXPECT_EQ(gguf::type_from_name("MXFP4"), GgufKQuantType::MXFP4);
    EXPECT_EQ(gguf::type_from_weight_quant(config::WeightQuant::gguf_mxfp4),
              GgufKQuantType::MXFP4);
    EXPECT_TRUE(gguf::is_gguf_weight_quant(config::WeightQuant::gguf_mxfp4));
    register_gguf_formats();
    const auto* fmt = find_format("gguf_mxfp4");
    ASSERT_NE(fmt, nullptr);
    EXPECT_EQ(fmt->weight_quant(), config::WeightQuant::gguf_mxfp4);
    // Uniform interface: V4-Flash expert = 3 x 2048x4096-shaped projections.
    auto quant = make_gguf_quant(GgufKQuantType::MXFP4, GgufKQuantType::MXFP4,
                                 GgufKQuantType::MXFP4);
    ExpertShape shape{};
    shape.hidden_size = 4096;
    shape.intermediate_size = 2048;
    EXPECT_EQ(quant.bytes_per_expert(shape), 3LL * 2048 * 128 * 17);
    EXPECT_EQ(quant.memory_layout().scale_dtype, ScaleDtype::ue8m0);
    EXPECT_EQ(quant.memory_layout().group_size, 32);
}

TEST(GgufKQuantMxfp4, E8m0HalfMatchesReference) {
    // ggml_e8m0_to_fp32_half: 2^(e-127)/2, with e<2 denormal patterns.
    EXPECT_FLOAT_EQ(gguf::e8m0_to_fp32_half(127), 0.5f);
    EXPECT_FLOAT_EQ(gguf::e8m0_to_fp32_half(128), 1.0f);
    EXPECT_FLOAT_EQ(gguf::e8m0_to_fp32_half(129), 2.0f);
    EXPECT_FLOAT_EQ(gguf::e8m0_to_fp32_half(121), 1.0f / 128.0f);
    EXPECT_FLOAT_EQ(gguf::e8m0_to_fp32_half(1), std::ldexp(1.0f, -127));
    EXPECT_FLOAT_EQ(gguf::e8m0_to_fp32_half(0), std::ldexp(1.0f, -128));
}

TEST(GgufKQuantMxfp4, DequantGoldenAllCodes) {
    // Block with e=128 (d = 1.0) and qs[j] = (hi=j's pair) — enumerate all 16
    // e2m1 codes in both nibble positions and pin the exact llama.cpp table:
    // {0, 1, 2, 3, 4, 6, 8, 12, -0, -1, -2, -3, -4, -6, -8, -12} (doubled
    // values un-doubled by the halved scale; here d=1.0 keeps them doubled).
    const float kExpected[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                 0, -1, -2, -3, -4, -6, -8, -12};
    uint8_t block[17];
    block[0] = 128;  // d = 1.0
    for (int j = 0; j < 16; ++j) {
        // low nibble = code j (elements 0..15); high nibble = code (15-j)
        // (elements 16..31 reversed) — covers every code in BOTH positions.
        block[1 + j] = static_cast<uint8_t>((j & 0xF) | (((15 - j) & 0xF) << 4));
    }
    float out[32];
    gguf::dequant_mxfp4_block(block, out);
    for (int j = 0; j < 16; ++j) {
        EXPECT_FLOAT_EQ(out[j], kExpected[j]) << "low nibble code " << j;
        EXPECT_FLOAT_EQ(out[16 + j], kExpected[15 - j]) << "high nibble code "
                                                        << (15 - j);
    }
}

TEST(GgufKQuantMxfp4, DequantGoldenHandComputed) {
    // Hand-computed spot checks at e=127 (d = 0.5).
    uint8_t block[17] = {};
    block[0] = 127;
    block[1] = 0x21;  // low=1 → +1*0.5 = 0.5 (elem 0); high=2 → +2*0.5 = 1.0 (elem 16)
    block[2] = 0x9F;  // low=15 → -12*0.5 = -6.0 (elem 1); high=9 → -1*0.5 = -0.5 (elem 17)
    float out[32];
    gguf::dequant_mxfp4_block(block, out);
    EXPECT_FLOAT_EQ(out[0], 0.5f);
    EXPECT_FLOAT_EQ(out[16], 1.0f);
    EXPECT_FLOAT_EQ(out[1], -6.0f);
    EXPECT_FLOAT_EQ(out[17], -0.5f);
    for (int i = 2; i < 16; ++i) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
        EXPECT_FLOAT_EQ(out[16 + i], 0.0f);
    }

    // dequant_mxfp4_row: two blocks, second at e=129 (d = 2.0), code 7 = +12
    // doubled → 24.0.
    uint8_t two[34] = {};
    std::memcpy(two, block, 17);
    two[17] = 129;
    two[18] = 0x07;  // low nibble code 7 → elem 32+0 = 12*2 = 24.0
    float out2[64];
    gguf::dequant_mxfp4_row(two, out2, 64);
    EXPECT_FLOAT_EQ(out2[0], 0.5f);   // block 0 unchanged
    EXPECT_FLOAT_EQ(out2[32], 24.0f);
    EXPECT_THROW(gguf::dequant_mxfp4_row(two, out2, 33), std::runtime_error);
}
