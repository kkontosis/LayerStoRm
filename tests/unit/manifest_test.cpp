#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "model/weight_pipeline/manifest.h"
#include "model/weight_pipeline/prepacked_format.h"
#include "model/quantization/nvfp4.h"
#include "model/quantization/fp8.h"

using namespace layerstorm::model;

// ── Helpers ─────────────────────────────────────────────────────────────────

static Manifest make_v32_nvfp4_manifest() {
    Nvfp4 quant;
    ExpertShape shape{7168, 2048};

    Manifest m;
    m.format_version = std::string{prepacked::kFormatVersion};
    m.engine_version = "9.65.1";
    m.source_model_path = "../DeepSeek-V3.2-NVFP4";
    m.source_freshness_timestamp = 1748793600;
    m.source_freshness_file = "model-00003-of-00012.safetensors";
    m.quant_format = std::string{prepacked::kNvfp4PackedQuantFormat};
    m.n_routed_experts = 256;
    m.n_expert_files = 256;
    m.expert_dimensions = {7168, 2048};
    m.moe_layers.count = 58;
    m.moe_layers.first_moe_layer = 3;
    m.moe_layers.last_moe_layer = 60;
    for (int i = 3; i <= 60; ++i) m.moe_layers.indices.push_back(i);
    m.slot = build_slot_from_quant(quant, shape);
    return m;
}

static Manifest make_v32_fp8e4m3_manifest() {
    Fp8E4M3 quant;
    ExpertShape shape{7168, 2048};

    Manifest m;
    m.format_version = std::string{prepacked::kFormatVersion};
    m.engine_version = "9.65.1";
    m.source_model_path = "../DeepSeek-V3.2-FP8";
    m.source_freshness_timestamp = 1748793600;
    m.source_freshness_file = "model-00003-of-00012.safetensors";
    m.quant_format = "fp8_e4m3";
    m.n_routed_experts = 256;
    m.n_expert_files = 256;
    m.expert_dimensions = {7168, 2048};
    m.moe_layers.count = 58;
    m.moe_layers.first_moe_layer = 3;
    m.moe_layers.last_moe_layer = 60;
    for (int i = 3; i <= 60; ++i) m.moe_layers.indices.push_back(i);
    m.slot = build_slot_from_quant(quant, shape);
    return m;
}

static Manifest make_v32_fp8e5m2_manifest() {
    Fp8E5M2 quant;
    ExpertShape shape{7168, 2048};

    Manifest m;
    m.format_version = std::string{prepacked::kFormatVersion};
    m.engine_version = "9.65.1";
    m.source_model_path = "../DeepSeek-V3.2-FP8-E5M2";
    m.source_freshness_timestamp = 1748793600;
    m.source_freshness_file = "model-00003-of-00012.safetensors";
    m.quant_format = "fp8_e5m2";
    m.n_routed_experts = 256;
    m.n_expert_files = 256;
    m.expert_dimensions = {7168, 2048};
    m.moe_layers.count = 58;
    m.moe_layers.first_moe_layer = 3;
    m.moe_layers.last_moe_layer = 60;
    for (int i = 3; i <= 60; ++i) m.moe_layers.indices.push_back(i);
    m.slot = build_slot_from_quant(quant, shape);
    return m;
}

// ── ManifestRoundTrip ───────────────────────────────────────────────────────

TEST(ManifestRoundTrip, Nvfp4) {
    auto m1 = make_v32_nvfp4_manifest();
    auto j = manifest_to_json(m1);
    auto m2 = manifest_from_json(j);

    EXPECT_EQ(m2.format_version, m1.format_version);
    EXPECT_EQ(m2.engine_version, m1.engine_version);
    EXPECT_EQ(m2.source_model_path, m1.source_model_path);
    EXPECT_EQ(m2.source_freshness_timestamp, m1.source_freshness_timestamp);
    EXPECT_EQ(m2.source_freshness_file, m1.source_freshness_file);
    EXPECT_EQ(m2.quant_format, m1.quant_format);
    EXPECT_EQ(m2.n_routed_experts, m1.n_routed_experts);
    EXPECT_EQ(m2.n_expert_files, m1.n_expert_files);
    EXPECT_EQ(m2.expert_dimensions.hidden_size, m1.expert_dimensions.hidden_size);
    EXPECT_EQ(m2.expert_dimensions.intermediate_size, m1.expert_dimensions.intermediate_size);
    EXPECT_EQ(m2.moe_layers.count, m1.moe_layers.count);
    EXPECT_EQ(m2.moe_layers.first_moe_layer, m1.moe_layers.first_moe_layer);
    EXPECT_EQ(m2.moe_layers.last_moe_layer, m1.moe_layers.last_moe_layer);
    EXPECT_EQ(m2.moe_layers.indices, m1.moe_layers.indices);
    EXPECT_EQ(m2.slot.slot_size_bytes, m1.slot.slot_size_bytes);
    EXPECT_EQ(m2.slot.alignment_bytes, m1.slot.alignment_bytes);
    ASSERT_EQ(m2.slot.projections.size(), m1.slot.projections.size());
    for (size_t i = 0; i < m1.slot.projections.size(); ++i) {
        EXPECT_EQ(m2.slot.projections[i].name, m1.slot.projections[i].name);
        EXPECT_EQ(m2.slot.projections[i].offset, m1.slot.projections[i].offset);
        EXPECT_EQ(m2.slot.projections[i].total_bytes, m1.slot.projections[i].total_bytes);
        EXPECT_EQ(m2.slot.projections[i].weight_bytes, m1.slot.projections[i].weight_bytes);
        EXPECT_EQ(m2.slot.projections[i].scale_bytes, m1.slot.projections[i].scale_bytes);
        EXPECT_EQ(m2.slot.projections[i].scalar_bytes, m1.slot.projections[i].scalar_bytes);
        EXPECT_EQ(m2.slot.projections[i].description, m1.slot.projections[i].description);
    }
}

TEST(ManifestRoundTrip, Fp8E4M3) {
    auto m1 = make_v32_fp8e4m3_manifest();
    auto j = manifest_to_json(m1);
    auto m2 = manifest_from_json(j);

    EXPECT_EQ(m2.format_version, m1.format_version);
    EXPECT_EQ(m2.quant_format, "fp8_e4m3");
    EXPECT_EQ(m2.slot.slot_size_bytes, m1.slot.slot_size_bytes);
    ASSERT_EQ(m2.slot.projections.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(m2.slot.projections[i].total_bytes, m1.slot.projections[i].total_bytes);
        EXPECT_EQ(m2.slot.projections[i].weight_bytes, m1.slot.projections[i].weight_bytes);
        EXPECT_EQ(m2.slot.projections[i].scalar_bytes, 0);
    }
}

TEST(ManifestRoundTrip, Fp8E5M2) {
    auto m1 = make_v32_fp8e5m2_manifest();
    auto j = manifest_to_json(m1);
    auto m2 = manifest_from_json(j);

    EXPECT_EQ(m2.format_version, m1.format_version);
    EXPECT_EQ(m2.quant_format, "fp8_e5m2");
    EXPECT_EQ(m2.slot.slot_size_bytes, m1.slot.slot_size_bytes);
    ASSERT_EQ(m2.slot.projections.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(m2.slot.projections[i].total_bytes, m1.slot.projections[i].total_bytes);
    }
}

// ── ManifestSlotVerification ────────────────────────────────────────────────

TEST(ManifestSlotVerification, Nvfp4Matches) {
    Nvfp4 quant;
    auto m = make_v32_nvfp4_manifest();
    auto result = verify_manifest(m, quant);
    EXPECT_TRUE(result.ok) << result.error;
}

TEST(ManifestSlotVerification, Fp8E4M3Matches) {
    Fp8E4M3 quant;
    auto m = make_v32_fp8e4m3_manifest();
    auto result = verify_manifest(m, quant);
    EXPECT_TRUE(result.ok) << result.error;
}

TEST(ManifestSlotVerification, Fp8E5M2Matches) {
    Fp8E5M2 quant;
    auto m = make_v32_fp8e5m2_manifest();
    auto result = verify_manifest(m, quant);
    EXPECT_TRUE(result.ok) << result.error;
}

TEST(ManifestSlotVerification, CrossFormatMismatchRejects) {
    Fp8E4M3 quant;
    auto m = make_v32_nvfp4_manifest();
    auto result = verify_manifest(m, quant);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("Quant format mismatch"), std::string::npos);
}

TEST(ManifestSlotVerification, WrongVersionRejects) {
    Nvfp4 quant;
    auto m = make_v32_nvfp4_manifest();
    m.format_version = "0.0.0";
    auto result = verify_manifest(m, quant);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("Unsupported format version"), std::string::npos);
}

// 9.66.0 raw-scale data must be refused (scales were stored row-major; the
// engine no longer has a runtime reformat).
TEST(ManifestSlotVerification, LegacyRawScaleVersionRejects) {
    Nvfp4 quant;
    auto m = make_v32_nvfp4_manifest();
    m.format_version = "9.66.0";
    auto result = verify_manifest(m, quant);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("Unsupported format version"), std::string::npos);
}

// A plain "nvfp4" quant_format means raw-layout scales — refuse it: the
// engine only consumes the Sm1xx-interleaved subcategory ("nvfp4-sm1xx").
TEST(ManifestSlotVerification, PlainNvfp4QuantFormatRejects) {
    Nvfp4 quant;
    auto m = make_v32_nvfp4_manifest();
    m.quant_format = "nvfp4";
    auto result = verify_manifest(m, quant);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("Quant format mismatch"), std::string::npos);
}

TEST(ManifestSlotVerification, IndicesCountMismatchRejects) {
    Nvfp4 quant;
    auto m = make_v32_nvfp4_manifest();
    m.moe_layers.count = 99;
    auto result = verify_manifest(m, quant);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("moe_layers.count"), std::string::npos);
}

TEST(ManifestSlotVerification, FirstMoeLayerMismatchRejects) {
    Nvfp4 quant;
    auto m = make_v32_nvfp4_manifest();
    m.moe_layers.first_moe_layer = 0;  // indices[0] is 3
    auto result = verify_manifest(m, quant);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("first_moe_layer"), std::string::npos);
}

TEST(ManifestSlotVerification, LastMoeLayerMismatchRejects) {
    Nvfp4 quant;
    auto m = make_v32_nvfp4_manifest();
    m.moe_layers.last_moe_layer = 99;  // indices.back() is 60
    auto result = verify_manifest(m, quant);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("last_moe_layer"), std::string::npos);
}

TEST(ManifestSlotVerification, UnsortedIndicesRejects) {
    Nvfp4 quant;
    auto m = make_v32_nvfp4_manifest();
    std::reverse(m.moe_layers.indices.begin(), m.moe_layers.indices.end());
    m.moe_layers.first_moe_layer = m.moe_layers.indices.front();  // 60
    m.moe_layers.last_moe_layer = m.moe_layers.indices.back();    // 3
    auto result = verify_manifest(m, quant);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("not sorted"), std::string::npos);
}

TEST(ManifestSlotVerification, ExpertFileCountMismatchRejects) {
    Nvfp4 quant;
    auto m = make_v32_nvfp4_manifest();
    m.n_expert_files = 128;  // n_routed_experts is 256
    auto result = verify_manifest(m, quant);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("n_expert_files"), std::string::npos);
}

// ── ManifestSlotConsistency ─────────────────────────────────────────────────

TEST(ManifestSlotConsistency, Nvfp4AlignmentIs128) {
    // NVFP4 bytes_per_projection enforces 128-byte TMA alignment (nvfp4.cpp kAlign).
    auto m = make_v32_nvfp4_manifest();
    EXPECT_EQ(m.slot.alignment_bytes, 128);
    for (const auto& p : m.slot.projections) {
        EXPECT_EQ(p.total_bytes % 128, 0) << "projection=" << p.name;
    }
}

TEST(ManifestSlotConsistency, AlignmentDividesAllProjections) {
    for (const auto make_fn : {make_v32_nvfp4_manifest,
                                make_v32_fp8e4m3_manifest,
                                make_v32_fp8e5m2_manifest}) {
        auto m = make_fn();
        EXPECT_GT(m.slot.alignment_bytes, 0) << "quant=" << m.quant_format;
        for (const auto& p : m.slot.projections) {
            EXPECT_EQ(p.total_bytes % m.slot.alignment_bytes, 0)
                << "projection=" << p.name << " quant=" << m.quant_format;
        }
    }
}

TEST(ManifestSlotConsistency, SlotSizeEqualsSumOfProjections) {
    for (const auto make_fn : {make_v32_nvfp4_manifest,
                                make_v32_fp8e4m3_manifest,
                                make_v32_fp8e5m2_manifest}) {
        auto m = make_fn();
        int64_t sum = 0;
        for (const auto& p : m.slot.projections) sum += p.total_bytes;
        EXPECT_EQ(m.slot.slot_size_bytes, sum) << "quant=" << m.quant_format;
    }
}

TEST(ManifestSlotConsistency, BytesPerProjectionCrossCheck) {
    {
        Nvfp4 quant;
        ExpertShape shape{7168, 2048};
        auto m = make_v32_nvfp4_manifest();
        EXPECT_EQ(m.slot.projections[0].total_bytes,
                  quant.bytes_per_projection(shape, Projection::gate));
        EXPECT_EQ(m.slot.projections[1].total_bytes,
                  quant.bytes_per_projection(shape, Projection::up));
        EXPECT_EQ(m.slot.projections[2].total_bytes,
                  quant.bytes_per_projection(shape, Projection::down));
    }
    {
        Fp8E4M3 quant;
        ExpertShape shape{7168, 2048};
        auto m = make_v32_fp8e4m3_manifest();
        EXPECT_EQ(m.slot.projections[0].total_bytes,
                  quant.bytes_per_projection(shape, Projection::gate));
        EXPECT_EQ(m.slot.projections[1].total_bytes,
                  quant.bytes_per_projection(shape, Projection::up));
        EXPECT_EQ(m.slot.projections[2].total_bytes,
                  quant.bytes_per_projection(shape, Projection::down));
    }
}

TEST(ManifestSlotConsistency, ProjectionOffsetsMonotonic) {
    auto m = make_v32_nvfp4_manifest();
    ASSERT_EQ(m.slot.projections.size(), 3u);
    EXPECT_EQ(m.slot.projections[0].offset, 0);
    EXPECT_EQ(m.slot.projections[1].offset, m.slot.projections[0].total_bytes);
    EXPECT_EQ(m.slot.projections[2].offset,
              m.slot.projections[0].total_bytes + m.slot.projections[1].total_bytes);
}

TEST(ManifestSlotConsistency, ScaleBytesNonNegative) {
    for (const auto make_fn : {make_v32_nvfp4_manifest,
                                make_v32_fp8e4m3_manifest,
                                make_v32_fp8e5m2_manifest}) {
        auto m = make_fn();
        for (const auto& p : m.slot.projections) {
            EXPECT_GE(p.scale_bytes, 0) << "projection=" << p.name
                                        << " quant=" << m.quant_format;
        }
    }
}

// ── ManifestMoeLayers ───────────────────────────────────────────────────────

TEST(ManifestMoeLayers, ContiguousPosition) {
    ManifestMoeLayers ml;
    ml.count = 58;
    ml.first_moe_layer = 3;
    ml.last_moe_layer = 60;
    for (int i = 3; i <= 60; ++i) ml.indices.push_back(i);

    EXPECT_TRUE(ml.is_contiguous());
    EXPECT_EQ(ml.layer_position(3), 0);
    EXPECT_EQ(ml.layer_position(10), 7);
    EXPECT_EQ(ml.layer_position(60), 57);
}

TEST(ManifestMoeLayers, NonContiguousPosition) {
    ManifestMoeLayers ml;
    ml.indices = {3, 5, 7, 10, 20};
    ml.count = 5;
    ml.first_moe_layer = 3;
    ml.last_moe_layer = 20;

    EXPECT_FALSE(ml.is_contiguous());
    EXPECT_EQ(ml.layer_position(3), 0);
    EXPECT_EQ(ml.layer_position(5), 1);
    EXPECT_EQ(ml.layer_position(7), 2);
    EXPECT_EQ(ml.layer_position(10), 3);
    EXPECT_EQ(ml.layer_position(20), 4);
}

TEST(ManifestMoeLayers, OutOfRangeReturnsNegOne) {
    ManifestMoeLayers ml;
    ml.count = 58;
    ml.first_moe_layer = 3;
    ml.last_moe_layer = 60;
    for (int i = 3; i <= 60; ++i) ml.indices.push_back(i);

    EXPECT_EQ(ml.layer_position(0), -1);
    EXPECT_EQ(ml.layer_position(2), -1);
    EXPECT_EQ(ml.layer_position(61), -1);
    EXPECT_EQ(ml.layer_position(100), -1);
}

TEST(ManifestMoeLayers, NonContiguousOutOfRange) {
    ManifestMoeLayers ml;
    ml.indices = {3, 5, 7};
    ml.count = 3;
    ml.first_moe_layer = 3;
    ml.last_moe_layer = 7;

    EXPECT_EQ(ml.layer_position(4), -1);
    EXPECT_EQ(ml.layer_position(6), -1);
    EXPECT_EQ(ml.layer_position(8), -1);
}

// ── PrepackedFormat ─────────────────────────────────────────────────────────

TEST(PrepackedFormat, ExpertFilename) {
    EXPECT_EQ(prepacked::expert_filename(0), "expert_000.bin");
    EXPECT_EQ(prepacked::expert_filename(42), "expert_042.bin");
    EXPECT_EQ(prepacked::expert_filename(255), "expert_255.bin");
}

TEST(PrepackedFormat, SlotOffset) {
    EXPECT_EQ(prepacked::slot_offset(0, 1000), 0);
    EXPECT_EQ(prepacked::slot_offset(1, 1000), 1000);
    EXPECT_EQ(prepacked::slot_offset(5, 1000), 5000);
    EXPECT_EQ(prepacked::slot_offset(57, 1835520), 57 * int64_t{1835520});
}

TEST(PrepackedFormat, ExpectedFileSize) {
    EXPECT_EQ(prepacked::expected_file_size(58, 1835520), 58 * int64_t{1835520});
    EXPECT_EQ(prepacked::expected_file_size(1, 100), 100);
    EXPECT_EQ(prepacked::expected_file_size(0, 100), 0);
}

TEST(PrepackedFormat, ManifestPath) {
    auto p = prepacked::manifest_path("/data/prepacked");
    EXPECT_EQ(p.filename(), "manifest.json");
}

TEST(PrepackedFormat, ExpertFilePath) {
    auto p = prepacked::expert_file_path("/data/prepacked", 42);
    EXPECT_EQ(p.filename(), "expert_042.bin");
    EXPECT_EQ(p.parent_path(), "/data/prepacked");
}
