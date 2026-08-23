// PrepackedSource unit tests (WP-3).
//
// Creates synthetic pre-processed files using prepack_experts (WP-2),
// then tests PrepackedSource resolve/has/error handling.

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include "model/quantization/nvfp4.h"
#include "model/quantization/fp8.h"
#include "model/model_config.h"
#include "model/weight_pipeline/expert_prepacker.h"
#include "model/weight_pipeline/manifest.h"
#include "model/weight_pipeline/prepacked_format.h"
#include "model/weight_pipeline/prepacked_source.h"
#include "model/weight_loader/weight_loader.h"
#include "weight_pipeline_test_helpers.h"

using namespace layerstorm::model;
namespace fs = std::filesystem;

// ── Test Fixture ───────────────────────────────────────────────────────────

class PrepackedSourceTest : public ::testing::Test,
                            public layerstorm::test::WeightPipelineHelpers {
protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "layerstorm_prepacked_source_test";
        fs::create_directories(tmp_dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmp_dir_, ec);
    }

    /// Prepack experts and return the output directory path.
    fs::path prepack(const std::string& name,
                     int n_layers, int n_experts,
                     int hidden, int intermediate,
                     int first_moe_layer,
                     SafetensorsDtype dtype) {
        ExpertShape shape{hidden, intermediate};
        auto cfg = make_config(n_layers, n_experts, hidden, intermediate, first_moe_layer);
        ModelConfig model_cfg(cfg);

        const QuantInterface* quant_ptr = nullptr;
        Nvfp4 nvfp4_quant;
        Fp8E4M3 fp8_quant;
        if (dtype == SafetensorsDtype::U8) {
            quant_ptr = &nvfp4_quant;
        } else {
            quant_ptr = &fp8_quant;
        }

        auto model = make_mock_model(n_layers, n_experts, shape,
                                     first_moe_layer, dtype);
        auto out = tmp_dir_ / name;
        auto result = prepack_experts(model, model_cfg, *quant_ptr, cfg, out);
        EXPECT_TRUE(result.error.empty()) << result.error;
        return out;
    }

    fs::path tmp_dir_;
};

// ── Resolve Tests ─────────────────────────────────────────────────────────

TEST_F(PrepackedSourceTest, ResolveCorrectPointer_NVFP4) {
    constexpr int kLayers = 4, kFirstMoe = 2, kExperts = 4;
    constexpr int kHidden = 64, kIntermediate = 32;

    auto dir = prepack("nvfp4", kLayers, kExperts, kHidden, kIntermediate,
                       kFirstMoe, SafetensorsDtype::U8);

    Nvfp4 quant;
    PrepackedSource src(dir, quant);

    EXPECT_EQ(src.num_expert_files(), kExperts);
    EXPECT_EQ(src.manifest().quant_format,
              std::string{prepacked::kNvfp4PackedQuantFormat});

    // All MoE experts should resolve to non-null.
    for (int l = kFirstMoe; l < kLayers; ++l) {
        for (int e = 0; e < kExperts; ++e) {
            layerstorm::memory::ExpertKey key{
                static_cast<uint32_t>(l), static_cast<uint16_t>(e)};
            const void* ptr = src.resolve(key);
            ASSERT_NE(ptr, nullptr) << "layer=" << l << " expert=" << e;
        }
    }
}

TEST_F(PrepackedSourceTest, ResolveCorrectPointer_FP8) {
    constexpr int kLayers = 4, kFirstMoe = 2, kExperts = 4;
    constexpr int kHidden = 64, kIntermediate = 32;

    auto dir = prepack("fp8", kLayers, kExperts, kHidden, kIntermediate,
                       kFirstMoe, SafetensorsDtype::F8_E4M3);

    Fp8E4M3 quant;
    PrepackedSource src(dir, quant);

    EXPECT_EQ(src.num_expert_files(), kExperts);
    EXPECT_EQ(src.manifest().quant_format, "fp8_e4m3");

    for (int l = kFirstMoe; l < kLayers; ++l) {
        for (int e = 0; e < kExperts; ++e) {
            layerstorm::memory::ExpertKey key{
                static_cast<uint32_t>(l), static_cast<uint16_t>(e)};
            ASSERT_NE(src.resolve(key), nullptr)
                << "layer=" << l << " expert=" << e;
        }
    }
}

TEST_F(PrepackedSourceTest, MultipleLayerOffsets) {
    constexpr int kLayers = 6, kFirstMoe = 2, kExperts = 2;
    constexpr int kHidden = 64, kIntermediate = 32;

    auto dir = prepack("offsets", kLayers, kExperts, kHidden, kIntermediate,
                       kFirstMoe, SafetensorsDtype::U8);

    Nvfp4 quant;
    PrepackedSource src(dir, quant);

    // resolve() steps layer slots by the on-disk STRIDE (padded,
    // kSlotAlignBytes-aligned), not the unpadded slot_size_bytes. Offsets must
    // be spaced by stride_bytes.
    int64_t stride = src.manifest().slot.stride_bytes;
    ASSERT_GT(stride, 0);

    // For expert 0, verify that layer offsets are strictly increasing
    // and spaced by the on-disk slot stride.
    const void* base = src.resolve({static_cast<uint32_t>(kFirstMoe), 0});
    ASSERT_NE(base, nullptr);

    int n_moe = kLayers - kFirstMoe;
    for (int i = 1; i < n_moe; ++i) {
        uint32_t layer_idx = static_cast<uint32_t>(kFirstMoe + i);
        const void* ptr = src.resolve({layer_idx, 0});
        ASSERT_NE(ptr, nullptr) << "layer_pos=" << i;

        auto diff = static_cast<const char*>(ptr)
                  - static_cast<const char*>(base);
        EXPECT_EQ(diff, i * stride)
            << "layer_pos=" << i << " expected offset="
            << (i * stride) << " got=" << diff;
    }
}

TEST_F(PrepackedSourceTest, ResolveNonMoeLayer_ReturnsNull) {
    constexpr int kLayers = 4, kFirstMoe = 2, kExperts = 2;
    constexpr int kHidden = 64, kIntermediate = 32;

    auto dir = prepack("dense", kLayers, kExperts, kHidden, kIntermediate,
                       kFirstMoe, SafetensorsDtype::U8);

    Nvfp4 quant;
    PrepackedSource src(dir, quant);

    // Dense layers (0, 1) should return nullptr.
    EXPECT_EQ(src.resolve({0, 0}), nullptr);
    EXPECT_EQ(src.resolve({1, 0}), nullptr);
}

TEST_F(PrepackedSourceTest, ResolveOutOfRange_ReturnsNull) {
    constexpr int kLayers = 4, kFirstMoe = 2, kExperts = 2;
    constexpr int kHidden = 64, kIntermediate = 32;

    auto dir = prepack("range", kLayers, kExperts, kHidden, kIntermediate,
                       kFirstMoe, SafetensorsDtype::U8);

    Nvfp4 quant;
    PrepackedSource src(dir, quant);

    // expert_idx out of range.
    EXPECT_EQ(src.resolve({static_cast<uint32_t>(kFirstMoe), 999}), nullptr);
    // layer_idx way out of range.
    EXPECT_EQ(src.resolve({999, 0}), nullptr);
}

TEST_F(PrepackedSourceTest, Has_ReturnsTrueForValid) {
    constexpr int kLayers = 4, kFirstMoe = 2, kExperts = 2;
    constexpr int kHidden = 64, kIntermediate = 32;

    auto dir = prepack("has", kLayers, kExperts, kHidden, kIntermediate,
                       kFirstMoe, SafetensorsDtype::U8);

    Nvfp4 quant;
    PrepackedSource src(dir, quant);

    // Valid keys.
    EXPECT_TRUE(src.has({static_cast<uint32_t>(kFirstMoe), 0}));
    EXPECT_TRUE(src.has({static_cast<uint32_t>(kFirstMoe), 1}));
    EXPECT_TRUE(src.has({static_cast<uint32_t>(kLayers - 1), 0}));

    // Invalid keys.
    EXPECT_FALSE(src.has({0, 0}));          // Dense layer
    EXPECT_FALSE(src.has({1, 0}));          // Dense layer
    EXPECT_FALSE(src.has({static_cast<uint32_t>(kFirstMoe), 999}));  // Bad expert
}

// ── Error Handling Tests ──────────────────────────────────────────────────

TEST_F(PrepackedSourceTest, ManifestValidationFailure_WrongQuant) {
    constexpr int kLayers = 4, kFirstMoe = 2, kExperts = 2;
    constexpr int kHidden = 64, kIntermediate = 32;

    // Prepack as NVFP4.
    auto dir = prepack("wrong_quant", kLayers, kExperts, kHidden, kIntermediate,
                       kFirstMoe, SafetensorsDtype::U8);

    // Try to open with FP8 quant — should throw.
    Fp8E4M3 fp8_quant;
    EXPECT_THROW(PrepackedSource(dir, fp8_quant), std::runtime_error);
}

TEST_F(PrepackedSourceTest, MissingFile_Throws) {
    constexpr int kLayers = 4, kFirstMoe = 2, kExperts = 2;
    constexpr int kHidden = 64, kIntermediate = 32;

    auto dir = prepack("missing_file", kLayers, kExperts, kHidden, kIntermediate,
                       kFirstMoe, SafetensorsDtype::U8);

    // Delete one expert file.
    fs::remove(prepacked::expert_file_path(dir, 0));

    Nvfp4 quant;
    EXPECT_THROW(PrepackedSource(dir, quant), std::runtime_error);
}

TEST_F(PrepackedSourceTest, CorruptFileSize_Throws) {
    constexpr int kLayers = 4, kFirstMoe = 2, kExperts = 2;
    constexpr int kHidden = 64, kIntermediate = 32;

    auto dir = prepack("corrupt_size", kLayers, kExperts, kHidden, kIntermediate,
                       kFirstMoe, SafetensorsDtype::U8);

    // Truncate one expert file.
    auto path = prepacked::expert_file_path(dir, 0);
    {
        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        ofs << "short";
    }

    Nvfp4 quant;
    EXPECT_THROW(PrepackedSource(dir, quant), std::runtime_error);
}

TEST_F(PrepackedSourceTest, MissingManifest_Throws) {
    auto dir = tmp_dir_ / "no_manifest";
    fs::create_directories(dir);

    Nvfp4 quant;
    EXPECT_THROW(PrepackedSource(dir, quant), std::runtime_error);
}

TEST_F(PrepackedSourceTest, DataMatchesFileContent) {
    constexpr int kLayers = 4, kFirstMoe = 2, kExperts = 2;
    constexpr int kHidden = 64, kIntermediate = 32;

    auto dir = prepack("data_match", kLayers, kExperts, kHidden, kIntermediate,
                       kFirstMoe, SafetensorsDtype::U8);

    Nvfp4 quant;
    PrepackedSource src(dir, quant);

    // Real packed bytes per slot (what resolve() exposes and what round-trips);
    // the on-disk slot stride is padded to kSlotAlignBytes, so file offsets must
    // step by stride_bytes while only slot_size_bytes of real content compares.
    int64_t slot_size = src.manifest().slot.slot_size_bytes;
    int64_t stride = src.manifest().slot.stride_bytes;

    // Read expert file 0 directly and compare with resolve() pointers.
    auto file_path = prepacked::expert_file_path(dir, 0);
    std::ifstream ifs(file_path, std::ios::binary);
    std::vector<char> file_data(
        (std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>());

    int n_moe = kLayers - kFirstMoe;
    for (int layer_pos = 0; layer_pos < n_moe; ++layer_pos) {
        uint32_t layer_idx = static_cast<uint32_t>(kFirstMoe + layer_pos);
        const void* ptr = src.resolve({layer_idx, 0});
        ASSERT_NE(ptr, nullptr);

        int64_t offset = prepacked::slot_offset(layer_pos, stride);
        ASSERT_EQ(std::memcmp(ptr, file_data.data() + offset,
                              static_cast<size_t>(slot_size)), 0)
            << "Data mismatch at layer_pos=" << layer_pos;
    }
}
