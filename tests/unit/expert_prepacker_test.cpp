#include <array>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "model/quantization/nvfp4.h"
#include "model/quantization/fp8.h"
#include "model/model_config.h"
#include "model/weight_pipeline/expert_prepacker.h"
#include "model/weight_pipeline/manifest.h"
#include "model/weight_pipeline/prepacked_format.h"
#include "model/weight_pipeline/prepacked_source.h"
#include "model/weight_loader/nvfp4_sfb_reformat.h"
#include "model/weight_loader/weight_loader.h"
#include "weight_pipeline_test_helpers.h"

using namespace layerstorm::model;
namespace fs = std::filesystem;

// ── Test Fixture ───────────────────────────────────────────────────────────

class ExpertPrepackerTest : public ::testing::Test,
                            public layerstorm::test::WeightPipelineHelpers {
protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "layerstorm_prepacker_test";
        fs::create_directories(tmp_dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmp_dir_, ec);
    }

    fs::path tmp_dir_;
};

// ── NVFP4 Mock Model ───────────────────────────────────────────────────────

TEST_F(ExpertPrepackerTest, MockNvfp4Model) {
    constexpr int kLayers = 4;
    constexpr int kFirstMoe = 2;
    constexpr int kExperts = 4;
    constexpr int kHidden = 64;
    constexpr int kIntermediate = 32;

    ExpertShape shape{kHidden, kIntermediate};
    Nvfp4 quant;

    auto cfg = make_config(kLayers, kExperts, kHidden, kIntermediate, kFirstMoe);
    ModelConfig model_cfg(cfg);
    auto model = make_mock_model(kLayers, kExperts, shape, kFirstMoe,
                                 SafetensorsDtype::U8);

    auto output = tmp_dir_ / "nvfp4_out";
    auto result = prepack_experts(model, model_cfg, quant, cfg, output);

    EXPECT_TRUE(result.error.empty()) << result.error;
    EXPECT_EQ(result.experts_written, kExperts);
    EXPECT_EQ(result.experts_skipped, 0);

    // Verify file existence and sizes. Files are written at the padded on-disk
    // stride (kSlotAlignBytes-aligned per slot), so the expected file size must
    // use aligned_slot_stride, NOT the unpadded bytes_per_expert.
    const int n_moe = kLayers - kFirstMoe;  // 2 MoE layers
    int64_t slot_size = quant.bytes_per_expert(shape);
    int64_t slot_stride = prepacked::aligned_slot_stride(slot_size);
    int64_t expected_size = prepacked::expected_file_size(n_moe, slot_stride);
    EXPECT_GT(expected_size, 0);

    for (int e = 0; e < kExperts; ++e) {
        auto path = prepacked::expert_file_path(output, e);
        EXPECT_TRUE(fs::exists(path)) << "Missing: " << path;
        EXPECT_EQ(fs::file_size(path), static_cast<uintmax_t>(expected_size))
            << "Expert " << e << " size mismatch";
    }

    // Verify manifest round-trips and verifies.
    auto manifest = read_manifest(output);
    EXPECT_EQ(manifest.quant_format,
              std::string{prepacked::kNvfp4PackedQuantFormat});
    EXPECT_EQ(manifest.n_routed_experts, kExperts);
    EXPECT_EQ(manifest.n_expert_files, kExperts);
    EXPECT_EQ(manifest.moe_layers.count, n_moe);
    EXPECT_EQ(manifest.expert_dimensions.hidden_size, kHidden);
    EXPECT_EQ(manifest.expert_dimensions.intermediate_size, kIntermediate);
    EXPECT_EQ(manifest.slot.slot_size_bytes, slot_size);

    auto verify_result = verify_manifest(manifest, quant);
    EXPECT_TRUE(verify_result.ok) << verify_result.error;
}

// ── FP8 Mock Model ─────────────────────────────────────────────────────────

TEST_F(ExpertPrepackerTest, MockFp8Model) {
    constexpr int kLayers = 4;
    constexpr int kFirstMoe = 2;
    constexpr int kExperts = 4;
    constexpr int kHidden = 64;
    constexpr int kIntermediate = 32;

    ExpertShape shape{kHidden, kIntermediate};
    Fp8E4M3 quant;

    auto cfg = make_config(kLayers, kExperts, kHidden, kIntermediate, kFirstMoe);
    ModelConfig model_cfg(cfg);
    auto model = make_mock_model(kLayers, kExperts, shape, kFirstMoe,
                                 SafetensorsDtype::F8_E4M3);

    auto output = tmp_dir_ / "fp8_out";
    auto result = prepack_experts(model, model_cfg, quant, cfg, output);

    EXPECT_TRUE(result.error.empty()) << result.error;
    EXPECT_EQ(result.experts_written, kExperts);
    EXPECT_EQ(result.experts_skipped, 0);

    const int n_moe = kLayers - kFirstMoe;
    int64_t slot_size = quant.bytes_per_expert(shape);
    // On-disk file size uses the padded (kSlotAlignBytes-aligned) slot stride.
    int64_t slot_stride = prepacked::aligned_slot_stride(slot_size);
    int64_t expected_size = prepacked::expected_file_size(n_moe, slot_stride);

    for (int e = 0; e < kExperts; ++e) {
        auto path = prepacked::expert_file_path(output, e);
        EXPECT_TRUE(fs::exists(path)) << "Missing: " << path;
        EXPECT_EQ(fs::file_size(path), static_cast<uintmax_t>(expected_size))
            << "Expert " << e << " size mismatch";
    }

    auto manifest = read_manifest(output);
    EXPECT_EQ(manifest.quant_format, "fp8_e4m3");
    EXPECT_EQ(manifest.slot.slot_size_bytes, slot_size);

    auto verify_result = verify_manifest(manifest, quant);
    EXPECT_TRUE(verify_result.ok) << verify_result.error;
}

// ── Resumable ──────────────────────────────────────────────────────────────

TEST_F(ExpertPrepackerTest, Resumable) {
    constexpr int kLayers = 3;
    constexpr int kFirstMoe = 1;
    constexpr int kExperts = 3;
    constexpr int kHidden = 64;
    constexpr int kIntermediate = 32;

    ExpertShape shape{kHidden, kIntermediate};
    Nvfp4 quant;

    auto cfg = make_config(kLayers, kExperts, kHidden, kIntermediate, kFirstMoe);
    ModelConfig model_cfg(cfg);

    auto output = tmp_dir_ / "resume_out";

    // First run.
    {
        auto model = make_mock_model(kLayers, kExperts, shape, kFirstMoe,
                                     SafetensorsDtype::U8);
        auto r = prepack_experts(model, model_cfg, quant, cfg, output);
        EXPECT_TRUE(r.error.empty()) << r.error;
        EXPECT_EQ(r.experts_written, kExperts);
        EXPECT_EQ(r.experts_skipped, 0);
    }

    // Second run — all should be skipped.
    {
        auto model = make_mock_model(kLayers, kExperts, shape, kFirstMoe,
                                     SafetensorsDtype::U8);
        auto r = prepack_experts(model, model_cfg, quant, cfg, output);
        EXPECT_TRUE(r.error.empty()) << r.error;
        EXPECT_EQ(r.experts_written, 0);
        EXPECT_EQ(r.experts_skipped, kExperts);
    }
}

// ── Partial Resume ─────────────────────────────────────────────────────────

TEST_F(ExpertPrepackerTest, PartialResume) {
    constexpr int kLayers = 3;
    constexpr int kFirstMoe = 1;
    constexpr int kExperts = 4;
    constexpr int kHidden = 64;
    constexpr int kIntermediate = 32;

    ExpertShape shape{kHidden, kIntermediate};
    Nvfp4 quant;

    auto cfg = make_config(kLayers, kExperts, kHidden, kIntermediate, kFirstMoe);
    ModelConfig model_cfg(cfg);

    auto output = tmp_dir_ / "partial_out";

    // First run — creates all files.
    {
        auto model = make_mock_model(kLayers, kExperts, shape, kFirstMoe,
                                     SafetensorsDtype::U8);
        auto r = prepack_experts(model, model_cfg, quant, cfg, output);
        ASSERT_TRUE(r.error.empty()) << r.error;
        ASSERT_EQ(r.experts_written, kExperts);
    }

    // Delete expert 1.
    fs::remove(prepacked::expert_file_path(output, 1));

    // Second run — should only rewrite expert 1.
    {
        auto model = make_mock_model(kLayers, kExperts, shape, kFirstMoe,
                                     SafetensorsDtype::U8);
        auto r = prepack_experts(model, model_cfg, quant, cfg, output);
        EXPECT_TRUE(r.error.empty()) << r.error;
        EXPECT_EQ(r.experts_written, 1);
        EXPECT_EQ(r.experts_skipped, kExperts - 1);
    }
}

// ── Corrupt Size Re-Packs ──────────────────────────────────────────────────

TEST_F(ExpertPrepackerTest, CorruptSizeRePacks) {
    constexpr int kLayers = 3;
    constexpr int kFirstMoe = 1;
    constexpr int kExperts = 3;
    constexpr int kHidden = 64;
    constexpr int kIntermediate = 32;

    ExpertShape shape{kHidden, kIntermediate};
    Nvfp4 quant;

    auto cfg = make_config(kLayers, kExperts, kHidden, kIntermediate, kFirstMoe);
    ModelConfig model_cfg(cfg);

    auto output = tmp_dir_ / "corrupt_out";

    // First run.
    {
        auto model = make_mock_model(kLayers, kExperts, shape, kFirstMoe,
                                     SafetensorsDtype::U8);
        auto r = prepack_experts(model, model_cfg, quant, cfg, output);
        ASSERT_TRUE(r.error.empty()) << r.error;
    }

    // Truncate expert 2 to wrong size.
    {
        auto path = prepacked::expert_file_path(output, 2);
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write("x", 1);
    }

    // Second run — should rewrite expert 2.
    {
        auto model = make_mock_model(kLayers, kExperts, shape, kFirstMoe,
                                     SafetensorsDtype::U8);
        auto r = prepack_experts(model, model_cfg, quant, cfg, output);
        EXPECT_TRUE(r.error.empty()) << r.error;
        EXPECT_EQ(r.experts_written, 1);
        EXPECT_EQ(r.experts_skipped, kExperts - 1);
    }
}

// ── Slot Content Verification ──────────────────────────────────────────────

TEST_F(ExpertPrepackerTest, SlotContentMatchesPacking) {
    // Verify that the file content matches what pack_nvfp4_expert produces.
    constexpr int kLayers = 2;
    constexpr int kFirstMoe = 0;
    constexpr int kExperts = 1;
    constexpr int kHidden = 64;
    constexpr int kIntermediate = 32;

    ExpertShape shape{kHidden, kIntermediate};
    Nvfp4 quant;

    auto cfg = make_config(kLayers, kExperts, kHidden, kIntermediate, kFirstMoe);
    ModelConfig model_cfg(cfg);

    // Create two separate models: one for prepacking, one for reference packing.
    auto model = make_mock_model(kLayers, kExperts, shape, kFirstMoe,
                                 SafetensorsDtype::U8);

    // Reference: pack layer 0, expert 0 manually.
    auto ref_bundles = make_nvfp4_bundles(shape);
    pack_nvfp4_expert(ref_bundles, shape);
    auto ref_data = ref_bundles[0].packed_slot;
    ASSERT_EQ(static_cast<int64_t>(ref_data.size()), quant.bytes_per_expert(shape));

    auto output = tmp_dir_ / "content_out";
    auto result = prepack_experts(model, model_cfg, quant, cfg, output);
    ASSERT_TRUE(result.error.empty()) << result.error;

    // Read back the file and compare the first slot (layer 0) against reference.
    auto path = prepacked::expert_file_path(output, 0);
    std::ifstream ifs(path, std::ios::binary);
    ASSERT_TRUE(ifs.good());

    std::vector<std::byte> file_data(quant.bytes_per_expert(shape));
    ifs.read(reinterpret_cast<char*>(file_data.data()), file_data.size());
    ASSERT_TRUE(ifs.good());

    // The data should be identical because both were packed from
    // identically-structured synthetic bundles with the same fill values.
    EXPECT_EQ(std::memcmp(ref_data.data(), file_data.data(), ref_data.size()), 0)
        << "File content does not match reference packing";
}

// ── nvfp4-sm1xx pack-time scale reformat + input_scale normalization ───────

// Packed slot group scales must be Sm1xx-interleaved (format 9.67.0): for an
// alignable shape, each projection's scale region equals reformat_nvfp4_sfb
// of the raw bytes. Random bytes — interleave bugs are invisible to uniform.
TEST_F(ExpertPrepackerTest, PackWritesSm1xxInterleavedScales) {
    // Both scale matrices alignable: gate/up [I=128, H/16=8], down [H=128, I/16=8].
    constexpr int kHidden = 128, kIntermediate = 128;
    ExpertShape shape{kHidden, kIntermediate};

    auto bundles = make_nvfp4_bundles(shape, /*scale_seed=*/0xC0FFEE);

    // Capture RAW scale bytes per projection before packing.
    auto raw_scales = [&](int proj_idx) {
        const auto* ws = bundles[static_cast<size_t>(proj_idx)].find_aux(
            TensorRole::weight_scale);
        EXPECT_NE(ws, nullptr);
        return std::vector<uint8_t>(
            reinterpret_cast<const uint8_t*>(ws->data.data()),
            reinterpret_cast<const uint8_t*>(ws->data.data()) + ws->data.size());
    };
    const std::array<std::vector<uint8_t>, 3> raw{
        raw_scales(0), raw_scales(1), raw_scales(2)};

    pack_nvfp4_expert(bundles, shape);
    const auto& slot = bundles[0].packed_slot;
    ASSERT_FALSE(slot.empty());

    Nvfp4 quant;
    const struct { Projection p; int64_t rows, groups; } projs[3] = {
        {Projection::gate, kIntermediate, kHidden / 16},
        {Projection::up,   kIntermediate, kHidden / 16},
        {Projection::down, kHidden,       kIntermediate / 16},
    };
    int64_t off = 0;
    for (int i = 0; i < 3; ++i) {
        const int64_t wb = quant.weight_bytes_per_projection(shape, projs[i].p);
        const int64_t total = quant.bytes_per_projection(shape, projs[i].p);
        const int64_t sb = projs[i].rows * projs[i].groups;
        std::vector<uint8_t> expected(static_cast<size_t>(sb));
        reformat_nvfp4_sfb(expected.data(), raw[static_cast<size_t>(i)].data(),
                           projs[i].rows, projs[i].groups);
        EXPECT_EQ(std::memcmp(slot.data() + off + wb, expected.data(),
                              static_cast<size_t>(sb)), 0)
            << "projection " << i << " scales not Sm1xx-interleaved";
        // Raw layout must NOT appear (the permutation is non-identity for
        // random bytes over >1 tile row).
        EXPECT_NE(std::memcmp(slot.data() + off + wb,
                              raw[static_cast<size_t>(i)].data(),
                              static_cast<size_t>(sb)), 0)
            << "projection " << i << " scales left RAW";
        off += total;
    }
}

// input_scale scalars must be normalized: gate == up == fc31 (layer max over
// gate∪up), down == fc2 (layer max over down) — per-slot last-8-bytes layout
// [ws2 | is] unchanged.
TEST_F(ExpertPrepackerTest, PackNormalizesInputScales) {
    constexpr int kHidden = 128, kIntermediate = 128;
    ExpertShape shape{kHidden, kIntermediate};
    Nvfp4 quant;

    // Two experts with diverging calibrations.
    std::vector<std::vector<WeightBundle>> layer_experts;
    layer_experts.push_back(
        make_nvfp4_bundles(shape, 0, /*gate=*/0.002f, /*up=*/0.002f, /*down=*/0.010f));
    layer_experts.push_back(
        make_nvfp4_bundles(shape, 0, /*gate=*/0.005f, /*up=*/0.005f, /*down=*/0.004f));

    auto norm = compute_nvfp4_input_scale_norm(layer_experts, /*layer_idx=*/0);
    EXPECT_FLOAT_EQ(norm.fc31, 0.005f);
    EXPECT_FLOAT_EQ(norm.fc2, 0.010f);

    auto read_scalar = [&](const std::span<const std::byte>& slot,
                           Projection p, int back_off) {
        int64_t end = 0;
        for (Projection q : {Projection::gate, Projection::up, Projection::down}) {
            end += quant.bytes_per_projection(shape, q);
            if (q == p) break;
        }
        float v;
        std::memcpy(&v, slot.data() + end - back_off, sizeof(float));
        return v;
    };

    for (auto& bundles : layer_experts) {
        pack_nvfp4_expert(bundles, shape, &norm);
        const auto& slot = bundles[0].packed_slot;
        ASSERT_FALSE(slot.empty());
        EXPECT_FLOAT_EQ(read_scalar(slot, Projection::gate, 4), 0.005f);
        EXPECT_FLOAT_EQ(read_scalar(slot, Projection::up, 4), 0.005f);
        EXPECT_FLOAT_EQ(read_scalar(slot, Projection::down, 4), 0.010f);
        // ws2 (at end-8) untouched by normalization.
        EXPECT_FLOAT_EQ(read_scalar(slot, Projection::gate, 8), 1.0f);
    }
}

// Without a layer norm, pack still merges gate/up locally (consistency with
// the shared quantized activation is a hard requirement).
TEST_F(ExpertPrepackerTest, PackLocalGateUpMergeWithoutNorm) {
    constexpr int kHidden = 128, kIntermediate = 128;
    ExpertShape shape{kHidden, kIntermediate};
    Nvfp4 quant;

    auto bundles = make_nvfp4_bundles(shape, 0, /*gate=*/0.003f, /*up=*/0.007f,
                                      /*down=*/0.002f);
    pack_nvfp4_expert(bundles, shape);
    const auto& slot = bundles[0].packed_slot;
    ASSERT_FALSE(slot.empty());

    const int64_t gate_total = quant.bytes_per_projection(shape, Projection::gate);
    const int64_t up_total = quant.bytes_per_projection(shape, Projection::up);
    float gate_is, up_is;
    std::memcpy(&gate_is, slot.data() + gate_total - 4, sizeof(float));
    std::memcpy(&up_is, slot.data() + gate_total + up_total - 4, sizeof(float));
    EXPECT_FLOAT_EQ(gate_is, 0.007f);  // max(gate, up)
    EXPECT_FLOAT_EQ(up_is, 0.007f);
}

// ── No MoE Layers ──────────────────────────────────────────────────────────

TEST_F(ExpertPrepackerTest, NoMoeLayersProducesNothing) {
    // All layers are dense — no MoE layers.
    constexpr int kLayers = 3;
    constexpr int kFirstMoe = 99;  // beyond num_hidden_layers
    constexpr int kExperts = 4;
    constexpr int kHidden = 64;
    constexpr int kIntermediate = 32;

    ExpertShape shape{kHidden, kIntermediate};
    Nvfp4 quant;

    auto cfg = make_config(kLayers, kExperts, kHidden, kIntermediate, kFirstMoe);
    ModelConfig model_cfg(cfg);

    LoadedModel model;
    model.layers.resize(kLayers);

    auto output = tmp_dir_ / "no_moe_out";
    auto result = prepack_experts(model, model_cfg, quant, cfg, output);

    EXPECT_TRUE(result.error.empty()) << result.error;
    EXPECT_EQ(result.experts_written, 0);
    EXPECT_EQ(result.experts_skipped, 0);
}

// ── ensure_expert_packed Lifecycle (TD-93e, TD-93h, TD-93b) ────────────────

TEST_F(ExpertPrepackerTest, EnsureExpertPackedNvfp4Lifecycle) {
    ExpertShape shape{64, 32};
    auto bundles = make_nvfp4_bundles(shape);

    // Initially unpacked: packed_slot empty, weight.data holds raw gate only.
    EXPECT_TRUE(bundles[0].packed_slot.empty());
    EXPECT_FALSE(bundles[0].weight.data.empty());

    // Pack.
    ensure_expert_packed(bundles, shape);
    EXPECT_TRUE(bundles[0].owned_buf);
    EXPECT_EQ(static_cast<int64_t>(bundles[0].packed_slot.size()),
              Nvfp4{}.bytes_per_expert(shape));
    // Raw weight.data is untouched.
    EXPECT_FALSE(bundles[0].weight.data.empty());

    // Second call is a no-op (packed_slot guard).
    auto* prev_ptr = bundles[0].packed_slot.data();
    ensure_expert_packed(bundles, shape);
    EXPECT_EQ(bundles[0].packed_slot.data(), prev_ptr);

    // Release (simulates post-H2D cleanup).
    bundles[0].owned_buf.reset();
    bundles[0].packed_slot = {};
    EXPECT_TRUE(bundles[0].packed_slot.empty());
    // Raw weight.data survives release (TD-93h fix).
    EXPECT_FALSE(bundles[0].weight.data.empty());

    // Re-pack after release: reads from preserved weight.data → correct.
    ensure_expert_packed(bundles, shape);
    EXPECT_TRUE(bundles[0].owned_buf);
    EXPECT_EQ(static_cast<int64_t>(bundles[0].packed_slot.size()),
              Nvfp4{}.bytes_per_expert(shape));
}

TEST_F(ExpertPrepackerTest, EnsureExpertPackedFp8Lifecycle) {
    ExpertShape shape{64, 32};
    auto bundles = make_fp8_bundles(shape);

    // Initially unpacked.
    EXPECT_TRUE(bundles[0].packed_slot.empty());
    EXPECT_FALSE(bundles[0].weight.data.empty());

    // Pack (non-contiguous synthetic data → allocates owned_buf).
    ensure_expert_packed(bundles, shape);
    EXPECT_TRUE(bundles[0].owned_buf);
    EXPECT_EQ(static_cast<int64_t>(bundles[0].packed_slot.size()),
              Fp8E4M3{}.bytes_per_expert(shape));
    EXPECT_FALSE(bundles[0].weight.data.empty());

    // Second call is a no-op (packed_slot guard, fixes TD-93b).
    auto* prev_ptr = bundles[0].packed_slot.data();
    ensure_expert_packed(bundles, shape);
    EXPECT_EQ(bundles[0].packed_slot.data(), prev_ptr);

    // Release.
    bundles[0].owned_buf.reset();
    bundles[0].packed_slot = {};
    EXPECT_TRUE(bundles[0].packed_slot.empty());
    EXPECT_FALSE(bundles[0].weight.data.empty());

    // Re-pack after release: reads from preserved weight.data → correct.
    ensure_expert_packed(bundles, shape);
    EXPECT_TRUE(bundles[0].owned_buf);
    EXPECT_EQ(static_cast<int64_t>(bundles[0].packed_slot.size()),
              Fp8E4M3{}.bytes_per_expert(shape));
}

TEST_F(ExpertPrepackerTest, EnsureExpertPackedIdempotent) {
    // Verify that calling ensure_expert_packed twice is idempotent
    // (packed_slot guard prevents re-entry for both formats).
    ExpertShape shape{64, 32};

    // NVFP4
    {
        auto bundles = make_nvfp4_bundles(shape);
        ensure_expert_packed(bundles, shape);
        auto* ptr1 = bundles[0].packed_slot.data();
        ensure_expert_packed(bundles, shape);
        EXPECT_EQ(bundles[0].packed_slot.data(), ptr1);
    }

    // FP8
    {
        auto bundles = make_fp8_bundles(shape);
        ensure_expert_packed(bundles, shape);
        auto* ptr1 = bundles[0].packed_slot.data();
        ensure_expert_packed(bundles, shape);
        EXPECT_EQ(bundles[0].packed_slot.data(), ptr1);
    }
}

// ── WP-6 guard tests (TD-97b, TD-97c) ────────────────────────────────────

TEST_F(ExpertPrepackerTest, PrepackEmptyRoutedExpertsFails) {
    // TD-97b: prepack_experts with empty routed_experts must fail, not crash.
    // This is the scenario the engine init guard protects against.
    constexpr int kLayers = 4;
    constexpr int kFirstMoe = 2;
    constexpr int kExperts = 4;
    constexpr int kHidden = 64;
    constexpr int kIntermediate = 32;

    ExpertShape shape{kHidden, kIntermediate};
    Nvfp4 quant;

    auto cfg = make_config(kLayers, kExperts, kHidden, kIntermediate, kFirstMoe);
    ModelConfig model_cfg(cfg);

    // Model with MoE layers but empty routed_experts (simulates WP-6 skip).
    LoadedModel model;
    model.layers.resize(kLayers);
    for (int l = 0; l < kLayers; ++l) {
        model.layers[l].layer_idx = l;
        // Deliberately leave routed_experts empty — WP-6 skip scenario.
    }

    auto output = tmp_dir_ / "empty_experts_out";
    auto result = prepack_experts(model, model_cfg, quant, cfg, output);

    // Must produce an error, not crash.
    EXPECT_FALSE(result.error.empty())
        << "prepack_experts should fail with empty routed_experts";
    EXPECT_EQ(result.experts_written, 0);
}

TEST_F(ExpertPrepackerTest, PrepackedSourceThrowsOnQuantMismatch) {
    // TD-97c: PrepackedSource init fails when manifest quant doesn't match.
    // This is the scenario the engine init guard protects against.
    constexpr int kLayers = 4;
    constexpr int kFirstMoe = 2;
    constexpr int kExperts = 4;
    constexpr int kHidden = 64;
    constexpr int kIntermediate = 32;

    ExpertShape shape{kHidden, kIntermediate};
    Nvfp4 quant;

    auto cfg = make_config(kLayers, kExperts, kHidden, kIntermediate, kFirstMoe);
    ModelConfig model_cfg(cfg);
    auto model = make_mock_model(kLayers, kExperts, shape, kFirstMoe,
                                 SafetensorsDtype::U8);

    // Prepack with NVFP4.
    auto output = tmp_dir_ / "quant_mismatch_out";
    auto result = prepack_experts(model, model_cfg, quant, cfg, output);
    ASSERT_TRUE(result.error.empty()) << result.error;

    // Try to open with FP8 quant — manifest says "nvfp4", mismatch → throw.
    Fp8E4M3 wrong_quant;
    EXPECT_THROW(PrepackedSource(output, wrong_quant), std::runtime_error);
}

// ── GGUF per-layer mixed k-quant types (GG-10) ──────────────────────────────

TEST_F(ExpertPrepackerTest, GgufPerLayerMixedTypesPackAndReadBack) {
    // Two MoE layers with DIFFERENT k-quant triples (Unsloth-Dynamic style):
    //   layer 2: gate/up Q4_K, down Q5_K
    //   layer 3: gate/up Q5_K, down Q6_K   (== the per-projection MAX triple)
    // The slot is sized at the global MAX; layer 2 packs smaller and its slot
    // tail must be zero padding.
    constexpr int kLayers = 4;
    constexpr int kFirstMoe = 2;
    constexpr int kExperts = 2;
    constexpr int kHidden = 256;        // QK=256 must divide both dims
    constexpr int kIntermediate = 512;

    ExpertShape shape{kHidden, kIntermediate};
    const GgufModelExpertTypes l2_types{
        GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q5_K};
    const GgufModelExpertTypes l3_types{
        GgufKQuantType::Q5_K, GgufKQuantType::Q5_K, GgufKQuantType::Q6_K};
    // Global slot-sizing interface = per-projection MAX (GG-9).
    GgufQuantInterface quant = make_gguf_quant(
        GgufKQuantType::Q5_K, GgufKQuantType::Q5_K, GgufKQuantType::Q6_K);

    auto cfg = make_config(kLayers, kExperts, kHidden, kIntermediate, kFirstMoe);
    ModelConfig model_cfg(cfg);

    LoadedModel model;
    model.layers.resize(kLayers);
    for (int l = 0; l < kLayers; ++l) model.layers[l].layer_idx = l;
    // Per-layer fill bytes make cross-layer/cross-projection mixups visible.
    model.layers[2].routed_experts.resize(kExperts);
    model.layers[3].routed_experts.resize(kExperts);
    for (int e = 0; e < kExperts; ++e) {
        model.layers[2].routed_experts[e] = make_gguf_bundles(
            shape, l2_types.gate, l2_types.up, l2_types.down,
            0xA2, 0xB2, 0xC2);
        model.layers[3].routed_experts[e] = make_gguf_bundles(
            shape, l3_types.gate, l3_types.up, l3_types.down,
            0xA3, 0xB3, 0xC3);
    }

    auto output = tmp_dir_ / "gguf_mixed_out";
    auto result = prepack_experts(model, model_cfg, quant, cfg, output);
    ASSERT_TRUE(result.error.empty()) << result.error;
    EXPECT_EQ(result.experts_written, kExperts);

    // Byte accounting.
    const int64_t slot_size = quant.bytes_per_expert(shape);
    const int64_t stride = prepacked::aligned_slot_stride(slot_size);
    GgufQuantInterface q2 =
        make_gguf_quant(l2_types.gate, l2_types.up, l2_types.down);
    const int64_t l2_gate = q2.bytes_per_projection(shape, Projection::gate);
    const int64_t l2_up   = q2.bytes_per_projection(shape, Projection::up);
    const int64_t l2_total = q2.bytes_per_expert(shape);
    ASSERT_LT(l2_total, slot_size);  // mixed layer really is smaller
    const int64_t l3_gate = quant.bytes_per_projection(shape, Projection::gate);
    const int64_t l3_up   = quant.bytes_per_projection(shape, Projection::up);
    const int64_t l3_total = quant.bytes_per_expert(shape);
    EXPECT_EQ(l3_total, slot_size);

    // File size = 2 MoE layers * stride.
    auto path = prepacked::expert_file_path(output, 0);
    ASSERT_TRUE(fs::exists(path));
    EXPECT_EQ(fs::file_size(path), static_cast<uintmax_t>(2 * stride));

    // Read back expert 0 raw and check per-layer offsets + zero padding.
    std::vector<char> file_data(static_cast<size_t>(2 * stride));
    {
        std::ifstream ifs(path, std::ios::binary);
        ASSERT_TRUE(ifs.read(file_data.data(),
                             static_cast<std::streamsize>(file_data.size())));
    }
    auto at = [&](int64_t off) {
        return static_cast<uint8_t>(file_data[static_cast<size_t>(off)]);
    };
    // Layer 2 (slot 0): gate|up|down back-to-back at the LAYER's sizes.
    EXPECT_EQ(at(0), 0xA2);
    EXPECT_EQ(at(l2_gate), 0xB2);
    EXPECT_EQ(at(l2_gate + l2_up), 0xC2);
    EXPECT_EQ(at(l2_total - 1), 0xC2);
    // Slot tail beyond the layer's real bytes is zero-padded up to the stride.
    for (int64_t off = l2_total; off < stride; ++off) {
        ASSERT_EQ(at(off), 0) << "slot 0 pad byte at " << off;
    }
    // Layer 3 (slot 1 at the aligned stride): the MAX triple fills the slot.
    EXPECT_EQ(at(stride), 0xA3);
    EXPECT_EQ(at(stride + l3_gate), 0xB3);
    EXPECT_EQ(at(stride + l3_gate + l3_up), 0xC3);
    EXPECT_EQ(at(stride + l3_total - 1), 0xC3);

    // Manifest: global MAX triple + ordered per-layer triples.
    auto manifest = read_manifest(output);
    EXPECT_EQ(manifest.format_version, std::string{prepacked::kFormatVersion});
    ASSERT_TRUE(manifest.gguf_types.has_value());
    EXPECT_EQ(manifest.gguf_types->gate, GgufKQuantType::Q5_K);
    EXPECT_EQ(manifest.gguf_types->down, GgufKQuantType::Q6_K);
    ASSERT_EQ(manifest.gguf_types_per_layer.size(), 2u);
    EXPECT_EQ(manifest.gguf_types_per_layer[0].gate, GgufKQuantType::Q4_K);
    EXPECT_EQ(manifest.gguf_types_per_layer[0].down, GgufKQuantType::Q5_K);
    EXPECT_EQ(manifest.gguf_types_per_layer[1].gate, GgufKQuantType::Q5_K);
    EXPECT_EQ(manifest.gguf_types_per_layer[1].down, GgufKQuantType::Q6_K);
    auto vr = verify_manifest(manifest, quant);
    EXPECT_TRUE(vr.ok) << vr.error;

    // PrepackedSource read side: per-layer triples by ABSOLUTE layer index.
    PrepackedSource src(output, quant);
    auto t2 = src.gguf_types_for_layer(2);
    ASSERT_TRUE(t2.has_value());
    EXPECT_EQ(t2->gate, GgufKQuantType::Q4_K);
    EXPECT_EQ(t2->up,   GgufKQuantType::Q4_K);
    EXPECT_EQ(t2->down, GgufKQuantType::Q5_K);
    auto t3 = src.gguf_types_for_layer(3);
    ASSERT_TRUE(t3.has_value());
    EXPECT_EQ(t3->gate, GgufKQuantType::Q5_K);
    EXPECT_EQ(t3->down, GgufKQuantType::Q6_K);
    EXPECT_FALSE(src.gguf_types_for_layer(1).has_value());  // dense layer
    EXPECT_FALSE(src.gguf_types_for_layer(9).has_value());  // out of range

    // resolve() sees the same per-layer layout (offsets from the LAYER triple).
    const auto* slot2 = static_cast<const uint8_t*>(
        src.resolve(layerstorm::memory::ExpertKey{2, 1}));
    ASSERT_NE(slot2, nullptr);
    EXPECT_EQ(slot2[0], 0xA2);
    EXPECT_EQ(slot2[l2_gate], 0xB2);
    EXPECT_EQ(slot2[l2_gate + l2_up], 0xC2);
    EXPECT_EQ(slot2[l2_total], 0x00);  // padded tail
}

TEST_F(ExpertPrepackerTest, GgufLayerExceedingSlotFails) {
    // If the "global" quant is NOT the per-projection MAX (a layer packs
    // bigger than the slot), prepack must fail loudly, not truncate.
    constexpr int kLayers = 3;
    constexpr int kFirstMoe = 2;
    constexpr int kExperts = 1;
    constexpr int kHidden = 256;
    constexpr int kIntermediate = 512;

    ExpertShape shape{kHidden, kIntermediate};
    GgufQuantInterface quant = make_gguf_quant(          // undersized global
        GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q4_K);

    auto cfg = make_config(kLayers, kExperts, kHidden, kIntermediate, kFirstMoe);
    ModelConfig model_cfg(cfg);

    LoadedModel model;
    model.layers.resize(kLayers);
    for (int l = 0; l < kLayers; ++l) model.layers[l].layer_idx = l;
    model.layers[2].routed_experts.resize(kExperts);
    model.layers[2].routed_experts[0] = make_gguf_bundles(
        shape, GgufKQuantType::Q6_K, GgufKQuantType::Q6_K, GgufKQuantType::Q6_K);

    auto output = tmp_dir_ / "gguf_oversize_out";
    auto result = prepack_experts(model, model_cfg, quant, cfg, output);
    EXPECT_FALSE(result.error.empty());
    EXPECT_NE(result.error.find("exceeds slot size"), std::string::npos)
        << result.error;
}
