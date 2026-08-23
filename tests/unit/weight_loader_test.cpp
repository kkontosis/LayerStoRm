#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "model/weight_loader/tensor_id.h"
#include "model/weight_loader/safetensors_reader.h"
#include "model/weight_loader/weight_handler.h"
#include "model/weight_loader/nvfp4_weight_handler.h"
#include "model/weight_loader/fp8_weight_handler.h"
#include "model/weight_loader/native_weight_handler.h"
#include "model/weight_loader/weight_loader.h"
#include "model/model_config.h"
#include "model/layer_registry.h"
#include "model/quantization/fp8.h"
#include "config/config_parser.h"

namespace fs = std::filesystem;
using layerstorm::model::TensorComponent;
using layerstorm::model::TensorRole;
using layerstorm::model::TensorOwner;
using layerstorm::model::TensorId;
using layerstorm::model::parse_hf_name;
using layerstorm::model::tensor_component_name;
using layerstorm::model::tensor_role_name;
using layerstorm::model::tensor_owner_name;
using layerstorm::model::SafetensorsDtype;
using layerstorm::model::SafetensorsReader;
using layerstorm::model::TensorEntry;
using layerstorm::model::read_shard_index;
using layerstorm::model::dtype_size;
using layerstorm::model::dtype_name;
using layerstorm::model::parse_dtype;
using layerstorm::model::handler_for_dtype;
using layerstorm::model::WeightBundle;
using layerstorm::model::RawTensor;
using layerstorm::model::NvFp4WeightHandler;
using layerstorm::model::Fp8WeightHandler;
using layerstorm::model::NativeWeightHandler;
using layerstorm::model::LayerRegistry;
using layerstorm::model::load_weights;
using layerstorm::config::Config;
using layerstorm::config::Architecture;
using layerstorm::config::WeightsFormat;
using layerstorm::config::WeightQuant;
using layerstorm::config::AttentionQuant;
using layerstorm::config::GatingQuant;
using layerstorm::config::GpuConfig;
using layerstorm::config::GpuType;
using MConfig = layerstorm::model::ModelConfig;

// ═══════════════════════════════════════════════════════════════════════════════
// TensorId parsing tests
// ═══════════════════════════════════════════════════════════════════════════════

class TensorIdTest : public ::testing::Test {};

// ── Attention projections ──

TEST_F(TensorIdTest, ParseQAProj) {
    auto id = parse_hf_name("model.layers.5.self_attn.q_a_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::q_a_proj);
    EXPECT_EQ(id->role, TensorRole::weight);
    EXPECT_EQ(id->owner, TensorOwner::attention);
    EXPECT_EQ(id->layer_idx, 5);
    EXPECT_EQ(id->expert_idx, -1);
}

TEST_F(TensorIdTest, ParseQANorm) {
    auto id = parse_hf_name("model.layers.0.self_attn.q_a_layernorm.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::q_a_norm);
    EXPECT_EQ(id->owner, TensorOwner::attention);
}

TEST_F(TensorIdTest, ParseQBProj) {
    auto id = parse_hf_name("model.layers.10.self_attn.q_b_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::q_b_proj);
    EXPECT_EQ(id->layer_idx, 10);
}

TEST_F(TensorIdTest, ParseKvAProjWithMqa) {
    auto id = parse_hf_name("model.layers.3.self_attn.kv_a_proj_with_mqa.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::kv_a_proj_with_mqa);
}

TEST_F(TensorIdTest, ParseKvANorm) {
    auto id = parse_hf_name("model.layers.0.self_attn.kv_a_layernorm.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::kv_a_norm);
}

TEST_F(TensorIdTest, ParseKvBProj) {
    auto id = parse_hf_name("model.layers.0.self_attn.kv_b_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::kv_b_proj);
}

TEST_F(TensorIdTest, ParseOProj) {
    auto id = parse_hf_name("model.layers.60.self_attn.o_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::o_proj);
    EXPECT_EQ(id->layer_idx, 60);
}

// ── NVFP4 scale tensors ──

TEST_F(TensorIdTest, ParseOProjWeightScale) {
    auto id = parse_hf_name("model.layers.0.self_attn.o_proj.weight_scale");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::o_proj);
    EXPECT_EQ(id->role, TensorRole::weight_scale);
    EXPECT_EQ(id->owner, TensorOwner::attention);
}

TEST_F(TensorIdTest, ParseOProjWeightScale2) {
    auto id = parse_hf_name("model.layers.0.self_attn.o_proj.weight_scale_2");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::o_proj);
    EXPECT_EQ(id->role, TensorRole::weight_scale_2);
}

TEST_F(TensorIdTest, ParseOProjInputScale) {
    auto id = parse_hf_name("model.layers.0.self_attn.o_proj.input_scale");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::o_proj);
    EXPECT_EQ(id->role, TensorRole::input_scale);
}

// ── DSA indexer ──

TEST_F(TensorIdTest, ParseIndexerWqB) {
    auto id = parse_hf_name("model.layers.0.self_attn.indexer.wq_b.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::indexer_wq_b);
    EXPECT_EQ(id->owner, TensorOwner::attention);
}

TEST_F(TensorIdTest, ParseIndexerWk) {
    auto id = parse_hf_name("model.layers.0.self_attn.indexer.wk.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::indexer_wk);
}

TEST_F(TensorIdTest, ParseIndexerKNormWeight) {
    auto id = parse_hf_name("model.layers.0.self_attn.indexer.k_norm.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::indexer_k_norm_weight);
    EXPECT_EQ(id->role, TensorRole::weight);
}

TEST_F(TensorIdTest, ParseIndexerKNormBias) {
    auto id = parse_hf_name("model.layers.0.self_attn.indexer.k_norm.bias");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::indexer_k_norm_bias);
    EXPECT_EQ(id->role, TensorRole::bias);
}

TEST_F(TensorIdTest, ParseIndexerWeightsProj) {
    auto id = parse_hf_name("model.layers.0.self_attn.indexer.weights_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::indexer_weights_proj);
}

// ── Layer norms ──

TEST_F(TensorIdTest, ParseInputLayernorm) {
    auto id = parse_hf_name("model.layers.0.input_layernorm.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::input_layernorm);
}

TEST_F(TensorIdTest, ParsePostAttentionLayernorm) {
    auto id = parse_hf_name("model.layers.30.post_attention_layernorm.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::post_attention_layernorm);
    EXPECT_EQ(id->layer_idx, 30);
}

// ── Gating ──

TEST_F(TensorIdTest, ParseGateWeight) {
    auto id = parse_hf_name("model.layers.3.mlp.gate.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::gate_weight);
    EXPECT_EQ(id->role, TensorRole::weight);
    EXPECT_EQ(id->owner, TensorOwner::gating);
}

TEST_F(TensorIdTest, ParseGateScoreCorrectionBias) {
    auto id = parse_hf_name("model.layers.3.mlp.gate.e_score_correction_bias");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::gate_e_score_correction_bias);
    EXPECT_EQ(id->role, TensorRole::bias);
    EXPECT_EQ(id->owner, TensorOwner::gating);
}

// ── Routed experts ──

TEST_F(TensorIdTest, ParseRoutedExpertGateProj) {
    auto id = parse_hf_name("model.layers.5.mlp.experts.42.gate_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::gate_proj);
    EXPECT_EQ(id->role, TensorRole::weight);
    EXPECT_EQ(id->owner, TensorOwner::routed_expert);
    EXPECT_EQ(id->layer_idx, 5);
    EXPECT_EQ(id->expert_idx, 42);
}

TEST_F(TensorIdTest, ParseRoutedExpertUpProj) {
    auto id = parse_hf_name("model.layers.60.mlp.experts.255.up_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::up_proj);
    EXPECT_EQ(id->owner, TensorOwner::routed_expert);
    EXPECT_EQ(id->expert_idx, 255);
}

TEST_F(TensorIdTest, ParseRoutedExpertDownProj) {
    auto id = parse_hf_name("model.layers.10.mlp.experts.0.down_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::down_proj);
    EXPECT_EQ(id->owner, TensorOwner::routed_expert);
    EXPECT_EQ(id->expert_idx, 0);
}

// ── Routed expert NVFP4 scale tensors ──

TEST_F(TensorIdTest, ParseRoutedExpertWeightScale) {
    auto id = parse_hf_name("model.layers.5.mlp.experts.42.gate_proj.weight_scale");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::gate_proj);
    EXPECT_EQ(id->role, TensorRole::weight_scale);
    EXPECT_EQ(id->owner, TensorOwner::routed_expert);
    EXPECT_EQ(id->expert_idx, 42);
}

TEST_F(TensorIdTest, ParseRoutedExpertWeightScale2) {
    auto id = parse_hf_name("model.layers.5.mlp.experts.42.gate_proj.weight_scale_2");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->role, TensorRole::weight_scale_2);
}

TEST_F(TensorIdTest, ParseRoutedExpertInputScale) {
    auto id = parse_hf_name("model.layers.5.mlp.experts.42.gate_proj.input_scale");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->role, TensorRole::input_scale);
}

// ── Shared expert ──

TEST_F(TensorIdTest, ParseSharedExpertGateProj) {
    auto id = parse_hf_name("model.layers.3.mlp.shared_experts.gate_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::gate_proj);
    EXPECT_EQ(id->owner, TensorOwner::shared_expert);
    EXPECT_EQ(id->expert_idx, -1);
}

TEST_F(TensorIdTest, ParseSharedExpertUpProj) {
    auto id = parse_hf_name("model.layers.3.mlp.shared_experts.up_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::up_proj);
    EXPECT_EQ(id->owner, TensorOwner::shared_expert);
}

TEST_F(TensorIdTest, ParseSharedExpertDownProj) {
    auto id = parse_hf_name("model.layers.3.mlp.shared_experts.down_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::down_proj);
    EXPECT_EQ(id->owner, TensorOwner::shared_expert);
}

// Shared expert with index (some models)
TEST_F(TensorIdTest, ParseSharedExpertWithIndex) {
    auto id = parse_hf_name("model.layers.3.mlp.shared_experts.0.gate_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::gate_proj);
    EXPECT_EQ(id->owner, TensorOwner::shared_expert);
}

// ── Dense FFN ──

TEST_F(TensorIdTest, ParseDenseGateProj) {
    auto id = parse_hf_name("model.layers.0.mlp.gate_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::gate_proj);
    EXPECT_EQ(id->owner, TensorOwner::dense_ffn);
    EXPECT_EQ(id->layer_idx, 0);
}

TEST_F(TensorIdTest, ParseDenseUpProj) {
    auto id = parse_hf_name("model.layers.1.mlp.up_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::up_proj);
    EXPECT_EQ(id->owner, TensorOwner::dense_ffn);
}

TEST_F(TensorIdTest, ParseDenseDownProj) {
    auto id = parse_hf_name("model.layers.2.mlp.down_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::down_proj);
    EXPECT_EQ(id->owner, TensorOwner::dense_ffn);
}

// ── Dense FFN NVFP4 scale tensors ──

TEST_F(TensorIdTest, ParseDenseDownProjWeightScale) {
    auto id = parse_hf_name("model.layers.0.mlp.down_proj.weight_scale");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::down_proj);
    EXPECT_EQ(id->role, TensorRole::weight_scale);
    EXPECT_EQ(id->owner, TensorOwner::dense_ffn);
}

TEST_F(TensorIdTest, ParseDenseGateProjInputScale) {
    auto id = parse_hf_name("model.layers.0.mlp.gate_proj.input_scale");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::gate_proj);
    EXPECT_EQ(id->role, TensorRole::input_scale);
    EXPECT_EQ(id->owner, TensorOwner::dense_ffn);
}

// ── Shared expert NVFP4 scale tensors ──

TEST_F(TensorIdTest, ParseSharedExpertWeightScale) {
    auto id = parse_hf_name("model.layers.3.mlp.shared_experts.gate_proj.weight_scale");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::gate_proj);
    EXPECT_EQ(id->role, TensorRole::weight_scale);
    EXPECT_EQ(id->owner, TensorOwner::shared_expert);
}

// ── Model-level tensors ──

TEST_F(TensorIdTest, ParseEmbedding) {
    auto id = parse_hf_name("model.embed_tokens.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::embedding);
    EXPECT_EQ(id->role, TensorRole::weight);
    EXPECT_EQ(id->owner, TensorOwner::model_level);
    EXPECT_EQ(id->layer_idx, -1);
}

TEST_F(TensorIdTest, ParseFinalNorm) {
    auto id = parse_hf_name("model.norm.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::final_norm);
    EXPECT_EQ(id->owner, TensorOwner::model_level);
}

TEST_F(TensorIdTest, ParseLmHead) {
    auto id = parse_hf_name("lm_head.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::output_head);
    EXPECT_EQ(id->owner, TensorOwner::model_level);
}

// ── MTP tensors ──

TEST_F(TensorIdTest, ParseMtpEnorm) {
    auto id = parse_hf_name("model.layers.61.enorm.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::mtp_enorm);
    EXPECT_EQ(id->owner, TensorOwner::mtp);
    EXPECT_EQ(id->layer_idx, 61);
}

TEST_F(TensorIdTest, ParseMtpHnorm) {
    auto id = parse_hf_name("model.layers.61.hnorm.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::mtp_hnorm);
    EXPECT_EQ(id->owner, TensorOwner::mtp);
}

TEST_F(TensorIdTest, ParseMtpEhProj) {
    auto id = parse_hf_name("model.layers.61.eh_proj.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::mtp_eh_proj);
    EXPECT_EQ(id->owner, TensorOwner::mtp);
}

TEST_F(TensorIdTest, ParseMtpSharedHeadWeight) {
    auto id = parse_hf_name("model.layers.61.shared_head.head.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::mtp_shared_head_weight);
    EXPECT_EQ(id->owner, TensorOwner::mtp);
}

TEST_F(TensorIdTest, ParseMtpSharedHeadNorm) {
    auto id = parse_hf_name("model.layers.61.shared_head.norm.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::mtp_shared_head_norm);
}

TEST_F(TensorIdTest, ParseMtpEmbedTokens) {
    auto id = parse_hf_name("model.layers.61.embed_tokens.weight");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(id->component, TensorComponent::mtp_embed_tokens);
    EXPECT_EQ(id->owner, TensorOwner::mtp);
}

// ── Edge cases ──

TEST_F(TensorIdTest, EmptyName) {
    auto id = parse_hf_name("");
    EXPECT_FALSE(id.has_value());
}

TEST_F(TensorIdTest, UnrecognizedName) {
    auto id = parse_hf_name("some.random.tensor.name");
    EXPECT_FALSE(id.has_value());
}

TEST_F(TensorIdTest, PartialName) {
    auto id = parse_hf_name("model.layers.0");
    EXPECT_FALSE(id.has_value());
}

TEST_F(TensorIdTest, InvalidLayerIndex) {
    auto id = parse_hf_name("model.layers.abc.self_attn.q_a_proj.weight");
    EXPECT_FALSE(id.has_value());
}

// ── same_logical_weight ──

TEST_F(TensorIdTest, SameLogicalWeight) {
    auto w = parse_hf_name("model.layers.5.mlp.experts.42.gate_proj.weight");
    auto s = parse_hf_name("model.layers.5.mlp.experts.42.gate_proj.weight_scale");
    ASSERT_TRUE(w.has_value());
    ASSERT_TRUE(s.has_value());
    EXPECT_TRUE(w->same_logical_weight(*s));
}

TEST_F(TensorIdTest, DifferentLogicalWeight) {
    auto a = parse_hf_name("model.layers.5.mlp.experts.42.gate_proj.weight");
    auto b = parse_hf_name("model.layers.5.mlp.experts.43.gate_proj.weight");
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_FALSE(a->same_logical_weight(*b));
}

// ═══════════════════════════════════════════════════════════════════════════════
// SafetensorsReader tests
// ═══════════════════════════════════════════════════════════════════════════════

// Helper: write a synthetic safetensors file
namespace {

void write_safetensors(const fs::path& path,
                       const std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>>& tensors) {
    // tensors: [(name, dtype_str, shape), ...]
    // Compute data layout
    nlohmann::json header;
    size_t offset = 0;

    struct TensorData {
        std::string name;
        size_t size;
    };
    std::vector<TensorData> data_entries;

    for (auto& [name, dtype_str, shape] : tensors) {
        size_t elem_size = 1;
        if (dtype_str == "F32") elem_size = 4;
        else if (dtype_str == "F16" || dtype_str == "BF16") elem_size = 2;
        else if (dtype_str == "F8_E4M3" || dtype_str == "F8_E5M2" || dtype_str == "U8") elem_size = 1;
        else if (dtype_str == "I64") elem_size = 8;

        size_t numel = 1;
        for (auto d : shape) numel *= static_cast<size_t>(d);
        // Scalars (empty shape) have numel=1
        size_t data_size = numel * elem_size;

        header[name] = {
            {"dtype", dtype_str},
            {"shape", shape},
            {"data_offsets", {offset, offset + data_size}},
        };

        data_entries.push_back({name, data_size});
        offset += data_size;
    }

    std::string header_json = header.dump();
    uint64_t header_size = header_json.size();

    std::ofstream ofs(path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(&header_size), 8);
    ofs.write(header_json.data(), header_json.size());

    // Write dummy data (zeros)
    std::vector<char> zeros(offset, 0);
    // Fill with a pattern so we can verify reads
    for (size_t i = 0; i < zeros.size(); ++i) {
        zeros[i] = static_cast<char>(i & 0xFF);
    }
    ofs.write(zeros.data(), zeros.size());
}

class TempDir {
public:
    TempDir() {
        path_ = fs::temp_directory_path() / ("layerstorm_test_" + std::to_string(getpid()));
        fs::create_directories(path_);
    }
    ~TempDir() { fs::remove_all(path_); }
    const fs::path& path() const { return path_; }
private:
    fs::path path_;
};

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// TD-VOCAB-AUTODETECT: resolve_vocab_size / detect_weights_vocab_rows
// (CPU seam: synthetic safetensors headers, no engine, no GPU)
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

Config vocab_test_config(const fs::path& model_dir) {
    Config cfg;
    cfg.model.weights_path = model_dir.string();
    cfg.model.weights_format = WeightsFormat::safetensors;
    cfg.model.vocab_size = 0;
    return cfg;
}

}  // namespace

TEST(VocabAutodetect, AdoptsEmbeddingRowsWhenConfigAbsent) {
    TempDir tmp;
    write_safetensors(tmp.path() / "model.safetensors", {
        {"model.embed_tokens.weight", "BF16", {1024, 8}},
        {"lm_head.weight", "BF16", {1024, 8}},
    });
    auto cfg = vocab_test_config(tmp.path());

    EXPECT_EQ(layerstorm::model::detect_weights_vocab_rows(cfg), 1024);
    layerstorm::model::resolve_vocab_size(cfg);
    EXPECT_EQ(cfg.model.vocab_size, 1024);  // adopted (0 = autodetect)
}

TEST(VocabAutodetect, ExplicitMatchPassesExplicitMismatchFailsLoud) {
    TempDir tmp;
    write_safetensors(tmp.path() / "model.safetensors", {
        {"model.embed_tokens.weight", "BF16", {1024, 8}},
    });
    auto cfg = vocab_test_config(tmp.path());

    cfg.model.vocab_size = 1024;              // explicit, matches
    EXPECT_NO_THROW(layerstorm::model::resolve_vocab_size(cfg));
    EXPECT_EQ(cfg.model.vocab_size, 1024);

    cfg.model.vocab_size = 129280;            // a DeepSeek vocab pasted
    try {                                     // into a mismatched model
        layerstorm::model::resolve_vocab_size(cfg);
        FAIL() << "mismatch must fail loud";
    } catch (const std::runtime_error& e) {
        // Both numbers must appear in the message (weights = ground truth).
        EXPECT_NE(std::string(e.what()).find("129280"), std::string::npos);
        EXPECT_NE(std::string(e.what()).find("1024"), std::string::npos);
    }
}

TEST(VocabAutodetect, OutputHeadFallbackWhenNoEmbedding) {
    TempDir tmp;
    write_safetensors(tmp.path() / "model.safetensors", {
        {"lm_head.weight", "BF16", {2048, 8}},
        {"model.norm.weight", "BF16", {8}},
    });
    auto cfg = vocab_test_config(tmp.path());
    EXPECT_EQ(layerstorm::model::detect_weights_vocab_rows(cfg), 2048);
}

TEST(VocabAutodetect, ShardedIndexResolvesOwningShard) {
    TempDir tmp;
    // Embedding lives in shard 2 of 2; the index names the owner.
    write_safetensors(tmp.path() / "model-00001-of-00002.safetensors", {
        {"model.layers.0.input_layernorm.weight", "BF16", {8}},
    });
    write_safetensors(tmp.path() / "model-00002-of-00002.safetensors", {
        {"model.embed_tokens.weight", "BF16", {4096, 8}},
    });
    nlohmann::json idx = {
        {"weight_map", {
            {"model.layers.0.input_layernorm.weight",
             "model-00001-of-00002.safetensors"},
            {"model.embed_tokens.weight",
             "model-00002-of-00002.safetensors"},
        }},
    };
    std::ofstream(tmp.path() / "model.safetensors.index.json") << idx.dump();

    auto cfg = vocab_test_config(tmp.path());
    EXPECT_EQ(layerstorm::model::detect_weights_vocab_rows(cfg), 4096);
}

TEST(VocabAutodetect, MissingTensorsFailLoud) {
    TempDir tmp;
    write_safetensors(tmp.path() / "model.safetensors", {
        {"model.norm.weight", "BF16", {8}},
    });
    auto cfg = vocab_test_config(tmp.path());
    EXPECT_THROW(layerstorm::model::resolve_vocab_size(cfg),
                 std::runtime_error);
}

class SafetensorsReaderTest : public ::testing::Test {};

TEST_F(SafetensorsReaderTest, ReadSingleFile) {
    TempDir tmp;
    auto fpath = tmp.path() / "test.safetensors";

    write_safetensors(fpath, {
        {"tensor_a", "F32", {4, 8}},
        {"tensor_b", "BF16", {2, 3}},
        {"tensor_c", "F32", {}},  // scalar
    });

    auto reader = SafetensorsReader::open(fpath);
    ASSERT_TRUE(reader.is_open());
    ASSERT_EQ(reader.entries().size(), 3u);

    // Find tensor_a
    const TensorEntry* a = nullptr;
    for (auto& e : reader.entries()) {
        if (e.name == "tensor_a") a = &e;
    }
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a->dtype, SafetensorsDtype::F32);
    EXPECT_EQ(a->shape, (std::vector<int64_t>{4, 8}));
    EXPECT_EQ(a->data_size_bytes, 4u * 8u * 4u);  // F32 = 4 bytes

    // Read tensor data via mmap
    auto data = reader.tensor_data(*a);
    EXPECT_EQ(data.size(), a->data_size_bytes);

    // Verify the pattern we wrote
    EXPECT_EQ(static_cast<uint8_t>(data[0]), 0);
    EXPECT_EQ(static_cast<uint8_t>(data[1]), 1);

    // scalar tensor
    const TensorEntry* c = nullptr;
    for (auto& e : reader.entries()) {
        if (e.name == "tensor_c") c = &e;
    }
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->shape, std::vector<int64_t>{});
    EXPECT_EQ(c->data_size_bytes, 4u);  // F32 scalar
}

TEST_F(SafetensorsReaderTest, ReadHeaderOnly) {
    TempDir tmp;
    auto fpath = tmp.path() / "test.safetensors";

    write_safetensors(fpath, {
        {"x", "U8", {16, 32}},
        {"y", "F8_E4M3", {16, 4}},
    });

    auto entries = SafetensorsReader::read_header(fpath);
    EXPECT_EQ(entries.size(), 2u);

    bool found_x = false, found_y = false;
    for (auto& e : entries) {
        if (e.name == "x") {
            found_x = true;
            EXPECT_EQ(e.dtype, SafetensorsDtype::U8);
            EXPECT_EQ(e.data_size_bytes, 16u * 32u);
        }
        if (e.name == "y") {
            found_y = true;
            EXPECT_EQ(e.dtype, SafetensorsDtype::F8_E4M3);
        }
    }
    EXPECT_TRUE(found_x);
    EXPECT_TRUE(found_y);
}

TEST_F(SafetensorsReaderTest, NonExistentFile) {
    EXPECT_THROW(SafetensorsReader::open("/nonexistent/path.safetensors"),
                 std::runtime_error);
}

TEST_F(SafetensorsReaderTest, MoveSemantics) {
    TempDir tmp;
    auto fpath = tmp.path() / "test.safetensors";
    write_safetensors(fpath, {{"t", "F32", {2, 2}}});

    auto r1 = SafetensorsReader::open(fpath);
    ASSERT_TRUE(r1.is_open());

    auto r2 = std::move(r1);
    EXPECT_FALSE(r1.is_open());
    EXPECT_TRUE(r2.is_open());
    EXPECT_EQ(r2.entries().size(), 1u);
}

// ── No-mmap (pread) path ──

TEST_F(SafetensorsReaderTest, ReadSingleFileNoMmap) {
    TempDir tmp;
    auto fpath = tmp.path() / "test.safetensors";

    write_safetensors(fpath, {
        {"tensor_a", "F32", {4, 8}},
        {"tensor_b", "BF16", {2, 3}},
        {"tensor_c", "F32", {}},  // scalar
    });

    auto reader = SafetensorsReader::open(fpath, /*use_mmap=*/false);
    ASSERT_TRUE(reader.is_open());
    EXPECT_FALSE(reader.is_mmap());
    ASSERT_EQ(reader.entries().size(), 3u);

    // Find tensor_a
    const TensorEntry* a = nullptr;
    for (auto& e : reader.entries()) {
        if (e.name == "tensor_a") a = &e;
    }
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a->dtype, SafetensorsDtype::F32);
    EXPECT_EQ(a->shape, (std::vector<int64_t>{4, 8}));
    EXPECT_EQ(a->data_size_bytes, 4u * 8u * 4u);

    // Read tensor data from heap buffer
    auto data = reader.tensor_data(*a);
    EXPECT_EQ(data.size(), a->data_size_bytes);

    // Verify the same pattern as mmap
    EXPECT_EQ(static_cast<uint8_t>(data[0]), 0);
    EXPECT_EQ(static_cast<uint8_t>(data[1]), 1);

    // scalar tensor
    const TensorEntry* c = nullptr;
    for (auto& e : reader.entries()) {
        if (e.name == "tensor_c") c = &e;
    }
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->data_size_bytes, 4u);
}

TEST_F(SafetensorsReaderTest, NoMmapMoveSemantics) {
    TempDir tmp;
    auto fpath = tmp.path() / "test.safetensors";
    write_safetensors(fpath, {{"t", "F32", {2, 2}}});

    auto r1 = SafetensorsReader::open(fpath, /*use_mmap=*/false);
    ASSERT_TRUE(r1.is_open());
    EXPECT_FALSE(r1.is_mmap());

    auto r2 = std::move(r1);
    EXPECT_FALSE(r1.is_open());
    EXPECT_TRUE(r2.is_open());
    EXPECT_FALSE(r2.is_mmap());
    EXPECT_EQ(r2.entries().size(), 1u);

    // Verify data is still readable after move
    auto data = r2.tensor_data(r2.entries()[0]);
    EXPECT_EQ(data.size(), 4u * 4u);  // 2*2*F32
}

TEST_F(SafetensorsReaderTest, MmapFlagCorrect) {
    TempDir tmp;
    auto fpath = tmp.path() / "test.safetensors";
    write_safetensors(fpath, {{"t", "F32", {2}}});

    auto mmap_reader = SafetensorsReader::open(fpath, true);
    EXPECT_TRUE(mmap_reader.is_mmap());

    auto pread_reader = SafetensorsReader::open(fpath, false);
    EXPECT_FALSE(pread_reader.is_mmap());
}

TEST_F(SafetensorsReaderTest, MmapAndPreadDataMatch) {
    TempDir tmp;
    auto fpath = tmp.path() / "test.safetensors";
    write_safetensors(fpath, {
        {"w", "F32", {8, 16}},
        {"s", "U8", {32}},
    });

    auto mmap_reader = SafetensorsReader::open(fpath, true);
    auto pread_reader = SafetensorsReader::open(fpath, false);

    ASSERT_EQ(mmap_reader.entries().size(), pread_reader.entries().size());

    for (size_t i = 0; i < mmap_reader.entries().size(); ++i) {
        auto mmap_data = mmap_reader.tensor_data(mmap_reader.entries()[i]);
        auto pread_data = pread_reader.tensor_data(pread_reader.entries()[i]);

        ASSERT_EQ(mmap_data.size(), pread_data.size());
        EXPECT_EQ(std::memcmp(mmap_data.data(), pread_data.data(), mmap_data.size()), 0)
            << "Data mismatch for tensor " << mmap_reader.entries()[i].name;
    }
}

// ── Shard index ──

TEST_F(SafetensorsReaderTest, ShardIndex) {
    TempDir tmp;

    // Write index.json
    nlohmann::json idx;
    idx["metadata"] = {};
    idx["weight_map"] = {
        {"tensor_a", "model-00001-of-00002.safetensors"},
        {"tensor_b", "model-00001-of-00002.safetensors"},
        {"tensor_c", "model-00002-of-00002.safetensors"},
    };

    std::ofstream(tmp.path() / "model.safetensors.index.json") << idx.dump();

    // Write shard files
    write_safetensors(tmp.path() / "model-00001-of-00002.safetensors", {
        {"tensor_a", "F32", {2, 2}},
        {"tensor_b", "BF16", {4}},
    });
    write_safetensors(tmp.path() / "model-00002-of-00002.safetensors", {
        {"tensor_c", "F32", {3}},
    });

    auto shard_idx = read_shard_index(tmp.path());
    EXPECT_EQ(shard_idx.shard_files.size(), 2u);
    EXPECT_EQ(shard_idx.tensor_to_shard.size(), 3u);
}

TEST_F(SafetensorsReaderTest, SingleFileIndex) {
    TempDir tmp;
    write_safetensors(tmp.path() / "model.safetensors", {
        {"tensor_a", "F32", {2}},
    });

    auto shard_idx = read_shard_index(tmp.path());
    EXPECT_EQ(shard_idx.shard_files.size(), 1u);
    EXPECT_EQ(shard_idx.shard_files[0], "model.safetensors");
}

TEST_F(SafetensorsReaderTest, NoSafetensorsFiles) {
    TempDir tmp;
    EXPECT_THROW(read_shard_index(tmp.path()), std::runtime_error);
}

// ═══════════════════════════════════════════════════════════════════════════════
// WeightHandler tests
// ═══════════════════════════════════════════════════════════════════════════════

class WeightHandlerTest : public ::testing::Test {};

TEST_F(WeightHandlerTest, HandlerForDtypeU8IsNvfp4) {
    auto& h = handler_for_dtype(SafetensorsDtype::U8);
    EXPECT_EQ(h.name(), "nvfp4");
}

TEST_F(WeightHandlerTest, HandlerForDtypeF8E4M3IsFp8) {
    auto& h = handler_for_dtype(SafetensorsDtype::F8_E4M3);
    EXPECT_EQ(h.name(), "fp8");
}

TEST_F(WeightHandlerTest, HandlerForDtypeF8E5M2IsFp8) {
    auto& h = handler_for_dtype(SafetensorsDtype::F8_E5M2);
    EXPECT_EQ(h.name(), "fp8");
}

TEST_F(WeightHandlerTest, HandlerForDtypeBF16IsNative) {
    auto& h = handler_for_dtype(SafetensorsDtype::BF16);
    EXPECT_EQ(h.name(), "native");
}

TEST_F(WeightHandlerTest, HandlerForDtypeF32IsNative) {
    auto& h = handler_for_dtype(SafetensorsDtype::F32);
    EXPECT_EQ(h.name(), "native");
}

// ── NVFP4 handler ──

TEST_F(WeightHandlerTest, NvFp4ExpectedAuxRoles) {
    NvFp4WeightHandler handler;
    auto roles = handler.expected_aux_roles();
    EXPECT_EQ(roles.size(), 3u);
    EXPECT_EQ(roles[0], TensorRole::weight_scale);
    EXPECT_EQ(roles[1], TensorRole::weight_scale_2);
    EXPECT_EQ(roles[2], TensorRole::input_scale);
}

TEST_F(WeightHandlerTest, NvFp4ValidBundle) {
    NvFp4WeightHandler handler;

    // Simulate a gate_proj [2048, 3584] (packed: 7168/2 = 3584)
    std::vector<std::byte> weight_data(2048 * 3584);
    std::vector<std::byte> ws_data(2048 * 448);  // 3584/8 = 448
    std::vector<std::byte> ws2_data(4);           // F32 scalar
    std::vector<std::byte> is_data(4);            // F32 scalar

    Config cfg;
    cfg.model.num_hidden_layers = 4;
    cfg.model.hidden_size = 7168;
    MConfig mcfg(cfg);

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::gate_proj, TensorRole::weight,
                         TensorOwner::routed_expert, 3, 0};
    bundle.weight = RawTensor{weight_data, SafetensorsDtype::U8, {2048, 3584}};
    bundle.aux = {
        {TensorRole::weight_scale, RawTensor{ws_data, SafetensorsDtype::F8_E4M3, {2048, 448}}},
        {TensorRole::weight_scale_2, RawTensor{ws2_data, SafetensorsDtype::F32, {}}},
        {TensorRole::input_scale, RawTensor{is_data, SafetensorsDtype::F32, {}}},
    };

    auto err = handler.validate(bundle, mcfg);
    EXPECT_TRUE(err.empty()) << err;
}

TEST_F(WeightHandlerTest, NvFp4MissingScale) {
    NvFp4WeightHandler handler;

    std::vector<std::byte> weight_data(100);

    Config cfg;
    cfg.model.num_hidden_layers = 4;
    MConfig mcfg(cfg);

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::gate_proj, TensorRole::weight,
                         TensorOwner::routed_expert, 3, 0};
    bundle.weight = RawTensor{weight_data, SafetensorsDtype::U8, {10, 10}};
    // No aux tensors

    auto err = handler.validate(bundle, mcfg);
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("weight_scale"), std::string::npos);
}

TEST_F(WeightHandlerTest, NvFp4WrongWeightDtype) {
    NvFp4WeightHandler handler;

    std::vector<std::byte> weight_data(100);

    Config cfg;
    cfg.model.num_hidden_layers = 4;
    MConfig mcfg(cfg);

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::gate_proj, TensorRole::weight,
                         TensorOwner::routed_expert, 3, 0};
    bundle.weight = RawTensor{weight_data, SafetensorsDtype::F32, {10, 10}};

    auto err = handler.validate(bundle, mcfg);
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("U8"), std::string::npos);
}

// ── FP8 handler ──

TEST_F(WeightHandlerTest, Fp8ValidBundle) {
    Fp8WeightHandler handler;

    std::vector<std::byte> weight_data(100);

    Config cfg;
    cfg.model.num_hidden_layers = 4;
    MConfig mcfg(cfg);

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::gate_proj, TensorRole::weight,
                         TensorOwner::routed_expert, 3, 0};
    bundle.weight = RawTensor{weight_data, SafetensorsDtype::F8_E4M3, {10, 10}};

    auto err = handler.validate(bundle, mcfg);
    EXPECT_TRUE(err.empty()) << err;
}

TEST_F(WeightHandlerTest, Fp8WrongDtype) {
    Fp8WeightHandler handler;

    std::vector<std::byte> weight_data(100);

    Config cfg;
    cfg.model.num_hidden_layers = 4;
    MConfig mcfg(cfg);

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::gate_proj, TensorRole::weight,
                         TensorOwner::routed_expert, 3, 0};
    bundle.weight = RawTensor{weight_data, SafetensorsDtype::F32, {10, 10}};

    auto err = handler.validate(bundle, mcfg);
    EXPECT_FALSE(err.empty());
}

TEST_F(WeightHandlerTest, Fp8AcceptsWeightScale) {
    Fp8WeightHandler handler;

    // Weight [10, 128]: 1280 bytes FP8.
    // Scale shape: [ceil(10/128), ceil(128/128)] = [1, 1] → 4 bytes F32.
    std::vector<std::byte> weight_data(1280);
    std::vector<std::byte> scale_data(4);

    Config cfg;
    cfg.model.num_hidden_layers = 4;
    MConfig mcfg(cfg);

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::gate_proj, TensorRole::weight,
                         TensorOwner::routed_expert, 3, 0};
    bundle.weight = RawTensor{weight_data, SafetensorsDtype::F8_E4M3, {10, 128}};
    bundle.aux = {{TensorRole::weight_scale, RawTensor{scale_data, SafetensorsDtype::F32, {1, 1}}}};

    auto err = handler.validate(bundle, mcfg);
    EXPECT_TRUE(err.empty()) << err;
}

TEST_F(WeightHandlerTest, Fp8AcceptsNoAux) {
    Fp8WeightHandler handler;

    std::vector<std::byte> weight_data(100);

    Config cfg;
    cfg.model.num_hidden_layers = 4;
    MConfig mcfg(cfg);

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::gate_proj, TensorRole::weight,
                         TensorOwner::routed_expert, 3, 0};
    bundle.weight = RawTensor{weight_data, SafetensorsDtype::F8_E4M3, {10, 10}};

    auto err = handler.validate(bundle, mcfg);
    EXPECT_TRUE(err.empty()) << err;
}

TEST_F(WeightHandlerTest, Fp8RejectsUnexpectedAux) {
    Fp8WeightHandler handler;

    std::vector<std::byte> weight_data(100);
    std::vector<std::byte> aux_data(4);

    Config cfg;
    cfg.model.num_hidden_layers = 4;
    MConfig mcfg(cfg);

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::gate_proj, TensorRole::weight,
                         TensorOwner::routed_expert, 3, 0};
    bundle.weight = RawTensor{weight_data, SafetensorsDtype::F8_E4M3, {10, 10}};
    bundle.aux = {{TensorRole::weight_scale_2, RawTensor{aux_data, SafetensorsDtype::F32, {}}}};

    auto err = handler.validate(bundle, mcfg);
    EXPECT_FALSE(err.empty());
}

TEST_F(WeightHandlerTest, Fp8ValidatesScaleShape) {
    Fp8WeightHandler handler;

    // Weight [10, 128]. Expected scale shape: [1, 1]. Give wrong shape [2, 2].
    std::vector<std::byte> weight_data(1280);
    std::vector<std::byte> scale_data(16);

    Config cfg;
    cfg.model.num_hidden_layers = 4;
    MConfig mcfg(cfg);

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::gate_proj, TensorRole::weight,
                         TensorOwner::routed_expert, 3, 0};
    bundle.weight = RawTensor{weight_data, SafetensorsDtype::F8_E4M3, {10, 128}};
    bundle.aux = {{TensorRole::weight_scale, RawTensor{scale_data, SafetensorsDtype::F32, {2, 2}}}};

    auto err = handler.validate(bundle, mcfg);
    EXPECT_FALSE(err.empty());
}

// ── Native handler ──

TEST_F(WeightHandlerTest, NativeValidBF16) {
    NativeWeightHandler handler;

    std::vector<std::byte> weight_data(100);

    Config cfg;
    cfg.model.num_hidden_layers = 4;
    MConfig mcfg(cfg);

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::q_a_proj, TensorRole::weight,
                         TensorOwner::attention, 0, -1};
    bundle.weight = RawTensor{weight_data, SafetensorsDtype::BF16, {50}};

    auto err = handler.validate(bundle, mcfg);
    EXPECT_TRUE(err.empty()) << err;
}

TEST_F(WeightHandlerTest, NativeValidF32) {
    NativeWeightHandler handler;

    std::vector<std::byte> weight_data(100);

    Config cfg;
    cfg.model.num_hidden_layers = 4;
    MConfig mcfg(cfg);

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::input_layernorm, TensorRole::weight,
                         TensorOwner::attention, 0, -1};
    bundle.weight = RawTensor{weight_data, SafetensorsDtype::F32, {25}};

    auto err = handler.validate(bundle, mcfg);
    EXPECT_TRUE(err.empty()) << err;
}

TEST_F(WeightHandlerTest, NativeRejectsU8) {
    NativeWeightHandler handler;

    std::vector<std::byte> weight_data(100);

    Config cfg;
    cfg.model.num_hidden_layers = 4;
    MConfig mcfg(cfg);

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::q_a_proj, TensorRole::weight,
                         TensorOwner::attention, 0, -1};
    bundle.weight = RawTensor{weight_data, SafetensorsDtype::U8, {100}};

    auto err = handler.validate(bundle, mcfg);
    EXPECT_FALSE(err.empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// WeightBundle tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(WeightBundleTest, TotalBytes) {
    std::vector<std::byte> w(1000), s1(100), s2(4);
    WeightBundle bundle;
    bundle.weight = RawTensor{w, SafetensorsDtype::U8, {10, 100}};
    bundle.aux = {
        {TensorRole::weight_scale, RawTensor{s1, SafetensorsDtype::F8_E4M3, {10, 10}}},
        {TensorRole::weight_scale_2, RawTensor{s2, SafetensorsDtype::F32, {}}},
    };
    EXPECT_EQ(bundle.total_bytes(), 1104);
}

TEST(WeightBundleTest, FindAux) {
    std::vector<std::byte> w(10), s(4);
    WeightBundle bundle;
    bundle.weight = RawTensor{w, SafetensorsDtype::U8, {10}};
    bundle.aux = {
        {TensorRole::weight_scale, RawTensor{s, SafetensorsDtype::F8_E4M3, {4}}},
    };

    auto* found = bundle.find_aux(TensorRole::weight_scale);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->dtype, SafetensorsDtype::F8_E4M3);

    auto* not_found = bundle.find_aux(TensorRole::input_scale);
    EXPECT_EQ(not_found, nullptr);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Full weight loading integration test (mini model)
// ═══════════════════════════════════════════════════════════════════════════════

class WeightLoaderIntegrationTest : public ::testing::Test {
protected:
    TempDir tmp;

    // Create a minimal 3-layer model (1 dense + 2 MoE with 2 experts each)
    void create_mini_model() {
        auto dir = tmp.path();

        // Shard 1: embedding, layer 0 (dense), layer 1 norms/attn
        std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> shard1;

        // Model-level
        shard1.push_back({"model.embed_tokens.weight", "BF16", {16, 8}});
        shard1.push_back({"model.norm.weight", "F32", {8}});
        shard1.push_back({"lm_head.weight", "BF16", {16, 8}});

        // Layer 0 (dense)
        shard1.push_back({"model.layers.0.input_layernorm.weight", "F32", {8}});
        shard1.push_back({"model.layers.0.post_attention_layernorm.weight", "F32", {8}});
        shard1.push_back({"model.layers.0.self_attn.q_a_proj.weight", "BF16", {8, 8}});
        shard1.push_back({"model.layers.0.self_attn.q_a_layernorm.weight", "F32", {8}});
        shard1.push_back({"model.layers.0.self_attn.q_b_proj.weight", "BF16", {8, 8}});
        shard1.push_back({"model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "BF16", {8, 8}});
        shard1.push_back({"model.layers.0.self_attn.kv_a_layernorm.weight", "F32", {8}});
        shard1.push_back({"model.layers.0.self_attn.kv_b_proj.weight", "BF16", {8, 8}});
        shard1.push_back({"model.layers.0.self_attn.o_proj.weight", "BF16", {8, 8}});
        shard1.push_back({"model.layers.0.mlp.gate_proj.weight", "BF16", {16, 8}});
        shard1.push_back({"model.layers.0.mlp.up_proj.weight", "BF16", {16, 8}});
        shard1.push_back({"model.layers.0.mlp.down_proj.weight", "BF16", {8, 16}});

        write_safetensors(dir / "model-00001-of-00002.safetensors", shard1);

        // Shard 2: layers 1-2 (MoE)
        std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> shard2;

        for (int l = 1; l <= 2; ++l) {
            auto ls = std::to_string(l);
            shard2.push_back({"model.layers." + ls + ".input_layernorm.weight", "F32", {8}});
            shard2.push_back({"model.layers." + ls + ".post_attention_layernorm.weight", "F32", {8}});
            shard2.push_back({"model.layers." + ls + ".self_attn.q_a_proj.weight", "BF16", {8, 8}});
            shard2.push_back({"model.layers." + ls + ".self_attn.q_a_layernorm.weight", "F32", {8}});
            shard2.push_back({"model.layers." + ls + ".self_attn.q_b_proj.weight", "BF16", {8, 8}});
            shard2.push_back({"model.layers." + ls + ".self_attn.kv_a_proj_with_mqa.weight", "BF16", {8, 8}});
            shard2.push_back({"model.layers." + ls + ".self_attn.kv_a_layernorm.weight", "F32", {8}});
            shard2.push_back({"model.layers." + ls + ".self_attn.kv_b_proj.weight", "BF16", {8, 8}});
            shard2.push_back({"model.layers." + ls + ".self_attn.o_proj.weight", "BF16", {8, 8}});
            shard2.push_back({"model.layers." + ls + ".mlp.gate.weight", "BF16", {8, 2}});
            shard2.push_back({"model.layers." + ls + ".mlp.shared_experts.gate_proj.weight", "BF16", {4, 8}});
            shard2.push_back({"model.layers." + ls + ".mlp.shared_experts.up_proj.weight", "BF16", {4, 8}});
            shard2.push_back({"model.layers." + ls + ".mlp.shared_experts.down_proj.weight", "BF16", {8, 4}});

            for (int e = 0; e < 2; ++e) {
                auto es = std::to_string(e);
                shard2.push_back({"model.layers." + ls + ".mlp.experts." + es + ".gate_proj.weight", "BF16", {4, 8}});
                shard2.push_back({"model.layers." + ls + ".mlp.experts." + es + ".up_proj.weight", "BF16", {4, 8}});
                shard2.push_back({"model.layers." + ls + ".mlp.experts." + es + ".down_proj.weight", "BF16", {8, 4}});
            }
        }

        write_safetensors(dir / "model-00002-of-00002.safetensors", shard2);

        // Write index.json
        nlohmann::json idx;
        idx["metadata"] = {};
        nlohmann::json wm;
        for (auto& [name, dtype, shape] : shard1) {
            wm[name] = "model-00001-of-00002.safetensors";
        }
        for (auto& [name, dtype, shape] : shard2) {
            wm[name] = "model-00002-of-00002.safetensors";
        }
        idx["weight_map"] = wm;
        std::ofstream(dir / "model.safetensors.index.json") << idx.dump();
    }

    Config make_mini_config() {
        Config cfg;
        cfg.model.architecture = Architecture::deepseek_v3;
        cfg.model.weights_path = tmp.path().string();
        cfg.model.weights_format = WeightsFormat::safetensors;
        cfg.model.num_hidden_layers = 3;
        cfg.model.hidden_size = 8;
        cfg.model.num_attention_heads = 1;
        cfg.model.kv_lora_rank = 4;
        cfg.model.qk_rope_head_dim = 4;
        cfg.model.qk_nope_head_dim = 4;
        cfg.model.v_head_dim = 4;
        cfg.model.q_lora_rank = 8;
        cfg.model.intermediate_size = 16;
        cfg.model.moe_intermediate_size = 4;
        cfg.model.n_routed_experts = 2;
        cfg.model.n_shared_experts = 1;
        cfg.model.num_experts_per_tok = 1;
        cfg.model.vocab_size = 16;
        cfg.model.max_position_embeddings = 128;
        cfg.model.first_k_dense_replace = 1;
        cfg.model.moe_layer_freq = 1;
        cfg.model.num_nextn_predict_layers = 0;
        cfg.model.index_topk = 0;  // no DSA

        cfg.quantization.weights = WeightQuant::fp8_e4m3;
        cfg.quantization.attention_compute = AttentionQuant::fp8_e4m3;
        cfg.quantization.gating_compute = GatingQuant::fp32;

        // Hardware (minimal for LayerRegistry)
        cfg.hardware.gpus.push_back(GpuConfig{.id = 0, .type = GpuType::rtx5090, .vram_gb = 32.0});
        cfg.hardware.tp_array = {0};

        return cfg;
    }
};

TEST_F(WeightLoaderIntegrationTest, LoadMiniModel) {
    create_mini_model();
    auto cfg = make_mini_config();
    MConfig mcfg(cfg);

    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    auto model = load_weights(cfg, mcfg, registry);

    // Check model-level tensors
    EXPECT_TRUE(model.embedding.has_value());
    EXPECT_TRUE(model.output_head.has_value());
    EXPECT_TRUE(model.final_norm.has_value());

    // Check layers
    EXPECT_EQ(model.layers.size(), 3u);

    // Layer 0: dense
    EXPECT_FALSE(model.layers[0].attention.empty());
    EXPECT_FALSE(model.layers[0].norms.empty());
    EXPECT_FALSE(model.layers[0].dense_ffn.empty());
    EXPECT_TRUE(model.layers[0].routed_experts.empty() ||
                model.layers[0].routed_experts[0].empty());

    // Layers 1-2: MoE
    for (int l = 1; l <= 2; ++l) {
        EXPECT_FALSE(model.layers[l].attention.empty());
        EXPECT_FALSE(model.layers[l].norms.empty());
        EXPECT_FALSE(model.layers[l].gating.empty());
        EXPECT_FALSE(model.layers[l].shared_expert.empty());
        EXPECT_EQ(model.layers[l].routed_experts.size(), 2u);
        for (int e = 0; e < 2; ++e) {
            EXPECT_EQ(model.layers[l].routed_experts[e].size(), 3u)  // gate, up, down
                << "Layer " << l << " expert " << e;
        }
    }

    // Stats
    EXPECT_GT(model.total_weight_bytes, 0);
    EXPECT_GT(model.total_tensors_loaded, 0);
    EXPECT_EQ(model.shards.size(), 2u);
}

TEST_F(WeightLoaderIntegrationTest, MissingEmbedding) {
    // Create a model missing the embedding tensor
    auto dir = tmp.path();

    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> tensors;
    // Deliberately omit model.embed_tokens.weight
    tensors.push_back({"model.norm.weight", "F32", {8}});
    tensors.push_back({"lm_head.weight", "BF16", {16, 8}});

    // Layer 0 (dense)
    tensors.push_back({"model.layers.0.input_layernorm.weight", "F32", {8}});
    tensors.push_back({"model.layers.0.post_attention_layernorm.weight", "F32", {8}});
    tensors.push_back({"model.layers.0.self_attn.q_a_proj.weight", "BF16", {8, 8}});
    tensors.push_back({"model.layers.0.self_attn.q_a_layernorm.weight", "F32", {8}});
    tensors.push_back({"model.layers.0.self_attn.q_b_proj.weight", "BF16", {8, 8}});
    tensors.push_back({"model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "BF16", {8, 8}});
    tensors.push_back({"model.layers.0.self_attn.kv_a_layernorm.weight", "F32", {8}});
    tensors.push_back({"model.layers.0.self_attn.kv_b_proj.weight", "BF16", {8, 8}});
    tensors.push_back({"model.layers.0.self_attn.o_proj.weight", "BF16", {8, 8}});
    tensors.push_back({"model.layers.0.mlp.gate_proj.weight", "BF16", {16, 8}});
    tensors.push_back({"model.layers.0.mlp.up_proj.weight", "BF16", {16, 8}});
    tensors.push_back({"model.layers.0.mlp.down_proj.weight", "BF16", {8, 16}});

    write_safetensors(dir / "model.safetensors", tensors);

    auto cfg = make_mini_config();
    cfg.model.weights_path = dir.string();
    cfg.model.num_hidden_layers = 1;
    cfg.model.first_k_dense_replace = 1;  // All layers dense
    cfg.model.n_routed_experts = 0;
    MConfig mcfg(cfg);

    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    // Should throw due to missing embedding
    EXPECT_THROW(load_weights(cfg, mcfg, registry), std::runtime_error);
}

TEST_F(WeightLoaderIntegrationTest, LoadMiniModelNoMmap) {
    create_mini_model();
    auto cfg = make_mini_config();
    cfg.model.use_mmap = false;
    MConfig mcfg(cfg);

    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    auto model = load_weights(cfg, mcfg, registry);

    // All shards should use pread
    for (auto& shard : model.shards) {
        EXPECT_FALSE(shard.is_mmap());
    }

    // Same structural checks as mmap test
    EXPECT_TRUE(model.embedding.has_value());
    EXPECT_TRUE(model.output_head.has_value());
    EXPECT_TRUE(model.final_norm.has_value());
    EXPECT_EQ(model.layers.size(), 3u);

    EXPECT_FALSE(model.layers[0].dense_ffn.empty());
    for (int l = 1; l <= 2; ++l) {
        EXPECT_EQ(model.layers[l].routed_experts.size(), 2u);
    }

    EXPECT_GT(model.total_weight_bytes, 0);
    EXPECT_EQ(model.shards.size(), 2u);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dtype helpers
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DtypeTest, Sizes) {
    EXPECT_EQ(dtype_size(SafetensorsDtype::F32), 4u);
    EXPECT_EQ(dtype_size(SafetensorsDtype::F16), 2u);
    EXPECT_EQ(dtype_size(SafetensorsDtype::BF16), 2u);
    EXPECT_EQ(dtype_size(SafetensorsDtype::F8_E4M3), 1u);
    EXPECT_EQ(dtype_size(SafetensorsDtype::U8), 1u);
    EXPECT_EQ(dtype_size(SafetensorsDtype::I64), 8u);
}

TEST(DtypeTest, ParseRoundTrip) {
    auto dtypes = {SafetensorsDtype::F32, SafetensorsDtype::F16, SafetensorsDtype::BF16,
                   SafetensorsDtype::F8_E4M3, SafetensorsDtype::F8_E5M2, SafetensorsDtype::U8};
    for (auto dt : dtypes) {
        auto name = dtype_name(dt);
        auto parsed = parse_dtype(name);
        ASSERT_TRUE(parsed.has_value()) << "Failed for " << name;
        EXPECT_EQ(*parsed, dt);
    }
}

TEST(DtypeTest, ParseUnknown) {
    EXPECT_FALSE(parse_dtype("FLOAT64").has_value());
    EXPECT_FALSE(parse_dtype("").has_value());
}

// ═══════════════════════════════════════════════════════════════════════════════
// String helpers
// ═══════════════════════════════════════════════════════════════════════════════

TEST(StringHelperTest, ComponentNames) {
    EXPECT_EQ(tensor_component_name(TensorComponent::q_a_proj), "q_a_proj");
    EXPECT_EQ(tensor_component_name(TensorComponent::embedding), "embedding");
    EXPECT_EQ(tensor_component_name(TensorComponent::mtp_enorm), "mtp_enorm");
}

TEST(StringHelperTest, RoleNames) {
    EXPECT_EQ(tensor_role_name(TensorRole::weight), "weight");
    EXPECT_EQ(tensor_role_name(TensorRole::weight_scale), "weight_scale");
    EXPECT_EQ(tensor_role_name(TensorRole::bias), "bias");
}

TEST(StringHelperTest, OwnerNames) {
    EXPECT_EQ(tensor_owner_name(TensorOwner::attention), "attention");
    EXPECT_EQ(tensor_owner_name(TensorOwner::routed_expert), "routed_expert");
}

// ═══════════════════════════════════════════════════════════════════════════════
// WP-6: skip_routed_experts tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(WeightLoaderIntegrationTest, SkipRoutedExperts) {
    create_mini_model();
    auto cfg = make_mini_config();
    MConfig mcfg(cfg);

    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    auto model = load_weights(cfg, mcfg, registry, /*skip_routed_experts=*/true);

    // Model-level tensors still loaded
    EXPECT_TRUE(model.embedding.has_value());
    EXPECT_TRUE(model.output_head.has_value());
    EXPECT_TRUE(model.final_norm.has_value());

    // All layers present
    EXPECT_EQ(model.layers.size(), 3u);

    // Layer 0 (dense): pinned weights loaded normally
    EXPECT_FALSE(model.layers[0].attention.empty());
    EXPECT_FALSE(model.layers[0].norms.empty());
    EXPECT_FALSE(model.layers[0].dense_ffn.empty());

    // MoE layers: pinned weights loaded, routed experts empty
    for (int l = 1; l <= 2; ++l) {
        EXPECT_FALSE(model.layers[l].attention.empty())
            << "Layer " << l << " missing attention";
        EXPECT_FALSE(model.layers[l].norms.empty())
            << "Layer " << l << " missing norms";
        EXPECT_FALSE(model.layers[l].gating.empty())
            << "Layer " << l << " missing gating";
        EXPECT_FALSE(model.layers[l].shared_expert.empty())
            << "Layer " << l << " missing shared expert";
        // WP-6: routed_experts must be empty
        EXPECT_TRUE(model.layers[l].routed_experts.empty())
            << "Layer " << l << " should have empty routed_experts";
    }

    // Stats: routed expert bytes NOT counted
    EXPECT_GT(model.total_weight_bytes, 0);
    EXPECT_GT(model.total_tensors_loaded, 0);
}

TEST_F(WeightLoaderIntegrationTest, SkipRoutedExpertsFewerTensors) {
    create_mini_model();
    auto cfg = make_mini_config();
    MConfig mcfg(cfg);

    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    auto full = load_weights(cfg, mcfg, registry, /*skip_routed_experts=*/false);
    auto skip = load_weights(cfg, mcfg, registry, /*skip_routed_experts=*/true);

    // Skipping routed experts should yield fewer tensors and bytes
    EXPECT_LT(skip.total_tensors_loaded, full.total_tensors_loaded);
    EXPECT_LT(skip.total_weight_bytes, full.total_weight_bytes);
}

TEST_F(WeightLoaderIntegrationTest, LegacyModeLoadsRoutedExperts) {
    // When skip_routed_experts=false (legacy), routed experts are populated
    create_mini_model();
    auto cfg = make_mini_config();
    MConfig mcfg(cfg);

    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    auto model = load_weights(cfg, mcfg, registry, /*skip_routed_experts=*/false);

    // MoE layers: routed experts present (legacy behavior)
    for (int l = 1; l <= 2; ++l) {
        EXPECT_EQ(model.layers[l].routed_experts.size(), 2u)
            << "Layer " << l << " should have 2 routed experts";
        for (int e = 0; e < 2; ++e) {
            EXPECT_EQ(model.layers[l].routed_experts[e].size(), 3u)
                << "Layer " << l << " expert " << e;
        }
    }
}
