#include <gtest/gtest.h>

#include "model/model_config.h"

namespace lc = layerstorm::config;
using layerstorm::model::ModelConfig;

// ── Helpers: build model configs for each architecture ───────────────────────

static ModelConfig deepseek_v3_2_config() {
    return ModelConfig{[] {
        auto j = nlohmann::json{
            {"model", {
                {"architecture",           "deepseek_v3"},
                {"weights_path",           "/data/models/deepseek-v3.2/"},
                {"weights_format",         "safetensors"},
                {"num_hidden_layers",      61},
                {"hidden_size",            7168},
                {"num_attention_heads",    128},
                {"num_key_value_heads",    128},
                {"intermediate_size",      18432},
                {"n_routed_experts",       256},
                {"n_shared_experts",       1},
                {"num_experts_per_tok",    8},
                {"n_group",                8},
                {"topk_group",             4},
                {"vocab_size",             129280},
                {"max_position_embeddings", 163840},
                {"kv_lora_rank",           512},
                {"q_lora_rank",            1536},
                {"qk_rope_head_dim",       64},
                {"qk_nope_head_dim",       128},
                {"v_head_dim",             128},
                {"first_k_dense_replace",  3},
                {"moe_layer_freq",         1},
                {"index_topk",             2048},
                {"index_n_heads",          64},
                {"index_head_dim",         128},
                {"num_nextn_predict_layers", 1},
                {"rms_norm_eps",           1e-6},
                {"rope_theta",             10000.0},
                {"routed_scaling_factor",  2.5},
                {"moe_intermediate_size",  2048},
            }},
            {"quantization", {{"weights", "nvfp4"}, {"attention_compute", "fp8_e4m3"},
                              {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
            {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                          {"system_ram_gb", 256}}},
        };
        return lc::parse_config(j);
    }()};
}

static ModelConfig glm5_config() {
    return ModelConfig{[] {
        auto j = nlohmann::json{
            {"model", {
                {"architecture",           "glm_moe_dsa"},
                {"weights_path",           "/data/models/glm-5/"},
                {"weights_format",         "safetensors"},
                {"num_hidden_layers",      78},
                {"hidden_size",            6144},
                {"num_attention_heads",    64},
                {"num_key_value_heads",    64},
                {"intermediate_size",      12288},
                {"n_routed_experts",       256},
                {"n_shared_experts",       1},
                {"num_experts_per_tok",    8},
                {"n_group",                1},
                {"topk_group",             1},
                {"vocab_size",             154880},
                {"max_position_embeddings", 202752},
                {"kv_lora_rank",           512},
                {"q_lora_rank",            2048},
                {"qk_rope_head_dim",       64},
                {"qk_nope_head_dim",       192},
                {"v_head_dim",             256},
                {"first_k_dense_replace",  3},
                {"moe_layer_freq",         1},
                {"index_topk",             2048},
                {"index_n_heads",          32},
                {"index_head_dim",         128},
                {"index_topk_freq",        4},
                {"index_skip_topk_offset", 3},
                {"rope_interleave",        true},
                {"indexer_rope_interleave", true},
                {"num_nextn_predict_layers", 1},
                {"rms_norm_eps",           1e-5},
                {"rope_theta",             1000000.0},
                {"routed_scaling_factor",  2.5},
                {"moe_intermediate_size",  2048},
            }},
            {"quantization", {{"weights", "nvfp4"}, {"attention_compute", "fp8_e4m3"},
                              {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
            {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                          {"system_ram_gb", 256}}},
        };
        return lc::parse_config(j);
    }()};
}

static ModelConfig kimi_k25_config() {
    return ModelConfig{[] {
        auto j = nlohmann::json{
            {"model", {
                {"architecture",           "kimi_k25"},
                {"weights_path",           "/data/models/kimi-k2.5/"},
                {"weights_format",         "safetensors"},
                {"num_hidden_layers",      61},
                {"hidden_size",            7168},
                {"num_attention_heads",    64},
                {"num_key_value_heads",    64},
                {"intermediate_size",      18432},
                {"n_routed_experts",       384},
                {"n_shared_experts",       1},
                {"num_experts_per_tok",    8},
                {"n_group",                1},
                {"topk_group",             1},
                {"vocab_size",             163840},
                {"max_position_embeddings", 262144},
                {"kv_lora_rank",           512},
                {"q_lora_rank",            1536},
                {"qk_rope_head_dim",       64},
                {"qk_nope_head_dim",       128},
                {"v_head_dim",             128},
                {"first_k_dense_replace",  1},
                {"moe_layer_freq",         1},
                {"index_topk",             0},
                {"num_nextn_predict_layers", 0},
                {"rms_norm_eps",           1e-5},
                {"rope_theta",             50000.0},
                {"routed_scaling_factor",  2.827},
                {"moe_intermediate_size",  2048},
                {"vision", {{"enabled", true}, {"hidden_size", 1152},
                            {"num_layers", 27}, {"num_heads", 16},
                            {"patch_size", 14}, {"image_size", 384}}},
                {"rope_scaling", {{"type", "yarn"}, {"factor", 64.0},
                                  {"beta_fast", 32.0}, {"beta_slow", 1.0},
                                  {"mscale", 1.0}}},
            }},
            {"quantization", {{"weights", "int4_symmetric"}, {"attention_compute", "fp8_e4m3"},
                              {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
            {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                          {"system_ram_gb", 512}}},
        };
        return lc::parse_config(j);
    }()};
}

// ── Tests: DeepSeek V3.2 ─────────────────────────────────────────────────────

class ModelConfigV32Test : public ::testing::Test {
   protected:
    ModelConfig cfg = deepseek_v3_2_config();
};

TEST_F(ModelConfigV32Test, DispatchHelpers) {
    EXPECT_TRUE(cfg.has_dsa());
    EXPECT_TRUE(cfg.has_mtp());
    EXPECT_TRUE(cfg.has_grouped_routing());
    EXPECT_FALSE(cfg.has_vision());
}

TEST_F(ModelConfigV32Test, IsMoeLayer) {
    // Layers 0-2 are dense, layers 3-60 are MoE
    EXPECT_FALSE(cfg.is_moe_layer(0));
    EXPECT_FALSE(cfg.is_moe_layer(1));
    EXPECT_FALSE(cfg.is_moe_layer(2));
    EXPECT_TRUE(cfg.is_moe_layer(3));
    EXPECT_TRUE(cfg.is_moe_layer(30));
    EXPECT_TRUE(cfg.is_moe_layer(60));
}

TEST_F(ModelConfigV32Test, OutOfBoundsLayerNotMoe) {
    EXPECT_FALSE(cfg.is_moe_layer(-1));
    EXPECT_FALSE(cfg.is_moe_layer(61));
    EXPECT_FALSE(cfg.is_moe_layer(100));
}

TEST_F(ModelConfigV32Test, LayerCounts) {
    EXPECT_EQ(cfg.num_moe_layers(), 58);   // layers 3-60
    EXPECT_EQ(cfg.num_dense_layers(), 3);   // layers 0-2
    EXPECT_EQ(cfg.num_moe_layers() + cfg.num_dense_layers(), 61);
}

TEST_F(ModelConfigV32Test, MoeLayerIndices) {
    const auto& moe = cfg.moe_layer_indices();
    EXPECT_EQ(static_cast<int>(moe.size()), 58);
    EXPECT_EQ(moe.front(), 3);
    EXPECT_EQ(moe.back(), 60);
}

TEST_F(ModelConfigV32Test, DenseLayerIndices) {
    const auto& dense = cfg.dense_layer_indices();
    EXPECT_EQ(static_cast<int>(dense.size()), 3);
    EXPECT_EQ(dense[0], 0);
    EXPECT_EQ(dense[1], 1);
    EXPECT_EQ(dense[2], 2);
}

TEST_F(ModelConfigV32Test, DerivedDimensions) {
    EXPECT_EQ(cfg.qk_head_dim(), 192);     // 128 + 64
    EXPECT_EQ(cfg.kv_cache_dim(), 576);     // 512 + 64
}

TEST_F(ModelConfigV32Test, RawAccess) {
    EXPECT_EQ(cfg.raw().hidden_size, 7168);
    EXPECT_EQ(cfg.raw().num_hidden_layers, 61);
    EXPECT_EQ(cfg.raw().n_routed_experts, 256);
    EXPECT_EQ(cfg.raw().num_experts_per_tok, 8);
    EXPECT_EQ(cfg.raw().n_group, 8);
    EXPECT_EQ(cfg.raw().topk_group, 4);
    EXPECT_EQ(cfg.raw().index_topk, 2048);
    EXPECT_EQ(cfg.raw().num_nextn_predict_layers, 1);
    EXPECT_DOUBLE_EQ(cfg.raw().rms_norm_eps, 1e-6);
    EXPECT_DOUBLE_EQ(cfg.raw().rope_theta, 10000.0);
}

// ── Tests: GLM-5 ─────────────────────────────────────────────────────────────

class ModelConfigGLM5Test : public ::testing::Test {
   protected:
    ModelConfig cfg = glm5_config();
};

TEST_F(ModelConfigGLM5Test, DispatchHelpers) {
    EXPECT_TRUE(cfg.has_dsa());
    EXPECT_TRUE(cfg.has_mtp());
    EXPECT_FALSE(cfg.has_grouped_routing());
    EXPECT_FALSE(cfg.has_vision());
}

TEST_F(ModelConfigGLM5Test, IsMoeLayer) {
    EXPECT_FALSE(cfg.is_moe_layer(0));
    EXPECT_FALSE(cfg.is_moe_layer(2));
    EXPECT_TRUE(cfg.is_moe_layer(3));
    EXPECT_TRUE(cfg.is_moe_layer(77));
}

TEST_F(ModelConfigGLM5Test, LayerCounts) {
    EXPECT_EQ(cfg.num_moe_layers(), 75);   // layers 3-77
    EXPECT_EQ(cfg.num_dense_layers(), 3);   // layers 0-2
    EXPECT_EQ(cfg.num_moe_layers() + cfg.num_dense_layers(), 78);
}

TEST_F(ModelConfigGLM5Test, DerivedDimensions) {
    EXPECT_EQ(cfg.qk_head_dim(), 256);     // 192 + 64
    EXPECT_EQ(cfg.kv_cache_dim(), 576);     // 512 + 64
}

TEST_F(ModelConfigGLM5Test, RopeInterleave) {
    EXPECT_TRUE(cfg.raw().rope_interleave);
    EXPECT_TRUE(cfg.raw().indexer_rope_interleave);
}

TEST_F(ModelConfigGLM5Test, IndexShareFullLayerSet) {
    // GLM-5.2's real IndexShare pattern (from the HF config's indexer_types,
    // dropped by the GGUF): 21 full layers, reconstructed from freq=4/offset=3.
    const std::vector<int> expected_full = {
        0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34,
        38, 42, 46, 50, 54, 58, 62, 66, 70, 74};
    EXPECT_EQ(cfg.num_full_index_layers(), 21);

    std::vector<int> got_full;
    for (int l = 0; l < 78; ++l)
        if (cfg.is_full_index_layer(l)) got_full.push_back(l);
    EXPECT_EQ(got_full, expected_full);

    // Spot-check shared layers reuse (not full).
    EXPECT_FALSE(cfg.is_full_index_layer(3));
    EXPECT_FALSE(cfg.is_full_index_layer(7));
    EXPECT_FALSE(cfg.is_full_index_layer(77));
    // Mask agrees with the query.
    const auto& mask = cfg.full_index_layer_mask();
    ASSERT_EQ(static_cast<int>(mask.size()), 78);
    for (int l = 0; l < 78; ++l)
        EXPECT_EQ(mask[l], cfg.is_full_index_layer(l)) << "layer " << l;
    // Out of bounds.
    EXPECT_FALSE(cfg.is_full_index_layer(-1));
    EXPECT_FALSE(cfg.is_full_index_layer(78));
}

TEST_F(ModelConfigGLM5Test, RawValues) {
    EXPECT_EQ(cfg.raw().hidden_size, 6144);
    EXPECT_EQ(cfg.raw().num_hidden_layers, 78);
    EXPECT_EQ(cfg.raw().q_lora_rank, 2048);
    EXPECT_EQ(cfg.raw().v_head_dim, 256);
    EXPECT_DOUBLE_EQ(cfg.raw().rms_norm_eps, 1e-5);
    EXPECT_DOUBLE_EQ(cfg.raw().rope_theta, 1000000.0);
}

// ── Tests: Kimi K2.5 ────────────────────────────────────────────────────────

class ModelConfigK25Test : public ::testing::Test {
   protected:
    ModelConfig cfg = kimi_k25_config();
};

TEST_F(ModelConfigK25Test, DispatchHelpers) {
    EXPECT_FALSE(cfg.has_dsa());
    EXPECT_FALSE(cfg.has_mtp());
    EXPECT_FALSE(cfg.has_grouped_routing());
    EXPECT_TRUE(cfg.has_vision());
}

TEST_F(ModelConfigK25Test, IsMoeLayer) {
    // first_k_dense_replace = 1, so only layer 0 is dense
    EXPECT_FALSE(cfg.is_moe_layer(0));
    EXPECT_TRUE(cfg.is_moe_layer(1));
    EXPECT_TRUE(cfg.is_moe_layer(60));
}

TEST_F(ModelConfigK25Test, LayerCounts) {
    EXPECT_EQ(cfg.num_moe_layers(), 60);   // layers 1-60
    EXPECT_EQ(cfg.num_dense_layers(), 1);   // layer 0
}

TEST_F(ModelConfigK25Test, DerivedDimensions) {
    EXPECT_EQ(cfg.qk_head_dim(), 192);     // 128 + 64
    EXPECT_EQ(cfg.kv_cache_dim(), 576);     // 512 + 64
}

TEST_F(ModelConfigK25Test, VisionConfig) {
    ASSERT_TRUE(cfg.raw().vision.has_value());
    EXPECT_TRUE(cfg.raw().vision->enabled);
    EXPECT_EQ(cfg.raw().vision->hidden_size, 1152);
    EXPECT_EQ(cfg.raw().vision->num_layers, 27);
    EXPECT_EQ(cfg.raw().vision->num_heads, 16);
    EXPECT_EQ(cfg.raw().vision->patch_size, 14);
    EXPECT_EQ(cfg.raw().vision->image_size, 384);
}

TEST_F(ModelConfigK25Test, RopeScaling) {
    ASSERT_TRUE(cfg.raw().rope_scaling.has_value());
    EXPECT_EQ(cfg.raw().rope_scaling->type, lc::RopeScalingType::yarn);
    EXPECT_DOUBLE_EQ(cfg.raw().rope_scaling->factor.value(), 64.0);
}

TEST_F(ModelConfigK25Test, RawValues) {
    EXPECT_EQ(cfg.raw().n_routed_experts, 384);
    EXPECT_EQ(cfg.raw().first_k_dense_replace, 1);
    EXPECT_EQ(cfg.raw().num_nextn_predict_layers, 0);
    EXPECT_DOUBLE_EQ(cfg.raw().rope_theta, 50000.0);
    EXPECT_DOUBLE_EQ(cfg.raw().routed_scaling_factor, 2.827);
}

// ── Tests: Edge cases ────────────────────────────────────────────────────────

TEST(ModelConfigEdge, MoeLayerFreqGreaterThanOne) {
    // Hypothetical model where only every other layer after dense is MoE
    auto j = nlohmann::json{
        {"model", {
            {"architecture",           "deepseek_v3"},
            {"weights_path",           "/tmp/test/"},
            {"weights_format",         "safetensors"},
            {"num_hidden_layers",      10},
            {"hidden_size",            7168},
            {"num_attention_heads",    128},
            {"num_key_value_heads",    128},
            {"intermediate_size",      18432},
            {"n_routed_experts",       256},
            {"num_experts_per_tok",    8},
            {"vocab_size",             129280},
            {"max_position_embeddings", 163840},
            {"first_k_dense_replace",  2},
            {"moe_layer_freq",         2},
        }},
        {"quantization", {{"weights", "nvfp4"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                      {"system_ram_gb", 256}}},
    };
    ModelConfig cfg{lc::parse_config(j)};

    // Dense: 0, 1 (before first_k_dense_replace)
    // After first_k_dense_replace=2, freq=2: MoE at 2, 4, 6, 8. Dense at 3, 5, 7, 9.
    EXPECT_FALSE(cfg.is_moe_layer(0));
    EXPECT_FALSE(cfg.is_moe_layer(1));
    EXPECT_TRUE(cfg.is_moe_layer(2));
    EXPECT_FALSE(cfg.is_moe_layer(3));
    EXPECT_TRUE(cfg.is_moe_layer(4));
    EXPECT_FALSE(cfg.is_moe_layer(5));
    EXPECT_TRUE(cfg.is_moe_layer(6));
    EXPECT_FALSE(cfg.is_moe_layer(7));
    EXPECT_TRUE(cfg.is_moe_layer(8));
    EXPECT_FALSE(cfg.is_moe_layer(9));

    EXPECT_EQ(cfg.num_moe_layers(), 4);   // 2, 4, 6, 8
    EXPECT_EQ(cfg.num_dense_layers(), 6); // 0, 1, 3, 5, 7, 9
}

TEST(ModelConfigEdge, AllDenseLayers) {
    // Model with no MoE layers (first_k_dense_replace >= num_hidden_layers)
    auto j = nlohmann::json{
        {"model", {
            {"architecture",           "deepseek_v3"},
            {"weights_path",           "/tmp/test/"},
            {"weights_format",         "safetensors"},
            {"num_hidden_layers",      10},
            {"hidden_size",            7168},
            {"num_attention_heads",    128},
            {"num_key_value_heads",    128},
            {"intermediate_size",      18432},
            {"n_routed_experts",       256},
            {"num_experts_per_tok",    8},
            {"vocab_size",             129280},
            {"max_position_embeddings", 163840},
            {"first_k_dense_replace",  10},
        }},
        {"quantization", {{"weights", "nvfp4"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                      {"system_ram_gb", 256}}},
    };
    ModelConfig cfg{lc::parse_config(j)};

    for (int l = 0; l < 10; ++l) {
        EXPECT_FALSE(cfg.is_moe_layer(l)) << "layer " << l;
    }
    EXPECT_EQ(cfg.num_moe_layers(), 0);
    EXPECT_EQ(cfg.num_dense_layers(), 10);
}

TEST(ModelConfigEdge, IndexShareDefaultAllFull) {
    // GGUF default / llama.cpp reference: index_topk_freq unset (0) ⇒ every
    // layer is full (no sharing).
    auto j = nlohmann::json{
        {"model", {
            {"architecture",           "glm_moe_dsa"},
            {"weights_path",           "/tmp/test/"},
            {"weights_format",         "gguf"},
            {"num_hidden_layers",      12},
            {"hidden_size",            6144},
            {"num_attention_heads",    64},
            {"num_key_value_heads",    64},
            {"intermediate_size",      12288},
            {"n_routed_experts",       256},
            {"num_experts_per_tok",    8},
            {"vocab_size",             154880},
            {"max_position_embeddings", 202752},
            {"index_topk",             2048},
        }},
        {"quantization", {{"weights", "gguf"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                      {"system_ram_gb", 256}}},
    };
    ModelConfig cfg{lc::parse_config(j)};
    EXPECT_EQ(cfg.num_full_index_layers(), 12);
    for (int l = 0; l < 12; ++l)
        EXPECT_TRUE(cfg.is_full_index_layer(l)) << "layer " << l;
}

TEST(ModelConfigEdge, VisionDisabled) {
    auto j = nlohmann::json{
        {"model", {
            {"architecture",           "kimi_k25"},
            {"weights_path",           "/tmp/test/"},
            {"weights_format",         "safetensors"},
            {"num_hidden_layers",      61},
            {"hidden_size",            7168},
            {"num_attention_heads",    64},
            {"num_key_value_heads",    64},
            {"intermediate_size",      18432},
            {"n_routed_experts",       384},
            {"num_experts_per_tok",    8},
            {"vocab_size",             163840},
            {"max_position_embeddings", 262144},
            {"vision", {{"enabled", false}}},
        }},
        {"quantization", {{"weights", "nvfp4"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                      {"system_ram_gb", 256}}},
    };
    ModelConfig cfg{lc::parse_config(j)};
    EXPECT_FALSE(cfg.has_vision());
}

TEST(ModelConfigEdge, NoVisionField) {
    auto j = nlohmann::json{
        {"model", {
            {"architecture",           "deepseek_v3"},
            {"weights_path",           "/tmp/test/"},
            {"weights_format",         "safetensors"},
            {"num_hidden_layers",      61},
            {"hidden_size",            7168},
            {"num_attention_heads",    128},
            {"num_key_value_heads",    128},
            {"intermediate_size",      18432},
            {"n_routed_experts",       256},
            {"num_experts_per_tok",    8},
            {"vocab_size",             129280},
            {"max_position_embeddings", 163840},
        }},
        {"quantization", {{"weights", "nvfp4"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                      {"system_ram_gb", 256}}},
    };
    ModelConfig cfg{lc::parse_config(j)};
    EXPECT_FALSE(cfg.has_vision());
}

TEST(ModelConfigEdge, ConstructFromModelConfigDirectly) {
    lc::ModelConfig raw;
    raw.architecture = lc::Architecture::deepseek_v3;
    raw.weights_path = "/tmp/test/";
    raw.weights_format = lc::WeightsFormat::safetensors;
    raw.num_hidden_layers = 61;
    raw.hidden_size = 7168;
    raw.num_attention_heads = 128;
    raw.num_key_value_heads = 128;
    raw.intermediate_size = 18432;
    raw.n_routed_experts = 256;
    raw.num_experts_per_tok = 8;
    raw.vocab_size = 129280;
    raw.max_position_embeddings = 163840;
    raw.first_k_dense_replace = 3;
    raw.moe_layer_freq = 1;
    raw.index_topk = 2048;
    raw.num_nextn_predict_layers = 1;
    raw.n_group = 8;

    ModelConfig cfg{raw};

    EXPECT_TRUE(cfg.has_dsa());
    EXPECT_TRUE(cfg.has_mtp());
    EXPECT_TRUE(cfg.has_grouped_routing());
    EXPECT_FALSE(cfg.has_vision());
    EXPECT_EQ(cfg.num_moe_layers(), 58);
    EXPECT_EQ(cfg.num_dense_layers(), 3);
}

// ── DeepSeek-V4 dispatch helpers (spec/DEEPSEEK4_PLAN.md V4-1b) ──────────────

using layerstorm::model::V4AttentionType;

static ModelConfig deepseek_v4_config() {
    return ModelConfig{[] {
        std::vector<int> ratios{0, 0};
        for (int i = 0; i < 20; ++i) { ratios.push_back(4); ratios.push_back(128); }
        ratios.push_back(4);  // 43 entries: [0,0,(4,128)x20,4]
        auto j = nlohmann::json{
            {"model", {
                {"architecture",           "deepseek_v4"},
                {"weights_path",           "/data/models/deepseek-v4-flash/"},
                {"weights_format",         "gguf"},
                {"num_hidden_layers",      43},
                {"hidden_size",            4096},
                {"num_attention_heads",    64},
                {"num_key_value_heads",    1},
                {"head_dim",               512},
                {"qk_rope_head_dim",       64},
                {"q_lora_rank",            1024},
                {"intermediate_size",      2048},
                {"n_routed_experts",       256},
                {"n_shared_experts",       1},
                {"num_experts_per_tok",    6},
                {"n_group",                1},
                {"topk_group",             1},
                {"vocab_size",             129280},
                {"max_position_embeddings", 1048576},
                {"first_k_dense_replace",  0},
                {"moe_layer_freq",         1},
                {"compress_ratios",        ratios},
                {"compress_rope_theta",    160000.0},
                {"sliding_window",         128},
                {"swiglu_limit",           10.0},
                {"num_hash_layers",        3},
                {"o_groups",               8},
                {"o_lora_rank",            1024},
                {"hc_mult",                4},
                {"hc_sinkhorn_iters",      20},
                {"hc_eps",                 1e-6},
                {"gating_score_fn",        "sqrtsoftplus"},
                {"index_topk",             512},
                {"index_n_heads",          64},
                {"index_head_dim",         128},
                {"num_nextn_predict_layers", 1},
                {"routed_scaling_factor",  1.5},
                {"moe_intermediate_size",  2048},
            }},
            {"quantization", {{"weights", "nvfp4"}, {"attention_compute", "fp8_e4m3"},
                              {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
            {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                          {"system_ram_gb", 256}}},
        };
        return lc::parse_config(j);
    }()};
}

TEST(ModelConfigV4, DispatchFlags) {
    auto cfg = deepseek_v4_config();
    EXPECT_TRUE(cfg.is_v4());
    EXPECT_FALSE(cfg.uses_mla());
    EXPECT_TRUE(cfg.has_csa_hca());
    EXPECT_TRUE(cfg.has_mhc());
    EXPECT_TRUE(cfg.has_grouped_o_proj());
    // V4's index_topk drives the Lightning Indexer INSIDE the CSA pipeline —
    // has_dsa() must be false despite index_topk = 512.
    EXPECT_FALSE(cfg.has_dsa());
    EXPECT_TRUE(cfg.has_mtp());
    EXPECT_FALSE(cfg.has_grouped_routing());  // n_group = 1
}

TEST(ModelConfigV4, AttentionTypeForLayer) {
    auto cfg = deepseek_v4_config();
    EXPECT_EQ(cfg.attention_type_for_layer(0),  V4AttentionType::kSwa);
    EXPECT_EQ(cfg.attention_type_for_layer(1),  V4AttentionType::kSwa);
    EXPECT_EQ(cfg.attention_type_for_layer(2),  V4AttentionType::kCsa);
    EXPECT_EQ(cfg.attention_type_for_layer(3),  V4AttentionType::kHca);
    EXPECT_EQ(cfg.attention_type_for_layer(40), V4AttentionType::kCsa);
    EXPECT_EQ(cfg.attention_type_for_layer(41), V4AttentionType::kHca);
    EXPECT_EQ(cfg.attention_type_for_layer(42), V4AttentionType::kCsa);
    // Out-of-range: the no-compression type.
    EXPECT_EQ(cfg.attention_type_for_layer(-1), V4AttentionType::kSwa);
    EXPECT_EQ(cfg.attention_type_for_layer(43), V4AttentionType::kSwa);
}

TEST(ModelConfigV4, DualRopeRule) {
    auto cfg = deepseek_v4_config();
    // Uncompressed layers: plain rope_theta, NO yarn (dossier §2.3).
    EXPECT_FALSE(cfg.layer_uses_compress_rope(0));
    EXPECT_FALSE(cfg.layer_uses_compress_rope(1));
    // Compressed layers: compress_rope_theta WITH yarn.
    EXPECT_TRUE(cfg.layer_uses_compress_rope(2));
    EXPECT_TRUE(cfg.layer_uses_compress_rope(3));
    EXPECT_TRUE(cfg.layer_uses_compress_rope(42));
}

TEST(ModelConfigV4, HashLayers) {
    auto cfg = deepseek_v4_config();
    EXPECT_TRUE(cfg.is_hash_layer(0));
    EXPECT_TRUE(cfg.is_hash_layer(1));
    EXPECT_TRUE(cfg.is_hash_layer(2));
    EXPECT_FALSE(cfg.is_hash_layer(3));
    EXPECT_FALSE(cfg.is_hash_layer(-1));
}

TEST(ModelConfigV4, AllLayersMoe) {
    auto cfg = deepseek_v4_config();  // first_k_dense_replace = 0
    EXPECT_EQ(cfg.num_moe_layers(), 43);
    EXPECT_EQ(cfg.num_dense_layers(), 0);
}

TEST(ModelConfigV4, V32Regression) {
    // No-collateral-damage: V3.2 helpers unchanged by the V4 additions.
    auto cfg = deepseek_v3_2_config();
    EXPECT_FALSE(cfg.is_v4());
    EXPECT_TRUE(cfg.uses_mla());
    EXPECT_FALSE(cfg.has_csa_hca());
    EXPECT_FALSE(cfg.has_mhc());
    EXPECT_FALSE(cfg.has_grouped_o_proj());
    EXPECT_FALSE(cfg.is_hash_layer(0));
    EXPECT_TRUE(cfg.has_dsa());  // index_topk 2048, non-V4
    EXPECT_EQ(cfg.attention_type_for_layer(0), V4AttentionType::kSwa);
    EXPECT_FALSE(cfg.layer_uses_compress_rope(0));
}
