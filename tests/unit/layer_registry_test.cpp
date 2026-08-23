#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

#include "model/layer_registry.h"
#include "model/quantization/fp8.h"
#include "model/quantization/nvfp4.h"
#include "model/quantization/registry.h"

namespace lc = layerstorm::config;
using layerstorm::model::LayerInfo;
using layerstorm::model::LayerRegistry;
using layerstorm::model::ModelConfig;
using layerstorm::model::QuantInterface;

// ── Helpers: build full Config objects for each architecture ─────────────────

static lc::Config deepseek_v3_2_full_config() {
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
}

static lc::Config glm5_full_config() {
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
}

static lc::Config kimi_k25_full_config() {
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
        {"quantization", {{"weights", "nvfp4"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                      {"system_ram_gb", 512}}},
    };
    return lc::parse_config(j);
}

// ── Test fixture ────────────────────────────────────────────────────────────

class LayerRegistryV32Test : public ::testing::Test {
protected:
    void SetUp() override {
        cfg_ = deepseek_v3_2_full_config();
        model_cfg_ = std::make_unique<ModelConfig>(cfg_);
        nvfp4_ = std::make_unique<layerstorm::model::Nvfp4>();
        reg_ = std::make_unique<LayerRegistry>(*model_cfg_, cfg_, *nvfp4_);
    }

    lc::Config cfg_;
    std::unique_ptr<ModelConfig> model_cfg_;
    std::unique_ptr<layerstorm::model::Nvfp4> nvfp4_;
    std::unique_ptr<LayerRegistry> reg_;
};

// ── Test 1: DeepSeek V3.2 layer counts ──────────────────────────────────────

TEST_F(LayerRegistryV32Test, LayerCounts) {
    EXPECT_EQ(reg_->num_layers(), 61);
    EXPECT_EQ(reg_->num_moe_layers(), 58);
    EXPECT_EQ(reg_->num_dense_layers(), 3);
}

// ── Test 2: Attention size ──────────────────────────────────────────────────

TEST_F(LayerRegistryV32Test, AttentionSize) {
    // MLA projections: q_a=11010048, q_b=37748736, kv_a=4128768, kv_b=16777216,
    // o=117440512, norms=16384. Total=187121664 params * 1 byte (FP8).
    // DSA indexer: q_idx_b=12582912, k_idx=917504, k_idx_norm=128,
    // k_idx_norm_bias=128, weights_proj=458752. Total=13959424 params * 1 byte (FP8).
    // Grand total: 187121664 + 13959424 = 201081088.
    for (int l = 0; l < reg_->num_layers(); ++l) {
        EXPECT_EQ(reg_->layer(l).attention_bytes, 201'081'088) << "layer " << l;
    }
}

// ── Test 3: Gating size ─────────────────────────────────────────────────────

TEST_F(LayerRegistryV32Test, GatingSize) {
    // hidden(7168) * n_routed(256) = 1835008 params * 4 bytes (FP32) = 7340032.
    for (int l = 0; l < reg_->num_layers(); ++l) {
        if (reg_->layer(l).is_moe) {
            EXPECT_EQ(reg_->layer(l).gating_bytes, 7'340'032) << "layer " << l;
        } else {
            EXPECT_EQ(reg_->layer(l).gating_bytes, 0) << "layer " << l;
        }
    }
}

// ── Test 4: Routed expert size ──────────────────────────────────────────────

TEST_F(LayerRegistryV32Test, RoutedExpertSize) {
    // ExpertShape{7168, 2048}: 3 projections, NVFP4 accounting.
    // Per projection: weight_bytes=7340032 + scale_bytes=917504 + scalars=8 = 8257544.
    // Total: 3 * 8257544 = 24772632.
    EXPECT_EQ(reg_->per_routed_expert_bytes(), 24'772'992);
    for (int l = 0; l < reg_->num_layers(); ++l) {
        if (reg_->layer(l).is_moe) {
            EXPECT_EQ(reg_->layer(l).per_routed_expert_bytes, 24'772'992) << "layer " << l;
        } else {
            EXPECT_EQ(reg_->layer(l).per_routed_expert_bytes, 0) << "layer " << l;
        }
    }
}

// ── Test 5: Shared expert size ──────────────────────────────────────────────

TEST_F(LayerRegistryV32Test, SharedExpertSize) {
    // Same shape as routed * n_shared_experts(1) = 24772632.
    for (int l = 0; l < reg_->num_layers(); ++l) {
        if (reg_->layer(l).is_moe) {
            EXPECT_EQ(reg_->layer(l).shared_expert_bytes, 24'772'992) << "layer " << l;
        } else {
            EXPECT_EQ(reg_->layer(l).shared_expert_bytes, 0) << "layer " << l;
        }
    }
}

// ── Test 6: Dense FFN size ──────────────────────────────────────────────────

TEST_F(LayerRegistryV32Test, DenseFfnSize) {
    // ExpertShape{7168, 18432}: per projection: weight=66060288 + scale=8257536 + scalars=8 = 74317832.
    // Total: 3 * 74317832 = 222953496.
    for (int l = 0; l < reg_->num_layers(); ++l) {
        if (!reg_->layer(l).is_moe) {
            EXPECT_EQ(reg_->layer(l).ffn_bytes, 222'953'856) << "layer " << l;
        } else {
            EXPECT_EQ(reg_->layer(l).ffn_bytes, 0) << "layer " << l;
        }
    }
}

// ── Test 7: Embedding size ──────────────────────────────────────────────────

TEST_F(LayerRegistryV32Test, EmbeddingSize) {
    // vocab(129280) * hidden(7168) * 2.0 (BF16) = 1853358080.
    EXPECT_EQ(reg_->embedding_bytes(), 1'853'358'080);
}

// ── Test 8: Output head size ────────────────────────────────────────────────

TEST_F(LayerRegistryV32Test, OutputHeadSize) {
    // vocab*hidden * 2.0 = 926679040 * 2 = 1853358080 (no bias for V3.2).
    EXPECT_EQ(reg_->output_head_bytes(), 1'853'358'080);
}

// ── Test 9: Pinning — attention "all" ───────────────────────────────────────

TEST_F(LayerRegistryV32Test, PinningAttentionAll) {
    for (int l = 0; l < reg_->num_layers(); ++l) {
        EXPECT_TRUE(reg_->layer(l).attention_pinned) << "layer " << l;
    }
}

// ── Test 10: Pinning — dense FFN ────────────────────────────────────────────

TEST_F(LayerRegistryV32Test, PinningDenseFfn) {
    // Default dense_ffn_layers = {0, 1, 2, 59, 60}.
    // Layers 0-2 are dense -> pinned. Layers 59-60 are MoE -> NOT pinned.
    EXPECT_TRUE(reg_->layer(0).ffn_pinned);
    EXPECT_TRUE(reg_->layer(1).ffn_pinned);
    EXPECT_TRUE(reg_->layer(2).ffn_pinned);
    // MoE layers have ffn_bytes=0 and ffn_pinned=false
    EXPECT_FALSE(reg_->layer(59).ffn_pinned);
    EXPECT_FALSE(reg_->layer(60).ffn_pinned);
    // Random MoE layer
    EXPECT_FALSE(reg_->layer(30).ffn_pinned);
}

// ── Test 11: Pinning — gating "all" ─────────────────────────────────────────

TEST_F(LayerRegistryV32Test, PinningGatingAll) {
    for (int l = 0; l < reg_->num_layers(); ++l) {
        if (reg_->layer(l).is_moe) {
            EXPECT_TRUE(reg_->layer(l).gating_pinned) << "layer " << l;
        } else {
            EXPECT_FALSE(reg_->layer(l).gating_pinned) << "layer " << l;
        }
    }
}

// ── Test 12: Pinning — shared experts always pinned ─────────────────────────

TEST_F(LayerRegistryV32Test, PinningSharedExpertsAlwaysPinned) {
    for (int l = 0; l < reg_->num_layers(); ++l) {
        if (reg_->layer(l).is_moe) {
            EXPECT_TRUE(reg_->layer(l).shared_expert_pinned) << "layer " << l;
        } else {
            EXPECT_FALSE(reg_->layer(l).shared_expert_pinned) << "layer " << l;
        }
    }
}

// ── Test 13: Pinning — embedding/output_head ────────────────────────────────

TEST_F(LayerRegistryV32Test, PinningEmbeddingOutputHead) {
    EXPECT_TRUE(reg_->embedding_pinned());
    EXPECT_TRUE(reg_->output_head_pinned());
}

// ── Test 14: total_pinned_bytes ─────────────────────────────────────────────

TEST_F(LayerRegistryV32Test, TotalPinnedBytes) {
    // attn:   61 * 201081088 = 12265946368  (includes DSA indexer + k_norm_bias)
    // gating: 58 * 7340032   =   425721856
    // dense:   3 * 222953496 =   668860488  (+8 per projection for NVFP4 scalars)
    // shared: 58 * 24772632  =  1436812656  (+8 per projection for NVFP4 scalars)
    // embed:                    1853358080
    // output:                   1853358080  (no lm_head.bias for V3.2)
    // total:                   18504057528
    EXPECT_EQ(reg_->total_pinned_bytes(), 18'504'079'488LL);
}

// ── Test 15: GLM-5 layer counts ─────────────────────────────────────────────

TEST(LayerRegistryGLM5, LayerCounts) {
    auto cfg = glm5_full_config();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    LayerRegistry reg{model_cfg, cfg, nvfp4};

    EXPECT_EQ(reg.num_layers(), 78);
    EXPECT_EQ(reg.num_moe_layers(), 75);
    EXPECT_EQ(reg.num_dense_layers(), 3);
}

// ── Test 16: K2.5 layer counts ──────────────────────────────────────────────

TEST(LayerRegistryK25, LayerCounts) {
    auto cfg = kimi_k25_full_config();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    LayerRegistry reg{model_cfg, cfg, nvfp4};

    EXPECT_EQ(reg.num_layers(), 61);
    EXPECT_EQ(reg.num_moe_layers(), 60);
    EXPECT_EQ(reg.num_dense_layers(), 1);
}

// ── Test 17: Custom pinning — attention as list ─────────────────────────────

TEST(LayerRegistryCustomPinning, AttentionAsList) {
    auto cfg = deepseek_v3_2_full_config();
    cfg.memory.pinned_layers.attention = std::vector<int>{0, 5, 10};
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    LayerRegistry reg{model_cfg, cfg, nvfp4};

    EXPECT_TRUE(reg.layer(0).attention_pinned);
    EXPECT_FALSE(reg.layer(1).attention_pinned);
    EXPECT_FALSE(reg.layer(4).attention_pinned);
    EXPECT_TRUE(reg.layer(5).attention_pinned);
    EXPECT_TRUE(reg.layer(10).attention_pinned);
    EXPECT_FALSE(reg.layer(11).attention_pinned);
    EXPECT_FALSE(reg.layer(60).attention_pinned);
}

// ── Test 18: Custom pinning — embedding=false ───────────────────────────────

TEST(LayerRegistryCustomPinning, EmbeddingFalse) {
    auto cfg = deepseek_v3_2_full_config();
    cfg.memory.pinned_layers.embedding = false;
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    LayerRegistry reg{model_cfg, cfg, nvfp4};

    EXPECT_FALSE(reg.embedding_pinned());
    // total_pinned should be reduced by embedding_bytes
    auto cfg2 = deepseek_v3_2_full_config();
    ModelConfig model_cfg2{cfg2};
    LayerRegistry reg2{model_cfg2, cfg2, nvfp4};

    EXPECT_EQ(reg.total_pinned_bytes(),
              reg2.total_pinned_bytes() - reg2.embedding_bytes());
}

// ── Test 19: Per-GPU budget estimate ────────────────────────────────────────

TEST(LayerRegistryBudget, MultiGpuBudget) {
    auto cfg = deepseek_v3_2_full_config();
    // 2x5090 + 2x5080, TP on 5090 pair
    cfg.hardware.gpus = {
        {0, lc::GpuType::rtx5090, 32.0},
        {1, lc::GpuType::rtx5090, 32.0},
        {2, lc::GpuType::rtx5080, 16.0},
        {3, lc::GpuType::rtx5080, 16.0},
    };
    cfg.hardware.tp_array = {0, 1};

    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    LayerRegistry reg{model_cfg, cfg, nvfp4};

    auto budgets = reg.estimate_gpu_budgets();
    ASSERT_EQ(budgets.size(), 4u);

    // KD-3d-fix: budget now uses PinnedRegionLayout — accounts for:
    //   - kv_b_proj excluded (absorbed by SnapMLA/TQ attention backends)
    //   - BF16 norms (q_a_norm, kv_a_norm) at 2 bytes/elem (not flat FP8)
    //   - Layer norms (input_layernorm + post_attention_layernorm): 2*H*2 per layer
    //   - Final norm: H*2 bytes
    //   - MTP block layer attention + norms (1 layer for V3.2)
    //   - Gating replicated (not ÷tp), shared expert/dense FFN TP-sharded (÷2)
    //
    // Verify the layout is what drives the budget, and it's within expected range.
    const auto& layout = reg.pinned_layout();
    EXPECT_GT(layout.total_bytes, 0);
    EXPECT_GT(layout.attention_bytes, 0);
    EXPECT_GT(layout.layer_norm_bytes, 0);
    EXPECT_GT(layout.gating_bytes, 0);
    EXPECT_GT(layout.shared_expert_bytes, 0);
    EXPECT_GT(layout.final_norm_bytes, 0);
    EXPECT_GT(layout.mtp_bytes, 0);
    int64_t expected_per_tp = layout.total_bytes;

    // GPU 0 (5090): has pinned bytes
    EXPECT_EQ(budgets[0].gpu_id, 0);
    EXPECT_EQ(budgets[0].pinned_bytes, expected_per_tp);
    EXPECT_GT(budgets[0].available_for_cache_bytes, 0);

    // GPU 1 (5090): same as GPU 0
    EXPECT_EQ(budgets[1].gpu_id, 1);
    EXPECT_EQ(budgets[1].pinned_bytes, expected_per_tp);
    EXPECT_GT(budgets[1].available_for_cache_bytes, 0);

    // GPU 2 (5080): no pinned bytes
    EXPECT_EQ(budgets[2].gpu_id, 2);
    EXPECT_EQ(budgets[2].pinned_bytes, 0);
    EXPECT_EQ(budgets[2].available_for_cache_bytes, budgets[2].total_vram_bytes);

    // GPU 3 (5080): no pinned bytes
    EXPECT_EQ(budgets[3].gpu_id, 3);
    EXPECT_EQ(budgets[3].pinned_bytes, 0);
    EXPECT_EQ(budgets[3].available_for_cache_bytes, budgets[3].total_vram_bytes);

    // With MLA replication, sum of per-GPU pinned > total_pinned_bytes
    // (replicated components counted on each TP GPU)
    int64_t sum_pinned = 0;
    for (const auto& b : budgets) sum_pinned += b.pinned_bytes;
    EXPECT_GT(sum_pinned, reg.total_pinned_bytes());

    // All budgets non-negative
    for (const auto& b : budgets) {
        EXPECT_GE(b.available_for_cache_bytes, 0) << "gpu " << b.gpu_id;
    }
}

// ── Test: Budget with no TP array ───────────────────────────────────────────

TEST(LayerRegistryBudget, NoTpArray) {
    auto cfg = deepseek_v3_2_full_config();
    cfg.hardware.gpus = {
        {0, lc::GpuType::rtx5090, 32.0},
        {1, lc::GpuType::rtx5090, 32.0},
    };
    cfg.hardware.tp_array = {};

    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    LayerRegistry reg{model_cfg, cfg, nvfp4};

    auto budgets = reg.estimate_gpu_budgets();
    ASSERT_EQ(budgets.size(), 2u);

    // Pinned distributed by VRAM weight (equal GPUs → equal share)
    EXPECT_EQ(budgets[0].pinned_bytes, budgets[1].pinned_bytes);
    EXPECT_GT(budgets[0].pinned_bytes, 0);

    // All budgets non-negative
    for (const auto& b : budgets) {
        EXPECT_GE(b.available_for_cache_bytes, 0) << "gpu " << b.gpu_id;
    }
}

// ── Test: total_routed_experts ──────────────────────────────────────────────

TEST_F(LayerRegistryV32Test, TotalRoutedExperts) {
    EXPECT_EQ(reg_->total_routed_experts(), 256);
}

// ── Test: LayerInfo::pinned_bytes ───────────────────────────────────────────

TEST_F(LayerRegistryV32Test, LayerInfoPinnedBytes) {
    // Dense layer 0: attention pinned + ffn pinned
    const auto& dense = reg_->layer(0);
    EXPECT_EQ(dense.pinned_bytes(), dense.attention_bytes + dense.ffn_bytes);

    // MoE layer 3: attention + gating + shared expert (all pinned)
    const auto& moe = reg_->layer(3);
    EXPECT_EQ(moe.pinned_bytes(),
              moe.attention_bytes + moe.gating_bytes + moe.shared_expert_bytes);
}

// ── Test: FP8 expert quant gives different sizes ────────────────────────────

TEST(LayerRegistryFP8, DifferentExpertQuant) {
    auto cfg = deepseek_v3_2_full_config();
    cfg.quantization.weights = lc::WeightQuant::fp8_e4m3;
    ModelConfig model_cfg{cfg};
    layerstorm::model::Fp8E4M3 fp8;
    LayerRegistry reg{model_cfg, cfg, fp8};

    // FP8: 1 byte/element + blockwise scales, ExpertShape{7168,2048}:
    // 3*(7168*2048 + 16*56*4) = 3*14683648 = 44050944 bytes
    EXPECT_EQ(reg.per_routed_expert_bytes(), 44'050'944);
    // Much larger than NVFP4's 24772608
}

// ── Test: FP16 attention quant doubles attention size ────────────────────────

TEST(LayerRegistryFP16Attn, AttentionFP16) {
    auto cfg = deepseek_v3_2_full_config();
    cfg.quantization.attention_compute = lc::AttentionQuant::fp16;
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    LayerRegistry reg{model_cfg, cfg, nvfp4};

    // FP16: 2 bytes/element -> double the FP8 size (incl. indexer projections + k_norm_bias)
    EXPECT_EQ(reg.layer(0).attention_bytes, 201'081'088LL * 2);
}

// ── Test: FP16 gating quant halves gating size ──────────────────────────────

TEST(LayerRegistryFP16Gate, GatingFP16) {
    auto cfg = deepseek_v3_2_full_config();
    cfg.quantization.gating_compute = lc::GatingQuant::fp16;
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    LayerRegistry reg{model_cfg, cfg, nvfp4};

    // FP16: 2 bytes/element -> half of FP32
    EXPECT_EQ(reg.layer(3).gating_bytes, 7'340'032LL / 2);
}

// ── Test: V3.2 attention includes DSA indexer weights ────────────────────────

TEST_F(LayerRegistryV32Test, DsaIndexerWeightsIncluded) {
    // V3.2 has DSA (index_topk=2048). Every layer's attention_bytes includes
    // indexer projection weights: q_idx_b + k_idx + k_idx_norm + k_idx_norm_bias + weights_proj.
    // Indexer params: 1536*64*128 + 7168*128 + 128 + 128 + 7168*64 = 13,959,424
    // At FP8 (1 byte/elem) = 13,959,424 bytes
    int64_t mla_base = 187'121'664;  // MLA projections only
    int64_t indexer_extra = 13'959'424;
    EXPECT_EQ(reg_->layer(0).attention_bytes, mla_base + indexer_extra);
}

// ── Test: Non-DSA model (K2.5) has no indexer in attention ──────────────────

TEST(LayerRegistryK25, NonDsaNoIndexerBytes) {
    auto cfg = kimi_k25_full_config();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    LayerRegistry reg{model_cfg, cfg, nvfp4};

    // K2.5 has index_topk=0 (no DSA). Attention bytes should be MLA-only.
    // K2.5: q_a=7168*1536=11010048, q_b=1536*64*(128+64)=18874368,
    // kv_a=7168*(512+64)=4128768, kv_b=512*64*(128+128)=8388608,
    // o=64*128*7168=58720256, norms=1536+512+7168*2=16384.
    // Total MLA params = 101138432 * 1.0 (FP8) = 101138432.
    // No indexer addition.
    EXPECT_EQ(reg.layer(0).attention_bytes, 101'138'432);
}

// ── Test: Custom gating pinning as list ─────────────────────────────────────

TEST(LayerRegistryCustomPinning, GatingAsList) {
    auto cfg = deepseek_v3_2_full_config();
    cfg.memory.pinned_layers.gating = std::vector<int>{3, 4, 5};
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    LayerRegistry reg{model_cfg, cfg, nvfp4};

    EXPECT_TRUE(reg.layer(3).gating_pinned);
    EXPECT_TRUE(reg.layer(4).gating_pinned);
    EXPECT_TRUE(reg.layer(5).gating_pinned);
    EXPECT_FALSE(reg.layer(6).gating_pinned);
    EXPECT_FALSE(reg.layer(60).gating_pinned);
}

// ═══════════════════════════════════════════════════════════════════════════
// DeepSeek-V4 (V4-3a): per-layer anatomy from compress_ratios
// ═══════════════════════════════════════════════════════════════════════════

#include "model/pinned_upload_plan.h"
#include "model/quantization/gguf_kquant.h"

namespace {

// Mirrors test-data/config/deepseek_v4_flash_gguf.json (ticket A surface).
// compress_ratios: [0,0] then alternating 4/128 ending at 4 → 2 SWA, 21 CSA,
// 20 HCA.
lc::Config deepseek_v4_full_config() {
    std::vector<int> ratios{0, 0};
    for (int l = 2; l < 43; ++l) ratios.push_back(l % 2 == 0 ? 4 : 128);
    auto j = nlohmann::json{
        {"model", {
            {"architecture",           "deepseek_v4"},
            {"weights_path",           "/data/models/deepseek-v4-flash.gguf"},
            {"weights_format",         "gguf"},
            {"num_hidden_layers",      43},
            {"hidden_size",            4096},
            {"num_attention_heads",    64},
            {"num_key_value_heads",    1},
            {"head_dim",               512},
            {"qk_rope_head_dim",       64},
            {"q_lora_rank",            1024},
            {"compress_ratios",        ratios},
            {"compress_rope_theta",    160000.0},
            {"sliding_window",         128},
            {"intermediate_size",      2048},
            {"n_routed_experts",       256},
            {"n_shared_experts",       1},
            {"num_experts_per_tok",    6},
            {"n_group",                1},
            {"topk_group",             1},
            {"vocab_size",             129280},
            {"max_position_embeddings", 1048576},
            {"rope_theta",             10000.0},
            {"rms_norm_eps",           1e-6},
            {"num_nextn_predict_layers", 1},
            {"first_k_dense_replace",  0},
            {"routed_scaling_factor",  1.5},
            {"moe_intermediate_size",  2048},
            {"moe_layer_freq",         1},
            {"gating_score_fn",        "sqrtsoftplus"},
            {"swiglu_limit",           10.0},
            {"num_hash_layers",        3},
            {"o_groups",               8},
            {"o_lora_rank",            1024},
            {"hc_mult",                4},
            {"hc_sinkhorn_iters",      20},
            {"hc_eps",                 1e-6},
            {"index_topk",             512},
            {"index_n_heads",          64},
            {"index_head_dim",         128},
        }},
        {"quantization", {{"weights", "gguf"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp16"}}},
        {"compute", {{"attention_backend", "csa_hca"}}},
        {"parallelism", {{"tensor_parallelism", 1}}},
        {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                      {"tp_array", {0}},
                      {"system_ram_gb", 512}}},
    };
    return lc::parse_config(j);
}

// Independent byte math from the GGUF census shapes (dossier §0.3), BF16=2 /
// F32=4 native, rounded to 16.
constexpr int64_t kV4Round16(int64_t b) { return (b + 15) & ~int64_t{15}; }

int64_t v4_expected_common_bytes() {
    int64_t b = 0;
    b += 4096LL * 1024 * 2;             // q_a
    b += 1024LL * 4;                    // q_a_norm
    b += 4096LL * 512 * 2;              // attn_kv
    b += 512LL * 4;                     // kv_a_norm
    b += 64LL * 4;                      // attn_sinks
    b += 1024LL * 64 * 512 * 2;         // q_b
    b += 64LL * 512 * 1024 * 2;         // o_a  [4096, 8192]
    b += 8LL * 1024 * 4096 * 2;         // o_b  [8192, 4096]
    const int64_t hc_set = 4LL * 4096 * 24 + 24 + 3;
    b += 2 * hc_set * 4;                // hc_attn_* + hc_ffn_* (F32)
    return b;
}

int64_t v4_expected_csa_extra() {
    int64_t b = 0;
    b += 2 * 4096LL * 1024 * 2;         // compressor wkv + wgate [4096, 1024]
    b += 1024LL * 4 * 4;                // APE [1024, 4] F32
    b += 512LL * 4;                     // compressor norm
    b += 4096LL * 64 * 2;               // indexer proj
    b += 1024LL * 8192 * 2;             // indexer q_b [1024, 8192]
    b += 2 * 4096LL * 256 * 2;          // idx-compressor wkv + wgate
    b += 256LL * 4 * 4;                 // idx-compressor APE [256, 4] F32
    b += 128LL * 4;                     // idx-compressor norm
    return b;
}

int64_t v4_expected_hca_extra() {
    int64_t b = 0;
    b += 2 * 4096LL * 512 * 2;          // compressor wkv + wgate [4096, 512]
    b += 512LL * 128 * 4;               // APE [512, 128] F32
    b += 512LL * 4;                     // compressor norm
    return b;
}

}  // namespace

TEST(LayerRegistryV4, LayerCounts) {
    auto cfg = deepseek_v4_full_config();
    ModelConfig model_cfg{cfg};
    layerstorm::model::GgufQuantInterface mxfp4{
        layerstorm::model::GgufKQuantType::MXFP4};
    LayerRegistry reg{model_cfg, cfg, mxfp4};

    EXPECT_EQ(reg.num_layers(), 43);
    EXPECT_EQ(reg.num_moe_layers(), 43);   // all-MoE (first_k_dense_replace=0)
    EXPECT_EQ(reg.num_dense_layers(), 0);
    EXPECT_EQ(reg.total_routed_experts(), 256);
}

TEST(LayerRegistryV4, PerLayerAttentionSizesByType) {
    auto cfg = deepseek_v4_full_config();
    ModelConfig model_cfg{cfg};
    layerstorm::model::GgufQuantInterface mxfp4{
        layerstorm::model::GgufKQuantType::MXFP4};
    LayerRegistry reg{model_cfg, cfg, mxfp4};

    const int64_t swa = kV4Round16(v4_expected_common_bytes());
    const int64_t csa = kV4Round16(v4_expected_common_bytes() +
                                   v4_expected_csa_extra());
    const int64_t hca = kV4Round16(v4_expected_common_bytes() +
                                   v4_expected_hca_extra());

    // Layers 0, 1 = SWA; even 2..42 = CSA; odd 3..41 = HCA.
    EXPECT_EQ(reg.layer(0).attention_bytes, swa);
    EXPECT_EQ(reg.layer(1).attention_bytes, swa);
    EXPECT_EQ(reg.layer(2).attention_bytes, csa);
    EXPECT_EQ(reg.layer(3).attention_bytes, hca);
    EXPECT_EQ(reg.layer(42).attention_bytes, csa);

    // Ordering sanity: CSA (compressor + indexer) > HCA (heavy compressor) > SWA.
    EXPECT_GT(csa, hca);
    EXPECT_GT(hca, swa);
}

TEST(LayerRegistryV4, HashLayerGatingIncludesTid2eid) {
    auto cfg = deepseek_v4_full_config();
    ModelConfig model_cfg{cfg};
    layerstorm::model::GgufQuantInterface mxfp4{
        layerstorm::model::GgufKQuantType::MXFP4};
    LayerRegistry reg{model_cfg, cfg, mxfp4};

    // gate weight: hidden × n_routed at fp16 gating = 4096*256*2
    const int64_t gate = 4096LL * 256 * 2;
    const int64_t tid2eid = 6LL * 129280 * 4;  // [experts_per_tok, vocab] I32
    EXPECT_EQ(reg.layer(0).gating_bytes, gate + tid2eid);
    EXPECT_EQ(reg.layer(2).gating_bytes, gate + tid2eid);
    EXPECT_EQ(reg.layer(3).gating_bytes, gate);   // non-hash
    EXPECT_EQ(reg.layer(42).gating_bytes, gate);
}

TEST(LayerRegistryV4, UploadPlanPerLayerSlotsAndModelLevel) {
    auto cfg = deepseek_v4_full_config();
    ModelConfig model_cfg{cfg};
    layerstorm::model::GgufQuantInterface mxfp4{
        layerstorm::model::GgufKQuantType::MXFP4};
    auto plan = layerstorm::model::build_upload_plan(model_cfg, cfg, mxfp4,
                                                     /*tp=*/1, /*rank=*/0);

    // Attention slots vary per layer type and match the registry sizes.
    LayerRegistry reg{model_cfg, cfg, mxfp4};
    for (int l : {0, 1, 2, 3, 41, 42}) {
        auto* slot = plan.find(layerstorm::model::PinnedComponent::attention, l);
        ASSERT_NE(slot, nullptr) << "layer " << l;
        EXPECT_EQ(slot->size_bytes, reg.layer(l).attention_bytes) << "layer " << l;
    }

    // Hash layers 0..2 carry gating_hash_table and NO gating_bias.
    for (int l : {0, 1, 2}) {
        EXPECT_NE(plan.find(
            layerstorm::model::PinnedComponent::gating_hash_table, l), nullptr);
        EXPECT_EQ(plan.find(
            layerstorm::model::PinnedComponent::gating_bias, l), nullptr);
    }
    EXPECT_EQ(plan.find(
        layerstorm::model::PinnedComponent::gating_hash_table, 3), nullptr);
    EXPECT_NE(plan.find(
        layerstorm::model::PinnedComponent::gating_bias, 3), nullptr);

    // Model-level output_hc present: (4·4096·4 + 4 + 1) F32.
    auto* hc = plan.find(layerstorm::model::PinnedComponent::output_hc, -1);
    ASSERT_NE(hc, nullptr);
    EXPECT_EQ(hc->size_bytes, (4LL * 4096 * 4 + 4 + 1) * 4);

    // NO MTP slots for V4 despite num_nextn_predict_layers=1 (GGUF has no
    // nextn tensors; embedded draft = dspark ticket J).
    EXPECT_EQ(plan.find(layerstorm::model::PinnedComponent::attention, 43), nullptr);
    EXPECT_EQ(plan.find(
        layerstorm::model::PinnedComponent::mtp_embed_tokens, 43), nullptr);
}

TEST(LayerRegistryV4, SharedExpertSizedAtBf16NotQ8UpperBound) {
    auto cfg = deepseek_v4_full_config();
    ModelConfig model_cfg{cfg};
    layerstorm::model::GgufQuantInterface mxfp4{
        layerstorm::model::GgufKQuantType::MXFP4};
    auto plan = layerstorm::model::build_upload_plan(model_cfg, cfg, mxfp4, 1, 0);

    // V4 shared experts are BF16-native: gate slot = inter × hidden × 2.
    int64_t se = 0;
    for (auto comp : {layerstorm::model::PinnedComponent::shared_expert_gate,
                      layerstorm::model::PinnedComponent::shared_expert_up,
                      layerstorm::model::PinnedComponent::shared_expert_down}) {
        auto* slot = plan.find(comp, 0);
        ASSERT_NE(slot, nullptr);
        se += slot->size_bytes;
    }
    EXPECT_EQ(se, 3LL * 2048 * 4096 * 2);
}

TEST(LayerRegistryV4, EstimateGpuBudgetsPinnedFitsVram) {
    auto cfg = deepseek_v4_full_config();
    ModelConfig model_cfg{cfg};
    layerstorm::model::GgufQuantInterface mxfp4{
        layerstorm::model::GgufKQuantType::MXFP4};
    LayerRegistry reg{model_cfg, cfg, mxfp4};

    auto budgets = reg.estimate_gpu_budgets();
    ASSERT_EQ(budgets.size(), 1u);
    // Pinned = embed + head + 43 variable attention layers + norms + gating
    // (incl. 3 hash tables) + shared experts + output_hc.  Sanity band:
    // attention alone is ~9.9 GB; the whole pinned region must fit a 5090.
    EXPECT_GT(budgets[0].pinned_bytes, 10LL << 30);
    EXPECT_LT(budgets[0].pinned_bytes, 20LL << 30);
    EXPECT_GT(budgets[0].available_for_cache_bytes, 0);
}
