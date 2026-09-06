#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <set>
#include <vector>

#include "model/pinned_region_layout.h"
#include "model/pinned_upload_plan.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/fp8.h"
#include "model/quantization/nvfp4.h"
#include "model/quantization/gguf_kquant.h"
#include "model/weight_loader/tp_weight_sharder.h"
#include "model/weight_loader/weight_loader.h"
#include "model/weight_loader/weight_handler.h"

namespace lc = layerstorm::config;
using layerstorm::model::compute_pinned_layout;
using layerstorm::model::has_kv_b_in_checkpoint;
using layerstorm::model::has_output_head_bias;
using layerstorm::model::ModelConfig;
using layerstorm::model::PinnedRegionLayout;
using layerstorm::model::PinnedUploadPlan;
using layerstorm::model::PinnedComponent;
using layerstorm::model::build_upload_plan;

// ── Config helpers ──────────────────────────────────────────────────────────

static lc::Config v32_config_tp2() {
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
        {"hardware", {{"gpus", {
            {{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}},
            {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 32}},
            {{"id", 2}, {"type", "rtx5080"}, {"vram_gb", 16}},
            {{"id", 3}, {"type", "rtx5080"}, {"vram_gb", 16}},
        }}, {"system_ram_gb", 256}}},
        {"parallelism", {{"tensor_parallelism", 2}}},
    };
    return lc::parse_config(j);
}

static lc::Config v32_config_tp1() {
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
        {"hardware", {{"gpus", {
            {{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}},
        }}, {"system_ram_gb", 256}}},
        {"parallelism", {{"tensor_parallelism", 1}}},
    };
    return lc::parse_config(j);
}

// ── Tests ───────────────────────────────────────────────────────────────────

TEST(PinnedRegionLayout, TotalBytesPositive_TP2) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;

    auto layout = compute_pinned_layout(model_cfg, cfg, nvfp4, 2, 0);

    EXPECT_GT(layout.total_bytes, 0);
    // KD-4f-d.1b: total includes 32 KB per-tensor alignment budget.
    using layerstorm::model::kUploadAlignBudget;
    EXPECT_EQ(layout.total_bytes,
              layout.embedding_bytes + layout.output_head_bytes +
              layout.attention_bytes + layout.layer_norm_bytes +
              layout.gating_bytes + layout.shared_expert_bytes +
              layout.dense_ffn_bytes + layout.final_norm_bytes +
              layout.mtp_bytes + kUploadAlignBudget);
}

TEST(PinnedRegionLayout, TotalBytesPositive_TP1) {
    auto cfg = v32_config_tp1();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;

    auto layout = compute_pinned_layout(model_cfg, cfg, nvfp4, 1, 0);

    EXPECT_GT(layout.total_bytes, 0);
    // KD-4f-d.1b: total includes 32 KB per-tensor alignment budget.
    using layerstorm::model::kUploadAlignBudget;
    EXPECT_EQ(layout.total_bytes,
              layout.embedding_bytes + layout.output_head_bytes +
              layout.attention_bytes + layout.layer_norm_bytes +
              layout.gating_bytes + layout.shared_expert_bytes +
              layout.dense_ffn_bytes + layout.final_norm_bytes +
              layout.mtp_bytes + kUploadAlignBudget);
}

TEST(PinnedRegionLayout, ComponentsReasonable_TP2) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;

    auto layout = compute_pinned_layout(model_cfg, cfg, nvfp4, 2, 0);

    // Embedding: 129280 * 7168 * 2 / 2 = ~927 MB
    EXPECT_GT(layout.embedding_bytes, 900'000'000);
    EXPECT_LT(layout.embedding_bytes, 950'000'000);

    // Layer norms: 61 layers * 2 * 7168 * 2 (BF16 after TD-73c F32→BF16 conversion)
    int64_t expected_layer_norms = 61LL * 2 * 7168 * 2;
    EXPECT_EQ(layout.layer_norm_bytes, expected_layer_norms);

    // Final norm: 7168 * 2 = 14336 (BF16 after TD-73c)
    EXPECT_EQ(layout.final_norm_bytes, 14336);

    // MTP bytes > 0 (has num_nextn_predict_layers = 1)
    EXPECT_GT(layout.mtp_bytes, 0);

    // Attention dominates the layout
    EXPECT_GT(layout.attention_bytes, layout.gating_bytes);
    EXPECT_GT(layout.attention_bytes, layout.shared_expert_bytes);
}

TEST(PinnedRegionLayout, TpShardingReducesSize) {
    auto cfg_tp1 = v32_config_tp1();
    auto cfg_tp2 = v32_config_tp2();
    ModelConfig mcfg1{cfg_tp1};
    ModelConfig mcfg2{cfg_tp2};
    layerstorm::model::Nvfp4 nvfp4;

    auto layout_tp1 = compute_pinned_layout(mcfg1, cfg_tp1, nvfp4, 1, 0);
    auto layout_tp2 = compute_pinned_layout(mcfg2, cfg_tp2, nvfp4, 2, 0);

    // TP=2 should produce smaller per-GPU total than TP=1
    EXPECT_LT(layout_tp2.total_bytes, layout_tp1.total_bytes);

    // Embedding halved
    EXPECT_EQ(layout_tp2.embedding_bytes, layout_tp1.embedding_bytes / 2);

    // Shared expert: shardable portion halved, replicated scalars unchanged
    // TP1: (bytes_per_proj - 8)/1 + 8 = bytes_per_proj. TP2: (bytes_per_proj - 8)/2 + 8.
    // So TP2 < TP1 but not exactly half (due to replicated scalar overhead).
    EXPECT_LT(layout_tp2.shared_expert_bytes, layout_tp1.shared_expert_bytes);
    EXPECT_GT(layout_tp2.shared_expert_bytes, layout_tp1.shared_expert_bytes / 2);

    // Layer norms same (replicated)
    EXPECT_EQ(layout_tp2.layer_norm_bytes, layout_tp1.layer_norm_bytes);

    // Final norm same (replicated)
    EXPECT_EQ(layout_tp2.final_norm_bytes, layout_tp1.final_norm_bytes);
}

TEST(PinnedRegionLayout, GatingReplicated) {
    auto cfg_tp1 = v32_config_tp1();
    auto cfg_tp2 = v32_config_tp2();
    ModelConfig mcfg1{cfg_tp1};
    ModelConfig mcfg2{cfg_tp2};
    layerstorm::model::Nvfp4 nvfp4;

    auto layout_tp1 = compute_pinned_layout(mcfg1, cfg_tp1, nvfp4, 1, 0);
    auto layout_tp2 = compute_pinned_layout(mcfg2, cfg_tp2, nvfp4, 2, 0);

    // Gating is replicated: same size regardless of TP
    EXPECT_EQ(layout_tp1.gating_bytes, layout_tp2.gating_bytes);
}

TEST(PinnedRegionLayout, KvBIncludedForMLA) {
    auto cfg = v32_config_tp1();
    // V3.2 (kv_lora_rank=512) ships kv_b_proj in checkpoint
    EXPECT_TRUE(has_kv_b_in_checkpoint(cfg));

    // Non-MLA model (kv_lora_rank=0) has no kv_b_proj
    cfg.model.kv_lora_rank = 0;
    EXPECT_FALSE(has_kv_b_in_checkpoint(cfg));
}

TEST(PinnedRegionLayout, MatchesLayerRegistryBudget) {
    auto cfg = v32_config_tp2();
    cfg.hardware.tp_array = {0, 1};
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;

    layerstorm::model::LayerRegistry reg{model_cfg, cfg, nvfp4};

    auto layout = compute_pinned_layout(model_cfg, cfg, nvfp4, 2, 0);
    auto budgets = reg.estimate_gpu_budgets();

    // TP GPU budget must equal pinned layout total_bytes
    EXPECT_EQ(budgets[0].pinned_bytes, layout.total_bytes);
    EXPECT_EQ(budgets[1].pinned_bytes, layout.total_bytes);

    // Non-TP GPUs have zero pinned
    EXPECT_EQ(budgets[2].pinned_bytes, 0);
    EXPECT_EQ(budgets[3].pinned_bytes, 0);
}

// ── Plan-structural tests ──────────────────────────────────────────────────

namespace {

struct LayoutSums {
    int64_t embedding = 0, output_head = 0, attention = 0, layer_norm = 0;
    int64_t gating = 0, shared_expert = 0, dense_ffn = 0, final_norm = 0, mtp = 0;
};

LayoutSums sum_plan_slots(const PinnedUploadPlan& plan, int num_hidden) {
    LayoutSums s{};
    for (const auto& slot : plan.slots) {
        const bool is_mtp = (slot.layer_idx >= num_hidden);
        switch (slot.component) {
            case PinnedComponent::embedding:
                s.embedding += slot.size_bytes; break;
            case PinnedComponent::output_head_weight:
            case PinnedComponent::output_head_bias:
                s.output_head += slot.size_bytes; break;
            case PinnedComponent::attention:
                (is_mtp ? s.mtp : s.attention) += slot.size_bytes; break;
            case PinnedComponent::layer_norm:
                (is_mtp ? s.mtp : s.layer_norm) += slot.size_bytes; break;
            case PinnedComponent::gating_weight:
            case PinnedComponent::gating_bias:
                (is_mtp ? s.mtp : s.gating) += slot.size_bytes; break;
            case PinnedComponent::shared_expert_gate:
            case PinnedComponent::shared_expert_up:
            case PinnedComponent::shared_expert_gate_scales:
            case PinnedComponent::shared_expert_up_scales:
            case PinnedComponent::shared_expert_gate_scalar:
            case PinnedComponent::shared_expert_up_scalar:
            case PinnedComponent::shared_expert_down:
                (is_mtp ? s.mtp : s.shared_expert) += slot.size_bytes; break;
            case PinnedComponent::dense_ffn_gate:
            case PinnedComponent::dense_ffn_up:
            case PinnedComponent::dense_ffn_gate_scales:
            case PinnedComponent::dense_ffn_up_scales:
            case PinnedComponent::dense_ffn_gate_scalar:
            case PinnedComponent::dense_ffn_up_scalar:
            case PinnedComponent::dense_ffn_down:
                s.dense_ffn += slot.size_bytes; break;
            case PinnedComponent::final_norm:
                s.final_norm += slot.size_bytes; break;
            case PinnedComponent::mtp_embed_tokens:
            case PinnedComponent::mtp_eh_proj:
            case PinnedComponent::mtp_enorm:
            case PinnedComponent::mtp_hnorm:
            case PinnedComponent::mtp_shared_head_weight:
            case PinnedComponent::mtp_shared_head_norm:
                s.mtp += slot.size_bytes; break;
        }
    }
    return s;
}

}  // namespace

TEST(PinnedUploadPlan, PlanSlotSumsMatchLayout_NVFP4_TP2) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;

    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);
    auto layout = compute_pinned_layout(model_cfg, cfg, nvfp4, 2, 0);
    auto s = sum_plan_slots(plan, cfg.model.num_hidden_layers);

    EXPECT_EQ(s.embedding, layout.embedding_bytes);
    EXPECT_EQ(s.output_head, layout.output_head_bytes);
    EXPECT_EQ(s.attention, layout.attention_bytes);
    EXPECT_EQ(s.layer_norm, layout.layer_norm_bytes);
    EXPECT_EQ(s.gating, layout.gating_bytes);
    EXPECT_EQ(s.shared_expert, layout.shared_expert_bytes);
    EXPECT_EQ(s.dense_ffn, layout.dense_ffn_bytes);
    EXPECT_EQ(s.final_norm, layout.final_norm_bytes);
    EXPECT_EQ(s.mtp, layout.mtp_bytes);
    // KD-4f-d.1b: layout includes 32 KB per-tensor alignment budget beyond plan.
    using layerstorm::model::kUploadAlignBudget;
    EXPECT_EQ(plan.total_bytes + kUploadAlignBudget, layout.total_bytes);
}

// ── FP8 config + tests ────────────────────────────────────────────────────

static lc::Config v32_config_fp8_tp2() {
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
        {"quantization", {{"weights", "fp8_e4m3"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {{"gpus", {
            {{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}},
            {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 32}},
            {{"id", 2}, {"type", "rtx5080"}, {"vram_gb", 16}},
            {{"id", 3}, {"type", "rtx5080"}, {"vram_gb", 16}},
        }}, {"system_ram_gb", 256}}},
        {"parallelism", {{"tensor_parallelism", 2}}},
    };
    return lc::parse_config(j);
}

TEST(PinnedUploadPlan, PlanSlotSumsMatchLayout_FP8_TP2) {
    auto cfg = v32_config_fp8_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Fp8E4M3 fp8;

    auto plan = build_upload_plan(model_cfg, cfg, fp8, 2, 0);
    auto layout = compute_pinned_layout(model_cfg, cfg, fp8, 2, 0);
    auto s = sum_plan_slots(plan, cfg.model.num_hidden_layers);

    EXPECT_EQ(s.embedding, layout.embedding_bytes);
    EXPECT_EQ(s.output_head, layout.output_head_bytes);
    EXPECT_EQ(s.attention, layout.attention_bytes);
    EXPECT_EQ(s.layer_norm, layout.layer_norm_bytes);
    EXPECT_EQ(s.gating, layout.gating_bytes);
    EXPECT_EQ(s.shared_expert, layout.shared_expert_bytes);
    EXPECT_EQ(s.dense_ffn, layout.dense_ffn_bytes);
    EXPECT_EQ(s.final_norm, layout.final_norm_bytes);
    EXPECT_EQ(s.mtp, layout.mtp_bytes);
    // KD-4f-d.1b: layout includes 32 KB per-tensor alignment budget beyond plan.
    using layerstorm::model::kUploadAlignBudget;
    EXPECT_EQ(plan.total_bytes + kUploadAlignBudget, layout.total_bytes);
}

TEST(PinnedUploadPlan, FP8_TotalMatchesLayout) {
    auto cfg = v32_config_fp8_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Fp8E4M3 fp8;

    auto layout = compute_pinned_layout(model_cfg, cfg, fp8, 2, 0);
    auto plan = build_upload_plan(model_cfg, cfg, fp8, 2, 0);

    // KD-4f-d.1b: layout includes 32 KB per-tensor alignment budget beyond plan.
    using layerstorm::model::kUploadAlignBudget;
    EXPECT_EQ(plan.total_bytes + kUploadAlignBudget, layout.total_bytes);
    EXPECT_GT(plan.total_bytes, 0);
}

TEST(PinnedUploadPlan, FP8_SlotsContiguous) {
    auto cfg = v32_config_fp8_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Fp8E4M3 fp8;
    auto plan = build_upload_plan(model_cfg, cfg, fp8, 2, 0);

    ASSERT_FALSE(plan.slots.empty());
    EXPECT_EQ(plan.slots[0].offset, 0);

    for (size_t i = 1; i < plan.slots.size(); ++i) {
        EXPECT_EQ(plan.slots[i].offset,
                  plan.slots[i - 1].offset + plan.slots[i - 1].size_bytes)
            << "Gap between slot " << (i - 1) << " and " << i;
    }

    const auto& last = plan.slots.back();
    EXPECT_EQ(plan.total_bytes, last.offset + last.size_bytes);
}

TEST(PinnedUploadPlan, FP8_NoSubComponentSlots) {
    auto cfg = v32_config_fp8_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Fp8E4M3 fp8;
    auto plan = build_upload_plan(model_cfg, cfg, fp8, 2, 0);

    for (int l = 0; l < cfg.model.num_hidden_layers; ++l) {
        EXPECT_EQ(plan.find(PinnedComponent::shared_expert_gate_scales, l), nullptr)
            << "FP8 should have no gate_scales slot at layer " << l;
        EXPECT_EQ(plan.find(PinnedComponent::shared_expert_up_scales, l), nullptr)
            << "FP8 should have no up_scales slot at layer " << l;
        EXPECT_EQ(plan.find(PinnedComponent::shared_expert_gate_scalar, l), nullptr)
            << "FP8 should have no gate_scalar slot at layer " << l;
        EXPECT_EQ(plan.find(PinnedComponent::shared_expert_up_scalar, l), nullptr)
            << "FP8 should have no up_scalar slot at layer " << l;
    }
}

// ── Existing PinnedUploadPlan tests ───────────────────────────────────────

TEST(PinnedUploadPlan, TotalMatchesLayout_TP2) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;

    auto layout = compute_pinned_layout(model_cfg, cfg, nvfp4, 2, 0);
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    // KD-4f-d.1b: layout includes 32 KB per-tensor alignment budget beyond plan.
    constexpr int64_t kBudget = 32 * 1024;
    EXPECT_EQ(plan.total_bytes + kBudget, layout.total_bytes)
        << "plan.total_bytes (" << plan.total_bytes
        << ") + budget != layout.total_bytes (" << layout.total_bytes << ")";
}

TEST(PinnedUploadPlan, TotalMatchesLayout_TP1) {
    auto cfg = v32_config_tp1();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;

    auto layout = compute_pinned_layout(model_cfg, cfg, nvfp4, 1, 0);
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 1, 0);

    // KD-4f-d.1b: layout includes 32 KB per-tensor alignment budget beyond plan.
    constexpr int64_t kBudget = 32 * 1024;
    EXPECT_EQ(plan.total_bytes + kBudget, layout.total_bytes)
        << "plan.total_bytes (" << plan.total_bytes
        << ") + budget != layout.total_bytes (" << layout.total_bytes << ")";
}

TEST(PinnedUploadPlan, SlotsContiguous) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    ASSERT_FALSE(plan.slots.empty());
    EXPECT_EQ(plan.slots[0].offset, 0);

    for (size_t i = 1; i < plan.slots.size(); ++i) {
        EXPECT_EQ(plan.slots[i].offset,
                  plan.slots[i - 1].offset + plan.slots[i - 1].size_bytes)
            << "Gap between slot " << (i - 1) << " and " << i;
    }

    const auto& last = plan.slots.back();
    EXPECT_EQ(plan.total_bytes, last.offset + last.size_bytes);
}

TEST(PinnedUploadPlan, ContiguityGroupsAdjacent) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    std::set<int> seen_groups;
    for (const auto& s : plan.slots) {
        if (s.contiguity_group != 0)
            seen_groups.insert(s.contiguity_group);
    }
    EXPECT_FALSE(seen_groups.empty()) << "Expected contiguity groups for MoE shared experts";

    for (int g : seen_groups) {
        int first = -1, last = -1;
        int count = 0;
        for (int i = 0; i < static_cast<int>(plan.slots.size()); ++i) {
            if (plan.slots[i].contiguity_group == g) {
                if (first < 0) first = i;
                last = i;
                ++count;
            }
        }
        EXPECT_EQ(last - first + 1, count)
            << "Contiguity group " << g << " has non-adjacent slots";
    }
}

TEST(PinnedUploadPlan, AllSlotsPositiveSize) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    for (size_t i = 0; i < plan.slots.size(); ++i) {
        EXPECT_GT(plan.slots[i].size_bytes, 0)
            << "Slot " << i << " has zero size";
    }
}

TEST(PinnedUploadPlan, LookupHelpers) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    auto* embed = plan.find(PinnedComponent::embedding, -1);
    ASSERT_NE(embed, nullptr);
    EXPECT_EQ(embed->offset, 0);
    EXPECT_GT(embed->size_bytes, 0);

    auto* fn = plan.find(PinnedComponent::final_norm, -1);
    ASSERT_NE(fn, nullptr);
    EXPECT_GT(fn->size_bytes, 0);

    auto* attn0 = plan.find(PinnedComponent::attention, 0);
    ASSERT_NE(attn0, nullptr);
    EXPECT_GT(attn0->size_bytes, 0);

    // Layer 0 is dense (first_k_dense_replace=3) — no gating
    EXPECT_EQ(plan.find(PinnedComponent::gating_weight, 0), nullptr);

    // Layer 3 is first MoE layer — has gating
    EXPECT_NE(plan.find(PinnedComponent::gating_weight, 3), nullptr);

    // slots_for_layer for a regular layer
    auto layer0_slots = plan.slots_for_layer(0);
    EXPECT_FALSE(layer0_slots.empty());

    // slots_for_layer(-1) returns empty (model-level slots are non-contiguous)
    auto model_slots = plan.slots_for_layer(-1);
    EXPECT_TRUE(model_slots.empty());

    // slots_for_layer for a MoE layer
    auto layer3_slots = plan.slots_for_layer(3);
    EXPECT_GE(layer3_slots.size(), 11u);  // attn, norm, gating_w, gating_b + 7 SE sub-slots (NVFP4)
}

TEST(PinnedUploadPlan, MtpIncludesIndexerWhenDsa) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    int num_hidden = cfg.model.num_hidden_layers;  // 61

    // MTP layer at index num_hidden_layers
    auto* mtp_attn = plan.find(PinnedComponent::attention, num_hidden);
    ASSERT_NE(mtp_attn, nullptr);

    // MTP attention is LARGER than regular because MTP o_proj is BF16
    // (force_bf16_oproj=true per INV-4f-6) while regular o_proj is NVFP4.
    // Both include DSA indexer.
    auto* reg_attn = plan.find(PinnedComponent::attention, 0);
    ASSERT_NE(reg_attn, nullptr);
    EXPECT_GT(mtp_attn->size_bytes, reg_attn->size_bytes);
}

TEST(PinnedUploadPlan, ScaleContiguity_TD55c) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    // Layer 3 is first MoE layer for V3.2 (first_k_dense_replace=3)
    auto* gate_scales = plan.find(PinnedComponent::shared_expert_gate_scales, 3);
    auto* up_scales = plan.find(PinnedComponent::shared_expert_up_scales, 3);
    ASSERT_NE(gate_scales, nullptr);
    ASSERT_NE(up_scales, nullptr);

    EXPECT_EQ(gate_scales->offset + gate_scales->size_bytes, up_scales->offset)
        << "TD-55c: gate UE8M0 must be immediately followed by up UE8M0";

    // Weight contiguity
    auto* gate_w = plan.find(PinnedComponent::shared_expert_gate, 3);
    auto* up_w = plan.find(PinnedComponent::shared_expert_up, 3);
    ASSERT_NE(gate_w, nullptr);
    ASSERT_NE(up_w, nullptr);

    EXPECT_EQ(gate_w->offset + gate_w->size_bytes, up_w->offset)
        << "gate weight must be immediately followed by up weight";

    // Scalar slots exist and have correct size (8 bytes each for n_shared=1)
    auto* gate_scalar = plan.find(PinnedComponent::shared_expert_gate_scalar, 3);
    auto* up_scalar = plan.find(PinnedComponent::shared_expert_up_scalar, 3);
    ASSERT_NE(gate_scalar, nullptr);
    ASSERT_NE(up_scalar, nullptr);
    EXPECT_EQ(gate_scalar->size_bytes, 8);
    EXPECT_EQ(up_scalar->size_bytes, 8);
}

// ── GG-4 / TD-GGUF-ATTN-UPLOAD-SIZING: GGUF attention/dense sizing ──────────

namespace {
using layerstorm::model::GgufKQuantType;
using layerstorm::model::AttentionDims;
using layerstorm::model::compute_attn_dims;
using layerstorm::model::attention_layer_bytes;
namespace gguf = layerstorm::model::gguf;
}  // namespace

// bytes_per_element(WeightQuant) now covers all six GGUF variants (was missing
// q2_k/q3_k/q8_0) and matches the single-source block table.
TEST(GgufAttentionSizing, BytesPerElementCoversAllVariants) {
    using WQ = lc::WeightQuant;
    EXPECT_DOUBLE_EQ(layerstorm::model::bytes_per_element(WQ::gguf_q2_k),
                     gguf::bytes_per_element(GgufKQuantType::Q2_K));  // 0.328125
    EXPECT_DOUBLE_EQ(layerstorm::model::bytes_per_element(WQ::gguf_q3_k),
                     gguf::bytes_per_element(GgufKQuantType::Q3_K));  // 0.4296875
    EXPECT_DOUBLE_EQ(layerstorm::model::bytes_per_element(WQ::gguf_q4_k),
                     gguf::bytes_per_element(GgufKQuantType::Q4_K));  // 0.5625
    EXPECT_DOUBLE_EQ(layerstorm::model::bytes_per_element(WQ::gguf_q5_k),
                     gguf::bytes_per_element(GgufKQuantType::Q5_K));  // 0.6875
    EXPECT_DOUBLE_EQ(layerstorm::model::bytes_per_element(WQ::gguf_q6_k),
                     gguf::bytes_per_element(GgufKQuantType::Q6_K));  // 0.8203125
    EXPECT_DOUBLE_EQ(layerstorm::model::bytes_per_element(WQ::gguf_q8_0),
                     gguf::bytes_per_element(GgufKQuantType::Q8_0));  // 1.0625
    // Q6_K must NOT be the old buggy 0.8125 (it is 210/256 = 0.8203125).
    EXPECT_NE(layerstorm::model::bytes_per_element(WQ::gguf_q6_k), 0.8125);
    // Generic gguf has no scalar bytes/element — must throw.
    EXPECT_THROW(layerstorm::model::bytes_per_element(WQ::gguf), std::runtime_error);
}

// attention_layer_bytes for a uniform GGUF variant equals the sum of exact
// per-projection gguf_packed_bytes (the source of truth), not a scalar estimate.
TEST(GgufAttentionSizing, AttentionLayerBytesUsesPackedBytes) {
    auto cfg = v32_config_tp2();   // V3.2 dims (hidden 7168, q_lora 1536, …)
    ModelConfig model_cfg{cfg};
    AttentionDims d = compute_attn_dims(model_cfg.raw());

    const int tp = 2;
    const auto t = GgufKQuantType::Q4_K;
    const int64_t got = attention_layer_bytes(
        d, lc::WeightQuant::gguf_q4_k, /*include_kv_b=*/true, tp,
        /*include_indexer=*/false);

    auto packed = [&](int64_t params, int64_t in) {
        return gguf::gguf_packed_bytes(params / in, in, t);
    };
    int64_t expect = packed(d.q_a_params, d.q_a_in)
                   + packed(d.kv_a_params, d.kv_a_in)
                   + packed(d.q_b_params, d.q_b_in) / tp
                   + packed(d.kv_b_params, d.kv_b_in) / tp
                   + packed(d.o_params, d.o_in) / tp;
    int64_t norms = (d.q_a_norm_params + d.kv_a_norm_params) * 2;
    expect = (expect + norms + 15) & ~int64_t{15};
    EXPECT_EQ(got, expect);
    EXPECT_GT(got, 0);
}

// Generic `gguf` attention sizing is a safe BF16 upper bound (no throw), since
// per-tensor types aren't visible at plan time and BF16 ≥ every GGUF type.
TEST(GgufAttentionSizing, GenericGgufIsBf16UpperBound) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    AttentionDims d = compute_attn_dims(model_cfg.raw());

    const int tp = 2;
    int64_t generic = attention_layer_bytes(
        d, lc::WeightQuant::gguf, true, tp, false);
    int64_t q4k = attention_layer_bytes(
        d, lc::WeightQuant::gguf_q4_k, true, tp, false);
    int64_t q8_0 = attention_layer_bytes(
        d, lc::WeightQuant::gguf_q8_0, true, tp, false);

    EXPECT_GT(generic, 0);
    // BF16 upper bound dominates every concrete k-quant / Q8_0 variant.
    EXPECT_GE(generic, q4k);
    EXPECT_GE(generic, q8_0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// glm5_next attention sizing (GF3.3) — spec/GLM-5.3-FLASH-MODELINFO.md
// §3a (KDA), §3b (NoPE sparse MLA), §3c (IndexPool indexer), §3e (mHC), §5 (MTP)
//
// Every expected number below is summed BY HAND from the verified per-tensor
// table (HF rev 04c4e9e9 shard headers) at the REAL GLM-5.3-Flash dimensions:
//   h = hidden = 4096, H = kda heads = 64, D = kda head_dim = 128, K = 4,
//   heads = 64, qk_head = 256, v_head_dim = 256, kv_lora = 512, q_lora = 1536,
//   index_n_heads = 32, index_head_dim = 128, index_kpool = 4, hc_mult = 4.
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

using layerstorm::model::glm5_next_attention_layer_bytes;
using layerstorm::model::TpWeightSharder;
using layerstorm::model::WeightBundle;
using layerstorm::model::RawTensor;
using layerstorm::model::TensorId;
using layerstorm::model::TensorComponent;
using layerstorm::model::TensorRole;
using layerstorm::model::TensorOwner;
using layerstorm::model::SafetensorsDtype;

constexpr int64_t glm_round16(int64_t b) { return (b + 15) & ~int64_t{15}; }

lc::Config glm53_flash_config(int tp) {
    nlohmann::json layer_types = nlohmann::json::array();
    for (int l = 0; l < 45; ++l)
        layer_types.push_back((l % 4 == 3) ? "deepseek_sparse_attention"
                                           : "linear_attention");
    nlohmann::json gpus = nlohmann::json::array();
    nlohmann::json tp_array = nlohmann::json::array();
    for (int i = 0; i < tp; ++i) {
        gpus.push_back({{"id", i}, {"type", "rtx5090"}, {"vram_gb", 32}});
        tp_array.push_back(i);
    }
    auto j = nlohmann::json{
        {"model", {
            {"architecture",           "glm5_next"},
            {"weights_path",           "/data/models/glm-5.3-flash/"},
            {"weights_format",         "safetensors"},
            {"num_hidden_layers",      45},
            {"hidden_size",            4096},
            {"num_attention_heads",    64},
            {"num_key_value_heads",    64},
            {"intermediate_size",      12288},
            {"n_routed_experts",       288},
            {"n_shared_experts",       1},
            {"num_experts_per_tok",    8},
            {"n_group",                1},
            {"topk_group",             1},
            {"vocab_size",             154880},
            {"max_position_embeddings", 1048576},
            {"kv_lora_rank",           512},
            {"q_lora_rank",            1536},
            {"qk_rope_head_dim",       0},
            {"qk_nope_head_dim",       256},
            {"v_head_dim",             256},
            {"first_k_dense_replace",  3},
            {"moe_layer_freq",         1},
            {"index_topk",             2048},
            {"index_n_heads",          32},
            {"index_head_dim",         128},
            {"index_kpool",            4},
            {"index_kpool_compress",   true},
            {"index_kpool_always_select_tail", true},
            {"mla_use_nope",           true},
            {"layer_types",            layer_types},
            {"linear_attn_config", {
                {"num_heads", 64}, {"head_dim", 128},
                {"short_conv_kernel_size", 4}, {"gate_lower_bound", -5.0}}},
            {"hc_mult",                4},
            {"hc_sinkhorn_iters",      20},
            {"hc_eps",                 1e-6},
            {"swiglu_limit",           10.0},
            {"num_nextn_predict_layers", 1},
            {"rms_norm_eps",           1e-5},
            {"routed_scaling_factor",  2.5},
            {"moe_intermediate_size",  2048},
        }},
        {"quantization", {{"weights", "fp8_e4m3"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {{"gpus", gpus}, {"system_ram_gb", 256},
                      {"tp_array", tp_array}}},
        {"parallelism", {{"tensor_parallelism", tp}}},
    };
    return lc::parse_config(j);
}

// mHC stream weights per HIDDEN layer (§3e): hc_mult 4 ⇒ hc_mix = (2+4)*4 = 24.
//   hc_attn_fn / hc_ffn_fn   [24, 4*4096 = 16384]  F32 on device (GF3.9:
//     the checkpoint ships BF16 (FP8) / Q8_0 (GGUF), but launch_mhc_pre's
//     contract is F32 — weight_loader widens at load, sizing follows)
//   hc_attn_base / hc_ffn_base [24]                F32
//   hc_attn_scale / hc_ffn_scale [3]               F32
int64_t glm_hc_bytes() {
    return 2 * (24LL * 16384 * 4 + 24LL * 4 + 3LL * 4);
}

// KDA linear-attention layer (§3a) — BF16 everywhere except A_log/dt_bias
// (F32); the whole layer is in the FP8 skip list so this never depends on wq.
int64_t glm_kda_bytes(int64_t t) {
    int64_t b = 0;
    b += 3 * (8192 / t) * 4096 * 2;   // q_proj/k_proj/v_proj [8192, 4096] BF16
    b += (64 / t) * 4096 * 2;         // b_proj [64, 4096] BF16
    b += 2 * 128LL * 4096 * 2;        // f_a_proj/g_a_proj [128, 4096] REPLICATED
    b += 2 * (8192 / t) * 128 * 2;    // f_b_proj/g_b_proj [8192, 128] BF16
    b += 3 * (8192 / t) * 4 * 2;      // q/k/v_conv1d [8192, 1, 4] BF16
    b += (64 / t) * 4;                // A_log [64] F32
    b += (8192 / t) * 4;              // dt_bias [8192] F32
    b += 128LL * 2;                   // o_norm [128] BF16 REPLICATED
    b += 4096LL * (8192 / t) * 2;     // o_proj [4096, 8192] BF16 (row-parallel)
    return b;
}

// NoPE sparse-MLA layer (§3b) + IndexPool indexer (§3c) at the FP8 checkpoint.
// FP8 = 1 B/elem; blockwise scale = F32 [ceil(N/128), ceil(K/128)].
int64_t glm_sparse_bytes(int64_t t) {
    int64_t b = 0;
    b += 1536LL * 4096;               // q_a_proj [1536, 4096] FP8 (replicated)
    b += 12LL * 32 * 4;               //   + weight_scale_inv [12, 32] F32
    b += 1536LL * 2;                  // q_a_layernorm [1536] BF16
    b += (16384 / t) * 1536LL;        // q_b_proj [16384, 1536] FP8 (col-parallel)
    b += (128 / t) * 12LL * 4;        //   + scale [128, 12] F32
    b += 512LL * 4096;                // kv_a_proj_with_mqa [512, 4096] FP8
    b += 4LL * 32 * 4;                //   + scale [4, 32] F32
    b += 512LL * 2;                   // kv_a_layernorm [512] BF16
    b += (32768 / t) * 512LL * 2;     // kv_b_proj [32768, 512] BF16, NO scale
    b += 4096LL * (16384 / t);        // o_proj [4096, 16384] FP8 (row-parallel)
    b += 32LL * (128 / t) * 4;        //   + scale [32, 128] F32
    // Indexer — ALL BF16, ALL replicated.
    b += 4096LL * 1536 * 2;           // wq_b [4096, 1536]
    b += 128LL * 4096 * 2;            // wk [128, 4096]
    b += 2 * 128LL * 2;               // k_norm.weight + k_norm.bias [128]
    b += 32LL * 4096 * 2;             // weights_proj [32, 4096]
    b += 128LL * 4096 * 2;            // index_kpool_compress_gate [128, 4096]
    b += 4LL * 128 * 2;               // index_kpool_compress_ape [4, 128]
    return b;
}

std::vector<std::byte> glm_buf(size_t n) { return std::vector<std::byte>(n); }

WeightBundle glm_bundle(const std::vector<std::byte>& buf,
                        std::vector<int64_t> shape, SafetensorsDtype dtype,
                        TensorComponent comp) {
    WeightBundle wb;
    wb.id = TensorId{comp, TensorRole::weight, TensorOwner::attention, 0, -1};
    wb.weight = RawTensor{std::span<const std::byte>(buf.data(), buf.size()),
                          dtype, std::move(shape), std::nullopt};
    return wb;
}

void glm_add_scale(WeightBundle& wb, const std::vector<std::byte>& buf,
                   std::vector<int64_t> shape) {
    wb.aux.emplace_back(TensorRole::weight_scale,
                        RawTensor{std::span<const std::byte>(buf.data(), buf.size()),
                                  SafetensorsDtype::F32, std::move(shape),
                                  std::nullopt});
}

}  // namespace

// ── Hand-derived byte counts ───────────────────────────────────────────────

TEST(Glm5NextAttentionSizing, HandDerivedTotalsMatchTable) {
    // Independent literals, summed by hand from the §3a/§3b/§3c/§3e tables.
    EXPECT_EQ(glm_hc_bytes(), 3'145'944LL);
    EXPECT_EQ(glm_kda_bytes(1), 275'481'088LL);
    EXPECT_EQ(glm_kda_bytes(2), 138'789'248LL);
    EXPECT_EQ(glm_sparse_bytes(1), 149'190'144LL);
    EXPECT_EQ(glm_sparse_bytes(2), 86'264'320LL);
}

TEST(Glm5NextAttentionSizing, KdaLayerBytes) {
    auto cfg = glm53_flash_config(1);
    const auto& m = cfg.model;
    const auto wq = cfg.quantization.weights;

    // tp = 1, hidden layer (mHC included).
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, /*linear=*/true,
                                              /*include_hc=*/true, 1),
              glm_round16(glm_kda_bytes(1) + glm_hc_bytes()));
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, true, true, 1),
              278'627'040LL);

    // Without mHC (used only by the MTP block, which is never KDA — sanity).
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, true, false, 1),
              glm_round16(glm_kda_bytes(1)));

    // The KDA layer is in the FP8 skip list: its size must NOT move with wq.
    auto cfg8 = cfg;
    cfg8.quantization.weights = lc::WeightQuant::fp8_e5m2;
    EXPECT_EQ(glm5_next_attention_layer_bytes(cfg8.model,
                                              lc::WeightQuant::fp8_e5m2,
                                              true, true, 1),
              278'627'040LL);
}

TEST(Glm5NextAttentionSizing, SparseLayerBytesFp8) {
    auto cfg = glm53_flash_config(1);
    const auto& m = cfg.model;
    const auto wq = cfg.quantization.weights;

    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, /*linear=*/false,
                                              /*include_hc=*/true, 1),
              glm_round16(glm_sparse_bytes(1) + glm_hc_bytes()));
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, true, 1),
              152'336'096LL);

    // Sparse layers are ~1.8x SMALLER than KDA layers: FP8 projections + the
    // 4x-cheaper index-pooled indexer vs an all-BF16 KDA block.
    EXPECT_LT(glm5_next_attention_layer_bytes(m, wq, false, true, 1),
              glm5_next_attention_layer_bytes(m, wq, true, true, 1));
}

TEST(Glm5NextAttentionSizing, MtpBlockHasNoMhc) {
    auto cfg = glm53_flash_config(1);
    const auto& m = cfg.model;
    const auto wq = cfg.quantization.weights;

    // MTP layer 45 = sparse-MLA anatomy WITHOUT the hc set (MODELINFO §5).
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, /*include_hc=*/false, 1),
              glm_round16(glm_sparse_bytes(1)));
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, false, 1),
              149'190'144LL);
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, false, 2),
              86'264'320LL);

    // Exactly one hc set separates a hidden sparse layer from the MTP block.
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, true, 1) -
                  glm5_next_attention_layer_bytes(m, wq, false, false, 1),
              glm_round16(glm_sparse_bytes(1) + glm_hc_bytes()) -
                  glm_round16(glm_sparse_bytes(1)));
}

TEST(Glm5NextAttentionSizing, Tp2SplitsOnlyTheShardedTensors) {
    auto cfg = glm53_flash_config(2);
    const auto& m = cfg.model;
    const auto wq = cfg.quantization.weights;

    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, true, true, 2),
              glm_round16(glm_kda_bytes(2) + glm_hc_bytes()));
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, true, true, 2),
              141'935'200LL);
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, true, 2),
              glm_round16(glm_sparse_bytes(2) + glm_hc_bytes()));
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, true, 2),
              89'410'272LL);

    // tp=4 still divides every sharded axis evenly (heads 64, HD 8192, ...).
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, true, true, 4),
              glm_round16(glm_kda_bytes(4) + glm_hc_bytes()));
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, true, 4),
              glm_round16(glm_sparse_bytes(4) + glm_hc_bytes()));

    // The replicated remainder never shrinks: halving tp=1 would be wrong.
    EXPECT_GT(glm5_next_attention_layer_bytes(m, wq, true, true, 2) * 2,
              glm5_next_attention_layer_bytes(m, wq, true, true, 1));
}

TEST(Glm5NextAttentionSizing, RejectsUnsupportedArtifactQuants) {
    auto cfg = glm53_flash_config(1);
    const auto& m = cfg.model;
    // NVFP4 has no glm5_next loader (no KDA requant story) — still a throw.
    // GGUF is served by the GF3.9 upper-bound arm below, so it must NOT throw.
    EXPECT_THROW(glm5_next_attention_layer_bytes(m, lc::WeightQuant::nvfp4,
                                                 true, true, 1),
                 std::runtime_error);
    EXPECT_THROW(glm5_next_attention_layer_bytes(m, lc::WeightQuant::nvfp4,
                                                 false, true, 1),
                 std::runtime_error);
    for (auto wq : {lc::WeightQuant::gguf, lc::WeightQuant::gguf_q4_k,
                    lc::WeightQuant::gguf_q8_0, lc::WeightQuant::gguf_q6_k,
                    lc::WeightQuant::gguf_mxfp4}) {
        EXPECT_NO_THROW(glm5_next_attention_layer_bytes(m, wq, true, true, 1));
        EXPECT_NO_THROW(glm5_next_attention_layer_bytes(m, wq, false, true, 1));
    }
}

// ── GF3.9: the GGUF upper-bound arm ────────────────────────────────────────
//
// LOCKED FP8 REGRESSION CONSTANTS (computed from the pre-GF3.9 function; the
// native-FP8 / GLM-5.2 / V4 paths must stay byte-identical forever).
constexpr int64_t kFp8KdaHiddenTp1   = 278'627'040LL;
constexpr int64_t kFp8KdaHiddenTp2   = 141'935'200LL;
constexpr int64_t kFp8SparseHiddenTp1 = 152'336'096LL;
constexpr int64_t kFp8SparseHiddenTp2 = 89'410'272LL;
constexpr int64_t kFp8SparseMtpTp1   = 149'190'144LL;
constexpr int64_t kFp8SparseMtpTp2   = 86'264'320LL;

TEST(Glm5NextAttentionSizing, Fp8NumbersAreByteIdenticalAfterGgufArm) {
    for (int tp : {1, 2}) {
        auto cfg = glm53_flash_config(tp);
        const auto& m = cfg.model;
        for (auto wq : {lc::WeightQuant::fp8_e4m3, lc::WeightQuant::fp8_e5m2}) {
            // KDA layers are dtype-fixed, so both FP8 flavours agree there.
            EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, true, true, tp),
                      tp == 1 ? kFp8KdaHiddenTp1 : kFp8KdaHiddenTp2);
        }
        const auto wq = lc::WeightQuant::fp8_e4m3;
        EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, true, tp),
                  tp == 1 ? kFp8SparseHiddenTp1 : kFp8SparseHiddenTp2);
        EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, false, tp),
                  tp == 1 ? kFp8SparseMtpTp1 : kFp8SparseMtpTp2);
        EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, true, false, tp),
                  glm_round16(glm_kda_bytes(tp)));
    }
}

namespace {

// GGUF KDA layer (SURVEY_BOOT §3.7 blk.0): every matrix packed Q8_0 → sized at
// the BF16 upper bound; the three convs, A_log, dt_bias and ssm_norm ship F32
// and are sized F32.
int64_t glm_kda_gguf_bytes(int64_t t) {
    int64_t b = 0;
    b += 3 * (8192 / t) * 4096 * 2;   // attn_q/attn_k/attn_v [8192,4096] Q8_0
    b += (64 / t) * 4096 * 2;         // ssm_beta [64,4096] Q8_0
    b += 2 * 128LL * 4096 * 2;        // ssm_f_a/ssm_g_a [128,4096] Q8_0 (repl.)
    b += 2 * (8192 / t) * 128 * 2;    // ssm_f_b/ssm_g_b [8192,128] Q8_0
    b += 3 * (8192 / t) * 4 * 4;      // ssm_conv1d_{q,k,v} [8192,1,4] F32
    b += (64 / t) * 4;                // ssm_a [64] F32
    b += (8192 / t) * 4;              // ssm_dt.bias [8192] F32
    b += 128LL * 4;                   // ssm_norm [128] F32
    b += 4096LL * (8192 / t) * 2;     // attn_output [4096,8192] Q8_0
    return b;
}

// GGUF sparse-MLA layer (§3.7 blk.3): matrices Q8_0 → BF16 upper bound;
// layernorms + indexer k_norm F32→BF16 (validate_plan halves them); indexer
// proj and the compressor APE stay F32.
int64_t glm_sparse_gguf_bytes(int64_t t) {
    int64_t b = 0;
    b += 1536LL * 4096 * 2;           // attn_q_a [1536,4096] Q8_0
    b += 1536LL * 2;                  // attn_q_a_norm [1536] F32→BF16
    b += (16384 / t) * 1536LL * 2;    // attn_q_b [16384,1536] Q8_0
    b += 512LL * 4096 * 2;            // attn_kv_a_mqa [512,4096] Q8_0
    b += 512LL * 2;                   // attn_kv_a_norm [512] F32→BF16
    b += (32768 / t) * 512LL * 2;     // attn_k_b + attn_v_b Q8_0
    b += 4096LL * (16384 / t) * 2;    // attn_output [4096,16384] Q8_0
    b += 4096LL * 1536 * 2;           // indexer.attn_q_b Q8_0
    b += 128LL * 4096 * 2;            // indexer.attn_k Q8_0
    b += 2 * 128LL * 2;               // indexer.k_norm weight+bias F32→BF16
    b += 32LL * 4096 * 4;             // indexer.proj [32,4096] F32
    b += 128LL * 4096 * 2;            // indexer_compressor_gate Q8_0→BF16
    b += 4LL * 128 * 4;               // indexer_compressor_ape [4,128] F32
    return b;
}

/// Packed Q8_0 bytes for an [out, in] matrix: out * (in/32) * 34.
int64_t q8_packed(int64_t out, int64_t in) { return out * (in / 32) * 34; }

}  // namespace

TEST(Glm5NextAttentionSizing, GgufHandDerivedTotals) {
    EXPECT_EQ(glm_kda_gguf_bytes(1), 275'677'952LL);
    EXPECT_EQ(glm_sparse_gguf_bytes(1), 250'092'032LL);

    auto cfg = glm53_flash_config(1);
    const auto& m = cfg.model;
    for (auto wq : {lc::WeightQuant::gguf, lc::WeightQuant::gguf_q4_k,
                    lc::WeightQuant::gguf_q8_0}) {
        // The upper bound is quant-independent (BF16 covers every GGUF type).
        EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, true, true, 1),
                  glm_round16(glm_kda_gguf_bytes(1) + glm_hc_bytes()));
        EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, true, true, 1),
                  278'823'904LL);
        EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, true, 1),
                  glm_round16(glm_sparse_gguf_bytes(1) + glm_hc_bytes()));
        EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, true, 1),
                  253'237'984LL);
        // MTP block = sparse anatomy without the mHC set (same code path).
        EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, false, 1),
                  glm_round16(glm_sparse_gguf_bytes(1)));
    }
}

TEST(Glm5NextAttentionSizing, GgufSlotsCoverThePackedUpload) {
    auto cfg = glm53_flash_config(1);
    const auto& m = cfg.model;
    const auto wq = lc::WeightQuant::gguf_q8_0;

    // Real per-tensor bytes the upload will place for a KDA layer:
    // Q8_0-packed matrices + verbatim F32 vectors + Q8_0 hc_*_fn.
    const int64_t kda_packed =
        3 * q8_packed(8192, 4096) + q8_packed(64, 4096)
        + 2 * q8_packed(128, 4096) + 2 * q8_packed(8192, 128)
        + q8_packed(4096, 8192)
        + 3 * 8192LL * 4 * 4 + 64LL * 4 + 8192LL * 4 + 128LL * 4
        + 2 * (q8_packed(24, 16384) + 24LL * 4 + 3LL * 4);
    const int64_t kda_slot =
        glm5_next_attention_layer_bytes(m, wq, true, true, 1);
    EXPECT_GT(kda_slot, 0);
    EXPECT_GE(kda_slot, kda_packed);
    // And it covers the F32 pieces the plan validator counts unconverted.
    EXPECT_GT(kda_slot, 3 * 8192LL * 4 * 4 + 8192LL * 4 + 128LL * 4);

    const int64_t sparse_packed =
        q8_packed(1536, 4096) + q8_packed(16384, 1536) + q8_packed(512, 4096)
        + q8_packed(32768, 512) + q8_packed(4096, 16384)
        + q8_packed(4096, 1536) + q8_packed(128, 4096)
        + 1536LL * 2 + 512LL * 2 + 2 * 128LL * 2      // norms after F32→BF16
        + 32LL * 4096 * 4                              // indexer.proj F32
        + 128LL * 4096 * 2                             // gate dequanted to BF16
        + 4LL * 128 * 4                                // APE F32 verbatim
        + 2 * (q8_packed(24, 16384) + 24LL * 4 + 3LL * 4);
    const int64_t sparse_slot =
        glm5_next_attention_layer_bytes(m, wq, false, true, 1);
    EXPECT_GT(sparse_slot, 0);
    EXPECT_GE(sparse_slot, sparse_packed);

    // The compressor APE is sized F32 on the GGUF path and BF16 on the native
    // FP8 path, so the GGUF slot leads the FP8 slot by at least that delta
    // (it leads by far more — the Q8_0 matrices are sized BF16, not FP8).
    const int64_t ape_delta = 4LL * 128 * 4 - 4LL * 128 * 2;
    EXPECT_GE(sparse_slot - kFp8SparseHiddenTp1, ape_delta);

    // tp=2 shards exactly the sharded tensors; the replicated remainder does
    // not halve.
    for (int tp : {2, 4}) {
        EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, true, true, tp),
                  glm_round16(glm_kda_gguf_bytes(tp) + glm_hc_bytes()));
        EXPECT_EQ(glm5_next_attention_layer_bytes(m, wq, false, true, tp),
                  glm_round16(glm_sparse_gguf_bytes(tp) + glm_hc_bytes()));
    }
    EXPECT_GT(glm5_next_attention_layer_bytes(m, wq, false, true, 2) * 2,
              glm5_next_attention_layer_bytes(m, wq, false, true, 1));
}

TEST(Glm5NextAttentionSizing, GgufKdaExceedsNativeOnlyByTheF32Widths) {
    auto cfg = glm53_flash_config(1);
    const auto& m = cfg.model;
    // KDA matrices are BF16 in both arms; the GGUF arm differs ONLY in the
    // three convs [8192,1,4] and o_norm [128] being sized F32 instead of BF16.
    const int64_t delta = 3 * 8192LL * 4 * 2 + 128LL * 2;
    EXPECT_EQ(glm5_next_attention_layer_bytes(m, lc::WeightQuant::gguf_q4_k,
                                              true, true, 1),
              kFp8KdaHiddenTp1 + delta);
}

// ── The arbiter: TpWeightSharder byte totals == the sizing formula ─────────
//
// validate_plan enforces exactly this equality at boot (Pass 5), so a drift
// between the sharder and the plan is a boot failure. Build REAL-shaped
// synthetic layers and compare.

namespace {

/// One KDA hidden layer: the 15 attention tensors (§3a) + the 6 mHC tensors.
struct Glm5KdaLayerFixture {
    std::vector<std::byte> qkv{glm_buf(8192ULL * 4096 * 2)};
    std::vector<std::byte> b{glm_buf(64ULL * 4096 * 2)};
    std::vector<std::byte> fga{glm_buf(128ULL * 4096 * 2)};
    std::vector<std::byte> fgb{glm_buf(8192ULL * 128 * 2)};
    std::vector<std::byte> conv{glm_buf(8192ULL * 4 * 2)};
    std::vector<std::byte> alog{glm_buf(64ULL * 4)};
    std::vector<std::byte> dt{glm_buf(8192ULL * 4)};
    std::vector<std::byte> onorm{glm_buf(128ULL * 2)};
    std::vector<std::byte> o{glm_buf(4096ULL * 8192 * 2)};
    std::vector<std::byte> hc_fn{glm_buf(24ULL * 16384 * 4)};
    std::vector<std::byte> hc_base{glm_buf(24ULL * 4)};
    std::vector<std::byte> hc_scale{glm_buf(3ULL * 4)};

    std::vector<WeightBundle> attention() const {
        std::vector<WeightBundle> a;
        for (auto c : {TensorComponent::kda_q_proj, TensorComponent::kda_k_proj,
                       TensorComponent::kda_v_proj})
            a.push_back(glm_bundle(qkv, {8192, 4096}, SafetensorsDtype::BF16, c));
        a.push_back(glm_bundle(b, {64, 4096}, SafetensorsDtype::BF16,
                               TensorComponent::kda_b_proj));
        for (auto c : {TensorComponent::kda_f_a_proj, TensorComponent::kda_g_a_proj})
            a.push_back(glm_bundle(fga, {128, 4096}, SafetensorsDtype::BF16, c));
        for (auto c : {TensorComponent::kda_f_b_proj, TensorComponent::kda_g_b_proj})
            a.push_back(glm_bundle(fgb, {8192, 128}, SafetensorsDtype::BF16, c));
        for (auto c : {TensorComponent::kda_q_conv1d, TensorComponent::kda_k_conv1d,
                       TensorComponent::kda_v_conv1d})
            a.push_back(glm_bundle(conv, {8192, 1, 4}, SafetensorsDtype::BF16, c));
        a.push_back(glm_bundle(alog, {64}, SafetensorsDtype::F32,
                               TensorComponent::kda_a_log));
        a.push_back(glm_bundle(dt, {8192}, SafetensorsDtype::F32,
                               TensorComponent::kda_dt_bias));
        a.push_back(glm_bundle(onorm, {128}, SafetensorsDtype::BF16,
                               TensorComponent::kda_o_norm));
        a.push_back(glm_bundle(o, {4096, 8192}, SafetensorsDtype::BF16,
                               TensorComponent::o_proj));
        for (auto c : {TensorComponent::hc_attn_fn, TensorComponent::hc_ffn_fn})
            // GF3.9: post-load reality — the loader widened fn to F32
            // (launch_mhc_pre contract); the sharder/plan see F32 bundles.
            a.push_back(glm_bundle(hc_fn, {24, 16384}, SafetensorsDtype::F32, c));
        for (auto c : {TensorComponent::hc_attn_base, TensorComponent::hc_ffn_base})
            a.push_back(glm_bundle(hc_base, {24}, SafetensorsDtype::F32, c));
        for (auto c : {TensorComponent::hc_attn_scale, TensorComponent::hc_ffn_scale})
            a.push_back(glm_bundle(hc_scale, {3}, SafetensorsDtype::F32, c));
        return a;
    }
};

/// One sparse-MLA hidden layer: 7 attention tensors + 7 indexer tensors + mHC.
struct Glm5SparseLayerFixture {
    std::vector<std::byte> q_a{glm_buf(1536ULL * 4096)};
    std::vector<std::byte> q_a_s{glm_buf(12ULL * 32 * 4)};
    std::vector<std::byte> q_a_n{glm_buf(1536ULL * 2)};
    std::vector<std::byte> q_b{glm_buf(16384ULL * 1536)};
    std::vector<std::byte> q_b_s{glm_buf(128ULL * 12 * 4)};
    std::vector<std::byte> kv_a{glm_buf(512ULL * 4096)};
    std::vector<std::byte> kv_a_s{glm_buf(4ULL * 32 * 4)};
    std::vector<std::byte> kv_a_n{glm_buf(512ULL * 2)};
    std::vector<std::byte> kv_b{glm_buf(32768ULL * 512 * 2)};
    std::vector<std::byte> o{glm_buf(4096ULL * 16384)};
    std::vector<std::byte> o_s{glm_buf(32ULL * 128 * 4)};
    std::vector<std::byte> wq_b{glm_buf(4096ULL * 1536 * 2)};
    std::vector<std::byte> wk{glm_buf(128ULL * 4096 * 2)};
    std::vector<std::byte> knorm{glm_buf(128ULL * 2)};
    std::vector<std::byte> wproj{glm_buf(32ULL * 4096 * 2)};
    std::vector<std::byte> kp_gate{glm_buf(128ULL * 4096 * 2)};
    std::vector<std::byte> kp_ape{glm_buf(4ULL * 128 * 2)};
    std::vector<std::byte> hc_fn{glm_buf(24ULL * 16384 * 4)};
    std::vector<std::byte> hc_base{glm_buf(24ULL * 4)};
    std::vector<std::byte> hc_scale{glm_buf(3ULL * 4)};

    std::vector<WeightBundle> attention(bool with_hc) const {
        std::vector<WeightBundle> a;
        auto qa = glm_bundle(q_a, {1536, 4096}, SafetensorsDtype::F8_E4M3,
                             TensorComponent::q_a_proj);
        glm_add_scale(qa, q_a_s, {12, 32});
        a.push_back(std::move(qa));
        a.push_back(glm_bundle(q_a_n, {1536}, SafetensorsDtype::BF16,
                               TensorComponent::q_a_norm));
        auto qb = glm_bundle(q_b, {16384, 1536}, SafetensorsDtype::F8_E4M3,
                             TensorComponent::q_b_proj);
        glm_add_scale(qb, q_b_s, {128, 12});
        a.push_back(std::move(qb));
        auto kva = glm_bundle(kv_a, {512, 4096}, SafetensorsDtype::F8_E4M3,
                              TensorComponent::kv_a_proj_with_mqa);
        glm_add_scale(kva, kv_a_s, {4, 32});
        a.push_back(std::move(kva));
        a.push_back(glm_bundle(kv_a_n, {512}, SafetensorsDtype::BF16,
                               TensorComponent::kv_a_norm));
        a.push_back(glm_bundle(kv_b, {32768, 512}, SafetensorsDtype::BF16,
                               TensorComponent::kv_b_proj));
        auto op = glm_bundle(o, {4096, 16384}, SafetensorsDtype::F8_E4M3,
                             TensorComponent::o_proj);
        glm_add_scale(op, o_s, {32, 128});
        a.push_back(std::move(op));
        if (with_hc) {
            for (auto c : {TensorComponent::hc_attn_fn, TensorComponent::hc_ffn_fn})
                // GF3.9: post-load reality — fn widened to F32 at load.
                a.push_back(glm_bundle(hc_fn, {24, 16384}, SafetensorsDtype::F32, c));
            for (auto c : {TensorComponent::hc_attn_base, TensorComponent::hc_ffn_base})
                a.push_back(glm_bundle(hc_base, {24}, SafetensorsDtype::F32, c));
            for (auto c : {TensorComponent::hc_attn_scale, TensorComponent::hc_ffn_scale})
                a.push_back(glm_bundle(hc_scale, {3}, SafetensorsDtype::F32, c));
        }
        return a;
    }

    std::vector<WeightBundle> indexer() const {
        std::vector<WeightBundle> v;
        v.push_back(glm_bundle(wq_b, {4096, 1536}, SafetensorsDtype::BF16,
                               TensorComponent::indexer_wq_b));
        v.push_back(glm_bundle(wk, {128, 4096}, SafetensorsDtype::BF16,
                               TensorComponent::indexer_wk));
        v.push_back(glm_bundle(knorm, {128}, SafetensorsDtype::BF16,
                               TensorComponent::indexer_k_norm_weight));
        v.push_back(glm_bundle(knorm, {128}, SafetensorsDtype::BF16,
                               TensorComponent::indexer_k_norm_bias));
        v.push_back(glm_bundle(wproj, {32, 4096}, SafetensorsDtype::BF16,
                               TensorComponent::indexer_weights_proj));
        v.push_back(glm_bundle(kp_gate, {128, 4096}, SafetensorsDtype::BF16,
                               TensorComponent::indexer_compressor_wgate));
        v.push_back(glm_bundle(kp_ape, {4, 128}, SafetensorsDtype::BF16,
                               TensorComponent::indexer_compressor_ape));
        return v;
    }
};

int64_t sharded_layer_bytes(const TpWeightSharder& sharder,
                            const std::vector<WeightBundle>& attention,
                            const std::vector<WeightBundle>& indexer,
                            int rank) {
    auto sharded = sharder.shard_attention_layer(attention, indexer, rank);
    int64_t total = 0;
    for (const auto& sb : sharded) total += sb.total_bytes();
    return total;
}

}  // namespace

TEST(Glm5NextAttentionSizing, SharderTotalsMatchSizingKdaLayer) {
    Glm5KdaLayerFixture fx;
    const auto attention = fx.attention();
    const std::vector<WeightBundle> no_indexer;  // KDA layers have no indexer

    for (int tp : {1, 2}) {
        auto cfg = glm53_flash_config(tp);
        ModelConfig model_cfg{cfg};
        TpWeightSharder sharder(model_cfg, tp);
        const int64_t expected = glm5_next_attention_layer_bytes(
            cfg.model, cfg.quantization.weights, /*linear=*/true,
            /*include_hc=*/true, tp);
        for (int rank = 0; rank < tp; ++rank) {
            const int64_t actual =
                sharded_layer_bytes(sharder, attention, no_indexer, rank);
            EXPECT_EQ(actual, glm_kda_bytes(tp) + glm_hc_bytes())
                << "tp=" << tp << " rank=" << rank;
            EXPECT_EQ(glm_round16(actual), expected)
                << "tp=" << tp << " rank=" << rank;
        }
    }
}

TEST(Glm5NextAttentionSizing, SharderTotalsMatchSizingSparseLayer) {
    Glm5SparseLayerFixture fx;
    const auto attention = fx.attention(/*with_hc=*/true);
    const auto indexer = fx.indexer();

    for (int tp : {1, 2}) {
        auto cfg = glm53_flash_config(tp);
        ModelConfig model_cfg{cfg};
        TpWeightSharder sharder(model_cfg, tp);
        const int64_t expected = glm5_next_attention_layer_bytes(
            cfg.model, cfg.quantization.weights, /*linear=*/false,
            /*include_hc=*/true, tp);
        for (int rank = 0; rank < tp; ++rank) {
            const int64_t actual =
                sharded_layer_bytes(sharder, attention, indexer, rank);
            EXPECT_EQ(actual, glm_sparse_bytes(tp) + glm_hc_bytes())
                << "tp=" << tp << " rank=" << rank;
            EXPECT_EQ(glm_round16(actual), expected)
                << "tp=" << tp << " rank=" << rank;
        }
    }
}

TEST(Glm5NextAttentionSizing, SharderTotalsMatchSizingMtpBlock) {
    Glm5SparseLayerFixture fx;
    const auto attention = fx.attention(/*with_hc=*/false);  // §5: no hc on MTP
    const auto indexer = fx.indexer();

    for (int tp : {1, 2}) {
        auto cfg = glm53_flash_config(tp);
        ModelConfig model_cfg{cfg};
        TpWeightSharder sharder(model_cfg, tp);
        const int64_t expected = glm5_next_attention_layer_bytes(
            cfg.model, cfg.quantization.weights, /*linear=*/false,
            /*include_hc=*/false, tp);
        for (int rank = 0; rank < tp; ++rank) {
            const int64_t actual =
                sharded_layer_bytes(sharder, attention, indexer, rank);
            EXPECT_EQ(actual, glm_sparse_bytes(tp)) << "tp=" << tp;
            EXPECT_EQ(glm_round16(actual), expected) << "tp=" << tp;
        }
    }
}

// ── Upload plan: hybrid per-layer slots, no output_hc, four MTP extras ─────

TEST(Glm5NextUploadPlan, PerLayerSlotsFollowLayerTypes) {
    auto cfg = glm53_flash_config(2);
    ModelConfig model_cfg{cfg};
    layerstorm::model::Fp8E4M3 fp8;
    auto plan = build_upload_plan(model_cfg, cfg, fp8, /*tp=*/2, /*rank=*/0);

    const int64_t kda = glm5_next_attention_layer_bytes(
        cfg.model, cfg.quantization.weights, true, true, 2);
    const int64_t sparse = glm5_next_attention_layer_bytes(
        cfg.model, cfg.quantization.weights, false, true, 2);

    for (int l = 0; l < 45; ++l) {
        auto* slot = plan.find(PinnedComponent::attention, l);
        ASSERT_NE(slot, nullptr) << "layer " << l;
        const bool linear = model_cfg.is_linear_attention_layer(l);
        EXPECT_EQ(slot->size_bytes, linear ? kda : sparse) << "layer " << l;
    }

    // The MTP block (layer 45) is sparse-MLA WITHOUT mHC.
    auto* mtp_attn = plan.find(PinnedComponent::attention, 45);
    ASSERT_NE(mtp_attn, nullptr);
    EXPECT_EQ(mtp_attn->size_bytes,
              glm5_next_attention_layer_bytes(cfg.model,
                                              cfg.quantization.weights,
                                              false, /*include_hc=*/false, 2));

    // glm5_next has NO model-level output_hc set (unlike V4) even though mHC
    // is on.
    EXPECT_TRUE(model_cfg.has_mhc());
    EXPECT_EQ(plan.find(PinnedComponent::output_hc, -1), nullptr);
}

TEST(Glm5NextUploadPlan, GgufPerLayerSlotsAndMtpUseTheUpperBoundArm) {
    // Exercises the two build_upload_plan call sites (hidden layers + the MTP
    // block) with a GGUF weight quant — the combination that used to throw.
    auto cfg = glm53_flash_config(2);
    cfg.quantization.weights = lc::WeightQuant::gguf_q4_k;
    ModelConfig model_cfg{cfg};
    layerstorm::model::GgufQuantInterface gq(
        layerstorm::model::GgufKQuantType::Q4_K);
    auto plan = build_upload_plan(model_cfg, cfg, gq, /*tp=*/2, /*rank=*/0);

    const int64_t kda = glm5_next_attention_layer_bytes(
        cfg.model, cfg.quantization.weights, true, true, 2);
    const int64_t sparse = glm5_next_attention_layer_bytes(
        cfg.model, cfg.quantization.weights, false, true, 2);
    EXPECT_EQ(kda, glm_round16(glm_kda_gguf_bytes(2) + glm_hc_bytes()));
    EXPECT_EQ(sparse, glm_round16(glm_sparse_gguf_bytes(2) + glm_hc_bytes()));

    for (int l = 0; l < 45; ++l) {
        auto* slot = plan.find(PinnedComponent::attention, l);
        ASSERT_NE(slot, nullptr) << "layer " << l;
        EXPECT_EQ(slot->size_bytes,
                  model_cfg.is_linear_attention_layer(l) ? kda : sparse)
            << "layer " << l;
    }
    auto* mtp_attn = plan.find(PinnedComponent::attention, 45);
    ASSERT_NE(mtp_attn, nullptr);
    EXPECT_EQ(mtp_attn->size_bytes,
              glm5_next_attention_layer_bytes(cfg.model,
                                              cfg.quantization.weights,
                                              false, /*include_hc=*/false, 2));
    EXPECT_EQ(mtp_attn->size_bytes, glm_round16(glm_sparse_gguf_bytes(2)));

    // The layout aggregator (and with it LayerRegistry's per-layer sizing)
    // walks the same slots without throwing.
    auto layout = compute_pinned_layout(model_cfg, cfg, gq, 2, 0);
    EXPECT_GT(layout.total_bytes, 0);
    EXPECT_EQ(layout.attention_bytes, 34 * kda + 11 * sparse);
    EXPECT_GT(layout.mtp_bytes, 0);
}

TEST(Glm5NextUploadPlan, MtpExtrasAreExactlyTheFourCheckpointTensors) {
    auto cfg = glm53_flash_config(2);
    ModelConfig model_cfg{cfg};
    layerstorm::model::Fp8E4M3 fp8;
    auto plan = build_upload_plan(model_cfg, cfg, fp8, /*tp=*/2, /*rank=*/0);

    // Present: eh_proj [4096, 8192] BF16 ÷tp, enorm/hnorm/shared_head.norm
    // [4096] BF16.
    auto* eh = plan.find(PinnedComponent::mtp_eh_proj, 45);
    ASSERT_NE(eh, nullptr);
    EXPECT_EQ(eh->size_bytes, 4096LL * 8192 * 2 / 2);
    for (auto comp : {PinnedComponent::mtp_enorm, PinnedComponent::mtp_hnorm,
                      PinnedComponent::mtp_shared_head_norm}) {
        auto* s = plan.find(comp, 45);
        ASSERT_NE(s, nullptr);
        EXPECT_EQ(s->size_bytes, 4096LL * 2);
    }

    // Absent: no per-MTP embed_tokens, no shared_head.head weight (shares
    // lm_head) — MODELINFO §5.
    EXPECT_EQ(plan.find(PinnedComponent::mtp_embed_tokens, 45), nullptr);
    EXPECT_EQ(plan.find(PinnedComponent::mtp_shared_head_weight, 45), nullptr);
}

TEST(Glm5NextUploadPlan, LayoutTotalsAreConsistentAndTpShrinks) {
    auto cfg1 = glm53_flash_config(1);
    ModelConfig mc1{cfg1};
    layerstorm::model::Fp8E4M3 fp8;
    auto layout1 = compute_pinned_layout(mc1, cfg1, fp8, 1, 0);

    auto cfg2 = glm53_flash_config(2);
    ModelConfig mc2{cfg2};
    auto layout2 = compute_pinned_layout(mc2, cfg2, fp8, 2, 0);

    EXPECT_GT(layout1.total_bytes, 0);
    EXPECT_LT(layout2.total_bytes, layout1.total_bytes);
    EXPECT_EQ(layout1.output_hc_bytes, 0);   // no model-level hc for glm5_next
    EXPECT_EQ(layout2.output_hc_bytes, 0);

    // Attention subtotal = 34 KDA + 11 sparse hidden layers (the MTP block's
    // attention lands in mtp_bytes).
    const int64_t kda = glm5_next_attention_layer_bytes(
        cfg1.model, cfg1.quantization.weights, true, true, 1);
    const int64_t sparse = glm5_next_attention_layer_bytes(
        cfg1.model, cfg1.quantization.weights, false, true, 1);
    EXPECT_EQ(layout1.attention_bytes, 34 * kda + 11 * sparse);
    EXPECT_GT(layout1.mtp_bytes, 0);
}

// ── GF3.15: checkpoint-exact attention slots (TD-AUTOCONFIG-PINNED-BYTES-
//    UPPER-BOUND / TD-GG9-ATTN-GGUF-UPPER-BOUND) ─────────────────────────────
//
// The pinned region is carved before any weight load, so the glm5_next GGUF arm
// sized every packed matrix at a BF16 upper bound. `gguf_non_expert_widths_from
// _path` reads the real per-tensor k-quant types from the tensor-info headers
// (the same headers the engine already scans for the routed experts), which
// makes the slot exact. These tests pin the three properties that make that
// safe: nullptr is unchanged, a supplied width never GROWS a slot, and the
// widths only ever apply to the packed matrices — never to a tensor the loader
// transforms at load.

TEST(Glm5NextGgufWidths, NullWidthsAreByteIdenticalToTheUpperBound) {
    // The defaulted nullptr must reproduce the pre-GF3.15 arithmetic exactly:
    // every pure-config caller (config_validator's VRAM budget pass, every
    // non-glm5_next architecture) still gets the old numbers.
    for (int tp : {1, 2}) {
        auto cfg = glm53_flash_config(tp);
        cfg.quantization.weights = lc::WeightQuant::gguf_q4_k;
        ModelConfig model_cfg{cfg};
        for (bool linear : {true, false}) {
            for (bool hc : {true, false}) {
                const int64_t with_null = glm5_next_attention_layer_bytes(
                    cfg.model, cfg.quantization.weights, linear, hc, tp,
                    /*widths=*/nullptr, /*layer_idx=*/3);
                const int64_t legacy = glm5_next_attention_layer_bytes(
                    cfg.model, cfg.quantization.weights, linear, hc, tp);
                EXPECT_EQ(with_null, legacy)
                    << "tp=" << tp << " linear=" << linear << " hc=" << hc;
            }
        }
    }
}

TEST(Glm5NextGgufWidths, EmptyWidthsAndUnknownLayersKeepTheBound) {
    auto cfg = glm53_flash_config(1);
    cfg.quantization.weights = lc::WeightQuant::gguf_q4_k;
    ModelConfig model_cfg{cfg};
    layerstorm::model::GgufNonExpertWidths empty;
    EXPECT_TRUE(empty.empty());

    // A width map that knows only layer 0 must leave layer 3's slot alone.
    layerstorm::model::GgufNonExpertWidths only_l0;
    only_l0.packed[layerstorm::model::GgufNonExpertWidths::key(
        0, layerstorm::model::TensorComponent::kda_q_proj)] =
        layerstorm::model::GgufKQuantType::Q8_0;

    const int64_t bound = glm5_next_attention_layer_bytes(
        cfg.model, cfg.quantization.weights, true, true, 1);
    EXPECT_EQ(glm5_next_attention_layer_bytes(
                  cfg.model, cfg.quantization.weights, true, true, 1,
                  &empty, 0), bound);
    EXPECT_EQ(glm5_next_attention_layer_bytes(
                  cfg.model, cfg.quantization.weights, true, true, 1,
                  &only_l0, 3), bound);
    // ...and must SHRINK layer 0's, because Q8_0 (34/32 B/elem) is narrower
    // than the BF16 bound it replaces.
    EXPECT_LT(glm5_next_attention_layer_bytes(
                  cfg.model, cfg.quantization.weights, true, true, 1,
                  &only_l0, 0), bound);
}

TEST(Glm5NextGgufWidths, EveryKnownWidthOnlyEverShrinksTheSlot) {
    // Q8_0 is the WIDEST k-quant this engine reads (34/32 = 1.0625 B/elem) and
    // still narrower than BF16, so no k-quant width may grow a slot. An
    // over-sized slot is trailing slack; an UNDER-sized one corrupts the KV
    // region, which is why validate_plan keeps throwing on actual > slot.
    auto cfg = glm53_flash_config(2);
    cfg.quantization.weights = lc::WeightQuant::gguf_q4_k;
    ModelConfig model_cfg{cfg};
    using TC = layerstorm::model::TensorComponent;
    using KQ = layerstorm::model::GgufKQuantType;
    for (KQ ty : {KQ::Q2_K, KQ::Q3_K, KQ::Q4_K, KQ::Q5_K, KQ::Q6_K, KQ::Q8_0}) {
        layerstorm::model::GgufNonExpertWidths w;
        for (TC c : {TC::kda_q_proj, TC::kda_k_proj, TC::kda_v_proj,
                     TC::kda_b_proj, TC::kda_f_a_proj, TC::kda_g_a_proj,
                     TC::kda_f_b_proj, TC::kda_g_b_proj, TC::o_proj,
                     TC::q_a_proj, TC::q_b_proj, TC::kv_a_proj_with_mqa,
                     TC::indexer_wq_b, TC::indexer_wk})
            w.packed[layerstorm::model::GgufNonExpertWidths::key(7, c)] = ty;
        for (bool linear : {true, false}) {
            const int64_t bound = glm5_next_attention_layer_bytes(
                cfg.model, cfg.quantization.weights, linear, true, 2);
            const int64_t exact = glm5_next_attention_layer_bytes(
                cfg.model, cfg.quantization.weights, linear, true, 2, &w, 7);
            EXPECT_LT(exact, bound) << "linear=" << linear;
            EXPECT_GT(exact, 0);
        }
    }
}

TEST(Glm5NextGgufWidths, TransformedTensorsKeepTheirUploadedWidth) {
    // The loader DEQUANTS or WIDENS four things at load, and their slot must
    // stay at the TRANSFORMED width even when the file says otherwise:
    //   kv_b halves -> one combined BF16 kv_b_proj (GLM-1)
    //   IndexPool compressor gate -> BF16 (the executor GEMM is BF16-only)
    //   hc_*_fn -> F32 (launch_mhc_pre's contract)
    //   the four dtype-halved norms -> BF16
    // Declaring a k-quant width for any of them must therefore change NOTHING.
    auto cfg = glm53_flash_config(1);
    cfg.quantization.weights = lc::WeightQuant::gguf_q4_k;
    ModelConfig model_cfg{cfg};
    using TC = layerstorm::model::TensorComponent;
    layerstorm::model::GgufNonExpertWidths w;
    for (TC c : {TC::mla_k_b_split, TC::mla_v_b_split, TC::kv_b_proj,
                 TC::indexer_compressor_wgate, TC::indexer_compressor_ape,
                 TC::hc_attn_fn, TC::hc_ffn_fn, TC::q_a_norm, TC::kv_a_norm,
                 TC::indexer_k_norm_weight, TC::indexer_k_norm_bias,
                 TC::indexer_weights_proj, TC::kda_o_norm, TC::kda_a_log,
                 TC::kda_dt_bias, TC::kda_q_conv1d, TC::kda_k_conv1d,
                 TC::kda_v_conv1d})
        w.packed[layerstorm::model::GgufNonExpertWidths::key(5, c)] =
            layerstorm::model::GgufKQuantType::Q4_K;
    for (bool linear : {true, false})
        EXPECT_EQ(glm5_next_attention_layer_bytes(
                      cfg.model, cfg.quantization.weights, linear, true, 1,
                      &w, 5),
                  glm5_next_attention_layer_bytes(
                      cfg.model, cfg.quantization.weights, linear, true, 1))
            << "linear=" << linear;
}

TEST(Glm5NextGgufWidths, LayoutAndRegistryShrinkTogether) {
    // The whole chain — build_upload_plan, compute_pinned_layout and the
    // LayerRegistry budget it feeds — must move as one, or the VRAM allocator
    // carves against a figure nothing else believes.
    auto cfg = glm53_flash_config(1);
    cfg.quantization.weights = lc::WeightQuant::gguf_q4_k;
    ModelConfig model_cfg{cfg};
    layerstorm::model::GgufQuantInterface gq(
        layerstorm::model::GgufKQuantType::Q4_K);
    using TC = layerstorm::model::TensorComponent;
    layerstorm::model::GgufNonExpertWidths w;
    for (int l = 0; l <= 45; ++l)
        for (TC c : {TC::kda_q_proj, TC::kda_k_proj, TC::kda_v_proj,
                     TC::o_proj, TC::q_a_proj, TC::q_b_proj,
                     TC::kv_a_proj_with_mqa, TC::indexer_wq_b, TC::indexer_wk})
            w.packed[layerstorm::model::GgufNonExpertWidths::key(l, c)] =
                layerstorm::model::GgufKQuantType::Q8_0;

    auto bound = compute_pinned_layout(model_cfg, cfg, gq, 1, 0);
    auto exact = compute_pinned_layout(model_cfg, cfg, gq, 1, 0, &w);
    EXPECT_LT(exact.total_bytes, bound.total_bytes);
    EXPECT_LT(exact.attention_bytes, bound.attention_bytes);
    EXPECT_LT(exact.mtp_bytes, bound.mtp_bytes);
    // Nothing OUTSIDE the attention/MTP slots may move — the widths are an
    // attention-slot fact only.
    EXPECT_EQ(exact.embedding_bytes, bound.embedding_bytes);
    EXPECT_EQ(exact.output_head_bytes, bound.output_head_bytes);
    EXPECT_EQ(exact.shared_expert_bytes, bound.shared_expert_bytes);
    EXPECT_EQ(exact.dense_ffn_bytes, bound.dense_ffn_bytes);
    EXPECT_EQ(exact.layer_norm_bytes, bound.layer_norm_bytes);
    EXPECT_EQ(exact.gating_bytes, bound.gating_bytes);

    layerstorm::model::LayerRegistry rb(model_cfg, cfg, gq);
    layerstorm::model::LayerRegistry re(model_cfg, cfg, gq, &w);
    EXPECT_EQ(rb.pinned_layout().total_bytes, bound.total_bytes);
    EXPECT_EQ(re.pinned_layout().total_bytes, exact.total_bytes);
    const auto bb = rb.estimate_gpu_budgets();
    const auto be = re.estimate_gpu_budgets();
    ASSERT_EQ(bb.size(), be.size());
    for (size_t i = 0; i < bb.size(); ++i)
        EXPECT_GE(be[i].available_for_cache_bytes, bb[i].available_for_cache_bytes)
            << "gpu " << i << " must gain (or keep) cache room, never lose it";
}

TEST(Glm5NextGgufWidths, NonGlm5ArchitecturesAreUntouched) {
    // V3.2 and V4 never consume the widths — their sizing must be identical
    // whether or not a width map exists (byte-identity for GLM-5.2 / V4).
    layerstorm::model::GgufNonExpertWidths w;
    using TC = layerstorm::model::TensorComponent;
    for (int l = 0; l < 64; ++l)
        for (TC c : {TC::q_a_proj, TC::q_b_proj, TC::kv_a_proj_with_mqa,
                     TC::o_proj})
            w.packed[layerstorm::model::GgufNonExpertWidths::key(l, c)] =
                layerstorm::model::GgufKQuantType::Q4_K;
    auto cfg = v32_config_tp2();
    cfg.quantization.weights = lc::WeightQuant::gguf;
    ModelConfig model_cfg{cfg};
    layerstorm::model::GgufQuantInterface gq(
        layerstorm::model::GgufKQuantType::Q4_K);
    EXPECT_EQ(compute_pinned_layout(model_cfg, cfg, gq, 2, 0, &w).total_bytes,
              compute_pinned_layout(model_cfg, cfg, gq, 2, 0).total_bytes);
}
