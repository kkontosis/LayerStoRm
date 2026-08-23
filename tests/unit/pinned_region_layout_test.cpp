#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

#include "model/pinned_region_layout.h"
#include "model/pinned_upload_plan.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/fp8.h"
#include "model/quantization/nvfp4.h"
#include "model/quantization/gguf_kquant.h"

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
