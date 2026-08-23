#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/model_config.h"
#include "model/pinned_upload_plan.h"
#include "model/quantization/nvfp4.h"
#include "model/weight_loader/tp_weight_sharder.h"
#include "model/weight_loader/weight_handler.h"
#include "model/weight_loader/weight_loader.h"

namespace lc = layerstorm::config;
using layerstorm::model::build_upload_plan;
using layerstorm::model::LoadedModel;
using layerstorm::model::ModelConfig;
using layerstorm::model::Nvfp4;
using layerstorm::model::PinnedComponent;
using layerstorm::model::RawTensor;
using layerstorm::model::SafetensorsDtype;
using layerstorm::model::ShardedWeightBundle;
using layerstorm::model::TensorComponent;
using layerstorm::model::TensorId;
using layerstorm::model::TensorOwner;
using layerstorm::model::TensorRole;
using layerstorm::model::TpWeightSharder;
using layerstorm::model::validate_plan;
using layerstorm::model::WeightBundle;

// ── Config helper ──────────────────────────────────────────────────────────

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

// ── Synthetic LoadedModel builder ──────────────────────────────────────────

struct SyntheticModelBuilder {
    std::vector<std::vector<std::byte>> buffers;
    LoadedModel model;

    std::span<const std::byte> alloc(size_t n) {
        buffers.emplace_back(n, std::byte{0x42});
        return {buffers.back().data(), buffers.back().size()};
    }

    WeightBundle make_fp8_bundle(std::vector<int64_t> shape, TensorComponent comp,
                                 TensorOwner owner, int layer) {
        int64_t size = 1;
        for (auto d : shape) size *= d;
        WeightBundle b;
        b.id = TensorId{comp, TensorRole::weight, owner, layer, -1};
        b.weight = RawTensor{
            .data = alloc(static_cast<size_t>(size)),
            .dtype = SafetensorsDtype::F8_E4M3,
            .shape = std::move(shape),
        };
        return b;
    }

    WeightBundle make_bf16_bundle(int64_t elems, TensorComponent comp,
                                  TensorOwner owner, int layer) {
        WeightBundle b;
        b.id = TensorId{comp, TensorRole::weight, owner, layer, -1};
        b.weight = RawTensor{
            .data = alloc(static_cast<size_t>(elems * 2)),
            .dtype = SafetensorsDtype::BF16,
            .shape = {elems},
        };
        return b;
    }

    WeightBundle make_bf16_bundle(std::vector<int64_t> shape, TensorComponent comp,
                                  TensorOwner owner, int layer) {
        int64_t size = 1;
        for (auto d : shape) size *= d;
        WeightBundle b;
        b.id = TensorId{comp, TensorRole::weight, owner, layer, -1};
        b.weight = RawTensor{
            .data = alloc(static_cast<size_t>(size * 2)),
            .dtype = SafetensorsDtype::BF16,
            .shape = std::move(shape),
        };
        return b;
    }

    WeightBundle make_f32_bundle(int64_t elems, TensorComponent comp,
                                 TensorOwner owner, int layer) {
        WeightBundle b;
        b.id = TensorId{comp, TensorRole::weight, owner, layer, -1};
        b.weight = RawTensor{
            .data = alloc(static_cast<size_t>(elems * 4)),
            .dtype = SafetensorsDtype::F32,
            .shape = {elems},
        };
        return b;
    }

    // NVFP4 bundle: U8 weight (params/2 bytes) + F8_E4M3 scale + F32 scalars.
    // U8 shape is storage shape (half the logical params): e.g. logical [16384, 7168]
    // stored as U8 [16384, 3584] because each byte holds 2 FP4 values.
    WeightBundle make_nvfp4_bundle(std::vector<int64_t> shape, TensorComponent comp,
                                   TensorOwner owner, int layer) {
        int64_t params = 1;
        for (auto d : shape) params *= d;
        int64_t weight_bytes = (params + 1) / 2;
        int64_t scale_elems = (params + 15) / 16;

        // U8 storage shape: last dim halved (FP4 packing)
        auto u8_shape = shape;
        if (!u8_shape.empty())
            u8_shape.back() /= 2;
        // Scale shape: last dim divided by 16
        auto scale_shape = shape;
        if (!scale_shape.empty())
            scale_shape.back() = (scale_shape.back() + 15) / 16;

        WeightBundle b;
        b.id = TensorId{comp, TensorRole::weight, owner, layer, -1};
        b.weight = RawTensor{
            .data = alloc(static_cast<size_t>(weight_bytes)),
            .dtype = SafetensorsDtype::U8,
            .shape = std::move(u8_shape),
        };
        b.aux.emplace_back(TensorRole::weight_scale, RawTensor{
            .data = alloc(static_cast<size_t>(scale_elems)),
            .dtype = SafetensorsDtype::F8_E4M3,
            .shape = std::move(scale_shape),
        });
        b.aux.emplace_back(TensorRole::weight_scale_2, RawTensor{
            .data = alloc(sizeof(float)),
            .dtype = SafetensorsDtype::F32,
            .shape = {},
        });
        b.aux.emplace_back(TensorRole::input_scale, RawTensor{
            .data = alloc(sizeof(float)),
            .dtype = SafetensorsDtype::F32,
            .shape = {},
        });
        return b;
    }

    void build_valid(const ModelConfig& model_cfg, const lc::Config& cfg, int tp) {
        const auto& m = model_cfg.raw();

        // Embedding (BF16, full unsharded size — validation divides by tp)
        int64_t embed_bytes = static_cast<int64_t>(m.vocab_size) * m.hidden_size * 2;
        model.embedding = WeightBundle{};
        model.embedding->id = TensorId{TensorComponent::embedding, TensorRole::weight,
                                        TensorOwner::model_level, -1, -1};
        model.embedding->weight = RawTensor{
            .data = alloc(static_cast<size_t>(embed_bytes)),
            .dtype = SafetensorsDtype::BF16,
            .shape = {static_cast<int64_t>(m.vocab_size),
                      static_cast<int64_t>(m.hidden_size)},
        };

        // Output head (BF16, same shape as embedding)
        model.output_head = WeightBundle{};
        model.output_head->id = TensorId{TensorComponent::output_head, TensorRole::weight,
                                          TensorOwner::model_level, -1, -1};
        model.output_head->weight = RawTensor{
            .data = alloc(static_cast<size_t>(embed_bytes)),
            .dtype = SafetensorsDtype::BF16,
            .shape = {static_cast<int64_t>(m.vocab_size),
                      static_cast<int64_t>(m.hidden_size)},
        };

        // Final norm (F32 in V3.2 checkpoints, hidden_size elements)
        int64_t fn_bytes = static_cast<int64_t>(m.hidden_size) * 4;
        model.final_norm = WeightBundle{};
        model.final_norm->id = TensorId{TensorComponent::final_norm, TensorRole::weight,
                                         TensorOwner::model_level, -1, -1};
        model.final_norm->weight = RawTensor{
            .data = alloc(static_cast<size_t>(fn_bytes)),
            .dtype = SafetensorsDtype::F32,
            .shape = {static_cast<int64_t>(m.hidden_size)},
        };

        // Build attention layers using the sharder to get correct sizes.
        // We build bundles that, when sharded, produce exactly the plan slot size.
        TpWeightSharder sharder(model_cfg, tp);

        auto build_attention_layer = [&](int layer_idx, bool include_indexer,
                                        bool bf16_oproj = false) {
            LoadedModel::LayerWeights lw;
            lw.layer_idx = layer_idx;

            int n_heads = m.num_attention_heads;
            int qk_head = m.qk_nope_head_dim + m.qk_rope_head_dim;

            // V3.2 NVFP4 checkpoint: q/kv projections are BF16, o_proj is NVFP4.
            // MTP layers: o_proj is BF16 (INV-4f-6).
            // q_a: [hidden, q_lora_rank] — replicated, BF16
            lw.attention.push_back(make_bf16_bundle(
                {static_cast<int64_t>(m.hidden_size), static_cast<int64_t>(m.q_lora_rank)},
                TensorComponent::q_a_proj, TensorOwner::attention, layer_idx));

            // q_b: [q_lora_rank, n_heads * qk_head] — column-parallel, BF16
            lw.attention.push_back(make_bf16_bundle(
                {static_cast<int64_t>(m.q_lora_rank), static_cast<int64_t>(n_heads * qk_head)},
                TensorComponent::q_b_proj, TensorOwner::attention, layer_idx));

            // kv_a: [hidden, kv_lora_rank + qk_rope_head_dim] — replicated, BF16
            lw.attention.push_back(make_bf16_bundle(
                {static_cast<int64_t>(m.hidden_size),
                 static_cast<int64_t>(m.kv_lora_rank + m.qk_rope_head_dim)},
                TensorComponent::kv_a_proj_with_mqa, TensorOwner::attention, layer_idx));

            // kv_b: [kv_lora_rank, n_heads * (qk_nope + v_head)] — shardable, BF16
            lw.attention.push_back(make_bf16_bundle(
                {static_cast<int64_t>(m.kv_lora_rank),
                 static_cast<int64_t>(n_heads) * (m.qk_nope_head_dim + m.v_head_dim)},
                TensorComponent::kv_b_proj, TensorOwner::attention, layer_idx));

            // o: [n_heads * v_head_dim, hidden] — row-parallel
            // Regular layers: NVFP4; MTP layers: BF16 (INV-4f-6)
            if (bf16_oproj) {
                lw.attention.push_back(make_bf16_bundle(
                    {static_cast<int64_t>(n_heads) * m.v_head_dim,
                     static_cast<int64_t>(m.hidden_size)},
                    TensorComponent::o_proj, TensorOwner::attention, layer_idx));
            } else {
                lw.attention.push_back(make_nvfp4_bundle(
                    {static_cast<int64_t>(n_heads) * m.v_head_dim,
                     static_cast<int64_t>(m.hidden_size)},
                    TensorComponent::o_proj, TensorOwner::attention, layer_idx));
            }

            // F32 internal norms (RMSNorm weights are F32 in V3.2 checkpoint)
            lw.attention.push_back(make_f32_bundle(
                m.q_lora_rank, TensorComponent::q_a_norm,
                TensorOwner::attention, layer_idx));
            lw.attention.push_back(make_f32_bundle(
                m.kv_lora_rank, TensorComponent::kv_a_norm,
                TensorOwner::attention, layer_idx));

            // DSA indexer (BF16 for projections, F32 for norms)
            if (include_indexer) {
                lw.indexer.push_back(make_bf16_bundle(
                    {static_cast<int64_t>(m.q_lora_rank),
                     static_cast<int64_t>(m.index_n_heads) * m.index_head_dim},
                    TensorComponent::indexer_wq_b, TensorOwner::attention, layer_idx));
                lw.indexer.push_back(make_bf16_bundle(
                    {static_cast<int64_t>(m.hidden_size), static_cast<int64_t>(m.index_head_dim)},
                    TensorComponent::indexer_wk, TensorOwner::attention, layer_idx));
                lw.indexer.push_back(make_f32_bundle(
                    m.index_head_dim, TensorComponent::indexer_k_norm_weight,
                    TensorOwner::attention, layer_idx));
                lw.indexer.push_back(make_f32_bundle(
                    m.index_head_dim, TensorComponent::indexer_k_norm_bias,
                    TensorOwner::attention, layer_idx));
                lw.indexer.push_back(make_bf16_bundle(
                    {static_cast<int64_t>(m.hidden_size), static_cast<int64_t>(m.index_n_heads)},
                    TensorComponent::indexer_weights_proj, TensorOwner::attention, layer_idx));
            }

            // Layer norms: input + post-attention (F32 in V3.2 checkpoint)
            lw.norms.push_back(make_f32_bundle(
                m.hidden_size, TensorComponent::input_layernorm,
                TensorOwner::attention, layer_idx));
            lw.norms.push_back(make_f32_bundle(
                m.hidden_size, TensorComponent::post_attention_layernorm,
                TensorOwner::attention, layer_idx));
            return lw;
        };

        // Hidden layers
        model.layers.reserve(m.num_hidden_layers);
        for (int l = 0; l < m.num_hidden_layers; ++l) {
            model.layers.push_back(build_attention_layer(l, model_cfg.has_dsa()));
        }

        // MTP block layers — full transformer blocks
        // V3.2 NVFP4 checkpoints store MTP as BF16/F32: o_proj BF16, shared expert BF16 (INV-4f-6)
        if (m.num_nextn_predict_layers > 0) {
            model.mtp.emplace();
            model.mtp->base_layer_idx = m.num_hidden_layers;
            for (int mi = 0; mi < m.num_nextn_predict_layers; ++mi) {
                int mtp_idx = m.num_hidden_layers + mi;
                auto blk = build_attention_layer(mtp_idx, model_cfg.has_dsa(),
                                                 /*bf16_oproj=*/true);

                // MTP gating (FP32 gate_weight + F32 correction bias)
                blk.gating.push_back(make_f32_bundle(
                    static_cast<int64_t>(m.hidden_size) * m.n_routed_experts,
                    TensorComponent::gate_weight, TensorOwner::gating, mtp_idx));
                blk.gating.push_back(make_f32_bundle(
                    m.n_routed_experts, TensorComponent::gate_e_score_correction_bias,
                    TensorOwner::gating, mtp_idx));

                // MTP shared expert (BF16 in V3.2 NVFP4 checkpoints, per INV-4f-6)
                blk.shared_expert.push_back(make_bf16_bundle(
                    {static_cast<int64_t>(m.moe_intermediate_size),
                     static_cast<int64_t>(m.hidden_size)},
                    TensorComponent::gate_proj, TensorOwner::shared_expert, mtp_idx));
                blk.shared_expert.push_back(make_bf16_bundle(
                    {static_cast<int64_t>(m.moe_intermediate_size),
                     static_cast<int64_t>(m.hidden_size)},
                    TensorComponent::up_proj, TensorOwner::shared_expert, mtp_idx));
                blk.shared_expert.push_back(make_bf16_bundle(
                    {static_cast<int64_t>(m.hidden_size),
                     static_cast<int64_t>(m.moe_intermediate_size)},
                    TensorComponent::down_proj, TensorOwner::shared_expert, mtp_idx));

                model.mtp->block_layers.push_back(std::move(blk));

                // MTP-specific tensors (BF16, stored in mtp->tensors)
                model.mtp->tensors.push_back(make_bf16_bundle(
                    {static_cast<int64_t>(m.vocab_size), static_cast<int64_t>(m.hidden_size)},
                    TensorComponent::mtp_embed_tokens, TensorOwner::mtp, mtp_idx));
                model.mtp->tensors.push_back(make_bf16_bundle(
                    {static_cast<int64_t>(m.vocab_size), static_cast<int64_t>(m.hidden_size)},
                    TensorComponent::mtp_shared_head_weight, TensorOwner::mtp, mtp_idx));
                // shared_head_norm, enorm, hnorm are F32 in V3.2 (per INV-4f-6)
                model.mtp->tensors.push_back(make_f32_bundle(
                    static_cast<int64_t>(m.hidden_size),
                    TensorComponent::mtp_shared_head_norm, TensorOwner::mtp, mtp_idx));
                // eh_proj: [hidden_size, 2*hidden_size] — projects concat(embed, hidden)
                model.mtp->tensors.push_back(make_bf16_bundle(
                    {static_cast<int64_t>(m.hidden_size), static_cast<int64_t>(m.hidden_size) * 2},
                    TensorComponent::mtp_eh_proj, TensorOwner::mtp, mtp_idx));
                model.mtp->tensors.push_back(make_f32_bundle(
                    static_cast<int64_t>(m.hidden_size),
                    TensorComponent::mtp_enorm, TensorOwner::mtp, mtp_idx));
                model.mtp->tensors.push_back(make_f32_bundle(
                    static_cast<int64_t>(m.hidden_size),
                    TensorComponent::mtp_hnorm, TensorOwner::mtp, mtp_idx));
            }
        }

        (void)cfg;
    }
};

// ── Tests ──────────────────────────────────────────────────────────────────

TEST(ValidatePlan, HappyPath) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    SyntheticModelBuilder builder;
    builder.build_valid(model_cfg, cfg, 2);

    EXPECT_NO_THROW(validate_plan(plan, builder.model, model_cfg, cfg, nvfp4, 2));
}

TEST(ValidatePlan, TD53q_EmbeddingDtypeMismatch) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    SyntheticModelBuilder builder;
    builder.build_valid(model_cfg, cfg, 2);

    // Double the embedding size to simulate F32 (4 bytes/elem instead of 2)
    int64_t f32_size = static_cast<int64_t>(cfg.model.vocab_size) * cfg.model.hidden_size * 4;
    auto data = builder.alloc(static_cast<size_t>(f32_size));
    builder.model.embedding->weight.data = data;
    builder.model.embedding->weight.dtype = SafetensorsDtype::F32;

    EXPECT_THROW({
        try {
            validate_plan(plan, builder.model, model_cfg, cfg, nvfp4, 2);
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("TD-53q"), std::string::npos);
            throw;
        }
    }, std::runtime_error);
}

TEST(ValidatePlan, TD53y_MtpCountMismatch) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    SyntheticModelBuilder builder;
    builder.build_valid(model_cfg, cfg, 2);

    // Remove MTP block layers (config says 1, loaded has 0)
    builder.model.mtp->block_layers.clear();

    EXPECT_THROW({
        try {
            validate_plan(plan, builder.model, model_cfg, cfg, nvfp4, 2);
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("TD-53y"), std::string::npos);
            throw;
        }
    }, std::runtime_error);
}

TEST(ValidatePlan, TD55b_MissingKNormBias) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    SyntheticModelBuilder builder;
    builder.build_valid(model_cfg, cfg, 2);

    // Remove k_norm_bias from layer 0's indexer
    auto& indexer = builder.model.layers[0].indexer;
    indexer.erase(
        std::remove_if(indexer.begin(), indexer.end(),
            [](const WeightBundle& b) {
                return b.id.component == TensorComponent::indexer_k_norm_bias;
            }),
        indexer.end());

    EXPECT_THROW({
        try {
            validate_plan(plan, builder.model, model_cfg, cfg, nvfp4, 2);
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("TD-55b"), std::string::npos);
            throw;
        }
    }, std::runtime_error);
}

TEST(ValidatePlan, TD55e_MtpIndexerIncludedInPlan) {
    // TD-55e resolved: MTP indexer is now included in plan when has_dsa().
    // Removing an indexer component should cause undersized sharded attention.
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    SyntheticModelBuilder builder;
    builder.build_valid(model_cfg, cfg, 2);

    // Remove one indexer component from MTP block layer 0 — should fail
    // because sharded attention size will be smaller than plan slot.
    auto& indexer = builder.model.mtp->block_layers[0].indexer;
    indexer.erase(
        std::remove_if(indexer.begin(), indexer.end(),
            [](const WeightBundle& b) {
                return b.id.component == TensorComponent::indexer_wk;
            }),
        indexer.end());

    EXPECT_THROW(
        validate_plan(plan, builder.model, model_cfg, cfg, nvfp4, 2),
        std::runtime_error);
}

TEST(ValidatePlan, AttentionSlotMismatch) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    SyntheticModelBuilder builder;
    builder.build_valid(model_cfg, cfg, 2);

    // Corrupt layer 0 attention — add an extra spurious bundle
    builder.model.layers[0].attention.push_back(
        builder.make_fp8_bundle({1024}, TensorComponent::q_a_proj,
                                TensorOwner::attention, 0));

    EXPECT_THROW({
        try {
            validate_plan(plan, builder.model, model_cfg, cfg, nvfp4, 2);
        } catch (const std::runtime_error& e) {
            std::string msg(e.what());
            EXPECT_NE(msg.find("attention slot mismatch"), std::string::npos);
            EXPECT_NE(msg.find("layer 0"), std::string::npos);
            throw;
        }
    }, std::runtime_error);
}

TEST(ValidatePlan, MtpGatingMismatch) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    SyntheticModelBuilder builder;
    builder.build_valid(model_cfg, cfg, 2);

    // Add extra gating data to MTP block layer 0
    int mtp_idx = cfg.model.num_hidden_layers;
    builder.model.mtp->block_layers[0].gating.push_back(
        builder.make_f32_bundle(1024, TensorComponent::gate_weight,
                                TensorOwner::gating, mtp_idx));

    EXPECT_THROW({
        try {
            validate_plan(plan, builder.model, model_cfg, cfg, nvfp4, 2);
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("MTP gating slot mismatch"),
                      std::string::npos);
            throw;
        }
    }, std::runtime_error);
}

TEST(ValidatePlan, MtpEmbedTokensMismatch) {
    auto cfg = v32_config_tp2();
    ModelConfig model_cfg{cfg};
    Nvfp4 nvfp4;
    auto plan = build_upload_plan(model_cfg, cfg, nvfp4, 2, 0);

    SyntheticModelBuilder builder;
    builder.build_valid(model_cfg, cfg, 2);

    // Corrupt mtp_embed_tokens — replace with wrong size
    int mtp_idx = cfg.model.num_hidden_layers;
    auto& tensors = builder.model.mtp->tensors;
    for (auto& t : tensors) {
        if (t.id.component == TensorComponent::mtp_embed_tokens &&
            t.id.layer_idx == mtp_idx) {
            t.weight.data = builder.alloc(1024);  // wrong size
            break;
        }
    }

    EXPECT_THROW({
        try {
            validate_plan(plan, builder.model, model_cfg, cfg, nvfp4, 2);
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("MTP tensor slot mismatch"),
                      std::string::npos);
            throw;
        }
    }, std::runtime_error);
}
