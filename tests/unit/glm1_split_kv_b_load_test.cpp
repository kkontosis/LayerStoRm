// GLM-1 real-model load test: load the GLM-4.7-Flash GGUF (which ships the MLA
// up-projection PRE-SPLIT as attn_k_b/attn_v_b) through the host weight loader
// and assert the combined kv_b_proj is now PRESENT (not silently skipped) for
// every layer, as a BF16 bundle (gguf_type unset → BF16 absorbed-MLA path) with
// the expected combined shape [n_head*(qk_nope+v_head), kv_lora].
//
// This is a HOST-ONLY load (mmap + dequant/assemble); it needs no GPU. It is
// GUARDED on the 21.7 GB GGUF + config being present, so the unit suite stays
// green on machines without the artifact (GTEST_SKIP otherwise). Full forward /
// golden parity is GG-9.

#include <gtest/gtest.h>

#include <filesystem>

#include "config/config_preset.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/registry.h"
#include "model/weight_loader/weight_loader.h"

namespace fs = std::filesystem;
using namespace layerstorm;

namespace {

const char* kConfigPath = "test-data/config/glm_4_7_flash.json";
const char* kGgufPath =
    "test-data/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q5_K_XL.gguf";

const model::WeightBundle* find_kv_b(const model::LoadedModel::LayerWeights& layer) {
    for (const auto& b : layer.attention) {
        if (b.id.component == model::TensorComponent::kv_b_proj) return &b;
    }
    return nullptr;
}

}  // namespace

TEST(Glm1SplitKvBLoad, CombinedKvBProjPresentForAllLayers) {
    if (!fs::exists(kConfigPath) || !fs::exists(kGgufPath)) {
        GTEST_SKIP() << "GLM-4.7-Flash GGUF / config not present; skipping "
                        "real-model GLM-1 load test";
    }

    config::Config cfg = config::load_config(std::string(kConfigPath));
    ASSERT_EQ(cfg.model.weights_format, config::WeightsFormat::gguf);
    model::ModelConfig model_cfg(cfg);

    // GGUF experts are per-tensor mixed; the generic `gguf` quant sentinel
    // cannot size them, so build the concrete mixed interface from the file's
    // per-projection k-quant types (mirrors engine.cpp init).
    auto types = model::gguf_expert_types_from_path(cfg.model.weights_path,
                                                    cfg.model.use_mmap);
    model::GgufQuantInterface quant =
        model::make_gguf_quant(types.gate, types.up, types.down);
    model::LayerRegistry registry(model_cfg, cfg, quant);

    // skip_routed_experts: we only need the attention path (kv_b_proj). This
    // avoids de-stacking the 64×47 expert tensors — much faster + less RAM.
    model::LoadedModel model =
        model::load_weights(cfg, model_cfg, registry, /*skip_routed_experts=*/true);

    const int num_layers = cfg.model.num_hidden_layers;  // 47
    ASSERT_EQ(static_cast<int>(model.layers.size()), num_layers);

    const int64_t P = cfg.model.qk_nope_head_dim;   // 192
    const int64_t V = cfg.model.v_head_dim;         // 256
    const int64_t L = cfg.model.kv_lora_rank;       // 512
    const int64_t Hh = cfg.model.num_attention_heads;  // 20
    const int64_t expect_rows = Hh * (P + V);       // 8960

    int present = 0;
    for (int l = 0; l < num_layers; ++l) {
        const auto* kv_b = find_kv_b(model.layers[l]);
        ASSERT_NE(kv_b, nullptr)
            << "layer " << l << " missing kv_b_proj (split attn_k_b/attn_v_b "
               "were silently skipped — GLM-1 regression)";
        // BF16, NOT a GGUF in-kernel-dequant tensor (so q_absorb/kv_bv take the
        // BF16 branch: weight_is_fp8=false, gguf_type=-1).
        EXPECT_EQ(kv_b->weight.dtype, model::SafetensorsDtype::BF16) << "layer " << l;
        EXPECT_FALSE(kv_b->gguf_type().has_value()) << "layer " << l;
        ASSERT_EQ(kv_b->weight.shape.size(), 2u) << "layer " << l;
        EXPECT_EQ(kv_b->weight.shape[0], expect_rows) << "layer " << l;
        EXPECT_EQ(kv_b->weight.shape[1], L) << "layer " << l;
        // Owned BF16 buffer backs the span.
        EXPECT_NE(kv_b->owned_buf, nullptr) << "layer " << l;
        EXPECT_EQ(static_cast<int64_t>(kv_b->weight.data.size()),
                  expect_rows * L * 2) << "layer " << l;
        ++present;
    }
    EXPECT_EQ(present, num_layers);
}
