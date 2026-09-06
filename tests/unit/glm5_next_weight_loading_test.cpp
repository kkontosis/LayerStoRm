//==============================================================================
// glm5_next (GLM-5.3-Flash) weight-loading tests — GF3.3
//
// Four groups, all CPU-only and self-contained (no network, no GPU):
//   1. TensorId parsing of the REAL checkpoint tensor names (KDA components,
//      bare parameter names, the weight_scale_inv alias, MTP, negatives).
//   2. Coverage of the REAL model.safetensors.index.json (76,108 names):
//      every non-vision name must parse, and the per-layer component sets must
//      match the hybrid anatomy of MODELINFO §3a/§3b/§3c/§5.
//   3. QuantSkipList against the REAL config.json — the namespace trap
//      (`model.layers.N.*` module paths vs `model.language_model.layers.N.*`
//      tensor paths), including a full 76,108-name census golden.
//   4. End-to-end load_weights() over a synthetic mini-glm5_next FP8
//      checkpoint that uses the REAL naming scheme: structure, byte-for-byte
//      checksum of every layer-0/layer-3 tensor, and the two load-stopping
//      negatives (poisoned precision, zero-match namespace).
//
// Ground truth: spec/GLM-5.3-FLASH-MODELINFO.md §1/§3a/§3b/§3c/§7 and the
// files in test-data/GLM-5.3-Flash/.
//==============================================================================

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "config/config_parser.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/fp8.h"
#include "model/weight_loader/quant_skip_list.h"
#include "model/weight_loader/safetensors_reader.h"
#include "model/weight_loader/tensor_id.h"
#include "model/weight_loader/weight_handler.h"
#include "model/weight_loader/weight_loader.h"

namespace fs = std::filesystem;

using layerstorm::model::LayerRegistry;
using layerstorm::model::LoadedModel;
using layerstorm::model::QuantSkipList;
using layerstorm::model::RawTensor;
using layerstorm::model::SafetensorsDtype;
using layerstorm::model::TensorComponent;
using layerstorm::model::TensorOwner;
using layerstorm::model::TensorRole;
using layerstorm::model::WeightBundle;
using layerstorm::model::load_weights;
using layerstorm::model::parse_dtype;
using layerstorm::model::parse_hf_name;
using layerstorm::model::tensor_component_name;
using MConfig = layerstorm::model::ModelConfig;

namespace {

// ── Shared helpers ───────────────────────────────────────────────────────────

constexpr const char* kLm = "model.language_model.";

/// Assert the full parsed identity of one tensor name.
void expect_id(std::string_view name, TensorComponent component, TensorRole role,
               TensorOwner owner, int layer_idx, int expert_idx = -1) {
    auto id = parse_hf_name(name);
    ASSERT_TRUE(id.has_value()) << "failed to parse: " << name;
    EXPECT_EQ(id->component, component)
        << name << ": got component " << tensor_component_name(id->component);
    EXPECT_EQ(id->role, role) << name;
    EXPECT_EQ(id->owner, owner) << name;
    EXPECT_EQ(id->layer_idx, layer_idx) << name;
    EXPECT_EQ(id->expert_idx, expert_idx) << name;
}

/// Resolve a file inside test-data/GLM-5.3-Flash/ regardless of cwd.
fs::path real_data_file(const std::string& leaf) {
    const std::string rel = "test-data/GLM-5.3-Flash/" + leaf;
    const std::vector<std::string> candidates = {
        "../../" + rel,
        "../" + rel,
        rel,
#ifdef LAYERSTORM_SOURCE_DIR
        std::string(LAYERSTORM_SOURCE_DIR) + "/" + rel,
#endif
    };
    for (const auto& c : candidates) {
        std::error_code ec;
        if (fs::exists(fs::path(c), ec)) return fs::path(c);
    }
    ADD_FAILURE() << "Could not locate " << rel;
    return {};
}

fs::path real_model_dir() { return real_data_file("config.json").parent_path(); }

/// All tensor names of the real safetensors index (76,108 entries).
const std::vector<std::string>& real_index_names() {
    static const std::vector<std::string> names = [] {
        std::ifstream f(real_data_file("model.safetensors.index.json"));
        nlohmann::json j = nlohmann::json::parse(f);
        std::vector<std::string> out;
        const auto& wm = j.at("weight_map");
        out.reserve(wm.size());
        for (auto it = wm.begin(); it != wm.end(); ++it) out.push_back(it.key());
        return out;
    }();
    return names;
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════════════
// Group 1 — TensorId parsing of the real glm5_next tensor names
// ═════════════════════════════════════════════════════════════════════════════

TEST(Glm5NextWeightLoadingTensorId, KdaProjectionsOfALinearLayer) {
    const std::string L = std::string(kLm) + "layers.4.self_attn.";
    struct Case {
        const char* leaf;
        TensorComponent comp;
    };
    const Case cases[] = {
        {"q_proj", TensorComponent::kda_q_proj},
        {"k_proj", TensorComponent::kda_k_proj},
        {"v_proj", TensorComponent::kda_v_proj},
        {"b_proj", TensorComponent::kda_b_proj},
        {"f_a_proj", TensorComponent::kda_f_a_proj},
        {"f_b_proj", TensorComponent::kda_f_b_proj},
        {"g_a_proj", TensorComponent::kda_g_a_proj},
        {"g_b_proj", TensorComponent::kda_g_b_proj},
        {"q_conv1d", TensorComponent::kda_q_conv1d},
        {"k_conv1d", TensorComponent::kda_k_conv1d},
        {"v_conv1d", TensorComponent::kda_v_conv1d},
        {"o_norm", TensorComponent::kda_o_norm},
        // A KDA layer's output projection reuses the shared MLA component
        // (same name, same row-parallel sharding) — MODELINFO §3a.
        {"o_proj", TensorComponent::o_proj},
    };
    for (const auto& c : cases) {
        expect_id(L + c.leaf + ".weight", c.comp, TensorRole::weight,
                  TensorOwner::attention, 4);
    }
}

TEST(Glm5NextWeightLoadingTensorId, BareParameterNamesOfALinearLayer) {
    const std::string L = std::string(kLm) + "layers.4.";
    // No `.weight` suffix in the checkpoint — the whole path IS the component.
    expect_id(L + "self_attn.A_log", TensorComponent::kda_a_log,
              TensorRole::weight, TensorOwner::attention, 4);
    expect_id(L + "self_attn.dt_bias", TensorComponent::kda_dt_bias,
              TensorRole::weight, TensorOwner::attention, 4);
    expect_id(L + "hc_attn_base", TensorComponent::hc_attn_base,
              TensorRole::weight, TensorOwner::attention, 4);
    expect_id(L + "hc_attn_fn", TensorComponent::hc_attn_fn, TensorRole::weight,
              TensorOwner::attention, 4);
    expect_id(L + "hc_attn_scale", TensorComponent::hc_attn_scale,
              TensorRole::weight, TensorOwner::attention, 4);
    expect_id(L + "hc_ffn_base", TensorComponent::hc_ffn_base,
              TensorRole::weight, TensorOwner::attention, 4);
    expect_id(L + "hc_ffn_fn", TensorComponent::hc_ffn_fn, TensorRole::weight,
              TensorOwner::attention, 4);
    expect_id(L + "hc_ffn_scale", TensorComponent::hc_ffn_scale,
              TensorRole::weight, TensorOwner::attention, 4);
}

TEST(Glm5NextWeightLoadingTensorId, IndexPoolCompressorBareNames) {
    const std::string L = std::string(kLm) + "layers.3.self_attn.indexer.";
    expect_id(L + "index_kpool_compress_gate",
              TensorComponent::indexer_compressor_wgate, TensorRole::weight,
              TensorOwner::attention, 3);
    expect_id(L + "index_kpool_compress_ape",
              TensorComponent::indexer_compressor_ape, TensorRole::weight,
              TensorOwner::attention, 3);
}

TEST(Glm5NextWeightLoadingTensorId, SparseLayerMlaAndScaleInvAlias) {
    const std::string L = std::string(kLm) + "layers.3.self_attn.";
    expect_id(L + "q_a_proj.weight", TensorComponent::q_a_proj,
              TensorRole::weight, TensorOwner::attention, 3);
    // `weight_scale_inv` is the DeepSeek/GLM blockwise-multiplier name; it maps
    // onto the SAME role as `weight_scale` (MODELINFO §7).
    expect_id(L + "q_a_proj.weight_scale_inv", TensorComponent::q_a_proj,
              TensorRole::weight_scale, TensorOwner::attention, 3);
    expect_id(L + "q_b_proj.weight", TensorComponent::q_b_proj,
              TensorRole::weight, TensorOwner::attention, 3);
    expect_id(L + "kv_a_proj_with_mqa.weight",
              TensorComponent::kv_a_proj_with_mqa, TensorRole::weight,
              TensorOwner::attention, 3);
    expect_id(L + "kv_a_proj_with_mqa.weight_scale_inv",
              TensorComponent::kv_a_proj_with_mqa, TensorRole::weight_scale,
              TensorOwner::attention, 3);
    expect_id(L + "kv_a_layernorm.weight", TensorComponent::kv_a_norm,
              TensorRole::weight, TensorOwner::attention, 3);
    expect_id(L + "kv_b_proj.weight", TensorComponent::kv_b_proj,
              TensorRole::weight, TensorOwner::attention, 3);
    expect_id(L + "o_proj.weight", TensorComponent::o_proj, TensorRole::weight,
              TensorOwner::attention, 3);
    expect_id(L + "q_a_layernorm.weight", TensorComponent::q_a_norm,
              TensorRole::weight, TensorOwner::attention, 3);
}

TEST(Glm5NextWeightLoadingTensorId, SparseLayerIndexer) {
    const std::string L = std::string(kLm) + "layers.3.self_attn.indexer.";
    expect_id(L + "wq_b.weight", TensorComponent::indexer_wq_b,
              TensorRole::weight, TensorOwner::attention, 3);
    expect_id(L + "wk.weight", TensorComponent::indexer_wk, TensorRole::weight,
              TensorOwner::attention, 3);
    expect_id(L + "k_norm.weight", TensorComponent::indexer_k_norm_weight,
              TensorRole::weight, TensorOwner::attention, 3);
    expect_id(L + "k_norm.bias", TensorComponent::indexer_k_norm_bias,
              TensorRole::bias, TensorOwner::attention, 3);
    expect_id(L + "weights_proj.weight", TensorComponent::indexer_weights_proj,
              TensorRole::weight, TensorOwner::attention, 3);
}

TEST(Glm5NextWeightLoadingTensorId, ModelLevelAndLegacyNamespace) {
    expect_id(std::string(kLm) + "embed_tokens.weight",
              TensorComponent::embedding, TensorRole::weight,
              TensorOwner::model_level, -1);
    expect_id(std::string(kLm) + "norm.weight", TensorComponent::final_norm,
              TensorRole::weight, TensorOwner::model_level, -1);
    expect_id("lm_head.weight", TensorComponent::output_head, TensorRole::weight,
              TensorOwner::model_level, -1);
    // The wrapper collapse must not disturb the legacy (V3.2 / GLM-5.2)
    // namespace, which stays byte-identical in behaviour.
    expect_id("model.layers.5.self_attn.q_a_proj.weight",
              TensorComponent::q_a_proj, TensorRole::weight,
              TensorOwner::attention, 5);
    expect_id("model.embed_tokens.weight", TensorComponent::embedding,
              TensorRole::weight, TensorOwner::model_level, -1);
}

TEST(Glm5NextWeightLoadingTensorId, MtpBlockLayer45) {
    const std::string L = std::string(kLm) + "layers.45.";
    expect_id(L + "eh_proj.weight", TensorComponent::mtp_eh_proj,
              TensorRole::weight, TensorOwner::mtp, 45);
    expect_id(L + "enorm.weight", TensorComponent::mtp_enorm, TensorRole::weight,
              TensorOwner::mtp, 45);
    expect_id(L + "hnorm.weight", TensorComponent::mtp_hnorm, TensorRole::weight,
              TensorOwner::mtp, 45);
    expect_id(L + "shared_head.norm.weight",
              TensorComponent::mtp_shared_head_norm, TensorRole::weight,
              TensorOwner::mtp, 45);
    // The MTP block layer is sparse MLA (MODELINFO §5) — its attention tensors
    // parse as ordinary attention tensors carrying layer index 45.
    expect_id(L + "self_attn.kv_b_proj.weight", TensorComponent::kv_b_proj,
              TensorRole::weight, TensorOwner::attention, 45);
    expect_id(L + "mlp.experts.287.down_proj.weight", TensorComponent::down_proj,
              TensorRole::weight, TensorOwner::routed_expert, 45, 287);
}

TEST(Glm5NextWeightLoadingTensorId, NegativesVisionAndNonsense) {
    // The vision tower is classified and skipped by load_weights (GF3.14
    // deferred) — the NAME PARSER deliberately does not recognize it.
    EXPECT_FALSE(
        parse_hf_name("model.visual.blocks.0.attn.qkv.weight").has_value());
    EXPECT_FALSE(parse_hf_name("model.visual.patch_embed.proj.bias").has_value());
    EXPECT_FALSE(parse_hf_name("model.layers.4.self_attn.nonsense").has_value());
    EXPECT_FALSE(parse_hf_name(std::string(kLm) + "layers.4.self_attn.nonsense")
                     .has_value());
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 2 — the real safetensors index: full coverage + per-layer anatomy
// ═════════════════════════════════════════════════════════════════════════════

namespace {

/// Count the parsed components of one layer's tensors, excluding routed-expert
/// tensors and quantization-scale roles (a `weight_scale_inv` rides with its
/// parent weight, so counting it would double every FP8 component). Both
/// `weight` and `bias` roles count — `indexer.k_norm.bias` is a component of
/// its own.
std::map<TensorComponent, int> layer_component_census(int layer) {
    const std::string prefix =
        std::string(kLm) + "layers." + std::to_string(layer) + ".";
    std::map<TensorComponent, int> census;
    for (const auto& n : real_index_names()) {
        if (n.rfind(prefix, 0) != 0) continue;
        if (n.find(".mlp.experts.") != std::string::npos) continue;
        auto id = parse_hf_name(n);
        if (!id) continue;
        if (id->role != TensorRole::weight && id->role != TensorRole::bias)
            continue;
        ++census[id->component];
    }
    return census;
}

void expect_present_once(const std::map<TensorComponent, int>& census,
                         TensorComponent c) {
    auto it = census.find(c);
    ASSERT_NE(it, census.end()) << "missing component " << tensor_component_name(c);
    EXPECT_EQ(it->second, 1) << tensor_component_name(c);
}

void expect_absent(const std::map<TensorComponent, int>& census,
                   TensorComponent c) {
    EXPECT_EQ(census.count(c), 0u)
        << "unexpected component " << tensor_component_name(c);
}

const TensorComponent kKdaComponents[] = {
    TensorComponent::kda_q_proj,   TensorComponent::kda_k_proj,
    TensorComponent::kda_v_proj,   TensorComponent::kda_b_proj,
    TensorComponent::kda_f_a_proj, TensorComponent::kda_f_b_proj,
    TensorComponent::kda_g_a_proj, TensorComponent::kda_g_b_proj,
    TensorComponent::kda_q_conv1d, TensorComponent::kda_k_conv1d,
    TensorComponent::kda_v_conv1d, TensorComponent::kda_a_log,
    TensorComponent::kda_dt_bias,  TensorComponent::kda_o_norm,
    TensorComponent::o_proj,
};

const TensorComponent kMlaComponents[] = {
    TensorComponent::q_a_proj,           TensorComponent::q_a_norm,
    TensorComponent::q_b_proj,           TensorComponent::kv_a_proj_with_mqa,
    TensorComponent::kv_a_norm,          TensorComponent::kv_b_proj,
    TensorComponent::o_proj,
};

const TensorComponent kIndexerComponents[] = {
    TensorComponent::indexer_wq_b,
    TensorComponent::indexer_wk,
    TensorComponent::indexer_k_norm_weight,
    TensorComponent::indexer_k_norm_bias,
    TensorComponent::indexer_weights_proj,
    TensorComponent::indexer_compressor_wgate,
    TensorComponent::indexer_compressor_ape,
};

const TensorComponent kHcComponents[] = {
    TensorComponent::hc_attn_base, TensorComponent::hc_attn_fn,
    TensorComponent::hc_attn_scale, TensorComponent::hc_ffn_base,
    TensorComponent::hc_ffn_fn, TensorComponent::hc_ffn_scale,
};

}  // namespace

TEST(Glm5NextWeightLoadingIndex, TotalsAndVisionSplit) {
    const auto& names = real_index_names();
    EXPECT_EQ(names.size(), 76108u);
    size_t vision = 0;
    for (const auto& n : names)
        if (n.rfind("model.visual.", 0) == 0) ++vision;
    EXPECT_EQ(vision, 347u);
}

TEST(Glm5NextWeightLoadingIndex, EveryNonVisionNameParses) {
    std::vector<std::string> failures;
    for (const auto& n : real_index_names()) {
        if (n.rfind("model.visual.", 0) == 0) continue;
        if (!parse_hf_name(n)) failures.push_back(n);
    }
    EXPECT_EQ(failures.size(), 0u);
    for (size_t i = 0; i < failures.size() && i < 20; ++i)
        ADD_FAILURE() << "unparsed tensor name: " << failures[i];
}

TEST(Glm5NextWeightLoadingIndex, Layer4IsPureKda) {
    const auto census = layer_component_census(4);
    for (auto c : kKdaComponents) expect_present_once(census, c);
    // ... and carries NOTHING of the MLA / indexer anatomy.
    for (auto c : {TensorComponent::q_a_proj, TensorComponent::q_b_proj,
                   TensorComponent::kv_a_proj_with_mqa,
                   TensorComponent::kv_b_proj})
        expect_absent(census, c);
    for (auto c : kIndexerComponents) expect_absent(census, c);
    // The layer-level pieces every non-MTP layer carries.
    expect_present_once(census, TensorComponent::input_layernorm);
    expect_present_once(census, TensorComponent::post_attention_layernorm);
    for (auto c : kHcComponents) expect_present_once(census, c);
    // Layer 4 is MoE (first_k_dense_replace = 3): router + shared expert.
    expect_present_once(census, TensorComponent::gate_weight);
    for (auto c : {TensorComponent::gate_proj, TensorComponent::up_proj,
                   TensorComponent::down_proj})
        expect_present_once(census, c);  // the shared expert's three
}

TEST(Glm5NextWeightLoadingIndex, Layer3IsSparseMlaWithIndexPool) {
    const auto census = layer_component_census(3);
    for (auto c : kMlaComponents) expect_present_once(census, c);
    for (auto c : kIndexerComponents) expect_present_once(census, c);
    for (auto c : kKdaComponents) {
        if (c == TensorComponent::o_proj) continue;  // shared with MLA
        expect_absent(census, c);
    }
    expect_present_once(census, TensorComponent::input_layernorm);
    expect_present_once(census, TensorComponent::post_attention_layernorm);
    expect_present_once(census, TensorComponent::gate_weight);
    for (auto c : kHcComponents) expect_present_once(census, c);
}

TEST(Glm5NextWeightLoadingIndex, Layer45IsMtpSparseMla) {
    const auto census = layer_component_census(45);
    for (auto c : {TensorComponent::mtp_eh_proj, TensorComponent::mtp_enorm,
                   TensorComponent::mtp_hnorm,
                   TensorComponent::mtp_shared_head_norm})
        expect_present_once(census, c);
    for (auto c : kMlaComponents) expect_present_once(census, c);
    for (auto c : kIndexerComponents) expect_present_once(census, c);
    for (auto c : kKdaComponents) {
        if (c == TensorComponent::o_proj) continue;
        expect_absent(census, c);
    }
    // MODELINFO §3e: the MTP layer runs the NON-mHC path — no hc_* tensors.
    for (auto c : kHcComponents) expect_absent(census, c);
}

TEST(Glm5NextWeightLoadingIndex, Layer3ExpertSweep) {
    const std::string prefix = std::string(kLm) + "layers.3.mlp.experts.";
    std::map<int, int> tensors_per_expert;
    std::map<int, std::set<std::pair<int, int>>> roles_seen;
    for (const auto& n : real_index_names()) {
        if (n.rfind(prefix, 0) != 0) continue;
        auto id = parse_hf_name(n);
        ASSERT_TRUE(id.has_value()) << n;
        EXPECT_EQ(id->owner, TensorOwner::routed_expert) << n;
        EXPECT_EQ(id->layer_idx, 3) << n;
        ++tensors_per_expert[id->expert_idx];
        roles_seen[id->expert_idx].insert(
            {static_cast<int>(id->component), static_cast<int>(id->role)});
    }
    ASSERT_EQ(tensors_per_expert.size(), 288u);
    EXPECT_EQ(tensors_per_expert.begin()->first, 0);
    EXPECT_EQ(tensors_per_expert.rbegin()->first, 287);
    for (const auto& [e, count] : tensors_per_expert) {
        // gate/up/down x {weight, weight_scale_inv}
        EXPECT_EQ(count, 6) << "expert " << e;
        EXPECT_EQ(roles_seen[e].size(), 6u) << "expert " << e;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 3 — QuantSkipList vs the real config.json (the namespace trap)
// ═════════════════════════════════════════════════════════════════════════════

namespace {

const QuantSkipList& real_skip_list() {
    static const QuantSkipList sl = [] {
        auto opt = QuantSkipList::load_from_model_dir(real_model_dir());
        EXPECT_TRUE(opt.has_value());
        return opt.value_or(QuantSkipList::from_entries({}, ""));
    }();
    return sl;
}

}  // namespace

TEST(Glm5NextWeightLoadingSkipList, LoadsRealConfig) {
    const auto& sl = real_skip_list();
    EXPECT_EQ(sl.entry_count(), 1509u);
    EXPECT_EQ(sl.quant_method(), "fp8");
}

TEST(Glm5NextWeightLoadingSkipList, Canonicalize) {
    EXPECT_EQ(QuantSkipList::canonicalize("model.language_model.layers.4.x"),
              "model.layers.4.x");
    EXPECT_EQ(QuantSkipList::canonicalize("model.visual.a"), "visual.a");
    EXPECT_EQ(QuantSkipList::canonicalize("model.visual"), "visual");
    EXPECT_EQ(QuantSkipList::canonicalize("lm_head.weight"), "lm_head.weight");
    // Boundary: `model.visuals` is a different module, not the vision tower.
    EXPECT_EQ(QuantSkipList::canonicalize("model.visuals"), "model.visuals");
}

TEST(Glm5NextWeightLoadingSkipList, SpotListStaysUnquantized) {
    const auto& sl = real_skip_list();
    const std::string P = kLm;
    const std::vector<std::string> keep_bf16 = {
        // whole KDA layers (incl. the bare F32 params)
        P + "layers.4.self_attn.q_proj.weight",
        P + "layers.4.self_attn.A_log",
        P + "layers.4.self_attn.dt_bias",
        // sparse-layer BF16 islands
        P + "layers.3.self_attn.kv_b_proj.weight",
        P + "layers.3.self_attn.indexer.wq_b.weight",
        P + "layers.3.self_attn.indexer.index_kpool_compress_ape",
        P + "layers.3.self_attn.indexer.k_norm.bias",
        // mHC, MTP extras, model level
        P + "layers.3.hc_attn_fn",
        P + "layers.45.eh_proj.weight",
        P + "layers.45.shared_head.norm.weight",
        P + "embed_tokens.weight",
        P + "norm.weight",
        "lm_head.weight",
        // the entire vision tower
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.patch_embed.proj.bias",
        // router + norms
        P + "layers.4.mlp.gate.weight",
        P + "layers.4.mlp.gate.e_score_correction_bias",
        P + "layers.3.input_layernorm.weight",
    };
    for (const auto& n : keep_bf16)
        EXPECT_TRUE(sl.matches(n)) << "expected skip-listed: " << n;
}

TEST(Glm5NextWeightLoadingSkipList, SpotListStaysFp8) {
    const auto& sl = real_skip_list();
    const std::string P = kLm;
    const std::vector<std::string> quantized = {
        P + "layers.3.self_attn.q_a_proj.weight",
        P + "layers.3.self_attn.q_b_proj.weight",
        P + "layers.3.self_attn.kv_a_proj_with_mqa.weight",
        P + "layers.3.self_attn.o_proj.weight",
        P + "layers.3.mlp.experts.0.gate_proj.weight",
        P + "layers.3.mlp.experts.287.down_proj.weight",
        P + "layers.3.mlp.shared_experts.up_proj.weight",
        P + "layers.45.self_attn.o_proj.weight",
    };
    for (const auto& n : quantized)
        EXPECT_FALSE(sl.matches(n)) << "expected FP8 (not skip-listed): " << n;
}

TEST(Glm5NextWeightLoadingSkipList, CensusGolden) {
    // Golden reconciliation over all 76,108 index names:
    //   31 KDA + MoE layers         x 25 = 775  (15 KDA + 2 norms + 6 hc
    //                                            + gate.weight + e_score bias)
    //    3 KDA + dense layers (0-2) x 23 =  69  (same minus the 2 router
    //                                            tensors)
    //   11 sparse layers            x 20 = 220  (kv_b + q_a/kv_a layernorms +
    //                                            7 indexer + 2 norms + 6 hc +
    //                                            gate.weight + e_score bias)
    //    1 MTP layer (45)                =  18  (no hc_*; plus eh_proj/enorm/
    //                                            hnorm/shared_head.norm)
    //    model level (embed, norm, lm_head) =  3
    //    vision tower                       = 347
    //   ------------------------------------------------------------------
    //   775 + 69 + 220 + 18 + 3 + 347        = 1432
    const auto& sl = real_skip_list();
    size_t matched = 0;
    for (const auto& n : real_index_names())
        if (sl.matches(n)) ++matched;
    EXPECT_EQ(matched, 1432u);
    EXPECT_EQ(775u + 69u + 220u + 18u + 3u + 347u, 1432u);
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 4 — end-to-end load_weights() on a synthetic mini-glm5_next FP8
//           checkpoint using the REAL naming scheme
// ═════════════════════════════════════════════════════════════════════════════

namespace {

// Reduced dims chosen so every FP8 blockwise scale shape
// [ceil(N/128), ceil(K/128)] validates against Fp8WeightHandler.
constexpr int64_t kHidden = 256;
constexpr int64_t kHeads = 4;
constexpr int64_t kKdaHeads = 4;
constexpr int64_t kKdaDim = 64;  // kKdaHeads * kKdaDim == kHidden
constexpr int64_t kConvK = 4;
constexpr int64_t kQLora = 128;
constexpr int64_t kKvLora = 128;
constexpr int64_t kQkNope = 64;
constexpr int64_t kQkRope = 0;  // NoPE MLA
constexpr int64_t kVHead = 64;
constexpr int64_t kIdxHeads = 4;
constexpr int64_t kIdxDim = 64;
constexpr int64_t kKpool = 4;
constexpr int64_t kIndexTopk = 16;
constexpr int64_t kHcMult = 4;
constexpr int64_t kVocab = 512;
constexpr int64_t kExperts = 4;
constexpr int64_t kMoeInter = 128;
constexpr int64_t kInter = 256;
constexpr int kLayers = 4;  // 0,1,2 linear_attention; 3 deepseek_sparse
constexpr int kSparseLayer = 3;

struct FixtureTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
};

int64_t ceil128(int64_t v) { return (v + 127) / 128; }

size_t elem_size_of(const std::string& dtype) {
    if (dtype == "F32") return 4;
    if (dtype == "F16" || dtype == "BF16") return 2;
    if (dtype == "F8_E4M3" || dtype == "F8_E5M2" || dtype == "U8") return 1;
    if (dtype == "I64") return 8;
    return 1;
}

size_t byte_size_of(const FixtureTensor& t) {
    size_t numel = 1;
    for (auto d : t.shape) numel *= static_cast<size_t>(d);
    return numel * elem_size_of(t.dtype);
}

/// Deterministic, name-derived payload: byte i of tensor `name` depends only on
/// the name and i, so the checksum check is independent of write order/layout.
uint64_t name_seed(std::string_view name) {
    uint64_t h = 1469598103934665603ull;
    for (char c : name) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ull;
    }
    return h;
}

std::vector<char> pattern_bytes(std::string_view name, size_t n) {
    uint64_t s = name_seed(name);
    std::vector<char> out(n);
    for (size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        out[i] = static_cast<char>((s >> 33) & 0xFF);
    }
    return out;
}

void write_fixture_shard(const fs::path& path,
                         const std::vector<FixtureTensor>& tensors) {
    nlohmann::json header;
    size_t offset = 0;
    for (const auto& t : tensors) {
        const size_t bytes = byte_size_of(t);
        header[t.name] = {
            {"dtype", t.dtype},
            {"shape", t.shape},
            {"data_offsets", {offset, offset + bytes}},
        };
        offset += bytes;
    }
    const std::string header_json = header.dump();
    const uint64_t header_size = header_json.size();

    std::ofstream ofs(path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(&header_size), 8);
    ofs.write(header_json.data(),
              static_cast<std::streamsize>(header_json.size()));
    for (const auto& t : tensors) {
        auto buf = pattern_bytes(t.name, byte_size_of(t));
        ofs.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }
}

class TempDir {
public:
    explicit TempDir(const std::string& tag) {
        path_ = fs::temp_directory_path() /
                ("ls_glm5next_" + tag + "_" + std::to_string(::getpid()));
        std::error_code ec;
        fs::remove_all(path_, ec);
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

/// The mini checkpoint's tensor list, in the REAL glm5_next naming scheme.
/// `poison_kv_b` writes layer-3 kv_b_proj as FP8 (+ a plausible scale) — the
/// checkpoint-vs-skip-list disagreement that must stop the load.
std::vector<FixtureTensor> mini_tensors(bool poison_kv_b) {
    std::vector<FixtureTensor> t;
    const std::string P = kLm;

    auto bf16 = [&](const std::string& n, std::vector<int64_t> s) {
        t.push_back({n, "BF16", std::move(s)});
    };
    auto f32 = [&](const std::string& n, std::vector<int64_t> s) {
        t.push_back({n, "F32", std::move(s)});
    };
    // FP8 weight + its blockwise `weight_scale_inv` multiplier.
    auto fp8 = [&](const std::string& base, int64_t N, int64_t K) {
        t.push_back({base + ".weight", "F8_E4M3", {N, K}});
        t.push_back({base + ".weight_scale_inv", "F32", {ceil128(N), ceil128(K)}});
    };

    // ── model level ──
    bf16(P + "embed_tokens.weight", {kVocab, kHidden});
    bf16(P + "norm.weight", {kHidden});
    bf16("lm_head.weight", {kVocab, kHidden});

    for (int l = 0; l < kLayers; ++l) {
        const std::string LP = P + "layers." + std::to_string(l) + ".";
        const std::string A = LP + "self_attn.";
        bf16(LP + "input_layernorm.weight", {kHidden});
        bf16(LP + "post_attention_layernorm.weight", {kHidden});
        // mHC hyper-connection wrap (bare names, every layer).
        for (const std::string w : {std::string("attn"), std::string("ffn")}) {
            f32(LP + "hc_" + w + "_base", {(2 + kHcMult) * kHcMult});
            bf16(LP + "hc_" + w + "_fn",
                 {(2 + kHcMult) * kHcMult, kHcMult * kHidden});
            f32(LP + "hc_" + w + "_scale", {3});
        }

        if (l != kSparseLayer) {
            // ── KDA linear attention: entirely BF16 (+ F32 A_log/dt_bias) ──
            bf16(A + "q_proj.weight", {kKdaHeads * kKdaDim, kHidden});
            bf16(A + "k_proj.weight", {kKdaHeads * kKdaDim, kHidden});
            bf16(A + "v_proj.weight", {kKdaHeads * kKdaDim, kHidden});
            bf16(A + "b_proj.weight", {kKdaHeads, kHidden});
            bf16(A + "f_a_proj.weight", {kKdaDim, kHidden});
            bf16(A + "f_b_proj.weight", {kKdaHeads * kKdaDim, kKdaDim});
            bf16(A + "g_a_proj.weight", {kKdaDim, kHidden});
            bf16(A + "g_b_proj.weight", {kKdaHeads * kKdaDim, kKdaDim});
            bf16(A + "q_conv1d.weight", {kKdaHeads * kKdaDim, 1, kConvK});
            bf16(A + "k_conv1d.weight", {kKdaHeads * kKdaDim, 1, kConvK});
            bf16(A + "v_conv1d.weight", {kKdaHeads * kKdaDim, 1, kConvK});
            f32(A + "A_log", {kKdaHeads});
            f32(A + "dt_bias", {kKdaHeads * kKdaDim});
            bf16(A + "o_norm.weight", {kKdaDim});
            bf16(A + "o_proj.weight", {kHidden, kKdaHeads * kKdaDim});
            // dense FFN stem (first_k_dense_replace = 3)
            fp8(LP + "mlp.gate_proj", kInter, kHidden);
            fp8(LP + "mlp.up_proj", kInter, kHidden);
            fp8(LP + "mlp.down_proj", kHidden, kInter);
            continue;
        }

        // ── NoPE sparse MLA: mixed precision WITHIN the layer ──
        fp8(A + "q_a_proj", kQLora, kHidden);
        bf16(A + "q_a_layernorm.weight", {kQLora});
        fp8(A + "q_b_proj", kHeads * (kQkNope + kQkRope), kQLora);
        fp8(A + "kv_a_proj_with_mqa", kKvLora + kQkRope, kHidden);
        bf16(A + "kv_a_layernorm.weight", {kKvLora});
        if (poison_kv_b) {
            fp8(A + "kv_b_proj", kHeads * (kQkNope + kVHead), kKvLora);
        } else {
            bf16(A + "kv_b_proj.weight", {kHeads * (kQkNope + kVHead), kKvLora});
        }
        fp8(A + "o_proj", kHidden, kHeads * kVHead);
        // indexer (all BF16), incl. the learned IndexPool compressor
        bf16(A + "indexer.wq_b.weight", {kIdxHeads * kIdxDim, kQLora});
        bf16(A + "indexer.wk.weight", {kIdxDim, kHidden});
        bf16(A + "indexer.k_norm.weight", {kIdxDim});
        bf16(A + "indexer.k_norm.bias", {kIdxDim});
        bf16(A + "indexer.weights_proj.weight", {kIdxHeads, kHidden});
        bf16(A + "indexer.index_kpool_compress_gate", {kIdxDim, kHidden});
        bf16(A + "indexer.index_kpool_compress_ape", {kKpool, kIdxDim});
        // MoE block
        bf16(LP + "mlp.gate.weight", {kExperts, kHidden});
        f32(LP + "mlp.gate.e_score_correction_bias", {kExperts});
        fp8(LP + "mlp.shared_experts.gate_proj", kMoeInter, kHidden);
        fp8(LP + "mlp.shared_experts.up_proj", kMoeInter, kHidden);
        fp8(LP + "mlp.shared_experts.down_proj", kHidden, kMoeInter);
        for (int e = 0; e < kExperts; ++e) {
            const std::string EP = LP + "mlp.experts." + std::to_string(e) + ".";
            fp8(EP + "gate_proj", kMoeInter, kHidden);
            fp8(EP + "up_proj", kMoeInter, kHidden);
            fp8(EP + "down_proj", kHidden, kMoeInter);
        }
    }

    // ── vision decoys: recognized and deliberately skipped (GF3.14) ──
    bf16("model.visual.patch_embed.proj.weight", {8, 8});
    bf16("model.visual.blocks.0.attn.qkv.weight", {8, 8});
    bf16("model.visual.post_layernorm.weight", {8});
    return t;
}

/// `modules_to_not_convert` in the MODULE namespace of the loaded HF model —
/// per-module granularity, exactly as the real config.json does it. `prefix` is
/// "model." for the true namespace and "foo." to reproduce the namespace trap.
std::vector<std::string> mini_skip_entries(const std::string& prefix) {
    const bool real_ns = (prefix == "model.");
    std::vector<std::string> e;
    for (int l = 0; l < kLayers; ++l) {
        const std::string L = prefix + "layers." + std::to_string(l) + ".";
        e.push_back(L + "input_layernorm");
        e.push_back(L + "post_attention_layernorm");
        for (const std::string w : {std::string("attn"), std::string("ffn")}) {
            e.push_back(L + "hc_" + w + "_base");
            e.push_back(L + "hc_" + w + "_fn");
            e.push_back(L + "hc_" + w + "_scale");
        }
        const std::string A = L + "self_attn.";
        if (l != kSparseLayer) {
            for (const char* leaf :
                 {"q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "f_b_proj",
                  "g_a_proj", "g_b_proj", "q_conv1d", "k_conv1d", "v_conv1d",
                  "A_log", "dt_bias", "o_norm", "o_proj"})
                e.push_back(A + leaf);
        } else {
            e.push_back(A + "kv_b_proj");
            e.push_back(A + "q_a_layernorm");
            e.push_back(A + "kv_a_layernorm");
            e.push_back(A + "indexer");  // subtree entry: the whole indexer
            e.push_back(L + "mlp.gate");
            e.push_back(L + "mlp.gate.e_score_correction_bias");
        }
    }
    e.push_back(prefix + "embed_tokens");
    e.push_back(prefix + "norm");
    // Module-class catch-alls (dotless in the real file). Under the trap prefix
    // they become dotted, so they cannot match either.
    e.push_back(real_ns ? "lm_head" : prefix + "lm_head");
    e.push_back(real_ns ? "visual" : prefix + "visual");
    e.push_back(real_ns ? "dt_bias" : prefix + "dt_bias");
    e.push_back(real_ns ? "weights_proj" : prefix + "weights_proj");
    return e;
}

void write_checkpoint(const fs::path& dir,
                      const std::vector<FixtureTensor>& tensors,
                      const std::string& skip_prefix) {
    // Two shards, so the load exercises the index.
    const size_t half = tensors.size() / 2;
    const std::vector<FixtureTensor> s1(tensors.begin(), tensors.begin() + half);
    const std::vector<FixtureTensor> s2(tensors.begin() + half, tensors.end());
    write_fixture_shard(dir / "model-00001-of-00002.safetensors", s1);
    write_fixture_shard(dir / "model-00002-of-00002.safetensors", s2);

    nlohmann::json idx;
    idx["metadata"] = nlohmann::json::object();
    nlohmann::json wm;
    for (const auto& t : s1) wm[t.name] = "model-00001-of-00002.safetensors";
    for (const auto& t : s2) wm[t.name] = "model-00002-of-00002.safetensors";
    idx["weight_map"] = wm;
    std::ofstream(dir / "model.safetensors.index.json") << idx.dump();

    // The CHECKPOINT's own config.json — only quantization_config matters here.
    nlohmann::json cfg;
    cfg["model_type"] = "glm5_next";
    cfg["quantization_config"] = {
        {"quant_method", "fp8"},
        {"fmt", "e4m3"},
        {"activation_scheme", "dynamic"},
        {"weight_block_size", {128, 128}},
        {"modules_to_not_convert", mini_skip_entries(skip_prefix)},
    };
    std::ofstream(dir / "config.json") << cfg.dump();
}

/// The engine-side config for the mini checkpoint (the glm53_flash_config()
/// fixture pattern, shrunk to the fixture's dims).
layerstorm::config::Config mini_engine_config(const fs::path& weights_dir) {
    nlohmann::json layer_types = nlohmann::json::array();
    for (int l = 0; l < kLayers; ++l)
        layer_types.push_back(l == kSparseLayer ? "deepseek_sparse_attention"
                                                : "linear_attention");
    nlohmann::json j = {
        {"model",
         {{"architecture", "glm5_next"},
          {"weights_path", weights_dir.string()},
          {"weights_format", "safetensors"},
          {"num_hidden_layers", kLayers},
          {"hidden_size", kHidden},
          {"num_attention_heads", kHeads},
          {"num_key_value_heads", kHeads},
          {"intermediate_size", kInter},
          {"n_routed_experts", kExperts},
          {"n_shared_experts", 1},
          {"num_experts_per_tok", 2},
          {"n_group", 1},
          {"topk_group", 1},
          {"vocab_size", kVocab},
          {"max_position_embeddings", 4096},
          {"kv_lora_rank", kKvLora},
          {"q_lora_rank", kQLora},
          {"qk_rope_head_dim", kQkRope},
          {"qk_nope_head_dim", kQkNope},
          {"v_head_dim", kVHead},
          {"first_k_dense_replace", kSparseLayer},
          {"moe_layer_freq", 1},
          {"index_topk", kIndexTopk},
          {"index_n_heads", kIdxHeads},
          {"index_head_dim", kIdxDim},
          {"index_kpool", kKpool},
          {"index_kpool_compress", true},
          {"index_kpool_always_select_tail", true},
          {"mla_use_nope", true},
          {"layer_types", layer_types},
          {"linear_attn_config",
           {{"num_heads", kKdaHeads},
            {"head_dim", kKdaDim},
            {"short_conv_kernel_size", kConvK},
            {"gate_lower_bound", -5.0}}},
          {"hc_mult", kHcMult},
          {"hc_sinkhorn_iters", 20},
          {"hc_eps", 1e-6},
          {"swiglu_limit", 10.0},
          {"num_nextn_predict_layers", 0},
          {"rms_norm_eps", 1e-5},
          {"routed_scaling_factor", 2.5},
          {"moe_intermediate_size", kMoeInter}}},
        {"quantization",
         {{"weights", "fp8_e4m3"},
          {"attention_compute", "fp8_e4m3"},
          {"kv_cache", "fp8_e4m3"},
          {"gating_compute", "fp32"}}},
        {"hardware",
         {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
          {"system_ram_gb", 256}}},
    };
    return layerstorm::config::parse_config(j);
}

/// Every bundle of a layer, flattened (attention + indexer + norms + gating +
/// shared expert + dense FFN + routed experts).
std::vector<const WeightBundle*> all_bundles(
    const LoadedModel::LayerWeights& layer) {
    std::vector<const WeightBundle*> out;
    auto add = [&](const std::vector<WeightBundle>& v) {
        for (const auto& b : v) out.push_back(&b);
    };
    add(layer.attention);
    add(layer.indexer);
    add(layer.norms);
    add(layer.gating);
    add(layer.shared_expert);
    add(layer.dense_ffn);
    for (const auto& e : layer.routed_experts) add(e);
    return out;
}

const WeightBundle* find_bundle(const LoadedModel::LayerWeights& layer,
                                TensorComponent c, TensorOwner owner,
                                int expert_idx) {
    for (const auto* b : all_bundles(layer))
        if (b->id.component == c && b->id.owner == owner &&
            b->id.expert_idx == expert_idx)
            return b;
    return nullptr;
}

bool has_component(const std::vector<WeightBundle>& v, TensorComponent c) {
    for (const auto& b : v)
        if (b.id.component == c) return true;
    return false;
}

/// One bundle per logical weight: vision tensors are skipped entirely, and each
/// `weight_scale_inv` folds into its parent weight's bundle.
size_t expected_bundle_count(const std::vector<FixtureTensor>& tensors) {
    constexpr std::string_view kScale = ".weight_scale_inv";
    size_t n = 0;
    for (const auto& t : tensors) {
        if (t.name.rfind("model.visual.", 0) == 0) continue;
        if (t.name.size() > kScale.size() &&
            std::string_view(t.name).substr(t.name.size() - kScale.size()) ==
                kScale)
            continue;
        ++n;
    }
    return n;
}

}  // namespace

class Glm5NextWeightLoadingE2E : public ::testing::Test {};

TEST_F(Glm5NextWeightLoadingE2E, LoadsHybridStackWithSkipListEnforced) {
    TempDir tmp("load");
    const auto tensors = mini_tensors(/*poison_kv_b=*/false);
    write_checkpoint(tmp.path(), tensors, "model.");

    auto cfg = mini_engine_config(tmp.path());
    MConfig mcfg(cfg);
    ASSERT_TRUE(mcfg.is_glm5_next());
    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    LoadedModel model;
    ASSERT_NO_THROW(model = load_weights(cfg, mcfg, registry));

    ASSERT_EQ(model.layers.size(), static_cast<size_t>(kLayers));
    EXPECT_TRUE(model.embedding.has_value());
    EXPECT_TRUE(model.output_head.has_value());
    EXPECT_TRUE(model.final_norm.has_value());

    // ── layer 0 (KDA linear attention) ──
    const auto& l0 = model.layers[0];
    for (auto c : kKdaComponents)
        EXPECT_TRUE(has_component(l0.attention, c))
            << "layer 0 missing " << tensor_component_name(c);
    EXPECT_TRUE(l0.indexer.empty());
    EXPECT_FALSE(l0.dense_ffn.empty());

    // THE skip-list assertion (PLAN GF3.3): inside an FP8 checkpoint the whole
    // KDA layer stays BF16 because modules_to_not_convert says so.
    const auto* q =
        find_bundle(l0, TensorComponent::kda_q_proj, TensorOwner::attention, -1);
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(q->weight.dtype, SafetensorsDtype::BF16);
    EXPECT_EQ(q->find_aux(TensorRole::weight_scale), nullptr);

    const auto* a_log =
        find_bundle(l0, TensorComponent::kda_a_log, TensorOwner::attention, -1);
    ASSERT_NE(a_log, nullptr);
    EXPECT_EQ(a_log->weight.dtype, SafetensorsDtype::F32);

    const auto* conv = find_bundle(l0, TensorComponent::kda_q_conv1d,
                                   TensorOwner::attention, -1);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->weight.shape,
              (std::vector<int64_t>{kKdaHeads * kKdaDim, 1, kConvK}));

    // ── layer 3 (NoPE sparse MLA + IndexPool indexer) ──
    const auto& l3 = model.layers[kSparseLayer];
    const auto* q_a =
        find_bundle(l3, TensorComponent::q_a_proj, TensorOwner::attention, -1);
    ASSERT_NE(q_a, nullptr);
    EXPECT_EQ(q_a->weight.dtype, SafetensorsDtype::F8_E4M3);
    const auto* q_a_scale = q_a->find_aux(TensorRole::weight_scale);
    ASSERT_NE(q_a_scale, nullptr);
    EXPECT_EQ(q_a_scale->dtype, SafetensorsDtype::F32);
    EXPECT_EQ(q_a_scale->shape,
              (std::vector<int64_t>{ceil128(kQLora), ceil128(kHidden)}));

    const auto* kv_b =
        find_bundle(l3, TensorComponent::kv_b_proj, TensorOwner::attention, -1);
    ASSERT_NE(kv_b, nullptr);
    EXPECT_EQ(kv_b->weight.dtype, SafetensorsDtype::BF16);
    EXPECT_EQ(kv_b->find_aux(TensorRole::weight_scale), nullptr);

    EXPECT_TRUE(
        has_component(l3.indexer, TensorComponent::indexer_compressor_wgate));
    EXPECT_TRUE(
        has_component(l3.indexer, TensorComponent::indexer_compressor_ape));
    EXPECT_TRUE(has_component(l3.indexer, TensorComponent::indexer_wq_b));
    EXPECT_TRUE(has_component(l3.indexer, TensorComponent::indexer_k_norm_bias));
    EXPECT_FALSE(l3.gating.empty());
    EXPECT_EQ(l3.routed_experts.size(), static_cast<size_t>(kExperts));

    // ── vision decoys were skipped, not bundled ──
    EXPECT_EQ(static_cast<size_t>(model.total_tensors_loaded),
              expected_bundle_count(tensors));
    for (const auto& layer : model.layers)
        for (const auto* b : all_bundles(layer))
            EXPECT_NE(b->weight.shape, (std::vector<int64_t>{8, 8}))
                << "a vision decoy leaked into layer " << layer.layer_idx;
    EXPECT_EQ(model.shards.size(), 2u);
}

TEST_F(Glm5NextWeightLoadingE2E, ByteForByteChecksumOfLayers0And3) {
    TempDir tmp("checksum");
    const auto tensors = mini_tensors(/*poison_kv_b=*/false);
    write_checkpoint(tmp.path(), tensors, "model.");

    auto cfg = mini_engine_config(tmp.path());
    MConfig mcfg(cfg);
    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);
    LoadedModel model = load_weights(cfg, mcfg, registry);

    int checked = 0;
    for (const auto& t : tensors) {
        auto id = parse_hf_name(t.name);
        if (!id) continue;
        if (id->layer_idx != 0 && id->layer_idx != kSparseLayer) continue;
        ASSERT_LT(id->layer_idx, static_cast<int>(model.layers.size()));
        const auto& layer = model.layers[id->layer_idx];
        const auto* bundle =
            find_bundle(layer, id->component, id->owner, id->expert_idx);
        ASSERT_NE(bundle, nullptr)
            << "no bundle for " << t.name << " ("
            << tensor_component_name(id->component) << ")";

        const RawTensor* raw = nullptr;
        if (id->role == TensorRole::weight || bundle->id.role == id->role) {
            // Bias-only bundles (indexer k_norm.bias, gate e_score bias) keep
            // the bias in `weight` with the bundle's role set to bias.
            raw = &bundle->weight;
        } else {
            raw = bundle->find_aux(id->role);
        }
        ASSERT_NE(raw, nullptr) << "no tensor for the role of " << t.name;

        EXPECT_EQ(raw->shape, t.shape) << t.name;
        const auto want_dtype = parse_dtype(t.dtype);
        ASSERT_TRUE(want_dtype.has_value());
        EXPECT_EQ(raw->dtype, *want_dtype) << t.name;

        const auto expected = pattern_bytes(t.name, byte_size_of(t));
        ASSERT_EQ(raw->data.size(), expected.size()) << t.name;
        EXPECT_EQ(
            std::memcmp(raw->data.data(), expected.data(), expected.size()), 0)
            << "payload mismatch for " << t.name;
        ++checked;
    }
    // layer 0 (KDA + dense FFN + norms + mHC) and layer 3 (MLA + indexer +
    // gating + shared expert + 4 routed experts), weights AND scales.
    EXPECT_EQ(checked, 87);
}

TEST_F(Glm5NextWeightLoadingE2E, PoisonedPrecisionKvBProjIsLoadStopping) {
    TempDir tmp("poison");
    const auto tensors = mini_tensors(/*poison_kv_b=*/true);
    write_checkpoint(tmp.path(), tensors, "model.");

    auto cfg = mini_engine_config(tmp.path());
    MConfig mcfg(cfg);
    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    try {
        load_weights(cfg, mcfg, registry);
        FAIL() << "expected load_weights to throw on an FP8 kv_b_proj that the "
                  "skip list keeps BF16";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("modules_to_not_convert"), std::string::npos)
            << "message was: " << msg;
        EXPECT_NE(msg.find("kv_b_proj"), std::string::npos)
            << "message was: " << msg;
    }
}

TEST_F(Glm5NextWeightLoadingE2E, NamespaceTrapZeroMatchIsLoadStopping) {
    TempDir tmp("trap");
    const auto tensors = mini_tensors(/*poison_kv_b=*/false);
    // Same checkpoint; skip list written in a namespace that matches nothing.
    write_checkpoint(tmp.path(), tensors, "foo.");

    auto cfg = mini_engine_config(tmp.path());
    MConfig mcfg(cfg);
    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    try {
        load_weights(cfg, mcfg, registry);
        FAIL() << "expected load_weights to refuse a skip list that matches no "
                  "tensor at all";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("matched ZERO"), std::string::npos)
            << "message was: " << msg;
    }
}
