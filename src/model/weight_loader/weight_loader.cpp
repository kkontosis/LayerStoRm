#include "model/weight_loader/weight_loader.h"

#include "core/parallel_for.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <map>
#include <optional>
#include <regex>
#include <stdexcept>
#include <tuple>

#include <spdlog/spdlog.h>

#include "config/config_parser.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/fp8.h"
#include "model/quantization/gguf_kquant.h"
#include "model/weight_loader/gguf_reader.h"
#include "model/weight_loader/nvfp4_sfb_reformat.h"

namespace layerstorm::model {

// ── Accumulator ──────────────────────────────────────────────────────────────
// Groups raw tensors by their logical identity before assembling WeightBundles.

namespace {

// Key for grouping tensors: same (component, owner, layer, expert) = same logical weight.
struct GroupKey {
    TensorComponent component;
    TensorOwner owner;
    int layer_idx;
    int expert_idx;

    auto operator<=>(const GroupKey&) const = default;
};

struct AccumulatedTensor {
    TensorRole role;
    RawTensor raw;
};

using TensorAccumulator = std::map<GroupKey, std::vector<AccumulatedTensor>>;

GroupKey key_from_id(const TensorId& id) {
    return {id.component, id.owner, id.layer_idx, id.expert_idx};
}

// Assemble a WeightBundle from accumulated tensors for one group.
// Returns empty string on success or error message.
std::pair<WeightBundle, std::string> assemble_bundle(
    const GroupKey& key,
    std::vector<AccumulatedTensor>& tensors,
    const ModelConfig& model_cfg) {

    WeightBundle bundle;
    bundle.id = TensorId{key.component, TensorRole::weight, key.owner, key.layer_idx, key.expert_idx};

    // Find the main weight
    bool found_weight = false;
    for (auto& t : tensors) {
        if (t.role == TensorRole::weight) {
            bundle.weight = std::move(t.raw);
            found_weight = true;
        } else {
            bundle.aux.emplace_back(t.role, std::move(t.raw));
        }
    }

    if (!found_weight) {
        // Some tensors are bias-only (e.g. gate_e_score_correction_bias).
        // For these, the "weight" is the bias itself and role was already set.
        // Check if there's a single bias tensor.
        if (tensors.size() == 1 && tensors[0].role == TensorRole::bias) {
            bundle.id.role = TensorRole::bias;
            // The loop above moved the bias into bundle.aux — move it to weight
            // and clear aux so the bundle has no phantom tensors.
            bundle.weight = std::move(bundle.aux[0].second);
            bundle.aux.clear();
            return {std::move(bundle), {}};
        }
        return {std::move(bundle),
                std::format("No weight tensor for {} layer={} expert={}",
                            tensor_component_name(key.component), key.layer_idx, key.expert_idx)};
    }

    // Validate via the appropriate handler
    auto& handler = handler_for_bundle(bundle);
    auto err = handler.validate(bundle, model_cfg);
    if (!err.empty()) {
        return {std::move(bundle),
                std::format("{} for {} layer={} expert={}: {}",
                            handler.name(), tensor_component_name(key.component),
                            key.layer_idx, key.expert_idx, err)};
    }

    return {std::move(bundle), {}};
}

// Place a bundle into the appropriate slot in LoadedModel.
void place_bundle(LoadedModel& model, WeightBundle&& bundle,
                  const ModelConfig& model_cfg) {
    auto& id = bundle.id;

    // Model-level tensors
    if (id.owner == TensorOwner::model_level) {
        switch (id.component) {
            case TensorComponent::embedding:
                model.embedding = std::move(bundle);
                return;
            case TensorComponent::output_head:
                model.output_head = std::move(bundle);
                return;
            case TensorComponent::final_norm:
                model.final_norm = std::move(bundle);
                return;
            case TensorComponent::output_hc_fn:
            case TensorComponent::output_hc_base:
            case TensorComponent::output_hc_scale:
                model.output_hc.push_back(std::move(bundle));
                return;
            default:
                break;
        }
        return;
    }

    // MTP tensors
    if (id.owner == TensorOwner::mtp) {
        if (!model.mtp) {
            model.mtp = LoadedModel::MtpWeights{};
            model.mtp->base_layer_idx = id.layer_idx;
        }
        model.mtp->tensors.push_back(std::move(bundle));
        return;
    }

    // Determine if this is a regular layer or an MTP block layer.
    // MTP block layers have layer_idx >= num_hidden_layers and reuse
    // attention/ffn patterns but belong to the MTP block.
    int num_hidden = model_cfg.raw().num_hidden_layers;
    bool is_mtp_block = (id.layer_idx >= num_hidden) && (id.owner != TensorOwner::mtp);

    if (is_mtp_block) {
        if (!model.mtp) {
            model.mtp = LoadedModel::MtpWeights{};
            model.mtp->base_layer_idx = id.layer_idx;
        }
        // Find or create the block layer
        int block_idx = id.layer_idx - num_hidden;
        while (static_cast<int>(model.mtp->block_layers.size()) <= block_idx) {
            model.mtp->block_layers.push_back({});
            model.mtp->block_layers.back().layer_idx =
                num_hidden + static_cast<int>(model.mtp->block_layers.size()) - 1;
        }
        auto& layer = model.mtp->block_layers[block_idx];
        // Place into the block layer using the same logic as regular layers
        // (fall through to layer placement below with a reference to the block layer)
        auto* target = &layer;

        switch (id.owner) {
            case TensorOwner::attention:
                if (id.component == TensorComponent::input_layernorm ||
                    id.component == TensorComponent::post_attention_layernorm) {
                    target->norms.push_back(std::move(bundle));
                } else if (id.component >= TensorComponent::indexer_wq_b &&
                           id.component <= TensorComponent::indexer_compressor_norm) {
                    target->indexer.push_back(std::move(bundle));
                } else {
                    target->attention.push_back(std::move(bundle));
                }
                return;
            case TensorOwner::gating:
                target->gating.push_back(std::move(bundle));
                return;
            case TensorOwner::routed_expert: {
                int eidx = id.expert_idx;
                if (eidx < 0) return;
                while (static_cast<int>(target->routed_experts.size()) <= eidx) {
                    target->routed_experts.emplace_back();
                }
                target->routed_experts[eidx].push_back(std::move(bundle));
                return;
            }
            case TensorOwner::shared_expert:
                target->shared_expert.push_back(std::move(bundle));
                return;
            case TensorOwner::dense_ffn:
                target->dense_ffn.push_back(std::move(bundle));
                return;
            default:
                return;
        }
    }

    // Regular layer
    if (id.layer_idx < 0 || id.layer_idx >= static_cast<int>(model.layers.size())) return;
    auto& layer = model.layers[id.layer_idx];

    switch (id.owner) {
        case TensorOwner::attention:
            if (id.component == TensorComponent::input_layernorm ||
                id.component == TensorComponent::post_attention_layernorm) {
                layer.norms.push_back(std::move(bundle));
            } else if (id.component >= TensorComponent::indexer_wq_b &&
                       id.component <= TensorComponent::indexer_compressor_norm) {
                layer.indexer.push_back(std::move(bundle));
            } else {
                layer.attention.push_back(std::move(bundle));
            }
            break;
        case TensorOwner::gating:
            layer.gating.push_back(std::move(bundle));
            break;
        case TensorOwner::routed_expert: {
            int eidx = id.expert_idx;
            if (eidx < 0) break;
            while (static_cast<int>(layer.routed_experts.size()) <= eidx) {
                layer.routed_experts.emplace_back();
            }
            layer.routed_experts[eidx].push_back(std::move(bundle));
            break;
        }
        case TensorOwner::shared_expert:
            layer.shared_expert.push_back(std::move(bundle));
            break;
        case TensorOwner::dense_ffn:
            layer.dense_ffn.push_back(std::move(bundle));
            break;
        default:
            break;
    }
}

// Validate that all expected tensors are present.
std::vector<std::string> validate_completeness(const LoadedModel& model,
                                               const ModelConfig& model_cfg,
                                               const LayerRegistry& registry,
                                               bool skip_routed_experts) {
    std::vector<std::string> errors;

    if (!model.embedding) errors.push_back("Missing embedding weight");
    if (!model.output_head) errors.push_back("Missing output head (lm_head) weight");
    if (!model.final_norm) errors.push_back("Missing final norm weight");

    int num_layers = model_cfg.raw().num_hidden_layers;
    if (static_cast<int>(model.layers.size()) != num_layers) {
        errors.push_back(std::format("Expected {} layers, got {}",
                                     num_layers, model.layers.size()));
    }

    for (int l = 0; l < num_layers && l < static_cast<int>(model.layers.size()); ++l) {
        auto& layer = model.layers[l];

        // Every layer needs attention projections and norms
        if (layer.attention.empty()) {
            errors.push_back(std::format("Layer {} missing attention weights", l));
        }
        if (layer.norms.empty()) {
            errors.push_back(std::format("Layer {} missing layer norms", l));
        }

        bool is_moe = model_cfg.is_moe_layer(l);
        if (is_moe) {
            if (layer.gating.empty()) {
                errors.push_back(std::format("MoE layer {} missing gating weights", l));
            }
            // WP-6: Skip routed expert count check when experts loaded from
            // prepacked source instead of safetensors.
            if (!skip_routed_experts) {
                int n_experts = model_cfg.raw().n_routed_experts;
                if (static_cast<int>(layer.routed_experts.size()) < n_experts) {
                    errors.push_back(std::format(
                        "MoE layer {} has {} experts, expected {}",
                        l, layer.routed_experts.size(), n_experts));
                }
            }
        } else {
            if (layer.dense_ffn.empty()) {
                errors.push_back(std::format("Dense layer {} missing FFN weights", l));
            }
        }
    }

    if (model_cfg.has_mtp() && !model.mtp) {
        if (model_cfg.is_v4()) {
            // The Unsloth V4-Flash GGUF strips the MTP/nextn layers despite
            // num_nextn_predict_layers=1 in the HF config (verified census —
            // DS4_DOSSIER.md §0.1). Skip gracefully: the embedded speculator,
            // when wanted, is sourced from the official safetensors release
            // (ticket J), never from this artifact.
            spdlog::warn(
                "V4 model config declares num_nextn_predict_layers={} but the "
                "weights carry no nextn/MTP tensors — continuing without MTP "
                "(embedded speculation unavailable from this artifact)",
                model_cfg.raw().num_nextn_predict_layers);
        } else {
            errors.push_back("Model has MTP but no MTP weights found");
        }
    }

    return errors;
}

}  // namespace

// ── Expert packing (external linkage) ──────────────────────────────────────
// Used by the lazy-pack path (ensure_expert_packed) and
// the offline pre-processor (WP-2 expert_prepacker).

// Pack FP8 routed expert bundles: linearize [weight | blockwise_scale] per projection
// into one contiguous buffer, matching ExpertCache slot layout for resolve_host_source().
void pack_fp8_expert(std::vector<WeightBundle>& bundles, const ExpertShape& shape) {
    // Find gate/up/down bundles by component.
    WeightBundle* gate = nullptr;
    WeightBundle* up = nullptr;
    WeightBundle* down = nullptr;
    for (auto& b : bundles) {
        switch (b.id.component) {
            case TensorComponent::gate_proj: gate = &b; break;
            case TensorComponent::up_proj:   up   = &b; break;
            case TensorComponent::down_proj:  down = &b; break;
            default: break;
        }
    }
    if (!gate || !up || !down) {
        spdlog::error("pack_fp8_expert: missing gate/up/down bundle — "
                      "gate={} up={} down={}",
                      static_cast<bool>(gate), static_cast<bool>(up),
                      static_cast<bool>(down));
        return;
    }

    // Compute scale bytes per projection.
    auto scale_bytes = [&](Projection proj) -> int64_t {
        int64_t N, K;
        switch (proj) {
            case Projection::gate: [[fallthrough]];
            case Projection::up:
                N = shape.intermediate_size;
                K = shape.hidden_size;
                break;
            case Projection::down:
                N = shape.hidden_size;
                K = shape.intermediate_size;
                break;
        }
        int64_t nb = (N + fp8::kBlockScaleTile - 1) / fp8::kBlockScaleTile;
        int64_t kb = (K + fp8::kBlockScaleTile - 1) / fp8::kBlockScaleTile;
        return nb * kb * static_cast<int64_t>(sizeof(float));
    };

    int64_t gate_wb = static_cast<int64_t>(gate->weight.data.size());
    int64_t up_wb   = static_cast<int64_t>(up->weight.data.size());
    int64_t down_wb = static_cast<int64_t>(down->weight.data.size());

    int64_t gate_sb = scale_bytes(Projection::gate);
    int64_t up_sb   = scale_bytes(Projection::up);
    int64_t down_sb = scale_bytes(Projection::down);

    int64_t total = (gate_wb + gate_sb) + (up_wb + up_sb) + (down_wb + down_sb);

    // Check if mmap already has the correct [weight|scale] layout contiguously.
    auto is_adjacent = [](const WeightBundle& b, int64_t sb) -> bool {
        const auto* ws = b.find_aux(TensorRole::weight_scale);
        if (!ws || static_cast<int64_t>(ws->data.size()) != sb) return false;
        return b.weight.data.data() + b.weight.data.size() == ws->data.data();
    };

    const auto* gate_ws = gate->find_aux(TensorRole::weight_scale);
    const auto* up_ws   = up->find_aux(TensorRole::weight_scale);

    bool already_contiguous =
        is_adjacent(*gate, gate_sb) && is_adjacent(*up, up_sb) && is_adjacent(*down, down_sb) &&
        gate_ws && up_ws &&
        gate_ws->data.data() + gate_ws->data.size() == up->weight.data.data() &&
        up_ws->data.data() + up_ws->data.size() == down->weight.data.data();

    if (already_contiguous) {
        // Zero-copy: mmap already matches slot layout.
        bundles[0].packed_slot = std::span<const std::byte>(
            gate->weight.data.data(), static_cast<size_t>(total));
        return;
    }

    // Allocate and pack into contiguous buffer.
    auto buf = std::make_shared<std::vector<std::byte>>(static_cast<size_t>(total));
    auto* dst = buf->data();
    int64_t offset = 0;

    auto pack_proj = [&](WeightBundle& b, int64_t wb, int64_t sb) {
        std::memcpy(dst + offset, b.weight.data.data(), static_cast<size_t>(wb));
        offset += wb;

        const auto* ws = b.find_aux(TensorRole::weight_scale);
        if (ws && static_cast<int64_t>(ws->data.size()) == sb) {
            std::memcpy(dst + offset, ws->data.data(), static_cast<size_t>(sb));
        } else {
            // No scale tensor — fill with 1.0f (identity scale).
            auto* scale_f = reinterpret_cast<float*>(dst + offset);
            int64_t n_scales = sb / static_cast<int64_t>(sizeof(float));
            for (int64_t i = 0; i < n_scales; ++i) scale_f[i] = 1.0f;
        }
        offset += sb;
    };

    pack_proj(*gate, gate_wb, gate_sb);
    pack_proj(*up,   up_wb,   up_sb);
    pack_proj(*down, down_wb, down_sb);

    // Expose via the first bundle (resolve_host_source reads bundles[0].packed_slot).
    bundles[0].owned_buf = buf;
    bundles[0].packed_slot = std::span<const std::byte>(buf->data(), buf->size());
}

// ── GGUF expert packing ─────────────────────────────────────────────────────
// GGUF k-quant blocks are copied verbatim; the slot is gate | up | down with no
// scale region (k-quant scales/mins pack inside each super-block). The
// per-projection byte sizes come from each bundle's OWN gguf_type (GG-10):
// per-layer mixed "XL" GGUFs vary the routed k-quant types across layers, so a
// global triple would mis-size most layers.

bool pack_gguf_expert(std::vector<WeightBundle>& bundles, const ExpertShape& shape,
                      const GgufModelExpertTypes* expected_types) {
    WeightBundle* gate = nullptr;
    WeightBundle* up = nullptr;
    WeightBundle* down = nullptr;
    for (auto& b : bundles) {
        switch (b.id.component) {
            case TensorComponent::gate_proj: gate = &b; break;
            case TensorComponent::up_proj:   up   = &b; break;
            case TensorComponent::down_proj:  down = &b; break;
            default: break;
        }
    }
    if (!gate || !up || !down) {
        spdlog::error("pack_gguf_expert: missing gate/up/down bundle — "
                      "gate={} up={} down={}",
                      static_cast<bool>(gate), static_cast<bool>(up),
                      static_cast<bool>(down));
        return false;
    }
    if (!gate->weight.gguf_type || !up->weight.gguf_type ||
        !down->weight.gguf_type) {
        spdlog::error("pack_gguf_expert: bundle missing gguf_type — "
                      "gate={} up={} down={}",
                      gate->weight.gguf_type.has_value(),
                      up->weight.gguf_type.has_value(),
                      down->weight.gguf_type.has_value());
        return false;
    }

    const GgufKQuantType gate_t = *gate->weight.gguf_type;
    const GgufKQuantType up_t   = *up->weight.gguf_type;
    const GgufKQuantType down_t = *down->weight.gguf_type;

    // Strict per-layer validation (GG-10): the caller's per-layer type table
    // must match the bundles' own types (catches cross-expert non-uniformity
    // within a layer and stale type tables).
    if (expected_types &&
        (gate_t != expected_types->gate || up_t != expected_types->up ||
         down_t != expected_types->down)) {
        spdlog::error("pack_gguf_expert: bundle k-quant types ({}/{}/{}) != "
                      "expected layer types ({}/{}/{})",
                      gguf::type_name(gate_t), gguf::type_name(up_t),
                      gguf::type_name(down_t),
                      gguf::type_name(expected_types->gate),
                      gguf::type_name(expected_types->up),
                      gguf::type_name(expected_types->down));
        return false;
    }

    // Per-projection sizes at THIS expert's own k-quant types.
    const GgufQuantInterface quant = make_gguf_quant(gate_t, up_t, down_t);
    const int64_t gate_b = quant.bytes_per_projection(shape, Projection::gate);
    const int64_t up_b   = quant.bytes_per_projection(shape, Projection::up);
    const int64_t down_b = quant.bytes_per_projection(shape, Projection::down);
    const int64_t total = gate_b + up_b + down_b;

    // Sanity: the loaded GGUF block bytes must match the type-derived sizes.
    auto check = [&](const WeightBundle& b, int64_t want, const char* nm) -> bool {
        if (static_cast<int64_t>(b.weight.data.size()) != want) {
            spdlog::error("pack_gguf_expert: {} block bytes {} != expected {}",
                          nm, b.weight.data.size(), want);
            return false;
        }
        return true;
    };
    if (!check(*gate, gate_b, "gate") || !check(*up, up_b, "up") ||
        !check(*down, down_b, "down")) {
        return false;
    }

    // Zero-copy when the three projections' blocks are already contiguous in
    // gate|up|down order (de-stacked spans into one mmap'd stacked tensor are
    // NOT adjacent across projections, so this rarely fires — kept for parity).
    bool contiguous =
        gate->weight.data.data() + gate->weight.data.size() == up->weight.data.data() &&
        up->weight.data.data() + up->weight.data.size() == down->weight.data.data();
    if (contiguous) {
        bundles[0].packed_slot = std::span<const std::byte>(
            gate->weight.data.data(), static_cast<size_t>(total));
        return true;
    }

    auto buf = std::make_shared<std::vector<std::byte>>(static_cast<size_t>(total));
    auto* dst = buf->data();
    std::memcpy(dst, gate->weight.data.data(), static_cast<size_t>(gate_b));
    std::memcpy(dst + gate_b, up->weight.data.data(), static_cast<size_t>(up_b));
    std::memcpy(dst + gate_b + up_b, down->weight.data.data(),
                static_cast<size_t>(down_b));

    bundles[0].owned_buf = buf;
    bundles[0].packed_slot = std::span<const std::byte>(buf->data(), buf->size());
    return true;
}

// ── NVFP4 input_scale helpers ────────────────────────────────────────────────

namespace {

/// Read a scalar float aux value from a bundle; fallback when absent.
float read_aux_scalar(const WeightBundle& b, TensorRole role, float fallback) {
    const auto* t = b.find_aux(role);
    if (t && t->data.size() >= sizeof(float)) {
        float v;
        std::memcpy(&v, t->data.data(), sizeof(float));
        return v;
    }
    return fallback;
}

}  // namespace

Nvfp4InputScaleNorm compute_nvfp4_input_scale_norm(
    const std::vector<std::vector<WeightBundle>>& layer_experts, int layer_idx) {
    Nvfp4InputScaleNorm norm;
    bool warned = false;
    for (size_t ei = 0; ei < layer_experts.size(); ++ei) {
        float is_gate = 0.f, is_up = 0.f, is_down = 0.f;
        for (const auto& b : layer_experts[ei]) {
            switch (b.id.component) {
                case TensorComponent::gate_proj:
                    is_gate = read_aux_scalar(b, TensorRole::input_scale, 0.f); break;
                case TensorComponent::up_proj:
                    is_up = read_aux_scalar(b, TensorRole::input_scale, 0.f); break;
                case TensorComponent::down_proj:
                    is_down = read_aux_scalar(b, TensorRole::input_scale, 0.f); break;
                default: break;
            }
        }
        // gate and up describe the SAME activation tensor; ModelOpt writes
        // identical values (TRT-LLM asserts equality at load).
        if (!warned && is_gate > 0.f && is_up > 0.f) {
            float diff = std::fabs(is_gate - is_up);
            if (diff > 1e-6f * std::max(is_gate, is_up)) {
                spdlog::warn("layer {} expert {}: gate input_scale {} != up "
                             "input_scale {} — calibrations should be identical; "
                             "using max (accuracy may be affected)",
                             layer_idx, ei, is_gate, is_up);
                warned = true;
            }
        }
        norm.fc31 = std::max({norm.fc31, is_gate, is_up});
        norm.fc2 = std::max(norm.fc2, is_down);
    }
    return norm;
}

// Pack NVFP4 routed expert bundles: linearize [weight | group_scale | weight_scale_2 | input_scale]
// per projection into one contiguous buffer, matching ExpertCache slot layout for resolve_host_source().
// Since format 9.67.0 ("nvfp4-sm1xx"): group scales are written Sm1xx-INTERLEAVED
// (the GEMM consumes them directly; no runtime reformat exists) and input_scale
// scalars are normalized (gate == up == fc31, down == fc2; TRT-LLM semantics).
void pack_nvfp4_expert(std::vector<WeightBundle>& bundles, const ExpertShape& shape,
                       const Nvfp4InputScaleNorm* norm) {
    // Find gate/up/down bundles by component.
    WeightBundle* gate = nullptr;
    WeightBundle* up = nullptr;
    WeightBundle* down = nullptr;
    for (auto& b : bundles) {
        switch (b.id.component) {
            case TensorComponent::gate_proj: gate = &b; break;
            case TensorComponent::up_proj:   up   = &b; break;
            case TensorComponent::down_proj:  down = &b; break;
            default: break;
        }
    }
    if (!gate || !up || !down) {
        spdlog::error("pack_nvfp4_expert: missing gate/up/down bundle — "
                      "gate={} up={} down={}",
                      static_cast<bool>(gate), static_cast<bool>(up),
                      static_cast<bool>(down));
        return;
    }

    // Resolve normalized input_scale values: layer max when provided,
    // otherwise per-expert with the mandatory gate/up merge (the two GEMMs
    // share ONE quantized activation — unequal fields would make alpha_up
    // inconsistent with the scale used in quantization).
    float is_fc31, is_fc2;
    if (norm && norm->fc31 > 0.f) {
        is_fc31 = norm->fc31;
    } else {
        is_fc31 = std::max(read_aux_scalar(*gate, TensorRole::input_scale, 1.0f),
                           read_aux_scalar(*up, TensorRole::input_scale, 1.0f));
        if (is_fc31 <= 0.f) is_fc31 = 1.0f;
    }
    if (norm && norm->fc2 > 0.f) {
        is_fc2 = norm->fc2;
    } else {
        is_fc2 = read_aux_scalar(*down, TensorRole::input_scale, 1.0f);
        if (is_fc2 <= 0.f) is_fc2 = 1.0f;
    }

    // Per-projection byte layout: [FP4_weight | group_scale | weight_scale_2(4B) | input_scale(4B)]
    // Matches Nvfp4::bytes_per_projection().
    // Per-projection aligned size: matches Nvfp4::bytes_per_projection() which aligns to 128.
    // Layout: [FP4_weight | group_scale | PADDING | weight_scale_2(4B) | input_scale(4B)]
    // Scalars at the last 8 bytes so launch_gather_alphas reads at (proj_bytes - 8).
    constexpr int64_t kAlign = 128;

    auto proj_layout = [&](int64_t params) -> std::tuple<int64_t, int64_t, int64_t> {
        int64_t wb = (params + 1) / 2;       // FP4 packed: 2 values per byte
        int64_t sb = (params + 15) / 16;     // group scale: 1 byte per 16 elements
        int64_t raw = wb + sb + 2 * static_cast<int64_t>(sizeof(float));
        int64_t aligned = (raw + kAlign - 1) & ~(kAlign - 1);
        return {wb, sb, aligned};
    };

    auto [gate_wb, gate_sb, gate_total] = proj_layout(shape.gate_params());
    auto [up_wb, up_sb, up_total]       = proj_layout(shape.up_params());
    auto [down_wb, down_sb, down_total] = proj_layout(shape.down_params());
    int64_t total = gate_total + up_total + down_total;

    // Allocate and pack into contiguous buffer (zero-initialized for padding).
    auto buf = std::make_shared<std::vector<std::byte>>(static_cast<size_t>(total), std::byte{0});
    auto* dst = buf->data();
    int64_t offset = 0;

    thread_local std::vector<uint8_t> sfb_scratch;

    auto pack_proj = [&](WeightBundle& b, int64_t wb, int64_t sb, int64_t aligned_total,
                         int64_t scale_rows, int64_t scale_groups, float input_scale) {
        int64_t proj_start = offset;

        // 1. FP4 weight data
        std::memcpy(dst + offset, b.weight.data.data(),
                    static_cast<size_t>(std::min(wb, static_cast<int64_t>(b.weight.data.size()))));
        offset += wb;

        // 2. Group scale (weight_scale), written Sm1xx-INTERLEAVED (nvfp4-sm1xx).
        const auto* ws = b.find_aux(TensorRole::weight_scale);
        if (ws && static_cast<int64_t>(ws->data.size()) >= sb) {
            std::memcpy(dst + offset, ws->data.data(), static_cast<size_t>(sb));
        } else {
            std::memset(dst + offset, 0x7E, static_cast<size_t>(sb));
        }
        if (scale_rows % 128 == 0 && scale_groups % 4 == 0 &&
            scale_rows * scale_groups == sb) {
            reformat_nvfp4_sfb_inplace(reinterpret_cast<uint8_t*>(dst + offset),
                                       scale_rows, scale_groups, sfb_scratch);
        } else {
            // The Sm1xx interleave (and the grouped GEMM itself) requires
            // 128/4-aligned scale matrices — raw scales here would produce
            // quietly-wrong GEMMs downstream, so fail loudly.
            spdlog::error("pack_nvfp4_expert: scale matrix [{}x{}] not "
                          "Sm1xx-alignable (need rows%128==0, groups%4==0) — "
                          "slot left with RAW scales",
                          scale_rows, scale_groups);
        }
        offset += sb;
        // Padding between group_scale and scalars is zero (from allocation).

        // 3. weight_scale_2 at (proj_start + aligned_total - 8)
        int64_t scalar_off = proj_start + aligned_total - 8;
        const auto* ws2 = b.find_aux(TensorRole::weight_scale_2);
        if (ws2 && ws2->data.size() >= sizeof(float)) {
            std::memcpy(dst + scalar_off, ws2->data.data(), sizeof(float));
        } else {
            float one = 1.0f;
            std::memcpy(dst + scalar_off, &one, sizeof(float));
        }

        // 4. input_scale at (proj_start + aligned_total - 4): the NORMALIZED
        // value (gate == up == fc31, down == fc2) — not the raw bundle aux.
        std::memcpy(dst + scalar_off + sizeof(float), &input_scale, sizeof(float));

        // Advance offset to next projection's aligned start.
        offset = proj_start + aligned_total;
    };

    // Scale matrix dims per projection: gate/up are [I, H] -> [I, H/16]
    // scales; down is [H, I] -> [H, I/16].
    const int64_t I = shape.intermediate_size;
    const int64_t H = shape.hidden_size;
    pack_proj(*gate, gate_wb, gate_sb, gate_total, I, H / 16, is_fc31);
    pack_proj(*up,   up_wb,   up_sb,   up_total,   I, H / 16, is_fc31);
    pack_proj(*down, down_wb, down_sb, down_total, H, I / 16, is_fc2);

    // Expose via the first bundle (resolve_host_source reads bundles[0].packed_slot).
    bundles[0].owned_buf = buf;
    bundles[0].packed_slot = std::span<const std::byte>(buf->data(), buf->size());
}

// ── GGUF whole-model loading (GG-6) ──────────────────────────────────────────

namespace {

// If `p` names a split-set member (`<stem>-NNNNN-of-MMMMM.gguf`), expand it to
// the full ordered member set `<stem>-00001-of-MMMMM.gguf` … `-MMMMM-of-MMMMM`
// in the same directory, throwing if any member is missing. Returns nullopt if
// the name is not a split member. `llama.cpp` accepts any single member as the
// entry point and resolves the rest by this convention.
std::optional<std::vector<std::filesystem::path>>
expand_gguf_split_members(const std::filesystem::path& p) {
    namespace fs = std::filesystem;
    static const std::regex kSplitRe(R"(^(.*)-(\d{5})-of-(\d{5})\.gguf$)");
    const std::string name = p.filename().string();
    std::smatch m;
    if (!std::regex_match(name, m, kSplitRe)) return std::nullopt;

    const std::string stem = m[1].str();
    const int member_no = std::stoi(m[2].str());
    const int total = std::stoi(m[3].str());
    if (total < 1) {
        throw std::runtime_error("GGUF split '" + name + "' has invalid -of-"
            + m[3].str());
    }
    if (member_no < 1 || member_no > total) {
        // e.g. -00012-of-00011: expanding 1..11 would silently EXCLUDE the file
        // the caller pointed at — reject instead.
        throw std::runtime_error("GGUF split member index " + m[2].str()
            + " out of range 1.." + m[3].str() + " in " + name);
    }
    const fs::path dir = p.parent_path();
    std::vector<fs::path> files;
    files.reserve(total);
    for (int i = 1; i <= total; ++i) {
        char buf[6];
        std::snprintf(buf, sizeof(buf), "%05d", i);
        fs::path member = dir / (stem + "-" + buf + "-of-" + m[3].str() + ".gguf");
        if (!fs::is_regular_file(member)) {
            throw std::runtime_error("GGUF split set incomplete: missing member "
                + member.string() + " (referenced from " + p.string() + ")");
        }
        files.push_back(std::move(member));
    }
    return files;
}

// Query the split-set's metadata: only shard 0 (the `-00001-` member) carries
// the full KV block. Returns nullopt if absent or non-scalar.
std::optional<GgufMetadataValue> shardset_metadata(
    const std::vector<GgufReader>& shards, std::string_view key) {
    if (shards.empty()) return std::nullopt;
    return shards.front().metadata(key);
}

// De-stack a stacked GGUF expert tensor into per-expert RawTensor spans and
// place them into the layer's routed_experts bundles. The stacked tensor has
// GGUF dims {in, out, n_experts}; each expert's packed block region is
// `per_expert_bytes` long, contiguous, at offset `e * per_expert_bytes`.
void destack_expert_tensor(LoadedModel& model, const TensorId& id,
                           const GgufTensorEntry& entry,
                           std::span<const std::byte> data,
                           int n_routed_experts) {
    if (id.layer_idx < 0 || id.layer_idx >= static_cast<int>(model.layers.size())) {
        return;
    }
    if (entry.dims.size() != 3) {
        throw std::runtime_error("GGUF stacked expert tensor '" + entry.name
            + "' expected 3 dims {in,out,n_experts}, got "
            + std::to_string(entry.dims.size()));
    }
    const int64_t in_features = entry.dims[0];
    const int64_t out_features = entry.dims[1];
    const int64_t n_experts = entry.dims[2];
    if (n_experts != n_routed_experts) {
        throw std::runtime_error("GGUF stacked expert tensor '" + entry.name
            + "' has " + std::to_string(n_experts) + " experts but config has "
            + std::to_string(n_routed_experts));
    }

    GgufKQuantType kt = entry.kquant_type();
    const int64_t per_expert_bytes =
        gguf::gguf_packed_bytes(out_features, in_features, kt);
    if (per_expert_bytes * n_experts != static_cast<int64_t>(data.size())) {
        throw std::runtime_error("GGUF stacked expert tensor '" + entry.name
            + "' size " + std::to_string(data.size()) + " != per_expert "
            + std::to_string(per_expert_bytes) + " x " + std::to_string(n_experts));
    }

    auto& layer = model.layers[id.layer_idx];
    if (static_cast<int>(layer.routed_experts.size()) < n_experts) {
        layer.routed_experts.resize(n_experts);
    }

    for (int64_t e = 0; e < n_experts; ++e) {
        RawTensor raw{
            .data = data.subspan(static_cast<size_t>(e * per_expert_bytes),
                                 static_cast<size_t>(per_expert_bytes)),
            .dtype = SafetensorsDtype::U8,  // packed k-quant bytes
            .shape = {out_features, in_features},
            .gguf_type = kt,
        };
        WeightBundle bundle;
        bundle.id = TensorId{id.component, TensorRole::weight, TensorOwner::routed_expert,
                             id.layer_idx, static_cast<int>(e)};
        bundle.weight = std::move(raw);
        layer.routed_experts[static_cast<size_t>(e)].push_back(std::move(bundle));
    }
}

// Build a RawTensor for one non-stacked GGUF tensor (attention/dense/norms/etc).
RawTensor raw_from_gguf(const GgufTensorEntry& entry, std::span<const std::byte> data) {
    RawTensor raw;
    raw.data = data;
    raw.shape.assign(entry.dims.rbegin(), entry.dims.rend());  // GGUF fastest-first → row-major
    if (entry.is_kquant()) {
        raw.dtype = SafetensorsDtype::U8;  // packed k-quant bytes
        raw.gguf_type = entry.kquant_type();
    } else {
        raw.dtype = gguf_float_dtype(entry.ggml_type);
    }
    return raw;
}

// ── GLM-1: split MLA up-projection (attn_k_b / attn_v_b) assembly ─────────────
// Host-side dequant + transpose + stack into a combined BF16 kv_b_proj. PURE
// host C++ (no CUDA) — runs once at load, off the hot path.

// fp16 (IEEE binary16) → f32. Q8_0 super-block scales and F16 tensors are fp16.
float fp16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // ±0
        } else {
            // Subnormal: normalize.
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x3FFu;
            bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / NaN
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// f32 → bf16 (round-to-nearest-even), matching core/bf16_convert.h's scalar
// algorithm. Scalar form avoids a per-element heap allocation on the ~430 MB
// assembly path.
uint16_t f32_to_bf16_scalar(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

// k-quant 6-bit packed scale/min extractor (ggml get_scale_min_k4, verbatim).
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Dequant a GGUF k-quant / float RawTensor into a flat f32 host vector in ggml
// row-major (fastest-dim-first) order. Q8_0 is exact (lossless); Q4_K/Q5_K/Q6_K
// follow the ggml dequantize_row_q*_K reference bit-for-bit. The general case
// handles each tensor by its own dtype/gguf_type so a split/whole GGUF in a
// different type works for the supported set; unsupported k-quant types throw
// with guidance (TD-GLM1-KQUANT-HOST-DEQUANT).
std::vector<float> dequant_split_to_f32(const RawTensor& t, const char* what) {
    int64_t n = 1;
    for (auto d : t.shape) n *= d;
    std::vector<float> out(static_cast<size_t>(n));

    if (t.gguf_type.has_value()) {
        switch (*t.gguf_type) {
            case GgufKQuantType::MXFP4: {
                // block_mxfp4 = { u8 e8m0; u8 qs[16] } → 17 bytes per 32 values.
                // Delegates to the host reference port (gguf_kquant.cpp).
                constexpr int kQK = 32;
                constexpr int kBlk = 17;
                if (n % kQK != 0) {
                    throw std::runtime_error(
                        std::string(what) + ": MXFP4 element count "
                        + std::to_string(n) + " is not a multiple of 32");
                }
                const int64_t nb = n / kQK;
                if (static_cast<int64_t>(t.data.size()) < nb * kBlk) {
                    throw std::runtime_error(
                        std::string(what) + ": MXFP4 buffer too small: have "
                        + std::to_string(t.data.size()) + " need "
                        + std::to_string(nb * kBlk));
                }
                const auto* p = reinterpret_cast<const uint8_t*>(t.data.data());
                core::parallel_for(static_cast<size_t>(nb), [&](size_t bi) {
                    gguf::dequant_mxfp4_block(
                        p + bi * kBlk, out.data() + bi * kQK);
                });
                return out;
            }
            case GgufKQuantType::Q8_0: {
                // block_q8_0 = { fp16 d; int8 qs[32] } → 34 bytes per 32 values.
                constexpr int kQK = 32;
                constexpr int kBlk = 34;
                if (n % kQK != 0) {
                    throw std::runtime_error(
                        std::string("assemble_split_kv_b_proj: ") + what
                        + " Q8_0 element count " + std::to_string(n)
                        + " is not a multiple of 32");
                }
                const int64_t nb = n / kQK;
                if (static_cast<int64_t>(t.data.size()) < nb * kBlk) {
                    throw std::runtime_error(
                        std::string("assemble_split_kv_b_proj: ") + what
                        + " Q8_0 buffer too small: have "
                        + std::to_string(t.data.size()) + " need "
                        + std::to_string(nb * kBlk));
                }
                const auto* p = reinterpret_cast<const uint8_t*>(t.data.data());
                core::parallel_for(static_cast<size_t>(nb), [&](size_t bi) {
                    const int64_t b = static_cast<int64_t>(bi);
                    uint16_t dh;
                    std::memcpy(&dh, p + b * kBlk, sizeof(dh));
                    const float d = fp16_to_f32(dh);
                    const auto* q = reinterpret_cast<const int8_t*>(p + b * kBlk + 2);
                    for (int i = 0; i < kQK; ++i) {
                        out[static_cast<size_t>(b * kQK + i)] =
                            d * static_cast<float>(q[i]);
                    }
                });
                return out;
            }
            case GgufKQuantType::Q4_K: {
                constexpr int QK_K = 256, kBlk = 144;
                const int64_t nb = n / QK_K;
                if (n % QK_K != 0 ||
                    static_cast<int64_t>(t.data.size()) < nb * kBlk)
                    throw std::runtime_error(std::string(what)
                        + ": Q4_K size/shape invalid for host dequant");
                const auto* base =
                    reinterpret_cast<const uint8_t*>(t.data.data());
                core::parallel_for(static_cast<size_t>(nb), [&](size_t bi) {
                    const int64_t b = static_cast<int64_t>(bi);
                    const uint8_t* blk = base + b * kBlk;
                    int64_t oi = b * QK_K;  // per-block output base
                    uint16_t dh, dmh;
                    std::memcpy(&dh, blk, 2);
                    std::memcpy(&dmh, blk + 2, 2);
                    const float d = fp16_to_f32(dh);
                    const float dmin = fp16_to_f32(dmh);
                    const uint8_t* scales = blk + 4;   // 12 bytes
                    const uint8_t* q = blk + 16;       // 128 bytes (qs)
                    int is = 0;
                    for (int j = 0; j < QK_K; j += 64) {
                        uint8_t sc, m;
                        get_scale_min_k4(is + 0, scales, &sc, &m);
                        const float d1 = d * sc, m1 = dmin * m;
                        get_scale_min_k4(is + 1, scales, &sc, &m);
                        const float d2 = d * sc, m2 = dmin * m;
                        for (int l = 0; l < 32; ++l)
                            out[oi++] = d1 * (q[l] & 0xF) - m1;
                        for (int l = 0; l < 32; ++l)
                            out[oi++] = d2 * (q[l] >> 4) - m2;
                        q += 32; is += 2;
                    }
                });
                return out;
            }
            case GgufKQuantType::Q5_K: {
                constexpr int QK_K = 256, kBlk = 176;
                const int64_t nb = n / QK_K;
                if (n % QK_K != 0 ||
                    static_cast<int64_t>(t.data.size()) < nb * kBlk)
                    throw std::runtime_error(std::string(what)
                        + ": Q5_K size/shape invalid for host dequant");
                const auto* base =
                    reinterpret_cast<const uint8_t*>(t.data.data());
                core::parallel_for(static_cast<size_t>(nb), [&](size_t bi) {
                    const int64_t b = static_cast<int64_t>(bi);
                    const uint8_t* blk = base + b * kBlk;
                    int64_t oi = b * QK_K;  // per-block output base
                    uint16_t dh, dmh;
                    std::memcpy(&dh, blk, 2);
                    std::memcpy(&dmh, blk + 2, 2);
                    const float d = fp16_to_f32(dh);
                    const float dmin = fp16_to_f32(dmh);
                    const uint8_t* scales = blk + 4;   // 12 bytes
                    const uint8_t* qh = blk + 16;      // 32 bytes (high bit)
                    const uint8_t* ql = blk + 48;      // 128 bytes (low 4 bits)
                    int is = 0;
                    uint8_t u1 = 1, u2 = 2;
                    for (int j = 0; j < QK_K; j += 64) {
                        uint8_t sc, m;
                        get_scale_min_k4(is + 0, scales, &sc, &m);
                        const float d1 = d * sc, m1 = dmin * m;
                        get_scale_min_k4(is + 1, scales, &sc, &m);
                        const float d2 = d * sc, m2 = dmin * m;
                        for (int l = 0; l < 32; ++l)
                            out[oi++] = d1 * ((ql[l] & 0xF)
                                       + ((qh[l] & u1) ? 16 : 0)) - m1;
                        for (int l = 0; l < 32; ++l)
                            out[oi++] = d2 * ((ql[l] >> 4)
                                       + ((qh[l] & u2) ? 16 : 0)) - m2;
                        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
                    }
                });
                return out;
            }
            case GgufKQuantType::Q6_K: {
                constexpr int QK_K = 256, kBlk = 210;
                const int64_t nb = n / QK_K;
                if (n % QK_K != 0 ||
                    static_cast<int64_t>(t.data.size()) < nb * kBlk)
                    throw std::runtime_error(std::string(what)
                        + ": Q6_K size/shape invalid for host dequant");
                const auto* base =
                    reinterpret_cast<const uint8_t*>(t.data.data());
                core::parallel_for(static_cast<size_t>(nb), [&](size_t bi) {
                    const int64_t b = static_cast<int64_t>(bi);
                    const uint8_t* blk = base + b * kBlk;
                    const int64_t oi = b * QK_K;  // per-block output base
                    const uint8_t* ql = blk;            // 128 bytes
                    const uint8_t* qh = blk + 128;      // 64 bytes
                    const int8_t* sc =
                        reinterpret_cast<const int8_t*>(blk + 192);  // 16
                    uint16_t dh;
                    std::memcpy(&dh, blk + 208, 2);
                    const float d = fp16_to_f32(dh);
                    int64_t yp = oi;
                    const uint8_t* qlp = ql;
                    const uint8_t* qhp = qh;
                    const int8_t* scp = sc;
                    for (int nn = 0; nn < QK_K; nn += 128) {
                        for (int l = 0; l < 32; ++l) {
                            const int is = l / 16;
                            const int8_t q1 = static_cast<int8_t>(
                                (qlp[l] & 0xF) | (((qhp[l] >> 0) & 3) << 4)) - 32;
                            const int8_t q2 = static_cast<int8_t>(
                                (qlp[l+32] & 0xF) | (((qhp[l] >> 2) & 3) << 4)) - 32;
                            const int8_t q3 = static_cast<int8_t>(
                                (qlp[l] >> 4) | (((qhp[l] >> 4) & 3) << 4)) - 32;
                            const int8_t q4 = static_cast<int8_t>(
                                (qlp[l+32] >> 4) | (((qhp[l] >> 6) & 3) << 4)) - 32;
                            out[yp + l +  0] = d * scp[is + 0] * q1;
                            out[yp + l + 32] = d * scp[is + 2] * q2;
                            out[yp + l + 64] = d * scp[is + 4] * q3;
                            out[yp + l + 96] = d * scp[is + 6] * q4;
                        }
                        yp += 128; qlp += 64; qhp += 32; scp += 8;
                    }
                });
                return out;
            }
            default:
                throw std::runtime_error(
                    std::string("assemble_split_kv_b_proj: ") + what + " GGUF type "
                    + std::string(gguf::type_name(*t.gguf_type))
                    + " is not supported for host dequant (supported: Q4_K, Q5_K, "
                      "Q6_K, Q8_0, and float). See TD-GLM1-KQUANT-HOST-DEQUANT.");
        }
    }

    // Non-quantized float tensor.
    const auto* p = reinterpret_cast<const uint8_t*>(t.data.data());
    const int64_t need = n * static_cast<int64_t>(dtype_size(t.dtype));
    if (static_cast<int64_t>(t.data.size()) < need) {
        throw std::runtime_error(
            std::string("assemble_split_kv_b_proj: ") + what
            + " float buffer too small: have " + std::to_string(t.data.size())
            + " need " + std::to_string(need));
    }
    switch (t.dtype) {
        case SafetensorsDtype::F32: {
            constexpr int64_t kChunk = 1 << 20;
            core::parallel_for(
                static_cast<size_t>((n + kChunk - 1) / kChunk), [&](size_t c) {
                const int64_t lo = static_cast<int64_t>(c) * kChunk;
                const int64_t hi = std::min<int64_t>(lo + kChunk, n);
                for (int64_t i = lo; i < hi; ++i) {
                float f;
                    std::memcpy(&f, p + i * 4, sizeof(f));
                    out[static_cast<size_t>(i)] = f;
                }
            });
            return out;
        }
        case SafetensorsDtype::F16: {
            constexpr int64_t kChunk = 1 << 20;
            core::parallel_for(
                static_cast<size_t>((n + kChunk - 1) / kChunk), [&](size_t c) {
                const int64_t lo = static_cast<int64_t>(c) * kChunk;
                const int64_t hi = std::min<int64_t>(lo + kChunk, n);
                for (int64_t i = lo; i < hi; ++i) {
                uint16_t h;
                    std::memcpy(&h, p + i * 2, sizeof(h));
                    out[static_cast<size_t>(i)] = fp16_to_f32(h);
                }
            });
            return out;
        }
        case SafetensorsDtype::BF16: {
            constexpr int64_t kChunk = 1 << 20;
            core::parallel_for(
                static_cast<size_t>((n + kChunk - 1) / kChunk), [&](size_t c) {
                const int64_t lo = static_cast<int64_t>(c) * kChunk;
                const int64_t hi = std::min<int64_t>(lo + kChunk, n);
                for (int64_t i = lo; i < hi; ++i) {
                uint16_t h;
                    std::memcpy(&h, p + i * 2, sizeof(h));
                    uint32_t bits = static_cast<uint32_t>(h) << 16;
                    float f;
                    std::memcpy(&f, &bits, sizeof(f));
                    out[static_cast<size_t>(i)] = f;
                }
            });
            return out;
        }
        default:
            throw std::runtime_error(
                std::string("assemble_split_kv_b_proj: ") + what
                + " dtype is not a supported float/Q8_0 type for the split MLA "
                  "up-projection");
    }
}

}  // namespace

// External-linkage (declared in weight_loader.h; exposed for testing). Calls the
// anonymous-namespace helper expand_gguf_split_members defined above.
std::vector<std::filesystem::path> resolve_gguf_files(const std::string& weights_path) {
    namespace fs = std::filesystem;
    fs::path p(weights_path);
    std::vector<fs::path> files;
    if (fs::is_directory(p)) {
        for (const auto& e : fs::directory_iterator(p)) {
            if (e.is_regular_file() && e.path().extension() == ".gguf") {
                files.push_back(e.path());
            }
        }
        std::sort(files.begin(), files.end());
    } else if (fs::is_regular_file(p)) {
        if (auto members = expand_gguf_split_members(p)) {
            files = std::move(*members);  // already ordered 00001..MMMMM
        } else {
            files.push_back(p);
        }
    }
    return files;
}

AssembledKvB assemble_split_kv_b_proj(const RawTensor& attn_k_b,
                                      const RawTensor& attn_v_b) {
    // RawTensor.shape is GGUF-reversed row-major:
    //   attn_k_b : [H, L, P]  (= W_UK transposed per head: rows=kv_lora, cols=qk_nope)
    //   attn_v_b : [H, V, L]  (= W_UV per head:           rows=v_head_dim, cols=kv_lora)
    if (attn_k_b.shape.size() != 3 || attn_v_b.shape.size() != 3) {
        throw std::runtime_error(
            "assemble_split_kv_b_proj: expected 3D attn_k_b/attn_v_b (got "
            + std::to_string(attn_k_b.shape.size()) + "D / "
            + std::to_string(attn_v_b.shape.size()) + "D)");
    }
    const int64_t H = attn_k_b.shape[0];
    const int64_t L = attn_k_b.shape[1];
    const int64_t P = attn_k_b.shape[2];
    const int64_t Hv = attn_v_b.shape[0];
    const int64_t V = attn_v_b.shape[1];
    const int64_t Lv = attn_v_b.shape[2];
    if (Hv != H || Lv != L) {
        throw std::runtime_error(std::format(
            "assemble_split_kv_b_proj: shape mismatch — attn_k_b [H={},L={},P={}] "
            "vs attn_v_b [H={},V={},L={}] (heads/kv_lora must agree)",
            H, L, P, Hv, V, Lv));
    }

    const std::vector<float> kf = dequant_split_to_f32(attn_k_b, "attn_k_b");  // [H,L,P]
    const std::vector<float> vf = dequant_split_to_f32(attn_v_b, "attn_v_b");  // [H,V,L]

    const int64_t rows = H * (P + V);
    auto buf = std::make_shared<std::vector<std::byte>>(
        static_cast<size_t>(rows * L) * sizeof(uint16_t));
    auto* out = reinterpret_cast<uint16_t*>(buf->data());

    for (int64_t h = 0; h < H; ++h) {
        const int64_t row_base = h * (P + V);
        // W_UK: transpose attn_k_b[h] from [L, P] → [P, L].
        for (int64_t p = 0; p < P; ++p) {
            uint16_t* dst = out + (row_base + p) * L;
            for (int64_t l = 0; l < L; ++l) {
                const float v = kf[static_cast<size_t>((h * L + l) * P + p)];
                dst[l] = f32_to_bf16_scalar(v);
            }
        }
        // W_UV: attn_v_b[h] is already [V, L] — copy directly.
        for (int64_t vrow = 0; vrow < V; ++vrow) {
            uint16_t* dst = out + (row_base + P + vrow) * L;
            const float* src = &vf[static_cast<size_t>((h * V + vrow) * L)];
            for (int64_t l = 0; l < L; ++l) {
                dst[l] = f32_to_bf16_scalar(src[l]);
            }
        }
    }

    return AssembledKvB{std::move(buf), {rows, L}};
}

namespace {

// GLM-1: find the paired split attn_k_b/attn_v_b groups in the accumulator,
// assemble each layer's combined BF16 kv_b_proj, place it into the model, and
// erase the consumed split groups so they never reach assemble_bundle (which
// would treat them as standalone weights).
void assemble_split_kv_b_groups(LoadedModel& model, TensorAccumulator& accumulator,
                                const ModelConfig& model_cfg) {
    auto weight_raw = [](std::vector<AccumulatedTensor>& ts) -> const RawTensor* {
        for (auto& t : ts) {
            if (t.role == TensorRole::weight) return &t.raw;
        }
        return ts.empty() ? nullptr : &ts.front().raw;
    };

    std::map<int, const RawTensor*> k_by_layer, v_by_layer;
    std::vector<GroupKey> consumed;
    for (auto& [key, tensors] : accumulator) {
        if (key.component == TensorComponent::mla_k_b_split) {
            k_by_layer[key.layer_idx] = weight_raw(tensors);
            consumed.push_back(key);
        } else if (key.component == TensorComponent::mla_v_b_split) {
            v_by_layer[key.layer_idx] = weight_raw(tensors);
            consumed.push_back(key);
        }
    }

    if (k_by_layer.empty() && v_by_layer.empty()) return;  // combined-kv_b model

    // Phase 1 (parallel, <=8 threads): per-layer dequant+transpose+stack is
    // independent; placement below stays serial (mutates model/accumulator).
    // parallel_for's inner dequant calls also fan out — nested fan-out is
    // bounded (outer threads mostly wait on their own inner loops), and layer
    // count dominates, so parallelize the OUTER loop and let the inner
    // dequants run inline for these small per-layer tensors.
    struct KvbItem { int layer; const RawTensor* kb; const RawTensor* vb; };
    std::vector<KvbItem> items;
    items.reserve(k_by_layer.size());
    for (const auto& [layer, kb] : k_by_layer) {
        auto vit = v_by_layer.find(layer);
        if (vit == v_by_layer.end() || vit->second == nullptr || kb == nullptr) {
            throw std::runtime_error(std::format(
                "GLM-1: layer {} has attn_k_b but is missing attn_v_b (split MLA "
                "up-projection requires both)", layer));
        }
        items.push_back({layer, kb, vit->second});
    }
    std::vector<AssembledKvB> assembled(items.size());
    core::parallel_for(items.size(), [&](size_t i) {
        assembled[i] = assemble_split_kv_b_proj(*items[i].kb, *items[i].vb);
    }, /*max_threads=*/core::parallel_for_default_threads());

    for (size_t ii = 0; ii < items.size(); ++ii) {
        const int layer = items[ii].layer;
        AssembledKvB& asm_kv = assembled[ii];

        // Defensive cross-check against the config dims the absorbed-MLA kernels
        // slice with (q_absorb d_nope_in/d_c/d_v): a config/file mismatch would
        // otherwise produce a silently-misaligned kv_b. The combined layout is
        // [n_head*(qk_nope+v_head), kv_lora].
        const auto& raw = model_cfg.raw();
        const int64_t expect_rows = static_cast<int64_t>(raw.num_attention_heads)
            * (static_cast<int64_t>(raw.qk_nope_head_dim) + raw.v_head_dim);
        if (asm_kv.shape[0] != expect_rows
            || asm_kv.shape[1] != raw.kv_lora_rank) {
            throw std::runtime_error(std::format(
                "GLM-1: layer {} assembled kv_b_proj shape [{}, {}] disagrees with "
                "config (n_head={} qk_nope={} v_head={} kv_lora={} → expected [{}, {}])",
                layer, asm_kv.shape[0], asm_kv.shape[1], raw.num_attention_heads,
                raw.qk_nope_head_dim, raw.v_head_dim, raw.kv_lora_rank,
                expect_rows, raw.kv_lora_rank));
        }

        WeightBundle bundle;
        bundle.id = TensorId{TensorComponent::kv_b_proj, TensorRole::weight,
                             TensorOwner::attention, layer, -1};
        bundle.owned_buf = asm_kv.buf;
        bundle.weight.data = std::span<const std::byte>(asm_kv.buf->data(),
                                                        asm_kv.buf->size());
        bundle.weight.dtype = SafetensorsDtype::BF16;  // BF16 branch (gguf_type unset)
        bundle.weight.shape = asm_kv.shape;            // [n_head*(P+V), kv_lora]
        // gguf_type intentionally left nullopt: q_absorb/kv_bv take the BF16
        // path (weight_is_fp8=false, gguf_type=-1), NOT the in-kernel GGUF
        // dequant path (which expects a combined-packed kv_b, not this split).

        model.total_weight_bytes += bundle.total_bytes();
        ++model.total_tensors_loaded;
        place_bundle(model, std::move(bundle), model_cfg);
    }

    // Any attn_v_b without a matching attn_k_b is an error too.
    for (const auto& [layer, vb] : v_by_layer) {
        if (!k_by_layer.count(layer)) {
            throw std::runtime_error(std::format(
                "GLM-1: layer {} has attn_v_b but is missing attn_k_b", layer));
        }
    }

    for (const auto& key : consumed) accumulator.erase(key);

    spdlog::info("GLM-1: assembled combined BF16 kv_b_proj for {} layer(s) from "
                 "split attn_k_b/attn_v_b", k_by_layer.size());
}

// GG-9: dequantize a GGUF-quantized embedding / lm_head bundle to BF16 in place.
// The pinned-upload plan and the embedding-lookup / output-head kernels are
// BF16-only (TD-53q rejects a k-quant here), but real GGUFs store token_embd /
// output quantized (Q5_K / Q6_K in GLM-4.7-Flash). Decode once at load to flat
// row-major BF16 — no transpose: the embedding lookup reads contiguous BF16 rows
// and the lm_head GEMM is row-major [vocab, hidden], matching the ggml layout.
void dequant_bundle_to_bf16(WeightBundle& b, const char* what) {
    if (!b.weight.gguf_type.has_value()) return;  // already a float tensor
    const std::vector<float> f = dequant_split_to_f32(b.weight, what);
    auto buf = std::make_shared<std::vector<std::byte>>(
        f.size() * sizeof(uint16_t));
    auto* out = reinterpret_cast<uint16_t*>(buf->data());
    {
        constexpr size_t kChunk = 1 << 20;
        core::parallel_for((f.size() + kChunk - 1) / kChunk, [&](size_t c) {
            const size_t lo = c * kChunk;
            const size_t hi = std::min(lo + kChunk, f.size());
            for (size_t i = lo; i < hi; ++i) out[i] = f32_to_bf16_scalar(f[i]);
        });
    }
    b.owned_buf = buf;
    b.weight.data = std::span<const std::byte>(buf->data(), buf->size());
    b.weight.dtype = SafetensorsDtype::BF16;
    b.weight.gguf_type = std::nullopt;
    spdlog::info("GG-9: dequantized {} to BF16 ({} elems, {:.0f} MB)", what,
                 f.size(), buf->size() / (1024.0 * 1024.0));
}

// f32 → fp16 (IEEE binary16, round-to-nearest-even). Inverse of fp16_to_f32;
// used to store the per-block Q8_0 scale `d` (always a small positive normal for
// real weights, but the conversion handles the full range for safety).
uint16_t f32_to_fp16_scalar(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t f_exp = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;
    if (f_exp == 0xFFu)  // inf / nan
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    int32_t exp = static_cast<int32_t>(f_exp) - 127 + 15;
    if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow→inf
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);  // underflow→±0
        mant |= 0x800000u;                                   // restore implicit 1
        const int shift = 14 - exp;
        uint32_t r = mant >> shift;
        const uint32_t round_bit = 1u << (shift - 1);
        if ((mant & round_bit) && ((mant & (round_bit - 1)) || (r & 1u))) ++r;
        return static_cast<uint16_t>(sign | r);
    }
    uint16_t r = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10)
                                       | (mant >> 13));
    const uint32_t round_bit = 1u << 12;
    if ((mant & round_bit) && ((mant & (round_bit - 1)) || (r & 1u))) ++r;
    return r;
}

// GG-9 (TD-GG9-F16-ATTN-PROJ-FP8-OOB): requantize a plain-float attention/dense
// projection weight to Q8_0 in place. A mixed "UD"/"XL" GGUF can store a handful
// of attention projections (e.g. attn_q_b on 5 of 47 GLM-4.7-Flash layers) as
// plain F16 while the rest are Q8_0. GG-4's attention dispatch only handles
// k-quant (→ GGUF GEMM) and fp8 (→ fp8 GEMM); a plain-float weight has neither
// gguf_type nor fp8 scales, so dcp_executor falls through to the FP8 block-scaled
// GEMM with a null scale_B → null-scale OOB read. Requantizing to Q8_0 (the SAME
// type the other layers' projections already use) routes them uniformly through
// the proven GGUF int/dequant path. Q8_0 is 8-bit and near-lossless for argmax;
// the layout matches ggml (32 contiguous values along the in/contraction dim,
// fp16 block scale + 32 int8) so it is byte-identical to a real Q8_0 layer.
void requant_bundle_to_q8_0(WeightBundle& b, const std::string& what) {
    if (b.weight.gguf_type.has_value()) return;  // already a k-quant
    if (b.weight.dtype != SafetensorsDtype::F16 &&
        b.weight.dtype != SafetensorsDtype::BF16 &&
        b.weight.dtype != SafetensorsDtype::F32)
        return;  // not a plain-float weight
    if (b.weight.shape.size() != 2)
        throw std::runtime_error("requant_bundle_to_q8_0: " + what
            + " expected a 2D weight, got rank "
            + std::to_string(b.weight.shape.size()));
    const int64_t out_f = b.weight.shape[0];
    const int64_t in_f = b.weight.shape[1];
    if (in_f % 32 != 0)
        throw std::runtime_error("requant_bundle_to_q8_0: " + what
            + " in_features " + std::to_string(in_f)
            + " is not a multiple of 32 (Q8_0 block size)");

    const std::vector<float> f = dequant_split_to_f32(b.weight, what.c_str());
    const int64_t n = out_f * in_f;
    const int64_t nb = n / 32;
    constexpr int kBlk = 34;  // block_q8_0 = { fp16 d; int8 qs[32] }
    auto buf = std::make_shared<std::vector<std::byte>>(
        static_cast<size_t>(nb) * kBlk);
    auto* p = reinterpret_cast<uint8_t*>(buf->data());
    core::parallel_for(static_cast<size_t>(nb), [&](size_t blki) {
        const int64_t blk = static_cast<int64_t>(blki);
        const float* x = f.data() + blk * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
        const float d = amax / 127.f;
        const float id = (d != 0.f) ? 1.f / d : 0.f;
        uint8_t* blkp = p + blk * kBlk;
        const uint16_t dh = f32_to_fp16_scalar(d);
        std::memcpy(blkp, &dh, 2);
        auto* qs = reinterpret_cast<int8_t*>(blkp + 2);
        for (int j = 0; j < 32; ++j) {
            int q = static_cast<int>(std::lround(x[j] * id));
            q = std::max(-127, std::min(127, q));
            qs[j] = static_cast<int8_t>(q);
        }
    });
    b.owned_buf = buf;
    b.weight.data = std::span<const std::byte>(buf->data(), buf->size());
    b.weight.dtype = SafetensorsDtype::U8;          // packed k-quant bytes
    b.weight.gguf_type = GgufKQuantType::Q8_0;
    spdlog::info("GG-9: requantized {} plain-float→Q8_0 ({} elems, {:.1f} MB)",
                 what, n, buf->size() / (1024.0 * 1024.0));
}

}  // namespace

std::vector<uint16_t> dequant_kquant_range_to_bf16(const RawTensor& t,
                                                   int64_t elem_off,
                                                   int64_t elem_cnt) {
    int64_t block_elems = 1, block_bytes = 0;
    if (t.gguf_type.has_value()) {
        switch (*t.gguf_type) {
            case GgufKQuantType::Q8_0: block_elems = 32;  block_bytes = 34;  break;
            case GgufKQuantType::Q4_K: block_elems = 256; block_bytes = 144; break;
            case GgufKQuantType::Q5_K: block_elems = 256; block_bytes = 176; break;
            case GgufKQuantType::Q6_K: block_elems = 256; block_bytes = 210; break;
            case GgufKQuantType::MXFP4: block_elems = 32; block_bytes = 17;  break;
            default:
                throw std::runtime_error(
                    "dequant_kquant_range_to_bf16: unsupported k-quant type");
        }
    } else {
        block_bytes = static_cast<int64_t>(dtype_size(t.dtype));
    }
    if (elem_off % block_elems != 0 || elem_cnt % block_elems != 0)
        throw std::runtime_error(
            "dequant_kquant_range_to_bf16: range not block-aligned (off=" +
            std::to_string(elem_off) + " cnt=" + std::to_string(elem_cnt) +
            " block=" + std::to_string(block_elems) + ")");
    // A block-aligned range of a 1-D-flattened quant tensor is itself a valid
    // quant tensor — dequant a VIEW (no refactor of the per-type kernels).
    RawTensor view = t;
    view.shape = {elem_cnt};
    view.data = t.data.subspan(
        static_cast<size_t>(elem_off / block_elems * block_bytes),
        static_cast<size_t>(elem_cnt / block_elems * block_bytes));
    const std::vector<float> f = dequant_split_to_f32(view, "range");
    std::vector<uint16_t> out(f.size());
    constexpr size_t kChunk = 1 << 20;
    core::parallel_for((f.size() + kChunk - 1) / kChunk, [&](size_t c) {
        const size_t lo = c * kChunk;
        const size_t hi = std::min(lo + kChunk, f.size());
        for (size_t i = lo; i < hi; ++i) out[i] = f32_to_bf16_scalar(f[i]);
    });
    return out;
}

LoadedModel load_weights_gguf(const config::Config& cfg,
                              const ModelConfig& model_cfg,
                              const LayerRegistry& registry,
                              bool skip_routed_experts) {
    namespace fs = std::filesystem;

    const auto& weights_path = cfg.model.weights_path;
    bool use_mmap = cfg.model.use_mmap;

    auto gguf_files = resolve_gguf_files(weights_path);
    if (gguf_files.empty()) {
        throw std::runtime_error("No .gguf file found at " + weights_path);
    }
    spdlog::info("Loading GGUF weights from {} ({} file(s), {})", weights_path,
                 gguf_files.size(), use_mmap ? "mmap" : "pread");

    LoadedModel model;
    model.gguf_shards.reserve(gguf_files.size());
    for (const auto& f : gguf_files) {
        model.gguf_shards.push_back(GgufReader::open(f, use_mmap));
    }

    // Defensive split validation: shard 0 carries split.count; a truncated
    // download (fewer members than declared) surfaces here with a clear error
    // instead of a confusing downstream completeness failure.
    if (auto sc = shardset_metadata(model.gguf_shards, "split.count")) {
        // split.count is u16 per the GGUF split convention, but tolerate a
        // writer that used a signed int type.
        const uint64_t declared =
            sc->kind == GgufMetadataValue::Kind::i64
                ? (sc->i > 0 ? static_cast<uint64_t>(sc->i) : 0)
                : sc->u;
        if (declared != 0 && declared != model.gguf_shards.size()) {
            throw std::runtime_error(
                "GGUF split.count=" + std::to_string(declared) + " but resolved "
                + std::to_string(model.gguf_shards.size())
                + " shard(s) from " + weights_path);
        }
    }

    const int num_layers = model_cfg.raw().num_hidden_layers;
    const int n_routed = model_cfg.raw().n_routed_experts;
    model.layers.resize(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        model.layers[l].layer_idx = l;
        if (model_cfg.is_moe_layer(l) && !skip_routed_experts) {
            model.layers[l].routed_experts.resize(n_routed);
        }
    }

    // Non-expert tensors are grouped (a logical weight may have a bias aux); the
    // stacked expert tensors are de-stacked immediately (no aux to group).
    TensorAccumulator accumulator;
    int total_entries = 0, unrecognized = 0;

    for (auto& shard : model.gguf_shards) {
        for (const auto& entry : shard.entries()) {
            ++total_entries;
            auto id_opt = parse_gguf_name(entry.name);
            if (!id_opt) {
                spdlog::warn("Unrecognized GGUF tensor name: {}", entry.name);
                ++unrecognized;
                continue;
            }

            auto data = shard.tensor_data(entry);

            // Stacked routed experts: de-stack into per-expert bundles.
            if (id_opt->owner == TensorOwner::routed_expert) {
                if (skip_routed_experts) continue;
                destack_expert_tensor(model, *id_opt, entry, data, n_routed);
                continue;
            }

            auto key = key_from_id(*id_opt);
            accumulator[key].push_back(
                AccumulatedTensor{id_opt->role, raw_from_gguf(entry, data)});
        }
    }

    spdlog::info("Scanned {} GGUF tensor entries, {} unrecognized",
                 total_entries, unrecognized);

    // GLM-1: if this GGUF ships the MLA up-projection PRE-SPLIT as
    // attn_k_b/attn_v_b (llama.cpp MLA-optimized layout), dequant + transpose +
    // stack them into a combined BF16 kv_b_proj BEFORE the generic assemble loop
    // (so the split tensors are not treated as standalone weights). No-op for
    // models that ship the combined attn_kv_b.
    assemble_split_kv_b_groups(model, accumulator, model_cfg);

    int validation_warnings = 0;
    for (auto& [key, tensors] : accumulator) {
        auto [bundle, err] = assemble_bundle(key, tensors, model_cfg);
        if (!err.empty()) {
            spdlog::warn("GGUF weight validation: {}", err);
            ++validation_warnings;
        }
        model.total_weight_bytes += bundle.total_bytes();
        ++model.total_tensors_loaded;
        place_bundle(model, std::move(bundle), model_cfg);
    }

    // Count de-stacked experts in the byte/tensor totals.
    for (const auto& layer : model.layers) {
        for (const auto& experts : layer.routed_experts) {
            for (const auto& b : experts) {
                model.total_weight_bytes += b.total_bytes();
                ++model.total_tensors_loaded;
            }
        }
    }

    // GG-9: embeddings / lm_head ship k-quantized in real GGUFs but the engine's
    // embedding-lookup + output-head paths are BF16-only — decode them now.
    // Round 2b: k-quant embed/lm_head are NOT dequanted here anymore — the
    // upload path streams them chunk-by-chunk (dequant_kquant_range_to_bf16 →
    // pinned staging → H2D), so the ~3.6 GB BF16 host copies (and the ~3.8 GB
    // F32 intermediate) never exist. Float-shipped tensors keep the old path.
    if (model.embedding && !model.embedding->weight.gguf_type.has_value())
        dequant_bundle_to_bf16(*model.embedding, "token_embd");
    if (model.output_head && !model.output_head->weight.gguf_type.has_value())
        dequant_bundle_to_bf16(*model.output_head, "output.weight");

    // GG-9 (TD-GG9-F16-ATTN-PROJ-FP8-OOB): some mixed "UD"/"XL" GGUFs store a few
    // attention projections as plain F16 while the rest are Q8_0 (unsloth quant
    // quirk). The GG-4 attention dispatch only handles k-quant + fp8; a plain-
    // float q_a/q_b/kv_a/o_proj falls through to the FP8 block-scaled GEMM with a
    // null scale_B → illegal memory access. Requantize any such plain-float
    // attention projection to Q8_0 so it routes uniformly through the proven GGUF
    // GEMM path. NOT kv_b_proj (GLM-1 assembles it to BF16 and q_absorb consumes
    // it directly via its own dequant branch), NOT norms/router.
    if (!model_cfg.is_v4())  // V4 attention ships BF16-native (census §0.3);
                             // its CSA/HCA device consumes BF16 directly —
                             // do NOT requantize V4 projections to Q8_0.
    {
        int n_requant = 0;
        for (auto& layer : model.layers) {
            for (auto& b : layer.attention) {
                switch (b.id.component) {
                    case TensorComponent::q_a_proj:
                    case TensorComponent::q_b_proj:
                    case TensorComponent::kv_a_proj_with_mqa:
                    case TensorComponent::o_proj: {
                        const bool was_float = !b.weight.gguf_type.has_value();
                        requant_bundle_to_q8_0(
                            b, std::format("layer {} {}", layer.layer_idx,
                                           tensor_component_name(b.id.component)));
                        if (was_float && b.weight.gguf_type.has_value())
                            ++n_requant;
                        break;
                    }
                    default:
                        break;
                }
            }
        }
        if (n_requant > 0)
            spdlog::info("GG-9: requantized {} plain-float attention projection(s)"
                         " to Q8_0 (TD-GG9-F16-ATTN-PROJ-FP8-OOB)", n_requant);
    }

    spdlog::info("Assembled GGUF bundles: {} tensors ({:.1f} GB), {} warnings",
                 model.total_tensors_loaded,
                 model.total_weight_bytes / (1024.0 * 1024.0 * 1024.0),
                 validation_warnings);

    auto completeness_errors = validate_completeness(model, model_cfg, registry,
                                                      skip_routed_experts);
    if (!completeness_errors.empty()) {
        std::string msg = "GGUF weight completeness errors:\n";
        for (auto& e : completeness_errors) {
            msg += "  - " + e + "\n";
            spdlog::error("Completeness: {}", e);
        }
        throw std::runtime_error(msg);
    }

    spdlog::info("GGUF weight loading complete: {} layers, {} bundles{}",
                 model.layers.size(), model.total_tensors_loaded,
                 skip_routed_experts ? " (routed experts skipped)" : "");
    return model;
}

// ── load_weights ─────────────────────────────────────────────────────────────

LoadedModel load_weights(const config::Config& cfg,
                         const ModelConfig& model_cfg,
                         const LayerRegistry& registry,
                         bool skip_routed_experts) {
    namespace fs = std::filesystem;

    const auto& weights_path = cfg.model.weights_path;
    if (weights_path.empty()) {
        throw std::runtime_error("model.weights_path is empty");
    }

    fs::path model_dir(weights_path);
    if (!fs::exists(model_dir)) {
        throw std::runtime_error("Model directory does not exist: " + weights_path);
    }

    // GGUF container path (GG-6): dispatch when the config selects GGUF weights.
    if (cfg.model.weights_format == config::WeightsFormat::gguf) {
        return load_weights_gguf(cfg, model_cfg, registry, skip_routed_experts);
    }

    bool use_mmap = cfg.model.use_mmap;
    spdlog::info("Loading weights from {} ({}{})", weights_path,
                 use_mmap ? "mmap" : "pread",
                 skip_routed_experts ? ", skip routed experts" : "");

    // Read shard index
    auto shard_index = read_shard_index(model_dir);

    // Open all shards
    LoadedModel model;
    model.shards.reserve(shard_index.shard_files.size());
    for (auto& shard_file : shard_index.shard_files) {
        auto shard_path = model_dir / shard_file;
        spdlog::debug("Opening shard: {}", shard_file);
        model.shards.push_back(SafetensorsReader::open(shard_path, use_mmap));
    }

    spdlog::info("Opened {} shards", model.shards.size());

    // Initialize layer vector
    int num_layers = model_cfg.raw().num_hidden_layers;
    model.layers.resize(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        model.layers[l].layer_idx = l;
        // WP-6: Skip routed expert allocation when prepacked source is active.
        if (model_cfg.is_moe_layer(l) && !skip_routed_experts) {
            model.layers[l].routed_experts.resize(model_cfg.raw().n_routed_experts);
        }
    }

    // Accumulate all tensors
    TensorAccumulator accumulator;
    int total_entries = 0;
    int unrecognized = 0;

    for (auto& shard : model.shards) {
        for (auto& entry : shard.entries()) {
            ++total_entries;

            auto id_opt = parse_hf_name(entry.name);
            if (!id_opt) {
                spdlog::warn("Unrecognized tensor name: {}", entry.name);
                ++unrecognized;
                continue;
            }

            // TD-97a: skip routed expert entries early — avoid accumulate,
            // assemble, tensor_data (pread I/O in non-mmap mode).
            if (skip_routed_experts &&
                id_opt->owner == TensorOwner::routed_expert)
                continue;

            auto key = key_from_id(*id_opt);
            RawTensor raw{
                .data = shard.tensor_data(entry),
                .dtype = entry.dtype,
                .shape = entry.shape,
            };

            accumulator[key].push_back(AccumulatedTensor{id_opt->role, std::move(raw)});
        }
    }

    spdlog::info("Scanned {} tensor entries, {} unrecognized", total_entries, unrecognized);

    // Assemble bundles and place into model
    int validation_warnings = 0;
    for (auto& [key, tensors] : accumulator) {
        auto [bundle, err] = assemble_bundle(key, tensors, model_cfg);
        if (!err.empty()) {
            spdlog::warn("Weight validation: {}", err);
            ++validation_warnings;
            // Still place the bundle — it may be partially valid
        }

        model.total_weight_bytes += bundle.total_bytes();
        ++model.total_tensors_loaded;
        place_bundle(model, std::move(bundle), model_cfg);
    }

    spdlog::info("Assembled {} weight bundles ({:.1f} GB), {} validation warnings",
                 model.total_tensors_loaded,
                 model.total_weight_bytes / (1024.0 * 1024.0 * 1024.0),
                 validation_warnings);

    // Expert packing is deferred to resolve_host_source() (lazy via
    // ensure_expert_packed) for both NVFP4 and FP8. Packing all experts
    // eagerly would exceed system RAM (~356 GB NVFP4, ~654 GB FP8 for V3.2).
    // TD-93a: FP8 was previously packed eagerly here; now lazy like NVFP4.

    // Validate completeness
    auto completeness_errors = validate_completeness(model, model_cfg, registry,
                                                     skip_routed_experts);
    if (!completeness_errors.empty()) {
        std::string msg = "Weight completeness errors:\n";
        for (auto& e : completeness_errors) {
            msg += "  - " + e + "\n";
            spdlog::error("Completeness: {}", e);
        }
        throw std::runtime_error(msg);
    }

    spdlog::info("Weight loading complete: {} layers, {} total bundles{}",
                 model.layers.size(), model.total_tensors_loaded,
                 skip_routed_experts ? " (routed experts skipped)" : "");

    return model;
}

// ── GGUF model expert-type scan (GG-6) ───────────────────────────────────────

GgufModelExpertTypes gguf_expert_types_from_model(const LoadedModel& model) {
    std::optional<GgufKQuantType> gate, up, down;

    auto record = [](std::optional<GgufKQuantType>& slot, const WeightBundle& b,
                     const char* name) {
        auto t = b.gguf_type();
        if (!t) {
            throw std::runtime_error(
                std::string("gguf_expert_types_from_model: ") + name
                + " expert weight has no GGUF k-quant type");
        }
        if (slot && *slot != *t) {
            throw std::runtime_error(
                std::string("gguf_expert_types_from_model: ") + name
                + " expert k-quant type is not uniform across layers/experts "
                  "(engine assumes one type per projection)");
        }
        slot = *t;
    };

    for (const auto& layer : model.layers) {
        for (const auto& experts : layer.routed_experts) {
            for (const auto& b : experts) {
                switch (b.id.component) {
                    case TensorComponent::gate_proj: record(gate, b, "gate"); break;
                    case TensorComponent::up_proj:   record(up, b, "up");     break;
                    case TensorComponent::down_proj: record(down, b, "down"); break;
                    default: break;
                }
            }
        }
    }

    if (!gate || !up || !down) {
        throw std::runtime_error(
            "gguf_expert_types_from_model: no routed experts with GGUF types "
            "found (was skip_routed_experts set, or is this not a GGUF MoE model?)");
    }
    return GgufModelExpertTypes{*gate, *up, *down};
}

GgufModelExpertTypes gguf_expert_types_from_path(const std::string& weights_path,
                                                 bool use_mmap) {
    auto files = resolve_gguf_files(weights_path);
    if (files.empty()) {
        throw std::runtime_error("gguf_expert_types_from_path: no .gguf file at "
                                 + weights_path);
    }

    // GG-9: keep the per-projection MAXIMUM-byte type (upper bound) — a Q5_K_XL
    // mix has different routed k-quants per layer (e.g. one layer's down is Q8_0
    // while most are Q5_K/Q6_K), and the expert-cache slot must hold the largest.
    auto rank = [](GgufKQuantType t) { return gguf::gguf_packed_bytes(1, 256, t); };
    auto upd = [&](std::optional<GgufKQuantType>& slot, GgufKQuantType t) {
        if (!slot || rank(t) > rank(*slot)) slot = t;
    };
    std::optional<GgufKQuantType> gate, up, down;
    for (const auto& f : files) {
        GgufReader r = GgufReader::open(f, use_mmap);
        for (const auto& e : r.entries()) {
            auto id = parse_gguf_name(e.name);
            if (!id || id->owner != TensorOwner::routed_expert) continue;
            if (!e.is_kquant()) continue;
            GgufKQuantType t = e.kquant_type();
            switch (id->component) {
                case TensorComponent::gate_proj: upd(gate, t); break;
                case TensorComponent::up_proj:   upd(up, t);   break;
                case TensorComponent::down_proj: upd(down, t); break;
                default: break;
            }
        }
    }

    if (!gate || !up || !down) {
        throw std::runtime_error(
            "gguf_expert_types_from_path: no stacked routed-expert k-quant "
            "tensors found in " + weights_path);
    }
    return GgufModelExpertTypes{*gate, *up, *down};
}

// ── TD-VOCAB-AUTODETECT: weights-derived vocab width + config resolve ────────

namespace {

// Header-only scan of the safetensors shards for the embedding row count
// (output-head fallback). Sharded models resolve the owning shard via the
// index; single-file models scan model.safetensors.
int64_t vocab_rows_from_safetensors(const config::Config& cfg) {
    namespace fs = std::filesystem;
    fs::path model_dir(cfg.model.weights_path);
    auto index = read_shard_index(model_dir);

    static constexpr const char* kNames[] = {"model.embed_tokens.weight",
                                             "lm_head.weight"};
    for (const char* name : kNames) {
        std::vector<std::string> candidates;
        if (!index.tensor_to_shard.empty()) {
            // Sharded: the index names the owning shard (skip this name if
            // it is absent from the weight_map).
            for (const auto& [tname, sfile] : index.tensor_to_shard)
                if (tname == name) { candidates.push_back(sfile); break; }
        } else {
            candidates = index.shard_files;  // single-file model
        }
        for (const auto& f : candidates) {
            for (const auto& e : SafetensorsReader::read_header(model_dir / f)) {
                if (e.name == name && !e.shape.empty())
                    return e.shape[0];  // [vocab, hidden] row-major
            }
        }
    }
    throw std::runtime_error(
        "detect_weights_vocab_rows: neither model.embed_tokens.weight nor "
        "lm_head.weight found in the safetensors headers under " +
        cfg.model.weights_path);
}

// Header-only scan of the GGUF shard(s). GGUF dims are reversed (dims[0] =
// fastest = hidden columns), so the vocab row count is the LAST dim.
int64_t vocab_rows_from_gguf(const config::Config& cfg) {
    auto files = resolve_gguf_files(cfg.model.weights_path);
    int64_t embd = 0, output = 0;
    for (const auto& f : files) {
        GgufReader r = GgufReader::open(f, cfg.model.use_mmap);
        for (const auto& e : r.entries()) {
            if (e.dims.size() < 2) continue;
            if (e.name == "token_embd.weight") embd = e.dims.back();
            else if (e.name == "output.weight") output = e.dims.back();
        }
        if (embd > 0) break;  // embedding is authoritative; stop early
    }
    if (embd > 0) return embd;
    if (output > 0) return output;
    throw std::runtime_error(
        "detect_weights_vocab_rows: neither token_embd.weight nor "
        "output.weight found in the GGUF tensor headers under " +
        cfg.model.weights_path);
}

}  // namespace

int64_t detect_weights_vocab_rows(const config::Config& cfg) {
    if (cfg.model.weights_path.empty())
        throw std::runtime_error(
            "detect_weights_vocab_rows: model.weights_path is empty");
    if (cfg.model.weights_format == config::WeightsFormat::gguf)
        return vocab_rows_from_gguf(cfg);
    return vocab_rows_from_safetensors(cfg);
}

void resolve_vocab_size(config::Config& cfg) {
    const int64_t rows = detect_weights_vocab_rows(cfg);
    if (rows <= 0)
        throw std::runtime_error(std::format(
            "resolve_vocab_size: weights report a non-positive "
            "embedding/output-head row count ({}) in {}",
            rows, cfg.model.weights_path));

    if (cfg.model.vocab_size == 0) {
        cfg.model.vocab_size = static_cast<int>(rows);
        spdlog::info("model.vocab_size autodetected from weights: {} "
                     "(embedding/output-head rows, padding included; config "
                     "field absent/0)", rows);
        return;
    }
    if (static_cast<int64_t>(cfg.model.vocab_size) != rows) {
        throw std::runtime_error(std::format(
            "model.vocab_size mismatch: config says {} but the weights' "
            "embedding/output-head row count is {} ({}). Weights are ground "
            "truth; a wrong config value silently mis-sizes logits "
            "readbacks, sampling, and guided-decode grammar masks — fix the "
            "config (or set vocab_size to 0 to autodetect).",
            cfg.model.vocab_size, rows, cfg.model.weights_path));
    }
    spdlog::debug("model.vocab_size {} confirmed against the weights", rows);
}

std::optional<GgufModelExpertTypes> gguf_owner_types_from_model(
    const LoadedModel& model, TensorOwner owner) {
    // Larger packed block = larger upper bound. gguf_packed_bytes(1, 256, t)
    // returns one super-block's bytes for any supported type (QK divides 256).
    auto rank = [](GgufKQuantType t) {
        return gguf::gguf_packed_bytes(1, 256, t);
    };
    std::optional<GgufKQuantType> gate, up, down;
    auto upd = [&](std::optional<GgufKQuantType>& slot, GgufKQuantType t) {
        if (!slot || rank(t) > rank(*slot)) slot = t;
    };
    auto scan = [&](const std::vector<WeightBundle>& bs) {
        for (const auto& b : bs) {
            auto t = b.gguf_type();
            if (!t) continue;
            switch (b.id.component) {
                case TensorComponent::gate_proj: upd(gate, *t); break;
                case TensorComponent::up_proj:   upd(up, *t);   break;
                case TensorComponent::down_proj: upd(down, *t); break;
                default: break;
            }
        }
    };
    for (const auto& layer : model.layers) {
        if (owner == TensorOwner::shared_expert) scan(layer.shared_expert);
        else if (owner == TensorOwner::dense_ffn) scan(layer.dense_ffn);
        else if (owner == TensorOwner::routed_expert)
            for (const auto& e : layer.routed_experts) scan(e);
    }
    if (!gate || !up || !down) return std::nullopt;
    return GgufModelExpertTypes{*gate, *up, *down};
}

// ── Lazy expert packing ─────────────────────────────────────────────────────

void ensure_expert_packed(std::vector<WeightBundle>& bundles,
                          const ExpertShape& shape,
                          const Nvfp4InputScaleNorm* norm) {
    if (bundles.empty()) return;
    // Already packed: packed_slot is non-empty.  Covers both allocated packing
    // (owned_buf set) and FP8 zero-copy (owned_buf null, packed_slot spans mmap).
    // TD-82a releases packed_slot after H2D, so this can fire multiple times.
    if (!bundles[0].packed_slot.empty()) return;

    // GGUF k-quants are packed bytes (dtype U8) but distinguished by gguf_type;
    // they pack by verbatim block copy sized by their own per-projection types
    // (pack_gguf_expert reads each bundle's gguf_type directly — GG-10).
    if (bundles[0].weight.gguf_type.has_value()) {
        pack_gguf_expert(bundles, shape);
        return;
    }

    switch (bundles[0].weight.dtype) {
        case SafetensorsDtype::U8:
            pack_nvfp4_expert(bundles, shape, norm);
            break;
        case SafetensorsDtype::F8_E4M3:
        case SafetensorsDtype::F8_E5M2:
            pack_fp8_expert(bundles, shape);
            break;
        default:
            break;  // BF16/F16/F32 — no packing needed
    }
}

void ensure_nvfp4_expert_packed(std::vector<WeightBundle>& bundles,
                                const ExpertShape& shape) {
    ensure_expert_packed(bundles, shape);
}


size_t LoadedModel::release_pinned_host_bytes() {
    size_t freed = 0;
    auto drop_bundle = [&](WeightBundle& b) {
        if (b.owned_buf) {
            freed += b.owned_buf->size();
            b.owned_buf.reset();
        }
        b.weight.data = {};
        for (auto& [role, aux] : b.aux) aux.data = {};
        b.packed_slot = {};
    };
    auto drop_vec = [&](std::vector<WeightBundle>& v) {
        for (auto& b : v) drop_bundle(b);
    };
    auto drop_layer = [&](LayerWeights& lw) {
        drop_vec(lw.attention);
        drop_vec(lw.indexer);
        drop_vec(lw.norms);
        drop_vec(lw.gating);
        drop_vec(lw.shared_expert);
        drop_vec(lw.dense_ffn);
        // routed_experts DELIBERATELY untouched (PrepackedSource path).
    };
    for (auto& lw : layers) drop_layer(lw);
    if (embedding) drop_bundle(*embedding);
    if (output_head) drop_bundle(*output_head);
    if (final_norm) drop_bundle(*final_norm);
    if (mtp) {
        drop_vec(mtp->tensors);
        for (auto& lw : mtp->block_layers) drop_layer(lw);
    }
    // NOTE (round 2b): the GGUF shard mmaps are deliberately NOT
    // MADV_DONTNEED'd — page cache is reclaimable under pressure anyway, and
    // force-dropping it made the NEXT boot re-read ~21 GB at the Gen3 disk
    // ceiling (~6 s). Only the non-reclaimable owned heap is freed here.
    return freed;
}

}  // namespace layerstorm::model
