#include "model/weight_loader/tensor_id.h"

#include <charconv>
#include <string_view>
#include <vector>

namespace layerstorm::model {

// ── String helpers ───────────────────────────────────────────────────────────

std::string_view tensor_component_name(TensorComponent c) {
    switch (c) {
        case TensorComponent::q_a_proj:              return "q_a_proj";
        case TensorComponent::q_a_norm:              return "q_a_norm";
        case TensorComponent::q_b_proj:              return "q_b_proj";
        case TensorComponent::kv_a_proj_with_mqa:    return "kv_a_proj_with_mqa";
        case TensorComponent::kv_a_norm:             return "kv_a_norm";
        case TensorComponent::kv_b_proj:             return "kv_b_proj";
        case TensorComponent::o_proj:                return "o_proj";
        case TensorComponent::indexer_wq_b:          return "indexer_wq_b";
        case TensorComponent::indexer_wk:            return "indexer_wk";
        case TensorComponent::indexer_k_norm_weight: return "indexer_k_norm_weight";
        case TensorComponent::indexer_k_norm_bias:   return "indexer_k_norm_bias";
        case TensorComponent::indexer_weights_proj:  return "indexer_weights_proj";
        case TensorComponent::input_layernorm:       return "input_layernorm";
        case TensorComponent::post_attention_layernorm: return "post_attention_layernorm";
        case TensorComponent::gate_weight:           return "gate_weight";
        case TensorComponent::gate_e_score_correction_bias: return "gate_e_score_correction_bias";
        case TensorComponent::gate_proj:             return "gate_proj";
        case TensorComponent::up_proj:               return "up_proj";
        case TensorComponent::down_proj:             return "down_proj";
        case TensorComponent::embedding:             return "embedding";
        case TensorComponent::output_head:           return "output_head";
        case TensorComponent::final_norm:            return "final_norm";
        case TensorComponent::mtp_enorm:             return "mtp_enorm";
        case TensorComponent::mtp_hnorm:             return "mtp_hnorm";
        case TensorComponent::mtp_eh_proj:           return "mtp_eh_proj";
        case TensorComponent::mtp_shared_head_weight: return "mtp_shared_head_weight";
        case TensorComponent::mtp_shared_head_norm:  return "mtp_shared_head_norm";
        case TensorComponent::mtp_embed_tokens:      return "mtp_embed_tokens";
        case TensorComponent::mla_k_b_split:         return "mla_k_b_split";
        case TensorComponent::mla_v_b_split:         return "mla_v_b_split";
        case TensorComponent::indexer_compressor_wkv:   return "indexer_compressor_wkv";
        case TensorComponent::indexer_compressor_wgate: return "indexer_compressor_wgate";
        case TensorComponent::indexer_compressor_ape:   return "indexer_compressor_ape";
        case TensorComponent::indexer_compressor_norm:  return "indexer_compressor_norm";
        case TensorComponent::o_proj_a:              return "o_proj_a";
        case TensorComponent::o_proj_b:              return "o_proj_b";
        case TensorComponent::attn_sinks:            return "attn_sinks";
        case TensorComponent::compressor_wkv:        return "compressor_wkv";
        case TensorComponent::compressor_wgate:      return "compressor_wgate";
        case TensorComponent::compressor_ape:        return "compressor_ape";
        case TensorComponent::compressor_norm:       return "compressor_norm";
        case TensorComponent::hc_attn_fn:            return "hc_attn_fn";
        case TensorComponent::hc_attn_base:          return "hc_attn_base";
        case TensorComponent::hc_attn_scale:         return "hc_attn_scale";
        case TensorComponent::hc_ffn_fn:             return "hc_ffn_fn";
        case TensorComponent::hc_ffn_base:           return "hc_ffn_base";
        case TensorComponent::hc_ffn_scale:          return "hc_ffn_scale";
        case TensorComponent::gate_tid2eid:          return "gate_tid2eid";
        case TensorComponent::output_hc_fn:          return "output_hc_fn";
        case TensorComponent::output_hc_base:        return "output_hc_base";
        case TensorComponent::output_hc_scale:       return "output_hc_scale";
        case TensorComponent::kda_q_proj:            return "kda_q_proj";
        case TensorComponent::kda_k_proj:            return "kda_k_proj";
        case TensorComponent::kda_v_proj:            return "kda_v_proj";
        case TensorComponent::kda_b_proj:            return "kda_b_proj";
        case TensorComponent::kda_f_a_proj:          return "kda_f_a_proj";
        case TensorComponent::kda_f_b_proj:          return "kda_f_b_proj";
        case TensorComponent::kda_g_a_proj:          return "kda_g_a_proj";
        case TensorComponent::kda_g_b_proj:          return "kda_g_b_proj";
        case TensorComponent::kda_q_conv1d:          return "kda_q_conv1d";
        case TensorComponent::kda_k_conv1d:          return "kda_k_conv1d";
        case TensorComponent::kda_v_conv1d:          return "kda_v_conv1d";
        case TensorComponent::kda_a_log:             return "kda_a_log";
        case TensorComponent::kda_dt_bias:           return "kda_dt_bias";
        case TensorComponent::kda_o_norm:            return "kda_o_norm";
    }
    return "unknown";
}

std::string_view tensor_role_name(TensorRole r) {
    switch (r) {
        case TensorRole::weight:         return "weight";
        case TensorRole::weight_scale:   return "weight_scale";
        case TensorRole::weight_scale_2: return "weight_scale_2";
        case TensorRole::input_scale:    return "input_scale";
        case TensorRole::bias:           return "bias";
    }
    return "unknown";
}

std::string_view tensor_owner_name(TensorOwner o) {
    switch (o) {
        case TensorOwner::attention:     return "attention";
        case TensorOwner::dense_ffn:     return "dense_ffn";
        case TensorOwner::routed_expert: return "routed_expert";
        case TensorOwner::shared_expert: return "shared_expert";
        case TensorOwner::gating:        return "gating";
        case TensorOwner::model_level:   return "model_level";
        case TensorOwner::mtp:           return "mtp";
    }
    return "unknown";
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

namespace {

// Split a string_view on '.' into a vector of segments.
std::vector<std::string_view> split_dot(std::string_view s) {
    std::vector<std::string_view> parts;
    while (!s.empty()) {
        auto pos = s.find('.');
        if (pos == std::string_view::npos) {
            parts.push_back(s);
            break;
        }
        parts.push_back(s.substr(0, pos));
        s.remove_prefix(pos + 1);
    }
    return parts;
}

// Parse an integer from a string_view. Returns nullopt on failure.
std::optional<int> parse_int(std::string_view s) {
    int val = 0;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
    if (ec != std::errc{} || ptr != s.data() + s.size()) return std::nullopt;
    return val;
}

// Determine TensorRole from the last segment(s) of a tensor name.
// Returns the role and how many trailing segments it consumed.
struct RoleParse {
    TensorRole role;
    int segments_consumed;  // how many trailing segments form the role
};

std::optional<RoleParse> parse_role_suffix(const std::vector<std::string_view>& parts) {
    if (parts.empty()) return std::nullopt;

    auto last = parts.back();

    // Two-segment suffixes: "weight_scale", "weight_scale_2", "input_scale"
    // In the safetensors format these are single dot-separated names:
    //   *.weight_scale, *.weight_scale_2, *.input_scale
    // But since we split on '.', "weight_scale" is one segment with underscore.

    if (last == "weight") return RoleParse{TensorRole::weight, 1};
    if (last == "weight_scale") return RoleParse{TensorRole::weight_scale, 1};
    // GF3.3: DeepSeek/GLM native-FP8 checkpoints name the blockwise
    // dequant multiplier `weight_scale_inv` (historical name; it IS the
    // multiplier w_bf16 = w_fp8 * scale — same semantics our FP8 handler
    // and kernels give TensorRole::weight_scale). Map it onto the same role.
    if (last == "weight_scale_inv") return RoleParse{TensorRole::weight_scale, 1};
    if (last == "weight_scale_2") return RoleParse{TensorRole::weight_scale_2, 1};
    if (last == "input_scale") return RoleParse{TensorRole::input_scale, 1};
    if (last == "bias") return RoleParse{TensorRole::bias, 1};
    if (last == "e_score_correction_bias") return RoleParse{TensorRole::bias, 1};

    return std::nullopt;
}

}  // namespace

// ── parse_hf_name ────────────────────────────────────────────────────────────

std::optional<TensorId> parse_hf_name(std::string_view name) {
    auto parts = split_dot(name);
    if (parts.empty()) return std::nullopt;

    // ── Namespace normalization (GF3.3, MODELINFO §1) ──
    // Multimodal checkpoints (glm5_next) wrap the text model: tensors live
    // under `model.language_model.layers.N.*` / `model.language_model.
    // embed_tokens` etc. Collapse the wrapper segment so every downstream
    // pattern matches both namespaces. (`model.visual.*` — the vision tower —
    // stays unrecognized here by design: GF3.14 loads it; load_weights
    // classifies and skips it without a warning.)
    if (parts.size() >= 3 && parts[0] == "model" && parts[1] == "language_model")
        parts.erase(parts.begin() + 1);

    // ── Model-level tensors (no "layers" prefix) ──

    // "model.embed_tokens.weight"
    if (parts.size() == 3 && parts[0] == "model" && parts[1] == "embed_tokens" && parts[2] == "weight") {
        return TensorId{TensorComponent::embedding, TensorRole::weight,
                        TensorOwner::model_level, -1, -1};
    }
    // "model.norm.weight"
    if (parts.size() == 3 && parts[0] == "model" && parts[1] == "norm" && parts[2] == "weight") {
        return TensorId{TensorComponent::final_norm, TensorRole::weight,
                        TensorOwner::model_level, -1, -1};
    }
    // "lm_head.weight"
    if (parts.size() == 2 && parts[0] == "lm_head" && parts[1] == "weight") {
        return TensorId{TensorComponent::output_head, TensorRole::weight,
                        TensorOwner::model_level, -1, -1};
    }

    // ── Layer tensors: "model.layers.{l}...." ──

    if (parts.size() < 4 || parts[0] != "model" || parts[1] != "layers") return std::nullopt;

    auto layer_opt = parse_int(parts[2]);
    if (!layer_opt) return std::nullopt;
    int layer_idx = *layer_opt;

    // Remaining segments after "model.layers.{l}"
    // parts[3..] is the component path + role suffix
    size_t path_start = 3;

    // Parse role from the trailing segment(s). Some glm5_next tensors carry
    // NO role suffix at all (bare parameter names in the checkpoint):
    // self_attn.A_log, self_attn.dt_bias, hc_{attn,ffn}_{base,fn,scale},
    // indexer.index_kpool_compress_{gate,ape}. For those the full remaining
    // path IS the component and the role is `weight`.
    auto role_parse = parse_role_suffix(parts);
    const bool bare_name = !role_parse.has_value();
    if (bare_name) role_parse = RoleParse{TensorRole::weight, 0};

    // Path segments = everything between layer index and role suffix
    size_t path_end = parts.size() - role_parse->segments_consumed;
    if (path_end <= path_start) return std::nullopt;

    // Build a path key by joining the middle segments with '.'
    // This avoids building a string in most cases via direct matching.

    // Helper: match a fixed path
    auto path_is = [&](std::initializer_list<std::string_view> expected) -> bool {
        if (path_end - path_start != expected.size()) return false;
        auto it = expected.begin();
        for (size_t i = path_start; i < path_end; ++i, ++it) {
            if (parts[i] != *it) return false;
        }
        return true;
    };

    // Helper: match path with one integer wildcard, returning the parsed int
    auto path_with_int = [&](std::initializer_list<std::string_view> before,
                             std::initializer_list<std::string_view> after)
        -> std::optional<int> {
        size_t total = before.size() + 1 + after.size();
        if (path_end - path_start != total) return std::nullopt;
        size_t idx = path_start;
        for (auto sv : before) {
            if (parts[idx] != sv) return std::nullopt;
            ++idx;
        }
        auto val = parse_int(parts[idx]);
        if (!val) return std::nullopt;
        ++idx;
        for (auto sv : after) {
            if (parts[idx] != sv) return std::nullopt;
            ++idx;
        }
        return val;
    };

    TensorRole role = role_parse->role;

    // ── self_attn paths ──

    if (path_is({"self_attn", "q_a_proj"}))
        return TensorId{TensorComponent::q_a_proj, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "q_a_layernorm"}))
        return TensorId{TensorComponent::q_a_norm, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "q_b_proj"}))
        return TensorId{TensorComponent::q_b_proj, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "kv_a_proj_with_mqa"}))
        return TensorId{TensorComponent::kv_a_proj_with_mqa, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "kv_a_layernorm"}))
        return TensorId{TensorComponent::kv_a_norm, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "kv_b_proj"}))
        return TensorId{TensorComponent::kv_b_proj, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "o_proj"}))
        return TensorId{TensorComponent::o_proj, role, TensorOwner::attention, layer_idx, -1};

    // ── glm5_next KDA linear attention (GF3.3; MODELINFO §3a) ──
    // Pure name→component mappings (no model gate — this parser maps names).
    // o_proj of a KDA layer uses the same `self_attn.o_proj` name as MLA and
    // maps to the shared TensorComponent::o_proj above (row-parallel there
    // and here).

    if (path_is({"self_attn", "q_proj"}))
        return TensorId{TensorComponent::kda_q_proj, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "k_proj"}))
        return TensorId{TensorComponent::kda_k_proj, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "v_proj"}))
        return TensorId{TensorComponent::kda_v_proj, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "b_proj"}))
        return TensorId{TensorComponent::kda_b_proj, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "f_a_proj"}))
        return TensorId{TensorComponent::kda_f_a_proj, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "f_b_proj"}))
        return TensorId{TensorComponent::kda_f_b_proj, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "g_a_proj"}))
        return TensorId{TensorComponent::kda_g_a_proj, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "g_b_proj"}))
        return TensorId{TensorComponent::kda_g_b_proj, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "q_conv1d"}))
        return TensorId{TensorComponent::kda_q_conv1d, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "k_conv1d"}))
        return TensorId{TensorComponent::kda_k_conv1d, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "v_conv1d"}))
        return TensorId{TensorComponent::kda_v_conv1d, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "o_norm"}))
        return TensorId{TensorComponent::kda_o_norm, role, TensorOwner::attention, layer_idx, -1};
    if (bare_name) {
        // Bare parameter names (no .weight suffix in the checkpoint).
        if (path_is({"self_attn", "A_log"}))
            return TensorId{TensorComponent::kda_a_log, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
        if (path_is({"self_attn", "dt_bias"}))
            return TensorId{TensorComponent::kda_dt_bias, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
        // mHC hyper-connection weights (glm5_next safetensors; V4 loads the
        // GGUF names below — same components).
        if (path_is({"hc_attn_base"}))
            return TensorId{TensorComponent::hc_attn_base, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
        if (path_is({"hc_attn_fn"}))
            return TensorId{TensorComponent::hc_attn_fn, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
        if (path_is({"hc_attn_scale"}))
            return TensorId{TensorComponent::hc_attn_scale, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
        if (path_is({"hc_ffn_base"}))
            return TensorId{TensorComponent::hc_ffn_base, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
        if (path_is({"hc_ffn_fn"}))
            return TensorId{TensorComponent::hc_ffn_fn, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
        if (path_is({"hc_ffn_scale"}))
            return TensorId{TensorComponent::hc_ffn_scale, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
        // IndexPool learned 4:1 compression (glm5_next; MODELINFO §3c) —
        // mapped onto the V4 indexer-compressor components (same semantics:
        // a gate projection + an additive positional bias).
        if (path_is({"self_attn", "indexer", "index_kpool_compress_gate"}))
            return TensorId{TensorComponent::indexer_compressor_wgate, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
        if (path_is({"self_attn", "indexer", "index_kpool_compress_ape"}))
            return TensorId{TensorComponent::indexer_compressor_ape, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
        return std::nullopt;
    }

    // ── DSA indexer paths ──

    if (path_is({"self_attn", "indexer", "wq_b"}))
        return TensorId{TensorComponent::indexer_wq_b, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"self_attn", "indexer", "wk"}))
        return TensorId{TensorComponent::indexer_wk, role, TensorOwner::attention, layer_idx, -1};

    // indexer.k_norm.weight / indexer.k_norm.bias
    // After role stripping: path = ["self_attn", "indexer", "k_norm"]
    // But the role suffix already consumed "weight" or "bias"
    if (path_is({"self_attn", "indexer", "k_norm"})) {
        if (role == TensorRole::weight)
            return TensorId{TensorComponent::indexer_k_norm_weight, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
        if (role == TensorRole::bias)
            return TensorId{TensorComponent::indexer_k_norm_bias, TensorRole::bias, TensorOwner::attention, layer_idx, -1};
    }

    if (path_is({"self_attn", "indexer", "weights_proj"}))
        return TensorId{TensorComponent::indexer_weights_proj, role, TensorOwner::attention, layer_idx, -1};

    // ── Layer norms ──

    if (path_is({"input_layernorm"}))
        return TensorId{TensorComponent::input_layernorm, role, TensorOwner::attention, layer_idx, -1};
    // Some HF models use "input_ln" instead
    if (path_is({"input_ln"}))
        return TensorId{TensorComponent::input_layernorm, role, TensorOwner::attention, layer_idx, -1};
    if (path_is({"post_attention_layernorm"}))
        return TensorId{TensorComponent::post_attention_layernorm, role, TensorOwner::attention, layer_idx, -1};

    // ── Gating (router) ──
    // "mlp.gate" with suffix "weight" or "e_score_correction_bias" (parsed as bias)
    if (path_is({"mlp", "gate"})) {
        // The trailing segment was "weight" or "e_score_correction_bias"
        // For "e_score_correction_bias", the role is bias.
        // Check original parts to distinguish gate.weight from gate.e_score_correction_bias
        auto original_last = parts.back();
        if (original_last == "e_score_correction_bias") {
            return TensorId{TensorComponent::gate_e_score_correction_bias, TensorRole::bias,
                            TensorOwner::gating, layer_idx, -1};
        }
        return TensorId{TensorComponent::gate_weight, role, TensorOwner::gating, layer_idx, -1};
    }

    // ── Routed experts: mlp.experts.{e}.{proj} ──

    if (auto eidx = path_with_int({"mlp", "experts"}, {"gate_proj"}))
        return TensorId{TensorComponent::gate_proj, role, TensorOwner::routed_expert, layer_idx, *eidx};
    if (auto eidx = path_with_int({"mlp", "experts"}, {"up_proj"}))
        return TensorId{TensorComponent::up_proj, role, TensorOwner::routed_expert, layer_idx, *eidx};
    if (auto eidx = path_with_int({"mlp", "experts"}, {"down_proj"}))
        return TensorId{TensorComponent::down_proj, role, TensorOwner::routed_expert, layer_idx, *eidx};

    // ── Shared experts: mlp.shared_experts.{proj} ──
    // Note: V3.2 has "mlp.shared_experts.gate_proj" (no index),
    // but some models may have "mlp.shared_experts.0.gate_proj".

    if (path_is({"mlp", "shared_experts", "gate_proj"}))
        return TensorId{TensorComponent::gate_proj, role, TensorOwner::shared_expert, layer_idx, -1};
    if (path_is({"mlp", "shared_experts", "up_proj"}))
        return TensorId{TensorComponent::up_proj, role, TensorOwner::shared_expert, layer_idx, -1};
    if (path_is({"mlp", "shared_experts", "down_proj"}))
        return TensorId{TensorComponent::down_proj, role, TensorOwner::shared_expert, layer_idx, -1};

    // Shared experts with index: mlp.shared_experts.{n}.{proj}
    if (auto _ = path_with_int({"mlp", "shared_experts"}, {"gate_proj"}))
        return TensorId{TensorComponent::gate_proj, role, TensorOwner::shared_expert, layer_idx, -1};
    if (auto _ = path_with_int({"mlp", "shared_experts"}, {"up_proj"}))
        return TensorId{TensorComponent::up_proj, role, TensorOwner::shared_expert, layer_idx, -1};
    if (auto _ = path_with_int({"mlp", "shared_experts"}, {"down_proj"}))
        return TensorId{TensorComponent::down_proj, role, TensorOwner::shared_expert, layer_idx, -1};

    // ── Dense FFN: mlp.{proj} (layers without MoE) ──

    if (path_is({"mlp", "gate_proj"}))
        return TensorId{TensorComponent::gate_proj, role, TensorOwner::dense_ffn, layer_idx, -1};
    if (path_is({"mlp", "up_proj"}))
        return TensorId{TensorComponent::up_proj, role, TensorOwner::dense_ffn, layer_idx, -1};
    if (path_is({"mlp", "down_proj"}))
        return TensorId{TensorComponent::down_proj, role, TensorOwner::dense_ffn, layer_idx, -1};

    // ── MTP-specific tensors ──
    // These appear on layers >= num_hidden_layers (e.g. layer 61 for V3.2)

    if (path_is({"enorm"}))
        return TensorId{TensorComponent::mtp_enorm, role, TensorOwner::mtp, layer_idx, -1};
    if (path_is({"hnorm"}))
        return TensorId{TensorComponent::mtp_hnorm, role, TensorOwner::mtp, layer_idx, -1};
    if (path_is({"eh_proj"}))
        return TensorId{TensorComponent::mtp_eh_proj, role, TensorOwner::mtp, layer_idx, -1};
    if (path_is({"shared_head", "head"}))
        return TensorId{TensorComponent::mtp_shared_head_weight, role, TensorOwner::mtp, layer_idx, -1};
    if (path_is({"shared_head", "norm"}))
        return TensorId{TensorComponent::mtp_shared_head_norm, role, TensorOwner::mtp, layer_idx, -1};
    if (path_is({"embed_tokens"}))
        return TensorId{TensorComponent::mtp_embed_tokens, role, TensorOwner::mtp, layer_idx, -1};

    // ── MTP block layers reuse the same patterns as regular layers ──
    // e.g. model.layers.61.block.self_attn.q_a_proj.weight
    // These are handled by the regular paths above since "block" layers
    // just reuse the layer index. But if there's a "block" prefix we need
    // to recurse. For now, parse "block.X.Y" by re-entering the path matcher.
    // The MTP block tensors use the pattern: model.layers.61.block.{sub_layers_path}
    // We handle this by stripping "block" and re-parsing the remainder.

    // Not recognized
    return std::nullopt;
}

// ── parse_gguf_name ──────────────────────────────────────────────────────────
// Maps the llama.cpp / gguf-py deepseek2 + glm-dsa tensor names onto the SAME
// canonical TensorId enums as parse_hf_name. The GGUF writer appends a
// ".weight"/".bias" suffix to each base name (see ref/llama.cpp/gguf-py).

std::optional<TensorId> parse_gguf_name(std::string_view name) {
    auto parts = split_dot(name);
    if (parts.empty()) return std::nullopt;

    auto role_parse = parse_role_suffix(parts);
    if (!role_parse) {
        // GF3.9: NARROW suffix-less escape. The glm5next GGUF writer emits the
        // KDA per-head decay base as a bare `blk.N.ssm_a` — the ONLY tensor in
        // any GGUF this engine loads that carries no `.weight`/`.bias` role
        // suffix (measured over the 1412 tensors of GLM-5.3-Flash-GGUF
        // UD-Q4_K_XL). Mirrors the HF path's `bare_name` branch for
        // `self_attn.A_log`; the role is normalized to `weight` so the bundle
        // groups as the main tensor. Every OTHER suffix-less name still
        // returns nullopt.
        if (parts.size() == 3 && parts[0] == "blk" && parts[2] == "ssm_a") {
            if (auto l = parse_int(parts[1]))
                return TensorId{TensorComponent::kda_a_log, TensorRole::weight,
                                TensorOwner::attention, *l, -1};
        }
        return std::nullopt;
    }
    TensorRole role = role_parse->role;

    // The "body" is everything before the role suffix (".weight"/".bias").
    size_t body_end = parts.size() - role_parse->segments_consumed;
    if (body_end == 0) return std::nullopt;

    // ── Model-level tensors (no "blk" prefix) ──
    if (parts[0] != "blk") {
        if (body_end == 1) {
            if (parts[0] == "token_embd")
                return TensorId{TensorComponent::embedding, TensorRole::weight,
                                TensorOwner::model_level, -1, -1};
            if (parts[0] == "output")
                return TensorId{TensorComponent::output_head, TensorRole::weight,
                                TensorOwner::model_level, -1, -1};
            if (parts[0] == "output_norm")
                return TensorId{TensorComponent::final_norm, TensorRole::weight,
                                TensorOwner::model_level, -1, -1};
            // DeepSeek-V4 model-level mHC output collapse (V4-2a).
            if (parts[0] == "output_hc_fn")
                return TensorId{TensorComponent::output_hc_fn, TensorRole::weight,
                                TensorOwner::model_level, -1, -1};
            if (parts[0] == "output_hc_base")
                return TensorId{TensorComponent::output_hc_base, TensorRole::weight,
                                TensorOwner::model_level, -1, -1};
            if (parts[0] == "output_hc_scale")
                return TensorId{TensorComponent::output_hc_scale, TensorRole::weight,
                                TensorOwner::model_level, -1, -1};
        }
        return std::nullopt;
    }

    // ── Layer tensors: "blk.{l}.<body>" ──
    if (body_end < 3) return std::nullopt;
    auto layer_opt = parse_int(parts[1]);
    if (!layer_opt) return std::nullopt;
    int layer_idx = *layer_opt;

    // Match the body segments [2, body_end) against a fixed component path.
    auto body_is = [&](std::initializer_list<std::string_view> expected) -> bool {
        if (body_end - 2 != expected.size()) return false;
        auto it = expected.begin();
        for (size_t i = 2; i < body_end; ++i, ++it) {
            if (parts[i] != *it) return false;
        }
        return true;
    };

    // ── Attention projections (MLA) ──
    if (body_is({"attn_q_a"}))
        return TensorId{TensorComponent::q_a_proj, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_q_b"}))
        return TensorId{TensorComponent::q_b_proj, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_kv_a_mqa"}))
        return TensorId{TensorComponent::kv_a_proj_with_mqa, role, TensorOwner::attention, layer_idx, -1};
    // DeepSeek-V4 (V4-2a): `attn_kv` is the single 512-dim MQA-over-latent KV
    // projection (llama.cpp deepseek4 wkv). Same canonical component as the
    // MLA kv_a latent projection; V4 has NO attn_kv_b decompression tensor.
    if (body_is({"attn_kv"}))
        return TensorId{TensorComponent::kv_a_proj_with_mqa, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_kv_b"}))
        return TensorId{TensorComponent::kv_b_proj, role, TensorOwner::attention, layer_idx, -1};
    // GLM-1: split MLA up-projection (llama.cpp MLA-optimized layout). Mapped
    // UNCONDITIONALLY (no model gate — this is a pure name→component mapper, and
    // attn_k_b/attn_v_b are the standard DeepSeek-V2/V3 + GLM MLA-split names).
    // The loader pairs them and assembles a combined BF16 kv_b_proj.
    if (body_is({"attn_k_b"}))
        return TensorId{TensorComponent::mla_k_b_split, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_v_b"}))
        return TensorId{TensorComponent::mla_v_b_split, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_output"}))
        return TensorId{TensorComponent::o_proj, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_q_a_norm"}))
        return TensorId{TensorComponent::q_a_norm, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_kv_a_norm"}))
        return TensorId{TensorComponent::kv_a_norm, role, TensorOwner::attention, layer_idx, -1};

    // ── glm5_next KDA linear attention (GF3.9; MODELINFO §8b) ──
    // The llama.cpp glm5next writer names the KDA anatomy with the `ssm_*`
    // family (it reuses the generic state-space naming slots) plus the BARE
    // MHA projection names for q/k/v. Mapped UNCONDITIONALLY, like every other
    // arm here — parse_gguf_name is a pure name→component mapper with no model
    // gate. COLLISION CAVEAT: `blk.N.attn_{q,k,v}.weight` is the standard
    // llama.cpp MHA convention, so a future dense-MHA architecture would land
    // on kda_{q,k,v}_proj. No architecture this engine loads today emits those
    // bare names (DeepSeek/GLM MLA uses attn_q_a/attn_q_b/attn_kv*; V4 uses
    // attn_kv), so the mapping is unambiguous for the current surface — revisit
    // (thread an arch hint) if a plain-MHA model is ever added.
    // KDA `o_proj` is `blk.N.attn_output.weight` and already maps to the shared
    // TensorComponent::o_proj above (same row-parallel sharding).
    if (body_is({"attn_q"}))
        return TensorId{TensorComponent::kda_q_proj, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_k"}))
        return TensorId{TensorComponent::kda_k_proj, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_v"}))
        return TensorId{TensorComponent::kda_v_proj, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"ssm_beta"}))
        return TensorId{TensorComponent::kda_b_proj, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"ssm_f_a"}))
        return TensorId{TensorComponent::kda_f_a_proj, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"ssm_f_b"}))
        return TensorId{TensorComponent::kda_f_b_proj, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"ssm_g_a"}))
        return TensorId{TensorComponent::kda_g_a_proj, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"ssm_g_b"}))
        return TensorId{TensorComponent::kda_g_b_proj, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"ssm_conv1d_q"}))
        return TensorId{TensorComponent::kda_q_conv1d, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"ssm_conv1d_k"}))
        return TensorId{TensorComponent::kda_k_conv1d, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"ssm_conv1d_v"}))
        return TensorId{TensorComponent::kda_v_conv1d, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"ssm_norm"}))
        return TensorId{TensorComponent::kda_o_norm, role, TensorOwner::attention, layer_idx, -1};
    // `blk.N.ssm_dt.bias` is the per-channel dt bias — a MAIN tensor that
    // happens to be written with the `.bias` suffix. Normalize the role to
    // `weight` (exactly what the HF path does for the bare `self_attn.dt_bias`
    // at the bare_name branch above) so the bundle groups as the logical weight
    // and the glm5_next completeness check finds kda_dt_bias.
    if (body_is({"ssm_dt"}))
        return TensorId{TensorComponent::kda_dt_bias, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
    // `blk.N.ssm_a` (no role suffix) is handled by the escape at the top.

    // ── Layer norms ──
    if (body_is({"attn_norm"}))
        return TensorId{TensorComponent::input_layernorm, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"ffn_norm"}))
        return TensorId{TensorComponent::post_attention_layernorm, role, TensorOwner::attention, layer_idx, -1};

    // ── DSA indexer ──
    if (body_is({"indexer", "attn_q_b"}))
        return TensorId{TensorComponent::indexer_wq_b, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"indexer", "attn_k"}))
        return TensorId{TensorComponent::indexer_wk, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"indexer", "k_norm"})) {
        if (role == TensorRole::bias)
            return TensorId{TensorComponent::indexer_k_norm_bias, TensorRole::bias, TensorOwner::attention, layer_idx, -1};
        return TensorId{TensorComponent::indexer_k_norm_weight, TensorRole::weight, TensorOwner::attention, layer_idx, -1};
    }
    if (body_is({"indexer", "proj"}))
        return TensorId{TensorComponent::indexer_weights_proj, role, TensorOwner::attention, layer_idx, -1};

    // ── DeepSeek-V4 attention families (V4-2a; shapes in DS4_DOSSIER.md §0.3) ──
    if (body_is({"attn_output_a"}))
        return TensorId{TensorComponent::o_proj_a, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_output_b"}))
        return TensorId{TensorComponent::o_proj_b, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_sinks"}))
        return TensorId{TensorComponent::attn_sinks, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_compressor_kv"}))
        return TensorId{TensorComponent::compressor_wkv, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_compressor_gate"}))
        return TensorId{TensorComponent::compressor_wgate, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_compressor_ape"}))
        return TensorId{TensorComponent::compressor_ape, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"attn_compressor_norm"}))
        return TensorId{TensorComponent::compressor_norm, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"indexer_compressor_kv"}))
        return TensorId{TensorComponent::indexer_compressor_wkv, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"indexer_compressor_gate"}))
        return TensorId{TensorComponent::indexer_compressor_wgate, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"indexer_compressor_ape"}))
        return TensorId{TensorComponent::indexer_compressor_ape, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"indexer_compressor_norm"}))
        return TensorId{TensorComponent::indexer_compressor_norm, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"hc_attn_fn"}))
        return TensorId{TensorComponent::hc_attn_fn, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"hc_attn_base"}))
        return TensorId{TensorComponent::hc_attn_base, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"hc_attn_scale"}))
        return TensorId{TensorComponent::hc_attn_scale, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"hc_ffn_fn"}))
        return TensorId{TensorComponent::hc_ffn_fn, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"hc_ffn_base"}))
        return TensorId{TensorComponent::hc_ffn_base, role, TensorOwner::attention, layer_idx, -1};
    if (body_is({"hc_ffn_scale"}))
        return TensorId{TensorComponent::hc_ffn_scale, role, TensorOwner::attention, layer_idx, -1};

    // ── MTP / NextN block tensors (llama.cpp naming: blk.N.nextn.*) ──
    // The MTP block's attention/FFN tensors use the regular blk.N.* names at
    // N >= num_hidden_layers (routed by place_bundle's is_mtp_block); only the
    // MTP-specific projections/norms carry the nextn. prefix.
    if (body_is({"nextn", "eh_proj"}))
        return TensorId{TensorComponent::mtp_eh_proj, role, TensorOwner::mtp, layer_idx, -1};
    if (body_is({"nextn", "embed_tokens"}))
        return TensorId{TensorComponent::mtp_embed_tokens, role, TensorOwner::mtp, layer_idx, -1};
    if (body_is({"nextn", "enorm"}))
        return TensorId{TensorComponent::mtp_enorm, role, TensorOwner::mtp, layer_idx, -1};
    if (body_is({"nextn", "hnorm"}))
        return TensorId{TensorComponent::mtp_hnorm, role, TensorOwner::mtp, layer_idx, -1};
    if (body_is({"nextn", "shared_head_head"}))
        return TensorId{TensorComponent::mtp_shared_head_weight, role, TensorOwner::mtp, layer_idx, -1};
    if (body_is({"nextn", "shared_head_norm"}))
        return TensorId{TensorComponent::mtp_shared_head_norm, role, TensorOwner::mtp, layer_idx, -1};

    // ── Gating (router) ──
    if (body_is({"ffn_gate_inp"}))
        return TensorId{TensorComponent::gate_weight, role, TensorOwner::gating, layer_idx, -1};
    if (body_is({"exp_probs_b"}))
        return TensorId{TensorComponent::gate_e_score_correction_bias, TensorRole::bias,
                        TensorOwner::gating, layer_idx, -1};
    // DeepSeek-V4 hash-layer routing table (V4-2a): I32 [num_experts_per_tok,
    // vocab] token-id → expert-id lookup; present only on hash layers.
    if (body_is({"ffn_gate_tid2eid"}))
        return TensorId{TensorComponent::gate_tid2eid, role, TensorOwner::gating, layer_idx, -1};

    // ── Routed experts (STACKED 3D: all experts in one tensor, expert_idx=-1) ──
    if (body_is({"ffn_gate_exps"}))
        return TensorId{TensorComponent::gate_proj, role, TensorOwner::routed_expert, layer_idx, -1};
    if (body_is({"ffn_up_exps"}))
        return TensorId{TensorComponent::up_proj, role, TensorOwner::routed_expert, layer_idx, -1};
    if (body_is({"ffn_down_exps"}))
        return TensorId{TensorComponent::down_proj, role, TensorOwner::routed_expert, layer_idx, -1};

    // ── Shared experts ──
    if (body_is({"ffn_gate_shexp"}))
        return TensorId{TensorComponent::gate_proj, role, TensorOwner::shared_expert, layer_idx, -1};
    if (body_is({"ffn_up_shexp"}))
        return TensorId{TensorComponent::up_proj, role, TensorOwner::shared_expert, layer_idx, -1};
    if (body_is({"ffn_down_shexp"}))
        return TensorId{TensorComponent::down_proj, role, TensorOwner::shared_expert, layer_idx, -1};

    // ── Dense FFN (layers without MoE) ──
    if (body_is({"ffn_gate"}))
        return TensorId{TensorComponent::gate_proj, role, TensorOwner::dense_ffn, layer_idx, -1};
    if (body_is({"ffn_up"}))
        return TensorId{TensorComponent::up_proj, role, TensorOwner::dense_ffn, layer_idx, -1};
    if (body_is({"ffn_down"}))
        return TensorId{TensorComponent::down_proj, role, TensorOwner::dense_ffn, layer_idx, -1};

    // Not recognized
    return std::nullopt;
}

}  // namespace layerstorm::model
