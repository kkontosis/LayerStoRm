#include "model/pinned_upload_plan.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "config/config_parser.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/pinned_region_layout.h"
#include "model/quantization/gguf_kquant.h"
#include "model/quantization/quant_interface.h"
#include "model/weight_loader/tp_weight_sharder.h"
#include "model/weight_loader/weight_loader.h"

namespace layerstorm::model {

// ── Lookup helpers ──────────────────────────────────────────────────────────

std::span<const PinnedSlot> PinnedUploadPlan::slots_for_layer(int layer_idx) const {
    if (layer_idx < 0 || slots.empty()) return {};
    auto it = std::find_if(slots.begin(), slots.end(),
        [layer_idx](const PinnedSlot& s) { return s.layer_idx == layer_idx; });
    if (it == slots.end()) return {};
    auto first = it;
    while (it != slots.end() && it->layer_idx == layer_idx) ++it;
    return {&*first, static_cast<size_t>(it - first)};
}

const PinnedSlot* PinnedUploadPlan::find(PinnedComponent comp, int layer_idx) const {
    for (const auto& s : slots)
        if (s.component == comp && s.layer_idx == layer_idx) return &s;
    return nullptr;
}

// ── Component names ─────────────────────────────────────────────────────────

std::string_view pinned_component_name(PinnedComponent c) {
    switch (c) {
        case PinnedComponent::embedding:          return "embedding";
        case PinnedComponent::output_head_weight:  return "output_head_weight";
        case PinnedComponent::output_head_bias:    return "output_head_bias";
        case PinnedComponent::attention:           return "attention";
        case PinnedComponent::layer_norm:          return "layer_norm";
        case PinnedComponent::gating_weight:       return "gating_weight";
        case PinnedComponent::gating_bias:         return "gating_bias";
        case PinnedComponent::shared_expert_gate:         return "shared_expert_gate";
        case PinnedComponent::shared_expert_up:           return "shared_expert_up";
        case PinnedComponent::shared_expert_gate_scales:  return "shared_expert_gate_scales";
        case PinnedComponent::shared_expert_up_scales:    return "shared_expert_up_scales";
        case PinnedComponent::shared_expert_gate_scalar:  return "shared_expert_gate_scalar";
        case PinnedComponent::shared_expert_up_scalar:    return "shared_expert_up_scalar";
        case PinnedComponent::shared_expert_down:         return "shared_expert_down";
        case PinnedComponent::dense_ffn_gate:          return "dense_ffn_gate";
        case PinnedComponent::dense_ffn_up:            return "dense_ffn_up";
        case PinnedComponent::dense_ffn_gate_scales:   return "dense_ffn_gate_scales";
        case PinnedComponent::dense_ffn_up_scales:     return "dense_ffn_up_scales";
        case PinnedComponent::dense_ffn_gate_scalar:   return "dense_ffn_gate_scalar";
        case PinnedComponent::dense_ffn_up_scalar:     return "dense_ffn_up_scalar";
        case PinnedComponent::dense_ffn_down:          return "dense_ffn_down";
        case PinnedComponent::final_norm:          return "final_norm";
        case PinnedComponent::gating_hash_table:   return "gating_hash_table";
        case PinnedComponent::output_hc:           return "output_hc";
        case PinnedComponent::mtp_embed_tokens:        return "mtp_embed_tokens";
        case PinnedComponent::mtp_eh_proj:             return "mtp_eh_proj";
        case PinnedComponent::mtp_enorm:               return "mtp_enorm";
        case PinnedComponent::mtp_hnorm:               return "mtp_hnorm";
        case PinnedComponent::mtp_shared_head_weight:  return "mtp_shared_head_weight";
        case PinnedComponent::mtp_shared_head_norm:    return "mtp_shared_head_norm";
    }
    return "unknown";
}

// ── Plan builder ────────────────────────────────────────────────────────────

PinnedUploadPlan build_upload_plan(
    const ModelConfig& model_cfg,
    const config::Config& cfg,
    const QuantInterface& expert_quant,
    int tp_degree,
    int rank,
    const GgufModelExpertTypes* gguf_shared_types,
    const GgufModelExpertTypes* gguf_dense_types) {

    (void)rank;

    const auto& m = model_cfg.raw();
    const auto& pin = cfg.memory.pinned_layers;
    const auto& tm = cfg.memory.tp_mode_per_layer;

    PinnedUploadPlan plan;
    int tp = std::max(1, tp_degree);
    int64_t offset = 0;
    int next_group = 1;

    auto resolve_tp = [tp](int mode) -> int {
        return (mode > 0) ? mode : tp;
    };
    int embed_tp   = resolve_tp(tm.embedding);
    int outhead_tp = resolve_tp(tm.output_head);
    int shared_tp  = resolve_tp(tm.shared_expert);
    int dense_tp   = resolve_tp(tm.pinned_dense_ffn);

    auto wq = cfg.quantization.weights;
    double gate_bpe = bytes_per_element(cfg.quantization.gating_compute);
    bool include_kv_b = has_kv_b_in_checkpoint(cfg);

    // V4 (V4-3a): attention anatomy varies by compress_ratios[l]; the MLA
    // AttentionDims below are meaningless for V4 (its config carries inert MLA
    // schema defaults) and are bypassed per layer.
    const bool v4 = model_cfg.is_v4();

    AttentionDims attn_dims = compute_attn_dims(m);

    auto emit = [&](PinnedComponent comp, int layer, int64_t size, int group = 0) {
        plan.slots.push_back({comp, layer, offset, size, group});
        offset += size;
    };

    auto is_pinned = [](const config::LayerPinSpec& spec, int l) {
        if (auto* s = std::get_if<std::string>(&spec))
            return *s == "all";
        auto& v = std::get<std::vector<int>>(spec);
        return std::find(v.begin(), v.end(), l) != v.end();
    };

    // ── 1. Embedding ────────────────────────────────────────────────────────
    if (pin.embedding) {
        int64_t embed_total = static_cast<int64_t>(m.vocab_size) * m.hidden_size *
                              kEmbeddingBytesPerElement;
        emit(PinnedComponent::embedding, -1, embed_total / embed_tp);
    }

    // ── 2. Output head ──────────────────────────────────────────────────────
    if (pin.output_head) {
        int64_t oh_weight = static_cast<int64_t>(m.vocab_size) * m.hidden_size *
                            kEmbeddingBytesPerElement;
        emit(PinnedComponent::output_head_weight, -1, oh_weight / outhead_tp);

        if (has_output_head_bias(cfg)) {
            int64_t oh_bias = static_cast<int64_t>(m.hidden_size) * kEmbeddingBytesPerElement;
            emit(PinnedComponent::output_head_bias, -1, oh_bias / outhead_tp);
        }
    }

    // ── 3. Per hidden layer ─────────────────────────────────────────────────
    int64_t attn_per_layer = attention_layer_bytes(attn_dims, wq, include_kv_b, tp);
    // Layer norms (input + post-attention): F32→BF16 during upload (TD-73c).
    int64_t per_layer_norm_bytes = static_cast<int64_t>(2 * m.hidden_size) * 2;

    int64_t gate_weight_bytes = static_cast<int64_t>(m.hidden_size) * m.n_routed_experts *
                                gate_bpe;
    int64_t gate_bias_bytes = static_cast<int64_t>(m.n_routed_experts) * 4;

    int64_t rep_per_proj = expert_quant.replicated_bytes_per_projection();

    // GG-9: per-projection byte count, optionally overriding the routed
    // expert_quant types with an owner's own GGUF k-quant types (shared/dense in
    // a mixed "XL" GGUF). Mirrors GgufQuantInterface::projection_nk: gate/up use
    // n=intermediate,k=hidden; down uses n=hidden,k=intermediate.
    auto proj_bytes = [&](const ExpertShape& sh, Projection p,
                          const GgufModelExpertTypes* ov) -> int64_t {
        // V4: shared-expert / dense-FFN tensors are BF16-NATIVE in the GGUF
        // (census §0.3; requant gated off for V4).  The GG-9 Q8_0 "upper
        // bound" (1.0625 B/elem) would UNDER-size them (BF16 = 2 B/elem) and
        // overflow at upload — size exactly at BF16 instead.
        if (v4) {
            const int64_t n = (p == Projection::down) ? sh.hidden_size
                                                      : sh.intermediate_size;
            const int64_t k = (p == Projection::down) ? sh.intermediate_size
                                                      : sh.hidden_size;
            return n * k * static_cast<int64_t>(kBf16BytesPerElement);
        }
        if (ov && wq == config::WeightQuant::gguf) {
            const int64_t n = (p == Projection::down) ? sh.hidden_size
                                                      : sh.intermediate_size;
            const int64_t k = (p == Projection::down) ? sh.intermediate_size
                                                      : sh.hidden_size;
            const GgufKQuantType t = (p == Projection::gate) ? ov->gate
                                   : (p == Projection::up)   ? ov->up
                                                             : ov->down;
            return gguf::gguf_packed_bytes(n, k, t);
        }
        return expert_quant.bytes_per_projection(sh, p);
    };

    ExpertShape moe_shape{m.hidden_size, m.moe_intermediate_size};
    int64_t se_gate_bytes = proj_bytes(moe_shape, Projection::gate, gguf_shared_types);
    int64_t se_up_bytes   = proj_bytes(moe_shape, Projection::up,   gguf_shared_types);
    int64_t se_down_bytes = proj_bytes(moe_shape, Projection::down, gguf_shared_types);
    int64_t se_gate_sharded = ((se_gate_bytes - rep_per_proj) / shared_tp + rep_per_proj) *
                              m.n_shared_experts;
    int64_t se_up_sharded = ((se_up_bytes - rep_per_proj) / shared_tp + rep_per_proj) *
                            m.n_shared_experts;
    int64_t se_down_sharded = ((se_down_bytes - rep_per_proj) / shared_tp + rep_per_proj) *
                              m.n_shared_experts;

    const bool has_se_subcomponents = (rep_per_proj > 0);
    int64_t se_gate_weight_only = 0, se_up_weight_only = 0;
    int64_t se_gate_scale_only = 0, se_up_scale_only = 0;
    int64_t se_gate_scalar_only = 0, se_up_scalar_only = 0;
    if (has_se_subcomponents) {
        int64_t gate_scale_unsharded = (moe_shape.gate_params() + 15) / 16;
        int64_t up_scale_unsharded = (moe_shape.up_params() + 15) / 16;
        se_gate_scale_only = gate_scale_unsharded / shared_tp * m.n_shared_experts;
        se_up_scale_only = up_scale_unsharded / shared_tp * m.n_shared_experts;
        se_gate_scalar_only = rep_per_proj * m.n_shared_experts;
        se_up_scalar_only = rep_per_proj * m.n_shared_experts;
        se_gate_weight_only = se_gate_sharded - se_gate_scale_only - se_gate_scalar_only;
        se_up_weight_only = se_up_sharded - se_up_scale_only - se_up_scalar_only;
    }

    ExpertShape dense_shape{m.hidden_size, m.intermediate_size};
    int64_t dense_gate = proj_bytes(dense_shape, Projection::gate, gguf_dense_types);
    int64_t dense_up   = proj_bytes(dense_shape, Projection::up,   gguf_dense_types);
    int64_t dense_down = proj_bytes(dense_shape, Projection::down, gguf_dense_types);
    int64_t dense_gate_sharded = (dense_gate - rep_per_proj) / dense_tp + rep_per_proj;
    int64_t dense_up_sharded = (dense_up - rep_per_proj) / dense_tp + rep_per_proj;
    int64_t dense_down_sharded = (dense_down - rep_per_proj) / dense_tp + rep_per_proj;

    // TD-62p: dense FFN sub-component sizes for NVFP4 (same pattern as shared expert).
    const bool has_dense_subcomponents = (rep_per_proj > 0);
    int64_t dense_gate_weight_only = 0, dense_up_weight_only = 0;
    int64_t dense_gate_scale_only = 0, dense_up_scale_only = 0;
    int64_t dense_gate_scalar_only = 0, dense_up_scalar_only = 0;
    if (has_dense_subcomponents) {
        int64_t dg_scale_unsharded = (dense_shape.gate_params() + 15) / 16;
        int64_t du_scale_unsharded = (dense_shape.up_params() + 15) / 16;
        dense_gate_scale_only = dg_scale_unsharded / dense_tp;
        dense_up_scale_only = du_scale_unsharded / dense_tp;
        dense_gate_scalar_only = rep_per_proj;
        dense_up_scalar_only = rep_per_proj;
        dense_gate_weight_only = dense_gate_sharded - dense_gate_scale_only - dense_gate_scalar_only;
        dense_up_weight_only = dense_up_sharded - dense_up_scale_only - dense_up_scalar_only;
    }

    for (int l = 0; l < m.num_hidden_layers; ++l) {
        bool is_moe = model_cfg.is_moe_layer(l);

        if (is_pinned(pin.attention, l)) {
            // V4: per-layer size from the layer's attention type (V4-3a).
            const int64_t attn_bytes = v4
                ? v4_attention_layer_bytes(
                      m, model_cfg.attention_type_for_layer(l), tp)
                : attn_per_layer;
            emit(PinnedComponent::attention, l, attn_bytes);
            emit(PinnedComponent::layer_norm, l, per_layer_norm_bytes);
        }

        if (is_moe && is_pinned(pin.gating, l)) {
            emit(PinnedComponent::gating_weight, l, gate_weight_bytes);
            if (v4 && model_cfg.is_hash_layer(l)) {
                // Hash layers route by ffn_gate_tid2eid[token_id] (I32) and
                // carry NO exp_probs_b bias (dossier §2.4).
                emit(PinnedComponent::gating_hash_table, l,
                     v4_hash_gating_table_bytes(m));
            } else {
                emit(PinnedComponent::gating_bias, l, gate_bias_bytes);
            }
        }

        if (is_moe) {
            if (has_se_subcomponents) {
                int wg = next_group++;
                int sg = next_group++;
                emit(PinnedComponent::shared_expert_gate, l, se_gate_weight_only, wg);
                emit(PinnedComponent::shared_expert_up, l, se_up_weight_only, wg);
                emit(PinnedComponent::shared_expert_gate_scales, l, se_gate_scale_only, sg);
                emit(PinnedComponent::shared_expert_up_scales, l, se_up_scale_only, sg);
                emit(PinnedComponent::shared_expert_gate_scalar, l, se_gate_scalar_only);
                emit(PinnedComponent::shared_expert_up_scalar, l, se_up_scalar_only);
                emit(PinnedComponent::shared_expert_down, l, se_down_sharded);
            } else {
                int group = next_group++;
                emit(PinnedComponent::shared_expert_gate, l, se_gate_sharded, group);
                emit(PinnedComponent::shared_expert_up, l, se_up_sharded, group);
                emit(PinnedComponent::shared_expert_down, l, se_down_sharded);
            }
        }

        if (!is_moe) {
            bool ffn_pinned = std::find(pin.dense_ffn_layers.begin(),
                                         pin.dense_ffn_layers.end(), l)
                              != pin.dense_ffn_layers.end();
            if (ffn_pinned) {
                if (has_dense_subcomponents) {
                    // TD-62p: separate weight and scale sub-slots so gate+up weights
                    // are contiguous in memory (required by grouped GEMM B_base layout).
                    int wg = next_group++;
                    int sg = next_group++;
                    emit(PinnedComponent::dense_ffn_gate, l, dense_gate_weight_only, wg);
                    emit(PinnedComponent::dense_ffn_up, l, dense_up_weight_only, wg);
                    emit(PinnedComponent::dense_ffn_gate_scales, l, dense_gate_scale_only, sg);
                    emit(PinnedComponent::dense_ffn_up_scales, l, dense_up_scale_only, sg);
                    emit(PinnedComponent::dense_ffn_gate_scalar, l, dense_gate_scalar_only);
                    emit(PinnedComponent::dense_ffn_up_scalar, l, dense_up_scalar_only);
                    emit(PinnedComponent::dense_ffn_down, l, dense_down_sharded);
                } else {
                    int group = next_group++;
                    emit(PinnedComponent::dense_ffn_gate, l, dense_gate_sharded, group);
                    emit(PinnedComponent::dense_ffn_up, l, dense_up_sharded, group);
                    emit(PinnedComponent::dense_ffn_down, l, dense_down_sharded);
                }
            }
        }
    }

    // ── 4. Final norm (F32→BF16 during upload, TD-73c) ────────
    emit(PinnedComponent::final_norm, -1,
         static_cast<int64_t>(m.hidden_size) * 2);

    // ── 4b. V4 mHC head collapse weights (model-level, F32) ──────────────
    if (model_cfg.has_mhc()) {
        emit(PinnedComponent::output_hc, -1, v4_output_hc_bytes(m));
    }

    // ── 5. MTP block layers ─────────────────────────────────────────────────
    // MTP block layers are full transformer blocks (attention + MoE + norms +
    // MTP-specific tensors: embed_tokens, eh_proj, enorm, hnorm, shared_head).
    // V4: the GGUF artifact carries NO nextn/mtp tensors (dossier §0.1) — the
    // embedded draft is sourced from the official safetensors via the dspark
    // checkpoint path (ticket J), never from pinned MTP slots.  Emitting slots
    // here would reserve ~1.2 GB of dead VRAM per rank.
    if (m.num_nextn_predict_layers > 0 && !v4) {
        bool mtp_has_indexer = model_cfg.has_dsa();
        // V3.2 NVFP4 checkpoints store MTP attention o_proj as BF16 (not NVFP4),
        // unlike regular layers where o_proj is quantized to NVFP4.
        bool mtp_bf16_oproj = (wq == config::WeightQuant::nvfp4);
        int64_t mtp_attn = attention_layer_bytes(attn_dims, wq, include_kv_b, tp,
                                                  mtp_has_indexer, mtp_bf16_oproj);

        // MTP-specific tensor sizes
        int64_t mtp_embed_bytes = static_cast<int64_t>(m.vocab_size) * m.hidden_size *
                                  kBf16BytesPerElement / embed_tp;
        int64_t mtp_shared_head_w_bytes = static_cast<int64_t>(m.vocab_size) * m.hidden_size *
                                          kBf16BytesPerElement / outhead_tp;
        int64_t mtp_shared_head_n_bytes = static_cast<int64_t>(m.hidden_size) * 2;  // BF16 (TD-74b)
        // eh_proj: [hidden_size, 2*hidden_size] — projects concat(embed, hidden)
        int64_t mtp_eh_proj_bytes = static_cast<int64_t>(m.hidden_size) * 2 *
                                    m.hidden_size * kBf16BytesPerElement / tp;
        int64_t mtp_norm_bytes = static_cast<int64_t>(m.hidden_size) * 2;  // BF16 (enorm, hnorm) (TD-74b)

        for (int mi = 0; mi < m.num_nextn_predict_layers; ++mi) {
            int mtp_layer = m.num_hidden_layers + mi;

            // 5a. Attention + layer norms
            emit(PinnedComponent::attention, mtp_layer, mtp_attn);
            emit(PinnedComponent::layer_norm, mtp_layer, per_layer_norm_bytes);

            // 5b. Gating (MTP blocks are MoE layers)
            emit(PinnedComponent::gating_weight, mtp_layer, gate_weight_bytes);
            emit(PinnedComponent::gating_bias, mtp_layer, gate_bias_bytes);

            // 5c. Shared expert
            // V3.2 NVFP4 checkpoints store MTP shared expert as BF16 (not NVFP4).
            if (mtp_bf16_oproj) {
                // BF16 shared expert: no scales, no sub-components
                int64_t mtp_se_gate = static_cast<int64_t>(m.moe_intermediate_size) *
                                      m.hidden_size * kBf16BytesPerElement /
                                      shared_tp * m.n_shared_experts;
                int64_t mtp_se_up = mtp_se_gate;
                int64_t mtp_se_down = static_cast<int64_t>(m.hidden_size) *
                                      m.moe_intermediate_size * kBf16BytesPerElement /
                                      shared_tp * m.n_shared_experts;
                int group = next_group++;
                emit(PinnedComponent::shared_expert_gate, mtp_layer, mtp_se_gate, group);
                emit(PinnedComponent::shared_expert_up, mtp_layer, mtp_se_up, group);
                emit(PinnedComponent::shared_expert_down, mtp_layer, mtp_se_down);
            } else if (has_se_subcomponents) {
                int wg = next_group++;
                int sg = next_group++;
                emit(PinnedComponent::shared_expert_gate, mtp_layer, se_gate_weight_only, wg);
                emit(PinnedComponent::shared_expert_up, mtp_layer, se_up_weight_only, wg);
                emit(PinnedComponent::shared_expert_gate_scales, mtp_layer, se_gate_scale_only, sg);
                emit(PinnedComponent::shared_expert_up_scales, mtp_layer, se_up_scale_only, sg);
                emit(PinnedComponent::shared_expert_gate_scalar, mtp_layer, se_gate_scalar_only);
                emit(PinnedComponent::shared_expert_up_scalar, mtp_layer, se_up_scalar_only);
                emit(PinnedComponent::shared_expert_down, mtp_layer, se_down_sharded);
            } else {
                int group = next_group++;
                emit(PinnedComponent::shared_expert_gate, mtp_layer, se_gate_sharded, group);
                emit(PinnedComponent::shared_expert_up, mtp_layer, se_up_sharded, group);
                emit(PinnedComponent::shared_expert_down, mtp_layer, se_down_sharded);
            }

            // 5d. MTP-specific tensors
            emit(PinnedComponent::mtp_embed_tokens, mtp_layer, mtp_embed_bytes);
            emit(PinnedComponent::mtp_shared_head_weight, mtp_layer, mtp_shared_head_w_bytes);
            emit(PinnedComponent::mtp_shared_head_norm, mtp_layer, mtp_shared_head_n_bytes);
            emit(PinnedComponent::mtp_eh_proj, mtp_layer, mtp_eh_proj_bytes);
            emit(PinnedComponent::mtp_enorm, mtp_layer, mtp_norm_bytes);
            emit(PinnedComponent::mtp_hnorm, mtp_layer, mtp_norm_bytes);
        }
    }

    plan.total_bytes = offset;
    return plan;
}

// ── Post-load plan validation ──────────────────────────────────────────────

void validate_plan(
    const PinnedUploadPlan& plan,
    const LoadedModel& loaded,
    const ModelConfig& model_cfg,
    const config::Config& cfg,
    const QuantInterface& /*expert_quant*/,
    int tp_degree) {

    const auto& m = model_cfg.raw();
    const int tp = std::max(1, tp_degree);

    // Pass 1: TD-53y — MTP count mismatch
    const int config_mtp = m.num_nextn_predict_layers;
    const int loaded_mtp = loaded.mtp
        ? static_cast<int>(loaded.mtp->block_layers.size()) : 0;
    if (config_mtp != loaded_mtp) {
        // V4: the GGUF carries no nextn tensors and the plan emits no MTP
        // slots (see build_upload_plan §5) — a nextn-configured V4 model with
        // zero loaded MTP layers is the EXPECTED state (loader warns; ticket J
        // sources the embedded draft from the official safetensors).
        if (!(model_cfg.is_v4() && loaded_mtp == 0)) {
            throw std::runtime_error(
                "[TD-53y] MTP layer count mismatch: config expects " +
                std::to_string(config_mtp) + " layers, loaded model has " +
                std::to_string(loaded_mtp));
        }
    }

    // Pass 2: TD-55e — MTP block layers include DSA indexer in the plan
    // when has_dsa() is true (matching the plan builder's mtp_has_indexer flag).
    // The indexer contribution is included in attention_layer_bytes via
    // include_indexer, so validation below uses has_dsa to stay consistent.

    // Pass 3: TD-55b — DSA layers must include k_norm_bias
    if (model_cfg.has_dsa()) {
        for (const auto& layer : loaded.layers) {
            if (layer.indexer.empty()) continue;
            bool has_k_norm_bias = false;
            for (const auto& bundle : layer.indexer) {
                if (bundle.id.component == TensorComponent::indexer_k_norm_bias) {
                    has_k_norm_bias = true;
                    break;
                }
            }
            if (!has_k_norm_bias) {
                throw std::runtime_error(
                    "[TD-55b] Layer " + std::to_string(layer.layer_idx) +
                    " indexer missing k_norm_bias: plan reserves space for it "
                    "but loaded checkpoint does not include it");
            }
        }
    }

    // Pass 4: TD-53q — Embedding/output_head size vs plan (dtype check).
    // Round 2b: a k-quant embed/lm_head is now UPLOADED STREAMED (dequanted
    // to BF16 chunk-by-chunk at upload — host bytes stay quant-sized), so
    // compare in ELEMENTS for k-quant bundles and in bytes for float ones.
    auto check_streamable = [&](const WeightBundle& b, const PinnedSlot* slot,
                                const char* what) {
        if (b.weight.gguf_type.has_value()) {
            const int64_t elems_per_rank = b.weight.numel() / tp;
            if (elems_per_rank * 2 != slot->size_bytes) {
                throw std::runtime_error(
                    std::string("[TD-53q] ") + what +
                    " element-count mismatch: plan expects " +
                    std::to_string(slot->size_bytes / 2) + " elems/rank (BF16), "
                    "loaded k-quant tensor has " +
                    std::to_string(elems_per_rank) + " elems/rank");
            }
            return;
        }
        const int64_t actual_per_rank =
            static_cast<int64_t>(b.weight.data.size()) / tp;
        if (actual_per_rank != slot->size_bytes) {
            throw std::runtime_error(
                std::string("[TD-53q] ") + what + " size mismatch: plan expects " +
                std::to_string(slot->size_bytes) + " bytes/rank, loaded tensor has " +
                std::to_string(actual_per_rank) +
                " bytes/rank (possible dtype mismatch)");
        }
    };
    if (auto* slot = plan.find(PinnedComponent::embedding, -1);
        slot && loaded.embedding)
        check_streamable(*loaded.embedding, slot, "Embedding");
    if (auto* slot = plan.find(PinnedComponent::output_head_weight, -1);
        slot && loaded.output_head)
        check_streamable(*loaded.output_head, slot, "Output head");

    // Pass 5: Per-layer attention slot size check
    TpWeightSharder sharder(model_cfg, tp);

    auto validate_attention_layer = [&](const LoadedModel::LayerWeights& lw,
                                       int idx, bool include_indexer) {
        auto* attn_slot = plan.find(PinnedComponent::attention, idx);
        if (!attn_slot) return;

        // Pass empty indexer when plan excludes it (layers without DSA).
        static const std::vector<WeightBundle> empty_indexer;
        auto sharded = sharder.shard_attention_layer(
            lw.attention, include_indexer ? lw.indexer : empty_indexer, 0);
        int64_t actual = 0;
        for (const auto& sb : sharded) {
            // TD-73c: Norm weight tensors are F32 in checkpoint but uploaded as BF16.
            // Subtract half the weight size for norm components (aux unchanged).
            const bool is_norm =
                sb.id.component == TensorComponent::q_a_norm ||
                sb.id.component == TensorComponent::kv_a_norm ||
                sb.id.component == TensorComponent::indexer_k_norm_weight ||
                sb.id.component == TensorComponent::indexer_k_norm_bias;
            actual += sb.total_bytes() - (is_norm ? sb.weight.size_bytes() / 2 : 0);
        }

        // Slot may include up to 15 bytes of alignment padding beyond actual data.
        // GG-9: for the generic `gguf` weight quant the attention slot is sized at
        // a BF16 UPPER BOUND (per-tensor k-quant types are unknown at plan time;
        // see attention_layer_bytes), so the actual packed upload is legitimately
        // smaller. The upload records real per-projection pointers and bumps the
        // cursor to the slot end, so an over-sized slot is structurally safe —
        // only an UNDER-sized slot (actual > slot, data overflows) is a bug.
        const bool gguf_upper_bound =
            (cfg.quantization.weights == config::WeightQuant::gguf);
        const bool too_big = actual > attn_slot->size_bytes;
        const bool too_small = !gguf_upper_bound
                             && actual < attn_slot->size_bytes - 15;
        if (too_big || too_small) {
            throw std::runtime_error(
                "Plan validation: attention slot mismatch layer " +
                std::to_string(idx) + ": loaded sharded size " +
                std::to_string(actual) + " != plan slot " +
                std::to_string(attn_slot->size_bytes));
        }

        auto* norm_slot = plan.find(PinnedComponent::layer_norm, idx);
        if (!norm_slot) return;

        // TD-73c: Layer norms are F32 in checkpoint, uploaded as BF16 (halved).
        int64_t norm_actual = 0;
        for (const auto& bundle : lw.norms) {
            norm_actual += static_cast<int64_t>(bundle.weight.data.size()) / 2;
        }
        if (norm_actual != norm_slot->size_bytes) {
            throw std::runtime_error(
                "Plan validation: layer_norm slot mismatch layer " +
                std::to_string(idx) + ": loaded size " +
                std::to_string(norm_actual) + " != plan slot " +
                std::to_string(norm_slot->size_bytes));
        }
    };

    bool has_dsa = model_cfg.has_dsa();
    // V4: Pass 5 shards through the MLA TpWeightSharder, which does not know
    // the V4 anatomy (grouped o_proj, compressor, mHC — sharder is V4-2b).
    // Skip the per-layer attention size cross-check until V4-2b lands; slot
    // sizing itself is exercised by layer_registry/pinned-layout unit tests.
    if (!model_cfg.is_v4()) {
        for (const auto& layer : loaded.layers) {
            validate_attention_layer(layer, layer.layer_idx, has_dsa);
        }
    }
    if (loaded.mtp) {
        for (size_t i = 0; i < loaded.mtp->block_layers.size(); ++i) {
            int mtp_idx = m.num_hidden_layers + static_cast<int>(i);
            const auto& blk = loaded.mtp->block_layers[i];

            // MTP attention + norms (include indexer when DSA enabled)
            validate_attention_layer(blk, mtp_idx, has_dsa);

            // MTP gating
            auto* gw_slot = plan.find(PinnedComponent::gating_weight, mtp_idx);
            auto* gb_slot = plan.find(PinnedComponent::gating_bias, mtp_idx);
            if (gw_slot || gb_slot) {
                int64_t gating_actual = 0;
                for (const auto& bundle : blk.gating) {
                    // GG-9 mirror of the upload (engine 5b, same as main
                    // layers): an F32 gate_weight is converted to BF16 at
                    // upload when the plan sized gating at 2 B/elem
                    // (gating_compute != fp32) — compare at the converted
                    // size then. Under fp32 gating (V3.2) the legacy raw-F32
                    // upload matches the F32-sized slot unchanged. The
                    // e_score_correction_bias stays F32 in both.
                    int64_t w = static_cast<int64_t>(bundle.weight.data.size());
                    if (bundle.id.component == TensorComponent::gate_weight &&
                        bundle.weight.dtype == SafetensorsDtype::F32 &&
                        cfg.quantization.gating_compute !=
                            config::GatingQuant::fp32)
                        w /= 2;
                    gating_actual += w;
                    for (const auto& [_, aux] : bundle.aux)
                        gating_actual += static_cast<int64_t>(aux.data.size());
                }
                int64_t gating_expected = (gw_slot ? gw_slot->size_bytes : 0)
                                        + (gb_slot ? gb_slot->size_bytes : 0);
                if (gating_actual != gating_expected) {
                    throw std::runtime_error(
                        "Plan validation: MTP gating slot mismatch layer " +
                        std::to_string(mtp_idx) + ": loaded " +
                        std::to_string(gating_actual) + " != plan " +
                        std::to_string(gating_expected));
                }
            }

            // MTP shared expert (TP-sharded)
            auto* se_slot = plan.find(PinnedComponent::shared_expert_gate, mtp_idx);
            if (se_slot && !blk.shared_expert.empty()) {
                auto sharded_se = sharder.shard_ffn_layer(blk.shared_expert, 0);
                int64_t se_actual = 0;
                for (const auto& sb : sharded_se) se_actual += sb.total_bytes();
                // Sum all shared expert plan slots for this layer
                int64_t se_expected = 0;
                for (auto comp : {PinnedComponent::shared_expert_gate,
                                  PinnedComponent::shared_expert_up,
                                  PinnedComponent::shared_expert_gate_scales,
                                  PinnedComponent::shared_expert_up_scales,
                                  PinnedComponent::shared_expert_gate_scalar,
                                  PinnedComponent::shared_expert_up_scalar,
                                  PinnedComponent::shared_expert_down}) {
                    if (auto* s = plan.find(comp, mtp_idx)) se_expected += s->size_bytes;
                }
                if (se_actual != se_expected) {
                    throw std::runtime_error(
                        "Plan validation: MTP shared_expert slot mismatch layer " +
                        std::to_string(mtp_idx) + ": loaded sharded " +
                        std::to_string(se_actual) + " != plan " +
                        std::to_string(se_expected));
                }
            }
        }

        // MTP-specific tensors (embed_tokens, eh_proj, shared_head, norms)
        for (const auto& bundle : loaded.mtp->tensors) {
            PinnedComponent comp;
            bool tp_sharded = false;
            switch (bundle.id.component) {
                case TensorComponent::mtp_embed_tokens:
                    comp = PinnedComponent::mtp_embed_tokens; tp_sharded = true; break;
                case TensorComponent::mtp_shared_head_weight:
                    comp = PinnedComponent::mtp_shared_head_weight; tp_sharded = true; break;
                case TensorComponent::mtp_shared_head_norm:
                    comp = PinnedComponent::mtp_shared_head_norm; break;
                case TensorComponent::mtp_eh_proj:
                    comp = PinnedComponent::mtp_eh_proj; tp_sharded = true; break;
                case TensorComponent::mtp_enorm:
                    comp = PinnedComponent::mtp_enorm; break;
                case TensorComponent::mtp_hnorm:
                    comp = PinnedComponent::mtp_hnorm; break;
                default: continue;
            }
            // TD-74b: norm tensors are F32→BF16 during upload, halve loaded size
            const bool is_norm =
                bundle.id.component == TensorComponent::mtp_shared_head_norm ||
                bundle.id.component == TensorComponent::mtp_enorm ||
                bundle.id.component == TensorComponent::mtp_hnorm;

            int layer_idx = bundle.id.layer_idx;
            auto* slot = plan.find(comp, layer_idx);
            if (!slot) continue;
            int64_t actual = static_cast<int64_t>(bundle.weight.data.size());
            if (tp_sharded) actual /= tp;
            if (is_norm) actual /= 2;  // F32→BF16
            // GG-9 (mirrors the attention-slot rule in Pass 5): with the
            // generic `gguf` weight quant the quantizable MTP tensors
            // (embed_tokens, shared_head, eh_proj) are plan-sized at a BF16
            // UPPER BOUND — the packed k-quant upload is legitimately smaller
            // (cursor-based upload records real pointers, so an over-sized
            // slot is structurally safe). Norms convert exactly (F32→BF16).
            const bool gguf_upper_bound = !is_norm
                && cfg.quantization.weights == config::WeightQuant::gguf;
            const bool too_big = actual > slot->size_bytes;
            const bool mismatch = !gguf_upper_bound
                                && actual != slot->size_bytes;
            if (too_big || mismatch) {
                throw std::runtime_error(
                    "Plan validation: MTP tensor slot mismatch (" +
                    std::string(pinned_component_name(comp)) + ") layer " +
                    std::to_string(layer_idx) + ": loaded " +
                    std::to_string(actual) + " != plan " +
                    std::to_string(slot->size_bytes));
            }
        }
    }

    // Pass 6: Final norm size (TD-73c: F32→BF16 during upload, so halved)
    if (auto* slot = plan.find(PinnedComponent::final_norm, -1);
        slot && loaded.final_norm) {
        int64_t actual = static_cast<int64_t>(loaded.final_norm->weight.data.size()) / 2;
        if (actual != slot->size_bytes) {
            throw std::runtime_error(
                "Plan validation: final_norm size mismatch: loaded " +
                std::to_string(actual) + " != plan slot " +
                std::to_string(slot->size_bytes));
        }
    }
}

}  // namespace layerstorm::model
