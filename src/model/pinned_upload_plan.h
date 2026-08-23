#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace layerstorm::config { struct Config; }

namespace layerstorm::model {

class ModelConfig;
class QuantInterface;

enum class PinnedComponent {
    embedding,
    output_head_weight,
    output_head_bias,
    attention,
    layer_norm,
    gating_weight,
    gating_bias,
    shared_expert_gate,
    shared_expert_up,
    shared_expert_gate_scales,
    shared_expert_up_scales,
    shared_expert_gate_scalar,
    shared_expert_up_scalar,
    shared_expert_down,
    dense_ffn_gate,
    dense_ffn_up,
    dense_ffn_gate_scales,
    dense_ffn_up_scales,
    dense_ffn_gate_scalar,
    dense_ffn_up_scalar,
    dense_ffn_down,
    final_norm,

    // DeepSeek-V4 (V4-3a)
    gating_hash_table,  // ffn_gate_tid2eid I32 (hash layers, replaces gating_bias)
    output_hc,          // output_hc_{fn,base,scale} F32 (model-level mHC head)

    // MTP-specific (per-MTP-block tensors)
    mtp_embed_tokens,
    mtp_eh_proj,
    mtp_enorm,
    mtp_hnorm,
    mtp_shared_head_weight,
    mtp_shared_head_norm,
};

struct PinnedSlot {
    PinnedComponent component;
    int layer_idx;          // -1 for model-level (embedding, output_head, final_norm)
    int64_t offset;
    int64_t size_bytes;
    int contiguity_group;   // 0 = no constraint; same non-zero value = must be adjacent
};

struct PinnedUploadPlan {
    std::vector<PinnedSlot> slots;
    int64_t total_bytes = 0;

    // All slots for a given layer (contiguous in the vector).
    // Returns empty span for layer_idx == -1 (model-level slots are non-contiguous;
    // use find() instead for embedding, output_head, final_norm).
    std::span<const PinnedSlot> slots_for_layer(int layer_idx) const;

    const PinnedSlot* find(PinnedComponent comp, int layer_idx) const;
};

struct GgufModelExpertTypes;  // weight_loader.h

PinnedUploadPlan build_upload_plan(
    const ModelConfig& model_cfg,
    const config::Config& cfg,
    const QuantInterface& expert_quant,
    int tp_degree,
    int rank,
    // GG-9: for the generic `gguf` weight quant, size the shared-expert / dense-FFN
    // slots from THEIR OWN per-projection k-quant types (a Q5_K_XL mix has shared
    // Q8_0, dense Q5_K/Q6_K, routed a per-layer mix), not the routed `expert_quant`.
    // nullptr → size them via expert_quant (uniform GGUF / non-GGUF, unchanged).
    const GgufModelExpertTypes* gguf_shared_types = nullptr,
    const GgufModelExpertTypes* gguf_dense_types = nullptr);

struct LoadedModel;

/// Post-load validation: compare plan slot sizes against loaded bundles.
/// Throws std::runtime_error on the first mismatch detected.
/// Catches: TD-53q (dtype/size), TD-53y (MTP count), TD-55b (k_norm_bias), TD-55e (MTP indexer).
void validate_plan(
    const PinnedUploadPlan& plan,
    const LoadedModel& loaded,
    const ModelConfig& model_cfg,
    const config::Config& cfg,
    const QuantInterface& expert_quant,
    int tp_degree);

std::string_view pinned_component_name(PinnedComponent c);

}  // namespace layerstorm::model
