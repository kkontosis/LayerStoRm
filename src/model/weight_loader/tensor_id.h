#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace layerstorm::model {

// ── TensorComponent ──────────────────────────────────────────────────────────
// Identifies what a tensor represents within the model architecture.

enum class TensorComponent {
    // Attention (MLA)
    q_a_proj,
    q_a_norm,
    q_b_proj,
    kv_a_proj_with_mqa,
    kv_a_norm,
    kv_b_proj,
    o_proj,

    // DSA indexer (V3.2/GLM) + V4 Lightning-Indexer compressor. The
    // [indexer_wq_b .. indexer_compressor_norm] range must stay CONTIGUOUS —
    // place_bundle routes it into LayerWeights::indexer by enum-range check.
    indexer_wq_b,          // V4 reuses: blk.N.indexer.attn_q_b
    indexer_wk,
    indexer_k_norm_weight,
    indexer_k_norm_bias,
    indexer_weights_proj,  // V4 reuses: blk.N.indexer.proj
    indexer_compressor_wkv,    // V4: blk.N.indexer_compressor_kv
    indexer_compressor_wgate,  // V4: blk.N.indexer_compressor_gate
    indexer_compressor_ape,    // V4: blk.N.indexer_compressor_ape
    indexer_compressor_norm,   // V4: blk.N.indexer_compressor_norm

    // Layer norms
    input_layernorm,
    post_attention_layernorm,

    // Gating (router)
    gate_weight,
    gate_e_score_correction_bias,

    // FFN projections (used by dense FFN, routed expert, and shared expert)
    gate_proj,
    up_proj,
    down_proj,

    // Model-level
    embedding,
    output_head,
    final_norm,

    // MTP (Multi-Token Prediction)
    mtp_enorm,
    mtp_hnorm,
    mtp_eh_proj,
    mtp_shared_head_weight,
    mtp_shared_head_norm,
    mtp_embed_tokens,

    // Split MLA up-projection (GLM-1) — llama.cpp's MLA-optimized layout ships
    // the KV up-projection PRE-SPLIT as two tensors instead of the combined
    // `kv_b_proj`/`attn_kv_b`: `attn_k_b` (W_UK, stored TRANSPOSED per head as
    // [kv_lora, qk_nope]) and `attn_v_b` (W_UV, [v_head_dim, kv_lora]). These map
    // to the two components below; they are NEVER placed as a WeightBundle
    // directly — the GGUF loader dequants both, transposes attn_k_b, and stacks
    // them into a combined BF16 `kv_b_proj` the absorbed-MLA path consumes.
    mla_k_b_split,
    mla_v_b_split,

    // ── DeepSeek-V4 (spec/DEEPSEEK4_PLAN.md V4-2a; scratchpad/DS4_DOSSIER.md
    // §0.3 for shapes). V4's `blk.N.attn_kv` (single 512-dim MQA-over-latent
    // KV projection) maps onto kv_a_proj_with_mqa; there is NO kv_b for V4.
    o_proj_a,           // blk.N.attn_output_a — grouped o_proj stage-1 (per-group down)
    o_proj_b,           // blk.N.attn_output_b — grouped o_proj stage-2 (shared up)
    attn_sinks,         // blk.N.attn_sinks — per-Q-head attention-sink logits
    compressor_wkv,     // blk.N.attn_compressor_kv
    compressor_wgate,   // blk.N.attn_compressor_gate
    compressor_ape,     // blk.N.attn_compressor_ape — additive positional bias
    compressor_norm,    // blk.N.attn_compressor_norm
    hc_attn_fn,         // blk.N.hc_attn_fn — mHC pre/post mixing weights (attn wrap)
    hc_attn_base,       // blk.N.hc_attn_base
    hc_attn_scale,      // blk.N.hc_attn_scale
    hc_ffn_fn,          // blk.N.hc_ffn_fn — mHC mixing weights (FFN wrap)
    hc_ffn_base,        // blk.N.hc_ffn_base
    hc_ffn_scale,       // blk.N.hc_ffn_scale
    gate_tid2eid,       // blk.N.ffn_gate_tid2eid — hash-layer token-id→expert table (I32)
    output_hc_fn,       // output_hc_fn — model-level mHC output collapse
    output_hc_base,     // output_hc_base
    output_hc_scale,    // output_hc_scale
};

// ── TensorRole ───────────────────────────────────────────────────────────────
// Distinguishes the main weight from auxiliary quantization tensors.

enum class TensorRole {
    weight,          // Main weight data
    weight_scale,    // Per-group scale factors (e.g. F8_E4M3 for NVFP4)
    weight_scale_2,  // Global scale factor (F32 scalar for NVFP4)
    input_scale,     // Activation scale (F32 scalar for NVFP4)
    bias,            // Bias vector (e.g. indexer k_norm, gate correction)
};

// ── TensorOwner ──────────────────────────────────────────────────────────────
// Which subsystem owns this tensor.

enum class TensorOwner {
    attention,
    dense_ffn,
    routed_expert,
    shared_expert,
    gating,
    model_level,
    mtp,
};

// ── TensorId ─────────────────────────────────────────────────────────────────
// Fully parsed identity of a tensor from its name.

struct TensorId {
    TensorComponent component;
    TensorRole role;
    TensorOwner owner;
    int layer_idx;    // -1 for model-level tensors
    int expert_idx;   // -1 if not expert-specific

    // Two TensorIds match for grouping if they share the same component,
    // owner, layer, and expert — differing only in role.
    bool same_logical_weight(const TensorId& other) const {
        return component == other.component &&
               owner == other.owner &&
               layer_idx == other.layer_idx &&
               expert_idx == other.expert_idx;
    }
};

// ── Name parsing ─────────────────────────────────────────────────────────────

/// Parse a HuggingFace-format tensor name into a TensorId.
/// Returns nullopt if the name is not recognized.
/// Examples:
///   "model.layers.3.self_attn.q_a_proj.weight" -> TensorId{q_a_proj, weight, attention, 3, -1}
///   "model.layers.5.mlp.experts.42.gate_proj.weight_scale" -> TensorId{gate_proj, weight_scale, routed_expert, 5, 42}
///   "model.embed_tokens.weight" -> TensorId{embedding, weight, model_level, -1, -1}
std::optional<TensorId> parse_hf_name(std::string_view name);

/// Parse a GGUF-format tensor name into a TensorId, mapping the llama.cpp /
/// gguf-py deepseek2/glm-dsa naming convention onto the SAME canonical enums as
/// parse_hf_name (so all downstream routing is reused). Returns nullopt if the
/// name is not recognized.
///
/// Stacked expert tensors (`blk.N.ffn_{gate,up,down}_exps.weight`) are 3D —
/// ALL experts in one tensor with ONE ggml_type per projection — so they map to
/// expert_idx = -1 (de-stacking into per-expert spans is the loader's job).
/// Examples:
///   "blk.3.attn_q_a.weight"        -> {q_a_proj, weight, attention, 3, -1}
///   "blk.5.ffn_gate_exps.weight"   -> {gate_proj, weight, routed_expert, 5, -1}
///   "blk.5.ffn_gate_shexp.weight"  -> {gate_proj, weight, shared_expert, 5, -1}
///   "token_embd.weight"            -> {embedding, weight, model_level, -1, -1}
///   "output.weight"                -> {output_head, weight, model_level, -1, -1}
///   "output_norm.weight"           -> {final_norm, weight, model_level, -1, -1}
std::optional<TensorId> parse_gguf_name(std::string_view name);

/// String representation of a TensorComponent for logging/debugging.
std::string_view tensor_component_name(TensorComponent c);

/// String representation of a TensorRole for logging/debugging.
std::string_view tensor_role_name(TensorRole r);

/// String representation of a TensorOwner for logging/debugging.
std::string_view tensor_owner_name(TensorOwner o);

}  // namespace layerstorm::model
