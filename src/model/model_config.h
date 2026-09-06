#pragma once

#include <vector>

#include "config/config_parser.h"

namespace layerstorm::model {

/// Per-layer attention mechanism for DeepSeek-V4 hybrid attention
/// (derived from model.compress_ratios; non-V4 models never query it).
enum class V4AttentionType {
    kSwa,  // compress_ratios[l] == 0  — sliding-window-only raw attention
    kCsa,  // compress_ratios[l] == 4  — compressed sparse (Lightning Indexer)
    kHca,  // compress_ratios[l] == 128 — heavily compressed, dense
};

/// Higher-level model configuration with computed dispatch helpers
/// and derived architectural values.
///
/// Wraps config::ModelConfig, adding layer-type queries and
/// precomputed counts used throughout the inference engine.
class ModelConfig {
   public:
    /// Construct from a parsed Config (uses cfg.model).
    explicit ModelConfig(const config::Config& cfg);

    /// Construct directly from a ModelConfig section.
    explicit ModelConfig(const config::ModelConfig& model_cfg);

    // ── Dispatch helpers (computed from underlying fields) ──────────────

    /// Layer l is MoE if l >= first_k_dense_replace and matches moe_layer_freq.
    bool is_moe_layer(int layer_idx) const;

    /// DSA enabled when index_topk > 0.
    bool has_dsa() const;

    /// IndexShare: layer l recomputes the DSA indexer ("full") vs reuses the
    /// preceding full layer's top-k ("shared"). Full-layer set =
    /// { l : l < index_skip_topk_offset OR (l - offset + 1) % index_topk_freq == 0 }.
    /// index_topk_freq <= 0 ⇒ every layer is full (no sharing; GGUF default,
    /// matches the llama.cpp reference which drops indexer_types).
    bool is_full_index_layer(int layer_idx) const;
    /// The indexer COMPUTING-layer set: layers that own indexer-K storage
    /// and compute selection (GF3.5). Legacy DSA (GLM-5.2/V3.2):
    /// IndexShare full ∪ {layer 0}; MTP layers never compute (shared by
    /// construction, index_share_for_mtp_iteration). glm5_next: exactly
    /// the sparse-MLA layers (KDA linear layers carry no indexer) PLUS the
    /// MTP layer, which is sparse MLA with its OWN indexer tensors
    /// (iteration 0 computes; iterations 1+ reuse — GF3.11 wires the
    /// epoch-keyed reuse). Accepts layer_idx in [0, num_hidden_layers +
    /// num_nextn_predict_layers).
    bool computes_indexer(int layer_idx) const;

    /// Number of full (indexer-recomputing) layers.
    int num_full_index_layers() const { return num_full_index_layers_; }

    /// Per-layer full/shared mask (size num_hidden_layers; true = full).
    const std::vector<bool>& full_index_layer_mask() const {
        return full_index_layer_mask_;
    }

    /// MTP enabled when num_nextn_predict_layers > 0.
    bool has_mtp() const;

    /// P-29 step 11 / OQ-3 phase A: LS_MTP_PROBE=1 makes glm5_next MTP/NextN
    /// layer(s) (indices >= num_hidden_layers) count as MoE layers, so
    /// their routed experts become arena tenants served through the normal
    /// FETCH_AND_RUN seam. Env read once (static). Default OFF — layer
    /// counts and arena slot totals are byte-identical to the historical
    /// behavior, so the champion's 12,096-slot warm-store identity is
    /// untouched unless the probe is explicitly armed.
    /// P-29 step 13 phase B: also armable from config (speculation.method ==
    /// "mtp" + speculation.mtp.enabled) via arm_mtp_experts() — the
    /// Config-taking constructor arms it before the first census is
    /// computed, so every later ModelConfig in the process agrees.
    static bool mtp_probe_experts_enabled();

    /// P-29 step 13 phase B: process-wide arming of the MTP expert census from
    /// config (equivalent to LS_MTP_PROBE=1). Must be called before any
    /// census is consulted; arming that would CHANGE an already-consulted
    /// census throws (a split census would corrupt every layer table).
    static void arm_mtp_experts();

    // ── DeepSeek-V4 dispatch helpers (spec/DEEPSEEK4_PLAN.md V4-1b) ─────

    /// Architecture is deepseek_v4.
    bool is_v4() const;

    // ── glm5_next dispatch helpers (PLAN.md Phase GF3, GF3.2) ───────────

    /// Architecture is glm5_next (GLM-5.3-Flash family: hybrid KDA
    /// linear attention + NoPE sparse MLA per model.layer_types).
    bool is_glm5_next() const;

    /// Layer l is a KDA linear-attention layer (glm5_next only:
    /// layer_types[l] == linear_attention). Linear layers keep NO KV
    /// cache and NO indexer — per-request recurrent + conv state instead.
    /// False for all other architectures and out-of-range indices.
    bool is_linear_attention_layer(int layer_idx) const;

    /// Number of KDA linear-attention layers (GLM-5.3-Flash: 34).
    int num_linear_attention_layers() const {
        return num_linear_attention_layers_;
    }

    /// GF3.9: model-layer → DENSE linear-layer ordinal map — the index
    /// KdaStateLayout::recurrent_offset / ring_offset take
    /// (vram_allocator.h: "l is the DENSE linear-layer ordinal
    /// [0, num_layers), not the model layer index"). Returns -1 for a
    /// non-linear layer or out-of-range index. GLM-5.3-Flash:
    /// 0,1,2 → 0,1,2; 4,5,6 → 3,4,5; … ; 44 → 33.
    int linear_attention_layer_ordinal(int layer_idx) const {
        if (layer_idx < 0
            || layer_idx >= static_cast<int>(linear_layer_ordinal_.size()))
            return -1;
        return linear_layer_ordinal_[layer_idx];
    }

    /// Number of KV-bearing attention layers: layers that append per-token
    /// KV state (page provisioning, KV metadata, tiering, DCP cover ONLY
    /// these). glm5_next: the sparse-MLA layers (GLM-5.3-Flash: 11);
    /// every other architecture: all hidden layers.
    int num_kv_layers() const { return num_kv_layers_; }

    /// IndexPool active (glm5_next: index_kpool > 1) — indexer keys pooled
    /// index_kpool:1 before scoring; index_topk stays a TOKEN budget
    /// (spec/GLM-5.3-FLASH-MODELINFO.md §3d).
    bool has_index_pool() const;

    /// MLA attention path (latent decompression, kv_b et al.).
    /// True for all non-V4 architectures; false for V4 (native MQA-over-latent).
    bool uses_mla() const;

    /// V4 hybrid CSA/HCA attention pipeline active.
    bool has_csa_hca() const;

    /// Per-layer V4 attention mechanism from compress_ratios[l].
    /// Precondition: is_v4() and 0 <= layer_idx < num_hidden_layers
    /// (out-of-range or non-V4 queries return kSwa — the no-compression type).
    V4AttentionType attention_type_for_layer(int layer_idx) const;

    /// Layer l applies compress_rope_theta + yarn (compressed layers);
    /// uncompressed V4 layers use rope_theta with NO yarn (dossier §2.3 /
    /// ref/llama.cpp/src/models/deepseek4.cpp dual-RoPE rule).
    bool layer_uses_compress_rope(int layer_idx) const;

    /// mHC hyper-connection residual streams active (hc_mult > 1).
    bool has_mhc() const;

    /// Grouped low-rank output projection active (o_groups > 1).
    bool has_grouped_o_proj() const;

    /// Layer l routes experts via the token-id hash table (l < num_hash_layers).
    bool is_hash_layer(int layer_idx) const;

    /// Grouped routing when n_group > 1.
    bool has_grouped_routing() const;

    /// Vision encoder present and enabled.
    bool has_vision() const;

    // ── Derived dimensions ─────────────────────────────────────────────

    /// qk_nope_head_dim + qk_rope_head_dim
    int qk_head_dim() const;

    /// kv_lora_rank + qk_rope_head_dim (compressed KV cache dimension per token)
    int kv_cache_dim() const;

    // ── Derived layer counts ───────────────────────────────────────────

    /// Number of MoE layers (where is_moe_layer returns true).
    int num_moe_layers() const { return num_moe_layers_; }

    /// Number of dense FFN layers (where is_moe_layer returns false).
    int num_dense_layers() const { return num_dense_layers_; }

    /// Sorted list of MoE layer indices.
    const std::vector<int>& moe_layer_indices() const { return moe_layer_indices_; }

    /// Sorted list of dense FFN layer indices.
    const std::vector<int>& dense_layer_indices() const { return dense_layer_indices_; }

    // ── Access underlying config ───────────────────────────────────────

    const config::ModelConfig& raw() const { return cfg_; }

   private:
    config::ModelConfig cfg_;
    int num_moe_layers_{};
    int num_dense_layers_{};
    int num_full_index_layers_{};
    int num_linear_attention_layers_{};
    int num_kv_layers_{};
    std::vector<int> moe_layer_indices_;
    std::vector<int> dense_layer_indices_;
    std::vector<bool> full_index_layer_mask_;
    std::vector<int> linear_layer_ordinal_;  ///< GF3.9: -1 = not linear

    void compute_layer_counts();
};

}  // namespace layerstorm::model
