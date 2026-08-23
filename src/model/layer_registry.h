#pragma once

#include <cstdint>
#include <vector>

#include "config/config_parser.h"
#include "model/model_config.h"
#include "model/pinned_region_layout.h"
#include "model/quantization/quant_interface.h"

namespace layerstorm::model {

// ── Quant precision helpers ─────────────────────────────────────────────────

/// Bytes per element for attention weight quantization format.
double bytes_per_element(config::AttentionQuant q);

/// Bytes per element for gating weight quantization format.
double bytes_per_element(config::GatingQuant q);

/// Bytes per element for weight quantization format (no per-projection scalars).
double bytes_per_element(config::WeightQuant q);

/// Embedding and output head are always BF16.
inline constexpr double kEmbeddingBytesPerElement = 2.0;

// ── LayerInfo ───────────────────────────────────────────────────────────────

/// Per-transformer-layer metadata: sizes and pinning policy.
struct LayerInfo {
    int layer_idx;
    bool is_moe;

    // Component sizes in bytes. Zero if not applicable for this layer type.
    int64_t attention_bytes;
    int64_t ffn_bytes;                 // Dense FFN (0 for MoE layers)
    int64_t gating_bytes;              // Routing network (0 for dense layers)
    int64_t shared_expert_bytes;       // All shared experts (0 for dense layers)
    int64_t per_routed_expert_bytes;   // One routed expert (0 for dense layers)

    // Pinning
    bool attention_pinned;
    bool ffn_pinned;
    bool gating_pinned;
    bool shared_expert_pinned;         // Always true for MoE layers

    /// Sum of all pinned component sizes for this layer.
    int64_t pinned_bytes() const;
};

// ── LayerRegistry ───────────────────────────────────────────────────────────

class LayerRegistry {
public:
    LayerRegistry(const ModelConfig& model_cfg,
                  const config::Config& cfg,
                  const QuantInterface& expert_quant);

    // Layer access
    int num_layers() const { return static_cast<int>(layers_.size()); }
    const LayerInfo& layer(int idx) const { return layers_[idx]; }
    const std::vector<LayerInfo>& layers() const { return layers_; }

    // Special components
    int64_t embedding_bytes() const { return embedding_bytes_; }
    int64_t output_head_bytes() const { return output_head_bytes_; }
    bool embedding_pinned() const { return embedding_pinned_; }
    bool output_head_pinned() const { return output_head_pinned_; }

    // Aggregate queries
    int64_t total_pinned_bytes() const { return total_pinned_bytes_; }
    int64_t per_routed_expert_bytes() const { return per_routed_expert_bytes_; }
    int total_routed_experts() const { return total_routed_experts_; }
    int num_moe_layers() const { return num_moe_layers_; }
    int num_dense_layers() const { return num_dense_layers_; }

    // Per-GPU VRAM budget estimate.
    // All pinned layers live on TP GPUs (split by tp_degree).
    // Non-TP GPUs hold only routed experts.
    struct GpuVramBudget {
        int gpu_id;
        int64_t total_vram_bytes;
        int64_t pinned_bytes;
        int64_t available_for_cache_bytes;  // total - pinned (KV deducted by #14)
    };
    std::vector<GpuVramBudget> estimate_gpu_budgets() const;

    /// The authoritative pinned layout for rank 0.
    const PinnedRegionLayout& pinned_layout() const { return pinned_layout_; }

    // DSpark draft accounting (DSP-2). Non-zero only when
    // speculation.method=dspark: the whole BF16 draft checkpoint is pinned on
    // ONE draft GPU (exact bytes from the checkpoint's safetensors header;
    // GPU = resolve_dspark_draft_gpu). estimate_gpu_budgets() charges it to
    // that GPU's pinned_bytes.
    int64_t dspark_draft_bytes() const {
        return dspark_rank_charges_.empty() ? 0 : dspark_rank_charges_[0];
    }
    int dspark_draft_gpu() const {
        return dspark_rank_gpus_.empty() ? -1 : dspark_rank_gpus_[0];
    }
    /// TD-DSPARK-DRAFT-SHARD: per-rank draft device set + pinned charges
    /// (weights shard + per-rank runtime scratch + align slack). One entry
    /// at the legacy single-rank placement.
    const std::vector<int>& dspark_rank_gpus() const {
        return dspark_rank_gpus_;
    }
    const std::vector<int64_t>& dspark_rank_charges() const {
        return dspark_rank_charges_;
    }

private:
    std::vector<LayerInfo> layers_;
    int64_t embedding_bytes_{};
    int64_t output_head_bytes_{};
    bool embedding_pinned_{};
    bool output_head_pinned_{};
    int64_t total_pinned_bytes_{};
    int64_t per_routed_expert_bytes_{};
    int total_routed_experts_{};
    int num_moe_layers_{};
    int num_dense_layers_{};

    const config::Config* cfg_{};
    PinnedRegionLayout pinned_layout_;

    // DSpark draft (DSP-2): exact draft weight bytes + hosting GPU position.
    std::vector<int> dspark_rank_gpus_;
    std::vector<int64_t> dspark_rank_charges_;
};

}  // namespace layerstorm::model
