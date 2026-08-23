#pragma once

#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "core/statistics/expert_stats.h"

namespace layerstorm::statistics {

// ── CoactivationGraph ────────────────────────────────────────────────────────

/// Per-MoE-layer symmetric co-activation matrix.
///
/// Tracks pairwise expert co-activation counts with exponential decay.
/// Used by the orchestrator (Phase 3) for statistics update, by the eviction
/// policy for co-activation scoring, and by the placement optimizer for
/// greedy graph partitioning.
///
/// Not thread-safe. Called exclusively by the single-threaded orchestrator
/// (INV-3.4.2, INV-4.13a). Diagonal entries are always zero (INV-4.13b).
class CoactivationGraph {
public:
    struct Options {
        uint32_t num_moe_layers = 58;     ///< Number of MoE layers.
        uint32_t num_experts = 256;       ///< Experts per MoE layer.
        uint32_t first_moe_layer = 3;     ///< First MoE layer index.
        double decay_factor = 0.999;      ///< Per-token exponential decay.
        double workload_shift_decay = 0.1;///< Aggressive decay on shift.
    };

    /// Affinity hint produced by greedy graph partitioning.
    struct AffinityHint {
        memory::ExpertKey key;
        int gpu_idx;    ///< Recommended GPU for this expert.
        double score;   ///< Partition score on the assigned GPU.
    };

    explicit CoactivationGraph(Options opts);

    // ── Core update (Phase 3) ────────────────────────────────────────────

    /// Update co-activation counts from completed gating results.
    /// For each layer's activated expert set {e_1, ..., e_K}, increments
    /// all C(K,2) pairwise entries by 1.0. Applies per-token decay lazily
    /// at the layer granularity.
    void update(std::span<const GatingResult> completed);

    // ── Workload shift decay ─────────────────────────────────────────────

    /// Multiply all matrices by factor. Called on workload shift (INV-0.8c).
    /// Typical factor: 0.1 (aggressive decay to forget stale patterns).
    void apply_shift_decay(double factor);

    // ── Point query ──────────────────────────────────────────────────────

    /// Co-activation weight between two experts on a layer.
    /// Returns 0.0 for invalid keys, diagonal (i==j), or out-of-range.
    /// Materializes any pending decay on-the-fly (const, no state mutation).
    float weight(uint32_t layer_idx, uint16_t expert_i, uint16_t expert_j) const;

    // ── Row access (for eviction scoring normalization) ──────────────────

    /// Materialized co-activation row for one expert on a layer.
    /// Returns span of num_experts floats (indexed by expert_idx).
    /// NOT reentrant: uses internal mutable buffer (safe under INV-3.4.2).
    /// Returns empty span for invalid keys.
    std::span<const float> row(uint32_t layer_idx, uint16_t expert_idx) const;

    // ── Greedy graph partitioning ────────────────────────────────────────

    /// Compute affinity hints via greedy partitioning.
    /// For each MoE layer independently:
    ///   1. Sort experts by total connectivity (row sum, descending).
    ///   2. Greedily assign each expert to the GPU with highest partition
    ///      score (sum of co-activation weights with already-assigned experts).
    ///   3. Respect per-GPU capacity constraints.
    /// Returns one AffinityHint per expert across all layers.
    /// O(num_experts * num_gpus) per layer.
    std::vector<AffinityHint> compute_affinity_hints(
        uint32_t num_gpus,
        std::span<const int> gpu_capacity_slots) const;

    // ── Accessors ────────────────────────────────────────────────────────

    uint64_t total_tokens_processed() const { return global_token_counter_; }
    const Options& options() const { return opts_; }

private:
    // ── Helpers ──────────────────────────────────────────────────────────

    /// Is the layer within MoE range?
    bool valid_layer(uint32_t layer_idx) const {
        return layer_idx >= opts_.first_moe_layer
            && layer_idx < opts_.first_moe_layer + opts_.num_moe_layers;
    }

    /// MoE layer offset (0-based) from absolute layer index.
    uint32_t layer_offset(uint32_t layer_idx) const {
        return layer_idx - opts_.first_moe_layer;
    }

    /// Base index into matrices_ for a given MoE layer offset.
    size_t matrix_base(uint32_t moe_offset) const {
        return static_cast<size_t>(moe_offset) * opts_.num_experts * opts_.num_experts;
    }

    /// Pointer to the start of a layer's matrix.
    float* layer_matrix(uint32_t moe_offset) {
        return matrices_.data() + matrix_base(moe_offset);
    }
    const float* layer_matrix(uint32_t moe_offset) const {
        return matrices_.data() + matrix_base(moe_offset);
    }

    /// Apply decay to a layer's matrix for the given token gap.
    void decay_layer(uint32_t moe_offset, uint64_t gap);

    /// Compute decay_factor^gap using pre-computed log_decay_.
    double compute_decay(uint64_t gap) const {
        if (gap == 0) return 1.0;
        if (gap == 1) return opts_.decay_factor;
        return std::exp(static_cast<double>(gap) * log_decay_);
    }

    // ── State ────────────────────────────────────────────────────────────

    Options opts_;
    double log_decay_;  ///< log(decay_factor), for fast pow.

    uint64_t global_token_counter_ = 0;
    uint64_t last_seen_token_id_ = 0;
    bool has_seen_any_token_ = false;

    /// Per-layer: last token at which decay was applied.
    std::vector<uint64_t> last_decayed_token_;

    /// Flat matrix storage: [moe_offset * N*N + i*N + j].
    /// Full symmetric storage. Diagonal always zero.
    std::vector<float> matrices_;

    /// Mutable row buffer for materialized reads via row().
    mutable std::vector<float> row_buf_;
};

}  // namespace layerstorm::statistics
