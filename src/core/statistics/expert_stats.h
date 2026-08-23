#pragma once

#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "core/memory/eviction_policy.h"

namespace layerstorm::statistics {

// ── Input types ──────────────────────────────────────────────────────────────

/// A single expert activation from gating output.
struct ExpertActivation {
    memory::ExpertKey key;
    float routing_weight;  ///< Router's weight for this expert on this token.
};

/// Gating result for one (token, layer) pair.
/// Multiple layers for the same token share the same token_id.
struct GatingResult {
    uint64_t token_id;   ///< Monotonic per-token ID. Advances temporal state.
    uint32_t layer_idx;  ///< MoE layer index.
    std::vector<ExpertActivation> activations;  ///< Variable length (INV-0.2).
};

// ── ExpertStats ──────────────────────────────────────────────────────────────

/// Per-expert statistics tracker.
///
/// Tracks four EWMA-smoothed metrics per expert (identified by ExpertKey):
///   - activation frequency
///   - recency (time since last use)
///   - average routing weight
///   - temporal autocorrelation (used in last 2 tokens)
///
/// All query methods return values normalized to [0,1] per INV-4.6b.
/// Not thread-safe. Called exclusively by the single-threaded orchestrator
/// (INV-3.4.2) in Phase 3 (STATISTICS UPDATE).
class ExpertStats {
public:
    struct Options {
        double ewma_alpha = 0.01;          ///< EWMA decay rate.
        uint32_t num_moe_layers = 58;      ///< Number of MoE layers.
        uint32_t num_experts = 256;        ///< Experts per MoE layer.
        uint32_t first_moe_layer = 3;      ///< First MoE layer index.
        uint32_t max_recency_tokens = 1024;///< Recency normalization window.
    };

    explicit ExpertStats(Options opts);

    // ── Core update ─────────────────────────────────────────────────────

    /// Called once per orchestrator cycle with completed gating results.
    /// Updates frequency (EWMA), recency, routing weight (EWMA), and
    /// temporal autocorrelation state for all activated experts.
    void update(std::span<const GatingResult> completed);

    // ── Normalized [0,1] queries ────────────────────────────────────────

    /// Activation frequency. 1.0 = most frequent, 0.0 = never.
    double frequency(memory::ExpertKey key) const;

    /// Recency. 1.0 = oldest (most stale), 0.0 = just used.
    double recency(memory::ExpertKey key) const;

    /// Average routing weight. 1.0 = highest impact, 0.0 = lowest.
    double routing_weight(memory::ExpertKey key) const;

    /// Temporal autocorrelation. 1.0 = used in last 2 tokens, 0.0 = not.
    double temporal_autocorr(memory::ExpertKey key) const;

    // ── Batch operations ────────────────────────────────────────────────

    /// Fill recency/frequency/routing_weight/temporal_autocorr in eviction
    /// inputs. Leaves key, zone, is_duplicate, gpu_id, coactivation, and
    /// prefetch_score untouched.
    void fill_eviction_scores(
        std::span<memory::ExpertEvictionInput> inputs) const;

    // ── Raw data (for WorkloadDetector #24) ─────────────────────────────

    /// Raw EWMA frequency (not normalized) with lazy decay materialized.
    double raw_frequency(memory::ExpertKey key) const;

    /// Materialized frequency vector for one MoE layer.
    /// Returns span of num_experts doubles, indexed by expert_idx.
    std::span<const double> frequency_vector(uint32_t layer_idx) const;

    // ── Frequency percentile (for duplication logic) ────────────────────

    /// Percentile rank of expert's frequency among all experts [0, 100].
    double frequency_percentile(memory::ExpertKey key) const;

    // ── Accessors ───────────────────────────────────────────────────────

    uint64_t total_tokens_processed() const { return global_token_counter_; }
    const Options& options() const { return opts_; }

private:
    struct PerExpertState {
        double ewma_frequency = 0.0;
        double ewma_routing_weight = 0.0;
        uint64_t last_used_token = 0;
        uint64_t last_freq_update_token = 0;
    };

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Index into states_ from an ExpertKey.
    size_t state_index(const memory::ExpertKey& key) const {
        return static_cast<size_t>(key.layer_idx - opts_.first_moe_layer)
               * opts_.num_experts + key.expert_idx;
    }

    /// Is the key within bounds of the dense array?
    bool valid_key(const memory::ExpertKey& key) const {
        return key.layer_idx >= opts_.first_moe_layer
            && key.layer_idx < opts_.first_moe_layer + opts_.num_moe_layers
            && key.expert_idx < opts_.num_experts;
    }

    /// Compute (1-alpha)^gap using pre-computed log_decay_.
    double decay_factor(uint64_t gap) const {
        if (gap == 0) return 1.0;
        if (gap == 1) return one_minus_alpha_;
        return std::exp(static_cast<double>(gap) * log_decay_);
    }

    /// Materialize the decayed EWMA frequency for an expert.
    double materialized_frequency(const PerExpertState& s) const {
        uint64_t gap = global_token_counter_ - s.last_freq_update_token;
        return s.ewma_frequency * decay_factor(gap);
    }

    /// Ensure freq_snapshots_ are fresh for the current token counter.
    void ensure_freq_snapshots_fresh() const;

    /// Recompute normalization maxima from scratch.
    void recompute_maxima();

    // ── State ───────────────────────────────────────────────────────────

    Options opts_;
    double one_minus_alpha_;  ///< 1 - ewma_alpha, cached.
    double log_decay_;        ///< log(1 - ewma_alpha), for fast pow.

    uint64_t global_token_counter_ = 0;
    uint64_t last_seen_token_id_ = 0;  ///< For token boundary detection.
    bool has_seen_any_token_ = false;

    std::vector<PerExpertState> states_;  ///< Dense: [moe_layer_offset * num_experts + expert_idx]

    // Normalization maxima (updated during update() and periodic recompute).
    double max_frequency_ = 0.0;
    double max_routing_weight_ = 0.0;
    uint64_t last_full_recompute_token_ = 0;
    static constexpr uint64_t kRecomputeInterval = 1000;

    // Lazy frequency vector snapshots for frequency_vector().
    mutable std::vector<std::vector<double>> freq_snapshots_;
    mutable uint64_t freq_snapshot_token_ = UINT64_MAX;  // Force recompute on first call.
};

}  // namespace layerstorm::statistics
