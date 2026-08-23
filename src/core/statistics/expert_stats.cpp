#include "core/statistics/expert_stats.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace layerstorm::statistics {

// ── Construction ─────────────────────────────────────────────────────────────

ExpertStats::ExpertStats(Options opts)
    : opts_(opts),
      one_minus_alpha_(1.0 - opts.ewma_alpha),
      log_decay_(std::log(1.0 - opts.ewma_alpha)) {
    const size_t total = static_cast<size_t>(opts_.num_moe_layers) * opts_.num_experts;
    states_.resize(total);

    // Pre-allocate frequency snapshot buffers.
    freq_snapshots_.resize(opts_.num_moe_layers);
    for (auto& v : freq_snapshots_) {
        v.resize(opts_.num_experts, 0.0);
    }
}

// ── Core update ──────────────────────────────────────────────────────────────

void ExpertStats::update(std::span<const GatingResult> completed) {
    for (const auto& result : completed) {
        // Advance global token counter on new token_id.
        if (!has_seen_any_token_ || result.token_id != last_seen_token_id_) {
            has_seen_any_token_ = true;
            last_seen_token_id_ = result.token_id;
            ++global_token_counter_;
        }

        for (const auto& act : result.activations) {
            if (!valid_key(act.key)) continue;

            const size_t idx = state_index(act.key);
            auto& s = states_[idx];

            // Lazy EWMA frequency: decay then add activation.
            const uint64_t gap = global_token_counter_ - s.last_freq_update_token;
            s.ewma_frequency = s.ewma_frequency * decay_factor(gap) + opts_.ewma_alpha;
            s.last_freq_update_token = global_token_counter_;

            // EWMA routing weight (updated only on activation).
            s.ewma_routing_weight = one_minus_alpha_ * s.ewma_routing_weight
                                  + opts_.ewma_alpha * static_cast<double>(act.routing_weight);

            // Recency: reset on activation.
            s.last_used_token = global_token_counter_;

            // Track normalization maxima.
            if (s.ewma_frequency > max_frequency_) {
                max_frequency_ = s.ewma_frequency;
            }
            if (s.ewma_routing_weight > max_routing_weight_) {
                max_routing_weight_ = s.ewma_routing_weight;
            }
        }
    }

    // Periodic full recompute of maxima to correct drift.
    if (global_token_counter_ - last_full_recompute_token_ >= kRecomputeInterval) {
        recompute_maxima();
    }
}

// ── Normalized queries ───────────────────────────────────────────────────────

double ExpertStats::frequency(memory::ExpertKey key) const {
    if (!valid_key(key) || max_frequency_ <= 0.0) return 0.0;
    const auto& s = states_[state_index(key)];
    return std::min(1.0, materialized_frequency(s) / max_frequency_);
}

double ExpertStats::recency(memory::ExpertKey key) const {
    if (!valid_key(key) || global_token_counter_ == 0) return 0.0;
    const auto& s = states_[state_index(key)];
    // Never-activated experts: last_used_token == 0, recency approaches 1.0.
    const double tokens_since = static_cast<double>(global_token_counter_ - s.last_used_token);
    return std::min(1.0, tokens_since / static_cast<double>(opts_.max_recency_tokens));
}

double ExpertStats::routing_weight(memory::ExpertKey key) const {
    if (!valid_key(key) || max_routing_weight_ <= 0.0) return 0.0;
    const auto& s = states_[state_index(key)];
    return std::min(1.0, s.ewma_routing_weight / max_routing_weight_);
}

double ExpertStats::temporal_autocorr(memory::ExpertKey key) const {
    if (!valid_key(key) || global_token_counter_ == 0) return 0.0;
    const auto& s = states_[state_index(key)];
    // Binary: 1.0 if used in last 2 tokens (distance 0 or 1).
    return (global_token_counter_ - s.last_used_token <= 1) ? 1.0 : 0.0;
}

// ── Batch fill ───────────────────────────────────────────────────────────────

void ExpertStats::fill_eviction_scores(
    std::span<memory::ExpertEvictionInput> inputs) const {
    for (auto& input : inputs) {
        input.frequency = frequency(input.key);
        input.recency = recency(input.key);
        input.routing_weight = routing_weight(input.key);
        input.temporal_autocorr = temporal_autocorr(input.key);
    }
}

// ── Raw data ─────────────────────────────────────────────────────────────────

double ExpertStats::raw_frequency(memory::ExpertKey key) const {
    if (!valid_key(key)) return 0.0;
    return materialized_frequency(states_[state_index(key)]);
}

std::span<const double> ExpertStats::frequency_vector(uint32_t layer_idx) const {
    const uint32_t moe_offset = layer_idx - opts_.first_moe_layer;
    if (moe_offset >= opts_.num_moe_layers) return {};

    ensure_freq_snapshots_fresh();
    return freq_snapshots_[moe_offset];
}

void ExpertStats::ensure_freq_snapshots_fresh() const {
    if (freq_snapshot_token_ == global_token_counter_) return;

    for (uint32_t layer = 0; layer < opts_.num_moe_layers; ++layer) {
        auto& snap = freq_snapshots_[layer];
        const size_t base = static_cast<size_t>(layer) * opts_.num_experts;
        for (uint32_t e = 0; e < opts_.num_experts; ++e) {
            snap[e] = materialized_frequency(states_[base + e]);
        }
    }
    freq_snapshot_token_ = global_token_counter_;
}

// ── Frequency percentile ─────────────────────────────────────────────────────

double ExpertStats::frequency_percentile(memory::ExpertKey key) const {
    if (!valid_key(key)) return 0.0;

    const double target = materialized_frequency(states_[state_index(key)]);
    size_t count_le = 0;
    const size_t total = states_.size();

    for (size_t i = 0; i < total; ++i) {
        if (materialized_frequency(states_[i]) <= target) {
            ++count_le;
        }
    }

    return (static_cast<double>(count_le) / static_cast<double>(total)) * 100.0;
}

// ── Internal helpers ─────────────────────────────────────────────────────────

void ExpertStats::recompute_maxima() {
    max_frequency_ = 0.0;
    max_routing_weight_ = 0.0;

    for (const auto& s : states_) {
        const double freq = materialized_frequency(s);
        if (freq > max_frequency_) max_frequency_ = freq;
        if (s.ewma_routing_weight > max_routing_weight_) {
            max_routing_weight_ = s.ewma_routing_weight;
        }
    }

    last_full_recompute_token_ = global_token_counter_;
}

}  // namespace layerstorm::statistics
