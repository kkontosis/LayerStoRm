#include "core/statistics/coactivation_graph.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace layerstorm::statistics {

// ── Construction ─────────────────────────────────────────────────────────────

CoactivationGraph::CoactivationGraph(Options opts)
    : opts_(opts),
      log_decay_(std::log(opts.decay_factor)) {
    const size_t total_cells =
        static_cast<size_t>(opts_.num_moe_layers) * opts_.num_experts * opts_.num_experts;
    matrices_.resize(total_cells, 0.0f);

    last_decayed_token_.resize(opts_.num_moe_layers, 0);
    row_buf_.resize(opts_.num_experts, 0.0f);
}

// ── Core update ──────────────────────────────────────────────────────────────

void CoactivationGraph::update(std::span<const GatingResult> completed) {
    for (const auto& result : completed) {
        // Advance global token counter on new token_id.
        if (!has_seen_any_token_ || result.token_id != last_seen_token_id_) {
            has_seen_any_token_ = true;
            last_seen_token_id_ = result.token_id;
            ++global_token_counter_;
        }

        if (!valid_layer(result.layer_idx)) continue;

        const uint32_t moe_off = layer_offset(result.layer_idx);
        const uint32_t N = opts_.num_experts;

        // Apply pending decay for this layer if stale.
        const uint64_t gap = global_token_counter_ - last_decayed_token_[moe_off];
        if (gap > 0) {
            decay_layer(moe_off, gap);
            last_decayed_token_[moe_off] = global_token_counter_;
        }

        // Increment all pairwise entries for co-activated experts.
        const auto& acts = result.activations;
        float* mat = layer_matrix(moe_off);

        for (size_t i = 0; i < acts.size(); ++i) {
            const uint16_t ei = acts[i].key.expert_idx;
            if (ei >= N) continue;

            for (size_t j = i + 1; j < acts.size(); ++j) {
                const uint16_t ej = acts[j].key.expert_idx;
                if (ej >= N || ei == ej) continue;

                mat[static_cast<size_t>(ei) * N + ej] += 1.0f;
                mat[static_cast<size_t>(ej) * N + ei] += 1.0f;
            }
        }
    }
}

// ── Workload shift decay ─────────────────────────────────────────────────────

void CoactivationGraph::apply_shift_decay(double factor) {
    const auto f = static_cast<float>(factor);
    for (auto& v : matrices_) {
        v *= f;
    }
}

// ── Point query ──────────────────────────────────────────────────────────────

float CoactivationGraph::weight(uint32_t layer_idx, uint16_t expert_i, uint16_t expert_j) const {
    if (!valid_layer(layer_idx)) return 0.0f;
    if (expert_i >= opts_.num_experts || expert_j >= opts_.num_experts) return 0.0f;
    if (expert_i == expert_j) return 0.0f;

    const uint32_t moe_off = layer_offset(layer_idx);
    const uint32_t N = opts_.num_experts;
    const float raw = matrices_[matrix_base(moe_off) + static_cast<size_t>(expert_i) * N + expert_j];

    // Materialize any pending decay without mutating state.
    const uint64_t gap = global_token_counter_ - last_decayed_token_[moe_off];
    if (gap > 0) {
        return raw * static_cast<float>(compute_decay(gap));
    }
    return raw;
}

// ── Row access ───────────────────────────────────────────────────────────────

std::span<const float> CoactivationGraph::row(uint32_t layer_idx, uint16_t expert_idx) const {
    if (!valid_layer(layer_idx) || expert_idx >= opts_.num_experts) return {};

    const uint32_t moe_off = layer_offset(layer_idx);
    const uint32_t N = opts_.num_experts;
    const float* mat = layer_matrix(moe_off);
    const float* row_start = mat + static_cast<size_t>(expert_idx) * N;

    // Check if decay is pending.
    const uint64_t gap = global_token_counter_ - last_decayed_token_[moe_off];
    if (gap > 0) {
        const auto decay = static_cast<float>(compute_decay(gap));
        for (uint32_t j = 0; j < N; ++j) {
            row_buf_[j] = row_start[j] * decay;
        }
        return {row_buf_.data(), N};
    }

    // No pending decay — copy directly into row_buf_ for stable lifetime.
    std::copy(row_start, row_start + N, row_buf_.begin());
    return {row_buf_.data(), N};
}

// ── Greedy graph partitioning ────────────────────────────────────────────────

std::vector<CoactivationGraph::AffinityHint> CoactivationGraph::compute_affinity_hints(
    uint32_t num_gpus,
    std::span<const int> gpu_capacity_slots) const {

    if (num_gpus == 0 || gpu_capacity_slots.size() < num_gpus) return {};

    const uint32_t N = opts_.num_experts;
    std::vector<AffinityHint> result;
    result.reserve(static_cast<size_t>(opts_.num_moe_layers) * N);

    // Per-GPU assignment tracking (reused across layers).
    // assigned_experts[gpu] = list of expert indices assigned to that GPU.
    std::vector<std::vector<uint16_t>> assigned_experts(num_gpus);
    std::vector<int> assigned_count(num_gpus);

    for (uint32_t moe_off = 0; moe_off < opts_.num_moe_layers; ++moe_off) {
        const uint32_t layer_idx = opts_.first_moe_layer + moe_off;

        // Clear per-GPU assignment.
        for (uint32_t g = 0; g < num_gpus; ++g) {
            assigned_experts[g].clear();
            assigned_count[g] = 0;
        }

        // Compute total connectivity (row sum) for each expert in this layer.
        // Materialize decay for consistent reads.
        const uint64_t gap = global_token_counter_ - last_decayed_token_[moe_off];
        const auto decay = (gap > 0) ? static_cast<float>(compute_decay(gap)) : 1.0f;
        const float* mat = layer_matrix(moe_off);

        std::vector<std::pair<float, uint16_t>> expert_connectivity(N);
        for (uint16_t e = 0; e < N; ++e) {
            float row_sum = 0.0f;
            const float* row_start = mat + static_cast<size_t>(e) * N;
            for (uint16_t j = 0; j < N; ++j) {
                row_sum += row_start[j];
            }
            expert_connectivity[e] = {row_sum * decay, e};
        }

        // Sort by total connectivity, descending (most connected first).
        std::sort(expert_connectivity.begin(), expert_connectivity.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Greedy assignment.
        for (const auto& [connectivity, expert] : expert_connectivity) {
            int best_gpu = -1;
            double best_score = -1.0;
            int best_count = INT32_MAX;  // For load-balanced tie-breaking.

            for (uint32_t g = 0; g < num_gpus; ++g) {
                if (assigned_count[g] >= gpu_capacity_slots[g]) continue;

                // Partition score: sum of co-activation weights with experts
                // already assigned to this GPU.
                double score = 0.0;
                const float* expert_row = mat + static_cast<size_t>(expert) * N;
                for (uint16_t assigned : assigned_experts[g]) {
                    score += static_cast<double>(expert_row[assigned]) * decay;
                }

                // Prefer higher score; break ties by fewer assigned (load balance).
                if (best_gpu < 0 || score > best_score
                    || (score == best_score && assigned_count[g] < best_count)) {
                    best_score = score;
                    best_gpu = static_cast<int>(g);
                    best_count = assigned_count[g];
                }
            }

            if (best_gpu >= 0) {
                assigned_experts[static_cast<uint32_t>(best_gpu)].push_back(expert);
                assigned_count[best_gpu]++;
                result.push_back({
                    .key = {layer_idx, expert},
                    .gpu_idx = best_gpu,
                    .score = best_score,
                });
            }
        }
    }

    return result;
}

// ── Internal helpers ─────────────────────────────────────────────────────────

void CoactivationGraph::decay_layer(uint32_t moe_offset, uint64_t gap) {
    const auto factor = static_cast<float>(compute_decay(gap));
    const size_t base = matrix_base(moe_offset);
    const size_t count = static_cast<size_t>(opts_.num_experts) * opts_.num_experts;

    float* data = matrices_.data() + base;
    for (size_t i = 0; i < count; ++i) {
        data[i] *= factor;
    }
}

}  // namespace layerstorm::statistics
