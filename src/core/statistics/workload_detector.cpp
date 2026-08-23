#include "core/statistics/workload_detector.h"

#include <cmath>

namespace layerstorm::statistics {

WorkloadDetector::WorkloadDetector(Options opts) : opts_(opts) {
    snapshot_.resize(opts_.num_moe_layers,
                     std::vector<double>(opts_.num_experts, 0.0));
}

void WorkloadDetector::update(const ExpertStats& stats) {
    uint64_t current_tokens = stats.total_tokens_processed();
    if (current_tokens == 0) return;

    // How many complete windows have elapsed?
    auto ws = static_cast<uint64_t>(opts_.token_window_size);
    uint64_t completed_windows = current_tokens / ws;
    if (completed_windows <= windows_processed_) return;

    // One or more new windows completed since last update.
    uint64_t new_windows = completed_windows - windows_processed_;
    windows_processed_ = completed_windows;

    // Advance cooldown counter.
    if (windows_since_last_shift_ < UINT64_MAX) {
        if (windows_since_last_shift_ + new_windows < windows_since_last_shift_)
            windows_since_last_shift_ = UINT64_MAX;  // Overflow guard.
        else
            windows_since_last_shift_ += new_windows;
    }

    if (!has_snapshot_) {
        // First window: just store snapshot, no distance to compute.
        for (uint32_t l = 0; l < opts_.num_moe_layers; ++l) {
            auto fv = stats.frequency_vector(opts_.first_moe_layer + l);
            for (uint32_t e = 0; e < opts_.num_experts; ++e) {
                snapshot_[l][e] = fv[e];
            }
        }
        has_snapshot_ = true;
        return;
    }

    // Compute per-layer L2 distance, averaged across layers.
    double total_distance = 0.0;
    for (uint32_t l = 0; l < opts_.num_moe_layers; ++l) {
        auto fv = stats.frequency_vector(opts_.first_moe_layer + l);
        double layer_sq_sum = 0.0;
        for (uint32_t e = 0; e < opts_.num_experts; ++e) {
            double diff = fv[e] - snapshot_[l][e];
            layer_sq_sum += diff * diff;
            snapshot_[l][e] = fv[e];  // Update snapshot in place.
        }
        total_distance += std::sqrt(layer_sq_sum);
    }
    double avg_distance = total_distance / opts_.num_moe_layers;
    last_distance_ = avg_distance;

    // Welford's online update.
    ++welford_count_;
    double delta_w = avg_distance - welford_mean_;
    welford_mean_ += delta_w / static_cast<double>(welford_count_);
    double delta_w2 = avg_distance - welford_mean_;
    welford_m2_ += delta_w * delta_w2;

    // Shift detection with min-history and cooldown guards.
    if (welford_count_ >= kMinHistorySamples &&
        windows_since_last_shift_ >= kCooldownWindows) {
        double std_dev = distance_std_dev();
        double threshold = welford_mean_ + opts_.shift_threshold_std_devs * std_dev;
        if (avg_distance > threshold && std_dev > 0.0) {
            shift_pending_ = true;
            windows_since_last_shift_ = 0;
        }
    }
}

bool WorkloadDetector::shift_detected() {
    bool result = shift_pending_;
    shift_pending_ = false;
    return result;
}

double WorkloadDetector::distance_std_dev() const {
    if (welford_count_ < 2) return 0.0;
    return std::sqrt(welford_m2_ / static_cast<double>(welford_count_ - 1));
}

}  // namespace layerstorm::statistics
