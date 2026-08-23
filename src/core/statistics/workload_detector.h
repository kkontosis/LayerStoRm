#pragma once

#include <cstdint>
#include <vector>

#include "core/statistics/expert_stats.h"

namespace layerstorm::statistics {

/// Detects workload shifts by monitoring expert activation distribution changes.
///
/// Consumes ExpertStats frequency vectors (INV-4.14a), computing L2 distance
/// between snapshots at token window boundaries. Uses Welford's online algorithm
/// for running mean/variance of distances, triggering a shift when the distance
/// exceeds mean + threshold * std_dev.
///
/// Single-threaded, orchestrator Phase 3 only (INV-4.14b).
class WorkloadDetector {
public:
    struct Options {
        uint32_t num_moe_layers = 58;
        uint32_t num_experts = 256;
        uint32_t first_moe_layer = 3;
        int token_window_size = 32;
        double shift_threshold_std_devs = 3.0;
    };

    explicit WorkloadDetector(Options opts);

    /// Feed current ExpertStats after an orchestrator cycle.
    /// Checks for window boundaries and performs shift detection.
    void update(const ExpertStats& stats);

    /// Returns true if a shift was detected since last call (INV-4.14c).
    /// One-shot: resets the flag after returning true.
    bool shift_detected();

    // ── Accessors ────────────────────────────────────────────────────────

    const Options& options() const { return opts_; }
    uint64_t windows_processed() const { return windows_processed_; }
    double last_distance() const { return last_distance_; }
    double distance_mean() const { return welford_mean_; }
    double distance_std_dev() const;

private:
    static constexpr uint64_t kMinHistorySamples = 4;
    static constexpr uint64_t kCooldownWindows = 4;

    Options opts_;

    // Token tracking.
    uint64_t last_token_count_ = 0;

    // Window state.
    uint64_t windows_processed_ = 0;

    // Snapshot: [layer][expert] raw EWMA frequencies at last window boundary.
    std::vector<std::vector<double>> snapshot_;
    bool has_snapshot_ = false;

    // Shift detection.
    bool shift_pending_ = false;
    double last_distance_ = 0.0;

    // Welford's online mean/variance of L2 distances.
    uint64_t welford_count_ = 0;
    double welford_mean_ = 0.0;
    double welford_m2_ = 0.0;  // Sum of squared differences from mean.

    // Cooldown (INV-4.14d).
    uint64_t windows_since_last_shift_ = UINT64_MAX;  // Start as "cooldown expired".
};

}  // namespace layerstorm::statistics
