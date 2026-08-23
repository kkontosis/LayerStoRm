#pragma once

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace layerstorm::statistics {

// ── Input types ──────────────────────────────────────────────────────────────

/// Per-position calibration training data for Kangaroo (#64, spec §4.7.1).
struct ConfidenceSample {
    float raw_confidence;  ///< Raw logit/softmax confidence from draft model.
    bool was_accepted;     ///< Whether this position passed verification.
};

/// Produced by verifier step 7, consumed by AcceptanceTracker::update().
struct VerificationResult {
    uint64_t request_id;       ///< Monotonic request identifier.
    uint32_t accepted_tokens;  ///< Prefix length accepted.
    uint32_t attempted_tokens; ///< Draft depth attempted.
    bool used_layer_skip;      ///< Whether layer skipping was active (INV-4.15a monitoring).
    std::vector<ConfidenceSample> confidence_samples; ///< Per-position, length = attempted_tokens.
};

// ── AcceptanceTracker ────────────────────────────────────────────────────────

/// Monitors per-request and global speculation acceptance rates.
///
/// Feeds into the online calibrator (for quality adjustment per INV-0.3) and
/// the utility scorer (for per-request speculation depth). Also monitors
/// layer-skip acceptance separately. Collects (raw_confidence, was_accepted)
/// calibration pairs for future Kangaroo confidence calibration (#64).
///
/// Single-threaded, orchestrator Phase 3 only (INV-3.4.2). No CUDA.
class AcceptanceTracker {
public:
    struct Options {
        double ema_alpha = 0.5;              ///< Global EMA (maps to calibration.acceptance_ema_alpha).
        double per_request_ema_alpha = 0.5;  ///< Per-request EMA smoothing.
        uint32_t window_size = 16;           ///< Ring buffer capacity (Lookahead: 16 iterations).
        uint32_t calibration_buffer_size = 4096; ///< Max calibration samples before wrap.
    };

    explicit AcceptanceTracker(Options opts);

    // ── Core update ──────────────────────────────────────────────────────

    /// Called once per orchestrator cycle (Phase 3).
    void update(std::span<const VerificationResult> results);

    // ── Global queries [0,1] ─────────────────────────────────────────────

    /// Global EMA. Consumer: online_calibrator.
    double global_rate() const;

    /// Recent-window rate from ring buffer.
    double windowed_rate() const;

    /// Lifetime accepted/attempted.
    double cumulative_rate() const;

    /// EMA for layer-skip verifications only. -1.0 if none seen.
    double layer_skip_rate() const;

    // ── Per-request query [0,1] ──────────────────────────────────────────

    /// Entry for per-request snapshot export (IPC-7).
    struct PerRequestEntry {
        uint64_t request_id;
        double acceptance_rate;
    };

    /// Per-request EMA. 0.0 for unknown. Consumer: utility_scorer.
    double rate(uint64_t request_id) const;

    /// Copy up to max_count per-request entries into out[].
    /// Returns actual count written. Unordered (hash map iteration).
    uint32_t per_request_snapshot(PerRequestEntry* out, uint32_t max_count) const;

    /// Cleanup on request completion.
    void remove_request(uint64_t request_id);

    // ── Calibration data (for Kangaroo #64) ──────────────────────────────

    uint32_t calibration_samples_count() const;
    const ConfidenceSample& calibration_sample(uint32_t index) const; ///< 0=oldest, count-1=newest.
    void drain_calibration_data(); ///< Reset buffer (called by calibrator after consuming).

    // ── Cumulative counters ──────────────────────────────────────────────

    uint64_t total_verifications() const { return total_verifications_; }
    uint64_t total_accepted_tokens() const { return total_accepted_; }
    uint64_t total_attempted_tokens() const { return total_attempted_; }

    const Options& options() const { return opts_; }

private:
    struct WindowEntry {
        uint32_t accepted;
        uint32_t attempted;
    };

    struct PerRequestState {
        double ema;
        bool has_data;
    };

    Options opts_;

    // Global EMA.
    double global_ema_ = 0.0;
    bool has_global_ = false;

    // Cumulative counters.
    uint64_t total_accepted_ = 0;
    uint64_t total_attempted_ = 0;
    uint64_t total_verifications_ = 0;

    // Layer-skip EMA.
    double layer_skip_ema_ = 0.0;
    bool has_layer_skip_ = false;

    // Window ring buffer.
    std::vector<WindowEntry> window_buf_;
    uint32_t window_head_ = 0;
    uint32_t window_count_ = 0;
    uint64_t window_accepted_sum_ = 0;
    uint64_t window_attempted_sum_ = 0;

    // Per-request state.
    std::unordered_map<uint64_t, PerRequestState> per_request_;

    // Calibration ring buffer.
    std::vector<ConfidenceSample> calibration_buf_;
    uint32_t calibration_head_ = 0;
    uint32_t calibration_count_ = 0;
};

}  // namespace layerstorm::statistics
