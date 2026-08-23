#include "core/statistics/acceptance_tracker.h"

namespace layerstorm::statistics {

// ── Construction ─────────────────────────────────────────────────────────────

AcceptanceTracker::AcceptanceTracker(Options opts) : opts_(opts) {
    window_buf_.resize(opts_.window_size, {0, 0});
    calibration_buf_.resize(opts_.calibration_buffer_size);
}

// ── Core update ──────────────────────────────────────────────────────────────

void AcceptanceTracker::update(std::span<const VerificationResult> results) {
    for (const auto& r : results) {
        const double ratio = r.attempted_tokens > 0
            ? static_cast<double>(r.accepted_tokens) / r.attempted_tokens
            : 0.0;

        // 1. Global EMA: seed on first, then blend.
        if (!has_global_) {
            global_ema_ = ratio;
            has_global_ = true;
        } else {
            global_ema_ = opts_.ema_alpha * ratio + (1.0 - opts_.ema_alpha) * global_ema_;
        }

        // 2. Cumulative counters.
        total_accepted_ += r.accepted_tokens;
        total_attempted_ += r.attempted_tokens;
        ++total_verifications_;

        // 3. Layer-skip EMA.
        if (r.used_layer_skip) {
            if (!has_layer_skip_) {
                layer_skip_ema_ = ratio;
                has_layer_skip_ = true;
            } else {
                layer_skip_ema_ = opts_.ema_alpha * ratio
                                + (1.0 - opts_.ema_alpha) * layer_skip_ema_;
            }
        }

        // 4. Window ring buffer: evict oldest if full, then insert.
        if (window_count_ == opts_.window_size) {
            // Evict oldest entry at head position.
            const auto& old = window_buf_[window_head_];
            window_accepted_sum_ -= old.accepted;
            window_attempted_sum_ -= old.attempted;
        } else {
            ++window_count_;
        }
        window_buf_[window_head_] = {r.accepted_tokens, r.attempted_tokens};
        window_accepted_sum_ += r.accepted_tokens;
        window_attempted_sum_ += r.attempted_tokens;
        window_head_ = (window_head_ + 1) % opts_.window_size;

        // 5. Per-request EMA.
        auto [it, inserted] = per_request_.try_emplace(
            r.request_id, PerRequestState{0.0, false});
        if (!it->second.has_data) {
            it->second.ema = ratio;
            it->second.has_data = true;
        } else {
            it->second.ema = opts_.per_request_ema_alpha * ratio
                           + (1.0 - opts_.per_request_ema_alpha) * it->second.ema;
        }

        // 6. Calibration ring buffer: append each sample.
        for (const auto& sample : r.confidence_samples) {
            calibration_buf_[calibration_head_] = sample;
            calibration_head_ = (calibration_head_ + 1) % opts_.calibration_buffer_size;
            if (calibration_count_ < opts_.calibration_buffer_size) {
                ++calibration_count_;
            }
        }
    }
}

// ── Global queries ───────────────────────────────────────────────────────────

double AcceptanceTracker::global_rate() const {
    return global_ema_;
}

double AcceptanceTracker::windowed_rate() const {
    if (window_attempted_sum_ == 0) return 0.0;
    return static_cast<double>(window_accepted_sum_)
         / static_cast<double>(window_attempted_sum_);
}

double AcceptanceTracker::cumulative_rate() const {
    if (total_attempted_ == 0) return 0.0;
    return static_cast<double>(total_accepted_)
         / static_cast<double>(total_attempted_);
}

double AcceptanceTracker::layer_skip_rate() const {
    if (!has_layer_skip_) return -1.0;
    return layer_skip_ema_;
}

// ── Per-request query ────────────────────────────────────────────────────────

double AcceptanceTracker::rate(uint64_t request_id) const {
    auto it = per_request_.find(request_id);
    if (it == per_request_.end()) return 0.0;
    return it->second.ema;
}

void AcceptanceTracker::remove_request(uint64_t request_id) {
    per_request_.erase(request_id);
}

uint32_t AcceptanceTracker::per_request_snapshot(
    PerRequestEntry* out, uint32_t max_count) const {
    uint32_t count = 0;
    for (const auto& [request_id, state] : per_request_) {
        if (count >= max_count) break;
        if (state.has_data) {
            out[count++] = {request_id, state.ema};
        }
    }
    return count;
}

// ── Calibration data ─────────────────────────────────────────────────────────

uint32_t AcceptanceTracker::calibration_samples_count() const {
    return calibration_count_;
}

const ConfidenceSample& AcceptanceTracker::calibration_sample(uint32_t index) const {
    // 0 = oldest, count-1 = newest.
    const uint32_t buf_size = opts_.calibration_buffer_size;
    const uint32_t pos = (calibration_head_ - calibration_count_ + index + buf_size) % buf_size;
    return calibration_buf_[pos];
}

void AcceptanceTracker::drain_calibration_data() {
    calibration_head_ = 0;
    calibration_count_ = 0;
}

}  // namespace layerstorm::statistics
