#pragma once

#include <compare>
#include <cstdint>

namespace layerstorm::memory {

// ── Expert identification ───────────────────────────────────────────────────

struct ExpertKey {
    uint32_t layer_idx{};
    uint16_t expert_idx{};

    bool operator==(const ExpertKey&) const = default;
    auto operator<=>(const ExpertKey&) const = default;
};

// ── Cache zone ──────────────────────────────────────────────────────────────

/// Two-zone cache partitioning (spec §4.6).
/// Stable: co-activation-graph placements, high-frequency experts, slow eviction.
/// Streaming: speculative prefetch, one-off loads, fast turnover.
enum class CacheZone { kStable, kStreaming };

// ── Eviction input ──────────────────────────────────────────────────────────

/// All scoring terms must be normalized to [0,1] by the caller (INV-4.6b).
struct ExpertEvictionInput {
    ExpertKey key;
    CacheZone zone = CacheZone::kStreaming;
    bool is_duplicate = false;   ///< Non-primary copy (INV-0.4, INV-4.6c).
    int gpu_idx = 0;  ///< GPU position index (INV-4.18).

    // ── Normalized scoring terms ──
    double recency = 0.0;          ///< time_since_last_use. 1.0 = oldest.
    double frequency = 0.0;        ///< access frequency. 1.0 = most frequent.
    double routing_weight = 0.0;   ///< avg routing weight. 1.0 = highest impact.
    double temporal_autocorr = 0.0;///< used in last 2 tokens. 1.0 = yes.
    double coactivation = 0.0;     ///< fraction of co-activated friends on GPU. 1.0 = all.
    double prefetch_score = 0.0;   ///< about to be needed. 1.0 = definitely.
};

// ── Eviction output ─────────────────────────────────────────────────────────

struct EvictionCandidate {
    ExpertKey key;
    CacheZone zone = CacheZone::kStreaming;
    bool is_duplicate = false;
    int gpu_idx = 0;  ///< GPU position index (INV-4.18).
    double score = 0.0;  ///< Higher = more evictable = evicted first.
};

// ── Constants ───────────────────────────────────────────────────────────────

/// Additive penalty ensuring duplicates always evict before primaries (INV-0.4).
/// Much larger than the scoring formula's output range (~2.0 max).
inline constexpr double kDuplicatePenalty = 1000.0;

/// Additive bonus for streaming-zone experts when both zones are eligible.
/// Ensures streaming evicts before stable. Smaller than kDuplicatePenalty.
inline constexpr double kStreamingZoneBonus = 500.0;

}  // namespace layerstorm::memory

// ── Hash specialization ─────────────────────────────────────────────────────

template <>
struct std::hash<layerstorm::memory::ExpertKey> {
    size_t operator()(const layerstorm::memory::ExpertKey& k) const noexcept {
        return std::hash<uint64_t>{}(
            (static_cast<uint64_t>(k.layer_idx) << 16) | k.expert_idx);
    }
};
