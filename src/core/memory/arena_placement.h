#pragma once

// Arena host-slot placement policy (Wave-2 M3, spec/reports/FETCH_XRAY.md):
// frequency-aware + HBM-first slot→node assignment for the pinned expert
// arena, raising the effective host source bandwidth for expert fetches.
//
// WHY (measured, FETCH_XRAY.md): expert fetches run at 1.09-1.20x of the
// per-command topology bound; the binding cap is the per-DDR-NUMA-node
// ~64 GiB/s read bandwidth with multiple GPU links ganging on one node
// (host slots are single-homed; the legacy fill order is frequency-blind
// expert-major, so a routed union's members collide on nodes at random).
// The Xeon-Max HBM nodes (much higher per-node bandwidth than DDR) hold
// only frequency-blind overflow spill. This policy:
//   (a) HBM-FIRST: the globally hottest (most-fetched) experts fill the HBM
//       node arenas to their (fraction_free-disciplined) planned capacity,
//       so the bytes most often on the wire come from nodes whose read
//       bandwidth the link gang cannot saturate;
//   (b) ANTI-GANG: within every MoE layer the remaining experts are spread
//       round-robin BY PER-LAYER FREQUENCY RANK across the DDR node arenas,
//       so a routed union's high-frequency members source from DISTINCT
//       nodes and per-node collisions are minimized.
//
// The frequency table is a static per-(layer,expert) fetch-count CSV derived
// from a representative traced run (tools/loader_xray/freq_table.py over an
// LS_PERF_TRACE dump). STALENESS: routing skew is only partially stable
// across prompts (hot-set Jaccard 0.43-0.79 adjacent, 0.28 long-range —
// prefill-routing-skew measurement); the table biases PLACEMENT only, never
// correctness — every expert remains resident, misplaced ones just source
// at legacy bandwidth.
//
// Configured via memory.arena_placement.freq_table (config JSON; "" = OFF =
// byte-for-byte the legacy behavior). Env LS_ARENA_PLACE_FREQ OVERRIDES the
// config: a path replaces the table; "off"/"none"/"0" force-disables (needed
// to switch a config-defaulted model back to the legacy or pure-online
// placement); unset/empty defers to the config. The policy identity (tag +
// file bytes — NOT the path or the source, so config vs env yields the same
// store) folds into the ArenaCache CONFIG-STABLE identity hash (P-24b) so a
// persisted store built under a different placement is wiped, never
// mis-adopted (see ArenaCache::hash_config).
//
// Placement is PERFORMANCE-ONLY: tokens must be byte-identical ON vs OFF
// (the same expert bytes DMA from a different host node).

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/memory/eviction_policy.h"  // ExpertKey

namespace layerstorm::memory {

using ArenaPlacementMap = std::unordered_map<ExpertKey, int>;

/// Placement policy resolved from config + env (see the header comment for
/// the precedence contract). When a table is selected the file MUST be
/// readable — a typo'd path silently reverting to the legacy frequency-blind
/// placement would invalidate measurements, so resolve() THROWS
/// std::runtime_error (fail-closed, surfaces at engine init).
struct ArenaPlacementPolicy {
    bool enabled = false;
    std::string freq_path;    ///< CSV: "layer,expert,count" (# comments ok)
    uint64_t identity = 0;    ///< FNV-1a-64 over policy tag + file bytes; 0=off

    /// The effective table path after env-override resolution ("" = off).
    /// Pure string logic — no file IO (usable for cheap "requested?" checks).
    static std::string resolved_path(const std::string& config_freq_table);

    /// Full policy: resolved_path + table load + identity hash. Throws on an
    /// unreadable/empty table (fail-closed).
    static ArenaPlacementPolicy resolve(const std::string& config_freq_table);

    /// Legacy env-only form (== resolve("")).
    static ArenaPlacementPolicy from_env();
};

/// Per-(layer,expert) fetch-frequency table. Missing keys read as 0.
struct ArenaFreqTable {
    std::unordered_map<ExpertKey, double> counts;

    /// Parse "layer,expert,count" CSV. Throws std::runtime_error on an
    /// unreadable file; skips blank/#-comment lines; malformed rows throw.
    static ArenaFreqTable load(const std::string& path);

    double at(ExpertKey k) const {
        auto it = counts.find(k);
        return it == counts.end() ? 0.0 : it->second;
    }
};

/// One placement-eligible node arena: id + free slot capacity + class.
struct ArenaPlacementNode {
    int    node = -1;
    size_t free_slots = 0;
    bool   is_hbm = false;
};

/// Compute the full key→node assignment (pure function — topology passed in,
/// deterministic for identical inputs):
///   1. HBM-FIRST: keys sorted by (freq desc, layer, expert); the top keys
///      fill the HBM nodes' free capacity, distributed per-layer round-robin
///      across HBM nodes (rank+layer offset) with capacity fallback.
///   2. ANTI-GANG DDR: per layer, remaining keys sorted by freq desc are
///      assigned round-robin by rank across DDR nodes ((rank+layer) % nDDR),
///      falling back to the DDR node with most free capacity when the target
///      is full.
/// Every key gets an assignment while ANY capacity remains (zero-freq tail
/// included — the arena stays full-fit); keys beyond total capacity are left
/// unassigned (caller falls back to the legacy tiered fill).
ArenaPlacementMap compute_arena_placement(
    const ArenaFreqTable& freq, const std::vector<ExpertKey>& keys,
    std::vector<ArenaPlacementNode> nodes);

}  // namespace layerstorm::memory
