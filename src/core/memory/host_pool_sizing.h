#pragma once

// Host expert pool sizing resolver (P-24 / INV-4.12f). Turns the user-facing
// `memory.pin_host_expert_pool_sizing` config (a global mode/value + optional
// per-NUMA-node overrides) into a concrete per-node byte budget. Shared by the
// PinnedExpertArena ctor so the three sizing modes live in one place.
//
// Kept as a free function (not a NumaManager method) so the low-level NUMA layer
// stays free of any `config::` dependency; this sits above both.

#include <cstddef>

#include "config/config_parser.h"  // PinHostExpertPoolSizingConfig, HostPoolSizingMode

namespace layerstorm::memory {

class NumaManager;

/// Resolve the pinned host-pool byte budget for one NUMA node. Applies a
/// `per_node` override matching `node` if present, else the global mode/value:
///   - fraction_free  → node_available_bytes(node) * value  (cache-aware free)
///   - fraction_total → node_total_bytes(node)     * value
///   - absolute_gb    → value GiB (decimals honored)
/// Throws std::runtime_error on an invalid value (fraction_* not in (0,1], or
/// absolute_gb <= 0).
size_t resolve_node_budget_bytes(
    const config::PinHostExpertPoolSizingConfig& sizing,
    const NumaManager& numa, int node);

}  // namespace layerstorm::memory
