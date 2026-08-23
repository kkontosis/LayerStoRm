#pragma once

#include <string>
#include <vector>

#include "config_parser.h"

namespace layerstorm::config {

/// A single validation issue (error or warning).
struct ValidationIssue {
    std::string field;    // JSON path, e.g. "hardware.tp_pair"
    std::string message;  // Human-readable description
};

/// Result of config validation.
struct ValidationResult {
    std::vector<ValidationIssue> errors;
    std::vector<ValidationIssue> warnings;

    /// Config is valid if there are no errors (warnings are advisory).
    bool valid() const { return errors.empty(); }
};

/// Resolve the effective memory.arena_attach.on_conflict mode (P-24b).
/// An explicit value wins; absent ("unset") derives from persist —
/// persist=false → kill (historical mismatch wipe + cold rebuild),
/// persist=true → fail (store-protecting fatal boot). This is the ONLY place
/// the derivation rule lives; the persist × on_conflict compatibility matrix
/// (persist=true permits only 'fail') is rejected at parse time by
/// finalize_config()/validate_config(), so 'unset' and 'fail' are the only
/// values this can return under persist=true.
inline ArenaOnConflict resolved_arena_on_conflict(const ArenaAttachConfig& attach) {
    if (attach.on_conflict != ArenaOnConflict::unset) return attach.on_conflict;
    return attach.persist ? ArenaOnConflict::fail : ArenaOnConflict::kill;
}

/// Validate a parsed Config for internal consistency and feasibility.
///
/// Checks include:
/// - TP array must be power-of-2 length, all same-type 5090 GPUs (INV-0.5)
/// - NVMe tier requires hardware.nvme_paths
/// - VRAM budget rough feasibility (pinned layers + minimum expert cache)
/// - Speculation pool sizing vs. max_concurrent_requests (INV-0.7)
/// - CUDA graph constraints (INV-0.6)
/// - Model dimension sanity
/// - Pinned layer index bounds
/// - Eviction weight and performance objective consistency
///
/// Returns all errors and warnings found (does not short-circuit).
ValidationResult validate_config(const Config& cfg);

}  // namespace layerstorm::config
