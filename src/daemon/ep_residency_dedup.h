// EP-within-TP residency dedup (INV-MOE-EP-DISJOINT enforcement).
//
// TD-MOE-EP-COMBINE-RESIDUAL fix (Phase-1c). The EP-within-TP routed MoE combine
// — BOTH the legacy bf16 partial-sum AND the canonical per-slot SUM-gather
// (INV-DRIFT-EPCOMBINE) — is correct ONLY when each routed expert is
// resident-and-computed on EXACTLY ONE GPU (disjoint EP). If the SAME routed
// expert is resident on >1 GPU (a cross-GPU DUPLICATE, e.g. the experimental ACT
// reroute leaving a stale home copy), every holder computes it and the SUM
// combine returns (#holders)·c_k — a real DOUBLE-COUNT.
//
// This is a pure, dependency-free, collective-free decision: ownership is decided
// from the per-rank residency bitsets the caller ALREADY built locally (each
// rank's own expert-cache residency). It adds NO cross-GPU collective / sync to
// the MoE hot path. On the disjoint path (production RUN_MOE, baseline e%tp) there
// are zero duplicates so the dedup is a byte-identical no-op.
//
// Header-only so it can be unit-tested on synthetic bitsets without any GPU.

#pragma once

#include <cstdint>

namespace layerstorm::daemon {

// Enforce disjoint EP residency: when a routed expert is resident on >1 rank,
// keep it ONLY on the lowest-rank holder (the canonical owner) and clear it from
// every higher-rank holder, so each c_k is computed/contributed exactly once.
//
//   bitsets[r] : pointer to ceil(n_experts/8) bytes for rank r (rank order,
//                little-endian bit-per-expert: expert e ⇒ byte e/8, bit e%8).
//                A null entry is treated as "rank absent" (skipped).
//   num_ranks  : number of rank bitsets.
//   n_experts  : number of routed experts.
//
// Mutates bitsets in place. Returns the number of experts that were resident on
// >1 rank (cross-GPU duplicates) — for observe-only telemetry; the dedup itself
// is unconditional regardless of the count.
inline int dedup_ep_residency(uint8_t* const* bitsets, int num_ranks,
                              int n_experts) {
    int dup_count = 0;
    for (int e = 0; e < n_experts; ++e) {
        const int byte = e / 8;
        const uint8_t mask = static_cast<uint8_t>(1u << (e % 8));
        int first_r = -1;
        bool dup = false;
        for (int r = 0; r < num_ranks; ++r) {
            if (!bitsets[r]) continue;
            if (bitsets[r][byte] & mask) {
                if (first_r < 0) {
                    first_r = r;  // lowest-rank holder = canonical owner; keep.
                } else {
                    // Higher-rank duplicate holder: clear so it does not compute.
                    bitsets[r][byte] &= static_cast<uint8_t>(~mask);
                    dup = true;
                }
            }
        }
        if (dup) ++dup_count;
    }
    return dup_count;
}

}  // namespace layerstorm::daemon
