#pragma once
// Reroute-eviction fallback — the SHARED, single-source eviction code path for
// the progressive FETCH_AND_RUN_MOE "make room" step (TD-FAR-EVICT /
// TD-FAR-EVICT-REROUTE). Lifted VERBATIM out of the daemon's
// CommandDispatcher::handle_fetch_and_run_moe so that BOTH the live daemon and
// the CPU offline-simulation integration test drive the identical victim
// selection over the identical engine classes (memory::ExpertCache truth +
// gpu_loader::EvictScoreBoard scores). The offline-sim reproducing the keeper
// hit-rates is the cross-validation that this move preserved daemon behaviour.
//
// CUDA-free (layerstorm_core / INV-GPU-1).
#include <cstdint>
#include <functional>
#include <span>

#include "core/memory/eviction_policy.h"  // ExpertKey

namespace layerstorm::memory { class ExpertCache; }

namespace layerstorm::gpu_loader {

class EvictScoreBoard;

// Per-fetch-entry view the eviction step consumes (one per routed expert this
// MoE layer-visit, index-aligned with the command's expert list). Built by the
// caller from its own state (daemon: ProgressiveMoeState + sideband evict map;
// offline-sim: the replayed layer instance + orchestrator LRU map).
struct FarEvictFetch {
  memory::ExpertKey key;        // this routed expert
  int target_gpu = 0;           // POST-reroute target (where it will be fetched)
  bool is_real_fetch = false;   // !is_arrived && fetch_requested (a true H2D)
  // Orchestrator-supplied victim (have_evict_map / 13c-2.0 Option A). When
  // has_victim is false the local fallback alone picks victims for this fetch.
  bool has_victim = false;
  memory::ExpertKey victim;     // resident to evict on target_gpu (if has_victim)
};

// Eviction counters (mirror the daemon's far_evict_* members), accumulated.
struct FarEvictStats {
  int honored = 0;     // orchestrator-supplied victim accepted
  int rejected = 0;    // orchestrator-supplied victim refused (guard / not resident)
  int fallback = 0;    // local-fallback eviction (board cheapest_keys / hash-order)
};

// Synchronous eviction action: returns true on success. In the daemon this is
// ExpertLifecycleManager::request_evict (which delegates to ExpertCache::evict);
// in the offline-sim it is a direct ExpertCache::evict. The cache's
// ResidencyListener (the board) is fired by evict() itself, keeping board
// residency in sync regardless of which caller drives this.
using EvictFn = std::function<bool(memory::ExpertKey, int)>;

// Free exactly enough unlocked, not-needed-this-layer STABLE residents per
// target GPU so the routed misses can be admitted. Mirrors the daemon block:
//   (1) honor the orchestrator-supplied victims first (have_evict_map), subject
//       to the needed-this-layer / locked / not-resident guards;
//   (2) local fallback for the residual shortfall — when board && lru_fallback,
//       the O(to_free·log N) cheapest_keys (LRU) walk over the board (which
//       authoritatively mirrors ExpertCache stable residency), widening to the
//       full ascending residency if guards skip some; else (board off / kill-
//       switch) the ExpertCache-truth hash-order scan (eviction_inputs) — the
//       SOLE, BYTE-IDENTICAL loader-off baseline path.
//
// have_evict_map gates step (1) (false → step (1) skipped entirely, byte-
// identical to the prior local-only path). lru_fallback is the A/B kill-switch
// (LS_EVICT_LRU_FALLBACK): true → board cheapest_keys path; false → legacy
// hash-order. cur_layer protects this-layer experts from eviction.
void apply_far_evictions(memory::ExpertCache& cache, EvictScoreBoard* board,
                         bool have_evict_map, bool lru_fallback,
                         std::span<const FarEvictFetch> fetches,
                         uint32_t cur_layer, const EvictFn& evict_fn,
                         FarEvictStats& stats);

}  // namespace layerstorm::gpu_loader
