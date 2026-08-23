#pragma once
// Abstract residency-change listener (TD-EVICT-BOARD-DESYNC fix —
// dependency-inversion / listener interface).
//
// ExpertCache is the SOLE authoritative mutator of routed-expert VRAM residency
// (INV-3.4.4). To let an external policy (the loader's EvictScoreBoard) track
// that residency WITHOUT the cache depending on it — and without relying on every
// eviction call site to remember to notify — ExpertCache fires this listener
// inside its OWN add/evict choke-points. "Mutate residency without the listener
// hearing" thus becomes structurally unrepresentable (compiler-enforced, not
// convention).
//
// The cache depends ONLY on this abstraction (dependency stays loader→core; no
// core→loader inversion). The cache drives MEMBERSHIP only via these hooks — it
// never feeds scores (least privilege; score-feeding is a separate loader→board
// API). The hooks track STABLE-zone residency (the eviction-fallback / evict_cum
// consumer is stable-zone only): an expert becomes "added" when it enters the
// stable zone (stable reserve / promote) and "removed" when it leaves it (evict
// of a stable entry / demote).
//
// CUDA-free (layerstorm_core / INV-GPU-1).
#include "core/memory/eviction_policy.h"  // ExpertKey

namespace layerstorm::memory {

class ResidencyListener {
 public:
    virtual ~ResidencyListener() = default;

    /// An expert just became resident in the STABLE zone on gpu_idx.
    virtual void on_resident_added(ExpertKey key, int gpu_idx) = 0;

    /// An expert just left the STABLE zone on gpu_idx (evicted / demoted /
    /// cancelled-before-ready).
    virtual void on_resident_removed(ExpertKey key, int gpu_idx) = 0;
};

}  // namespace layerstorm::memory
