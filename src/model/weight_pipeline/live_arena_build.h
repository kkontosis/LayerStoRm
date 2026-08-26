#pragma once

// live_prepack_build — the LIVE PREPACK bulk pipeline: fill the pinned host
// expert arena DIRECTLY from the source GGUF shards, in parallel.
//
//   plan (init thread)  : PinnedExpertArena::plan_preload — reserve every
//                         coverable slot with the SAME ordering/tiered rules
//                         as the prepacked preload (byte-behavior parity).
//   read + transform    : NUMA-aware worker pool — workers are PINNED to the
//     (worker threads)    destination slot's NUMA node (the standing rule:
//                         writes destined for a node run node-local), read
//                         the de-stacked expert tensor ranges with O_DIRECT
//                         (io_uring-batched per slot when built with
//                         liburing), run pack_gguf_expert, and commit the
//                         bytes + zero tail into the pinned slot
//                         (LiveGgufExpertSource::load_into).
//   commit (init thread): drain worker completions and mark_ready each key —
//                         arena/ArenaCache state stays single-writer
//                         (INV-ARENA-CACHE-ORDER: records were killed at
//                         reserve, stamped ACTIVE only after the full write).
//
// Failed slots are never marked ready (reserved-empty, reloadable) — the same
// semantics as a failed prepacked preload read.

#include <cstdint>
#include <unordered_map>

#include "core/memory/eviction_policy.h"  // ExpertKey

namespace layerstorm::memory {
class NumaManager;
class PinnedExpertArena;
}  // namespace layerstorm::memory

namespace layerstorm::model {

class LiveGgufExpertSource;

struct LiveBuildStats {
    size_t filled = 0;
    size_t failed = 0;
    size_t skipped_full = 0;    ///< every node arena full (extend-only)
    size_t adopted = 0;         ///< already-ready (warm attach) keys skipped
    size_t placed = 0;          ///< placement-map directed reservations
    size_t place_fallback = 0;  ///< planned node full → tiered fill
    double seconds = 0.0;       ///< wall time of the parallel fill
    double gigabytes = 0.0;     ///< filled slots × stride / 2^30
};

/// Build the arena from `src`. `threads` <= 0 picks an automatic worker count
/// (min(hardware_concurrency, 32), split across destination NUMA nodes
/// proportionally to their slot counts). Init-thread only (drives arena
/// state); worker threads touch only slot bytes + the source.
LiveBuildStats live_prepack_build(
    memory::PinnedExpertArena& arena, memory::NumaManager& numa,
    const LiveGgufExpertSource& src, uint32_t num_layers,
    uint32_t num_experts,
    const std::unordered_map<memory::ExpertKey, int>* placement = nullptr,
    int threads = 0);

}  // namespace layerstorm::model
