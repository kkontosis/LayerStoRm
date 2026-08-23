#pragma once

// Shared types and resolution logic for host-side expert weight sources.
// Used by ExpertLifecycleManager and CommandDispatcher.

#include <cstddef>
#include <memory>
#include <vector>

#include "core/memory/eviction_policy.h"      // ExpertKey
#include "model/quantization/quant_interface.h" // ExpertShape

namespace layerstorm::model {
struct LoadedModel;
class PrepackedSource;
class PackedBufferCache;
}

namespace layerstorm::memory {
class NvmeTier;
class PinnedExpertArena;
class ArenaLoader;
}

namespace layerstorm::transfer {

/// J-1: outcome of host-source resolution.
///  - kReady:   `ptr` is a usable H2D source NOW (enqueue the H2D).
///  - kPending:  the arena slot is reserved and an async file→slot read is in
///               flight (J-1). NOT a source yet — the caller must defer the H2D
///               and retry once the load completes. `ptr` is null.
///  - kBackpressure: (direct/mmap-free mode) the arena is momentarily full —
///               every slot is pinned mid-H2D — and there is no mmap fallback.
///               No load was submitted; the caller must defer and retry when a
///               slot frees (an in-flight H2D completes), NOT when a load
///               completes. Distinct from kPending so it is re-driven on H2D
///               completion rather than arena-load completion. `ptr` is null.
///  - kAbsent:   no source found at any tier (caller fails the request).
enum class HostSourceStatus { kReady, kPending, kBackpressure, kAbsent };

/// Result of resolving the host-side source for an expert's weight data.
/// `ptr` points to the data (mmap, NvmeTier warm cache, or lazily-packed buffer).
/// `owned_buf` holds a reference to keep the lazily-packed buffer alive during DMA.
struct HostSourceResult {
    const void* ptr = nullptr;
    std::shared_ptr<std::vector<std::byte>> owned_buf;  // non-null when heap-allocated (null for zero-copy/NvmeTier)
    bool pinned = false;  ///< True if ptr is in pinned (page-locked) memory (WP-7b).
    HostSourceStatus status = HostSourceStatus::kAbsent;  ///< J-1; kReady iff ptr!=null below.
    /// Actual byte count of the source buffer when known (GG-9). A mixed "XL"
    /// GGUF packs each layer's experts at its OWN per-projection k-quant sizes,
    /// which can be SMALLER than the global-max cache slot (expert_bytes()). The
    /// H2D must copy exactly these bytes — copying the larger slot size overruns
    /// the (smaller) host packed buffer → segfault. 0 = unknown (sources that
    /// store full slot-sized data, e.g. prepacked/NvmeTier); caller then copies
    /// the full slot. See TD-GG9-EXPERT-PERLAYER-COPYSIZE.
    int64_t bytes = 0;
};

/// Dependencies for host-source resolution (extracted from ELM/CommandDispatcher).
/// All pointers are nullable — resolution skips absent sources.
struct HostSourceDeps {
    // P-24: when set (pin_host_expert_pool on), the per-NUMA-node pinned arena is
    // the priority-0 warm tier. resolve fills a slot on miss (reads the prepacked
    // file bytes in via prepacked_source) and returns the pinned slot pointer, so
    // H2D DMAs straight from page-locked NUMA-local memory (no staging copy).
    memory::PinnedExpertArena*  pinned_arena     = nullptr;
    // J-1: when set, arena misses submit the file→slot read here (async, off the
    // daemon thread) and resolve returns kPending instead of blocking on the
    // synchronous copy. When null, the synchronous load_into fallback is used.
    memory::ArenaLoader*        arena_loader     = nullptr;
    model::PrepackedSource*     prepacked_source = nullptr;
    memory::NvmeTier*           nvme_tier        = nullptr;
    model::PackedBufferCache*   packed_cache     = nullptr;
    model::LoadedModel*         loaded_model     = nullptr;
    model::ExpertShape          expert_shape{};
};

/// Resolve the host-side source for an expert's packed weight data.
/// Priority chain: PrepackedSource → NvmeTier → PackedBufferCache → LoadedModel.
/// On cache miss with LoadedModel fallback, lazy-packs and inserts into cache.
/// `target_gpu` (>=0) enables consumer-aware pinned-arena placement (INV-MoE-NUMA):
/// the arena slot is taken from the GPU's local NUMA node so the H2D is local.
/// -1 (default) falls back to expert_home_node placement (TD-NUMA-align).
HostSourceResult resolve_expert_host_source(
    memory::ExpertKey key, const HostSourceDeps& deps, int target_gpu = -1);

}  // namespace layerstorm::transfer
