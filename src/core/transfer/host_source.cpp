// Host-source resolution — shared by ELM and CommandDispatcher.
// See host_source.h for the HostSourceDeps / HostSourceResult types.

#include "core/transfer/host_source.h"

#include <spdlog/spdlog.h>

#include "core/memory/arena_loader.h"
#include "core/memory/nvme_tier.h"
#include "core/memory/pinned_expert_arena.h"
#include "model/weight_loader/weight_loader.h"
#include "model/weight_pipeline/packed_buffer_cache.h"
#include "model/weight_pipeline/prepacked_source.h"

namespace layerstorm::transfer {

HostSourceResult resolve_expert_host_source(
        memory::ExpertKey key, const HostSourceDeps& deps, int target_gpu) {
    // Priority -1 (P-24): pinned per-NUMA-node arena. When pin_host_expert_pool
    // is on, this is the warm DMA tier: page-locked, NUMA-local, no staging copy.
    // The arena only homes experts whose home node has an arena AND that the
    // prepacked cold loader can fill — so on a miss/absence we fall through to
    // the (non-pinned) mmap path below.
    if (deps.pinned_arena && deps.prepacked_source) {
        memory::PinnedExpertArena* pa = deps.pinned_arena;
        memory::PinnedNodeArena* a = nullptr;
        void* slot = nullptr;

        // 1. GLOBAL resolve: find the expert in ANY node arena — local OR a
        //    cross-node spill arena (Stage 1). The location index makes residency
        //    GPU/node-independent; a spilled slot is still a valid pinned DMA
        //    source (cross-NUMA H2D, INV-4.12h — slower than local but resident).
        if ((a = pa->locate(key))) {
            slot = a->resolve(key);  // resident + ready?
            if (slot) {
                a->touch(key);
            } else if (a->is_loading(key)) {
                // J-1: an async file→slot read is already in flight; defer the H2D.
                return {nullptr, nullptr, false, HostSourceStatus::kPending};
            }
            // else: reserved-but-not-ready, non-loading (rare) → fall to miss path
            // (reserve_for_fill's located branch returns the same slot to re-load).
        }

        // 1.5 (P-24b): a FREE slot may still hold this key's exact bytes from a
        //     previous run (ArenaCache identity-verified record). Adopt it —
        //     resident+ready with zero NVMe I/O. No-op without a wired cache.
        if (!slot) {
            slot = pa->try_adopt(key);
            if (slot) {
                a = pa->locate(key);
                a->touch(key);
            }
        }

        // 2. MISS: tiered weighted placement — prefer the computing GPU's local
        //    node while it has a free slot (extend), else a weighted cross-node
        //    spill arena, else evict the weighted-best node's LRU victim.
        if (!slot) {
            int node = -1;
            slot = pa->reserve_for_fill(key, target_gpu, &node);
            a = (node >= 0) ? pa->node_arena(node) : nullptr;
            if (!slot && deps.prepacked_source->is_direct()) {
                // Arena momentarily full (every candidate slot pinned mid-H2D);
                // direct mode has no mmap fallback. Signal back-pressure so the
                // caller re-drives when a slot frees (H2D completion), NOT when a
                // load completes (none was submitted). Eviction can't help here:
                // in-flight slots are non-evictable (INV-4.12i).
                return {nullptr, nullptr, false, HostSourceStatus::kBackpressure};
            }
            if (slot && a && deps.arena_loader) {
                // J-1 async path: mark loading + hand the cold copy to the pool.
                a->mark_loading(key);
                if (deps.arena_loader->submit(key, target_gpu, slot,
                                              deps.prepacked_source)) {
                    return {nullptr, nullptr, false, HostSourceStatus::kPending};
                }
                // submit() declined (dedup) — single daemon thread makes this
                // unreachable; defensively clear loading + sync-load below.
                a->clear_loading(key);
            }
            if (slot && a) {
                if (deps.prepacked_source->load_into(key, slot)) {
                    a->mark_ready(key);   // synchronous fallback (compiled)
                } else {
                    slot = nullptr;       // arena full / load failed → fall through
                    a = nullptr;
                }
            }
        }

        if (slot && a) {
            // Pin the slot for the duration of this H2D (released via the owned_buf
            // keep-alive, TD-82a). The guard captures the CONCRETE arena `a` (the
            // node actually holding the slot — local or spill), so release hits the
            // right node regardless of the computing GPU. Multi-GPU safe.
            a->acquire_inflight(key);
            std::shared_ptr<void> guard(
                static_cast<void*>(nullptr),
                [arena = a, key](void*) { arena->release_inflight(key); });
            std::shared_ptr<std::vector<std::byte>> keepalive(
                guard, static_cast<std::vector<std::byte>*>(nullptr));
            return {slot, std::move(keepalive), /*pinned=*/true,
                    HostSourceStatus::kReady};
        }
    }

    // Priority 0: PrepackedSource (mmap'd prepacked files — always warm, WP-3).
    if (deps.prepacked_source) {
        const void* ptr = deps.prepacked_source->resolve(key);
        // mmap lifetime = process lifetime; pinned when the region was
        // NUMA-registered (481-1) → pinned source skips the staging copy.
        if (ptr) return {ptr, nullptr, deps.prepacked_source->is_pinned(key),
                         HostSourceStatus::kReady};
        // WP-6: PrepackedSource active but expert missing — prepacked files
        // incomplete. Log error; lower-priority sources may or may not help.
        spdlog::error("PrepackedSource active but expert ({},{}) not found — "
                      "falling through to lower-priority sources",
                      key.layer_idx, key.expert_idx);
    }
    // Priority 1: NvmeTier warm cache (pinned memory — fast DMA source).
    if (deps.nvme_tier) {
        const void* ptr = deps.nvme_tier->host_ptr(key);
        if (ptr) return {ptr, nullptr, /*pinned=*/true, HostSourceStatus::kReady};
    }
    // Priority 1.5: PackedBufferCache (WP-4) — retained packed buffers.
    if (deps.packed_cache && !deps.packed_cache->is_mmap()) {
        auto buf = deps.packed_cache->lookup(key);
        if (buf) return {buf->data(), buf, deps.packed_cache->pins(),
                         HostSourceStatus::kReady,
                         static_cast<int64_t>(buf->size())};
    }
    // Priority 2: LoadedModel mmap'd safetensors data (lazy-packed for NVFP4/FP8).
    if (deps.loaded_model) {
        auto li = static_cast<size_t>(key.layer_idx);
        auto ei = static_cast<size_t>(key.expert_idx);
        if (li < deps.loaded_model->layers.size()) {
            auto& layer = deps.loaded_model->layers[li];
            if (ei < layer.routed_experts.size() &&
                !layer.routed_experts[ei].empty()) {
                // NVFP4 packs need the per-layer input_scale normalization
                // (TRT-LLM semantics). Compute it only when this expert will
                // actually pack (scan over the layer's experts, once per pack).
                model::Nvfp4InputScaleNorm norm;
                const model::Nvfp4InputScaleNorm* norm_ptr = nullptr;
                if (layer.routed_experts[ei][0].packed_slot.empty()) {
                    norm = model::compute_nvfp4_input_scale_norm(
                        layer.routed_experts, key.layer_idx);
                    norm_ptr = &norm;
                }
                model::ensure_expert_packed(
                    layer.routed_experts[ei], deps.expert_shape, norm_ptr);
                // WP-4: insert into cache for future reuse. When the cache pins
                // (481-1), insert NUMA-binds + page-locks owned_buf, so the slot
                // (a span within owned_buf) becomes a pinned DMA source.
                bool pinned = false;
                if (deps.packed_cache && !deps.packed_cache->is_mmap() &&
                    layer.routed_experts[ei][0].owned_buf) {
                    deps.packed_cache->insert(
                        key, layer.routed_experts[ei][0].owned_buf);
                    pinned = deps.packed_cache->pins();
                }
                return {layer.routed_experts[ei][0].packed_slot.data(),
                        layer.routed_experts[ei][0].owned_buf, pinned,
                        HostSourceStatus::kReady,
                        static_cast<int64_t>(
                            layer.routed_experts[ei][0].packed_slot.size())};
            }
        }
    }
    return {};
}

}  // namespace layerstorm::transfer
