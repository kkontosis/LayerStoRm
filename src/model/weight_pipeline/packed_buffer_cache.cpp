// PackedBufferCache — see packed_buffer_cache.h.

#include "model/weight_pipeline/packed_buffer_cache.h"

#include <sys/mman.h>  // madvise

#include <spdlog/spdlog.h>

#include "core/cuda_hardware_query.h"      // host_register_pinned
#include "core/memory/numa_manager.h"
#include "model/weight_loader/weight_loader.h"
#include "model/weight_pipeline/prepacked_source.h"

namespace layerstorm::model {

PackedBufferCache::PackedBufferCache(Options opts)
    : opts_(std::move(opts)) {}

PackedBufferCache::~PackedBufferCache() {
    // Unregister all pinned buffers before they free (481-1 teardown safety).
    for (auto& [key, entry] : entries_) unpin_entry(entry);
}

// ── NUMA pin / unpin (explicit mode) ─────────────────────────────────────────

void PackedBufferCache::pin_entry(memory::ExpertKey key, CacheEntry& entry) {
    if (!opts_.numa || !entry.buf || entry.buf->empty() || entry.pinned) return;
    int node = opts_.numa->expert_home_node(key.expert_idx);
    opts_.numa->bind_range(entry.buf->data(), entry.buf->size(), node);
    if (core::host_register_pinned(entry.buf->data(), entry.buf->size()) == 0) {
        entry.pinned = true;
    } else {
        spdlog::warn("PackedBufferCache: cudaHostRegister failed for expert "
                     "({},{}) — staging fallback", key.layer_idx, key.expert_idx);
    }
}

void PackedBufferCache::unpin_entry(CacheEntry& entry) {
    if (!entry.pinned || !entry.buf) return;
    core::host_unregister_pinned(entry.buf->data());
    entry.pinned = false;
}

// ── Core operations ────────────────────────────────────────────────────────

std::shared_ptr<std::vector<std::byte>>
PackedBufferCache::lookup(memory::ExpertKey key) {
    if (opts_.mode == Mode::kMmap) return nullptr;
    auto it = entries_.find(key);
    if (it == entries_.end()) return nullptr;
    it->second.lru_tick = ++lru_clock_;
    return it->second.buf;
}

void PackedBufferCache::insert(memory::ExpertKey key,
                               std::shared_ptr<std::vector<std::byte>> buf) {
    if (opts_.mode == Mode::kMmap || !buf) return;

    auto it = entries_.find(key);
    if (it != entries_.end()) {
        // Replace existing entry (unpin the old buffer first).
        unpin_entry(it->second);
        used_bytes_ -= static_cast<int64_t>(it->second.buf->size());
        it->second.buf = std::move(buf);
        used_bytes_ += static_cast<int64_t>(it->second.buf->size());
        it->second.lru_tick = ++lru_clock_;
        pin_entry(key, it->second);
        evict_if_needed();
        return;
    }

    auto entry_size = static_cast<int64_t>(buf->size());
    auto [ins, _] = entries_.emplace(key, CacheEntry{std::move(buf), ++lru_clock_});
    used_bytes_ += entry_size;
    pin_entry(key, ins->second);
    evict_if_needed();
}

void PackedBufferCache::release(memory::ExpertKey key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) return;
    unpin_entry(it->second);
    used_bytes_ -= static_cast<int64_t>(it->second.buf->size());
    entries_.erase(it);
}

// ── Eviction ───────────────────────────────────────────────────────────────

void PackedBufferCache::evict_if_needed() {
    if (opts_.budget_bytes < 0) return;  // unlimited
    while (used_bytes_ > opts_.budget_bytes) {
        // Find LRU entry with use_count() == 1 (only the cache holds it).
        auto victim = entries_.end();
        uint64_t min_tick = UINT64_MAX;
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->second.buf.use_count() == 1 && it->second.lru_tick < min_tick) {
                min_tick = it->second.lru_tick;
                victim = it;
            }
        }
        if (victim == entries_.end()) break;  // all in-flight, can't evict
        unpin_entry(victim->second);
        used_bytes_ -= static_cast<int64_t>(victim->second.buf->size());
        entries_.erase(victim);
    }
}

// ── Preload ────────────────────────────────────────────────────────────────

int PackedBufferCache::preload_mmap(const PrepackedSource& source) {
    int count = 0;
    for (const auto& region : source.mmap_regions()) {
        if (!region.base || region.size == 0) continue;
        if (::madvise(region.base, region.size, MADV_WILLNEED) != 0) {
            spdlog::warn("PackedBufferCache: madvise failed on region {} ({} bytes)",
                         static_cast<void*>(region.base), region.size);
            continue;
        }
        ++count;
    }
    return count;
}

int PackedBufferCache::preload_explicit(LoadedModel& model,
                                        uint32_t first_moe_layer,
                                        uint32_t num_moe_layers,
                                        uint32_t num_experts) {
    if (opts_.mode == Mode::kMmap) return 0;
    if (opts_.budget_bytes == 0) return 0;  // passthrough: would pack then evict

    int inserted = 0;
    for (uint32_t li = 0; li < num_moe_layers; ++li) {
        uint32_t layer_idx = first_moe_layer + li;
        if (layer_idx >= model.layers.size()) break;
        auto& layer = model.layers[layer_idx];

        for (uint32_t ei = 0; ei < num_experts; ++ei) {
            if (ei >= layer.routed_experts.size() ||
                layer.routed_experts[ei].empty())
                continue;

            // Check budget before packing (avoid wasted work).
            if (opts_.budget_bytes > 0 &&
                used_bytes_ + opts_.slot_size_bytes > opts_.budget_bytes) {
                spdlog::warn("PackedBufferCache: preload stopped at {} experts "
                             "(budget {} MB exhausted)", inserted,
                             opts_.budget_bytes / (1024 * 1024));
                return inserted;
            }

            auto& bundles = layer.routed_experts[ei];
            ensure_expert_packed(bundles, opts_.expert_shape);

            if (bundles[0].owned_buf) {
                memory::ExpertKey key{layer_idx, static_cast<uint16_t>(ei)};
                insert(key, bundles[0].owned_buf);
                ++inserted;
            }
        }
    }
    return inserted;
}

}  // namespace layerstorm::model
