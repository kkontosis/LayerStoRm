#pragma once

// PackedBufferCache: centralized host RAM cache for packed expert weight
// buffers (WP-4).  Replaces the scattered owned_buf pattern with unified
// lifecycle management, LRU eviction, and optional startup preloading.
//
// Two modes:
//   kExplicit — malloc + LRU eviction bounded by budget_bytes.
//   kMmap    — OS page cache manages residency (no engine-side LRU).
//
// Thread safety: NOT thread-safe. Called exclusively from daemon thread
// (INV-ELM-1).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/memory/eviction_policy.h"  // ExpertKey
#include "model/quantization/quant_interface.h"  // ExpertShape

namespace layerstorm::memory { class NumaManager; }

namespace layerstorm::model {

struct LoadedModel;
class PrepackedSource;

class PackedBufferCache {
public:
    enum class Mode { kExplicit, kMmap };

    struct Options {
        Mode mode = Mode::kExplicit;
        int64_t budget_bytes = 0;      // 0=passthrough, -1=unlimited, >0=bounded LRU
        int64_t slot_size_bytes = 0;   // bytes per expert (from layer_registry)
        ExpertShape expert_shape{};    // for lazy packing
        // When set (explicit mode), retained buffers are NUMA-bound to their
        // expert's home node and page-locked (cudaHostRegister) so H2D needs no
        // staging copy (481-1 / P-22 / P-23). Null → plain unpinned malloc.
        memory::NumaManager* numa = nullptr;
    };

    explicit PackedBufferCache(Options opts);
    ~PackedBufferCache();

    PackedBufferCache(const PackedBufferCache&) = delete;
    PackedBufferCache& operator=(const PackedBufferCache&) = delete;

    // ── Core operations (explicit mode only; mmap mode returns nullptr) ──

    /// Lookup cached buffer.  Returns shared_ptr (null if miss).
    /// Caller holding the shared_ptr prevents LRU eviction of that entry
    /// (use_count() > 1 detected during eviction scan).
    std::shared_ptr<std::vector<std::byte>> lookup(memory::ExpertKey key);

    /// Insert a packed buffer into the cache.  Triggers LRU eviction if over
    /// budget.  Eviction skips entries with use_count() > 1 (in-flight DMA).
    void insert(memory::ExpertKey key, std::shared_ptr<std::vector<std::byte>> buf);

    /// Remove a specific entry (for passthrough/budget=0 after H2D release).
    void release(memory::ExpertKey key);

    /// True if budget=0 (passthrough — identical to legacy TD-82a behavior).
    bool is_passthrough() const { return opts_.mode == Mode::kExplicit && opts_.budget_bytes == 0; }

    /// True if mode is kMmap.
    bool is_mmap() const { return opts_.mode == Mode::kMmap; }

    /// True if retained buffers are NUMA-bound + page-locked (so a cache hit's
    /// data() is a pinned H2D source — no staging copy needed).
    bool pins() const { return opts_.numa != nullptr && opts_.mode == Mode::kExplicit; }

    // ── Preload ─────────────────────────────────────────────────────────

    /// Mmap mode: madvise(MADV_WILLNEED) on all PrepackedSource regions.
    /// Returns number of regions advised.
    int preload_mmap(const PrepackedSource& source);

    /// Explicit mode: pack all experts from LoadedModel and insert.
    /// Stops when budget is full (budget>0) or after all experts (budget=-1).
    /// Returns number of experts inserted.
    int preload_explicit(LoadedModel& model, uint32_t first_moe_layer,
                         uint32_t num_moe_layers, uint32_t num_experts);

    // ── Diagnostics ─────────────────────────────────────────────────────

    int64_t used_bytes() const { return used_bytes_; }
    size_t entry_count() const { return entries_.size(); }
    bool is_over_budget() const {
        return opts_.budget_bytes >= 0 && used_bytes_ > opts_.budget_bytes;
    }

private:
    struct CacheEntry {
        std::shared_ptr<std::vector<std::byte>> buf;
        uint64_t lru_tick = 0;
        bool pinned = false;   ///< cudaHostRegister'd (must unpin before free)
    };

    /// NUMA-bind buf to key's home node + cudaHostRegister. Sets entry.pinned.
    void pin_entry(memory::ExpertKey key, CacheEntry& entry);
    /// cudaHostUnregister entry's buffer if pinned. Idempotent.
    void unpin_entry(CacheEntry& entry);

    /// Evict LRU entries until used_bytes_ <= budget_bytes (or no evictable
    /// entries remain).  Skips entries with use_count() > 1.
    void evict_if_needed();

    Options opts_;
    std::unordered_map<memory::ExpertKey, CacheEntry> entries_;
    uint64_t lru_clock_ = 0;
    int64_t used_bytes_ = 0;
};

}  // namespace layerstorm::model
