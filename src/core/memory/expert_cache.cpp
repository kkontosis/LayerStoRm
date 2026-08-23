#include "core/memory/expert_cache.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace layerstorm::memory {

// ── TD-91f: bounds-checked gpu_state accessor ──────────────────────────────

ExpertCache::GpuState& ExpertCache::gpu_state(int gpu_idx) {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size()))
        throw std::out_of_range(
            "ExpertCache: gpu_idx " + std::to_string(gpu_idx) +
            " out of range [0, " + std::to_string(gpus_.size()) + ")");
    return gpus_[gpu_idx];
}

const ExpertCache::GpuState& ExpertCache::gpu_state(int gpu_idx) const {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size()))
        throw std::out_of_range(
            "ExpertCache: gpu_idx " + std::to_string(gpu_idx) +
            " out of range [0, " + std::to_string(gpus_.size()) + ")");
    return gpus_[gpu_idx];
}

// ── SlotAllocator ───────────────────────────────────────────────────────────

void ExpertCache::SlotAllocator::init(void* base_ptr, int64_t bytes_per_slot,
                                       int64_t zone_bytes) {
    base = base_ptr;
    slot_bytes = bytes_per_slot;
    total_slots_ = (bytes_per_slot > 0) ? static_cast<int>(zone_bytes / bytes_per_slot) : 0;

    free_list.clear();
    free_list.reserve(total_slots_);
    for (int i = total_slots_ - 1; i >= 0; --i) {
        free_list.push_back(i);
    }
}

int ExpertCache::SlotAllocator::allocate() {
    if (free_list.empty()) return -1;
    int idx = free_list.back();
    free_list.pop_back();
    return idx;
}

void ExpertCache::SlotAllocator::free(int slot_idx) {
    free_list.push_back(slot_idx);
}

void* ExpertCache::SlotAllocator::address(int slot_idx) const {
    return static_cast<char*>(base) + static_cast<int64_t>(slot_idx) * slot_bytes;
}

int ExpertCache::SlotAllocator::free_count() const {
    return static_cast<int>(free_list.size());
}

// ── Constructor (proportional offsets) ──────────────────────────────────────

ExpertCache::ExpertCache(const VramAllocator& vram,
                         const config::Config& cfg,
                         int64_t expert_bytes)
    : expert_bytes_(expert_bytes),
      duplication_enabled_(cfg.parallelism.expert_duplication.enabled),
      max_duplicated_fraction_(cfg.parallelism.expert_duplication.max_duplicated_fraction),
      duplication_freq_threshold_percentile_(
          cfg.memory.expert_cache.duplication_frequency_threshold_percentile) {
    // Proportional per-projection offsets (equal thirds).
    gate_bytes_ = expert_bytes / 3;
    up_bytes_ = expert_bytes / 3;
    down_bytes_ = expert_bytes - gate_bytes_ - up_bytes_;

    const auto& layout = vram.layout();
    gpus_.resize(vram.gpu_count());

    for (int i = 0; i < vram.gpu_count(); ++i) {
        auto& gs = gpus_[i];
        const auto& region = vram.region(i);
        const auto& gpu_layout = layout.gpus[i];

        gs.stable_slots.init(region.expert_stable, expert_bytes,
                             gpu_layout.expert_stable_bytes);

        // Streaming zone split: spill (top, adjacent to KV main) + prefetch (bottom)
        // Region layout: expert_streaming points to top of streaming zone
        // Spill zone: [expert_streaming, expert_streaming + spill_bytes)
        // Prefetch zone: [expert_streaming + spill_bytes, expert_streaming + streaming_total)
        auto* streaming_base = static_cast<char*>(region.expert_streaming);
        gs.spill_slots.init(streaming_base, expert_bytes,
                            gpu_layout.streaming_spill_bytes);
        gs.prefetch_slots.init(streaming_base + gpu_layout.streaming_spill_bytes,
                               expert_bytes,
                               gpu_layout.streaming_prefetch_bytes);

        // Store scratch region info (available when entering spill mode)
        gs.scratch.ptr = streaming_base;
        gs.scratch.size_bytes = gpu_layout.streaming_spill_bytes;
    }
}

// ── Constructor (precise offsets from QuantInterface) ───────────────────────

ExpertCache::ExpertCache(const VramAllocator& vram,
                         const config::Config& cfg,
                         int64_t expert_bytes,
                         const model::QuantInterface& quant,
                         const model::ExpertShape& shape)
    : ExpertCache(vram, cfg, expert_bytes) {
    // Override proportional offsets with precise per-projection sizes.
    gate_bytes_ = quant.bytes_per_projection(shape, model::Projection::gate);
    up_bytes_ = quant.bytes_per_projection(shape, model::Projection::up);
    down_bytes_ = quant.bytes_per_projection(shape, model::Projection::down);

    // Weight-only byte counts (for B/scale_B split within each slot region).
    gate_weight_bytes_ = quant.weight_bytes_per_projection(shape, model::Projection::gate);
    up_weight_bytes_ = quant.weight_bytes_per_projection(shape, model::Projection::up);
    down_weight_bytes_ = quant.weight_bytes_per_projection(shape, model::Projection::down);
}

// ── Core operations ─────────────────────────────────────────────────────────

void* ExpertCache::reserve(ExpertKey key, int gpu_idx, CacheZone zone,
                            bool is_duplicate) {
    auto& gs = gpu_state(gpu_idx);

    // Already resident on this GPU — return nullptr.
    if (gs.residents.contains(key)) return nullptr;

    SlotAllocator* alloc = nullptr;
    bool in_spill = false;

    if (zone == CacheZone::kStable) {
        alloc = &gs.stable_slots;
    } else {
        // Streaming zone: route to spill or prefetch
        if (gs.mode == PrefillMode::kSpillActive) {
            // Spill mode: only prefetch slots available
            alloc = &gs.prefetch_slots;
        } else {
            // Normal mode: try spill first, then prefetch
            int slot = gs.spill_slots.allocate();
            if (slot >= 0) {
                alloc = &gs.spill_slots;
                in_spill = true;

                void* addr = alloc->address(slot);
                CacheEntry entry;
                entry.key = key;
                entry.gpu_idx = gpu_idx;
                entry.zone = zone;
                entry.is_duplicate = is_duplicate;
                entry.in_spill_zone = true;
                entry.sub_components_ready = 0;
                entry.vram_address = addr;
                entry.slot_idx = slot;
                entry.gate_offset = 0;
                entry.up_offset = gate_bytes_;
                entry.down_offset = gate_bytes_ + up_bytes_;

                gs.residents.emplace(key, entry);
                if (is_duplicate) ++gs.duplicate_count_;
                return addr;
            }
            // Spill full → fall through to prefetch
            alloc = &gs.prefetch_slots;
        }
    }

    int slot = alloc->allocate();
    if (slot < 0) return nullptr;

    void* addr = alloc->address(slot);

    CacheEntry entry;
    entry.key = key;
    entry.gpu_idx = gpu_idx;
    entry.zone = zone;
    entry.is_duplicate = is_duplicate;
    entry.in_spill_zone = in_spill;
    entry.sub_components_ready = 0;
    entry.vram_address = addr;
    entry.slot_idx = slot;
    entry.gate_offset = 0;
    entry.up_offset = gate_bytes_;
    entry.down_offset = gate_bytes_ + up_bytes_;

    gs.residents.emplace(key, entry);
    if (is_duplicate) ++gs.duplicate_count_;

    // TD-EVICT-BOARD-DESYNC: a stable-zone reserve makes the expert resident in
    // the stable zone — notify the residency listener (the streaming early-return
    // path above is intentionally NOT notified; the listener tracks stable only).
    if (listener_ && zone == CacheZone::kStable)
        listener_->on_resident_added(key, gpu_idx);

    return addr;
}

void ExpertCache::mark_ready(ExpertKey key, int gpu_idx,
                              SubComponent component) {
    auto it = gpu_state(gpu_idx).residents.find(key);
    assert(it != gpu_state(gpu_idx).residents.end());
    it->second.sub_components_ready |= component;
}

void ExpertCache::mark_all_ready(ExpertKey key, int gpu_idx) {
    auto it = gpu_state(gpu_idx).residents.find(key);
    assert(it != gpu_state(gpu_idx).residents.end());
    it->second.sub_components_ready = kAll;
}

// ── Eviction lock (#90) ─────────────────────────────────────────────────────

bool ExpertCache::lock(ExpertKey key, int gpu_idx) {
    auto& gs = gpu_state(gpu_idx);
    auto it = gs.residents.find(key);
    if (it == gs.residents.end()) return false;
    ++it->second.lock_count;
    return true;
}

bool ExpertCache::unlock(ExpertKey key, int gpu_idx) {
    auto& gs = gpu_state(gpu_idx);
    auto it = gs.residents.find(key);
    if (it == gs.residents.end()) return false;
    if (it->second.lock_count == 0) return false;
    --it->second.lock_count;
    return true;
}

bool ExpertCache::is_locked(ExpertKey key, int gpu_idx) const {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size())) return false;
    const auto& gs = gpu_state(gpu_idx);
    auto it = gs.residents.find(key);
    if (it == gs.residents.end()) return false;
    return it->second.lock_count > 0;
}

bool ExpertCache::evict(ExpertKey key, int gpu_idx) {
    auto& gs = gpu_state(gpu_idx);
    auto it = gs.residents.find(key);
    if (it == gs.residents.end()) return false;

    const auto& entry = it->second;

    // #90: refuse to evict locked experts.
    if (entry.lock_count > 0) return false;
    const bool was_stable = (entry.zone == CacheZone::kStable);
    if (entry.zone == CacheZone::kStable) {
        gs.stable_slots.free(entry.slot_idx);
    } else {
        // Streaming zone: return to correct sub-allocator
        if (entry.in_spill_zone) {
            // During spill mode, don't return to frozen spill slots
            if (gs.mode != PrefillMode::kSpillActive) {
                gs.spill_slots.free(entry.slot_idx);
            }
        } else {
            gs.prefetch_slots.free(entry.slot_idx);
        }
    }

    if (entry.is_duplicate) --gs.duplicate_count_;

    gs.residents.erase(it);

    // TD-EVICT-BOARD-DESYNC: notify the listener that a stable-zone resident left
    // (covers ordinary evicts AND the cancel / never-arrived path, which routes
    // through this same evict()). Streaming evicts are not tracked.
    if (listener_ && was_stable)
        listener_->on_resident_removed(key, gpu_idx);
    return true;
}

// ── Queries ─────────────────────────────────────────────────────────────────

bool ExpertCache::is_resident(ExpertKey key, int gpu_idx) const {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size())) return false;
    return gpu_state(gpu_idx).residents.contains(key);
}

bool ExpertCache::is_resident(ExpertKey key) const {
    for (const auto& gs : gpus_) {
        if (gs.residents.contains(key)) return true;
    }
    return false;
}

const CacheEntry* ExpertCache::lookup(ExpertKey key, int gpu_idx) const {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size())) return nullptr;
    const auto& gs = gpu_state(gpu_idx);
    auto it = gs.residents.find(key);
    if (it == gs.residents.end()) return nullptr;
    return &it->second;
}

std::vector<int> ExpertCache::resident_gpus(ExpertKey key) const {
    std::vector<int> result;
    for (int i = 0; i < static_cast<int>(gpus_.size()); ++i) {
        if (gpus_[i].residents.contains(key)) result.push_back(i);
    }
    return result;
}

std::vector<ResidencyInfo> ExpertCache::residency_snapshot() const {
    std::vector<ResidencyInfo> result;
    for (const auto& gs : gpus_) {
        result.reserve(result.size() + gs.residents.size());
        for (const auto& [k, e] : gs.residents) {
            result.push_back({e.key, e.gpu_idx, e.zone, e.is_duplicate,
                              e.sub_components_ready, e.vram_address});
        }
    }
    return result;
}

std::vector<ResidencyInfo> ExpertCache::residency_snapshot(int gpu_idx) const {
    std::vector<ResidencyInfo> result;
    const auto& gs = gpu_state(gpu_idx);
    result.reserve(gs.residents.size());
    for (const auto& [k, e] : gs.residents) {
        result.push_back({e.key, e.gpu_idx, e.zone, e.is_duplicate,
                          e.sub_components_ready, e.vram_address});
    }
    return result;
}

// ── Zone management ─────────────────────────────────────────────────────────

bool ExpertCache::promote(ExpertKey key, int gpu_idx) {
    auto it = gpu_state(gpu_idx).residents.find(key);
    if (it == gpu_state(gpu_idx).residents.end()) return false;
    if (it->second.zone != CacheZone::kStreaming) return false;
    it->second.zone = CacheZone::kStable;
    // TD-EVICT-BOARD-DESYNC: promotion makes this a stable-zone resident.
    if (listener_) listener_->on_resident_added(key, gpu_idx);
    return true;
}

bool ExpertCache::demote(ExpertKey key, int gpu_idx) {
    auto it = gpu_state(gpu_idx).residents.find(key);
    if (it == gpu_state(gpu_idx).residents.end()) return false;
    if (it->second.zone != CacheZone::kStable) return false;
    it->second.zone = CacheZone::kStreaming;
    // TD-EVICT-BOARD-DESYNC: demotion removes this from the stable zone.
    if (listener_) listener_->on_resident_removed(key, gpu_idx);
    return true;
}

// ── Prefill spill mode ──────────────────────────────────────────────────────

ScratchRegion ExpertCache::enter_spill_mode(int gpu_idx) {
    auto& gs = gpu_state(gpu_idx);
    assert(gs.mode == PrefillMode::kNormal);

    // Evict all spill-zone experts (bookkeeping only — caller handles D2H).
    // #90: skip locked experts — they are in active use by progressive MoE.
    std::vector<ExpertKey> to_evict;
    for (const auto& [k, e] : gs.residents) {
        if (e.zone == CacheZone::kStreaming && e.in_spill_zone
            && e.lock_count == 0) {
            to_evict.push_back(k);
        }
    }
    for (const auto& k : to_evict) {
        auto it = gs.residents.find(k);
        if (it->second.is_duplicate) --gs.duplicate_count_;
        gs.residents.erase(it);
    }

    // Freeze spill slots — clear free list (no allocations allowed)
    gs.spill_slots.free_list.clear();

    gs.mode = PrefillMode::kSpillActive;
    return gs.scratch;
}

void ExpertCache::exit_spill_mode(int gpu_idx) {
    auto& gs = gpu_state(gpu_idx);
    assert(gs.mode == PrefillMode::kSpillActive);

    // Re-init spill slots free list (all slots free)
    gs.spill_slots.free_list.clear();
    gs.spill_slots.free_list.reserve(gs.spill_slots.total_slots_);
    for (int i = gs.spill_slots.total_slots_ - 1; i >= 0; --i) {
        gs.spill_slots.free_list.push_back(i);
    }

    gs.mode = PrefillMode::kNormal;
}

PrefillMode ExpertCache::prefill_mode(int gpu_idx) const {
    return gpu_state(gpu_idx).mode;
}

const ScratchRegion& ExpertCache::spill_scratch(int gpu_idx) const {
    return gpu_state(gpu_idx).scratch;
}

// ── Eviction support ────────────────────────────────────────────────────────

std::vector<ExpertEvictionInput> ExpertCache::eviction_inputs(
    int gpu_idx) const {
    const auto& gs = gpu_state(gpu_idx);
    std::vector<ExpertEvictionInput> result;
    result.reserve(gs.residents.size());
    for (const auto& [k, e] : gs.residents) {
        ExpertEvictionInput inp;
        inp.key = e.key;
        inp.zone = e.zone;
        inp.is_duplicate = e.is_duplicate;
        inp.gpu_idx = gpu_idx;
        result.push_back(inp);
    }
    return result;
}

std::vector<ExpertEvictionInput> ExpertCache::eviction_inputs(
    int gpu_idx, CacheZone zone) const {
    const auto& gs = gpu_state(gpu_idx);
    std::vector<ExpertEvictionInput> result;
    for (const auto& [k, e] : gs.residents) {
        if (e.zone != zone) continue;
        ExpertEvictionInput inp;
        inp.key = e.key;
        inp.zone = e.zone;
        inp.is_duplicate = e.is_duplicate;
        inp.gpu_idx = gpu_idx;
        result.push_back(inp);
    }
    return result;
}

// ── Duplication ─────────────────────────────────────────────────────────────

double ExpertCache::duplication_benefit(const DuplicationInput& input) {
    return input.frequency * input.avg_cross_gpu_latency
         - input.vram_cost * input.eviction_pressure;
}

bool ExpertCache::can_duplicate(int target_gpu) const {
    if (!duplication_enabled_) return false;
    const auto& gs = gpus_[target_gpu];
    int total = gs.stable_slots.total_slots_ +
                gs.spill_slots.total_slots_ +
                gs.prefetch_slots.total_slots_;
    if (total == 0) return false;
    double dup_fraction = static_cast<double>(gs.duplicate_count_) / total;
    return dup_fraction < max_duplicated_fraction_;
}

int ExpertCache::duplicate_count(int gpu_idx) const {
    return gpu_state(gpu_idx).duplicate_count_;
}

// ── Affinity hints ──────────────────────────────────────────────────────────

void ExpertCache::invalidate_affinity_hints() {
    affinity_hints_valid_ = false;
}

bool ExpertCache::affinity_hints_valid() const {
    return affinity_hints_valid_;
}

// ── Capacity queries ────────────────────────────────────────────────────────

int ExpertCache::free_slots(int gpu_idx, CacheZone zone) const {
    const auto& gs = gpu_state(gpu_idx);
    if (zone == CacheZone::kStable) {
        return gs.stable_slots.free_count();
    }
    // Streaming: sum of both sub-allocators (spill frozen during spill mode)
    if (gs.mode == PrefillMode::kSpillActive) {
        return gs.prefetch_slots.free_count();
    }
    return gs.spill_slots.free_count() + gs.prefetch_slots.free_count();
}

int ExpertCache::total_slots(int gpu_idx, CacheZone zone) const {
    const auto& gs = gpu_state(gpu_idx);
    if (zone == CacheZone::kStable) {
        return gs.stable_slots.total_slots_;
    }
    return gs.spill_slots.total_slots_ + gs.prefetch_slots.total_slots_;
}

int ExpertCache::used_slots(int gpu_idx, CacheZone zone) const {
    return total_slots(gpu_idx, zone) - free_slots(gpu_idx, zone);
}

int ExpertCache::gpu_count() const {
    return static_cast<int>(gpus_.size());
}

int64_t ExpertCache::expert_bytes() const {
    return expert_bytes_;
}

int ExpertCache::total_resident() const {
    int total = 0;
    for (const auto& gs : gpus_) {
        total += static_cast<int>(gs.residents.size());
    }
    return total;
}

}  // namespace layerstorm::memory
