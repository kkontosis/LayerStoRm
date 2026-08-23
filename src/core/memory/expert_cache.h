#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "config/config_parser.h"
#include "core/memory/eviction_policy.h"
#include "core/memory/residency_listener.h"
#include "core/memory/vram_allocator.h"
#include "model/quantization/quant_interface.h"

namespace layerstorm::memory {

// ── Prefill spill mode types ─────────────────────────────────────────────────

enum class PrefillMode { kNormal, kSpillActive };

struct ScratchRegion {
    void* ptr = nullptr;
    int64_t size_bytes = 0;
};

// ── Sub-component readiness ──────────────────────────────────────────────────

/// Bitfield for tracking per-projection transfer completion.
/// Gate ready -> gate GEMM. Gate+Up -> fused gate-up. All -> full expert FFN.
enum SubComponent : uint8_t {
    kGate = 0x01,
    kUp   = 0x02,
    kDown = 0x04,
    kAll  = kGate | kUp | kDown,
};

// ── Cache entry (per-resident expert on a GPU) ──────────────────────────────

struct CacheEntry {
    ExpertKey key;
    int gpu_idx = -1;
    CacheZone zone = CacheZone::kStreaming;
    bool is_duplicate = false;           ///< Non-primary copy (INV-0.4, INV-4.6c).
    bool in_spill_zone = false;          ///< true = spill sub-allocator, false = prefetch.
    uint8_t sub_components_ready = 0;    ///< Bitfield of SubComponent.
    void* vram_address = nullptr;        ///< Base address of expert slot in VRAM.
    int slot_idx = -1;                   ///< Index into zone's slot array.
    uint16_t lock_count = 0;             ///< Eviction lock refcount (#90).

    /// Per-projection byte offsets within the expert slot.
    /// gate at vram_address+0, up at vram_address+gate_offset, etc.
    int64_t gate_offset = 0;
    int64_t up_offset = 0;
    int64_t down_offset = 0;
};

// ── Residency snapshot element (lightweight read-only view) ─────────────────

struct ResidencyInfo {
    ExpertKey key;
    int gpu_idx = -1;
    CacheZone zone = CacheZone::kStreaming;
    bool is_duplicate = false;
    uint8_t sub_components_ready = 0;
    void* vram_address = nullptr;
};

// ── Duplication evaluation input ────────────────────────────────────────────

struct DuplicationInput {
    ExpertKey key;
    int target_gpu = -1;
    double frequency = 0.0;              ///< Normalized [0,1].
    double frequency_percentile = 0.0;   ///< 0-100.
    double avg_cross_gpu_latency = 0.0;  ///< Seconds (estimated).
    double vram_cost = 0.0;              ///< Expert bytes / zone capacity.
    double eviction_pressure = 0.0;      ///< Fraction of zone occupied [0,1].
};

// ── Expert cache ────────────────────────────────────────────────────────────

/// Expert weight cache managing routed expert residency in GPU VRAM.
///
/// Manages two zones per GPU (stable + streaming) within pre-allocated VRAM
/// regions from VramAllocator. Tracks per-expert sub-component readiness for
/// sub-component readiness tracking. Supports expert duplication across GPUs.
///
/// Owned and exclusively mutated by the single-threaded orchestrator (INV-4.6a).
/// Not thread-safe (INV-3.4.2).
///
/// Two-phase protocol: reserve() allocates a slot (no DMA), then mark_ready()
/// is called after transfer completion (INV-4.5.1).
class ExpertCache {
public:
    /// Construct from VRAM allocator, config, and expert byte size.
    /// Per-projection offsets default to proportional thirds.
    ExpertCache(const VramAllocator& vram,
                const config::Config& cfg,
                int64_t expert_bytes);

    /// Construct with precise per-projection offsets from QuantInterface.
    ExpertCache(const VramAllocator& vram,
                const config::Config& cfg,
                int64_t expert_bytes,
                const model::QuantInterface& quant,
                const model::ExpertShape& shape);

    // ── Core operations ─────────────────────────────────────────────────

    /// Reserve a slot for an expert on a GPU in a given zone.
    /// Returns the VRAM base address for the expert slot.
    /// Returns nullptr if the zone is full (caller must evict first).
    /// Entry created with sub_components_ready=0 (INV-4.5.1: no DMA here).
    void* reserve(ExpertKey key, int gpu_idx, CacheZone zone,
                  bool is_duplicate = false);

    /// Mark a sub-component as ready after its transfer completes.
    void mark_ready(ExpertKey key, int gpu_idx, SubComponent component);

    /// Mark all sub-components ready at once (for bulk loads).
    void mark_all_ready(ExpertKey key, int gpu_idx);

    /// Evict an expert from a specific GPU. Frees the slot.
    /// Returns false if the expert was not resident, or if locked (#90).
    bool evict(ExpertKey key, int gpu_idx);

    // ── Eviction lock (#90) ─────────────────────────────────────────────

    /// Increment eviction lock refcount. While lock_count > 0, evict()
    /// refuses to remove this expert. Returns false if not resident.
    bool lock(ExpertKey key, int gpu_idx);

    /// Decrement eviction lock refcount. Returns false if not resident
    /// or lock_count was already 0.
    bool unlock(ExpertKey key, int gpu_idx);

    /// Query whether an expert is locked against eviction.
    bool is_locked(ExpertKey key, int gpu_idx) const;

    // ── Queries ─────────────────────────────────────────────────────────

    /// Check if an expert is resident on a specific GPU.
    bool is_resident(ExpertKey key, int gpu_idx) const;

    /// Check if an expert is resident on any GPU.
    bool is_resident(ExpertKey key) const;

    /// Get the cache entry for a resident expert. Returns nullptr if not found.
    const CacheEntry* lookup(ExpertKey key, int gpu_idx) const;

    /// Get all GPU indices where an expert is resident.
    std::vector<int> resident_gpus(ExpertKey key) const;

    /// Full residency snapshot across all GPUs.
    std::vector<ResidencyInfo> residency_snapshot() const;

    /// Per-GPU residency snapshot.
    std::vector<ResidencyInfo> residency_snapshot(int gpu_idx) const;

    // ── Zone management (metadata-only, no data transfer) ───────────────

    /// Promote an expert from streaming to stable zone (zone tag change only).
    /// Returns false if the expert is not in the streaming zone on that GPU.
    bool promote(ExpertKey key, int gpu_idx);

    /// Demote an expert from stable to streaming zone (zone tag change only).
    /// Returns false if the expert is not in the stable zone on that GPU.
    bool demote(ExpertKey key, int gpu_idx);

    // ── Eviction support ────────────────────────────────────────────────

    /// Build ExpertEvictionInput entries for all residents on a GPU.
    /// Fills key, zone, is_duplicate, gpu_id from cache state.
    /// Scoring terms (recency, frequency, etc.) left at 0.0 for the caller.
    std::vector<ExpertEvictionInput> eviction_inputs(int gpu_idx) const;

    /// Build eviction inputs for a specific zone on a GPU.
    std::vector<ExpertEvictionInput> eviction_inputs(
        int gpu_idx, CacheZone zone) const;

    // ── Duplication ─────────────────────────────────────────────────────

    /// Evaluate duplication benefit (spec section 4.6 formula).
    /// Returns positive value if duplication is beneficial.
    static double duplication_benefit(const DuplicationInput& input);

    /// Check if duplicating would exceed max_duplicated_fraction on target GPU.
    bool can_duplicate(int target_gpu) const;

    /// Count of duplicate entries on a GPU.
    int duplicate_count(int gpu_idx) const;

    // ── Prefill spill mode ───────────────────────────────────────────────

    /// Enter spill mode: evict all spill-zone experts (bookkeeping only —
    /// caller handles D2H transfers), freeze spill slot allocation, return
    /// spill zone as scratch. The returned ScratchRegion.ptr is contiguous
    /// with the KV main scratch tail above it.
    ScratchRegion enter_spill_mode(int gpu_idx);

    /// Exit spill mode: restore spill zone for expert cache, re-init free list.
    void exit_spill_mode(int gpu_idx);

    /// Current prefill mode for a GPU.
    PrefillMode prefill_mode(int gpu_idx) const;

    /// Get the spill scratch region (valid only during kSpillActive).
    const ScratchRegion& spill_scratch(int gpu_idx) const;

    // ── Affinity hints ──────────────────────────────────────────────────

    /// Invalidate all affinity hints. Called on workload shift (INV-0.8c).
    void invalidate_affinity_hints();

    /// Whether affinity hints are currently valid.
    bool affinity_hints_valid() const;

    // ── Capacity queries ────────────────────────────────────────────────

    int free_slots(int gpu_idx, CacheZone zone) const;
    int total_slots(int gpu_idx, CacheZone zone) const;
    int used_slots(int gpu_idx, CacheZone zone) const;
    int gpu_count() const;
    int64_t expert_bytes() const;

    /// Total bytes per projection (weight + scales + scalar scales).
    int64_t gate_bytes() const { return gate_bytes_; }
    int64_t up_bytes() const { return up_bytes_; }
    int64_t down_bytes() const { return down_bytes_; }

    /// Weight-only bytes per projection (for computing B/scale_B split within slot).
    /// Returns 0 when constructed without QuantInterface.
    int64_t gate_weight_bytes() const { return gate_weight_bytes_; }
    int64_t up_weight_bytes() const { return up_weight_bytes_; }
    int64_t down_weight_bytes() const { return down_weight_bytes_; }

    /// Total resident experts across all GPUs (including duplicates).
    int total_resident() const;

    // ── Residency listener (TD-EVICT-BOARD-DESYNC) ──────────────────────
    /// Register a residency-change listener (nullable). Default is nullptr,
    /// in which case every cache operation is BYTE-IDENTICAL to having no
    /// listener at all (no fire) — the loader-off baseline is unaffected. When
    /// set, the cache fires on_resident_added / on_resident_removed at its
    /// stable-zone membership choke-points (reserve/promote → added; evict/demote
    /// → removed), so the listener authoritatively mirrors stable residency.
    void set_residency_listener(ResidencyListener* listener) {
        listener_ = listener;
    }
    ResidencyListener* residency_listener() const { return listener_; }

private:
    struct SlotAllocator {
        void* base = nullptr;
        int64_t slot_bytes = 0;
        int total_slots_ = 0;
        std::vector<int> free_list;

        void init(void* base_ptr, int64_t bytes_per_slot, int64_t zone_bytes);
        int allocate();
        void free(int slot_idx);
        void* address(int slot_idx) const;
        int free_count() const;
    };

    struct GpuState {
        SlotAllocator stable_slots;
        SlotAllocator spill_slots;       ///< Top of streaming zone (repurposable for prefill)
        SlotAllocator prefetch_slots;    ///< Bottom of streaming zone (permanent)
        PrefillMode mode = PrefillMode::kNormal;
        ScratchRegion scratch;           ///< Valid only during kSpillActive
        std::unordered_map<ExpertKey, CacheEntry> residents;
        int duplicate_count_ = 0;
    };

    std::vector<GpuState> gpus_;

    // TD-91f: bounds-checked accessor — throws on invalid gpu_idx.
    GpuState& gpu_state(int gpu_idx);
    const GpuState& gpu_state(int gpu_idx) const;

    int64_t expert_bytes_;
    bool affinity_hints_valid_ = true;

    // Residency-change listener (nullable; default null = no notification, the
    // byte-identical baseline). Fired only at the stable-zone choke-points.
    ResidencyListener* listener_ = nullptr;

    // Per-projection byte counts (computed once in constructor).
    int64_t gate_bytes_ = 0;
    int64_t up_bytes_ = 0;
    int64_t down_bytes_ = 0;

    // Weight-only bytes per projection (excluding group scales and scalar scales).
    // Used by the dispatcher to compute B/scale_B split within each projection slot.
    // Zero when constructed without QuantInterface (test-only path).
    int64_t gate_weight_bytes_ = 0;
    int64_t up_weight_bytes_ = 0;
    int64_t down_weight_bytes_ = 0;

    // Duplication config.
    bool duplication_enabled_;
    double max_duplicated_fraction_;
    int duplication_freq_threshold_percentile_;
};

}  // namespace layerstorm::memory
