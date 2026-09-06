#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "config/config_parser.h"
#include "core/memory/eviction_policy.h"
#include "core/memory/expert_zone_math.h"
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

    /// 44z: id of the elastic zone backing this slot, or -1 = boot carve.
    /// slot_idx is zone-local when this is >= 0.
    int elastic_zone = -1;

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
///
/// ── Elastic zones (44z: KV <-> expert zone rebalancing) ─────────────────────
///
/// Beyond the boot carve, a GPU may hold additional ELASTIC ZONES: stable-
/// capacity slot regions handed in at runtime by the 44z rebalancer, each
/// backing onto a contiguous slab run granted from the shared KV pool
/// (geometry per ExpertZoneGeometry in expert_zone_math.h). Semantics:
///
/// - An elastic zone's entries are ORDINARY stable-zone residents: valid and
///   dispatchable until the moment they are evicted through the ordinary
///   evict() path (metadata-only, listener fires as today). There is NO
///   intermediate "resident but invalid" state. The reclaim hazard windows
///   (in-flight DMA, cached pointer tables, CUDA-graph staging) are closed
///   OUTSIDE the cache, by the rebalancer's quiesce+barrier protocol and by
///   the generation counter below.
/// - A DRAINING zone stops accepting new reserves immediately; its residents
///   are evicted lazily by the rebalancer's drain loop (ready+unlocked ones at
///   once, locked or transfer-pending ones as they become evictable).
/// - elastic_generation() is the ELASTIC TOPOLOGY GENERATION — the one cheap
///   staleness discriminator for anything that snapshots elastic-affected
///   state across daemon-loop cycles. It bumps on every add_elastic_zone(),
///   begin_drain_elastic_zone() and remove_elastic_zone(), so "generation
///   unchanged" proves BOTH properties its two consumer classes need:
///     * CAPACITY consumers (the REEF solver caps, TD-KVXP-CAPACITY-REPUBLISH)
///       — total_slots(kStable) is unchanged since the last read, so a
///       boot-latched per-GPU cap is still the truth;
///     * ADDRESS consumers (anything caching slot addresses — moe_big's
///       per-superchunk pointer tables, moe_driver's CUDA-graph pinned
///       staging) — no elastic capacity was retired, so every cached address
///       still points at live capacity. For this class the add_elastic_zone
///       bump merely OVER-invalidates (added capacity cannot invalidate an
///       address), which is safe; under-invalidating is not — and the
///       capacity class strictly needs the add bump, so one counter serves
///       both.
///
/// With no elastic zones registered, every operation takes the same branches
/// with the same outcomes as before 44z — the boot-carve baseline is unchanged.
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

    // ── Elastic zones (44z) ─────────────────────────────────────────────

    /// Drain progress for one elastic zone.
    struct ElasticDrainStatus {
        int residents = 0;    ///< Entries still holding a slot in the zone.
        bool drained = false; ///< residents == 0 (false for an unknown id).
    };

    /// Register a new ACTIVE elastic zone over a granted slab run and return
    /// its per-GPU id. `base`/`bytes` describe the whole granted region;
    /// `slot_stride` is ExpertZoneGeometry::slot_stride() and `num_slots` the
    /// slot count the caller derived from the SAME geometry. Slot 0 is placed
    /// at align_up(base, kExpertZoneSlotAlign) (the 716-misalignment lesson),
    /// so the aligned slots are guaranteed to fit inside `bytes`.
    /// `cookie` is an opaque caller value (the rebalancer maps zone -> slab
    /// run with it); the cache never interprets it.
    /// Bumps elastic_generation(): total_slots(kStable) grows, and capacity
    /// consumers latching it must see the change (TD-KVXP-CAPACITY-REPUBLISH).
    /// For address-caching consumers the bump merely over-invalidates, which
    /// is safe.
    int add_elastic_zone(int gpu_idx, void* base, int64_t bytes,
                         int64_t slot_stride, int num_slots, int64_t cookie);

    /// Move a zone kActive -> kDraining: it stops accepting new reserves
    /// immediately, but its residents stay ordinary valid stable residents
    /// until the caller evicts them. Idempotent (true if already draining);
    /// false for an unknown id. Bumps elastic_generation() on success.
    bool begin_drain_elastic_zone(int gpu_idx, int zone_id);

    /// Drain progress; {0, false} for an unknown id.
    ElasticDrainStatus elastic_drain_status(int gpu_idx, int zone_id) const;

    /// Keys of every entry still holding a slot in the zone, ascending — the
    /// worklist for the rebalancer's drain loop. Empty for an unknown id.
    std::vector<ExpertKey> elastic_zone_residents(int gpu_idx,
                                                  int zone_id) const;

    /// Erase a fully drained zone (the caller may then release its slab run).
    /// Returns false and logs on an unknown id or a zone with residents left.
    /// Bumps elastic_generation() on success.
    bool remove_elastic_zone(int gpu_idx, int zone_id);

    /// Number of registered elastic zones on a GPU (active + draining).
    int elastic_zone_count(int gpu_idx) const;

    /// The cookie handed to add_elastic_zone(); 0 for an unknown id.
    int64_t elastic_zone_cookie(int gpu_idx, int zone_id) const;

    /// The elastic TOPOLOGY generation: bumps on add_elastic_zone,
    /// begin_drain_elastic_zone and remove_elastic_zone. Unchanged since a
    /// previous read proves (a) stable capacity (total_slots(kStable)) has
    /// not changed — the capacity-republish discriminator
    /// (TD-KVXP-CAPACITY-REPUBLISH) — and (b) no elastic capacity was
    /// retired, so every cached slot address still points at a live region
    /// (the add bump only over-invalidates this class, which is safe).
    uint64_t elastic_generation() const { return elastic_generation_; }

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

    /// kStable counts the boot carve plus every ACTIVE elastic zone. DRAINING
    /// zones are excluded from BOTH free and total, so their still-resident
    /// entries deliberately vanish from used_slots = total - free: occupancy
    /// stats UNDER-report during a drain window rather than over-report, which
    /// keeps admission decisions from spending capacity that is on its way out.
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

        /// 44z elastic-zone form: the caller already derived the slot count
        /// from ExpertZoneGeometry, so take base/stride/count directly instead
        /// of floor-dividing a byte span. Free list uses the SAME convention
        /// as init() (LIFO descending — slot 0 pops first).
        void init_with_count(void* base_ptr, int64_t stride, int count);

        int allocate();
        void free(int slot_idx);
        void* address(int slot_idx) const;
        int free_count() const;
    };

    // ── Elastic zone (44z) ──────────────────────────────────────────────

    enum class ElasticZoneState { kActive, kDraining };

    /// One runtime stable-capacity region over a granted slab run.
    struct ElasticZone {
        int id = -1;                 ///< Unique per GPU, monotonically assigned.
        void* base = nullptr;        ///< Region device base (slab-run base).
        int64_t bytes = 0;           ///< Full region bytes.
        int64_t slot_stride = 0;     ///< Aligned stride (ExpertZoneGeometry::slot_stride).
        ElasticZoneState state = ElasticZoneState::kActive;
        SlotAllocator slots;         ///< Over align_up(base, kExpertZoneSlotAlign).
        int resident_entries = 0;    ///< Entries (incl. reserved-not-ready) here.
        int64_t cookie = 0;          ///< Opaque caller cookie (zone -> slab run).
    };

    struct GpuState {
        SlotAllocator stable_slots;
        SlotAllocator spill_slots;       ///< Top of streaming zone (repurposable for prefill)
        SlotAllocator prefetch_slots;    ///< Bottom of streaming zone (permanent)
        PrefillMode mode = PrefillMode::kNormal;
        ScratchRegion scratch;           ///< Valid only during kSpillActive
        std::unordered_map<ExpertKey, CacheEntry> residents;
        int duplicate_count_ = 0;

        /// 44z: extra stable-capacity regions, always kept in ascending id
        /// order (ids are monotonic and erase preserves order), which is the
        /// order reserve() offers them in.
        std::vector<ElasticZone> elastic_zones;
        int next_elastic_id = 0;
    };

    // Elastic-zone lookup by id (nullptr when unknown).
    static ElasticZone* find_elastic_zone(GpuState& gs, int zone_id);
    static const ElasticZone* find_elastic_zone(const GpuState& gs,
                                                int zone_id);

    std::vector<GpuState> gpus_;

    // TD-91f: bounds-checked accessor — throws on invalid gpu_idx.
    GpuState& gpu_state(int gpu_idx);
    const GpuState& gpu_state(int gpu_idx) const;

    int64_t expert_bytes_;
    bool affinity_hints_valid_ = true;

    /// 44z elastic topology generation — see elastic_generation().
    uint64_t elastic_generation_ = 0;

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
