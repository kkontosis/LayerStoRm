#pragma once

// PinnedExpertArena — one pinned, NUMA-bound anonymous arena per GPU-attached
// NUMA node, slab-managed (P-24). Replaces the 481-1 per-file mmap registration
// path: instead of cudaHostRegister-ing each prepacked expert *file mmap*
// individually (~5.7 s/GB, TD-100c), lock one big anonymous arena per node,
// register it ONCE with cudaHostRegisterPortable (~tens of ms/GB), and manage
// it as fixed expert slots with LRU eviction + a multi-GPU in-flight refcount.
//
// Topology (ties to P-22): each GPU-attached NUMA node gets exactly ONE arena,
// sized to that node's expert budget. A node backing two GPUs (share_degree=2)
// keeps a SINGLE shared arena — no 50/50 split — so both GPUs DMA the one copy
// of any expert homed on that node (dedup), and the refcount must count both
// GPUs' outstanding H2Ds (a slot frees only when neither GPU is mid-copy).
//
// This is the host warm tier of the SSD → pinned-arena → VRAM 3-tier model:
// preload = a plain file read into a pre-pinned slot; the H2D DMAs straight
// from the slot at full PCIe bandwidth (pinned, NUMA-local).
//
// Thread safety: reserve/touch/free mutate slab state and are NOT thread-safe;
// call them only from the daemon thread (INV-ELM-1). The in-flight refcount
// (acquire_inflight/release_inflight) IS atomic so transfer-completion
// callbacks on other threads can decrement it.

#include <atomic>
#include <cstddef>
#include <functional>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "core/memory/eviction_policy.h"   // ExpertKey
#include "core/memory/numa_manager.h"       // NumaManager, NumaBuffer

namespace layerstorm::model { class PrepackedSource; }

namespace layerstorm::memory {

class ArenaLoader;  // fwd decl: optional io_uring batch reader for preload
class ArenaCache;   // fwd decl: P-24b persistence meta layer (arena_cache.h)

/// Sentinel for "no slot was evicted" returned via PinnedNodeArena::reserve's
/// evicted_out param (a real key never has these maxed-out indices).
inline constexpr ExpertKey kNoEvictedKey{UINT32_MAX, UINT16_MAX};

/// P-24b: how a node arena's RAM is backed.
///  - kPrivate:      MAP_PRIVATE|MAP_ANONYMOUS (classic, process-lifetime).
///  - kSharedCreate: fresh MAP_SHARED memfd segments (fds then STOREd with the
///                   arena holder so the pages survive this process).
///  - kAdopt:        pre-existing memfd segments received from the holder
///                   (warm attach). Content is PRESERVED: the parallel memset
///                   prefault MUST NOT run (pages are resident already).
enum class ArenaBackingMode { kPrivate, kSharedCreate, kAdopt };

/// One planned node arena (geometry). Public so the engine can compute the
/// slab layout BEFORE building the arena (cold boots), and so a warm attach
/// can carry the STORED geometry into ArenaBacking::adopt_plans.
struct ArenaNodePlan {
    int    node = -1;
    bool   is_spill = false;
    size_t num_slots = 0;
    int    share_degree = 1;
};

/// P-24b: backing selection for PinnedExpertArena. For kAdopt, `adopted` must
/// hold exactly one NumaBuffer per adopted node (ownership moves into the
/// arena) and `adopt_plans` carries the STORED geometry — it OVERRIDES
/// plan_geometry entirely, because resolved slot counts derive from volatile
/// free RAM (fraction_free banks drift ±slots run-to-run) while adoption
/// allocates nothing and thus always fits the stored layout. For the other
/// modes both fields are ignored.
struct ArenaBacking {
    ArenaBackingMode mode = ArenaBackingMode::kPrivate;
    std::unordered_map<int, NumaBuffer> adopted;  // node → segment (kAdopt)
    std::vector<ArenaNodePlan> adopt_plans;       // stored geometry (kAdopt)
};

/// A single pinned, NUMA-bound, slab-managed arena for one NUMA node.
class PinnedNodeArena {
public:
    /// Allocate `extra_scratch_bytes (page-rounded) + num_slots *
    /// slot_size_bytes` anonymous bytes bound to `numa_node` (via NumaManager),
    /// then page-lock the whole arena with ONE cudaHostRegisterPortable.
    /// `share_degree` records how many GPUs source from this node (≥1; 2 for the
    /// shared node) — used by the multi-GPU in-flight refcount semantics (purely
    /// informational here; the refcount is per-slot and counts any GPU).
    /// `extra_scratch_bytes` (default 0) reserves a GENERAL shared scratch region
    /// at the FRONT of the slab (before slot 0); when 0 the layout is byte-for-byte
    /// the prior behaviour. Throws std::runtime_error if registration fails or the
    /// node has insufficient free RAM.
    /// When `defer_register` is true the ctor allocates + mbinds the arena but
    /// SKIPS the cudaHostRegister; the caller prefaults the pages NUMA-local (in
    /// parallel across nodes) and then calls finalize_registration(). This makes
    /// registration ~6× cheaper (it otherwise faults every page serially under a
    /// per-mm lock). Default false = allocate + register in one shot.
    /// P-24b `backing`/`adopted`: kSharedCreate allocates the slab as a fresh
    /// MAP_SHARED memfd (NumaManager::allocate_on_node_shared); kAdopt takes
    /// ownership of `adopted` (a holder-received segment mapped via
    /// NumaManager::adopt_shared) — its size must cover the slab, its CONTENT
    /// is preserved (no prefault/zeroing here or in the facade).
    PinnedNodeArena(NumaManager& numa, int numa_node, size_t slot_size_bytes,
                    size_t num_slots, int share_degree,
                    size_t extra_scratch_bytes = 0, bool defer_register = false,
                    ArenaBackingMode backing = ArenaBackingMode::kPrivate,
                    NumaBuffer adopted = {});
    ~PinnedNodeArena();

    /// Page-lock the arena (cudaHostRegister Portable) if not already registered.
    /// Idempotent. Throws std::runtime_error on failure (frees the arena first).
    /// Called by the ctor unless `defer_register` was set.
    void finalize_registration();

    PinnedNodeArena(const PinnedNodeArena&) = delete;
    PinnedNodeArena& operator=(const PinnedNodeArena&) = delete;
    PinnedNodeArena(PinnedNodeArena&&) = delete;
    PinnedNodeArena& operator=(PinnedNodeArena&&) = delete;

    int    node() const { return numa_node_; }
    int    share_degree() const { return share_degree_; }
    size_t slot_size() const { return slot_size_bytes_; }
    size_t num_slots() const { return slots_.size(); }
    void*  base() const { return base_; }
    size_t total_bytes() const { return total_bytes_; }
    bool   registered() const { return registered_; }

    /// General shared scratch region at the FRONT of this node's arena slab
    /// (reserved before slot 0). nullptr if extra_scratch_bytes was 0. This memory
    /// is pinned + mbind'd to this node like the rest of the arena. The first
    /// consumer is MultiNumaCpuExpertDevice (staging per-node weight slices);
    /// callers coordinate sub-allocation themselves (it is raw shared space).
    void*  node_scratch() const { return scratch_bytes_ ? base_ : nullptr; }
    /// Byte span of node_scratch() (the requested bytes, page-rounded). 0 when no
    /// scratch was reserved.
    size_t node_scratch_bytes() const { return scratch_bytes_; }

    /// Resolve a resident expert's slot pointer, or nullptr if not resident.
    /// Does NOT update LRU (use touch() for that). const + cheap.
    void* resolve(ExpertKey key) const;

    /// True if `key` currently occupies a slot.
    bool resident(ExpertKey key) const { return index_.find(key) != index_.end(); }

    /// Reserve a slot for `key` and return its pointer (the caller then reads
    /// the expert's bytes into it). If `key` is already resident, returns its
    /// existing slot (LRU-touched). Otherwise picks a free slot, or evicts the
    /// LRU slot whose in-flight refcount is 0 AND is not mid async-load (a slot
    /// mid-H2D or mid-load is never evicted — multi-GPU + J-1 safe). Returns
    /// nullptr only if every slot is pinned in-flight or loading (arena
    /// momentarily full). The slot's bytes are uninitialised on a fresh reserve;
    /// `ready` is left false until mark_ready. A fresh reserve also sets the
    /// slot's `loading` flag false — the caller flips it via mark_loading() when
    /// it submits an async fill (J-1), so a concurrent reserve() won't re-pick it.
    /// If `evicted_out` is non-null, it is set to the key of the slot that was
    /// evicted to make room (so a caller maintaining a cross-arena location index
    /// can drop the stale entry), or `kNoEvictedKey` if a free slot was used or
    /// the key was already resident.
    void* reserve(ExpertKey key, ExpertKey* evicted_out = nullptr);

    /// True if this arena has at least one unoccupied slot (can take a new expert
    /// without evicting). Used by the facade to prefer extend-over-evict.
    bool has_free_slot() const { return index_.size() < slots_.size(); }

    /// J-1: mark a reserved slot as having an async file→slot read in flight, so
    /// pick_slot() will not evict/reuse it until the load completes. No-op if
    /// key is not resident.
    void mark_loading(ExpertKey key);

    /// J-1: clear the loading flag (async read completed or failed) WITHOUT
    /// marking the slot ready. After a failed load, the slot is loadable again /
    /// evictable. No-op if key is not resident.
    void clear_loading(ExpertKey key);

    /// J-1: true if the slot for `key` is resident AND has an async load in
    /// flight (reserved, not yet ready). False if not resident.
    bool is_loading(ExpertKey key) const;

    /// Mark a reserved slot as holding valid (fully-loaded) data, so resolve()
    /// returns it. Also clears the `loading` flag (J-1). No-op if key is not
    /// resident.
    void mark_ready(ExpertKey key);

    /// TD-GOLDEN bug #6: hook invoked on the not-ready -> ready transition in
    /// mark_ready — used to reformat NVFP4 weight scales in the freshly
    /// filled slot into the Sm1xx interleaved SFB layout the grouped GEMM
    /// consumes. Runs on the filling thread (CPU-only work).
    void set_post_fill(std::function<void(ExpertKey, void*)> hook) {
        post_fill_ = std::move(hook);
    }

    /// True if the slot for `key` is resident AND its data was marked ready.
    bool is_ready(ExpertKey key) const;

    /// M3b online migration: drop `key`'s slot (occupied=false) WITHOUT
    /// touching its bytes or its ArenaCache record. Refuses (returns false)
    /// if the key is not resident, the slot has an in-flight H2D
    /// (inflight>0) or an async load in flight (`loading`). The record
    /// intentionally stays as-is — an ACTIVE record over intact bytes is
    /// still truthful (the "free cell still holds it" try_adopt contract);
    /// the next reserve() of the slot stamps it EMPTY before any byte is
    /// overwritten (INV-ARENA-CACHE-ORDER). Daemon thread only.
    bool evict(ExpertKey key);

    /// P-24b warm attach: occupy slot `slot_idx` for `key` as resident+READY
    /// WITHOUT any load — the slot's bytes were preserved across runs (kAdopt
    /// backing) and validated by ArenaCache. The slot must be unoccupied and
    /// `key` not already resident; returns false otherwise. Does NOT touch the
    /// cache record (it is already kActive — that's why we're adopting).
    bool adopt_slot(ExpertKey key, size_t slot_idx);

    /// True if slot `slot_idx` currently holds an expert (bounds-checked;
    /// out-of-range reads as occupied so callers never adopt into it).
    bool slot_occupied(size_t slot_idx) const {
        return slot_idx >= slots_.size() || slots_[slot_idx].occupied;
    }

    /// P-24b: the backing memfd of this arena's slab (-1 for private backing).
    /// Used by the engine to STORE the segment with the arena holder.
    int backing_fd() const { return buf_.fd; }

    /// P-24b: the slab's ACTUAL allocation size (page-aligned, == the memfd's
    /// fstat size for shared backing). Stored in the holder's SegmentDesc so a
    /// later warm attach can validate byte-exactly.
    size_t backing_bytes() const { return buf_.size; }

    /// P-24b: wire the persistence meta layer. The arena stamps records on
    /// every slot state change (INV-ARENA-CACHE-ORDER): reserve-reuse → EMPTY
    /// (before the caller writes bytes), mark_loading → LOADING, mark_ready →
    /// ACTIVE. nullptr (default) = no cache, byte-for-byte prior behavior.
    void set_cache(ArenaCache* cache) { cache_ = cache; }

    /// LRU-touch a resident slot (mark most-recently-used). No-op if not
    /// resident.
    void touch(ExpertKey key);

    /// Increment the in-flight refcount for the slot holding `key` (a GPU is
    /// about to / is copying it). Pins the slot against eviction until released.
    /// Multi-GPU: each GPU's H2D increments independently. No-op if not resident.
    void acquire_inflight(ExpertKey key);

    /// Decrement the in-flight refcount for the slot holding `key` (one GPU's
    /// H2D completed). Atomic — safe from a transfer-completion callback on any
    /// thread. No-op if not resident.
    void release_inflight(ExpertKey key);

    /// Current in-flight refcount for `key` (0 if not resident). For tests.
    int inflight(ExpertKey key) const;

    /// Number of slots currently occupied.
    size_t occupied() const { return index_.size(); }

private:
    struct Slot {
        ExpertKey key{};
        bool      occupied = false;
        bool      ready = false;
        bool      loading = false;   ///< J-1: async file→slot read in flight.
        uint64_t  lru_tick = 0;
        std::atomic<int> inflight{0};

        Slot() = default;
        // Movable so the slot vector can be sized once at construction (before
        // any concurrency). The atomic is copied by value — only ever invoked
        // single-threaded during the initial resize(), never under contention.
        Slot(Slot&& o) noexcept
            : key(o.key), occupied(o.occupied), ready(o.ready),
              loading(o.loading), lru_tick(o.lru_tick),
              inflight(o.inflight.load(std::memory_order_relaxed)) {}
        Slot& operator=(Slot&& o) noexcept {
            key = o.key; occupied = o.occupied; ready = o.ready;
            loading = o.loading; lru_tick = o.lru_tick;
            inflight.store(o.inflight.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
            return *this;
        }
        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;
    };

    NumaManager& numa_;
    int    numa_node_;
    int    share_degree_;
    size_t slot_size_bytes_;
    size_t scratch_bytes_ = 0;   ///< front-of-slab general scratch (0 = none)
    size_t total_bytes_ = 0;
    NumaBuffer buf_{};           ///< owning handle for the mbind'd arena
    void*  base_ = nullptr;      ///< == buf_.data (cached for slot math)
    bool   registered_ = false;
    ArenaCache* cache_ = nullptr;  ///< P-24b meta layer (nullable)

    std::vector<Slot> slots_;
    std::unordered_map<ExpertKey, size_t> index_;
    std::function<void(ExpertKey, void*)> post_fill_;  // key → slot index
    uint64_t lru_clock_ = 0;

    void* slot_ptr(size_t i) const;
    // Find a free slot, or evict the LRU evictable (inflight==0) slot.
    // Returns slots_.size() if none available (all in-flight).
    size_t pick_slot();
};

/// Set of per-node arenas (one per GPU-attached NUMA node) plus the routing
/// from an expert key to its home node (P-22 expert_home_node). Owns the
/// arenas and provides a single resolve/reserve facade used by host_source.
class PinnedExpertArena {
public:
    /// Build one arena per GPU-attached NUMA node. Each node's slot count is
    /// `node_budget_bytes(node) / slot_size_bytes`, where the per-node budget is
    /// `min(total_arena_bytes / num_gpu_attached_nodes, resolve_node_budget_bytes(
    /// sizing, node))` — i.e. the configured per-node ceiling (fraction of
    /// total/available RAM or an absolute amount, with optional per-node
    /// overrides), clamped so the arena never exceeds the full routed set
    /// (INV-4.12f: per-node RAM is the ceiling, not RLIMIT_MEMLOCK).
    /// `share_degree` for each node = number of GPUs whose numa_node maps to it
    /// (≥1; 2 for the shared node). Throws on failure of any node's
    /// allocation/registration.
    /// `spill` (default = disabled) adds a weighted cross-NUMA-node RAM spill
    /// tier: arenas are also built on each spill node (incl. GPU-less nodes),
    /// sized by `spill.sizing_*` (no /total clamp). When disabled this is a
    /// strict no-op (GPU-attached nodes only).
    /// `extra_scratch_bytes` (default 0) reserves a general shared scratch region
    /// at the FRONT of EVERY node arena (home + spill) — see
    /// PinnedNodeArena::node_scratch. 0 = byte-for-byte the prior layout.
    /// `defer_registration` (default false): allocate + prefault the arenas but
    /// SKIP the per-node cudaHostRegister — the caller runs register_all() later,
    /// concurrently with the io_uring preload (the two touch disjoint hardware:
    /// GPU page-tables vs NVMe; both only need the pages faulted, which the
    /// prefault already did). Slots are usable for preload reads while unpinned.
    /// `backing` (P-24b, default null = kPrivate — byte-for-byte the prior
    /// behavior): kSharedCreate backs every node arena with a fresh memfd
    /// (fds exposed via node_backing_fd for the holder STORE); kAdopt consumes
    /// `backing->adopted` (one segment per planned node — the engine validates
    /// the plan against the holder's geometry FIRST) and SKIPS the prefault
    /// memset so the preserved slot bytes survive.
    PinnedExpertArena(NumaManager& numa, size_t slot_size_bytes,
                      size_t total_arena_bytes,
                      const config::PinHostExpertPoolSizingConfig& sizing,
                      const config::CrossNodeSpillConfig& spill = {},
                      size_t extra_scratch_bytes = 0,
                      bool defer_registration = false,
                      ArenaBacking* backing = nullptr);
    ~PinnedExpertArena() = default;

    /// P-24b: compute the exact per-node slab plan this constructor would
    /// build for these parameters — WITHOUT allocating. Deterministic order
    /// (ascending node id). The engine hashes this (ArenaCache::hash_geometry)
    /// to validate an adopted meta segment before construction. Throws the
    /// same sizing errors as the constructor (zero-slot budget, INV-4.12f).
    static std::vector<ArenaNodePlan> plan_geometry(
        NumaManager& numa, size_t slot_size_bytes, size_t total_arena_bytes,
        const config::PinHostExpertPoolSizingConfig& sizing,
        const config::CrossNodeSpillConfig& spill = {});

    /// Page-lock every node arena (cudaHostRegister Portable) if deferred at
    /// construction. Idempotent per node. Throws on registration failure WITHOUT
    /// freeing the arena (a concurrent preload may still hold slot pointers — the
    /// dtor reclaims it). Safe to call concurrently with preload(): it touches
    /// only each arena's base registration state, never the slot index/LRU.
    void register_all();

    PinnedExpertArena(const PinnedExpertArena&) = delete;
    PinnedExpertArena& operator=(const PinnedExpertArena&) = delete;

    /// The node-arena that homes `key` (per P-22 expert_home_node), or nullptr
    /// if no GPU-attached nodes exist or the expert's home node has no arena.
    PinnedNodeArena* arena_for(ExpertKey key);
    const PinnedNodeArena* arena_for(ExpertKey key) const;

    /// Consumer-aware placement (INV-MoE-NUMA): the node-arena local to the GPU
    /// that will COMPUTE the expert (`gpu`'s NUMA node), so the H2D is
    /// local-bandwidth (INV-4.12h). Preferred over `arena_for(key)` — the latter
    /// places by expert_idx alone and is only coincidentally GPU-local
    /// (TD-NUMA-align). Returns nullptr if `gpu`'s node has no arena.
    PinnedNodeArena* arena_for_gpu(int gpu);
    const PinnedNodeArena* arena_for_gpu(int gpu) const;

    /// Resolve a resident, ready expert's slot pointer (nullptr if not resident
    /// or not yet ready). Does not touch LRU.
    void* resolve(ExpertKey key) const;

    /// True if `key` is resident AND ready in its home arena.
    bool is_ready(ExpertKey key) const;

    /// TD-GOLDEN bug #6: install the post-fill hook on ALL node arenas
    /// (home + spill). See PinnedNodeArena::set_post_fill.
    void set_post_fill(std::function<void(ExpertKey, void*)> hook);

    /// Reserve a slot for `key` in its home arena (see PinnedNodeArena::reserve).
    /// nullptr if the home arena is momentarily full of in-flight slots, or no
    /// home arena exists. Updates the location index. Used by preload (home
    /// placement); the offload miss path uses reserve_for_fill instead.
    void* reserve(ExpertKey key);

    /// Tiered placement for an offload miss on computing GPU `gpu`: prefer the
    /// GPU's local-node arena while it has a FREE slot (extend), else spill into
    /// a weighted cross-node arena that has a free slot, else evict the LRU
    /// victim of the weighted-best node (extend-before-evict). Updates the
    /// location index (dropping any evicted key). Sets `*out_node` to the chosen
    /// node (or -1 if every candidate slot is pinned in-flight). Returns the slot
    /// pointer or nullptr. `extend_only` (preload) skips the evict phase — returns
    /// null when no free slot exists anywhere, so a slot is never evicted.
    void* reserve_for_fill(ExpertKey key, int gpu, int* out_node,
                           bool extend_only = false);

    /// Placement-directed EXTEND-ONLY reserve on a SPECIFIC node's arena
    /// (arena host placement policy, arena_placement.h): reserves a slot for
    /// `key` on `node` iff that arena exists and has a FREE slot (never
    /// evicts, never spills elsewhere — the caller falls back to
    /// reserve_for_fill). Already-located keys route to their holding node
    /// (LRU-touch, no move). Updates the location index. nullptr on any miss.
    void* reserve_on_node(ExpertKey key, int node);

    // ── M3b online placement migration (arena_migrator.h) ────────────────
    // Composite slot-relocation primitives (INV-ARENA-MIGRATE). Contract:
    // daemon-thread-only; the moving key stays resident+ready+fetchable at
    // its SOURCE slot for the whole window (migrate_begin pins the source
    // slot via the in-flight refcount, so it cannot be evicted, and
    // `location_` keeps routing every facade path to it); the TARGET slot
    // fills through the standard reserve→mark_loading→mark_ready record
    // discipline (INV-ARENA-CACHE-ORDER holds at every instant); the flip
    // (migrate_commit) happens only when the migration pin is the ONLY
    // outstanding reference on the source slot (no demand H2D mid-read),
    // atomically within one daemon tick. Placement stays performance-only:
    // the target slot receives byte-identical content (same prepacked
    // source read as any cold fill).

    /// Read-only view of the authoritative key→node residency map.
    /// Daemon-thread only (the map mutates on reserve/evict/migrate).
    const std::unordered_map<ExpertKey, int>& locations() const {
        return location_;
    }

    /// TD-BRIDGE-CPP-GAP Q1 trace probe: optional sink invoked on EVERY
    /// `location_` change, from the same (daemon) thread that mutates it.
    /// kind: 'R' reserve/adopt insert, 'E' evict/erase (new_node == -1),
    /// 'C' migrate-commit flip (the online-migrator epoch boundary).
    /// Unset (default) = zero-cost.
    std::function<void(ExpertKey key, int old_node, int new_node, char kind)>
        location_change_sink;

    // ── Epoch-latched bank snapshot (INV-REEF-BANK) ──────────────────────
    // The REEF solver's bank inputs must be FROZEN per placement epoch:
    // per-solve live `location_` reads under the online migrator churn the
    // solver's H2D-cost inputs → placement/residency churn → fetch waits
    // (the measured bridge-vs-C++ gap). Consumers read a PUBLISHED
    // immutable copy of the location map; publication happens on the
    // daemon thread only — once lazily at first consumption (first epoch)
    // and again at every migrate_commit (epoch boundary).

    /// Publish the current `location_` map as the new bank snapshot and
    /// bump the epoch. DAEMON THREAD ONLY (reads `location_` unlocked).
    void publish_bank_snapshot();

    /// Latest published snapshot (nullptr before the first publish) —
    /// raw holding nodes; callers apply HBM→CPU pairing themselves.
    /// Any thread.
    std::shared_ptr<const std::unordered_map<ExpertKey, int>>
    bank_snapshot() const;

    /// Publication epoch (0 = never published). Any thread; cheap —
    /// consumers poll this per solve and refetch only on change.
    uint64_t bank_epoch() const {
        return bank_epoch_.load(std::memory_order_acquire);
    }

    /// Evict `key` wherever it is located (bytes + record left intact — see
    /// PinnedNodeArena::evict). Drops the location entry. False if the key
    /// is not located or its slot is pinned (inflight/loading).
    bool evict_key(ExpertKey key);

    /// Begin migrating `key` to `target_node`: pins the source slot
    /// (acquire_inflight) and reserves+mark_loading()s a FREE slot on the
    /// target arena (never evicts — the caller frees a victim slot first).
    /// Returns the target slot pointer for the caller's async fill, or
    /// nullptr (nothing pinned, nothing reserved) if the key is not
    /// located+ready, the target equals the source, the target arena does
    /// not exist, or it has no free slot.
    void* migrate_begin(ExpertKey key, int target_node);

    /// Outcome of migrate_commit.
    enum class MigrateCommit { kCommitted, kRetry };

    /// After the target slot's bytes are FULLY written: flip residency to
    /// the target. kRetry if a demand H2D still holds the source slot
    /// (inflight > 1) — call again next tick. On kCommitted: the target
    /// slot is mark_ready'd (ACTIVE record — transiently the key has TWO
    /// ACTIVE records, which is crash-safe: both describe identical bytes
    /// and scan_adoptable dedups), the source pin is released, the source
    /// slot evicted (bytes/record intact per evict()), and `location_`
    /// routes to the target — all within this call.
    MigrateCommit migrate_commit(ExpertKey key, int target_node);

    /// Abandon a begun migration (async fill failed): clears the target
    /// slot (record EMPTY via clear_loading — its bytes are undefined) and
    /// releases the source pin. The key remains resident at the source as
    /// if nothing happened.
    void migrate_abort(ExpertKey key, int target_node);

    /// The node currently holding `key` (location index), or -1. For host_source
    /// global resolve, diagnostics, and tests.
    int location_node(ExpertKey key) const;
    /// The arena currently holding `key`, or nullptr.
    PinnedNodeArena* locate(ExpertKey key);
    const PinnedNodeArena* locate(ExpertKey key) const;

    /// Mark a reserved slot ready (data loaded).
    void mark_ready(ExpertKey key);

    // ── P-24b persistence (arena holder + ArenaCache) ────────────────────

    /// Wire the persistence meta layer into every node arena (see
    /// PinnedNodeArena::set_cache). nullptr = disabled.
    void set_cache(ArenaCache* cache);

    /// Warm attach: adopt slot `slot_idx` of `node`'s arena as resident+ready
    /// for `key` (content preserved across runs, validated by ArenaCache).
    /// Updates the location index. False if the node has no arena, the slot is
    /// occupied, or the key is already resident elsewhere.
    bool adopt_ready(ExpertKey key, int node, size_t slot_idx);

    /// Runtime re-adopt (P-24b, the "free cell still holds it" path): if the
    /// cache knows an ACTIVE, identity-verified record for `key` in a slot the
    /// arena currently has UNOCCUPIED, adopt it (resident+ready, no file
    /// read) and return its pointer. nullptr when no adoptable record exists.
    /// Requires set_cache(); steady-state this hits ~never (eviction reuses
    /// slots immediately) — it guards attach-time races and future eviction
    /// models, and completes the ArenaCache contract.
    void* try_adopt(ExpertKey key);

    /// The backing memfd of `node`'s arena slab (-1 if private / no arena).
    int node_backing_fd(int node) const;

    /// The actual (page-aligned) allocation size of `node`'s slab (0 if none).
    size_t node_backing_bytes(int node) const;

    /// P-24b: the exact slab byte size the ctor would allocate for one node
    /// with these parameters — page-rounded scratch + slots, then page-aligned
    /// as NumaManager mmaps it. Warm-attach validation compares the holder's
    /// stored segment sizes against this BEFORE adopting.
    static size_t slab_bytes(size_t slot_size_bytes, size_t num_slots,
                             size_t extra_scratch_bytes);

    /// J-1: async-load state forwarded to the GPU-local / home arena. Placement
    /// is consumer-aware (INV-MoE-NUMA): callers that know the computing GPU pass
    /// it so the loading slot is the same one resolve()/reserve() returned.
    void mark_loading(ExpertKey key, int gpu = -1);
    void clear_loading(ExpertKey key, int gpu = -1);
    bool is_loading(ExpertKey key, int gpu = -1) const;
    void mark_ready(ExpertKey key, int gpu);

    /// In-flight refcount management (forwarded to the home arena).
    void acquire_inflight(ExpertKey key);
    void release_inflight(ExpertKey key);

    /// Total pinned bytes across all node arenas.
    size_t total_pinned_bytes() const;
    /// Total slot count across all node arenas.
    size_t total_slots() const;
    /// Number of node arenas built.
    size_t num_arenas() const { return arenas_.size(); }

    /// Sorted (ascending) node ids of every built node arena (home + spill).
    /// For placement-policy computation and diagnostics.
    std::vector<int> arena_nodes() const;

    /// Access the arena for a specific NUMA node (nullptr if none). For
    /// diagnostics / tests / P-22a migration substrate.
    PinnedNodeArena* node_arena(int node);
    const PinnedNodeArena* node_arena(int node) const;

    /// Front-of-slab general scratch base / size for `node`, or {nullptr, 0} if
    /// the node has no arena or no scratch was reserved. Forwards to
    /// PinnedNodeArena::node_scratch. First consumer: MultiNumaCpuExpertDevice.
    void*  node_scratch(int node) const;
    size_t node_scratch_bytes(int node) const;

    /// Eagerly load expert WEIGHTS into the arena slots from `src` at startup
    /// (warm-by-definition): for each (layer, expert) in [0,num_layers)×
    /// [0,num_experts), route to its home arena and reserve+load_into+mark_ready
    /// until that node arena is at capacity (experts beyond capacity are skipped —
    /// they stay on the cold path). Returns the number of slots loaded. NOTE:
    /// "preload" here = warming the host-RAM expert cache; unrelated to LLM prefill
    /// (prompt / KV-cache processing). Daemon-thread/init only (not thread-safe).
    /// Pairs with `PrepackedSource::advise_dontneed()` to drop the now-redundant
    /// mmap page cache.
    ///
    /// `loader`: when non-null AND in io_uring mode (`Backend::kIoUring`), the
    /// slot reads are submitted through it (the SAME async path the runtime miss
    /// path uses — "imitate spill") instead of a synchronous per-slot pread. Bulk
    /// preload is the ideal io_uring workload (deep queue saturating the NVMe),
    /// so this is faster than the single-threaded pread for the full-set load.
    /// A worker-pool loader is ignored (preload stays synchronous).
    ///
    /// `placement` (arena host placement policy, arena_placement.h; nullptr =
    /// legacy behavior, byte-for-byte): a key→node map consulted FIRST for
    /// every key — the slot is reserved on the planned node (extend-only,
    /// reserve_on_node). Keys absent from the map, or whose planned node is
    /// full, fall back to the legacy tiered home→spill fill.
    size_t preload(const model::PrepackedSource& src, uint32_t num_layers,
                   uint32_t num_experts, ArenaLoader* loader = nullptr,
                   const std::unordered_map<ExpertKey, int>* placement = nullptr);

private:
    NumaManager& numa_;
    ArenaCache* cache_ = nullptr;  ///< P-24b meta layer (nullable)
    // node → arena. unique_ptr so PinnedNodeArena need not be movable.
    std::unordered_map<int, std::unique_ptr<PinnedNodeArena>> arenas_;

    // Cross-node spill (Stage 1). Authoritative residency: key → node holding it.
    // All facade state-mutators route by this first (a spilled expert lives on
    // neither its home node nor the computing GPU's node).
    std::unordered_map<ExpertKey, int> location_;
    // INV-REEF-BANK published snapshot (see the public section above).
    mutable std::mutex bank_snap_mu_;
    std::shared_ptr<const std::unordered_map<ExpertKey, int>> bank_snap_;
    std::atomic<uint64_t> bank_epoch_{0};
    // Spill candidates sorted DESC by weight (built in ctor; empty when disabled).
    std::vector<std::pair<int, int>> spill_order_;  // (node, weight)
    // Per-node round-robin tick for deterministic equal-weight interleave.
    std::unordered_map<int, uint64_t> rr_counter_;

    /// Route a location_ change to the (optional) trace sink.
    void note_loc(ExpertKey key, int old_node, int new_node, char kind) {
        if (location_change_sink)
            location_change_sink(key, old_node, new_node, kind);
    }

    PinnedNodeArena* located_arena(ExpertKey key);
    const PinnedNodeArena* located_arena(ExpertKey key) const;
    // Spill nodes (weight>0, arena exists) in selection order, excluding `exclude`
    // (the local node). Highest weight first; equal weights interleave by
    // (fewest-occupied, rr_counter, node). Advances rr_counter_.
    std::vector<int> spill_candidates_in_order(int exclude);
};

}  // namespace layerstorm::memory
