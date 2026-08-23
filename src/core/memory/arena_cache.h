#pragma once

// ArenaCache (P-24b) — the persistence meta layer below the arena loaders.
//
// A tiny (~40 B/slot) slot-indexed record table living in its OWN shared memfd
// segment (kept alive by the arena holder alongside the node arena segments).
// Each record describes what a specific arena slot holds: expert identity
// (layer, expert) + the source file's identity (mtime_ns, size) + a 64-bit
// key hash. Because the record table survives engine restarts with the arena
// RAM, a warm attach can ADOPT still-valid slots (skip the NVMe preload), and
// a runtime miss can re-adopt a free slot whose bytes still match.
//
// INV-ARENA-CACHE-ORDER (crash safety at any kill point):
//   - a slot's record is invalidated (kEmpty) BEFORE the first byte of a
//     reuse/fresh fill is written (hooked at PinnedNodeArena::reserve);
//   - kLoading is stamped when an async fill is submitted (mark_loading);
//   - kActive (with the full identity tuple) is stamped only AFTER the load
//     fully completed (mark_ready).
//   Any interruption leaves the slot either valid-and-described or
//   invalid-and-reloadable — never a stale key over clobbered bytes.
//
// The key hash is an INDEX ONLY (fast in-memory map); every hit verifies the
// full stored tuple, so hash collisions cost a miss, never corruption.
//
// Thread safety: daemon-thread only (INV-ELM-1), same as the arena slot state
// it mirrors. The meta segment is written with plain stores + a release fence
// before state transitions to kActive (single-writer process).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "config/config_parser.h"         // sizing/spill config (hash_config)
#include "core/memory/eviction_policy.h"  // ExpertKey

namespace layerstorm::memory {

/// Identity of one prepacked expert file (expert_NNN.bin). Captured once at
/// startup by stat(); a re-packed/re-quantized file changes mtime/size and
/// invalidates every cached slot sourced from it.
struct ExpertFileIdentity {
    uint64_t mtime_ns = 0;
    uint64_t size_bytes = 0;
    bool valid() const { return size_bytes != 0; }
};

/// Geometry of one node arena, as seen by the cache (must match the arena's
/// actual slab layout — hashed into the header).
struct ArenaCacheNodeGeom {
    int32_t  node = -1;
    uint64_t num_slots = 0;
};

/// RAII owner of the meta memfd + its MAP_SHARED mapping. Created fresh on a
/// cold boot (fd then STOREd with the holder) or adopted from the holder on a
/// warm attach. The meta table is tiny (~40 B/slot ≈ ~1 MB) and needs no NUMA
/// binding. Move-only; the dtor munmaps + closes the fd.
class ArenaMetaSegment {
public:
    /// Fresh zero-filled segment of `bytes` (memfd_create + ftruncate + mmap).
    /// Returns an invalid segment on failure (logged).
    static ArenaMetaSegment create(size_t bytes);

    /// Map an existing memfd received from the holder (size from fstat).
    /// Takes ownership of `fd` (closed even on failure).
    static ArenaMetaSegment adopt(int fd);

    ArenaMetaSegment() = default;
    ~ArenaMetaSegment();
    ArenaMetaSegment(ArenaMetaSegment&& o) noexcept;
    ArenaMetaSegment& operator=(ArenaMetaSegment&& o) noexcept;
    ArenaMetaSegment(const ArenaMetaSegment&) = delete;
    ArenaMetaSegment& operator=(const ArenaMetaSegment&) = delete;

    bool   valid() const { return base_ != nullptr; }
    void*  base() const { return base_; }
    size_t bytes() const { return bytes_; }
    int    fd() const { return fd_; }

    /// Grow the segment to at least `min_bytes` (page-aligned) via ftruncate
    /// + remap (TD-ARENA-MIGRATE-EMA-PERSIST: append the EMA trailer to a
    /// pre-trailer store's meta segment in place — memfd growth persists with
    /// the holder's fd, and the holder never maps the segment
    /// (INV-ARENA-HOLD), so no other mapping needs the new size). The new
    /// mapping is established BEFORE the old one is released; on any failure
    /// the existing mapping stays valid and false is returned (EMA
    /// persistence simply stays disabled). New bytes read as zero (=
    /// trailer state invalid = start cold). No-op true when already big
    /// enough.
    bool ensure_bytes(size_t min_bytes);

private:
    void*  base_ = nullptr;
    size_t bytes_ = 0;
    int    fd_ = -1;
};

class ArenaCache {
public:
    /// v2: geometry_hash covers CONFIG-STABLE inputs only (slot size, scratch,
    /// node set + share degrees, sizing/spill config) — NOT resolved slot
    /// counts, which depend on volatile free RAM (fraction_free HBM banks
    /// drifted ±1 slot run-to-run and wiped a valid 504 GB store). On warm
    /// attach the STORED per-node slot counts are authoritative (adoption
    /// allocates nothing, so the stored geometry always fits).
    static constexpr uint32_t kSchemaVersion = 2;

    enum RecordState : uint32_t { kEmpty = 0, kLoading = 1, kActive = 2 };

    /// On-disk (on-shm) per-slot record. 40 bytes, plain-old-data.
    struct SlotRecord {
        uint32_t state = kEmpty;      ///< RecordState
        uint32_t layer_idx = 0;
        uint16_t expert_idx = 0;
        uint16_t pad0 = 0;
        uint32_t pad1 = 0;
        uint64_t file_mtime_ns = 0;
        uint64_t file_size = 0;
        uint64_t key_hash = 0;
    };
    static_assert(sizeof(SlotRecord) == 40);

    /// Meta segment header. `geometry_hash` covers slot_size + scratch bytes +
    /// the per-node (node, num_slots) list; `source_id` covers the prepacked
    /// dir identity + manifest format + stride + this schema version. Any
    /// mismatch on attach ⇒ the stored arena is WIPED (cold rebuild) — old
    /// cells are never remapped onto new geometry.
    struct Header {
        uint64_t magic = kMagicExpected;
        uint32_t schema_version = kSchemaVersion;
        uint32_t num_nodes = 0;
        uint64_t geometry_hash = 0;
        uint64_t source_id = 0;
        uint64_t record_count = 0;
    };
    struct NodeSpan {
        int32_t  node = -1;
        uint32_t pad = 0;
        uint64_t first_record = 0;
        uint64_t num_slots = 0;
    };
    static constexpr uint64_t kMagicExpected = 0x314E455241534C1BULL;  // "\x1bLSARENA1"

    // ── Construction ─────────────────────────────────────────────────────

    /// Bytes the meta segment needs for this geometry.
    static size_t required_bytes(const std::vector<ArenaCacheNodeGeom>& geom);

    /// FNV-1a-64 over the geometry parameters (deterministic across runs).
    static uint64_t hash_geometry(size_t slot_size_bytes, size_t scratch_bytes,
                                  const std::vector<ArenaCacheNodeGeom>& geom);

    /// One (node, share_degree) pair for hash_config — derived from topology
    /// + config only (NO RAM-dependent budgets).
    struct NodeIdentity {
        int32_t node = -1;
        int32_t share_degree = 1;
    };

    /// FNV-1a-64 over the CONFIG-STABLE arena identity: slot size, scratch,
    /// the (sorted) node set with share degrees, and the sizing + spill config
    /// knobs (mode/value/per-node overrides, spill nodes/weights). Resolved
    /// slot counts are deliberately EXCLUDED — they derive from volatile free
    /// RAM; the stored spans carry them instead (see open_stored).
    /// `placement_id` (arena host placement policy,
    /// ArenaPlacementPolicy::identity): non-zero when a frequency-aware
    /// slot→node placement is active — it folds into the identity so a store
    /// built under a DIFFERENT placement (or none) is wiped, never
    /// mis-adopted. 0 (policy off) is NOT folded, so every pre-placement
    /// store keeps its exact historical hash (no spurious one-time wipe).
    static uint64_t hash_config(size_t slot_size_bytes, size_t scratch_bytes,
                                const std::vector<NodeIdentity>& nodes,
                                const config::PinHostExpertPoolSizingConfig& sizing,
                                const config::CrossNodeSpillConfig& spill,
                                uint64_t placement_id = 0);

    /// Warm attach, step 1: validate the header (magic, schema, geometry/
    /// source hashes) and the spans' SELF-consistency (record count, bounds
    /// within the segment), and build the span table — WITHOUT comparing
    /// against a recomputed plan. The stored geometry is then read back via
    /// stored_geometry() and adopted as authoritative. False ⇒ wipe.
    bool open_stored(uint64_t geometry_hash, uint64_t source_id);

    /// The per-node geometry recorded in the meta segment (valid after
    /// open_stored / format / validate).
    std::vector<ArenaCacheNodeGeom> stored_geometry() const;

    /// FNV-1a-64 source identity: prepacked dir (absolute path string), the
    /// manifest format_version, the on-disk slot stride, and kSchemaVersion
    /// (so any future in-slot transform bump invalidates old content).
    static uint64_t hash_source_id(const std::string& prepacked_dir_abs,
                                   const std::string& manifest_format_version,
                                   int64_t slot_stride_bytes);

    /// Wrap a mapped meta segment (`base`,`bytes`). Does NOT initialize —
    /// call format() for a fresh segment or validate() for an adopted one.
    ArenaCache(void* base, size_t bytes);

    ArenaCache(const ArenaCache&) = delete;
    ArenaCache& operator=(const ArenaCache&) = delete;

    /// Initialize a fresh meta segment: write header + node spans, zero all
    /// records. Returns false if `bytes` is too small for the geometry.
    bool format(uint64_t geometry_hash, uint64_t source_id,
                const std::vector<ArenaCacheNodeGeom>& geom);

    /// Validate an adopted meta segment against the CURRENT geometry/source.
    /// False ⇒ wipe (caller cold-rebuilds + format()s a new segment). Also
    /// (re)builds the in-memory node-span table on success.
    bool validate(uint64_t geometry_hash, uint64_t source_id,
                  const std::vector<ArenaCacheNodeGeom>& geom);

    /// Install the per-expert-file identity table (index = expert_idx).
    /// Records whose stored identity mismatches are treated as invalid.
    void set_file_identities(std::vector<ExpertFileIdentity> idents);

    /// Stat every expert_NNN.bin under `prepacked_dir` → identity table
    /// (index = expert_idx). Missing files yield invalid identities.
    static std::vector<ExpertFileIdentity> stat_expert_files(
        const std::string& prepacked_dir, uint32_t num_experts);

    // ── Record ops (INV-ARENA-CACHE-ORDER) ───────────────────────────────

    /// A slot is about to be (re)used for new content: kill its record BEFORE
    /// any byte is written. Also drops the in-memory index entry.
    void on_reuse(int node, size_t slot);

    /// An async fill was submitted for (node, slot) ← key: stamp kLoading with
    /// the full identity (not adoptable until kActive).
    void on_load_start(int node, size_t slot, ExpertKey key);

    /// The fill for (node, slot) ← key completed (bytes are final): stamp the
    /// full record kActive (release-fenced) and index it.
    void on_load_ready(int node, size_t slot, ExpertKey key);

    // ── Queries ──────────────────────────────────────────────────────────

    struct Hit {
        int node = -1;
        size_t slot = 0;
    };

    /// Find a currently-ACTIVE record for `key` whose stored identity matches
    /// the CURRENT file identity. In-memory index lookup + full-tuple verify.
    std::optional<Hit> lookup(ExpertKey key) const;

    /// Attach-time adoption scan: invoke `fn(node, slot, key)` for every
    /// ACTIVE record whose identity matches the current file identity table
    /// and whose (layer, expert) is within (num_layers, num_experts). Also
    /// builds the in-memory index for those records; all other records are
    /// reset to kEmpty (stale identity, out-of-range, or interrupted loads).
    /// Returns the number of adoptable records.
    size_t scan_adoptable(uint32_t num_layers, uint32_t num_experts,
                          const std::function<void(int, size_t, ExpertKey)>& fn);

    /// 64-bit key hash (index only — never trusted without tuple verify).
    uint64_t key_hash(ExpertKey key) const;

    size_t active_records() const { return index_.size(); }
    const Header* header() const { return header_; }

    // ── EMA persistence trailer (TD-ARENA-MIGRATE-EMA-PERSIST) ───────────
    //
    // Optional region after the slot-record table: the ArenaMigrator's
    // per-(layer,expert) demand-fetch EMA (~78 KiB for GLM-5.2), persisted
    // across boots with the holder store so a fresh engine does not re-learn
    // placement from zero. Presence is detected by SEGMENT SIZE plus the
    // trailer's own magic/version — a store whose meta segment predates the
    // trailer simply has no room (ema_load → nullopt, ema_save → false;
    // NEVER a wipe: the record table and its schema are untouched). Crash
    // safety follows the slot-record discipline (INV-ARENA-CACHE-ORDER):
    // `state` is killed BEFORE the first payload byte of a save and re-armed
    // only AFTER the full payload landed (release-fenced) — a kill at any
    // point leaves the trailer either valid-and-consistent or
    // invalid-and-cold, never a stale-shape blob under a valid stamp.
    static constexpr uint64_t kEmaMagic = 0x414D45414E455241ULL;  // "ARENAEMA"
    static constexpr uint32_t kEmaVersion = 1;

    struct EmaTrailer {
        uint64_t magic = 0;
        uint32_t version = 0;
        uint32_t state = 0;          ///< 0 = invalid, 1 = valid
        uint32_t num_layers = 0;
        uint32_t num_experts = 0;
        uint64_t saved_unix_ns = 0;  ///< CLOCK_REALTIME at save (age info)
        double   fetches_seen = 0.0; ///< un-decayed total (warm-up gate)
        uint64_t reserved = 0;
        // float data[num_layers * num_experts] follows (64-B-aligned start).
    };
    static_assert(sizeof(EmaTrailer) == 48);

    /// Bytes the trailer region needs (header + dense float table).
    static size_t ema_trailer_bytes(uint32_t num_layers, uint32_t num_experts);

    /// Segment bytes for geometry + EMA trailer (cold-create / grow sizing).
    static size_t required_bytes_with_ema(
        const std::vector<ArenaCacheNodeGeom>& geom, uint32_t num_layers,
        uint32_t num_experts);

    struct EmaView {
        const float* data = nullptr;  ///< points INTO the meta segment
        uint32_t num_layers = 0;
        uint32_t num_experts = 0;
        double   fetches_seen = 0.0;
        uint64_t saved_unix_ns = 0;
    };

    /// The persisted EMA, iff present + valid + EXACTLY (num_layers,
    /// num_experts)-shaped. Any absence/mismatch → nullopt (start cold —
    /// never wrong). Layout must be open (format/open_stored/validate).
    std::optional<EmaView> ema_load(uint32_t num_layers,
                                    uint32_t num_experts) const;

    /// Persist `data` (num_layers*num_experts floats) into the trailer.
    /// False when the segment has no trailer room (pre-trailer store) or no
    /// layout is open. Single-writer (daemon thread / post-join engine).
    bool ema_save(const float* data, uint32_t num_layers,
                  uint32_t num_experts, double fetches_seen);

private:
    SlotRecord* record(int node, size_t slot);
    const SlotRecord* record(int node, size_t slot) const;
    /// Trailer pointer iff the open layout leaves >= `payload_floats`-shaped
    /// trailer room in the segment; nullptr otherwise.
    EmaTrailer* ema_trailer(uint32_t num_layers, uint32_t num_experts) const;
    bool identity_matches(const SlotRecord& r) const;
    void index_erase(ExpertKey key, int node, size_t slot);

    void*  base_ = nullptr;
    size_t bytes_ = 0;
    Header* header_ = nullptr;
    NodeSpan* spans_ = nullptr;      // [header_->num_nodes]
    SlotRecord* records_ = nullptr;  // [header_->record_count]
    std::unordered_map<int, NodeSpan*> span_by_node_;
    std::vector<ExpertFileIdentity> file_ident_;  // index = expert_idx
    // key (layer<<16|expert) → flat record index. ACTIVE records only.
    std::unordered_map<uint64_t, uint64_t> index_;
};

}  // namespace layerstorm::memory
