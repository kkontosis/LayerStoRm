#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef LAYERSTORM_HAS_URING
#include <liburing.h>
#endif

#include "core/memory/eviction_policy.h"
#include "core/memory/numa_manager.h"

namespace layerstorm::model { class PrepackedSource; }

namespace layerstorm::memory {

// ── Expert tier in the 3-level hierarchy ────────────────────────────────────

/// Location of an expert in the tiered memory hierarchy.
enum class ExpertTier : uint8_t {
    kNone,       ///< Not tracked by NvmeTier (only in VRAM or not loaded).
    kHostRam,    ///< In host RAM (mmap-backed — process-lifetime pointer).
    kNvme,       ///< On NVMe only (file exists but not yet mmapped).
    kInflight,   ///< Write I/O in progress.
};

// ── Async I/O token and completion ──────────────────────────────────────────

/// Opaque token identifying an in-flight I/O operation.
using IoToken = uint64_t;

/// Completion status for an async I/O operation.
struct IoCompletion {
    IoToken token = 0;
    ExpertKey key;
    bool success = false;
    int error_code = 0;
    enum class Op : uint8_t { kRead, kWrite } op = Op::kRead;
};

// ── NVMe tier (mmap-backed, per-expert-index files) ───────────────────────

/// NVMe tier for expert weight storage.
///
/// Manages the lower two tiers of the expert memory hierarchy:
///   Host RAM (mmap) --> NVMe (cold storage)
/// GPU VRAM is managed by ExpertCache (separate).
///
/// File layout (WP-5): one file per expert_idx, containing all MoE layers:
///   <drive>/layerstorm/experts/expert_042.bin
///   Slot at byte offset: (layer_idx - first_moe_layer) * slot_size_bytes
///
/// Reads are mmap-based: host_ptr() returns a pointer into a process-lifetime
/// mmap region (INV-WP-7).  This eliminates the buffer lifetime race (TD-82n)
/// because mmap pointers are valid for the entire process lifetime — no
/// refcounting or pinning needed during H2D DMA.
///
/// Writes (spill path) use io_uring pwrite at the correct slot offset.
/// The mmap (MAP_SHARED) sees pwritten data immediately.
///
/// Bidirectional: NVMe-as-source (prepacked files live on NVMe drives, shared
/// mmap with PrepackedSource via attach_prepacked_source()) or NVMe-as-spill
/// (RAM→NVMe eviction writes slot data via pwrite).
///
/// Expert-to-drive assignment: expert_idx % num_drives (INV-4.12c).
///
/// Owned and exclusively called by the single-threaded daemon (INV-3.4.2).
/// Not thread-safe.
///
/// Build-time: requires LAYERSTORM_HAS_URING for async writes.
/// Without it, writes use synchronous pwrite fallback.
class NvmeTier {
public:
    struct Options {
        std::vector<std::string> drive_paths;   ///< NVMe mount points.
        int queue_depth = 64;                   ///< Per-drive io_uring queue depth.
        int64_t slot_size_bytes = 0;            ///< Size of one expert slot (one layer's data).
        int64_t host_ram_budget_bytes = 0;      ///< Advisory budget (for prefetch pacing).
        int num_moe_layers = 0;                 ///< Number of MoE layers in the model.
        int num_experts_per_layer = 0;          ///< Number of routed experts per layer.
        int first_moe_layer = 0;                ///< Absolute layer index of the first MoE layer.
    };

    /// Construct the NVMe tier.
    /// Throws std::invalid_argument if drive_paths is empty or slot_size_bytes <= 0.
    /// Scans existing expert files on all drives and mmaps them.
    NvmeTier(Options opts, NumaManager& numa);
    ~NvmeTier();

    NvmeTier(const NvmeTier&) = delete;
    NvmeTier& operator=(const NvmeTier&) = delete;

    // ── Startup / Shutdown ───────────────────────────────────────────────

    /// Pre-create directory structure on all drives (flat: <drive>/layerstorm/experts/).
    void init_directories();

    /// Drain all in-flight I/O (blocking). Called during shutdown only.
    void drain();

    /// Borrow mmap regions from a PrepackedSource whose files live on our
    /// NVMe drives.  For each expert file on a matching drive, the mmap is
    /// shared (not re-opened) and all slots are marked as written.
    /// Regions borrowed this way are not munmapped by NvmeTier's destructor.
    void attach_prepacked_source(const model::PrepackedSource& source);

    // ── Write: Host RAM → NVMe (spill) ──────────────────────────────────

    /// Asynchronously write an expert slot from host_ptr to NVMe.
    /// Writes at the correct offset within the per-expert-index file.
    /// The host_ptr must remain valid until the completion is polled.
    /// Returns nullopt if ring is full, expert is already inflight,
    /// or io_uring is unavailable.
    std::optional<IoToken> write_expert(ExpertKey key, const void* host_ptr);

    /// Owning overload: takes ownership of buf unconditionally.
    /// On async success, buf is moved into PendingOp (freed on CQE).
    /// On async failure, falls back to sync write and frees buf.
    /// Returns IoToken on async success, sentinel IoToken{0} on sync
    /// fallback, or nullopt only if the expert is already inflight.
    std::optional<IoToken> write_expert(ExpertKey key, NumaBuffer buf);

    // ── Read: NVMe → Host RAM (mmap + madvise) ─────────────────────────

    /// Make an expert's host pointer available.  With mmap, this is
    /// essentially a prefetch hint: issues madvise(MADV_WILLNEED) on the
    /// slot's pages.
    /// Returns IoToken{0} (immediate completion) if the slot is available,
    /// or nullopt if the expert has no data on NVMe.
    std::optional<IoToken> read_expert(ExpertKey key, int gpu_hint = -1);

    // ── Completion polling (non-blocking) ────────────────────────────────

    /// Poll all io_uring rings for write completions. Non-blocking.
    /// Reads are synchronous (mmap), so only writes produce completions.
    std::vector<IoCompletion> poll_completions();

    // ── Host pointer access (mmap-backed, process-lifetime) ─────────────

    /// Get the host RAM pointer for an expert slot.
    /// Returns a pointer into a process-lifetime mmap region, or nullptr
    /// if the slot has no data.  The returned pointer is valid for the
    /// entire process lifetime (INV-WP-7: fixes TD-82n).
    const void* host_ptr(ExpertKey key) const;

    // ── Tier queries ────────────────────────────────────────────────────

    ExpertTier tier(ExpertKey key) const;
    bool is_on_nvme(ExpertKey key) const;
    bool is_in_host_ram(ExpertKey key) const;
    bool is_inflight(ExpertKey key) const;

    // ── NUMA queries ──────────────────────────────────────────────────

    /// NUMA node of an mmap-backed entry.  Returns -1 (OS-managed pages).
    int host_numa_node(ExpertKey key) const;

    // ── Latency estimation (INV-4.12) ───────────────────────────────────

    /// Estimated microseconds for NVMe read of one expert slot.
    double estimate_read_us() const;

    /// Estimated microseconds for host-RAM-to-VRAM transfer.
    static double estimate_host_to_vram_us(int64_t slot_size_bytes,
                                           double pcie_bw_gbps);

    // ── Capacity queries ────────────────────────────────────────────────

    int num_drives() const;
    int64_t slot_size_bytes() const;
    int64_t total_nvme_capacity_bytes() const;
    int64_t total_nvme_used_bytes() const;

    int host_ram_entry_count() const;
    int64_t host_ram_used_bytes() const;
    int64_t host_ram_budget_bytes() const;

    int inflight_count() const;

    // ── Cancellation ────────────────────────────────────────────────────

    /// Cancel an in-flight write I/O by token.  Returns true if found and
    /// cancelled, false if the token is unknown (already completed or
    /// never issued).
    bool cancel(IoToken token);

    // ── Drive assignment (exposed for testing) ──────────────────────────

    int drive_for_expert(ExpertKey key) const;
    std::string expert_path(ExpertKey key) const;

private:
    // ── Internal types ──────────────────────────────────────────────────

    /// Per-expert-index mmap region.  One file per expert_idx, containing
    /// all MoE layers' slot data contiguously.
    struct MmapRegion {
        void* base = nullptr;     ///< mmap base pointer (or nullptr if not mapped).
        size_t size = 0;          ///< mmap size (= num_moe_layers * slot_size_bytes).
        int fd = -1;              ///< Open fd for pwrite (spill path). -1 if borrowed.
        bool borrowed = false;    ///< If true, borrowed from PrepackedSource — don't munmap.
    };

    struct PendingOp {
        IoToken token = 0;
        ExpertKey key;
        int drive_idx = -1;
        NumaBuffer write_buf;   ///< Owned source data (writes only).
        int fd = -1;            ///< Open file descriptor (per-op, for pwrite).
    };

#ifdef LAYERSTORM_HAS_URING
    struct DriveState {
        std::string path;
        struct io_uring ring{};
        bool ring_initialized = false;
        int num_inflight = 0;
    };
    std::vector<DriveState> drives_;
#else
    struct DriveState {
        std::string path;
    };
    std::vector<DriveState> drives_;
#endif

    // ── Data ────────────────────────────────────────────────────────────

    Options opts_;
    NumaManager& numa_;

    /// Per-expert-index mmap regions.  Indexed by expert_idx.
    std::vector<MmapRegion> mmap_regions_;

    /// Per-expert, per-layer slot written status.  [expert_idx][layer_pos].
    /// layer_pos = layer_idx - first_moe_layer.
    std::vector<std::vector<bool>> slot_written_;

    /// Total NVMe bytes written (for capacity queries).
    int64_t nvme_used_bytes_ = 0;

    // In-flight tracking (writes only).
    IoToken next_token_ = 1;
    std::unordered_map<IoToken, PendingOp> pending_;
    std::unordered_map<ExpertKey, IoToken> inflight_keys_;

    // Deferred cleanup for cancelled writes: buffer freed when CQE arrives.
    struct CancelledWrite {
        NumaBuffer buf;
        int drive_idx = -1;
    };
    std::unordered_map<IoToken, CancelledWrite> cancelled_writes_;

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Compute layer position within an expert file from absolute layer_idx.
    /// Returns -1 if layer_idx is out of MoE range.
    int layer_pos(ExpertKey key) const;

    /// Ensure the expert file is created and mmapped.  No-op if already mapped.
    void ensure_mmap(int expert_idx);

    /// Scan existing expert files on all drives and mmap them.
    void scan_and_mmap_existing_files();

    /// Build filesystem path for an expert file on its assigned drive.
    std::string expert_path_for(int expert_idx) const;

    /// Synchronous pwrite fallback for write_expert().
    void write_expert_sync(ExpertKey key, const void* data);
};

}  // namespace layerstorm::memory
