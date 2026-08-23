#pragma once

#include <sched.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "config/config_parser.h"

namespace layerstorm::memory {

/// Owned handle for a NUMA-allocated buffer.
struct NumaBuffer {
    void* data = nullptr;
    size_t size = 0;       ///< Page-aligned allocation size.
    int numa_node = -1;    ///< NUMA node, or -1 for interleaved/fallback.
    /// P-24b: backing memfd for MAP_SHARED (persistent-arena) buffers; -1 for
    /// the classic MAP_PRIVATE|MAP_ANONYMOUS allocations. Owned: free() closes
    /// it after the munmap (the fd may live on elsewhere via SCM_RIGHTS dups —
    /// e.g. the arena holder — which is exactly the persistence mechanism).
    int fd = -1;
};

/// A detected CPU-less HBM / device-memory NUMA node (TD-NUMA-HBM-BANKS,
/// INV-NUMA-HOSTBANK). These are valid, bindable host memory (empirically
/// proven: a strict MPOL_BIND fills a full 16 GiB node with no OOM) — the only
/// constraint is CAPACITY: any bound allocation must be sized to the node's
/// free memory, and any thread that should run "local" to the node must be
/// pinned to the node's nearest CPU node (the node itself has no CPUs).
struct HbmNodeInfo {
    int node = -1;               ///< NUMA node id.
    int cpu_affinity_node = -1;  ///< Nearest CPU-ful node: the HMAT access0
                                 ///< initiator when present, else the CPU node
                                 ///< with min numa_distance (ties → smallest id).
    size_t total_bytes = 0;      ///< Node MemTotal at detection time.
};

/// NUMA-aware host-side memory allocator.
///
/// Allocates buffers via mmap + mbind() for NUMA affinity, with transparent
/// fallback to plain mmap when libnuma is unavailable. Designed for expert
/// weight buffers (~18 MB each) that benefit from placement near the GPU's
/// NUMA node to minimise cross-socket PCIe traffic.
class NumaManager {
public:
    /// Construct from hardware config (uses gpus[].numa_node mapping).
    explicit NumaManager(const config::HardwareConfig& hw);
    ~NumaManager();

    NumaManager(const NumaManager&) = delete;
    NumaManager& operator=(const NumaManager&) = delete;

    // ── Topology queries ─────────────────────────────────────────────────

    /// Number of distinct NUMA nodes across all GPUs (0 if no GPUs).
    int num_nodes() const;

    /// NUMA node for a GPU position index (INV-4.18). Returns -1 if out of range.
    int gpu_numa_node(int gpu_position) const;

    /// Number of GPUs this manager was built from (size of the gpu→node map).
    int num_gpus() const { return static_cast<int>(gpu_to_numa_.size()); }

    /// GPU-attached NUMA nodes (sorted, unique). Excludes GPU-less/overflow
    /// nodes. Empty if no GPUs have a known NUMA node.
    const std::vector<int>& gpu_attached_nodes() const { return all_nodes_; }

    /// ALL host-RAM NUMA nodes that have memory (libnuma `numa_node_size64 > 0`)
    /// AND CPUs, sorted — including GPU-less nodes that can still hold spilled
    /// experts. CPU-less memory-only nodes (HBM / device memory) are NOT in this
    /// list: they are valid banks but need capacity-capped allocations + explicit
    /// CPU pinning (INV-NUMA-HOSTBANK) — enumerate them via hbm_nodes(), or use
    /// all_banks_including_hbm() for the full bank set. Falls back to
    /// gpu_attached_nodes() if libnuma is unavailable.
    std::vector<int> all_memory_nodes() const;

    /// Detected CPU-less HBM / device-memory NUMA nodes with their nearest-CPU
    /// affinity mapping (TD-NUMA-HBM-BANKS). Empty when the LAYERSTORM_DETECT_HBM
    /// CMake gate is OFF, libnuma is unavailable, or the box has none. Detection
    /// is pure sysfs + libnuma: a node qualifies iff it has memory, an empty
    /// cpumask, and is in the kernel's has_normal_memory mask (i.e. normal,
    /// bindable RAM — not a non-coherent aperture); the affinity node comes from
    /// the HMAT access0 initiator list, falling back to min numa_distance.
    const std::vector<HbmNodeInfo>& hbm_nodes() const { return hbm_nodes_; }

    /// True iff `node` is a detected HBM / device-memory node (see hbm_nodes()).
    bool node_is_hbm(int node) const;

    /// The nearest-CPU node for a detected HBM node (its measurement/first-touch
    /// thread should be pinned there — the node itself has no CPUs). Returns -1
    /// if `node` is not a detected HBM node.
    int hbm_cpu_affinity_node(int node) const;

    /// The full bank set: all_memory_nodes() ∪ hbm_nodes() ids, sorted unique.
    /// Callers using HBM banks MUST size per-node allocations to the node's free
    /// memory and pin local work via hbm_cpu_affinity_node() (INV-NUMA-HOSTBANK).
    std::vector<int> all_banks_including_hbm() const;

    /// Static home-node assignment for an expert index (P-22 partition):
    /// round-robin over the GPU-attached nodes, balanced equally per NUMA node
    /// (not per GPU count). Returns -1 if no GPU-attached nodes exist.
    int expert_home_node(uint32_t expert_idx) const;

    /// Read a /sys/devices/system/node/nodeN/meminfo field in kB (e.g. "MemFree",
    /// "MemTotal", "Active(file)"). Pass the key WITHOUT the trailing colon.
    /// Returns 0 if unavailable. Matches the full key + ':' (so "Active(file)"
    /// never collides with "Active").
    size_t node_meminfo_kb(int node, const char* key) const;

    /// Free bytes on a NUMA node (meminfo MemFree). Returns 0 if unavailable.
    size_t node_free_bytes(int node) const;

    /// Total bytes on a NUMA node (meminfo MemTotal). Returns 0 if unavailable.
    size_t node_total_bytes(int node) const;

    /// Available bytes on a NUMA node ≈ MemFree + reclaimable page cache
    /// (Active(file) + Inactive(file)) + SReclaimable. The /proc MemAvailable
    /// analogue (per-node meminfo has no MemAvailable line). Returns 0 if
    /// unavailable. Use this — not node_free_bytes — when page cache is large.
    size_t node_available_bytes(int node) const;

    /// True if libnuma is available and functional at runtime.
    bool numa_available() const;

    // ── Allocation ───────────────────────────────────────────────────────

    /// Allocate on a specific NUMA node (MPOL_BIND).
    /// Throws std::bad_alloc on failure.
    NumaBuffer allocate_on_node(size_t size, int numa_node);

    /// Allocate with interleaved policy across all NUMA nodes (MPOL_INTERLEAVE).
    /// Throws std::bad_alloc on failure.
    NumaBuffer allocate_interleaved(size_t size);

    /// Convenience: allocate on the NUMA node of a given GPU.
    /// Falls back to interleaved if gpu_position is invalid (INV-4.18).
    NumaBuffer allocate_for_gpu(size_t size, int gpu_position);

    // ── Shared (memfd-backed) allocation — persistent arena, P-24b ───────

    /// Enable transparent huge pages for the SHARED (memfd) allocations below
    /// (MADV_HUGEPAGE before first touch — 512× fewer PTEs to pin, so
    /// cudaHostRegister over a warm arena drops from minutes-scale to seconds,
    /// and the CPU-expert GEMV gets TLB relief). No-op with a warning when the
    /// kernel's shmem THP mode disallows madvise (shmem_enabled=never/deny) —
    /// the fallback is plain 4K pages, functionally identical. Pages already
    /// faulted small (a pre-THP store) stay small until a cold rebuild.
    void set_shared_thp(bool enabled) { shared_thp_ = enabled; }
    bool shared_thp() const { return shared_thp_; }

    /// True when /sys/kernel/mm/transparent_hugepage/shmem_enabled selects a
    /// mode where MADV_HUGEPAGE is honored for shmem (anything but
    /// [never]/[deny]). False when unreadable (conservative).
    static bool shmem_thp_available();

    /// Allocate a MAP_SHARED memfd-backed buffer bound to `numa_node` (mbind
    /// MPOL_BIND, pages placed at first touch exactly like allocate_on_node).
    /// The returned NumaBuffer owns both the mapping and the fd; the fd can be
    /// SCM_RIGHTS-dup'd to the arena holder so the tmpfs pages (and their NUMA
    /// placement) survive this process. `name` labels the memfd for /proc
    /// diagnostics. Throws std::bad_alloc on failure.
    NumaBuffer allocate_on_node_shared(size_t size, int numa_node,
                                       const char* name = "ls-arena");

    /// Map an EXISTING memfd (received from the arena holder) as MAP_SHARED and
    /// take ownership of `fd` (closed by free()). `size` must be the segment's
    /// exact byte size (page-aligned; validated against fstat). The pages keep
    /// their historical NUMA placement; `numa_node` is recorded and re-bound
    /// (mbind — a no-op for already-placed pages, and it pins the policy for
    /// any future faults). Throws std::bad_alloc / std::runtime_error.
    NumaBuffer adopt_shared(int fd, size_t size, int numa_node);

    // ── Deallocation ─────────────────────────────────────────────────────

    /// Free a buffer. Safe to call on a default-constructed (null) buffer.
    void free(NumaBuffer& buf);

    // ── Migration ────────────────────────────────────────────────────────

    /// Migrate a buffer to a different NUMA node (alloc new, memcpy, free old).
    /// No-op if already on the target node. Throws std::invalid_argument
    /// if buf is null, std::bad_alloc if the new allocation fails.
    void migrate(NumaBuffer& buf, int new_numa_node);

    // ── In-place binding (for externally-owned memory: mmap, malloc) ──────

    /// Bind an existing memory range to a NUMA node (mbind MPOL_BIND with
    /// MPOL_MF_MOVE — moves already-faulted pages). No-op if node < 0 or
    /// libnuma is unavailable. Used to place mmap'd prepacked files and
    /// packed host buffers on an expert's home node before pinning.
    void bind_range(void* ptr, size_t size, int numa_node);

    /// Pin the CALLING thread to run on `numa_node`'s CPUs (numa_run_on_node), so
    /// its first-touch writes fault pages NUMA-local. No-op if node < 0 or libnuma
    /// is unavailable. Used by the parallel arena prefault.
    void pin_current_thread_to_node(int numa_node);

    // ── Statistics ───────────────────────────────────────────────────────

    /// Total bytes currently allocated via this manager.
    size_t total_allocated_bytes() const;

    /// Bytes allocated on a specific NUMA node (-1 for interleaved/fallback).
    size_t allocated_bytes_on_node(int node) const;

private:
    std::vector<int> gpu_to_numa_;       ///< GPU id -> NUMA node
    std::vector<int> all_nodes_;         ///< Sorted unique NUMA nodes
    std::vector<HbmNodeInfo> hbm_nodes_; ///< CPU-less HBM/device-memory nodes (sorted by id)
    bool numa_available_ = false;
    size_t page_size_;

    bool shared_thp_ = false;  ///< MADV_HUGEPAGE on shared arena segments (P-24b)
    std::atomic<size_t> total_allocated_{0};
    // Pre-populated at construction; no insertions during alloc/free.
    std::unordered_map<int, std::atomic<size_t>> per_node_allocated_;

    void detect_hbm_nodes();  ///< Fills hbm_nodes_ (no-op unless LAYERSTORM_HBM_NODES).
    size_t align_to_page(size_t size) const;
    void* raw_mmap(size_t aligned_size);
    void raw_munmap(void* ptr, size_t size);
    void apply_mbind_bind(void* ptr, size_t size, int numa_node);
    void apply_mbind_interleave(void* ptr, size_t size);
};

/// RAII: pin the calling thread to `numa_node`'s CPUs for a scope, restoring the
/// thread's EXACT previous CPU affinity on destruction (sched_getaffinity save →
/// numa_run_on_node → sched_setaffinity restore). No-op when numa_node < 0 or
/// libnuma is unavailable — so wrapping a non-HBM path changes nothing. Used by
/// the loader calibration to measure a CPU-less HBM bank from its nearest CPU
/// node without perturbing the subsequent (unpinned) DDR-bank measurements.
class ScopedThreadNodeBind {
public:
    ScopedThreadNodeBind(NumaManager& numa, int numa_node);
    ~ScopedThreadNodeBind();
    ScopedThreadNodeBind(const ScopedThreadNodeBind&) = delete;
    ScopedThreadNodeBind& operator=(const ScopedThreadNodeBind&) = delete;

private:
    cpu_set_t saved_{};
    bool active_ = false;
};

}  // namespace layerstorm::memory
