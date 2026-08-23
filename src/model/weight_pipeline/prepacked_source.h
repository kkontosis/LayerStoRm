#pragma once

// PrepackedSource: mmap-based reader for pre-processed expert weight files
// produced by WP-2 (expert_prepacker). Provides O(1) host-source resolution
// for ExpertLifecycleManager and CommandDispatcher, eliminating runtime
// packing overhead.
//
// Thread safety: resolve() and has() are const and safe to call from any
// thread. register_for_pinned_dma() / unregister_pinned_dma() are init/
// shutdown only (not concurrent with resolve).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include "core/memory/eviction_policy.h"  // ExpertKey
#include "model/weight_pipeline/manifest.h"

namespace layerstorm::model {

class QuantInterface;

class PrepackedSource {
public:
    /// Construct from a prepacked directory containing manifest.json +
    /// expert_NNN.bin files. Validates manifest against the QuantInterface.
    /// Throws std::runtime_error on any error (missing files, size mismatch,
    /// manifest validation failure).
    /// `direct_io` (Stage 2, TD-J-1-f): do NOT mmap the prepacked files; keep an
    /// open fd per file and read slots via pread (so the page cache never holds a
    /// second copy of the arena's data). `o_direct` additionally opens with
    /// O_DIRECT to bypass the cache entirely (requires 4096-aligned slot size; if
    /// the slot size is not aligned it falls back to buffered pread + fadvise).
    PrepackedSource(const std::filesystem::path& prepacked_dir,
                    const QuantInterface& quant,
                    bool direct_io = false, bool o_direct = false);
    ~PrepackedSource();

    PrepackedSource(const PrepackedSource&) = delete;
    PrepackedSource& operator=(const PrepackedSource&) = delete;
    PrepackedSource(PrepackedSource&&) = delete;
    PrepackedSource& operator=(PrepackedSource&&) = delete;

    /// Returns pointer to the packed expert slot data for a given key,
    /// or nullptr if the key is out of range or layer_idx is not a MoE layer.
    const void* resolve(memory::ExpertKey key) const;

    /// Returns true if this source covers the given expert key.
    bool has(memory::ExpertKey key) const;

    /// Bytes per packed expert slot (one MoE layer's worth for one expert).
    int64_t slot_size_bytes() const { return slot_size_bytes_; }

    /// P-24 cold loader: copy this expert's packed slot bytes from the mmap'd
    /// prepacked file into `dst` (a pre-pinned arena slot of at least
    /// slot_size_bytes()). Returns true on success, false if the key is not
    /// covered. This is the SSD→pinned-arena step of the 3-tier model: the
    /// mmap stays the cold backing store; the arena slot becomes the DMA source.
    bool load_into(memory::ExpertKey key, void* dst) const;

    /// Register all mmap regions as pinned memory for faster DMA.
    /// Requires CUDA runtime to be initialized. Returns the number of
    /// regions successfully registered, or 0 if CUDA is unavailable.
    /// Logs warnings on per-region failure but does not throw.
    int register_for_pinned_dma();

    /// Stage 2: read `key`'s packed slot bytes directly from the file into `dst`
    /// via pread (no mmap). Buffered mode posix_fadvise(DONTNEED)s the range after
    /// the read so it doesn't linger in the page cache; O_DIRECT mode bypasses the
    /// cache. Returns false if the key is not covered or the read is short. Only
    /// valid in direct_io mode (fd kept open).
    bool pread_into(memory::ExpertKey key, void* dst) const;

    /// True if this source is in direct-I/O (mmap-free) mode (Stage 2). In that
    /// mode resolve()/host_ptr() return null and loads go through pread.
    bool is_direct() const { return direct_io_; }

    /// Stage 3 (io_uring): resolve `key` to a raw read descriptor — the open file
    /// descriptor, byte offset, and length (the on-disk stride) — so a caller can
    /// issue an io_uring_prep_read straight into a pinned slot. Returns false if
    /// not in direct mode (no kept fd) or the key is not covered.
    bool read_descriptor(memory::ExpertKey key, int& fd, int64_t& offset,
                         int64_t& length) const;

    /// True if the region backing `key`'s expert is page-locked (so the H2D
    /// source needs no staging copy).
    bool is_pinned(memory::ExpertKey key) const;

    /// Unregister all pinned regions. Called automatically by destructor.
    void unregister_pinned_dma();

    /// MADV_DONTNEED every (non-pinned) mmap region — drops the prepacked file
    /// page cache. Use after a full arena preload so the redundant cache can be
    /// reclaimed (it would otherwise compete with the pinned arena for RAM).
    /// Pages re-fault from the file on any later access. Skips pinned regions
    /// (page-locked → DONTNEED would be undefined). No-op on null regions.
    void advise_dontneed() const;

    /// GG-10: the routed-expert k-quant triple THIS layer's slots are packed
    /// at. `layer_idx` is the ABSOLUTE model layer index (same indexing as
    /// ExpertKey.layer_idx / moe_layers.indices). Returns nullopt if layer_idx
    /// is not a MoE layer or the manifest has no per-layer block (non-GGUF).
    /// The global manifest gguf_types triple remains the per-projection MAX
    /// used for slot/cache sizing.
    std::optional<GgufExpertTypes> gguf_types_for_layer(int layer_idx) const;

    /// Access the manifest (for logging/diagnostics).
    const Manifest& manifest() const { return manifest_; }

    /// Number of expert files mmapped.
    int num_expert_files() const { return static_cast<int>(mmaps_.size()); }

    /// Mmap region descriptor (for engine-level pinned DMA registration).
    /// In direct_io mode `base` is null and `fd` is the open file descriptor
    /// used by pread_into (mmap-free).
    struct MmapRegion {
        void*  base = nullptr;
        size_t size = 0;
        bool   pinned = false;
        int    numa_node = -1;   ///< home node when NUMA-bound + pinned
        int    fd = -1;          ///< open fd (direct_io mode); -1 otherwise
    };

    /// Access mmap regions (for external pinned DMA registration).
    std::span<const MmapRegion> mmap_regions() const { return mmaps_; }

private:
    Manifest manifest_;
    std::vector<MmapRegion> mmaps_;  // indexed by expert_idx
    int64_t slot_size_bytes_ = 0;
    bool direct_io_ = false;   ///< Stage 2: pread instead of mmap.
    bool o_direct_ = false;    ///< direct_io_ via O_DIRECT (else buffered+fadvise).
};

}  // namespace layerstorm::model
