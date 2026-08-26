#pragma once

// ExpertSlotSource — the abstract host-side source of packed expert SLOT bytes
// (one MoE layer's worth for one expert, in the prepacked slot layout the
// arena / H2D path consumes).
//
// Two implementations:
//   PrepackedSource   (WP-3): reads slots from the on-disk prepacked set
//                     (expert_NNN.bin + manifest.json) via mmap / pread /
//                     O_DIRECT io_uring descriptors.
//   LiveGgufExpertSource (LIVE PREPACK): synthesizes the SAME slot bytes
//                     directly from the source GGUF shards, running the
//                     prepack transform (pack_gguf_expert) in memory — no
//                     prepacked files on disk. Byte-identical output is an
//                     invariant (INV-LIVE-PREPACK-IDENTITY).
//
// Consumers (host_source, ArenaLoader, ELM, PinnedExpertArena::preload,
// CommandDispatcher) program against this interface; which source exists is
// an engine-init decision (preprocessing.prepacked_dir vs
// preprocessing.live_prepack).
//
// Thread safety: has()/load_into()/read_descriptor()/gguf_types_for_layer()
// must be safe from any thread (ArenaLoader worker pools call load_into off
// the daemon thread; the live bulk build calls it from many workers).

#include <cstdint>
#include <optional>

#include "core/memory/eviction_policy.h"  // ExpertKey
#include "model/weight_pipeline/manifest.h"  // GgufExpertTypes

namespace layerstorm::model {

class ExpertSlotSource {
public:
    virtual ~ExpertSlotSource() = default;

    /// True if this source covers the given expert key.
    virtual bool has(memory::ExpertKey key) const = 0;

    /// Pointer to resident packed slot bytes (mmap mode), or nullptr when the
    /// source has no in-memory representation (direct/live modes).
    virtual const void* resolve(memory::ExpertKey key) const = 0;

    /// Copy/synthesize `key`'s full packed slot bytes into `dst` (a buffer of
    /// at least slot_size_bytes()). Returns false if the key is not covered or
    /// the read fails. Must be callable from any thread.
    virtual bool load_into(memory::ExpertKey key, void* dst) const = 0;

    /// Resolve `key` to a raw contiguous read descriptor (fd, offset, length)
    /// for io_uring reads straight into a pinned slot. Returns false when the
    /// source has no contiguous on-disk slot (live source) or is not in
    /// direct-fd mode — callers then fall back to load_into().
    virtual bool read_descriptor(memory::ExpertKey key, int& fd,
                                 int64_t& offset, int64_t& length) const = 0;

    /// True if this source is mmap-free (loads go through load_into/pread; no
    /// resolve() pointers). The arena is then the only warm tier.
    virtual bool is_direct() const = 0;

    /// True if the memory behind resolve(key) is page-locked.
    virtual bool is_pinned(memory::ExpertKey key) const = 0;

    /// Bytes per packed expert slot — the O_DIRECT-aligned STRIDE (the arena
    /// slot size). The real expert content may be smaller (zero tail).
    virtual int64_t slot_size_bytes() const = 0;

    /// GG-10: the routed-expert k-quant triple THIS layer's slots are packed
    /// at (absolute model layer index). nullopt for non-MoE layers/non-GGUF.
    virtual std::optional<GgufExpertTypes> gguf_types_for_layer(
        int layer_idx) const = 0;
};

}  // namespace layerstorm::model
