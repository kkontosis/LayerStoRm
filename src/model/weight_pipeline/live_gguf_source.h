#pragma once

// LiveGgufExpertSource (LIVE PREPACK) — builds packed expert SLOT bytes
// DIRECTLY from the source GGUF shards, running the prepack transform
// (pack_gguf_expert — the exact code prepack_experts/WP-2 uses before writing
// expert_NNN.bin) in memory. No prepacked files on disk are read or written.
//
// Motivation: an arena cold build was bounded by the PREPACKED set's disk
// (46 min from a USB HDD) while the source GGUF sits on NVMe at ~3.3 GB/s.
// Live prepack makes the cold build source-disk-bound: read the stacked
// routed-expert tensor ranges with O_DIRECT (io_uring when built with
// liburing), transform per (layer, expert) on a NUMA-aware worker pool, and
// commit the bytes straight into the pinned arena slots.
//
// INV-LIVE-PREPACK-IDENTITY: for every covered key, the slot bytes produced
// here (content + zero tail up to the aligned stride) are BYTE-IDENTICAL to
// the slot a PrepackedSource would serve from a prepack_experts output of the
// same GGUF. Guaranteed by construction: the per-expert source ranges are the
// same de-stacked spans (destack_expert_tensor geometry), the transform is
// pack_gguf_expert with the same strict per-layer type validation (GG-10),
// and the tail is zero-padded to prepacked::aligned_slot_stride like the
// on-disk writer.
//
// Scope: GGUF sources only (covers all GGUF k-quants — Q4_K/Q5_K/Q6_K/Q8_0/
// MXFP4/... — via the verbatim block-copy pack). Safetensors NVFP4/FP8 keep
// the offline prepack path (their transform needs the full safetensors aux
// tensors; see prepack_experts).
//
// Thread safety: has()/gguf_types_for_layer()/slot_size_bytes() are
// const-immutable after construction. load_into() is callable from ANY thread
// (per-thread scratch); the bulk build in live_arena_build.h drives it from
// many workers. No CUDA anywhere here (INV-GPU-1).

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

#include "core/memory/eviction_policy.h"  // ExpertKey
#include "model/weight_loader/weight_loader.h"  // GgufModelExpertTypes
#include "model/weight_pipeline/expert_slot_source.h"

namespace layerstorm::model {

class GgufReader;
class ModelConfig;
class QuantInterface;

class LiveGgufExpertSource final : public ExpertSlotSource {
public:
    /// One projection of one layer's STACKED routed-expert tensor: where the
    /// per-expert block ranges live on disk. Expert `e`'s bytes are the
    /// contiguous range [abs_offset + e*per_expert_bytes, +per_expert_bytes)
    /// of shard `shard_idx` (destack_expert_tensor geometry).
    struct ProjSource {
        int      shard_idx = -1;
        int64_t  abs_offset = 0;        ///< absolute file offset of expert 0
        int64_t  per_expert_bytes = 0;  ///< packed k-quant bytes per expert
        GgufKQuantType kquant{};        ///< this layer's type for the proj
    };

    /// `shards` are BORROWED (typically LoadedModel::gguf_shards — their
    /// header metadata is used; tensor DATA pages are never touched through
    /// the mmap) and must outlive this object. `quant` is the global
    /// per-projection MAX interface that sizes the slot (GG-9). Throws
    /// std::runtime_error when any MoE layer lacks a complete stacked
    /// gate/up/down k-quant triple, an expert count mismatches, or a layer's
    /// packed total exceeds the slot size (same checks as prepack_experts).
    /// `o_direct`: open the shard data fds O_DIRECT (per-file fallback to
    /// buffered pread + fadvise when the filesystem refuses O_DIRECT).
    LiveGgufExpertSource(const std::vector<GgufReader>& shards,
                         const ModelConfig& model_cfg,
                         const QuantInterface& quant,
                         int n_routed_experts,
                         bool o_direct);
    ~LiveGgufExpertSource() override;

    LiveGgufExpertSource(const LiveGgufExpertSource&) = delete;
    LiveGgufExpertSource& operator=(const LiveGgufExpertSource&) = delete;

    // ── ExpertSlotSource ────────────────────────────────────────────────
    bool has(memory::ExpertKey key) const override;
    const void* resolve(memory::ExpertKey) const override { return nullptr; }
    /// Read + transform + zero-pad `key`'s slot into `dst` (>= stride bytes).
    /// Any-thread safe (per-thread scratch buffers).
    bool load_into(memory::ExpertKey key, void* dst) const override;
    bool read_descriptor(memory::ExpertKey, int&, int64_t&,
                         int64_t&) const override { return false; }
    bool is_direct() const override { return true; }
    bool is_pinned(memory::ExpertKey) const override { return false; }
    /// The O_DIRECT-aligned slot STRIDE (arena slot size) — identical to the
    /// prepacked on-disk stride for the same quant/shape.
    int64_t slot_size_bytes() const override { return slot_stride_bytes_; }
    std::optional<GgufExpertTypes> gguf_types_for_layer(
        int layer_idx) const override;

    // ── Live-prepack extras ─────────────────────────────────────────────

    /// Unpadded slot content size (quant.bytes_per_expert — the global MAX).
    int64_t raw_slot_bytes() const { return slot_size_bytes_raw_; }

    /// The source shard file paths (for ArenaCache source identities).
    const std::vector<std::filesystem::path>& shard_paths() const {
        return shard_paths_;
    }

    /// Per-layer projection sources for `layer_idx` (absolute index), or
    /// nullptr for non-MoE layers. Order: gate, up, down.
    const std::array<ProjSource, 3>* layer_sources(int layer_idx) const;

    /// Number of routed experts per MoE layer.
    int num_experts() const { return n_experts_; }

    /// The layer's packed content total (gate+up+down bytes), or 0.
    int64_t layer_packed_bytes(int layer_idx) const;

    /// True when shard `shard_idx`'s data fd is O_DIRECT.
    bool shard_is_o_direct(int shard_idx) const;

    /// Raw data fd for shard `shard_idx` (owned by this source).
    int shard_fd(int shard_idx) const;
    /// File size of shard `shard_idx` (for O_DIRECT EOF clamping).
    int64_t shard_file_size(int shard_idx) const;

    /// The expert shape used for pack validation.
    const ExpertShape& shape() const { return shape_; }

private:
    struct LayerInfo {
        bool is_moe = false;
        std::array<ProjSource, 3> proj{};  // gate, up, down
        GgufModelExpertTypes types{};
        int64_t packed_total = 0;          // gate+up+down bytes (<= raw slot)
    };

    // Per-shard open data fd (O_DIRECT when granted).
    struct ShardFd {
        int  fd = -1;
        bool o_direct = false;
        int64_t file_size = 0;
    };

    ExpertShape shape_{};
    int n_experts_ = 0;
    int64_t slot_size_bytes_raw_ = 0;   // quant.bytes_per_expert (unpadded)
    int64_t slot_stride_bytes_ = 0;     // aligned_slot_stride(raw)
    std::vector<LayerInfo> layers_;     // index = absolute layer idx
    std::vector<ShardFd> fds_;          // parallel to shards
    std::vector<std::filesystem::path> shard_paths_;
};

}  // namespace layerstorm::model
