#pragma once

#include <cstdint>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>

namespace layerstorm::model::prepacked {

// ── Format constants ────────────────────────────────────────────────────────

// 9.69.0 (GG-10): PER-LAYER mixed GGUF k-quant types (Unsloth-Dynamic files).
// The global gguf_types block is now documented as the per-projection MAXIMUM
// triple (slot/cache sizing upper bound, GG-9); a NEW mandatory (for gguf)
// gguf_types_per_layer array carries one {gate,up,down} triple per MoE layer
// (ascending layer index). Each slot is packed gate|up|down back-to-back at
// the LAYER's real sizes; the slot tail up to the global-max stride is zero
// padding. 9.68.0 gguf manifests (which packed/validated every layer at the
// global triple) are refused — regenerate.
// 9.68.0 (GG-6): GGUF k-quant experts — quant_format "gguf" + a 3-field
// per-projection gguf_types block (gate/up/down k-quant) in the manifest;
// slot = raw GGUF blocks copied verbatim (scales pack in-block, no scale region,
// replicated==0). NVFP4/FP8 manifests unchanged structurally. 9.67.0 and earlier
// manifests have no gguf_types block (absent → not a GGUF manifest).
// 9.67.0: NVFP4 group scales stored Sm1xx-interleaved on disk (quant_format
// "nvfp4-sm1xx"); per-slot input_scale scalars normalized to the per-layer max
// (TRT-LLM semantics). 9.66.0 raw-scale data is refused — regenerate.
inline constexpr std::string_view kFormatVersion = "9.69.0";
inline constexpr std::string_view kManifestFilename = "manifest.json";

// quant_format marker for GGUF k-quant experts. Unlike NVFP4 (which has a
// packed-layout subcategory "nvfp4-sm1xx"), GGUF blocks are copied verbatim, so
// the marker is just "gguf"; the per-projection k-quant types live in the
// manifest's gguf_types block (the slot layout is reconstructed from them via
// make_gguf_quant(gate,up,down)).
inline constexpr std::string_view kGgufPackedQuantFormat = "gguf";

// Packed-layout subcategory of the quant type for NVFP4: group scales are
// pre-interleaved in the CUTLASS Sm1xx SFB atom layout (128 rows x 4 groups),
// consumed directly by all Sm1xx-family devices (SM100..SM120). The raw
// checkpoint quant stays "nvfp4"; a future device family with a different
// scale layout must introduce its own subcategory and repack.
inline constexpr std::string_view kNvfp4PackedQuantFormat = "nvfp4-sm1xx";

// On-disk slot stride alignment. Each expert slot is padded so its stride (and
// thus every slot_offset) is a multiple of this, enabling O_DIRECT reads (which
// require offset/length/buffer aligned to the device block size). 4096 covers
// both common logical block sizes (512/4096) and the host page size. The padding
// is dead tail bytes; the real expert content is described by slot.projections
// and the manifest's (unpadded) slot_size_bytes.
inline constexpr int64_t kSlotAlignBytes = 4096;

/// Round a raw slot size up to the kSlotAlignBytes stride.
inline constexpr int64_t aligned_slot_stride(int64_t raw_slot_bytes) {
    return (raw_slot_bytes + kSlotAlignBytes - 1) / kSlotAlignBytes * kSlotAlignBytes;
}

// ── Path helpers ────────────────────────────────────────────────────────────

/// Returns "expert_042.bin" for expert_idx=42.
inline std::string expert_filename(int expert_idx) {
    return std::format("expert_{:03d}.bin", expert_idx);
}

/// Returns full path: dir / "expert_042.bin".
inline std::filesystem::path expert_file_path(const std::filesystem::path& dir,
                                               int expert_idx) {
    return dir / expert_filename(expert_idx);
}

/// Returns full path: dir / "manifest.json".
inline std::filesystem::path manifest_path(const std::filesystem::path& dir) {
    return dir / std::string{kManifestFilename};
}

// ── Directory helpers ───────────────────────────────────────────────────────

/// Returns the expert storage directory under a drive mount point.
inline std::filesystem::path expert_dir(const std::filesystem::path& drive) {
    return drive / "layerstorm" / "experts";
}

// ── Offset computation ──────────────────────────────────────────────────────

/// Byte offset within an expert file for a given layer position.
/// layer_pos is the index into moe_layers.indices (0-based).
inline int64_t slot_offset(int layer_pos, int64_t slot_size_bytes) {
    return static_cast<int64_t>(layer_pos) * slot_size_bytes;
}

/// Expected file size for one expert_NNN.bin file.
inline int64_t expected_file_size(int n_moe_layers, int64_t slot_size_bytes) {
    return static_cast<int64_t>(n_moe_layers) * slot_size_bytes;
}

}  // namespace layerstorm::model::prepacked
