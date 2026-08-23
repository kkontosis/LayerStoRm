#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "model/quantization/gguf_kquant.h"
#include "model/quantization/quant_interface.h"

namespace layerstorm::model {

// ── Manifest sub-structures ─────────────────────────────────────────────────

struct ManifestExpertDimensions {
    int hidden_size = 0;
    int intermediate_size = 0;

    ExpertShape to_expert_shape() const;
};

struct ManifestMoeLayers {
    int count = 0;
    int first_moe_layer = 0;
    int last_moe_layer = 0;
    std::vector<int> indices;

    /// Returns the position of layer_idx within indices, or -1 if not found.
    /// For contiguous ranges, this is layer_idx - first_moe_layer.
    int layer_position(int layer_idx) const;

    /// Returns true if indices form a contiguous range [first, last].
    bool is_contiguous() const;
};

struct ManifestProjection {
    std::string name;          // "gate", "up", "down"
    int64_t offset = 0;        // byte offset within slot
    int64_t total_bytes = 0;   // total bytes for this projection (aligned)
    int64_t weight_bytes = 0;  // raw weight bytes (pre-scale)
    int64_t scale_bytes = 0;   // scale/group-scale bytes (includes alignment padding)
    int64_t scalar_bytes = 0;  // per-projection scalar bytes (e.g. 8 for NVFP4)
    std::string description;   // human-readable layout description
};

// ── GgufExpertTypes ─────────────────────────────────────────────────────────
// The three per-projection GGUF k-quant types, serialized by name in the
// manifest. Reconstructs the mixed/uniform QuantInterface for slot sizing.
// Experts are stacked 3D tensors with ONE ggml_type per projection (uniform
// across experts), so three fields suffice.

struct GgufExpertTypes {
    GgufKQuantType gate = GgufKQuantType::Q4_K;
    GgufKQuantType up   = GgufKQuantType::Q4_K;
    GgufKQuantType down = GgufKQuantType::Q4_K;

    /// Build the GgufQuantInterface this manifest's experts are sized by.
    GgufQuantInterface to_quant() const { return make_gguf_quant(gate, up, down); }
};

struct ManifestSlot {
    int64_t slot_size_bytes = 0;   ///< real packed expert bytes (sum of projections)
    // On-disk per-slot stride (>= slot_size_bytes, multiple of kSlotAlignBytes).
    // slot_offset = layer_pos * stride_bytes. The tail [slot_size_bytes, stride)
    // is zero padding for O_DIRECT alignment. 0 = legacy (stride == slot_size_bytes).
    int64_t stride_bytes = 0;
    int alignment_bytes = 0;
    std::vector<ManifestProjection> projections;  // exactly 3: gate, up, down
};

// ── Manifest ────────────────────────────────────────────────────────────────

struct Manifest {
    // Versioning
    std::string format_version;
    std::string engine_version;

    // Source linkage
    std::string source_model_path;
    int64_t source_freshness_timestamp = 0;
    std::string source_freshness_file;

    // Model identity
    std::string quant_format;     // "nvfp4", "fp8_e4m3", "fp8_e5m2", "gguf"
    int n_routed_experts = 0;
    int n_expert_files = 0;
    ManifestExpertDimensions expert_dimensions;

    // Layer mapping
    ManifestMoeLayers moe_layers;

    // GGUF per-projection k-quant types (gate/up/down). Present ONLY for
    // quant_format == "gguf". This is the per-projection MAXIMUM triple across
    // all MoE layers (GG-9): it sizes the slot / expert-cache entry (an upper
    // bound every layer fits in) via make_gguf_quant(gate, up, down). It is NOT
    // necessarily any single layer's actual triple — see gguf_types_per_layer.
    std::optional<GgufExpertTypes> gguf_types;

    // GG-10: per-MoE-layer routed-expert k-quant triples, one entry per MoE
    // layer in ascending layer order (parallel to moe_layers.indices). A
    // per-layer mixed "XL" GGUF (Unsloth-Dynamic) varies the routed types
    // across layers; each slot is packed gate|up|down at ITS layer's real
    // sizes, zero-padded to the global-max stride. Mandatory (count ==
    // moe_layers.count) when quant_format == "gguf"; empty otherwise.
    std::vector<GgufExpertTypes> gguf_types_per_layer;

    // Slot layout
    ManifestSlot slot;
};

// ── Serialization ───────────────────────────────────────────────────────────

nlohmann::json manifest_to_json(const Manifest& m);

/// Throws std::runtime_error on missing/invalid fields.
Manifest manifest_from_json(const nlohmann::json& j);

// ── File I/O ────────────────────────────────────────────────────────────────

/// Read manifest.json from a pre-processed directory.
Manifest read_manifest(const std::filesystem::path& prepacked_dir);

/// Write manifest.json to a pre-processed directory.
void write_manifest(const std::filesystem::path& prepacked_dir, const Manifest& m);

// ── Verification ────────────────────────────────────────────────────────────

struct ManifestVerifyResult {
    bool ok = false;
    std::string error;   // empty if ok
};

/// Verify manifest slot layout matches engine's QuantInterface.
ManifestVerifyResult verify_manifest(const Manifest& m, const QuantInterface& quant);

// ── Builder ─────────────────────────────────────────────────────────────────

/// Build a ManifestSlot from a QuantInterface (single source of truth for slot layout).
ManifestSlot build_slot_from_quant(const QuantInterface& quant, const ExpertShape& shape);

}  // namespace layerstorm::model
