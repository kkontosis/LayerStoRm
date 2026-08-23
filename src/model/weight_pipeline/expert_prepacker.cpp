#include "model/weight_pipeline/expert_prepacker.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <optional>
#include <unistd.h>
#include <unordered_map>

#include <spdlog/spdlog.h>

#include "config/config_parser.h"
#include "model/model_config.h"
#include "model/quantization/gguf_kquant.h"
#include "model/quantization/quant_interface.h"
#include "model/weight_loader/weight_loader.h"
#include "model/weight_pipeline/manifest.h"
#include "model/weight_pipeline/prepacked_format.h"

namespace layerstorm::model {

namespace {

/// Compute source freshness: max mtime across all safetensors shard files.
/// Returns {unix_timestamp_seconds, filename_of_newest_shard}.
std::pair<int64_t, std::string> compute_freshness(const LoadedModel& model) {
    namespace fs = std::filesystem;

    int64_t max_ts = 0;
    std::string max_file;

    auto consider = [&](const fs::path& path) {
        std::error_code ec;
        auto ftime = fs::last_write_time(path, ec);
        if (ec) return;
        // Convert file_time_type → system_clock → Unix epoch seconds.
        auto sys_time = std::chrono::clock_cast<std::chrono::system_clock>(ftime);
        auto epoch_s = std::chrono::duration_cast<std::chrono::seconds>(
            sys_time.time_since_epoch()).count();
        if (epoch_s > max_ts) {
            max_ts = epoch_s;
            max_file = path.filename().string();
        }
    };

    for (const auto& shard : model.shards) consider(shard.path());
    for (const auto& shard : model.gguf_shards) consider(shard.path());

    return {max_ts, max_file};
}

/// Extract the gate/up/down k-quant triple from one expert's bundles.
/// Returns std::nullopt (and sets result.error) if a projection is missing
/// or lacks a GGUF type.
std::optional<GgufModelExpertTypes> gguf_triple_from_bundles(
        const std::vector<WeightBundle>& bundles,
        PrepackResult& result, int expert_idx, int layer_idx) {
    std::optional<GgufKQuantType> gate, up, down;
    for (const auto& b : bundles) {
        auto t = b.gguf_type();
        if (!t) continue;
        switch (b.id.component) {
            case TensorComponent::gate_proj: gate = *t; break;
            case TensorComponent::up_proj:   up   = *t; break;
            case TensorComponent::down_proj: down = *t; break;
            default: break;
        }
    }
    if (!gate || !up || !down) {
        result.error = std::format(
            "expert {} layer {}: routed expert missing GGUF k-quant type "
            "(gate={} up={} down={})",
            expert_idx, layer_idx, gate.has_value(), up.has_value(),
            down.has_value());
        return std::nullopt;
    }
    return GgufModelExpertTypes{*gate, *up, *down};
}

/// Pack a single expert's bundles based on dtype and write the slot data to `os`.
/// Returns true on success.  On success, the packed data has been written and
/// the owned buffer released.  On failure, result.error is set.
/// `layer_gguf_types` (GG-10): non-null for GGUF — THIS layer's k-quant triple;
/// the slot then holds gate|up|down at the layer's real sizes (which may be
/// smaller than slot_size_bytes, the global-max upper bound) and the tail is
/// zero-padded to stride_bytes.
bool pack_and_write_slot(std::vector<WeightBundle>& bundles,
                         const ExpertShape& shape,
                         int64_t slot_size_bytes,
                         int64_t stride_bytes,
                         std::ostream& os,
                         PrepackResult& result,
                         int expert_idx, int layer_idx,
                         const Nvfp4InputScaleNorm* norm,
                         const GgufModelExpertTypes* layer_gguf_types) {
    if (bundles.empty()) {
        result.error = std::format(
            "expert {} layer {}: empty bundles", expert_idx, layer_idx);
        return false;
    }

    // GGUF k-quants take priority over the dtype switch: they are stored as
    // packed bytes (RawTensor.dtype == U8) but carry a gguf_type, and pack by
    // verbatim block copy (no scales/reformat) sized by each bundle's OWN
    // per-tensor k-quant type — validated strictly against the layer's triple.
    const bool is_gguf = bundles[0].weight.gguf_type.has_value();
    if (is_gguf) {
        if (!pack_gguf_expert(bundles, shape, layer_gguf_types)) {
            result.error = std::format(
                "expert {} layer {}: GGUF pack failed (k-quant type/size "
                "mismatch — see log)", expert_idx, layer_idx);
            return false;
        }
    } else {
        const auto dtype = bundles[0].weight.dtype;
        switch (dtype) {
            case SafetensorsDtype::U8:
                // NVFP4 — needs packing from separate gate/up/down + aux tensors.
                // Scales come out Sm1xx-interleaved with per-layer-normalized
                // input_scale scalars ("nvfp4-sm1xx").
                pack_nvfp4_expert(bundles, shape, norm);
                break;
            case SafetensorsDtype::F8_E4M3:
            case SafetensorsDtype::F8_E5M2:
                // FP8 — linearize weight + blockwise scale per projection.
                pack_fp8_expert(bundles, shape);
                break;
            default:
                result.error = std::format(
                    "expert {} layer {}: unsupported dtype {}",
                    expert_idx, layer_idx, static_cast<int>(dtype));
                return false;
        }
    }

    // After packing, bundles[0].packed_slot holds the contiguous slot.
    // GGUF (GG-10): the layer's real packed size may be SMALLER than the
    // global-max slot (per-layer mixed k-quants) — require 0 < size <= slot and
    // zero-pad the tail. All other formats pack exactly slot_size_bytes.
    const auto& data = bundles[0].packed_slot;
    const int64_t data_bytes = static_cast<int64_t>(data.size());
    if (is_gguf ? (data_bytes <= 0 || data_bytes > slot_size_bytes)
                : (data_bytes != slot_size_bytes)) {
        result.error = std::format(
            "expert {} layer {}: packed size {} {} expected slot size {}",
            expert_idx, layer_idx, data.size(),
            is_gguf ? "exceeds (or empty)" : "!=", slot_size_bytes);
        return false;
    }

    os.write(reinterpret_cast<const char*>(data.data()), data_bytes);
    if (!os) {
        result.error = std::format(
            "expert {} layer {}: write failed (disk full?)",
            expert_idx, layer_idx);
        return false;
    }

    // Zero-pad to the aligned on-disk stride so slot_offset and read length are
    // kSlotAlignBytes-aligned (O_DIRECT-able). For GGUF this also covers the
    // slot tail between the layer's real size and the global-max slot size.
    if (stride_bytes > data_bytes) {
        static const std::array<char, prepacked::kSlotAlignBytes> kZeroPad{};
        int64_t remaining = stride_bytes - data_bytes;
        while (remaining > 0) {
            const int64_t chunk = std::min<int64_t>(
                remaining, prepacked::kSlotAlignBytes);
            os.write(kZeroPad.data(), chunk);
            if (!os) {
                result.error = std::format(
                    "expert {} layer {}: pad write failed (disk full?)",
                    expert_idx, layer_idx);
                return false;
            }
            remaining -= chunk;
        }
    }

    // Release the packed buffer to keep memory bounded.
    bundles[0].owned_buf.reset();
    bundles[0].packed_slot = {};

    return true;
}

}  // namespace

// ── prepack_experts ─────────────────────────────────────────────────────────

PrepackResult prepack_experts(
    LoadedModel& model,
    const ModelConfig& model_cfg,
    const QuantInterface& quant,
    const config::Config& cfg,
    const std::filesystem::path& output_dir)
{
    namespace fs = std::filesystem;
    PrepackResult result;

    // 1. Create output directory.
    {
        std::error_code ec;
        fs::create_directories(output_dir, ec);
        if (ec) {
            result.error = std::format("cannot create directory {}: {}",
                                       output_dir.string(), ec.message());
            return result;
        }
    }

    // 2. Build ExpertShape from model config.
    const auto& raw = model_cfg.raw();
    const ExpertShape shape{raw.hidden_size, raw.moe_intermediate_size};

    // 3. Compute slot size and expected file size.
    const int64_t slot_size = quant.bytes_per_expert(shape);
    // On-disk stride padded to kSlotAlignBytes (O_DIRECT alignment); each slot is
    // written as slot_size real bytes + zero pad to slot_stride.
    const int64_t slot_stride = prepacked::aligned_slot_stride(slot_size);
    const auto& moe_indices = model_cfg.moe_layer_indices();
    const int n_moe_layers = static_cast<int>(moe_indices.size());
    const int64_t expected_size = prepacked::expected_file_size(n_moe_layers, slot_stride);
    const int n_experts = raw.n_routed_experts;

    if (n_moe_layers == 0 || n_experts == 0) {
        spdlog::warn("prepack_experts: nothing to do (moe_layers={}, experts={})",
                     n_moe_layers, n_experts);
        return result;
    }

    spdlog::info("Pre-processing {} experts × {} MoE layers → {} "
                 "(slot={} B, file={} B)",
                 n_experts, n_moe_layers, output_dir.string(),
                 slot_size, expected_size);

    // 4. Compute source freshness.
    auto [freshness_ts, freshness_file] = compute_freshness(model);

    // 4a. GGUF (GG-10): collect each MoE layer's routed-expert k-quant triple
    // (from expert 0 — pack_gguf_expert enforces cross-expert uniformity within
    // a layer against this table). `quant` is the per-projection MAX interface
    // (GG-9) that sizes the slot; individual layers may pack smaller.
    const bool is_gguf =
        quant.weight_quant() == config::WeightQuant::gguf ||
        (quant.name().size() >= 5 && quant.name().substr(0, 5) == "gguf_");
    std::vector<GgufModelExpertTypes> layer_gguf_types;  // parallel to moe_indices
    if (is_gguf) {
        layer_gguf_types.reserve(moe_indices.size());
        for (int layer_idx : moe_indices) {
            if (layer_idx < 0 ||
                layer_idx >= static_cast<int>(model.layers.size()) ||
                model.layers[layer_idx].routed_experts.empty() ||
                model.layers[layer_idx].routed_experts[0].empty()) {
                result.error = std::format(
                    "GGUF prepack: layer {} has no loaded routed experts to "
                    "read k-quant types from", layer_idx);
                return result;
            }
            auto triple = gguf_triple_from_bundles(
                model.layers[layer_idx].routed_experts[0], result,
                /*expert_idx=*/0, layer_idx);
            if (!triple) return result;  // result.error already set
            const int64_t layer_total =
                make_gguf_quant(triple->gate, triple->up, triple->down)
                    .bytes_per_expert(shape);
            if (layer_total > slot_size) {
                result.error = std::format(
                    "GGUF prepack: layer {} packed total {} exceeds slot size "
                    "{} (global gguf_types must be the per-projection MAX)",
                    layer_idx, layer_total, slot_size);
                return result;
            }
            layer_gguf_types.push_back(*triple);
        }
    }

    // 4b. Per-layer NVFP4 input_scale normalization (TRT-LLM semantics):
    // fc31 = max over experts of (gate ∪ up), fc2 = max over experts of down.
    // The expert loop below is expert-major, so compute each layer's norm once.
    std::unordered_map<int, Nvfp4InputScaleNorm> layer_norms;
    for (int layer_idx : moe_indices) {
        if (layer_idx >= 0 && layer_idx < static_cast<int>(model.layers.size())) {
            layer_norms[layer_idx] = compute_nvfp4_input_scale_norm(
                model.layers[layer_idx].routed_experts, layer_idx);
        }
    }

    // 5. Process each expert index.
    for (int ei = 0; ei < n_experts; ++ei) {
        const auto final_path = prepacked::expert_file_path(output_dir, ei);

        // Resumable: skip if the file already exists with correct size.
        {
            std::error_code ec;
            auto fsize = fs::file_size(final_path, ec);
            if (!ec && static_cast<int64_t>(fsize) == expected_size) {
                ++result.experts_skipped;
                continue;
            }
        }

        // Write to a temp file; rename on success.
        const auto tmp_path = fs::path(final_path.string() + ".tmp");

        {
            std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
            if (!ofs) {
                result.error = std::format("cannot open {} for writing",
                                           tmp_path.string());
                return result;
            }

            // Write one slot per MoE layer.
            for (size_t layer_pos = 0; layer_pos < moe_indices.size();
                 ++layer_pos) {
                const int layer_idx = moe_indices[layer_pos];
                if (layer_idx < 0 ||
                    layer_idx >= static_cast<int>(model.layers.size())) {
                    result.error = std::format(
                        "expert {}: layer_idx {} out of range (model has {} layers)",
                        ei, layer_idx, model.layers.size());
                    std::error_code ec;
                    fs::remove(tmp_path, ec);
                    return result;
                }

                auto& experts = model.layers[layer_idx].routed_experts;
                if (ei >= static_cast<int>(experts.size())) {
                    result.error = std::format(
                        "expert {}: layer {} has only {} routed experts",
                        ei, layer_idx, experts.size());
                    std::error_code ec;
                    fs::remove(tmp_path, ec);
                    return result;
                }

                auto norm_it = layer_norms.find(layer_idx);
                const Nvfp4InputScaleNorm* norm =
                    (norm_it != layer_norms.end()) ? &norm_it->second : nullptr;
                const GgufModelExpertTypes* layer_types =
                    is_gguf ? &layer_gguf_types[layer_pos] : nullptr;
                if (!pack_and_write_slot(experts[ei], shape, slot_size,
                                         slot_stride, ofs, result, ei, layer_idx,
                                         norm, layer_types)) {
                    std::error_code ec;
                    fs::remove(tmp_path, ec);
                    return result;
                }
            }
        }  // ofs closed

        // Flush data to disk before rename so a crash can't leave a
        // correct-size file with incomplete content (TD-93f).
        {
            int fd = ::open(tmp_path.c_str(), O_RDONLY);
            if (fd >= 0) {
                ::fdatasync(fd);
                ::close(fd);
            }
        }

        // Verify written size.
        {
            std::error_code ec;
            auto written_size = fs::file_size(tmp_path, ec);
            if (ec || static_cast<int64_t>(written_size) != expected_size) {
                result.error = std::format(
                    "expert {}: written file size {} != expected {}",
                    ei, ec ? -1 : static_cast<int64_t>(written_size),
                    expected_size);
                fs::remove(tmp_path, ec);
                return result;
            }
        }

        // Atomic rename.
        {
            std::error_code ec;
            fs::rename(tmp_path, final_path, ec);
            if (ec) {
                result.error = std::format("expert {}: rename failed: {}",
                                           ei, ec.message());
                fs::remove(tmp_path, ec);
                return result;
            }
        }

        ++result.experts_written;
        result.bytes_written += expected_size;

        spdlog::info("Pre-processed expert {}/{} ({:.1f} MB)",
                     ei + 1, n_experts, expected_size / 1e6);
    }

    // 6. Build and write manifest (only after all expert files succeed).
    Manifest manifest;
    manifest.format_version = std::string{prepacked::kFormatVersion};
    manifest.engine_version = std::string{prepacked::kFormatVersion};
    manifest.source_model_path = cfg.model.weights_path;
    manifest.source_freshness_timestamp = freshness_ts;
    manifest.source_freshness_file = freshness_file;
    // NVFP4 slots carry the Sm1xx layout subcategory: scales on disk are
    // pre-interleaved + input_scale scalars normalized (see prepacked_format.h).
    // GGUF slots carry the family marker "gguf" + the global-MAX gguf_types
    // block (slot sizing, GG-9) + the per-layer gguf_types_per_layer array
    // (GG-10 — the types each layer's slot content is actually packed at).
    if (quant.name() == "nvfp4") {
        manifest.quant_format = std::string{prepacked::kNvfp4PackedQuantFormat};
    } else if (is_gguf) {
        manifest.quant_format = std::string{prepacked::kGgufPackedQuantFormat};
        if (const auto* gq = dynamic_cast<const GgufQuantInterface*>(&quant)) {
            GgufExpertTypes types;
            types.gate = gq->projection_type(Projection::gate);
            types.up   = gq->projection_type(Projection::up);
            types.down = gq->projection_type(Projection::down);
            manifest.gguf_types = types;
        } else {
            result.error = "GGUF prepack requires a GgufQuantInterface to record "
                           "per-projection k-quant types in the manifest";
            return result;
        }
        manifest.gguf_types_per_layer.reserve(layer_gguf_types.size());
        for (const auto& t : layer_gguf_types) {
            manifest.gguf_types_per_layer.push_back(
                GgufExpertTypes{t.gate, t.up, t.down});
        }
    } else {
        manifest.quant_format = std::string{quant.name()};
    }
    manifest.n_routed_experts = n_experts;
    manifest.n_expert_files = n_experts;
    manifest.expert_dimensions = ManifestExpertDimensions{
        raw.hidden_size, raw.moe_intermediate_size};
    manifest.moe_layers.count = n_moe_layers;
    manifest.moe_layers.first_moe_layer = moe_indices.front();
    manifest.moe_layers.last_moe_layer = moe_indices.back();
    manifest.moe_layers.indices = moe_indices;
    manifest.slot = build_slot_from_quant(quant, shape);

    write_manifest(output_dir, manifest);

    // Verify the manifest we just wrote.
    auto verify = verify_manifest(manifest, quant);
    if (!verify.ok) {
        result.error = std::format("manifest self-verification failed: {}",
                                   verify.error);
        return result;
    }

    spdlog::info("Pre-processing complete: {} written, {} skipped, {:.2f} GB total",
                 result.experts_written, result.experts_skipped,
                 result.bytes_written / (1024.0 * 1024.0 * 1024.0));

    return result;
}

}  // namespace layerstorm::model
