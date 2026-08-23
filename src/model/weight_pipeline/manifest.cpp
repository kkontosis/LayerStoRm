#include "model/weight_pipeline/manifest.h"
#include "model/weight_pipeline/prepacked_format.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace layerstorm::model {

// ── ManifestExpertDimensions ────────────────────────────────────────────────

ExpertShape ManifestExpertDimensions::to_expert_shape() const {
    return ExpertShape{hidden_size, intermediate_size};
}

// ── ManifestMoeLayers ───────────────────────────────────────────────────────

int ManifestMoeLayers::layer_position(int layer_idx) const {
    if (is_contiguous()) {
        if (layer_idx < first_moe_layer || layer_idx > last_moe_layer) return -1;
        return layer_idx - first_moe_layer;
    }
    auto it = std::lower_bound(indices.begin(), indices.end(), layer_idx);
    if (it == indices.end() || *it != layer_idx) return -1;
    return static_cast<int>(it - indices.begin());
}

bool ManifestMoeLayers::is_contiguous() const {
    return count == (last_moe_layer - first_moe_layer + 1);
}

// ── JSON serialization ──────────────────────────────────────────────────────

nlohmann::json manifest_to_json(const Manifest& m) {
    nlohmann::json j;

    j["format_version"] = m.format_version;
    j["engine_version"] = m.engine_version;

    j["source_model_path"] = m.source_model_path;
    j["source_freshness_timestamp"] = m.source_freshness_timestamp;
    j["source_freshness_file"] = m.source_freshness_file;

    j["quant_format"] = m.quant_format;
    j["n_routed_experts"] = m.n_routed_experts;
    j["n_expert_files"] = m.n_expert_files;

    j["expert_dimensions"] = {
        {"hidden_size", m.expert_dimensions.hidden_size},
        {"intermediate_size", m.expert_dimensions.intermediate_size}
    };

    j["moe_layers"] = {
        {"count", m.moe_layers.count},
        {"first_moe_layer", m.moe_layers.first_moe_layer},
        {"last_moe_layer", m.moe_layers.last_moe_layer},
        {"indices", m.moe_layers.indices}
    };

    // GGUF per-projection k-quant types (present only for quant_format "gguf").
    // This is the per-projection MAXIMUM triple (slot-sizing upper bound, GG-9).
    if (m.gguf_types) {
        j["gguf_types"] = {
            {"gate", std::string{gguf::type_name(m.gguf_types->gate)}},
            {"up",   std::string{gguf::type_name(m.gguf_types->up)}},
            {"down", std::string{gguf::type_name(m.gguf_types->down)}}
        };
    }

    // GG-10: per-MoE-layer k-quant triples (ascending layer order, parallel to
    // moe_layers.indices). Mandatory for quant_format "gguf" as of 9.69.0.
    if (!m.gguf_types_per_layer.empty()) {
        nlohmann::json per_layer = nlohmann::json::array();
        for (const auto& t : m.gguf_types_per_layer) {
            per_layer.push_back({
                {"gate", std::string{gguf::type_name(t.gate)}},
                {"up",   std::string{gguf::type_name(t.up)}},
                {"down", std::string{gguf::type_name(t.down)}}
            });
        }
        j["gguf_types_per_layer"] = std::move(per_layer);
    }

    nlohmann::json projs = nlohmann::json::array();
    for (const auto& p : m.slot.projections) {
        projs.push_back({
            {"name", p.name},
            {"offset", p.offset},
            {"total_bytes", p.total_bytes},
            {"weight_bytes", p.weight_bytes},
            {"scale_bytes", p.scale_bytes},
            {"scalar_bytes", p.scalar_bytes},
            {"description", p.description}
        });
    }

    j["slot"] = {
        {"slot_size_bytes", m.slot.slot_size_bytes},
        {"stride_bytes", m.slot.stride_bytes},
        {"alignment_bytes", m.slot.alignment_bytes},
        {"projections", projs}
    };

    return j;
}

Manifest manifest_from_json(const nlohmann::json& j) {
    Manifest m;

    m.format_version = j.at("format_version").get<std::string>();
    m.engine_version = j.at("engine_version").get<std::string>();

    m.source_model_path = j.at("source_model_path").get<std::string>();
    m.source_freshness_timestamp = j.at("source_freshness_timestamp").get<int64_t>();
    m.source_freshness_file = j.at("source_freshness_file").get<std::string>();

    m.quant_format = j.at("quant_format").get<std::string>();
    m.n_routed_experts = j.at("n_routed_experts").get<int>();
    m.n_expert_files = j.at("n_expert_files").get<int>();

    const auto& ed = j.at("expert_dimensions");
    m.expert_dimensions.hidden_size = ed.at("hidden_size").get<int>();
    m.expert_dimensions.intermediate_size = ed.at("intermediate_size").get<int>();

    const auto& ml = j.at("moe_layers");
    m.moe_layers.count = ml.at("count").get<int>();
    m.moe_layers.first_moe_layer = ml.at("first_moe_layer").get<int>();
    m.moe_layers.last_moe_layer = ml.at("last_moe_layer").get<int>();
    m.moe_layers.indices = ml.at("indices").get<std::vector<int>>();

    if (j.contains("gguf_types")) {
        const auto& gt = j.at("gguf_types");
        GgufExpertTypes types;
        types.gate = gguf::type_from_name(gt.at("gate").get<std::string>());
        types.up   = gguf::type_from_name(gt.at("up").get<std::string>());
        types.down = gguf::type_from_name(gt.at("down").get<std::string>());
        m.gguf_types = types;
    }

    // GG-10: per-MoE-layer triples. type_from_name throws on an invalid name,
    // so a corrupt entry is rejected here at parse time.
    if (j.contains("gguf_types_per_layer")) {
        for (const auto& e : j.at("gguf_types_per_layer")) {
            GgufExpertTypes types;
            types.gate = gguf::type_from_name(e.at("gate").get<std::string>());
            types.up   = gguf::type_from_name(e.at("up").get<std::string>());
            types.down = gguf::type_from_name(e.at("down").get<std::string>());
            m.gguf_types_per_layer.push_back(types);
        }
    }

    const auto& sl = j.at("slot");
    m.slot.slot_size_bytes = sl.at("slot_size_bytes").get<int64_t>();
    m.slot.alignment_bytes = sl.at("alignment_bytes").get<int>();
    // stride_bytes: on-disk slot stride (>= slot_size_bytes). Absent in legacy
    // manifests → 0 (stride == slot_size_bytes, unpadded).
    m.slot.stride_bytes = sl.value("stride_bytes", static_cast<int64_t>(0));

    for (const auto& pj : sl.at("projections")) {
        ManifestProjection p;
        p.name = pj.at("name").get<std::string>();
        p.offset = pj.at("offset").get<int64_t>();
        p.total_bytes = pj.at("total_bytes").get<int64_t>();
        p.weight_bytes = pj.at("weight_bytes").get<int64_t>();
        p.scale_bytes = pj.at("scale_bytes").get<int64_t>();
        p.scalar_bytes = pj.at("scalar_bytes").get<int64_t>();
        p.description = pj.at("description").get<std::string>();
        m.slot.projections.push_back(std::move(p));
    }

    return m;
}

// ── File I/O ────────────────────────────────────────────────────────────────

Manifest read_manifest(const std::filesystem::path& prepacked_dir) {
    auto path = prepacked::manifest_path(prepacked_dir);
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open manifest: " + path.string());
    }
    return manifest_from_json(nlohmann::json::parse(f));
}

void write_manifest(const std::filesystem::path& prepacked_dir, const Manifest& m) {
    auto path = prepacked::manifest_path(prepacked_dir);
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot write manifest: " + path.string());
    }
    f << manifest_to_json(m).dump(2) << '\n';
}

// ── Verification ────────────────────────────────────────────────────────────

ManifestVerifyResult verify_manifest(const Manifest& m, const QuantInterface& quant) {
    if (m.format_version != prepacked::kFormatVersion) {
        return {false, "Unsupported format version: " + m.format_version
                       + " (engine supports " + std::string{prepacked::kFormatVersion} + ")"};
    }

    // NVFP4 packed data must carry the Sm1xx layout subcategory: slots store
    // group scales pre-interleaved (no runtime reformat exists). A plain
    // "nvfp4" manifest means raw-layout scales — refuse it loudly rather than
    // serve quietly-wrong GEMMs.
    // GGUF: the engine-side quant interface is a (possibly mixed) GgufQuantInterface
    // whose name() is "gguf" (mixed) or "gguf_qX_k" (uniform). Either way the
    // manifest marker is the family marker "gguf"; the per-projection types are
    // carried in the gguf_types block (validated below).
    const bool is_gguf_quant =
        quant.weight_quant() == config::WeightQuant::gguf ||
        (quant.name().size() >= 5 && quant.name().substr(0, 5) == "gguf_");
    const std::string expected_quant =
        (quant.name() == "nvfp4") ? std::string{prepacked::kNvfp4PackedQuantFormat}
      : is_gguf_quant             ? std::string{prepacked::kGgufPackedQuantFormat}
                                  : std::string{quant.name()};
    if (m.quant_format != expected_quant) {
        return {false, "Quant format mismatch: manifest=" + m.quant_format
                       + " engine expects=" + expected_quant};
    }

    // GGUF: the gguf_types block (the per-projection MAX triple that sizes the
    // slot, GG-9) is mandatory and must match the engine quant's per-projection
    // types (the slot is sized from them). This catches a manifest written for
    // one k-quant mix being verified against another.
    if (is_gguf_quant) {
        if (!m.gguf_types) {
            return {false, "GGUF manifest missing gguf_types block (gate/up/down "
                           "k-quant types) — regenerate"};
        }
        if (const auto* gq = dynamic_cast<const GgufQuantInterface*>(&quant)) {
            const std::pair<Projection, GgufKQuantType> checks[] = {
                {Projection::gate, m.gguf_types->gate},
                {Projection::up,   m.gguf_types->up},
                {Projection::down, m.gguf_types->down},
            };
            for (auto [proj, mtype] : checks) {
                if (gq->projection_type(proj) != mtype) {
                    return {false, "GGUF gguf_types mismatch: manifest projection "
                                   "type differs from engine quant interface"};
                }
            }
        }

        // GG-10: the per-layer triples block is mandatory (one entry per MoE
        // layer, ascending layer order) and every layer's real packed total
        // must fit the global-max slot.
        if (m.gguf_types_per_layer.empty()) {
            return {false, "GGUF manifest missing gguf_types_per_layer block "
                           "(per-MoE-layer gate/up/down k-quant types) — "
                           "regenerate"};
        }
        if (static_cast<int>(m.gguf_types_per_layer.size()) != m.moe_layers.count) {
            return {false, "gguf_types_per_layer has "
                           + std::to_string(m.gguf_types_per_layer.size())
                           + " entries but moe_layers.count="
                           + std::to_string(m.moe_layers.count)};
        }
        auto layer_shape = m.expert_dimensions.to_expert_shape();
        for (size_t i = 0; i < m.gguf_types_per_layer.size(); ++i) {
            int64_t layer_total =
                m.gguf_types_per_layer[i].to_quant().bytes_per_expert(layer_shape);
            if (layer_total > m.slot.slot_size_bytes) {
                return {false, "gguf_types_per_layer[" + std::to_string(i)
                               + "] packed total " + std::to_string(layer_total)
                               + " exceeds slot size "
                               + std::to_string(m.slot.slot_size_bytes)
                               + " (gguf_types must be the per-projection MAX)"};
            }
        }
    }

    auto shape = m.expert_dimensions.to_expert_shape();

    int64_t expected_slot = quant.bytes_per_expert(shape);
    if (m.slot.slot_size_bytes != expected_slot) {
        return {false, "Slot size mismatch: manifest=" + std::to_string(m.slot.slot_size_bytes)
                       + " expected=" + std::to_string(expected_slot)};
    }

    // stride_bytes (on-disk slot stride) is MANDATORY: the strided/padded layout
    // is the only supported format. A legacy unpadded manifest (stride_bytes==0)
    // is rejected — regenerate via prepack_experts.
    {
        int64_t expected_stride = prepacked::aligned_slot_stride(expected_slot);
        if (m.slot.stride_bytes != expected_stride ||
            m.slot.stride_bytes % prepacked::kSlotAlignBytes != 0) {
            return {false, "Slot stride mismatch (legacy/unpadded data not "
                           "supported — regenerate): manifest="
                           + std::to_string(m.slot.stride_bytes)
                           + " expected=" + std::to_string(expected_stride)};
        }
    }

    if (m.slot.projections.size() != 3) {
        return {false, "Expected 3 projections, got "
                       + std::to_string(m.slot.projections.size())};
    }

    static constexpr Projection kProjs[] = {Projection::gate, Projection::up, Projection::down};
    static constexpr const char* kProjNames[] = {"gate", "up", "down"};

    for (int i = 0; i < 3; ++i) {
        const auto& mp = m.slot.projections[i];
        if (mp.name != kProjNames[i]) {
            return {false, "Projection " + std::to_string(i) + " name mismatch: "
                           + mp.name + " vs " + kProjNames[i]};
        }
        int64_t expected_total = quant.bytes_per_projection(shape, kProjs[i]);
        if (mp.total_bytes != expected_total) {
            return {false, "Projection " + mp.name + " total_bytes mismatch: manifest="
                           + std::to_string(mp.total_bytes)
                           + " expected=" + std::to_string(expected_total)};
        }
    }

    if (static_cast<int>(m.moe_layers.indices.size()) != m.moe_layers.count) {
        return {false, "moe_layers.count=" + std::to_string(m.moe_layers.count)
                       + " but indices has " + std::to_string(m.moe_layers.indices.size())
                       + " entries"};
    }

    if (m.moe_layers.count > 0) {
        if (m.moe_layers.indices.front() != m.moe_layers.first_moe_layer) {
            return {false, "first_moe_layer=" + std::to_string(m.moe_layers.first_moe_layer)
                           + " but indices[0]=" + std::to_string(m.moe_layers.indices.front())};
        }
        if (m.moe_layers.indices.back() != m.moe_layers.last_moe_layer) {
            return {false, "last_moe_layer=" + std::to_string(m.moe_layers.last_moe_layer)
                           + " but indices[last]=" + std::to_string(m.moe_layers.indices.back())};
        }
        if (!std::is_sorted(m.moe_layers.indices.begin(), m.moe_layers.indices.end())) {
            return {false, "moe_layers.indices is not sorted ascending"};
        }
    }

    if (m.n_expert_files != m.n_routed_experts) {
        return {false, "n_expert_files=" + std::to_string(m.n_expert_files)
                       + " != n_routed_experts=" + std::to_string(m.n_routed_experts)};
    }

    return {true, ""};
}

// ── Builder ─────────────────────────────────────────────────────────────────

ManifestSlot build_slot_from_quant(const QuantInterface& quant, const ExpertShape& shape) {
    ManifestSlot slot;
    slot.slot_size_bytes = quant.bytes_per_expert(shape);
    // On-disk stride padded to kSlotAlignBytes for O_DIRECT-able reads.
    slot.stride_bytes = prepacked::aligned_slot_stride(slot.slot_size_bytes);

    static constexpr Projection kProjs[] = {Projection::gate, Projection::up, Projection::down};
    static constexpr const char* kProjNames[] = {"gate", "up", "down"};

    int64_t offset = 0;
    for (int i = 0; i < 3; ++i) {
        ManifestProjection p;
        p.name = kProjNames[i];
        p.offset = offset;
        p.total_bytes = quant.bytes_per_projection(shape, kProjs[i]);
        p.weight_bytes = quant.weight_bytes_per_projection(shape, kProjs[i]);
        p.scalar_bytes = quant.replicated_bytes_per_projection();
        p.scale_bytes = p.total_bytes - p.weight_bytes - p.scalar_bytes;

        const bool is_gguf =
            quant.weight_quant() == config::WeightQuant::gguf ||
            (quant.name().size() >= 5 && quant.name().substr(0, 5) == "gguf_");
        if (quant.name() == "nvfp4") {
            p.description = "[FP4_weight | UE8M0_group_scale | PAD | weight_scale_2_F32 | input_scale_F32]";
        } else if (is_gguf) {
            // K-quant super-blocks carry their scales/mins in-block: one
            // contiguous region of raw GGUF blocks, no separate scale region.
            p.description = "[GGUF_kquant_blocks (scales/mins packed in-block)]";
        } else {
            p.description = "[FP8_weight | F32_blockwise_scale]";
        }

        offset += p.total_bytes;
        slot.projections.push_back(std::move(p));
    }

    // Derive per-projection alignment from actual sizes (largest power-of-2
    // factor common to all projections). NVFP4: 128 (TMA B-pointer alignment),
    // FP8: depends on model dimensions.
    int64_t g = slot.projections[0].total_bytes;
    for (size_t i = 1; i < slot.projections.size(); ++i)
        g = std::gcd(g, slot.projections[i].total_bytes);
    slot.alignment_bytes = static_cast<int>(g & -g);

    return slot;
}

}  // namespace layerstorm::model
