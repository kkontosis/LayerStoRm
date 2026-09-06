#include "model/weight_loader/quant_skip_list.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace layerstorm::model {

namespace {

// Trailing role suffixes a MODULE-path skip entry does not carry but a
// TENSOR path does. `weight_scale`/`weight_scale_inv` ride with their parent
// module's decision; `bias` only strips when the module entry itself covers
// it (entries that DO name a bias — `...proj.bias` — hit the exact match on
// the full name first).
constexpr std::string_view kRoleSuffixes[] = {
    ".weight_scale_inv", ".weight_scale", ".weight", ".bias"};

}  // namespace

std::string QuantSkipList::canonicalize(std::string_view name) {
    // `model.language_model.X` -> `model.X` (multimodal text wrapper).
    constexpr std::string_view kLm = "model.language_model.";
    if (name.substr(0, kLm.size()) == kLm) {
        std::string out = "model.";
        out.append(name.substr(kLm.size()));
        // A wrapped vision path cannot occur (`model.language_model.visual`
        // does not exist) — no second pass needed.
        return out;
    }
    // `model.visual` / `model.visual.X` -> `visual` / `visual.X` (the skip
    // list uses BOTH `visual.*` and `model.visual`; tensors use
    // `model.visual.*` — fold all three into `visual...`).
    constexpr std::string_view kMv = "model.visual";
    if (name.substr(0, kMv.size()) == kMv &&
        (name.size() == kMv.size() || name[kMv.size()] == '.')) {
        return std::string(name.substr(std::string_view("model.").size()));
    }
    return std::string(name);
}

QuantSkipList QuantSkipList::from_entries(std::vector<std::string> entries,
                                          std::string quant_method) {
    QuantSkipList sl;
    sl.quant_method_ = std::move(quant_method);
    sl.entry_count_ = entries.size();
    for (auto& e : entries) {
        if (e.empty()) continue;
        if (e.find('.') == std::string::npos) {
            sl.segments_.insert(std::move(e));
        } else {
            sl.exact_.insert(canonicalize(e));
        }
    }
    return sl;
}

std::optional<QuantSkipList> QuantSkipList::load_from_model_dir(
    const std::filesystem::path& model_dir) {
    const auto cfg_path = model_dir / "config.json";
    std::ifstream ifs(cfg_path);
    if (!ifs.good()) return std::nullopt;

    nlohmann::json j;
    try {
        ifs >> j;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Malformed " + cfg_path.string() + ": " +
                                 e.what());
    }

    // quantization_config lives either at the top level (GLM-5.3-Flash) or
    // would be absent entirely for BF16 checkpoints.
    if (!j.contains("quantization_config")) return std::nullopt;
    const auto& qc = j["quantization_config"];
    if (!qc.is_object() || !qc.contains("modules_to_not_convert"))
        return std::nullopt;
    const auto& list = qc["modules_to_not_convert"];
    if (!list.is_array())
        throw std::runtime_error(
            cfg_path.string() +
            ": quantization_config.modules_to_not_convert is not an array");

    std::vector<std::string> entries;
    entries.reserve(list.size());
    for (const auto& e : list) {
        if (!e.is_string())
            throw std::runtime_error(
                cfg_path.string() +
                ": non-string entry in modules_to_not_convert");
        entries.push_back(e.get<std::string>());
    }
    std::string method = qc.value("quant_method", std::string{});
    spdlog::info(
        "Quantization skip list: {} modules_to_not_convert entries "
        "(quant_method '{}') from {}",
        entries.size(), method, cfg_path.string());
    return from_entries(std::move(entries), std::move(method));
}

bool QuantSkipList::matches(std::string_view tensor_name) const {
    const std::string canon = canonicalize(tensor_name);

    // Candidate module paths: the full name, and the name minus one trailing
    // role suffix.
    std::string_view candidates[2];
    size_t n_cand = 0;
    candidates[n_cand++] = canon;
    for (auto suf : kRoleSuffixes) {
        if (canon.size() > suf.size() &&
            std::string_view(canon).substr(canon.size() - suf.size()) == suf) {
            candidates[n_cand++] =
                std::string_view(canon).substr(0, canon.size() - suf.size());
            break;
        }
    }

    for (size_t i = 0; i < n_cand; ++i) {
        std::string_view c = candidates[i];
        // 1. exact match.
        if (exact_.count(std::string(c))) return true;
        // 2. subtree match: any dot-prefix of c that is an entry covers c.
        for (size_t pos = c.find('.'); pos != std::string_view::npos;
             pos = c.find('.', pos + 1)) {
            if (exact_.count(std::string(c.substr(0, pos)))) return true;
        }
    }

    // 3. module-class catch-alls: match any path segment of the full name.
    if (!segments_.empty()) {
        std::string_view rest = canon;
        while (!rest.empty()) {
            const size_t pos = rest.find('.');
            const std::string_view seg =
                (pos == std::string_view::npos) ? rest : rest.substr(0, pos);
            if (segments_.count(std::string(seg))) return true;
            if (pos == std::string_view::npos) break;
            rest.remove_prefix(pos + 1);
        }
    }
    return false;
}

}  // namespace layerstorm::model
