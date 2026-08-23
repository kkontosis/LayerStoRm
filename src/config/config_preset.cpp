#include "config_preset.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace layerstorm::config {

namespace {

// Compile-time preset directory from CMake.
#ifndef LAYERSTORM_PRESET_DIR
#define LAYERSTORM_PRESET_DIR ""
#endif

constexpr const char* kForbiddenSections[] = {
    "model", "quantization", "hardware", "preset"
};

}  // namespace

void finalize_config(Config& cfg) {
    // INV-4.18: populate GpuRef from parsed fields
    for (int i = 0; i < static_cast<int>(cfg.hardware.gpus.size()); ++i) {
        auto& g = cfg.hardware.gpus[i];
        g.ref = GpuRef{.position = i, .id = g.id, .type = g.type};
    }

    // P-24b memory.arena_attach.on_conflict × persist compatibility — a
    // contradiction is a config error and must fail at PARSE time, never at
    // attach (mirrored as a structured error in validate_arena_attach).
    // persist=true permits ONLY on_conflict='fail' (or absent, which derives
    // to 'fail').
    const auto& aa = cfg.memory.arena_attach;
    if (aa.persist && aa.on_conflict == ArenaOnConflict::kill) {
        throw std::runtime_error(
            "memory.arena_attach.on_conflict='kill' contradicts persist=true: "
            "persist means the holder store is NEVER wiped. Use "
            "on_conflict='fail' (or omit it); to wipe + cold-rebuild "
            "intentionally, boot once with persist=false.");
    }
    if (aa.persist && aa.on_conflict == ArenaOnConflict::new_arena) {
        throw std::runtime_error(
            "memory.arena_attach.on_conflict='new' contradicts persist=true: "
            "the deployment declared the persistent holder store as this "
            "engine's arena — silently serving from a throwaway "
            "process-private arena would bypass it and double-allocate "
            "~500 GB host RAM behind the operator. Use on_conflict='fail' "
            "(or omit it), or set persist=false if a private arena is really "
            "intended.");
    }
}

void validate_preset_json(const nlohmann::json& preset_json,
                          const std::string& preset_name) {
    for (const char* sec : kForbiddenSections) {
        if (preset_json.contains(sec)) {
            throw std::runtime_error(
                "Preset '" + preset_name + "' contains forbidden key '" +
                sec + "'");
        }
    }
}

nlohmann::json load_preset(const std::string& preset_name,
                           const std::vector<std::string>& search_paths) {
    namespace fs = std::filesystem;

    const std::string filename = preset_name + ".json";

    for (const auto& dir : search_paths) {
        if (dir.empty()) continue;
        fs::path candidate = fs::path(dir) / filename;
        if (fs::exists(candidate)) {
            std::ifstream f(candidate);
            if (!f.is_open()) {
                throw std::runtime_error(
                    "Cannot open preset file: " + candidate.string());
            }
            return nlohmann::json::parse(f);
        }
    }

    std::string msg = "Preset '" + preset_name + "' not found. Searched:";
    for (const auto& dir : search_paths) {
        if (!dir.empty()) msg += "\n  " + dir + "/" + filename;
    }
    throw std::runtime_error(msg);
}

Config load_config(const nlohmann::json& user_json,
                   const std::vector<std::string>& preset_search_paths) {
    Config cfg;
    if (!user_json.contains("preset") || user_json["preset"].is_null()) {
        cfg = parse_config(user_json);
    } else {
        const std::string preset_name = user_json["preset"].get<std::string>();
        nlohmann::json preset_json = load_preset(preset_name, preset_search_paths);
        validate_preset_json(preset_json, preset_name);

        // User values win: merge user config on top of preset
        preset_json.merge_patch(user_json);
        // Remove consumed preset key (not a config section)
        preset_json.erase("preset");

        cfg = parse_config(preset_json);
    }
    finalize_config(cfg);
    return cfg;
}

Config load_config(const nlohmann::json& user_json) {
    return load_config(user_json, {LAYERSTORM_PRESET_DIR});
}

Config load_config(const std::string& config_path) {
    namespace fs = std::filesystem;

    std::ifstream f(config_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open config file: " + config_path);
    }
    nlohmann::json user_json = nlohmann::json::parse(f);

    std::string config_dir = fs::path(config_path).parent_path().string();
    std::string local_presets = (fs::path(config_dir) / "presets").string();

    return load_config(user_json, {local_presets, LAYERSTORM_PRESET_DIR});
}

void resolve_config_file_links(nlohmann::json& json,
                               const std::string& config_dir) {
    namespace fs = std::filesystem;

    for (size_t i = 0; i < kConfigFileLinkCount; ++i) {
        const auto& link = kConfigFileLinks[i];
        if (json.contains(link.section_key)) continue;

        fs::path file_path = fs::path(config_dir) / link.filename;
        if (!fs::exists(file_path)) continue;

        std::ifstream f(file_path);
        if (!f.is_open()) continue;

        json[link.section_key] = nlohmann::json::parse(f);
    }
}

}  // namespace layerstorm::config
