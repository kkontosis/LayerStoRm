#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "config_parser.h"

namespace layerstorm::config {

/// Load a config from a JSON file, applying preset overlay if specified.
///
/// If the JSON contains a "preset" key, the named preset file is loaded
/// and merged underneath the user config (user values win).
///
/// Preset search order:
///   1. <config_dir>/presets/<name>.json  (sibling to user config)
///   2. LAYERSTORM_PRESET_DIR/<name>.json (compile-time path)
///
/// Throws std::runtime_error on unknown preset, malformed JSON, or
/// preset containing forbidden sections (model/quantization/hardware).
Config load_config(const std::string& config_path);

/// Load from a JSON object with only the compile-time preset search path.
Config load_config(const nlohmann::json& user_json);

/// Load from a JSON object with explicit preset search paths.
Config load_config(const nlohmann::json& user_json,
                   const std::vector<std::string>& preset_search_paths);

/// Post-parse fixup: sets GpuConfig.position = array index (INV-4.18).
/// Called automatically by load_config(). Tests that call parse_config()
/// directly should call this before using the Config.
void finalize_config(Config& cfg);

/// Find and load a preset JSON by name, searching the given directories.
/// Returns the parsed preset JSON. Throws std::runtime_error if not found.
nlohmann::json load_preset(const std::string& preset_name,
                           const std::vector<std::string>& search_paths);

/// Validate that a preset JSON does not contain forbidden sections.
/// Throws std::runtime_error if it contains model, quantization,
/// hardware, or preset keys.
void validate_preset_json(const nlohmann::json& preset_json,
                          const std::string& preset_name);

/// Resolve x-configFile sections: for each known file-linked section,
/// if the _internal-<name> key is absent in the JSON, try loading
/// <filename> from config_dir and insert it. No-op if no file-linked
/// sections are registered or if all sections are already present.
void resolve_config_file_links(nlohmann::json& json,
                               const std::string& config_dir);

}  // namespace layerstorm::config
