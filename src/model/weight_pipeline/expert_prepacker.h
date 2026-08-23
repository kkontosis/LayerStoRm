#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace layerstorm::model {

struct LoadedModel;
class ModelConfig;
class QuantInterface;

}  // namespace layerstorm::model

namespace layerstorm::config {
struct Config;
}  // namespace layerstorm::config

namespace layerstorm::model {

// ── ExpertPrepacker ────────────────────────────────────────────────────────
// Offline pre-processing engine: converts raw safetensors expert weights
// into the pre-packed on-disk format (expert_NNN.bin + manifest.json).
//
// Processes one expert-index at a time (streaming), never holding all
// packed buffers simultaneously. Resumable: skips expert files that
// already have the correct size on disk.

struct PrepackResult {
    int experts_written = 0;      // newly written expert files
    int experts_skipped = 0;      // skipped (already correct size)
    int64_t bytes_written = 0;    // total bytes newly written to disk
    std::string error;            // non-empty on failure
};

/// Pre-process all routed experts from a loaded model into the target directory.
///
/// Writes one expert_NNN.bin file per expert index (each containing all MoE
/// layers' slot data) plus a manifest.json.  Processes experts one at a time
/// so peak memory is O(1 slot).  Resumable: skips files that already exist
/// with the expected size.
///
/// @param model       LoadedModel with raw safetensors data (mmap'd shards).
///                    Non-const: packing mutates bundles[0].owned_buf.
/// @param model_cfg   Model config (MoE layer indices, expert count, dimensions).
/// @param quant       Quant interface for the weight format (NVFP4 or FP8).
/// @param cfg         Full config (for source_model_path).
/// @param output_dir  Target directory for expert_NNN.bin + manifest.json.
///                    Created if it does not exist.
/// @return            PrepackResult with counts and any error message.
PrepackResult prepack_experts(
    LoadedModel& model,
    const ModelConfig& model_cfg,
    const QuantInterface& quant,
    const config::Config& cfg,
    const std::filesystem::path& output_dir);

}  // namespace layerstorm::model
