// Offline expert pre-processing tool.
//
// Converts raw safetensors routed expert weights into the pre-packed on-disk
// format (expert_NNN.bin + manifest.json) for fast mmap-based loading.
//
// Usage: prepack_experts <config.json> <output_dir>
//
// The config.json must contain model.weights_path pointing to the safetensors
// model directory. All other config fields are used for model metadata only.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "daemon/engine.h"
#include "model/weight_pipeline/expert_prepacker.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <config.json> <output_dir>\n";
        return 1;
    }

    const std::string config_path = argv[1];
    const std::filesystem::path output_dir = argv[2];

    if (!std::filesystem::exists(config_path)) {
        std::cerr << "Config not found: " << config_path << "\n";
        return 1;
    }

    auto result = layerstorm::daemon::Engine::run_prepack(config_path, output_dir);

    if (!result.error.empty()) {
        std::cerr << "FAILED: " << result.error << "\n";
        return 1;
    }

    std::cout << "Done: " << result.experts_written << " written, "
              << result.experts_skipped << " skipped\n";
    return 0;
}
