#include "config_resolver.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/cuda_hardware_query.h"
#include "core/hardware_detect.h"

namespace layerstorm::config {

// ── Hardware query helpers ────────────────────────────────────────────────

int64_t total_vram_bytes(const HardwareConfig& hw) {
    int64_t total = 0;
    for (const auto& g : hw.gpus) total += vram_gb_to_bytes(g.vram_gb);
    return total;
}

std::vector<int> indices_by_type(const HardwareConfig& hw, GpuType type) {
    std::vector<int> out;
    for (int i = 0; i < static_cast<int>(hw.gpus.size()); ++i) {
        if (hw.gpus[i].type == type) out.push_back(i);
    }
    return out;
}

// ── tp_array auto-detect ──────────────────────────────────────────────────

/// Normalized speed factor by GPU type (SM ratio).
static double normalized_speed(GpuType type) {
    switch (type) {
        case GpuType::rtx5090: return 1.0;       // 170 SMs
        case GpuType::rtx5080: return 0.494;      // 84 SMs
    }
    return 0.0;
}

/// Largest power of 2 <= n.
static int floor_pow2(int n) {
    if (n <= 0) return 0;
    int p = 1;
    while (p * 2 <= n) p *= 2;
    return p;
}

static std::vector<int> auto_detect_tp_array(const HardwareConfig& hw) {
    if (hw.gpus.empty()) return {};

    // Find max VRAM for normalization.
    double max_vram = 0.0;
    for (const auto& g : hw.gpus)
        max_vram = std::max(max_vram, g.vram_gb);
    if (max_vram <= 0.0) return {};

    // Score each GPU: 3 * (vram / max_vram) + 1 * normalized_speed(type)
    struct ScoredGpu {
        int id;
        GpuType type;
        double score;
    };
    std::vector<ScoredGpu> scored;
    scored.reserve(hw.gpus.size());
    for (const auto& g : hw.gpus) {
        double s = 3.0 * (g.vram_gb / max_vram) + 1.0 * normalized_speed(g.type);
        scored.push_back({g.id, g.type, s});
    }

    // Sort by score descending, then by ID ascending for stability.
    std::sort(scored.begin(), scored.end(), [](const ScoredGpu& a, const ScoredGpu& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.id < b.id;
    });

    // Pick largest power-of-2 count of top-scoring same-type GPUs.
    GpuType top_type = scored[0].type;
    int same_type_count = 0;
    for (const auto& sg : scored) {
        if (sg.type == top_type) ++same_type_count;
        else break;
    }

    int tp_count = floor_pow2(same_type_count);
    if (tp_count < 1) return {};

    std::vector<int> result;
    result.reserve(tp_count);
    for (const auto& sg : scored) {
        if (sg.type == top_type) {
            result.push_back(sg.id);
            if (static_cast<int>(result.size()) == tp_count) break;
        }
    }

    // Sort IDs for deterministic ordering.
    std::sort(result.begin(), result.end());
    return result;
}

// ── resolve_config ────────────────────────────────────────────────────────

void resolve_config(Config& cfg) {
    int cuda_count = core::query_gpu_count();  // throws on CUDA failure
    if (cuda_count == 0) {
        throw std::runtime_error("No CUDA devices found (count == 0)");
    }

    // Resolve per-GPU fields.
    for (auto& g : cfg.hardware.gpus) {
        if (g.id < 0 || g.id >= cuda_count) continue;

        auto hw = core::query_gpu_info(g.id);

        // vram_gb: sentinel 0.0 → auto-detect
        if (g.vram_gb <= 0.0) {
            g.vram_gb = hw.vram_gb;
        }

        // type: always detect from hardware name
        g.type = core::detect_gpu_type(hw.device_name);

        // PCI bus ID: always fill (internal field)
        g.pci_bus_id = hw.pci_bus_id;

        // NUMA node: sentinel -1 → auto-detect
        if (g.numa_node < 0) {
            g.numa_node = core::read_numa_node(g.pci_bus_id);
        }

        // PCIe info: sentinel 0 → auto-detect (kickstart first for accurate current speed)
        if (g.pcie_gen == 0 || g.pcie_width == 0) {
            core::kickstart_pcie_link(g.id);
            auto pcie = core::read_pcie_info(g.pci_bus_id);
            if (g.pcie_gen == 0) g.pcie_gen = pcie.gen_current;
            if (g.pcie_width == 0) g.pcie_width = pcie.width_current;
            g.pcie_gen_max = pcie.gen_max;
            g.pcie_width_max = pcie.width_max;
        } else {
            // User specified pcie_gen/width; still read max from hardware.
            auto pcie = core::read_pcie_info(g.pci_bus_id);
            g.pcie_gen_max = pcie.gen_max;
            g.pcie_width_max = pcie.width_max;
        }

        // Warn if link trained below capability.
        if (g.pcie_gen > 0 && g.pcie_gen_max > 0 && g.pcie_gen < g.pcie_gen_max) {
            spdlog::warn(
                "GPU {}: PCIe link trained at Gen {} x{} but device supports Gen {} x{} "
                "— possible riser cable or slot bandwidth limitation",
                g.id, g.pcie_gen, g.pcie_width, g.pcie_gen_max, g.pcie_width_max);
        }
    }

    // system_ram_gb: sentinel 0 → auto-detect
    if (cfg.hardware.system_ram_gb <= 0) {
        cfg.hardware.system_ram_gb = static_cast<int>(core::read_system_ram() >> 30);
    }

    // tp_array: empty → auto-detect via scoring
    if (cfg.hardware.tp_array.empty()) {
        cfg.hardware.tp_array = auto_detect_tp_array(cfg.hardware);
    }

    // DCP is always enabled when tensor_parallelism >= 2 (TP+DCP on same GPUs)
    cfg.hardware.dcp_enabled = (cfg.parallelism.tensor_parallelism >= 2);

    // tp_mode_per_layer: sentinel 0 → inherit from parallelism.tensor_parallelism
    fill_tp_mode_per_layer(cfg);
}

}  // namespace layerstorm::config
