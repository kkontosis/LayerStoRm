#pragma once

#include <cstdint>
#include <vector>

#include "config_parser.h"

namespace layerstorm::config {

/// Fill sentinel-valued fields in cfg.hardware from real hardware.
///
/// For each gpu: if vram_gb == 0.0 → fill from CUDA hardware query
///               if pcie_gen == 0  → fill from sysfs current_link_speed (with PCIe kickstart)
///               if pcie_width == 0 → fill from sysfs current_link_width
///               if numa_node == -1 → fill from sysfs
///               always fills pcie_gen_max, pcie_width_max, pci_bus_id
/// If system_ram_gb == 0 → fill from /proc/meminfo
/// If tp_array empty → auto-detect via scoring
///
/// Throws std::runtime_error if no CUDA devices are available.
void resolve_config(Config& cfg);

// ── Hardware query helpers (free functions replacing GpuTopology methods) ──

/// True if tp_array is non-empty.
inline bool has_tp_array(const HardwareConfig& hw) { return !hw.tp_array.empty(); }

/// Number of GPUs in the TP group (0 if none).
inline int tp_degree(const HardwareConfig& hw) { return static_cast<int>(hw.tp_array.size()); }

/// Number of GPUs.
inline int gpu_count(const HardwareConfig& hw) { return static_cast<int>(hw.gpus.size()); }

/// Sum of vram_gb across all GPUs, in bytes.
int64_t total_vram_bytes(const HardwareConfig& hw);

/// Indices of GPUs matching the given type, in order.
std::vector<int> indices_by_type(const HardwareConfig& hw, GpuType type);

/// Convert vram_gb (double GiB) to bytes.
inline int64_t vram_gb_to_bytes(double gb) {
    return static_cast<int64_t>(gb * 1024.0 * 1024.0 * 1024.0);
}

/// Fill sentinel tp_mode_per_layer fields (0) from parallelism.tensor_parallelism.
/// Called by resolve_config(); also available for unit testing without CUDA.
inline void fill_tp_mode_per_layer(Config& cfg) {
    int tp = cfg.parallelism.tensor_parallelism;
    auto& tm = cfg.memory.tp_mode_per_layer;
    if (tm.default_mode == 0)     tm.default_mode     = tp;
    if (tm.gating == 0)           tm.gating           = tp;
    if (tm.pinned_dense_ffn == 0) tm.pinned_dense_ffn = tp;
    if (tm.attention == 0)        tm.attention        = tp;
    if (tm.shared_expert == 0)    tm.shared_expert    = tp;
    if (tm.embedding == 0)        tm.embedding        = tp;
    if (tm.output_head == 0)      tm.output_head      = tp;
}

}  // namespace layerstorm::config
