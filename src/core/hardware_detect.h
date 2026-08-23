#pragma once

#include <cstdint>
#include <string>

#include "config/config_parser.h"

namespace layerstorm::core {

/// PCIe link details read from sysfs.
struct PcieDetails {
    int gen_current = 0;
    int width_current = 0;
    int gen_max = 0;
    int width_max = 0;
};

/// Lowercase a PCI bus ID for sysfs path compatibility.
/// CUDA returns uppercase hex (e.g. "0000:6A:00.0"); sysfs uses lowercase.
std::string sysfs_pci_id(const std::string& pci_bus_id);

/// Read NUMA node from /sys/bus/pci/devices/<pci_bus_id>/numa_node.
/// Returns -1 if the file is missing or unreadable.
int read_numa_node(const std::string& pci_bus_id);

/// Map sysfs link_speed string (e.g. "32.0 GT/s PCIe") to PCIe generation.
/// Returns 0 if not recognised.
int parse_pcie_gen(const std::string& s);

// kickstart_pcie_link() moved to cuda_hardware_query.h (#86e).

/// Read current and max PCIe gen and width from sysfs.
/// Call kickstart_pcie_link() first to get meaningful current_link_speed.
PcieDetails read_pcie_info(const std::string& pci_bus_id);

/// Read total system RAM from /proc/meminfo. Returns 0 if unreadable.
int64_t read_system_ram();

/// Infer GPU type from CUDA device name string.
config::GpuType detect_gpu_type(const std::string& name);

}  // namespace layerstorm::core
