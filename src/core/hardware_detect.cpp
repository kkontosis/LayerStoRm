#include "core/hardware_detect.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace layerstorm::core {

std::string sysfs_pci_id(const std::string& pci_bus_id) {
    std::string s = pci_bus_id;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

int read_numa_node(const std::string& pci_bus_id) {
    std::ifstream f("/sys/bus/pci/devices/" + sysfs_pci_id(pci_bus_id) + "/numa_node");
    if (!f) return -1;
    int node = -1;
    f >> node;
    return node;
}

int parse_pcie_gen(const std::string& s) {
    if (s.find("64") != std::string::npos) return 6;
    if (s.find("32") != std::string::npos) return 5;
    if (s.find("16") != std::string::npos) return 4;
    if (s.find("8.0") != std::string::npos || s.find("8 ") != std::string::npos) return 3;
    if (s.find("2.5") != std::string::npos) return 1;
    if (s.find("5.0") != std::string::npos || s.find("5 ") != std::string::npos) return 2;
    return 0;
}

PcieDetails read_pcie_info(const std::string& pci_bus_id) {
    PcieDetails d;
    const std::string id = sysfs_pci_id(pci_bus_id);

    if (std::ifstream sf("/sys/bus/pci/devices/" + id + "/current_link_speed"); sf) {
        std::string line;
        std::getline(sf, line);
        d.gen_current = parse_pcie_gen(line);
    }
    if (std::ifstream wf("/sys/bus/pci/devices/" + id + "/current_link_width"); wf) {
        wf >> d.width_current;
    }
    if (std::ifstream sf("/sys/bus/pci/devices/" + id + "/max_link_speed"); sf) {
        std::string line;
        std::getline(sf, line);
        d.gen_max = parse_pcie_gen(line);
    }
    if (std::ifstream wf("/sys/bus/pci/devices/" + id + "/max_link_width"); wf) {
        wf >> d.width_max;
    }

    return d;
}

int64_t read_system_ram() {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("MemTotal:", 0) == 0) {
            std::istringstream iss(line.substr(9));
            int64_t kb = 0;
            iss >> kb;
            return kb * 1024;
        }
    }
    return 0;
}

config::GpuType detect_gpu_type(const std::string& name) {
    if (name.find("5090") != std::string::npos) return config::GpuType::rtx5090;
    if (name.find("5080") != std::string::npos) return config::GpuType::rtx5080;
    return config::GpuType::rtx5080;
}

}  // namespace layerstorm::core
