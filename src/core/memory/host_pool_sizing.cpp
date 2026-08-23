// Host expert pool sizing resolver — see host_pool_sizing.h.

#include "core/memory/host_pool_sizing.h"

#include <stdexcept>
#include <string>

#include "core/memory/numa_manager.h"

namespace layerstorm::memory {

namespace {

size_t budget_for(config::HostPoolSizingMode mode, double value,
                  const NumaManager& numa, int node) {
    switch (mode) {
        case config::HostPoolSizingMode::fraction_free:
            if (!(value > 0.0 && value <= 1.0))
                throw std::runtime_error(
                    "pin_host_expert_pool_sizing fraction_free value must be in "
                    "(0,1], got " + std::to_string(value) + " (node " +
                    std::to_string(node) + ")");
            return static_cast<size_t>(
                static_cast<double>(numa.node_available_bytes(node)) * value);
        case config::HostPoolSizingMode::fraction_total:
            if (!(value > 0.0 && value <= 1.0))
                throw std::runtime_error(
                    "pin_host_expert_pool_sizing fraction_total value must be in "
                    "(0,1], got " + std::to_string(value) + " (node " +
                    std::to_string(node) + ")");
            return static_cast<size_t>(
                static_cast<double>(numa.node_total_bytes(node)) * value);
        case config::HostPoolSizingMode::absolute_gb:
            if (!(value > 0.0))
                throw std::runtime_error(
                    "pin_host_expert_pool_sizing absolute_gb value must be > 0, "
                    "got " + std::to_string(value) + " (node " +
                    std::to_string(node) + ")");
            return static_cast<size_t>(value * 1073741824.0);  // GiB → bytes
    }
    return 0;  // unreachable (all enum cases handled)
}

}  // namespace

size_t resolve_node_budget_bytes(
        const config::PinHostExpertPoolSizingConfig& sizing,
        const NumaManager& numa, int node) {
    // Per-node override wins over the global setting.
    for (const auto& pn : sizing.per_node) {
        if (pn.node == node)
            return budget_for(pn.mode, pn.value, numa, node);
    }
    return budget_for(sizing.mode, sizing.value, numa, node);
}

}  // namespace layerstorm::memory
