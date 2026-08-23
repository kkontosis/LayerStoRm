#include "compute/graphs/graph_registry.h"

#include <spdlog/spdlog.h>
#include <stdexcept>

namespace layerstorm::compute {

const char* graph_type_name(GraphType type) {
    switch (type) {
        case GraphType::kAttentionDecode: return "AttentionDecode";
        case GraphType::kDcpAllreduce:    return "DcpAllreduce";
        default:                          return "Unknown";
    }
}

GraphRegistry::~GraphRegistry() {
    clear();
}

// ── Insert ──────────────────────────────────────────────────────────────────

void GraphRegistry::insert(GraphKey key, GraphEntry entry) {
    if (entries_.count(key)) {
        throw std::runtime_error(
            "GraphRegistry::insert: duplicate key ("
            + std::string(graph_type_name(key.type))
            + ", gpu=" + std::to_string(key.gpu_idx)
            + ", bs=" + std::to_string(key.batch_size) + ")");
    }
    entries_.emplace(key, std::move(entry));
}

// ── Lookup ──────────────────────────────────────────────────────────────────

GraphEntry* GraphRegistry::find(const GraphKey& key) {
    auto it = entries_.find(key);
    return it != entries_.end() ? &it->second : nullptr;
}

const GraphEntry* GraphRegistry::find(const GraphKey& key) const {
    auto it = entries_.find(key);
    return it != entries_.end() ? &it->second : nullptr;
}

GraphEntry& GraphRegistry::get(const GraphKey& key) {
    auto it = entries_.find(key);
    if (it != entries_.end()) return it->second;
    throw std::runtime_error(
        "GraphRegistry::get: key not found ("
        + std::string(graph_type_name(key.type))
        + ", gpu=" + std::to_string(key.gpu_idx)
        + ", bs=" + std::to_string(key.batch_size) + ")");
}

const GraphEntry& GraphRegistry::get(const GraphKey& key) const {
    auto it = entries_.find(key);
    if (it != entries_.end()) return it->second;
    throw std::runtime_error(
        "GraphRegistry::get: key not found ("
        + std::string(graph_type_name(key.type))
        + ", gpu=" + std::to_string(key.gpu_idx)
        + ", bs=" + std::to_string(key.batch_size) + ")");
}

bool GraphRegistry::contains(const GraphKey& key) const {
    return entries_.count(key) > 0;
}

// ── Removal ─────────────────────────────────────────────────────────────────

void GraphRegistry::remove(const GraphKey& key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) return;
    if (it->second.destroy) {
        it->second.destroy(it->second.runner);
    }
    entries_.erase(it);
}

void GraphRegistry::clear() {
    for (auto& [key, entry] : entries_) {
        if (entry.destroy) {
            entry.destroy(entry.runner);
        }
    }
    entries_.clear();
}

// ── Queries ─────────────────────────────────────────────────────────────────

size_t GraphRegistry::size() const {
    return entries_.size();
}

bool GraphRegistry::empty() const {
    return entries_.empty();
}

std::vector<GraphKey> GraphRegistry::keys() const {
    std::vector<GraphKey> result;
    result.reserve(entries_.size());
    for (const auto& [key, _] : entries_) {
        result.push_back(key);
    }
    return result;
}

size_t GraphRegistry::count_by_type(GraphType type) const {
    size_t count = 0;
    for (const auto& [key, _] : entries_) {
        if (key.type == type) ++count;
    }
    return count;
}

std::vector<GraphKey> GraphRegistry::keys_by_type(GraphType type) const {
    std::vector<GraphKey> result;
    for (const auto& [key, _] : entries_) {
        if (key.type == type) result.push_back(key);
    }
    return result;
}

}  // namespace layerstorm::compute
