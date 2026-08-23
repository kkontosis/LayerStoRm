#include "daemon/buffer_registry.h"

namespace layerstorm::daemon {

uint32_t BufferRegistry::register_buffer(void* device_ptr, int64_t size_bytes,
                                         int gpu_idx, const char* name) {
    uint32_t id = next_id_++;
    entries_[id] = BufferEntry{device_ptr, size_bytes, gpu_idx};
    if (name) {
        names_[id] = name;
    }
    return id;
}

void BufferRegistry::deregister(uint32_t buf_id) {
    entries_.erase(buf_id);
    names_.erase(buf_id);
}

const BufferEntry* BufferRegistry::lookup(uint32_t buf_id) const {
    auto it = entries_.find(buf_id);
    return it != entries_.end() ? &it->second : nullptr;
}

void* BufferRegistry::resolve(uint32_t buf_id) const {
    auto it = entries_.find(buf_id);
    return it != entries_.end() ? it->second.device_ptr : nullptr;
}

void* BufferRegistry::resolve_checked(uint32_t buf_id,
                                      int64_t required_bytes) const {
    auto it = entries_.find(buf_id);
    if (it == entries_.end()) return nullptr;
    if (it->second.size_bytes < required_bytes) return nullptr;
    return it->second.device_ptr;
}

std::vector<std::pair<uint32_t, std::string>>
BufferRegistry::all_named_entries() const {
    std::vector<std::pair<uint32_t, std::string>> result;
    result.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) {
        auto name_it = names_.find(id);
        result.emplace_back(
            id, name_it != names_.end() ? name_it->second : std::string{});
    }
    return result;
}

void BufferRegistry::clear() {
    entries_.clear();
    names_.clear();
    // Do NOT reset next_id_ — deregistered IDs are never reused.
}

}  // namespace layerstorm::daemon
