#include "model/quantization/registry.h"

#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace layerstorm::model {
namespace {

// Construct-on-first-use to avoid static initialization order fiasco.
// Self-registering formats (nvfp4.cpp, fp8.cpp) call register_format() from
// static initializers, which may run before raw namespace-scope globals.
std::shared_mutex& mutex() {
    static std::shared_mutex m;
    return m;
}
std::unordered_map<std::string_view, const QuantInterface*>& by_name() {
    static std::unordered_map<std::string_view, const QuantInterface*> m;
    return m;
}
std::unordered_map<int, const QuantInterface*>& by_enum() {
    static std::unordered_map<int, const QuantInterface*> m;
    return m;
}

// Caller must hold mutex() (shared or exclusive).
std::string names_list_unlocked() {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (auto& [name, _] : by_name()) {
        if (!first) oss << ", ";
        oss << name;
        first = false;
    }
    oss << "]";
    return oss.str();
}

} // namespace

void register_format(const QuantInterface* impl) {
    if (!impl) {
        throw std::runtime_error("register_format: null QuantInterface");
    }
    std::unique_lock lock(mutex());
    auto n = impl->name();
    if (by_name().count(n)) {
        throw std::runtime_error(
            "register_format: duplicate name '" + std::string(n) + "'");
    }
    by_name()[n] = impl;
    by_enum()[static_cast<int>(impl->weight_quant())] = impl;
}

const QuantInterface* find_format(std::string_view name) {
    std::shared_lock lock(mutex());
    auto it = by_name().find(name);
    return it != by_name().end() ? it->second : nullptr;
}

const QuantInterface* find_format(config::WeightQuant wq) {
    std::shared_lock lock(mutex());
    auto it = by_enum().find(static_cast<int>(wq));
    return it != by_enum().end() ? it->second : nullptr;
}

const QuantInterface& get_format(std::string_view name) {
    std::shared_lock lock(mutex());
    auto it = by_name().find(name);
    if (it != by_name().end()) return *it->second;
    throw std::runtime_error(
        "get_format: unknown format '" + std::string(name)
        + "'; registered: " + names_list_unlocked());
}

const QuantInterface& get_format(config::WeightQuant wq) {
    std::shared_lock lock(mutex());
    auto it = by_enum().find(static_cast<int>(wq));
    if (it != by_enum().end()) return *it->second;
    throw std::runtime_error(
        "get_format: unknown WeightQuant enum "
        + std::to_string(static_cast<int>(wq))
        + "; registered: " + names_list_unlocked());
}

std::vector<std::string_view> registered_names() {
    std::shared_lock lock(mutex());
    std::vector<std::string_view> names;
    names.reserve(by_name().size());
    for (auto& [n, _] : by_name()) {
        names.push_back(n);
    }
    return names;
}

void clear_registry() {
    std::unique_lock lock(mutex());
    by_name().clear();
    by_enum().clear();
}

} // namespace layerstorm::model
