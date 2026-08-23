#pragma once

#include <string_view>
#include <vector>

#include "model/quantization/quant_interface.h"

namespace layerstorm::model {

/// Register a quantization format implementation.
/// Populates both name and WeightQuant lookup maps.
/// Throws std::runtime_error on null or duplicate name.
void register_format(const QuantInterface* impl);

/// Find a format by name. Returns nullptr if not registered.
const QuantInterface* find_format(std::string_view name);

/// Find a format by WeightQuant enum. Returns nullptr if not registered.
const QuantInterface* find_format(config::WeightQuant wq);

/// Get a format by name. Throws std::runtime_error if not found.
const QuantInterface& get_format(std::string_view name);

/// Get a format by WeightQuant enum. Throws std::runtime_error if not found.
const QuantInterface& get_format(config::WeightQuant wq);

/// List all registered format names.
std::vector<std::string_view> registered_names();

/// Clear registry state (for test isolation).
void clear_registry();

} // namespace layerstorm::model
