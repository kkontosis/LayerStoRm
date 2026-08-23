#include "model/weight_loader/gguf_weight_handler.h"

#include <format>

#include "model/model_config.h"

namespace layerstorm::model {

std::vector<TensorRole> GgufWeightHandler::expected_aux_roles() const {
    // K-quant scales/mins are packed in-block: no separate scale tensors.
    return {};
}

std::string GgufWeightHandler::validate(const WeightBundle& bundle,
                                        const ModelConfig& /*model_cfg*/) const {
    if (!bundle.weight.gguf_type.has_value()) {
        return "GGUF weight handler invoked on a bundle with no gguf_type set";
    }
    if (!bundle.aux.empty()) {
        return std::format("GGUF weight should have no aux tensors, got {}",
                           bundle.aux.size());
    }
    return {};  // valid
}

}  // namespace layerstorm::model
