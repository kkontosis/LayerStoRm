#include "model/weight_loader/native_weight_handler.h"

#include <format>

#include "model/model_config.h"

namespace layerstorm::model {

std::vector<TensorRole> NativeWeightHandler::expected_aux_roles() const {
    return {};
}

std::string NativeWeightHandler::validate(const WeightBundle& bundle,
                                           const ModelConfig& /*model_cfg*/) const {
    switch (bundle.weight.dtype) {
        case SafetensorsDtype::BF16:
        case SafetensorsDtype::F16:
        case SafetensorsDtype::F32:
            break;  // ok
        default:
            return std::format("Native weight dtype must be BF16/F16/F32, got {}",
                               dtype_name(bundle.weight.dtype));
    }

    if (!bundle.aux.empty()) {
        return std::format("Native weight should have no aux tensors, got {}",
                           bundle.aux.size());
    }

    return {};  // valid
}

}  // namespace layerstorm::model
