#pragma once

#include "model/weight_loader/weight_handler.h"

namespace layerstorm::model {

/// WeightHandler for native (unquantized) tensors: BF16, F16, F32.
/// Used for attention projections excluded from quantization, norms, embedding, etc.
class NativeWeightHandler final : public WeightHandler {
public:
    std::string_view name() const override { return "native"; }
    std::vector<TensorRole> expected_aux_roles() const override;
    std::string validate(const WeightBundle& bundle, const ModelConfig& model_cfg) const override;
};

}  // namespace layerstorm::model
