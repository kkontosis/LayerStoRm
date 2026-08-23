#pragma once

#include "model/weight_loader/weight_handler.h"

namespace layerstorm::model {

/// WeightHandler for FP8-quantized tensors (E4M3 or E5M2).
/// No auxiliary scale tensors — just the weight.
class Fp8WeightHandler final : public WeightHandler {
public:
    std::string_view name() const override { return "fp8"; }
    std::vector<TensorRole> expected_aux_roles() const override;
    std::string validate(const WeightBundle& bundle, const ModelConfig& model_cfg) const override;
};

}  // namespace layerstorm::model
