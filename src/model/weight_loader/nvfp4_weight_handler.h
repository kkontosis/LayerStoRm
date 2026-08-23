#pragma once

#include "model/weight_loader/weight_handler.h"

namespace layerstorm::model {

/// WeightHandler for NVFP4-quantized tensors.
/// Expects: weight(U8 packed), weight_scale(F8_E4M3), weight_scale_2(F32), input_scale(F32).
/// Weight shape has last dim halved (2 FP4 values per byte).
/// Scale shape has last dim = original / group_size (group_size = 16).
class NvFp4WeightHandler final : public WeightHandler {
public:
    std::string_view name() const override { return "nvfp4"; }
    std::vector<TensorRole> expected_aux_roles() const override;
    std::string validate(const WeightBundle& bundle, const ModelConfig& model_cfg) const override;
};

}  // namespace layerstorm::model
