#pragma once

#include "model/weight_loader/weight_handler.h"

namespace layerstorm::model {

/// WeightHandler for GGUF k-quant tensors. GGUF super-blocks carry their scales
/// and mins INSIDE each block, so — like the native handler — there are NO
/// auxiliary scale tensors. Validation confirms the bundle carries a GGUF
/// k-quant type and no aux tensors. Selected when a bundle's main weight has a
/// gguf_type set (see handler_for_bundle).
class GgufWeightHandler final : public WeightHandler {
public:
    std::string_view name() const override { return "gguf"; }
    std::vector<TensorRole> expected_aux_roles() const override;
    std::string validate(const WeightBundle& bundle, const ModelConfig& model_cfg) const override;
};

}  // namespace layerstorm::model
