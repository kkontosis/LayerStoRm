#include "model/weight_loader/nvfp4_weight_handler.h"

#include <format>
#include <string>

#include "model/model_config.h"

namespace layerstorm::model {

std::vector<TensorRole> NvFp4WeightHandler::expected_aux_roles() const {
    return {TensorRole::weight_scale, TensorRole::weight_scale_2, TensorRole::input_scale};
}

std::string NvFp4WeightHandler::validate(const WeightBundle& bundle,
                                          const ModelConfig& /*model_cfg*/) const {
    // Main weight must be U8 (packed FP4: 2 values per byte)
    if (bundle.weight.dtype != SafetensorsDtype::U8) {
        return std::format("NVFP4 weight dtype must be U8, got {}",
                           dtype_name(bundle.weight.dtype));
    }

    // Check all expected aux tensors are present
    auto* ws = bundle.find_aux(TensorRole::weight_scale);
    auto* ws2 = bundle.find_aux(TensorRole::weight_scale_2);
    auto* is = bundle.find_aux(TensorRole::input_scale);

    if (!ws) return "NVFP4 missing weight_scale tensor";
    if (!ws2) return "NVFP4 missing weight_scale_2 tensor";
    if (!is) return "NVFP4 missing input_scale tensor";

    // weight_scale must be F8_E4M3
    if (ws->dtype != SafetensorsDtype::F8_E4M3) {
        return std::format("NVFP4 weight_scale dtype must be F8_E4M3, got {}",
                           dtype_name(ws->dtype));
    }

    // weight_scale_2 must be F32 scalar
    if (ws2->dtype != SafetensorsDtype::F32) {
        return std::format("NVFP4 weight_scale_2 dtype must be F32, got {}",
                           dtype_name(ws2->dtype));
    }
    if (!ws2->shape.empty()) {
        return std::format("NVFP4 weight_scale_2 must be scalar, got {} dims",
                           ws2->shape.size());
    }

    // input_scale must be F32 scalar
    if (is->dtype != SafetensorsDtype::F32) {
        return std::format("NVFP4 input_scale dtype must be F32, got {}",
                           dtype_name(is->dtype));
    }
    if (!is->shape.empty()) {
        return std::format("NVFP4 input_scale must be scalar, got {} dims",
                           is->shape.size());
    }

    // Shape validation: weight shape last dim should be half of logical dim
    // and weight_scale last dim should be logical_dim / 16 (group_size = 16).
    // For a 2D weight [M, N_packed]:
    //   N_logical = N_packed * 2
    //   weight_scale shape: [M, N_logical / 16] = [M, N_packed / 8]
    if (bundle.weight.shape.size() == 2 && ws->shape.size() == 2) {
        int64_t m = bundle.weight.shape[0];
        int64_t n_packed = bundle.weight.shape[1];

        if (ws->shape[0] != m) {
            return std::format("NVFP4 weight_scale rows ({}) != weight rows ({})",
                               ws->shape[0], m);
        }

        int64_t expected_scale_cols = n_packed / 8;  // (N_packed * 2) / 16
        if (ws->shape[1] != expected_scale_cols) {
            return std::format("NVFP4 weight_scale cols ({}) != expected ({}) "
                               "for weight cols {} (group_size=16)",
                               ws->shape[1], expected_scale_cols, n_packed);
        }
    }

    return {};  // valid
}

}  // namespace layerstorm::model
