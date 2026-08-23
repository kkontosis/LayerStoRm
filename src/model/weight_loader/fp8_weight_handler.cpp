#include "model/weight_loader/fp8_weight_handler.h"

#include <cstdint>
#include <format>

#include "model/model_config.h"
#include "model/quantization/fp8.h"

namespace layerstorm::model {

std::vector<TensorRole> Fp8WeightHandler::expected_aux_roles() const {
    return {TensorRole::weight_scale};
}

std::string Fp8WeightHandler::validate(const WeightBundle& bundle,
                                        const ModelConfig& /*model_cfg*/) const {
    if (bundle.weight.dtype != SafetensorsDtype::F8_E4M3 &&
        bundle.weight.dtype != SafetensorsDtype::F8_E5M2) {
        return std::format("FP8 weight dtype must be F8_E4M3 or F8_E5M2, got {}",
                           dtype_name(bundle.weight.dtype));
    }

    // Check for unexpected aux roles (only weight_scale is allowed).
    for (const auto& [role, _] : bundle.aux) {
        if (role != TensorRole::weight_scale) {
            return std::format("FP8 weight has unexpected aux tensor role '{}'",
                               tensor_role_name(role));
        }
    }

    // Validate optional blockwise scale tensor if present.
    const auto* ws = bundle.find_aux(TensorRole::weight_scale);
    if (ws) {
        if (ws->dtype != SafetensorsDtype::F32) {
            return std::format("FP8 weight_scale dtype must be F32, got {}",
                               dtype_name(ws->dtype));
        }

        // Shape must be [ceil(N/128), ceil(K/128)] for 2D weights [N, K].
        if (bundle.weight.shape.size() == 2 && ws->shape.size() == 2) {
            int64_t N = bundle.weight.shape[0];
            int64_t K = bundle.weight.shape[1];
            int64_t expected_n = (N + fp8::kBlockScaleTile - 1) / fp8::kBlockScaleTile;
            int64_t expected_k = (K + fp8::kBlockScaleTile - 1) / fp8::kBlockScaleTile;

            if (ws->shape[0] != expected_n || ws->shape[1] != expected_k) {
                return std::format(
                    "FP8 weight_scale shape [{}, {}] != expected [{}, {}] "
                    "for weight shape [{}, {}] (tile_size={})",
                    ws->shape[0], ws->shape[1], expected_n, expected_k,
                    N, K, fp8::kBlockScaleTile);
            }
        }
    }

    return {};  // valid
}

}  // namespace layerstorm::model
