#include "model/weight_loader/weight_handler.h"

#include "model/weight_loader/fp8_weight_handler.h"
#include "model/weight_loader/gguf_weight_handler.h"
#include "model/weight_loader/native_weight_handler.h"
#include "model/weight_loader/nvfp4_weight_handler.h"

namespace layerstorm::model {

// ── RawTensor ────────────────────────────────────────────────────────────────

int64_t RawTensor::numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// ── WeightBundle ─────────────────────────────────────────────────────────────

int64_t WeightBundle::total_bytes() const {
    int64_t total = static_cast<int64_t>(weight.data.size());
    for (auto& [role, tensor] : aux) {
        total += static_cast<int64_t>(tensor.data.size());
    }
    return total;
}

const RawTensor* WeightBundle::find_aux(TensorRole role) const {
    for (auto& [r, tensor] : aux) {
        if (r == role) return &tensor;
    }
    return nullptr;
}

// ── handler_for_dtype ────────────────────────────────────────────────────────

const WeightHandler& handler_for_dtype(SafetensorsDtype dtype) {
    static const NvFp4WeightHandler nvfp4;
    static const Fp8WeightHandler fp8;
    static const NativeWeightHandler native;

    switch (dtype) {
        case SafetensorsDtype::U8:
            return nvfp4;
        case SafetensorsDtype::F8_E4M3:
        case SafetensorsDtype::F8_E5M2:
            return fp8;
        case SafetensorsDtype::BF16:
        case SafetensorsDtype::F16:
        case SafetensorsDtype::F32:
            return native;
        default:
            return native;  // fallback
    }
}

const WeightHandler& handler_for_bundle(const WeightBundle& bundle) {
    static const GgufWeightHandler gguf;
    // GGUF k-quants are packed bytes (dtype U8) but must NOT use the NVFP4
    // handler — distinguish by the gguf_type marker the GGUF reader sets.
    if (bundle.weight.gguf_type.has_value()) {
        return gguf;
    }
    return handler_for_dtype(bundle.weight.dtype);
}

}  // namespace layerstorm::model
