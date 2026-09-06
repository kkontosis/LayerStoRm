// Ordinal-checked bridge: model::GgufKQuantType -> compute::GgufQuantType.
//
// The two enums are independently declared (model/quantization/gguf_kquant.h
// vs core/expert_device.h) but share ONE canonical value order
// {Q2_K=0, Q3_K=1, Q4_K=2, Q5_K=3, Q6_K=4, Q8_0=5, MXFP4=6}, and the
// engine/kernel boundary maps them by value. TD-GGUF-ENUM-MXFP4-DIVERGENCE:
// MXFP4 was added to the model enum (V4 QAT routed experts) and not to the
// compute enum, so raw ordinal casts silently carried a value the compute enum
// could not name. This header is the ONLY sanctioned conversion: every
// enumerator and the enum length are pinned by static_asserts, so any future
// divergence (added/reordered/removed enumerator) is a COMPILE ERROR here —
// not a silent mis-mapping at a cast site. Do not static_cast between the two
// enums anywhere else. The third leg of the contract (engine enum -> kernel
// compute::GgufType, which lives in deps/LayerStoRmGemmKernels and needs CUDA
// headers) is pinned the same way inside the CUDA device TUs
// (cuda_sm120_expert_device.cu / cuda_sm120_device_backend.cu) so this header
// stays CUDA-free (INV-GPU-1).

#pragma once

#include "core/expert_device.h"
#include "model/quantization/gguf_kquant.h"

namespace layerstorm::model::gguf {

#define LS_GGUF_ENUM_PIN(name)                                          \
    static_assert(static_cast<int>(compute::GgufQuantType::name) ==     \
                      static_cast<int>(GgufKQuantType::name),           \
                  "compute::GgufQuantType::" #name                      \
                  " diverged from model::GgufKQuantType::" #name)
LS_GGUF_ENUM_PIN(Q2_K);
LS_GGUF_ENUM_PIN(Q3_K);
LS_GGUF_ENUM_PIN(Q4_K);
LS_GGUF_ENUM_PIN(Q5_K);
LS_GGUF_ENUM_PIN(Q6_K);
LS_GGUF_ENUM_PIN(Q8_0);
LS_GGUF_ENUM_PIN(MXFP4);
#undef LS_GGUF_ENUM_PIN

static_assert(compute::kNumGgufQuantTypes == kNumGgufKQuantTypes,
              "compute::GgufQuantType and model::GgufKQuantType stopped being "
              "the same length — add the new enumerator to BOTH enums (and to "
              "every switch over compute::GgufQuantType) before casting "
              "(TD-GGUF-ENUM-MXFP4-DIVERGENCE)");

/// The one sanctioned model -> engine GGUF quant-type conversion. Value-
/// preserving by the static_asserts above.
constexpr compute::GgufQuantType to_compute_gguf(GgufKQuantType t) {
    return static_cast<compute::GgufQuantType>(static_cast<int>(t));
}

}  // namespace layerstorm::model::gguf
