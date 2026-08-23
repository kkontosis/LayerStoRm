// fp8_e4m3_decode.h — minimal, self-contained declaration of the single FP8
// E4M3 decode entry point that the CPU NVFP4 kernel needs.
//
// The CPU NVFP4 grouped-GEMM kernel dequantizes per-group scale bytes via
// layerstorm::model::fp8_e4m3::decode(uint8_t). That decode is a pure,
// arch-independent table/bit-twiddle function whose DEFINITION lives in the
// engine (src/model/quantization/fp8.cpp). This header declares ONLY that one
// symbol so this dependency stays self-contained: it does NOT pull in the
// engine's quant_interface.h / ExpertShape / QuantInterface machinery.
//
// Symbol/namespace are kept IDENTICAL to the engine's model/quantization/fp8.h
// so the engine's fp8.cpp resolves the reference at final link with no shim.
//
// CPU-only. No CUDA. No engine headers.
#pragma once

#include <cstdint>

namespace layerstorm::model::fp8_e4m3 {

/// Decode a single FP8 E4M3 byte to float.
/// Returns NaN for 0x7F and 0xFF. Subnormals supported.
/// (Definition: engine src/model/quantization/fp8.cpp.)
float decode(uint8_t byte);

}  // namespace layerstorm::model::fp8_e4m3
