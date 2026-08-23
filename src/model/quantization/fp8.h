#pragma once

#include <cstdint>
#include <string_view>

#include "model/quantization/quant_interface.h"

namespace layerstorm::model {

// ── FP8 shared constants ────────────────────────────────────────────────────
// Both E4M3 and E5M2 are 8 bits per weight with no per-group scale overhead.

namespace fp8 {

inline constexpr int kBitsPerWeight = 8;
inline constexpr int kGroupSize = 0;              // no per-group scales
inline constexpr int kBlockScaleTile = 128;        // SM120 FP8 blockwise scale tile dim
inline constexpr int kAlignmentBytes = 64;         // DMA/tensor core alignment
inline constexpr double kBytesPerElement = 1.0 + 4.0 / (128.0 * 128.0);  // weight + amortized blockwise scale
inline constexpr double kDequantFlopsPerElement = 1.0;  // single FP8->FP32 conversion

} // namespace fp8

// ── FP8 E4M3fn (NVIDIA flavor) ─────────────────────────────────────────────
// Bit layout: [sign:1][exp:4][mant:3]. Bias = 7.
// NaN: exp=15, mant=7 only (0x7F/0xFF). No infinity.
// Max finite: ±448.0 (exp=15, mant=6).

namespace fp8_e4m3 {

inline constexpr int kExponentBits = 4;
inline constexpr int kMantissaBits = 3;
inline constexpr int kBias = 7;
inline constexpr float kMaxFinite = 448.0f;

/// Decode a single FP8 E4M3 byte to float.
/// Returns NaN for 0x7F and 0xFF. Subnormals supported.
float decode(uint8_t byte);

/// Dequantize packed E4M3 weights to float (no scales).
/// @param packed  num_elements bytes
/// @param out     num_elements float output
void dequantize(const uint8_t* packed, float* out, int64_t num_elements);

} // namespace fp8_e4m3

// ── FP8 E5M2 (IEEE-like) ───────────────────────────────────────────────────
// Bit layout: [sign:1][exp:5][mant:2]. Bias = 15.
// Infinity: exp=31, mant=0 (0x7C/0xFC).
// NaN: exp=31, mant!=0 (0x7D-0x7F, 0xFD-0xFF).
// Max finite: ±57344.0 (exp=30, mant=3).

namespace fp8_e5m2 {

inline constexpr int kExponentBits = 5;
inline constexpr int kMantissaBits = 2;
inline constexpr int kBias = 15;
inline constexpr float kMaxFinite = 57344.0f;

/// Decode a single FP8 E5M2 byte to float.
/// Returns ±inf for 0x7C/0xFC, NaN for 0x7D-0x7F/0xFD-0xFF.
float decode(uint8_t byte);

/// Dequantize packed E5M2 weights to float (no scales).
void dequantize(const uint8_t* packed, float* out, int64_t num_elements);

} // namespace fp8_e5m2

// ── Fp8E4M3 QuantInterface implementation ───────────────────────────────────

class Fp8E4M3 final : public QuantInterface {
public:
    std::string_view name() const override;
    config::WeightQuant weight_quant() const override;
    int64_t bytes_per_expert(const ExpertShape& shape) const override;
    int64_t bytes_per_projection(const ExpertShape& shape, Projection proj) const override;
    int64_t weight_bytes_per_projection(const ExpertShape& shape,
                                        Projection proj) const override;
    double bytes_per_element() const override;
    double dequant_flops_per_element() const override;
    bool tensor_core_eligible() const override;
    MemoryLayout memory_layout() const override;
};

// ── Fp8E5M2 QuantInterface implementation ───────────────────────────────────

class Fp8E5M2 final : public QuantInterface {
public:
    std::string_view name() const override;
    config::WeightQuant weight_quant() const override;
    int64_t bytes_per_expert(const ExpertShape& shape) const override;
    int64_t bytes_per_projection(const ExpertShape& shape, Projection proj) const override;
    int64_t weight_bytes_per_projection(const ExpertShape& shape,
                                        Projection proj) const override;
    double bytes_per_element() const override;
    double dequant_flops_per_element() const override;
    bool tensor_core_eligible() const override;
    MemoryLayout memory_layout() const override;
};

} // namespace layerstorm::model
