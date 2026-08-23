#pragma once

#include <cstdint>
#include <string_view>

#include "model/quantization/quant_interface.h"

namespace layerstorm::model {

// ── NVFP4 format constants ──────────────────────────────────────────────────
// FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit.
// Block-scaled with UE8M0 scale factors, one per 16 elements.

namespace nvfp4 {

inline constexpr int kBitsPerWeight = 4;
inline constexpr int kGroupSize = 16;       // elements per UE8M0 scale factor
inline constexpr int kAlignmentBytes = 32;  // 64 FP4 elements (CUTLASS AlignmentA)
inline constexpr double kBytesPerElement = 0.5 + 1.0 / 16.0;  // 0.5625
inline constexpr double kDequantFlopsPerElement = 2.0;         // lookup + multiply

// E2M1 magnitude table: indexed by 3-bit unsigned magnitude (bits [2:0]).
// Bit 3 is the sign bit (0 = positive, 1 = negative).
inline constexpr float kE2M1Magnitudes[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};
inline constexpr float kE2M1Max = 6.0f;

// ── CPU reference dequantization ────────────────────────────────────────────
// For test validation and weight loading. GPU dequant is fused into CUTLASS
// GEMMs (PLAN #31).

/// Decode a single 4-bit E2M1 nibble to float.
/// @param nibble Raw 4-bit value [0..15]. Bits: [sign:1][exp:2][mant:1].
float decode_e2m1(uint8_t nibble);

/// Interpret a UE8M0 scale byte as float. UE8M0 = 2^(byte - 127).
/// byte=0 → 0.0f, byte=127 → 1.0f, byte=128 → 2.0f.
float decode_ue8m0(uint8_t scale_byte);

/// Dequantize packed FP4 weights with per-group UE8M0 scales.
/// @param packed  ceil(num_elements/2) bytes, 2 FP4 per byte (low nibble first)
/// @param scales  ceil(num_elements/16) UE8M0 bytes, one per group
/// @param out     num_elements float output
/// @param num_elements  Total element count
void dequantize(const uint8_t* packed, const uint8_t* scales,
                float* out, int64_t num_elements);

} // namespace nvfp4

// ── Nvfp4 QuantInterface implementation ─────────────────────────────────────

class Nvfp4 final : public QuantInterface {
public:
    std::string_view name() const override;
    config::WeightQuant weight_quant() const override;
    int64_t bytes_per_expert(const ExpertShape& shape) const override;
    int64_t bytes_per_projection(const ExpertShape& shape, Projection proj) const override;
    double bytes_per_element() const override;
    double dequant_flops_per_element() const override;
    bool tensor_core_eligible() const override;
    MemoryLayout memory_layout() const override;
    int64_t weight_bytes_per_projection(const ExpertShape& shape,
                                        Projection proj) const override;
    int64_t replicated_bytes_per_projection() const override;
};

} // namespace layerstorm::model
