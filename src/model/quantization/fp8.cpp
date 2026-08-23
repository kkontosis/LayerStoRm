#include "model/quantization/fp8.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "model/quantization/registry.h"

namespace layerstorm::model {

// ── Self-registration ───────────────────────────────────────────────────────

namespace {
const Fp8E4M3 g_e4m3_instance;
[[maybe_unused]] const int g_e4m3_reg = (register_format(&g_e4m3_instance), 0);

const Fp8E5M2 g_e5m2_instance;
[[maybe_unused]] const int g_e5m2_reg = (register_format(&g_e5m2_instance), 0);
} // namespace

// ── Shared helpers ──────────────────────────────────────────────────────────

namespace {

int64_t fp8_weight_bytes(const ExpertShape& shape, Projection proj) {
    switch (proj) {
        case Projection::gate: return shape.gate_params();
        case Projection::up:   return shape.up_params();
        case Projection::down: return shape.down_params();
    }
    return 0;
}

int64_t fp8_blockwise_scale_bytes(const ExpertShape& shape, Projection proj) {
    int64_t N, K;
    switch (proj) {
        case Projection::gate: [[fallthrough]];
        case Projection::up:
            N = shape.intermediate_size;
            K = shape.hidden_size;
            break;
        case Projection::down:
            N = shape.hidden_size;
            K = shape.intermediate_size;
            break;
    }
    int64_t n_blocks = (N + fp8::kBlockScaleTile - 1) / fp8::kBlockScaleTile;
    int64_t k_blocks = (K + fp8::kBlockScaleTile - 1) / fp8::kBlockScaleTile;
    return n_blocks * k_blocks * static_cast<int64_t>(sizeof(float));
}

} // namespace

// ── Fp8E4M3 QuantInterface overrides ────────────────────────────────────────

std::string_view Fp8E4M3::name() const { return "fp8_e4m3"; }

config::WeightQuant Fp8E4M3::weight_quant() const {
    return config::WeightQuant::fp8_e4m3;
}

int64_t Fp8E4M3::bytes_per_projection(const ExpertShape& shape, Projection proj) const {
    return fp8_weight_bytes(shape, proj) + fp8_blockwise_scale_bytes(shape, proj);
}

int64_t Fp8E4M3::weight_bytes_per_projection(const ExpertShape& shape, Projection proj) const {
    return fp8_weight_bytes(shape, proj);
}

int64_t Fp8E4M3::bytes_per_expert(const ExpertShape& shape) const {
    return bytes_per_projection(shape, Projection::gate)
         + bytes_per_projection(shape, Projection::up)
         + bytes_per_projection(shape, Projection::down);
}

double Fp8E4M3::bytes_per_element() const { return fp8::kBytesPerElement; }
double Fp8E4M3::dequant_flops_per_element() const { return fp8::kDequantFlopsPerElement; }
bool Fp8E4M3::tensor_core_eligible() const { return true; }

MemoryLayout Fp8E4M3::memory_layout() const {
    return {fp8::kBitsPerWeight, fp8::kGroupSize, false,
            ScaleDtype::none, fp8::kAlignmentBytes};
}

// ── Fp8E5M2 QuantInterface overrides ────────────────────────────────────────

std::string_view Fp8E5M2::name() const { return "fp8_e5m2"; }

config::WeightQuant Fp8E5M2::weight_quant() const {
    return config::WeightQuant::fp8_e5m2;
}

int64_t Fp8E5M2::bytes_per_projection(const ExpertShape& shape, Projection proj) const {
    return fp8_weight_bytes(shape, proj) + fp8_blockwise_scale_bytes(shape, proj);
}

int64_t Fp8E5M2::weight_bytes_per_projection(const ExpertShape& shape, Projection proj) const {
    return fp8_weight_bytes(shape, proj);
}

int64_t Fp8E5M2::bytes_per_expert(const ExpertShape& shape) const {
    return bytes_per_projection(shape, Projection::gate)
         + bytes_per_projection(shape, Projection::up)
         + bytes_per_projection(shape, Projection::down);
}

double Fp8E5M2::bytes_per_element() const { return fp8::kBytesPerElement; }
double Fp8E5M2::dequant_flops_per_element() const { return fp8::kDequantFlopsPerElement; }
bool Fp8E5M2::tensor_core_eligible() const { return true; }

MemoryLayout Fp8E5M2::memory_layout() const {
    return {fp8::kBitsPerWeight, fp8::kGroupSize, false,
            ScaleDtype::none, fp8::kAlignmentBytes};
}

// ── CPU reference dequantization: E4M3fn ────────────────────────────────────
// FP8 E4M3fn (NVIDIA): [sign:1][exp:4][mant:3], bias=7.
// NaN: exp=15 AND mant=7 only. No infinity. Subnormals supported.

namespace fp8_e4m3 {

float decode(uint8_t byte) {
    bool    sign = (byte >> 7) & 1;
    uint8_t exp  = (byte >> 3) & 0x0F;  // 4 bits
    uint8_t mant = byte & 0x07;         // 3 bits

    // NaN: exponent all-ones AND mantissa all-ones
    if (exp == 0x0F && mant == 0x07) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    float value;
    if (exp == 0) {
        // Subnormal or zero: 2^(1-bias) * (mant / 2^3) = mant * 2^(-9)
        value = std::ldexp(static_cast<float>(mant), -kBias - kMantissaBits + 1);
    } else {
        // Normal: 2^(exp-bias) * (1 + mant/2^3) = (8 + mant) * 2^(exp-10)
        value = std::ldexp(static_cast<float>((1 << kMantissaBits) + mant),
                           exp - kBias - kMantissaBits);
    }

    return sign ? -value : value;
}

void dequantize(const uint8_t* packed, float* out, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; ++i) {
        out[i] = decode(packed[i]);
    }
}

} // namespace fp8_e4m3

// ── CPU reference dequantization: E5M2 ──────────────────────────────────────
// FP8 E5M2 (IEEE-like): [sign:1][exp:5][mant:2], bias=15.
// Infinity: exp=31 AND mant=0. NaN: exp=31 AND mant!=0. Subnormals supported.

namespace fp8_e5m2 {

float decode(uint8_t byte) {
    bool    sign = (byte >> 7) & 1;
    uint8_t exp  = (byte >> 2) & 0x1F;  // 5 bits
    uint8_t mant = byte & 0x03;         // 2 bits

    if (exp == 0x1F) {
        if (mant == 0) {
            return sign ? -std::numeric_limits<float>::infinity()
                        :  std::numeric_limits<float>::infinity();
        }
        return std::numeric_limits<float>::quiet_NaN();
    }

    float value;
    if (exp == 0) {
        // Subnormal or zero: 2^(1-bias) * (mant / 2^2) = mant * 2^(-16)
        value = std::ldexp(static_cast<float>(mant), -kBias - kMantissaBits + 1);
    } else {
        // Normal: 2^(exp-bias) * (1 + mant/2^2) = (4 + mant) * 2^(exp-17)
        value = std::ldexp(static_cast<float>((1 << kMantissaBits) + mant),
                           exp - kBias - kMantissaBits);
    }

    return sign ? -value : value;
}

void dequantize(const uint8_t* packed, float* out, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; ++i) {
        out[i] = decode(packed[i]);
    }
}

} // namespace fp8_e5m2

} // namespace layerstorm::model
