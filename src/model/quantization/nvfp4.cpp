#include "model/quantization/nvfp4.h"

#include <bit>
#include <cstdint>

#include "model/quantization/registry.h"

namespace layerstorm::model {

// ── Self-registration ───────────────────────────────────────────────────────

namespace {
const Nvfp4 g_instance;
[[maybe_unused]] const int g_reg = (register_format(&g_instance), 0);
} // namespace

// ── QuantInterface overrides ────────────────────────────────────────────────

std::string_view Nvfp4::name() const { return "nvfp4"; }

config::WeightQuant Nvfp4::weight_quant() const {
    return config::WeightQuant::nvfp4;
}

int64_t Nvfp4::bytes_per_projection(const ExpertShape& shape, Projection proj) const {
    int64_t params = 0;
    switch (proj) {
        case Projection::gate: params = shape.gate_params(); break;
        case Projection::up:   params = shape.up_params();   break;
        case Projection::down: params = shape.down_params(); break;
    }
    int64_t weight_bytes = (params + 1) / 2;   // ceil: 2 FP4 values per byte
    int64_t scale_bytes  = (params + 15) / 16; // ceil: 1 UE8M0 byte per 16 elements
    // +8: weight_scale_2 (F32) + input_scale (F32) per projection.
    // Align total to 128 bytes for CUTLASS TMA B-pointer alignment on SM120.
    // Layout: [FP4_weight | group_scale | PADDING | weight_scale_2 | input_scale]
    // Scalars occupy the last 8 bytes so launch_gather_alphas reads at (proj_bytes - 8).
    constexpr int64_t kAlign = 128;
    int64_t raw = weight_bytes + scale_bytes + 2 * static_cast<int64_t>(sizeof(float));
    return (raw + kAlign - 1) & ~(kAlign - 1);
}

int64_t Nvfp4::bytes_per_expert(const ExpertShape& shape) const {
    return bytes_per_projection(shape, Projection::gate)
         + bytes_per_projection(shape, Projection::up)
         + bytes_per_projection(shape, Projection::down);
}

double Nvfp4::bytes_per_element() const {
    return nvfp4::kBytesPerElement;
}

double Nvfp4::dequant_flops_per_element() const {
    return nvfp4::kDequantFlopsPerElement;
}

bool Nvfp4::tensor_core_eligible() const { return true; }

int64_t Nvfp4::weight_bytes_per_projection(const ExpertShape& shape,
                                            Projection proj) const {
    int64_t params = 0;
    switch (proj) {
        case Projection::gate: params = shape.gate_params(); break;
        case Projection::up:   params = shape.up_params();   break;
        case Projection::down: params = shape.down_params(); break;
    }
    return (params + 1) / 2;  // ceil: 2 FP4 values per byte
}

int64_t Nvfp4::replicated_bytes_per_projection() const {
    return 2 * static_cast<int64_t>(sizeof(float));  // weight_scale_2 + input_scale
}

MemoryLayout Nvfp4::memory_layout() const {
    return {nvfp4::kBitsPerWeight, nvfp4::kGroupSize, false,
            ScaleDtype::ue8m0, nvfp4::kAlignmentBytes};
}

// ── CPU reference dequantization ────────────────────────────────────────────

namespace nvfp4 {

float decode_e2m1(uint8_t nibble) {
    uint8_t mag  = nibble & 0x07;        // bits [2:0] = magnitude index
    bool    sign = (nibble & 0x08) != 0; // bit  [3]   = sign
    float val = kE2M1Magnitudes[mag];
    return sign ? -val : val;
}

float decode_ue8m0(uint8_t scale_byte) {
    // UE8M0: unsigned 8-bit exponent, 0 mantissa bits.
    // Encodes 2^(byte - 127). Reconstruct by placing the 8 exponent bits
    // into the IEEE 754 float32 exponent field (bits [30:23]).
    // byte=0 → float bits 0x00000000 = 0.0f
    // byte=127 → 0x3F800000 = 1.0f
    // byte=128 → 0x40000000 = 2.0f
    uint32_t bits = static_cast<uint32_t>(scale_byte) << 23;
    return std::bit_cast<float>(bits);
}

void dequantize(const uint8_t* packed, const uint8_t* scales,
                float* out, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; ++i) {
        int64_t byte_idx = i / 2;
        uint8_t nibble = (i % 2 == 0)
            ? (packed[byte_idx] & 0x0F)
            : ((packed[byte_idx] >> 4) & 0x0F);

        float scale = decode_ue8m0(scales[i / kGroupSize]);
        out[i] = decode_e2m1(nibble) * scale;
    }
}

} // namespace nvfp4

} // namespace layerstorm::model
