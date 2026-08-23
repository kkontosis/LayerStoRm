// CPU-side F32→BF16 conversion utility.
// Used during weight upload to convert checkpoint F32 norm weights to BF16.

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace layerstorm {

/// Convert F32 buffer to BF16 (round-to-nearest-even). Returns BF16 data.
inline std::vector<uint16_t> f32_to_bf16(const void* src, size_t num_elements) {
    std::vector<uint16_t> out(num_elements);
    const auto* f = static_cast<const uint8_t*>(src);
    for (size_t i = 0; i < num_elements; ++i) {
        uint32_t bits;
        std::memcpy(&bits, f + i * 4, 4);
        bits += 0x7FFF + ((bits >> 16) & 1);
        out[i] = static_cast<uint16_t>(bits >> 16);
    }
    return out;
}

}  // namespace layerstorm
