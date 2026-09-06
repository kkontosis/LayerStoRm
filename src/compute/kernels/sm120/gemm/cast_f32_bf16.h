// FP32 -> BF16 elementwise round (TD-GLM5-TP-COMBINE-PRECISION).
//
// The TP fp32 partial-combine path stores o_proj partials in FP32, allreduces
// them in FP32 (kCollFloat32 sum), and then rounds the summed hidden to BF16
// exactly ONCE — this kernel is that single rounding. dst[i] =
// __float2bfloat16_rn(src[i]); deterministic, graph-capturable, no host
// allocation.

#pragma once

#include <cstdint>

namespace layerstorm::compute {

// dst: [count] BF16, src: [count] FP32. Launches on `stream`.
void launch_cast_f32_to_bf16(void* dst_bf16, const void* src_f32,
                             int64_t count, void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
