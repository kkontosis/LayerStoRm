#pragma once
// PART-1 multi-arch NVFP4 probe: three arch tiers (AVX-512 / AVX2 / scalar) of
// the NVFP4 decode + M=1 GEMV, each compiled in its OWN TU with forced arch
// flags so all three are built and comparable on the same host (see
// tests/unit/nvfp4_arch_kernel.inc + the three nvfp4_arch_*.cpp TUs).
//
// Each tier exposes the same three entrypoints. ws2 is the per-projection
// weight_scale_2; `base` points at the packed nvfp4-sm1xx projection block
// (FP4 region then Sm1xx-interleaved E4M3 scales). out / row_scratch are caller-
// owned FP32 buffers (decode_full needs N*K floats; gemv needs N; scratch K).

#include <cstdint>

namespace nvfp4_arch {

struct ArchKernel {
    const char* name;
    void (*decode_full)(const uint8_t* base, int N, int K, float ws2, float* out);
    void (*gemv_m1_fused)(const uint8_t* base, int N, int K, float ws2,
                          const uint16_t* a, float* out);
    void (*gemv_m1_materialize)(const uint8_t* base, int N, int K, float ws2,
                                const uint16_t* a, float* out, float* row_scratch);
};

// Always built (forced -march=native => AVX-512 on this host; falls to whatever
// the host supports otherwise).
const ArchKernel& avx512_kernel();
// Forced -mavx2 -mno-avx512f => AVX2 tier even on an AVX-512 host.
const ArchKernel& avx2_kernel();
// Forced -mno-avx2 -mno-avx512f (-mno-sse... kept; pure scalar fallback path).
const ArchKernel& scalar_kernel();

}  // namespace nvfp4_arch
