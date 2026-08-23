// AVX-512 tier of the NVFP4 multi-arch probe. Compiled with -march=native (the
// LAYERSTORM_FAST_CPU_FLAGS), so on a host with AVX-512F the #if __AVX512F__
// branch of nvfp4_arch_kernel.inc is selected.
#define LS_NVFP4_ARCH_NS ls_nvfp4_avx512
#include "nvfp4_arch_kernel.inc"
#include "nvfp4_arch_probe.h"

namespace nvfp4_arch {
const ArchKernel& avx512_kernel() {
    static const ArchKernel k{
        "avx512",
        &ls_nvfp4_avx512::decode_full,
        &ls_nvfp4_avx512::gemv_m1_fused,
        &ls_nvfp4_avx512::gemv_m1_materialize,
    };
    return k;
}
}  // namespace nvfp4_arch
