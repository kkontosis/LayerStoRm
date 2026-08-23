// AVX2 tier of the NVFP4 multi-arch probe. Compiled with -mavx2 -mfma
// -mno-avx512f (see tests/unit/CMakeLists.txt), so __AVX512F__ is UNDEFINED and
// the #elif __AVX2__ branch of nvfp4_arch_kernel.inc is selected — exercising
// the 256-bit two-half vpermps E2M1 decode + 8-wide FMA GEMV on this AVX-512
// host where the production kernel would otherwise always take the AVX-512 path.
#define LS_NVFP4_ARCH_NS ls_nvfp4_avx2
#include "nvfp4_arch_kernel.inc"
#include "nvfp4_arch_probe.h"

namespace nvfp4_arch {
const ArchKernel& avx2_kernel() {
    static const ArchKernel k{
        "avx2",
        &ls_nvfp4_avx2::decode_full,
        &ls_nvfp4_avx2::gemv_m1_fused,
        &ls_nvfp4_avx2::gemv_m1_materialize,
    };
    return k;
}
}  // namespace nvfp4_arch
