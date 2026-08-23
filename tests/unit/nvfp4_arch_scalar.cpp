// Scalar tier of the NVFP4 multi-arch probe. Compiled with -mno-avx2
// -mno-avx512f (see tests/unit/CMakeLists.txt), so BOTH __AVX512F__ and __AVX2__
// are undefined and the #else scalar fallback of nvfp4_arch_kernel.inc is taken
// — the per-nibble scalar table-lookup decode + scalar dot.
#define LS_NVFP4_ARCH_NS ls_nvfp4_scalar
#include "nvfp4_arch_kernel.inc"
#include "nvfp4_arch_probe.h"

namespace nvfp4_arch {
const ArchKernel& scalar_kernel() {
    static const ArchKernel k{
        "scalar",
        &ls_nvfp4_scalar::decode_full,
        &ls_nvfp4_scalar::gemv_m1_fused,
        &ls_nvfp4_scalar::gemv_m1_materialize,
    };
    return k;
}
}  // namespace nvfp4_arch
