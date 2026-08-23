// ExpertDevice base default implementations.
//
// gguf_grouped_gemm() default-throws: it is a CPU-only path (vendored ik kernels)
// overridden by NumaCpuExpertDevice / MultiNumaCpuExpertDevice. Every other
// device (CudaSm120ExpertDevice, NullExpertDevice, RecordingExpertDevice)
// inherits this throw — so the interface stays satisfiable and the method is
// "usable for the future" without forcing each device to define a stub.
//
// CPU-only TU — no CUDA (INV-GPU-1).

#include "core/expert_device.h"

#include <stdexcept>

namespace layerstorm::compute {

void ExpertDevice::gguf_grouped_gemm(const GgufGroupedGemmParams&,
                                     void*, size_t, void*) {
    throw std::runtime_error(
        "gguf_grouped_gemm is not supported on this ExpertDevice "
        "(GGUF CPU GEMM is implemented only by the NUMA CPU expert devices)");
}

}  // namespace layerstorm::compute
