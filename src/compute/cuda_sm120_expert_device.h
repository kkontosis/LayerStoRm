// CudaSm120ExpertDevice: concrete ExpertDevice for SM120 (RTX 5090/5080).
//
// Delegates to existing kernel launchers (grouped_gemm, fused_swiglu,
// moe_permute/unpermute).  Header is CUDA-free — implementation in
// compute/kernels/sm120/device/cuda_sm120_expert_device.cu.

#pragma once

#include "core/expert_device.h"

#include <memory>

namespace layerstorm::compute {

/// Factory: creates a CudaSm120ExpertDevice for the given GPU.
std::unique_ptr<ExpertDevice> make_cuda_sm120_expert_device(
    config::GpuRef gpu);

}  // namespace layerstorm::compute
