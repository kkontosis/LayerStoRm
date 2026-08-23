// CUDA hardware query abstraction (#86e).
//
// CUDA-free header — declares functions whose implementations live in
// compute/kernels/smxx/hardware/cuda_hardware_query.cu.  Lets config
// and core TUs query GPU hardware without including <cuda_runtime.h>.

#pragma once

#include <string>

namespace layerstorm::core {

/// Per-GPU hardware info returned by query_gpu_info().
struct GpuHardwareInfo {
    int id = -1;
    double vram_gb = 0.0;         ///< Total global memory in GiB.
    std::string device_name;      ///< CUDA device name (e.g. "NVIDIA GeForce RTX 5090").
    std::string pci_bus_id;       ///< PCI bus ID string (e.g. "0000:6A:00.0").
    std::string uuid;             ///< CUDA device UUID ("GPU-xxxxxxxx-...."); stable across reorders.
};

/// Return the number of available CUDA devices.
/// Throws std::runtime_error on CUDA failure (driver not loaded, etc.).
int query_gpu_count();

/// Query device properties for a single GPU.
/// Throws std::runtime_error on CUDA failure.
GpuHardwareInfo query_gpu_info(int device_id);

/// Perform a small device allocation to wake up the PCIe link to full speed.
/// Moved from hardware_detect.cpp (#86e) — needs CUDA runtime.
void kickstart_pcie_link(int device_id);

/// Register a host memory region for pinned DMA (cudaHostRegister).
/// Returns 0 on success, or the cudaError_t value on failure.
/// WP-3: used by PrepackedSource for mmap'd expert files.
int host_register_pinned(void* ptr, size_t bytes);

/// Register a host memory region as PORTABLE pinned DMA
/// (cudaHostRegister(..., cudaHostRegisterPortable)). Portable so every GPU
/// context can DMA the region — required for a per-NUMA-node arena shared by
/// multiple GPUs (P-24). Returns 0 on success, or the cudaError_t value on
/// failure. The region must be writable anonymous host memory (an mbind'd
/// arena); file-backed COW mmaps are far slower to register (TD-100c).
int host_register_pinned_portable(void* ptr, size_t bytes);

/// Unregister a previously registered host memory region.
/// Returns 0 on success, or the cudaError_t value on failure.
int host_unregister_pinned(void* ptr);

}  // namespace layerstorm::core
