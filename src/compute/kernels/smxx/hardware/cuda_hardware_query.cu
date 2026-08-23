// CUDA hardware query implementation (#86e).
//
// Wraps CUDA runtime device queries behind a CUDA-free C++ interface
// declared in core/cuda_hardware_query.h.  This is the ONLY TU that
// calls cudaGetDeviceCount / cudaGetDeviceProperties / cudaDeviceGetPCIBusId.

#include "core/cuda_hardware_query.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace layerstorm::core {

int query_gpu_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err));
    }
    return count;
}

GpuHardwareInfo query_gpu_info(int device_id) {
    cudaDeviceProp props{};
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaGetDeviceProperties failed for device ") +
            std::to_string(device_id) + ": " + cudaGetErrorString(err));
    }

    GpuHardwareInfo info;
    info.id = device_id;
    info.vram_gb = static_cast<double>(props.totalGlobalMem) /
                   (1024.0 * 1024.0 * 1024.0);
    info.device_name = props.name;

    char pci_buf[16] = {};
    err = cudaDeviceGetPCIBusId(pci_buf, sizeof(pci_buf), device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaDeviceGetPCIBusId failed for device ") +
            std::to_string(device_id) + ": " + cudaGetErrorString(err));
    }
    info.pci_bus_id = pci_buf;

    // Device UUID — stable across PCI reorder / driver restart (unlike the bus id
    // or the enumeration ordinal). Format as nvidia-smi does: "GPU-" + 8-4-4-4-12
    // lowercase hex of the 16 raw bytes. Used by the loader calibration to detect a
    // wrong-machine / reordered-device JSON (I8b device-identity check).
    char uuid_buf[40] = {};
    const unsigned char* b =
        reinterpret_cast<const unsigned char*>(props.uuid.bytes);
    std::snprintf(uuid_buf, sizeof(uuid_buf),
                  "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                  b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    info.uuid = uuid_buf;

    return info;
}

void kickstart_pcie_link(int device_id) {
    cudaSetDevice(device_id);
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, 4096) == cudaSuccess) {
        cudaFree(ptr);
        cudaDeviceSynchronize();
    }
}

int host_register_pinned(void* ptr, size_t bytes) {
    // Prefer ReadOnly (no device write mapping needed), but it is unsupported on
    // some platforms (e.g. SM120 returns cudaErrorNotSupported=801). Fall back
    // to the default flag, which succeeds for writable host memory — including
    // COW (PROT_READ|PROT_WRITE MAP_PRIVATE) file mmaps. A read-only (PROT_READ)
    // file mapping cannot be registered with either flag (cudaErrorInvalidValue);
    // callers that want pinning must map COW-writable.
    auto err = cudaHostRegister(ptr, bytes, cudaHostRegisterReadOnly);
    if (err == cudaErrorNotSupported) {
        cudaGetLastError();  // clear the sticky error before retrying
        err = cudaHostRegister(ptr, bytes, cudaHostRegisterDefault);
    }
    if (err != cudaSuccess)
        cudaGetLastError();  // Clear runtime error state so it doesn't contaminate later ops
    return static_cast<int>(err);
}

int host_register_pinned_portable(void* ptr, size_t bytes) {
    // Portable pinning over a whole anonymous arena (P-24): every GPU context
    // can DMA the region, so a shared-node arena serves multiple GPUs with one
    // registration. Anonymous (mbind'd) writable memory registers fast
    // (~tens of ms/GB), unlike file-backed COW mmaps (TD-100c).
    auto err = cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);
    if (err != cudaSuccess)
        cudaGetLastError();  // clear sticky error so later ops aren't contaminated
    return static_cast<int>(err);
}

int host_unregister_pinned(void* ptr) {
    return static_cast<int>(cudaHostUnregister(ptr));
}

}  // namespace layerstorm::core
