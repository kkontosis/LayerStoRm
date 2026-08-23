// No-op DeviceBackend for unit tests (no CUDA required).
//
// Compute methods are no-ops.  Alloc/free use std::malloc/std::free.
// Header-only — include directly in test code.

#pragma once

#include "core/device_backend.h"

#include <cstdlib>
#include <cstring>
#include <memory>

namespace layerstorm::compute {

class NullDeviceBackend : public DeviceBackend {
public:
    explicit NullDeviceBackend(config::GpuRef gpu) : gpu_(gpu) {}
    ~NullDeviceBackend() override = default;

    void set_device() override {}
    void synchronize_device() override {}
    int  peek_last_error() override { return 0; }
    const config::GpuRef& gpu() const override { return gpu_; }

    void gemm(const Fp8GemmParams&, void*, void*) override {}
    void rmsnorm(void*, const void*, const void*, float,
                 int, int, int, void*) override {}
    void quantize_fp8(const DynamicFp8QuantParams&, void*) override {}
    void weight_quantize_fp8(const WeightFp8QuantParams&, void*) override {}
    void nvfp4_dequant_bf16(const Nvfp4DequantBf16Params&, void*) override {}
    void nvfp4_grouped_gemm(const Nvfp4GroupedGemmParams&, void*, size_t, void*) override {}
    void bf16_to_nvfp4_grouped(const Bf16ToNvfp4GroupedParams&, void*) override {}

    void* device_alloc(size_t bytes) override {
        // 256-byte alignment matches CUDA's cudaMalloc guarantee (INV-GPU-1).
        size_t aligned = (bytes + 255) & ~size_t{255};
        if (aligned == 0) aligned = 256;
        return std::aligned_alloc(256, aligned);
    }
    void  device_free(void* ptr) override { std::free(ptr); }

    void* host_alloc_pinned(size_t bytes) override {
        size_t aligned = (bytes + 255) & ~size_t{255};
        if (aligned == 0) aligned = 256;
        return std::aligned_alloc(256, aligned);
    }
    void host_free_pinned(void* ptr) override { std::free(ptr); }

    void  device_sync() override {}

    // ── Stream lifecycle ────────────────────────────────────────────────────
    void* create_stream() override { return static_cast<void*>(new int{0}); }
    void  destroy_stream(void* s) override { delete static_cast<int*>(s); }

    // ── Event lifecycle ─────────────────────────────────────────────────────
    void* create_event() override { return static_cast<void*>(new int{0}); }
    void  destroy_event(void* e) override { delete static_cast<int*>(e); }
    void  record_event(void*, void*) override {}
    EventQueryResult query_event(void*) override { return {EventStatus::kReady, 0}; }
    void  stream_wait_event(void*, void*) override {}

    // ── Async data transfer ─────────────────────────────────────────────────
    void memcpy_h2d_async(void* dst, const void* src,
                          size_t bytes, void*) override {
        if (dst && src && bytes > 0) std::memcpy(dst, src, bytes);
    }
    void memcpy_d2h_async(void* dst, const void* src,
                          size_t bytes, void*) override {
        if (dst && src && bytes > 0) std::memcpy(dst, src, bytes);
    }
    void memcpy_d2d_async(void* dst, const void* src,
                          size_t bytes, void*) override {
        if (dst && src && bytes > 0) std::memcpy(dst, src, bytes);
    }

    // ── Async memset ────────────────────────────────────────────────────────
    void memset_async(void* dst, int value, size_t bytes, void*) override {
        if (dst && bytes > 0) std::memset(dst, value, bytes);
    }

    // ── Synchronous data transfer ───────────────────────────────────────────
    void memcpy_h2d(void* dst, const void* src, size_t bytes) override {
        if (dst && src && bytes > 0) std::memcpy(dst, src, bytes);
    }
    void memcpy_d2d(void* dst, const void* src, size_t bytes) override {
        if (dst && src && bytes > 0) std::memcpy(dst, src, bytes);
    }

    // memcpy_async: inherited default (delegates to memcpy_d2d_async → memcpy)

private:
    config::GpuRef gpu_;
};

/// Factory: creates a NullDeviceBackend (mirrors CudaSm120 factory pattern).
inline std::unique_ptr<DeviceBackend> make_null_device_backend(config::GpuRef gpu) {
    return std::make_unique<NullDeviceBackend>(std::move(gpu));
}

}  // namespace layerstorm::compute
