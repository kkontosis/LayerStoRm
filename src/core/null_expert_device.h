// No-op ExpertDevice for unit tests (no CUDA required).
//
// All compute methods are no-ops.  Alloc/free use std::malloc/std::free.
// Header-only — include directly in test code.

#pragma once

#include "core/expert_device.h"

#include <cstdlib>
#include <memory>

namespace layerstorm::compute {

class NullExpertDevice final : public ExpertDevice {
public:
    explicit NullExpertDevice(config::GpuRef gpu) : gpu_(gpu) {}
    ~NullExpertDevice() override = default;

    // ── Device selection + identity ─────────────────────────────────────────
    void set_device() override {}
    const config::GpuRef& gpu() const override { return gpu_; }

    // ── Grouped GEMM ────────────────────────────────────────────────────────
    void nvfp4_grouped_gemm(const Nvfp4GroupedGemmParams&,
                            void*, size_t, void*) override {}
    void fp8_grouped_gemm(const Fp8GroupedGemmParams&,
                          void*, size_t, void*) override {}

    // ── Activation ──────────────────────────────────────────────────────────
    void fused_swiglu(void*, const void*, const FusedSwigluParams&,
                      int, void*) override {}

    // ── Token permutation ───────────────────────────────────────────────────
    void moe_permute(void*, int32_t*, int32_t*, int32_t*,
                     const void*, const int32_t*,
                     int, int, int, int, int,
                     void*, void*) override {}
    void moe_unpermute(void*, const void*, const float*, const int32_t*,
                       int, int, int, int, void*,
                       MoeCombineMode = MoeCombineMode::kReducedBf16) override {}

    // ── Device memory ───────────────────────────────────────────────────────
    void* device_alloc(size_t bytes) override { return std::malloc(bytes); }
    void  device_free(void* ptr) override { std::free(ptr); }

private:
    config::GpuRef gpu_;
};

/// Factory: creates a NullExpertDevice.
inline std::unique_ptr<ExpertDevice> make_null_expert_device(
        config::GpuRef gpu) {
    return std::make_unique<NullExpertDevice>(std::move(gpu));
}

}  // namespace layerstorm::compute
