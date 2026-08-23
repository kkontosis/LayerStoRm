// No-op AttentionDevice for unit tests (no CUDA required).
//
// Compute and attention methods are no-ops.  Alloc/free use std::malloc/std::free.
// Header-only — include directly in test code.

#pragma once

#include "core/attention_device.h"

#include <cstdlib>
#include <cstring>
#include <memory>

namespace layerstorm::compute {

class NullAttentionDevice final : public AttentionDevice {
public:
    explicit NullAttentionDevice(config::GpuRef gpu) : gpu_(gpu) {}
    ~NullAttentionDevice() override = default;

    // ── Device selection + identity ─────────────────────────────────────────
    void set_device() override {}
    const config::GpuRef& gpu() const override { return gpu_; }

    // ── Compute kernels ─────────────────────────────────────────────────────
    void gemm(const Fp8GemmParams&, void*, void*) override {}
    void gguf_mmvq(const GgufGemmParams&, void*, void*) override {}
    void gguf_mmq(const GgufGemmParams&, void*, void*) override {}
    void gguf_dequant_gemm(const GgufGemmParams&, void*) override {}
    void rmsnorm(void*, const void*, const void*, float,
                 int, int, int, void*) override {}
    void quantize_fp8(const DynamicFp8QuantParams&, void*) override {}
    void weight_quantize_fp8(const WeightFp8QuantParams&, void*) override {}
    void nvfp4_dequant_bf16(const Nvfp4DequantBf16Params&, void*) override {}
    void nvfp4_grouped_gemm(const Nvfp4GroupedGemmParams&, void*, size_t, void*) override {}
    void bf16_to_nvfp4_grouped(const Bf16ToNvfp4GroupedParams&, void*) override {}
    void kv_bv_extract_dequant(const KvBvExtractDequantParams&, void*) override {}
    void batched_gemm_bf16(const StridedBatchedGemmBf16Params&, void*) override {}
    void absorb_q(const QAbsorbParams&, void*) override {}
    void rope_rotate(const RopeRotateParams&, void*) override {}

    // ── Device memory ───────────────────────────────────────────────────────
    void* device_alloc(size_t bytes) override { return std::malloc(bytes); }
    void  device_free(void* ptr) override { std::free(ptr); }
    void  device_sync() override {}
    void  memcpy_h2d(void* dst, const void* src, size_t bytes) override {
        if (dst && src && bytes > 0) std::memcpy(dst, src, bytes);
    }
    void  memcpy_2d_d2d_async(void* dst, size_t dpitch,
                              const void* src, size_t spitch,
                              size_t width, size_t height,
                              void*) override {
        if (!dst || !src) return;
        for (size_t r = 0; r < height; ++r)
            std::memcpy(static_cast<char*>(dst) + r * dpitch,
                        static_cast<const char*>(src) + r * spitch, width);
    }

    // ── KV cache append ─────────────────────────────────────────────────────
    void k_append(const void*, const void*, void*,
                  int64_t, int, const int*, int,
                  int, int, int, int, int,
                  int, void*) override {}

    // ── Prefill attention ───────────────────────────────────────────────────
    void prefill_attention(const void*, int, int,
                           const int*, const int*,
                           int,
                           void*, int64_t, int,
                           int, bool, bool,
                           const int*, const int*, int,
                           void*, float*,
                           int, void*) override {}

    // ── Decode graph ops ────────────────────────────────────────────────────
    void decode_graph_update(GraphEntry&, const void*,
                             const int*, const int*,
                             const int*,
                             int, void*) override {}
    void decode_graph_replay(GraphEntry&, void*) override {}
    void* decode_graph_out_ptr(GraphEntry&) override { return nullptr; }
    float* decode_graph_lse_ptr(GraphEntry&) override { return nullptr; }

    // ── DCP allreduce graph ─────────────────────────────────────────────────
    void dcp_graph_replay(GraphEntry&, void*) override {}

private:
    config::GpuRef gpu_;
};

/// Factory: creates a NullAttentionDevice.
inline std::unique_ptr<AttentionDevice> make_null_attention_device(
        config::GpuRef gpu) {
    return std::make_unique<NullAttentionDevice>(std::move(gpu));
}

}  // namespace layerstorm::compute
