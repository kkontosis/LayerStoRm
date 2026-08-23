// Recording AttentionDevice for unit tests (KD-R3).
//
// Logs every operation as a DeviceOp record.  Alloc returns tagged pointers
// (kDevicePtrTag) so std::memcpy on device memory segfaults at test time.
// Header-only — include directly in test code.
//
// Guarded by LAYERSTORM_RECORDING_BACKEND.  Zero cost in production builds.

#pragma once

#ifdef LAYERSTORM_RECORDING_BACKEND

#include "core/attention_device.h"
#include "core/recording_device_ops.h"

#include <cstdlib>
#include <memory>
#include <span>
#include <vector>

namespace layerstorm::compute {

class RecordingAttentionDevice final : public AttentionDevice {
public:
    explicit RecordingAttentionDevice(config::GpuRef gpu) : gpu_(gpu) {}
    ~RecordingAttentionDevice() override = default;

    // ── Device selection + identity ─────────────────────────────────────────
    void set_device() override {}
    const config::GpuRef& gpu() const override { return gpu_; }

    // ── Compute kernels ─────────────────────────────────────────────────────
    void gemm(const Fp8GemmParams& params, void* /*workspace*/,
              void* stream) override {
        ops_.push_back({DeviceOp::kGemm, params.D, params.A, 0, stream});
    }

    // GGUF linear GEMMs (GG-4): recorded as kGemm (same C←A projection
    // semantic) so op-trace tests treat them like any projection GEMM.
    void gguf_mmvq(const GgufGemmParams& params, void* /*ws*/,
                   void* stream) override {
        ops_.push_back({DeviceOp::kGemm, params.C, params.A, 0, stream});
    }
    void gguf_mmq(const GgufGemmParams& params, void* /*ws*/,
                  void* stream) override {
        ops_.push_back({DeviceOp::kGemm, params.C, params.A, 0, stream});
    }
    void gguf_dequant_gemm(const GgufGemmParams& params,
                           void* stream) override {
        ops_.push_back({DeviceOp::kGemm, params.C, params.A, 0, stream});
    }

    void rmsnorm(void* out, const void* input, const void* /*weight*/,
                 float /*eps*/, int num_tokens, int hidden_size,
                 int /*row_stride*/, void* stream) override {
        ops_.push_back({DeviceOp::kRmsnorm, out, input,
                        static_cast<size_t>(num_tokens) * hidden_size * 2,
                        stream});
    }

    void quantize_fp8(const DynamicFp8QuantParams& params,
                      void* stream) override {
        ops_.push_back({DeviceOp::kQuantize, params.output, params.input,
                        0, stream});
    }

    void weight_quantize_fp8(const WeightFp8QuantParams& params,
                              void* stream) override {
        ops_.push_back({DeviceOp::kQuantize, params.output, params.input,
                        0, stream});
    }

    void nvfp4_dequant_bf16(const Nvfp4DequantBf16Params&, void*) override {}
    void nvfp4_grouped_gemm(const Nvfp4GroupedGemmParams&, void*, size_t,
                            void*) override {}
    void bf16_to_nvfp4_grouped(const Bf16ToNvfp4GroupedParams&,
                               void*) override {}
    void kv_bv_extract_dequant(const KvBvExtractDequantParams&,
                                void*) override {}
    void batched_gemm_bf16(const StridedBatchedGemmBf16Params&,
                            void*) override {}
    void absorb_q(const QAbsorbParams&, void*) override {}
    void rope_rotate(const RopeRotateParams&, void*) override {}

    // ── Device memory (tagged pointers) ─────────────────────────────────────
    void* device_alloc(size_t bytes) override {
        void* raw = std::malloc(bytes);
        if (!raw) return nullptr;
        void* tagged = tag_device_ptr(raw);
        ops_.push_back({DeviceOp::kAlloc, tagged, nullptr, bytes, nullptr});
        return tagged;
    }

    void device_free(void* ptr) override {
        if (!ptr) return;
        ops_.push_back({DeviceOp::kFree, ptr, nullptr, 0, nullptr});
        std::free(untag_device_ptr(ptr));
    }

    void device_sync() override {}
    void memcpy_h2d(void* dst, const void* src, size_t bytes) override {
        ops_.push_back({DeviceOp::kMemcpyH2D, dst, src, bytes, nullptr});
        void* real_dst = untag_device_ptr(dst);
        if (real_dst && src && bytes > 0) std::memcpy(real_dst, src, bytes);
    }
    void memcpy_2d_d2d_async(void* dst, size_t dpitch,
                             const void* src, size_t spitch,
                             size_t width, size_t height,
                             void* stream) override {
        ops_.push_back({DeviceOp::kMemcpy2D, dst, src, width * height, stream});
        char* real_dst = static_cast<char*>(untag_device_ptr(dst));
        const char* real_src =
            static_cast<const char*>(untag_device_ptr(const_cast<void*>(src)));
        if (!real_dst || !real_src) return;
        for (size_t r = 0; r < height; ++r)
            std::memcpy(real_dst + r * dpitch, real_src + r * spitch, width);
    }

    // ── KV cache append ─────────────────────────────────────────────────────
    void k_append(const void* c_kv, const void* /*k_rope*/, void* kv_cache,
                  int64_t /*cache_stride_block*/, int /*cache_stride_row*/,
                  const int* /*slot_mapping*/, int /*num_tokens*/,
                  int /*d_c*/, int /*d_rope*/,
                  int /*c_kv_row_stride*/, int /*k_rope_row_stride*/,
                  int /*page_size*/,
                  int /*layer_idx*/, void* stream) override {
        ops_.push_back({DeviceOp::kKAppend, kv_cache, c_kv, 0, stream});
    }

    // ── Prefill attention ───────────────────────────────────────────────────
    void prefill_attention(const void* q_compressed, int /*batch_size*/,
                           int /*seq_len_kv*/,
                           const int* /*seqlens_k*/, const int* /*block_tables*/,
                           int /*max_blocks_per_seq*/,
                           void* /*kv_cache*/, int64_t /*cache_stride_block*/,
                           int /*cache_stride_row*/,
                           int /*page_size*/, bool /*is_sparse*/,
                           bool /*chunk_causal*/,
                           const int* /*sparse_indices*/,
                           const int* /*topk_lengths*/, int /*topk*/,
                           void* out, float* /*lse*/,
                           int /*layer_idx*/, void* stream) override {
        ops_.push_back({DeviceOp::kAttention, out, q_compressed, 0, stream});
    }

    // ── Decode graph ops (no-op) ────────────────────────────────────────────
    void decode_graph_update(GraphEntry&, const void*,
                             const int*, const int*,
                             const int*,
                             int, void*) override {}
    void decode_graph_replay(GraphEntry&, void*) override {}
    void* decode_graph_out_ptr(GraphEntry&) override { return nullptr; }
    float* decode_graph_lse_ptr(GraphEntry&) override { return nullptr; }

    // ── DCP allreduce graph (no-op) ─────────────────────────────────────────
    void dcp_graph_replay(GraphEntry&, void*) override {}

    // ── Recording query API ─────────────────────────────────────────────────
    std::span<const DeviceOp> ops() const { return ops_; }
    void clear() { ops_.clear(); }

    bool has_op(DeviceOp::Kind k, void* dst = nullptr) const {
        for (const auto& op : ops_) {
            if (op.kind == k && (dst == nullptr || op.dst == dst))
                return true;
        }
        return false;
    }

private:
    config::GpuRef gpu_;
    std::vector<DeviceOp> ops_;
};

/// Factory: creates a RecordingAttentionDevice.
inline std::unique_ptr<AttentionDevice> make_recording_attention_device(
        config::GpuRef gpu) {
    return std::make_unique<RecordingAttentionDevice>(std::move(gpu));
}

}  // namespace layerstorm::compute

#endif  // LAYERSTORM_RECORDING_BACKEND
