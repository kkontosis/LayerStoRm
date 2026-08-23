// Recording ExpertDevice for unit tests (KD-R3).
//
// Logs every operation as a DeviceOp record.  Alloc returns tagged pointers
// (kDevicePtrTag) so std::memcpy on device memory segfaults at test time.
// Header-only — include directly in test code.
//
// Guarded by LAYERSTORM_RECORDING_BACKEND.  Zero cost in production builds.

#pragma once

#ifdef LAYERSTORM_RECORDING_BACKEND

#include "core/expert_device.h"
#include "core/recording_device_ops.h"

#include <cstdlib>
#include <memory>
#include <span>
#include <vector>

namespace layerstorm::compute {

class RecordingExpertDevice final : public ExpertDevice {
public:
    explicit RecordingExpertDevice(config::GpuRef gpu) : gpu_(gpu) {}
    ~RecordingExpertDevice() override = default;

    // ── Device selection + identity ─────────────────────────────────────────
    void set_device() override {}
    const config::GpuRef& gpu() const override { return gpu_; }

    // ── Grouped GEMM ────────────────────────────────────────────────────────
    void nvfp4_grouped_gemm(const Nvfp4GroupedGemmParams& params,
                            void* /*workspace*/, size_t /*workspace_bytes*/,
                            void* stream) override {
        ops_.push_back({DeviceOp::kGemm, params.D_base, params.A_base,
                        0, stream});
    }

    void fp8_grouped_gemm(const Fp8GroupedGemmParams& params,
                          void* /*workspace*/, size_t /*workspace_bytes*/,
                          void* stream) override {
        ops_.push_back({DeviceOp::kGemm, params.D_base, params.A_base,
                        0, stream});
    }

    // ── Activation ──────────────────────────────────────────────────────────
    void fused_swiglu(void* output, const void* input,
                      const FusedSwigluParams& params,
                      int elem_size_bytes, void* stream) override {
        ops_.push_back({DeviceOp::kSwiglu, output, input,
                        static_cast<size_t>(params.num_tokens) * params.d
                            * elem_size_bytes,
                        stream});
    }

    // ── Token permutation ───────────────────────────────────────────────────
    void moe_permute(void* permuted_input, int32_t* /*expert_offsets*/,
                     int32_t* /*src_to_dest_map*/, int32_t* /*permuted_idx*/,
                     const void* hidden_states, const int32_t* /*topk_indices*/,
                     int num_tokens, int topk, int hidden_dim,
                     int /*num_experts*/, int elem_size_bytes,
                     void* /*workspace*/, void* stream) override {
        ops_.push_back({DeviceOp::kPermute, permuted_input, hidden_states,
                        static_cast<size_t>(num_tokens) * topk * hidden_dim
                            * elem_size_bytes,
                        stream});
    }

    void moe_unpermute(void* output, const void* permuted_output,
                       const float* /*topk_weights*/,
                       const int32_t* /*src_to_dest_map*/,
                       int num_tokens, int topk, int hidden_dim,
                       int elem_size_bytes, void* stream,
                       MoeCombineMode combine_mode =
                           MoeCombineMode::kReducedBf16) override {
        // Per-element output size: kReducedBf16 → [num_tokens, hidden] (bf16);
        // per-slot modes → [num_tokens, topk, hidden] at 4 (fp32) or 2 (bf16) bytes.
        const size_t bytes =
            combine_mode == MoeCombineMode::kPerSlotFp32
                ? static_cast<size_t>(num_tokens) * topk * hidden_dim * 4
            : combine_mode == MoeCombineMode::kPerSlotBf16
                ? static_cast<size_t>(num_tokens) * topk * hidden_dim * 2
                : static_cast<size_t>(num_tokens) * hidden_dim * elem_size_bytes;
        ops_.push_back({DeviceOp::kUnpermute, output, permuted_output,
                        bytes, stream});
    }

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

/// Factory: creates a RecordingExpertDevice.
inline std::unique_ptr<ExpertDevice> make_recording_expert_device(
        config::GpuRef gpu) {
    return std::make_unique<RecordingExpertDevice>(std::move(gpu));
}

}  // namespace layerstorm::compute

#endif  // LAYERSTORM_RECORDING_BACKEND
