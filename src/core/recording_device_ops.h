// Shared types for recording device backends (KD-R3).
//
// DeviceOp captures kernel/memcpy/event operations for test assertions.
// Tagged pointer utilities distinguish device vs host memory at test time.
//
// Test-only — guarded by LAYERSTORM_RECORDING_BACKEND.  Zero cost in
// production builds.

#pragma once

#ifdef LAYERSTORM_RECORDING_BACKEND

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "core/null_device_backend.h"

namespace layerstorm::compute {

// ── DeviceOp ────────────────────────────────────────────────────────────────

/// A single recorded device operation.
struct DeviceOp {
    enum Kind {
        kAlloc,        // device_alloc
        kFree,         // device_free
        kGemm,         // gemm / grouped_gemm
        kRmsnorm,      // rmsnorm
        kQuantize,     // quantize_fp8
        kMemcpyD2D,    // device-to-device copy
        kMemcpyH2D,    // host-to-device copy
        kMemcpyD2H,    // device-to-host copy
        kEventRecord,  // record_event
        kEventWait,    // stream_wait_event
        kAttention,    // prefill_attention
        kKAppend,      // k_append
        kSwiglu,       // fused_swiglu
        kPermute,      // moe_permute
        kUnpermute,    // moe_unpermute
        kMemcpy2D,     // 2D strided device-to-device copy
        kMemset,       // memset_async
    };

    Kind        kind;
    void*       dst;     ///< output pointer (or event for event ops)
    const void* src;     ///< input pointer
    size_t      bytes;   ///< byte count (0 when not meaningful)
    void*       stream;  ///< stream handle (nullptr for alloc/free)
};

// ── Tagged pointer utilities ────────────────────────────────────────────────
//
// device_alloc returns heap pointers ORed with kDevicePtrTag.  The tag sits
// in the upper 16 bits of the x86-64 virtual address space which the kernel
// never maps for userspace.  Any std::memcpy on a tagged pointer segfaults,
// immediately surfacing host-vs-device confusion in tests.

inline constexpr uintptr_t kDevicePtrTag = 0x8000'0000'0000'0000ULL;

inline bool is_device_ptr(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & kDevicePtrTag) != 0;
}

inline void* tag_device_ptr(void* p) {
    return reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(p) | kDevicePtrTag);
}

inline void* untag_device_ptr(void* p) {
    return reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(p) & ~kDevicePtrTag);
}

inline const void* untag_device_ptr(const void* p) {
    return reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(p) & ~kDevicePtrTag);
}

// ── RecordingStreamContext ──────────────────────────────────────────────────
//
// Shared ops log for RecordingDeviceBackend.  Both the backend and the test
// code hold a shared_ptr to this.

struct RecordingStreamContext {
    std::vector<DeviceOp> ops;

    void record(DeviceOp op) { ops.push_back(op); }

    std::span<const DeviceOp> view() const { return ops; }

    void clear() { ops.clear(); }

    bool has_op(DeviceOp::Kind k, void* dst = nullptr) const {
        for (const auto& op : ops) {
            if (op.kind == k && (dst == nullptr || op.dst == dst))
                return true;
        }
        return false;
    }
};

// ── Recording device backend ────────────────────────────────────────────────
//
// Extends NullDeviceBackend with recording of record_event, stream_wait_event,
// and memcpy_d2h_async.  Use context() to access the recorded ops in tests.

class RecordingDeviceBackend : public NullDeviceBackend {
public:
    explicit RecordingDeviceBackend(config::GpuRef gpu)
        : NullDeviceBackend(std::move(gpu)),
          ctx_(std::make_shared<RecordingStreamContext>()) {}

    std::shared_ptr<RecordingStreamContext> context() const { return ctx_; }

    void record_event(void* event, void* stream) override {
        ctx_->record({DeviceOp::kEventRecord, event, nullptr, 0, stream});
    }

    void stream_wait_event(void* stream, void* event) override {
        ctx_->record({DeviceOp::kEventWait, event, nullptr, 0, stream});
    }

    void memcpy_2d_async(void* dst, size_t dpitch,
                         const void* src, size_t spitch,
                         size_t width, size_t height,
                         void* stream) override {
        ctx_->record({DeviceOp::kMemcpy2D, dst, src,
                     width * height, stream});
        // Delegate to per-row copy for data correctness in tests.
        NullDeviceBackend::memcpy_2d_async(dst, dpitch, src, spitch,
                                            width, height, stream);
    }

    void memcpy_d2h_async(void* host_dst, const void* device_src,
                           size_t bytes, void* stream) override {
        ctx_->record({DeviceOp::kMemcpyD2H, host_dst, device_src,
                     bytes, stream});
        // Still perform the copy so data flows correctly in tests.
        if (host_dst && device_src && bytes > 0) {
            const void* real_src = is_device_ptr(device_src)
                ? untag_device_ptr(device_src) : device_src;
            std::memcpy(host_dst, real_src, bytes);
        }
    }

    void memset_async(void* dst, int value, size_t bytes, void* stream) override {
        ctx_->record({DeviceOp::kMemset, dst, nullptr, bytes, stream});
        // Untag device pointers from RecordingExpertDevice::device_alloc
        // before std::memset (tagged pointers segfault on host access).
        if (dst && bytes > 0) {
            void* real_dst = is_device_ptr(dst) ? untag_device_ptr(dst) : dst;
            std::memset(real_dst, value, bytes);
        }
    }

    void memcpy_d2d_async(void* dst, const void* src,
                           size_t bytes, void* stream) override {
        ctx_->record({DeviceOp::kMemcpyD2D, dst, src, bytes, stream});
        if (dst && src && bytes > 0) {
            void* real_dst = is_device_ptr(dst) ? untag_device_ptr(dst) : dst;
            const void* real_src = is_device_ptr(src)
                ? untag_device_ptr(src) : src;
            std::memcpy(real_dst, real_src, bytes);
        }
    }

private:
    std::shared_ptr<RecordingStreamContext> ctx_;
};

}  // namespace layerstorm::compute

#endif  // LAYERSTORM_RECORDING_BACKEND
