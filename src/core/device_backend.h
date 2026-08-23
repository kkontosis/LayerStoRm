// Abstract interface for per-GPU hardware operations.
//
// DeviceBackend provides a device-agnostic abstraction over GPU hardware:
// GEMM, RMSNorm, FP8 quantization, device memory, and device selection.
// Each GPU in the system has its own instance (INV-BH-1).
//
// The abstract interface uses void* streams and generic POD param structs —
// no CUDA/HIP/ROCm types — compilable without any GPU SDK (INV-BH-2).
//
// Concrete implementations:
//   CudaSm120DeviceBackend  — SM120 (RTX 5090/5080) via CUDA
//   NullDeviceBackend       — no-op for unit tests

#pragma once

#include "core/gpu_ref.h"
#include "sm120/gemm/fp8/fp8_gemm.h"
#include "sm120/gemm/grouped_gemm.h"
#include "smxx/quant/dynamic_fp8_quant.h"
#include "smxx/quant/weight_fp8_quant.h"
#include "smxx/quant/nvfp4_dequant_bf16.h"
#include "smxx/quant/bf16_to_nvfp4_grouped.h"

#include <cstddef>

namespace layerstorm::compute {

/// Result of a non-blocking event query.
enum class EventStatus { kNotReady, kReady, kError };

/// Return value of query_event: status + vendor error code on failure.
struct EventQueryResult {
    EventStatus status = EventStatus::kNotReady;
    int error_code = 0;  ///< Vendor error code (e.g. cudaError_t) when status == kError.
};

class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;

    // ── Device selection ──────���───────────────────────────────────────────────

    /// Activate this backend's GPU as the current device.
    /// The GPU identity is fixed at construction (per-GPU, INV-BH-1).
    virtual void set_device() = 0;

    /// Block until all pending work on this GPU completes.
    /// Called during shutdown before freeing device memory.
    virtual void synchronize_device() = 0;

    /// Return the last async error code without clearing it (diagnostic).
    /// Uses cudaPeekAtLastError semantics: 0 = cudaSuccess.
    virtual int peek_last_error() = 0;

    // ── GPU identity ────���─────────────────────────────────��───────────────────

    /// Returns the GpuRef for this backend's GPU.
    virtual const config::GpuRef& gpu() const = 0;

    // ── Compute kernels ─────────────���─────────────────────────────��───────────

    /// FP8 blockwise-scaled GEMM.
    virtual void gemm(const Fp8GemmParams& params,
                      void* workspace, void* stream) = 0;

    /// RMS normalization (concrete impl determines dtype).
    /// row_stride: elements between consecutive token rows of BOTH input and
    /// out (== hidden_size for tight rows); only the first hidden_size
    /// elements of each row are read/written (TD-PREFILL-CHUNK-ATTN).
    virtual void rmsnorm(void* out, const void* input,
                         const void* weight, float eps,
                         int num_tokens, int hidden_size, int row_stride,
                         void* stream) = 0;

    /// Dynamic BF16 → FP8 E4M3 blockwise quantization (activations).
    virtual void quantize_fp8(const DynamicFp8QuantParams& params,
                              void* stream) = 0;

    /// Offline BF16 → FP8 E4M3 tile-level quantization (weights).
    virtual void weight_quantize_fp8(const WeightFp8QuantParams& params,
                                     void* stream) = 0;

    /// NVFP4 → BF16 weight dequantization (for online NVFP4→BF16→FP8 conversion).
    virtual void nvfp4_dequant_bf16(const Nvfp4DequantBf16Params& params,
                                    void* stream) = 0;

    /// NVFP4 grouped GEMM.
    virtual void nvfp4_grouped_gemm(
        const Nvfp4GroupedGemmParams& params,
        void* workspace, size_t workspace_bytes,
        void* stream) = 0;

    /// BF16 → NVFP4 grouped activation quantization.
    virtual void bf16_to_nvfp4_grouped(const Bf16ToNvfp4GroupedParams& params,
                                       void* stream) = 0;

    // ── Device memory ────────���──────────────────────────���─────────────────────

    /// Allocate device memory.  Returns nullptr on failure.
    virtual void* device_alloc(size_t bytes) = 0;

    /// Free device memory.  Must tolerate nullptr.
    virtual void device_free(void* ptr) = 0;

    /// Allocate pinned (page-locked) host memory for fast DMA.
    /// Returns nullptr on failure.  Used by TransferEngine staging pool (WP-7b).
    virtual void* host_alloc_pinned(size_t bytes) = 0;

    /// Free pinned host memory allocated by host_alloc_pinned().
    /// Must tolerate nullptr.
    virtual void host_free_pinned(void* ptr) = 0;

    /// Synchronize the device (all streams).  Init-time only (INV-5b exemption).
    virtual void device_sync() = 0;

    // ── Stream lifecycle ──────────────────────────────────────────────────

    /// Create a non-blocking stream on this backend's GPU.
    virtual void* create_stream() = 0;

    /// Create a non-blocking stream at the device's LOWEST scheduling
    /// priority (kernels on it yield block-dispatch to default/higher
    /// priority streams). For best-effort background pipelines sharing a
    /// GPU with latency-critical work (e.g. the DSpark draft block on a TP
    /// draft GPU — LS_DSPARK_DRAFT_LOWPRI). Default: plain create_stream()
    /// (backends without priority support). Destroy via destroy_stream().
    virtual void* create_stream_low_priority() { return create_stream(); }

    /// Destroy a stream previously created by create_stream().
    virtual void destroy_stream(void* stream) = 0;

    // ── Event lifecycle ───────────────────────────────────────────────────

    /// Create a timing-disabled event on this backend's GPU.
    virtual void* create_event() = 0;

    /// Create a TIMING-ENABLED event (for cudaEventElapsedTime). Profiling-only
    /// (perf_trace pure-DMA x-ray). Default: unsupported → nullptr (callers must
    /// null-check and skip). Destroy via destroy_event().
    virtual void* create_timing_event() { return nullptr; }

    /// Elapsed milliseconds between two timing events, both already completed.
    /// Returns <0 if unsupported or on error. Profiling-only.
    virtual float event_elapsed_ms(void* /*start*/, void* /*end*/) { return -1.0f; }

    /// Destroy an event previously created by create_event()/create_timing_event().
    virtual void destroy_event(void* event) = 0;

    /// Record an event on a stream.
    virtual void record_event(void* event, void* stream) = 0;

    /// Query whether an event has completed.
    /// Returns {kReady} if complete, {kNotReady} if pending,
    /// {kError, vendor_code} on device-fatal.
    virtual EventQueryResult query_event(void* event) = 0;

    /// Make a stream wait until an event completes (inter-stream dependency).
    virtual void stream_wait_event(void* stream, void* event) = 0;

    // ── Async data transfer (replaces raw cudaMemcpyAsync) ──────────────────

    /// Host-to-device async memcpy on a stream.
    virtual void memcpy_h2d_async(void* dst, const void* src,
                                  size_t bytes, void* stream) = 0;

    /// Device-to-host async memcpy on a stream.
    virtual void memcpy_d2h_async(void* dst, const void* src,
                                  size_t bytes, void* stream) = 0;

    /// Device-to-device async memcpy on a stream.
    virtual void memcpy_d2d_async(void* dst, const void* src,
                                  size_t bytes, void* stream) = 0;

    // ── Synchronous data transfer ──────────────────────────────────────────

    /// Synchronous host-to-device memcpy.
    virtual void memcpy_h2d(void* dst, const void* src, size_t bytes) = 0;

    /// Synchronous device-to-device memcpy.
    virtual void memcpy_d2d(void* dst, const void* src, size_t bytes) = 0;

    /// Async memset on a stream (e.g. zero-fill device buffers).
    virtual void memset_async(void* dst, int value, size_t bytes, void* stream) = 0;

    /// Direction-agnostic async memcpy (cudaMemcpyDefault semantics).
    /// Default delegates to memcpy_d2d_async; CUDA override uses cudaMemcpyDefault.
    virtual void memcpy_async(void* dst, const void* src,
                              size_t bytes, void* stream) {
        memcpy_d2d_async(dst, src, bytes, stream);
    }

    /// 2D device-to-device async memcpy (strided copy).
    /// Copies `height` rows, each `width` bytes wide, from src (pitch spitch)
    /// to dst (pitch dpitch). Default: per-row memcpy_d2d_async loop.
    /// CUDA override uses cudaMemcpy2DAsync.
    virtual void memcpy_2d_async(void* dst, size_t dpitch,
                                 const void* src, size_t spitch,
                                 size_t width, size_t height,
                                 void* stream) {
        auto* d = static_cast<char*>(dst);
        auto* s = static_cast<const char*>(src);
        for (size_t r = 0; r < height; ++r) {
            memcpy_d2d_async(d + r * dpitch, s + r * spitch, width, stream);
        }
    }

    // Non-copyable (polymorphic base).
    DeviceBackend(const DeviceBackend&) = delete;
    DeviceBackend& operator=(const DeviceBackend&) = delete;

protected:
    DeviceBackend() = default;
};

}  // namespace layerstorm::compute
