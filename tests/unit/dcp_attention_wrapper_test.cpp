// Unit tests for DCP attention wrapper.
//
// Group 1: Construction/passthrough (null attention device, no GPU)
// Group 2: Call sequencing (recording attention device)
// Group 3: Multi-GPU correctness (REQUIRES_MULTI_GPU(2))

#include "compute/kernels/attention/dcp_attention_wrapper.h"
#include "compute/snapmla_sm120_attention_device.h"
#include "core/attention_device.h"
#include "core/device_backend.h"
#include "core/null_attention_device.h"
#include "core/null_device_backend.h"
#include "parallelism/collective_backend.h"
#include "parallelism/dcp_communicator.h"
#include "parallelism/null_collective_backend.h"
#include "recording_collective_backend.h"

// Multi-GPU backends (guarded by REQUIRES_MULTI_GPU)
#include "compute/cuda_sm120_device_backend.h"
#include "parallelism/nccl_collective_backend.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/gpu_ref.h"
#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;
namespace lcomp = layerstorm::compute;
namespace lp = layerstorm::parallelism;

static std::vector<layerstorm::config::GpuRef> make_gpu_refs(int count) {
    std::vector<layerstorm::config::GpuRef> v;
    for (int i = 0; i < count; ++i)
        v.push_back({.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090});
    return v;
}

// -- Helpers -----------------------------------------------------------------

#define CUDA_CHECK(expr) \
    ASSERT_EQ((expr), cudaSuccess) << cudaGetErrorString(expr)

#define REQUIRES_MULTI_GPU(n) \
    do { int c_ = 0; cudaGetDeviceCount(&c_); \
         if (c_ < (n)) GTEST_SKIP() << "Need " << (n) << " GPUs"; } while(0)

static uint16_t float_to_bf16(float v) {
    uint32_t f; std::memcpy(&f, &v, sizeof(f));
    f += 0x7FFF + ((f >> 16) & 1);
    return static_cast<uint16_t>(f >> 16);
}
static float bf16_to_float(uint16_t b) {
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float r; std::memcpy(&r, &f, sizeof(r)); return r;
}

// -- Null-backend communicator helper ----------------------------------------

// Callers must keep device_backends and collective alive for the lifetime of
// the DcpCommunicator constructed from the returned Options.
static lp::DcpCommunicator::Options null_comm_opts(
        int dcp_size,
        std::vector<lcomp::DeviceBackend*> device_backends,
        lp::CollectiveBackend* collective,
        int max_batch = 8) {
    return {
        .dcp_size         = dcp_size,
        .device_backends  = std::move(device_backends),
        .max_batch_size   = max_batch,
        .num_heads        = 128,
        .attn_output_dim  = 512,
        .hidden_size      = 7168,
        .collective       = collective,
    };
}

static lc::DcpAttentionWrapperConfig wrapper_cfg(int dcp_size,
                                                   int max_batch = 8) {
    return {
        .num_heads_local = 128 / std::max(dcp_size, 1),
        .head_dim        = 512,
        .hidden_size     = 7168,
        .max_batch_size  = max_batch,
        .gpus            = make_gpu_refs(dcp_size),
    };
}

// -- Null AttentionDevice helpers for wrapper tests --------------------------

static std::vector<std::unique_ptr<lc::NullAttentionDevice>> g_wrapper_null_devs;

static std::vector<lc::AttentionDevice*> make_null_devs(int count) {
    g_wrapper_null_devs.clear();
    std::vector<lc::AttentionDevice*> ptrs;
    for (int i = 0; i < count; ++i) {
        g_wrapper_null_devs.push_back(std::make_unique<lc::NullAttentionDevice>(
            layerstorm::config::GpuRef{.position = i, .id = i,
                                        .type = layerstorm::config::GpuType::rtx5090}));
        ptrs.push_back(g_wrapper_null_devs.back().get());
    }
    return ptrs;
}

// ============================================================================
// Group 1: Construction / passthrough (null backend, no GPU)
// ============================================================================

TEST(DcpAttentionWrapper, DcpSize1Passthrough) {
    // Device backends + collective must outlive DcpCommunicator.
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    for (int i = 0; i < 1; ++i) {
        owned_devbacks.push_back(lcomp::make_null_device_backend(
            {.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090}));
        devback_ptrs.push_back(owned_devbacks.back().get());
    }
    auto null_collective = lp::make_null_collective_backend();

    auto comm = lp::DcpCommunicator(
        null_comm_opts(1, devback_ptrs, null_collective.get()));
    auto devs = make_null_devs(1);
    auto wrapper = lc::DcpAttentionWrapper(
        &comm, wrapper_cfg(1), devs, lc::null_launch_correction());

    EXPECT_FALSE(wrapper.is_active());
    EXPECT_EQ(wrapper.dcp_size(), 1);

    // No-op calls with nullptr — should not crash
    const float* lse = nullptr;
    void* out = nullptr;
    void* stream = nullptr;
    wrapper.correct_output(&out, &lse, 1, &stream);
    wrapper.reduce_hidden(&out, 1, &stream);
}

TEST(DcpAttentionWrapper, NullCommPassthrough) {
    // No communicator — wrapper should be inactive regardless.
    auto devs = make_null_devs(2);
    auto wrapper = lc::DcpAttentionWrapper(
        nullptr, wrapper_cfg(2), devs, lc::null_launch_correction());

    EXPECT_FALSE(wrapper.is_active());
    EXPECT_EQ(wrapper.dcp_size(), 1);
}

TEST(DcpAttentionWrapper, DcpSize2Active) {
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    for (int i = 0; i < 2; ++i) {
        owned_devbacks.push_back(lcomp::make_null_device_backend(
            {.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090}));
        devback_ptrs.push_back(owned_devbacks.back().get());
    }
    auto null_collective = lp::make_null_collective_backend();

    auto comm = lp::DcpCommunicator(
        null_comm_opts(2, devback_ptrs, null_collective.get()));
    auto devs = make_null_devs(2);
    auto wrapper = lc::DcpAttentionWrapper(
        &comm, wrapper_cfg(2), devs, lc::null_launch_correction());

    EXPECT_TRUE(wrapper.is_active());
    EXPECT_EQ(wrapper.dcp_size(), 2);
}

TEST(DcpAttentionWrapper, ThrowsOnTooFewDevices) {
    // TD-87c: constructor must reject attention_devices.size() < dcp_size.
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    for (int i = 0; i < 2; ++i) {
        owned_devbacks.push_back(lcomp::make_null_device_backend(
            {.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090}));
        devback_ptrs.push_back(owned_devbacks.back().get());
    }
    auto null_collective = lp::make_null_collective_backend();
    auto comm = lp::DcpCommunicator(
        null_comm_opts(2, devback_ptrs, null_collective.get()));

    // Only 1 device for dcp_size=2 → should throw.
    auto devs = make_null_devs(1);
    EXPECT_THROW(
        lc::DcpAttentionWrapper(&comm, wrapper_cfg(2), devs,
                                lc::null_launch_correction()),
        std::invalid_argument);
}

TEST(DcpAttentionWrapper, ThrowsOnTooFewGpuRefs) {
    // TD-87c: constructor must reject cfg.gpus.size() < dcp_size.
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    for (int i = 0; i < 2; ++i) {
        owned_devbacks.push_back(lcomp::make_null_device_backend(
            {.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090}));
        devback_ptrs.push_back(owned_devbacks.back().get());
    }
    auto null_collective = lp::make_null_collective_backend();
    auto comm = lp::DcpCommunicator(
        null_comm_opts(2, devback_ptrs, null_collective.get()));

    // 2 devices but cfg has only 1 GPU ref → should throw.
    auto devs = make_null_devs(2);
    auto bad_cfg = wrapper_cfg(1);  // gpus.size()=1
    EXPECT_THROW(
        lc::DcpAttentionWrapper(&comm, bad_cfg, devs,
                                lc::null_launch_correction()),
        std::invalid_argument);
}

TEST(DcpAttentionWrapper, GlobalLseBuffersAllocated) {
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    for (int i = 0; i < 2; ++i) {
        owned_devbacks.push_back(lcomp::make_null_device_backend(
            {.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090}));
        devback_ptrs.push_back(owned_devbacks.back().get());
    }
    auto null_collective = lp::make_null_collective_backend();

    auto comm = lp::DcpCommunicator(
        null_comm_opts(2, devback_ptrs, null_collective.get()));
    auto devs = make_null_devs(2);
    {
        auto wrapper = lc::DcpAttentionWrapper(
            &comm, wrapper_cfg(2), devs, lc::null_launch_correction());
        EXPECT_TRUE(wrapper.is_active());
    }
    // Destructor frees buffers — no crash
}

TEST(DcpAttentionWrapper, DestroyDefaultInactive) {
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    for (int i = 0; i < 1; ++i) {
        owned_devbacks.push_back(lcomp::make_null_device_backend(
            {.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090}));
        devback_ptrs.push_back(owned_devbacks.back().get());
    }
    auto null_collective = lp::make_null_collective_backend();

    auto comm = lp::DcpCommunicator(
        null_comm_opts(1, devback_ptrs, null_collective.get()));
    auto devs = make_null_devs(1);
    {
        auto wrapper = lc::DcpAttentionWrapper(
            &comm, wrapper_cfg(1), devs, lc::null_launch_correction());
        EXPECT_FALSE(wrapper.is_active());
    }
}

// ============================================================================
// Group 2: Call sequencing (recording backend)
// ============================================================================

// Recording collective backend that appends to a shared call log.

// Shared recording collective backend (tests/unit/recording_collective_backend.h).
using RecordingCollectiveBackend = layerstorm::test::RecordingCollectiveBackend;

// Recording AttentionDevice that logs set_device calls.
class WrapperRecordingAttentionDevice final : public lc::AttentionDevice {
public:
    WrapperRecordingAttentionDevice(layerstorm::config::GpuRef gpu,
                                     std::vector<std::string>& log)
        : gpu_(gpu), log_(log) {}
    void set_device() override {
        log_.push_back("set_device(" + std::to_string(gpu_.id) + ")");
    }
    const layerstorm::config::GpuRef& gpu() const override { return gpu_; }
    void gemm(const lc::Fp8GemmParams&, void*, void*) override {}
    void gguf_mmvq(const lc::GgufGemmParams&, void*, void*) override {}
    void gguf_mmq(const lc::GgufGemmParams&, void*, void*) override {}
    void gguf_dequant_gemm(const lc::GgufGemmParams&, void*) override {}
    void rmsnorm(void*, const void*, const void*, float, int, int, int, void*) override {}
    void quantize_fp8(const lc::DynamicFp8QuantParams&, void*) override {}
    void weight_quantize_fp8(const lc::WeightFp8QuantParams&, void*) override {}
    void nvfp4_dequant_bf16(const lc::Nvfp4DequantBf16Params&, void*) override {}
    void nvfp4_grouped_gemm(const lc::Nvfp4GroupedGemmParams&, void*, size_t,
                            void*) override {}
    void bf16_to_nvfp4_grouped(const lc::Bf16ToNvfp4GroupedParams&,
                               void*) override {}
    void kv_bv_extract_dequant(const lc::KvBvExtractDequantParams&,
                                void*) override {}
    void batched_gemm_bf16(const lc::StridedBatchedGemmBf16Params&,
                            void*) override {}
    void absorb_q(const lc::QAbsorbParams&, void*) override {}
    void rope_rotate(const lc::RopeRotateParams&, void*) override {}
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
    void k_append(const void*, const void*, void*,
                  int64_t, int, const int*, int,
                  int, int, int, int, int,
                  int, void*) override {}
    void prefill_attention(const void*, int, int,
                           const int*, const int*,
                           int,
                           void*, int64_t, int,
                           int, bool, bool,
                           const int*, const int*, int,
                           void*, float*,
                           int, void*) override {}
    void decode_graph_update(lc::GraphEntry&, const void*,
                             const int*, const int*,
                             const int*,
                             int, void*) override {}
    void decode_graph_replay(lc::GraphEntry&, void*) override {}
    void* decode_graph_out_ptr(lc::GraphEntry&) override { return nullptr; }
    float* decode_graph_lse_ptr(lc::GraphEntry&) override { return nullptr; }
    void dcp_graph_replay(lc::GraphEntry&, void*) override {}
private:
    layerstorm::config::GpuRef gpu_;
    std::vector<std::string>& log_;
};

static lc::LaunchCorrectionFn recording_correction(std::vector<std::string>& log) {
    return [&log](void*, const float*, float*, int, int, int, int, int rank, void*) {
        log.push_back("correction(rank=" + std::to_string(rank) + ")");
    };
}

TEST(DcpAttentionWrapper, CorrectOutputSequence) {
    std::vector<std::string> log;

    // Owned backends — must outlive DcpCommunicator.
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    for (int i = 0; i < 2; ++i) {
        owned_devbacks.push_back(lcomp::make_null_device_backend(
            {.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090}));
        devback_ptrs.push_back(owned_devbacks.back().get());
    }
    auto recording_collective = std::make_unique<RecordingCollectiveBackend>(log);

    auto comm = lp::DcpCommunicator(
        null_comm_opts(2, devback_ptrs, recording_collective.get()));

    std::vector<std::unique_ptr<WrapperRecordingAttentionDevice>> rec_devs;
    std::vector<lc::AttentionDevice*> dev_ptrs;
    for (int i = 0; i < 2; ++i) {
        rec_devs.push_back(std::make_unique<WrapperRecordingAttentionDevice>(
            layerstorm::config::GpuRef{.position = i, .id = i,
                                        .type = layerstorm::config::GpuType::rtx5090},
            log));
        dev_ptrs.push_back(rec_devs.back().get());
    }

    auto wrapper = lc::DcpAttentionWrapper(
        &comm, wrapper_cfg(2), dev_ptrs, recording_correction(log));

    log.clear();  // clear any init calls

    void* outputs[2] = {nullptr, nullptr};
    const float* lses[2] = {nullptr, nullptr};
    void* streams[2] = {nullptr, nullptr};
    wrapper.correct_output(outputs, lses, 4, streams);

    // Expected sequence:
    // allgather_lse (grouped, comm's set_device NOT recorded — separate backend):
    //   group_begin, allgather, allgather, group_end
    // per-rank correction (wrapper's set_device IS recorded):
    //   set_device(0), correction(rank=0), set_device(1), correction(rank=1)
    // allreduce_output (grouped):
    //   group_begin, allreduce, allreduce, group_end
    std::vector<std::string> expected = {
        "group_begin", "allgather", "allgather", "group_end",
        "set_device(0)", "correction(rank=0)",
        "set_device(1)", "correction(rank=1)",
        "group_begin", "allreduce", "allreduce", "group_end",
    };

    ASSERT_EQ(log.size(), expected.size())
        << "Log size mismatch: got " << log.size();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(log[i], expected[i]) << "at index " << i;
    }
}

// INV-KVS-QAG: under sharded KV the combine runs over ALL dcp*HL heads —
// combine_num_heads must size the global_lse buffers and flow into the
// correction kernel's H (and the communicator's per-call head counts).
TEST(DcpAttentionWrapper, CombineNumHeadsPropagatesToCorrection) {
    std::vector<std::string> log;

    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    for (int i = 0; i < 2; ++i) {
        owned_devbacks.push_back(lcomp::make_null_device_backend(
            {.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090}));
        devback_ptrs.push_back(owned_devbacks.back().get());
    }
    auto recording_collective = std::make_unique<RecordingCollectiveBackend>(log);
    auto comm = lp::DcpCommunicator(
        null_comm_opts(2, devback_ptrs, recording_collective.get()));

    auto dev_ptrs = make_null_devs(2);

    std::vector<int> seen_H;
    auto capture_h = [&seen_H](void*, const float*, float*,
                               int, int H, int, int, int, void*) {
        seen_H.push_back(H);
    };

    auto cfg = wrapper_cfg(2);           // num_heads_local = 64
    cfg.combine_num_heads = 128;         // sharded KV: all heads
    auto wrapper = lc::DcpAttentionWrapper(&comm, cfg, dev_ptrs, capture_h);

    void* outputs[2] = {nullptr, nullptr};
    const float* lses[2] = {nullptr, nullptr};
    void* streams[2] = {nullptr, nullptr};
    wrapper.correct_output(outputs, lses, 4, streams);

    ASSERT_EQ(seen_H.size(), 2u);
    EXPECT_EQ(seen_H[0], 128);
    EXPECT_EQ(seen_H[1], 128);

    // Default (combine_num_heads = 0): correction runs at num_heads_local.
    seen_H.clear();
    auto wrapper_legacy = lc::DcpAttentionWrapper(
        &comm, wrapper_cfg(2), dev_ptrs, capture_h);
    wrapper_legacy.correct_output(outputs, lses, 4, streams);
    ASSERT_EQ(seen_H.size(), 2u);
    EXPECT_EQ(seen_H[0], 64);
    EXPECT_EQ(seen_H[1], 64);
}

TEST(DcpAttentionWrapper, ReduceHiddenSequence) {
    std::vector<std::string> log;

    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    for (int i = 0; i < 2; ++i) {
        owned_devbacks.push_back(lcomp::make_null_device_backend(
            {.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090}));
        devback_ptrs.push_back(owned_devbacks.back().get());
    }
    auto recording_collective = std::make_unique<RecordingCollectiveBackend>(log);

    auto comm = lp::DcpCommunicator(
        null_comm_opts(2, devback_ptrs, recording_collective.get()));

    std::vector<std::unique_ptr<WrapperRecordingAttentionDevice>> rec_devs;
    std::vector<lc::AttentionDevice*> dev_ptrs;
    for (int i = 0; i < 2; ++i) {
        rec_devs.push_back(std::make_unique<WrapperRecordingAttentionDevice>(
            layerstorm::config::GpuRef{.position = i, .id = i,
                                        .type = layerstorm::config::GpuType::rtx5090},
            log));
        dev_ptrs.push_back(rec_devs.back().get());
    }

    auto wrapper = lc::DcpAttentionWrapper(
        &comm, wrapper_cfg(2), dev_ptrs, recording_correction(log));

    log.clear();

    void* hiddens[2] = {nullptr, nullptr};
    void* streams[2] = {nullptr, nullptr};
    wrapper.reduce_hidden(hiddens, 4, streams);

    // Expected: group_begin, allreduce, allreduce, group_end
    std::vector<std::string> expected = {
        "group_begin", "allreduce", "allreduce", "group_end",
    };

    ASSERT_EQ(log.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(log[i], expected[i]) << "at index " << i;
    }
}

TEST(DcpAttentionWrapper, PassthroughSkipsCalls) {
    std::vector<std::string> log;

    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    owned_devbacks.push_back(lcomp::make_null_device_backend(
        {.position = 0, .id = 0, .type = layerstorm::config::GpuType::rtx5090}));
    devback_ptrs.push_back(owned_devbacks.back().get());
    auto recording_collective = std::make_unique<RecordingCollectiveBackend>(log);

    auto comm = lp::DcpCommunicator(
        null_comm_opts(1, devback_ptrs, recording_collective.get()));

    std::vector<std::unique_ptr<WrapperRecordingAttentionDevice>> rec_devs;
    std::vector<lc::AttentionDevice*> dev_ptrs;
    rec_devs.push_back(std::make_unique<WrapperRecordingAttentionDevice>(
        layerstorm::config::GpuRef{.position = 0, .id = 0,
                                    .type = layerstorm::config::GpuType::rtx5090},
        log));
    dev_ptrs.push_back(rec_devs.back().get());

    auto wrapper = lc::DcpAttentionWrapper(
        &comm, wrapper_cfg(1), dev_ptrs, recording_correction(log));

    log.clear();

    const float* lse = nullptr;
    void* out = nullptr;
    void* stream = nullptr;
    wrapper.correct_output(&out, &lse, 1, &stream);
    wrapper.reduce_hidden(&out, 1, &stream);

    EXPECT_TRUE(log.empty()) << "Expected no calls for dcp_size=1";
}

// ============================================================================
// Group 3: Multi-GPU correctness (real NCCL + correction kernel)
// ============================================================================

class DcpAttentionWrapperMultiGpu : public ::testing::Test {
protected:
    static constexpr int kN = 2;

    void SetUp() override {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count < kN)
            GTEST_SKIP() << "Need " << kN << " GPUs";
    }
};

// Helper: create SnapMlaSm120AttentionDevice instances for multi-GPU tests.
struct CudaAttentionDevices {
    std::vector<std::unique_ptr<lc::AttentionDevice>> owned;
    std::vector<lc::AttentionDevice*> ptrs;

    explicit CudaAttentionDevices(int count) {
        for (int i = 0; i < count; ++i) {
            owned.push_back(lc::make_snapmla_sm120_attention_device(
                {.position = i, .id = i,
                 .type = layerstorm::config::GpuType::rtx5090}));
            ptrs.push_back(owned.back().get());
        }
    }
};

// CPU reference: same as dcp_allreduce_ref from other tests
static std::vector<float> dcp_ref(
    const std::vector<std::vector<float>>& outs,
    const std::vector<std::vector<float>>& lses,
    int B, int H, int D, int N) {
    std::vector<float> result(B * H * D, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            float mx = -1e30f;
            for (int n = 0; n < N; ++n)
                mx = std::max(mx, lses[n][b * H + h]);
            float se = 0.0f;
            for (int n = 0; n < N; ++n)
                se += std::exp(lses[n][b * H + h] - mx);
            float gl = mx + std::log(se);
            for (int n = 0; n < N; ++n) {
                float s = std::exp(lses[n][b * H + h] - gl);
                for (int d_ = 0; d_ < D; ++d_)
                    result[b * H * D + h * D + d_] += outs[n][b * H * D + h * D + d_] * s;
            }
        }
    }
    return result;
}

TEST_F(DcpAttentionWrapperMultiGpu, TwoGpuCorrectOutput) {
    const int B = 2, H = 4, D = 32;

    // Real CUDA device backends + NCCL collective — must outlive DcpCommunicator.
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    for (int i = 0; i < kN; ++i) {
        owned_devbacks.push_back(lcomp::make_cuda_sm120_device_backend(
            {.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090}));
        devback_ptrs.push_back(owned_devbacks.back().get());
    }
    auto nccl_collective = lp::make_nccl_collective_backend();

    auto comm = lp::DcpCommunicator({
        .dcp_size         = kN,
        .device_backends  = devback_ptrs,
        .max_batch_size   = B,
        .num_heads        = H * kN,
        .attn_output_dim  = D,
        .hidden_size      = 7168,
        .collective       = nccl_collective.get(),
    });

    CudaAttentionDevices cuda_devs(kN);
    auto wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = H, .head_dim = D, .hidden_size = 7168,
         .max_batch_size = B, .gpus = make_gpu_refs(kN)},
        cuda_devs.ptrs, lc::cuda_launch_correction());

    EXPECT_TRUE(wrapper.is_active());

    // Per-rank data: rank0=1.0/lse=2.0, rank1=3.0/lse=1.0
    std::vector<std::vector<float>> h_outs = {
        std::vector<float>(B * H * D, 1.0f),
        std::vector<float>(B * H * D, 3.0f),
    };
    std::vector<std::vector<float>> h_lses = {
        std::vector<float>(B * H, 2.0f),
        std::vector<float>(B * H, 1.0f),
    };
    auto expected = dcp_ref(h_outs, h_lses, B, H, D, kN);

    void*  d_out[kN] = {};
    float* d_lse[kN] = {};
    cudaStream_t streams[kN] = {};
    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamCreate(&streams[r]));
        CUDA_CHECK(cudaMalloc(&d_out[r], B * H * D * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_lse[r], B * H * sizeof(float)));

        std::vector<uint16_t> bf(B * H * D, float_to_bf16(h_outs[r][0]));
        CUDA_CHECK(cudaMemcpy(d_out[r], bf.data(),
                               bf.size() * sizeof(uint16_t),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lse[r], h_lses[r].data(),
                               B * H * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    const float* lse_ptrs[kN] = {d_lse[0], d_lse[1]};
    void* stream_ptrs[kN] = {streams[0], streams[1]};

    wrapper.correct_output(d_out, lse_ptrs, B, stream_ptrs);

    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }

    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        std::vector<uint16_t> result(B * H * D);
        CUDA_CHECK(cudaMemcpy(result.data(), d_out[r],
                               result.size() * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost));
        for (int i = 0; i < B * H * D; ++i) {
            EXPECT_NEAR(bf16_to_float(result[i]), expected[i], 0.15f)
                << "rank=" << r << " i=" << i;
        }
    }

    for (int r = 0; r < kN; ++r) {
        cudaSetDevice(r);
        cudaFree(d_out[r]);
        cudaFree(d_lse[r]);
        cudaStreamDestroy(streams[r]);
    }
}

TEST_F(DcpAttentionWrapperMultiGpu, TwoGpuReduceHidden) {
    const int B = 2, hidden = 7168;

    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    for (int i = 0; i < kN; ++i) {
        owned_devbacks.push_back(lcomp::make_cuda_sm120_device_backend(
            {.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090}));
        devback_ptrs.push_back(owned_devbacks.back().get());
    }
    auto nccl_collective = lp::make_nccl_collective_backend();

    auto comm = lp::DcpCommunicator({
        .dcp_size         = kN,
        .device_backends  = devback_ptrs,
        .max_batch_size   = B,
        .num_heads        = 128,
        .attn_output_dim  = 512,
        .hidden_size      = hidden,
        .collective       = nccl_collective.get(),
    });

    CudaAttentionDevices cuda_devs(kN);
    auto wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = 64, .head_dim = 512, .hidden_size = hidden,
         .max_batch_size = B, .gpus = make_gpu_refs(kN)},
        cuda_devs.ptrs, lc::cuda_launch_correction());

    const size_t elems = B * hidden;

    std::vector<uint16_t> h0(elems, float_to_bf16(2.0f));
    std::vector<uint16_t> h1(elems, float_to_bf16(5.0f));

    void* d_hidden[kN] = {};
    cudaStream_t streams[kN] = {};
    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamCreate(&streams[r]));
        CUDA_CHECK(cudaMalloc(&d_hidden[r], elems * sizeof(uint16_t)));
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(d_hidden[0], h0.data(),
                           elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMemcpy(d_hidden[1], h1.data(),
                           elems * sizeof(uint16_t), cudaMemcpyHostToDevice));

    void* stream_ptrs[kN] = {streams[0], streams[1]};
    wrapper.reduce_hidden(d_hidden, B, stream_ptrs);

    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }

    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        std::vector<uint16_t> result(elems);
        CUDA_CHECK(cudaMemcpy(result.data(), d_hidden[r],
                               elems * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost));
        for (int i = 0; i < 64; ++i) {
            EXPECT_NEAR(bf16_to_float(result[i]), 7.0f, 0.15f)
                << "r=" << r << " i=" << i;
        }
    }

    for (int r = 0; r < kN; ++r) {
        cudaSetDevice(r);
        cudaFree(d_hidden[r]);
        cudaStreamDestroy(streams[r]);
    }
}

TEST_F(DcpAttentionWrapperMultiGpu, TwoGpuAsymmetricLse) {
    const int B = 1, H = 4, D = 16;

    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devbacks;
    std::vector<lcomp::DeviceBackend*> devback_ptrs;
    for (int i = 0; i < kN; ++i) {
        owned_devbacks.push_back(lcomp::make_cuda_sm120_device_backend(
            {.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090}));
        devback_ptrs.push_back(owned_devbacks.back().get());
    }
    auto nccl_collective = lp::make_nccl_collective_backend();

    auto comm = lp::DcpCommunicator({
        .dcp_size         = kN,
        .device_backends  = devback_ptrs,
        .max_batch_size   = B,
        .num_heads        = H * kN,
        .attn_output_dim  = D,
        .hidden_size      = 7168,
        .collective       = nccl_collective.get(),
    });

    CudaAttentionDevices cuda_devs(kN);
    auto wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = H, .head_dim = D, .hidden_size = 7168,
         .max_batch_size = B, .gpus = make_gpu_refs(kN)},
        cuda_devs.ptrs, lc::cuda_launch_correction());

    // rank0: out=1.0, lse=50.0 (dominates)
    // rank1: out=100.0, lse=-50.0 (negligible)
    std::vector<std::vector<float>> h_outs = {
        std::vector<float>(B * H * D, 1.0f),
        std::vector<float>(B * H * D, 100.0f),
    };
    std::vector<std::vector<float>> h_lses = {
        std::vector<float>(B * H, 50.0f),
        std::vector<float>(B * H, -50.0f),
    };
    auto expected = dcp_ref(h_outs, h_lses, B, H, D, kN);

    void*  d_out[kN] = {};
    float* d_lse[kN] = {};
    cudaStream_t streams[kN] = {};
    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamCreate(&streams[r]));
        CUDA_CHECK(cudaMalloc(&d_out[r], B * H * D * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_lse[r], B * H * sizeof(float)));

        std::vector<uint16_t> bf(B * H * D, float_to_bf16(h_outs[r][0]));
        CUDA_CHECK(cudaMemcpy(d_out[r], bf.data(),
                               bf.size() * sizeof(uint16_t),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lse[r], h_lses[r].data(),
                               B * H * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    const float* lse_ptrs[kN] = {d_lse[0], d_lse[1]};
    void* stream_ptrs[kN] = {streams[0], streams[1]};
    wrapper.correct_output(d_out, lse_ptrs, B, stream_ptrs);

    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }

    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        std::vector<uint16_t> result(B * H * D);
        CUDA_CHECK(cudaMemcpy(result.data(), d_out[r],
                               result.size() * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost));
        for (int i = 0; i < B * H * D; ++i) {
            float val = bf16_to_float(result[i]);
            EXPECT_FALSE(std::isnan(val)) << "NaN r=" << r << " i=" << i;
            EXPECT_FALSE(std::isinf(val)) << "Inf r=" << r << " i=" << i;
            EXPECT_NEAR(val, expected[i], 0.1f) << "r=" << r << " i=" << i;
        }
    }

    for (int r = 0; r < kN; ++r) {
        cudaSetDevice(r);
        cudaFree(d_out[r]);
        cudaFree(d_lse[r]);
        cudaStreamDestroy(streams[r]);
    }
}
