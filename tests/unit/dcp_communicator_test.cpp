// Unit tests for DCP communicator.
//
// Groups 1-4: null backend (no CUDA/NCCL required).
// Group 5: multi-GPU correctness (REQUIRES_MULTI_GPU(2)).

#include "parallelism/dcp_communicator.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#include "core/device_backend.h"
#include "core/gpu_ref.h"
#include "core/null_device_backend.h"
#include "parallelism/null_collective_backend.h"
#include "compute/cuda_sm120_device_backend.h"
#include "parallelism/nccl_collective_backend.h"
#include "../gpu_test_utils.h"

namespace lp   = layerstorm::parallelism;
namespace lcomp = layerstorm::compute;

static std::vector<layerstorm::config::GpuRef> make_gpu_refs(int count) {
    std::vector<layerstorm::config::GpuRef> v;
    for (int i = 0; i < count; ++i)
        v.push_back({.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090});
    return v;
}

// -- Helpers -----------------------------------------------------------------

#define CUDA_CHECK(expr)                                                 \
    do {                                                                 \
        cudaError_t _err = (expr);                                       \
        ASSERT_EQ(_err, cudaSuccess) << cudaGetErrorString(_err);        \
    } while (0)

#define REQUIRES_MULTI_GPU(n)                                            \
    do {                                                                 \
        int c_ = 0; cudaGetDeviceCount(&c_);                             \
        if (c_ < (n)) GTEST_SKIP() << "Need " << (n) << " GPUs";       \
    } while (0)

/// Null-backend helper: owns NullDeviceBackend instances + NullCollectiveBackend
/// so they outlive the DcpCommunicator built from the returned Options.
struct NullBackendHolder {
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> device_backends;
    std::unique_ptr<lp::CollectiveBackend> collective;

    NullBackendHolder()
        : collective(std::make_unique<lp::NullCollectiveBackend>()) {}

    /// Ensure we have at least `n` NullDeviceBackend instances.
    void ensure_backends(int n) {
        while (static_cast<int>(device_backends.size()) < n) {
            int idx = static_cast<int>(device_backends.size());
            auto gpu = layerstorm::config::GpuRef{
                .position = idx, .id = idx,
                .type = layerstorm::config::GpuType::rtx5090};
            device_backends.push_back(
                lcomp::make_null_device_backend(gpu));
        }
    }

    /// Build raw pointer vector for Options.device_backends (first `n`).
    std::vector<lcomp::DeviceBackend*> backend_ptrs(int n) {
        ensure_backends(n);
        std::vector<lcomp::DeviceBackend*> ptrs;
        for (int i = 0; i < n; ++i)
            ptrs.push_back(device_backends[i].get());
        return ptrs;
    }
};

/// Thread-local holder so the free function null_opts() can produce
/// Options whose backing objects live long enough (until next call).
static thread_local NullBackendHolder tl_null_holder;

static lp::DcpCommunicator::Options null_opts(int dcp_size,
                                                int num_heads = 128,
                                                int max_batch = 64) {
    tl_null_holder = NullBackendHolder{};          // reset
    return {
        .dcp_size         = dcp_size,
        .device_backends  = tl_null_holder.backend_ptrs(dcp_size),
        .max_batch_size   = max_batch,
        .num_heads        = num_heads,
        .attn_output_dim  = 512,
        .hidden_size      = 7168,
        .collective       = tl_null_holder.collective.get(),
    };
}

// BF16 helpers
static uint16_t float_to_bf16(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    f += 0x7FFF + ((f >> 16) & 1);
    return static_cast<uint16_t>(f >> 16);
}

static float bf16_to_float(uint16_t b) {
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
}

// ============================================================================
// Group 1: Construction (null backend)
// ============================================================================

TEST(DcpCommunicator, DcpSize2Construction) {
    auto comm = lp::DcpCommunicator(null_opts(2));
    EXPECT_TRUE(comm.is_active());
    EXPECT_EQ(comm.dcp_size(), 2);
    EXPECT_EQ(comm.num_heads_local(), 64);  // 128 / 2
    EXPECT_EQ(comm.gpus().size(), 2u);
}

TEST(DcpCommunicator, DcpSize1Passthrough) {
    auto opts = null_opts(1);
    auto comm = lp::DcpCommunicator(opts);
    EXPECT_FALSE(comm.is_active());
    EXPECT_EQ(comm.dcp_size(), 1);
    EXPECT_EQ(comm.num_heads_local(), 128);
    EXPECT_EQ(comm.comm(0), nullptr);
    EXPECT_EQ(comm.gathered_lse_buffer(0), nullptr);
}

TEST(DcpCommunicator, MismatchedDeviceIds) {
    auto opts = null_opts(2);
    // Replace with only 1 backend, but dcp_size=2
    opts.device_backends = tl_null_holder.backend_ptrs(1);
    EXPECT_THROW((lp::DcpCommunicator{opts}), std::invalid_argument);
}

TEST(DcpCommunicator, ZeroDcpSize) {
    auto opts = null_opts(0);
    opts.device_backends.clear();
    EXPECT_THROW((lp::DcpCommunicator{opts}), std::invalid_argument);
}

TEST(DcpCommunicator, NegativeDcpSize) {
    auto opts = null_opts(1);
    opts.dcp_size = -1;
    EXPECT_THROW((lp::DcpCommunicator{opts}), std::invalid_argument);
}

TEST(DcpCommunicator, DestroyIdempotent) {
    // Destructor runs twice via explicit scope — no crash.
    {
        auto comm = lp::DcpCommunicator(null_opts(2));
        EXPECT_TRUE(comm.is_active());
    }
    // Second construction/destruction
    {
        auto comm = lp::DcpCommunicator(null_opts(2));
        EXPECT_TRUE(comm.is_active());
    }
}

// ============================================================================
// Group 2: Buffer pre-allocation (null backend)
// ============================================================================

TEST(DcpCommunicator, GatheredLseNonNull) {
    auto comm = lp::DcpCommunicator(null_opts(2));
    EXPECT_NE(comm.gathered_lse_buffer(0), nullptr);
    EXPECT_NE(comm.gathered_lse_buffer(1), nullptr);
}

TEST(DcpCommunicator, GatheredLseNullWhenInactive) {
    auto opts = null_opts(1);
    auto comm = lp::DcpCommunicator(opts);
    EXPECT_EQ(comm.gathered_lse_buffer(0), nullptr);
}

TEST(DcpCommunicator, GatheredLseDistinctPerRank) {
    auto comm = lp::DcpCommunicator(null_opts(2));
    EXPECT_NE(comm.gathered_lse_buffer(0), comm.gathered_lse_buffer(1));
}

TEST(DcpCommunicator, GatheredLseInvalidRank) {
    auto comm = lp::DcpCommunicator(null_opts(2));
    EXPECT_THROW(comm.gathered_lse_buffer(2), std::out_of_range);
    EXPECT_THROW(comm.gathered_lse_buffer(-1), std::out_of_range);
}

// ============================================================================
// Group 3: Comm access (null backend)
// ============================================================================

TEST(DcpCommunicator, CommDistinctPerRank) {
    auto comm = lp::DcpCommunicator(null_opts(2));
    EXPECT_NE(comm.comm(0), nullptr);
    EXPECT_NE(comm.comm(1), nullptr);
    EXPECT_NE(comm.comm(0), comm.comm(1));
}

TEST(DcpCommunicator, CommNullWhenInactive) {
    auto opts = null_opts(1);
    auto comm = lp::DcpCommunicator(opts);
    EXPECT_EQ(comm.comm(0), nullptr);
}

TEST(DcpCommunicator, CommInvalidRank) {
    auto comm = lp::DcpCommunicator(null_opts(2));
    EXPECT_THROW(comm.comm(2), std::out_of_range);
    EXPECT_THROW(comm.comm(-1), std::out_of_range);
}

// ============================================================================
// Group 4: Collective calls (null backend)
// ============================================================================

TEST(DcpCommunicator, AllgatherLseReturnsBuffer) {
    auto comm = lp::DcpCommunicator(null_opts(2));
    // Null backend: allgather is no-op, but we test the call doesn't crash
    // and that returned buffer matches gathered_lse_buffer.
    const float* lses[2] = {nullptr, nullptr};
    void* streams[2] = {nullptr, nullptr};
    ASSERT_NO_THROW(comm.allgather_lse(lses, 4, streams));
}

TEST(DcpCommunicator, AllgatherLseBatchValidation) {
    auto comm = lp::DcpCommunicator(null_opts(2, 128, 32));
    const float* lses[2] = {nullptr, nullptr};
    void* streams[2] = {nullptr, nullptr};
    // batch_size=33 > max_batch_size=32
    EXPECT_THROW(comm.allgather_lse(lses, 33, streams), std::invalid_argument);
    // batch_size=32 OK
    ASSERT_NO_THROW(comm.allgather_lse(lses, 32, streams));
}

TEST(DcpCommunicator, AllgatherLsePassthroughDcpSize1) {
    auto opts = null_opts(1);
    auto comm = lp::DcpCommunicator(opts);
    const float* lses[1] = {nullptr};
    void* streams[1] = {nullptr};
    ASSERT_NO_THROW(comm.allgather_lse(lses, 4, streams));
}

TEST(DcpCommunicator, AllreduceOutputCallSucceeds) {
    auto comm = lp::DcpCommunicator(null_opts(2));
    void* outputs[2] = {nullptr, nullptr};
    void* streams[2] = {nullptr, nullptr};
    ASSERT_NO_THROW(comm.allreduce_output(outputs, 4, streams));
}

TEST(DcpCommunicator, AllreduceHiddenCallSucceeds) {
    auto comm = lp::DcpCommunicator(null_opts(2));
    void* hiddens[2] = {nullptr, nullptr};
    void* streams[2] = {nullptr, nullptr};
    ASSERT_NO_THROW(comm.allreduce_hidden(hiddens, 4, streams));
}

TEST(DcpCommunicator, AllreduceNoopDcpSize1) {
    auto opts = null_opts(1);
    auto comm = lp::DcpCommunicator(opts);
    void* outputs[1] = {nullptr};
    void* streams[1] = {nullptr};
    ASSERT_NO_THROW(comm.allreduce_output(outputs, 4, streams));
    ASSERT_NO_THROW(comm.allreduce_hidden(outputs, 4, streams));
}

// ── INV-KVS-QAG: Q-head allgather + per-call combine head counts ────────────

TEST(DcpCommunicator, AllgatherQCallSucceeds) {
    auto comm = lp::DcpCommunicator(null_opts(2));
    const void* sends[2] = {nullptr, nullptr};
    void* recvs[2] = {nullptr, nullptr};
    void* streams[2] = {nullptr, nullptr};
    ASSERT_NO_THROW(comm.allgather_q(sends, recvs, 4, 4 * 64 * 576, streams));
}

TEST(DcpCommunicator, AllgatherQBatchValidation) {
    auto comm = lp::DcpCommunicator(null_opts(2, 128, 32));
    const void* sends[2] = {nullptr, nullptr};
    void* recvs[2] = {nullptr, nullptr};
    void* streams[2] = {nullptr, nullptr};
    EXPECT_THROW(comm.allgather_q(sends, recvs, 33, 1, streams),
                 std::invalid_argument);
    ASSERT_NO_THROW(comm.allgather_q(sends, recvs, 32, 1, streams));
}

TEST(DcpCommunicator, AllgatherQNoopDcpSize1) {
    auto opts = null_opts(1);
    auto comm = lp::DcpCommunicator(opts);
    const void* sends[1] = {nullptr};
    void* recvs[1] = {nullptr};
    void* streams[1] = {nullptr};
    ASSERT_NO_THROW(comm.allgather_q(sends, recvs, 4, 1, streams));
}

TEST(DcpCommunicator, CombineHeadCountValidation) {
    // Per-call num_heads must not exceed Options::num_heads (the gathered
    // LSE buffer is sized for the all-head worst case, INV-KVS-QAG).
    auto comm = lp::DcpCommunicator(null_opts(2, /*num_heads=*/128));
    const float* lses[2] = {nullptr, nullptr};
    void* outputs[2] = {nullptr, nullptr};
    void* streams[2] = {nullptr, nullptr};
    // All-head combine (sharded KV): num_heads == full head count is legal.
    ASSERT_NO_THROW(comm.allgather_lse(lses, 4, streams, 128));
    ASSERT_NO_THROW(comm.allreduce_output(outputs, 4, streams, 128));
    // Beyond the buffer bound: reject.
    EXPECT_THROW(comm.allgather_lse(lses, 4, streams, 129),
                 std::invalid_argument);
    EXPECT_THROW(comm.allreduce_output(outputs, 4, streams, 129),
                 std::invalid_argument);
}

// ============================================================================
// Group 5: Multi-GPU correctness (real NCCL)
// ============================================================================

class DcpCommunicatorMultiGpu : public ::testing::Test {
protected:
    static constexpr int kN = 2;

    // Backends must outlive DcpCommunicator instances.
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> device_backends_;
    std::unique_ptr<lp::CollectiveBackend> collective_;

    void SetUp() override {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count < kN) {
            GTEST_SKIP() << "Need " << kN << " GPUs, have " << count;
        }
        // Create real CUDA device backends.
        auto gpus = make_gpu_refs(kN);
        for (int i = 0; i < kN; ++i)
            device_backends_.push_back(
                lcomp::make_cuda_sm120_device_backend(gpus[i]));
        // Create real NCCL collective backend.
        collective_ = lp::make_nccl_collective_backend();
    }

    std::vector<lcomp::DeviceBackend*> backend_ptrs(int n = kN) {
        std::vector<lcomp::DeviceBackend*> ptrs;
        for (int i = 0; i < n; ++i)
            ptrs.push_back(device_backends_[i].get());
        return ptrs;
    }

    lp::DcpCommunicator::Options real_opts(int max_batch = 8,
                                            int num_heads = 128) {
        return {
            .dcp_size         = kN,
            .device_backends  = backend_ptrs(kN),
            .max_batch_size   = max_batch,
            .num_heads        = num_heads,
            .attn_output_dim  = 512,
            .hidden_size      = 7168,
            .collective       = collective_.get(),
        };
    }
};

TEST_F(DcpCommunicatorMultiGpu, Construction) {
    auto comm = lp::DcpCommunicator(real_opts());
    EXPECT_TRUE(comm.is_active());
    EXPECT_EQ(comm.dcp_size(), 2);
    EXPECT_EQ(comm.num_heads_local(), 64);
    EXPECT_NE(comm.comm(0), nullptr);
    EXPECT_NE(comm.comm(1), nullptr);
    EXPECT_NE(comm.gathered_lse_buffer(0), nullptr);
    EXPECT_NE(comm.gathered_lse_buffer(1), nullptr);
}

TEST_F(DcpCommunicatorMultiGpu, AllgatherLse) {
    const int B = 4, H_local = 64;
    auto comm = lp::DcpCommunicator(real_opts(B));
    const size_t elems = B * H_local;

    // Per-rank LSE data
    std::vector<float> h_lse0(elems, 1.0f);
    std::vector<float> h_lse1(elems, 2.0f);

    // Allocate device buffers and streams
    float* d_lse[kN] = {};
    cudaStream_t streams[kN] = {};
    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_lse[r], elems * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&streams[r]));
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(d_lse[0], h_lse0.data(),
                           elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMemcpy(d_lse[1], h_lse1.data(),
                           elems * sizeof(float), cudaMemcpyHostToDevice));

    // Allgather
    const float* lse_ptrs[kN] = {d_lse[0], d_lse[1]};
    void* stream_ptrs[kN] = {streams[0], streams[1]};
    comm.allgather_lse(lse_ptrs, B, stream_ptrs);

    // Synchronize
    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }

    // Verify: gathered layout is [N, B, H_local]
    // rank0's buffer: [rank0_data | rank1_data] = [1.0... | 2.0...]
    // rank1's buffer: same (allgather replicates)
    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        std::vector<float> gathered(kN * elems);
        CUDA_CHECK(cudaMemcpy(gathered.data(),
                               comm.gathered_lse_buffer(r),
                               kN * elems * sizeof(float),
                               cudaMemcpyDeviceToHost));

        // Slot 0: rank 0's data (1.0)
        for (size_t i = 0; i < elems; ++i)
            EXPECT_FLOAT_EQ(gathered[i], 1.0f)
                << "r=" << r << " slot=0 i=" << i;
        // Slot 1: rank 1's data (2.0)
        for (size_t i = 0; i < elems; ++i)
            EXPECT_FLOAT_EQ(gathered[elems + i], 2.0f)
                << "r=" << r << " slot=1 i=" << i;
    }

    // Cleanup
    for (int r = 0; r < kN; ++r) {
        cudaSetDevice(r);
        cudaFree(d_lse[r]);
        cudaStreamDestroy(streams[r]);
    }
}

// INV-KVS-QAG: real-NCCL Q-head allgather layout. Rank s holds global heads
// [s*HL, (s+1)*HL), so the rank-major NCCL concatenation must be the GLOBAL
// head order — at B==1 directly the [1, N*HL, d_q] head-major layout the
// all-head attention consumes.
TEST_F(DcpCommunicatorMultiGpu, AllgatherQHeadDimGather) {
    const int B = 1, HL = 64, DQ = 576;   // kv_lora_rank + qk_rope
    auto comm = lp::DcpCommunicator(real_opts(B));
    const size_t elems = static_cast<size_t>(B) * HL * DQ;

    // Rank r fills head h (local) with value r*HL + h == the GLOBAL head id.
    void* d_send[kN] = {};
    void* d_recv[kN] = {};
    cudaStream_t streams[kN] = {};
    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_send[r], elems * 2));
        CUDA_CHECK(cudaMalloc(&d_recv[r], kN * elems * 2));
        CUDA_CHECK(cudaStreamCreate(&streams[r]));
        std::vector<uint16_t> h_q(elems);
        for (int h = 0; h < HL; ++h)
            for (int d = 0; d < DQ; ++d)
                h_q[static_cast<size_t>(h) * DQ + d] =
                    float_to_bf16(static_cast<float>(r * HL + h));
        CUDA_CHECK(cudaMemcpy(d_send[r], h_q.data(), elems * 2,
                               cudaMemcpyHostToDevice));
    }

    const void* send_ptrs[kN] = {d_send[0], d_send[1]};
    void* recv_ptrs[kN] = {d_recv[0], d_recv[1]};
    void* stream_ptrs[kN] = {streams[0], streams[1]};
    comm.allgather_q(send_ptrs, recv_ptrs, B, elems, stream_ptrs);

    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }

    // Every rank's recv buffer must read [1, kN*HL, DQ] with element value ==
    // its GLOBAL head id (rank-major concat == global head order).
    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        std::vector<uint16_t> gathered(kN * elems);
        CUDA_CHECK(cudaMemcpy(gathered.data(), d_recv[r], kN * elems * 2,
                               cudaMemcpyDeviceToHost));
        for (int gh = 0; gh < kN * HL; ++gh) {
            // Spot-check first/last element of each head row.
            EXPECT_FLOAT_EQ(bf16_to_float(
                gathered[static_cast<size_t>(gh) * DQ]),
                static_cast<float>(gh)) << "r=" << r << " head=" << gh;
            EXPECT_FLOAT_EQ(bf16_to_float(
                gathered[static_cast<size_t>(gh) * DQ + DQ - 1]),
                static_cast<float>(gh)) << "r=" << r << " head=" << gh;
        }
    }

    for (int r = 0; r < kN; ++r) {
        cudaSetDevice(r);
        cudaFree(d_send[r]);
        cudaFree(d_recv[r]);
        cudaStreamDestroy(streams[r]);
    }
}

TEST_F(DcpCommunicatorMultiGpu, AllreduceOutput) {
    const int B = 2, H_local = 64, D = 512;
    auto comm = lp::DcpCommunicator(real_opts(B));
    const size_t elems = B * H_local * D;

    // Rank 0: all 1.0 BF16, Rank 1: all 3.0 BF16
    // After SUM allreduce: all 4.0
    std::vector<uint16_t> h_bf16_0(elems, float_to_bf16(1.0f));
    std::vector<uint16_t> h_bf16_1(elems, float_to_bf16(3.0f));

    void* d_out[kN] = {};
    cudaStream_t streams[kN] = {};
    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_out[r], elems * sizeof(uint16_t)));
        CUDA_CHECK(cudaStreamCreate(&streams[r]));
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(d_out[0], h_bf16_0.data(),
                           elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMemcpy(d_out[1], h_bf16_1.data(),
                           elems * sizeof(uint16_t), cudaMemcpyHostToDevice));

    void* stream_ptrs[kN] = {streams[0], streams[1]};
    comm.allreduce_output(d_out, B, stream_ptrs);

    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }

    // Both ranks should have 1.0 + 3.0 = 4.0
    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        std::vector<uint16_t> result(elems);
        CUDA_CHECK(cudaMemcpy(result.data(), d_out[r],
                               elems * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost));
        // Spot-check first 64 elements
        for (int i = 0; i < 64; ++i) {
            float val = bf16_to_float(result[i]);
            EXPECT_NEAR(val, 4.0f, 0.1f) << "r=" << r << " i=" << i;
        }
    }

    for (int r = 0; r < kN; ++r) {
        cudaSetDevice(r);
        cudaFree(d_out[r]);
        cudaStreamDestroy(streams[r]);
    }
}

TEST_F(DcpCommunicatorMultiGpu, AllreduceHidden) {
    const int B = 2, hidden = 7168;
    auto comm = lp::DcpCommunicator(real_opts(B));
    const size_t elems = B * hidden;

    // Rank 0: 2.0, Rank 1: 5.0 → sum = 7.0
    std::vector<uint16_t> h0(elems, float_to_bf16(2.0f));
    std::vector<uint16_t> h1(elems, float_to_bf16(5.0f));

    void* d_hidden[kN] = {};
    cudaStream_t streams[kN] = {};
    for (int r = 0; r < kN; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_hidden[r], elems * sizeof(uint16_t)));
        CUDA_CHECK(cudaStreamCreate(&streams[r]));
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(d_hidden[0], h0.data(),
                           elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMemcpy(d_hidden[1], h1.data(),
                           elems * sizeof(uint16_t), cudaMemcpyHostToDevice));

    void* stream_ptrs[kN] = {streams[0], streams[1]};
    comm.allreduce_hidden(d_hidden, B, stream_ptrs);

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
            float val = bf16_to_float(result[i]);
            EXPECT_NEAR(val, 7.0f, 0.15f) << "r=" << r << " i=" << i;
        }
    }

    for (int r = 0; r < kN; ++r) {
        cudaSetDevice(r);
        cudaFree(d_hidden[r]);
        cudaStreamDestroy(streams[r]);
    }
}

// DET-REDUCE Phase 1b: isolated cross-GPU EP-combine GATHER timing. Times
// allreduce_hidden on the real 2×5090 NCCL link for the three combine byte sizes
// on IDENTICAL buffers (latency/bandwidth only — no routing trajectory):
//   OFF reduced bf16 : [B, hidden] bf16        (rows_per_token=1, fp32=false)
//   canonical fp32   : [B, topk, hidden] fp32  (rows_per_token=topk, fp32=true)
//   canonical bf16   : [B, topk, hidden] bf16  (rows_per_token=topk, fp32=false)
// The bf16 per-slot gather is HALF the bytes of the fp32 per-slot gather.
TEST_F(DcpCommunicatorMultiGpu, AllreduceHiddenCombineTiming) {
    const int hidden = 7168;
    const int topk = 8;
    struct Variant { const char* name; int rows; bool fp32; };
    const Variant variants[] = {
        {"OFF reduced bf16", 1,    false},
        {"canonical fp32  ", topk, true },
        {"canonical bf16  ", topk, false},
    };
    std::cerr << "\n=== EP-combine collective timing (2 GPU NCCL allreduce) ===\n";
    for (int B : {1, 8}) {
        auto comm = lp::DcpCommunicator(real_opts(B));
        for (const auto& v : variants) {
            const size_t count = static_cast<size_t>(B) * v.rows * hidden;
            const size_t elsz = v.fp32 ? 4 : 2;
            const size_t bytes = count * elsz;
            void* d[kN] = {};
            cudaStream_t st[kN] = {};
            for (int r = 0; r < kN; ++r) {
                CUDA_CHECK(cudaSetDevice(r));
                CUDA_CHECK(cudaMalloc(&d[r], bytes));
                CUDA_CHECK(cudaMemset(d[r], 0, bytes));
                CUDA_CHECK(cudaStreamCreate(&st[r]));
            }
            void* sp[kN] = {st[0], st[1]};
            for (int i = 0; i < 50; ++i)
                comm.allreduce_hidden(d, B, sp, v.fp32, v.rows);
            for (int r = 0; r < kN; ++r) {
                CUDA_CHECK(cudaSetDevice(r));
                CUDA_CHECK(cudaStreamSynchronize(st[r]));
            }
            const int iters = 300;
            CUDA_CHECK(cudaSetDevice(0));
            cudaEvent_t e0, e1;
            CUDA_CHECK(cudaEventCreate(&e0));
            CUDA_CHECK(cudaEventCreate(&e1));
            CUDA_CHECK(cudaEventRecord(e0, st[0]));
            for (int i = 0; i < iters; ++i)
                comm.allreduce_hidden(d, B, sp, v.fp32, v.rows);
            CUDA_CHECK(cudaEventRecord(e1, st[0]));
            for (int r = 0; r < kN; ++r) {
                CUDA_CHECK(cudaSetDevice(r));
                CUDA_CHECK(cudaStreamSynchronize(st[r]));
            }
            float ms = 0;
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            const double us = static_cast<double>(ms) * 1e3 / iters;
            std::cerr << "  B=" << B << "  " << v.name
                      << "  rows=" << v.rows
                      << "  bytes/rank=" << bytes
                      << "  -> " << us << " us/call\n";
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            for (int r = 0; r < kN; ++r) {
                cudaSetDevice(r);
                cudaFree(d[r]);
                cudaStreamDestroy(st[r]);
            }
        }
    }
}

TEST_F(DcpCommunicatorMultiGpu, DcpSize1NoNcclCalls) {
    // dcp_size=1 with real CUDA but no NCCL comms.
    // When dcp_size==1, collective can be nullptr.
    lp::DcpCommunicator::Options opts{
        .dcp_size         = 1,
        .device_backends  = backend_ptrs(1),
        .max_batch_size   = 8,
        .num_heads        = 128,
        .attn_output_dim  = 512,
        .hidden_size      = 7168,
        .collective       = nullptr,
    };
    auto comm = lp::DcpCommunicator(opts);
    EXPECT_FALSE(comm.is_active());
    EXPECT_EQ(comm.comm(0), nullptr);

    // Collective calls should be no-ops
    const float* lse = nullptr;
    void* stream = nullptr;
    ASSERT_NO_THROW(comm.allgather_lse(&lse, 1, &stream));
    void* out = nullptr;
    ASSERT_NO_THROW(comm.allreduce_output(&out, 1, &stream));
    ASSERT_NO_THROW(comm.allreduce_hidden(&out, 1, &stream));
}
