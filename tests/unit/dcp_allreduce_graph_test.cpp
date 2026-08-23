// Unit tests for DCP allreduce CUDA graph runner.
//
// Tests: config validation, graph capture/replay, multi-GPU correctness,
// numerical stability, registry integration.
//
// Multi-GPU tests use ncclCommInitRank + threads (one thread per rank)
// to mimic the production multi-process NCCL setup.

#include "compute/graphs/dcp_allreduce_graph.h"
#include "compute/graphs/graph_registry.h"
#include "compute/cuda_sm120_device_backend.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;

// -- CUDA error checking -----------------------------------------------------

#define CUDA_CHECK(expr)                                                 \
    do {                                                                 \
        cudaError_t _err = (expr);                                       \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_err); \
    } while (0)

// -- Host-side BF16 conversion -----------------------------------------------

static uint16_t float_to_bf16_bits(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint32_t lsb = (f >> 16) & 1;
    f += 0x7FFF + lsb;
    return static_cast<uint16_t>(f >> 16);
}

static float bf16_bits_to_float(uint16_t bits) {
    uint32_t f = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

// -- BF16 device buffer helpers ----------------------------------------------

static void* alloc_bf16_device(int device, const std::vector<float>& host) {
    std::vector<uint16_t> bf16(host.size());
    for (size_t i = 0; i < host.size(); ++i)
        bf16[i] = float_to_bf16_bits(host[i]);
    cudaSetDevice(device);
    void* d_ptr = nullptr;
    cudaMalloc(&d_ptr, bf16.size() * sizeof(uint16_t));
    cudaMemcpy(d_ptr, bf16.data(), bf16.size() * sizeof(uint16_t),
               cudaMemcpyHostToDevice);
    return d_ptr;
}

static std::vector<float> read_bf16_device(int device, void* d_ptr,
                                            size_t count) {
    cudaSetDevice(device);
    std::vector<uint16_t> bf16(count);
    cudaMemcpy(bf16.data(), d_ptr, count * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i)
        result[i] = bf16_bits_to_float(bf16[i]);
    return result;
}

static void write_bf16_device(int device, void* d_ptr,
                               const std::vector<float>& host) {
    std::vector<uint16_t> bf16(host.size());
    for (size_t i = 0; i < host.size(); ++i)
        bf16[i] = float_to_bf16_bits(host[i]);
    cudaSetDevice(device);
    cudaMemcpy(d_ptr, bf16.data(), bf16.size() * sizeof(uint16_t),
               cudaMemcpyHostToDevice);
}

// -- CPU reference for DCP correction + allreduce ----------------------------

static std::vector<float> dcp_allreduce_ref(
    const std::vector<std::vector<float>>& rank_outputs,  // per-rank [B*H*D]
    const std::vector<std::vector<float>>& rank_lses,     // per-rank [B*H]
    int B, int H, int D, int N) {

    std::vector<float> result(B * H * D, 0.0f);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            // Global LSE = logsumexp over all ranks
            float max_lse = -1e30f;
            for (int n = 0; n < N; ++n)
                max_lse = std::max(max_lse, rank_lses[n][b * H + h]);
            float sum_exp = 0.0f;
            for (int n = 0; n < N; ++n)
                sum_exp += std::exp(rank_lses[n][b * H + h] - max_lse);
            float global_lse = max_lse + std::log(sum_exp);

            // Correct each rank's output and sum
            for (int n = 0; n < N; ++n) {
                float scale = std::exp(rank_lses[n][b * H + h] - global_lse);
                for (int d_idx = 0; d_idx < D; ++d_idx) {
                    int idx = b * H * D + h * D + d_idx;
                    result[idx] += rank_outputs[n][idx] * scale;
                }
            }
        }
    }
    return result;
}

// ============================================================================
// Config validation tests (no GPU required)
// ============================================================================

TEST(DcpAllreduceGraph, DcpSize1Rejected) {
    lc::DcpAllreduceGraphRunner runner;
    lc::DcpAllreduceConfig cfg{};
    cfg.batch_size = 1;
    cfg.num_heads  = 4;
    cfg.head_dim   = 32;
    cfg.dcp_size   = 1;
    cfg.rank       = 0;
    cfg.nccl_comm      = reinterpret_cast<void*>(0x1);  // dummy
    cfg.partial_output = reinterpret_cast<void*>(0x1);
    cfg.partial_lse    = reinterpret_cast<float*>(0x1);
    EXPECT_THROW(runner.init(cfg, nullptr, nullptr), std::invalid_argument);
}

TEST(DcpAllreduceGraph, DcpSize0Rejected) {
    lc::DcpAllreduceGraphRunner runner;
    lc::DcpAllreduceConfig cfg{};
    cfg.dcp_size = 0;
    cfg.nccl_comm      = reinterpret_cast<void*>(0x1);
    cfg.partial_output = reinterpret_cast<void*>(0x1);
    cfg.partial_lse    = reinterpret_cast<float*>(0x1);
    EXPECT_THROW(runner.init(cfg, nullptr, nullptr), std::invalid_argument);
}

TEST(DcpAllreduceGraph, NullCommRejected) {
    lc::DcpAllreduceGraphRunner runner;
    lc::DcpAllreduceConfig cfg{};
    cfg.dcp_size       = 2;
    cfg.nccl_comm      = nullptr;
    cfg.partial_output = reinterpret_cast<void*>(0x1);
    cfg.partial_lse    = reinterpret_cast<float*>(0x1);
    EXPECT_THROW(runner.init(cfg, nullptr, nullptr), std::invalid_argument);
}

TEST(DcpAllreduceGraph, NullOutputRejected) {
    lc::DcpAllreduceGraphRunner runner;
    lc::DcpAllreduceConfig cfg{};
    cfg.dcp_size       = 2;
    cfg.nccl_comm      = reinterpret_cast<void*>(0x1);
    cfg.partial_output = nullptr;
    cfg.partial_lse    = reinterpret_cast<float*>(0x1);
    EXPECT_THROW(runner.init(cfg, nullptr, nullptr), std::invalid_argument);
}

TEST(DcpAllreduceGraph, NullLseRejected) {
    lc::DcpAllreduceGraphRunner runner;
    lc::DcpAllreduceConfig cfg{};
    cfg.dcp_size       = 2;
    cfg.nccl_comm      = reinterpret_cast<void*>(0x1);
    cfg.partial_output = reinterpret_cast<void*>(0x1);
    cfg.partial_lse    = nullptr;
    EXPECT_THROW(runner.init(cfg, nullptr, nullptr), std::invalid_argument);
}

TEST(DcpAllreduceGraph, NullDeviceBackendRejected) {
    lc::DcpAllreduceGraphRunner runner;
    lc::DcpAllreduceConfig cfg{};
    cfg.dcp_size       = 2;
    cfg.nccl_comm      = reinterpret_cast<void*>(0x1);
    cfg.partial_output = reinterpret_cast<void*>(0x1);
    cfg.partial_lse    = reinterpret_cast<float*>(0x1);
    EXPECT_THROW(runner.init(cfg, nullptr, nullptr), std::invalid_argument);
}

TEST(DcpAllreduceGraph, DestroyDefaultConstructed) {
    lc::DcpAllreduceGraphRunner runner;
    EXPECT_FALSE(runner.is_captured());
    runner.destroy();  // no-op
    runner.destroy();  // still no-op
    EXPECT_FALSE(runner.is_captured());
}

// ============================================================================
// Multi-GPU test fixture
//
// Uses ncclCommInitRank + threads to mimic multi-process NCCL.
// This avoids the ncclGroupStart/End requirement of ncclCommInitAll.
// ============================================================================

class DcpAllreduceGraphMultiGpu : public ::testing::Test {
protected:
    static constexpr int kNumGpus = 2;

    ncclComm_t    comms_[kNumGpus]   = {};
    cudaStream_t  streams_[kNumGpus] = {};
    std::unique_ptr<lc::DeviceBackend> device_backends_[kNumGpus];
    bool initialized_ = false;

    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count < kNumGpus) {
            GTEST_SKIP() << "Need " << kNumGpus << " GPUs, have "
                         << device_count;
        }

        // ncclCommInitRank requires all ranks to call concurrently
        ncclUniqueId id;
        ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);

        std::atomic<int> errors{0};
        std::vector<std::thread> threads;
        for (int r = 0; r < kNumGpus; ++r) {
            threads.emplace_back([&, r]() {
                if (cudaSetDevice(r) != cudaSuccess) { errors++; return; }
                if (ncclCommInitRank(&comms_[r], kNumGpus, id, r) !=
                    ncclSuccess) { errors++; return; }
                if (cudaStreamCreate(&streams_[r]) != cudaSuccess) {
                    errors++; return;
                }
                device_backends_[r] = lc::make_cuda_sm120_device_backend(
                    layerstorm::config::GpuRef{r, r, layerstorm::config::GpuType::rtx5090});
            });
        }
        for (auto& t : threads) t.join();
        ASSERT_EQ(errors.load(), 0) << "NCCL/CUDA init failed";
        initialized_ = true;
    }

    void TearDown() override {
        if (!initialized_) return;
        for (int r = 0; r < kNumGpus; ++r) {
            ncclCommDestroy(comms_[r]);
            cudaSetDevice(r);
            cudaStreamDestroy(streams_[r]);
        }
    }

    // Helper: build config for a given rank
    lc::DcpAllreduceConfig make_config(int rank, int B, int H, int D,
                                        void* output, float* lse) {
        return lc::DcpAllreduceConfig{
            .batch_size     = B,
            .num_heads      = H,
            .head_dim       = D,
            .dcp_size       = kNumGpus,
            .rank           = rank,
            .nccl_comm      = comms_[rank],
            .partial_output = output,
            .partial_lse    = lse,
        };
    }

    // Helper: capture all runners concurrently (one thread per rank)
    void capture_all(lc::DcpAllreduceGraphRunner* runners,
                     lc::DcpAllreduceConfig* cfgs) {
        std::atomic<int> errors{0};
        std::string error_msgs[kNumGpus];
        std::vector<std::thread> threads;
        for (int r = 0; r < kNumGpus; ++r) {
            threads.emplace_back([&, r]() {
                try {
                    cudaSetDevice(r);
                    runners[r].init(cfgs[r], streams_[r],
                                    device_backends_[r].get());
                } catch (const std::exception& e) {
                    error_msgs[r] = e.what();
                    errors++;
                }
            });
        }
        for (auto& t : threads) t.join();
        for (int r = 0; r < kNumGpus; ++r) {
            ASSERT_TRUE(error_msgs[r].empty())
                << "Rank " << r << " init failed: " << error_msgs[r];
        }
    }

    // Helper: replay all runners and synchronize
    void replay_and_sync(lc::DcpAllreduceGraphRunner* runners) {
        // Launch all graphs (non-blocking at host level)
        for (int r = 0; r < kNumGpus; ++r) {
            ASSERT_EQ(cudaSetDevice(r), cudaSuccess);
            runners[r].replay(streams_[r]);
        }
        // Synchronize
        for (int r = 0; r < kNumGpus; ++r) {
            ASSERT_EQ(cudaSetDevice(r), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(streams_[r]), cudaSuccess);
        }
    }
};

// ============================================================================
// Multi-GPU correctness tests
// ============================================================================

TEST_F(DcpAllreduceGraphMultiGpu, TwoGpuCaptureReplay) {
    const int B = 2, H = 4, D = 32;

    // Rank 0: output=1.0, lse=2.0  |  Rank 1: output=3.0, lse=1.0
    std::vector<std::vector<float>> h_outputs = {
        std::vector<float>(B * H * D, 1.0f),
        std::vector<float>(B * H * D, 3.0f),
    };
    std::vector<std::vector<float>> h_lses = {
        std::vector<float>(B * H, 2.0f),
        std::vector<float>(B * H, 1.0f),
    };
    auto expected = dcp_allreduce_ref(h_outputs, h_lses, B, H, D, kNumGpus);

    // Allocate per-GPU buffers
    void*  d_output[kNumGpus] = {};
    float* d_lse[kNumGpus]    = {};
    for (int r = 0; r < kNumGpus; ++r) {
        d_output[r] = alloc_bf16_device(r, h_outputs[r]);
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_lse[r], B * H * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_lse[r], h_lses[r].data(),
                               B * H * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    // Capture concurrently
    lc::DcpAllreduceGraphRunner runners[kNumGpus];
    lc::DcpAllreduceConfig cfgs[kNumGpus];
    for (int r = 0; r < kNumGpus; ++r)
        cfgs[r] = make_config(r, B, H, D, d_output[r], d_lse[r]);
    capture_all(runners, cfgs);

    for (int r = 0; r < kNumGpus; ++r) {
        EXPECT_TRUE(runners[r].is_captured());
        EXPECT_NE(runners[r].gathered_lse_ptr(), nullptr);
        EXPECT_NE(runners[r].global_lse_ptr(), nullptr);
    }

    // Replay and verify
    replay_and_sync(runners);

    for (int r = 0; r < kNumGpus; ++r) {
        auto result = read_bf16_device(r, d_output[r], B * H * D);
        for (int i = 0; i < B * H * D; ++i) {
            EXPECT_NEAR(result[i], expected[i], 0.1f)
                << "rank=" << r << " index=" << i;
        }
    }

    // Cleanup
    for (int r = 0; r < kNumGpus; ++r) {
        runners[r].destroy();
        cudaSetDevice(r);
        cudaFree(d_output[r]);
        cudaFree(d_lse[r]);
    }
}

TEST_F(DcpAllreduceGraphMultiGpu, MultiReplay) {
    const int B = 2, H = 4, D = 16;

    // Allocate fixed-address buffers
    void*  d_output[kNumGpus] = {};
    float* d_lse[kNumGpus]    = {};
    for (int r = 0; r < kNumGpus; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_output[r], B * H * D * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_lse[r], B * H * sizeof(float)));
    }

    // Write initial data and capture
    std::vector<float> init_out(B * H * D, 1.0f);
    std::vector<float> init_lse(B * H, 0.0f);
    for (int r = 0; r < kNumGpus; ++r) {
        write_bf16_device(r, d_output[r], init_out);
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMemcpy(d_lse[r], init_lse.data(),
                               B * H * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    lc::DcpAllreduceGraphRunner runners[kNumGpus];
    lc::DcpAllreduceConfig cfgs[kNumGpus];
    for (int r = 0; r < kNumGpus; ++r)
        cfgs[r] = make_config(r, B, H, D, d_output[r], d_lse[r]);
    capture_all(runners, cfgs);

    // Replay 5 times with different data each iteration
    for (int iter = 0; iter < 5; ++iter) {
        float val0 = 1.0f + iter;
        float val1 = 2.0f + iter;
        float lse0 = 1.0f + 0.5f * iter;
        float lse1 = 0.5f + 0.3f * iter;

        std::vector<std::vector<float>> h_outputs = {
            std::vector<float>(B * H * D, val0),
            std::vector<float>(B * H * D, val1),
        };
        std::vector<std::vector<float>> h_lses = {
            std::vector<float>(B * H, lse0),
            std::vector<float>(B * H, lse1),
        };
        auto expected = dcp_allreduce_ref(h_outputs, h_lses, B, H, D,
                                           kNumGpus);

        // Write new data into fixed buffers
        for (int r = 0; r < kNumGpus; ++r) {
            write_bf16_device(r, d_output[r], h_outputs[r]);
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaMemcpy(d_lse[r], h_lses[r].data(),
                                   B * H * sizeof(float),
                                   cudaMemcpyHostToDevice));
        }

        replay_and_sync(runners);

        for (int r = 0; r < kNumGpus; ++r) {
            auto result = read_bf16_device(r, d_output[r], B * H * D);
            for (int i = 0; i < B * H * D; ++i) {
                EXPECT_NEAR(result[i], expected[i], 0.15f)
                    << "iter=" << iter << " rank=" << r << " i=" << i;
            }
        }
    }

    for (int r = 0; r < kNumGpus; ++r) {
        runners[r].destroy();
        cudaSetDevice(r);
        cudaFree(d_output[r]);
        cudaFree(d_lse[r]);
    }
}

TEST_F(DcpAllreduceGraphMultiGpu, AsymmetricLse) {
    // One rank has much larger LSE -> dominates.  No NaN/Inf.
    const int B = 1, H = 2, D = 16;

    std::vector<std::vector<float>> h_outputs = {
        std::vector<float>(B * H * D, 1.0f),    // rank 0
        std::vector<float>(B * H * D, 100.0f),  // rank 1
    };
    std::vector<std::vector<float>> h_lses = {
        std::vector<float>(B * H, 50.0f),   // rank 0 dominates
        std::vector<float>(B * H, -50.0f),  // rank 1 negligible
    };
    auto expected = dcp_allreduce_ref(h_outputs, h_lses, B, H, D, kNumGpus);

    void*  d_output[kNumGpus] = {};
    float* d_lse[kNumGpus]    = {};
    for (int r = 0; r < kNumGpus; ++r) {
        d_output[r] = alloc_bf16_device(r, h_outputs[r]);
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_lse[r], B * H * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_lse[r], h_lses[r].data(),
                               B * H * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    lc::DcpAllreduceGraphRunner runners[kNumGpus];
    lc::DcpAllreduceConfig cfgs[kNumGpus];
    for (int r = 0; r < kNumGpus; ++r)
        cfgs[r] = make_config(r, B, H, D, d_output[r], d_lse[r]);
    capture_all(runners, cfgs);
    replay_and_sync(runners);

    // Verify no NaN/Inf and match reference
    for (int r = 0; r < kNumGpus; ++r) {
        auto result = read_bf16_device(r, d_output[r], B * H * D);
        for (int i = 0; i < B * H * D; ++i) {
            EXPECT_FALSE(std::isnan(result[i]))
                << "NaN at rank=" << r << " i=" << i;
            EXPECT_FALSE(std::isinf(result[i]))
                << "Inf at rank=" << r << " i=" << i;
            EXPECT_NEAR(result[i], expected[i], 0.1f)
                << "rank=" << r << " i=" << i;
        }
    }

    // Rank 0 dominates: result should be close to rank 0's original output
    auto result = read_bf16_device(0, d_output[0], B * H * D);
    for (int i = 0; i < B * H * D; ++i) {
        EXPECT_NEAR(result[i], 1.0f, 0.01f)
            << "Dominant rank not preserved at i=" << i;
    }

    for (int r = 0; r < kNumGpus; ++r) {
        runners[r].destroy();
        cudaSetDevice(r);
        cudaFree(d_output[r]);
        cudaFree(d_lse[r]);
    }
}

TEST_F(DcpAllreduceGraphMultiGpu, EqualLse) {
    // Both ranks have identical LSE -> each scaled by 0.5, result = average.
    const int B = 2, H = 2, D = 8;

    std::vector<std::vector<float>> h_outputs = {
        std::vector<float>(B * H * D, 2.0f),  // rank 0
        std::vector<float>(B * H * D, 6.0f),  // rank 1
    };
    std::vector<std::vector<float>> h_lses = {
        std::vector<float>(B * H, 3.0f),  // equal
        std::vector<float>(B * H, 3.0f),  // equal
    };
    auto expected = dcp_allreduce_ref(h_outputs, h_lses, B, H, D, kNumGpus);
    // With equal LSE: result = 0.5 * 2.0 + 0.5 * 6.0 = 4.0

    void*  d_output[kNumGpus] = {};
    float* d_lse[kNumGpus]    = {};
    for (int r = 0; r < kNumGpus; ++r) {
        d_output[r] = alloc_bf16_device(r, h_outputs[r]);
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_lse[r], B * H * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_lse[r], h_lses[r].data(),
                               B * H * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    lc::DcpAllreduceGraphRunner runners[kNumGpus];
    lc::DcpAllreduceConfig cfgs[kNumGpus];
    for (int r = 0; r < kNumGpus; ++r)
        cfgs[r] = make_config(r, B, H, D, d_output[r], d_lse[r]);
    capture_all(runners, cfgs);
    replay_and_sync(runners);

    for (int r = 0; r < kNumGpus; ++r) {
        auto result = read_bf16_device(r, d_output[r], B * H * D);
        for (int i = 0; i < B * H * D; ++i) {
            EXPECT_NEAR(result[i], 4.0f, 0.1f)
                << "rank=" << r << " i=" << i;
        }
    }

    for (int r = 0; r < kNumGpus; ++r) {
        runners[r].destroy();
        cudaSetDevice(r);
        cudaFree(d_output[r]);
        cudaFree(d_lse[r]);
    }
}

TEST_F(DcpAllreduceGraphMultiGpu, DestroyIdempotent) {
    const int B = 1, H = 2, D = 8;

    void*  d_output[kNumGpus] = {};
    float* d_lse[kNumGpus]    = {};
    for (int r = 0; r < kNumGpus; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_output[r], B * H * D * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_lse[r], B * H * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_output[r], 0, B * H * D * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemset(d_lse[r], 0, B * H * sizeof(float)));
    }

    lc::DcpAllreduceGraphRunner runners[kNumGpus];
    lc::DcpAllreduceConfig cfgs[kNumGpus];
    for (int r = 0; r < kNumGpus; ++r)
        cfgs[r] = make_config(r, B, H, D, d_output[r], d_lse[r]);
    capture_all(runners, cfgs);

    // Destroy rank 0's runner twice
    EXPECT_TRUE(runners[0].is_captured());
    runners[0].destroy();
    EXPECT_FALSE(runners[0].is_captured());
    EXPECT_EQ(runners[0].gathered_lse_ptr(), nullptr);
    EXPECT_EQ(runners[0].global_lse_ptr(), nullptr);

    runners[0].destroy();  // second call: no crash
    EXPECT_FALSE(runners[0].is_captured());

    // Rank 1 still valid
    EXPECT_TRUE(runners[1].is_captured());
    runners[1].destroy();

    for (int r = 0; r < kNumGpus; ++r) {
        cudaSetDevice(r);
        cudaFree(d_output[r]);
        cudaFree(d_lse[r]);
    }
}

TEST_F(DcpAllreduceGraphMultiGpu, RegistryIntegration) {
    const int B = 1, H = 2, D = 8;

    void*  d_output[kNumGpus] = {};
    float* d_lse[kNumGpus]    = {};
    for (int r = 0; r < kNumGpus; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_output[r], B * H * D * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_lse[r], B * H * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_output[r], 0, B * H * D * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemset(d_lse[r], 0, B * H * sizeof(float)));
    }

    // Capture runners concurrently
    auto* runner0 = new lc::DcpAllreduceGraphRunner();
    auto* runner1 = new lc::DcpAllreduceGraphRunner();
    {
        lc::DcpAllreduceConfig cfgs[kNumGpus];
        for (int r = 0; r < kNumGpus; ++r)
            cfgs[r] = make_config(r, B, H, D, d_output[r], d_lse[r]);

        std::atomic<int> errors{0};
        std::string error_msgs[kNumGpus];
        lc::DcpAllreduceGraphRunner* ptrs[kNumGpus] = {runner0, runner1};
        std::vector<std::thread> threads;
        for (int r = 0; r < kNumGpus; ++r) {
            threads.emplace_back([&, r]() {
                try {
                    cudaSetDevice(r);
                    ptrs[r]->init(cfgs[r], streams_[r],
                                   device_backends_[r].get());
                } catch (const std::exception& e) {
                    error_msgs[r] = e.what();
                    errors++;
                }
            });
        }
        for (auto& t : threads) t.join();
        for (int r = 0; r < kNumGpus; ++r)
            ASSERT_TRUE(error_msgs[r].empty())
                << "Rank " << r << ": " << error_msgs[r];
    }

    // Register in GraphRegistry
    lc::GraphRegistry registry;
    lc::GraphKey key0{lc::GraphType::kDcpAllreduce, 0, B};
    lc::GraphKey key1{lc::GraphType::kDcpAllreduce, 1, B};

    registry.insert(key0, lc::GraphEntry{
        .runner  = runner0,
        .destroy = [](std::any& r) {
            delete std::any_cast<lc::DcpAllreduceGraphRunner*>(r);
        }
    });
    registry.insert(key1, lc::GraphEntry{
        .runner  = runner1,
        .destroy = [](std::any& r) {
            delete std::any_cast<lc::DcpAllreduceGraphRunner*>(r);
        }
    });

    EXPECT_EQ(registry.count_by_type(lc::GraphType::kDcpAllreduce), 2u);

    auto* found = registry.find_as<lc::DcpAllreduceGraphRunner>(key0);
    ASSERT_NE(found, nullptr);
    EXPECT_TRUE(found->is_captured());
    EXPECT_EQ(found->config().batch_size, B);
    EXPECT_EQ(found->config().dcp_size, kNumGpus);
    EXPECT_EQ(found->config().rank, 0);

    auto* found1 = registry.find_as<lc::DcpAllreduceGraphRunner>(key1);
    ASSERT_NE(found1, nullptr);
    EXPECT_EQ(found1->config().rank, 1);

    registry.clear();  // calls destroy callbacks

    for (int r = 0; r < kNumGpus; ++r) {
        cudaSetDevice(r);
        cudaFree(d_output[r]);
        cudaFree(d_lse[r]);
    }
}
