// Smoke test: DCP allreduce graph with V3.2 production dimensions.
//
// Verifies the DcpAllreduceGraphRunner at full V3.2 scale:
//   - B=64 (max decode batch), H=64 (TP=2 local heads), D=512 (v_head_dim)
//   - Real NCCL allgather + LSE correction + allreduce via CUDA graph
//
// Buffer layout matches DecodeGraphRunner's output buffers:
//   out_ptr()  = [B, s_q=1, h_q=64, d_v=512] BF16  (decode_graph.h line 209)
//   lse_ptr()  = [B, s_q=1, h_q=64]           FP32  (decode_graph.h line 210)
//
// We allocate identically-shaped buffers to validate the DCP runner
// accepts the same layout.  Direct DecodeGraphRunner inclusion is blocked
// by ODR conflicts (decode_graph.h transitively includes .cu kernel defs
// that clash with the snapmla_kernels object library).  End-to-end wiring
// through DecodeGraphRunner will be tested in #40 (DCP executor).
//
// Requires: 2 GPUs.
// Build:    cmake --build build --target dcp_allreduce_graph_smoke
// Run:      ./build/tests/smoke/dcp_allreduce_graph_smoke

#include "compute/graphs/dcp_allreduce_graph.h"
#include "compute/cuda_sm120_device_backend.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;

// -- Macros ------------------------------------------------------------------

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

// -- BF16 helpers ------------------------------------------------------------

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

static void* alloc_bf16(int dev, const std::vector<float>& h) {
    std::vector<uint16_t> buf(h.size());
    for (size_t i = 0; i < h.size(); ++i) buf[i] = float_to_bf16(h[i]);
    cudaSetDevice(dev);
    void* d = nullptr;
    cudaMalloc(&d, buf.size() * sizeof(uint16_t));
    cudaMemcpy(d, buf.data(), buf.size() * sizeof(uint16_t),
               cudaMemcpyHostToDevice);
    return d;
}

static std::vector<float> read_bf16(int dev, void* d, size_t n) {
    cudaSetDevice(dev);
    std::vector<uint16_t> buf(n);
    cudaMemcpy(buf.data(), d, n * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    std::vector<float> r(n);
    for (size_t i = 0; i < n; ++i) r[i] = bf16_to_float(buf[i]);
    return r;
}

// -- CPU reference -----------------------------------------------------------

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

// ============================================================================
// V3.2 full-scale smoke test
// ============================================================================

TEST(DcpAllreduceSmoke, V32FullDimensions) {
    REQUIRES_MULTI_GPU(2);

    // V3.2 production parameters (TP=2, DCP=2):
    //   128 Q heads / 2 GPUs = 64 local heads
    //   d_v = 512 (MLA compressed latent dim)
    //   max decode batch = 64
    constexpr int N = 2;       // dcp_size
    constexpr int B = 64;      // max decode batch
    constexpr int H = 64;      // local heads per GPU (128 / TP=2)
    constexpr int D = 512;     // v_head_dim

    // Buffer sizes match DecodeGraphRunner::allocate_fixed_buffers():
    //   buf_out_ = B * s_q * h_q * d_v * sizeof(BF16)   = 64*1*64*512*2 = 4 MB
    //   buf_lse_ = B * s_q * h_q * sizeof(float)         = 64*1*64*4     = 16 KB
    constexpr size_t out_elems = B * H * D;       // 2,097,152
    constexpr size_t lse_elems = B * H;            // 4,096

    // Create NCCL comms via ncclCommInitRank (multi-process-like)
    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);

    ncclComm_t   comms[N]   = {};
    cudaStream_t streams[N] = {};
    {
        std::atomic<int> errs{0};
        std::vector<std::thread> threads;
        for (int r = 0; r < N; ++r) {
            threads.emplace_back([&, r]() {
                if (cudaSetDevice(r) != cudaSuccess) { errs++; return; }
                if (ncclCommInitRank(&comms[r], N, id, r) != ncclSuccess)
                    { errs++; return; }
                if (cudaStreamCreate(&streams[r]) != cudaSuccess)
                    { errs++; return; }
            });
        }
        for (auto& t : threads) t.join();
        ASSERT_EQ(errs.load(), 0) << "NCCL/CUDA init failed";
    }

    // Generate random per-rank data
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> out_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> lse_dist(-3.0f, 3.0f);

    std::vector<std::vector<float>> h_outs(N), h_lses(N);
    for (int r = 0; r < N; ++r) {
        h_outs[r].resize(out_elems);
        h_lses[r].resize(lse_elems);
        for (auto& v : h_outs[r]) v = out_dist(rng);
        for (auto& v : h_lses[r]) v = lse_dist(rng);
    }
    auto expected = dcp_ref(h_outs, h_lses, B, H, D, N);

    // Allocate device buffers (matching DecodeGraphRunner layout)
    void*  d_out[N] = {};
    float* d_lse[N] = {};
    for (int r = 0; r < N; ++r) {
        d_out[r] = alloc_bf16(r, h_outs[r]);
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_lse[r], lse_elems * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_lse[r], h_lses[r].data(),
                               lse_elems * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    // Create per-GPU DeviceBackend instances (#86b)
    std::unique_ptr<lc::DeviceBackend> dev_backends[N];
    for (int r = 0; r < N; ++r) {
        dev_backends[r] = lc::make_cuda_sm120_device_backend(
            layerstorm::config::GpuRef{r, r, layerstorm::config::GpuType::rtx5090});
    }

    // Capture DCP allreduce graphs (concurrent, one thread per rank)
    lc::DcpAllreduceGraphRunner runners[N];
    lc::DcpAllreduceConfig cfgs[N];
    for (int r = 0; r < N; ++r) {
        cfgs[r] = {
            .batch_size     = B,
            .num_heads      = H,
            .head_dim       = D,
            .dcp_size       = N,
            .rank           = r,
            .nccl_comm      = comms[r],
            .partial_output = d_out[r],
            .partial_lse    = d_lse[r],
        };
    }
    {
        std::atomic<int> errs{0};
        std::string msgs[N];
        std::vector<std::thread> threads;
        for (int r = 0; r < N; ++r) {
            threads.emplace_back([&, r]() {
                try {
                    cudaSetDevice(r);
                    runners[r].init(cfgs[r], streams[r],
                                    dev_backends[r].get());
                } catch (const std::exception& e) {
                    msgs[r] = e.what();
                    errs++;
                }
            });
        }
        for (auto& t : threads) t.join();
        for (int r = 0; r < N; ++r)
            ASSERT_TRUE(msgs[r].empty()) << "Rank " << r << ": " << msgs[r];
    }

    // Replay
    for (int r = 0; r < N; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        runners[r].replay(streams[r]);
    }
    for (int r = 0; r < N; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }

    // Verify: spot-check first 1024 and last 1024 elements
    // (checking all 2M elements is slow in a smoke test)
    for (int r = 0; r < N; ++r) {
        auto result = read_bf16(r, d_out[r], out_elems);

        for (int i = 0; i < 1024; ++i) {
            EXPECT_NEAR(result[i], expected[i], 0.15f)
                << "rank=" << r << " i=" << i;
        }
        for (size_t i = out_elems - 1024; i < out_elems; ++i) {
            EXPECT_NEAR(result[i], expected[i], 0.15f)
                << "rank=" << r << " i=" << i;
        }
    }

    // Cleanup
    for (int r = 0; r < N; ++r) {
        runners[r].destroy();
        cudaSetDevice(r);
        cudaFree(d_out[r]);
        cudaFree(d_lse[r]);
        cudaStreamDestroy(streams[r]);
        ncclCommDestroy(comms[r]);
    }
}

TEST(DcpAllreduceSmoke, V32MultiReplayStability) {
    REQUIRES_MULTI_GPU(2);

    // Smaller batch for multi-replay (still V3.2 head/dim)
    constexpr int N = 2, B = 8, H = 64, D = 512;
    constexpr size_t out_elems = B * H * D;
    constexpr size_t lse_elems = B * H;

    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);

    ncclComm_t   comms[N]   = {};
    cudaStream_t streams[N] = {};
    {
        std::atomic<int> errs{0};
        std::vector<std::thread> threads;
        for (int r = 0; r < N; ++r) {
            threads.emplace_back([&, r]() {
                if (cudaSetDevice(r) != cudaSuccess) { errs++; return; }
                if (ncclCommInitRank(&comms[r], N, id, r) != ncclSuccess)
                    { errs++; return; }
                if (cudaStreamCreate(&streams[r]) != cudaSuccess)
                    { errs++; return; }
            });
        }
        for (auto& t : threads) t.join();
        ASSERT_EQ(errs.load(), 0);
    }

    // Allocate fixed-address buffers
    void*  d_out[N] = {};
    float* d_lse[N] = {};
    for (int r = 0; r < N; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_out[r], out_elems * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_lse[r], lse_elems * sizeof(float)));
    }

    // Capture with dummy data
    std::vector<float> zeros_out(out_elems, 0.0f);
    std::vector<float> zeros_lse(lse_elems, 0.0f);
    for (int r = 0; r < N; ++r) {
        // Write zeros — just need valid buffers for capture
        std::vector<uint16_t> zbuf(out_elems, float_to_bf16(0.0f));
        cudaSetDevice(r);
        cudaMemcpy(d_out[r], zbuf.data(), out_elems * sizeof(uint16_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_lse[r], zeros_lse.data(), lse_elems * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    std::unique_ptr<lc::DeviceBackend> dev_backends[N];
    for (int r = 0; r < N; ++r) {
        dev_backends[r] = lc::make_cuda_sm120_device_backend(
            layerstorm::config::GpuRef{r, r, layerstorm::config::GpuType::rtx5090});
    }

    lc::DcpAllreduceGraphRunner runners[N];
    lc::DcpAllreduceConfig cfgs[N];
    for (int r = 0; r < N; ++r) {
        cfgs[r] = {B, H, D, N, r, comms[r], d_out[r], d_lse[r]};
    }
    {
        std::atomic<int> errs{0};
        std::string msgs[N];
        std::vector<std::thread> threads;
        for (int r = 0; r < N; ++r) {
            threads.emplace_back([&, r]() {
                try {
                    cudaSetDevice(r);
                    runners[r].init(cfgs[r], streams[r],
                                    dev_backends[r].get());
                } catch (const std::exception& e) {
                    msgs[r] = e.what(); errs++;
                }
            });
        }
        for (auto& t : threads) t.join();
        for (int r = 0; r < N; ++r)
            ASSERT_TRUE(msgs[r].empty()) << "Rank " << r << ": " << msgs[r];
    }

    // 10 replays with different random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (int iter = 0; iter < 10; ++iter) {
        std::vector<std::vector<float>> h_outs(N), h_lses(N);
        for (int r = 0; r < N; ++r) {
            h_outs[r].resize(out_elems);
            h_lses[r].resize(lse_elems);
            for (auto& v : h_outs[r]) v = dist(rng);
            for (auto& v : h_lses[r]) v = dist(rng);
        }
        auto expected = dcp_ref(h_outs, h_lses, B, H, D, N);

        // Write into fixed buffers
        for (int r = 0; r < N; ++r) {
            std::vector<uint16_t> bf(out_elems);
            for (size_t i = 0; i < out_elems; ++i)
                bf[i] = float_to_bf16(h_outs[r][i]);
            cudaSetDevice(r);
            cudaMemcpy(d_out[r], bf.data(), out_elems * sizeof(uint16_t),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_lse[r], h_lses[r].data(), lse_elems * sizeof(float),
                       cudaMemcpyHostToDevice);
        }

        // Replay
        for (int r = 0; r < N; ++r) {
            cudaSetDevice(r);
            runners[r].replay(streams[r]);
        }
        for (int r = 0; r < N; ++r) {
            cudaSetDevice(r);
            ASSERT_EQ(cudaStreamSynchronize(streams[r]), cudaSuccess);
        }

        // Spot-check 256 elements
        for (int r = 0; r < N; ++r) {
            auto result = read_bf16(r, d_out[r], out_elems);
            for (int i = 0; i < 256; ++i) {
                EXPECT_NEAR(result[i], expected[i], 0.15f)
                    << "iter=" << iter << " rank=" << r << " i=" << i;
            }
        }
    }

    for (int r = 0; r < N; ++r) {
        runners[r].destroy();
        cudaSetDevice(r);
        cudaFree(d_out[r]);
        cudaFree(d_lse[r]);
        cudaStreamDestroy(streams[r]);
        ncclCommDestroy(comms[r]);
    }
}
