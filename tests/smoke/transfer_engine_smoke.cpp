// Smoke test: TransferEngine with real CUDA streams, events, and cudaMemcpyAsync.
//
// Verifies that CudaSm120DeviceBackend actually performs async DMA between
// pinned host memory and GPU VRAM. Uses cudaMallocHost + cudaMalloc for real
// PCIe H2D/D2H transfers.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "compute/cuda_sm120_device_backend.h"
#include "core/gpu_ref.h"
#include "core/transfer/transfer_engine.h"

namespace lc = layerstorm::compute;
namespace lt = layerstorm::transfer;
namespace lmem = layerstorm::memory;

static std::vector<layerstorm::config::GpuRef> make_gpu_refs(int count) {
    std::vector<layerstorm::config::GpuRef> v;
    for (int i = 0; i < count; ++i)
        v.push_back({.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090});
    return v;
}

// ── Fixture ────────────────────────────────────────────────────────────────

class TransferEngineSmoke : public ::testing::Test {
protected:
    void SetUp() override {
        if (cudaGetDeviceCount(&gpu_count_) != cudaSuccess || gpu_count_ == 0) {
            GTEST_SKIP() << "No CUDA GPU — cannot run transfer engine smoke test";
        }
        std::cout << "\nDetected " << gpu_count_ << " GPUs\n";
    }

    int gpu_count_ = 0;
};

static lmem::ExpertKey ek(uint32_t layer, uint16_t expert) {
    return {layer, expert};
}

// ── H2D round-trip: write pattern to host, DMA to GPU, read back ─────────

TEST_F(TransferEngineSmoke, H2DRoundTrip) {
    constexpr size_t kBytes = 4 * 1024 * 1024;  // 4 MiB

    for (int gpu = 0; gpu < gpu_count_; ++gpu) {
        cudaSetDevice(gpu);

        // Allocate pinned host memory and GPU memory.
        uint8_t* host_src = nullptr;
        uint8_t* dev_dst = nullptr;
        uint8_t* host_readback = nullptr;
        ASSERT_EQ(cudaMallocHost(&host_src, kBytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dev_dst, kBytes), cudaSuccess);
        ASSERT_EQ(cudaMallocHost(&host_readback, kBytes), cudaSuccess);

        // Fill host with pattern.
        std::iota(host_src, host_src + kBytes, static_cast<uint8_t>(gpu));

        // Build engine with real CUDA backend.
        auto gpu_refs = make_gpu_refs(gpu_count_);
        std::vector<std::unique_ptr<lc::DeviceBackend>> dev_owners;
        std::vector<lc::DeviceBackend*> dev_ptrs;
        for (auto& ref : gpu_refs) {
            dev_owners.push_back(lc::make_cuda_sm120_device_backend(ref));
            dev_ptrs.push_back(dev_owners.back().get());
        }
        lt::TransferEngine::Options opts;
        opts.device_backends = dev_ptrs;
        lt::TransferEngine engine(std::move(opts));

        // Enqueue H2D transfer.
        bool callback_fired = false;
        auto token = engine.enqueue_h2d(
            ek(0, static_cast<uint16_t>(gpu)), gpu,
            dev_dst, host_src, kBytes,
            [&](lt::TransferToken, bool success) {
                EXPECT_TRUE(success);
                callback_fired = true;
            });
        ASSERT_TRUE(token.has_value());
        EXPECT_TRUE(engine.is_inflight_h2d(ek(0, static_cast<uint16_t>(gpu)), gpu));

        // Poll until complete (real async — may need multiple polls).
        std::vector<lt::TransferCompletion> completions;
        for (int i = 0; i < 10000 && completions.empty(); ++i) {
            completions = engine.poll_completions();
        }
        ASSERT_EQ(completions.size(), 1u);
        EXPECT_TRUE(callback_fired);
        EXPECT_EQ(completions[0].bytes, static_cast<int64_t>(kBytes));
        EXPECT_TRUE(completions[0].success);

        // Read back from GPU to verify data arrived correctly.
        ASSERT_EQ(cudaMemcpy(host_readback, dev_dst, kBytes,
                             cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(std::memcmp(host_src, host_readback, kBytes), 0)
            << "GPU " << gpu << ": H2D data mismatch";

        cudaFreeHost(host_src);
        cudaFree(dev_dst);
        cudaFreeHost(host_readback);

        std::cout << "  GPU " << gpu << ": H2D " << (kBytes / 1024)
                  << " KiB round-trip OK\n";
    }
}

// ── D2H round-trip: fill GPU via cudaMemcpy, DMA back via engine ─────────

TEST_F(TransferEngineSmoke, D2HRoundTrip) {
    constexpr size_t kBytes = 2 * 1024 * 1024;  // 2 MiB

    for (int gpu = 0; gpu < gpu_count_; ++gpu) {
        cudaSetDevice(gpu);

        uint8_t* host_pattern = nullptr;
        uint8_t* dev_src = nullptr;
        uint8_t* host_dst = nullptr;
        ASSERT_EQ(cudaMallocHost(&host_pattern, kBytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dev_src, kBytes), cudaSuccess);
        ASSERT_EQ(cudaMallocHost(&host_dst, kBytes), cudaSuccess);

        // Fill pattern and upload to GPU.
        std::fill(host_pattern, host_pattern + kBytes,
                  static_cast<uint8_t>(0xAB + gpu));
        ASSERT_EQ(cudaMemcpy(dev_src, host_pattern, kBytes,
                             cudaMemcpyHostToDevice), cudaSuccess);

        // Zero host_dst so we can verify the DMA wrote it.
        std::memset(host_dst, 0, kBytes);

        auto gpu_refs = make_gpu_refs(gpu_count_);
        std::vector<std::unique_ptr<lc::DeviceBackend>> dev_owners;
        std::vector<lc::DeviceBackend*> dev_ptrs;
        for (auto& ref : gpu_refs) {
            dev_owners.push_back(lc::make_cuda_sm120_device_backend(ref));
            dev_ptrs.push_back(dev_owners.back().get());
        }
        lt::TransferEngine::Options opts;
        opts.device_backends = dev_ptrs;
        lt::TransferEngine engine(std::move(opts));

        // Enqueue D2H transfer.
        auto token = engine.enqueue_d2h(
            ek(1, static_cast<uint16_t>(gpu)), gpu,
            host_dst, dev_src, kBytes);
        ASSERT_TRUE(token.has_value());

        // Drain (blocking poll).
        engine.drain();
        EXPECT_EQ(engine.inflight_count(), 0);

        // Verify data.
        EXPECT_EQ(std::memcmp(host_pattern, host_dst, kBytes), 0)
            << "GPU " << gpu << ": D2H data mismatch";

        cudaFreeHost(host_pattern);
        cudaFree(dev_src);
        cudaFreeHost(host_dst);

        std::cout << "  GPU " << gpu << ": D2H " << (kBytes / 1024)
                  << " KiB round-trip OK\n";
    }
}

// ── Concurrent H2D across multiple GPUs ──────────────────────────────────

TEST_F(TransferEngineSmoke, ConcurrentMultiGpuH2D) {
    if (gpu_count_ < 2) {
        GTEST_SKIP() << "Need >= 2 GPUs for concurrent multi-GPU test";
    }

    constexpr size_t kBytes = 1 * 1024 * 1024;  // 1 MiB per GPU

    struct GpuBuffers {
        uint8_t* host_src = nullptr;
        uint8_t* dev_dst = nullptr;
        uint8_t* host_readback = nullptr;
    };
    std::vector<GpuBuffers> bufs(gpu_count_);

    // Allocate on all GPUs.
    for (int g = 0; g < gpu_count_; ++g) {
        cudaSetDevice(g);
        ASSERT_EQ(cudaMallocHost(&bufs[g].host_src, kBytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&bufs[g].dev_dst, kBytes), cudaSuccess);
        ASSERT_EQ(cudaMallocHost(&bufs[g].host_readback, kBytes), cudaSuccess);
        // Each GPU gets a distinct fill pattern.
        std::memset(bufs[g].host_src, 0x10 + g, kBytes);
    }

    auto gpu_refs = make_gpu_refs(gpu_count_);
    std::vector<std::unique_ptr<lc::DeviceBackend>> dev_owners;
    std::vector<lc::DeviceBackend*> dev_ptrs;
    for (auto& ref : gpu_refs) {
        dev_owners.push_back(lc::make_cuda_sm120_device_backend(ref));
        dev_ptrs.push_back(dev_owners.back().get());
    }
    lt::TransferEngine::Options opts;
    opts.device_backends = dev_ptrs;
    lt::TransferEngine engine(std::move(opts));

    // Enqueue H2D on all GPUs simultaneously.
    for (int g = 0; g < gpu_count_; ++g) {
        auto token = engine.enqueue_h2d(
            ek(2, static_cast<uint16_t>(g)), g,
            bufs[g].dev_dst, bufs[g].host_src, kBytes);
        ASSERT_TRUE(token.has_value());
    }
    EXPECT_EQ(engine.inflight_count(), gpu_count_);

    // Drain all.
    engine.drain();
    EXPECT_EQ(engine.inflight_count(), 0);

    // Verify each GPU's data.
    for (int g = 0; g < gpu_count_; ++g) {
        cudaSetDevice(g);
        ASSERT_EQ(cudaMemcpy(bufs[g].host_readback, bufs[g].dev_dst, kBytes,
                             cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(std::memcmp(bufs[g].host_src, bufs[g].host_readback, kBytes), 0)
            << "GPU " << g << ": concurrent H2D data mismatch";
    }

    // Cleanup.
    for (int g = 0; g < gpu_count_; ++g) {
        cudaFreeHost(bufs[g].host_src);
        cudaFree(bufs[g].dev_dst);
        cudaFreeHost(bufs[g].host_readback);
    }

    std::cout << "  " << gpu_count_ << " GPUs: concurrent H2D "
              << (kBytes / 1024) << " KiB each — all verified\n";
}

// ── Dedup with real CUDA: second enqueue returns nullopt ─────────────────

TEST_F(TransferEngineSmoke, DedupWithRealCuda) {
    cudaSetDevice(0);

    constexpr size_t kBytes = 64 * 1024;
    uint8_t* host_src = nullptr;
    uint8_t* dev_dst = nullptr;
    ASSERT_EQ(cudaMallocHost(&host_src, kBytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dev_dst, kBytes), cudaSuccess);

    auto gpu_refs = make_gpu_refs(gpu_count_);
    std::vector<std::unique_ptr<lc::DeviceBackend>> dev_owners;
    std::vector<lc::DeviceBackend*> dev_ptrs;
    for (auto& ref : gpu_refs) {
        dev_owners.push_back(lc::make_cuda_sm120_device_backend(ref));
        dev_ptrs.push_back(dev_owners.back().get());
    }
    lt::TransferEngine::Options opts;
    opts.device_backends = dev_ptrs;
    lt::TransferEngine engine(std::move(opts));

    auto t1 = engine.enqueue_h2d(ek(5, 5), 0, dev_dst, host_src, kBytes);
    ASSERT_TRUE(t1.has_value());

    // Same key+gpu+direction → dedup.
    auto t2 = engine.enqueue_h2d(ek(5, 5), 0, dev_dst, host_src, kBytes);
    EXPECT_FALSE(t2.has_value());

    engine.drain();

    // After completion, re-enqueue succeeds.
    auto t3 = engine.enqueue_h2d(ek(5, 5), 0, dev_dst, host_src, kBytes);
    EXPECT_TRUE(t3.has_value());

    engine.drain();

    cudaFreeHost(host_src);
    cudaFree(dev_dst);

    std::cout << "  Dedup with real CUDA verified\n";
}
