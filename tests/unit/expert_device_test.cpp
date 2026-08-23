// Unit tests for ExpertDevice abstraction.
//
// Tests NullExpertDevice construction, polymorphism, factory function,
// and verifies all methods are callable through the base pointer.

#include <gtest/gtest.h>

#include "../gpu_test_utils.h"
#include "core/expert_device.h"
#include "core/null_expert_device.h"
#include "compute/cuda_sm120_expert_device.h"

namespace lc = layerstorm::compute;
namespace cfg = layerstorm::config;

namespace {

cfg::GpuRef make_gpu(int pos = 0, int id = 0,
                     cfg::GpuType type = cfg::GpuType::rtx5090) {
    return {pos, id, type};
}

}  // namespace

// ── NullExpertDevice ────────────────────────────────────────────────────────

TEST(ExpertDeviceTest, NullConstruction) {
    auto dev = lc::make_null_expert_device(make_gpu());
    ASSERT_NE(dev, nullptr);
    EXPECT_EQ(dev->gpu().position, 0);
}

TEST(ExpertDeviceTest, NullGpuIdentity) {
    auto dev = lc::make_null_expert_device(make_gpu(3, 5));
    EXPECT_EQ(dev->gpu().position, 3);
    EXPECT_EQ(dev->gpu().id, 5);
}

TEST(ExpertDeviceTest, NullSetDeviceNoop) {
    auto dev = lc::make_null_expert_device(make_gpu());
    dev->set_device();
}

TEST(ExpertDeviceTest, NullGroupedGemmNoop) {
    auto dev = lc::make_null_expert_device(make_gpu());

    lc::Nvfp4GroupedGemmParams nvfp4_params{};
    dev->nvfp4_grouped_gemm(nvfp4_params, nullptr, 0, nullptr);

    lc::Fp8GroupedGemmParams fp8_params{};
    dev->fp8_grouped_gemm(fp8_params, nullptr, 0, nullptr);
}

TEST(ExpertDeviceTest, NullSwigluNoop) {
    auto dev = lc::make_null_expert_device(make_gpu());
    lc::FusedSwigluParams params{};
    dev->fused_swiglu(nullptr, nullptr, params, 2, nullptr);
}

TEST(ExpertDeviceTest, NullPermuteNoop) {
    auto dev = lc::make_null_expert_device(make_gpu());
    dev->moe_permute(nullptr, nullptr, nullptr, nullptr,
                     nullptr, nullptr, 0, 0, 0, 0, 2, nullptr, nullptr);
    dev->moe_unpermute(nullptr, nullptr, nullptr, nullptr,
                       0, 0, 0, 2, nullptr);
}

TEST(ExpertDeviceTest, NullAllocFree) {
    auto dev = lc::make_null_expert_device(make_gpu());
    void* ptr = dev->device_alloc(512);
    ASSERT_NE(ptr, nullptr);
    dev->device_free(ptr);
    dev->device_free(nullptr);
}

// ── Polymorphism ────────────────────────────────────────────────────────────

TEST(ExpertDeviceTest, Polymorphism) {
    std::unique_ptr<lc::ExpertDevice> dev =
        lc::make_null_expert_device(make_gpu(2, 4));
    EXPECT_EQ(dev->gpu().position, 2);
    dev->set_device();
    void* p = dev->device_alloc(128);
    ASSERT_NE(p, nullptr);
    dev->device_free(p);
}

TEST(ExpertDeviceTest, MultipleDifferentGpus) {
    auto dev0 = lc::make_null_expert_device(make_gpu(0, 0));
    auto dev1 = lc::make_null_expert_device(make_gpu(1, 1));
    auto dev2 = lc::make_null_expert_device(make_gpu(2, 2));
    EXPECT_EQ(dev0->gpu().position, 0);
    EXPECT_EQ(dev1->gpu().position, 1);
    EXPECT_EQ(dev2->gpu().position, 2);
}

TEST(ExpertDeviceTest, UniqueOwnership) {
    auto dev = lc::make_null_expert_device(make_gpu());
    lc::ExpertDevice* raw = dev.get();
    ASSERT_NE(raw, nullptr);
    dev.reset();
}

// ── CudaSm120ExpertDevice (GPU required) ───────────────────────────────────

TEST(CudaSm120ExpertDevice, ConstructAndSetDevice) {
    REQUIRES_GPU();
    auto dev = lc::make_cuda_sm120_expert_device(make_gpu(0, 0));
    ASSERT_NE(dev, nullptr);
    EXPECT_NO_THROW(dev->set_device());
}

TEST(CudaSm120ExpertDevice, GpuReturnsConstructorRef) {
    REQUIRES_GPU();
    auto dev = lc::make_cuda_sm120_expert_device(
        make_gpu(1, 0, cfg::GpuType::rtx5080));
    EXPECT_EQ(dev->gpu().position, 1);
    EXPECT_EQ(dev->gpu().id, 0);
    EXPECT_EQ(dev->gpu().type, cfg::GpuType::rtx5080);
}

TEST(CudaSm120ExpertDevice, AllocFree) {
    REQUIRES_GPU();
    auto dev = lc::make_cuda_sm120_expert_device(make_gpu(0, 0));
    dev->set_device();
    void* ptr = dev->device_alloc(4096);
    ASSERT_NE(ptr, nullptr);
    EXPECT_NO_THROW(dev->device_free(ptr));
}

TEST(CudaSm120ExpertDevice, FreeNull) {
    REQUIRES_GPU();
    auto dev = lc::make_cuda_sm120_expert_device(make_gpu(0, 0));
    EXPECT_NO_THROW(dev->device_free(nullptr));
}

TEST(CudaSm120ExpertDevice, FactoryPattern) {
    REQUIRES_GPU();
    std::unique_ptr<lc::ExpertDevice> dev =
        lc::make_cuda_sm120_expert_device(make_gpu(0, 0));
    ASSERT_NE(dev, nullptr);
    EXPECT_NO_THROW(dev->set_device());
    EXPECT_EQ(dev->gpu().position, 0);
}
