// Unit tests for DeviceBackend hierarchy.
//
// Group 1: NullDeviceBackend (CPU-only, no GPU)
// Group 2: Per-GPU instantiation (INV-BH-1)
// Group 3: Polymorphism through base pointer
// Group 4: CudaSm120DeviceBackend (GPU required)

#include "core/device_backend.h"
#include "core/null_device_backend.h"
#include "compute/cuda_sm120_device_backend.h"
#include "core/gpu_ref.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <memory>

namespace lc = layerstorm::compute;

// ── Helpers ─────────────────────────────────────────────────────────────────

static layerstorm::config::GpuRef make_ref(int pos, int id,
                                            layerstorm::config::GpuType type =
                                                layerstorm::config::GpuType::rtx5090) {
    return {.position = pos, .id = id, .type = type};
}

// ── Group 1: NullDeviceBackend ──────────────────────────────────────────────

TEST(NullDeviceBackend, ConstructsWithGpuRef) {
    auto ref = make_ref(0, 0);
    lc::NullDeviceBackend backend(ref);
    EXPECT_EQ(backend.gpu().position, 0);
    EXPECT_EQ(backend.gpu().id, 0);
    EXPECT_EQ(backend.gpu().type, layerstorm::config::GpuType::rtx5090);
}

TEST(NullDeviceBackend, SetDeviceIsNoop) {
    lc::NullDeviceBackend backend(make_ref(0, 0));
    EXPECT_NO_THROW(backend.set_device());
}

TEST(NullDeviceBackend, GemmIsNoop) {
    lc::NullDeviceBackend backend(make_ref(0, 0));
    lc::Fp8GemmParams params{};
    EXPECT_NO_THROW(backend.gemm(params, nullptr, nullptr));
}

TEST(NullDeviceBackend, RmsnormIsNoop) {
    lc::NullDeviceBackend backend(make_ref(0, 0));
    EXPECT_NO_THROW(backend.rmsnorm(nullptr, nullptr, nullptr, 1e-5f, 1, 64, 64, nullptr));
}

TEST(NullDeviceBackend, QuantizeFp8IsNoop) {
    lc::NullDeviceBackend backend(make_ref(0, 0));
    lc::DynamicFp8QuantParams params{};
    EXPECT_NO_THROW(backend.quantize_fp8(params, nullptr));
}

TEST(NullDeviceBackend, AllocFree) {
    lc::NullDeviceBackend backend(make_ref(0, 0));
    void* ptr = backend.device_alloc(1024);
    ASSERT_NE(ptr, nullptr);
    EXPECT_NO_THROW(backend.device_free(ptr));
}

TEST(NullDeviceBackend, FreeNull) {
    lc::NullDeviceBackend backend(make_ref(0, 0));
    EXPECT_NO_THROW(backend.device_free(nullptr));
}

// ── Group 2: Per-GPU instantiation (INV-BH-1) ──────────────────────────────

TEST(NullDeviceBackend, PerGpuInstantiation) {
    lc::NullDeviceBackend b0(make_ref(0, 0));
    lc::NullDeviceBackend b1(make_ref(1, 1));
    EXPECT_EQ(b0.gpu().position, 0);
    EXPECT_EQ(b1.gpu().position, 1);
    EXPECT_NE(&b0, &b1);
}

TEST(NullDeviceBackend, FactoryReturnsUniqueInstances) {
    auto p0 = lc::make_null_device_backend(make_ref(0, 0));
    auto p1 = lc::make_null_device_backend(make_ref(1, 1));
    ASSERT_NE(p0, nullptr);
    ASSERT_NE(p1, nullptr);
    EXPECT_NE(p0.get(), p1.get());
    EXPECT_EQ(p0->gpu().position, 0);
    EXPECT_EQ(p1->gpu().position, 1);
}

// ── Group 3: Polymorphism ───────────────────────────────────────────────────

TEST(DeviceBackendPolymorphism, ThroughBasePointer) {
    auto backend = lc::make_null_device_backend(make_ref(2, 3));
    lc::DeviceBackend* base = backend.get();

    EXPECT_EQ(base->gpu().position, 2);
    EXPECT_EQ(base->gpu().id, 3);
    EXPECT_NO_THROW(base->set_device());

    lc::Fp8GemmParams gp{};
    EXPECT_NO_THROW(base->gemm(gp, nullptr, nullptr));
    EXPECT_NO_THROW(base->rmsnorm(nullptr, nullptr, nullptr, 1e-5f, 1, 64, 64, nullptr));

    lc::DynamicFp8QuantParams qp{};
    EXPECT_NO_THROW(base->quantize_fp8(qp, nullptr));

    void* ptr = base->device_alloc(256);
    ASSERT_NE(ptr, nullptr);
    base->device_free(ptr);
}

TEST(DeviceBackendPolymorphism, UniquePtrLifetime) {
    std::unique_ptr<lc::DeviceBackend> p = lc::make_null_device_backend(make_ref(0, 0));
    EXPECT_EQ(p->gpu().position, 0);
    // Destructor runs when p goes out of scope — no crash.
}

// ── Group 4: CudaSm120DeviceBackend (GPU required) ─────────────────────────

static bool has_cuda_gpu() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

TEST(CudaSm120DeviceBackend, ConstructAndSetDevice) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    lc::CudaSm120DeviceBackend backend(make_ref(0, 0));
    EXPECT_NO_THROW(backend.set_device());
}

TEST(CudaSm120DeviceBackend, GpuReturnsConstructorRef) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    auto ref = make_ref(1, 0, layerstorm::config::GpuType::rtx5080);
    lc::CudaSm120DeviceBackend backend(ref);
    EXPECT_EQ(backend.gpu().position, 1);
    EXPECT_EQ(backend.gpu().id, 0);
    EXPECT_EQ(backend.gpu().type, layerstorm::config::GpuType::rtx5080);
}

TEST(CudaSm120DeviceBackend, AllocFree) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    lc::CudaSm120DeviceBackend backend(make_ref(0, 0));
    backend.set_device();
    void* ptr = backend.device_alloc(4096);
    ASSERT_NE(ptr, nullptr);
    EXPECT_NO_THROW(backend.device_free(ptr));
}

TEST(CudaSm120DeviceBackend, FreeNull) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    lc::CudaSm120DeviceBackend backend(make_ref(0, 0));
    EXPECT_NO_THROW(backend.device_free(nullptr));
}

TEST(CudaSm120DeviceBackend, FactoryPattern) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    auto p = lc::make_cuda_sm120_device_backend(make_ref(0, 0));
    ASSERT_NE(p, nullptr);
    EXPECT_NO_THROW(p->set_device());
    EXPECT_EQ(p->gpu().position, 0);
}
