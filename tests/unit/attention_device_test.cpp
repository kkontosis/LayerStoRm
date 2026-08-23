// Unit tests for AttentionDevice abstraction.
//
// Tests NullAttentionDevice construction, polymorphism, factory function,
// and verifies all methods are callable through the base pointer.

#include <gtest/gtest.h>

#include "../gpu_test_utils.h"
#include "core/attention_device.h"
#include "core/null_attention_device.h"
#include "compute/graphs/graph_registry.h"
#include "compute/snapmla_sm120_attention_device.h"
#include "compute/tq_sm120_attention_device.h"

namespace lc = layerstorm::compute;
namespace cfg = layerstorm::config;

namespace {

cfg::GpuRef make_gpu(int pos = 0, int id = 0,
                     cfg::GpuType type = cfg::GpuType::rtx5090) {
    return {pos, id, type};
}

}  // namespace

// ── NullAttentionDevice ─────────────────────────────────────────────────────

TEST(AttentionDeviceTest, NullConstruction) {
    auto dev = lc::make_null_attention_device(make_gpu());
    ASSERT_NE(dev, nullptr);
    EXPECT_EQ(dev->gpu().position, 0);
    EXPECT_EQ(dev->gpu().id, 0);
}

TEST(AttentionDeviceTest, NullGpuIdentity) {
    auto dev = lc::make_null_attention_device(make_gpu(2, 3));
    EXPECT_EQ(dev->gpu().position, 2);
    EXPECT_EQ(dev->gpu().id, 3);
}

TEST(AttentionDeviceTest, NullSetDeviceNoop) {
    auto dev = lc::make_null_attention_device(make_gpu());
    dev->set_device();  // should not crash
}

TEST(AttentionDeviceTest, NullComputeMethodsNoop) {
    auto dev = lc::make_null_attention_device(make_gpu());
    lc::Fp8GemmParams gemm_params{};
    dev->gemm(gemm_params, nullptr, nullptr);
    dev->rmsnorm(nullptr, nullptr, nullptr, 1e-6f, 1, 128, 128, nullptr);
    lc::DynamicFp8QuantParams quant_params{};
    dev->quantize_fp8(quant_params, nullptr);
}

TEST(AttentionDeviceTest, NullAllocFree) {
    auto dev = lc::make_null_attention_device(make_gpu());
    void* ptr = dev->device_alloc(256);
    ASSERT_NE(ptr, nullptr);
    dev->device_free(ptr);
    dev->device_free(nullptr);  // must tolerate nullptr
}

TEST(AttentionDeviceTest, NullAttentionMethodsNoop) {
    auto dev = lc::make_null_attention_device(make_gpu());

    // k_append
    dev->k_append(nullptr, nullptr, nullptr, 0, 0, nullptr, 0, 0, 0, 0, 0, 0, 0, nullptr);

    // prefill_attention
    dev->prefill_attention(nullptr, 0, 0, nullptr, nullptr, 0,
                           nullptr, 0, 0, 0, false, false,
                           nullptr, nullptr, 0,
                           nullptr, nullptr,
                           0, nullptr);

    // decode graph ops — use a dummy GraphEntry
    lc::GraphEntry entry{};
    dev->decode_graph_update(entry, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    dev->decode_graph_replay(entry, nullptr);
    EXPECT_EQ(dev->decode_graph_out_ptr(entry), nullptr);
    EXPECT_EQ(dev->decode_graph_lse_ptr(entry), nullptr);

    // DCP allreduce graph
    dev->dcp_graph_replay(entry, nullptr);
}

// ── Polymorphism ────────────────────────────────────────────────────────────

TEST(AttentionDeviceTest, Polymorphism) {
    std::unique_ptr<lc::AttentionDevice> dev =
        lc::make_null_attention_device(make_gpu(1, 2));
    EXPECT_EQ(dev->gpu().position, 1);
    dev->set_device();
    void* p = dev->device_alloc(64);
    ASSERT_NE(p, nullptr);
    dev->device_free(p);
}

TEST(AttentionDeviceTest, MultipleDifferentGpus) {
    auto dev0 = lc::make_null_attention_device(make_gpu(0, 0));
    auto dev1 = lc::make_null_attention_device(make_gpu(1, 1));
    EXPECT_EQ(dev0->gpu().position, 0);
    EXPECT_EQ(dev1->gpu().position, 1);
    EXPECT_NE(dev0->gpu().id, dev1->gpu().id);
}

TEST(AttentionDeviceTest, UniqueOwnership) {
    auto dev = lc::make_null_attention_device(make_gpu());
    lc::AttentionDevice* raw = dev.get();
    ASSERT_NE(raw, nullptr);
    dev.reset();
    // unique_ptr destroyed cleanly — no double-free or leak
}

// ── SnapMlaSm120AttentionDevice (GPU required) ────────────────────────────

TEST(SnapMlaSm120AttentionDevice, ConstructAndSetDevice) {
    REQUIRES_GPU();
    auto dev = lc::make_snapmla_sm120_attention_device(make_gpu(0, 0));
    ASSERT_NE(dev, nullptr);
    EXPECT_NO_THROW(dev->set_device());
}

TEST(SnapMlaSm120AttentionDevice, GpuReturnsConstructorRef) {
    REQUIRES_GPU();
    auto dev = lc::make_snapmla_sm120_attention_device(
        make_gpu(1, 0, cfg::GpuType::rtx5080));
    EXPECT_EQ(dev->gpu().position, 1);
    EXPECT_EQ(dev->gpu().id, 0);
    EXPECT_EQ(dev->gpu().type, cfg::GpuType::rtx5080);
}

TEST(SnapMlaSm120AttentionDevice, AllocFree) {
    REQUIRES_GPU();
    auto dev = lc::make_snapmla_sm120_attention_device(make_gpu(0, 0));
    dev->set_device();
    void* ptr = dev->device_alloc(4096);
    ASSERT_NE(ptr, nullptr);
    EXPECT_NO_THROW(dev->device_free(ptr));
}

TEST(SnapMlaSm120AttentionDevice, FreeNull) {
    REQUIRES_GPU();
    auto dev = lc::make_snapmla_sm120_attention_device(make_gpu(0, 0));
    EXPECT_NO_THROW(dev->device_free(nullptr));
}

TEST(SnapMlaSm120AttentionDevice, FactoryPattern) {
    REQUIRES_GPU();
    std::unique_ptr<lc::AttentionDevice> dev =
        lc::make_snapmla_sm120_attention_device(make_gpu(0, 0));
    ASSERT_NE(dev, nullptr);
    EXPECT_NO_THROW(dev->set_device());
    EXPECT_EQ(dev->gpu().position, 0);
}

// ── TqSm120AttentionDevice (GPU required) ──────────────────────────────────

TEST(TqSm120AttentionDevice, ConstructAndSetDevice) {
    REQUIRES_GPU();
    auto dev = lc::make_tq_sm120_attention_device(make_gpu(0, 0));
    ASSERT_NE(dev, nullptr);
    EXPECT_NO_THROW(dev->set_device());
}

TEST(TqSm120AttentionDevice, GpuReturnsConstructorRef) {
    REQUIRES_GPU();
    auto dev = lc::make_tq_sm120_attention_device(
        make_gpu(1, 0, cfg::GpuType::rtx5080));
    EXPECT_EQ(dev->gpu().position, 1);
    EXPECT_EQ(dev->gpu().id, 0);
    EXPECT_EQ(dev->gpu().type, cfg::GpuType::rtx5080);
}

TEST(TqSm120AttentionDevice, AllocFree) {
    REQUIRES_GPU();
    auto dev = lc::make_tq_sm120_attention_device(make_gpu(0, 0));
    dev->set_device();
    void* ptr = dev->device_alloc(4096);
    ASSERT_NE(ptr, nullptr);
    EXPECT_NO_THROW(dev->device_free(ptr));
}

TEST(TqSm120AttentionDevice, FreeNull) {
    REQUIRES_GPU();
    auto dev = lc::make_tq_sm120_attention_device(make_gpu(0, 0));
    EXPECT_NO_THROW(dev->device_free(nullptr));
}

TEST(TqSm120AttentionDevice, FactoryPattern) {
    REQUIRES_GPU();
    std::unique_ptr<lc::AttentionDevice> dev =
        lc::make_tq_sm120_attention_device(make_gpu(0, 0));
    ASSERT_NE(dev, nullptr);
    EXPECT_NO_THROW(dev->set_device());
    EXPECT_EQ(dev->gpu().position, 0);
}

// R0H-1d: free-function accessors for TQ-specific setters
TEST(TqSm120AttentionDevice, TqDeviceSetResources) {
    REQUIRES_GPU();
    auto dev = lc::make_tq_sm120_attention_device(make_gpu(0, 0));
    // null TqResources is safe — just stores the pointer
    EXPECT_NO_THROW(lc::tq_device_set_resources(dev.get(), nullptr));
}

TEST(TqSm120AttentionDevice, TqDeviceSetModelDims) {
    REQUIRES_GPU();
    auto dev = lc::make_tq_sm120_attention_device(make_gpu(0, 0));
    EXPECT_NO_THROW(lc::tq_device_set_model_dims(dev.get(), 2, 512, 64, 128));
    EXPECT_NO_THROW(lc::tq_device_set_model_dims(dev.get(), 4, 512, 64, 128, 2));
}

// tq_device_set_layer_context removed in #35h — layer_idx now passed
// directly to k_append / prefill_attention / decode_graph_update.
