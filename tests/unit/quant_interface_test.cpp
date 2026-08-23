#include <gtest/gtest.h>

#include <memory>

#include "model/quantization/quant_interface.h"

using namespace layerstorm::model;
using namespace layerstorm;

// ── Mock implementations ─────────────────────────────────────────────────────

class MockNvfp4 : public QuantInterface {
public:
    std::string_view name() const override { return "nvfp4"; }
    config::WeightQuant weight_quant() const override { return config::WeightQuant::nvfp4; }

    int64_t bytes_per_expert(const ExpertShape& shape) const override {
        return bytes_per_projection(shape, Projection::gate)
             + bytes_per_projection(shape, Projection::up)
             + bytes_per_projection(shape, Projection::down);
    }

    int64_t bytes_per_projection(const ExpertShape& shape, Projection proj) const override {
        int64_t params = 0;
        switch (proj) {
            case Projection::gate: params = shape.gate_params(); break;
            case Projection::up:   params = shape.up_params();   break;
            case Projection::down: params = shape.down_params(); break;
        }
        // 4 bits per weight + UE8M0 scale per 128-element group (1 byte each)
        int64_t weight_bytes = (params + 1) / 2; // ceil(params * 4 / 8)
        int64_t scale_bytes  = (params + 127) / 128; // one byte per group
        return weight_bytes + scale_bytes;
    }

    int64_t weight_bytes_per_projection(const ExpertShape& shape, Projection proj) const override {
        int64_t params = 0;
        switch (proj) {
            case Projection::gate: params = shape.gate_params(); break;
            case Projection::up:   params = shape.up_params();   break;
            case Projection::down: params = shape.down_params(); break;
        }
        return (params + 1) / 2;
    }

    double bytes_per_element() const override { return 0.5 + 1.0 / 128.0; }
    double dequant_flops_per_element() const override { return 2.0; }
    bool tensor_core_eligible() const override { return true; }

    MemoryLayout memory_layout() const override {
        return {4, 128, false, ScaleDtype::ue8m0, 128};
    }
};

class MockFp8 : public QuantInterface {
public:
    std::string_view name() const override { return "fp8_e4m3"; }
    config::WeightQuant weight_quant() const override { return config::WeightQuant::fp8_e4m3; }

    int64_t bytes_per_expert(const ExpertShape& shape) const override {
        return bytes_per_projection(shape, Projection::gate)
             + bytes_per_projection(shape, Projection::up)
             + bytes_per_projection(shape, Projection::down);
    }

    int64_t bytes_per_projection(const ExpertShape& shape, Projection proj) const override {
        return weight_bytes_per_projection(shape, proj) + scale_bytes(shape, proj);
    }

    int64_t weight_bytes_per_projection(const ExpertShape& shape, Projection proj) const override {
        switch (proj) {
            case Projection::gate: return shape.gate_params();
            case Projection::up:   return shape.up_params();
            case Projection::down: return shape.down_params();
        }
        return 0;
    }

    double bytes_per_element() const override { return 1.0 + 4.0 / (128.0 * 128.0); }
    double dequant_flops_per_element() const override { return 1.0; }
    bool tensor_core_eligible() const override { return true; }

    MemoryLayout memory_layout() const override {
        return {8, 0, false, ScaleDtype::none, 64};
    }

private:
    static int64_t scale_bytes(const ExpertShape& shape, Projection proj) {
        int64_t N, K;
        switch (proj) {
            case Projection::gate: [[fallthrough]];
            case Projection::up:   N = shape.intermediate_size; K = shape.hidden_size; break;
            case Projection::down: N = shape.hidden_size; K = shape.intermediate_size; break;
        }
        return ((N + 127) / 128) * ((K + 127) / 128) * int64_t{4};
    }
};

// ── ExpertShape tests ────────────────────────────────────────────────────────

TEST(QuantInterface, ExpertShapeV32) {
    constexpr ExpertShape v3{7168, 2048};
    static_assert(v3.gate_params() == 7168LL * 2048);
    static_assert(v3.up_params()   == 7168LL * 2048);
    static_assert(v3.down_params() == 2048LL * 7168);
    static_assert(v3.total_params() == 3LL * 7168 * 2048);
    EXPECT_EQ(v3.total_params(), 44'040'192);
}

TEST(QuantInterface, ExpertShapeGlm5) {
    constexpr ExpertShape glm{6144, 2048};
    static_assert(glm.total_params() == 3LL * 6144 * 2048);
    EXPECT_EQ(glm.total_params(), 37'748'736);
}

// ── Mock construction & method calls ─────────────────────────────────────────

TEST(QuantInterface, MockNvfp4Methods) {
    MockNvfp4 q;
    EXPECT_EQ(q.name(), "nvfp4");
    EXPECT_EQ(q.weight_quant(), config::WeightQuant::nvfp4);
    EXPECT_TRUE(q.tensor_core_eligible());
    EXPECT_DOUBLE_EQ(q.dequant_flops_per_element(), 2.0);

    auto layout = q.memory_layout();
    EXPECT_EQ(layout.bits_per_weight, 4);
    EXPECT_EQ(layout.group_size, 128);
    EXPECT_FALSE(layout.has_zero_point);
    EXPECT_EQ(layout.scale_dtype, ScaleDtype::ue8m0);
    EXPECT_EQ(layout.alignment_bytes, 128);
}

TEST(QuantInterface, MockFp8Methods) {
    MockFp8 q;
    EXPECT_EQ(q.name(), "fp8_e4m3");
    EXPECT_EQ(q.weight_quant(), config::WeightQuant::fp8_e4m3);
    EXPECT_TRUE(q.tensor_core_eligible());
    EXPECT_DOUBLE_EQ(q.dequant_flops_per_element(), 1.0);
    EXPECT_NEAR(q.bytes_per_element(), 1.0, 0.001);  // ~1.000244 (blockwise scale overhead)

    auto layout = q.memory_layout();
    EXPECT_EQ(layout.bits_per_weight, 8);
    EXPECT_EQ(layout.group_size, 0);
    EXPECT_FALSE(layout.has_zero_point);
    EXPECT_EQ(layout.scale_dtype, ScaleDtype::none);
    EXPECT_EQ(layout.alignment_bytes, 64);
}

// ── bytes_per_expert consistency ─────────────────────────────────────────────

TEST(QuantInterface, BytesPerExpertConsistencyNvfp4) {
    MockNvfp4 q;
    ExpertShape shape{7168, 2048};
    int64_t total = q.bytes_per_expert(shape);
    int64_t sum   = q.bytes_per_projection(shape, Projection::gate)
                  + q.bytes_per_projection(shape, Projection::up)
                  + q.bytes_per_projection(shape, Projection::down);
    EXPECT_EQ(total, sum);
    EXPECT_GT(total, 0);
}

TEST(QuantInterface, BytesPerExpertConsistencyFp8) {
    MockFp8 q;
    ExpertShape shape{7168, 2048};
    int64_t total = q.bytes_per_expert(shape);
    int64_t sum   = q.bytes_per_projection(shape, Projection::gate)
                  + q.bytes_per_projection(shape, Projection::up)
                  + q.bytes_per_projection(shape, Projection::down);
    EXPECT_EQ(total, sum);
    // FP8: 1 byte per param + blockwise scales (3584 per projection)
    EXPECT_EQ(total, 3LL * (7168 * 2048 + 3584));
}

// ── Virtual dispatch through base pointer ────────────────────────────────────

TEST(QuantInterface, VirtualDispatch) {
    std::unique_ptr<QuantInterface> q = std::make_unique<MockNvfp4>();
    EXPECT_EQ(q->name(), "nvfp4");

    ExpertShape shape{7168, 2048};
    EXPECT_GT(q->bytes_per_expert(shape), 0);
    EXPECT_TRUE(q->tensor_core_eligible());

    // Swap to FP8
    q = std::make_unique<MockFp8>();
    EXPECT_EQ(q->name(), "fp8_e4m3");
    EXPECT_NEAR(q->bytes_per_element(), 1.0, 0.001);
}

// ── Two mocks through same interface ─────────────────────────────────────────

TEST(QuantInterface, TwoMocksCompare) {
    MockNvfp4 nvfp4;
    MockFp8 fp8;
    ExpertShape shape{7168, 2048};

    // NVFP4 should use roughly half the bytes of FP8
    int64_t nvfp4_bytes = nvfp4.bytes_per_expert(shape);
    int64_t fp8_bytes   = fp8.bytes_per_expert(shape);
    EXPECT_LT(nvfp4_bytes, fp8_bytes);

    // Verify approximate ratio (~0.5 + scale overhead)
    double ratio = static_cast<double>(nvfp4_bytes) / fp8_bytes;
    EXPECT_GT(ratio, 0.49);
    EXPECT_LT(ratio, 0.52);

    // bytes_per_element reflects the difference
    EXPECT_LT(nvfp4.bytes_per_element(), fp8.bytes_per_element());
}

// ── MemoryLayout fields ──────────────────────────────────────────────────────

TEST(QuantInterface, MemoryLayoutFields) {
    MockNvfp4 nvfp4;
    MockFp8 fp8;

    auto nvfp4_layout = nvfp4.memory_layout();
    auto fp8_layout   = fp8.memory_layout();

    // NVFP4: 4-bit weights, 128-element groups, UE8M0 scales
    EXPECT_EQ(nvfp4_layout.bits_per_weight, 4);
    EXPECT_EQ(nvfp4_layout.group_size, 128);
    EXPECT_EQ(nvfp4_layout.scale_dtype, ScaleDtype::ue8m0);
    EXPECT_GT(nvfp4_layout.alignment_bytes, 0);

    // FP8: 8-bit weights, no grouping, no scales
    EXPECT_EQ(fp8_layout.bits_per_weight, 8);
    EXPECT_EQ(fp8_layout.group_size, 0);
    EXPECT_EQ(fp8_layout.scale_dtype, ScaleDtype::none);
    EXPECT_GT(fp8_layout.alignment_bytes, 0);
}
