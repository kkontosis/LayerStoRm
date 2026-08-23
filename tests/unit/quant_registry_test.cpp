#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "model/quantization/registry.h"

using namespace layerstorm::model;
using namespace layerstorm;

// ── Minimal mock implementations (local to this test file) ──────────────────

class MockNvfp4Reg : public QuantInterface {
public:
    std::string_view name() const override { return "nvfp4"; }
    config::WeightQuant weight_quant() const override { return config::WeightQuant::nvfp4; }
    int64_t bytes_per_expert(const ExpertShape&) const override { return 0; }
    int64_t bytes_per_projection(const ExpertShape&, Projection) const override { return 0; }
    int64_t weight_bytes_per_projection(const ExpertShape&, Projection) const override { return 0; }
    double bytes_per_element() const override { return 0.5; }
    double dequant_flops_per_element() const override { return 2.0; }
    bool tensor_core_eligible() const override { return true; }
    MemoryLayout memory_layout() const override { return {4, 128, false, ScaleDtype::ue8m0, 128}; }
};

class MockFp8Reg : public QuantInterface {
public:
    std::string_view name() const override { return "fp8_e4m3"; }
    config::WeightQuant weight_quant() const override { return config::WeightQuant::fp8_e4m3; }
    int64_t bytes_per_expert(const ExpertShape&) const override { return 0; }
    int64_t bytes_per_projection(const ExpertShape&, Projection) const override { return 0; }
    int64_t weight_bytes_per_projection(const ExpertShape&, Projection) const override { return 0; }
    double bytes_per_element() const override { return 1.0; }
    double dequant_flops_per_element() const override { return 1.0; }
    bool tensor_core_eligible() const override { return true; }
    MemoryLayout memory_layout() const override { return {8, 0, false, ScaleDtype::none, 64}; }
};

// ── Fixture ─────────────────────────────────────────────────────────────────

class QuantRegistryTest : public ::testing::Test {
protected:
    void SetUp() override { clear_registry(); }
    void TearDown() override { clear_registry(); }

    static MockNvfp4Reg nvfp4;
    static MockFp8Reg fp8;
};

MockNvfp4Reg QuantRegistryTest::nvfp4;
MockFp8Reg QuantRegistryTest::fp8;

// ── Tests ───────────────────────────────────────────────────────────────────

TEST_F(QuantRegistryTest, RegisterAndLookupByName) {
    register_format(&nvfp4);
    auto* p = find_format("nvfp4");
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->name(), "nvfp4");
    EXPECT_EQ(p, &nvfp4);
}

TEST_F(QuantRegistryTest, RegisterAndLookupByEnum) {
    register_format(&nvfp4);
    auto* p = find_format(config::WeightQuant::nvfp4);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->weight_quant(), config::WeightQuant::nvfp4);
    EXPECT_EQ(p, &nvfp4);
}

TEST_F(QuantRegistryTest, FindUnknownNameReturnsNull) {
    EXPECT_EQ(find_format("nonexistent"), nullptr);
}

TEST_F(QuantRegistryTest, FindUnknownEnumReturnsNull) {
    EXPECT_EQ(find_format(config::WeightQuant::fp8_e5m2), nullptr);
}

TEST_F(QuantRegistryTest, GetUnknownNameThrows) {
    register_format(&nvfp4);
    try {
        get_format("nonexistent");
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("nonexistent"), std::string::npos);
        EXPECT_NE(msg.find("nvfp4"), std::string::npos);
    }
}

TEST_F(QuantRegistryTest, GetUnknownEnumThrows) {
    register_format(&nvfp4);
    try {
        get_format(config::WeightQuant::fp8_e5m2);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("registered:"), std::string::npos);
        EXPECT_NE(msg.find("nvfp4"), std::string::npos);
    }
}

TEST_F(QuantRegistryTest, DuplicateRegistrationThrows) {
    register_format(&nvfp4);
    EXPECT_THROW(register_format(&nvfp4), std::runtime_error);
}

TEST_F(QuantRegistryTest, MultipleFormatsCoexist) {
    register_format(&nvfp4);
    register_format(&fp8);

    EXPECT_EQ(find_format("nvfp4"), &nvfp4);
    EXPECT_EQ(find_format("fp8_e4m3"), &fp8);
    EXPECT_EQ(find_format(config::WeightQuant::nvfp4), &nvfp4);
    EXPECT_EQ(find_format(config::WeightQuant::fp8_e4m3), &fp8);

    // get_format returns references
    EXPECT_EQ(&get_format("nvfp4"), &nvfp4);
    EXPECT_EQ(&get_format("fp8_e4m3"), &fp8);
}

TEST_F(QuantRegistryTest, RegisteredNamesContainsAll) {
    register_format(&nvfp4);
    register_format(&fp8);

    auto names = registered_names();
    EXPECT_EQ(names.size(), 2u);

    std::vector<std::string> sorted;
    for (auto sv : names) sorted.emplace_back(sv);
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted[0], "fp8_e4m3");
    EXPECT_EQ(sorted[1], "nvfp4");
}
