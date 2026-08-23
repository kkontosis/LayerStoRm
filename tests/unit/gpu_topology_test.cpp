#include <gtest/gtest.h>

#include "config/config_resolver.h"

namespace lc = layerstorm::config;

// ── Helpers ────────────────────────────────────────────────────────────────

namespace {

lc::GpuConfig make_gpu(int id, lc::GpuType type, int vram_gb,
                        int pcie_gen = 5, int pcie_width = 16, int numa_node = 0) {
    lc::GpuConfig g;
    g.id = id;
    g.type = type;
    g.vram_gb = vram_gb;
    g.pcie_gen = pcie_gen;
    g.pcie_width = pcie_width;
    g.numa_node = numa_node;
    return g;
}

// Standard 4-GPU config: 2x5090 (NUMA 0) + 2x5080 (NUMA 1), explicit TP array.
lc::HardwareConfig standard_hw() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 256;
    hw.gpus = {
        make_gpu(0, lc::GpuType::rtx5090, 32, 5, 16, 0),
        make_gpu(1, lc::GpuType::rtx5090, 32, 5, 16, 0),
        make_gpu(2, lc::GpuType::rtx5080, 16, 5, 16, 1),
        make_gpu(3, lc::GpuType::rtx5080, 16, 5, 16, 1),
    };
    hw.tp_array = {0, 1};
    return hw;
}

}  // namespace

// ── Hardware query helpers ──────────────────────────────────────────────

TEST(HardwareHelpers, GpuCount) {
    auto hw = standard_hw();
    EXPECT_EQ(lc::gpu_count(hw), 4);
}

TEST(HardwareHelpers, TotalVramBytes) {
    auto hw = standard_hw();
    // 2x32 GiB + 2x16 GiB = 96 GiB
    int64_t expected = (2 * 32LL + 2 * 16LL) * (1LL << 30);
    EXPECT_EQ(lc::total_vram_bytes(hw), expected);
}

TEST(HardwareHelpers, IndicesByType5090) {
    auto hw = standard_hw();
    auto fives = lc::indices_by_type(hw, lc::GpuType::rtx5090);
    ASSERT_EQ(fives.size(), 2u);
    EXPECT_EQ(fives[0], 0);
    EXPECT_EQ(fives[1], 1);
}

TEST(HardwareHelpers, IndicesByType5080) {
    auto hw = standard_hw();
    auto eighties = lc::indices_by_type(hw, lc::GpuType::rtx5080);
    ASSERT_EQ(eighties.size(), 2u);
    EXPECT_EQ(eighties[0], 2);
    EXPECT_EQ(eighties[1], 3);
}

TEST(HardwareHelpers, HasTpArray) {
    auto hw = standard_hw();
    EXPECT_TRUE(lc::has_tp_array(hw));

    hw.tp_array.clear();
    EXPECT_FALSE(lc::has_tp_array(hw));
}

TEST(HardwareHelpers, TpDegree) {
    auto hw = standard_hw();
    EXPECT_EQ(lc::tp_degree(hw), 2);

    hw.tp_array.clear();
    EXPECT_EQ(lc::tp_degree(hw), 0);
}

TEST(HardwareHelpers, VramGbToBytes) {
    EXPECT_EQ(lc::vram_gb_to_bytes(32.0), 32LL * (1LL << 30));
    EXPECT_EQ(lc::vram_gb_to_bytes(0.0), 0);
}

// ── GpuConfig fields ──────────────────────────────────────────────────────

TEST(GpuConfigFields, DeviceIds) {
    auto hw = standard_hw();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(hw.gpus[i].id, i);
}

TEST(GpuConfigFields, GpuTypes) {
    auto hw = standard_hw();
    EXPECT_EQ(hw.gpus[0].type, lc::GpuType::rtx5090);
    EXPECT_EQ(hw.gpus[1].type, lc::GpuType::rtx5090);
    EXPECT_EQ(hw.gpus[2].type, lc::GpuType::rtx5080);
    EXPECT_EQ(hw.gpus[3].type, lc::GpuType::rtx5080);
}

TEST(GpuConfigFields, VramGb) {
    auto hw = standard_hw();
    EXPECT_DOUBLE_EQ(hw.gpus[0].vram_gb, 32.0);
    EXPECT_DOUBLE_EQ(hw.gpus[2].vram_gb, 16.0);
}

TEST(GpuConfigFields, PcieInfo) {
    auto hw = standard_hw();
    for (const auto& g : hw.gpus) {
        EXPECT_EQ(g.pcie_gen, 5);
        EXPECT_EQ(g.pcie_width, 16);
    }
}

TEST(GpuConfigFields, InternalFieldsDefault) {
    lc::GpuConfig g;
    EXPECT_EQ(g.pcie_gen_max, 0);
    EXPECT_EQ(g.pcie_width_max, 0);
    EXPECT_TRUE(g.pci_bus_id.empty());
}

TEST(GpuConfigFields, PciBusIdEmptyByDefault) {
    auto hw = standard_hw();
    for (const auto& g : hw.gpus)
        EXPECT_TRUE(g.pci_bus_id.empty());
}

// ── NUMA mapping ──────────────────────────────────────────────────────────

TEST(HardwareHelpers, NumaMappingPerGpu) {
    auto hw = standard_hw();
    EXPECT_EQ(hw.gpus[0].numa_node, 0);
    EXPECT_EQ(hw.gpus[1].numa_node, 0);
    EXPECT_EQ(hw.gpus[2].numa_node, 1);
    EXPECT_EQ(hw.gpus[3].numa_node, 1);
}

TEST(HardwareHelpers, NumaNodePreservedExactly) {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 128;
    hw.gpus = {make_gpu(0, lc::GpuType::rtx5090, 32, 5, 16, 3)};
    EXPECT_EQ(hw.gpus[0].numa_node, 3);
}

// ── TP array ──────────────────────────────────────────────────────────────

TEST(HardwareHelpers, TpArrayPresent) {
    auto hw = standard_hw();
    ASSERT_TRUE(lc::has_tp_array(hw));
    ASSERT_EQ(hw.tp_array.size(), 2u);
    EXPECT_EQ(hw.tp_array[0], 0);
    EXPECT_EQ(hw.tp_array[1], 1);
}

TEST(HardwareHelpers, TpArrayPointsTo5090s) {
    auto hw = standard_hw();
    ASSERT_TRUE(lc::has_tp_array(hw));
    int a = hw.tp_array[0];
    int b = hw.tp_array[1];
    EXPECT_EQ(hw.gpus[a].type, lc::GpuType::rtx5090);
    EXPECT_EQ(hw.gpus[b].type, lc::GpuType::rtx5090);
}

TEST(HardwareHelpers, NoTpArrayWhenNotConfigured) {
    auto hw = standard_hw();
    hw.tp_array.clear();
    EXPECT_FALSE(lc::has_tp_array(hw));
}

TEST(HardwareHelpers, TpArrayCustomIndices) {
    auto hw = standard_hw();
    hw.tp_array = {1, 3};
    ASSERT_TRUE(lc::has_tp_array(hw));
    ASSERT_EQ(hw.tp_array.size(), 2u);
    EXPECT_EQ(hw.tp_array[0], 1);
    EXPECT_EQ(hw.tp_array[1], 3);
}

// ── Edge cases ─────────────────────────────────────────────────────────

TEST(HardwareHelpers, EmptyHardware) {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 0;
    EXPECT_EQ(lc::gpu_count(hw), 0);
    EXPECT_FALSE(lc::has_tp_array(hw));
    EXPECT_EQ(lc::total_vram_bytes(hw), 0);
    EXPECT_TRUE(lc::indices_by_type(hw, lc::GpuType::rtx5090).empty());
    EXPECT_TRUE(lc::indices_by_type(hw, lc::GpuType::rtx5080).empty());
}

TEST(HardwareHelpers, Single5090NoTpArray) {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 128;
    hw.gpus = {make_gpu(0, lc::GpuType::rtx5090, 32)};
    EXPECT_EQ(lc::gpu_count(hw), 1);
    EXPECT_FALSE(lc::has_tp_array(hw));
    EXPECT_EQ(lc::total_vram_bytes(hw), 32LL * (1LL << 30));
}

TEST(HardwareHelpers, Only5080sNoTpArray) {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 128;
    hw.gpus = {
        make_gpu(0, lc::GpuType::rtx5080, 16),
        make_gpu(1, lc::GpuType::rtx5080, 16),
    };
    EXPECT_EQ(lc::gpu_count(hw), 2);
    EXPECT_FALSE(lc::has_tp_array(hw));
    EXPECT_TRUE(lc::indices_by_type(hw, lc::GpuType::rtx5090).empty());
    ASSERT_EQ(lc::indices_by_type(hw, lc::GpuType::rtx5080).size(), 2u);
}

// ── PCIe gen round-trip ────────────────────────────────────────────────

TEST(GpuConfigFields, PcieGen1PreservedNotMistakenForGen2) {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 32;
    hw.gpus = {make_gpu(0, lc::GpuType::rtx5090, 32, 1, 16, 0)};
    EXPECT_EQ(hw.gpus[0].pcie_gen, 1);
}

TEST(GpuConfigFields, MixedPcieGenerations) {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 64;
    hw.gpus = {
        make_gpu(0, lc::GpuType::rtx5090, 32, 5, 16, 0),
        make_gpu(1, lc::GpuType::rtx5080, 16, 4, 8, 0),
    };
    EXPECT_EQ(hw.gpus[0].pcie_gen, 5);
    EXPECT_EQ(hw.gpus[0].pcie_width, 16);
    EXPECT_EQ(hw.gpus[1].pcie_gen, 4);
    EXPECT_EQ(hw.gpus[1].pcie_width, 8);
}
