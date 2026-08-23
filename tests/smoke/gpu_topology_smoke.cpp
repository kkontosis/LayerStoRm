// Smoke test: resolve_config() on real hardware.
//
// Expected hardware (dev machine):
//   GPU 0: RTX 5090, 31 GiB VRAM, PCIe 5.0 x16, NUMA 2, PCI 0000:6a:00.0
//   GPU 1: RTX 5090, 31 GiB VRAM, PCIe 5.0 x16, NUMA 3, PCI 0000:94:00.0
//   GPU 2: RTX 5080, 15 GiB VRAM, PCIe 5.0 x16, NUMA 0, PCI 0000:16:00.0
//   GPU 3: RTX 5080, 15 GiB VRAM, PCIe 5.0 x16, NUMA 2, PCI 0000:40:00.0
//   TP array: GPU 0 + GPU 1
//   System RAM: ~503 GiB
//
// If this matches your hardware the test passes.
// If your hardware differs, the assertions will fail and show what was found.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "config/config_resolver.h"

namespace lc = layerstorm::config;

// ── Fixture ────────────────────────────────────────────────────────────────

class GpuTopologySmoke : public ::testing::Test {
   protected:
    void SetUp() override {
        std::cout << "\nExpected hardware (dev machine):\n"
                  << "  GPU 0: RTX 5090, 31 GiB, PCIe 5.0 x16, NUMA 2, PCI 0000:6a:00.0\n"
                  << "  GPU 1: RTX 5090, 31 GiB, PCIe 5.0 x16, NUMA 3, PCI 0000:94:00.0\n"
                  << "  GPU 2: RTX 5080, 15 GiB, PCIe 5.0 x16, NUMA 0, PCI 0000:16:00.0\n"
                  << "  GPU 3: RTX 5080, 15 GiB, PCIe 5.0 x16, NUMA 2, PCI 0000:40:00.0\n"
                  << "  TP array: GPU 0 + GPU 1 | System RAM: ~503 GiB\n\n";

        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            GTEST_SKIP() << "No CUDA GPU present — cannot run hardware smoke test";
        }

        // Build skeleton config with one entry per GPU, all sentinels.
        for (int i = 0; i < count; ++i) {
            lc::GpuConfig g;
            g.id = i;
            cfg.hardware.gpus.push_back(g);
        }
        lc::resolve_config(cfg);

        std::cout << "Detected hardware:\n";
        for (const auto& g : cfg.hardware.gpus) {
            std::cout << "  GPU " << g.id << ": "
                      << (g.type == lc::GpuType::rtx5090 ? "RTX 5090" : "RTX 5080")
                      << ", " << static_cast<int>(g.vram_gb) << " GiB"
                      << ", PCIe " << g.pcie_gen << ".0 x" << g.pcie_width
                      << " (max Gen " << g.pcie_gen_max << " x" << g.pcie_width_max << ")"
                      << ", NUMA " << g.numa_node
                      << ", PCI " << g.pci_bus_id << "\n";
        }
        if (lc::has_tp_array(cfg.hardware)) {
            std::cout << "  TP array:";
            for (int id : cfg.hardware.tp_array) std::cout << " GPU " << id;
            std::cout << "\n";
        }
        std::cout << "  System RAM: " << cfg.hardware.system_ram_gb << " GiB\n\n";
    }

    lc::Config cfg;
};

// ── Tests ──────────────────────────────────────────────────────────────────

TEST_F(GpuTopologySmoke, FourGpusDetected) {
    EXPECT_EQ(lc::gpu_count(cfg.hardware), 4);
}

TEST_F(GpuTopologySmoke, TwoRtx5090s) {
    auto fives = lc::indices_by_type(cfg.hardware, lc::GpuType::rtx5090);
    EXPECT_EQ(fives.size(), 2u);
}

TEST_F(GpuTopologySmoke, TwoRtx5080s) {
    auto eighties = lc::indices_by_type(cfg.hardware, lc::GpuType::rtx5080);
    EXPECT_EQ(eighties.size(), 2u);
}

TEST_F(GpuTopologySmoke, Gpu0Is5090With31GiB) {
    ASSERT_GE(lc::gpu_count(cfg.hardware), 1);
    EXPECT_EQ(cfg.hardware.gpus[0].type, lc::GpuType::rtx5090);
    EXPECT_EQ(static_cast<int>(cfg.hardware.gpus[0].vram_gb), 31);
}

TEST_F(GpuTopologySmoke, Gpu1Is5090With31GiB) {
    ASSERT_GE(lc::gpu_count(cfg.hardware), 2);
    EXPECT_EQ(cfg.hardware.gpus[1].type, lc::GpuType::rtx5090);
    EXPECT_EQ(static_cast<int>(cfg.hardware.gpus[1].vram_gb), 31);
}

TEST_F(GpuTopologySmoke, Gpu2Is5080With15GiB) {
    ASSERT_GE(lc::gpu_count(cfg.hardware), 3);
    EXPECT_EQ(cfg.hardware.gpus[2].type, lc::GpuType::rtx5080);
    EXPECT_EQ(static_cast<int>(cfg.hardware.gpus[2].vram_gb), 15);
}

TEST_F(GpuTopologySmoke, Gpu3Is5080With15GiB) {
    ASSERT_GE(lc::gpu_count(cfg.hardware), 4);
    EXPECT_EQ(cfg.hardware.gpus[3].type, lc::GpuType::rtx5080);
    EXPECT_EQ(static_cast<int>(cfg.hardware.gpus[3].vram_gb), 15);
}

TEST_F(GpuTopologySmoke, TpArrayIsGpu0And1) {
    ASSERT_TRUE(lc::has_tp_array(cfg.hardware));
    ASSERT_EQ(cfg.hardware.tp_array.size(), 2u);
    EXPECT_EQ(cfg.hardware.tp_array[0], 0);
    EXPECT_EQ(cfg.hardware.tp_array[1], 1);
}

TEST_F(GpuTopologySmoke, NumaNodesCorrect) {
    ASSERT_GE(lc::gpu_count(cfg.hardware), 4);
    EXPECT_EQ(cfg.hardware.gpus[0].numa_node, 2);
    EXPECT_EQ(cfg.hardware.gpus[1].numa_node, 3);
    EXPECT_EQ(cfg.hardware.gpus[2].numa_node, 0);
    EXPECT_EQ(cfg.hardware.gpus[3].numa_node, 2);
}

TEST_F(GpuTopologySmoke, PciBusIdsPresent) {
    for (const auto& g : cfg.hardware.gpus) {
        EXPECT_FALSE(g.pci_bus_id.empty()) << "GPU " << g.id << " has empty PCI bus ID";
    }
}

TEST_F(GpuTopologySmoke, SystemRamAbove400GiB) {
    EXPECT_GT(cfg.hardware.system_ram_gb, 400);
}

TEST_F(GpuTopologySmoke, TotalVramInExpectedRange) {
    // 2x5090 + 2x5080 after firmware overhead.
    int64_t gib = lc::total_vram_bytes(cfg.hardware) >> 30;
    EXPECT_GE(gib, 88) << "Total VRAM unexpectedly low: " << gib << " GiB";
    EXPECT_LE(gib, 100) << "Total VRAM unexpectedly high: " << gib << " GiB";
}

TEST_F(GpuTopologySmoke, PcieGenCurrentNonZero) {
    for (const auto& g : cfg.hardware.gpus) {
        EXPECT_GT(g.pcie_gen, 0) << "GPU " << g.id << " has unknown current PCIe gen";
    }
}

TEST_F(GpuTopologySmoke, PcieGenMaxNonZero) {
    for (const auto& g : cfg.hardware.gpus) {
        EXPECT_GT(g.pcie_gen_max, 0) << "GPU " << g.id << " has unknown max PCIe gen";
    }
}

TEST_F(GpuTopologySmoke, PcieGenCurrentLeMax) {
    for (const auto& g : cfg.hardware.gpus) {
        EXPECT_LE(g.pcie_gen, g.pcie_gen_max)
            << "GPU " << g.id << ": current gen " << g.pcie_gen
            << " > max gen " << g.pcie_gen_max << " (impossible)";
    }
}

TEST_F(GpuTopologySmoke, PcieGen5x16AfterKickstart) {
    for (const auto& g : cfg.hardware.gpus) {
        EXPECT_EQ(g.pcie_gen, 5)
            << "GPU " << g.id << " current PCIe gen is " << g.pcie_gen
            << " (expected 5 after kickstart)";
        EXPECT_EQ(g.pcie_width, 16)
            << "GPU " << g.id << " current PCIe width is " << g.pcie_width;
        EXPECT_EQ(g.pcie_gen_max, 5)
            << "GPU " << g.id << " max PCIe gen is " << g.pcie_gen_max;
        EXPECT_EQ(g.pcie_width_max, 16)
            << "GPU " << g.id << " max PCIe width is " << g.pcie_width_max;
    }
}
