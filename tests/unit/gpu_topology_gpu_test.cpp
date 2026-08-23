#include "../gpu_test_utils.h"

#include "config/config_resolver.h"

namespace lc = layerstorm::config;

// ── resolve_config: basic sanity ─────────────────────────────────────────

static lc::Config make_skeleton_config(int gpu_count) {
    lc::Config cfg;
    for (int i = 0; i < gpu_count; ++i) {
        lc::GpuConfig g;
        g.id = i;
        // Leave all fields at sentinel values for auto-detect.
        cfg.hardware.gpus.push_back(g);
    }
    return cfg;
}

TEST(ConfigResolverGpu, ResolveFillsVramGb) {
    REQUIRES_GPU();
    int count = 0;
    cudaGetDeviceCount(&count);
    auto cfg = make_skeleton_config(count);
    lc::resolve_config(cfg);
    for (const auto& g : cfg.hardware.gpus) {
        EXPECT_GT(g.vram_gb, 0.0)
            << "GPU " << g.id << " vram_gb not filled";
        // Sanity: 1 GiB < vram < 256 GiB
        EXPECT_GE(g.vram_gb, 1.0);
        EXPECT_LE(g.vram_gb, 256.0);
    }
}

TEST(ConfigResolverGpu, ResolveFillsPciBusId) {
    REQUIRES_GPU();
    int count = 0;
    cudaGetDeviceCount(&count);
    auto cfg = make_skeleton_config(count);
    lc::resolve_config(cfg);
    for (const auto& g : cfg.hardware.gpus)
        EXPECT_FALSE(g.pci_bus_id.empty()) << "GPU " << g.id << " has empty PCI bus ID";
}

TEST(ConfigResolverGpu, ResolveFillsSystemRamGb) {
    REQUIRES_GPU();
    auto cfg = make_skeleton_config(1);
    lc::resolve_config(cfg);
    EXPECT_GT(cfg.hardware.system_ram_gb, 0);
}

// ── PCIe bandwidth sanity ─────────────────────────────────────────────────

namespace {
int64_t pcie_lane_mb_per_sec(int gen) {
    switch (gen) {
        case 1: return 250;
        case 2: return 500;
        case 3: return 985;
        case 4: return 1969;
        case 5: return 3938;
        case 6: return 7877;
        default: return 0;
    }
}
}  // namespace

TEST(ConfigResolverGpu, PcieBandwidthInSaneRange) {
    REQUIRES_GPU();
    int count = 0;
    cudaGetDeviceCount(&count);
    auto cfg = make_skeleton_config(count);
    lc::resolve_config(cfg);
    for (const auto& g : cfg.hardware.gpus) {
        if (g.pcie_gen == 0 || g.pcie_width == 0) continue;
        int64_t bw_mb = pcie_lane_mb_per_sec(g.pcie_gen) * g.pcie_width;
        EXPECT_GE(bw_mb, 985LL)
            << "GPU " << g.id << " bandwidth implausibly low: " << bw_mb
            << " MB/s (gen=" << g.pcie_gen << " x" << g.pcie_width << ")";
        EXPECT_LE(bw_mb, 130'000LL)
            << "GPU " << g.id << " bandwidth implausibly high: " << bw_mb << " MB/s";
    }
}

TEST(ConfigResolverGpu, MaxBandwidthGeCurrentBandwidth) {
    REQUIRES_GPU();
    int count = 0;
    cudaGetDeviceCount(&count);
    auto cfg = make_skeleton_config(count);
    lc::resolve_config(cfg);
    for (const auto& g : cfg.hardware.gpus) {
        if (g.pcie_gen == 0 || g.pcie_gen_max == 0) continue;
        int64_t bw_cur = pcie_lane_mb_per_sec(g.pcie_gen) * g.pcie_width;
        int64_t bw_max = pcie_lane_mb_per_sec(g.pcie_gen_max) * g.pcie_width_max;
        EXPECT_LE(bw_cur, bw_max)
            << "GPU " << g.id << " current bandwidth " << bw_cur
            << " MB/s exceeds device max " << bw_max << " MB/s (impossible)";
    }
}

// ── vram_gb override ───────────────────────────────────────────────────

TEST(ConfigResolverGpu, VramOverrideRespected) {
    REQUIRES_GPU();
    lc::Config cfg;
    lc::GpuConfig gc;
    gc.id = 0;
    gc.vram_gb = 1.0;  // deliberately tiny — clearly differs from real hardware
    cfg.hardware.gpus.push_back(gc);
    lc::resolve_config(cfg);
    EXPECT_DOUBLE_EQ(cfg.hardware.gpus[0].vram_gb, 1.0);
}

TEST(ConfigResolverGpu, ZeroVramGbUsesHardware) {
    REQUIRES_GPU();
    // First resolve with auto-detect to get real value.
    lc::Config real_cfg;
    lc::GpuConfig gc_real;
    gc_real.id = 0;
    gc_real.vram_gb = 0.0;  // sentinel: auto-detect
    real_cfg.hardware.gpus.push_back(gc_real);
    lc::resolve_config(real_cfg);

    // Second resolve with explicit override.
    lc::Config override_cfg;
    lc::GpuConfig gc_override;
    gc_override.id = 0;
    gc_override.vram_gb = 0.0;  // sentinel: auto-detect again
    override_cfg.hardware.gpus.push_back(gc_override);
    lc::resolve_config(override_cfg);

    EXPECT_DOUBLE_EQ(real_cfg.hardware.gpus[0].vram_gb,
                     override_cfg.hardware.gpus[0].vram_gb)
        << "Zero vram_gb (sentinel) should fall back to same hardware value";
}
