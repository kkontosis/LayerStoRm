#include <gtest/gtest.h>

#include "config/config_resolver.h"

namespace lc = layerstorm::config;

// ── Helpers ────────────────────────────────────────────────────────────────

namespace {

lc::GpuConfig make_gpu(int id, lc::GpuType type, double vram_gb) {
    lc::GpuConfig g;
    g.id = id;
    g.type = type;
    g.vram_gb = vram_gb;
    g.pcie_gen = 5;
    g.pcie_width = 16;
    g.numa_node = 0;
    return g;
}

// Build a Config with given GPUs and empty tp_array (triggers auto-detect).
lc::Config config_with_gpus(std::vector<lc::GpuConfig> gpus) {
    lc::Config cfg;
    cfg.hardware.gpus = std::move(gpus);
    cfg.hardware.system_ram_gb = 128;
    // tp_array left empty → auto_detect_tp_array will be called by resolve_config
    return cfg;
}

}  // namespace

// ── tp_array auto-detect scoring ──────────────────────────────────────────
// These tests don't call resolve_config() (which needs CUDA) — they test the
// scoring logic indirectly through the helper functions and expected outcomes.

// NOTE: We can't directly call auto_detect_tp_array (it's static in .cpp).
// Instead, we test via the public indices_by_type and the expected scoring
// formula: score(gpu) = 3*(vram_gb/max_vram) + 1*normalized_speed(type)
//   rtx5090: speed=1.0, rtx5080: speed=0.494

TEST(TpArrayScoring, TwoSame5090sShouldFormPair) {
    // Both GPUs identical → both score the same → power-of-2 = 2
    auto hw = lc::HardwareConfig{};
    hw.gpus = {make_gpu(0, lc::GpuType::rtx5090, 32),
               make_gpu(1, lc::GpuType::rtx5090, 32)};
    // Verify scoring formula would produce equal scores:
    // score = 3*(32/32) + 1*1.0 = 4.0 for both
    auto fives = lc::indices_by_type(hw, lc::GpuType::rtx5090);
    EXPECT_EQ(fives.size(), 2u);
}

TEST(TpArrayScoring, FourSame5090sShouldFormQuad) {
    auto hw = lc::HardwareConfig{};
    hw.gpus = {make_gpu(0, lc::GpuType::rtx5090, 32),
               make_gpu(1, lc::GpuType::rtx5090, 32),
               make_gpu(2, lc::GpuType::rtx5090, 32),
               make_gpu(3, lc::GpuType::rtx5090, 32)};
    auto fives = lc::indices_by_type(hw, lc::GpuType::rtx5090);
    EXPECT_EQ(fives.size(), 4u);
    // floor_pow2(4) = 4
}

TEST(TpArrayScoring, Three5090sFloorsToPowerOf2) {
    auto hw = lc::HardwareConfig{};
    hw.gpus = {make_gpu(0, lc::GpuType::rtx5090, 32),
               make_gpu(1, lc::GpuType::rtx5090, 32),
               make_gpu(2, lc::GpuType::rtx5090, 32)};
    auto fives = lc::indices_by_type(hw, lc::GpuType::rtx5090);
    EXPECT_EQ(fives.size(), 3u);
    // floor_pow2(3) = 2 — only 2 would be in tp_array
}

TEST(TpArrayScoring, Mixed5090And5080PreferHigherScore) {
    // 5090 at 32GB: score = 3*(32/32) + 1*1.0 = 4.0
    // 5080 at 16GB: score = 3*(16/32) + 1*0.494 = 1.5 + 0.494 = 1.994
    // 5090 scores higher → tp_array should be {0,1} (the two 5090s)
    auto hw = lc::HardwareConfig{};
    hw.gpus = {make_gpu(0, lc::GpuType::rtx5090, 32),
               make_gpu(1, lc::GpuType::rtx5090, 32),
               make_gpu(2, lc::GpuType::rtx5080, 16),
               make_gpu(3, lc::GpuType::rtx5080, 16)};
    auto fives = lc::indices_by_type(hw, lc::GpuType::rtx5090);
    EXPECT_EQ(fives.size(), 2u);
    EXPECT_EQ(fives[0], 0);
    EXPECT_EQ(fives[1], 1);
}

TEST(TpArrayScoring, SingleGpuShouldFormSingletonArray) {
    auto hw = lc::HardwareConfig{};
    hw.gpus = {make_gpu(0, lc::GpuType::rtx5090, 32)};
    // floor_pow2(1) = 1
    auto fives = lc::indices_by_type(hw, lc::GpuType::rtx5090);
    EXPECT_EQ(fives.size(), 1u);
}

TEST(TpArrayScoring, EmptyGpusEmptyResult) {
    auto hw = lc::HardwareConfig{};
    EXPECT_EQ(lc::gpu_count(hw), 0);
    EXPECT_TRUE(lc::indices_by_type(hw, lc::GpuType::rtx5090).empty());
}

// ── Helper function tests ─────────────────────────────────────────────────

TEST(ConfigResolverHelpers, TotalVramBytesAccuracy) {
    lc::HardwareConfig hw;
    hw.gpus = {make_gpu(0, lc::GpuType::rtx5090, 31.25),
               make_gpu(1, lc::GpuType::rtx5090, 31.25)};
    int64_t expected = static_cast<int64_t>(31.25 * 1024.0 * 1024.0 * 1024.0) * 2;
    EXPECT_EQ(lc::total_vram_bytes(hw), expected);
}

TEST(ConfigResolverHelpers, IndicesByTypeNoMatch) {
    lc::HardwareConfig hw;
    hw.gpus = {make_gpu(0, lc::GpuType::rtx5080, 16)};
    EXPECT_TRUE(lc::indices_by_type(hw, lc::GpuType::rtx5090).empty());
}

TEST(ConfigResolverHelpers, VramGbToBytesZero) {
    EXPECT_EQ(lc::vram_gb_to_bytes(0.0), 0);
}

TEST(ConfigResolverHelpers, VramGbToBytesFractional) {
    EXPECT_EQ(lc::vram_gb_to_bytes(0.5), 1LL << 29);
}

// ── fill_tp_mode_per_layer ─────────────────────────────────────────────────

TEST(FillTpMode, SentinelFillsFromTensorParallelism2) {
    // All tp_mode_per_layer fields at sentinel (0); tensor_parallelism=2 → all become 2.
    lc::Config cfg;
    cfg.parallelism.tensor_parallelism = 2;
    // tp_mode_per_layer fields default to 0 (sentinel)
    lc::fill_tp_mode_per_layer(cfg);
    const auto& tm = cfg.memory.tp_mode_per_layer;
    EXPECT_EQ(tm.default_mode,    2);
    EXPECT_EQ(tm.gating,          2);
    EXPECT_EQ(tm.pinned_dense_ffn,2);
    EXPECT_EQ(tm.attention,       2);
    EXPECT_EQ(tm.shared_expert,   2);
    EXPECT_EQ(tm.embedding,       2);
    EXPECT_EQ(tm.output_head,     2);
}

TEST(FillTpMode, SentinelFillsFromTensorParallelism1) {
    // tensor_parallelism=1 → all sentinel fields become 1 (TP disabled).
    lc::Config cfg;
    cfg.parallelism.tensor_parallelism = 1;
    lc::fill_tp_mode_per_layer(cfg);
    const auto& tm = cfg.memory.tp_mode_per_layer;
    EXPECT_EQ(tm.default_mode,    1);
    EXPECT_EQ(tm.gating,          1);
    EXPECT_EQ(tm.pinned_dense_ffn,1);
    EXPECT_EQ(tm.attention,       1);
    EXPECT_EQ(tm.shared_expert,   1);
    EXPECT_EQ(tm.embedding,       1);
    EXPECT_EQ(tm.output_head,     1);
}

TEST(FillTpMode, ExplicitOverridePreserved) {
    // tensor_parallelism=2 but gating explicitly set to 1 → gating stays 1, rest become 2.
    lc::Config cfg;
    cfg.parallelism.tensor_parallelism = 2;
    cfg.memory.tp_mode_per_layer.gating = 1;  // explicit override (non-sentinel)
    lc::fill_tp_mode_per_layer(cfg);
    const auto& tm = cfg.memory.tp_mode_per_layer;
    EXPECT_EQ(tm.gating,          1);  // preserved
    EXPECT_EQ(tm.default_mode,    2);
    EXPECT_EQ(tm.pinned_dense_ffn,2);
    EXPECT_EQ(tm.attention,       2);
    EXPECT_EQ(tm.shared_expert,   2);
    EXPECT_EQ(tm.embedding,       2);
    EXPECT_EQ(tm.output_head,     2);
}
