#include <gtest/gtest.h>

#include <stdexcept>

#include "core/memory/host_pool_sizing.h"
#include "core/memory/numa_manager.h"

namespace lc = layerstorm::config;
namespace lm = layerstorm::memory;

namespace {

constexpr double kGiB = 1073741824.0;

lc::GpuConfig make_gpu(int id, int numa_node) {
    lc::GpuConfig g;
    g.id = id;
    g.type = lc::GpuType::rtx5090;
    g.vram_gb = 32.0;
    g.numa_node = numa_node;
    return g;
}

// One GPU on node 0 — enough to exercise the resolver. NUMA accessors read /sys
// for node 0 (present on the dev box; gracefully 0 on non-NUMA CI).
lc::HardwareConfig hw_node0() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 128;
    hw.gpus = {make_gpu(0, 0)};
    return hw;
}

lc::PinHostExpertPoolSizingConfig sizing(lc::HostPoolSizingMode mode, double value) {
    lc::PinHostExpertPoolSizingConfig s;
    s.mode = mode;
    s.value = value;
    return s;
}

}  // namespace

// ── absolute_gb (NUMA-independent, fully deterministic) ──────────────────────

TEST(HostPoolSizing, AbsoluteGbExact) {
    lm::NumaManager numa(hw_node0());
    auto s = sizing(lc::HostPoolSizingMode::absolute_gb, 16.0);
    EXPECT_EQ(lm::resolve_node_budget_bytes(s, numa, 0),
              static_cast<size_t>(16.0 * kGiB));
}

TEST(HostPoolSizing, AbsoluteGbHonorsDecimals) {
    lm::NumaManager numa(hw_node0());
    auto s = sizing(lc::HostPoolSizingMode::absolute_gb, 1.5);
    EXPECT_EQ(lm::resolve_node_budget_bytes(s, numa, 0),
              static_cast<size_t>(1.5 * kGiB));
}

// ── per-node override wins over the global setting ───────────────────────────

TEST(HostPoolSizing, PerNodeOverride) {
    lm::NumaManager numa(hw_node0());
    auto s = sizing(lc::HostPoolSizingMode::absolute_gb, 8.0);
    lc::PerNodeConfig pn;
    pn.node = 0;
    pn.mode = lc::HostPoolSizingMode::absolute_gb;
    pn.value = 32.0;
    s.per_node.push_back(pn);

    // node 0 hits the override; any other node falls back to the global value.
    EXPECT_EQ(lm::resolve_node_budget_bytes(s, numa, 0),
              static_cast<size_t>(32.0 * kGiB));
    EXPECT_EQ(lm::resolve_node_budget_bytes(s, numa, 1),
              static_cast<size_t>(8.0 * kGiB));
}

// ── fraction modes: resolver = accessor × value (consistency, not magic value) ─

TEST(HostPoolSizing, FractionTotalMatchesAccessor) {
    lm::NumaManager numa(hw_node0());
    size_t total = numa.node_total_bytes(0);
    if (total == 0) GTEST_SKIP() << "node 0 MemTotal unavailable (non-NUMA host)";
    auto s = sizing(lc::HostPoolSizingMode::fraction_total, 0.5);
    EXPECT_EQ(lm::resolve_node_budget_bytes(s, numa, 0),
              static_cast<size_t>(static_cast<double>(total) * 0.5));
}

TEST(HostPoolSizing, FractionFreeMatchesAccessor) {
    lm::NumaManager numa(hw_node0());
    size_t avail = numa.node_available_bytes(0);
    if (avail == 0) GTEST_SKIP() << "node 0 meminfo unavailable (non-NUMA host)";
    auto s = sizing(lc::HostPoolSizingMode::fraction_free, 0.25);
    EXPECT_EQ(lm::resolve_node_budget_bytes(s, numa, 0),
              static_cast<size_t>(static_cast<double>(avail) * 0.25));
}

// available ≥ free (it adds reclaimable cache) and ≤ total.
TEST(HostPoolSizing, AvailableBetweenFreeAndTotal) {
    lm::NumaManager numa(hw_node0());
    size_t total = numa.node_total_bytes(0);
    if (total == 0) GTEST_SKIP() << "node 0 meminfo unavailable (non-NUMA host)";
    EXPECT_GE(numa.node_available_bytes(0), numa.node_free_bytes(0));
    EXPECT_LE(numa.node_available_bytes(0), total);
}

// ── validation: out-of-range values throw before any allocation ──────────────

TEST(HostPoolSizing, RejectsInvalidFraction) {
    lm::NumaManager numa(hw_node0());
    EXPECT_THROW(lm::resolve_node_budget_bytes(
        sizing(lc::HostPoolSizingMode::fraction_total, 0.0), numa, 0),
        std::runtime_error);
    EXPECT_THROW(lm::resolve_node_budget_bytes(
        sizing(lc::HostPoolSizingMode::fraction_total, 1.5), numa, 0),
        std::runtime_error);
    EXPECT_THROW(lm::resolve_node_budget_bytes(
        sizing(lc::HostPoolSizingMode::fraction_free, -0.1), numa, 0),
        std::runtime_error);
}

TEST(HostPoolSizing, RejectsNonPositiveAbsolute) {
    lm::NumaManager numa(hw_node0());
    EXPECT_THROW(lm::resolve_node_budget_bytes(
        sizing(lc::HostPoolSizingMode::absolute_gb, 0.0), numa, 0),
        std::runtime_error);
}
