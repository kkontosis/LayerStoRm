// Unit tests for PinnedExpertArena / PinnedNodeArena (P-24).
//
// These exercise the per-NUMA-node pinned, slab-managed expert arena:
//   - one anonymous mbind'd arena per node, registered once (fast — tens of
//     ms/GB, not s/GB),
//   - slab reserve / free / LRU eviction,
//   - the multi-GPU in-flight refcount that blocks reuse,
//   - move_pages confirming slots land on the expected node.
//
// All tests require a CUDA device (the arena page-locks via cudaHostRegister),
// so the suite is in the unit-CMake GPU filter and skips on headless CI.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <vector>

#ifdef LAYERSTORM_HAS_NUMA
#include <numaif.h>  // move_pages
#endif

#include "../gpu_test_utils.h"
#include "core/memory/numa_manager.h"
#include "core/memory/pinned_expert_arena.h"

namespace lc = layerstorm::config;
namespace lm = layerstorm::memory;

namespace {

lc::GpuConfig make_gpu(int id, int numa_node) {
    lc::GpuConfig g;
    g.id = id;
    g.type = lc::GpuType::rtx5090;
    g.vram_gb = 32.0;
    g.numa_node = numa_node;
    return g;
}

// Mirrors the dev box: 4 GPUs across 3 GPU-attached nodes; node 2 is shared by
// GPU1+GPU2 (share_degree=2) — the single-shared-arena case P-24 cares about.
lc::HardwareConfig dev_box_hw() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 512;
    hw.gpus = {make_gpu(0, 0), make_gpu(1, 2), make_gpu(2, 2), make_gpu(3, 3)};
    hw.tp_array = {1, 2};
    return hw;
}

// Single-node topology for the basic slab/refcount tests (deterministic
// home-node routing: every expert lands on node 0).
lc::HardwareConfig single_node_hw() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 128;
    hw.gpus = {make_gpu(0, 0), make_gpu(1, 0)};
    return hw;
}

constexpr size_t kSlot = 64 * 1024;  // 64 KiB slots — tiny, page-multiple.

// A sizing budget large enough that the per-node RAM ceiling never binds, so the
// slot count is governed by the test's `total` (mirrors the old free×0.85 on a
// box with ample RAM). fraction_total 0.85 of node RAM ≫ the few KiB these tests use.
lc::PinHostExpertPoolSizingConfig big_sizing() {
    lc::PinHostExpertPoolSizingConfig s;
    s.mode = lc::HostPoolSizingMode::fraction_total;
    s.value = 0.85;
    return s;
}

lm::ExpertKey key(uint32_t layer, uint16_t expert) { return {layer, expert}; }

// Cross-node spill config: each listed spill node is sized via absolute_gb to hold
// exactly `slots_per_node` slots (no /total clamp), for deterministic tests.
lc::CrossNodeSpillConfig spill_cfg(std::vector<std::pair<int, int>> nodes_w,
                                   size_t slots_per_node) {
    lc::CrossNodeSpillConfig s;
    s.enabled = true;
    for (auto [n, w] : nodes_w) {
        lc::NodesConfig nc;
        nc.node = n;
        nc.weight = w;
        s.nodes.push_back(nc);
    }
    s.sizing_mode = lc::HostPoolSizingMode::absolute_gb;
    s.sizing_value = static_cast<double>(slots_per_node * kSlot) / 1073741824.0;
    return s;
}

}  // namespace

// ── Arena allocation + one registration ──────────────────────────────────────

TEST(PinnedExpertArenaTest, BuildsOneArenaPerGpuAttachedNode) {
    REQUIRES_GPU();
    lm::NumaManager numa(dev_box_hw());
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    // 4 slots per node → total = 3 nodes * 4 * kSlot.
    const size_t total = 3 * 4 * kSlot;
    lm::PinnedExpertArena arena(numa, kSlot, total, big_sizing());

    EXPECT_EQ(arena.num_arenas(), 3u);  // nodes 0, 2, 3
    EXPECT_EQ(arena.total_slots(), 12u);
    EXPECT_GE(arena.total_pinned_bytes(), total);

    // Shared node (2) records share_degree=2; distinct nodes are 1.
    ASSERT_NE(arena.node_arena(2), nullptr);
    EXPECT_EQ(arena.node_arena(2)->share_degree(), 2);
    ASSERT_NE(arena.node_arena(0), nullptr);
    EXPECT_EQ(arena.node_arena(0)->share_degree(), 1);
    ASSERT_NE(arena.node_arena(3), nullptr);
    EXPECT_EQ(arena.node_arena(3)->share_degree(), 1);
    EXPECT_EQ(arena.node_arena(1), nullptr);  // node 1 is overflow-only, no GPU
}

// ── Registration throughput (P-24 acceptance: tens of ms/GB, not s/GB) ───────

TEST(PinnedExpertArenaTest, RegistrationIsFast) {
    REQUIRES_GPU();
    lm::NumaManager numa(single_node_hw());
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    // ~1 GiB single arena so the per-GB figure is meaningful.
    const size_t gib = 1ull << 30;
    const size_t slot = 16ull << 20;       // 16 MiB slots
    const size_t total = gib;              // one node → ~1 GiB

    auto t0 = std::chrono::steady_clock::now();
    lm::PinnedExpertArena arena(numa, slot, total, big_sizing());
    auto t1 = std::chrono::steady_clock::now();

    double gb = arena.total_pinned_bytes() / 1073741824.0;
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_per_gb = gb > 0 ? ms / gb : 0.0;
    // Anonymous arena registration is ~37 ms/GB (INV-4.12g); the old per-file
    // COW path was ~5700 ms/GB (TD-100c). Assert well under 1 s/GB — orders of
    // magnitude faster than the file-pin path, with generous slack for the box.
    EXPECT_GT(gb, 0.0);
    EXPECT_LT(ms_per_gb, 1000.0)
        << "arena register " << ms_per_gb << " ms/GB (expected ~tens of ms/GB)";
}

// ── Slab reserve / resolve / ready ───────────────────────────────────────────

TEST(PinnedExpertArenaTest, ReserveResolveReady) {
    REQUIRES_GPU();
    lm::NumaManager numa(single_node_hw());
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    lm::PinnedExpertArena arena(numa, kSlot, 4 * kSlot, big_sizing());
    auto* a = arena.node_arena(0);
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a->num_slots(), 4u);

    auto k = key(3, 7);
    EXPECT_EQ(arena.resolve(k), nullptr);   // not resident
    EXPECT_FALSE(arena.is_ready(k));

    void* slot = arena.reserve(k);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(arena.resolve(k), nullptr);   // reserved but not ready yet
    arena.mark_ready(k);
    EXPECT_EQ(arena.resolve(k), slot);      // now resolvable
    EXPECT_TRUE(arena.is_ready(k));
    EXPECT_EQ(a->occupied(), 1u);

    // Re-reserving the same key returns the same slot (idempotent).
    EXPECT_EQ(arena.reserve(k), slot);
    EXPECT_EQ(a->occupied(), 1u);
}

// ── LRU eviction when slab is full ───────────────────────────────────────────

TEST(PinnedExpertArenaTest, LruEvictsLeastRecentlyUsed) {
    REQUIRES_GPU();
    lm::NumaManager numa(single_node_hw());
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    // 2 slots → forces eviction on the 3rd distinct expert.
    lm::PinnedExpertArena arena(numa, kSlot, 2 * kSlot, big_sizing());
    auto* a = arena.node_arena(0);
    ASSERT_NE(a, nullptr);
    ASSERT_EQ(a->num_slots(), 2u);

    auto k0 = key(3, 0), k1 = key(3, 1), k2 = key(3, 2);
    arena.reserve(k0); arena.mark_ready(k0);
    arena.reserve(k1); arena.mark_ready(k1);
    EXPECT_EQ(a->occupied(), 2u);

    // Touch k0 so k1 is now the LRU victim.
    a->touch(k0);
    void* s2 = arena.reserve(k2);  // must evict k1 (LRU), not k0
    arena.mark_ready(k2);
    ASSERT_NE(s2, nullptr);

    EXPECT_TRUE(a->resident(k0));
    EXPECT_FALSE(a->resident(k1));  // evicted
    EXPECT_TRUE(a->resident(k2));
    EXPECT_EQ(a->occupied(), 2u);
}

// ── In-flight refcount blocks reuse (multi-GPU safety) ───────────────────────

TEST(PinnedExpertArenaTest, InflightRefcountBlocksEviction) {
    REQUIRES_GPU();
    lm::NumaManager numa(single_node_hw());
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    lm::PinnedExpertArena arena(numa, kSlot, 2 * kSlot, big_sizing());
    auto* a = arena.node_arena(0);
    ASSERT_NE(a, nullptr);
    ASSERT_EQ(a->num_slots(), 2u);

    auto k0 = key(3, 0), k1 = key(3, 1), k2 = key(3, 2);
    arena.reserve(k0); arena.mark_ready(k0);
    arena.reserve(k1); arena.mark_ready(k1);

    // Two GPUs both copying k0 (shared-node dedup): refcount 2.
    arena.acquire_inflight(k0);
    arena.acquire_inflight(k0);
    EXPECT_EQ(a->inflight(k0), 2);

    // k1 not in flight; k0 is. Reserving k2 must evict k1 (the only evictable
    // slot), never the in-flight k0 — even though k0 may be older.
    a->touch(k1);            // make k0 the LRU by tick, yet k0 is pinned
    void* s2 = arena.reserve(k2);
    ASSERT_NE(s2, nullptr);
    arena.mark_ready(k2);
    EXPECT_TRUE(a->resident(k0));   // protected by in-flight refcount
    EXPECT_FALSE(a->resident(k1));  // evicted instead
    EXPECT_TRUE(a->resident(k2));

    // Release one GPU's copy → still pinned (other GPU mid-copy).
    arena.release_inflight(k0);
    EXPECT_EQ(a->inflight(k0), 1);

    // With both k0 and k2 resident and k0 still in flight, reserving a 3rd key
    // can only evict k2; k0 stays.
    auto k3 = key(3, 3);
    void* s3 = arena.reserve(k3);
    ASSERT_NE(s3, nullptr);
    EXPECT_TRUE(a->resident(k0));
    EXPECT_FALSE(a->resident(k2));

    // Release the last copy → k0 now evictable.
    arena.release_inflight(k0);
    EXPECT_EQ(a->inflight(k0), 0);
    auto k4 = key(3, 4);
    arena.reserve(k4);
    EXPECT_FALSE(a->resident(k0));  // finally evictable
}

// ── J-1: async-load (kLoading) state ─────────────────────────────────────────

// A reserved slot whose async file→slot read is in flight (mark_loading) must
// not be evicted by another expert's reserve(), and resolve() must return null
// until mark_ready. mark_ready clears loading.
TEST(PinnedExpertArenaTest, LoadingSlotNotEvictedAndNotReady) {
    REQUIRES_GPU();
    lm::NumaManager numa(single_node_hw());
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    lm::PinnedExpertArena arena(numa, kSlot, 2 * kSlot, big_sizing());
    auto* a = arena.node_arena(0);
    ASSERT_NE(a, nullptr);
    ASSERT_EQ(a->num_slots(), 2u);

    auto k0 = key(5, 0), k1 = key(5, 1), k2 = key(5, 2);

    // k0: reserved + loading (async read in flight). Not ready, not resolvable.
    void* s0 = arena.reserve(k0);
    ASSERT_NE(s0, nullptr);
    a->mark_loading(k0);
    EXPECT_TRUE(a->is_loading(k0));
    EXPECT_FALSE(arena.is_ready(k0));
    EXPECT_EQ(arena.resolve(k0), nullptr);   // loading → not a source yet

    // k1: a plain ready slot, made the LRU victim.
    arena.reserve(k1); arena.mark_ready(k1);
    a->touch(k1);  // k1 newest by tick — but k0 is loading, so k0 is unevictable
    a->touch(k0);  // (no-op for loading protection; just exercise touch)

    // Reserving k2 must evict k1 (ready, evictable), NOT the loading k0.
    void* s2 = arena.reserve(k2);
    ASSERT_NE(s2, nullptr);
    arena.mark_ready(k2);
    EXPECT_TRUE(a->resident(k0));   // protected while loading
    EXPECT_FALSE(a->resident(k1));  // evicted instead
    EXPECT_TRUE(a->resident(k2));

    // Load completes: mark_ready clears loading and makes it resolvable.
    arena.mark_ready(k0);
    EXPECT_FALSE(a->is_loading(k0));
    EXPECT_TRUE(arena.is_ready(k0));
    EXPECT_EQ(arena.resolve(k0), s0);
}

// A failed async load → clear_loading: the slot stays NOT ready and becomes
// evictable again.
TEST(PinnedExpertArenaTest, ClearLoadingLeavesSlotUnreadyAndEvictable) {
    REQUIRES_GPU();
    lm::NumaManager numa(single_node_hw());
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    lm::PinnedExpertArena arena(numa, kSlot, 1 * kSlot, big_sizing());  // 1 slot
    auto* a = arena.node_arena(0);
    ASSERT_NE(a, nullptr);
    ASSERT_EQ(a->num_slots(), 1u);

    auto k0 = key(5, 0), k1 = key(5, 1);
    ASSERT_NE(arena.reserve(k0), nullptr);
    a->mark_loading(k0);
    // While loading, the only slot is unevictable → reserve(k1) fails.
    EXPECT_EQ(arena.reserve(k1), nullptr);

    // Load failed: clear_loading. Slot is still not ready, but now evictable.
    arena.clear_loading(k0);
    EXPECT_FALSE(a->is_loading(k0));
    EXPECT_FALSE(arena.is_ready(k0));
    void* s1 = arena.reserve(k1);  // can now reuse the slot
    EXPECT_NE(s1, nullptr);
    EXPECT_FALSE(a->resident(k0));
}

// ── Arena full of in-flight slots → reserve returns nullptr (fall through) ────

TEST(PinnedExpertArenaTest, AllInflightReserveFails) {
    REQUIRES_GPU();
    lm::NumaManager numa(single_node_hw());
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    lm::PinnedExpertArena arena(numa, kSlot, 2 * kSlot, big_sizing());
    auto* a = arena.node_arena(0);
    ASSERT_NE(a, nullptr);

    auto k0 = key(3, 0), k1 = key(3, 1), k2 = key(3, 2);
    arena.reserve(k0); arena.mark_ready(k0); arena.acquire_inflight(k0);
    arena.reserve(k1); arena.mark_ready(k1); arena.acquire_inflight(k1);

    // Both slots pinned in-flight → no evictable slot.
    EXPECT_EQ(arena.reserve(k2), nullptr);

    arena.release_inflight(k0);
    arena.release_inflight(k1);
}

// ── move_pages: slots land on the expected NUMA node ─────────────────────────

#ifdef LAYERSTORM_HAS_NUMA
TEST(PinnedExpertArenaTest, SlotsAreOnExpectedNode) {
    REQUIRES_GPU();
    lm::NumaManager numa(dev_box_hw());
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    lm::PinnedExpertArena arena(numa, kSlot, 3 * 4 * kSlot, big_sizing());

    // Reserve one expert homed on each GPU-attached node and fault its first
    // page in (write through the returned slot pointer), then query move_pages.
    for (int node : {0, 2, 3}) {
        // expert_home_node round-robins over sorted nodes {0,2,3}; pick an
        // expert index that maps to `node`.
        std::vector<int> nodes = {0, 2, 3};
        int pos = 0;
        for (size_t i = 0; i < nodes.size(); ++i)
            if (nodes[i] == node) pos = static_cast<int>(i);
        auto k = key(3, static_cast<uint16_t>(pos));
        ASSERT_EQ(numa.expert_home_node(k.expert_idx), node);

        void* slot = arena.reserve(k);
        ASSERT_NE(slot, nullptr);
        // Touch the page so it is backed by a physical frame on the bound node.
        static_cast<char*>(slot)[0] = 1;

        void* pages[1] = {slot};
        int status[1] = {-1};
        int rc = move_pages(0, 1, pages, nullptr, status, 0);
        ASSERT_EQ(rc, 0) << "move_pages failed";
        EXPECT_EQ(status[0], node)
            << "slot for expert homed on node " << node
            << " is physically on node " << status[0];
    }
}
#endif

// ── Cross-node RAM spill tier (Stage 1) ──────────────────────────────────────

TEST(PinnedExpertArenaTest, SpillArenaBuiltOnGpuLessNode) {
    REQUIRES_GPU();
    lm::NumaManager numa(dev_box_hw());           // GPUs on nodes 0,2,3; node 1 GPU-less
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    // Spill onto the GPU-less node 1 (4 slots). GPU nodes built as usual.
    lm::PinnedExpertArena arena(numa, kSlot, 3 * 4 * kSlot, big_sizing(),
                                spill_cfg({{1, 10}}, 4));
    ASSERT_NE(arena.node_arena(1), nullptr) << "spill arena on GPU-less node 1";
    EXPECT_EQ(arena.node_arena(1)->num_slots(), 4u);
    EXPECT_EQ(arena.node_arena(1)->share_degree(), 1);
    // GPU nodes still present.
    EXPECT_NE(arena.node_arena(0), nullptr);
    EXPECT_NE(arena.node_arena(2), nullptr);
    EXPECT_NE(arena.node_arena(3), nullptr);
}

TEST(PinnedExpertArenaTest, ExtendToSpillBeforeEvictingLocal) {
    REQUIRES_GPU();
    lm::NumaManager numa(single_node_hw());       // GPUs 0,1 on node 0
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    // node 0 (local) = 2 slots; spill node 1 = 2 slots.
    lm::PinnedExpertArena arena(numa, kSlot, 2 * kSlot, big_sizing(),
                                spill_cfg({{1, 5}}, 2));
    auto k0 = key(3, 0), k1 = key(3, 1), k2 = key(3, 2);
    int n0 = -1, n1 = -1, n2 = -1;
    EXPECT_NE(arena.reserve_for_fill(k0, /*gpu=*/0, &n0), nullptr);
    EXPECT_NE(arena.reserve_for_fill(k1, 0, &n1), nullptr);
    EXPECT_NE(arena.reserve_for_fill(k2, 0, &n2), nullptr);  // node 0 full → spill

    EXPECT_EQ(n0, 0);
    EXPECT_EQ(n1, 0);
    EXPECT_EQ(n2, 1) << "3rd expert spills to node 1 instead of evicting k0/k1";
    EXPECT_EQ(arena.location_node(k0), 0);   // not evicted
    EXPECT_EQ(arena.location_node(k1), 0);
    EXPECT_EQ(arena.node_arena(0)->occupied(), 2u);
    EXPECT_EQ(arena.node_arena(1)->occupied(), 1u);
}

TEST(PinnedExpertArenaTest, WeightedSpillPrefersHigherWeightUntilFull) {
    REQUIRES_GPU();
    lm::NumaManager numa(single_node_hw());
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    // node 0 = 1 slot; spill nodes: node 1 (weight 10, 2 slots), node 2 (weight 5, 2 slots).
    lm::PinnedExpertArena arena(numa, kSlot, 1 * kSlot, big_sizing(),
                                spill_cfg({{1, 10}, {2, 5}}, 2));
    int n = -1;
    arena.reserve_for_fill(key(3, 0), 0, &n);  // node 0
    EXPECT_EQ(n, 0);
    // Next experts overflow node 0 → must fill node 1 (higher weight) fully first.
    arena.reserve_for_fill(key(3, 1), 0, &n); EXPECT_EQ(n, 1);
    arena.reserve_for_fill(key(3, 2), 0, &n); EXPECT_EQ(n, 1);
    // node 1 now full (2/2) → next spills to node 2.
    arena.reserve_for_fill(key(3, 3), 0, &n); EXPECT_EQ(n, 2)
        << "node 1 full → lower-weight node 2";
}

TEST(PinnedExpertArenaTest, EqualWeightSpillInterleaves) {
    REQUIRES_GPU();
    lm::NumaManager numa(single_node_hw());
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    // node 0 = 1 slot; equal-weight spill nodes 1 and 2 (4 slots each).
    lm::PinnedExpertArena arena(numa, kSlot, 1 * kSlot, big_sizing(),
                                spill_cfg({{1, 5}, {2, 5}}, 4));
    int n = -1;
    arena.reserve_for_fill(key(3, 0), 0, &n);  // node 0
    int na = -1, nb = -1;
    arena.reserve_for_fill(key(3, 1), 0, &na);  // first spill
    arena.reserve_for_fill(key(3, 2), 0, &nb);  // second spill
    EXPECT_TRUE((na == 1 && nb == 2) || (na == 2 && nb == 1))
        << "equal weights interleave by fewest-occupied (na=" << na
        << " nb=" << nb << ")";
    EXPECT_NE(na, nb);
}

TEST(PinnedExpertArenaTest, CrossNodeResolveAndLocationEraseOnEvict) {
    REQUIRES_GPU();
    lm::NumaManager numa(single_node_hw());
    if (!numa.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    // node 0 = 1 slot; spill node 1 = 1 slot.
    lm::PinnedExpertArena arena(numa, kSlot, 1 * kSlot, big_sizing(),
                                spill_cfg({{1, 5}}, 1));
    auto k0 = key(3, 0), k1 = key(3, 1), k2 = key(3, 2);
    int n = -1;
    void* s0 = arena.reserve_for_fill(k0, 0, &n); arena.mark_ready(k0);  // node 0
    ASSERT_EQ(n, 0);
    void* s1 = arena.reserve_for_fill(k1, 0, &n); arena.mark_ready(k1);  // spill node 1
    ASSERT_EQ(n, 1);

    // Cross-node resolve: facade finds k1 on the spill node.
    EXPECT_EQ(arena.resolve(k0), s0);
    EXPECT_EQ(arena.resolve(k1), s1);
    EXPECT_EQ(arena.location_node(k1), 1);

    // Both nodes full (1 slot each) → k2 evicts the local LRU (k0), erases its index.
    void* s2 = arena.reserve_for_fill(k2, 0, &n);
    ASSERT_NE(s2, nullptr);
    EXPECT_EQ(n, 0) << "evict local LRU (candidate order: local first)";
    EXPECT_EQ(arena.location_node(k0), -1) << "evicted key dropped from location index";
    EXPECT_EQ(arena.location_node(k2), 0);
    EXPECT_EQ(arena.location_node(k1), 1) << "spill resident untouched";
}
