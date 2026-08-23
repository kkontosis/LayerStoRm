// Unit tests for the persistent-arena wiring in PinnedExpertArena (P-24b):
// shared (memfd) backing, cross-"run" adopt of preserved slot bytes, the
// ArenaCache hooks on reserve/mark_loading/mark_ready/evict, warm-slot
// preload skipping (via is_ready), and the runtime try_adopt path.
//
// All tests run with defer_registration=true and never register — no CUDA
// needed (slots are plain host memory until finalize_registration). The test
// itself plays the arena holder: it dup()s the segment fds across arena
// destruction, exactly like the holder keeps them across engine runs.

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstring>
#include <memory>
#include <vector>

#include "core/memory/arena_cache.h"
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

lc::HardwareConfig single_node_hw() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 128;
    hw.gpus = {make_gpu(0, 0), make_gpu(1, 0)};
    return hw;
}

constexpr size_t kSlot = 64 * 1024;

lc::PinHostExpertPoolSizingConfig big_sizing() {
    lc::PinHostExpertPoolSizingConfig s;
    s.mode = lc::HostPoolSizingMode::fraction_total;
    s.value = 0.85;
    return s;
}

lm::ExpertKey key(uint32_t layer, uint16_t expert) { return {layer, expert}; }

std::vector<lm::ExpertFileIdentity> idents(uint16_t n) {
    std::vector<lm::ExpertFileIdentity> v(n);
    for (uint16_t e = 0; e < n; ++e) v[e] = {7000ULL + e, 4096ULL * (e + 1)};
    return v;
}

std::unique_ptr<lm::PinnedExpertArena> make_arena(lm::NumaManager& numa,
                                                  size_t slots,
                                                  lm::ArenaBacking* backing) {
    return std::make_unique<lm::PinnedExpertArena>(
        numa, kSlot, kSlot * slots, big_sizing(),
        lc::CrossNodeSpillConfig{}, /*extra_scratch_bytes=*/0,
        /*defer_registration=*/true, backing);
}

}  // namespace

TEST(PinnedArenaPersist, SharedCreateExposesBackingFds) {
    lm::NumaManager numa(single_node_hw());
    lm::ArenaBacking backing;
    backing.mode = lm::ArenaBackingMode::kSharedCreate;
    auto arena = make_arena(numa, 4, &backing);
    EXPECT_GE(arena->node_backing_fd(0), 0);
    EXPECT_GT(arena->node_backing_bytes(0), 0u);
    // Private backing (default) has no fd.
    auto priv = make_arena(numa, 4, nullptr);
    EXPECT_EQ(priv->node_backing_fd(0), -1);
}

TEST(PinnedArenaPersist, SlabBytesMatchesActualAllocation) {
    lm::NumaManager numa(single_node_hw());
    lm::ArenaBacking backing;
    backing.mode = lm::ArenaBackingMode::kSharedCreate;
    auto arena = make_arena(numa, 4, &backing);
    EXPECT_EQ(arena->node_backing_bytes(0),
              lm::PinnedExpertArena::slab_bytes(kSlot, 4, 0));
}

TEST(PinnedArenaPersist, CacheHooksFollowSlotLifecycle) {
    lm::NumaManager numa(single_node_hw());
    lm::ArenaBacking backing;
    backing.mode = lm::ArenaBackingMode::kSharedCreate;
    auto arena = make_arena(numa, /*slots=*/1, &backing);  // 1 slot → evictions

    std::vector<lm::ArenaCacheNodeGeom> geom{{0, 1}};
    auto seg = lm::ArenaMetaSegment::create(lm::ArenaCache::required_bytes(geom));
    lm::ArenaCache cache(seg.base(), seg.bytes());
    ASSERT_TRUE(cache.format(0xA, 0xB, geom));
    cache.set_file_identities(idents(8));
    arena->set_cache(&cache);

    // Fill A synchronously: reserve → write → mark_ready ⇒ record ACTIVE.
    const auto a = key(0, 1);
    void* slot = arena->reserve(a);
    ASSERT_NE(slot, nullptr);
    std::memset(slot, 0x11, kSlot);
    arena->mark_ready(a);
    EXPECT_TRUE(cache.lookup(a).has_value());

    // Reserve B evicts A (single slot): A's record must die BEFORE any write.
    const auto b = key(0, 2);
    void* slot_b = arena->reserve(b);
    ASSERT_EQ(slot_b, slot);  // same physical slot, reused
    EXPECT_FALSE(cache.lookup(a).has_value());
    EXPECT_FALSE(cache.lookup(b).has_value());  // not ready yet
    arena->mark_loading(b, /*gpu=*/0);
    EXPECT_FALSE(cache.lookup(b).has_value());  // LOADING ≠ adoptable
    arena->mark_ready(b, /*gpu=*/0);
    EXPECT_TRUE(cache.lookup(b).has_value());

    // A failed async load kills the record.
    const auto c = key(0, 3);
    ASSERT_NE(arena->reserve(c), nullptr);
    arena->mark_loading(c, 0);
    arena->clear_loading(c, 0);
    EXPECT_FALSE(cache.lookup(c).has_value());
}

TEST(PinnedArenaPersist, AdoptRoundtripPreservesBytesAcrossArenaLifetimes) {
    lm::NumaManager numa(single_node_hw());
    constexpr size_t kSlots = 4;
    std::vector<lm::ArenaCacheNodeGeom> geom{{0, kSlots}};
    auto seg = lm::ArenaMetaSegment::create(lm::ArenaCache::required_bytes(geom));
    auto cache = std::make_unique<lm::ArenaCache>(seg.base(), seg.bytes());
    ASSERT_TRUE(cache->format(0xC0FE, 0xBEEF, geom));
    cache->set_file_identities(idents(8));

    int keeper_fd = -1;
    size_t seg_bytes = 0;
    const auto k1 = key(3, 1);
    const auto k2 = key(9, 2);

    {  // ── "run 1": fill two slots, then exit (destroy the arena) ──
        lm::ArenaBacking backing;
        backing.mode = lm::ArenaBackingMode::kSharedCreate;
        auto arena = make_arena(numa, kSlots, &backing);
        arena->set_cache(cache.get());
        keeper_fd = ::dup(arena->node_backing_fd(0));  // the test IS the holder
        ASSERT_GE(keeper_fd, 0);
        seg_bytes = arena->node_backing_bytes(0);

        void* s1 = arena->reserve(k1);
        std::memset(s1, 0xAA, kSlot);
        arena->mark_ready(k1);
        void* s2 = arena->reserve(k2);
        std::memset(s2, 0xBB, kSlot);
        arena->mark_ready(k2);
    }  // arena destroyed: munmap + fd close — pages live via keeper_fd

    // ── "run 2": adopt the kept segment, validate meta, adopt slots ──
    lm::ArenaCache cache2(seg.base(), seg.bytes());
    ASSERT_TRUE(cache2.validate(0xC0FE, 0xBEEF, geom));
    cache2.set_file_identities(idents(8));

    lm::ArenaBacking backing2;
    backing2.mode = lm::ArenaBackingMode::kAdopt;
    backing2.adopted[0] = numa.adopt_shared(keeper_fd, seg_bytes, 0);
    backing2.adopt_plans = {{0, false, kSlots, 1}};  // stored geometry
    auto arena2 = make_arena(numa, kSlots, &backing2);
    arena2->set_cache(&cache2);

    size_t adopted = cache2.scan_adoptable(
        /*num_layers=*/16, /*num_experts=*/8,
        [&](int node, size_t slot, lm::ExpertKey k) {
            EXPECT_TRUE(arena2->adopt_ready(k, node, slot));
        });
    EXPECT_EQ(adopted, 2u);

    // Both keys resolve ready with their bytes intact — zero reloads.
    ASSERT_TRUE(arena2->is_ready(k1));
    ASSERT_TRUE(arena2->is_ready(k2));
    EXPECT_EQ(static_cast<unsigned char*>(arena2->resolve(k1))[0], 0xAA);
    EXPECT_EQ(static_cast<unsigned char*>(arena2->resolve(k1))[kSlot - 1], 0xAA);
    EXPECT_EQ(static_cast<unsigned char*>(arena2->resolve(k2))[0], 0xBB);

    // Adopted slots participate in the normal lifecycle: evicting one for a
    // new key kills its record (INV-ARENA-CACHE-ORDER via reserve).
    const auto k3 = key(1, 3);
    ASSERT_NE(arena2->reserve(k3), nullptr);  // takes a free slot, no evict
    EXPECT_TRUE(cache2.lookup(k1).has_value());
}

TEST(PinnedArenaPersist, AdoptStoredGeometryOverridesPlan) {
    // Free-RAM drift: today's plan would say 6 slots, but the stored arena has
    // 4 — adoption must use the STORED geometry (it always fits; recomputed
    // budgets are volatile: the fraction_free HBM ±1-slot wipe bug).
    lm::NumaManager numa(single_node_hw());
    int keeper_fd = -1;
    size_t seg_bytes = 0;
    const auto k = key(1, 1);
    {
        lm::ArenaBacking backing;
        backing.mode = lm::ArenaBackingMode::kSharedCreate;
        auto arena = make_arena(numa, /*slots=*/4, &backing);
        keeper_fd = ::dup(arena->node_backing_fd(0));
        seg_bytes = arena->node_backing_bytes(0);
        std::memset(arena->reserve(k), 0xEE, kSlot);
        arena->mark_ready(k);
    }
    lm::ArenaBacking backing2;
    backing2.mode = lm::ArenaBackingMode::kAdopt;
    backing2.adopted[0] = numa.adopt_shared(keeper_fd, seg_bytes, 0);
    backing2.adopt_plans = {{0, false, 4, 1}};      // stored: 4 slots
    auto arena2 = make_arena(numa, /*slots=*/6, &backing2);  // plan says 6
    EXPECT_EQ(arena2->total_slots(), 4u);           // stored geometry wins
    ASSERT_TRUE(arena2->adopt_ready(k, 0, 0));
    EXPECT_EQ(static_cast<unsigned char*>(arena2->resolve(k))[0], 0xEE);
}

TEST(PinnedArenaPersist, TryAdoptRuntimePath) {
    lm::NumaManager numa(single_node_hw());
    constexpr size_t kSlots = 3;
    std::vector<lm::ArenaCacheNodeGeom> geom{{0, kSlots}};
    auto seg = lm::ArenaMetaSegment::create(lm::ArenaCache::required_bytes(geom));
    lm::ArenaCache cache(seg.base(), seg.bytes());
    ASSERT_TRUE(cache.format(1, 2, geom));
    cache.set_file_identities(idents(8));

    int keeper_fd = -1;
    size_t seg_bytes = 0;
    const auto k = key(2, 4);
    {
        lm::ArenaBacking backing;
        backing.mode = lm::ArenaBackingMode::kSharedCreate;
        auto arena = make_arena(numa, kSlots, &backing);
        arena->set_cache(&cache);
        keeper_fd = ::dup(arena->node_backing_fd(0));
        seg_bytes = arena->node_backing_bytes(0);
        void* s = arena->reserve(k);
        std::memset(s, 0xCD, kSlot);
        arena->mark_ready(k);
    }

    lm::ArenaCache cache2(seg.base(), seg.bytes());
    ASSERT_TRUE(cache2.validate(1, 2, geom));
    cache2.set_file_identities(idents(8));
    // Index the records WITHOUT adopting into the arena (empty scan fn) — the
    // runtime try_adopt must then find the free-slot record on demand.
    ASSERT_EQ(cache2.scan_adoptable(16, 8, nullptr), 1u);

    lm::ArenaBacking backing2;
    backing2.mode = lm::ArenaBackingMode::kAdopt;
    backing2.adopted[0] = numa.adopt_shared(keeper_fd, seg_bytes, 0);
    backing2.adopt_plans = {{0, false, kSlots, 1}};  // stored geometry
    auto arena2 = make_arena(numa, kSlots, &backing2);
    arena2->set_cache(&cache2);

    EXPECT_FALSE(arena2->is_ready(k));
    void* p = arena2->try_adopt(k);
    ASSERT_NE(p, nullptr);
    EXPECT_TRUE(arena2->is_ready(k));
    EXPECT_EQ(static_cast<unsigned char*>(p)[0], 0xCD);
    // Second try_adopt for the same key: already resident → slot occupied.
    EXPECT_EQ(arena2->try_adopt(k), nullptr);
    // Unknown key: no record → no adopt.
    EXPECT_EQ(arena2->try_adopt(key(5, 5)), nullptr);
}
