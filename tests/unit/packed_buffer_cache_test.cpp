// Unit tests for PackedBufferCache (WP-4).

#include <gtest/gtest.h>

#include "model/weight_pipeline/packed_buffer_cache.h"
#include "model/weight_loader/weight_loader.h"
#include "model/weight_loader/weight_handler.h"

using namespace layerstorm::model;
using layerstorm::memory::ExpertKey;

// ── Helpers ────────────────────────────────────────────────────────────────

static ExpertKey mk(uint32_t layer, uint16_t expert) {
    return ExpertKey{layer, expert};
}

static std::shared_ptr<std::vector<std::byte>> make_buf(size_t size) {
    return std::make_shared<std::vector<std::byte>>(size, std::byte{0x42});
}

// ── Test: explicit mode, LRU eviction ──────────────────────────────────────

TEST(PackedBufferCacheTest, ExplicitBudgetLruEviction) {
    // Budget = 100 MB, slot = 24 MB each.  5 experts = 120 MB > 100 MB.
    constexpr int64_t MB = 1024 * 1024;
    PackedBufferCache cache(PackedBufferCache::Options{
        .mode = PackedBufferCache::Mode::kExplicit,
        .budget_bytes = 100 * MB,
        .slot_size_bytes = 24 * MB,
    });

    // Insert 5 experts (A through E, oldest first).
    for (int i = 0; i < 5; ++i) {
        cache.insert(mk(3, static_cast<uint16_t>(i)), make_buf(24 * MB));
    }

    // Budget exceeded (120 > 100) — at least one must have been evicted.
    EXPECT_LE(cache.used_bytes(), 100 * MB);
    // 4 experts fit in 96 MB ≤ 100 MB.
    EXPECT_EQ(cache.entry_count(), 4u);

    // Oldest (expert 0) should be evicted.
    EXPECT_EQ(cache.lookup(mk(3, 0)), nullptr);
    // Most recent 4 should still be present.
    for (int i = 1; i < 5; ++i) {
        EXPECT_NE(cache.lookup(mk(3, static_cast<uint16_t>(i))), nullptr);
    }
}

// ── Test: budget=0 (passthrough, matches TD-82a) ──────────────────────────

TEST(PackedBufferCacheTest, BudgetZeroPassthrough) {
    PackedBufferCache cache(PackedBufferCache::Options{
        .mode = PackedBufferCache::Mode::kExplicit,
        .budget_bytes = 0,
    });

    EXPECT_TRUE(cache.is_passthrough());

    auto buf = make_buf(1024);
    cache.insert(mk(3, 0), buf);
    // Budget is 0, so insert immediately triggers eviction.
    // But the caller still holds a shared_ptr (use_count > 1), so eviction
    // can't remove it.  Verify it's still accessible:
    auto found = cache.lookup(mk(3, 0));
    EXPECT_NE(found, nullptr);

    // Release the external reference — now the cache can evict on release().
    buf.reset();
    found.reset();
    cache.release(mk(3, 0));
    EXPECT_EQ(cache.lookup(mk(3, 0)), nullptr);
    EXPECT_EQ(cache.entry_count(), 0u);
}

// ── Test: budget=-1 (unlimited, never evicts) ─────────────────────────────

TEST(PackedBufferCacheTest, BudgetUnlimitedNeverEvicts) {
    PackedBufferCache cache(PackedBufferCache::Options{
        .mode = PackedBufferCache::Mode::kExplicit,
        .budget_bytes = -1,
    });

    EXPECT_FALSE(cache.is_passthrough());

    for (int i = 0; i < 50; ++i) {
        cache.insert(mk(3, static_cast<uint16_t>(i)), make_buf(1024));
    }

    EXPECT_EQ(cache.entry_count(), 50u);
    for (int i = 0; i < 50; ++i) {
        EXPECT_NE(cache.lookup(mk(3, static_cast<uint16_t>(i))), nullptr);
    }
}

// ── Test: mmap mode is a no-op ────────────────────────────────────────────

TEST(PackedBufferCacheTest, MmapModeNoop) {
    PackedBufferCache cache(PackedBufferCache::Options{
        .mode = PackedBufferCache::Mode::kMmap,
    });

    EXPECT_TRUE(cache.is_mmap());
    EXPECT_FALSE(cache.is_passthrough());

    // Lookup always returns nullptr.
    EXPECT_EQ(cache.lookup(mk(3, 0)), nullptr);

    // Insert is a no-op.
    cache.insert(mk(3, 0), make_buf(1024));
    EXPECT_EQ(cache.entry_count(), 0u);
    EXPECT_EQ(cache.lookup(mk(3, 0)), nullptr);
}

// ── Test: pin prevents LRU eviction during DMA ───────────────────────────

TEST(PackedBufferCacheTest, PinPreventsEviction) {
    // Budget fits exactly 2 slots.
    constexpr int64_t SLOT = 1024;
    PackedBufferCache cache(PackedBufferCache::Options{
        .mode = PackedBufferCache::Mode::kExplicit,
        .budget_bytes = 2 * SLOT,
        .slot_size_bytes = SLOT,
    });

    // Insert A and hold a reference (simulating in-flight DMA).
    cache.insert(mk(3, 0), make_buf(SLOT));
    auto pin_a = cache.lookup(mk(3, 0));  // use_count = 2 (cache + pin_a)
    ASSERT_NE(pin_a, nullptr);

    // Insert B — fits within budget (2 slots total).
    cache.insert(mk(3, 1), make_buf(SLOT));
    EXPECT_EQ(cache.entry_count(), 2u);

    // Insert C — over budget (3 > 2).  Must evict one.
    // A is pinned (use_count > 1), so B gets evicted instead.
    cache.insert(mk(3, 2), make_buf(SLOT));

    EXPECT_NE(cache.lookup(mk(3, 0)), nullptr);  // A survived (pinned)
    EXPECT_EQ(cache.lookup(mk(3, 1)), nullptr);   // B evicted
    EXPECT_NE(cache.lookup(mk(3, 2)), nullptr);   // C is new
    EXPECT_EQ(cache.entry_count(), 2u);
}

// ── Test: insert deduplicates ─────────────────────────────────────────────

TEST(PackedBufferCacheTest, InsertDeduplicates) {
    PackedBufferCache cache(PackedBufferCache::Options{
        .mode = PackedBufferCache::Mode::kExplicit,
        .budget_bytes = -1,
    });

    cache.insert(mk(3, 0), make_buf(1000));
    EXPECT_EQ(cache.used_bytes(), 1000);

    // Re-insert same key with larger buffer.
    cache.insert(mk(3, 0), make_buf(2000));
    EXPECT_EQ(cache.entry_count(), 1u);
    EXPECT_EQ(cache.used_bytes(), 2000);
}

// ── Test: release removes entry ───────────────────────────────────────────

TEST(PackedBufferCacheTest, ReleaseRemovesEntry) {
    PackedBufferCache cache(PackedBufferCache::Options{
        .mode = PackedBufferCache::Mode::kExplicit,
        .budget_bytes = -1,
    });

    cache.insert(mk(3, 0), make_buf(1024));
    EXPECT_EQ(cache.entry_count(), 1u);
    EXPECT_EQ(cache.used_bytes(), 1024);

    cache.release(mk(3, 0));
    EXPECT_EQ(cache.entry_count(), 0u);
    EXPECT_EQ(cache.used_bytes(), 0);

    // Release of non-existent key is a no-op.
    cache.release(mk(3, 99));
    EXPECT_EQ(cache.entry_count(), 0u);
}

// ── Test: multiple layers and experts ─────────────────────────────────────

TEST(PackedBufferCacheTest, MultiLayerMultiExpert) {
    PackedBufferCache cache(PackedBufferCache::Options{
        .mode = PackedBufferCache::Mode::kExplicit,
        .budget_bytes = -1,
    });

    // Insert experts across multiple layers.
    for (uint32_t l = 3; l <= 5; ++l) {
        for (uint16_t e = 0; e < 4; ++e) {
            cache.insert(mk(l, e), make_buf(100));
        }
    }

    EXPECT_EQ(cache.entry_count(), 12u);
    EXPECT_EQ(cache.used_bytes(), 1200);

    // Verify each is independently accessible.
    EXPECT_NE(cache.lookup(mk(3, 0)), nullptr);
    EXPECT_NE(cache.lookup(mk(5, 3)), nullptr);
    EXPECT_EQ(cache.lookup(mk(6, 0)), nullptr);  // not inserted
}

// ── Test: preload_explicit with bounded budget (TD-95a) ──────────────────

TEST(PackedBufferCacheTest, PreloadExplicitStopsAtBudget) {
    // Build a fake LoadedModel with 2 MoE layers (idx 3,4), 4 experts each.
    // Each expert has a pre-set owned_buf of 1000 bytes.
    LoadedModel model;
    model.layers.resize(5);  // layers 0-4
    for (int l = 3; l <= 4; ++l) {
        model.layers[l].layer_idx = l;
        model.layers[l].routed_experts.resize(4);
        for (int e = 0; e < 4; ++e) {
            auto buf = make_buf(1000);
            WeightBundle wb;
            wb.owned_buf = buf;
            wb.packed_slot = std::span<const std::byte>(buf->data(), buf->size());
            model.layers[l].routed_experts[e].push_back(std::move(wb));
        }
    }

    // Budget fits 5 experts (5000 bytes) out of 8 total.
    PackedBufferCache cache(PackedBufferCache::Options{
        .mode = PackedBufferCache::Mode::kExplicit,
        .budget_bytes = 5000,
        .slot_size_bytes = 1000,
        .expert_shape = {7168, 2048},
    });

    int n = cache.preload_explicit(model, /*first_moe_layer=*/3,
                                   /*num_moe_layers=*/2, /*num_experts=*/4);
    EXPECT_EQ(n, 5);  // stopped after 5 (budget exhausted before 6th)
    EXPECT_EQ(cache.entry_count(), 5u);
    EXPECT_LE(cache.used_bytes(), 5000);
}

// ── Test: preload_explicit with budget=0 returns immediately ─────────────

TEST(PackedBufferCacheTest, PreloadExplicitPassthroughNoop) {
    LoadedModel model;
    model.layers.resize(5);
    model.layers[3].layer_idx = 3;
    model.layers[3].routed_experts.resize(2);
    for (int e = 0; e < 2; ++e) {
        auto buf = make_buf(1000);
        WeightBundle wb;
        wb.owned_buf = buf;
        wb.packed_slot = std::span<const std::byte>(buf->data(), buf->size());
        model.layers[3].routed_experts[e].push_back(std::move(wb));
    }

    PackedBufferCache cache(PackedBufferCache::Options{
        .mode = PackedBufferCache::Mode::kExplicit,
        .budget_bytes = 0,
        .slot_size_bytes = 1000,
    });

    int n = cache.preload_explicit(model, 3, 1, 2);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(cache.entry_count(), 0u);
}

// ── Test: all entries pinned — budget exceeded gracefully (TD-95c) ───────

TEST(PackedBufferCacheTest, AllPinnedExceedsBudgetGracefully) {
    constexpr int64_t SLOT = 1024;
    PackedBufferCache cache(PackedBufferCache::Options{
        .mode = PackedBufferCache::Mode::kExplicit,
        .budget_bytes = SLOT,  // fits 1
        .slot_size_bytes = SLOT,
    });

    // Insert A, pin it.  Hold an external ref (simulating WeightBundle.owned_buf)
    // so that use_count > 1 at insert time — mirrors the real resolve_host_source
    // path where the bundle retains a reference alongside the cache.
    auto buf_a = make_buf(SLOT);
    cache.insert(mk(3, 0), buf_a);  // copy: cache + buf_a → use_count=2
    auto pin_a = cache.lookup(mk(3, 0));
    ASSERT_NE(pin_a, nullptr);

    // Insert B with external ref — over budget, but A is pinned.
    auto buf_b = make_buf(SLOT);
    cache.insert(mk(3, 1), buf_b);  // copy: cache + buf_b → use_count=2
    auto pin_b = cache.lookup(mk(3, 1));
    ASSERT_NE(pin_b, nullptr);

    // Both survive — budget exceeded but no crash, no data loss.
    EXPECT_EQ(cache.entry_count(), 2u);
    EXPECT_EQ(cache.used_bytes(), 2 * SLOT);
    EXPECT_TRUE(cache.is_over_budget());

    // Release all external refs — next insert should recover.
    pin_a.reset();
    pin_b.reset();
    buf_a.reset();
    buf_b.reset();
    cache.insert(mk(3, 2), make_buf(SLOT));
    // Both old entries evictable now; evict until within budget.
    EXPECT_LE(cache.used_bytes(), SLOT);
    EXPECT_FALSE(cache.is_over_budget());
}
