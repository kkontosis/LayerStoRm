// Unit tests for BufferRegistry (IPC-6).
//
// Pure data structure tests — no CUDA, no IPC rings.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

#include "daemon/buffer_registry.h"

namespace ldam = layerstorm::daemon;

// ── Construction ────────────────────────────────────────────────────────────

TEST(BufferRegistry, DefaultConstructionEmpty) {
    ldam::BufferRegistry reg;
    EXPECT_TRUE(reg.empty());
    EXPECT_EQ(reg.size(), 0u);
}

TEST(BufferRegistry, InvalidBufIdIsZero) {
    EXPECT_EQ(ldam::kInvalidBufId, 0u);
}

// ── Registration ────────────────────────────────────────────────────────────

TEST(BufferRegistry, RegisterSingleBuffer) {
    ldam::BufferRegistry reg;
    int dummy = 42;
    uint32_t id = reg.register_buffer(&dummy, 1024, 0, "test_buf");

    EXPECT_NE(id, ldam::kInvalidBufId);
    EXPECT_EQ(reg.size(), 1u);
    EXPECT_FALSE(reg.empty());

    const auto* entry = reg.lookup(id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->device_ptr, &dummy);
    EXPECT_EQ(entry->size_bytes, 1024);
    EXPECT_EQ(entry->gpu_idx, 0);
}

TEST(BufferRegistry, RegisterMultipleBuffers) {
    ldam::BufferRegistry reg;
    int a = 1, b = 2, c = 3;
    uint32_t id_a = reg.register_buffer(&a, 100, 0, "a");
    uint32_t id_b = reg.register_buffer(&b, 200, 1, "b");
    uint32_t id_c = reg.register_buffer(&c, 300, 2, "c");

    EXPECT_EQ(reg.size(), 3u);

    // All IDs unique.
    std::set<uint32_t> ids{id_a, id_b, id_c};
    EXPECT_EQ(ids.size(), 3u);

    // Each resolves correctly.
    EXPECT_EQ(reg.resolve(id_a), &a);
    EXPECT_EQ(reg.resolve(id_b), &b);
    EXPECT_EQ(reg.resolve(id_c), &c);
}

TEST(BufferRegistry, RegisterReturnsMonotonicIds) {
    ldam::BufferRegistry reg;
    int d = 0;
    uint32_t id1 = reg.register_buffer(&d, 10, 0);
    uint32_t id2 = reg.register_buffer(&d, 20, 0);
    uint32_t id3 = reg.register_buffer(&d, 30, 0);
    EXPECT_EQ(id1, 1u);
    EXPECT_EQ(id2, 2u);
    EXPECT_EQ(id3, 3u);
}

TEST(BufferRegistry, RegisterNullPtrAllowed) {
    ldam::BufferRegistry reg;
    uint32_t id = reg.register_buffer(nullptr, 0, 0, "collapsed_region");
    EXPECT_NE(id, ldam::kInvalidBufId);

    const auto* entry = reg.lookup(id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->device_ptr, nullptr);
    EXPECT_EQ(entry->size_bytes, 0);
}

// ── Lookup ──────────────────────────────────────────────────────────────────

TEST(BufferRegistry, LookupMissingReturnsNull) {
    ldam::BufferRegistry reg;
    EXPECT_EQ(reg.lookup(999), nullptr);
}

TEST(BufferRegistry, LookupInvalidIdReturnsNull) {
    ldam::BufferRegistry reg;
    EXPECT_EQ(reg.lookup(ldam::kInvalidBufId), nullptr);
}

TEST(BufferRegistry, ResolveReturnsDevicePtr) {
    ldam::BufferRegistry reg;
    int d = 7;
    uint32_t id = reg.register_buffer(&d, 64, 0);
    EXPECT_EQ(reg.resolve(id), &d);
}

TEST(BufferRegistry, ResolveMissingReturnsNull) {
    ldam::BufferRegistry reg;
    EXPECT_EQ(reg.resolve(42), nullptr);
}

TEST(BufferRegistry, ResolveCheckedPass) {
    ldam::BufferRegistry reg;
    int d = 0;
    uint32_t id = reg.register_buffer(&d, 1024, 0);
    EXPECT_EQ(reg.resolve_checked(id, 1024), &d);
    EXPECT_EQ(reg.resolve_checked(id, 512), &d);
    EXPECT_EQ(reg.resolve_checked(id, 0), &d);
}

TEST(BufferRegistry, ResolveCheckedFailTooSmall) {
    ldam::BufferRegistry reg;
    int d = 0;
    uint32_t id = reg.register_buffer(&d, 100, 0);
    EXPECT_EQ(reg.resolve_checked(id, 101), nullptr);
}

TEST(BufferRegistry, ResolveCheckedMissing) {
    ldam::BufferRegistry reg;
    EXPECT_EQ(reg.resolve_checked(42, 0), nullptr);
}

// ── Deregistration ──────────────────────────────────────────────────────────

TEST(BufferRegistry, DeregisterRemovesEntry) {
    ldam::BufferRegistry reg;
    int d = 0;
    uint32_t id = reg.register_buffer(&d, 64, 0, "temp");
    EXPECT_TRUE(reg.contains(id));

    reg.deregister(id);
    EXPECT_FALSE(reg.contains(id));
    EXPECT_EQ(reg.lookup(id), nullptr);
    EXPECT_EQ(reg.resolve(id), nullptr);
    EXPECT_EQ(reg.size(), 0u);
}

TEST(BufferRegistry, DeregisterUnknownIsNoop) {
    ldam::BufferRegistry reg;
    // Should not crash.
    reg.deregister(999);
    reg.deregister(ldam::kInvalidBufId);
    EXPECT_EQ(reg.size(), 0u);
}

TEST(BufferRegistry, DeregisterDoesNotReuseId) {
    ldam::BufferRegistry reg;
    int d = 0;
    uint32_t id1 = reg.register_buffer(&d, 10, 0);
    reg.deregister(id1);

    uint32_t id2 = reg.register_buffer(&d, 20, 0);
    EXPECT_GT(id2, id1);
}

// ── Contains / size ─────────────────────────────────────────────────────────

TEST(BufferRegistry, ContainsTrueAfterRegister) {
    ldam::BufferRegistry reg;
    int d = 0;
    uint32_t id = reg.register_buffer(&d, 10, 0);
    EXPECT_TRUE(reg.contains(id));
}

TEST(BufferRegistry, ContainsFalseAfterDeregister) {
    ldam::BufferRegistry reg;
    int d = 0;
    uint32_t id = reg.register_buffer(&d, 10, 0);
    reg.deregister(id);
    EXPECT_FALSE(reg.contains(id));
}

TEST(BufferRegistry, SizeTracksDynamically) {
    ldam::BufferRegistry reg;
    int d = 0;
    uint32_t id1 = reg.register_buffer(&d, 10, 0);
    EXPECT_EQ(reg.size(), 1u);
    uint32_t id2 = reg.register_buffer(&d, 20, 1);
    EXPECT_EQ(reg.size(), 2u);
    reg.deregister(id1);
    EXPECT_EQ(reg.size(), 1u);
    reg.deregister(id2);
    EXPECT_EQ(reg.size(), 0u);
}

// ── Clear ───────────────────────────────────────────────────────────────────

TEST(BufferRegistry, ClearRemovesAll) {
    ldam::BufferRegistry reg;
    int d = 0;
    uint32_t id1 = reg.register_buffer(&d, 10, 0, "a");
    uint32_t id2 = reg.register_buffer(&d, 20, 1, "b");
    EXPECT_EQ(reg.size(), 2u);

    reg.clear();
    EXPECT_TRUE(reg.empty());
    EXPECT_FALSE(reg.contains(id1));
    EXPECT_FALSE(reg.contains(id2));

    // IDs still not reused after clear.
    uint32_t id3 = reg.register_buffer(&d, 30, 0);
    EXPECT_GT(id3, id2);
}

// ── all_named_entries ───────────────────────────────────────────────────────

TEST(BufferRegistry, AllNamedEntriesReturnsAll) {
    ldam::BufferRegistry reg;
    int d = 0;
    reg.register_buffer(&d, 10, 0, "alpha");
    reg.register_buffer(&d, 20, 1);  // unnamed
    reg.register_buffer(&d, 30, 2, "gamma");

    auto entries = reg.all_named_entries();
    EXPECT_EQ(entries.size(), 3u);

    // Find the named ones.
    int named_count = 0;
    for (const auto& [id, name] : entries) {
        if (!name.empty()) ++named_count;
    }
    EXPECT_EQ(named_count, 2);
}
