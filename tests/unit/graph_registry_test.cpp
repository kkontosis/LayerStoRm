#include "compute/graphs/graph_registry.h"

#include <algorithm>
#include <set>

#include <gtest/gtest.h>

namespace lc = layerstorm::compute;

// ── Mock runner ─────────────────────────────────────────────────────────────

struct MockRunner {
    int  id{};
    bool* destroyed{};  // set to true when destroy callback fires
};

static lc::GraphEntry make_mock_entry(MockRunner* runner) {
    return lc::GraphEntry{
        .runner  = runner,
        .destroy = [](std::any& r) {
            auto* p = std::any_cast<MockRunner*>(r);
            if (p->destroyed) *p->destroyed = true;
            delete p;
        }
    };
}

static lc::GraphKey attn_key(int gpu, int bs) {
    return {lc::GraphType::kAttentionDecode, gpu, bs};
}

static lc::GraphKey dcp_key(int gpu, int bs) {
    return {lc::GraphType::kDcpAllreduce, gpu, bs};
}

// ── Construction ────────────────────────────────────────────────────────────

TEST(GraphRegistry, DefaultConstructionEmpty) {
    lc::GraphRegistry reg;
    EXPECT_TRUE(reg.empty());
    EXPECT_EQ(reg.size(), 0u);
}

// ── Insert + find ───────────────────────────────────────────────────────────

TEST(GraphRegistry, InsertAndFind) {
    lc::GraphRegistry reg;
    bool destroyed = false;
    auto* runner = new MockRunner{42, &destroyed};
    reg.insert(attn_key(0, 1), make_mock_entry(runner));

    EXPECT_EQ(reg.size(), 1u);
    EXPECT_FALSE(reg.empty());

    auto* entry = reg.find(attn_key(0, 1));
    ASSERT_NE(entry, nullptr);
    auto* found = std::any_cast<MockRunner*>(entry->runner);
    EXPECT_EQ(found->id, 42);
    EXPECT_FALSE(destroyed);
}

TEST(GraphRegistry, FindAs) {
    lc::GraphRegistry reg;
    bool destroyed = false;
    auto* runner = new MockRunner{7, &destroyed};
    reg.insert(attn_key(1, 4), make_mock_entry(runner));

    auto* found = reg.find_as<MockRunner>(attn_key(1, 4));
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->id, 7);
}

TEST(GraphRegistry, FindAsMissing) {
    lc::GraphRegistry reg;
    EXPECT_EQ(reg.find_as<MockRunner>(attn_key(0, 1)), nullptr);
}

TEST(GraphRegistry, FindAsWrongType) {
    lc::GraphRegistry reg;
    // Store an int* instead of MockRunner*
    int value = 99;
    reg.insert(attn_key(0, 1), lc::GraphEntry{
        .runner = &value,
        .destroy = [](std::any&) {}
    });
    EXPECT_THROW(reg.find_as<MockRunner>(attn_key(0, 1)), std::bad_any_cast);
}

// ── Duplicate insert ────────────────────────────────────────────────────────

TEST(GraphRegistry, DuplicateInsertThrows) {
    lc::GraphRegistry reg;
    bool d1 = false, d2 = false;
    reg.insert(attn_key(0, 1), make_mock_entry(new MockRunner{1, &d1}));
    EXPECT_THROW(
        reg.insert(attn_key(0, 1), make_mock_entry(new MockRunner{2, &d2})),
        std::runtime_error);
    // The second runner was not inserted, but we created it with new — clean up
    // (In real code the caller would handle this; here we just leak-check via ASAN)
}

// ── Find / get missing ─────────────────────────────────────────────────────

TEST(GraphRegistry, FindMissingReturnsNull) {
    lc::GraphRegistry reg;
    EXPECT_EQ(reg.find(attn_key(0, 1)), nullptr);
}

TEST(GraphRegistry, GetMissingThrows) {
    lc::GraphRegistry reg;
    EXPECT_THROW(reg.get(attn_key(0, 1)), std::runtime_error);
}

TEST(GraphRegistry, GetExisting) {
    lc::GraphRegistry reg;
    bool destroyed = false;
    reg.insert(attn_key(0, 8), make_mock_entry(new MockRunner{8, &destroyed}));
    auto& entry = reg.get(attn_key(0, 8));
    EXPECT_EQ(std::any_cast<MockRunner*>(entry.runner)->id, 8);
}

// ── Contains ────────────────────────────────────────────────────────────────

TEST(GraphRegistry, Contains) {
    lc::GraphRegistry reg;
    EXPECT_FALSE(reg.contains(attn_key(0, 1)));
    bool d = false;
    reg.insert(attn_key(0, 1), make_mock_entry(new MockRunner{1, &d}));
    EXPECT_TRUE(reg.contains(attn_key(0, 1)));
    EXPECT_FALSE(reg.contains(attn_key(0, 2)));
    EXPECT_FALSE(reg.contains(attn_key(1, 1)));
    EXPECT_FALSE(reg.contains(dcp_key(0, 1)));
}

// ── Remove ──────────────────────────────────────────────────────────────────

TEST(GraphRegistry, RemoveCallsDestroy) {
    lc::GraphRegistry reg;
    bool destroyed = false;
    reg.insert(attn_key(0, 1), make_mock_entry(new MockRunner{1, &destroyed}));
    EXPECT_FALSE(destroyed);

    reg.remove(attn_key(0, 1));
    EXPECT_TRUE(destroyed);
    EXPECT_TRUE(reg.empty());
    EXPECT_EQ(reg.find(attn_key(0, 1)), nullptr);
}

TEST(GraphRegistry, RemoveNonExistentIsNoop) {
    lc::GraphRegistry reg;
    reg.remove(attn_key(0, 1));  // no throw, no crash
    EXPECT_TRUE(reg.empty());
}

// ── Clear ───────────────────────────────────────────────────────────────────

TEST(GraphRegistry, ClearDestroysAll) {
    lc::GraphRegistry reg;
    bool d1 = false, d2 = false, d3 = false;
    reg.insert(attn_key(0, 1), make_mock_entry(new MockRunner{1, &d1}));
    reg.insert(attn_key(0, 4), make_mock_entry(new MockRunner{2, &d2}));
    reg.insert(dcp_key(0, 1),  make_mock_entry(new MockRunner{3, &d3}));
    EXPECT_EQ(reg.size(), 3u);

    reg.clear();
    EXPECT_TRUE(d1);
    EXPECT_TRUE(d2);
    EXPECT_TRUE(d3);
    EXPECT_TRUE(reg.empty());
}

// ── Destructor ──────────────────────────────────────────────────────────────

TEST(GraphRegistry, DestructorCallsDestroy) {
    bool d1 = false, d2 = false;
    {
        lc::GraphRegistry reg;
        reg.insert(attn_key(0, 1), make_mock_entry(new MockRunner{1, &d1}));
        reg.insert(dcp_key(1, 8), make_mock_entry(new MockRunner{2, &d2}));
    }  // destructor fires
    EXPECT_TRUE(d1);
    EXPECT_TRUE(d2);
}

// ── Size / empty tracking ───────────────────────────────────────────────────

TEST(GraphRegistry, SizeTracking) {
    lc::GraphRegistry reg;
    EXPECT_EQ(reg.size(), 0u);
    EXPECT_TRUE(reg.empty());

    bool d1 = false, d2 = false;
    reg.insert(attn_key(0, 1), make_mock_entry(new MockRunner{1, &d1}));
    EXPECT_EQ(reg.size(), 1u);
    EXPECT_FALSE(reg.empty());

    reg.insert(attn_key(0, 4), make_mock_entry(new MockRunner{2, &d2}));
    EXPECT_EQ(reg.size(), 2u);

    reg.remove(attn_key(0, 1));
    EXPECT_EQ(reg.size(), 1u);

    reg.clear();
    EXPECT_EQ(reg.size(), 0u);
    EXPECT_TRUE(reg.empty());
}

// ── Keys enumeration ────────────────────────────────────────────────────────

TEST(GraphRegistry, KeysEnumeration) {
    lc::GraphRegistry reg;
    bool d1 = false, d2 = false, d3 = false;
    reg.insert(attn_key(0, 1), make_mock_entry(new MockRunner{1, &d1}));
    reg.insert(attn_key(1, 4), make_mock_entry(new MockRunner{2, &d2}));
    reg.insert(dcp_key(0, 1),  make_mock_entry(new MockRunner{3, &d3}));

    auto all_keys = reg.keys();
    EXPECT_EQ(all_keys.size(), 3u);

    // Verify all 3 keys are present (order not guaranteed)
    auto has = [&](lc::GraphKey k) {
        return std::find(all_keys.begin(), all_keys.end(), k) != all_keys.end();
    };
    EXPECT_TRUE(has(attn_key(0, 1)));
    EXPECT_TRUE(has(attn_key(1, 4)));
    EXPECT_TRUE(has(dcp_key(0, 1)));
}

// ── Count / keys by type ────────────────────────────────────────────────────

TEST(GraphRegistry, CountByType) {
    lc::GraphRegistry reg;
    bool d1 = false, d2 = false, d3 = false;
    reg.insert(attn_key(0, 1), make_mock_entry(new MockRunner{1, &d1}));
    reg.insert(attn_key(1, 4), make_mock_entry(new MockRunner{2, &d2}));
    reg.insert(dcp_key(0, 1),  make_mock_entry(new MockRunner{3, &d3}));

    EXPECT_EQ(reg.count_by_type(lc::GraphType::kAttentionDecode), 2u);
    EXPECT_EQ(reg.count_by_type(lc::GraphType::kDcpAllreduce), 1u);
}

TEST(GraphRegistry, KeysByType) {
    lc::GraphRegistry reg;
    bool d1 = false, d2 = false, d3 = false;
    reg.insert(attn_key(0, 1), make_mock_entry(new MockRunner{1, &d1}));
    reg.insert(attn_key(1, 4), make_mock_entry(new MockRunner{2, &d2}));
    reg.insert(dcp_key(0, 1),  make_mock_entry(new MockRunner{3, &d3}));

    auto attn_keys = reg.keys_by_type(lc::GraphType::kAttentionDecode);
    EXPECT_EQ(attn_keys.size(), 2u);

    auto dcp_keys = reg.keys_by_type(lc::GraphType::kDcpAllreduce);
    EXPECT_EQ(dcp_keys.size(), 1u);
    EXPECT_EQ(dcp_keys[0], dcp_key(0, 1));
}

// ── Multi-GPU / multi-batch isolation ───────────────────────────────────────

TEST(GraphRegistry, MultiGpuMultiBatchIsolation) {
    lc::GraphRegistry reg;
    bool d[6] = {};
    // 2 GPUs x 3 batch sizes
    for (int gpu = 0; gpu < 2; ++gpu) {
        for (int i = 0; i < 3; ++i) {
            int bs = 1 << i;  // 1, 2, 4
            int idx = gpu * 3 + i;
            reg.insert(attn_key(gpu, bs),
                       make_mock_entry(new MockRunner{idx, &d[idx]}));
        }
    }
    EXPECT_EQ(reg.size(), 6u);

    // Each is independently retrievable
    for (int gpu = 0; gpu < 2; ++gpu) {
        for (int i = 0; i < 3; ++i) {
            int bs = 1 << i;
            int idx = gpu * 3 + i;
            auto* r = reg.find_as<MockRunner>(attn_key(gpu, bs));
            ASSERT_NE(r, nullptr);
            EXPECT_EQ(r->id, idx);
        }
    }
}

// ── graph_type_name ─────────────────────────────────────────────────────────

TEST(GraphRegistry, GraphTypeName) {
    EXPECT_STREQ(lc::graph_type_name(lc::GraphType::kAttentionDecode), "AttentionDecode");
    EXPECT_STREQ(lc::graph_type_name(lc::GraphType::kDcpAllreduce), "DcpAllreduce");
    EXPECT_STREQ(lc::graph_type_name(static_cast<lc::GraphType>(255)), "Unknown");
}

// ── Hash sanity ─────────────────────────────────────────────────────────────

TEST(GraphRegistry, HashDistinct) {
    // Test indirectly: insert distinct keys and verify all are retrievable.
    lc::GraphRegistry reg;
    bool d[4] = {};
    reg.insert(attn_key(0, 1), make_mock_entry(new MockRunner{0, &d[0]}));
    reg.insert(attn_key(0, 2), make_mock_entry(new MockRunner{1, &d[1]}));
    reg.insert(attn_key(1, 1), make_mock_entry(new MockRunner{2, &d[2]}));
    reg.insert(dcp_key(0, 1),  make_mock_entry(new MockRunner{3, &d[3]}));

    EXPECT_EQ(reg.size(), 4u);
    EXPECT_EQ(reg.find_as<MockRunner>(attn_key(0, 1))->id, 0);
    EXPECT_EQ(reg.find_as<MockRunner>(attn_key(0, 2))->id, 1);
    EXPECT_EQ(reg.find_as<MockRunner>(attn_key(1, 1))->id, 2);
    EXPECT_EQ(reg.find_as<MockRunner>(dcp_key(0, 1))->id, 3);
}

// ── Const access ────────────────────────────────────────────────────────────

TEST(GraphRegistry, ConstAccess) {
    lc::GraphRegistry reg;
    bool d = false;
    reg.insert(attn_key(0, 1), make_mock_entry(new MockRunner{10, &d}));

    const auto& creg = reg;
    const auto* entry = creg.find(attn_key(0, 1));
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(std::any_cast<MockRunner*>(entry->runner)->id, 10);

    const auto& ref = creg.get(attn_key(0, 1));
    EXPECT_EQ(std::any_cast<MockRunner*>(ref.runner)->id, 10);

    EXPECT_TRUE(creg.contains(attn_key(0, 1)));
    EXPECT_FALSE(creg.contains(attn_key(0, 2)));

    EXPECT_EQ(creg.find(attn_key(0, 2)), nullptr);
    EXPECT_THROW(creg.get(attn_key(0, 2)), std::runtime_error);
}

// ── Null destroy callback ───────────────────────────────────────────────────

TEST(GraphRegistry, NullDestroyCallbackSafe) {
    lc::GraphRegistry reg;
    reg.insert(attn_key(0, 1), lc::GraphEntry{
        .runner = 42,
        .destroy = nullptr
    });
    // remove and clear should not crash with null destroy
    reg.remove(attn_key(0, 1));
    EXPECT_TRUE(reg.empty());
}
