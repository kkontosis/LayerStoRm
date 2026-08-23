// Arena host-slot placement policy tests (Wave-2 M3, arena_placement.h):
// frequency table parsing, HBM-first + per-layer anti-gang assignment,
// determinism, capacity discipline, and the env-policy fail-closed contract.

#include "core/memory/arena_placement.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>

namespace lm = layerstorm::memory;

namespace {

// Write a temp freq CSV; returns the path (caller unlinks).
std::string write_csv(const std::string& body) {
    static int seq = 0;
    std::string path = ::testing::TempDir() + "arena_place_freq_" +
                       std::to_string(::getpid()) + "_" +
                       std::to_string(seq++) + ".csv";
    std::ofstream f(path);
    f << body;
    return path;
}

struct EnvGuard {
    explicit EnvGuard(const char* v) {
        if (v) ::setenv("LS_ARENA_PLACE_FREQ", v, 1);
        else   ::unsetenv("LS_ARENA_PLACE_FREQ");
    }
    ~EnvGuard() { ::unsetenv("LS_ARENA_PLACE_FREQ"); }
};

}  // namespace

// ── ArenaFreqTable ───────────────────────────────────────────────────────────

TEST(ArenaPlacementTest, FreqTableParsesRowsCommentsAndAccumulates) {
    const std::string p = write_csv(
        "# layer,expert,count\n"
        "3,0,10\n"
        "3,1,2.5\n"
        "\n"
        "4,255,7\n"
        "3,0,5\n");  // duplicate accumulates
    auto t = lm::ArenaFreqTable::load(p);
    EXPECT_EQ(t.counts.size(), 3u);
    EXPECT_DOUBLE_EQ(t.at({3, 0}), 15.0);
    EXPECT_DOUBLE_EQ(t.at({3, 1}), 2.5);
    EXPECT_DOUBLE_EQ(t.at({4, 255}), 7.0);
    EXPECT_DOUBLE_EQ(t.at({9, 9}), 0.0);  // missing → 0
    ::unlink(p.c_str());
}

TEST(ArenaPlacementTest, FreqTableThrowsOnMissingFileAndMalformedRow) {
    EXPECT_THROW(lm::ArenaFreqTable::load("/nonexistent/freq.csv"),
                 std::runtime_error);
    const std::string bad = write_csv("3,notanumber,1\n");
    EXPECT_THROW(lm::ArenaFreqTable::load(bad), std::runtime_error);
    ::unlink(bad.c_str());
    const std::string empty = write_csv("# only comments\n");
    EXPECT_THROW(lm::ArenaFreqTable::load(empty), std::runtime_error);
    ::unlink(empty.c_str());
}

// ── ArenaPlacementPolicy::from_env ───────────────────────────────────────────

TEST(ArenaPlacementTest, PolicyDisabledWhenEnvUnset) {
    EnvGuard g(nullptr);
    auto p = lm::ArenaPlacementPolicy::from_env();
    EXPECT_FALSE(p.enabled);
    EXPECT_EQ(p.identity, 0u);
}

TEST(ArenaPlacementTest, PolicyFailsClosedOnUnreadableFile) {
    EnvGuard g("/nonexistent/freq.csv");
    EXPECT_THROW(lm::ArenaPlacementPolicy::from_env(), std::runtime_error);
}

TEST(ArenaPlacementTest, PolicyIdentityTracksFileContentNotPath) {
    const std::string a = write_csv("3,0,10\n");
    const std::string b = write_csv("3,0,10\n");   // same content, other path
    const std::string c = write_csv("3,0,11\n");   // different content
    uint64_t ia, ib, ic;
    { EnvGuard g(a.c_str()); ia = lm::ArenaPlacementPolicy::from_env().identity; }
    { EnvGuard g(b.c_str()); ib = lm::ArenaPlacementPolicy::from_env().identity; }
    { EnvGuard g(c.c_str()); ic = lm::ArenaPlacementPolicy::from_env().identity; }
    EXPECT_NE(ia, 0u);
    EXPECT_EQ(ia, ib);
    EXPECT_NE(ia, ic);
    ::unlink(a.c_str()); ::unlink(b.c_str()); ::unlink(c.c_str());
}

// ── ArenaPlacementPolicy::resolve (config + env precedence) ─────────────────

TEST(ArenaPlacementTest, ResolveUsesConfigWhenEnvUnset) {
    EnvGuard g(nullptr);
    const std::string a = write_csv("3,0,10\n");
    EXPECT_EQ(lm::ArenaPlacementPolicy::resolved_path(a), a);
    auto p = lm::ArenaPlacementPolicy::resolve(a);
    EXPECT_TRUE(p.enabled);
    EXPECT_EQ(p.freq_path, a);
    EXPECT_NE(p.identity, 0u);
    // Same table via env vs config → identical identity (source-agnostic).
    uint64_t ienv;
    { EnvGuard e(a.c_str()); ienv = lm::ArenaPlacementPolicy::from_env().identity; }
    EXPECT_EQ(p.identity, ienv);
    ::unlink(a.c_str());
}

TEST(ArenaPlacementTest, ResolveEnvPathOverridesConfig) {
    const std::string cfg = write_csv("3,0,10\n");
    const std::string env = write_csv("3,0,11\n");
    EnvGuard g(env.c_str());
    auto p = lm::ArenaPlacementPolicy::resolve(cfg);
    EXPECT_TRUE(p.enabled);
    EXPECT_EQ(p.freq_path, env);
    ::unlink(cfg.c_str()); ::unlink(env.c_str());
}

TEST(ArenaPlacementTest, ResolveEnvOffDisablesConfigDefault) {
    const std::string cfg = write_csv("3,0,10\n");
    for (const char* off : {"off", "none", "0"}) {
        EnvGuard g(off);
        EXPECT_EQ(lm::ArenaPlacementPolicy::resolved_path(cfg), "");
        auto p = lm::ArenaPlacementPolicy::resolve(cfg);
        EXPECT_FALSE(p.enabled);
        EXPECT_EQ(p.identity, 0u);
    }
    ::unlink(cfg.c_str());
}

TEST(ArenaPlacementTest, ResolveDisabledWhenNeitherSet) {
    EnvGuard g(nullptr);
    EXPECT_EQ(lm::ArenaPlacementPolicy::resolved_path(""), "");
    auto p = lm::ArenaPlacementPolicy::resolve("");
    EXPECT_FALSE(p.enabled);
    EXPECT_EQ(p.identity, 0u);
}

TEST(ArenaPlacementTest, ResolveFailsClosedOnUnreadableConfigTable) {
    EnvGuard g(nullptr);
    EXPECT_THROW(lm::ArenaPlacementPolicy::resolve("/nonexistent/freq.csv"),
                 std::runtime_error);
}

// ── compute_arena_placement ──────────────────────────────────────────────────

namespace {

// 2 MoE layers × 8 experts; layer-L expert-e freq = descending in e so the
// per-layer frequency rank is exactly the expert index order.
std::vector<lm::ExpertKey> make_keys(uint32_t layers, uint16_t experts) {
    std::vector<lm::ExpertKey> keys;
    for (uint32_t L = 0; L < layers; ++L)
        for (uint16_t e = 0; e < experts; ++e) keys.push_back({L, e});
    return keys;
}

}  // namespace

TEST(ArenaPlacementTest, HbmFirstTakesGloballyHottestUpToCapacity) {
    // freq: key (0,0)=100, (0,1)=90, (1,0)=80, everything else 1.
    lm::ArenaFreqTable f;
    f.counts[{0, 0}] = 100;
    f.counts[{0, 1}] = 90;
    f.counts[{1, 0}] = 80;
    auto keys = make_keys(2, 8);
    for (const auto& k : keys)
        if (!f.counts.count(k)) f.counts[k] = 1;

    std::vector<lm::ArenaPlacementNode> nodes = {
        {0, 8, false}, {2, 8, false},          // DDR
        {4, 2, true},  {5, 1, true},           // HBM: 3 total slots
    };
    auto m = lm::compute_arena_placement(f, keys, nodes);
    ASSERT_EQ(m.size(), keys.size());  // full fit (16 keys, 19 slots)

    // The 3 hottest keys are on HBM nodes.
    std::set<int> hbm{4, 5};
    EXPECT_TRUE(hbm.count(m.at({0, 0})));
    EXPECT_TRUE(hbm.count(m.at({0, 1})));
    EXPECT_TRUE(hbm.count(m.at({1, 0})));
    // Nobody else is (capacity exhausted).
    int on_hbm = 0;
    for (const auto& [k, n] : m) on_hbm += hbm.count(n);
    EXPECT_EQ(on_hbm, 3);
}

TEST(ArenaPlacementTest, AntiGangSpreadsPerLayerRanksAcrossDdrNodes) {
    // No HBM. 4 DDR nodes; one layer, 8 experts, freq desc in expert idx.
    lm::ArenaFreqTable f;
    auto keys = make_keys(1, 8);
    for (const auto& k : keys) f.counts[k] = 100.0 - k.expert_idx;
    std::vector<lm::ArenaPlacementNode> nodes = {
        {0, 4, false}, {1, 4, false}, {2, 4, false}, {3, 4, false}};
    auto m = lm::compute_arena_placement(f, keys, nodes);
    ASSERT_EQ(m.size(), 8u);
    // The top-4 (rank 0..3) land on 4 DISTINCT nodes, as do rank 4..7.
    std::set<int> top4{m.at({0, 0}), m.at({0, 1}), m.at({0, 2}), m.at({0, 3})};
    std::set<int> next4{m.at({0, 4}), m.at({0, 5}), m.at({0, 6}), m.at({0, 7})};
    EXPECT_EQ(top4.size(), 4u);
    EXPECT_EQ(next4.size(), 4u);
}

TEST(ArenaPlacementTest, LayerOffsetRotatesStartNode) {
    // Two layers with identical freq patterns start their round-robin on
    // DIFFERENT nodes ((rank + layer) % n).
    lm::ArenaFreqTable f;
    auto keys = make_keys(2, 2);
    for (const auto& k : keys) f.counts[k] = 10.0 - k.expert_idx;
    std::vector<lm::ArenaPlacementNode> nodes = {
        {0, 4, false}, {1, 4, false}, {2, 4, false}, {3, 4, false}};
    auto m = lm::compute_arena_placement(f, keys, nodes);
    EXPECT_NE(m.at({0, 0}), m.at({1, 0}));  // rank-0 of each layer differs
}

TEST(ArenaPlacementTest, CapacityRespectedAndOverflowUnassigned) {
    lm::ArenaFreqTable f;
    auto keys = make_keys(1, 8);
    for (const auto& k : keys) f.counts[k] = 8.0 - k.expert_idx;
    std::vector<lm::ArenaPlacementNode> nodes = {{0, 3, false}, {1, 2, false}};
    auto m = lm::compute_arena_placement(f, keys, nodes);
    EXPECT_EQ(m.size(), 5u);  // 5 slots total
    // Per-node counts never exceed capacity.
    int n0 = 0, n1 = 0;
    for (const auto& [k, n] : m) (n == 0 ? n0 : n1)++;
    EXPECT_EQ(n0, 3);
    EXPECT_EQ(n1, 2);
    // The unassigned keys are the 3 COLDEST (placement favors hot keys).
    EXPECT_TRUE(m.count({0, 0}));
    EXPECT_TRUE(m.count({0, 1}));
    EXPECT_FALSE(m.count({0, 7}));
}

TEST(ArenaPlacementTest, DeterministicAcrossCalls) {
    lm::ArenaFreqTable f;
    auto keys = make_keys(3, 16);
    for (const auto& k : keys)
        f.counts[k] = static_cast<double>((k.layer_idx * 7 + k.expert_idx * 13)
                                          % 11);
    std::vector<lm::ArenaPlacementNode> nodes = {
        {0, 12, false}, {1, 12, false}, {2, 12, false},
        {4, 6, true}, {5, 6, true}};
    auto a = lm::compute_arena_placement(f, keys, nodes);
    auto b = lm::compute_arena_placement(f, keys, nodes);
    EXPECT_EQ(a, b);
    EXPECT_EQ(a.size(), keys.size());
}

TEST(ArenaPlacementTest, ZeroFreqTailStillPlacedFullFit) {
    lm::ArenaFreqTable f;
    f.counts[{0, 0}] = 5;  // only one key has signal
    auto keys = make_keys(1, 6);
    std::vector<lm::ArenaPlacementNode> nodes = {{0, 4, false}, {1, 4, false}};
    auto m = lm::compute_arena_placement(f, keys, nodes);
    EXPECT_EQ(m.size(), 6u);  // zero-freq keys are still assigned
}
