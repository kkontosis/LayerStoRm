// Unit tests for ArenaCache (P-24b) — the persistence meta layer.
// Covers: meta segment RAII, format/validate + every wipe condition,
// INV-ARENA-CACHE-ORDER record ops, hash-is-index-only (full-tuple verify),
// identity (mtime/size) invalidation, and the adoption scan — including
// cross-"run" survival of the meta table via a dup'd memfd.

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstring>
#include <optional>
#include <vector>

#include "core/memory/arena_cache.h"

namespace lm = layerstorm::memory;

namespace {

lm::ExpertKey key(uint32_t layer, uint16_t expert) { return {layer, expert}; }

std::vector<lm::ArenaCacheNodeGeom> two_node_geom() {
    return {{0, 4}, {2, 3}};  // node 0: 4 slots, node 2: 3 slots
}

std::vector<lm::ExpertFileIdentity> idents(uint16_t n, uint64_t salt = 0) {
    std::vector<lm::ExpertFileIdentity> v(n);
    for (uint16_t e = 0; e < n; ++e)
        v[e] = {1000000ULL + e + salt, 4096ULL * (e + 1) + salt};
    return v;
}

struct CacheFixture {
    lm::ArenaMetaSegment seg;
    std::unique_ptr<lm::ArenaCache> cache;
    uint64_t gh = 0x1234, sid = 0x5678;

    explicit CacheFixture(const std::vector<lm::ArenaCacheNodeGeom>& geom =
                              two_node_geom()) {
        seg = lm::ArenaMetaSegment::create(lm::ArenaCache::required_bytes(geom));
        EXPECT_TRUE(seg.valid());
        cache = std::make_unique<lm::ArenaCache>(seg.base(), seg.bytes());
        EXPECT_TRUE(cache->format(gh, sid, geom));
        cache->set_file_identities(idents(16));
    }
};

}  // namespace

TEST(ArenaMetaSegment, CreateAdoptRoundtrip) {
    auto seg = lm::ArenaMetaSegment::create(1000);
    ASSERT_TRUE(seg.valid());
    EXPECT_EQ(seg.bytes() % 4096, 0u);  // page-rounded
    std::memset(seg.base(), 0x7E, 100);

    const int dup_fd = ::dup(seg.fd());
    ASSERT_GE(dup_fd, 0);
    auto seg2 = lm::ArenaMetaSegment::adopt(dup_fd);
    ASSERT_TRUE(seg2.valid());
    EXPECT_EQ(seg2.bytes(), seg.bytes());
    EXPECT_EQ(static_cast<unsigned char*>(seg2.base())[42], 0x7E);
}

TEST(ArenaMetaSegment, AdoptBadFdInvalid) {
    EXPECT_FALSE(lm::ArenaMetaSegment::adopt(-1).valid());
}

TEST(ArenaCacheTest, FormatThenValidateSucceeds) {
    CacheFixture f;
    lm::ArenaCache reopened(f.seg.base(), f.seg.bytes());
    EXPECT_TRUE(reopened.validate(f.gh, f.sid, two_node_geom()));
}

TEST(ArenaCacheTest, ValidateWipesOnAnyMismatch) {
    CacheFixture f;
    lm::ArenaCache c(f.seg.base(), f.seg.bytes());
    EXPECT_FALSE(c.validate(f.gh + 1, f.sid, two_node_geom()));  // geometry
    EXPECT_FALSE(c.validate(f.gh, f.sid + 1, two_node_geom()));  // source
    auto other = two_node_geom();
    other[1].num_slots = 5;                                      // span shape
    EXPECT_FALSE(c.validate(f.gh, f.sid, other));
    other = two_node_geom();
    other[1].node = 3;                                           // node id
    EXPECT_FALSE(c.validate(f.gh, f.sid, other));
}

TEST(ArenaCacheTest, LoadReadyThenLookupHit) {
    CacheFixture f;
    const auto k = key(5, 7);
    f.cache->on_load_start(0, 2, k);
    EXPECT_FALSE(f.cache->lookup(k).has_value());  // LOADING ≠ adoptable
    f.cache->on_load_ready(0, 2, k);
    auto hit = f.cache->lookup(k);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->node, 0);
    EXPECT_EQ(hit->slot, 2u);
}

TEST(ArenaCacheTest, SyncPathReadyWithoutLoadStart) {
    CacheFixture f;  // the synchronous loader never calls mark_loading
    f.cache->on_load_ready(2, 1, key(3, 4));
    ASSERT_TRUE(f.cache->lookup(key(3, 4)).has_value());
}

TEST(ArenaCacheTest, ReuseKillsRecordBeforeRewrite) {
    CacheFixture f;
    const auto k = key(1, 2);
    f.cache->on_load_ready(0, 0, k);
    ASSERT_TRUE(f.cache->lookup(k).has_value());
    f.cache->on_reuse(0, 0);  // INV-ARENA-CACHE-ORDER: before any byte lands
    EXPECT_FALSE(f.cache->lookup(k).has_value());
}

TEST(ArenaCacheTest, IdentityChangeInvalidatesLookup) {
    CacheFixture f;
    const auto k = key(2, 3);
    f.cache->on_load_ready(0, 1, k);
    ASSERT_TRUE(f.cache->lookup(k).has_value());
    // "touch expert_003.bin": new mtime/size ⇒ the stored tuple mismatches.
    f.cache->set_file_identities(idents(16, /*salt=*/999));
    EXPECT_FALSE(f.cache->lookup(k).has_value());
}

TEST(ArenaCacheTest, ScanAdoptableSurvivesReopenAndFiltersInvalid) {
    CacheFixture f;
    f.cache->on_load_ready(0, 0, key(0, 1));   // adoptable
    f.cache->on_load_ready(2, 2, key(4, 8));   // adoptable
    f.cache->on_load_start(0, 3, key(6, 6));   // interrupted (LOADING) → reset
    f.cache->on_load_ready(2, 0, key(50, 2));  // out of layer range → reset

    // "Engine restart": a fresh ArenaCache over the SAME memory.
    lm::ArenaCache c2(f.seg.base(), f.seg.bytes());
    ASSERT_TRUE(c2.validate(f.gh, f.sid, two_node_geom()));
    c2.set_file_identities(idents(16));

    std::vector<std::tuple<int, size_t, uint32_t, uint16_t>> adopted;
    const size_t n = c2.scan_adoptable(
        /*num_layers=*/40, /*num_experts=*/16,
        [&](int node, size_t slot, lm::ExpertKey k) {
            adopted.emplace_back(node, slot, k.layer_idx, k.expert_idx);
        });
    EXPECT_EQ(n, 2u);
    ASSERT_EQ(adopted.size(), 2u);
    // Post-scan: adoptable records are indexed, invalid ones were reset.
    EXPECT_TRUE(c2.lookup(key(0, 1)).has_value());
    EXPECT_TRUE(c2.lookup(key(4, 8)).has_value());
    EXPECT_FALSE(c2.lookup(key(6, 6)).has_value());
    EXPECT_FALSE(c2.lookup(key(50, 2)).has_value());
    // A second scan is idempotent (invalid records already EMPTY).
    EXPECT_EQ(c2.scan_adoptable(40, 16, nullptr), 2u);
}

TEST(ArenaCacheTest, ScanRejectsStaleIdentity) {
    CacheFixture f;
    f.cache->on_load_ready(0, 0, key(0, 1));
    lm::ArenaCache c2(f.seg.base(), f.seg.bytes());
    ASSERT_TRUE(c2.validate(f.gh, f.sid, two_node_geom()));
    c2.set_file_identities(idents(16, /*salt=*/7));  // files were re-packed
    EXPECT_EQ(c2.scan_adoptable(40, 16, nullptr), 0u);
}

TEST(ArenaCacheTest, OpenStoredIgnoresSlotCountDrift) {
    // The whole point of open_stored: slot counts live in the SPANS, not the
    // hash — a free-RAM-drifted recomputation must not wipe a valid store.
    CacheFixture f;
    f.cache->on_load_ready(0, 1, key(2, 2));
    lm::ArenaCache c2(f.seg.base(), f.seg.bytes());
    ASSERT_TRUE(c2.open_stored(f.gh, f.sid));  // no geometry argument at all
    const auto stored = c2.stored_geometry();
    ASSERT_EQ(stored.size(), 2u);
    EXPECT_EQ(stored[0].node, 0);
    EXPECT_EQ(stored[0].num_slots, 4u);
    EXPECT_EQ(stored[1].node, 2);
    EXPECT_EQ(stored[1].num_slots, 3u);
    c2.set_file_identities(idents(16));
    EXPECT_EQ(c2.scan_adoptable(40, 16, nullptr), 1u);
    // Config-level mismatch still wipes.
    lm::ArenaCache c3(f.seg.base(), f.seg.bytes());
    EXPECT_FALSE(c3.open_stored(f.gh + 1, f.sid));
    EXPECT_FALSE(c3.open_stored(f.gh, f.sid + 1));
}

TEST(ArenaCacheTest, HashConfigSensitivity) {
    layerstorm::config::PinHostExpertPoolSizingConfig sizing;
    layerstorm::config::CrossNodeSpillConfig spill;
    std::vector<lm::ArenaCache::NodeIdentity> nodes{{0, 2}, {2, 1}};
    const uint64_t h = lm::ArenaCache::hash_config(64, 0, nodes, sizing, spill);
    EXPECT_EQ(h, lm::ArenaCache::hash_config(64, 0, nodes, sizing, spill));
    EXPECT_NE(h, lm::ArenaCache::hash_config(128, 0, nodes, sizing, spill));
    auto sizing2 = sizing;
    sizing2.value = 0.6;  // fraction knob changed → different identity
    EXPECT_NE(h, lm::ArenaCache::hash_config(64, 0, nodes, sizing2, spill));
    auto spill2 = spill;
    spill2.nodes.push_back({4, 1});
    EXPECT_NE(h, lm::ArenaCache::hash_config(64, 0, nodes, sizing, spill2));
    std::vector<lm::ArenaCache::NodeIdentity> nodes2{{0, 1}, {2, 1}};
    EXPECT_NE(h, lm::ArenaCache::hash_config(64, 0, nodes2, sizing, spill));
    // Arena host placement policy identity (arena_placement.h): id 0 (off)
    // must NOT change the hash (pre-placement stores keep their identity —
    // no spurious one-time wipe); any non-zero id folds in and distinct ids
    // give distinct identities.
    EXPECT_EQ(h, lm::ArenaCache::hash_config(64, 0, nodes, sizing, spill, 0));
    const uint64_t hp1 =
        lm::ArenaCache::hash_config(64, 0, nodes, sizing, spill, 0x1234);
    const uint64_t hp2 =
        lm::ArenaCache::hash_config(64, 0, nodes, sizing, spill, 0x1235);
    EXPECT_NE(h, hp1);
    EXPECT_NE(hp1, hp2);
}

TEST(ArenaCacheTest, KeyHashDependsOnIdentityAndKey) {
    CacheFixture f;
    const uint64_t h1 = f.cache->key_hash(key(1, 2));
    EXPECT_NE(h1, f.cache->key_hash(key(1, 3)));
    EXPECT_NE(h1, f.cache->key_hash(key(2, 2)));
    f.cache->set_file_identities(idents(16, /*salt=*/1));
    EXPECT_NE(h1, f.cache->key_hash(key(1, 2)));  // identity feeds the hash
}

// ── EMA persistence trailer (TD-ARENA-MIGRATE-EMA-PERSIST) ──────────────────

namespace {

std::vector<float> ema_pattern(uint32_t nl, uint32_t ne) {
    std::vector<float> v(static_cast<size_t>(nl) * ne);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = 0.25f * static_cast<float>(i % 97);
    return v;
}

}  // namespace

TEST(ArenaCacheEma, RoundtripSurvivesReopenViaAdoptedFd) {
    constexpr uint32_t kL = 6, kE = 8;
    const auto geom = two_node_geom();
    auto seg = lm::ArenaMetaSegment::create(
        lm::ArenaCache::required_bytes_with_ema(geom, kL, kE));
    ASSERT_TRUE(seg.valid());
    lm::ArenaCache cache(seg.base(), seg.bytes());
    ASSERT_TRUE(cache.format(0x1234, 0x5678, geom));
    cache.set_file_identities(idents(16));
    cache.on_load_ready(0, 1, key(2, 3));  // records coexist with the trailer

    EXPECT_FALSE(cache.ema_load(kL, kE).has_value());  // fresh: no blob
    const auto data = ema_pattern(kL, kE);
    ASSERT_TRUE(cache.ema_save(data.data(), kL, kE, /*fetches_seen=*/42017.0));

    // "Next boot": adopt the memfd (dup — the holder's copy), reopen.
    const int dup_fd = ::dup(seg.fd());
    ASSERT_GE(dup_fd, 0);
    auto seg2 = lm::ArenaMetaSegment::adopt(dup_fd);
    ASSERT_TRUE(seg2.valid());
    lm::ArenaCache reopened(seg2.base(), seg2.bytes());
    ASSERT_TRUE(reopened.open_stored(0x1234, 0x5678));
    const auto v = reopened.ema_load(kL, kE);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->num_layers, kL);
    EXPECT_EQ(v->num_experts, kE);
    EXPECT_DOUBLE_EQ(v->fetches_seen, 42017.0);
    EXPECT_GT(v->saved_unix_ns, 0u);
    EXPECT_EQ(std::memcmp(v->data, data.data(), data.size() * sizeof(float)),
              0);
    // The record table is untouched by the trailer (index rebuilds via the
    // adoption scan, as on a real warm attach).
    reopened.set_file_identities(idents(16));
    EXPECT_EQ(reopened.scan_adoptable(64, 16, nullptr), 1u);
    EXPECT_TRUE(reopened.lookup(key(2, 3)).has_value());
}

TEST(ArenaCacheEma, PreTrailerStoreHasNoRoomAndNeverWipes) {
    CacheFixture f;  // sized required_bytes only (pre-trailer store)
    // A production-shaped trailer (~78 KiB) exceeds the segment's page-round
    // slack: no room → save refused. (A blob small enough for the slack
    // would land in zero-initialized bytes anyway — state invalid ⇒ load
    // still starts cold; either way, never wrong.)
    const auto data = ema_pattern(78, 256);
    EXPECT_FALSE(f.cache->ema_save(data.data(), 78, 256, 1.0));
    EXPECT_FALSE(f.cache->ema_load(78, 256).has_value());
    EXPECT_FALSE(f.cache->ema_load(6, 8).has_value());  // slack reads invalid
    // The store still opens fine — absence of the trailer is never a wipe.
    lm::ArenaCache reopened(f.seg.base(), f.seg.bytes());
    EXPECT_TRUE(reopened.open_stored(f.gh, f.sid));
}

TEST(ArenaCacheEma, ShapeMismatchAndInvalidStateRejected) {
    constexpr uint32_t kL = 6, kE = 8;
    const auto geom = two_node_geom();
    auto seg = lm::ArenaMetaSegment::create(
        lm::ArenaCache::required_bytes_with_ema(geom, kL, kE));
    ASSERT_TRUE(seg.valid());
    lm::ArenaCache cache(seg.base(), seg.bytes());
    ASSERT_TRUE(cache.format(1, 2, geom));
    const auto data = ema_pattern(kL, kE);
    ASSERT_TRUE(cache.ema_save(data.data(), kL, kE, 5.0));
    // A DIFFERENT model shape must never adopt this blob (even when smaller,
    // i.e. the room check alone would pass).
    EXPECT_FALSE(cache.ema_load(kL, kE - 1).has_value());
    EXPECT_FALSE(cache.ema_load(kL - 1, kE).has_value());
    EXPECT_FALSE(cache.ema_load(100, 100).has_value());  // no room either
    EXPECT_TRUE(cache.ema_load(kL, kE).has_value());
    // Interrupted-save simulation (INV-ARENA-CACHE-ORDER): state invalid ⇒
    // start cold.
    const size_t off =
        (lm::ArenaCache::required_bytes(geom) + 63) & ~size_t{63};
    auto* trailer = reinterpret_cast<lm::ArenaCache::EmaTrailer*>(
        static_cast<char*>(seg.base()) + off);
    trailer->state = 0;
    EXPECT_FALSE(cache.ema_load(kL, kE).has_value());
    trailer->state = 1;
    trailer->version = lm::ArenaCache::kEmaVersion + 1;  // future format
    EXPECT_FALSE(cache.ema_load(kL, kE).has_value());
}

TEST(ArenaCacheEma, EnsureBytesGrowsInPlacePreservingRecords) {
    constexpr uint32_t kL = 64, kE = 64;  // trailer 16 KiB > page slack
    const auto geom = two_node_geom();
    // Old-generation store: no trailer room.
    auto seg = lm::ArenaMetaSegment::create(lm::ArenaCache::required_bytes(geom));
    ASSERT_TRUE(seg.valid());
    {
        lm::ArenaCache cache(seg.base(), seg.bytes());
        ASSERT_TRUE(cache.format(9, 9, geom));
        cache.set_file_identities(idents(16));
        cache.on_load_ready(2, 1, key(4, 5));
        EXPECT_FALSE(cache.ema_save(ema_pattern(kL, kE).data(), kL, kE, 1.0));
    }
    // Grow (the warm-attach path) — same fd, remapped base.
    const size_t want =
        lm::ArenaCache::required_bytes_with_ema(geom, kL, kE);
    ASSERT_TRUE(seg.ensure_bytes(want));
    ASSERT_GE(seg.bytes(), want);
    EXPECT_TRUE(seg.ensure_bytes(want));  // idempotent no-op
    lm::ArenaCache cache(seg.base(), seg.bytes());
    ASSERT_TRUE(cache.open_stored(9, 9));
    cache.set_file_identities(idents(16));
    // Records survived the remap; the trailer now has room and works.
    size_t adopted = cache.scan_adoptable(64, 16, nullptr);
    EXPECT_EQ(adopted, 1u);
    EXPECT_FALSE(cache.ema_load(kL, kE).has_value());  // new bytes read zero
    const auto data = ema_pattern(kL, kE);
    ASSERT_TRUE(cache.ema_save(data.data(), kL, kE, 7.0));
    const auto v = cache.ema_load(kL, kE);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(std::memcmp(v->data, data.data(), data.size() * sizeof(float)),
              0);
}
