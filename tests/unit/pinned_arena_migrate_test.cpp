// Unit tests for the M3b online-migration primitives on PinnedExpertArena
// (INV-ARENA-MIGRATE): evict / evict_key / migrate_begin / migrate_commit /
// migrate_abort, their interlock with the in-flight H2D refcount and the J-1
// loading flag, and the ArenaCache record discipline across a migration
// (INV-ARENA-CACHE-ORDER: the moving key is adoptable at EVERY instant, from
// exactly one truthful record — transiently two identical ones).
//
// All tests run with defer_registration=true and never register — no CUDA
// needed (slots are plain host memory until finalize_registration).

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
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

// Two GPU-attached nodes (0 and 1) → one arena per node. Stands in for the
// production DDR→HBM promotion pair (the primitives are node-class-agnostic).
lc::HardwareConfig two_node_hw() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 128;
    hw.gpus = {make_gpu(0, 0), make_gpu(1, 1)};
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

// `slots_per_node` slots on EACH of the two node arenas (total budget is split
// per-node by the facade: per_node_target = total / n_gpu_nodes).
std::unique_ptr<lm::PinnedExpertArena> make_arena(lm::NumaManager& numa,
                                                  size_t slots_per_node) {
    return std::make_unique<lm::PinnedExpertArena>(
        numa, kSlot, kSlot * slots_per_node * 2, big_sizing(),
        lc::CrossNodeSpillConfig{}, /*extra_scratch_bytes=*/0,
        /*defer_registration=*/true, nullptr);
}

// Synchronous fill helper: reserve on `node`, tag the bytes, mark ready.
void fill_on_node(lm::PinnedExpertArena& a, lm::ExpertKey k, int node,
                  unsigned char tag) {
    void* s = a.reserve_on_node(k, node);
    ASSERT_NE(s, nullptr) << "no free slot on node " << node;
    std::memset(s, tag, kSlot);
    a.mark_ready(k);
}

}  // namespace

// ── evict (node) / evict_key (facade) guards ─────────────────────────────────

TEST(PinnedArenaMigrate, EvictKeyGuardsPins) {
    lm::NumaManager numa(two_node_hw());
    auto arena = make_arena(numa, 2);
    const auto k = key(0, 1);
    fill_on_node(*arena, k, 0, 0xAA);

    // In-flight H2D pins the slot against eviction.
    arena->acquire_inflight(k);
    EXPECT_FALSE(arena->evict_key(k));
    EXPECT_EQ(arena->location_node(k), 0);
    arena->release_inflight(k);

    // Loading pins it too.
    arena->mark_loading(k);
    EXPECT_FALSE(arena->evict_key(k));
    arena->mark_ready(k);

    // Unpinned: evicts, drops the location, slot becomes free.
    EXPECT_TRUE(arena->evict_key(k));
    EXPECT_EQ(arena->location_node(k), -1);
    EXPECT_EQ(arena->resolve(k), nullptr);
    EXPECT_TRUE(arena->node_arena(0)->has_free_slot());

    // Not located → false.
    EXPECT_FALSE(arena->evict_key(k));
}

TEST(PinnedArenaMigrate, EvictLeavesRecordAdoptable) {
    // evict() must NOT stamp the record: the bytes are intact, so the ACTIVE
    // record stays truthful and try_adopt can restore the key ("free cell
    // still holds it") — the migrator's victim-selection failure recovery.
    lm::NumaManager numa(two_node_hw());
    auto arena = make_arena(numa, 2);
    std::vector<lm::ArenaCacheNodeGeom> geom{{0, 2}, {1, 2}};
    auto seg = lm::ArenaMetaSegment::create(lm::ArenaCache::required_bytes(geom));
    lm::ArenaCache cache(seg.base(), seg.bytes());
    ASSERT_TRUE(cache.format(0xA, 0xB, geom));
    cache.set_file_identities(idents(8));
    arena->set_cache(&cache);

    const auto k = key(0, 1);
    fill_on_node(*arena, k, 0, 0xAA);
    ASSERT_TRUE(arena->evict_key(k));
    EXPECT_TRUE(cache.lookup(k).has_value());  // record intact after evict

    void* restored = arena->try_adopt(k);
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(static_cast<unsigned char*>(restored)[0], 0xAA);
    EXPECT_EQ(arena->location_node(k), 0);
}

// ── migrate_begin / migrate_commit happy path ────────────────────────────────

TEST(PinnedArenaMigrate, BeginCommitMovesKeyAcrossNodes) {
    lm::NumaManager numa(two_node_hw());
    auto arena = make_arena(numa, 2);
    const auto hot = key(0, 1);
    const auto cold = key(0, 2);
    fill_on_node(*arena, hot, 0, 0x11);
    // Target node full of cold keys: the migrator evicts a victim first.
    fill_on_node(*arena, cold, 1, 0x22);
    fill_on_node(*arena, key(0, 3), 1, 0x33);
    ASSERT_FALSE(arena->node_arena(1)->has_free_slot());

    // No free slot on the target → begin refuses (extend-only).
    EXPECT_EQ(arena->migrate_begin(hot, 1), nullptr);

    ASSERT_TRUE(arena->evict_key(cold));  // victim demotion (frees a slot)
    void* dst = arena->migrate_begin(hot, 1);
    ASSERT_NE(dst, nullptr);

    // Mid-window: the key still resolves READY at the SOURCE (old bytes),
    // and the source slot is pinned (a reserve storm cannot evict it).
    void* src_ptr = arena->resolve(hot);
    ASSERT_NE(src_ptr, nullptr);
    EXPECT_EQ(static_cast<unsigned char*>(src_ptr)[0], 0x11);
    EXPECT_EQ(arena->location_node(hot), 0);
    EXPECT_GE(arena->node_arena(0)->inflight(hot), 1);

    // Async fill lands the bytes, then commit flips residency.
    std::memset(dst, 0x11, kSlot);
    EXPECT_EQ(arena->migrate_commit(hot, 1),
              lm::PinnedExpertArena::MigrateCommit::kCommitted);
    EXPECT_EQ(arena->location_node(hot), 1);
    ASSERT_TRUE(arena->is_ready(hot));
    EXPECT_EQ(arena->resolve(hot), dst);
    EXPECT_EQ(static_cast<unsigned char*>(arena->resolve(hot))[0], 0x11);
    // Source slot freed and unpinned; source arena has a free slot for the
    // victim's reload.
    EXPECT_TRUE(arena->node_arena(0)->has_free_slot());
    EXPECT_EQ(arena->node_arena(0)->inflight(hot), 0);
    EXPECT_EQ(arena->node_arena(1)->inflight(hot), 0);
}

TEST(PinnedArenaMigrate, CommitDefersWhileDemandH2dHoldsSource) {
    lm::NumaManager numa(two_node_hw());
    auto arena = make_arena(numa, 2);
    const auto hot = key(0, 1);
    fill_on_node(*arena, hot, 0, 0x11);

    void* dst = arena->migrate_begin(hot, 1);
    ASSERT_NE(dst, nullptr);
    std::memset(dst, 0x11, kSlot);

    // A demand fetch pinned the source mid-migration: commit must retry
    // (the DMA is still reading the source slot).
    arena->acquire_inflight(hot);  // routes via location_ → source arena
    EXPECT_EQ(arena->migrate_commit(hot, 1),
              lm::PinnedExpertArena::MigrateCommit::kRetry);
    EXPECT_EQ(arena->location_node(hot), 0);  // unflipped

    arena->release_inflight(hot);  // demand H2D completed
    EXPECT_EQ(arena->migrate_commit(hot, 1),
              lm::PinnedExpertArena::MigrateCommit::kCommitted);
    EXPECT_EQ(arena->location_node(hot), 1);
}

TEST(PinnedArenaMigrate, AbortRestoresSourceUntouched) {
    lm::NumaManager numa(two_node_hw());
    auto arena = make_arena(numa, 2);
    const auto hot = key(0, 1);
    fill_on_node(*arena, hot, 0, 0x11);

    void* dst = arena->migrate_begin(hot, 1);
    ASSERT_NE(dst, nullptr);
    arena->migrate_abort(hot, 1);

    // Source exactly as before; target slot free again; pin released.
    EXPECT_EQ(arena->location_node(hot), 0);
    ASSERT_TRUE(arena->is_ready(hot));
    EXPECT_EQ(static_cast<unsigned char*>(arena->resolve(hot))[0], 0x11);
    EXPECT_EQ(arena->node_arena(0)->inflight(hot), 0);
    EXPECT_FALSE(arena->node_arena(1)->resident(hot));
    EXPECT_TRUE(arena->node_arena(1)->has_free_slot());
    EXPECT_TRUE(arena->evict_key(hot));  // no dangling pin
}

// ── ArenaCache record discipline across a migration ──────────────────────────

TEST(PinnedArenaMigrate, RecordsAdoptableAtEveryInstant) {
    lm::NumaManager numa(two_node_hw());
    auto arena = make_arena(numa, 2);
    std::vector<lm::ArenaCacheNodeGeom> geom{{0, 2}, {1, 2}};
    auto seg = lm::ArenaMetaSegment::create(lm::ArenaCache::required_bytes(geom));
    lm::ArenaCache cache(seg.base(), seg.bytes());
    ASSERT_TRUE(cache.format(0xA, 0xB, geom));
    cache.set_file_identities(idents(8));
    arena->set_cache(&cache);

    const auto hot = key(2, 1);
    const auto cold = key(3, 2);
    fill_on_node(*arena, hot, 0, 0x11);
    fill_on_node(*arena, cold, 1, 0x22);

    auto adoptable_count = [&](bool expect_hot) {
        // A post-"crash" boot scan. scan_adoptable is DESTRUCTIVE by design
        // (stale/dup/interrupted records reset to EMPTY — it runs once at
        // attach), so inspect a byte-copy of the segment, never the live one.
        std::vector<unsigned char> snap(
            static_cast<unsigned char*>(seg.base()),
            static_cast<unsigned char*>(seg.base()) + seg.bytes());
        lm::ArenaCache scan(snap.data(), snap.size());
        EXPECT_TRUE(scan.validate(0xA, 0xB, geom));
        scan.set_file_identities(idents(8));
        size_t hot_hits = 0, total = 0;
        scan.scan_adoptable(16, 8, [&](int, size_t, lm::ExpertKey k) {
            ++total;
            if (k == hot) ++hot_hits;
        });
        EXPECT_EQ(hot_hits, expect_hot ? 1u : 0u)
            << "hot key must adopt from exactly " << (expect_hot ? 1 : 0)
            << " slot(s)";
        return total;
    };

    ASSERT_EQ(adoptable_count(true), 2u);    // baseline: hot + cold
    ASSERT_TRUE(arena->evict_key(cold));     // victim out — record INTACT
    ASSERT_EQ(adoptable_count(true), 2u);    // crash here: nothing lost
    void* dst = arena->migrate_begin(hot, 1);
    ASSERT_NE(dst, nullptr);
    // Target slot reused the victim's slot: its record died (EMPTY→LOADING),
    // hot's source record is the single truthful one.
    EXPECT_EQ(adoptable_count(true), 1u);    // crash here: hot @ source only
    std::memset(dst, 0x11, kSlot);
    ASSERT_EQ(arena->migrate_commit(hot, 1),
              lm::PinnedExpertArena::MigrateCommit::kCommitted);
    // Post-commit: hot ACTIVE at target; the stale source record (identical
    // bytes) transiently coexists — the boot scan dedups to ONE adoption.
    EXPECT_EQ(adoptable_count(true), 1u);
    // The freed source slot re-fills (the victim's reload): its reserve
    // stamps the stale hot record EMPTY before the first byte lands.
    void* back = arena->reserve_on_node(cold, 0);
    ASSERT_NE(back, nullptr);
    arena->mark_loading(cold);
    EXPECT_EQ(adoptable_count(true), 1u);    // crash mid-reload: hot only
    std::memset(back, 0x22, kSlot);
    arena->mark_ready(cold);
    EXPECT_EQ(adoptable_count(true), 2u);    // converged: hot@1, cold@0
    EXPECT_EQ(arena->location_node(hot), 1);
    EXPECT_EQ(arena->location_node(cold), 0);
}

// ── Stress: concurrent fetch-vs-migrate race ─────────────────────────────────
//
// Faithful miniature of the production concurrency model (INV-ELM-1 /
// INV-ARENA-MIGRATE): ALL slot-state mutation (reserve/evict/migrate/
// acquire/release) happens on ONE thread — the "daemon" (this test's main
// thread). The genuinely concurrent actors are:
//   - a DMA-emulator thread: for every pinned demand fetch it repeatedly
//     verifies the slot bytes through the RAW POINTER captured at enqueue
//     time (exactly like the transfer engine holds a bare `src`), catching
//     any migration that overwrites a slot an H2D is still reading;
//   - a loader-worker thread: writes bytes into `loading` slots off-thread
//     (the J-1 ArenaLoader contract), catching any reuse of a mid-fill slot.
// The daemon randomly interleaves demand fetches (pin → DMA job → release on
// completion), migrations (victim evict → begin → async fill → commit with
// kRetry deferral → victim reload), and readers.

namespace {

struct DmaJob {       // captured-at-enqueue H2D: reader verifies via raw ptr
    lm::ExpertKey key;
    const unsigned char* ptr;
    unsigned char tag;
};
struct FillJob {      // async loader fill into a loading slot
    uint64_t id;
    unsigned char* ptr;
    unsigned char tag;
};

}  // namespace

TEST(PinnedArenaMigrate, StressFetchVsMigrateRace) {
    lm::NumaManager numa(two_node_hw());
    constexpr size_t kPerNode = 4;
    auto arena = make_arena(numa, kPerNode);

    std::vector<lm::ExpertKey> keys;
    for (uint16_t e = 0; e < 2 * kPerNode; ++e) {
        const auto k = key(0, e);
        const int node = e < kPerNode ? 0 : 1;
        fill_on_node(*arena, k, node, static_cast<unsigned char>(0x10 + e));
        keys.push_back(k);
    }

    std::atomic<bool> stop{false};
    std::atomic<size_t> dma_errors{0};

    // DMA emulator: verify pinned slots through captured pointers.
    std::mutex dma_mu;
    std::vector<DmaJob> dma_jobs;      // submitted by daemon
    std::vector<lm::ExpertKey> dma_done;  // completions daemon reaps
    std::thread dma_thread([&] {
        while (!stop.load(std::memory_order_acquire)) {
            std::vector<DmaJob> batch;
            {
                std::lock_guard<std::mutex> g(dma_mu);
                batch.swap(dma_jobs);
            }
            for (const auto& j : batch) {
                // Read the whole slot a few times — the DMA window.
                for (int pass = 0; pass < 3; ++pass)
                    for (size_t i = 0; i < kSlot; i += 4096)
                        if (j.ptr[i] != j.tag)
                            dma_errors.fetch_add(1, std::memory_order_relaxed);
                std::lock_guard<std::mutex> g(dma_mu);
                dma_done.push_back(j.key);
            }
            if (batch.empty()) std::this_thread::yield();
        }
    });

    // Loader worker: fills loading slots off-thread (J-1).
    std::mutex fill_mu;
    std::vector<FillJob> fill_jobs;
    std::vector<uint64_t> fill_done;
    std::thread fill_thread([&] {
        while (!stop.load(std::memory_order_acquire)) {
            std::vector<FillJob> batch;
            {
                std::lock_guard<std::mutex> g(fill_mu);
                batch.swap(fill_jobs);
            }
            for (const auto& j : batch) {
                std::memset(j.ptr, j.tag, kSlot);
                std::lock_guard<std::mutex> g(fill_mu);
                fill_done.push_back(j.id);
            }
            if (batch.empty()) std::this_thread::yield();
        }
    });

    auto tag_of = [&](lm::ExpertKey k) {
        return static_cast<unsigned char>(0x10 + k.expert_idx);
    };

    // Daemon-side migration state machine (at most one active, like the
    // production migrator's serial pipeline).
    enum class MigState { kIdle, kFilling, kCommitting, kReloadingVictim };
    MigState mig = MigState::kIdle;
    lm::ExpertKey mig_key{}, mig_victim{};
    int mig_src = -1, mig_dst = -1;
    uint64_t mig_fill_id = 0, next_fill_id = 1;
    std::unordered_map<uint64_t, bool> fills_landed;

    std::mt19937 rng(42);
    size_t committed = 0, retried = 0;
    int outstanding_pins = 0;

    for (int iter = 0; iter < 60000 || mig != MigState::kIdle ||
                       outstanding_pins > 0; ++iter) {
        ASSERT_LT(iter, 500000) << "livelock: migration never drained";
        // 1. Reap DMA completions → release pins (daemon thread, like ELM
        //    poll dropping host_buf_ref).
        {
            std::vector<lm::ExpertKey> done;
            {
                std::lock_guard<std::mutex> g(dma_mu);
                done.swap(dma_done);
            }
            for (auto k : done) {
                arena->release_inflight(k);
                --outstanding_pins;
            }
        }
        // 2. Reap loader completions.
        {
            std::vector<uint64_t> done;
            {
                std::lock_guard<std::mutex> g(fill_mu);
                done.swap(fill_done);
            }
            for (auto id : done) fills_landed[id] = true;
        }
        // 3. Advance the migration state machine.
        switch (mig) {
            case MigState::kIdle:
                break;
            case MigState::kFilling:
                if (fills_landed.count(mig_fill_id)) {
                    fills_landed.erase(mig_fill_id);
                    mig = MigState::kCommitting;
                }
                break;
            case MigState::kCommitting: {
                const auto r = arena->migrate_commit(mig_key, mig_dst);
                if (r == lm::PinnedExpertArena::MigrateCommit::kCommitted) {
                    ++committed;
                    // Victim reload into the freed source slot, async.
                    void* back = arena->reserve_on_node(mig_victim, mig_src);
                    ASSERT_NE(back, nullptr);
                    arena->mark_loading(mig_victim);
                    mig_fill_id = next_fill_id++;
                    {
                        std::lock_guard<std::mutex> g(fill_mu);
                        fill_jobs.push_back({mig_fill_id,
                                             static_cast<unsigned char*>(back),
                                             tag_of(mig_victim)});
                    }
                    mig = MigState::kReloadingVictim;
                } else {
                    ++retried;  // demand H2D still pins the source — defer
                }
                break;
            }
            case MigState::kReloadingVictim:
                if (fills_landed.count(mig_fill_id)) {
                    fills_landed.erase(mig_fill_id);
                    arena->mark_ready(mig_victim);
                    mig = MigState::kIdle;
                }
                break;
        }
        // 4. Random daemon work (stop injecting near the end to drain).
        if (iter >= 60000) {
            std::this_thread::yield();
            continue;
        }
        const auto k = keys[rng() % keys.size()];
        switch (rng() % 3) {
            case 0: {  // demand fetch: resolve + pin + captured-ptr DMA job
                void* p = arena->resolve(k);
                if (!p) {  // mid-victim-reload — the production miss path
                    break;
                }
                ASSERT_EQ(static_cast<unsigned char*>(p)[0], tag_of(k));
                arena->acquire_inflight(k);
                ++outstanding_pins;
                std::lock_guard<std::mutex> g(dma_mu);
                dma_jobs.push_back(
                    {k, static_cast<const unsigned char*>(p), tag_of(k)});
                break;
            }
            case 1: {  // start a migration if idle
                if (mig != MigState::kIdle) break;
                const int src = arena->location_node(k);
                if (src < 0) break;              // mid-victim-reload
                const int dst_node = src == 0 ? 1 : 0;
                // Victim: an unpinned co-resident on the target node.
                lm::ExpertKey victim = lm::kNoEvictedKey;
                for (auto v : keys)
                    if (arena->location_node(v) == dst_node &&
                        arena->is_ready(v) &&
                        arena->node_arena(dst_node)->inflight(v) == 0) {
                        victim = v;
                        break;
                    }
                if (victim == lm::kNoEvictedKey) break;
                if (!arena->evict_key(victim)) break;
                void* dst = arena->migrate_begin(k, dst_node);
                if (!dst) {  // k not migratable — restore victim, skip
                    void* back = arena->reserve_on_node(victim, dst_node);
                    ASSERT_NE(back, nullptr);
                    std::memset(back, tag_of(victim), kSlot);
                    arena->mark_ready(victim);
                    break;
                }
                mig_key = k;
                mig_victim = victim;
                mig_src = src;
                mig_dst = dst_node;
                mig_fill_id = next_fill_id++;
                {
                    std::lock_guard<std::mutex> g(fill_mu);
                    fill_jobs.push_back({mig_fill_id,
                                         static_cast<unsigned char*>(dst),
                                         tag_of(k)});
                }
                mig = MigState::kFilling;
                break;
            }
            case 2: {  // reader
                if (void* p = arena->resolve(k))
                    ASSERT_EQ(static_cast<unsigned char*>(p)[0], tag_of(k));
                break;
            }
        }
    }
    stop.store(true, std::memory_order_release);
    dma_thread.join();
    fill_thread.join();

    EXPECT_EQ(dma_errors.load(), 0u)
        << "a migration overwrote bytes a pinned H2D was reading";
    EXPECT_GT(committed, 50u);
    (void)retried;  // deferral is timing-dependent here; covered
                    // deterministically by CommitDefersWhileDemandH2dHoldsSource
    size_t on0 = 0, on1 = 0;
    for (auto k : keys) {
        ASSERT_TRUE(arena->is_ready(k));
        ASSERT_EQ(static_cast<unsigned char*>(arena->resolve(k))[0], tag_of(k));
        const int n = arena->location_node(k);
        ASSERT_TRUE(n == 0 || n == 1);
        (n == 0 ? on0 : on1)++;
        EXPECT_EQ(arena->node_arena(n)->inflight(k), 0)
            << "leaked pin on (" << k.layer_idx << "," << k.expert_idx << ")";
    }
    EXPECT_EQ(on0, kPerNode);
    EXPECT_EQ(on1, kPerNode);
}
