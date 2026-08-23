// Unit tests for ArenaMigrator (M3b online self-tuning arena placement):
// EMA signal + decay, hysteresis-gated candidate selection, budget cap,
// free-slot-first promotion, and the end-to-end promote/demote pipeline
// against a REAL PinnedExpertArena + mock-prepacked PrepackedSource (bytes
// verified against the source after every move — placement must be
// performance-only).
//
// No CUDA needed (defer_registration=true).

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include "core/memory/arena_migrator.h"
#include "core/memory/numa_manager.h"
#include "core/memory/pinned_expert_arena.h"
#include "model/model_config.h"
#include "model/quantization/nvfp4.h"
#include "model/weight_pipeline/expert_prepacker.h"
#include "model/weight_pipeline/prepacked_source.h"
#include "weight_pipeline_test_helpers.h"

namespace fs = std::filesystem;
using namespace layerstorm::model;
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

lc::HardwareConfig two_node_hw() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 128;
    hw.gpus = {make_gpu(0, 0), make_gpu(1, 1)};
    return hw;
}

lc::PinHostExpertPoolSizingConfig big_sizing() {
    lc::PinHostExpertPoolSizingConfig s;
    s.mode = lc::HostPoolSizingMode::fraction_total;
    s.value = 0.85;
    return s;
}

lm::ExpertKey key(uint32_t layer, uint16_t expert) { return {layer, expert}; }

}  // namespace

class ArenaMigratorTest : public ::testing::Test,
                          public layerstorm::test::WeightPipelineHelpers {
protected:
    static constexpr int kLayers = 4, kFirstMoe = 2, kExperts = 4;
    static constexpr int kHidden = 64, kIntermediate = 32;

    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "layerstorm_arena_migrator_test";
        fs::create_directories(tmp_dir_);
        ExpertShape shape{kHidden, kIntermediate};
        auto cfg = make_config(kLayers, kExperts, kHidden, kIntermediate,
                               kFirstMoe);
        ModelConfig model_cfg(cfg);
        Nvfp4 quant;
        auto model = make_mock_model(kLayers, kExperts, shape, kFirstMoe,
                                     SafetensorsDtype::U8);
        auto out = tmp_dir_ / "src";
        auto result = prepack_experts(model, model_cfg, quant, cfg, out);
        ASSERT_TRUE(result.error.empty()) << result.error;
        quant_ = std::make_unique<Nvfp4>();
        src_ = std::make_unique<PrepackedSource>(out, *quant_);
        slot_ = static_cast<size_t>(src_->slot_size_bytes());
        ASSERT_GT(slot_, 0u);
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmp_dir_, ec);
    }

    // Arena with `per_node` slots on each of nodes {0, 1}; slot size = the
    // prepacked stride so the migrator's async fills land exactly.
    std::unique_ptr<lm::PinnedExpertArena> make_arena(size_t per_node) {
        numa_ = std::make_unique<lm::NumaManager>(two_node_hw());
        return std::make_unique<lm::PinnedExpertArena>(
            *numa_, slot_, slot_ * per_node * 2, big_sizing(),
            lc::CrossNodeSpillConfig{}, /*extra_scratch_bytes=*/0,
            /*defer_registration=*/true, nullptr);
    }

    // Synchronously place `k` on `node` with its REAL prepacked bytes.
    void place(lm::PinnedExpertArena& a, lm::ExpertKey k, int node) {
        void* s = a.reserve_on_node(k, node);
        ASSERT_NE(s, nullptr);
        ASSERT_TRUE(src_->load_into(k, s));
        a.mark_ready(k);
    }

    void expect_bytes_match(lm::PinnedExpertArena& a, lm::ExpertKey k) {
        void* p = a.resolve(k);
        ASSERT_NE(p, nullptr);
        const void* sp = src_->resolve(k);
        ASSERT_NE(sp, nullptr);
        EXPECT_EQ(std::memcmp(p, sp, slot_), 0)
            << "expert (" << k.layer_idx << "," << k.expert_idx
            << ") bytes diverged from the prepacked source";
    }

    // Pump the migrator until it goes idle with an empty plan (bounded).
    void pump(lm::ArenaMigrator& m, int max_ms = 5000) {
        const auto t0 = std::chrono::steady_clock::now();
        int settle = 0;
        // The idle fast path only does real work every 64th tick — a true
        // "drained" verdict needs several full idle passes (>3*64 ticks).
        while (settle < 256) {
            m.tick();
            if (!m.active())
                ++settle;
            else
                settle = 0;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            ASSERT_LT(std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0)
                          .count(),
                      max_ms / 1000.0)
                << "migrator did not settle";
        }
    }

    lm::ArenaMigrator::Config fast_cfg() {
        lm::ArenaMigrator::Config c;
        c.budget_mbps = 1e6;        // effectively unlimited
        c.half_life_s = 3600.0;     // no decay during the test
        c.refresh_period_s = 0.0;   // rescan whenever idle
        c.min_total = 1.0;          // warm-up gate off (tested separately)
        return c;
    }

    fs::path tmp_dir_;
    std::unique_ptr<Nvfp4> quant_;
    std::unique_ptr<PrepackedSource> src_;
    std::unique_ptr<lm::NumaManager> numa_;
    size_t slot_ = 0;
};

TEST_F(ArenaMigratorTest, ConfigFromEnvParsesAndRejectsGarbage) {
    ::setenv("LS_ARENA_MIGRATE_MBPS", "123.5", 1);
    ::setenv("LS_ARENA_MIGRATE_HALF_LIFE_S", "7", 1);
    ::setenv("LS_ARENA_MIGRATE_RATIO", "-3", 1);    // invalid → default
    ::setenv("LS_ARENA_MIGRATE_MARGIN", "junk", 1); // invalid → default
    auto c = lm::ArenaMigrator::Config::from_env();
    EXPECT_DOUBLE_EQ(c.budget_mbps, 123.5);
    EXPECT_DOUBLE_EQ(c.half_life_s, 7.0);
    EXPECT_DOUBLE_EQ(c.ratio, 3.0);
    EXPECT_DOUBLE_EQ(c.margin, 10.0);
    ::unsetenv("LS_ARENA_MIGRATE_MBPS");
    ::unsetenv("LS_ARENA_MIGRATE_HALF_LIFE_S");
    ::unsetenv("LS_ARENA_MIGRATE_RATIO");
    ::unsetenv("LS_ARENA_MIGRATE_MARGIN");
}

TEST_F(ArenaMigratorTest, PromotesHotDemotesColdViaSpareFullFit) {
    // Full-fit: 4 keys per node, node 1 plays "HBM". The first swap must
    // mint the floating spare by evicting the globally coldest DDR key
    // (deterministic: f=0, key asc → ddr_keys[0]); afterwards NO key ever
    // goes non-resident (the byte-identity-critical property).
    auto arena = make_arena(4);
    std::vector<lm::ExpertKey> ddr_keys, hbm_keys;
    for (uint16_t e = 0; e < kExperts; ++e) {
        place(*arena, key(kFirstMoe, e), 0);
        ddr_keys.push_back(key(kFirstMoe, e));
        place(*arena, key(kFirstMoe + 1, e), 1);
        hbm_keys.push_back(key(kFirstMoe + 1, e));
    }
    lm::ArenaMigrator mig(fast_cfg(), *arena, /*hbm_nodes=*/{1}, kLayers,
                          kExperts, slot_);

    // Two DDR keys are hot (fetch often); everything else silent. Give the
    // spare-donor candidate ddr_keys[0] zero signal so the mint choice is
    // deterministic; ddr_keys[2] gets a sub-margin trickle.
    for (int i = 0; i < 100; ++i) {
        mig.on_demand_fetch(ddr_keys[1]);
        mig.on_demand_fetch(ddr_keys[3]);
    }
    for (int i = 0; i < 3; ++i) mig.on_demand_fetch(ddr_keys[2]);

    // Pump while asserting the zero-non-residency contract: every key
    // except the spare donor resolves READY at every observation point.
    const auto t0 = std::chrono::steady_clock::now();
    int settle = 0;
    while (settle < 256) {
        mig.tick();
        for (auto k : ddr_keys)
            if (k != ddr_keys[0])
                ASSERT_TRUE(arena->is_ready(k))
                    << "key (" << k.layer_idx << "," << k.expert_idx
                    << ") went non-resident mid-migration";
        for (auto k : hbm_keys)
            ASSERT_TRUE(arena->is_ready(k));
        if (!mig.active())
            ++settle;
        else
            settle = 0;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        ASSERT_LT(std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0).count(), 10.0);
    }

    // The hot keys moved to the HBM node; two cold HBM victims demoted out
    // through the spare; the donor was evicted exactly once.
    EXPECT_EQ(arena->location_node(ddr_keys[1]), 1);
    EXPECT_EQ(arena->location_node(ddr_keys[3]), 1);
    EXPECT_EQ(mig.stats().committed, 2u);
    EXPECT_EQ(mig.stats().demotions, 2u);
    EXPECT_EQ(mig.stats().spare_steals, 1u);
    EXPECT_EQ(mig.stats().aborted, 0u);
    EXPECT_EQ(arena->location_node(ddr_keys[0]), -1);  // spare donor evicted
    // Bytes of every RESIDENT key identical to the prepacked source —
    // placement is performance-only.
    size_t on0 = 0, on1 = 0;
    for (auto k : ddr_keys) {
        if (k == ddr_keys[0]) continue;
        (arena->location_node(k) == 0 ? on0 : on1)++;
        expect_bytes_match(*arena, k);
    }
    for (auto k : hbm_keys) {
        (arena->location_node(k) == 0 ? on0 : on1)++;
        expect_bytes_match(*arena, k);
    }
    EXPECT_EQ(on0, 3u);  // 2 demoted victims + ddr_keys[2]; +1 floating spare
    EXPECT_EQ(on1, 4u);  // HBM stays full-fit
    EXPECT_TRUE(arena->node_arena(0)->has_free_slot());  // the spare
    // No leaked pins anywhere.
    for (auto k : ddr_keys) {
        if (k == ddr_keys[0]) continue;
        EXPECT_EQ(arena->locate(k)->inflight(k), 0);
    }
    for (auto k : hbm_keys) EXPECT_EQ(arena->locate(k)->inflight(k), 0);
}

TEST_F(ArenaMigratorTest, FreeHbmSlotsFillBeforeAnyEviction) {
    auto arena = make_arena(4);
    // Node 1 has 4 slots but only 2 residents → 2 free slots.
    for (uint16_t e = 0; e < kExperts; ++e) place(*arena, key(kFirstMoe, e), 0);
    place(*arena, key(kFirstMoe + 1, 0), 1);
    place(*arena, key(kFirstMoe + 1, 1), 1);

    lm::ArenaMigrator mig(fast_cfg(), *arena, {1}, kLayers, kExperts,
                          slot_);
    for (int i = 0; i < 50; ++i) mig.on_demand_fetch(key(kFirstMoe, 2));
    pump(mig);

    EXPECT_EQ(arena->location_node(key(kFirstMoe, 2)), 1);
    EXPECT_EQ(mig.stats().committed, 1u);
    EXPECT_EQ(mig.stats().demotions, 0u);       // free slot: no demote leg
    EXPECT_EQ(mig.stats().spare_steals, 0u);    // and no spare needed
    expect_bytes_match(*arena, key(kFirstMoe, 2));
}

TEST_F(ArenaMigratorTest, HysteresisBlocksMarginalAndSymmetricSwaps) {
    auto arena = make_arena(2);
    place(*arena, key(kFirstMoe, 0), 0);
    place(*arena, key(kFirstMoe, 1), 0);
    place(*arena, key(kFirstMoe + 1, 0), 1);
    place(*arena, key(kFirstMoe + 1, 1), 1);

    auto cfg = fast_cfg();
    cfg.ratio = 1.5;
    cfg.margin = 4.0;
    lm::ArenaMigrator mig(cfg, *arena, {1}, kLayers, kExperts, slot_);

    // Symmetric load: DDR key f=20, HBM keys f=20 → 20 <= 1.5*20+4 → no move.
    for (int i = 0; i < 20; ++i) {
        mig.on_demand_fetch(key(kFirstMoe, 0));
        mig.on_demand_fetch(key(kFirstMoe + 1, 0));
        mig.on_demand_fetch(key(kFirstMoe + 1, 1));
    }
    pump(mig);
    EXPECT_EQ(mig.stats().committed, 0u);
    EXPECT_EQ(arena->location_node(key(kFirstMoe, 0)), 0);

    // Sub-margin signal (f=3 < margin=4) must not move either.
    for (int i = 0; i < 3; ++i) mig.on_demand_fetch(key(kFirstMoe, 1));
    pump(mig);
    EXPECT_EQ(mig.stats().committed, 0u);
}

TEST_F(ArenaMigratorTest, WarmUpGateHoldsMigrationsUntilSignalAccumulates) {
    auto arena = make_arena(2);
    place(*arena, key(kFirstMoe, 0), 0);
    place(*arena, key(kFirstMoe + 1, 0), 1);

    auto cfg = fast_cfg();
    cfg.min_total = 100.0;
    lm::ArenaMigrator mig(cfg, *arena, {1}, kLayers, kExperts, slot_);

    // 50 fetches < gate: demonstrably hot, but the plan must stay empty.
    for (int i = 0; i < 50; ++i) mig.on_demand_fetch(key(kFirstMoe, 0));
    pump(mig);
    EXPECT_EQ(mig.stats().committed, 0u);
    EXPECT_EQ(arena->location_node(key(kFirstMoe, 0)), 0);

    // Crossing the gate releases the (same) plan.
    for (int i = 0; i < 60; ++i) mig.on_demand_fetch(key(kFirstMoe, 0));
    pump(mig);
    EXPECT_EQ(mig.stats().committed, 1u);
    EXPECT_EQ(arena->location_node(key(kFirstMoe, 0)), 1);
}

TEST_F(ArenaMigratorTest, BudgetGatesMigrationStart) {
    auto arena = make_arena(2);
    place(*arena, key(kFirstMoe, 0), 0);
    place(*arena, key(kFirstMoe + 1, 0), 1);

    auto cfg = fast_cfg();
    // ~0 budget: the token bucket cannot reach 2*slot bytes in test time.
    cfg.budget_mbps = 1e-6;
    lm::ArenaMigrator mig(cfg, *arena, {1}, kLayers, kExperts, slot_);
    for (int i = 0; i < 100; ++i) mig.on_demand_fetch(key(kFirstMoe, 0));
    for (int i = 0; i < 2000; ++i) mig.tick();
    EXPECT_EQ(mig.stats().committed, 0u);
    EXPECT_EQ(arena->location_node(key(kFirstMoe, 0)), 0);
}

TEST_F(ArenaMigratorTest, DecayForgetsOldBursts) {
    auto arena = make_arena(2);
    place(*arena, key(kFirstMoe, 0), 0);
    auto cfg = fast_cfg();
    cfg.half_life_s = 0.02;  // 20 ms half-life → a burst fades fast
    lm::ArenaMigrator mig(cfg, *arena, {1}, kLayers, kExperts, slot_);
    for (int i = 0; i < 64; ++i) mig.on_demand_fetch(key(kFirstMoe, 0));
    EXPECT_FLOAT_EQ(mig.freq(key(kFirstMoe, 0)), 64.0f);
    const auto t0 = std::chrono::steady_clock::now();
    while (mig.freq(key(kFirstMoe, 0)) > 1.0f) {
        mig.tick();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ASSERT_LT(std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0).count(), 5.0)
            << "signal never decayed";
    }
    EXPECT_LE(mig.freq(key(kFirstMoe, 0)), 1.0f);
}

TEST_F(ArenaMigratorTest, AdoptedEmaSkipsWarmUpGateAndRanksImmediately) {
    // TD-ARENA-MIGRATE-EMA-PERSIST: a persisted (mature) EMA snapshot adopted
    // at boot must re-arm planning immediately — no re-learning window, no
    // fresh churn. Mirror of WarmUpGateHoldsMigrationsUntilSignalAccumulates
    // with the signal arriving via adopt_ema instead of live fetches.
    auto arena = make_arena(2);
    place(*arena, key(kFirstMoe, 0), 0);
    place(*arena, key(kFirstMoe + 1, 0), 1);

    auto cfg = fast_cfg();
    cfg.min_total = 100.0;
    lm::ArenaMigrator mig(cfg, *arena, {1}, kLayers, kExperts, slot_);

    // Snapshot from a "previous boot": DDR key hot, everything else cold;
    // fetches_seen far past the gate.
    std::vector<float> snap(static_cast<size_t>(kLayers) * kExperts, 0.0f);
    snap[static_cast<size_t>(kFirstMoe) * kExperts + 0] = 500.0f;
    mig.adopt_ema(snap.data(), snap.size(), /*fetches_seen=*/5000.0);
    EXPECT_FLOAT_EQ(mig.freq(key(kFirstMoe, 0)), 500.0f);
    EXPECT_DOUBLE_EQ(mig.fetches_seen(), 5000.0);

    pump(mig);  // NOT a single live fetch — the adopted signal alone plans
    EXPECT_EQ(mig.stats().committed, 1u);
    EXPECT_EQ(arena->location_node(key(kFirstMoe, 0)), 1);
    expect_bytes_match(*arena, key(kFirstMoe, 0));

    // A shape-mismatched snapshot is ignored (start cold, never wrong).
    lm::ArenaMigrator mig2(cfg, *arena, {1}, kLayers, kExperts, slot_);
    std::vector<float> bad(snap.size() + 1, 1.0f);
    mig2.adopt_ema(bad.data(), bad.size(), 5000.0);
    EXPECT_DOUBLE_EQ(mig2.fetches_seen(), 0.0);
}

TEST_F(ArenaMigratorTest, PersistSinkFiresOnNewSignalAndStaysQuietWhenIdle) {
    auto arena = make_arena(2);
    place(*arena, key(kFirstMoe, 0), 0);
    auto cfg = fast_cfg();
    lm::ArenaMigrator mig(cfg, *arena, {1}, kLayers, kExperts, slot_);
    int calls = 0;
    mig.set_persist_sink([&] { ++calls; }, /*period_s=*/0.01);

    // No signal yet: the sink must stay quiet (nothing new to persist).
    for (int i = 0; i < 200; ++i) {
        mig.tick();
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    EXPECT_EQ(calls, 0);

    // New fetches → the sink fires (rate-limited by period_s).
    for (int i = 0; i < 20; ++i) mig.on_demand_fetch(key(kFirstMoe, 0));
    const auto t0 = std::chrono::steady_clock::now();
    while (calls == 0) {
        mig.tick();
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        ASSERT_LT(std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0).count(), 5.0)
            << "persist sink never fired";
    }
    const int after_first = calls;
    // With no FURTHER fetches the sink stays quiet again.
    for (int i = 0; i < 200; ++i) {
        mig.tick();
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    EXPECT_EQ(calls, after_first);
}
