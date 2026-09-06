// 44z stage 4: the KV <-> expert zone REBALANCER policy.
//
// What these tests are actually about: the HYSTERESIS contract and the
// RECLAIM RATCHET, not the two mechanisms underneath (those are covered by
// page_allocator_expert_zone_test.cpp and the expert-cache elastic-zone
// suite). Specifically —
//   * a grant may only happen ABOVE the high-water mark, so a grant can
//     never immediately trigger the reclaim it just caused;
//   * a reclaim starts at the LOW-water mark, in the BACKGROUND, before
//     anybody blocks — the band between the marks is the anti-oscillation
//     device;
//   * capacity moves ONLY at the end of the ratchet: begin_drain -> evict as
//     ready+unlocked -> MoE quiesce -> h2d + FFN stream barriers -> release.
//     Every test that reclaims asserts the ORDER, because each earlier step
//     is reversible bookkeeping and only the last one hands KV the bytes;
//   * an admission refused while grants are outstanding is a POLICY FAILURE
//     and is counted as one.
//
// CPU-only: real PageAllocator + real ExpertCache over NullDeviceBackend
// regions (the cache never dereferences slot memory), and the whole device
// stream/event surface injected as counting stubs.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "config/config_parser.h"
#include "config/config_resolver.h"
#include "core/device_backend.h"
#include "core/gpu_ref.h"
#include "core/memory/expert_cache.h"
#include "core/memory/expert_zone_math.h"
#include "core/memory/page_allocator.h"
#include "core/memory/vram_allocator.h"
#include "core/null_device_backend.h"
#include "daemon/expert_zone_rebalancer.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/fp8.h"

namespace lc = layerstorm::config;
namespace lcomp = layerstorm::compute;
namespace ld = layerstorm::daemon;
namespace lmem = layerstorm::memory;
namespace lmod = layerstorm::model;

namespace {

// ── Fixture ────────────────────────────────────────────────────────────────
// Tiny glm5_next shape WITH an indexer pool, which is what makes the kMain
// region SLABBED (a grant is a whole-slab run — without slabs there is no
// mechanism to exercise). Same shape the page-allocator expert-zone suite
// uses; rebuilt locally because all unit *_test.cpp share one executable and
// each suite's helpers live in its own anonymous namespace.

lc::Config rb_config(int num_gpus) {
    auto j = nlohmann::json{
        {"model", {
            {"architecture",            "deepseek_v3"},
            {"weights_path",            "/data/models/test/"},
            {"weights_format",          "safetensors"},
            {"num_hidden_layers",       6},
            {"hidden_size",             256},
            {"num_attention_heads",     4},
            {"num_key_value_heads",     4},
            {"intermediate_size",       512},
            {"n_routed_experts",        8},
            {"n_shared_experts",        1},
            {"num_experts_per_tok",     2},
            {"n_group",                 1},
            {"topk_group",              1},
            {"vocab_size",              1024},
            {"max_position_embeddings", 2048},
            {"kv_lora_rank",            0},
            {"q_lora_rank",             0},
            {"qk_rope_head_dim",        32},
            {"qk_nope_head_dim",        32},
            {"v_head_dim",              64},
            {"first_k_dense_replace",   1},
            {"moe_layer_freq",          1},
            {"index_topk",              0},
            {"num_nextn_predict_layers", 0},
            {"moe_intermediate_size",   128},
        }},
        {"quantization", {{"weights", "fp8_e4m3"},
                          {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"},
                          {"gating_compute", "fp32"}}},
        {"hardware", {{"system_ram_gb", 64}}},
        {"serving", {{"max_concurrent_requests", 2},
                     {"max_sequence_length", 2048}}},
        {"memory", {{"vram_safety_margin_gb", 0.1},
                    {"kv_cache", {{"page_growth_chunk_tokens", 0}}}}},
    };
    nlohmann::json gpus = nlohmann::json::array();
    nlohmann::json tp = nlohmann::json::array();
    for (int i = 0; i < num_gpus; ++i) {
        gpus.push_back({{"id", i}, {"type", "rtx5090"}, {"vram_gb", 1}});
        tp.push_back(i);
    }
    j["hardware"]["gpus"] = gpus;
    j["hardware"]["tp_array"] = tp;

    auto cfg = lc::parse_config(j);
    cfg.model.architecture = lc::Architecture::glm5_next;
    cfg.model.layer_types.clear();
    for (int l = 0; l < 6; ++l)
        cfg.model.layer_types.push_back(
            (l % 3 == 2) ? lc::LayerAttentionType::deepseek_sparse_attention
                         : lc::LayerAttentionType::linear_attention);
    lc::LinearAttnConfig la;
    la.num_heads = 4;
    la.head_dim = 32;
    la.short_conv_kernel_size = 4;
    la.gate_lower_bound = -5.0;
    cfg.model.linear_attn_config = la;
    // The indexer pool is what makes the shared region slabbed.
    cfg.model.index_topk = 64;
    cfg.model.index_n_heads = 2;
    cfg.model.index_head_dim = 32;
    cfg._internal_kda_state.mapped = true;
    if (num_gpus >= 2) cfg.parallelism.tensor_parallelism = num_gpus;
    return cfg;
}

struct RbCtx {
    lc::Config cfg;
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned;
    std::vector<lcomp::DeviceBackend*> ptrs;
    std::unique_ptr<lmem::VramAllocator> vram;
    std::unique_ptr<lmem::PageAllocator> pa;
    std::unique_ptr<lmem::ExpertCache> cache;
    int64_t expert_bytes = 0;
    std::vector<int> tp_gpus;
};

std::unique_ptr<RbCtx> build_ctx(lc::Config cfg, int num_gpus) {
    auto ctx = std::make_unique<RbCtx>();
    ctx->cfg = std::move(cfg);
    lmod::ModelConfig mcfg(ctx->cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, ctx->cfg, fp8);
    ctx->expert_bytes = reg.per_routed_expert_bytes();
    auto layout = lmem::compute_vram_layout(ctx->cfg, reg, mcfg);
    for (size_t i = 0; i < layout.gpus.size(); ++i) {
        lc::GpuRef gref{.position = static_cast<int>(i),
                        .id = layout.gpus[i].gpu_id};
        ctx->owned.push_back(lcomp::make_null_device_backend(gref));
        ctx->ptrs.push_back(ctx->owned.back().get());
    }
    ctx->vram = std::make_unique<lmem::VramAllocator>(std::move(layout),
                                                     ctx->ptrs);
    ctx->pa = std::make_unique<lmem::PageAllocator>(*ctx->vram, ctx->ptrs[0]);
    ctx->cache = std::make_unique<lmem::ExpertCache>(*ctx->vram, ctx->cfg,
                                                     ctx->expert_bytes);
    for (int i = 0; i < num_gpus; ++i) ctx->tp_gpus.push_back(i);
    return ctx;
}

/// Slots-per-grant ceiling used by every fixture below. Small on purpose:
/// the policy under test is WHEN a grant happens, not how big it can get.
/// It bounds the ASK — which feeds grant_slabs, the smallest admissible SLAB
/// count meeting that ask under the worst-case base pad — not the fill: on
/// this fixture an expert slot is far SMALLER than a slab, so the claimed
/// slabs hold more slots than the cap and the grant takes all of them
/// (capping the fill would strand claimed capacity as waste, breaking the
/// 44z §3 bound).
constexpr int kMaxSlotsPerGrant = 2;

/// Two-pass build: pass 1 learns the model's real slab/expert geometry, then
/// the kMain pool is capped to a slab count that is (a) big enough for a
/// two-slot grant to fit above the high-water mark and (b) small enough that
/// a test can walk free slabs down past the low-water mark cheaply. Deriving
/// the size from the geometry (instead of hardcoding pages) keeps the suite
/// correct if the model shape or the layout ever changes.
/// @param size_mult scales the pool. The mark-shape tests need TWO pools of
///   different sizes built from the SAME geometry, to show the absolute
///   marks do not move with the pool while a fraction would.
std::unique_ptr<RbCtx> make_ctx(int num_gpus = 1, int size_mult = 1) {
    auto probe = build_ctx(rb_config(num_gpus), num_gpus);
    const int pps = probe->pa->pages_per_slab();
    const int64_t slab_bytes = probe->pa->slab_bytes();
    EXPECT_GT(pps, 0) << "fixture model must produce a SLABBED kMain region";
    EXPECT_GT(slab_bytes, 0);
    if (pps <= 0 || slab_bytes <= 0) return probe;

    lmem::ExpertZoneGeometry geo{.expert_slot_bytes = probe->expert_bytes,
                                 .slab_bytes = slab_bytes,
                                 .max_waste = 0.10};
    // Size against grant_slabs, not slabs_for_slots: worst-case-pad
    // admissibility can push S well past the bare slot fit (on a 3-slots-per-
    // slab geometry the smallest admissible S is 4, not 1), and the pool must
    // still leave the high-water cushion plus room to walk below low water.
    //
    // The x4 also has to bracket the BAND-WIDTH GUARD, which raises the high
    // mark to low + 2S when the configured band is narrower. With total = 4S
    // and the default 0.15 low mark that leaves a grant budget of
    // 4S - (0.6S + 2S) = 1.4S >= S, so a grant still fits above the widened
    // mark. Anything tighter than x4 would starve the grant tests.
    const int slabs_per_grant =
        std::max(1, geo.grant_slabs(kMaxSlotsPerGrant, 1 << 20));
    const int want_slabs = std::max(24, slabs_per_grant * 4) * size_mult;

    auto cfg = rb_config(num_gpus);
    cfg.memory.kv_cache.max_pages_per_gpu = want_slabs * pps;
    return build_ctx(std::move(cfg), num_gpus);
}

lmem::ExpertZoneGeometry geometry_of(const RbCtx& ctx, double max_waste) {
    return lmem::ExpertZoneGeometry{.expert_slot_bytes = ctx.expert_bytes,
                                    .slab_bytes = ctx.pa->slab_bytes(),
                                    .max_waste = max_waste};
}

// ── Injected device surface (counting stubs) ───────────────────────────────

struct Stubs {
    bool quiesced = true;
    bool events_done = true;
    int h2d_calls = 0;
    int ffn_calls = 0;
    int destroys = 0;
    int complete_queries = 0;
    uintptr_t next_event = 0x100000;

    void* fresh_event() {
        next_event += 64;
        return reinterpret_cast<void*>(next_event);
    }
};

ld::ExpertZoneRebalancer::Config rb_cfg(double low = 0.15, double high = 0.30,
                                        double max_waste = 0.10) {
    ld::ExpertZoneRebalancer::Config c;
    c.enabled = true;
    c.max_waste = max_waste;
    c.low_water_frac = low;
    c.high_water_frac = high;
    c.max_slots_per_grant = kMaxSlotsPerGrant;
    c.min_tick_interval_us = 0;  // every tick() is an effective pass
    // Pinned OFF rather than inherited: the production default (30 s) is a
    // live-serving anti-churn choice, not a property any of these tests are
    // asserting. GrantCooldownAfterReclaim sets its own value; the rest must
    // not silently depend on whatever that default happens to be.
    c.grant_cooldown_ms = 0;
    return c;
}

ld::ExpertZoneRebalancer::Deps rb_deps(RbCtx& ctx, Stubs& s,
                                       double max_waste = 0.10) {
    ld::ExpertZoneRebalancer::Deps d;
    d.page_allocator = ctx.pa.get();
    d.expert_cache = ctx.cache.get();
    d.tp_gpus = ctx.tp_gpus;
    d.geometry = geometry_of(ctx, max_waste);
    d.moe_quiesced = [&s] { return s.quiesced; };
    d.record_h2d_barrier = [&s](int) -> void* {
        ++s.h2d_calls;
        return s.fresh_event();
    };
    d.record_ffn_barrier = [&s](int) -> void* {
        ++s.ffn_calls;
        return s.fresh_event();
    };
    d.event_complete = [&s](int, void*) {
        ++s.complete_queries;
        return s.events_done;
    };
    d.destroy_event = [&s](int, void*) { ++s.destroys; };
    return d;
}

// ── Pool shaping helpers ───────────────────────────────────────────────────

int free_slabs_min(const RbCtx& ctx) {
    int f = INT32_MAX;
    for (const int g : ctx.tp_gpus)
        f = std::min(f, ctx.pa->kv_fragmentation(g).free_slabs);
    return f;
}

int total_slabs_of(const RbCtx& ctx) {
    return ctx.pa->kv_fragmentation(ctx.tp_gpus.front()).total_slabs;
}

/// Pin free slabs from the BOTTOM (grants land at the TOP, so this never
/// collides with a granted run) in lockstep on every tp rank, until the
/// minimum free-slab count reaches `target_free`.
std::vector<lmem::PageHandle> pin_down_to(RbCtx& ctx, int target_free) {
    std::vector<lmem::PageHandle> pins;
    const int total = total_slabs_of(ctx);
    for (int sid = 0; sid < total && free_slabs_min(ctx) > target_free; ++sid) {
        std::vector<lmem::PageHandle> round;
        bool all = true;
        for (const int g : ctx.tp_gpus) {
            auto h = ctx.pa->claim_one_slab_for_test(g, sid);
            if (!h) { all = false; break; }
            round.push_back(*h);
        }
        if (!all) {  // slab busy on some rank — undo the partial round
            for (auto& h : round) ctx.pa->free(h);
            continue;
        }
        for (auto& h : round) pins.push_back(h);
    }
    return pins;
}

/// Reserve into every free kStable slot (the elastic zone included) and mark
/// the entries fully arrived, so the drain's "ready + unlocked" gate passes.
void fill_stable(RbCtx& ctx, int gpu) {
    const int cap = ctx.cache->total_slots(gpu, lmem::CacheZone::kStable);
    for (int e = 0; e < cap + 8; ++e) {
        lmem::ExpertKey k{0u, static_cast<uint16_t>(e)};
        if (ctx.cache->is_resident(k, gpu)) continue;
        if (!ctx.cache->reserve(k, gpu, lmem::CacheZone::kStable)) break;
        ctx.cache->mark_all_ready(k, gpu);
    }
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// ExpertZoneRebalancer
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertZoneRebalancer, DisabledNoOp) {
    auto ctx = make_ctx();
    Stubs s;
    auto cfg = rb_cfg();
    cfg.enabled = false;
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), cfg);

    ASSERT_FALSE(rb.enabled());
    for (int i = 0; i < 5; ++i) rb.tick();

    // Default OFF must be BYTE-IDENTICAL to the mechanism not existing.
    const auto& ez = ctx->pa->expert_zone_stats(0);
    EXPECT_EQ(ez.grants, 0);
    EXPECT_EQ(ez.releases, 0);
    EXPECT_EQ(ez.refusals, 0);
    EXPECT_EQ(ez.granted_slabs, 0);
    EXPECT_EQ(rb.stats().grants, 0);
    EXPECT_EQ(rb.stats().grant_refusals, 0);
    EXPECT_EQ(s.h2d_calls, 0);
    EXPECT_EQ(s.ffn_calls, 0);
}

TEST(ExpertZoneRebalancer, GrantAboveHighWater) {
    auto ctx = make_ctx();
    Stubs s;
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), rb_cfg());
    ASSERT_TRUE(rb.enabled());

    const int total = total_slabs_of(*ctx);
    ASSERT_GE(total, 24) << "fixture must give the band room to exist";
    const int base_stable = ctx->cache->total_slots(0, lmem::CacheZone::kStable);
    const int high = rb.high_water_slabs();
    const auto geo = geometry_of(*ctx, 0.10);
    ASSERT_GT(free_slabs_min(*ctx), high) << "fresh pool must be above high";
    // Reproduce the run the rebalancer will size against: the budget above
    // the high-water mark, capped by the longest contiguous free run.
    const int run0 = std::min(ctx->pa->largest_free_run(0),
                              free_slabs_min(*ctx) - high);
    ASSERT_GT(run0, 0);

    rb.tick();

    const auto& st = rb.stats();
    ASSERT_EQ(st.grants, 1);
    EXPECT_EQ(st.grant_refusals, 0);
    EXPECT_GT(st.granted_slabs, 0);
    EXPECT_GE(st.granted_slots, 1);
    // Sizing: the cap bounds the ASK, and the slab count is the SMALLEST S
    // that both meets that ask and stays admissible under the WORST-CASE base
    // pad — the pad is unknown at sizing time because slab_bytes is not a
    // 4096-multiple on any served model.
    EXPECT_EQ(st.granted_slabs,
              geo.grant_slabs(
                  std::min(geo.best_fit(run0, /*base_misalign=*/1).num_slots,
                           kMaxSlotsPerGrant),
                  run0));

    // The grant is CAPACITY, not an allocation: total slabs unchanged, and
    // the cache's stable pool grew by exactly the slots we handed it.
    EXPECT_EQ(total_slabs_of(*ctx), total);
    EXPECT_EQ(ctx->cache->total_slots(0, lmem::CacheZone::kStable),
              base_stable + st.granted_slots);
    EXPECT_EQ(ctx->pa->expert_zone_stats(0).grants, 1);
    EXPECT_EQ(ctx->pa->expert_zone_stats(0).granted_slabs, st.granted_slabs);
    EXPECT_EQ(free_slabs_min(*ctx), total - st.granted_slabs);
    // Placement is TOP-DOWN, so the untouched bottom is one contiguous run.
    EXPECT_EQ(ctx->pa->largest_free_run(0), total - st.granted_slabs);

    // Waste bound (44z §3), evaluated at the run's ACTUAL base alignment.
    // FILL semantics: the grant takes every slot the claimed slabs hold, so
    // the capacity it did NOT hand the cache is only the geometry's own
    // remainder — which is exactly what waste_frac must keep under max_waste.
    const int start = total - st.granted_slabs;
    const auto base = static_cast<char*>(ctx->pa->kv_main_base(0))
                      + static_cast<int64_t>(start) * ctx->pa->slab_bytes();
    const auto misalign = static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(base) % lmem::kExpertZoneSlotAlign);
    EXPECT_EQ(st.granted_slots, geo.slots_in(st.granted_slabs, misalign));
    EXPECT_LE(geo.waste_frac(st.granted_slabs, st.granted_slots), 0.10);

    // A grant must never itself trip the reclaim it would have caused: free
    // slabs are still above the high-water mark after it lands.
    EXPECT_GE(free_slabs_min(*ctx), high);
    EXPECT_EQ(st.reclaims, 0);
    EXPECT_EQ(st.draining_zones, 0);
}

TEST(ExpertZoneRebalancer, HysteresisBandHolds) {
    // Inside the band — above low, at/below high — NOTHING happens. This is
    // the whole point of two marks instead of one.
    auto ctx = make_ctx();
    Stubs s;
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), rb_cfg());

    const int high = rb.high_water_slabs();
    const int low = rb.low_water_slabs();
    ASSERT_GT(high, low) << "the band must have width";
    auto pins = pin_down_to(*ctx, high);
    ASSERT_EQ(free_slabs_min(*ctx), high);
    ASSERT_GE(free_slabs_min(*ctx), low);

    for (int i = 0; i < 4; ++i) rb.tick();

    EXPECT_EQ(rb.stats().grants, 0) << "grants require free > high";
    EXPECT_EQ(rb.stats().reclaims, 0) << "reclaims require free < low";
    EXPECT_EQ(rb.stats().grant_refusals, 0);
    EXPECT_EQ(rb.stats().draining_zones, 0);
    EXPECT_EQ(ctx->pa->expert_zone_stats(0).grants, 0);
    for (auto& h : pins) ctx->pa->free(h);
}

TEST(ExpertZoneRebalancer, NoGrantWhileDraining) {
    // Handing slabs out while capacity is moving the other way would make the
    // reclaim pointless and re-fragment the region being cleared.
    auto ctx = make_ctx();
    Stubs s;
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), rb_cfg());

    rb.tick();
    ASSERT_EQ(rb.stats().grants, 1);

    // Arm the drain, then hold it open by refusing quiesce.
    s.quiesced = false;
    rb.note_pool_pressure_refusal(0);
    rb.tick();
    ASSERT_EQ(rb.stats().draining_zones, 1);
    ASSERT_GT(free_slabs_min(*ctx), rb.high_water_slabs())
        << "the pool is still comfortable — only the drain blocks a grant";

    const int64_t grants_before = rb.stats().grants;
    for (int i = 0; i < 4; ++i) rb.tick();
    EXPECT_EQ(rb.stats().grants, grants_before);
    EXPECT_EQ(rb.stats().draining_zones, 1);
    EXPECT_EQ(rb.stats().reclaims, 0);
}

TEST(ExpertZoneRebalancer, ReclaimBelowLowWater) {
    auto ctx = make_ctx();
    Stubs s;
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), rb_cfg());

    const int total = total_slabs_of(*ctx);
    const int base_stable = ctx->cache->total_slots(0, lmem::CacheZone::kStable);
    rb.tick();
    ASSERT_EQ(rb.stats().grants, 1);
    const int granted_slabs = rb.stats().granted_slabs;
    const int granted_slots = rb.stats().granted_slots;
    ASSERT_GT(granted_slabs, 0);

    // Put real residents in the lent region: the reclaim must EVICT them,
    // not merely forget the zone.
    fill_stable(*ctx, 0);
    ASSERT_GT(ctx->cache->used_slots(0, lmem::CacheZone::kStable), 0);

    // Walk the pool under the low-water mark.
    auto pins = pin_down_to(*ctx, rb.low_water_slabs() - 1);
    ASSERT_LT(free_slabs_min(*ctx), rb.low_water_slabs());

    rb.tick();
    EXPECT_EQ(rb.stats().draining_zones, 1) << "drain starts in the background";

    for (int i = 0; i < 6 && rb.stats().reclaims == 0; ++i) rb.tick();

    EXPECT_EQ(rb.stats().reclaims, 1);
    EXPECT_EQ(rb.stats().granted_slabs, 0);
    EXPECT_EQ(rb.stats().granted_slots, 0);
    EXPECT_EQ(rb.stats().draining_zones, 0);
    EXPECT_EQ(ctx->pa->expert_zone_stats(0).releases, 1);
    EXPECT_EQ(ctx->pa->expert_zone_stats(0).granted_slabs, 0);
    EXPECT_EQ(ctx->cache->total_slots(0, lmem::CacheZone::kStable),
              base_stable);
    EXPECT_EQ(total_slabs_of(*ctx), total);
    // The slabs came back WHOLE and re-coalesced with the free top.
    EXPECT_GE(ctx->pa->largest_free_run(0), granted_slabs);
    EXPECT_EQ(s.destroys, s.h2d_calls + s.ffn_calls);
    (void)granted_slots;
    for (auto& h : pins) ctx->pa->free(h);
}

TEST(ExpertZoneRebalancer, QuiesceDoesNotWaitForStalledPipeline) {
    // THE LIVELOCK (measured: engine wedged 9+ min, GPU 0%, drain stuck at
    // START). Quiesce means "no MoE kernel in flight", NOT "pipeline idle".
    // A prefill stalled on a KV page claim holds its progressive state ACTIVE
    // while it waits — and it is waiting for the very slabs this drain is
    // trying to hand back. If quiesce demanded an idle pipeline the cycle
    // never breaks: drain waits on quiesce, quiesce waits on the prefill, the
    // prefill waits on the drain. The stub here is TRUE because no kernel is
    // running, which is exactly what the stalled-but-active case reports.
    auto ctx = make_ctx();
    Stubs s;
    s.quiesced = true;  // pending_compute == 0, progressive state still ACTIVE
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), rb_cfg());

    rb.tick();
    ASSERT_EQ(rb.stats().grants, 1);
    fill_stable(*ctx, 0);

    // Pressure arrives (this IS the stalled prefill's page claim failing).
    auto pins = pin_down_to(*ctx, rb.low_water_slabs() - 1);
    for (int i = 0; i < 8 && rb.stats().reclaims == 0; ++i) rb.tick();

    // The drain MUST complete and give the slabs back — that is what unblocks
    // the stalled admission.
    EXPECT_EQ(rb.stats().reclaims, 1)
        << "a stalled-but-active pipeline must not block the reclaim that "
           "would unstall it";
    EXPECT_EQ(rb.stats().granted_slabs, 0);
    EXPECT_EQ(ctx->pa->expert_zone_stats(0).releases, 1);
    // The safety steps still ran: barriers on the h2d and FFN streams are what
    // actually prove no kernel or DMA can still touch the region.
    EXPECT_GT(s.h2d_calls, 0);
    EXPECT_GT(s.ffn_calls, 0);
    EXPECT_EQ(s.destroys, s.h2d_calls + s.ffn_calls);
    for (auto& h : pins) ctx->pa->free(h);
}

TEST(ExpertZoneRebalancer, ReclaimGatedOnQuiesce) {
    // The ORDER is the assertion: no barrier before quiesce, no release
    // before the barrier events complete.
    auto ctx = make_ctx();
    Stubs s;
    s.quiesced = false;
    s.events_done = false;
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), rb_cfg());

    rb.tick();
    ASSERT_EQ(rb.stats().grants, 1);
    const int granted_slabs = rb.stats().granted_slabs;
    auto pins = pin_down_to(*ctx, rb.low_water_slabs() - 1);

    // Drained (no residents) but NOT quiesced: bookkeeping only.
    for (int i = 0; i < 3; ++i) rb.tick();
    EXPECT_EQ(rb.stats().draining_zones, 1);
    EXPECT_EQ(s.h2d_calls, 0) << "a barrier before quiesce would be a lie: "
                                 "cached pointer tables outlive residency";
    EXPECT_EQ(s.ffn_calls, 0);
    EXPECT_EQ(rb.stats().reclaims, 0);
    EXPECT_EQ(ctx->pa->expert_zone_stats(0).releases, 0);

    // Quiesced, events still pending: barriers recorded, capacity NOT moved.
    s.quiesced = true;
    rb.tick();
    EXPECT_EQ(s.h2d_calls, static_cast<int>(ctx->tp_gpus.size()));
    EXPECT_EQ(s.ffn_calls, static_cast<int>(ctx->tp_gpus.size()));
    EXPECT_EQ(rb.stats().reclaims, 0);
    EXPECT_EQ(rb.stats().granted_slabs, granted_slabs);
    EXPECT_EQ(ctx->pa->expert_zone_stats(0).releases, 0);
    EXPECT_EQ(s.destroys, 0);

    rb.tick();
    EXPECT_EQ(rb.stats().reclaims, 0) << "cancel() does not stop a dispatched "
                                         "DMA — only the event does";
    EXPECT_EQ(s.h2d_calls, static_cast<int>(ctx->tp_gpus.size()))
        << "barriers are recorded ONCE";

    // Events complete: now, and only now, the slabs are KV's again.
    s.events_done = true;
    rb.tick();
    EXPECT_EQ(rb.stats().reclaims, 1);
    EXPECT_EQ(rb.stats().granted_slabs, 0);
    EXPECT_EQ(ctx->pa->expert_zone_stats(0).releases, 1);
    EXPECT_EQ(s.destroys, s.h2d_calls + s.ffn_calls);
    for (auto& h : pins) ctx->pa->free(h);
}

TEST(ExpertZoneRebalancer, LockedResidentDefersReclaim) {
    // #90: a locked entry is NAMED by an in-flight dispatch. The drain waits;
    // it never forces the eviction.
    auto ctx = make_ctx();
    Stubs s;
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), rb_cfg());

    rb.tick();
    ASSERT_EQ(rb.stats().grants, 1);

    // Fill EVERY stable slot (the granted zone's included) and lock them all
    // BEFORE the drain starts, so at least one locked resident is certainly
    // inside the elastic zone.
    fill_stable(*ctx, 0);
    const int cap = ctx->cache->total_slots(0, lmem::CacheZone::kStable);
    ASSERT_EQ(ctx->cache->used_slots(0, lmem::CacheZone::kStable), cap)
        << "granted slots must be reachable through the normal reserve path";
    std::vector<lmem::ExpertKey> locked;
    for (int e = 0; e < cap + 8; ++e) {
        lmem::ExpertKey k{0u, static_cast<uint16_t>(e)};
        if (ctx->cache->is_resident(k, 0) && ctx->cache->lock(k, 0))
            locked.push_back(k);
    }
    ASSERT_FALSE(locked.empty());

    auto pins = pin_down_to(*ctx, rb.low_water_slabs() - 1);

    rb.tick();
    ASSERT_EQ(rb.stats().draining_zones, 1);
    const int h2d_before = s.h2d_calls;
    for (int i = 0; i < 3; ++i) rb.tick();

    // The drain STALLS — it never forces a locked eviction, and it never
    // records a barrier over a region the cache still hands out.
    EXPECT_EQ(rb.stats().reclaims, 0);
    EXPECT_EQ(rb.stats().draining_zones, 1);
    EXPECT_EQ(s.h2d_calls, h2d_before)
        << "no barrier while the zone still holds a resident";
    for (const auto& k : locked)
        EXPECT_TRUE(ctx->cache->is_resident(k, 0))
            << "a locked resident must survive the drain";

    for (const auto& k : locked) ctx->cache->unlock(k, 0);
    for (int i = 0; i < 6 && rb.stats().reclaims == 0; ++i) rb.tick();
    EXPECT_EQ(rb.stats().reclaims, 1) << "unlocking releases the drain";
    EXPECT_EQ(rb.stats().granted_slabs, 0);
    EXPECT_GT(s.h2d_calls, h2d_before);
    for (auto& h : pins) ctx->pa->free(h);
}

TEST(ExpertZoneRebalancer, DrainEvictsThroughLifecycleSeam) {
    // INV-ELM-EVICT (TD-KVXP-RECLAIM-REGRANT-WEDGE): step (2) of the drain
    // evicts THROUGH the injected lifecycle seam, never ExpertCache::evict
    // directly. A cache-direct evict leaves the lifecycle manager's
    // (key,gpu) tier kHot for a key the cache no longer holds;
    // ensure_resident then never re-fetches the key and every MoE layer
    // routing it silently burns its full fetch deadline — the measured
    // GPU-0% crawl that looked like a hard engine hang. Also proves the
    // REFUSAL semantics: a seam that declines (pending interests, in-flight
    // transfer) defers the drain to a later tick — never a forced eviction,
    // never a fallback to the cache-direct path.
    auto ctx = make_ctx();
    Stubs s;
    auto d = rb_deps(*ctx, s);
    int seam_calls = 0;
    bool seam_allow = false;
    d.evict_expert = [&](lmem::ExpertKey k, int gpu) {
        ++seam_calls;
        return seam_allow ? ctx->cache->evict(k, gpu) : false;
    };
    ld::ExpertZoneRebalancer rb(std::move(d), rb_cfg());

    rb.tick();
    ASSERT_EQ(rb.stats().grants, 1);
    fill_stable(*ctx, 0);  // residents land in the granted zone too
    auto pins = pin_down_to(*ctx, rb.low_water_slabs() - 1);

    // Seam refuses: the drain DEFERS — nothing is evicted behind its back.
    // (used_slots is the wrong witness here: begin_drain flips the zone to
    // kDraining, which removes its slots from the STABLE TOTALS by
    // accounting alone. total_resident() counts actual cache entries.)
    const int residents_before = ctx->cache->total_resident();
    for (int i = 0; i < 3; ++i) rb.tick();
    EXPECT_GT(seam_calls, 0) << "drain must route evictions to the seam";
    EXPECT_EQ(rb.stats().draining_zones, 1);
    EXPECT_EQ(rb.stats().reclaims, 0);
    EXPECT_EQ(ctx->cache->total_resident(), residents_before)
        << "a refused seam evict must NOT fall back to cache-direct";

    // Seam allows: the drain completes through it.
    seam_allow = true;
    for (int i = 0; i < 8 && rb.stats().reclaims == 0; ++i) rb.tick();
    EXPECT_EQ(rb.stats().reclaims, 1);
    EXPECT_LT(ctx->cache->total_resident(), residents_before);
    for (auto& h : pins) ctx->pa->free(h);
}

TEST(ExpertZoneRebalancer, ForcedImmediateCounter) {
    auto ctx = make_ctx();
    Stubs s;
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), rb_cfg());

    rb.tick();
    ASSERT_EQ(rb.stats().grants, 1);
    ASSERT_GE(free_slabs_min(*ctx), rb.low_water_slabs())
        << "the pool is comfortable — this refusal is the POLICY failing";
    EXPECT_EQ(rb.stats().forced_immediate_reclaims, 0);

    rb.note_pool_pressure_refusal(0);
    EXPECT_EQ(rb.stats().forced_immediate_reclaims, 1);

    // Eager: the drain starts even though free never crossed low water.
    s.quiesced = false;
    rb.tick();
    EXPECT_EQ(rb.stats().draining_zones, 1);

    // No grant outstanding = nothing the rebalancer could have handed back,
    // so it is NOT a policy failure and must not be counted as one.
    auto ctx2 = make_ctx();
    Stubs s2;
    ld::ExpertZoneRebalancer rb2(rb_deps(*ctx2, s2), rb_cfg());
    rb2.note_pool_pressure_refusal(0);
    rb2.note_pool_pressure_refusal(0);
    EXPECT_EQ(rb2.stats().forced_immediate_reclaims, 0);
}

TEST(ExpertZoneRebalancer, EagerReclaimDrainsToTheReportedDemand) {
    // TD-KVXP-BOOT-OVERGRANT-FIRST-ADMISSION (P-30 step 3). The old eager
    // drain was DEMAND-BLIND: it force-drained ONE grant and stopped at
    // `covered >= high`, so a refusal whose shortage spanned several grants
    // (the 97k prefill: ~2,470 slabs against ~340-540-slab grants) got one
    // span back, the cooldown expired, a re-grant took it again, and the
    // admission was blocked FOREVER. The refusal now reports its shortfall
    // and the drain hands back as many grants as that demand needs.
    // NEGATIVE CONTROL: on the old code this test fails at both the
    // `reclaims >= 3` and the `free >= demand` asserts (one reclaim, one
    // grant's worth of slabs).
    auto ctx = make_ctx(/*num_gpus=*/1, /*size_mult=*/4);
    Stubs s;
    auto cfg = rb_cfg();
    // Production-shaped cooldown so a completed reclaim cannot be re-granted
    // away inside this test's tick window (the tick interval is 0 => 1 ms
    // per tick, so this is 60,000 effective ticks).
    cfg.grant_cooldown_ms = 60000;
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), cfg);

    // Let the boot grant loop park the pool: grants until nothing fits.
    for (int i = 0; i < 32; ++i) rb.tick();
    const int64_t grants_out = rb.stats().grants;
    ASSERT_GE(grants_out, 4) << "fixture must park multiple grants";
    const int slabs_per_grant =
        static_cast<int>(rb.stats().granted_slabs / grants_out);
    ASSERT_GT(slabs_per_grant, 0);

    // A refusal three grants short of its claim.
    const int free0 = free_slabs_min(*ctx);
    const int64_t shortfall = 3LL * slabs_per_grant;
    rb.note_pool_pressure_refusal(0, shortfall);
    EXPECT_EQ(rb.stats().forced_immediate_reclaims, 1);
    // Demand-aware target: today's free + the shortfall (+ the admission
    // margin, 0 on this fixture — transient/growth unwired).
    EXPECT_EQ(rb.eager_demand_free_slabs(), free0 + shortfall);

    for (int i = 0; i < 16; ++i) rb.tick();

    EXPECT_GE(rb.stats().reclaims, 3)
        << "the drain must hand back as many grants as the demand needs, "
           "not force-drain one and stop at high water";
    EXPECT_GE(free_slabs_min(*ctx), free0 + shortfall)
        << "free must reach the refusal's demand";
    EXPECT_EQ(rb.eager_demand_free_slabs(), 0) << "disarmed once satisfied";
}

TEST(ExpertZoneRebalancer, GrantsNeverParkBelowTheAdmissionDemandFloor) {
    // TD-KVXP-BOOT-OVERGRANT-FIRST-ADMISSION candidate fix 2 (the grant
    // cap): before any request exists to size against, the boot grant loop
    // used to grant until free parked JUST ABOVE HIGH WATER (measured:
    // free 420-455 vs high 414), where the FIRST admission's non-deferrable
    // upfront claims (state transient 306 + KV + indexer) no longer fit —
    // every first admission then rode the refusal seam. Grants must leave
    // low + one admission's upfront margin free.
    // NEGATIVE CONTROL: the old budget was free - high, so on the old code
    // the pool parks below the floor and the GE assert fails.
    auto ctx = make_ctx(/*num_gpus=*/1, /*size_mult=*/4);
    const int total = total_slabs_of(*ctx);
    const auto geo = geometry_of(*ctx, 0.10);
    const int64_t grant_span = geo.grant_slabs(kMaxSlotsPerGrant, total);
    ASSERT_GT(grant_span, 0);

    // Per-step-floor shape: upfront state transient of two grant spans, a
    // small growth term. floor = low + T + G sits a whole 2-grant demand
    // above high = low + 2G, so the old code would grant twice more than
    // the new code allows.
    const int64_t T = 2 * grant_span;
    const int64_t G = 2;
    Stubs s;
    auto d = rb_deps(*ctx, s);
    d.admission_transient_slabs = T;
    d.per_step_growth_slabs = G;
    d.max_concurrent_admissions = 1;
    ld::ExpertZoneRebalancer rb(std::move(d),
                                rb_cfg(/*low=*/0.0, /*high=*/0.0));
    ASSERT_TRUE(rb.enabled());
    const int64_t floor = rb.low_water_slabs() + T + G;
    ASSERT_EQ(rb.grant_floor_slabs(), floor);
    ASSERT_GT(floor, rb.high_water_slabs())
        << "the fixture must make the floor BITE (floor > high), or the "
           "negative control is vacuous";

    for (int i = 0; i < 32; ++i) rb.tick();

    EXPECT_GT(rb.stats().grants, 0) << "the floor must not stop ALL grants";
    EXPECT_GE(free_slabs_min(*ctx), floor)
        << "grants must never park free below low + one admission's "
           "upfront margin";
}

TEST(ExpertZoneRebalancer, MirrorLockstep) {
    // Replicated KV claims pages BY INDEX: a per-rank-divergent span would
    // collapse replicated capacity to the intersection of the free sets.
    auto ctx = make_ctx(/*num_gpus=*/2);
    ASSERT_EQ(ctx->tp_gpus.size(), 2u);
    Stubs s;
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), rb_cfg());

    const int total = total_slabs_of(*ctx);
    const int base0 = ctx->cache->total_slots(0, lmem::CacheZone::kStable);
    const int base1 = ctx->cache->total_slots(1, lmem::CacheZone::kStable);
    rb.tick();
    ASSERT_EQ(rb.stats().grants, 1);
    const int slabs = rb.stats().granted_slabs;
    const int slots = rb.stats().granted_slots;
    ASSERT_GT(slabs, 0);

    for (const int g : ctx->tp_gpus) {
        EXPECT_EQ(ctx->pa->expert_zone_stats(g).grants, 1) << "gpu " << g;
        EXPECT_EQ(ctx->pa->expert_zone_stats(g).granted_slabs, slabs)
            << "gpu " << g;
        EXPECT_EQ(ctx->pa->kv_fragmentation(g).free_slabs, total - slabs)
            << "gpu " << g;
        // Same START on both ranks: the topmost slab is claimed everywhere.
        EXPECT_FALSE(ctx->pa->claim_one_slab_for_test(g, total - 1).has_value())
            << "gpu " << g << " must have the SAME top slab claimed";
    }
    EXPECT_EQ(ctx->cache->total_slots(0, lmem::CacheZone::kStable),
              base0 + slots);
    EXPECT_EQ(ctx->cache->total_slots(1, lmem::CacheZone::kStable),
              base1 + slots);

    // And the reclaim gives BOTH ranks back.
    auto pins = pin_down_to(*ctx, rb.low_water_slabs() - 1);
    for (int i = 0; i < 8 && rb.stats().reclaims == 0; ++i) rb.tick();
    EXPECT_EQ(rb.stats().reclaims, 1);
    for (const int g : ctx->tp_gpus) {
        EXPECT_EQ(ctx->pa->expert_zone_stats(g).releases, 1) << "gpu " << g;
        EXPECT_EQ(ctx->pa->expert_zone_stats(g).granted_slabs, 0) << "gpu " << g;
    }
    EXPECT_EQ(ctx->cache->total_slots(0, lmem::CacheZone::kStable), base0);
    EXPECT_EQ(ctx->cache->total_slots(1, lmem::CacheZone::kStable), base1);
    // Barriers on EVERY rank, and every event destroyed.
    EXPECT_EQ(s.h2d_calls, 2);
    EXPECT_EQ(s.ffn_calls, 2);
    EXPECT_EQ(s.destroys, 4);
    for (auto& h : pins) ctx->pa->free(h);
}

TEST(ExpertZoneRebalancer, GrantRespectsExtraReserve) {
    // The grant carries the high-water cushion into the allocator as
    // extra_reserve_pages ON TOP of the INV-4.9f floor. Raise the floor so
    // the two together cover the whole pool: the claim must refuse rather
    // than eat KV's growth room, and the refusal must be COUNTED.
    auto ctx = make_ctx();
    Stubs s;
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), rb_cfg());

    const int total = total_slabs_of(*ctx);
    const int high = rb.high_water_slabs();
    const int pps = ctx->pa->pages_per_slab();
    ASSERT_GT(total - high, 0);
    // main_reserved = everything above the high-water mark ⇒ free_pages minus
    // any claim can never clear reserved + extra_reserve.
    ctx->pa->configure_headroom(
        lmem::PageAllocator::HeadroomConfig{
            .max_concurrent_forks = (total - high) * pps,
            .max_concurrent_sequences = 0,
            .page_growth_chunk_pages = 0});
    ASSERT_GT(free_slabs_min(*ctx), high) << "the SLAB budget still looks fine";

    rb.tick();

    EXPECT_EQ(rb.stats().grants, 0);
    EXPECT_EQ(rb.stats().granted_slabs, 0);
    EXPECT_EQ(rb.stats().grant_refusals, 1);
    EXPECT_GE(ctx->pa->expert_zone_stats(0).refusals, 1);
    EXPECT_EQ(ctx->pa->expert_zone_stats(0).grants, 0);
    EXPECT_EQ(ctx->pa->kv_fragmentation(0).free_slabs, total);
}

TEST(ExpertZoneRebalancer, BandGuardWidensNarrowBand) {
    // A band narrower than one admission's transient is not hysteresis: the
    // admission crosses it end to end, so the pool reclaims on the way down
    // and re-grants the same span on the way up, once PER REQUEST. Measured
    // on a 3-percentage-point band: residents evicted mid-service, 12 layers
    // past the CPU-overlap deadline, 1.54 tok/s. So the width is CHECKED.
    auto ctx = make_ctx(/*num_gpus=*/1, /*size_mult=*/4);
    const int total = total_slabs_of(*ctx);
    const auto geo = geometry_of(*ctx, 0.10);
    const int64_t two_grants =
        2 * geo.grant_slabs(kMaxSlotsPerGrant, total);
    ASSERT_GT(two_grants, 0) << "fixture must admit at least one grant";

    // Narrow requested marks (2 percentage points), no transient wired: the
    // two-grants-wide floor alone must set the band, because a grant that
    // straddles the band lands free under low water by itself.
    Stubs s1;
    auto narrow = rb_cfg(/*low=*/0.10, /*high=*/0.12);
    ld::ExpertZoneRebalancer rb1(rb_deps(*ctx, s1), narrow);
    ASSERT_TRUE(rb1.enabled());
    const int low1 = rb1.low_water_slabs();
    EXPECT_EQ(rb1.min_band_slabs(), two_grants);
    EXPECT_EQ(rb1.high_water_slabs(), low1 + static_cast<int>(two_grants));
    // The requested high floor is subordinate: the band decides, and the
    // floor is only ever honored when it sits ABOVE the derived mark.
    EXPECT_GE(rb1.high_water_slabs(),
              static_cast<int>(std::ceil(0.12 * total)));

    // An admission transient that DOMINATES the two-grants floor: the band
    // tracks 2x the transient instead.
    Stubs s2;
    auto deps2 = rb_deps(*ctx, s2);
    const int64_t transient = two_grants;  // = 2 grants, so 2x it dominates
    deps2.admission_transient_slabs = transient;
    ld::ExpertZoneRebalancer rb2(std::move(deps2), narrow);
    ASSERT_TRUE(rb2.enabled()) << "these marks must still fit the pool";
    EXPECT_EQ(rb2.min_band_slabs(), 2 * transient);
    EXPECT_GT(rb2.min_band_slabs(), two_grants) << "transient must dominate";
    EXPECT_EQ(rb2.high_water_slabs(),
              rb2.low_water_slabs() + static_cast<int>(2 * transient));

    // A transient the pool cannot possibly bracket: refuse to run at all
    // rather than run marks that are known to thrash.
    Stubs s3;
    auto deps3 = rb_deps(*ctx, s3);
    deps3.admission_transient_slabs = total;
    ld::ExpertZoneRebalancer rb3(std::move(deps3), narrow);
    EXPECT_FALSE(rb3.enabled())
        << "unsafe marks must disable the rebalancer, not run anyway";
    rb3.tick();
    EXPECT_EQ(rb3.stats().grants, 0);
    EXPECT_EQ(ctx->pa->expert_zone_stats(0).grants, 0);
}

TEST(ExpertZoneRebalancer, FracFloorStillRaises) {
    // The fractions did not disappear — they became optional FLOORS. A high
    // floor ABOVE the absolute derivation still wins, which is what keeps the
    // old pool-fraction behavior (and the tight-marks experiment) available.
    auto ctx = make_ctx(/*num_gpus=*/1, /*size_mult=*/4);
    const int total = total_slabs_of(*ctx);
    const auto geo = geometry_of(*ctx, 0.10);
    const int64_t two_grants = 2 * geo.grant_slabs(kMaxSlotsPerGrant, total);

    Stubs s;
    auto raised = rb_cfg(/*low=*/0.05, /*high=*/0.90);
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), raised);
    ASSERT_TRUE(rb.enabled());

    EXPECT_EQ(rb.low_water_slabs(), static_cast<int>(std::ceil(0.05 * total)));
    EXPECT_EQ(rb.high_water_slabs(), static_cast<int>(std::ceil(0.90 * total)))
        << "an explicit high floor above the derived mark must win";
    EXPECT_GT(rb.high_water_slabs(),
              rb.low_water_slabs() + static_cast<int>(two_grants))
        << "this floor must genuinely sit above the absolute derivation";
    // The band the guard requires is still satisfied — raising a floor can
    // only widen the band, never narrow it.
    EXPECT_GE(rb.high_water_slabs() - rb.low_water_slabs(),
              static_cast<int>(rb.min_band_slabs()));
}

TEST(ExpertZoneRebalancer, AbsoluteMarksRecoverEmptyPool) {
    // THE SHAPE OF THE RESERVE. What has to stay free is one admission's
    // transient plus the band — a property of an ADMISSION, not of the pool.
    // A fraction-shaped reserve strands proportionally MORE capacity the
    // emptier the pool is, which is exactly backwards: an idle pool is
    // precisely when the expert cache should be allowed to borrow. So the
    // marks must NOT move when the pool grows.
    auto small = make_ctx(/*num_gpus=*/1, /*size_mult=*/4);
    auto big = make_ctx(/*num_gpus=*/1, /*size_mult=*/8);
    const int total_small = total_slabs_of(*small);
    const int total_big = total_slabs_of(*big);
    ASSERT_GT(total_big, total_small) << "the two pools must differ in size";

    // No fraction floors: the marks come out of the transient alone.
    const auto geo = geometry_of(*small, 0.10);
    const int64_t transient = geo.grant_slabs(kMaxSlotsPerGrant, total_small);
    ASSERT_GT(transient, 0);

    Stubs s1;
    auto deps1 = rb_deps(*small, s1);
    deps1.admission_transient_slabs = transient;
    ld::ExpertZoneRebalancer rb_small(std::move(deps1),
                                      rb_cfg(/*low=*/0.0, /*high=*/0.0));
    Stubs s2;
    auto deps2 = rb_deps(*big, s2);
    deps2.admission_transient_slabs = transient;
    ld::ExpertZoneRebalancer rb_big(std::move(deps2),
                                    rb_cfg(/*low=*/0.0, /*high=*/0.0));
    ASSERT_TRUE(rb_small.enabled());
    ASSERT_TRUE(rb_big.enabled());

    // Identical marks on pools of different sizes — that is the whole point.
    EXPECT_EQ(rb_small.low_water_slabs(), rb_big.low_water_slabs());
    EXPECT_EQ(rb_small.high_water_slabs(), rb_big.high_water_slabs());
    EXPECT_EQ(rb_small.min_band_slabs(), rb_big.min_band_slabs());
    EXPECT_EQ(rb_small.low_water_slabs(), static_cast<int>(2 * transient));
    EXPECT_EQ(rb_small.high_water_slabs(), static_cast<int>(4 * transient));

    // Contrast: the OLD pool-fraction shape would have reserved strictly more
    // on the bigger pool, stranding capacity precisely where there was most
    // of it going spare.
    EXPECT_LT(static_cast<int>(std::floor(0.30 * total_small)),
              static_cast<int>(std::floor(0.30 * total_big)));
    EXPECT_LT(rb_big.high_water_slabs(),
              static_cast<int>(std::floor(0.30 * total_big)))
        << "the absolute high mark must free capacity a 0.30 fraction held";
}

TEST(ExpertZoneRebalancer, LowWaterDerivedFromConcurrency) {
    // The low-water reserve exists so that every admission the engine may
    // have IN FLIGHT can land without waiting on a reclaim. So its multiplier
    // is the engine's own admission parallelism (serving.
    // max_concurrent_requests — the same worst-case multiplier the KV pool is
    // sized by), not a constant. The BAND is a different question — how far a
    // single admission swings the pool — so it does NOT scale with
    // concurrency.
    auto ctx = make_ctx(/*num_gpus=*/1, /*size_mult=*/4);
    const int total = total_slabs_of(*ctx);
    const auto geo = geometry_of(*ctx, 0.10);
    const int64_t transient = geo.grant_slabs(kMaxSlotsPerGrant, total);
    ASSERT_GT(transient, 0);
    const int64_t band = std::max<int64_t>(
        2 * transient, 2 * geo.grant_slabs(kMaxSlotsPerGrant, total));

    auto build = [&](Stubs& st, int concurrency) {
        auto d = rb_deps(*ctx, st);
        d.admission_transient_slabs = transient;
        d.max_concurrent_admissions = concurrency;
        return ld::ExpertZoneRebalancer(std::move(d),
                                        rb_cfg(/*low=*/0.0, /*high=*/0.0));
    };

    Stubs s1, s2, s4;
    auto rb1 = build(s1, 1);
    auto rb2 = build(s2, 2);
    auto rb4 = build(s4, 4);
    ASSERT_TRUE(rb1.enabled());
    ASSERT_TRUE(rb2.enabled());
    ASSERT_TRUE(rb4.enabled());

    // Low water is exactly concurrency x transient.
    EXPECT_EQ(rb1.low_water_slabs(), static_cast<int>(transient));
    EXPECT_EQ(rb2.low_water_slabs(), static_cast<int>(2 * transient));
    EXPECT_EQ(rb4.low_water_slabs(), static_cast<int>(4 * transient));

    // The band is a property of ONE admission's excursion — concurrency must
    // not inflate it.
    EXPECT_EQ(rb1.min_band_slabs(), band);
    EXPECT_EQ(rb2.min_band_slabs(), band);
    EXPECT_EQ(rb4.min_band_slabs(), band);

    // High rides on low, so it shifts by exactly the extra reserve.
    EXPECT_EQ(rb2.high_water_slabs() - rb1.high_water_slabs(),
              static_cast<int>(transient));
    EXPECT_EQ(rb4.high_water_slabs() - rb2.high_water_slabs(),
              static_cast<int>(2 * transient));
    EXPECT_EQ(rb4.high_water_slabs(),
              rb4.low_water_slabs() + static_cast<int>(band));

    // B=1-class serving (concurrency 2) reproduces the previously hardcoded
    // 2x, so the arms measured before this change remain valid.
    EXPECT_EQ(rb2.low_water_slabs(), static_cast<int>(2 * transient));
}

TEST(ExpertZoneRebalancer, GrantCooldownIsADurationNotATickCount) {
    // The cooldown must mean the same WALL-CLOCK window whatever the tick
    // interval is. Held as a tick count it silently shortened whenever
    // LS_KVXP_TICK_MS was lowered — quietly reintroducing the very
    // post-reclaim re-grant churn it exists to prevent.
    auto ctx = make_ctx();
    Stubs s1, s2, s3;

    // Same 3 s duration, three different tick intervals -> three different
    // tick counts, all denoting 3 s.
    auto slow = rb_cfg();
    slow.grant_cooldown_ms = 3000;
    slow.min_tick_interval_us = 200000;  // 200 ms
    ld::ExpertZoneRebalancer rb_slow(rb_deps(*ctx, s1), slow);
    EXPECT_EQ(rb_slow.grant_cooldown_ticks(), 15);  // 3000 / 200

    auto fast = rb_cfg();
    fast.grant_cooldown_ms = 3000;
    fast.min_tick_interval_us = 20000;  // 20 ms — 10x more ticks, same 3 s
    ld::ExpertZoneRebalancer rb_fast(rb_deps(*ctx, s2), fast);
    EXPECT_EQ(rb_fast.grant_cooldown_ticks(), 150);  // 3000 / 20
    EXPECT_GT(rb_fast.grant_cooldown_ticks(), rb_slow.grant_cooldown_ticks())
        << "a shorter tick must yield MORE ticks, not a shorter cooldown";

    // Unbounded tick rate (interval 0): no ms-per-tick to divide by, so a
    // tick counts as 1 ms. Deterministic rather than a division by zero.
    auto unbounded = rb_cfg();
    unbounded.grant_cooldown_ms = 3000;
    unbounded.min_tick_interval_us = 0;
    ld::ExpertZoneRebalancer rb_unbounded(rb_deps(*ctx, s3), unbounded);
    EXPECT_EQ(rb_unbounded.grant_cooldown_ticks(), 3000);

    // A sub-tick duration still costs a whole tick — it must never round away
    // to no cooldown at all.
    Stubs s4;
    auto tiny = rb_cfg();
    tiny.grant_cooldown_ms = 1;
    tiny.min_tick_interval_us = 200000;  // 200 ms per tick
    ld::ExpertZoneRebalancer rb_tiny(rb_deps(*ctx, s4), tiny);
    EXPECT_EQ(rb_tiny.grant_cooldown_ticks(), 1);

    // Zero means zero: explicitly disabled, not "one tick".
    Stubs s5;
    auto off = rb_cfg();  // rb_cfg pins grant_cooldown_ms = 0
    ld::ExpertZoneRebalancer rb_off(rb_deps(*ctx, s5), off);
    EXPECT_EQ(rb_off.grant_cooldown_ticks(), 0);
}

TEST(ExpertZoneRebalancer, GrantCooldownAfterReclaim) {
    // Band-edge damping: after handing slabs back, do NOT re-grant the span
    // on the very next tick just because free crossed the high mark again.
    auto ctx = make_ctx();
    Stubs s;
    auto cfg = rb_cfg();
    // The cooldown is a DURATION; the tick count is derived from it. rb_cfg
    // leaves min_tick_interval_us at 0 (every tick() is an effective pass),
    // which maps a tick to 1 ms — so 3 ms is exactly 3 effective ticks, and
    // the assertions below stay a deterministic tick count rather than a
    // wall-clock race.
    cfg.grant_cooldown_ms = 3;
    ld::ExpertZoneRebalancer rb(rb_deps(*ctx, s), cfg);

    rb.tick();
    ASSERT_EQ(rb.stats().grants, 1);

    // Drive a full reclaim.
    auto pins = pin_down_to(*ctx, rb.low_water_slabs() - 1);
    for (int i = 0; i < 8 && rb.stats().reclaims == 0; ++i) rb.tick();
    ASSERT_EQ(rb.stats().reclaims, 1);

    // Pool is comfortable again — a grant is now blocked ONLY by cooldown.
    for (auto& h : pins) ctx->pa->free(h);
    ASSERT_GT(free_slabs_min(*ctx), rb.high_water_slabs());

    rb.tick();
    EXPECT_EQ(rb.stats().grants, 1) << "cooldown tick 1 must not re-grant";
    rb.tick();
    EXPECT_EQ(rb.stats().grants, 1) << "cooldown tick 2 must not re-grant";
    rb.tick();
    EXPECT_EQ(rb.stats().grants, 2) << "grants resume once the cooldown ends";

    // The cooldown gates grants ONLY: a reclaim is never delayed by it.
    auto pins2 = pin_down_to(*ctx, rb.low_water_slabs() - 1);
    rb.tick();
    EXPECT_EQ(rb.stats().draining_zones, 1)
        << "pressure must start a drain immediately, cooldown or not";
    for (auto& h : pins2) ctx->pa->free(h);
}

TEST(ExpertZoneRebalancer, PerStepFloorMarks) {
    // TD-KVXP-PER-STEP-FLOOR. KV grows INCREMENTALLY — a decode step claims a
    // page per KV-bearing layer only when it crosses a page boundary — so the
    // standing reserve does not have to hold a whole admission's KV. It has to
    // hold the NON-DEFERRABLE upfront part (the mapped KDA state, claimed
    // whole at seq_create) plus ONE growth event of headroom per concurrent
    // admission, and the band shrinks to two growth events. The difference
    // against the legacy transient-derived marks is capacity DONATED back to
    // the expert cache.
    auto ctx = make_ctx(/*num_gpus=*/1, /*size_mult=*/8);
    const int total = total_slabs_of(*ctx);
    const auto geo = geometry_of(*ctx, 0.10);
    const int64_t two_grants = 2 * geo.grant_slabs(kMaxSlotsPerGrant, total);
    ASSERT_GT(two_grants, 0) << "fixture must admit at least one grant";

    // The measured arm's shape (upfront state ~578 slabs against a ~36-slab
    // growth event) scaled onto this pool. Derived from the pool rather than
    // hardcoded so the arithmetic stays exact — and the marks stay INSIDE the
    // pool — whatever slab geometry the fixture model produces.
    const int64_t transient = std::max<int64_t>(16, total / 8);
    const int64_t growth = std::max<int64_t>(2, transient / 13);
    ASSERT_LT(growth, transient) << "growth must be the SMALL term";

    Stubs s;
    auto d = rb_deps(*ctx, s);
    d.admission_transient_slabs = transient;
    d.per_step_growth_slabs = growth;
    d.max_concurrent_admissions = 2;
    ld::ExpertZoneRebalancer rb(std::move(d),
                                rb_cfg(/*low=*/0.0, /*high=*/0.0));
    ASSERT_TRUE(rb.enabled());

    // low = concurrency x (upfront state + one growth event).
    EXPECT_EQ(rb.low_water_slabs(),
              static_cast<int>(2 * (transient + growth)));
    // The band is TWO GROWTH EVENTS — deliberately narrower than one
    // admission's excursion, which is the whole point of the ticket.
    EXPECT_EQ(rb.min_band_slabs(), 2 * growth);
    EXPECT_EQ(rb.high_water_slabs(),
              rb.low_water_slabs() + static_cast<int>(2 * growth));

    // THE DONATION CLAIM. The legacy derivation holds low = 2 x transient with
    // a band of max(2 x transient, two grants); the per-step high mark must sit
    // well below that, and the gap is what the expert cache gets back.
    const int64_t legacy_high =
        2 * transient + std::max<int64_t>(2 * transient, two_grants);
    EXPECT_LT(rb.high_water_slabs(), static_cast<int>(legacy_high))
        << "the per-step floor must reserve strictly LESS than the legacy one";
    EXPECT_GE(legacy_high - rb.high_water_slabs(), transient)
        << "the donation must be a whole admission transient's worth, not a "
           "rounding difference";
}

TEST(ExpertZoneRebalancer, PerStepFloorFracFloorsStillRaise) {
    // The fractions remain optional FLOORS under the per-step derivation,
    // exactly as they are under the legacy one (FracFloorStillRaises): a user
    // who raises them gets the old pool-fraction reserve back.
    auto ctx = make_ctx(/*num_gpus=*/1, /*size_mult=*/8);
    const int total = total_slabs_of(*ctx);
    const int64_t transient = std::max<int64_t>(16, total / 8);
    const int64_t growth = std::max<int64_t>(2, transient / 13);

    Stubs s;
    auto d = rb_deps(*ctx, s);
    d.admission_transient_slabs = transient;
    d.per_step_growth_slabs = growth;
    d.max_concurrent_admissions = 2;
    // Both floors sit ABOVE the per-step derivation (whose low mark is about a
    // quarter of the pool), so both must engage.
    ld::ExpertZoneRebalancer rb(std::move(d),
                                rb_cfg(/*low=*/0.50, /*high=*/0.80));
    ASSERT_TRUE(rb.enabled());

    const auto frac_low = static_cast<int>(std::ceil(0.50 * total));
    const auto frac_high = static_cast<int>(std::ceil(0.80 * total));
    ASSERT_GT(frac_low, static_cast<int>(2 * (transient + growth)))
        << "the low floor must genuinely sit above the per-step derivation";

    EXPECT_EQ(rb.low_water_slabs(), frac_low);
    EXPECT_EQ(rb.high_water_slabs(), frac_high)
        << "an explicit high floor above the per-step mark must win";
    // A floor can only WIDEN the band: the per-step band floor still holds.
    EXPECT_EQ(rb.min_band_slabs(), 2 * growth);
    EXPECT_GE(rb.high_water_slabs() - rb.low_water_slabs(),
              static_cast<int>(rb.min_band_slabs()));
}

TEST(ExpertZoneRebalancer, PerStepFloorZeroGrowthFallsBackToLegacy) {
    // The growth term is ENGINE-WIRED, so an engine that does not supply it
    // (0 = unwired) must land on the pre-ticket marks VERBATIM — the per-step
    // derivation is an addition, never a silent change of the default.
    auto ctx = make_ctx(/*num_gpus=*/1, /*size_mult=*/8);
    const int total = total_slabs_of(*ctx);
    const auto geo = geometry_of(*ctx, 0.10);
    const int64_t two_grants = 2 * geo.grant_slabs(kMaxSlotsPerGrant, total);
    ASSERT_GT(two_grants, 0) << "fixture must admit at least one grant";
    const int64_t transient = std::max<int64_t>(16, total / 8);
    const int64_t growth = std::max<int64_t>(2, transient / 13);

    auto build = [&](Stubs& st, int64_t per_step) {
        auto d = rb_deps(*ctx, st);
        d.admission_transient_slabs = transient;
        d.per_step_growth_slabs = per_step;
        d.max_concurrent_admissions = 2;
        return ld::ExpertZoneRebalancer(std::move(d),
                                        rb_cfg(/*low=*/0.0, /*high=*/0.0));
    };

    Stubs s_off, s_on;
    auto rb_legacy = build(s_off, 0);
    auto rb_per_step = build(s_on, growth);
    ASSERT_TRUE(rb_legacy.enabled());
    ASSERT_TRUE(rb_per_step.enabled());

    // Legacy, spelled out the old way: low = concurrency x transient, band =
    // max(2 x transient, two grants), high = low + band.
    const int64_t legacy_band = std::max<int64_t>(2 * transient, two_grants);
    EXPECT_EQ(rb_legacy.low_water_slabs(), static_cast<int>(2 * transient));
    EXPECT_EQ(rb_legacy.min_band_slabs(), legacy_band);
    EXPECT_EQ(rb_legacy.high_water_slabs(),
              static_cast<int>(2 * transient + legacy_band));

    // Per-step, identical deps otherwise: the growth term enters BOTH terms.
    EXPECT_EQ(rb_per_step.low_water_slabs(),
              static_cast<int>(2 * (transient + growth)));
    EXPECT_EQ(rb_per_step.min_band_slabs(), 2 * growth);
    EXPECT_EQ(rb_per_step.high_water_slabs(),
              static_cast<int>(2 * (transient + growth) + 2 * growth));

    // Same transient, same concurrency, same pool — only the wiring differs,
    // and it moves the marks in the direction the ticket claims.
    EXPECT_GT(rb_per_step.low_water_slabs(), rb_legacy.low_water_slabs())
        << "the growth headroom is ADDED to the upfront-state reserve";
    EXPECT_LT(rb_per_step.min_band_slabs(), rb_legacy.min_band_slabs())
        << "the band is the term that shrinks";
    EXPECT_LT(rb_per_step.high_water_slabs(), rb_legacy.high_water_slabs())
        << "net effect: the pool holds strictly less hostage";
}

TEST(ExpertZoneRebalancer, PerStepFloorDisablesWhenMarksExceedPool) {
    // The over-pool check governs the per-step derivation too: marks that
    // cannot fit are refused outright rather than run in a shape known to
    // thrash. Here the upfront state alone already fills the pool, and the
    // per-admission growth headroom only makes it worse.
    auto ctx = make_ctx(/*num_gpus=*/1, /*size_mult=*/8);
    const int total = total_slabs_of(*ctx);

    Stubs s;
    auto d = rb_deps(*ctx, s);
    d.admission_transient_slabs = total;
    d.per_step_growth_slabs = std::max<int64_t>(2, total / 4);
    d.max_concurrent_admissions = 2;
    ld::ExpertZoneRebalancer rb(std::move(d),
                                rb_cfg(/*low=*/0.0, /*high=*/0.0));

    EXPECT_FALSE(rb.enabled())
        << "unsafe per-step marks must disable the rebalancer, not run anyway";
    rb.tick();
    EXPECT_EQ(rb.stats().grants, 0);
    EXPECT_EQ(ctx->pa->expert_zone_stats(0).grants, 0);
}

TEST(ExpertZoneRebalancer, TinyPoolGrantsBelowTheCapAsk) {
    // TD-KVXP-FAT-KV-ARM (2026-09-02): the GLM-5.2 champion boot
    // self-disabled with "no admissible grant of up to 8 slot(s) fits the
    // 154-slab pool" — but try_grant sizes its ask with best_fit(run) and
    // would have granted 1-6 admissible slots (slot/slab ratio 25.41). The
    // resolve-time viability check must size at the grant the pool can
    // ACTUALLY produce — min(cap, best_fit(total)) — not at the raw cap.
    auto ctx = make_ctx();
    const int total = total_slabs_of(*ctx);
    const int64_t slab = ctx->pa->slab_bytes();
    ASSERT_GE(total, 12);

    // A slot spanning ~55% of the pool: a ONE-slot grant fits, the
    // kMaxSlotsPerGrant-slot cap ask cannot — the champion shape in
    // miniature.
    const int64_t s1_slabs = std::max<int64_t>(3, (total * 55) / 100);
    lmem::ExpertZoneGeometry big{
        .expert_slot_bytes = s1_slabs * slab - lmem::kExpertZoneSlotAlign,
        .slab_bytes = slab,
        .max_waste = 0.10};
    ASSERT_EQ(big.grant_slabs(kMaxSlotsPerGrant, total), 0)
        << "fixture must reproduce the champion shape: the cap ask must "
           "NOT fit the pool";
    ASSERT_GT(big.grant_slabs(1, total), 0)
        << "a one-slot grant must fit, or the disable is correct";

    Stubs s;
    auto d = rb_deps(*ctx, s);
    d.geometry = big;
    d.admission_transient_slabs = 0;  // no mapped KDA state (GLM-5.2 shape)
    d.per_step_growth_slabs = 1;      // per-step floor: low 2, band 2, high 4
    d.max_concurrent_admissions = 2;
    ld::ExpertZoneRebalancer rb(std::move(d),
                                rb_cfg(/*low=*/0.0, /*high=*/0.0));
    ASSERT_TRUE(rb.enabled())
        << "a pool that admits a smaller-than-cap grant must not "
           "self-disable";
    EXPECT_EQ(rb.low_water_slabs(), 2);
    EXPECT_EQ(rb.high_water_slabs(), 4);

    const int base_stable =
        ctx->cache->total_slots(0, lmem::CacheZone::kStable);
    const int run0 = std::min(ctx->pa->largest_free_run(0),
                              free_slabs_min(*ctx) - rb.high_water_slabs());
    ASSERT_GT(run0, 0);

    rb.tick();

    const auto& st = rb.stats();
    ASSERT_EQ(st.grants, 1);
    EXPECT_EQ(st.granted_slots, 1) << "the pool-fitting ask is ONE slot";
    EXPECT_EQ(st.granted_slabs, big.grant_slabs(1, run0));
    EXPECT_EQ(ctx->cache->total_slots(0, lmem::CacheZone::kStable),
              base_stable + 1);
    EXPECT_GE(free_slabs_min(*ctx), rb.high_water_slabs());
    EXPECT_EQ(st.grant_refusals, 0);
}

// ── TD-KVXP-SCHEMA-KNOBS: config base + env override precedence ────────────

namespace {

/// Scoped save/clear/restore for the rebalancer's env surface so these
/// tests neither see nor leak ambient LS_* state.
class KvxpEnvGuard {
 public:
    KvxpEnvGuard() {
        for (const char* n : kVars) {
            const char* v = std::getenv(n);
            saved_.emplace_back(n, v ? std::optional<std::string>(v)
                                     : std::nullopt);
            ::unsetenv(n);
        }
    }
    ~KvxpEnvGuard() {
        for (const auto& [n, v] : saved_) {
            if (v) ::setenv(n, v->c_str(), 1);
            else   ::unsetenv(n);
        }
    }

 private:
    static constexpr const char* kVars[] = {
        "LS_KV_EXPERT_REBALANCE",     "LS_KVXP_MAX_WASTE",
        "LS_KVXP_LOW_FRAC",           "LS_KVXP_HIGH_FRAC",
        "LS_KVXP_MAX_SLOTS_PER_GRANT", "LS_KVXP_TICK_MS",
        "LS_KVXP_GRANT_COOLDOWN_MS",  "LS_KVXP_GRANT_COOLDOWN_TICKS"};
    std::vector<std::pair<const char*, std::optional<std::string>>> saved_;
};

}  // namespace

TEST(ExpertZoneRebalancerConfig, ConfigBaseHonoredWhenEnvUnset) {
    // The parsed _internal-kv_expert_rebalance section IS the policy when
    // no env var is set — a measured recipe can pin the rebalancer.
    KvxpEnvGuard guard;
    ld::ExpertZoneRebalancerConfig base;
    base.enabled = true;
    base.max_waste = 0.25;
    base.low_water_frac = 0.05;
    base.high_water_frac = 0.20;
    base.max_slots_per_grant = 4;
    base.min_tick_interval_us = 0;      // tick_ms 0: live, reachable value
    base.grant_cooldown_ms = 5000;
    const auto c = ld::ExpertZoneRebalancerConfig::from_config_env(base);
    EXPECT_TRUE(c.enabled);
    EXPECT_DOUBLE_EQ(c.max_waste, 0.25);
    EXPECT_DOUBLE_EQ(c.low_water_frac, 0.05);
    EXPECT_DOUBLE_EQ(c.high_water_frac, 0.20);
    EXPECT_EQ(c.max_slots_per_grant, 4);
    EXPECT_EQ(c.min_tick_interval_us, 0);
    EXPECT_EQ(c.grant_cooldown_ms, 5000);
}

TEST(ExpertZoneRebalancerConfig, EnvOverridesConfigEitherWay) {
    // A SET env var wins over the config value in BOTH directions (the
    // kda_state.mapped precedence): config-on + env-off => off, and
    // config-off + env-on => on.  Value knobs follow the same rule.
    KvxpEnvGuard guard;
    ld::ExpertZoneRebalancerConfig base;
    base.enabled = true;
    base.max_waste = 0.25;
    ::setenv("LS_KV_EXPERT_REBALANCE", "0", 1);
    ::setenv("LS_KVXP_MAX_WASTE", "0.5", 1);
    auto c = ld::ExpertZoneRebalancerConfig::from_config_env(base);
    EXPECT_FALSE(c.enabled) << "env off must beat config on";
    EXPECT_DOUBLE_EQ(c.max_waste, 0.5);
    ::setenv("LS_KV_EXPERT_REBALANCE", "1", 1);
    base.enabled = false;
    c = ld::ExpertZoneRebalancerConfig::from_config_env(base);
    EXPECT_TRUE(c.enabled) << "env on must beat config off";
}

TEST(ExpertZoneRebalancerConfig, FromEnvMatchesSchemaDefaults) {
    // from_env() (env-only boots, tests) is from_config_env over the
    // built-in defaults, which mirror the schema section's defaults.
    KvxpEnvGuard guard;
    const auto c = ld::ExpertZoneRebalancerConfig::from_env();
    EXPECT_TRUE(c.enabled);   // default ON since 2026-09-03 (44z promotion)
    EXPECT_DOUBLE_EQ(c.max_waste, 0.10);
    EXPECT_DOUBLE_EQ(c.low_water_frac, 0.02);
    EXPECT_DOUBLE_EQ(c.high_water_frac, 0.0);
    EXPECT_EQ(c.max_slots_per_grant, 8);
    EXPECT_EQ(c.min_tick_interval_us, 200000);
    EXPECT_EQ(c.grant_cooldown_ms, 30000);
}

TEST(ExpertZoneRebalancerConfig, ConfigValuesAreClampedLikeEnvValues) {
    // The documented-range clamps run on the MERGED value regardless of
    // source, so a config file cannot smuggle in what the env could not.
    KvxpEnvGuard guard;
    ld::ExpertZoneRebalancerConfig base;
    base.max_waste = 1.5;               // > 1.0
    base.max_slots_per_grant = 0;       // < 1
    const auto c = ld::ExpertZoneRebalancerConfig::from_config_env(base);
    EXPECT_DOUBLE_EQ(c.max_waste, 1.0);
    EXPECT_EQ(c.max_slots_per_grant, 1);
}
