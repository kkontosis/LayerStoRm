// ── 44z: ExpertCache elastic zones ──────────────────────────────────────────
//
// Elastic zones are runtime stable-capacity regions handed to the cache by the
// KV<->expert rebalancer, each backing onto a slab run granted from the shared
// KV pool. These tests pin the contract the rebalancer depends on: capacity
// accounting, the drain lifecycle, the address arithmetic (including the
// align_up(base, kExpertZoneSlotAlign) placement of slot 0), the generation
// counter used as a staleness discriminator by pointer-table consumers, and
// the byte-identity of the zero-zone baseline.
//
// The cache never dereferences a slot address, so the zones here are built
// over synthetic bases; only the arithmetic is under test.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "config/config_resolver.h"
#include "core/device_backend.h"
#include "core/memory/eviction_policy.h"
#include "core/memory/expert_cache.h"
#include "core/memory/expert_zone_math.h"
#include "core/memory/vram_allocator.h"
#include "core/null_device_backend.h"
#include "model/quantization/fp8.h"

namespace lc = layerstorm::config;
namespace lcomp = layerstorm::compute;
namespace lmem = layerstorm::memory;
namespace lmod = layerstorm::model;

namespace {

// Small MoE model: 2 GPUs, tiny VRAM — keeps the boot-carve fill loops short.
lc::Config small_moe_config() {
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
        {"quantization", {{"weights", "fp8_e4m3"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {
            {"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 1}},
                      {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 1}}}},
            {"tp_array", {0, 1}},
            {"system_ram_gb", 64}}},
        {"memory", {{"vram_safety_margin_gb", 0.1}}},
    };
    return lc::parse_config(j);
}

/// Build NullDeviceBackend instances for a VramLayout.
struct NullBackends {
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned;
    std::vector<lcomp::DeviceBackend*> ptrs;
    explicit NullBackends(const lmem::VramLayout& layout) {
        for (size_t i = 0; i < layout.gpus.size(); ++i) {
            lc::GpuRef ref{.position = static_cast<int>(i),
                           .id = layout.gpus[i].gpu_id};
            owned.push_back(lcomp::make_null_device_backend(ref));
            ptrs.push_back(owned.back().get());
        }
    }
};

struct TestContext {
    lc::Config cfg;
    lmod::ModelConfig mcfg;
    int64_t expert_bytes;
    NullBackends backends;
    lmem::VramAllocator vram;
    lmem::ExpertCache cache;
};

TestContext make_context() {
    auto cfg = small_moe_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);
    int64_t expert_bytes = reg.per_routed_expert_bytes();
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);
    auto cache = lmem::ExpertCache(vram, cfg, expert_bytes);
    return TestContext{std::move(cfg), std::move(mcfg), expert_bytes,
                       std::move(nb), std::move(vram), std::move(cache)};
}

lmem::ExpertKey key(uint32_t layer, uint16_t expert) {
    return {layer, expert};
}

/// Keys used only to fill the boot carve — layer >= 1000 keeps them clear of
/// every key a test asserts on.
lmem::ExpertKey filler_key(int i) {
    return key(static_cast<uint32_t>(1000 + i / 256),
               static_cast<uint16_t>(i % 256));
}

/// Reserve `count` stable slots, exhausting the boot carve.
void fill_stable(lmem::ExpertCache& cache, int gpu, int count) {
    for (int i = 0; i < count; ++i) {
        auto* addr = cache.reserve(filler_key(i), gpu, lmem::CacheZone::kStable);
        ASSERT_NE(addr, nullptr) << "boot-carve slot " << i << " should succeed";
    }
}

/// The geometry the rebalancer would use for this fixture's experts.
lmem::ExpertZoneGeometry geometry(int64_t expert_bytes) {
    return lmem::ExpertZoneGeometry{.expert_slot_bytes = expert_bytes,
                                    .slab_bytes = 272448,
                                    .max_waste = 0.10};
}

/// A deliberately slab-granular (NOT 4096-aligned) synthetic region base, so
/// the align_up(base, kExpertZoneSlotAlign) placement of slot 0 is exercised.
char* const kZoneBaseA = reinterpret_cast<char*>(0x10000040);
char* const kZoneBaseB = reinterpret_cast<char*>(0x20000000);

char* aligned_base(char* base) {
    const int64_t misalign = static_cast<int64_t>(
        reinterpret_cast<std::uintptr_t>(base)
        % static_cast<std::uintptr_t>(lmem::kExpertZoneSlotAlign));
    return base + (misalign == 0 ? 0 : lmem::kExpertZoneSlotAlign - misalign);
}

/// Region bytes generous enough to hold `num_slots` aligned slots.
int64_t region_bytes(int64_t stride, int num_slots) {
    return lmem::kExpertZoneSlotAlign + static_cast<int64_t>(num_slots) * stride;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// a. Capacity + address arithmetic
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheElastic, AddZoneExtendsStableCapacity) {
    auto ctx = make_context();
    const int gpu = 0;
    const int64_t stride = geometry(ctx.expert_bytes).slot_stride();
    constexpr int kSlots = 4;

    const int boot_total = ctx.cache.total_slots(gpu, lmem::CacheZone::kStable);
    const int boot_free = ctx.cache.free_slots(gpu, lmem::CacheZone::kStable);
    EXPECT_EQ(ctx.cache.elastic_zone_count(gpu), 0);

    const int zid = ctx.cache.add_elastic_zone(
        gpu, kZoneBaseA, region_bytes(stride, kSlots), stride, kSlots,
        /*cookie=*/0xBEEF);
    EXPECT_GE(zid, 0);
    EXPECT_EQ(ctx.cache.elastic_zone_count(gpu), 1);
    EXPECT_EQ(ctx.cache.elastic_zone_cookie(gpu, zid), 0xBEEF);

    // An active zone is ordinary stable capacity.
    EXPECT_EQ(ctx.cache.total_slots(gpu, lmem::CacheZone::kStable),
              boot_total + kSlots);
    EXPECT_EQ(ctx.cache.free_slots(gpu, lmem::CacheZone::kStable),
              boot_free + kSlots);
    // The other GPU is untouched.
    EXPECT_EQ(ctx.cache.elastic_zone_count(1), 0);

    // The boot carve is spent first...
    ASSERT_NO_FATAL_FAILURE(fill_stable(ctx.cache, gpu, boot_total));
    if (boot_total > 0) {
        const auto* last = ctx.cache.lookup(filler_key(boot_total - 1), gpu);
        ASSERT_NE(last, nullptr);
        EXPECT_EQ(last->elastic_zone, -1) << "boot-carve entries carry no zone";
    }
    EXPECT_EQ(ctx.cache.free_slots(gpu, lmem::CacheZone::kStable), kSlots);

    // ...then reserves land in the zone, slot 0 first, at the ALIGNED base.
    char* const abase = aligned_base(kZoneBaseA);
    for (int i = 0; i < kSlots; ++i) {
        auto* addr = ctx.cache.reserve(key(3, static_cast<uint16_t>(i)), gpu,
                                       lmem::CacheZone::kStable);
        ASSERT_NE(addr, nullptr) << "elastic slot " << i;
        EXPECT_EQ(addr, abase + static_cast<int64_t>(i) * stride);

        const auto* e = ctx.cache.lookup(key(3, static_cast<uint16_t>(i)), gpu);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->elastic_zone, zid);
        EXPECT_EQ(e->slot_idx, i) << "slot_idx is zone-local";
        EXPECT_EQ(e->zone, lmem::CacheZone::kStable)
            << "elastic residents are ordinary stable residents";
        EXPECT_FALSE(e->in_spill_zone);
        EXPECT_EQ(e->vram_address, addr);
    }

    // Zone full and boot carve full → the stable zone is full.
    EXPECT_EQ(ctx.cache.free_slots(gpu, lmem::CacheZone::kStable), 0);
    EXPECT_EQ(ctx.cache.used_slots(gpu, lmem::CacheZone::kStable),
              boot_total + kSlots);
    EXPECT_EQ(ctx.cache.reserve(key(4, 0), gpu, lmem::CacheZone::kStable),
              nullptr);

    // Elastic residents are visible through the ordinary residency queries.
    EXPECT_TRUE(ctx.cache.is_resident(key(3, 0), gpu));
    auto snap = ctx.cache.residency_snapshot(gpu);
    EXPECT_EQ(static_cast<int>(snap.size()), boot_total + kSlots);
}

// ═══════════════════════════════════════════════════════════════════════════
// b. Eviction returns the slot to its own zone
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheElastic, EvictReturnsSlotToZone) {
    auto ctx = make_context();
    const int gpu = 0;
    const int64_t stride = geometry(ctx.expert_bytes).slot_stride();
    constexpr int kSlots = 2;

    const int boot_total = ctx.cache.total_slots(gpu, lmem::CacheZone::kStable);
    const int zid = ctx.cache.add_elastic_zone(
        gpu, kZoneBaseA, region_bytes(stride, kSlots), stride, kSlots, 7);
    ASSERT_NO_FATAL_FAILURE(fill_stable(ctx.cache, gpu, boot_total));

    auto* addr = ctx.cache.reserve(key(3, 0), gpu, lmem::CacheZone::kStable);
    ASSERT_NE(addr, nullptr);
    ASSERT_EQ(ctx.cache.lookup(key(3, 0), gpu)->elastic_zone, zid);
    EXPECT_EQ(ctx.cache.free_slots(gpu, lmem::CacheZone::kStable), kSlots - 1);
    EXPECT_EQ(ctx.cache.elastic_drain_status(gpu, zid).residents, 1);

    // Ordinary evict() — metadata only.
    EXPECT_TRUE(ctx.cache.evict(key(3, 0), gpu));
    EXPECT_FALSE(ctx.cache.is_resident(key(3, 0), gpu));
    EXPECT_EQ(ctx.cache.free_slots(gpu, lmem::CacheZone::kStable), kSlots);
    EXPECT_EQ(ctx.cache.elastic_drain_status(gpu, zid).residents, 0);

    // The slot went back to the zone, not to the boot carve: re-reserving
    // hands out the very same zone address.
    auto* again = ctx.cache.reserve(key(3, 1), gpu, lmem::CacheZone::kStable);
    ASSERT_NE(again, nullptr);
    EXPECT_EQ(again, addr);
    EXPECT_EQ(ctx.cache.lookup(key(3, 1), gpu)->elastic_zone, zid);
}

// ═══════════════════════════════════════════════════════════════════════════
// c. Draining zones refuse new reserves
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheElastic, DrainStopsNewReserves) {
    auto ctx = make_context();
    const int gpu = 0;
    const int64_t stride = geometry(ctx.expert_bytes).slot_stride();
    constexpr int kSlots = 3;

    const int boot_total = ctx.cache.total_slots(gpu, lmem::CacheZone::kStable);
    const int zid = ctx.cache.add_elastic_zone(
        gpu, kZoneBaseA, region_bytes(stride, kSlots), stride, kSlots, 1);
    ASSERT_NO_FATAL_FAILURE(fill_stable(ctx.cache, gpu, boot_total));

    EXPECT_TRUE(ctx.cache.begin_drain_elastic_zone(gpu, zid));
    // Idempotent.
    EXPECT_TRUE(ctx.cache.begin_drain_elastic_zone(gpu, zid));

    // A draining zone is out of the capacity arithmetic entirely — occupancy
    // stats under-report during the drain window.
    EXPECT_EQ(ctx.cache.total_slots(gpu, lmem::CacheZone::kStable), boot_total);
    EXPECT_EQ(ctx.cache.free_slots(gpu, lmem::CacheZone::kStable), 0);

    // ...and it is never offered, even though all its slots are free.
    EXPECT_EQ(ctx.cache.reserve(key(3, 0), gpu, lmem::CacheZone::kStable),
              nullptr);
    EXPECT_FALSE(ctx.cache.is_resident(key(3, 0), gpu));
    EXPECT_EQ(ctx.cache.elastic_drain_status(gpu, zid).residents, 0);

    // Streaming reserves are unaffected by the stable-side drain.
    if (ctx.cache.total_slots(gpu, lmem::CacheZone::kStreaming) > 0) {
        auto* s = ctx.cache.reserve(key(3, 1), gpu, lmem::CacheZone::kStreaming);
        EXPECT_NE(s, nullptr);
        EXPECT_EQ(ctx.cache.lookup(key(3, 1), gpu)->elastic_zone, -1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// d. Drain status, resident listing, removal
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheElastic, DrainStatusAndRemove) {
    auto ctx = make_context();
    const int gpu = 0;
    const int64_t stride = geometry(ctx.expert_bytes).slot_stride();
    constexpr int kSlots = 2;

    // Unknown ids are inert.
    EXPECT_FALSE(ctx.cache.begin_drain_elastic_zone(gpu, 99));
    EXPECT_FALSE(ctx.cache.remove_elastic_zone(gpu, 99));
    EXPECT_EQ(ctx.cache.elastic_drain_status(gpu, 99).residents, 0);
    EXPECT_FALSE(ctx.cache.elastic_drain_status(gpu, 99).drained);
    EXPECT_TRUE(ctx.cache.elastic_zone_residents(gpu, 99).empty());
    EXPECT_EQ(ctx.cache.elastic_zone_cookie(gpu, 99), 0);

    const int boot_total = ctx.cache.total_slots(gpu, lmem::CacheZone::kStable);
    const int zid = ctx.cache.add_elastic_zone(
        gpu, kZoneBaseA, region_bytes(stride, kSlots), stride, kSlots, 42);
    ASSERT_NO_FATAL_FAILURE(fill_stable(ctx.cache, gpu, boot_total));

    ASSERT_NE(ctx.cache.reserve(key(3, 5), gpu, lmem::CacheZone::kStable),
              nullptr);
    ASSERT_NE(ctx.cache.reserve(key(3, 2), gpu, lmem::CacheZone::kStable),
              nullptr);

    auto st = ctx.cache.elastic_drain_status(gpu, zid);
    EXPECT_EQ(st.residents, 2);
    EXPECT_FALSE(st.drained);

    // Exactly the zone's keys, ascending — nothing from the boot carve.
    auto residents = ctx.cache.elastic_zone_residents(gpu, zid);
    ASSERT_EQ(residents.size(), 2u);
    EXPECT_EQ(residents[0], key(3, 2));
    EXPECT_EQ(residents[1], key(3, 5));

    EXPECT_TRUE(ctx.cache.begin_drain_elastic_zone(gpu, zid));
    // Draining does NOT evict: residents stay valid, dispatchable entries.
    EXPECT_TRUE(ctx.cache.is_resident(key(3, 5), gpu));
    EXPECT_EQ(ctx.cache.lookup(key(3, 5), gpu)->zone, lmem::CacheZone::kStable);
    EXPECT_EQ(ctx.cache.elastic_drain_status(gpu, zid).residents, 2);

    // Premature removal is refused.
    EXPECT_FALSE(ctx.cache.remove_elastic_zone(gpu, zid));
    EXPECT_EQ(ctx.cache.elastic_zone_count(gpu), 1);

    // The rebalancer's drain loop: evict the listed keys.
    for (const auto& k : residents) EXPECT_TRUE(ctx.cache.evict(k, gpu));

    st = ctx.cache.elastic_drain_status(gpu, zid);
    EXPECT_EQ(st.residents, 0);
    EXPECT_TRUE(st.drained);
    EXPECT_TRUE(ctx.cache.elastic_zone_residents(gpu, zid).empty());

    EXPECT_TRUE(ctx.cache.remove_elastic_zone(gpu, zid));
    EXPECT_EQ(ctx.cache.elastic_zone_count(gpu), 0);
    // Gone for good.
    EXPECT_FALSE(ctx.cache.remove_elastic_zone(gpu, zid));
    EXPECT_FALSE(ctx.cache.begin_drain_elastic_zone(gpu, zid));
    EXPECT_EQ(ctx.cache.total_slots(gpu, lmem::CacheZone::kStable), boot_total);
}

// ═══════════════════════════════════════════════════════════════════════════
// e. Generation counter
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheElastic, GenerationBumps) {
    auto ctx = make_context();
    const int gpu = 0;
    const int64_t stride = geometry(ctx.expert_bytes).slot_stride();

    const uint64_t g0 = ctx.cache.elastic_generation();

    // Adding capacity bumps: total_slots(kStable) grew, and capacity
    // consumers latch it off this counter (TD-KVXP-CAPACITY-REPUBLISH).
    const int zid = ctx.cache.add_elastic_zone(
        gpu, kZoneBaseA, region_bytes(stride, 1), stride, 1, 0);
    const uint64_t ga = ctx.cache.elastic_generation();
    EXPECT_GT(ga, g0);

    // Ordinary cache traffic does not bump.
    ctx.cache.reserve(key(3, 0), gpu, lmem::CacheZone::kStreaming);
    ctx.cache.promote(key(3, 0), gpu);
    ctx.cache.demote(key(3, 0), gpu);
    ctx.cache.evict(key(3, 0), gpu);
    EXPECT_EQ(ctx.cache.elastic_generation(), ga);

    // Retiring capacity bumps too.
    EXPECT_TRUE(ctx.cache.begin_drain_elastic_zone(gpu, zid));
    const uint64_t g1 = ctx.cache.elastic_generation();
    EXPECT_GT(g1, ga);

    EXPECT_TRUE(ctx.cache.remove_elastic_zone(gpu, zid));
    EXPECT_GT(ctx.cache.elastic_generation(), g1);

    // Failed calls on a now-unknown id leave the generation alone.
    const uint64_t g2 = ctx.cache.elastic_generation();
    EXPECT_FALSE(ctx.cache.remove_elastic_zone(gpu, zid));
    EXPECT_EQ(ctx.cache.elastic_generation(), g2);
}

// ═══════════════════════════════════════════════════════════════════════════
// f. Locked residents hold a drain open
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheElastic, LockedEntryBlocksDrainEviction) {
    auto ctx = make_context();
    const int gpu = 0;
    const int64_t stride = geometry(ctx.expert_bytes).slot_stride();

    const int boot_total = ctx.cache.total_slots(gpu, lmem::CacheZone::kStable);
    const int zid = ctx.cache.add_elastic_zone(
        gpu, kZoneBaseA, region_bytes(stride, 1), stride, 1, 0);
    ASSERT_NO_FATAL_FAILURE(fill_stable(ctx.cache, gpu, boot_total));

    ASSERT_NE(ctx.cache.reserve(key(3, 0), gpu, lmem::CacheZone::kStable),
              nullptr);
    ASSERT_EQ(ctx.cache.lookup(key(3, 0), gpu)->elastic_zone, zid);

    // #90 lock: in active use by progressive MoE.
    EXPECT_TRUE(ctx.cache.lock(key(3, 0), gpu));
    EXPECT_TRUE(ctx.cache.is_locked(key(3, 0), gpu));

    EXPECT_TRUE(ctx.cache.begin_drain_elastic_zone(gpu, zid));

    // The drain loop cannot reclaim it yet — and it stays dispatchable.
    EXPECT_FALSE(ctx.cache.evict(key(3, 0), gpu));
    EXPECT_TRUE(ctx.cache.is_resident(key(3, 0), gpu));
    auto st = ctx.cache.elastic_drain_status(gpu, zid);
    EXPECT_EQ(st.residents, 1);
    EXPECT_FALSE(st.drained);
    EXPECT_FALSE(ctx.cache.remove_elastic_zone(gpu, zid));

    // Once unlocked it becomes evictable and the drain completes.
    EXPECT_TRUE(ctx.cache.unlock(key(3, 0), gpu));
    EXPECT_TRUE(ctx.cache.evict(key(3, 0), gpu));
    EXPECT_TRUE(ctx.cache.elastic_drain_status(gpu, zid).drained);
    EXPECT_TRUE(ctx.cache.remove_elastic_zone(gpu, zid));
}

// ═══════════════════════════════════════════════════════════════════════════
// g. Zero-zone baseline is unchanged
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheElastic, BaselineByteIdentical) {
    // Two independently constructed caches over an identical layout, run one
    // after the other. The `probe` arm is asked about elastic state
    // throughout; the control arm never hears of it. With no zones registered
    // every outcome must match exactly.
    struct Trace {
        std::vector<int64_t> offsets;  // addresses relative to the first one.
        std::vector<int> flags;
        std::vector<int> counts;
    };

    auto run = [](lmem::ExpertCache& cache, bool probe) {
        Trace t;
        char* origin = nullptr;
        auto note_addr = [&](void* p) {
            if (p == nullptr) { t.offsets.push_back(-1); return; }
            if (origin == nullptr) origin = static_cast<char*>(p);
            t.offsets.push_back(static_cast<char*>(p) - origin);
        };

        for (int g = 0; g < cache.gpu_count(); ++g) {
            if (probe) {
                EXPECT_EQ(cache.elastic_zone_count(g), 0);
                EXPECT_EQ(cache.elastic_generation(), 0u);
            }
            t.counts.push_back(cache.total_slots(g, lmem::CacheZone::kStable));
            t.counts.push_back(cache.free_slots(g, lmem::CacheZone::kStable));
            t.counts.push_back(cache.total_slots(g, lmem::CacheZone::kStreaming));
            t.counts.push_back(cache.free_slots(g, lmem::CacheZone::kStreaming));
        }

        note_addr(cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable));
        note_addr(cache.reserve(key(3, 1), 0, lmem::CacheZone::kStreaming));
        note_addr(cache.reserve(key(3, 2), 0, lmem::CacheZone::kStable, true));

        t.flags.push_back(cache.promote(key(3, 1), 0));
        t.flags.push_back(cache.promote(key(3, 0), 0));   // already stable
        t.flags.push_back(cache.demote(key(3, 0), 0));
        t.flags.push_back(cache.demote(key(3, 0), 0));    // already streaming
        t.flags.push_back(cache.lock(key(3, 2), 0));
        t.flags.push_back(cache.evict(key(3, 2), 0));     // locked → refused
        t.flags.push_back(cache.unlock(key(3, 2), 0));
        t.flags.push_back(cache.evict(key(3, 2), 0));
        t.flags.push_back(cache.evict(key(3, 0), 0));
        t.flags.push_back(cache.evict(key(9, 9), 0));     // not resident

        // Every entry created on the no-zone path carries no zone.
        for (int g = 0; g < cache.gpu_count(); ++g) {
            for (const auto& info : cache.residency_snapshot(g)) {
                const auto* e = cache.lookup(info.key, g);
                t.flags.push_back(e != nullptr ? e->elastic_zone : -999);
            }
        }

        t.counts.push_back(cache.total_resident());
        t.counts.push_back(cache.duplicate_count(0));
        for (int g = 0; g < cache.gpu_count(); ++g) {
            t.counts.push_back(cache.total_slots(g, lmem::CacheZone::kStable));
            t.counts.push_back(cache.free_slots(g, lmem::CacheZone::kStable));
            t.counts.push_back(cache.total_slots(g, lmem::CacheZone::kStreaming));
            t.counts.push_back(cache.free_slots(g, lmem::CacheZone::kStreaming));
        }
        return t;
    };

    Trace probed;
    {
        auto a = make_context();
        probed = run(a.cache, /*probe=*/true);
        // Nothing in the sequence touched elastic state.
        EXPECT_EQ(a.cache.elastic_generation(), 0u);
        EXPECT_EQ(a.cache.elastic_zone_count(0), 0);
    }
    Trace control;
    {
        auto b = make_context();
        control = run(b.cache, /*probe=*/false);
    }

    EXPECT_EQ(probed.offsets, control.offsets);
    EXPECT_EQ(probed.flags, control.flags);
    EXPECT_EQ(probed.counts, control.counts);
    for (const int f : probed.flags) EXPECT_NE(f, -999);
}

// ═══════════════════════════════════════════════════════════════════════════
// h. Multiple zones are offered in id order
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheElastic, TwoZonesIdOrder) {
    auto ctx = make_context();
    const int gpu = 0;
    const int64_t stride = geometry(ctx.expert_bytes).slot_stride();

    const int boot_total = ctx.cache.total_slots(gpu, lmem::CacheZone::kStable);
    const int z0 = ctx.cache.add_elastic_zone(
        gpu, kZoneBaseA, region_bytes(stride, 1), stride, 1, /*cookie=*/100);
    const int z1 = ctx.cache.add_elastic_zone(
        gpu, kZoneBaseB, region_bytes(stride, 1), stride, 1, /*cookie=*/200);
    EXPECT_LT(z0, z1) << "ids are monotonically assigned";
    EXPECT_EQ(ctx.cache.elastic_zone_count(gpu), 2);
    EXPECT_EQ(ctx.cache.elastic_zone_cookie(gpu, z0), 100);
    EXPECT_EQ(ctx.cache.elastic_zone_cookie(gpu, z1), 200);
    EXPECT_EQ(ctx.cache.total_slots(gpu, lmem::CacheZone::kStable),
              boot_total + 2);

    ASSERT_NO_FATAL_FAILURE(fill_stable(ctx.cache, gpu, boot_total));

    // Overflow lands in the LOWER id first.
    auto* a0 = ctx.cache.reserve(key(3, 0), gpu, lmem::CacheZone::kStable);
    ASSERT_NE(a0, nullptr);
    EXPECT_EQ(a0, aligned_base(kZoneBaseA));
    EXPECT_EQ(ctx.cache.lookup(key(3, 0), gpu)->elastic_zone, z0);

    auto* a1 = ctx.cache.reserve(key(3, 1), gpu, lmem::CacheZone::kStable);
    ASSERT_NE(a1, nullptr);
    EXPECT_EQ(a1, aligned_base(kZoneBaseB));
    EXPECT_EQ(ctx.cache.lookup(key(3, 1), gpu)->elastic_zone, z1);

    // Free both, drain z0: the survivor keeps serving, z0 is skipped.
    EXPECT_TRUE(ctx.cache.evict(key(3, 0), gpu));
    EXPECT_TRUE(ctx.cache.evict(key(3, 1), gpu));
    EXPECT_TRUE(ctx.cache.begin_drain_elastic_zone(gpu, z0));
    EXPECT_EQ(ctx.cache.total_slots(gpu, lmem::CacheZone::kStable),
              boot_total + 1);

    auto* a2 = ctx.cache.reserve(key(3, 2), gpu, lmem::CacheZone::kStable);
    ASSERT_NE(a2, nullptr);
    EXPECT_EQ(a2, aligned_base(kZoneBaseB)) << "draining z0 must be skipped";
    EXPECT_EQ(ctx.cache.lookup(key(3, 2), gpu)->elastic_zone, z1);

    // z1 now full, z0 draining → stable is full.
    EXPECT_EQ(ctx.cache.reserve(key(3, 3), gpu, lmem::CacheZone::kStable),
              nullptr);

    // z0 has no residents, so it can retire immediately.
    EXPECT_TRUE(ctx.cache.elastic_drain_status(gpu, z0).drained);
    EXPECT_TRUE(ctx.cache.remove_elastic_zone(gpu, z0));
    EXPECT_EQ(ctx.cache.elastic_zone_count(gpu), 1);
    EXPECT_EQ(ctx.cache.elastic_zone_cookie(gpu, z1), 200)
        << "removing z0 must not disturb z1";
    EXPECT_EQ(ctx.cache.lookup(key(3, 2), gpu)->elastic_zone, z1);
    EXPECT_EQ(ctx.cache.lookup(key(3, 2), gpu)->vram_address, a2);
}
