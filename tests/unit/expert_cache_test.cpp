#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "config/config_resolver.h"
#include "core/device_backend.h"
#include "core/memory/eviction_policy.h"
#include "core/memory/expert_cache.h"
#include "core/memory/vram_allocator.h"
#include "core/null_device_backend.h"
#include "model/quantization/fp8.h"
#include "model/quantization/nvfp4.h"

namespace lc = layerstorm::config;
namespace lcomp = layerstorm::compute;
namespace lmem = layerstorm::memory;
namespace lmod = layerstorm::model;

// ── Config helpers ──────────────────────────────────────────────────────────

namespace {

lc::Config v32_config() {
    auto j = nlohmann::json{
        {"model", {
            {"architecture",            "deepseek_v3"},
            {"weights_path",            "/data/models/deepseek-v3.2/"},
            {"weights_format",          "safetensors"},
            {"num_hidden_layers",       61},
            {"hidden_size",             7168},
            {"num_attention_heads",     128},
            {"num_key_value_heads",     128},
            {"intermediate_size",       18432},
            {"n_routed_experts",        256},
            {"n_shared_experts",        1},
            {"num_experts_per_tok",     8},
            {"n_group",                 8},
            {"topk_group",              4},
            {"vocab_size",              129280},
            {"max_position_embeddings", 163840},
            {"kv_lora_rank",            512},
            {"q_lora_rank",             1536},
            {"qk_rope_head_dim",        64},
            {"qk_nope_head_dim",        128},
            {"v_head_dim",              128},
            {"first_k_dense_replace",   3},
            {"moe_layer_freq",          1},
            {"index_topk",              2048},
            {"index_n_heads",           64},
            {"index_head_dim",          128},
            {"num_nextn_predict_layers", 1},
            {"rms_norm_eps",            1e-6},
            {"rope_theta",              10000.0},
            {"routed_scaling_factor",   2.5},
            {"moe_intermediate_size",   2048},
        }},
        {"quantization", {{"weights", "nvfp4"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"serving", {
            {"max_sequence_length", 4096},
            {"max_concurrent_requests", 4}}},
        {"hardware", {
            {"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}},
                      {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 32}},
                      {{"id", 2}, {"type", "rtx5080"}, {"vram_gb", 16}},
                      {{"id", 3}, {"type", "rtx5080"}, {"vram_gb", 16}}}},
            {"tp_array", {0, 1}},
            {"system_ram_gb", 256}}},
    };
    return lc::parse_config(j);
}

// Smaller model for focused tests: 2 GPUs, MoE layers, tiny VRAM.
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

TestContext make_v32_context() {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    int64_t expert_bytes = reg.per_routed_expert_bytes();
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);
    auto cache = lmem::ExpertCache(vram, cfg, expert_bytes);
    return TestContext{std::move(cfg), std::move(mcfg), expert_bytes,
                       std::move(nb), std::move(vram), std::move(cache)};
}

TestContext make_small_context() {
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

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCache, ConstructionV32) {
    auto ctx = make_v32_context();
    EXPECT_EQ(ctx.cache.gpu_count(), 4);
    EXPECT_EQ(ctx.cache.expert_bytes(), ctx.expert_bytes);
    EXPECT_EQ(ctx.cache.total_resident(), 0);

    // Each GPU should have some expert cache slots.
    for (int g = 0; g < ctx.cache.gpu_count(); ++g) {
        int stable = ctx.cache.total_slots(g, lmem::CacheZone::kStable);
        int streaming = ctx.cache.total_slots(g, lmem::CacheZone::kStreaming);
        EXPECT_GT(stable + streaming, 0)
            << "GPU " << g << " should have expert cache slots";
        EXPECT_EQ(ctx.cache.free_slots(g, lmem::CacheZone::kStable), stable);
        EXPECT_EQ(ctx.cache.free_slots(g, lmem::CacheZone::kStreaming), streaming);
    }
}

TEST(ExpertCache, ConstructionSmall) {
    auto ctx = make_small_context();
    EXPECT_EQ(ctx.cache.gpu_count(), 2);
    EXPECT_EQ(ctx.cache.expert_bytes(), ctx.expert_bytes);
}

TEST(ExpertCache, ConstructionStableStreamingSplit) {
    auto ctx = make_v32_context();
    for (int g = 0; g < ctx.cache.gpu_count(); ++g) {
        const auto& gpu_layout = ctx.vram.layout().gpus[g];
        int expected_stable = static_cast<int>(
            gpu_layout.expert_stable_bytes / ctx.expert_bytes);
        // Streaming total = spill_slots + prefetch_slots (each floor-divided)
        int expected_spill = static_cast<int>(
            gpu_layout.streaming_spill_bytes / ctx.expert_bytes);
        int expected_prefetch = static_cast<int>(
            gpu_layout.streaming_prefetch_bytes / ctx.expert_bytes);
        EXPECT_EQ(ctx.cache.total_slots(g, lmem::CacheZone::kStable),
                  expected_stable);
        EXPECT_EQ(ctx.cache.total_slots(g, lmem::CacheZone::kStreaming),
                  expected_spill + expected_prefetch);
    }
}

TEST(ExpertCache, ConstructionWithQuantInterface) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    int64_t expert_bytes = reg.per_routed_expert_bytes();
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);

    lmod::ExpertShape shape{mcfg.raw().hidden_size, mcfg.raw().moe_intermediate_size};
    auto cache = lmem::ExpertCache(vram, cfg, expert_bytes, nvfp4, shape);

    // Verify the cache constructed successfully with precise offsets.
    EXPECT_EQ(cache.gpu_count(), 4);
    EXPECT_EQ(cache.expert_bytes(), expert_bytes);

    // Reserve an expert and check that projection offsets are set from quant.
    auto* addr = cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ASSERT_NE(addr, nullptr);
    auto* entry = cache.lookup(key(3, 0), 0);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->gate_offset, 0);
    EXPECT_EQ(entry->up_offset,
              nvfp4.bytes_per_projection(shape, lmod::Projection::gate));
    EXPECT_EQ(entry->down_offset,
              nvfp4.bytes_per_projection(shape, lmod::Projection::gate) +
              nvfp4.bytes_per_projection(shape, lmod::Projection::up));
}

// ═══════════════════════════════════════════════════════════════════════════
// Reserve & Lookup
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCache, ReserveBasicStable) {
    auto ctx = make_v32_context();
    auto* addr = ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStable);
    ASSERT_NE(addr, nullptr);
    EXPECT_TRUE(ctx.cache.is_resident(key(3, 10), 0));
    EXPECT_TRUE(ctx.cache.is_resident(key(3, 10)));

    auto* entry = ctx.cache.lookup(key(3, 10), 0);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->key, key(3, 10));
    EXPECT_EQ(entry->gpu_idx, 0);
    EXPECT_EQ(entry->zone, lmem::CacheZone::kStable);
    EXPECT_FALSE(entry->is_duplicate);
    EXPECT_EQ(entry->sub_components_ready, 0);
    EXPECT_EQ(entry->vram_address, addr);
}

TEST(ExpertCache, ReserveBasicStreaming) {
    auto ctx = make_v32_context();
    auto* addr = ctx.cache.reserve(key(5, 42), 2, lmem::CacheZone::kStreaming);
    ASSERT_NE(addr, nullptr);
    EXPECT_TRUE(ctx.cache.is_resident(key(5, 42), 2));

    auto* entry = ctx.cache.lookup(key(5, 42), 2);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->zone, lmem::CacheZone::kStreaming);
}

TEST(ExpertCache, ReserveReturnedAddressInRange) {
    auto ctx = make_v32_context();
    int gpu = 0;
    auto* addr = ctx.cache.reserve(key(3, 0), gpu, lmem::CacheZone::kStable);
    ASSERT_NE(addr, nullptr);

    // Address should be within the expert_stable region.
    auto* stable_base = static_cast<char*>(ctx.vram.region(gpu).expert_stable);
    int64_t stable_size = ctx.vram.layout().gpus[gpu].expert_stable_bytes;
    auto* addr_c = static_cast<char*>(addr);
    EXPECT_GE(addr_c, stable_base);
    EXPECT_LT(addr_c, stable_base + stable_size);
}

TEST(ExpertCache, ReserveMultipleSameGpu) {
    auto ctx = make_v32_context();
    std::unordered_set<void*> addrs;
    for (uint16_t e = 0; e < 5; ++e) {
        auto* addr = ctx.cache.reserve(key(3, e), 0, lmem::CacheZone::kStable);
        ASSERT_NE(addr, nullptr);
        EXPECT_TRUE(addrs.insert(addr).second) << "Addresses must be distinct";
    }
    EXPECT_EQ(ctx.cache.used_slots(0, lmem::CacheZone::kStable), 5);
}

TEST(ExpertCache, ReserveSameExpertDifferentGpus) {
    auto ctx = make_v32_context();
    auto* addr0 = ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStable);
    auto* addr2 = ctx.cache.reserve(key(3, 10), 2, lmem::CacheZone::kStable,
                                    /*is_duplicate=*/true);
    ASSERT_NE(addr0, nullptr);
    ASSERT_NE(addr2, nullptr);
    EXPECT_NE(addr0, addr2);
    EXPECT_TRUE(ctx.cache.is_resident(key(3, 10), 0));
    EXPECT_TRUE(ctx.cache.is_resident(key(3, 10), 2));
}

TEST(ExpertCache, ReserveFullZoneReturnsNull) {
    auto ctx = make_small_context();
    int total = ctx.cache.total_slots(0, lmem::CacheZone::kStreaming);

    // Fill all streaming slots using (layer, expert) pairs to avoid uint16_t overflow.
    for (int i = 0; i < total; ++i) {
        uint32_t layer = static_cast<uint32_t>(i / 256);
        uint16_t expert = static_cast<uint16_t>(i % 256);
        auto* addr = ctx.cache.reserve(
            key(layer, expert), 0, lmem::CacheZone::kStreaming);
        ASSERT_NE(addr, nullptr) << "Slot " << i << " should succeed";
    }

    // Next reserve should fail.
    auto* addr = ctx.cache.reserve(
        key(9999, 0), 0, lmem::CacheZone::kStreaming);
    EXPECT_EQ(addr, nullptr);
}

TEST(ExpertCache, ReserveAlreadyResidentReturnsNull) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStable);
    auto* addr = ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStreaming);
    EXPECT_EQ(addr, nullptr);
}

TEST(ExpertCache, ReserveDuplicateFlag) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 10), 2, lmem::CacheZone::kStable, true);

    auto* primary = ctx.cache.lookup(key(3, 10), 0);
    auto* dup = ctx.cache.lookup(key(3, 10), 2);
    EXPECT_FALSE(primary->is_duplicate);
    EXPECT_TRUE(dup->is_duplicate);
}

// ═══════════════════════════════════════════════════════════════════════════
// Mark Ready
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCache, MarkReadySingleComponent) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.mark_ready(key(3, 0), 0, lmem::kGate);
    auto* e = ctx.cache.lookup(key(3, 0), 0);
    EXPECT_EQ(e->sub_components_ready, lmem::kGate);
}

TEST(ExpertCache, MarkReadyProgressive) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);

    ctx.cache.mark_ready(key(3, 0), 0, lmem::kGate);
    EXPECT_EQ(ctx.cache.lookup(key(3, 0), 0)->sub_components_ready, lmem::kGate);

    ctx.cache.mark_ready(key(3, 0), 0, lmem::kUp);
    EXPECT_EQ(ctx.cache.lookup(key(3, 0), 0)->sub_components_ready,
              lmem::kGate | lmem::kUp);

    ctx.cache.mark_ready(key(3, 0), 0, lmem::kDown);
    EXPECT_EQ(ctx.cache.lookup(key(3, 0), 0)->sub_components_ready, lmem::kAll);
}

TEST(ExpertCache, MarkAllReady) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.mark_all_ready(key(3, 0), 0);
    EXPECT_EQ(ctx.cache.lookup(key(3, 0), 0)->sub_components_ready, lmem::kAll);
}

TEST(ExpertCache, MarkReadyGatePlusUp) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.mark_ready(key(3, 0), 0, lmem::kGate);
    ctx.cache.mark_ready(key(3, 0), 0, lmem::kUp);
    auto ready = ctx.cache.lookup(key(3, 0), 0)->sub_components_ready;
    EXPECT_TRUE(ready & lmem::kGate);
    EXPECT_TRUE(ready & lmem::kUp);
    EXPECT_FALSE(ready & lmem::kDown);
}

TEST(ExpertCache, MarkReadyIdempotent) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.mark_ready(key(3, 0), 0, lmem::kGate);
    ctx.cache.mark_ready(key(3, 0), 0, lmem::kGate);
    EXPECT_EQ(ctx.cache.lookup(key(3, 0), 0)->sub_components_ready, lmem::kGate);
}

// ═══════════════════════════════════════════════════════════════════════════
// Eviction
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCache, EvictBasic) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStable);
    EXPECT_TRUE(ctx.cache.is_resident(key(3, 10), 0));

    EXPECT_TRUE(ctx.cache.evict(key(3, 10), 0));
    EXPECT_FALSE(ctx.cache.is_resident(key(3, 10), 0));
    EXPECT_FALSE(ctx.cache.is_resident(key(3, 10)));
    EXPECT_EQ(ctx.cache.lookup(key(3, 10), 0), nullptr);
}

TEST(ExpertCache, EvictNonResident) {
    auto ctx = make_v32_context();
    EXPECT_FALSE(ctx.cache.evict(key(99, 99), 0));
}

TEST(ExpertCache, EvictFreesSlot) {
    auto ctx = make_v32_context();
    int before = ctx.cache.free_slots(0, lmem::CacheZone::kStreaming);
    ASSERT_GT(before, 0) << "Need at least 1 streaming slot";

    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStreaming);
    EXPECT_EQ(ctx.cache.free_slots(0, lmem::CacheZone::kStreaming), before - 1);

    // Evict one — slot returns to free list.
    ctx.cache.evict(key(3, 0), 0);
    EXPECT_EQ(ctx.cache.free_slots(0, lmem::CacheZone::kStreaming), before);

    // Can re-reserve into the freed slot.
    auto* addr = ctx.cache.reserve(key(5, 0), 0, lmem::CacheZone::kStreaming);
    EXPECT_NE(addr, nullptr);
    EXPECT_EQ(ctx.cache.free_slots(0, lmem::CacheZone::kStreaming), before - 1);
}

TEST(ExpertCache, EvictDuplicate) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 10), 2, lmem::CacheZone::kStable, true);
    EXPECT_EQ(ctx.cache.duplicate_count(2), 1);

    ctx.cache.evict(key(3, 10), 2);
    EXPECT_EQ(ctx.cache.duplicate_count(2), 0);
    EXPECT_TRUE(ctx.cache.is_resident(key(3, 10), 0));  // Primary unaffected.
}

TEST(ExpertCache, EvictCoherenceAfterEvict) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 20), 0, lmem::CacheZone::kStable);
    ctx.cache.mark_all_ready(key(3, 20), 0);

    ctx.cache.evict(key(3, 10), 0);

    // Expert 20 should be completely unaffected.
    EXPECT_TRUE(ctx.cache.is_resident(key(3, 20), 0));
    auto* e = ctx.cache.lookup(key(3, 20), 0);
    EXPECT_EQ(e->sub_components_ready, lmem::kAll);
}

TEST(ExpertCache, EvictAllSlots) {
    auto ctx = make_small_context();
    int total = ctx.cache.total_slots(0, lmem::CacheZone::kStreaming);

    for (int i = 0; i < total; ++i) {
        uint32_t layer = static_cast<uint32_t>(i / 256);
        uint16_t expert = static_cast<uint16_t>(i % 256);
        ctx.cache.reserve(key(layer, expert), 0, lmem::CacheZone::kStreaming);
    }
    for (int i = 0; i < total; ++i) {
        uint32_t layer = static_cast<uint32_t>(i / 256);
        uint16_t expert = static_cast<uint16_t>(i % 256);
        EXPECT_TRUE(ctx.cache.evict(key(layer, expert), 0));
    }
    EXPECT_EQ(ctx.cache.free_slots(0, lmem::CacheZone::kStreaming), total);
    EXPECT_EQ(ctx.cache.total_resident(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Zone Promote/Demote
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCache, PromoteStreamingToStable) {
    auto ctx = make_v32_context();
    auto* addr = ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStreaming);
    EXPECT_TRUE(ctx.cache.promote(key(3, 10), 0));

    auto* e = ctx.cache.lookup(key(3, 10), 0);
    EXPECT_EQ(e->zone, lmem::CacheZone::kStable);
    EXPECT_EQ(e->vram_address, addr);  // No data movement.
}

TEST(ExpertCache, PromoteAlreadyStable) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStable);
    EXPECT_FALSE(ctx.cache.promote(key(3, 10), 0));
}

TEST(ExpertCache, DemoteStableToStreaming) {
    auto ctx = make_v32_context();
    auto* addr = ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStable);
    EXPECT_TRUE(ctx.cache.demote(key(3, 10), 0));

    auto* e = ctx.cache.lookup(key(3, 10), 0);
    EXPECT_EQ(e->zone, lmem::CacheZone::kStreaming);
    EXPECT_EQ(e->vram_address, addr);  // No data movement.
}

TEST(ExpertCache, DemoteAlreadyStreaming) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStreaming);
    EXPECT_FALSE(ctx.cache.demote(key(3, 10), 0));
}

TEST(ExpertCache, PromoteDemoteRoundTrip) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStreaming);
    EXPECT_TRUE(ctx.cache.promote(key(3, 10), 0));
    EXPECT_EQ(ctx.cache.lookup(key(3, 10), 0)->zone, lmem::CacheZone::kStable);
    EXPECT_TRUE(ctx.cache.demote(key(3, 10), 0));
    EXPECT_EQ(ctx.cache.lookup(key(3, 10), 0)->zone, lmem::CacheZone::kStreaming);
}

// ═══════════════════════════════════════════════════════════════════════════
// Eviction Inputs
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCache, EvictionInputsAllGpu) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 1), 0, lmem::CacheZone::kStreaming);
    ctx.cache.reserve(key(5, 0), 0, lmem::CacheZone::kStable, true);

    auto inputs = ctx.cache.eviction_inputs(0);
    EXPECT_EQ(inputs.size(), 3u);

    // All scoring terms should be zero (caller fills).
    for (const auto& inp : inputs) {
        EXPECT_DOUBLE_EQ(inp.recency, 0.0);
        EXPECT_DOUBLE_EQ(inp.frequency, 0.0);
        EXPECT_DOUBLE_EQ(inp.routing_weight, 0.0);
        EXPECT_EQ(inp.gpu_idx, 0);
    }
}

TEST(ExpertCache, EvictionInputsZoneFiltered) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 1), 0, lmem::CacheZone::kStreaming);
    ctx.cache.reserve(key(3, 2), 0, lmem::CacheZone::kStreaming);

    auto streaming_inputs = ctx.cache.eviction_inputs(0, lmem::CacheZone::kStreaming);
    EXPECT_EQ(streaming_inputs.size(), 2u);
    for (const auto& inp : streaming_inputs) {
        EXPECT_EQ(inp.zone, lmem::CacheZone::kStreaming);
    }

    auto stable_inputs = ctx.cache.eviction_inputs(0, lmem::CacheZone::kStable);
    EXPECT_EQ(stable_inputs.size(), 1u);
    EXPECT_EQ(stable_inputs[0].zone, lmem::CacheZone::kStable);
}

TEST(ExpertCache, EvictionInputsDuplicateFlag) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 0), 2, lmem::CacheZone::kStable, true);

    auto inputs_gpu0 = ctx.cache.eviction_inputs(0);
    EXPECT_EQ(inputs_gpu0.size(), 1u);
    EXPECT_FALSE(inputs_gpu0[0].is_duplicate);

    auto inputs_gpu2 = ctx.cache.eviction_inputs(2);
    EXPECT_EQ(inputs_gpu2.size(), 1u);
    EXPECT_TRUE(inputs_gpu2[0].is_duplicate);
}

TEST(ExpertCache, EvictionInputsEmptyGpu) {
    auto ctx = make_v32_context();
    auto inputs = ctx.cache.eviction_inputs(3);
    EXPECT_TRUE(inputs.empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Duplication
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCache, DuplicationBenefitPositive) {
    lmem::DuplicationInput inp;
    inp.frequency = 0.9;
    inp.avg_cross_gpu_latency = 0.001;  // 1ms
    inp.vram_cost = 0.01;
    inp.eviction_pressure = 0.3;
    double benefit = lmem::ExpertCache::duplication_benefit(inp);
    // 0.9 * 0.001 - 0.01 * 0.3 = 0.0009 - 0.003 = -0.0021 (actually negative)
    // Let's use higher latency for positive result.
    inp.avg_cross_gpu_latency = 0.1;
    benefit = lmem::ExpertCache::duplication_benefit(inp);
    // 0.9 * 0.1 - 0.01 * 0.3 = 0.09 - 0.003 = 0.087
    EXPECT_GT(benefit, 0.0);
}

TEST(ExpertCache, DuplicationBenefitNegative) {
    lmem::DuplicationInput inp;
    inp.frequency = 0.1;
    inp.avg_cross_gpu_latency = 0.001;
    inp.vram_cost = 0.5;
    inp.eviction_pressure = 0.9;
    double benefit = lmem::ExpertCache::duplication_benefit(inp);
    // 0.1 * 0.001 - 0.5 * 0.9 = 0.0001 - 0.45 = -0.4499
    EXPECT_LT(benefit, 0.0);
}

TEST(ExpertCache, CanDuplicateUnderLimit) {
    auto ctx = make_v32_context();
    // No duplicates yet; should be allowed.
    EXPECT_TRUE(ctx.cache.can_duplicate(0));
}

TEST(ExpertCache, CanDuplicateAtLimit) {
    auto ctx = make_v32_context();
    int total_stable = ctx.cache.total_slots(0, lmem::CacheZone::kStable);
    int total_streaming = ctx.cache.total_slots(0, lmem::CacheZone::kStreaming);
    int total = total_stable + total_streaming;
    double max_frac = ctx.cfg.parallelism.expert_duplication.max_duplicated_fraction;

    // Fill duplicates until can_duplicate returns false.
    ASSERT_TRUE(ctx.cache.can_duplicate(0));
    int added = 0;
    while (ctx.cache.can_duplicate(0) && added < total) {
        uint32_t layer = static_cast<uint32_t>(added / 256);
        uint16_t expert = static_cast<uint16_t>(added % 256);
        // Primary on GPU 2, duplicate on GPU 0.
        ctx.cache.reserve(key(layer, expert), 2, lmem::CacheZone::kStable);
        ctx.cache.reserve(key(layer, expert), 0, lmem::CacheZone::kStable,
                          /*is_duplicate=*/true);
        ++added;
    }
    // We should have stopped when dup_fraction >= max_frac.
    EXPECT_FALSE(ctx.cache.can_duplicate(0));
    double dup_frac = static_cast<double>(ctx.cache.duplicate_count(0)) / total;
    EXPECT_GE(dup_frac, max_frac);
}

TEST(ExpertCache, CanDuplicateDisabled) {
    auto cfg = v32_config();
    cfg.parallelism.expert_duplication.enabled = false;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);
    lmem::ExpertCache cache(vram, cfg, reg.per_routed_expert_bytes());
    EXPECT_FALSE(cache.can_duplicate(0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Capacity Queries
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCache, FreeSlotsInitially) {
    auto ctx = make_v32_context();
    for (int g = 0; g < ctx.cache.gpu_count(); ++g) {
        EXPECT_EQ(ctx.cache.free_slots(g, lmem::CacheZone::kStable),
                  ctx.cache.total_slots(g, lmem::CacheZone::kStable));
        EXPECT_EQ(ctx.cache.free_slots(g, lmem::CacheZone::kStreaming),
                  ctx.cache.total_slots(g, lmem::CacheZone::kStreaming));
    }
}

TEST(ExpertCache, FreeSlotsDecreasesOnReserve) {
    auto ctx = make_v32_context();
    int before = ctx.cache.free_slots(0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    EXPECT_EQ(ctx.cache.free_slots(0, lmem::CacheZone::kStable), before - 1);
}

TEST(ExpertCache, FreeSlotsIncreasesOnEvict) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    int after_reserve = ctx.cache.free_slots(0, lmem::CacheZone::kStable);
    ctx.cache.evict(key(3, 0), 0);
    EXPECT_EQ(ctx.cache.free_slots(0, lmem::CacheZone::kStable),
              after_reserve + 1);
}

TEST(ExpertCache, TotalResidentAcrossGpus) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 1), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(5, 0), 2, lmem::CacheZone::kStreaming);
    EXPECT_EQ(ctx.cache.total_resident(), 3);
}

// ═══════════════════════════════════════════════════════════════════════════
// Affinity Hints
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCache, AffinityHintsInitiallyValid) {
    auto ctx = make_v32_context();
    EXPECT_TRUE(ctx.cache.affinity_hints_valid());
}

TEST(ExpertCache, AffinityHintsInvalidateAndCheck) {
    auto ctx = make_v32_context();
    ctx.cache.invalidate_affinity_hints();
    EXPECT_FALSE(ctx.cache.affinity_hints_valid());
}

// ═══════════════════════════════════════════════════════════════════════════
// Residency Snapshot
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCache, SnapshotAllGpus) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 1), 2, lmem::CacheZone::kStreaming);
    ctx.cache.mark_all_ready(key(3, 0), 0);

    auto snap = ctx.cache.residency_snapshot();
    EXPECT_EQ(snap.size(), 2u);

    // Find the entry on GPU 0.
    auto it = std::find_if(snap.begin(), snap.end(),
        [](const lmem::ResidencyInfo& r) { return r.gpu_idx == 0; });
    ASSERT_NE(it, snap.end());
    EXPECT_EQ(it->key, key(3, 0));
    EXPECT_EQ(it->sub_components_ready, lmem::kAll);
}

TEST(ExpertCache, SnapshotSingleGpu) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 1), 2, lmem::CacheZone::kStreaming);

    auto snap0 = ctx.cache.residency_snapshot(0);
    EXPECT_EQ(snap0.size(), 1u);
    EXPECT_EQ(snap0[0].key, key(3, 0));

    auto snap2 = ctx.cache.residency_snapshot(2);
    EXPECT_EQ(snap2.size(), 1u);
    EXPECT_EQ(snap2[0].key, key(3, 1));
}

TEST(ExpertCache, SnapshotEmpty) {
    auto ctx = make_v32_context();
    EXPECT_TRUE(ctx.cache.residency_snapshot().empty());
    EXPECT_TRUE(ctx.cache.residency_snapshot(0).empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-GPU Scenarios
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCache, MultiGpuIsolatedReserveEvict) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(5, 0), 2, lmem::CacheZone::kStable);

    // Evicting on GPU 0 should not affect GPU 2.
    ctx.cache.evict(key(3, 0), 0);
    EXPECT_FALSE(ctx.cache.is_resident(key(3, 0), 0));
    EXPECT_TRUE(ctx.cache.is_resident(key(5, 0), 2));
}

TEST(ExpertCache, MultiGpuSameExpertOnMultiple) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 10), 2, lmem::CacheZone::kStable, true);

    // Evict from GPU 0 only.
    ctx.cache.evict(key(3, 10), 0);
    EXPECT_FALSE(ctx.cache.is_resident(key(3, 10), 0));
    EXPECT_TRUE(ctx.cache.is_resident(key(3, 10), 2));
    EXPECT_TRUE(ctx.cache.is_resident(key(3, 10)));  // Still resident somewhere.
}

TEST(ExpertCache, MultiGpuResidentGpusList) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 10), 2, lmem::CacheZone::kStable, true);
    ctx.cache.reserve(key(3, 10), 3, lmem::CacheZone::kStreaming, true);

    auto gpus = ctx.cache.resident_gpus(key(3, 10));
    EXPECT_EQ(gpus.size(), 3u);
    EXPECT_NE(std::find(gpus.begin(), gpus.end(), 0), gpus.end());
    EXPECT_NE(std::find(gpus.begin(), gpus.end(), 2), gpus.end());
    EXPECT_NE(std::find(gpus.begin(), gpus.end(), 3), gpus.end());
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration: Eviction inputs from ExpertCache
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCache, EvictionInputsPopulatedCorrectly) {
    auto ctx = make_v32_context();
    ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(3, 1), 0, lmem::CacheZone::kStreaming);
    ctx.cache.reserve(key(3, 2), 0, lmem::CacheZone::kStreaming, true);

    auto inputs = ctx.cache.eviction_inputs(0);
    ASSERT_EQ(inputs.size(), 3u);

    bool found_dup = false;
    for (const auto& inp : inputs) {
        if (inp.key == key(3, 2)) {
            EXPECT_TRUE(inp.is_duplicate);
            found_dup = true;
        } else {
            EXPECT_FALSE(inp.is_duplicate);
        }
        EXPECT_EQ(inp.gpu_idx, 0);
    }
    EXPECT_TRUE(found_dup);
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration: Full eviction pipeline
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheIntegration, FullEvictionPipeline) {
    auto ctx = make_v32_context();

    constexpr int gpu = 0;
    int num_experts = ctx.cache.total_slots(gpu, lmem::CacheZone::kStreaming);
    ASSERT_GE(num_experts, 2) << "Need at least 2 streaming slots for pipeline test";

    // Fill streaming zone with progressive sub-component readiness.
    for (int i = 0; i < num_experts; ++i) {
        auto ek = key(3, static_cast<uint16_t>(i));
        auto* addr = ctx.cache.reserve(ek, gpu, lmem::CacheZone::kStreaming);
        ASSERT_NE(addr, nullptr) << "Failed to reserve expert " << i;
        ctx.cache.mark_ready(ek, gpu, lmem::kGate);
        ctx.cache.mark_ready(ek, gpu, lmem::kUp);
        ctx.cache.mark_ready(ek, gpu, lmem::kDown);
    }
    EXPECT_EQ(ctx.cache.free_slots(gpu, lmem::CacheZone::kStreaming), 0);

    // eviction_inputs() returns valid inputs for all cached experts.
    auto inputs = ctx.cache.eviction_inputs(gpu, lmem::CacheZone::kStreaming);
    ASSERT_EQ(static_cast<int>(inputs.size()), num_experts);

    // Evict expert (3,0) directly (scoring now done in Python).
    EXPECT_TRUE(ctx.cache.evict(key(3, 0), gpu));
    EXPECT_FALSE(ctx.cache.is_resident(key(3, 0), gpu));

    // Load a replacement expert.
    auto new_expert = key(5, 100);
    auto* addr = ctx.cache.reserve(new_expert, gpu, lmem::CacheZone::kStreaming);
    ASSERT_NE(addr, nullptr);
    ctx.cache.mark_all_ready(new_expert, gpu);
    EXPECT_TRUE(ctx.cache.is_resident(new_expert, gpu));

    // All other experts still resident and fully ready.
    for (int i = 1; i < num_experts; ++i) {
        auto ek = key(3, static_cast<uint16_t>(i));
        EXPECT_TRUE(ctx.cache.is_resident(ek, gpu));
        auto* e = ctx.cache.lookup(ek, gpu);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->sub_components_ready, lmem::kAll);
    }

    // Slot accounting: zone is full again after evict+load.
    EXPECT_EQ(ctx.cache.free_slots(gpu, lmem::CacheZone::kStreaming), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration: Zone boundary enforcement under pressure
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheIntegration, ZoneBoundaryEnforcement) {
    auto ctx = make_v32_context();
    constexpr int gpu = 0;
    int stable_total = ctx.cache.total_slots(gpu, lmem::CacheZone::kStable);
    int streaming_total = ctx.cache.total_slots(gpu, lmem::CacheZone::kStreaming);
    ASSERT_GE(stable_total, 2) << "Need at least 2 stable slots";
    ASSERT_GE(streaming_total, 2) << "Need at least 2 streaming slots";

    // Load experts into both zones — use min(total, 3) to adapt to available slots.
    int n_stable = std::min(stable_total, 3);
    int n_streaming = std::min(streaming_total, 3);

    // Stable: experts (10, 0..n_stable-1)
    for (int e = 0; e < n_stable; ++e) {
        ctx.cache.reserve(key(10, static_cast<uint16_t>(e)), gpu,
                          lmem::CacheZone::kStable);
        ctx.cache.mark_all_ready(key(10, static_cast<uint16_t>(e)), gpu);
    }
    // Streaming: experts (20, 0..n_streaming-1)
    for (int e = 0; e < n_streaming; ++e) {
        ctx.cache.reserve(key(20, static_cast<uint16_t>(e)), gpu,
                          lmem::CacheZone::kStreaming);
        ctx.cache.mark_all_ready(key(20, static_cast<uint16_t>(e)), gpu);
    }

    // Zone-filtered eviction inputs: streaming-only should not include stable.
    auto streaming_inputs = ctx.cache.eviction_inputs(gpu,
                                                      lmem::CacheZone::kStreaming);
    EXPECT_EQ(static_cast<int>(streaming_inputs.size()), n_streaming);
    for (const auto& inp : streaming_inputs) {
        EXPECT_EQ(inp.zone, lmem::CacheZone::kStreaming);
        EXPECT_EQ(inp.key.layer_idx, 20u);
    }

    // Full eviction inputs include both zones.
    auto all_inputs = ctx.cache.eviction_inputs(gpu);
    EXPECT_EQ(static_cast<int>(all_inputs.size()), n_stable + n_streaming);

    // Now promote streaming expert (20,0) to stable.
    EXPECT_TRUE(ctx.cache.promote(key(20, 0), gpu));

    // Re-query streaming-only inputs — promoted expert should NOT appear.
    auto streaming_after = ctx.cache.eviction_inputs(gpu,
                                                     lmem::CacheZone::kStreaming);
    for (const auto& inp : streaming_after) {
        EXPECT_NE(inp.key, key(20, 0))
            << "Promoted expert should not appear in streaming eviction inputs";
    }
    EXPECT_EQ(static_cast<int>(streaming_after.size()), n_streaming - 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration: Duplication eviction ordering (INV-0.4 end-to-end)
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheIntegration, DuplicateMarking) {
    auto ctx = make_v32_context();
    // Expert (5,42): primary on GPU 0, duplicates on GPU 2 and GPU 3.
    ctx.cache.reserve(key(5, 42), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(5, 42), 2, lmem::CacheZone::kStable, true);
    ctx.cache.reserve(key(5, 42), 3, lmem::CacheZone::kStreaming, true);
    ctx.cache.reserve(key(5, 43), 0, lmem::CacheZone::kStable);
    ctx.cache.reserve(key(5, 44), 0, lmem::CacheZone::kStreaming);

    // GPU 2: eviction_inputs marks the expert as duplicate.
    auto inputs_gpu2 = ctx.cache.eviction_inputs(2);
    ASSERT_EQ(inputs_gpu2.size(), 1u);
    EXPECT_TRUE(inputs_gpu2[0].is_duplicate);
    EXPECT_EQ(inputs_gpu2[0].key, key(5, 42));

    // GPU 3: also a duplicate.
    auto inputs_gpu3 = ctx.cache.eviction_inputs(3);
    ASSERT_EQ(inputs_gpu3.size(), 1u);
    EXPECT_TRUE(inputs_gpu3[0].is_duplicate);

    // GPU 0: primary (5,42) is NOT marked as duplicate.
    auto inputs_gpu0 = ctx.cache.eviction_inputs(0);
    for (const auto& inp : inputs_gpu0) {
        EXPECT_FALSE(inp.is_duplicate)
            << "No entries on GPU 0 should be marked as duplicates";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration: VRAM budget consistency (ExpertCache + VramAllocator + LayerRegistry)
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheIntegration, VramBudgetConsistency) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    int64_t expert_bytes = reg.per_routed_expert_bytes();
    ASSERT_GT(expert_bytes, 0);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);
    lmem::ExpertCache cache(vram, cfg, expert_bytes);

    for (int g = 0; g < cache.gpu_count(); ++g) {
        const auto& gpu_layout = vram.layout().gpus[g];

        // Stable zone: slots * expert_bytes must fit within stable region.
        int stable_slots = cache.total_slots(g, lmem::CacheZone::kStable);
        EXPECT_LE(static_cast<int64_t>(stable_slots) * expert_bytes,
                  gpu_layout.expert_stable_bytes)
            << "GPU " << g << ": stable slots overflow stable region";

        // No wasted full-expert gap (at most expert_bytes-1 bytes unused).
        int64_t stable_waste = gpu_layout.expert_stable_bytes -
                               static_cast<int64_t>(stable_slots) * expert_bytes;
        EXPECT_LT(stable_waste, expert_bytes)
            << "GPU " << g << ": too much wasted stable space";

        // Streaming zone: same checks.
        int streaming_slots = cache.total_slots(g, lmem::CacheZone::kStreaming);
        EXPECT_LE(static_cast<int64_t>(streaming_slots) * expert_bytes,
                  gpu_layout.expert_streaming_bytes)
            << "GPU " << g << ": streaming slots overflow streaming region";

        // Streaming waste: up to 2 * expert_bytes (one per sub-zone: spill + prefetch)
        int64_t streaming_waste = gpu_layout.expert_streaming_bytes -
                                  static_cast<int64_t>(streaming_slots) * expert_bytes;
        EXPECT_LT(streaming_waste, 2 * expert_bytes)
            << "GPU " << g << ": too much wasted streaming space";

        // Verify zone ratio matches config.
        double expected_stable_frac = cfg.memory.expert_cache.stable_zone_fraction;
        int64_t expert_total = gpu_layout.expert_total_bytes();
        if (expert_total > 0) {
            double actual_stable_frac =
                static_cast<double>(gpu_layout.expert_stable_bytes) / expert_total;
            EXPECT_NEAR(actual_stable_frac, expected_stable_frac, 0.01)
                << "GPU " << g << ": stable/total ratio mismatch";
        }

        // Free slots should equal total slots initially.
        EXPECT_EQ(cache.free_slots(g, lmem::CacheZone::kStable), stable_slots);
        EXPECT_EQ(cache.free_slots(g, lmem::CacheZone::kStreaming), streaming_slots);
    }

    // With QuantInterface: verify projection offset consistency.
    lmod::ExpertShape shape{mcfg.raw().hidden_size, mcfg.raw().moe_intermediate_size};
    int64_t gate_bytes = nvfp4.bytes_per_projection(shape, lmod::Projection::gate);
    int64_t up_bytes = nvfp4.bytes_per_projection(shape, lmod::Projection::up);
    int64_t down_bytes = nvfp4.bytes_per_projection(shape, lmod::Projection::down);
    EXPECT_EQ(gate_bytes + up_bytes + down_bytes, expert_bytes)
        << "Projection sizes must sum to expert_bytes";
}

// ═══════════════════════════════════════════════════════════════════════════
// Smoke: Orchestrator cycle simulation
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheIntegration, OrchestratorCycleSimulation) {
    // Simulate 200 iterations of the orchestrator's cache management cycle:
    // Each iteration: load some experts, evict when full, promote hot ones,
    // demote cold ones, periodically invalidate affinity hints.
    // After every cycle, verify structural invariants hold.
    auto ctx = make_v32_context();
    int gpu = 0;

    auto verify_invariants = [&](const char* phase) {
        // INV: total_resident == sum of used_slots across zones.
        int used_stable = ctx.cache.used_slots(gpu, lmem::CacheZone::kStable);
        int used_streaming = ctx.cache.used_slots(gpu, lmem::CacheZone::kStreaming);
        auto snap = ctx.cache.residency_snapshot(gpu);
        EXPECT_EQ(static_cast<int>(snap.size()), used_stable + used_streaming)
            << "Invariant violated at " << phase
            << ": snapshot size != used slots sum";

        // INV: free + used == total for each zone.
        EXPECT_EQ(ctx.cache.free_slots(gpu, lmem::CacheZone::kStable) + used_stable,
                  ctx.cache.total_slots(gpu, lmem::CacheZone::kStable))
            << "Invariant violated at " << phase << ": stable free+used != total";
        EXPECT_EQ(ctx.cache.free_slots(gpu, lmem::CacheZone::kStreaming) + used_streaming,
                  ctx.cache.total_slots(gpu, lmem::CacheZone::kStreaming))
            << "Invariant violated at " << phase << ": streaming free+used != total";

        // INV: duplicate_count matches actual duplicates in snapshot.
        int dup_count = 0;
        for (const auto& r : snap) {
            if (r.is_duplicate) ++dup_count;
        }
        EXPECT_EQ(dup_count, ctx.cache.duplicate_count(gpu))
            << "Invariant violated at " << phase << ": duplicate count mismatch";

        // INV: every resident expert is findable via is_resident and lookup.
        for (const auto& r : snap) {
            EXPECT_TRUE(ctx.cache.is_resident(r.key, gpu))
                << "Invariant violated at " << phase
                << ": snapshot entry not findable via is_resident";
            EXPECT_NE(ctx.cache.lookup(r.key, gpu), nullptr)
                << "Invariant violated at " << phase
                << ": snapshot entry not findable via lookup";
        }
    };

    verify_invariants("initial");

    // Simple PRNG for deterministic test.
    uint32_t rng = 42;
    auto next_rng = [&]() -> uint32_t {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };

    int total_loads = 0;
    int total_evictions = 0;
    int total_promotions = 0;
    int total_demotions = 0;

    for (int cycle = 0; cycle < 200; ++cycle) {
        // --- LOAD phase: try to load 1-3 experts into streaming ---
        int to_load = 1 + static_cast<int>(next_rng() % 3);
        for (int j = 0; j < to_load; ++j) {
            uint32_t layer = 3 + (next_rng() % 58);  // MoE layers 3..60
            uint16_t expert = static_cast<uint16_t>(next_rng() % 256);
            auto ek = key(layer, expert);

            if (ctx.cache.is_resident(ek, gpu)) continue;

            auto* addr = ctx.cache.reserve(ek, gpu,
                                           lmem::CacheZone::kStreaming);
            if (addr == nullptr) {
                // Zone full — evict one (pick first from inputs).
                auto inputs = ctx.cache.eviction_inputs(gpu,
                                                        lmem::CacheZone::kStreaming);
                if (inputs.empty()) break;
                ctx.cache.evict(inputs[0].key, gpu);
                ++total_evictions;
                addr = ctx.cache.reserve(ek, gpu, lmem::CacheZone::kStreaming);
            }
            if (addr) {
                // Simulate progressive component readiness.
                ctx.cache.mark_ready(ek, gpu, lmem::kGate);
                ctx.cache.mark_ready(ek, gpu, lmem::kUp);
                ctx.cache.mark_ready(ek, gpu, lmem::kDown);
                ++total_loads;
            }
        }

        // --- PROMOTE phase: promote ~10% of streaming experts to stable ---
        if (cycle % 10 == 0) {
            auto snap = ctx.cache.residency_snapshot(gpu);
            for (const auto& r : snap) {
                if (r.zone == lmem::CacheZone::kStreaming &&
                    (next_rng() % 10) == 0) {
                    if (ctx.cache.promote(r.key, gpu))
                        ++total_promotions;
                }
            }
        }

        // --- DEMOTE phase: demote ~5% of stable experts ---
        if (cycle % 20 == 0) {
            auto snap = ctx.cache.residency_snapshot(gpu);
            for (const auto& r : snap) {
                if (r.zone == lmem::CacheZone::kStable &&
                    (next_rng() % 20) == 0) {
                    if (ctx.cache.demote(r.key, gpu))
                        ++total_demotions;
                }
            }
        }

        // --- INVALIDATE phase: workload shift every 50 cycles ---
        if (cycle % 50 == 0 && cycle > 0) {
            ctx.cache.invalidate_affinity_hints();
            EXPECT_FALSE(ctx.cache.affinity_hints_valid());
        }

        verify_invariants(("cycle " + std::to_string(cycle)).c_str());
    }

    // Verify the simulation actually exercised the code paths.
    EXPECT_GT(total_loads, 50) << "Should have loaded many experts";
    EXPECT_GT(total_evictions, 0) << "Should have evicted at least once";
    // Promotions/demotions may be 0 depending on RNG, but loads+evictions
    // must have happened for the test to be meaningful.

    // Final invariant check: no slot leaks.
    int used_stable = ctx.cache.used_slots(gpu, lmem::CacheZone::kStable);
    int used_streaming = ctx.cache.used_slots(gpu, lmem::CacheZone::kStreaming);
    int total_stable = ctx.cache.total_slots(gpu, lmem::CacheZone::kStable);
    int total_streaming = ctx.cache.total_slots(gpu, lmem::CacheZone::kStreaming);
    EXPECT_EQ(ctx.cache.free_slots(gpu, lmem::CacheZone::kStable),
              total_stable - used_stable);
    EXPECT_EQ(ctx.cache.free_slots(gpu, lmem::CacheZone::kStreaming),
              total_streaming - used_streaming);
    EXPECT_EQ(ctx.cache.total_resident(),
              used_stable + used_streaming);
}

// ═══════════════════════════════════════════════════════════════════════════
// Prefill spill mode tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExpertCacheSpill, InitialModeIsNormal) {
    auto ctx = make_v32_context();
    for (int g = 0; g < ctx.cache.gpu_count(); ++g) {
        EXPECT_EQ(ctx.cache.prefill_mode(g), lmem::PrefillMode::kNormal);
    }
}

TEST(ExpertCacheSpill, EnterReturnsValidScratch) {
    auto ctx = make_v32_context();
    const auto& gpu_layout = ctx.vram.layout().gpus[0];

    auto scratch = ctx.cache.enter_spill_mode(0);
    EXPECT_NE(scratch.ptr, nullptr);
    EXPECT_EQ(scratch.size_bytes, gpu_layout.streaming_spill_bytes);
    EXPECT_EQ(ctx.cache.prefill_mode(0), lmem::PrefillMode::kSpillActive);

    ctx.cache.exit_spill_mode(0);
    EXPECT_EQ(ctx.cache.prefill_mode(0), lmem::PrefillMode::kNormal);
}

TEST(ExpertCacheSpill, SpillEvictsSpillZoneExperts) {
    auto ctx = make_v32_context();

    // Fill some experts into streaming zone (some will land in spill)
    int total_streaming = ctx.cache.total_slots(0, lmem::CacheZone::kStreaming);
    ASSERT_GT(total_streaming, 0);

    int reserved = 0;
    for (int i = 0; i < total_streaming; ++i) {
        auto* addr = ctx.cache.reserve(key(100, i), 0, lmem::CacheZone::kStreaming);
        if (addr) {
            ctx.cache.mark_all_ready(key(100, i), 0);
            ++reserved;
        }
    }
    ASSERT_GT(reserved, 0);
    EXPECT_EQ(ctx.cache.total_resident(), reserved);

    // Count how many are in spill zone
    int spill_count = 0;
    for (int i = 0; i < reserved; ++i) {
        const auto* entry = ctx.cache.lookup(key(100, i), 0);
        if (entry && entry->in_spill_zone) ++spill_count;
    }

    // Enter spill mode evicts spill-zone experts
    ctx.cache.enter_spill_mode(0);
    EXPECT_EQ(ctx.cache.total_resident(), reserved - spill_count);

    ctx.cache.exit_spill_mode(0);
}

TEST(ExpertCacheSpill, PrefetchExpertsUnaffectedDuringSpill) {
    auto ctx = make_v32_context();

    // Fill streaming — some go to spill, some to prefetch
    int total = ctx.cache.total_slots(0, lmem::CacheZone::kStreaming);
    for (int i = 0; i < total; ++i) {
        auto* addr = ctx.cache.reserve(key(200, i), 0, lmem::CacheZone::kStreaming);
        if (addr) ctx.cache.mark_all_ready(key(200, i), 0);
    }

    // Count prefetch-zone experts
    int prefetch_count = 0;
    for (int i = 0; i < total; ++i) {
        const auto* entry = ctx.cache.lookup(key(200, i), 0);
        if (entry && !entry->in_spill_zone) ++prefetch_count;
    }

    ctx.cache.enter_spill_mode(0);

    // Prefetch experts should still be resident
    EXPECT_EQ(ctx.cache.total_resident(), prefetch_count);
    for (int i = 0; i < total; ++i) {
        const auto* entry = ctx.cache.lookup(key(200, i), 0);
        if (entry) {
            EXPECT_FALSE(entry->in_spill_zone)
                << "Only prefetch zone experts should remain";
        }
    }

    ctx.cache.exit_spill_mode(0);
}

TEST(ExpertCacheSpill, ReserveDuringSpillUsePrefetchOnly) {
    auto ctx = make_v32_context();
    ctx.cache.enter_spill_mode(0);

    // Reserve should only go to prefetch zone
    auto* addr = ctx.cache.reserve(key(300, 0), 0, lmem::CacheZone::kStreaming);
    if (addr) {
        const auto* entry = ctx.cache.lookup(key(300, 0), 0);
        ASSERT_NE(entry, nullptr);
        EXPECT_FALSE(entry->in_spill_zone)
            << "During spill mode, new streaming entries go to prefetch only";
    }

    ctx.cache.exit_spill_mode(0);
}

TEST(ExpertCacheSpill, StableZoneUnaffected) {
    auto ctx = make_v32_context();

    // Reserve a stable expert
    ctx.cache.reserve(key(400, 0), 0, lmem::CacheZone::kStable);
    ctx.cache.mark_all_ready(key(400, 0), 0);

    int stable_before = ctx.cache.used_slots(0, lmem::CacheZone::kStable);

    ctx.cache.enter_spill_mode(0);
    EXPECT_EQ(ctx.cache.used_slots(0, lmem::CacheZone::kStable), stable_before);
    EXPECT_TRUE(ctx.cache.is_resident(key(400, 0), 0));

    // Can still allocate stable during spill mode
    auto* addr = ctx.cache.reserve(key(400, 1), 0, lmem::CacheZone::kStable);
    EXPECT_NE(addr, nullptr);

    ctx.cache.exit_spill_mode(0);
    EXPECT_TRUE(ctx.cache.is_resident(key(400, 0), 0));
    EXPECT_TRUE(ctx.cache.is_resident(key(400, 1), 0));
}

TEST(ExpertCacheSpill, ExitRestoresSpillCapacity) {
    auto ctx = make_v32_context();
    const auto& gpu_layout = ctx.vram.layout().gpus[0];
    int expected_spill_slots = static_cast<int>(
        gpu_layout.streaming_spill_bytes / ctx.expert_bytes);

    int free_before = ctx.cache.free_slots(0, lmem::CacheZone::kStreaming);

    ctx.cache.enter_spill_mode(0);
    // During spill, free_slots(kStreaming) only counts prefetch
    int free_during = ctx.cache.free_slots(0, lmem::CacheZone::kStreaming);
    EXPECT_LT(free_during, free_before);

    ctx.cache.exit_spill_mode(0);
    // After exit, all spill slots should be free again
    int free_after = ctx.cache.free_slots(0, lmem::CacheZone::kStreaming);
    EXPECT_EQ(free_after, free_before);
}

TEST(ExpertCacheSpill, ModeTransitionEnforced) {
    auto ctx = make_v32_context();

    // Normal → spill: OK
    ctx.cache.enter_spill_mode(0);
    EXPECT_EQ(ctx.cache.prefill_mode(0), lmem::PrefillMode::kSpillActive);

    // Spill → normal: OK
    ctx.cache.exit_spill_mode(0);
    EXPECT_EQ(ctx.cache.prefill_mode(0), lmem::PrefillMode::kNormal);
}

TEST(ExpertCacheSpill, SpillScratchContiguousWithKvMain) {
    auto ctx = make_v32_context();
    const auto& region = ctx.vram.region(0);
    const auto& gpu_layout = ctx.vram.layout().gpus[0];

    auto scratch = ctx.cache.enter_spill_mode(0);

    // Spill zone starts at expert_streaming (which is contiguous with KV main end)
    auto* kv_main_end = static_cast<char*>(region.kv_main) + gpu_layout.kv_main_bytes;
    EXPECT_EQ(scratch.ptr, kv_main_end)
        << "Spill scratch must be contiguous with KV main tail";

    ctx.cache.exit_spill_mode(0);
}
