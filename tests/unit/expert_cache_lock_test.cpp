// #90: Tests for ExpertCache eviction lock (lock/unlock refcount semantics).

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "config/config_resolver.h"
#include "core/device_backend.h"
#include "core/memory/expert_cache.h"
#include "core/memory/vram_allocator.h"
#include "core/null_device_backend.h"
#include "model/quantization/fp8.h"

namespace lc = layerstorm::config;
namespace lcomp = layerstorm::compute;
namespace lmem = layerstorm::memory;
namespace lmod = layerstorm::model;

namespace {

lc::Config small_config() {
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

struct Ctx {
    lc::Config cfg;
    lmod::ModelConfig mcfg;
    int64_t expert_bytes;
    NullBackends backends;
    lmem::VramAllocator vram;
    lmem::ExpertCache cache;
};

Ctx make_ctx() {
    auto cfg = small_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);
    int64_t expert_bytes = reg.per_routed_expert_bytes();
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);
    auto cache = lmem::ExpertCache(vram, cfg, expert_bytes);
    return Ctx{std::move(cfg), std::move(mcfg), expert_bytes,
               std::move(nb), std::move(vram), std::move(cache)};
}

lmem::ExpertKey key(uint32_t layer, uint16_t expert) {
    return {layer, expert};
}

}  // namespace

// ── Basic lock/unlock ──────────────────────────────────────────────────────

TEST(ExpertCacheLock, LockResidentExpert) {
    auto ctx = make_ctx();
    auto k = key(1, 0);
    ctx.cache.reserve(k, 0, lmem::CacheZone::kStable);
    ctx.cache.mark_all_ready(k, 0);

    EXPECT_FALSE(ctx.cache.is_locked(k, 0));
    EXPECT_TRUE(ctx.cache.lock(k, 0));
    EXPECT_TRUE(ctx.cache.is_locked(k, 0));
    EXPECT_TRUE(ctx.cache.unlock(k, 0));
    EXPECT_FALSE(ctx.cache.is_locked(k, 0));
}

TEST(ExpertCacheLock, LockNonResidentReturnsFalse) {
    auto ctx = make_ctx();
    auto k = key(1, 0);
    EXPECT_FALSE(ctx.cache.lock(k, 0));
    EXPECT_FALSE(ctx.cache.is_locked(k, 0));
}

TEST(ExpertCacheLock, UnlockNonLockedReturnsFalse) {
    auto ctx = make_ctx();
    auto k = key(1, 0);
    ctx.cache.reserve(k, 0, lmem::CacheZone::kStable);
    ctx.cache.mark_all_ready(k, 0);

    // Expert is resident but not locked — unlock should fail.
    EXPECT_FALSE(ctx.cache.unlock(k, 0));
}

TEST(ExpertCacheLock, UnlockNonResidentReturnsFalse) {
    auto ctx = make_ctx();
    EXPECT_FALSE(ctx.cache.unlock(key(1, 0), 0));
}

// ── Refcounting ────────────────────────────────────────────────────────────

TEST(ExpertCacheLock, MultipleLocksRequireMatchingUnlocks) {
    auto ctx = make_ctx();
    auto k = key(1, 0);
    ctx.cache.reserve(k, 0, lmem::CacheZone::kStable);
    ctx.cache.mark_all_ready(k, 0);

    // Lock 3 times.
    EXPECT_TRUE(ctx.cache.lock(k, 0));
    EXPECT_TRUE(ctx.cache.lock(k, 0));
    EXPECT_TRUE(ctx.cache.lock(k, 0));
    EXPECT_TRUE(ctx.cache.is_locked(k, 0));

    // Unlock twice — still locked.
    EXPECT_TRUE(ctx.cache.unlock(k, 0));
    EXPECT_TRUE(ctx.cache.unlock(k, 0));
    EXPECT_TRUE(ctx.cache.is_locked(k, 0));

    // Final unlock — no longer locked.
    EXPECT_TRUE(ctx.cache.unlock(k, 0));
    EXPECT_FALSE(ctx.cache.is_locked(k, 0));

    // Additional unlock should fail (refcount=0).
    EXPECT_FALSE(ctx.cache.unlock(k, 0));
}

// ── Eviction guard ─────────────────────────────────────────────────────────

TEST(ExpertCacheLock, EvictRefusesLockedExpert) {
    auto ctx = make_ctx();
    auto k = key(1, 0);
    ctx.cache.reserve(k, 0, lmem::CacheZone::kStable);
    ctx.cache.mark_all_ready(k, 0);

    ctx.cache.lock(k, 0);
    EXPECT_FALSE(ctx.cache.evict(k, 0));
    EXPECT_TRUE(ctx.cache.is_resident(k, 0));

    // After unlock, eviction succeeds.
    ctx.cache.unlock(k, 0);
    EXPECT_TRUE(ctx.cache.evict(k, 0));
    EXPECT_FALSE(ctx.cache.is_resident(k, 0));
}

TEST(ExpertCacheLock, EvictRefusesMultiLocked) {
    auto ctx = make_ctx();
    auto k = key(1, 0);
    ctx.cache.reserve(k, 0, lmem::CacheZone::kStable);
    ctx.cache.mark_all_ready(k, 0);

    ctx.cache.lock(k, 0);
    ctx.cache.lock(k, 0);

    // Both unlock calls needed before eviction works.
    EXPECT_FALSE(ctx.cache.evict(k, 0));
    ctx.cache.unlock(k, 0);
    EXPECT_FALSE(ctx.cache.evict(k, 0));
    ctx.cache.unlock(k, 0);
    EXPECT_TRUE(ctx.cache.evict(k, 0));
}

// ── Spill mode skips locked ────────────────────────────────────────────────

TEST(ExpertCacheLock, SpillModeSkipsLockedExperts) {
    auto ctx = make_ctx();

    // Reserve one expert in the streaming zone (goes to spill sub-allocator).
    auto k0 = key(2, 0);
    ctx.cache.reserve(k0, 0, lmem::CacheZone::kStreaming);
    ctx.cache.mark_all_ready(k0, 0);

    // Verify it's in spill zone.
    const auto* entry = ctx.cache.lookup(k0, 0);
    ASSERT_NE(entry, nullptr);
    // If it didn't end up in spill zone (config too small), skip this test.
    if (!entry->in_spill_zone) {
        GTEST_SKIP() << "Expert not allocated in spill zone (config has no spill slots)";
    }

    // Lock it — should survive enter_spill_mode.
    ctx.cache.lock(k0, 0);
    ctx.cache.enter_spill_mode(0);

    EXPECT_TRUE(ctx.cache.is_resident(k0, 0))
        << "Locked expert in spill zone must survive enter_spill_mode";

    ctx.cache.unlock(k0, 0);
    ctx.cache.exit_spill_mode(0);
}

// ── is_locked on invalid GPU ───────────────────────────────────────────────

TEST(ExpertCacheLock, IsLockedInvalidGpuReturnsFalse) {
    auto ctx = make_ctx();
    EXPECT_FALSE(ctx.cache.is_locked(key(0, 0), 99));
    EXPECT_FALSE(ctx.cache.is_locked(key(0, 0), -1));
}
