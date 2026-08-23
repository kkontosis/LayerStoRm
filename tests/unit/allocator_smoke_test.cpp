#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "config/config_resolver.h"
#include "core/device_backend.h"
#include "core/memory/eviction_policy.h"
#include "core/memory/expert_cache.h"
#include "core/memory/page_allocator.h"
#include "core/memory/vram_allocator.h"
#include "core/null_device_backend.h"
#include "model/quantization/nvfp4.h"

namespace lc = layerstorm::config;
namespace lcomp = layerstorm::compute;
namespace lmem = layerstorm::memory;
namespace lmod = layerstorm::model;

namespace {

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

}  // namespace

// ── Config helper ──────────────────────────────────────────────────────────

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

lmem::ExpertKey key(uint32_t layer, uint16_t expert) {
    return {layer, expert};
}

/// Returns true if [a_start, a_start+a_len) overlaps [b_start, b_start+b_len).
bool ranges_overlap(uintptr_t a_start, int64_t a_len,
                    uintptr_t b_start, int64_t b_len) {
    if (a_len == 0 || b_len == 0) return false;
    return a_start < b_start + b_len && b_start < a_start + a_len;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Smoke: VRAM partitioning integrity
// ═══════════════════════════════════════════════════════════════════════════
//
// Stand up VramAllocator + PageAllocator + ExpertCache from the same V3.2
// layout. Allocate every KV page and every expert slot. Verify:
//   (a) Every page address falls within KV region bounds
//   (b) Every expert slot address falls within expert region bounds
//   (c) No KV address overlaps any expert address range
//   (d) No two pages or slots share the same address

TEST(AllocatorCrossModule, VramPartitioningIntegrity) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    int64_t expert_bytes = reg.per_routed_expert_bytes();
    ASSERT_GT(expert_bytes, 0);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);

    lmem::PageAllocator pages(vram, nb.ptrs[0]);
    lmem::ExpertCache cache(vram, cfg, expert_bytes);

    ASSERT_EQ(pages.gpu_count(), vram.gpu_count());
    ASSERT_EQ(cache.gpu_count(), vram.gpu_count());

    for (int g = 0; g < vram.gpu_count(); ++g) {
        const auto& region = vram.region(g);
        const auto& gpu_layout = vram.layout().gpus[g];

        // ── KV region bounds (two separate segments after region reorder) ─
        // Layout: pinned | kv_speculation | indexer_k | kv_main | streaming | stable
        auto kv_main_start = reinterpret_cast<uintptr_t>(region.kv_main);
        int64_t kv_main_len = gpu_layout.kv_main_bytes;
        auto kv_spec_start = reinterpret_cast<uintptr_t>(region.kv_speculation);
        int64_t kv_spec_len = gpu_layout.kv_speculation_bytes;

        // ── Expert region bounds ────────────────────────────────────────
        auto expert_stable_start =
            reinterpret_cast<uintptr_t>(region.expert_stable);
        auto expert_streaming_start =
            reinterpret_cast<uintptr_t>(region.expert_streaming);
        int64_t expert_stable_len = gpu_layout.expert_stable_bytes;
        int64_t expert_streaming_len = gpu_layout.expert_streaming_bytes;

        // (c) KV regions must not overlap expert regions.
        EXPECT_FALSE(ranges_overlap(kv_main_start, kv_main_len,
                                    expert_stable_start, expert_stable_len))
            << "GPU " << g << ": KV main overlaps expert stable region";
        EXPECT_FALSE(ranges_overlap(kv_main_start, kv_main_len,
                                    expert_streaming_start, expert_streaming_len))
            << "GPU " << g << ": KV main overlaps expert streaming region";
        EXPECT_FALSE(ranges_overlap(kv_spec_start, kv_spec_len,
                                    expert_stable_start, expert_stable_len))
            << "GPU " << g << ": KV spec overlaps expert stable region";
        EXPECT_FALSE(ranges_overlap(kv_spec_start, kv_spec_len,
                                    expert_streaming_start, expert_streaming_len))
            << "GPU " << g << ": KV spec overlaps expert streaming region";
        EXPECT_FALSE(ranges_overlap(expert_stable_start, expert_stable_len,
                                    expert_streaming_start, expert_streaming_len))
            << "GPU " << g << ": expert stable overlaps expert streaming";

        // ── Allocate all KV pages ───────────────────────────────────────
        std::set<uintptr_t> all_addrs;

        int main_pages = pages.total_pages(g, lmem::Pool::kMain);
        int spec_pages = pages.total_pages(g, lmem::Pool::kSpeculation);

        std::vector<lmem::PageHandle> page_handles;
        page_handles.reserve(main_pages + spec_pages);

        for (int p = 0; p < main_pages; ++p) {
            auto h = pages.allocate(g, lmem::Pool::kMain);
            ASSERT_TRUE(h.has_value()) << "GPU " << g << ": main page " << p
                                       << " allocation failed";
            auto addr = reinterpret_cast<uintptr_t>(h->gpu_ptr);
            // (a) Page address within KV main region.
            EXPECT_GE(addr, kv_main_start)
                << "GPU " << g << ": page below KV main region";
            EXPECT_LT(addr, kv_main_start + kv_main_len)
                << "GPU " << g << ": page above KV main region";
            // (d) No duplicate addresses.
            EXPECT_TRUE(all_addrs.insert(addr).second)
                << "GPU " << g << ": duplicate page address";
            page_handles.push_back(*h);
        }
        for (int p = 0; p < spec_pages; ++p) {
            auto h = pages.allocate(g, lmem::Pool::kSpeculation);
            ASSERT_TRUE(h.has_value()) << "GPU " << g << ": spec page " << p
                                       << " allocation failed";
            auto addr = reinterpret_cast<uintptr_t>(h->gpu_ptr);
            // (a) Page address within KV speculation region.
            EXPECT_GE(addr, kv_spec_start);
            EXPECT_LT(addr, kv_spec_start + kv_spec_len);
            EXPECT_TRUE(all_addrs.insert(addr).second)
                << "GPU " << g << ": duplicate page address";
            page_handles.push_back(*h);
        }

        // Pools should be exhausted.
        EXPECT_FALSE(pages.allocate(g, lmem::Pool::kMain).has_value());
        EXPECT_FALSE(pages.allocate(g, lmem::Pool::kSpeculation).has_value());

        // ── Allocate all expert slots ───────────────────────────────────
        int stable_slots = cache.total_slots(g, lmem::CacheZone::kStable);
        int streaming_slots = cache.total_slots(g, lmem::CacheZone::kStreaming);

        for (int s = 0; s < stable_slots; ++s) {
            auto ek = key(static_cast<uint32_t>(g * 1000 + s), 0);
            auto* addr = cache.reserve(ek, g, lmem::CacheZone::kStable);
            ASSERT_NE(addr, nullptr) << "GPU " << g << ": stable slot " << s
                                     << " allocation failed";
            auto uaddr = reinterpret_cast<uintptr_t>(addr);
            // (b) Expert address within expert stable region.
            EXPECT_GE(uaddr, expert_stable_start)
                << "GPU " << g << ": expert below stable region";
            EXPECT_LT(uaddr, expert_stable_start + expert_stable_len)
                << "GPU " << g << ": expert above stable region";
            // (c) Expert address must not be in any KV range.
            EXPECT_FALSE(ranges_overlap(uaddr, expert_bytes,
                                        kv_main_start, kv_main_len))
                << "GPU " << g << ": expert slot overlaps KV main region";
            EXPECT_FALSE(ranges_overlap(uaddr, expert_bytes,
                                        kv_spec_start, kv_spec_len))
                << "GPU " << g << ": expert slot overlaps KV spec region";
            // (d) No duplicate with any KV page address.
            EXPECT_TRUE(all_addrs.insert(uaddr).second)
                << "GPU " << g << ": expert address collides with a page address";
        }

        for (int s = 0; s < streaming_slots; ++s) {
            auto ek = key(static_cast<uint32_t>(g * 1000 + stable_slots + s), 1);
            auto* addr = cache.reserve(ek, g, lmem::CacheZone::kStreaming);
            ASSERT_NE(addr, nullptr) << "GPU " << g << ": streaming slot " << s
                                     << " allocation failed";
            auto uaddr = reinterpret_cast<uintptr_t>(addr);
            // (b) Expert address within expert streaming region.
            EXPECT_GE(uaddr, expert_streaming_start);
            EXPECT_LT(uaddr, expert_streaming_start + expert_streaming_len);
            EXPECT_FALSE(ranges_overlap(uaddr, expert_bytes,
                                        kv_main_start, kv_main_len));
            EXPECT_FALSE(ranges_overlap(uaddr, expert_bytes,
                                        kv_spec_start, kv_spec_len));
            EXPECT_TRUE(all_addrs.insert(uaddr).second)
                << "GPU " << g << ": expert address collides with another address";
        }

        // Expert zones should be exhausted.
        EXPECT_EQ(cache.free_slots(g, lmem::CacheZone::kStable), 0);
        EXPECT_EQ(cache.free_slots(g, lmem::CacheZone::kStreaming), 0);

        // ── Cleanup for this GPU ────────────────────────────────────────
        for (auto& h : page_handles) pages.free(h);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Smoke: Allocation churn with leak detection
// ═══════════════════════════════════════════════════════════════════════════
//
// Simulate N cycles of mixed page + expert slot allocation/deallocation.
// After each cycle: verify free + used == total for both allocators on
// every GPU. After final cycle: free everything, verify all counts return
// to initial state.

TEST(AllocatorCrossModule, AllocationChurnLeakDetection) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    int64_t expert_bytes = reg.per_routed_expert_bytes();

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);

    lmem::PageAllocator pages(vram, nb.ptrs[0]);
    lmem::ExpertCache cache(vram, cfg, expert_bytes);

    // Deterministic PRNG.
    uint32_t rng = 0xDEADBEEF;
    auto next = [&]() -> uint32_t {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };

    // Track live handles for cleanup.
    struct LivePages {
        std::vector<lmem::PageHandle> main_handles;
        std::vector<lmem::PageHandle> spec_handles;
    };
    struct LiveExperts {
        std::vector<lmem::ExpertKey> stable_keys;
        std::vector<lmem::ExpertKey> streaming_keys;
    };

    std::vector<LivePages> live_pages(vram.gpu_count());
    std::vector<LiveExperts> live_experts(vram.gpu_count());

    auto verify_invariants = [&](const char* phase) {
        for (int g = 0; g < vram.gpu_count(); ++g) {
            // PageAllocator: free + used == total for each pool.
            int main_free = pages.free_pages(g, lmem::Pool::kMain);
            int main_used = pages.used_pages(g, lmem::Pool::kMain);
            int main_total = pages.total_pages(g, lmem::Pool::kMain);
            EXPECT_EQ(main_free + main_used, main_total)
                << phase << " GPU " << g << ": main page accounting broken";

            int spec_free = pages.free_pages(g, lmem::Pool::kSpeculation);
            int spec_used = pages.used_pages(g, lmem::Pool::kSpeculation);
            int spec_total = pages.total_pages(g, lmem::Pool::kSpeculation);
            EXPECT_EQ(spec_free + spec_used, spec_total)
                << phase << " GPU " << g << ": spec page accounting broken";

            // ExpertCache: free + used == total for each zone.
            int stable_free = cache.free_slots(g, lmem::CacheZone::kStable);
            int stable_used = cache.used_slots(g, lmem::CacheZone::kStable);
            int stable_total = cache.total_slots(g, lmem::CacheZone::kStable);
            EXPECT_EQ(stable_free + stable_used, stable_total)
                << phase << " GPU " << g << ": stable expert accounting broken";

            int str_free = cache.free_slots(g, lmem::CacheZone::kStreaming);
            int str_used = cache.used_slots(g, lmem::CacheZone::kStreaming);
            int str_total = cache.total_slots(g, lmem::CacheZone::kStreaming);
            EXPECT_EQ(str_free + str_used, str_total)
                << phase << " GPU " << g << ": streaming expert accounting broken";
        }
    };

    verify_invariants("initial");

    constexpr int kCycles = 100;
    uint32_t expert_id_counter = 0;

    for (int cycle = 0; cycle < kCycles; ++cycle) {
        int g = static_cast<int>(next() % vram.gpu_count());

        // ── Allocate some pages ─────────────────────────────────────────
        int page_allocs = 1 + static_cast<int>(next() % 5);
        for (int i = 0; i < page_allocs; ++i) {
            bool use_main = (next() % 2) == 0;
            auto pool = use_main ? lmem::Pool::kMain : lmem::Pool::kSpeculation;
            auto h = pages.allocate(g, pool);
            if (h.has_value()) {
                if (use_main)
                    live_pages[g].main_handles.push_back(*h);
                else
                    live_pages[g].spec_handles.push_back(*h);
            }
        }

        // ── Allocate some expert slots ──────────────────────────────────
        int expert_allocs = 1 + static_cast<int>(next() % 3);
        for (int i = 0; i < expert_allocs; ++i) {
            bool use_stable = (next() % 2) == 0;
            auto zone = use_stable ? lmem::CacheZone::kStable
                                   : lmem::CacheZone::kStreaming;
            auto ek = key(expert_id_counter / 256, expert_id_counter % 256);
            ++expert_id_counter;
            auto* addr = cache.reserve(ek, g, zone);
            if (addr != nullptr) {
                cache.mark_all_ready(ek, g);
                if (use_stable)
                    live_experts[g].stable_keys.push_back(ek);
                else
                    live_experts[g].streaming_keys.push_back(ek);
            }
        }

        // ── Free some pages (randomly) ──────────────────────────────────
        for (auto* handles : {&live_pages[g].main_handles,
                              &live_pages[g].spec_handles}) {
            if (!handles->empty() && (next() % 3) == 0) {
                // Free ~1/3 of live handles.
                int to_free = 1 + static_cast<int>(next() % handles->size());
                to_free = std::min(to_free, static_cast<int>(handles->size()));
                for (int f = 0; f < to_free; ++f) {
                    pages.free(handles->back());
                    handles->pop_back();
                }
            }
        }

        // ── Evict some experts (randomly) ───────────────────────────────
        for (auto* keys : {&live_experts[g].stable_keys,
                           &live_experts[g].streaming_keys}) {
            if (!keys->empty() && (next() % 3) == 0) {
                int to_evict = 1 + static_cast<int>(next() % keys->size());
                to_evict = std::min(to_evict, static_cast<int>(keys->size()));
                for (int f = 0; f < to_evict; ++f) {
                    EXPECT_TRUE(cache.evict(keys->back(), g));
                    keys->pop_back();
                }
            }
        }

        verify_invariants(("cycle " + std::to_string(cycle)).c_str());
    }

    // ── Final cleanup: free everything ──────────────────────────────────
    for (int g = 0; g < vram.gpu_count(); ++g) {
        for (auto& h : live_pages[g].main_handles) pages.free(h);
        for (auto& h : live_pages[g].spec_handles) pages.free(h);
        live_pages[g].main_handles.clear();
        live_pages[g].spec_handles.clear();

        for (auto& ek : live_experts[g].stable_keys) cache.evict(ek, g);
        for (auto& ek : live_experts[g].streaming_keys) cache.evict(ek, g);
        live_experts[g].stable_keys.clear();
        live_experts[g].streaming_keys.clear();
    }

    // ── Verify return to initial state: no leaks ────────────────────────
    for (int g = 0; g < vram.gpu_count(); ++g) {
        EXPECT_EQ(pages.free_pages(g, lmem::Pool::kMain),
                  pages.total_pages(g, lmem::Pool::kMain))
            << "GPU " << g << ": main page leak after full cleanup";
        EXPECT_EQ(pages.free_pages(g, lmem::Pool::kSpeculation),
                  pages.total_pages(g, lmem::Pool::kSpeculation))
            << "GPU " << g << ": spec page leak after full cleanup";

        EXPECT_EQ(cache.free_slots(g, lmem::CacheZone::kStable),
                  cache.total_slots(g, lmem::CacheZone::kStable))
            << "GPU " << g << ": stable expert slot leak after full cleanup";
        EXPECT_EQ(cache.free_slots(g, lmem::CacheZone::kStreaming),
                  cache.total_slots(g, lmem::CacheZone::kStreaming))
            << "GPU " << g << ": streaming expert slot leak after full cleanup";

        EXPECT_EQ(cache.total_resident(), 0)
            << "GPU " << g << ": experts still resident after full cleanup";
    }
}
