// Smoke test: VramAllocator + PageAllocator + ExpertCache on real GPU memory.
//
// Expected hardware (dev machine):
//   GPU 0: RTX 5090, 31 GiB VRAM, NUMA 2, PCI 0000:6a:00.0
//   GPU 1: RTX 5090, 31 GiB VRAM, NUMA 3, PCI 0000:94:00.0
//   GPU 2: RTX 5080, 15 GiB VRAM, NUMA 0, PCI 0000:16:00.0
//   GPU 3: RTX 5080, 15 GiB VRAM, NUMA 2, PCI 0000:40:00.0
//   TP pair: GPU 0 + GPU 1 | System RAM: ~503 GiB
//
// Tests allocate real CUDA memory via CudaSm120DeviceBackend and exercise
// all three allocators (VramAllocator, PageAllocator, ExpertCache) on the
// actual GPU address space.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "compute/cuda_sm120_device_backend.h"
#include "config/config_resolver.h"
#include "core/device_backend.h"
#include "core/memory/expert_cache.h"
#include "core/memory/page_allocator.h"
#include "core/memory/vram_allocator.h"
#include "model/quantization/nvfp4.h"

namespace lc = layerstorm::config;
namespace lcomp = layerstorm::compute;
namespace lmem = layerstorm::memory;
namespace lmod = layerstorm::model;

// ── Fixture ────────────────────────────────────────────────────────────────

class AllocatorSmoke : public ::testing::Test {
protected:
    void SetUp() override {
        std::cout << "\nExpected hardware (dev machine):\n"
                  << "  GPU 0: RTX 5090, 31 GiB\n"
                  << "  GPU 1: RTX 5090, 31 GiB\n"
                  << "  GPU 2: RTX 5080, 15 GiB\n"
                  << "  GPU 3: RTX 5080, 15 GiB\n\n";

        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            GTEST_SKIP() << "No CUDA GPU present — cannot run hardware smoke test";
        }

        // Build hardware config from real GPUs (skeleton entries, resolve fills).
        for (int i = 0; i < count; ++i) {
            lc::GpuConfig g;
            g.id = i;
            cfg.hardware.gpus.push_back(g);
        }
        lc::resolve_config(cfg);

        // Parse model/quant config from JSON (same V3.2 config as unit tests).
        // Hardware section uses dummy GPUs — we overwrite it below.
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
                {"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                {"system_ram_gb", 256}}},
        };
        auto parsed = lc::parse_config(j);

        // Copy model + quant from parsed config, keep hardware from resolve_config.
        cfg.model = parsed.model;
        cfg.quantization = parsed.quantization;
        cfg.memory = parsed.memory;
        cfg.orchestrator = parsed.orchestrator;
        cfg.prefetch = parsed.prefetch;
        cfg.speculation = parsed.speculation;
        cfg.parallelism = parsed.parallelism;
        cfg.compute = parsed.compute;
        cfg.serving = parsed.serving;

        // TP array: first two 5090s if available.
        cfg.hardware.tp_array.clear();
        for (const auto& g : cfg.hardware.gpus) {
            if (g.type == lc::GpuType::rtx5090 &&
                static_cast<int>(cfg.hardware.tp_array.size()) < 2)
                cfg.hardware.tp_array.push_back(g.id);
        }

        mcfg = std::make_unique<lmod::ModelConfig>(cfg);
        reg = std::make_unique<lmod::LayerRegistry>(*mcfg, cfg, nvfp4);
        expert_bytes = reg->per_routed_expert_bytes();

        std::cout << "Detected " << count << " GPUs, expert_bytes="
                  << expert_bytes << " ("
                  << (expert_bytes / 1048576.0) << " MB)\n";
        for (const auto& g : cfg.hardware.gpus) {
            std::cout << "  GPU " << g.id << ": "
                      << static_cast<int>(g.vram_gb) << " GiB VRAM\n";
        }
        std::cout << std::endl;
    }

    lc::Config cfg;
    lmod::Nvfp4 nvfp4;
    std::unique_ptr<lmod::ModelConfig> mcfg;
    std::unique_ptr<lmod::LayerRegistry> reg;
    int64_t expert_bytes = 0;
};

// ── Helper ─────────────────────────────────────────────────────────────────

namespace {

lmem::ExpertKey key(uint32_t layer, uint16_t expert) {
    return {layer, expert};
}

bool is_device_ptr(void* ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    return err == cudaSuccess && attr.type == cudaMemoryTypeDevice;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Smoke: Real CUDA VRAM partitioning
// ═══════════════════════════════════════════════════════════════════════════
//
// Allocate real CUDA memory via VramAllocator. Build PageAllocator and
// ExpertCache on top. Verify:
//   (a) All region base pointers are real device pointers
//   (b) All allocated page/slot addresses are device pointers
//   (c) Address ranges don't overlap between KV and expert regions
//   (d) No duplicate addresses across pages and slots
//   (e) All addresses freed cleanly (no CUDA errors on teardown)

TEST_F(AllocatorSmoke, VramPartitioningOnRealGpu) {
    auto layout = lmem::compute_vram_layout(cfg, *reg, *mcfg);

    // Real CUDA device backends.
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devs;
    std::vector<lcomp::DeviceBackend*> dev_ptrs;
    for (const auto& g : cfg.hardware.gpus) {
        owned_devs.push_back(lcomp::make_cuda_sm120_device_backend(g.ref));
        dev_ptrs.push_back(owned_devs.back().get());
    }

    auto vram = lmem::VramAllocator(std::move(layout), dev_ptrs);

    lmem::PageAllocator pages(vram, dev_ptrs[0]);
    lmem::ExpertCache cache(vram, cfg, expert_bytes);

    for (int g = 0; g < vram.gpu_count(); ++g) {
        const auto& region = vram.region(g);
        const auto& gpu_layout = vram.layout().gpus[g];

        // (a) Region base pointers are device memory.
        EXPECT_TRUE(is_device_ptr(region.base))
            << "GPU " << g << ": base is not a device pointer";
        if (region.kv_main)
            EXPECT_TRUE(is_device_ptr(region.kv_main))
                << "GPU " << g << ": kv_main is not a device pointer";
        if (region.expert_stable)
            EXPECT_TRUE(is_device_ptr(region.expert_stable))
                << "GPU " << g << ": expert_stable is not a device pointer";

        // KV region bounds.
        auto kv_start = reinterpret_cast<uintptr_t>(region.kv_main);
        int64_t kv_len = gpu_layout.kv_main_bytes + gpu_layout.kv_speculation_bytes;

        // Expert region bounds.
        auto stable_start = reinterpret_cast<uintptr_t>(region.expert_stable);
        auto streaming_start = reinterpret_cast<uintptr_t>(region.expert_streaming);

        // (c) KV and expert regions must not overlap.
        if (kv_len > 0 && gpu_layout.expert_stable_bytes > 0) {
            EXPECT_FALSE(kv_start < stable_start + gpu_layout.expert_stable_bytes &&
                         stable_start < kv_start + kv_len)
                << "GPU " << g << ": KV overlaps expert stable";
        }

        // ── Allocate all KV pages ───────────────────────────────────────
        std::set<uintptr_t> all_addrs;
        std::vector<lmem::PageHandle> page_handles;

        int main_pages = pages.total_pages(g, lmem::Pool::kMain);
        int spec_pages = pages.total_pages(g, lmem::Pool::kSpeculation);

        for (int p = 0; p < main_pages; ++p) {
            auto h = pages.allocate(g, lmem::Pool::kMain);
            ASSERT_TRUE(h.has_value());
            auto addr = reinterpret_cast<uintptr_t>(h->gpu_ptr);
            // (b) Page pointer is device memory.
            EXPECT_TRUE(is_device_ptr(h->gpu_ptr))
                << "GPU " << g << ": KV page " << p << " not on device";
            // (d) No duplicate.
            EXPECT_TRUE(all_addrs.insert(addr).second)
                << "GPU " << g << ": duplicate KV page address";
            page_handles.push_back(*h);
        }
        for (int p = 0; p < spec_pages; ++p) {
            auto h = pages.allocate(g, lmem::Pool::kSpeculation);
            ASSERT_TRUE(h.has_value());
            EXPECT_TRUE(is_device_ptr(h->gpu_ptr));
            EXPECT_TRUE(all_addrs.insert(reinterpret_cast<uintptr_t>(h->gpu_ptr)).second);
            page_handles.push_back(*h);
        }

        // ── Allocate all expert slots ───────────────────────────────────
        int stable_slots = cache.total_slots(g, lmem::CacheZone::kStable);
        int streaming_slots = cache.total_slots(g, lmem::CacheZone::kStreaming);

        for (int s = 0; s < stable_slots; ++s) {
            auto ek = key(static_cast<uint32_t>(g * 1000 + s), 0);
            auto* addr = cache.reserve(ek, g, lmem::CacheZone::kStable);
            ASSERT_NE(addr, nullptr);
            // (b) Expert slot is device memory.
            EXPECT_TRUE(is_device_ptr(addr))
                << "GPU " << g << ": stable slot " << s << " not on device";
            // (d) No collision with KV pages.
            EXPECT_TRUE(all_addrs.insert(reinterpret_cast<uintptr_t>(addr)).second)
                << "GPU " << g << ": expert collides with KV page";
        }
        for (int s = 0; s < streaming_slots; ++s) {
            auto ek = key(static_cast<uint32_t>(g * 1000 + stable_slots + s), 1);
            auto* addr = cache.reserve(ek, g, lmem::CacheZone::kStreaming);
            ASSERT_NE(addr, nullptr);
            EXPECT_TRUE(is_device_ptr(addr));
            EXPECT_TRUE(all_addrs.insert(reinterpret_cast<uintptr_t>(addr)).second);
        }

        std::cout << "  GPU " << g << ": " << main_pages << " main pages, "
                  << spec_pages << " spec pages, "
                  << stable_slots << " stable slots, "
                  << streaming_slots << " streaming slots — "
                  << all_addrs.size() << " unique addresses\n";

        // ── Cleanup ─────────────────────────────────────────────────────
        for (auto& h : page_handles) pages.free(h);
        // Expert eviction not strictly needed (VramAllocator frees the region),
        // but verifies the evict path works.
        auto snap = cache.residency_snapshot(g);
        for (const auto& r : snap) cache.evict(r.key, g);
    }

    // (e) VramAllocator destructor will cudaFree — no CUDA errors expected.
}

// ═══════════════════════════════════════════════════════════════════════════
// Smoke: Allocation churn on real GPU memory
// ═══════════════════════════════════════════════════════════════════════════
//
// 50 cycles of mixed page + expert alloc/free on real CUDA memory.
// Verifies no CUDA errors, no leaks, and accounting stays consistent.

TEST_F(AllocatorSmoke, AllocationChurnOnRealGpu) {
    auto layout = lmem::compute_vram_layout(cfg, *reg, *mcfg);

    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_devs;
    std::vector<lcomp::DeviceBackend*> dev_ptrs;
    for (const auto& g : cfg.hardware.gpus) {
        owned_devs.push_back(lcomp::make_cuda_sm120_device_backend(g.ref));
        dev_ptrs.push_back(owned_devs.back().get());
    }

    auto vram = lmem::VramAllocator(std::move(layout), dev_ptrs);

    lmem::PageAllocator pages(vram, dev_ptrs[0]);
    lmem::ExpertCache cache(vram, cfg, expert_bytes);

    uint32_t rng = 42;
    auto next = [&]() -> uint32_t {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return rng;
    };

    struct LivePages { std::vector<lmem::PageHandle> handles; };
    struct LiveExperts { std::vector<lmem::ExpertKey> keys; };

    std::vector<LivePages> live_pages(vram.gpu_count());
    std::vector<LiveExperts> live_experts(vram.gpu_count());
    uint32_t expert_id = 0;

    auto verify = [&](const char* phase) {
        for (int g = 0; g < vram.gpu_count(); ++g) {
            int mf = pages.free_pages(g, lmem::Pool::kMain);
            int mu = pages.used_pages(g, lmem::Pool::kMain);
            int mt = pages.total_pages(g, lmem::Pool::kMain);
            EXPECT_EQ(mf + mu, mt)
                << phase << " GPU " << g << ": main page accounting";

            int sf = cache.free_slots(g, lmem::CacheZone::kStable);
            int su = cache.used_slots(g, lmem::CacheZone::kStable);
            int st = cache.total_slots(g, lmem::CacheZone::kStable);
            EXPECT_EQ(sf + su, st)
                << phase << " GPU " << g << ": stable slot accounting";

            int rf = cache.free_slots(g, lmem::CacheZone::kStreaming);
            int ru = cache.used_slots(g, lmem::CacheZone::kStreaming);
            int rt = cache.total_slots(g, lmem::CacheZone::kStreaming);
            EXPECT_EQ(rf + ru, rt)
                << phase << " GPU " << g << ": streaming slot accounting";
        }
        // Check no CUDA errors accumulated.
        cudaError_t err = cudaGetLastError();
        EXPECT_EQ(err, cudaSuccess)
            << phase << ": CUDA error: " << cudaGetErrorString(err);
    };

    verify("initial");

    for (int cycle = 0; cycle < 50; ++cycle) {
        int g = static_cast<int>(next() % vram.gpu_count());

        // Allocate pages.
        int to_alloc = 1 + static_cast<int>(next() % 10);
        for (int i = 0; i < to_alloc; ++i) {
            auto h = pages.allocate(g, lmem::Pool::kMain);
            if (h.has_value()) live_pages[g].handles.push_back(*h);
        }

        // Allocate expert slots.
        int to_reserve = 1 + static_cast<int>(next() % 3);
        for (int i = 0; i < to_reserve; ++i) {
            auto ek = key(expert_id / 256, expert_id % 256);
            ++expert_id;
            auto zone = (next() % 2 == 0) ? lmem::CacheZone::kStable
                                           : lmem::CacheZone::kStreaming;
            if (cache.reserve(ek, g, zone) != nullptr) {
                cache.mark_all_ready(ek, g);
                live_experts[g].keys.push_back(ek);
            }
        }

        // Free some pages.
        auto& ph = live_pages[g].handles;
        if (!ph.empty() && (next() % 2) == 0) {
            int n = 1 + static_cast<int>(next() % ph.size());
            n = std::min(n, static_cast<int>(ph.size()));
            for (int f = 0; f < n; ++f) { pages.free(ph.back()); ph.pop_back(); }
        }

        // Evict some experts.
        auto& ek_vec = live_experts[g].keys;
        if (!ek_vec.empty() && (next() % 2) == 0) {
            int n = 1 + static_cast<int>(next() % ek_vec.size());
            n = std::min(n, static_cast<int>(ek_vec.size()));
            for (int f = 0; f < n; ++f) { cache.evict(ek_vec.back(), g); ek_vec.pop_back(); }
        }

        verify(("cycle " + std::to_string(cycle)).c_str());
    }

    // Full cleanup.
    for (int g = 0; g < vram.gpu_count(); ++g) {
        for (auto& h : live_pages[g].handles) pages.free(h);
        for (auto& ek : live_experts[g].keys) cache.evict(ek, g);
    }

    // Final: everything returned.
    for (int g = 0; g < vram.gpu_count(); ++g) {
        EXPECT_EQ(pages.free_pages(g, lmem::Pool::kMain),
                  pages.total_pages(g, lmem::Pool::kMain))
            << "GPU " << g << ": main page leak";
        EXPECT_EQ(cache.free_slots(g, lmem::CacheZone::kStable),
                  cache.total_slots(g, lmem::CacheZone::kStable))
            << "GPU " << g << ": stable slot leak";
        EXPECT_EQ(cache.free_slots(g, lmem::CacheZone::kStreaming),
                  cache.total_slots(g, lmem::CacheZone::kStreaming))
            << "GPU " << g << ": streaming slot leak";
    }
    EXPECT_EQ(cache.total_resident(), 0);

    std::cout << "\n  Churn complete: 50 cycles, no leaks, no CUDA errors.\n";
}
