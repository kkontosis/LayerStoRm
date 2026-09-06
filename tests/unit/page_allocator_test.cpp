#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "config/config_resolver.h"
#include "core/device_backend.h"
#include "core/memory/page_allocator.h"
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

// Small config for focused tests: 1 GPU, explicit page counts
lc::Config small_config(int max_pages = 100) {
    auto j = nlohmann::json{
        {"model", {
            {"architecture",            "deepseek_v3"},
            {"weights_path",            "/data/models/test/"},
            {"weights_format",          "safetensors"},
            {"num_hidden_layers",       4},
            {"hidden_size",             256},
            {"num_attention_heads",     4},
            {"num_key_value_heads",     4},
            {"intermediate_size",       512},
            {"n_routed_experts",        0},
            {"n_shared_experts",        0},
            {"num_experts_per_tok",     0},
            {"n_group",                 1},
            {"topk_group",              1},
            {"vocab_size",              1024},
            {"max_position_embeddings", 2048},
            {"kv_lora_rank",            0},
            {"q_lora_rank",             0},
            {"qk_rope_head_dim",        32},
            {"qk_nope_head_dim",        32},
            {"v_head_dim",              64},
            {"first_k_dense_replace",   999},
            {"moe_layer_freq",          1},
            {"index_topk",              0},
            {"num_nextn_predict_layers", 0},
            {"moe_intermediate_size",   512},
        }},
        {"quantization", {{"weights", "fp8_e4m3"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {
            {"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
            {"system_ram_gb", 64}}},
    };
    auto cfg = lc::parse_config(j);
    cfg.memory.kv_cache.max_pages_per_gpu = max_pages;
    return cfg;
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

struct TestAllocators {
    NullBackends backends;
    lmem::VramAllocator vram;
    lmem::PageAllocator pages;
};

TestAllocators make_test_allocators(const lc::Config& cfg) {
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);
    auto pages = lmem::PageAllocator(vram, nb.ptrs[0]);
    return TestAllocators{std::move(nb), std::move(vram), std::move(pages)};
}

TestAllocators make_v32_test_allocators() {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);
    auto pages = lmem::PageAllocator(vram, nb.ptrs[0]);
    return TestAllocators{std::move(nb), std::move(vram), std::move(pages)};
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, ConstructionGpuCount) {
    auto [nb_, vram, pages] = make_v32_test_allocators();
    EXPECT_EQ(pages.gpu_count(), 4);
}

TEST(PageAllocator, ConstructionFreeCountsMatchLayout) {
    auto [nb_, vram, pages] = make_v32_test_allocators();
    for (int i = 0; i < pages.gpu_count(); ++i) {
        const auto& gpu = vram.layout().gpus[i];
        EXPECT_EQ(pages.total_pages(i, lmem::Pool::kMain), gpu.kv_main_pages)
            << "GPU " << i;
        EXPECT_EQ(pages.total_pages(i, lmem::Pool::kSpeculation),
                  gpu.kv_speculation_pages)
            << "GPU " << i;
        EXPECT_EQ(pages.free_pages(i, lmem::Pool::kMain), gpu.kv_main_pages)
            << "GPU " << i;
        EXPECT_EQ(pages.free_pages(i, lmem::Pool::kSpeculation),
                  gpu.kv_speculation_pages)
            << "GPU " << i;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Basic allocation
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, AllocateMainReturnsValidHandle) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    auto handle = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(handle->gpu_idx, 0);
    EXPECT_GE(handle->page_idx, 0);
    EXPECT_NE(handle->gpu_ptr, nullptr);
    EXPECT_EQ(pages.meta(*handle).refcount, 1u);
    EXPECT_EQ(pages.meta(*handle).pool, lmem::Pool::kMain);
}

TEST(PageAllocator, AllocateSpecReturnsValidHandle) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    auto handle = pages.allocate(0, lmem::Pool::kSpeculation);
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(handle->gpu_idx, 0);
    EXPECT_NE(handle->gpu_ptr, nullptr);
    EXPECT_EQ(pages.meta(*handle).refcount, 1u);
    EXPECT_EQ(pages.meta(*handle).pool, lmem::Pool::kSpeculation);
}

TEST(PageAllocator, AllocateDecrementsFreeCount) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    int before = pages.free_pages(0, lmem::Pool::kMain);
    auto handle = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), before - 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Free
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, FreeReturnsToFreeList) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    int before = pages.free_pages(0, lmem::Pool::kMain);
    auto handle = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(handle.has_value());
    pages.free(*handle);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), before);
}

TEST(PageAllocator, FreeAndReallocate) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    auto h1 = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(h1.has_value());
    int idx1 = h1->page_idx;
    pages.free(*h1);

    // Reallocate — should get the same page back (stack LIFO)
    auto h2 = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(h2.has_value());
    EXPECT_EQ(h2->page_idx, idx1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Pool isolation
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, PoolIsolationExhaustMain) {
    auto [nb_, vram, pages] = make_test_allocators(small_config(10));
    int main_total = pages.total_pages(0, lmem::Pool::kMain);
    int spec_before = pages.free_pages(0, lmem::Pool::kSpeculation);

    // Exhaust main pool
    std::vector<lmem::PageHandle> handles;
    for (int i = 0; i < main_total; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kMain);
        ASSERT_TRUE(h.has_value()) << "Failed at allocation " << i;
        handles.push_back(*h);
    }

    // Main exhausted
    EXPECT_FALSE(pages.allocate(0, lmem::Pool::kMain).has_value());

    // Spec pool unaffected
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation), spec_before);
    auto spec_h = pages.allocate(0, lmem::Pool::kSpeculation);
    EXPECT_TRUE(spec_h.has_value());
}

TEST(PageAllocator, PoolIsolationExhaustSpec) {
    auto [nb_, vram, pages] = make_test_allocators(small_config(10));
    int spec_total = pages.total_pages(0, lmem::Pool::kSpeculation);
    int main_before = pages.free_pages(0, lmem::Pool::kMain);

    // Exhaust spec pool
    std::vector<lmem::PageHandle> handles;
    for (int i = 0; i < spec_total; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kSpeculation);
        ASSERT_TRUE(h.has_value()) << "Failed at allocation " << i;
        handles.push_back(*h);
    }

    // Spec exhausted
    EXPECT_FALSE(pages.allocate(0, lmem::Pool::kSpeculation).has_value());

    // Main pool unaffected
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), main_before);
}

TEST(PageAllocator, ExhaustBothPools) {
    auto [nb_, vram, pages] = make_test_allocators(small_config(10));
    int main_total = pages.total_pages(0, lmem::Pool::kMain);
    int spec_total = pages.total_pages(0, lmem::Pool::kSpeculation);

    std::vector<lmem::PageHandle> handles;
    for (int i = 0; i < main_total; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kMain);
        ASSERT_TRUE(h.has_value());
        handles.push_back(*h);
    }
    for (int i = 0; i < spec_total; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kSpeculation);
        ASSERT_TRUE(h.has_value());
        handles.push_back(*h);
    }

    EXPECT_FALSE(pages.allocate(0, lmem::Pool::kMain).has_value());
    EXPECT_FALSE(pages.allocate(0, lmem::Pool::kSpeculation).has_value());
}

// ═══════════════════════════════════════════════════════════════════════════
// Promotion
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, PromoteChangesPoolTag) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    auto handle = pages.allocate(0, lmem::Pool::kSpeculation);
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(pages.meta(*handle).pool, lmem::Pool::kSpeculation);

    pages.promote(*handle);
    EXPECT_EQ(pages.meta(*handle).pool, lmem::Pool::kMain);
}

TEST(PageAllocator, PromoteNoDataCopy) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    auto handle = pages.allocate(0, lmem::Pool::kSpeculation);
    ASSERT_TRUE(handle.has_value());

    // Write a known pattern
    std::memset(handle->gpu_ptr, 0xAB,
                vram.layout().kv_bytes_per_page);
    void* original_ptr = handle->gpu_ptr;

    pages.promote(*handle);

    // Pointer unchanged, data unchanged
    EXPECT_EQ(handle->gpu_ptr, original_ptr);
    EXPECT_EQ(static_cast<uint8_t*>(handle->gpu_ptr)[0], 0xAB);
}

TEST(PageAllocator, PromoteLeavesHandlePoolFieldStale) {
    // TD-GOLDEN-KV-SPEC documentation: promote() updates the allocator's
    // PageMeta, but COPIES of the PageHandle held by callers (e.g. the
    // dispatcher's seq_pages_) keep their original pool tag. A future
    // CMD_SEQ_PROMOTE handler must update every stored handle's .pool to
    // kMain itself — otherwise ensure_pages (which routes growth via
    // pages[0].pool) keeps drawing from the speculation pool.
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    auto handle = pages.allocate(0, lmem::Pool::kSpeculation);
    ASSERT_TRUE(handle.has_value());
    lmem::PageHandle stored = *handle;  // caller-side copy (as in seq_pages_)

    pages.promote(*handle);

    EXPECT_EQ(pages.meta(stored).pool, lmem::Pool::kMain);   // meta updated
    EXPECT_EQ(stored.pool, lmem::Pool::kSpeculation);        // copy is stale
    pages.free(stored);
}

TEST(PageAllocator, PromoteLayerMajorGroupFreesToMainPool) {
    // TD-GOLDEN-KV-SPEC: promoting a whole layer-major group (all L physical
    // pages of one logical page, INV-KV-LAYER) moves every page to the main
    // pool — frees land in the main free list, spec free list regains none.
    constexpr int kL = 6;
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    const int main_before = pages.free_pages(0, lmem::Pool::kMain);
    const int spec_before = pages.free_pages(0, lmem::Pool::kSpeculation);

    std::vector<lmem::PageHandle> group;
    for (int l = 0; l < kL; ++l) {
        auto h = pages.allocate(0, lmem::Pool::kSpeculation);
        ASSERT_TRUE(h.has_value());
        pages.meta(*h).layer_index = static_cast<uint32_t>(l);
        group.push_back(*h);
    }
    ASSERT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation), spec_before - kL);

    for (auto& h : group) pages.promote(h);
    for (auto& h : group) pages.free(h);

    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), main_before + kL);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation), spec_before - kL);
}

TEST(PageAllocator, PromotedPageFreesToMainPool) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    int main_free_before = pages.free_pages(0, lmem::Pool::kMain);
    int spec_free_before = pages.free_pages(0, lmem::Pool::kSpeculation);

    auto handle = pages.allocate(0, lmem::Pool::kSpeculation);
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation),
              spec_free_before - 1);

    pages.promote(*handle);
    pages.free(*handle);

    // Page went to main free list, not spec
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), main_free_before + 1);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation),
              spec_free_before - 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Copy-on-write
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, AddRefIncrementsRefcount) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    auto handle = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(pages.meta(*handle).refcount, 1u);

    pages.add_ref(*handle);
    EXPECT_EQ(pages.meta(*handle).refcount, 2u);

    pages.add_ref(*handle);
    EXPECT_EQ(pages.meta(*handle).refcount, 3u);
}

TEST(PageAllocator, FreeWithRefcountGt1OnlyDecrements) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    int free_before = pages.free_pages(0, lmem::Pool::kMain);

    auto handle = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(handle.has_value());
    pages.add_ref(*handle);  // refcount = 2
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), free_before - 1);

    pages.free(*handle);  // refcount = 1
    EXPECT_EQ(pages.meta(*handle).refcount, 1u);
    // Still not on free list
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), free_before - 1);

    pages.free(*handle);  // refcount = 0 → freed
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), free_before);
}

TEST(PageAllocator, CowCopyRefcount1NoOp) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    auto handle = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(handle.has_value());

    auto result = pages.cow_copy(*handle);
    EXPECT_EQ(result.gpu_idx, handle->gpu_idx);
    EXPECT_EQ(result.page_idx, handle->page_idx);
    EXPECT_EQ(result.gpu_ptr, handle->gpu_ptr);
}

TEST(PageAllocator, CowCopyRefcountGt1AllocatesNew) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    auto handle = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(handle.has_value());

    // Write known data
    int64_t bpp = vram.layout().kv_bytes_per_page;
    std::memset(handle->gpu_ptr, 0xCD, bpp);

    pages.add_ref(*handle);  // refcount = 2

    int free_before = pages.free_pages(0, lmem::Pool::kMain);
    auto new_handle = pages.cow_copy(*handle);

    // New page allocated
    EXPECT_NE(new_handle.page_idx, handle->page_idx);
    EXPECT_NE(new_handle.gpu_ptr, handle->gpu_ptr);
    EXPECT_EQ(new_handle.gpu_idx, handle->gpu_idx);

    // Data copied
    EXPECT_EQ(static_cast<uint8_t*>(new_handle.gpu_ptr)[0], 0xCD);
    EXPECT_EQ(static_cast<uint8_t*>(new_handle.gpu_ptr)[bpp - 1], 0xCD);

    // Old refcount decremented
    EXPECT_EQ(pages.meta(*handle).refcount, 1u);
    // New refcount = 1
    EXPECT_EQ(pages.meta(new_handle).refcount, 1u);

    // One more page consumed from free list
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), free_before - 1);
}

TEST(PageAllocator, CowCopyPreservesMetadata) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    auto handle = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(handle.has_value());

    auto& m = pages.meta(*handle);
    m.layer_index = 42;
    m.sequence_id = 12345;
    m.token_start = 100;
    m.token_end = 116;

    pages.add_ref(*handle);
    auto new_handle = pages.cow_copy(*handle);

    const auto& nm = pages.meta(new_handle);
    EXPECT_EQ(nm.layer_index, 42u);
    EXPECT_EQ(nm.sequence_id, 12345u);
    EXPECT_EQ(nm.token_start, 100u);
    EXPECT_EQ(nm.token_end, 116u);
    EXPECT_EQ(nm.refcount, 1u);
    EXPECT_EQ(nm.pool, lmem::Pool::kMain);
}

// ═══════════════════════════════════════════════════════════════════════════
// Bulk free
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, FreeSequenceFreesMatching) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    int free_before = pages.free_pages(0, lmem::Pool::kMain);

    // Allocate 3 pages for sequence 100
    std::vector<lmem::PageHandle> seq100;
    for (int i = 0; i < 3; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kMain);
        ASSERT_TRUE(h.has_value());
        pages.meta(*h).sequence_id = 100;
        seq100.push_back(*h);
    }

    // Allocate 2 pages for sequence 200
    std::vector<lmem::PageHandle> seq200;
    for (int i = 0; i < 2; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kMain);
        ASSERT_TRUE(h.has_value());
        pages.meta(*h).sequence_id = 200;
        seq200.push_back(*h);
    }

    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), free_before - 5);

    // Free sequence 100
    pages.free_sequence(0, 100);

    // 3 pages returned for seq 100, seq 200 still allocated
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), free_before - 2);

    // seq 200 pages still alive
    for (const auto& h : seq200) {
        EXPECT_EQ(pages.meta(h).refcount, 1u);
    }
}

TEST(PageAllocator, FreeSequenceHandlesCoW) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());

    auto h1 = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(h1.has_value());
    pages.meta(*h1).sequence_id = 100;
    pages.add_ref(*h1);  // refcount = 2 (shared with another sequence)

    int free_before = pages.free_pages(0, lmem::Pool::kMain);

    pages.free_sequence(0, 100);

    // Refcount decremented to 1, page NOT freed
    EXPECT_EQ(pages.meta(*h1).refcount, 1u);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), free_before);
}

// ═══════════════════════════════════════════════════════════════════════════
// Metadata
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, MetadataReadWrite) {
    auto [nb_, vram, pages] = make_test_allocators(small_config());
    auto handle = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(handle.has_value());

    auto& m = pages.meta(*handle);
    m.layer_index = 60;
    m.sequence_id = 999999;
    m.token_start = 1024;
    m.token_end = 1040;

    const auto& cm = pages.meta(*handle);
    EXPECT_EQ(cm.layer_index, 60u);
    EXPECT_EQ(cm.sequence_id, 999999u);
    EXPECT_EQ(cm.token_start, 1024u);
    EXPECT_EQ(cm.token_end, 1040u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-GPU
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, MultiGpuIndependence) {
    auto [nb_, vram, pages] = make_v32_test_allocators();
    ASSERT_GE(pages.gpu_count(), 2);

    int gpu0_before = pages.free_pages(0, lmem::Pool::kMain);
    int gpu1_before = pages.free_pages(1, lmem::Pool::kMain);

    auto h0 = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(h0.has_value());
    EXPECT_EQ(h0->gpu_idx, 0);

    // GPU 0 decremented, GPU 1 unaffected
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), gpu0_before - 1);
    EXPECT_EQ(pages.free_pages(1, lmem::Pool::kMain), gpu1_before);
}

// ═══════════════════════════════════════════════════════════════════════════
// Pointer arithmetic
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, GpuPtrArithmetic) {
    auto [nb_, vram, pages] = make_test_allocators(small_config(20));
    int64_t bpp = vram.layout().kv_bytes_per_page;

    // Allocate several pages and verify pointer arithmetic
    for (int i = 0; i < 5; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kMain);
        ASSERT_TRUE(h.has_value());
        void* expected = static_cast<char*>(vram.region(0).kv_main) +
                         h->page_idx * bpp;
        EXPECT_EQ(h->gpu_ptr, expected) << "Page " << h->page_idx;
    }

    // Same for speculation pool (uses kv_speculation base)
    int main_pages = vram.layout().gpus[0].kv_main_pages;
    for (int i = 0; i < 3; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kSpeculation);
        ASSERT_TRUE(h.has_value());
        void* expected = static_cast<char*>(vram.region(0).kv_speculation) +
                         (h->page_idx - main_pages) * bpp;
        EXPECT_EQ(h->gpu_ptr, expected) << "Spec page " << h->page_idx;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Used pages count
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, UsedPagesCount) {
    auto [nb_, vram, pages] = make_test_allocators(small_config(20));

    EXPECT_EQ(pages.used_pages(0, lmem::Pool::kMain), 0);

    std::vector<lmem::PageHandle> handles;
    for (int i = 0; i < 5; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kMain);
        ASSERT_TRUE(h.has_value());
        handles.push_back(*h);
    }
    EXPECT_EQ(pages.used_pages(0, lmem::Pool::kMain), 5);

    pages.free(handles[0]);
    pages.free(handles[1]);
    EXPECT_EQ(pages.used_pages(0, lmem::Pool::kMain), 3);
}

// ═══════════════════════════════════════════════════════════════════════════
// Zero speculation pages
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, ZeroSpeculationPages) {
    auto cfg = small_config(20);
    cfg.memory.kv_cache.speculation_pool_fraction = 0.0;
    auto [nb_, vram, pages] = make_test_allocators(cfg);

    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kSpeculation), 0);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation), 0);
    EXPECT_FALSE(pages.allocate(0, lmem::Pool::kSpeculation).has_value());

    // Main pool still works
    auto h = pages.allocate(0, lmem::Pool::kMain);
    EXPECT_TRUE(h.has_value());
}

// ═══════════════════════════════════════════════════════════════════════════
// V3.2 realistic config
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocator, V32RealisticPageCounts) {
    auto [nb_, vram, pages] = make_v32_test_allocators();

    for (int i = 0; i < pages.gpu_count(); ++i) {
        int main_total = pages.total_pages(i, lmem::Pool::kMain);
        int spec_total = pages.total_pages(i, lmem::Pool::kSpeculation);

        // All pages should be free initially
        EXPECT_EQ(pages.free_pages(i, lmem::Pool::kMain), main_total)
            << "GPU " << i;
        EXPECT_EQ(pages.free_pages(i, lmem::Pool::kSpeculation), spec_total)
            << "GPU " << i;
        EXPECT_EQ(pages.used_pages(i, lmem::Pool::kMain), 0) << "GPU " << i;
        EXPECT_EQ(pages.used_pages(i, lmem::Pool::kSpeculation), 0)
            << "GPU " << i;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Indexer K pool tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocatorIndexerK, AllocateAndFree) {
    auto [nb_, vram, pages] = make_v32_test_allocators();

    // TP GPUs (0,1) should have indexer K pages (V3.2 is DSA)
    int ik_total_0 = pages.total_pages(0, lmem::Pool::kIndexerK);
    EXPECT_GT(ik_total_0, 0) << "TP GPU 0 should have indexer K pages";
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kIndexerK), ik_total_0);

    // Allocate a page
    auto h = pages.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(h->pool, lmem::Pool::kIndexerK);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kIndexerK), ik_total_0 - 1);
    EXPECT_EQ(pages.used_pages(0, lmem::Pool::kIndexerK), 1);

    // Free it
    pages.free(*h);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kIndexerK), ik_total_0);
    EXPECT_EQ(pages.used_pages(0, lmem::Pool::kIndexerK), 0);
}

TEST(PageAllocatorIndexerK, SharesTheSlabPoolWithMain) {
    // S4 (TD-INDEXER-POOL-ELASTIC): on slabbed models the indexer-K pool
    // is ELASTIC — an indexer page claims a WHOLE slab from the SHARED
    // free-slab list, so kMain's free count drops by pages_per_slab and
    // returns whole on free. Speculation stays separate.
    auto [nb_, vram, pages] = make_v32_test_allocators();

    int main_free_before = pages.free_pages(0, lmem::Pool::kMain);
    int spec_free_before = pages.free_pages(0, lmem::Pool::kSpeculation);

    auto h = pages.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(h.has_value());

    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain),
              main_free_before - pages.pages_per_slab());
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation), spec_free_before);

    pages.free(*h);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), main_free_before);
}

TEST(PageAllocatorIndexerK, DifferentBytesPerPage) {
    auto [nb_, vram, pages] = make_v32_test_allocators();

    // Indexer K page size: (128 B FP8 K + 4 B F32 scale)/token * 8192 tokens
    // (coarse default — ~1 MiB chunks make pool growth behave like amortized
    // doubling with no copies; see TD-GLM-INDEXER-PAGED).
    // KV page size: 644 bytes/token * 16 tokens = 10304 bytes
    int64_t kv_bpp = vram.layout().kv_bytes_per_page;
    int64_t ik_bpp = vram.layout().indexer_k_bytes_per_page;
    EXPECT_NE(kv_bpp, ik_bpp);
    EXPECT_EQ(ik_bpp, (128 + 4) * 8192);
    EXPECT_EQ(kv_bpp, 644 * 16);    // 10304
}

TEST(PageAllocatorIndexerK, SlabStrideAndGeometry) {
    // S1 (TD-INDEXER-POOL-ELASTIC): the indexer pool is slabbed — physical
    // stride = slab_bytes (105 kMain pages on V3.2 SnapMLA), one indexer
    // page per slab; allocate == whole-slab claim, free == whole-slab
    // release back to the S1 free-slab list.
    auto [nb_, vram, pages] = make_v32_test_allocators();

    EXPECT_EQ(pages.pages_per_slab(), 105);
    EXPECT_EQ(pages.slab_bytes(), 105 * vram.layout().kv_bytes_per_page);
    EXPECT_GE(pages.slab_bytes(), vram.layout().indexer_k_bytes_per_page);
    // kMain span is a whole number of slabs on both TP GPUs.
    for (int g = 0; g < 2; ++g)
        EXPECT_EQ(pages.total_pages(g, lmem::Pool::kMain) %
                      pages.pages_per_slab(), 0)
            << "GPU " << g;

    // Consecutive claims stride by slab_bytes from the indexer span base.
    auto h0 = pages.allocate(0, lmem::Pool::kIndexerK);
    auto h1 = pages.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(h0 && h1);
    const auto base = static_cast<char*>(vram.region(0).indexer_k);
    EXPECT_EQ(static_cast<char*>(h0->gpu_ptr) - base,
              static_cast<int64_t>(h0->page_idx) * pages.slab_bytes());
    EXPECT_EQ(static_cast<char*>(h1->gpu_ptr) - base,
              static_cast<int64_t>(h1->page_idx) * pages.slab_bytes());
    // The slab is an aligned run of flat kMain page indices: the span END
    // (kv_main base) sits exactly indexer_k_pages slabs past the base.
    EXPECT_EQ(static_cast<char*>(vram.region(0).kv_main) - base,
              static_cast<int64_t>(vram.layout().gpus[0].indexer_k_pages) *
                  pages.slab_bytes());
    pages.free(*h0);
    pages.free(*h1);
}

TEST(PageAllocatorIndexerK, ZeroPagesForNonDsa) {
    // small_config has index_topk=0 (non-DSA)
    auto [nb_, vram, pages] = make_test_allocators(small_config(20));

    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kIndexerK), 0);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kIndexerK), 0);

    // Allocation returns nullopt
    auto h = pages.allocate(0, lmem::Pool::kIndexerK);
    EXPECT_FALSE(h.has_value());
}

TEST(PageAllocatorIndexerK, NonTpGpuZeroIndexerK) {
    auto [nb_, vram, pages] = make_v32_test_allocators();

    // Non-TP GPUs (2,3) should have zero indexer K pages
    EXPECT_EQ(pages.total_pages(2, lmem::Pool::kIndexerK), 0);
    EXPECT_EQ(pages.total_pages(3, lmem::Pool::kIndexerK), 0);
}

TEST(PageAllocatorIndexerK, PtrWithinSharedSlabRegion) {
    // S4 elastic: no dedicated indexer span — the page is a whole slab of
    // the shared kv_main region, at kv_main base + slab_id * slab_bytes.
    auto [nb_, vram, pages] = make_v32_test_allocators();

    auto kv_base = reinterpret_cast<uintptr_t>(vram.region(0).kv_main);
    const int64_t span =
        static_cast<int64_t>(vram.layout().gpus[0].kv_main_pages) *
        vram.layout().kv_bytes_per_page;

    auto h = pages.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(h.has_value());
    auto addr = reinterpret_cast<uintptr_t>(h->gpu_ptr);
    EXPECT_GE(addr, kv_base);
    EXPECT_LT(addr, kv_base + static_cast<uintptr_t>(span));
    EXPECT_EQ(addr - kv_base,
              static_cast<uintptr_t>(h->page_idx) *
                  static_cast<uintptr_t>(pages.slab_bytes()));

    pages.free(*h);
}

// ═══ S4 elastic indexer-K locks (TD-INDEXER-POOL-ELASTIC) ═══════════════════

TEST(PageAllocatorIndexerK, ElasticHonorsHeadroomFloor) {
    // An elastic indexer claim must respect the INV-4.9f kMain headroom
    // reservation: a claim that would dip kv_free_pages below the floor is
    // refused (retryable exhaustion upstream), and lifting the floor lets
    // the identical claim succeed.
    auto [nb_, vram, pages] = make_v32_test_allocators();
    const int free_before = pages.free_pages(0, lmem::Pool::kMain);
    ASSERT_GT(free_before, 0);

    // Floor above (free - one slab): the next side claim must refuse.
    pages.configure_headroom(lmem::PageAllocator::HeadroomConfig{
        .max_concurrent_forks = 0,
        .max_concurrent_sequences = 1,
        .page_growth_chunk_pages = free_before - pages.pages_per_slab() + 1,
    });
    EXPECT_FALSE(pages.allocate(0, lmem::Pool::kIndexerK).has_value());

    pages.configure_headroom(lmem::PageAllocator::HeadroomConfig{});
    auto h = pages.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(h.has_value());
    pages.free(*h);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), free_before);
}

TEST(PageAllocatorIndexerK, ElasticSlabInvisibleToKvPlacement) {
    // A side-claimed slab is marked fully used: KV sequence allocation must
    // never land a page inside it, and the slab returns whole to the shared
    // list on free — claimable by KV afterwards.
    auto [nb_, vram, pages] = make_v32_test_allocators();
    auto ik = pages.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(ik.has_value());
    const int ik_slab = ik->page_idx;
    const int pps = pages.pages_per_slab();

    // Fill a KV sequence run past one slab — no page may fall in ik_slab.
    std::vector<lmem::PageHandle> kv;
    for (int i = 0; i < pps + 1; ++i) {
        auto h = pages.allocate_for_sequence(
            0, lmem::Pool::kMain, /*seq=*/7, /*layer=*/0,
            /*token_start=*/static_cast<uint32_t>(i * 16),
            /*token_end=*/static_cast<uint32_t>(i * 16 + 16));
        ASSERT_TRUE(h.has_value());
        EXPECT_NE(h->page_idx / pps, ik_slab)
            << "KV page landed inside a side-claimed slab";
        kv.push_back(*h);
    }
    for (auto& h : kv) pages.free(h);
    pages.free(*ik);

    // The released slab is claimable again (by anyone).
    auto again = pages.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(again.has_value());
    pages.free(*again);
}

TEST(PageAllocatorIndexerK, ElasticFragmentationCountsSideSlabs) {
    // kv_fragmentation must stay consistent: a side-claimed slab is a live,
    // fully-used slab (no fragmented free pages inside it).
    auto [nb_, vram, pages] = make_v32_test_allocators();
    const auto before = pages.kv_fragmentation(0);
    auto ik = pages.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(ik.has_value());
    const auto during = pages.kv_fragmentation(0);
    EXPECT_EQ(during.free_slabs, before.free_slabs - 1);
    EXPECT_EQ(during.live_slabs, before.live_slabs + 1);
    EXPECT_EQ(during.used_pages, before.used_pages + pages.pages_per_slab());
    EXPECT_EQ(during.fragmented_free_pages, before.fragmented_free_pages);
    pages.free(*ik);
    const auto after = pages.kv_fragmentation(0);
    EXPECT_EQ(after.free_slabs, before.free_slabs);
    EXPECT_EQ(after.used_pages, before.used_pages);
}

TEST(PageAllocatorIndexerK, MetadataAccess) {
    auto [nb_, vram, pages] = make_v32_test_allocators();

    auto h = pages.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(h.has_value());

    auto& m = pages.meta(*h);
    EXPECT_EQ(m.refcount, 1u);
    EXPECT_EQ(m.pool, lmem::Pool::kIndexerK);
    m.layer_index = 42;
    m.sequence_id = 100;
    EXPECT_EQ(pages.meta(*h).layer_index, 42u);

    pages.free(*h);
}

TEST(PageAllocatorIndexerK, FreeSequenceScansIndexerK) {
    auto [nb_, vram, pages] = make_v32_test_allocators();

    int ik_total = pages.total_pages(0, lmem::Pool::kIndexerK);
    ASSERT_GT(ik_total, 2);

    auto h1 = pages.allocate(0, lmem::Pool::kIndexerK);
    auto h2 = pages.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(h1.has_value());
    ASSERT_TRUE(h2.has_value());

    pages.meta(*h1).sequence_id = 999;
    pages.meta(*h2).sequence_id = 888;

    EXPECT_EQ(pages.used_pages(0, lmem::Pool::kIndexerK), 2);

    // free_sequence should free only the matching one
    pages.free_sequence(0, 999);
    EXPECT_EQ(pages.used_pages(0, lmem::Pool::kIndexerK), 1);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kIndexerK), ik_total - 1);

    pages.free(*h2);
}

// ═══════════════════════════════════════════════════════════════════════════
// TurboQuant KV cache format
// ═══════════════════════════════════════════════════════════════════════════

namespace {

lc::Config v32_tq_config() {
    auto cfg = v32_config();
    cfg.compute.attention_backend = lc::AttentionBackendType::turboquant_mla;
    return cfg;
}

TestAllocators make_v32_tq_test_allocators() {
    auto cfg = v32_tq_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);
    auto pages = lmem::PageAllocator(vram, nb.ptrs[0]);
    return TestAllocators{std::move(nb), std::move(vram), std::move(pages)};
}

}  // namespace

TEST(PageAllocatorTQ, FormatIsSnapMlaByDefault) {
    auto [nb_, vram, pages] = make_v32_test_allocators();
    EXPECT_EQ(pages.kv_cache_format(), lmem::KvCacheFormat::kSnapMlaFp8);
    EXPECT_EQ(vram.layout().kv_cache_format, lmem::KvCacheFormat::kSnapMlaFp8);
}

TEST(PageAllocatorTQ, FormatIsTurboQuant) {
    auto [nb_, vram, pages] = make_v32_tq_test_allocators();
    EXPECT_EQ(pages.kv_cache_format(), lmem::KvCacheFormat::kTurboQuantMse4);
    EXPECT_EQ(vram.layout().kv_cache_format, lmem::KvCacheFormat::kTurboQuantMse4);
}

TEST(PageAllocatorTQ, TqPageSize386x16) {
    auto [nb_, vram, pages] = make_v32_tq_test_allocators();
    // TQ: 386 bytes/token × 16 tokens/page = 6176 bytes/page
    EXPECT_EQ(vram.layout().kv_bytes_per_page, 386 * 16);
}

TEST(PageAllocatorTQ, AllocFreeCycle) {
    auto [nb_, vram, pages] = make_v32_tq_test_allocators();
    int before = pages.free_pages(0, lmem::Pool::kMain);
    ASSERT_GT(before, 0);

    auto h = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), before - 1);

    // Write to full page extent (6176 bytes)
    int64_t bpp = vram.layout().kv_bytes_per_page;
    EXPECT_EQ(bpp, 6176);
    std::memset(h->gpu_ptr, 0xBE, bpp);
    EXPECT_EQ(static_cast<uint8_t*>(h->gpu_ptr)[bpp - 1], 0xBE);

    pages.free(*h);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), before);
}

TEST(PageAllocatorTQ, MorePagesThanSnapMla) {
    auto [nb_snap, vram_snap, pages_snap] = make_v32_test_allocators();
    auto [nb_tq, vram_tq, pages_tq] = make_v32_tq_test_allocators();

    // TQ pages are smaller (6176 vs 10304), so more pages fit in same VRAM
    for (int i = 0; i < pages_snap.gpu_count(); ++i) {
        int snap_total = pages_snap.total_pages(i, lmem::Pool::kMain) +
                         pages_snap.total_pages(i, lmem::Pool::kSpeculation);
        int tq_total = pages_tq.total_pages(i, lmem::Pool::kMain) +
                       pages_tq.total_pages(i, lmem::Pool::kSpeculation);
        if (snap_total > 0)
            EXPECT_GT(tq_total, snap_total) << "GPU " << i;
    }
}

TEST(PageAllocatorTQ, DcpCompatTokenRouting) {
    auto [nb_, vram, pages] = make_v32_tq_test_allocators();

    lmem::DcpConfig dcp;
    dcp.dcp_size = 2;
    dcp.dcp_chunk_size = 16;
    dcp.page_size_tokens = 16;
    dcp.tp_gpu_indices = {0, 1};
    pages.set_dcp_config(dcp);

    // Token routing is format-independent
    EXPECT_EQ(pages.dcp_gpu_for_token(0), 0);
    EXPECT_EQ(pages.dcp_gpu_for_token(16), 1);
    EXPECT_EQ(pages.dcp_gpu_for_token(32), 0);

    // DCP append works
    auto h = pages.allocate_for_dcp_append(/*seq_id=*/1, /*token_pos=*/0,
                                            /*layer_index=*/0);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(h->gpu_idx, 0);
    pages.free(*h);
}

TEST(PageAllocatorTQ, PointerArithmeticUsesTqPageSize) {
    auto [nb_, vram, pages] = make_v32_tq_test_allocators();
    int64_t bpp = vram.layout().kv_bytes_per_page;
    EXPECT_EQ(bpp, 6176);

    auto h1 = pages.allocate(0, lmem::Pool::kMain);
    auto h2 = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(h1.has_value());
    ASSERT_TRUE(h2.has_value());

    // Both pointers should be at expected offsets from kv_main base
    void* base = vram.region(0).kv_main;
    auto off1 = static_cast<char*>(h1->gpu_ptr) - static_cast<char*>(base);
    auto off2 = static_cast<char*>(h2->gpu_ptr) - static_cast<char*>(base);
    EXPECT_EQ(off1 % bpp, 0);
    EXPECT_EQ(off2 % bpp, 0);

    pages.free(*h1);
    pages.free(*h2);
}

// ═══════════════════════════════════════════════════════════════════════════
// DeepSeek-V4 (V4-3c): 3-bucket page pools (kMain=CSA, kHca, kSwa, kIndexerK)
// ═══════════════════════════════════════════════════════════════════════════

#include "model/quantization/gguf_kquant.h"

namespace {

// Small V4 config: 5 layers [0,4,128,4,128], tiny serving load so pool
// counts stay test-sized.  Mirrors the shipped Flash surface otherwise.
lc::Config v4_small_config() {
    auto j = nlohmann::json{
        {"model", {
            {"architecture",           "deepseek_v4"},
            {"weights_path",           "/data/models/deepseek-v4-flash.gguf"},
            {"weights_format",         "gguf"},
            {"num_hidden_layers",      5},
            {"hidden_size",            4096},
            {"num_attention_heads",    64},
            {"num_key_value_heads",    1},
            {"head_dim",               512},
            {"qk_rope_head_dim",       64},
            {"q_lora_rank",            1024},
            {"compress_ratios",        {0, 4, 128, 4, 128}},
            {"compress_rope_theta",    160000.0},
            {"sliding_window",         128},
            {"intermediate_size",      2048},
            {"n_routed_experts",       256},
            {"n_shared_experts",       1},
            {"num_experts_per_tok",    6},
            {"n_group",                1},
            {"topk_group",             1},
            {"vocab_size",             129280},
            {"max_position_embeddings", 1048576},
            {"rope_theta",             10000.0},
            {"rms_norm_eps",           1e-6},
            {"num_nextn_predict_layers", 0},
            {"first_k_dense_replace",  0},
            {"routed_scaling_factor",  1.5},
            {"moe_intermediate_size",  2048},
            {"moe_layer_freq",         1},
            {"gating_score_fn",        "sqrtsoftplus"},
            {"swiglu_limit",           10.0},
            {"num_hash_layers",        3},
            {"o_groups",               8},
            {"o_lora_rank",            1024},
            {"hc_mult",                4},
            {"index_topk",             512},
            {"index_n_heads",          64},
            {"index_head_dim",         128},
        }},
        {"quantization", {{"weights", "gguf"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp16"}}},
        {"compute", {{"attention_backend", "csa_hca"}}},
        {"parallelism", {{"tensor_parallelism", 1}}},
        {"memory", {{"kv_cache", {
            {"page_size_tokens", 16},
            {"speculation_pool_fraction", 0.15},
            {"indexer_k_page_size_tokens", 8192},
        }}}},
        {"serving", {{"max_concurrent_requests", 2},
                     {"max_sequence_length", 8192}}},
        {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                      {"tp_array", {0}},
                      {"system_ram_gb", 512}}},
    };
    return lc::parse_config(j);
}

TestAllocators make_v4_test_allocators() {
    auto cfg = v4_small_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::GgufQuantInterface mxfp4{lmod::GgufKQuantType::MXFP4};
    lmod::LayerRegistry reg(mcfg, cfg, mxfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);
    auto pages = lmem::PageAllocator(vram, nb.ptrs[0]);
    return TestAllocators{std::move(nb), std::move(vram), std::move(pages)};
}

}  // namespace

TEST(PageAllocatorV4, PoolCountsMatchLayout) {
    auto [nb_, vram, pages] = make_v4_test_allocators();
    const auto& g = vram.layout().gpus[0];

    // 2 req × ceil(8192/256)=32 blocks × {2 CSA, 2 HCA} layers.  Side tiers
    // additionally carry the prefix-holder budget (serving.prefix_cache
    // default max_entries=8): a V4 holder owns a COMPLETE copy-on-fork
    // kSwa/kHca/LID set (INV-PREFIX-CACHE-3, 53756713), while kMain (CSA)
    // is refcount-shared with holders and NOT holder-scaled.
    constexpr int kHolders = 8;                 // serving.prefix_cache default
    // S4 (TD-INDEXER-POOL-ELASTIC): the LID share is folded into the CSA
    // (kMain) span — 1 page/seq × 2 CSA layers × (2 req + 8 holders) = 20
    // slabs of LID share on top of the demand-driven 2 req × 32 blocks ×
    // 2 CSA layers = 128 pages, S1-quantized to whole slabs first.
    const int pps = pages.pages_per_slab();
    ASSERT_GT(pps, 0);
    const int csa_demand = (2 * 32 * 2 / pps) * pps;  // quantized down
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kMain),
              csa_demand + 20 * pps);
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kHca), (2 + kHolders) * 32 * 2);
    // SWA: per seq — 1 SWA layer×2 + 2 CSA×3 + 2 HCA×3 = 14 pages.
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kSwa), (2 + kHolders) * 14);
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kMain), g.kv_main_pages);
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kHca), g.kv_hca_pages);
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kSwa), g.kv_swa_pages);
    // Elastic indexer pool: NO dedicated span (layout carries 0 pages);
    // capacity = the shared region's slab count.
    EXPECT_EQ(g.indexer_k_pages, 0);
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kIndexerK),
              g.kv_main_pages / pps);
    EXPECT_GT(pages.total_pages(0, lmem::Pool::kIndexerK), 0);
    EXPECT_EQ(pages.kv_cache_format(), lmem::KvCacheFormat::kV4Fp8);
}

TEST(PageAllocatorV4, AllocFreeCycleAllPools) {
    auto [nb_, vram, pages] = make_v4_test_allocators();
    for (auto pool : {lmem::Pool::kMain, lmem::Pool::kSpeculation,
                      lmem::Pool::kHca, lmem::Pool::kSwa,
                      lmem::Pool::kIndexerK}) {
        const int before = pages.free_pages(0, pool);
        ASSERT_GT(before, 0) << static_cast<int>(pool);
        auto h = pages.allocate(0, pool);
        ASSERT_TRUE(h.has_value()) << static_cast<int>(pool);
        EXPECT_EQ(h->pool, pool);
        EXPECT_NE(h->gpu_ptr, nullptr);
        EXPECT_EQ(pages.free_pages(0, pool), before - 1);
        EXPECT_EQ(pages.used_pages(0, pool), 1);
        pages.free(*h);
        EXPECT_EQ(pages.free_pages(0, pool), before);
        EXPECT_EQ(pages.used_pages(0, pool), 0);
    }
}

TEST(PageAllocatorV4, TierPointerArithmeticAndRegions) {
    auto [nb_, vram, pages] = make_v4_test_allocators();
    const auto& reg0 = vram.region(0);
    const auto& v4 = vram.layout().v4;

    // Two HCA pages: consecutive indices stride by hca_bytes_per_page and
    // stay inside [kv_hca, kv_swa).
    auto h0 = pages.allocate(0, lmem::Pool::kHca);
    auto h1 = pages.allocate(0, lmem::Pool::kHca);
    ASSERT_TRUE(h0 && h1);
    auto diff = std::abs(static_cast<char*>(h1->gpu_ptr) -
                         static_cast<char*>(h0->gpu_ptr));
    EXPECT_EQ(diff, v4.hca_bytes_per_page);
    EXPECT_GE(h0->gpu_ptr, reg0.kv_hca);
    EXPECT_LT(h0->gpu_ptr, reg0.kv_swa);

    // SWA pages live in [kv_swa, indexer_k) — S1 moved the indexer span
    // between kv_swa and kv_main (shared slab region).
    auto s0 = pages.allocate(0, lmem::Pool::kSwa);
    auto s1 = pages.allocate(0, lmem::Pool::kSwa);
    ASSERT_TRUE(s0 && s1);
    EXPECT_EQ(std::abs(static_cast<char*>(s1->gpu_ptr) -
                       static_cast<char*>(s0->gpu_ptr)),
              v4.swa_bytes_per_page);
    EXPECT_GE(s0->gpu_ptr, reg0.kv_swa);
    EXPECT_LT(s0->gpu_ptr, reg0.indexer_k);

    // Main (CSA) pages in [kv_main, expert_streaming).
    auto m0 = pages.allocate(0, lmem::Pool::kMain);
    ASSERT_TRUE(m0);
    EXPECT_GE(m0->gpu_ptr, reg0.kv_main);
    EXPECT_LT(m0->gpu_ptr, reg0.expert_streaming);
}

TEST(PageAllocatorV4, PoolIsolationAndExhaustion) {
    auto [nb_, vram, pages] = make_v4_test_allocators();
    // Exhaust the SWA pool; other pools unaffected.
    const int swa_total = pages.total_pages(0, lmem::Pool::kSwa);
    std::vector<lmem::PageHandle> held;
    for (int i = 0; i < swa_total; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kSwa);
        ASSERT_TRUE(h.has_value()) << i;
        held.push_back(*h);
    }
    EXPECT_FALSE(pages.allocate(0, lmem::Pool::kSwa).has_value());
    EXPECT_TRUE(pages.allocate(0, lmem::Pool::kHca).has_value());
    EXPECT_TRUE(pages.allocate(0, lmem::Pool::kMain).has_value());
    for (auto& h : held) pages.free(h);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSwa), swa_total);
}

TEST(PageAllocatorV4, FreeSequenceCoversTierPools) {
    auto [nb_, vram, pages] = make_v4_test_allocators();
    const uint64_t seq = 77;
    for (auto pool : {lmem::Pool::kMain, lmem::Pool::kHca, lmem::Pool::kSwa,
                      lmem::Pool::kIndexerK}) {
        auto h = pages.allocate(0, pool);
        ASSERT_TRUE(h);
        pages.meta(*h).sequence_id = seq;
    }
    auto other = pages.allocate(0, lmem::Pool::kHca);
    ASSERT_TRUE(other);
    pages.meta(*other).sequence_id = 78;

    pages.free_sequence(0, seq);
    EXPECT_EQ(pages.used_pages(0, lmem::Pool::kMain), 0);
    EXPECT_EQ(pages.used_pages(0, lmem::Pool::kSwa), 0);
    EXPECT_EQ(pages.used_pages(0, lmem::Pool::kIndexerK), 0);
    EXPECT_EQ(pages.used_pages(0, lmem::Pool::kHca), 1);  // seq 78 survives
    pages.free(*other);
}

TEST(PageAllocatorV4, CowCopyOnTierPool) {
    auto [nb_, vram, pages] = make_v4_test_allocators();
    auto h = pages.allocate(0, lmem::Pool::kHca);
    ASSERT_TRUE(h);
    pages.meta(*h).sequence_id = 5;
    pages.meta(*h).token_start = 256;
    pages.meta(*h).token_end = 512;

    // refcount 1 → no-op.
    EXPECT_EQ(pages.cow_copy(*h).gpu_ptr, h->gpu_ptr);

    pages.add_ref(*h);
    auto split = pages.cow_copy(*h);
    EXPECT_NE(split.gpu_ptr, h->gpu_ptr);
    EXPECT_EQ(split.pool, lmem::Pool::kHca);
    EXPECT_EQ(pages.meta(split).sequence_id, 5u);
    EXPECT_EQ(pages.meta(split).token_start, 256u);
    EXPECT_EQ(pages.meta(split).token_end, 512u);
    EXPECT_EQ(pages.meta(*h).refcount, 1u);
    pages.free(split);
    pages.free(*h);
}

TEST(PageAllocatorV4, MetadataAccessTierPools) {
    auto [nb_, vram, pages] = make_v4_test_allocators();
    auto h = pages.allocate(0, lmem::Pool::kSwa);
    ASSERT_TRUE(h);
    pages.meta(*h).layer_index = 3;
    pages.meta(*h).sequence_id = 9;
    const auto& cpages = pages;
    EXPECT_EQ(cpages.meta(*h).layer_index, 3u);
    EXPECT_EQ(cpages.meta(*h).sequence_id, 9u);
    pages.free(*h);
}

TEST(PageAllocatorV4, NonV4ModelsHaveEmptyTierPools) {
    auto [nb_, vram, pages] = make_v32_test_allocators();
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kHca), 0);
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kSwa), 0);
    EXPECT_FALSE(pages.allocate(0, lmem::Pool::kHca).has_value());
    EXPECT_FALSE(pages.allocate(0, lmem::Pool::kSwa).has_value());
}

// ═══════════════════════════════════════════════════════════════════════════
// S2 position-major per-sequence bump runs (TD-INDEXER-POOL-ELASTIC +
// TD-SLAB-S2-GLM-MIN-FOOTPRINT, RADIX_SLAB_DESIGN §5 S2): the slabbed kMain
// span is bump-allocated per SEQUENCE, position-major — one slab holds all
// layers' pages for a contiguous token range of one sequence. Rule 3: a
// slab is claimed by exactly one sequence; freeing stays per-sequence and
// returns whole slabs; a cold token range drains whole slabs; the CoW
// split colocates with its source page's slab (fork-family tenancy).
// ═══════════════════════════════════════════════════════════════════════════

namespace {

/// v3.2 (DSA → slabbed) fixture with a SMALL kMain pool: explicit physical
/// page cap keeps the slab count test-sized (pps = 105 on the SnapMLA arm).
TestAllocators make_v32_small_kv_allocators(int max_pages = 1500) {
    auto cfg = v32_config();
    cfg.memory.kv_cache.max_pages_per_gpu = max_pages;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);
    auto pages = lmem::PageAllocator(vram, nb.ptrs[0]);
    return TestAllocators{std::move(nb), std::move(vram), std::move(pages)};
}

}  // namespace

TEST(PageAllocatorSlabRun, BumpIsContiguousWithinARun) {
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int pps = pages.pages_per_slab();
    ASSERT_GT(pps, 1);
    ASSERT_GE(pages.total_pages(0, lmem::Pool::kMain), 2 * pps);

    std::vector<lmem::PageHandle> hs;
    for (int i = 0; i <= pps; ++i) {
        auto h = pages.allocate_for_sequence(
            0, lmem::Pool::kMain, /*seq=*/1, /*layer=*/0,
            static_cast<uint32_t>(i * 16), static_cast<uint32_t>(i * 16 + 16));
        ASSERT_TRUE(h.has_value()) << i;
        EXPECT_EQ(pages.meta(*h).sequence_id, 1u);
        EXPECT_EQ(pages.meta(*h).layer_index, 0u);
        hs.push_back(*h);
    }
    // First pps pages: one slab, strictly ascending bump offsets.
    for (int i = 0; i < pps; ++i) {
        EXPECT_EQ(hs[i].page_idx, hs[0].page_idx + i) << i;
        EXPECT_EQ(hs[i].page_idx / pps, hs[0].page_idx / pps) << i;
    }
    // Page pps rolled into a NEW slab.
    EXPECT_NE(hs[pps].page_idx / pps, hs[0].page_idx / pps);

    const auto f = pages.kv_fragmentation(0);
    EXPECT_EQ(f.live_slabs, 2);
    EXPECT_EQ(f.used_pages, pps + 1);
    EXPECT_EQ(f.fragmented_free_pages, pps - 1);
    for (auto& h : hs) pages.free(h);
}

TEST(PageAllocatorSlabRun, SequencesNeverShareASlabButLayersDo) {
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int pps = pages.pages_per_slab();

    // Position-major rule 3: DIFFERENT LAYERS of one sequence share a slab
    // (that is the whole point — the min footprint is ~1 slab, not
    // kv_layers slabs); different SEQUENCES never do.
    std::vector<lmem::PageHandle> a, b, c;
    for (int i = 0; i < 3; ++i) {
        a.push_back(*pages.allocate_for_sequence(0, lmem::Pool::kMain, 1, 0,
                                                 0, 16));
        b.push_back(*pages.allocate_for_sequence(0, lmem::Pool::kMain, 1, 1,
                                                 0, 16));
        c.push_back(*pages.allocate_for_sequence(0, lmem::Pool::kMain, 2, 0,
                                                 0, 16));
    }
    const int slab_a = a[0].page_idx / pps;
    const int slab_b = b[0].page_idx / pps;
    const int slab_c = c[0].page_idx / pps;
    EXPECT_EQ(slab_a, slab_b) << "layers of one sequence share a slab";
    EXPECT_NE(slab_a, slab_c) << "sequences never share a slab";
    for (auto& h : a) EXPECT_EQ(h.page_idx / pps, slab_a);
    for (auto& h : b) EXPECT_EQ(h.page_idx / pps, slab_a);
    for (auto& h : c) EXPECT_EQ(h.page_idx / pps, slab_c);
    EXPECT_EQ(pages.kv_fragmentation(0).live_slabs, 2);
    for (auto* v : {&a, &b, &c})
        for (auto& h : *v) pages.free(h);
    EXPECT_EQ(pages.kv_fragmentation(0).live_slabs, 0);
}

TEST(PageAllocatorSlabRun, MinFootprintIsOneSlabAcrossAllLayers) {
    // THE TD-SLAB-S2-GLM-MIN-FOOTPRINT acceptance shape: a 1-token sequence
    // allocates one page per kMain layer (GLM champion: 79) and must fit in
    // ceil(kv_layers / pps) slabs — ~1 — not kv_layers slabs as the
    // reverted layer-major packing required.
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int pps = pages.pages_per_slab();
    const int kv_layers = std::min(pps, 79);  // GLM champion layer count

    std::vector<lmem::PageHandle> hs;
    for (int l = 0; l < kv_layers; ++l) {
        auto h = pages.allocate_for_sequence(
            0, lmem::Pool::kMain, /*seq=*/1, static_cast<uint32_t>(l),
            /*token_start=*/0, /*token_end=*/16);
        ASSERT_TRUE(h.has_value()) << l;
        hs.push_back(*h);
    }
    const auto f = pages.kv_fragmentation(0);
    EXPECT_EQ(f.live_slabs, 1) << "1-token sequence must occupy ONE slab";
    EXPECT_EQ(f.used_pages, kv_layers);
    for (auto& h : hs) pages.free(h);
    EXPECT_EQ(pages.kv_fragmentation(0).live_slabs, 0);
}

TEST(PageAllocatorSlabRun, ColdRangeDemotionDrainsWholeSlabs) {
    // Position-major cohort property: KVT demotes by POSITION across all
    // layers ("fully behind the retention window"), so freeing every
    // layer's pages of an old token range must drain COMPLETE slabs — the
    // packing the layer-major shape could never achieve (it spread a token
    // range across a slice of every layer's slab).
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int pps = pages.pages_per_slab();
    const int L = 4;
    const int blocks = (2 * pps) / L;  // ~2 slabs of position-major pages

    std::vector<lmem::PageHandle> hs;
    for (int b = 0; b < blocks; ++b)          // position-outer …
        for (int l = 0; l < L; ++l)           // … layer-inner (append order)
            hs.push_back(*pages.allocate_for_sequence(
                0, lmem::Pool::kMain, 1, static_cast<uint32_t>(l),
                static_cast<uint32_t>(b * 16),
                static_cast<uint32_t>(b * 16 + 16)));
    const auto before = pages.kv_fragmentation(0);
    ASSERT_GE(before.live_slabs, 2);

    // Demote the oldest token range: all L layers of the first pps pages'
    // worth of blocks — exactly the first slab's occupancy by construction.
    for (int i = 0; i < pps; ++i) pages.free(hs[i]);
    const auto after = pages.kv_fragmentation(0);
    EXPECT_EQ(after.live_slabs, before.live_slabs - 1)
        << "a cold token range must return its slab(s) WHOLE";
    EXPECT_EQ(after.free_slabs, before.free_slabs + 1);
    EXPECT_EQ(after.fragmented_free_pages, before.fragmented_free_pages)
        << "no stranded free pages inside surviving slabs";
    for (size_t i = pps; i < hs.size(); ++i) pages.free(hs[i]);
    EXPECT_EQ(pages.kv_fragmentation(0).live_slabs, 0);
}

TEST(PageAllocatorSlabRun, FreeSequenceReturnsWholeSlabs) {
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int pps = pages.pages_per_slab();
    const int total = pages.total_pages(0, lmem::Pool::kMain);
    const int n = 2 * pps + pps / 2;  // 2.5 slabs
    ASSERT_GE(total, 3 * pps);

    for (int i = 0; i < n; ++i) {
        auto h = pages.allocate_for_sequence(0, lmem::Pool::kMain, 7, 0,
                                             0, 16);
        ASSERT_TRUE(h.has_value()) << i;
    }
    EXPECT_EQ(pages.kv_fragmentation(0).live_slabs, 3);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), total - n);

    // Per-sequence bulk free returns the slabs WHOLE (rule 3 consequence).
    pages.free_sequence(0, 7);
    const auto f = pages.kv_fragmentation(0);
    EXPECT_EQ(f.live_slabs, 0);
    EXPECT_EQ(f.free_slabs, f.total_slabs);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), total);
}

TEST(PageAllocatorSlabRun, HoleReuseStaysInRun) {
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int pps = pages.pages_per_slab();

    std::vector<lmem::PageHandle> hs;
    for (int i = 0; i < 5; ++i)
        hs.push_back(*pages.allocate_for_sequence(
            0, lmem::Pool::kMain, 1, 0, static_cast<uint32_t>(i * 16),
            static_cast<uint32_t>(i * 16 + 16)));
    const int freed_idx = hs[2].page_idx;
    pages.free(hs[2]);  // rewind-style free at position 32

    // Another sequence must NOT take the hole (the slab belongs to seq 1) …
    auto other = pages.allocate_for_sequence(0, lmem::Pool::kMain, 2, 0,
                                             0, 16);
    ASSERT_TRUE(other.has_value());
    EXPECT_NE(other->page_idx, freed_idx);
    EXPECT_NE(other->page_idx / pps, freed_idx / pps);

    // … while the owning sequence re-allocating that POSITION (rewind
    // refill / re-promotion) position-matches back into the hole.
    auto again = pages.allocate_for_sequence(0, lmem::Pool::kMain, 1, 0,
                                             32, 48);
    ASSERT_TRUE(again.has_value());
    EXPECT_EQ(again->page_idx, freed_idx);

    pages.free(*other);
    pages.free(*again);
    for (int i : {0, 1, 3, 4}) pages.free(hs[i]);
    EXPECT_EQ(pages.kv_fragmentation(0).live_slabs, 0);
}

TEST(PageAllocatorSlabRun, RepromotionReturnsToItsCohortSlab) {
    // A partially drained OLD slab (some of its token range demoted) must
    // receive a re-promoted page of that range back — position-matched —
    // instead of the frontier slab, so the cohort stays coherent.
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int pps = pages.pages_per_slab();

    std::vector<lmem::PageHandle> hs;
    for (int i = 0; i < pps + pps / 2; ++i)  // slab A full + slab B half
        hs.push_back(*pages.allocate_for_sequence(
            0, lmem::Pool::kMain, 1, 0, static_cast<uint32_t>(i * 16),
            static_cast<uint32_t>(i * 16 + 16)));
    const int slab_a = hs[0].page_idx / pps;
    const int slab_b = hs[pps].page_idx / pps;
    ASSERT_NE(slab_a, slab_b);

    // Demote part of slab A's range (not all — the slab stays claimed).
    for (int i = 3; i < 6; ++i) pages.free(hs[i]);

    // Re-promote position 4*16: must land back in slab A, not slab B.
    auto rp = pages.allocate_for_sequence(0, lmem::Pool::kMain, 1, 0,
                                          4 * 16, 4 * 16 + 16);
    ASSERT_TRUE(rp.has_value());
    EXPECT_EQ(rp->page_idx / pps, slab_a);

    // A frontier APPEND (new position beyond every watermark) must NOT
    // fill slab A's remaining holes — it stays in the frontier slab.
    auto ap = pages.allocate_for_sequence(
        0, lmem::Pool::kMain, 1, 0,
        static_cast<uint32_t>((pps + pps / 2) * 16),
        static_cast<uint32_t>((pps + pps / 2) * 16 + 16));
    ASSERT_TRUE(ap.has_value());
    EXPECT_EQ(ap->page_idx / pps, slab_b);

    pages.free(*rp);
    pages.free(*ap);
    for (int i = 0; i < 3; ++i) pages.free(hs[i]);
    for (size_t i = 6; i < hs.size(); ++i) pages.free(hs[i]);
    EXPECT_EQ(pages.kv_fragmentation(0).live_slabs, 0);
}

TEST(PageAllocatorSlabRun, PressureFallbackUsesOwnOlderSlabSpace) {
    // With NO free slabs and NO loose pages left, an append must fall back
    // to free space in the sequence's own older slabs (never fail while
    // the sequence owns free space; never touch another sequence's slab).
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int pps = pages.pages_per_slab();
    const int total = pages.total_pages(0, lmem::Pool::kMain);

    std::vector<lmem::PageHandle> hs;
    for (int i = 0; i < total; ++i)
        hs.push_back(*pages.allocate_for_sequence(
            0, lmem::Pool::kMain, 1, 0, static_cast<uint32_t>(i * 16),
            static_cast<uint32_t>(i * 16 + 16)));
    ASSERT_EQ(pages.kv_fragmentation(0).free_slabs, 0);

    // Punch a hole in the OLDEST slab (position 0), then append a NEW
    // frontier position: no fresh slab and no position match exist, so the
    // pressure fallback takes the old hole.
    const int old_idx = hs[0].page_idx;
    pages.free(hs[0]);
    auto h = pages.allocate_for_sequence(
        0, lmem::Pool::kMain, 1, 0, static_cast<uint32_t>(total * 16),
        static_cast<uint32_t>(total * 16 + 16));
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(h->page_idx, old_idx);

    // Another sequence still may not take seq 1's in-slab space.
    EXPECT_FALSE(pages.allocate_for_sequence(0, lmem::Pool::kMain, 2, 0,
                                             0, 16)
                     .has_value());

    pages.free(*h);
    for (size_t i = 1; i < hs.size(); ++i) pages.free(hs[i]);
    EXPECT_EQ(pages.kv_fragmentation(0).live_slabs, 0);
}

TEST(PageAllocatorSlabRun, CowSplitColocatesWithSourceSlab) {
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int pps = pages.pages_per_slab();

    auto h0 = pages.allocate_for_sequence(0, lmem::Pool::kMain, 1, 3, 0, 16);
    auto h1 = pages.allocate_for_sequence(0, lmem::Pool::kMain, 1, 3, 16, 32);
    ASSERT_TRUE(h0 && h1);
    pages.add_ref(*h1);  // shared with the (future) fork child

    // The split lands NEXT TO its source page — no fresh slab per fork
    // child (this is what keeps GLM prefix holders refcount-cheap,
    // INV-PREFIX-CACHE-3).
    auto split = pages.cow_copy(*h1, /*dst_seq_id=*/2);
    EXPECT_NE(split.page_idx, h1->page_idx);
    EXPECT_EQ(split.page_idx / pps, h1->page_idx / pps);
    EXPECT_EQ(pages.meta(split).refcount, 1u);
    EXPECT_EQ(pages.meta(*h1).refcount, 1u);
    EXPECT_EQ(pages.kv_fragmentation(0).live_slabs, 1);

    pages.free(split);
    pages.free(*h1);
    pages.free(*h0);
}

TEST(PageAllocatorSlabRun, CowSplitFallsBackToDstRunSlab) {
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int pps = pages.pages_per_slab();

    // Fill the source slab completely so colocation cannot serve the split.
    std::vector<lmem::PageHandle> hs;
    for (int i = 0; i < pps; ++i)
        hs.push_back(*pages.allocate_for_sequence(0, lmem::Pool::kMain, 1, 0,
                                                  0, 16));
    pages.add_ref(hs.back());
    auto split = pages.cow_copy(hs.back(), /*dst_seq_id=*/2);
    EXPECT_NE(split.page_idx / pps, hs.back().page_idx / pps);

    // The fallback slab belongs to the CHILD's run: the child's next append
    // bumps inside it.
    auto next = pages.allocate_for_sequence(0, lmem::Pool::kMain, 2, 0,
                                            0, 16);
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next->page_idx / pps, split.page_idx / pps);

    pages.free(*next);
    pages.free(split);
    for (auto& h : hs) pages.free(h);
}

TEST(PageAllocatorSlabRun, PromotedSpecPageBecomesLooseAndRecycles) {
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int total = pages.total_pages(0, lmem::Pool::kMain);
    const int spec_before = pages.free_pages(0, lmem::Pool::kSpeculation);
    ASSERT_GT(spec_before, 0);

    // INV-4.9b: promote a spec page, free it — capacity moves to kMain as a
    // LOOSE page (physically outside the slab span).
    auto sp = pages.allocate(0, lmem::Pool::kSpeculation);
    ASSERT_TRUE(sp.has_value());
    const int loose_idx = sp->page_idx;
    pages.promote(*sp);
    pages.free(*sp);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), total + 1);
    EXPECT_EQ(pages.kv_fragmentation(0).loose_free_pages, 1);

    // Exhaust every slab page; the loose page is the LAST resort and keeps
    // the recycled capacity allocatable.
    std::vector<lmem::PageHandle> hs;
    for (int i = 0; i < total; ++i) {
        auto h = pages.allocate_for_sequence(0, lmem::Pool::kMain, 1, 0,
                                             0, 16);
        ASSERT_TRUE(h.has_value()) << i;
        EXPECT_LT(h->page_idx, total) << "slab pages first";
        hs.push_back(*h);
    }
    auto last = pages.allocate_for_sequence(0, lmem::Pool::kMain, 1, 0,
                                            0, 16);
    ASSERT_TRUE(last.has_value());
    EXPECT_EQ(last->page_idx, loose_idx);
    EXPECT_FALSE(pages.allocate_for_sequence(0, lmem::Pool::kMain, 1, 0,
                                             0, 16)
                     .has_value());
    pages.free(*last);
    for (auto& h : hs) pages.free(h);
}

TEST(PageAllocatorSlabRun, HeadroomReservationHolds) {
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int total = pages.total_pages(0, lmem::Pool::kMain);

    lmem::PageAllocator::HeadroomConfig hc;
    hc.max_concurrent_forks = 3;
    hc.max_concurrent_sequences = 2;
    hc.page_growth_chunk_pages = 5;
    pages.configure_headroom(hc);
    const int reserved = pages.reserved_pages(0, lmem::Pool::kMain);
    ASSERT_EQ(reserved, 13);

    // INV-4.9f: reserved claims refuse once free would dip to the headroom.
    std::vector<lmem::PageHandle> hs;
    while (true) {
        auto h = pages.allocate_for_sequence(0, lmem::Pool::kMain, 1, 0,
                                             0, 16);
        if (!h) break;
        hs.push_back(*h);
    }
    EXPECT_EQ(static_cast<int>(hs.size()), total - reserved);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), reserved);
    // Unreserved (CoW / page growth) still allocates from the headroom.
    auto h = pages.allocate_for_sequence(0, lmem::Pool::kMain, 1, 0, 0, 16,
                                         /*unreserved=*/true);
    EXPECT_TRUE(h.has_value());
    if (h) pages.free(*h);
    for (auto& hh : hs) pages.free(hh);
}

TEST(PageAllocatorSlabRun, AnonymousAllocateExhaustsExactly) {
    auto [nb_, vram, pages] = make_v32_small_kv_allocators();
    const int total = pages.total_pages(0, lmem::Pool::kMain);

    // allocate() with no sequence identity (trash page / legacy callers)
    // shares the anonymous run — capacity is exactly the pool size.
    std::vector<lmem::PageHandle> hs;
    for (int i = 0; i < total; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kMain);
        ASSERT_TRUE(h.has_value()) << i;
        hs.push_back(*h);
    }
    EXPECT_FALSE(pages.allocate(0, lmem::Pool::kMain).has_value());
    for (auto& h : hs) pages.free(h);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), total);
    EXPECT_EQ(pages.kv_fragmentation(0).live_slabs, 0);
}
