#include <gtest/gtest.h>

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

TEST(PageAllocatorIndexerK, SeparateFromMainSpec) {
    auto [nb_, vram, pages] = make_v32_test_allocators();

    int main_free_before = pages.free_pages(0, lmem::Pool::kMain);
    int spec_free_before = pages.free_pages(0, lmem::Pool::kSpeculation);

    // Allocate from indexer K should not affect main/spec
    auto h = pages.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(h.has_value());

    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), main_free_before);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation), spec_free_before);

    pages.free(*h);
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

TEST(PageAllocatorIndexerK, PtrWithinIndexerKRegion) {
    auto [nb_, vram, pages] = make_v32_test_allocators();

    auto ik_base = reinterpret_cast<uintptr_t>(vram.region(0).indexer_k);
    int64_t ik_bytes = vram.layout().gpus[0].indexer_k_bytes;

    auto h = pages.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(h.has_value());
    auto addr = reinterpret_cast<uintptr_t>(h->gpu_ptr);
    EXPECT_GE(addr, ik_base);
    EXPECT_LT(addr, ik_base + ik_bytes);

    pages.free(*h);
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

    // 2 req × ceil(8192/256)=32 blocks × {2 CSA, 2 HCA} layers.
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kMain), 2 * 32 * 2);
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kHca), 2 * 32 * 2);
    // SWA: per seq — 1 SWA layer×2 + 2 CSA×3 + 2 HCA×3 = 14 pages.
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kSwa), 2 * 14);
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kMain), g.kv_main_pages);
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kHca), g.kv_hca_pages);
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kSwa), g.kv_swa_pages);
    EXPECT_EQ(pages.total_pages(0, lmem::Pool::kIndexerK), g.indexer_k_pages);
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

    // SWA pages live in [kv_swa, kv_main).
    auto s0 = pages.allocate(0, lmem::Pool::kSwa);
    auto s1 = pages.allocate(0, lmem::Pool::kSwa);
    ASSERT_TRUE(s0 && s1);
    EXPECT_EQ(std::abs(static_cast<char*>(s1->gpu_ptr) -
                       static_cast<char*>(s0->gpu_ptr)),
              v4.swa_bytes_per_page);
    EXPECT_GE(s0->gpu_ptr, reg0.kv_swa);
    EXPECT_LT(s0->gpu_ptr, reg0.kv_main);

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
