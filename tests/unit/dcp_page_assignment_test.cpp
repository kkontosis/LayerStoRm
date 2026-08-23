#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "config/config_parser.h"
#include "core/device_backend.h"
#include "core/memory/page_allocator.h"
#include "core/memory/vram_allocator.h"
#include "core/null_device_backend.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/fp8.h"
#include "model/quantization/nvfp4.h"

namespace lc = layerstorm::config;
namespace lcomp = layerstorm::compute;
namespace lmem = layerstorm::memory;
namespace lmod = layerstorm::model;

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

lc::Config dcp_v32_config() {
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
            {"topk_group",             4},
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
        {"parallelism", {{"tensor_parallelism", 2}}},
    };
    return lc::parse_config(j);
}

struct TestAllocators {
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned_backends;
    lmem::VramAllocator vram;
    lmem::PageAllocator pages;
};

TestAllocators make_dcp_allocators(const lc::Config& cfg) {
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);

    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned;
    std::vector<lcomp::DeviceBackend*> ptrs;
    for (size_t i = 0; i < layout.gpus.size(); ++i) {
        lc::GpuRef ref{.position = static_cast<int>(i),
                       .id = layout.gpus[i].gpu_id};
        owned.push_back(lcomp::make_null_device_backend(ref));
        ptrs.push_back(owned.back().get());
    }

    auto vram = lmem::VramAllocator(std::move(layout), ptrs);
    auto pages = lmem::PageAllocator(vram, ptrs[0]);
    return TestAllocators{std::move(owned), std::move(vram), std::move(pages)};
}

lmem::DcpConfig make_dcp_config(int dcp_size = 2, int chunk_size = 16,
                                 int page_size = 16,
                                 std::vector<int> tp_gpus = {0, 1},
                                 bool indexer_k_sharded = false,
                                 bool kv_sharded = false,
                                 int indexer_k_page_size_tokens = 0) {
    return lmem::DcpConfig{
        .dcp_size = dcp_size,
        .dcp_chunk_size = chunk_size,
        .page_size_tokens = page_size,
        .tp_gpu_indices = std::move(tp_gpus),
        .indexer_k_sharded = indexer_k_sharded,
        .indexer_k_page_size_tokens = indexer_k_page_size_tokens,
        .kv_sharded = kv_sharded,
    };
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Test 1: Round-robin chunk boundary
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, RankForToken_ChunkBoundary) {
    auto dcp = make_dcp_config(2, 16);

    // Tokens 0-15 → rank 0 (chunk 0)
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(0, dcp), 0);
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(15, dcp), 0);

    // Tokens 16-31 → rank 1 (chunk 1)
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(16, dcp), 1);
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(31, dcp), 1);

    // Tokens 32-47 → rank 0 (chunk 2, wraps)
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(32, dcp), 0);
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(47, dcp), 0);

    // Tokens 48-63 → rank 1 (chunk 3)
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(48, dcp), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 2: DCP disabled
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, RankForToken_Disabled) {
    auto dcp = make_dcp_config(1, 16, 16, {0});

    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(0, dcp), 0);
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(100, dcp), 0);
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(999, dcp), 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 3: Custom chunk size
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, RankForToken_CustomChunk) {
    auto dcp = make_dcp_config(2, 8, 8);

    // chunk_size=8: tokens 0-7 → rank 0, 8-15 → rank 1, 16-23 → rank 0
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(0, dcp), 0);
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(7, dcp), 0);
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(8, dcp), 1);
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(15, dcp), 1);
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(16, dcp), 0);
    EXPECT_EQ(lmem::PageAllocator::dcp_rank_for_token(23, dcp), 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 4: GPU mapping with different tp_gpu_indices
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, GpuForToken_MapsRankToGpuIdx) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);

    // Normal order: {0, 1}
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1}));
    EXPECT_EQ(pages.dcp_gpu_for_token(0), 0);
    EXPECT_EQ(pages.dcp_gpu_for_token(16), 1);
    EXPECT_EQ(pages.dcp_gpu_for_token(32), 0);

    // Reversed: {1, 0}
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {1, 0}));
    EXPECT_EQ(pages.dcp_gpu_for_token(0), 1);   // rank 0 → gpu 1
    EXPECT_EQ(pages.dcp_gpu_for_token(16), 0);   // rank 1 → gpu 0
    EXPECT_EQ(pages.dcp_gpu_for_token(32), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 5: assign_dcp_range sets metadata correctly
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, AssignRange_SetsMetadata) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1}));

    // Allocate 3 pages on GPU 0
    std::vector<lmem::PageHandle> handles;
    for (int i = 0; i < 3; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kMain);
        ASSERT_TRUE(h.has_value());
        handles.push_back(*h);
    }

    // Assign range [0, 48) across 3 pages (16 tokens each)
    pages.assign_dcp_range(0, 42, handles, 0, 48);

    EXPECT_EQ(pages.meta(handles[0]).token_start, 0u);
    EXPECT_EQ(pages.meta(handles[0]).token_end, 16u);
    EXPECT_EQ(pages.meta(handles[0]).sequence_id, 42u);

    EXPECT_EQ(pages.meta(handles[1]).token_start, 16u);
    EXPECT_EQ(pages.meta(handles[1]).token_end, 32u);

    EXPECT_EQ(pages.meta(handles[2]).token_start, 32u);
    EXPECT_EQ(pages.meta(handles[2]).token_end, 48u);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 6: Partial last page
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, AssignRange_PartialLastPage) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1}));

    std::vector<lmem::PageHandle> handles;
    for (int i = 0; i < 2; ++i) {
        auto h = pages.allocate(0, lmem::Pool::kMain);
        ASSERT_TRUE(h.has_value());
        handles.push_back(*h);
    }

    // Range [0, 20) — page 1 is partial (only 4 tokens)
    pages.assign_dcp_range(0, 1, handles, 0, 20);

    EXPECT_EQ(pages.meta(handles[0]).token_start, 0u);
    EXPECT_EQ(pages.meta(handles[0]).token_end, 16u);

    EXPECT_EQ(pages.meta(handles[1]).token_start, 16u);
    EXPECT_EQ(pages.meta(handles[1]).token_end, 20u);  // partial
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 7: allocate_for_dcp_append routes to correct GPU (SHARDED KV — owner
// routing is the dcp_kv_mode=sharded placement; replicated claims every GPU)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, AllocateAppend_RoutesToCorrectGpu) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1},
                                         /*indexer_k_sharded=*/false,
                                         /*kv_sharded=*/true));

    // token 0 → gpu 0
    auto h0 = pages.allocate_for_dcp_append(100, 0, 5);
    ASSERT_TRUE(h0.has_value());
    EXPECT_EQ(h0->gpu_idx, 0);
    EXPECT_EQ(pages.meta(*h0).sequence_id, 100u);
    EXPECT_EQ(pages.meta(*h0).token_start, 0u);
    EXPECT_EQ(pages.meta(*h0).token_end, 16u);
    EXPECT_EQ(pages.meta(*h0).layer_index, 5u);

    // token 16 → gpu 1
    auto h1 = pages.allocate_for_dcp_append(100, 16, 5);
    ASSERT_TRUE(h1.has_value());
    EXPECT_EQ(h1->gpu_idx, 1);
    EXPECT_EQ(pages.meta(*h1).token_start, 16u);

    // token 32 → gpu 0 (wraps)
    auto h2 = pages.allocate_for_dcp_append(100, 32, 5);
    ASSERT_TRUE(h2.has_value());
    EXPECT_EQ(h2->gpu_idx, 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 8: Non-DCP fallback
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, AllocateAppend_NonDcpFallback) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    // DCP disabled
    pages.set_dcp_config(make_dcp_config(1, 16, 16, {0}));

    // All tokens go to default_gpu (0)
    auto h0 = pages.allocate_for_dcp_append(1, 0, 0, 0);
    ASSERT_TRUE(h0.has_value());
    EXPECT_EQ(h0->gpu_idx, 0);

    auto h1 = pages.allocate_for_dcp_append(1, 16, 0, 0);
    ASSERT_TRUE(h1.has_value());
    EXPECT_EQ(h1->gpu_idx, 0);

    auto h2 = pages.allocate_for_dcp_append(1, 32, 0, 0);
    ASSERT_TRUE(h2.has_value());
    EXPECT_EQ(h2->gpu_idx, 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 9: Disjoint ranges, no cross-GPU overlap (SHARDED KV)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, DisjointRanges_NoCrossGpuOverlap) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1},
                                         /*indexer_k_sharded=*/false,
                                         /*kv_sharded=*/true));

    // Allocate 8 pages covering 128 tokens (8 chunks of 16)
    std::set<uint32_t> gpu0_tokens, gpu1_tokens;

    for (uint32_t pos = 0; pos < 128; pos += 16) {
        auto h = pages.allocate_for_dcp_append(1, pos, 0);
        ASSERT_TRUE(h.has_value());
        auto& m = pages.meta(*h);
        for (uint32_t t = m.token_start; t < m.token_end; ++t) {
            if (h->gpu_idx == 0) {
                EXPECT_EQ(gpu1_tokens.count(t), 0u) << "Overlap at token " << t;
                gpu0_tokens.insert(t);
            } else {
                EXPECT_EQ(gpu0_tokens.count(t), 0u) << "Overlap at token " << t;
                gpu1_tokens.insert(t);
            }
        }
    }

    // Union should cover [0, 128)
    EXPECT_EQ(gpu0_tokens.size() + gpu1_tokens.size(), 128u);
    // Each GPU should have 64 tokens
    EXPECT_EQ(gpu0_tokens.size(), 64u);
    EXPECT_EQ(gpu1_tokens.size(), 64u);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 9b: REPLICATED KV (INV-KV-REP, TD-KV-REPLICATED-PAGE-ALIAS fix) —
// every allocation claims the SAME canonical index on EVERY TP GPU in
// lockstep, so replicated block tables never alias and each rank holds the
// full KV.
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, ReplicatedKv_LockstepUniqueIndices) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1}));  // kv_sharded=false

    const int free0 = pages.free_pages(0, lmem::Pool::kMain);
    const int free1 = pages.free_pages(1, lmem::Pool::kMain);

    // Alternating token positions (the aliasing trigger pre-fix): every
    // handle must carry a DISTINCT canonical index, and both GPUs' free
    // lists must drain in lockstep.
    std::vector<lmem::PageHandle> handles;
    std::set<int> indices;
    for (uint32_t pos = 0; pos < 96; pos += 16) {
        auto h = pages.allocate_for_dcp_append(7, pos, 0);
        ASSERT_TRUE(h.has_value()) << "pos " << pos;
        EXPECT_EQ(h->gpu_idx, 0) << "canonical handle must be rank-0 GPU";
        EXPECT_TRUE(indices.insert(h->page_idx).second)
            << "physical page " << h->page_idx << " aliased at pos " << pos;
        EXPECT_TRUE(pages.meta(*h).replicated);
        handles.push_back(*h);
    }
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), free0 - 6);
    EXPECT_EQ(pages.free_pages(1, lmem::Pool::kMain), free1 - 6);

    // Mirrored free: both GPUs' pools drain back completely.
    for (auto& h : handles) pages.free(h);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), free0);
    EXPECT_EQ(pages.free_pages(1, lmem::Pool::kMain), free1);
}

TEST(DcpPageAssignment, ReplicatedKv_CowCopyMirrors) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1}));

    auto h = pages.allocate_for_dcp_append(9, 0, 0);
    ASSERT_TRUE(h.has_value());
    const int free0 = pages.free_pages(0, lmem::Pool::kMain);
    const int free1 = pages.free_pages(1, lmem::Pool::kMain);

    // Shared page (refcount 2 on every replica): cow_copy must allocate a
    // NEW replicated index on both GPUs and keep the old one alive.
    pages.add_ref(*h);
    auto split = pages.cow_copy(*h);
    EXPECT_NE(split.page_idx, h->page_idx);
    EXPECT_EQ(split.gpu_idx, 0);
    EXPECT_TRUE(pages.meta(split).replicated);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), free0 - 1);
    EXPECT_EQ(pages.free_pages(1, lmem::Pool::kMain), free1 - 1);

    pages.free(split);
    pages.free(*h);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), free0 + 1);
    EXPECT_EQ(pages.free_pages(1, lmem::Pool::kMain), free1 + 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 9c: REPLICATED KV × kSpeculation (INV-KV-REP extended,
// TD-KV-REPLICATED-SPEC fix) — draft-pool pages follow the SAME lockstep
// same-index-on-every-TP-GPU discipline as kMain, so replicated block
// tables of draft sequences never alias; promote/cow_copy/free all mirror.
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, ReplicatedKv_SpecPool_LockstepUniqueIndices) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1}));  // kv_sharded=false

    const int free0 = pages.free_pages(0, lmem::Pool::kSpeculation);
    const int free1 = pages.free_pages(1, lmem::Pool::kSpeculation);
    ASSERT_GT(free0, 6);
    const int main0 = pages.free_pages(0, lmem::Pool::kMain);
    const int main1 = pages.free_pages(1, lmem::Pool::kMain);

    // Every draft page must carry a DISTINCT canonical index claimed on BOTH
    // GPUs (the pre-fix single-GPU allocation left the mirror's index free
    // for another sequence → physical aliasing in the replicated tables).
    std::vector<lmem::PageHandle> handles;
    std::set<int> indices;
    for (uint32_t pos = 0; pos < 96; pos += 16) {
        auto h = pages.allocate_replicated(11, pos, 0,
                                           lmem::Pool::kSpeculation);
        ASSERT_TRUE(h.has_value()) << "pos " << pos;
        EXPECT_EQ(h->gpu_idx, 0) << "canonical handle must be rank-0 GPU";
        EXPECT_EQ(h->pool, lmem::Pool::kSpeculation);
        EXPECT_TRUE(indices.insert(h->page_idx).second)
            << "physical page " << h->page_idx << " aliased at pos " << pos;
        EXPECT_TRUE(pages.meta(*h).replicated);
        // Mirror's meta claimed in lockstep with identical fields.
        lmem::PageHandle mirror{1, h->page_idx, nullptr,
                                lmem::Pool::kSpeculation};
        EXPECT_TRUE(pages.meta(mirror).replicated);
        EXPECT_EQ(pages.meta(mirror).sequence_id, 11u);
        EXPECT_EQ(pages.meta(mirror).pool, lmem::Pool::kSpeculation);
        handles.push_back(*h);
    }
    // Both GPUs' SPEC pools drain in lockstep; main pools untouched.
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation), free0 - 6);
    EXPECT_EQ(pages.free_pages(1, lmem::Pool::kSpeculation), free1 - 6);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), main0);
    EXPECT_EQ(pages.free_pages(1, lmem::Pool::kMain), main1);

    // Mirrored free: both GPUs' spec pools drain back completely.
    for (auto& h : handles) pages.free(h);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation), free0);
    EXPECT_EQ(pages.free_pages(1, lmem::Pool::kSpeculation), free1);
}

TEST(DcpPageAssignment, ReplicatedKv_SpecPool_CowCopyMirrors) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1}));

    auto h = pages.allocate_replicated(13, 0, 0, lmem::Pool::kSpeculation);
    ASSERT_TRUE(h.has_value());
    const int free0 = pages.free_pages(0, lmem::Pool::kSpeculation);
    const int free1 = pages.free_pages(1, lmem::Pool::kSpeculation);

    // Shared draft page: cow_copy must allocate a NEW replicated index from
    // the SPECULATION pool on both GPUs (not from kMain).
    pages.add_ref(*h);
    auto split = pages.cow_copy(*h);
    EXPECT_NE(split.page_idx, h->page_idx);
    EXPECT_EQ(split.gpu_idx, 0);
    EXPECT_EQ(split.pool, lmem::Pool::kSpeculation);
    EXPECT_TRUE(pages.meta(split).replicated);
    EXPECT_EQ(pages.meta(split).pool, lmem::Pool::kSpeculation);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation), free0 - 1);
    EXPECT_EQ(pages.free_pages(1, lmem::Pool::kSpeculation), free1 - 1);

    pages.free(split);
    pages.free(*h);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation), free0 + 1);
    EXPECT_EQ(pages.free_pages(1, lmem::Pool::kSpeculation), free1 + 1);
}

TEST(DcpPageAssignment, ReplicatedKv_SpecPool_PromoteMirrors) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1}));

    const int main0 = pages.free_pages(0, lmem::Pool::kMain);
    const int main1 = pages.free_pages(1, lmem::Pool::kMain);
    const int spec0 = pages.free_pages(0, lmem::Pool::kSpeculation);
    const int spec1 = pages.free_pages(1, lmem::Pool::kSpeculation);

    auto h = pages.allocate_replicated(17, 0, 0, lmem::Pool::kSpeculation);
    ASSERT_TRUE(h.has_value());

    // Promotion (metadata-only, INV-4.9b) must flip the pool on EVERY TP
    // GPU's mirror — a canonical-only flip would route the later free to
    // main on rank 0 but back to spec on rank 1 (free-list desync).
    pages.promote(*h);
    EXPECT_EQ(pages.meta(*h).pool, lmem::Pool::kMain);
    lmem::PageHandle mirror{1, h->page_idx, nullptr, lmem::Pool::kMain};
    EXPECT_EQ(pages.meta(mirror).pool, lmem::Pool::kMain);
    EXPECT_TRUE(pages.meta(mirror).replicated);

    // Free lands in the MAIN free list on BOTH GPUs (zero-copy promotion
    // migrates the index; the spec pool regains nothing).
    pages.free(*h);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kMain), main0 + 1);
    EXPECT_EQ(pages.free_pages(1, lmem::Pool::kMain), main1 + 1);
    EXPECT_EQ(pages.free_pages(0, lmem::Pool::kSpeculation), spec0 - 1);
    EXPECT_EQ(pages.free_pages(1, lmem::Pool::kSpeculation), spec1 - 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 10: Indexer K replicated mode — allocates on all TP GPUs
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, IndexerK_Replicated_AllTpGpus) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1}, false));

    auto handles = pages.allocate_indexer_k_for_dcp(42, 0, 3);

    // Replicated: should get one handle per TP GPU
    ASSERT_EQ(handles.size(), 2u);
    EXPECT_EQ(handles[0].gpu_idx, 0);
    EXPECT_EQ(handles[1].gpu_idx, 1);
    EXPECT_EQ(handles[0].pool, lmem::Pool::kIndexerK);
    EXPECT_EQ(handles[1].pool, lmem::Pool::kIndexerK);
    EXPECT_EQ(pages.meta(handles[0]).sequence_id, 42u);
    EXPECT_EQ(pages.meta(handles[1]).sequence_id, 42u);
    EXPECT_EQ(pages.meta(handles[0]).layer_index, 3u);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 11: Indexer K local mode — allocates on owning GPU only
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, IndexerK_Local_OwningGpuOnly) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1}, true));

    // Token 0 → rank 0 → gpu 0
    auto h0 = pages.allocate_indexer_k_for_dcp(1, 0, 0);
    ASSERT_EQ(h0.size(), 1u);
    EXPECT_EQ(h0[0].gpu_idx, 0);

    // Token 16 → rank 1 → gpu 1
    auto h1 = pages.allocate_indexer_k_for_dcp(1, 16, 0);
    ASSERT_EQ(h1.size(), 1u);
    EXPECT_EQ(h1[0].gpu_idx, 1);

    // Token 32 → rank 0 → gpu 0 (wraps)
    auto h2 = pages.allocate_indexer_k_for_dcp(1, 32, 0);
    ASSERT_EQ(h2.size(), 1u);
    EXPECT_EQ(h2[0].gpu_idx, 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 11b: Indexer K local mode routes round-robin by INDEXER PAGE
// (TD-GLM-INDEXER-LOCAL-MERGE) — INDEPENDENT of the KV chunk partition.
// With the production shape (indexer page ≫ KV chunk) chunk routing would
// map every page-aligned start to rank 0; page routing keeps pages
// rank-atomic and alternating. Must match the executor/dispatcher owner
// math: owner(pos) = (pos / indexer_k_page_size_tokens) % dcp.
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, IndexerK_Local_PageRoundRobin) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    // KV chunk = 16, indexer page = 32: page routing must IGNORE the chunk.
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1}, true,
                                         /*kv_sharded=*/false,
                                         /*indexer_k_page_size_tokens=*/32));

    // Page 0 (tokens 0..31) → rank 0. Chunk routing would send token 16 to
    // rank 1 — page routing must keep the whole page on rank 0.
    auto h0 = pages.allocate_indexer_k_for_dcp(1, 0, 0);
    ASSERT_EQ(h0.size(), 1u);
    EXPECT_EQ(h0[0].gpu_idx, 0);
    auto h0b = pages.allocate_indexer_k_for_dcp(1, 16, 0);
    ASSERT_EQ(h0b.size(), 1u);
    EXPECT_EQ(h0b[0].gpu_idx, 0);

    // Page 1 (tokens 32..63) → rank 1. Chunk routing would send token 32 to
    // rank 0 ((32/16) % 2 == 0) — page routing must send it to rank 1.
    auto h1 = pages.allocate_indexer_k_for_dcp(1, 32, 0);
    ASSERT_EQ(h1.size(), 1u);
    EXPECT_EQ(h1[0].gpu_idx, 1);

    // Page 2 (tokens 64..95) → rank 0 (wraps).
    auto h2 = pages.allocate_indexer_k_for_dcp(1, 64, 0);
    ASSERT_EQ(h2.size(), 1u);
    EXPECT_EQ(h2[0].gpu_idx, 0);

    // Degenerate case the chunk fallback gets WRONG at the production shape
    // (page = 2 * chunk * dcp): page starts 0, 64, 128 are all chunk-rank 0;
    // page routing must alternate 0, 1, 0.
    pages.set_dcp_config(make_dcp_config(2, 16, 16, {0, 1}, true,
                                         /*kv_sharded=*/false,
                                         /*indexer_k_page_size_tokens=*/64));
    auto p0 = pages.allocate_indexer_k_for_dcp(2, 0, 0);
    auto p1 = pages.allocate_indexer_k_for_dcp(2, 64, 0);
    auto p2 = pages.allocate_indexer_k_for_dcp(2, 128, 0);
    ASSERT_EQ(p0.size(), 1u);
    ASSERT_EQ(p1.size(), 1u);
    ASSERT_EQ(p2.size(), 1u);
    EXPECT_EQ(p0[0].gpu_idx, 0);
    EXPECT_EQ(p1[0].gpu_idx, 1);
    EXPECT_EQ(p2[0].gpu_idx, 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 12: Non-DCP indexer K — single GPU
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DcpPageAssignment, IndexerK_NonDcp_SingleGpu) {
    auto cfg = dcp_v32_config();
    auto [backends_, vram, pages] = make_dcp_allocators(cfg);
    pages.set_dcp_config(make_dcp_config(1, 16, 16, {0}));

    auto handles = pages.allocate_indexer_k_for_dcp(1, 0, 0);
    ASSERT_EQ(handles.size(), 1u);
    EXPECT_EQ(handles[0].gpu_idx, 0);
    EXPECT_EQ(handles[0].pool, lmem::Pool::kIndexerK);
}
