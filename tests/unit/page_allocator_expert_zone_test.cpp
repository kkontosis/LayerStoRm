// 44z: expert-zone grants — the FOURTH tenant class of the shared slab
// region (after per-sequence KV bump runs, elastic indexer-K single slabs,
// and mapped KDA state runs).
//
// A grant is a contiguous run of whole kMain slabs LENT to the expert cache.
// The asymmetry under test: an expert slot is a CACHE LINE over host RAM
// (reclaim = re-fetch, nothing lost), a KV page is STATE (no host copy) — so
// pressure flows expert -> KV and never the reverse. Concretely that means
// a grant must (a) be invisible to KV placement while held, (b) come back
// WHOLE and re-coalescing when released, and (c) never shrink the capacity
// the admissibility model quotes (total_pages is untouched by a grant).
//
// Placement is TOP-DOWN so long-lived grants pack against the top of the
// region, with the churning KDA state band beneath them and bottom-up
// KV/indexer claims below that — the TD-KDA-MAPPED-FRAG (b2) segregation
// family. Released runs FRONT-INSERT into the free-slab list so the next
// single-slab pop cannot colonize and split them.
//
// All CPU-only (NullDeviceBackend) — no _GPU_FILTER registration.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "config/config_parser.h"
#include "config/config_resolver.h"
#include "core/device_backend.h"
#include "core/gpu_ref.h"
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

namespace {

// ── Fixtures ────────────────────────────────────────────────────────────────
// Rebuilt locally (all unit *_test.cpp share ONE executable, so the sibling
// suites' helpers live in their own anonymous namespaces and are not
// reachable here).

/// v3.2 (DSA ⇒ slabbed kMain) with a SMALL physical page cap so the slab
/// count stays test-sized — the make_v32_small_kv_allocators shape.
lc::Config ez_v32_config() {
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
        {"quantization", {{"weights", "nvfp4"},
                          {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"},
                          {"gating_compute", "fp32"}}},
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

struct EzAllocators {
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned;
    std::vector<lcomp::DeviceBackend*> ptrs;
    std::unique_ptr<lmem::VramAllocator> vram;
    std::unique_ptr<lmem::PageAllocator> pages;
};

EzAllocators build(const lc::Config& cfg, const lmod::QuantInterface& quant) {
    EzAllocators a;
    lmod::ModelConfig mcfg(cfg);
    lmod::LayerRegistry reg(mcfg, cfg, quant);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    for (size_t i = 0; i < layout.gpus.size(); ++i) {
        lc::GpuRef gref{.position = static_cast<int>(i),
                        .id = layout.gpus[i].gpu_id};
        a.owned.push_back(lcomp::make_null_device_backend(gref));
        a.ptrs.push_back(a.owned.back().get());
    }
    a.vram = std::make_unique<lmem::VramAllocator>(std::move(layout), a.ptrs);
    a.pages = std::make_unique<lmem::PageAllocator>(*a.vram, a.ptrs[0]);
    return a;
}

EzAllocators make_slabbed(int max_pages = 1500) {
    auto cfg = ez_v32_config();
    cfg.memory.kv_cache.max_pages_per_gpu = max_pages;
    lmod::Nvfp4 nvfp4;
    return build(cfg, nvfp4);
}

/// Tiny glm5_next shape WITH an indexer pool (so slab geometry engages) and
/// the mapped KDA state switched on — the make_mapped_kda_allocators shape.
lc::Config ez_glm5n_mapped_config() {
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
        {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"},
                                 {"vram_gb", 1}}}},
                      {"system_ram_gb", 64}}},
        {"serving", {{"max_concurrent_requests", 2},
                     {"max_sequence_length", 2048}}},
        {"memory", {{"vram_safety_margin_gb", 0.1},
                    {"kv_cache", {{"page_growth_chunk_tokens", 0}}}}},
    };
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
    // The indexer pool is what makes the region slabbed (mapped unit = slab).
    cfg.model.index_topk = 64;
    cfg.model.index_n_heads = 2;
    cfg.model.index_head_dim = 32;
    cfg._internal_kda_state.mapped = true;
    return cfg;
}

EzAllocators make_mapped_kda() {
    unsetenv("LS_KDA_STATE_MAPPED");
    lmod::Fp8E4M3 fp8;
    return build(ez_glm5n_mapped_config(), fp8);
}

/// Flat page indices covered by a grant: [start*pps, (start+n)*pps).
bool in_grant(int page_idx, int start_slab, int num_slabs, int pps) {
    return page_idx >= start_slab * pps
        && page_idx < (start_slab + num_slabs) * pps;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// PageAllocatorExpertZone
// ═══════════════════════════════════════════════════════════════════════════

TEST(PageAllocatorExpertZone, ClaimStampsTopmostRun) {
    auto a = make_slabbed();
    auto& pa = *a.pages;
    const int pps = pa.pages_per_slab();
    ASSERT_GT(pps, 0);
    const auto f0 = pa.kv_fragmentation(0);
    const int total_slabs = f0.total_slabs;
    ASSERT_GE(total_slabs, 6);
    ASSERT_EQ(f0.free_slabs, total_slabs) << "fresh pool must be all-free";
    const int free_pages_before = pa.free_pages(0, lmem::Pool::kMain);

    const int n = 3;
    auto run = pa.claim_expert_zone(0, n, /*extra_reserve_pages=*/0);
    ASSERT_TRUE(run.has_value());
    EXPECT_EQ(run->gpu_idx, 0);
    EXPECT_EQ(run->num_slabs, n);
    // TOP-DOWN first-fit on a fresh pool lands on the LAST n slabs.
    EXPECT_EQ(run->start_slab, total_slabs - n);
    EXPECT_EQ(run->base,
              static_cast<char*>(pa.kv_main_base(0))
                  + static_cast<int64_t>(run->start_slab) * pa.slab_bytes());

    // Whole slabs left the shared pool; the free-slab list shrank by n.
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), free_pages_before - n * pps);
    const auto f1 = pa.kv_fragmentation(0);
    EXPECT_EQ(f1.total_slabs, total_slabs) << "a grant is not a capacity loss";
    EXPECT_EQ(f1.free_slabs, total_slabs - n);
    EXPECT_EQ(f1.live_slabs, n);
    // Side-tenant stamp: fully used, so no fragmented free space inside.
    EXPECT_EQ(f1.used_pages, n * pps);
    EXPECT_EQ(f1.fragmented_free_pages, 0);

    const auto& st = pa.expert_zone_stats(0);
    EXPECT_EQ(st.grants, 1);
    EXPECT_EQ(st.releases, 0);
    EXPECT_EQ(st.refusals, 0);
    EXPECT_EQ(st.granted_slabs, n);

    pa.release_expert_zone(0, run->start_slab, run->num_slabs);
}

TEST(PageAllocatorExpertZone, ClaimDoesNotShrinkTotalPages) {
    // The admissibility model quotes total_pages(); a grant is reclaimable
    // CACHE, so the boot capacity — and with it the NON-RETRYABLE seq_create
    // refusal class — must read identically with a grant outstanding.
    auto a = make_slabbed();
    auto& pa = *a.pages;
    const int total_before = pa.total_pages(0, lmem::Pool::kMain);
    const int admissible_before = pa.admissible_ctx_tokens();

    auto run = pa.claim_expert_zone(0, 4, 0);
    ASSERT_TRUE(run.has_value());
    EXPECT_EQ(pa.total_pages(0, lmem::Pool::kMain), total_before);
    EXPECT_EQ(pa.admissible_ctx_tokens(), admissible_before);

    pa.release_expert_zone(0, run->start_slab, run->num_slabs);
    EXPECT_EQ(pa.total_pages(0, lmem::Pool::kMain), total_before);
}

TEST(PageAllocatorExpertZone, InvisibleToKvPlacement) {
    // While held, a granted slab must be unreachable by KV placement: the
    // side-tenant stamp (used == bump == pps) hides it from every step of
    // the S2 placement order, right up to full exhaustion of the pool.
    auto a = make_slabbed();
    auto& pa = *a.pages;
    const int pps = pa.pages_per_slab();
    const int total_slabs = pa.kv_fragmentation(0).total_slabs;
    const int total_pages = pa.total_pages(0, lmem::Pool::kMain);

    const int n = 2;
    auto run = pa.claim_expert_zone(0, n, 0);
    ASSERT_TRUE(run.has_value());
    const int start = run->start_slab;

    std::vector<lmem::PageHandle> kv;
    for (int i = 0; i < total_pages + 1; ++i) {
        auto h = pa.allocate_for_sequence(
            0, lmem::Pool::kMain, /*seq=*/7, /*layer=*/0,
            static_cast<uint32_t>(i * 16), static_cast<uint32_t>(i * 16 + 16));
        if (!h) break;
        EXPECT_FALSE(in_grant(h->page_idx, start, n, pps))
            << "KV page " << h->page_idx << " landed inside the grant";
        kv.push_back(*h);
    }
    // Exhaustion arrives exactly one grant early — the grant is held, not lost.
    EXPECT_EQ(static_cast<int>(kv.size()), total_pages - n * pps);
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), 0);

    for (auto& h : kv) pa.free(h);
    EXPECT_EQ(pa.kv_fragmentation(0).live_slabs, n) << "only the grant remains";
    pa.release_expert_zone(0, start, n);
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), total_pages);
    EXPECT_EQ(pa.kv_fragmentation(0).free_slabs, total_slabs);
}

TEST(PageAllocatorExpertZone, ReleaseFrontInsertsAndRestores) {
    // TD-KDA-MAPPED-FRAG (b2): a released grant goes to the free-list
    // BOTTOM, so the next single-slab side claim (which pops the LIFO top)
    // cannot colonize and split the just-freed run — it re-coalesces.
    auto a = make_slabbed();
    auto& pa = *a.pages;
    const int pps = pa.pages_per_slab();
    const int total_slabs = pa.kv_fragmentation(0).total_slabs;
    const int free_pages_before = pa.free_pages(0, lmem::Pool::kMain);

    const int n = 3;
    auto run = pa.claim_expert_zone(0, n, 0);
    ASSERT_TRUE(run.has_value());
    const int start = run->start_slab;
    ASSERT_EQ(pa.free_pages(0, lmem::Pool::kMain),
              free_pages_before - n * pps);

    pa.release_expert_zone(0, start, n);
    const auto f = pa.kv_fragmentation(0);
    EXPECT_EQ(f.free_slabs, total_slabs);
    EXPECT_EQ(f.live_slabs, 0);
    EXPECT_EQ(f.granted_slabs, 0);
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), free_pages_before);
    const auto& st = pa.expert_zone_stats(0);
    EXPECT_EQ(st.releases, 1);
    EXPECT_EQ(st.granted_slabs, 0);

    // The released ids sit at the FRONT: the next elastic single-slab claim
    // must come off the far end of the list, outside the released span.
    auto ik = pa.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(ik.has_value());
    // An elastic indexer page_idx IS the slab id.
    const bool inside = ik->page_idx >= start && ik->page_idx < start + n;
    EXPECT_FALSE(inside)
        << "a single-slab claim colonized the just-released grant (slab "
        << ik->page_idx << " in [" << start << ", " << start + n << "))";
    pa.free(*ik);

    // Still perfectly claimable under pressure — merely LAST in line.
    auto again = pa.claim_expert_zone(0, n, 0);
    ASSERT_TRUE(again.has_value());
    EXPECT_EQ(again->start_slab, start) << "the run re-coalesced whole";
    pa.release_expert_zone(0, start, n);
}

TEST(PageAllocatorExpertZone, HeadroomRefusalCounts) {
    // extra_reserve_pages is the rebalancer's KV growth margin on top of the
    // INV-4.9f floor. A grant that would breach it is refused, and refusing
    // claims NOTHING.
    auto a = make_slabbed();
    auto& pa = *a.pages;
    const int free_before = pa.free_pages(0, lmem::Pool::kMain);
    const int total_slabs = pa.kv_fragmentation(0).total_slabs;

    auto refused = pa.claim_expert_zone(0, 2, /*extra_reserve_pages=*/
                                        static_cast<int64_t>(free_before));
    EXPECT_FALSE(refused.has_value());
    const auto& st = pa.expert_zone_stats(0);
    EXPECT_EQ(st.refusals, 1);
    EXPECT_EQ(st.grants, 0);
    EXPECT_EQ(st.granted_slabs, 0);
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), free_before);
    EXPECT_EQ(pa.kv_fragmentation(0).free_slabs, total_slabs);
    EXPECT_EQ(pa.largest_free_run(0), total_slabs);

    // The identical claim with no extra margin succeeds.
    auto ok = pa.claim_expert_zone(0, 2, 0);
    ASSERT_TRUE(ok.has_value());
    EXPECT_EQ(st.refusals, 1);
    EXPECT_EQ(st.grants, 1);
    pa.release_expert_zone(0, ok->start_slab, ok->num_slabs);
}

TEST(PageAllocatorExpertZone, RefusesWhenNoContiguousRunExists) {
    // Bytes are not adjacency: with enough free slabs but no run of the
    // requested length, the claim refuses rather than splitting.
    auto a = make_slabbed();
    auto& pa = *a.pages;
    const int total_slabs = pa.kv_fragmentation(0).total_slabs;
    ASSERT_GE(total_slabs, 6);

    // Fragment: claim every other slab, leaving max run length 1.
    std::vector<lmem::PageHandle> pins;
    for (int s = 1; s < total_slabs; s += 2) {
        auto h = pa.claim_one_slab_for_test(0, s);
        ASSERT_TRUE(h.has_value()) << s;
        pins.push_back(*h);
    }
    ASSERT_EQ(pa.largest_free_run(0), 1);
    EXPECT_FALSE(pa.claim_expert_zone(0, 2, 0).has_value());
    EXPECT_EQ(pa.expert_zone_stats(0).refusals, 1);
    // A 1-slab grant still fits.
    auto one = pa.claim_expert_zone(0, 1, 0);
    ASSERT_TRUE(one.has_value());
    pa.release_expert_zone(0, one->start_slab, one->num_slabs);
    for (auto& h : pins) pa.free(h);
}

TEST(PageAllocatorExpertZone, TargetedMirrorClaim) {
    // TP lockstep: replicated KV claims slabs BY INDEX on every rank, so the
    // rebalancer mirrors rank 0's chosen run verbatim. An overlapping mirror
    // claim must fail whole (S4 / INV-KVT-14b: divergent free sets would
    // collapse replicated capacity to the ranks' intersection).
    auto a = make_slabbed();
    auto& pa = *a.pages;
    const int pps = pa.pages_per_slab();
    const int total_slabs = pa.kv_fragmentation(0).total_slabs;
    ASSERT_GE(total_slabs, 8);
    const int free_before = pa.free_pages(0, lmem::Pool::kMain);

    const int start = 2;
    const int n = 3;
    ASSERT_TRUE(pa.claim_expert_zone_at(0, start, n, 0));
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), free_before - n * pps);
    const auto& st = pa.expert_zone_stats(0);
    EXPECT_EQ(st.grants, 1);
    EXPECT_EQ(st.granted_slabs, n);
    EXPECT_EQ(pa.kv_fragmentation(0).granted_slabs, n);

    // Overlapping (shares slab `start + n - 1`): refused, nothing claimed.
    const int free_now = pa.free_pages(0, lmem::Pool::kMain);
    EXPECT_FALSE(pa.claim_expert_zone_at(0, start + n - 1, 3, 0));
    EXPECT_EQ(pa.expert_zone_stats(0).refusals, 1);
    EXPECT_EQ(pa.expert_zone_stats(0).grants, 1);
    EXPECT_EQ(pa.expert_zone_stats(0).granted_slabs, n);
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), free_now);
    // No slab outside the first grant was touched by the refused claim.
    EXPECT_EQ(pa.kv_fragmentation(0).live_slabs, n);

    // Out-of-range spans refuse too.
    EXPECT_FALSE(pa.claim_expert_zone_at(0, total_slabs - 1, 4, 0));
    EXPECT_EQ(pa.expert_zone_stats(0).refusals, 2);

    // A disjoint mirror claim succeeds alongside the first.
    ASSERT_TRUE(pa.claim_expert_zone_at(0, start + n, 2, 0));
    EXPECT_EQ(pa.expert_zone_stats(0).granted_slabs, n + 2);
    pa.release_expert_zone(0, start + n, 2);
    pa.release_expert_zone(0, start, n);
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), free_before);
    EXPECT_EQ(pa.expert_zone_stats(0).granted_slabs, 0);
}

TEST(PageAllocatorExpertZone, CoexistsWithMappedKdaState) {
    // The two contiguous-run tenants share the region: grants pack top-down
    // above the KDA state band. Neither may intersect the other, and the
    // pool must return to all-free once both hand back.
    auto a = make_mapped_kda();
    auto& pa = *a.pages;
    ASSERT_TRUE(pa.kda_state_mapped());
    const int pps = pa.pages_per_slab();
    ASSERT_GT(pps, 0);
    const int unit_slabs = pa.kda_unit_slabs();
    const int units = pa.kda_layout().num_layers;
    ASSERT_GT(unit_slabs, 0);
    ASSERT_GT(units, 0);

    const int free_before = pa.free_pages(0, lmem::Pool::kMain);
    const auto f0 = pa.kv_fragmentation(0);
    ASSERT_EQ(f0.free_slabs, f0.total_slabs);
    if (f0.free_slabs < units * unit_slabs + 1)
        GTEST_SKIP() << "fixture pool too small for a state claim + a grant";

    const int n = 1;
    auto run = pa.claim_expert_zone(0, n, 0);
    ASSERT_TRUE(run.has_value());
    const int start = run->start_slab;

    auto hs = pa.allocate_kda_state(0, /*seq_id=*/42);
    ASSERT_EQ(static_cast<int>(hs.size()), units)
        << "the grant must not starve the state claim";
    for (const auto& h : hs) {
        // Mapped page_idx is the run-START slab id; the unit spans
        // [page_idx, page_idx + unit_slabs).
        for (int j = 0; j < unit_slabs; ++j) {
            EXPECT_FALSE(in_grant((h.page_idx + j) * pps, start, n, pps))
                << "KDA state unit at slab " << (h.page_idx + j)
                << " intersects the expert-zone grant";
        }
    }
    EXPECT_EQ(pa.kv_fragmentation(0).granted_slabs, n)
        << "state runs are NOT grants";

    for (auto& h : hs) pa.free(h);
    pa.release_expert_zone(0, start, n);
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), free_before);
    const auto f1 = pa.kv_fragmentation(0);
    EXPECT_EQ(f1.free_slabs, f1.total_slabs);
    EXPECT_EQ(f1.live_slabs, 0);
    EXPECT_EQ(f1.granted_slabs, 0);
}

TEST(PageAllocatorExpertZone, FragmentationReportsGrantedSlabs) {
    auto a = make_slabbed();
    auto& pa = *a.pages;
    EXPECT_EQ(pa.kv_fragmentation(0).granted_slabs, 0);

    auto r1 = pa.claim_expert_zone(0, 2, 0);
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(pa.kv_fragmentation(0).granted_slabs, 2);

    auto r2 = pa.claim_expert_zone(0, 3, 0);
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(pa.kv_fragmentation(0).granted_slabs, 5);
    // granted_slabs is a SUBSET of live_slabs, not a separate population.
    EXPECT_EQ(pa.kv_fragmentation(0).live_slabs, 5);

    pa.release_expert_zone(0, r1->start_slab, r1->num_slabs);
    EXPECT_EQ(pa.kv_fragmentation(0).granted_slabs, 3);
    pa.release_expert_zone(0, r2->start_slab, r2->num_slabs);
    EXPECT_EQ(pa.kv_fragmentation(0).granted_slabs, 0);

    // An unslabbed / out-of-range GPU reports inert zeros.
    EXPECT_EQ(pa.kv_fragmentation(3).granted_slabs, 0);
    EXPECT_EQ(pa.expert_zone_stats(3).grants, 0);
}

TEST(PageAllocatorExpertZone, LargestFreeRun) {
    auto a = make_slabbed();
    auto& pa = *a.pages;
    const int total_slabs = pa.kv_fragmentation(0).total_slabs;
    ASSERT_GE(total_slabs, 8);
    EXPECT_EQ(pa.largest_free_run(0), total_slabs) << "fresh pool is one run";

    // Fragment the middle: slab m splits the region into [0, m) and
    // (m, total_slabs).
    const int m = total_slabs / 2;
    auto pin = pa.claim_one_slab_for_test(0, m);
    ASSERT_TRUE(pin.has_value());
    EXPECT_EQ(pa.largest_free_run(0), std::max(m, total_slabs - 1 - m));

    // A top grant of n eats the top of the upper segment.
    const int n = 2;
    ASSERT_GE(total_slabs - 1 - m, n);
    auto run = pa.claim_expert_zone(0, n, 0);
    ASSERT_TRUE(run.has_value());
    EXPECT_EQ(run->start_slab, total_slabs - n);
    EXPECT_EQ(pa.largest_free_run(0), std::max(m, total_slabs - n - 1 - m));

    pa.release_expert_zone(0, run->start_slab, run->num_slabs);
    EXPECT_EQ(pa.largest_free_run(0), std::max(m, total_slabs - 1 - m));
    pa.free(*pin);
    EXPECT_EQ(pa.largest_free_run(0), total_slabs);

    // Unslabbed / out-of-range GPUs report 0.
    EXPECT_EQ(pa.largest_free_run(3), 0);
    EXPECT_EQ(pa.largest_free_run(99), 0);
}
