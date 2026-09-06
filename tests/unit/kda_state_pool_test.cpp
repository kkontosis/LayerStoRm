// GF3.8: KDA per-request state pool — unit tests (all CPU-only).
//
// The KDA recurrent state is PER REQUEST, not per token: one fixed-size fp32
// slot per sequence (recurrent [H/tp][D][D] + 3 conv rings per linear layer),
// claimed at seq_create, ZEROED on claim (INV-V4-DET obligation (2)),
// D2D-copied on EVERY fork (frozen included), freed at seq_free, retryable
// kKvPoolExhausted on capacity. The slot is the GF3.6 checkpoint unit
// (INV-KDA-REWIND): seq_snapshot/restore refuse loudly until GF3.12 owns
// state checkpoints.
//
// Suites (all CPU — no _GPU_FILTER registration):
//   KdaStateLayoutMath        — compute_kda_state_layout pure math
//   VramAllocatorKdaState     — boot sizing, clamp, regions, non-KDA zeroes
//   PageAllocatorKdaState     — Pool::kKdaState side-pool mechanics
//   CommandDispatcherKdaState — seq lifecycle through the dispatcher
//     (NullDeviceBackend memset/memcpy are REAL host ops, so zero-on-claim
//      and copy-on-fork are asserted on content, not just op counts).

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <memory>
#include <optional>
#include <vector>

#include "compute/stream_manager.h"
#include "config/config_parser.h"
#include "core/gpu_ref.h"
#include "core/memory/page_allocator.h"
#include "core/memory/vram_allocator.h"
#include "core/null_attention_device.h"
#include "core/null_device_backend.h"
#include "daemon/command_dispatcher.h"
#include "daemon/ipc_protocol.h"
#include "daemon/spsc_ring.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/fp8.h"

namespace lipc = layerstorm::ipc;
namespace ldam = layerstorm::daemon;
namespace lmem = layerstorm::memory;
namespace lmod = layerstorm::model;
namespace lcomp = layerstorm::compute;
namespace lc = layerstorm::config;

namespace {

constexpr uint32_t kTestSlots = 128;
/// KV page granularity handed to the fixture's DcpConfig at tp >= 2.
/// MUST equal memory::DcpConfig's default page_size_tokens: at tp == 1 the
/// fixture never calls set_dcp_config, so handle_seq_create reads that
/// default — pinning the same value here keeps the 1-GPU and 2-GPU arms on
/// IDENTICAL seq_create page arithmetic (and satisfies set_dcp_config's
/// dcp_chunk_size % page_size_tokens == 0 assert).
constexpr int kTestPageTokens = 16;

void* aligned_alloc_zeroed(size_t bytes) {
    void* p = std::aligned_alloc(64, bytes);
    std::memset(p, 0, bytes);
    return p;
}

/// GLM-5.3-Flash-shaped glm5_next config (full geometry: 45 layers, KDA
/// 64 heads x 128, conv 4) on 32 GB GPUs — the vram_allocator_test
/// glm5n_config() shape, rebuilt here so the suites stay independent.
lc::Config glm5n_full_config() {
    auto j = nlohmann::json{
        {"model", {
            {"architecture",            "deepseek_v3"},
            {"weights_path",            "/data/models/test/"},
            {"weights_format",          "safetensors"},
            {"num_hidden_layers",       45},
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
            {"qk_rope_head_dim",        0},
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
        {"quantization", {{"weights", "fp8_e4m3"},
                          {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"},
                          {"gating_compute", "fp32"}}},
        {"hardware", {
            {"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}},
                      {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 32}},
                      {{"id", 2}, {"type", "rtx5080"}, {"vram_gb", 16}}}},
            {"tp_array", {0, 1}},
            {"system_ram_gb", 256}}},
    };
    auto cfg = lc::parse_config(j);
    cfg.model.architecture = lc::Architecture::glm5_next;
    cfg.model.index_kpool = 4;
    cfg.model.layer_types.clear();
    for (int l = 0; l < 45; ++l)
        cfg.model.layer_types.push_back(
            (l % 4 == 3) || l == 43
                ? lc::LayerAttentionType::deepseek_sparse_attention
                : lc::LayerAttentionType::linear_attention);
    lc::LinearAttnConfig la;
    la.num_heads = 64;
    la.head_dim = 128;
    la.short_conv_kernel_size = 4;
    la.gate_lower_bound = -5.0;
    cfg.model.linear_attn_config = la;
    return cfg;
}

/// Tiny glm5_next config for the dispatcher fixture: 6 layers (4 linear +
/// 2 sparse-typed), MINIATURE KDA geometry (4 heads x 32, conv 4 ⇒ slot =
/// 4 x (4*32*32*4 + 3*3*4*32*4) = 83,968 B), index_topk 0 (no indexer
/// machinery — the state pool does not depend on it), ONE GPU by default.
///
/// TD-KDA-MAPPED-MULTIGPU: num_gpus == 2 declares a second identical GPU
/// AND hardware.tp_array {0,1} — tp_degree 2 shards the KDA head count
/// per rank (4 → 2 heads, halving slot_bytes) exactly as production TP
/// does, and both GPUs are TP members so both carry a KV/slab region for
/// the mapped state to claim from.
lc::Config glm5n_small_config(int num_gpus = 1) {
    auto gpus_j = nlohmann::json::array();
    for (int i = 0; i < num_gpus; ++i)
        gpus_j.push_back(nlohmann::json{{"id", i},
                                        {"type", "rtx5090"},
                                        {"vram_gb", 1}});
    auto hw_j = nlohmann::json{{"gpus", gpus_j}, {"system_ram_gb", 64}};
    if (num_gpus >= 2) {
        auto tp_j = nlohmann::json::array();
        for (int i = 0; i < num_gpus; ++i) tp_j.push_back(i);
        hw_j["tp_array"] = tp_j;
    }
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
        {"hardware", hw_j},
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
            (l % 3 == 2)
                ? lc::LayerAttentionType::deepseek_sparse_attention
                : lc::LayerAttentionType::linear_attention);
    lc::LinearAttnConfig la;
    la.num_heads = 4;
    la.head_dim = 32;
    la.short_conv_kernel_size = 4;
    la.gate_lower_bound = -5.0;
    cfg.model.linear_attn_config = la;
    return cfg;
}

}  // namespace

// ── KdaStateLayoutMath ──────────────────────────────────────────────────────

TEST(KdaStateLayoutMath, Glm53FlashGeometry) {
    auto cfg = glm5n_full_config();
    lmod::ModelConfig mcfg(cfg);
    ASSERT_EQ(mcfg.num_linear_attention_layers(), 34);

    const auto k = lmem::compute_kda_state_layout(mcfg, /*tp=*/1,
                                                  /*spec_columns=*/0);
    ASSERT_TRUE(k.enabled);
    EXPECT_EQ(k.num_layers, 34);
    EXPECT_EQ(k.heads_per_rank, 64);
    EXPECT_EQ(k.head_dim, 128);
    EXPECT_EQ(k.conv_kernel, 4);
    // PLAN.md GF3.8 numbers: recurrent 64x128x128 fp32 = 4 MiB/layer,
    // 3 conv rings (kernel-1 = 3 columns) x 64*128 fp32 = 288 KiB/layer,
    // ~4.28 MiB/layer, ~146 MiB/request over 34 layers.
    EXPECT_EQ(k.recurrent_bytes_per_layer, 64LL * 128 * 128 * 4);
    EXPECT_EQ(k.ring_bytes_per_layer, 3LL * 64 * 128 * 4);
    EXPECT_EQ(k.per_layer_bytes, 4489216);
    EXPECT_EQ(k.slot_bytes, 152633344);  // already 256-aligned
    // Offsets: per-layer contiguous [recurrent | ring_q | ring_k | ring_v].
    EXPECT_EQ(k.recurrent_offset(0), 0);
    EXPECT_EQ(k.recurrent_offset(1), k.per_layer_bytes);
    EXPECT_EQ(k.ring_offset(0, 0), k.recurrent_bytes_per_layer);
    EXPECT_EQ(k.ring_offset(0, 2),
              k.recurrent_bytes_per_layer + 2 * k.ring_bytes_per_layer);
    EXPECT_EQ(k.ring_offset(33, 2) + k.ring_bytes_per_layer,
              33 * k.per_layer_bytes + k.per_layer_bytes);

    // TP=2 halves the per-rank shard.
    const auto k2 = lmem::compute_kda_state_layout(mcfg, 2, 0);
    EXPECT_EQ(k2.heads_per_rank, 32);
    EXPECT_EQ(k2.per_layer_bytes, k.per_layer_bytes / 2);
}

TEST(KdaStateLayoutMath, DisabledForNonLinearModels) {
    auto cfg = glm5n_full_config();
    cfg.model.architecture = lc::Architecture::deepseek_v3;
    cfg.model.layer_types.clear();  // no linear layers
    lmod::ModelConfig mcfg(cfg);
    const auto k = lmem::compute_kda_state_layout(mcfg, 1, 0);
    EXPECT_FALSE(k.enabled);
    EXPECT_EQ(k.slot_bytes, 0);
    EXPECT_EQ(k.num_layers, 0);
}

TEST(KdaStateLayoutMath, TpMustDivideHeads) {
    auto cfg = glm5n_full_config();
    lmod::ModelConfig mcfg(cfg);
    EXPECT_THROW(lmem::compute_kda_state_layout(mcfg, 3, 0),
                 std::invalid_argument);
    EXPECT_THROW(lmem::compute_kda_state_layout(mcfg, 1, -1),
                 std::invalid_argument);
}

TEST(KdaStateLayoutMath, SpecColumnsWidenConvRings) {
    // The GF3.11 seam: a replay-ring design widens the conv rings by S
    // columns; the recurrent block is untouched. Wired 0 in production
    // until GF3.11 decides (snapshot-per-round was the recommendation).
    auto cfg = glm5n_full_config();
    lmod::ModelConfig mcfg(cfg);
    const auto k0 = lmem::compute_kda_state_layout(mcfg, 1, 0);
    const auto k8 = lmem::compute_kda_state_layout(mcfg, 1, 8);
    EXPECT_EQ(k8.recurrent_bytes_per_layer, k0.recurrent_bytes_per_layer);
    EXPECT_EQ(k8.ring_bytes_per_layer,
              (3 + 8) * 64LL * 128 * 4);
    EXPECT_GT(k8.slot_bytes, k0.slot_bytes);
}

// ── VramAllocatorKdaState ───────────────────────────────────────────────────

TEST(VramAllocatorKdaState, PoolSizedForConcurrencyPlusHolders) {
    // Mapped is the DEFAULT since 2026-08-31 — this test exercises the
    // surviving CARVE off-path (LS_KDA_STATE_MAPPED=0 / mapped=false), so
    // it pins the switch explicitly.
    unsetenv("LS_KDA_STATE_MAPPED");
    auto cfg = glm5n_full_config();
    cfg._internal_kda_state.mapped = false;
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);

    ASSERT_TRUE(layout.kda.enabled);
    // tp_array {0,1} ⇒ per-rank H/tp = 32 heads.
    EXPECT_EQ(layout.kda.heads_per_rank, 32);
    const int policy = cfg.serving.max_concurrent_requests
                     + cfg.serving.prefix_cache.max_entries;
    for (int g = 0; g < 2; ++g) {
        const auto& gl = layout.gpus[g];
        EXPECT_GE(gl.kda_state_slots, 1) << "gpu " << g;
        EXPECT_LE(gl.kda_state_slots, policy) << "gpu " << g;
        EXPECT_EQ(gl.kda_state_bytes,
                  static_cast<int64_t>(gl.kda_state_slots)
                      * layout.kda.slot_bytes) << "gpu " << g;
    }
    // Non-TP GPU: no attention, no state pool.
    EXPECT_EQ(layout.gpus[2].kda_state_slots, 0);
    EXPECT_EQ(layout.gpus[2].kda_state_bytes, 0);
}

TEST(VramAllocatorKdaState, NonKdaModelsCarveNothing) {
    // Byte-identity leg (the GF3.2 proof shape): on a non-glm5_next model
    // the KDA branch contributes literally zero bytes to every region, so
    // the layout arithmetic is unchanged — the whole pre-existing
    // VramAllocator/PageAllocator suite green is the rest of the proof.
    auto cfg = glm5n_full_config();
    cfg.model.architecture = lc::Architecture::deepseek_v3;
    cfg.model.layer_types.clear();
    cfg.model.linear_attn_config.reset();
    cfg.model.index_kpool = 1;
    cfg.model.qk_rope_head_dim = 64;  // back to the V3.2 MLA shape
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    EXPECT_FALSE(layout.kda.enabled);
    EXPECT_EQ(layout.kda.slot_bytes, 0);
    for (const auto& gl : layout.gpus) {
        EXPECT_EQ(gl.kda_state_bytes, 0);
        EXPECT_EQ(gl.kda_state_slots, 0);
    }
}

TEST(VramAllocatorKdaState, ThrowsWhenNotEvenOneSlotFits) {
    // A glm5_next boot that cannot hold ONE sequence's state must fail AT
    // BOOT, not at the first seq_create — on BOTH paths. 1 GB TP GPUs
    // cannot host a ~146 MiB slot after pinned weights of a 7168-hidden
    // 45-layer stack.
    unsetenv("LS_KDA_STATE_MAPPED");
    // Carve off-path: the sizing itself refuses (not even one slot).
    {
        auto cfg = glm5n_full_config();
        cfg._internal_kda_state.mapped = false;
        for (auto& g : cfg.hardware.gpus) g.vram_gb = 1;
        lmod::ModelConfig mcfg(cfg);
        lmod::Fp8E4M3 fp8;
        lmod::LayerRegistry reg(mcfg, cfg, fp8);
        EXPECT_THROW(lmem::compute_vram_layout(cfg, reg, mcfg),
                     std::runtime_error);
    }
    // Mapped default: sizing has no slot count to refuse on, so the
    // fail-loud parity check lives in the PageAllocator ctor (region
    // cannot hold num_layers contiguous unit runs). Either stage may
    // throw first on this degenerate 1 GB shape — the contract is only
    // that SOMETHING refuses at boot.
    {
        auto cfg = glm5n_full_config();
        for (auto& g : cfg.hardware.gpus) g.vram_gb = 1;
        lmod::ModelConfig mcfg(cfg);
        lmod::Fp8E4M3 fp8;
        lmod::LayerRegistry reg(mcfg, cfg, fp8);
        auto boot = [&] {
            auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
            EXPECT_TRUE(layout.kda.mapped);
            std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned;
            std::vector<lcomp::DeviceBackend*> ptrs;
            for (size_t i = 0; i < layout.gpus.size(); ++i) {
                lc::GpuRef gref{static_cast<int>(i), static_cast<int>(i),
                                lc::GpuType::rtx5090};
                owned.push_back(lcomp::make_null_device_backend(gref));
                ptrs.push_back(owned.back().get());
            }
            lmem::VramAllocator vram(std::move(layout), ptrs);
            lmem::PageAllocator pages(vram, ptrs[0]);
        };
        EXPECT_ANY_THROW(boot());
    }
}

// ── PageAllocatorKdaState + regions ─────────────────────────────────────────

namespace {

struct SmallAllocators {
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned;
    std::vector<lcomp::DeviceBackend*> ptrs;
    std::unique_ptr<lmem::VramAllocator> vram;
    std::unique_ptr<lmem::PageAllocator> pages;
    lmem::VramLayout layout_copy;
};

SmallAllocators make_small_kda_allocators() {
    SmallAllocators a;
    auto cfg = glm5n_small_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    a.layout_copy = layout;
    for (size_t i = 0; i < layout.gpus.size(); ++i) {
        lc::GpuRef gref{static_cast<int>(i), static_cast<int>(i),
                        lc::GpuType::rtx5090};
        a.owned.push_back(lcomp::make_null_device_backend(gref));
        a.ptrs.push_back(a.owned.back().get());
    }
    a.vram = std::make_unique<lmem::VramAllocator>(std::move(layout), a.ptrs);
    a.pages = std::make_unique<lmem::PageAllocator>(*a.vram, a.ptrs[0]);
    return a;
}

}  // namespace

TEST(PageAllocatorKdaState, AllocFreeCycleAndExhaustion) {
    auto a = make_small_kda_allocators();
    auto& pa = *a.pages;
    const int total = pa.total_pages(0, lmem::Pool::kKdaState);
    ASSERT_GT(total, 0);
    EXPECT_EQ(pa.kda_state_slot_bytes(), a.layout_copy.kda.slot_bytes);
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kKdaState), total);

    std::vector<lmem::PageHandle> hs;
    for (int i = 0; i < total; ++i) {
        auto h = pa.allocate(0, lmem::Pool::kKdaState);
        ASSERT_TRUE(h.has_value()) << i;
        EXPECT_EQ(h->pool, lmem::Pool::kKdaState);
        ASSERT_NE(h->gpu_ptr, nullptr);
        hs.push_back(*h);
    }
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kKdaState), 0);
    EXPECT_FALSE(pa.allocate(0, lmem::Pool::kKdaState).has_value());
    // Uniform stride from one base — the GF3.7 base + slot * stride kernel
    // contract. page_idx IS the kernel slot index.
    const auto& reg0 = a.vram->region(0);
    for (const auto& h : hs) {
        EXPECT_EQ(h.gpu_ptr,
                  static_cast<char*>(reg0.kda_state)
                      + static_cast<int64_t>(h.page_idx)
                            * a.layout_copy.kda.slot_bytes);
    }
    for (auto& h : hs) pa.free(h);
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kKdaState), total);
}

TEST(PageAllocatorKdaState, FreeSequenceSweepsKdaState) {
    auto a = make_small_kda_allocators();
    auto& pa = *a.pages;
    auto h = pa.allocate(0, lmem::Pool::kKdaState);
    ASSERT_TRUE(h.has_value());
    pa.meta(*h).sequence_id = 77;
    const int used_before = pa.used_pages(0, lmem::Pool::kKdaState);
    pa.free_sequence(0, 77);
    EXPECT_EQ(pa.used_pages(0, lmem::Pool::kKdaState), used_before - 1);
}

TEST(PageAllocatorKdaState, RegionPlacementAndSlotBytesZeroOffKda) {
    auto a = make_small_kda_allocators();
    const auto& reg0 = a.vram->region(0);
    ASSERT_NE(reg0.kda_state, nullptr);
    // kda_state sits between kv_speculation and kv_hca (collapsed V4 tiers
    // follow immediately for non-V4 models).
    EXPECT_GE(static_cast<char*>(reg0.kda_state),
              static_cast<char*>(reg0.kv_speculation));
    EXPECT_LE(static_cast<char*>(reg0.kda_state)
                  + a.layout_copy.gpus[0].kda_state_bytes,
              static_cast<char*>(reg0.kv_main));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(reg0.kda_state) % 256, 0u);
}

// ── CommandDispatcherKdaState ───────────────────────────────────────────────

class CommandDispatcherKdaState : public ::testing::Test {
protected:
    void SetUp() override { build_world(glm5n_small_config(), false); }

    /// TD-KDA-STATE-MAPPED-SLABS: tests that exercise the mapped mode
    /// rebuild the world on a SLABBED glm5_next config with the switch on
    /// (the config channel; the env channel is cleared for hermeticity).
    /// TD-KDA-MAPPED-MULTIGPU: `num_gpus` must match the GPU count the
    /// config declares (build_world asserts it) — 1 keeps every pre-existing
    /// test byte-identical, 2 builds the TP {0,1} world.
    /// TD-KDA-MAPPED-NONTP-GPUS: `tp_degree` decouples the TP SET from the
    /// GPU COUNT (default: every GPU is a TP rank, the pre-existing shape).
    /// tp_degree == 1 with num_gpus == 2 is the EP shape autoconfig derives:
    /// one attention rank plus EXPERT-ONLY GPUs.
    void rebuild(lc::Config base_cfg, bool mapped, int num_gpus = 1,
                 int tp_degree = -1) {
        teardown_world();
        build_world(std::move(base_cfg), mapped, num_gpus, tp_degree);
    }

    /// Builds an N-null-backend-GPU world. At N >= 2 the PageAllocator gets
    /// the ENGINE's DcpConfig (engine.cpp's `tp >= 2` branch, same fields):
    /// dcp_size = N, tp_gpu_indices = {0..N-1}, replicated KV. That vector
    /// is precisely what CommandDispatcher::claim_kda_state iterates, so
    /// setting it here is what engages EVERY rank's state claim — without
    /// it the dispatcher falls back to the create-command GPU alone.
    void build_world(lc::Config base_cfg, bool mapped, int num_gpus = 1,
                     int tp_degree = -1) {
        base_cfg._internal_kda_state.mapped = mapped;
        unsetenv("LS_KDA_STATE_MAPPED");
        ipc_bytes_ = lipc::IpcLayout::total_size(kTestSlots, kTestSlots);
        ipc_region_ = static_cast<uint8_t*>(aligned_alloc_zeroed(ipc_bytes_));
        void* cmp_ptr =
            ipc_region_ + lipc::IpcLayout::cmp_ring_offset(kTestSlots);
        lipc::CompletionRing::init(cmp_ptr, kTestSlots);
        cmp_ring_ = std::make_unique<lipc::CompletionRing>(cmp_ptr);
        sideband_ = ipc_region_
                  + lipc::IpcLayout::sideband_offset(kTestSlots, kTestSlots);

        cfg_ = std::make_unique<lc::Config>(std::move(base_cfg));
        mcfg_ = std::make_unique<lmod::ModelConfig>(*cfg_);
        fp8_ = std::make_unique<lmod::Fp8E4M3>();
        layer_reg_ = std::make_unique<lmod::LayerRegistry>(*mcfg_, *cfg_,
                                                           *fp8_);

        std::vector<lc::GpuRef> grefs;
        std::vector<lcomp::DeviceBackend*> dev_ptrs;
        for (int i = 0; i < num_gpus; ++i) {
            grefs.push_back(lc::GpuRef{i, i, lc::GpuType::rtx5090});
            backends_.push_back(
                lcomp::make_null_device_backend(grefs.back()));
        }
        for (auto& b : backends_) dev_ptrs.push_back(b.get());
        auto layout = lmem::compute_vram_layout(*cfg_, *layer_reg_, *mcfg_);
        ASSERT_EQ(static_cast<int>(layout.gpus.size()), num_gpus)
            << "build_world(num_gpus) disagrees with hardware.gpus";
        slot_bytes_ = layout.kda.slot_bytes;
        kda_slots_ = layout.gpus[0].kda_state_slots;
        vram_ = std::make_unique<lmem::VramAllocator>(std::move(layout),
                                                      dev_ptrs);
        page_allocator_ = std::make_unique<lmem::PageAllocator>(
            *vram_, dev_ptrs[0]);
        num_gpus_ = num_gpus;
        // Mirror engine.cpp EXACTLY: the DcpConfig is installed only at
        // tp >= 2. At tp == 1 the allocator keeps its default (empty
        // tp_gpu_indices) and claim_kda_state falls back to the
        // create-command GPU — the production tp=1 + expert-hosts shape
        // (TD-KDA-MAPPED-NONTP-GPUS).
        const int tp = tp_degree > 0 ? tp_degree : num_gpus;
        if (tp >= 2) {
            std::vector<int> tp_gpus;
            for (int i = 0; i < tp; ++i) tp_gpus.push_back(i);
            page_allocator_->set_dcp_config(lmem::DcpConfig{
                .dcp_size           = tp,
                .dcp_chunk_size     = kTestPageTokens,
                .page_size_tokens   = kTestPageTokens,
                .tp_gpu_indices     = std::move(tp_gpus),
                .indexer_k_sharded  = false,
                .kv_sharded         = false,  // INV-KV-REP, the default
            });
        }

        lcomp::StreamManager::Options sm_opts{.device_backends = dev_ptrs};
        stream_manager_ = std::make_unique<lcomp::StreamManager>(
            std::move(sm_opts));
        std::vector<lcomp::AttentionDevice*> attn_ptrs;
        for (const auto& g : grefs)
            attn_devices_.push_back(lcomp::make_null_attention_device(g));
        for (auto& a : attn_devices_) attn_ptrs.push_back(a.get());

        ldam::CommandDispatcher::Deps deps{
            .cmp_ring             = cmp_ring_.get(),
            .stream_manager       = stream_manager_.get(),
            .page_allocator       = page_allocator_.get(),
            .sideband_base        = sideband_,
            .live_config          = cfg_.get(),
            .attention_devices    = attn_ptrs,
            .device_backends      = dev_ptrs,
            .cuda_kernels_enabled = false,
        };
        // GF3.12: seq_restore's config-mismatch check compares the file
        // header against this stride; the KV body upload itself is
        // skipped in this fixture (no dcp_executor => no kv base ptrs),
        // which is fine — the state section is what these tests assert.
        deps.kv_cache_stride_block = kTestKvStride;
        dispatcher_ = std::make_unique<ldam::CommandDispatcher>(
            std::move(deps));
    }

    void TearDown() override { teardown_world(); }

    void teardown_world() {
        dispatcher_.reset();
        attn_devices_.clear();
        stream_manager_.reset();
        page_allocator_.reset();
        vram_.reset();
        backends_.clear();
        cmp_ring_.reset();
        completions_.clear();
        if (ipc_region_) { std::free(ipc_region_); ipc_region_ = nullptr; }
    }

    void drain() {
        dispatcher_->poll_compute_completions();
        lipc::Completion cmp{};
        while (cmp_ring_->try_read(&cmp)) completions_.push_back(cmp);
    }
    std::optional<lipc::Completion> last_of(uint32_t cmp_type) {
        for (auto it = completions_.rbegin(); it != completions_.rend(); ++it)
            if (it->cmp_type == cmp_type) return *it;
        return std::nullopt;
    }
    void clear_completions() { completions_.clear(); }

    lipc::Completion create_seq(uint64_t seq_id, uint32_t prompt_len = 8,
                                uint32_t pool = 0) {
        clear_completions();
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(lipc::CMD_SEQ_CREATE);
        c.cmd_seq = next_seq_++;
        c.seq_create.seq_id = seq_id;
        c.seq_create.prompt_len = prompt_len;
        c.seq_create.pool = pool;
        c.seq_create.reserve_tokens = 0;
        dispatcher_->dispatch(c);
        drain();
        auto out = last_of(lipc::CMP_SEQ_OP_DONE);
        if (!out) out = last_of(lipc::CMP_ERROR);
        EXPECT_TRUE(out.has_value()) << "no completion for seq_create";
        return out.value_or(lipc::Completion{});
    }
    lipc::Completion fork_seq(uint64_t src, uint64_t dst,
                              uint32_t prefix_len = 0, bool frozen = false) {
        clear_completions();
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(
            frozen ? lipc::CMD_SEQ_FORK_FROZEN : lipc::CMD_SEQ_FORK);
        c.cmd_seq = next_seq_++;
        c.seq_fork.src_seq_id = src;
        c.seq_fork.dst_seq_id = dst;
        c.seq_fork.prefix_len = prefix_len;
        dispatcher_->dispatch(c);
        drain();
        auto out = last_of(lipc::CMP_SEQ_OP_DONE);
        if (!out) out = last_of(lipc::CMP_ERROR);
        EXPECT_TRUE(out.has_value()) << "no completion for seq_fork";
        return out.value_or(lipc::Completion{});
    }
    lipc::Completion hibernate_seq(uint64_t seq_id, uint32_t kv_len) {
        clear_completions();
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(lipc::CMD_SEQ_HIBERNATE);
        c.cmd_seq = next_seq_++;
        c.seq_hibernate.seq_id = seq_id;
        c.seq_hibernate.kv_len = kv_len;
        c.seq_hibernate.spill = 0;
        dispatcher_->dispatch(c);
        drain();
        auto out = last_of(lipc::CMP_SEQ_OP_DONE);
        if (!out) out = last_of(lipc::CMP_ERROR);
        EXPECT_TRUE(out.has_value()) << "no completion for seq_hibernate";
        return out.value_or(lipc::Completion{});
    }
    /// P-29 step 24 (LS_KDA_PREFIX_CKPT): D_CMD_KDA_CKPT — capture a
    /// position-keyed host-RAM prefix checkpoint of the live slot.
    lipc::Completion kda_ckpt_cmd(uint64_t seq_id, uint32_t pos) {
        clear_completions();
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(lipc::D_CMD_KDA_CKPT);
        c.cmd_seq = next_seq_++;
        c.kda_anchor.seq_id = seq_id;
        c.kda_anchor.pos = pos;
        dispatcher_->dispatch(c);
        drain();
        auto out = last_of(lipc::CMP_COMPUTE_DONE);
        if (!out) out = last_of(lipc::CMP_ERROR);
        EXPECT_TRUE(out.has_value()) << "no completion for kda_ckpt";
        return out.value_or(lipc::Completion{});
    }
    lipc::Completion seq_ckpt_cmd(uint32_t cmd_type, uint64_t seq_id,
                                  uint32_t token_count = 0) {
        clear_completions();
        lipc::Command c{};
        c.cmd_type = cmd_type;
        c.cmd_seq = next_seq_++;
        c.seq_ckpt.seq_id = seq_id;
        c.seq_ckpt.token_count = token_count;
        dispatcher_->dispatch(c);
        drain();
        auto out = last_of(lipc::CMP_SEQ_OP_DONE);
        if (!out) out = last_of(lipc::CMP_ERROR);
        EXPECT_TRUE(out.has_value()) << "no completion for seq ckpt cmd";
        return out.value_or(lipc::Completion{});
    }

    static constexpr int64_t kTestKvStride = 512;

    /// GF3.12 checkpoint file layout (mirrors dispatch_lifecycle.cpp's
    /// SeqCkptHeader/SeqCkptKdaExt — the test IS the cross-check that the
    /// format holds its documented shape).
    struct CkptHeader {
        uint32_t magic = 0x4B43534CU;
        uint32_t version = 1;
        uint32_t token_count = 0;
        uint32_t kv_layers = 0;
        uint32_t kv_page_size = 0;
        uint64_t kv_stride_block = 0;
        uint32_t logical_pages = 0;
        uint32_t ik_page_tokens = 0;
        uint64_t ik_page_bytes = 0;
        uint32_t ik_groups = 0;
        uint32_t dcp_size = 0;
    };
    struct CkptKdaExt {
        uint64_t slot_bytes = 0;
        uint32_t ranks = 0;
        uint32_t frontier = 0;
    };
    /// Write a crafted checkpoint file. Geometry matches the fixture:
    /// kv_layers = 6 (no nextn), page 64, stride kTestKvStride, dcp 1.
    /// `ranks` is the SeqCkptKdaExt rank count — the state section then
    /// carries `ranks` whole-slot blobs back to back, so `state_blob` must
    /// be ranks * slot_bytes long (TD-KDA-MAPPED-MULTIGPU: at tp = 2 the
    /// reader demands ranks == 2 and scatters blob r onto rank r).
    void write_ckpt(const std::string& path, uint32_t version,
                    uint32_t token_count,
                    const std::vector<char>* state_blob,
                    uint32_t ranks = 1) {
        CkptHeader h{};
        h.version = version;
        h.token_count = token_count;
        h.kv_layers = 6;
        h.kv_page_size = 64;
        h.kv_stride_block = static_cast<uint64_t>(kTestKvStride);
        h.logical_pages = (token_count + 63) / 64;
        h.dcp_size = 1;
        std::FILE* f = std::fopen(path.c_str(), "wb");
        ASSERT_NE(f, nullptr);
        std::fwrite(&h, sizeof(h), 1, f);
        if (version >= 3) {
            CkptKdaExt e{};
            e.slot_bytes = static_cast<uint64_t>(slot_bytes_);
            e.ranks = ranks;
            e.frontier = token_count;
            std::fwrite(&e, sizeof(e), 1, f);
        }
        std::vector<char> zero(static_cast<size_t>(kTestKvStride), 0);
        for (uint32_t j = 0; j < h.logical_pages * h.kv_layers; ++j)
            std::fwrite(zero.data(), 1, zero.size(), f);
        if (version >= 3 && state_blob)
            std::fwrite(state_blob->data(), 1, state_blob->size(), f);
        std::fclose(f);
    }

    void free_seq(uint64_t seq_id) {
        clear_completions();
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(lipc::CMD_SEQ_FREE);
        c.cmd_seq = next_seq_++;
        c.seq_free.seq_id = seq_id;
        dispatcher_->dispatch(c);
        drain();
        ASSERT_TRUE(last_of(lipc::CMP_SEQ_OP_DONE).has_value());
    }

    int used_kda() const {
        return page_allocator_->used_pages(0, lmem::Pool::kKdaState);
    }
    int free_kda() const {
        return page_allocator_->free_pages(0, lmem::Pool::kKdaState);
    }
    /// Slot pointer by pool index (NullDeviceBackend: plain host memory).
    char* slot_ptr(int idx) const {
        return static_cast<char*>(vram_->region(0).kda_state)
             + static_cast<int64_t>(idx) * slot_bytes_;
    }

    // ── TD-KDA-MAPPED-MULTIGPU: PER-RANK observation ──────────────────
    /// Mapped units currently claimed on GPU `g` (one per linear layer per
    /// live sequence) — the per-rank claim count, not a slab count.
    int used_kda_on(int g) const {
        return page_allocator_->used_pages(g, lmem::Pool::kKdaState);
    }
    /// Free pages of the SHARED kv_main pool on GPU `g`. This is the EXACT
    /// accounting channel: a mapped state run takes whole slabs
    /// (pages_per_slab pages each) while a KV page takes exactly one, so a
    /// claim's per-rank cost is arithmetic, not a packing guess.
    int free_kv_pages(int g) const {
        return page_allocator_->free_pages(g, lmem::Pool::kMain);
    }
    /// Unclaimed whole slabs of the shared region on GPU `g`.
    int free_kv_slabs(int g) const {
        return page_allocator_->kv_fragmentation(g).free_slabs;
    }
    /// Layers that consume a kMain KV page per logical page (glm5_next:
    /// the sparse-MLA layers + any MTP layer; the linear layers carry
    /// recurrent state instead) — mirrors CommandDispatcher's kmain_layer_.
    int kmain_layers() const {
        int n = cfg_->model.num_nextn_predict_layers;
        for (auto t : cfg_->model.layer_types)
            if (t != lc::LayerAttentionType::linear_attention) ++n;
        return n;
    }
    /// Claim slabs on GPU `g` (bottom-up, via the S4 test hook) until only
    /// `target_free` whole slabs remain free there. Returns the handles;
    /// free them to release the pressure. Other ranks are untouched — this
    /// is how a SINGLE rank is made unable to satisfy a state claim.
    std::vector<lmem::PageHandle> squeeze_slabs_to(int g, int target_free) {
        std::vector<lmem::PageHandle> held;
        const int total = page_allocator_->total_pages(g, lmem::Pool::kMain)
                        / std::max(1, page_allocator_->pages_per_slab());
        for (int sid = 0; sid < total && free_kv_slabs(g) > target_free;
             ++sid)
            if (auto h = page_allocator_->claim_one_slab_for_test(g, sid))
                held.push_back(*h);
        return held;
    }

    size_t ipc_bytes_ = 0;
    uint8_t* ipc_region_ = nullptr;
    uint8_t* sideband_ = nullptr;
    std::unique_ptr<lipc::CompletionRing> cmp_ring_;
    std::unique_ptr<lc::Config> cfg_;
    std::unique_ptr<lmod::ModelConfig> mcfg_;
    std::unique_ptr<lmod::Fp8E4M3> fp8_;
    std::unique_ptr<lmod::LayerRegistry> layer_reg_;
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> backends_;
    std::unique_ptr<lmem::VramAllocator> vram_;
    std::unique_ptr<lmem::PageAllocator> page_allocator_;
    std::unique_ptr<lcomp::StreamManager> stream_manager_;
    std::vector<std::unique_ptr<lcomp::AttentionDevice>> attn_devices_;
    std::unique_ptr<ldam::CommandDispatcher> dispatcher_;
    std::vector<lipc::Completion> completions_;
    int64_t slot_bytes_ = 0;
    int kda_slots_ = 0;
    int num_gpus_ = 1;
    uint32_t next_seq_ = 1;
};

TEST_F(CommandDispatcherKdaState, CreateClaimsAndZeroesSlot) {
    ASSERT_GT(kda_slots_, 0);
    ASSERT_GT(slot_bytes_, 0);
    // Poison the first slot BEFORE the claim (free list hands out index 0
    // first) — zero-on-claim must erase it.
    std::memset(slot_ptr(0), 0xAB, static_cast<size_t>(slot_bytes_));
    auto c = create_seq(1);
    ASSERT_EQ(c.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(used_kda(), 1);
    const char* p = slot_ptr(0);
    for (int64_t i = 0; i < slot_bytes_; ++i)
        ASSERT_EQ(p[i], 0) << "slot byte " << i << " not zeroed on claim";
}

TEST_F(CommandDispatcherKdaState, ZeroOnReclaimRegression) {
    // PLAN.md GF3.8 verify clause: allocate → free → re-allocate → zeros
    // (INV-V4-DET obligation (2): pool reuse must never hand a sequence the
    // previous holder's residue).
    ASSERT_EQ(create_seq(1).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    // Simulate a stepped sequence: dirty its state slot.
    std::memset(slot_ptr(0), 0x5C, static_cast<size_t>(slot_bytes_));
    free_seq(1);
    EXPECT_EQ(used_kda(), 0);
    ASSERT_EQ(create_seq(2).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(used_kda(), 1);
    const char* p = slot_ptr(0);
    for (int64_t i = 0; i < slot_bytes_; ++i)
        ASSERT_EQ(p[i], 0) << "reclaimed slot byte " << i << " has residue";
    free_seq(2);
    EXPECT_EQ(used_kda(), 0);
}

TEST_F(CommandDispatcherKdaState, ForkCopiesStateFrozenIncluded) {
    ASSERT_EQ(create_seq(1).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    // Give the parent a recognizable state.
    for (int64_t i = 0; i < slot_bytes_; ++i)
        slot_ptr(0)[i] = static_cast<char>(i * 31 + 7);
    // LIVE fork: child owns slot 1 with the parent's bytes.
    ASSERT_EQ(fork_seq(1, 2).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(used_kda(), 2);
    EXPECT_EQ(std::memcmp(slot_ptr(0), slot_ptr(1),
                          static_cast<size_t>(slot_bytes_)), 0);
    // Divergence: the parent keeps stepping — the child must not follow
    // (this is WHY a frozen holder cannot refcount-share either).
    slot_ptr(0)[0] = static_cast<char>(~slot_ptr(0)[0]);
    EXPECT_NE(std::memcmp(slot_ptr(0), slot_ptr(1),
                          static_cast<size_t>(slot_bytes_)), 0);
    // FROZEN fork (prefix holder registration) also takes a full copy.
    ASSERT_EQ(fork_seq(1, 3, 0, /*frozen=*/true).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(used_kda(), 3);
    EXPECT_EQ(std::memcmp(slot_ptr(0), slot_ptr(2),
                          static_cast<size_t>(slot_bytes_)), 0);
    free_seq(3);
    free_seq(2);
    free_seq(1);
    EXPECT_EQ(used_kda(), 0);
}

TEST_F(CommandDispatcherKdaState, TruncatedForkRejected) {
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    auto c = fork_seq(1, 2, /*prefix_len=*/16);
    ASSERT_EQ(c.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(c.error.error_category,
              static_cast<uint32_t>(lipc::CmpErrorCategory::kSeqFork));
    EXPECT_EQ(used_kda(), 1);  // no leaked child slot
    free_seq(1);
}

TEST_F(CommandDispatcherKdaState, ExhaustionIsRetryableAndRollsBack) {
    // Hog every slot directly at the pool.
    std::vector<lmem::PageHandle> hogs;
    while (auto h = page_allocator_->allocate(0, lmem::Pool::kKdaState))
        hogs.push_back(*h);
    ASSERT_EQ(free_kda(), 0);
    const int kv_free_before =
        page_allocator_->free_pages(0, lmem::Pool::kMain);

    auto c = create_seq(9);
    ASSERT_EQ(c.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    // RETRYABLE category + "exhausted" in the message (INV-IPC-ERRMSG-80:
    // classification is by category; the message narrates).
    EXPECT_EQ(c.error.error_category,
              static_cast<uint32_t>(lipc::CmpErrorCategory::kKvPoolExhausted));
    EXPECT_NE(std::strstr(c.error.message, "exhausted"), nullptr)
        << c.error.message;
    // Rollback: the KV pages claimed before the state-slot refusal are
    // returned (never admitted-then-degraded, never leaked).
    EXPECT_EQ(page_allocator_->free_pages(0, lmem::Pool::kMain),
              kv_free_before);

    // Holder eviction frees a slot → the SAME create retries clean.
    page_allocator_->free(hogs.back());
    hogs.pop_back();
    ASSERT_EQ(create_seq(9).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    free_seq(9);
    for (auto& h : hogs) page_allocator_->free(h);
}

TEST_F(CommandDispatcherKdaState, DraftSequenceClaimsNoSlot) {
    // kSpeculation drafts carry NO recurrent state: the glm5_next MTP draft
    // layer is sparse MLA (MODELINFO section 5).
    auto c = create_seq(5, 8, /*pool=*/1);
    ASSERT_EQ(c.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(used_kda(), 0);
    free_seq(5);
}

// ── GF3.12: checkpoint format v3/v4 + hibernated-holder state spill ────────
//
// The KDA state section: a state-carrying sequence checkpoints as version
// 3 (replicated KV) / 4 (sharded) = the v1/v2 layout + SeqCkptKdaExt after
// the header + per-rank WHOLE-SLOT state blobs after the KV/indexer bodies.
// The versioning rule is honest BOTH ways (INV-KDA-STATE (f)): an old
// reader rejects v3/v4 outright; a new reader refuses v1/v2 onto a
// state-carrying sequence and v3/v4 onto a stateless one. The full
// snapshot->restore->decode bit-identity proof is GPU-level
// (glm53flash_gguf_golden_test, real engine); these CPU suites pin the
// refusal matrix, the restore state path, and the holder spill lifecycle.

TEST_F(CommandDispatcherKdaState, SnapshotRefusesMismatchedFrontier) {
    // seq_snapshot's precondition: every linear layer's frontier == the
    // claimed token_count — a never-stepped or mid-step sequence would
    // checkpoint a state that does not match its KV (a checkpoint that
    // lies; INV-KDA-REWIND anchor semantics).
    setenv("LS_SEQ_CKPT_PATH", "/tmp/gf312_kda_frontier_probe.ckpt", 1);
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    // Never stepped: kda_next_pos unsized -> refuse.
    auto snap = seq_ckpt_cmd(
        static_cast<uint32_t>(lipc::CMD_SEQ_SNAPSHOT), 1, 8);
    ASSERT_EQ(snap.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_NE(std::strstr(snap.error.message, "frontier"), nullptr)
        << snap.error.message;
    // Stepped to 4 but snapshotting 8: still a mismatch -> refuse.
    dispatcher_->kda_test_set_frontier(1, 4);
    snap = seq_ckpt_cmd(static_cast<uint32_t>(lipc::CMD_SEQ_SNAPSHOT), 1, 8);
    ASSERT_EQ(snap.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_NE(std::strstr(snap.error.message, "frontier"), nullptr)
        << snap.error.message;
    unsetenv("LS_SEQ_CKPT_PATH");
    free_seq(1);
}

TEST_F(CommandDispatcherKdaState, RestoreRefusesV1CkptOnStateCarryingSeq) {
    // The honest-versioning rule, reader side: a v1/v2 file has no state
    // section, and a recurrent-state sequence restored without its state
    // would resume on ZEROS — silently wrong everywhere (INV-KDA-REWIND).
    const std::string path = "/tmp/gf312_kda_v1_reject.ckpt";
    setenv("LS_SEQ_CKPT_PATH", path.c_str(), 1);
    write_ckpt(path, /*version=*/1, /*token_count=*/64, nullptr);
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    auto rest = seq_ckpt_cmd(
        static_cast<uint32_t>(lipc::CMD_SEQ_RESTORE), 1);
    ASSERT_EQ(rest.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_NE(std::strstr(rest.error.message, "no KDA state"), nullptr)
        << rest.error.message;
    unsetenv("LS_SEQ_CKPT_PATH");
    std::remove(path.c_str());
    free_seq(1);
}

TEST_F(CommandDispatcherKdaState, RestoreRefusesV3CkptOnStatelessSeq) {
    // ...and the other direction: a v3 file carries a state section, but a
    // slotless sequence (kSpeculation draft here) has nowhere to put it.
    const std::string path = "/tmp/gf312_kda_v3_reject.ckpt";
    setenv("LS_SEQ_CKPT_PATH", path.c_str(), 1);
    std::vector<char> blob(static_cast<size_t>(slot_bytes_), '\x11');
    write_ckpt(path, /*version=*/3, /*token_count=*/64, &blob);
    ASSERT_EQ(create_seq(7, 64, /*pool=*/1).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    ASSERT_EQ(dispatcher_->kda_seq_slots(7).first, 0);
    auto rest = seq_ckpt_cmd(
        static_cast<uint32_t>(lipc::CMD_SEQ_RESTORE), 7);
    ASSERT_EQ(rest.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_NE(std::strstr(rest.error.message, "no state slot"), nullptr)
        << rest.error.message;
    unsetenv("LS_SEQ_CKPT_PATH");
    std::remove(path.c_str());
    free_seq(7);
}

TEST_F(CommandDispatcherKdaState, RestoreUploadsStateAndSetsFrontier) {
    // v3 restore, state path: the whole-slot blob lands byte-exact in the
    // target's slot (NullDeviceBackend memcpys are REAL host ops) and the
    // per-layer INV-KDA-REWIND frontier is rebuilt at token_count.
    const std::string path = "/tmp/gf312_kda_v3_roundtrip.ckpt";
    setenv("LS_SEQ_CKPT_PATH", path.c_str(), 1);
    std::vector<char> blob(static_cast<size_t>(slot_bytes_));
    for (size_t i = 0; i < blob.size(); ++i)
        blob[i] = static_cast<char>(i * 131 + 5);
    write_ckpt(path, /*version=*/3, /*token_count=*/64, &blob);
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(dispatcher_->kda_seq_frontier(1), UINT32_MAX);
    auto rest = seq_ckpt_cmd(
        static_cast<uint32_t>(lipc::CMD_SEQ_RESTORE), 1);
    ASSERT_EQ(rest.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE))
        << rest.error.message;
    EXPECT_EQ(std::memcmp(slot_ptr(0), blob.data(), blob.size()), 0)
        << "restored slot bytes differ from the checkpoint blob";
    EXPECT_EQ(dispatcher_->kda_seq_frontier(1), 64u);
    // A snapshot from the restored sequence now passes the frontier gate
    // ... up to the KV read, which this fixture cannot serve (no kv base
    // pointers) — the refusal must be the KV one, never the KDA one.
    auto snap = seq_ckpt_cmd(
        static_cast<uint32_t>(lipc::CMD_SEQ_SNAPSHOT), 1, 64);
    ASSERT_EQ(snap.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_NE(std::strstr(snap.error.message, "kv state incomplete"),
              nullptr) << snap.error.message;
    unsetenv("LS_SEQ_CKPT_PATH");
    std::remove(path.c_str());
    free_seq(1);
}

TEST_F(CommandDispatcherKdaState, RestoreRefusesGeometryMismatch) {
    // SeqCkptKdaExt is an exact-match contract: a different slot size (TP
    // shard / layer geometry) cannot resume. Craft a v3 file whose ext
    // disagrees with the pool's slot_bytes.
    const std::string path = "/tmp/gf312_kda_v3_geom.ckpt";
    setenv("LS_SEQ_CKPT_PATH", path.c_str(), 1);
    {   // write_ckpt with a WRONG slot size in the ext
        CkptHeader h{};
        h.version = 3;
        h.token_count = 64;
        h.kv_layers = 6;
        h.kv_page_size = 64;
        h.kv_stride_block = static_cast<uint64_t>(kTestKvStride);
        h.logical_pages = 1;
        h.dcp_size = 1;
        CkptKdaExt e{};
        e.slot_bytes = static_cast<uint64_t>(slot_bytes_) + 256;
        e.ranks = 1;
        e.frontier = 64;
        std::FILE* f = std::fopen(path.c_str(), "wb");
        ASSERT_NE(f, nullptr);
        std::fwrite(&h, sizeof(h), 1, f);
        std::fwrite(&e, sizeof(e), 1, f);
        std::fclose(f);
    }
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    auto rest = seq_ckpt_cmd(
        static_cast<uint32_t>(lipc::CMD_SEQ_RESTORE), 1);
    ASSERT_EQ(rest.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_NE(std::strstr(rest.error.message, "geometry mismatch"), nullptr)
        << rest.error.message;
    unsetenv("LS_SEQ_CKPT_PATH");
    std::remove(path.c_str());
    free_seq(1);
}

TEST_F(CommandDispatcherKdaState, HibernateSpillsStateToHostAndFreesSlot) {
    // GF3.12 holder-cost decision: hibernation moves the FULL slot to host
    // RAM and returns the VRAM slot — holders stop competing with live
    // sequences for the pool that clamps concurrency (INV-PREFIX-CACHE-3
    // third cost class, host-resident form). Kill switch honored.
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    for (int64_t i = 0; i < slot_bytes_; ++i)
        slot_ptr(0)[i] = static_cast<char>(i * 37 + 3);
    std::vector<char> want(slot_ptr(0), slot_ptr(0) + slot_bytes_);
    dispatcher_->kda_test_set_frontier(1, 64);

    // Kill switch OFF path first: slot stays in VRAM.
    setenv("LS_KDA_HOLDER_SPILL", "0", 1);
    ASSERT_EQ(hibernate_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(used_kda(), 1);
    EXPECT_EQ(dispatcher_->kda_seq_slots(1), (std::pair<int, int>{1, 0}));

    // Default ON: spill to host, slot returned, frontier retained.
    unsetenv("LS_KDA_HOLDER_SPILL");
    ASSERT_EQ(hibernate_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(used_kda(), 0);
    EXPECT_EQ(dispatcher_->kda_seq_slots(1), (std::pair<int, int>{0, 1}));
    EXPECT_EQ(dispatcher_->kda_seq_frontier(1), 64u);
    // The freed slot may be reused by a new sequence (zeroed on claim) —
    // the holder's bytes must survive on host regardless.
    ASSERT_EQ(create_seq(2, 8).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(used_kda(), 1);
    free_seq(2);
    // Fork from the spilled holder: fresh child slot, H2D from the host
    // copy — byte-exact, frontier inherited; the holder stays spilled.
    ASSERT_EQ(fork_seq(1, 3).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(dispatcher_->kda_seq_slots(3), (std::pair<int, int>{1, 0}));
    EXPECT_EQ(dispatcher_->kda_seq_slots(1), (std::pair<int, int>{0, 1}));
    EXPECT_EQ(std::memcmp(slot_ptr(0), want.data(), want.size()), 0)
        << "fork-from-spilled child bytes differ from the holder's state";
    EXPECT_EQ(dispatcher_->kda_seq_frontier(3), 64u);
    // Frozen fork from the spilled holder works the same way.
    ASSERT_EQ(fork_seq(1, 4, 0, /*frozen=*/true).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(std::memcmp(slot_ptr(1), want.data(), want.size()), 0);
    // Truncating fork on a SPILLED holder: rejected exactly like on a
    // live slot (INV-SEQ-FORK-TRUNC belt-and-braces — the R4 exclusion
    // follows the state, not its residence).
    auto trunc = fork_seq(1, 5, /*prefix_len=*/16);
    ASSERT_EQ(trunc.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(trunc.error.error_category,
              static_cast<uint32_t>(lipc::CmpErrorCategory::kSeqFork));
    free_seq(4);
    free_seq(3);
    free_seq(1);  // releases the host spill bytes
    EXPECT_EQ(used_kda(), 0);
}

// ═══ P-29 step 24 (LS_KDA_PREFIX_CKPT): host-RAM KDA prefix checkpoints ════

TEST_F(CommandDispatcherKdaState, KdaCkptCaptureRefusals) {
    // The capture tripwire class (CMP_ERROR, kComputeValidation): unknown
    // sequence, bad grid position, never-stepped sequence, and a frontier
    // that is not uniformly AT the requested position. Every refusal must
    // be loud — the orchestrator's tripwire counter keys on CMP_ERROR.
    auto unknown = kda_ckpt_cmd(99, 64);
    ASSERT_EQ(unknown.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(unknown.error.error_category,
              static_cast<uint32_t>(
                  lipc::CmpErrorCategory::kComputeValidation));

    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    // Never stepped: kda_next_pos unsized — refused (no frontier to trust).
    auto fresh = kda_ckpt_cmd(1, 64);
    ASSERT_EQ(fresh.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));

    dispatcher_->kda_test_set_frontier(1, 64);
    // pos 0 and non-multiple-of-64 violate the INV-KDA-CARRY grid.
    ASSERT_EQ(kda_ckpt_cmd(1, 0).cmp_type,
              static_cast<uint32_t>(lipc::CMP_ERROR));
    ASSERT_EQ(kda_ckpt_cmd(1, 33).cmp_type,
              static_cast<uint32_t>(lipc::CMP_ERROR));
    // Frontier mismatch (a stale or mid-sweep capture point).
    ASSERT_EQ(kda_ckpt_cmd(1, 128).cmp_type,
              static_cast<uint32_t>(lipc::CMP_ERROR));
    // Nothing was captured by any refusal.
    EXPECT_EQ(dispatcher_->kda_ckpt_count(1), 0u);
    EXPECT_EQ(dispatcher_->kda_ckpt_host_bytes(), 0u);
    free_seq(1);
}

TEST_F(CommandDispatcherKdaState, KdaCkptCaptureStoresBlobAndCounts) {
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    for (int64_t i = 0; i < slot_bytes_; ++i)
        slot_ptr(0)[i] = static_cast<char>(i * 41 + 11);
    dispatcher_->kda_test_set_frontier(1, 64);
    auto cap = kda_ckpt_cmd(1, 64);
    ASSERT_EQ(cap.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cap.status, 0u);
    EXPECT_EQ(cap.compute.data_bytes, static_cast<uint32_t>(slot_bytes_))
        << "capture completion must report the host bytes consumed";
    EXPECT_EQ(dispatcher_->kda_ckpt_count(1), 1u);
    EXPECT_EQ(dispatcher_->kda_ckpt_host_bytes(),
              static_cast<size_t>(slot_bytes_));
    // Duplicate position: idempotent success, nothing re-captured.
    auto dup = kda_ckpt_cmd(1, 64);
    ASSERT_EQ(dup.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(dup.status, 0u);
    EXPECT_EQ(dup.compute.data_bytes, 0u);
    EXPECT_EQ(dispatcher_->kda_ckpt_count(1), 1u);
    EXPECT_EQ(dispatcher_->kda_ckpt_host_bytes(),
              static_cast<size_t>(slot_bytes_));
    // A second position accumulates.
    dispatcher_->kda_test_set_frontier(1, 128);
    ASSERT_EQ(kda_ckpt_cmd(1, 128).cmp_type,
              static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(dispatcher_->kda_ckpt_count(1), 2u);
    EXPECT_EQ(dispatcher_->kda_ckpt_host_bytes(),
              2u * static_cast<size_t>(slot_bytes_));
    // seq_free releases the blobs and settles the global byte counter.
    free_seq(1);
    EXPECT_EQ(dispatcher_->kda_ckpt_count(1), static_cast<size_t>(-1));
    EXPECT_EQ(dispatcher_->kda_ckpt_host_bytes(), 0u);
}

TEST_F(CommandDispatcherKdaState, TruncatingForkAdmittedAtCheckpointOnly) {
    // The GF3.12 gate rework: a truncating fork on a KDA sequence is
    // admitted IFF the source holds a checkpoint at EXACTLY prefix_len —
    // and the child gets the CHECKPOINT bytes at the CHECKPOINT frontier,
    // not the source's live state.
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    for (int64_t i = 0; i < slot_bytes_; ++i)
        slot_ptr(0)[i] = static_cast<char>(i * 31 + 7);   // pattern A
    std::vector<char> at64(slot_ptr(0), slot_ptr(0) + slot_bytes_);
    dispatcher_->kda_test_set_frontier(1, 64);
    ASSERT_EQ(kda_ckpt_cmd(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    // The source keeps stepping: live slot mutates to pattern B @128.
    for (int64_t i = 0; i < slot_bytes_; ++i)
        slot_ptr(0)[i] = static_cast<char>(i * 13 + 99);  // pattern B
    std::vector<char> at128(slot_ptr(0), slot_ptr(0) + slot_bytes_);
    dispatcher_->kda_test_set_frontier(1, 128);

    // NEGATIVE CONTROLS first (state must be unchanged by refusals):
    // no checkpoint at 128 or 16 => refused, kSeqFork, live state intact.
    for (uint32_t bad : {128u, 16u}) {
        auto r = fork_seq(1, 9, /*prefix_len=*/bad);
        ASSERT_EQ(r.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR))
            << "prefix_len " << bad << " must be refused (no checkpoint)";
        EXPECT_EQ(r.error.error_category,
                  static_cast<uint32_t>(lipc::CmpErrorCategory::kSeqFork));
    }

    // Admitted at the checkpoint: child = pattern A, frontier 64.
    ASSERT_EQ(fork_seq(1, 2, /*prefix_len=*/64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    ASSERT_EQ(dispatcher_->kda_seq_unit_ptrs(2).size(), 1u);
    EXPECT_EQ(std::memcmp(dispatcher_->kda_seq_unit_ptrs(2)[0],
                          at64.data(), at64.size()), 0)
        << "checkpoint-fork child must carry the CHECKPOINT bytes";
    EXPECT_EQ(dispatcher_->kda_seq_frontier(2), 64u);
    // The source keeps its blob: a second divergence forks again.
    ASSERT_EQ(fork_seq(1, 3, /*prefix_len=*/64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(dispatcher_->kda_ckpt_count(1), 1u);
    // A FULL fork still clones the live state at the live frontier.
    ASSERT_EQ(fork_seq(1, 4).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    ASSERT_EQ(dispatcher_->kda_seq_unit_ptrs(4).size(), 1u);
    EXPECT_EQ(std::memcmp(dispatcher_->kda_seq_unit_ptrs(4)[0],
                          at128.data(), at128.size()), 0)
        << "full fork must carry the LIVE bytes, not the checkpoint";
    EXPECT_EQ(dispatcher_->kda_seq_frontier(4), 128u);
    // Children were born without checkpoints of their own.
    EXPECT_EQ(dispatcher_->kda_ckpt_count(2), 0u);
    free_seq(4);
    free_seq(3);
    free_seq(2);
    free_seq(1);
    EXPECT_EQ(dispatcher_->kda_ckpt_host_bytes(), 0u);
}

TEST_F(CommandDispatcherKdaState, FrozenForkMovesCkptsToHolder) {
    // Registration semantics: checkpoints belong to the PREFIX ENTRY — a
    // FROZEN fork (the registration fork) moves the parent's whole map to
    // the holder child; the parent (freed at end of request) keeps none.
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    for (int64_t i = 0; i < slot_bytes_; ++i)
        slot_ptr(0)[i] = static_cast<char>(i * 31 + 7);   // pattern A @64
    std::vector<char> at64(slot_ptr(0), slot_ptr(0) + slot_bytes_);
    dispatcher_->kda_test_set_frontier(1, 64);
    ASSERT_EQ(kda_ckpt_cmd(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    // Prefill continues to the registration grid: pattern B @128.
    for (int64_t i = 0; i < slot_bytes_; ++i)
        slot_ptr(0)[i] = static_cast<char>(i * 13 + 99);  // pattern B @128
    std::vector<char> at128(slot_ptr(0), slot_ptr(0) + slot_bytes_);
    dispatcher_->kda_test_set_frontier(1, 128);

    ASSERT_EQ(fork_seq(1, 2, 0, /*frozen=*/true).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(dispatcher_->kda_ckpt_count(1), 0u)
        << "frozen registration fork must MOVE the checkpoints away";
    EXPECT_EQ(dispatcher_->kda_ckpt_count(2), 1u);
    EXPECT_EQ(dispatcher_->kda_ckpt_host_bytes(),
              static_cast<size_t>(slot_bytes_))
        << "ownership moved, bytes did not";
    // The parent can no longer serve the divergence...
    ASSERT_EQ(fork_seq(1, 8, /*prefix_len=*/64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_ERROR));
    // ...the holder can: truncating hit-child = pattern A @64.
    ASSERT_EQ(fork_seq(2, 3, /*prefix_len=*/64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    ASSERT_EQ(dispatcher_->kda_seq_unit_ptrs(3).size(), 1u);
    EXPECT_EQ(std::memcmp(dispatcher_->kda_seq_unit_ptrs(3)[0],
                          at64.data(), at64.size()), 0);
    EXPECT_EQ(dispatcher_->kda_seq_frontier(3), 64u);

    // Compose with HIBERNATION (the production holder shape): the spill
    // and the checkpoint are DIFFERENT host blobs at different positions.
    ASSERT_EQ(hibernate_seq(2, 128).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(dispatcher_->kda_seq_slots(2), (std::pair<int, int>{0, 1}));
    // Truncating hit on the HIBERNATED holder: checkpoint bytes @64.
    ASSERT_EQ(fork_seq(2, 4, /*prefix_len=*/64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    ASSERT_EQ(dispatcher_->kda_seq_unit_ptrs(4).size(), 1u);
    EXPECT_EQ(std::memcmp(dispatcher_->kda_seq_unit_ptrs(4)[0],
                          at64.data(), at64.size()), 0)
        << "hibernated-holder checkpoint fork must restore the @64 blob";
    EXPECT_EQ(dispatcher_->kda_seq_frontier(4), 64u);
    // Whole-node hit on the same holder: SPILL bytes @128 (unchanged
    // behavior — the two host sources must not cross).
    ASSERT_EQ(fork_seq(2, 5).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    ASSERT_EQ(dispatcher_->kda_seq_unit_ptrs(5).size(), 1u);
    EXPECT_EQ(std::memcmp(dispatcher_->kda_seq_unit_ptrs(5)[0],
                          at128.data(), at128.size()), 0)
        << "whole-node hit must restore the registered-length spill";
    EXPECT_EQ(dispatcher_->kda_seq_frontier(5), 128u);
    free_seq(5);
    free_seq(4);
    free_seq(3);
    free_seq(2);  // releases spill AND checkpoint bytes
    free_seq(1);
    EXPECT_EQ(dispatcher_->kda_ckpt_host_bytes(), 0u);
}

// ═══ TD-KDA-STATE-MAPPED-SLABS: mapped-state mode ═══════════════════════════
//
// The mapped mode backs the per-request state with per-(request, linear
// layer) CONTIGUOUS whole-slab runs over the shared kv_main slab region —
// no dedicated carve. The kernel contract survives verbatim (one base +
// uniform stride per launch; slot ids become run-start slab ids), and every
// host-visible byte (zero-on-claim, fork copy, spill blob, ckpt file) is
// IDENTICAL to the carve path — the mapping changes only WHERE bytes live.

namespace {

/// The small glm5_next shape WITH an indexer pool: slab geometry engages,
/// so the mapped KDA state has a shared region to claim from. Carve mode
/// boots on this config too — cross-mode tests use it for BOTH arms so the
/// state geometry (slot_bytes, per-layer layout) is identical.
lc::Config glm5n_small_slabbed_config(int num_gpus = 1) {
    auto cfg = glm5n_small_config(num_gpus);
    cfg.model.index_topk = 64;
    cfg.model.index_n_heads = 2;
    cfg.model.index_head_dim = 32;
    return cfg;
}

/// TD-KDA-MAPPED-NONTP-GPUS: THE SHAPE THE MAPPED DEFAULT COULD NOT BOOT.
/// Two GPUs, but hardware.tp_array = {0} — GPU 1 is EXPERT-ONLY: it hosts
/// no attention, no KV pool, no indexer share and no KDA state, so its
/// kMain is legitimately EMPTY. This is exactly what autoconfig derives
/// for glm5_next (one attention rank on a 5090, experts spread over every
/// GPU), and it is the COMPLEMENT of the tp=2 shape TD-KDA-MAPPED-MULTIGPU
/// covered — where both GPUs were TP members and therefore both slabbed.
lc::Config glm5n_small_nontp_config() {
    auto cfg = glm5n_small_slabbed_config(2);
    cfg.hardware.tp_array = {0};   // GPU 1 stays OUT of the TP set
    return cfg;
}

SmallAllocators make_mapped_kda_allocators() {
    SmallAllocators a;
    auto cfg = glm5n_small_slabbed_config();
    cfg._internal_kda_state.mapped = true;
    unsetenv("LS_KDA_STATE_MAPPED");
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    a.layout_copy = layout;
    for (size_t i = 0; i < layout.gpus.size(); ++i) {
        lc::GpuRef gref{static_cast<int>(i), static_cast<int>(i),
                        lc::GpuType::rtx5090};
        a.owned.push_back(lcomp::make_null_device_backend(gref));
        a.ptrs.push_back(a.owned.back().get());
    }
    a.vram = std::make_unique<lmem::VramAllocator>(std::move(layout), a.ptrs);
    a.pages = std::make_unique<lmem::PageAllocator>(*a.vram, a.ptrs[0]);
    return a;
}

}  // namespace

TEST(VramAllocatorKdaState, MappedBootsWithNonTpExpertOnlyGpus) {
    // TD-KDA-MAPPED-NONTP-GPUS (the ★★BLOCKER): with an expert-only GPU in
    // the world, the PageAllocator ctor threw
    //   "PageAllocator GPU 1: mapped KDA state requires slabbed kMain"
    // for that GPU — an UNSLABBED-MODEL guard firing on a NON-PARTICIPATING
    // GPU, which made every tp=1 + expert-host EP shape unbootable on the
    // 2026-08-31 mapped default (workaround: LS_KDA_STATE_MAPPED=0).
    // The sizing had always gated the mapped share on the participation
    // predicate; the ctor had not. Both now read the SAME published bit,
    // GpuVramLayout::attention_host.
    unsetenv("LS_KDA_STATE_MAPPED");
    auto cfg = glm5n_small_nontp_config();   // untouched ⇒ mapped default
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    ASSERT_TRUE(layout.kda.enabled);
    ASSERT_TRUE(layout.kda.mapped) << "mapped is the default since 2026-08-31";
    ASSERT_EQ(layout.gpus.size(), 2u);

    // The shared predicate, as published to every consumer.
    EXPECT_TRUE(layout.gpus[0].attention_host);
    EXPECT_FALSE(layout.gpus[1].attention_host);
    // ... and what it means: the expert-only GPU carries NO attention-side
    // tenant at all, while still getting the whole expert arena.
    EXPECT_GT(layout.gpus[0].kv_main_pages, 0);
    EXPECT_EQ(layout.gpus[1].kv_main_pages, 0);
    EXPECT_EQ(layout.gpus[1].kv_speculation_pages, 0);
    EXPECT_EQ(layout.gpus[1].kda_state_slots, 0);
    EXPECT_EQ(layout.gpus[1].kda_state_bytes, 0);
    EXPECT_EQ(layout.gpus[1].indexer_share_slabs, 0);
    EXPECT_EQ(layout.gpus[1].indexer_k_pages, 0);
    EXPECT_GT(layout.gpus[1].expert_total_bytes(), 0);

    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned;
    std::vector<lcomp::DeviceBackend*> ptrs;
    for (size_t i = 0; i < layout.gpus.size(); ++i) {
        lc::GpuRef gref{static_cast<int>(i), static_cast<int>(i),
                        lc::GpuType::rtx5090};
        owned.push_back(lcomp::make_null_device_backend(gref));
        ptrs.push_back(owned.back().get());
    }
    const int expect_units = layout.kda.num_layers;
    std::unique_ptr<lmem::VramAllocator> vram;
    ASSERT_NO_THROW(vram = std::make_unique<lmem::VramAllocator>(
                        std::move(layout), ptrs));
    // THE REGRESSION LINE: this ctor is what used to throw.
    std::unique_ptr<lmem::PageAllocator> pages;
    ASSERT_NO_THROW(pages = std::make_unique<lmem::PageAllocator>(
                        *vram, ptrs[0]));
    EXPECT_TRUE(pages->kda_state_mapped());
    EXPECT_EQ(pages->total_pages(1, lmem::Pool::kMain), 0);
    EXPECT_EQ(pages->total_pages(1, lmem::Pool::kKdaState), 0);

    // The attention rank serves a whole request's mapped state; the
    // expert-only GPU has nothing to claim from and refuses RETRYABLY
    // (empty vector — the admission seam), never aliasing another region.
    auto claim = pages->allocate_kda_state(0, /*seq_id=*/1);
    EXPECT_EQ(static_cast<int>(claim.size()), expect_units);
    EXPECT_TRUE(pages->allocate_kda_state(1, /*seq_id=*/2).empty());
    for (auto& h : claim) pages->free(h);
    EXPECT_EQ(pages->used_pages(0, lmem::Pool::kKdaState), 0);
}

TEST(VramAllocatorKdaState, MappedSkipsCarveAndGrowsSharedRegion) {
    unsetenv("LS_KDA_STATE_MAPPED");
    auto cfg_carve = glm5n_small_slabbed_config();
    cfg_carve._internal_kda_state.mapped = false;  // the surviving off-path
    auto cfg_mapped = glm5n_small_slabbed_config();
    cfg_mapped._internal_kda_state.mapped = true;
    // THE DEFAULT IS MAPPED (2026-08-31 promotion): an untouched config on
    // a slabbed model must land on the mapped path with no carve.
    {
        auto cfg_default = glm5n_small_slabbed_config();
        lmod::ModelConfig md(cfg_default);
        lmod::Fp8E4M3 f8;
        lmod::LayerRegistry rd(md, cfg_default, f8);
        auto ld = lmem::compute_vram_layout(cfg_default, rd, md);
        EXPECT_TRUE(ld.kda.mapped);
        EXPECT_EQ(ld.gpus[0].kda_state_slots, 0);
    }
    lmod::Fp8E4M3 fp8;
    lmod::ModelConfig mc(cfg_carve), mm(cfg_mapped);
    lmod::LayerRegistry rc(mc, cfg_carve, fp8), rm(mm, cfg_mapped, fp8);
    auto lc_ = lmem::compute_vram_layout(cfg_carve, rc, mc);
    auto lm_ = lmem::compute_vram_layout(cfg_mapped, rm, mm);
    ASSERT_TRUE(lc_.kda.enabled);
    EXPECT_FALSE(lc_.kda.mapped);
    ASSERT_TRUE(lm_.kda.enabled);
    EXPECT_TRUE(lm_.kda.mapped);
    // No carve under mapped; logical slot geometry untouched.
    EXPECT_GT(lc_.gpus[0].kda_state_slots, 0);
    EXPECT_EQ(lm_.gpus[0].kda_state_slots, 0);
    EXPECT_EQ(lm_.gpus[0].kda_state_bytes, 0);
    EXPECT_EQ(lm_.kda.slot_bytes, lc_.kda.slot_bytes);
    EXPECT_EQ(lm_.kda.per_layer_bytes, lc_.kda.per_layer_bytes);
    // The state's POLICY SHARE rides in kv_main as shared slab capacity
    // (the S4 indexer-share pattern): kv_main must GROW by at least the
    // per-request slab demand x policy slots — without this a demand-
    // capped-KV boot could admit ZERO sequences under mapped.
    const int64_t unit_slabs =
        (lm_.kda.per_layer_bytes + lm_.slab_bytes - 1) / lm_.slab_bytes;
    EXPECT_GE(lm_.gpus[0].kv_main_bytes - lc_.gpus[0].kv_main_bytes,
              static_cast<int64_t>(lm_.kda.num_layers) * unit_slabs
                  * lm_.slab_bytes);
    EXPECT_EQ(lm_.gpus[0].kv_speculation_bytes,
              lc_.gpus[0].kv_speculation_bytes);
    // Env overrides either way.
    setenv("LS_KDA_STATE_MAPPED", "1", 1);
    auto le = lmem::compute_vram_layout(cfg_carve, rc, mc);
    EXPECT_TRUE(le.kda.mapped);
    setenv("LS_KDA_STATE_MAPPED", "0", 1);
    auto le0 = lmem::compute_vram_layout(cfg_mapped, rm, mm);
    EXPECT_FALSE(le0.kda.mapped);
    unsetenv("LS_KDA_STATE_MAPPED");
}

TEST(VramAllocatorKdaState, MappedFallsBackToCarveOnUnslabbed) {
    // The mapped unit IS the slab — a glm5_next shape with no indexer pool
    // has no slab region. With mapped the DEFAULT, that shape must FALL
    // BACK to the dedicated carve (boot warning), never fail the boot: the
    // default cannot make previously-bootable models unbootable.
    unsetenv("LS_KDA_STATE_MAPPED");
    auto cfg = glm5n_small_config();  // index_topk == 0: unslabbed
    cfg._internal_kda_state.mapped = true;  // explicit AND default: same
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    EXPECT_FALSE(layout.kda.mapped);
    ASSERT_TRUE(layout.kda.enabled);
    EXPECT_GT(layout.gpus[0].kda_state_slots, 0);
    EXPECT_GT(layout.gpus[0].kda_state_bytes, 0);
}

TEST(PageAllocatorKdaMapped, ClaimShapeStrideAndSharedAccounting) {
    auto a = make_mapped_kda_allocators();
    auto& pa = *a.pages;
    ASSERT_TRUE(pa.kda_state_mapped());
    const auto& kl = pa.kda_layout();
    ASSERT_GT(pa.slab_bytes(), 0);
    EXPECT_EQ(pa.kda_units_per_rank(), kl.num_layers);
    EXPECT_EQ(pa.kda_unit_bytes(), kl.per_layer_bytes);
    EXPECT_EQ(pa.kda_state_stride_bytes(), pa.slab_bytes());
    EXPECT_EQ(pa.kda_state_slot_bytes(), kl.slot_bytes);  // logical size
    EXPECT_EQ(pa.kda_unit_slabs(),
              static_cast<int>((kl.per_layer_bytes + pa.slab_bytes() - 1)
                               / pa.slab_bytes()));
    // The base + slot * stride kernel contract over the SHARED region.
    EXPECT_EQ(pa.kda_state_base(0), pa.kv_main_base(0));

    const int free_kv_before = pa.free_pages(0, lmem::Pool::kMain);
    auto hs = pa.allocate_kda_state(0, 42);
    ASSERT_EQ(static_cast<int>(hs.size()), kl.num_layers);
    for (const auto& h : hs) {
        EXPECT_EQ(h.pool, lmem::Pool::kKdaState);
        ASSERT_NE(h.gpu_ptr, nullptr);
        EXPECT_EQ(h.gpu_ptr,
                  static_cast<char*>(pa.kda_state_base(0))
                      + static_cast<int64_t>(h.page_idx) * pa.slab_bytes());
        EXPECT_EQ(pa.meta(h).sequence_id, 42u);
        // 16-byte alignment for the kernels' float4 state pass.
        EXPECT_EQ(reinterpret_cast<uintptr_t>(h.gpu_ptr) % 16, 0u);
    }
    // Distinct, non-overlapping runs.
    for (size_t i = 0; i < hs.size(); ++i)
        for (size_t j = i + 1; j < hs.size(); ++j)
            EXPECT_GE(std::abs(hs[i].page_idx - hs[j].page_idx),
                      pa.kda_unit_slabs());
    // Whole slabs left the SHARED pool, and return whole on free.
    EXPECT_EQ(free_kv_before - pa.free_pages(0, lmem::Pool::kMain),
              kl.num_layers * pa.kda_unit_slabs() * pa.pages_per_slab());
    for (auto& h : hs) pa.free(h);
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), free_kv_before);
    EXPECT_EQ(pa.used_pages(0, lmem::Pool::kKdaState), 0);
}

TEST(PageAllocatorKdaMapped, ExhaustionIsAllOrNothingAndClaimsNothing) {
    auto a = make_mapped_kda_allocators();
    auto& pa = *a.pages;
    ASSERT_TRUE(pa.kda_state_mapped());
    const int free_before = pa.free_pages(0, lmem::Pool::kMain);
    std::vector<std::vector<lmem::PageHandle>> claims;
    for (;;) {
        auto hs = pa.allocate_kda_state(
            0, 100 + static_cast<uint64_t>(claims.size()));
        if (hs.empty()) break;
        ASSERT_EQ(static_cast<int>(hs.size()), pa.kda_layout().num_layers);
        claims.push_back(std::move(hs));
        ASSERT_LT(claims.size(), 100000u) << "runaway claim loop";
    }
    ASSERT_GT(claims.size(), 0u);
    // The FAILED claim must have claimed nothing.
    const int free_at_fail = pa.free_pages(0, lmem::Pool::kMain);
    auto again = pa.allocate_kda_state(0, 999);
    EXPECT_TRUE(again.empty());
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), free_at_fail);
    for (auto& hs : claims)
        for (auto& h : hs) pa.free(h);
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), free_before);
}

TEST(PageAllocatorKdaMapped, FreeSequenceSweepsMappedUnits) {
    auto a = make_mapped_kda_allocators();
    auto& pa = *a.pages;
    const int free_before = pa.free_pages(0, lmem::Pool::kMain);
    auto hs = pa.allocate_kda_state(0, 77);
    ASSERT_FALSE(hs.empty());
    EXPECT_EQ(pa.used_pages(0, lmem::Pool::kKdaState),
              pa.kda_layout().num_layers);
    pa.free_sequence(0, 77);
    EXPECT_EQ(pa.used_pages(0, lmem::Pool::kKdaState), 0);
    EXPECT_EQ(pa.free_pages(0, lmem::Pool::kMain), free_before);
}

TEST_F(CommandDispatcherKdaState, MappedCreateClaimsUnitsAndZeroesOnReclaim) {
    rebuild(glm5n_small_slabbed_config(), true);
    ASSERT_TRUE(page_allocator_->kda_state_mapped());
    const int L = page_allocator_->kda_layout().num_layers;
    const auto ub =
        static_cast<size_t>(page_allocator_->kda_unit_bytes());
    ASSERT_EQ(create_seq(1, 8).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(dispatcher_->kda_seq_slots(1).first, L);
    auto ptrs = dispatcher_->kda_seq_unit_ptrs(1);
    ASSERT_EQ(static_cast<int>(ptrs.size()), L);
    // Dirty every unit, free, re-create: the reclaim must ZERO (mapped
    // slabs may also carry stale KV bytes — INV-V4-DET obligation (2)).
    for (auto* p : ptrs) std::memset(p, 0xAB, ub);
    free_seq(1);
    ASSERT_EQ(create_seq(2, 8).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    auto ptrs2 = dispatcher_->kda_seq_unit_ptrs(2);
    ASSERT_EQ(static_cast<int>(ptrs2.size()), L);
    for (auto* p : ptrs2) {
        const char* c = static_cast<const char*>(p);
        for (size_t i = 0; i < ub; ++i)
            ASSERT_EQ(c[i], 0) << "unit byte " << i << " not zeroed";
    }
    free_seq(2);
}

TEST_F(CommandDispatcherKdaState, MappedForkCopiesEveryUnitAndDiverges) {
    rebuild(glm5n_small_slabbed_config(), true);
    const int L = page_allocator_->kda_layout().num_layers;
    const auto ub =
        static_cast<size_t>(page_allocator_->kda_unit_bytes());
    ASSERT_EQ(create_seq(1, 8).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    auto pp = dispatcher_->kda_seq_unit_ptrs(1);
    ASSERT_EQ(static_cast<int>(pp.size()), L);
    for (int u = 0; u < L; ++u)
        for (size_t i = 0; i < ub; ++i)
            static_cast<char*>(pp[u])[i] =
                static_cast<char>((i * 31 + u * 7 + 1) & 0xFF);
    ASSERT_EQ(fork_seq(1, 2, 0, /*frozen=*/true).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    auto cp = dispatcher_->kda_seq_unit_ptrs(2);
    ASSERT_EQ(static_cast<int>(cp.size()), L);
    for (int u = 0; u < L; ++u) {
        EXPECT_NE(cp[u], pp[u]);
        EXPECT_EQ(std::memcmp(cp[u], pp[u], ub), 0) << "unit " << u;
    }
    // Divergence: mutating the parent must not touch the frozen child.
    static_cast<char*>(pp[0])[0] = static_cast<char>(0x5A);
    EXPECT_NE(std::memcmp(cp[0], pp[0], ub), 0);
    free_seq(2);
    free_seq(1);
}

TEST_F(CommandDispatcherKdaState, MappedSpillGathersAndForkRestoresBitExact) {
    // The spill blob is the whole-slot layout in BOTH modes: mapped
    // hibernation GATHERS the per-layer units at their slot offsets, and a
    // fork from the spilled holder SCATTERS them back — the NullDevice
    // backend memcpys are real host ops, so this asserts gather/scatter
    // byte-exactness (the same blob shape the GF3.12 ckpt file carries).
    rebuild(glm5n_small_slabbed_config(), true);
    const int L = page_allocator_->kda_layout().num_layers;
    const auto ub =
        static_cast<size_t>(page_allocator_->kda_unit_bytes());
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    auto pp = dispatcher_->kda_seq_unit_ptrs(1);
    ASSERT_EQ(static_cast<int>(pp.size()), L);
    std::vector<std::vector<char>> want(static_cast<size_t>(L));
    for (int u = 0; u < L; ++u) {
        want[u].resize(ub);
        for (size_t i = 0; i < ub; ++i)
            want[u][i] = static_cast<char>((i * 131 + u * 17 + 5) & 0xFF);
        std::memcpy(pp[u], want[u].data(), ub);
    }
    dispatcher_->kda_test_set_frontier(1, 64);
    ASSERT_EQ(hibernate_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(dispatcher_->kda_seq_slots(1),
              (std::pair<int, int>{0, 1}));  // spilled: VRAM units freed
    EXPECT_EQ(used_kda(), 0);
    ASSERT_EQ(fork_seq(1, 2).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    auto cp = dispatcher_->kda_seq_unit_ptrs(2);
    ASSERT_EQ(static_cast<int>(cp.size()), L);
    for (int u = 0; u < L; ++u)
        EXPECT_EQ(std::memcmp(cp[u], want[u].data(), ub), 0)
            << "unit " << u << " lost bytes across spill+fork";
    EXPECT_EQ(dispatcher_->kda_seq_frontier(2), 64u);
    free_seq(2);
    free_seq(1);
}

TEST_F(CommandDispatcherKdaState, MappedRestoresCarveFormatCheckpoint) {
    // Cross-mode checkpoint portability: the v3 file format is mode-blind
    // (whole-slot blobs). A file written against the CARVE geometry must
    // restore onto a MAPPED sequence bit-exactly, unit by unit.
    rebuild(glm5n_small_slabbed_config(), true);
    const int L = page_allocator_->kda_layout().num_layers;
    const auto ub =
        static_cast<size_t>(page_allocator_->kda_unit_bytes());
    const std::string path = "/tmp/kda_mapped_xmode.ckpt";
    setenv("LS_SEQ_CKPT_PATH", path.c_str(), 1);
    std::vector<char> blob(static_cast<size_t>(slot_bytes_));
    for (size_t i = 0; i < blob.size(); ++i)
        blob[i] = static_cast<char>(i * 131 + 5);
    write_ckpt(path, /*version=*/3, /*token_count=*/64, &blob);
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    auto rest = seq_ckpt_cmd(
        static_cast<uint32_t>(lipc::CMD_SEQ_RESTORE), 1);
    ASSERT_EQ(rest.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE))
        << rest.error.message;
    auto ptrs = dispatcher_->kda_seq_unit_ptrs(1);
    ASSERT_EQ(static_cast<int>(ptrs.size()), L);
    for (int u = 0; u < L; ++u)
        EXPECT_EQ(std::memcmp(ptrs[u],
                              blob.data() + static_cast<size_t>(u) * ub,
                              ub), 0)
            << "restored unit " << u << " differs from the file blob";
    EXPECT_EQ(dispatcher_->kda_seq_frontier(1), 64u);
    unsetenv("LS_SEQ_CKPT_PATH");
    std::remove(path.c_str());
    free_seq(1);
}

TEST_F(CommandDispatcherKdaState, MappedExhaustionIsRetryableAndRollsBack) {
    rebuild(glm5n_small_slabbed_config(), true);
    // Drain the shared free-slab pool with mapped claims, then expect the
    // admission refusal to be the retryable kKvPoolExhausted category and
    // the failing create to leave no partial claim behind.
    std::vector<uint64_t> ids;
    uint64_t next = 10;
    for (;;) {
        auto c = create_seq(next, 8);
        if (c.cmp_type != static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE)) {
            EXPECT_EQ(c.error.error_category,
                      static_cast<uint32_t>(
                          lipc::CmpErrorCategory::kKvPoolExhausted));
            // The shared pool may exhaust on the KV side or the KDA
            // side first — either way the category is the retryable one
            // and "exhausted" is in the message.
            EXPECT_NE(std::strstr(c.error.message, "exhausted"), nullptr)
                << c.error.message;
            break;
        }
        ids.push_back(next++);
        ASSERT_LT(ids.size(), 100000u) << "runaway create loop";
    }
    ASSERT_GT(ids.size(), 0u);
    EXPECT_EQ(used_kda(),
              static_cast<int>(ids.size())
                  * page_allocator_->kda_layout().num_layers)
        << "a failed create must leave no partial mapped claim";
    // Retry after freeing one sequence succeeds (the retry seam).
    free_seq(ids.back());
    ids.pop_back();
    ASSERT_EQ(create_seq(9999, 8).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    free_seq(9999);
    for (auto id : ids) free_seq(id);
    EXPECT_EQ(used_kda(), 0);
}

// ── TD-KDA-MAPPED-FRAG (b2) + observability: segregation and the
//    byte-vs-contiguity discriminator ─────────────────────────────────────

namespace {

/// A shape whose per-layer unit spans MULTIPLE slabs (unit_slabs >= 2), so
/// contiguity can actually be broken. glm5n_small_slabbed has unit_slabs = 1
/// (any free slab is a run — fragmentation-immune), which is exactly why a
/// discriminator test needs this bigger-head variant: 128 heads x 32 dim →
/// per_layer 671,744 B over ~294,912 B slabs = 3 slabs/unit.
lc::Config glm5n_frag_config() {
    auto cfg = glm5n_small_slabbed_config();
    cfg.model.linear_attn_config->num_heads = 128;
    return cfg;
}

}  // namespace

TEST(PageAllocatorKdaMapped, SegregationTopDownAndFreedRunsLast) {
    // (b2) both levers: state claims land at the TOP of the region while
    // KV/indexer single-slab claims pop the boot LIFO from slab 0 up; and
    // a freed state run's slabs go to the free-list BOTTOM so the next
    // single-slab claim does NOT colonize the just-freed run.
    auto a = make_mapped_kda_allocators();
    auto& pa = *a.pages;
    ASSERT_TRUE(pa.kda_state_mapped());
    const int total = pa.total_pages(0, lmem::Pool::kMain)
                    / pa.pages_per_slab();
    ASSERT_GT(total, 0);
    const int units = pa.kda_units_per_rank();
    const int n = pa.kda_unit_slabs();

    // State claim: every unit run sits in the TOP units*n slabs.
    auto hs = pa.allocate_kda_state(0, 42);
    ASSERT_EQ(static_cast<int>(hs.size()), units);
    for (const auto& h : hs)
        EXPECT_GE(h.page_idx, total - units * n)
            << "state unit landed outside the top region";
    // Indexer claim: pops the boot LIFO — the BOTTOM of the region.
    auto ik1 = pa.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(ik1.has_value());
    EXPECT_EQ(ik1->page_idx, 0) << "indexer claim should start at slab 0";

    // Free the state claim; its slabs must go to the free-list BOTTOM:
    // the next single-slab claim takes boot-order slab 1, NOT a freed
    // state slab from the top region.
    std::vector<int> freed_ids;
    for (auto& h : hs) {
        for (int j = 0; j < n; ++j) freed_ids.push_back(h.page_idx + j);
        pa.free(h);
    }
    auto ik2 = pa.allocate(0, lmem::Pool::kIndexerK);
    ASSERT_TRUE(ik2.has_value());
    EXPECT_EQ(ik2->page_idx, 1)
        << "single-slab claim colonized a freed state run";
    for (int id : freed_ids) EXPECT_NE(ik2->page_idx, id);
    // Stats: one successful claim, no refusals.
    EXPECT_EQ(pa.kda_mapped_stats(0).claims, 1u);
    EXPECT_EQ(pa.kda_mapped_stats(0).refusals_capacity, 0u);
    EXPECT_EQ(pa.kda_mapped_stats(0).refusals_contiguity, 0u);
    pa.free(*ik1);
    pa.free(*ik2);
}

TEST(PageAllocatorKdaMapped, RefusalDiscriminatorContiguityVsCapacity) {
    // Craft both refusal classes on a unit_slabs >= 2 shape and assert the
    // TD-KDA-MAPPED-FRAG discriminator: capacity = not enough free slabs
    // (eviction helps); contiguity = enough slabs, too few intact runs
    // (eviction returns bytes, not adjacency).
    SmallAllocators a;
    {
        auto cfg = glm5n_frag_config();
        cfg._internal_kda_state.mapped = true;
        unsetenv("LS_KDA_STATE_MAPPED");
        lmod::ModelConfig mcfg(cfg);
        lmod::Fp8E4M3 fp8;
        lmod::LayerRegistry reg(mcfg, cfg, fp8);
        auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
        a.layout_copy = layout;
        for (size_t i = 0; i < layout.gpus.size(); ++i) {
            lc::GpuRef gref{static_cast<int>(i), static_cast<int>(i),
                            lc::GpuType::rtx5090};
            a.owned.push_back(lcomp::make_null_device_backend(gref));
            a.ptrs.push_back(a.owned.back().get());
        }
        a.vram = std::make_unique<lmem::VramAllocator>(std::move(layout),
                                                       a.ptrs);
        a.pages = std::make_unique<lmem::PageAllocator>(*a.vram, a.ptrs[0]);
    }
    auto& pa = *a.pages;
    ASSERT_TRUE(pa.kda_state_mapped());
    const int units = pa.kda_units_per_rank();
    const int n = pa.kda_unit_slabs();
    ASSERT_GE(n, 2) << "frag config must span multiple slabs per unit";

    // Fill: claim whole-request states until the pool refuses — that
    // refusal is CAPACITY class (free slabs below the whole demand).
    std::vector<std::vector<lmem::PageHandle>> claims;
    for (;;) {
        auto hs = pa.allocate_kda_state(
            0, 100 + static_cast<uint64_t>(claims.size()));
        if (hs.empty()) break;
        claims.push_back(std::move(hs));
        ASSERT_LT(claims.size(), 10000u);
    }
    ASSERT_GE(claims.size(), 2u)
        << "frag config too small to hold two requests — resize it";
    // Mop up the leftover free slabs (the fill stopped with < need free,
    // in a contiguous bottom run that could otherwise donate runs and
    // spoil the crafted shape) — after this, free slabs == 0 exactly.
    std::vector<lmem::PageHandle> mopped;
    {
        const int total = pa.total_pages(0, lmem::Pool::kMain)
                        / pa.pages_per_slab();
        for (int id = 0; id < total; ++id)
            if (auto m = pa.claim_one_slab_for_test(0, id))
                mopped.push_back(*m);
    }
    const auto& st0 = pa.kda_mapped_stats(0);
    EXPECT_EQ(st0.refusals_capacity, 1u);
    EXPECT_EQ(st0.refusals_contiguity, 0u);
    EXPECT_FALSE(st0.last_was_contiguity);

    // Free TWO requests (2*units runs), then break units+1 DISTINCT runs
    // by claiming one MIDDLE slab of each (the test hook: production
    // claims cannot target a slab). Free slabs = 2*units*n - (units+1)
    // >= units*n whenever units*n >= units+1 (n >= 2) — bytes suffice,
    // runs do not: the discriminator must say CONTIGUITY.
    std::vector<int> run_starts;
    for (int r = 0; r < 2; ++r) {
        for (auto& h : claims.back()) {
            run_starts.push_back(h.page_idx);
            pa.free(h);
        }
        claims.pop_back();
    }
    ASSERT_EQ(static_cast<int>(run_starts.size()), 2 * units);
    std::vector<lmem::PageHandle> breakers;
    for (int r = 0; r <= units; ++r) {
        auto b = pa.claim_one_slab_for_test(0, run_starts[r] + 1);
        ASSERT_TRUE(b.has_value()) << "breaker " << r;
        breakers.push_back(*b);
    }
    auto refused = pa.allocate_kda_state(0, 999);
    EXPECT_TRUE(refused.empty());
    const auto& st1 = pa.kda_mapped_stats(0);
    EXPECT_EQ(st1.refusals_contiguity, 1u);
    EXPECT_TRUE(st1.last_was_contiguity);
    EXPECT_EQ(st1.last_needed_runs, units);
    EXPECT_LT(st1.last_found_runs, units);
    EXPECT_GE(st1.last_free_slabs, units * n);

    // Remove the breakers: the same claim succeeds again (the shortage
    // was adjacency, not bytes — exactly what eviction cannot fix but
    // freeing the colonizers does).
    for (auto& b : breakers) pa.free(b);
    auto ok = pa.allocate_kda_state(0, 1000);
    EXPECT_EQ(static_cast<int>(ok.size()), units);
    for (auto& h : ok) pa.free(h);
    for (auto& b : mopped) pa.free(b);
    for (auto& hs : claims)
        for (auto& h : hs) pa.free(h);
}

// ═══ TD-KDA-MAPPED-MULTIGPU: the MULTI-RANK mapped legs ═════════════════════
//
// Everything above this line runs on ONE GPU. Mapped KDA state claims
// per-(request, linear layer) contiguous whole-slab runs on EVERY TP rank,
// ALL-OR-NOTHING, and keeps the handles RANK-MAJOR ([rank][unit], every
// downstream consumer indexing [rank * units + unit]) — and until this
// section no test had ever executed CommandDispatcher::claim_kda_state's
// loop with more than one entry in dcp_config().tp_gpu_indices, so
// multi-GPU mapped correctness was inherited by ARGUMENT ("per-rank
// independent claims"), not verified.
//
// The world: two null-backend GPUs, hardware.tp_array {0,1} (so the KDA
// layout shards heads per rank), and the ENGINE's DcpConfig on the
// PageAllocator — tp_gpu_indices {0,1} is the switch claim_kda_state reads.
// KV rides the production default (replicated, INV-KV-REP), so the shared
// slab pool is exercised by BOTH tenants on BOTH ranks, exactly as it is on
// hardware.
//
// The five legs: rank-major claim shape, the all-or-nothing rollback when
// rank 1 alone refuses, per-rank fork copies, per-rank spill + fork-restore,
// and per-rank checkpoint blob order.

TEST_F(CommandDispatcherKdaState, MappedTp2CreateClaimsBothRanksRankMajor) {
    rebuild(glm5n_small_slabbed_config(2), true, 2);
    ASSERT_EQ(num_gpus_, 2);
    ASSERT_TRUE(page_allocator_->kda_state_mapped());
    // The switch under test: BOTH GPUs are in the claim loop's TP set.
    ASSERT_EQ(page_allocator_->dcp_config().tp_gpu_indices,
              (std::vector<int>{0, 1}));
    const int L = page_allocator_->kda_layout().num_layers;
    const int n = page_allocator_->kda_unit_slabs();
    const int pps = page_allocator_->pages_per_slab();
    ASSERT_GT(L, 0);
    ASSERT_GT(n, 0);
    ASSERT_GT(pps, 0);
    const int fp0 = free_kv_pages(0), fp1 = free_kv_pages(1);
    const int fs0 = free_kv_slabs(0), fs1 = free_kv_slabs(1);

    ASSERT_EQ(create_seq(1, 8).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));

    // Shape: num_layers unit runs per rank, 2 * num_layers handles total.
    EXPECT_EQ(dispatcher_->kda_seq_slots(1).first, 2 * L);
    EXPECT_EQ(used_kda_on(0), L);
    EXPECT_EQ(used_kda_on(1), L);
    // Order: RANK-MAJOR — every rank-0 unit precedes every rank-1 unit.
    auto gpus = dispatcher_->kda_seq_unit_gpus(1);
    ASSERT_EQ(static_cast<int>(gpus.size()), 2 * L);
    for (int u = 0; u < L; ++u) {
        EXPECT_EQ(gpus[static_cast<size_t>(u)], 0)
            << "handle " << u << " should be a rank-0 unit";
        EXPECT_EQ(gpus[static_cast<size_t>(L + u)], 1)
            << "handle " << (L + u) << " should be a rank-1 unit";
    }
    // Per-rank slot indices are INDEPENDENT, but no two units — on the same
    // rank or across ranks — may alias.
    auto ptrs = dispatcher_->kda_seq_unit_ptrs(1);
    ASSERT_EQ(static_cast<int>(ptrs.size()), 2 * L);
    for (size_t i = 0; i < ptrs.size(); ++i) {
        ASSERT_NE(ptrs[i], nullptr) << "unit " << i << " has no pointer";
        for (size_t j = i + 1; j < ptrs.size(); ++j)
            EXPECT_NE(ptrs[i], ptrs[j]) << "units " << i << " and " << j
                                        << " alias";
    }
    // Shared-pool accounting, EXACT and per rank: the state took
    // num_layers x unit_slabs WHOLE slabs (pages_per_slab pages each) and
    // the replicated KV took one page per (logical page, kMain layer) on
    // EVERY rank — prompt_len 8 < page 16 is exactly one logical page.
    // (Pages, not slabs, carry the exact assertion: the KV pages of one
    // sequence pack into its own run, whose slab count depends on
    // pages_per_slab, while the page count is pure arithmetic.)
    const int kv_pages = kmain_layers();
    EXPECT_EQ(fp0 - free_kv_pages(0), L * n * pps + kv_pages);
    EXPECT_EQ(fp1 - free_kv_pages(1), L * n * pps + kv_pages);
    // ... and the slab cost is the SAME on both ranks and covers the runs.
    EXPECT_EQ(fs0 - free_kv_slabs(0), fs1 - free_kv_slabs(1));
    EXPECT_GE(fs0 - free_kv_slabs(0), L * n);

    // Release returns every rank's slabs whole.
    free_seq(1);
    EXPECT_EQ(used_kda_on(0), 0);
    EXPECT_EQ(used_kda_on(1), 0);
    EXPECT_EQ(free_kv_pages(0), fp0);
    EXPECT_EQ(free_kv_pages(1), fp1);
    EXPECT_EQ(free_kv_slabs(0), fs0);
    EXPECT_EQ(free_kv_slabs(1), fs1);
}

TEST_F(CommandDispatcherKdaState, MappedTp2Rank1FailRollsBackRank0) {
    // THE load-bearing multi-rank leg: the claim is ALL-OR-NOTHING ACROSS
    // RANKS. Rank 0 succeeds, rank 1 refuses — rank 0's units must come
    // back (a leak here would burn num_layers whole slabs of GPU 0 per
    // refused admission, i.e. an unbounded capacity leak on the retry seam
    // the orchestrator is designed to hammer), the category must stay the
    // retryable one, and the retry must actually work once the pressure is
    // gone (TD-KDA-MAPPED-RETRY-EFFECTIVE).
    rebuild(glm5n_small_slabbed_config(2), true, 2);
    ASSERT_EQ(num_gpus_, 2);
    ASSERT_TRUE(page_allocator_->kda_state_mapped());
    const int L = page_allocator_->kda_layout().num_layers;
    const int n = page_allocator_->kda_unit_slabs();
    const int need_slabs = L * n;
    ASSERT_GE(need_slabs, 3)
        << "the squeeze must leave room for the sequence's KV slab";

    // Pressure on RANK 1 ONLY: leave it one slab short of the state demand
    // (still enough for the KV claim, which runs first and must succeed —
    // otherwise the refusal under test would be a kSeqCreate KV refusal).
    auto pressure = squeeze_slabs_to(1, need_slabs - 1);
    ASSERT_EQ(free_kv_slabs(1), need_slabs - 1);
    const int fp0 = free_kv_pages(0);
    const int fs0 = free_kv_slabs(0);
    ASSERT_GT(fs0, need_slabs) << "rank 0 must have plenty";

    auto c = create_seq(42, 8);
    ASSERT_EQ(c.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    // (a) retryable category + the per-GPU message naming the REFUSING
    // rank (INV-IPC-ERRMSG-80: classification by category, message
    // narrates; the 80-byte field truncates the tail, so only the head of
    // the message is asserted).
    EXPECT_EQ(c.error.error_category,
              static_cast<uint32_t>(lipc::CmpErrorCategory::kKvPoolExhausted));
    EXPECT_NE(std::strstr(c.error.message, "exhausted KDA state gpu 1"),
              nullptr) << c.error.message;
    // (b) NO LEAK on rank 0: the all-or-nothing loop freed the units it had
    // already claimed there, and seq_create rolled back the KV pages.
    EXPECT_EQ(used_kda_on(0), 0) << "rank-0 units leaked by the rollback";
    EXPECT_EQ(used_kda_on(1), 0);
    EXPECT_EQ(free_kv_pages(0), fp0);
    EXPECT_EQ(free_kv_slabs(0), fs0);
    // The refused sequence does not exist at all.
    EXPECT_EQ(dispatcher_->kda_seq_slots(42), (std::pair<int, int>{0, 0}));

    // (c) RETRYABLE IN FACT: release rank 1's pressure, re-issue the SAME
    // create — it must go through, on both ranks.
    for (auto& h : pressure) page_allocator_->free(h);
    ASSERT_EQ(create_seq(42, 8).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(dispatcher_->kda_seq_slots(42).first, 2 * L);
    EXPECT_EQ(used_kda_on(0), L);
    EXPECT_EQ(used_kda_on(1), L);
    free_seq(42);
    EXPECT_EQ(used_kda_on(0), 0);
    EXPECT_EQ(used_kda_on(1), 0);
    EXPECT_EQ(free_kv_pages(0), fp0);
}

TEST_F(CommandDispatcherKdaState, MappedTp2ForkCopiesEveryUnitOnBothRanks) {
    // Copy-on-fork is per UNIT per RANK: each rank's D2D runs on its own
    // backend/stream. A rank-blind copy (only rank 0 cloned, or rank 1's
    // child pointed at rank 0's bytes) would hand the child a HALF-STALE
    // state — invisible at tp=1. The patterns below encode the RANK, so a
    // cross-rank copy fails the memcmp instead of passing by symmetry.
    rebuild(glm5n_small_slabbed_config(2), true, 2);
    ASSERT_EQ(num_gpus_, 2);
    const int L = page_allocator_->kda_layout().num_layers;
    const auto ub = static_cast<size_t>(page_allocator_->kda_unit_bytes());
    ASSERT_GT(ub, 0u);
    ASSERT_EQ(create_seq(1, 8).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    auto pp = dispatcher_->kda_seq_unit_ptrs(1);
    ASSERT_EQ(static_cast<int>(pp.size()), 2 * L);
    for (int r = 0; r < 2; ++r)
        for (int u = 0; u < L; ++u) {
            char* p = static_cast<char*>(pp[static_cast<size_t>(r * L + u)]);
            for (size_t i = 0; i < ub; ++i)
                p[i] = static_cast<char>((i * 31 + u * 7 + r * 101 + 1)
                                         & 0xFF);
        }

    // Frozen fork (prefix-holder registration) — a full copy, both ranks.
    ASSERT_EQ(fork_seq(1, 2, 0, /*frozen=*/true).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(dispatcher_->kda_seq_slots(2).first, 2 * L);
    EXPECT_EQ(used_kda_on(0), 2 * L);
    EXPECT_EQ(used_kda_on(1), 2 * L);
    auto cg = dispatcher_->kda_seq_unit_gpus(2);
    ASSERT_EQ(static_cast<int>(cg.size()), 2 * L);
    for (int u = 0; u < L; ++u) {
        EXPECT_EQ(cg[static_cast<size_t>(u)], 0);
        EXPECT_EQ(cg[static_cast<size_t>(L + u)], 1);
    }
    auto cp = dispatcher_->kda_seq_unit_ptrs(2);
    ASSERT_EQ(static_cast<int>(cp.size()), 2 * L);
    for (int r = 0; r < 2; ++r)
        for (int u = 0; u < L; ++u) {
            const size_t k = static_cast<size_t>(r * L + u);
            EXPECT_NE(cp[k], pp[k]) << "rank " << r << " unit " << u
                                    << " shares the parent's storage";
            EXPECT_EQ(std::memcmp(cp[k], pp[k], ub), 0)
                << "rank " << r << " unit " << u << " was not copied";
        }
    // Divergence, checked ON RANK 1: a parent step must not follow into the
    // child there — and must not disturb rank 0's copy either.
    static_cast<char*>(pp[static_cast<size_t>(L)])[0] =
        static_cast<char>(0x5A);
    EXPECT_NE(std::memcmp(cp[static_cast<size_t>(L)],
                          pp[static_cast<size_t>(L)], ub), 0)
        << "rank-1 child followed the parent's write";
    EXPECT_EQ(std::memcmp(cp[0], pp[0], ub), 0)
        << "a rank-1 parent write reached rank 0's units";
    free_seq(2);
    free_seq(1);
    EXPECT_EQ(used_kda_on(0), 0);
    EXPECT_EQ(used_kda_on(1), 0);
}

TEST_F(CommandDispatcherKdaState, MappedTp2SpillGathersBothRanksAndForkRestores) {
    // Hibernation gathers EVERY rank's units into its OWN whole-slot host
    // blob and returns that rank's VRAM slabs; a later fork claims fresh
    // units on each rank and H2D-scatters rank r's blob onto rank r. The
    // rank axis is the whole risk: one blob for two ranks, or a scatter
    // that ignores the source GPU, corrupts half the state.
    rebuild(glm5n_small_slabbed_config(2), true, 2);
    ASSERT_EQ(num_gpus_, 2);
    const int L = page_allocator_->kda_layout().num_layers;
    const int n = page_allocator_->kda_unit_slabs();
    const int pps = page_allocator_->pages_per_slab();
    const auto ub = static_cast<size_t>(page_allocator_->kda_unit_bytes());
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    auto pp = dispatcher_->kda_seq_unit_ptrs(1);
    ASSERT_EQ(static_cast<int>(pp.size()), 2 * L);
    std::vector<std::vector<char>> want(static_cast<size_t>(2 * L));
    for (int r = 0; r < 2; ++r)
        for (int u = 0; u < L; ++u) {
            const size_t k = static_cast<size_t>(r * L + u);
            want[k].resize(ub);
            for (size_t i = 0; i < ub; ++i)
                want[k][i] = static_cast<char>((i * 131 + u * 17 + r * 53
                                                + 5) & 0xFF);
            std::memcpy(pp[k], want[k].data(), ub);
        }
    dispatcher_->kda_test_set_frontier(1, 64);
    const int fp0 = free_kv_pages(0), fp1 = free_kv_pages(1);

    ASSERT_EQ(hibernate_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    // BOTH ranks freed, and the host blob covers BOTH ranks (one whole-slot
    // buffer per rank — {VRAM handles, spilled rank buffers}).
    EXPECT_EQ(dispatcher_->kda_seq_slots(1), (std::pair<int, int>{0, 2}));
    EXPECT_EQ(used_kda_on(0), 0);
    EXPECT_EQ(used_kda_on(1), 0);
    EXPECT_EQ(free_kv_pages(0) - fp0, L * n * pps);
    EXPECT_EQ(free_kv_pages(1) - fp1, L * n * pps);

    // Fork from the spilled holder: fresh units on each rank, bit-exact per
    // rank, frontier inherited; the holder stays spilled.
    ASSERT_EQ(fork_seq(1, 2).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(dispatcher_->kda_seq_slots(1), (std::pair<int, int>{0, 2}));
    EXPECT_EQ(dispatcher_->kda_seq_slots(2).first, 2 * L);
    EXPECT_EQ(used_kda_on(0), L);
    EXPECT_EQ(used_kda_on(1), L);
    auto cg = dispatcher_->kda_seq_unit_gpus(2);
    ASSERT_EQ(static_cast<int>(cg.size()), 2 * L);
    auto cp = dispatcher_->kda_seq_unit_ptrs(2);
    ASSERT_EQ(static_cast<int>(cp.size()), 2 * L);
    for (int r = 0; r < 2; ++r)
        for (int u = 0; u < L; ++u) {
            const size_t k = static_cast<size_t>(r * L + u);
            EXPECT_EQ(cg[k], r) << "restored unit " << k << " on wrong rank";
            EXPECT_EQ(std::memcmp(cp[k], want[k].data(), ub), 0)
                << "rank " << r << " unit " << u
                << " lost bytes across spill + fork";
        }
    EXPECT_EQ(dispatcher_->kda_seq_frontier(2), 64u);
    free_seq(2);
    free_seq(1);  // releases both host spill buffers
    EXPECT_EQ(used_kda_on(0), 0);
    EXPECT_EQ(used_kda_on(1), 0);
}

TEST_F(CommandDispatcherKdaState, MappedTp2CkptBlobsRankMajor) {
    // The v3/v4 state section is PER-RANK whole-slot blobs in rank order,
    // and SeqCkptKdaExt::ranks is an exact-match restore requirement. At
    // tp = 2 that means: ranks == 2 or refuse, and blob r scatters onto
    // rank r's units. The two blobs here are DIFFERENT, so a rank swap (or
    // a rank-blind broadcast of blob 0) fails the compare instead of
    // passing by symmetry.
    //
    // NOTE (checked, not assumed): this fixture cannot take a REAL
    // seq_snapshot — handle_seq_snapshot needs kv_cache_base_ptrs_, which
    // CommandDispatcher only populates when a DcpExecutor is wired, so a
    // snapshot here always refuses with "kv state incomplete" before the
    // state section is reached (the same limitation the tp=1 checkpoint
    // tests document). The WRITE side of the per-rank blob order is
    // therefore proven on hardware (TP=2 CheckpointRestoreBitExact-
    // Continuation); the READ side is pinned here.
    rebuild(glm5n_small_slabbed_config(2), true, 2);
    ASSERT_EQ(num_gpus_, 2);
    const int L = page_allocator_->kda_layout().num_layers;
    const auto ub = static_cast<size_t>(page_allocator_->kda_unit_bytes());
    const auto slot = static_cast<size_t>(slot_bytes_);
    // The units tile the slot from offset 0 (any alignment tail past
    // num_layers * per_layer_bytes is deterministic zero, never read).
    ASSERT_LE(static_cast<size_t>(L) * ub, slot);
    const std::string path = "/tmp/kda_mapped_tp2_ckpt.ckpt";
    setenv("LS_SEQ_CKPT_PATH", path.c_str(), 1);

    // Two DISTINCT whole-slot blobs, concatenated in rank order.
    std::vector<char> blobs(2 * slot);
    for (size_t i = 0; i < slot; ++i) {
        blobs[i] = static_cast<char>((i * 131 + 5) & 0xFF);
        blobs[slot + i] = static_cast<char>((i * 197 + 61) & 0xFF);
    }
    ASSERT_NE(std::memcmp(blobs.data(), blobs.data() + slot, slot), 0);

    // A ranks == 1 file must be REFUSED on a tp = 2 sequence: the ext is an
    // exact-match contract (a 1-rank file restored onto 2 ranks would leave
    // rank 1 on its zeroed claim — silently wrong on half the heads).
    ASSERT_EQ(create_seq(1, 64).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    ASSERT_EQ(dispatcher_->kda_seq_slots(1).first, 2 * L);
    {
        std::vector<char> one(blobs.begin(), blobs.begin() + slot);
        write_ckpt(path, /*version=*/3, /*token_count=*/64, &one,
                   /*ranks=*/1);
        auto bad = seq_ckpt_cmd(
            static_cast<uint32_t>(lipc::CMD_SEQ_RESTORE), 1);
        ASSERT_EQ(bad.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
        EXPECT_NE(std::strstr(bad.error.message, "geometry mismatch"),
                  nullptr) << bad.error.message;
    }

    // The real round trip: ranks == 2, blob r onto rank r, unit by unit.
    write_ckpt(path, /*version=*/3, /*token_count=*/64, &blobs, /*ranks=*/2);
    auto rest = seq_ckpt_cmd(
        static_cast<uint32_t>(lipc::CMD_SEQ_RESTORE), 1);
    ASSERT_EQ(rest.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE))
        << rest.error.message;
    auto gpus = dispatcher_->kda_seq_unit_gpus(1);
    auto ptrs = dispatcher_->kda_seq_unit_ptrs(1);
    ASSERT_EQ(static_cast<int>(ptrs.size()), 2 * L);
    for (int r = 0; r < 2; ++r)
        for (int u = 0; u < L; ++u) {
            const size_t k = static_cast<size_t>(r * L + u);
            EXPECT_EQ(gpus[k], r);
            EXPECT_EQ(std::memcmp(ptrs[k],
                                  blobs.data()
                                      + static_cast<size_t>(r) * slot
                                      + static_cast<size_t>(u) * ub,
                                  ub), 0)
                << "rank " << r << " unit " << u
                << " differs from the file's rank-" << r << " blob";
        }
    EXPECT_EQ(dispatcher_->kda_seq_frontier(1), 64u);
    unsetenv("LS_SEQ_CKPT_PATH");
    std::remove(path.c_str());
    free_seq(1);
}

// ═══ TD-KDA-MAPPED-NONTP-GPUS: the EXPERT-ONLY-HOST shape ═══════════════════
//
// The complement of the section above. Everything there had EVERY GPU in the
// TP set; this leg has GPUs that are NOT — hardware.tp_array = {0} with two
// GPUs, i.e. one attention rank plus an expert-only host, the shape autoconfig
// derives for glm5_next EP (one 5090 attends, experts spread over every card
// and every PCIe link). An expert-only GPU has NO KV pool, so its kMain is
// legitimately EMPTY, and the mapped-state precondition ("requires slabbed
// kMain") must not fire there — it is a guard against UNSLABBED MODELS, not
// against non-participating GPUs. That divergence between the sizing side
// (which always gated on participation) and the PageAllocator ctor (which did
// not) is the whole bug; both now read GpuVramLayout::attention_host.
//
// The dispatcher half matters independently of the ctor half: at tp == 1 the
// engine never installs a DcpConfig, so claim_kda_state runs on the
// create-command GPU alone — the expert-only GPU must simply stay out of the
// lifecycle entirely.

TEST_F(CommandDispatcherKdaState, MappedNonTpExpertHostRunsTheLifecycleOnRank0) {
    rebuild(glm5n_small_nontp_config(), /*mapped=*/true, /*num_gpus=*/2,
            /*tp_degree=*/1);
    ASSERT_EQ(num_gpus_, 2);
    ASSERT_TRUE(page_allocator_->kda_state_mapped());
    // tp == 1 ⇒ no DcpConfig (engine.cpp installs one only at tp >= 2), so
    // the claim loop degenerates to the create-command GPU.
    ASSERT_TRUE(page_allocator_->dcp_config().tp_gpu_indices.empty());
    // The expert-only GPU really is empty — the condition the old guard
    // mistook for an unslabbed model.
    ASSERT_EQ(page_allocator_->total_pages(1, lmem::Pool::kMain), 0);
    ASSERT_EQ(page_allocator_->total_pages(1, lmem::Pool::kKdaState), 0);
    ASSERT_GT(page_allocator_->total_pages(0, lmem::Pool::kMain), 0);

    const int L = page_allocator_->kda_layout().num_layers;
    const int n = page_allocator_->kda_unit_slabs();
    const int pps = page_allocator_->pages_per_slab();
    ASSERT_GT(L, 0);
    ASSERT_GT(n, 0);
    ASSERT_GT(pps, 0);
    const int fp0 = free_kv_pages(0);
    const int fs0 = free_kv_slabs(0);

    ASSERT_EQ(create_seq(1, 8).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    // One rank's worth of units — NOT two: the expert-only GPU is not a
    // state host and must never be claimed on.
    EXPECT_EQ(dispatcher_->kda_seq_slots(1).first, L);
    EXPECT_EQ(used_kda_on(0), L);
    EXPECT_EQ(used_kda_on(1), 0);
    auto gpus = dispatcher_->kda_seq_unit_gpus(1);
    ASSERT_EQ(static_cast<int>(gpus.size()), L);
    for (int u = 0; u < L; ++u)
        EXPECT_EQ(gpus[static_cast<size_t>(u)], 0)
            << "unit " << u << " must live on the attention rank";
    // Exact shared-pool accounting on the attention rank: the state took
    // L x n whole slabs and the KV claim one page per (logical page, kMain
    // layer) — prompt_len 8 < page 16 is exactly one logical page.
    EXPECT_EQ(fp0 - free_kv_pages(0), L * n * pps + kmain_layers());
    EXPECT_GE(fs0 - free_kv_slabs(0), L * n);
    EXPECT_EQ(free_kv_pages(1), 0);

    // Fork and free stay rank-local too.
    ASSERT_EQ(fork_seq(1, 2).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(used_kda_on(0), 2 * L);
    EXPECT_EQ(used_kda_on(1), 0);
    free_seq(2);
    free_seq(1);
    EXPECT_EQ(used_kda_on(0), 0);
    EXPECT_EQ(free_kv_pages(0), fp0);
    EXPECT_EQ(free_kv_slabs(0), fs0);
}


// ── TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA: the admissible-context ceiling ────

TEST(VramAllocatorKdaState, AdmissibleCeilingComputedWithAllThreeTenants) {
    // The kMain pool serves THREE tenants at admission (KV pages, the
    // full-length indexer-K reservation, the mapped KDA state runs), so the
    // boot must publish the single-request ceiling over their SUM — the GF3
    // 1M arm advertised max_sequence_length=500000 while the pool could not
    // hold one such request, and the first sign was an admission 500 at
    // 425k tokens of prefill.
    unsetenv("LS_KDA_STATE_MAPPED");
    auto cfg = glm5n_small_slabbed_config();   // mapped default ON
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    ASSERT_TRUE(layout.kda.enabled);
    ASSERT_TRUE(layout.kda.mapped);
    const auto& gl = layout.gpus[0];
    ASSERT_GT(gl.kv_main_pages, 0);
    ASSERT_GT(layout.pages_per_slab, 0);

    // Components at max_sequence_length, recomputed from first principles.
    const int page = cfg.memory.kv_cache.page_size_tokens;
    const int idx_page = cfg.memory.kv_cache.indexer_k_page_size_tokens;
    const int64_t max_seq = cfg.serving.max_sequence_length;
    const int64_t unit_slabs =
        (layout.kda.per_layer_bytes + layout.slab_bytes - 1)
        / layout.slab_bytes;
    const int64_t state_pages = static_cast<int64_t>(layout.kda.num_layers)
                                * unit_slabs * layout.pages_per_slab;
    EXPECT_EQ(gl.admission_state_pages, state_pages);
    // glm5n_small_slabbed: 2 DSA layers are both the KV-bearing and the
    // indexer-computing set (no nextn).
    const int64_t kv_pages = ((max_seq + page - 1) / page) * 2;
    EXPECT_EQ(gl.admission_kv_pages, kv_pages);
    const int64_t idx_pages = ((max_seq + idx_page - 1) / idx_page) * 2
                              * layout.pages_per_slab;
    EXPECT_EQ(gl.admission_indexer_pages, idx_pages);
    EXPECT_EQ(gl.admission_demand_pages, kv_pages + idx_pages + state_pages);

    // The ceiling is exact: demand(ceiling) fits, demand(ceiling + 1) does
    // not (all three terms are monotone step functions of T).
    ASSERT_GT(gl.admissible_ctx_tokens, 0);
    auto demand = [&](int64_t T) {
        return ((T + page - 1) / page) * 2
               + ((T + idx_page - 1) / idx_page) * 2 * layout.pages_per_slab
               + state_pages;
    };
    EXPECT_LE(demand(gl.admissible_ctx_tokens), gl.kv_main_pages);
    EXPECT_GT(demand(gl.admissible_ctx_tokens + 1), gl.kv_main_pages);
}

TEST(VramAllocatorKdaState, AdmissibleCeilingFlagsAnInadmissibleMaxSeq) {
    // A VRAM-clamped pool smaller than ONE max-length request must be
    // called out at boot (warn + ceiling), not discovered at request time.
    unsetenv("LS_KDA_STATE_MAPPED");
    auto cfg = glm5n_small_slabbed_config();
    cfg.serving.max_sequence_length = 10'000'000;  // 1 GB GPU cannot hold it
    // Keep the toy boot out of two unrelated 1-GB edge conditions so the
    // layout computes instead of throwing: an explicit (small) page pool
    // and a modest scratch keep the expert-cache minimum fed.
    cfg.memory.kv_cache.max_pages_per_gpu = 20000;
    cfg.memory.kv_cache.prefill_scratch_preallocated_gb = 0.01;
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    const auto& gl = layout.gpus[0];
    ASSERT_GT(gl.kv_main_pages, 0);
    ASSERT_GT(gl.admissible_ctx_tokens, 0);
    EXPECT_LT(gl.admissible_ctx_tokens, cfg.serving.max_sequence_length);
    EXPECT_GT(gl.admission_demand_pages, gl.kv_main_pages);
    // Expert-only GPUs never compute a ceiling.
    auto cfg2 = glm5n_small_nontp_config();
    lmod::ModelConfig mcfg2(cfg2);
    lmod::LayerRegistry reg2(mcfg2, cfg2, fp8);
    auto layout2 = lmem::compute_vram_layout(cfg2, reg2, mcfg2);
    EXPECT_GT(layout2.gpus[0].admissible_ctx_tokens, 0);
    EXPECT_EQ(layout2.gpus[1].admissible_ctx_tokens, 0);
}

TEST_F(CommandDispatcherKdaState, SelfInflictedOverCapacityFailsFastNonRetryable) {
    // TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA defect (2): a request whose OWN
    // whole-life demand exceeds the pool's TOTAL capacity was routed to the
    // retryable evict-a-holder remedy, which cannot help — no eviction
    // frees the request's own KV. It must refuse FAST (before claiming
    // anything), NON-RETRYABLE (kSeqCreate, and the message must avoid the
    // word "exhausted" — is_pool_exhaustion() substring-matches it), and
    // quote the admissible ceiling.
    // A 10M-token serving window on a 1 GB GPU: the pool is a fraction of
    // one max-length request (the GF3 1M-arm shape, scaled down).
    auto cfg = glm5n_small_slabbed_config();
    cfg.serving.max_sequence_length = 10'000'000;
    cfg.memory.kv_cache.max_pages_per_gpu = 20000;
    cfg.memory.kv_cache.prefill_scratch_preallocated_gb = 0.01;
    rebuild(std::move(cfg), true);
    ASSERT_TRUE(page_allocator_->kda_state_mapped());
    ASSERT_GT(page_allocator_->admissible_ctx_tokens(), 0);
    const int cap = page_allocator_->total_pages(0, lmem::Pool::kMain);
    ASSERT_GT(cap, 0);
    const int free_before = page_allocator_->free_pages(0, lmem::Pool::kMain);

    // A 10M-token prompt's own KV alone dwarfs the whole pool.
    auto c = create_seq(77, 10'000'000u);
    ASSERT_EQ(c.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(c.error.error_category,
              static_cast<uint32_t>(lipc::CmpErrorCategory::kSeqCreate))
        << "self-inflicted shortage must NOT be the retryable "
           "kKvPoolExhausted class";
    EXPECT_EQ(std::strstr(c.error.message, "exhausted"), nullptr)
        << "message must not re-arm the substring retry seam: "
        << c.error.message;
    EXPECT_NE(std::strstr(c.error.message, "over kMain pool capacity"),
              nullptr) << c.error.message;
    EXPECT_NE(std::strstr(c.error.message, "admissible"), nullptr)
        << "the refusal must carry the ceiling: " << c.error.message;

    // FAST: nothing was claimed, nothing to roll back.
    EXPECT_EQ(page_allocator_->free_pages(0, lmem::Pool::kMain), free_before);
    EXPECT_EQ(used_kda(), 0);
    // The dispatcher is undamaged: a servable request still admits.
    ASSERT_EQ(create_seq(78, 8).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    free_seq(78);
}

TEST_F(CommandDispatcherKdaState,
       BulkKvExhaustionIsRetryableAndReportsShortfall) {
    // TD-KVXP-BOOT-OVERGRANT-FIRST-ADMISSION (P-30 step 3). The seq_create
    // BULK-KV allocation refusal — a WITHIN-capacity request against a
    // currently-drained pool — is the seam a 97k prefill rides while the
    // 44z rebalancer holds the pool's slabs as expert-zone grants. It used
    // to carry kSeqCreate with NO pool-pressure poke: the orchestrator's
    // bounded large-prefill wait then waited on an eager drain nothing had
    // armed (189 refusals, zero pokes, permanent 503). It must (a) carry
    // the retryable kKvPoolExhausted CATEGORY (classification survives the
    // 80-byte truncation, INV-IPC-ERRMSG-80), (b) fire the pool-pressure
    // callback through write_error's choke point, (c) report the slab
    // SHORTFALL so the eager drain is demand-aware, and (d) roll back
    // cleanly. NEGATIVE CONTROL: on the old code (a) reads kSeqCreate and
    // (b)/(c) never fire.
    rebuild(glm5n_small_slabbed_config(), true);
    ASSERT_TRUE(page_allocator_->kda_state_mapped());
    const int pps = page_allocator_->pages_per_slab();
    ASSERT_GT(pps, 0);
    const int L = kmain_layers();
    ASSERT_GT(L, 0);

    int pokes = 0;
    int poke_gpu = -1;
    int64_t poke_shortfall = -1;
    dispatcher_->set_pool_pressure_callback(
        [&](int g, int64_t sf) { ++pokes; poke_gpu = g; poke_shortfall = sf; });

    // CALIBRATE the max-length prompt's actual upfront KV demand (the
    // dispatcher clamps num_pages at max_blocks_per_seq, whose page
    // granularity differs from the claim loop's — measure, don't
    // replicate): admit one from a comfortable pool and subtract the
    // known mapped-state footprint.
    const auto max_seq = static_cast<int>(cfg_->serving.max_sequence_length);
    const int state_pages = page_allocator_->kda_layout().num_layers
                            * page_allocator_->kda_unit_slabs() * pps;
    const int fp_calib = free_kv_pages(0);
    ASSERT_EQ(create_seq(70, static_cast<uint32_t>(max_seq)).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    const int need_kv_pages = fp_calib - free_kv_pages(0) - state_pages;
    free_seq(70);
    ASSERT_GT(need_kv_pages, pps)
        << "fixture must demand more than one slab of KV, or the squeeze "
           "cannot leave the claim short";
    ASSERT_EQ(free_kv_pages(0), fp_calib) << "calibration must roll back";

    // Squeeze free BELOW the KV demand — but the demand stays far under
    // the pool's TOTAL capacity, so the non-retryable over-capacity fast
    // check (SelfInflictedOverCapacity...) stays out of the way.
    const int target_free = (need_kv_pages - 1) / pps;  // free < KV demand
    auto pressure = squeeze_slabs_to(0, target_free);
    ASSERT_EQ(free_kv_slabs(0), target_free);
    const int free_pages = free_kv_pages(0);
    ASSERT_LT(free_pages, need_kv_pages)
        << "the squeeze must leave the KV claim short";

    auto c = create_seq(77, static_cast<uint32_t>(max_seq));
    ASSERT_EQ(c.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(c.error.error_category,
              static_cast<uint32_t>(lipc::CmpErrorCategory::kKvPoolExhausted))
        << "a within-capacity bulk-KV shortage is the RETRYABLE pool class";
    EXPECT_NE(std::strstr(c.error.message, "pool exhausted"), nullptr)
        << "the substring retry seam must keep matching: " << c.error.message;
    EXPECT_EQ(pokes, 1)
        << "write_error's kKvPoolExhausted choke point must poke the "
           "rebalancer (the old per-site form missed this path entirely)";
    EXPECT_EQ(poke_gpu, 0);
    const int64_t want_shortfall =
        (static_cast<int64_t>(need_kv_pages) - free_pages + pps - 1) / pps;
    EXPECT_EQ(poke_shortfall, want_shortfall)
        << "the refusal must report its slab shortfall for the "
           "demand-aware eager drain";
    EXPECT_GE(poke_shortfall, 1);

    // Clean rollback: every allocated page returned, no sequence exists.
    EXPECT_EQ(free_kv_pages(0), free_pages);
    EXPECT_EQ(used_kda_on(0), 0);

    // A KDA-STATE refusal still pokes through the same choke point (the
    // per-site call it replaced): free the KV pressure down to just under
    // one state claim.
    for (auto& h : pressure) page_allocator_->free(h);
    const int state_slabs = page_allocator_->kda_layout().num_layers
                            * page_allocator_->kda_unit_slabs();
    auto pressure2 = squeeze_slabs_to(0, state_slabs - 1);
    pokes = 0;
    poke_shortfall = -1;
    auto c2 = create_seq(79, 8);
    ASSERT_EQ(c2.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(c2.error.error_category,
              static_cast<uint32_t>(lipc::CmpErrorCategory::kKvPoolExhausted));
    EXPECT_EQ(pokes, 1) << "the KDA-state refusal lost its poke in the "
                           "choke-point move";
    EXPECT_EQ(poke_shortfall, 0)
        << "sites that cannot size their shortage report 0 (unknown)";
    for (auto& h : pressure2) page_allocator_->free(h);
}

// ── TD-GLM5-KDA-SLOTS-EXPORT: kda_state_export reduction (pure struct math) ──
// The boot-metadata view the orchestrator's admission reads
// (EngineInfo.kda_state_*): min across attention-host GPUs, expert-only
// hosts excluded, mapped geometry vs dedicated carve.

TEST(KdaStateExportGeometry, NonKdaModelExportsAllZero) {
    lmem::VramLayout layout{};
    layout.gpus.push_back(lmem::GpuVramLayout{});
    const auto ks = lmem::kda_state_export(layout);
    EXPECT_FALSE(ks.mapped);
    EXPECT_EQ(ks.slots, 0);
    EXPECT_EQ(ks.slot_bytes, 0);
    EXPECT_EQ(ks.pages_per_seq, 0);
    EXPECT_EQ(ks.pool_pages, 0);
}

TEST(KdaStateExportGeometry, CarveTakesMinAcrossAttentionHostsOnly) {
    lmem::VramLayout layout{};
    layout.kda.enabled = true;
    layout.kda.mapped = false;
    layout.kda.slot_bytes = 146 << 20;
    lmem::GpuVramLayout a{};
    a.attention_host = true;
    a.kda_state_slots = 10;
    lmem::GpuVramLayout b{};
    b.attention_host = true;
    b.kda_state_slots = 7;   // the binding rank
    lmem::GpuVramLayout e{};
    e.attention_host = false;  // expert-only host: must not dilute the min
    e.kda_state_slots = 3;
    layout.gpus = {a, b, e};
    const auto ks = lmem::kda_state_export(layout);
    EXPECT_FALSE(ks.mapped);
    EXPECT_EQ(ks.slots, 7);
    EXPECT_EQ(ks.slot_bytes, 146 << 20);
    // carve mode draws nothing from kMain
    EXPECT_EQ(ks.pages_per_seq, 0);
    EXPECT_EQ(ks.pool_pages, 0);
}

TEST(KdaStateExportGeometry, MappedExportsPoolMinAndAdmissionDemand) {
    lmem::VramLayout layout{};
    layout.kda.enabled = true;
    layout.kda.mapped = true;
    layout.kda.slot_bytes = 73 << 20;
    lmem::GpuVramLayout a{};
    a.attention_host = true;
    a.kv_main_pages = 1000;
    a.admission_state_pages = 146;
    lmem::GpuVramLayout b{};
    b.attention_host = true;
    b.kv_main_pages = 800;   // the binding pool
    b.admission_state_pages = 146;
    lmem::GpuVramLayout e{};
    e.attention_host = false;
    e.kv_main_pages = 50;    // expert-only host: excluded
    layout.gpus = {a, b, e};
    const auto ks = lmem::kda_state_export(layout);
    EXPECT_TRUE(ks.mapped);
    EXPECT_EQ(ks.slots, 0);  // mapped: capacity is pool geometry, not slots
    EXPECT_EQ(ks.pool_pages, 800);
    EXPECT_EQ(ks.pages_per_seq, 146);
}

TEST(KdaStateExportGeometry, MappedFallsBackToLayoutGeometry) {
    // admissible-ctx pass skipped (admission_state_pages == 0): the demand
    // comes from the slab arithmetic — ceil(per_layer/slab) slabs per layer.
    lmem::VramLayout layout{};
    layout.kda.enabled = true;
    layout.kda.mapped = true;
    layout.kda.num_layers = 34;
    layout.kda.per_layer_bytes = 2500;
    layout.kda.slot_bytes = 34 * 2500;
    layout.slab_bytes = 1024;      // -> 3 slabs per layer unit
    layout.pages_per_slab = 2;
    lmem::GpuVramLayout a{};
    a.attention_host = true;
    a.kv_main_pages = 4096;
    layout.gpus = {a};
    const auto ks = lmem::kda_state_export(layout);
    EXPECT_TRUE(ks.mapped);
    EXPECT_EQ(ks.pool_pages, 4096);
    EXPECT_EQ(ks.pages_per_seq, 34 * 3 * 2);
}
