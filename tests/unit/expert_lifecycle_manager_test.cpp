// Unit tests for ExpertLifecycleManager (ELM-2).
//
// All tests run without CUDA by using null/heap backends.
// The null transfer backend completes transfers on the next poll().

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "config/config_parser.h"
#include "core/gpu_ref.h"
#include "core/null_device_backend.h"
#include "core/memory/expert_cache.h"
#include "core/memory/numa_manager.h"
#include "core/memory/nvme_tier.h"
#include "core/memory/vram_allocator.h"
#include "core/transfer/transfer_engine.h"

#include <filesystem>
#include "daemon/expert_lifecycle_manager.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/fp8.h"
#include "model/weight_loader/weight_loader.h"

namespace lmem  = layerstorm::memory;
namespace ltr   = layerstorm::transfer;
namespace lmod  = layerstorm::model;
namespace lc    = layerstorm::config;
namespace ldam  = layerstorm::daemon;
namespace lcomp = layerstorm::compute;

// ── Helpers ─────────────────────────────────────────────────────────────────

static lc::Config small_config() {
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

/// Backing storage for fake model data (must outlive LoadedModel).
static std::vector<std::vector<std::byte>> g_fake_data;

/// Build a fake LoadedModel with data for (layer, expert).
/// Each expert has 3 bundles (gate/up/down) with proper components so that
/// ensure_expert_packed can re-pack after release.  packed_slot is pre-set
/// on bundles[0] so initial resolve_host_source skips packing.
/// Caller must ensure g_fake_data outlives the returned model.
static lmod::LoadedModel fake_model(int num_layers, int num_experts,
                                     int64_t expert_bytes) {
    g_fake_data.clear();
    // 3 projection buffers + 1 packed_slot buffer per expert.
    g_fake_data.reserve(num_layers * num_experts * 4);

    // Split expert_bytes roughly into 3 projections.
    // FP8: weight bytes only, no scales needed for fake data.
    int64_t proj_bytes = expert_bytes / 3;
    int64_t last_proj_bytes = expert_bytes - 2 * proj_bytes;

    lmod::LoadedModel model;
    model.layers.resize(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        model.layers[l].layer_idx = l;
        model.layers[l].routed_experts.resize(num_experts);
        for (int e = 0; e < num_experts; ++e) {
            auto make_bundle = [&](lmod::TensorComponent comp, int64_t bytes) {
                g_fake_data.emplace_back(bytes, std::byte{0xAB});
                auto& data = g_fake_data.back();
                lmod::RawTensor rt{
                    .data = std::span<const std::byte>(data),
                    .dtype = lmod::SafetensorsDtype::F8_E4M3,
                    .shape = {bytes}};
                lmod::WeightBundle wb;
                wb.id.component = comp;
                wb.weight = rt;
                return wb;
            };

            auto gate = make_bundle(lmod::TensorComponent::gate_proj, proj_bytes);
            auto up   = make_bundle(lmod::TensorComponent::up_proj,   proj_bytes);
            auto down  = make_bundle(lmod::TensorComponent::down_proj, last_proj_bytes);

            // Pre-set packed_slot on bundles[0] (gate) to simulate already-packed state.
            g_fake_data.emplace_back(expert_bytes, std::byte{0xAB});
            auto& packed = g_fake_data.back();
            gate.packed_slot = std::span<const std::byte>(packed);

            model.layers[l].routed_experts[e].push_back(std::move(gate));
            model.layers[l].routed_experts[e].push_back(std::move(up));
            model.layers[l].routed_experts[e].push_back(std::move(down));
        }
    }
    return model;
}

// ── Test fixture ────────────────────────────────────────────────────────────

class ExpertLifecycleManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        cfg_ = std::make_unique<lc::Config>(small_config());
        mcfg_ = std::make_unique<lmod::ModelConfig>(*cfg_);
        fp8_ = std::make_unique<lmod::Fp8E4M3>();
        layer_reg_ = std::make_unique<lmod::LayerRegistry>(*mcfg_, *cfg_, *fp8_);
        expert_bytes_ = layer_reg_->per_routed_expert_bytes();

        lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
        lc::GpuRef gpu1{1, 1, lc::GpuType::rtx5090};
        null_backends_.push_back(lcomp::make_null_device_backend(gpu0));
        null_backends_.push_back(lcomp::make_null_device_backend(gpu1));
        std::vector<lcomp::DeviceBackend*> backends{
            null_backends_[0].get(), null_backends_[1].get()};

        auto layout = lmem::compute_vram_layout(*cfg_, *layer_reg_, *mcfg_);
        vram_ = std::make_unique<lmem::VramAllocator>(
            std::move(layout), backends);
        cache_ = std::make_unique<lmem::ExpertCache>(
            *vram_, *cfg_, expert_bytes_);
        ltr::TransferEngine::Options te_opts{
            .device_backends = backends,
            .pcie_info = {{.pcie_gen = 5, .pcie_width = 16},
                          {.pcie_gen = 5, .pcie_width = 16}},
            .min_dispatch_per_gpu = 8,
        };
        transfer_engine_ = std::make_unique<ltr::TransferEngine>(std::move(te_opts));

        model_ = fake_model(6, 8, expert_bytes_);
    }

    /// Poll ELM with pre-polled subsystem completions (ELM-5 API).
    ldam::PollResult poll_elm(ldam::ExpertLifecycleManager& elm) {
        auto xfer = transfer_engine_->poll_completions();
        // No NvmeTier in default fixture.
        std::vector<lmem::IoCompletion> nvme;
        return elm.poll(xfer, nvme);
    }

    /// Create ELM with default deps (mmap source, no NVMe).
    std::unique_ptr<ldam::ExpertLifecycleManager> make_elm() {
        ldam::ExpertLifecycleManager::Deps deps{
            .expert_cache    = cache_.get(),
            .nvme_tier       = nullptr,
            .transfer_engine = transfer_engine_.get(),
            .loaded_model    = &model_,
        };
        return std::make_unique<ldam::ExpertLifecycleManager>(std::move(deps));
    }

    /// Create ELM with snapshot attached (ELM-8 tests).
    std::unique_ptr<ldam::ExpertLifecycleManager> make_elm_with_snapshot(
            layerstorm::ipc::StateSnapshot& snap) {
        ldam::ExpertLifecycleManager::Deps deps{
            .expert_cache    = cache_.get(),
            .nvme_tier       = nullptr,
            .transfer_engine = transfer_engine_.get(),
            .loaded_model    = &model_,
            .snapshot        = &snap,
            .first_moe_layer = 1,   // matches small_config first_k_dense_replace
            .num_moe_layers  = 5,   // layers 1..5 are MoE (6 total - 1 dense)
            .num_experts     = 8,
            .num_gpus        = 2,
        };
        return std::make_unique<ldam::ExpertLifecycleManager>(std::move(deps));
    }

    std::unique_ptr<lc::Config>          cfg_;
    std::unique_ptr<lmod::ModelConfig>    mcfg_;
    std::unique_ptr<lmod::Fp8E4M3>       fp8_;
    std::unique_ptr<lmod::LayerRegistry>  layer_reg_;
    int64_t                              expert_bytes_ = 0;
    // null_backends_ must precede vram_ (VramAllocator holds raw ptrs to them)
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> null_backends_;
    std::unique_ptr<lmem::VramAllocator> vram_;
    std::unique_ptr<lmem::ExpertCache>   cache_;
    std::unique_ptr<ltr::TransferEngine> transfer_engine_;
    lmod::LoadedModel                    model_;
};

// ═════════════════════════════════════════════════════════════════════════════
// State transitions

TEST_F(ExpertLifecycleManagerTest, EnsureResident_WarmToHot) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 0};

    auto token = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    EXPECT_NE(token, 0u);

    // State should be RESERVED or TRANSFERRING after ensure_resident.
    auto st = elm->state(key, 0);
    EXPECT_EQ(st.host_tier, ldam::HostTier::kWarm);  // mmap source
    EXPECT_EQ(st.interest_count, 1u);

    // Poll to complete the transfer (null backend completes immediately).
    auto completions = poll_elm(*elm).lifecycle;
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_EQ(completions[0].token, token);
    EXPECT_EQ(completions[0].cmd_seq, 100u);
    EXPECT_TRUE(completions[0].success);
    EXPECT_EQ(completions[0].key, key);
    EXPECT_EQ(completions[0].gpu_idx, 0);

    // Now HOT.
    st = elm->state(key, 0);
    EXPECT_EQ(st.gpu_tier, ldam::GpuTier::kHot);
    EXPECT_EQ(st.interest_count, 0u);
}

TEST_F(ExpertLifecycleManagerTest, EnsureResident_AlreadyHot) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 0};

    // Load it first.
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    poll_elm(*elm);

    // Now it's HOT — second ensure_resident should be immediate.
    auto token2 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 200);
    auto completions = poll_elm(*elm).lifecycle;
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_EQ(completions[0].token, token2);
    EXPECT_EQ(completions[0].cmd_seq, 200u);
    EXPECT_TRUE(completions[0].success);
}

TEST_F(ExpertLifecycleManagerTest, EnsureResident_AlreadyTransferring) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 2};

    auto token1 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    // Before polling, issue second ensure_resident.
    auto token2 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 200);

    EXPECT_NE(token1, token2);

    auto st = elm->state(key, 0);
    EXPECT_EQ(st.interest_count, 2u);

    // Poll — both should complete.
    auto completions = poll_elm(*elm).lifecycle;
    EXPECT_GE(completions.size(), 2u);

    bool found1 = false, found2 = false;
    for (const auto& c : completions) {
        if (c.token == token1) { found1 = true; EXPECT_TRUE(c.success); }
        if (c.token == token2) { found2 = true; EXPECT_TRUE(c.success); }
    }
    EXPECT_TRUE(found1);
    EXPECT_TRUE(found2);
}

// ELB demand-join: a higher-priority interest joining an entry whose H2D is
// still STAGED at speculative-prefetch priority re-asserts the priority in the
// transfer engine's staged admission queue (demand must never ride a
// prefetch's low priority).
TEST_F(ExpertLifecycleManagerTest, EnsureResident_DemandJoinReassertsPriority) {
    // Staging-forced engine: min_dispatch 0 (everything stages) + single-slot
    // admission so the flush order is observable.
    std::vector<lcomp::DeviceBackend*> backends{null_backends_[0].get(),
                                                null_backends_[1].get()};
    ltr::TransferEngine::Options te_opts{
        .device_backends = backends,
        .pcie_info = {{.pcie_gen = 5, .pcie_width = 16},
                      {.pcie_gen = 5, .pcie_width = 16}},
        .min_dispatch_per_gpu = 0,
        .max_inflight_per_gpu = 1,
    };
    transfer_engine_ = std::make_unique<ltr::TransferEngine>(std::move(te_opts));
    auto elm = make_elm();

    // Speculative prefetch through the ELM — staged at -1.0 (the fixture's
    // stable zone is a single slot, so the competing transfer below is a raw
    // engine enqueue: same GPU, same staged admission queue, no slot needed).
    lmem::ExpertKey pf{1, 3};
    lmem::ExpertKey other{1, 4};
    std::vector<uint8_t> buf(64);
    elm->ensure_resident(pf, 0, lmem::CacheZone::kStable, 100,
                         /*priority=*/-1.0f);
    transfer_engine_->enqueue_h2d(other, 0, buf.data(), buf.data(), 64,
                                  /*priority=*/0.0f);
    ASSERT_EQ(transfer_engine_->staged_count(0), 2);
    EXPECT_EQ(elm->state(pf, 0).gpu_tier, ldam::GpuTier::kTransferring);

    // Demand joins the prefetch at demand priority.
    elm->ensure_resident(pf, 0, lmem::CacheZone::kStable, 102,
                         std::numeric_limits<float>::max());

    // Single-slot flush must admit the demand-joined transfer, not `other`
    // (without the re-assert, `other` at 0.0 would beat the -1.0 prefetch).
    transfer_engine_->flush_staged();
    EXPECT_TRUE(transfer_engine_->is_dispatched_h2d(pf, 0));
    EXPECT_FALSE(transfer_engine_->is_dispatched_h2d(other, 0));

    // Drain: both interests on pf complete successfully.
    auto completions = poll_elm(*elm).lifecycle;
    transfer_engine_->flush_staged();
    auto more = poll_elm(*elm).lifecycle;
    completions.insert(completions.end(), more.begin(), more.end());
    EXPECT_EQ(completions.size(), 2u);
    for (const auto& c : completions) EXPECT_TRUE(c.success);
}

TEST_F(ExpertLifecycleManagerTest, EnsureResident_MmapSource) {
    auto elm = make_elm();
    lmem::ExpertKey key{2, 5};

    // Host state should be WARM (mmap available).
    EXPECT_EQ(elm->host_state(key), ldam::HostTier::kWarm);

    auto token = elm->ensure_resident(key, 1, lmem::CacheZone::kStreaming, 300);
    auto completions = poll_elm(*elm).lifecycle;
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_TRUE(completions[0].success);
    EXPECT_EQ(completions[0].gpu_idx, 1);
}

TEST_F(ExpertLifecycleManagerTest, EnsureResident_NoHostSource) {
    // ELM with no LoadedModel and no NvmeTier — should fail.
    ldam::ExpertLifecycleManager::Deps deps{
        .expert_cache    = cache_.get(),
        .transfer_engine = transfer_engine_.get(),
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(deps));

    lmem::ExpertKey key{1, 0};
    auto token = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 400);

    auto completions = poll_elm(*elm).lifecycle;
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_FALSE(completions[0].success);
}

// ═════════════════════════════════════════════════════════════════════════════
// Interest refcounting

TEST_F(ExpertLifecycleManagerTest, DuplicateEnsureResident_BothNotified) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 3};

    auto t1 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 10);
    auto t2 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 20);
    auto t3 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 30);

    auto st = elm->state(key, 0);
    EXPECT_EQ(st.interest_count, 3u);

    auto completions = poll_elm(*elm).lifecycle;
    EXPECT_GE(completions.size(), 3u);

    for (const auto& c : completions)
        EXPECT_TRUE(c.success);
}

TEST_F(ExpertLifecycleManagerTest, CancelOne_OtherStillCompletes) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 4};

    auto t1 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 10);
    auto t2 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 20);

    // Cancel the first interest.
    elm->cancel(t1);

    auto st = elm->state(key, 0);
    EXPECT_EQ(st.interest_count, 1u);

    // Poll — second should still complete.
    auto completions = poll_elm(*elm).lifecycle;
    bool found_t2 = false;
    for (const auto& c : completions) {
        EXPECT_NE(c.token, t1);  // cancelled token should not appear
        if (c.token == t2) { found_t2 = true; EXPECT_TRUE(c.success); }
    }
    EXPECT_TRUE(found_t2);
}

TEST_F(ExpertLifecycleManagerTest, CancelAll_TransferCancelled) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 5};

    auto t1 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 10);

    // Cancel the only interest — should revert to ABSENT.
    elm->cancel(t1);

    auto st = elm->state(key, 0);
    EXPECT_EQ(st.gpu_tier, ldam::GpuTier::kAbsent);
    EXPECT_EQ(st.interest_count, 0u);

    // Poll should return nothing for this expert.
    auto completions = poll_elm(*elm).lifecycle;
    for (const auto& c : completions)
        EXPECT_NE(c.token, t1);
}

// ═════════════════════════════════════════════════════════════════════════════
// Eviction

TEST_F(ExpertLifecycleManagerTest, Evict_HotExpert) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 0};

    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    poll_elm(*elm);

    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kHot);

    bool ok = elm->request_evict(key, 0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kAbsent);
}

TEST_F(ExpertLifecycleManagerTest, Evict_WhileTransferring_Blocked) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 1};

    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    // Don't poll — still transferring.

    bool ok = elm->request_evict(key, 0);
    EXPECT_FALSE(ok);  // Interest pending (refcount > 0)
}

TEST_F(ExpertLifecycleManagerTest, Evict_AfterCancelAll_Succeeds) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 2};

    auto token = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    elm->cancel(token);  // Reverts to ABSENT

    // Entry is ABSENT after cancel — nothing to evict via ELM.
    // But ExpertCache might still have it (cancel calls evict internally).
    auto st = elm->state(key, 0);
    EXPECT_EQ(st.gpu_tier, ldam::GpuTier::kAbsent);
}

// ═════════════════════════════════════════════════════════════════════════════
// Multi-GPU

TEST_F(ExpertLifecycleManagerTest, EnsureResident_TwoGpus) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 0};

    auto t0 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 10);
    auto t1 = elm->ensure_resident(key, 1, lmem::CacheZone::kStable, 20);

    auto completions = poll_elm(*elm).lifecycle;
    EXPECT_GE(completions.size(), 2u);

    auto st0 = elm->state(key, 0);
    auto st1 = elm->state(key, 1);
    EXPECT_EQ(st0.gpu_tier, ldam::GpuTier::kHot);
    EXPECT_EQ(st1.gpu_tier, ldam::GpuTier::kHot);
}

// ═════════════════════════════════════════════════════════════════════════════
// State queries

TEST_F(ExpertLifecycleManagerTest, HostState_MmapIsWarm) {
    auto elm = make_elm();
    EXPECT_EQ(elm->host_state({1, 0}), ldam::HostTier::kWarm);
    EXPECT_EQ(elm->host_state({2, 3}), ldam::HostTier::kWarm);
}

TEST_F(ExpertLifecycleManagerTest, HostState_NoSource_IsCold) {
    ldam::ExpertLifecycleManager::Deps deps{
        .expert_cache    = cache_.get(),
        .transfer_engine = transfer_engine_.get(),
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(deps));
    EXPECT_EQ(elm->host_state({1, 0}), ldam::HostTier::kCold);
}

TEST_F(ExpertLifecycleManagerTest, State_AbsentByDefault) {
    auto elm = make_elm();
    auto st = elm->state({1, 0}, 0);
    EXPECT_EQ(st.gpu_tier, ldam::GpuTier::kAbsent);
    EXPECT_EQ(st.interest_count, 0u);
}

// ═════════════════════════════════════════════════════════════════════════════
// Zone management

TEST_F(ExpertLifecycleManagerTest, Promote_WhileHot) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 0};

    elm->ensure_resident(key, 0, lmem::CacheZone::kStreaming, 100);
    poll_elm(*elm);

    // Promote from streaming → stable.
    bool ok = elm->promote(key, 0);
    EXPECT_TRUE(ok);
}

TEST_F(ExpertLifecycleManagerTest, Promote_WhileTransferring_Blocked) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 1};

    elm->ensure_resident(key, 0, lmem::CacheZone::kStreaming, 100);
    // Don't poll — still transferring.

    bool ok = elm->promote(key, 0);
    EXPECT_FALSE(ok);
}

TEST_F(ExpertLifecycleManagerTest, Demote_WhileHot) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 0};

    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    poll_elm(*elm);

    bool ok = elm->demote(key, 0);
    EXPECT_TRUE(ok);
}

// ═════════════════════════════════════════════════════════════════════════════
// Poll mechanics

TEST_F(ExpertLifecycleManagerTest, Poll_EmptyReturnsNothing) {
    auto elm = make_elm();
    auto completions = poll_elm(*elm).lifecycle;
    EXPECT_TRUE(completions.empty());
}

TEST_F(ExpertLifecycleManagerTest, Poll_AdvancesTransferCompletions) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 0};

    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 500);

    // First poll: transfer completes via null backend.
    auto c1 = poll_elm(*elm).lifecycle;
    ASSERT_EQ(c1.size(), 1u);
    EXPECT_TRUE(c1[0].success);

    // ExpertCache should be marked ready.
    const auto* entry = cache_->lookup(key, 0);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->sub_components_ready,
              static_cast<uint8_t>(lmem::SubComponent::kAll));
}

TEST_F(ExpertLifecycleManagerTest, Poll_MultipleExpertsInOneCycle) {
    auto elm = make_elm();

    elm->ensure_resident({1, 0}, 0, lmem::CacheZone::kStable, 10);
    elm->ensure_resident({1, 1}, 0, lmem::CacheZone::kStable, 20);
    elm->ensure_resident({1, 2}, 0, lmem::CacheZone::kStable, 30);

    auto completions = poll_elm(*elm).lifecycle;
    EXPECT_EQ(completions.size(), 3u);

    for (const auto& c : completions)
        EXPECT_TRUE(c.success);
}

// ═════════════════════════════════════════════════════════════════════════════
// Race resolution: stage then prefetch same expert

TEST_F(ExpertLifecycleManagerTest, StageAndPrefetch_SameExpert_Deduped) {
    // Both should become ensure_resident(); second adds interest.
    auto elm = make_elm();
    lmem::ExpertKey key{2, 3};

    auto t1 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 10);
    auto t2 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 20);

    // Only one reserve + one H2D should happen.
    auto st = elm->state(key, 0);
    EXPECT_EQ(st.interest_count, 2u);

    auto completions = poll_elm(*elm).lifecycle;
    EXPECT_GE(completions.size(), 2u);
}

// ═════════════════════════════════════════════════════════════════════════════
// Race resolution: evict while transferring

TEST_F(ExpertLifecycleManagerTest, EvictWhileInterestPending_Blocked) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 6};

    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    // Interest pending → evict should fail.
    EXPECT_FALSE(elm->request_evict(key, 0));

    // Complete the transfer.
    poll_elm(*elm);
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kHot);

    // Now evict should succeed.
    EXPECT_TRUE(elm->request_evict(key, 0));
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kAbsent);
}

// ═════════════════════════════════════════════════════════════════════════════
// Race resolution: cancel then callback suppressed

TEST_F(ExpertLifecycleManagerTest, CancelThenPoll_NoCompletion) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 7};

    auto token = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    elm->cancel(token);

    // Poll — should not produce a completion for the cancelled token.
    auto completions = poll_elm(*elm).lifecycle;
    for (const auto& c : completions)
        EXPECT_NE(c.token, token);
}

// ═════════════════════════════════════════════════════════════════════════════
// Lifecycle: load, evict, re-load

TEST_F(ExpertLifecycleManagerTest, ReloadAfterEvict) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 0};

    // Load.
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    poll_elm(*elm);
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kHot);

    // Evict.
    EXPECT_TRUE(elm->request_evict(key, 0));
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kAbsent);

    // Re-load.
    auto token = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 200);
    auto completions = poll_elm(*elm).lifecycle;
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_TRUE(completions[0].success);
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kHot);
}

// ═════════════════════════════════════════════════════════════════════════════
// ELM-4: find_by_cmd_seq

TEST_F(ExpertLifecycleManagerTest, FindByCmdSeq_Found) {
    auto elm = make_elm();
    auto token = elm->ensure_resident({1, 0}, 0, lmem::CacheZone::kStable, 100);
    EXPECT_EQ(elm->find_by_cmd_seq(100), token);
}

TEST_F(ExpertLifecycleManagerTest, FindByCmdSeq_NotFound) {
    auto elm = make_elm();
    EXPECT_EQ(elm->find_by_cmd_seq(999), 0u);
}

TEST_F(ExpertLifecycleManagerTest, CancelByCmdSeq_CleansUp) {
    auto elm = make_elm();
    auto token = elm->ensure_resident({1, 0}, 0, lmem::CacheZone::kStable, 100);
    EXPECT_NE(elm->find_by_cmd_seq(100), 0u);

    elm->cancel(token);
    EXPECT_EQ(elm->find_by_cmd_seq(100), 0u);
}

// ═════════════════════════════════════════════════════════════════════════════
// ELM-9: batch cmd_seq — multimap support

TEST_F(ExpertLifecycleManagerTest, FindAllByCmdSeq_BatchSharedCmdSeq) {
    auto elm = make_elm();
    // Simulate a batch: 3 experts with the same cmd_seq.
    auto t1 = elm->ensure_resident({1, 0}, 0, lmem::CacheZone::kStable, 42);
    auto t2 = elm->ensure_resident({1, 1}, 0, lmem::CacheZone::kStable, 42);
    auto t3 = elm->ensure_resident({1, 2}, 0, lmem::CacheZone::kStable, 42);

    auto all = elm->find_all_by_cmd_seq(42);
    EXPECT_EQ(all.size(), 3u);

    // Each token should appear exactly once.
    std::sort(all.begin(), all.end());
    std::vector<ldam::LifecycleToken> expected{t1, t2, t3};
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(all, expected);

    // find_by_cmd_seq returns one of them.
    auto single = elm->find_by_cmd_seq(42);
    EXPECT_NE(single, 0u);
}

TEST_F(ExpertLifecycleManagerTest, FindAllByCmdSeq_Empty) {
    auto elm = make_elm();
    auto all = elm->find_all_by_cmd_seq(999);
    EXPECT_TRUE(all.empty());
}

TEST_F(ExpertLifecycleManagerTest, CancelOneBatchEntry_OthersRemain) {
    auto elm = make_elm();
    auto t1 = elm->ensure_resident({1, 0}, 0, lmem::CacheZone::kStable, 42);
    auto t2 = elm->ensure_resident({1, 1}, 0, lmem::CacheZone::kStable, 42);
    auto t3 = elm->ensure_resident({1, 2}, 0, lmem::CacheZone::kStable, 42);

    // Cancel one entry — the other two remain findable.
    elm->cancel(t2);
    auto remaining = elm->find_all_by_cmd_seq(42);
    EXPECT_EQ(remaining.size(), 2u);

    // The cancelled token is gone.
    EXPECT_EQ(std::count(remaining.begin(), remaining.end(), t2), 0);
}

TEST_F(ExpertLifecycleManagerTest, CancelAllBatchEntries_CmdSeqGone) {
    auto elm = make_elm();
    auto t1 = elm->ensure_resident({1, 0}, 0, lmem::CacheZone::kStable, 42);
    auto t2 = elm->ensure_resident({1, 1}, 0, lmem::CacheZone::kStable, 42);

    elm->cancel(t1);
    elm->cancel(t2);

    EXPECT_TRUE(elm->find_all_by_cmd_seq(42).empty());
    EXPECT_EQ(elm->find_by_cmd_seq(42), 0u);
}

// ═════════════════════════════════════════════════════════════════════════════
// ELM-5: input-driven poll

TEST_F(ExpertLifecycleManagerTest, Poll_InputDriven_HandlesTransfers) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 0};

    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 500);

    // Manually poll subsystems and pass to ELM (simulates DaemonLoop).
    auto xfer = transfer_engine_->poll_completions();
    std::vector<lmem::IoCompletion> nvme;
    auto result = elm->poll(xfer, nvme);

    ASSERT_EQ(result.lifecycle.size(), 1u);
    EXPECT_TRUE(result.lifecycle[0].success);
    EXPECT_EQ(result.lifecycle[0].cmd_seq, 500u);
    EXPECT_TRUE(result.unhandled_transfers.empty());
}

TEST_F(ExpertLifecycleManagerTest, Poll_UnhandledTransfers_PassedThrough) {
    auto elm = make_elm();

    // Construct a transfer completion for a non-ELM expert.
    ltr::TransferCompletion tc{};
    tc.token     = 42;
    tc.key       = {3, 7};  // Not managed by ELM
    tc.gpu_idx   = 0;
    tc.direction = ltr::TransferDirection::kH2D;
    tc.bytes     = 1024;
    tc.success   = true;

    std::vector<ltr::TransferCompletion> xfer{tc};
    std::vector<lmem::IoCompletion> nvme;
    auto result = elm->poll(xfer, nvme);

    EXPECT_TRUE(result.lifecycle.empty());
    ASSERT_EQ(result.unhandled_transfers.size(), 1u);
    EXPECT_EQ(result.unhandled_transfers[0].key, tc.key);
}

// ═════════════════════════════════════════════════════════════════════════════
// ELM-8: Snapshot publication tests
// ═════════════════════════════════════════════════════════════════════════════

namespace lipc = layerstorm::ipc;

TEST_F(ExpertLifecycleManagerTest, Snapshot_NullNoCrash) {
    // ELM with null snapshot — all operations should succeed without crash.
    auto elm = make_elm();
    lmem::ExpertKey key{1, 0};
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    poll_elm(*elm);
    elm->request_evict(key, 0);
}

TEST_F(ExpertLifecycleManagerTest, Snapshot_EnsureResidentUpdatesGpuTier) {
    lipc::StateSnapshot snap{};
    std::memset(&snap, 0, sizeof(snap));
    auto elm = make_elm_with_snapshot(snap);

    lmem::ExpertKey key{1, 0};  // moe_layer=0 (layer_idx=1, first_moe=1)

    // GPU triple index: moe_layer=0, expert=0, gpu=0
    size_t flat = 0 * lipc::kMaxExperts * lipc::kMaxGpus
                + 0 * lipc::kMaxGpus + 0;

    // Before: should be ABSENT (from bootstrap — mmap-warm but no cache entry).
    // Actually bootstrap populates HOT for pre-cached entries, ABSENT otherwise.
    EXPECT_EQ(snap.expert_gpu_tier[flat],
              static_cast<uint8_t>(ldam::GpuTier::kAbsent));

    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);

    // After ensure_resident: should be TRANSFERRING (reserved then H2D started).
    EXPECT_EQ(snap.expert_gpu_tier[flat],
              static_cast<uint8_t>(ldam::GpuTier::kTransferring));
    // Bitmap should NOT be set yet (TRANSFERRING < kHot).
    EXPECT_EQ(snap.residency_bitmap[flat / 8] & (1u << (flat % 8)), 0u);

    // Poll H2D completion.
    poll_elm(*elm);
    EXPECT_EQ(snap.expert_gpu_tier[flat],
              static_cast<uint8_t>(ldam::GpuTier::kHot));
    // Bitmap should be set now (HOT = fully ready).
    EXPECT_NE(snap.residency_bitmap[flat / 8] & (1u << (flat % 8)), 0u);
}

TEST_F(ExpertLifecycleManagerTest, Snapshot_InterestCountTracked) {
    lipc::StateSnapshot snap{};
    std::memset(&snap, 0, sizeof(snap));
    auto elm = make_elm_with_snapshot(snap);

    lmem::ExpertKey key{2, 3};  // moe_layer=1, expert=3
    size_t flat = 1 * lipc::kMaxExperts * lipc::kMaxGpus
                + 3 * lipc::kMaxGpus + 0;

    auto t1 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 10);
    EXPECT_EQ(snap.expert_interest_count[flat], 1u);

    auto t2 = elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 20);
    EXPECT_EQ(snap.expert_interest_count[flat], 2u);

    elm->cancel(t1);
    EXPECT_EQ(snap.expert_interest_count[flat], 1u);

    (void)t2;
}

TEST_F(ExpertLifecycleManagerTest, Snapshot_EvictUpdatesGpuTier) {
    lipc::StateSnapshot snap{};
    std::memset(&snap, 0, sizeof(snap));
    auto elm = make_elm_with_snapshot(snap);

    lmem::ExpertKey key{1, 2};  // moe_layer=0, expert=2
    size_t flat = 0 * lipc::kMaxExperts * lipc::kMaxGpus
                + 2 * lipc::kMaxGpus + 0;

    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    poll_elm(*elm);  // HOT
    EXPECT_EQ(snap.expert_gpu_tier[flat],
              static_cast<uint8_t>(ldam::GpuTier::kHot));
    EXPECT_NE(snap.residency_bitmap[flat / 8] & (1u << (flat % 8)), 0u);

    elm->request_evict(key, 0);
    EXPECT_EQ(snap.expert_gpu_tier[flat],
              static_cast<uint8_t>(ldam::GpuTier::kAbsent));
    EXPECT_EQ(snap.expert_interest_count[flat], 0u);
    // Bitmap cleared on evict.
    EXPECT_EQ(snap.residency_bitmap[flat / 8] & (1u << (flat % 8)), 0u);
}

TEST_F(ExpertLifecycleManagerTest, Snapshot_HostTierUpdated) {
    lipc::StateSnapshot snap{};
    std::memset(&snap, 0, sizeof(snap));
    auto elm = make_elm_with_snapshot(snap);

    // With mmap LoadedModel, host tier should be WARM at bootstrap.
    size_t idx = 0 * lipc::kMaxExperts + 0;  // moe_layer=0, expert=0
    EXPECT_EQ(snap.host_tier[idx],
              static_cast<uint8_t>(ldam::HostTier::kWarm));
}

TEST_F(ExpertLifecycleManagerTest, Snapshot_HostResidentBitmap) {
    lipc::StateSnapshot snap{};
    std::memset(&snap, 0, sizeof(snap));
    auto elm = make_elm_with_snapshot(snap);

    // With mmap, all experts should be warm in bitmap.
    size_t idx = 0 * lipc::kMaxExperts + 0;
    EXPECT_NE(snap.host_resident_bitmap[idx / 8] & (1u << (idx % 8)), 0u);
}

TEST_F(ExpertLifecycleManagerTest, Snapshot_TimestampUpdated) {
    lipc::StateSnapshot snap{};
    std::memset(&snap, 0, sizeof(snap));
    auto elm = make_elm_with_snapshot(snap);

    lmem::ExpertKey key{1, 0};
    size_t expert_idx = 0 * lipc::kMaxExperts + 0;

    // Bootstrap sets timestamp for warm experts.
    uint64_t ts_bootstrap = snap.expert_last_change_ns[expert_idx];
    EXPECT_GT(ts_bootstrap, 0u);

    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    uint64_t ts_after = snap.expert_last_change_ns[expert_idx];
    EXPECT_GE(ts_after, ts_bootstrap);
}

TEST_F(ExpertLifecycleManagerTest, Snapshot_InitBootstrap) {
    // Pre-populate ExpertCache, then create ELM — snapshot should reflect.
    lmem::ExpertKey key{1, 0};
    cache_->reserve(key, 0, lmem::CacheZone::kStable);
    cache_->mark_all_ready(key, 0);

    lipc::StateSnapshot snap{};
    std::memset(&snap, 0, sizeof(snap));
    auto elm = make_elm_with_snapshot(snap);

    size_t flat = 0 * lipc::kMaxExperts * lipc::kMaxGpus
                + 0 * lipc::kMaxGpus + 0;
    EXPECT_EQ(snap.expert_gpu_tier[flat],
              static_cast<uint8_t>(ldam::GpuTier::kHot));
    // Bitmap set for HOT entries at bootstrap.
    EXPECT_NE(snap.residency_bitmap[flat / 8] & (1u << (flat % 8)), 0u);
}

// ═════════════════════════════════════════════════════════════════════════════
// ELM-8b: Per-NUMA host tier snapshot tests
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(ExpertLifecycleManagerTest, Snapshot_HostNumaTierBootstrap) {
    lipc::StateSnapshot snap{};
    std::memset(&snap, 0, sizeof(snap));
    auto elm = make_elm_with_snapshot(snap);

    // mmap experts should be warm on NUMA 0.
    size_t idx = 0 * lipc::kMaxExperts + 0;  // moe_layer=0, expert=0
    size_t numa_base = idx * lipc::kMaxNuma;
    EXPECT_EQ(snap.host_numa_tier[numa_base + 0],
              static_cast<uint8_t>(ldam::HostTier::kWarm));
    // Other NUMA nodes should be cold.
    for (int n = 1; n < lipc::kMaxNuma; ++n) {
        EXPECT_EQ(snap.host_numa_tier[numa_base + n],
                  static_cast<uint8_t>(ldam::HostTier::kCold));
    }
}

TEST_F(ExpertLifecycleManagerTest, Snapshot_HostNumaTierColdExpert) {
    lipc::StateSnapshot snap{};
    std::memset(&snap, 0, sizeof(snap));
    auto elm = make_elm_with_snapshot(snap);

    // Expert beyond the model's range (num_experts_=8) stays all-cold.
    size_t idx = 0 * lipc::kMaxExperts + 9;  // expert=9
    size_t numa_base = idx * lipc::kMaxNuma;
    for (int n = 0; n < lipc::kMaxNuma; ++n) {
        EXPECT_EQ(snap.host_numa_tier[numa_base + n],
                  static_cast<uint8_t>(ldam::HostTier::kCold));
    }
}

TEST_F(ExpertLifecycleManagerTest, Snapshot_HostNumaTierMultipleLayers) {
    lipc::StateSnapshot snap{};
    std::memset(&snap, 0, sizeof(snap));
    auto elm = make_elm_with_snapshot(snap);

    // Check a warm expert in a different MoE layer.
    // moe_layer=2 (layer_idx=3, first_moe=1), expert=5
    size_t idx = 2 * lipc::kMaxExperts + 5;
    size_t numa_base = idx * lipc::kMaxNuma;
    EXPECT_EQ(snap.host_numa_tier[numa_base + 0],
              static_cast<uint8_t>(ldam::HostTier::kWarm));
    for (int n = 1; n < lipc::kMaxNuma; ++n) {
        EXPECT_EQ(snap.host_numa_tier[numa_base + n],
                  static_cast<uint8_t>(ldam::HostTier::kCold));
    }
}

TEST_F(ExpertLifecycleManagerTest, Snapshot_HostNumaTierWithNvmeTier) {
    // NvmeTier allocates host buffers on a specific NUMA node (via gpu_hint).
    // Verify that the per-NUMA tier reflects the NvmeTier's NUMA placement.
    std::string tmpdir = [] {
        char tmpl[] = "/tmp/layerstorm_elm_numa_XXXXXX";
        char* dir = ::mkdtemp(tmpl);
        if (!dir) throw std::runtime_error("mkdtemp failed");
        return std::string(dir);
    }();

    lc::HardwareConfig hw;
    lc::GpuConfig g;
    g.id = 0; g.type = lc::GpuType::rtx5090; g.numa_node = 0;
    hw.gpus.push_back(g);
    lmem::NumaManager numa(hw);

    lmem::NvmeTier::Options nvme_opts;
    nvme_opts.drive_paths = {tmpdir};
    nvme_opts.slot_size_bytes = expert_bytes_;
    nvme_opts.host_ram_budget_bytes = 10 * expert_bytes_;
    nvme_opts.num_moe_layers = 6;
    nvme_opts.num_experts_per_layer = 8;
    nvme_opts.first_moe_layer = 0;
    auto nvme = std::make_unique<lmem::NvmeTier>(std::move(nvme_opts), numa);

    lipc::StateSnapshot snap{};
    std::memset(&snap, 0, sizeof(snap));

    ldam::ExpertLifecycleManager::Deps deps{
        .expert_cache    = cache_.get(),
        .nvme_tier       = nvme.get(),
        .transfer_engine = transfer_engine_.get(),
        .numa_manager    = &numa,
        .loaded_model    = &model_,
        .snapshot        = &snap,
        .first_moe_layer = 1,
        .num_moe_layers  = 5,
        .num_experts     = 8,
        .num_gpus        = 2,
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(deps));

    // Load expert to HOT, then drain to host RAM.
    lmem::ExpertKey key{1, 0};
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    auto r1 = poll_elm(*elm);
    ASSERT_EQ(r1.lifecycle.size(), 1u);
    EXPECT_TRUE(r1.lifecycle[0].success);

    elm->request_drain(key, 0, 200);
    auto r2 = poll_elm(*elm);
    ASSERT_EQ(r2.lifecycle.size(), 1u);
    EXPECT_TRUE(r2.lifecycle[0].success);

    // Expert is now in host via NvmeTier (mmap-backed, WP-5).
    EXPECT_TRUE(nvme->is_in_host_ram(key));
    int host_numa = nvme->host_numa_node(key);
    // WP-5: mmap pages are OS-managed — NUMA node is -1.
    EXPECT_EQ(host_numa, -1);

    // WP-5: NvmeTier returns NUMA=-1 (mmap, OS-managed).  The ELM falls
    // through to the loaded_model path which assumes NUMA 0 for mmap.
    // So NUMA 0 should show kWarm, others kCold.
    size_t idx = 0 * lipc::kMaxExperts + 0;  // moe_layer=0, expert=0
    size_t numa_base = idx * lipc::kMaxNuma;
    for (int n = 0; n < lipc::kMaxNuma; ++n) {
        if (n == 0) {
            EXPECT_EQ(snap.host_numa_tier[numa_base + n],
                      static_cast<uint8_t>(ldam::HostTier::kWarm))
                << "NUMA node 0 should be warm (loaded_model fallback)";
        } else {
            EXPECT_EQ(snap.host_numa_tier[numa_base + n],
                      static_cast<uint8_t>(ldam::HostTier::kCold))
                << "NUMA node " << n << " should be cold";
        }
    }

    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
}

// ═════════════════════════════════════════════════════════════════════════════
// Drain (ELM-6): HOT → kDraining → ABSENT
// ═════════════════════════════════════════════════════════════════════════════

static std::string make_temp_dir() {
    char tmpl[] = "/tmp/layerstorm_elm_drain_XXXXXX";
    char* dir = ::mkdtemp(tmpl);
    if (!dir) throw std::runtime_error("mkdtemp failed");
    return std::string(dir);
}

TEST_F(ExpertLifecycleManagerTest, RequestDrain_HotToAbsent) {
    // Setup NvmeTier + NumaManager for drain path.
    std::string tmpdir = make_temp_dir();
    lc::HardwareConfig hw;
    lc::GpuConfig g;
    g.id = 0; g.type = lc::GpuType::rtx5090; g.numa_node = 0;
    hw.gpus.push_back(g);
    lmem::NumaManager numa(hw);

    lmem::NvmeTier::Options nvme_opts;
    nvme_opts.drive_paths = {tmpdir};
    nvme_opts.slot_size_bytes = expert_bytes_;
    nvme_opts.host_ram_budget_bytes = 10 * expert_bytes_;
    nvme_opts.num_moe_layers = 6;
    nvme_opts.num_experts_per_layer = 8;
    nvme_opts.first_moe_layer = 0;
    auto nvme = std::make_unique<lmem::NvmeTier>(std::move(nvme_opts), numa);

    ldam::ExpertLifecycleManager::Deps deps{
        .expert_cache    = cache_.get(),
        .nvme_tier       = nvme.get(),
        .transfer_engine = transfer_engine_.get(),
        .numa_manager    = &numa,
        .loaded_model    = &model_,
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(deps));

    // Load expert to HOT.
    lmem::ExpertKey key{1, 0};
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 100);
    auto r1 = poll_elm(*elm);
    ASSERT_EQ(r1.lifecycle.size(), 1u);
    EXPECT_TRUE(r1.lifecycle[0].success);
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kHot);

    // Initiate drain.
    EXPECT_TRUE(elm->request_drain(key, 0, 200));
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kDraining);

    // Poll to complete D2H.
    auto r2 = poll_elm(*elm);
    ASSERT_EQ(r2.lifecycle.size(), 1u);
    EXPECT_TRUE(r2.lifecycle[0].success);
    EXPECT_TRUE(r2.lifecycle[0].is_eviction);
    EXPECT_EQ(r2.lifecycle[0].cmd_seq, 200u);
    EXPECT_EQ(r2.lifecycle[0].key, key);
    EXPECT_GE(r2.lifecycle[0].dma_us, 0u);  // may round to 0µs with null backend

    // Expert evicted from VRAM, now in host warm cache.
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kAbsent);
    EXPECT_EQ(cache_->lookup(key, 0), nullptr);
    EXPECT_TRUE(nvme->is_in_host_ram(key));

    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
}

TEST_F(ExpertLifecycleManagerTest, RequestDrain_NotHot_Fails) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 0};

    // Expert is ABSENT — drain should fail.
    EXPECT_FALSE(elm->request_drain(key, 0, 300));
}

TEST_F(ExpertLifecycleManagerTest, RequestDrain_InterestsPending_Fails) {
    auto elm = make_elm();
    lmem::ExpertKey key{1, 0};

    // Load expert but don't poll — interest is pending (TRANSFERRING).
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 400);
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kTransferring);
    EXPECT_FALSE(elm->request_drain(key, 0, 401));

    // Poll to HOT, add a second interest, then try drain.
    auto r1 = poll_elm(*elm);
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kHot);

    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 402);
    // Interest count is 1 (the second ensure_resident adds an interest).
    EXPECT_FALSE(elm->request_drain(key, 0, 403));

    // Drain the second interest.
    auto r2 = poll_elm(*elm);
}

TEST_F(ExpertLifecycleManagerTest, EnsureResident_WhileDraining_Fails) {
    std::string tmpdir = make_temp_dir();
    lc::HardwareConfig hw;
    lc::GpuConfig g;
    g.id = 0; g.type = lc::GpuType::rtx5090; g.numa_node = 0;
    hw.gpus.push_back(g);
    lmem::NumaManager numa(hw);

    lmem::NvmeTier::Options nvme_opts;
    nvme_opts.drive_paths = {tmpdir};
    nvme_opts.slot_size_bytes = expert_bytes_;
    nvme_opts.host_ram_budget_bytes = 10 * expert_bytes_;
    nvme_opts.num_moe_layers = 6;
    nvme_opts.num_experts_per_layer = 8;
    nvme_opts.first_moe_layer = 0;
    auto nvme = std::make_unique<lmem::NvmeTier>(std::move(nvme_opts), numa);

    ldam::ExpertLifecycleManager::Deps deps{
        .expert_cache    = cache_.get(),
        .nvme_tier       = nvme.get(),
        .transfer_engine = transfer_engine_.get(),
        .numa_manager    = &numa,
        .loaded_model    = &model_,
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(deps));

    lmem::ExpertKey key{1, 0};
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 500);
    poll_elm(*elm);
    ASSERT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kHot);

    // Start drain.
    ASSERT_TRUE(elm->request_drain(key, 0, 501));
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kDraining);

    // Try ensure_resident while draining — should get immediate failure.
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 502);
    auto r = poll_elm(*elm);

    // Should have drain completion + failure completion.
    int evictions = 0, failures = 0;
    for (const auto& lc : r.lifecycle) {
        if (lc.is_eviction) ++evictions;
        if (!lc.success && !lc.is_eviction) ++failures;
    }
    EXPECT_EQ(evictions, 1);
    EXPECT_EQ(failures, 1);

    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
}

TEST_F(ExpertLifecycleManagerTest, RequestEvict_WhileDraining_Blocked) {
    std::string tmpdir = make_temp_dir();
    lc::HardwareConfig hw;
    lc::GpuConfig g;
    g.id = 0; g.type = lc::GpuType::rtx5090; g.numa_node = 0;
    hw.gpus.push_back(g);
    lmem::NumaManager numa(hw);

    lmem::NvmeTier::Options nvme_opts;
    nvme_opts.drive_paths = {tmpdir};
    nvme_opts.slot_size_bytes = expert_bytes_;
    nvme_opts.host_ram_budget_bytes = 10 * expert_bytes_;
    nvme_opts.num_moe_layers = 6;
    nvme_opts.num_experts_per_layer = 8;
    nvme_opts.first_moe_layer = 0;
    auto nvme = std::make_unique<lmem::NvmeTier>(std::move(nvme_opts), numa);

    ldam::ExpertLifecycleManager::Deps deps{
        .expert_cache    = cache_.get(),
        .nvme_tier       = nvme.get(),
        .transfer_engine = transfer_engine_.get(),
        .numa_manager    = &numa,
        .loaded_model    = &model_,
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(deps));

    lmem::ExpertKey key{1, 0};
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 600);
    poll_elm(*elm);

    ASSERT_TRUE(elm->request_drain(key, 0, 601));
    EXPECT_FALSE(elm->request_evict(key, 0));

    // Complete drain.
    poll_elm(*elm);
    EXPECT_EQ(elm->state(key, 0).gpu_tier, ldam::GpuTier::kAbsent);

    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
}

TEST_F(ExpertLifecycleManagerTest, Drain_SnapshotPublished) {
    std::string tmpdir = make_temp_dir();
    lc::HardwareConfig hw;
    lc::GpuConfig g;
    g.id = 0; g.type = lc::GpuType::rtx5090; g.numa_node = 0;
    hw.gpus.push_back(g);
    lmem::NumaManager numa(hw);

    lmem::NvmeTier::Options nvme_opts;
    nvme_opts.drive_paths = {tmpdir};
    nvme_opts.slot_size_bytes = expert_bytes_;
    nvme_opts.host_ram_budget_bytes = 10 * expert_bytes_;
    nvme_opts.num_moe_layers = 6;
    nvme_opts.num_experts_per_layer = 8;
    nvme_opts.first_moe_layer = 0;
    auto nvme = std::make_unique<lmem::NvmeTier>(std::move(nvme_opts), numa);

    lipc::StateSnapshot snap{};
    std::memset(&snap, 0, sizeof(snap));

    ldam::ExpertLifecycleManager::Deps deps{
        .expert_cache    = cache_.get(),
        .nvme_tier       = nvme.get(),
        .transfer_engine = transfer_engine_.get(),
        .numa_manager    = &numa,
        .loaded_model    = &model_,
        .snapshot        = &snap,
        .first_moe_layer = 1,
        .num_moe_layers  = 5,
        .num_experts     = 8,
        .num_gpus        = 2,
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(deps));

    lmem::ExpertKey key{1, 0};
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 700);
    poll_elm(*elm);

    // flat index: (layer - first_moe) * experts * gpus + expert * gpus + gpu
    size_t flat = 0 * lipc::kMaxExperts * lipc::kMaxGpus
                + 0 * lipc::kMaxGpus + 0;

    EXPECT_EQ(snap.expert_gpu_tier[flat],
              static_cast<uint8_t>(ldam::GpuTier::kHot));

    // Drain → kDraining published.
    ASSERT_TRUE(elm->request_drain(key, 0, 701));
    EXPECT_EQ(snap.expert_gpu_tier[flat],
              static_cast<uint8_t>(ldam::GpuTier::kDraining));
    // Residency bitmap cleared (kDraining < kHot).
    EXPECT_EQ(snap.residency_bitmap[flat / 8] & (1u << (flat % 8)), 0u);

    // Complete drain → kAbsent published.
    poll_elm(*elm);
    EXPECT_EQ(snap.expert_gpu_tier[flat],
              static_cast<uint8_t>(ldam::GpuTier::kAbsent));

    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
}

// ═══════════════════════════════════════════════════════════════════════════
// ELM-13: ProgressEvent emission
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(ExpertLifecycleManagerTest, Progress_EmitH2dStarted) {
    ldam::ExpertLifecycleManager::Deps deps{
        .expert_cache    = cache_.get(),
        .transfer_engine = transfer_engine_.get(),
        .loaded_model    = &model_,
        .emit_progress   = true,
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(deps));

    auto token = elm->ensure_resident({1, 0}, 0,
                                      lmem::CacheZone::kStable, 900);
    EXPECT_NE(token, 0u);

    auto result = poll_elm(*elm);
    ASSERT_EQ(result.progress.size(), 1u);
    EXPECT_EQ(result.progress[0].key, (lmem::ExpertKey{1, 0}));
    EXPECT_EQ(result.progress[0].gpu_idx, 0);
    EXPECT_EQ(result.progress[0].phase, 3);
    EXPECT_EQ(result.progress[0].cmd_seq, 900u);

    ASSERT_EQ(result.lifecycle.size(), 1u);
    EXPECT_TRUE(result.lifecycle[0].success);
}

TEST_F(ExpertLifecycleManagerTest, Progress_DisabledByDefault) {
    auto elm = make_elm();

    elm->ensure_resident({1, 0}, 0, lmem::CacheZone::kStable, 901);

    auto result = poll_elm(*elm);
    EXPECT_TRUE(result.progress.empty());
    ASSERT_EQ(result.lifecycle.size(), 1u);
    EXPECT_TRUE(result.lifecycle[0].success);
}

// ═════════════════════════════════════════════════════════════════════════════
// TD-82a: owned_buf release after H2D completion

TEST_F(ExpertLifecycleManagerTest, TD93a_FP8_OwnedBufReleasedAfterH2D) {
    // TD-93a: FP8 owned_buf IS released after H2D (same as NVFP4).
    // Previously FP8 was permanent; now lazy-pack-and-release.
    lmem::ExpertKey key{1, 0};
    auto buf = std::make_shared<std::vector<std::byte>>(
        static_cast<size_t>(expert_bytes_), std::byte{0xAB});
    model_.layers[1].routed_experts[0][0].owned_buf = buf;
    model_.layers[1].routed_experts[0][0].packed_slot =
        std::span<const std::byte>(buf->data(), buf->size());

    auto elm = make_elm();
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 500);
    auto result = poll_elm(*elm);
    ASSERT_EQ(result.lifecycle.size(), 1u);
    EXPECT_TRUE(result.lifecycle[0].success);

    // FP8 owned_buf released after H2D (TD-93a).
    EXPECT_EQ(model_.layers[1].routed_experts[0][0].owned_buf, nullptr)
        << "FP8 owned_buf should be released after H2D (TD-93a)";
    EXPECT_TRUE(model_.layers[1].routed_experts[0][0].packed_slot.empty())
        << "packed_slot should be cleared after H2D";
}

TEST_F(ExpertLifecycleManagerTest, TD82a_OwnedBufReleasedAfterH2D) {
    // Set dtype to U8 and give it an owned_buf + packed_slot.
    lmem::ExpertKey key{1, 0};
    auto buf = std::make_shared<std::vector<std::byte>>(
        static_cast<size_t>(expert_bytes_), std::byte{0xAB});
    model_.layers[1].routed_experts[0][0].weight.dtype = lmod::SafetensorsDtype::U8;
    model_.layers[1].routed_experts[0][0].owned_buf = buf;
    model_.layers[1].routed_experts[0][0].packed_slot =
        std::span<const std::byte>(buf->data(), buf->size());

    auto elm = make_elm();

    // Prefetch expert to GPU 0.
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 500);

    // Poll completes H2D (null backend is immediate).
    auto result = poll_elm(*elm);
    ASSERT_EQ(result.lifecycle.size(), 1u);
    EXPECT_TRUE(result.lifecycle[0].success);

    // After H2D completion, owned_buf should be released (TD-82a).
    EXPECT_EQ(model_.layers[1].routed_experts[0][0].owned_buf, nullptr)
        << "owned_buf should be released after H2D completion";
}

TEST_F(ExpertLifecycleManagerTest, TD82a_TP2_BufferAliveUntilBothComplete) {
    // With TP=2, both GPUs transfer the same expert. The buffer must stay
    // alive until both transfers complete (shared_ptr refcounting via
    // GpuEntry::host_buf_ref).
    lmem::ExpertKey key{1, 0};
    auto buf = std::make_shared<std::vector<std::byte>>(
        static_cast<size_t>(expert_bytes_), std::byte{0xAB});
    model_.layers[1].routed_experts[0][0].weight.dtype = lmod::SafetensorsDtype::U8;
    model_.layers[1].routed_experts[0][0].owned_buf = buf;
    model_.layers[1].routed_experts[0][0].packed_slot =
        std::span<const std::byte>(buf->data(), buf->size());

    auto elm = make_elm();

    // Prefetch to both GPUs.
    elm->ensure_resident(key, 0, lmem::CacheZone::kStable, 600);
    elm->ensure_resident(key, 1, lmem::CacheZone::kStable, 601);

    // Both complete immediately with null backend.
    auto result = poll_elm(*elm);
    ASSERT_EQ(result.lifecycle.size(), 2u);
    EXPECT_TRUE(result.lifecycle[0].success);
    EXPECT_TRUE(result.lifecycle[1].success);

    // LoadedModel's owned_buf released (first completion resets it).
    EXPECT_EQ(model_.layers[1].routed_experts[0][0].owned_buf, nullptr);

    // The underlying buffer should be freed (only held by our local `buf` now).
    // If GpuEntry host_buf_ref was properly released, use_count == 1 (our local).
    EXPECT_EQ(buf.use_count(), 1)
        << "Buffer should only be held by test-local shared_ptr after all H2D complete";
}
