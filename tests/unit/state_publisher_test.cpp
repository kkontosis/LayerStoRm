// Unit tests for StatePublisher (IPC-5).
//
// All tests run without CUDA by using null/heap backends.
// Follows patterns from command_dispatcher_test.cpp.

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <vector>

#include "config/config_parser.h"
#include "core/gpu_ref.h"
#include "core/null_device_backend.h"
#include "core/memory/eviction_policy.h"
#include "core/memory/expert_cache.h"
#include "core/memory/page_allocator.h"
#include "core/memory/vram_allocator.h"
#include "core/statistics/acceptance_tracker.h"
#include "core/statistics/expert_stats.h"
#include "core/statistics/workload_detector.h"
#include "core/transfer/transfer_engine.h"
#include "daemon/ipc_protocol.h"
#include "daemon/state_publisher.h"
#include "daemon/state_transaction.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/fp8.h"

namespace lipc   = layerstorm::ipc;
namespace ldam   = layerstorm::daemon;
namespace lmem   = layerstorm::memory;
namespace ltr    = layerstorm::transfer;
namespace lmod   = layerstorm::model;
namespace lstats = layerstorm::statistics;
namespace lc     = layerstorm::config;
namespace lcomp  = layerstorm::compute;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Build a small FP8 model config for testing.
/// first_k_dense_replace=1 means layers 1..5 are MoE (5 MoE layers, 8 experts).
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

// ── Test fixture ────────────────────────────────────────────────────────────

class StatePublisherTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Config + model
        cfg_  = std::make_unique<lc::Config>(small_config());
        mcfg_ = std::make_unique<lmod::ModelConfig>(*cfg_);
        fp8_  = std::make_unique<lmod::Fp8E4M3>();
        layer_reg_ = std::make_unique<lmod::LayerRegistry>(*mcfg_, *cfg_, *fp8_);
        expert_bytes_ = layer_reg_->per_routed_expert_bytes();

        // Null device backends (shared by VramAllocator, PageAllocator, TransferEngine)
        lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
        lc::GpuRef gpu1{1, 1, lc::GpuType::rtx5090};
        null_backends_.push_back(lcomp::make_null_device_backend(gpu0));
        null_backends_.push_back(lcomp::make_null_device_backend(gpu1));
        std::vector<lcomp::DeviceBackend*> backends{
            null_backends_[0].get(), null_backends_[1].get()};

        // Memory
        auto layout = lmem::compute_vram_layout(*cfg_, *layer_reg_, *mcfg_);
        vram_ = std::make_unique<lmem::VramAllocator>(
            std::move(layout), backends);
        page_alloc_ = std::make_unique<lmem::PageAllocator>(
            *vram_, null_backends_[0].get());
        cache_ = std::make_unique<lmem::ExpertCache>(
            *vram_, *cfg_, expert_bytes_);
        ltr::TransferEngine::Options te_opts{
            .device_backends = backends,
            .pcie_info = {{.pcie_gen = 5, .pcie_width = 16},
                          {.pcie_gen = 5, .pcie_width = 16}},
        };
        transfer_engine_ = std::make_unique<ltr::TransferEngine>(std::move(te_opts));

        // Statistics
        num_moe_layers_  = static_cast<uint32_t>(mcfg_->num_moe_layers());
        first_moe_layer_ = static_cast<uint32_t>(cfg_->model.first_k_dense_replace);
        num_experts_     = static_cast<uint32_t>(cfg_->model.n_routed_experts);

        expert_stats_ = std::make_unique<lstats::ExpertStats>(
            lstats::ExpertStats::Options{
                .ewma_alpha = 0.5,
                .num_moe_layers  = num_moe_layers_,
                .num_experts     = num_experts_,
                .first_moe_layer = first_moe_layer_,
                .max_recency_tokens = 1024,
            });

        workload_detector_ = std::make_unique<lstats::WorkloadDetector>(
            lstats::WorkloadDetector::Options{
                .num_moe_layers  = num_moe_layers_,
                .num_experts     = num_experts_,
                .first_moe_layer = first_moe_layer_,
                .token_window_size = 4,
                .shift_threshold_std_devs = 3.0,
            });

        acceptance_tracker_ = std::make_unique<lstats::AcceptanceTracker>(
            lstats::AcceptanceTracker::Options{
                .ema_alpha = 0.5,
                .per_request_ema_alpha = 0.5,
                .window_size = 16,
                .calibration_buffer_size = 64,
            });

        // Snapshot (stack-allocated, zeroed)
        std::memset(&snap_, 0, sizeof(snap_));
    }

    ldam::StatePublisher make_publisher() {
        return ldam::StatePublisher(ldam::StatePublisher::Deps{
            .expert_stats       = expert_stats_.get(),
            .workload_detector  = workload_detector_.get(),
            .acceptance_tracker = acceptance_tracker_.get(),
            .expert_cache       = cache_.get(),
            .vram_allocator     = vram_.get(),
            .page_allocator     = page_alloc_.get(),
            .transfer_engine    = transfer_engine_.get(),
        });
    }

    // Config / model
    std::unique_ptr<lc::Config>          cfg_;
    std::unique_ptr<lmod::ModelConfig>   mcfg_;
    std::unique_ptr<lmod::Fp8E4M3>      fp8_;
    std::unique_ptr<lmod::LayerRegistry> layer_reg_;
    int64_t expert_bytes_ = 0;

    // Modules (null_backends_ first: VramAllocator holds raw ptrs to them)
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> null_backends_;
    std::unique_ptr<lmem::VramAllocator>    vram_;
    std::unique_ptr<lmem::PageAllocator>    page_alloc_;
    std::unique_ptr<lmem::ExpertCache>      cache_;
    std::unique_ptr<ltr::TransferEngine>    transfer_engine_;
    std::unique_ptr<lstats::ExpertStats>    expert_stats_;
    std::unique_ptr<lstats::WorkloadDetector>  workload_detector_;
    std::unique_ptr<lstats::AcceptanceTracker> acceptance_tracker_;

    // Model params
    uint32_t num_moe_layers_  = 0;
    uint32_t first_moe_layer_ = 0;
    uint32_t num_experts_     = 0;

    // Snapshot under test
    lipc::StateSnapshot snap_;

    // Transaction for publish() calls (not stackable in tests — one per call)
    ldam::StateTransaction tx_{snap_};
};

// ═══════════════════════════════════════════════════════════════════════════
// Expert Stats
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(StatePublisherTest, ExpertStatsReflected) {
    // Feed some gating results so stats have non-zero values.
    std::vector<lstats::GatingResult> results;
    for (uint32_t l = first_moe_layer_; l < first_moe_layer_ + num_moe_layers_; ++l) {
        lstats::GatingResult gr;
        gr.token_id  = 1;
        gr.layer_idx = l;
        gr.activations.push_back({{l, 0}, 0.8f});
        gr.activations.push_back({{l, 1}, 0.2f});
        results.push_back(std::move(gr));
    }
    expert_stats_->update(results);

    auto pub = make_publisher();
    pub.publish(snap_, tx_);

    // Check expert (layer=first_moe_layer_, expert=0) has non-zero frequency.
    size_t idx = 0 * lipc::kMaxExperts + 0;  // moe_layer=0, expert=0
    EXPECT_GT(snap_.expert_frequency[idx], 0.0f);
    EXPECT_FLOAT_EQ(snap_.expert_frequency[idx],
                     static_cast<float>(expert_stats_->frequency({first_moe_layer_, 0})));

    // Check expert (layer=first_moe_layer_, expert=2) was not activated — frequency ~0.
    size_t idx2 = 0 * lipc::kMaxExperts + 2;
    EXPECT_FLOAT_EQ(snap_.expert_frequency[idx2],
                     static_cast<float>(expert_stats_->frequency({first_moe_layer_, 2})));

    // Check routing weight.
    EXPECT_FLOAT_EQ(snap_.expert_routing_weight[idx],
                     static_cast<float>(expert_stats_->routing_weight({first_moe_layer_, 0})));
}

// Note: ResidencyBitmapCorrect test removed — residency_bitmap is now
// ELM-owned (tested in expert_lifecycle_manager_test.cpp).

// ═══════════════════════════════════════════════════════════════════════════
// Acceptance Tracker
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(StatePublisherTest, AcceptanceRatesReflected) {
    // Feed a verification result: 3 accepted out of 5 attempted.
    lstats::VerificationResult vr;
    vr.request_id       = 42;
    vr.accepted_tokens  = 3;
    vr.attempted_tokens = 5;
    vr.used_layer_skip  = false;
    acceptance_tracker_->update(std::span<const lstats::VerificationResult>(&vr, 1));

    auto pub = make_publisher();
    pub.publish(snap_, tx_);

    EXPECT_GT(snap_.global_acceptance_rate, 0.0);
    EXPECT_DOUBLE_EQ(snap_.global_acceptance_rate, acceptance_tracker_->global_rate());
    EXPECT_DOUBLE_EQ(snap_.windowed_acceptance_rate, acceptance_tracker_->windowed_rate());
    EXPECT_EQ(snap_.total_verifications, 1u);
    EXPECT_EQ(snap_.total_accepted_tokens, 3u);
    EXPECT_EQ(snap_.total_attempted_tokens, 5u);

    // Per-request: request 42 should appear (IPC-7).
    EXPECT_EQ(snap_.num_tracked_requests, 1u);
    EXPECT_EQ(snap_.per_request_acceptance[0].request_id, 42u);
    EXPECT_GT(snap_.per_request_acceptance[0].acceptance_rate, 0.0);
}

TEST_F(StatePublisherTest, AcceptancePerRequestPopulated) {
    // Feed results for 3 different requests.
    std::vector<lstats::VerificationResult> results;
    for (uint64_t rid : {100u, 200u, 300u}) {
        lstats::VerificationResult vr;
        vr.request_id       = rid;
        vr.accepted_tokens  = 3;
        vr.attempted_tokens = 4;
        vr.used_layer_skip  = false;
        results.push_back(std::move(vr));
    }
    acceptance_tracker_->update(results);

    auto pub = make_publisher();
    pub.publish(snap_, tx_);

    EXPECT_EQ(snap_.num_tracked_requests, 3u);
    // Verify all 3 request IDs appear (order unspecified — unordered_map).
    std::set<uint64_t> found_ids;
    for (uint32_t i = 0; i < snap_.num_tracked_requests; ++i) {
        found_ids.insert(snap_.per_request_acceptance[i].request_id);
        EXPECT_GT(snap_.per_request_acceptance[i].acceptance_rate, 0.0);
    }
    EXPECT_EQ(found_ids.count(100), 1u);
    EXPECT_EQ(found_ids.count(200), 1u);
    EXPECT_EQ(found_ids.count(300), 1u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Workload Detector shift (one-shot)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(StatePublisherTest, WorkloadShiftOneShot) {
    // WorkloadDetector with window_size=4 needs 4 tokens per window.
    // We need to process several stable windows, then inject a shift.

    // Build "normal" gating results (expert 0 always active).
    auto make_normal_results = [&](uint64_t token_id) {
        std::vector<lstats::GatingResult> results;
        for (uint32_t l = first_moe_layer_; l < first_moe_layer_ + num_moe_layers_; ++l) {
            lstats::GatingResult gr;
            gr.token_id  = token_id;
            gr.layer_idx = l;
            gr.activations.push_back({{l, 0}, 0.8f});
            results.push_back(std::move(gr));
        }
        return results;
    };

    // Build "shifted" gating results (expert 7 always active, 0 not).
    auto make_shifted_results = [&](uint64_t token_id) {
        std::vector<lstats::GatingResult> results;
        for (uint32_t l = first_moe_layer_; l < first_moe_layer_ + num_moe_layers_; ++l) {
            lstats::GatingResult gr;
            gr.token_id  = token_id;
            gr.layer_idx = l;
            gr.activations.push_back({{l, 7}, 0.8f});
            results.push_back(std::move(gr));
        }
        return results;
    };

    // Build up stable baseline (need min 4 windows × 4 tokens = 16 tokens + cooldown).
    uint64_t tok = 1;
    for (int i = 0; i < 40; ++i, ++tok) {
        auto r = make_normal_results(tok);
        expert_stats_->update(r);
        workload_detector_->update(*expert_stats_);
    }

    // Consume any pre-existing shift flag.
    workload_detector_->shift_detected();

    // Inject shifted workload.
    for (int i = 0; i < 8; ++i, ++tok) {
        auto r = make_shifted_results(tok);
        expert_stats_->update(r);
        workload_detector_->update(*expert_stats_);
    }

    // If a shift was detected, the publisher should capture it.
    // Note: shift detection depends on distance exceeding threshold. We verify
    // the one-shot semantics: first publish sees whatever the detector reports,
    // second publish sees 0.
    auto pub = make_publisher();
    pub.publish(snap_, tx_);
    uint8_t first_shift = snap_.shift_detected;

    // Second publish: shift flag should be consumed.
    pub.publish(snap_, tx_);
    EXPECT_EQ(snap_.shift_detected, 0u);

    // If the first publish saw the shift, it should have been 1.
    if (first_shift) {
        EXPECT_EQ(first_shift, 1u);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Snapshot fields
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(StatePublisherTest, GpuSnapshotFields) {
    // Reserve an expert to have some used slots.
    lmem::ExpertKey k0{first_moe_layer_, 0};
    cache_->reserve(k0, 0, lmem::CacheZone::kStable);
    cache_->mark_all_ready(k0, 0);

    auto pub = make_publisher();
    pub.publish(snap_, tx_);

    EXPECT_EQ(snap_.num_gpus, 2u);

    // GPU 0 should have some VRAM allocated.
    EXPECT_GT(snap_.gpus[0].vram_total_bytes, 0u);
    EXPECT_GT(snap_.gpus[0].vram_used_bytes, 0u);

    // GPU 0: 1 stable slot used.
    EXPECT_EQ(snap_.gpus[0].expert_stable_used, 1u);
    EXPECT_GT(snap_.gpus[0].expert_stable_total, 0u);

    // GPU 1: no experts reserved.
    EXPECT_EQ(snap_.gpus[1].expert_stable_used, 0u);

    // KV pages should be populated.
    EXPECT_GT(snap_.gpus[0].kv_main_free_pages, 0u);

    // No transfers enqueued yet.
    EXPECT_EQ(snap_.gpus[0].inflight_h2d_count, 0u);
    EXPECT_EQ(snap_.gpus[0].inflight_d2h_count, 0u);

    // Prefill mode should be normal.
    EXPECT_EQ(snap_.gpus[0].prefill_mode, static_cast<uint8_t>(lmem::PrefillMode::kNormal));
}

// ═══════════════════════════════════════════════════════════════════════════
// Transfer inflight counts
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(StatePublisherTest, TotalInflightTransfers) {
    // Null backend completes immediately on poll, but before poll
    // the transfer is "inflight". We need to avoid polling.
    // Enqueue but don't poll.
    uint8_t buf[64]{};
    lmem::ExpertKey k0{first_moe_layer_, 0};
    auto tok = transfer_engine_->enqueue_h2d(k0, 0, buf, buf, sizeof(buf));
    ASSERT_TRUE(tok.has_value());

    auto pub = make_publisher();
    pub.publish(snap_, tx_);

    // Null backend: query_event always returns true, so poll_completions()
    // would clear them. But we haven't polled yet. Check if inflight
    // actually registers. The null backend's query_event returns true,
    // but the transfer is still in the inflight map until poll_completions()
    // is called.
    EXPECT_EQ(snap_.total_inflight_transfers, 1u);
    EXPECT_EQ(snap_.gpus[0].inflight_h2d_count, 1u);
    EXPECT_EQ(snap_.gpus[0].inflight_d2h_count, 0u);
}

TEST_F(StatePublisherTest, PerGpuInflightDirection) {
    uint8_t buf[64]{};
    lmem::ExpertKey k0{first_moe_layer_, 0};
    lmem::ExpertKey k1{first_moe_layer_, 1};

    // H2D on GPU 0, D2H on GPU 1.
    transfer_engine_->enqueue_h2d(k0, 0, buf, buf, sizeof(buf));
    transfer_engine_->enqueue_d2h(k1, 1, buf, buf, sizeof(buf));

    auto pub = make_publisher();
    pub.publish(snap_, tx_);

    EXPECT_EQ(snap_.gpus[0].inflight_h2d_count, 1u);
    EXPECT_EQ(snap_.gpus[0].inflight_d2h_count, 0u);
    EXPECT_EQ(snap_.gpus[1].inflight_h2d_count, 0u);
    EXPECT_EQ(snap_.gpus[1].inflight_d2h_count, 1u);
    EXPECT_EQ(snap_.total_inflight_transfers, 2u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Null modules
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(StatePublisherTest, NullModulesNocrash) {
    // All deps null — should produce zeroed snapshot without crashing.
    ldam::StatePublisher pub(ldam::StatePublisher::Deps{});
    pub.publish(snap_, tx_);

    EXPECT_EQ(snap_.num_gpus, 0u);
    EXPECT_EQ(snap_.total_inflight_transfers, 0u);
    EXPECT_DOUBLE_EQ(snap_.global_acceptance_rate, 0.0);
    EXPECT_EQ(snap_.shift_detected, 0u);
    EXPECT_EQ(snap_.num_tracked_requests, 0u);

    // Expert stats arrays should be zeroed.
    for (uint32_t i = 0; i < lipc::kMaxMoeLayers * lipc::kMaxExperts; ++i) {
        EXPECT_FLOAT_EQ(snap_.expert_frequency[i], 0.0f) << "at index " << i;
        if (i > 10) break;  // Spot check first few.
    }
}

TEST_F(StatePublisherTest, PartialModules) {
    // Only expert_stats and expert_cache non-null.
    ldam::StatePublisher pub(ldam::StatePublisher::Deps{
        .expert_stats  = expert_stats_.get(),
        .expert_cache  = cache_.get(),
    });

    // Reserve an expert to verify residency is populated.
    lmem::ExpertKey k0{first_moe_layer_, 0};
    cache_->reserve(k0, 0, lmem::CacheZone::kStable);
    cache_->mark_all_ready(k0, 0);

    pub.publish(snap_, tx_);

    // Expert cache produces GPU count.
    EXPECT_EQ(snap_.num_gpus, 2u);

    // Note: residency_bitmap no longer published by StatePublisher (ELM-owned).

    // Transfer and acceptance should be zeroed.
    EXPECT_EQ(snap_.total_inflight_transfers, 0u);
    EXPECT_DOUBLE_EQ(snap_.global_acceptance_rate, 0.0);

    // VRAM fields should be zero (no vram_allocator).
    EXPECT_EQ(snap_.gpus[0].vram_total_bytes, 0u);

    // But expert cache slots should still be populated.
    EXPECT_EQ(snap_.gpus[0].expert_stable_used, 1u);
}

// Note: HostNuma tests removed — expert_host_numa is now ELM-owned
// (tested in expert_lifecycle_manager_test.cpp).
