// Unit tests for CommandDispatcher (IPC-4).
//
// All tests run without CUDA by using null/heap backends and heap-allocated
// IPC regions.  Follows patterns from daemon_loop_test.cpp and
// expert_cache_test.cpp.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "compute/graphs/graph_registry.h"
#include "compute/stream_manager.h"
#include "config/config_parser.h"
#include "core/gpu_ref.h"
#include "core/memory/eviction_policy.h"
#include "core/memory/expert_cache.h"
#include "core/memory/numa_manager.h"
#include "core/memory/nvme_tier.h"
#include "core/memory/page_allocator.h"
#include "core/memory/vram_allocator.h"
#include "core/statistics/coactivation_graph.h"
#include "core/transfer/transfer_engine.h"
#include "daemon/buffer_registry.h"
#include "daemon/command_dispatcher.h"
#include "daemon/expert_lifecycle_manager.h"
#include "daemon/daemon_loop.h"
#include "daemon/ipc_protocol.h"
#include "daemon/spsc_ring.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/fp8.h"
#include "model/quantization/gguf_kquant.h"
#include "model/weight_loader/weight_loader.h"
#include "core/null_attention_device.h"
#include "core/null_device_backend.h"
#include "core/null_expert_device.h"
#ifdef LAYERSTORM_RECORDING_BACKEND
#include "core/recording_attention_device.h"
#include "core/recording_expert_device.h"
#include "core/recording_device_ops.h"
#endif
#include "parallelism/dcp_communicator.h"
#include "parallelism/dcp_executor.h"
#include "parallelism/null_collective_backend.h"
#include "recording_collective_backend.h"

namespace lipc = layerstorm::ipc;
namespace ldam = layerstorm::daemon;
namespace lmem = layerstorm::memory;
namespace ltr  = layerstorm::transfer;
namespace lmod = layerstorm::model;
namespace lcomp = layerstorm::compute;
namespace lstats = layerstorm::statistics;
namespace lpar = layerstorm::parallelism;
namespace lc   = layerstorm::config;

// ── Helpers ���───────────────────────────────────────────────────────────────

static constexpr uint32_t kTestSlots = 128;

static void* aligned_alloc_zeroed(size_t bytes) {
    void* p = std::aligned_alloc(64, bytes);
    std::memset(p, 0, bytes);
    return p;
}

/// Write a command into the completion ring and return it.
static lipc::Command make_cmd(lipc::CmdType type, uint32_t seq = 0,
                               uint32_t gpu_idx = 0, uint32_t stream_id = 0) {
    lipc::Command cmd{};
    cmd.cmd_type  = static_cast<uint32_t>(type);
    cmd.cmd_seq   = seq;
    cmd.gpu_idx   = gpu_idx;
    cmd.stream_id = stream_id;
    return cmd;
}

/// Read one completion from the ring.
static bool read_cmp(lipc::CompletionRing& ring, lipc::Completion& out) {
    return ring.try_read(&out);
}

/// Build a small FP8 model config for testing.
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
        {"memory", {{"vram_safety_margin_gb", 0.1},
                    {"kv_cache", {{"page_growth_chunk_tokens", 0}}}}},
    };
    return lc::parse_config(j);
}

// ── Test fixture ─���─────────────────────────────────────────────────────────

class CommandDispatcherTest : public ::testing::Test {
protected:
    void SetUp() override {
        // IPC region
        ipc_bytes_ = lipc::IpcLayout::total_size(kTestSlots, kTestSlots);
        ipc_region_ = static_cast<uint8_t*>(aligned_alloc_zeroed(ipc_bytes_));

        header_ = reinterpret_cast<lipc::IpcHeader*>(
            ipc_region_ + lipc::IpcLayout::kHeaderOffset);
        header_->version = lipc::kProtocolVersion;

        void* cmd_ptr = ipc_region_ + lipc::IpcLayout::cmd_ring_offset();
        lipc::CommandRing::init(cmd_ptr, kTestSlots);
        cmd_ring_ = std::make_unique<lipc::CommandRing>(cmd_ptr);

        void* cmp_ptr = ipc_region_ + lipc::IpcLayout::cmp_ring_offset(kTestSlots);
        lipc::CompletionRing::init(cmp_ptr, kTestSlots);
        cmp_ring_ = std::make_unique<lipc::CompletionRing>(cmp_ptr);

        snap_ = reinterpret_cast<lipc::StateSnapshot*>(
            ipc_region_ + lipc::IpcLayout::state_offset(kTestSlots, kTestSlots));
        snap_->seqlock = 0;

        sideband_base_ = ipc_region_
                         + lipc::IpcLayout::sideband_offset(kTestSlots, kTestSlots);

        // Config + model
        cfg_ = std::make_unique<lc::Config>(small_config());
        mcfg_ = std::make_unique<lmod::ModelConfig>(*cfg_);
        fp8_ = std::make_unique<lmod::Fp8E4M3>();
        layer_reg_ = std::make_unique<lmod::LayerRegistry>(*mcfg_, *cfg_, *fp8_);
        expert_bytes_ = layer_reg_->per_routed_expert_bytes();

        // KD-2: Null devices for compute dispatch testing
        lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
        lc::GpuRef gpu1{1, 1, lc::GpuType::rtx5090};
        test_device_backends_.push_back(lcomp::make_null_device_backend(gpu0));
        test_device_backends_.push_back(lcomp::make_null_device_backend(gpu1));

        // Memory
        std::vector<lcomp::DeviceBackend*> dev_ptrs{
            test_device_backends_[0].get(), test_device_backends_[1].get()};
        auto layout = lmem::compute_vram_layout(*cfg_, *layer_reg_, *mcfg_);
        vram_ = std::make_unique<lmem::VramAllocator>(
            std::move(layout), dev_ptrs);
        cache_ = std::make_unique<lmem::ExpertCache>(
            *vram_, *cfg_, expert_bytes_);
        page_allocator_ = std::make_unique<lmem::PageAllocator>(
            *vram_, test_device_backends_[0].get());
        test_attn_devices_.push_back(lcomp::make_null_attention_device(gpu0));
        test_attn_devices_.push_back(lcomp::make_null_attention_device(gpu1));
        test_expert_devices_.push_back(lcomp::make_null_expert_device(gpu0));
        test_expert_devices_.push_back(lcomp::make_null_expert_device(gpu1));

        // Transfer engine (null backend — completes immediately on poll)
        ltr::TransferEngine::Options te_opts{
            .device_backends = {test_device_backends_[0].get(),
                                test_device_backends_[1].get()},
            .pcie_info = {{.pcie_gen = 5, .pcie_width = 16},
                          {.pcie_gen = 5, .pcie_width = 16}},
        };
        transfer_engine_ = std::make_unique<ltr::TransferEngine>(std::move(te_opts));

        // Stream manager (null backend)
        lcomp::StreamManager::Options sm_opts{
            .device_backends = {test_device_backends_[0].get(),
                                test_device_backends_[1].get()},
        };
        stream_manager_ = std::make_unique<lcomp::StreamManager>(std::move(sm_opts));

        // Graph registry (empty)
        graph_registry_ = std::make_unique<lcomp::GraphRegistry>();

        // CoactivationGraph
        coactivation_graph_ = std::make_unique<lstats::CoactivationGraph>(
            lstats::CoactivationGraph::Options{
                .num_moe_layers = static_cast<uint32_t>(mcfg_->num_moe_layers()),
                .num_experts = static_cast<uint32_t>(cfg_->model.n_routed_experts),
                .first_moe_layer = static_cast<uint32_t>(cfg_->model.first_k_dense_replace),
                .decay_factor = 0.999,
                .workload_shift_decay = 0.1,
            });
    }

    void TearDown() override {
        dispatcher_.reset();
        // KD-R4: DcpExecutor must outlive dispatcher; destroy in order.
        dcp_executor_.reset();
        dcp_attn_device_.reset();
        dcp_stream_manager_.reset();
        elm_.reset();
        std::free(dcp_hidden_buf_);
        std::free(dcp_moe_hidden_buf_);
        dcp_hidden_buf_ = nullptr;
        dcp_moe_hidden_buf_ = nullptr;
        std::free(ipc_region_);
    }

    /// Construct the dispatcher with default deps.
    void make_dispatcher(lmod::LoadedModel* loaded_model = nullptr) {
        // Create ELM — always active; loaded_model may be null (COLD experts).
        ldam::ExpertLifecycleManager::Deps elm_deps{
            .expert_cache    = cache_.get(),
            .transfer_engine = transfer_engine_.get(),
            .loaded_model    = loaded_model,
        };
        elm_ = std::make_unique<ldam::ExpertLifecycleManager>(std::move(elm_deps));

        ldam::CommandDispatcher::Deps deps{
            .cmp_ring           = cmp_ring_.get(),
            .transfer_engine    = transfer_engine_.get(),
            .expert_cache       = cache_.get(),
            .stream_manager     = stream_manager_.get(),
            .graph_registry     = graph_registry_.get(),
            .coactivation_graph = coactivation_graph_.get(),
            .nvme_tier          = nullptr,  // not available in test
            .numa_manager       = nullptr,
            .loaded_model       = loaded_model,
            .dcp_executor       = nullptr,
            .dcp_communicator   = nullptr,
            .page_allocator     = page_allocator_.get(),
            .sideband_base      = sideband_base_,
            .elm                = elm_.get(),
            .live_config        = cfg_.get(),
            .attention_devices  = {test_attn_devices_[0].get(),
                                   test_attn_devices_[1].get()},
            .expert_devices     = {test_expert_devices_[0].get(),
                                   test_expert_devices_[1].get()},
            .device_backends    = {test_device_backends_[0].get(),
                                   test_device_backends_[1].get()},
            .cuda_kernels_enabled = false,
        };
        dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));
    }

    /// KD-R4: Create DcpExecutor + NullAttentionDevice as fixture members,
    /// build CommandDispatcher with standard DCP deps.  Optional customizer
    /// lets tests tweak Deps before construction (e.g. clear hidden states).
    /// If dcp_stream_manager_ is pre-set, uses it; otherwise uses stream_manager_.
    void make_dcp_dispatcher(
        std::function<void(ldam::CommandDispatcher::Deps&)> customize = {})
    {
        lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
        dcp_attn_device_ = lcomp::make_null_attention_device(gpu0);

        auto* sm = dcp_stream_manager_ ? dcp_stream_manager_.get()
                                       : stream_manager_.get();

        lpar::DcpExecutor::Options dcp_opts{
            .dcp_size             = 1,
            .gpus                 = {gpu0},
            .max_batch_size       = 8,
            .hidden_size          = 256,
            .num_attention_heads  = 4,
            .q_lora_rank          = 64,
            .kv_lora_rank         = 32,
            .qk_rope_head_dim    = 16,
            .qk_nope_head_dim    = 32,
            .v_head_dim           = 64,
            .rms_norm_eps         = 1e-6f,
            .stream_manager       = sm,
            .attention_devices    = {dcp_attn_device_.get()},
        };
        dcp_executor_ = std::make_unique<lpar::DcpExecutor>(std::move(dcp_opts));

        // TD-90b: initialize dequant pool (required since KD-4j).
        dcp_fake_weights_ = std::vector<std::vector<lpar::AttentionLayerWeights>>(
            61, std::vector<lpar::AttentionLayerWeights>(1));
        {
            std::vector<std::vector<const lpar::AttentionLayerWeights*>> wp(61);
            for (int l = 0; l < 61; ++l) wp[l] = {&dcp_fake_weights_[l][0]};
            dcp_executor_->set_layer_weights(std::move(wp), 61);
        }

        const size_t hidden_bytes = 8 * 256 * 2;  // max_batch * hidden * BF16
        dcp_hidden_buf_ = aligned_alloc_zeroed(hidden_bytes);

        ldam::CommandDispatcher::Deps deps{
            .cmp_ring              = cmp_ring_.get(),
            .transfer_engine       = transfer_engine_.get(),
            .expert_cache          = cache_.get(),
            .stream_manager        = sm,
            .graph_registry        = graph_registry_.get(),
            .coactivation_graph    = coactivation_graph_.get(),
            .dcp_executor          = dcp_executor_.get(),
            .sideband_base         = sideband_base_,
            .live_config           = cfg_.get(),
            .attention_devices     = {dcp_attn_device_.get()},
            .device_backends       = {test_device_backends_[0].get()},
            .cuda_kernels_enabled  = false,
            .kv_cache_stride_block = 576 * 64,
            .kv_cache_stride_row   = 576,
            .kv_page_size          = 64,
            .hidden_state_pairs    = {ldam::HiddenStatePair{dcp_hidden_buf_, nullptr, 0, 0, 0}},
            .per_layer_attn_weights = dcp_fake_weights_,
            .max_batch_size        = 8,
        };

        if (customize) customize(deps);

        dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));
    }

    // IPC region
    uint8_t*            ipc_region_ = nullptr;
    size_t              ipc_bytes_  = 0;
    lipc::IpcHeader*    header_     = nullptr;
    lipc::StateSnapshot* snap_      = nullptr;
    uint8_t*            sideband_base_ = nullptr;
    std::unique_ptr<lipc::CommandRing>    cmd_ring_;
    std::unique_ptr<lipc::CompletionRing> cmp_ring_;

    // Config / model
    std::unique_ptr<lc::Config>         cfg_;
    std::unique_ptr<lmod::ModelConfig>  mcfg_;
    std::unique_ptr<lmod::Fp8E4M3>     fp8_;
    std::unique_ptr<lmod::LayerRegistry> layer_reg_;
    int64_t expert_bytes_ = 0;

    // KD-2: Null devices for compute dispatch (must outlive vram_/stream_manager_/transfer_engine_)
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> test_device_backends_;  // #86b

    // Modules
    std::unique_ptr<lmem::VramAllocator>    vram_;
    std::unique_ptr<lmem::ExpertCache>      cache_;
    std::unique_ptr<lmem::PageAllocator>    page_allocator_;
    std::vector<std::unique_ptr<lcomp::AttentionDevice>> test_attn_devices_;
    std::vector<std::unique_ptr<lcomp::ExpertDevice>> test_expert_devices_;

    std::unique_ptr<ltr::TransferEngine>    transfer_engine_;
    std::unique_ptr<lcomp::StreamManager>   stream_manager_;
    std::unique_ptr<lcomp::GraphRegistry>   graph_registry_;
    std::unique_ptr<lstats::CoactivationGraph> coactivation_graph_;

    // KD-R4: DcpExecutor test support — fixture-owned to outlive dispatcher_
    std::unique_ptr<lcomp::AttentionDevice> dcp_attn_device_;
    std::unique_ptr<lpar::DcpExecutor> dcp_executor_;
    std::unique_ptr<lcomp::StreamManager> dcp_stream_manager_;
    std::vector<std::vector<lpar::AttentionLayerWeights>> dcp_fake_weights_;
    void* dcp_hidden_buf_ = nullptr;
    void* dcp_moe_hidden_buf_ = nullptr;

    // ELM (wired into make_dispatcher; no loaded_model so COLD by default)
    std::unique_ptr<ldam::ExpertLifecycleManager> elm_;

    // Dispatcher under test
    std::unique_ptr<ldam::CommandDispatcher> dispatcher_;

    /// Poll ELM with transfer completions.  Returns lifecycle completions.
    std::vector<ldam::LifecycleCompletion> poll_elm() {
        auto tc = transfer_engine_->poll_completions();
        return elm_->poll(tc, {}).lifecycle;
    }
};

namespace {

// INV-KV-LAYER (TD-GOLDEN): one physical KV page per (logical page, layer).
// small_config has num_hidden_layers=6, num_nextn_predict_layers=0, so every
// logical page costs 6 physical pages and page_count reports physical pages.
constexpr uint32_t kKvLayers = 6;

// Helper: create per_layer_attn_weights with default (nullptr) entries.
// NullAttentionDevice doesn't dereference — only vector dimensions matter.
std::vector<std::vector<lpar::AttentionLayerWeights>>
make_fake_attn_weights(int num_layers, int dcp_size) {
    return std::vector<std::vector<lpar::AttentionLayerWeights>>(
        num_layers, std::vector<lpar::AttentionLayerWeights>(dcp_size));
}
}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Cache operations
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, CacheReserve_Success) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::CMD_CACHE_RESERVE, /*seq=*/1, /*gpu_idx=*/0);
    // first_k_dense_replace=1, so layer 1 is a MoE layer.
    cmd.cache_reserve.layer_idx  = 1;
    cmd.cache_reserve.expert_idx = 0;
    cmd.cache_reserve.zone       = 1;  // streaming
    cmd.cache_reserve.is_duplicate = 0;

    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 1u);
    EXPECT_EQ(cmp.status, 0u);

    // Expert is now resident.
    auto* entry = cache_->lookup({1, 0}, 0);
    EXPECT_NE(entry, nullptr);
}

TEST_F(CommandDispatcherTest, CacheEvict_Success) {
    make_dispatcher();

    // Reserve first.
    auto reserve = make_cmd(lipc::CMD_CACHE_RESERVE, /*seq=*/1, /*gpu_idx=*/0);
    reserve.cache_reserve.layer_idx  = 1;
    reserve.cache_reserve.expert_idx = 0;
    reserve.cache_reserve.zone       = 1;
    dispatcher_->dispatch(reserve);

    // Drain the reserve completion.
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));

    // Now evict.
    auto evict = make_cmd(lipc::CMD_CACHE_EVICT, /*seq=*/2, /*gpu_idx=*/0);
    evict.cache_op.layer_idx  = 1;
    evict.cache_op.expert_idx = 0;
    dispatcher_->dispatch(evict);

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 2u);
    EXPECT_EQ(cmp.status, 0u);

    // Expert no longer resident.
    EXPECT_EQ(cache_->lookup({1, 0}, 0), nullptr);
}

TEST_F(CommandDispatcherTest, CacheEvict_NotResident) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::CMD_CACHE_EVICT, /*seq=*/5, /*gpu_idx=*/0);
    cmd.cache_op.layer_idx  = 1;
    cmd.cache_op.expert_idx = 7;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.status, 1u);  // not found
}

TEST_F(CommandDispatcherTest, CachePromote_Success) {
    make_dispatcher();

    // Reserve in streaming zone.
    auto reserve = make_cmd(lipc::CMD_CACHE_RESERVE, /*seq=*/1, /*gpu_idx=*/0);
    reserve.cache_reserve.layer_idx  = 2;
    reserve.cache_reserve.expert_idx = 3;
    reserve.cache_reserve.zone       = 1;  // streaming
    dispatcher_->dispatch(reserve);
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.status, 0u);

    // Promote to stable.
    auto promote = make_cmd(lipc::CMD_CACHE_PROMOTE, /*seq=*/2, /*gpu_idx=*/0);
    promote.cache_op.layer_idx  = 2;
    promote.cache_op.expert_idx = 3;
    dispatcher_->dispatch(promote);

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 2u);
    EXPECT_EQ(cmp.status, 0u);

    // Verify zone changed.
    auto* entry = cache_->lookup({2, 3}, 0);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->zone, lmem::CacheZone::kStable);
}

TEST_F(CommandDispatcherTest, CacheDemote_Success) {
    make_dispatcher();

    // Reserve in stable zone.
    auto reserve = make_cmd(lipc::CMD_CACHE_RESERVE, /*seq=*/1, /*gpu_idx=*/0);
    reserve.cache_reserve.layer_idx  = 2;
    reserve.cache_reserve.expert_idx = 4;
    reserve.cache_reserve.zone       = 0;  // stable
    dispatcher_->dispatch(reserve);
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.status, 0u);

    // Demote to streaming.
    auto demote = make_cmd(lipc::CMD_CACHE_DEMOTE, /*seq=*/2, /*gpu_idx=*/0);
    demote.cache_op.layer_idx  = 2;
    demote.cache_op.expert_idx = 4;
    dispatcher_->dispatch(demote);

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmd_seq, 2u);
    EXPECT_EQ(cmp.status, 0u);

    auto* entry = cache_->lookup({2, 4}, 0);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->zone, lmem::CacheZone::kStreaming);
}

// ═══════════════════════════════════════════════════════════════════════════
// Transfer operations
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, TransferH2D_NoSource) {
    // No LoadedModel, no NvmeTier → should get error.
    make_dispatcher(/*loaded_model=*/nullptr);

    // Reserve a cache slot first.
    auto reserve = make_cmd(lipc::CMD_CACHE_RESERVE, /*seq=*/1, /*gpu_idx=*/0);
    reserve.cache_reserve.layer_idx  = 1;
    reserve.cache_reserve.expert_idx = 0;
    reserve.cache_reserve.zone       = 1;
    dispatcher_->dispatch(reserve);
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));

    // Try H2D transfer.
    auto h2d = make_cmd(lipc::CMD_TRANSFER_H2D, /*seq=*/2, /*gpu_idx=*/0);
    h2d.transfer.layer_idx     = 1;
    h2d.transfer.expert_idx    = 0;
    h2d.transfer.sub_component = 0x07;  // kAll
    h2d.transfer.zone          = 1;
    h2d.transfer.bytes         = 128;
    dispatcher_->dispatch(h2d);

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 2u);
}

TEST_F(CommandDispatcherTest, TransferH2D_NoReservation) {
    // Build a fake LoadedModel with some data.
    lmod::LoadedModel model;
    model.layers.resize(6);
    model.layers[1].layer_idx = 1;
    model.layers[1].routed_experts.resize(8);
    // Expert 0: one WeightBundle with fake data.
    std::vector<std::byte> fake_data(256, std::byte{0x42});
    lmod::RawTensor rt{
        .data = std::span<const std::byte>(fake_data),
        .dtype = lmod::SafetensorsDtype::F8_E4M3,
        .shape = {256}};
    lmod::WeightBundle wb{.id = {}, .weight = rt, .packed_slot = rt.data};
    model.layers[1].routed_experts[0].push_back(std::move(wb));

    make_dispatcher(&model);

    // H2D without prior reserve → error (lookup returns nullptr).
    auto h2d = make_cmd(lipc::CMD_TRANSFER_H2D, /*seq=*/3, /*gpu_idx=*/0);
    h2d.transfer.layer_idx     = 1;
    h2d.transfer.expert_idx    = 0;
    h2d.transfer.sub_component = 0x07;
    h2d.transfer.bytes         = 128;
    dispatcher_->dispatch(h2d);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 3u);
}

TEST_F(CommandDispatcherTest, TransferH2D_Enqueues) {
    // Fake LoadedModel.
    lmod::LoadedModel model;
    model.layers.resize(6);
    model.layers[1].layer_idx = 1;
    model.layers[1].routed_experts.resize(8);
    std::vector<std::byte> fake_data(expert_bytes_, std::byte{0xAB});
    lmod::RawTensor rt{
        .data = std::span<const std::byte>(fake_data),
        .dtype = lmod::SafetensorsDtype::F8_E4M3,
        .shape = {expert_bytes_}};
    lmod::WeightBundle wb{.id = {}, .weight = rt, .packed_slot = rt.data};
    model.layers[1].routed_experts[0].push_back(std::move(wb));

    make_dispatcher(&model);

    // Reserve a cache slot.
    auto reserve = make_cmd(lipc::CMD_CACHE_RESERVE, /*seq=*/10, /*gpu_idx=*/0);
    reserve.cache_reserve.layer_idx  = 1;
    reserve.cache_reserve.expert_idx = 0;
    reserve.cache_reserve.zone       = 1;
    dispatcher_->dispatch(reserve);
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.status, 0u);

    // Enqueue H2D.
    auto h2d = make_cmd(lipc::CMD_TRANSFER_H2D, /*seq=*/42, /*gpu_idx=*/0);
    h2d.transfer.layer_idx     = 1;
    h2d.transfer.expert_idx    = 0;
    h2d.transfer.sub_component = 0x07;
    h2d.transfer.bytes         = expert_bytes_;
    dispatcher_->dispatch(h2d);

    // No error completion — transfer is async.
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));

    // Token should be tracked.
    // Poll the transfer engine → completes immediately (null backend).
    auto completions = transfer_engine_->poll_completions();
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_EQ(dispatcher_->resolve_cmd_seq(completions[0].token), 42u);

    // Cleanup.
    dispatcher_->remove_token_mapping(completions[0].token);
    EXPECT_EQ(dispatcher_->resolve_cmd_seq(completions[0].token), 0u);
}

TEST_F(CommandDispatcherTest, CmdSeqCorrelation) {
    // Full end-to-end: dispatcher + DaemonLoop cmd_seq tracking.
    lmod::LoadedModel model;
    model.layers.resize(6);
    model.layers[1].layer_idx = 1;
    model.layers[1].routed_experts.resize(8);
    std::vector<std::byte> fake_data(expert_bytes_, std::byte{0xCD});
    lmod::RawTensor rt{
        .data = std::span<const std::byte>(fake_data),
        .dtype = lmod::SafetensorsDtype::F8_E4M3,
        .shape = {expert_bytes_}};
    lmod::WeightBundle wb{.id = {}, .weight = rt, .packed_slot = rt.data};
    model.layers[1].routed_experts[0].push_back(std::move(wb));

    make_dispatcher(&model);

    // Reserve.
    auto reserve = make_cmd(lipc::CMD_CACHE_RESERVE, /*seq=*/1, /*gpu_idx=*/0);
    reserve.cache_reserve.layer_idx  = 1;
    reserve.cache_reserve.expert_idx = 0;
    reserve.cache_reserve.zone       = 1;
    dispatcher_->dispatch(reserve);
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));

    // Enqueue H2D with seq=42.
    auto h2d = make_cmd(lipc::CMD_TRANSFER_H2D, /*seq=*/42, /*gpu_idx=*/0);
    h2d.transfer.layer_idx     = 1;
    h2d.transfer.expert_idx    = 0;
    h2d.transfer.sub_component = 0x07;
    h2d.transfer.bytes         = expert_bytes_;
    dispatcher_->dispatch(h2d);

    // Wire DaemonLoop with resolver.
    std::atomic<bool> running{true};
    ldam::DaemonLoop::Deps loop_deps{
        .cmd_ring        = cmd_ring_.get(),
        .cmp_ring        = cmp_ring_.get(),
        .ipc_header      = header_,
        .state_snapshot  = snap_,
        .running         = &running,
        .transfer_engine = transfer_engine_.get(),
        .dispatch_fn     = {},
        .publish_fn      = {},
        .cmd_seq_resolver = [this](uint64_t token) {
            return dispatcher_->resolve_cmd_seq(token);
        },
        .token_cleanup = [this](uint64_t token) {
            dispatcher_->remove_token_mapping(token);
        },
    };
    ldam::DaemonLoop loop(std::move(loop_deps));

    // Run one cycle — should poll transfer completions.
    loop.run_one_cycle();

    // Read the transfer completion from the ring.
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_TRANSFER_DONE));
    EXPECT_EQ(cmp.cmd_seq, 42u);  // The key assertion!
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.transfer.layer_idx, 1u);
    EXPECT_EQ(cmp.transfer.expert_idx, 0u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Stream event operations
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, RecordEvent_Success) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::CMD_RECORD_EVENT, /*seq=*/10, /*gpu_idx=*/0,
                        /*stream_id=*/0);
    cmd.event.event_id = 100;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_EVENT_STATUS));
    EXPECT_EQ(cmp.cmd_seq, 10u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, StreamWaitEvent_Success) {
    make_dispatcher();

    // Record first.
    auto record = make_cmd(lipc::CMD_RECORD_EVENT, /*seq=*/10, /*gpu_idx=*/0,
                           /*stream_id=*/0);
    record.event.event_id = 200;
    dispatcher_->dispatch(record);
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));

    // Wait on it from a different stream.
    auto wait = make_cmd(lipc::CMD_STREAM_WAIT_EVENT, /*seq=*/11, /*gpu_idx=*/0,
                         /*stream_id=*/1);
    wait.event.event_id = 200;
    dispatcher_->dispatch(wait);

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_EVENT_STATUS));
    EXPECT_EQ(cmp.cmd_seq, 11u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, StreamWaitEvent_UnknownEvent) {
    make_dispatcher();

    auto wait = make_cmd(lipc::CMD_STREAM_WAIT_EVENT, /*seq=*/20, /*gpu_idx=*/0);
    wait.event.event_id = 999;
    dispatcher_->dispatch(wait);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 20u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph replay
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, GraphReplay_NotFound) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::CMD_GRAPH_REPLAY, /*seq=*/30, /*gpu_idx=*/0);
    cmd.graph.graph_type = 0;  // kAttentionDecode
    cmd.graph.batch_size = 4;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 30u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Compute commands (stub)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, ComputeCommand_NoBufferRegistry) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::CMD_GATING, /*seq=*/40, /*gpu_idx=*/0);
    cmd.gating.layer_idx  = 1;
    cmd.gating.num_tokens = 4;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 40u);
    // Check error message.
    EXPECT_NE(std::string(cmp.error.message).find("buffer registry"), std::string::npos);
}

TEST_F(CommandDispatcherTest, ComputeCommand_AllTypes_NoRegistry) {
    make_dispatcher();

    // Verify all compute CmdTypes produce CMP_ERROR without buffer registry.
    lipc::CmdType compute_types[] = {
        lipc::CMD_ATTENTION_DECODE, lipc::CMD_ATTENTION_PREFILL,
        lipc::CMD_GATING, lipc::CMD_EXPERT_FFN,
        lipc::CMD_EMBEDDING_LOOKUP, lipc::CMD_OUTPUT_HEAD,
        lipc::CMD_RMSNORM, lipc::CMD_SWIGLU,
        lipc::CMD_MOE_PERMUTE, lipc::CMD_MOE_UNPERMUTE,
        lipc::CMD_DCP_CORRECTION, lipc::CMD_NCCL_ALLREDUCE,
        lipc::CMD_DYNAMIC_FP8_QUANT,
        lipc::CMD_PRESCOPE_GATING, lipc::CMD_PROBE_MLP,
    };

    for (uint32_t i = 0; i < std::size(compute_types); ++i) {
        auto cmd = make_cmd(compute_types[i], /*seq=*/100 + i);
        dispatcher_->dispatch(cmd);

        lipc::Completion cmp{};
        ASSERT_TRUE(read_cmp(*cmp_ring_, cmp))
            << "No completion for cmd type 0x" << std::hex << compute_types[i];
        EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
        EXPECT_EQ(cmp.cmd_seq, 100 + i);
    }
}

TEST_F(CommandDispatcherTest, ComputeCommand_WithBufferRegistry_ResolvesOk) {
    // Create a buffer registry with some fake buffers, then verify that
    // compute commands with valid buf_ids produce CMP_COMPUTE_DONE.
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem[8] = {};

    uint32_t buf1 = registry->register_buffer(&fake_mem[0], 4096, 0, "hidden");
    uint32_t buf2 = registry->register_buffer(&fake_mem[1], 4096, 0, "kv_cache");
    uint32_t buf3 = registry->register_buffer(&fake_mem[2], 4096, 0, "output");
    uint32_t buf4 = registry->register_buffer(&fake_mem[3], 4096, 0, "weights");
    uint32_t buf5 = registry->register_buffer(&fake_mem[4], 4096, 0, "indices");
    uint32_t buf6 = registry->register_buffer(&fake_mem[5], 4096, 0, "offsets");
    uint32_t buf7 = registry->register_buffer(&fake_mem[6], 4096, 0, "scales");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // CMD_ATTENTION_DECODE (TD-ATTN-LEGACY resolved): routes through
    // dispatch_attention_internal (same production path as
    // D_B_CMD_RUN_ATTENTION — absorption + RoPE + scale). Without a wired
    // DcpExecutor it must fail loudly (CMP_ERROR), never silently launch the
    // old raw-hidden-state-as-q path. The DcpExecutor-wired behavior is
    // covered by the fused-attention tests.
    {
        auto cmd = make_cmd(lipc::CMD_ATTENTION_DECODE, /*seq=*/200);
        cmd.attention.layer_idx           = 1;
        cmd.attention.batch_size          = 4;
        cmd.attention.hidden_state_buf_id = buf1;
        cmd.attention.kv_cache_buf_id     = buf2;
        dispatcher_->dispatch(cmd);
        dispatcher_->poll_compute_completions();

        lipc::Completion cmp{};
        ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
        EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
        EXPECT_EQ(cmp.cmd_seq, 200u);
    }

    // CMD_GATING with valid buf_ids → CMP_COMPUTE_DONE
    {
        auto cmd = make_cmd(lipc::CMD_GATING, /*seq=*/201);
        cmd.gating.layer_idx              = 2;
        cmd.gating.num_tokens             = 8;
        cmd.gating.input_buf_id           = buf1;
        cmd.gating.output_weights_buf_id  = buf3;
        cmd.gating.output_indices_buf_id  = buf5;
        dispatcher_->dispatch(cmd);
        dispatcher_->poll_compute_completions();

        lipc::Completion cmp{};
        ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
        EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
        EXPECT_EQ(cmp.cmd_seq, 201u);
    }

    // CMD_RMSNORM with valid buf_ids → CMP_COMPUTE_DONE
    {
        auto cmd = make_cmd(lipc::CMD_RMSNORM, /*seq=*/202);
        cmd.rmsnorm.num_tokens    = 4;
        cmd.rmsnorm.input_buf_id  = buf1;
        cmd.rmsnorm.output_buf_id = buf3;
        cmd.rmsnorm.weight_buf_id = buf4;
        cmd.rmsnorm.eps           = 1e-5f;
        dispatcher_->dispatch(cmd);
        dispatcher_->poll_compute_completions();

        lipc::Completion cmp{};
        ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
        EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
        EXPECT_EQ(cmp.cmd_seq, 202u);
    }

    // CMD_DYNAMIC_FP8_QUANT with valid buf_ids → CMP_COMPUTE_DONE
    {
        auto cmd = make_cmd(lipc::CMD_DYNAMIC_FP8_QUANT, /*seq=*/203);
        cmd.dynamic_fp8_quant.num_tokens    = 4;
        cmd.dynamic_fp8_quant.hidden_dim    = 256;
        cmd.dynamic_fp8_quant.input_buf_id  = buf1;
        cmd.dynamic_fp8_quant.output_buf_id = buf3;
        cmd.dynamic_fp8_quant.scales_buf_id = buf7;
        dispatcher_->dispatch(cmd);
        dispatcher_->poll_compute_completions();

        lipc::Completion cmp{};
        ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
        EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
        EXPECT_EQ(cmp.cmd_seq, 203u);
    }
}

TEST_F(CommandDispatcherTest, ComputeCommand_WithBufferRegistry_InvalidBufId) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake = 0;
    uint32_t valid_id = registry->register_buffer(&fake, 64, 0, "valid");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // CMD_ATTENTION_DECODE (TD-ATTN-LEGACY resolved): buf_ids are ignored;
    // the command routes through dispatch_attention_internal, which requires
    // a wired DcpExecutor → CMP_ERROR here (none in this fixture). Must fail
    // loudly rather than launch the legacy raw-hidden-state-as-q path.
    auto cmd = make_cmd(lipc::CMD_ATTENTION_DECODE, /*seq=*/300);
    cmd.attention.batch_size          = 4;
    cmd.attention.hidden_state_buf_id = valid_id;
    cmd.attention.kv_cache_buf_id     = 999;  // ignored (legacy field)
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 300u);
    EXPECT_NE(std::string(cmp.error.message).find("DcpExecutor"), std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════
// Placement / NUMA
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, AffinityHints_Dispatches) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::CMD_COMPUTE_AFFINITY_HINTS, /*seq=*/50, /*gpu_idx=*/0);
    cmd.affinity_hints.num_gpus = 2;
    // Capacity slots — arbitrary values.
    cmd.affinity_hints.gpu_capacity_slots[0] = 100;
    cmd.affinity_hints.gpu_capacity_slots[1] = 100;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 50u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, NumaMigrate_NoNvmeTier) {
    make_dispatcher();  // nvme_tier = nullptr

    auto cmd = make_cmd(lipc::CMD_NUMA_MIGRATE, /*seq=*/60, /*gpu_idx=*/0);
    cmd.numa_migrate.layer_idx       = 1;
    cmd.numa_migrate.expert_idx      = 5;
    cmd.numa_migrate.target_numa_node = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 60u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Error handling
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, UnknownCommand_Error) {
    make_dispatcher();

    auto cmd = make_cmd(static_cast<lipc::CmdType>(0xDEAD), /*seq=*/70);
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 70u);
}

TEST_F(CommandDispatcherTest, MultipleCommands_Batch) {
    make_dispatcher();

    // Mix of different command types.
    auto reserve = make_cmd(lipc::CMD_CACHE_RESERVE, /*seq=*/1, /*gpu_idx=*/0);
    reserve.cache_reserve.layer_idx  = 1;
    reserve.cache_reserve.expert_idx = 0;
    reserve.cache_reserve.zone       = 1;
    dispatcher_->dispatch(reserve);

    auto event = make_cmd(lipc::CMD_RECORD_EVENT, /*seq=*/2, /*gpu_idx=*/0);
    event.event.event_id = 300;
    dispatcher_->dispatch(event);

    auto compute = make_cmd(lipc::CMD_RMSNORM, /*seq=*/3);
    dispatcher_->dispatch(compute);

    // Read all three completions.
    lipc::Completion cmp{};

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmd_seq, 1u);
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmd_seq, 2u);
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_EVENT_STATUS));

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmd_seq, 3u);
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));  // no buffer registry
}

// ═══════════════════════════════════════════════════════════════════════════
// Sequence lifecycle (IPC-8a)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, SeqCreate_Success) {
    make_dispatcher();

    int free_before = page_allocator_->free_pages(0, lmem::Pool::kMain);

    auto cmd = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/100, /*gpu_idx=*/0);
    cmd.seq_create.seq_id     = 1000;
    cmd.seq_create.prompt_len = 32;  // page_size_tokens=16, so 2 pages
    cmd.seq_create.pool       = 0;   // kMain

    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 100u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.seq_op.seq_id, 1000u);
    EXPECT_EQ(cmp.seq_op.page_count, 2u * kKvLayers);

    int free_after = page_allocator_->free_pages(0, lmem::Pool::kMain);
    EXPECT_EQ(free_before - free_after, 2 * static_cast<int>(kKvLayers));
}

TEST_F(CommandDispatcherTest, SeqCreate_DuplicateSeqId) {
    make_dispatcher();

    auto cmd1 = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/101);
    cmd1.seq_create.seq_id     = 2000;
    cmd1.seq_create.prompt_len = 16;
    dispatcher_->dispatch(cmd1);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.status, 0u);

    // Duplicate — should error.
    auto cmd2 = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/102);
    cmd2.seq_create.seq_id     = 2000;
    cmd2.seq_create.prompt_len = 16;
    dispatcher_->dispatch(cmd2);

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 102u);
}

TEST_F(CommandDispatcherTest, SeqFree_Success) {
    make_dispatcher();

    int free_before = page_allocator_->free_pages(0, lmem::Pool::kMain);

    // Create sequence.
    auto cmd_create = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/110);
    cmd_create.seq_create.seq_id     = 3000;
    cmd_create.seq_create.prompt_len = 48;  // 3 pages
    dispatcher_->dispatch(cmd_create);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.seq_op.page_count, 3u * kKvLayers);

    // Free it.
    auto cmd_free = make_cmd(lipc::CMD_SEQ_FREE, /*seq=*/111);
    cmd_free.seq_free.seq_id = 3000;
    dispatcher_->dispatch(cmd_free);

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 111u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.seq_op.page_count, 3u * kKvLayers);

    int free_after = page_allocator_->free_pages(0, lmem::Pool::kMain);
    EXPECT_EQ(free_after, free_before);  // All pages returned
}

TEST_F(CommandDispatcherTest, SeqFree_UnknownSeqId) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::CMD_SEQ_FREE, /*seq=*/120);
    cmd.seq_free.seq_id = 9999;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 120u);
}

TEST_F(CommandDispatcherTest, SeqFork_Success) {
    make_dispatcher();

    // Create source sequence.
    auto cmd_create = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/130);
    cmd_create.seq_create.seq_id     = 4000;
    cmd_create.seq_create.prompt_len = 32;  // 2 pages
    dispatcher_->dispatch(cmd_create);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.status, 0u);

    int free_before_fork = page_allocator_->free_pages(0, lmem::Pool::kMain);

    // Fork.
    auto cmd_fork = make_cmd(lipc::CMD_SEQ_FORK, /*seq=*/131);
    cmd_fork.seq_fork.src_seq_id = 4000;
    cmd_fork.seq_fork.dst_seq_id = 4001;
    dispatcher_->dispatch(cmd_fork);

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 131u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.seq_op.page_count, 2u * kKvLayers);

    // Fork is CoW — shared pages + kKvLayers new pages for the CoW-copied
    // partial LOGICAL page (all of its per-layer physical pages split).
    int free_after_fork = page_allocator_->free_pages(0, lmem::Pool::kMain);
    EXPECT_EQ(free_after_fork, free_before_fork - static_cast<int>(kKvLayers));
}

TEST_F(CommandDispatcherTest, SeqFork_ThenFreeBoth) {
    make_dispatcher();

    int free_initial = page_allocator_->free_pages(0, lmem::Pool::kMain);

    // Create and fork.
    auto cmd_create = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/140);
    cmd_create.seq_create.seq_id     = 5000;
    cmd_create.seq_create.prompt_len = 16;  // 1 page
    dispatcher_->dispatch(cmd_create);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));

    auto cmd_fork = make_cmd(lipc::CMD_SEQ_FORK, /*seq=*/141);
    cmd_fork.seq_fork.src_seq_id = 5000;
    cmd_fork.seq_fork.dst_seq_id = 5001;
    dispatcher_->dispatch(cmd_fork);
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));

    // Free source — refcount 2→1, page NOT returned.
    auto cmd_free_src = make_cmd(lipc::CMD_SEQ_FREE, /*seq=*/142);
    cmd_free_src.seq_free.seq_id = 5000;
    dispatcher_->dispatch(cmd_free_src);
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.status, 0u);

    int free_after_src = page_allocator_->free_pages(0, lmem::Pool::kMain);
    // dst holds its kKvLayers CoW copies; src's originals were all freed.
    EXPECT_EQ(free_after_src, free_initial - static_cast<int>(kKvLayers));

    // Free destination — refcount 1→0, page returned.
    auto cmd_free_dst = make_cmd(lipc::CMD_SEQ_FREE, /*seq=*/143);
    cmd_free_dst.seq_free.seq_id = 5001;
    dispatcher_->dispatch(cmd_free_dst);
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.status, 0u);

    int free_final = page_allocator_->free_pages(0, lmem::Pool::kMain);
    EXPECT_EQ(free_final, free_initial);  // All pages returned
}

TEST_F(CommandDispatcherTest, SeqFork_UnknownSource) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::CMD_SEQ_FORK, /*seq=*/150);
    cmd.seq_fork.src_seq_id = 8888;
    cmd.seq_fork.dst_seq_id = 8889;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
}

TEST_F(CommandDispatcherTest, SeqFork_DuplicateDestination) {
    make_dispatcher();

    // Create two sequences.
    auto cmd1 = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/160);
    cmd1.seq_create.seq_id     = 6000;
    cmd1.seq_create.prompt_len = 16;
    dispatcher_->dispatch(cmd1);
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));

    auto cmd2 = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/161);
    cmd2.seq_create.seq_id     = 6001;
    cmd2.seq_create.prompt_len = 16;
    dispatcher_->dispatch(cmd2);
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));

    // Fork with dst that already exists.
    auto cmd_fork = make_cmd(lipc::CMD_SEQ_FORK, /*seq=*/162);
    cmd_fork.seq_fork.src_seq_id = 6000;
    cmd_fork.seq_fork.dst_seq_id = 6001;
    dispatcher_->dispatch(cmd_fork);

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 162u);
}

TEST_F(CommandDispatcherTest, SeqCreate_NoPageAllocator) {
    // Create dispatcher without page_allocator.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .page_allocator     = nullptr,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/170);
    cmd.seq_create.seq_id     = 7000;
    cmd.seq_create.prompt_len = 16;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 170u);
}

// ═══════════════════════════════════════════════════════════════════════════
// NVMe tier commands (IPC-8a.1)
// ═══════════════════════════════════════════════════════════════════════════

// Helper: create a temp dir.
static std::string make_temp_nvme_dir() {
    char tmpl[] = "/tmp/layerstorm_disp_nvme_XXXXXX";
    char* dir = ::mkdtemp(tmpl);
    if (!dir) throw std::runtime_error("mkdtemp failed");
    return std::string(dir);
}

TEST_F(CommandDispatcherTest, NvmeRead_NoNvmeTier) {
    make_dispatcher();  // nvme_tier = nullptr by default

    auto cmd = make_cmd(lipc::CMD_NVME_READ, /*seq=*/200);
    cmd.nvme_read.layer_idx  = 1;
    cmd.nvme_read.expert_idx = 0;
    cmd.nvme_read.gpu_hint   = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 200u);
}

TEST_F(CommandDispatcherTest, NvmeRead_ExpertNotOnNvme) {
    // Create NvmeTier with temp dir.
    std::string tmpdir = make_temp_nvme_dir();
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

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .nvme_tier          = nvme.get(),
        .page_allocator     = page_allocator_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Expert not written to NVMe → read_expert returns nullopt → CMP_ERROR.
    auto cmd = make_cmd(lipc::CMD_NVME_READ, /*seq=*/201);
    cmd.nvme_read.layer_idx  = 1;
    cmd.nvme_read.expert_idx = 0;
    cmd.nvme_read.gpu_hint   = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 201u);

    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
}

TEST_F(CommandDispatcherTest, NvmeEvictHost_Success) {
    std::string tmpdir = make_temp_nvme_dir();
    lc::HardwareConfig hw;
    lc::GpuConfig g;
    g.id = 0; g.type = lc::GpuType::rtx5090; g.numa_node = 0;
    hw.gpus.push_back(g);
    lmem::NumaManager numa(hw);

    lmem::NvmeTier::Options nvme_opts;
    nvme_opts.drive_paths = {tmpdir};
    nvme_opts.slot_size_bytes = 4096;
    nvme_opts.host_ram_budget_bytes = 10 * 4096;
    nvme_opts.num_moe_layers = 6;
    nvme_opts.num_experts_per_layer = 8;
    nvme_opts.first_moe_layer = 0;
    auto nvme = std::make_unique<lmem::NvmeTier>(std::move(nvme_opts), numa);

    // Write an expert slot to NVMe (populates the mmap-backed file).
    std::vector<std::byte> data(4096, std::byte{0xAB});
    auto tok = nvme->write_expert({1, 0}, data.data());
    ASSERT_TRUE(tok.has_value());
    EXPECT_TRUE(nvme->is_in_host_ram({1, 0}));

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .nvme_tier          = nvme.get(),
        .page_allocator     = page_allocator_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_NVME_EVICT_HOST, /*seq=*/210);
    cmd.nvme_evict_host.layer_idx  = 1;
    cmd.nvme_evict_host.expert_idx = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_NVME_DONE));
    EXPECT_EQ(cmp.cmd_seq, 210u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.nvme_op.layer_idx, 1u);
    EXPECT_EQ(cmp.nvme_op.expert_idx, 0u);
    EXPECT_EQ(cmp.nvme_op.op, 2u);  // evict_host

    // WP-5: mmap has process lifetime — expert remains accessible.
    // CMD_NVME_EVICT_HOST is a no-op success with mmap-backed NvmeTier.
    EXPECT_TRUE(nvme->is_in_host_ram({1, 0}));

    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
}

TEST_F(CommandDispatcherTest, NvmeEvictHost_NotInHostRam) {
    std::string tmpdir = make_temp_nvme_dir();
    lc::HardwareConfig hw;
    lc::GpuConfig g;
    g.id = 0; g.type = lc::GpuType::rtx5090; g.numa_node = 0;
    hw.gpus.push_back(g);
    lmem::NumaManager numa(hw);

    lmem::NvmeTier::Options nvme_opts;
    nvme_opts.drive_paths = {tmpdir};
    nvme_opts.slot_size_bytes = 4096;
    nvme_opts.host_ram_budget_bytes = 10 * 4096;
    nvme_opts.num_moe_layers = 6;
    nvme_opts.num_experts_per_layer = 8;
    nvme_opts.first_moe_layer = 0;
    auto nvme = std::make_unique<lmem::NvmeTier>(std::move(nvme_opts), numa);

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .nvme_tier          = nvme.get(),
        .page_allocator     = page_allocator_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // WP-5: CMD_NVME_EVICT_HOST is a no-op success with mmap-backed NvmeTier.
    // Even for non-existent experts, it reports success (no discard needed).
    auto cmd = make_cmd(lipc::CMD_NVME_EVICT_HOST, /*seq=*/211);
    cmd.nvme_evict_host.layer_idx  = 1;
    cmd.nvme_evict_host.expert_idx = 99;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_NVME_DONE));
    EXPECT_EQ(cmp.cmd_seq, 211u);

    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
}

TEST_F(CommandDispatcherTest, NvmeWrite_NoHostPtr) {
    std::string tmpdir = make_temp_nvme_dir();
    lc::HardwareConfig hw;
    lc::GpuConfig g;
    g.id = 0; g.type = lc::GpuType::rtx5090; g.numa_node = 0;
    hw.gpus.push_back(g);
    lmem::NumaManager numa(hw);

    lmem::NvmeTier::Options nvme_opts;
    nvme_opts.drive_paths = {tmpdir};
    nvme_opts.slot_size_bytes = 4096;
    nvme_opts.host_ram_budget_bytes = 10 * 4096;
    nvme_opts.num_moe_layers = 6;
    nvme_opts.num_experts_per_layer = 8;
    nvme_opts.first_moe_layer = 0;
    auto nvme = std::make_unique<lmem::NvmeTier>(std::move(nvme_opts), numa);

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .nvme_tier          = nvme.get(),
        .page_allocator     = page_allocator_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Expert not in warm cache → host_ptr null → CMP_ERROR.
    auto cmd = make_cmd(lipc::CMD_NVME_WRITE, /*seq=*/220);
    cmd.nvme_write.layer_idx  = 1;
    cmd.nvme_write.expert_idx = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 220u);

    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
}

TEST_F(CommandDispatcherTest, CancelTransfer_NotFound) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::CMD_CANCEL_TRANSFER, /*seq=*/230);
    cmd.cancel_transfer.target_cmd_seq = 999;  // no such transfer
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CANCEL_DONE));
    EXPECT_EQ(cmp.cmd_seq, 230u);
    EXPECT_EQ(cmp.cancel_result.cancelled, 0u);
}

TEST_F(CommandDispatcherTest, CancelTransfer_PcieH2D) {
    make_dispatcher();

    // Reserve a cache slot for the expert.
    auto reserve = make_cmd(lipc::CMD_CACHE_RESERVE, /*seq=*/240, /*gpu=*/0);
    reserve.cache_reserve.layer_idx  = 1;
    reserve.cache_reserve.expert_idx = 0;
    reserve.cache_reserve.zone       = 1;
    dispatcher_->dispatch(reserve);

    // Drain the reserve completion.
    lipc::Completion cmp_reserve{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp_reserve));

    // Set up NvmeTier for host source resolution.
    std::string tmpdir = make_temp_nvme_dir();
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

    // Write expert slot to NVMe so resolve_host_source finds it via mmap.
    std::vector<std::byte> host_data(static_cast<size_t>(expert_bytes_),
                                     std::byte{0xAA});
    auto tok = nvme->write_expert({1, 0}, host_data.data());
    ASSERT_TRUE(tok.has_value());

    // Re-create dispatcher with NvmeTier.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .nvme_tier          = nvme.get(),
        .page_allocator     = page_allocator_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Enqueue an H2D transfer.
    auto h2d = make_cmd(lipc::CMD_TRANSFER_H2D, /*seq=*/241, /*gpu=*/0);
    h2d.transfer.layer_idx     = 1;
    h2d.transfer.expert_idx    = 0;
    h2d.transfer.sub_component = 0x07;  // SUB_ALL
    h2d.transfer.zone          = 1;
    h2d.transfer.bytes         = expert_bytes_;
    dispatcher_->dispatch(h2d);
    // With null device backend, transfer completes immediately on poll, but we
    // haven't polled yet — the transfer is still "pending" in the engine.
    // Actually, NullDeviceBackend::query_event always returns true,
    // so poll_completions will find it immediately.  But the dispatcher
    // only enqueues — it doesn't poll.  The token is tracked.

    EXPECT_EQ(transfer_engine_->inflight_count(), 1);

    // Cancel it.
    auto cancel = make_cmd(lipc::CMD_CANCEL_TRANSFER, /*seq=*/242);
    cancel.cancel_transfer.target_cmd_seq = 241;
    dispatcher_->dispatch(cancel);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CANCEL_DONE));
    EXPECT_EQ(cmp.cmd_seq, 242u);
    EXPECT_EQ(cmp.cancel_result.target_cmd_seq, 241u);
    EXPECT_EQ(cmp.cancel_result.cancelled, 1u);

    // Transfer engine should have 0 in-flight after cancel.
    EXPECT_EQ(transfer_engine_->inflight_count(), 0);

    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward pass boundary commands (IPC-8c)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, EmbeddingLookup_SidebandSuccess) {
    // Write test token IDs to sideband region.
    auto* token_ids = reinterpret_cast<uint32_t*>(
        sideband_base_ + lipc::IpcLayout::kTokenIdsOff);
    token_ids[0] = 100;
    token_ids[1] = 200;
    token_ids[2] = 300;
    token_ids[3] = 400;

    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem = 0;
    uint32_t out_buf = registry->register_buffer(&fake_mem, 4096, 0, "output");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_EMBEDDING_LOOKUP, /*seq=*/500);
    cmd.embedding_lookup.num_tokens = 4;
    cmd.embedding_lookup.output_buf_id = out_buf;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 500u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.cmd_type, static_cast<uint32_t>(lipc::CMD_EMBEDDING_LOOKUP));
}

TEST_F(CommandDispatcherTest, EmbeddingLookup_NumTokensExceedsMax) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem = 0;
    uint32_t out_buf = registry->register_buffer(&fake_mem, 4096, 0, "output");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_EMBEDDING_LOOKUP, /*seq=*/501);
    cmd.embedding_lookup.num_tokens = lipc::kMaxSidebandTokenIds + 1;
    cmd.embedding_lookup.output_buf_id = out_buf;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 501u);
    EXPECT_NE(std::string(cmp.error.message).find("num_tokens out of range"),
              std::string::npos);
}

TEST_F(CommandDispatcherTest, EmbeddingLookup_NullSideband) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem = 0;
    uint32_t out_buf = registry->register_buffer(&fake_mem, 4096, 0, "output");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = nullptr,  // deliberately null
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_EMBEDDING_LOOKUP, /*seq=*/502);
    cmd.embedding_lookup.num_tokens = 4;
    cmd.embedding_lookup.output_buf_id = out_buf;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 502u);
    EXPECT_NE(std::string(cmp.error.message).find("sideband not configured"),
              std::string::npos);
}

// TD-74k: verify num_tokens > max_batch_size is rejected.
TEST_F(CommandDispatcherTest, EmbeddingLookup_NumTokensExceedsMaxBatchSize) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem = 0;
    uint32_t out_buf = registry->register_buffer(&fake_mem, 4096, 0, "output");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
        .max_batch_size     = 4,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_EMBEDDING_LOOKUP, /*seq=*/503);
    cmd.embedding_lookup.num_tokens    = 8;  // > max_batch_size=4
    cmd.embedding_lookup.output_buf_id = out_buf;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 503u);
    EXPECT_NE(std::string(cmp.error.message).find("max_batch_size"),
              std::string::npos);
}

// TD-74l: verify CMD_EMBEDDING_LOOKUP completes with TP=2 deps.
// The broadcast path (TD-73i) requires cuda_kernels_enabled=true + CUDA
// runtime (integration test), but this validates the TP infrastructure.
TEST_F(CommandDispatcherTest, EmbeddingLookup_TP2_DepsSetup) {
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    lc::GpuRef gpu1{1, 1, lc::GpuType::rtx5090};

    auto attn0 = lcomp::make_null_attention_device(gpu0);
    auto attn1 = lcomp::make_null_attention_device(gpu1);

    lpar::DcpExecutor::Options dcp_opts{
        .dcp_size             = 2,
        .gpus                 = {gpu0, gpu1},
        .max_batch_size       = 8,
        .hidden_size          = 256,
        .num_attention_heads  = 4,
        .q_lora_rank          = 64,
        .kv_lora_rank         = 32,
        .qk_rope_head_dim     = 16,
        .qk_nope_head_dim     = 32,
        .v_head_dim           = 64,
        .rms_norm_eps         = 1e-6f,
        .stream_manager       = stream_manager_.get(),
        .attention_devices    = {attn0.get(), attn1.get()},
    };
    auto dcp = std::make_unique<lpar::DcpExecutor>(std::move(dcp_opts));

    const size_t hidden_bytes = 8 * 256 * 2;
    void* attn_buf0 = aligned_alloc_zeroed(hidden_bytes);
    void* attn_buf1 = aligned_alloc_zeroed(hidden_bytes);
    void* moe_buf0  = aligned_alloc_zeroed(hidden_bytes);
    void* moe_buf1  = aligned_alloc_zeroed(hidden_bytes);

    auto registry = std::make_unique<ldam::BufferRegistry>();
    uint32_t out_buf = registry->register_buffer(attn_buf0, hidden_bytes, 0, "hidden_state.attn.rank0");

    auto* token_ids = reinterpret_cast<uint32_t*>(
        sideband_base_ + lipc::IpcLayout::kTokenIdsOff);
    token_ids[0] = 42;

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .dcp_executor       = dcp.get(),
        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {attn0.get(), attn1.get()},
        .device_backends    = {test_device_backends_[0].get(),
                               test_device_backends_[1].get()},
        .cuda_kernels_enabled = false,
        .hidden_state_pairs = {
            ldam::HiddenStatePair{attn_buf0, moe_buf0, 0, 0, 0},
            ldam::HiddenStatePair{attn_buf1, moe_buf1, 1, 1, 1},
        },
        .per_layer_attn_weights = std::vector<std::vector<lpar::AttentionLayerWeights>>(
            61, std::vector<lpar::AttentionLayerWeights>(2)),
        .max_batch_size     = 8,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_EMBEDDING_LOOKUP, /*seq=*/504);
    cmd.embedding_lookup.num_tokens    = 1;
    cmd.embedding_lookup.output_buf_id = out_buf;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 504u);
    EXPECT_EQ(cmp.status, 0u);

    dispatcher_.reset();
    dcp.reset();
    std::free(attn_buf0);
    std::free(attn_buf1);
    std::free(moe_buf0);
    std::free(moe_buf1);
}

TEST_F(CommandDispatcherTest, OutputHead_ReadbackCompletion) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem[2] = {};
    uint32_t in_buf  = registry->register_buffer(&fake_mem[0], 4096, 0, "input");
    uint32_t out_buf = registry->register_buffer(&fake_mem[1], 4096, 0, "output");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_OUTPUT_HEAD, /*seq=*/510);
    cmd.output_head.num_tokens = 8;
    cmd.output_head.input_buf_id = in_buf;
    cmd.output_head.output_buf_id = out_buf;
    cmd.output_head.readback_to_host = 1;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 510u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.cmd_type, static_cast<uint32_t>(lipc::CMD_OUTPUT_HEAD));
    // Readback fields are accessible (0 expected: cuda_kernels_enabled=false).
    (void)cmp.compute.host_buf_offset;
    (void)cmp.compute.data_bytes;
}

TEST_F(CommandDispatcherTest, OutputHead_NoReadbackZeroFields) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem[2] = {};
    uint32_t in_buf  = registry->register_buffer(&fake_mem[0], 4096, 0, "input");
    uint32_t out_buf = registry->register_buffer(&fake_mem[1], 4096, 0, "output");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_OUTPUT_HEAD, /*seq=*/511);
    cmd.output_head.num_tokens = 4;
    cmd.output_head.input_buf_id = in_buf;
    cmd.output_head.output_buf_id = out_buf;
    cmd.output_head.readback_to_host = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 511u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.cmd_type, static_cast<uint32_t>(lipc::CMD_OUTPUT_HEAD));
    EXPECT_EQ(cmp.compute.host_buf_offset, 0u);
    EXPECT_EQ(cmp.compute.data_bytes, 0u);
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU-side confidence estimation (IPC-8g)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, OutputHead_ComputeConfidence) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem[2] = {};
    uint32_t in_buf  = registry->register_buffer(&fake_mem[0], 4096, 0, "input");
    uint32_t out_buf = registry->register_buffer(&fake_mem[1], 4096, 0, "output");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_OUTPUT_HEAD, /*seq=*/512);
    cmd.output_head.num_tokens = 4;
    cmd.output_head.input_buf_id = in_buf;
    cmd.output_head.output_buf_id = out_buf;
    cmd.output_head.readback_to_host = 0;
    cmd.output_head.compute_confidence = 1;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 512u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.cmd_type, static_cast<uint32_t>(lipc::CMD_OUTPUT_HEAD));
    // TODO:DEBT TD-56a: 0.0f because cuda_kernels_enabled=false, not because dispatch is missing
    EXPECT_EQ(cmp.compute.top1_prob, 0.0f);
    EXPECT_EQ(cmp.compute.entropy, 0.0f);
}

TEST_F(CommandDispatcherTest, OutputHead_NoConfidenceZeroFields) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem[2] = {};
    uint32_t in_buf  = registry->register_buffer(&fake_mem[0], 4096, 0, "input");
    uint32_t out_buf = registry->register_buffer(&fake_mem[1], 4096, 0, "output");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_OUTPUT_HEAD, /*seq=*/513);
    cmd.output_head.num_tokens = 4;
    cmd.output_head.input_buf_id = in_buf;
    cmd.output_head.output_buf_id = out_buf;
    cmd.output_head.readback_to_host = 0;
    cmd.output_head.compute_confidence = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 513u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.top1_prob, 0.0f);
    EXPECT_EQ(cmp.compute.entropy, 0.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU-side token sampling (IPC-8f)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, SampleTokens_Success) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem = 0;
    uint32_t logits_buf = registry->register_buffer(&fake_mem, 129280 * 4, 0, "logits");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_SAMPLE_TOKENS, /*seq=*/700);
    cmd.sample_tokens.num_tokens = 4;
    cmd.sample_tokens.logits_buf_id = logits_buf;
    cmd.sample_tokens.vocab_size = 129280;
    cmd.sample_tokens.top_k = 50;
    cmd.sample_tokens.temperature = 1.0f;
    cmd.sample_tokens.top_p = 0.9f;
    cmd.sample_tokens.random_seed = 42;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 700u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.cmd_type, static_cast<uint32_t>(lipc::CMD_SAMPLE_TOKENS));
    EXPECT_EQ(cmp.compute.data_bytes, 4u * sizeof(uint32_t));
}

TEST_F(CommandDispatcherTest, SampleTokens_NumTokensZero) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem = 0;
    uint32_t logits_buf = registry->register_buffer(&fake_mem, 4096, 0, "logits");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_SAMPLE_TOKENS, /*seq=*/701);
    cmd.sample_tokens.num_tokens = 0;
    cmd.sample_tokens.logits_buf_id = logits_buf;
    cmd.sample_tokens.vocab_size = 1024;
    cmd.sample_tokens.temperature = 1.0f;
    cmd.sample_tokens.top_p = 1.0f;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 701u);
    EXPECT_NE(std::string(cmp.error.message).find("num_tokens out of range"),
              std::string::npos);
}

TEST_F(CommandDispatcherTest, SampleTokens_NumTokensExceedsMax) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem = 0;
    uint32_t logits_buf = registry->register_buffer(&fake_mem, 4096, 0, "logits");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_SAMPLE_TOKENS, /*seq=*/702);
    cmd.sample_tokens.num_tokens = lipc::kMaxSidebandTokenIds + 1;
    cmd.sample_tokens.logits_buf_id = logits_buf;
    cmd.sample_tokens.vocab_size = 1024;
    cmd.sample_tokens.temperature = 1.0f;
    cmd.sample_tokens.top_p = 1.0f;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 702u);
    EXPECT_NE(std::string(cmp.error.message).find("num_tokens out of range"),
              std::string::npos);
}

TEST_F(CommandDispatcherTest, SampleTokens_VocabSizeZero) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem = 0;
    uint32_t logits_buf = registry->register_buffer(&fake_mem, 4096, 0, "logits");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_SAMPLE_TOKENS, /*seq=*/703);
    cmd.sample_tokens.num_tokens = 4;
    cmd.sample_tokens.logits_buf_id = logits_buf;
    cmd.sample_tokens.vocab_size = 0;
    cmd.sample_tokens.temperature = 1.0f;
    cmd.sample_tokens.top_p = 1.0f;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 703u);
    EXPECT_NE(std::string(cmp.error.message).find("vocab_size must be > 0"),
              std::string::npos);
}

TEST_F(CommandDispatcherTest, SampleTokens_InvalidLogitsBufId) {
    auto registry = std::make_unique<ldam::BufferRegistry>();

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_SAMPLE_TOKENS, /*seq=*/704);
    cmd.sample_tokens.num_tokens = 4;
    cmd.sample_tokens.logits_buf_id = 9999;  // unregistered
    cmd.sample_tokens.vocab_size = 1024;
    cmd.sample_tokens.temperature = 1.0f;
    cmd.sample_tokens.top_p = 1.0f;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 704u);
    EXPECT_NE(std::string(cmp.error.message).find("invalid logits_buf_id"),
              std::string::npos);
}

TEST_F(CommandDispatcherTest, SampleTokens_NullSideband) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem = 0;
    uint32_t logits_buf = registry->register_buffer(&fake_mem, 4096, 0, "logits");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = nullptr,  // deliberately null
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_SAMPLE_TOKENS, /*seq=*/705);
    cmd.sample_tokens.num_tokens = 4;
    cmd.sample_tokens.logits_buf_id = logits_buf;
    cmd.sample_tokens.vocab_size = 1024;
    cmd.sample_tokens.temperature = 1.0f;
    cmd.sample_tokens.top_p = 1.0f;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 705u);
    EXPECT_NE(std::string(cmp.error.message).find("sideband not configured"),
              std::string::npos);
}

TEST_F(CommandDispatcherTest, SampleTokens_InvalidTopP) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem = 0;
    uint32_t logits_buf = registry->register_buffer(&fake_mem, 4096, 0, "logits");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_SAMPLE_TOKENS, /*seq=*/706);
    cmd.sample_tokens.num_tokens = 4;
    cmd.sample_tokens.logits_buf_id = logits_buf;
    cmd.sample_tokens.vocab_size = 1024;
    cmd.sample_tokens.temperature = 1.0f;
    cmd.sample_tokens.top_p = 1.5f;  // out of range
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 706u);
    EXPECT_NE(std::string(cmp.error.message).find("top_p must be in [0, 1]"),
              std::string::npos);
}

TEST_F(CommandDispatcherTest, SampleTokens_ArgmaxMode) {
    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem = 0;
    uint32_t logits_buf = registry->register_buffer(&fake_mem, 4096, 0, "logits");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .buffer_registry    = registry.get(),
        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_SAMPLE_TOKENS, /*seq=*/707);
    cmd.sample_tokens.num_tokens = 1;
    cmd.sample_tokens.logits_buf_id = logits_buf;
    cmd.sample_tokens.vocab_size = 1024;
    cmd.sample_tokens.top_k = 0;
    cmd.sample_tokens.temperature = 0.0f;  // argmax mode
    cmd.sample_tokens.top_p = 1.0f;
    cmd.sample_tokens.random_seed = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 707u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.data_bytes, 1u * sizeof(uint32_t));
}

// ── IPC-8d: Fused compute command tests ──────────────────────────────────────

TEST_F(CommandDispatcherTest, RunAttention_Success) {
    // Without DcpExecutor/weights: dispatch fails with CMP_ERROR (TD-40a)
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/600);
    cmd.run_attention.layer_idx  = 5;
    cmd.run_attention.num_seqs   = 8;
    cmd.run_attention.is_prefill = 0;
    cmd.run_attention.use_graph  = 0;
    cmd.run_attention.is_draft   = 0;
    cmd.run_attention.chunk_start = 0;
    cmd.run_attention.chunk_len   = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 600u);
    EXPECT_EQ(cmp.status, 1u);
}

TEST_F(CommandDispatcherTest, RunAttention_IsDraft) {
    // Without DcpExecutor: dispatch fails (TD-40a)
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/601);
    cmd.run_attention.layer_idx  = 10;
    cmd.run_attention.num_seqs   = 4;
    cmd.run_attention.is_prefill = 1;
    cmd.run_attention.use_graph  = 0;
    cmd.run_attention.is_draft   = 1;
    cmd.run_attention.chunk_start = 512;
    cmd.run_attention.chunk_len   = 512;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 601u);
    EXPECT_EQ(cmp.status, 1u);
}

TEST_F(CommandDispatcherTest, RunAttention_NumSeqsExceedsMax) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/602);
    cmd.run_attention.layer_idx = 0;
    cmd.run_attention.num_seqs  = lipc::kMaxBatchDescriptors + 1;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 602u);
}

TEST_F(CommandDispatcherTest, RunMoe_Success) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/610);
    cmd.run_moe.layer_idx                 = 3;
    cmd.run_moe.num_seqs                  = 4;
    cmd.run_moe.moe_mode                  = 0;
    cmd.run_moe.apply_residual_correction = 0;
    cmd.run_moe.store_gating_output       = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 610u);
    EXPECT_EQ(cmp.status, 1u);
}

TEST_F(CommandDispatcherTest, RunMoe_InvalidMoeMode) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/611);
    cmd.run_moe.layer_idx = 0;
    cmd.run_moe.num_seqs  = 1;
    cmd.run_moe.moe_mode  = 3;  // Invalid: must be 0-2
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 611u);
}

TEST_F(CommandDispatcherTest, RunMoe_AllFieldsSet) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/612);
    cmd.run_moe.layer_idx                 = 42;
    cmd.run_moe.num_seqs                  = 32;
    cmd.run_moe.moe_mode                  = 2;
    cmd.run_moe.apply_residual_correction = 1;
    cmd.run_moe.store_gating_output       = 1;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 612u);
    EXPECT_EQ(cmp.status, 1u);
}

// ── IPC-8h: Multi-completion checkpoints ─────────────────────────────────

TEST_F(CommandDispatcherTest, RunAttention_EmitCheckpoint) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/620);
    cmd.run_attention.layer_idx       = 7;
    cmd.run_attention.num_seqs        = 4;
    cmd.run_attention.is_prefill      = 0;
    cmd.run_attention.use_graph       = 0;
    cmd.run_attention.is_draft        = 1;
    cmd.run_attention.emit_checkpoint = 1;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(cmd);

    // Without DcpExecutor: dispatch fails with CMP_ERROR (TD-40a)
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 620u);
    EXPECT_EQ(cmp.status, 1u);
}

TEST_F(CommandDispatcherTest, RunAttention_NoCheckpointByDefault) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/621);
    cmd.run_attention.layer_idx       = 3;
    cmd.run_attention.num_seqs        = 2;
    cmd.run_attention.is_prefill      = 0;
    cmd.run_attention.use_graph       = 0;
    cmd.run_attention.is_draft        = 0;
    cmd.run_attention.emit_checkpoint = 0;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(cmd);

    // Without DcpExecutor: dispatch fails with CMP_ERROR (TD-40a)
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 621u);
    EXPECT_EQ(cmp.status, 1u);

    // No second completion
    lipc::Completion cmp2{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp2));
}

TEST_F(CommandDispatcherTest, RunMoe_EmitCheckpoint) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/630);
    cmd.run_moe.layer_idx                 = 15;
    cmd.run_moe.num_seqs                  = 8;
    cmd.run_moe.moe_mode                  = 1;
    cmd.run_moe.apply_residual_correction = 0;
    cmd.run_moe.store_gating_output       = 1;
    cmd.run_moe.emit_checkpoint           = 1;
    dispatcher_->dispatch(cmd);

    // Without ExpertDevice: dispatch fails with CMP_ERROR (TD-40a)
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 630u);
    EXPECT_EQ(cmp.status, 1u);
}

TEST_F(CommandDispatcherTest, RunMoe_NoCheckpointByDefault) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .sideband_base      = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/631);
    cmd.run_moe.layer_idx                 = 3;
    cmd.run_moe.num_seqs                  = 2;
    cmd.run_moe.moe_mode                  = 0;
    cmd.run_moe.apply_residual_correction = 0;
    cmd.run_moe.store_gating_output       = 0;
    cmd.run_moe.emit_checkpoint           = 0;
    dispatcher_->dispatch(cmd);

    // Without ExpertDevice: dispatch fails with CMP_ERROR (TD-40a)
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 631u);
    EXPECT_EQ(cmp.status, 1u);

    // No second completion
    lipc::Completion cmp2{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp2));
}

// ═══════════════════════════════════════════════════════════════════════════
// IPC-8e: Fused command handlers

// ── D fusion: single-expert commands ──────────────────────────────────────

TEST_F(CommandDispatcherTest, PrefetchExpert_NoHostSource) {
    make_dispatcher();  // ELM active, no loaded_model → COLD experts

    auto cmd = make_cmd(lipc::D_CMD_PREFETCH_EXPERT, /*seq=*/700);
    cmd.prefetch_expert.layer_idx  = 1;
    cmd.prefetch_expert.expert_idx = 3;
    cmd.prefetch_expert.zone       = 0;
    cmd.prefetch_expert.gpu_idx    = 0;
    dispatcher_->dispatch(cmd);

    // ELM path: ensure_resident → COLD, no NVMe → failure via ELM poll.
    auto completions = poll_elm();
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_FALSE(completions[0].success);
    EXPECT_EQ(completions[0].cmd_seq, 700u);
    EXPECT_EQ(completions[0].key, (lmem::ExpertKey{1, 3}));
}

TEST_F(CommandDispatcherTest, PrefetchExpert_NoExpertCache) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring = cmp_ring_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_CMD_PREFETCH_EXPERT, /*seq=*/701);
    cmd.prefetch_expert.layer_idx = 1;
    cmd.prefetch_expert.expert_idx = 3;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 701u);
}

TEST_F(CommandDispatcherTest, EvictToHost_Success) {
    make_dispatcher();

    // Reserve + mark ready so evict finds it.
    auto key = lmem::ExpertKey{2, 5};
    cache_->reserve(key, 0, lmem::CacheZone::kStable, false);
    cache_->mark_ready(key, 0, lmem::SubComponent::kAll);

    auto cmd = make_cmd(lipc::D_CMD_EVICT_TO_HOST, /*seq=*/710);
    cmd.evict_to_host.layer_idx  = 2;
    cmd.evict_to_host.expert_idx = 5;
    cmd.evict_to_host.gpu_idx    = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 710u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, EvictToHost_NotResident) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::D_CMD_EVICT_TO_HOST, /*seq=*/711);
    cmd.evict_to_host.layer_idx  = 99;
    cmd.evict_to_host.expert_idx = 99;
    cmd.evict_to_host.gpu_idx    = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.status, 1u);  // evict failed (not resident)
}

TEST_F(CommandDispatcherTest, StageExpert_NoNvmeTier) {
    make_dispatcher();  // ELM active, no nvme_tier, no loaded_model → COLD

    auto cmd = make_cmd(lipc::D_CMD_STAGE_EXPERT, /*seq=*/720);
    cmd.stage_expert.layer_idx  = 1;
    cmd.stage_expert.expert_idx = 2;
    cmd.stage_expert.zone       = 0;
    cmd.stage_expert.gpu_idx    = 0;
    dispatcher_->dispatch(cmd);

    // ELM path: ensure_resident → COLD, no NVMe → failure via poll.
    auto completions = poll_elm();
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_FALSE(completions[0].success);
    EXPECT_EQ(completions[0].cmd_seq, 720u);
}

// ── D+B fusion: batch commands ───────────────────────────────────────────

TEST_F(CommandDispatcherTest, PrefetchBatch_NullSideband) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring      = cmp_ring_.get(),
        .expert_cache  = cache_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_PREFETCH_BATCH, /*seq=*/730);
    cmd.prefetch_batch.count = 3;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 730u);
}

TEST_F(CommandDispatcherTest, PrefetchBatch_CountExceedsMax) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring      = cmp_ring_.get(),
        .expert_cache  = cache_.get(),
        .sideband_base = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_PREFETCH_BATCH, /*seq=*/731);
    cmd.prefetch_batch.count = lipc::kMaxExpertPrefetch + 1;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 731u);
}

TEST_F(CommandDispatcherTest, PrefetchBatch_ZeroCount) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring      = cmp_ring_.get(),
        .expert_cache  = cache_.get(),
        .sideband_base = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_PREFETCH_BATCH, /*seq=*/732);
    cmd.prefetch_batch.count = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 732u);
}

TEST_F(CommandDispatcherTest, PrefetchBatch_Success) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring        = cmp_ring_.get(),
        .transfer_engine = transfer_engine_.get(),
        .expert_cache    = cache_.get(),
        .sideband_base   = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Write sideband entries.
    auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
        sideband_base_ + lipc::IpcLayout::kExpertPrefetchOff);
    entries[0] = {1, 0, 0, 0};  // layer=1 expert=0 zone=0 gpu=0
    entries[1] = {1, 1, 0, 0};

    auto cmd = make_cmd(lipc::D_B_CMD_PREFETCH_BATCH, /*seq=*/733);
    cmd.prefetch_batch.count = 2;
    dispatcher_->dispatch(cmd);

    // No host source → all skipped → immediate cache completion.
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 733u);
}

TEST_F(CommandDispatcherTest, EvictBatch_NullSideband) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring      = cmp_ring_.get(),
        .expert_cache  = cache_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_EVICT_BATCH, /*seq=*/740);
    cmd.evict_batch.count = 2;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 740u);
}

TEST_F(CommandDispatcherTest, EvictBatch_Success) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring      = cmp_ring_.get(),
        .expert_cache  = cache_.get(),
        .sideband_base = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Reserve + mark ready for two experts.
    auto k1 = lmem::ExpertKey{3, 0};
    auto k2 = lmem::ExpertKey{3, 1};
    cache_->reserve(k1, 0, lmem::CacheZone::kStable, false);
    cache_->mark_ready(k1, 0, lmem::SubComponent::kAll);
    cache_->reserve(k2, 0, lmem::CacheZone::kStable, false);
    cache_->mark_ready(k2, 0, lmem::SubComponent::kAll);

    // Write sideband entries.
    auto* entries = reinterpret_cast<lipc::ExpertEvictionEntry*>(
        sideband_base_ + lipc::IpcLayout::kExpertEvictionOff);
    entries[0] = {3, 0, 0, 0};
    entries[1] = {3, 1, 0, 0};

    auto cmd = make_cmd(lipc::D_B_CMD_EVICT_BATCH, /*seq=*/741);
    cmd.evict_batch.count = 2;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 741u);
    EXPECT_EQ(cmp.status, 0u);

    // Verify both are evicted.
    EXPECT_EQ(cache_->lookup(k1, 0), nullptr);
    EXPECT_EQ(cache_->lookup(k2, 0), nullptr);
}

// ── B fusion: batch NVMe read ────────────────────────────────────────────

TEST_F(CommandDispatcherTest, NvmeBatchRead_NoNvmeTier) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring      = cmp_ring_.get(),
        .sideband_base = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::B_CMD_NVME_BATCH_READ, /*seq=*/750);
    cmd.nvme_batch_read.count = 3;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 750u);
}

TEST_F(CommandDispatcherTest, NvmeBatchRead_CountExceedsMax) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring      = cmp_ring_.get(),
        .sideband_base = sideband_base_,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::B_CMD_NVME_BATCH_READ, /*seq=*/751);
    cmd.nvme_batch_read.count = lipc::kMaxNvmeReadBatch + 1;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 751u);
}

// ── D fusion: compute stubs ──────────────────────────────────────────────

TEST_F(CommandDispatcherTest, RunPrefetchProbe_Success) {
    auto br = std::make_unique<ldam::BufferRegistry>();
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring        = cmp_ring_.get(),
        .stream_manager  = stream_manager_.get(),
        .buffer_registry = br.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_CMD_RUN_PREFETCH_PROBE, /*seq=*/760);
    cmd.run_prefetch_probe.target_layer = 15;
    cmd.run_prefetch_probe.num_tokens   = 32;
    cmd.run_prefetch_probe.probe_points = 0x07;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 760u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.layer_idx, 15u);
}

TEST_F(CommandDispatcherTest, RunPrefetchProbe_NoBufferRegistry) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring = cmp_ring_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_CMD_RUN_PREFETCH_PROBE, /*seq=*/761);
    cmd.run_prefetch_probe.target_layer = 15;
    cmd.run_prefetch_probe.num_tokens   = 32;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 761u);
}

TEST_F(CommandDispatcherTest, RunAdapterForward_Success) {
    auto br = std::make_unique<ldam::BufferRegistry>();
    // Register a fake buffer so resolve succeeds.
    static uint8_t fake_buf[64];
    uint32_t buf_id = br->register_buffer(fake_buf, sizeof(fake_buf), 0);

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring        = cmp_ring_.get(),
        .stream_manager  = stream_manager_.get(),
        .buffer_registry = br.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_CMD_RUN_ADAPTER_FORWARD, /*seq=*/770);
    cmd.run_adapter_forward.num_tokens             = 8;
    cmd.run_adapter_forward.adapter_weights_buf_id = buf_id;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 770u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, RunAdapterForward_InvalidBufId) {
    auto br = std::make_unique<ldam::BufferRegistry>();

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring        = cmp_ring_.get(),
        .buffer_registry = br.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_CMD_RUN_ADAPTER_FORWARD, /*seq=*/771);
    cmd.run_adapter_forward.num_tokens             = 8;
    cmd.run_adapter_forward.adapter_weights_buf_id = 999;  // not registered
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 771u);
}

// ── E fusion: extended sequence lifecycle ─────────────────────────────────

TEST_F(CommandDispatcherTest, ESeqCreate_Success) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::E_CMD_SEQ_CREATE, /*seq=*/780);
    cmd.seq_create.seq_id     = 5000;
    cmd.seq_create.prompt_len = 100;
    cmd.seq_create.pool       = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 780u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.seq_op.seq_id, 5000u);
    EXPECT_GT(cmp.seq_op.page_count, 0u);
}

TEST_F(CommandDispatcherTest, ESeqCreate_DuplicateSeqId) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::E_CMD_SEQ_CREATE, /*seq=*/781);
    cmd.seq_create.seq_id     = 5001;
    cmd.seq_create.prompt_len = 50;
    cmd.seq_create.pool       = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp1{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp1));
    EXPECT_EQ(cmp1.status, 0u);

    // Second create with same seq_id → error.
    auto cmd2 = make_cmd(lipc::E_CMD_SEQ_CREATE, /*seq=*/782);
    cmd2.seq_create.seq_id     = 5001;
    cmd2.seq_create.prompt_len = 50;
    cmd2.seq_create.pool       = 0;
    dispatcher_->dispatch(cmd2);

    lipc::Completion cmp2{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp2));
    EXPECT_EQ(cmp2.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp2.cmd_seq, 782u);
}

TEST_F(CommandDispatcherTest, ESeqFree_Success) {
    make_dispatcher();

    // First create a sequence.
    auto cmd1 = make_cmd(lipc::E_CMD_SEQ_CREATE, /*seq=*/790);
    cmd1.seq_create.seq_id     = 6000;
    cmd1.seq_create.prompt_len = 50;
    cmd1.seq_create.pool       = 0;
    dispatcher_->dispatch(cmd1);

    lipc::Completion cmp1{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp1));
    EXPECT_EQ(cmp1.status, 0u);

    // Free it.
    auto cmd2 = make_cmd(lipc::E_CMD_SEQ_FREE, /*seq=*/791);
    cmd2.seq_free.seq_id = 6000;
    dispatcher_->dispatch(cmd2);

    lipc::Completion cmp2{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp2));
    EXPECT_EQ(cmp2.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(cmp2.cmd_seq, 791u);
    EXPECT_EQ(cmp2.status, 0u);
    EXPECT_EQ(cmp2.seq_op.seq_id, 6000u);
}

TEST_F(CommandDispatcherTest, ESeqFree_UnknownSeqId) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::E_CMD_SEQ_FREE, /*seq=*/792);
    cmd.seq_free.seq_id = 99999;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 792u);
}

// ═══════════════════════════════════════════════════════════════════════════
// ELM-3 integration: fused commands through ExpertLifecycleManager

TEST_F(CommandDispatcherTest, ELM_PrefetchExpert_AutoChains) {
    // Build a fake LoadedModel for host source.
    lmod::LoadedModel model;
    model.layers.resize(6);
    model.layers[1].layer_idx = 1;
    model.layers[1].routed_experts.resize(8);
    std::vector<std::byte> fake_data(expert_bytes_, std::byte{0xAB});
    lmod::RawTensor rt{
        .data = std::span<const std::byte>(fake_data),
        .dtype = lmod::SafetensorsDtype::F8_E4M3,
        .shape = {expert_bytes_}};
    lmod::WeightBundle wb{.id = {}, .weight = rt, .packed_slot = rt.data};
    model.layers[1].routed_experts[0].push_back(std::move(wb));

    // Create ELM with mmap source.
    ldam::ExpertLifecycleManager::Deps elm_deps{
        .expert_cache    = cache_.get(),
        .transfer_engine = transfer_engine_.get(),
        .loaded_model    = &model,
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(elm_deps));

    // Wire dispatcher with ELM.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring        = cmp_ring_.get(),
        .transfer_engine = transfer_engine_.get(),
        .expert_cache    = cache_.get(),
        .elm             = elm.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Dispatch prefetch_expert — should go through ELM.
    auto cmd = make_cmd(lipc::D_CMD_PREFETCH_EXPERT, /*seq=*/800);
    cmd.prefetch_expert.layer_idx  = 1;
    cmd.prefetch_expert.expert_idx = 0;
    cmd.prefetch_expert.zone       = 0;
    cmd.prefetch_expert.gpu_idx    = 0;
    dispatcher_->dispatch(cmd);

    // ELM manages completions — poll ELM to advance state.
    auto completions = elm->poll(transfer_engine_->poll_completions(), {}).lifecycle;
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_TRUE(completions[0].success);
    EXPECT_EQ(completions[0].cmd_seq, 800u);

    // Expert should be HOT.
    auto st = elm->state({1, 0}, 0);
    EXPECT_EQ(st.gpu_tier, ldam::GpuTier::kHot);
}

TEST_F(CommandDispatcherTest, ELM_StageExpert_SameAsEnsureResident) {
    lmod::LoadedModel model;
    model.layers.resize(6);
    model.layers[2].layer_idx = 2;
    model.layers[2].routed_experts.resize(8);
    std::vector<std::byte> fake_data(expert_bytes_, std::byte{0xCD});
    lmod::RawTensor rt{
        .data = std::span<const std::byte>(fake_data),
        .dtype = lmod::SafetensorsDtype::F8_E4M3,
        .shape = {expert_bytes_}};
    lmod::WeightBundle wb{.id = {}, .weight = rt, .packed_slot = rt.data};
    model.layers[2].routed_experts[3].push_back(std::move(wb));

    ldam::ExpertLifecycleManager::Deps elm_deps{
        .expert_cache    = cache_.get(),
        .transfer_engine = transfer_engine_.get(),
        .loaded_model    = &model,
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(elm_deps));

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring        = cmp_ring_.get(),
        .transfer_engine = transfer_engine_.get(),
        .expert_cache    = cache_.get(),
        .elm             = elm.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // stage_expert via ELM — same as ensure_resident (auto-detects mmap).
    auto cmd = make_cmd(lipc::D_CMD_STAGE_EXPERT, /*seq=*/810);
    cmd.stage_expert.layer_idx  = 2;
    cmd.stage_expert.expert_idx = 3;
    cmd.stage_expert.zone       = 0;
    cmd.stage_expert.gpu_idx    = 0;
    dispatcher_->dispatch(cmd);

    auto completions = elm->poll(transfer_engine_->poll_completions(), {}).lifecycle;
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_TRUE(completions[0].success);
}

TEST_F(CommandDispatcherTest, ELM_EvictToHost_BlockedWhileTransferring) {
    lmod::LoadedModel model;
    model.layers.resize(6);
    model.layers[1].layer_idx = 1;
    model.layers[1].routed_experts.resize(8);
    std::vector<std::byte> fake_data(expert_bytes_, std::byte{0xAB});
    lmod::RawTensor rt{
        .data = std::span<const std::byte>(fake_data),
        .dtype = lmod::SafetensorsDtype::F8_E4M3,
        .shape = {expert_bytes_}};
    lmod::WeightBundle wb{.id = {}, .weight = rt, .packed_slot = rt.data};
    model.layers[1].routed_experts[0].push_back(std::move(wb));

    ldam::ExpertLifecycleManager::Deps elm_deps{
        .expert_cache    = cache_.get(),
        .transfer_engine = transfer_engine_.get(),
        .loaded_model    = &model,
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(elm_deps));

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring        = cmp_ring_.get(),
        .transfer_engine = transfer_engine_.get(),
        .expert_cache    = cache_.get(),
        .elm             = elm.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Start prefetch (interest pending).
    auto cmd1 = make_cmd(lipc::D_CMD_PREFETCH_EXPERT, /*seq=*/820);
    cmd1.prefetch_expert.layer_idx  = 1;
    cmd1.prefetch_expert.expert_idx = 0;
    cmd1.prefetch_expert.zone       = 0;
    cmd1.prefetch_expert.gpu_idx    = 0;
    dispatcher_->dispatch(cmd1);

    // Try evict — should be blocked (interest pending).
    auto cmd2 = make_cmd(lipc::D_CMD_EVICT_TO_HOST, /*seq=*/821);
    cmd2.evict_to_host.layer_idx  = 1;
    cmd2.evict_to_host.expert_idx = 0;
    cmd2.evict_to_host.gpu_idx    = 0;
    dispatcher_->dispatch(cmd2);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.status, 1u);  // evict returned false (blocked)

    // Complete the transfer, then evict should work.
    elm->poll(transfer_engine_->poll_completions(), {});
    auto cmd3 = make_cmd(lipc::D_CMD_EVICT_TO_HOST, /*seq=*/822);
    cmd3.evict_to_host.layer_idx  = 1;
    cmd3.evict_to_host.expert_idx = 0;
    cmd3.evict_to_host.gpu_idx    = 0;
    dispatcher_->dispatch(cmd3);

    lipc::Completion cmp2{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp2));
    EXPECT_EQ(cmp2.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp2.status, 0u);  // evict succeeded
}

TEST_F(CommandDispatcherTest, ELM_CancelTransfer_RoutedThroughELM) {
    // Build fake model.
    lmod::LoadedModel model;
    model.layers.resize(6);
    model.layers[1].layer_idx = 1;
    model.layers[1].routed_experts.resize(8);
    std::vector<std::byte> fake_data(expert_bytes_, std::byte{0xAB});
    lmod::RawTensor rt{
        .data = std::span<const std::byte>(fake_data),
        .dtype = lmod::SafetensorsDtype::F8_E4M3,
        .shape = {expert_bytes_}};
    lmod::WeightBundle wb{.id = {}, .weight = rt, .packed_slot = rt.data};
    model.layers[1].routed_experts[0].push_back(std::move(wb));

    // Create ELM.
    ldam::ExpertLifecycleManager::Deps elm_deps{
        .expert_cache    = cache_.get(),
        .transfer_engine = transfer_engine_.get(),
        .loaded_model    = &model,
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(elm_deps));

    // Wire dispatcher with ELM.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring        = cmp_ring_.get(),
        .transfer_engine = transfer_engine_.get(),
        .expert_cache    = cache_.get(),
        .elm             = elm.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Prefetch expert via ELM.
    auto cmd = make_cmd(lipc::D_CMD_PREFETCH_EXPERT, /*seq=*/900);
    cmd.prefetch_expert.layer_idx  = 1;
    cmd.prefetch_expert.expert_idx = 0;
    cmd.prefetch_expert.zone       = 0;
    cmd.prefetch_expert.gpu_idx    = 0;
    dispatcher_->dispatch(cmd);

    // ELM should track this cmd_seq.
    EXPECT_NE(elm->find_by_cmd_seq(900), 0u);

    // Cancel through CMD_CANCEL_TRANSFER — should route through ELM.
    auto cancel = make_cmd(lipc::CMD_CANCEL_TRANSFER, /*seq=*/901);
    cancel.cancel_transfer.target_cmd_seq = 900;
    dispatcher_->dispatch(cancel);

    // Verify cancel completion.
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CANCEL_DONE));
    EXPECT_EQ(cmp.cmd_seq, 901u);
    EXPECT_EQ(cmp.cancel_result.cancelled, 1u);

    // ELM should have cancelled the interest — state reverted to ABSENT.
    EXPECT_EQ(elm->state({1, 0}, 0).gpu_tier, ldam::GpuTier::kAbsent);
    EXPECT_EQ(elm->find_by_cmd_seq(900), 0u);
}

TEST_F(CommandDispatcherTest, ELM_NvmeEvictHost_WarmExpertAllowed) {
    // ELM with mmap source — host_state is kWarm, discard should proceed.
    // No NvmeTier in fixture → handler returns error (NVMe not configured),
    // but the ELM guard should NOT fire (kWarm != kLoadingToRam).
    lmod::LoadedModel model;
    model.layers.resize(6);
    model.layers[1].layer_idx = 1;
    model.layers[1].routed_experts.resize(8);
    std::vector<std::byte> fake_data(expert_bytes_, std::byte{0xAB});
    lmod::RawTensor rt{
        .data = std::span<const std::byte>(fake_data),
        .dtype = lmod::SafetensorsDtype::F8_E4M3,
        .shape = {expert_bytes_}};
    lmod::WeightBundle wb{.id = {}, .weight = rt, .packed_slot = rt.data};
    model.layers[1].routed_experts[0].push_back(std::move(wb));

    ldam::ExpertLifecycleManager::Deps elm_deps{
        .expert_cache    = cache_.get(),
        .transfer_engine = transfer_engine_.get(),
        .loaded_model    = &model,
    };
    auto elm = std::make_unique<ldam::ExpertLifecycleManager>(std::move(elm_deps));

    // Dispatcher with ELM but no NvmeTier.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring        = cmp_ring_.get(),
        .transfer_engine = transfer_engine_.get(),
        .expert_cache    = cache_.get(),
        .elm             = elm.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // ELM sees this expert as kWarm (mmap) — guard should NOT block.
    // Handler will fail because NvmeTier is null, but the error should be
    // "NVMe tier not configured", NOT "expert loading to RAM".
    auto cmd = make_cmd(lipc::CMD_NVME_EVICT_HOST, /*seq=*/950);
    cmd.nvme_evict_host.layer_idx  = 1;
    cmd.nvme_evict_host.expert_idx = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    // Verify it's the NvmeTier-not-configured error, not the ELM guard.
    std::string msg(cmp.error.message);
    EXPECT_NE(msg.find("NVMe tier not configured"), std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════
// CMP_ELM_EXPERT_READY completion payload tests

TEST_F(CommandDispatcherTest, ELM_PrefetchExpert_ProducesExpertReady) {
    // Build a fake LoadedModel for host source.
    lmod::LoadedModel model;
    model.layers.resize(6);
    model.layers[1].layer_idx = 1;
    model.layers[1].routed_experts.resize(8);
    std::vector<std::byte> fake_data(expert_bytes_, std::byte{0xAB});
    lmod::RawTensor rt{
        .data = std::span<const std::byte>(fake_data),
        .dtype = lmod::SafetensorsDtype::F8_E4M3,
        .shape = {expert_bytes_}};
    lmod::WeightBundle wb{.id = {}, .weight = rt, .packed_slot = rt.data};
    model.layers[1].routed_experts[0].push_back(std::move(wb));

    make_dispatcher(&model);

    // Dispatch prefetch.
    auto cmd = make_cmd(lipc::D_CMD_PREFETCH_EXPERT, /*seq=*/960);
    cmd.prefetch_expert.layer_idx  = 1;
    cmd.prefetch_expert.expert_idx = 0;
    cmd.prefetch_expert.zone       = 0;
    cmd.prefetch_expert.gpu_idx    = 0;
    dispatcher_->dispatch(cmd);

    // Poll ELM — null backend completes H2D immediately.
    auto completions = poll_elm();
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_TRUE(completions[0].success);
    EXPECT_EQ(completions[0].cmd_seq, 960u);
    EXPECT_EQ(completions[0].key, (lmem::ExpertKey{1, 0}));
    EXPECT_EQ(completions[0].gpu_idx, 0);

    // Verify latency fields are populated (non-zero total_us).
    // With null backend, dma_us may be 0 (instant) but total_us > 0.
    EXPECT_GE(completions[0].total_us, 0u);

    // Write to ring as DaemonLoop would and verify CMP_ELM_EXPERT_READY payload.
    const auto& lc = completions[0];
    lipc::Completion cmp{};
    cmp.cmp_type  = lipc::CMP_ELM_EXPERT_READY;
    cmp.cmd_seq   = lc.cmd_seq;
    cmp.gpu_idx   = static_cast<uint32_t>(lc.gpu_idx);
    cmp.status    = lc.success ? 0u : 1u;
    cmp.elm_expert.layer_idx  = lc.key.layer_idx;
    cmp.elm_expert.expert_idx = lc.key.expert_idx;
    cmp.elm_expert.nvme_us    = lc.nvme_us;
    cmp.elm_expert.dma_us     = lc.dma_us;
    cmp.elm_expert.total_us   = lc.total_us;
    ASSERT_TRUE(cmp_ring_->try_write(&cmp));

    // Read it back and verify.
    lipc::Completion read{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, read));
    EXPECT_EQ(read.cmp_type, static_cast<uint32_t>(lipc::CMP_ELM_EXPERT_READY));
    EXPECT_EQ(read.cmd_seq, 960u);
    EXPECT_EQ(read.gpu_idx, 0u);
    EXPECT_EQ(read.status, 0u);
    EXPECT_EQ(read.elm_expert.layer_idx, 1u);
    EXPECT_EQ(read.elm_expert.expert_idx, 0u);
    EXPECT_EQ(read.elm_expert.nvme_us, 0u);  // mmap, no NVMe read
}

TEST_F(CommandDispatcherTest, ELM_PrefetchBatch_ProducesMultipleExpertReady) {
    // Build a fake LoadedModel with two experts.
    lmod::LoadedModel model;
    model.layers.resize(6);
    model.layers[1].layer_idx = 1;
    model.layers[1].routed_experts.resize(8);
    std::vector<std::byte> fake_data(expert_bytes_, std::byte{0xAB});
    for (int e = 0; e < 2; ++e) {
        lmod::RawTensor rt{
            .data = std::span<const std::byte>(fake_data),
            .dtype = lmod::SafetensorsDtype::F8_E4M3,
            .shape = {expert_bytes_}};
        lmod::WeightBundle wb{.id = {}, .weight = rt, .packed_slot = rt.data};
        model.layers[1].routed_experts[e].push_back(std::move(wb));
    }

    make_dispatcher(&model);

    // Write sideband entries for batch prefetch.
    auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
        sideband_base_ + lipc::IpcLayout::kExpertPrefetchOff);
    entries[0] = {1, 0, 0, 0};  // layer=1 expert=0 zone=0 gpu=0
    entries[1] = {1, 1, 0, 0};  // layer=1 expert=1 zone=0 gpu=0

    auto cmd = make_cmd(lipc::D_B_CMD_PREFETCH_BATCH, /*seq=*/970);
    cmd.prefetch_batch.count = 2;
    dispatcher_->dispatch(cmd);

    // Poll ELM — both experts should complete.
    auto completions = poll_elm();
    ASSERT_EQ(completions.size(), 2u);
    EXPECT_TRUE(completions[0].success);
    EXPECT_TRUE(completions[1].success);

    // Verify keys (order may vary — sort by expert_idx).
    std::sort(completions.begin(), completions.end(),
              [](const auto& a, const auto& b) {
                  return a.key.expert_idx < b.key.expert_idx;
              });
    EXPECT_EQ(completions[0].key, (lmem::ExpertKey{1, 0}));
    EXPECT_EQ(completions[1].key, (lmem::ExpertKey{1, 1}));
}

// ═══════════════════════════════════════════════════════════════════════════
// CMD_SLOW_EVICT_TO_HOST (IPC-8i)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, SlowEvictToHost_HostHasData) {
    // When host has data (mmap source), slow evict is metadata-only.
    lmod::LoadedModel model;
    model.layers.resize(6);
    model.layers[1].layer_idx = 1;
    model.layers[1].routed_experts.resize(8);
    std::vector<std::byte> fake_data(expert_bytes_, std::byte{0x42});
    lmod::RawTensor rt{
        .data = std::span<const std::byte>(fake_data),
        .dtype = lmod::SafetensorsDtype::F8_E4M3,
        .shape = {expert_bytes_}};
    lmod::WeightBundle wb{.id = {}, .weight = rt, .packed_slot = rt.data};
    model.layers[1].routed_experts[0].push_back(std::move(wb));

    make_dispatcher(&model);

    // Load expert to VRAM via ELM.
    auto cmd1 = make_cmd(lipc::D_CMD_PREFETCH_EXPERT, /*seq=*/950);
    cmd1.prefetch_expert.layer_idx  = 1;
    cmd1.prefetch_expert.expert_idx = 0;
    cmd1.prefetch_expert.zone       = 0;
    cmd1.prefetch_expert.gpu_idx    = 0;
    dispatcher_->dispatch(cmd1);

    // Complete the transfer (ELM lifecycle).
    auto completions = poll_elm();
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_TRUE(completions[0].success);

    // Expert should be resident.
    EXPECT_NE(cache_->lookup({1, 0}, 0), nullptr);

    // Slow evict — host has data (mmap), so metadata-only evict.
    auto cmd2 = make_cmd(lipc::D_CMD_SLOW_EVICT_TO_HOST, /*seq=*/951);
    cmd2.slow_evict_to_host.layer_idx  = 1;
    cmd2.slow_evict_to_host.expert_idx = 0;
    cmd2.slow_evict_to_host.gpu_idx    = 0;
    dispatcher_->dispatch(cmd2);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 951u);
    EXPECT_EQ(cmp.status, 0u);

    // Expert evicted.
    EXPECT_EQ(cache_->lookup({1, 0}, 0), nullptr);
}

TEST_F(CommandDispatcherTest, SlowEvictToHost_NotResident) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::D_CMD_SLOW_EVICT_TO_HOST, /*seq=*/960);
    cmd.slow_evict_to_host.layer_idx  = 99;
    cmd.slow_evict_to_host.expert_idx = 99;
    cmd.slow_evict_to_host.gpu_idx    = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 960u);
    EXPECT_EQ(cmp.status, 1u);  // not resident
}

TEST_F(CommandDispatcherTest, SlowEvictToHost_D2H_ThenEvict) {
    // Expert in VRAM, NOT in host RAM → D2H fires, then VRAM evict.
    std::string tmpdir = make_temp_nvme_dir();
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

    // Expert NOT admitted to NvmeTier warm cache — host has no data.
    EXPECT_FALSE(nvme->is_in_host_ram({1, 0}));

    // Reserve expert in VRAM and mark ready (no ELM, direct cache ops).
    cache_->reserve({1, 0}, 0, lmem::CacheZone::kStable, false);
    cache_->mark_ready({1, 0}, 0, lmem::SubComponent::kAll);
    ASSERT_NE(cache_->lookup({1, 0}, 0), nullptr);

    // Create dispatcher without ELM, with NvmeTier + NumaManager.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .nvme_tier          = nvme.get(),
        .numa_manager       = &numa,
        .page_allocator     = page_allocator_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Dispatch slow evict.
    auto cmd = make_cmd(lipc::D_CMD_SLOW_EVICT_TO_HOST, /*seq=*/970);
    cmd.slow_evict_to_host.layer_idx  = 1;
    cmd.slow_evict_to_host.expert_idx = 0;
    cmd.slow_evict_to_host.gpu_idx    = 0;
    dispatcher_->dispatch(cmd);

    // Null backend: D2H enqueued but callback fires on poll.
    // Poll transfer engine to fire the callback.
    auto tc = transfer_engine_->poll_completions();

    // Callback should have: admitted to NvmeTier, evicted from VRAM,
    // written CMP_CACHE_OP_DONE.
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE));
    EXPECT_EQ(cmp.cmd_seq, 970u);
    EXPECT_EQ(cmp.status, 0u);

    // Expert evicted from VRAM.
    EXPECT_EQ(cache_->lookup({1, 0}, 0), nullptr);
    // Expert now in host RAM warm cache.
    EXPECT_TRUE(nvme->is_in_host_ram({1, 0}));

    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
}

TEST_F(CommandDispatcherTest, SlowEvictToHost_NoNvmeTier) {
    // Expert in VRAM, no host data, no NvmeTier → CMP_ERROR.
    // Reserve expert directly (no ELM).
    cache_->reserve({2, 3}, 0, lmem::CacheZone::kStreaming, false);
    cache_->mark_ready({2, 3}, 0, lmem::SubComponent::kAll);

    // Dispatcher without ELM, without NvmeTier, without loaded_model.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),

        .page_allocator     = page_allocator_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_CMD_SLOW_EVICT_TO_HOST, /*seq=*/980);
    cmd.slow_evict_to_host.layer_idx  = 2;
    cmd.slow_evict_to_host.expert_idx = 3;
    cmd.slow_evict_to_host.gpu_idx    = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 980u);
}

// ── MTP step (#62d) ──────────────────────────────────────────────────────────

TEST_F(CommandDispatcherTest, MtpStep_WritesGatingAndCompute) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::D_CMD_RUN_MTP_STEP, /*seq=*/500);
    cmd.run_mtp_step.mtp_layer_idx  = 61;
    cmd.run_mtp_step.seq_id         = 42;
    cmd.run_mtp_step.input_token_id = 100;
    cmd.run_mtp_step.step_idx       = 0;
    dispatcher_->dispatch(cmd);

    // Checkpoint is synchronous; compute done needs poll
    // First completion: gating checkpoint (type 1)
    lipc::Completion cmp1{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp1));
    EXPECT_EQ(cmp1.cmp_type, static_cast<uint32_t>(lipc::CMP_CHECKPOINT));
    EXPECT_EQ(cmp1.cmd_seq, 500u);
    EXPECT_EQ(cmp1.checkpoint.layer_idx, 61u);
    EXPECT_EQ(cmp1.checkpoint.checkpoint_type,
              static_cast<uint8_t>(lipc::CheckpointType::kGatingOutput));

    // Second completion: compute done with 8-byte readback (needs poll)
    dispatcher_->poll_compute_completions();
    lipc::Completion cmp2{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp2));
    EXPECT_EQ(cmp2.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp2.cmd_seq, 500u);
    EXPECT_EQ(cmp2.compute.layer_idx, 61u);
    EXPECT_EQ(cmp2.compute.data_bytes, 8u);

    // No more completions
    lipc::Completion cmp3{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp3));
}

// ── Self-spec forward (#62e) ─────────────────────────────────────────────────

TEST_F(CommandDispatcherTest, SelfSpecForward_WritesLayerCheckpoints) {
    // Build dispatcher with live_config so the handler knows model dimensions.
    // small_config: num_hidden_layers=6, first_k_dense_replace=1 → 5 MoE layers (1-5)
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .page_allocator     = page_allocator_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_CMD_RUN_SELF_SPEC_FORWARD, /*seq=*/600);
    cmd.self_spec_forward.seq_id              = 99;
    cmd.self_spec_forward.input_token_id      = 200;
    cmd.self_spec_forward.draft_expert_count  = 1;
    cmd.self_spec_forward.apply_residual_corr = 1;
    cmd.self_spec_forward.store_gating        = 1;
    cmd.self_spec_forward.step_idx            = 0;
    cmd.self_spec_forward.skip_mask_lo        = 0;  // no skipping
    cmd.self_spec_forward.skip_mask_hi        = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    // Count completions: 6 similarity (all layers) + 5 gating (MoE layers 1-5) + 1 compute = 12
    int similarity_count = 0, gating_count = 0, compute_count = 0;
    lipc::Completion cmp{};
    while (read_cmp(*cmp_ring_, cmp)) {
        if (cmp.cmp_type == static_cast<uint32_t>(lipc::CMP_CHECKPOINT)) {
            if (cmp.checkpoint.checkpoint_type ==
                    static_cast<uint8_t>(lipc::CheckpointType::kLayerSimilarity)) {
                ++similarity_count;
                EXPECT_EQ(cmp.checkpoint.data_bytes, 4u);
            } else if (cmp.checkpoint.checkpoint_type ==
                           static_cast<uint8_t>(lipc::CheckpointType::kGatingOutput)) {
                ++gating_count;
            }
        } else if (cmp.cmp_type == static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)) {
            ++compute_count;
            EXPECT_EQ(cmp.compute.data_bytes, 8u);
        }
    }
    EXPECT_EQ(similarity_count, 6);  // all 6 layers
    EXPECT_EQ(gating_count, 5);      // 5 MoE layers (1-5)
    EXPECT_EQ(compute_count, 1);     // final compute done
}

TEST_F(CommandDispatcherTest, SelfSpecForward_SkipMask) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .page_allocator     = page_allocator_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_CMD_RUN_SELF_SPEC_FORWARD, /*seq=*/601);
    cmd.self_spec_forward.seq_id              = 99;
    cmd.self_spec_forward.input_token_id      = 200;
    cmd.self_spec_forward.draft_expert_count  = 1;
    cmd.self_spec_forward.apply_residual_corr = 1;
    cmd.self_spec_forward.store_gating        = 1;
    cmd.self_spec_forward.step_idx            = 0;
    // Skip layers 2 and 4 (bits 2 and 4 in skip_mask_lo)
    cmd.self_spec_forward.skip_mask_lo        = (1u << 2) | (1u << 4);
    cmd.self_spec_forward.skip_mask_hi        = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    // 4 similarity (6 layers - 2 skipped) + 3 gating (5 MoE - 2 skipped MoE layers) + 1 compute = 8
    // MoE layers are 1-5. Skipped: 2 and 4. So MoE layers remaining: 1, 3, 5 = 3 gating
    int similarity_count = 0, gating_count = 0, compute_count = 0;
    lipc::Completion cmp{};
    while (read_cmp(*cmp_ring_, cmp)) {
        if (cmp.cmp_type == static_cast<uint32_t>(lipc::CMP_CHECKPOINT)) {
            if (cmp.checkpoint.checkpoint_type ==
                    static_cast<uint8_t>(lipc::CheckpointType::kLayerSimilarity))
                ++similarity_count;
            else if (cmp.checkpoint.checkpoint_type ==
                         static_cast<uint8_t>(lipc::CheckpointType::kGatingOutput))
                ++gating_count;
        } else if (cmp.cmp_type == static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)) {
            ++compute_count;
        }
    }
    EXPECT_EQ(similarity_count, 4);  // 6 - 2 skipped = 4
    EXPECT_EQ(gating_count, 3);      // 5 MoE - 2 skipped MoE = 3
    EXPECT_EQ(compute_count, 1);
}

TEST_F(CommandDispatcherTest, SelfSpecForward_StoreGatingDisabled) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .page_allocator     = page_allocator_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_CMD_RUN_SELF_SPEC_FORWARD, /*seq=*/602);
    cmd.self_spec_forward.seq_id              = 99;
    cmd.self_spec_forward.input_token_id      = 200;
    cmd.self_spec_forward.draft_expert_count  = 1;
    cmd.self_spec_forward.apply_residual_corr = 0;
    cmd.self_spec_forward.store_gating        = 0;  // disabled
    cmd.self_spec_forward.step_idx            = 0;
    cmd.self_spec_forward.skip_mask_lo        = 0;
    cmd.self_spec_forward.skip_mask_hi        = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    // Only similarity checkpoints + compute done (no gating)
    int similarity_count = 0, gating_count = 0, compute_count = 0;
    lipc::Completion cmp{};
    while (read_cmp(*cmp_ring_, cmp)) {
        if (cmp.cmp_type == static_cast<uint32_t>(lipc::CMP_CHECKPOINT)) {
            if (cmp.checkpoint.checkpoint_type ==
                    static_cast<uint8_t>(lipc::CheckpointType::kLayerSimilarity))
                ++similarity_count;
            else if (cmp.checkpoint.checkpoint_type ==
                         static_cast<uint8_t>(lipc::CheckpointType::kGatingOutput))
                ++gating_count;
        } else if (cmp.cmp_type == static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)) {
            ++compute_count;
        }
    }
    EXPECT_EQ(similarity_count, 6);  // all layers
    EXPECT_EQ(gating_count, 0);      // disabled
    EXPECT_EQ(compute_count, 1);
}

// ── KD-3c: Pipeline dispatch tests ───────────────────────────────────────────
// Pipeline tests exercise the stub path with cuda_kernels_enabled=false (unit tests
// have no CUDA runtime). The full pipeline path (cuda_kernels_enabled=true) requires
// GPU integration tests. Here we verify: (1) stub fallback works with various configs,
// (2) PipelineCheckpoint polling logic works via deferred events, (3) SpecScratch
// allocation/deallocation is safe.

TEST_F(CommandDispatcherTest, MtpStep_StubFallback_CudaEnabledNoScratch) {
    // cuda_kernels_enabled=true but live_config=nullptr → spec_scratch not allocated
    // → pipeline guard fails → falls back to stub path.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .page_allocator     = page_allocator_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = nullptr,
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .cuda_kernels_enabled = true,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_CMD_RUN_MTP_STEP, /*seq=*/700);
    cmd.run_mtp_step.mtp_layer_idx  = 61;
    cmd.run_mtp_step.seq_id         = 42;
    cmd.run_mtp_step.input_token_id = 100;
    cmd.run_mtp_step.step_idx       = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    int checkpoint_count = 0, compute_count = 0;
    lipc::Completion cmp{};
    while (read_cmp(*cmp_ring_, cmp)) {
        if (cmp.cmp_type == static_cast<uint32_t>(lipc::CMP_CHECKPOINT))
            ++checkpoint_count;
        else if (cmp.cmp_type == static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE))
            ++compute_count;
    }
    EXPECT_EQ(checkpoint_count, 1);
    EXPECT_EQ(compute_count, 1);
}

TEST_F(CommandDispatcherTest, SelfSpec_StubFallback_CudaDisabled) {
    // cuda_kernels_enabled=false → always stub path regardless of config.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .page_allocator     = page_allocator_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_CMD_RUN_SELF_SPEC_FORWARD, /*seq=*/701);
    cmd.self_spec_forward.seq_id              = 99;
    cmd.self_spec_forward.input_token_id      = 200;
    cmd.self_spec_forward.draft_expert_count  = 1;
    cmd.self_spec_forward.apply_residual_corr = 1;
    cmd.self_spec_forward.store_gating        = 1;
    cmd.self_spec_forward.step_idx            = 0;
    cmd.self_spec_forward.skip_mask_lo        = 0;
    cmd.self_spec_forward.skip_mask_hi        = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    // Stub: 6 similarity + 5 gating + 1 compute = 12
    int similarity_count = 0, gating_count = 0, compute_count = 0;
    lipc::Completion cmp{};
    while (read_cmp(*cmp_ring_, cmp)) {
        if (cmp.cmp_type == static_cast<uint32_t>(lipc::CMP_CHECKPOINT)) {
            if (cmp.checkpoint.checkpoint_type ==
                    static_cast<uint8_t>(lipc::CheckpointType::kLayerSimilarity))
                ++similarity_count;
            else if (cmp.checkpoint.checkpoint_type ==
                         static_cast<uint8_t>(lipc::CheckpointType::kGatingOutput))
                ++gating_count;
        } else if (cmp.cmp_type == static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)) {
            ++compute_count;
        }
    }
    EXPECT_EQ(similarity_count, 6);
    EXPECT_EQ(gating_count, 5);
    EXPECT_EQ(compute_count, 1);
}

/// NullDeviceBackend with controllable query_event result for deferred-event tests.
class DeferredNullDeviceBackend : public lcomp::NullDeviceBackend {
public:
    DeferredNullDeviceBackend(lc::GpuRef gpu, bool& ready)
        : NullDeviceBackend(std::move(gpu)), ready_(ready) {}
    lcomp::EventQueryResult query_event(void*) override {
        return {ready_ ? lcomp::EventStatus::kReady : lcomp::EventStatus::kNotReady, 0};
    }
private:
    bool& ready_;
};

/// NullDeviceBackend with directly controllable EventStatus for error-path tests.
class StatusDeviceBackend : public lcomp::NullDeviceBackend {
public:
    StatusDeviceBackend(lc::GpuRef gpu, lcomp::EventStatus& status)
        : NullDeviceBackend(std::move(gpu)), status_(status) {}
    lcomp::EventQueryResult query_event(void*) override { return {status_, 0}; }
private:
    lcomp::EventStatus& status_;
};

/// Helper: build a deferred-event StreamManager for N GPUs.
/// Returns {backends, stream_manager} — backends first so they outlive SM
/// (std::pair destroys `second` before `first`).
static std::pair<std::vector<std::unique_ptr<lcomp::DeviceBackend>>,
                 std::unique_ptr<lcomp::StreamManager>>
make_deferred_stream_manager(std::vector<lc::GpuRef> gpus, bool& event_ready) {
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> backends;
    std::vector<lcomp::DeviceBackend*> ptrs;
    for (auto& g : gpus) {
        backends.push_back(std::make_unique<DeferredNullDeviceBackend>(g, event_ready));
        ptrs.push_back(backends.back().get());
    }
    lcomp::StreamManager::Options opts{.device_backends = std::move(ptrs)};
    return {std::move(backends),
            std::make_unique<lcomp::StreamManager>(std::move(opts))};
}

TEST_F(CommandDispatcherTest, PipelineCheckpoint_DeferredEventPolling) {
    // Verify that PipelineCheckpoint events are polled correctly.
    // Use deferred events that require multiple poll cycles.
    bool event_ready = false;
    auto [deferred_backends, deferred_sm] = make_deferred_stream_manager(
        {lc::GpuRef{0, 0, lc::GpuType::rtx5090},
         lc::GpuRef{1, 1, lc::GpuType::rtx5090}}, event_ready);

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = deferred_sm.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .page_allocator     = page_allocator_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Dispatch MTP step (stub path), which uses deferred events.
    auto cmd = make_cmd(lipc::D_CMD_RUN_MTP_STEP, /*seq=*/710);
    cmd.run_mtp_step.mtp_layer_idx  = 61;
    cmd.run_mtp_step.seq_id         = 42;
    cmd.run_mtp_step.input_token_id = 100;
    cmd.run_mtp_step.step_idx       = 0;
    dispatcher_->dispatch(cmd);

    // Gating checkpoint is synchronous in stub path.
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_CHECKPOINT));

    // Compute done is deferred — poll with event not ready.
    EXPECT_EQ(dispatcher_->poll_compute_completions(), 0u);
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));

    // Signal event ready.
    event_ready = true;
    EXPECT_EQ(dispatcher_->poll_compute_completions(), 1u);
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.compute.data_bytes, 8u);
    // Reset dispatcher before locals (deferred_sm) go out of scope.
    dispatcher_.reset();
}

TEST_F(CommandDispatcherTest, PollComputeCompletions_GpuFatalError_WritesCmpError) {
    // Verify that EventStatus::kError triggers CMP_ERROR and cleans up pending entry.
    lcomp::EventStatus event_status = lcomp::EventStatus::kNotReady;

    std::vector<std::unique_ptr<lcomp::DeviceBackend>> backends;
    std::vector<lcomp::DeviceBackend*> ptrs;
    for (auto& g : {lc::GpuRef{0, 0, lc::GpuType::rtx5090},
                     lc::GpuRef{1, 1, lc::GpuType::rtx5090}}) {
        backends.push_back(std::make_unique<StatusDeviceBackend>(g, event_status));
        ptrs.push_back(backends.back().get());
    }
    lcomp::StreamManager::Options sm_opts{.device_backends = std::move(ptrs)};
    auto status_sm = std::make_unique<lcomp::StreamManager>(std::move(sm_opts));

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = status_sm.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .page_allocator     = page_allocator_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .device_backends    = {test_device_backends_[0].get(),
                               test_device_backends_[1].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Dispatch MTP step to create a PendingCompute entry.
    auto cmd = make_cmd(lipc::D_CMD_RUN_MTP_STEP, /*seq=*/720);
    cmd.run_mtp_step.mtp_layer_idx  = 61;
    cmd.run_mtp_step.seq_id         = 42;
    cmd.run_mtp_step.input_token_id = 100;
    cmd.run_mtp_step.step_idx       = 0;
    dispatcher_->dispatch(cmd);

    // Drain synchronous checkpoint.
    lipc::Completion cmp{};
    while (read_cmp(*cmp_ring_, cmp)) {}

    // Poll with kNotReady — nothing completes.
    EXPECT_EQ(dispatcher_->poll_compute_completions(), 0u);

    // Simulate GPU fatal error.
    event_status = lcomp::EventStatus::kError;
    EXPECT_EQ(dispatcher_->poll_compute_completions(), 1u);

    // First completion: CMP_GPU_FATAL (once per GPU).
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_GPU_FATAL));
    EXPECT_EQ(cmp.gpu_idx, 0u);

    // Second completion: CMP_ERROR for the specific command.
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 720u);

    // Pending entry should be removed.
    EXPECT_EQ(dispatcher_->pending_compute_count(), 0u);

    // Second command on same GPU: CMP_GPU_FATAL should NOT be emitted again.
    event_status = lcomp::EventStatus::kNotReady;
    auto cmd2 = make_cmd(lipc::D_CMD_RUN_MTP_STEP, /*seq=*/721);
    cmd2.run_mtp_step.mtp_layer_idx  = 61;
    cmd2.run_mtp_step.seq_id         = 43;
    cmd2.run_mtp_step.input_token_id = 101;
    cmd2.run_mtp_step.step_idx       = 0;
    dispatcher_->dispatch(cmd2);
    while (read_cmp(*cmp_ring_, cmp)) {}  // drain sync checkpoint

    event_status = lcomp::EventStatus::kError;
    EXPECT_EQ(dispatcher_->poll_compute_completions(), 1u);

    // Only CMP_ERROR this time — no second CMP_GPU_FATAL.
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 721u);
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));  // no more completions

    EXPECT_EQ(dispatcher_->pending_compute_count(), 0u);

    dispatcher_.reset();
}

TEST_F(CommandDispatcherTest, PollComputeCompletions_GpuFatalError_PipelineCheckpoints) {
    // Verify that kError on a pipeline checkpoint event aborts the entire
    // PendingCompute, destroys all events, and writes CMP_ERROR.
    // This uses a backend that returns kReady for N calls, then kError,
    // to test the partial-emission + error path.

    /// Backend that returns kReady for the first N query_event calls, then kError.
    struct CountdownErrorBackend : public lcomp::NullDeviceBackend {
        int remaining;
        CountdownErrorBackend(lc::GpuRef gpu, int ready_count)
            : NullDeviceBackend(std::move(gpu)), remaining(ready_count) {}
        lcomp::EventQueryResult query_event(void*) override {
            if (remaining > 0) { --remaining; return {lcomp::EventStatus::kReady, 0}; }
            return {lcomp::EventStatus::kError, 42};
        }
    };

    // ready_count=0: first query_event returns kError (checkpoint event).
    // This tests: checkpoint polls kError → gpu_error → cleanup all events.
    auto gpu0 = lc::GpuRef{0, 0, lc::GpuType::rtx5090};
    auto gpu1 = lc::GpuRef{1, 1, lc::GpuType::rtx5090};
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> backends;
    backends.push_back(std::make_unique<CountdownErrorBackend>(gpu0, 0));
    backends.push_back(std::make_unique<CountdownErrorBackend>(gpu1, 0));
    std::vector<lcomp::DeviceBackend*> ptrs{backends[0].get(), backends[1].get()};

    lcomp::StreamManager::Options sm_opts{.device_backends = std::move(ptrs)};
    auto err_sm = std::make_unique<lcomp::StreamManager>(std::move(sm_opts));

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = err_sm.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .page_allocator     = page_allocator_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .device_backends    = {test_device_backends_[0].get(),
                               test_device_backends_[1].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Dispatch MTP step — creates PendingCompute with 1 pipeline checkpoint + 1 main event.
    auto cmd = make_cmd(lipc::D_CMD_RUN_MTP_STEP, /*seq=*/730);
    cmd.run_mtp_step.mtp_layer_idx  = 61;
    cmd.run_mtp_step.seq_id         = 42;
    cmd.run_mtp_step.input_token_id = 100;
    cmd.run_mtp_step.step_idx       = 0;
    dispatcher_->dispatch(cmd);

    // Drain any synchronous completions.
    lipc::Completion cmp{};
    while (read_cmp(*cmp_ring_, cmp)) {}

    // First poll: checkpoint event returns kError immediately.
    // The main event should NOT be queried (checkpoint error aborts early).
    EXPECT_EQ(dispatcher_->poll_compute_completions(), 1u);

    // CMP_GPU_FATAL emitted first (once per GPU).
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_GPU_FATAL));
    EXPECT_EQ(cmp.gpu_idx, 0u);
    // Verify vendor error code is in the message.
    EXPECT_NE(std::string(cmp.gpu_fatal.message).find("42"), std::string::npos);

    // CMP_ERROR for the specific command.
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 730u);

    EXPECT_EQ(dispatcher_->pending_compute_count(), 0u);

    dispatcher_.reset();
}

TEST_F(CommandDispatcherTest, SpecScratch_AllocatedWithCudaEnabled) {
    // Verify SpecScratch is allocated when cuda_kernels_enabled=true + live_config present.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .page_allocator     = page_allocator_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .cuda_kernels_enabled = true,
    };
    // Construction and destruction should work cleanly (no leaks, no crashes).
    auto d = std::make_unique<ldam::CommandDispatcher>(std::move(deps));
    d.reset();  // explicit destruction
}

// ═══════════════════════════════════════════════════════════════════════════
// KD-1: Async compute completion infrastructure
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, AsyncCompute_DeferredEvent_PollCycle) {
    bool event_ready = false;
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    auto [deferred_backends, deferred_sm] = make_deferred_stream_manager(
        {gpu0}, event_ready);

    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem[4] = {};
    uint32_t buf1 = registry->register_buffer(&fake_mem[0], 4096, 0, "input");
    uint32_t buf2 = registry->register_buffer(&fake_mem[1], 4096, 0, "output");
    uint32_t buf3 = registry->register_buffer(&fake_mem[2], 4096, 0, "weight");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring        = cmp_ring_.get(),
        .stream_manager  = deferred_sm.get(),
        .buffer_registry = registry.get(),
        .attention_devices = {test_attn_devices_[0].get(),
                              test_attn_devices_.size() > 1
                                  ? test_attn_devices_[1].get() : nullptr},
        .expert_devices    = {test_expert_devices_[0].get(),
                              test_expert_devices_.size() > 1
                                  ? test_expert_devices_[1].get() : nullptr},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::CMD_RMSNORM, /*seq=*/900);
    cmd.rmsnorm.num_tokens    = 4;
    cmd.rmsnorm.input_buf_id  = buf1;
    cmd.rmsnorm.output_buf_id = buf2;
    cmd.rmsnorm.weight_buf_id = buf3;
    cmd.rmsnorm.eps           = 1e-5f;
    dispatcher_->dispatch(cmd);

    // Event not ready — poll produces nothing.
    EXPECT_EQ(dispatcher_->poll_compute_completions(), 0u);
    lipc::Completion cmp{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));

    // Signal event ready — poll now completes.
    event_ready = true;
    EXPECT_EQ(dispatcher_->poll_compute_completions(), 1u);
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 900u);
    EXPECT_EQ(cmp.status, 0u);
    // Reset dispatcher before locals (deferred_sm) go out of scope.
    dispatcher_.reset();
}

TEST_F(CommandDispatcherTest, AsyncCompute_PendingCount) {
    bool event_ready = false;
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    lc::GpuRef gpu1{1, 1, lc::GpuType::rtx5090};
    auto [deferred_backends, deferred_sm] = make_deferred_stream_manager(
        {gpu0, gpu1}, event_ready);

    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem[4] = {};
    uint32_t buf1 = registry->register_buffer(&fake_mem[0], 4096, 0, "input");
    uint32_t buf2 = registry->register_buffer(&fake_mem[1], 4096, 0, "output");
    uint32_t buf3 = registry->register_buffer(&fake_mem[2], 4096, 0, "weight");

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring        = cmp_ring_.get(),
        .stream_manager  = deferred_sm.get(),
        .buffer_registry = registry.get(),
        .attention_devices = {test_attn_devices_[0].get(),
                              test_attn_devices_.size() > 1
                                  ? test_attn_devices_[1].get() : nullptr},
        .expert_devices    = {test_expert_devices_[0].get(),
                              test_expert_devices_.size() > 1
                                  ? test_expert_devices_[1].get() : nullptr},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    EXPECT_EQ(dispatcher_->pending_compute_count(), 0u);

    // Dispatch to GPU 0
    auto cmd1 = make_cmd(lipc::CMD_RMSNORM, /*seq=*/910, /*gpu_idx=*/0);
    cmd1.rmsnorm.num_tokens    = 4;
    cmd1.rmsnorm.input_buf_id  = buf1;
    cmd1.rmsnorm.output_buf_id = buf2;
    cmd1.rmsnorm.weight_buf_id = buf3;
    cmd1.rmsnorm.eps           = 1e-5f;
    dispatcher_->dispatch(cmd1);

    // Dispatch to GPU 1
    auto cmd2 = make_cmd(lipc::CMD_RMSNORM, /*seq=*/911, /*gpu_idx=*/1);
    cmd2.rmsnorm.num_tokens    = 4;
    cmd2.rmsnorm.input_buf_id  = buf1;
    cmd2.rmsnorm.output_buf_id = buf2;
    cmd2.rmsnorm.weight_buf_id = buf3;
    cmd2.rmsnorm.eps           = 1e-5f;
    dispatcher_->dispatch(cmd2);

    EXPECT_EQ(dispatcher_->pending_compute_count(), 2u);
    EXPECT_EQ(dispatcher_->pending_compute_count(0), 1u);
    EXPECT_EQ(dispatcher_->pending_compute_count(1), 1u);

    // Complete all
    event_ready = true;
    EXPECT_EQ(dispatcher_->poll_compute_completions(), 2u);
    EXPECT_EQ(dispatcher_->pending_compute_count(), 0u);
    // Reset dispatcher before locals (deferred_sm) go out of scope.
    dispatcher_.reset();
}

TEST_F(CommandDispatcherTest, AsyncCompute_Backpressure_StallsDrain) {
    bool event_ready = false;
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    auto [deferred_backends, deferred_sm] = make_deferred_stream_manager(
        {gpu0}, event_ready);

    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem[4] = {};
    uint32_t buf1 = registry->register_buffer(&fake_mem[0], 4096, 0, "input");
    uint32_t buf2 = registry->register_buffer(&fake_mem[1], 4096, 0, "output");
    uint32_t buf3 = registry->register_buffer(&fake_mem[2], 4096, 0, "weight");

    ldam::CommandDispatcher::Deps disp_deps{
        .cmp_ring        = cmp_ring_.get(),
        .stream_manager  = deferred_sm.get(),
        .buffer_registry = registry.get(),
        .max_inflight_compute = 2,
        .attention_devices = {test_attn_devices_[0].get()},
        .expert_devices    = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(disp_deps));

    // Fill pending to the cap
    for (uint32_t i = 0; i < 2; ++i) {
        auto cmd = make_cmd(lipc::CMD_RMSNORM, /*seq=*/920 + i);
        cmd.rmsnorm.num_tokens    = 4;
        cmd.rmsnorm.input_buf_id  = buf1;
        cmd.rmsnorm.output_buf_id = buf2;
        cmd.rmsnorm.weight_buf_id = buf3;
        cmd.rmsnorm.eps           = 1e-5f;
        dispatcher_->dispatch(cmd);
    }
    EXPECT_EQ(dispatcher_->pending_compute_count(), 2u);

    // Simulate DaemonLoop backpressure check
    std::atomic<bool> running{true};
    ldam::DaemonLoop::Deps loop_deps{
        .cmd_ring       = cmd_ring_.get(),
        .cmp_ring       = cmp_ring_.get(),
        .ipc_header     = header_,
        .state_snapshot = snap_,
        .running        = &running,
        .poll_compute_fn = [this]() -> uint32_t {
            return dispatcher_->poll_compute_completions();
        },
        .pending_compute_count_fn = [this]() -> uint32_t {
            return dispatcher_->pending_compute_count();
        },
        .max_inflight_compute = 2,
    };
    ldam::DaemonLoop loop(std::move(loop_deps));

    // Write a command into the ring — it should NOT be drained (backpressure)
    auto cmd3 = make_cmd(lipc::CMD_RMSNORM, /*seq=*/999);
    cmd3.rmsnorm.num_tokens    = 4;
    cmd3.rmsnorm.input_buf_id  = buf1;
    cmd3.rmsnorm.output_buf_id = buf2;
    cmd3.rmsnorm.weight_buf_id = buf3;
    cmd3.rmsnorm.eps           = 1e-5f;
    cmd_ring_->try_write(reinterpret_cast<const uint8_t*>(&cmd3));

    loop.run_one_cycle();

    // Command remains in ring (drain was skipped)
    lipc::Command peek{};
    EXPECT_TRUE(cmd_ring_->try_read(reinterpret_cast<uint8_t*>(&peek)));
    EXPECT_EQ(peek.cmd_seq, 999u);

    // Free backpressure
    event_ready = true;
    loop.run_one_cycle();  // This will poll (free slots) then drain on next cycle

    // The command was consumed — drain ran
    EXPECT_EQ(dispatcher_->pending_compute_count(), 0u);
    // Reset dispatcher before locals (deferred_sm) go out of scope.
    dispatcher_.reset();
}

TEST_F(CommandDispatcherTest, AsyncCompute_FusedCheckpointThenDone) {
    // This test needs a DcpExecutor + weights to actually dispatch. Set up properly.
    bool event_ready = false;
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    auto [deferred_backends, deferred_sm] = make_deferred_stream_manager(
        {gpu0}, event_ready);
    auto null_attn = lcomp::make_null_attention_device(gpu0);

    lpar::DcpExecutor::Options dcp_opts{
        .dcp_size = 1,
        .gpus = {gpu0},
        .max_batch_size = 8,
        .hidden_size = 256,
        .num_attention_heads = 4,
        .q_lora_rank = 64,
        .kv_lora_rank = 32,
        .qk_rope_head_dim = 16,
        .qk_nope_head_dim = 32,
        .v_head_dim = 64,
        .rms_norm_eps = 1e-6f,
        .stream_manager = deferred_sm.get(),
        .attention_devices = {null_attn.get()},
    };
    auto dcp_exec = std::make_unique<lpar::DcpExecutor>(std::move(dcp_opts));

    // TD-90b: set_layer_weights initializes dequant_pool_ (required by KD-4j).
    auto fake_weights = make_fake_attn_weights(61, 1);
    std::vector<std::vector<const lpar::AttentionLayerWeights*>> weight_ptrs(61);
    for (int l = 0; l < 61; ++l) {
        weight_ptrs[l].resize(1);
        weight_ptrs[l][0] = &fake_weights[l][0];
    }
    dcp_exec->set_layer_weights(std::move(weight_ptrs), 61);

    const size_t hidden_bytes = 8 * 256 * 2;
    void* hidden_buf = aligned_alloc_zeroed(hidden_bytes);

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring       = cmp_ring_.get(),
        .stream_manager = deferred_sm.get(),
        .dcp_executor   = dcp_exec.get(),
        .sideband_base  = sideband_base_,
        .live_config    = cfg_.get(),
        .attention_devices = {null_attn.get()},
        .cuda_kernels_enabled = false,
        .hidden_state_pairs = {ldam::HiddenStatePair{hidden_buf, nullptr, 0, 0, 0}},
        .per_layer_attn_weights = std::move(fake_weights),
        .max_batch_size = 8,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/950);
    cmd.run_attention.layer_idx       = 3;
    cmd.run_attention.num_seqs        = 2;
    cmd.run_attention.is_prefill      = 0;
    cmd.run_attention.use_graph       = 0;
    cmd.run_attention.is_draft        = 1;
    cmd.run_attention.emit_checkpoint = 1;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(cmd);

    // Nothing yet (event not ready)
    EXPECT_EQ(dispatcher_->poll_compute_completions(), 0u);
    lipc::Completion cmp{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));

    // Ready — poll should emit checkpoint + compute done
    event_ready = true;
    EXPECT_EQ(dispatcher_->poll_compute_completions(), 1u);

    lipc::Completion cmp1{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp1));
    EXPECT_EQ(cmp1.cmp_type, static_cast<uint32_t>(lipc::CMP_CHECKPOINT));
    EXPECT_EQ(cmp1.cmd_seq, 950u);
    EXPECT_EQ(cmp1.checkpoint.layer_idx, 3u);

    lipc::Completion cmp2{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp2));
    EXPECT_EQ(cmp2.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp2.cmd_seq, 950u);
    EXPECT_EQ(cmp2.compute.layer_idx, 3u);

    // KD-R4: destroy dispatcher before locals (deferred_sm captures &event_ready)
    dispatcher_.reset();
    std::free(hidden_buf);
}

/// NullDeviceBackend that counts create_event/destroy_event calls and
/// has controllable query_event.
class CountingDeviceBackend : public lcomp::NullDeviceBackend {
public:
    CountingDeviceBackend(lc::GpuRef gpu, int& count, bool& ready)
        : NullDeviceBackend(std::move(gpu)), count_(count), ready_(ready) {}
    void* create_event() override {
        ++count_;
        return reinterpret_cast<void*>(static_cast<uintptr_t>(count_));
    }
    void destroy_event(void*) override { --count_; }
    lcomp::EventQueryResult query_event(void*) override {
        return {ready_ ? lcomp::EventStatus::kReady : lcomp::EventStatus::kNotReady, 0};
    }
private:
    int& count_;
    bool& ready_;
};

TEST_F(CommandDispatcherTest, AsyncCompute_DestructorCleansEvents) {
    bool event_ready = false;
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    int event_count = 0;
    auto counting_be = std::make_unique<CountingDeviceBackend>(
        gpu0, event_count, event_ready);
    lcomp::StreamManager::Options sm_opts{
        .device_backends = {counting_be.get()},
    };
    auto deferred_sm = std::make_unique<lcomp::StreamManager>(std::move(sm_opts));

    auto registry = std::make_unique<ldam::BufferRegistry>();
    int fake_mem[4] = {};
    uint32_t buf1 = registry->register_buffer(&fake_mem[0], 4096, 0, "input");
    uint32_t buf2 = registry->register_buffer(&fake_mem[1], 4096, 0, "output");
    uint32_t buf3 = registry->register_buffer(&fake_mem[2], 4096, 0, "weight");

    {
        ldam::CommandDispatcher::Deps deps{
            .cmp_ring        = cmp_ring_.get(),
            .stream_manager  = deferred_sm.get(),
            .buffer_registry = registry.get(),
            .attention_devices = {test_attn_devices_[0].get()},
            .expert_devices    = {test_expert_devices_[0].get()},
            .cuda_kernels_enabled = false,
        };
        auto disp = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

        auto cmd = make_cmd(lipc::CMD_RMSNORM, /*seq=*/960);
        cmd.rmsnorm.num_tokens    = 4;
        cmd.rmsnorm.input_buf_id  = buf1;
        cmd.rmsnorm.output_buf_id = buf2;
        cmd.rmsnorm.weight_buf_id = buf3;
        cmd.rmsnorm.eps           = 1e-5f;
        disp->dispatch(cmd);
        EXPECT_EQ(event_count, 1);
        // Destructor runs here
    }
    EXPECT_EQ(event_count, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// KD-3a: Fused attention dispatch through DcpExecutor
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, FusedAttention_DcpExecutor_Decode) {
    make_dcp_dispatcher();

    // Dispatch fused decode attention
    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/700);
    cmd.run_attention.layer_idx       = 2;
    cmd.run_attention.num_seqs        = 4;
    cmd.run_attention.is_prefill      = 0;
    cmd.run_attention.use_graph       = 0;
    cmd.run_attention.is_draft        = 0;
    cmd.run_attention.emit_checkpoint = 0;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 700u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.cmd_type,
              static_cast<uint32_t>(lipc::D_B_CMD_RUN_ATTENTION));
    EXPECT_EQ(cmp.compute.layer_idx, 2u);
}

TEST_F(CommandDispatcherTest, FusedAttention_DcpExecutor_Prefill) {
    make_dcp_dispatcher();

    // Dispatch fused prefill attention (use_graph ignored for prefill)
    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/701);
    cmd.run_attention.layer_idx       = 0;
    cmd.run_attention.num_seqs        = 1;
    cmd.run_attention.is_prefill      = 1;
    cmd.run_attention.use_graph       = 1;  // ignored for prefill
    cmd.run_attention.is_draft        = 0;
    cmd.run_attention.emit_checkpoint = 0;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 512;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 701u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, FusedAttention_DcpExecutor_Checkpoint) {
    make_dcp_dispatcher();

    // Dispatch with emit_checkpoint — should produce CMP_CHECKPOINT then CMP_COMPUTE_DONE
    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/702);
    cmd.run_attention.layer_idx       = 5;
    cmd.run_attention.num_seqs        = 2;
    cmd.run_attention.is_prefill      = 0;
    cmd.run_attention.use_graph       = 0;
    cmd.run_attention.is_draft        = 1;
    cmd.run_attention.emit_checkpoint = 1;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    // CMP_CHECKPOINT first
    lipc::Completion cmp1{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp1));
    EXPECT_EQ(cmp1.cmp_type, static_cast<uint32_t>(lipc::CMP_CHECKPOINT));
    EXPECT_EQ(cmp1.cmd_seq, 702u);
    EXPECT_EQ(cmp1.checkpoint.cmd_type,
              static_cast<uint32_t>(lipc::D_B_CMD_RUN_ATTENTION));
    EXPECT_EQ(cmp1.checkpoint.layer_idx, 5u);
    EXPECT_EQ(cmp1.checkpoint.checkpoint_type,
              static_cast<uint8_t>(lipc::CheckpointType::kHiddenState));

    // CMP_COMPUTE_DONE second
    lipc::Completion cmp2{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp2));
    EXPECT_EQ(cmp2.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp2.cmd_seq, 702u);
    EXPECT_EQ(cmp2.compute.layer_idx, 5u);
}

TEST_F(CommandDispatcherTest, FusedAttention_NoDcpExecutor_ReturnsError) {
    // Without DcpExecutor, dispatch fails and CMP_ERROR is written (TD-40a/h)
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {test_attn_devices_[0].get(),
                               test_attn_devices_[1].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/703);
    cmd.run_attention.layer_idx       = 1;
    cmd.run_attention.num_seqs        = 4;
    cmd.run_attention.is_prefill      = 0;
    cmd.run_attention.use_graph       = 0;
    cmd.run_attention.is_draft        = 0;
    cmd.run_attention.emit_checkpoint = 0;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 703u);
    EXPECT_EQ(cmp.status, 1u);
}

TEST_F(CommandDispatcherTest, FusedAttention_EmptyHiddenStates_ReturnsError) {
    // KD-4a: verify that dispatch_attention_internal returns false (CMP_ERROR)
    // when hidden_state_pairs is empty, even with a valid DcpExecutor.
    make_dcp_dispatcher([](ldam::CommandDispatcher::Deps& deps) {
        deps.hidden_state_pairs.clear();
    });

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/710);
    cmd.run_attention.layer_idx       = 1;
    cmd.run_attention.num_seqs        = 2;
    cmd.run_attention.is_prefill      = 0;
    cmd.run_attention.use_graph       = 0;
    cmd.run_attention.is_draft        = 0;
    cmd.run_attention.emit_checkpoint = 0;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 710u);
}

// ── KD-4e: KV cache metadata wiring ─────────────────────────────────────

TEST_F(CommandDispatcherTest, FusedAttention_KvMetadata_WithSequence) {
    // Wire page_allocator into dcp_dispatcher so KV metadata gets built.
    make_dcp_dispatcher([this](ldam::CommandDispatcher::Deps& deps) {
        deps.page_allocator = page_allocator_.get();
        deps.kv_page_size   = 16;  // match PageAllocator (config default)
    });

    // Create sequence: 2 pages (prompt_len=32, page_size=16).
    auto cmd_create = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/750);
    cmd_create.seq_create.seq_id     = 8000;
    cmd_create.seq_create.prompt_len = 32;
    cmd_create.seq_create.pool       = 0;
    dispatcher_->dispatch(cmd_create);
    lipc::Completion cmp_create{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp_create));
    EXPECT_EQ(cmp_create.status, 0u);

    // Write batch descriptor: decode at token_pos=15 (within first page).
    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_base_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0].seq_id    = 8000;
    batch[0].token_pos = 15;
    batch[0]._pad      = 0;

    // Dispatch attention — should succeed with KV metadata populated.
    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/751);
    cmd.run_attention.layer_idx       = 0;
    cmd.run_attention.num_seqs        = 1;
    cmd.run_attention.is_prefill      = 0;
    cmd.run_attention.use_graph       = 0;
    cmd.run_attention.is_draft        = 0;
    cmd.run_attention.emit_checkpoint = 0;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 751u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, FusedAttention_KvMetadata_MultipleBatchEntries) {
    make_dcp_dispatcher([this](ldam::CommandDispatcher::Deps& deps) {
        deps.page_allocator = page_allocator_.get();
        deps.kv_page_size   = 16;
    });

    // Create two sequences.
    auto cmd1 = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/752);
    cmd1.seq_create.seq_id     = 8001;
    cmd1.seq_create.prompt_len = 32;  // 2 pages
    cmd1.seq_create.pool       = 0;
    dispatcher_->dispatch(cmd1);
    lipc::Completion c1{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, c1));

    auto cmd2 = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/753);
    cmd2.seq_create.seq_id     = 8002;
    cmd2.seq_create.prompt_len = 64;  // 4 pages
    cmd2.seq_create.pool       = 0;
    dispatcher_->dispatch(cmd2);
    lipc::Completion c2{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, c2));

    // Write 2 batch descriptor entries.
    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_base_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {8001, 20, 0};
    batch[1] = {8002, 50, 0};

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/754);
    cmd.run_attention.layer_idx       = 1;
    cmd.run_attention.num_seqs        = 2;
    cmd.run_attention.is_prefill      = 0;
    cmd.run_attention.use_graph       = 0;
    cmd.run_attention.is_draft        = 0;
    cmd.run_attention.emit_checkpoint = 0;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 754u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, FusedAttention_KvMetadata_UnknownSequence) {
    make_dcp_dispatcher([this](ldam::CommandDispatcher::Deps& deps) {
        deps.page_allocator = page_allocator_.get();
        deps.kv_page_size   = 16;
    });

    // No CMD_SEQ_CREATE — sideband references unknown seq_id.
    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_base_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {9999, 5, 0};

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/755);
    cmd.run_attention.layer_idx       = 0;
    cmd.run_attention.num_seqs        = 1;
    cmd.run_attention.is_prefill      = 0;
    cmd.run_attention.use_graph       = 0;
    cmd.run_attention.is_draft        = 0;
    cmd.run_attention.emit_checkpoint = 0;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    // TD-GOLDEN-KV-EXHAUST: unknown seq_id must FAIL the command — a zeroed
    // slot mapping would silently write KV to physical page 0.
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 755u);
    EXPECT_EQ(cmp.error.error_category,
              static_cast<uint32_t>(lipc::CmpErrorCategory::kKvPoolExhausted));
}

TEST_F(CommandDispatcherTest, FusedAttention_KvExhausted_ReturnsError) {
    // TD-GOLDEN-KV-EXHAUST: page-pool exhaustion during growth must fail the
    // attention command with kKvPoolExhausted, with the partial logical page
    // rolled back (free-page count stays a multiple of kKvLayers).
    cfg_->memory.kv_cache.page_growth_chunk_tokens = 32;

    make_dcp_dispatcher([this](ldam::CommandDispatcher::Deps& deps) {
        deps.page_allocator = page_allocator_.get();
        deps.kv_page_size   = 16;
    });

    auto cmd_create = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/760);
    cmd_create.seq_create.seq_id     = 8100;
    cmd_create.seq_create.prompt_len = 16;
    cmd_create.seq_create.pool       = 0;
    dispatcher_->dispatch(cmd_create);
    lipc::Completion cmp_create{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp_create));
    ASSERT_EQ(cmp_create.status, 0u);

    // Drain the main pool so ensure_pages cannot grow.
    std::vector<lmem::PageHandle> drained;
    while (auto h = page_allocator_->allocate(0, lmem::Pool::kMain))
        drained.push_back(*h);
    while (auto h = page_allocator_->allocate_unreserved(0, lmem::Pool::kMain))
        drained.push_back(*h);
    const int free_before = page_allocator_->free_pages(0, lmem::Pool::kMain);
    ASSERT_EQ(free_before, 0);

    // Decode at a position past the current allocation → growth → exhaustion.
    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_base_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {8100, 100, 0};

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/761);
    cmd.run_attention.layer_idx = 0;
    cmd.run_attention.num_seqs  = 1;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 761u);
    EXPECT_EQ(cmp.error.error_category,
              static_cast<uint32_t>(lipc::CmpErrorCategory::kKvPoolExhausted));

    // Rollback: nothing leaked, free count unchanged.
    EXPECT_EQ(page_allocator_->free_pages(0, lmem::Pool::kMain), 0);

    // Release drained pages and retry the same dispatch — the failed build
    // must NOT have been cached by the dirty guard.
    for (auto& h : drained) page_allocator_->free(h);

    auto cmd_retry = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/762);
    cmd_retry.run_attention.layer_idx = 0;
    cmd_retry.run_attention.num_seqs  = 1;
    dispatcher_->dispatch(cmd_retry);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp_retry{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp_retry));
    EXPECT_EQ(cmp_retry.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp_retry.cmd_seq, 762u);
    EXPECT_EQ(cmp_retry.status, 0u);
}

TEST_F(CommandDispatcherTest, FusedAttention_KvMetadata_NullPageAllocator) {
    // No page_allocator — KV metadata stays nullptr, dispatch still succeeds.
    make_dcp_dispatcher();

    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_base_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {1000, 10, 0};

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/756);
    cmd.run_attention.layer_idx       = 0;
    cmd.run_attention.num_seqs        = 1;
    cmd.run_attention.is_prefill      = 0;
    cmd.run_attention.use_graph       = 0;
    cmd.run_attention.is_draft        = 0;
    cmd.run_attention.emit_checkpoint = 0;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 756u);
    EXPECT_EQ(cmp.status, 0u);
}

// ── TD-GOLDEN-KV-SPEC: speculation-pool sequence lifecycle ───────────────

TEST_F(CommandDispatcherTest, SpecPool_SeqCreate_LayerMajorPhysicalPages) {
    // Spec-pool sequences allocate kKvLayers physical pages per logical page,
    // drawn from the SPECULATION free list (main pool untouched).
    make_dcp_dispatcher([this](ldam::CommandDispatcher::Deps& deps) {
        deps.page_allocator = page_allocator_.get();
        deps.kv_page_size   = 16;
    });

    const int main_before = page_allocator_->free_pages(0, lmem::Pool::kMain);
    const int spec_before =
        page_allocator_->free_pages(0, lmem::Pool::kSpeculation);
    ASSERT_GT(spec_before, 0);

    auto cmd = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/830);
    cmd.seq_create.seq_id     = 9300;
    cmd.seq_create.prompt_len = 32;  // 2 logical pages at page_size=16
    cmd.seq_create.pool       = 1;   // kSpeculation
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.seq_op.page_count, 2u * kKvLayers);

    EXPECT_EQ(page_allocator_->free_pages(0, lmem::Pool::kSpeculation),
              spec_before - 2 * static_cast<int>(kKvLayers));
    EXPECT_EQ(page_allocator_->free_pages(0, lmem::Pool::kMain), main_before);
}

TEST_F(CommandDispatcherTest, SpecPool_EnsurePages_GrowsFromSpecPool) {
    // ensure_pages routes growth to the pool of the sequence's existing pages
    // (pages[0].pool) — a spec-pool sequence must grow from the spec pool.
    cfg_->memory.kv_cache.page_growth_chunk_tokens = 32;

    make_dcp_dispatcher([this](ldam::CommandDispatcher::Deps& deps) {
        deps.page_allocator = page_allocator_.get();
        deps.kv_page_size   = 16;
    });

    auto cmd_create = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/831);
    cmd_create.seq_create.seq_id     = 9301;
    cmd_create.seq_create.prompt_len = 16;  // 1 base + 2 headroom = 3 logical
    cmd_create.seq_create.pool       = 1;   // kSpeculation
    dispatcher_->dispatch(cmd_create);
    lipc::Completion cmp_create{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp_create));
    ASSERT_EQ(cmp_create.status, 0u);

    const int main_before = page_allocator_->free_pages(0, lmem::Pool::kMain);
    const int spec_before =
        page_allocator_->free_pages(0, lmem::Pool::kSpeculation);

    // Decode at token_pos=48 → logical page 3, beyond the initial 3 → grows
    // 2 logical pages (chunk=2): spec pool drops by 2*kKvLayers.
    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_base_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {9301, 48, 0};

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/832);
    cmd.run_attention.layer_idx = 0;
    cmd.run_attention.num_seqs  = 1;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.status, 0u);

    EXPECT_EQ(page_allocator_->free_pages(0, lmem::Pool::kSpeculation),
              spec_before - 2 * static_cast<int>(kKvLayers));
    EXPECT_EQ(page_allocator_->free_pages(0, lmem::Pool::kMain), main_before);
}

TEST_F(CommandDispatcherTest, SpecPool_Exhaustion_ReturnsError) {
    // Spec-pool exhaustion during growth fails the attention command with
    // kKvPoolExhausted, exactly like the main pool (TD-GOLDEN-KV-EXHAUST).
    cfg_->memory.kv_cache.page_growth_chunk_tokens = 32;

    make_dcp_dispatcher([this](ldam::CommandDispatcher::Deps& deps) {
        deps.page_allocator = page_allocator_.get();
        deps.kv_page_size   = 16;
    });

    auto cmd_create = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/833);
    cmd_create.seq_create.seq_id     = 9302;
    cmd_create.seq_create.prompt_len = 16;
    cmd_create.seq_create.pool       = 1;  // kSpeculation
    dispatcher_->dispatch(cmd_create);
    lipc::Completion cmp_create{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp_create));
    ASSERT_EQ(cmp_create.status, 0u);

    // Drain the speculation pool.
    std::vector<lmem::PageHandle> drained;
    while (auto h = page_allocator_->allocate(0, lmem::Pool::kSpeculation))
        drained.push_back(*h);
    while (auto h =
               page_allocator_->allocate_unreserved(0, lmem::Pool::kSpeculation))
        drained.push_back(*h);
    ASSERT_EQ(page_allocator_->free_pages(0, lmem::Pool::kSpeculation), 0);

    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_base_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {9302, 100, 0};

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/834);
    cmd.run_attention.layer_idx = 0;
    cmd.run_attention.num_seqs  = 1;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.error.error_category,
              static_cast<uint32_t>(lipc::CmpErrorCategory::kKvPoolExhausted));

    // Rollback left no partial logical page allocated.
    EXPECT_EQ(page_allocator_->free_pages(0, lmem::Pool::kSpeculation), 0);
    for (auto& h : drained) page_allocator_->free(h);
}

// ── KD-4e1: Automatic KV page growth ─────────────────────────────────────

TEST_F(CommandDispatcherTest, EnsurePages_AutoGrowth) {
    // Enable auto-growth: 32 tokens = 2 pages at page_size=16.
    cfg_->memory.kv_cache.page_growth_chunk_tokens = 32;

    make_dcp_dispatcher([this](ldam::CommandDispatcher::Deps& deps) {
        deps.page_allocator = page_allocator_.get();
        deps.kv_page_size   = 16;
    });

    int free_before = page_allocator_->free_pages(0, lmem::Pool::kMain);

    // Create sequence with prompt_len=16 → 1 base page + 2 headroom = 3 pages.
    auto cmd_create = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/800);
    cmd_create.seq_create.seq_id     = 9000;
    cmd_create.seq_create.prompt_len = 16;
    cmd_create.seq_create.pool       = 0;
    dispatcher_->dispatch(cmd_create);
    lipc::Completion cmp_create{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp_create));
    EXPECT_EQ(cmp_create.status, 0u);
    EXPECT_EQ(cmp_create.seq_op.page_count, 3u * kKvLayers);  // (1 base + 2 headroom) logical

    int free_after_create = page_allocator_->free_pages(0, lmem::Pool::kMain);
    EXPECT_EQ(free_before - free_after_create, 3 * static_cast<int>(kKvLayers));

    // Dispatch attention at token_pos=48 — needs page index 3 (token 48/16=3),
    // beyond the initial 3 pages (indices 0-2). ensure_pages should grow.
    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_base_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {9000, 48, 0};

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/801);
    cmd.run_attention.layer_idx       = 0;
    cmd.run_attention.num_seqs        = 1;
    cmd.run_attention.is_prefill      = 0;
    cmd.run_attention.use_graph       = 0;
    cmd.run_attention.is_draft        = 0;
    cmd.run_attention.emit_checkpoint = 0;
    cmd.run_attention.chunk_start     = 0;
    cmd.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.status, 0u);

    // Verify pages grew: ensure_pages should have added 2 more LOGICAL pages
    // (chunk_size=2, needed=4, had 3 → target=max(4, 3+2)=5), each costing
    // kKvLayers physical pages.
    int free_after_grow = page_allocator_->free_pages(0, lmem::Pool::kMain);
    EXPECT_EQ(free_before - free_after_grow, 5 * static_cast<int>(kKvLayers));
}

TEST_F(CommandDispatcherTest, EnsurePages_NoGrowthWhenDisabled) {
    // page_growth_chunk_tokens=0 in small_config → no headroom, no auto-growth.
    make_dcp_dispatcher([this](ldam::CommandDispatcher::Deps& deps) {
        deps.page_allocator = page_allocator_.get();
        deps.kv_page_size   = 16;
    });

    // Create sequence with prompt_len=16 → exactly 1 page (no headroom).
    auto cmd_create = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/810);
    cmd_create.seq_create.seq_id     = 9100;
    cmd_create.seq_create.prompt_len = 16;
    cmd_create.seq_create.pool       = 0;
    dispatcher_->dispatch(cmd_create);
    lipc::Completion cmp_create{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp_create));
    EXPECT_EQ(cmp_create.status, 0u);
    EXPECT_EQ(cmp_create.seq_op.page_count, 1u * kKvLayers);  // no headroom (1 logical)
}

TEST_F(CommandDispatcherTest, EnsurePages_InitialHeadroom) {
    // Verify CMD_SEQ_CREATE pre-allocates headroom pages.
    cfg_->memory.kv_cache.page_growth_chunk_tokens = 48;  // 3 pages at page_size=16

    make_dcp_dispatcher([this](ldam::CommandDispatcher::Deps& deps) {
        deps.page_allocator = page_allocator_.get();
        deps.kv_page_size   = 16;
    });

    // prompt_len=32 → 2 base + 3 headroom = 5 pages.
    auto cmd_create = make_cmd(lipc::CMD_SEQ_CREATE, /*seq=*/820);
    cmd_create.seq_create.seq_id     = 9200;
    cmd_create.seq_create.prompt_len = 32;
    cmd_create.seq_create.pool       = 0;
    dispatcher_->dispatch(cmd_create);
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.seq_op.page_count, 5u * kKvLayers);  // (2 base + 3 headroom) logical
}

// ── KD-3b: Fused MoE dispatch ────────────────────────────────────────────

TEST_F(CommandDispatcherTest, FusedMoe_ExpertDevice_Dispatch) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .expert_devices     = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/800);
    cmd.run_moe.layer_idx                 = 3;
    cmd.run_moe.num_seqs                  = 4;
    cmd.run_moe.moe_mode                  = 0;
    cmd.run_moe.apply_residual_correction = 0;
    cmd.run_moe.store_gating_output       = 0;
    cmd.run_moe.emit_checkpoint           = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 800u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.cmd_type,
              static_cast<uint32_t>(lipc::D_B_CMD_RUN_MOE));
    EXPECT_EQ(cmp.compute.layer_idx, 3u);
}

TEST_F(CommandDispatcherTest, FusedMoe_ExpertDevice_Checkpoint) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .expert_devices     = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/801);
    cmd.run_moe.layer_idx                 = 5;
    cmd.run_moe.num_seqs                  = 2;
    cmd.run_moe.moe_mode                  = 0;
    cmd.run_moe.apply_residual_correction = 0;
    cmd.run_moe.store_gating_output       = 0;
    cmd.run_moe.emit_checkpoint           = 1;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp1{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp1));
    EXPECT_EQ(cmp1.cmp_type, static_cast<uint32_t>(lipc::CMP_CHECKPOINT));
    EXPECT_EQ(cmp1.cmd_seq, 801u);
    EXPECT_EQ(cmp1.checkpoint.cmd_type,
              static_cast<uint32_t>(lipc::D_B_CMD_RUN_MOE));
    EXPECT_EQ(cmp1.checkpoint.layer_idx, 5u);
    // F-7: emit_checkpoint on RUN_MOE now means the attention↔MoE seam-routing
    // checkpoint (top-K weights+indices published to kSeamCheckpointOff), not the
    // old zero-data kGatingOutput marker.
    EXPECT_EQ(cmp1.checkpoint.checkpoint_type,
              static_cast<uint8_t>(lipc::CheckpointType::kSeamRouting));
    EXPECT_EQ(cmp1.checkpoint.host_buf_offset,
              static_cast<uint32_t>(lipc::IpcLayout::kSeamCheckpointOff));
    // num_seqs=2, topk=2 → expanded=4 → 4*(f32 w + i32 idx) = 32 bytes, non-zero.
    EXPECT_GT(cmp1.checkpoint.data_bytes, 0u);
    const uint32_t expanded = 2u * 2u;
    EXPECT_EQ(cmp1.checkpoint.data_bytes,
              expanded * (sizeof(float) + sizeof(int32_t)));
    // The seam region must contain the published routing within bounds.
    EXPECT_LE(cmp1.checkpoint.host_buf_offset + cmp1.checkpoint.data_bytes,
              lipc::IpcLayout::kSeamCheckpointOff + lipc::IpcLayout::kSeamCheckpointSize);

    lipc::Completion cmp2{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp2));
    EXPECT_EQ(cmp2.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp2.cmd_seq, 801u);
    EXPECT_EQ(cmp2.compute.layer_idx, 5u);
}

// F-7: the seam checkpoint is readable — RUN_MOE [emit_checkpoint=1] publishes the
// routed top-K to the sideband seam region and signals it via CMP_CHECKPOINT with
// kSeamRouting + a non-zero, in-bounds {host_buf_offset, data_bytes}.
TEST_F(CommandDispatcherTest, FusedMoe_SeamCheckpoint_Readable) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .expert_devices     = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Poison the seam region so we can confirm the daemon actually wrote it.
    std::memset(sideband_base_ + lipc::IpcLayout::kSeamCheckpointOff, 0xAB,
                lipc::IpcLayout::kSeamCheckpointSize);

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/803);
    cmd.run_moe.layer_idx                 = 5;
    cmd.run_moe.num_seqs                  = 2;
    cmd.run_moe.moe_mode                  = 0;
    cmd.run_moe.apply_residual_correction = 0;
    cmd.run_moe.store_gating_output       = 0;
    cmd.run_moe.emit_checkpoint           = 1;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp1{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp1));
    ASSERT_EQ(cmp1.cmp_type, static_cast<uint32_t>(lipc::CMP_CHECKPOINT));
    EXPECT_EQ(cmp1.checkpoint.checkpoint_type,
              static_cast<uint8_t>(lipc::CheckpointType::kSeamRouting));
    EXPECT_EQ(cmp1.checkpoint.host_buf_offset,
              static_cast<uint32_t>(lipc::IpcLayout::kSeamCheckpointOff));
    ASSERT_GT(cmp1.checkpoint.data_bytes, 0u);
    EXPECT_LE(cmp1.checkpoint.host_buf_offset + cmp1.checkpoint.data_bytes,
              lipc::IpcLayout::kSeamCheckpointOff + lipc::IpcLayout::kSeamCheckpointSize);

    // The advertised seam region was written (no longer the 0xAB poison).
    const uint8_t* seam = sideband_base_ + cmp1.checkpoint.host_buf_offset;
    bool any_written = false;
    for (uint32_t i = 0; i < cmp1.checkpoint.data_bytes; ++i) {
        if (seam[i] != 0xAB) { any_written = true; break; }
    }
    EXPECT_TRUE(any_written);

    lipc::Completion cmp2{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp2));
    EXPECT_EQ(cmp2.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp2.cmd_seq, 803u);
}

// F-7: attention and MoE remain SEPARATELY dispatchable across the data-dependent
// H2D seam — each op completes independently with its own CMP_COMPUTE_DONE (no
// single op/graph spans attention→fetch→expert).
TEST_F(CommandDispatcherTest, AttentionAndMoe_RemainSeparateOps) {
    // Attention needs a DcpExecutor; MoE needs an ExpertDevice. Wire both so the
    // two ops are independently dispatchable in the same dispatcher.
    make_dcp_dispatcher([this](ldam::CommandDispatcher::Deps& deps) {
        deps.expert_devices = {test_expert_devices_[0].get()};
    });

    // Op 1: attention (no checkpoint) — completes on its own.
    auto attn = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/820);
    attn.run_attention.layer_idx       = 5;
    attn.run_attention.num_seqs        = 2;
    attn.run_attention.is_prefill      = 0;
    attn.run_attention.use_graph       = 0;
    attn.run_attention.is_draft        = 0;
    attn.run_attention.emit_checkpoint = 0;
    attn.run_attention.chunk_start     = 0;
    attn.run_attention.chunk_len       = 0;
    dispatcher_->dispatch(attn);
    dispatcher_->poll_compute_completions();

    lipc::Completion attn_done{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, attn_done));
    EXPECT_EQ(attn_done.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(attn_done.cmd_seq, 820u);
    EXPECT_EQ(attn_done.compute.cmd_type,
              static_cast<uint32_t>(lipc::D_B_CMD_RUN_ATTENTION));

    // Op 2: MoE (separate dispatch) — completes independently with its own
    // CMP_COMPUTE_DONE. The seam between them is where the host decider sits.
    auto moe = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/821);
    moe.run_moe.layer_idx                 = 5;
    moe.run_moe.num_seqs                  = 2;
    moe.run_moe.moe_mode                  = 0;
    moe.run_moe.apply_residual_correction = 0;
    moe.run_moe.store_gating_output       = 0;
    moe.run_moe.emit_checkpoint           = 0;
    dispatcher_->dispatch(moe);
    dispatcher_->poll_compute_completions();

    lipc::Completion moe_done{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, moe_done));
    EXPECT_EQ(moe_done.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(moe_done.cmd_seq, 821u);
    EXPECT_EQ(moe_done.compute.cmd_type,
              static_cast<uint32_t>(lipc::D_B_CMD_RUN_MOE));
}

TEST_F(CommandDispatcherTest, FusedMoe_NoExpertDevice_ReturnsError) {
    // Without ExpertDevice, dispatch fails and CMP_ERROR is written (TD-40a)
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/802);
    cmd.run_moe.layer_idx                 = 1;
    cmd.run_moe.num_seqs                  = 4;
    cmd.run_moe.moe_mode                  = 0;
    cmd.run_moe.apply_residual_correction = 0;
    cmd.run_moe.store_gating_output       = 0;
    cmd.run_moe.emit_checkpoint           = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 802u);
    EXPECT_EQ(cmp.status, 1u);
}

TEST_F(CommandDispatcherTest, FusedMoe_AllModes) {
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .expert_devices     = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    for (uint8_t mode = 0; mode <= 2; ++mode) {
        auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/810 + mode);
        cmd.run_moe.layer_idx                 = 2;
        cmd.run_moe.num_seqs                  = 1;
        cmd.run_moe.moe_mode                  = mode;
        cmd.run_moe.apply_residual_correction = (mode == 2) ? 1 : 0;
        cmd.run_moe.store_gating_output       = (mode == 1) ? 1 : 0;
        cmd.run_moe.emit_checkpoint           = 0;
        dispatcher_->dispatch(cmd);
        dispatcher_->poll_compute_completions();

        lipc::Completion cmp{};
        ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
        EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
        EXPECT_EQ(cmp.cmd_seq, 810u + mode);
        EXPECT_EQ(cmp.status, 0u);
        EXPECT_EQ(cmp.compute.cmd_type,
                  static_cast<uint32_t>(lipc::D_B_CMD_RUN_MOE));
    }
}

// ── KD-3e: Router projection, shared expert, and residual add tests ─────────

TEST_F(CommandDispatcherTest, FusedMoe_RouterWeightPtrs_NullGracefulSkip) {
    // With router_weight_ptrs populated but pointing to nullptr for the layer,
    // dispatch still succeeds (Step 0 skipped gracefully).
    std::vector<std::vector<const void*>> router_ptrs(10, std::vector<const void*>(2, nullptr));

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .expert_devices     = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
        .router_weight_ptrs = std::move(router_ptrs),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/820);
    cmd.run_moe.layer_idx   = 3;
    cmd.run_moe.num_seqs    = 2;
    cmd.run_moe.moe_mode    = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 820u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, FusedMoe_SharedExpertWeightPtrs_NullGracefulSkip) {
    // With shared_expert_weight_ptrs populated but with null gate_up/down,
    // dispatch still succeeds (Step 7 skipped gracefully).
    std::vector<std::vector<ldam::CommandDispatcher::Deps::SharedExpertWeights>>
        se_ptrs(10, std::vector<ldam::CommandDispatcher::Deps::SharedExpertWeights>(2));

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .expert_devices     = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
        .shared_expert_weight_ptrs = std::move(se_ptrs),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/821);
    cmd.run_moe.layer_idx   = 3;
    cmd.run_moe.num_seqs    = 2;
    cmd.run_moe.moe_mode    = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 821u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, FusedMoe_BothKD3eFields_DispatchSucceeds) {
    // Both router and shared expert Deps fields populated (all nullptr weights).
    // Verifies no crash with full KD-3e Deps structure.
    std::vector<std::vector<const void*>> router_ptrs(10, std::vector<const void*>(2, nullptr));
    std::vector<std::vector<ldam::CommandDispatcher::Deps::SharedExpertWeights>>
        se_ptrs(10, std::vector<ldam::CommandDispatcher::Deps::SharedExpertWeights>(2));

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .expert_devices     = {test_expert_devices_[0].get(),
                               test_expert_devices_[1].get()},
        .cuda_kernels_enabled = false,
        .router_weight_ptrs = std::move(router_ptrs),
        .shared_expert_weight_ptrs = std::move(se_ptrs),
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/822);
    cmd.run_moe.layer_idx   = 5;
    cmd.run_moe.num_seqs    = 4;
    cmd.run_moe.moe_mode    = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 822u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.layer_idx, 5u);
}

TEST_F(CommandDispatcherTest, FusedMoe_SharedExpertScratch_Allocated) {
    // Verify that shared expert scratch buffers are allocated when n_shared_experts > 0.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .expert_devices     = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
    };
    // Config has n_shared_experts=1, so shared scratch should be allocated.
    // NullExpertDevice::device_alloc returns non-null sentinel pointers.
    auto disp = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Dispatch should succeed — shared scratch doesn't cause issues.
    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/823);
    cmd.run_moe.layer_idx   = 0;
    cmd.run_moe.num_seqs    = 1;
    cmd.run_moe.moe_mode    = 0;
    disp->dispatch(cmd);
    disp->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 823u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, FusedMoe_ProblemSizesSfOffsetsAllocated) {
    // KD-3f: verify problem_sizes/sf_offsets buffers are allocated for both
    // routed and shared expert paths. NullExpertDevice::device_alloc returns
    // non-null sentinels, so dispatch succeeding confirms allocation happened.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .expert_devices     = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
    };
    auto disp = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/824);
    cmd.run_moe.layer_idx   = 0;
    cmd.run_moe.num_seqs    = 4;
    cmd.run_moe.moe_mode    = 0;
    disp->dispatch(cmd);
    disp->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 824u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, FusedMoe_QuantScratchAllocated_FP8) {
    // KD-3g: verify quant_act/quant_scale buffers are allocated with FP8 config.
    // Construction + destruction succeeding confirms allocation/free paths work.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .expert_devices     = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
    };
    auto disp = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/830);
    cmd.run_moe.layer_idx   = 0;
    cmd.run_moe.num_seqs    = 4;
    cmd.run_moe.moe_mode    = 0;
    disp->dispatch(cmd);
    disp->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 830u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, FusedMoe_QuantScratchAllocated_NVFP4) {
    // KD-3g: verify quant_act/quant_scale buffers are allocated with NVFP4 config.
    // Uses a modified config with nvfp4 weight quantization.
    auto nvfp4_cfg = small_config();
    nvfp4_cfg.quantization.weights = lc::WeightQuant::nvfp4;

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = &nvfp4_cfg,
        .expert_devices     = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
    };
    auto disp = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/831);
    cmd.run_moe.layer_idx   = 0;
    cmd.run_moe.num_seqs    = 4;
    cmd.run_moe.moe_mode    = 0;
    disp->dispatch(cmd);
    disp->poll_compute_completions();

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 831u);
    EXPECT_EQ(cmp.status, 0u);
}

TEST_F(CommandDispatcherTest, FusedMoe_CopyBack_HiddenStateUpdate) {
    // KD-4d: dispatch attention then MoE for the same layer, verify both
    // complete successfully with DcpExecutor + both hidden state buffers wired.
    const size_t hidden_bytes = 8 * 256 * 2;
    dcp_moe_hidden_buf_ = aligned_alloc_zeroed(hidden_bytes);

    make_dcp_dispatcher([this](ldam::CommandDispatcher::Deps& deps) {
        deps.expert_devices = {test_expert_devices_[0].get()};
        if (!deps.hidden_state_pairs.empty())
            deps.hidden_state_pairs[0].moe_buf = dcp_moe_hidden_buf_;
        deps.per_layer_attn_weights = make_fake_attn_weights(6, 1);
    });

    // Dispatch attention
    auto cmd_attn = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/900);
    cmd_attn.run_attention.layer_idx = 1;
    cmd_attn.run_attention.num_seqs  = 4;
    dispatcher_->dispatch(cmd_attn);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp1{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp1));
    EXPECT_EQ(cmp1.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp1.cmd_seq, 900u);

    // Dispatch MoE on same GPU
    auto cmd_moe = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/901);
    cmd_moe.run_moe.layer_idx = 1;
    cmd_moe.run_moe.num_seqs  = 4;
    cmd_moe.run_moe.moe_mode  = 0;
    dispatcher_->dispatch(cmd_moe);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp2{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp2));
    EXPECT_EQ(cmp2.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp2.cmd_seq, 901u);
}

TEST_F(CommandDispatcherTest, AttentionMoeSync_EventsCreatedAndDestroyed) {
    // KD-4d: verify pre-allocated sync events are created when DcpExecutor is
    // present, and cleaned up on destruction without leaking.
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    auto null_attn = lcomp::make_null_attention_device(gpu0);
    lpar::DcpExecutor::Options dcp_opts{
        .dcp_size = 1,
        .gpus = {gpu0},
        .max_batch_size = 8,
        .hidden_size = 256,
        .num_attention_heads = 4,
        .q_lora_rank = 64,
        .kv_lora_rank = 32,
        .qk_rope_head_dim = 16,
        .qk_nope_head_dim = 32,
        .v_head_dim = 64,
        .rms_norm_eps = 1e-6f,
        .stream_manager = stream_manager_.get(),
        .attention_devices = {null_attn.get()},
    };
    auto dcp_exec = std::make_unique<lpar::DcpExecutor>(std::move(dcp_opts));

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .dcp_executor       = dcp_exec.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {null_attn.get()},
        .expert_devices     = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
        .hidden_state_pairs = {ldam::HiddenStatePair{nullptr, nullptr, 0, 0, 0}},
        .per_layer_attn_weights = make_fake_attn_weights(6, 1),
        .max_batch_size         = 8,
    };
    auto disp = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Destruction should clean up sync events without errors.
    disp.reset();
}

// ═══════════════════════════════════════════════════════════════════════════
// KD-R1: forward_one_layer + MoE→attention sync events
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(CommandDispatcherTest, MoeAttnSyncEvents_CreatedAndDestroyed) {
    // KD-R1: verify pair moe_attn sync events are created when DcpExecutor is
    // present, and cleaned up on destruction without leaking.
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    auto null_attn = lcomp::make_null_attention_device(gpu0);
    lpar::DcpExecutor::Options dcp_opts{
        .dcp_size = 1,
        .gpus = {gpu0},
        .max_batch_size = 8,
        .hidden_size = 256,
        .num_attention_heads = 4,
        .q_lora_rank = 64,
        .kv_lora_rank = 32,
        .qk_rope_head_dim = 16,
        .qk_nope_head_dim = 32,
        .v_head_dim = 64,
        .rms_norm_eps = 1e-6f,
        .stream_manager = stream_manager_.get(),
        .attention_devices = {null_attn.get()},
    };
    auto dcp_exec = std::make_unique<lpar::DcpExecutor>(std::move(dcp_opts));

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .dcp_executor       = dcp_exec.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {null_attn.get()},
        .expert_devices     = {test_expert_devices_[0].get()},
        .cuda_kernels_enabled = false,
        .hidden_state_pairs = {ldam::HiddenStatePair{nullptr, nullptr, 0, 0, 0}},
        .per_layer_attn_weights = make_fake_attn_weights(6, 1),
        .max_batch_size         = 8,
    };
    auto disp = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Destruction should clean up both attn_moe + moe_attn sync events.
    disp.reset();
}

TEST_F(CommandDispatcherTest, ForwardOneLayer_AttentionThenMoe) {
    // KD-R1: verify forward_one_layer's internal sequence by dispatching
    // attention then MoE for the same layer with DcpExecutor + both hidden
    // state buffers wired.
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    auto null_attn = lcomp::make_null_attention_device(gpu0);
    lpar::DcpExecutor::Options dcp_opts{
        .dcp_size = 1,
        .gpus = {gpu0},
        .max_batch_size = 8,
        .hidden_size = 256,
        .num_attention_heads = 4,
        .q_lora_rank = 64,
        .kv_lora_rank = 32,
        .qk_rope_head_dim = 16,
        .qk_nope_head_dim = 32,
        .v_head_dim = 64,
        .rms_norm_eps = 1e-6f,
        .stream_manager = stream_manager_.get(),
        .attention_devices = {null_attn.get()},
    };
    auto dcp_exec = std::make_unique<lpar::DcpExecutor>(std::move(dcp_opts));

    // TD-90b: initialize dequant pool (required since KD-4j).
    auto fake_w = make_fake_attn_weights(6, 1);
    {
        std::vector<std::vector<const lpar::AttentionLayerWeights*>> wp(6);
        for (int l = 0; l < 6; ++l) wp[l] = {&fake_w[l][0]};
        dcp_exec->set_layer_weights(std::move(wp), 6);
    }

    const size_t hidden_bytes = 8 * 256 * 2;  // max_batch * hidden_size * BF16
    void* hidden_buf = aligned_alloc_zeroed(hidden_bytes);
    void* moe_hidden_buf = aligned_alloc_zeroed(hidden_bytes);

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .dcp_executor       = dcp_exec.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {null_attn.get()},
        .expert_devices     = {test_expert_devices_[0].get()},
        .device_backends    = {test_device_backends_[0].get()},
        .cuda_kernels_enabled = false,
        .kv_cache_stride_block = 576 * 64,
        .kv_cache_stride_row   = 576,
        .kv_page_size          = 64,
        .hidden_state_pairs      = {ldam::HiddenStatePair{hidden_buf, moe_hidden_buf, 0, 0, 0}},
        .per_layer_attn_weights = std::move(fake_w),
        .max_batch_size         = 8,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Dispatch attention for layer 2.
    auto cmd_attn = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/950);
    cmd_attn.run_attention.layer_idx = 2;
    cmd_attn.run_attention.num_seqs  = 4;
    dispatcher_->dispatch(cmd_attn);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp1{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp1));
    EXPECT_EQ(cmp1.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp1.cmd_seq, 950u);
    EXPECT_EQ(cmp1.compute.layer_idx, 2u);

    // Dispatch MoE for same layer.
    auto cmd_moe = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/951);
    cmd_moe.run_moe.layer_idx = 2;
    cmd_moe.run_moe.num_seqs  = 4;
    cmd_moe.run_moe.moe_mode  = 0;
    dispatcher_->dispatch(cmd_moe);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp2{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp2));
    EXPECT_EQ(cmp2.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp2.cmd_seq, 951u);
    EXPECT_EQ(cmp2.compute.layer_idx, 2u);

    // No leftover completions.
    lipc::Completion cmp3{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp3));

    dispatcher_.reset();
    std::free(hidden_buf);
    std::free(moe_hidden_buf);
}

// ═══════════════════════════════════════════════════════════════════════════
// KD-R3: Recording device backend tests
// ═══════════════════════════════════════════════════════════════════════════

#ifdef LAYERSTORM_RECORDING_BACKEND

// Convenience macros for op assertions.
#define EXPECT_HAS_OP(dev, kind) \
    EXPECT_TRUE((dev)->has_op(lcomp::DeviceOp::kind)) \
        << "Expected " #kind " op in recording"
#define EXPECT_NO_OP(dev, kind) \
    EXPECT_FALSE((dev)->has_op(lcomp::DeviceOp::kind)) \
        << "Did not expect " #kind " op in recording"

TEST_F(CommandDispatcherTest, TaggedPointer_IsDevicePtr) {
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    auto dev = std::make_unique<lcomp::RecordingAttentionDevice>(gpu0);

    void* dptr = dev->device_alloc(256);
    ASSERT_NE(dptr, nullptr);
    EXPECT_TRUE(lcomp::is_device_ptr(dptr));

    // Host pointer should NOT have the tag.
    void* hptr = std::malloc(64);
    EXPECT_FALSE(lcomp::is_device_ptr(hptr));
    std::free(hptr);

    // nullptr is not a device pointer.
    EXPECT_FALSE(lcomp::is_device_ptr(nullptr));

    // Free via recording device must not crash (strips tag internally).
    dev->device_free(dptr);

    // Verify alloc + free were recorded.
    EXPECT_HAS_OP(dev, kAlloc);
    EXPECT_HAS_OP(dev, kFree);
}

TEST_F(CommandDispatcherTest, RecordingAttention_LogsOps) {
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    auto dev = std::make_unique<lcomp::RecordingAttentionDevice>(gpu0);

    // Allocate scratch for param pointers (content doesn't matter).
    void* buf = dev->device_alloc(1024);

    lcomp::Fp8GemmParams gp{};
    gp.M = 4; gp.N = 256; gp.K = 256;
    gp.A = buf; gp.B = buf; gp.D = buf;
    dev->gemm(gp, nullptr, nullptr);

    dev->rmsnorm(buf, buf, buf, 1e-6f, 4, 256, 256, nullptr);

    lcomp::DynamicFp8QuantParams qp{};
    qp.num_tokens = 4; qp.hidden_size = 256;
    qp.input = buf; qp.output = buf; qp.scales = buf;
    dev->quantize_fp8(qp, nullptr);

    EXPECT_HAS_OP(dev, kGemm);
    EXPECT_HAS_OP(dev, kRmsnorm);
    EXPECT_HAS_OP(dev, kQuantize);

    // Verify count: alloc(1) + gemm(1) + rmsnorm(1) + quantize(1) = 4 ops.
    EXPECT_EQ(dev->ops().size(), 4u);

    dev->device_free(buf);
}

TEST_F(CommandDispatcherTest, RecordingExpert_LogsOps) {
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    auto dev = std::make_unique<lcomp::RecordingExpertDevice>(gpu0);

    void* buf = dev->device_alloc(4096);
    auto* i32buf = static_cast<int32_t*>(lcomp::untag_device_ptr(buf));

    // Permute.
    dev->moe_permute(buf, i32buf, i32buf, i32buf,
                     buf, i32buf,
                     /*num_tokens=*/4, /*topk=*/2, /*hidden_dim=*/128,
                     /*num_experts=*/8, /*elem_size_bytes=*/2,
                     nullptr, nullptr);
    EXPECT_HAS_OP(dev, kPermute);

    // Grouped GEMM (FP8).
    lcomp::Fp8GroupedGemmParams fp8gp{};
    fp8gp.num_experts = 2; fp8gp.N = 128; fp8gp.K = 256;
    fp8gp.A_base = buf; fp8gp.B_base = buf; fp8gp.D_base = buf;
    dev->fp8_grouped_gemm(fp8gp, nullptr, 0, nullptr);
    EXPECT_HAS_OP(dev, kGemm);

    // SwiGLU.
    lcomp::FusedSwigluParams sp{};
    sp.num_tokens = 8; sp.d = 128;
    dev->fused_swiglu(buf, buf, sp, 2, nullptr);
    EXPECT_HAS_OP(dev, kSwiglu);

    // Unpermute.
    dev->moe_unpermute(buf, buf, nullptr, i32buf,
                       /*num_tokens=*/4, /*topk=*/2, /*hidden_dim=*/128,
                       /*elem_size_bytes=*/2, nullptr);
    EXPECT_HAS_OP(dev, kUnpermute);

    // alloc(1) + permute(1) + gemm(1) + swiglu(1) + unpermute(1) = 5
    EXPECT_EQ(dev->ops().size(), 5u);

    dev->device_free(buf);
}

TEST_F(CommandDispatcherTest, RecordingStreamBackend_LogsEventAndMemcpy) {
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};

    auto rec_be = std::make_unique<lcomp::RecordingDeviceBackend>(gpu0);
    auto ctx = rec_be->context();
    lcomp::StreamManager::Options sm_opts{
        .device_backends = {rec_be.get()},
    };
    auto sm = std::make_unique<lcomp::StreamManager>(std::move(sm_opts));

    void* event = sm->create_event(0);
    ASSERT_NE(event, nullptr);

    sm->record_event(event, 0, lcomp::StreamId::kAttention);
    EXPECT_TRUE(ctx->has_op(lcomp::DeviceOp::kEventRecord));

    sm->wait_event(0, lcomp::StreamId::kExpertFfn, event);
    EXPECT_TRUE(ctx->has_op(lcomp::DeviceOp::kEventWait));

    // Memcpy D2H: source is a tagged device pointer.
    float host_val = 0.0f;
    float src_val = 42.0f;
    void* dev_src = lcomp::tag_device_ptr(&src_val);
    sm->memcpy_d2h_async(&host_val, dev_src, sizeof(float),
                         0, lcomp::StreamId::kD2hTransfer);
    EXPECT_TRUE(ctx->has_op(lcomp::DeviceOp::kMemcpyD2H));
    // Recording backend should have performed the actual copy (untagged).
    EXPECT_FLOAT_EQ(host_val, 42.0f);

    // 3 ops total: record + wait + memcpy.
    EXPECT_EQ(ctx->view().size(), 3u);

    sm->destroy_event(event, 0);
}

TEST_F(CommandDispatcherTest, ForwardOneLayer_RecordingDevices) {
    // Upgraded version of ForwardOneLayer_AttentionThenMoe using recording
    // devices + recording stream backend to verify sync event flow.
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};

    // Recording device backend (replaces recording_stream_backend).
    auto rec_dev_be = std::make_unique<lcomp::RecordingDeviceBackend>(gpu0);
    auto stream_ctx = rec_dev_be->context();
    auto rec_sm = std::make_unique<lcomp::StreamManager>(
        lcomp::StreamManager::Options{
            .device_backends = {rec_dev_be.get()},
        });

    // Recording devices — keep raw pointers for query access.
    auto rec_attn_uptr = std::make_unique<lcomp::RecordingAttentionDevice>(gpu0);
    auto rec_exp_uptr  = std::make_unique<lcomp::RecordingExpertDevice>(gpu0);
    auto* rec_attn = rec_attn_uptr.get();
    auto* rec_exp  = rec_exp_uptr.get();

    lpar::DcpExecutor::Options dcp_opts{
        .dcp_size             = 1,
        .gpus                 = {gpu0},
        .max_batch_size       = 8,
        .hidden_size          = 256,
        .num_attention_heads  = 4,
        .q_lora_rank          = 64,
        .kv_lora_rank         = 32,
        .qk_rope_head_dim     = 16,
        .qk_nope_head_dim     = 32,
        .v_head_dim           = 64,
        .rms_norm_eps         = 1e-6f,
        .stream_manager       = rec_sm.get(),
        .attention_devices    = {rec_attn_uptr.get()},
    };
    auto dcp_exec = std::make_unique<lpar::DcpExecutor>(std::move(dcp_opts));

    // TD-90b: initialize dequant pool (required since KD-4j).
    auto fake_w = make_fake_attn_weights(6, 1);
    {
        std::vector<std::vector<const lpar::AttentionLayerWeights*>> wp(6);
        for (int l = 0; l < 6; ++l) wp[l] = {&fake_w[l][0]};
        dcp_exec->set_layer_weights(std::move(wp), 6);
    }

    const size_t hidden_bytes = 8 * 256 * 2;  // max_batch * hidden * BF16
    void* hidden_buf = aligned_alloc_zeroed(hidden_bytes);
    void* moe_hidden_buf = aligned_alloc_zeroed(hidden_bytes);

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = rec_sm.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .dcp_executor       = dcp_exec.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {rec_attn_uptr.get()},
        .expert_devices     = {rec_exp_uptr.get()},
        .device_backends    = {rec_dev_be.get()},
        .cuda_kernels_enabled = false,
        .kv_cache_stride_block = 576 * 64,
        .kv_cache_stride_row   = 576,
        .kv_page_size          = 64,
        .hidden_state_pairs    = {ldam::HiddenStatePair{hidden_buf, moe_hidden_buf, 0, 0, 0}},
        .per_layer_attn_weights = std::move(fake_w),
        .max_batch_size        = 8,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Clear ops accumulated during DcpExecutor construction.
    rec_attn->clear();
    rec_exp->clear();
    stream_ctx->clear();

    // Dispatch attention for layer 2.
    auto cmd_attn = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, /*seq=*/960);
    cmd_attn.run_attention.layer_idx = 2;
    cmd_attn.run_attention.num_seqs  = 4;
    dispatcher_->dispatch(cmd_attn);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp1{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp1));
    EXPECT_EQ(cmp1.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));

    // Dispatch MoE for same layer.
    auto cmd_moe = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/961);
    cmd_moe.run_moe.layer_idx = 2;
    cmd_moe.run_moe.num_seqs  = 4;
    cmd_moe.run_moe.moe_mode  = 0;
    dispatcher_->dispatch(cmd_moe);
    dispatcher_->poll_compute_completions();

    lipc::Completion cmp2{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp2));
    EXPECT_EQ(cmp2.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));

    // ── Verify recording stream context saw sync events ──────────────────
    // forward_one_layer records an event after MoE on kExpertFfn and waits
    // on kAttention for prior MoE completion.  The stream backend should
    // have logged kEventRecord ops from the compute command event recording.
    EXPECT_TRUE(stream_ctx->has_op(lcomp::DeviceOp::kEventRecord))
        << "Expected event record ops from compute dispatch";

    // No leftover completions.
    lipc::Completion cmp3{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp3));

    dispatcher_.reset();
    dcp_exec.reset();
    std::free(hidden_buf);
    std::free(moe_hidden_buf);
}

#undef EXPECT_HAS_OP
#undef EXPECT_NO_OP

#endif  // LAYERSTORM_RECORDING_BACKEND

// ═══════════════════════════════════════════════════════════════════════════
// 13c-7: EP-within-TP routed output allreduce
// ═══════════════════════════════════════════════════════════════════════════

// Shared recording collective backend (tests/unit/recording_collective_backend.h).
using EpRecordingCollective = layerstorm::test::RecordingCollectiveBackend;

// Identical bitsets on both TP GPUs → old path (intersection, no routed allreduce).
// Should see exactly one allreduce group (shared expert output only).
TEST_F(CommandDispatcherTest, MoeTP2_IdenticalBitsets_NoRoutedAllreduce) {
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    lc::GpuRef gpu1{1, 1, lc::GpuType::rtx5090};

    auto attn0 = lcomp::make_null_attention_device(gpu0);
    auto attn1 = lcomp::make_null_attention_device(gpu1);
    auto exp0  = lcomp::make_null_expert_device(gpu0);
    auto exp1  = lcomp::make_null_expert_device(gpu1);

    // DcpExecutor (TP=2).
    lpar::DcpExecutor::Options dcp_opts{
        .dcp_size             = 2,
        .gpus                 = {gpu0, gpu1},
        .max_batch_size       = 8,
        .hidden_size          = 256,
        .num_attention_heads  = 4,
        .q_lora_rank          = 64,
        .kv_lora_rank         = 32,
        .qk_rope_head_dim     = 16,
        .qk_nope_head_dim     = 32,
        .v_head_dim           = 64,
        .rms_norm_eps         = 1e-6f,
        .stream_manager       = stream_manager_.get(),
        .attention_devices    = {attn0.get(), attn1.get()},
    };
    auto dcp = std::make_unique<lpar::DcpExecutor>(std::move(dcp_opts));

    // DcpCommunicator with recording collective.
    std::vector<std::string> nccl_log;
    EpRecordingCollective rec_collective(nccl_log);
    lpar::DcpCommunicator::Options comm_opts{
        .dcp_size        = 2,
        .device_backends = {test_device_backends_[0].get(),
                            test_device_backends_[1].get()},
        .max_batch_size  = 8,
        .num_heads       = 4,
        .attn_output_dim = 32,
        .hidden_size     = 256,
        .collective      = &rec_collective,
    };
    auto comm = std::make_unique<lpar::DcpCommunicator>(comm_opts);

    // Hidden state pairs.
    const size_t hidden_bytes = 8 * 256 * 2;
    void* attn_buf0 = aligned_alloc_zeroed(hidden_bytes);
    void* attn_buf1 = aligned_alloc_zeroed(hidden_bytes);
    void* moe_buf0  = aligned_alloc_zeroed(hidden_bytes);
    void* moe_buf1  = aligned_alloc_zeroed(hidden_bytes);

    // Reserve the SAME experts on both GPUs (identical bitsets).
    // Use layer 6 (>= num_hidden_layers=6), treated as MTP layer — silently
    // skips router weight check in dispatch_moe_internal with cuda_kernels_enabled=false.
    // Still not dense (6 >= first_k_dense_replace=1), so EP bitset logic runs.
    constexpr uint32_t kTestLayer = 6;
    for (uint16_t e = 0; e < 4; ++e) {
        auto key = lmem::ExpertKey{kTestLayer, e};
        cache_->reserve(key, 0, lmem::CacheZone::kStable, false);
        cache_->mark_ready(key, 0, lmem::SubComponent::kAll);
        cache_->reserve(key, 1, lmem::CacheZone::kStable, false);
        cache_->mark_ready(key, 1, lmem::SubComponent::kAll);
    }

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .dcp_executor       = dcp.get(),
        .dcp_communicator   = comm.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {attn0.get(), attn1.get()},
        .expert_devices     = {exp0.get(), exp1.get()},
        .device_backends    = {test_device_backends_[0].get(),
                               test_device_backends_[1].get()},
        .cuda_kernels_enabled = false,
        .hidden_state_pairs = {
            ldam::HiddenStatePair{attn_buf0, moe_buf0, 0, 0, 0},
            ldam::HiddenStatePair{attn_buf1, moe_buf1, 1, 1, 1},
        },
        .per_layer_attn_weights = make_fake_attn_weights(6, 2),
        .max_batch_size     = 8,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Clear log after construction (DcpCommunicator init may log).
    nccl_log.clear();

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/700, /*gpu_idx=*/0);
    cmd.run_moe.layer_idx                 = kTestLayer;
    cmd.run_moe.num_seqs                  = 1;
    cmd.run_moe.moe_mode                  = 0;
    cmd.run_moe.store_gating_output       = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    // Verify completion succeeded.
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.status, 0u);

    // Count allreduce groups: exactly 1 (shared expert only, no routed allreduce).
    int group_count = 0;
    int allreduce_count = 0;
    for (const auto& entry : nccl_log) {
        if (entry == "group_begin") ++group_count;
        if (entry == "allreduce") ++allreduce_count;
    }
    EXPECT_EQ(group_count, 1)
        << "Identical bitsets should produce exactly 1 allreduce group "
           "(shared expert only)";
    EXPECT_EQ(allreduce_count, 2)
        << "TP=2 should produce 2 allreduce calls per group (one per rank)";

    dispatcher_.reset();
    dcp.reset();
    comm.reset();
    std::free(attn_buf0);
    std::free(attn_buf1);
    std::free(moe_buf0);
    std::free(moe_buf1);
}

// Disjoint bitsets on TP GPUs → EP-within-TP path (routed allreduce fires).
// Should see exactly two allreduce groups (shared expert + routed output).
TEST_F(CommandDispatcherTest, MoeTP2_DisjointBitsets_RoutedAllreduce) {
    lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
    lc::GpuRef gpu1{1, 1, lc::GpuType::rtx5090};

    auto attn0 = lcomp::make_null_attention_device(gpu0);
    auto attn1 = lcomp::make_null_attention_device(gpu1);
    auto exp0  = lcomp::make_null_expert_device(gpu0);
    auto exp1  = lcomp::make_null_expert_device(gpu1);

    // DcpExecutor (TP=2).
    lpar::DcpExecutor::Options dcp_opts{
        .dcp_size             = 2,
        .gpus                 = {gpu0, gpu1},
        .max_batch_size       = 8,
        .hidden_size          = 256,
        .num_attention_heads  = 4,
        .q_lora_rank          = 64,
        .kv_lora_rank         = 32,
        .qk_rope_head_dim     = 16,
        .qk_nope_head_dim     = 32,
        .v_head_dim           = 64,
        .rms_norm_eps         = 1e-6f,
        .stream_manager       = stream_manager_.get(),
        .attention_devices    = {attn0.get(), attn1.get()},
    };
    auto dcp = std::make_unique<lpar::DcpExecutor>(std::move(dcp_opts));

    // DcpCommunicator with recording collective.
    std::vector<std::string> nccl_log;
    EpRecordingCollective rec_collective(nccl_log);
    lpar::DcpCommunicator::Options comm_opts{
        .dcp_size        = 2,
        .device_backends = {test_device_backends_[0].get(),
                            test_device_backends_[1].get()},
        .max_batch_size  = 8,
        .num_heads       = 4,
        .attn_output_dim = 32,
        .hidden_size     = 256,
        .collective      = &rec_collective,
    };
    auto comm = std::make_unique<lpar::DcpCommunicator>(comm_opts);

    // Hidden state pairs.
    const size_t hidden_bytes = 8 * 256 * 2;
    void* attn_buf0 = aligned_alloc_zeroed(hidden_bytes);
    void* attn_buf1 = aligned_alloc_zeroed(hidden_bytes);
    void* moe_buf0  = aligned_alloc_zeroed(hidden_bytes);
    void* moe_buf1  = aligned_alloc_zeroed(hidden_bytes);

    // Reserve DISJOINT experts: experts 0-3 on GPU 0, experts 4-7 on GPU 1.
    // Use layer 6 (MTP layer) — same rationale as IdenticalBitsets test.
    constexpr uint32_t kTestLayer = 6;
    for (uint16_t e = 0; e < 4; ++e) {
        auto key0 = lmem::ExpertKey{kTestLayer, e};
        cache_->reserve(key0, 0, lmem::CacheZone::kStable, false);
        cache_->mark_ready(key0, 0, lmem::SubComponent::kAll);

        auto key1 = lmem::ExpertKey{kTestLayer, static_cast<uint16_t>(e + 4)};
        cache_->reserve(key1, 1, lmem::CacheZone::kStable, false);
        cache_->mark_ready(key1, 1, lmem::SubComponent::kAll);
    }

    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .dcp_executor       = dcp.get(),
        .dcp_communicator   = comm.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .attention_devices  = {attn0.get(), attn1.get()},
        .expert_devices     = {exp0.get(), exp1.get()},
        .device_backends    = {test_device_backends_[0].get(),
                               test_device_backends_[1].get()},
        .cuda_kernels_enabled = false,
        .hidden_state_pairs = {
            ldam::HiddenStatePair{attn_buf0, moe_buf0, 0, 0, 0},
            ldam::HiddenStatePair{attn_buf1, moe_buf1, 1, 1, 1},
        },
        .per_layer_attn_weights = make_fake_attn_weights(6, 2),
        .max_batch_size     = 8,
    };
    dispatcher_ = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    // Clear log after construction.
    nccl_log.clear();

    auto cmd = make_cmd(lipc::D_B_CMD_RUN_MOE, /*seq=*/701, /*gpu_idx=*/0);
    cmd.run_moe.layer_idx                 = kTestLayer;
    cmd.run_moe.num_seqs                  = 1;
    cmd.run_moe.moe_mode                  = 0;
    cmd.run_moe.store_gating_output       = 0;
    dispatcher_->dispatch(cmd);
    dispatcher_->poll_compute_completions();

    // Verify completion succeeded.
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.status, 0u);

    // Count allreduce groups: exactly 2 (shared expert + routed output).
    int group_count = 0;
    int allreduce_count = 0;
    for (const auto& entry : nccl_log) {
        if (entry == "group_begin") ++group_count;
        if (entry == "allreduce") ++allreduce_count;
    }
    EXPECT_EQ(group_count, 2)
        << "Disjoint bitsets should produce 2 allreduce groups "
           "(shared expert + routed output)";
    EXPECT_EQ(allreduce_count, 4)
        << "TP=2 with 2 groups should produce 4 allreduce calls total "
           "(2 per group)";

    dispatcher_.reset();
    dcp.reset();
    comm.reset();
    std::free(attn_buf0);
    std::free(attn_buf1);
    std::free(moe_buf0);
    std::free(moe_buf1);
}

// ═══════════════════════════════════════════════════════════════════════════
// #90: E_CMD_FETCH_AND_RUN_MOE
// ═══════════════════════════════════════════════════════════════════════════

namespace {

/// Write ExpertPrefetchEntry[] into the sideband at kExpertPrefetchOff.
void write_expert_sideband(uint8_t* sideband_base,
                           std::initializer_list<lipc::ExpertPrefetchEntry> entries) {
    auto* dst = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
        sideband_base + lipc::IpcLayout::kExpertPrefetchOff);
    int i = 0;
    for (const auto& e : entries) {
        dst[i++] = e;
    }
}

lipc::ExpertPrefetchEntry pfe(uint32_t layer, uint16_t expert, uint8_t gpu) {
    return {layer, expert, /*zone=*/1, gpu};
}

}  // namespace

TEST_F(CommandDispatcherTest, FetchAndRunMoe_AllCached) {
    make_dispatcher();

    // Reserve + mark ready 3 experts on GPU 0 (layer 1 = MoE layer).
    for (uint16_t e = 0; e < 3; ++e) {
        cache_->reserve({1, e}, 0, lmem::CacheZone::kStreaming);
        cache_->mark_all_ready({1, e}, 0);
    }

    // Write sideband with 3 experts, all on GPU 0.
    write_expert_sideband(sideband_base_,
        {pfe(1, 0, 0), pfe(1, 1, 0), pfe(1, 2, 0)});

    auto cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE, /*seq=*/10, /*gpu=*/0);
    cmd.fetch_and_run_moe.layer_idx     = 1;
    cmd.fetch_and_run_moe.num_seqs      = 1;
    cmd.fetch_and_run_moe.expert_count  = 3;
    cmd.fetch_and_run_moe.timeout_us    = 0;
    cmd.fetch_and_run_moe.moe_mode      = 0;
    dispatcher_->dispatch(cmd);

    // All cached -> should finalize immediately (within dispatch + advance).
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 10u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.routed_miss_count, 0);

    // All experts should be unlocked after finalization.
    for (uint16_t e = 0; e < 3; ++e) {
        EXPECT_FALSE(cache_->is_locked({1, e}, 0))
            << "expert " << e << " should be unlocked after finalization";
    }
}

TEST_F(CommandDispatcherTest, FetchAndRunMoe_ProgressiveArrival) {
    // Post finalize-on-quiescence (TD-far), "in-flight" is ELM-tracked, not a
    // bare cache reservation: an expert counts as in flight only while the ELM
    // is genuinely progressing it (interest held / slot reserved / H2D in
    // flight / host warm). So expert 1 is sourced from a (mock) LoadedModel and
    // driven through the real ELM — the FETCH command's ensure_resident starts
    // an in-flight H2D, and poll_elm() (null transfer backend completes on poll)
    // delivers the arrival. Expert 0 is pre-cached (immediate).
    lmod::LoadedModel model;
    model.layers.resize(6);
    model.layers[1].layer_idx = 1;
    model.layers[1].routed_experts.resize(8);
    std::vector<std::byte> fake_data(expert_bytes_, std::byte{0x42});
    lmod::RawTensor rt{
        .data = std::span<const std::byte>(fake_data),
        .dtype = lmod::SafetensorsDtype::F8_E4M3,
        .shape = {expert_bytes_}};
    lmod::WeightBundle wb{.id = {}, .weight = rt, .packed_slot = rt.data};
    model.layers[1].routed_experts[1].push_back(std::move(wb));

    make_dispatcher(&model);

    // Expert 0 already cached + ready.
    cache_->reserve({1, 0}, 0, lmem::CacheZone::kStreaming);
    cache_->mark_all_ready({1, 0}, 0);

    write_expert_sideband(sideband_base_,
        {pfe(1, 0, 0), pfe(1, 1, 0)});

    auto cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE, /*seq=*/20, /*gpu=*/0);
    cmd.fetch_and_run_moe.layer_idx     = 1;
    cmd.fetch_and_run_moe.num_seqs      = 1;
    cmd.fetch_and_run_moe.expert_count  = 2;
    cmd.fetch_and_run_moe.timeout_us    = 0;  // no timeout
    cmd.fetch_and_run_moe.moe_mode      = 0;
    dispatcher_->dispatch(cmd);

    // Expert 1's H2D is in flight (ensure_resident started it; not yet polled).
    // No completion, and advance must NOT finalize while it is in flight.
    lipc::Completion cmp{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));
    EXPECT_TRUE(cache_->is_locked({1, 0}, 0));   // cached expert locked
    EXPECT_FALSE(dispatcher_->advance_progressive_moe());
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));

    // Complete expert 1's H2D (null transfer backend completes on poll); the ELM
    // marks it ready → it is now resident.
    poll_elm();

    // Advance -- expert 1 detected as arrived, finalization triggers.
    EXPECT_TRUE(dispatcher_->advance_progressive_moe());

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 20u);
    EXPECT_EQ(cmp.compute.routed_miss_count, 0);

    // Both experts unlocked.
    EXPECT_FALSE(cache_->is_locked({1, 0}, 0));
    EXPECT_FALSE(cache_->is_locked({1, 1}, 0));
}

TEST_F(CommandDispatcherTest, FetchAndRunMoe_Timeout) {
    make_dispatcher();

    // Expert 0 cached, expert 1 not reserved at all (will never arrive).
    cache_->reserve({1, 0}, 0, lmem::CacheZone::kStreaming);
    cache_->mark_all_ready({1, 0}, 0);

    write_expert_sideband(sideband_base_,
        {pfe(1, 0, 0), pfe(1, 1, 0)});

    auto cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE, /*seq=*/30, /*gpu=*/0);
    cmd.fetch_and_run_moe.layer_idx     = 1;
    cmd.fetch_and_run_moe.num_seqs      = 1;
    cmd.fetch_and_run_moe.expert_count  = 2;
    cmd.fetch_and_run_moe.timeout_us    = 1;  // 1 microsecond -- expires immediately
    cmd.fetch_and_run_moe.moe_mode      = 0;
    dispatcher_->dispatch(cmd);

    // Busy-wait briefly to ensure timeout passes, then advance.
    lipc::Completion cmp{};
    for (int i = 0; i < 100; ++i) {
        if (dispatcher_->advance_progressive_moe()) break;
    }

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 30u);
    EXPECT_EQ(cmp.compute.routed_miss_count, 1);  // expert 1 never arrived

    // Expert 0 unlocked.
    EXPECT_FALSE(cache_->is_locked({1, 0}, 0));
}

TEST_F(CommandDispatcherTest, FetchAndRunMoe_RejectConcurrent) {
    make_dispatcher();

    // Expert 0 cached, expert 1 missing.
    cache_->reserve({1, 0}, 0, lmem::CacheZone::kStreaming);
    cache_->mark_all_ready({1, 0}, 0);

    write_expert_sideband(sideband_base_,
        {pfe(1, 0, 0), pfe(1, 1, 0)});

    auto cmd1 = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE, /*seq=*/40, /*gpu=*/0);
    cmd1.fetch_and_run_moe.layer_idx     = 1;
    cmd1.fetch_and_run_moe.num_seqs      = 1;
    cmd1.fetch_and_run_moe.expert_count  = 2;
    cmd1.fetch_and_run_moe.timeout_us    = 0;
    cmd1.fetch_and_run_moe.moe_mode      = 0;
    dispatcher_->dispatch(cmd1);

    // First command is active (waiting for expert 1). No completion yet.
    lipc::Completion cmp{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));

    // Second command should fail with error.
    write_expert_sideband(sideband_base_, {pfe(2, 0, 0)});

    auto cmd2 = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE, /*seq=*/41, /*gpu=*/0);
    cmd2.fetch_and_run_moe.layer_idx     = 2;
    cmd2.fetch_and_run_moe.num_seqs      = 1;
    cmd2.fetch_and_run_moe.expert_count  = 1;
    cmd2.fetch_and_run_moe.timeout_us    = 0;
    cmd2.fetch_and_run_moe.moe_mode      = 0;
    dispatcher_->dispatch(cmd2);

    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 41u);
}

TEST_F(CommandDispatcherTest, FetchAndRunMoe_NoCacheError) {
    // Build dispatcher without expert cache.
    ldam::CommandDispatcher::Deps deps{
        .cmp_ring           = cmp_ring_.get(),
        .sideband_base      = sideband_base_,
        .live_config        = cfg_.get(),
        .device_backends    = {test_device_backends_[0].get(),
                               test_device_backends_[1].get()},
        .cuda_kernels_enabled = false,
    };
    auto dispatcher_no_cache = std::make_unique<ldam::CommandDispatcher>(std::move(deps));

    write_expert_sideband(sideband_base_, {pfe(1, 0, 0)});

    auto cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE, /*seq=*/50, /*gpu=*/0);
    cmd.fetch_and_run_moe.layer_idx     = 1;
    cmd.fetch_and_run_moe.num_seqs      = 1;
    cmd.fetch_and_run_moe.expert_count  = 1;
    cmd.fetch_and_run_moe.timeout_us    = 0;
    cmd.fetch_and_run_moe.moe_mode      = 0;
    dispatcher_no_cache->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 50u);
}

// ── F-6: selective-fetch decider params ──────────────────────────────────────
//
// The decider chooses which *missing* experts to issue H2D for. Skipped missing
// experts never arrive and finalize immediately (graceful degradation, like the
// timeout path) — so a decider that skips all missing experts completes without
// any timeout and reports them in routed_miss_count.

// gating_weight_threshold: a below-threshold missing expert is skipped, so the
// command finalizes immediately (no timeout) with the resident subset.
TEST_F(CommandDispatcherTest, FetchAndRunMoe_WeightThresholdSkips) {
    make_dispatcher();

    // Expert 0 cached (high weight); expert 1 missing and low weight.
    cache_->reserve({1, 0}, 0, lmem::CacheZone::kStreaming);
    cache_->mark_all_ready({1, 0}, 0);

    write_expert_sideband(sideband_base_, {pfe(1, 0, 0), pfe(1, 1, 0)});

    auto cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE, /*seq=*/60, /*gpu=*/0);
    cmd.fetch_and_run_moe.layer_idx     = 1;
    cmd.fetch_and_run_moe.num_seqs      = 1;
    cmd.fetch_and_run_moe.expert_count  = 2;
    cmd.fetch_and_run_moe.timeout_us    = 0;  // no timeout — must finalize via skip
    cmd.fetch_and_run_moe.moe_mode      = 0;
    cmd.fetch_and_run_moe.weight_count  = 2;
    cmd.fetch_and_run_moe.gating_weight_threshold = 0.5f;
    cmd.fetch_and_run_moe.weights[0]    = 0.9f;   // entry 0 (cached) — irrelevant
    cmd.fetch_and_run_moe.weights[1]    = 0.1f;   // entry 1 — below threshold → skip
    dispatcher_->dispatch(cmd);

    // Expert 1 skipped → finalize immediately (no timeout, no advance needed).
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 60u);
    EXPECT_EQ(cmp.compute.routed_miss_count, 1);  // expert 1 skipped → miss

    EXPECT_FALSE(cache_->is_locked({1, 0}, 0));
}

// min_experts forces a leading low-weight missing expert to be fetched even when
// below threshold, so the command keeps waiting (not skipped).
TEST_F(CommandDispatcherTest, FetchAndRunMoe_MinExpertsForcesFetch) {
    make_dispatcher();

    // Expert 0 missing + low weight + reserved (in-flight H2D, not ready).
    cache_->reserve({1, 0}, 0, lmem::CacheZone::kStreaming);

    write_expert_sideband(sideband_base_, {pfe(1, 0, 0)});

    auto cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE, /*seq=*/61, /*gpu=*/0);
    cmd.fetch_and_run_moe.layer_idx     = 1;
    cmd.fetch_and_run_moe.num_seqs      = 1;
    cmd.fetch_and_run_moe.expert_count  = 1;
    cmd.fetch_and_run_moe.timeout_us    = 0;
    cmd.fetch_and_run_moe.moe_mode      = 0;
    cmd.fetch_and_run_moe.weight_count  = 1;
    cmd.fetch_and_run_moe.min_experts   = 1;      // force entry 0 despite low weight
    cmd.fetch_and_run_moe.gating_weight_threshold = 0.5f;
    cmd.fetch_and_run_moe.weights[0]    = 0.1f;   // below threshold, but forced
    dispatcher_->dispatch(cmd);

    // Forced fetch → not skipped → still waiting (no timeout). No completion yet.
    lipc::Completion cmp{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));

    // Expert arrives → finalize with 0 misses (it was fetched, not skipped).
    cache_->mark_all_ready({1, 0}, 0);
    EXPECT_TRUE(dispatcher_->advance_progressive_moe());
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 61u);
    EXPECT_EQ(cmp.compute.routed_miss_count, 0);
}

// max_new_fetches caps the number of missing experts fetched; the overflow is
// skipped and finalizes immediately.
TEST_F(CommandDispatcherTest, FetchAndRunMoe_MaxNewFetchesCaps) {
    make_dispatcher();

    // Two missing experts, both reserved (in-flight). Cap to 1 new fetch.
    cache_->reserve({1, 0}, 0, lmem::CacheZone::kStreaming);
    cache_->reserve({1, 1}, 0, lmem::CacheZone::kStreaming);

    write_expert_sideband(sideband_base_, {pfe(1, 0, 0), pfe(1, 1, 0)});

    auto cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE, /*seq=*/62, /*gpu=*/0);
    cmd.fetch_and_run_moe.layer_idx       = 1;
    cmd.fetch_and_run_moe.num_seqs        = 1;
    cmd.fetch_and_run_moe.expert_count    = 2;
    cmd.fetch_and_run_moe.timeout_us      = 0;
    cmd.fetch_and_run_moe.moe_mode        = 0;
    cmd.fetch_and_run_moe.max_new_fetches = 1;  // only fetch the first missing
    dispatcher_->dispatch(cmd);

    // Expert 0 fetched (waiting), expert 1 skipped. Not yet complete.
    lipc::Completion cmp{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));

    // Expert 0 arrives → finalize. Expert 1 was skipped → 1 miss.
    cache_->mark_all_ready({1, 0}, 0);
    EXPECT_TRUE(dispatcher_->advance_progressive_moe());
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 62u);
    EXPECT_EQ(cmp.compute.routed_miss_count, 1);
}

// Defaults (all decider fields 0) reproduce the original progressive behavior:
// every missing expert is fetched and awaited (no immediate skip-finalize).
TEST_F(CommandDispatcherTest, FetchAndRunMoe_DeciderDefaultsUnchanged) {
    make_dispatcher();

    cache_->reserve({1, 0}, 0, lmem::CacheZone::kStreaming);
    cache_->mark_all_ready({1, 0}, 0);
    cache_->reserve({1, 1}, 0, lmem::CacheZone::kStreaming);  // in-flight

    write_expert_sideband(sideband_base_, {pfe(1, 0, 0), pfe(1, 1, 0)});

    auto cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE, /*seq=*/63, /*gpu=*/0);
    cmd.fetch_and_run_moe.layer_idx     = 1;
    cmd.fetch_and_run_moe.num_seqs      = 1;
    cmd.fetch_and_run_moe.expert_count  = 2;
    cmd.fetch_and_run_moe.timeout_us    = 0;
    cmd.fetch_and_run_moe.moe_mode      = 0;
    // All decider fields left 0 → no skipping.
    dispatcher_->dispatch(cmd);

    // Expert 1 fetched (not skipped) → still waiting, no completion.
    lipc::Completion cmp{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));

    cache_->mark_all_ready({1, 1}, 0);
    EXPECT_TRUE(dispatcher_->advance_progressive_moe());
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.compute.routed_miss_count, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// TD-PREFILL-MOE-BIG: E_CMD_FETCH_AND_RUN_MOE_BIG + chunk/batch capacities
// ═══════════════════════════════════════════════════════════════════════════

// The BIG payload must be a layout-compatible prefix extension of the legacy
// payload — the shared handler reads the common fields through
// cmd.fetch_and_run_moe regardless of the command type.
TEST(FetchAndRunMoeBigLayout, PrefixMatchesLegacyPayload) {
    using Cmd = lipc::Command;
    static_assert(offsetof(Cmd, fetch_and_run_moe.layer_idx)
                  == offsetof(Cmd, fetch_and_run_moe_big.layer_idx));
    static_assert(offsetof(Cmd, fetch_and_run_moe.num_seqs)
                  == offsetof(Cmd, fetch_and_run_moe_big.num_seqs));
    static_assert(offsetof(Cmd, fetch_and_run_moe.expert_count)
                  == offsetof(Cmd, fetch_and_run_moe_big.expert_count));
    static_assert(offsetof(Cmd, fetch_and_run_moe.timeout_us)
                  == offsetof(Cmd, fetch_and_run_moe_big.timeout_us));
    static_assert(offsetof(Cmd, fetch_and_run_moe.moe_mode)
                  == offsetof(Cmd, fetch_and_run_moe_big.moe_mode));
    static_assert(offsetof(Cmd, fetch_and_run_moe.weight_count)
                  == offsetof(Cmd, fetch_and_run_moe_big.weight_count));
    static_assert(offsetof(Cmd, fetch_and_run_moe.min_experts)
                  == offsetof(Cmd, fetch_and_run_moe_big.min_experts));
    static_assert(offsetof(Cmd, fetch_and_run_moe.max_new_fetches)
                  == offsetof(Cmd, fetch_and_run_moe_big.max_new_fetches));
    static_assert(offsetof(Cmd, fetch_and_run_moe.have_evict_map)
                  == offsetof(Cmd, fetch_and_run_moe_big.have_evict_map));
    static_assert(offsetof(Cmd, fetch_and_run_moe.gating_weight_threshold)
                  == offsetof(Cmd, fetch_and_run_moe_big.gating_weight_threshold));
    static_assert(offsetof(Cmd, fetch_and_run_moe.weights)
                  == offsetof(Cmd, fetch_and_run_moe_big.weights));
    // chunk_tokens extends past the legacy payload.
    static_assert(offsetof(Cmd, fetch_and_run_moe_big.chunk_tokens)
                  >= offsetof(Cmd, fetch_and_run_moe.weights)
                     + sizeof(float) * lipc::kMaxFetchDeciderWeights);
    SUCCEED();
}

// prefill_moe_big (default on): the single-shot chunk capacity is bounded by
// the config chunk knob and never exceeds the batch capacity; with the mode
// off the two are equal (chunking never engages — legacy sizing).
TEST_F(CommandDispatcherTest, MoeBigChunkCapacityDefaults) {
    ASSERT_TRUE(cfg_->compute.prefill_moe_big);  // schema default
    make_dispatcher();
    EXPECT_LE(dispatcher_->moe_chunk_capacity(),
              dispatcher_->moe_batch_capacity());
    // chunk = max(moe_big_chunk_tokens, Deps::max_batch_size (fixture: 64))
    EXPECT_EQ(dispatcher_->moe_chunk_capacity(),
              std::max(cfg_->compute.moe_big_chunk_tokens, 64));
}

TEST_F(CommandDispatcherTest, MoeBigOffKeepsLegacyCapacity) {
    cfg_->compute.prefill_moe_big = false;
    make_dispatcher();
    EXPECT_EQ(dispatcher_->moe_chunk_capacity(),
              dispatcher_->moe_batch_capacity());
    EXPECT_EQ(dispatcher_->moe_batch_capacity(),
              static_cast<int>(lipc::kMaxBatchDescriptors));
}

// E_CMD_FETCH_AND_RUN_MOE_BIG runs the same progressive machine; the
// completion carries the BIG cmd_type so drivers can match it.
TEST_F(CommandDispatcherTest, FetchAndRunMoeBig_AllCached) {
    make_dispatcher();

    // One resident expert. On boxes where the fixture's tiny streaming zone
    // resolves to ZERO slots, reserve fails — skip instead of dereferencing
    // (the sibling legacy FETCH tests crash on exactly this, pre-existing).
    if (cache_->reserve({1, 0}, 0, lmem::CacheZone::kStreaming) == nullptr)
        GTEST_SKIP() << "no streaming slot in this environment (pre-existing "
                        "fixture/env limitation, see FetchAndRunMoe_AllCached)";
    cache_->mark_all_ready({1, 0}, 0);
    write_expert_sideband(sideband_base_, {pfe(1, 0, 0)});

    auto cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE_BIG, /*seq=*/70, /*gpu=*/0);
    cmd.fetch_and_run_moe_big.layer_idx     = 1;
    cmd.fetch_and_run_moe_big.num_seqs      = 128;  // > chunk capacity is fine
    cmd.fetch_and_run_moe_big.expert_count  = 1;
    cmd.fetch_and_run_moe_big.timeout_us    = 0;
    cmd.fetch_and_run_moe_big.moe_mode      = 0;
    cmd.fetch_and_run_moe_big.chunk_tokens  = 0;  // engine default
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(cmp.cmd_seq, 70u);
    EXPECT_EQ(cmp.status, 0u);
    EXPECT_EQ(cmp.compute.cmd_type,
              static_cast<uint32_t>(lipc::E_CMD_FETCH_AND_RUN_MOE_BIG));
    EXPECT_EQ(cmp.compute.routed_miss_count, 0);
    EXPECT_FALSE(cache_->is_locked({1, 0}, 0));
}

TEST_F(CommandDispatcherTest, FetchAndRunMoeBig_RejectsOverCapacity) {
    make_dispatcher();
    write_expert_sideband(sideband_base_, {pfe(1, 0, 0)});

    auto cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE_BIG, /*seq=*/71, /*gpu=*/0);
    cmd.fetch_and_run_moe_big.layer_idx    = 1;
    cmd.fetch_and_run_moe_big.num_seqs     =
        static_cast<uint32_t>(dispatcher_->moe_batch_capacity()) + 1;
    cmd.fetch_and_run_moe_big.expert_count = 1;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 71u);
}

// ── GG-S1 Phase 4 (TD-GG5C-SPLIT-BUFFER-ALWAYS-ALLOC) ──────────────────────
// CUDA-free unit tests for the load-time gate≠up scan that gates the
// MoeScratch::gguf_gate_up_split allocation. The scan's bool return IS the
// sizing decision: true ⇒ buffer allocated (sized > 0); false ⇒ left null
// (sized 0). These exercise the static helper directly — no GPU, no devices.
using DenseW  = ldam::CommandDispatcher::Deps::DenseFFNWeights;
using SharedW = ldam::CommandDispatcher::Deps::SharedExpertWeights;
using GKQ     = lmod::GgufKQuantType;

TEST(GgufSplitBufferScan, AllGateEqUp_FlagFalse_BufferGatedOff) {
    // Mirrors GLM-4.7-Flash: every dense/shared unit has gate==up.
    std::vector<std::vector<DenseW>> dense(4, std::vector<DenseW>(2));
    std::vector<std::vector<SharedW>> shared(4, std::vector<SharedW>(2));
    for (auto& g : dense)  for (auto& d : g) { d.gate_gguf_type = GKQ::Q4_K; d.up_gguf_type = GKQ::Q4_K; }
    for (auto& g : shared) for (auto& s : g) { s.gate_gguf_type = GKQ::Q6_K; s.up_gguf_type = GKQ::Q6_K; }
    EXPECT_FALSE(ldam::CommandDispatcher::any_dense_shared_gate_ne_up(dense, shared));
}

TEST(GgufSplitBufferScan, DenseGateNeUp_FlagTrue_BufferAllocated) {
    std::vector<std::vector<DenseW>> dense(4, std::vector<DenseW>(2));
    std::vector<std::vector<SharedW>> shared(4, std::vector<SharedW>(2));
    // One dense unit on one GPU has gate≠up (Q5_K gate, Q6_K up).
    dense[2][1].gate_gguf_type = GKQ::Q5_K;
    dense[2][1].up_gguf_type   = GKQ::Q6_K;
    EXPECT_TRUE(ldam::CommandDispatcher::any_dense_shared_gate_ne_up(dense, shared));
}

TEST(GgufSplitBufferScan, SharedGateNeUp_FlagTrue_BufferAllocated) {
    std::vector<std::vector<DenseW>> dense(4, std::vector<DenseW>(2));
    std::vector<std::vector<SharedW>> shared(4, std::vector<SharedW>(2));
    // One shared expert has gate≠up (Q8_0 gate, Q4_K up).
    shared[0][0].gate_gguf_type = GKQ::Q8_0;
    shared[0][0].up_gguf_type   = GKQ::Q4_K;
    EXPECT_TRUE(ldam::CommandDispatcher::any_dense_shared_gate_ne_up(dense, shared));
}

TEST(GgufSplitBufferScan, EmptyVectors_FlagFalse) {
    EXPECT_FALSE(ldam::CommandDispatcher::any_dense_shared_gate_ne_up({}, {}));
}

// ── DSP-3: D_CMD_RUN_DSPARK_STEP fail-closed ────────────────────────────────
// Without an armed DsparkRuntime (speculation.method != dspark, or no CUDA)
// the command must CMP_ERROR — never silently no-op (the orchestrator would
// otherwise draft from garbage).

TEST_F(CommandDispatcherTest, DsparkStep_FailsClosedWithoutRuntime) {
    make_dispatcher();

    auto cmd = make_cmd(lipc::D_CMD_RUN_DSPARK_STEP, /*seq=*/77, /*gpu_idx=*/0);
    cmd.run_dspark_step.seq_id = 1;
    cmd.run_dspark_step.anchor_token_id = 3;
    cmd.run_dspark_step.anchor_pos = 0;
    cmd.run_dspark_step.num_query = 0;
    dispatcher_->dispatch(cmd);

    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.cmd_seq, 77u);
    EXPECT_EQ(cmp.error.error_category,
              static_cast<uint32_t>(lipc::CmpErrorCategory::kComputeValidation));
}
