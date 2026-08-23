// TD-GLM-INDEXER-LOCAL-MERGE: local-mode indexer-K PROVISIONING shape.
//
// Under hardware.dcp_indexer_mode=local at dcp>=2, the dispatcher's
// ensure_indexer_pages must allocate ONE kIndexerK page per (logical page,
// computing layer) — on the OWNER rank's GPU only (round-robin by INDEXER
// page: owner(pg) = pg % dcp) — instead of the replicated dcp_size replicas.
// Observable via the PageAllocator: after covering N positions, GPU g holds
// exactly (#pages owned by rank g) × n_computing_layers kIndexerK pages, and
// the two GPUs' counts sum to pages × layers (HALF the replicated total —
// the entire point of local mode). Also locks:
//   - idempotent re-dispatch (later layer, same step) allocates nothing new;
//   - decode growth allocates the next page on the alternating owner;
//   - the coverage guard blesses the paged mode (never the arena — the
//     executor arena is replicated-shape and illegal under local).
//
// CPU-only: null device backends/attention devices, real PageAllocator +
// dispatcher provisioning path (mirrors indexer_coverage_guard_test at
// dcp_size=2).

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "compute/graphs/graph_registry.h"
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
#include "parallelism/dcp_executor.h"

namespace lipc = layerstorm::ipc;
namespace ldam = layerstorm::daemon;
namespace lmem = layerstorm::memory;
namespace lmod = layerstorm::model;
namespace lcomp = layerstorm::compute;
namespace lpar = layerstorm::parallelism;
namespace lc   = layerstorm::config;

namespace {

constexpr uint32_t kTestSlots = 128;
constexpr int kModePaged = 1;  // CommandDispatcher::IndexerSeqMode::kPaged
constexpr int kPT = 16;        // indexer_k_page_size_tokens (ownership unit)
constexpr int kLayers = 6;     // all computing (no IndexShare mask)

void* aligned_alloc_zeroed(size_t bytes) {
    void* p = std::aligned_alloc(64, bytes);
    std::memset(p, 0, bytes);
    return p;
}

/// DSA-enabled dcp=2 config with dcp_indexer_mode=local and a tiny indexer
/// page so a short prompt spans several ownership units.
lc::Config dsa_local_config() {
    auto j = nlohmann::json{
        {"model", {
            {"architecture",            "deepseek_v3"},
            {"weights_path",            "/data/models/test/"},
            {"weights_format",          "safetensors"},
            {"num_hidden_layers",       kLayers},
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
            {"index_topk",              2048},
            {"num_nextn_predict_layers", 0},
            {"moe_intermediate_size",   128},
        }},
        {"quantization", {{"weights", "fp8_e4m3"},
                          {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"},
                          {"gating_compute", "fp32"}}},
        {"hardware", {
            {"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 2}},
                      {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 2}}}},
            {"tp_array", {0, 1}},
            {"dcp_indexer_mode", "local"},
            // Replicated KV explicitly: this fixture is replicated-shaped
            // (PageAllocator DcpConfig.kv_sharded = false, dispatcher
            // kv_page_size 64 vs dcp_chunk_size 16) and isolates INDEXER
            // sharding. Pinned because the schema default is now 'sharded'.
            {"dcp_kv_mode", "replicated"},
            {"system_ram_gb", 64}}},
        {"parallelism", {{"tensor_parallelism", 2}}},
        {"serving", {{"max_sequence_length", 1024}}},
        {"memory", {{"vram_safety_margin_gb", 0.1},
                    {"kv_cache", {{"page_growth_chunk_tokens", 64},
                                  {"indexer_k_page_size_tokens", kPT}}}}},
    };
    return lc::parse_config(j);
}

}  // namespace

class IndexerLocalProvisioningTest : public ::testing::Test {
protected:
    void SetUp() override {
        ipc_bytes_ = lipc::IpcLayout::total_size(kTestSlots, kTestSlots);
        ipc_region_ = static_cast<uint8_t*>(aligned_alloc_zeroed(ipc_bytes_));
        void* cmp_ptr = ipc_region_
                      + lipc::IpcLayout::cmp_ring_offset(kTestSlots);
        lipc::CompletionRing::init(cmp_ptr, kTestSlots);
        cmp_ring_ = std::make_unique<lipc::CompletionRing>(cmp_ptr);
        sideband_ = ipc_region_
                  + lipc::IpcLayout::sideband_offset(kTestSlots, kTestSlots);

        cfg_ = std::make_unique<lc::Config>(dsa_local_config());
        mcfg_ = std::make_unique<lmod::ModelConfig>(*cfg_);
        fp8_ = std::make_unique<lmod::Fp8E4M3>();
        layer_reg_ = std::make_unique<lmod::LayerRegistry>(*mcfg_, *cfg_, *fp8_);

        lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
        lc::GpuRef gpu1{1, 1, lc::GpuType::rtx5090};
        backends_.push_back(lcomp::make_null_device_backend(gpu0));
        backends_.push_back(lcomp::make_null_device_backend(gpu1));

        std::vector<lcomp::DeviceBackend*> dev_ptrs{backends_[0].get(),
                                                    backends_[1].get()};
        auto layout = lmem::compute_vram_layout(*cfg_, *layer_reg_, *mcfg_);
        vram_ = std::make_unique<lmem::VramAllocator>(std::move(layout),
                                                      dev_ptrs);
        page_allocator_ = std::make_unique<lmem::PageAllocator>(
            *vram_, backends_[0].get());
        page_allocator_->set_dcp_config(lmem::DcpConfig{
            .dcp_size = 2,
            .dcp_chunk_size = cfg_->memory.kv_cache.dcp_chunk_size,
            .page_size_tokens = cfg_->memory.kv_cache.page_size_tokens,
            .tp_gpu_indices = {0, 1},
            .indexer_k_sharded = true,
            .indexer_k_page_size_tokens = kPT,
            .kv_sharded = false,
        });

        lcomp::StreamManager::Options sm_opts{
            .device_backends = {backends_[0].get(), backends_[1].get()},
        };
        stream_manager_ = std::make_unique<lcomp::StreamManager>(
            std::move(sm_opts));

        attn_devices_.push_back(lcomp::make_null_attention_device(gpu0));
        attn_devices_.push_back(lcomp::make_null_attention_device(gpu1));
        lpar::DcpExecutor::Options dcp_opts{
            .dcp_size            = 2,
            .gpus                = {gpu0, gpu1},
            .max_batch_size      = 64,
            .num_layers          = kLayers,
            .hidden_size         = 256,
            .num_attention_heads = 4,
            .q_lora_rank         = 64,
            .kv_lora_rank        = 32,
            .qk_rope_head_dim    = 32,
            .qk_nope_head_dim    = 32,
            .v_head_dim          = 64,
            .rms_norm_eps        = 1e-6f,
            .stream_manager      = stream_manager_.get(),
            .attention_devices   = {attn_devices_[0].get(),
                                    attn_devices_[1].get()},
        };
        dcp_executor_ = std::make_unique<lpar::DcpExecutor>(
            std::move(dcp_opts));

        weights_ = std::vector<std::vector<lpar::AttentionLayerWeights>>(
            kLayers, std::vector<lpar::AttentionLayerWeights>(2));
        {
            std::vector<std::vector<const lpar::AttentionLayerWeights*>> wp(
                kLayers);
            for (int l = 0; l < kLayers; ++l)
                wp[l] = {&weights_[l][0], &weights_[l][1]};
            dcp_executor_->set_layer_weights(std::move(wp), kLayers);
        }

        hidden_buf0_ = aligned_alloc_zeroed(64 * 256 * 2);
        hidden_buf1_ = aligned_alloc_zeroed(64 * 256 * 2);

        ldam::CommandDispatcher::Deps deps{
            .cmp_ring               = cmp_ring_.get(),
            .stream_manager         = stream_manager_.get(),
            .dcp_executor           = dcp_executor_.get(),
            .page_allocator         = page_allocator_.get(),
            .sideband_base          = sideband_,
            .live_config            = cfg_.get(),
            .attention_devices      = {attn_devices_[0].get(),
                                       attn_devices_[1].get()},
            .device_backends        = {backends_[0].get(), backends_[1].get()},
            .cuda_kernels_enabled   = false,
            .kv_cache_stride_block  = 576 * 64,
            .kv_cache_stride_row    = 576,
            .kv_page_size           = 64,
            .hidden_state_pairs     = {
                ldam::HiddenStatePair{hidden_buf0_, nullptr, 0, 0, 0,
                                      nullptr, nullptr},
                ldam::HiddenStatePair{hidden_buf1_, nullptr, 1, 1, 1,
                                      nullptr, nullptr}},
            .per_layer_attn_weights = weights_,
            .max_batch_size         = 64,
        };
        dispatcher_ = std::make_unique<ldam::CommandDispatcher>(
            std::move(deps));
    }

    void TearDown() override {
        dispatcher_.reset();
        dcp_executor_.reset();
        attn_devices_.clear();
        std::free(hidden_buf0_);
        std::free(hidden_buf1_);
        std::free(ipc_region_);
    }

    void drain_completions() {
        dispatcher_->poll_compute_completions();
        lipc::Completion cmp{};
        while (cmp_ring_->try_read(&cmp)) {
            ASSERT_NE(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR))
                << "CMP_ERROR: " << cmp.error.message;
        }
    }

    void create_seq(uint64_t seq_id, uint32_t prompt_len) {
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(lipc::CMD_SEQ_CREATE);
        c.cmd_seq = next_seq_++;
        c.seq_create.seq_id = seq_id;
        c.seq_create.prompt_len = prompt_len;
        c.seq_create.pool = 0;
        dispatcher_->dispatch(c);
        drain_completions();
    }

    void attention_step(uint64_t seq_id, uint32_t pos0, uint32_t n,
                        bool prefill, uint32_t layer = 0) {
        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        for (uint32_t b = 0; b < n; ++b) {
            batch[b].seq_id = seq_id;
            batch[b].token_pos = pos0 + b;
            batch[b]._pad = 0;
        }
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(lipc::D_B_CMD_RUN_ATTENTION);
        c.cmd_seq = next_seq_++;
        c.run_attention.layer_idx = layer;
        c.run_attention.num_seqs = n;
        c.run_attention.is_prefill = prefill ? 1 : 0;
        c.run_attention.use_graph = 0;
        c.run_attention.is_draft = 0;
        c.run_attention.chunk_start = 0;
        c.run_attention.chunk_len = prefill ? n : 0;
        dispatcher_->dispatch(c);
        drain_completions();
    }

    int used_ik(int gpu) const {
        return page_allocator_->used_pages(gpu, lmem::Pool::kIndexerK);
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
    std::unique_ptr<lpar::DcpExecutor> dcp_executor_;
    std::vector<std::vector<lpar::AttentionLayerWeights>> weights_;
    void* hidden_buf0_ = nullptr;
    void* hidden_buf1_ = nullptr;
    std::unique_ptr<ldam::CommandDispatcher> dispatcher_;
    uint32_t next_seq_ = 1;
};

// A 40-position prefill chunk at PT=16 needs 3 indexer pages (0,1,2) per
// computing layer. Local mode: page 0,2 → GPU 0 (owner rank 0), page 1 →
// GPU 1. Every layer computes (no IndexShare mask): counts × kLayers.
TEST_F(IndexerLocalProvisioningTest, PrefillProvisionsOwnerShardedPages) {
    create_seq(1, 40);
    ASSERT_EQ(used_ik(0), 0);
    ASSERT_EQ(used_ik(1), 0);

    attention_step(1, /*pos0=*/0, /*n=*/40, /*prefill=*/true, /*layer=*/0);

    // Coverage blessed as PAGED (never the arena in local mode) + advanced.
    const auto cov = dispatcher_->indexer_coverage(1);
    EXPECT_EQ(cov.first, kModePaged);
    EXPECT_EQ(cov.second, 40u);

    // 3 pages × 6 layers, split 2:1 across the two GPUs by page ownership —
    // NOT 3×6 on EACH GPU (the replicated shape this mode exists to halve).
    EXPECT_EQ(used_ik(0), 2 * kLayers);
    EXPECT_EQ(used_ik(1), 1 * kLayers);

    // Idempotent re-dispatch for a later layer of the SAME step: no growth.
    attention_step(1, 0, 40, true, /*layer=*/1);
    EXPECT_EQ(used_ik(0), 2 * kLayers);
    EXPECT_EQ(used_ik(1), 1 * kLayers);

    // Contiguous decode steps: positions 40..47 stay inside page 2 (no
    // growth); position 48 opens page 3 → owner rank 1 → GPU 1 grows.
    for (uint32_t pos = 40; pos < 48; ++pos)
        attention_step(1, pos, 1, /*prefill=*/false, /*layer=*/0);
    EXPECT_EQ(used_ik(0), 2 * kLayers);
    EXPECT_EQ(used_ik(1), 1 * kLayers);

    attention_step(1, 48, 1, /*prefill=*/false, /*layer=*/0);
    EXPECT_EQ(used_ik(0), 2 * kLayers);
    EXPECT_EQ(used_ik(1), 2 * kLayers) << "page 3 must land on owner rank 1";

    // The sequence is still paged-sparse-eligible after growth.
    const auto cov2 = dispatcher_->indexer_coverage(1);
    EXPECT_EQ(cov2.first, kModePaged);
    EXPECT_EQ(cov2.second, 49u);
}
