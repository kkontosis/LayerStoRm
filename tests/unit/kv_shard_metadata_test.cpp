// KVS-2: unit tests for DCP sequence-sharded KV routing.
//
// Covers the pure token→rank math (kv_shard_math.h vs the PageAllocator's
// canonical dcp_rank_for_token) and the dispatcher's sharded
// build_kv_metadata branch (per-rank DIVERGENT seqlens/block tables/slot
// mappings, owner-only k_append via trash slots, disjoint coverage,
// INV-4.9e), plus a replicated-mode regression (rank rows identical — the
// validated default must not move).
//
// All tests run without CUDA (null backends, heap IPC) — same pattern as
// command_dispatcher_test.cpp.

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
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
#include "daemon/kv_shard_math.h"
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
namespace kvs  = layerstorm::daemon::kvshard;

// ═══════════════════════════════════════════════════════════════════════════
// Pure math: kv_shard_math.h
// ═══════════════════════════════════════════════════════════════════════════

TEST(KvShardMath, OwnerRankMatchesPageAllocator) {
    for (int dcp : {2, 3, 4}) {
        lmem::DcpConfig dc{};
        dc.dcp_size = dcp;
        dc.dcp_chunk_size = 16;
        dc.page_size_tokens = 16;
        for (uint32_t pos = 0; pos < 1000; ++pos) {
            EXPECT_EQ(kvs::owner_rank(pos, 16, dcp),
                      lmem::PageAllocator::dcp_rank_for_token(pos, dc))
                << "pos=" << pos << " dcp=" << dcp;
        }
    }
}

TEST(KvShardMath, OwnedLenDisjointComplete) {
    // Sum over ranks == n_tokens (INV-4.9e) and matches brute force.
    for (int dcp : {2, 3}) {
        for (int chunk : {16, 32}) {
            for (uint32_t n = 0; n <= 300; ++n) {
                int sum = 0;
                for (int r = 0; r < dcp; ++r) {
                    int brute = 0;
                    for (uint32_t p = 0; p < n; ++p)
                        if (kvs::owner_rank(p, chunk, dcp) == r) ++brute;
                    const int got = kvs::owned_len(r, n, chunk, dcp);
                    ASSERT_EQ(got, brute)
                        << "r=" << r << " n=" << n << " chunk=" << chunk
                        << " dcp=" << dcp;
                    sum += got;
                }
                ASSERT_EQ(sum, static_cast<int>(n));
            }
        }
    }
}

TEST(KvShardMath, LocalGlobalPageRoundtrip) {
    for (int dcp : {2, 3}) {
        for (int ppc : {1, 2, 4}) {
            std::vector<int> next_local(dcp, 0);
            for (int g = 0; g < 120; ++g) {
                const int owner = kvs::page_owner_rank(g, ppc, dcp);
                // Owner matches token-level owner of the page's first token
                // (page size 16, chunk = ppc * 16).
                EXPECT_EQ(owner, kvs::owner_rank(
                                     static_cast<uint32_t>(g) * 16,
                                     ppc * 16, dcp));
                const int local = kvs::local_page_index(g, ppc, dcp);
                // Local indices are dense per rank, in global order.
                EXPECT_EQ(local, next_local[owner]) << "g=" << g;
                ++next_local[owner];
                // Roundtrip.
                EXPECT_EQ(kvs::global_page_of_local(owner, local, ppc, dcp),
                          g);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dispatcher-level tests (dcp=2, null backends)
// ═══════════════════════════════════════════════════════════════════════════

namespace {

constexpr uint32_t kTestSlots = 128;
constexpr int kKvLayers = 6;    // num_hidden_layers below
constexpr int kPage = 16;       // page_size_tokens == dcp_chunk_size
constexpr int kMaxBatch = 8;

void* aligned_alloc_zeroed(size_t bytes) {
    void* p = std::aligned_alloc(64, bytes);
    std::memset(p, 0, bytes);
    return p;
}

lipc::Command make_cmd(lipc::CmdType type, uint32_t seq = 0) {
    lipc::Command cmd{};
    cmd.cmd_type = static_cast<uint32_t>(type);
    cmd.cmd_seq  = seq;
    return cmd;
}

lc::Config shard_test_config(const char* dcp_kv_mode) {
    auto j = nlohmann::json{
        {"model", {
            {"architecture",            "deepseek_v3"},
            {"weights_path",            "/data/models/test/"},
            {"weights_format",          "safetensors"},
            {"num_hidden_layers",       kKvLayers},
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
        {"hardware", {
            {"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 1}},
                      {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 1}}}},
            {"tp_array", {0, 1}},
            {"system_ram_gb", 64},
            {"dcp_kv_mode", dcp_kv_mode}}},
        {"memory", {{"vram_safety_margin_gb", 0.1},
                    {"kv_cache", {{"page_growth_chunk_tokens", 0},
                                  {"dcp_chunk_size", 16},
                                  {"page_size_tokens", 16}}}}},
    };
    return lc::parse_config(j);
}

class KvShardMetadataTest : public ::testing::Test {
protected:
    void SetUp() override {
        ipc_bytes_ = lipc::IpcLayout::total_size(kTestSlots, kTestSlots);
        ipc_region_ = static_cast<uint8_t*>(aligned_alloc_zeroed(ipc_bytes_));
        void* cmp_ptr =
            ipc_region_ + lipc::IpcLayout::cmp_ring_offset(kTestSlots);
        lipc::CompletionRing::init(cmp_ptr, kTestSlots);
        cmp_ring_ = std::make_unique<lipc::CompletionRing>(cmp_ptr);
        sideband_base_ =
            ipc_region_ + lipc::IpcLayout::sideband_offset(kTestSlots,
                                                           kTestSlots);
    }

    void TearDown() override {
        dispatcher_.reset();
        dcp_executor_.reset();
        attn_devices_.clear();
        std::free(hidden_buf0_);
        std::free(hidden_buf1_);
        std::free(ipc_region_);
    }

    /// Build a full dcp=2 stack with the given dcp_kv_mode.
    void make_stack(const char* dcp_kv_mode) {
        cfg_ = std::make_unique<lc::Config>(shard_test_config(dcp_kv_mode));
        mcfg_ = std::make_unique<lmod::ModelConfig>(*cfg_);
        fp8_ = std::make_unique<lmod::Fp8E4M3>();
        layer_reg_ = std::make_unique<lmod::LayerRegistry>(*mcfg_, *cfg_,
                                                           *fp8_);

        lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
        lc::GpuRef gpu1{1, 1, lc::GpuType::rtx5090};
        device_backends_.push_back(lcomp::make_null_device_backend(gpu0));
        device_backends_.push_back(lcomp::make_null_device_backend(gpu1));
        std::vector<lcomp::DeviceBackend*> dev_ptrs{
            device_backends_[0].get(), device_backends_[1].get()};

        auto layout = lmem::compute_vram_layout(*cfg_, *layer_reg_, *mcfg_);
        vram_ = std::make_unique<lmem::VramAllocator>(std::move(layout),
                                                      dev_ptrs);
        page_allocator_ = std::make_unique<lmem::PageAllocator>(
            *vram_, device_backends_[0].get());
        // Engine-equivalent DCP routing config (TP >= 2).
        page_allocator_->set_dcp_config(lmem::DcpConfig{
            .dcp_size = 2,
            .dcp_chunk_size = 16,
            .page_size_tokens = kPage,
            .tp_gpu_indices = {0, 1},
            .indexer_k_sharded = false,
            .kv_sharded = (std::string(dcp_kv_mode) == "sharded"),
        });

        attn_devices_.push_back(lcomp::make_null_attention_device(gpu0));
        attn_devices_.push_back(lcomp::make_null_attention_device(gpu1));

        lcomp::StreamManager::Options sm_opts{.device_backends = dev_ptrs};
        stream_manager_ =
            std::make_unique<lcomp::StreamManager>(std::move(sm_opts));
        graph_registry_ = std::make_unique<lcomp::GraphRegistry>();

        lpar::DcpExecutor::Options dcp_opts{
            .dcp_size            = 2,
            .gpus                = {gpu0, gpu1},
            .max_batch_size      = kMaxBatch,
            .hidden_size         = 256,
            .num_attention_heads = 4,
            .q_lora_rank         = 64,
            .kv_lora_rank        = 32,
            .qk_rope_head_dim   = 16,
            .qk_nope_head_dim   = 32,
            .v_head_dim          = 64,
            .rms_norm_eps        = 1e-6f,
            .stream_manager      = stream_manager_.get(),
            .attention_devices   = {attn_devices_[0].get(),
                                    attn_devices_[1].get()},
        };
        dcp_executor_ =
            std::make_unique<lpar::DcpExecutor>(std::move(dcp_opts));

        fake_weights_ =
            std::vector<std::vector<lpar::AttentionLayerWeights>>(
                kKvLayers, std::vector<lpar::AttentionLayerWeights>(2));
        {
            std::vector<std::vector<const lpar::AttentionLayerWeights*>> wp(
                kKvLayers);
            for (int l = 0; l < kKvLayers; ++l)
                wp[l] = {&fake_weights_[l][0], &fake_weights_[l][1]};
            dcp_executor_->set_layer_weights(std::move(wp), kKvLayers);
        }

        const size_t hidden_bytes = kMaxBatch * 256 * 2;
        hidden_buf0_ = aligned_alloc_zeroed(hidden_bytes);
        hidden_buf1_ = aligned_alloc_zeroed(hidden_bytes);

        ldam::CommandDispatcher::Deps deps{
            .cmp_ring              = cmp_ring_.get(),
            .stream_manager        = stream_manager_.get(),
            .graph_registry        = graph_registry_.get(),
            .dcp_executor          = dcp_executor_.get(),
            .page_allocator        = page_allocator_.get(),
            .sideband_base         = sideband_base_,
            .live_config           = cfg_.get(),
            .attention_devices     = {attn_devices_[0].get(),
                                      attn_devices_[1].get()},
            .device_backends       = dev_ptrs,
            .cuda_kernels_enabled  = false,
            .kv_cache_stride_block = 576 * kPage,
            .kv_cache_stride_row   = 576,
            .kv_page_size          = kPage,
            .hidden_state_pairs    =
                {ldam::HiddenStatePair{hidden_buf0_, nullptr, 0, 0, 0},
                 ldam::HiddenStatePair{hidden_buf1_, nullptr, 1, 1, 1}},
            .per_layer_attn_weights = fake_weights_,
            .max_batch_size        = kMaxBatch,
        };
        dispatcher_ =
            std::make_unique<ldam::CommandDispatcher>(std::move(deps));
    }

    /// CMD_SEQ_CREATE and assert success.  pool: 0=kMain, 1=kSpeculation.
    void create_seq(uint64_t seq_id, uint32_t prompt_len, uint32_t seq = 10,
                    uint8_t pool = 0) {
        auto cmd = make_cmd(lipc::CMD_SEQ_CREATE, seq);
        cmd.seq_create.seq_id     = seq_id;
        cmd.seq_create.prompt_len = prompt_len;
        cmd.seq_create.pool       = pool;
        dispatcher_->dispatch(cmd);
        lipc::Completion cmp{};
        ASSERT_TRUE(cmp_ring_->try_read(&cmp));
        ASSERT_EQ(cmp.status, 0u) << "seq_create failed";
    }

    /// Write batch descriptors and dispatch RUN_ATTENTION; assert success.
    void run_attention(
        const std::vector<std::pair<uint64_t, uint32_t>>& rows,
        uint32_t seq, bool is_prefill = false) {
        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_base_ + lipc::IpcLayout::kBatchDescriptorOff);
        for (size_t b = 0; b < rows.size(); ++b)
            batch[b] = {rows[b].first, rows[b].second, 0};

        auto cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION, seq);
        cmd.run_attention.layer_idx  = 0;
        cmd.run_attention.num_seqs   = static_cast<uint32_t>(rows.size());
        cmd.run_attention.is_prefill = is_prefill ? 1 : 0;
        dispatcher_->dispatch(cmd);
        dispatcher_->poll_compute_completions();
        lipc::Completion cmp{};
        ASSERT_TRUE(cmp_ring_->try_read(&cmp));
        ASSERT_EQ(cmp.cmp_type,
                  static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE))
            << "attention dispatch failed";
    }

    uint8_t* ipc_region_ = nullptr;
    size_t ipc_bytes_ = 0;
    uint8_t* sideband_base_ = nullptr;
    std::unique_ptr<lipc::CompletionRing> cmp_ring_;

    std::unique_ptr<lc::Config> cfg_;
    std::unique_ptr<lmod::ModelConfig> mcfg_;
    std::unique_ptr<lmod::Fp8E4M3> fp8_;
    std::unique_ptr<lmod::LayerRegistry> layer_reg_;
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> device_backends_;
    std::unique_ptr<lmem::VramAllocator> vram_;
    std::unique_ptr<lmem::PageAllocator> page_allocator_;
    std::vector<std::unique_ptr<lcomp::AttentionDevice>> attn_devices_;
    std::unique_ptr<lcomp::StreamManager> stream_manager_;
    std::unique_ptr<lcomp::GraphRegistry> graph_registry_;
    std::unique_ptr<lpar::DcpExecutor> dcp_executor_;
    std::vector<std::vector<lpar::AttentionLayerWeights>> fake_weights_;
    void* hidden_buf0_ = nullptr;
    void* hidden_buf1_ = nullptr;
    std::unique_ptr<ldam::CommandDispatcher> dispatcher_;
};

}  // namespace

// Decode at pos 20 (owner = rank 1): per-rank divergent lengths, owner-only
// real slot, trash slot on the non-owner, block tables reference only local
// pages, global seqlens carry the position.
TEST_F(KvShardMetadataTest, Sharded_Decode_DivergentPerRankMetadata) {
    make_stack("sharded");
    create_seq(9000, 48);  // 3 logical pages: g0→rank0, g1→rank1, g2→rank0

    run_attention({{9000, 20}}, 100);  // N=21, owner(20)=rank1

    auto v0 = dispatcher_->kv_meta_host_view(0);
    auto v1 = dispatcher_->kv_meta_host_view(1);
    ASSERT_NE(v0.seqlens_k, nullptr);
    ASSERT_NE(v1.seqlens_k, nullptr);

    // Local shard lengths: rank0 owns [0,16) → 16; rank1 owns [16,21) → 5.
    EXPECT_EQ(v0.seqlens_k[0], 16);
    EXPECT_EQ(v1.seqlens_k[0], 5);
    // Disjoint + complete (INV-4.9e).
    EXPECT_EQ(v0.seqlens_k[0] + v1.seqlens_k[0], 21);

    // Global lengths present and identical.
    ASSERT_NE(v0.global_seqlens, nullptr);
    ASSERT_NE(v1.global_seqlens, nullptr);
    EXPECT_EQ(v0.global_seqlens[0], 21);
    EXPECT_EQ(v1.global_seqlens[0], 21);

    // Trash slots exist and differ per rank pool.
    ASSERT_GE(v0.trash_slot_base, 0);
    ASSERT_GE(v1.trash_slot_base, 0);

    const int B = 1;
    for (int l = 0; l < kKvLayers; ++l) {
        // Owner (rank1): real slot = its local page 0's physical page.
        const int bt1_0 = v1.block_tables[(l * B + 0)
                                          * v1.max_blocks_per_seq + 0];
        const int slot1 = v1.slot_mappings[l * B + 0];
        EXPECT_EQ(slot1, bt1_0 * kPage + (20 % kPage)) << "layer " << l;
        EXPECT_NE(slot1 / kPage, v1.trash_slot_base / kPage);

        // Non-owner (rank0): slot points at ITS trash page.
        const int slot0 = v0.slot_mappings[l * B + 0];
        EXPECT_EQ(slot0 / kPage, v0.trash_slot_base / kPage) << "layer " << l;

        // No block table entry may reference the rank's trash page.
        for (int j = 0; j * kPage < v0.seqlens_k[0]; ++j)
            EXPECT_NE(v0.block_tables[(l * B) * v0.max_blocks_per_seq + j],
                      v0.trash_slot_base / kPage);
        for (int j = 0; j * kPage < v1.seqlens_k[0]; ++j)
            EXPECT_NE(v1.block_tables[(l * B) * v1.max_blocks_per_seq + j],
                      v1.trash_slot_base / kPage);
    }
}

// Owner alternates with position: pos 15 → rank0 appends, pos 16 → rank1.
TEST_F(KvShardMetadataTest, Sharded_KAppend_OwnerAlternatesByChunk) {
    make_stack("sharded");
    create_seq(9001, 48);

    run_attention({{9001, 15}}, 101);  // owner(15) = rank0
    {
        auto v0 = dispatcher_->kv_meta_host_view(0);
        auto v1 = dispatcher_->kv_meta_host_view(1);
        EXPECT_EQ(v0.seqlens_k[0], 16);
        EXPECT_EQ(v1.seqlens_k[0], 0);   // rank1 owns nothing yet
        // rank0 slot is real; rank1 goes to trash.
        const int bt0_0 = v0.block_tables[0];
        EXPECT_EQ(v0.slot_mappings[0], bt0_0 * kPage + 15);
        EXPECT_EQ(v1.slot_mappings[0] / kPage, v1.trash_slot_base / kPage);
    }

    run_attention({{9001, 16}}, 102);  // owner(16) = rank1
    {
        auto v0 = dispatcher_->kv_meta_host_view(0);
        auto v1 = dispatcher_->kv_meta_host_view(1);
        EXPECT_EQ(v0.seqlens_k[0], 16);
        EXPECT_EQ(v1.seqlens_k[0], 1);
        const int bt1_0 = v1.block_tables[0];
        EXPECT_EQ(v1.slot_mappings[0], bt1_0 * kPage + 0);
        EXPECT_EQ(v0.slot_mappings[0] / kPage, v0.trash_slot_base / kPage);
    }
}

// Multi-row prefill chunk spanning the chunk boundary: per-row owners split
// between ranks; per-rank per-row local causal lengths are correct.
TEST_F(KvShardMetadataTest, Sharded_PrefillChunk_PerRowOwnership) {
    make_stack("sharded");
    create_seq(9002, 48);

    // Rows at positions 14, 15, 16 (one sequence, consecutive).
    run_attention({{9002, 14}, {9002, 15}, {9002, 16}}, 103,
                  /*is_prefill=*/true);

    auto v0 = dispatcher_->kv_meta_host_view(0);
    auto v1 = dispatcher_->kv_meta_host_view(1);

    // Rank-local lengths per row: owned_len(r, pos+1).
    EXPECT_EQ(v0.seqlens_k[0], 15);  // rank0: [0,15)
    EXPECT_EQ(v0.seqlens_k[1], 16);
    EXPECT_EQ(v0.seqlens_k[2], 16);
    EXPECT_EQ(v1.seqlens_k[0], 0);
    EXPECT_EQ(v1.seqlens_k[1], 0);
    EXPECT_EQ(v1.seqlens_k[2], 1);   // rank1 owns pos 16

    // Global lengths ascend per row.
    ASSERT_NE(v0.global_seqlens, nullptr);
    EXPECT_EQ(v0.global_seqlens[0], 15);
    EXPECT_EQ(v0.global_seqlens[1], 16);
    EXPECT_EQ(v0.global_seqlens[2], 17);

    // Row owners: 14→r0, 15→r0, 16→r1. Check slots on layer 0, B=3.
    const int B = 3;
    // rank0: rows 0,1 real; row 2 trash.
    EXPECT_EQ(v0.slot_mappings[0] % kPage, 14);
    EXPECT_EQ(v0.slot_mappings[1] % kPage, 15);
    EXPECT_EQ(v0.slot_mappings[2] / kPage, v0.trash_slot_base / kPage);
    // rank1: rows 0,1 trash; row 2 real (its local page 0, row 0).
    EXPECT_EQ(v1.slot_mappings[0] / kPage, v1.trash_slot_base / kPage);
    EXPECT_EQ(v1.slot_mappings[1] / kPage, v1.trash_slot_base / kPage);
    const int bt1_row2 = v1.block_tables[(0 * B + 2)
                                         * v1.max_blocks_per_seq + 0];
    EXPECT_EQ(v1.slot_mappings[2], bt1_row2 * kPage + 0);
}

// Auto-growth past the pre-allocated pages keeps routing owner-correct.
TEST_F(KvShardMetadataTest, Sharded_AutoGrowth_RoutesToOwner) {
    make_stack("sharded");
    create_seq(9003, 16);  // 1 logical page pre-allocated

    // Decode at pos 40 → global page 2 (rank0-owned, local page 1); pages
    // for g1 (rank1) and g2 (rank0) must be auto-grown on their owners.
    run_attention({{9003, 40}}, 104);

    auto v0 = dispatcher_->kv_meta_host_view(0);
    auto v1 = dispatcher_->kv_meta_host_view(1);
    EXPECT_EQ(v0.seqlens_k[0], 25);  // rank0: [0,16) + [32,41) = 16+9
    EXPECT_EQ(v1.seqlens_k[0], 16);  // rank1: [16,32)
    EXPECT_EQ(v0.seqlens_k[0] + v1.seqlens_k[0], 41);

    // Owner slot: rank0 local page 1 (global page 2), row 40%16=8.
    const int bt0_1 = v0.block_tables[1];
    EXPECT_EQ(v0.slot_mappings[0], bt0_1 * kPage + 8);
    EXPECT_EQ(v1.slot_mappings[0] / kPage, v1.trash_slot_base / kPage);
}

// TD-PREFILL-NONDET regression (root cause, fixed 2026-08-02): a chunked
// prefill advances token_pos WITHOUT page growth (pages preallocated at
// creation), and the bt_dirty guard deliberately has no token_pos term
// (TD-PREFILL-KVMETA-BT-SPLIT) — so each row's block table must cover the
// FULL allocation at the first build. The pre-fix sharded builder truncated
// coverage at the row's CURRENT owned length: the next chunk reused the
// stale table, linearized its ZEROED entries into physical page 0, and the
// staged ctx KV became wrong AND run-to-run nondeterministic (the chunk-2
// first-router-gate fork). The decisive consistency: the READ-side block
// table and the WRITE-side slot mapping must reference the SAME physical
// page for the same token.
TEST_F(KvShardMetadataTest,
       Sharded_ChunkedPrefill_TableCoversAllocationAcrossChunks) {
    make_stack("sharded");
    create_seq(9004, 64);  // 4 logical pages preallocated (g0..g3)

    // Chunk 1: rows 0..7 (prefill) — first build stages the tables.
    run_attention({{9004, 0}, {9004, 1}, {9004, 2}, {9004, 3},
                   {9004, 4}, {9004, 5}, {9004, 6}, {9004, 7}},
                  105, /*is_prefill=*/true);
    const int B = 8;
    auto v0 = dispatcher_->kv_meta_host_view(0);
    // Full-coverage ground truth from the fresh build: rank0 owns global
    // pages 0 (local 0) and 2 (local 1) of the 4-page allocation.
    const int r0_lp1_fresh =
        v0.block_tables[(0 * B + 7) * v0.max_blocks_per_seq + 1];

    // Chunk 2: rows 32..39 — same batch size / seq / page epoch → the
    // block tables are intentionally NOT rebuilt.
    run_attention({{9004, 32}, {9004, 33}, {9004, 34}, {9004, 35},
                   {9004, 36}, {9004, 37}, {9004, 38}, {9004, 39}},
                  106, /*is_prefill=*/true);
    v0 = dispatcher_->kv_meta_host_view(0);
    auto v1 = dispatcher_->kv_meta_host_view(1);

    // Row 7 = token 39: rank0 local len = [0,16) + [32,40) = 24 → its
    // union staging needs local pages 0 AND 1; rank1 holds [16,32) = 16.
    EXPECT_EQ(v0.seqlens_k[7], 24);
    EXPECT_EQ(v1.seqlens_k[7], 16);

    // The (not-rebuilt) table must still cover local page 1 — and agree
    // with the always-fresh append slot for token 39 (owner rank0, global
    // page 2 = local page 1, offset 39 % 16 = 7).
    const int bt_lp1 =
        v0.block_tables[(0 * B + 7) * v0.max_blocks_per_seq + 1];
    EXPECT_EQ(bt_lp1, r0_lp1_fresh)
        << "stale block table lost local-page-1 coverage across chunks";
    EXPECT_EQ(v0.slot_mappings[7], bt_lp1 * kPage + 7)
        << "read-side table and write-side slot disagree on token 39's "
           "physical page (pre-fix: bt entry zeroed -> phys page 0)";

    // Beyond the allocation the row's table stays zeroed (loop exits at
    // num_logical, it does not error): rank0 local page 2 would be global
    // page 4 >= 4.
    EXPECT_EQ(v0.block_tables[(0 * B + 7) * v0.max_blocks_per_seq + 2], 0);
}

// Replicated regression (validated default): every rank's rows are
// IDENTICAL, seqlens are global, slots derive from the (single) logical
// page — and no sharded extras leak in.
TEST_F(KvShardMetadataTest, Replicated_RankRowsIdentical) {
    make_stack("replicated");
    create_seq(9004, 48);

    run_attention({{9004, 20}}, 105);

    auto v0 = dispatcher_->kv_meta_host_view(0);
    auto v1 = dispatcher_->kv_meta_host_view(1);
    ASSERT_NE(v0.seqlens_k, nullptr);

    // Global (replicated) lengths on both ranks.
    EXPECT_EQ(v0.seqlens_k[0], 21);
    EXPECT_EQ(v1.seqlens_k[0], 21);
    // No sharded extras.
    EXPECT_EQ(v0.global_seqlens, nullptr);
    EXPECT_EQ(v0.trash_slot_base, -1);

    // Rank rows byte-identical (the KD-4f-d.1b replication contract).
    const int B = 1;
    for (int l = 0; l < kKvLayers; ++l) {
        EXPECT_EQ(v0.slot_mappings[l * B], v1.slot_mappings[l * B]);
        for (int j = 0; j < 3; ++j)
            EXPECT_EQ(v0.block_tables[(l * B) * v0.max_blocks_per_seq + j],
                      v1.block_tables[(l * B) * v1.max_blocks_per_seq + j])
                << "l=" << l << " j=" << j;
        // Slot self-consistency: logical page of pos 20 is page 1.
        const int bt_1 = v0.block_tables[(l * B) * v0.max_blocks_per_seq + 1];
        EXPECT_EQ(v0.slot_mappings[l * B], bt_1 * kPage + 4);
    }
}

// TD-KV-REPLICATED-PAGE-ALIAS: under replicated KV every rank's block table
// must reference a DISTINCT physical page per logical page.  The pre-fix
// owner-routed allocation popped identical indices from the two GPUs'
// independent free lists, so a 48-token sequence produced bt=[0,0,6] —
// tokens 0-15 and 16-31 aliased physical page 0 on EVERY rank and the later
// chunk's k_append clobbered the earlier chunk's KV (INV-KV-REP).
TEST_F(KvShardMetadataTest, Replicated_BlockTable_NoPhysicalPageAliasing) {
    make_stack("replicated");
    create_seq(9006, 48);  // 3 logical pages

    run_attention({{9006, 40}}, 107);  // covers all 3 logical pages

    const int B = 1;
    for (int rank = 0; rank < 2; ++rank) {
        auto v = dispatcher_->kv_meta_host_view(rank);
        ASSERT_NE(v.block_tables, nullptr);
        for (int l = 0; l < kKvLayers; ++l) {
            const int* bt = v.block_tables
                          + (static_cast<size_t>(l) * B) * v.max_blocks_per_seq;
            std::set<int> distinct(bt, bt + 3);
            EXPECT_EQ(distinct.size(), 3u)
                << "rank " << rank << " layer " << l
                << " block table aliases physical pages: ["
                << bt[0] << ", " << bt[1] << ", " << bt[2] << "]";
        }
    }
}

// INV-KV-REP: replicated allocation must consume a page on EVERY TP GPU per
// (logical page, layer) — each rank holds the full KV — and seq_free must
// drain both pools back completely.
TEST_F(KvShardMetadataTest, Replicated_AllocationMirroredAcrossRanks) {
    make_stack("replicated");

    const int free0_before = page_allocator_->free_pages(0, lmem::Pool::kMain);
    const int free1_before = page_allocator_->free_pages(1, lmem::Pool::kMain);

    create_seq(9007, 48);
    const int used0 = free0_before
                    - page_allocator_->free_pages(0, lmem::Pool::kMain);
    const int used1 = free1_before
                    - page_allocator_->free_pages(1, lmem::Pool::kMain);
    EXPECT_EQ(used0, used1) << "replicated alloc must mirror on all TP GPUs";
    EXPECT_GT(used0, 0);

    auto cmd = make_cmd(lipc::CMD_SEQ_FREE, 108);
    cmd.seq_free.seq_id = 9007;
    dispatcher_->dispatch(cmd);
    lipc::Completion cmp{};
    ASSERT_TRUE(cmp_ring_->try_read(&cmp));
    ASSERT_EQ(cmp.status, 0u);

    EXPECT_EQ(page_allocator_->free_pages(0, lmem::Pool::kMain), free0_before);
    EXPECT_EQ(page_allocator_->free_pages(1, lmem::Pool::kMain), free1_before);
}

// TD-KV-REPLICATED-SPEC (INV-KV-REP extended to kSpeculation): a DRAFT
// (speculation-pool) sequence's replicated block tables must reference a
// DISTINCT physical page per logical page on every rank — pre-fix the spec
// pages were allocated on ONE GPU while build_kv_metadata replicated their
// page_idx to all ranks, the same aliasing family as the kMain HIGH bug.
TEST_F(KvShardMetadataTest, Replicated_SpecPool_BlockTable_NoPhysicalPageAliasing) {
    make_stack("replicated");
    create_seq(9010, 48, 10, /*pool=*/1);  // 3 logical pages, kSpeculation

    run_attention({{9010, 40}}, 110);  // covers all 3 logical pages

    const int B = 1;
    for (int rank = 0; rank < 2; ++rank) {
        auto v = dispatcher_->kv_meta_host_view(rank);
        ASSERT_NE(v.block_tables, nullptr);
        for (int l = 0; l < kKvLayers; ++l) {
            const int* bt = v.block_tables
                          + (static_cast<size_t>(l) * B) * v.max_blocks_per_seq;
            std::set<int> distinct(bt, bt + 3);
            EXPECT_EQ(distinct.size(), 3u)
                << "rank " << rank << " layer " << l
                << " spec-pool block table aliases physical pages: ["
                << bt[0] << ", " << bt[1] << ", " << bt[2] << "]";
        }
    }
    // Rank rows identical (replication contract holds for the spec pool).
    auto v0 = dispatcher_->kv_meta_host_view(0);
    auto v1 = dispatcher_->kv_meta_host_view(1);
    for (int l = 0; l < kKvLayers; ++l) {
        for (int j = 0; j < 3; ++j)
            EXPECT_EQ(v0.block_tables[(l * B) * v0.max_blocks_per_seq + j],
                      v1.block_tables[(l * B) * v1.max_blocks_per_seq + j])
                << "l=" << l << " j=" << j;
        EXPECT_EQ(v0.slot_mappings[l * B], v1.slot_mappings[l * B]);
    }
}

// INV-KV-REP × kSpeculation: replicated draft allocation consumes a spec
// page on EVERY TP GPU per (logical page, layer) — and seq_free drains both
// spec pools back completely (mirrored free).
TEST_F(KvShardMetadataTest, Replicated_SpecPool_AllocationMirroredAcrossRanks) {
    make_stack("replicated");

    const int spec0_before =
        page_allocator_->free_pages(0, lmem::Pool::kSpeculation);
    const int spec1_before =
        page_allocator_->free_pages(1, lmem::Pool::kSpeculation);
    const int main0_before = page_allocator_->free_pages(0, lmem::Pool::kMain);
    const int main1_before = page_allocator_->free_pages(1, lmem::Pool::kMain);
    ASSERT_GT(spec0_before, 0);

    create_seq(9011, 48, 10, /*pool=*/1);
    const int used0 = spec0_before
                    - page_allocator_->free_pages(0, lmem::Pool::kSpeculation);
    const int used1 = spec1_before
                    - page_allocator_->free_pages(1, lmem::Pool::kSpeculation);
    EXPECT_EQ(used0, used1)
        << "replicated spec alloc must mirror on all TP GPUs";
    EXPECT_GT(used0, 0);
    // Main pools untouched: drafts draw exclusively from the spec pool.
    EXPECT_EQ(page_allocator_->free_pages(0, lmem::Pool::kMain), main0_before);
    EXPECT_EQ(page_allocator_->free_pages(1, lmem::Pool::kMain), main1_before);

    auto cmd = make_cmd(lipc::CMD_SEQ_FREE, 111);
    cmd.seq_free.seq_id = 9011;
    dispatcher_->dispatch(cmd);
    lipc::Completion cmp{};
    ASSERT_TRUE(cmp_ring_->try_read(&cmp));
    ASSERT_EQ(cmp.status, 0u);

    EXPECT_EQ(page_allocator_->free_pages(0, lmem::Pool::kSpeculation),
              spec0_before);
    EXPECT_EQ(page_allocator_->free_pages(1, lmem::Pool::kSpeculation),
              spec1_before);
}

// SHARDED KV keeps the pre-existing owner-only single-GPU spec allocation
// (the lockstep-replicated claim applies ONLY under replicated metadata).
TEST_F(KvShardMetadataTest, Sharded_SpecPool_SingleGpuOwnerOnly) {
    make_stack("sharded");

    const int spec0_before =
        page_allocator_->free_pages(0, lmem::Pool::kSpeculation);
    const int spec1_before =
        page_allocator_->free_pages(1, lmem::Pool::kSpeculation);
    ASSERT_GT(spec0_before, 0);

    create_seq(9012, 48, 10, /*pool=*/1);  // cmd.gpu_idx = 0
    EXPECT_LT(page_allocator_->free_pages(0, lmem::Pool::kSpeculation),
              spec0_before)
        << "spec pages must come from the command's GPU";
    EXPECT_EQ(page_allocator_->free_pages(1, lmem::Pool::kSpeculation),
              spec1_before)
        << "sharded spec allocation must NOT touch the other GPU";
}

// seq_free returns all shard pages (both GPUs' pools drain back).
TEST_F(KvShardMetadataTest, Sharded_SeqFree_ReturnsAllShardPages) {
    make_stack("sharded");

    const int free0_before = page_allocator_->free_pages(0, lmem::Pool::kMain);
    const int free1_before = page_allocator_->free_pages(1, lmem::Pool::kMain);

    create_seq(9005, 48);
    EXPECT_LT(page_allocator_->free_pages(0, lmem::Pool::kMain), free0_before);
    EXPECT_LT(page_allocator_->free_pages(1, lmem::Pool::kMain), free1_before);

    auto cmd = make_cmd(lipc::CMD_SEQ_FREE, 106);
    cmd.seq_free.seq_id = 9005;
    dispatcher_->dispatch(cmd);
    lipc::Completion cmp{};
    ASSERT_TRUE(cmp_ring_->try_read(&cmp));
    ASSERT_EQ(cmp.status, 0u);

    EXPECT_EQ(page_allocator_->free_pages(0, lmem::Pool::kMain), free0_before);
    EXPECT_EQ(page_allocator_->free_pages(1, lmem::Pool::kMain), free1_before);
}

