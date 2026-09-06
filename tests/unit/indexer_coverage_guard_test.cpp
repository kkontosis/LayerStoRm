// TD-GLM-INDEXER-PREFILL / TD-GLM-INDEXER-COV: coverage-guard state machine.
//
// Verifies the dispatcher's per-sequence indexer-K coverage bookkeeping
// (CommandDispatcher::indexer_coverage introspection) across the step kinds:
//   - a prefill / chunked-prefill step ADVANCES coverage by chunk_len (the
//     chunk appender path, TD-GLM-INDEXER-PREFILL) instead of killing the
//     sequence;
//   - a subsequent contiguous decode step keeps the sequence sparse-eligible
//     (mode != dead) and advances next_pos by 1;
//   - re-dispatching the same step for a later layer (repeat) neither
//     re-advances nor kills;
//   - a position GAP permanently downgrades the sequence to dense (kDead);
//   - a malformed prefill shape (mixed sequences / non-consecutive
//     positions — a broken command stream no live producer can express) is
//     REFUSED with kComputeValidation, every involved sequence's coverage
//     untouched (TD-KVT-BATCH-COHORT indexer rider: healthy sequences must
//     not die for a neighbour's fault, and no dense step may be served off
//     a malformed shape).
//
// CPU-only: null device backends, null attention device, real PageAllocator +
// KV metadata path (the guard is gated on kv_meta_ok).

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "compute/graphs/graph_registry.h"
#include "compute/stream_manager.h"
#include "config/config_parser.h"
#include "core/gpu_ref.h"
#include "core/memory/expert_cache.h"
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

// Coverage mode codes (CommandDispatcher::indexer_coverage contract).
constexpr int kModeUnknown = -1;
constexpr int kModeDead = 2;  // IndexerSeqMode::kDead (S4: kArena deleted)

// CmpErrorCategory (ipc_protocol.h) — the malformed-shape refusal class.
constexpr uint32_t kErrCatComputeValidation = 10;

void* aligned_alloc_zeroed(size_t bytes) {
    void* p = std::aligned_alloc(64, bytes);
    std::memset(p, 0, bytes);
    return p;
}

/// Small DSA-enabled config (index_topk > 0 arms the coverage guard).
lc::Config dsa_config() {
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
            {"system_ram_gb", 64}}},
        {"serving", {{"max_sequence_length", 1024}}},
        {"memory", {{"vram_safety_margin_gb", 0.1},
                    {"kv_cache", {{"page_growth_chunk_tokens", 64}}}}},
    };
    return lc::parse_config(j);
}

}  // namespace

class IndexerCoverageGuardTest : public ::testing::Test {
protected:
    /// Subclass hook: adjust the parsed config before the dispatcher (and
    /// everything downstream of the config) is built.
    virtual void mutate_config(lc::Config&) {}

    void SetUp() override {
        ipc_bytes_ = lipc::IpcLayout::total_size(kTestSlots, kTestSlots);
        ipc_region_ = static_cast<uint8_t*>(aligned_alloc_zeroed(ipc_bytes_));
        void* cmp_ptr = ipc_region_
                      + lipc::IpcLayout::cmp_ring_offset(kTestSlots);
        lipc::CompletionRing::init(cmp_ptr, kTestSlots);
        cmp_ring_ = std::make_unique<lipc::CompletionRing>(cmp_ptr);
        sideband_ = ipc_region_
                  + lipc::IpcLayout::sideband_offset(kTestSlots, kTestSlots);

        cfg_ = std::make_unique<lc::Config>(dsa_config());
        mutate_config(*cfg_);  // subclass hook (e.g. dsa_indexer_rewind)
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

        lcomp::StreamManager::Options sm_opts{
            .device_backends = {backends_[0].get(), backends_[1].get()},
        };
        stream_manager_ = std::make_unique<lcomp::StreamManager>(
            std::move(sm_opts));

        attn_device_ = lcomp::make_null_attention_device(gpu0);
        lpar::DcpExecutor::Options dcp_opts{
            .dcp_size            = 1,
            .gpus                = {gpu0},
            .max_batch_size      = 8,
            .num_layers          = 6,
            .hidden_size         = 256,
            .num_attention_heads = 4,
            .q_lora_rank         = 64,
            .kv_lora_rank        = 32,
            .qk_rope_head_dim    = 32,
            .qk_nope_head_dim    = 32,
            .v_head_dim          = 64,
            .rms_norm_eps        = 1e-6f,
            .stream_manager      = stream_manager_.get(),
            .attention_devices   = {attn_device_.get()},
        };
        dcp_executor_ = std::make_unique<lpar::DcpExecutor>(
            std::move(dcp_opts));

        weights_ = std::vector<std::vector<lpar::AttentionLayerWeights>>(
            6, std::vector<lpar::AttentionLayerWeights>(1));
        {
            std::vector<std::vector<const lpar::AttentionLayerWeights*>> wp(6);
            for (int l = 0; l < 6; ++l) wp[l] = {&weights_[l][0]};
            dcp_executor_->set_layer_weights(std::move(wp), 6);
        }

        hidden_buf_ = aligned_alloc_zeroed(8 * 256 * 2);

        ldam::CommandDispatcher::Deps deps{
            .cmp_ring               = cmp_ring_.get(),
            .stream_manager         = stream_manager_.get(),
            .dcp_executor           = dcp_executor_.get(),
            .page_allocator         = page_allocator_.get(),
            .sideband_base          = sideband_,
            .live_config            = cfg_.get(),
            .attention_devices      = {attn_device_.get()},
            .device_backends        = {backends_[0].get()},
            .cuda_kernels_enabled   = false,
            .kv_cache_stride_block  = 576 * 64,
            .kv_cache_stride_row    = 576,
            .kv_page_size           = 64,
            .hidden_state_pairs     = {ldam::HiddenStatePair{
                hidden_buf_, nullptr, 0, 0, 0, nullptr, nullptr}},
            .per_layer_attn_weights = weights_,
            .max_batch_size         = 8,
        };
        dispatcher_ = std::make_unique<ldam::CommandDispatcher>(
            std::move(deps));
    }

    void TearDown() override {
        dispatcher_.reset();
        dcp_executor_.reset();
        attn_device_.reset();
        std::free(hidden_buf_);
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

    /// Like drain_completions but TOLERATES errors and returns the last
    /// completion — for steps the dispatcher is expected to REFUSE.
    lipc::Completion drain_capture() {
        dispatcher_->poll_compute_completions();
        lipc::Completion cmp{}, last{};
        bool any = false;
        while (cmp_ring_->try_read(&cmp)) {
            last = cmp;
            any = true;
        }
        EXPECT_TRUE(any) << "no completion drained";
        return last;
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

    void free_seq(uint64_t seq_id) {
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(lipc::CMD_SEQ_FREE);
        c.cmd_seq = next_seq_++;
        c.seq_free.seq_id = seq_id;
        dispatcher_->dispatch(c);
        drain_completions();
    }

    /// R4a: CMD_SEQ_FORK with an optional truncation prefix (0 = full).
    void fork_seq(uint64_t src, uint64_t dst, uint32_t prefix_len = 0) {
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(lipc::CMD_SEQ_FORK);
        c.cmd_seq = next_seq_++;
        c.seq_fork.src_seq_id = src;
        c.seq_fork.dst_seq_id = dst;
        c.seq_fork.prefix_len = prefix_len;
        dispatcher_->dispatch(c);
        drain_completions();
    }

    /// One attention dispatch over `n` batch entries starting at (seq, pos0)
    /// with consecutive positions (a decode step for n == 1 / prefill == 0,
    /// a prefill chunk otherwise).
    void attention_step(uint64_t seq_id, uint32_t pos0, uint32_t n,
                        bool prefill, uint32_t layer = 0,
                        uint8_t superchunk = 0) {
        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        for (uint32_t b = 0; b < n; ++b) {
            batch[b].seq_id = seq_id;
            batch[b].token_pos = pos0 + b;
            batch[b]._pad = 0;
        }
        dispatch_attention(n, prefill, layer, /*chunk_len=*/prefill ? n : 0,
                           superchunk);
    }

    /// TD-DECODE-GRAPH coverage rider: a B=1 decode step carrying explicit
    /// use_graph / is_draft flags (the two non-appending step kinds).
    void attention_step_flags(uint64_t seq_id, uint32_t pos, uint32_t layer,
                              uint8_t use_graph, uint8_t is_draft) {
        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        batch[0].seq_id = seq_id;
        batch[0].token_pos = pos;
        batch[0]._pad = 0;
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(lipc::D_B_CMD_RUN_ATTENTION);
        c.cmd_seq = next_seq_++;
        c.run_attention.layer_idx = layer;
        c.run_attention.num_seqs = 1;
        c.run_attention.is_prefill = 0;
        c.run_attention.use_graph = use_graph;
        c.run_attention.is_draft = is_draft;
        c.run_attention.chunk_start = 0;
        c.run_attention.chunk_len = 0;
        c.run_attention.superchunk = 0;
        dispatcher_->dispatch(c);
        drain_completions();
    }

    void dispatch_attention(uint32_t num_seqs, bool prefill, uint32_t layer,
                            uint32_t chunk_len, uint8_t superchunk = 0) {
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(lipc::D_B_CMD_RUN_ATTENTION);
        c.cmd_seq = next_seq_++;
        c.run_attention.layer_idx = layer;
        c.run_attention.num_seqs = num_seqs;
        c.run_attention.is_prefill = prefill ? 1 : 0;
        c.run_attention.use_graph = 0;
        c.run_attention.is_draft = 0;
        c.run_attention.chunk_start = 0;
        c.run_attention.chunk_len = prefill ? chunk_len : 0;
        c.run_attention.superchunk = superchunk;
        dispatcher_->dispatch(c);
        drain_completions();
    }

    /// dispatch_attention that expects the step to be refused: returns the
    /// last completion (the CMP_ERROR) instead of asserting none occurred.
    lipc::Completion dispatch_attention_capture(uint32_t num_seqs,
                                                bool prefill, uint32_t layer,
                                                uint32_t chunk_len,
                                                uint8_t superchunk = 0) {
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(lipc::D_B_CMD_RUN_ATTENTION);
        c.cmd_seq = next_seq_++;
        c.run_attention.layer_idx = layer;
        c.run_attention.num_seqs = num_seqs;
        c.run_attention.is_prefill = prefill ? 1 : 0;
        c.run_attention.use_graph = 0;
        c.run_attention.is_draft = 0;
        c.run_attention.chunk_start = 0;
        c.run_attention.chunk_len = prefill ? chunk_len : 0;
        c.run_attention.superchunk = superchunk;
        dispatcher_->dispatch(c);
        return drain_capture();
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
    std::unique_ptr<lcomp::AttentionDevice> attn_device_;
    std::unique_ptr<lpar::DcpExecutor> dcp_executor_;
    std::vector<std::vector<lpar::AttentionLayerWeights>> weights_;
    void* hidden_buf_ = nullptr;
    std::unique_ptr<ldam::CommandDispatcher> dispatcher_;
    uint32_t next_seq_ = 1;
};

// Prefill chunk → contiguous decode: the sequence stays sparse-eligible and
// coverage advances by chunk_len, then by 1 per decode step.
TEST_F(IndexerCoverageGuardTest, PrefillThenDecodeStaysSparseEligible) {
    create_seq(1, 8);

    // Untracked before any step.
    EXPECT_EQ(dispatcher_->indexer_coverage(1).first, kModeUnknown);

    // Prefill chunk: positions 0..3 (layer 0).
    attention_step(1, /*pos0=*/0, /*n=*/4, /*prefill=*/true, /*layer=*/0);
    auto cov = dispatcher_->indexer_coverage(1);
    EXPECT_NE(cov.first, kModeDead) << "prefill chunk must not kill coverage";
    EXPECT_NE(cov.first, kModeUnknown);
    EXPECT_EQ(cov.second, 4u) << "coverage must advance by chunk_len";
    const int pinned_mode = cov.first;

    // Same chunk re-dispatched for a later layer (repeat): no re-advance.
    attention_step(1, 0, 4, true, /*layer=*/1);
    cov = dispatcher_->indexer_coverage(1);
    EXPECT_EQ(cov.first, pinned_mode);
    EXPECT_EQ(cov.second, 4u);

    // Contiguous decode at position 4: still alive, advance to 5.
    attention_step(1, 4, 1, false, 0);
    cov = dispatcher_->indexer_coverage(1);
    EXPECT_EQ(cov.first, pinned_mode) << "storage mode is pinned";
    EXPECT_EQ(cov.second, 5u);

    // Decode repeat (later layer): unchanged.
    attention_step(1, 4, 1, false, 1);
    cov = dispatcher_->indexer_coverage(1);
    EXPECT_EQ(cov.second, 5u);

    // Second prefill chunk contiguous with coverage: 5..6.
    attention_step(1, 5, 2, true, 0);
    cov = dispatcher_->indexer_coverage(1);
    EXPECT_EQ(cov.first, pinned_mode);
    EXPECT_EQ(cov.second, 7u);
}

// A position gap (decode or prefill) permanently downgrades to dense.
TEST_F(IndexerCoverageGuardTest, GapKillsSequence) {
    create_seq(2, 8);

    attention_step(2, 0, 3, true, 0);   // prefill 0..2
    EXPECT_EQ(dispatcher_->indexer_coverage(2).second, 3u);

    attention_step(2, 5, 1, false, 0);  // decode at 5 — gap (3, 4 missing)
    EXPECT_EQ(dispatcher_->indexer_coverage(2).first, kModeDead);

    // Dead is permanent: a later contiguous-looking step cannot revive it.
    attention_step(2, 3, 1, false, 0);
    EXPECT_EQ(dispatcher_->indexer_coverage(2).first, kModeDead);
}

TEST_F(IndexerCoverageGuardTest, PrefillGapKillsSequence) {
    create_seq(3, 8);

    attention_step(3, 0, 2, true, 0);   // prefill 0..1
    EXPECT_EQ(dispatcher_->indexer_coverage(3).second, 2u);

    attention_step(3, 4, 2, true, 0);   // prefill 4..5 — gap (2, 3 missing)
    EXPECT_EQ(dispatcher_->indexer_coverage(3).first, kModeDead);
}

// TD-GLM-INDEXER-B1CASCADE resolved (INV-DSA-ROWMIX): a B>1 decode cohort
// mixing a live and a coverage-dead sequence must NOT cascade — the live row
// keeps appending (coverage advances) and stays sparse-eligible while only
// the dead row runs dense. Pre-fix, the dead row suppressed the whole cohort:
// the live row's key was never appended, its coverage stalled, and the next
// step's gap check killed it permanently.
TEST_F(IndexerCoverageGuardTest, MixedCohortKeepsLiveRowSparseEligible) {
    create_seq(6, 8);   // stays live
    create_seq(7, 8);   // killed by a gap below

    attention_step(6, 0, 2, true, 0);   // live: prefill 0..1 → coverage 2
    attention_step(7, 0, 2, true, 0);   // dead-to-be: prefill 0..1
    attention_step(7, 4, 1, false, 0);  // gap (2, 3 missing) → kDead
    ASSERT_EQ(dispatcher_->indexer_coverage(7).first, kModeDead);
    ASSERT_EQ(dispatcher_->indexer_dense_total(), 1u)
        << "the gap transition is the only witness so far";
    const int live_mode = dispatcher_->indexer_coverage(6).first;
    ASSERT_NE(live_mode, kModeDead);

    // Mixed decode cohort: live row (seq 6 @ pos 2, contiguous) + dead row
    // (seq 7 @ pos 5).
    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {6, 2, 0};
    batch[1] = {7, 5, 0};
    dispatch_attention(/*num_seqs=*/2, /*prefill=*/false, /*layer=*/0,
                       /*chunk_len=*/0);

    auto cov = dispatcher_->indexer_coverage(6);
    EXPECT_EQ(cov.first, live_mode) << "live row's storage mode is pinned";
    EXPECT_EQ(cov.second, 3u)
        << "live row must append + advance in a mixed cohort (no cascade)";
    EXPECT_EQ(dispatcher_->indexer_coverage(7).first, kModeDead);

    // Same cohort re-dispatched for a later layer (repeat): no re-advance,
    // still alive.
    dispatch_attention(2, false, /*layer=*/1, 0);
    cov = dispatcher_->indexer_coverage(6);
    EXPECT_EQ(cov.first, live_mode);
    EXPECT_EQ(cov.second, 3u);

    // The live sequence continues sparse-eligible on the next contiguous
    // decode step (pre-fix it would be permanently dead here).
    attention_step(6, 3, 1, false, 0);
    cov = dispatcher_->indexer_coverage(6);
    EXPECT_EQ(cov.first, live_mode);
    EXPECT_EQ(cov.second, 4u);

    // The dead sequence stays dense forever — solo steps included.
    attention_step(7, 5, 1, false, 0);
    EXPECT_EQ(dispatcher_->indexer_coverage(7).first, kModeDead);

    // The witness fired EXACTLY ONCE (the original gap transition): an
    // already-dead row riding later cohorts must not re-count.
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 1u);
}

// TD-KVT-BATCH-COHORT: a row that gaps INSIDE a mixed decode cohort dies
// ALONE — the witness fires exactly once for the bad row, the healthy row
// keeps appending, stays sparse-eligible, and advances across subsequent
// steps (INV-DSA-ROWMIX applied to the in-cohort transition itself).
TEST_F(IndexerCoverageGuardTest, CohortRowGapWitnessFiresOnceHealthyAdvances) {
    create_seq(8, 8);
    create_seq(9, 8);
    attention_step(8, 0, 2, true, 0);   // both at coverage 2
    attention_step(9, 0, 2, true, 0);
    ASSERT_EQ(dispatcher_->indexer_dense_total(), 0u);

    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {8, 2, 0};   // advancing — healthy
    batch[1] = {9, 5, 0};   // gap (2, 3, 4 missing) — dies, alone
    dispatch_attention(/*num_seqs=*/2, /*prefill=*/false, /*layer=*/0,
                       /*chunk_len=*/0);

    auto cov = dispatcher_->indexer_coverage(8);
    EXPECT_NE(cov.first, kModeDead) << "healthy row must survive";
    EXPECT_EQ(cov.second, 3u) << "healthy row must append + advance";
    EXPECT_EQ(dispatcher_->indexer_coverage(9).first, kModeDead);
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 1u)
        << "exactly one witness for the bad row";

    // Later layer of the same step: no re-count, no re-kill, no re-advance.
    dispatch_attention(2, false, /*layer=*/1, 0);
    EXPECT_EQ(dispatcher_->indexer_coverage(8).second, 3u);
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 1u);

    // The healthy sequence continues sparse on its own.
    attention_step(8, 3, 1, false, 0);
    cov = dispatcher_->indexer_coverage(8);
    EXPECT_NE(cov.first, kModeDead);
    EXPECT_EQ(cov.second, 4u);
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 1u);
}

// TD-KVT-BATCH-COHORT (indexer-coverage rider): a malformed prefill chunk —
// batch descriptors that are not ONE sequence at consecutive ascending
// positions — is a broken command stream (no live descriptor writer can
// express it; they all take (seq_id, pos0, n)), not a real batch
// composition. It must be REFUSED with kComputeValidation and every
// involved sequence's coverage left UNTOUCHED: a healthy sequence must not
// die for its neighbour's fault, no dense step may be served off the
// malformed shape, and the zero-kDead witness must not fire.
TEST_F(IndexerCoverageGuardTest, MalformedPrefillShapeRefusedCoverageIntact) {
    create_seq(4, 8);
    create_seq(5, 8);

    // Establish healthy sparse coverage on both sequences first.
    attention_step(4, 0, 2, true, 0);   // coverage 2
    attention_step(5, 0, 2, true, 0);   // coverage 2
    const int mode4 = dispatcher_->indexer_coverage(4).first;
    const int mode5 = dispatcher_->indexer_coverage(5).first;
    ASSERT_NE(mode4, kModeDead);
    ASSERT_NE(mode5, kModeDead);
    ASSERT_EQ(dispatcher_->indexer_dense_total(), 0u);

    // Mixed-sequence chunk: seq 4 @ 2 and seq 5 @ 2 in ONE prefill step.
    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {4, 2, 0};
    batch[1] = {5, 2, 0};
    const auto cmp = dispatch_attention_capture(
        /*num_seqs=*/2, /*prefill=*/true, /*layer=*/0, /*chunk_len=*/2);
    ASSERT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR))
        << "a malformed prefill shape must be refused, not served dense";
    EXPECT_EQ(cmp.error.error_category, kErrCatComputeValidation);

    // Nobody died, nothing advanced, the dense witness did not fire.
    EXPECT_EQ(dispatcher_->indexer_coverage(4).first, mode4);
    EXPECT_EQ(dispatcher_->indexer_coverage(4).second, 2u);
    EXPECT_EQ(dispatcher_->indexer_coverage(5).first, mode5);
    EXPECT_EQ(dispatcher_->indexer_coverage(5).second, 2u);
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 0u);

    // Both sequences continue SPARSE and advancing on well-formed steps.
    attention_step(4, 2, 1, false, 0);
    attention_step(5, 2, 1, false, 0);
    EXPECT_EQ(dispatcher_->indexer_coverage(4).first, mode4);
    EXPECT_EQ(dispatcher_->indexer_coverage(4).second, 3u);
    EXPECT_EQ(dispatcher_->indexer_coverage(5).first, mode5);
    EXPECT_EQ(dispatcher_->indexer_coverage(5).second, 3u);
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 0u);
}

// Non-consecutive positions of ONE sequence inside a chunk are the same
// malformed class (the appender assumes row b sits at pos0 + b): refused,
// coverage intact, and a well-formed retry of the same window advances.
TEST_F(IndexerCoverageGuardTest, NonConsecutivePrefillShapeRefused) {
    create_seq(6, 8);
    attention_step(6, 0, 2, true, 0);   // coverage 2
    const int mode6 = dispatcher_->indexer_coverage(6).first;
    ASSERT_NE(mode6, kModeDead);

    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {6, 2, 0};
    batch[1] = {6, 4, 0};   // hole at 3 INSIDE one chunk
    const auto cmp = dispatch_attention_capture(
        /*num_seqs=*/2, /*prefill=*/true, /*layer=*/0, /*chunk_len=*/2);
    ASSERT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.error.error_category, kErrCatComputeValidation);
    EXPECT_EQ(dispatcher_->indexer_coverage(6).first, mode6);
    EXPECT_EQ(dispatcher_->indexer_coverage(6).second, 2u);

    // The well-formed retry of the same window proceeds sparse.
    attention_step(6, 2, 2, true, 0);
    EXPECT_EQ(dispatcher_->indexer_coverage(6).first, mode6);
    EXPECT_EQ(dispatcher_->indexer_coverage(6).second, 4u);
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 0u);
}

// INV-DSA-REWIND config-off byte-identity: with compute.dsa_indexer_rewind
// explicitly false (schema default is TRUE since the 2026-08-18 A/B) and no
// env override, the speculative-verify overwrite-rewind shape stays a
// permanent dense downgrade (legacy semantics remain reachable).
class IndexerCoverageGuardRewindConfigOffTest
    : public IndexerCoverageGuardTest {
protected:
    void mutate_config(lc::Config& cfg) override {
        cfg.compute.dsa_indexer_rewind = false;
    }
};

TEST_F(IndexerCoverageGuardRewindConfigOffTest, RewindKillsSequence) {
    create_seq(8, 8);

    attention_step(8, 0, 8, true, 0);   // prefill 0..7 → coverage 8
    ASSERT_EQ(dispatcher_->indexer_coverage(8).second, 8u);

    // Verify-chunk re-feed behind the frontier: [5, 9) — a rewind.
    attention_step(8, 5, 4, true, 0);
    EXPECT_EQ(dispatcher_->indexer_coverage(8).first, kModeDead);
}

// Env precedence: LS_INDEXER_REWIND=0 force-disables even though the
// config carries the schema default TRUE (the LAYERSTORM_DETERMINISTIC_
// EP_COMBINE '0'-wins pattern).
class IndexerCoverageGuardRewindEnvOffTest : public IndexerCoverageGuardTest {
protected:
    void SetUp() override {
        ::setenv("LS_INDEXER_REWIND", "0", 1);
        IndexerCoverageGuardTest::SetUp();  // dispatcher reads the env here
    }
    void TearDown() override {
        IndexerCoverageGuardTest::TearDown();
        ::unsetenv("LS_INDEXER_REWIND");
    }
};

TEST_F(IndexerCoverageGuardRewindEnvOffTest, EnvZeroOverridesConfigTrue) {
    create_seq(8, 8);

    attention_step(8, 0, 8, true, 0);   // prefill 0..7 → coverage 8
    ASSERT_EQ(dispatcher_->indexer_coverage(8).second, 8u);

    attention_step(8, 5, 4, true, 0);   // rewind [5, 9)
    EXPECT_EQ(dispatcher_->indexer_coverage(8).first, kModeDead);
}

// Schema default TRUE (config-first, no env): rewinds are blessed out of
// the box — the dsp52 champion speculative re-feed shape stays sparse.
TEST_F(IndexerCoverageGuardTest, DefaultConfigBlessesRewind) {
    ASSERT_EQ(nullptr, std::getenv("LS_INDEXER_REWIND"));
    create_seq(8, 8);

    attention_step(8, 0, 8, true, 0);   // prefill 0..7 → coverage 8
    const int pinned_mode = dispatcher_->indexer_coverage(8).first;
    ASSERT_NE(pinned_mode, kModeDead);

    attention_step(8, 5, 4, true, 0);   // rewind [5, 9): overwrite, alive
    auto cov = dispatcher_->indexer_coverage(8);
    EXPECT_EQ(cov.first, pinned_mode);
    EXPECT_EQ(cov.second, 9u);
}

// ── INV-DSA-REWIND (LS_INDEXER_REWIND=1): overwrite-rewind blessing ──────
// The dsp52 speculative batched-verify round shape: after partial
// acceptance the next step re-feeds positions BEHIND the coverage frontier
// (same sequence, contiguous). Position-keyed appends overwrite those
// indexer-K rows in place (exactly like the KV rows the same re-feed
// overwrites), so the sequence must stay sparse-eligible; next_pos tracks
// the high-water mark. Gaps / mixed sequences / superchunk shapes must stay
// fail-closed.
class IndexerCoverageGuardRewindTest : public IndexerCoverageGuardTest {
protected:
    void SetUp() override {
        ::setenv("LS_INDEXER_REWIND", "1", 1);
        IndexerCoverageGuardTest::SetUp();  // dispatcher reads the env here
    }
    void TearDown() override {
        IndexerCoverageGuardTest::TearDown();
        ::unsetenv("LS_INDEXER_REWIND");
    }
};

// Decode-step rewind (the overlap-mode plain step after a partially
// accepted chunk): pos < next_pos overwrites in place, no advance, alive.
TEST_F(IndexerCoverageGuardRewindTest, DecodeRewindOverwriteStaysSparse) {
    create_seq(1, 8);

    attention_step(1, 0, 8, true, 0);    // prefill 0..7 → coverage 8
    attention_step(1, 8, 1, false, 0);   // decode @8 → 9
    ASSERT_EQ(dispatcher_->indexer_coverage(1).second, 9u);
    const int pinned_mode = dispatcher_->indexer_coverage(1).first;
    ASSERT_NE(pinned_mode, kModeDead);

    // Rewound decode @6 (< 9): overwrite, frontier stands.
    attention_step(1, 6, 1, false, 0);
    auto cov = dispatcher_->indexer_coverage(1);
    EXPECT_EQ(cov.first, pinned_mode) << "rewind must not kill coverage";
    EXPECT_EQ(cov.second, 9u) << "next_pos is the high-water mark";

    // Same rewound step, later layer: unchanged.
    attention_step(1, 6, 1, false, 1);
    cov = dispatcher_->indexer_coverage(1);
    EXPECT_EQ(cov.first, pinned_mode);
    EXPECT_EQ(cov.second, 9u);

    // Frontier decode @9: advancing again.
    attention_step(1, 9, 1, false, 0);
    cov = dispatcher_->indexer_coverage(1);
    EXPECT_EQ(cov.first, pinned_mode);
    EXPECT_EQ(cov.second, 10u);

    // A genuine gap still kills (@12 > 10).
    attention_step(1, 12, 1, false, 0);
    EXPECT_EQ(dispatcher_->indexer_coverage(1).first, kModeDead);
}

// Chunk rewind (the batched-verify chunk after partial acceptance):
// [pos0, end) with pos0 < next_pos overwrites; end > next_pos extends the
// high-water mark; a later-layer replay and an interior chunk don't move it.
TEST_F(IndexerCoverageGuardRewindTest, ChunkRewindOverwriteExtendsHighWater) {
    create_seq(2, 8);

    attention_step(2, 0, 8, true, 0);    // prefill 0..7 → coverage 8
    const int pinned_mode = dispatcher_->indexer_coverage(2).first;
    ASSERT_NE(pinned_mode, kModeDead);

    // Verify chunk [4, 10): rewind + extension → high water 10.
    attention_step(2, 4, 6, true, 0);
    auto cov = dispatcher_->indexer_coverage(2);
    EXPECT_EQ(cov.first, pinned_mode) << "rewind chunk must stay blessed";
    EXPECT_EQ(cov.second, 10u) << "frontier extends to chunk end";

    // Same chunk, later layer (end == next_pos → legacy repeat): no move.
    attention_step(2, 4, 6, true, 1);
    cov = dispatcher_->indexer_coverage(2);
    EXPECT_EQ(cov.first, pinned_mode);
    EXPECT_EQ(cov.second, 10u);

    // Interior overwrite chunk [6, 9) (end < next_pos, no superchunk flag):
    // valid rewind under the flag, frontier stands.
    attention_step(2, 6, 3, true, 0);
    cov = dispatcher_->indexer_coverage(2);
    EXPECT_EQ(cov.first, pinned_mode);
    EXPECT_EQ(cov.second, 10u);

    // Decode at the frontier: still advancing.
    attention_step(2, 10, 1, false, 0);
    cov = dispatcher_->indexer_coverage(2);
    EXPECT_EQ(cov.first, pinned_mode);
    EXPECT_EQ(cov.second, 11u);
}

// Fail-closed set unchanged under the flag: chunk gaps and
// superchunk-flagged rewind shapes still kill; a malformed mixed-sequence
// chunk is refused outright (coverage untouched).
TEST_F(IndexerCoverageGuardRewindTest, InvalidShapesStillFailClosed) {
    create_seq(3, 8);
    attention_step(3, 0, 4, true, 0);    // prefill 0..3 → coverage 4
    ASSERT_NE(dispatcher_->indexer_coverage(3).first, kModeDead);

    // Chunk gap: [6, 8) with next_pos == 4 → dead.
    attention_step(3, 6, 2, true, 0);
    EXPECT_EQ(dispatcher_->indexer_coverage(3).first, kModeDead);

    // Superchunk-flagged chunk crossing the frontier is NOT a rewind (the
    // relaxed repeat is end <= next_pos only): pos0 < next_pos < end with
    // the superchunk flag → dead.
    create_seq(4, 8);
    attention_step(4, 0, 4, true, 0);    // coverage 4
    ASSERT_NE(dispatcher_->indexer_coverage(4).first, kModeDead);
    attention_step(4, 2, 4, true, 0, /*superchunk=*/1);  // [2, 6), end > 4
    EXPECT_EQ(dispatcher_->indexer_coverage(4).first, kModeDead);

    // Mixed-sequence chunk: refused with coverage untouched under the
    // rewind flag too (TD-KVT-BATCH-COHORT — the malformed-shape refusal
    // is orthogonal to INV-DSA-REWIND).
    create_seq(5, 8);
    create_seq(6, 8);
    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {5, 0, 0};
    batch[1] = {6, 0, 0};
    const auto cmp = dispatch_attention_capture(
        2, /*prefill=*/true, /*layer=*/0, /*chunk_len=*/2);
    ASSERT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(cmp.error.error_category, kErrCatComputeValidation);
    EXPECT_EQ(dispatcher_->indexer_coverage(5).first, kModeUnknown)
        << "untouched: neither sequence ever appended, none may die";
    EXPECT_EQ(dispatcher_->indexer_coverage(6).first, kModeUnknown);
}

// ── TD-INDEXER-STEPKEY-TOKEN-BLIND (INV-DSA-EPOCH): rewind-epoch keying ──
// The IndexShare reuse-validity key must hash WHAT could have changed, not
// just WHERE: every step of one (sequence, epoch) emits one key across all
// its layers, and any overwrite re-feed — a step whose FIRST layer sees the
// sequence behind its coverage frontier — advances the epoch so the re-fed
// step (which may carry different tokens at the same positions) can never
// validate a selection cached from the overwritten content. Bumping must
// not kill coverage (kDead is terminal — TD-INDEXER-NO-DENSE-FALLBACK).
// Rewind shapes are blessed by the schema default (DefaultConfigBlessesRewind).

// All layers of one step emit one key; a new advancing step emits a new key.
TEST_F(IndexerCoverageGuardTest, StepKeyStableAcrossLayersOfOneStep) {
    create_seq(11, 8);

    attention_step(11, 0, 8, true, 0);            // prefill [0, 8), layer 0
    const uint64_t p0 = dispatcher_->last_indexer_step_key();
    ASSERT_NE(p0, 0u) << "blessed prefill chunk must carry a key";
    attention_step(11, 0, 8, true, 1);            // same chunk, layer 1
    EXPECT_EQ(dispatcher_->last_indexer_step_key(), p0)
        << "a later layer of the same step must reuse the step's key";

    attention_step(11, 8, 1, false, 0);           // decode @8, layer 0
    const uint64_t d0 = dispatcher_->last_indexer_step_key();
    ASSERT_NE(d0, 0u);
    EXPECT_NE(d0, p0) << "different step, different key";
    attention_step(11, 8, 1, false, 1);           // decode @8, layer 1
    EXPECT_EQ(dispatcher_->last_indexer_step_key(), d0)
        << "decode repeat (later layer) must reuse the step's key";
}

// The dsp52 partial-acceptance trajectory: a verify chunk re-fed over the
// SAME positions (draft tokens differ round to round) must carry a NEW key
// — while each re-fed step still keys consistently across its own layers
// and coverage stays alive.
TEST_F(IndexerCoverageGuardTest, ChunkRefeedSamePositionsYieldsNewKey) {
    create_seq(12, 8);
    attention_step(12, 0, 4, true, 0);            // prefill [0, 4)
    ASSERT_NE(dispatcher_->indexer_coverage(12).first, kModeDead);

    attention_step(12, 4, 6, true, 0);            // verify chunk [4, 10)
    const uint64_t v1 = dispatcher_->last_indexer_step_key();
    ASSERT_NE(v1, 0u);
    ASSERT_EQ(dispatcher_->indexer_coverage(12).second, 10u);
    attention_step(12, 4, 6, true, 1);            // later layer, same step
    EXPECT_EQ(dispatcher_->last_indexer_step_key(), v1);

    // Re-fed verify chunk over the SAME window (0 accepted, new drafts):
    // identical (seq, pos) rows — the token-blind key would collide.
    attention_step(12, 4, 6, true, 0);
    const uint64_t v2 = dispatcher_->last_indexer_step_key();
    ASSERT_NE(v2, 0u);
    EXPECT_NE(v2, v1) << "overwrite re-feed must not reproduce the stale key";
    attention_step(12, 4, 6, true, 1);            // its own later layer
    EXPECT_EQ(dispatcher_->last_indexer_step_key(), v2)
        << "the re-fed step keys consistently across its layers";

    // A THIRD identical-position re-feed gets yet another key (consecutive
    // re-feeds of one window must not collide with each other either).
    attention_step(12, 4, 6, true, 0);
    const uint64_t v3 = dispatcher_->last_indexer_step_key();
    ASSERT_NE(v3, 0u);
    EXPECT_NE(v3, v2);

    // The bump must invalidate reuse, never coverage.
    auto cov = dispatcher_->indexer_coverage(12);
    EXPECT_NE(cov.first, kModeDead) << "epoch bump must not kill coverage";
    EXPECT_EQ(cov.second, 10u);
    attention_step(12, 10, 1, false, 0);          // frontier decode advances
    EXPECT_NE(dispatcher_->indexer_coverage(12).first, kModeDead);
    EXPECT_EQ(dispatcher_->indexer_coverage(12).second, 11u);
}

// Decode re-feeds behind the frontier — the guided bonus-violation re-feed
// at (fed − 1) (pos + 1 == next_pos at the step's FIRST layer is a
// re-dispatch, not a later-layer repeat) and the deeper rewind row — must
// each draw a new key.
TEST_F(IndexerCoverageGuardTest, DecodeRefeedBehindFrontierYieldsNewKey) {
    create_seq(13, 8);
    attention_step(13, 0, 8, true, 0);            // prefill [0, 8)
    attention_step(13, 8, 1, false, 0);           // decode @8 → next_pos 9
    const uint64_t a = dispatcher_->last_indexer_step_key();
    ASSERT_NE(a, 0u);
    attention_step(13, 8, 1, false, 1);           // later layer: same key
    ASSERT_EQ(dispatcher_->last_indexer_step_key(), a);

    // Guided re-feed shape: decode @8 again from layer 0 (fed-1 == 8).
    attention_step(13, 8, 1, false, 0);
    const uint64_t b = dispatcher_->last_indexer_step_key();
    ASSERT_NE(b, 0u);
    EXPECT_NE(b, a) << "layer-0 re-dispatch of a position is an overwrite";
    attention_step(13, 8, 1, false, 1);
    EXPECT_EQ(dispatcher_->last_indexer_step_key(), b);

    // Deeper rewind row @6 (< 9): new key again; frontier stands, alive.
    attention_step(13, 6, 1, false, 0);
    const uint64_t c = dispatcher_->last_indexer_step_key();
    ASSERT_NE(c, 0u);
    EXPECT_NE(c, b);
    EXPECT_NE(c, a);
    auto cov = dispatcher_->indexer_coverage(13);
    EXPECT_NE(cov.first, kModeDead) << "epoch bump must not kill coverage";
    EXPECT_EQ(cov.second, 9u) << "next_pos is still the high-water mark";
}

// A recycled seq_id is a NEW incarnation: identical (seq, pos) steps of the
// two lives must not share keys (the executor's reuse map outlives seq_free).
TEST_F(IndexerCoverageGuardTest, RecycledSeqIdYieldsNewKey) {
    create_seq(14, 8);
    attention_step(14, 0, 4, true, 0);            // prefill [0, 4)
    const uint64_t life1 = dispatcher_->last_indexer_step_key();
    ASSERT_NE(life1, 0u);
    free_seq(14);

    create_seq(14, 8);                            // recycled id
    attention_step(14, 0, 4, true, 0);            // identical step shape
    const uint64_t life2 = dispatcher_->last_indexer_step_key();
    ASSERT_NE(life2, 0u);
    EXPECT_NE(life2, life1)
        << "a recycled seq_id must not reproduce a dead incarnation's key";
}

// ═══ R4a: truncating fork — indexer coverage truncation ════════════════════
// A truncated fork's coverage must be exactly [0, min(parent frontier,
// prefix_len)) with a correct frontier — never the parent's — and the
// child must continue exactly like a sequence that only ever saw
// prefix_len tokens: contiguous append at prefix_len stays sparse-
// eligible, anything past it is a GAP and fail-closes dense.  The child
// draws a FRESH rewind epoch (INV-DSA-EPOCH — same unconditional line as
// a full fork; RecycledSeqIdYieldsNewKey pins the mechanism).

TEST_F(IndexerCoverageGuardTest, TruncatedForkClampsCoverageToPrefix) {
    create_seq(11, 8);
    attention_step(11, 0, 6, true, 0);  // prefill 0..5 → coverage 6
    auto pcov = dispatcher_->indexer_coverage(11);
    ASSERT_EQ(pcov.second, 6u);
    ASSERT_NE(pcov.first, kModeDead);

    fork_seq(11, 12, /*prefix_len=*/4);
    auto cov = dispatcher_->indexer_coverage(12);
    EXPECT_EQ(cov.second, 4u) << "child coverage must clamp to prefix_len";
    EXPECT_EQ(cov.first, pcov.first)
        << "mode inherited (S4: every live parent is kPaged)";
    // Parent untouched.
    EXPECT_EQ(dispatcher_->indexer_coverage(11).second, 6u);

    if (cov.first != kModeDead) {
        // Contiguous decode at the truncation boundary: alive, advance.
        attention_step(12, 4, 1, false, 0);
        cov = dispatcher_->indexer_coverage(12);
        EXPECT_NE(cov.first, kModeDead)
            << "append at prefix_len must stay sparse-eligible";
        EXPECT_EQ(cov.second, 5u);
    }
}

TEST_F(IndexerCoverageGuardTest, TruncatedForkStepPastPrefixIsAGap) {
    create_seq(13, 8);
    attention_step(13, 0, 6, true, 0);  // coverage 6
    fork_seq(13, 14, /*prefix_len=*/4);
    ASSERT_EQ(dispatcher_->indexer_coverage(14).second, 4u);

    // Decode at 5 — position 4 was truncated away: GAP → dense forever.
    attention_step(14, 5, 1, false, 0);
    EXPECT_EQ(dispatcher_->indexer_coverage(14).first, kModeDead);
    // Parent unaffected: contiguous decode at ITS frontier still works.
    attention_step(13, 6, 1, false, 0);
    EXPECT_EQ(dispatcher_->indexer_coverage(13).second, 7u);
}

TEST_F(IndexerCoverageGuardTest, TruncatedForkBeyondParentCoverageInheritsShorter) {
    // prefix_len ABOVE the parent's covered frontier: the child inherits
    // the shorter frontier (coverage is contiguous — nothing exists past
    // it), and its first append at prefix_len is a gap → dense.  This is
    // the fail-closed answer for forking a coverage-lagging parent.
    create_seq(15, 8);
    attention_step(15, 0, 3, true, 0);  // coverage 3
    fork_seq(15, 16, /*prefix_len=*/5);
    auto cov = dispatcher_->indexer_coverage(16);
    EXPECT_EQ(cov.second, 3u) << "min(parent frontier, prefix_len)";

    attention_step(16, 5, 1, false, 0);  // gap (3, 4 never appended)
    EXPECT_EQ(dispatcher_->indexer_coverage(16).first, kModeDead);
}

// TD-DECODE-GRAPH coverage rider (P-29 step 7): a decode step arriving with
// use_graph=1 on a DSA model must MAINTAIN indexer coverage exactly as the
// eager path — the dispatcher forces it NONGRAPH until decode-graph capture
// wires the indexer append. Pre-rider behaviour (the bug this test would
// catch): the graphed step skips the coverage block, the next nongraph step
// sees pos > next_pos, marks the sequence kDead, and every later step
// silently serves DENSE — no error, ~10x slower, inflated-looking rates.
TEST_F(IndexerCoverageGuardTest, GraphedDecodeStepMaintainsCoverage) {
    create_seq(17, 8);
    attention_step(17, 0, 4, true, 0);  // prefill 0..3
    ASSERT_EQ(dispatcher_->indexer_coverage(17).second, 4u);
    const int pinned_mode = dispatcher_->indexer_coverage(17).first;

    // Decode at the frontier with use_graph=1: coverage must advance as if
    // the step were eager (the rider forces nongraph → append blessed).
    attention_step_flags(17, 4, /*layer=*/0, /*use_graph=*/1, /*is_draft=*/0);
    auto cov = dispatcher_->indexer_coverage(17);
    EXPECT_EQ(cov.first, pinned_mode)
        << "graphed decode step must not disturb the storage mode";
    EXPECT_EQ(cov.second, 5u)
        << "graphed decode step must advance coverage like the eager path";

    // Same graphed step re-dispatched for a later layer (repeat shape).
    attention_step_flags(17, 4, /*layer=*/1, /*use_graph=*/1, /*is_draft=*/0);
    EXPECT_EQ(dispatcher_->indexer_coverage(17).second, 5u);

    // The NEXT nongraph step is the silent-dense tripwire: pre-rider it saw
    // a gap here and killed the sequence.
    attention_step(17, 5, 1, false, 0);
    cov = dispatcher_->indexer_coverage(17);
    EXPECT_NE(cov.first, kModeDead)
        << "silent dense fallback: the step after a graphed decode step saw "
           "a coverage gap (TD-DECODE-GRAPH rider regression)";
    EXPECT_EQ(cov.second, 6u);
}

// Negative control for the rider test above: a DRAFT decode step is the
// still-sanctioned non-appending step kind — it must NOT advance coverage,
// and the next non-draft step at the frontier+1 must see the gap and kill
// the sequence. Proves the harness detects the kDead transition the rider
// test asserts against (the pass above is not vacuous), and pins the rider's
// boundary: use_graph is protected, drafts intentionally are not.
TEST_F(IndexerCoverageGuardTest, DraftDecodeStepStillGapsCoverage) {
    create_seq(18, 8);
    attention_step(18, 0, 4, true, 0);  // prefill 0..3
    ASSERT_EQ(dispatcher_->indexer_coverage(18).second, 4u);

    // Draft step at the frontier: no append, coverage untouched.
    attention_step_flags(18, 4, /*layer=*/0, /*use_graph=*/0, /*is_draft=*/1);
    EXPECT_EQ(dispatcher_->indexer_coverage(18).second, 4u)
        << "draft steps must not advance coverage";

    // Non-draft decode PAST the never-appended position: gap → kDead.
    attention_step(18, 5, 1, false, 0);
    EXPECT_EQ(dispatcher_->indexer_coverage(18).first, kModeDead)
        << "the harness must detect the silent-dense transition";
}
