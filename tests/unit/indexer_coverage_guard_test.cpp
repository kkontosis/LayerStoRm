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
//   - an unsupported prefill shape (mixed sequences) kills the involved
//     sequences (no silent stale-key hazard).
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
constexpr int kModeDead = 3;

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
}

// Unsupported prefill shape (mixed sequences in one chunk) kills the involved
// sequences — nothing was appended, so they must never score sparsely.
TEST_F(IndexerCoverageGuardTest, MixedSequencePrefillKillsBoth) {
    create_seq(4, 8);
    create_seq(5, 8);

    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {4, 0, 0};
    batch[1] = {5, 0, 0};
    dispatch_attention(/*num_seqs=*/2, /*prefill=*/true, /*layer=*/0,
                       /*chunk_len=*/2);

    EXPECT_EQ(dispatcher_->indexer_coverage(4).first, kModeDead);
    EXPECT_EQ(dispatcher_->indexer_coverage(5).first, kModeDead);
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

// Fail-closed set unchanged under the flag: chunk gaps, mixed-sequence
// chunks, and superchunk-flagged rewind shapes still kill.
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

    // Mixed-sequence chunk still kills both (shape_ok guard untouched).
    create_seq(5, 8);
    create_seq(6, 8);
    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {5, 0, 0};
    batch[1] = {6, 0, 0};
    dispatch_attention(2, /*prefill=*/true, /*layer=*/0, /*chunk_len=*/2);
    EXPECT_EQ(dispatcher_->indexer_coverage(5).first, kModeDead);
    EXPECT_EQ(dispatcher_->indexer_coverage(6).first, kModeDead);
}
