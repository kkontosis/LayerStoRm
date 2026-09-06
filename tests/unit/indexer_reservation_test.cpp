// TD-INDEXER-NO-DENSE-FALLBACK (Route 1): RESERVE AT ADMISSION.
//
// A sequence's indexer-K pages are committed at CMD_SEQ_CREATE /
// CMD_SEQ_FORK (payload reserve_tokens) for the context it may ever reach,
// so per-step provisioning (ensure_indexer_pages) can never fail later.
// Locks:
//   - reservation allocates the page groups UP FRONT and later steps
//     allocate nothing (the guarantee);
//   - a create that cannot reserve is REFUSED with the RETRYABLE
//     kKvPoolExhausted category (the evict_for_admission seam's contract)
//     and leaves no partial state — after capacity is freed the identical
//     create succeeds;
//   - the case that FAILED SILENTLY before this ticket: an admitted
//     UNRESERVED sequence whose mid-prefill growth hits pool exhaustion
//     goes IndexerSeqMode::kDead (permanent dense) — now a loud BUG
//     witness: indexer_dense_total() counts it and the step's completion
//     carries Completion.compute.indexer_dense = 1;
//   - a RESERVED sequence never goes dense: an overrun past its
//     reservation raises the retryable exhaustion error (coverage mode
//     intact), and past the serving window raises kComputeValidation —
//     serve sparse or refuse, never quietly 10x slower;
//   - fork-time reservation (hit children): growth beyond the inherited
//     groups, retryable refusal + full rollback on exhaustion, truncated
//     children (INV-SEQ-FORK-TRUNC) resume growth from the truncated
//     grid, frozen holders ignore reserve_tokens;
//   - the granted total (clamped to serving.max_sequence_length) rides
//     Completion.seq_op.reserved_tokens.
//
// CPU-only: null device backends, real PageAllocator + dispatcher
// provisioning (mirrors indexer_local_provisioning_test at dcp=2,
// REPLICATED indexer mode — the champion shape).

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
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
constexpr int kModeUnknown = -1;   // indexer_coverage: untracked / no seq
constexpr int kModePaged = 1;      // IndexerSeqMode::kPaged
constexpr int kModeDead = 2;       // IndexerSeqMode::kDead (S4: kArena deleted)
constexpr int kPT = 16;            // indexer_k_page_size_tokens
constexpr int kLayers = 6;         // all computing (no IndexShare mask)
constexpr uint32_t kMaxSeq = 1024; // serving window -> 64 groups/seq
constexpr uint32_t kErrCatKvPoolExhausted = 29;  // CmpErrorCategory
constexpr uint32_t kErrCatComputeValidation = 10;  // malformed-shape refusal

void* aligned_alloc_zeroed(size_t bytes) {
    void* p = std::aligned_alloc(64, bytes);
    std::memset(p, 0, bytes);
    return p;
}

/// DSA-enabled dcp=2 config, REPLICATED indexer, tiny indexer page.
lc::Config dsa_config() {
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
            // Replicated KV: this fixture isolates the RESERVATION path;
            // page-claim lockstep is INV-KV-REP's own suite.
            {"dcp_kv_mode", "replicated"},
            {"system_ram_gb", 64}}},
        {"parallelism", {{"tensor_parallelism", 2}}},
        // max_concurrent_requests 2 keeps the INV-KVT-14b pool small (128
        // groups) so the exhaustion tests can fill it with a few hogs.
        {"serving", {{"max_sequence_length", static_cast<int>(kMaxSeq)},
                     {"max_concurrent_requests", 2}}},
        {"memory", {{"vram_safety_margin_gb", 0.1},
                    {"kv_cache", {{"page_growth_chunk_tokens", 64},
                                  {"indexer_k_page_size_tokens", kPT}}}}},
    };
    return lc::parse_config(j);
}

}  // namespace

class IndexerReservationTest : public ::testing::Test {
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

        cfg_ = std::make_unique<lc::Config>(dsa_config());
        mcfg_ = std::make_unique<lmod::ModelConfig>(*cfg_);
        fp8_ = std::make_unique<lmod::Fp8E4M3>();
        layer_reg_ = std::make_unique<lmod::LayerRegistry>(*mcfg_, *cfg_,
                                                           *fp8_);

        lc::GpuRef gpu0{0, 0, lc::GpuType::rtx5090};
        lc::GpuRef gpu1{1, 1, lc::GpuType::rtx5090};
        backends_.push_back(lcomp::make_null_device_backend(gpu0));
        backends_.push_back(lcomp::make_null_device_backend(gpu1));

        std::vector<lcomp::DeviceBackend*> dev_ptrs{backends_[0].get(),
                                                    backends_[1].get()};
        auto layout = lmem::compute_vram_layout(*cfg_, *layer_reg_, *mcfg_);
        // S4 (TD-INDEXER-POOL-ELASTIC): these tests exercise the
        // reservation/refusal STATE MACHINE against an exactly-bounded
        // indexer pool, so pin the legacy UNSLABBED fixed carve (the
        // elastic shared-pool shape is covered by the PageAllocator
        // elastic suite and the GPU gate). 768 pages = 64 groups/seq x
        // kLayers x 2 sequences — the pre-S4 carve for this config.
        layout.slab_bytes = 0;
        layout.pages_per_slab = 0;
        for (auto& g : layout.gpus) {
            if (g.kv_main_pages == 0) continue;
            g.indexer_k_pages = 768;
            g.indexer_k_bytes = static_cast<int64_t>(g.indexer_k_pages) *
                                layout.indexer_k_bytes_per_page;
            g.indexer_k_pad_bytes =
                (256 - g.indexer_k_bytes % 256) % 256;
            g.kv_main_bytes -=
                g.indexer_k_bytes + g.indexer_k_pad_bytes;
            g.kv_main_pages = static_cast<int>(
                (g.kv_main_bytes - g.prefill_scratch_preallocated_bytes) /
                layout.kv_bytes_per_page);
        }
        vram_ = std::make_unique<lmem::VramAllocator>(std::move(layout),
                                                      dev_ptrs);
        page_allocator_ = std::make_unique<lmem::PageAllocator>(
            *vram_, backends_[0].get());
        page_allocator_->set_dcp_config(lmem::DcpConfig{
            .dcp_size = 2,
            .dcp_chunk_size = cfg_->memory.kv_cache.dcp_chunk_size,
            .page_size_tokens = cfg_->memory.kv_cache.page_size_tokens,
            .tp_gpu_indices = {0, 1},
            .indexer_k_sharded = false,   // replicated indexer
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
            .device_backends        = {backends_[0].get(),
                                       backends_[1].get()},
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

    /// Drain: flush pending computes, collect every completion (errors
    /// included) into completions_.
    void drain() {
        dispatcher_->poll_compute_completions();
        lipc::Completion cmp{};
        while (cmp_ring_->try_read(&cmp)) completions_.push_back(cmp);
    }

    /// Last completion of a given type since the last clear, or nullopt.
    std::optional<lipc::Completion> last_of(uint32_t cmp_type) {
        for (auto it = completions_.rbegin(); it != completions_.rend();
             ++it)
            if (it->cmp_type == cmp_type) return *it;
        return std::nullopt;
    }

    void clear_completions() { completions_.clear(); }

    /// CMD_SEQ_CREATE with a reservation.  Returns the CMP (seq-op done or
    /// error).
    lipc::Completion create_seq(uint64_t seq_id, uint32_t prompt_len,
                                uint32_t reserve_tokens) {
        clear_completions();
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(lipc::CMD_SEQ_CREATE);
        c.cmd_seq = next_seq_++;
        c.seq_create.seq_id = seq_id;
        c.seq_create.prompt_len = prompt_len;
        c.seq_create.pool = 0;
        c.seq_create.reserve_tokens = reserve_tokens;
        dispatcher_->dispatch(c);
        drain();
        auto out = last_of(lipc::CMP_SEQ_OP_DONE);
        if (!out) out = last_of(lipc::CMP_ERROR);
        EXPECT_TRUE(out.has_value()) << "no completion for seq_create";
        return out.value_or(lipc::Completion{});
    }

    lipc::Completion fork_seq(uint64_t src, uint64_t dst,
                              uint32_t reserve_tokens,
                              uint32_t prefix_len = 0,
                              bool frozen = false) {
        clear_completions();
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(
            frozen ? lipc::CMD_SEQ_FORK_FROZEN : lipc::CMD_SEQ_FORK);
        c.cmd_seq = next_seq_++;
        c.seq_fork.src_seq_id = src;
        c.seq_fork.dst_seq_id = dst;
        c.seq_fork.prefix_len = prefix_len;
        c.seq_fork.reserve_tokens = reserve_tokens;
        dispatcher_->dispatch(c);
        drain();
        auto out = last_of(lipc::CMP_SEQ_OP_DONE);
        if (!out) out = last_of(lipc::CMP_ERROR);
        EXPECT_TRUE(out.has_value()) << "no completion for seq_fork";
        return out.value_or(lipc::Completion{});
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

    /// One attention step; returns its completion (compute-done or error).
    lipc::Completion attention_step(uint64_t seq_id, uint32_t pos0,
                                    uint32_t n, bool prefill,
                                    uint32_t layer = 0) {
        clear_completions();
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
        drain();
        auto out = last_of(lipc::CMP_COMPUTE_DONE);
        if (!out) out = last_of(lipc::CMP_ERROR);
        EXPECT_TRUE(out.has_value()) << "no completion for attention step";
        return out.value_or(lipc::Completion{});
    }

    /// Fill the kIndexerK pool with full-window reserved hog sequences
    /// until exactly `groups_left` whole groups remain free.  Returns the
    /// hog seq ids (free them to model holder eviction).
    std::vector<uint64_t> fill_pool_leaving(int groups_left) {
        std::vector<uint64_t> hogs;
        uint64_t next_hog = 1000;
        const int window_groups = static_cast<int>(kMaxSeq) / kPT;
        while (free_groups() - groups_left >= window_groups) {
            auto c = create_seq(next_hog, 8, kMaxSeq);
            EXPECT_EQ(c.cmp_type,
                      static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
            hogs.push_back(next_hog++);
        }
        const int extra = free_groups() - groups_left;
        if (extra > 0) {
            auto c = create_seq(next_hog, 8,
                                static_cast<uint32_t>(extra) * kPT);
            EXPECT_EQ(c.cmp_type,
                      static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
            hogs.push_back(next_hog++);
        }
        EXPECT_EQ(free_groups(), groups_left);
        return hogs;
    }

    int used_ik(int gpu) const {
        return page_allocator_->used_pages(gpu, lmem::Pool::kIndexerK);
    }
    int free_ik(int gpu) const {
        return page_allocator_->free_pages(gpu, lmem::Pool::kIndexerK);
    }
    /// Free whole (page, layer)-groups left in the pool (replicated: the
    /// same count on each GPU by lockstep).
    int free_groups() const { return free_ik(0) / kLayers; }

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
    std::vector<lipc::Completion> completions_;
    uint32_t next_seq_ = 1;
};

// Reservation commits the pages at create; later steps allocate NOTHING.
TEST_F(IndexerReservationTest, CreateReservesUpfrontAndStepsNeverAllocate) {
    const auto out = create_seq(1, /*prompt_len=*/40, /*reserve=*/48);
    ASSERT_EQ(out.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(out.seq_op.reserved_tokens, 48u);
    // 48 tokens at PT=16 -> 3 groups x 6 layers, replicated on both GPUs.
    EXPECT_EQ(used_ik(0), 3 * kLayers);
    EXPECT_EQ(used_ik(1), 3 * kLayers);

    // Prefill + decode across a page boundary: coverage advances SPARSE,
    // zero further pool allocation (the pages already exist).
    attention_step(1, 0, 40, /*prefill=*/true, /*layer=*/0);
    auto cov = dispatcher_->indexer_coverage(1);
    EXPECT_EQ(cov.first, kModePaged);
    EXPECT_EQ(cov.second, 40u);
    for (uint32_t pos = 40; pos < 48; ++pos)
        attention_step(1, pos, 1, /*prefill=*/false, /*layer=*/0);
    cov = dispatcher_->indexer_coverage(1);
    EXPECT_EQ(cov.first, kModePaged);
    EXPECT_EQ(cov.second, 48u);
    EXPECT_EQ(used_ik(0), 3 * kLayers);
    EXPECT_EQ(used_ik(1), 3 * kLayers);
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 0u);
}

// Granted reservation is clamped to serving.max_sequence_length.
TEST_F(IndexerReservationTest, ReserveClampedToServingWindow) {
    ASSERT_GE(free_groups(), static_cast<int>(kMaxSeq) / kPT)
        << "fixture pool too small for a full-window reservation";
    const auto out = create_seq(1, 8, /*reserve=*/0xFFFFFFFFu);
    ASSERT_EQ(out.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(out.seq_op.reserved_tokens, kMaxSeq);
    EXPECT_EQ(used_ik(0), (static_cast<int>(kMaxSeq) / kPT) * kLayers);
}

// THE CASE THAT FAILED SILENTLY BEFORE THIS TICKET: an admitted UNRESERVED
// sequence hits pool exhaustion mid-prefill -> permanent dense (kDead).
// Now it is a loud witness: transition counter + completion byte.  And the
// same admission with a RESERVATION is refused UP FRONT with the retryable
// category instead — after freeing capacity the identical create succeeds
// and the sequence serves sparse end to end.
TEST_F(IndexerReservationTest,
       UnreservedMidPrefillDeathBecomesReservedAdmissionRefusal) {
    // Fill the pool to ONE remaining group with reserved hogs.
    const auto hogs = fill_pool_leaving(1);

    // Legacy (reserve=0) admission: first chunk grabs the last group and
    // blesses kPaged; the SECOND chunk's growth fails -> kDead, exactly the
    // old silent 10x downgrade — now witnessed.
    const auto legacy = create_seq(1, 32, /*reserve=*/0);
    ASSERT_EQ(legacy.cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(legacy.seq_op.reserved_tokens, 0u);
    attention_step(1, 0, kPT, /*prefill=*/true);
    ASSERT_EQ(dispatcher_->indexer_coverage(1).first, kModePaged);
    ASSERT_EQ(dispatcher_->indexer_dense_total(), 0u);

    const auto step = attention_step(1, kPT, kPT, /*prefill=*/true);
    EXPECT_EQ(dispatcher_->indexer_coverage(1).first, kModeDead)
        << "unreserved growth failure must fail closed to dense (legacy)";
    EXPECT_GE(dispatcher_->indexer_dense_total(), 1u)
        << "the dense downgrade must be counted (bug witness)";
    ASSERT_EQ(step.cmp_type, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE));
    EXPECT_EQ(step.compute.indexer_dense, 1)
        << "the step's completion must carry the witness byte";
    free_seq(1);

    // NEW: the same admission WITH a reservation is refused at create —
    // retryable category, no partial state.
    const auto refused = create_seq(2, 32, /*reserve=*/32);
    ASSERT_EQ(refused.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(refused.error.error_category, kErrCatKvPoolExhausted);
    EXPECT_EQ(dispatcher_->indexer_coverage(2).first, kModeUnknown)
        << "a refused create must leave no sequence behind";

    // The orchestrator's evict-retry: free a hog (models holder
    // eviction), re-issue -> succeeds, and the sequence serves SPARSE
    // across the same boundary that killed the legacy one.
    free_seq(hogs.back());
    const auto ok = create_seq(2, 32, /*reserve=*/32);
    ASSERT_EQ(ok.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(ok.seq_op.reserved_tokens, 32u);
    attention_step(2, 0, kPT, /*prefill=*/true);
    attention_step(2, kPT, kPT, /*prefill=*/true);
    const auto cov = dispatcher_->indexer_coverage(2);
    EXPECT_EQ(cov.first, kModePaged);
    EXPECT_EQ(cov.second, 32u);
}

// A RESERVED sequence never goes dense: an overrun past its reservation
// under a full pool raises the RETRYABLE exhaustion error and leaves
// coverage intact (sparse), never kDead.
TEST_F(IndexerReservationTest, ReservedOverrunRaisesRetryableNotDense) {
    const auto out = create_seq(1, 8, /*reserve=*/2 * kPT);
    ASSERT_EQ(out.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    // Hog the remainder so the overrun cannot silently grow.
    fill_pool_leaving(0);

    attention_step(1, 0, 2 * kPT, /*prefill=*/true);
    ASSERT_EQ(dispatcher_->indexer_coverage(1).first, kModePaged);

    // Position 2*kPT needs a THIRD group: past the reservation, pool full.
    const auto step = attention_step(1, 2 * kPT, 1, /*prefill=*/false);
    ASSERT_EQ(step.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR))
        << "a reserved sequence must refuse, not degrade";
    EXPECT_EQ(step.error.error_category, kErrCatKvPoolExhausted);
    const auto cov = dispatcher_->indexer_coverage(1);
    EXPECT_EQ(cov.first, kModePaged) << "coverage must survive the refusal";
    EXPECT_EQ(cov.second, 2u * kPT);
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 0u);
}

// A RESERVED sequence stepping past the SERVING WINDOW is refused with a
// validation error (permanent shape, not retryable) — never dense.
TEST_F(IndexerReservationTest, ReservedBeyondWindowRefusedNotDense) {
    ASSERT_GE(free_groups(), static_cast<int>(kMaxSeq) / kPT);
    const auto out = create_seq(1, 8, kMaxSeq);
    ASSERT_EQ(out.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    // Bless coverage, then jump the frontier to the window edge via the
    // reserved pages (chunked prefill in window-sized strides).
    for (uint32_t pos = 0; pos < kMaxSeq; pos += 64)
        attention_step(1, pos, 64, /*prefill=*/true);
    ASSERT_EQ(dispatcher_->indexer_coverage(1).second, kMaxSeq);

    const auto step = attention_step(1, kMaxSeq, 1, /*prefill=*/false);
    // The refusal may come from the kMain kv-meta layer (which also
    // fail-closes past the window, kKvPoolExhausted) or from the indexer
    // reserved-window guard (kComputeValidation) — what this test locks is
    // that the step is REFUSED and coverage survives sparse, never a
    // silent dense downgrade.
    ASSERT_EQ(step.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(dispatcher_->indexer_coverage(1).first, kModePaged);
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 0u);
}

// Fork-time reservation: a hit child grows past its inherited groups at
// fork; exhaustion is a retryable refusal with FULL rollback; after
// eviction the identical fork succeeds.
TEST_F(IndexerReservationTest, ForkChildReservationRefusesThenSucceeds) {
    const auto parent = create_seq(1, 8, 2 * kPT);
    ASSERT_EQ(parent.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    attention_step(1, 0, 2 * kPT, /*prefill=*/true);
    ASSERT_EQ(dispatcher_->indexer_coverage(1).first, kModePaged);

    // Hog all but ONE group; the child wants TWO more than the parent's 2.
    const auto hogs = fill_pool_leaving(1);

    const int used0 = used_ik(0);
    const auto refused = fork_seq(1, 2, /*reserve=*/4 * kPT);
    ASSERT_EQ(refused.cmp_type, static_cast<uint32_t>(lipc::CMP_ERROR));
    EXPECT_EQ(refused.error.error_category, kErrCatKvPoolExhausted);
    EXPECT_EQ(dispatcher_->indexer_coverage(2).first, kModeUnknown)
        << "refused fork must roll the child back completely";
    EXPECT_EQ(used_ik(0), used0) << "refused fork must free its growth";

    // Evict a hog (the orchestrator's fork-evict-retry) -> fork succeeds
    // and the child decodes SPARSE past the parent frontier.
    free_seq(hogs.back());
    const auto ok = fork_seq(1, 2, /*reserve=*/4 * kPT);
    ASSERT_EQ(ok.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(ok.seq_op.reserved_tokens, 4u * kPT);
    for (uint32_t pos = 2 * kPT; pos < 4 * kPT; ++pos)
        attention_step(2, pos, 1, /*prefill=*/false);
    const auto cov = dispatcher_->indexer_coverage(2);
    EXPECT_EQ(cov.first, kModePaged);
    EXPECT_EQ(cov.second, 4u * kPT);
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 0u);
}

// R4a truncating fork x reservation: the child keeps groups
// [0, ceil(prefix/PT)) and the reservation grows it from there; coverage
// clamps to the prefix and the delta prefill stays sparse.
TEST_F(IndexerReservationTest, TruncatedForkChildReservationGrows) {
    const auto parent = create_seq(1, 8, 2 * kPT);
    ASSERT_EQ(parent.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    attention_step(1, 0, 2 * kPT, /*prefill=*/true);

    const auto ok = fork_seq(1, 2, /*reserve=*/3 * kPT,
                             /*prefix_len=*/kPT);
    ASSERT_EQ(ok.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(ok.seq_op.reserved_tokens, 3u * kPT);
    auto cov = dispatcher_->indexer_coverage(2);
    EXPECT_EQ(cov.first, kModePaged);
    EXPECT_EQ(cov.second, static_cast<uint32_t>(kPT))
        << "truncated child coverage = [0, prefix_len) (INV-SEQ-FORK-TRUNC)";

    // Delta prefill from the truncation point through the reserved window.
    attention_step(2, kPT, kPT, /*prefill=*/true);
    attention_step(2, 2 * kPT, kPT, /*prefill=*/true);
    cov = dispatcher_->indexer_coverage(2);
    EXPECT_EQ(cov.first, kModePaged);
    EXPECT_EQ(cov.second, 3u * kPT);
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 0u);
}

// Frozen holders ignore reserve_tokens: registration stays a pure refcount
// share (INV-PREFIX-CACHE-3) with zero new indexer pages and grant 0.
TEST_F(IndexerReservationTest, FrozenForkIgnoresReservation) {
    const auto parent = create_seq(1, 8, 2 * kPT);
    ASSERT_EQ(parent.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    attention_step(1, 0, 2 * kPT, /*prefill=*/true);
    const int used0 = used_ik(0);

    const auto holder = fork_seq(1, 50, /*reserve=*/8 * kPT,
                                 /*prefix_len=*/0, /*frozen=*/true);
    ASSERT_EQ(holder.cmp_type, static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    EXPECT_EQ(holder.seq_op.reserved_tokens, 0u);
    EXPECT_EQ(used_ik(0), used0)
        << "a frozen holder must not claim reservation pages";
}

// TD-KVT-BATCH-COHORT (indexer-coverage rider): a malformed prefill cohort
// (mixed sequences in one chunk — a shape no live descriptor writer can
// express) touching RESERVED sequences is REFUSED with kComputeValidation.
// A reserved sequence must never be densified by a broken command stream
// (INV-DSA-RESERVE (c)): both coverage states survive intact, the dense
// witness stays at zero, and both sequences keep serving sparse afterwards.
TEST_F(IndexerReservationTest, MalformedPrefillCohortRefusedNotDense) {
    ASSERT_EQ(create_seq(1, 8, 2 * kPT).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    ASSERT_EQ(create_seq(2, 8, 2 * kPT).cmp_type,
              static_cast<uint32_t>(lipc::CMP_SEQ_OP_DONE));
    attention_step(1, 0, kPT, /*prefill=*/true);
    attention_step(2, 0, kPT, /*prefill=*/true);
    ASSERT_EQ(dispatcher_->indexer_coverage(1).first, kModePaged);
    ASSERT_EQ(dispatcher_->indexer_coverage(2).first, kModePaged);

    // Hand-craft the malformed cohort: rows of TWO sequences in ONE chunk.
    clear_completions();
    auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
        sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
    batch[0] = {1, static_cast<uint32_t>(kPT), 0};
    batch[1] = {2, static_cast<uint32_t>(kPT), 0};
    lipc::Command c{};
    c.cmd_type = static_cast<uint32_t>(lipc::D_B_CMD_RUN_ATTENTION);
    c.cmd_seq = next_seq_++;
    c.run_attention.layer_idx = 0;
    c.run_attention.num_seqs = 2;
    c.run_attention.is_prefill = 1;
    c.run_attention.use_graph = 0;
    c.run_attention.is_draft = 0;
    c.run_attention.chunk_start = 0;
    c.run_attention.chunk_len = 2;
    dispatcher_->dispatch(c);
    drain();
    const auto step = last_of(lipc::CMP_ERROR);
    ASSERT_TRUE(step.has_value())
        << "a malformed prefill cohort must be refused";
    EXPECT_EQ(step->error.error_category, kErrCatComputeValidation);

    // Coverage untouched, witness silent — never densified.
    auto cov1 = dispatcher_->indexer_coverage(1);
    auto cov2 = dispatcher_->indexer_coverage(2);
    EXPECT_EQ(cov1.first, kModePaged);
    EXPECT_EQ(cov1.second, static_cast<uint32_t>(kPT));
    EXPECT_EQ(cov2.first, kModePaged);
    EXPECT_EQ(cov2.second, static_cast<uint32_t>(kPT));
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 0u);

    // Both keep serving SPARSE on well-formed chunks inside their
    // reservations.
    attention_step(1, kPT, kPT, /*prefill=*/true);
    attention_step(2, kPT, kPT, /*prefill=*/true);
    EXPECT_EQ(dispatcher_->indexer_coverage(1).second, 2u * kPT);
    EXPECT_EQ(dispatcher_->indexer_coverage(2).second, 2u * kPT);
    EXPECT_EQ(dispatcher_->indexer_dense_total(), 0u);
}
