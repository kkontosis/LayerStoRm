// Unit tests for DCP executor.
//
// Group 1: Construction/passthrough (null backend, no GPU)
// Group 2: Call sequencing (recording backend)
// Group 3: Multi-GPU correctness (REQUIRES_MULTI_GPU(2))

#include "parallelism/dcp_executor.h"
#include "parallelism/dcp_communicator.h"
#include "parallelism/kv_tiering_hook.h"  // GLM-25k (TD-KVT-PREFILL tests)
#include "compute/kernels/attention/dcp_attention_wrapper.h"
#include "config/config_parser.h"
#include "compute/stream_manager.h"
#include "core/attention_device.h"
#include "core/null_attention_device.h"
#include "core/device_backend.h"
#include "core/null_device_backend.h"
#include "parallelism/null_collective_backend.h"
#include "recording_collective_backend.h"
#include "compute/cuda_sm120_device_backend.h"
#include "parallelism/nccl_collective_backend.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <sstream>
#include <string>
#include <vector>

#include "core/gpu_ref.h"

namespace lp = layerstorm::parallelism;
namespace lc = layerstorm::compute;

// ── Helpers ─────────────────────────────────────────────────────────────────

// Reduced-dimension model config for tests.
// V3.2-like but smaller to avoid large allocations.
static constexpr int kHidden = 256;
static constexpr int kHeads = 8;
static constexpr int kQLora = 64;
static constexpr int kKVLora = 32;
static constexpr int kQkRope = 16;
static constexpr int kQkNope = 32;
static constexpr int kVHeadDim = 32;
static constexpr int kMaxBatch = 4;

static std::vector<layerstorm::config::GpuRef> make_gpu_refs(int count) {
    std::vector<layerstorm::config::GpuRef> v;
    for (int i = 0; i < count; ++i)
        v.push_back({.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090});
    return v;
}

// ── Owned NullDeviceBackend storage (outlives DcpCommunicator) ──────────────
static std::vector<std::unique_ptr<lc::DeviceBackend>> g_null_device_backends;

static std::vector<lc::DeviceBackend*> make_null_device_backends(int count) {
    g_null_device_backends.clear();
    std::vector<lc::DeviceBackend*> ptrs;
    for (int i = 0; i < count; ++i) {
        g_null_device_backends.push_back(lc::make_null_device_backend(
            layerstorm::config::GpuRef{.position = i, .id = i,
                                        .type = layerstorm::config::GpuType::rtx5090}));
        ptrs.push_back(g_null_device_backends.back().get());
    }
    return ptrs;
}

// ── Owned NullCollectiveBackend storage (outlives DcpCommunicator) ──────────
static std::unique_ptr<lp::CollectiveBackend> g_null_collective;

static lp::CollectiveBackend* ensure_null_collective() {
    if (!g_null_collective)
        g_null_collective = lp::make_null_collective_backend();
    return g_null_collective.get();
}

// ── Recording AttentionDevice ───────────────────────────────────────────────
// Logs set_device, gemm, rmsnorm, quantize_fp8, prefill_attention calls.
// Attention pipeline methods (k_append, decode graph, dcp graph) are no-ops.

class RecordingAttentionDevice : public lc::AttentionDevice {
public:
    RecordingAttentionDevice(layerstorm::config::GpuRef gpu,
                             std::vector<std::string>& log)
        : gpu_(gpu), log_(log) {}

    // ── Device selection + identity ─────────────────────────────────────────
    void set_device() override {
        log_.push_back("set_device(" + std::to_string(gpu_.id) + ")");
    }
    const layerstorm::config::GpuRef& gpu() const override { return gpu_; }

    // ── Compute kernels ─────────────────────────────────────────────────────
    void gemm(const lc::Fp8GemmParams& p, void*, void*) override {
        log_.push_back("gemm(M=" + std::to_string(p.M) +
                       ",N=" + std::to_string(p.N) +
                       ",K=" + std::to_string(p.K) + ")");
    }
    void gguf_mmvq(const lc::GgufGemmParams& p, void*, void*) override {
        log_.push_back("gguf_mmvq(M=" + std::to_string(p.M) +
                       ",N=" + std::to_string(p.N) +
                       ",K=" + std::to_string(p.K) +
                       ",t=" + std::to_string(static_cast<int>(p.type)) + ")");
    }
    void gguf_mmq(const lc::GgufGemmParams& p, void*, void*) override {
        log_.push_back("gguf_mmq(M=" + std::to_string(p.M) +
                       ",N=" + std::to_string(p.N) +
                       ",K=" + std::to_string(p.K) +
                       ",t=" + std::to_string(static_cast<int>(p.type)) + ")");
    }
    void gguf_dequant_gemm(const lc::GgufGemmParams& p, void*) override {
        log_.push_back("gguf_dequant_gemm(M=" + std::to_string(p.M) +
                       ",N=" + std::to_string(p.N) +
                       ",K=" + std::to_string(p.K) +
                       ",t=" + std::to_string(static_cast<int>(p.type)) + ")");
    }
    void rmsnorm(void*, const void*, const void*, float,
                 int tokens, int dim, int stride, void*) override {
        log_.push_back("rmsnorm(tokens=" + std::to_string(tokens) +
                       ",dim=" + std::to_string(dim) +
                       ",stride=" + std::to_string(stride) + ")");
    }
    void quantize_fp8(const lc::DynamicFp8QuantParams& p, void*) override {
        log_.push_back("quant_fp8(M=" + std::to_string(p.num_tokens) +
                       ",K=" + std::to_string(p.hidden_size) + ")");
    }
    void weight_quantize_fp8(const lc::WeightFp8QuantParams&, void*) override {}
    void nvfp4_dequant_bf16(const lc::Nvfp4DequantBf16Params&, void*) override {}
    void nvfp4_grouped_gemm(const lc::Nvfp4GroupedGemmParams&, void*, size_t,
                            void*) override {}
    void bf16_to_nvfp4_grouped(const lc::Bf16ToNvfp4GroupedParams&,
                               void*) override {}
    void kv_bv_extract_dequant(const lc::KvBvExtractDequantParams&,
                                void*) override {}
    void batched_gemm_bf16(const lc::StridedBatchedGemmBf16Params& p,
                            void*) override {
        log_.push_back("batched_gemm_bf16(m=" + std::to_string(p.m) +
                       ",n=" + std::to_string(p.n) +
                       ",k=" + std::to_string(p.k) +
                       ",batch=" + std::to_string(p.batch_count) + ")");
    }
    // KVS-3 (INV-KVS-POS): log the POSITION-SOURCE pointer so tests can
    // assert sharded mode feeds RoPE the GLOBAL seqlens array (not the
    // rank-local shard lengths).
    void absorb_q(const lc::QAbsorbParams& p, void*) override {
        std::ostringstream os;
        os << "absorb_q(seqlens=" << static_cast<const void*>(p.seqlens_k)
           << ",rope=" << (p.apply_rope ? 1 : 0) << ")";
        log_.push_back(os.str());
    }
    void rope_rotate(const lc::RopeRotateParams& p, void*) override {
        std::ostringstream os;
        os << "rope_rotate(seqlens=" << static_cast<const void*>(p.seqlens_k)
           << ")";
        log_.push_back(os.str());
    }

    // ── Device memory ───────────────────────────────────────────────────────
    void* device_alloc(size_t bytes) override { return std::malloc(bytes); }
    void  device_free(void* ptr) override { std::free(ptr); }
    void  device_sync() override {}
    void  memcpy_h2d(void* dst, const void* src, size_t bytes) override {
        if (dst && src && bytes > 0) std::memcpy(dst, src, bytes);
    }
    void  memcpy_2d_d2d_async(void* dst, size_t dpitch,
                              const void* src, size_t spitch,
                              size_t width, size_t height,
                              void*) override {
        log_.push_back("memcpy2d(w=" + std::to_string(width) +
                       ",h=" + std::to_string(height) + ")");
        if (!dst || !src) return;
        for (size_t r = 0; r < height; ++r)
            std::memcpy(static_cast<char*>(dst) + r * dpitch,
                        static_cast<const char*>(src) + r * spitch, width);
    }

    // ── KV cache append ─────────────────────────────────────────────────────
    void k_append(const void*, const void*, void*, int64_t, int,
                  const int*, int, int, int, int, int, int,
                  int, void*) override {}

    // ── Prefill attention ───────────────────────────────────────────────────
    void prefill_attention(const void*, int batch_size, int seq_len_kv,
                           const int*, const int*, int,
                           void*, int64_t, int,
                           int, bool is_sparse, bool chunk_causal,
                           const int*, const int*,
                           int, void*, float*,
                           int, void*) override {
        log_.push_back("prefill(B=" + std::to_string(batch_size) +
                       ",skv=" + std::to_string(seq_len_kv) +
                       ",sparse=" + std::to_string(is_sparse) +
                       ",causal=" + std::to_string(chunk_causal) + ")");
    }

    // ── DSA indexer (TD-SPARSE-CHUNK-PREFILL wiring probe) ──────────────────
    // Logs the per-row score/top-k shape (bound + causal cutoff) the sparse-
    // prefill producer issues; the other indexer kernels stay base no-ops.
    void indexer_score_topk(const lc::IndexerScoreTopkArgs& a,
                            void*) override {
        log_.push_back("idx_topk(nb=" + std::to_string(a.num_blocks) +
                       ",qpos=" + std::to_string(a.query_position_base) + ")");
    }
    // TD-SPARSE-PREFILL-SCORE-BATCH wiring probe: ONE batched score+top-k
    // launch pair per wave. device_alloc is host malloc and memcpy_h2d is
    // memcpy here, so the staged per-row bound/cutoff device arrays are
    // readable — the log carries each row's OWN bound + cutoff, proving the
    // batched call preserves the per-row shapes the retired loop issued.
    void indexer_score_topk_batched(const lc::IndexerScoreTopkBatchedArgs& a,
                                    void*) override {
        std::string nb, qp;
        const int* rb = static_cast<const int*>(a.row_num_blocks);
        const int* rq = static_cast<const int*>(a.row_query_position);
        for (int i = 0; i < a.num_rows; ++i) {
            nb += (i ? "," : "") + std::to_string(rb[i]);
            qp += (i ? "," : "") + std::to_string(rq[i]);
        }
        log_.push_back("idx_topk_batched(rows=" + std::to_string(a.num_rows) +
                       ",nb=[" + nb + "],qpos=[" + qp + "],paged=" +
                       std::to_string(a.k_page_table != nullptr) + ")");
    }
    // TD-SPARSE-PREFILL-LOCAL-INDEXER wiring probe: the per-row cross-rank
    // merge shape (row, global causal bound + cutoff).
    void indexer_topk_merge(const lc::IndexerTopkMergeArgs& a,
                            void*) override {
        log_.push_back("idx_merge(tok=" + std::to_string(a.token) +
                       ",nb=" + std::to_string(a.num_blocks) +
                       ",qpos=" + std::to_string(a.query_position) + ")");
    }

    // ── Decode graph ops ────────────────────────────────────────────────────
    void decode_graph_update(lc::GraphEntry&, const void*,
                             const int*, const int*,
                             const int*,
                             int, void*) override {}
    void decode_graph_replay(lc::GraphEntry&, void*) override {}
    void* decode_graph_out_ptr(lc::GraphEntry&) override { return nullptr; }
    float* decode_graph_lse_ptr(lc::GraphEntry&) override { return nullptr; }

    // ── DCP allreduce graph ─────────────────────────────────────────────────
    void dcp_graph_replay(lc::GraphEntry&, void*) override {}

private:
    layerstorm::config::GpuRef gpu_;
    std::vector<std::string>& log_;
};

// ── Helpers for building Options with NullAttentionDevice ────────────────────

// Owned NullAttentionDevice storage for tests (per-rank, reused across tests).
// Tests that need recording devices create their own.
static std::vector<std::unique_ptr<lc::AttentionDevice>> g_null_devices;

static std::vector<lc::AttentionDevice*> make_null_attention_devices(int count) {
    g_null_devices.clear();
    std::vector<lc::AttentionDevice*> ptrs;
    for (int i = 0; i < count; ++i) {
        g_null_devices.push_back(lc::make_null_attention_device(
            layerstorm::config::GpuRef{.position = i, .id = i,
                                        .type = layerstorm::config::GpuType::rtx5090}));
        ptrs.push_back(g_null_devices.back().get());
    }
    return ptrs;
}

static lp::DcpCommunicator::Options null_comm_opts(int dcp_size) {
    return {
        .dcp_size         = dcp_size,
        .device_backends  = make_null_device_backends(dcp_size),
        .max_batch_size   = kMaxBatch,
        .num_heads        = kHeads,
        .attn_output_dim  = kVHeadDim,
        .hidden_size      = kHidden,
        .collective       = ensure_null_collective(),
    };
}

static lp::DcpExecutor::Options executor_opts(int dcp_size) {
    return {
        .dcp_size            = dcp_size,
        .gpus                = make_gpu_refs(dcp_size),
        .max_batch_size      = kMaxBatch,
        .hidden_size         = kHidden,
        .num_attention_heads = kHeads,
        .q_lora_rank         = kQLora,
        .kv_lora_rank        = kKVLora,
        .qk_rope_head_dim    = kQkRope,
        .qk_nope_head_dim    = kQkNope,
        .v_head_dim          = kVHeadDim,
        .rms_norm_eps        = 1e-6f,
        .attention_devices   = make_null_attention_devices(dcp_size),
    };
}

/// TD-90b: KD-4j requires dequant_pool_ init via set_layer_weights().
/// Call after constructing DcpExecutor for any test that calls execute_attention().
static std::vector<std::vector<lp::AttentionLayerWeights>>
        fake_attn_weights_storage;  // kept alive for test duration

static void init_dequant_pool(lp::DcpExecutor& exec, int dcp_size,
                               int num_layers = 6) {
    fake_attn_weights_storage.assign(
        num_layers, std::vector<lp::AttentionLayerWeights>(dcp_size));
    std::vector<std::vector<const lp::AttentionLayerWeights*>> ptrs(num_layers);
    for (int l = 0; l < num_layers; ++l)
        for (int r = 0; r < dcp_size; ++r)
            ptrs[l].push_back(&fake_attn_weights_storage[l][r]);
    exec.set_layer_weights(std::move(ptrs), num_layers);
}

// ============================================================================
// Group 1: Construction / passthrough (null backend, no GPU)
// ============================================================================

TEST(DcpExecutor, DcpSize1Passthrough) {
    auto opts = executor_opts(1);
    auto exec = lp::DcpExecutor(opts);

    EXPECT_FALSE(exec.is_active());
    EXPECT_EQ(exec.dcp_size(), 1);
    EXPECT_EQ(exec.num_heads_local(), kHeads);
}

TEST(DcpExecutor, DcpSize2Active) {
    auto comm = lp::DcpCommunicator(null_comm_opts(2));
    auto opts = executor_opts(2);
    opts.communicator = &comm;
    auto exec = lp::DcpExecutor(opts);

    EXPECT_TRUE(exec.is_active());
    EXPECT_EQ(exec.dcp_size(), 2);
    EXPECT_EQ(exec.num_heads_local(), kHeads / 2);
}

TEST(DcpExecutor, InvalidGpusSizeThrows) {
    auto opts = executor_opts(2);
    opts.gpus = make_gpu_refs(1);  // Only 1 GPU for dcp_size=2
    EXPECT_THROW({
        [[maybe_unused]] auto exec = lp::DcpExecutor(std::move(opts));
    }, std::invalid_argument);
}

TEST(DcpExecutor, EmptyGpusThrows) {
    auto opts = executor_opts(1);
    opts.gpus.clear();
    EXPECT_THROW({
        [[maybe_unused]] auto exec = lp::DcpExecutor(std::move(opts));
    }, std::invalid_argument);
}

TEST(DcpExecutor, InvalidAttentionDevicesSizeThrows) {
    auto opts = executor_opts(2);
    opts.attention_devices.resize(1);  // Mismatch: dcp_size=2, devices=1
    EXPECT_THROW({
        [[maybe_unused]] auto exec = lp::DcpExecutor(std::move(opts));
    }, std::invalid_argument);
}

TEST(DcpExecutor, DestroyCleanup) {
    // Verify construction + destruction of dcp_size=2 doesn't crash
    auto comm = lp::DcpCommunicator(null_comm_opts(2));
    {
        auto opts = executor_opts(2);
        opts.communicator = &comm;
        auto exec = lp::DcpExecutor(opts);
        EXPECT_TRUE(exec.is_active());
    }
    // Destructor frees all buffers — no crash
}

// ============================================================================
// Group 2: Call sequencing (recording backend)
// ============================================================================

namespace {

// Shared recording collective backend (tests/unit/recording_collective_backend.h).
using RecordingCollectiveBackend = layerstorm::test::RecordingCollectiveBackend;

}  // anonymous namespace

static lc::LaunchCorrectionFn recording_correction_fn(
    std::vector<std::string>& log) {
    return [&log](void*, const float*, float*,
                  int, int, int, int, int rank, void*) {
        log.push_back("correction(rank=" + std::to_string(rank) + ")");
    };
}

// Helper to build AttentionExecParams for testing.
// Uses heap-allocated dummy data that survives the test.
struct TestExecParamsBuilder {
    int dcp_size;
    int batch_size;

    // Per-rank dummy data
    std::vector<std::vector<char>> hidden_data;
    std::vector<std::vector<int>> seqlens_data;
    std::vector<std::vector<int>> block_tables_data;
    std::vector<std::vector<int>> slot_mapping_data;
    std::vector<lp::AttentionLayerWeights> weights_data;

    // Pointer arrays
    std::vector<void*> hidden_ptrs;
    std::vector<const int*> seqlens_ptrs;
    std::vector<const int*> block_table_ptrs;
    std::vector<const int*> slot_mapping_ptrs;
    std::vector<const lp::AttentionLayerWeights*> weight_ptrs;
    std::vector<void*> kv_cache_ptrs;

    TestExecParamsBuilder(int dcp, int batch)
        : dcp_size(dcp), batch_size(batch) {
        hidden_data.resize(dcp_size);
        seqlens_data.resize(dcp_size);
        block_tables_data.resize(dcp_size);
        slot_mapping_data.resize(dcp_size);
        weights_data.resize(dcp_size);
        hidden_ptrs.resize(dcp_size);
        seqlens_ptrs.resize(dcp_size);
        block_table_ptrs.resize(dcp_size);
        slot_mapping_ptrs.resize(dcp_size);
        weight_ptrs.resize(dcp_size);
        kv_cache_ptrs.resize(dcp_size);

        for (int r = 0; r < dcp_size; ++r) {
            hidden_data[r].resize(batch_size * kHidden * 2, 0);  // BF16
            seqlens_data[r].resize(batch_size, 100);
            block_tables_data[r].resize(batch_size * 4, 0);
            slot_mapping_data[r].resize(batch_size, 0);

            hidden_ptrs[r] = hidden_data[r].data();
            seqlens_ptrs[r] = seqlens_data[r].data();
            block_table_ptrs[r] = block_tables_data[r].data();
            slot_mapping_ptrs[r] = slot_mapping_data[r].data();
            weight_ptrs[r] = &weights_data[r];
            kv_cache_ptrs[r] = nullptr;
        }
    }

    lp::AttentionExecParams build(bool use_graph = false, bool sparse = false) {
        return {
            .layer_idx = 0,
            .batch_size = batch_size,
            .hidden_states = hidden_ptrs.data(),
            .seqlens_k = seqlens_ptrs.data(),
            .block_tables = block_table_ptrs.data(),
            .slot_mappings = slot_mapping_ptrs.data(),
            .kv_cache_ptrs = kv_cache_ptrs.data(),
            .weights = weight_ptrs.data(),
            .use_graph = use_graph,
            .is_sparse = sparse,
        };
    }
};

// Helper: build recording options (creates recording attention devices).
// Caller must keep returned struct alive while DcpExecutor is in use.
struct RecordingBackends {
    std::vector<std::unique_ptr<RecordingAttentionDevice>> devices;
    std::vector<lc::AttentionDevice*> attn_ptrs;

    RecordingBackends(int dcp_size, std::vector<std::string>& log) {
        for (int i = 0; i < dcp_size; ++i) {
            devices.push_back(std::make_unique<RecordingAttentionDevice>(
                layerstorm::config::GpuRef{.position = i, .id = i,
                                            .type = layerstorm::config::GpuType::rtx5090},
                log));
            attn_ptrs.push_back(devices.back().get());
        }
    }

    void apply(lp::DcpExecutor::Options& opts) {
        opts.attention_devices = attn_ptrs;
    }
};

TEST(DcpExecutor, PrefillCallSequenceDcpSize1) {
    // Single GPU: verify the call sequence for non-graph mode.
    std::vector<std::string> log;

    auto opts = executor_opts(1);
    RecordingBackends rec(1, log);
    rec.apply(opts);

    // Need a DcpAttentionWrapper for reduce_hidden
    auto comm = lp::DcpCommunicator(null_comm_opts(1));
    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .gpus = make_gpu_refs(1)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;
    opts.communicator = &comm;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());
    log.clear();

    TestExecParamsBuilder builder(1, 2);
    auto params = builder.build(/*use_graph=*/false);
    exec.execute_attention(params);

    // Verify key elements in order
    ASSERT_GE(log.size(), 5u) << "Expected at least several calls";

    // First call should be set_device(0)
    EXPECT_EQ(log[0], "set_device(0)");

    bool found_input_norm = false;
    bool found_hidden_quant = false;
    bool found_q_a_gemm = false;
    bool found_q_b_gemm = false;
    bool found_kv_a_gemm = false;
    bool found_q_a_norm = false;
    bool found_kv_a_norm = false;
    bool found_oproj_gemm = false;

    int idx_input_norm = -1;
    int idx_hidden_quant = -1;

    int HL = kHeads;  // dcp_size=1, so all heads local
    int qk_dim = kQkNope + kQkRope;

    for (size_t i = 0; i < log.size(); ++i) {
        if (log[i] == "rmsnorm(tokens=2,dim=" + std::to_string(kHidden) +
                       ",stride=" + std::to_string(kHidden) + ")" &&
            idx_input_norm < 0) {
            found_input_norm = true;
            idx_input_norm = static_cast<int>(i);
        }
        if (log[i] == "quant_fp8(M=2,K=" + std::to_string(kHidden) + ")") {
            found_hidden_quant = true;
            if (idx_hidden_quant < 0) idx_hidden_quant = static_cast<int>(i);
        }
        if (log[i] == "gemm(M=2,N=" + std::to_string(kQLora) +
                       ",K=" + std::to_string(kHidden) + ")")
            found_q_a_gemm = true;
        if (log[i] == "rmsnorm(tokens=2,dim=" + std::to_string(kQLora) +
                       ",stride=" + std::to_string(kQLora) + ")")
            found_q_a_norm = true;
        if (log[i] == "gemm(M=2,N=" + std::to_string(HL * qk_dim) +
                       ",K=" + std::to_string(kQLora) + ")")
            found_q_b_gemm = true;
        if (log[i] == "gemm(M=2,N=" + std::to_string(kKVLora + kQkRope) +
                       ",K=" + std::to_string(kHidden) + ")")
            found_kv_a_gemm = true;
        if (log[i] == "rmsnorm(tokens=2,dim=" + std::to_string(kKVLora) +
                       ",stride=" + std::to_string(kKVLora + kQkRope) + ")")
            found_kv_a_norm = true;
        if (log[i] == "gemm(M=2,N=" + std::to_string(kHidden) +
                       ",K=" + std::to_string(HL * kVHeadDim) + ")")
            found_oproj_gemm = true;
    }

    EXPECT_TRUE(found_input_norm) << "Missing input_layernorm";
    EXPECT_TRUE(found_hidden_quant) << "Missing hidden quantization";
    EXPECT_TRUE(found_q_a_gemm) << "Missing q_a_proj GEMM";
    EXPECT_TRUE(found_q_a_norm) << "Missing q_a_layernorm";
    EXPECT_TRUE(found_q_b_gemm) << "Missing q_b_proj GEMM";
    EXPECT_TRUE(found_kv_a_gemm) << "Missing kv_a_proj GEMM";
    EXPECT_TRUE(found_kv_a_norm) << "Missing kv_a_layernorm";
    EXPECT_TRUE(found_oproj_gemm) << "Missing o_proj GEMM";

    if (idx_input_norm >= 0 && idx_hidden_quant >= 0) {
        EXPECT_LT(idx_input_norm, idx_hidden_quant)
            << "input_layernorm must precede FP8 quantization";
    }
}

// ── GG-4: GGUF attention GEMM routing ───────────────────────────────────────
//
// Verify the dcp_executor routing-by-(strategy, M) for GGUF-quantized
// projections (q_a/q_b/kv_a/o_proj), the per-projection type selection, and
// that non-GGUF projections still take the FP8 gemm path. Uses the recording
// device (CPU-only; no kernel runs).

namespace {

// Tag the four plain projections in a per-rank weights vector as GGUF with the
// given per-projection k-quant types. kv_b_proj is intentionally left non-GGUF
// (it is consumed by q_absorb, GG-7).
void set_gguf_projections(std::vector<lp::AttentionLayerWeights>& weights,
                          layerstorm::model::GgufKQuantType q_a,
                          layerstorm::model::GgufKQuantType q_b,
                          layerstorm::model::GgufKQuantType kv_a,
                          layerstorm::model::GgufKQuantType o) {
    for (auto& w : weights) {
        w.q_a_is_gguf = true;    w.q_a_gguf_type = q_a;
        w.q_b_is_gguf = true;    w.q_b_gguf_type = q_b;
        w.kv_a_is_gguf = true;   w.kv_a_gguf_type = kv_a;
        w.o_proj_is_gguf = true; w.o_proj_gguf_type = o;
    }
}

// Count log entries whose text begins with `prefix`.
int count_prefix(const std::vector<std::string>& log, const std::string& prefix) {
    int n = 0;
    for (const auto& s : log)
        if (s.rfind(prefix, 0) == 0) ++n;
    return n;
}

}  // namespace

TEST(DcpExecutor, GgufRoutingIntDecodeUsesMmvq) {
    // int strategy, M ≤ 8 (decode) → all four projections go through gguf_mmvq;
    // NO FP8 gemm and NO gguf_mmq/dequant. Per-projection types preserved.
    using GT = layerstorm::model::GgufKQuantType;
    std::vector<std::string> log;

    auto opts = executor_opts(1);
    opts.gguf_active = true;
    opts.gguf_strategy = layerstorm::config::GgufStrategy::int_strategy;
    RecordingBackends rec(1, log);
    rec.apply(opts);

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());

    TestExecParamsBuilder builder(1, /*batch=*/2);  // M=2 ≤ 8 → decode
    set_gguf_projections(builder.weights_data,
                         GT::Q4_K, GT::Q6_K, GT::Q5_K, GT::Q8_0);
    log.clear();
    auto params = builder.build(/*use_graph=*/false);
    exec.execute_attention(params);

    EXPECT_EQ(count_prefix(log, "gemm("), 0) << "no FP8 gemm for GGUF projections";
    EXPECT_EQ(count_prefix(log, "gguf_mmq("), 0);
    EXPECT_EQ(count_prefix(log, "gguf_dequant_gemm("), 0);
    EXPECT_EQ(count_prefix(log, "gguf_mmvq("), 4)
        << "q_a + q_b + kv_a + o_proj all routed to mmvq";

    const int HL = kHeads;  // dcp_size=1
    const int qk = kQkNope + kQkRope;
    // Per-projection N/K/type checks (verify the right type reaches each call).
    EXPECT_EQ(count_prefix(log, "gguf_mmvq(M=2,N=" + std::to_string(kQLora) +
                                ",K=" + std::to_string(kHidden) +
                                ",t=" + std::to_string((int)GT::Q4_K) + ")"), 1)
        << "q_a_proj: Q4_K";
    EXPECT_EQ(count_prefix(log, "gguf_mmvq(M=2,N=" + std::to_string(HL * qk) +
                                ",K=" + std::to_string(kQLora) +
                                ",t=" + std::to_string((int)GT::Q6_K) + ")"), 1)
        << "q_b_proj: Q6_K";
    EXPECT_EQ(count_prefix(log, "gguf_mmvq(M=2,N=" + std::to_string(kKVLora + kQkRope) +
                                ",K=" + std::to_string(kHidden) +
                                ",t=" + std::to_string((int)GT::Q5_K) + ")"), 1)
        << "kv_a_proj: Q5_K";
    EXPECT_EQ(count_prefix(log, "gguf_mmvq(M=2,N=" + std::to_string(kHidden) +
                                ",K=" + std::to_string(HL * kVHeadDim) +
                                ",t=" + std::to_string((int)GT::Q8_0) + ")"), 1)
        << "o_proj: Q8_0";
}

TEST(DcpExecutor, GgufRoutingIntPrefillUsesMmq) {
    // int strategy, M > 8 (prefill) → all four projections go through gguf_mmq.
    using GT = layerstorm::model::GgufKQuantType;
    std::vector<std::string> log;

    auto opts = executor_opts(1);
    opts.gguf_active = true;
    opts.gguf_strategy = layerstorm::config::GgufStrategy::int_strategy;
    opts.max_batch_size = 16;  // allow batch > 8
    RecordingBackends rec(1, log);
    rec.apply(opts);

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());

    TestExecParamsBuilder builder(1, /*batch=*/12);  // M=12 > 8 → prefill
    set_gguf_projections(builder.weights_data,
                         GT::Q4_K, GT::Q4_K, GT::Q4_K, GT::Q4_K);
    log.clear();
    auto params = builder.build(/*use_graph=*/false);
    exec.execute_attention(params);

    EXPECT_EQ(count_prefix(log, "gemm("), 0);
    EXPECT_EQ(count_prefix(log, "gguf_mmvq("), 0);
    EXPECT_EQ(count_prefix(log, "gguf_dequant_gemm("), 0);
    EXPECT_EQ(count_prefix(log, "gguf_mmq("), 4);
}

TEST(DcpExecutor, GgufRoutingDequantUsesDequantGemm) {
    // dequant strategy → gguf_dequant_gemm regardless of M; no Q8_1 workspace.
    using GT = layerstorm::model::GgufKQuantType;
    std::vector<std::string> log;

    auto opts = executor_opts(1);
    opts.gguf_active = true;
    opts.gguf_strategy = layerstorm::config::GgufStrategy::dequant;
    RecordingBackends rec(1, log);
    rec.apply(opts);

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());

    TestExecParamsBuilder builder(1, /*batch=*/2);
    set_gguf_projections(builder.weights_data,
                         GT::Q2_K, GT::Q3_K, GT::Q4_K, GT::Q5_K);
    log.clear();
    auto params = builder.build(/*use_graph=*/false);
    exec.execute_attention(params);

    EXPECT_EQ(count_prefix(log, "gemm("), 0);
    EXPECT_EQ(count_prefix(log, "gguf_mmvq("), 0);
    EXPECT_EQ(count_prefix(log, "gguf_mmq("), 0);
    EXPECT_EQ(count_prefix(log, "gguf_dequant_gemm("), 4);
}

TEST(DcpExecutor, GgufRoutingMixedWithFp8Projections) {
    // Only q_a + o_proj are GGUF; q_b + kv_a stay FP8. Verify the executor mixes
    // both paths and uses the FP8 gemm for the non-GGUF projections.
    using GT = layerstorm::model::GgufKQuantType;
    std::vector<std::string> log;

    auto opts = executor_opts(1);
    opts.gguf_active = true;
    opts.gguf_strategy = layerstorm::config::GgufStrategy::int_strategy;
    RecordingBackends rec(1, log);
    rec.apply(opts);

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());

    TestExecParamsBuilder builder(1, /*batch=*/2);
    for (auto& w : builder.weights_data) {
        w.q_a_is_gguf = true;    w.q_a_gguf_type = GT::Q4_K;
        w.o_proj_is_gguf = true; w.o_proj_gguf_type = GT::Q6_K;
        // q_b + kv_a left non-GGUF → FP8 gemm.
    }
    log.clear();
    auto params = builder.build(/*use_graph=*/false);
    exec.execute_attention(params);

    // 2 GGUF projections via mmvq, 2 FP8 projections (q_b, kv_a) via gemm.
    EXPECT_EQ(count_prefix(log, "gguf_mmvq("), 2);
    EXPECT_EQ(count_prefix(log, "gemm("), 2)
        << "q_b + kv_a still take the FP8 GEMM path";
}

// ── GG-7: GGUF kv_b (q_absorb) L%QK guard ───────────────────────────────────
//
// kv_b_proj is consumed in-kernel by q_absorb's GGUF dequant branch, which
// indexes blocks_per_row = kv_lora_rank / QK and therefore requires
// kv_lora_rank % QK == 0 (no partial super-block on the latent axis). The test
// config has kv_lora_rank = 32, so a k-quant (QK=256) MUST trip the executor's
// guard, while Q8_0 (QK=32, 32%32==0) MUST pass. Both are CPU-only (recording
// device; no kernel runs) — the guard throws before absorb_q is dispatched.
TEST(DcpExecutor, GgufKvBAbsorbGuardThrowsOnBadShape) {
    using GT = layerstorm::model::GgufKQuantType;
    std::vector<std::string> log;

    auto opts = executor_opts(1);
    opts.gguf_active = true;
    opts.gguf_strategy = layerstorm::config::GgufStrategy::dequant;
    RecordingBackends rec(1, log);
    rec.apply(opts);

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());

    TestExecParamsBuilder builder(1, /*batch=*/2);
    for (auto& w : builder.weights_data) {
        // kv_lora_rank (kKVLora) == 32; Q4_K's QK == 256 ⇒ 32 % 256 != 0.
        w.kv_b_is_gguf = true;
        w.kv_b_gguf_type = GT::Q4_K;
    }
    auto params = builder.build(/*use_graph=*/false);
    EXPECT_THROW(exec.execute_attention(params), std::runtime_error)
        << "GGUF kv_b q_absorb must reject kv_lora_rank % QK != 0";
}

TEST(DcpExecutor, GgufKvBAbsorbGuardAcceptsDivisibleShape) {
    using GT = layerstorm::model::GgufKQuantType;
    std::vector<std::string> log;

    auto opts = executor_opts(1);
    opts.gguf_active = true;
    opts.gguf_strategy = layerstorm::config::GgufStrategy::dequant;
    RecordingBackends rec(1, log);
    rec.apply(opts);

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());

    TestExecParamsBuilder builder(1, /*batch=*/2);
    for (auto& w : builder.weights_data) {
        // Q8_0's QK == 32 ⇒ 32 % 32 == 0: the guard must NOT fire.
        w.kv_b_is_gguf = true;
        w.kv_b_gguf_type = GT::Q8_0;
    }
    auto params = builder.build(/*use_graph=*/false);
    EXPECT_NO_THROW(exec.execute_attention(params))
        << "GGUF kv_b q_absorb must accept kv_lora_rank % QK == 0";
}

TEST(DcpExecutor, PrefillCallSequenceDcpSize2) {
    // DCP=2: verify DCP correction calls appear between attention and o_proj.
    std::vector<std::string> log;

    RecordingCollectiveBackend rec_collective(log, "nccl_");
    lp::DcpCommunicator::Options comm_opts = null_comm_opts(2);
    comm_opts.collective = &rec_collective;
    auto comm = lp::DcpCommunicator(comm_opts);

    auto opts = executor_opts(2);
    opts.dcp_kv_sharded = true;  // exercises the KV-SHARDED DCP combine
    RecordingBackends rec(2, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads / 2, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         // INV-KVS-QAG: sharded-KV combine runs over ALL heads.
         .combine_num_heads = kHeads,
         .gpus = make_gpu_refs(2)},
        opts.attention_devices, recording_correction_fn(log));
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());
    log.clear();

    TestExecParamsBuilder builder(2, 2);
    auto params = builder.build(/*use_graph=*/false);
    exec.execute_attention(params);

    // Find the indices of key events
    int first_correction_idx = -1;
    int first_oproj_gemm_idx = -1;
    int first_nccl_allgather_idx = -1;
    int last_nccl_allreduce_idx = -1;

    int HL = kHeads / 2;

    for (size_t i = 0; i < log.size(); ++i) {
        if (log[i].find("correction(rank=") != std::string::npos &&
            first_correction_idx < 0) {
            first_correction_idx = static_cast<int>(i);
        }
        if (log[i] == "gemm(M=2,N=" + std::to_string(kHidden) +
                       ",K=" + std::to_string(HL * kVHeadDim) + ")" &&
            first_oproj_gemm_idx < 0) {
            first_oproj_gemm_idx = static_cast<int>(i);
        }
        if (log[i] == "nccl_allgather" && first_nccl_allgather_idx < 0)
            first_nccl_allgather_idx = static_cast<int>(i);
        if (log[i] == "nccl_allreduce")
            last_nccl_allreduce_idx = static_cast<int>(i);
    }

    // DCP correction (allgather + correction + allreduce) must come before o_proj
    EXPECT_GE(first_nccl_allgather_idx, 0) << "Expected NCCL allgather call";
    EXPECT_GE(first_correction_idx, 0) << "Expected DCP correction kernel call";
    EXPECT_GE(first_oproj_gemm_idx, 0) << "Expected o_proj GEMM call";

    if (first_correction_idx >= 0 && first_oproj_gemm_idx >= 0) {
        EXPECT_LT(first_correction_idx, first_oproj_gemm_idx)
            << "DCP correction must precede o_proj GEMM (INV-DCP-8)";
    }

    // TP allreduce (reduce_hidden) must come after o_proj
    EXPECT_GT(last_nccl_allreduce_idx, first_oproj_gemm_idx)
        << "TP allreduce must follow o_proj GEMM";

    // input_layernorm must be called (dim=kHidden, distinct from q_a/kv_a norms)
    bool found_input_norm = false;
    for (const auto& entry : log) {
        if (entry == "rmsnorm(tokens=2,dim=" + std::to_string(kHidden) +
                     ",stride=" + std::to_string(kHidden) + ")") {
            found_input_norm = true;
            break;
        }
    }
    EXPECT_TRUE(found_input_norm) << "Missing input_layernorm for DCP size 2";
}

TEST(DcpExecutor, DcpSize1SkipsCommunication) {
    std::vector<std::string> log;

    RecordingCollectiveBackend rec_collective(log, "nccl_");
    lp::DcpCommunicator::Options comm_opts = null_comm_opts(1);
    comm_opts.collective = &rec_collective;
    auto comm = lp::DcpCommunicator(comm_opts);

    auto opts = executor_opts(1);
    RecordingBackends rec(1, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .gpus = make_gpu_refs(1)},
        opts.attention_devices, recording_correction_fn(log));
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());
    log.clear();

    TestExecParamsBuilder builder(1, 2);
    auto params = builder.build(/*use_graph=*/false);
    exec.execute_attention(params);

    // No NCCL calls should appear
    for (const auto& entry : log) {
        EXPECT_TRUE(entry.find("nccl_") == std::string::npos)
            << "Unexpected NCCL call for dcp_size=1: " << entry;
        EXPECT_TRUE(entry.find("correction(") == std::string::npos)
            << "Unexpected correction call for dcp_size=1: " << entry;
    }
}

TEST(DcpExecutor, CommonPrefixPerRank) {
    // DCP=2: verify both ranks get their own set_device + projection calls.
    std::vector<std::string> log;

    auto comm = lp::DcpCommunicator(null_comm_opts(2));
    auto opts = executor_opts(2);
    RecordingBackends rec(2, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads / 2, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .gpus = make_gpu_refs(2)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());
    log.clear();

    TestExecParamsBuilder builder(2, 1);
    auto params = builder.build(/*use_graph=*/false);
    exec.execute_attention(params);

    // Count set_device calls per GPU
    int dev0_count = 0, dev1_count = 0;
    for (const auto& entry : log) {
        if (entry == "set_device(0)") ++dev0_count;
        if (entry == "set_device(1)") ++dev1_count;
    }

    // Both GPUs must be addressed (common prefix + non-graph + o_proj)
    EXPECT_GE(dev0_count, 2) << "GPU 0 should have multiple set_device calls";
    EXPECT_GE(dev1_count, 2) << "GPU 1 should have multiple set_device calls";
}

TEST(DcpExecutor, PrefillCallbackInvoked) {
    // Non-graph mode should call prefill_attention for each rank.
    std::vector<std::string> log;

    auto comm = lp::DcpCommunicator(null_comm_opts(1));
    auto opts = executor_opts(1);
    RecordingBackends rec(1, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .gpus = make_gpu_refs(1)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());
    log.clear();

    TestExecParamsBuilder builder(1, 3);
    auto params = builder.build(/*use_graph=*/false);
    exec.execute_attention(params);

    bool found_prefill = false;
    for (const auto& entry : log) {
        if (entry == "prefill(B=3,skv=3,sparse=0,causal=0)") {
            found_prefill = true;
        }
    }
    EXPECT_TRUE(found_prefill) << "Expected prefill_attention callback for non-graph mode";
}

TEST(DcpExecutor, PrefillCallbackSparse) {
    // Sparse mode should pass is_sparse=true to prefill callback.
    std::vector<std::string> log;

    auto comm = lp::DcpCommunicator(null_comm_opts(1));
    auto opts = executor_opts(1);
    RecordingBackends rec(1, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .gpus = make_gpu_refs(1)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());
    log.clear();

    TestExecParamsBuilder builder(1, 2);
    auto params = builder.build(/*use_graph=*/false, /*sparse=*/true);
    exec.execute_attention(params);

    bool found_sparse = false;
    for (const auto& entry : log) {
        if (entry == "prefill(B=2,skv=2,sparse=1,causal=0)") {
            found_sparse = true;
        }
    }
    EXPECT_TRUE(found_sparse) << "Expected sparse prefill_attention callback";
}

TEST(DcpExecutor, ChunkStepSingleBatchedCausalCall) {
    // TD-PREFILL-CHUNK-ATTN perf fix: a multi-token prefill chunk must issue
    // exactly ONE batched prefill_attention call per rank with
    // chunk_causal=true and seq_len_kv = max_seqlen_k (union prefix), NOT the
    // retired B per-row calls (O(B·s_kv) staging) and NOT a flat non-causal
    // batched call.
    std::vector<std::string> log;

    auto comm = lp::DcpCommunicator(null_comm_opts(1));
    auto opts = executor_opts(1);
    RecordingBackends rec(1, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .gpus = make_gpu_refs(1)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());
    log.clear();

    TestExecParamsBuilder builder(1, 3);
    auto params = builder.build(/*use_graph=*/false);
    // Chunk descriptor: 3 consecutive positions of one sequence starting at
    // token_pos 4 → per-row visible KV prefixes {5, 6, 7}, union = 7.
    std::vector<int> host_seqlens{5, 6, 7};
    params.host_seqlens_k = host_seqlens.data();
    params.chunk_start = 4;
    params.chunk_len = 3;
    params.max_seqlen_k = 7;
    exec.execute_attention(params);

    int prefill_count = 0;
    bool found_causal = false;
    for (const auto& entry : log) {
        if (entry.find("prefill(") != std::string::npos) ++prefill_count;
        if (entry == "prefill(B=3,skv=7,sparse=0,causal=1)")
            found_causal = true;
    }
    EXPECT_EQ(prefill_count, 1)
        << "Chunk step must be ONE batched causal call, not per-row calls";
    EXPECT_TRUE(found_causal)
        << "Expected prefill(B=3,skv=7,sparse=0,causal=1) in log";
}

// TD-SPARSE-CHUNK-PREFILL executor wiring: on a dispatcher-blessed prefill
// chunk of a DSA model, Options::sparse_prefill=true must (a) run the
// producer's per-row score/top-k with each row's OWN bound + causal cutoff
// (nb = len_b, qpos = len_b − 1) and (b) issue ONE batched prefill call with
// is_sparse=1 AND chunk_causal=1. With the gate OFF the identical step stays
// the dense chunk path (sparse=0), byte-identical legacy behavior.
static void run_sparse_chunk_prefill_case(bool gate_on,
                                          std::vector<std::string>& log) {
    auto comm = lp::DcpCommunicator(null_comm_opts(1));
    auto opts = executor_opts(1);
    opts.has_dsa = true;
    opts.index_topk = 8;
    opts.index_n_heads = 4;
    opts.index_head_dim = 16;
    opts.sparse_prefill = gate_on;
    RecordingBackends rec(1, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .gpus = make_gpu_refs(1)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());

    TestExecParamsBuilder builder(1, 3);
    // Indexer weights present (never dereferenced by the recorder).
    for (auto& w : builder.weights_data) {
        w.q_idx_b = reinterpret_cast<const void*>(0x10);
        w.k_idx = reinterpret_cast<const void*>(0x20);
        w.k_idx_norm = reinterpret_cast<const void*>(0x30);
        w.k_idx_norm_bias = reinterpret_cast<const void*>(0x40);
        w.weights_proj = reinterpret_cast<const void*>(0x50);
    }
    log.clear();

    auto params = builder.build(/*use_graph=*/false);
    // Blessed chunk: 3 consecutive positions starting at token_pos 4 —
    // per-row prefixes {5, 6, 7} (arena storage: no indexer_k_pages).
    std::vector<int> host_seqlens{5, 6, 7};
    params.host_seqlens_k = host_seqlens.data();
    params.chunk_start = 4;
    params.chunk_len = 3;
    params.max_seqlen_k = 7;
    params.indexer_step_key = 0xC0FFEE;
    params.indexer_prefill_append = true;
    exec.execute_attention(params);
}

TEST(DcpExecutor, SparseChunkPrefillGateOn) {
    std::vector<std::string> log;
    run_sparse_chunk_prefill_case(/*gate_on=*/true, log);

    // TD-SPARSE-PREFILL-SCORE-BATCH: ONE batched score/top-k launch pair for
    // the whole chunk — with each row keeping its OWN bound (nb = len_b) and
    // causal cutoff (len_b − 1), exactly the shapes the retired per-row loop
    // issued (B per-row calls).
    int topk_calls = 0, batched_calls = 0;
    bool batched_rows = false;
    int prefill_count = 0;
    bool found_sparse_causal = false;
    for (const auto& entry : log) {
        if (entry.find("idx_topk(") != std::string::npos) ++topk_calls;
        if (entry.find("idx_topk_batched(") != std::string::npos)
            ++batched_calls;
        if (entry == "idx_topk_batched(rows=3,nb=[5,6,7],qpos=[4,5,6],paged=0)")
            batched_rows = true;
        if (entry.find("prefill(") != std::string::npos) ++prefill_count;
        if (entry == "prefill(B=3,skv=7,sparse=1,causal=1)")
            found_sparse_causal = true;
    }
    EXPECT_EQ(batched_calls, 1) << "ONE batched score/top-k pair per chunk";
    EXPECT_EQ(topk_calls, 0) << "no per-row score/top-k launches";
    EXPECT_TRUE(batched_rows)
        << "each chunk row must keep ITS OWN causal bound + cutoff";
    EXPECT_EQ(prefill_count, 1) << "still ONE batched chunk call";
    EXPECT_TRUE(found_sparse_causal)
        << "Expected prefill(B=3,skv=7,sparse=1,causal=1) in log";
}

TEST(DcpExecutor, SparseChunkPrefillGateOffStaysDense) {
    std::vector<std::string> log;
    run_sparse_chunk_prefill_case(/*gate_on=*/false, log);

    bool found_dense_causal = false;
    int topk_calls = 0;
    for (const auto& entry : log) {
        if (entry.find("idx_topk(") != std::string::npos
            || entry.find("idx_topk_batched(") != std::string::npos)
            ++topk_calls;
        if (entry == "prefill(B=3,skv=7,sparse=0,causal=1)")
            found_dense_causal = true;
    }
    EXPECT_EQ(topk_calls, 0) << "gate off: no producer scoring on chunks";
    EXPECT_TRUE(found_dense_causal)
        << "gate off must keep the DENSE chunk-causal call";
}

// TD-SPARSE-PREFILL-LOCAL-INDEXER executor wiring: on a dispatcher-blessed
// prefill chunk under dcp_indexer_mode=local (dcp=2, paged indexer-K), the
// producer must (a) score each rank's shard with the PER-ROW owned_len
// bound — NOT len_b (the replicated bound) and NOT the full stored shard
// (which already holds the chunk's later keys — the exactness hazard) — at
// the row's own causal cutoff, (b) run the cross-rank merge PER ROW with
// the row's GLOBAL bound/cutoff on every rank, and (c) still consume the
// chunk SPARSE chunk-causal in ONE batched call.
TEST(DcpExecutor, SparseChunkPrefillLocalIndexerMergesPerRow) {
    std::vector<std::string> log;
    auto comm = lp::DcpCommunicator(null_comm_opts(2));
    auto opts = executor_opts(2);
    opts.has_dsa = true;
    opts.index_topk = 8;
    opts.index_n_heads = 4;
    opts.index_head_dim = 16;
    opts.sparse_prefill = true;
    opts.indexer_local = true;
    // PT=4, dcp=2: indexer pages 0,2,… → rank 0; 1,3,… → rank 1.
    opts.indexer_k_page_tokens = 4;
    RecordingBackends rec(2, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads / 2, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .gpus = make_gpu_refs(2)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());

    TestExecParamsBuilder builder(2, 3);
    for (auto& w : builder.weights_data) {
        w.q_idx_b = reinterpret_cast<const void*>(0x10);
        w.k_idx = reinterpret_cast<const void*>(0x20);
        w.k_idx_norm = reinterpret_cast<const void*>(0x30);
        w.k_idx_norm_bias = reinterpret_cast<const void*>(0x40);
        w.weights_proj = reinterpret_cast<const void*>(0x50);
    }

    // Paged indexer-K host tables (local mode is paged-only, LOCAL-compacted
    // slots): [batch * batch_stride + layer * page_stride + slot], layer 0
    // only. Fake distinct device pointers — never dereferenced (recorder
    // kernels are no-ops).
    constexpr int kPageStride = 2;
    std::vector<std::vector<const void*>> tables(2);
    std::vector<const void* const*> table_bases(2);
    for (int r = 0; r < 2; ++r) {
        tables[r].resize(3 * kPageStride);
        for (size_t s = 0; s < tables[r].size(); ++s)
            tables[r][s] = reinterpret_cast<const void*>(
                0x1000 + r * 0x100 + s * 0x10);
        table_bases[r] = tables[r].data();
    }

    log.clear();
    auto params = builder.build(/*use_graph=*/false);
    // Blessed chunk: 3 consecutive positions starting at token_pos 4 —
    // per-row prefixes {5, 6, 7}. Shard math at PT=4: rank 0 owns page 0
    // (positions 0-3) → owned_len = 4 for every row; rank 1 owns page 1
    // (positions 4-6 stored so far) → owned_len = {1, 2, 3}.
    std::vector<int> host_seqlens{5, 6, 7};
    params.host_seqlens_k = host_seqlens.data();
    params.chunk_start = 4;
    params.chunk_len = 3;
    params.max_seqlen_k = 7;
    params.indexer_step_key = 0xC0FFEE;
    params.indexer_prefill_append = true;
    params.indexer_k_pages = table_bases.data();
    params.indexer_k_page_stride = kPageStride;
    params.indexer_k_batch_stride = kPageStride;  // single layer row
    params.indexer_k_page_tokens = 4;
    exec.execute_attention(params);

    auto count = [&](const std::string& s) {
        return std::count(log.begin(), log.end(), s);
    };
    // (a) Per-rank shard scoring at the PER-ROW owned_len bound — now ONE
    // batched launch pair per rank (TD-SPARSE-PREFILL-SCORE-BATCH), each row
    // keeping its own owned_len bound + causal cutoff (the exact shapes the
    // retired per-row loop issued).
    EXPECT_EQ(
        count("idx_topk_batched(rows=3,nb=[4,4,4],qpos=[4,5,6],paged=1)"), 1)
        << "rank 0: per-row owned_len bounds {4,4,4}";
    EXPECT_EQ(
        count("idx_topk_batched(rows=3,nb=[1,2,3],qpos=[4,5,6],paged=1)"), 1)
        << "rank 1: per-row owned_len bounds {1,2,3}";
    int topk_calls = 0, batched_calls = 0, merge_calls = 0;
    for (const auto& e : log) {
        if (e.find("idx_topk(") != std::string::npos) ++topk_calls;
        if (e.find("idx_topk_batched(") != std::string::npos) ++batched_calls;
        if (e.find("idx_merge(") != std::string::npos) ++merge_calls;
    }
    EXPECT_EQ(batched_calls, 2) << "one batched shard score/top-k per rank";
    EXPECT_EQ(topk_calls, 0) << "no per-row score/top-k launches";
    // (b) Per-row merge on EVERY rank at the row's GLOBAL bound/cutoff.
    EXPECT_EQ(count("idx_merge(tok=0,nb=5,qpos=4)"), 2);
    EXPECT_EQ(count("idx_merge(tok=1,nb=6,qpos=5)"), 2);
    EXPECT_EQ(count("idx_merge(tok=2,nb=7,qpos=6)"), 2);
    EXPECT_EQ(merge_calls, 6) << "one merge per rank per chunk row";
    // (c) ONE batched sparse chunk-causal consumer call per rank.
    EXPECT_EQ(count("prefill(B=3,skv=7,sparse=1,causal=1)"), 2)
        << "chunk must consume SPARSE chunk-causal on both ranks";
}

// ── TD-KVT-PREFILL: tiered SPARSE prefill chunks (B==1) ─────────────────────
// A blessed B==1 sparse prefill chunk is decode-shaped to the consumer
// (chunk_rows is false at B==1), so with a tiering hook set the executor
// must (a) call prepare() right after the per-row top-k is enqueued,
// (b) materialize the selection and consume the returned fake view
// (selection-only staging: skv == view rows, causal=0 — never the full
// union prefix), and (c) notify on_dense_layer() when the chunk's sparse
// production fails and it falls back to the dense chunk path (INV-KVT-2
// fail-loud seam).

namespace {

struct FakeTieringHook final : lp::KvTieringHook {
    int prepares = 0;
    int materializes = 0;
    int dense_notifies = 0;
    bool serve_tiered = true;
    int view_rows = 5;

    void prepare(int, int, const int*, const int*, int, void*,
                 bool selection_fresh) override {
        ++prepares;
        last_fresh = selection_fresh;
    }
    bool materialize(int, int, const int*, const int* topk_lengths_dev,
                     int, void*, lp::TieredKvView* out) override {
        ++materializes;
        if (!serve_tiered) return false;
        out->kv_cache = reinterpret_cast<void*>(0x1000);
        out->block_tables = reinterpret_cast<const int*>(0x2000);
        out->seqlens_k = topk_lengths_dev;
        out->max_blocks_per_seq = 1;
        out->seq_len_kv = view_rows;
        out->sparse_indices = reinterpret_cast<const int*>(0x3000);
        return true;
    }
    void on_dense_layer(int) override { ++dense_notifies; }

    bool last_fresh = false;
};

// One B==1 blessed sparse-prefill chunk step (token_pos 4 → prefix len 5)
// against a recording backend with `hook` attached. `with_q_weights` toggles
// the producer's Q-side weight gate (false → per-layer sparse production
// fails → dense chunk fallback).
void run_tiered_prefill_chunk_case(FakeTieringHook& hook,
                                   std::vector<std::string>& log,
                                   bool with_q_weights, int batch = 1) {
    auto comm = lp::DcpCommunicator(null_comm_opts(1));
    auto opts = executor_opts(1);
    opts.has_dsa = true;
    opts.index_topk = 8;
    opts.index_n_heads = 4;
    opts.index_head_dim = 16;
    opts.sparse_prefill = true;
    RecordingBackends rec(1, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .gpus = make_gpu_refs(1)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());

    TestExecParamsBuilder builder(1, batch);
    for (auto& w : builder.weights_data) {
        w.k_idx = reinterpret_cast<const void*>(0x20);
        w.k_idx_norm = reinterpret_cast<const void*>(0x30);
        w.k_idx_norm_bias = reinterpret_cast<const void*>(0x40);
        if (with_q_weights) {
            w.q_idx_b = reinterpret_cast<const void*>(0x10);
            w.weights_proj = reinterpret_cast<const void*>(0x50);
        }
    }
    log.clear();

    auto params = builder.build(/*use_graph=*/false);
    std::vector<int> host_seqlens(static_cast<size_t>(batch));
    for (int b = 0; b < batch; ++b) host_seqlens[static_cast<size_t>(b)] = 5 + b;
    params.host_seqlens_k = host_seqlens.data();
    params.chunk_start = 4;
    params.chunk_len = batch;
    params.max_seqlen_k = 4 + batch;
    params.indexer_step_key = 0xC0FFEE;
    params.indexer_prefill_append = true;
    params.kv_tiering = &hook;
    exec.execute_attention(params);
}

}  // namespace

TEST(DcpExecutor, SparseChunkPrefillTieredB1ConsumesTieredView) {
    FakeTieringHook hook;
    std::vector<std::string> log;
    run_tiered_prefill_chunk_case(hook, log, /*with_q_weights=*/true);

    EXPECT_EQ(hook.prepares, 1) << "prepare() must run right after the "
                                   "chunk's per-row top-k is enqueued";
    EXPECT_TRUE(hook.last_fresh) << "fresh (non-IndexShare-reuse) production";
    EXPECT_EQ(hook.materializes, 1);
    EXPECT_EQ(hook.dense_notifies, 0);
    bool found_tiered = false;
    int prefill_count = 0, per_row_topk = 0, batched_topk = 0;
    for (const auto& e : log) {
        if (e.find("prefill(") != std::string::npos) ++prefill_count;
        if (e == "prefill(B=1,skv=5,sparse=1,causal=0)") found_tiered = true;
        if (e.find("idx_topk(") != std::string::npos) ++per_row_topk;
        if (e.find("idx_topk_batched(") != std::string::npos) ++batched_topk;
    }
    // TD-SPARSE-PREFILL-SCORE-BATCH: B==1 has nothing to batch — the
    // per-row (single-launch-pair) path stays.
    EXPECT_EQ(per_row_topk, 1) << "B==1 chunk keeps the per-row producer";
    EXPECT_EQ(batched_topk, 0);
    EXPECT_EQ(prefill_count, 1);
    EXPECT_TRUE(found_tiered)
        << "tiered chunk must consume the fake view (skv = view rows, "
           "sparse, non-chunk-causal — selection-only staging)";
}

TEST(DcpExecutor, SparseChunkPrefillTieredNoColdKeepsOriginalPath) {
    FakeTieringHook hook;
    hook.serve_tiered = false;  // materialize: no cold pages in this layer
    std::vector<std::string> log;
    run_tiered_prefill_chunk_case(hook, log, /*with_q_weights=*/true);

    EXPECT_EQ(hook.materializes, 1);
    EXPECT_EQ(hook.dense_notifies, 0);
    bool found_orig = false;
    for (const auto& e : log)
        if (e == "prefill(B=1,skv=5,sparse=1,causal=0)") found_orig = true;
    EXPECT_TRUE(found_orig)
        << "materialize()==false must keep the original full-residency "
           "sparse call (B==1 chunk: skv = prefix len 5, causal=0)";
}

TEST(DcpExecutor, SparseChunkPrefillDenseFallbackNotifiesTieringHook) {
    FakeTieringHook hook;
    std::vector<std::string> log;
    run_tiered_prefill_chunk_case(hook, log, /*with_q_weights=*/false);

    EXPECT_EQ(hook.materializes, 0)
        << "dense chunk fallback must not materialize";
    EXPECT_EQ(hook.dense_notifies, 1)
        << "a tiered prefill chunk that fell back to DENSE must notify "
           "on_dense_layer (throws iff the layer has cold pages, INV-KVT-2)";
    bool found_dense = false;
    for (const auto& e : log)
        if (e == "prefill(B=1,skv=5,sparse=0,causal=0)") found_dense = true;
    EXPECT_TRUE(found_dense) << "gate-on production failure keeps the dense "
                                "B==1 chunk call";
}

TEST(DcpExecutor, SparseChunkPrefillTieredB3NeverMaterializes) {
    // Defensive: the dispatcher only sets kv_tiering on B==1 steps, but the
    // consumer's tiered branch must ALSO exclude B>1 chunk cohorts on its
    // own (chunk_rows) — a B>1 materialize would need per-row fake views
    // (TD-KVT-BATCH-COHORT).
    FakeTieringHook hook;
    std::vector<std::string> log;
    run_tiered_prefill_chunk_case(hook, log, /*with_q_weights=*/true,
                                  /*batch=*/3);

    EXPECT_EQ(hook.materializes, 0);
    bool found_sparse_causal = false;
    for (const auto& e : log)
        if (e == "prefill(B=3,skv=7,sparse=1,causal=1)")
            found_sparse_causal = true;
    EXPECT_TRUE(found_sparse_causal)
        << "B>1 chunk keeps the full-prefix sparse chunk-causal call";
}

// ── TD-GLM-INDEXER-B1CASCADE resolved (INV-DSA-ROWMIX): mixed cohorts ────────
// A B>1 decode cohort mixing a sparse-eligible row and a coverage-dead row
// (AttentionExecParams::indexer_row_dense) must (a) run the producer ONLY for
// the live row (its own bound + causal cutoff; the dead row is never appended
// or scored) and (b) split the consumer into per-row batch-of-1 sub-dispatches
// — SPARSE for the live row, DENSE for the dead row — instead of one flat
// batched call (invalid for multiple sequences, TD-DECODE-NONGRAPH-BATCH) or
// the retired whole-cohort dense suppression.

namespace {

// Builds a dcp=1 DSA executor + a blessed 2-row decode step (host prefixes
// {5, 7}, paged indexer-K tables for both rows) with `mask` as the per-row
// dense mask (nullptr = uniform sparse cohort). Storage for the tables /
// host lengths lives in the caller-provided vectors.
void run_mixed_cohort_case(std::vector<std::string>& log,
                           const uint8_t* mask,
                           std::vector<std::vector<const void*>>& tables,
                           std::vector<const void* const*>& table_bases,
                           std::vector<int>& host_seqlens) {
    auto comm = lp::DcpCommunicator(null_comm_opts(1));
    auto opts = executor_opts(1);
    opts.has_dsa = true;
    opts.index_topk = 8;
    opts.index_n_heads = 4;
    opts.index_head_dim = 16;
    RecordingBackends rec(1, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .gpus = make_gpu_refs(1)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());

    TestExecParamsBuilder builder(1, 2);
    for (auto& w : builder.weights_data) {
        w.q_idx_b = reinterpret_cast<const void*>(0x10);
        w.k_idx = reinterpret_cast<const void*>(0x20);
        w.k_idx_norm = reinterpret_cast<const void*>(0x30);
        w.k_idx_norm_bias = reinterpret_cast<const void*>(0x40);
        w.weights_proj = reinterpret_cast<const void*>(0x50);
    }
    // Per-seq prefixes {5, 7} at PT=4 → 2 pages per row; layer 0 only.
    constexpr int kPageStride = 2;
    tables[0].assign(2 * kPageStride, nullptr);
    for (size_t s = 0; s < tables[0].size(); ++s)
        tables[0][s] = reinterpret_cast<const void*>(0x1000 + s * 0x10);
    table_bases[0] = tables[0].data();
    host_seqlens = {5, 7};

    log.clear();
    auto params = builder.build(/*use_graph=*/false);
    params.host_seqlens_k = host_seqlens.data();
    params.max_seqlen_k = 7;
    params.indexer_step_key = 0xC0FFEE;
    params.indexer_k_pages = table_bases.data();
    params.indexer_k_page_stride = kPageStride;
    params.indexer_k_batch_stride = kPageStride;  // single layer row
    params.indexer_k_page_tokens = 4;
    params.indexer_row_dense = mask;
    exec.execute_attention(params);
}

}  // anonymous namespace

TEST(DcpExecutor, MixedCohortSplitsSparseAndDenseRows) {
    std::vector<std::string> log;
    std::vector<std::vector<const void*>> tables(1);
    std::vector<const void* const*> table_bases(1);
    std::vector<int> host_seqlens;
    const uint8_t mask[2] = {0, 1};  // row 0 live, row 1 coverage-dead
    run_mixed_cohort_case(log, mask, tables, table_bases, host_seqlens);

    auto count = [&](const std::string& s) {
        return std::count(log.begin(), log.end(), s);
    };
    // (a) Producer runs ONLY the live row (bound 5, cutoff 4).
    int topk_calls = 0, prefill_calls = 0;
    for (const auto& e : log) {
        if (e.find("idx_topk(") != std::string::npos) ++topk_calls;
        if (e.find("prefill(") != std::string::npos) ++prefill_calls;
    }
    EXPECT_EQ(topk_calls, 1) << "dead row must never be scored";
    EXPECT_EQ(count("idx_topk(nb=5,qpos=4)"), 1);
    // (b) Consumer splits per row: live row SPARSE at its own prefix, dead
    // row DENSE at its own prefix — each a batch-of-1 sub-dispatch.
    EXPECT_EQ(prefill_calls, 2) << "mixed cohort = one sub-dispatch per row";
    EXPECT_EQ(count("prefill(B=1,skv=5,sparse=1,causal=0)"), 1)
        << "live row must stay SPARSE (not silently dense)";
    EXPECT_EQ(count("prefill(B=1,skv=7,sparse=0,causal=0)"), 1)
        << "dead row must run DENSE over its own prefix";
}

TEST(DcpExecutor, UniformCohortKeepsSingleBatchedSparseCall) {
    // No mask (all rows sparse-eligible): legacy single batched sparse call —
    // the split engages ONLY on mixed cohorts.
    std::vector<std::string> log;
    std::vector<std::vector<const void*>> tables(1);
    std::vector<const void* const*> table_bases(1);
    std::vector<int> host_seqlens;
    run_mixed_cohort_case(log, /*mask=*/nullptr, tables, table_bases,
                          host_seqlens);

    auto count = [&](const std::string& s) {
        return std::count(log.begin(), log.end(), s);
    };
    int topk_calls = 0, prefill_calls = 0;
    for (const auto& e : log) {
        if (e.find("idx_topk(") != std::string::npos) ++topk_calls;
        if (e.find("prefill(") != std::string::npos) ++prefill_calls;
    }
    EXPECT_EQ(topk_calls, 2) << "both rows scored";
    EXPECT_EQ(prefill_calls, 1) << "uniform cohort keeps ONE batched call";
    EXPECT_EQ(count("prefill(B=2,skv=7,sparse=1,causal=0)"), 1);
}

// ── INV-DSA-ROWMIX scatter correctness: per-row pointer offsets ──────────────
// The split writes each row's output into its own slice of the batch buffers
// — bit-identical to running the sequences separately BY CONSTRUCTION iff
// every per-row pointer (q, seqlens, block table, indices, lengths, out, lse)
// is the batch pointer at row b's offset. Capture the exact pointers of both
// sub-dispatches and verify the offsets, then verify the content each
// sub-call wrote landed in the right row slice (deterministic per-(skv,
// sparse) fill) and matches a stand-alone batch-of-1 run of each sequence.

namespace {

class SplitCapturingAttentionDevice : public RecordingAttentionDevice {
public:
    struct Call {
        const void* q; int B; int skv;
        const int* sl; const int* bt;
        bool sparse;
        const int* si; const int* tl;
        void* out; float* lse;
    };
    SplitCapturingAttentionDevice(layerstorm::config::GpuRef gpu,
                                  std::vector<std::string>& log)
        : RecordingAttentionDevice(gpu, log) {}
    std::vector<Call> calls;

    void prefill_attention(const void* q, int batch_size, int seq_len_kv,
                           const int* seqlens_k, const int* block_tables,
                           int /*max_blocks_per_seq*/,
                           void* /*kv_cache*/, int64_t /*csb*/, int /*csr*/,
                           int /*page_size*/, bool is_sparse,
                           bool /*chunk_causal*/,
                           const int* sparse_indices, const int* topk_lengths,
                           int /*topk*/, void* out, float* lse,
                           int /*layer_idx*/, void* /*stream*/) override {
        calls.push_back({q, batch_size, seq_len_kv, seqlens_k, block_tables,
                         is_sparse, sparse_indices, topk_lengths, out, lse});
        // Deterministic fill of the EXACT region a real kernel would write:
        // [batch_size, kHeads * kKVLora] BF16 out + [batch_size, kHeads] lse,
        // value = f(skv, sparse, element index). device_alloc is malloc in
        // the recording backend, so these writes are plain host stores.
        auto* o = static_cast<uint16_t*>(out);
        const size_t n =
            static_cast<size_t>(batch_size) * kHeads * kKVLora;
        for (size_t i = 0; i < n; ++i)
            o[i] = static_cast<uint16_t>(
                (seq_len_kv * 131) ^ (is_sparse ? 0x4000 : 0) ^ (i & 0xFF));
        const size_t nl = static_cast<size_t>(batch_size) * kHeads;
        for (size_t i = 0; i < nl; ++i)
            lse[i] = static_cast<float>(seq_len_kv)
                   + (is_sparse ? 0.5f : 0.0f) + static_cast<float>(i);
    }
};

}  // anonymous namespace

TEST(DcpExecutor, MixedCohortSplitScatterBitIdenticalToSeparateRuns) {
    std::vector<std::string> log;
    auto comm = lp::DcpCommunicator(null_comm_opts(1));

    auto make_exec_opts = [&](SplitCapturingAttentionDevice& dev) {
        auto opts = executor_opts(1);
        opts.has_dsa = true;
        opts.index_topk = 8;
        opts.index_n_heads = 4;
        opts.index_head_dim = 16;
        opts.attention_devices = {&dev};
        opts.communicator = &comm;
        return opts;
    };
    auto gpu0 = layerstorm::config::GpuRef{
        .position = 0, .id = 0, .type = layerstorm::config::GpuType::rtx5090};

    // Shared paged indexer-K table (layer 0, PT=4, 2 pages per row).
    constexpr int kPageStride = 2;
    std::vector<std::vector<const void*>> tables(1);
    std::vector<const void* const*> table_bases(1);
    tables[0].assign(2 * kPageStride, nullptr);
    for (size_t s = 0; s < tables[0].size(); ++s)
        tables[0][s] = reinterpret_cast<const void*>(0x1000 + s * 0x10);
    table_bases[0] = tables[0].data();

    auto set_indexer_weights = [](TestExecParamsBuilder& b) {
        for (auto& w : b.weights_data) {
            w.q_idx_b = reinterpret_cast<const void*>(0x10);
            w.k_idx = reinterpret_cast<const void*>(0x20);
            w.k_idx_norm = reinterpret_cast<const void*>(0x30);
            w.k_idx_norm_bias = reinterpret_cast<const void*>(0x40);
            w.weights_proj = reinterpret_cast<const void*>(0x50);
        }
    };

    // ── Mixed cohort run: rows {seq A len 5 SPARSE, seq B len 7 DENSE} ──
    SplitCapturingAttentionDevice dev_mixed(gpu0, log);
    auto exec_mixed = lp::DcpExecutor(make_exec_opts(dev_mixed));
    init_dequant_pool(exec_mixed, 1);
    TestExecParamsBuilder bm(1, 2);
    set_indexer_weights(bm);
    std::vector<int> hs_mixed{5, 7};
    const uint8_t mask[2] = {0, 1};
    auto pm = bm.build(/*use_graph=*/false);
    pm.host_seqlens_k = hs_mixed.data();
    pm.max_seqlen_k = 7;
    pm.max_blocks_per_seq = 4;
    pm.indexer_step_key = 0xC0FFEE;
    pm.indexer_k_pages = table_bases.data();
    pm.indexer_k_page_stride = kPageStride;
    pm.indexer_k_batch_stride = kPageStride;
    pm.indexer_k_page_tokens = 4;
    pm.indexer_row_dense = mask;
    exec_mixed.execute_attention(pm);

    ASSERT_EQ(dev_mixed.calls.size(), 2u);
    const auto& c0 = dev_mixed.calls[0];
    const auto& c1 = dev_mixed.calls[1];
    EXPECT_EQ(c0.B, 1);
    EXPECT_EQ(c1.B, 1);
    EXPECT_TRUE(c0.sparse);
    EXPECT_FALSE(c1.sparse);
    EXPECT_EQ(c0.skv, 5);
    EXPECT_EQ(c1.skv, 7);

    // Per-row pointer offsets: row 1's pointers are the batch pointers at
    // exactly one row's stride (scatter by original row order).
    const int kQkHead = kKVLora + kQkRope;  // absorbed q row dim per head
    EXPECT_EQ(static_cast<const char*>(c1.q) - static_cast<const char*>(c0.q),
              static_cast<ptrdiff_t>(kHeads) * kQkHead * 2)
        << "q must advance one [H, d_qk] BF16 row";
    EXPECT_EQ(c1.sl - c0.sl, 1) << "device seqlens must advance one entry";
    EXPECT_EQ(c1.bt - c0.bt, 4) << "block table must advance one row "
                                   "(max_blocks_per_seq)";
    EXPECT_EQ(static_cast<char*>(c1.out) - static_cast<char*>(c0.out),
              static_cast<ptrdiff_t>(kHeads) * kKVLora * 2)
        << "out must advance one [H, d_c] BF16 row";
    EXPECT_EQ(c1.lse - c0.lse, kHeads) << "lse must advance one [H] row";
    EXPECT_NE(c0.si, nullptr) << "live row consumes its sparse indices";
    EXPECT_NE(c0.tl, nullptr);
    EXPECT_EQ(c1.si, nullptr) << "dense row must not see sparse indices";
    EXPECT_EQ(c1.tl, nullptr);

    // ── Separate stand-alone runs of each sequence ──
    // Sequence A alone: B==1 blessed sparse decode (same paged row).
    SplitCapturingAttentionDevice dev_a(gpu0, log);
    auto exec_a = lp::DcpExecutor(make_exec_opts(dev_a));
    init_dequant_pool(exec_a, 1);
    TestExecParamsBuilder ba(1, 1);
    set_indexer_weights(ba);
    std::vector<int> hs_a{5};
    auto pa = ba.build(/*use_graph=*/false);
    pa.host_seqlens_k = hs_a.data();
    pa.max_seqlen_k = 5;
    pa.max_blocks_per_seq = 4;
    pa.indexer_step_key = 0xC0FFEE;
    pa.indexer_k_pages = table_bases.data();
    pa.indexer_k_page_stride = kPageStride;
    pa.indexer_k_batch_stride = kPageStride;
    pa.indexer_k_page_tokens = 4;
    exec_a.execute_attention(pa);
    ASSERT_EQ(dev_a.calls.size(), 1u);
    EXPECT_TRUE(dev_a.calls[0].sparse);
    EXPECT_EQ(dev_a.calls[0].skv, 5);

    // Sequence B alone: B==1 coverage-dead decode (dispatcher would set
    // indexer_sparse_suppress) — plain dense call.
    SplitCapturingAttentionDevice dev_b(gpu0, log);
    auto exec_b = lp::DcpExecutor(make_exec_opts(dev_b));
    init_dequant_pool(exec_b, 1);
    TestExecParamsBuilder bb(1, 1);
    set_indexer_weights(bb);
    std::vector<int> hs_b{7};
    auto pb = bb.build(/*use_graph=*/false);
    pb.host_seqlens_k = hs_b.data();
    pb.max_seqlen_k = 7;
    pb.max_blocks_per_seq = 4;
    pb.indexer_sparse_suppress = true;  // coverage-dead at B==1
    exec_b.execute_attention(pb);
    ASSERT_EQ(dev_b.calls.size(), 1u);
    EXPECT_FALSE(dev_b.calls[0].sparse);
    EXPECT_EQ(dev_b.calls[0].skv, 7);

    // ── Bit-identity: each cohort row's out/lse slice equals the same
    // sequence's stand-alone run (the sub-call IS the B==1 call shape; the
    // deterministic fill is a pure function of the per-call inputs). ──
    const size_t out_row_bytes = static_cast<size_t>(kHeads) * kKVLora * 2;
    EXPECT_EQ(std::memcmp(c0.out, dev_a.calls[0].out, out_row_bytes), 0)
        << "live row slice must be bit-identical to its separate run";
    EXPECT_EQ(std::memcmp(c1.out, dev_b.calls[0].out, out_row_bytes), 0)
        << "dead row slice must be bit-identical to its separate run";
    EXPECT_EQ(std::memcmp(c0.lse, dev_a.calls[0].lse,
                          kHeads * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(c1.lse, dev_b.calls[0].lse,
                          kHeads * sizeof(float)), 0);
}

TEST(DcpExecutor, PrefillBeforeDcpCorrection) {
    // Verify prefill comes before DCP correction in the call sequence.
    std::vector<std::string> log;

    RecordingCollectiveBackend rec_collective(log, "nccl_");
    lp::DcpCommunicator::Options comm_opts = null_comm_opts(2);
    comm_opts.collective = &rec_collective;
    auto comm = lp::DcpCommunicator(comm_opts);

    auto opts = executor_opts(2);
    opts.dcp_kv_sharded = true;  // exercises the KV-SHARDED DCP combine
    RecordingBackends rec(2, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads / 2, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         // INV-KVS-QAG: sharded-KV combine runs over ALL heads.
         .combine_num_heads = kHeads,
         .gpus = make_gpu_refs(2)},
        opts.attention_devices, recording_correction_fn(log));
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());
    log.clear();

    TestExecParamsBuilder builder(2, 2);
    auto params = builder.build(/*use_graph=*/false);
    exec.execute_attention(params);

    // Find indices
    int first_prefill_idx = -1;
    int first_correction_idx = -1;
    int first_oproj_idx = -1;

    int HL = kHeads / 2;

    for (size_t i = 0; i < log.size(); ++i) {
        if (log[i].find("prefill(") != std::string::npos &&
            first_prefill_idx < 0)
            first_prefill_idx = static_cast<int>(i);
        if (log[i].find("correction(") != std::string::npos &&
            first_correction_idx < 0)
            first_correction_idx = static_cast<int>(i);
        if (log[i] == "gemm(M=2,N=" + std::to_string(kHidden) +
                       ",K=" + std::to_string(HL * kVHeadDim) + ")" &&
            first_oproj_idx < 0)
            first_oproj_idx = static_cast<int>(i);
    }

    EXPECT_GE(first_prefill_idx, 0) << "Expected prefill call";
    EXPECT_GE(first_correction_idx, 0) << "Expected DCP correction call";
    EXPECT_GE(first_oproj_idx, 0) << "Expected o_proj GEMM call";

    if (first_prefill_idx >= 0 && first_correction_idx >= 0) {
        EXPECT_LT(first_prefill_idx, first_correction_idx)
            << "Prefill must precede DCP correction";
    }
    if (first_correction_idx >= 0 && first_oproj_idx >= 0) {
        EXPECT_LT(first_correction_idx, first_oproj_idx)
            << "DCP correction must precede o_proj (INV-DCP-8)";
    }
}

TEST(DcpExecutor, PrefillPerRankDcpSize2) {
    // DCP=2: both ranks should get their own prefill callback.
    std::vector<std::string> log;

    auto comm = lp::DcpCommunicator(null_comm_opts(2));
    auto opts = executor_opts(2);
    RecordingBackends rec(2, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads / 2, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .gpus = make_gpu_refs(2)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());
    log.clear();

    TestExecParamsBuilder builder(2, 2);
    auto params = builder.build(/*use_graph=*/false);
    exec.execute_attention(params);

    int prefill_count = 0;
    for (const auto& entry : log) {
        if (entry.find("prefill(") != std::string::npos)
            ++prefill_count;
    }
    EXPECT_EQ(prefill_count, 2)
        << "Expected one prefill_attention call per rank for dcp_size=2";
}

// KVS-3 (TD-KV-SHARDED-EXEC): under sequence-sharded KV the executor must
//   (1) INV-KVS-POS — feed RoPE (q_absorb + step-5b k_pe rotate) the GLOBAL
//       seqlens array on EVERY rank (seqlens_k[r] is the rank-LOCAL shard
//       length there — position ≠ local length);
//   (2) bound each rank's KV gather/attend by its OWN local max
//       (host_local_seqlens_k[r]), including 0 for an EMPTY shard
//       (INV-KVS-EMPTY) — never the global max, which would overrun the
//       rank-local block table.
TEST(DcpExecutor, ShardedGlobalPositionsAndLocalKvBounds) {
    std::vector<std::string> log;

    auto comm = lp::DcpCommunicator(null_comm_opts(2));
    auto opts = executor_opts(2);
    RecordingBackends rec(2, log);
    rec.apply(opts);
    opts.communicator = &comm;
    opts.dcp_kv_sharded = true;
    // RoPE table so absorb_q / rope_rotate receive the position source.
    std::vector<float> rope_table(static_cast<size_t>(64) * kQkRope, 0.0f);
    opts.rope_cos_sin_host = rope_table.data();
    opts.rope_max_pos = 64;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads / 2, .head_dim = kKVLora,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .combine_num_heads = kHeads,  // INV-KVS-QAG all-head combine
         .gpus = make_gpu_refs(2)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());

    // One 5-token sequence at dcp_chunk_size=16: rank 0 owns positions 0-4
    // (local len 5), rank 1 owns NOTHING (local len 0 — the empty shard).
    TestExecParamsBuilder builder(2, /*batch=*/1);
    builder.seqlens_data[0][0] = 5;   // rank-local shard lengths
    builder.seqlens_data[1][0] = 0;
    std::vector<int> global_seqlens{5};
    std::vector<const int*> global_ptrs{global_seqlens.data(),
                                        global_seqlens.data()};
    std::vector<const int*> host_local_ptrs{builder.seqlens_data[0].data(),
                                            builder.seqlens_data[1].data()};
    auto params = builder.build(/*use_graph=*/false);
    params.max_seqlen_k = 5;  // global max (dispatcher-provided)
    params.global_seqlens_k = global_ptrs.data();
    params.host_local_seqlens_k = host_local_ptrs.data();

    log.clear();
    exec.execute_attention(params);

    // (1) Positions from the GLOBAL array on both ranks: q_absorb fused RoPE
    //     and the step-5b k_pe rotate.
    std::ostringstream qa_want, rr_want;
    qa_want << "absorb_q(seqlens="
            << static_cast<const void*>(global_seqlens.data()) << ",rope=1)";
    rr_want << "rope_rotate(seqlens="
            << static_cast<const void*>(global_seqlens.data()) << ")";
    EXPECT_EQ(count_prefix(log, qa_want.str()), 2)
        << "q_absorb RoPE must use GLOBAL positions on every rank";
    EXPECT_EQ(count_prefix(log, rr_want.str()), 2)
        << "step-5b k_pe rotate must use GLOBAL positions on every rank";
    // No call may consume the LOCAL arrays as positions.
    std::ostringstream local0, local1;
    local0 << "(seqlens=" << static_cast<const void*>(host_local_ptrs[0]);
    local1 << "(seqlens=" << static_cast<const void*>(host_local_ptrs[1]);
    for (const auto& e : log) {
        if (e.rfind("absorb_q", 0) == 0 || e.rfind("rope_rotate", 0) == 0) {
            EXPECT_EQ(e.find(local0.str()), std::string::npos) << e;
            EXPECT_EQ(e.find(local1.str()), std::string::npos) << e;
        }
    }

    // (2) Per-rank LOCAL KV bounds: rank 0 attends its 5 local tokens, rank 1
    //     runs the empty-shard shape (skv=0) — NOT the global max.
    EXPECT_EQ(count_prefix(log, "prefill(B=1,skv=5,"), 1)
        << "rank 0 must attend its local shard (5)";
    EXPECT_EQ(count_prefix(log, "prefill(B=1,skv=0,"), 1)
        << "rank 1 (empty shard) must attend 0 tokens, not the global max";
}

// INV-KVS-QAG (TD-KVS-Q-ALLGATHER fix): under sharded KV the executor must
// allgather the absorbed query in the HEAD dim BEFORE attention — otherwise
// no rank computes another rank's heads over its own token shard and the
// combine mixes different heads. Verifies:
//   (1) a Q allgather (one NCCL allgather per rank) precedes the first
//       prefill_attention call, and the LSE allgather follows it;
//   (2) at B > 1 the rank-major NCCL layout is rearranged token-major via
//       dcp strided copies per rank before attention.
TEST(DcpExecutor, ShardedQAllgatherBeforeAttention) {
    std::vector<std::string> log;

    RecordingCollectiveBackend rec_collective(log, "nccl_");
    lp::DcpCommunicator::Options comm_opts = null_comm_opts(2);
    comm_opts.collective = &rec_collective;
    auto comm = lp::DcpCommunicator(comm_opts);

    auto opts = executor_opts(2);
    opts.dcp_kv_sharded = true;
    RecordingBackends rec(2, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads / 2, .head_dim = kKVLora,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .combine_num_heads = kHeads,  // INV-KVS-QAG all-head combine
         .gpus = make_gpu_refs(2)},
        opts.attention_devices, recording_correction_fn(log));
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());

    // ── B == 1: NCCL rank-major recv IS head-major; no rearrange copies ──
    {
        TestExecParamsBuilder builder(2, /*batch=*/1);
        auto params = builder.build(/*use_graph=*/false);
        log.clear();
        exec.execute_attention(params);

        int first_prefill = -1, first_ag = -1;
        int ag_before_prefill = 0, ag_total = 0;
        for (size_t i = 0; i < log.size(); ++i) {
            if (log[i].rfind("prefill(", 0) == 0 && first_prefill < 0)
                first_prefill = static_cast<int>(i);
            if (log[i] == "nccl_allgather") {
                ++ag_total;
                if (first_ag < 0) first_ag = static_cast<int>(i);
                if (first_prefill < 0) ++ag_before_prefill;
            }
        }
        ASSERT_GE(first_prefill, 0) << "Expected prefill_attention calls";
        ASSERT_GE(first_ag, 0) << "Expected NCCL allgather calls";
        EXPECT_LT(first_ag, first_prefill)
            << "Q-head allgather must precede attention (INV-KVS-QAG)";
        // One allgather per rank for Q, one per rank for LSE.
        EXPECT_EQ(ag_before_prefill, 2)
            << "Q allgather = one NCCL allgather per rank before attention";
        EXPECT_EQ(ag_total, 4)
            << "Expected Q allgather (2) + LSE allgather (2)";
        // No head rearrangement at B == 1.
        const std::string qcopy =
            "memcpy2d(w=" +
            std::to_string((kHeads / 2) * (kKVLora + kQkRope) * 2) + ",h=1)";
        EXPECT_EQ(count_prefix(log, qcopy), 0)
            << "B==1 must consume the NCCL stage buffer directly";
    }

    // ── B == 2: rank-major → token-major rearrange (dcp copies per rank) ──
    {
        TestExecParamsBuilder builder(2, /*batch=*/2);
        auto params = builder.build(/*use_graph=*/false);
        log.clear();
        exec.execute_attention(params);

        // Each copy moves [B, HL, KV] rows: width = HL*KV*2 bytes, height = B.
        const std::string qcopy =
            "memcpy2d(w=" +
            std::to_string((kHeads / 2) * (kKVLora + kQkRope) * 2) + ",h=2)";
        int copies = 0, first_copy = -1, first_prefill = -1;
        for (size_t i = 0; i < log.size(); ++i) {
            if (log[i] == qcopy) {
                ++copies;
                if (first_copy < 0) first_copy = static_cast<int>(i);
            }
            if (log[i].rfind("prefill(", 0) == 0 && first_prefill < 0)
                first_prefill = static_cast<int>(i);
        }
        EXPECT_EQ(copies, 4)
            << "Expected dcp copies per rank (2*2) for the head rearrange";
        ASSERT_GE(first_prefill, 0);
        EXPECT_LT(first_copy, first_prefill)
            << "Rearrange must precede attention";
    }
}

// TD-KVS-QAG-GRAPH: sharded KV is nongraph-only — use_graph must be forced
// off so the QAG sequence runs (the graph path would replay per-rank-HL
// decode graphs + a combine graph that is no longer captured).
TEST(DcpExecutor, ShardedForcesNongraphPath) {
    std::vector<std::string> log;

    auto comm = lp::DcpCommunicator(null_comm_opts(2));
    auto opts = executor_opts(2);
    opts.dcp_kv_sharded = true;
    RecordingBackends rec(2, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads / 2, .head_dim = kKVLora,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .combine_num_heads = kHeads,
         .gpus = make_gpu_refs(2)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());
    log.clear();

    TestExecParamsBuilder builder(2, 1);
    auto params = builder.build(/*use_graph=*/true);  // graph requested
    exec.execute_attention(params);

    int prefill_count = 0;
    for (const auto& e : log)
        if (e.rfind("prefill(", 0) == 0) ++prefill_count;
    EXPECT_EQ(prefill_count, 2)
        << "Sharded KV must force the nongraph path (one prefill per rank)";
}

// Sharded KV at dcp>=2 without a communicator cannot run the Q-head
// allgather — construction must fail fast (INV-KVS-QAG).
TEST(DcpExecutor, ShardedWithoutCommunicatorThrows) {
    auto opts = executor_opts(2);
    opts.dcp_kv_sharded = true;
    opts.communicator = nullptr;
    EXPECT_THROW(lp::DcpExecutor{opts}, std::invalid_argument);
}

TEST(DcpExecutor, GraphModeSkipsPrefill) {
    // Graph mode (use_graph=true) should NOT call prefill_attention.
    std::vector<std::string> log;

    auto comm = lp::DcpCommunicator(null_comm_opts(1));
    auto opts = executor_opts(1);
    RecordingBackends rec(1, log);
    rec.apply(opts);
    opts.communicator = &comm;

    auto dcp_wrapper = lc::DcpAttentionWrapper(
        &comm,
        {.num_heads_local = kHeads, .head_dim = kVHeadDim,
         .hidden_size = kHidden, .max_batch_size = kMaxBatch,
         .gpus = make_gpu_refs(1)},
        opts.attention_devices, lc::null_launch_correction());
    opts.dcp_wrapper = &dcp_wrapper;

    auto exec = lp::DcpExecutor(opts);
    init_dequant_pool(exec, exec.dcp_size());
    log.clear();

    TestExecParamsBuilder builder(1, 2);
    auto params = builder.build(/*use_graph=*/true);
    exec.execute_attention(params);

    for (const auto& entry : log) {
        EXPECT_TRUE(entry.find("prefill(") == std::string::npos)
            << "Graph mode should not call prefill_attention: " << entry;
    }
}

TEST(DcpExecutor, CaptureGraphsNoop) {
    // dcp_size=1: capture_dcp_graphs is a no-op, doesn't crash
    auto opts = executor_opts(1);
    auto exec = lp::DcpExecutor(opts);
    exec.capture_dcp_graphs();  // No-op, should not crash
}

// ============================================================================
// Group 3: Multi-GPU smoke tests (REQUIRES 2 GPUs)
// ============================================================================
//
// Verifies the full DcpExecutor pipeline on 2 real GPUs with NCCL:
//   - DCP correction (allgather LSE + correction kernel + allreduce output)
//   - Comparison against CPU reference (non-DCP simulation)
//   - Dense and sparse attention modes
//
// Backend: SmokeCudaAttentionDevice for real device memory, simulated compute
// kernels (no-op GEMMs, norms, quantization), custom prefill callback
// with known partial outputs. DCP correction and NCCL communication are real.

// ── Smoke AttentionDevice: real CUDA memory, no-op compute ──────────────────
// Uses cudaSetDevice + cudaMalloc/cudaFree for real device memory,
// but GEMMs/norms/quant are no-ops (content irrelevant for DCP correction tests).
// Attention pipeline methods are no-ops except prefill_attention which is
// injected via a callable (set by tests to write known partial outputs).

class SmokeCudaAttentionDevice : public lc::AttentionDevice {
public:
    using PrefillFn = std::function<void(
        const void*, int, int, const int*, const int*,
        void*, int64_t, int, int, bool,
        const int*, const int*, int, void*, float*, int, void*)>;

    explicit SmokeCudaAttentionDevice(layerstorm::config::GpuRef gpu,
                                       PrefillFn fn = {})
        : gpu_(gpu), prefill_fn_(std::move(fn)) {}

    // ── Device selection + identity ─────────────────────────────────────────
    void set_device() override { cudaSetDevice(gpu_.id); }
    const layerstorm::config::GpuRef& gpu() const override { return gpu_; }

    // ── Compute kernels (no-op) ─────────────────────────────────────────────
    void gemm(const lc::Fp8GemmParams&, void*, void*) override {}
    void gguf_mmvq(const lc::GgufGemmParams&, void*, void*) override {}
    void gguf_mmq(const lc::GgufGemmParams&, void*, void*) override {}
    void gguf_dequant_gemm(const lc::GgufGemmParams&, void*) override {}
    void rmsnorm(void*, const void*, const void*, float,
                 int, int, int, void*) override {}
    void quantize_fp8(const lc::DynamicFp8QuantParams&, void*) override {}
    void weight_quantize_fp8(const lc::WeightFp8QuantParams&, void*) override {}
    void nvfp4_dequant_bf16(const lc::Nvfp4DequantBf16Params&, void*) override {}
    void nvfp4_grouped_gemm(const lc::Nvfp4GroupedGemmParams&, void*, size_t,
                            void*) override {}
    void bf16_to_nvfp4_grouped(const lc::Bf16ToNvfp4GroupedParams&,
                               void*) override {}
    void kv_bv_extract_dequant(const lc::KvBvExtractDequantParams&,
                                void*) override {}
    void batched_gemm_bf16(const lc::StridedBatchedGemmBf16Params&,
                            void*) override {}
    void absorb_q(const lc::QAbsorbParams&, void*) override {}
    void rope_rotate(const lc::RopeRotateParams&, void*) override {}

    // ── Device memory (real CUDA) ───────────────────────────────────────────
    void* device_alloc(size_t bytes) override {
        void* ptr = nullptr;
        cudaMalloc(&ptr, bytes);
        return ptr;
    }
    void device_free(void* ptr) override { if (ptr) cudaFree(ptr); }
    void device_sync() override { cudaDeviceSynchronize(); }
    void memcpy_h2d(void* dst, const void* src, size_t bytes) override {
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    }
    void memcpy_2d_d2d_async(void* dst, size_t dpitch,
                             const void* src, size_t spitch,
                             size_t width, size_t height,
                             void* stream) override {
        cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                          cudaMemcpyDeviceToDevice,
                          static_cast<cudaStream_t>(stream));
    }

    // ── KV cache append ─────────────────────────────────────────────────────
    void k_append(const void*, const void*, void*, int64_t, int,
                  const int*, int, int, int, int, int, int,
                  int, void*) override {}

    // ── Prefill attention ───────────────────────────────────────────────────
    void prefill_attention(
        const void* q, int B, int skv, const int* sl, const int* bt,
        int /*max_blocks_per_seq*/,
        void* kv, int64_t csb, int csr, int ps, bool sp,
        bool /*chunk_causal*/,
        const int* si, const int* tl, int tk,
        void* out, float* lse,
        int layer_idx, void* stream) override {
        if (prefill_fn_)
            prefill_fn_(q, B, skv, sl, bt, kv, csb, csr, ps, sp, si, tl, tk, out, lse, layer_idx, stream);
    }

    // ── Decode graph ops ────────────────────────────────────────────────────
    void decode_graph_update(lc::GraphEntry&, const void*,
                             const int*, const int*,
                             const int*,
                             int, void*) override {}
    void decode_graph_replay(lc::GraphEntry&, void*) override {}
    void* decode_graph_out_ptr(lc::GraphEntry&) override { return nullptr; }
    float* decode_graph_lse_ptr(lc::GraphEntry&) override { return nullptr; }

    // ── DCP allreduce graph ─────────────────────────────────────────────────
    void dcp_graph_replay(lc::GraphEntry&, void*) override {}

private:
    layerstorm::config::GpuRef gpu_;
    PrefillFn prefill_fn_;
};

#define SMOKE_CUDA_CHECK(expr)                                            \
    do {                                                                  \
        cudaError_t _err = (expr);                                        \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA: " << cudaGetErrorString(_err); \
    } while (0)

// ── BF16 host conversion ────────────────────────────────────────────────────

static uint16_t smoke_f2bf16(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    f += 0x7FFF + ((f >> 16) & 1);
    return static_cast<uint16_t>(f >> 16);
}

static float smoke_bf162f(uint16_t b) {
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
}

// ── CPU reference: DCP correction formula ───────────────────────────────────

static std::vector<float> smoke_dcp_ref(
    const std::vector<std::vector<float>>& rank_outputs,
    const std::vector<std::vector<float>>& rank_lses,
    int B, int H, int D, int N) {
    std::vector<float> result(B * H * D, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            float mx = -1e30f;
            for (int n = 0; n < N; ++n)
                mx = std::max(mx, rank_lses[n][b * H + h]);
            float se = 0.0f;
            for (int n = 0; n < N; ++n)
                se += std::exp(rank_lses[n][b * H + h] - mx);
            float gl = mx + std::log(se);
            for (int n = 0; n < N; ++n) {
                float s = std::exp(rank_lses[n][b * H + h] - gl);
                for (int d = 0; d < D; ++d)
                    result[b * H * D + h * D + d] +=
                        rank_outputs[n][b * H * D + h * D + d] * s;
            }
        }
    }
    return result;
}

// ── Fixture ─────────────────────────────────────────────────────────────────

class DcpExecutorMultiGpu : public ::testing::Test {
protected:
    static constexpr int kDcp = 2;

    void SetUp() override {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count < kDcp)
            GTEST_SKIP() << "Need " << kDcp << " GPUs, have " << count;
    }
};

// ── Helper: run DCP=2 smoke and verify ──────────────────────────────────────
//
// INV-KVS-QAG (sharded KV): the executor allgathers Q in the head dim, so
// EVERY rank's attention partial covers ALL kHeads heads over its LOCAL
// (disjoint) token shard; the combine then merges SAME-head partials across
// ranks. This smoke simulates the two disjoint-shard partials with per-head
// DISTINCT values so any residual head-mixing (the pre-QAG failure mode,
// TD-KVS-Q-ALLGATHER) breaks the per-head reference match.

static void run_smoke(
    float out_r0, float lse_r0,
    float out_r1, float lse_r1,
    bool is_sparse) {

    constexpr int N = 2;
    constexpr int B = 2;
    constexpr int HA = kHeads;      // all heads per rank (post-Q-allgather)
    constexpr int V = kVHeadDim;
    constexpr int elems = B * HA * V;

    // Per-(rank, head) values: bf16-exact steps, magnitude < 4 so the bf16
    // correct+allreduce rounding stays within tolerance.
    auto out_val = [](float base, int h) { return base + 0.125f * h; };
    auto lse_val = [](float base, int h) { return base + 0.05f * h; };

    // ── CPU reference: same-head merge over disjoint token shards ─────────
    std::vector<std::vector<float>> ref_outs(N, std::vector<float>(elems));
    std::vector<std::vector<float>> ref_lses(N, std::vector<float>(B * HA));
    const float bases[N] = {out_r0, out_r1};
    const float lbases[N] = {lse_r0, lse_r1};
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < HA; ++h) {
                ref_lses[n][b * HA + h] = lse_val(lbases[n], h);
                for (int d = 0; d < V; ++d)
                    ref_outs[n][b * HA * V + h * V + d] =
                        out_val(bases[n], h);
            }
        }
    }
    auto expected = smoke_dcp_ref(ref_outs, ref_lses, B, HA, V, N);

    // ── Phase 1: DCP=2 ──────────────────────────────────────────────────

    void* captured[N] = {};

    auto prefill2 = [&](const void*, int batch, int, const int*, const int*,
                        void*, int64_t, int, int, bool sp,
                        const int*, const int*, int,
                        void* out, float* lse, int, void*) {
        if (is_sparse)
            EXPECT_TRUE(sp) << "Expected is_sparse=true";
        else
            EXPECT_FALSE(sp) << "Expected is_sparse=false";

        int dev = -1;
        cudaGetDevice(&dev);
        captured[dev] = out;

        // All-head partial: [batch, HA, V] out + [batch, HA] LSE.
        std::vector<uint16_t> bf(static_cast<size_t>(batch) * HA * V);
        std::vector<float> lv(static_cast<size_t>(batch) * HA);
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < HA; ++h) {
                lv[b * HA + h] = lse_val(lbases[dev], h);
                for (int d = 0; d < V; ++d)
                    bf[b * HA * V + h * V + d] =
                        smoke_f2bf16(out_val(bases[dev], h));
            }
        }
        cudaMemcpy(out, bf.data(), bf.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(lse, lv.data(), lv.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
    };

    // Real backends: CudaSm120DeviceBackend for device alloc/free,
    // NcclCollectiveBackend for real NCCL communication.
    auto cuda_backend0 = lc::make_cuda_sm120_device_backend(
        layerstorm::config::GpuRef{.position = 0, .id = 0, .type = layerstorm::config::GpuType::rtx5090});
    auto cuda_backend1 = lc::make_cuda_sm120_device_backend(
        layerstorm::config::GpuRef{.position = 1, .id = 1, .type = layerstorm::config::GpuType::rtx5090});
    std::vector<lc::DeviceBackend*> cuda_backends2 = {cuda_backend0.get(), cuda_backend1.get()};

    auto nccl_collective = lp::make_nccl_collective_backend();

    auto comm2 = lp::DcpCommunicator({
        .dcp_size = N, .device_backends = cuda_backends2,
        .max_batch_size = B, .num_heads = kHeads,
        .attn_output_dim = V, .hidden_size = kHidden,
        .collective = nccl_collective.get(),
    });

    // SmokeCudaAttentionDevice: real CUDA memory, no-op compute, custom prefill
    auto cuda_dev0 = std::make_unique<SmokeCudaAttentionDevice>(
        layerstorm::config::GpuRef{.position = 0, .id = 0, .type = layerstorm::config::GpuType::rtx5090},
        prefill2);
    auto cuda_dev1 = std::make_unique<SmokeCudaAttentionDevice>(
        layerstorm::config::GpuRef{.position = 1, .id = 1, .type = layerstorm::config::GpuType::rtx5090},
        prefill2);
    std::vector<lc::AttentionDevice*> cuda_devs2 = {cuda_dev0.get(), cuda_dev1.get()};

    auto wrapper2 = lc::DcpAttentionWrapper(
        &comm2,
        {.num_heads_local = kHeads / N, .head_dim = V,
         .hidden_size = kHidden, .max_batch_size = B,
         // INV-KVS-QAG: the combine runs over ALL heads under sharded KV.
         .combine_num_heads = HA,
         .gpus = make_gpu_refs(2)},
        cuda_devs2, lc::cuda_launch_correction());

    auto exec2 = lp::DcpExecutor({
        .dcp_size = N, .gpus = make_gpu_refs(2),
        // These tests validate the KV-SHARDED DCP combine machinery itself.
        .dcp_kv_sharded = true,
        .max_batch_size = B, .hidden_size = kHidden,
        .num_attention_heads = kHeads, .q_lora_rank = kQLora,
        .kv_lora_rank = kKVLora, .qk_rope_head_dim = kQkRope,
        .qk_nope_head_dim = kQkNope, .v_head_dim = V,
        .communicator = &comm2,
        .dcp_wrapper = &wrapper2,
        .attention_devices = cuda_devs2,
    });
    init_dequant_pool(exec2, N);

    // Allocate dummy hidden_states (content irrelevant — GEMMs are no-op)
    void* h_states[N] = {};
    for (int r = 0; r < N; ++r) {
        SMOKE_CUDA_CHECK(cudaSetDevice(r));
        SMOKE_CUDA_CHECK(cudaMalloc(&h_states[r], B * kHidden * 2));
        SMOKE_CUDA_CHECK(cudaMemset(h_states[r], 0, B * kHidden * 2));
    }

    lp::AttentionLayerWeights w2[N] = {};
    const lp::AttentionLayerWeights* wp2[N] = {&w2[0], &w2[1]};
    int sl0[B] = {100, 100};
    int sl1[B] = {100, 100};
    const int* slp2[N] = {sl0, sl1};

    lp::AttentionExecParams params2 = {
        .layer_idx = 0, .batch_size = B,
        .hidden_states = h_states, .seqlens_k = slp2,
        .weights = wp2,
        .use_graph = false, .is_sparse = is_sparse,
    };

    exec2.execute_attention(params2);

    for (int r = 0; r < N; ++r) {
        SMOKE_CUDA_CHECK(cudaSetDevice(r));
        SMOKE_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ── Verify DCP=2 corrected output ────────────────────────────────────

    std::vector<uint16_t> dcp2_rank[N];
    for (int r = 0; r < N; ++r) {
        ASSERT_NE(captured[r], nullptr)
            << "Prefill callback not invoked for rank " << r;

        SMOKE_CUDA_CHECK(cudaSetDevice(r));
        dcp2_rank[r].resize(elems);
        SMOKE_CUDA_CHECK(cudaMemcpy(dcp2_rank[r].data(), captured[r],
                                     elems * sizeof(uint16_t),
                                     cudaMemcpyDeviceToHost));

        for (int i = 0; i < elems; ++i) {
            EXPECT_NEAR(smoke_bf162f(dcp2_rank[r][i]), expected[i], 0.03f)
                << "DCP=2 rank=" << r << " i=" << i;
        }
    }

    // Both ranks must agree (allreduce guarantee)
    for (int i = 0; i < elems; ++i) {
        EXPECT_EQ(dcp2_rank[0][i], dcp2_rank[1][i])
            << "Ranks diverge at i=" << i;
    }

    // ── Phase 2: DCP=1 comparison (non-DCP simulation) ──────────────────

    void* captured1 = nullptr;

    auto prefill1 = [&](const void*, int batch, int, const int*, const int*,
                        void*, int64_t, int, int, bool,
                        const int*, const int*, int,
                        void* out, float*, int, void*) {
        captured1 = out;
        // Write the expected DCP-corrected values directly (no correction
        // needed — the single rank IS the full-KV attention).
        std::vector<uint16_t> bf(static_cast<size_t>(batch) * kHeads * V);
        for (size_t i = 0; i < bf.size(); ++i)
            bf[i] = smoke_f2bf16(expected[i % expected.size()]);
        cudaMemcpy(out, bf.data(), bf.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice);
    };

    auto cuda_dev_solo = std::make_unique<SmokeCudaAttentionDevice>(
        layerstorm::config::GpuRef{.position = 0, .id = 0, .type = layerstorm::config::GpuType::rtx5090},
        prefill1);
    std::vector<lc::AttentionDevice*> cuda_devs1 = {cuda_dev_solo.get()};

    auto comm1 = lp::DcpCommunicator(null_comm_opts(1));
    auto wrap1 = lc::DcpAttentionWrapper(
        &comm1,
        {.num_heads_local = kHeads, .head_dim = V,
         .hidden_size = kHidden, .max_batch_size = B,
         .gpus = make_gpu_refs(1)},
        cuda_devs1, lc::null_launch_correction());

    auto exec1 = lp::DcpExecutor({
        .dcp_size = 1, .gpus = make_gpu_refs(1),
        .max_batch_size = B, .hidden_size = kHidden,
        .num_attention_heads = kHeads, .q_lora_rank = kQLora,
        .kv_lora_rank = kKVLora, .qk_rope_head_dim = kQkRope,
        .qk_nope_head_dim = kQkNope, .v_head_dim = V,
        .communicator = &comm1,
        .dcp_wrapper = &wrap1,
        .attention_devices = cuda_devs1,
    });
    init_dequant_pool(exec1, 1);

    void* h1[1] = {h_states[0]};
    lp::AttentionLayerWeights w1{};
    const lp::AttentionLayerWeights* wp1[1] = {&w1};
    const int* slp1[1] = {sl0};

    lp::AttentionExecParams params1 = {
        .layer_idx = 0, .batch_size = B,
        .hidden_states = h1, .seqlens_k = slp1,
        .weights = wp1,
        .use_graph = false, .is_sparse = is_sparse,
    };

    exec1.execute_attention(params1);

    SMOKE_CUDA_CHECK(cudaSetDevice(0));
    SMOKE_CUDA_CHECK(cudaDeviceSynchronize());

    ASSERT_NE(captured1, nullptr) << "DCP=1 prefill not invoked";
    std::vector<uint16_t> dcp1_data(B * kHeads * V);
    SMOKE_CUDA_CHECK(cudaMemcpy(dcp1_data.data(), captured1,
                                 dcp1_data.size() * sizeof(uint16_t),
                                 cudaMemcpyDeviceToHost));

    // DCP=1 output ≈ expected corrected values
    for (int i = 0; i < static_cast<int>(dcp1_data.size()); ++i) {
        EXPECT_NEAR(smoke_bf162f(dcp1_data[i]),
                    expected[i % expected.size()], 0.03f)
            << "DCP=1 diverges at i=" << i;
    }

    // DCP=2 corrected (any rank) ≈ DCP=1 uncorrected (same expected values)
    for (int i = 0; i < elems; ++i) {
        EXPECT_NEAR(smoke_bf162f(dcp2_rank[0][i]),
                    smoke_bf162f(dcp1_data[i]), 0.03f)
            << "DCP=2 vs DCP=1 diverge at i=" << i;
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    for (int r = 0; r < N; ++r) {
        cudaSetDevice(r);
        cudaFree(h_states[r]);
    }
}

// ── Dense smoke test ────────────────────────────────────────────────────────

TEST_F(DcpExecutorMultiGpu, TwoGpuDenseSmoke) {
    run_smoke(1.0f, 0.0f,
              3.0f, std::log(2.0f),
              /*is_sparse=*/false);
}

// ── Sparse smoke test ───────────────────────────────────────────────────────

TEST_F(DcpExecutorMultiGpu, TwoGpuSparseSmoke) {
    run_smoke(2.0f, 1.0f,
              4.0f, 2.0f,
              /*is_sparse=*/true);
}
