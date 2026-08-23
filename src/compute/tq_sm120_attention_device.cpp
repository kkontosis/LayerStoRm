// TurboQuant MLA SM120 concrete AttentionDevice.
//
// Composes CudaSm120DeviceBackend for hardware ops and implements TQ
// attention pipeline methods using TQ kernel launchers.
//
// Per-instance state (tq_resources_, model dims) set once at init via
// free-function accessors.  layer_idx passed as a parameter to virtual
// methods (added in #35h).
//
// MUST NOT include mla_attention.h (spec TURBOQUANT_DETAILS.md §17.4).

#include "compute/tq_sm120_attention_device.h"
#include "compute/cuda_sm120_device_backend.h"
#include "compute/kernels/attention/tq_mla_attention.h"
#include "compute/kernels/attention/linearize_block_tables.h"
#include "compute/prefill_params.h"
#include "compute/kernels/sm120/attention/tq_prep_params.h"
#include "compute/tq_init.h"
#include "compute/graphs/dcp_allreduce_graph.h"
#include "compute/graphs/graph_registry.h"
#include "config/config_parser.h"

#include <sm120/decode/tq_dense/params.h>
#include <sm120/decode/tq_sparse/params.h>
#include <sm120/graph/tq_decode_graph.h>

#include <any>
#include <cassert>
#include <cstdio>    // TD-PREFILL-NONDET seam_tq_dump
#include <cstdlib>   // TD-PREFILL-NONDET getenv gate
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>    // TD-PREFILL-NONDET D2H staging
#include <cuda_runtime.h>

namespace layerstorm::compute {

namespace {
// TD-PREFILL-NONDET diagnostic (LS_SEAM_DUMP_TQ=<path>, off by default):
// dump a TQ prefill-chain buffer (q input / linearized slots / dequant
// staging) for chunk-shaped calls at layers <= LS_SEAM_DUMP_MAXLAYER
// (default 4). Record (LE): int32 hdr[5]={tag4cc,layer,cuda_dev,1,bytes/2};
// raw payload. Diagnosis-only — syncs the stream when enabled.
void seam_tq_dump(uint32_t tag, int layer, const void* dev, size_t bytes,
                  cudaStream_t stream) {
    static const char* path = std::getenv("LS_SEAM_DUMP_TQ");
    if (!path || !*path || !dev || !bytes) return;
    static const int max_layer = [] {
        const char* e = std::getenv("LS_SEAM_DUMP_MAXLAYER");
        return (e && *e) ? std::atoi(e) : 4;
    }();
    if (layer > max_layer) return;
    static std::FILE* fp = std::fopen(path, "wb");
    if (!fp) return;
    std::vector<uint8_t> host(bytes);
    cudaMemcpyAsync(host.data(), dev, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int dev_id = -1;
    cudaGetDevice(&dev_id);
    int32_t hdr[5] = {static_cast<int32_t>(tag), layer, dev_id, 1,
                      static_cast<int32_t>(bytes / 2)};
    std::fwrite(hdr, sizeof(int32_t), 5, fp);
    std::fwrite(host.data(), 1, bytes, fp);
    std::fflush(fp);
}
}  // namespace

class TqSm120AttentionDevice final : public AttentionDevice {
public:
    explicit TqSm120AttentionDevice(config::GpuRef gpu)
        : device_(gpu)
    {
        // TD-71g: fail loud on an invalid device — an unchecked failure
        // leaves num_sm_ = 0, which the prefill kernel consumes as a grid
        // dimension (division by zero / degenerate tile scheduling).
        int sm = 0;
        const cudaError_t err = cudaDeviceGetAttribute(
            &sm, cudaDevAttrMultiProcessorCount, gpu.id);
        if (err != cudaSuccess || sm <= 0) {
            throw std::runtime_error(
                "TqSm120AttentionDevice: cudaDeviceGetAttribute("
                "MultiProcessorCount) failed for CUDA device "
                + std::to_string(gpu.id) + ": "
                + cudaGetErrorString(err));
        }
        num_sm_ = sm;
    }

    ~TqSm120AttentionDevice() override {
        if (tq_dec_qrot_) device_.device_free(tq_dec_qrot_);
        if (tq_dec_orot_) device_.device_free(tq_dec_orot_);
        if (tq_dec_indices_) device_.device_free(tq_dec_indices_);
        for (auto& v : tq_dec_graphs_)
            for (auto& g : v)
                if (g.exec) cudaGraphExecDestroy(g.exec);
        if (prefill_indices_scratch_) device_.device_free(prefill_indices_scratch_);
        if (prefill_kout_scratch_) device_.device_free(prefill_kout_scratch_);
        if (prefill_seq_offsets_scratch_) device_.device_free(prefill_seq_offsets_scratch_);
        if (prefill_num_fetch_scratch_) device_.device_free(prefill_num_fetch_scratch_);
    }

    // ── TQ-specific configuration (non-virtual) ─────────────────────────────

    /// Set TQ resources (codebook + per-layer Pi). Called by engine at init.
    void set_tq_resources(const TqResources* res) { tq_resources_ = res; }

    // DET-REDUCE (TD-TQ-PREFILL-DETREDUCE-WIRING): enable the gated
    // deterministic (bit-reproducible) softmax denominator reduction for this
    // device's prefill kernels (mirrors the SnapMLA twin).
    void set_deterministic_reduce(bool enable) { deterministic_reduce_ = enable; }

    /// Set model dimensions for graph update pointer arithmetic and prefill.
    void set_model_dims(int batch_size, int d_c, int d_rope, int h_q, int s_q = 1,
                        float sm_scale = 0.0f) {
        batch_size_ = batch_size;
        d_c_ = d_c;
        d_rope_ = d_rope;
        h_q_ = h_q;
        s_q_ = s_q;
        sm_scale_ = sm_scale;  // 0 → legacy 1/√(d_c+d_rope)
    }

    /// Allocate prefill scratch buffers for dequant index linearization + BF16 output.
    void set_prefill_scratch(int max_kv_tokens) {
        if (max_kv_tokens <= prefill_scratch_capacity_) return;
        // Free old buffers
        if (prefill_indices_scratch_) device_.device_free(prefill_indices_scratch_);
        if (prefill_kout_scratch_) device_.device_free(prefill_kout_scratch_);
        if (prefill_seq_offsets_scratch_) device_.device_free(prefill_seq_offsets_scratch_);
        if (prefill_num_fetch_scratch_) device_.device_free(prefill_num_fetch_scratch_);

        // indices: int[max_kv_tokens]
        prefill_indices_scratch_ = static_cast<int*>(
            device_.device_alloc(static_cast<size_t>(max_kv_tokens) * sizeof(int)));
        // k_out: bf16[max_kv_tokens * (d_c + d_rope)]
        const size_t kout_bytes = static_cast<size_t>(max_kv_tokens) *
            static_cast<size_t>(d_c_ + d_rope_) * 2;
        prefill_kout_scratch_ = device_.device_alloc(kout_bytes);
        // seq_offsets: int[batch_size] — batch_size << max_kv_tokens, reuse max_batch_size_
        prefill_seq_offsets_scratch_ = static_cast<int*>(
            device_.device_alloc(static_cast<size_t>(batch_size_) * sizeof(int)));
        // num_fetch: int[1]
        prefill_num_fetch_scratch_ = static_cast<int*>(
            device_.device_alloc(sizeof(int)));

        // FAIL-LOUD (1M-cap lesson, mirrors SnapMLA): a silently-null
        // scratch is consumed as "skip the dequant" downstream — stale-read
        // corruption, not an error. Abort init instead.
        if (!prefill_indices_scratch_ || !prefill_kout_scratch_
            || !prefill_seq_offsets_scratch_ || !prefill_num_fetch_scratch_) {
            throw std::runtime_error(
                "TqSm120AttentionDevice: prefill scratch allocation failed ("
                + std::to_string(kout_bytes)
                + " B k_out — scales with serving.max_sequence_length)");
        }

        prefill_scratch_capacity_ = max_kv_tokens;
    }

    // ── Non-graph decode (full pipeline with v_rotate_back epilogue) ────────

    void decode_ungraphed(
        const __nv_bfloat16* q_nope,
        const __nv_bfloat16* q_rope,
        const uint8_t* kv_cache,
        int64_t cache_stride_block, int cache_stride_row,
        const int* block_table, int block_table_batch_stride,
        int page_block_size,
        const int* seqlens_k,
        __nv_bfloat16* out_bf16, float* lse,
        float* scratch_fp32, float* q_rot_fp32,
        int layer_idx,
        cudaStream_t stream) const
    {
        assert(tq_resources_ && "TqResources not set — call set_tq_resources()");
        assert(layer_idx >= 0 && "layer_idx must be non-negative");

        const int batch_heads = batch_size_ * s_q_ * h_q_;
        const float* Pi = tq_resources_->device_Pi(layer_idx);

        // Step 1: Pre-rotate Q nope by Pi^T  →  FP32 rotated query
        sm120::prep::TqQRotateParams qr{};
        qr.q_nope = q_nope;
        qr.Pi = Pi;
        qr.q_rot = q_rot_fp32;
        qr.batch_heads = batch_heads;
        qr.d_c = d_c_;
        launch_tq_q_rotate(qr, stream);

        // Step 2: TQ dense decode  →  FP32 output in rotated space
        sm120::decode::tq_dense::TqDenseDecodeParams dp{};
        std::memset(&dp, 0, sizeof(dp));
        dp.b = batch_size_;
        dp.s_q = s_q_;
        dp.h_q = h_q_;
        dp.h_kv = 1;
        dp.d_c = d_c_;
        dp.d_rope = d_rope_;
        dp.sm_scale = sm_scale_ > 0.0f
            ? sm_scale_ : 1.0f / std::sqrt(static_cast<float>(d_c_ + d_rope_));
        dp.q_rot = q_rot_fp32;
        dp.q_rope = q_rope;
        dp.kv_cache = kv_cache;
        dp.cache_stride_block = cache_stride_block;
        dp.cache_stride_row = cache_stride_row;
        dp.block_table = block_table;
        dp.block_table_batch_stride = block_table_batch_stride;
        dp.page_block_size = page_block_size;
        dp.seqlens_k = seqlens_k;
        dp.centroids = tq_resources_->device_centroids();
        dp.out = scratch_fp32;
        dp.lse = lse;
        dp.stride_o_b = s_q_ * h_q_ * d_c_;
        dp.stride_o_s_q = h_q_ * d_c_;
        dp.stride_o_h_q = d_c_;
        dp.stride_lse_b = s_q_ * h_q_;
        dp.stride_lse_s_q = h_q_;
        dp.num_sm_parts = 1;
        dp.stream = stream;
        launch_decode_dense_tq(dp);

        // Step 3: Inverse rotate  →  BF16 output in original space
        sm120::prep::TqVRotateBackParams vr{};
        vr.out_rotated = scratch_fp32;
        vr.Pi = Pi;
        vr.Pi_t = tq_resources_->device_Pi_t(layer_idx);
        vr.out_final = out_bf16;
        vr.batch_heads = batch_heads;
        vr.d_c = d_c_;
        launch_tq_v_rotate_back(vr, stream);
    }

    // ── Device selection + identity (delegated to CudaSm120DeviceBackend) ──

    void set_device() override { device_.set_device(); }
    const config::GpuRef& gpu() const override { return device_.gpu(); }

    // ── Compute kernels (delegated) ─────────────────────────────────────────

    void gemm(const Fp8GemmParams& params,
              void* workspace, void* stream) override {
        device_.gemm(params, workspace, stream);
    }

    void gguf_mmvq(const GgufGemmParams& params,
                   void* q8_1_workspace, void* stream) override {
        device_.gguf_mmvq(params, q8_1_workspace, stream);
    }
    void gguf_mmq(const GgufGemmParams& params,
                  void* q8_1_workspace, void* stream) override {
        device_.gguf_mmq(params, q8_1_workspace, stream);
    }
    void gguf_dequant_gemm(const GgufGemmParams& params,
                           void* stream) override {
        device_.gguf_dequant_gemm(params, stream);
    }

    void rmsnorm(void* out, const void* input,
                 const void* weight, float eps,
                 int num_tokens, int hidden_size, int row_stride,
                 void* stream) override {
        device_.rmsnorm(out, input, weight, eps, num_tokens, hidden_size,
                        row_stride, stream);
    }

    void quantize_fp8(const DynamicFp8QuantParams& params,
                      void* stream) override {
        device_.quantize_fp8(params, stream);
    }
    void weight_quantize_fp8(const WeightFp8QuantParams& params,
                              void* stream) override {
        device_.weight_quantize_fp8(params, stream);
    }
    void nvfp4_dequant_bf16(const Nvfp4DequantBf16Params& params,
                             void* stream) override {
        device_.nvfp4_dequant_bf16(params, stream);
    }
    void nvfp4_grouped_gemm(
        const Nvfp4GroupedGemmParams& params,
        void* workspace, size_t workspace_bytes,
        void* stream) override {
        device_.nvfp4_grouped_gemm(params, workspace, workspace_bytes, stream);
    }
    void bf16_to_nvfp4_grouped(const Bf16ToNvfp4GroupedParams& params,
                                void* stream) override {
        device_.bf16_to_nvfp4_grouped(params, stream);
    }
    void kv_bv_extract_dequant(const KvBvExtractDequantParams& params,
                                void* stream) override {
        device_.kv_bv_extract_dequant(params, stream);
    }
    void batched_gemm_bf16(const StridedBatchedGemmBf16Params& params,
                            void* stream) override {
        device_.batched_gemm_bf16(params, stream);
    }
    void absorb_q(const QAbsorbParams& params, void* stream) override {
        device_.absorb_q(params, stream);
    }
    void rope_rotate(const RopeRotateParams& params, void* stream) override {
        device_.rope_rotate(params, stream);
    }
    void indexer_layernorm_bias(void* x, const void* w, const void* b,
                                int rows, int dim, float eps, void* s) override {
        device_.indexer_layernorm_bias(x, w, b, rows, dim, eps, s);
    }
    void indexer_hadamard(void* x, int rows, int dim, void* s) override {
        device_.indexer_hadamard(x, rows, dim, s);
    }
    void indexer_k_quant_append(const void* k, const void* slots, void* cache,
                                void* scales, int tokens, int dim, int slot_bias,
                                void* s) override {
        device_.indexer_k_quant_append(k, slots, cache, scales, tokens, dim,
                                       slot_bias, s);
    }
    void indexer_scale_weights(const void* in, void* out, int rows, int n,
                               float scale, void* s) override {
        device_.indexer_scale_weights(in, out, rows, n, scale, s);
    }
    void indexer_score_topk(const IndexerScoreTopkArgs& a, void* s) override {
        device_.indexer_score_topk(a, s);
    }
    void indexer_score_topk_batched(const IndexerScoreTopkBatchedArgs& a,
                                    void* s) override {
        device_.indexer_score_topk_batched(a, s);
    }
    void indexer_shard_translate(const void* gi, const void* gl,
                                 void* li, void* ll, int nt, int topk,
                                 int chunk, int dcp, int rank,
                                 void* s) override {
        device_.indexer_shard_translate(gi, gl, li, ll, nt, topk, chunk, dcp,
                                        rank, s);
    }
    void indexer_topk_merge(const IndexerTopkMergeArgs& a, void* s) override {
        device_.indexer_topk_merge(a, s);
    }

    // ── Device memory (delegated) ───────────────────────────────────────────

    void* device_alloc(size_t bytes) override { return device_.device_alloc(bytes); }
    void  device_free(void* ptr) override { device_.device_free(ptr); }
    void  device_sync() override { device_.device_sync(); }
    void  memcpy_h2d(void* dst, const void* src, size_t bytes) override {
        device_.memcpy_h2d(dst, src, bytes);
    }
    void  memcpy_2d_d2d_async(void* dst, size_t dpitch,
                              const void* src, size_t spitch,
                              size_t width, size_t height,
                              void* stream) override {
        device_.memcpy_2d_async(dst, dpitch, src, spitch, width, height, stream);
    }
    void  memcpy_h2d_async(void* dst, const void* src, size_t bytes,
                           void* stream) override {
        device_.memcpy_h2d_async(dst, src, bytes, stream);
    }

    // ── KV cache append ─────────────────────────────────────────────────────

    void k_append(
        const void* c_kv, const void* k_rope, void* kv_cache,
        int64_t cache_stride_block, int cache_stride_row,
        const int* slot_mapping, int num_tokens,
        int d_c, int d_rope,
        int c_kv_row_stride, int k_rope_row_stride,
        int page_size,
        int layer_idx, void* stream) override
    {
        assert(tq_resources_ && "TqResources not set — call set_tq_resources()");
        assert(layer_idx >= 0 && "layer_idx must be non-negative");

        sm120::prep::TqFusedKAppendParams p{};
        p.c_kv = static_cast<const __nv_bfloat16*>(c_kv);
        p.k_rope = static_cast<const __nv_bfloat16*>(k_rope);
        p.src_stride_ckv = c_kv_row_stride;
        p.src_stride_rope = k_rope_row_stride;
        p.kv_cache = static_cast<uint8_t*>(kv_cache);
        p.cache_stride_block = cache_stride_block;
        p.cache_stride_row = cache_stride_row;
        p.slot_mapping = slot_mapping;
        p.Pi = tq_resources_->device_Pi(layer_idx);
        p.centroids = tq_resources_->device_centroids();
        p.decision_boundaries = tq_resources_->device_boundaries();
        p.num_tokens = num_tokens;
        p.d_c = d_c;
        p.d_rope = d_rope;
        p.page_size = page_size;
        p.num_centroids = tq_resources_->codebook.n_clusters;
        launch_tq_k_append(p, static_cast<cudaStream_t>(stream));
    }

    // ── Prefill attention ───────────────────────────────────────────────────

    void prefill_attention(
        const void* q_compressed, int batch_size, int seq_len_kv,
        const int* seqlens_k, const int* block_tables,
        int max_blocks_per_seq,
        void* kv_cache, int64_t cache_stride_block, int cache_stride_row,
        int page_size, bool is_sparse, bool chunk_causal,
        const int* sparse_indices, const int* topk_lengths, int topk,
        void* out, float* lse,
        int layer_idx, void* stream) override
    {
        assert(tq_resources_ && "TqResources not set");
        assert(layer_idx >= 0 && "layer_idx must be non-negative");
        auto s = static_cast<cudaStream_t>(stream);

        // Linearize block_tables → flat slot indices for indexed dequant.
        // chunk_causal: stage the UNION prefix once via the LAST row only
        // (see the chunk_causal contract in attention_device.h and the
        // SnapMLA twin — kept in lockstep).
        const int lin_row = chunk_causal ? batch_size - 1 : 0;
        const int lin_batch = chunk_causal ? 1 : batch_size;
        const int max_blocks = prefill_max_blocks(
            max_blocks_per_seq, seq_len_kv, page_size);
        if (block_tables && seqlens_k && prefill_indices_scratch_) {
            launch_linearize_block_tables(
                block_tables + static_cast<size_t>(lin_row) * max_blocks,
                seqlens_k + lin_row, lin_batch, max_blocks, page_size,
                seq_len_kv, prefill_indices_scratch_,
                prefill_seq_offsets_scratch_, prefill_num_fetch_scratch_,
                stream);
        }

        if (!kv_cache || !prefill_kout_scratch_ || !prefill_indices_scratch_)
            return;

        // ── Direct-TQ sparse decode — the ONLY TQ decode path for this
        // shape (user decision 2026-07-14, §12m-r5): codebook-space attention
        // on the quantized cache is the TQ paper's math; the dequant+phase1
        // chain below serves PREFILL, chunk-causal, and B>1 shapes only (its
        // staged-bf16 error profile is NOT decode-validated). Chain: translate
        // DSA positions → pool slots (composes the linearization above),
        // pre-rotate Q (strided over the interleaved [nope|rope] rows), attend
        // in codebook space, inverse-rotate into `out`. LSE is natural-log,
        // layout [s_q, h_q] — identical contract to the phase1 kernel
        // (INV-LSE-NAT; max_logits is consumer-free per KVS-1).
        if (is_sparse && batch_size == 1 && !chunk_causal && sparse_indices) {
            const int d_qk = d_c_ + d_rope_;
            if (!tq_dec_qrot_) {
                tq_dec_qrot_ = static_cast<float*>(device_.device_alloc(
                    static_cast<size_t>(h_q_) * d_c_ * sizeof(float)));
                tq_dec_orot_ = static_cast<float*>(device_.device_alloc(
                    static_cast<size_t>(h_q_) * d_c_ * sizeof(float)));
            }
            if (topk > tq_dec_topk_cap_) {
                if (tq_dec_indices_) device_.device_free(tq_dec_indices_);
                tq_dec_indices_ = static_cast<int*>(device_.device_alloc(
                    static_cast<size_t>(topk) * sizeof(int)));
                tq_dec_topk_cap_ = topk;
            }
            if (!tq_dec_qrot_ || !tq_dec_orot_ || !tq_dec_indices_)
                throw std::runtime_error(
                    "TqSm120AttentionDevice: sparse-decode scratch "
                    "allocation failed (INV-SCRATCH-LOUD — a silent fallback "
                    "to the dequant chain would switch decode numerics "
                    "mid-run)");
            // Device-read causal bound ONLY under graph mode: the toggle-OFF
            // path keeps the host-scalar bound bit-identical to the
            // established route.
            const int* seq_len_dev = tq_dec_graph_on_ ? seqlens_k : nullptr;
            auto launch_chain = [&](cudaStream_t cs) {
                launch_tq_sparse_translate_indices(
                    sparse_indices, prefill_indices_scratch_, topk_lengths,
                    topk, seq_len_kv, seq_len_dev, tq_dec_indices_, cs);

                sm120::prep::TqQRotateParams qr{};
                qr.q_nope = static_cast<const __nv_bfloat16*>(q_compressed);
                qr.Pi = tq_resources_->device_Pi(layer_idx);
                qr.q_rot = tq_dec_qrot_;
                qr.batch_heads = h_q_;
                qr.d_c = d_c_;
                qr.q_row_stride = d_qk;   // interleaved [nope|rope] head rows
                launch_tq_q_rotate(qr, cs);

                sm120::decode::tq_sparse::TqSparseDecodeParams sp{};
                std::memset(&sp, 0, sizeof(sp));
                sp.b = 1; sp.s_q = 1;
                sp.h_q = h_q_; sp.h_kv = 1;
                sp.d_c = d_c_; sp.d_rope = d_rope_;
                sp.sm_scale = sm_scale_ > 0.0f
                    ? sm_scale_
                    : 1.0f / std::sqrt(static_cast<float>(d_qk));
                sp.q_rot = tq_dec_qrot_;
                sp.q_rope = static_cast<const __nv_bfloat16*>(q_compressed)
                            + d_c_;
                sp.q_rope_row_stride = d_qk;
                sp.kv_cache = static_cast<const uint8_t*>(kv_cache);
                sp.cache_stride_block = cache_stride_block;
                sp.cache_stride_row = cache_stride_row;
                sp.page_block_size = page_size;
                sp.indices = tq_dec_indices_;
                sp.topk = topk;
                sp.stride_indices_b = 0;
                sp.stride_indices_s_q = 0;
                sp.centroids = tq_resources_->device_centroids();
                sp.out = tq_dec_orot_;
                sp.lse = lse;
                sp.stride_o_b = h_q_ * d_c_;
                sp.stride_o_s_q = h_q_ * d_c_;
                sp.stride_o_h_q = d_c_;
                sp.stride_lse_b = h_q_;
                sp.stride_lse_s_q = h_q_;
                sp.stream = cs;
                launch_decode_sparse_tq(sp);

                sm120::prep::TqVRotateBackParams vr{};
                vr.out_rotated = tq_dec_orot_;
                vr.Pi = tq_resources_->device_Pi(layer_idx);
                vr.Pi_t = tq_resources_->device_Pi_t(layer_idx);
                vr.out_final = static_cast<__nv_bfloat16*>(out);
                vr.batch_heads = h_q_;
                vr.d_c = d_c_;
                launch_tq_v_rotate_back(vr, cs);
            };

            // Graph replay needs the causal bound read from the device
            // (seqlens_k restages every token); without it the bound would be
            // baked stale into the capture — fall back to direct launches.
            if (!tq_dec_graph_on_ || !seq_len_dev) {
                launch_chain(s);
                return;
            }
            if (layer_idx >= static_cast<int>(tq_dec_graphs_.size()))
                tq_dec_graphs_.resize(layer_idx + 1);
            auto& cache = tq_dec_graphs_[layer_idx];
            auto fits = [&](const TqDecGraph& g) {
                return g.exec
                    && g.q == q_compressed && g.out == out && g.lse == lse
                    && g.sidx == sparse_indices && g.tkl == topk_lengths
                    && g.kv == kv_cache && g.seqlens == seq_len_dev
                    && g.lin == prefill_indices_scratch_
                    && g.dec_idx == tq_dec_indices_ && g.topk == topk
                    && g.csb == cache_stride_block
                    && g.csr == cache_stride_row && g.ps == page_size;
            };
            TqDecGraph* hit = nullptr;
            for (auto& g : cache)
                if (fits(g)) { hit = &g; break; }
            if (!hit && static_cast<int>(cache.size()) >= kTqDecGraphCap) {
                // Cache full with foreign fingerprints — eager, no churn.
                launch_chain(s);
                return;
            }
            if (!hit) {
                TqDecGraph g{};
                cudaError_t err = cudaStreamBeginCapture(
                    s, cudaStreamCaptureModeThreadLocal);
                if (err != cudaSuccess)
                    throw std::runtime_error(
                        "LS_TQ_DECODE_GRAPH: capture begin failed: "
                        + std::string(cudaGetErrorString(err)));
                launch_chain(s);
                cudaGraph_t graph = nullptr;
                err = cudaStreamEndCapture(s, &graph);
                if (err != cudaSuccess || !graph)
                    throw std::runtime_error(
                        "LS_TQ_DECODE_GRAPH: capture end failed: "
                        + std::string(cudaGetErrorString(err)));
                err = cudaGraphInstantiateWithFlags(&g.exec, graph, 0);
                cudaGraphDestroy(graph);
                if (err != cudaSuccess || !g.exec)
                    throw std::runtime_error(
                        "LS_TQ_DECODE_GRAPH: instantiate failed: "
                        + std::string(cudaGetErrorString(err)));
                g.q = q_compressed; g.out = out; g.lse = lse;
                g.sidx = sparse_indices; g.tkl = topk_lengths;
                g.kv = kv_cache; g.seqlens = seq_len_dev;
                g.lin = prefill_indices_scratch_; g.dec_idx = tq_dec_indices_;
                g.topk = topk; g.csb = cache_stride_block;
                g.csr = cache_stride_row; g.ps = page_size;
                cache.push_back(g);
                hit = &cache.back();
            }
            if (cudaError_t err = cudaGraphLaunch(hit->exec, s);
                err != cudaSuccess)
                throw std::runtime_error(
                    "LS_TQ_DECODE_GRAPH: replay failed: "
                    + std::string(cudaGetErrorString(err)));
            return;
        }

        // TD-PREFILL-NONDET diagnostic (chunk-shaped calls only): per-rank q
        // input ('Qin '), linearized slot indices ('Lin ') — splits q-side
        // vs KV-side divergence, and catches run-varying physical page
        // assignment. Env-gated, zero work off.
        if (batch_size > 1) {
            const int d_qk_dump = d_c_ + d_rope_;
            seam_tq_dump(0x206e6951u /*'Qin '*/, layer_idx, q_compressed,
                         static_cast<size_t>(batch_size) * h_q_ * d_qk_dump * 2,
                         s);
            seam_tq_dump(0x206e694cu /*'Lin '*/, layer_idx,
                         prefill_indices_scratch_,
                         static_cast<size_t>(seq_len_kv) * sizeof(int), s);
        }

        sm120::prep::TqDequantCKVIndexedParams dq{};
        dq.kv_cache = static_cast<const uint8_t*>(kv_cache);
        dq.cache_stride_block = cache_stride_block;
        dq.cache_stride_row = cache_stride_row;
        dq.page_size = page_size;
        dq.Pi = tq_resources_->device_Pi(layer_idx);
        dq.centroids = tq_resources_->device_centroids();
        dq.d_c = tq_resources_->d_c();
        dq.d_rope = d_rope_;
        dq.indices = prefill_indices_scratch_;
        dq.num_fetch = seq_len_kv;
        dq.k_out = static_cast<__nv_bfloat16*>(prefill_kout_scratch_);

        const PrefillDims dims{d_c_, d_rope_, h_q_, num_sm_, sm_scale_};

        if (!is_sparse) {
            sm120::prefill::dense::head64::DenseAttnFwdParams p{};
            populate_dense_prefill_params(p, dims, q_compressed,
                prefill_kout_scratch_, batch_size, seq_len_kv, out, lse, s);
            p.deterministic_reduce = deterministic_reduce_;  // DET-REDUCE gate
            // chunk_causal: per-query-row causal bound — row b attends the
            // staged union prefix [0, seqlens_k[b]).
            if (chunk_causal) p.s_kv_per_row = seqlens_k;
            launch_prefill_dense_tq(dq, p, s);
            // TD-PREFILL-NONDET diagnostic: dequant staging ('Kst ') — the
            // BF16 K/V rows the phase1 kernel consumed (persists post-run).
            if (batch_size > 1)
                seam_tq_dump(0x2074734bu /*'Kst '*/, layer_idx,
                             prefill_kout_scratch_,
                             static_cast<size_t>(seq_len_kv)
                                 * (d_c_ + d_rope_) * 2, s);
        } else {
            SparseAttnFwdParams p{};
            populate_sparse_prefill_params(p, dims, q_compressed,
                prefill_kout_scratch_, sparse_indices, topk_lengths, topk,
                batch_size, seq_len_kv, out, lse, s);
            p.deterministic_reduce = deterministic_reduce_;  // DET-REDUCE gate
            // chunk_causal × sparse (TD-SPARSE-CHUNK-PREFILL): per-query-row
            // causal bound — same shared kernel contract as the SnapMLA twin
            // (kept in lockstep, INV-SPARSE-CHUNK-CAUSAL).
            if (chunk_causal) p.s_kv_per_row = seqlens_k;
            launch_prefill_sparse_tq(dq, p, s);
        }
    }

    // ── Decode graph ops ────────────────────────────────────────────────────

    void decode_graph_update(
        GraphEntry& entry, const void* q_bf16,
        const int* seqlens_k, const int* block_table,
        const int* /*indices*/,
        int /*layer_idx*/, void* stream) override
    {
        assert(d_c_ > 0 && "set_model_dims not called");
        auto** p = std::any_cast<sm120::graph::TqDecodeGraphRunner*>(&entry.runner);
        if (p && *p) {
            // NOTE (q-absorption, 481): q_bf16 is now the W_UK-absorbed query in the
            // PER-HEAD INTERLEAVED layout [s_q, h_q, d_c + d_rope] (produced by
            // DcpExecutor::execute_common_prefix → absorb_q), matching the prefill and
            // SnapMLA-decode contract. The split [all-nope | all-rope] slicing below is
            // STALE for that layout, but this graph path is currently unreachable —
            // engine never captures decode graphs (see TD-DECODE-GRAPH). When decode-graph
            // capture is wired, the TQ runner must consume the interleaved query (strided
            // q_nope/q_rope with row stride d_c+d_rope), not these contiguous blocks.
            auto* q_nope = q_bf16;
            const size_t rope_offset =
                static_cast<size_t>(batch_size_) * s_q_ * h_q_ * d_c_
                * sizeof(__nv_bfloat16);
            auto* q_rope = static_cast<const void*>(
                static_cast<const char*>(q_bf16) + rope_offset);
            (*p)->update(q_nope, q_rope, seqlens_k, block_table,
                         static_cast<cudaStream_t>(stream));
        }
    }

    void decode_graph_replay(GraphEntry& entry, void* stream) override {
        auto** p = std::any_cast<sm120::graph::TqDecodeGraphRunner*>(&entry.runner);
        if (p && *p) {
            (*p)->replay(static_cast<cudaStream_t>(stream));
        }
    }

    void* decode_graph_out_ptr(GraphEntry& entry) override {
        auto** p = std::any_cast<sm120::graph::TqDecodeGraphRunner*>(&entry.runner);
        return (p && *p) ? static_cast<void*>((*p)->out_ptr()) : nullptr;
    }

    float* decode_graph_lse_ptr(GraphEntry& entry) override {
        auto** p = std::any_cast<sm120::graph::TqDecodeGraphRunner*>(&entry.runner);
        return (p && *p) ? (*p)->lse_ptr() : nullptr;
    }

    // ── DCP allreduce graph ─────────────────────────────────────────────────

    void dcp_graph_replay(GraphEntry& entry, void* stream) override {
        // DCP correction is backend-agnostic (INV-DCP-1)
        auto** p = std::any_cast<DcpAllreduceGraphRunner*>(&entry.runner);
        if (p && *p) {
            (*p)->replay(stream);
        }
    }

private:
    CudaSm120DeviceBackend device_;

    // TQ per-instance state (set once at init via free-function accessors)
    const TqResources* tq_resources_ = nullptr;
    int batch_size_ = 1;
    int d_c_    = 512;
    int d_rope_ = 64;
    int h_q_    = 128;
    int s_q_    = 1;
    float sm_scale_ = 0.0f;  // model softmax scale ((qk_nope+qk_rope)^-0.5 · mscale²)
    int num_sm_ = 0;       // GPU SM count (queried at construction)
    bool deterministic_reduce_ = false;  // DET-REDUCE gate (config/env, set at init)

    // TQ sparse-decode fast path scratch (TD-TQ-SPARSE-DECODE-UNWIRED;
    // lazy-allocated on first use, sized h_q*d_c fp32 ×2 + topk ints).
    float* tq_dec_qrot_   = nullptr;
    float* tq_dec_orot_   = nullptr;
    int*   tq_dec_indices_ = nullptr;
    int    tq_dec_topk_cap_ = 0;

    // ── LS_TQ_DECODE_GRAPH (§12n): per-layer CUDA graph of the 4-kernel
    // sparse-decode chain (translate → q_rotate → decode → v_rotate_back).
    // All shapes are fixed at B=1 and every per-token variation is device
    // buffer CONTENT (indices, q, cache, seqlens_k[0] causal bound), so one
    // replay replaces 4 host launches. The fingerprint guards every pointer
    // and host scalar baked into the capture — any drift re-captures.
    struct TqDecGraph {
        cudaGraphExec_t exec = nullptr;
        const void* q = nullptr;   const void* out = nullptr;
        const float* lse = nullptr;
        const int* sidx = nullptr; const int* tkl = nullptr;
        const void* kv = nullptr;  const int* seqlens = nullptr;
        const int* lin = nullptr;  const int* dec_idx = nullptr;
        int topk = 0; int64_t csb = 0; int csr = 0; int ps = 0;
    };
    // Per-layer graph CACHE (not a single slot): B>1 sparse decode is
    // row-sliced upstream into B batch-of-1 calls with per-row pointer sets
    // (dcp_executor mixed_rows, TD-DECODE-NONGRAPH-BATCH) — a single slot
    // would re-capture every row every layer. Small linear-scan cache keyed
    // by the full fingerprint; beyond the cap, eager launches (no capture
    // churn). Cap 64 covers max decode batch rows.
    static constexpr int kTqDecGraphCap = 64;
    std::vector<std::vector<TqDecGraph>> tq_dec_graphs_;
    // DEFAULT ON (user decision 2026-07-20; INV-0.6(b) — trajectory
    // bit-identical, engagement nsys-proven §12n). LS_TQ_DECODE_GRAPH=0
    // opts out.
    bool tq_dec_graph_on_ = [] {
        const char* v = std::getenv("LS_TQ_DECODE_GRAPH");
        return !(v && *v && v[0] == '0');
    }();

    // Prefill dequant scratch buffers (allocated by set_prefill_scratch)
    int* prefill_indices_scratch_       = nullptr;
    void* prefill_kout_scratch_         = nullptr;
    int* prefill_seq_offsets_scratch_   = nullptr;
    int* prefill_num_fetch_scratch_     = nullptr;
    int prefill_scratch_capacity_       = 0;
};

// ── Factory ─────────────────────────────────────────────────────────────────

std::unique_ptr<AttentionDevice> make_tq_sm120_attention_device(
        config::GpuRef gpu) {
    return std::make_unique<TqSm120AttentionDevice>(std::move(gpu));
}

// ── R0H-1d: free-function accessors for TQ-specific state ──────────────────

void tq_device_set_resources(AttentionDevice* dev, const TqResources* res) {
    static_cast<TqSm120AttentionDevice*>(dev)->set_tq_resources(res);
}

void tq_device_set_model_dims(AttentionDevice* dev,
        int batch_size, int d_c, int d_rope, int h_q, int s_q, float sm_scale) {
    static_cast<TqSm120AttentionDevice*>(dev)->set_model_dims(
        batch_size, d_c, d_rope, h_q, s_q, sm_scale);
}

void tq_device_set_prefill_scratch(AttentionDevice* dev, int max_kv_tokens) {
    static_cast<TqSm120AttentionDevice*>(dev)->set_prefill_scratch(max_kv_tokens);
}

void tq_device_set_deterministic_reduce(AttentionDevice* dev, bool enable) {
    static_cast<TqSm120AttentionDevice*>(dev)->set_deterministic_reduce(enable);
}

}  // namespace layerstorm::compute
