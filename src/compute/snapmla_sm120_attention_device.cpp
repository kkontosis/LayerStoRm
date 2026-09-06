// SnapMLA SM120 concrete AttentionDevice.
//
// Composes CudaSm120DeviceBackend for hardware ops (gemm, rmsnorm,
// quantize_fp8, alloc/free) and implements attention pipeline methods
// using SnapMLA kernel launchers.
//
// Body of attention methods taken from SnapMlaCudaSm120Backend.

#include "compute/snapmla_sm120_attention_device.h"
#include "compute/cuda_sm120_device_backend.h"
#include "compute/kernels/attention/mla_attention.h"
#include "compute/kernels/attention/tq_mla_attention.h"  // translate_indices (codec-agnostic)
#include "compute/kernels/attention/linearize_block_tables.h"
#include "compute/prefill_params.h"
#include "compute/graphs/dcp_allreduce_graph.h"
#include "compute/graphs/graph_registry.h"
#include "compute/kernels/sm120/attention/prep_params.h"
#include "config/config_parser.h"

#include <sm120/graph/decode_graph.h>
#include <sm120/decode/sparse_fp8/params.h>
#include <sm120/prep/fused_q_quant.h>
#include <smxx/mla_combine.h>
#include <smxx/params.h>

#include <algorithm>
#include <any>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace layerstorm::compute {

class SnapMlaSm120AttentionDevice final : public AttentionDevice {
public:
    explicit SnapMlaSm120AttentionDevice(config::GpuRef gpu)
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
                "SnapMlaSm120AttentionDevice: cudaDeviceGetAttribute("
                "MultiProcessorCount) failed for CUDA device "
                + std::to_string(gpu.id) + ": "
                + cudaGetErrorString(err));
        }
        num_sm_ = sm;

        // TD-SNAPMLA-ATCTX-DECODE-PATH env gates (read once at construction):
        //   LS_SNAPMLA_BUDGET_STAGING (default ON; 0 = full-context staging)
        //     — stage (a), bit-identical budget-bound dequant staging.
        //   LS_SNAPMLA_FP8_DECODE (default ON since P-29 step 14 / OQ-7,
        //     user-granted 2026-09-04: +60–66% @>=8k at −0.0063 nats vs the
        //     exact route, keeping +0.0219 of snapMLA's edge over TQ;
        //     0 = off) — stage (b), the FP8 split-KV decode kernel
        //     (different numerics profile). LS_REFERENCE_TRAJECTORY_IDENTITY=1
        //     forces this OFF (the tier's canonical numerics stay exact —
        //     src/core/determinism.cpp).
        //   LS_SNAPMLA_NSP (default 0 = auto: one 64-row block per split)
        //     — split count request for stage (b).
        if (const char* e = std::getenv("LS_SNAPMLA_BUDGET_STAGING"))
            budget_staging_on_ = !(e[0] == '0' && e[1] == '\0');
        if (const char* e = std::getenv("LS_SNAPMLA_FP8_DECODE"))
            fp8_decode_on_ = !(e[0] == '0' && e[1] == '\0');
        if (const char* e = std::getenv("LS_SNAPMLA_NSP"))
            nsp_request_ = std::atoi(e);
    }

    ~SnapMlaSm120AttentionDevice() override {
        if (prefill_kv_staging_) device_.device_free(prefill_kv_staging_);
        if (prefill_indices_scratch_) device_.device_free(prefill_indices_scratch_);
        if (prefill_seq_offsets_scratch_) device_.device_free(prefill_seq_offsets_scratch_);
        if (prefill_num_fetch_scratch_) device_.device_free(prefill_num_fetch_scratch_);
        free_decode_scratch();
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
    void gguf_mmvq_multi(const GgufGemmMultiParams& params,
                         void* q8_1_workspace, void* stream) override {
        device_.gguf_mmvq_multi(params, q8_1_workspace, stream);
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
    void cast_f32_to_bf16(void* dst_bf16, const void* src_f32,
                          int64_t count, void* stream) override {
        device_.cast_f32_to_bf16(dst_bf16, src_f32, count, stream);
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
    void indexer_kpool_append(const IndexerKpoolAppendArgs& a,
                              void* s) override {
        device_.indexer_kpool_append(a, s);
    }
    void indexer_kpool_chunk_append(const IndexerKpoolChunkAppendArgs& a,
                                    void* s) override {
        device_.indexer_kpool_chunk_append(a, s);
    }
    void indexer_kpool_expand(const IndexerKpoolExpandArgs& a,
                              void* s) override {
        device_.indexer_kpool_expand(a, s);
    }
    void indexer_kpool_merge(const IndexerKpoolMergeArgs& a,
                             void* s) override {
        device_.indexer_kpool_merge(a, s);
    }

    // GF3.9 KDA linear attention (glm5_next) — forwarded verbatim, like the
    // kpool ops: the backend translates the CUDA-free PODs and launches the
    // GF3.7 wrappers on the caller's (kAttention) stream.
    void kda_conv_prefill(const KdaConvPrefillArgs& a, void* s) override {
        device_.kda_conv_prefill(a, s);
    }
    void kda_conv_decode(const KdaConvDecodeArgs& a, void* s) override {
        device_.kda_conv_decode(a, s);
    }
    void kda_chunked_scan(const KdaChunkedScanArgs& a, void* s) override {
        device_.kda_chunked_scan(a, s);
    }
    void kda_gated_rmsnorm(const KdaGatedRmsNormArgs& a, void* s) override {
        device_.kda_gated_rmsnorm(a, s);
    }
    void kda_decode_step(const KdaDecodeStepArgs& a, void* s) override {
        device_.kda_decode_step(a, s);
    }
    size_t kda_prefill_workspace_bytes(int t_len,
                                       int num_heads) const override {
        return device_.kda_prefill_workspace_bytes(t_len, num_heads);
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
        int /*layer_idx*/, void* stream) override
    {
        sm120::prep::FusedKAppendParams fka{};
        fka.c_kv = static_cast<const __nv_bfloat16*>(c_kv);
        fka.k_rope = static_cast<const __nv_bfloat16*>(k_rope);
        fka.src_stride_ckv = c_kv_row_stride;
        fka.src_stride_rope = k_rope_row_stride;
        fka.kv_cache = static_cast<__nv_fp8_e4m3*>(kv_cache);
        fka.cache_stride_block = cache_stride_block;
        fka.cache_stride_row = cache_stride_row;
        fka.slot_mapping = slot_mapping;
        fka.num_tokens = num_tokens;
        fka.d_c = d_c;
        fka.d_rope = d_rope;
        fka.page_size = page_size;
        launch_fused_k_append(fka, static_cast<cudaStream_t>(stream));
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
        int /*layer_idx*/, void* stream) override
    {
        assert(d_c_ > 0 && "snapmla_device_set_model_dims not called");
        auto s = static_cast<cudaStream_t>(stream);

        // Step 1: Linearize block_tables → flat slot indices for dequant.
        //
        // chunk_causal: stage the UNION prefix [0, seq_len_kv) exactly ONCE by
        // linearizing only the LAST batch row (its prefix IS the union — see
        // the chunk_causal contract in attention_device.h). Linearizing all B
        // rows would concatenate per-row prefixes while the dequant below
        // fetches only the first seq_len_kv staging rows (the original
        // TD-PREFILL-CHUNK-ATTN staging bug) — and would cost O(B·seq_len_kv).
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

        // ── B=1 sparse decode gate (TD-SNAPMLA-ATCTX-DECODE-PATH) ──────────
        // The exact TQ decode gate (tq_sm120_attention_device.cpp:483 — kept
        // in lockstep): a single query row attending its own DSA selection.
        // `!chunk_causal` excludes the multi-row union-prefix semantics;
        // per-row prefill sub-dispatches (tiered_chunk / mixed_rows sparse
        // legs) match too and are equally covered — each is a genuine s_q=1
        // sparse call over its own selection.
        const bool b1_sparse_decode =
            is_sparse && batch_size == 1 && !chunk_causal && sparse_indices;

        // Stage (b): budget-bound FP8 split-KV decode (LS_SNAPMLA_FP8_DECODE,
        // DEFAULT ON since P-29 step 14 / OQ-7 — the champion decode route;
        // =0 falls back to the exact dequant route below).
        // A DIFFERENT numerics profile from the dequant route
        // (FP8 Q / FP8 P / per-tile V requant vs BF16 staging) — quality
        // barred by the TF-NLL harness, never silently substituted.
        if (b1_sparse_decode && fp8_decode_on_ && d_rope_ == 0 && d_c_ == 512
            && kv_cache && prefill_indices_scratch_) {
            fp8_sparse_decode(q_compressed, seq_len_kv, kv_cache,
                              cache_stride_block, cache_stride_row, page_size,
                              sparse_indices, topk_lengths, topk, out, lse, s);
            return;
        }

        // Step 2: Dequant FP8 paged cache → contiguous BF16 staging.
        if (kv_cache && prefill_kv_staging_ && prefill_indices_scratch_) {
            sm120::prep::DequantCKVIndexedParams dq{};
            dq.kv_cache = static_cast<const __nv_fp8_e4m3*>(kv_cache);
            dq.cache_stride_block = cache_stride_block;
            dq.cache_stride_row = cache_stride_row;
            dq.page_size = page_size;
            dq.indices = prefill_indices_scratch_;
            dq.num_fetch = seq_len_kv;
            dq.k_out = static_cast<__nv_bfloat16*>(prefill_kv_staging_);
            dq.d_c = d_c_;
            dq.d_rope = d_rope_;
            if (b1_sparse_decode && budget_staging_on_) {
                // Stage (a): budget-bound staging — dequant ONLY the selected
                // rows (scatter to their original staging positions).
                // BIT-IDENTICAL to the full-context staging: the sparse
                // kernel's read set is bounded by the same predicate the
                // gather mirrors (sValid: g < topk_length && 0 <= idx < s_kv),
                // masked lanes substitute a compile-time constant, and the
                // per-row dequant body is the shared device function. Kills
                // the O(ctx) staging law (seq_len_kv → topk rows).
                dq.gather_rows = sparse_indices;
                dq.gather_count = topk_lengths;  // nullptr → max_gather rows
                dq.max_gather = topk;
                dq.row_bound = seq_len_kv;
                if (!logged_budget_) {
                    logged_budget_ = true;
                    std::fprintf(stderr,
                                 "[snapmla] B=1 sparse decode: budget-bound "
                                 "staging ON (gather %d of %d rows/step; "
                                 "LS_SNAPMLA_BUDGET_STAGING=0 restores "
                                 "full-context staging)\n",
                                 topk, seq_len_kv);
                }
            }
            launch_dequant_ckv_indexed(dq, s);
        }

        // Step 3: Launch prefill kernel with fully populated params.
        const PrefillDims dims{d_c_, d_rope_, h_q_, num_sm_, sm_scale_};

        if (!is_sparse) {
            sm120::prefill::dense::head64::DenseAttnFwdParams p{};
            populate_dense_prefill_params(p, dims, q_compressed,
                prefill_kv_staging_, batch_size, seq_len_kv, out, lse, s);
            p.deterministic_reduce = deterministic_reduce_;  // DET-REDUCE gate
            // chunk_causal: per-query-row causal bound — row b attends the
            // staged union prefix [0, seqlens_k[b]).
            if (chunk_causal) p.s_kv_per_row = seqlens_k;
            launch_prefill_dense(p);
        } else {
            SparseAttnFwdParams p{};
            populate_sparse_prefill_params(p, dims, q_compressed,
                prefill_kv_staging_, sparse_indices, topk_lengths, topk,
                batch_size, seq_len_kv, out, lse, s);
            p.deterministic_reduce = deterministic_reduce_;  // DET-REDUCE gate
            // chunk_causal × sparse (TD-SPARSE-CHUNK-PREFILL): per-query-row
            // causal bound — row b attends only its selected indices inside
            // the staged union prefix [0, seqlens_k[b])
            // (INV-SPARSE-CHUNK-CAUSAL).
            if (chunk_causal) p.s_kv_per_row = seqlens_k;
            launch_prefill_sparse(p);
        }
    }

    // ── Decode graph ops ────────────────────────────────────────────────────

    void decode_graph_update(
        GraphEntry& entry, const void* q_bf16,
        const int* seqlens_k, const int* block_table,
        const int* indices,
        int /*layer_idx*/, void* stream) override
    {
        auto** p = std::any_cast<sm120::graph::DecodeGraphRunner*>(&entry.runner);
        if (p && *p) {
            (*p)->update(q_bf16, seqlens_k, block_table, indices,
                         static_cast<cudaStream_t>(stream));
        }
    }

    void decode_graph_replay(
        GraphEntry& entry, void* stream) override
    {
        auto** p = std::any_cast<sm120::graph::DecodeGraphRunner*>(&entry.runner);
        if (p && *p) {
            (*p)->replay(static_cast<cudaStream_t>(stream));
        }
    }

    void* decode_graph_out_ptr(GraphEntry& entry) override {
        auto** p = std::any_cast<sm120::graph::DecodeGraphRunner*>(&entry.runner);
        return (p && *p) ? static_cast<void*>((*p)->out_ptr()) : nullptr;
    }

    float* decode_graph_lse_ptr(GraphEntry& entry) override {
        auto** p = std::any_cast<sm120::graph::DecodeGraphRunner*>(&entry.runner);
        return (p && *p) ? (*p)->lse_ptr() : nullptr;
    }

    // ── DCP allreduce graph ─────────────────────────────────────────────────

    void dcp_graph_replay(GraphEntry& entry, void* stream) override {
        auto** p = std::any_cast<DcpAllreduceGraphRunner*>(&entry.runner);
        if (p && *p) {
            (*p)->replay(stream);
        }
    }

    // ── KD-4f-d.1a: model dims + prefill scratch (set at init via free functions) ──

    void set_model_dims(int batch_size, int d_c, int d_rope, int h_q,
                        float sm_scale) {
        batch_size_ = batch_size;
        d_c_ = d_c;
        d_rope_ = d_rope;
        h_q_ = h_q;
        sm_scale_ = sm_scale;  // 0 → legacy 1/√d_qk in populate_*_prefill_params
    }

    // DET-REDUCE: enable the gated deterministic (bit-reproducible) softmax
    // denominator reduction for this device's attention kernels.
    void set_deterministic_reduce(bool enable) { deterministic_reduce_ = enable; }

    void set_prefill_scratch(int max_kv_tokens) {
        assert(d_c_ > 0 && "set_model_dims must be called before set_prefill_scratch");
        // TD-71h: skip re-allocation when capacity already suffices
        // (mirrors the TQ twin).
        if (max_kv_tokens <= prefill_scratch_capacity_) return;
        device_.set_device();

        // Free existing (re-entrant for capacity changes)
        if (prefill_kv_staging_) { device_.device_free(prefill_kv_staging_); prefill_kv_staging_ = nullptr; }
        if (prefill_indices_scratch_) { device_.device_free(prefill_indices_scratch_); prefill_indices_scratch_ = nullptr; }
        if (prefill_seq_offsets_scratch_) { device_.device_free(prefill_seq_offsets_scratch_); prefill_seq_offsets_scratch_ = nullptr; }
        if (prefill_num_fetch_scratch_) { device_.device_free(prefill_num_fetch_scratch_); prefill_num_fetch_scratch_ = nullptr; }

        const size_t kv_staging_bytes = static_cast<size_t>(max_kv_tokens) * (d_c_ + d_rope_) * 2;  // BF16
        prefill_kv_staging_ = device_.device_alloc(kv_staging_bytes);
        prefill_indices_scratch_ = static_cast<int*>(device_.device_alloc(
            static_cast<size_t>(max_kv_tokens) * sizeof(int)));
        prefill_seq_offsets_scratch_ = static_cast<int*>(device_.device_alloc(
            static_cast<size_t>(batch_size_) * sizeof(int)));
        prefill_num_fetch_scratch_ = static_cast<int*>(device_.device_alloc(sizeof(int)));
        // FAIL-LOUD (1M-cap lesson): kv staging is max_kv × 576 × 2 B —
        // 1.2 GB at max_sequence_length=1M — and prefill_attention treats a
        // null staging pointer as "skip the dequant" (stale-read corruption,
        // not an error). Abort init instead.
        if (!prefill_kv_staging_ || !prefill_indices_scratch_
            || !prefill_seq_offsets_scratch_ || !prefill_num_fetch_scratch_) {
            throw std::runtime_error(
                "SnapMlaSm120AttentionDevice: prefill scratch allocation "
                "failed (" + std::to_string(kv_staging_bytes)
                + " B staging — scales with serving.max_sequence_length)");
        }

        prefill_scratch_capacity_ = max_kv_tokens;  // TD-71h
    }

private:
    // ── Stage (b): budget-bound FP8 split-KV sparse decode ──────────────────
    // (TD-SNAPMLA-ATCTX-DECODE-PATH; the dormant decode/sparse_fp8 kernel
    // stack wired eagerly — chain mirrors graph/decode_graph.cu + the
    // csa_hca sparse splitkv orchestration.)

    void free_decode_scratch() {
        if (dec_indices_) { device_.device_free(dec_indices_); dec_indices_ = nullptr; }
        if (dec_q_fp8_) { device_.device_free(dec_q_fp8_); dec_q_fp8_ = nullptr; }
        if (dec_q_scales_) { device_.device_free(dec_q_scales_); dec_q_scales_ = nullptr; }
        if (dec_o_accum_) { device_.device_free(dec_o_accum_); dec_o_accum_ = nullptr; }
        if (dec_lse_accum_) { device_.device_free(dec_lse_accum_); dec_lse_accum_ = nullptr; }
        if (dec_meta_) { device_.device_free(dec_meta_); dec_meta_ = nullptr; }
        if (dec_splits_) { device_.device_free(dec_splits_); dec_splits_ = nullptr; }
        dec_topk_pad_ = 0;
        dec_nsp_ = 0;
    }

    // Exact integer replica of deps smxx/get_mla_metadata.cu with topk > 0
    // (uniform), block_size_n = 64, fixed_overhead = 1, rows = 1 — copied
    // from the proven csa_hca sparse_meta host replica so the {meta, splits}
    // content can never disagree with what the metadata kernel would write.
    static void sched_meta_host(int topk, int nsp,
                                std::vector<int>& meta,
                                std::vector<int>& splits) {
        constexpr int rows = 1;
        const int block_size_n = 64;
        const int fixed_overhead = 1;
        const int last_block_idx = (std::max(topk - 1, 0)) / block_size_n;
        const int num_blocks = last_block_idx + 1;
        const int total_num_blocks = rows * (num_blocks + fixed_overhead);
        const int payload = (total_num_blocks + nsp - 1) / nsp + fixed_overhead;

        meta.assign(static_cast<size_t>(nsp) * TileSchedulerMetaDataSize, 0);
        splits.assign(rows + 1, 0);
        int now_idx = 0, now_block = 0, now_n_split = 0, cum = 0;
        for (int i = 0; i < nsp; ++i) {
            int* m = meta.data() +
                     static_cast<size_t>(i) * TileSchedulerMetaDataSize;
            m[0] = now_idx;
            m[1] = now_block;
            m[4] = now_n_split;
            int remain = payload;
            while (now_idx < rows) {
                int rem_blocks = num_blocks - now_block;
                if (remain >= rem_blocks + fixed_overhead) {
                    cum += now_n_split + 1;
                    splits[now_idx + 1] = cum;
                    remain -= rem_blocks + fixed_overhead;
                    ++now_idx;
                    now_block = 0;
                    now_n_split = 0;
                } else {
                    if (remain - fixed_overhead > 0) {
                        now_block += remain - fixed_overhead;
                        ++now_n_split;
                        remain = 0;
                    }
                    break;
                }
            }
            m[2] = now_block > 0 ? now_idx : now_idx - 1;
            m[3] = now_block > 0 ? now_block : last_block_idx + 1;
        }
        if (now_idx != rows || now_block != 0 || now_n_split != 0) {
            throw std::runtime_error(
                "snapmla sched_meta_host: scheduler replica did not consume "
                "all work (topk=" + std::to_string(topk) +
                " nsp=" + std::to_string(nsp) + ")");
        }
    }

    void ensure_decode_scratch(int topk_pad) {
        const int num_blocks = topk_pad / 64;
        // Auto split request: one 64-row block per split (num_blocks + 1
        // parts makes the scheduler payload exactly one block); combine cap
        // 192 (deps mla_combine MAX_SPLITS).
        int nsp = nsp_request_ > 0 ? nsp_request_ : num_blocks + 1;
        nsp = std::max(1, std::min(nsp, 192));
        if (dec_topk_pad_ == topk_pad && dec_nsp_ == nsp) return;
        free_decode_scratch();

        dec_indices_ = static_cast<int*>(device_.device_alloc(
            static_cast<size_t>(topk_pad) * sizeof(int)));
        dec_q_fp8_ = device_.device_alloc(
            static_cast<size_t>(h_q_) * d_c_);  // FP8 = 1 B/elem
        dec_q_scales_ = static_cast<float*>(device_.device_alloc(
            static_cast<size_t>(h_q_) * sizeof(float)));
        dec_o_accum_ = static_cast<float*>(device_.device_alloc(
            static_cast<size_t>(nsp) * h_q_ * d_c_ * sizeof(float)));
        dec_lse_accum_ = static_cast<float*>(device_.device_alloc(
            static_cast<size_t>(nsp) * h_q_ * sizeof(float)));
        dec_meta_ = static_cast<int*>(device_.device_alloc(
            static_cast<size_t>(nsp) * TileSchedulerMetaDataSize
            * sizeof(int)));
        dec_splits_ = static_cast<int*>(device_.device_alloc(2 * sizeof(int)));
        if (!dec_indices_ || !dec_q_fp8_ || !dec_q_scales_ || !dec_o_accum_
            || !dec_lse_accum_ || !dec_meta_ || !dec_splits_) {
            throw std::runtime_error(
                "SnapMlaSm120AttentionDevice: FP8 sparse-decode scratch "
                "allocation failed (INV-SCRATCH-LOUD — a silent fallback to "
                "the dequant chain would switch decode numerics mid-run)");
        }
        // One-time -1 fill: translate writes [0, topk) every step; the
        // 64-alignment pad [topk, topk_pad) must stay -1 forever (the kernel
        // walks whole 64-row blocks; -1 rows are masked, and the fixed
        // rescale guard makes fully-masked blocks a no-op).
        cudaMemset(dec_indices_, 0xFF,
                   static_cast<size_t>(topk_pad) * sizeof(int));

        std::vector<int> meta, splits;
        sched_meta_host(topk_pad, nsp, meta, splits);
        device_.memcpy_h2d(dec_meta_, meta.data(), meta.size() * sizeof(int));
        device_.memcpy_h2d(dec_splits_, splits.data(),
                           splits.size() * sizeof(int));
        dec_total_splits_ = splits[1];
        dec_topk_pad_ = topk_pad;
        dec_nsp_ = nsp;
        if (!logged_fp8_) {
            logged_fp8_ = true;
            std::fprintf(stderr,
                         "[snapmla] B=1 sparse decode: FP8 split-KV kernel ON "
                         "(topk_pad=%d, nsp=%d, splits=%d, det=%d; "
                         "LS_SNAPMLA_FP8_DECODE unset/0 restores the "
                         "dequant+prefill route)\n",
                         topk_pad, nsp, dec_total_splits_,
                         deterministic_reduce_ ? 1 : 0);
        }
    }

    void fp8_sparse_decode(const void* q_compressed, int seq_len_kv,
                           void* kv_cache, int64_t cache_stride_block,
                           int cache_stride_row, int page_size,
                           const int* sparse_indices, const int* topk_lengths,
                           int topk, void* out, float* lse, cudaStream_t s) {
        const int topk_pad = ((topk + 63) / 64) * 64;
        ensure_decode_scratch(topk_pad);
        const int d_qk = d_c_ + d_rope_;
        const int d_v = d_c_;  // absorbed geometry: V is the latent

        // 1. Translate DSA token positions → pool slots (composes the
        // linearization above); -1 outside topk_length/causal bound. The
        // [topk, topk_pad) tail is the persistent -1 fill.
        launch_tq_sparse_translate_indices(
            sparse_indices, prefill_indices_scratch_, topk_lengths, topk,
            seq_len_kv, /*seq_len_dev=*/nullptr, dec_indices_, s);

        // 2. Quantize Q: BF16 → FP8 NOPE + per-head scales (rope leg empty
        // at d_rope == 0; q_rope output stays null).
        sm120::prep::FusedQQuantParams qq{};
        qq.q_bf16 = static_cast<const __nv_bfloat16*>(q_compressed);
        qq.q_nope_fp8 = static_cast<__nv_fp8_e4m3*>(dec_q_fp8_);
        qq.q_rope_bf16 = nullptr;
        qq.q_scales = dec_q_scales_;
        qq.s_q = 1;
        qq.h_q = h_q_;
        qq.d_qk = d_qk;
        qq.d_nope = d_c_;
        launch_fused_q_quant(qq, s);

        // 3. The split-KV sparse FP8 kernel (flat top-k; reads the paged FP8
        // cache directly — no staging).
        sm120::decode::sparse_fp8::SparseAttnDecodeParams p{};
        p.b = 1;
        p.s_q = 1;
        p.h_q = h_q_;
        p.h_kv = 1;
        p.d_qk = d_qk;
        p.d_v = d_v;
        p.sm_scale = sm_scale_ > 0.0f
            ? sm_scale_ : 1.0f / std::sqrt(static_cast<float>(d_qk));
        p.sm_scale_div_log2 = p.sm_scale * 1.44269504088896340736f;
        p.num_blocks = 0;  // unread by the sparse kernel
        p.page_block_size = page_size;
        p.topk = topk_pad;  // 64-aligned walk; tail rows are -1 (masked)
        p.model_type = sm120::decode::sparse_fp8::ModelType::GLM5N;
        p.q = reinterpret_cast<cutlass::bfloat16_t*>(dec_q_fp8_);  // FP8 bytes
        p.q_rope = nullptr;
        p.q_scales = dec_q_scales_;
        p.kv = static_cast<cutlass::bfloat16_t*>(kv_cache);  // FP8 bytes
        p.indices = dec_indices_;
        p.topk_length = nullptr;   // flat top-k ignores it; -1 mask governs
        p.attn_sink = nullptr;
        p.lse = lse;
        p.out = static_cast<cutlass::bfloat16_t*>(out);
        p.stride_q_b = p.stride_q_s_q = h_q_ * d_qk;
        p.stride_q_h_q = d_qk;
        p.stride_kv_block = static_cast<int>(cache_stride_block);  // bytes
        p.stride_kv_row = cache_stride_row;                        // bytes
        p.stride_indices_b = p.stride_indices_s_q = topk_pad;
        p.stride_lse_b = p.stride_lse_s_q = h_q_;
        p.stride_o_b = p.stride_o_s_q = h_q_ * d_v;
        p.stride_o_h_q = d_v;
        p.lse_accum = dec_lse_accum_;
        p.o_accum = dec_o_accum_;
        p.stride_lse_accum_split = h_q_;       // num_q_seqs = s_q * h_q
        p.stride_lse_accum_s_q = h_q_;
        p.stride_o_accum_split = h_q_ * d_v;
        p.stride_o_accum_s_q = h_q_ * d_v;
        p.stride_o_accum_h_q = d_v;
        p.tile_scheduler_metadata_ptr =
            reinterpret_cast<sm120::decode::sparse_fp8::DecodingSchedMeta*>(
                dec_meta_);
        p.num_splits_ptr = dec_splits_;
        p.num_sm_parts = dec_nsp_;
        p.deterministic_reduce = deterministic_reduce_;
        p.stream = s;
        launch_decode_sparse_fp8(p);

        // 4. Merge split partials (fixed-order, deterministic). With one
        // split the kernel wrote out/lse directly and the combine early-outs
        // — skip the launch entirely.
        if (dec_total_splits_ > 1) {
            MlaCombineParams cp{};
            cp.b = 1;
            cp.h_q = h_q_;
            cp.h_k = 1;
            cp.q_seq_per_hk = h_q_;  // h_q / h_k * s_q
            cp.d_v = d_v;
            cp.o_ptr = out;
            cp.softmax_lse_ptr = lse;
            cp.o_batch_stride = static_cast<int64_t>(h_q_) * d_v;
            cp.o_row_stride = d_v;
            cp.o_head_stride = d_v;  // s_q=1: heads are the row dimension
            cp.num_splits_ptr = dec_splits_;
            cp.num_sm_parts = dec_nsp_;
            cp.softmax_lseaccum_ptr = dec_lse_accum_;
            cp.oaccum_ptr = dec_o_accum_;
            launch_mla_combine(cp, s);
        }
    }

    CudaSm120DeviceBackend device_;

    // Model dimensions (KD-4f-d.1a, set at init)
    int batch_size_ = 0;
    int d_c_ = 0;          // kv_lora_rank
    int d_rope_ = 0;       // qk_rope_head_dim
    int h_q_ = 0;          // num_heads_local (after TP split)
    float sm_scale_ = 0.0f;  // model softmax scale ((qk_nope+qk_rope)^-0.5 · mscale²)
    int num_sm_ = 0;       // GPU SM count
    bool deterministic_reduce_ = false;  // DET-REDUCE gate (config/env, set at init)

    // Prefill scratch buffers (KD-4f-d.1a)
    void* prefill_kv_staging_ = nullptr;         // [max_kv, d_c + d_rope] BF16
    int*  prefill_indices_scratch_ = nullptr;    // [max_kv] int
    int*  prefill_seq_offsets_scratch_ = nullptr; // [batch_size] int
    int*  prefill_num_fetch_scratch_ = nullptr;  // [1] int
    int   prefill_scratch_capacity_ = 0;         // TD-71h: tokens covered

    // ── B=1 sparse decode (TD-SNAPMLA-ATCTX-DECODE-PATH) ────────────────────
    bool budget_staging_on_ = true;   // LS_SNAPMLA_BUDGET_STAGING (stage a)
    bool fp8_decode_on_ = true;       // LS_SNAPMLA_FP8_DECODE (stage b);
                                      // DEFAULT ON since P-29 step 14 (OQ-7)
    int  nsp_request_ = 0;            // LS_SNAPMLA_NSP (0 = auto)
    bool logged_budget_ = false;
    bool logged_fp8_ = false;
    // Stage (b) scratch (allocated at first decode; shapes fixed per boot)
    int*   dec_indices_ = nullptr;    // [topk_pad] pool slots, -1 padded
    void*  dec_q_fp8_ = nullptr;      // [h_q, d_c] FP8
    float* dec_q_scales_ = nullptr;   // [h_q]
    float* dec_o_accum_ = nullptr;    // [nsp, h_q, d_v] f32
    float* dec_lse_accum_ = nullptr;  // [nsp, h_q] f32 (log2 units)
    int*   dec_meta_ = nullptr;       // [nsp, TileSchedulerMetaDataSize]
    int*   dec_splits_ = nullptr;     // [b+1] = [2]
    int    dec_topk_pad_ = 0;
    int    dec_nsp_ = 0;
    int    dec_total_splits_ = 0;
};

// ── Factory ─────────────────────────────────────────────────────────────────

std::unique_ptr<AttentionDevice> make_snapmla_sm120_attention_device(
        config::GpuRef gpu) {
    return std::make_unique<SnapMlaSm120AttentionDevice>(std::move(gpu));
}

// ── KD-4f-d.1a: Free-function bridges (same pattern as TQ R0H-1d) ─────────

void snapmla_device_set_model_dims(AttentionDevice* dev,
    int batch_size, int d_c, int d_rope, int h_q, float sm_scale) {
    static_cast<SnapMlaSm120AttentionDevice*>(dev)->set_model_dims(
        batch_size, d_c, d_rope, h_q, sm_scale);
}

void snapmla_device_set_prefill_scratch(AttentionDevice* dev, int max_kv_tokens) {
    static_cast<SnapMlaSm120AttentionDevice*>(dev)->set_prefill_scratch(max_kv_tokens);
}

void snapmla_device_set_deterministic_reduce(AttentionDevice* dev, bool enable) {
    static_cast<SnapMlaSm120AttentionDevice*>(dev)->set_deterministic_reduce(enable);
}

}  // namespace layerstorm::compute
