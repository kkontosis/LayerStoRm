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
#include "compute/kernels/attention/linearize_block_tables.h"
#include "compute/prefill_params.h"
#include "compute/graphs/dcp_allreduce_graph.h"
#include "compute/graphs/graph_registry.h"
#include "compute/kernels/sm120/attention/prep_params.h"
#include "config/config_parser.h"

#include <sm120/graph/decode_graph.h>

#include <any>
#include <cassert>
#include <stdexcept>
#include <string>
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
    }

    ~SnapMlaSm120AttentionDevice() override {
        if (prefill_kv_staging_) device_.device_free(prefill_kv_staging_);
        if (prefill_indices_scratch_) device_.device_free(prefill_indices_scratch_);
        if (prefill_seq_offsets_scratch_) device_.device_free(prefill_seq_offsets_scratch_);
        if (prefill_num_fetch_scratch_) device_.device_free(prefill_num_fetch_scratch_);
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
