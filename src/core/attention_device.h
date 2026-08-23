// Abstract interface for per-GPU attention device.
//
// AttentionDevice merges hardware ops (GEMM, RMSNorm, FP8 quantization,
// device memory) with attention pipeline ops (KV cache append, prefill,
// decode graph, DCP allreduce graph) into a single per-GPU abstraction.
//
// Each TP GPU has its own instance (INV-BH-1).  Algorithm selection
// (SnapMLA vs TurboQuant) is baked into the concrete type at construction
// (INV-BH-7) — no runtime dispatch or registry lookup after init.
//
// The abstract interface uses void* streams and generic POD param structs —
// no CUDA/HIP/ROCm types — compilable without any GPU SDK (INV-BH-2).
//
// Concrete implementations:
//   SnapMlaSm120AttentionDevice  — SM120 + SnapMLA FP8 MLA
//   TqSm120AttentionDevice       — SM120 + TurboQuant 4-bit MLA
//   NullAttentionDevice           — no-op for unit tests

#pragma once

#include "core/gpu_ref.h"
#include "sm120/gemm/fp8/fp8_gemm.h"
#include "sm120/gemm/grouped_gemm.h"
#include "smxx/quant/dynamic_fp8_quant.h"
#include "smxx/quant/weight_fp8_quant.h"
#include "smxx/quant/nvfp4_dequant_bf16.h"
#include "smxx/quant/bf16_to_nvfp4_grouped.h"

#include <cstddef>
#include <cstdint>
#include <memory>

// Forward-declare only the enum we need — avoids pulling config_parser.h
// into CUDA kernel TUs that lack the include path.
namespace layerstorm::config { enum class AttentionBackendType : int; }

// Forward-declare GgufKQuantType (defined in model/quantization/gguf_kquant.h).
// That header pulls quant_interface.h → config_parser.h → nlohmann/json, which
// is NOT on the CUDA kernel TUs' include path; the GgufGemmParams below only
// needs the enum name (the backend .cu casts it to int via its fixed underlying
// type), so a forward declaration keeps attention_device.h CUDA-TU-safe.
namespace layerstorm::model { enum class GgufKQuantType : int; }

namespace layerstorm::compute {

// Forward-declare GraphEntry (defined in graph_registry.h).
struct GraphEntry;
// Forward-declare KvBvExtractDequantParams (defined in kv_bv_extract_dequant.h).
struct KvBvExtractDequantParams;

/// Parameters for BF16 strided batched GEMM (per-head kv_b_v projection):
/// [HL, B, D_c] x [HL, D_c, V] -> [HL, B, V]. Implemented by the self-contained
/// bf16_gemm.cu kernel (FP32 accumulate, graph-capturable); formerly cuBLAS.
///
/// Per batch the kernel computes the NT contraction
///   C[n*ldc + m] = sum_k A[m*lda + k] * B[n*ldb + k]
/// i.e. A is [m, k] row-major (lda = k) and B is [n, k] row-major with the
/// head-interleaved leading dim ldb.
struct StridedBatchedGemmBf16Params {
    int m;               ///< output rows per batch (v_head_dim)
    int n;               ///< output cols per batch (batch_size)
    int k;               ///< inner dim (kv_lora_rank)
    const void* A;       ///< weight: BF16, per-batch shape [m, k] row-major
    int lda;             ///< leading dim of A in memory (= k for row-major [m, k])
    int64_t strideA;     ///< element stride between batches in A
    const void* B;       ///< input: BF16, per-batch shape [n, k] with head-interleaved layout
    int ldb;             ///< leading dim of B (= batch_count * k for [n, batch_count, k])
    int64_t strideB;     ///< element stride between batches in B
    void* C;             ///< output: BF16, per-batch shape [n, m] with head-interleaved layout
    int ldc;             ///< leading dim of C (= batch_count * m for [n, batch_count, m])
    int64_t strideC;     ///< element stride between batches in C
    int batch_count;     ///< number of batches (num_heads_local)
};

/// Parameters for the MLA W_UK query absorption (q_absorb kernel).
/// Folds the K-up projection into the query so attention runs in the kv_lora latent
/// space: ql_nope[s,h,k] = sum_d q_heads_nope[s,h,d] * W_UK[h,d,k]; the rope half is
/// copied through. Output q_absorbed = [s_q, h_q, d_c + d_rope] BF16.
struct QAbsorbParams {
    const void* q_heads;          ///< [s_q, h_q, d_nope_in + d_rope] BF16 (q_b_proj output)
    const void* kv_b_proj;        ///< [h_q*(d_nope_in + d_v), d_c] BF16 or FP8 E4M3 (W_UK = first d_nope_in rows/head)
    const void* kv_b_proj_scales; ///< FP8 blockwise scales (K-major); nullptr if BF16
    void* q_absorbed;             ///< [s_q, h_q, d_c + d_rope] BF16 output
    int  s_q;
    int  h_q;
    int  d_nope_in;               ///< qk_nope_head_dim (contraction dim)
    int  d_c;                     ///< kv_lora_rank (absorbed output dim)
    int  d_rope;                  ///< qk_rope_head_dim
    int  d_v;                     ///< v_head_dim (sets kv_b_proj per-head row stride)
    bool weight_is_fp8;

    // GG-7: GGUF W_UK path selector. -1 = not GGUF (use the BF16/FP8 branch
    // above); otherwise a model::GgufKQuantType value cast to int (0=Q2_K …
    // 5=Q8_0). When set, kv_b_proj is a packed GGUF k-quant and the kernel
    // dequants W_UK per element (dequant-only; kv_b_proj_scales ignored —
    // GGUF scales are in-block). Requires d_c (kv_lora_rank) % QK == 0.
    int  gguf_type = -1;

    // Optional fused RoPE on the rope half (interleaved adjacent-pair, DeepSeek
    // convention). pos = seqlens_k[s] − 1; cos_sin is the pure [max_pos][d_rope]
    // table (d_rope/2 cos | d_rope/2 sin per row). Off by default.
    bool        apply_rope = false;
    const void* seqlens_k = nullptr;   ///< [s_q] int32 device pointer
    const void* cos_sin = nullptr;     ///< [max_pos][d_rope] float32 device pointer
    int         max_pos = 0;
};

/// Parameters for the in-place strided RoPE rotation (rope_rotate kernel).
/// Rotates the d_rope-wide rope slice of each row by the owning token's position
/// (pos = seqlens_k[t] − 1) using the interleaved adjacent-pair convention.
/// `x` points at the FIRST rope element of row 0; row r starts at x + r*row_stride.
struct RopeRotateParams {
    void*       x;              ///< BF16, in-place
    const void* seqlens_k;      ///< [num_tokens] int32 device pointer
    const void* cos_sin;        ///< [max_pos][d_rope] float32 device pointer
    int     num_tokens;
    int     rows_per_token;     ///< 1 for k_pe, h_q for per-head q_pe
    int64_t row_stride;         ///< elements between consecutive rope rows
    int     d_rope;
    int     max_pos;
};

// GLM-25a: DSA lightning-indexer score+top-k device op (CUDA-free params). All
// pointers are device pointers.
struct IndexerScoreTopkArgs {
    const void* q_all;           ///< [num_tokens, n_heads*head_dim] BF16
    const void* k_cache;         ///< [num_blocks, head_dim] FP8 e4m3 (MQA single-K)
    const void* k_scales;        ///< [num_blocks] f32
    const void* score_proj_all;  ///< [num_tokens, n_heads] f32
    const void* block_endpoints; ///< [num_blocks] int32
    void* scores_scratch;        ///< [num_blocks] f32 (one query's scores)
    void* topk_scores_scratch;   ///< [topk] f32
    void* sparse_indices_out;    ///< [num_tokens, topk] int32
    void* topk_lengths_out;      ///< [num_tokens] int32
    int num_tokens;
    int num_blocks;
    int n_heads;
    int head_dim;
    int topk;
    int query_position_base = 0;  ///< query t's causal cutoff = base + t

    // TD-GLM-INDEXER-PAGED: paged indexer-K mode. When k_pages is set, K lives
    // in coarse pool pages instead of the contiguous k_cache/k_scales above:
    // k_pages is a HOST array of device page base pointers; each page holds
    // [page_tokens × head_dim] FP8 K rows followed by [page_tokens] F32 scales
    // (scales at base + page_tokens*head_dim). Scoring launches per page over
    // contiguous ranges; top-k is unchanged (scores_scratch spans all blocks).
    const void* const* k_pages = nullptr;  ///< [num_k_pages] device bases (host array)
    int num_k_pages = 0;
    int page_tokens = 0;                   ///< K rows per page
};

// TD-SPARSE-PREFILL-SCORE-BATCH: batched multi-row variant of
// IndexerScoreTopkArgs — ONE score launch + ONE top-k launch cover num_rows
// rows, each row r with its OWN causal block bound (row_num_blocks[r]) and
// causality cutoff (row_query_position[r]), both DEVICE int32 arrays. Row r
// reads q_all/score_proj_all row r, scores into scores_scratch row r (stride
// scores_stride floats) and writes sparse_indices_out + r*topk /
// topk_scores_out + r*topk / topk_lengths_out + r. Output is BIT-IDENTICAL
// to num_rows sequential indexer_score_topk calls (num_tokens=1 each) with
// the same per-row bounds. K storage is UNIFORM across rows: contiguous
// k_cache/k_scales (global block addressing) or, when k_page_table is set, a
// per-row DEVICE page-pointer table [num_rows, page_table_stride] whose
// pages hold [page_tokens × head_dim] FP8 rows + [page_tokens] f32 scales
// (same page layout as IndexerScoreTopkArgs paged mode). All pointers are
// device pointers.
struct IndexerScoreTopkBatchedArgs {
    const void* q_all;              ///< [num_rows, n_heads*head_dim] BF16
    const void* score_proj_all;     ///< [num_rows, n_heads] f32
    const void* row_num_blocks;     ///< [num_rows] int32 device — per-row bound
    const void* row_query_position; ///< [num_rows] int32 device — per-row cutoff
    const void* k_cache = nullptr;  ///< contiguous FP8 K (when not paged)
    const void* k_scales = nullptr; ///< contiguous f32 scales (when not paged)
    const void* k_page_table = nullptr; ///< [num_rows*page_table_stride] device ptr array
    int page_table_stride = 0;      ///< page slots per row in k_page_table
    int page_tokens = 0;            ///< K rows per page
    const void* block_endpoints;    ///< [max_num_blocks] int32 iota
    void* scores_scratch;           ///< [num_rows, scores_stride] f32
    int64_t scores_stride;          ///< floats between consecutive rows
    void* topk_scores_out;          ///< [num_rows, topk] f32
    void* sparse_indices_out;       ///< [num_rows, topk] int32
    void* topk_lengths_out;         ///< [num_rows] int32
    int num_rows;
    int max_num_blocks;             ///< max over row_num_blocks (grid sizing)
    int n_heads;
    int head_dim;
    int topk;
};

// TD-GLM-INDEXER-LOCAL-MERGE (dcp_indexer_mode=local): exact cross-rank merge
// of per-rank shard-local top-k candidate lists into the global top-k. All
// pointers are device pointers; CUDA-free params (mirrors
// IndexerScoreTopkArgs). `gathered` is the NCCL-allgather output: dcp_size
// rank-major segments of seg_words 32-bit words, each laid out
// [batch*topk int32 LOCAL indices][batch*topk f32 scores]; row `token` of
// every segment is merged. Local index l of rank n maps to global position
// (l / page_tokens * dcp_size + n) * page_tokens + l % page_tokens
// (round-robin-by-indexer-page ownership). The merge scatters candidates
// over a -inf-filled [num_blocks] scratch and re-runs the standard radix
// top-k — output = GLOBAL sorted-ascending indices, -1-padded, with
// effective length (identical to a replicated-indexer full-history top-k;
// see topk_merge.h for the exactness argument).
struct IndexerTopkMergeArgs {
    const void* gathered;        ///< [dcp_size * seg_words] int32/f32 words
    int seg_words;               ///< words per rank segment (= 2*batch*topk)
    int batch;                   ///< rows per segment
    int token;                   ///< row to merge
    void* scores_scratch;        ///< [num_blocks] f32 (overwritten)
    const void* block_endpoints; ///< [num_blocks] int32 global iota
    void* topk_scores_scratch;   ///< [topk] f32 (merge output scores, unused)
    void* indices_out;           ///< [topk] int32 GLOBAL, sorted asc, pad -1
    void* length_out;            ///< int32* effective length
    int num_blocks;              ///< global causal length for this row
    int topk;
    int query_position;          ///< causality cutoff (global)
    int dcp_size;
    int page_tokens;             ///< indexer_k_page_size_tokens
};

/// CUDA-free parameter struct for a GGUF-quantized linear GEMM (attention
/// projection). Mirrors how `Fp8GemmParams` is a plain POD passed across the
/// AttentionDevice seam so `dcp_executor` (a CUDA-free .cpp) can construct it
/// without including any kernel/CUDA header (INV-GPU-1). The concrete CUDA
/// backend converts this into the kernel's `GgufMmvqParams` /
/// `GgufDequantGemmParams` (and maps `model::GgufKQuantType` →
/// `compute::GgufType`) inside the .cu.
///
/// Computes  C[M,N] = A[M,K] @ dequant(B)^T  with A,C in BF16. The packed GGUF
/// weight B is row-major `[N, (K/QK)*block_bytes]` (one row = one output
/// channel). Requires K % QK == 0 (256 for k-quants, 32 for Q8_0); true for the
/// attention projection shapes (GG-4).
struct GgufGemmParams {
    int M;                       ///< rows of A / rows of C (tokens)
    int N;                       ///< output channels (rows of packed B)
    int K;                       ///< input dim (cols of A; weight values per row)
    const void* A;               ///< [M, K] BF16 activations, row-major
    const void* B;               ///< [N, (K/QK)*block_bytes] packed GGUF weight
    void*       C;               ///< [M, N] BF16 output, row-major
    model::GgufKQuantType type;  ///< per-projection GGUF k-quant type
};

class AttentionDevice {
public:
    virtual ~AttentionDevice() = default;

    // ── Device selection ────────────────────────────────────────────────────

    /// Activate this device's GPU as the current device.
    virtual void set_device() = 0;

    // ── GPU identity ────────────────────────────────────────────────────────

    /// Returns the GpuRef for this device's GPU.
    virtual const config::GpuRef& gpu() const = 0;

    // ── Compute kernels (attention projections) ─────────────────────────────

    /// FP8 blockwise-scaled GEMM (for q_a, q_b, kv_a; o_proj uses nvfp4_grouped_gemm).
    virtual void gemm(const Fp8GemmParams& params,
                      void* workspace, void* stream) = 0;

    /// GGUF integer mat-vec (mmvq) — decode path (M ≤ 8), strategy `int`.
    /// Quantizes the BF16 activation to Q8_1 in `q8_1_workspace` (must be at
    /// least `gguf_mmvq_workspace_bytes(M, K)`), then int8 dp4a against the
    /// packed weight. Forwards to compute::launch_gguf_mmvq in the CUDA backend.
    virtual void gguf_mmvq(const GgufGemmParams& params,
                           void* q8_1_workspace, void* stream) = 0;

    /// GGUF integer mat-mat (mmq) — prefill path (M > 8), strategy `int`.
    /// Same Q8_1 activation workspace as gguf_mmvq; forwards to the fast int8
    /// tensor-core variant compute::launch_gguf_mmq_mma in the CUDA backend.
    virtual void gguf_mmq(const GgufGemmParams& params,
                          void* q8_1_workspace, void* stream) = 0;

    /// GGUF dequant-to-BF16 GEMM — strategy `dequant` (lossless activation).
    /// No activation quantization, no workspace. Forwards to
    /// compute::launch_gguf_dequant_gemm in the CUDA backend.
    virtual void gguf_dequant_gemm(const GgufGemmParams& params,
                                   void* stream) = 0;

    /// RMS normalization (for input_layernorm, q_a_norm, kv_a_norm).
    /// row_stride: elements between consecutive token rows of BOTH input and
    /// out (== hidden_size for tight rows). Only the first hidden_size
    /// elements of each row are read/written — required when normalizing a
    /// sub-slice of interleaved rows (kv_a_norm over the c_kv half of the
    /// [B, kv_lora + rope] kv_a output; TD-PREFILL-CHUNK-ATTN).
    virtual void rmsnorm(void* out, const void* input,
                         const void* weight, float eps,
                         int num_tokens, int hidden_size, int row_stride,
                         void* stream) = 0;

    /// Dynamic BF16 → FP8 E4M3 blockwise quantization (activations).
    virtual void quantize_fp8(const DynamicFp8QuantParams& params,
                              void* stream) = 0;

    /// Offline BF16 → FP8 E4M3 tile-level quantization (weights).
    virtual void weight_quantize_fp8(const WeightFp8QuantParams& params,
                                     void* stream) = 0;

    /// NVFP4 → BF16 weight dequantization (for online NVFP4→BF16→FP8 conversion).
    virtual void nvfp4_dequant_bf16(const Nvfp4DequantBf16Params& params,
                                    void* stream) = 0;

    /// NVFP4 grouped GEMM (for o_proj when weight is NVFP4).
    virtual void nvfp4_grouped_gemm(
        const Nvfp4GroupedGemmParams& params,
        void* workspace, size_t workspace_bytes,
        void* stream) = 0;

    /// BF16 → NVFP4 grouped activation quantization (for o_proj NVFP4 path).
    virtual void bf16_to_nvfp4_grouped(const Bf16ToNvfp4GroupedParams& params,
                                       void* stream) = 0;

    /// FP8 → BF16 strided dequant: extract V rows from kv_b_proj.
    virtual void kv_bv_extract_dequant(const KvBvExtractDequantParams& params,
                                       void* stream) = 0;

    // ── BF16 batched GEMM (kv_b_v projection) ──────────────────────────────

    /// BF16 strided batched GEMM for kv_b_v projection.
    virtual void batched_gemm_bf16(const StridedBatchedGemmBf16Params& params,
                                   void* stream) = 0;

    // ── MLA query W_UK absorption ──────────────────────────────────────────

    /// Absorb W_UK into the query (q_b_proj output → absorbed [s_q, h_q, d_c+d_rope]).
    virtual void absorb_q(const QAbsorbParams& params, void* stream) = 0;

    /// In-place RoPE rotation of a strided rope slice (k_pe before k_append).
    virtual void rope_rotate(const RopeRotateParams& params, void* stream) = 0;

    // ── DSA lightning-indexer producer ops (GLM-25a; CUDA-free params) ────────
    // Default no-ops so non-DSA backends (Null) need not implement them; the
    // SM120 device overrides them (kernels in sm120/indexer/indexer_prep.cu +
    // lightning_indexer.cu). All pointers are device pointers.

    /// Full LayerNorm (mean-center, weight+bias) in place over the last `dim`,
    /// `num_rows` BF16 rows. Used for the indexer k_norm (LayerNorm, not RMS).
    virtual void indexer_layernorm_bias(void* x, const void* weight,
                                        const void* bias, int num_rows, int dim,
                                        float eps, void* stream) {}

    /// In-place orthonormal Walsh-Hadamard rotation (FWHT scaled by
    /// 1/sqrt(dim)) of `rows` BF16 rows of width `dim` (power of two).
    /// Applied to BOTH indexer q and k after RoPE (and the k LayerNorm),
    /// before the FP8 cache append — QuaRot-style, matches llama.cpp
    /// deepseek32.cpp:281-286 (TD-GLM-INDEXER-HADAMARD).
    virtual void indexer_hadamard(void* x, int rows, int dim, void* stream) {}

    /// Per-token FP8-E4M3 quant of single-head keys [num_tokens, head_dim] with
    /// one scale each, scattered into k_cache[slot_mapping[t] + slot_bias]
    /// (MQA layout). slot_bias = −1 lets the decode path pass seqlens_k directly.
    virtual void indexer_k_quant_append(const void* k_in, const void* slot_mapping,
                                        void* k_cache, void* k_scales,
                                        int num_tokens, int index_head_dim,
                                        int slot_bias, void* stream) {}

    /// out[f32] = in[bf16] * scale, over [num_rows, n] (score_proj prep).
    virtual void indexer_scale_weights(const void* in, void* out, int num_rows,
                                       int n, float scale, void* stream) {}

    /// For each of num_tokens queries: MQA score over num_blocks resident keys
    /// then causal top-k (query_position = token index). Writes
    /// sparse_indices_out[t*topk..] (padded −1) + topk_lengths_out[t].
    virtual void indexer_score_topk(const IndexerScoreTopkArgs& args, void* stream) {}

    /// TD-SPARSE-PREFILL-SCORE-BATCH: batched multi-row score + causal top-k
    /// — ONE score launch + ONE top-k launch over num_rows rows, each with
    /// its own device-array bound/cutoff. Bit-identical to num_rows
    /// sequential indexer_score_topk calls (see IndexerScoreTopkBatchedArgs).
    virtual void indexer_score_topk_batched(
        const IndexerScoreTopkBatchedArgs& args, void* stream) {}

    /// KVS-4 (sharded KV): translate a GLOBAL sorted top-k index list
    /// ([num_tokens, topk] device, −1-padded, lengths in global_lengths) into
    /// this rank's LOCAL slot indices — the subset of positions owned by
    /// `rank` under the INV-4.9e round-robin-by-chunk partition, rewritten as
    /// indices into the rank's KV staging (compacted, ascending, −1-padded;
    /// local_lengths[t] may be 0 when the rank owns none of the selection).
    /// All pointers are device pointers.
    virtual void indexer_shard_translate(
        const void* global_indices, const void* global_lengths,
        void* local_indices, void* local_lengths,
        int num_tokens, int topk, int chunk_tokens, int dcp_size, int rank,
        void* stream) {}

    /// TD-GLM-INDEXER-LOCAL-MERGE (dcp_indexer_mode=local): exact cross-rank
    /// top-k merge — scatter allgathered per-rank shard candidates back to
    /// global positions and re-select the global top-k (see
    /// IndexerTopkMergeArgs for the contract).
    virtual void indexer_topk_merge(const IndexerTopkMergeArgs& args,
                                    void* stream) {}

    // ── Device memory ───────────────────────────────────────────────────────

    /// Allocate device memory.  Returns nullptr on failure.
    virtual void* device_alloc(size_t bytes) = 0;

    /// Free device memory.  Must tolerate nullptr.
    virtual void device_free(void* ptr) = 0;

    /// Synchronize the device (all streams).  Init-time only (INV-5b exemption).
    virtual void device_sync() = 0;

    /// Synchronous host-to-device memcpy.  Used by init-time upload paths.
    virtual void memcpy_h2d(void* dst, const void* src, size_t bytes) = 0;

    /// Asynchronous device-to-device 2D copy: `height` rows of `width` bytes
    /// from src (row pitch spitch) to dst (row pitch dpitch).  Used for row
    /// compaction when a GEMM writes an N-padded output (e.g. kv_a 576→640).
    virtual void memcpy_2d_d2d_async(void* dst, size_t dpitch,
                                     const void* src, size_t spitch,
                                     size_t width, size_t height,
                                     void* stream) = 0;

    /// Asynchronous stream-ordered host-to-device memcpy.  The host source
    /// must outlive the copy (persistent staging, not a stack temporary).
    /// Default falls back to the synchronous path; CUDA devices override.
    virtual void memcpy_h2d_async(void* dst, const void* src, size_t bytes,
                                  void* /*stream*/) {
        memcpy_h2d(dst, src, bytes);
    }

    // ── KV cache append ─────────────────────────────────────────────────────

    /// Write compressed KV (LoRA + RoPE) to paged KV cache.
    /// c_kv_row_stride / k_rope_row_stride: elements between consecutive
    /// tokens' source rows (== d_c / d_rope for tight arrays). The engine
    /// passes both pointers into ONE interleaved [num_tokens, d_c + d_rope]
    /// kv_a output, so both strides are d_c + d_rope — assuming tight rows was
    /// exact only at num_tokens == 1 (TD-PREFILL-CHUNK-ATTN).
    /// layer_idx: layer index for per-layer resources (TQ uses for Π lookup;
    ///            SnapMLA ignores).
    virtual void k_append(
        const void* c_kv, const void* k_rope, void* kv_cache,
        int64_t cache_stride_block, int cache_stride_row,
        const int* slot_mapping, int num_tokens,
        int d_c, int d_rope,
        int c_kv_row_stride, int k_rope_row_stride,
        int page_size,
        int layer_idx, void* stream) = 0;

    // ── Prefill attention ───────────────────────────────────────────────────

    /// Steps 7-9 for non-graph path.  Dispatches dense or sparse internally.
    /// layer_idx: layer index for per-layer resources (TQ uses for Π lookup;
    ///            SnapMLA ignores).
    ///
    /// chunk_causal (TD-PREFILL-CHUNK-ATTN perf fix): batched causal prefill
    /// for a multi-token prefill CHUNK. Contract: the batch_size rows are
    /// consecutive positions of ONE sequence — seqlens_k[b] (device) is row
    /// b's visible KV prefix length, ascending in b, with seqlens_k[B-1] ==
    /// seq_len_kv == the union prefix length, and block table row B-1 covers
    /// the whole union [0, seq_len_kv). The device stages the union prefix
    /// ONCE (linearize+dequant of row B-1 only — O(seq_len_kv), not
    /// O(B·seq_len_kv)) and the kernel masks row b to [0, seqlens_k[b])
    /// per CTA, bit-equal to B per-row batch-of-1 calls.
    /// chunk_causal composes with BOTH kernels:
    ///   - dense: row b attends the full staged prefix [0, seqlens_k[b]).
    ///   - sparse (TD-SPARSE-CHUNK-PREFILL, INV-SPARSE-CHUNK-CAUSAL): row b
    ///     attends only its own top-k selection (sparse_indices row b,
    ///     topk_lengths[b]) INTERSECTED with [0, seqlens_k[b]) — the per-row
    ///     causal bound is enforced in-kernel, so a selected index at or past
    ///     the row's own position is masked, never attended.
    /// false → legacy flat behavior: every row attends all seq_len_kv staged
    /// rows (dense) / any selected index < seq_len_kv (sparse) — valid for
    /// batch_size == 1.
    virtual void prefill_attention(
        const void* q_compressed, int batch_size, int seq_len_kv,
        const int* seqlens_k, const int* block_tables,
        int max_blocks_per_seq,
        void* kv_cache, int64_t cache_stride_block, int cache_stride_row,
        int page_size, bool is_sparse, bool chunk_causal,
        const int* sparse_indices, const int* topk_lengths, int topk,
        void* out, float* lse,
        int layer_idx, void* stream) = 0;

    // ── Decode graph ops ────────────────────────────────────────────────────

    /// Update decode graph runner's fixed buffers with per-step data.
    /// layer_idx: layer index for per-layer resources (TQ uses for Π lookup;
    ///            SnapMLA ignores).
    virtual void decode_graph_update(
        GraphEntry& entry, const void* q_bf16,
        const int* seqlens_k, const int* block_table,
        const int* indices,
        int layer_idx, void* stream) = 0;

    /// Replay the captured decode CUDA graph.
    virtual void decode_graph_replay(
        GraphEntry& entry, void* stream) = 0;

    /// Get the decode graph runner's output buffer pointer.
    virtual void* decode_graph_out_ptr(
        GraphEntry& entry) = 0;

    /// Get the decode graph runner's LSE buffer pointer.
    virtual float* decode_graph_lse_ptr(
        GraphEntry& entry) = 0;

    // ── DCP allreduce graph ─────────────────────────────────────────────────

    /// Replay the DCP allreduce CUDA graph (steps 10-12).
    virtual void dcp_graph_replay(
        GraphEntry& entry, void* stream) = 0;

    // Non-copyable (polymorphic base).
    AttentionDevice(const AttentionDevice&) = delete;
    AttentionDevice& operator=(const AttentionDevice&) = delete;

protected:
    AttentionDevice() = default;
};

// ── Factory ─────────────────────────────────────────────────────────────────

/// Create an AttentionDevice for the given GPU and algorithm.
/// Algorithm selection baked into concrete type (INV-BH-7).
/// Requires forward decl; see config_parser.h for AttentionBackendType.
std::unique_ptr<AttentionDevice> make_attention_device(
    config::AttentionBackendType type, config::GpuRef gpu);

/// Create a NullAttentionDevice for unit tests.
std::unique_ptr<AttentionDevice> make_null_attention_device(
    config::GpuRef gpu);

}  // namespace layerstorm::compute
