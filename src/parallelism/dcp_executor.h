#pragma once

//==============================================================================
// DCP Executor
//
// Orchestrates per-layer TP+DCP attention execution for the TP GPU pair.
// Sequences projection GEMMs, attention kernels, DCP correction, o_proj
// GEMM, and TP allreduce — all as non-blocking stream-ordered operations.
//
// With tensor_parallelism=2: TP=2 (head-parallel sharding for q_b_proj,
// kv_b_proj, o_proj — 64 of 128 heads per GPU) + DCP=2 (KV cache
// sequence-sharded by token position).
//
// Two execution paths:
//   Graph mode (decode):   DecodeGraphRunner + DcpAllreduceGraphRunner + o_proj
//   Non-graph mode (prefill): Individual kernels + DcpAttentionWrapper
//
// With dcp_size=1: single GPU, no communication, no DCP correction.
//
// INV-DCP-6:  tensor_parallelism controls both TP and DCP
// INV-DCP-8:  o_proj TP allreduce separate from DCP correction (HOP-B)
// INV-DCP-14: Graph capture requires concurrent threads for both ranks
// INV-3.4.0:  All GPU calls non-blocking, returns immediately
// INV-3.4.1:  No blocking calls in main loop
// INV-3.4.2:  NOT thread-safe (single-threaded orchestrator)
//==============================================================================

#include "core/gpu_ref.h"
#include "config/config_parser.h"             // config::GgufStrategy (CUDA-free)
#include "model/quantization/gguf_kquant.h"   // model::GgufKQuantType (CUDA-free)

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace layerstorm::compute {
class DeviceBackend;
class StreamManager;
class GraphRegistry;
class AttentionDevice;
class DcpAttentionWrapper;
class NcclGroupGraphRunner;  // INV-NCCL-GRAPH: captured o_proj TP reduce
struct GraphEntry;
}  // namespace layerstorm::compute

namespace layerstorm::daemon {
class BufferRegistry;
}  // namespace layerstorm::daemon

namespace layerstorm::parallelism {

class DcpCommunicator;
class KvBvDequantPool;
class KvTieringHook;  // GLM-25k (kv_tiering_hook.h)
class V4KvTieringHook;  // TD-V4-KVT (v4_kv_tiering_hook.h)

// ── Per-layer weight descriptor ──────────────────────────────────────────────

/// Device pointers for one attention layer's weights on a single GPU.
/// Filled by orchestrator from loaded weights.  All pointers are device memory.
///
/// For replicated weights (q_a, kv_a, norms, DSA indexer): both ranks hold
/// identical copies.
/// For TP-sharded weights (q_b, o_proj): each rank holds its head shard.
struct AttentionLayerWeights {
    // Projection weights (FP8 E4M3)
    const void* q_a_proj = nullptr;       ///< [q_lora_rank, hidden_size] replicated
    const void* q_b_proj = nullptr;       ///< [H_local*(qk_nope+qk_rope), q_lora_rank] TP-sharded
    const void* kv_a_proj = nullptr;      ///< [kv_lora_rank+qk_rope, hidden_size] replicated
    const void* kv_b_proj = nullptr;      ///< [H_local*(qk_nope+v_head_dim), kv_lora_rank] TP-sharded (column-parallel)
    const void* o_proj = nullptr;         ///< [hidden_size, H_local*v_head_dim] TP-sharded (row-parallel, non-absorbed INV-MLA-1)

    // Per-block float32 weight scales (for FP8 GEMMs)
    const void* q_a_proj_scales = nullptr;
    const void* q_b_proj_scales = nullptr;
    const void* kv_a_proj_scales = nullptr;
    const void* kv_b_proj_scales = nullptr; ///< FP8 blockwise scales (nullptr if BF16)
    const void* o_proj_scales = nullptr;

    // Norm weights (BF16)
    const void* q_a_norm = nullptr;       ///< [q_lora_rank] replicated
    const void* kv_a_norm = nullptr;      ///< [kv_lora_rank] replicated

    // DSA indexer weights (replicated, nullptr if not DSA)
    const void* q_idx_b = nullptr;
    const void* k_idx = nullptr;
    const void* k_idx_norm = nullptr;       ///< indexer k_norm weight [index_head_dim] BF16
    const void* k_idx_norm_bias = nullptr;  ///< indexer k_norm bias   [index_head_dim] BF16 (TD-74j)
    const void* weights_proj = nullptr;     ///< [hidden_size, index_n_heads] — per-token score weights

    // kv_b_proj metadata
    bool kv_b_proj_is_fp8 = false;      ///< true when kv_b_proj is FP8 (needs dequant for v_proj)

    // kv_a_proj padded row count: set (to a 128 multiple, e.g. 576→640) when
    // the weight rows were zero-padded at init so the kv_a GEMM runs at the
    // padded N on the unpadded fast path. 0 = weight is tight (FP8
    // checkpoints) and the GEMM must use the real N.
    int kv_a_n_padded = 0;

    // GGUF projection metadata (GG-4). For a GGUF-quantized checkpoint each
    // attention projection's weight is packed in one of the six k-quant types;
    // the executor routes that projection through the GGUF GEMM virtuals
    // (mmvq/mmq/dequant) using the per-projection type below. *_is_gguf is true
    // only when that projection was loaded as a GGUF k-quant. The four plain
    // projections (q_a/q_b/kv_a/o_proj) route through the GGUF GEMM virtuals;
    // kv_b_proj is consumed in-kernel by q_absorb (GG-7) via its GGUF dequant
    // branch, not a GEMM. When false the projection takes the existing
    // FP8/NVFP4 path. The type is meaningful only when the matching *_is_gguf
    // flag is set.
    bool q_a_is_gguf = false;
    bool q_b_is_gguf = false;
    bool kv_a_is_gguf = false;
    bool o_proj_is_gguf = false;
    model::GgufKQuantType q_a_gguf_type{};
    model::GgufKQuantType q_b_gguf_type{};
    model::GgufKQuantType kv_a_gguf_type{};
    model::GgufKQuantType o_proj_gguf_type{};

    // GG-7: kv_b_proj GGUF metadata. When kv_b_is_gguf the packed kv_b weight is
    // a GGUF k-quant [h_q*(P+V), (L/QK)*bytes] and the q_absorb kernel dequants
    // W_UK per element in-kernel (dequant-only, bit-equal to a load-time dequant
    // to BF16 — no activation quant). Requires kv_lora_rank % QK == 0. The type
    // is meaningful only when kv_b_is_gguf is set. GG-7b: the W_UV value path
    // (kv_b_v batched GEMM via the KvBvDequantPool, feeding o_proj) now also
    // handles GGUF kv_b — the pool's V-extract kernel dequants the packed V rows
    // into the BF16 slot (same dequant-only philosophy), so the batched value
    // GEMM + o_proj run unchanged.
    bool kv_b_is_gguf = false;
    model::GgufKQuantType kv_b_gguf_type{};

    // GLM-25a: DSA indexer projection GGUF metadata. wq_b (q_lora→n_head*head_dim)
    // and wk (hidden→head_dim) route through the GGUF GEMM virtuals like the main
    // projections; indexer_proj + k_norm are small and loaded BF16 (no GGUF type).
    bool q_idx_b_is_gguf = false;
    bool k_idx_is_gguf = false;
    model::GgufKQuantType q_idx_b_gguf_type{};
    model::GgufKQuantType k_idx_gguf_type{};

    // NVFP4 o_proj metadata (runtime NVFP4 GEMM path)
    bool o_proj_is_nvfp4 = false;       ///< true when o_proj is in NVFP4 format
    float o_proj_nvfp4_scale_2 = 0.f;   ///< weight_scale_2 scalar
    float o_proj_nvfp4_input_scale = 0.f; ///< input_scale scalar
    float o_proj_nvfp4_alpha = 1.f;     ///< weight_scale_2 * input_scale

    // Layer norms (BF16, replicated)
    const void* input_layernorm = nullptr;          ///< [hidden_size]
    const void* post_attention_layernorm = nullptr; ///< [hidden_size]

    // ── DeepSeek-V4 per-layer weights (nullptr for every non-V4 arch) ──────
    // V4-5b mHC mixing weights (F32, replicated; fn [(2+hc)*hc, hc*hidden],
    // base [(2+hc)*hc], scale [3]).
    const void* hc_attn_fn = nullptr;
    const void* hc_attn_base = nullptr;
    const void* hc_attn_scale = nullptr;
    const void* hc_ffn_fn = nullptr;
    const void* hc_ffn_base = nullptr;
    const void* hc_ffn_scale = nullptr;

    // V4-5a attention-sink logits (F32 [num_attention_heads], replicated).
    const void* attn_sinks = nullptr;

    // V4-5c grouped o_proj factorization (BF16; ticket G consumes).
    const void* o_proj_a = nullptr;   ///< [o_groups*o_lora_rank, group_dim]
    const void* o_proj_b = nullptr;   ///< [hidden, o_groups*o_lora_rank]

    // V4-5a compressor projections (BF16) + APE/norm (F32, replicated).
    // CSA: wkv/wgate [2*head_dim, hidden], APE [ratio, 2*head_dim];
    // HCA: wkv/wgate [head_dim, hidden],   APE [ratio, head_dim].
    const void* compressor_wkv = nullptr;
    const void* compressor_wgate = nullptr;
    const void* compressor_ape = nullptr;
    const void* compressor_norm = nullptr;   ///< [head_dim] F32

    // V4-5a lightning-indexer compressor (CSA layers only; dims at
    // index_head_dim: wkv/wgate [2*128, hidden] BF16, APE [4, 256] F32,
    // norm [128] F32).
    const void* indexer_compressor_wkv = nullptr;
    const void* indexer_compressor_wgate = nullptr;
    const void* indexer_compressor_ape = nullptr;
    const void* indexer_compressor_norm = nullptr;
};

// ── Per-layer execution parameters ───────────────────────────────────────────

/// Runtime parameters for a single attention layer execution.
/// Passed by orchestrator each layer.  Per-rank arrays are indexed by DCP rank.
struct AttentionExecParams {
    int layer_idx = 0;
    int batch_size = 0;             ///< Number of tokens in this step

    // TD-PREFILL-SUPERCHUNK: global ROW offset of this sub-chunk within the
    // superchunk. The dispatcher already offsets hidden_states by
    // row_offset·H, so projections/attention stay row-offset-agnostic; the
    // executor uses this ONLY to address the PERSISTENT per-superchunk sparse
    // top-k buffers (sparse_indices_dev_/topk_lengths_dev_ rows
    // [batch_row_offset, batch_row_offset+batch_size)) and to key the
    // IndexShare reuse per sub-chunk (a shared layer must reuse the full
    // layer's selection of ITS OWN sub-chunk, not the last-produced one).
    // 0 = legacy behavior.
    int batch_row_offset = 0;

    // Input: per-rank [batch_size, hidden_size] BF16
    void* const* hidden_states = nullptr;   ///< [dcp_size]

    // KV cache metadata per rank
    const int* const* seqlens_k = nullptr;      ///< [dcp_size] → [batch_size]
    int max_seqlen_k = 0;           ///< host-side max(seqlens_k[0..batch)) — bounds the
                                    ///< nongraph KV gather/dequant/attend length; 0 → unknown
    const int* const* block_tables = nullptr;    ///< [dcp_size] → [batch_size, max_blocks_per_seq]
    int max_blocks_per_seq = 0;                 ///< inner stride of block_tables (TD-51cf)
    const int* const* slot_mappings = nullptr;   ///< [dcp_size] → [batch_size]

    // KV cache base pointers per rank (for fused_k_append)
    void* const* kv_cache_ptrs = nullptr;        ///< [dcp_size]
    int64_t cache_stride_block = 0;
    int cache_stride_row = 0;
    int page_size = 64;

    // KVS-2 (sharded KV, Options::dcp_kv_sharded): non-null ONLY under
    // sequence-sharded KV. There, seqlens_k[r] holds rank r's LOCAL shard
    // length (attention/k-gather bound over its own token subset), while
    // ROPE / position math must use the GLOBAL length: position =
    // global_seqlens_k[·][b] − 1. Per-rank device arrays with IDENTICAL
    // values — each rank dereferences its own GPU's copy. nullptr under
    // replication (seqlens_k IS global there and doubles as both).
    const int* const* global_seqlens_k = nullptr;   ///< [dcp_size] → [batch_size]
    // KVS-2 (sharded KV): per-rank HOST mirror of the LOCAL shard lengths
    // (kv-meta staging) for host-side bounds/causal math (chunk prefill's
    // per-row local causal length, KVS-3). nullptr under replication.
    const int* const* host_local_seqlens_k = nullptr;  ///< [dcp_size] → [batch_size]

    // DSA sparse indices (nullptr if dense or non-DSA). Per-rank device
    // buffers — every rank dereferences ITS OWN GPU's copy
    // (TD-GLM-INDEXER-DCP: a single cross-rank pointer is illegal).
    const int* const* sparse_indices = nullptr;  ///< [dcp_size] → [batch_size, topk]
    const int* const* topk_lengths = nullptr;    ///< [dcp_size] → [batch_size]

    // TD-GLM-INDEXER-PAGED/-BATCH: paged indexer-K provisioned by the
    // dispatcher from Pool::kIndexerK. HOST table of device page base pointers
    // laid out [batch * indexer_k_batch_stride + layer * indexer_k_page_stride
    // + logical_page]; rows exist only for layers that COMPUTE the indexer
    // (IndexShare full ∪ layer 0), nullptr otherwise. Page layout:
    // [page_tokens × head_dim FP8 | page_tokens F32 scales]. When absent the
    // producer falls back to its executor arena (B==1 only — the arena is
    // structurally single-sequence).
    // Per-RANK host tables (TD-GLM-INDEXER-DCP replicated mode: each rank
    // holds its own GPU's replica pages): indexer_k_pages[r] is rank r's
    // table with the batch/layer/page layout described above.
    const void* const* const* indexer_k_pages = nullptr;  ///< [dcp_size] → table
    int indexer_k_page_stride = 0;    ///< logical pages per layer row
    int indexer_k_batch_stride = 0;   ///< table row stride per batch entry
    int indexer_k_page_tokens = 0;    ///< positions per page

    // TD-GLM-INDEXER-BATCH: per-entry HOST seqlens (kv-meta staging; entry
    // b's current length) and a nonzero fingerprint identifying this step's
    // batch composition — the IndexShare reuse-validity key (seq_id-aware,
    // unlike a bare seqlen match).
    const int* host_seqlens_k = nullptr;   ///< HOST [batch_size]
    uint64_t indexer_step_key = 0;

    // TD-GLM-INDEXER-COV: set by the dispatcher when EVERY sequence in this
    // step has an indexer-K coverage gap — the producer must not run sparse
    // from ANY storage (paged or arena), because skipped positions were never
    // appended and would be scored as garbage.
    bool indexer_sparse_suppress = false;

    // TD-GLM-INDEXER-B1CASCADE (resolved) / INV-DSA-ROWMIX: per-row dense
    // mask for a MIXED B>1 decode cohort. HOST array [batch_size]; non-null
    // ONLY when the dispatcher found a decode cohort mixing sparse-eligible
    // and coverage-dead sequences: row b == 1 → that row is coverage-dead
    // and MUST run DENSE (its indexer-K storage is stale/absent — the
    // producer skips its append and scoring, and writes topk_length[b] = 0
    // so batch-wide consumers see a bounded empty selection); row b == 0 →
    // sparse-eligible as usual. The nongraph consumer then splits the step
    // into per-row batch-of-1 sub-dispatches (sparse for live rows, dense
    // for dead rows) writing into each row's slice of the batch output —
    // bit-identical to running the sequences separately (each sub-call IS
    // the B==1 call shape; a flat batched call over MULTIPLE sequences is
    // invalid for dense AND sparse, TD-DECODE-NONGRAPH-BATCH staging).
    // nullptr = uniform batch (all-sparse or all-dense), legacy behavior.
    // Never set at B==1, on prefill/chunk steps, or under use_graph.
    const uint8_t* indexer_row_dense = nullptr;

    // TD-GLM-INDEXER-PREFILL: set by the dispatcher on a prefill / chunked-
    // prefill step whose coverage guard blessed an indexer-K CHUNK APPEND
    // (single sequence, consecutive positions contiguous with prior
    // coverage). The executor then runs the producer's K half batched over
    // all chunk rows — append first, always. Requires indexer_step_key
    // != 0 (and indexer_k_pages for the paged mode; absent → arena).
    // With Options::sparse_prefill OFF (default) the chunk's own attention
    // stays dense prefill (append only). With it ON
    // (TD-SPARSE-CHUNK-PREFILL) the executor additionally runs the
    // producer's Q half + per-chunk-row causal top-k and the chunk attends
    // SPARSE chunk-causal (each row: its own top-k ∩ [0, its position]).
    bool indexer_prefill_append = false;

    // Per-rank weights for this layer
    const AttentionLayerWeights* const* weights = nullptr;  ///< [dcp_size]

    // Execution mode flags
    bool use_graph = false;         ///< true = decode with captured graphs
    bool is_sparse = false;         ///< true = DSA sparse attention

    // Chunked prefill (0 = full sequence)
    int chunk_start = 0;
    int chunk_len = 0;

    // SC (superchunk port): this sub-launch is one sub-chunk of a superchunk
    // layer sweep (row_offset semantics handled dispatcher-side). The V4
    // executor uses it to (a) accept layer-sweep REPLAYS of already-covered
    // step windows (L1 revisits w0 after L0 advanced the frontier) and (b)
    // never snapshot superchunk windows (rewind into a superchunk is
    // fail-closed). Inert for non-V4 paths.
    bool superchunk = false;

    // ── V4-7b (ticket H): DeepSeek-V4 per-step fields ─────────────────────
    // Filled by the dispatcher ONLY for deepseek_v4 dispatches (nullptr
    // otherwise). The V4 pipeline is B==1 decode-shaped (prompt-fed prefill
    // = one step per prompt token); chunk/batched/draft/graph shapes fail
    // loud (TD-V4-CHUNK-PREFILL / TD-V4-PREFILL-PERF).
    // V4-2c TP: tier metadata is PER RANK (each rank keeps replicated KV
    // tiers on its own GPU; side-pool page ids are rank-independent).
    struct V4StepRank {
        // CSA compressed tier (kMain): HOST block-table row for this layer —
        // physical page ids per 256-token logical block. Entry slot for CSA
        // block e = bt[e/64]*64 + e%64.
        const int* host_csa_bt = nullptr;
        int host_csa_bt_len = 0;       ///< valid entries in the row
        // SWA ring page for this layer (kSwa pool-relative page id; ONE ring
        // page per (seq, layer): slot = page*swa_page_tokens + pos%window).
        int swa_page_idx = -1;
        // HCA tier pages for this layer (pool-relative ids, logical order).
        const int* hca_page_ids = nullptr;
        int hca_page_count = 0;
        // Lightning-indexer tier pages for this layer (CSA layers only):
        // pool-relative ids (compress-insert slot math) + device-addressable
        // page base pointers (lightning score page table), logical order.
        const int* lid_page_ids = nullptr;
        const void* const* lid_page_ptrs = nullptr;
        int lid_page_count = 0;
    };
    struct V4Step {
        uint64_t seq_id = 0;           ///< ring-state key (0 = invalid)
        int token_pos = -1;            ///< this step's token position
        const V4StepRank* ranks = nullptr;  ///< [num_ranks] per-rank tiers
        int num_ranks = 0;             ///< must equal dcp_size
    };
    const V4Step* v4 = nullptr;

    // GLM-25k: DSA-guided KV tiering hook (kv_tiering_hook.h). Set by the
    // dispatcher on tierable steps only: B==1 non-draft non-graph decode
    // steps, and (TD-KVT-PREFILL) blessed B==1 SPARSE prefill chunks
    // (Options::sparse_prefill on, replicated KV) — a single-row chunk's
    // sparse consumption is decode-shaped, so the same materialize seam
    // covers it.  When set and a layer runs SPARSE, the executor
    // materializes the selected rows through the hook and runs the
    // unmodified sparse attention over the returned dense fake-paged view
    // with identity indices (INV-KVT-1 placement-only).  When set and a
    // DSA-capable layer falls back to DENSE (decode fallback or a failed
    // per-layer sparse-prefill production), on_dense_layer() is notified
    // (throws if the layer has cold pages, INV-KVT-2).  nullptr =
    // non-tiered path, byte-identical to pre-GLM-25k behavior.
    KvTieringHook* kv_tiering = nullptr;

    /// TD-V4-KVT (P3): V4 CSA-bucket tiering hook. Set by the dispatcher
    /// on tiered V4 steps; the executor calls ensure_hot() with the
    /// lightning selection (or IOTA visibility) BEFORE the block-table
    /// staging so cold pages repromote in place. nullptr = untiered,
    /// byte-identical.
    V4KvTieringHook* v4_tiering = nullptr;
};

// ── DcpExecutor ──────────────────────────────────────────────────────────────

class DcpExecutor {
public:
    struct Options {
        int dcp_size = 1;                          ///< parallelism.tensor_parallelism
        std::vector<config::GpuRef> gpus;       ///< TP GPU configs (INV-4.18)

        /// KV placement mode (INV-DCP-KVREP / INV-KVS-QAG).
        /// false (default): KV is replicated on every TP rank — each rank
        /// computes COMPLETE attention for its own head shard, so the DCP
        /// LSE combine (steps 10-12) MUST NOT run: it implements KV-SHARDED
        /// union semantics and under replication it would cross-mix
        /// DIFFERENT heads (rank0 head h with rank1 head HL+h) — mild at
        /// short context, catastrophic at long context.
        /// true (sequence-sharded KV, KVS-2/-3): each rank holds only its
        /// token shard and the executor runs the Q-head-allgather structure
        /// (INV-KVS-QAG, vLLM DCP): allgather q_absorbed in the HEAD dim →
        /// every rank attends ALL dcp*HL heads over its LOCAL shard →
        /// LSE combine over the gathered head set (genuine same-head
        /// partials over disjoint token shards) → each rank's kv_bv/o_proj
        /// consume only its own HL-head slice of the combined output.
        /// Sharded is NONGRAPH-ONLY (TD-KVS-QAG-GRAPH): use_graph is forced
        /// off and capture_dcp_graphs() is a no-op under this mode.
        /// Requires `communicator` at dcp_size >= 2 (Q allgather).
        bool dcp_kv_sharded = false;

        /// INV-4.9e round-robin chunk size (memory.kv_cache.dcp_chunk_size).
        /// Required under dcp_kv_sharded for the KVS-4 GLOBAL→LOCAL sparse-
        /// index translation (owner(g) = (g/chunk) % dcp); 0/unset under
        /// sharding fails DSA closed to dense (never silently wrong).
        int dcp_chunk_tokens = 0;

        /// TD-GLM-INDEXER-LOCAL-MERGE (hardware.dcp_indexer_mode=local at
        /// dcp>=2): indexer-K is POSITION-SHARDED round-robin by INDEXER PAGE
        /// (owner(pos) = (pos / indexer_k_page_tokens) % dcp — independent of
        /// the KV chunk partition above; the merge output is GLOBAL indices,
        /// so KV-mode translation is unaffected). Each rank scores only its
        /// own shard, then the per-rank candidate lists are allgathered and
        /// exactly re-merged into the global top-k (identical to replicated
        /// mode's — see topk_merge.h). Requires `communicator` and
        /// indexer_k_page_tokens > 0; paged storage only (no arena fallback).
        bool indexer_local = false;
        /// memory.kv_cache.indexer_k_page_size_tokens — the local-mode
        /// ownership unit; must match the dispatcher's provisioning PT.
        int indexer_k_page_tokens = 0;

        /// TD-SPARSE-CHUNK-PREFILL (compute.dsa_sparse_prefill): DSA sparse
        /// CHUNK PREFILL attention. On a dispatcher-blessed prefill chunk
        /// (indexer_prefill_append), after the indexer-K chunk append the
        /// executor also runs the producer's Q half + per-chunk-row causal
        /// top-k, and the chunk's attention runs SPARSE chunk-causal: each
        /// chunk row attends its own ≤index_topk selection ∩ its causal
        /// prefix (INV-SPARSE-CHUNK-CAUSAL) instead of the DENSE full
        /// prefix. Native to DSA (the model is trained with this sparse
        /// pattern) — validated against the DSA reference, and the enabler
        /// for prefill KV tiering (TD-KVT-PREFILL, resolved: B==1 sparse
        /// prefill chunks are tierable steps, INV-KVT-13).
        /// Default OFF: chunks stay dense (append-only), byte-identical to
        /// pre-feature behavior. BOTH indexer modes: replicated emits the
        /// per-row global top-k directly; local (TD-SPARSE-PREFILL-LOCAL-
        /// INDEXER) emits per-row shard candidates + the same cross-rank
        /// merge as decode (per-row owned_len scoring bound → exact global
        /// equivalence, INV-SPARSE-LOCAL-PREFILL). Replicated KV only:
        /// dcp_kv_sharded (TD-SPARSE-PREFILL-KVS) falls back to dense
        /// chunks.
        bool sparse_prefill = false;

        /// TD-KVT-ADMISSION-UPFRONT (memory.kv_tiering.tiered_prefill):
        /// with sparse_prefill, a blessed sparse prefill chunk is ALSO
        /// allowed under SHARDED KV — the producer's per-row global top-k
        /// is KVS-4-translated per row (indexer_shard_translate is batched)
        /// and consumption decomposes into per-row B==1 sparse
        /// sub-dispatches when the dispatcher staged a tier step
        /// (AttentionExecParams::kv_tiering on a chunk), so cold rows
        /// materialize per row (INV-KVT-13 extended to chunk cohorts).
        bool tiered_prefill = false;

        int max_batch_size = 64;                   ///< Max decode batch for buffer sizing
        /// TD-PREFILL-SUPERCHUNK: total superchunk token capacity (0 = off).
        /// Sizes the PERSISTENT sparse top-k buffers (sparse_indices_dev_/
        /// topk_lengths_dev_) to max(max_batch_size, superchunk_tokens) rows so
        /// every sub-chunk's per-row selection survives until the IndexShare
        /// shared layers of the same superchunk consume it.
        int superchunk_tokens = 0;
        int num_layers = 0;                        ///< Total layers incl. MTP (per-layer meta sizing); 0 → 1

        // Model dimensions
        int hidden_size = 7168;
        int num_attention_heads = 128;             ///< Total heads (before TP split)
        int q_lora_rank = 1536;
        int kv_lora_rank = 512;
        int qk_rope_head_dim = 64;
        int qk_nope_head_dim = 128;
        int v_head_dim = 128;
        float rms_norm_eps = 1e-6f;

        // RoPE (TD-ROPE): pure cos/sin table built by the engine
        // (compute/rope_table.h), uploaded per rank in allocate_buffers.
        // Layout [rope_max_pos][qk_rope_head_dim] float32 (cos|sin halves,
        // interleaved-pair convention). Rotation disabled when null.
        const float* rope_cos_sin_host = nullptr;
        int rope_max_pos = 0;
        // V4-4c dual RoPE: SECOND pure cos/sin table (compress_rope_theta +
        // yarn) for compressed layers (compress_ratios[l] != 0). Same layout
        // and max_pos as rope_cos_sin_host. Null for non-V4 models. Per-layer
        // selection = ModelConfig::layer_uses_compress_rope(l). NOTE: V4
        // tables carry NO mscale anywhere — llama.cpp's dsv4_rope_attn_factor
        // exactly cancels ggml's yarn mscale, so cos/sin stay pure and the
        // V4 softmax scale is 1/sqrt(head_dim) without yarn mscale².
        const float* rope_cos_sin_compress_host = nullptr;

        // V4-5c grouped o_proj (ticket G). All three default 0 = non-V4:
        // no scratch is allocated and execute_v4_grouped_oproj fails loud.
        // Engine sets them from model config when has_grouped_o_proj().
        // group_dim = (num_attention_heads / v4_o_groups) * v4_head_dim
        // (V4-Flash: 8 heads * 512 = 4096).
        int v4_head_dim = 0;     ///< V4 attention head dim (512)
        int v4_o_groups = 0;     ///< o_proj stage-1 group count (8)
        int v4_o_lora_rank = 0;  ///< per-group stage-1 output rank (1024)

        // ── V4-7b (ticket H): DeepSeek-V4 attention pipeline config ────────
        // enabled ⇒ execute_attention_v4 is callable; scratch + per-seq state
        // rings are allocated. All geometry mirrors VramLayout.v4 / the
        // configured CsaHcaSm120AttentionDevice.
        struct V4Exec {
            bool enabled = false;
            int num_layers = 0;                ///< hidden layers (43)
            std::vector<uint8_t> attn_type;    ///< per layer: 0 swa/1 csa/2 hca
            // V4-5T (TD-V4-TQ-DEVICE): per-tier compressed-KV codec —
            // true = 644-B TQ entries (deps csa_tq family; device runs
            // the two-pass TQ+SWA merge). Mirrors VramLayout.v4 formats.
            bool csa_tq = false;
            bool hca_tq = false;
            int sliding_window = 128;
            int csa_ratio = 4;
            int hca_ratio = 128;
            int csa_entries_per_page = 64;     ///< kMain page = 256-tok block
            int hca_entries_per_page = 2;
            int swa_page_tokens = 128;
            int idx_entries_per_page = 0;      ///< LID entries per pool page
            int64_t idx_page_bytes = 0;
            int topk = 512;                    ///< lightning top-k (64-mult)
            int max_seq = 0;                   ///< bounds staging tables
            // Per-rank tier region bases (kMain/CSA base arrives per call via
            // AttentionExecParams::kv_cache_ptrs).
            std::vector<void*> hca_base;
            std::vector<void*> swa_base;
            std::vector<void*> idx_base;
            // Ticket J (V4 speculation): arm the per-step ring/SWA-slot
            // snapshots that make speculative REWIND re-feeds lossless.
            // The pos%capacity ring slots alias across a window boundary —
            // a speculative write at pos p+k clobbers the committed slot of
            // p+k−capacity, which a post-rewind boundary compress / SWA
            // window still needs. With snapshots armed, every step's
            // about-to-be-overwritten slots are saved per layer BEFORE the
            // writes and restored on the next rewind. OFF (default): plain
            // decode is byte-identical and rewinds fail loud.
            bool spec_snapshots = false;
            int max_verify_rows = 16;  ///< multi-row verify-chunk row bound
        };
        V4Exec v4;

        // DSA
        bool has_dsa = false;
        int index_topk = 2048;
        int index_n_heads = 64;    ///< DSA indexer query heads (GLM-5.2: 32)
        int index_head_dim = 128;  ///< DSA indexer head dim
        /// IndexShare (GLM-25b) per-layer full/shared mask (size num_layers).
        /// full[l]=true → layer recomputes the indexer; false → reuse the
        /// preceding full layer's top-k. Empty → every layer is full (GGUF
        /// default / llama.cpp reference).
        std::vector<uint8_t> indexer_full_layers;

        // GGUF attention GEMM dispatch (GG-4). gguf_active is true when the
        // checkpoint's weights_format is gguf / weights is a gguf* type; it
        // gates allocation of the Q8_1 activation workspace used by the int
        // strategy. gguf_strategy selects mmvq/mmq (int) vs dequant_gemm.
        bool gguf_active = false;
        config::GgufStrategy gguf_strategy = config::GgufStrategy::int_strategy;

        // Graph capture
        std::vector<int> graph_batch_sizes;        ///< Batch sizes to pre-capture (empty = none)

        // External dependencies (non-owning)
        DcpCommunicator* communicator = nullptr;
        compute::StreamManager* stream_manager = nullptr;
        compute::GraphRegistry* graph_registry = nullptr;
        compute::DcpAttentionWrapper* dcp_wrapper = nullptr;

        // Per-rank attention devices (one per TP GPU, INV-BH-1). Non-owning.
        // Algorithm selection (SnapMLA/TQ) baked into concrete type (INV-BH-7).
        std::vector<compute::AttentionDevice*> attention_devices;
        std::vector<compute::DeviceBackend*>  device_backends;  // #86b: for graph alloc/free
    };

    explicit DcpExecutor(Options opts);
    ~DcpExecutor();

    // Non-copyable, non-movable (owns device buffers)
    DcpExecutor(const DcpExecutor&) = delete;
    DcpExecutor& operator=(const DcpExecutor&) = delete;
    DcpExecutor(DcpExecutor&&) = delete;
    DcpExecutor& operator=(DcpExecutor&&) = delete;

    // ── Graph capture at init ──────────────────────────────────────────

    /// Capture the DCP combine graph for each (gpu_idx, batch_size).
    /// Currently a NO-OP in every mode (TD-KVS-QAG-GRAPH): the combine only
    /// exists under sharded KV, and sharded KV runs nongraph-only — the QAG
    /// sequence (Q-head allgather → all-head attention → combine → slice)
    /// is not graph-captured yet. Decode graphs are engine-dormant anyway
    /// (TD-DECODE-GRAPH). No-op when dcp_size < 2 or replicated KV.
    void capture_dcp_graphs();

    // ── Per-layer attention execution ──────────────────────────────────

    /// Execute the full attention flow for one layer (DCP_GUIDE §5 steps 1-14).
    /// All work enqueued on attention streams (Stream 0). Returns immediately.
    void execute_attention(const AttentionExecParams& params);

    /// V4-5c (ticket G, resolves TD-V4-OPROJ): DeepSeek-V4 grouped o_proj —
    /// the V4 equivalent of the MLA path's execute_oproj_and_reduce tail.
    /// Pure 2-stage grouped low-rank factorization (deepseek4.cpp:1066-1074;
    /// NO base+LoRA sum):
    ///   stage 1: per-group batched GEMM — attn_out [rows, h_q, head_dim]
    ///            viewed as [rows, o_groups, group_dim] (group g = heads
    ///            [g*h_q/o_groups, (g+1)*h_q/o_groups), contiguous);
    ///            oa[rows, g, :] = o_proj_a slab g [o_lora_rank, group_dim]
    ///            @ attn_out group g → oa scratch [rows, o_groups*o_lora_rank];
    ///   stage 2: ONE shared GEMM — out [rows, hidden] = oa @ o_proj_b^T
    ///            (o_proj_b [hidden, o_groups*o_lora_rank] row-major).
    /// Both stages run on batched_gemm_bf16 (BF16 in, FP32 accumulate).
    /// V4-7b's execute_attention_v4 calls this after the inverse-roped
    /// attention output; `out` is the caller's [rows, hidden] BF16 buffer
    /// (the attention-side hc_post input — hidden_out_ is max_batch-sized
    /// and prefill chunk rows exceed it, so no internal output buffer).
    /// `stream` nullptr → this rank's attention stream.
    /// Fails loud: dcp_size > 1 (TD-V4-TP — V4-2c shards stage 1 by group +
    /// row-parallel o_proj_b + TP allreduce; until then tp is pinned 1 and
    /// there is NO reduce here), unconfigured Options (v4_o_groups == 0),
    /// rows over the scratch bound, or null o_proj_a/o_proj_b.
    void execute_v4_grouped_oproj(int rank, const AttentionLayerWeights& w,
                                  const void* attn_out, int rows, void* out,
                                  void* stream = nullptr);

    /// V4-7b (ticket H, resolves TD-V4-ATTN-ROUTING): the DeepSeek-V4
    /// per-layer attention pipeline — the V4 sibling of execute_attention.
    /// B==1 decode-shaped steps only (prompt-fed prefill = one step per
    /// prompt token); requires Options::v4.enabled and params.v4. Per layer:
    /// rmsnorm → q_a/q_a_norm/q_b → v4_q_prep → attn_kv/kv_a_norm →
    /// v4_raw_kv_append (SWA ring); compressor state GEMMs + ring writes +
    /// stride-boundary v4_compress_insert (CSA/HCA main entry + CSA LID);
    /// CSA lightning select (logical→physical translate, trap #11);
    /// csa_hca_device_attention (sinks + inverse-rope epilogues) →
    /// execute_v4_grouped_oproj into hidden_out_[0]. The dispatcher tail
    /// (mHC hc_post residual + fused gate) consumes hidden_out() unchanged.
    /// Accepted shapes: B==1 decode steps; same-seq consecutive-position
    /// chunks (chunk_len == rows) — dspark verify chunks (ring-clamped,
    /// rewind-snapshotted) and chunked prefill (monotone, up to
    /// max(max_batch, superchunk_tokens) rows; TD-V4-CHUNK-PREFILL lifted
    /// 2026-08-21). Throws on graph shapes, row-bound violations, and
    /// ring-state position discontinuities (illegal rewinds).
    void execute_attention_v4(const AttentionExecParams& params);

    /// V4-7b: release the per-sequence compressor state rings. Called from
    /// the dispatcher's seq-free path; no-op for unknown ids / non-V4.
    void v4_free_sequence(uint64_t seq_id);
    /// TD-V4-SERVE-PREFIX: clone src's per-seq V4 state (per-rank ring +
    /// snapshot blocks, D2D on kAttention; host step tracking) into dst at
    /// CMD_SEQ_FORK. No-op when src has no state; throws on allocation
    /// failure or an existing dst.
    void v4_fork_sequence(uint64_t src_id, uint64_t dst_id);

    // ── Buffer registration (IPC-6) ─────────────────────────────────

    /// Register all intermediate buffers with the given BufferRegistry.
    /// Called by Engine at init time.  buf_ids are assigned by the registry.
    void register_buffers(daemon::BufferRegistry& registry);

    // ── Layer weight storage for predictive dequant ────────────────────

    /// Store per-layer attention weights for all layers (enables predictive
    /// kv_bv dequant scheduling).  Called by Engine after weight upload.
    void set_layer_weights(
        std::vector<std::vector<const AttentionLayerWeights*>> all_weights,
        int total_layers);

    /// Pre-dequant kv_bv for layers 0 and 1 into permanent pool slots.
    /// Must be called after set_layer_weights().
    void prime_dequant_pool();

    // ── Queries ────────────────────────────────────────────────────────

    int dcp_size() const { return dcp_size_; }
    bool is_active() const { return dcp_size_ >= 2; }
    int num_heads_local() const { return num_heads_local_; }
    const std::vector<config::GpuRef>& gpus() const { return opts_.gpus; }
    const std::vector<void*>& hidden_out() const { return hidden_out_; }

    // V4-4c dual RoPE: per-rank device cos/sin tables. `rope_table_device` is
    // the base-theta table (all models); `rope_table_compress_device` is the
    // compress-theta table (V4 compressed layers only; nullptr otherwise).
    // Consumers select per layer via ModelConfig::layer_uses_compress_rope.
    const void* rope_table_device(int rank) const {
        return (rank >= 0 && static_cast<size_t>(rank) < rope_cos_sin_.size())
                   ? rope_cos_sin_[static_cast<size_t>(rank)] : nullptr;
    }
    const void* rope_table_compress_device(int rank) const {
        return (rank >= 0
                && static_cast<size_t>(rank) < rope_cos_sin_compress_.size())
                   ? rope_cos_sin_compress_[static_cast<size_t>(rank)]
                   : nullptr;
    }

    // Route one GGUF-quantized projection GEMM to the right kernel (GG-4).
    // Strategy `int`: M ≤ 8 (decode) → gguf_mmvq, else (prefill) → gguf_mmq,
    // both using the per-rank Q8_1 workspace; strategy `dequant` →
    // gguf_dequant_gemm (no workspace). A is BF16, C is BF16, B is the packed
    // GGUF weight; `type` is the projection's own k-quant type.  PUBLIC
    // (#16): the CommandDispatcher's MTP eh_proj projection reuses this
    // single-GEMM router + workspace instead of duplicating the strategy/M
    // dispatch (see dispatch_mtp_projection).
    void route_gguf_gemm(compute::AttentionDevice* attn,
                         int rank, int M, int N, int K,
                         const void* A, const void* B, void* C,
                         model::GgufKQuantType type, void* stream) const;

private:
    void execute_attention_graph(const AttentionExecParams& params);
    void execute_attention_nongraph(const AttentionExecParams& params);
    void execute_common_prefix(const AttentionExecParams& params);
    // in_ld_heads: head count of corrected_outputs' per-token leading dim.
    // HL for the legacy per-rank layout; dcp*HL under sharded KV where
    // corrected_outputs[r] points at rank r's HL-head slice INSIDE the
    // all-head combined buffer [B, dcp*HL, D_c] (INV-KVS-QAG slicing).
    void execute_oproj_and_reduce(const AttentionExecParams& params,
                                   void* const* corrected_outputs,
                                   int in_ld_heads);

    void allocate_buffers();
    void free_buffers();

    // GLM-25a: run the DSA lightning indexer for one rank/layer to produce this
    // step's sparse block indices. Reads the q-a-norm latent (q_compressed_) and
    // normed hidden (normed_hidden_), appends the token's key to the persistent
    // per-layer indexer-K cache at its position, scores all cached positions and
    // writes sparse_indices_/topk_lengths_. Single-sequence step model (B==1,
    // dcp_size==1, seqlen ≤ indexer_cache_tokens_); returns false → dense
    // otherwise or when the layer has no indexer weights. Returns true when
    // sparse indices were produced (caller flips params.is_sparse). The
    // emitted indices are GLOBAL positions; under sharded KV (KVS-4) the
    // caller translates them per rank via indexer_shard_translate before the
    // sparse consumer runs.
    bool produce_sparse_indices(compute::AttentionDevice* attn, int rank,
                                const AttentionExecParams& params);

    // TD-GLM-INDEXER-PREFILL: prefill/chunked-prefill indexer-K chunk
    // appender — the producer's K half (k-proj → LayerNorm(w+b) → RoPE →
    // Hadamard → FP8 quant-append) batched over all chunk rows, appending
    // each row's key at its own position into the SAME storage the decode
    // producer scores (paged rows or executor arena). NO scoring, NO sparse
    // output. Only layers that OWN indexer-K storage append (IndexShare full
    // ∪ layer 0); shared layers no-op (return true). Returns false when the
    // append could not run (missing blessing/weights/positions).
    bool append_indexer_chunk(compute::AttentionDevice* attn, int rank,
                              const AttentionExecParams& params);

    // TD-SPARSE-CHUNK-PREFILL: prefill/chunked-prefill sparse-index producer
    // — the decode producer's Q half (q-proj → RoPE → Hadamard) + score
    // weights batched over all chunk rows, then a per-row score + causal
    // top-k: row b scores its own prefix [0, host_seqlens_k[b]) and selects
    // at query position host_seqlens_k[b] − 1 → row b of sparse_indices_dev_
    // / topk_lengths_dev_. Requires the chunk's indexer keys to be already
    // appended (append_indexer_chunk ran first this layer). Same IndexShare
    // reuse + dispatcher-blessing + storage-resolution rules as the decode
    // producer; chunk = ONE sequence, so the arena is valid at any B (in
    // replicated mode; local mode is paged-only). LOCAL indexer mode
    // (TD-SPARSE-PREFILL-LOCAL-INDEXER): row b scores only this rank's
    // owned shard bounded PER ROW at owned_len(rank, len_b) — the shard
    // already holds the chunk's LATER keys, so the per-row bound is an
    // exactness requirement, not just causality — and emits row b's shard
    // top-k as a CANDIDATE row into the packed send buffer; the caller then
    // runs merge_local_indexer_candidates. Returns true when row-wise
    // sparse indices (or candidates, local mode) were produced (caller
    // merges if local, then flips params.is_sparse; the nongraph consumer
    // runs is_sparse + chunk_causal).
    bool produce_sparse_indices_prefill(compute::AttentionDevice* attn,
                                        int rank,
                                        const AttentionExecParams& params);

    // TD-SPARSE-PREFILL-SCORE-BATCH: batched replacement for the per-row
    // score+top-k loop of produce_sparse_indices_prefill — same per-row
    // bounds/cutoffs/outputs, BIT-IDENTICAL selection (INV-DSA-BATCH), in
    // ceil(B / rows_per_wave) score+top-k launch PAIRS instead of B pairs
    // (each per-row pair itself was ceil(len_b/PT) score launches when
    // paged). rows_per_wave = batched-scratch floats / max per-row bound.
    // Serves BOTH indexer modes: replicated (writes sparse_indices_dev_/
    // topk_lengths_dev_ rows directly) and local (writes per-row candidate
    // rows into the packed send buffer for the cross-rank merge). Caller
    // must have resolved storage into indexer_page_rows_ and passed the
    // arena slot. Returns false → caller runs the per-row loop, which stays
    // authoritative (B < 2, scratch unallocated, per-row dense mask present,
    // mixed paged/arena rows, bound beyond the endpoints iota, page-table
    // overflow, or rows_per_wave < 2).
    bool prefill_score_topk_batched(compute::AttentionDevice* attn, int rank,
                                    const AttentionExecParams& params,
                                    int slot);

    // TD-GLM-INDEXER-LOCAL-MERGE: allgather the per-rank shard candidate
    // buffers and run the exact cross-rank top-k merge on every rank, per
    // batch/chunk row (row b's causal bound from host_seqlens_k[b]), into
    // sparse_indices_dev_/topk_lengths_dev_; blesses the IndexShare reuse
    // key post-merge. Shared by the decode producer path and the sparse-
    // prefill producer path (TD-SPARSE-PREFILL-LOCAL-INDEXER). Call only
    // when indexer_local_ && all producers succeeded && indexer_step_fresh_.
    void merge_local_indexer_candidates(const AttentionExecParams& params);

    // Resolve batch entry b's dispatcher-provisioned indexer-K page row for
    // `layer` on `rank`, requiring coverage of [0, len). nullptr when
    // unprovisioned/short (→ arena fallback). Shared by the decode producer
    // and the chunk appender.
    const void* const* indexer_page_row(const AttentionExecParams& params,
                                        int rank, int layer, int b,
                                        int len) const;

    Options opts_;
    int dcp_size_ = 1;
    int num_heads_local_ = 0;
    // Heads entering the ATTENTION kernels (INV-KVS-QAG): num_heads_local_
    // under replicated KV; dcp*num_heads_local_ (= num_attention_heads) under
    // sharded KV, where the Q-head allgather hands every rank all heads. The
    // engine configures the AttentionDevice h_q to match.
    int attn_num_heads_ = 0;
    int qk_head_dim_ = 0;       // qk_nope + qk_rope
    // kv_a GEMM N (kv_lora_rank + qk_rope) rounded up to 128 so the FP8 GEMM
    // takes the unpadded fast path; the weight rows are zero-padded to match
    // at init (engine quantize_attention_weights). 576 → 640 for DeepSeek.
    int kv_a_n_pad_ = 0;

    // Per-rank intermediate buffers [dcp_size]
    std::vector<void*> normed_hidden_;            // [max_batch, hidden_size] BF16
    std::vector<void*> fp8_hidden_;             // [max_batch, hidden_size] FP8
    std::vector<void*> fp8_hidden_scales_;      // [max_batch, ceil(hidden/128)] float32
    std::vector<void*> q_compressed_;           // [max_batch, q_lora_rank] BF16
    std::vector<void*> fp8_q_compressed_;       // [max_batch, q_lora_rank] FP8
    std::vector<void*> fp8_q_compressed_scales_; // float32
    std::vector<void*> q_heads_;                // [max_batch, H_local * qk_head_dim] BF16
    std::vector<void*> q_absorbed_;             // [max_batch, H_local * (kv_lora_rank + qk_rope)] BF16 (W_UK-absorbed query)
    std::vector<void*> rope_cos_sin_;           // [rope_max_pos, qk_rope] float32 cos|sin table (per rank)
    std::vector<void*> rope_cos_sin_compress_;  // V4-4c: compress-theta table (per rank; empty ptrs for non-V4)
    std::vector<void*> kv_compressed_;          // [max_batch, kv_lora_rank + qk_rope] BF16 (alloc rows padded to kv_a_n_pad_; B==1 takes the kv_a GEMM write directly)
    std::vector<void*> kv_a_pad_out_;           // [max_batch, kv_a_n_pad_] BF16 kv_a GEMM scratch, only when max_batch > 1 and padding active
    std::vector<void*> hidden_out_;             // [max_batch, hidden_size] BF16
    // V4-5c grouped o_proj stage-1 output scratch (ticket G), allocated only
    // when opts_.v4_o_groups > 0. Rows bound = max(max_batch,
    // superchunk_tokens) so V4-7b prefill chunk rows flow through.
    std::vector<void*> v4_oproj_oa_;            // [v4_oa_rows_, o_groups*o_lora_rank] BF16
    int v4_oa_rows_ = 0;                        // allocated row bound (0 = non-V4)

    // ── V4-7b pipeline scratch (allocated only when opts_.v4.enabled) ──────
    // Per-rank device buffers (tp is pinned 1 for V4 — vectors keep the
    // rank-indexed convention).
    std::vector<void*> v4_q_nope_;      // [mb, h_q, head_dim] BF16 ([448|0])
    std::vector<void*> v4_q_rope_;      // [mb, h_q, rope] BF16
    std::vector<void*> v4_state_kv_;    // [mb, 2*head_dim] BF16 (CSA max)
    std::vector<void*> v4_state_score_; // [mb, 2*head_dim] BF16
    std::vector<void*> v4_lid_kv_;      // [mb, 2*index_head_dim] BF16
    std::vector<void*> v4_lid_score_;   // [mb, 2*index_head_dim] BF16
    std::vector<void*> v4_iq_;          // [mb, n_idx_heads*index_head_dim] BF16
    std::vector<void*> v4_iw_bf_;       // [mb, n_idx_heads] BF16
    std::vector<void*> v4_iw_f32_;      // [mb, n_idx_heads] F32 (prescaled)
    std::vector<void*> v4_attn_out_;    // [mb, h_q, head_dim] BF16
    std::vector<float*> v4_lse_;        // [mb, h_q] F32
    std::vector<int*> v4_logical_idx_;  // [mb, topk] int32 (lightning out)
    std::vector<int*> v4_phys_idx_;     // [mb, idx_cap] int32 (translated)
    int v4_idx_cap_ = 0;                // max(topk, pad64(max HCA entries))
    // Small-int staging: one device block per rank with named slots (see
    // V4IntSlots in the .cpp) + device page-id table + LID page-ptr table +
    // static block-endpoints iota (4j+3).
    std::vector<int*> v4_ints_dev_;
    std::vector<int*> v4_pt_dev_;       // [max_pages] page-id table
    std::vector<const void**> v4_lid_ptrs_dev_;  // [max_lid_pages]
    std::vector<int*> v4_endpoints_;    // [max_index_blocks] iota 4j+3
    int v4_max_pages_ = 0;
    int v4_max_lid_pages_ = 0;
    int v4_max_index_blocks_ = 0;
    // Host staging SLOT RING for the per-row async H2Ds (ticket J
    // determinism, suspect S5): pageable cudaMemcpyAsync is NOT a
    // synchronous-source-consumption contract — on HMM-capable drivers the
    // DMA may read the host buffer at stream-execution time, so a single
    // buffer rewritten once per layer races its own in-flight copies.
    // Each execute_attention_v4_row call claims one fresh slot; the ring
    // covers >= 2 full steps of row calls, and wrap-around reuse is
    // separated by the per-step logits readback sync.
    std::vector<int> v4_host_ints_;              // [slots, v4_ints_stride_]
    std::vector<int> v4_host_pt_;                // [slots, max(max_pages,1)]
    std::vector<const void*> v4_host_lid_ptrs_;  // [slots, max(lid_pages,1)]
    int v4_ints_stride_ = 0;        // ints per slot (5*B_alloc + 8)
    int v4_staging_slots_ = 0;      // slot-ring capacity
    uint64_t v4_staging_next_ = 0;  // monotone per-row-call slot cursor

    // Per-seq compressor state rings (kv+score per compressor layer; LID pair
    // on CSA layers). One device block per seq, offsets precomputed per layer.
    struct V4RingOffsets {
        int64_t kv = -1, score = -1, lid_kv = -1, lid_score = -1;
        int capacity = 0, dim = 0;         // main ring geometry
        int lid_capacity = 0, lid_dim = 0; // LID ring geometry (CSA only)
    };
    std::vector<V4RingOffsets> v4_ring_off_;   // per layer
    int64_t v4_ring_bytes_per_seq_ = 0;
    struct V4SeqState {
        // V4-2c TP: one state-ring block (and spec-snapshot block) PER RANK
        // — every rank keeps its own replicated compressor/LID rings on its
        // GPU (replicated computation is bit-identical across ranks).
        std::vector<void*> block;   // [dcp] device allocs (rank r's GPU)
        std::vector<void*> snap;    // [dcp] snapshot blocks (spec_snapshots)
        int last_pos = -1;       // step-monotony guard (highest fed pos)
        // Ticket J: step-window tracking (a step = one consecutive-position
        // row window dispatched once per layer; multi-row = verify chunk).
        int step_lo = -1, step_hi = -1;    // current step window
        int prev_lo = -1, prev_hi = -1;    // previous step window
        int step_serial = 0;               // bumps at each NEW window
        bool step_rewind = false;          // current step re-feeds positions
        // TD-V4-CHUNK-PREFILL: whether the CURRENT window is snapshot-
        // covered (rows <= the ring-clamped bound). Prefill chunks exceed
        // it — they skip snapshotting and may never be rewound into.
        bool step_snapshotted = false;
        std::vector<int> layer_snap_serial;  // per-layer snapshot marker
    };
    std::unordered_map<uint64_t, V4SeqState> v4_seq_state_;
    /// Ticket J: per-layer snapshot offsets (swa entries + state rings +
    /// LID rings, max_verify_rows rows each) into V4SeqState::snap.
    struct V4SnapOffsets {
        int64_t swa = -1, kv = -1, score = -1, lid_kv = -1, lid_score = -1;
    };
    std::vector<V4SnapOffsets> v4_snap_off_;
    int64_t v4_snap_bytes_per_seq_ = 0;
    int v4_spec_rows_max_ = 1;  ///< effective verify-row bound (ring-clamped)
    /// TD-V4-CHUNK-PREFILL: chunked-prefill row bound = max(max_batch,
    /// superchunk_tokens) — matches hidden_out_'s V4 row sizing and the
    /// engine's hidden-pair-buffer bound.
    int v4_prefill_rows_max_ = 1;

    /// Ticket J: one row (position step.token_pos + row) of the V4 pipeline
    /// on ONE rank — the B==1 body execute_attention_v4 loops rows × ranks
    /// (V4-2c: rank r computes its head shard; replicated projections and
    /// tier writes run identically on every rank).
    void execute_attention_v4_row(const AttentionExecParams& params, int row,
                                  V4SeqState& st, int r);
    /// SC (superchunk port, TD-V4-PREFILL-PERF a): TRUE batch-shaped prefill
    /// body — one batched pipeline (GEMMs/prep/ring-writes/span compress/
    /// selection) + ONE decode-kernel attention call per layer over all
    /// chunk rows. Taken for chunk windows with rows > v4_spec_rows_max_ at
    /// dcp_size 1 with unpadded head tiles (TP >= 2 and snapshotted verify
    /// windows keep the per-row loop — zero numeric change to decode/spec).
    /// LS_V4_ROW_PREFILL=1 forces the per-row loop (bisect arm).
    void execute_attention_v4_chunk(const AttentionExecParams& params,
                                    V4SeqState& st);
    /// SC: batch-call host int staging (slot ring, one slot per layer call):
    /// pos[R] | seql[R] | swa_len[R] | row_nb[R] | stage_slots[R] |
    /// ring_slots[R] | comp_slots[NB] | lid_slots[NB] | prefix_slots[W] |
    /// pt[max_pages].
    std::vector<int> v4_batch_host_ints_;
    std::vector<int*> v4_batch_ints_dev_;      // [stride] per rank
    int v4_batch_ints_stride_ = 0;
    int v4_batch_nb_max_ = 0;                  // boundary-block bound per call
    int v4_batch_staging_slots_ = 0;
    uint64_t v4_batch_staging_next_ = 0;
    std::vector<int*> v4_swa_bt_dev_;          // [R, sliding_window] per rank
    /// Ticket J: per-(seq, layer, step) rewind restore + snapshot of the
    /// slots this step's rows will overwrite (SWA tier entries + state
    /// rings), on EVERY rank. No-op unless Options.v4.spec_snapshots.
    void v4_spec_layer_guard(const AttentionExecParams& params,
                             V4SeqState& st);
    /// V4-2c: padded decode-kernel head tile (64 or 128) — the q/out/lse
    /// scratch row bound; num_heads_local_ is the REAL per-rank count.
    int v4_hq_pad_ = 0;
    /// TD-V4-KVT: host readback scratch for the lightning selection when a
    /// tiered sequence needs cold-page repromotes (selection-driven).
    std::vector<int> v4_tier_ids_host_;
    bool v4_active_logged_ = false;
    bool v4_batch_logged_ = false;   // SC batched-prefill body first-use log
    std::vector<void*> fp8_corrected_;          // [max_batch, H_local * v_head_dim] FP8 (post kv_b_v projection)
    std::vector<void*> fp8_corrected_scales_;   // float32
    std::vector<void*> gemm_workspace_;         // max across all GEMMs

    // GLM-25a: DSA indexer producer scratch (per rank), allocated when has_dsa.
    // The indexer-K cache is PERSISTENT per layer × position (engine batch model:
    // one new token per sequence per step; single-sequence path, B==1), holding
    // up to indexer_cache_tokens_ positions per layer. Long-context / multi-seq
    // uses the paged kIndexerK pool (TD-GLM-INDEXER-PAGED / -BATCH).
    int indexer_cache_tokens_ = 0;              // positions per slot in the arena
    // IndexShare (GLM-25b): per-rank step key under which sparse_indices_dev_
    // was last written by a FULL layer. A shared layer reuses that buffer iff
    // its step key matches (else it recomputes — always correct). The key is
    // the dispatcher's batch fingerprint when provided (seq_id-aware,
    // TD-GLM-INDEXER-BATCH), else derived from seqlen (single-seq legacy).
    // 0 = no valid result.
    // TD-PREFILL-SUPERCHUNK: keyed PER ROW-RANGE (batch_row_offset) — a
    // superchunk interleaves K sub-chunks per layer, each with its own step
    // key and its own persistent row range in sparse_indices_dev_; a shared
    // layer must match the key of ITS sub-chunk's rows. Decode and legacy
    // prefill always use offset 0 (scalar semantics unchanged).
    std::vector<std::unordered_map<uint32_t, uint64_t>> indexer_reuse_key_;
    // Per-entry page-row scratch for the producer (avoids per-call allocation;
    // sized max_batch at allocate_buffers).
    std::vector<const void* const*> indexer_page_rows_;
    // Arena slot per layer (−1 = shared layer that never computes → no K
    // storage). Only IndexShare full layers + layer 0 get slots — GLM-5.2:
    // 21 of 79. Paged migration notes live at the allocation site.
    std::vector<int> indexer_layer_slot_;
    int indexer_arena_slots_ = 0;
    std::vector<void*> indexer_q_;              // [max_batch, index_n_heads*index_head_dim] BF16
    std::vector<void*> indexer_k_;              // [max_batch, index_head_dim] BF16
    std::vector<void*> indexer_weights_;        // [max_batch, index_n_heads] BF16 (indexer_proj out)
    std::vector<void*> indexer_score_proj_;     // [max_batch, index_n_heads] F32 (scaled score weights)
    std::vector<void*> indexer_k_cache_;        // [num_layers, cache_tokens, index_head_dim] FP8 (persistent)
    std::vector<void*> indexer_k_scales_;       // [num_layers, cache_tokens] F32 per-position scale
    std::vector<void*> indexer_scores_;         // [cache_tokens] F32 one-query score scratch
    std::vector<void*> indexer_block_endpoints_;// [cache_tokens] int32 static iota (position ids)
    std::vector<void*> indexer_topk_scores_;    // [index_topk] F32 topk output scratch
                                                // ([max_batch, index_topk] under
                                                // sparse_prefill — the batched
                                                // top-k writes per-row score rows)
    // TD-SPARSE-PREFILL-SCORE-BATCH scratch (allocated only when has_dsa &&
    // sparse_prefill; nullptr → per-row fallback). The batched producer
    // stages per-row bounds+cutoffs ([2*max_batch] int32: bounds then
    // cutoffs) and, for paged storage, a row-major per-row page-pointer
    // table, then issues ONE batched score + ONE batched top-k launch per
    // wave of rows (wave = scratch_floats / max row bound).
    std::vector<void*> indexer_scores_batched_;  // [batched_floats] F32
    size_t indexer_scores_batched_floats_ = 0;
    std::vector<void*> indexer_row_bounds_dev_;  // [2*max_batch] int32
    std::vector<void*> indexer_page_table_dev_;  // [max_batch*pages_cap] ptrs
    size_t indexer_page_table_entries_ = 0;
    std::vector<int> indexer_row_bounds_host_;   // pageable staging mirror
    std::vector<const void*> indexer_page_table_host_;
    bool indexer_batch_logged_ = false;  // one-time "batched producer" line
    std::vector<void*> sparse_indices_dev_;     // [max_batch, index_topk] int32
    std::vector<void*> topk_lengths_dev_;       // [max_batch] int32
    // KVS-4 (sharded KV only): rank-LOCAL translation of the global top-k —
    // sparse_indices_dev_ holds GLOBAL positions (identical on every rank,
    // replicated indexer); the sparse consumer indexes the rank's LOCAL KV
    // staging, so under dcp_kv_sharded these hold the per-rank
    // indexer_shard_translate output and the consumer pointers below are
    // repointed at them. nullptr under replication.
    std::vector<void*> sparse_local_indices_dev_; // [max_batch, index_topk] int32
    std::vector<void*> topk_local_lengths_dev_;   // [max_batch] int32
    std::vector<const int*> sparse_indices_ptrs_; // per-rank view for AttentionExecParams
    std::vector<const int*> topk_lengths_ptrs_;   // per-rank view (TD-GLM-INDEXER-DCP)
    bool dsa_active_logged_ = false;  // one-time "DSA ACTIVE" info line
    bool sparse_prefill_logged_ = false;  // one-time "SPARSE CHUNK PREFILL" line

    // TD-GLM-INDEXER-LOCAL-MERGE (dcp_indexer_mode=local at dcp>=2): each
    // rank's shard-local top-k CANDIDATES (LOCAL slot indices + scores) are
    // written into its packed send buffer, allgathered rank-major into the
    // recv buffer, then merged (indexer_topk_merge) into the GLOBAL top-k in
    // sparse_indices_dev_/topk_lengths_dev_ — from where the replicated-mode
    // consumers (IndexShare reuse, KVS-4 translation, sparse attention) run
    // unchanged. Send layout at runtime batch B: [B*ITK int32][B*ITK f32]
    // (buffers sized for max_batch); recv = dcp_size send segments.
    bool indexer_local_ = false;                 // local mode active (dcp>=2)
    bool indexer_step_fresh_ = false;            // last produce computed (vs reuse)
    bool indexer_merge_logged_ = false;          // one-time merge-active info line
    std::vector<void*> indexer_cand_send_;       // [2 * max_batch * ITK] words
    std::vector<void*> indexer_cand_recv_;       // [dcp, 2 * max_batch * ITK]

    // GGUF Q8_1 activation workspace [dcp_size] (GG-4). Allocated only when
    // gguf_active && int strategy. Sized for the largest (M=max_batch, K) over
    // the four GGUF projections: gguf_mmvq/gguf_mmq quantize the BF16
    // activation into M*(K/32) Q8_1 blocks (36 B each) here before the int8
    // dot-product. nullptr (and 0 bytes) when not needed.
    std::vector<void*> gguf_q8_1_ws_;
    size_t gguf_q8_1_ws_bytes_ = 0;

    // Prefill-specific buffers [dcp_size]. Sized with attn_num_heads_
    // (H_local replicated; dcp*H_local under sharded KV — the all-head
    // attention output that the INV-KVS-QAG combine corrects in place).
    std::vector<void*> prefill_out_;            // [max_batch, attn_heads * kv_lora_rank] BF16
    std::vector<float*> prefill_lse_;           // [max_batch, attn_heads] FP32

    // INV-KVS-QAG Q-head allgather buffers [dcp_size], allocated only under
    // sharded KV at dcp_size >= 2.
    // q_gathered_stage_: NCCL allgather target, rank-major
    //   [dcp, max_batch, H_local, kv_lora_rank + qk_rope] BF16. At B==1 this
    //   is directly the head-major [1, dcp*H_local, d_q] layout attention
    //   consumes (rank s holds global heads s*HL..(s+1)*HL-1).
    // q_gathered_: token-major rearrangement [max_batch, dcp*H_local, d_q]
    //   BF16 for B>1 (chunked prefill); null when max_batch == 1.
    std::vector<void*> q_gathered_stage_;
    std::vector<void*> q_gathered_;

    // NVFP4 o_proj buffers (for native NVFP4 GEMM when o_proj_is_nvfp4) [dcp_size]
    std::vector<void*> nvfp4_oproj_act_;        // [max_batch, H_local*v_head_dim/2] FP4 packed
    std::vector<void*> nvfp4_oproj_scales_;     // Sm1xx interleaved UE4M3 activation scales
    // Per-layer o_proj NVFP4 meta block: [num_layers][kOprojMetaStride] per
    // rank. Each slot: expert_offsets(2) + problem_sizes(3) + sf_offsets(2)
    // int32 + alpha + input_scale float. Uploaded lazily ONCE per (layer, B)
    // — decode steady state issues zero meta H2Ds (was 5 sync cudaMemcpy per
    // layer per rank per token).
    static constexpr size_t kOprojMetaStride = 48;  // 36 B used, 16-aligned
    std::vector<void*> nvfp4_oproj_meta_;
    std::vector<std::vector<int>> oproj_meta_resident_b_;       // [dcp][layers]; -1 = not uploaded
    std::vector<std::vector<unsigned char>> oproj_meta_host_;   // persistent H2D staging
    int total_layers_alloc_ = 1;                                 // meta slots per rank
    size_t nvfp4_oproj_gemm_ws_bytes_ = 0;

    // kv_b_v projection buffers [dcp_size]
    std::vector<void*> kv_bv_out_;             // [max_batch, H_local * v_head_dim] BF16

    // kv_b_v dequant pool (predictive FP8→BF16 for kv_b_proj V portion)
    std::unique_ptr<KvBvDequantPool> dequant_pool_;

    // Per-layer weight pointers for predictive dequant scheduling.
    // Outer: [num_layers], inner: [dcp_size] → AttentionLayerWeights*.
    std::vector<std::vector<const AttentionLayerWeights*>> all_layer_weights_;
    int total_layers_ = 0;

    // Cached stream pointers [dcp_size]
    std::vector<void*> attn_streams_;

    // INV-NCCL-GRAPH (env LS_NCCL_GRAPH, default OFF): captured per-rank
    // graphs of the Step-14 o_proj TP allreduce (fixed hidden_out_ buffers,
    // B==1 decode). Capture failure fails open to the eager reduce_hidden.
    std::unique_ptr<compute::NcclGroupGraphRunner> oproj_reduce_graph_;
    bool oproj_reduce_graph_failed_ = false;
};

}  // namespace layerstorm::parallelism
