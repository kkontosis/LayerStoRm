// NOTE (attention refactor V2 P1 code motion): the MLA and DeepSeek-V4
// attention pipelines moved to src/daemon/attention/arch_mla.cpp and
// src/daemon/attention/arch_deepseek_v4.cpp; shared file-local helpers
// moved to parallelism/dcp_executor_internal.h.

#include "parallelism/dcp_executor.h"
#include "parallelism/kv_bv_dequant_pool.h"
#include "parallelism/kv_tiering_hook.h"
#include "parallelism/v4_kv_tiering_hook.h"

#include "compute/graphs/graph_registry.h"
#include "compute/graphs/nccl_group_graph.h"  // INV-NCCL-GRAPH
#include "compute/csa_hca_sm120_attention_device.h"   // V4-7b bridge API
#include "compute/kernels/attention/v4_prep.h"        // V4-7b prep kernels
#include "core/attention_device.h"
#include "model/quantization/gguf_kquant.h"
#include "sm120/gemm/nvfp4/nvfp4_gemm.h"
#include "compute/kernels/attention/dcp_attention_wrapper.h"
#include "compute/stream_manager.h"
#include "daemon/buffer_registry.h"
#include "core/memory/vram_allocator.h"  // kV4Fp8EntryBytes (SWA tier — always FP8)
#include "daemon/kv_shard_math.h"  // round-robin ownership math (local indexer)
#include "parallelism/dcp_communicator.h"
#include "parallelism/dcp_executor_internal.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>   // LS_CHUNK_SMALLM gate (read once)
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace layerstorm::parallelism {

// ── Constructor / Destructor ────────────────────────────────────────────────

DcpExecutor::DcpExecutor(Options opts) : opts_(std::move(opts)) {
    dcp_size_ = std::max(opts_.dcp_size, 1);

    if (dcp_size_ >= 2 && static_cast<int>(opts_.gpus.size()) != dcp_size_) {
        throw std::invalid_argument(
            "DcpExecutor: gpus.size() must equal dcp_size");
    }

    if (opts_.gpus.empty()) {
        throw std::invalid_argument("DcpExecutor: gpus must not be empty");
    }

    if (static_cast<int>(opts_.attention_devices.size()) != dcp_size_) {
        throw std::invalid_argument(
            "DcpExecutor: attention_devices.size() must equal dcp_size");
    }

    num_heads_local_ = opts_.num_attention_heads / std::max(dcp_size_, 1);
    // INV-KVS-QAG: under sharded KV the Q-head allgather hands every rank ALL
    // dcp*HL heads for attention; under replication attention runs the rank's
    // own HL heads. The engine sets the AttentionDevice h_q to match.
    const bool qag = opts_.dcp_kv_sharded && dcp_size_ >= 2;
    attn_num_heads_ = qag ? num_heads_local_ * dcp_size_ : num_heads_local_;
    if (qag && !opts_.communicator) {
        throw std::invalid_argument(
            "DcpExecutor: dcp_kv_sharded at dcp_size >= 2 requires a "
            "communicator (INV-KVS-QAG Q-head allgather)");
    }
    qk_head_dim_ = opts_.qk_nope_head_dim + opts_.qk_rope_head_dim;

    allocate_buffers();

    // Cache attention stream pointers (nullptr if no stream_manager)
    // INV-4.18: StreamManager uses position indices via GpuRef.position.
    attn_streams_.resize(dcp_size_, nullptr);
    if (opts_.stream_manager) {
        for (int r = 0; r < dcp_size_; ++r) {
            attn_streams_[r] = opts_.stream_manager->stream(
                opts_.gpus[r].position, compute::StreamId::kAttention);
        }
    }
}

DcpExecutor::~DcpExecutor() {
    free_buffers();
}

// ── Buffer management ───────────────────────────────────────────────────────

void DcpExecutor::allocate_buffers() {
    const int B = opts_.max_batch_size;
    const int H = opts_.hidden_size;
    const int Q = opts_.q_lora_rank;
    const int KV = opts_.kv_lora_rank + opts_.qk_rope_head_dim;
    const int HL = num_heads_local_;
    const int V = opts_.v_head_dim;
    // D_c: compressed attention output dim (kv_lora_rank).  Used for prefill_out_
    // and DCP allreduce which operate on attention output [B, HL, kv_lora_rank].
    // V: value head dim (v_head_dim).  Used for o_proj buffers because the
    // non-absorbed o_proj weight is [H, HL * v_head_dim] (INV-MLA-1).
    // A kv_b_v projection step converts [B, HL, D_c] → [B, HL, V] before o_proj.
    const int D_c = opts_.kv_lora_rank;

    // FP8 + scale sizes
    const size_t fp8_hidden_bytes = static_cast<size_t>(B) * H;  // 1 byte per FP8
    const size_t fp8_hidden_scale_bytes = static_cast<size_t>(B) * ceildiv(H, kFp8ScaleBlockSize) * sizeof(float);
    const size_t q_compressed_bytes = static_cast<size_t>(B) * Q * 2;  // BF16
    const size_t fp8_q_compressed_bytes = static_cast<size_t>(B) * Q;
    const size_t fp8_q_compressed_scale_bytes = static_cast<size_t>(B) * ceildiv(Q, kFp8ScaleBlockSize) * sizeof(float);
    const size_t q_heads_bytes = static_cast<size_t>(B) * HL * qk_head_dim_ * 2;  // BF16
    // W_UK-absorbed query: [B, HL, kv_lora_rank + qk_rope] = [B, HL, KV] BF16.
    const size_t q_absorbed_bytes = static_cast<size_t>(B) * HL * KV * 2;  // BF16
    // kv_a GEMM N padded to 128 so fp8_gemm takes the unpadded pingpong path
    // (N=576 otherwise falls into run_padded_cooperative: 5 sync cudaMallocs,
    // ~60 strided copies and a host cudaStreamSynchronize per layer per token).
    // The weight rows are zero-padded to match (engine quantize_attention_weights).
    kv_a_n_pad_ = ceildiv(KV, 128) * 128;
    // Rows allocated at the padded width: for B==1 the GEMM writes row 0
    // directly (cols KV..pad are zeros from the zero weight rows); for B>1 a
    // 2D compaction copies [B, KV] out of the padded scratch.
    const size_t kv_compressed_bytes = static_cast<size_t>(B) * kv_a_n_pad_ * 2;  // BF16
    const bool kv_a_pad_active = (kv_a_n_pad_ != KV);
    const size_t kv_a_pad_out_bytes = (B > 1 && kv_a_pad_active)
        ? static_cast<size_t>(B) * kv_a_n_pad_ * 2 : 0;
    const size_t hidden_out_bytes = static_cast<size_t>(B) * H * 2;  // BF16
    const size_t fp8_corrected_bytes = static_cast<size_t>(B) * HL * V;
    const size_t fp8_corrected_scale_bytes = static_cast<size_t>(B) * ceildiv(HL * V, kFp8ScaleBlockSize) * sizeof(float);

    // Workspace: max across all 4 GEMMs
    size_t max_ws = 0;
    auto update_ws = [&](int M, int N, int K) {
        max_ws = std::max(max_ws,
            compute::query_fp8_gemm_workspace_size(M, N, K,
                compute::GemmOutputDtype::kBFloat16));
    };
    update_ws(B, Q, H);                          // q_a_proj
    update_ws(B, HL * qk_head_dim_, Q);          // q_b_proj
    update_ws(B, kv_a_n_pad_, H);                 // kv_a_proj (N-padded weights)
    update_ws(B, KV, H);                          // kv_a_proj (tight weights, FP8 checkpoints)
    update_ws(B, H, HL * V);                      // o_proj (FP8 path)

    // NVFP4 o_proj workspace (for native NVFP4 GEMM when o_proj_is_nvfp4)
    nvfp4_oproj_gemm_ws_bytes_ = compute::query_nvfp4_grouped_gemm_workspace_size(
        1, H, HL * V, compute::GemmOutputDtype::kBFloat16);
    max_ws = std::max(max_ws, nvfp4_oproj_gemm_ws_bytes_);

    fp8_hidden_.resize(dcp_size_);
    fp8_hidden_scales_.resize(dcp_size_);
    q_compressed_.resize(dcp_size_);
    fp8_q_compressed_.resize(dcp_size_);
    fp8_q_compressed_scales_.resize(dcp_size_);
    q_heads_.resize(dcp_size_);
    q_absorbed_.resize(dcp_size_);
    rope_cos_sin_.resize(dcp_size_);
    rope_cos_sin_compress_.resize(dcp_size_);
    kv_compressed_.resize(dcp_size_);
    kv_a_pad_out_.resize(dcp_size_, nullptr);
    hidden_out_.resize(dcp_size_);
    fp8_corrected_.resize(dcp_size_);
    fp8_corrected_scales_.resize(dcp_size_);
    normed_hidden_.resize(dcp_size_);
    gemm_workspace_.resize(dcp_size_);
    prefill_out_.resize(dcp_size_);
    prefill_lse_.resize(dcp_size_);
    kv_bv_out_.resize(dcp_size_);
    v4_oproj_oa_.resize(dcp_size_, nullptr);
    gguf_q8_1_ws_.resize(dcp_size_, nullptr);
    q_gathered_stage_.resize(dcp_size_, nullptr);
    q_gathered_.resize(dcp_size_, nullptr);

    // GLM-25a: DSA indexer producer scratch (per rank). Allocated only when the
    // model has DSA. The indexer-K cache is a persistent slot × position ARENA
    // (single-sequence step model), sized to the serving context
    // (rope_max_pos = serving.max_sequence_length — attention cannot run past
    // the rope table anyway). Slots are assigned only to layers that ever
    // COMPUTE the indexer: IndexShare full layers, plus layer 0 if shared
    // (ascending layer order means layer 0 always computes when no full result
    // precedes it; later shared layers reuse within the step). GLM-5.2 preset:
    // 21 slots × 32K × 132 B ≈ 89 MB/rank.
    //
    // Paged migration path (do with TD-GLM-INDEXER-BATCH, not before): replace
    // this arena with Pool::kIndexerK — the pool is pre-provisioned at init
    // (same upfront VRAM as the arena), so paging buys nothing at B==1; its
    // value is per-SEQUENCE pages (PageMeta.sequence_id, free_sequence) shared
    // across a multi-sequence budget + DCP sharding via
    // allocate_indexer_k_for_dcp. Steps: (1) budget the per-position F32 scale
    // into indexer_k_bytes_per_page, (2) dispatcher allocates pages per
    // seq/layer and builds indexer block-tables/slot-mappings alongside
    // build_kv_metadata (same TD-51cb dirty guard), (3) indexer_k_quant_append
    // takes a slot mapping and lightning_score_mqa takes a block table.
    // An arena can also stretch to SMALL fixed B>1 (per-seq slot column,
    // reservation = B_max × cap × n_slots × 132 B — fine for B ≤ 8 at ≤128K)
    // before pooled pages become structurally necessary (many seqs / 1M multi-
    // tenancy, where worst-case × B reservation stops fitting).
    const int NIH = opts_.index_n_heads;
    const int IHD = opts_.index_head_dim;
    // The sparse-attention consumer reads sparse_indices with stride
    // opts_.index_topk (prefill_attention topk arg), so the top-k output must be
    // laid out and padded to that stride.
    const int ITK = opts_.index_topk;
    // Arena ceiling (GLM-25k 1M capacity smoke): the arena is the B==1
    // FALLBACK for the paged Pool::kIndexerK path (the production path,
    // provisioned to the full serving window by VramAllocator). Sizing the
    // fallback to the serving context too DOUBLES the indexer-K VRAM at
    // long caps (21 slots × 1M × 132 B ≈ 2.84 GB/rank at 1M — allocation
    // failure on a 32 GB card). Cap it at 128K tokens: below the cap the
    // historical behavior is unchanged (preset 32768 ≪ cap); beyond it a
    // pool-miss falls back to dense (produce_sparse_indices false via the
    // seqlen > indexer_cache_tokens_ ceiling) exactly as a pool-miss past
    // the serving window always did — and under KV tiering dense-with-cold
    // stays fail-closed (INV-KVT-2).
    constexpr int kIndexerArenaCapTokens = 131072;
    indexer_cache_tokens_ = opts_.has_dsa
        ? std::min(std::max(opts_.rope_max_pos, 4096), kIndexerArenaCapTokens)
        : 0;
    // TD-GLM-INDEXER-LOCAL-MERGE: local (position-sharded) indexer mode is
    // meaningful only at dcp>=2 with a valid page ownership unit and a
    // communicator for the candidate allgather; otherwise fall back to the
    // replicated producer shape (dcp==1: identical by definition).
    indexer_local_ = opts_.indexer_local && dcp_size_ >= 2 && opts_.has_dsa
                  && opts_.indexer_k_page_tokens > 0
                  && opts_.communicator != nullptr;
    indexer_reuse_key_.assign(dcp_size_, {});
    indexer_page_rows_.assign(std::max(opts_.max_batch_size, 1), nullptr);
    const int total_layers = std::max(opts_.num_layers, 1);

    // Slot map: layer → arena slot (−1 = never computes → no K storage).
    // Uses the same full/shared rule as produce_sparse_indices.
    indexer_layer_slot_.assign(total_layers, -1);
    int indexer_slots = 0;
    if (opts_.has_dsa) {
        for (int l = 0; l < total_layers; ++l) {
            const bool full = opts_.indexer_full_layers.empty()
                || (l < static_cast<int>(opts_.indexer_full_layers.size())
                    && opts_.indexer_full_layers[l]);
            if (full || l == 0) indexer_layer_slot_[l] = indexer_slots++;
        }
    }
    indexer_arena_slots_ = indexer_slots;
    indexer_q_.resize(dcp_size_, nullptr);
    indexer_k_.resize(dcp_size_, nullptr);
    indexer_weights_.resize(dcp_size_, nullptr);
    indexer_score_proj_.resize(dcp_size_, nullptr);
    indexer_k_cache_.resize(dcp_size_, nullptr);
    indexer_k_scales_.resize(dcp_size_, nullptr);
    indexer_scores_.resize(dcp_size_, nullptr);
    indexer_block_endpoints_.resize(dcp_size_, nullptr);
    indexer_topk_scores_.resize(dcp_size_, nullptr);
    indexer_scores_batched_.resize(dcp_size_, nullptr);
    indexer_row_bounds_dev_.resize(dcp_size_, nullptr);
    indexer_page_table_dev_.resize(dcp_size_, nullptr);
    sparse_indices_dev_.resize(dcp_size_, nullptr);
    topk_lengths_dev_.resize(dcp_size_, nullptr);
    sparse_local_indices_dev_.resize(dcp_size_, nullptr);
    topk_local_lengths_dev_.resize(dcp_size_, nullptr);
    sparse_indices_ptrs_.resize(dcp_size_, nullptr);
    topk_lengths_ptrs_.resize(dcp_size_, nullptr);
    indexer_cand_send_.resize(dcp_size_, nullptr);
    indexer_cand_recv_.resize(dcp_size_, nullptr);

    // GGUF Q8_1 activation workspace: only the int strategy needs it (dequant
    // feeds BF16 activations straight to the kernel). Size for the worst-case
    // (M=B, K=max projection input dim): q_a/kv_a use K=H, q_b uses K=Q,
    // o_proj uses K=HL*V. One reusable buffer per rank — no per-call malloc.
    gguf_q8_1_ws_bytes_ = 0;
    if (opts_.gguf_active
        && opts_.gguf_strategy == config::GgufStrategy::int_strategy) {
        const int max_k = std::max({H, Q, HL * V});
        gguf_q8_1_ws_bytes_ = gguf_q8_1_workspace_bytes(B, max_k);
    }

    // Prefill output/LSE sizes — absorbed MLA outputs [B, h_attn, d_c] BF16
    // (d_c = kv_lora_rank, NOT v_head_dim, because attention operates in
    // compressed KV space). h_attn = attn_num_heads_: HL replicated;
    // dcp*HL under sharded KV (INV-KVS-QAG all-head attention output).
    const int HA = attn_num_heads_;
    const size_t prefill_out_bytes = static_cast<size_t>(B) * HA * opts_.kv_lora_rank * 2;
    const size_t prefill_lse_bytes = static_cast<size_t>(B) * HA * sizeof(float);
    // kv_b_v projection intermediate: [B, HL, V] BF16 (always the rank's own
    // HL heads — under sharded KV o_proj consumes the rank's combined slice).
    const size_t kv_bv_out_bytes = static_cast<size_t>(B) * HL * V * 2;

    // V4-5c grouped o_proj stage-1 scratch (ticket G): [rows_bound,
    // o_groups*o_lora_rank] BF16 per rank, only when configured. Rows bound
    // covers prefill chunk rows (superchunk staging), matching the engine's
    // hidden-pair-buffer row bound — V4-7b prefill o_proj flows through here.
    v4_oa_rows_ = (opts_.v4_o_groups > 0)
        ? std::max(B, std::max(opts_.superchunk_tokens, 1)) : 0;
    const size_t v4_oa_bytes = static_cast<size_t>(v4_oa_rows_)
        * opts_.v4_o_groups * opts_.v4_o_lora_rank * 2;  // BF16

    // ── V4-7b (ticket H): V4 attention pipeline scratch ────────────────────
    // The pipeline body is per-row (B==1 scratch at base offsets); only the
    // row-indexed output (hidden_out_) needs the chunk row bound.
    // TD-V4-CHUNK-PREFILL lift (2026-08-21): chunked prefill flows through
    // the per-row loop with rows up to max(max_batch, superchunk_tokens) —
    // the same row bound the engine sizes the hidden-pair buffers to.
    const bool v4on = opts_.v4.enabled;
    const int v4H = v4on ? opts_.v4_head_dim : 0;          // 512
    const int v4Rope = opts_.qk_rope_head_dim;             // 64
    // Cap at 512 = the IPC batch-descriptor / sideband-token-id bound —
    // no command can carry more chunk rows, and the elastic superchunk
    // capacity (tens of thousands of tokens) must not inflate the V4
    // hidden_out_/staging allocations.
    constexpr int kV4MaxChunkRows = 512;
    v4_prefill_rows_max_ = v4on
        ? std::min(std::max(B, std::max(opts_.superchunk_tokens, 1)),
                   kV4MaxChunkRows)
        : 1;
    // V4-2c TP: padded decode-kernel head tile (trap #9 — h_q must be 64
    // or 128; sub-64 per-rank head counts run zero-q-padded).
    v4_hq_pad_ = v4on ? (num_heads_local_ <= 64 ? 64 : 128) : 0;
    if (v4on && num_heads_local_ > 128) {
        throw std::invalid_argument(
            "DcpExecutor: V4 per-rank head count > 128 exceeds the csa_fp8 "
            "decode-kernel tile bound");
    }
    if (v4on) {
        v4_q_nope_.resize(dcp_size_, nullptr);
        v4_q_rope_.resize(dcp_size_, nullptr);
        v4_state_kv_.resize(dcp_size_, nullptr);
        v4_state_score_.resize(dcp_size_, nullptr);
        v4_lid_kv_.resize(dcp_size_, nullptr);
        v4_lid_score_.resize(dcp_size_, nullptr);
        v4_iq_.resize(dcp_size_, nullptr);
        v4_iw_bf_.resize(dcp_size_, nullptr);
        v4_iw_f32_.resize(dcp_size_, nullptr);
        v4_attn_out_.resize(dcp_size_, nullptr);
        v4_lse_.resize(dcp_size_, nullptr);
        v4_logical_idx_.resize(dcp_size_, nullptr);
        v4_phys_idx_.resize(dcp_size_, nullptr);
        v4_ints_dev_.resize(dcp_size_, nullptr);
        v4_pt_dev_.resize(dcp_size_, nullptr);
        v4_lid_ptrs_dev_.resize(dcp_size_, nullptr);
        v4_endpoints_.resize(dcp_size_, nullptr);

        const int max_seq = std::max(opts_.v4.max_seq, 1);
        // HCA dense visibility bound, padded to the decode kernel's
        // 64-multiple topk contract; CSA uses opts_.v4.topk (already 64-mult).
        const int hca_max = ceildiv(max_seq, std::max(opts_.v4.hca_ratio, 1));
        v4_idx_cap_ = std::max(opts_.v4.topk, ceildiv(hca_max, 64) * 64);
        // One 256-token logical block per kMain page (CSA, 64 entries) — and
        // one kHca page per 2 HCA entries = per 256 tokens: same count.
        v4_max_pages_ = ceildiv(max_seq, opts_.v4.csa_entries_per_page
                                             * opts_.v4.csa_ratio);
        v4_max_index_blocks_ = ceildiv(max_seq, std::max(opts_.v4.csa_ratio, 1));
        v4_max_lid_pages_ = opts_.v4.idx_entries_per_page > 0
            ? ceildiv(v4_max_index_blocks_, opts_.v4.idx_entries_per_page)
            : 0;
        v4_ints_stride_ = 5 * B + 8;
        // Host staging slot rings are sized after v4_spec_rows_max_ is
        // known (end of this block) — see the ticket-J S5 note in the .h.

        // Per-seq compressor state-ring layout (offsets into one device
        // block). SC (superchunk port): capacities carry the batched-prefill
        // row bound ON TOP of the model window — a batched chunk writes all
        // R rows' states BEFORE the span compress-inserts read them, so the
        // ring must hold window + R positions without aliasing (slot =
        // pos mod capacity is self-consistent everywhere: writes, compress
        // gathers, spec-guard snapshots — placement-only, bit-identical).
        // Cost: ≈(6 + ~76·R/512) MB/seq/rank for Flash (was ≈6 MB pre-SC).
        const int L = static_cast<int>(opts_.v4.attn_type.size());
        const int ring_extra = std::max(v4_prefill_rows_max_, 0);
        v4_ring_off_.assign(static_cast<size_t>(L), {});
        int64_t off = 0;
        for (int l = 0; l < L; ++l) {
            auto& ro = v4_ring_off_[static_cast<size_t>(l)];
            const uint8_t t = opts_.v4.attn_type[static_cast<size_t>(l)];
            if (t == 1) {          // CSA: overlap window 2*ratio, 2*D state
                ro.capacity = 2 * opts_.v4.csa_ratio + ring_extra;
                ro.dim = 2 * v4H;
                ro.lid_capacity = 2 * opts_.v4.csa_ratio + ring_extra;
                ro.lid_dim = 2 * opts_.index_head_dim;
            } else if (t == 2) {   // HCA: window == stride, single-half state
                ro.capacity = opts_.v4.hca_ratio + ring_extra;
                ro.dim = v4H;
            } else {
                continue;          // SWA-only: no compressor
            }
            ro.kv = off;    off += int64_t(ro.capacity) * ro.dim * 2;
            ro.score = off; off += int64_t(ro.capacity) * ro.dim * 2;
            if (ro.lid_capacity > 0) {
                ro.lid_kv = off;
                off += int64_t(ro.lid_capacity) * ro.lid_dim * 2;
                ro.lid_score = off;
                off += int64_t(ro.lid_capacity) * ro.lid_dim * 2;
            }
        }
        v4_ring_bytes_per_seq_ = off;

        // Ticket J (V4 speculation): per-seq snapshot layout — the slots a
        // step's rows overwrite (SWA-tier 1160-B entries + state-ring rows),
        // max_verify_rows rows per layer. Saved before every step's writes
        // and restored on rewind (see v4_spec_layer_guard).
        if (opts_.v4.spec_snapshots) {
            // SWA tier is ALWAYS FP8 (all TQ arms) — deps V4CacheLayout.
            constexpr int64_t kV4EntryBytes = memory::kV4Fp8EntryBytes;
            // Verify-row bound: a window longer than the SMALLEST ring
            // capacity would alias two window positions onto one slot and
            // break the snapshot/restore invariant — clamp. SC note: the
            // superchunk ring growth (capacity += prefill rows) makes
            // max_verify_rows (16) the binding bound now — γ ≤ 15 verify
            // windows snapshot correctly (was ring-clamped to 8 pre-SC).
            int rm = std::max(1, opts_.v4.max_verify_rows);
            for (int l = 0; l < L; ++l) {
                const auto& rc = v4_ring_off_[static_cast<size_t>(l)];
                if (rc.capacity > 0) rm = std::min(rm, rc.capacity);
                if (rc.lid_capacity > 0) rm = std::min(rm, rc.lid_capacity);
            }
            v4_spec_rows_max_ = rm;
            const int RM = rm;
            v4_snap_off_.assign(static_cast<size_t>(L), {});
            int64_t soff = 0;
            for (int l = 0; l < L; ++l) {
                auto& so = v4_snap_off_[static_cast<size_t>(l)];
                const auto& ro = v4_ring_off_[static_cast<size_t>(l)];
                so.swa = soff;
                soff += RM * kV4EntryBytes;
                if (ro.capacity > 0) {
                    so.kv = soff;
                    soff += int64_t(RM) * ro.dim * 2;
                    so.score = soff;
                    soff += int64_t(RM) * ro.dim * 2;
                }
                if (ro.lid_capacity > 0) {
                    so.lid_kv = soff;
                    soff += int64_t(RM) * ro.lid_dim * 2;
                    so.lid_score = soff;
                    soff += int64_t(RM) * ro.lid_dim * 2;
                }
            }
            v4_snap_bytes_per_seq_ = soff;
        }

        // Ticket J determinism (S5): size the host staging slot rings —
        // one slot per execute_attention_v4_row call, ring >= 2 full steps
        // (L layers x max verify rows) so no in-flight pageable async H2D
        // ever has its source rewritten before stream execution.
        // TD-V4-CHUNK-PREFILL: prefill chunks (rows > the spec bound) end
        // with a device sync inside execute_attention_v4, so their in-flight
        // window is ONE call — 2x the chunk row bound covers it.
        const int v4_rows_bound =
            opts_.v4.spec_snapshots ? v4_spec_rows_max_ : 1;
        v4_staging_slots_ = std::max(2 * L * std::max(v4_rows_bound, 1),
                                     2 * v4_prefill_rows_max_) + 8;
        v4_host_ints_.assign(static_cast<size_t>(v4_staging_slots_)
                                 * static_cast<size_t>(v4_ints_stride_), 0);
        v4_host_pt_.assign(static_cast<size_t>(v4_staging_slots_)
                               * static_cast<size_t>(
                                     std::max(v4_max_pages_, 1)), 0);
        v4_host_lid_ptrs_.assign(
            static_cast<size_t>(v4_staging_slots_)
                * static_cast<size_t>(std::max(v4_max_lid_pages_, 1)),
            nullptr);

        // SC (superchunk port): batched-prefill host int staging — one slot
        // per LAYER call (pos/seql/swa_len/row_nb/stage_slots/ring_slots +
        // span compress slots + ring-prefix gather slots + the block-table
        // page ids). Batched calls end with a device sync, so the in-flight
        // window is one call; 4 slots for slack.
        const int R = v4_prefill_rows_max_;
        const int Wd = std::max(opts_.v4.sliding_window, 1);
        const int min_stride = std::max(std::min(opts_.v4.csa_ratio,
                                                 opts_.v4.hca_ratio), 1);
        v4_batch_nb_max_ = R / min_stride + 2;
        v4_batch_ints_stride_ = 6 * R + 2 * v4_batch_nb_max_ + Wd
                              + std::max(v4_max_pages_, 1);
        v4_batch_staging_slots_ = 4;
        v4_batch_host_ints_.assign(
            static_cast<size_t>(v4_batch_staging_slots_)
                * static_cast<size_t>(v4_batch_ints_stride_), 0);
        v4_batch_ints_dev_.resize(dcp_size_, nullptr);
        v4_swa_bt_dev_.resize(dcp_size_, nullptr);
    }

    // INV-KVS-QAG: Q-head allgather buffers (sharded KV only). Stage is the
    // rank-major NCCL target [dcp, B, HL, KV]; q_gathered_ is the token-major
    // rearrangement [B, dcp*HL, KV], needed only at B > 1 (at B == 1 the
    // stage layout IS head-major).
    const bool qag = opts_.dcp_kv_sharded && dcp_size_ >= 2;
    const size_t q_gathered_bytes = qag
        ? static_cast<size_t>(dcp_size_) * B * HL * KV * 2 : 0;

    // SC (superchunk port): the batched V4 prefill pipeline runs the
    // projection/norm chain over up to v4_prefill_rows_max_ rows in one
    // call — grow the shared per-rank scratch to the chunk bound (V4 only;
    // non-V4 sizing byte-identical).
    const size_t v4R = v4on ? static_cast<size_t>(v4_prefill_rows_max_) : 0;
    const size_t normed_alloc = std::max(hidden_out_bytes, v4R * H * 2);
    const size_t q_cmp_alloc = std::max(q_compressed_bytes, v4R * Q * 2);
    const size_t q_heads_alloc = std::max(
        q_heads_bytes, v4R * HL * static_cast<size_t>(qk_head_dim_) * 2);
    const size_t kv_cmp_alloc = std::max(
        kv_compressed_bytes, v4R * static_cast<size_t>(kv_a_n_pad_) * 2);
    for (int r = 0; r < dcp_size_; ++r) {
        auto* attn = opts_.attention_devices[r];
        attn->set_device();
        normed_hidden_[r]           = attn->device_alloc(normed_alloc);
        fp8_hidden_[r]              = attn->device_alloc(fp8_hidden_bytes);
        fp8_hidden_scales_[r]       = attn->device_alloc(fp8_hidden_scale_bytes);
        q_compressed_[r]            = attn->device_alloc(q_cmp_alloc);
        fp8_q_compressed_[r]        = attn->device_alloc(fp8_q_compressed_bytes);
        fp8_q_compressed_scales_[r] = attn->device_alloc(fp8_q_compressed_scale_bytes);
        q_heads_[r]                 = attn->device_alloc(q_heads_alloc);
        q_absorbed_[r]              = attn->device_alloc(q_absorbed_bytes);
        kv_compressed_[r]           = attn->device_alloc(kv_cmp_alloc);
        if (kv_a_pad_out_bytes > 0)
            kv_a_pad_out_[r]        = attn->device_alloc(kv_a_pad_out_bytes);
        // RoPE cos/sin table: upload once per rank (pure table, see rope_table.h).
        if (opts_.rope_cos_sin_host && opts_.rope_max_pos > 0) {
            const size_t rope_bytes = static_cast<size_t>(opts_.rope_max_pos)
                                    * opts_.qk_rope_head_dim * sizeof(float);
            rope_cos_sin_[r] = attn->device_alloc(rope_bytes);
            if (rope_cos_sin_[r])
                attn->memcpy_h2d(rope_cos_sin_[r], opts_.rope_cos_sin_host,
                                 rope_bytes);
            // V4-4c dual RoPE: compress-theta table (compressed layers).
            if (opts_.rope_cos_sin_compress_host) {
                rope_cos_sin_compress_[r] = attn->device_alloc(rope_bytes);
                if (rope_cos_sin_compress_[r])
                    attn->memcpy_h2d(rope_cos_sin_compress_[r],
                                     opts_.rope_cos_sin_compress_host,
                                     rope_bytes);
            }
        }
        // TD-V4-CHUNK-PREFILL: V4 chunked prefill writes hidden_out_ rows
        // [0, chunk) — grow the row bound to the chunk capacity (V4 only;
        // non-V4 sizing byte-identical).
        hidden_out_[r]              = attn->device_alloc(
            v4on ? static_cast<size_t>(v4_prefill_rows_max_) * H * 2
                 : hidden_out_bytes);
        fp8_corrected_[r]           = attn->device_alloc(fp8_corrected_bytes);
        fp8_corrected_scales_[r]    = attn->device_alloc(fp8_corrected_scale_bytes);
        gemm_workspace_[r]          = (max_ws > 0)
            ? attn->device_alloc(max_ws) : nullptr;
        prefill_out_[r]             = attn->device_alloc(prefill_out_bytes);
        prefill_lse_[r]             = static_cast<float*>(
            attn->device_alloc(prefill_lse_bytes));
        kv_bv_out_[r]              = attn->device_alloc(kv_bv_out_bytes);
        if (v4_oa_bytes > 0)
            v4_oproj_oa_[r]        = attn->device_alloc(v4_oa_bytes);

        // V4-7b pipeline scratch (see sizing block above). SC: sized at the
        // batched-prefill row bound (the per-row loop uses row 0 only).
        if (v4on) {
            const size_t rows = std::max(
                static_cast<size_t>(v4_prefill_rows_max_),
                static_cast<size_t>(1));
            // V4-2c TP padding: the deps decode kernel loads full 64-head Q
            // tiles — q/out/lse buffers are sized at the padded tile bound
            // and the q buffers are ZEROED once (q_prep writes only the HL
            // real heads per call; zero-q pad heads produce ignored rows).
            const size_t HLp = static_cast<size_t>(v4_hq_pad_);
            v4_q_nope_[r] = attn->device_alloc(rows * HLp * v4H * 2);
            v4_q_rope_[r] = attn->device_alloc(rows * HLp * v4Rope * 2);
            // Default-stream memsets (attn_streams_ is cached AFTER
            // allocate_buffers) — ordered before any later kernel use.
            if (!opts_.device_backends.empty() && opts_.device_backends[r]) {
                auto* be = opts_.device_backends[r];
                if (v4_q_nope_[r])
                    be->memset_async(v4_q_nope_[r], 0, rows * HLp * v4H * 2,
                                     nullptr);
                if (v4_q_rope_[r])
                    be->memset_async(v4_q_rope_[r], 0,
                                     rows * HLp * v4Rope * 2, nullptr);
            }
            v4_state_kv_[r] = attn->device_alloc(rows * 2 * v4H * 2);
            v4_state_score_[r] = attn->device_alloc(rows * 2 * v4H * 2);
            v4_lid_kv_[r] = attn->device_alloc(rows * 2 * IHD * 2);
            v4_lid_score_[r] = attn->device_alloc(rows * 2 * IHD * 2);
            v4_iq_[r] = attn->device_alloc(rows * NIH * IHD * 2);
            v4_iw_bf_[r] = attn->device_alloc(rows * NIH * 2);
            v4_iw_f32_[r] = attn->device_alloc(rows * NIH * sizeof(float));
            v4_attn_out_[r] = attn->device_alloc(rows * HLp * v4H * 2);
            v4_lse_[r] = static_cast<float*>(
                attn->device_alloc(rows * HLp * sizeof(float)));
            v4_logical_idx_[r] = static_cast<int*>(
                attn->device_alloc(rows * opts_.v4.topk * sizeof(int)));
            v4_phys_idx_[r] = static_cast<int*>(
                attn->device_alloc(rows * v4_idx_cap_ * sizeof(int)));
            v4_ints_dev_[r] = static_cast<int*>(attn->device_alloc(
                static_cast<size_t>(v4_ints_stride_) * sizeof(int)));
            v4_pt_dev_[r] = static_cast<int*>(attn->device_alloc(
                static_cast<size_t>(v4_max_pages_) * sizeof(int)));
            v4_lid_ptrs_dev_[r] = static_cast<const void**>(attn->device_alloc(
                static_cast<size_t>(std::max(v4_max_lid_pages_, 1))
                * sizeof(void*)));
            // SC (superchunk port): batched-prefill device int staging +
            // per-row SWA block table ([rows, window] index list at
            // swa_page_block_size 1).
            v4_batch_ints_dev_[r] = static_cast<int*>(attn->device_alloc(
                static_cast<size_t>(v4_batch_ints_stride_) * sizeof(int)));
            v4_swa_bt_dev_[r] = static_cast<int*>(attn->device_alloc(
                rows * static_cast<size_t>(
                           std::max(opts_.v4.sliding_window, 1))
                * sizeof(int)));
            // Static block-endpoints iota: CSA block j's endpoint token
            // 4j+3 (causality cutoffs for the lightning select).
            v4_endpoints_[r] = static_cast<int*>(attn->device_alloc(
                static_cast<size_t>(v4_max_index_blocks_) * sizeof(int)));
            {
                std::vector<int> ep(static_cast<size_t>(v4_max_index_blocks_));
                for (int j = 0; j < v4_max_index_blocks_; ++j)
                    ep[static_cast<size_t>(j)] =
                        j * opts_.v4.csa_ratio + opts_.v4.csa_ratio - 1;
                attn->memcpy_h2d(v4_endpoints_[r], ep.data(),
                                 ep.size() * sizeof(int));
            }
        }
        if (q_gathered_bytes > 0) {
            q_gathered_stage_[r]   = attn->device_alloc(q_gathered_bytes);
            if (B > 1)
                q_gathered_[r]     = attn->device_alloc(q_gathered_bytes);
        }
        gguf_q8_1_ws_[r]           = (gguf_q8_1_ws_bytes_ > 0)
            ? attn->device_alloc(gguf_q8_1_ws_bytes_) : nullptr;

        // GLM-25a: DSA indexer producer scratch + persistent slot-mapped K arena
        // (IndexShare-aware: only computing layers get a slot; see slot map
        // above and the paged-migration comment).
        if (opts_.has_dsa) {
            const size_t CT = static_cast<size_t>(indexer_cache_tokens_);
            const size_t NS = static_cast<size_t>(std::max(indexer_arena_slots_, 1));
            indexer_q_[r]        = attn->device_alloc(size_t(B) * NIH * IHD * 2);  // BF16
            indexer_k_[r]        = attn->device_alloc(size_t(B) * IHD * 2);        // BF16
            indexer_weights_[r]  = attn->device_alloc(size_t(B) * NIH * 2);        // BF16
            indexer_score_proj_[r] = attn->device_alloc(size_t(B) * NIH * sizeof(float));
            indexer_k_cache_[r]  = attn->device_alloc(NS * CT * IHD);  // FP8 e4m3
            indexer_k_scales_[r] = attn->device_alloc(NS * CT * sizeof(float));
            indexer_scores_[r]   = attn->device_alloc(CT * sizeof(float));
            indexer_block_endpoints_[r] = attn->device_alloc(CT * sizeof(int));
            // TD-SPARSE-PREFILL-SCORE-BATCH: under sparse_prefill the batched
            // top-k writes per-ROW score rows concurrently (one CTA per row)
            // — size [max_batch, ITK]; otherwise the historical single-row
            // scratch (per-row loop overwrites it sequentially).
            const size_t tk_rows = opts_.sparse_prefill ? size_t(B) : 1;
            indexer_topk_scores_[r] = attn->device_alloc(
                tk_rows * size_t(std::max(ITK, 1)) * sizeof(float));
            // TD-PREFILL-SUPERCHUNK: the per-row top-k selection PERSISTS for
            // the whole superchunk (IndexShare shared layers of every
            // sub-chunk consume it later in the layer-wise sweep) — size the
            // rows for max(max_batch, superchunk_tokens).
            const size_t sparse_rows = static_cast<size_t>(
                std::max(B, std::max(opts_.superchunk_tokens, 1)));
            sparse_indices_dev_[r] = attn->device_alloc(
                sparse_rows * std::max(ITK, 1) * sizeof(int));
            topk_lengths_dev_[r] = attn->device_alloc(
                sparse_rows * sizeof(int));
            // KVS-4: rank-local translation targets — only sharded KV
            // consumes them (the global buffers above stay the producer's
            // output in both modes).
            if (opts_.dcp_kv_sharded && dcp_size_ >= 2) {
                sparse_local_indices_dev_[r] = attn->device_alloc(
                    size_t(B) * std::max(ITK, 1) * sizeof(int));
                topk_local_lengths_dev_[r] =
                    attn->device_alloc(size_t(B) * sizeof(int));
            }
            // TD-GLM-INDEXER-LOCAL-MERGE: per-rank candidate send buffer
            // ([B*ITK int32 local indices][B*ITK f32 scores], packed at the
            // runtime batch size) + rank-major allgather target.
            if (indexer_local_) {
                const size_t cand_bytes =
                    2 * size_t(B) * std::max(ITK, 1) * sizeof(int);
                indexer_cand_send_[r] = attn->device_alloc(cand_bytes);
                indexer_cand_recv_[r] =
                    attn->device_alloc(size_t(dcp_size_) * cand_bytes);
            }

            // TD-SPARSE-PREFILL-SCORE-BATCH (sparse chunk prefill only):
            // batched-producer scratch. Scores: sized so a full max_batch
            // chunk batches in ONE wave up to 16K-token per-row bounds and
            // degrades to ceil(B/rows_per_wave) waves beyond (never below
            // one CT row — a single row always fits; rows_per_wave < 2
            // falls back to the per-row loop). Bounds: [2*max_batch] int32
            // (per-row bound then per-row cutoff), staged H2D per call.
            // Page table: [max_batch, ceil(CT/PT)] device page pointers
            // (PT floor 64 when the ownership unit is unset at init — the
            // runtime PT is checked against the allocation per call).
            if (opts_.sparse_prefill) {
                indexer_scores_batched_floats_ =
                    std::max(CT, size_t(B) * size_t(16384));
                indexer_scores_batched_[r] = attn->device_alloc(
                    indexer_scores_batched_floats_ * sizeof(float));
                indexer_row_bounds_dev_[r] = attn->device_alloc(
                    2 * size_t(B) * sizeof(int));
                const size_t pt_floor = static_cast<size_t>(
                    std::max(opts_.indexer_k_page_tokens, 64));
                const size_t pages_cap = (CT + pt_floor - 1) / pt_floor;
                indexer_page_table_entries_ = size_t(B) * pages_cap;
                indexer_page_table_dev_[r] = attn->device_alloc(
                    indexer_page_table_entries_ * sizeof(void*));
                indexer_row_bounds_host_.assign(2 * size_t(B), 0);
                if (!indexer_scores_batched_[r] || !indexer_row_bounds_dev_[r]
                    || !indexer_page_table_dev_[r]) {
                    throw std::runtime_error(
                        "DcpExecutor: sparse-prefill batched indexer scratch "
                        "allocation failed (rank " + std::to_string(r) + ")");
                }
            }

            if (!indexer_q_[r] || !indexer_k_[r] || !indexer_weights_[r]
                || !indexer_score_proj_[r] || !indexer_k_cache_[r]
                || !indexer_k_scales_[r] || !indexer_scores_[r]
                || !indexer_block_endpoints_[r] || !indexer_topk_scores_[r]
                || !sparse_indices_dev_[r] || !topk_lengths_dev_[r]
                || (opts_.dcp_kv_sharded && dcp_size_ >= 2
                    && (!sparse_local_indices_dev_[r]
                        || !topk_local_lengths_dev_[r]))
                || (indexer_local_
                    && (!indexer_cand_send_[r] || !indexer_cand_recv_[r]))) {
                throw std::runtime_error(
                    "DcpExecutor: DSA indexer scratch allocation failed (rank "
                    + std::to_string(r) + ")");
            }

            // block_endpoints are position ids — a static iota, uploaded once.
            std::vector<int> iota(indexer_cache_tokens_);
            for (int i = 0; i < indexer_cache_tokens_; ++i) iota[i] = i;
            attn->memcpy_h2d(indexer_block_endpoints_[r], iota.data(),
                             CT * sizeof(int));
        }

        // TD-74r: uniform null-check matching NVFP4 pattern.
        if (!normed_hidden_[r] || !fp8_hidden_[r] || !fp8_hidden_scales_[r]
            || !q_compressed_[r] || !fp8_q_compressed_[r]
            || !fp8_q_compressed_scales_[r] || !q_heads_[r] || !q_absorbed_[r]
            || (opts_.rope_cos_sin_host && opts_.rope_max_pos > 0
                && !rope_cos_sin_[r])
            || (opts_.rope_cos_sin_compress_host && opts_.rope_cos_sin_host
                && opts_.rope_max_pos > 0 && !rope_cos_sin_compress_[r])
            || !kv_compressed_[r] || !hidden_out_[r]
            || !fp8_corrected_[r] || !fp8_corrected_scales_[r]
            || !prefill_out_[r] || !prefill_lse_[r]
            || !kv_bv_out_[r]
            || (q_gathered_bytes > 0 && !q_gathered_stage_[r])
            || (q_gathered_bytes > 0 && B > 1 && !q_gathered_[r])
            || (gguf_q8_1_ws_bytes_ > 0 && !gguf_q8_1_ws_[r])
            || (max_ws > 0 && !gemm_workspace_[r])) {
            throw std::runtime_error(
                "DcpExecutor: device_alloc failed for attention buffers "
                "(rank " + std::to_string(r) + ")");
        }
    }

    // NVFP4 o_proj buffers (very small: single-token sized + metadata)
    const size_t nvfp4_act_bytes = static_cast<size_t>(B) * HL * V / 2;  // FP4 packed
    // Activation scale size: query_sf_buffer_size_a gives the Sm1xx padded size
    const size_t nvfp4_scale_bytes = compute::query_sf_buffer_size_a(B, H, HL * V);
    // Per-layer metadata slots: expert_offsets[2] + problem_sizes[3] +
    // sf_offsets[2] + alpha[1] + input_scale[1] (FP4-ACT-SCALE: feeds the
    // quantizer; the alpha at meta+28 already includes the same factor).
    // One slot per layer so the contents are uploaded once per (layer, B)
    // instead of 5 sync cudaMemcpys per layer per rank per token.
    total_layers_alloc_ = std::max(1, opts_.num_layers);
    const size_t nvfp4_meta_bytes =
        kOprojMetaStride * static_cast<size_t>(total_layers_alloc_);
    nvfp4_oproj_act_.resize(dcp_size_);
    nvfp4_oproj_scales_.resize(dcp_size_);
    nvfp4_oproj_meta_.resize(dcp_size_);
    oproj_meta_resident_b_.assign(
        static_cast<size_t>(dcp_size_),
        std::vector<int>(static_cast<size_t>(total_layers_alloc_), -1));
    oproj_meta_host_.assign(static_cast<size_t>(dcp_size_),
                            std::vector<unsigned char>(nvfp4_meta_bytes, 0));
    for (int r = 0; r < dcp_size_; ++r) {
        auto* attn = opts_.attention_devices[r];
        attn->set_device();
        nvfp4_oproj_act_[r]    = attn->device_alloc(nvfp4_act_bytes);
        nvfp4_oproj_scales_[r] = attn->device_alloc(nvfp4_scale_bytes);
        nvfp4_oproj_meta_[r]   = attn->device_alloc(nvfp4_meta_bytes);
        if (!nvfp4_oproj_act_[r] || !nvfp4_oproj_scales_[r]
            || !nvfp4_oproj_meta_[r]) {
            throw std::runtime_error(
                "DcpExecutor: device_alloc failed for NVFP4 o_proj buffers "
                "(rank " + std::to_string(r) + ")");
        }
    }
}

void DcpExecutor::free_buffers() {
    // Per-rank free: each buffer at index r was allocated on attention_devices[r].
    auto free_vec = [&](std::vector<void*>& v) {
        for (int r = 0; r < static_cast<int>(v.size()); ++r) {
            if (v[r]) opts_.attention_devices[r]->device_free(v[r]);
        }
        v.clear();
    };
    free_vec(normed_hidden_);
    free_vec(fp8_hidden_);
    free_vec(fp8_hidden_scales_);
    free_vec(q_compressed_);
    free_vec(fp8_q_compressed_);
    free_vec(fp8_q_compressed_scales_);
    free_vec(q_heads_);
    free_vec(q_absorbed_);
    free_vec(rope_cos_sin_);
    free_vec(rope_cos_sin_compress_);
    free_vec(kv_compressed_);
    free_vec(kv_a_pad_out_);
    free_vec(hidden_out_);
    free_vec(fp8_corrected_);
    free_vec(fp8_corrected_scales_);
    free_vec(gemm_workspace_);
    free_vec(prefill_out_);
    // prefill_lse_ is float* but allocated via device_alloc
    for (int r = 0; r < static_cast<int>(prefill_lse_.size()); ++r) {
        if (prefill_lse_[r]) opts_.attention_devices[r]->device_free(prefill_lse_[r]);
    }
    prefill_lse_.clear();
    free_vec(kv_bv_out_);
    free_vec(v4_oproj_oa_);
    // V4-7b pipeline scratch + per-seq state rings.
    free_vec(v4_q_nope_);
    free_vec(v4_q_rope_);
    free_vec(v4_state_kv_);
    free_vec(v4_state_score_);
    free_vec(v4_lid_kv_);
    free_vec(v4_lid_score_);
    free_vec(v4_iq_);
    free_vec(v4_iw_bf_);
    free_vec(v4_iw_f32_);
    free_vec(v4_attn_out_);
    for (int r = 0; r < static_cast<int>(v4_lse_.size()); ++r)
        if (v4_lse_[r]) opts_.attention_devices[r]->device_free(v4_lse_[r]);
    v4_lse_.clear();
    auto free_int_vec = [&](auto& v) {
        for (int r = 0; r < static_cast<int>(v.size()); ++r)
            if (v[r]) opts_.attention_devices[r]->device_free(
                reinterpret_cast<void*>(v[r]));
        v.clear();
    };
    free_int_vec(v4_logical_idx_);
    free_int_vec(v4_phys_idx_);
    free_int_vec(v4_ints_dev_);
    free_int_vec(v4_pt_dev_);
    free_int_vec(v4_lid_ptrs_dev_);
    free_int_vec(v4_endpoints_);
    free_int_vec(v4_batch_ints_dev_);   // SC batched-prefill staging
    free_int_vec(v4_swa_bt_dev_);
    if (!v4_seq_state_.empty() && !opts_.attention_devices.empty()) {
        for (auto& [sid, st] : v4_seq_state_) {
            for (int q = 0; q < dcp_size_ && static_cast<size_t>(q)
                     < opts_.attention_devices.size(); ++q) {
                opts_.attention_devices[q]->set_device();
                if (static_cast<size_t>(q) < st.block.size()
                    && st.block[static_cast<size_t>(q)])
                    opts_.attention_devices[q]->device_free(
                        st.block[static_cast<size_t>(q)]);
                if (static_cast<size_t>(q) < st.snap.size()
                    && st.snap[static_cast<size_t>(q)])
                    opts_.attention_devices[q]->device_free(
                        st.snap[static_cast<size_t>(q)]);
            }
        }
        v4_seq_state_.clear();
    }
    free_vec(q_gathered_stage_);
    free_vec(q_gathered_);
    free_vec(gguf_q8_1_ws_);
    free_vec(nvfp4_oproj_act_);
    free_vec(nvfp4_oproj_scales_);
    free_vec(nvfp4_oproj_meta_);
    // GLM-25a: DSA indexer producer scratch.
    free_vec(indexer_q_);
    free_vec(indexer_k_);
    free_vec(indexer_weights_);
    free_vec(indexer_score_proj_);
    free_vec(indexer_k_cache_);
    free_vec(indexer_k_scales_);
    free_vec(indexer_scores_);
    free_vec(indexer_block_endpoints_);
    free_vec(indexer_topk_scores_);
    // TD-SPARSE-PREFILL-SCORE-BATCH scratch.
    free_vec(indexer_scores_batched_);
    free_vec(indexer_row_bounds_dev_);
    free_vec(indexer_page_table_dev_);
    free_vec(sparse_indices_dev_);
    free_vec(topk_lengths_dev_);
    free_vec(sparse_local_indices_dev_);
    free_vec(topk_local_lengths_dev_);
    free_vec(indexer_cand_send_);
    free_vec(indexer_cand_recv_);
    sparse_indices_ptrs_.clear();
    topk_lengths_ptrs_.clear();
    // dequant_pool_ destroyed by unique_ptr in destructor.
}

// ── GGUF GEMM routing (GG-4) ────────────────────────────────────────────────
//
// CUDA-FREE routing-by-(strategy, M): the .cpp picks the kernel; the CUDA
// backend (.cu) owns the kernel calls. int strategy splits decode (mmvq) vs
// prefill (mmq) at M ≤ 8 (mirrors the kernel's GEMV/GEMM split at M=8);
// dequant strategy always uses the lossless-activation dequant GEMM. The
// per-projection k-quant type is passed through verbatim.
void DcpExecutor::route_gguf_gemm(compute::AttentionDevice* attn,
                                  int rank, int M, int N, int K,
                                  const void* A, const void* B, void* C,
                                  model::GgufKQuantType type,
                                  void* stream) const {
    compute::GgufGemmParams p{};
    p.M = M; p.N = N; p.K = K;
    p.A = A; p.B = B; p.C = C;
    p.type = type;

    if (opts_.gguf_strategy == config::GgufStrategy::dequant) {
        attn->gguf_dequant_gemm(p, stream);
        return;
    }
    // int strategy: mmvq (decode, small M) vs mmq (prefill, large M).
    // Crossover 32 by DEFAULT since 2026-08-17 (TD-CHUNK-SMALLM-DEFAULT
    // resolved): the speculative-verify chunk shapes (M = chunk rows, 9..32)
    // launched MMQ with an N-tile-only grid (q_a 16 CTAs / kv_a 5 / o_proj 48
    // on a 170-SM GPU, measured 409/412/540 us per call at M=16) while the
    // mmvq kernel already register-tiles M in MT=8 passes at ~17 us/pass. NOT
    // bit-identical across the route flip (different reduction structure),
    // hence the escape hatch LS_NO_CHUNK_SMALLM=1 → crossover back to 8.
    // Shares its gate rule with the small-M GEMV (bf16_gemm.cu
    // small_m_gemv_enabled(); explicit LS_CHUNK_SMALLM=0 also disables) —
    // keep the two in lockstep.
    static const int mmvq_max_m = [] {
        const char* no = std::getenv("LS_NO_CHUNK_SMALLM");
        if (no && *no && no[0] != '0') return 8;
        const char* e = std::getenv("LS_CHUNK_SMALLM");
        if (e && *e && e[0] == '0') return 8;
        return 32;
    }();
    void* ws = (rank < static_cast<int>(gguf_q8_1_ws_.size()))
                 ? gguf_q8_1_ws_[rank] : nullptr;
    if (M <= mmvq_max_m) {
        attn->gguf_mmvq(p, ws, stream);
    } else {
        attn->gguf_mmq(p, ws, stream);
    }
}

// ── Buffer registration (IPC-6) ────────────────────────────────────────────

void DcpExecutor::register_buffers(daemon::BufferRegistry& registry) {
    const int B = opts_.max_batch_size;
    const int H = opts_.hidden_size;
    const int Q = opts_.q_lora_rank;
    const int KV = opts_.kv_lora_rank + opts_.qk_rope_head_dim;
    const int HL = num_heads_local_;
    const int V = opts_.v_head_dim;
    const int D_c = opts_.kv_lora_rank;  // compressed attention output dim

    const auto sz_fp8_hidden         = static_cast<int64_t>(B) * H;
    const auto sz_fp8_hidden_scales  = static_cast<int64_t>(B) * ceildiv(H, kFp8ScaleBlockSize) * int{sizeof(float)};
    const auto sz_q_compressed       = static_cast<int64_t>(B) * Q * 2;
    const auto sz_fp8_q_compressed   = static_cast<int64_t>(B) * Q;
    const auto sz_fp8_q_comp_scales  = static_cast<int64_t>(B) * ceildiv(Q, kFp8ScaleBlockSize) * int{sizeof(float)};
    const auto sz_q_heads            = static_cast<int64_t>(B) * HL * qk_head_dim_ * 2;
    const auto sz_q_absorbed         = static_cast<int64_t>(B) * HL * KV * 2;
    const auto sz_kv_compressed      = static_cast<int64_t>(B) * kv_a_n_pad_ * 2;  // rows padded for the kv_a GEMM fast path
    const auto sz_hidden_out         = static_cast<int64_t>(B) * H * 2;
    const auto sz_fp8_corrected      = static_cast<int64_t>(B) * HL * V;
    const auto sz_fp8_corr_scales    = static_cast<int64_t>(B) * ceildiv(HL * V, kFp8ScaleBlockSize) * int{sizeof(float)};
    // attn_num_heads_: HL replicated; dcp*HL under sharded KV (INV-KVS-QAG).
    const int HA = attn_num_heads_;
    const auto sz_prefill_out        = static_cast<int64_t>(B) * HA * D_c * 2;  // TD-74a: kv_lora_rank (attention output)
    const auto sz_prefill_lse        = static_cast<int64_t>(B) * HA * int{sizeof(float)};

    for (int r = 0; r < dcp_size_; ++r) {
        int gpu = opts_.gpus[static_cast<size_t>(r)].position;
        auto name = [&](const char* buf) {
            return std::string("dcp.") + buf + ".rank" + std::to_string(r);
        };

        registry.register_buffer(normed_hidden_[r],            sz_hidden_out,         gpu, name("normed_hidden").c_str());
        registry.register_buffer(fp8_hidden_[r],              sz_fp8_hidden,         gpu, name("fp8_hidden").c_str());
        registry.register_buffer(fp8_hidden_scales_[r],       sz_fp8_hidden_scales,  gpu, name("fp8_hidden_scales").c_str());
        registry.register_buffer(q_compressed_[r],            sz_q_compressed,       gpu, name("q_compressed").c_str());
        registry.register_buffer(fp8_q_compressed_[r],        sz_fp8_q_compressed,   gpu, name("fp8_q_compressed").c_str());
        registry.register_buffer(fp8_q_compressed_scales_[r], sz_fp8_q_comp_scales,  gpu, name("fp8_q_compressed_scales").c_str());
        registry.register_buffer(q_heads_[r],                 sz_q_heads,            gpu, name("q_heads").c_str());
        registry.register_buffer(q_absorbed_[r],              sz_q_absorbed,         gpu, name("q_absorbed").c_str());
        if (rope_cos_sin_[r]) {
            const auto sz_rope = static_cast<int64_t>(opts_.rope_max_pos)
                               * opts_.qk_rope_head_dim * int{sizeof(float)};
            registry.register_buffer(rope_cos_sin_[r], sz_rope, gpu, name("rope_cos_sin").c_str());
            if (rope_cos_sin_compress_[r])
                registry.register_buffer(rope_cos_sin_compress_[r], sz_rope, gpu,
                                         name("rope_cos_sin_compress").c_str());
        }
        registry.register_buffer(kv_compressed_[r],           sz_kv_compressed,      gpu, name("kv_compressed").c_str());
        registry.register_buffer(hidden_out_[r],
                                 opts_.v4.enabled
                                     ? static_cast<int64_t>(
                                           v4_prefill_rows_max_) * H * 2
                                     : sz_hidden_out,
                                 gpu, name("hidden_out").c_str());
        registry.register_buffer(fp8_corrected_[r],           sz_fp8_corrected,      gpu, name("fp8_corrected").c_str());
        registry.register_buffer(fp8_corrected_scales_[r],    sz_fp8_corr_scales,    gpu, name("fp8_corrected_scales").c_str());
        if (gemm_workspace_[r]) {
            // Workspace size not recomputable without CUDA headers; register with 0 size.
            registry.register_buffer(gemm_workspace_[r], 0, gpu, name("gemm_workspace").c_str());
        }
        registry.register_buffer(prefill_out_[r],             sz_prefill_out,        gpu, name("prefill_out").c_str());
        registry.register_buffer(prefill_lse_[r],             sz_prefill_lse,        gpu, name("prefill_lse").c_str());
        // INV-KVS-QAG: Q-head allgather buffers (sharded KV only).
        if (q_gathered_stage_[r])
            registry.register_buffer(
                q_gathered_stage_[r],
                static_cast<int64_t>(dcp_size_) * B * HL * KV * 2,
                gpu, name("q_gathered_stage").c_str());
        if (q_gathered_[r])
            registry.register_buffer(
                q_gathered_[r],
                static_cast<int64_t>(dcp_size_) * B * HL * KV * 2,
                gpu, name("q_gathered").c_str());
        // TD-GOLDEN: kv_b_v projection output (input to o_proj) for debug readback.
        registry.register_buffer(kv_bv_out_[r],
                                 static_cast<int64_t>(B) * HL * V * 2,
                                 gpu, name("kv_bv_out").c_str());
        // V4-5c grouped o_proj stage-1 output (ticket G) for debug readback.
        if (v4_oproj_oa_[r])
            registry.register_buffer(
                v4_oproj_oa_[r],
                static_cast<int64_t>(v4_oa_rows_) * opts_.v4_o_groups
                    * opts_.v4_o_lora_rank * 2,
                gpu, name("v4_oproj_oa").c_str());
        // V4-7b: pre-o_proj attention output — the layer-bisection readback
        // seam for the V4 pipeline (out [B, h_q, head_dim] BF16 + lse,
        // plus the prepped q halves).
        if (opts_.v4.enabled && r < static_cast<int>(v4_attn_out_.size())
            && v4_attn_out_[r]) {
            registry.register_buffer(
                v4_attn_out_[r],
                static_cast<int64_t>(B) * HL * opts_.v4_head_dim * 2,
                gpu, name("v4_attn_out").c_str());
            registry.register_buffer(
                v4_lse_[r], static_cast<int64_t>(B) * HL * 4,
                gpu, name("v4_attn_lse").c_str());
            registry.register_buffer(
                v4_q_nope_[r],
                static_cast<int64_t>(B) * HL * opts_.v4_head_dim * 2,
                gpu, name("v4_q_nope").c_str());
            registry.register_buffer(
                v4_q_rope_[r],
                static_cast<int64_t>(B) * HL * opts_.qk_rope_head_dim * 2,
                gpu, name("v4_q_rope").c_str());
        }
        if (nvfp4_oproj_act_[r])
            registry.register_buffer(nvfp4_oproj_act_[r],
                                     static_cast<int64_t>(B) * HL * V / 2,
                                     gpu, name("nvfp4_oproj_act").c_str());
        if (nvfp4_oproj_scales_[r])
            registry.register_buffer(
                nvfp4_oproj_scales_[r],
                static_cast<int64_t>(
                    compute::query_sf_buffer_size_a(B, H, HL * V)),
                gpu, name("nvfp4_oproj_scales").c_str());
        if (nvfp4_oproj_meta_[r])
            registry.register_buffer(
                nvfp4_oproj_meta_[r],
                static_cast<int64_t>(kOprojMetaStride) * total_layers_alloc_,
                gpu, name("nvfp4_oproj_meta").c_str());
    }
}

// ── Graph capture ───────────────────────────────────────────────────────────
//
// GLM-25c (IndexCache × decode graphs) HOOK: DSA currently runs on the
// non-graph path only (execute_attention gates the producer on
// !params.use_graph; correct + default since decode-graph capture is DCP≥2
// and DSA is dcp==1-gated, TD-GLM-INDEXER-DCP). When DSA-under-graphs lands:
//  - the indexer producer runs OUTSIDE the captured graph each step (its
//    top-k is data-dependent), writing into the SAME fixed per-rank
//    sparse_indices_dev_/topk_lengths_dev_ buffers;
//  - decode_graph_update already threads a sparse_indices pointer into the
//    captured graph as an indirection input, so the graph body needs no
//    per-pattern variants for IndexShare: full and shared layers read the
//    identical fixed buffer address (shared reuse = same pointer, no copy).
//    Only the graph's attention kernel must be the sparse variant for DSA
//    models — select at capture time from opts_.indexer_full_layers/has_dsa.

void DcpExecutor::capture_dcp_graphs() {
    if (dcp_size_ < 2) return;
    // INV-DCP-KVREP: the DCP combine graph is KV-SHARDED semantics only —
    // under replicated KV it must not exist.
    if (!opts_.dcp_kv_sharded) return;

    // TD-KVS-QAG-GRAPH: sharded KV is NONGRAPH-ONLY. The QAG combine sequence
    // (allgather Q in the head dim → all-head attention → allgather-LSE +
    // lse-correct over dcp*HL heads → output allreduce + per-rank head slice)
    // is not graph-captured yet: the pre-QAG capture here (per-rank-HL
    // allgather-LSE + correct + allreduce over the DecodeGraphRunner buffers,
    // see git history) implemented the mathematically-invalid per-rank-HL
    // combine (TD-KVS-Q-ALLGATHER) and decode graphs are engine-dormant
    // anyway (TD-DECODE-GRAPH). execute_attention forces the nongraph path
    // under sharded KV, so a captured kDcpAllreduce graph would never replay.
    spdlog::debug("DcpExecutor::capture_dcp_graphs: skipped — sharded KV is "
                  "nongraph-only (TD-KVS-QAG-GRAPH)");
}

// ── Layer weight storage + dequant pool management ──────────────────────────

void DcpExecutor::set_layer_weights(
    std::vector<std::vector<const AttentionLayerWeights*>> all_weights,
    int total_layers) {
    all_layer_weights_ = std::move(all_weights);
    total_layers_ = total_layers;

    // Weights replaced → per-layer o_proj meta (alpha/input_scale) is stale.
    for (auto& v : oproj_meta_resident_b_) std::fill(v.begin(), v.end(), -1);

    // Construct the dequant pool now that we know the layer count.
    KvBvDequantPool::Options pool_opts;
    pool_opts.dcp_size = dcp_size_;
    pool_opts.num_slots = 5;
    pool_opts.num_heads_local = num_heads_local_;
    pool_opts.qk_nope_head_dim = opts_.qk_nope_head_dim;
    pool_opts.v_head_dim = opts_.v_head_dim;
    pool_opts.kv_lora_rank = opts_.kv_lora_rank;
    pool_opts.attention_devices = opts_.attention_devices;
    pool_opts.stream_manager = opts_.stream_manager;
    dequant_pool_ = std::make_unique<KvBvDequantPool>(std::move(pool_opts));
}

void DcpExecutor::prime_dequant_pool() {
    if (!dequant_pool_ || all_layer_weights_.empty()) return;

    // Pre-dequant layers 0 and 1 (permanent slots).
    for (int l = 0; l < 2 && l < total_layers_; ++l) {
        dequant_pool_->prime(l, all_layer_weights_[l].data());
    }
}

}  // namespace layerstorm::parallelism
