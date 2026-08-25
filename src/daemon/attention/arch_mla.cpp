// MLA architecture attention pipeline (DeepSeek-V3.2 / GLM geometry):
// projections+norms+k_append common prefix, DSA lightning indexer producers,
// sparse merge, LSE combine seams, o_proj+reduce, graph/nongraph drivers.
// Moved verbatim from parallelism/dcp_executor.cpp (attention refactor V2 P1);
// still DcpExecutor members — arch classes arrive in later phases.
// CUDA-free TU (INV-GPU-1).

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
#include "daemon/kv_shard_math.h"  // round-robin ownership math (local indexer)
#include "parallelism/dcp_communicator.h"
#include "parallelism/dcp_executor_internal.h"

#include "daemon/attention/arch_mla.h"
#include "daemon/kv_tiering_manager.h"  // GLM-25k KV tiering (P2 hook move)

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>   // LS_CHUNK_SMALLM gate (read once)
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace layerstorm::parallelism {

// ── Per-layer attention: common prefix (steps 1-6) ──────────────────────────

void DcpExecutor::execute_common_prefix(const AttentionExecParams& params) {
    const int B = params.batch_size;
    const int H = opts_.hidden_size;
    const int Q = opts_.q_lora_rank;
    const int KV_dim = opts_.kv_lora_rank + opts_.qk_rope_head_dim;
    const int HL = num_heads_local_;

    for (int r = 0; r < dcp_size_; ++r) {
        auto* attn = opts_.attention_devices[r];
        attn->set_device();
        void* stream = attn_streams_[r];
        const auto& w = *params.weights[r];

        // input_layernorm(hidden_states) → normed_hidden [REPLICATED]
        attn->rmsnorm(normed_hidden_[r], params.hidden_states[r], w.input_layernorm,
                      opts_.rms_norm_eps, B, H, /*row_stride=*/H, stream);

        // Quantize normed hidden → FP8 for q_a_proj and kv_a_proj.
        // m_major_scales: these scales feed fp8_gemm scale_A (CUTLASS SFA is
        // M-major); the row-major default scrambles per-token scales at B > 1.
        // Skipped entirely when BOTH consumers (q_a, kv_a) are GGUF — those take
        // the BF16 activation directly, so the FP8 quant would be dead work.
        if (!(w.q_a_is_gguf && w.kv_a_is_gguf)) {
            attn->quantize_fp8(
                {.num_tokens = B, .hidden_size = H,
                 .input = normed_hidden_[r],
                 .output = fp8_hidden_[r],
                 .scales = fp8_hidden_scales_[r],
                 .m_major_scales = true}, stream);
        }

        // Step 1: q_a_proj(hidden) → q_compressed [REPLICATED]
        // GGUF projections consume the BF16 activation (normed_hidden_) directly
        // (no FP8 quant); the GGUF kernel quantizes to Q8_1 internally (int) or
        // dequants the weight (dequant). Otherwise: FP8 blockwise GEMM.
        if (w.q_a_is_gguf) {
            route_gguf_gemm(attn, r, B, Q, H,
                            normed_hidden_[r], w.q_a_proj, q_compressed_[r],
                            w.q_a_gguf_type, stream);
        } else {
            attn->gemm(
                {.M = B, .N = Q, .K = H,
                 .A = fp8_hidden_[r], .B = w.q_a_proj, .D = q_compressed_[r],
                 .scale_A = fp8_hidden_scales_[r], .scale_B = w.q_a_proj_scales,
                 .output_dtype = compute::GemmOutputDtype::kBFloat16},
                gemm_workspace_[r], stream);
        }

        // Step 2: q_a_layernorm(q_compressed) [REPLICATED]
        attn->rmsnorm(
            q_compressed_[r], q_compressed_[r], w.q_a_norm,
            opts_.rms_norm_eps, B, Q, /*row_stride=*/Q, stream);

        // Quantize q_compressed → FP8 for q_b_proj (m_major_scales: SFA layout).
        // Skipped when q_b is GGUF (it consumes the BF16 q_compressed directly).
        if (!w.q_b_is_gguf) {
            attn->quantize_fp8(
                {.num_tokens = B, .hidden_size = Q,
                 .input = q_compressed_[r],
                 .output = fp8_q_compressed_[r],
                 .scales = fp8_q_compressed_scales_[r],
                 .m_major_scales = true}, stream);
        }

        // Step 3: q_b_proj(q_compressed) → q_heads [TP-SHARDED, H_local heads]
        if (w.q_b_is_gguf) {
            route_gguf_gemm(attn, r, B, HL * qk_head_dim_, Q,
                            q_compressed_[r], w.q_b_proj, q_heads_[r],
                            w.q_b_gguf_type, stream);
        } else {
            attn->gemm(
                {.M = B, .N = HL * qk_head_dim_, .K = Q,
                 .A = fp8_q_compressed_[r], .B = w.q_b_proj, .D = q_heads_[r],
                 .scale_A = fp8_q_compressed_scales_[r], .scale_B = w.q_b_proj_scales,
                 .output_dtype = compute::GemmOutputDtype::kBFloat16},
                gemm_workspace_[r], stream);
        }

        // Step 3b: W_UK query absorption → q_absorbed [B, HL, kv_lora_rank + qk_rope].
        // Folds the K-up projection (K-half of kv_b_proj) into the query so attention
        // runs in the compressed kv_lora latent space — the layout SnapMLA/TQ prefill
        // and decode kernels require. Per-rank/per-head, replicated like q_heads_ (DCP-safe).
        {
            compute::QAbsorbParams qa{};
            qa.q_heads        = q_heads_[r];
            qa.kv_b_proj      = w.kv_b_proj;
            qa.q_absorbed     = q_absorbed_[r];
            qa.s_q            = B;
            qa.h_q            = HL;
            qa.d_nope_in      = opts_.qk_nope_head_dim;
            qa.d_c            = opts_.kv_lora_rank;
            qa.d_rope         = opts_.qk_rope_head_dim;
            qa.d_v            = opts_.v_head_dim;
            if (w.kv_b_is_gguf) {
                // GG-7: packed GGUF kv_b → q_absorb dequant-only branch. The
                // kernel dequants W_UK per element bit-equal to a load-time
                // BF16 dequant; GGUF scales are in-block, so pass no scales and
                // leave weight_is_fp8 false (the gguf_type selector takes over).
                // This is independent of gguf_strategy — kv_b absorption is
                // always dequant-only (no activation quant / int path).
                const int qk = model::gguf::block_values(w.kv_b_gguf_type);
                if (opts_.kv_lora_rank % qk != 0) {
                    throw std::runtime_error(
                        "DcpExecutor: GGUF kv_b q_absorb requires kv_lora_rank ("
                        + std::to_string(opts_.kv_lora_rank)
                        + ") % QK (" + std::to_string(qk)
                        + ") == 0 for type "
                        + std::string(model::gguf::type_name(w.kv_b_gguf_type))
                        + "; the in-kernel per-element dequant path cannot span "
                          "a partial super-block on the kv_lora axis (GG-7)");
                }
                qa.kv_b_proj_scales = nullptr;
                qa.weight_is_fp8    = false;
                qa.gguf_type        = static_cast<int>(w.kv_b_gguf_type);
            } else {
                qa.kv_b_proj_scales = w.kv_b_proj_scales;
                qa.weight_is_fp8    = w.kv_b_proj_is_fp8;
                qa.gguf_type        = -1;
            }
            // Fused RoPE on the rope half (TD-ROPE): pos = seqlens[s] − 1.
            // INV-KVS-POS: positions derive from the GLOBAL sequence length.
            // Under sharded KV (KVS-3) seqlens_k[r] is the rank-LOCAL shard
            // length — the dispatcher supplies global_seqlens_k for position
            // math. Under replication global_seqlens_k is null and seqlens_k
            // IS global. Requires the uploaded cos/sin table and per-seq
            // lengths; without either, the rope half passes through
            // unrotated (legacy behavior).
            const int* seqlens = params.global_seqlens_k
                ? params.global_seqlens_k[r]
                : (params.seqlens_k ? params.seqlens_k[r] : nullptr);
            if (rope_cos_sin_[r] && seqlens) {
                qa.apply_rope = true;
                qa.seqlens_k  = seqlens;
                qa.cos_sin    = rope_cos_sin_[r];
                qa.max_pos    = opts_.rope_max_pos;
            } else {
                // Silent-skip would mean position-incorrect attention — surface it.
                static std::once_flag rope_off_logged;
                std::call_once(rope_off_logged, [&] {
                    spdlog::warn("DcpExecutor: RoPE SKIPPED (cos_sin {}, seqlens "
                                 "{}) — attention runs without positions",
                                 rope_cos_sin_[r] ? "ok" : "missing",
                                 seqlens ? "ok" : "missing");
                });
            }
            attn->absorb_q(qa, stream);
        }

        // Step 4: kv_a_proj(hidden) → kv_compressed [REPLICATED]
        // Reuse fp8_hidden from step 1 (same quantized input).
        // N runs at kv_a_n_pad_ (576→640): the weight rows are zero-padded at
        // init so fp8_gemm takes the unpadded pingpong path instead of the
        // padded-cooperative fallback (which cost 5 sync cudaMallocs, ~60
        // strided copies and a host cudaStreamSynchronize per layer per token).
        // B==1 writes kv_compressed_ directly (its rows are pad-width; the pad
        // columns are exact zeros). B>1 writes the pad scratch, then one 2D
        // compaction restores the tight [B, KV_dim] layout downstream expects.
        if (w.kv_a_is_gguf) {
            // GGUF kv_a: packed weight is tight [KV_dim, H] (k-quants are never
            // N-zero-padded — the N-pad fast path is an FP8/NVFP4-only trick),
            // so it writes kv_compressed_ at the real N=KV_dim. BF16 activation.
            route_gguf_gemm(attn, r, B, KV_dim, H,
                            normed_hidden_[r], w.kv_a_proj, kv_compressed_[r],
                            w.kv_a_gguf_type, stream);
        } else {
            // Per-weight: kv_a_n_padded is set only when the rows were
            // actually zero-padded at init (NVFP4 checkpoints); tight FP8
            // checkpoint weights run at the real N (pad fallback inside GK).
            const int kv_a_N_gemm =
                (w.kv_a_n_padded > 0) ? w.kv_a_n_padded : KV_dim;
            const bool pad_indirect = (kv_a_N_gemm != KV_dim) && B > 1;
            void* kv_a_dst = pad_indirect ? kv_a_pad_out_[r] : kv_compressed_[r];
            attn->gemm(
                {.M = B, .N = kv_a_N_gemm, .K = H,
                 .A = fp8_hidden_[r], .B = w.kv_a_proj, .D = kv_a_dst,
                 .scale_A = fp8_hidden_scales_[r], .scale_B = w.kv_a_proj_scales,
                 .output_dtype = compute::GemmOutputDtype::kBFloat16},
                gemm_workspace_[r], stream);
            if (pad_indirect) {
                attn->memcpy_2d_d2d_async(
                    kv_compressed_[r], static_cast<size_t>(KV_dim) * 2,
                    kv_a_pad_out_[r], static_cast<size_t>(kv_a_N_gemm) * 2,
                    static_cast<size_t>(KV_dim) * 2, static_cast<size_t>(B),
                    stream);
            }
        }

        // Step 5: kv_a_layernorm(c_kv) — first kv_lora_rank elements of each
        // [KV_dim]-strided row only [REPLICATED]. kv_compressed_ rows are
        // interleaved [c_kv | k_pe] at stride KV_dim; normalizing them as
        // tight kv_lora_rank rows was exact only at B == 1 — at B > 1 it read
        // row 0's rope half into row 1's norm AND clobbered every row's rope
        // half with normalized output (TD-PREFILL-CHUNK-ATTN).
        attn->rmsnorm(
            kv_compressed_[r], kv_compressed_[r], w.kv_a_norm,
            opts_.rms_norm_eps, B, opts_.kv_lora_rank,
            /*row_stride=*/KV_dim, stream);

        // Step 6: fused_k_append(c_kv, k_pe) → paged KV cache [DCP-LOCAL]
        if (params.slot_mappings && params.kv_cache_ptrs) {
            // kv_compressed layout: [B, kv_lora_rank + qk_rope] BF16
            const void* c_kv = kv_compressed_[r];
            void* k_rope = static_cast<char*>(kv_compressed_[r]) +
                static_cast<size_t>(opts_.kv_lora_rank) * 2;  // BF16 = 2 bytes

            // Step 5b (TD-ROPE): rotate k_pe in place by its token position before
            // the cache write — the cache stores ROTATED rope (SnapMLA paper Eq. 2;
            // fused_k_append then pre-scales it by the inverse content scale).
            // INV-KVS-POS: position = GLOBAL length − 1 (see step 3b note).
            const int* seqlens = params.global_seqlens_k
                ? params.global_seqlens_k[r]
                : (params.seqlens_k ? params.seqlens_k[r] : nullptr);
            if (rope_cos_sin_[r] && seqlens) {
                compute::RopeRotateParams rr{};
                rr.x              = k_rope;
                rr.seqlens_k      = seqlens;
                rr.cos_sin        = rope_cos_sin_[r];
                rr.num_tokens     = B;
                rr.rows_per_token = 1;
                rr.row_stride     = opts_.kv_lora_rank + opts_.qk_rope_head_dim;
                rr.d_rope         = opts_.qk_rope_head_dim;
                rr.max_pos        = opts_.rope_max_pos;
                attn->rope_rotate(rr, stream);
            }

            // Both c_kv and k_rope live in the SAME interleaved kv_compressed_
            // row [c_kv | k_pe], so both source row strides are KV_dim.
            // Passing tight strides (d_c / d_rope) was exact only at B == 1
            // (TD-PREFILL-CHUNK-ATTN: every later chunk row appended garbage).
            attn->k_append(
                c_kv, k_rope, params.kv_cache_ptrs[r],
                params.cache_stride_block, params.cache_stride_row,
                params.slot_mappings[r], B,
                opts_.kv_lora_rank, opts_.qk_rope_head_dim,
                /*c_kv_row_stride=*/KV_dim, /*k_rope_row_stride=*/KV_dim,
                params.page_size,
                params.layer_idx, stream);
        }
    }
}

// ── DSA lightning indexer producer (GLM-25a) ────────────────────────────────

bool DcpExecutor::produce_sparse_indices(compute::AttentionDevice* attn, int rank,
                                         const AttentionExecParams& params) {
    const int r = rank;
    const int B = params.batch_size;
    const int H = opts_.hidden_size;
    const int Q = opts_.q_lora_rank;
    const int NIH = opts_.index_n_heads;
    const int IHD = opts_.index_head_dim;
    const int ITK = opts_.index_topk;
    const int d_rope = opts_.qk_rope_head_dim;  // indexer rope width = n_rot()
    void* stream = attn_streams_[r];

    // Step model: engine batch entries are separate SEQUENCES, one new token
    // each ("one new token per seq per step"). TD-GLM-INDEXER-BATCH: B>1 is
    // supported when the dispatcher provisioned per-entry indexer-K page rows
    // AND per-entry host seqlens — each entry appends/scores against ITS OWN
    // sequence's pages. The executor arena remains a B==1-only fallback (it is
    // structurally single-sequence). TD-GLM-INDEXER-DCP (replicated mode):
    // dcp>=2 runs this producer ON EVERY RANK against that rank's own replica
    // storage — KV metadata is replicated across TP ranks (KD-4f-d.1b), the
    // per-rank activations are identical, so every rank derives the identical
    // global top-k into its own device buffers. Chunked prefill is NOT this
    // producer's shape — blessed chunks go through
    // produce_sparse_indices_prefill (TD-SPARSE-CHUNK-PREFILL); unblessed
    // ones stay dense.
    if (params.chunk_len > 0) return false;
    if (params.indexer_sparse_suppress) return false;  // coverage gap → dense
    if (B < 1 || B > opts_.max_batch_size) return false;
    // TD-GLM-INDEXER-B1CASCADE / INV-DSA-ROWMIX: per-row dense mask of a
    // MIXED cohort. Masked (coverage-dead) rows are excluded from storage
    // resolution, the key append and the scoring below — they run DENSE in
    // the consumer's per-row split. Their topk_length row is zeroed so any
    // batch-wide consumer (KVS-4 shard translate) sees a bounded empty
    // selection instead of stale garbage.
    const uint8_t* row_dense = params.indexer_row_dense;
    const int layer = params.layer_idx;
    const int total_layers = std::max(opts_.num_layers, 1);
    if (layer < 0 || layer >= total_layers) return false;

    // INV-KVS-POS: RoPE positions and indexer append slots are GLOBAL token
    // positions (replicated indexer-K is indexed by global position; local
    // mode's in-page offsets are also global — only whole pages are
    // rank-routed). Under sharded KV seqlens_k[r] is rank-local — use
    // global_seqlens_k. KVS-4: under sharded KV the top-k this step ends
    // with is the GLOBAL one (emitted directly under replicated indexer;
    // reconstructed by the cross-rank merge under local indexer);
    // execute_attention translates it to rank-local staging indices via
    // indexer_shard_translate before the sparse consumer runs.
    const int* seqlens = params.global_seqlens_k
        ? params.global_seqlens_k[r]
        : (params.seqlens_k ? params.seqlens_k[r] : nullptr);
    if (!seqlens) return false;  // positions required (RoPE + append slot)

    // Per-entry host lengths: required for B>1; B==1 may fall back to the
    // step-level max_seqlen_k.
    const int* hseq = params.host_seqlens_k;
    if (B > 1 && !hseq) return false;
    const int seqlen0 = hseq ? hseq[0] : params.max_seqlen_k;
    if (seqlen0 < 1) return false;

    // IndexShare (GLM-25b): a "shared" layer reuses the preceding full layer's
    // top-k. This branch MUST precede the weight-presence gate — the HF
    // safetensors checkpoint ships indexer weights ONLY on full layers, so a
    // shared layer legitimately has none (the GGUF ships them everywhere and
    // would mask that ordering bug). Full layers run in ascending order and
    // overwrite the shared per-rank sparse_indices_dev_/topk_lengths_dev_
    // buffers; a shared layer reuses whatever the most recent full layer left
    // there — no recompute — provided that result belongs to THIS step
    // (seqlen match). If no full layer has produced this step (degenerate
    // all-shared config), it falls through and recomputes as if full (always
    // correct). The MTP layer (layer == num_hidden_layers ≥ mask size) is
    // shared by construction — matching GLM-5.2's
    // index_share_for_mtp_iteration=true.
    const bool is_full = opts_.indexer_full_layers.empty()
        || (layer < static_cast<int>(opts_.indexer_full_layers.size())
            && opts_.indexer_full_layers[layer]);
    // Sparse requires DISPATCHER BLESSING: a nonzero step key proves the
    // coverage guard ran for this step (contiguous appends, pinned storage
    // mode). Without it — non-dispatcher callers, or any path that skipped
    // provisioning — the arena/pages may hold garbage for earlier positions,
    // so the only safe answer is dense.
    const uint64_t step_key = params.indexer_step_key;
    if (step_key == 0) return false;
    // TD-PREFILL-SUPERCHUNK: reuse keys are per row-range (decode = offset 0).
    const uint32_t row_off = static_cast<uint32_t>(
        params.batch_row_offset > 0 ? params.batch_row_offset : 0);
    if (!is_full) {
        auto it = indexer_reuse_key_[r].find(row_off);
        if (it != indexer_reuse_key_[r].end() && it->second == step_key) {
            // Reuse the buffer in place — no compute. In local mode the reuse
            // key is only set AFTER a successful cross-rank merge, so the
            // reused sparse_indices_dev_ always holds the MERGED global top-k.
            indexer_step_fresh_ = false;
            return true;
        }
    }

    // Storage per entry: paged (dispatcher-provisioned Pool::kIndexerK rows,
    // TD-GLM-INDEXER-PAGED/-BATCH) when the table covers [0, len_b) for this
    // layer; the executor arena covers the B==1 case only. Only computing
    // layers (full ∪ {layer 0}) have K storage — a shared layer with no
    // storage and no reusable result cannot compute → dense.
    const int PT = params.indexer_k_page_tokens;

    // TD-GLM-INDEXER-LOCAL-MERGE: local mode requires the dispatcher-
    // provisioned page shape at the OWNERSHIP unit this executor was built
    // with (the local→global index math depends on it) — the executor arena
    // is replicated-shape only, so no arena fallback exists here.
    if (indexer_local_ && PT != opts_.indexer_k_page_tokens) return false;

    auto& rows = indexer_page_rows_;  // preallocated [max_batch]
    bool all_paged = true;
    for (int b = 0; b < B; ++b) {
        if (row_dense && row_dense[b]) {  // dense row: no storage needed
            rows[b] = nullptr;
            continue;
        }
        const int len_b = hseq ? hseq[b] : seqlen0;
        if (len_b < 1) return false;
        rows[b] = indexer_page_row(params, r, layer, b, len_b);
        if (!rows[b]) all_paged = false;
    }

    int slot = -1;
    if (!all_paged) {
        if (indexer_local_) return false;  // paged-only (no arena in local)
        if (B != 1) return false;  // arena is single-sequence only
        slot = (layer < static_cast<int>(indexer_layer_slot_.size()))
            ? indexer_layer_slot_[layer] : -1;
        if (slot < 0) return false;
        if (seqlen0 > indexer_cache_tokens_) return false;  // arena ceiling
    }

    const auto& w = *params.weights[rank];
    // Gate on actual indexer-weight presence (NOT opts_.has_dsa — index_topk
    // defaults to 2048, so has_dsa is true even for non-DSA GGUFs like GLM-4.7).
    if (!w.q_idx_b || !w.k_idx || !w.k_idx_norm || !w.k_idx_norm_bias || !w.weights_proj)
        return false;
    auto rope = [&](void* x, int rows_per_token) {
        if (!rope_cos_sin_[r] || !seqlens || d_rope <= 0) return;
        compute::RopeRotateParams rr{};
        rr.x = x; rr.seqlens_k = seqlens; rr.cos_sin = rope_cos_sin_[r];
        rr.num_tokens = B; rr.rows_per_token = rows_per_token; rr.row_stride = IHD;
        rr.d_rope = d_rope; rr.max_pos = opts_.rope_max_pos;
        attn->rope_rotate(rr, stream);
    };
    auto bf16_gemm = [&](const void* A, const void* B_in, void* C, int m, int k) {
        compute::StridedBatchedGemmBf16Params g{};
        g.m = m; g.n = B; g.k = k;
        g.A = A; g.lda = k; g.strideA = 0;
        g.B = B_in; g.ldb = k; g.strideB = 0;
        g.C = C; g.ldc = m; g.strideC = 0; g.batch_count = 1;
        attn->batched_gemm_bf16(g, stream);
    };

    // 1) indexer_q = wq_b · q_a_norm_latent → [B, NIH*IHD]; RoPE the rope slice.
    if (w.q_idx_b_is_gguf)
        route_gguf_gemm(attn, r, B, NIH * IHD, Q, q_compressed_[r], w.q_idx_b,
                        indexer_q_[r], w.q_idx_b_gguf_type, stream);
    else
        bf16_gemm(w.q_idx_b, q_compressed_[r], indexer_q_[r], NIH * IHD, Q);
    rope(indexer_q_[r], NIH);
    // 1b) Hadamard-rotate ALL NIH q head rows (QuaRot-style, orthonormal:
    //     Sylvester H_128 / sqrt(128), an involution). Matches
    //     ref/llama.cpp/src/models/deepseek32.cpp:281-286 — applied AFTER
    //     RoPE on the full concat(pe,nope) 128-dim row; matrix from
    //     ref/llama.cpp/src/llama-kv-cache.cpp:20-58 (ggml_gen_hadamard).
    //     Scores q·k are mathematically unchanged; the rotation only reduces
    //     the FP8 quantization error of the STORED key, so q and k must both
    //     be rotated before append/score (TD-GLM-INDEXER-HADAMARD).
    attn->indexer_hadamard(indexer_q_[r], B * NIH, IHD, stream);

    // 2) indexer_k = wk · normed_hidden → [B, IHD]; LayerNorm(w+bias); RoPE.
    if (w.k_idx_is_gguf)
        route_gguf_gemm(attn, r, B, IHD, H, normed_hidden_[r], w.k_idx,
                        indexer_k_[r], w.k_idx_gguf_type, stream);
    else
        bf16_gemm(w.k_idx, normed_hidden_[r], indexer_k_[r], IHD, H);
    attn->indexer_layernorm_bias(indexer_k_[r], w.k_idx_norm, w.k_idx_norm_bias,
                                 B, IHD, opts_.rms_norm_eps, stream);
    rope(indexer_k_[r], 1);
    // 2b) Hadamard-rotate k (single head row per token): AFTER LayerNorm +
    //     RoPE, BEFORE the FP8 quant append — the stored-K convention is the
    //     ROTATED key (ref/llama.cpp/src/models/deepseek32.cpp:281-290:
    //     Hadamard then cpy_k to the indexer KV cache).
    attn->indexer_hadamard(indexer_k_[r], B, IHD, stream);

    // 3) FP8-quantize each entry's key + append into ITS sequence's storage at
    //    its position (slot = seqlens_k[b] − 1: the seqlens array is the
    //    slot_mapping, slot_bias subtracts 1 plus the page start under
    //    paging). Per-entry launches because each entry's destination page —
    //    and therefore base pointer — differs. Local mode: only the OWNER
    //    rank of the new position stores its indexer-K (page-granular
    //    round-robin, owner(pos) = (pos/PT) % dcp); the owner's page row is
    //    LOCAL-compacted, so page pg maps to row slot pg/dcp while the
    //    in-page offset keeps the GLOBAL page-start bias.
    for (int b = 0; b < B; ++b) {
        if (row_dense && row_dense[b]) continue;  // dead row: never append —
        // its coverage was not advanced and its storage is permanently stale.
        const int len_b = hseq ? hseq[b] : seqlen0;
        const int pos_b = len_b - 1;
        if (indexer_local_
            && daemon::kvshard::owner_rank(static_cast<uint32_t>(pos_b), PT,
                                           dcp_size_) != r)
            continue;
        void* k_dst;
        void* s_dst;
        int bias;
        if (rows[b]) {
            const int pg = pos_b / PT;
            const int pg_slot = indexer_local_ ? pg / dcp_size_ : pg;
            auto* base = const_cast<std::byte*>(
                static_cast<const std::byte*>(rows[b][pg_slot]));
            k_dst = base;                                   // FP8 rows
            s_dst = base + static_cast<size_t>(PT) * IHD;   // F32 tail
            bias = -1 - pg * PT;  // slot = seqlens[b]−1−page_start
        } else {
            k_dst = static_cast<std::byte*>(indexer_k_cache_[r])
                + static_cast<size_t>(slot) * indexer_cache_tokens_ * IHD;
            s_dst = static_cast<float*>(indexer_k_scales_[r])
                + static_cast<size_t>(slot) * indexer_cache_tokens_;
            bias = -1;
        }
        attn->indexer_k_quant_append(
            static_cast<std::byte*>(indexer_k_[r])
                + static_cast<size_t>(b) * IHD * 2,          // BF16 row b
            seqlens + b, k_dst, s_dst,
            /*num_tokens=*/1, IHD, bias, stream);
    }

    // 4) indexer_weights = indexer_proj · normed_hidden → [B, NIH] (BF16), then
    //    scale by 1/sqrt(IHD*NIH) into the F32 score_proj (per-token weights).
    bf16_gemm(w.weights_proj, normed_hidden_[r], indexer_weights_[r], NIH, H);
    attn->indexer_scale_weights(
        indexer_weights_[r], indexer_score_proj_[r], B, NIH,
        1.0f / std::sqrt(static_cast<float>(IHD) * static_cast<float>(NIH)), stream);

    // 5) Per entry: score all len_b cached positions of ITS sequence (MQA
    //    single-K, paged per-page launches in the backend or the contiguous
    //    arena slice) then causal top-k at its position → row b of
    //    sparse_indices / topk_lengths. The scores scratch is reused
    //    sequentially — all launches are stream-ordered.
    //    LOCAL MODE (TD-GLM-INDEXER-LOCAL-MERGE): score only THIS RANK's
    //    owned shard (its LOCAL-compacted page row; per-position scores are
    //    exact global scores — the head sum is position-local and the
    //    indexer weights are replicated, INV-DCP-5) and emit the shard's
    //    top-k as a CANDIDATE list (LOCAL slot indices + scores) into the
    //    packed send buffer. The iota endpoints keep causality vacuous
    //    (local slot i < owned <= len_b, qpos = len_b−1) — every stored
    //    position is causal, exactly as in replicated mode. A rank owning
    //    ZERO positions launches the top-k at num_blocks 0, which writes the
    //    valid empty candidate row (all −1, length 0). execute_attention
    //    then allgathers the candidates and merges them into the global
    //    top-k in sparse_indices_dev_/topk_lengths_dev_.
    for (int b = 0; b < B; ++b) {
        if (row_dense && row_dense[b]) {
            // Dense row in a mixed cohort: nothing is scored for it — the
            // consumer's per-row split runs it dense. Zero its top-k length
            // so batch-wide consumers (KVS-4 indexer_shard_translate, the
            // local-mode merge skip) see a bounded EMPTY selection rather
            // than a stale/uninitialized length. Static source: address-
            // stable for the async stream-ordered copy.
            static constexpr int kZeroLen = 0;
            attn->memcpy_h2d_async(
                static_cast<int*>(topk_lengths_dev_[r]) + b, &kZeroLen,
                sizeof(int), stream);
            continue;
        }
        const int len_b = hseq ? hseq[b] : seqlen0;
        compute::IndexerScoreTopkArgs sa{};
        sa.q_all = static_cast<std::byte*>(indexer_q_[r])
            + static_cast<size_t>(b) * NIH * IHD * 2;        // BF16 row b
        sa.score_proj_all = static_cast<float*>(indexer_score_proj_[r])
            + static_cast<size_t>(b) * NIH;
        int nb = len_b;
        if (indexer_local_) {
            nb = daemon::kvshard::owned_len(
                r, static_cast<uint32_t>(len_b), PT, dcp_size_);
            sa.k_pages = rows[b];
            sa.num_k_pages = (nb + PT - 1) / PT;
            sa.page_tokens = PT;
            // Candidate outputs into the packed send buffer:
            // [B*ITK int32 local indices][B*ITK f32 scores], row b.
            int* cand = static_cast<int*>(indexer_cand_send_[r]);
            sa.sparse_indices_out = cand + static_cast<size_t>(b) * ITK;
            sa.topk_scores_scratch = reinterpret_cast<float*>(
                cand + static_cast<size_t>(B) * ITK)
                + static_cast<size_t>(b) * ITK;
            // Scratch only — the merge overwrites it with the final length.
            // (Offset by the sub-chunk row range so it never clobbers another
            // sub-chunk's persisted lengths, TD-PREFILL-SUPERCHUNK.)
            sa.topk_lengths_out = static_cast<int*>(topk_lengths_dev_[r])
                + row_off + b;
        } else if (rows[b]) {
            sa.k_pages = rows[b];
            sa.num_k_pages = (len_b + PT - 1) / PT;
            sa.page_tokens = PT;
        } else {
            sa.k_cache = static_cast<std::byte*>(indexer_k_cache_[r])
                + static_cast<size_t>(slot) * indexer_cache_tokens_ * IHD;
            sa.k_scales = static_cast<float*>(indexer_k_scales_[r])
                + static_cast<size_t>(slot) * indexer_cache_tokens_;
        }
        if (!indexer_local_) {
            sa.topk_scores_scratch = indexer_topk_scores_[r];
            sa.sparse_indices_out = static_cast<int*>(sparse_indices_dev_[r])
                + static_cast<size_t>(b) * ITK;
            sa.topk_lengths_out = static_cast<int*>(topk_lengths_dev_[r]) + b;
        }
        sa.block_endpoints = indexer_block_endpoints_[r];  // static iota
        sa.scores_scratch = indexer_scores_[r];
        sa.num_tokens = 1; sa.num_blocks = nb;
        sa.n_heads = NIH; sa.head_dim = IHD; sa.topk = ITK;
        sa.query_position_base = len_b - 1;  // entry b's causal cutoff
        attn->indexer_score_topk(sa, stream);
    }
    // Shared layers may now reuse this step's result. Local mode defers the
    // reuse blessing to execute_attention (post-merge): candidates alone are
    // not a consumable top-k.
    if (!indexer_local_) indexer_reuse_key_[r][row_off] = step_key;
    indexer_step_fresh_ = true;
    return true;
}

// Shared by the decode producer and the chunk appender: batch entry b's page
// row for `layer` on `rank`, valid only when it covers [0, len). Local
// indexer mode (TD-GLM-INDEXER-LOCAL-MERGE): the rank's row is LOCAL-
// compacted — slot j holds its j-th OWNED global page — so coverage of
// [0, len) means the first ceil(owned_len(rank, len)/PT) slots are non-null
// (a rank owning zero positions has a trivially-valid empty row).
const void* const* DcpExecutor::indexer_page_row(
    const AttentionExecParams& params, int rank, int layer, int b,
    int len) const {
    const int PT = params.indexer_k_page_tokens;
    if (!params.indexer_k_pages || !params.indexer_k_pages[rank] || PT <= 0
        || params.indexer_k_page_stride <= 0)
        return nullptr;
    const void* const* row = params.indexer_k_pages[rank]
        + static_cast<size_t>(b) * params.indexer_k_batch_stride
        + static_cast<size_t>(layer) * params.indexer_k_page_stride;
    int need = (len + PT - 1) / PT;
    if (indexer_local_) {
        const int owned = daemon::kvshard::owned_len(
            rank, static_cast<uint32_t>(len), PT, dcp_size_);
        need = (owned + PT - 1) / PT;
    }
    if (need > params.indexer_k_page_stride) return nullptr;
    for (int p = 0; p < need; ++p)
        if (!row[p]) return nullptr;
    return row;
}

// ── DSA indexer chunk appender (TD-GLM-INDEXER-PREFILL) ─────────────────────
//
// Prefill / chunked-prefill steps carry one batch row per PROMPT POSITION
// (consecutive positions of one sequence — seqlens_k[b] = pos_b + 1). This
// runs the decode producer's K half over ALL chunk rows in one pass —
// k-proj (same GEMM routing), indexer LayerNorm(w+bias), RoPE (per-row
// position via the same seqlens array the decode producer uses), Hadamard,
// FP8 quant-append — into the SAME storage the decode producer scores later
// (dispatcher-provisioned Pool::kIndexerK page rows, replicated per rank
// under dcp>=2; executor arena for a blessed single-sequence chunk). NO
// scoring, NO sparse consumption: the chunk's own attention stays dense
// prefill. IndexShare: only layers that OWN indexer-K storage append
// (slot-map rule: full ∪ layer 0); shared layers store nothing → no-op.
//
// Contract with the dispatcher's coverage guard: blessing (indexer_step_key
// != 0 + indexer_prefill_append) means contiguity and storage pinning were
// verified and coverage was advanced by chunk_len — like the decode guard,
// this is optimistic. A false return here can only come from step-invariant
// causes (weights absent on this layer, positions missing), which then fail
// identically on EVERY step of the sequence — that layer's storage is never
// written and never scored (the producer's own weight gate keeps it dense),
// so a stranded coverage bump cannot cause stale-key scoring.
bool DcpExecutor::append_indexer_chunk(compute::AttentionDevice* attn, int rank,
                                       const AttentionExecParams& params) {
    const int r = rank;
    const int B = params.batch_size;
    const int H = opts_.hidden_size;
    const int IHD = opts_.index_head_dim;
    const int d_rope = opts_.qk_rope_head_dim;
    void* stream = attn_streams_[r];

    if (params.indexer_step_key == 0) return false;  // dispatcher blessing
    if (B < 1 || B > opts_.max_batch_size) return false;
    const int layer = params.layer_idx;
    const int total_layers = std::max(opts_.num_layers, 1);
    if (layer < 0 || layer >= total_layers) return false;

    // IndexShare: shared layers own no indexer-K storage — nothing to append.
    const int slot = (layer < static_cast<int>(indexer_layer_slot_.size()))
        ? indexer_layer_slot_[layer] : -1;
    if (slot < 0) return true;

    // INV-KVS-POS: GLOBAL positions (see produce_sparse_indices note).
    const int* seqlens = params.global_seqlens_k
        ? params.global_seqlens_k[r]
        : (params.seqlens_k ? params.seqlens_k[r] : nullptr);
    if (!seqlens) return false;  // positions required (RoPE + append slot)
    const int* hseq = params.host_seqlens_k;
    if (!hseq) return false;     // per-row positions required (page routing)

    const auto& w = *params.weights[rank];
    // Same weight-presence gate as the decode producer's K side.
    if (!w.k_idx || !w.k_idx_norm || !w.k_idx_norm_bias) return false;

    // Storage per row (same resolution as the decode producer): page rows
    // when provisioned, else the executor arena (the dispatcher blesses the
    // arena only for a single-sequence chunk — the arena is structurally
    // single-sequence).
    // TD-GLM-INDEXER-LOCAL-MERGE: local mode is paged-only at the executor's
    // ownership unit (no arena — replicated shape).
    const int PT_check = params.indexer_k_page_tokens;
    if (indexer_local_ && PT_check != opts_.indexer_k_page_tokens)
        return false;

    auto& rows = indexer_page_rows_;  // preallocated [max_batch]
    for (int b = 0; b < B; ++b) {
        const int len_b = hseq[b];
        if (len_b < 1) return false;
        rows[b] = indexer_page_row(params, r, layer, b, len_b);
        if (indexer_local_ && !rows[b]) return false;  // paged-only
        if (!rows[b] && len_b > indexer_cache_tokens_)
            return false;  // arena ceiling
    }

    // K producer chain — mirrors produce_sparse_indices steps 2/2b exactly.
    // Every kernel below operates row-independently (one CTA per row), so a
    // chunk append is bit-equal to appending the same positions one-by-one
    // through the decode producer.
    if (w.k_idx_is_gguf) {
        route_gguf_gemm(attn, r, B, IHD, H, normed_hidden_[r], w.k_idx,
                        indexer_k_[r], w.k_idx_gguf_type, stream);
    } else {
        compute::StridedBatchedGemmBf16Params g{};
        g.m = IHD; g.n = B; g.k = H;
        g.A = w.k_idx;           g.lda = H;   g.strideA = 0;
        g.B = normed_hidden_[r]; g.ldb = H;   g.strideB = 0;
        g.C = indexer_k_[r];     g.ldc = IHD; g.strideC = 0;
        g.batch_count = 1;
        attn->batched_gemm_bf16(g, stream);
    }
    attn->indexer_layernorm_bias(indexer_k_[r], w.k_idx_norm, w.k_idx_norm_bias,
                                 B, IHD, opts_.rms_norm_eps, stream);
    // RoPE rotates each row at ITS OWN position (pos = seqlens_k[t] − 1 per
    // token inside the kernel) — chunk rows carry consecutive positions
    // through the per-row seqlens array, so the decode producer's
    // positioning generalizes with no kernel change.
    if (rope_cos_sin_[r] && d_rope > 0) {
        compute::RopeRotateParams rr{};
        rr.x = indexer_k_[r]; rr.seqlens_k = seqlens;
        rr.cos_sin = rope_cos_sin_[r];
        rr.num_tokens = B; rr.rows_per_token = 1; rr.row_stride = IHD;
        rr.d_rope = d_rope; rr.max_pos = opts_.rope_max_pos;
        attn->rope_rotate(rr, stream);
    }
    attn->indexer_hadamard(indexer_k_[r], B, IHD, stream);

    // FP8 quant-append, batched per destination page: a run of rows whose
    // positions land in the SAME physical page (same base pointer + same
    // slot_bias) goes in ONE launch — the kernel is one CTA per row with
    // slot = seqlens_k[t] + slot_bias, identical math to the decode
    // producer's per-row launches. Arena rows share one base — single run.
    // Local mode: each rank appends ONLY the chunk rows it OWNS (page-
    // granular round-robin); rows within one page share the owner, so the
    // same-page run grouping is automatically owner-uniform. The owner's
    // row is LOCAL-compacted (page pg at slot pg/dcp); the slot bias keeps
    // the GLOBAL page start (in-page offsets are mode-invariant).
    const int PT = params.indexer_k_page_tokens;
    for (int b = 0; b < B; ) {
        if (indexer_local_
            && daemon::kvshard::owner_rank(
                   static_cast<uint32_t>(hseq[b] - 1), PT, dcp_size_) != r) {
            ++b;
            continue;
        }
        void* k_dst;
        void* s_dst;
        int bias;
        int e = b + 1;
        if (rows[b]) {
            const int pg = (hseq[b] - 1) / PT;
            const int pg_slot = indexer_local_ ? pg / dcp_size_ : pg;
            auto* base = const_cast<std::byte*>(
                static_cast<const std::byte*>(rows[b][pg_slot]));
            k_dst = base;                                   // FP8 rows
            s_dst = base + static_cast<size_t>(PT) * IHD;   // F32 tail
            bias = -1 - pg * PT;
            while (e < B && rows[e] && (hseq[e] - 1) / PT == pg
                   && rows[e][pg_slot] == rows[b][pg_slot])
                ++e;
        } else {
            k_dst = static_cast<std::byte*>(indexer_k_cache_[r])
                + static_cast<size_t>(slot) * indexer_cache_tokens_ * IHD;
            s_dst = static_cast<float*>(indexer_k_scales_[r])
                + static_cast<size_t>(slot) * indexer_cache_tokens_;
            bias = -1;
            while (e < B && !rows[e]) ++e;
        }
        attn->indexer_k_quant_append(
            static_cast<std::byte*>(indexer_k_[r])
                + static_cast<size_t>(b) * IHD * 2,          // BF16 row b
            seqlens + b, k_dst, s_dst,
            /*num_tokens=*/e - b, IHD, bias, stream);
        b = e;
    }
    return true;
}

// ── DSA sparse-prefill producer (TD-SPARSE-CHUNK-PREFILL) ───────────────────
//
// The decode producer's Q half over ALL chunk rows in one pass (q-proj →
// RoPE at each row's own position → Hadamard; score-weight proj) followed by
// a PER-ROW score + causal top-k: chunk row b scores its own causal prefix
// [0, host_seqlens_k[b]) — which by this point includes the chunk's own keys
// up to and including row b's (append_indexer_chunk ran first) — and selects
// at query position host_seqlens_k[b] − 1, writing row b of
// sparse_indices_dev_ / topk_lengths_dev_. Identical math to running the
// decode producer once per chunk position (the score and top-k kernels are
// per-query; the bound nb = len_b keeps later chunk rows' stored keys out of
// row b's candidate set, so causality holds by construction AND is enforced
// again in the consumer kernel via s_kv_per_row, INV-SPARSE-CHUNK-CAUSAL).
//
// IndexShare: same full/shared reuse rule as the decode producer — a shared
// layer reuses the per-row top-k the most recent full layer produced under
// this step key. Storage: dispatcher-provisioned page rows (row b's slice
// covers [0, len_b)) or the executor arena (a blessed chunk is ONE sequence,
// so the arena is valid at any B — unlike decode's multi-sequence B>1).
//
// LOCAL indexer mode (TD-SPARSE-PREFILL-LOCAL-INDEXER): same shard-score →
// shard-top-k → cross-rank-merge structure as the decode producer, PER CHUNK
// ROW. Rank r scores only its LOCAL-compacted owned pages, bounded PER ROW at
// nb = owned_len(r, len_b): the appender already stored the WHOLE chunk's
// keys, so the shard holds keys at positions ≥ len_b — the per-row bound is
// required for EXACTNESS, not just causality (scoring the chunk's later keys
// could crowd true causal candidates out of the ≤ITK candidate list before
// the merge's causal cutoff; with the bound, every scored position is causal
// and the decode merge-exactness argument applies row-by-row). Row b's shard
// top-k lands as a CANDIDATE row (LOCAL slot indices + scores) in the packed
// send buffer; execute_attention then allgathers and merges per row
// (merge_local_indexer_candidates) into the same GLOBAL per-row top-k the
// replicated producer emits. Paged storage only (no arena — replicated
// shape); the IndexShare reuse blessing is deferred to post-merge.
bool DcpExecutor::produce_sparse_indices_prefill(
    compute::AttentionDevice* attn, int rank,
    const AttentionExecParams& params) {
    const int r = rank;
    const int B = params.batch_size;
    const int H = opts_.hidden_size;
    const int Q = opts_.q_lora_rank;
    const int NIH = opts_.index_n_heads;
    const int IHD = opts_.index_head_dim;
    const int ITK = opts_.index_topk;
    const int d_rope = opts_.qk_rope_head_dim;
    void* stream = attn_streams_[r];

    if (params.indexer_step_key == 0) return false;  // dispatcher blessing
    if (B < 1 || B > opts_.max_batch_size) return false;
    const int layer = params.layer_idx;
    const int total_layers = std::max(opts_.num_layers, 1);
    if (layer < 0 || layer >= total_layers) return false;

    // INV-KVS-POS: GLOBAL positions (RoPE); per-row host lengths (bounds).
    const int* seqlens = params.global_seqlens_k
        ? params.global_seqlens_k[r]
        : (params.seqlens_k ? params.seqlens_k[r] : nullptr);
    if (!seqlens) return false;
    const int* hseq = params.host_seqlens_k;
    if (!hseq) return false;

    // IndexShare reuse — same rule as the decode producer: a shared layer
    // reuses the per-row buffers the most recent full layer wrote under this
    // step key; otherwise it falls through and recomputes as if full (only
    // possible when it owns storage — layer 0; a storage-less shared layer
    // fails the resolution below → that layer's chunk stays dense).
    const bool is_full = opts_.indexer_full_layers.empty()
        || (layer < static_cast<int>(opts_.indexer_full_layers.size())
            && opts_.indexer_full_layers[layer]);
    const uint64_t step_key = params.indexer_step_key;
    // TD-PREFILL-SUPERCHUNK: reuse keys + persistent top-k rows are per
    // ROW-RANGE — sub-chunk k of a superchunk owns rows
    // [batch_row_offset, batch_row_offset+B) of sparse_indices_dev_ and its
    // own step key, so a shared layer reuses ITS sub-chunk's selection even
    // though other sub-chunks produced in between. Legacy prefill = offset 0.
    const uint32_t row_off = static_cast<uint32_t>(
        params.batch_row_offset > 0 ? params.batch_row_offset : 0);
    if (!is_full) {
        auto it = indexer_reuse_key_[r].find(row_off);
        if (it != indexer_reuse_key_[r].end() && it->second == step_key) {
            indexer_step_fresh_ = false;
            return true;
        }
    }

    // Storage per chunk row (same resolution as append_indexer_chunk, which
    // already stored this chunk's keys there): page rows when provisioned,
    // else the arena slot (single-sequence chunk).
    // TD-SPARSE-PREFILL-LOCAL-INDEXER: local mode requires the dispatcher-
    // provisioned page shape at the OWNERSHIP unit this executor was built
    // with (same rule as the decode producer) — the arena is replicated-
    // shape, so no arena fallback exists in local mode.
    const int PT = params.indexer_k_page_tokens;
    if (indexer_local_ && PT != opts_.indexer_k_page_tokens) return false;
    const int slot = (layer < static_cast<int>(indexer_layer_slot_.size()))
        ? indexer_layer_slot_[layer] : -1;
    auto& rows = indexer_page_rows_;  // preallocated [max_batch]
    for (int b = 0; b < B; ++b) {
        const int len_b = hseq[b];
        if (len_b < 1) return false;
        rows[b] = indexer_page_row(params, r, layer, b, len_b);
        if (!rows[b]) {
            if (indexer_local_) return false;  // paged-only (no arena)
            if (slot < 0) return false;  // no storage (shared layer)
            if (len_b > indexer_cache_tokens_) return false;  // arena ceiling
        }
    }

    const auto& w = *params.weights[rank];
    // Q-side weight gate (the K side was gated by append_indexer_chunk).
    if (!w.q_idx_b || !w.weights_proj) return false;

    // 1) indexer_q = wq_b · q_a_norm_latent → [B, NIH*IHD]; RoPE each row at
    //    its own position; Hadamard-rotate all NIH head rows (identical to
    //    decode producer steps 1/1b — per-row kernels generalize over B).
    if (w.q_idx_b_is_gguf) {
        route_gguf_gemm(attn, r, B, NIH * IHD, Q, q_compressed_[r], w.q_idx_b,
                        indexer_q_[r], w.q_idx_b_gguf_type, stream);
    } else {
        compute::StridedBatchedGemmBf16Params g{};
        g.m = NIH * IHD; g.n = B; g.k = Q;
        g.A = w.q_idx_b;        g.lda = Q;         g.strideA = 0;
        g.B = q_compressed_[r]; g.ldb = Q;         g.strideB = 0;
        g.C = indexer_q_[r];    g.ldc = NIH * IHD; g.strideC = 0;
        g.batch_count = 1;
        attn->batched_gemm_bf16(g, stream);
    }
    if (rope_cos_sin_[r] && d_rope > 0) {
        compute::RopeRotateParams rr{};
        rr.x = indexer_q_[r]; rr.seqlens_k = seqlens;
        rr.cos_sin = rope_cos_sin_[r];
        rr.num_tokens = B; rr.rows_per_token = NIH; rr.row_stride = IHD;
        rr.d_rope = d_rope; rr.max_pos = opts_.rope_max_pos;
        attn->rope_rotate(rr, stream);
    }
    attn->indexer_hadamard(indexer_q_[r], B * NIH, IHD, stream);

    // 2) score weights = indexer_proj · normed_hidden → [B, NIH], scaled by
    //    1/sqrt(IHD*NIH) into the F32 score_proj (decode producer step 4).
    {
        compute::StridedBatchedGemmBf16Params g{};
        g.m = NIH; g.n = B; g.k = H;
        g.A = w.weights_proj;    g.lda = H;   g.strideA = 0;
        g.B = normed_hidden_[r]; g.ldb = H;   g.strideB = 0;
        g.C = indexer_weights_[r]; g.ldc = NIH; g.strideC = 0;
        g.batch_count = 1;
        attn->batched_gemm_bf16(g, stream);
    }
    attn->indexer_scale_weights(
        indexer_weights_[r], indexer_score_proj_[r], B, NIH,
        1.0f / std::sqrt(static_cast<float>(IHD) * static_cast<float>(NIH)),
        stream);

    // 3) Per chunk row: score its causal prefix [0, len_b) and causal-top-k
    //    at query position len_b − 1 → row b outputs. Sequential launches on
    //    the attention stream — the shared scores scratch is safe to reuse.
    //    LOCAL MODE (TD-SPARSE-PREFILL-LOCAL-INDEXER): score only THIS
    //    RANK's owned shard, bounded PER ROW at nb = owned_len(r, len_b) —
    //    the appender already stored the WHOLE chunk's keys, so local slots
    //    ≥ nb hold positions ≥ len_b; the bound keeps them out of the ≤ITK
    //    candidate list (exactness — see the function comment) exactly as
    //    the replicated bound nb = len_b does. Row b's shard top-k goes to
    //    the packed candidate send buffer (LOCAL slot indices + scores);
    //    the caller allgathers + merges per row into the global top-k. A
    //    rank owning zero causal positions launches at num_blocks 0 → the
    //    valid empty candidate row (all −1).
    //    TD-SPARSE-PREFILL-SCORE-BATCH: batched path first — ONE score +
    //    ONE top-k launch per wave of rows, each row keeping its OWN bound
    //    and cutoff (INV-DSA-BATCH: bit-identical outputs to this loop; the
    //    kernels run the exact single-query device bodies per row). The
    //    per-row loop below is the authoritative fallback.
    if (prefill_score_topk_batched(attn, r, params, slot)) {
        if (!indexer_local_) indexer_reuse_key_[r][row_off] = step_key;
        indexer_step_fresh_ = true;
        return true;
    }
    for (int b = 0; b < B; ++b) {
        const int len_b = hseq[b];
        compute::IndexerScoreTopkArgs sa{};
        sa.q_all = static_cast<std::byte*>(indexer_q_[r])
            + static_cast<size_t>(b) * NIH * IHD * 2;        // BF16 row b
        sa.score_proj_all = static_cast<float*>(indexer_score_proj_[r])
            + static_cast<size_t>(b) * NIH;
        int nb = len_b;
        if (indexer_local_) {
            nb = daemon::kvshard::owned_len(
                r, static_cast<uint32_t>(len_b), PT, dcp_size_);
            sa.k_pages = rows[b];
            sa.num_k_pages = (nb + PT - 1) / PT;
            sa.page_tokens = PT;
            // Candidate outputs into the packed send buffer:
            // [B*ITK int32 local indices][B*ITK f32 scores], row b.
            int* cand = static_cast<int*>(indexer_cand_send_[r]);
            sa.sparse_indices_out = cand + static_cast<size_t>(b) * ITK;
            sa.topk_scores_scratch = reinterpret_cast<float*>(
                cand + static_cast<size_t>(B) * ITK)
                + static_cast<size_t>(b) * ITK;
            // Scratch only — the merge overwrites it with the final length.
            // (Offset by the sub-chunk row range so it never clobbers another
            // sub-chunk's persisted lengths, TD-PREFILL-SUPERCHUNK.)
            sa.topk_lengths_out = static_cast<int*>(topk_lengths_dev_[r])
                + row_off + b;
        } else if (rows[b]) {
            sa.k_pages = rows[b];
            sa.num_k_pages = (len_b + PT - 1) / PT;
            sa.page_tokens = PT;
        } else {
            sa.k_cache = static_cast<std::byte*>(indexer_k_cache_[r])
                + static_cast<size_t>(slot) * indexer_cache_tokens_ * IHD;
            sa.k_scales = static_cast<float*>(indexer_k_scales_[r])
                + static_cast<size_t>(slot) * indexer_cache_tokens_;
        }
        if (!indexer_local_) {
            sa.topk_scores_scratch = indexer_topk_scores_[r];
            // TD-PREFILL-SUPERCHUNK: persistent rows at the sub-chunk's
            // global row range (row_off = 0 for legacy prefill).
            sa.sparse_indices_out = static_cast<int*>(sparse_indices_dev_[r])
                + (static_cast<size_t>(row_off) + b) * ITK;
            sa.topk_lengths_out = static_cast<int*>(topk_lengths_dev_[r])
                + row_off + b;
        }
        sa.block_endpoints = indexer_block_endpoints_[r];  // static iota
        sa.scores_scratch = indexer_scores_[r];
        sa.num_tokens = 1; sa.num_blocks = nb;
        sa.n_heads = NIH; sa.head_dim = IHD; sa.topk = ITK;
        sa.query_position_base = len_b - 1;  // row b's causal cutoff
        attn->indexer_score_topk(sa, stream);
    }
    // Shared layers may reuse this chunk step's per-row result. Local mode
    // defers the reuse blessing to execute_attention (post-merge):
    // candidates alone are not a consumable top-k.
    if (!indexer_local_) indexer_reuse_key_[r][row_off] = step_key;
    indexer_step_fresh_ = true;
    return true;
}

// ── Batched sparse-prefill producer (TD-SPARSE-PREFILL-SCORE-BATCH) ─────────
//
// Replaces the per-row score+top-k loop of produce_sparse_indices_prefill
// with ONE batched score launch + ONE batched top-k launch per wave of chunk
// rows. Each row keeps its OWN causal block bound (nb = len_b replicated,
// owned_len(rank, len_b) local — the same values the per-row loop computes)
// and its own cutoff (len_b − 1); the bounds are staged host→device once per
// call as [B bounds | B cutoffs] int32. The batched kernels execute the
// exact single-query device bodies per (row, block) / per row CTA, so the
// selection is BIT-IDENTICAL to the per-row loop (INV-DSA-BATCH). Waves cap
// the per-row scores scratch: rows_per_wave = scratch_floats / max_row_bound
// (self-balancing — small prefixes, where launch overhead dominates, batch
// the whole chunk; huge prefixes, where kernel time dominates anyway, split
// into a few waves). Wave launches are stream-ordered on the rank's
// attention stream, so reusing the scores scratch across waves is safe —
// exactly like the retired sequential per-row reuse.
//
// Paged storage stages a row-major [B, pages_per_row] device page-pointer
// table (per-row page rows may differ under the dispatcher's batch-strided
// tables even for one sequence). Arena storage addresses the contiguous
// slot directly — identical to the single-query contiguous mode.
//
// Returns false → caller runs the authoritative per-row loop: B < 2 (no
// batching win), batched scratch unallocated (sparse_prefill off — cannot
// happen on this path — or allocation was skipped), per-row dense mask
// present (INV-DSA-ROWMIX defensive: the dispatcher never sets it on
// prefill steps; the per-row loop preserves exact legacy behavior if that
// ever changes), mixed paged/arena rows, a bound beyond the endpoints iota
// (indexer_cache_tokens_ — the same latent ceiling the per-row scratch
// has), page-table overflow, or rows_per_wave < 2.
bool DcpExecutor::prefill_score_topk_batched(
    compute::AttentionDevice* attn, int rank,
    const AttentionExecParams& params, int slot) {
    const int r = rank;
    const int B = params.batch_size;
    const int NIH = opts_.index_n_heads;
    const int IHD = opts_.index_head_dim;
    const int ITK = opts_.index_topk;
    const int PT = params.indexer_k_page_tokens;
    void* stream = attn_streams_[r];
    const int* hseq = params.host_seqlens_k;  // caller verified non-null
    // TD-PREFILL-SUPERCHUNK: global row range of this sub-chunk (0 = legacy).
    const uint32_t row_off = static_cast<uint32_t>(
        params.batch_row_offset > 0 ? params.batch_row_offset : 0);

    if (B < 2 || ITK < 1) return false;
    if (params.indexer_row_dense) return false;  // INV-DSA-ROWMIX (defensive)
    if (!indexer_scores_batched_[r] || !indexer_row_bounds_dev_[r])
        return false;
    if (static_cast<int>(indexer_row_bounds_host_.size()) < 2 * B)
        return false;

    // Uniform storage across rows (a blessed chunk is one sequence, so this
    // holds in practice; fall back on any mix).
    auto& rows = indexer_page_rows_;
    const bool paged = rows[0] != nullptr;
    for (int b = 1; b < B; ++b)
        if ((rows[b] != nullptr) != paged) return false;
    if (!paged && slot < 0) return false;   // caller-guaranteed; keep safe
    if (paged && PT <= 0) return false;

    // Per-row bounds + cutoffs — the same values the per-row loop passes.
    int nb_max = 0;
    for (int b = 0; b < B; ++b) {
        const int len_b = hseq[b];
        int nb = len_b;
        if (indexer_local_)
            nb = static_cast<int>(daemon::kvshard::owned_len(
                r, static_cast<uint32_t>(len_b), PT, dcp_size_));
        indexer_row_bounds_host_[static_cast<size_t>(b)] = nb;
        indexer_row_bounds_host_[static_cast<size_t>(B) + b] = len_b - 1;
        nb_max = std::max(nb_max, nb);
    }
    if (nb_max > indexer_cache_tokens_) return false;  // endpoints ceiling

    // Wave capacity from the batched scores scratch.
    int rpw = B;
    if (nb_max > 0)
        rpw = static_cast<int>(std::min<size_t>(
            static_cast<size_t>(B),
            indexer_scores_batched_floats_ / static_cast<size_t>(nb_max)));
    if (rpw < 2) return false;

    // Paged: stage the row-major per-row page-pointer table.
    int pages_per_row = 0;
    if (paged) {
        pages_per_row = (nb_max + PT - 1) / PT;
        if (!indexer_page_table_dev_[r] || pages_per_row < 1
            || static_cast<size_t>(B) * pages_per_row
                   > indexer_page_table_entries_)
            return false;
        indexer_page_table_host_.assign(
            static_cast<size_t>(B) * pages_per_row, nullptr);
        for (int b = 0; b < B; ++b) {
            const int np =
                (indexer_row_bounds_host_[static_cast<size_t>(b)] + PT - 1)
                / PT;
            for (int p = 0; p < np; ++p)
                indexer_page_table_host_[
                    static_cast<size_t>(b) * pages_per_row + p] = rows[b][p];
        }
        attn->memcpy_h2d_async(
            indexer_page_table_dev_[r], indexer_page_table_host_.data(),
            static_cast<size_t>(B) * pages_per_row * sizeof(void*), stream);
    }
    attn->memcpy_h2d_async(
        indexer_row_bounds_dev_[r], indexer_row_bounds_host_.data(),
        2 * static_cast<size_t>(B) * sizeof(int), stream);

    int* bounds_dev = static_cast<int*>(indexer_row_bounds_dev_[r]);
    for (int w0 = 0; w0 < B; w0 += rpw) {
        const int wave = std::min(rpw, B - w0);
        int wave_nb_max = 0;
        for (int b = w0; b < w0 + wave; ++b)
            wave_nb_max = std::max(
                wave_nb_max, indexer_row_bounds_host_[static_cast<size_t>(b)]);

        compute::IndexerScoreTopkBatchedArgs ba{};
        ba.q_all = static_cast<std::byte*>(indexer_q_[r])
            + static_cast<size_t>(w0) * NIH * IHD * 2;       // BF16 rows
        ba.score_proj_all = static_cast<float*>(indexer_score_proj_[r])
            + static_cast<size_t>(w0) * NIH;
        ba.row_num_blocks = bounds_dev + w0;
        ba.row_query_position = bounds_dev + B + w0;
        if (paged) {
            ba.k_page_table = static_cast<const void* const*>(
                indexer_page_table_dev_[r])
                + static_cast<size_t>(w0) * pages_per_row;
            ba.page_table_stride = pages_per_row;
            ba.page_tokens = PT;
        } else {
            ba.k_cache = static_cast<std::byte*>(indexer_k_cache_[r])
                + static_cast<size_t>(slot) * indexer_cache_tokens_ * IHD;
            ba.k_scales = static_cast<float*>(indexer_k_scales_[r])
                + static_cast<size_t>(slot) * indexer_cache_tokens_;
        }
        ba.block_endpoints = indexer_block_endpoints_[r];
        ba.scores_scratch = indexer_scores_batched_[r];
        ba.scores_stride = nb_max;
        if (indexer_local_) {
            // Candidate rows into the packed send buffer, exactly as the
            // per-row loop: [B*ITK int32 local indices][B*ITK f32 scores].
            int* cand = static_cast<int*>(indexer_cand_send_[r]);
            ba.sparse_indices_out = cand + static_cast<size_t>(w0) * ITK;
            ba.topk_scores_out = reinterpret_cast<float*>(
                cand + static_cast<size_t>(B) * ITK)
                + static_cast<size_t>(w0) * ITK;
        } else {
            // TD-PREFILL-SUPERCHUNK: persistent rows at the sub-chunk's global
            // row range (row_off = 0 for legacy prefill). topk_scores is a
            // per-launch scratch — wave-local rows suffice.
            ba.sparse_indices_out = static_cast<int*>(sparse_indices_dev_[r])
                + (static_cast<size_t>(row_off) + w0) * ITK;
            ba.topk_scores_out = static_cast<float*>(indexer_topk_scores_[r])
                + static_cast<size_t>(w0) * ITK;
        }
        ba.topk_lengths_out = static_cast<int*>(topk_lengths_dev_[r])
            + row_off + w0;
        ba.num_rows = wave;
        ba.max_num_blocks = wave_nb_max;
        ba.n_heads = NIH;
        ba.head_dim = IHD;
        ba.topk = ITK;
        attn->indexer_score_topk_batched(ba, stream);
    }

    if (!indexer_batch_logged_) {
        indexer_batch_logged_ = true;
        spdlog::info("DcpExecutor: sparse-prefill BATCHED indexer score+topk "
                     "ACTIVE (B={}, rows/wave={}, waves={}, paged={}, "
                     "local={})",
                     B, rpw, (B + rpw - 1) / rpw, paged, indexer_local_);
    }
    return true;
}

// ── Local-indexer cross-rank top-k merge (TD-GLM-INDEXER-LOCAL-MERGE) ───────
//
// Shared by the decode producer and the sparse-prefill producer
// (TD-SPARSE-PREFILL-LOCAL-INDEXER): after every rank's producer emitted its
// SHARD candidates (LOCAL slot indices + scores) into the packed send
// buffers, allgather them and run the exact cross-rank merge on every rank,
// PER ROW (row b's causal bound/cutoff from its own host length), into
// sparse_indices_dev_/topk_lengths_dev_ — the same buffers the replicated
// producer writes, so everything downstream (IndexShare reuse, the KVS-4
// sharded-KV translation, the sparse consumers) is mode-agnostic.
// Deterministic identical inputs per rank → identical merged output. The
// IndexShare reuse key is blessed HERE, post-merge: the reused buffers must
// hold a MERGED global top-k, never raw candidates.
void DcpExecutor::merge_local_indexer_candidates(
    const AttentionExecParams& params) {
    const int ITK = opts_.index_topk;
    const int B = params.batch_size;
    // TD-PREFILL-SUPERCHUNK: merged output rows land at the sub-chunk's global
    // row range (candidate send/recv buffers stay launch-local, rows [0, B)).
    const uint32_t row_off = static_cast<uint32_t>(
        params.batch_row_offset > 0 ? params.batch_row_offset : 0);
    const size_t cand_words = 2 * static_cast<size_t>(B) * ITK;
    std::vector<const void*> sends(dcp_size_);
    std::vector<void*> recvs(dcp_size_);
    for (int r = 0; r < dcp_size_; ++r) {
        sends[r] = indexer_cand_send_[r];
        recvs[r] = indexer_cand_recv_[r];
    }
    opts_.communicator->allgather_indexer_candidates(
        sends.data(), recvs.data(), cand_words, attn_streams_.data());
    for (int r = 0; r < dcp_size_; ++r) {
        auto* attn = opts_.attention_devices[r];
        attn->set_device();
        for (int b = 0; b < B; ++b) {
            // INV-DSA-ROWMIX: a masked dense row has no candidates (the
            // producer skipped it and zeroed its top-k length) — merging
            // would overwrite that zero with garbage-derived output.
            if (params.indexer_row_dense && params.indexer_row_dense[b])
                continue;
            const int len_b = params.host_seqlens_k
                ? params.host_seqlens_k[b] : params.max_seqlen_k;
            compute::IndexerTopkMergeArgs ma{};
            ma.gathered = indexer_cand_recv_[r];
            ma.seg_words = static_cast<int>(cand_words);
            ma.batch = B;
            ma.token = b;
            ma.scores_scratch = indexer_scores_[r];
            ma.block_endpoints = indexer_block_endpoints_[r];
            ma.topk_scores_scratch = indexer_topk_scores_[r];
            ma.indices_out = static_cast<int*>(sparse_indices_dev_[r])
                + (static_cast<size_t>(row_off) + b) * ITK;
            ma.length_out = static_cast<int*>(topk_lengths_dev_[r])
                + row_off + b;
            ma.num_blocks = len_b;
            ma.topk = ITK;
            ma.query_position = len_b - 1;
            ma.dcp_size = dcp_size_;
            ma.page_tokens = opts_.indexer_k_page_tokens;
            attn->indexer_topk_merge(ma, attn_streams_[r]);
        }
        // Bless IndexShare reuse only now: the reused buffers must hold a
        // MERGED global top-k, never raw candidates.
        indexer_reuse_key_[r][row_off] = params.indexer_step_key;
    }
    if (!indexer_merge_logged_) {
        indexer_merge_logged_ = true;
        spdlog::info("DcpExecutor: local-indexer cross-rank top-k merge "
                     "ACTIVE (dcp={}, B={}, layer={}, page_tokens={})",
                     dcp_size_, B, params.layer_idx,
                     opts_.indexer_k_page_tokens);
    }
}

// ── o_proj GEMM + TP allreduce (steps 13-14) ────────────────────────────────

void DcpExecutor::execute_oproj_and_reduce(
    const AttentionExecParams& params, void* const* corrected_outputs,
    int in_ld_heads) {

    const int B = params.batch_size;
    const int H = opts_.hidden_size;
    const int HL = num_heads_local_;
    // V = v_head_dim: o_proj weight is [H, HL * v_head_dim] (non-absorbed, INV-MLA-1).
    // The kv_b_v projection converts [B, HL, kv_lora_rank] → [B, HL, V] before this.
    const int V = opts_.v_head_dim;
    const int D_c = opts_.kv_lora_rank;

    // ── Step 12b: kv_b_v projection [B, HL, D_c] → [B, HL, V] ──
    //
    // Per-head batched GEMM using the V portion of kv_b_proj weight.
    // For FP8 weights: acquires dequanted BF16 from the pool.
    // For BF16 weights: reads directly from pinned kv_b_proj (no copy).
    if (!dequant_pool_) {
        throw std::runtime_error(
            "DcpExecutor::execute_oproj_and_reduce: dequant_pool_ is null — "
            "call set_layer_weights() before execute_attention()");
    }
    for (int r = 0; r < dcp_size_; ++r) {
        auto* attn = opts_.attention_devices[r];
        attn->set_device();
        void* stream = attn_streams_[r];
        const auto& w = *params.weights[r];

        auto acq = dequant_pool_->acquire(params.layer_idx, r, w, stream);

        compute::StridedBatchedGemmBf16Params bp{};
        bp.m = V;
        bp.n = B;
        bp.k = D_c;
        bp.A = acq.weight_ptr;
        bp.lda = D_c;           // row-major [V, D_c] → leading dim = D_c
        bp.strideA = acq.stride_a;
        bp.B = corrected_outputs[r];
        // [B, in_ld_heads, D_c] interleaved: per-token leading dim is
        // in_ld_heads * D_c. Legacy layout: in_ld_heads == HL. Sharded KV
        // (INV-KVS-QAG): corrected_outputs[r] points at rank r's HL-head
        // slice inside the all-head combined buffer, in_ld_heads == dcp*HL.
        bp.ldb = in_ld_heads * D_c;
        bp.strideB = D_c;      // stride between heads in B
        bp.C = kv_bv_out_[r];
        bp.ldc = HL * V;       // [B, HL, V] interleaved → ldc = HL * V
        bp.strideC = V;        // stride between heads in C
        bp.batch_count = HL;

        attn->batched_gemm_bf16(bp, stream);
    }

    // ── Step 13: o_proj GEMM + Step 14: TP allreduce ──
    // Input is now kv_bv_out_ [B, HL, V] instead of corrected_outputs [B, HL, D_c].
    for (int r = 0; r < dcp_size_; ++r) {
        auto* attn = opts_.attention_devices[r];
        attn->set_device();
        void* stream = attn_streams_[r];
        const auto& w = *params.weights[r];

        if (w.o_proj_is_gguf) {
            // ── GGUF path: BF16 activation (kv_bv_out_) straight into the GGUF
            // GEMM (mmvq/mmq/dequant). o_proj weight is packed [H, HL*V]. ──
            route_gguf_gemm(attn, r, B, H, HL * V,
                            kv_bv_out_[r], w.o_proj, hidden_out_[r],
                            w.o_proj_gguf_type, stream);
        } else if (w.o_proj_is_nvfp4) {
            // ── NVFP4 path: BF16 → FP4 activation quant, then NVFP4 grouped GEMM ──

            // Per-layer meta slot, uploaded lazily once per (layer, B): the
            // contents (offsets/problem_sizes/sf_offsets/alpha/input_scale)
            // only depend on the layer's weights and B, so the decode steady
            // state issues ZERO meta H2Ds (this block used to be 5 sync
            // cudaMemcpys per layer per rank per token).
            int slot = params.layer_idx;
            bool force_upload = false;
            if (slot < 0 || slot >= total_layers_alloc_) {
                slot = 0;          // shared fallback slot — always refresh
                force_upload = true;
            }
            auto* meta = static_cast<char*>(nvfp4_oproj_meta_[r])
                       + static_cast<size_t>(slot) * kOprojMetaStride;
            if (force_upload || oproj_meta_resident_b_[r][slot] != B) {
                const int sf_padded = ((B + 127) / 128) * 128;
                // Layout: expert_offsets{0,B} | problem_sizes{B,H,HL*V} |
                //         sf_offsets{0,sf_padded} | alpha | input_scale.
                const int32_t ints[7] = {0, B, B, H, HL * V, 0, sf_padded};
                // FP4-ACT-SCALE: the calibrated input_scale the quantizer
                // divides by; alpha already multiplies it back.
                const float scalars[2] = {
                    w.o_proj_nvfp4_alpha,
                    (w.o_proj_nvfp4_input_scale > 0.f)
                        ? w.o_proj_nvfp4_input_scale : 1.0f};
                unsigned char* hb = oproj_meta_host_[r].data()
                                  + static_cast<size_t>(slot) * kOprojMetaStride;
                std::memcpy(hb, ints, sizeof(ints));
                std::memcpy(hb + sizeof(ints), scalars, sizeof(scalars));
                // Stream-ordered: the quant kernel + GEMM below read meta on
                // the same stream. hb is persistent staging (async-safe).
                attn->memcpy_h2d_async(meta, hb, sizeof(ints) + sizeof(scalars),
                                       stream);
                oproj_meta_resident_b_[r][slot] = force_upload ? -1 : B;
            }

            compute::Bf16ToNvfp4GroupedParams qp{};
            qp.input          = kv_bv_out_[r];
            qp.output_packed  = nvfp4_oproj_act_[r];
            qp.output_scales  = nvfp4_oproj_scales_[r];
            qp.expert_offsets = meta;
            qp.sf_offsets     = meta + 5 * sizeof(int32_t);
            qp.total_tokens   = B;
            qp.num_experts    = 1;
            qp.K              = HL * V;
            qp.input_scales   = meta + 7 * sizeof(int32_t) + sizeof(float);
            attn->bf16_to_nvfp4_grouped(qp, stream);

            compute::Nvfp4GroupedGemmParams gp{};
            gp.num_experts    = 1;
            gp.N              = H;
            gp.K              = HL * V;
            gp.A_base         = nvfp4_oproj_act_[r];
            gp.B_base         = w.o_proj;
            gp.D_base         = hidden_out_[r];
            gp.scale_A_base   = nvfp4_oproj_scales_[r];
            gp.scale_B_base   = w.o_proj_scales;
            gp.alphas         = reinterpret_cast<const float*>(
                                    meta + 7 * sizeof(int32_t));
            gp.expert_offsets = static_cast<const int32_t*>(
                                    static_cast<void*>(meta));
            gp.sf_offsets     = reinterpret_cast<const int32_t*>(
                                    meta + 5 * sizeof(int32_t));
            gp.problem_sizes  = reinterpret_cast<const int32_t*>(
                                    meta + 2 * sizeof(int32_t));
            gp.output_dtype   = compute::GemmOutputDtype::kBFloat16;
            attn->nvfp4_grouped_gemm(gp, gemm_workspace_[r],
                                     nvfp4_oproj_gemm_ws_bytes_, stream);
        } else {
            // ── FP8 path ──
            attn->quantize_fp8(
                {.num_tokens = B, .hidden_size = HL * V,
                 .input = kv_bv_out_[r],
                 .output = fp8_corrected_[r],
                 .scales = fp8_corrected_scales_[r],
                 .m_major_scales = true}, stream);

            attn->gemm(
                {.M = B, .N = H, .K = HL * V,
                 .A = fp8_corrected_[r], .B = w.o_proj, .D = hidden_out_[r],
                 .scale_A = fp8_corrected_scales_[r], .scale_B = w.o_proj_scales,
                 .output_dtype = compute::GemmOutputDtype::kBFloat16},
                gemm_workspace_[r], stream);
        }
    }

    // Release the dequant pool slot for this layer.
    if (dequant_pool_) dequant_pool_->release(params.layer_idx);

    // Step 14: TP allreduce (separate per INV-DCP-8)
    if (opts_.dcp_wrapper) {
        // INV-NCCL-GRAPH (env LS_NCCL_GRAPH, default OFF): decode (B==1)
        // replays a captured per-rank graph of this fixed-buffer allreduce —
        // hidden_out_ addresses and count (hidden_size) are invariant across
        // layers and steps, so ONE capture serves every layer. Removes the
        // eager NCCL enqueue host cost + launch stagger. Fails open to the
        // eager reduce_hidden on any capture failure.
        static const bool nccl_graph = [] {
            const char* e = std::getenv("LS_NCCL_GRAPH");
            return e && e[0] && e[0] != '0';
        }();
        bool replayed = false;
        if (nccl_graph && B == 1 && dcp_size_ >= 2 && opts_.communicator
            && !oproj_reduce_graph_failed_) {
            if (!oproj_reduce_graph_) {
                std::vector<int> ids(dcp_size_);
                std::vector<void*> comms(dcp_size_), strms(dcp_size_);
                std::vector<std::vector<compute::NcclGroupGraphRunner::Op>>
                    ops(dcp_size_);
                bool have_all = true;
                for (int r = 0; r < dcp_size_; ++r) {
                    ids[r] = opts_.gpus[r].id;
                    comms[r] = opts_.communicator->comm(r);
                    strms[r] = attn_streams_[r];
                    if (!comms[r] || !strms[r] || !hidden_out_[r])
                        have_all = false;
                    ops[r] = {{hidden_out_[r],
                               static_cast<size_t>(opts_.hidden_size),
                               /*fp32=*/false}};
                }
                if (have_all) {
                    oproj_reduce_graph_ =
                        std::make_unique<compute::NcclGroupGraphRunner>();
                    if (oproj_reduce_graph_->init(ids, comms, strms, ops)) {
                        spdlog::warn("INV-NCCL-GRAPH: captured o_proj TP "
                                     "reduce graphs ({} ranks)", dcp_size_);
                    } else {
                        oproj_reduce_graph_.reset();
                        oproj_reduce_graph_failed_ = true;
                    }
                } else {
                    oproj_reduce_graph_failed_ = true;
                }
            }
            if (oproj_reduce_graph_ && oproj_reduce_graph_->is_captured()) {
                oproj_reduce_graph_->replay(attn_streams_);
                replayed = true;
            }
        }
        if (!replayed) {
            opts_.dcp_wrapper->reduce_hidden(
                hidden_out_.data(), B, attn_streams_.data());
        }
    }
}

// ── Graph-mode execution (decode) ───────────────────────────────────────────

// TODO:DEBT TD-35a: Graph path dereferences seqlens_k/block_tables without null check
void DcpExecutor::execute_attention_graph(const AttentionExecParams& params) {
    const int B = params.batch_size;

    if (!opts_.graph_registry) return;

    // Per-rank: update + replay DecodeGraphRunner (steps 7-9)
    std::vector<void*> corrected_ptrs(dcp_size_);

    for (int r = 0; r < dcp_size_; ++r) {
        auto* attn = opts_.attention_devices[r];
        attn->set_device();
        void* stream = attn_streams_[r];

        compute::GraphKey decode_key{compute::GraphType::kAttentionDecode,
                                      opts_.gpus[r].position, B};
        auto* decode_entry = opts_.graph_registry->find(decode_key);
        if (!decode_entry) continue;

        attn->decode_graph_update(
            *decode_entry, q_absorbed_[r],
            params.seqlens_k[r], params.block_tables[r],
            params.sparse_indices ? params.sparse_indices[r] : nullptr,
            params.layer_idx, stream);
        attn->decode_graph_replay(*decode_entry, stream);
        corrected_ptrs[r] = attn->decode_graph_out_ptr(*decode_entry);
    }

    // Steps 10-12: DcpAllreduceGraphRunner — KV-SHARDED mode only
    // (INV-DCP-KVREP, same gate as the nongraph correct_output).
    // UNREACHABLE under sharded KV today: execute_attention forces the
    // nongraph path (TD-KVS-QAG-GRAPH) and capture_dcp_graphs no longer
    // captures kDcpAllreduce entries, so find() below always misses. Kept
    // as the replay seam for the future QAG graph capture.
    if (dcp_size_ >= 2 && opts_.dcp_kv_sharded) {
        for (int r = 0; r < dcp_size_; ++r) {
            auto* attn = opts_.attention_devices[r];
            attn->set_device();

            compute::GraphKey dcp_key{compute::GraphType::kDcpAllreduce,
                                       opts_.gpus[r].position, B};
            auto* dcp_entry = opts_.graph_registry->find(dcp_key);
            if (!dcp_entry) continue;

            attn->dcp_graph_replay(*dcp_entry, attn_streams_[r]);
        }
    }

    // Steps 13-14: o_proj + TP allreduce
    execute_oproj_and_reduce(params, corrected_ptrs.data(),
                              /*in_ld_heads=*/num_heads_local_);
}

// ── Non-graph-mode execution (prefill / variable batch) ─────────────────────

void DcpExecutor::execute_attention_nongraph(const AttentionExecParams& params) {
    const int B = params.batch_size;

    std::vector<void*> partial_outputs(dcp_size_);
    std::vector<const float*> partial_lses(dcp_size_);

    // ── Step 7b (INV-KVS-QAG, sharded KV only): allgather Q in the HEAD dim ──
    //
    // Rank r's q_absorbed_ holds ONLY its own HL-head shard (q_b_proj is
    // TP-sharded). Under sharded KV each rank also holds only a token shard,
    // so without this gather NO rank would compute head h over another rank's
    // tokens — the combine's cross terms would be absent and the per-head LSE
    // merge would mix DIFFERENT heads (TD-KVS-Q-ALLGATHER). Gathering
    // q_absorbed across ranks in the head dim makes every rank attend ALL
    // dcp*HL heads over its LOCAL shard (vLLM mla_attention.py:802-830); the
    // combine below is then a genuine same-head merge over disjoint token
    // shards. Per-rank attention FLOPs are unchanged: H×T/dcp vs HL×T.
    //
    // Layout: NCCL concatenates rank-major → stage[r] = [dcp, B, HL, KV] with
    // chunk s = rank s's global heads [s*HL, (s+1)*HL). At B == 1 that IS the
    // head-major [1, dcp*HL, KV] attention layout; at B > 1 a per-source-rank
    // strided copy rearranges token-major into q_gathered_[r].
    const bool qag = opts_.dcp_kv_sharded && dcp_size_ >= 2;
    std::vector<const void*> q_attn(dcp_size_);
    for (int r = 0; r < dcp_size_; ++r) q_attn[r] = q_absorbed_[r];
    if (qag) {
        const int HL = num_heads_local_;
        const int KV = opts_.kv_lora_rank + opts_.qk_rope_head_dim;
        const size_t per_rank_elems =
            static_cast<size_t>(B) * HL * KV;      // BF16 elements
        opts_.communicator->allgather_q(
            const_cast<const void* const*>(q_absorbed_.data()),
            q_gathered_stage_.data(), B, per_rank_elems,
            attn_streams_.data());
        if (B == 1) {
            for (int r = 0; r < dcp_size_; ++r)
                q_attn[r] = q_gathered_stage_[r];
        } else {
            const size_t head_row = static_cast<size_t>(HL) * KV * 2;  // bytes
            for (int r = 0; r < dcp_size_; ++r) {
                auto* attn = opts_.attention_devices[r];
                attn->set_device();
                for (int s = 0; s < dcp_size_; ++s) {
                    // stage chunk s = [B, HL, KV] → rows of q_gathered_[r]'s
                    // [B, dcp*HL, KV] at head offset s*HL.
                    attn->memcpy_2d_d2d_async(
                        static_cast<char*>(q_gathered_[r]) + s * head_row,
                        head_row * dcp_size_,
                        static_cast<const char*>(q_gathered_stage_[r])
                            + s * head_row * B,
                        head_row, head_row, static_cast<size_t>(B),
                        attn_streams_[r]);
                }
                q_attn[r] = q_gathered_[r];
            }
        }
    }

    for (int r = 0; r < dcp_size_; ++r) {
        auto* attn = opts_.attention_devices[r];
        attn->set_device();
        void* stream = attn_streams_[r];

        // KV length for the gather/dequant/attend: the max cached length across
        // the batch (host-side, from the batch descriptor). The old `= B`
        // placeholder truncated decode attention to B tokens — the query never
        // saw its cached history (caught by GreedyContinuationGolden).
        //
        // KVS-3 (sharded KV): the bound is rank-LOCAL — rank r's block table
        // covers only its own token shard, so staging/attending to the GLOBAL
        // max would read past the local page list (zeroed entries → physical
        // page 0 garbage). host_local_seqlens_k[r] carries the per-row local
        // lengths; the per-rank bound is their max. May be 0 (empty shard:
        // every sequence shorter than this rank's first chunk) — the kernels
        // then emit zero output + lse=+inf and the DCP combine weights this
        // rank's contribution to 0 (INV-KVS-EMPTY).
        int seq_len_kv = params.max_seqlen_k > 0 ? params.max_seqlen_k : B;
        if (params.host_local_seqlens_k && params.host_local_seqlens_k[r]) {
            int local_max = 0;
            for (int b = 0; b < B; ++b)
                local_max = std::max(local_max,
                                     params.host_local_seqlens_k[r][b]);
            seq_len_kv = local_max;
        }

        // TD-PREFILL-CHUNK-ATTN: a multi-token prefill CHUNK (B query rows,
        // one per prompt position of ONE sequence) runs as a SINGLE batched
        // CAUSAL call: the device stages the union KV prefix [0, seq_len_kv)
        // once and the dense kernel masks row b to [0, seqlens_k[b]) per CTA.
        // KVS-3: under sharded KV seqlens_k[r] holds the rank-LOCAL per-row
        // shard lengths, so the per-row causal bound is automatically over
        // the rank's OWN shard (local ordering is global-ascending), and
        // seq_len_kv above is the local union bound.
        // (host_seqlens_k[b] = chunk_start + b + 1, ascending, so row B-1's
        // block-table row covers the union — the chunk_causal contract in
        // attention_device.h). Bit-equal to the retired per-row loop (row b
        // attends exactly [0, len_b), causal by construction) at O(seq_len_kv)
        // staging instead of O(B·seq_len_kv). The plain batched call remains
        // only valid at B==1 for prefill: the dense kernel's flat mask is
        // non-causal, and at B>1 multi-sequence (nongraph decode) the
        // linearize/dequant staging is also wrong (TD-DECODE-NONGRAPH-BATCH).
        // INV-KVS-QAG: under sharded KV the query is the head-gathered
        // [B, dcp*HL, KV] buffer and the device's h_q is dcp*HL (engine sets
        // it to the FULL head count when dcp_kv_mode=sharded); under
        // replication it is the rank's own [B, HL, KV] shard with h_q = HL.
        const bool chunk_rows =
            params.chunk_len > 0 && B > 1 && params.host_seqlens_k;

        // GLM-25k: tiered sparse consumption. Materialize the ≤index_topk
        // selected rows into the manager's dense fake-paged scratch (hot D2D
        // + cold staged H2D) and run the UNMODIFIED sparse attention over it
        // with IDENTITY indices — topk_lengths unchanged, kernel-change-free,
        // bit-identical to the full-residency path (INV-KVT-1). false =
        // no cold pages in this layer → original path below is valid.
        // TD-KVT-PREFILL: a blessed B==1 SPARSE prefill chunk takes this
        // same branch (chunk_rows is false at B==1; its per-row causal
        // top-k is a decode-shaped single-row selection) — only the
        // selected rows are staged, never the full union prefix, so pages
        // behind the chunk frontier may be cold during long prefill.
        // INV-DSA-ROWMIX (TD-GLM-INDEXER-B1CASCADE resolved): a MIXED
        // sparse/dense decode cohort is executed as B PER-ROW batch-of-1
        // sub-dispatches — sparse for live rows, dense for coverage-dead
        // rows — each writing into its own row slice of the batch output
        // (scatter by pointer offset; rows are laid out exactly as one
        // batched call would write them, so the DCP combine and o_proj
        // downstream are untouched). A batch-of-1 call is the ONLY correct
        // multi-sequence shape here: the flat batched call stages ONE
        // sequence's prefix for all rows (TD-DECODE-NONGRAPH-BATCH — dense
        // AND sparse), while a per-row call linearizes/dequants/attends the
        // row's OWN prefix — bit-identical to running that sequence
        // separately. Scratch reuse across the sub-calls is safe: all
        // launches are stream-ordered on this rank's attention stream.
        const bool mixed_rows = params.indexer_row_dense && B > 1
            && !chunk_rows && params.is_sparse;

        parallelism::TieredKvView tv{};
        bool tiered = false;
        if (params.kv_tiering && params.is_sparse && !chunk_rows
            && !mixed_rows
            && params.sparse_indices && params.topk_lengths) {
            tiered = params.kv_tiering->materialize(
                r, params.layer_idx, params.sparse_indices[r],
                params.topk_lengths[r], B, stream, &tv);
        }
        // TD-KVT-ADMISSION-UPFRONT: a blessed SPARSE prefill chunk on a tier
        // step (the dispatcher stages kv_tiering only when the tiered-
        // prefill mode blessed this chunk) consumes PER ROW as B==1 sparse
        // sub-dispatches — the INV-DSA-ROWMIX shape: row b attends exactly
        // its own causal top-k (INV-SPARSE-CHUNK-CAUSAL), through the real
        // block tables while the layer is fully resident, or through the
        // manager's B==1 fake view when it has cold pages (INV-KVT-13
        // extended to chunk cohorts; placement-only, INV-KVT-1 — bit-
        // identical to the same per-row consumption with everything hot).
        // Row outputs land in their batch row slices, so the downstream
        // DCP combine / o_proj are untouched (mixed-rows precedent).
        // HYBRID gate: per-row consumption only when the layer actually
        // holds cold pages (or LS_KVT_COHORT_ALWAYS=1 forces the strict
        // per-row arm) — an all-hot layer keeps the batched sparse chunk
        // kernel below (its full-local-prefix linearize is legal, and the
        // per-row shape costs ~decode-class attention per row).
        const bool tiered_chunk = chunk_rows && params.kv_tiering
            && params.is_sparse
            && params.sparse_indices && params.topk_lengths
            && params.kv_tiering->cohort_layer_tiered(params.layer_idx);
        // TD-KVT-COHORT-BATCHED-MATERIALIZE: batched union consumer — the
        // manager materializes the UNION of the cohort's selections once
        // and rewrites each row's indices to union slots (order-preserving),
        // so the ONE batched sparse chunk-causal kernel runs over the fake
        // union view exactly like the resident-layer batched call below:
        // per-row topk_lengths unchanged, per-row seqlens all = the union
        // extent (constant satisfies the chunk_causal ascending contract;
        // the causal bound never bites — the producer's selection is already
        // causal, INV-SPARSE-CHUNK-CAUSAL).  False (no cold pages / arm
        // disabled via LS_KVT_COHORT_ROWWISE=1 / empty or over-capacity
        // union) falls back to the per-row loop below — always correct.
        bool tiered_union = false;
        parallelism::TieredKvView utv{};
        if (tiered_chunk) {
            tiered_union = params.kv_tiering->materialize_cohort(
                r, params.layer_idx, B, params.sparse_indices[r],
                params.topk_lengths[r], indexer_step_fresh_, stream, &utv);
        }
        if (tiered_union) {
            attn->prefill_attention(
                q_attn[r], B, utv.seq_len_kv,
                utv.seqlens_k, utv.block_tables, utv.max_blocks_per_seq,
                utv.kv_cache,
                params.cache_stride_block, params.cache_stride_row,
                params.page_size,
                /*is_sparse=*/true, /*chunk_causal=*/true,
                utv.sparse_indices, params.topk_lengths[r],
                opts_.index_topk,
                prefill_out_[r], prefill_lse_[r],
                params.layer_idx, stream);
        } else if (tiered_chunk) {
            const int KV = opts_.kv_lora_rank + opts_.qk_rope_head_dim;
            const size_t q_row_b   =
                static_cast<size_t>(attn_num_heads_) * KV * 2;   // BF16
            const size_t out_row_b =
                static_cast<size_t>(attn_num_heads_) * opts_.kv_lora_rank * 2;
            for (int b = 0; b < B; ++b) {
                // Per-row KV bound: the row's own (rank-local under sharded
                // KV, KVS-3) cached length — what a stand-alone B==1 step
                // of this position would use.
                int skv_b = seq_len_kv;
                if (params.host_local_seqlens_k
                    && params.host_local_seqlens_k[r])
                    skv_b = params.host_local_seqlens_k[r][b];
                else if (params.host_seqlens_k)
                    skv_b = params.host_seqlens_k[b];
                parallelism::TieredKvView rtv{};
                const bool rt = params.kv_tiering->materialize_row(
                    r, params.layer_idx, b, B, params.sparse_indices[r],
                    params.topk_lengths[r], indexer_step_fresh_, stream,
                    &rtv);
                attn->prefill_attention(
                    static_cast<const char*>(q_attn[r]) + b * q_row_b,
                    /*batch_size=*/1,
                    rt ? rtv.seq_len_kv : skv_b,
                    rt ? rtv.seqlens_k
                       : (params.seqlens_k ? params.seqlens_k[r] + b
                                           : nullptr),
                    rt ? rtv.block_tables
                       : (params.block_tables
                              ? params.block_tables[r]
                                    + static_cast<size_t>(b)
                                          * params.max_blocks_per_seq
                              : nullptr),
                    rt ? rtv.max_blocks_per_seq : params.max_blocks_per_seq,
                    rt ? rtv.kv_cache
                       : (params.kv_cache_ptrs ? params.kv_cache_ptrs[r]
                                               : nullptr),
                    params.cache_stride_block, params.cache_stride_row,
                    params.page_size,
                    /*is_sparse=*/true, /*chunk_causal=*/false,
                    rt ? rtv.sparse_indices
                       : params.sparse_indices[r]
                             + static_cast<size_t>(b) * opts_.index_topk,
                    params.topk_lengths[r] + b,
                    opts_.index_topk,
                    static_cast<char*>(prefill_out_[r]) + b * out_row_b,
                    prefill_lse_[r]
                        + static_cast<size_t>(b) * attn_num_heads_,
                    params.layer_idx, stream);
            }
        } else if (mixed_rows) {
            const int KV = opts_.kv_lora_rank + opts_.qk_rope_head_dim;
            const size_t q_row_b   =
                static_cast<size_t>(attn_num_heads_) * KV * 2;   // BF16
            const size_t out_row_b =
                static_cast<size_t>(attn_num_heads_) * opts_.kv_lora_rank * 2;
            for (int b = 0; b < B; ++b) {
                const bool row_sparse = params.indexer_row_dense[b] == 0;
                // Per-row KV bound: exactly what a stand-alone B==1 step of
                // this sequence would use — its own (rank-local under
                // sharded KV, KVS-3) cached length.
                int skv_b = seq_len_kv;
                if (params.host_local_seqlens_k
                    && params.host_local_seqlens_k[r])
                    skv_b = params.host_local_seqlens_k[r][b];
                else if (params.host_seqlens_k)
                    skv_b = params.host_seqlens_k[b];
                attn->prefill_attention(
                    static_cast<const char*>(q_attn[r]) + b * q_row_b,
                    /*batch_size=*/1, skv_b,
                    params.seqlens_k ? params.seqlens_k[r] + b : nullptr,
                    params.block_tables
                        ? params.block_tables[r]
                              + static_cast<size_t>(b)
                                    * params.max_blocks_per_seq
                        : nullptr,
                    params.max_blocks_per_seq,
                    params.kv_cache_ptrs ? params.kv_cache_ptrs[r] : nullptr,
                    params.cache_stride_block, params.cache_stride_row,
                    params.page_size,
                    /*is_sparse=*/row_sparse, /*chunk_causal=*/false,
                    row_sparse && params.sparse_indices
                        ? params.sparse_indices[r]
                              + static_cast<size_t>(b) * opts_.index_topk
                        : nullptr,
                    row_sparse && params.topk_lengths
                        ? params.topk_lengths[r] + b : nullptr,
                    opts_.index_topk,
                    static_cast<char*>(prefill_out_[r]) + b * out_row_b,
                    prefill_lse_[r]
                        + static_cast<size_t>(b) * attn_num_heads_,
                    params.layer_idx, stream);
            }
        } else if (tiered) {
            attn->prefill_attention(
                q_attn[r], B, tv.seq_len_kv,
                tv.seqlens_k, tv.block_tables, tv.max_blocks_per_seq,
                tv.kv_cache,
                params.cache_stride_block, params.cache_stride_row,
                params.page_size,
                /*is_sparse=*/true, /*chunk_causal=*/false,
                tv.sparse_indices, params.topk_lengths[r],
                opts_.index_topk,
                prefill_out_[r], prefill_lse_[r],
                params.layer_idx, stream);
        } else {
        // TD-SPARSE-CHUNK-PREFILL: a prefill chunk runs SPARSE when the
        // prefill producer emitted per-chunk-row top-k (execute_attention
        // set is_sparse + the per-rank index/length pointers): the sparse
        // kernel then combines each row's selection with its causal bound
        // (s_kv_per_row = seqlens_k inside the device,
        // INV-SPARSE-CHUNK-CAUSAL). Without produced indices a chunk stays
        // DENSE chunk-causal (legacy TD-PREFILL-CHUNK-ATTN path).
        const bool sparse_chunk = chunk_rows && params.is_sparse
            && params.sparse_indices && params.topk_lengths;
        attn->prefill_attention(
            q_attn[r], B, seq_len_kv,
            params.seqlens_k ? params.seqlens_k[r] : nullptr,
            params.block_tables ? params.block_tables[r] : nullptr,
            params.max_blocks_per_seq,
            params.kv_cache_ptrs ? params.kv_cache_ptrs[r] : nullptr,
            params.cache_stride_block, params.cache_stride_row,
            params.page_size,
            /*is_sparse=*/chunk_rows ? sparse_chunk : params.is_sparse,
            /*chunk_causal=*/chunk_rows,
            (chunk_rows && !sparse_chunk) || !params.sparse_indices
                ? nullptr : params.sparse_indices[r],
            (chunk_rows && !sparse_chunk) || !params.topk_lengths
                ? nullptr : params.topk_lengths[r],
            opts_.index_topk,
            prefill_out_[r], prefill_lse_[r],
            params.layer_idx, stream);
        }

        partial_outputs[r] = prefill_out_[r];
        partial_lses[r] = prefill_lse_[r];
    }

    // Steps 10-12: DCP correction via DcpAttentionWrapper. KV-SHARDED mode
    // only (INV-DCP-KVREP): with the step-7b Q-head allgather above, every
    // rank's partial covers ALL dcp*HL heads over its LOCAL token shard, so
    // the LSE combine (allgather LSE [dcp, B, dcp*HL] → lse-correct →
    // output allreduce) is a genuine SAME-head merge over disjoint token
    // shards (INV-KVS-QAG; the wrapper's combine_num_heads = dcp*HL). Under
    // replicated KV every rank's attention is already complete for its own
    // head shard — running the combine there would cross-mix different heads
    // and corrupt long-context logits.
    if (opts_.dcp_wrapper && opts_.dcp_kv_sharded) {
        opts_.dcp_wrapper->correct_output(
            partial_outputs.data(),
            partial_lses.data(),
            B,
            attn_streams_.data());
    }

    // Steps 13-14: o_proj + TP allreduce.
    // INV-KVS-QAG slice (the reduce-scatter step, realized as allreduce +
    // slice): after the combine every rank holds the FULL combined
    // [B, dcp*HL, D_c]; rank r's kv_bv/o_proj consume ONLY its own HL-head
    // slice (global heads [r*HL, (r+1)*HL) — matching its TP o_proj shard),
    // addressed in place via a head offset + the dcp*HL per-token leading
    // dim. No copy needed.
    if (qag) {
        const int D_c = opts_.kv_lora_rank;
        std::vector<void*> sliced(dcp_size_);
        for (int r = 0; r < dcp_size_; ++r) {
            sliced[r] = static_cast<char*>(prefill_out_[r])
                + static_cast<size_t>(r) * num_heads_local_ * D_c * 2;
        }
        execute_oproj_and_reduce(params, sliced.data(),
                                  /*in_ld_heads=*/attn_num_heads_);
    } else {
        execute_oproj_and_reduce(params, partial_outputs.data(),
                                  /*in_ld_heads=*/num_heads_local_);
    }
}

// ── Main entry point ────────────────────────────────────────────────────────

void DcpExecutor::execute_attention(const AttentionExecParams& params) {
    // Steps 1-6: common prefix (projections + norms + k_append)
    execute_common_prefix(params);

    // Step 6b (GLM-25a/b): DSA lightning indexer. On a FULL layer, append this
    // token's indexer key to the layer's persistent position cache, score all
    // cached positions and produce the causal top-k; on a SHARED layer
    // (IndexShare, GLM-25b) reuse the preceding full layer's top-k with no
    // recompute. Then flip is_sparse so the attention kernels take the sparse
    // path. Non-graph only; produce_sparse_indices returns false (→ dense) for
    // non-DSA layers, chunked prefill, unprovisioned storage, or sequences
    // beyond the window. dcp>=2: replicated indexer mode runs every rank over
    // its own full replica (identical top-k per rank, TD-GLM-INDEXER-DCP);
    // local indexer mode (TD-GLM-INDEXER-LOCAL-MERGE) runs every rank over
    // its OWN position shard and the cross-rank merge below reconstructs the
    // identical global top-k.
    AttentionExecParams p = params;
    if (opts_.has_dsa && !params.use_graph && params.indexer_prefill_append) {
        // TD-GLM-INDEXER-PREFILL: prefill/chunked-prefill step blessed by the
        // dispatcher's coverage guard — append the chunk's indexer keys on
        // EVERY rank (replicated storage, TD-GLM-INDEXER-DCP), batched over
        // all chunk rows. With Options::sparse_prefill OFF the chunk's own
        // attention stays dense prefill (append only).
        bool all_appended = true;
        for (int r = 0; r < dcp_size_; ++r) {
            opts_.attention_devices[r]->set_device();
            if (!append_indexer_chunk(opts_.attention_devices[r], r, params))
                all_appended = false;
        }
        if (!all_appended) {
            // Step-invariant causes only (weights/positions absent for this
            // layer) — such a layer's storage is never written OR scored, so
            // this cannot corrupt sparse decode; surface it once regardless.
            static std::once_flag chunk_append_warned;
            std::call_once(chunk_append_warned, [&] {
                spdlog::warn("DcpExecutor: indexer chunk append skipped after "
                             "dispatcher blessing (layer={}, B={}) — this "
                             "layer stays dense-only",
                             params.layer_idx, params.batch_size);
            });
        }
        // TD-SPARSE-CHUNK-PREFILL: sparse CHUNK PREFILL attention. After the
        // chunk's keys are appended, run the producer's Q half + per-chunk-
        // row causal top-k on every rank, then flip is_sparse: the nongraph
        // consumer runs the chunk SPARSE + chunk-causal (each row attends
        // its own top-k ∩ [0, its position), INV-SPARSE-CHUNK-CAUSAL).
        // Indexer mode: BOTH — replicated produces the per-row global top-k
        // directly; local (TD-SPARSE-PREFILL-LOCAL-INDEXER) produces per-row
        // SHARD candidates which the cross-rank merge below reconstructs
        // into the identical global top-k. Replicated KV only in this pass —
        // sharded KV would need per-row GLOBAL→LOCAL translation + local
        // per-row bounds (TD-SPARSE-PREFILL-KVS) and keeps the dense chunk
        // path.
        // TD-KVT-ADMISSION-UPFRONT (Options::tiered_prefill): sharded KV no
        // longer forces dense chunks — the per-row global top-k is KVS-4-
        // translated below (indexer_shard_translate is batched per row; the
        // per-row causal bound survives translation because local ordering
        // is global-ascending) and the consumer runs per-row sub-dispatches
        // on tier steps.  Without the flag, sharded keeps the dense chunk
        // path (TD-SPARSE-PREFILL-KVS legacy behavior, byte-identical).
        const bool sharded_kv = opts_.dcp_kv_sharded && dcp_size_ >= 2;
        const bool sharded_pf_ok = !sharded_kv
            || (opts_.tiered_prefill
                && params.batch_size <= opts_.max_batch_size
                && opts_.dcp_chunk_tokens > 0 && sparse_local_indices_dev_[0]);
        if (all_appended && opts_.sparse_prefill && sharded_pf_ok) {
            // TD-PREFILL-SUPERCHUNK: this sub-chunk's persistent selection
            // rows live at its global row range — the consumer indexes rows
            // [0, B) off the OFFSET base (0 for legacy prefill).
            const size_t sc_off = static_cast<size_t>(
                params.batch_row_offset > 0 ? params.batch_row_offset : 0);
            const int itk = std::max(opts_.index_topk, 1);
            bool all = true;
            for (int r = 0; r < dcp_size_; ++r) {
                opts_.attention_devices[r]->set_device();
                if (produce_sparse_indices_prefill(
                        opts_.attention_devices[r], r, params)) {
                    sparse_indices_ptrs_[r] =
                        static_cast<const int*>(sparse_indices_dev_[r])
                        + sc_off * itk;
                    topk_lengths_ptrs_[r] =
                        static_cast<const int*>(topk_lengths_dev_[r]) + sc_off;
                } else {
                    sparse_indices_ptrs_[r] = nullptr;
                    topk_lengths_ptrs_[r] = nullptr;
                    all = false;
                }
            }
            // TD-SPARSE-PREFILL-LOCAL-INDEXER: allgather the per-chunk-row
            // shard candidates and merge them per row into the global top-k
            // (same seam as the decode path; skipped on IndexShare reuse —
            // the buffers already hold this step's merged result).
            if (all && indexer_local_ && indexer_step_fresh_)
                merge_local_indexer_candidates(params);
            // TD-KVT-ADMISSION-UPFRONT: sharded KV — translate the chunk's
            // per-row GLOBAL selection to each rank's LOCAL slot indices
            // (KVS-4; same kernel as decode, batched over the chunk rows).
            // Local ordering is global-ascending, so each row's causal
            // global selection lands strictly below its rank-local bound
            // (seqlens_k[r], KVS-3) — the sparse consumer stays exact.
            // Runs every layer (IndexShare shared layers reuse the global
            // buffers; the translation is recomputed, exactly like decode).
            if (all && sharded_kv) {
                for (int r = 0; r < dcp_size_; ++r) {
                    auto* attn = opts_.attention_devices[r];
                    attn->set_device();
                    attn->indexer_shard_translate(
                        sparse_indices_ptrs_[r], topk_lengths_ptrs_[r],
                        sparse_local_indices_dev_[r],
                        topk_local_lengths_dev_[r],
                        params.batch_size, opts_.index_topk,
                        opts_.dcp_chunk_tokens, dcp_size_, r,
                        attn_streams_[r]);
                    sparse_indices_ptrs_[r] =
                        static_cast<const int*>(sparse_local_indices_dev_[r]);
                    topk_lengths_ptrs_[r] =
                        static_cast<const int*>(topk_local_lengths_dev_[r]);
                }
            }
            if (all) {
                p.is_sparse = true;
                p.sparse_indices = sparse_indices_ptrs_.data();
                p.topk_lengths = topk_lengths_ptrs_.data();
                // GLM-25k (TD-KVT-PREFILL): the chunk's per-row selection is
                // now fully enqueued — same overlapped-readback seam as the
                // decode branch below (prepare() no-ops at B > 1; the
                // dispatcher only sets kv_tiering on B==1 steps anyway).
                if (p.kv_tiering) {
                    for (int r = 0; r < dcp_size_; ++r) {
                        opts_.attention_devices[r]->set_device();
                        p.kv_tiering->prepare(
                            r, p.layer_idx, p.sparse_indices[r],
                            p.topk_lengths[r], p.batch_size,
                            attn_streams_[r], indexer_step_fresh_);
                    }
                }
                if (!sparse_prefill_logged_) {
                    sparse_prefill_logged_ = true;
                    spdlog::info("DcpExecutor: DSA sparse CHUNK PREFILL "
                                 "ACTIVE (dcp={}, B={}, first layer={}, "
                                 "indexer={})",
                                 dcp_size_, params.batch_size,
                                 params.layer_idx,
                                 indexer_local_ ? "local+merged"
                                                : "replicated");
                }
            }
        }
    } else if (opts_.has_dsa && !params.use_graph) {
        // TD-GLM-INDEXER-DCP: ALL ranks must produce (each into its own GPU's
        // buffers) or the whole layer stays dense — a partial set would hand
        // some rank null indices under is_sparse.
        bool all = true;
        for (int r = 0; r < dcp_size_; ++r) {
            if (produce_sparse_indices(opts_.attention_devices[r], r, params)) {
                sparse_indices_ptrs_[r] =
                    static_cast<const int*>(sparse_indices_dev_[r]);
                topk_lengths_ptrs_[r] =
                    static_cast<const int*>(topk_lengths_dev_[r]);
            } else {
                sparse_indices_ptrs_[r] = nullptr;
                topk_lengths_ptrs_[r] = nullptr;
                all = false;
            }
        }
        // TD-GLM-INDEXER-LOCAL-MERGE (dcp_indexer_mode=local): the producers
        // above emitted per-rank SHARD candidates (local indices + scores)
        // into the packed send buffers — allgather them and run the exact
        // cross-rank merge on every rank into sparse_indices_dev_/
        // topk_lengths_dev_ (the same buffers replicated mode's producer
        // writes), so everything downstream — IndexShare reuse, the KVS-4
        // sharded-KV translation, the sparse consumers — is mode-agnostic.
        // Skipped on IndexShare reuse (the buffers already hold the merged
        // result of this step's full layer; reuse keys are only set inside
        // the merge, AFTER it succeeded).
        if (all && indexer_local_ && indexer_step_fresh_)
            merge_local_indexer_candidates(params);
        const bool sharded_kv = opts_.dcp_kv_sharded && dcp_size_ >= 2;
        if (all && sharded_kv) {
            // KVS-4 (replicated indexer × sharded KV): the produced top-k is
            // GLOBAL and identical on every rank (replicated indexer-K), but
            // rank r's sparse consumer indexes its LOCAL KV staging (its own
            // token shard, INV-4.9e). Translate the global list to each
            // rank's owned subset as LOCAL slot indices and repoint the
            // consumer at the translated buffers. A rank owning none of the
            // selection gets length 0 → zero output + lse=+inf → the QAG
            // combine weights it 0 (INV-KVS-EMPTY). Missing shard config
            // fails CLOSED to dense (never silently-wrong indices).
            if (opts_.dcp_chunk_tokens <= 0 || !sparse_local_indices_dev_[0]) {
                static std::once_flag translate_cfg_warned;
                std::call_once(translate_cfg_warned, [&] {
                    spdlog::warn("DcpExecutor: sharded KV without "
                                 "dcp_chunk_tokens/translation buffers — DSA "
                                 "forced DENSE (KVS-4 fail-closed)");
                });
                all = false;
                for (int r = 0; r < dcp_size_; ++r) {
                    sparse_indices_ptrs_[r] = nullptr;
                    topk_lengths_ptrs_[r] = nullptr;
                }
            } else {
                for (int r = 0; r < dcp_size_; ++r) {
                    auto* attn = opts_.attention_devices[r];
                    attn->set_device();
                    attn->indexer_shard_translate(
                        sparse_indices_dev_[r], topk_lengths_dev_[r],
                        sparse_local_indices_dev_[r],
                        topk_local_lengths_dev_[r],
                        params.batch_size, opts_.index_topk,
                        opts_.dcp_chunk_tokens, dcp_size_, r,
                        attn_streams_[r]);
                    sparse_indices_ptrs_[r] =
                        static_cast<const int*>(sparse_local_indices_dev_[r]);
                    topk_lengths_ptrs_[r] =
                        static_cast<const int*>(topk_local_lengths_dev_[r]);
                }
            }
        }
        if (all) {
            p.is_sparse = true;
            p.sparse_indices = sparse_indices_ptrs_.data();
            p.topk_lengths = topk_lengths_ptrs_.data();
            // GLM-25k (TD-KVT-SYNC): the selection for this layer is now
            // fully enqueued on each rank's attention stream (top-k, local
            // merge if any) — let the tiering hook start its overlapped
            // selection readback BEFORE we enqueue the projection/staging
            // work between here and the sparse consumer.  On an IndexShare
            // SHARED layer the producer reused the previous buffers
            // (indexer_step_fresh_ == false: byte-identical content), so the
            // hook can skip the readback and reuse its host copy.
            if (p.kv_tiering) {
                for (int r = 0; r < dcp_size_; ++r) {
                    opts_.attention_devices[r]->set_device();
                    p.kv_tiering->prepare(
                        r, p.layer_idx, p.sparse_indices[r],
                        p.topk_lengths[r], p.batch_size, attn_streams_[r],
                        indexer_step_fresh_);
                }
            }
            if (!dsa_active_logged_) {
                dsa_active_logged_ = true;
                spdlog::info("DcpExecutor: DSA sparse attention ACTIVE "
                             "(dcp={}, B={}, first layer={}, kv={}, "
                             "indexer={})",
                             dcp_size_, params.batch_size, params.layer_idx,
                             sharded_kv ? "sharded+translated" : "replicated",
                             indexer_local_ ? "local+merged" : "replicated");
            }
        }
    }
    // GLM-25k (INV-KVT-2): a DSA-capable layer that fell back to DENSE under
    // tiering must be surfaced — dense staging reads the FULL prefix through
    // the real block tables, which is only legal while every page of the
    // layer is VRAM-resident. This covers decode steps AND (TD-KVT-PREFILL)
    // blessed sparse-prefill chunks whose per-layer production failed (the
    // chunk then runs the dense chunk path). The hook throws iff the layer
    // has cold pages (a silent fallback would read stale/reused pages);
    // otherwise it marks the layer sticky-dense (never demoted). The
    // dispatcher only sets kv_tiering on non-graph B==1 steps.
    if (p.kv_tiering && opts_.has_dsa && !p.use_graph && !p.is_sparse) {
        p.kv_tiering->on_dense_layer(p.layer_idx);
    }
    // TD-KVS-QAG-GRAPH: sharded KV is nongraph-only — the graph path replays
    // per-rank-HL DecodeGraphRunners + a kDcpAllreduce combine that no longer
    // exists (capture_dcp_graphs is a no-op under sharding); the QAG sequence
    // (Q-head allgather → all-head attention → combine → slice) must run on
    // the nongraph path. Decode graphs are engine-dormant anyway
    // (TD-DECODE-GRAPH).
    if (p.use_graph && opts_.dcp_kv_sharded && dcp_size_ >= 2)
        p.use_graph = false;
    const AttentionExecParams& params2 = p;

    // Steps 7-14: graph or non-graph path
    if (params2.use_graph) {
        execute_attention_graph(params2);
    } else {
        execute_attention_nongraph(params2);
    }

    // Predictive kv_bv dequant: schedule layer+2 on async stream.
    // Overlaps with expert FFN compute on other GPUs.
    if (dequant_pool_ && !all_layer_weights_.empty()) {
        const int lookahead = params.layer_idx + 2;
        if (lookahead < total_layers_
            && lookahead < static_cast<int>(all_layer_weights_.size())) {
            dequant_pool_->schedule_dequant(
                lookahead, all_layer_weights_[lookahead].data());
        }
    }
}

}  // namespace layerstorm::parallelism

// ── Dispatcher-side MLA phase hooks (arch_mla.h; attention refactor V2 P1) ──
// Bodies are the verbatim blocks carved from the former dispatch_attention.cpp
// driver — dispatcher members accessed via d_ (friend); byte-identical
// behavior to the inline blocks they replace.

namespace layerstorm::daemon {

bool ArchMla::validate_shape(
        const CommandDispatcher::InternalAttentionParams& p, int& batch_cap) {
    (void)p;
    // TD-40g: validate batch_size against DcpExecutor buffer cap — MLA
    // shapes keep the max_batch cap.
    batch_cap = d_.deps_.max_batch_size;
    return true;
}

bool ArchMla::stage_step(
        const CommandDispatcher::InternalAttentionParams& p,
        parallelism::AttentionExecParams& params,
        int batch_size, int layer, int dcp_size, bool kv_meta_ok) {
    using IndexerSeqMode = CommandDispatcher::IndexerSeqMode;
    using IndexerPageResult = CommandDispatcher::IndexerPageResult;

    // TD-INDEXER-POOL-EVICT: a Pool::kIndexerK exhaustion that would
    // downgrade a KV-DEMOTED sequence to dense is FATAL, not merely lossy.
    // Two reasons compound: (a) a dense step skips the indexer append, so
    // the sequence's contiguous coverage gets a permanent hole (kDead — it
    // can never return to sparse); (b) dense reads the whole prefix through
    // the REAL block tables, so under tiering the TD-KVT-PREFILL-REPROMOTE
    // lift must pull the ENTIRE cold prefix back into VRAM — exactly the
    // capacity that forced the demotion — and fail-closes (the 2026-08-24
    // serving incident: kIndexerK exhausted at a forked 25k prefix ⇒
    // repromote_seq stalled at 5018/29952 pages ⇒ CMP_ERROR).
    // The pool is sized for serving.max_concurrent_requests sequences, so
    // exhaustion means it is full of OTHER live sequences — prefix-cache
    // holders pin a CoW frontier page group each. That is RECLAIMABLE
    // capacity, so surface a RETRYABLE pool-exhaustion error instead: the
    // orchestrator frees one holder (its indexer pages die with its
    // sequence) and re-issues the identical step, which provisions and
    // stays SPARSE. Sequences with no demotions keep the old lossy dense
    // downgrade — legal there, and no behavior change off tiering.
    auto indexer_exhaustion_fatal = [&](uint64_t sid) {
        return d_.kv_tiering_ && d_.kv_tiering_->seq_has_demotions(sid);
    };
    auto raise_indexer_exhausted = [&]() {
        d_.last_internal_error_cat_ = ipc::CmpErrorCategory::kKvPoolExhausted;
        // Keep "exhausted" inside the 80-byte CMP_ERROR message field — the
        // orchestrator's evict-retry matches on it.
        d_.last_internal_error_msg_ =
            "indexer-K pool exhausted (demoted seq) — retryable, evict a "
            "prefix holder";
    };
    // TD-GLM-INDEXER-PAGED/-BATCH/-DCP/-COV: provision paged indexer-K per
    // batch entry (each entry is its own sequence; at dcp>=2 every rank gets
    // its own GPU's replica pages) and hand the producer the per-rank host
    // page tables, per-entry host seqlens, and a seq_id-aware step
    // fingerprint (the IndexShare reuse key).
    //
    // Coverage guard (TD-GLM-INDEXER-COV): the producer appends exactly one
    // key per entry per qualifying step, so sparse scoring is valid only if
    // this step ADVANCES each sequence's contiguous coverage (pos ==
    // next_pos; pos+1 == next_pos is the same step seen by a later layer) in
    // the SAME storage mode as all prior steps. Prefill / chunked-prefill
    // steps now APPEND their chunk's indexer keys (TD-GLM-INDEXER-PREFILL,
    // append-only branch below) and advance coverage by chunk_len, so
    // prefill + sparse decode compose. Non-appending steps (graph replay,
    // drafts, unsupported prefill shapes) leave a gap that permanently
    // downgrades the sequence to DENSE. With LS_INDEXER_REWIND=1
    // (INV-DSA-REWIND) a same-sequence contiguous OVERWRITE-REWIND (pos0 <=
    // next_pos — the speculative-verify partial-acceptance re-feed) is
    // additionally valid: position-keyed appends overwrite the re-fed rows
    // in place and next_pos becomes the high-water mark. A sequence that ever fell back to
    // the arena (pool exhaustion) is pinned there: switching to
    // later-allocated pages would score never-written memory.
    // INV-DCP-5 pairing legality (KVS-4; TD-GLM-INDEXER-LOCAL-MERGE
    // resolved): BOTH dcp_indexer_mode values are legal with BOTH KV modes.
    // replicated: every rank's producer emits the identical GLOBAL top-k.
    // local: indexer-K is position-sharded (round-robin by INDEXER PAGE,
    // halving resident indexer-K per rank); each rank scores its own shard
    // and the executor's candidate allgather + EXACT cross-rank merge
    // reconstructs the identical global top-k BEFORE any KV-mode handling.
    // Under sharded KV the (merged) global top-k is then translated per rank
    // to LOCAL staging indices (indexer_shard_translate) before the sparse
    // consumer runs; the QAG combine merges the per-rank sparse partials
    // exactly like dense ones. Local mode fails CLOSED to dense only when
    // the page ownership unit is unset (defensive; schema default 8192).
    const bool indexer_local_mode = dcp_size >= 2 && d_.deps_.live_config
        && d_.deps_.live_config->hardware.dcp_indexer_mode
               == config::DcpIndexerMode::local;
    const bool indexer_mode_ok = !indexer_local_mode
        || d_.deps_.live_config->memory.kv_cache.indexer_k_page_size_tokens > 0;
    if (!indexer_mode_ok && d_.deps_.live_config
        && d_.deps_.live_config->model.index_topk > 0) {
        static std::once_flag local_mode_warned;
        std::call_once(local_mode_warned, [] {
            spdlog::warn("dispatch_attention: dcp_indexer_mode=local without "
                         "a valid indexer_k_page_size_tokens — DSA forced "
                         "DENSE (fail-closed)");
        });
    }
    const bool indexer_prefill_step = (p.is_prefill != 0) || (p.chunk_len > 0);
    // V4-7b: V4 carries index_topk for its Lightning indexer but the DSA
    // provisioning/coverage machinery is MLA-only — the V4 pipeline manages
    // its own LID tier (ensure_v4_tier_pages).
    if (d_.deps_.live_config && d_.deps_.live_config->model.index_topk > 0
        && indexer_mode_ok && !params.use_graph && p.is_draft == 0
        && kv_meta_ok && d_.deps_.sideband_base) {
        const auto* be = reinterpret_cast<const ipc::BatchDescriptorEntry*>(
            d_.deps_.sideband_base + ipc::IpcLayout::kBatchDescriptorOff);

        // Step fingerprint: FNV-1a over (seq, pos) — the IndexShare reuse key
        // and the producer/appender blessing token.
        uint64_t key = 1469598103934665603ULL;
        for (int b = 0; b < batch_size; ++b) {
            key = (key ^ be[b].seq_id) * 1099511628211ULL;
            key = (key ^ be[b].token_pos) * 1099511628211ULL;
        }

        if (!indexer_prefill_step) {
            // TD-GLM-INDEXER-B1CASCADE RESOLVED (INV-DSA-ROWMIX): the guard
            // is PER ROW. A dead/gapped row no longer suppresses sparse for
            // its whole cohort — it is marked DENSE in the per-row mask and
            // only an ALL-dense cohort suppresses the step entirely. Live
            // rows keep appending and advancing coverage; the executor
            // splits a mixed cohort into per-row batch-of-1 sub-dispatches
            // (sparse live rows / dense dead rows). B==1 semantics are
            // unchanged (a single dead row IS the all-dense case).
            d_.indexer_row_dense_.assign(static_cast<size_t>(batch_size), 0);
            int n_dense = 0;
            bool pages_ok = true;   // every SPARSE row on the paged path
            for (int b = 0; b < batch_size; ++b) {
                auto& cov = d_.sequences_[be[b].seq_id].indexer_cov;
                const uint32_t pos = be[b].token_pos;
                if (cov.mode == IndexerSeqMode::kDead) {
                    d_.indexer_row_dense_[b] = 1;
                    ++n_dense;
                    continue;
                }
                const bool advancing = (pos == cov.next_pos);
                const bool repeat    = (pos + 1 == cov.next_pos);
                // INV-DSA-REWIND: an overwrite-rewind (pos < next_pos, this
                // sequence) re-appends AT its position — row `pos` of the
                // pinned storage is overwritten in place and next_pos (the
                // high-water mark) stands. Only with LS_INDEXER_REWIND=1;
                // `repeat` is its next_pos−1 special case either way.
                const bool rewind = d_.indexer_rewind_ok_
                    && pos < cov.next_pos;
                if (!advancing && !repeat && !rewind) {  // gap
                    cov.mode = IndexerSeqMode::kDead;
                    d_.indexer_row_dense_[b] = 1;
                    ++n_dense;
                    continue;
                }
                if (cov.mode == IndexerSeqMode::kArena) {
                    if (batch_size > 1) {  // arena is B==1-only → this row
                        cov.mode = IndexerSeqMode::kDead;  // is dense → gap
                        d_.indexer_row_dense_[b] = 1;
                        ++n_dense;
                        continue;
                    }
                    pages_ok = false;  // stay pinned to the arena
                    continue;
                }
                // kUnset or kPaged: (re)provision — idempotent for repeats,
                // grows one page group at page boundaries otherwise.
                // Local indexer mode never blesses the arena (the executor
                // arena is replicated-shape) — provisioning failure is a
                // permanent dense downgrade instead.
                const auto ipr =
                    d_.ensure_indexer_pages(be[b].seq_id, pos, b, dcp_size);
                if (ipr == IndexerPageResult::kOk) {
                    cov.mode = IndexerSeqMode::kPaged;
                } else if (ipr == IndexerPageResult::kExhausted
                           && indexer_exhaustion_fatal(be[b].seq_id)) {
                    raise_indexer_exhausted();  // leave cov.mode intact
                    return false;
                } else if (cov.mode == IndexerSeqMode::kUnset
                           && batch_size == 1 && advancing
                           && !indexer_local_mode) {
                    cov.mode = IndexerSeqMode::kArena;  // arena from pos 0
                    pages_ok = false;
                } else {
                    // kPaged growth failure (can't switch to arena: earlier
                    // positions live in pages) or B>1 without pages → dense
                    // row → gap.
                    cov.mode = IndexerSeqMode::kDead;
                    d_.indexer_row_dense_[b] = 1;
                    ++n_dense;
                }
            }

            if (n_dense == batch_size) {
                // Whole cohort dense (B==1 dead row included) — same
                // suppression contract as before.
                params.indexer_sparse_suppress = true;
            } else {
                if (pages_ok) {
                    params.indexer_k_pages        = d_.indexer_table_bases_.data();
                    params.indexer_k_page_stride  = d_.indexer_page_stride_;
                    params.indexer_k_batch_stride = d_.indexer_batch_stride_;
                    params.indexer_k_page_tokens  = d_.deps_.live_config
                        ->memory.kv_cache.indexer_k_page_size_tokens;
                }
                // Commit coverage for SPARSE rows only: paged batches append
                // every live entry; the B==1 arena path appends via the
                // executor arena. A dense row's key is never appended — its
                // coverage stays behind (it is kDead already). (pages_ok=
                // false at B>1 is unreachable here — the arena branch above
                // kills it.)
                params.indexer_step_key = key ? key : 1;
                for (int b = 0; b < batch_size; ++b) {
                    if (d_.indexer_row_dense_[b]) continue;
                    auto& cov = d_.sequences_[be[b].seq_id].indexer_cov;
                    if (be[b].token_pos == cov.next_pos) ++cov.next_pos;
                }
                // MIXED cohort (only possible at B>1): hand the executor the
                // per-row mask (INV-DSA-ROWMIX split). Uniform all-sparse
                // cohorts keep the legacy nullptr (batched sparse path).
                if (n_dense > 0)
                    params.indexer_row_dense = d_.indexer_row_dense_.data();
            }
        } else {
            // TD-GLM-INDEXER-PREFILL: prefill / chunked-prefill step — the
            // batch rows are the chunk's PROMPT POSITIONS. Bless an indexer-K
            // chunk APPEND (executor runs the producer's K half, batched; no
            // scoring, no sparse consumption — attention stays dense prefill)
            // and advance coverage by chunk_len. Supported shape: ONE
            // sequence, consecutive ascending positions (the only shape the
            // engine's prefill emits). Anything else cannot append → the
            // skipped positions form a permanent gap (legacy behavior).
            bool shape_ok = batch_size >= 1;
            for (int b = 1; b < batch_size && shape_ok; ++b)
                shape_ok = be[b].seq_id == be[0].seq_id
                        && be[b].token_pos
                               == be[0].token_pos + static_cast<uint32_t>(b);
            if (!shape_ok) {
                for (int b = 0; b < batch_size; ++b)
                    d_.sequences_[be[b].seq_id].indexer_cov.mode =
                        IndexerSeqMode::kDead;
            } else {
                auto& cov = d_.sequences_[be[0].seq_id].indexer_cov;
                const uint32_t pos0 = be[0].token_pos;
                const uint32_t end  = pos0 + static_cast<uint32_t>(batch_size);
                const bool advancing = (pos0 == cov.next_pos);
                // TD-PREFILL-SUPERCHUNK: a superchunk advances the frontier by
                // K sub-chunks at layer 0, then every later layer REPLAYS the
                // same sub-chunks behind the advanced frontier — accept any
                // fully-covered window (end <= next_pos) as the layer-replay
                // `repeat` when the command carries the superchunk flag. The
                // legacy single-chunk replay (end == next_pos) is its K == 1
                // special case and stays the only repeat shape without the
                // flag (a bare end < next_pos would otherwise be a rewind).
                const bool repeat    = (end == cov.next_pos)   // later layer,
                                                               // same step
                    || (p.superchunk && end < cov.next_pos);   // later layer,
                                                               // earlier sub-
                                                               // chunk of this
                                                               // superchunk
                // INV-DSA-REWIND (LS_INDEXER_REWIND=1): a same-sequence
                // contiguous chunk starting BEHIND the frontier (pos0 <
                // next_pos, no gap — the dsp52 partial-acceptance re-feed
                // shape) OVERWRITES rows [pos0, end) in place (position-
                // keyed appends, exactly like the KV rows the same re-feed
                // overwrites) and extends the high-water mark when end >
                // next_pos. Superchunk sub-chunks keep their own repeat
                // shape and are never rewinds (fail-closed together).
                const bool rewind = d_.indexer_rewind_ok_
                    && !p.superchunk && pos0 < cov.next_pos;
                if (cov.mode == IndexerSeqMode::kDead) {
                    // stays dead — no append, no blessing
                } else if (!advancing && !repeat && !rewind) {
                    cov.mode = IndexerSeqMode::kDead;  // gap (or rewind with
                                                       // LS_INDEXER_REWIND
                                                       // unset)
                } else {
                    bool pages_ok = true;
                    if (cov.mode == IndexerSeqMode::kArena) {
                        pages_ok = false;  // pinned to the arena
                    } else {
                        // Provision pages covering [pos0, end): per-row calls
                        // grow the pool allocation cumulatively AND fill each
                        // batch row's table slice (the appender reads row b's
                        // slice for position b). Idempotent for repeats.
                        auto ipr = IndexerPageResult::kOk;
                        for (int b = 0;
                             b < batch_size && ipr == IndexerPageResult::kOk;
                             ++b)
                            ipr = d_.ensure_indexer_pages(be[0].seq_id,
                                                      be[b].token_pos, b,
                                                      dcp_size);
                        const bool ok = ipr == IndexerPageResult::kOk;
                        if (ok) {
                            cov.mode = IndexerSeqMode::kPaged;
                        } else if (ipr == IndexerPageResult::kExhausted
                                   && indexer_exhaustion_fatal(
                                          be[0].seq_id)) {
                            raise_indexer_exhausted();  // cov.mode intact
                            return false;
                        } else if (cov.mode == IndexerSeqMode::kUnset
                                   && advancing && !indexer_local_mode) {
                            // Fresh sequence (next_pos == pos0 == 0): the
                            // arena is fine for a single-sequence chunk.
                            // (Never in local mode — replicated-shape arena.)
                            cov.mode = IndexerSeqMode::kArena;
                            pages_ok = false;
                        } else {
                            cov.mode = IndexerSeqMode::kDead;  // growth failed
                        }
                    }
                    if (cov.mode != IndexerSeqMode::kDead) {
                        if (pages_ok) {
                            params.indexer_k_pages = d_.indexer_table_bases_.data();
                            params.indexer_k_page_stride  = d_.indexer_page_stride_;
                            params.indexer_k_batch_stride = d_.indexer_batch_stride_;
                            params.indexer_k_page_tokens  = d_.deps_.live_config
                                ->memory.kv_cache.indexer_k_page_size_tokens;
                        }
                        params.indexer_step_key = key ? key : 1;
                        params.indexer_prefill_append = true;
                        if (advancing) {
                            cov.next_pos = end;
                        } else if (rewind && end > cov.next_pos) {
                            // Overwrite-rewind past the old frontier: rows
                            // [pos0, next_pos) overwritten, (next_pos, end)
                            // freshly appended — contiguous either way.
                            cov.next_pos = end;
                        }
                    }
                }
            }
        }
    }

    // TD-40j: chunked prefill support (MLA arch — the V4 copy of the first
    // branch lives in ArchDeepseekV4::stage_step; V4 never synthesizes a
    // whole-prompt chunk descriptor).
    if (p.is_prefill && p.chunk_len > 0) {
        params.chunk_start = static_cast<int>(p.chunk_start);
        params.chunk_len   = static_cast<int>(p.chunk_len);
    } else if (p.is_prefill) {
        // Whole-prompt prefill: synthesize the chunk descriptor so the
        // executor takes the chunk path (per-row causal attention,
        // TD-PREFILL-CHUNK-ATTN; indexer chunk append). The batch rows ARE
        // the chunk: one entry per prompt position.
        params.chunk_len = batch_size;
        if (d_.deps_.sideband_base) {
            const auto* be0 = reinterpret_cast<const ipc::BatchDescriptorEntry*>(
                d_.deps_.sideband_base + ipc::IpcLayout::kBatchDescriptorOff);
            params.chunk_start = static_cast<int>(be0[0].token_pos);
        }
    }

    // ── GLM-25k: DSA-guided KV tiering ─────────────────────────────────────
    // Engage only on sparse-blessed B==1 steps (non-draft, non-graph):
    // decode steps, and — TD-KVT-PREFILL — blessed B==1 SPARSE prefill
    // chunks (compute.dsa_sparse_prefill): a single-row chunk's sparse
    // consumption is decode-shaped (per-row causal top-k + the same
    // materialize seam), so long/chunked prefill demotes behind the chunk
    // frontier and prefill INTO a sequence with cold pages works (cold rows
    // come from the pinned pool via materialize — no re-promotion needed).
    // begin_layer stages the per-layer step context (host block-table row
    // per rank) and polls in-flight demotions; the executor then
    // materializes selected rows through the hook (INV-KVT-1).
    // TD-KVT-BATCH: tiering state is per-sequence — any number of sequences
    // may be tiered across interleaved B==1 steps.  Once a sequence has
    // demoted pages, a step shape that would read its full prefix through
    // the real block tables must not run against neutralized handles:
    // NON-TIERABLE PREFILL shapes (dense prefill chunks, B>1 chunk cohorts,
    // sharded-KV dense fallback) are lifted by FULL cold-page re-promotion
    // at the gate below (TD-KVT-PREFILL-REPROMOTE); drafts, graph replay,
    // B>1 DECODE cohorts and coverage-dead dense decode fail CLOSED
    // (TD-KVT-SPEC-DRAFT Phase-12 / TD-KVT-BATCH-COHORT) — placement must
    // never change results.  Fork / rewind / restore are LIFTED upstream
    // (refcounted cold-slot sharing + cold-page re-promotion).  Under sharded KV
    // (TD-KVT-DCP-SHARDED) the same seam holds for DECODE (begin_layer
    // receives each rank's LOCAL block-table row and the hook consumes the
    // KVS-4 translated local selection, INV-KVT-9), but prefill chunks stay
    // non-tierable there: the executor's sparse-prefill pass falls back to
    // DENSE chunks under sharded KV (TD-SPARSE-PREFILL-KVS).
    tier_seq_ = 0;
    tier_pos_ = 0;
    tier_step_ = false;
    tier_rows_ = 1;
    if (d_.kv_tiering_ && kv_meta_ok && d_.deps_.sideband_base) {
        d_.kv_tiering_->poll_demotions();
        const auto* tbe = reinterpret_cast<const ipc::BatchDescriptorEntry*>(
            d_.deps_.sideband_base + ipc::IpcLayout::kBatchDescriptorOff);
        tier_seq_ = tbe[0].seq_id;
        tier_pos_ = tbe[0].token_pos;
        // TD-KVT-PREFILL: a blessed prefill chunk is tierable only when the
        // executor will actually run it SPARSE — gate mirrors the
        // executor's own sparse-prefill pass (Options::sparse_prefill on,
        // replicated KV; replicated indexer is guaranteed by the tiering
        // construction gate).  A dense chunk under tiering would stage the
        // full prefix through the real block tables (INV-KVT-2).
        // TD-KVT-ADMISSION-UPFRONT (memory.kv_tiering.tiered_prefill): a
        // blessed sparse prefill CHUNK COHORT (B>1 rows of ONE sequence —
        // the indexer_prefill_append blessing already validated same-seq
        // ascending shape) is ALSO a tier step, under BOTH KV modes: the
        // executor consumes it per row (B==1 sub-dispatches; sharded KV
        // rides the KVS-4 per-row translation).  Rows must fit the
        // executor's translation buffers (max_batch_size) AND the
        // manager's cohort staging (begin_layer re-checks its cap).
        const auto* lc = d_.deps_.live_config;
        const bool tiered_prefill_mode = lc
            && lc->memory.kv_tiering.tiered_prefill
            && lc->compute.dsa_sparse_prefill
            && batch_size <= std::max(1, d_.deps_.max_batch_size);
        const bool prefill_sparse_tierable = params.indexer_prefill_append
            && lc && lc->compute.dsa_sparse_prefill
            && (!(d_.kv_sharded_ && dcp_size >= 2) || tiered_prefill_mode)
            && (batch_size == 1 || tiered_prefill_mode);
        const bool tierable
            = (batch_size == 1
               || (params.indexer_prefill_append && batch_size > 1))
            && p.is_draft == 0
            && !params.use_graph
            && params.indexer_step_key != 0
            && !params.indexer_sparse_suppress
            && (params.indexer_prefill_append ? prefill_sparse_tierable
                                              : params.chunk_len == 0);
        if (tierable) {
            const int L = d_.kv_layers_ > 0 ? d_.kv_layers_ : 1;
            const int lyr = (layer >= 0 && layer < L) ? layer : 0;
            const size_t bt_off = static_cast<size_t>(lyr) * batch_size
                                * d_.max_blocks_per_seq_;
            d_.tier_bt_scratch_.resize(d_.kv_meta_scratch_.size());
            for (size_t r = 0; r < d_.kv_meta_scratch_.size(); ++r)
                d_.tier_bt_scratch_[r] =
                    d_.kv_meta_scratch_[r].host_block_tables.data() + bt_off;
            if (d_.kv_tiering_->begin_layer(layer, tier_seq_, tier_pos_,
                                         d_.tier_bt_scratch_.data(),
                                         batch_size)) {
                params.kv_tiering = d_.kv_tiering_.get();
                tier_step_ = true;
                tier_rows_ = batch_size;
            }
        }
        if (!tier_step_ && d_.kv_tiering_->has_demotions()) {
            // Narrowed fail-closed set (TD-KVT-SPEC-FORK): fork, rewind and
            // restore no longer trip this — they are lifted upstream
            // (on_seq_fork refcounted cold sharing; repromote_for_rewind /
            // repromote_seq cold re-promotion).
            bool touches_demoted = false;
            for (int b = 0; b < batch_size; ++b)
                if (d_.kv_tiering_->seq_has_demotions(tbe[b].seq_id))
                    touches_demoted = true;
            // TD-KVT-PREFILL-REPROMOTE: a NON-TIERABLE PREFILL shape on a
            // demoted sequence (dense prefill chunk — sparse gate off,
            // coverage-dead, or the sharded-KV dense fallback — or a B>1
            // chunk cohort) is lifted by FULL re-promotion: every cold page
            // returns to a fresh VRAM page with its exact demoted bytes
            // (repromote_seq(seq, 0) — VRAM must fit the full prefix again;
            // the alloc seam un-neutralizes seq_pages_ and poisons the
            // kv-meta dirty guard), then the kv metadata is rebuilt IN
            // PLACE from the fresh handles (same pre-sized host vectors +
            // device buffers — the layer-slice pointers derived above stay
            // valid; the H2D re-upload is attention-stream-ordered before
            // this step's kernels), so the dense chunk reads real, hot
            // block tables.  The lift runs at THIS gate — after
            // tierability is decided — never pre-kv-meta: a TIERABLE
            // sparse chunk must keep the no-re-promotion materialize path
            // (INV-KVT-13; re-promoting it would need the full prefix in
            // VRAM and break long-context capacity).  Allocation failure
            // keeps the step fail-closed (capacity, not correctness).
            const bool prefill_shape =
                (p.is_prefill != 0) || (p.chunk_len > 0);
            if (touches_demoted && prefill_shape && p.is_draft == 0
                && !params.use_graph) {
                bool ok = true;
                for (int b = 0; b < batch_size && ok; ++b) {
                    if (!d_.kv_tiering_->seq_has_demotions(tbe[b].seq_id))
                        continue;
                    ok = d_.kv_tiering_->repromote_seq(tbe[b].seq_id,
                                                    /*keep_frontier=*/0);
                }
                if (ok) {
                    d_.invalidate_kv_meta();  // force a real rebuild
                    ok = d_.build_kv_metadata(batch_size, dcp_size)
                         == CommandDispatcher::KvMetaResult::kOk;
                }
                if (!ok) {
                    d_.last_internal_error_cat_ =
                        ipc::CmpErrorCategory::kKvPoolExhausted;
                    d_.last_internal_error_msg_ =
                        "attention: non-tierable prefill on a demoted "
                        "sequence — full cold-page re-promotion failed "
                        "(VRAM capacity; fail-closed, "
                        "TD-KVT-PREFILL-REPROMOTE)";
                    return false;
                }
                touches_demoted = false;  // every touched seq is hot now
            }
            if (touches_demoted) {
                // What remains reads the full prefix through the real block
                // tables and stays ILLEGAL on a demoted sequence
                // (INV-KVT-2): drafts + graph replay (Phase-12 speculation,
                // TD-KVT-SPEC-DRAFT), B>1 DECODE cohorts
                // (TD-KVT-BATCH-COHORT) and coverage-dead dense decode.
                d_.last_internal_error_cat_ =
                    ipc::CmpErrorCategory::kComputeValidation;
                d_.last_internal_error_msg_ =
                    "attention: non-tierable step (draft/graph-replay "
                    "[TD-KVT-SPEC-DRAFT Phase-12] / B>1 decode cohort "
                    "[TD-KVT-BATCH-COHORT] / dense-coverage decode) on a "
                    "sequence with demoted KV pages — fail-closed "
                    "(INV-KVT-2)";
                return false;
            }
        }
    }


    return true;
}

bool ArchMla::execute(
        const CommandDispatcher::InternalAttentionParams& p,
        parallelism::AttentionExecParams& params,
        int batch_size, int layer, int dcp_size, bool kv_meta_ok) {
    (void)p; (void)batch_size; (void)dcp_size; (void)kv_meta_ok;
    d_.deps_.dcp_executor->execute_attention(params);

    // GLM-25k: demote this layer's pages that fell fully behind the retention
    // window (background D2H ordered after this layer's attention-stream
    // work; the device page returns to the allocator when the copy lands).
    // TD-KVT-ADMISSION-UPFRONT: a chunk cohort demotes behind its LAST row's
    // position — long served prefill demotes at every sub-chunk boundary,
    // bounding the hot set by retention + active chunk (the retention window
    // itself keeps the active superchunk hot, INV-KVT-4 unchanged).
    if (tier_step_ && params.kv_tiering) {
        auto tit = d_.sequences_.find(tier_seq_);
        if (tit != d_.sequences_.end()) {
            const int L = d_.kv_layers_ > 0 ? d_.kv_layers_ : 1;
            const int lyr = (layer >= 0 && layer < L) ? layer : 0;
            const auto& pgs = tit->second.kv_pages;
            const int num_logical = static_cast<int>(pgs.size()) / L;
            const uint32_t demote_pos = tier_pos_
                + static_cast<uint32_t>(tier_rows_ > 0 ? tier_rows_ - 1 : 0);
            if (num_logical > 0)
                d_.kv_tiering_->after_attention(layer, tier_seq_, demote_pos,
                                             pgs.data() + lyr, num_logical, L);
        }
    }

    return true;
}

}  // namespace layerstorm::daemon
