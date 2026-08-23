// DeepSeek-V4 attention pipeline: compressor inserts, SWA/CSA/HCA arm
// routing, dual rope, mHC seams, grouped o_proj, fork/free/spec-guard.
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

#include "daemon/attention/arch_deepseek_v4.h"
#include "daemon/v4_kv_tiering.h"
#include "core/memory/vram_allocator.h"  // kV4Fp8EntryBytes (SWA tier — always FP8)

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>   // LS_CHUNK_SMALLM gate (read once)
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace layerstorm::parallelism {

// ── V4-5c grouped o_proj (ticket G, resolves TD-V4-OPROJ) ───────────────────
//
// DeepSeek-V4 output projection: pure 2-stage grouped low-rank factorization
// (ref/llama.cpp/src/models/deepseek4.cpp:1066-1074 — NO base+LoRA sum).
// The inverse-roped attention output [rows, h_q, head_dim] is viewed as
// [rows, o_groups, group_dim] (group g = the contiguous head span
// [g*h_q/o_groups, (g+1)*h_q/o_groups)); stage 1 runs one strided batched
// BF16 GEMM over the o_groups slabs of o_proj_a (GGUF attn_output_a, memory
// [o_groups*o_lora_rank, group_dim] row-major, groups-major slabs); stage 2
// is a single shared GEMM against o_proj_b (GGUF attn_output_b, memory
// [hidden, o_groups*o_lora_rank] row-major). FP32 accumulation in both
// stages (launch_bf16_strided_batched_gemm_nt semantics: per batch
// C[n,m] = B[n,k] @ A[m,k]^T).
//
// TP (V4-2c, TD-V4-TP resolved 2026-08-21): stage 1 is sharded BY GROUP —
// rank r's o_proj_a slice carries o_groups/tp contiguous group slabs
// (matching its column-sharded q_b head span), stage 2 runs against the
// rank's row-parallel o_proj_b K-slice and produces a PARTIAL hidden; the
// CALLER (execute_attention_v4) allreduces hidden_out_ once per layer call
// (step-14 pattern). At dcp_size 1 the math is byte-identical to ticket G.
void DcpExecutor::execute_v4_grouped_oproj(int rank,
                                           const AttentionLayerWeights& w,
                                           const void* attn_out, int rows,
                                           void* out, void* stream) {
    if (opts_.v4_o_groups <= 0 || opts_.v4_o_lora_rank <= 0
        || opts_.v4_head_dim <= 0) {
        throw std::runtime_error(
            "DcpExecutor::execute_v4_grouped_oproj: V4 grouped o_proj is not "
            "configured (Options.v4_o_groups/v4_o_lora_rank/v4_head_dim) — "
            "engine sets these only for has_grouped_o_proj() models");
    }
    if (opts_.v4_o_groups % std::max(dcp_size_, 1) != 0) {
        throw std::runtime_error(
            "DcpExecutor::execute_v4_grouped_oproj: o_groups must be "
            "divisible by dcp_size (validator-enforced)");
    }
    if (rank < 0 || rank >= dcp_size_) {
        throw std::runtime_error(
            "DcpExecutor::execute_v4_grouped_oproj: bad rank");
    }
    if (rows <= 0 || rows > v4_oa_rows_) {
        throw std::runtime_error(
            "DcpExecutor::execute_v4_grouped_oproj: rows "
            + std::to_string(rows) + " outside the oa scratch bound "
            + std::to_string(v4_oa_rows_));
    }
    if (!w.o_proj_a || !w.o_proj_b) {
        throw std::runtime_error(
            "DcpExecutor::execute_v4_grouped_oproj: o_proj_a/o_proj_b weight "
            "pointers are null for this layer");
    }
    if (!attn_out || !out || !v4_oproj_oa_[rank]) {
        throw std::runtime_error(
            "DcpExecutor::execute_v4_grouped_oproj: null attn_out/out/oa "
            "scratch");
    }
    // V4-2c: per-rank geometry — the rank owns o_groups/tp groups whose
    // head spans are exactly its num_heads_local_ heads (group_dim is
    // tp-invariant: heads-per-group never changes).
    const int h_q = num_heads_local_;
    const int groups_local = opts_.v4_o_groups / std::max(dcp_size_, 1);
    if (groups_local <= 0 || h_q % groups_local != 0) {
        throw std::runtime_error(
            "DcpExecutor::execute_v4_grouped_oproj: h_q "
            + std::to_string(h_q) + " not divisible by per-rank o_groups "
            + std::to_string(groups_local));
    }

    auto* attn = opts_.attention_devices[rank];
    attn->set_device();
    if (!stream) stream = attn_streams_[rank];

    const int groups    = groups_local;                // 8/tp (rank-local)
    const int olr       = opts_.v4_o_lora_rank;                  // 1024
    const int group_dim = (h_q / groups) * opts_.v4_head_dim;    // 4096
    const int oa_width  = groups * olr;                // 8192/tp (rank-local)

    // Stage 1: per-group batched GEMM — oa[t, g, :] = A_g @ attn_out[t, g, :].
    compute::StridedBatchedGemmBf16Params s1{};
    s1.m           = olr;
    s1.n           = rows;
    s1.k           = group_dim;
    s1.A           = w.o_proj_a;                 // slab g at g*olr*group_dim
    s1.lda         = group_dim;
    s1.strideA     = static_cast<int64_t>(olr) * group_dim;
    s1.B           = attn_out;                   // [rows, h_q*head_dim]
    s1.ldb         = h_q * opts_.v4_head_dim;    // per-token leading dim
    s1.strideB     = group_dim;                  // group g span within a token
    s1.C           = v4_oproj_oa_[rank];         // [rows, groups*olr]
    s1.ldc         = oa_width;
    s1.strideC     = olr;
    s1.batch_count = groups;
    attn->batched_gemm_bf16(s1, stream);

    // Stage 2: shared GEMM — out[t, :] = o_proj_b @ oa[t, :].
    compute::StridedBatchedGemmBf16Params s2{};
    s2.m           = opts_.hidden_size;
    s2.n           = rows;
    s2.k           = oa_width;
    s2.A           = w.o_proj_b;                 // [hidden, oa_width] row-major
    s2.lda         = oa_width;
    s2.strideA     = 0;
    s2.B           = v4_oproj_oa_[rank];
    s2.ldb         = oa_width;
    s2.strideB     = 0;
    s2.C           = out;                        // [rows, hidden]
    s2.ldc         = opts_.hidden_size;
    s2.strideC     = 0;
    s2.batch_count = 1;
    attn->batched_gemm_bf16(s2, stream);
}

// ── V4-7b (ticket H): DeepSeek-V4 per-layer attention pipeline ──────────────
//
// B==1 decode-shaped steps only (prompt-fed prefill = one step per prompt
// token). Model reference: ref/llama.cpp/src/models/deepseek4.cpp
// build_attention (:796) + the compress/index builders; entry mapping and
// kernel contracts per compute/kernels/attention/v4_prep.h and
// compute/csa_hca_sm120_attention_device.h. Chunk/batched prefill fails loud
// (TD-V4-CHUNK-PREFILL): the chunk arm needs gather-staging visibility and a
// ring-capacity re-architecture (a stride-capacity ring cannot hold a whole
// chunk's compressor states) — perf work tracked by TD-V4-PREFILL-PERF.

void DcpExecutor::v4_fork_sequence(uint64_t src_id, uint64_t dst_id) {
    // TD-V4-SERVE-PREFIX: clone the executor-owned per-seq V4 state at
    // CMD_SEQ_FORK — per-rank compressor/LID state-ring blocks + spec-
    // snapshot blocks (device D2D on each rank's kAttention stream, ordered
    // after the parent's last writes) + the host step-window tracking, so a
    // forked child resumes at the parent's frontier (its first window at
    // step_hi + 1 passes the monotony gate) instead of failing the pos-0
    // entry requirement. A parent with no state (never stepped) is a no-op:
    // the child lazily allocates at pos 0 like any fresh sequence.
    auto it = v4_seq_state_.find(src_id);
    if (it == v4_seq_state_.end()) return;
    if (v4_seq_state_.count(dst_id))
        throw std::runtime_error(
            "v4_fork_sequence: destination seq state already exists");
    const auto& src = it->second;
    V4SeqState st;
    st.block.assign(static_cast<size_t>(dcp_size_), nullptr);
    if (!src.snap.empty())
        st.snap.assign(static_cast<size_t>(dcp_size_), nullptr);
    auto fail_alloc = [&]() {
        for (int q = 0; q < dcp_size_; ++q) {
            opts_.attention_devices[q]->set_device();
            if (q < static_cast<int>(st.block.size()) && st.block[q])
                opts_.attention_devices[q]->device_free(st.block[q]);
            if (q < static_cast<int>(st.snap.size()) && st.snap[q])
                opts_.attention_devices[q]->device_free(st.snap[q]);
        }
        throw std::runtime_error(
            "v4_fork_sequence: state-ring allocation failed");
    };
    for (int q = 0; q < dcp_size_; ++q) {
        auto* dev = opts_.attention_devices[q];
        dev->set_device();
        auto copy = [&](void* dst, const void* srcp, int64_t bytes) {
            dev->memcpy_2d_d2d_async(dst, static_cast<size_t>(bytes), srcp,
                                     static_cast<size_t>(bytes),
                                     static_cast<size_t>(bytes), 1,
                                     attn_streams_[q]);
        };
        st.block[static_cast<size_t>(q)] =
            dev->device_alloc(static_cast<size_t>(v4_ring_bytes_per_seq_));
        if (!st.block[static_cast<size_t>(q)]) fail_alloc();
        copy(st.block[static_cast<size_t>(q)],
             src.block[static_cast<size_t>(q)], v4_ring_bytes_per_seq_);
        if (!st.snap.empty()) {
            st.snap[static_cast<size_t>(q)] = dev->device_alloc(
                static_cast<size_t>(v4_snap_bytes_per_seq_));
            if (!st.snap[static_cast<size_t>(q)]) fail_alloc();
            copy(st.snap[static_cast<size_t>(q)],
                 src.snap[static_cast<size_t>(q)], v4_snap_bytes_per_seq_);
        }
    }
    st.last_pos = src.last_pos;
    st.step_lo = src.step_lo;
    st.step_hi = src.step_hi;
    st.prev_lo = src.prev_lo;
    st.prev_hi = src.prev_hi;
    st.step_serial = src.step_serial;
    st.step_rewind = src.step_rewind;
    st.step_snapshotted = src.step_snapshotted;
    st.layer_snap_serial = src.layer_snap_serial;
    v4_seq_state_.emplace(dst_id, std::move(st));
}

void DcpExecutor::v4_free_sequence(uint64_t seq_id) {
    auto it = v4_seq_state_.find(seq_id);
    if (it == v4_seq_state_.end()) return;
    for (int q = 0; q < dcp_size_
             && static_cast<size_t>(q) < opts_.attention_devices.size(); ++q) {
        opts_.attention_devices[q]->set_device();
        if (static_cast<size_t>(q) < it->second.block.size()
            && it->second.block[static_cast<size_t>(q)])
            opts_.attention_devices[q]->device_free(
                it->second.block[static_cast<size_t>(q)]);
        if (static_cast<size_t>(q) < it->second.snap.size()
            && it->second.snap[static_cast<size_t>(q)])
            opts_.attention_devices[q]->device_free(
                it->second.snap[static_cast<size_t>(q)]);
    }
    v4_seq_state_.erase(it);
}

void DcpExecutor::v4_spec_layer_guard(const AttentionExecParams& params,
                                      V4SeqState& st) {
    // Ticket J: rewind-lossless speculation. The pos%capacity ring slots
    // (SWA tier entries, compressor/LID state rings) alias across a window
    // boundary, so a speculative row at p+k destroys the committed slot of
    // p+k−capacity — which a post-rewind SWA window / boundary compress
    // still needs. Per (seq, layer, step): (1) on a rewind, restore the
    // not-committed tail [step_lo, prev_hi] from the previous step's
    // snapshot; (2) snapshot the slots this step's rows [step_lo, step_hi]
    // are about to overwrite. Compressed CSA/HCA/LID ENTRIES need no
    // snapshot: they are block-index-addressed, invisible beyond
    // (pos+1)/ratio, and recomputed from restored ring states on re-cross.
    if (!opts_.v4.spec_snapshots || st.snap.empty()) return;
    // TD-V4-CHUNK-PREFILL: prefill-chunk windows exceed the ring-clamped
    // snapshot capacity — skip both restore and snapshot (the window is
    // monotone; rewinding INTO it is rejected at the window-tracking gate).
    if (!st.step_snapshotted) return;
    const int layer = params.layer_idx;
    if (st.layer_snap_serial[static_cast<size_t>(layer)] == st.step_serial)
        return;
    st.layer_snap_serial[static_cast<size_t>(layer)] = st.step_serial;

    const auto& so = v4_snap_off_[static_cast<size_t>(layer)];
    const auto& ro = v4_ring_off_[static_cast<size_t>(layer)];
    const auto& step = *params.v4;
    // SWA tier is ALWAYS FP8 (all TQ arms) — deps V4CacheLayout.
    constexpr int64_t kEntry = memory::kV4Fp8EntryBytes;
    const int W = opts_.v4.swa_page_tokens;
    const int64_t rb = static_cast<int64_t>(ro.dim) * 2;
    const int64_t lb = static_cast<int64_t>(ro.lid_dim) * 2;

    // V4-2c: restore/snapshot EVERY rank's replicated tier + rings.
    for (int r = 0; r < dcp_size_; ++r) {
        auto* attn = opts_.attention_devices[r];
        attn->set_device();
        void* stream = attn_streams_[r];
        const auto& rk = step.ranks[r];
        auto* snap = static_cast<uint8_t*>(st.snap[static_cast<size_t>(r)]);
        auto* swa_base = static_cast<uint8_t*>(opts_.v4.swa_base[r]);
        auto* ring = static_cast<uint8_t*>(st.block[static_cast<size_t>(r)]);
        if (rk.swa_page_idx < 0) {
            throw std::runtime_error(
                "v4_spec_layer_guard: SWA ring page missing for layer "
                + std::to_string(layer));
        }

        auto swa_slot = [&](int q) -> uint8_t* {
            return swa_base +
                   (static_cast<int64_t>(rk.swa_page_idx) * W + q % W)
                       * kEntry;
        };
        auto copy = [&](void* dst, const void* src, int64_t bytes) {
            attn->memcpy_2d_d2d_async(dst, static_cast<size_t>(bytes), src,
                                      static_cast<size_t>(bytes),
                                      static_cast<size_t>(bytes), 1, stream);
        };

        if (st.step_rewind) {
            for (int q = st.step_lo; q <= st.prev_hi; ++q) {
                const int64_t srow = q - st.prev_lo;
                if (srow < 0) continue;  // guarded upstream (p0 >= prev_lo)
                copy(swa_slot(q), snap + so.swa + srow * kEntry, kEntry);
                if (ro.capacity > 0) {
                    copy(ring + ro.kv + (q % ro.capacity) * rb,
                         snap + so.kv + srow * rb, rb);
                    copy(ring + ro.score + (q % ro.capacity) * rb,
                         snap + so.score + srow * rb, rb);
                }
                if (ro.lid_capacity > 0) {
                    copy(ring + ro.lid_kv + (q % ro.lid_capacity) * lb,
                         snap + so.lid_kv + srow * lb, lb);
                    copy(ring + ro.lid_score + (q % ro.lid_capacity) * lb,
                         snap + so.lid_score + srow * lb, lb);
                }
            }
        }
        for (int q = st.step_lo; q <= st.step_hi; ++q) {
            const int64_t srow = q - st.step_lo;
            copy(snap + so.swa + srow * kEntry, swa_slot(q), kEntry);
            if (ro.capacity > 0) {
                copy(snap + so.kv + srow * rb,
                     ring + ro.kv + (q % ro.capacity) * rb, rb);
                copy(snap + so.score + srow * rb,
                     ring + ro.score + (q % ro.capacity) * rb, rb);
            }
            if (ro.lid_capacity > 0) {
                copy(snap + so.lid_kv + srow * lb,
                     ring + ro.lid_kv + (q % ro.lid_capacity) * lb, lb);
                copy(snap + so.lid_score + srow * lb,
                     ring + ro.lid_score + (q % ro.lid_capacity) * lb, lb);
            }
        }
    }
}

void DcpExecutor::execute_attention_v4(const AttentionExecParams& params) {
    // ── Shape + config guards (fail loud, never silently wrong) ──────────
    if (!opts_.v4.enabled) {
        throw std::runtime_error(
            "execute_attention_v4: Options.v4 not configured (engine arms it "
            "only for deepseek_v4 + csa_hca)");
    }
    if (!params.v4 || params.v4->seq_id == 0 || params.v4->token_pos < 0) {
        throw std::runtime_error(
            "execute_attention_v4: missing per-step V4 fields (dispatcher "
            "must fill AttentionExecParams::v4)");
    }
    // Multi-row shapes (same-seq consecutive positions, internal per-row
    // loop): dspark verify chunks (ticket J, rewind-snapshotted, ring-
    // clamped) AND chunked prefill (TD-V4-CHUNK-PREFILL lift 2026-08-21 —
    // monotone windows only, bounded by the row scratch). Graph shapes stay
    // fail-closed (TD-DECODE-GRAPH).
    const int n_rows = params.batch_size;
    const int max_rows = std::max(v4_prefill_rows_max_, 1);
    if (n_rows < 1 || n_rows > max_rows || params.use_graph
        || (params.chunk_len > 0 && params.chunk_len != n_rows)) {
        throw std::runtime_error(
            "execute_attention_v4: rows must be 1..max(max_batch, "
            "superchunk_tokens) with chunk_len == rows; graph shapes are "
            "fail-closed (TD-DECODE-GRAPH)");
    }
    if (n_rows > 1 && params.chunk_len != n_rows) {
        throw std::runtime_error(
            "execute_attention_v4: multi-row steps must be single-sequence "
            "chunks (chunk_len == rows)");
    }
    if (params.v4->num_ranks != dcp_size_ || !params.v4->ranks) {
        throw std::runtime_error(
            "execute_attention_v4: per-rank V4 tier metadata missing or "
            "rank count mismatch (dispatcher must fill V4Step::ranks for "
            "every TP rank)");
    }
    if (opts_.v4.sliding_window != opts_.v4.swa_page_tokens) {
        throw std::runtime_error(
            "execute_attention_v4: single-ring-page SWA scheme requires "
            "sliding_window == swa_page_tokens");
    }
    if (!params.kv_cache_ptrs || !params.kv_cache_ptrs[0]) {
        throw std::runtime_error(
            "execute_attention_v4: kv_cache_ptrs (CSA/kMain tier base) "
            "missing");
    }
    if (!params.seqlens_k || !params.seqlens_k[0]) {
        throw std::runtime_error("execute_attention_v4: seqlens_k missing");
    }

    const int layer = params.layer_idx;
    const auto& v4o = opts_.v4;
    if (layer < 0 || layer >= static_cast<int>(v4o.attn_type.size())) {
        throw std::runtime_error("execute_attention_v4: layer "
                                 + std::to_string(layer)
                                 + " outside the V4 attention-type table");
    }
    const auto& step = *params.v4;
    const int p0 = step.token_pos;
    const int p1 = p0 + n_rows - 1;

    // ── Per-seq state blocks (lazy alloc, ONE PER RANK) + step tracking ──
    auto& st = v4_seq_state_[step.seq_id];
    if (st.block.empty()) {
        if (p0 != 0) {
            v4_seq_state_.erase(step.seq_id);
            throw std::runtime_error(
                "execute_attention_v4: sequence " + std::to_string(step.seq_id)
                + " first seen at pos " + std::to_string(p0)
                + " — compressor state rings require processing from pos 0 "
                  "(restore/mid-stream entry is fail-closed)");
        }
        // V4-2c: every rank keeps replicated rings/snapshots on its GPU.
        st.block.assign(static_cast<size_t>(dcp_size_), nullptr);
        if (opts_.v4.spec_snapshots && v4_snap_bytes_per_seq_ > 0)
            st.snap.assign(static_cast<size_t>(dcp_size_), nullptr);
        auto fail_alloc = [&](const char* what) {
            for (int q = 0; q < dcp_size_; ++q) {
                opts_.attention_devices[q]->set_device();
                if (q < static_cast<int>(st.block.size()) && st.block[q])
                    opts_.attention_devices[q]->device_free(st.block[q]);
                if (q < static_cast<int>(st.snap.size()) && st.snap[q])
                    opts_.attention_devices[q]->device_free(st.snap[q]);
            }
            v4_seq_state_.erase(step.seq_id);
            throw std::runtime_error(std::string(
                "execute_attention_v4: ") + what + " allocation failed");
        };
        for (int q = 0; q < dcp_size_; ++q) {
            auto* dev = opts_.attention_devices[q];
            dev->set_device();
            st.block[static_cast<size_t>(q)] = dev->device_alloc(
                static_cast<size_t>(v4_ring_bytes_per_seq_));
            if (!st.block[static_cast<size_t>(q)]) fail_alloc("state-ring");
            if (!st.snap.empty()) {
                st.snap[static_cast<size_t>(q)] = dev->device_alloc(
                    static_cast<size_t>(v4_snap_bytes_per_seq_));
                if (!st.snap[static_cast<size_t>(q)])
                    fail_alloc("spec-snapshot");
            }
            // Ticket J determinism: device_alloc REUSES freed blocks — a
            // new sequence must not inherit the previous holder's ring/
            // snapshot residue.
            if (static_cast<size_t>(q) < opts_.device_backends.size()
                && opts_.device_backends[q]) {
                auto* be = opts_.device_backends[q];
                be->memset_async(st.block[static_cast<size_t>(q)], 0,
                                 static_cast<size_t>(v4_ring_bytes_per_seq_),
                                 attn_streams_[q]);
                if (!st.snap.empty())
                    be->memset_async(
                        st.snap[static_cast<size_t>(q)], 0,
                        static_cast<size_t>(v4_snap_bytes_per_seq_),
                        attn_streams_[q]);
            }
        }
        st.last_pos = -1;
        st.layer_snap_serial.assign(opts_.v4.attn_type.size(), -1);
    }
    if (!(p0 == st.step_lo && p1 == st.step_hi)) {
        // SC (superchunk port): a superchunk LAYER SWEEP re-dispatches every
        // sub-chunk window at each layer — layer L > 0 revisits windows the
        // frontier layer already advanced past. Accept those as REPLAYS
        // without touching the window tracking: per-layer rings/tiers see
        // monotone positions within their own layer, superchunk windows are
        // never snapshotted (the spec guard early-outs), and the tracked
        // window keeps the frontier so post-superchunk decode resumes at
        // step_hi + 1.
        const bool sc_replay = params.superchunk && st.step_hi >= 0
            && p1 <= st.step_hi;
        if (!sc_replay) {
        // First layer of a NEW step window. Ticket J rewind rule: a window
        // starting at p0 <= step_hi is a speculative RE-FEED (post-verify
        // rejection) — legal only with snapshots armed, only back to the
        // previous window's start, and only into a window that WAS
        // snapshotted (TD-V4-CHUNK-PREFILL: prefill chunks longer than the
        // ring-clamped verify bound skip snapshotting — they are monotone
        // by construction and never rewound into).
        if (p0 == st.step_hi + 1) {
            st.step_rewind = false;
        } else if (opts_.v4.spec_snapshots && st.step_snapshotted
                   && st.step_hi >= 0
                   && p0 >= st.step_lo && p0 <= st.step_hi) {
            st.step_rewind = true;
        } else {
            throw std::runtime_error(
                "execute_attention_v4: position discontinuity for seq "
                + std::to_string(step.seq_id) + " (window ["
                + std::to_string(p0) + ", " + std::to_string(p1)
                + "], previous [" + std::to_string(st.step_lo) + ", "
                + std::to_string(st.step_hi)
                + "]) — rewinds require armed spec snapshots, may not reach "
                  "behind the previous window's start, and may not enter an "
                  "un-snapshotted (prefill-chunk/superchunk) window");
        }
        st.prev_lo = st.step_lo;
        st.prev_hi = st.step_hi;
        st.step_lo = p0;
        st.step_hi = p1;
        st.last_pos = p1;
        // Snapshot coverage of THIS window: armed snapshots + rows within
        // the ring-clamped bound (prefill chunks exceed it and are skipped
        // by v4_spec_layer_guard). SC: superchunk windows NEVER snapshot —
        // the layer sweep would guard against a stale tracked window.
        st.step_snapshotted = opts_.v4.spec_snapshots && !st.snap.empty()
            && n_rows <= v4_spec_rows_max_ && !params.superchunk;
        ++st.step_serial;
        }
    }

    // Per-(layer, step) rewind restore + snapshot (all ranks), then the
    // pipeline body. SC (superchunk port, TD-V4-PREFILL-PERF a): prefill
    // chunk windows beyond the snapshot bound take the BATCH-shaped body
    // (one batched kernel chain + ONE decode-kernel attention call per
    // layer) at dcp_size 1 with unpadded head tiles; decode, verify
    // (snapshotted) and TP>=2 shapes keep the exact per-row loop.
    // LS_V4_ROW_PREFILL=1 forces the per-row loop (bisect arm).
    v4_spec_layer_guard(params, st);
    static const bool force_row_prefill = [] {
        const char* p = std::getenv("LS_V4_ROW_PREFILL");
        return p && p[0] == '1';
    }();
    const bool batch_body = n_rows > v4_spec_rows_max_ && dcp_size_ == 1
        && num_heads_local_ == v4_hq_pad_ && !force_row_prefill;
    if (batch_body) {
        execute_attention_v4_chunk(params, st);
    } else {
        for (int row = 0; row < n_rows; ++row)
            for (int r = 0; r < dcp_size_; ++r)
                execute_attention_v4_row(params, row, st, r);
    }

    // V4-2c step 14: the grouped o_proj stage-2 partials sum across ranks —
    // ONE allreduce per layer call over the [n_rows, hidden] output block.
    if (dcp_size_ >= 2 && opts_.dcp_wrapper) {
        opts_.dcp_wrapper->reduce_hidden(hidden_out_.data(), n_rows,
                                         attn_streams_.data());
    }

    // TD-V4-CHUNK-PREFILL (ticket-J S5 staging discipline): a chunk larger
    // than the spec row bound can lap the host staging slot ring while its
    // pageable async H2Ds are still in flight — drain the device before
    // returning so the in-flight window never exceeds one call. Prefill-
    // chunk-only (decode and verify shapes are unchanged); cost is noise
    // against the chunk's expert-streaming wall.
    if (n_rows > v4_spec_rows_max_ && !opts_.device_backends.empty()
        && opts_.device_backends[0]) {
        opts_.device_backends[0]->device_sync();
    }
}

void DcpExecutor::execute_attention_v4_row(const AttentionExecParams& params,
                                           int row, V4SeqState& st, int r) {
    const auto& v4o = opts_.v4;
    const int layer = params.layer_idx;
    const uint8_t atype = v4o.attn_type[static_cast<size_t>(layer)];
    const auto& step = *params.v4;
    const auto& rk = step.ranks[r];   // V4-2c: this rank's tier metadata
    const int pos = step.token_pos + row;
    const int B = 1;
    const int H = opts_.hidden_size;
    const int Q = opts_.q_lora_rank;
    const int HL = num_heads_local_;
    const int D = opts_.v4_head_dim;        // 512
    const int R = opts_.qk_rope_head_dim;   // 64
    const int NIH = opts_.index_n_heads;
    const int IHD = opts_.index_head_dim;

    const auto& w = *params.weights[r];
    auto* attn = opts_.attention_devices[r];
    attn->set_device();
    void* stream = attn_streams_[r];
    const auto& ro = v4_ring_off_[static_cast<size_t>(layer)];
    auto ring_at = [&](int64_t off) -> void* {
        return static_cast<uint8_t*>(st.block[static_cast<size_t>(r)]) + off;
    };

    // Per-layer rope table: compressed layers (CSA/HCA) use the
    // compress-theta table for EVERYTHING in the layer (q/kv/compress/
    // indexer/inverse — the dual-rope rule); SWA-only layers use base theta.
    const void* table = (atype != 0)
        ? rope_cos_sin_compress_[r] : rope_cos_sin_[r];
    if (!table) {
        throw std::runtime_error(
            "execute_attention_v4: missing rope table for layer "
            + std::to_string(layer));
    }

    // Ticket-J diagnostic (LS_V4_STAGE_DUMP=<path>, off by default): append
    // {tag4cc, layer, pos, bytes} + raw device bytes after each pipeline
    // stage — bisects WHICH stage first diverges between two probe runs.
    // Debug seam only: syncs the device around every D2H.
    static FILE* stage_dump_f = [] {
        const char* p = std::getenv("LS_V4_STAGE_DUMP");
        return p ? std::fopen(p, "ab") : nullptr;
    }();
    // Two capture modes: default = LOW-PERTURBATION (stream-ordered async
    // D2H into a preallocated pinned-free host arena, flushed with ONE
    // device_sync at the next step boundary — keeps cross-stream overlap
    // intact so timing races stay reproducible under observation);
    // LS_V4_STAGE_SYNC=1 = fully synchronous per record (serializes the
    // device — heisenbug check). LS_V4_STAGE_TAGS=CSV filters stages.
    static const bool stage_sync_mode = [] {
        const char* p = std::getenv("LS_V4_STAGE_SYNC");
        return p && p[0] == '1';
    }();
    static const char* stage_tag_filter = std::getenv("LS_V4_STAGE_TAGS");
    struct StageRec { char tag[4]; int layer, pos; uint32_t bytes;
                      size_t off; };
    static std::vector<uint8_t> stage_arena;
    static std::vector<StageRec> stage_recs;
    static size_t stage_used = 0;
    static int stage_last_flushed_pos = -1;
    auto stage_flush = [&]() {
        if (!stage_dump_f || stage_recs.empty()) return;
        opts_.device_backends[r]->device_sync();
        for (const auto& rec : stage_recs) {
            int hdr[3] = {0, rec.layer, rec.pos};
            std::memcpy(hdr, rec.tag, 4);
            std::fwrite(hdr, sizeof(hdr), 1, stage_dump_f);
            std::fwrite(&rec.bytes, sizeof(uint32_t), 1, stage_dump_f);
            std::fwrite(stage_arena.data() + rec.off, 1, rec.bytes,
                        stage_dump_f);
        }
        std::fflush(stage_dump_f);
        stage_recs.clear();
        stage_used = 0;
    };
    if (stage_dump_f && pos != stage_last_flushed_pos) {
        stage_flush();
        stage_last_flushed_pos = pos;
    }
    auto stage_dump = [&](const char tag[5], const void* dev, size_t bytes) {
        if (!stage_dump_f || !dev || bytes == 0) return;
        if (opts_.device_backends.empty() || !opts_.device_backends[r]) return;
        if (stage_tag_filter && !std::strstr(stage_tag_filter, tag)) return;
        auto* be = opts_.device_backends[r];
        if (stage_sync_mode) {
            static std::vector<uint8_t> hbuf;
            hbuf.resize(bytes);
            be->device_sync();
            be->memcpy_d2h_async(hbuf.data(), dev, bytes, stream);
            be->device_sync();
            int hdr[3] = {0, layer, pos};
            std::memcpy(hdr, tag, 4);
            std::fwrite(hdr, sizeof(hdr), 1, stage_dump_f);
            std::fwrite(&bytes, sizeof(uint32_t), 1, stage_dump_f);
            std::fwrite(hbuf.data(), 1, bytes, stage_dump_f);
            std::fflush(stage_dump_f);
            return;
        }
        if (stage_arena.empty())
            stage_arena.resize(size_t(768) << 20);  // fixed: no realloc races
        if (stage_used + bytes > stage_arena.size()) stage_flush();
        be->memcpy_d2h_async(stage_arena.data() + stage_used, dev, bytes,
                             stream);
        StageRec rec{};
        std::memcpy(rec.tag, tag, 4);
        rec.layer = layer; rec.pos = pos;
        rec.bytes = static_cast<uint32_t>(bytes);
        rec.off = stage_used;
        stage_recs.push_back(rec);
        stage_used += bytes;
    };

    // ── Small-int staging (pageable host → device, driver-staged) ────────
    // Layout: [0,B) positions | [B,2B) swa slots | [2B,3B) swa block table |
    // [3B,4B) swa seqlens | [4B,5B) row_num_blocks | [5B,5B+8) comp slots.
    const int W = v4o.sliding_window;
    if (rk.swa_page_idx < 0) {
        throw std::runtime_error(
            "execute_attention_v4: SWA ring page missing for layer "
            + std::to_string(layer));
    }
    // Ticket J determinism (S5): claim a FRESH staging slot for this row
    // call. The async H2Ds below may read the host source at
    // stream-execution time (pageable-async is not a synchronous-staging
    // contract on HMM drivers); reusing one buffer per layer let a copy for
    // layer L pick up layer L+2k's ints/page-ids depending on GPU lag —
    // the run-to-run V4 decode drift (first surfacing mid-sweep, layer 30).
    const size_t slot_i = static_cast<size_t>(
        v4_staging_next_++ % static_cast<uint64_t>(v4_staging_slots_));
    int* hi = v4_host_ints_.data()
            + slot_i * static_cast<size_t>(v4_ints_stride_);
    int* hpt = v4_host_pt_.data()
             + slot_i * static_cast<size_t>(std::max(v4_max_pages_, 1));
    const void** hlid = v4_host_lid_ptrs_.data()
        + slot_i * static_cast<size_t>(std::max(v4_max_lid_pages_, 1));
    hi[0] = pos;
    hi[static_cast<size_t>(B)] = rk.swa_page_idx * v4o.swa_page_tokens
                               + pos % W;
    hi[static_cast<size_t>(2 * B)] = rk.swa_page_idx;
    hi[static_cast<size_t>(3 * B)] = std::min(pos + 1, W);
    const int n_vis_csa = (pos + 1) / v4o.csa_ratio;   // completed CSA blocks
    const int n_vis_hca = (pos + 1) / v4o.hca_ratio;   // completed HCA blocks
    hi[static_cast<size_t>(4 * B)] = n_vis_csa;
    // Stride-boundary compress: block j completes when (pos+1) % stride == 0.
    const int stride = (atype == 1) ? v4o.csa_ratio
                     : (atype == 2) ? v4o.hca_ratio : 0;
    const bool boundary = stride > 0 && (pos + 1) % stride == 0;
    const int block_j = boundary ? (pos + 1) / stride - 1 : -1;
    int comp_slot = -1, lid_slot = -1;
    if (boundary) {
        if (atype == 1) {
            const int epp = v4o.csa_entries_per_page;
            const int lp = block_j / epp;
            if (!rk.host_csa_bt || lp >= rk.host_csa_bt_len) {
                throw std::runtime_error(
                    "execute_attention_v4: CSA block table too short (layer "
                    + std::to_string(layer) + ", block "
                    + std::to_string(block_j) + ")");
            }
            comp_slot = rk.host_csa_bt[lp] * epp + block_j % epp;
            const int lepp = v4o.idx_entries_per_page;
            if (lepp <= 0 || !rk.lid_page_ids
                || block_j / lepp >= rk.lid_page_count) {
                throw std::runtime_error(
                    "execute_attention_v4: LID pages missing for layer "
                    + std::to_string(layer));
            }
            lid_slot = rk.lid_page_ids[block_j / lepp] * lepp
                     + block_j % lepp;
        } else {
            const int epp = v4o.hca_entries_per_page;
            if (!rk.hca_page_ids || block_j / epp >= rk.hca_page_count) {
                throw std::runtime_error(
                    "execute_attention_v4: HCA pages missing for layer "
                    + std::to_string(layer));
            }
            comp_slot = rk.hca_page_ids[block_j / epp] * epp
                      + block_j % epp;
        }
    }
    hi[static_cast<size_t>(5 * B)] = comp_slot;
    hi[static_cast<size_t>(5 * B) + 1] = lid_slot;
    // Ticket J: per-ROW rope position for the lightning-q rotation
    // (rope_rotate reads seqlens_k[token]-1; params.seqlens_k carries the
    // batch-row-0 metadata, wrong for verify-chunk rows > 0).
    hi[static_cast<size_t>(5 * B) + 2] = pos + 1;
    int* di = v4_ints_dev_[r];
    attn->memcpy_h2d_async(di, hi,
                           static_cast<size_t>(v4_ints_stride_) * sizeof(int),
                           stream);
    const int* d_pos     = di;
    const int* d_swa_sl  = di + B;       // append slot
    const int* d_swa_bt  = di + 2 * B;   // block table (stride 1)
    const int* d_swa_len = di + 3 * B;
    const int* d_row_nb  = di + 4 * B;
    const int* d_comp    = di + 5 * B;
    const int* d_lid     = di + 5 * B + 1;
    const int* d_seql    = di + 5 * B + 2;  // ticket J: this ROW's pos+1

    // ── Projections (all BF16 — batched_gemm_bf16, batch_count 1) ────────
    auto v4_gemm = [&](const void* weight, int n_out, int k_in,
                       const void* act, void* out) {
        compute::StridedBatchedGemmBf16Params g{};
        g.m = n_out;  g.n = B;  g.k = k_in;
        g.A = weight; g.lda = k_in; g.strideA = 0;
        g.B = act;    g.ldb = k_in; g.strideB = 0;
        g.C = out;    g.ldc = n_out; g.strideC = 0;
        g.batch_count = 1;
        attn->batched_gemm_bf16(g, stream);
    };
    if (!w.q_a_proj || !w.q_b_proj || !w.kv_a_proj || !w.q_a_norm
        || !w.kv_a_norm || !w.input_layernorm) {
        throw std::runtime_error(
            "execute_attention_v4: missing core projection/norm weights for "
            "layer " + std::to_string(layer));
    }

    // 1. attn_norm(x) → normed_hidden (x is the hc_pre-collapsed input;
    //    ticket J: row `row` of the verify chunk).
    const void* x_row = static_cast<const uint8_t*>(params.hidden_states[r])
                        + static_cast<size_t>(row) * H * 2;
    attn->rmsnorm(normed_hidden_[r], x_row,
                  w.input_layernorm, opts_.rms_norm_eps, B, H, H, stream);
    stage_dump("NRMH", normed_hidden_[r], static_cast<size_t>(B) * H * 2);

    // 2. q_a → q_a_norm → q_b → v4_q_prep (parameterless per-head RMS +
    //    interleaved rope + [448|0]/[rope] split).
    v4_gemm(w.q_a_proj, Q, H, normed_hidden_[r], q_compressed_[r]);
    attn->rmsnorm(q_compressed_[r], q_compressed_[r], w.q_a_norm,
                  opts_.rms_norm_eps, B, Q, Q, stream);
    stage_dump("QCMP", q_compressed_[r], static_cast<size_t>(Q) * 2);
    v4_gemm(w.q_b_proj, HL * D, Q, q_compressed_[r], q_heads_[r]);
    stage_dump("QHDS", q_heads_[r], static_cast<size_t>(HL) * D * 2);
    compute::launch_v4_q_prep(v4_q_nope_[r], v4_q_rope_[r], q_heads_[r],
                              d_pos, table, opts_.rms_norm_eps, B, HL, D, R,
                              stream);
    stage_dump("QNOP", v4_q_nope_[r], static_cast<size_t>(HL) * D * 2);
    stage_dump("QROP", v4_q_rope_[r], static_cast<size_t>(HL) * R * 2);

    // 3. kv latent → FULL-512 weighted RMS (V4 norms the pe half too) →
    //    duplicated-rope FP8 append into the SWA ring.
    v4_gemm(w.kv_a_proj, D, H, normed_hidden_[r], kv_compressed_[r]);
    attn->rmsnorm(kv_compressed_[r], kv_compressed_[r], w.kv_a_norm,
                  opts_.rms_norm_eps, B, D, D, stream);
    stage_dump("KVC ", kv_compressed_[r], static_cast<size_t>(D) * 2);
    compute::launch_v4_raw_kv_append(kv_compressed_[r], d_pos, d_swa_sl,
                                     v4o.swa_base[r], table, B, D, R, stream);
    stage_dump("SWAR",
               static_cast<const uint8_t*>(v4o.swa_base[r])
                   + static_cast<size_t>(hi[B])
                         * memory::kV4Fp8EntryBytes,
               memory::kV4Fp8EntryBytes);

    // 4. Compressor states (CSA/HCA) + LID states (CSA): GEMMs from the SAME
    //    normed hidden, ring writes at pos, boundary compress-inserts.
    if (atype != 0) {
        if (!w.compressor_wkv || !w.compressor_wgate || !w.compressor_ape
            || !w.compressor_norm) {
            throw std::runtime_error(
                "execute_attention_v4: compressor weights missing for layer "
                + std::to_string(layer));
        }
        const int sdim = ro.dim;  // 2*D (CSA overlap) or D (HCA)
        v4_gemm(w.compressor_wkv, sdim, H, normed_hidden_[r],
                v4_state_kv_[r]);
        v4_gemm(w.compressor_wgate, sdim, H, normed_hidden_[r],
                v4_state_score_[r]);
        compute::launch_v4_state_ring_write(ring_at(ro.kv), v4_state_kv_[r],
                                            d_pos, ro.capacity, sdim, B,
                                            stream);
        compute::launch_v4_state_ring_write(ring_at(ro.score),
                                            v4_state_score_[r], d_pos,
                                            ro.capacity, sdim, B, stream);
        stage_dump("SKV ", v4_state_kv_[r], static_cast<size_t>(sdim) * 2);
        stage_dump("SSC ", v4_state_score_[r],
                   static_cast<size_t>(sdim) * 2);
        if (atype == 1) {
            if (!w.indexer_compressor_wkv || !w.indexer_compressor_wgate
                || !w.indexer_compressor_ape || !w.indexer_compressor_norm) {
                throw std::runtime_error(
                    "execute_attention_v4: LID compressor weights missing "
                    "for layer " + std::to_string(layer));
            }
            const int ldim = ro.lid_dim;  // 2*IHD
            v4_gemm(w.indexer_compressor_wkv, ldim, H, normed_hidden_[r],
                    v4_lid_kv_[r]);
            v4_gemm(w.indexer_compressor_wgate, ldim, H, normed_hidden_[r],
                    v4_lid_score_[r]);
            compute::launch_v4_state_ring_write(
                ring_at(ro.lid_kv), v4_lid_kv_[r], d_pos, ro.lid_capacity,
                ldim, B, stream);
            compute::launch_v4_state_ring_write(
                ring_at(ro.lid_score), v4_lid_score_[r], d_pos,
                ro.lid_capacity, ldim, B, stream);
        }
        if (boundary) {
            compute::V4CompressArgs ca{};
            ca.kv_state = ring_at(ro.kv);
            ca.score_state = ring_at(ro.score);
            ca.ring_capacity = ro.capacity;
            ca.state_dim = ro.dim;
            ca.overlap = (atype == 1);
            ca.stride = stride;
            ca.ape = w.compressor_ape;
            ca.norm_w = w.compressor_norm;
            ca.cos_sin = table;
            ca.rms_eps = opts_.rms_norm_eps;
            ca.D = D;
            ca.rope_dim = R;
            ca.first_block = block_j;
            ca.num_blocks = 1;
            ca.slots = d_comp;
            ca.kv_cache = (atype == 1) ? params.kv_cache_ptrs[r]
                                       : v4o.hca_base[r];
            // V4-5T codec split: TQ tiers pack 644-B entries via the
            // device bridge (compress → BF16 staging → v4_tq_k_append);
            // FP8 tiers keep the in-kernel entry write.
            if ((atype == 1) ? v4o.csa_tq : v4o.hca_tq) {
                compute::csa_hca_device_tq_compress_insert(attn, ca, layer,
                                                           stream);
            } else {
                ca.out_mode = compute::V4CompressArgs::Out::kFp8Entry;
                compute::launch_v4_compress_insert(ca, stream);
            }
            if (atype == 1) {
                compute::V4CompressArgs li{};
                li.kv_state = ring_at(ro.lid_kv);
                li.score_state = ring_at(ro.lid_score);
                li.ring_capacity = ro.lid_capacity;
                li.state_dim = ro.lid_dim;
                li.overlap = true;
                li.stride = stride;
                li.ape = w.indexer_compressor_ape;
                li.norm_w = w.indexer_compressor_norm;
                li.cos_sin = table;
                li.rms_eps = opts_.rms_norm_eps;
                li.D = IHD;
                li.rope_dim = R;
                li.first_block = block_j;
                li.num_blocks = 1;
                li.slots = d_lid;
                li.out_mode = compute::V4CompressArgs::Out::kIndexerPaged;
                li.idx_pages = v4o.idx_base[r];
                li.idx_page_tokens = v4o.idx_entries_per_page;
                li.idx_page_bytes = v4o.idx_page_bytes;
                compute::launch_v4_compress_insert(li, stream);
            }
        }
    }

    // ── 5. Visibility indices per attention type ─────────────────────────
    const void* comp_cache = nullptr;
    const int* sparse = nullptr;
    int topk = 0;
    if (atype == 1 && n_vis_csa > 0 && n_vis_csa <= v4o.topk) {
        // TD-V4-KVT: IOTA visibility touches EVERY completed block — a
        // tiered sequence must repromote all its cold pages in range
        // before the block-table staging below (the hook updates
        // rk.host_csa_bt in place).
        if (params.v4_tiering
            && params.v4_tiering->has_demotions(step.seq_id, layer)) {
            params.v4_tiering->ensure_hot(step.seq_id, layer, nullptr,
                                          n_vis_csa);
        }
        // Ticket J: when EVERY visible block fits the top-k budget the
        // selection is "all of them" — take the deterministic IOTA path
        // (identical set, ascending order) instead of the lightning select.
        // The top-k kernel's index ORDER is not deterministic across runs,
        // and the FP8 gather/attention reduction order follows the index
        // order — the source of the run-to-run V4 decode drift first seen
        // in the ticket-J probes (pos 9+, deep layers). Semantically exact:
        // top-512 of <= 512 candidates selects all; skips the iq/iw GEMMs
        // and the score/topk kernels entirely.
        const int epp = v4o.csa_entries_per_page;
        const int need_pages = ceildiv(n_vis_csa, epp);
        if (!rk.host_csa_bt || need_pages > rk.host_csa_bt_len
            || need_pages > v4_max_pages_) {
            throw std::runtime_error(
                "execute_attention_v4: CSA block table short for visibility "
                "(layer " + std::to_string(layer) + ")");
        }
        for (int p2 = 0; p2 < need_pages; ++p2)
            hpt[static_cast<size_t>(p2)] = rk.host_csa_bt[p2];
        attn->memcpy_h2d_async(v4_pt_dev_[r], hpt,
                               static_cast<size_t>(need_pages) * sizeof(int),
                               stream);
        topk = ceildiv(n_vis_csa, 64) * 64;
        if (topk > v4_idx_cap_) {
            throw std::runtime_error(
                "execute_attention_v4: CSA visibility exceeds the index "
                "scratch bound");
        }
        compute::launch_v4_slot_translate(
            v4_phys_idx_[r], nullptr, v4_pt_dev_[r], epp, n_vis_csa, topk,
            stream);
        comp_cache = params.kv_cache_ptrs[r];
        sparse = v4_phys_idx_[r];
    } else if (atype == 1 && n_vis_csa > 0) {
        // Lightning select over the LID tier → LOGICAL block ids →
        // physical CSA entry slots (trap #11).
        if (!w.q_idx_b || !w.weights_proj) {
            throw std::runtime_error(
                "execute_attention_v4: lightning indexer weights missing for "
                "layer " + std::to_string(layer));
        }
        const int lepp = v4o.idx_entries_per_page;
        const int need_lid = ceildiv(n_vis_csa, lepp);
        if (need_lid > rk.lid_page_count) {
            throw std::runtime_error(
                "execute_attention_v4: LID pages short for visibility "
                "(layer " + std::to_string(layer) + ")");
        }
        v4_gemm(w.q_idx_b, NIH * IHD, Q, q_compressed_[r], v4_iq_[r]);
        compute::RopeRotateParams rr{};
        rr.x = static_cast<uint8_t*>(v4_iq_[r])
             + static_cast<size_t>(IHD - R) * 2;  // last R dims of head 0
        rr.seqlens_k = d_seql;  // ticket J: per-row position (verify chunks)
        rr.cos_sin = table;
        rr.num_tokens = B;
        rr.rows_per_token = NIH;
        rr.row_stride = IHD;
        rr.d_rope = R;
        rr.max_pos = opts_.rope_max_pos;
        attn->rope_rotate(rr, stream);
        v4_gemm(w.weights_proj, NIH, H, normed_hidden_[r], v4_iw_bf_[r]);
        attn->indexer_scale_weights(
            v4_iw_bf_[r], v4_iw_f32_[r], B, NIH,
            1.0f / std::sqrt(static_cast<float>(IHD) * NIH), stream);
        // Stage the per-row LID page-base table.
        for (int p = 0; p < rk.lid_page_count; ++p)
            hlid[static_cast<size_t>(p)] = rk.lid_page_ptrs[p];
        attn->memcpy_h2d_async(
            v4_lid_ptrs_dev_[r], hlid,
            static_cast<size_t>(std::max(rk.lid_page_count, 1))
                * sizeof(void*),
            stream);
        compute::V4LightningArgs la{};
        la.rows = B;
        la.q_proj = v4_iq_[r];
        la.score_w = v4_iw_f32_[r];
        la.row_num_blocks = d_row_nb;
        la.k_page_table =
            reinterpret_cast<const void* const*>(v4_lid_ptrs_dev_[r]);
        la.page_table_stride = std::max(rk.lid_page_count, 1);
        la.block_endpoints = v4_endpoints_[r];
        la.query_positions = d_pos;
        la.topk = v4o.topk;
        la.indices_out = v4_logical_idx_[r];
        compute::csa_hca_device_lightning_select(attn, la, stream);
        // TD-V4-KVT: selection-driven repromote — read back the selected
        // LOGICAL block ids and make their pages hot before the
        // block-table staging (only when this (seq, layer) actually has
        // demotions; the untiered path is untouched). The sync is the
        // price of exact selection knowledge (v1; INV-KVT-2: placement
        // must never change results).
        if (params.v4_tiering
            && params.v4_tiering->has_demotions(step.seq_id, layer)) {
            v4_tier_ids_host_.resize(static_cast<size_t>(v4o.topk));
            if (!opts_.device_backends.empty()
                && opts_.device_backends[r]) {
                auto* be = opts_.device_backends[r];
                be->memcpy_d2h_async(v4_tier_ids_host_.data(),
                                     v4_logical_idx_[r],
                                     static_cast<size_t>(v4o.topk)
                                         * sizeof(int),
                                     stream);
                be->device_sync();
            }
            params.v4_tiering->ensure_hot(step.seq_id, layer,
                                          v4_tier_ids_host_.data(),
                                          v4o.topk);
        }
        // LOGICAL → PHYSICAL CSA entry slots via the kMain block-table row.
        const int need_pages = ceildiv(n_vis_csa, v4o.csa_entries_per_page);
        if (!rk.host_csa_bt || need_pages > rk.host_csa_bt_len
            || need_pages > v4_max_pages_) {
            throw std::runtime_error(
                "execute_attention_v4: CSA block table short for visibility "
                "(layer " + std::to_string(layer) + ")");
        }
        for (int p = 0; p < need_pages; ++p)
            hpt[static_cast<size_t>(p)] = rk.host_csa_bt[p];
        attn->memcpy_h2d_async(v4_pt_dev_[r], hpt,
                               static_cast<size_t>(need_pages) * sizeof(int),
                               stream);
        compute::launch_v4_slot_translate(
            v4_phys_idx_[r], v4_logical_idx_[r], v4_pt_dev_[r],
            v4o.csa_entries_per_page, n_vis_csa, v4o.topk, stream);
        comp_cache = params.kv_cache_ptrs[r];
        sparse = v4_phys_idx_[r];
        topk = v4o.topk;
    } else if (atype == 2 && n_vis_hca > 0) {
        // Dense over ALL completed HCA blocks: iota → physical entry slots,
        // -1-padded to the kernel's 64-multiple topk contract.
        const int epp = v4o.hca_entries_per_page;
        const int need_pages = ceildiv(n_vis_hca, epp);
        if (!rk.hca_page_ids || need_pages > rk.hca_page_count
            || need_pages > v4_max_pages_) {
            throw std::runtime_error(
                "execute_attention_v4: HCA pages short for visibility "
                "(layer " + std::to_string(layer) + ")");
        }
        for (int p = 0; p < need_pages; ++p)
            hpt[static_cast<size_t>(p)] = rk.hca_page_ids[p];
        attn->memcpy_h2d_async(v4_pt_dev_[r], hpt,
                               static_cast<size_t>(need_pages) * sizeof(int),
                               stream);
        topk = ceildiv(n_vis_hca, 64) * 64;
        if (topk > v4_idx_cap_) {
            throw std::runtime_error(
                "execute_attention_v4: HCA visibility exceeds the index "
                "scratch bound");
        }
        compute::launch_v4_slot_translate(
            v4_phys_idx_[r], nullptr, v4_pt_dev_[r], epp, n_vis_hca, topk,
            stream);
        comp_cache = v4o.hca_base[r];
        sparse = v4_phys_idx_[r];
    }

    // ── 6. Attention (one deps kernel behind all three arms) + epilogues ─
    compute::V4AttentionArgs aa{};
    aa.rows = B;
    aa.q_nope = v4_q_nope_[r];
    aa.q_rope = v4_q_rope_[r];
    aa.comp_cache = comp_cache;
    aa.sparse_indices = sparse;
    aa.topk = topk;
    aa.swa_cache = v4o.swa_base[r];
    aa.swa_block_table = d_swa_bt;
    aa.swa_block_table_stride = 1;
    aa.swa_seqlens = d_swa_len;
    aa.sinks = w.attn_sinks;
    // V4-2c: sinks are replicated full-width — rank r's real heads are the
    // global span [HL*r, HL*(r+1)); epilogues run over the real heads only
    // (pad heads of the 64-tile are never consumed).
    aa.sink_head_offset = HL * r;
    aa.num_heads_real = HL;
    aa.positions = d_pos;
    aa.rope_table = table;   // per-layer table for the inverse rope
    aa.out = v4_attn_out_[r];
    aa.lse = v4_lse_[r];
    // V4-5T: tell the device when THIS call's compressed tier is TQ-coded
    // (CSA → csa_tq, HCA → hca_tq; SWA-only calls carry no comp tier).
    aa.comp_tq = (atype == 1) ? v4o.csa_tq
               : (atype == 2) ? v4o.hca_tq : false;
    aa.layer_idx = layer;
    aa.stream = stream;
    compute::csa_hca_device_attention(attn, aa);
    stage_dump("PIDX", sparse,
               sparse ? static_cast<size_t>(topk) * sizeof(int) : 0);
    stage_dump("ATTO", v4_attn_out_[r], static_cast<size_t>(HL) * D * 2);
    stage_dump("LSE ", v4_lse_[r], static_cast<size_t>(HL) * sizeof(float));

    // ── 7. Grouped o_proj into hidden_out_ row (dispatcher tail = hc_post) ─
    execute_v4_grouped_oproj(
        r, w, v4_attn_out_[r], B,
        static_cast<uint8_t*>(hidden_out_[r]) +
            static_cast<size_t>(row) * H * 2,
        stream);
    stage_dump("OPRJ",
               static_cast<uint8_t*>(hidden_out_[r])
                   + static_cast<size_t>(row) * H * 2,
               static_cast<size_t>(H) * 2);

    if (!v4_active_logged_) {
        v4_active_logged_ = true;
        spdlog::info("DcpExecutor: V4 attention pipeline ACTIVE (layer {}, "
                     "type {}, pos {}, topk {})",
                     layer, atype == 0 ? "swa" : atype == 1 ? "csa" : "hca",
                     pos, topk);
    }
}

// ── SC (superchunk port): TRUE batch-shaped V4 prefill body ────────────────
//
// One batched pipeline per layer call over all chunk rows (TD-V4-PREFILL-
// PERF item a): projections/norms/prep at nt = R, one compressor state
// write + ONE span compress-insert per arm, batched lightning (or per-row
// deterministic IOTA), and ONE decode-kernel attention call with per-row
// -1-padded compressed indices + a per-token SWA index list over the
// chunk's raw-entry staging (swa_page_block_size = 1). Semantics per row
// are identical to the sequential per-row loop (sglang expand_prefill_
// casually visibility: compressed blocks (pos+1)/ratio, SWA min(pos+1, W));
// numerics differ at bf16-lsb (split-KV partitioning + reduction shape) —
// the token-identity golden is the gate, LS_V4_ROW_PREFILL=1 the bisect
// arm. dcp_size 1 + HL == padded-tile only (callers guarantee).
void DcpExecutor::execute_attention_v4_chunk(const AttentionExecParams& params,
                                             V4SeqState& st) {
    const auto& v4o = opts_.v4;
    const int layer = params.layer_idx;
    const uint8_t atype = v4o.attn_type[static_cast<size_t>(layer)];
    const auto& step = *params.v4;
    const int r = 0;
    const auto& rk = step.ranks[r];
    const int R = params.batch_size;
    const int p0 = step.token_pos;
    const int p1 = p0 + R - 1;
    const int H = opts_.hidden_size;
    const int Q = opts_.q_lora_rank;
    const int HL = num_heads_local_;
    const int D = opts_.v4_head_dim;
    const int RD = opts_.qk_rope_head_dim;
    const int NIH = opts_.index_n_heads;
    const int IHD = opts_.index_head_dim;
    const auto& w = *params.weights[r];
    auto* attn = opts_.attention_devices[r];
    attn->set_device();
    void* stream = attn_streams_[r];
    const auto& ro = v4_ring_off_[static_cast<size_t>(layer)];
    auto ring_at = [&](int64_t off) -> void* {
        return static_cast<uint8_t*>(st.block[static_cast<size_t>(r)]) + off;
    };
    const void* table = (atype != 0)
        ? rope_cos_sin_compress_[r] : rope_cos_sin_[r];
    if (!table) {
        throw std::runtime_error(
            "execute_attention_v4_chunk: missing rope table for layer "
            + std::to_string(layer));
    }
    const int W = v4o.sliding_window;
    if (rk.swa_page_idx < 0) {
        throw std::runtime_error(
            "execute_attention_v4_chunk: SWA ring page missing for layer "
            + std::to_string(layer));
    }
    if (!w.q_a_proj || !w.q_b_proj || !w.kv_a_proj || !w.q_a_norm
        || !w.kv_a_norm || !w.input_layernorm) {
        throw std::runtime_error(
            "execute_attention_v4_chunk: missing core projection/norm "
            "weights for layer " + std::to_string(layer));
    }

    // ── Host int staging (fresh batch slot; layout per dcp_executor.h) ───
    const int RA = std::max(v4_prefill_rows_max_, 1);   // allocation stride
    const int NB = v4_batch_nb_max_;
    const int OP = 0, OS = RA, OW = 2 * RA, ON = 3 * RA, OSS = 4 * RA,
              ORS = 5 * RA, OC = 6 * RA, OL = 6 * RA + NB,
              OPX = 6 * RA + 2 * NB, OPT = 6 * RA + 2 * NB + W;
    const size_t slot_i = static_cast<size_t>(
        v4_batch_staging_next_++
        % static_cast<uint64_t>(std::max(v4_batch_staging_slots_, 1)));
    int* hb = v4_batch_host_ints_.data()
            + slot_i * static_cast<size_t>(v4_batch_ints_stride_);
    const int W_pref = std::min(p0, W);
    const int stride = (atype == 1) ? v4o.csa_ratio
                     : (atype == 2) ? v4o.hca_ratio : 0;
    for (int i = 0; i < R; ++i) {
        const int pos = p0 + i;
        hb[OP + i] = pos;
        hb[OS + i] = pos + 1;
        hb[OW + i] = std::min(pos + 1, W);
        hb[ON + i] = (atype == 1) ? (pos + 1) / v4o.csa_ratio
                   : (atype == 2) ? (pos + 1) / v4o.hca_ratio : 0;
        hb[OSS + i] = W_pref + i;    // staging append slot for position pos
    }
    // SWA ring appends: ONLY the chunk's last min(R, W) rows — distinct
    // ring slots (a batched launch must never write one slot twice; the
    // earlier rows' entries are superseded within this very chunk).
    const int ring_rows = std::min(R, W);
    const int ring_row0 = R - ring_rows;
    for (int i = 0; i < ring_rows; ++i) {
        const int pos = p0 + ring_row0 + i;
        hb[ORS + i] = rk.swa_page_idx * v4o.swa_page_tokens + pos % W;
    }
    // Ring-prefix gather slots (positions [p0 - W_pref, p0) — exactly the
    // ring's live window at chunk start).
    for (int i = 0; i < W_pref; ++i) {
        const int pos = p0 - W_pref + i;
        hb[OPX + i] = rk.swa_page_idx * v4o.swa_page_tokens + pos % W;
    }
    // Stride-boundary blocks completing inside [p0, p1]: block j completes
    // at pos (j+1)*stride - 1.
    int first_block = -1, n_boundary = 0;
    if (stride > 0) {
        const int j_min = (p0 + stride) / stride - 1;   // ceil((p0+1)/s) - 1
        const int j_max = (p1 + 1) / stride - 1;
        if (j_max >= j_min && j_max >= 0) {
            first_block = std::max(j_min, 0);
            n_boundary = j_max - first_block + 1;
            if (n_boundary > NB) {
                throw std::runtime_error(
                    "execute_attention_v4_chunk: boundary-block span "
                    "exceeds the staging bound");
            }
            for (int j = first_block; j <= j_max; ++j) {
                const int k = j - first_block;
                if (atype == 1) {
                    const int epp = v4o.csa_entries_per_page;
                    const int lp = j / epp;
                    if (!rk.host_csa_bt || lp >= rk.host_csa_bt_len) {
                        throw std::runtime_error(
                            "execute_attention_v4_chunk: CSA block table "
                            "too short (layer " + std::to_string(layer)
                            + ", block " + std::to_string(j) + ")");
                    }
                    hb[OC + k] = rk.host_csa_bt[lp] * epp + j % epp;
                    const int lepp = v4o.idx_entries_per_page;
                    if (lepp <= 0 || !rk.lid_page_ids
                        || j / lepp >= rk.lid_page_count) {
                        throw std::runtime_error(
                            "execute_attention_v4_chunk: LID pages missing "
                            "for layer " + std::to_string(layer));
                    }
                    hb[OL + k] = rk.lid_page_ids[j / lepp] * lepp + j % lepp;
                } else {
                    const int epp = v4o.hca_entries_per_page;
                    if (!rk.hca_page_ids || j / epp >= rk.hca_page_count) {
                        throw std::runtime_error(
                            "execute_attention_v4_chunk: HCA pages missing "
                            "for layer " + std::to_string(layer));
                    }
                    hb[OC + k] = rk.hca_page_ids[j / epp] * epp + j % epp;
                }
            }
        }
    }
    // CSA/HCA block-table page ids for the logical→physical translation
    // (staged AFTER any tiering repromote below rewrites host_csa_bt —
    // the write happens right before the H2D at the arm sites).
    int* d = v4_batch_ints_dev_[r];
    const int* d_pos = d + OP;
    const int* d_seql = d + OS;
    const int* d_swa_len = d + OW;
    const int* d_row_nb = d + ON;
    const int* d_stage_slots = d + OSS;
    const int* d_ring_slots = d + ORS;
    const int* d_comp = d + OC;
    const int* d_lid = d + OL;
    const int* d_pfx = d + OPX;
    int* d_pt = d + OPT;
    attn->memcpy_h2d_async(
        d, hb, static_cast<size_t>(OPT) * sizeof(int), stream);

    // ── Batched projections (BF16 GEMMs, nt = R) ─────────────────────────
    auto v4_gemm = [&](const void* weight, int n_out, int k_in,
                       const void* act, void* out) {
        compute::StridedBatchedGemmBf16Params g{};
        g.m = n_out;  g.n = R;  g.k = k_in;
        g.A = weight; g.lda = k_in; g.strideA = 0;
        g.B = act;    g.ldb = k_in; g.strideB = 0;
        g.C = out;    g.ldc = n_out; g.strideC = 0;
        g.batch_count = 1;
        attn->batched_gemm_bf16(g, stream);
    };
    attn->rmsnorm(normed_hidden_[r], params.hidden_states[r],
                  w.input_layernorm, opts_.rms_norm_eps, R, H, H, stream);
    v4_gemm(w.q_a_proj, Q, H, normed_hidden_[r], q_compressed_[r]);
    attn->rmsnorm(q_compressed_[r], q_compressed_[r], w.q_a_norm,
                  opts_.rms_norm_eps, R, Q, Q, stream);
    v4_gemm(w.q_b_proj, HL * D, Q, q_compressed_[r], q_heads_[r]);
    compute::launch_v4_q_prep(v4_q_nope_[r], v4_q_rope_[r], q_heads_[r],
                              d_pos, table, opts_.rms_norm_eps, R, HL, D, RD,
                              stream);
    v4_gemm(w.kv_a_proj, D, H, normed_hidden_[r], kv_compressed_[r]);
    attn->rmsnorm(kv_compressed_[r], kv_compressed_[r], w.kv_a_norm,
                  opts_.rms_norm_eps, R, D, D, stream);

    // ── Raw-entry staging: ring-prefix gather → chunk append → ring tail ─
    // Order matters on the stream: the prefix must be gathered BEFORE the
    // ring-tail append rewrites aliased ring slots.
    void* staging_base = nullptr;
    {
        compute::V4EntryGatherArgs ga{};
        ga.src_cache = v4o.swa_base[r];
        ga.slots = d_pfx;
        ga.count = W_pref;
        ga.dst_row_offset = 0;
        compute::csa_hca_device_gather_entries(attn, ga, &staging_base,
                                               stream);
    }
    compute::launch_v4_raw_kv_append(kv_compressed_[r], d_pos, d_stage_slots,
                                     staging_base, table, R, D, RD, stream);
    compute::launch_v4_raw_kv_append(
        static_cast<uint8_t*>(kv_compressed_[r])
            + static_cast<size_t>(ring_row0) * D * 2,
        d_pos + ring_row0, d_ring_slots, v4o.swa_base[r], table, ring_rows,
        D, RD, stream);

    // ── Compressor states + ring writes + span compress-inserts ──────────
    if (atype != 0) {
        if (!w.compressor_wkv || !w.compressor_wgate || !w.compressor_ape
            || !w.compressor_norm) {
            throw std::runtime_error(
                "execute_attention_v4_chunk: compressor weights missing for "
                "layer " + std::to_string(layer));
        }
        const int sdim = ro.dim;
        v4_gemm(w.compressor_wkv, sdim, H, normed_hidden_[r],
                v4_state_kv_[r]);
        v4_gemm(w.compressor_wgate, sdim, H, normed_hidden_[r],
                v4_state_score_[r]);
        compute::launch_v4_state_ring_write(ring_at(ro.kv), v4_state_kv_[r],
                                            d_pos, ro.capacity, sdim, R,
                                            stream);
        compute::launch_v4_state_ring_write(ring_at(ro.score),
                                            v4_state_score_[r], d_pos,
                                            ro.capacity, sdim, R, stream);
        if (atype == 1) {
            if (!w.indexer_compressor_wkv || !w.indexer_compressor_wgate
                || !w.indexer_compressor_ape || !w.indexer_compressor_norm) {
                throw std::runtime_error(
                    "execute_attention_v4_chunk: LID compressor weights "
                    "missing for layer " + std::to_string(layer));
            }
            const int ldim = ro.lid_dim;
            v4_gemm(w.indexer_compressor_wkv, ldim, H, normed_hidden_[r],
                    v4_lid_kv_[r]);
            v4_gemm(w.indexer_compressor_wgate, ldim, H, normed_hidden_[r],
                    v4_lid_score_[r]);
            compute::launch_v4_state_ring_write(
                ring_at(ro.lid_kv), v4_lid_kv_[r], d_pos, ro.lid_capacity,
                ldim, R, stream);
            compute::launch_v4_state_ring_write(
                ring_at(ro.lid_score), v4_lid_score_[r], d_pos,
                ro.lid_capacity, ldim, R, stream);
        }
        if (n_boundary > 0) {
            compute::V4CompressArgs ca{};
            ca.kv_state = ring_at(ro.kv);
            ca.score_state = ring_at(ro.score);
            ca.ring_capacity = ro.capacity;
            ca.state_dim = ro.dim;
            ca.overlap = (atype == 1);
            ca.stride = stride;
            ca.ape = w.compressor_ape;
            ca.norm_w = w.compressor_norm;
            ca.cos_sin = table;
            ca.rms_eps = opts_.rms_norm_eps;
            ca.D = D;
            ca.rope_dim = RD;
            ca.first_block = first_block;
            ca.num_blocks = n_boundary;
            ca.slots = d_comp;
            ca.kv_cache = (atype == 1) ? params.kv_cache_ptrs[r]
                                       : v4o.hca_base[r];
            // V4-5T codec split: TQ tiers pack 644-B entries via the
            // device bridge (compress → BF16 staging → v4_tq_k_append);
            // FP8 tiers keep the in-kernel entry write.
            if ((atype == 1) ? v4o.csa_tq : v4o.hca_tq) {
                compute::csa_hca_device_tq_compress_insert(attn, ca, layer,
                                                           stream);
            } else {
                ca.out_mode = compute::V4CompressArgs::Out::kFp8Entry;
                compute::launch_v4_compress_insert(ca, stream);
            }
            if (atype == 1) {
                compute::V4CompressArgs li{};
                li.kv_state = ring_at(ro.lid_kv);
                li.score_state = ring_at(ro.lid_score);
                li.ring_capacity = ro.lid_capacity;
                li.state_dim = ro.lid_dim;
                li.overlap = true;
                li.stride = stride;
                li.ape = w.indexer_compressor_ape;
                li.norm_w = w.indexer_compressor_norm;
                li.cos_sin = table;
                li.rms_eps = opts_.rms_norm_eps;
                li.D = IHD;
                li.rope_dim = RD;
                li.first_block = first_block;
                li.num_blocks = n_boundary;
                li.slots = d_lid;
                li.out_mode = compute::V4CompressArgs::Out::kIndexerPaged;
                li.idx_pages = v4o.idx_base[r];
                li.idx_page_tokens = v4o.idx_entries_per_page;
                li.idx_page_bytes = v4o.idx_page_bytes;
                compute::launch_v4_compress_insert(li, stream);
            }
        }
    }

    // ── Per-row compressed visibility / selection ────────────────────────
    const void* comp_cache = nullptr;
    const int* sparse = nullptr;
    int topk = 0;
    if (atype == 1 && (p1 + 1) / v4o.csa_ratio > 0) {
        const int n_vis_max = (p1 + 1) / v4o.csa_ratio;
        const int epp = v4o.csa_entries_per_page;
        const bool all_iota = n_vis_max <= v4o.topk;
        if (all_iota) {
            topk = ceildiv(n_vis_max, 64) * 64;
        } else {
            // Batched lightning over the LID tier (shared page table via
            // stride 0), then per-row deterministic IOTA overwrite for
            // rows whose whole visibility fits the top-k budget (the
            // ticket-J rule applied per row).
            if (!w.q_idx_b || !w.weights_proj) {
                throw std::runtime_error(
                    "execute_attention_v4_chunk: lightning indexer weights "
                    "missing for layer " + std::to_string(layer));
            }
            const int lepp = v4o.idx_entries_per_page;
            if (lepp <= 0 || ceildiv(n_vis_max, lepp) > rk.lid_page_count) {
                throw std::runtime_error(
                    "execute_attention_v4_chunk: LID pages short for "
                    "visibility (layer " + std::to_string(layer) + ")");
            }
            v4_gemm(w.q_idx_b, NIH * IHD, Q, q_compressed_[r], v4_iq_[r]);
            compute::RopeRotateParams rr{};
            rr.x = static_cast<uint8_t*>(v4_iq_[r])
                 + static_cast<size_t>(IHD - RD) * 2;
            rr.seqlens_k = d_seql;         // per-row pos+1
            rr.cos_sin = table;
            rr.num_tokens = R;
            rr.rows_per_token = NIH;
            rr.row_stride = IHD;
            rr.d_rope = RD;
            rr.max_pos = opts_.rope_max_pos;
            attn->rope_rotate(rr, stream);
            v4_gemm(w.weights_proj, NIH, H, normed_hidden_[r], v4_iw_bf_[r]);
            attn->indexer_scale_weights(
                v4_iw_bf_[r], v4_iw_f32_[r], R, NIH,
                1.0f / std::sqrt(static_cast<float>(IHD) * NIH), stream);
            const size_t lid_slot = static_cast<size_t>(
                v4_staging_next_++
                % static_cast<uint64_t>(v4_staging_slots_));
            const void** hlid = v4_host_lid_ptrs_.data()
                + lid_slot * static_cast<size_t>(
                                 std::max(v4_max_lid_pages_, 1));
            for (int p = 0; p < rk.lid_page_count; ++p)
                hlid[static_cast<size_t>(p)] = rk.lid_page_ptrs[p];
            attn->memcpy_h2d_async(
                v4_lid_ptrs_dev_[r], hlid,
                static_cast<size_t>(std::max(rk.lid_page_count, 1))
                    * sizeof(void*),
                stream);
            compute::V4LightningArgs la{};
            la.rows = R;
            la.q_proj = v4_iq_[r];
            la.score_w = v4_iw_f32_[r];
            la.row_num_blocks = d_row_nb;
            la.k_page_table =
                reinterpret_cast<const void* const*>(v4_lid_ptrs_dev_[r]);
            la.page_table_stride = 0;   // one shared table for all rows
            la.block_endpoints = v4_endpoints_[r];
            la.query_positions = d_pos;
            la.topk = v4o.topk;
            la.indices_out = v4_logical_idx_[r];
            compute::csa_hca_device_lightning_select(attn, la, stream);
            topk = v4o.topk;
        }
        if (topk > v4_idx_cap_) {
            throw std::runtime_error(
                "execute_attention_v4_chunk: CSA visibility exceeds the "
                "index scratch bound");
        }
        // Per-row logical indices: IOTA rows (deterministic all-visible)
        // merged over the lightning rows (in-place legal).
        compute::launch_v4_prefill_indices(
            v4_logical_idx_[r], all_iota ? nullptr : v4_logical_idx_[r],
            d_row_nb, R, topk, stream);
        // TD-V4-KVT: make every page the gather/attention will touch hot
        // BEFORE staging the block table. IOTA rows touch all visible
        // blocks; lightning rows contribute their selection (host
        // readback — the batch path unions all rows).
        if (params.v4_tiering
            && params.v4_tiering->has_demotions(step.seq_id, layer)) {
            if (all_iota) {
                params.v4_tiering->ensure_hot(step.seq_id, layer, nullptr,
                                              n_vis_max);
            } else {
                const size_t n_ids =
                    static_cast<size_t>(R) * static_cast<size_t>(topk);
                v4_tier_ids_host_.resize(n_ids);
                if (!opts_.device_backends.empty()
                    && opts_.device_backends[r]) {
                    auto* be = opts_.device_backends[r];
                    be->memcpy_d2h_async(v4_tier_ids_host_.data(),
                                         v4_logical_idx_[r],
                                         n_ids * sizeof(int), stream);
                    be->device_sync();
                }
                params.v4_tiering->ensure_hot(
                    step.seq_id, layer, v4_tier_ids_host_.data(),
                    static_cast<int>(n_ids));
            }
        }
        const int need_pages = ceildiv(n_vis_max, epp);
        if (!rk.host_csa_bt || need_pages > rk.host_csa_bt_len
            || need_pages > v4_max_pages_) {
            throw std::runtime_error(
                "execute_attention_v4_chunk: CSA block table short for "
                "visibility (layer " + std::to_string(layer) + ")");
        }
        for (int p = 0; p < need_pages; ++p)
            hb[OPT + p] = rk.host_csa_bt[p];
        attn->memcpy_h2d_async(d_pt, hb + OPT,
                               static_cast<size_t>(need_pages) * sizeof(int),
                               stream);
        compute::launch_v4_slot_translate(
            v4_phys_idx_[r], v4_logical_idx_[r], d_pt, epp, n_vis_max,
            R * topk, stream);
        comp_cache = params.kv_cache_ptrs[r];
        sparse = v4_phys_idx_[r];
    } else if (atype == 2 && (p1 + 1) / v4o.hca_ratio > 0) {
        const int n_vis_max = (p1 + 1) / v4o.hca_ratio;
        const int epp = v4o.hca_entries_per_page;
        const int need_pages = ceildiv(n_vis_max, epp);
        if (!rk.hca_page_ids || need_pages > rk.hca_page_count
            || need_pages > v4_max_pages_) {
            throw std::runtime_error(
                "execute_attention_v4_chunk: HCA pages short for visibility "
                "(layer " + std::to_string(layer) + ")");
        }
        topk = ceildiv(n_vis_max, 64) * 64;
        if (topk > v4_idx_cap_) {
            throw std::runtime_error(
                "execute_attention_v4_chunk: HCA visibility exceeds the "
                "index scratch bound");
        }
        compute::launch_v4_prefill_indices(v4_logical_idx_[r], nullptr,
                                           d_row_nb, R, topk, stream);
        for (int p = 0; p < need_pages; ++p)
            hb[OPT + p] = rk.hca_page_ids[p];
        attn->memcpy_h2d_async(d_pt, hb + OPT,
                               static_cast<size_t>(need_pages) * sizeof(int),
                               stream);
        compute::launch_v4_slot_translate(
            v4_phys_idx_[r], v4_logical_idx_[r], d_pt, epp, n_vis_max,
            R * topk, stream);
        comp_cache = v4o.hca_base[r];
        sparse = v4_phys_idx_[r];
    }

    // ── Per-row SWA index list over the staging + batched attention ──────
    compute::launch_v4_prefill_swa_bt(v4_swa_bt_dev_[r], d_swa_len, R, W,
                                      W_pref, stream);
    compute::V4AttentionArgs aa{};
    aa.rows = R;
    aa.q_nope = v4_q_nope_[r];
    aa.q_rope = v4_q_rope_[r];
    aa.comp_cache = comp_cache;
    aa.sparse_indices = sparse;
    aa.topk = topk;
    aa.swa_cache = staging_base;
    aa.swa_block_table = v4_swa_bt_dev_[r];
    aa.swa_block_table_stride = W;
    aa.swa_seqlens = d_swa_len;
    aa.swa_page_block_size = 1;     // per-token index list (SC)
    aa.sinks = w.attn_sinks;
    aa.sink_head_offset = 0;        // dcp_size 1 (caller-guaranteed)
    aa.num_heads_real = 0;          // HL == h_q tile (caller-guaranteed)
    aa.positions = d_pos;
    aa.rope_table = table;
    aa.out = v4_attn_out_[r];
    aa.lse = v4_lse_[r];
    // V4-5T codec (see the row path).
    aa.comp_tq = (atype == 1) ? v4o.csa_tq
               : (atype == 2) ? v4o.hca_tq : false;
    aa.layer_idx = layer;
    aa.stream = stream;
    compute::csa_hca_device_attention(attn, aa);

    // ── Grouped o_proj over all rows into hidden_out_ [0, R) ─────────────
    execute_v4_grouped_oproj(r, w, v4_attn_out_[r], R, hidden_out_[r],
                             stream);

    if (!v4_batch_logged_) {
        v4_batch_logged_ = true;
        spdlog::info("DcpExecutor: V4 BATCH prefill body ACTIVE (layer {}, "
                     "type {}, rows {}, window [{}, {}], topk {})",
                     layer, atype == 0 ? "swa" : atype == 1 ? "csa" : "hca",
                     R, p0, p1, topk);
    }
}

}  // namespace layerstorm::parallelism

// ── Dispatcher-side V4 phase hooks (arch_deepseek_v4.h; refactor V2 P1) ─────
// Bodies are the verbatim blocks carved from the former dispatch_attention.cpp
// driver — dispatcher members accessed via d_ (friend); byte-identical
// behavior to the inline blocks they replace.

namespace layerstorm::daemon {

bool ArchDeepseekV4::validate_shape(
        const CommandDispatcher::InternalAttentionParams& p, int& batch_cap) {
    // Ticket J + TD-V4-CHUNK-PREFILL lift (2026-08-21) + SC superchunk port:
    // beyond B==1 decode-shaped steps, V4 accepts the single-sequence CHUNK
    // shape — is_prefill with chunk_len == num_seqs (consecutive positions
    // of ONE sequence, validated in execute()). This covers dspark verify
    // chunks (small, rewind-snapshotted), chunked PREFILL, and SUPERCHUNK
    // sub-launches (row_offset > 0 and/or the superchunk flag — the GLM
    // superchunk machinery in the driver already places hc collapse, moe_buf
    // rows, fused-gate topk and the routing export at row_offset; the
    // executor accepts the layer-sweep window replays). Drafts and
    // multi-sequence batches stay fail-closed.
    const bool v4_chunk = p.is_prefill
        && p.chunk_len > 0 && p.chunk_len == p.num_seqs
        && !p.is_draft;
    if (!v4_chunk
        && (p.num_seqs != 1 || p.chunk_len > 0 || p.is_draft
            || p.row_offset != 0 || p.superchunk)) {
        d_.last_internal_error_cat_ =
            ipc::CmpErrorCategory::kComputeValidation;
        d_.last_internal_error_msg_ =
            "attention: deepseek_v4 supports only B==1 decode-shaped steps, "
            "single-sequence chunked prefill / superchunk sub-launches "
            "(chunk_len == num_seqs) and dspark verify chunks; draft/"
            "multi-sequence shapes are fail-closed";
        return false;
    }
    // TD-V4-CHUNK-PREFILL: the V4 chunk shape flows per-row through the
    // executor (which enforces its own row bound = max(max_batch,
    // superchunk_tokens) capped at 512) — allow up to the descriptor bound
    // there; every other shape keeps the max_batch cap (TD-40g).
    batch_cap = v4_chunk
        ? static_cast<int>(ipc::kMaxBatchDescriptors)
        : d_.deps_.max_batch_size;
    return true;
}

bool ArchDeepseekV4::stage_step(
        const CommandDispatcher::InternalAttentionParams& p,
        parallelism::AttentionExecParams& params,
        int batch_size, int layer, int dcp_size, bool kv_meta_ok) {
    (void)batch_size; (void)layer; (void)dcp_size; (void)kv_meta_ok;
    // TD-40j: chunked prefill support. V4: a B==1 is_prefill step (prompt
    // feeding) is decode-shaped for the V4 pipeline — do NOT synthesize the
    // chunk descriptor (chunk shapes are fail-closed there); a verify/
    // prefill CHUNK carries its descriptor in the command.
    if (p.is_prefill && p.chunk_len > 0) {
        params.chunk_start = static_cast<int>(p.chunk_start);
        params.chunk_len   = static_cast<int>(p.chunk_len);
    }
    return true;
}

bool ArchDeepseekV4::execute(
        const CommandDispatcher::InternalAttentionParams& p,
        parallelism::AttentionExecParams& params,
        int batch_size, int layer, int dcp_size, bool kv_meta_ok) {
    (void)batch_size;
    // V4-7b: provision the side tiers, fill the per-step V4 fields and
    // route to the executor's V4 pipeline. Errors surface as command
    // completions (never a daemon crash).
    if (!kv_meta_ok || !d_.deps_.sideband_base) {
        d_.last_internal_error_cat_ =
            ipc::CmpErrorCategory::kComputeValidation;
        d_.last_internal_error_msg_ =
            "attention: deepseek_v4 requires KV metadata (CSA tier block "
            "tables)";
        return false;
    }
    const auto* vbe = reinterpret_cast<const ipc::BatchDescriptorEntry*>(
        d_.deps_.sideband_base + ipc::IpcLayout::kBatchDescriptorOff);
    const uint64_t v4_seq = vbe[0].seq_id;
    const uint32_t v4_pos = vbe[0].token_pos;
    // Ticket J: a verify chunk's rows must be ONE sequence at
    // consecutive positions (the executor loops rows internally); the
    // side tiers provision through the LAST row's position.
    const int v4_rows = static_cast<int>(p.num_seqs);
    for (int b = 1; b < v4_rows; ++b) {
        if (vbe[b].seq_id != v4_seq
            || vbe[b].token_pos != v4_pos + static_cast<uint32_t>(b)) {
            d_.last_internal_error_cat_ =
                ipc::CmpErrorCategory::kComputeValidation;
            d_.last_internal_error_msg_ =
                "attention: V4 verify chunk rows must be one sequence at "
                "consecutive positions";
            return false;
        }
    }
    if (!d_.ensure_v4_tier_pages(
            v4_seq, v4_pos + static_cast<uint32_t>(v4_rows - 1))) {
        d_.last_internal_error_cat_ = ipc::CmpErrorCategory::kKvPoolExhausted;
        d_.last_internal_error_msg_ =
            "attention: V4 side-tier page provisioning failed "
            "(kSwa/kHca/kIndexerK pool exhausted — fail-closed)";
        return false;
    }
    const int L = d_.kv_layers_ > 0 ? d_.kv_layers_ : 1;
    const int lyr = (layer >= 0 && layer < L) ? layer : 0;
    auto sp = d_.sequences_.find(v4_seq);
    const int num_logical = sp != d_.sequences_.end() && L > 0
        ? static_cast<int>(sp->second.kv_pages.size()) / L : 0;
    d_.v4_step_ = {};
    d_.v4_step_.seq_id = v4_seq;
    d_.v4_step_.token_pos = static_cast<int>(v4_pos);
    // V4-2c TP: fill the per-rank tier metadata — rank r's kMain block
    // tables (replicated lockstep, per-rank host rows) + its own
    // side-tier page ids/ptrs.
    const auto& tiers = d_.sequences_[v4_seq].v4_tiers;
    const int n_ranks = dcp_size;
    d_.v4_step_ranks_.assign(static_cast<size_t>(n_ranks), {});
    d_.v4_hca_ids_.resize(static_cast<size_t>(n_ranks));
    d_.v4_lid_ids_.resize(static_cast<size_t>(n_ranks));
    d_.v4_lid_ptrs_.resize(static_cast<size_t>(n_ranks));
    for (int rr = 0; rr < n_ranks; ++rr) {
        auto& rk = d_.v4_step_ranks_[static_cast<size_t>(rr)];
        // Host CSA (kMain) block-table row for this layer, batch row 0
        // — built by build_kv_metadata above (persists between
        // bt_dirty rebuilds).
        rk.host_csa_bt =
            d_.kv_meta_scratch_[static_cast<size_t>(rr)]
                .host_block_tables.data()
            + (static_cast<size_t>(lyr) * batch_size + 0)
                  * d_.max_blocks_per_seq_;
        rk.host_csa_bt_len = std::min(num_logical, d_.max_blocks_per_seq_);
        if (rr < static_cast<int>(tiers.swa.size())
            && layer < static_cast<int>(
                   tiers.swa[static_cast<size_t>(rr)].size()))
            rk.swa_page_idx =
                tiers.swa[static_cast<size_t>(rr)]
                     [static_cast<size_t>(layer)].page_idx;
        if (rr < static_cast<int>(tiers.hca.size())
            && layer < static_cast<int>(
                   tiers.hca[static_cast<size_t>(rr)].size())) {
            const auto& hv = tiers.hca[static_cast<size_t>(rr)]
                                      [static_cast<size_t>(layer)];
            auto& ids = d_.v4_hca_ids_[static_cast<size_t>(rr)];
            ids.resize(hv.size());
            for (size_t i = 0; i < hv.size(); ++i)
                ids[i] = hv[i].page_idx;
            rk.hca_page_ids = ids.data();
            rk.hca_page_count = static_cast<int>(hv.size());
        }
        if (rr < static_cast<int>(tiers.lid.size())
            && layer < static_cast<int>(
                   tiers.lid[static_cast<size_t>(rr)].size())) {
            const auto& lv = tiers.lid[static_cast<size_t>(rr)]
                                      [static_cast<size_t>(layer)];
            auto& ids = d_.v4_lid_ids_[static_cast<size_t>(rr)];
            auto& ptrs = d_.v4_lid_ptrs_[static_cast<size_t>(rr)];
            ids.resize(lv.size());
            ptrs.resize(lv.size());
            for (size_t i = 0; i < lv.size(); ++i) {
                ids[i] = lv[i].page_idx;
                ptrs[i] = lv[i].gpu_ptr;
            }
            rk.lid_page_ids = ids.data();
            rk.lid_page_ptrs = ptrs.data();
            rk.lid_page_count = static_cast<int>(lv.size());
        }
    }
    d_.v4_step_.ranks = d_.v4_step_ranks_.data();
    d_.v4_step_.num_ranks = n_ranks;
    params.v4 = &d_.v4_step_;
    // TD-V4-KVT: arm the CSA-bucket tiering hook for this step — the
    // executor repromotes selected cold pages in place (the manager
    // mutates these SAME host bt rows before the per-row pt upload).
    if (d_.v4_kv_tiering_) {
        std::vector<int*> bt_rows;
        bt_rows.reserve(static_cast<size_t>(n_ranks));
        for (int rr = 0; rr < n_ranks; ++rr) {
            bt_rows.push_back(
                d_.kv_meta_scratch_[static_cast<size_t>(rr)]
                    .host_block_tables.data()
                + (static_cast<size_t>(lyr) * batch_size + 0)
                      * d_.max_blocks_per_seq_);
        }
        d_.v4_kv_tiering_->set_step_bt(v4_seq, layer, std::move(bt_rows),
                                    std::min(num_logical,
                                             d_.max_blocks_per_seq_));
        params.v4_tiering = d_.v4_kv_tiering_.get();
    }
    params.use_graph = false;  // decode graphs: TD-DECODE-GRAPH dormant
    try {
        d_.deps_.dcp_executor->execute_attention_v4(params);
    } catch (const std::exception& e) {
        spdlog::error("dispatch_attention: V4 pipeline failed: {}",
                      e.what());
        d_.last_internal_error_cat_ =
            ipc::CmpErrorCategory::kComputeValidation;
        d_.last_internal_error_msg_ =
            "attention: V4 pipeline failed (see log)";
        return false;
    }
    // TD-V4-KVT: demote sweep — CSA pages fully behind the retention
    // frontier (relative to this step's LAST row) go cold. Runs after
    // the layer's attention so every read of this step saw hot pages.
    if (d_.v4_kv_tiering_) {
        auto sit = d_.sequences_.find(v4_seq);
        if (sit != d_.sequences_.end()) {
            const int Lk = d_.kv_layers_ > 0 ? d_.kv_layers_ : 1;
            const int nlog =
                static_cast<int>(sit->second.kv_pages.size()) / Lk;
            d_.v4_kv_tiering_->after_attention(
                v4_seq, layer,
                static_cast<int>(v4_pos) + v4_rows - 1,
                sit->second.kv_pages, nlog, Lk);
        }
    }
    return true;
}

}  // namespace layerstorm::daemon
