// glm5_next (GLM-5.3-Flash) attention arch — GF3.9 hybrid execution.
//
// Two namespaces, mirroring arch_mla.cpp / arch_deepseek_v4.cpp:
//   layerstorm::parallelism — DcpExecutor::execute_attention_kda, the KDA
//     linear-attention executor leg (input_layernorm → q/k/v/b/f/g
//     projections → fused conv → chunked WY scan or fused O(1) decode →
//     gated RMSNorm → o_proj into hidden_out_);
//   layerstorm::daemon — the three ArchGlm5Next phase hooks (per-layer
//     KDA-vs-sparse-MLA dispatch; sparse layers delegate to the composed
//     ArchMla).
//
// CUDA-free TU (INV-GPU-1): all device work goes through AttentionDevice
// (the GF3.9 kda_* virtuals mirror deps kda_scan.h as CUDA-free PODs).
// Every kernel launch rides the per-rank kAttention stream
// (DcpExecutor::attn_streams_) — the KDA state pool's zero-on-claim and
// fork D2D are ordered on that stream (INV-KDA-STATE (b)/(c)).

#include "daemon/attention/arch_glm5_next.h"

#include "compute/kernels/attention/dcp_attention_wrapper.h"
#include "core/attention_device.h"
#include "core/device_backend.h"
#include "daemon/command_dispatcher.h"
#include "daemon/ipc_protocol.h"
#include "model/model_config.h"
#include "parallelism/dcp_executor.h"
#include "parallelism/dcp_executor_internal.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace layerstorm::parallelism {

namespace {

// LS_KDA_PROJ_FUSED (default ON, P-29 step 15): fuse the six same-input KDA
// projections (q/k/v/b/f_a/g_a, all reading the SAME normed hidden state)
// into ONE activation quantize + ONE multi-segment mmvq launch. Bit-identical
// by construction (kernel-side contract in gguf_mmvq.h; GgufMmvqMulti unit
// suite) — =0 restores the serial 8-launch ladder byte-for-byte, including
// the shared f/g low-rank staging buffer.
bool kda_proj_fused_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("LS_KDA_PROJ_FUSED");
        return !(e && *e && e[0] == '0');
    }();
    return on;
}

}  // namespace

// ── The KDA executor leg ────────────────────────────────────────────────────
//
// One KDA linear layer for the whole step: prefill (chunked WY scan over
// batch_size consecutive rows of ONE sequence — chunk-64 launch boundaries
// are the bitwise state-carry points, INV-KDA-CARRY) or decode (fused O(1)
// step per row via base + slots[b] * stride slot indirection). Writes the
// [B, hidden] BF16 result into hidden_out_[r] rows [0, B) — the driver's
// shared residual / mHC-post tail consumes it exactly like the MLA output.
// THROWS std::runtime_error on wiring/geometry errors (the dispatch loop's
// top-level catch surfaces them as kException CMP_ERRORs).

void DcpExecutor::execute_attention_kda(const AttentionExecParams& params) {
    if (!opts_.kda_enabled)
        throw std::runtime_error(
            "execute_attention_kda: KDA is not enabled on this executor "
            "(non-glm5_next boot?)");
    const auto* kda = params.kda;
    if (!kda || kda->num_ranks != dcp_size_ || !kda->ranks)
        throw std::runtime_error(
            "execute_attention_kda: params.kda not staged "
            "(ArchGlm5Next::stage_step contract)");
    const int B = params.batch_size;
    if (B <= 0 || B > kda_rows_max_)
        throw std::runtime_error(
            "execute_attention_kda: batch " + std::to_string(B)
            + " outside the KDA scratch bound "
            + std::to_string(kda_rows_max_));
    const int Hk = opts_.kda_heads_per_rank;
    const int C = Hk * 128;
    const int H = opts_.hidden_size;
    const int ord = kda->linear_ordinal;
    if (ord < 0)
        throw std::runtime_error(
            "execute_attention_kda: negative linear ordinal");
    // TD-KDA-STATE-MAPPED-SLABS: carve mode addresses layer `ord` at its
    // offset inside the whole-request slot; mapped mode gets a PER-LAYER
    // base/slot (the handle/slot-id already selects the layer unit), so
    // the intra-unit offset starts at 0. Ring offsets are intra-unit in
    // both modes ([recurrent | ring_q | ring_k | ring_v] — the slot's
    // per-layer section layout, verbatim).
    const int64_t rec_off = opts_.kda_mapped
        ? 0
        : static_cast<int64_t>(ord) * opts_.kda_per_layer_bytes;
    const int64_t ring0_off = rec_off + opts_.kda_recurrent_bytes_per_layer;
    const float scale = 1.0f / std::sqrt(128.0f);  // host-computed (GF3.7)

    // LS_TP_HIDDEN_PROBE (TD-GLM5-TP-COMBINE-PRECISION diagnostic): dump the
    // layer INPUT row-0 checksum per rank — bisects TP drift to a stage.
    tp_hidden_probe("kda-in", params.layer_idx, params.hidden_states, B);

    for (int r = 0; r < dcp_size_; ++r) {
        auto* attn = opts_.attention_devices[r];
        attn->set_device();
        void* stream = attn_streams_[r];
        const auto& w = *params.weights[r];
        const auto& kr = kda->ranks[r];

        if (!w.kda_q_proj || !w.kda_k_proj || !w.kda_v_proj || !w.kda_b_proj
            || !w.kda_f_a_proj || !w.kda_f_b_proj || !w.kda_g_a_proj
            || !w.kda_g_b_proj || !w.kda_q_conv1d || !w.kda_k_conv1d
            || !w.kda_v_conv1d || !w.kda_a_log || !w.kda_dt_bias
            || !w.kda_o_norm || !w.o_proj)
            throw std::runtime_error(
                "execute_attention_kda: KDA weights missing for layer "
                + std::to_string(params.layer_idx) + " rank "
                + std::to_string(r)
                + " (engine upload / assign_attn_weight_ptr gap)");

        // Projection helper: C[B, out] BF16 = normed/act [B, k] @ W[out, k]^T.
        auto proj = [&](const void* act, const void* weight, void* out,
                        int out_dim, int k, bool is_gguf,
                        model::GgufKQuantType type, bool c_fp32 = false) {
            if (is_gguf) {
                route_gguf_gemm(attn, r, B, out_dim, k, act, weight, out,
                                type, stream, c_fp32);
            } else {
                compute::StridedBatchedGemmBf16Params g{};
                g.m = out_dim; g.n = B; g.k = k;
                g.A = weight; g.lda = k; g.strideA = 0;
                g.B = act;    g.ldb = k; g.strideB = 0;
                g.C = out;    g.ldc = out_dim; g.strideC = 0;
                g.batch_count = 1;
                g.c_fp32 = c_fp32;
                attn->batched_gemm_bf16(g, stream);
            }
        };

        // q/k/v projections → the three pre-conv sections of kda_x_bf16_.
        const size_t x_sec = static_cast<size_t>(kda_rows_max_) * C * 2;
        auto* x_base = static_cast<char*>(kda_x_bf16_[r]);
        void* x_q = x_base;
        void* x_k = x_base + x_sec;
        void* x_v = x_base + 2 * x_sec;

        // P-29 step 15 (LS_KDA_PROJ_FUSED): the six projections off the SAME
        // normed hidden state (q/k/v/b/f_a/g_a — all K=H, int-strategy mmvq
        // at this M) fuse into ONE quantize + ONE multi-segment launch when
        // every segment is GGUF of one shared quant type. Fused g_a stages
        // into kda_lowrank2_ (f/g chains leave one launch concurrently); the
        // serial fallback keeps today's ladder byte-for-byte, including the
        // kda_lowrank_ reuse.
        void* q8_ws = r < static_cast<int>(gguf_q8_1_ws_.size())
                          ? gguf_q8_1_ws_[r] : nullptr;
        const bool proj_fused =
            kda_proj_fused_enabled()
            && opts_.gguf_strategy == config::GgufStrategy::int_strategy
            && B <= gguf_mmvq_max_m()
            && q8_ws
            && w.kda_q_is_gguf && w.kda_k_is_gguf && w.kda_v_is_gguf
            && w.kda_b_is_gguf && w.kda_f_a_is_gguf && w.kda_g_a_is_gguf
            && w.kda_k_gguf_type == w.kda_q_gguf_type
            && w.kda_v_gguf_type == w.kda_q_gguf_type
            && w.kda_b_gguf_type == w.kda_q_gguf_type
            && w.kda_f_a_gguf_type == w.kda_q_gguf_type
            && w.kda_g_a_gguf_type == w.kda_q_gguf_type;

        // P-29 step 7 (INV-0.6(b) span graph): the norm + projection head of
        // the layer, shared verbatim by the prefill arm (eager) and the
        // decode arm (inside the captured span).
        auto emit_qkv = [&] {
            // input_layernorm(hidden) → normed_hidden [B, H] BF16.
            attn->rmsnorm(normed_hidden_[r], params.hidden_states[r],
                          w.input_layernorm, opts_.rms_norm_eps, B, H,
                          /*row_stride=*/H, stream);
            if (proj_fused) {
                compute::GgufGemmMultiParams mp{};
                mp.M = B; mp.K = H;
                mp.A = normed_hidden_[r];
                mp.type = w.kda_q_gguf_type;
                mp.nseg = 6;
                mp.B[0] = w.kda_q_proj;   mp.C[0] = x_q;            mp.N[0] = C;
                mp.B[1] = w.kda_k_proj;   mp.C[1] = x_k;            mp.N[1] = C;
                mp.B[2] = w.kda_v_proj;   mp.C[2] = x_v;            mp.N[2] = C;
                mp.B[3] = w.kda_b_proj;   mp.C[3] = kda_beta_[r];   mp.N[3] = Hk;
                mp.B[4] = w.kda_f_a_proj; mp.C[4] = kda_lowrank_[r];  mp.N[4] = 128;
                mp.B[5] = w.kda_g_a_proj; mp.C[5] = kda_lowrank2_[r]; mp.N[5] = 128;
                attn->gguf_mmvq_multi(mp, q8_ws, stream);
                // Low-rank tails: raw_g = f_b(f_a(x)), g2 = g_b(g_a(x)).
                proj(kda_lowrank_[r], w.kda_f_b_proj, kda_rawg_[r], C, 128,
                     w.kda_f_b_is_gguf, w.kda_f_b_gguf_type);
                proj(kda_lowrank2_[r], w.kda_g_b_proj, kda_g2_[r], C, 128,
                     w.kda_g_b_is_gguf, w.kda_g_b_gguf_type);
                return;
            }
            proj(normed_hidden_[r], w.kda_q_proj, x_q, C, H,
                 w.kda_q_is_gguf, w.kda_q_gguf_type);
            proj(normed_hidden_[r], w.kda_k_proj, x_k, C, H,
                 w.kda_k_is_gguf, w.kda_k_gguf_type);
            proj(normed_hidden_[r], w.kda_v_proj, x_v, C, H,
                 w.kda_v_is_gguf, w.kda_v_gguf_type);
            // beta (raw pre-sigmoid write strength) [B, Hk].
            proj(normed_hidden_[r], w.kda_b_proj, kda_beta_[r], Hk, H,
                 w.kda_b_is_gguf, w.kda_b_gguf_type);
            // Low-rank decay gate: raw_g = f_b(f_a(x)) [B, C].
            proj(normed_hidden_[r], w.kda_f_a_proj, kda_lowrank_[r], 128, H,
                 w.kda_f_a_is_gguf, w.kda_f_a_gguf_type);
            proj(kda_lowrank_[r], w.kda_f_b_proj, kda_rawg_[r], C, 128,
                 w.kda_f_b_is_gguf, w.kda_f_b_gguf_type);
            // Low-rank output gate: g2 = g_b(g_a(x)) [B, C]. kda_lowrank_ is
            // reused — stream-ordered, the f_b GEMM has consumed it.
            proj(normed_hidden_[r], w.kda_g_a_proj, kda_lowrank_[r], 128, H,
                 w.kda_g_a_is_gguf, w.kda_g_a_gguf_type);
            proj(kda_lowrank_[r], w.kda_g_b_proj, kda_g2_[r], C, 128,
                 w.kda_g_b_is_gguf, w.kda_g_b_gguf_type);
        };

        const size_t f_sec = static_cast<size_t>(kda_rows_max_) * C * 4;
        auto* cf_base = static_cast<char*>(kda_conv_f32_[r]);
        void* cq = cf_base;
        void* ck = cf_base + f_sec;
        void* cv = cf_base + 2 * f_sec;

        // o_proj emitter (shared tail; the NVFP4 legality check is hoisted
        // to each arm so the span lambda below stays throw-free once
        // capture begins).
        auto emit_oproj = [&] {
            // o_proj: [B, H] = onorm_out [B, C] @ o_proj[H, C_local]^T →
            // hidden_out_. GF3.10: o_proj is ROW-parallel (its K axis IS the
            // head axis, tp_weight_sharder shared arm), so with C = H/tp *
            // 128 rank-local channels this GEMM produces rank r's PARTIAL
            // hidden sum. The cross-rank combine happens ONCE below, after
            // the rank loop.
            // TD-GLM5-TP-COMBINE-PRECISION: under the fp32 TP combine the
            // partial goes to the FP32 staging buffer un-rounded; the single
            // bf16 rounding happens after the cross-rank sum below.
            if (tp_combine_fp32_active_)
                proj(kda_onorm_bf16_[r], w.o_proj, hidden_f32_[r], H, C,
                     w.o_proj_is_gguf, w.o_proj_gguf_type, /*c_fp32=*/true);
            else
                proj(kda_onorm_bf16_[r], w.o_proj, hidden_out_[r], H, C,
                     w.o_proj_is_gguf, w.o_proj_gguf_type);
        };

        if (!kda->decode) {
            emit_qkv();
            // ── Prefill: conv (rings live IN the slot) + chunked scan ──
            if (!kr.slot_base)
                throw std::runtime_error(
                    "execute_attention_kda: prefill without a slot base "
                    "(rank " + std::to_string(r) + ")");
            auto* slot = static_cast<char*>(kr.slot_base);
            compute::KdaConvPrefillArgs ca{};
            ca.x_q = x_q; ca.x_k = x_k; ca.x_v = x_v;
            ca.w_q = w.kda_q_conv1d; ca.w_k = w.kda_k_conv1d;
            ca.w_v = w.kda_v_conv1d;
            ca.ring_q = slot + ring0_off;
            ca.ring_k = slot + ring0_off + opts_.kda_ring_bytes_per_layer;
            ca.ring_v = slot + ring0_off + 2 * opts_.kda_ring_bytes_per_layer;
            ca.out_q = cq; ca.out_k = ck; ca.out_v = cv;
            ca.t_len = B; ca.channels = C;
            attn->kda_conv_prefill(ca, stream);

            compute::KdaChunkedScanArgs sa{};
            sa.q = cq; sa.k = ck; sa.v = cv;
            sa.raw_g = kda_rawg_[r];
            sa.beta = kda_beta_[r];
            sa.a_log = w.kda_a_log;
            sa.dt_bias = w.kda_dt_bias;
            sa.state = slot + rec_off;
            sa.core_out = kda_core_f32_[r];
            sa.workspace = kda_workspace_[r];
            sa.workspace_bytes = kda_workspace_bytes_;
            sa.t_len = B;
            sa.num_heads = Hk;
            sa.lower_bound = opts_.kda_gate_lower_bound;
            sa.l2_eps = opts_.kda_l2_eps;
            sa.scale = scale;
            attn->kda_chunked_scan(sa, stream);

            compute::KdaGatedRmsNormArgs na{};
            na.core = kda_core_f32_[r];
            na.g2 = kda_g2_[r];
            na.w = w.kda_o_norm;
            na.out_bf16 = kda_onorm_bf16_[r];
            na.out_f32 = nullptr;
            na.t_len = B;
            na.num_heads = Hk;
            na.eps = opts_.rms_norm_eps;
            attn->kda_gated_rmsnorm(na, stream);
        } else {
            // ── Decode: slot-indirect conv + fused O(1) step ──
            if (!kr.slots_host)
                throw std::runtime_error(
                    "execute_attention_kda: decode without a slot table "
                    "(rank " + std::to_string(r) + ")");
            void* region = r < static_cast<int>(opts_.kda_state_bases.size())
                ? opts_.kda_state_bases[r] : nullptr;
            if (!region)
                throw std::runtime_error(
                    "execute_attention_kda: no KDA state region base "
                    "(rank " + std::to_string(r) + ")");
            // Stage the per-row slot ids: pinned host → device, stream-
            // ordered on kAttention ahead of the conv/step launches.
            std::memcpy(kda_slots_host_[r], kr.slots_host,
                        static_cast<size_t>(B) * sizeof(int));
            attn->memcpy_h2d_async(kda_slots_dev_[r], kda_slots_host_[r],
                                   static_cast<size_t>(B) * sizeof(int),
                                   stream);
            const int64_t slot_stride_f =
                (opts_.kda_state_stride_bytes > 0
                     ? opts_.kda_state_stride_bytes
                     : opts_.kda_slot_bytes) / 4;
            auto* region_c = static_cast<char*>(region);
            // Hoisted o_proj legality check (emit_oproj runs inside the
            // captured span below — it must be throw-free once capture
            // begins).
            if (w.o_proj_is_nvfp4)
                throw std::runtime_error(
                    "execute_attention_kda: NVFP4 o_proj unsupported on KDA "
                    "layers");

            // P-29 step 7 (INV-0.6(b) span graph): the whole per-rank decode
            // chain — norm, 8 projections, slot-indirect conv, fused O(1)
            // step, o_proj — is a fixed-shape single-stream span: every grid
            // is a function of (B, C, H) only, and the only per-step
            // variation is the CONTENT of kda_slots_dev_ (restaged by the
            // eager H2D above, which stays OUTSIDE the span). Capture once
            // per (layer, rank), replay every token: ~20 host launches → 1.
            auto emit_span = [&] {
                emit_qkv();

                compute::KdaConvDecodeArgs ca{};
                ca.x_q = x_q; ca.x_k = x_k; ca.x_v = x_v;
                ca.w_q = w.kda_q_conv1d; ca.w_k = w.kda_k_conv1d;
                ca.w_v = w.kda_v_conv1d;
                ca.ring_q = region_c + ring0_off;
                ca.ring_k = region_c + ring0_off
                          + opts_.kda_ring_bytes_per_layer;
                ca.ring_v = region_c + ring0_off
                          + 2 * opts_.kda_ring_bytes_per_layer;
                ca.slots = kda_slots_dev_[r];
                ca.ring_slot_stride = slot_stride_f;
                ca.out_q = cq; ca.out_k = ck; ca.out_v = cv;
                ca.batch = B; ca.channels = C;
                attn->kda_conv_decode(ca, stream);

                compute::KdaDecodeStepArgs da{};
                da.q = cq; da.k = ck; da.v = cv;
                da.raw_g = kda_rawg_[r];
                da.beta = kda_beta_[r];
                da.g2 = kda_g2_[r];
                da.a_log = w.kda_a_log;
                da.dt_bias = w.kda_dt_bias;
                da.onorm_w = w.kda_o_norm;
                da.state_base = region_c + rec_off;
                da.state_slot_stride = slot_stride_f;
                da.slots = kda_slots_dev_[r];
                da.core_out = nullptr;
                da.out_bf16 = kda_onorm_bf16_[r];
                da.out_f32 = nullptr;
                da.batch = B;
                da.num_heads = Hk;
                da.lower_bound = opts_.kda_gate_lower_bound;
                da.l2_eps = opts_.kda_l2_eps;
                da.onorm_eps = opts_.rms_norm_eps;
                da.scale = scale;
                attn->kda_decode_step(da, stream);

                emit_oproj();
            };

            // GGUF-dequant is NOT graph-eligible (INV-0.6a: host-loop syncs
            // mid-sequence) — spans stay eager under that strategy.
            const bool span_ok = span_graphs_.enabled()
                && opts_.gguf_strategy != config::GgufStrategy::dequant;
            if (span_ok) {
                compute::DecodeSpanGraphs::Fp fp;
                // Buffers (per-rank persistent; content varies, addresses
                // fingerprinted).
                fp.add(params.hidden_states[r]);
                fp.add(normed_hidden_[r]);
                fp.add(kda_x_bf16_[r]);
                fp.add(kda_beta_[r]);
                fp.add(kda_lowrank_[r]);
                fp.add(kda_lowrank2_[r]);
                fp.add(kda_rawg_[r]);
                fp.add(kda_g2_[r]);
                fp.add(kda_conv_f32_[r]);
                fp.add(kda_onorm_bf16_[r]);
                fp.add(kda_slots_dev_[r]);
                fp.add(region);
                fp.add(tp_combine_fp32_active_ ? hidden_f32_[r]
                                               : hidden_out_[r]);
                // Weights (per-layer constants — drift means re-upload).
                fp.add(w.input_layernorm);
                fp.add(w.kda_q_proj);  fp.add(w.kda_k_proj);
                fp.add(w.kda_v_proj);  fp.add(w.kda_b_proj);
                fp.add(w.kda_f_a_proj); fp.add(w.kda_f_b_proj);
                fp.add(w.kda_g_a_proj); fp.add(w.kda_g_b_proj);
                fp.add(w.kda_q_conv1d); fp.add(w.kda_k_conv1d);
                fp.add(w.kda_v_conv1d);
                fp.add(w.kda_a_log);   fp.add(w.kda_dt_bias);
                fp.add(w.kda_o_norm);  fp.add(w.o_proj);
                fp.add(r < static_cast<int>(gguf_q8_1_ws_.size())
                           ? gguf_q8_1_ws_[r] : nullptr);
                // Scalars baked into launch params.
                fp.add_s(B);
                fp.add_s(rec_off);
                fp.add_s(ring0_off);
                fp.add_s(slot_stride_f);
                fp.add_s(tp_combine_fp32_active_ ? 1 : 0);
                // P-29 step 15: fused-projection route is part of the
                // captured launch structure.
                fp.add_s(proj_fused ? 1 : 0);
                span_graphs_.run(
                    stream,
                    compute::DecodeSpanGraphs::make_key(
                        compute::DecodeSpanGraphs::kKdaLayer,
                        params.layer_idx,
                        // P-29 step 13: row axis (spec-verify variants).
                        r + 16 * (params.batch_row_offset & 7)),
                    fp, emit_span);
            } else {
                emit_span();
            }
        }

        // LS_KDA_XRAY: env-gated stage magnitudes (debug-only D2H+sync;
        // never on unless explicitly requested — first-boot triage tool).
        static const bool kda_xray = [] {
            const char* e = std::getenv("LS_KDA_XRAY");
            return e && *e && *e != '0';
        }();
        if (kda_xray && r < static_cast<int>(opts_.device_backends.size())
            && opts_.device_backends[r]) {
            auto* be = opts_.device_backends[r];
            auto rms_bf16 = [&](const void* p_, size_t n) {
                std::vector<uint16_t> h(n);
                be->memcpy_d2h_async(h.data(), p_, n * 2, stream);
                attn->device_sync();
                double acc = 0;
                for (auto v : h) {
                    uint32_t u = static_cast<uint32_t>(v) << 16;
                    float f;
                    std::memcpy(&f, &u, 4);
                    acc += static_cast<double>(f) * f;
                }
                return std::sqrt(acc / std::max<size_t>(n, 1));
            };
            auto rms_f32 = [&](const void* p_, size_t n) {
                std::vector<float> h(n);
                be->memcpy_d2h_async(h.data(), p_, n * 4, stream);
                attn->device_sync();
                double acc = 0;
                for (auto f : h) acc += static_cast<double>(f) * f;
                return std::sqrt(acc / std::max<size_t>(n, 1));
            };
            const size_t nC = static_cast<size_t>(B) * C;
            spdlog::info(
                "[kda-xray] L{} ord{} {} B{} | normed {:.4f} q {:.4f} k "
                "{:.4f} v {:.4f} beta {:.4f} rawg {:.4f} g2 {:.4f} | convq "
                "{:.4f} convk {:.4f} convv {:.4f} | core {:.4f} onorm "
                "{:.4f}",
                params.layer_idx, ord, kda->decode ? "dec" : "pre", B,
                rms_bf16(normed_hidden_[r], static_cast<size_t>(B) * H),
                rms_bf16(x_q, nC), rms_bf16(x_k, nC), rms_bf16(x_v, nC),
                rms_bf16(kda_beta_[r], static_cast<size_t>(B) * Hk),
                rms_bf16(kda_rawg_[r], nC), rms_bf16(kda_g2_[r], nC),
                rms_f32(cq, nC), rms_f32(ck, nC), rms_f32(cv, nC),
                rms_f32(kda_core_f32_[r], nC),
                rms_bf16(kda_onorm_bf16_[r], nC));
        }

        // Prefill arm o_proj (the decode arm already emitted it inside the
        // span above). The NVFP4 legality check runs here for prefill; the
        // decode arm hoisted its own copy ahead of the span.
        if (!kda->decode) {
            if (w.o_proj_is_nvfp4)
                throw std::runtime_error(
                    "execute_attention_kda: NVFP4 o_proj unsupported on KDA "
                    "layers");
            emit_oproj();
        }
    }

    // GF3.10 — the KDA TP allreduce (the mirror of MLA step 14,
    // arch_mla.cpp execute_oproj_and_reduce): sum the per-rank partial
    // hidden_out_ rows [0, B) across TP ranks, in place, on the per-rank
    // kAttention streams. INV-KDA-TP discipline: this is the ONLY
    // cross-rank combine a KDA layer performs — projections, conv, scan
    // and o_norm are all rank-local over the head slice (state pool,
    // rings, A_log/dt_bias and beta are sharded by the SAME head split,
    // GF3.3/GF3.8), so one deterministic fixed-topology NCCL sum
    // reconstructs the full o_proj output on every rank. dcp_size == 1:
    // reduce_hidden is a structural no-op (is_active() false).
    // TD-GLM5-TP-COMBINE-PRECISION: fp32 partials -> fp32 allreduce -> ONE
    // bf16 round into hidden_out_ (per rank, same kAttention stream). The
    // bf16 arm is byte-identical to the pre-fix behavior.
    if (tp_combine_fp32_active_) {
        if (opts_.dcp_wrapper)
            opts_.dcp_wrapper->reduce_hidden(hidden_f32_.data(), B,
                                             attn_streams_.data(),
                                             /*fp32=*/true);
        for (int r = 0; r < dcp_size_; ++r) {
            auto* attn = opts_.attention_devices[r];
            attn->set_device();
            attn->cast_f32_to_bf16(hidden_out_[r], hidden_f32_[r],
                                   static_cast<int64_t>(B) * H,
                                   attn_streams_[r]);
        }
    } else if (opts_.dcp_wrapper) {
        opts_.dcp_wrapper->reduce_hidden(hidden_out_.data(), B,
                                         attn_streams_.data());
    }
    tp_hidden_probe("kda-out", params.layer_idx, hidden_out_.data(), B);
}

}  // namespace layerstorm::parallelism

// ── The dispatcher-side hooks ───────────────────────────────────────────────

namespace layerstorm::daemon {

bool ArchGlm5Next::refuse(const char* msg) {
    d_.last_internal_error_cat_ = ipc::CmpErrorCategory::kComputeValidation;
    d_.last_internal_error_msg_ = msg;
    spdlog::error("{}", msg);
    return false;
}

bool ArchGlm5Next::ensure_layer_axis() {
    if (axis_ready_) return true;
    if (!d_.deps_.live_config) return false;
    const auto& m = d_.deps_.live_config->model;
    model::ModelConfig mc(m);
    const int n = m.num_hidden_layers;
    is_linear_.assign(static_cast<size_t>(std::max(n, 0)), 0);
    linear_ordinal_.assign(static_cast<size_t>(std::max(n, 0)), -1);
    num_linear_ = 0;
    for (int l = 0; l < n; ++l) {
        if (mc.is_linear_attention_layer(l)) {
            is_linear_[static_cast<size_t>(l)] = 1;
            linear_ordinal_[static_cast<size_t>(l)] = num_linear_++;
        }
    }
    axis_ready_ = true;
    return true;
}

bool ArchGlm5Next::validate_shape(
    const CommandDispatcher::InternalAttentionParams& p, int& batch_cap) {
    batch_cap = 0;
    if (!ensure_layer_axis())
        return refuse("glm5_next: live config unavailable at dispatch");
    // GF3.10 — TP >= 2 with REPLICATED KV is executable: the KDA tensors
    // are head-sharded (GF3.3), the state pool is per-rank by the same
    // head split (GF3.8), and execute_attention_kda now performs the
    // row-parallel o_proj partial + ONE cross-rank combine per KDA layer
    // (INV-KDA-TP). Sequence-SHARDED KV stays fail-closed: a
    // whole-sequence recurrent state has no token shard to own, so the
    // QAG structure the sharded mode drags the sparse layers into has no
    // meaning for 34 of the 45 layers — the engine also refuses this at
    // boot (engine.cpp, mirror of TD-V4-DCP-KV), so this arm is
    // belt-and-braces for a driver wired past that throw.
    const int dcp_size =
        d_.deps_.dcp_executor ? d_.deps_.dcp_executor->dcp_size() : 1;
    if (dcp_size > 1 && d_.kv_sharded_)
        return refuse(
            "glm5_next: sequence-sharded KV is not supported (the KDA "
            "recurrent state is whole-sequence; replicated KV is the "
            "glm5_next TP mode) — set hardware.dcp_kv_mode = replicated");
    // Decode graphs have no KDA arms (and are engine-dormant,
    // TD-DECODE-GRAPH) — refuse rather than replay an MLA-only capture.
    if (p.use_graph != 0 && p.is_prefill == 0)
        return refuse(
            "glm5_next: decode-graph steps are unsupported (no KDA graph "
            "arms; TD-DECODE-GRAPH)");
    batch_cap = d_.deps_.max_batch_size;
    return true;
}

bool ArchGlm5Next::stage_step(
    const CommandDispatcher::InternalAttentionParams& p,
    parallelism::AttentionExecParams& params,
    int batch_size, int layer, int dcp_size, bool kv_meta_ok) {
    if (!ensure_layer_axis())
        return refuse("glm5_next: live config unavailable at dispatch");

    const bool linear =
        layer >= 0 && layer < static_cast<int>(is_linear_.size())
        && is_linear_[static_cast<size_t>(layer)] != 0;
    if (!linear) {
        // Sparse NoPE-MLA layer (incl. the MTP layer >= num_hidden_layers):
        // the full ArchMla staging — DSA coverage machine, IndexPool
        // provisioning, chunk synthesis. Tiering gates inside are inert
        // while kv_tiering_ is unconstructed (GF3.9 default-OFF decision).
        return mla_.stage_step(p, params, batch_size, layer, dcp_size,
                               kv_meta_ok);
    }

    // ── KDA layer ──
    // No KV metadata consumption, no coverage machine, no sparse/dense
    // selection, no tiering, no indexer witness (step_indexer_dense_
    // untouched — the TD-INDEXER-NO-DENSE-FALLBACK counter must not be
    // polluted by KDA dispatches).
    if (p.is_draft != 0)
        return refuse(
            "glm5_next: draft step on a KDA layer — the MTP draft is "
            "sparse MLA and drafts claim no recurrent state (INV-KDA-STATE "
            "(e)); this is a wiring error");
    if (!d_.deps_.sideband_base)
        return refuse("glm5_next: KDA step without sideband descriptors");
    const auto* be = reinterpret_cast<const ipc::BatchDescriptorEntry*>(
        d_.deps_.sideband_base + ipc::IpcLayout::kBatchDescriptorOff);
    const int ord = linear_ordinal_[static_cast<size_t>(layer)];
    const bool prefill_shape = p.is_prefill != 0 || p.chunk_len > 0;

    kda_ranks_.assign(static_cast<size_t>(dcp_size), {});
    kda_slots_.assign(static_cast<size_t>(dcp_size), {});

    if (prefill_shape) {
        // ONE sequence at consecutive ascending positions (the same shape
        // law the MLA prefill arm enforces).
        const uint64_t sid = be[0].seq_id;
        const uint32_t start = be[0].token_pos;
        for (int b = 1; b < batch_size; ++b) {
            if (be[b].seq_id != sid
                || be[b].token_pos != start + static_cast<uint32_t>(b))
                return refuse(
                    "glm5_next: malformed KDA prefill chunk — descriptors "
                    "must be ONE sequence at consecutive positions");
        }
        auto* st = d_.find_seq(sid);
        if (!st)
            return refuse("glm5_next: KDA step on unknown seq_id");
        // TD-KDA-STATE-MAPPED-SLABS: kda_state holds [rank][unit]
        // handles — 1 unit/rank on the carve (the whole slot; the
        // executor adds the per-layer offset), num_layers units/rank
        // mapped (the handle IS the layer unit; offset 0). Same shape
        // check, unit-aware.
        const size_t kda_units = static_cast<size_t>(std::max(
            1, d_.deps_.page_allocator
                   ? d_.deps_.page_allocator->kda_units_per_rank() : 1));
        const bool kda_mapped = d_.deps_.page_allocator
            && d_.deps_.page_allocator->kda_state_mapped();
        if (st->kda_state.size()
            != static_cast<size_t>(dcp_size) * kda_units)
            return refuse(
                "glm5_next: sequence carries no KDA state slot (draft or "
                "pre-GF3.8 create?)");
        if (st->kda_next_pos.empty())
            st->kda_next_pos.assign(static_cast<size_t>(num_linear_), 0);
        // INV-KDA-REWIND frontier guard, per LAYER: the launch must start
        // exactly where this layer's recurrent state stands. A retry-seam
        // re-feed, an unanchored rewind, or a mis-chunked prefill lands
        // here — refused, never re-applied (state mutation is not
        // idempotent). GF3.11 owns anchor-and-replay.
        if (st->kda_next_pos[static_cast<size_t>(ord)] != start)
            return refuse(
                "glm5_next: KDA prefill start is not this layer's state "
                "frontier — re-feeding a recurrent state is refused "
                "(INV-KDA-REWIND; anchor-and-replay lands in GF3.11)");
        st->kda_next_pos[static_cast<size_t>(ord)] =
            start + static_cast<uint32_t>(batch_size);
        for (int r = 0; r < dcp_size; ++r)
            kda_ranks_[static_cast<size_t>(r)].slot_base =
                st->kda_state[static_cast<size_t>(r) * kda_units
                              + (kda_mapped ? static_cast<size_t>(ord)
                                            : 0)].gpu_ptr;
        kda_step_.decode = false;
    } else {
        // Decode: one token per row; every row its own sequence (two rows
        // of one sequence in a single step would race the slot).
        for (int b = 0; b < batch_size; ++b) {
            for (int b2 = 0; b2 < b; ++b2)
                if (be[b2].seq_id == be[b].seq_id)
                    return refuse(
                        "glm5_next: duplicate sequence in a KDA decode "
                        "cohort — the state slot admits one step per "
                        "dispatch");
            auto* st = d_.find_seq(be[b].seq_id);
            if (!st)
                return refuse("glm5_next: KDA step on unknown seq_id");
            const size_t kda_units = static_cast<size_t>(std::max(
                1, d_.deps_.page_allocator
                       ? d_.deps_.page_allocator->kda_units_per_rank()
                       : 1));
            const bool kda_mapped = d_.deps_.page_allocator
                && d_.deps_.page_allocator->kda_state_mapped();
            if (st->kda_state.size()
                != static_cast<size_t>(dcp_size) * kda_units)
                return refuse(
                    "glm5_next: sequence carries no KDA state slot (draft "
                    "or pre-GF3.8 create?)");
            if (st->kda_next_pos.empty())
                st->kda_next_pos.assign(static_cast<size_t>(num_linear_), 0);
            if (st->kda_next_pos[static_cast<size_t>(ord)]
                != be[b].token_pos)
                return refuse(
                    "glm5_next: KDA decode position is not this layer's "
                    "state frontier (INV-KDA-REWIND; anchor-and-replay "
                    "lands in GF3.11)");
            st->kda_next_pos[static_cast<size_t>(ord)] =
                be[b].token_pos + 1;
            // Mapped mode: the slot id is THIS LAYER's run-start slab
            // id (per-layer slot tables — the per-launch indirection the
            // ticket blessed); carve mode: the whole-slot index, as ever.
            for (int r = 0; r < dcp_size; ++r)
                kda_slots_[static_cast<size_t>(r)].push_back(
                    st->kda_state[static_cast<size_t>(r) * kda_units
                                  + (kda_mapped ? static_cast<size_t>(ord)
                                                : 0)].page_idx);
        }
        for (int r = 0; r < dcp_size; ++r)
            kda_ranks_[static_cast<size_t>(r)].slots_host =
                kda_slots_[static_cast<size_t>(r)].data();
        kda_step_.decode = true;
    }

    kda_step_.linear_ordinal = ord;
    kda_step_.ranks = kda_ranks_.data();
    kda_step_.num_ranks = dcp_size;
    params.kda = &kda_step_;
    return true;
}

bool ArchGlm5Next::execute(
    const CommandDispatcher::InternalAttentionParams& p,
    parallelism::AttentionExecParams& params,
    int batch_size, int layer, int dcp_size, bool kv_meta_ok) {
    const bool linear =
        layer >= 0 && layer < static_cast<int>(is_linear_.size())
        && is_linear_[static_cast<size_t>(layer)] != 0;
    if (!linear)
        return mla_.execute(p, params, batch_size, layer, dcp_size,
                            kv_meta_ok);
    if (!params.kda)
        return refuse(
            "glm5_next: execute on a KDA layer without a staged KdaStep "
            "(stage_step contract)");
    d_.deps_.dcp_executor->execute_attention_kda(params);
    return true;
}

}  // namespace layerstorm::daemon
