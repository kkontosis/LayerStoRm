// DeepSeek-V4 CSA/HCA/SWA hybrid AttentionDevice (V4-5a/5d/5e) — SM120.
//
// Composes CudaSm120DeviceBackend for hardware ops and drives the deps
// V4K kernels through their C++ entry points:
//   run_csa_fp8_decode_kernel<64|128>   (sm120/decode/csa_fp8/params.h)
//   run_get_mla_metadata_kernel / run_mla_combine_kernel (smxx)
//   launch_lightning_score_mqa_batched / launch_lightning_topk_batched
//   CsaFp8DecodeGraphRunner             (sm120/graph/csa_fp8_decode_graph.h)
// plus the engine-side V4 prep epilogues (v4_attn_sinks, v4_out_inverse_rope,
// v4_entry_gather). See csa_hca_sm120_attention_device.h for contracts.
//
// MLA-only AttentionDevice virtuals (absorb_q, k_append, prefill_attention,
// decode_graph_*) throw: the V4 pipeline runs through the bridge API — the
// MLA seams are dimensionally meaningless for V4 (single 512-latent MQA,
// duplicated-rope entries) and silently mis-executing them is worse than
// failing loud (TD-V4-ATTN-ROUTING owns the executor-side wiring, V4-7b).

#include "compute/csa_hca_sm120_attention_device.h"

#include "compute/cuda_sm120_device_backend.h"
#include "compute/kernels/attention/v4_prep.h"
#include "compute/kernels/sm120/indexer/lightning_indexer.h"
#include "config/config_parser.h"

#include "compute/kernels/attention/tq_mla_attention.h"  // V4-5T q/v rotates
#include "compute/tq_init.h"                             // V4-5T TqResources

#include <sm120/decode/csa_fp8/params.h>
#include <sm120/decode/csa_tq/params.h>   // V4-5T run_csa_tq_decode
#include <sm120/prep/tq_q_rotate.h>       // V4-5T Π-rotate q
#include <sm120/prep/tq_v_rotate_back.h>  // V4-5T Π^T rotate-back
#include <sm120/graph/csa_fp8_decode_graph.h>
#include <smxx/get_mla_metadata.h>
#include <smxx/mla_combine.h>
#include <smxx/params.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace layerstorm::compute {

namespace {
constexpr int kEntryBytes = 1160;    // deps V4CacheLayout::ENTRY_BYTES (FP8)
constexpr int kTqEntryBytes = 644;   // deps V4 TQ entry (V4-5T)
constexpr int kMaxCombineSplits = 192;  // deps mla_combine MAX_SPLITS
}  // namespace

class CsaHcaSm120AttentionDevice final : public AttentionDevice {
public:
    explicit CsaHcaSm120AttentionDevice(config::GpuRef gpu) : device_(gpu) {
        int sm = 0;
        const cudaError_t err = cudaDeviceGetAttribute(
            &sm, cudaDevAttrMultiProcessorCount, gpu.id);
        if (err != cudaSuccess || sm <= 0) {
            throw std::runtime_error(
                "CsaHcaSm120AttentionDevice: cudaDeviceGetAttribute failed "
                "for CUDA device " + std::to_string(gpu.id) + ": " +
                cudaGetErrorString(err));
        }
        num_sm_ = sm;
    }

    ~CsaHcaSm120AttentionDevice() override {
        for (auto& r : graph_runners_) {
            if (r) r->destroy();
        }
        free_scratch();
    }

    // ── Hardware ops (delegated to CudaSm120DeviceBackend) ─────────────────

    void set_device() override { device_.set_device(); }
    const config::GpuRef& gpu() const override { return device_.gpu(); }

    void gemm(const Fp8GemmParams& p, void* ws, void* s) override {
        device_.gemm(p, ws, s);
    }
    void gguf_mmvq(const GgufGemmParams& p, void* ws, void* s) override {
        device_.gguf_mmvq(p, ws, s);
    }
    void gguf_mmq(const GgufGemmParams& p, void* ws, void* s) override {
        device_.gguf_mmq(p, ws, s);
    }
    void gguf_dequant_gemm(const GgufGemmParams& p, void* s) override {
        device_.gguf_dequant_gemm(p, s);
    }
    void rmsnorm(void* out, const void* in, const void* w, float eps, int nt,
                 int hs, int rs, void* s) override {
        device_.rmsnorm(out, in, w, eps, nt, hs, rs, s);
    }
    void quantize_fp8(const DynamicFp8QuantParams& p, void* s) override {
        device_.quantize_fp8(p, s);
    }
    void weight_quantize_fp8(const WeightFp8QuantParams& p, void* s) override {
        device_.weight_quantize_fp8(p, s);
    }
    void nvfp4_dequant_bf16(const Nvfp4DequantBf16Params& p, void* s) override {
        device_.nvfp4_dequant_bf16(p, s);
    }
    void nvfp4_grouped_gemm(const Nvfp4GroupedGemmParams& p, void* ws,
                            size_t wb, void* s) override {
        device_.nvfp4_grouped_gemm(p, ws, wb, s);
    }
    void bf16_to_nvfp4_grouped(const Bf16ToNvfp4GroupedParams& p,
                               void* s) override {
        device_.bf16_to_nvfp4_grouped(p, s);
    }
    void batched_gemm_bf16(const StridedBatchedGemmBf16Params& p,
                           void* s) override {
        device_.batched_gemm_bf16(p, s);
    }
    void rope_rotate(const RopeRotateParams& p, void* s) override {
        device_.rope_rotate(p, s);
    }
    // DSA indexer hardware ops: harmless delegation (V4 does not call them;
    // the lightning path goes through csa_hca_device_lightning_select).
    void indexer_layernorm_bias(void* x, const void* w, const void* b, int r,
                                int d, float e, void* s) override {
        device_.indexer_layernorm_bias(x, w, b, r, d, e, s);
    }
    void indexer_hadamard(void* x, int r, int d, void* s) override {
        device_.indexer_hadamard(x, r, d, s);
    }
    void indexer_k_quant_append(const void* k, const void* sl, void* c,
                                void* sc, int t, int d, int sb,
                                void* s) override {
        device_.indexer_k_quant_append(k, sl, c, sc, t, d, sb, s);
    }
    void indexer_scale_weights(const void* in, void* out, int r, int n,
                               float sc, void* s) override {
        device_.indexer_scale_weights(in, out, r, n, sc, s);
    }
    void indexer_score_topk(const IndexerScoreTopkArgs& a, void* s) override {
        device_.indexer_score_topk(a, s);
    }
    void indexer_score_topk_batched(const IndexerScoreTopkBatchedArgs& a,
                                    void* s) override {
        device_.indexer_score_topk_batched(a, s);
    }
    void indexer_shard_translate(const void* gi, const void* gl, void* li,
                                 void* ll, int nt, int tk, int ch, int dc,
                                 int rk, void* s) override {
        device_.indexer_shard_translate(gi, gl, li, ll, nt, tk, ch, dc, rk, s);
    }
    void indexer_topk_merge(const IndexerTopkMergeArgs& a, void* s) override {
        device_.indexer_topk_merge(a, s);
    }

    void* device_alloc(size_t bytes) override {
        return device_.device_alloc(bytes);
    }
    void device_free(void* p) override { device_.device_free(p); }
    void device_sync() override { device_.device_sync(); }
    void memcpy_h2d(void* d, const void* s, size_t b) override {
        device_.memcpy_h2d(d, s, b);
    }
    void memcpy_2d_d2d_async(void* d, size_t dp, const void* s, size_t sp,
                             size_t w, size_t h, void* st) override {
        device_.memcpy_2d_async(d, dp, s, sp, w, h, st);
    }
    void memcpy_h2d_async(void* d, const void* s, size_t b, void* st) override {
        device_.memcpy_h2d_async(d, s, b, st);
    }

    // ── MLA-only virtuals: fail loud (see header note) ──────────────────────

    void absorb_q(const QAbsorbParams&, void*) override {
        throw std::logic_error("CsaHcaSm120AttentionDevice: absorb_q is "
                               "MLA-only — V4 uses v4_q_prep");
    }
    void kv_bv_extract_dequant(const KvBvExtractDequantParams&,
                               void*) override {
        throw std::logic_error("CsaHcaSm120AttentionDevice: kv_bv_extract_"
                               "dequant is MLA-only (no kv_b in V4)");
    }
    void k_append(const void*, const void*, void*, int64_t, int, const int*,
                  int, int, int, int, int, int, int, void*) override {
        throw std::logic_error("CsaHcaSm120AttentionDevice: MLA k_append "
                               "unsupported — V4 uses launch_v4_raw_kv_append"
                               " + launch_v4_compress_insert");
    }
    void prefill_attention(const void*, int, int, const int*, const int*, int,
                           void*, int64_t, int, int, bool, bool, const int*,
                           const int*, int, void*, float*, int,
                           void*) override {
        throw std::logic_error("CsaHcaSm120AttentionDevice: MLA prefill_"
                               "attention unsupported — V4 prefill runs "
                               "csa_hca_device_attention per-row "
                               "(TD-V4-ATTN-ROUTING)");
    }
    void decode_graph_update(GraphEntry&, const void*, const int*, const int*,
                             const int*, int, void*) override {
        throw std::logic_error("CsaHcaSm120AttentionDevice: MLA decode graph "
                               "seam unsupported — use csa_hca_device_graph_*");
    }
    void decode_graph_replay(GraphEntry&, void*) override {
        throw std::logic_error("CsaHcaSm120AttentionDevice: MLA decode graph "
                               "seam unsupported — use csa_hca_device_graph_*");
    }
    void* decode_graph_out_ptr(GraphEntry&) override {
        throw std::logic_error("CsaHcaSm120AttentionDevice: MLA decode graph "
                               "seam unsupported — use csa_hca_device_graph_*");
    }
    float* decode_graph_lse_ptr(GraphEntry&) override {
        throw std::logic_error("CsaHcaSm120AttentionDevice: MLA decode graph "
                               "seam unsupported — use csa_hca_device_graph_*");
    }
    void dcp_graph_replay(GraphEntry&, void*) override {
        throw std::logic_error("CsaHcaSm120AttentionDevice: DCP is fail-closed"
                               " for V4 (TD-V4-DCP-KV)");
    }

    // ── V4 configuration ────────────────────────────────────────────────────

    void configure(const V4DeviceOptions& o) {
        if (o.h_q != 64 && o.h_q != 128) {
            // The deps decode kernel loads full BLOCK_SIZE_M(=64)-head Q
            // tiles (only the scale/output paths are masked) — h_q must
            // match an instantiation exactly.
            throw std::runtime_error(
                "csa_hca configure: h_q must be 64 or 128 (deps csa_fp8 "
                "decode tile contract), got " + std::to_string(o.h_q));
        }
        if (o.topk % 64 != 0) {
            throw std::runtime_error(
                "csa_hca configure: topk must be a multiple of 64 (deps "
                "TOPK_BLOCK_SIZE)");
        }
        opts_ = o;
        if (opts_.num_sm_parts < 1) opts_.num_sm_parts = 1;
        if (opts_.num_sm_parts > kMaxCombineSplits) {
            opts_.num_sm_parts = kMaxCombineSplits;
        }
    }

    void set_scratch(int max_staged_entries) {
        if (opts_.h_q <= 0) {
            throw std::runtime_error(
                "csa_hca set_scratch: configure() must run first");
        }
        if (max_staged_entries <= staged_capacity_ &&
            attn_rows_capacity_ > 0) {
            return;
        }
        device_.set_device();
        free_scratch();

        const int rows = opts_.max_attn_rows;
        const int h = opts_.h_q;
        const int dv = opts_.head_dim;
        const int accum_rows = rows + opts_.num_sm_parts + 1;

        staging_ = device_.device_alloc(
            static_cast<size_t>(max_staged_entries) * kEntryBytes);
        o_accum_ = static_cast<float*>(device_.device_alloc(
            static_cast<size_t>(accum_rows) * h * dv * sizeof(float)));
        lse_accum_ = static_cast<float*>(device_.device_alloc(
            static_cast<size_t>(accum_rows) * h * sizeof(float)));
        sched_meta_ = static_cast<int*>(device_.device_alloc(
            static_cast<size_t>(opts_.num_sm_parts) *
            TileSchedulerMetaDataSize * sizeof(int)));
        num_splits_ = static_cast<int*>(device_.device_alloc(
            static_cast<size_t>(rows + 1) * sizeof(int)));
        scores_scratch_ = static_cast<float*>(device_.device_alloc(
            static_cast<size_t>(rows) *
            std::max(opts_.max_index_blocks, 1) * sizeof(float)));
        topk_scores_scratch_ = static_cast<float*>(device_.device_alloc(
            static_cast<size_t>(rows) * std::max(opts_.topk, 1) *
            sizeof(float)));
        topk_effk_scratch_ = static_cast<int*>(device_.device_alloc(
            static_cast<size_t>(rows) * sizeof(int)));
        if (!staging_ || !o_accum_ || !lse_accum_ || !sched_meta_ ||
            !num_splits_ || !scores_scratch_ || !topk_scores_scratch_ ||
            !topk_effk_scratch_) {
            throw std::runtime_error(
                "CsaHcaSm120AttentionDevice: scratch allocation failed");
        }
        staged_capacity_ = max_staged_entries;
        attn_rows_capacity_ = rows;

        // V4-5T TQ scratch: rotated-q FP32, rotated FP32 partial + its
        // BF16 rotate-back + LSE, and the compress-insert BF16 staging
        // rows (bounded by max_attn_rows compress blocks per call).
        if (opts_.csa_codec == V4TierCodec::kTq4 ||
            opts_.hca_codec == V4TierCodec::kTq4) {
            const size_t rh = static_cast<size_t>(rows) * h;
            tq_q_rot_ = static_cast<float*>(
                device_.device_alloc(rh * dv * sizeof(float)));
            tq_out_f32_ = static_cast<float*>(
                device_.device_alloc(rh * dv * sizeof(float)));
            tq_out_bf16_ = device_.device_alloc(rh * dv * 2);
            tq_lse_ = static_cast<float*>(
                device_.device_alloc(rh * sizeof(float)));
            tq_stage_rows_ = device_.device_alloc(
                static_cast<size_t>(rows) * dv * 2);
            tq_stage_rope_ = device_.device_alloc(
                static_cast<size_t>(rows) * opts_.rope_dim * 2);
            if (!tq_q_rot_ || !tq_out_f32_ || !tq_out_bf16_ || !tq_lse_ ||
                !tq_stage_rows_ || !tq_stage_rope_) {
                throw std::runtime_error(
                    "CsaHcaSm120AttentionDevice: TQ scratch alloc failed");
            }
        }
    }

    void set_tq(const TqResources* res) { tq_res_ = res; }

    // V4-5T: TQ-tier compress-insert — compressor → BF16 staging rows →
    // deps v4_tq_k_append (normalize → Π-rotate → 4-bit pack, 644-B
    // entries at ca.slots of ca.kv_cache).
    void tq_compress_insert(const V4CompressArgs& ca_in, int layer_idx,
                            void* stream) {
        if (!tq_res_) {
            throw std::runtime_error(
                "csa_hca tq_compress_insert: TqResources not set "
                "(csa_hca_device_set_tq)");
        }
        if (!tq_stage_rows_ || ca_in.num_blocks > attn_rows_capacity_) {
            throw std::runtime_error(
                "csa_hca tq_compress_insert: staging missing/short (blocks "
                + std::to_string(ca_in.num_blocks) + ")");
        }
        V4CompressArgs ca = ca_in;
        ca.out_mode = V4CompressArgs::Out::kBf16Rows;
        ca.bf16_rows = tq_stage_rows_;
        ca.bf16_rope_rows = tq_stage_rope_;
        void* tier_base = ca.kv_cache;
        ca.kv_cache = nullptr;
        launch_v4_compress_insert(ca, stream);

        V4TqAppendArgs ta{};
        ta.rows = tq_stage_rows_;
        ta.rope_rows = tq_stage_rope_;
        ta.kv_cache = tier_base;
        ta.slots = ca_in.slots;
        ta.num_tokens = ca_in.num_blocks;
        ta.head_dim = ca_in.D;
        ta.rope_dim = ca_in.rope_dim;
        ta.Pi = tq_res_->device_Pi(layer_idx);
        ta.centroids = tq_res_->device_centroids();
        ta.boundaries = tq_res_->device_boundaries();
        ta.num_centroids = tq_res_->codebook.n_clusters;
        launch_v4_tq_entry_append(ta, stream);
    }

    // SWA-only decode arm: single-partition scheduler metadata (all rows of
    // ONE call in one partition, zero compressed blocks — end_req_idx MUST
    // equal the call's rows-1, not the capacity, or the kernel batch loop
    // reads swa_seqlens past the caller's arrays) + zero cumulative splits
    // (every row no-split; combine early-outs). Cached per batch size.
    std::pair<const int*, const int*> swa_only_meta(int rows) {
        auto it = swa_meta_cache_.find(rows);
        if (it != swa_meta_cache_.end()) {
            return {it->second.first, it->second.second};
        }
        int* meta_d = static_cast<int*>(device_.device_alloc(
            static_cast<size_t>(TileSchedulerMetaDataSize) * sizeof(int)));
        int* splits_d = static_cast<int*>(device_.device_alloc(
            static_cast<size_t>(rows + 1) * sizeof(int)));
        if (!meta_d || !splits_d) {
            throw std::runtime_error(
                "CsaHcaSm120AttentionDevice: SWA metadata alloc failed");
        }
        std::vector<int> meta(TileSchedulerMetaDataSize, 0);
        meta[0] = 0;         // begin_req_idx
        meta[1] = 0;         // begin_block_idx
        meta[2] = rows - 1;  // end_req_idx
        meta[3] = 0;         // end_block_idx
        meta[4] = 0;         // begin_n_split_idx
        device_.memcpy_h2d(meta_d, meta.data(), meta.size() * sizeof(int));
        std::vector<int> zeros(static_cast<size_t>(rows) + 1, 0);
        device_.memcpy_h2d(splits_d, zeros.data(), zeros.size() * sizeof(int));
        swa_meta_cache_[rows] = {meta_d, splits_d};
        return {meta_d, splits_d};
    }

    // ── Attention (decode + prefill-as-decode) ──────────────────────────────

    void attention(const V4AttentionArgs& a) {
        if (a.rows <= 0) return;
        if (a.rows > attn_rows_capacity_) {
            throw std::runtime_error(
                "csa_hca attention: rows " + std::to_string(a.rows) +
                " > capacity " + std::to_string(attn_rows_capacity_) +
                " (set max_attn_rows / sub-batch the call)");
        }
        if (a.topk % 64 != 0) {
            throw std::runtime_error(
                "csa_hca attention: topk must be a multiple of 64");
        }
        if (!a.out || !a.lse) {
            throw std::runtime_error("csa_hca attention: out/lse required");
        }
        void* stream = a.stream;

        const bool swa_only = (a.topk == 0);
        int nsp = a.num_sm_parts > 0 ? a.num_sm_parts : opts_.num_sm_parts;
        if (nsp > kMaxCombineSplits) nsp = kMaxCombineSplits;

        const int* sched = nullptr;
        const int* splits = nullptr;
        if (swa_only) {
            const auto meta = swa_only_meta(a.rows);
            sched = meta.first;
            splits = meta.second;
            nsp = 1;
        } else {
            GetMlaMetadataParams mp{};
            mp.seqlens_k_ptr = nullptr;  // unread when topk >= 0
            mp.tile_scheduler_metadata_ptr = sched_meta_;
            mp.num_splits_ptr = num_splits_;
            mp.batch_size = a.rows;
            mp.block_size_n = 64;
            mp.fixed_overhead_num_blocks = 1;
            mp.num_sm_parts = nsp;
            mp.topk = a.topk;
            run_get_mla_metadata_kernel(mp, static_cast<cudaStream_t>(stream));
            sched = sched_meta_;
            splits = num_splits_;
        }

        // V4-5T TQ codec: the compressed tier of this call is TQ-coded —
        // TWO passes + LSE merge (the deps csa_tq kernel carries no SWA
        // support by design; its header names this composition):
        //   (1) csa_tq over comp_cache: q_nope Π-rotated to FP32, sparse
        //       indices, out FP32 in ROTATED space + natural-log LSE;
        //       Π^T rotate-back to the normal V space (BF16);
        //   (2) the FP8 SWA-only arm (topk == 0) into a.out/a.lse;
        //   (3) natural-units LSE merge of (1) into (2);
        // then the SHARED sinks + inverse-rope epilogues on the merged
        // output — identical epilogue semantics to the FP8 arm.
        if (a.comp_tq && !swa_only) {
            if (!tq_res_ || !tq_q_rot_) {
                throw std::runtime_error(
                    "csa_hca attention: TQ tier without TQ resources/"
                    "scratch (csa_hca_device_set_tq / set_scratch)");
            }
            if (a.layer_idx < 0) {
                throw std::runtime_error(
                    "csa_hca attention: TQ tier requires layer_idx (Π)");
            }
            const int h = opts_.h_q;
            const int dv = opts_.head_dim;
            const int bh = a.rows * h;

            // (1a) Π-rotate q_nope → FP32 rotated query.
            sm120::prep::TqQRotateParams qr{};
            qr.q_nope = static_cast<const __nv_bfloat16*>(a.q_nope);
            qr.Pi = tq_res_->device_Pi(a.layer_idx);
            qr.q_rot = tq_q_rot_;
            qr.batch_heads = bh;
            qr.d_c = dv;
            qr.q_row_stride = 0;  // tight [rows, h, dv]
            launch_tq_q_rotate(qr, static_cast<cudaStream_t>(stream));

            // (1b) csa_tq sparse decode over the TQ compressed tier.
            // num_sm_parts = 1 (single-partition arm writes out/lse
            // directly, natural-log LSE) — split-KV for the TQ pass is a
            // perf follow-up (TD-V4-TQ-DEVICE resolution note).
            sm120::decode::csa_tq::CsaTqDecodeParams tp;
            std::memset(&tp, 0, sizeof(tp));
            tp.b = a.rows;
            tp.s_q = 1;
            tp.h_q = h;
            tp.head_dim = dv;
            tp.qk_rope_head_dim = opts_.rope_dim;
            tp.sm_scale = opts_.sm_scale;
            tp.q_rot = tq_q_rot_;
            tp.q_rope = static_cast<const __nv_bfloat16*>(a.q_rope);
            tp.kv_cache = static_cast<const uint8_t*>(a.comp_cache);
            tp.indices = a.sparse_indices;
            tp.topk = a.topk;
            tp.stride_indices_b = a.topk;
            tp.stride_indices_s_q = a.topk;
            tp.centroids = tq_res_->device_centroids();
            tp.out = tq_out_f32_;
            tp.lse = tq_lse_;
            tp.stride_o_b = h * dv;
            tp.stride_o_s_q = h * dv;
            tp.stride_o_h_q = dv;
            tp.stride_lse_b = h;
            tp.stride_lse_s_q = h;
            tp.num_sm_parts = 1;
            tp.o_accum = o_accum_;        // unused at nsp=1; valid memory
            tp.lse_accum = lse_accum_;
            tp.num_splits_ptr = num_splits_;
            tp.tile_scheduler_metadata_ptr = sched_meta_;
            tp.stream = static_cast<cudaStream_t>(stream);
            sm120::decode::csa_tq::run_csa_tq_decode(tp);
            cudaError_t terr = cudaGetLastError();
            if (terr != cudaSuccess) {
                throw std::runtime_error(
                    std::string("csa_tq_decode launch failed: ") +
                    cudaGetErrorString(terr));
            }

            // (1c) Π^T rotate-back to the normal V space.
            sm120::prep::TqVRotateBackParams vr{};
            vr.out_rotated = tq_out_f32_;
            vr.Pi = tq_res_->device_Pi(a.layer_idx);
            vr.Pi_t = tq_res_->device_Pi_t(a.layer_idx);
            vr.out_final = static_cast<__nv_bfloat16*>(tq_out_bf16_);
            vr.batch_heads = bh;
            vr.d_c = dv;
            launch_tq_v_rotate_back(vr, static_cast<cudaStream_t>(stream));

            // (2) FP8 SWA-only pass into the caller buffers.
            V4AttentionArgs sa = a;
            sa.comp_cache = nullptr;
            sa.sparse_indices = nullptr;
            sa.topk = 0;
            const auto meta = swa_only_meta(a.rows);
            run_decode(sa, /*nsp=*/1, meta.first, meta.second, a.out,
                       a.lse, stream);

            // (3) fold the TQ partial into the SWA partial.
            V4LseMerge2Args ma{};
            ma.out_a = a.out;
            ma.lse_a = a.lse;
            ma.out_b = tq_out_bf16_;
            ma.lse_b = tq_lse_;
            ma.rows = a.rows;
            ma.heads = h;
            ma.head_dim = dv;
            launch_v4_lse_merge2(ma, stream);

            run_epilogues(a.rows, a.out, a.lse, a.sinks,
                          a.sink_head_offset, a.positions, a.rope_table,
                          stream, a.num_heads_real);
            return;
        }

        run_decode(a, nsp, sched, splits, a.out, a.lse, stream);
        run_epilogues(a.rows, a.out, a.lse, a.sinks, a.sink_head_offset,
                      a.positions, a.rope_table, stream, a.num_heads_real);
    }

    void gather_entries(const V4EntryGatherArgs& a, void** staging_base_out,
                        void* stream) {
        if (a.dst_row_offset + a.count > staged_capacity_) {
            throw std::runtime_error("csa_hca gather_entries: staging overflow");
        }
        launch_v4_entry_gather(staging_, a.src_cache, a.slots, a.count,
                               kEntryBytes, a.dst_row_offset, stream);
        if (staging_base_out) *staging_base_out = staging_;
    }

    // ── Lightning indexer selection ─────────────────────────────────────────

    void lightning_select(const V4LightningArgs& a, void* stream) {
        if (a.rows <= 0) return;
        if (a.rows > attn_rows_capacity_) {
            throw std::runtime_error("csa_hca lightning_select: rows > cap");
        }
        sm120::indexer::LightningScoreMqaBatchedParams sp{};
        sp.q_all = static_cast<const __nv_bfloat16*>(a.q_proj);
        sp.score_proj_all = static_cast<const float*>(a.score_w);
        sp.row_num_blocks = a.row_num_blocks;
        sp.k_cache = nullptr;
        sp.k_scales = nullptr;
        sp.k_page_table = a.k_page_table;
        sp.page_table_stride = a.page_table_stride;
        sp.page_tokens = opts_.idx_entries_per_page;
        sp.scores_out = scores_scratch_;
        sp.scores_stride = opts_.max_index_blocks;
        sp.num_rows = a.rows;
        sp.max_num_blocks = opts_.max_index_blocks;
        sp.index_n_heads = opts_.index_n_heads;
        sp.index_head_dim = opts_.index_head_dim;
        launch_lightning_score_mqa_batched(sp,
                                           static_cast<cudaStream_t>(stream));

        sm120::indexer::LightningTopkBatchedParams tp{};
        tp.scores = scores_scratch_;
        tp.scores_stride = opts_.max_index_blocks;
        tp.block_endpoints = a.block_endpoints;
        tp.row_num_blocks = a.row_num_blocks;
        tp.row_query_position = a.query_positions;
        tp.output_indices = a.indices_out;
        tp.output_scores = topk_scores_scratch_;
        tp.effective_k_out = topk_effk_scratch_;
        tp.num_rows = a.rows;
        tp.topk = a.topk;
        if (a.topk > opts_.topk) {
            throw std::runtime_error("csa_hca lightning_select: topk > cap");
        }
        launch_lightning_topk_batched(tp, static_cast<cudaStream_t>(stream));
    }

    // ── Graph runners (V4-5e) ───────────────────────────────────────────────

    void* graph_init(const V4GraphInitArgs& a) {
        device_.set_device();
        auto runner = std::make_unique<sm120::graph::CsaFp8DecodeGraphRunner>();
        sm120::graph::CsaFp8DecodeGraphConfig cfg{};
        cfg.batch_size = a.batch;
        cfg.s_q = 1;
        cfg.h_q = opts_.h_q;
        cfg.topk = a.topk;
        cfg.sm_scale = opts_.sm_scale;
        cfg.num_sm_parts = a.num_sm_parts > 0 ? a.num_sm_parts : 1;
        cfg.compressed_kv = const_cast<void*>(a.comp_cache);
        cfg.compressed_page_block_size = opts_.csa_entries_per_page;
        cfg.swa_kv = const_cast<void*>(a.swa_cache);
        cfg.swa_page_block_size = opts_.swa_page_tokens;
        cfg.max_swa_blocks = a.max_swa_blocks;
        cudaStream_t init_stream = nullptr;
        runner->init(cfg, init_stream);

        // One-shot scheduler metadata into the runner's buffers (constant
        // for the fixed topk; SWA growth never reschedules — see dossier
        // E+F correction 8).
        if (a.topk > 0) {
            GetMlaMetadataParams mp{};
            mp.seqlens_k_ptr = nullptr;
            mp.tile_scheduler_metadata_ptr = runner->sched_meta_ptr()
                ? static_cast<int*>(runner->sched_meta_ptr()) : nullptr;
            mp.num_splits_ptr = runner->num_splits_ptr();
            mp.batch_size = a.batch;
            mp.block_size_n = 64;
            mp.fixed_overhead_num_blocks = 1;
            mp.num_sm_parts = cfg.num_sm_parts;
            mp.topk = a.topk;
            run_get_mla_metadata_kernel(mp, init_stream);
        } else {
            std::vector<int> meta(TileSchedulerMetaDataSize, 0);
            meta[2] = a.batch - 1;
            device_.memcpy_h2d(runner->sched_meta_ptr(), meta.data(),
                               meta.size() * sizeof(int));
            std::vector<int> zeros(static_cast<size_t>(a.batch) + 1, 0);
            device_.memcpy_h2d(runner->num_splits_ptr(), zeros.data(),
                               zeros.size() * sizeof(int));
        }
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("csa_hca graph_init: ") +
                                     cudaGetErrorString(err));
        }
        graph_runners_.push_back(std::move(runner));
        return graph_runners_.back().get();
    }

    void graph_update(void* runner, const void* q_nope, const void* q_rope,
                      const int* indices, const int* swa_bt,
                      const int* swa_sl, void* stream) {
        static_cast<sm120::graph::CsaFp8DecodeGraphRunner*>(runner)->update(
            q_nope, q_rope, indices, swa_bt, swa_sl,
            static_cast<cudaStream_t>(stream));
    }
    void graph_replay(void* runner, void* stream) {
        static_cast<sm120::graph::CsaFp8DecodeGraphRunner*>(runner)->replay(
            static_cast<cudaStream_t>(stream));
    }
    void graph_outputs(void* runner, void** out, float** lse) {
        auto* r = static_cast<sm120::graph::CsaFp8DecodeGraphRunner*>(runner);
        if (out) *out = r->out_ptr();
        if (lse) *lse = r->lse_ptr();
    }
    void graph_epilogue(void* runner, int rows, const void* sinks,
                        int sink_head_offset, const int* positions,
                        const void* rope_table, void* stream) {
        auto* r = static_cast<sm120::graph::CsaFp8DecodeGraphRunner*>(runner);
        run_epilogues(rows, r->out_ptr(), r->lse_ptr(), sinks,
                      sink_head_offset, positions, rope_table, stream);
    }

private:
    void free_scratch() {
        if (staging_) device_.device_free(staging_);
        if (o_accum_) device_.device_free(o_accum_);
        if (lse_accum_) device_.device_free(lse_accum_);
        if (sched_meta_) device_.device_free(sched_meta_);
        if (num_splits_) device_.device_free(num_splits_);
        if (scores_scratch_) device_.device_free(scores_scratch_);
        if (topk_scores_scratch_) device_.device_free(topk_scores_scratch_);
        if (topk_effk_scratch_) device_.device_free(topk_effk_scratch_);
        for (auto& [k, v] : swa_meta_cache_) {
            device_.device_free(v.first);
            device_.device_free(v.second);
        }
        swa_meta_cache_.clear();
        staging_ = nullptr;
        o_accum_ = lse_accum_ = scores_scratch_ = topk_scores_scratch_ = nullptr;
        topk_effk_scratch_ = nullptr;
        sched_meta_ = num_splits_ = nullptr;
        if (tq_q_rot_) device_.device_free(tq_q_rot_);
        if (tq_out_f32_) device_.device_free(tq_out_f32_);
        if (tq_out_bf16_) device_.device_free(tq_out_bf16_);
        if (tq_lse_) device_.device_free(tq_lse_);
        if (tq_stage_rows_) device_.device_free(tq_stage_rows_);
        if (tq_stage_rope_) device_.device_free(tq_stage_rope_);
        tq_q_rot_ = tq_out_f32_ = tq_lse_ = nullptr;
        tq_out_bf16_ = tq_stage_rows_ = tq_stage_rope_ = nullptr;
        staged_capacity_ = 0;
        attn_rows_capacity_ = 0;
    }

    void run_decode(const V4AttentionArgs& a, int nsp, const int* sched,
                    const int* splits, void* out, float* lse, void* stream) {
        const int h = opts_.h_q;
        const int dv = opts_.head_dim;

        sm120::decode::csa_fp8::CsaFp8DecodeParams p;
        std::memset(&p, 0, sizeof(p));
        p.b = a.rows;
        p.s_q = 1;
        p.h_q = h;
        p.sm_scale = opts_.sm_scale;
        p.sm_scale_log2 = opts_.sm_scale * 1.4426950408889634f;
        p.q_nope = static_cast<cutlass::bfloat16_t*>(
            const_cast<void*>(a.q_nope));
        p.q_rope = static_cast<cutlass::bfloat16_t*>(
            const_cast<void*>(a.q_rope));
        p.q_scales = nullptr;

        // Compressed cache: when SWA-only, point at the SWA cache (never
        // read — zero compressed iterations) so the pointer stays valid.
        p.compressed_kv = static_cast<const char*>(
            a.comp_cache ? a.comp_cache : a.swa_cache);
        p.compressed_page_block_size = opts_.csa_entries_per_page;
        p.stride_compressed_block =
            static_cast<int64_t>(opts_.csa_entries_per_page) * kEntryBytes;
        p.stride_compressed_row = kEntryBytes;

        p.sparse_indices = a.sparse_indices;
        p.topk = a.topk;
        p.stride_indices_b = a.topk;  // s_q = 1
        p.stride_indices_s_q = a.topk;

        p.swa_kv = static_cast<const char*>(
            a.swa_cache ? a.swa_cache : a.comp_cache);
        // Superchunk batched prefill overrides the SWA page granularity
        // (1 = per-token index-list block table over the entry staging).
        const int spbs = a.swa_page_block_size > 0 ? a.swa_page_block_size
                                                   : opts_.swa_page_tokens;
        p.swa_page_block_size = spbs;
        p.stride_swa_block = static_cast<int64_t>(spbs) * kEntryBytes;
        p.stride_swa_row = kEntryBytes;
        p.swa_block_table = a.swa_block_table;
        p.swa_block_table_stride = a.swa_block_table_stride;
        p.swa_seqlens = a.swa_seqlens;

        p.out = static_cast<cutlass::bfloat16_t*>(out);
        p.lse = lse;
        p.stride_q_b = h * opts_.head_dim;
        p.stride_q_s_q = h * opts_.head_dim;
        p.stride_q_h_q = opts_.head_dim;
        p.stride_o_b = h * dv;
        p.stride_o_s_q = h * dv;
        p.stride_o_h_q = dv;
        p.stride_lse_b = h;
        p.stride_lse_s_q = h;

        p.lse_accum = lse_accum_;
        p.o_accum = o_accum_;
        p.stride_lse_accum_split = h;
        p.stride_lse_accum_s_q = h;
        p.stride_o_accum_split = h * dv;
        p.stride_o_accum_s_q = h * dv;
        p.stride_o_accum_h_q = dv;
        p.tile_scheduler_metadata_ptr =
            reinterpret_cast<sm120::decode::csa_fp8::DecodingSchedMeta*>(
                const_cast<int*>(sched));
        p.num_splits_ptr = const_cast<int*>(splits);
        p.num_sm_parts = nsp;
        p.deterministic_reduce = opts_.deterministic_reduce;  // DET-REDUCE
        p.stream = static_cast<cudaStream_t>(stream);

        if (h <= 64) {
            sm120::decode::csa_fp8::run_csa_fp8_decode_kernel<64>(p);
        } else {
            sm120::decode::csa_fp8::run_csa_fp8_decode_kernel<128>(p);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("csa_fp8_decode launch failed: ") +
                cudaGetErrorString(err));
        }

        if (nsp > 1) {
            MlaCombineParams cp{};
            cp.b = a.rows;
            cp.h_q = h;
            cp.h_k = 1;
            cp.q_seq_per_hk = h;  // h_q / h_k * s_q
            cp.d_v = dv;
            cp.o_ptr = out;
            cp.softmax_lse_ptr = lse;
            cp.o_batch_stride = static_cast<int64_t>(h) * dv;
            cp.o_row_stride = dv;
            cp.o_head_stride = dv;  // s_q=1: heads are the row dimension
            cp.num_splits_ptr = const_cast<int*>(splits);
            cp.num_sm_parts = nsp;
            cp.softmax_lseaccum_ptr = lse_accum_;
            cp.oaccum_ptr = o_accum_;
            run_mla_combine_kernel<cutlass::bfloat16_t>(
                cp, static_cast<cudaStream_t>(stream));
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("mla_combine launch failed: ") +
                    cudaGetErrorString(err));
            }
        }
    }

    void run_epilogues(int rows, void* out, float* lse, const void* sinks,
                       int sink_head_offset, const int* positions,
                       const void* rope_table, void* stream,
                       int num_heads_real = 0) {
        // V4-2c TP padding: run the epilogues over the REAL heads only.
        // With rows == 1 (the V4 per-row contract) head-major addressing
        // stays exact even though the buffers are h_q-padded.
        const int h = (num_heads_real > 0 && num_heads_real <= opts_.h_q)
                          ? num_heads_real : opts_.h_q;
        if (h < opts_.h_q && rows != 1) {
            throw std::runtime_error(
                "csa_hca run_epilogues: padded head counts require rows==1");
        }
        if (sinks) {
            launch_v4_attn_sinks(out, lse, sinks, sink_head_offset, rows,
                                 h, opts_.head_dim, stream);
        }
        if (positions && rope_table) {
            launch_v4_out_inverse_rope(out, positions, rope_table, rows,
                                       h, opts_.head_dim,
                                       opts_.rope_dim, stream);
        }
    }

    CudaSm120DeviceBackend device_;
    V4DeviceOptions opts_{};
    int num_sm_ = 0;

    // Workspaces.
    void* staging_ = nullptr;        // [staged_capacity, 1160] entry rows
    float* o_accum_ = nullptr;       // [rows+nsp+1, h, dv] f32
    float* lse_accum_ = nullptr;     // [rows+nsp+1, h] f32
    int* sched_meta_ = nullptr;      // [nsp, 8]
    int* num_splits_ = nullptr;      // [rows+1]
    float* scores_scratch_ = nullptr;  // [rows, max_index_blocks]
    float* topk_scores_scratch_ = nullptr;  // [rows, topk]
    int* topk_effk_scratch_ = nullptr;       // [rows]
    std::unordered_map<int, std::pair<int*, int*>> swa_meta_cache_;
    int staged_capacity_ = 0;
    int attn_rows_capacity_ = 0;

    std::vector<std::unique_ptr<sm120::graph::CsaFp8DecodeGraphRunner>>
        graph_runners_;

    // V4-5T TQ codec state (null unless a tier codec is kTq4).
    const TqResources* tq_res_ = nullptr;
    float* tq_q_rot_ = nullptr;     // [rows, h, dv] F32 rotated query
    float* tq_out_f32_ = nullptr;   // [rows, h, dv] F32 rotated partial
    void* tq_out_bf16_ = nullptr;   // [rows, h, dv] BF16 rotate-back
    float* tq_lse_ = nullptr;       // [rows, h] F32 natural-log LSE
    void* tq_stage_rows_ = nullptr; // [rows, dv] BF16 compress staging
    void* tq_stage_rope_ = nullptr; // [rows, rope_dim] BF16 roped tails
};

// ── Factory + bridges ───────────────────────────────────────────────────────

std::unique_ptr<AttentionDevice> make_csa_hca_sm120_attention_device(
    config::GpuRef gpu) {
    return std::make_unique<CsaHcaSm120AttentionDevice>(std::move(gpu));
}

void csa_hca_device_configure(AttentionDevice* dev, const V4DeviceOptions& o) {
    static_cast<CsaHcaSm120AttentionDevice*>(dev)->configure(o);
}
void csa_hca_device_set_scratch(AttentionDevice* dev, int max_staged_entries) {
    static_cast<CsaHcaSm120AttentionDevice*>(dev)->set_scratch(
        max_staged_entries);
}
void csa_hca_device_attention(AttentionDevice* dev, const V4AttentionArgs& a) {
    static_cast<CsaHcaSm120AttentionDevice*>(dev)->attention(a);
}
void csa_hca_device_set_tq(AttentionDevice* dev, const TqResources* res) {
    static_cast<CsaHcaSm120AttentionDevice*>(dev)->set_tq(res);
}
void csa_hca_device_tq_compress_insert(AttentionDevice* dev,
                                       const V4CompressArgs& ca,
                                       int layer_idx, void* stream) {
    static_cast<CsaHcaSm120AttentionDevice*>(dev)->tq_compress_insert(
        ca, layer_idx, stream);
}
void csa_hca_device_gather_entries(AttentionDevice* dev,
                                   const V4EntryGatherArgs& a,
                                   void** staging_base_out, void* stream) {
    static_cast<CsaHcaSm120AttentionDevice*>(dev)->gather_entries(
        a, staging_base_out, stream);
}
void csa_hca_device_lightning_select(AttentionDevice* dev,
                                     const V4LightningArgs& a, void* stream) {
    static_cast<CsaHcaSm120AttentionDevice*>(dev)->lightning_select(a, stream);
}
void* csa_hca_device_graph_init(AttentionDevice* dev,
                                const V4GraphInitArgs& a) {
    return static_cast<CsaHcaSm120AttentionDevice*>(dev)->graph_init(a);
}
void csa_hca_device_graph_update(AttentionDevice* dev, void* runner,
                                 const void* q_nope, const void* q_rope,
                                 const int* sparse_indices,
                                 const int* swa_block_table,
                                 const int* swa_seqlens, void* stream) {
    static_cast<CsaHcaSm120AttentionDevice*>(dev)->graph_update(
        runner, q_nope, q_rope, sparse_indices, swa_block_table, swa_seqlens,
        stream);
}
void csa_hca_device_graph_replay(AttentionDevice* dev, void* runner,
                                 void* stream) {
    static_cast<CsaHcaSm120AttentionDevice*>(dev)->graph_replay(runner,
                                                                stream);
}
void csa_hca_device_graph_outputs(AttentionDevice* dev, void* runner,
                                  void** out, float** lse) {
    static_cast<CsaHcaSm120AttentionDevice*>(dev)->graph_outputs(runner, out,
                                                                 lse);
}
void csa_hca_device_graph_epilogue(AttentionDevice* dev, void* runner,
                                   int rows, const void* sinks,
                                   int sink_head_offset, const int* positions,
                                   const void* rope_table, void* stream) {
    static_cast<CsaHcaSm120AttentionDevice*>(dev)->graph_epilogue(
        runner, rows, sinks, sink_head_offset, positions, rope_table, stream);
}

}  // namespace layerstorm::compute
