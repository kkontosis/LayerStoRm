// MoE driver — dense FFN early-out, routed-expert dispatch (single rank),
// shared experts, seam/routing publishers, and env gates.
// Part of CommandDispatcher — see command_dispatcher.h. Split from
// dispatch_moe.cpp (pure code motion).

#include "daemon/command_dispatcher.h"
#include "daemon/dispatch_detail.h"
#include "daemon/moe/arch_mla_moe.h"
#include "daemon/moe/arch_deepseek_v4_moe.h"
#include "daemon/moe/quant_routes.h"
#include "daemon/ep_residency_dedup.h"  // INV-MOE-EP-DISJOINT force-ON dedup
#include "daemon/expert_lifecycle_manager.h"
#include "core/perf_trace.h"

#include <algorithm>
#include <atomic>   // TD-DRIFT-ROOTCAUSE diagnostic dump sequence counter
#include <chrono>
#include <cstdio>  // I8b shadow-dump JSONL sink (fopen/fprintf)
#include <cstdlib>  // TD-DRIFT-ROOTCAUSE getenv gate
#include <cstring>
#include <limits>
#include <span>
#include <string>  // I8 shadow log assembly
#include <unordered_map>  // TD-PREFILL-FETCH-SEAM-SCALING wave scheduling
#include <unordered_set>  // TD-FAR-SLOT-RESERVE-STALL needed-key set
#include <vector>  // TD-DRIFT-ROOTCAUSE D2H staging

#include <spdlog/spdlog.h>

#include "compute/stream_manager.h"
#include "core/device_backend.h"
#include "core/expert_device.h"
#include "core/gpu_loader/far_evict.h"  // shared reroute-eviction fallback
#include "core/memory/expert_cache.h"
#include "core/transfer/transfer_engine.h"  // TD-FAR-STREAM-GATE: h2d barrier
#include "core/memory/numa_manager.h"  // I8 shadow: expert_home_node()
#include "model/quantization/gguf_kquant.h"  // GG-5b: GgufQuantInterface (per-projection type)
#include "core/statistics/expert_stats.h"  // ExpertStats recency feed (FETCH path)
#include "parallelism/dcp_communicator.h"
#include "compute/kernels/elementwise/residual_add.h"
#include "compute/kernels/mhc/mhc.h"
#include "smxx/permute/moe_permute.h"  // DET-REDUCE Phase 1b: fp32→bf16 EP-combine cast
#include "compute/kernels/moe/moe_gemm_meta.h"
#include "compute/kernels/moe/router_projection.h"
#include "compute/kernels/norm/rmsnorm.h"
#include "compute/kernels/moe/hash_gating.h"  // V4-4 hash-layer gating
#include "compute/kernels/sm120/gemm/bf16_gemm.h"  // V4-7b raw-BF16 shexp/dense FFN
#include "sm120/gating/topk_gating.h"

namespace layerstorm::daemon {

namespace {
// ── TD-DRIFT-ROOTCAUSE: DIAGNOSTIC-ONLY routing dump (off by default) ──────────
// Gated on LS_DRIFT_DUMP=<path>. When set, dumps a binary record per
// (sequence, layer, gpu) immediately after the top-K gate runs: the full
// pre-argmax router logit vector + the selected top-K expert ids + their gating
// weights. Two runs (same config twice, OR decay-on vs decay-off) can then be
// diffed at the logit level to locate the FIRST divergence and classify it
// (ULP-scale near-tie argmax flip vs large/garbage). UNSET = zero work, zero
// production impact. When SET it adds a D2H + full device sync, which DOES change
// timing — used only to compare dump-on vs dump-on, and (deliberately) as the
// race probe: forcing this serialization must not change the dumped routing if
// the pipeline is race-free. Record layout (little-endian):
//   int32 hdr[6] = {seq, layer_idx, gpu, num_tokens, n_experts, topk}
//   float logits[num_tokens*n_experts]
//   int32 idx   [num_tokens*topk]
//   float w     [num_tokens*topk]
void drift_dump_routing(compute::DeviceBackend* dev_be, void* stream,
                        const void* router_logits_dev,
                        const void* topk_indices_dev,
                        const void* topk_weights_dev,
                        int num_tokens, int n_experts, int topk,
                        uint32_t layer_idx, int gpu) {
    static const char* path = std::getenv("LS_DRIFT_DUMP");
    if (!path || !*path) return;
    if (!dev_be || !router_logits_dev || !topk_indices_dev || !topk_weights_dev)
        return;
    if (num_tokens <= 0 || n_experts <= 0 || topk <= 0) return;
    static std::FILE* fp = std::fopen(path, "wb");
    if (!fp) return;
    static std::atomic<uint64_t> seq{0};
    const size_t nlog = static_cast<size_t>(num_tokens) * n_experts;
    const size_t nk   = static_cast<size_t>(num_tokens) * topk;
    std::vector<float>   logits(nlog);
    std::vector<int32_t> idx(nk);
    std::vector<float>   w(nk);
    dev_be->set_device();
    dev_be->memcpy_d2h_async(logits.data(), router_logits_dev,
                             nlog * sizeof(float), stream);
    dev_be->memcpy_d2h_async(idx.data(), topk_indices_dev,
                             nk * sizeof(int32_t), stream);
    dev_be->memcpy_d2h_async(w.data(), topk_weights_dev,
                             nk * sizeof(float), stream);
    dev_be->synchronize_device();
    const uint64_t s = seq.fetch_add(1);
    int32_t hdr[6] = {static_cast<int32_t>(s), static_cast<int32_t>(layer_idx),
                      gpu, num_tokens, n_experts, topk};
    std::fwrite(hdr, sizeof(int32_t), 6, fp);
    std::fwrite(logits.data(), sizeof(float), nlog, fp);
    std::fwrite(idx.data(), sizeof(int32_t), nk, fp);
    std::fwrite(w.data(), sizeof(float), nk, fp);
    std::fflush(fp);
}
}  // namespace

// ── TD-DECODE-FFN-GRAPH (experiment) ──────────────────────────────────────
// Read once: is the routed-expert FFN decode CUDA graph enabled? Gated on the
// LS_MOE_FFN_GRAPH env var. INTENTIONALLY violates INV-0.6 for measurement.
bool CommandDispatcher::moe_ffn_graph_enabled() {
    if (moe_ffn_graph_enabled_ < 0) {
        // DEFAULT ON (user decision 2026-07-20; INV-0.6(a) is a permitted,
        // validated subgraph — ~2.5-3% MoE-phase win, bit-identical replay).
        // LS_MOE_FFN_GRAPH=0 opts out.
        const char* e = std::getenv("LS_MOE_FFN_GRAPH");
        moe_ffn_graph_enabled_ = (e && e[0] && e[0] == '0') ? 0 : 1;
        if (!moe_ffn_graph_enabled_)
            spdlog::info("routed-expert FFN decode CUDA graph DISABLED "
                         "(LS_MOE_FFN_GRAPH=0)");
    }
    return moe_ffn_graph_enabled_ == 1;
}

// ── INV-NCCL-GRAPH ──────────────────────────────────────────────────────────
// Read once: replay the per-layer fused MoE-combine collectives through
// captured per-rank CUDA graphs (LS_NCCL_GRAPH, default OFF until measured).
bool CommandDispatcher::nccl_graph_enabled() {
    if (nccl_graph_enabled_ < 0) {
        const char* e = std::getenv("LS_NCCL_GRAPH");
        nccl_graph_enabled_ = (e && e[0] && e[0] != '0') ? 1 : 0;
        if (nccl_graph_enabled_)
            spdlog::warn("INV-NCCL-GRAPH: captured NCCL combine graphs "
                         "ENABLED (LS_NCCL_GRAPH)");
    }
    return nccl_graph_enabled_ == 1;
}

// ── INV-MOE-OVERLAP: decode fetch-overlap split ────────────────────────────
// Read once: run resident-expert compute (wave-partial pass) concurrently with
// the missing-expert H2D fetch at decode. Default ON; LS_MOE_RESIDENT_OVERLAP=0
// restores the wait-then-single-pass behavior byte-identically.
bool CommandDispatcher::moe_resident_overlap_enabled() {
    if (moe_resident_overlap_enabled_ < 0) {
        const char* e = std::getenv("LS_MOE_RESIDENT_OVERLAP");
        moe_resident_overlap_enabled_ = (e && e[0] == '0') ? 0 : 1;
        if (!moe_resident_overlap_enabled_)
            spdlog::info("INV-MOE-OVERLAP: decode resident-overlap pass "
                         "DISABLED (LS_MOE_RESIDENT_OVERLAP=0)");
    }
    return moe_resident_overlap_enabled_ == 1;
}

// ── KD-3b: Fused MoE dispatch ────────────────────────────────────────────

void CommandDispatcher::publish_seam_routing(const InternalMoeParams& mp,
                                             uint32_t gpu, int expanded_tokens,
                                             void* stream) {
    if (!mp.emit_seam_checkpoint || !deps_.sideband_base)
        return;
    if (gpu >= moe_scratch_.size())
        return;
    const auto& scratch = moe_scratch_[gpu];
    if (!scratch.topk_weights || !scratch.topk_indices)
        return;

    const uint32_t expanded = static_cast<uint32_t>(std::min<int>(
        std::max(expanded_tokens, 0),
        static_cast<int>(ipc::IpcLayout::kMaxSeamRoutingExpanded)));
    if (expanded == 0)
        return;

    const uint32_t weights_bytes = expanded * static_cast<uint32_t>(sizeof(float));
    const uint32_t indices_bytes = expanded * static_cast<uint32_t>(sizeof(int32_t));
    uint8_t* w_dst = deps_.sideband_base + ipc::IpcLayout::kSeamCheckpointOff;
    uint8_t* i_dst = w_dst + weights_bytes;
    if (deps_.cuda_kernels_enabled) {
        auto* dev_be = deps_.device_backends[gpu];
        dev_be->memcpy_d2h_async(w_dst, scratch.topk_weights, weights_bytes, stream);
        dev_be->memcpy_d2h_async(i_dst, scratch.topk_indices, indices_bytes, stream);
    } else {
        std::memcpy(w_dst, scratch.topk_weights, weights_bytes);
        std::memcpy(i_dst, scratch.topk_indices, indices_bytes);
    }
    last_seam_checkpoint_ = SeamCheckpoint{
        .checkpoint_type = static_cast<uint8_t>(ipc::CheckpointType::kSeamRouting),
        .host_buf_offset = static_cast<uint32_t>(ipc::IpcLayout::kSeamCheckpointOff),
        .data_bytes      = weights_bytes + indices_bytes,
    };
}

// F-4/F-3: publish the routed top-K (RoutingExportHeader + weights f32 + indices
// i32) into the canonical sideband routing-export slot (kRoutingExportOff). Read
// from moe_scratch_[gpu] (written by either the MoE gate, F-4, or the attention
// gate, F-1/F-3). Same slot + layout regardless of producer (cluster-unified).
void CommandDispatcher::publish_routing_export(uint32_t gpu, int num_tokens,
                                               int topk, uint32_t layer_idx,
                                               void* stream,
                                               int src_row_offset) {
    if (!deps_.sideband_base || gpu >= moe_scratch_.size())
        return;
    const auto& scratch = moe_scratch_[gpu];
    if (!scratch.topk_weights || !scratch.topk_indices)
        return;
    const int export_tokens =
        std::min(num_tokens, static_cast<int>(ipc::kMaxRoutingExportTokens));
    const int export_topk =
        std::min(topk, static_cast<int>(ipc::kRoutingExportMaxTopk));
    if (export_tokens <= 0 || export_topk <= 0)
        return;
    // TD-PREFILL-SUPERCHUNK: export a sub-chunk's rows stored at its row
    // offset in moe_scratch_ (sideband rows stay [0, export_tokens)).
    const void* topk_w_src = static_cast<const float*>(scratch.topk_weights)
        + static_cast<size_t>(src_row_offset) * topk;
    const void* topk_i_src = static_cast<const int32_t*>(scratch.topk_indices)
        + static_cast<size_t>(src_row_offset) * topk;
    auto* hdr = reinterpret_cast<ipc::RoutingExportHeader*>(
        deps_.sideband_base + ipc::IpcLayout::kRoutingExportOff);
    hdr->num_tokens = static_cast<uint32_t>(export_tokens);
    hdr->topk       = static_cast<uint32_t>(export_topk);
    hdr->layer_idx  = layer_idx;
    hdr->_pad       = 0;
    void* w_dst = deps_.sideband_base + ipc::IpcLayout::kRoutingExportWeightsOff;
    void* i_dst = deps_.sideband_base + ipc::IpcLayout::kRoutingExportIndicesOff;
    const size_t w_bytes =
        static_cast<size_t>(export_tokens) * export_topk * sizeof(float);
    const size_t i_bytes =
        static_cast<size_t>(export_tokens) * export_topk * sizeof(int32_t);
    if (export_topk == topk) {
        if (deps_.cuda_kernels_enabled) {
            auto* dev_be = deps_.device_backends[gpu];
            dev_be->memcpy_d2h_async(w_dst, topk_w_src, w_bytes, stream);
            dev_be->memcpy_d2h_async(i_dst, topk_i_src, i_bytes, stream);
        } else {
            std::memcpy(w_dst, topk_w_src, w_bytes);
            std::memcpy(i_dst, topk_i_src, i_bytes);
        }
    } else {
        const auto* w_src = static_cast<const float*>(topk_w_src);
        const auto* i_src = static_cast<const int32_t*>(topk_i_src);
        auto* w_out = static_cast<float*>(w_dst);
        auto* i_out = static_cast<int32_t*>(i_dst);
        for (int t = 0; t < export_tokens; ++t) {
            const size_t rb_w = static_cast<size_t>(export_topk) * sizeof(float);
            const size_t rb_i = static_cast<size_t>(export_topk) * sizeof(int32_t);
            if (deps_.cuda_kernels_enabled) {
                auto* dev_be = deps_.device_backends[gpu];
                dev_be->memcpy_d2h_async(w_out + t * export_topk,
                                         w_src + t * topk, rb_w, stream);
                dev_be->memcpy_d2h_async(i_out + t * export_topk,
                                         i_src + t * topk, rb_i, stream);
            } else {
                std::memcpy(w_out + t * export_topk, w_src + t * topk, rb_w);
                std::memcpy(i_out + t * export_topk, i_src + t * topk, rb_i);
            }
        }
    }
}

bool CommandDispatcher::dispatch_moe_internal(const InternalMoeParams& mp) {
    const uint32_t gpu = mp.gpu_idx;

    auto* dev = expert_dev(gpu);
    if (!dev) {
        spdlog::warn("dispatch_moe: no expert device for gpu {}", gpu);
        return false;
    }
    if (gpu >= moe_scratch_.size() || !moe_scratch_[gpu].topk_weights) {
        spdlog::warn("dispatch_moe: scratch not ready for gpu {}", gpu);
        return false;
    }
    if (!deps_.live_config) {
        spdlog::warn("dispatch_moe: no live_config");
        return false;
    }

    // TD-PREFILL-MOE-BIG: batches beyond the single-shot chunk capacity run
    // the chunked grouped-GEMM pipeline (dispatch_moe_big.cpp) — the transient
    // scratch is sized for moe_chunk_capacity_ tokens only. CUDA-gated: the
    // null-backend path below launches no kernels, so it stays shape-safe at
    // any num_seqs (test parity). Batches <= the chunk capacity (all decode)
    // keep the byte-identical legacy single-shot path (INV-MOE-BIG-1).
    if (deps_.cuda_kernels_enabled
        && static_cast<int>(mp.num_seqs) > moe_chunk_capacity_) {
        return dispatch_moe_chunked_internal(mp);
    }

    const auto& mc = deps_.live_config->model;
    const auto& scratch = moe_scratch_[gpu];
    const int num_tokens = static_cast<int>(mp.num_seqs);
    const int topk = (mp.topk_override > 0) ? mp.topk_override : mc.num_experts_per_tok;
    const int hidden = mc.hidden_size;
    const int intermediate = mc.moe_intermediate_size;
    const int n_experts = mc.n_routed_experts;
    const int expanded_tokens = num_tokens * topk;

    // MoE by-model split (moe/arch_base.h): the model-special driver phases
    // run through the MoeArch hooks. Selection mirrors the attention
    // driver's is_v4 condition; every hook body keeps its original
    // data-gated condition verbatim and falls back to the common base body
    // (INV-MOE-ARCH), so behavior is identical under any config. The arch
    // objects are stateless facades over this dispatcher (friends),
    // constructed once.
    const bool is_v4 =
        mc.architecture == config::Architecture::deepseek_v4;
    if (!moe_arch_mla_) {
        moe_arch_mla_ = std::make_unique<ArchMlaMoe>(*this);
        moe_arch_v4_  = std::make_unique<ArchDeepseekV4Moe>(*this);
    }
    MoeArch& arch = is_v4 ? *moe_arch_v4_ : *moe_arch_mla_;

    dev->set_device();
    void* stream = deps_.stream_manager->stream(
        static_cast<int>(gpu), compute::StreamId::kExpertFfn);

    // KD-R2: resolve pair for this GPU (if TP).
    const int pair_idx = resolve_pair_idx(gpu);

    // KD-R2: wait for attention residual on kAttention before reading.
    if (pair_idx >= 0) {
        const auto& pair = deps_.hidden_state_pairs[pair_idx];
        if (pair.attn_moe_event) {
            deps_.stream_manager->wait_event(
                static_cast<int>(gpu),
                compute::StreamId::kExpertFfn,
                pair.attn_moe_event);
        }
    }

    // Resolve hidden state input (needed by Step 0 and Step 2).
    void* hidden_input = nullptr;
    if (pair_idx >= 0) {
        hidden_input = deps_.hidden_state_pairs[pair_idx].moe_buf;
    } else if (gpu < deps_.fused_moe_hidden_states.size()) {
        hidden_input = deps_.fused_moe_hidden_states[gpu];
    }

    // TD-PREFILL-NONDET diagnostic: MoE-INPUT hidden (= the attention
    // command's committed output for this layer) — 'Min '. Env-gated.
    // V4-5b mHC: TP ranks hold the hc-stream residual (hc_streams*hidden wide);
    // EP-xTP extras hold the collapsed 4096-wide broadcast.
    seam_dump_hidden(0x206e694du /*'Min '*/, static_cast<int>(mp.layer_idx),
                     static_cast<int>(gpu), hidden_input, num_tokens,
                     pair_idx >= 0 ? hidden * deps_.hc_streams : hidden);

    // KD-3h-b: Pre-MoE RMSNorm (ffn_norm). Normalized output goes to router,
    // permute, and shared expert. Raw hidden_input kept for Step 8 residual add.
    // V4-5b mHC: hc_pre collapses the hc-stream residual to the module input
    // first (emitting the post/comb coefficients the Step-8 hc_post consumes);
    // the norm then reads the collapsed x. Gated on norm_w presence — EP-xTP
    // extras (no norm weights) receive the already collapsed+normed broadcast.
    void* norm_input = hidden_input;
    if (deps_.cuda_kernels_enabled && hidden_input && scratch.normalized_hidden) {
        const parallelism::AttentionLayerWeights* lw = nullptr;
        const int layer = static_cast<int>(mp.layer_idx);
        // KD-R2: use pair.rank directly instead of scanning tp_gpus.
        if (pair_idx >= 0 && layer < static_cast<int>(deps_.per_layer_attn_weights.size())) {
            const int r = deps_.hidden_state_pairs[pair_idx].rank;
            if (r >= 0 && r < static_cast<int>(deps_.per_layer_attn_weights[layer].size())) {
                lw = &deps_.per_layer_attn_weights[layer][r];
            }
        } else if (deps_.dcp_executor && layer < static_cast<int>(deps_.per_layer_attn_weights.size())) {
            const auto& tp_gpus = deps_.dcp_executor->gpus();
            for (int r = 0; r < static_cast<int>(tp_gpus.size()); ++r) {
                if (tp_gpus[r].position == static_cast<int>(gpu) &&
                    r < static_cast<int>(deps_.per_layer_attn_weights[layer].size())) {
                    lw = &deps_.per_layer_attn_weights[layer][r];
                    break;
                }
            }
        }
        const void* norm_w = lw ? lw->post_attention_layernorm : nullptr;
        const void* rms_src = hidden_input;
        // V4-5b mHC hc_pre collapse (ArchDeepseekV4Moe::collapse_hidden);
        // base arch: identity (rms_src stays hidden_input).
        if (!arch.collapse_hidden(mp, gpu, lw, norm_w, hidden_input,
                                  num_tokens, hidden, stream, rms_src))
            return false;
        if (norm_w) {
            compute::launch_rmsnorm(
                scratch.normalized_hidden, rms_src, norm_w,
                deps_.live_config->model.rms_norm_eps,
                num_tokens, hidden,
                compute::NormDtype::kBFloat16, stream);
            norm_input = scratch.normalized_hidden;
        }
        // C-6 Task A: record the input-ready event right after the RMSNorm so
        // the CPU-expert fold's input D2H never races the norm — even on the
        // zero-resident skip_routed path below (where the router never runs and
        // the later, post-router record is skipped). The post-router record
        // above overwrites this on the normal path (covers norm+router).
        record_cpu_input_event(gpu, num_tokens, mp.layer_idx, stream);
    }

    // ── C-6 early-kick: CPU-INPUT PRIME pass ────────────────────────────────
    // The ffn_norm above produced normalized_hidden on this rank's kExpertFfn
    // stream and its input-ready event is recorded. The routed top-K already sits
    // in moe_scratch_ (attention emit_gating), so the host CPU-expert FFN has all
    // its inputs. Return NOW — before the router / permute / expert GEMMs — so the
    // caller (handle_fetch_and_run) can kick the host FFN at fetch-launch time and
    // overlap the missing-expert H2D window. Nothing else was touched (accumulator
    // / moe_output / bitset untouched), and the finalize's own Phase-1 recomputes
    // the (bit-identical) norm, so this prime is a pure no-side-effect input hoist.
    if (mp.prime_cpu_input_only) {
        return deps_.cuda_kernels_enabled && hidden_input
               && scratch.normalized_hidden
               && norm_input == scratch.normalized_hidden;
    }

    // GG-5b/GG-5c/GG-9: weight-quant route derivation — the shared single
    // source (moe/quant_routes.h; formerly a per-file copy that moe_big.cpp
    // mirrored under a keep-in-sync comment). Locals below alias the struct
    // so the pipeline sites read exactly as before.
    const MoeQuantRoutes qr = build_moe_quant_routes(
        deps_, mp.layer_idx, intermediate, hidden, "dispatch_moe_internal");
    const bool use_fp8 = qr.use_fp8;
    const bool use_gguf = qr.use_gguf;
    const compute::GgufGemmStrategy gguf_strategy = qr.gguf_strategy;
    // GG-5c: ordinal-preserving cast for the dense/shared structs' OWN
    // per-projection types (may differ from the routed types on a mixed GGUF).
    auto to_gguf_compute = [](model::GgufKQuantType t) -> compute::GgufQuantType {
        return MoeQuantRoutes::to_gguf_compute(t);
    };
    const compute::GgufQuantType gguf_gate_type = qr.gguf_gate_type;
    const compute::GgufQuantType gguf_up_type   = qr.gguf_up_type;
    const compute::GgufQuantType gguf_down_type = qr.gguf_down_type;
    // Per-projection in-slot offset: GGUF uses the per-layer offsets; all
    // other formats keep the CacheEntry's uniform global offsets (GG-9).
    auto gate_off_fn = [&](const memory::CacheEntry* e) { return qr.gate_off(e); };
    auto up_off_fn   = [&](const memory::CacheEntry* e) { return qr.up_off(e); };
    auto down_off_fn = [&](const memory::CacheEntry* e) { return qr.down_off(e); };

    // TD-89af: Reset miss count at entry so dense layers and error paths
    // don't carry stale values from previous MoE dispatches.
    last_moe_miss_count_ = 0;

    // F-7: Reset seam checkpoint so a stale {offset,bytes} from a prior dispatch
    // is never reported when this op publishes no routing.
    last_seam_checkpoint_ = SeamCheckpoint{};

    // KD-4f-c: Dense layer early-out — use DenseFFNWeights instead of MoE pipeline.
    // Dense layers (< first_k_dense_replace) have no routed or shared experts.
    const int first_k_dense = deps_.live_config->model.first_k_dense_replace;
    if (static_cast<int>(mp.layer_idx) < first_k_dense) {
        const Deps::DenseFFNWeights* dw = nullptr;
        if (mp.layer_idx < deps_.dense_ffn_weight_ptrs.size() &&
            gpu < deps_.dense_ffn_weight_ptrs[mp.layer_idx].size()) {
            dw = &deps_.dense_ffn_weight_ptrs[mp.layer_idx][gpu];
        }
        if (!dw || !dw->gate_up || !dw->down) {
            // No dense weights available. Under null backends (cuda_kernels_enabled=false),
            // this is expected — just do the residual identity (same as MoE no-op path).
            if (!deps_.cuda_kernels_enabled) {
                if (pair_idx >= 0) {
                    deps_.hidden_state_pairs[pair_idx].commit(
                        static_cast<size_t>(num_tokens) * hidden * 2, stream,
                        deps_.device_backends[gpu]);
                    // TD-74c: Record moe_attn_event for dense layers too.
                    if (deps_.hidden_state_pairs[pair_idx].moe_attn_event) {
                        deps_.stream_manager->record_event(
                            deps_.hidden_state_pairs[pair_idx].moe_attn_event,
                            static_cast<int>(gpu), compute::StreamId::kExpertFfn);
                    }
                }
                return true;
            }
            spdlog::warn("dispatch_moe: dense layer {} missing DenseFFNWeights on gpu {}",
                         mp.layer_idx, gpu);
            return false;
        }

        // KD-4g: pre-allreduce phase — run gate+up, SwiGLU, down GEMM.
        if (mp.phase != MoeDispatchPhase::kPostAllreduce) {
        const int tp_val = std::max(1, deps_.live_config->parallelism.tensor_parallelism);
        const int I_dense_local = deps_.live_config->model.intermediate_size / tp_val;

        // Write expert_offsets = {0, num_tokens} for single-expert grouped GEMM.
        // Reuse shared_expert_offsets/problem_sizes/sf_offsets (no conflict: dense
        // layers have no shared expert, these buffers are written before use).
        // Guard: these buffers are only allocated when n_shared_experts > 0.
        auto* gpu_dev = deps_.device_backends[gpu];
        if (!scratch.shared_expert_offsets || !scratch.shared_problem_sizes ||
            !scratch.shared_sf_offsets) {
            spdlog::warn("dispatch_moe: dense layer {} missing shared expert "
                         "metadata buffers on gpu {}", mp.layer_idx, gpu);
            return false;
        }
        const int32_t d_offsets[2] = {0, num_tokens};
        gpu_dev->memcpy_h2d_async(scratch.shared_expert_offsets, d_offsets,
                                  2 * sizeof(int32_t), stream);
        const int32_t d_gu_ps[3] = {num_tokens, 2 * I_dense_local, hidden};
        gpu_dev->memcpy_h2d_async(scratch.shared_problem_sizes, d_gu_ps,
                                  3 * sizeof(int32_t), stream);
        const int32_t d_gu_sf[2] = {0, ((num_tokens + 127) / 128) * 128};
        gpu_dev->memcpy_h2d_async(scratch.shared_sf_offsets, d_gu_sf,
                                  2 * sizeof(int32_t), stream);

        // Upload NVFP4 alpha (ws2 * input_scale, precomputed at engine load)
        // and the activation input_scale for the quantizer (FP4-ACT-SCALE).
        if (!use_fp8 && scratch.nvfp4_alpha) {
            gpu_dev->memcpy_h2d_async(scratch.nvfp4_alpha,
                                       &dw->alpha, sizeof(float), stream);
            gpu_dev->memcpy_h2d_async(scratch.moe_input_scales,
                                       &dw->input_scale, sizeof(float), stream);
        }

        // V4-7b (ticket H): raw-BF16 dense FFN inside a GGUF checkpoint has
        // no route here (no shipped model needs it — V4 has zero dense
        // layers). Fail loud rather than decode BF16 bytes as k-quants.
        if (use_gguf && !dw->gate_is_gguf) {
            spdlog::critical("dispatch_moe: dense FFN layer {} carries raw "
                             "(non-k-quant) weights in a GGUF checkpoint — "
                             "no BF16 dense route is wired", mp.layer_idx);
            std::abort();
        }

        // Quantize norm_input [B, H] BF16 → FP8/FP4 for dense gate+up.
        // GG-5b: GGUF feeds BF16 norm_input directly (no pre-quant).
        if (scratch.quant_act && !use_gguf) {
            launch_activation_quant({use_fp8, num_tokens, hidden,
                norm_input, scratch.quant_act, scratch.quant_scale,
                scratch.quant_scale_bytes, scratch.shared_expert_offsets,
                scratch.shared_sf_offsets, num_tokens, 1, gpu_dev, stream,
                use_fp8 ? nullptr : scratch.moe_input_scales});
        }

        // Dense gate+up GEMM. GGUF (GG-5c): the gate-block-then-up-block weight is
        // uploaded contiguously (upload_ffn assert_contiguous). When this dense
        // FFN's gate and up tensors share a k-quant type it is byte-identical to
        // one [2*I, H] GGUF block → fused single GEMM; when they differ we split
        // into two single-type GEMMs (the helper reads dw's OWN per-projection
        // types, NOT the routed types). FP8/NVFP4: quant_act × W_gate_up.
        if (use_gguf) {
            GgufDenseGateUpArgs gu{};
            gu.num_tokens = num_tokens;
            gu.intermediate_local = I_dense_local;
            gu.hidden = hidden;
            gu.a_base = norm_input;
            gu.gate_up_output = scratch.gate_up_output;
            gu.gate_scratch = scratch.activation_output;   // [B, I_dense]
            gu.up_scratch = scratch.gguf_gate_up_split;     // [B, I_dense]
            gu.gate_type = to_gguf_compute(dw->gate_gguf_type);
            gu.up_type   = to_gguf_compute(dw->up_gguf_type);
            gu.up_block_offset = model::gguf::gguf_packed_bytes(
                I_dense_local, hidden, dw->gate_gguf_type);
            gu.strategy = gguf_strategy;
            gu.gate_up_weight = dw->gate_up;
            gu.single_b_ptr = scratch.gguf_single_b_ptr;
            gu.expert_offsets =
                static_cast<const int32_t*>(scratch.shared_expert_offsets);
            gu.dev = dev;
            gu.gpu_dev = gpu_dev;
            gu.gemm_workspace = scratch.gemm_workspace;
            gu.gemm_workspace_bytes = scratch.gemm_workspace_bytes;
            gu.stream = stream;
            launch_gguf_dense_gate_up(gu);
        } else {
            GroupedGemmArgs gargs{use_fp8, 1, 2 * I_dense_local, hidden,
                scratch.quant_act,
                dw->gate_up, scratch.gate_up_output,
                scratch.quant_scale, dw->gate_up_scales,
                static_cast<const float*>(scratch.nvfp4_alpha),
                static_cast<const int32_t*>(scratch.shared_expert_offsets),
                static_cast<const int32_t*>(scratch.shared_sf_offsets),
                static_cast<const int32_t*>(scratch.shared_problem_sizes),
                dev, scratch.gemm_workspace, scratch.gemm_workspace_bytes, stream};
            launch_grouped_gemm(gargs);
        }

        // Dense SwiGLU: gate_up_output → activation_output
        launch_swiglu(dev, scratch.activation_output, scratch.gate_up_output,
                      num_tokens, I_dense_local, stream,
                      static_cast<float>(mc.swiglu_limit));

        // Re-populate problem_sizes for down GEMM dimensions.
        const int32_t d_dn_ps[3] = {num_tokens, hidden, I_dense_local};
        gpu_dev->memcpy_h2d_async(scratch.shared_problem_sizes, d_dn_ps,
                                  3 * sizeof(int32_t), stream);

        // Upload down_proj alpha + input_scale (may differ from gate+up).
        if (!use_fp8 && scratch.nvfp4_alpha) {
            gpu_dev->memcpy_h2d_async(scratch.nvfp4_alpha,
                                       &dw->alpha_down, sizeof(float), stream);
            gpu_dev->memcpy_h2d_async(scratch.moe_input_scales,
                                       &dw->input_scale_down, sizeof(float), stream);
        }

        // Quantize activation_output [B, I_dense_local] BF16 → FP8/FP4 for down.
        // GG-5b: GGUF feeds BF16 activation_output directly (no pre-quant).
        if (scratch.quant_act && !use_gguf) {
            launch_activation_quant({use_fp8, num_tokens, I_dense_local,
                scratch.activation_output, scratch.quant_act, scratch.quant_scale,
                scratch.quant_scale_bytes, scratch.shared_expert_offsets,
                scratch.shared_sf_offsets, num_tokens, 1, gpu_dev, stream,
                use_fp8 ? nullptr : scratch.moe_input_scales});
        }
        // GG-5b: rebind the 1-element B_ptrs array to the dense down weight.
        if (use_gguf) {
            gpu_dev->memcpy_h2d_async(scratch.gguf_single_b_ptr, &dw->down,
                                      sizeof(void*), stream);
        }

        // Dense down GEMM. GGUF: BF16 activation_output → expert_output.
        // FP8/NVFP4: quant_act × W_down → expert_output [B, H].
        {
            GroupedGemmArgs gargs{use_fp8, 1, hidden, I_dense_local,
                use_gguf ? scratch.activation_output : scratch.quant_act,
                dw->down, scratch.expert_output,
                scratch.quant_scale, dw->down_scales,
                static_cast<const float*>(scratch.nvfp4_alpha),
                static_cast<const int32_t*>(scratch.shared_expert_offsets),
                static_cast<const int32_t*>(scratch.shared_sf_offsets),
                static_cast<const int32_t*>(scratch.shared_problem_sizes),
                dev, scratch.gemm_workspace, scratch.gemm_workspace_bytes, stream};
            if (use_gguf) {
                gargs.use_gguf = true;
                gargs.gguf_type = to_gguf_compute(dw->down_gguf_type);  // GG-5c: dense's OWN down type
                gargs.gguf_strategy = gguf_strategy;
                gargs.gguf_total_tokens = num_tokens;  // dense: 1 expert, B rows
                gargs.B_ptrs = static_cast<const void**>(scratch.gguf_single_b_ptr);
            }
            launch_grouped_gemm(gargs);
        }

        // KD-4g: stop here — caller will allreduce expert_output across TP ranks.
        if (mp.phase == MoeDispatchPhase::kPreAllreduce)
            return true;
        } // end kPreAllreduce block

        // KD-4g: post-allreduce phase (and kFull) — residual add + commit.
        // expert_output now holds the allreduced sum (or full result for TP=1).
        // V4-5b mHC: the FFN residual update is hc_post (doubly-stochastic
        // stream mix), not an add — ArchDeepseekV4Moe::residual_update; base
        // arch: plain residual add.
        arch.residual_update(gpu, hidden_input, scratch.expert_output,
                             num_tokens, hidden, pair_idx, stream);

        // KD-R2: commit to attention buffer for next layer.
        if (pair_idx >= 0) {
            deps_.hidden_state_pairs[pair_idx].commit(
                static_cast<size_t>(num_tokens) * hidden * deps_.hc_streams * 2,
                stream,
                deps_.device_backends[gpu]);
            // TD-74c: Record moe_attn_event so next layer's kAttention can wait,
            // matching the MoE path (TD-73l).
            if (deps_.hidden_state_pairs[pair_idx].moe_attn_event) {
                deps_.stream_manager->record_event(
                    deps_.hidden_state_pairs[pair_idx].moe_attn_event,
                    static_cast<int>(gpu), compute::StreamId::kExpertFfn);
            }
        }
        return true;
    }

    // Declared before goto to avoid crossing initialization (C++ rule).
    int total_resident = 0;
    bool skip_routed = false;
    // TD-52w: tracks whether moe_output contains valid data (written by Step 6
    // or Step 7d). Step 8 residual add is gated on this to prevent adding
    // uninitialized/stale device memory to the hidden state.
    bool moe_valid = false;

    // KD-4g: for kPostAllreduce, skip directly to post-allreduce work.
    if (mp.phase == MoeDispatchPhase::kPostAllreduce)
        goto moe_post_allreduce;

    // TD-89m: Build per-expert resident bitset. Replaces old all-or-nothing check.
    // When bitset_precomputed is true (TP path), the bitset was pre-populated by
    // dispatch_moe_all_ranks with the cross-GPU intersection (TD-89p).
    {
    auto& bitset = moe_scratch_[gpu].expert_resident_bitset;
    if (!mp.bitset_precomputed) {
        std::memset(bitset.data(), 0, bitset.size());
        if (deps_.expert_cache && deps_.cuda_kernels_enabled) {
            for (int e = 0; e < n_experts; ++e) {
                memory::ExpertKey key{static_cast<uint32_t>(mp.layer_idx),
                                      static_cast<uint16_t>(e)};
                const auto* entry = deps_.expert_cache->lookup(key, static_cast<int>(gpu));
                if (entry && (entry->sub_components_ready & memory::SubComponent::kAll)
                              == memory::SubComponent::kAll) {
                    bitset[e / 8] |= static_cast<uint8_t>(1u << (e % 8));
                    ++total_resident;
                }
            }
        }
    } else {
        // Count set bits in precomputed bitset.
        for (int e = 0; e < n_experts; ++e) {
            if ((bitset[e / 8] >> (e % 8)) & 1)
                ++total_resident;
        }
    }
    }

    // If zero experts are resident, zero moe_output and skip routed pipeline.
    // Also skip when CUDA kernels are disabled (test/null backend path).
    // TD-PREFILL-FETCH-SEAM-SCALING: a kFinal wave pass must NOT take this
    // shortcut even with zero residents THIS pass — earlier waves already
    // accumulated this rank's expert rows into moe_wave_accum, and the final
    // pass must still unpermute them (an empty bitset just adds zero rows).
    if (total_resident == 0 && mp.wave_pass != MoeWavePass::kFinal) {
        if (deps_.cuda_kernels_enabled && scratch.moe_output) {
            deps_.device_backends[gpu]->memset_async(
                scratch.moe_output, 0,
                static_cast<size_t>(num_tokens) * hidden * 2, stream);
        }
        // TD-MOE-EP-COMBINE-RESIDUAL (full-pool): when the canonical EP-within-TP
        // combine is active, the cross-GPU gather SUMs each rank's PER-SLOT buffer
        // (mp.ep_combine_mode 1=fp32 / 2=bf16), NOT moe_output. The skip below does
        // NOT run moe_unpermute, so a zero-resident rank would otherwise contribute
        // its STALE per-slot rows (from a prior layer/token) — a real corruption of
        // the routed output (large near-tie fork), not benign FP. A rank reaches
        // zero residents only under non-e%tp placement (the ACT reroute can empty a
        // GPU for a layer); e%tp keeps each GPU non-empty so baseline never hit it.
        // Zero the per-slot buffer this rank will gather so its contribution = 0.
        // Gated default-ON; LS_NO_PERSLOT_ZEROFIX=1 reproduces the pre-fix bug for
        // the load-bearing A/B (control = fork; fix = collapse-to-baseline).
        static const bool perslot_zerofix = [] {
            const char* v = std::getenv("LS_NO_PERSLOT_ZEROFIX");
            return !(v && v[0] && v[0] != '0');
        }();
        if (perslot_zerofix && deps_.cuda_kernels_enabled) {
            const size_t perslot_elems =
                static_cast<size_t>(num_tokens) * topk * hidden;
            if (mp.ep_combine_mode == 2 && scratch.moe_output_bf16_perslot) {
                deps_.device_backends[gpu]->memset_async(
                    scratch.moe_output_bf16_perslot, 0, perslot_elems * 2, stream);
            } else if (mp.ep_combine_mode == 1 && scratch.moe_output_fp32) {
                deps_.device_backends[gpu]->memset_async(
                    scratch.moe_output_fp32, 0, perslot_elems * 4, stream);
            }
        }
        last_moe_miss_count_ = static_cast<uint8_t>(std::min(topk, 255));
        skip_routed = true;
        // F-7: gate skipped (no residents / null backend) — still publish the seam
        // routing region+bytes so the seam telemetry contract holds.
        publish_seam_routing(mp, gpu, expanded_tokens, stream);
    }

    if (!skip_routed) {

    // F-2: precomputed-gating consume path. When set, topk_weights/topk_indices
    // are already present in moe_scratch_[gpu] (written by an earlier gating pass —
    // e.g. RUN_ATTENTION emit_gating in F-1). Skip Step 0 (router projection) and
    // Step 1 (TopK gating) entirely and treat the routing as valid, so the rest of
    // the pipeline (miss-count, permute, GEMMs, unpermute, commit) is byte-for-byte
    // identical to the self-gating path. Default (flag clear) self-gates below.
    bool router_valid = false;
    if (mp.use_precomputed_gating) {
        // For real MoE layers, valid top-K must already be in scratch; for
        // MTP/out-of-range layers there is no routing and we fall through to the
        // same silent-skip behaviour as the self-gating path.
        const int num_layers = mc.num_hidden_layers;
        const bool is_real_moe_layer = static_cast<int>(mp.layer_idx) >= first_k_dense
                                    && static_cast<int>(mp.layer_idx) < num_layers;
        router_valid = is_real_moe_layer && deps_.cuda_kernels_enabled;
    } else {

    // Step 0: Router projection — hidden → router_logits (TD-50a)
    // TD-52aa: track whether Step 0 produced valid router_logits. When false,
    // Steps 1-6 are skipped to prevent gating from reading stale/uninitialized data.
    if (deps_.cuda_kernels_enabled && norm_input) {
        const void* router_w = nullptr;
        if (mp.layer_idx < deps_.router_weight_ptrs.size() &&
            gpu < deps_.router_weight_ptrs[mp.layer_idx].size()) {
            router_w = deps_.router_weight_ptrs[mp.layer_idx][gpu];
        }
        if (router_w) {
            compute::launch_router_projection(
                static_cast<float*>(scratch.router_logits),
                norm_input, router_w,
                num_tokens, n_experts, hidden, stream);
            router_valid = true;
        }
    }

    // TD-53u/TD-52aa: if router projection failed on a real MoE layer
    // (layer_idx >= first_k_dense and < num_hidden_layers), this indicates
    // a configuration or weight upload bug — report error to the orchestrator.
    // For MTP/out-of-range layers, silently skip (expected: no router weights).
    if (!router_valid) {
        const int num_layers = mc.num_hidden_layers;
        const bool is_real_moe_layer = static_cast<int>(mp.layer_idx) >= first_k_dense
                                    && static_cast<int>(mp.layer_idx) < num_layers;
        if (is_real_moe_layer) {
            spdlog::error("dispatch_moe: router projection failed for MoE layer {} "
                          "on gpu {} — missing router weights or norm input",
                          mp.layer_idx, gpu);
            return false;
        }
        // MTP/out-of-range layer: no router weights expected. Skip Steps 1-6.
    }

    // Step 1: expert selection — router_logits → topk_weights, topk_indices.
    // Base arch (arch_mla_moe.cpp): learned top-K gating (V4-4a scoring_func
    // parametrizes sigmoid vs sqrtsoftplus). V4 override: V4-4 hash-layer
    // gating (tid2eid) for layer_idx < num_hash_layers, base body otherwise.
    if (!arch.select_experts(mp, gpu, num_tokens, topk, n_experts,
                             router_valid, stream))
        return false;

    } // F-2: end self-gating branch (else of use_precomputed_gating)
    // F-7: Seam-routing checkpoint — between Step 1 (gating) and the expert GEMM,
    // publish the routed top-K (weights f32 + indices i32) to the sideband so the
    // host/daemon fetch decider sits at the attention↔MoE seam. This is the point
    // where routing becomes known; the H2D fetch it drives is data-dependent, so
    // no single CUDA graph can span attention→fetch→expert (INV-F-7-1 / Decision in
    // F_FORWARD_LAYER_FUSION.md). Called here (gate ran → real top-K) and also in
    // the skip_routed path above (gate skipped → publish the region/bytes so the
    // seam contract still holds). Mirrors the MTP store_gating export
    // (dispatch_forward.cpp:543): host memcpy when CUDA is off, D2H otherwise.
    publish_seam_routing(mp, gpu, expanded_tokens, stream);

    // TD-52aa: Steps 1-6 and miss counting are only valid when router_valid.
    // When router_valid is false (MTP/out-of-range layer with no router weights),
    // moe_output stays at whatever the skip_routed path set (zeroed if total_resident==0).
    if (router_valid) {

    // TD-89m: count routed expert misses via D2H of topk_indices.
    // Skipped when all experts are resident (zero overhead on happy path).
    // INV-MOE-OVERLAP: also skipped for wave-PARTIAL passes — the D2H event
    // spin would stall the daemon at overlap-issue time (it should be polling
    // fetch arrivals), and last_moe_miss_count_ is overwritten by the finalize
    // pass anyway (FETCH completions report miss counts from ProgressiveMoeState,
    // not from this counter).
    {
    auto& bitset = moe_scratch_[gpu].expert_resident_bitset;
    if (deps_.cuda_kernels_enabled && total_resident < n_experts
        && mp.wave_pass != MoeWavePass::kPartial && !mp.skip_miss_probe) {
        auto& indices_host = moe_scratch_[gpu].topk_indices_host;
        const size_t copy_bytes = static_cast<size_t>(num_tokens) * topk * sizeof(int32_t);
        auto* dev_be = deps_.device_backends[gpu];
        dev_be->memcpy_d2h_async(indices_host.data(), scratch.topk_indices,
                                 copy_bytes, stream);
        // Event-based sync: spin-poll until D2H completes (~1us for 32-byte decode).
        void* sync_event = deps_.stream_manager->create_event(static_cast<int>(gpu));
        deps_.stream_manager->record_event(sync_event, static_cast<int>(gpu),
                                            compute::StreamId::kExpertFfn);
        // TD-91b: bounded spin-poll — 10k iterations ≈ 10ms at ~1μs/cudaEventQuery.
        // Covers any legitimate GPU backlog; prevents daemon hang on GPU stall.
        bool d2h_ok = false;
        constexpr int kMaxD2hPollIters = 10000;
        for (int poll_i = 0; poll_i < kMaxD2hPollIters; ++poll_i) {
            auto [status, err] = deps_.stream_manager->query_event(
                sync_event, static_cast<int>(gpu));
            if (status == compute::EventStatus::kReady) { d2h_ok = true; break; }
            if (status == compute::EventStatus::kError) {
                // TD-89aa: don't read potentially invalid host data after D2H error.
                spdlog::error("dispatch_moe: D2H sync event error on gpu {}", gpu);
                break;
            }
        }
        if (!d2h_ok) {
            spdlog::warn("dispatch_moe: D2H poll exhausted after {} iters on gpu {} "
                         "(layer {}), reporting 0 misses", kMaxD2hPollIters, gpu, mp.layer_idx);
        }
        deps_.stream_manager->destroy_event(sync_event, static_cast<int>(gpu));

        // Count unique missing experts among top-K selections.
        // TD-89aa: skip counting when D2H failed — host buffer may contain garbage.
        if (d2h_ok) {
        uint8_t seen_missing[32] = {};
        uint8_t miss_count = 0;
        for (int i = 0; i < num_tokens * topk; ++i) {
            const int e = indices_host[i];
            if (e >= 0 && e < n_experts) {
                const bool is_resident = (bitset[e / 8] >> (e % 8)) & 1;
                const bool already_seen = (seen_missing[e / 8] >> (e % 8)) & 1;
                if (!is_resident && !already_seen) {
                    seen_missing[e / 8] |= static_cast<uint8_t>(1u << (e % 8));
                    if (miss_count < 255) ++miss_count;
                }
            }
        }
        last_moe_miss_count_ = miss_count;
        } // d2h_ok
    } else {
        last_moe_miss_count_ = 0;
    }

    // TD-89o: moe_mode behavioral variants.
    // Mode 0: partial execution + report miss count (default).
    // Mode 1: same as mode 0 but suppress miss count in completion.
    // Mode 2: same as mode 0 for now (zero-fill missing experts).
    //         Shared expert already adds to moe_output, partially compensating.
    if (mp.moe_mode == 1) {
        last_moe_miss_count_ = 0;
    }

    // F-4: routing-export contract. When store_gating is requested, publish the
    // top-K this MoE is about to consume into the documented sideband slot
    // (IpcLayout::kRoutingExportOff), signalled by the op's CMP_COMPUTE_DONE.
    // This is the zero-copy routing seam read back by the orchestrator/tests; it
    // does NOT ride in the Completion struct. F-1/F-3 will relocate the producer
    // to RUN_ATTENTION [emit_gating] but keep this exact slot + layout.
    if (mp.store_gating)
        publish_routing_export(gpu, num_tokens, topk, mp.layer_idx, stream);
    }

    auto* gpu_dev = deps_.device_backends[gpu];
    auto& b_host  = moe_scratch_[gpu].routed_b_ptrs_host;
    auto& sb_host = moe_scratch_[gpu].routed_sb_ptrs_host;
    auto& bitset = moe_scratch_[gpu].expert_resident_bitset;

    // GG-S1: one emitter owns the routed grouped-GEMM quant pick (fp8/nvfp4/gguf
    // param-build) and derives the GGUF permuted-row count ONCE (= expanded_
    // tokens) — every routed gate/up/down GEMM, graph or eager, goes through it
    // so the per-site `gguf_total_tokens` is never forgotten
    // (TD-GG5D-GGUF-TOTAL-TOKENS-SILENT-ZERO). Flat POD, no heap/virtual.
    const MoeGemmEmitter routed_gemm_emit{
        dev, scratch.gemm_workspace, scratch.gemm_workspace_bytes, stream,
        n_experts, use_fp8, use_gguf, gguf_strategy,
        /*total_tokens=*/expanded_tokens,
        static_cast<const int32_t*>(scratch.expert_offsets),
        static_cast<const int32_t*>(scratch.sf_offsets),
        static_cast<const int32_t*>(scratch.problem_sizes)};

    // TD-DECODE-FFN-GRAPH (experiment): is this dispatch eligible for the routed
    // FFN decode CUDA graph? Decode batch=1 only (num_tokens==1 → fixed shape:
    // each active expert sees M_e=1 token, expert_offsets/problem_sizes/sf_offsets
    // are recomputed device-side by permute+populate_meta from the per-step
    // topk_indices fed at fixed device addresses). NVFP4 only (the experiment
    // target). The ONLY per-step host→device feed inside the Step 2..6 boundary
    // is the per-projection b_ptrs/sb_ptrs — routed through the fixed-address
    // pinned indirection buffers (g_b_ptrs[]/g_b_ptrs_host[]). Everything else is
    // device-resident and recomputed by captured kernels, so ONE graph replays
    // correctly across all MoE layers and decode steps (layer/step-specific
    // expert weights flow entirely through the b_ptrs indirection).
    // GG-5d: GGUF is now graph-eligible. The device-fused int kernel reads the
    // expert offsets on-device (no host D2H — TD-GG5-GROUPED-HOST-SYNC resolved),
    // and at decode (M_e=1) the host-side mmvq-vs-mmq pick (avg_m=total_tokens/E≪8)
    // always selects mmvq, which issues only stream kernels — NO cudaFuncSetAttribute
    // (that fires lazily inside the mmq instances, a prefill-only path; see the GGUF
    // branch of emit_routed_ffn + TD-GG5-MMQ-CAPTURE-WARMUP). So the captured GGUF FFN
    // sequence is fully capture-safe without a warmup. quant_act/nvfp4_alpha are
    // allocated for every MoE model (incl. GGUF), so they don't gate GGUF out.
    // TD-DECODE-FFN-GRAPH-GGUF-CAPTURE-BROKEN fix (a): the GGUF dequant
    // strategy is a HOST-LOOP (launch_gguf_grouped_gemm_hostloop syncs the
    // stream to read expert offsets/B ptrs) — fundamentally not capture-safe,
    // so it is excluded here instead of aborting mid-capture.
    // INV-MOE-OVERLAP: wave passes are now graph-eligible at decode (B==1) —
    // the resident-overlap split runs kPartial (Steps 2..5 + accumulate) and
    // kFinal (… + unpermute-from-accumulator); both are fixed-shape at B==1
    // and capture-safe (memset_async/residual_add are stream ops). The pass
    // kind + the per-GPU wave-first bit are part of the variant key below so
    // each captured graph has fixed control flow. Prefill waves never reach
    // here (num_tokens > 1). kPartial uses the SEPARATE *_w b_ptrs staging.
    const bool ffn_graph_eligible =
        moe_ffn_graph_enabled() && deps_.cuda_kernels_enabled && !use_fp8 &&
        num_tokens == 1 && scratch.quant_act && scratch.nvfp4_alpha &&
        scratch.g_b_ptrs[0] && scratch.g_b_ptrs_host[0] &&
        (mp.wave_pass == MoeWavePass::kNone ||
         (scratch.g_b_ptrs_w[0] && scratch.g_b_ptrs_host_w[0] &&
          scratch.moe_wave_accum)) &&
        gpu < routed_ffn_graphs_.size() &&
        (!use_gguf ||
         gguf_strategy == compute::GgufGemmStrategy::int_strategy);

    // TD-81a/89m: build B_ptrs + scale_B_ptrs for one projection.
    // Missing experts (bitset=0) get zero buffer → GEMM reads zeros, gather_alphas
    // reads alpha=0.0f → zero contribution. Non-graph path writes the shared
    // host vector + H2D into the shared routed_b_ptrs device array (as before).
    //
    // INV-MOE-OVERLAP null-skip: for GGUF DECODE wave passes the excluded
    // experts get a NULL pointer instead of the zero buffer — the grouped GEMV
    // early-returns their CTAs (deps kernels honor NULL) instead of paying the
    // full latency-bound K-walk over zeros. The pass's GEMM output rows are
    // pre-zeroed in emit_routed_ffn so the skipped rows stay exact zeros
    // (bit-identical to the zero-weight-buffer result). kNone keeps the zero
    // buffer (byte-identical legacy path).
    const bool wave_null_skip = use_gguf && num_tokens == 1
        && mp.wave_pass != MoeWavePass::kNone;
    void* const excluded_b = wave_null_skip
        ? nullptr : static_cast<void*>(scratch.zero_weight_buf);
    auto build_routed_b_ptrs = [&](auto proj_offset_fn, int64_t weight_bytes) {
        for (int e = 0; e < n_experts; ++e) {
            const bool is_resident = (bitset[e / 8] >> (e % 8)) & 1;
            if (is_resident) {
                memory::ExpertKey key{static_cast<uint32_t>(mp.layer_idx),
                                      static_cast<uint16_t>(e)};
                const auto* entry = deps_.expert_cache->lookup(key, static_cast<int>(gpu));
                // TD-91d: guard against concurrent eviction between bitset snapshot
                // and lookup — fall back to zero buffer (same as non-resident path).
                if (!entry || !entry->vram_address) {
                    b_host[e]  = excluded_b;
                    sb_host[e] = excluded_b;
                    continue;
                }
                auto* base = static_cast<uint8_t*>(entry->vram_address);
                int64_t off = proj_offset_fn(entry);
                b_host[e]  = base + off;                  // FP4/FP8 weight
                sb_host[e] = base + off + weight_bytes;   // group scales
            } else {
                b_host[e]  = excluded_b;
                sb_host[e] = excluded_b;
            }
        }
        gpu_dev->memcpy_h2d_async(scratch.routed_b_ptrs, b_host.data(),
                                  n_experts * sizeof(void*), stream);
        gpu_dev->memcpy_h2d_async(scratch.routed_sb_ptrs, sb_host.data(),
                                  n_experts * sizeof(void*), stream);
    };

    // Graph path: fill the PINNED per-projection host staging (HOST work, runs
    // OUTSIDE capture each step) — the captured H2D below copies these into the
    // fixed device arrays at replay time, so fresh routing/weights flow in.
    // INV-MOE-OVERLAP: kPartial (the resident-overlap pass) uses the *_w
    // staging set — its graph H2D nodes read the pinned buffers asynchronously
    // while the host refills the kFinal set for the same layer.
    const bool use_wave_bptr_set = (mp.wave_pass == MoeWavePass::kPartial);
    auto fill_graph_b_ptrs = [&](int proj, auto proj_offset_fn, int64_t weight_bytes) {
        const void** bh  = use_wave_bptr_set ? scratch.g_b_ptrs_host_w[proj]
                                             : scratch.g_b_ptrs_host[proj];
        const void** sbh = use_wave_bptr_set ? scratch.g_sb_ptrs_host_w[proj]
                                             : scratch.g_sb_ptrs_host[proj];
        for (int e = 0; e < n_experts; ++e) {
            const bool is_resident = (bitset[e / 8] >> (e % 8)) & 1;
            if (is_resident) {
                memory::ExpertKey key{static_cast<uint32_t>(mp.layer_idx),
                                      static_cast<uint16_t>(e)};
                const auto* entry = deps_.expert_cache->lookup(key, static_cast<int>(gpu));
                if (!entry || !entry->vram_address) {
                    bh[e] = excluded_b; sbh[e] = excluded_b;
                    continue;
                }
                auto* base = static_cast<uint8_t*>(entry->vram_address);
                int64_t off = proj_offset_fn(entry);
                bh[e]  = base + off;
                sbh[e] = base + off + weight_bytes;
            } else {
                bh[e] = excluded_b; sbh[e] = excluded_b;
            }
        }
    };

    // ── GG-S1: the ONE routed-FFN op sequence (Step 2..6) ───────────────────
    // Emitted IDENTICALLY for the eager and the CUDA-graph-captured paths — this
    // is the single source that kills the former run_routed_ffn-vs-eager-else
    // duplication (TD-GG5D-GRAPH-SEQUENCE-DUP): graph≡eager is now structural
    // (same emit), not re-proven by hand. The ONLY per-mode difference is how the
    // per-projection expert B-pointers are staged:
    //   · kEager — build_routed_b_ptrs() fills the SHARED routed_b_ptrs device
    //              array (host vector → live H2D) right before each projection;
    //              gate/up/down reuse that one array, refilled in place.
    //   · kGraph — the three pinned per-projection host buffers are refilled
    //              OUTSIDE capture (fill_graph_b_ptrs, in the caller); the H2D
    //              into the DISTINCT g_b_ptrs[proj] device arrays is captured
    //              here so replay pulls fresh routing/weights. Pinned host keeps
    //              the copy capture-safe (pageable would force an illegal sync).
    // Everything else (permute, populate_meta, gather, quant, GEMMs, SwiGLU,
    // unpermute) is byte-identical between modes; the cuda-gated steps keep their
    // guards (always true under kGraph). emit() stays free of host-side
    // data-dependent branching that differs capture-vs-replay (capture-safety),
    // and the per-step B-ptr refill stays OUTSIDE capture exactly as before.
    enum class MoeBPtrMode { kEager, kGraph };
    // INV-MOE-OVERLAP: wave-accumulation control, hoisted OUT of the emit so
    // the captured graphs have fixed control flow (the per-GPU
    // moe_wave_accum_used_ host branch must not differ capture-vs-replay).
    // wave_first ⇒ this pass memsets the accumulator before adding; it is part
    // of the graph variant key. The used-flag is set after the graph-or-eager
    // block below (exactly once per dispatch, replays included).
    const bool wave_accum_active = mp.wave_pass != MoeWavePass::kNone
        && scratch.moe_wave_accum
        && gpu < moe_wave_accum_used_.size();
    const bool wave_first = wave_accum_active && !moe_wave_accum_used_[gpu];
    auto emit_routed_ffn = [&](MoeBPtrMode mode) {
        const bool graph = (mode == MoeBPtrMode::kGraph);
        // Per-projection expert B/scale-B device arrays the GEMMs+gather read.
        auto bp  = [&](int p) -> void* {
            return graph ? (use_wave_bptr_set ? scratch.g_b_ptrs_w[p]
                                              : scratch.g_b_ptrs[p])
                         : scratch.routed_b_ptrs;  };
        auto sbp = [&](int p) -> void* {
            return graph ? (use_wave_bptr_set ? scratch.g_sb_ptrs_w[p]
                                              : scratch.g_sb_ptrs[p])
                         : scratch.routed_sb_ptrs; };
        // Stage projection p's expert B-pointers (and, non-GGUF, scale-B ptrs).
        auto prep_bptrs = [&](int p) {
            if (!graph) {
                // Eager: fill the shared array + live H2D (also primes scale-B).
                if (p == 0)      build_routed_b_ptrs(gate_off_fn,
                                     deps_.expert_cache->gate_weight_bytes());
                else if (p == 1) build_routed_b_ptrs(up_off_fn,
                                     deps_.expert_cache->up_weight_bytes());
                else             build_routed_b_ptrs(down_off_fn,
                                     deps_.expert_cache->down_weight_bytes());
                return;
            }
            // Graph: captured H2D from pinned per-projection host staging into
            // the distinct device arrays (GGUF has no scale-B trailer → b only).
            // INV-MOE-OVERLAP: kPartial uses the *_w set (see fill_graph_b_ptrs).
            gpu_dev->memcpy_h2d_async(
                bp(p),
                use_wave_bptr_set ? scratch.g_b_ptrs_host_w[p]
                                  : scratch.g_b_ptrs_host[p],
                n_experts * sizeof(void*), stream);
            if (!use_gguf)
                gpu_dev->memcpy_h2d_async(
                    sbp(p),
                    use_wave_bptr_set ? scratch.g_sb_ptrs_host_w[p]
                                      : scratch.g_sb_ptrs_host[p],
                    n_experts * sizeof(void*), stream);
        };

        // INV-MOE-OVERLAP null-skip: NULL-skipped experts' CTAs write NOTHING
        // (vs the zero-weight buffer which wrote zeros), so this pass's GEMM
        // output rows are pre-zeroed — the skipped rows then flow as exact
        // zeros through interleave → SwiGLU(0,0)=0 → down(skip) → accumulate
        // (+0), bit-identical to the zero-buffer result. Capture-safe.
        if (wave_null_skip) {
            gpu_dev->memset_async(scratch.activation_output, 0,
                static_cast<size_t>(expanded_tokens) * intermediate * 2, stream);
            gpu_dev->memset_async(scratch.expert_output, 0,
                static_cast<size_t>(expanded_tokens) * hidden * 2, stream);
        }

        // §12h sub-seam M1: entry+gating done, routed emit begins.
        perf_trace::record(perf_trace::kMoeSegRankPre,
                           static_cast<uint16_t>(mp.gpu_idx), 0,
                           mp.layer_idx << 16, 10);
        // Step 2: Permute — reorder hidden states by expert assignment.
        dev->moe_permute(
            scratch.permuted_input,
            static_cast<int32_t*>(scratch.expert_offsets),
            static_cast<int32_t*>(scratch.src_to_dest_map),
            static_cast<int32_t*>(scratch.permuted_idx),
            norm_input,
            static_cast<const int32_t*>(scratch.topk_indices),
            num_tokens, topk, hidden, n_experts,
            /*elem_size_bytes=*/2,
            scratch.permute_workspace, stream);

        // -- Gate B_ptrs --
        prep_bptrs(0);
        // KD-3f: populate problem_sizes with N=intermediate (not 2*intermediate).
        // GGUF ignores problem_sizes (int kernel reads expert_offsets on-device);
        // kept here so the op list is identical to the NVFP4 path.
        if (deps_.cuda_kernels_enabled) {
            compute::launch_populate_gemm_meta(
                static_cast<int32_t*>(scratch.problem_sizes),
                use_fp8 ? nullptr : static_cast<int32_t*>(scratch.sf_offsets),
                static_cast<const int32_t*>(scratch.expert_offsets),
                intermediate, hidden, n_experts, stream);
        }
        // TD-81d/69w + FP4-ACT-SCALE: gather gate alphas + input_scales. GGUF has
        // no per-expert scale trailer (ignores nvfp4_alpha); FP8 needs none.
        if (!use_fp8 && !use_gguf && scratch.nvfp4_alpha) {
            compute::launch_gather_alphas_scaled(
                static_cast<float*>(scratch.nvfp4_alpha),
                static_cast<float*>(scratch.moe_input_scales),
                static_cast<const void* const*>(bp(0)),
                deps_.expert_cache->gate_bytes() - 8,
                deps_.expert_cache->gate_bytes() - 4,
                n_experts, stream);
        }
        // KD-3g: quantize permuted_input [expanded_tokens, H] BF16 → FP8/FP4.
        // GGUF feeds BF16 permuted_input directly (int kernel self-quantizes).
        if (deps_.cuda_kernels_enabled && scratch.quant_act && !use_gguf) {
            launch_activation_quant({use_fp8, expanded_tokens, hidden,
                scratch.permuted_input, scratch.quant_act, scratch.quant_scale,
                scratch.quant_scale_bytes, scratch.expert_offsets,
                scratch.sf_offsets, expanded_tokens, n_experts,
                gpu_dev, stream,
                use_fp8 ? nullptr : scratch.moe_input_scales});
        }
        // Step 3a: Gate GEMM → activation_output.
        routed_gemm_emit.routed_gemm(intermediate, hidden,
            use_gguf ? scratch.permuted_input : scratch.quant_act,
            scratch.activation_output,
            static_cast<const void**>(bp(0)), static_cast<const void**>(sbp(0)),
            scratch.quant_scale,
            use_fp8 ? nullptr : static_cast<const float*>(scratch.nvfp4_alpha),
            gguf_gate_type);

        // -- Up B_ptrs --
        prep_bptrs(1);
        if (!use_fp8 && !use_gguf && scratch.nvfp4_alpha) {
            compute::launch_gather_alphas_scaled(
                static_cast<float*>(scratch.nvfp4_alpha), nullptr,
                static_cast<const void* const*>(bp(1)),
                deps_.expert_cache->up_bytes() - 8,
                deps_.expert_cache->up_bytes() - 4,
                n_experts, stream);
        }
        // Step 3b: Up GEMM → expert_output.
        routed_gemm_emit.routed_gemm(intermediate, hidden,
            use_gguf ? scratch.permuted_input : scratch.quant_act,
            scratch.expert_output,
            static_cast<const void**>(bp(1)), static_cast<const void**>(sbp(1)),
            scratch.quant_scale,
            use_fp8 ? nullptr : static_cast<const float*>(scratch.nvfp4_alpha),
            gguf_up_type);

        // -- Down B_ptrs + down-GEMM problem_sizes/sf_offsets + down alphas. --
        prep_bptrs(2);
        if (deps_.cuda_kernels_enabled) {
            compute::launch_populate_gemm_meta(
                static_cast<int32_t*>(scratch.problem_sizes),
                use_fp8 ? nullptr : static_cast<int32_t*>(scratch.sf_offsets),
                static_cast<const int32_t*>(scratch.expert_offsets),
                hidden, intermediate, n_experts, stream);
        }
        if (!use_fp8 && !use_gguf && scratch.nvfp4_alpha) {
            compute::launch_gather_alphas_scaled(
                static_cast<float*>(scratch.nvfp4_alpha),
                static_cast<float*>(scratch.moe_input_scales),
                static_cast<const void* const*>(bp(2)),
                deps_.expert_cache->down_bytes() - 8,
                deps_.expert_cache->down_bytes() - 4,
                n_experts, stream);
        }
        // Step 4: SwiGLU (+ optional quant), per quant mode. GGUF: plain BF16
        // SwiGLU (down GGUF GEMM consumes BF16 directly). NVFP4: fused SwiGLU +
        // FP4 quant. FP8: interleave + SwiGLU + FP8 quant.
        if (deps_.cuda_kernels_enabled && (scratch.quant_act || use_gguf)) {
            if (use_gguf) {
                if (expanded_tokens > 0) {
                    const size_t I_bytes = static_cast<size_t>(intermediate) * 2;
                    gpu_dev->memcpy_2d_async(
                        scratch.gate_up_output, I_bytes * 2,
                        scratch.activation_output, I_bytes,
                        I_bytes, expanded_tokens, stream);
                    gpu_dev->memcpy_2d_async(
                        static_cast<uint8_t*>(scratch.gate_up_output) + I_bytes, I_bytes * 2,
                        scratch.expert_output, I_bytes,
                        I_bytes, expanded_tokens, stream);
                }
                launch_swiglu(dev, scratch.activation_output, scratch.gate_up_output,
                              expanded_tokens, intermediate, stream,
                              static_cast<float>(mc.swiglu_limit));
                // INV-MOE-OVERLAP null-skip: re-zero expert_output AFTER the
                // interleave consumed the up outputs — the down GEMM (stride H)
                // and the up GEMM (stride I) share this buffer, so a skipped
                // expert's down-row range [r*H, (r+1)*H) overlaps live up
                // outputs (rows r < expanded*I/H). The zero-weight-buffer path
                // overwrote every down row with zeros; NULL-skip writes
                // nothing, so the skipped rows must be zeroed here or they
                // accumulate stale up data (measured trajectory corruption).
                if (wave_null_skip) {
                    gpu_dev->memset_async(scratch.expert_output, 0,
                        static_cast<size_t>(expanded_tokens) * hidden * 2,
                        stream);
                }
            } else if (!use_fp8) {
                // V4-4b note: the fused SiLU·mul→NVFP4 kernel has NO
                // swiglu_limit support. swiglu_limit>0 + NVFP4 expert
                // weights is rejected at config validation (V4 experts are
                // GGUF MXFP4 — this path is V4-unreachable).
                launch_fused_swiglu_nvfp4_quant({
                    scratch.activation_output, scratch.expert_output,
                    scratch.quant_act, scratch.quant_scale, scratch.quant_scale_bytes,
                    scratch.expert_offsets, scratch.sf_offsets,
                    expanded_tokens, n_experts, intermediate,
                    scratch.moe_input_scales, gpu_dev, stream});
            } else {
                if (expanded_tokens > 0) {
                    const size_t I_bytes = static_cast<size_t>(intermediate) * 2;
                    gpu_dev->memcpy_2d_async(
                        scratch.gate_up_output, I_bytes * 2,
                        scratch.activation_output, I_bytes,
                        I_bytes, expanded_tokens, stream);
                    gpu_dev->memcpy_2d_async(
                        static_cast<uint8_t*>(scratch.gate_up_output) + I_bytes, I_bytes * 2,
                        scratch.expert_output, I_bytes,
                        I_bytes, expanded_tokens, stream);
                }
                launch_swiglu(dev, scratch.activation_output, scratch.gate_up_output,
                              expanded_tokens, intermediate, stream,
                              static_cast<float>(mc.swiglu_limit));
                launch_activation_quant({use_fp8, expanded_tokens, intermediate,
                    scratch.activation_output, scratch.quant_act, scratch.quant_scale,
                    scratch.quant_scale_bytes, scratch.expert_offsets,
                    scratch.sf_offsets, expanded_tokens, n_experts,
                    gpu_dev, stream, nullptr});
            }
        }
        // Step 5: Down grouped GEMM → expert_output.
        routed_gemm_emit.routed_gemm(hidden, intermediate,
            use_gguf ? scratch.activation_output : scratch.quant_act,
            scratch.expert_output,
            static_cast<const void**>(bp(2)), static_cast<const void**>(sbp(2)),
            scratch.quant_scale,
            use_fp8 ? nullptr : static_cast<const float*>(scratch.nvfp4_alpha),
            gguf_down_type);

        // TD-PREFILL-FETCH-SEAM-SCALING: rolling-wave accumulation. Each
        // permuted row belongs to exactly ONE expert, so exactly one wave
        // writes it non-zero (absent experts' rows are zeros from the
        // zero-weight GEMM) — the elementwise add is bit-exact (x + 0 = x).
        // kPartial stops here (no unpermute/shared/residual); kFinal adds its
        // rows then feeds the ACCUMULATOR to the unchanged Step 6+ chain, so
        // the multi-wave result is bit-identical to an all-resident single
        // pass. kNone (in-capacity unions, decode, RUN_MOE) never enters.
        // INV-MOE-OVERLAP: the memset decision (wave_first) is hoisted out of
        // the emit (fixed per graph variant); the used-flag is set by the
        // caller after the graph-or-eager block, once per dispatch.
        const bool wave_accum = wave_accum_active;
        if (wave_accum) {
            if (wave_first) {
                gpu_dev->memset_async(scratch.moe_wave_accum, 0,
                                      static_cast<size_t>(expanded_tokens) * hidden * 2,
                                      stream);
            }
            compute::launch_residual_add(scratch.moe_wave_accum,
                                         scratch.expert_output,
                                         expanded_tokens * hidden, stream);
            if (mp.wave_pass == MoeWavePass::kPartial)
                return;  // wave-partial: accumulated only; caller returns
        }

        // Step 6: Unpermute — expert_output + topk_weights → moe_output.
        // DET-REDUCE Phase 1b: fp32 partial → moe_output_fp32 for the
        // placement-invariant EP combine (else bf16 → moe_output).
        // Rolling-wave kFinal reads the accumulated rows instead.
        dev->moe_unpermute(
            (mp.ep_combine_mode == 2 ? scratch.moe_output_bf16_perslot : mp.ep_combine_mode == 1 ? scratch.moe_output_fp32 : scratch.moe_output),
            wave_accum ? scratch.moe_wave_accum : scratch.expert_output,
            static_cast<const float*>(scratch.topk_weights),
            static_cast<const int32_t*>(scratch.src_to_dest_map),
            num_tokens, topk, hidden,
            /*elem_size_bytes=*/2, stream,
            static_cast<compute::MoeCombineMode>(mp.ep_combine_mode));
    };

    // TD-DECODE-FFN-GRAPH-GGUF-CAPTURE-BROKEN fix (b): the captured kernel
    // instantiations depend on the per-LAYER GGUF k-quant triple (mixed "XL"
    // packs) and on the unpermute combine mode — one graph per (GPU, variant
    // key), NOT one per GPU. NVFP4 collapses to a single key. The map is
    // bounded (kMaxFfnGraphVariants) — beyond the cap, run eager.
    // INV-MOE-OVERLAP: the wave-pass kind + wave-first bit are baked control
    // flow (memset / accumulate / unpermute-source) → part of the variant key.
    const uint32_t ffn_graph_wave_bits =
        (static_cast<uint32_t>(mp.wave_pass) << 11)
        | (wave_first ? (1u << 13) : 0u);
    const uint32_t ffn_graph_key = ffn_graph_wave_bits | (!use_gguf
        ? (0x8000u | static_cast<uint32_t>(mp.ep_combine_mode) << 9)
        : (static_cast<uint32_t>(gguf_gate_type)
           | static_cast<uint32_t>(gguf_up_type) << 3
           | static_cast<uint32_t>(gguf_down_type) << 6
           | static_cast<uint32_t>(mp.ep_combine_mode) << 9));
    // ── C-6 Task A (graph-hoist overlap): record the input-ready event ──────
    // norm (RMSNorm) + router (topk) are enqueued on THIS gpu's `stream` above
    // and are the ONLY inputs the host CPU-expert fold consumes. Record here —
    // BEFORE the captured FFN replay / eager emit below (covers norm+router) —
    // so the fold's input D2H can ride a side stream keyed to this event and the
    // host compute overlaps the in-flight GPU graph replay instead of draining
    // the device. (A prior record after the RMSNorm protects the norm on the
    // zero-resident skip_routed path where the router does not run.) No-op unless
    // forced-CPU decode ⇒ champion path unchanged.
    record_cpu_input_event(gpu, num_tokens, mp.layer_idx, stream);

    compute::RoutedFfnGraphRunner* ffn_graph_runner = nullptr;
    if (ffn_graph_eligible) {
        auto& gmap = routed_ffn_graphs_[gpu];
        auto it = gmap.find(ffn_graph_key);
        if (it == gmap.end() && gmap.size() < kMaxFfnGraphVariants)
            it = gmap.emplace(ffn_graph_key,
                              std::make_unique<compute::RoutedFfnGraphRunner>())
                     .first;
        if (it != gmap.end()) ffn_graph_runner = it->second.get();
    }

    if (ffn_graph_eligible && ffn_graph_runner) {
        // Fill all three pinned b_ptrs sets (host work, outside capture).
        fill_graph_b_ptrs(0, gate_off_fn,
                          deps_.expert_cache->gate_weight_bytes());
        fill_graph_b_ptrs(1, up_off_fn,
                          deps_.expert_cache->up_weight_bytes());
        fill_graph_b_ptrs(2, down_off_fn,
                          deps_.expert_cache->down_weight_bytes());
        // §12h sub-seam M1b: b_ptrs fills done, graph replay next.
        perf_trace::record(perf_trace::kMoeSegRankPre,
                           static_cast<uint16_t>(mp.gpu_idx), 0,
                           mp.layer_idx << 16, 12);

        auto& runner = *ffn_graph_runner;
        if (!runner.is_captured()) {
            // First eligible decode dispatch on this GPU: capture Step 2..6 then
            // launch the captured graph (executes this step's work). INV-0.6
            // override is logged in moe_ffn_graph_enabled().
            // GG-S1: pre-capture warmup hook (TD-GG5-MMQ-CAPTURE-WARMUP, engine
            // side). No-op today (decode hits only mmvq; the GemmKernels warmup
            // entry point is a follow-up) but issued BEFORE begin_capture so the
            // non-capturable mmq cudaFuncSetAttribute has a home the moment a
            // large-M GGUF GEMM is ever captured.
            runner.warmup(stream);
            if (runner.begin_capture(stream)) {
                emit_routed_ffn(MoeBPtrMode::kGraph);
                if (runner.end_capture(stream)) {
                    runner.replay(stream);  // execute this step via the graph
                    spdlog::warn("TD-DECODE-FFN-GRAPH: captured routed FFN graph "
                                 "on gpu {} (layer {}, variant key 0x{:x})",
                                 gpu, mp.layer_idx, ffn_graph_key);
                } else {
                    // Capture failed at end/instantiate — run live this step and
                    // every step after (runner stays uncaptured).
                    emit_routed_ffn(MoeBPtrMode::kGraph);
                }
            } else {
                emit_routed_ffn(MoeBPtrMode::kGraph);
            }
        } else {
            // Steady state: pinned buffers refilled above → replay copies fresh
            // routing/weights via the captured H2D nodes and recomputes Step 2..6.
            runner.replay(stream);
        }
        // §12h sub-seam M1c: graph replay enqueued.
        perf_trace::record(perf_trace::kMoeSegRankPre,
                           static_cast<uint16_t>(mp.gpu_idx), 0,
                           mp.layer_idx << 16, 13);
    } else {
        // ── Non-graph (default, INV-0.6) path: the SAME op list, emitted eagerly.
        // build_routed_b_ptrs issues each projection's shared-array H2D live; the
        // kernel sequence (permute → populate_meta → gather → quant → gate/up
        // GEMM → SwiGLU → down GEMM → unpermute) is byte-identical to the captured
        // path because both come from emit_routed_ffn (GG-S1). ──
        emit_routed_ffn(MoeBPtrMode::kEager);
    }

    // INV-MOE-OVERLAP: mark the accumulator touched exactly once per dispatch
    // (graph replay or eager — the memset decision was baked via wave_first).
    if (wave_accum_active)
        moe_wave_accum_used_[gpu] = 1;

    // TD-PREFILL-FETCH-SEAM-SCALING: wave-partial pass ends after Step 5 +
    // accumulate — no unpermute/shared expert/allreduce/residual/commit (those
    // run exactly once, in the kFinal pass).
    if (mp.wave_pass == MoeWavePass::kPartial)
        return true;

    moe_valid = true;  // TD-52w: Step 6 produced valid moe_output.

    } else if (deps_.cuda_kernels_enabled && scratch.moe_output) {
        // router_valid is false but skip_routed is false (total_resident > 0 with
        // no router weights — MTP/out-of-range layer). Zero moe_output so Step 7d's
        // residual add (moe_output += shared_expert_output) doesn't read stale data.
        deps_.device_backends[gpu]->memset_async(
            scratch.moe_output, 0,
            static_cast<size_t>(num_tokens) * hidden * 2, stream);
    }
    } // end if (!skip_routed)

    // §12h sub-seam M2: routed emit done, shared expert begins.
    perf_trace::record(perf_trace::kMoeSegRankPre,
                       static_cast<uint16_t>(mp.gpu_idx), 0,
                       mp.layer_idx << 16, 11);

    // Step 7: Shared expert FFN (TD-50b)
    // Always-on expert(s) executed on original hidden states, output added to moe_output.
    // KD-3d-fix: uses TP-local intermediate size (column-parallel gate/up, row-parallel down).
    if (deps_.cuda_kernels_enabled && mc.n_shared_experts > 0
        && hidden_input && scratch.shared_gate_up_output) {
        const Deps::SharedExpertWeights* se = nullptr;
        if (mp.layer_idx < deps_.shared_expert_weight_ptrs.size() &&
            gpu < deps_.shared_expert_weight_ptrs[mp.layer_idx].size()) {
            se = &deps_.shared_expert_weight_ptrs[mp.layer_idx][gpu];
        }
        if (se && se->gate_up && se->down) {
            const int tp = std::max(1, deps_.live_config->parallelism.tensor_parallelism);
            const int intermediate_local = intermediate / tp;

            // V4-7b (ticket H): the raw-BF16 shared-expert route is the
            // ArchDeepseekV4Moe::try_shexp_raw_bf16 hook (DeepSeek-V4 shexp
            // is BF16-native inside a GGUF checkpoint; mixed raw/packed
            // projections fail loud there). Base arch: not handled — the
            // quantized route below runs.
            bool shexp_return_early = false;
            if (arch.try_shexp_raw_bf16(mp, gpu, se, use_gguf, norm_input,
                                        num_tokens, hidden, intermediate_local,
                                        moe_valid, shexp_return_early,
                                        stream)) {
                if (shexp_return_early)
                    return true;
            } else {
            // Write shared_expert_offsets = {0, num_tokens} for single-expert grouped GEMM.
            auto* gpu_dev = deps_.device_backends[gpu];
            const int32_t offsets[2] = {0, num_tokens};
            gpu_dev->memcpy_h2d_async(scratch.shared_expert_offsets, offsets,
                                      2 * sizeof(int32_t), stream);

            // KD-3f: populate shared problem_sizes + sf_offsets for gate+up.
            const int32_t shared_gu_ps[3] = {num_tokens, 2 * intermediate_local, hidden};
            gpu_dev->memcpy_h2d_async(scratch.shared_problem_sizes, shared_gu_ps,
                                      3 * sizeof(int32_t), stream);
            const int32_t shared_gu_sf[2] = {0, ((num_tokens + 127) / 128) * 128};
            gpu_dev->memcpy_h2d_async(scratch.shared_sf_offsets, shared_gu_sf,
                                      2 * sizeof(int32_t), stream);

            // Upload NVFP4 alpha (ws2 * input_scale, precomputed at engine
            // load) + the activation input_scale for the quantizer.
            if (!use_fp8 && scratch.nvfp4_alpha) {
                gpu_dev->memcpy_h2d_async(scratch.nvfp4_alpha,
                                           &se->alpha, sizeof(float), stream);
                gpu_dev->memcpy_h2d_async(scratch.moe_input_scales,
                                           &se->input_scale, sizeof(float), stream);
            }

            // KD-3g: quantize norm_input [B, H] BF16 → FP8/FP4 for shared gate+up.
            // GG-5b: GGUF feeds BF16 norm_input directly (no pre-quant).
            if (scratch.quant_act && !use_gguf) {
                launch_activation_quant({use_fp8, num_tokens, hidden,
                    norm_input, scratch.quant_act, scratch.quant_scale,
                    scratch.quant_scale_bytes, scratch.shared_expert_offsets,
                    scratch.shared_sf_offsets, num_tokens, 1, gpu_dev, stream,
                    use_fp8 ? nullptr : scratch.moe_input_scales});
            }
            // 7a: Gate+Up GEMM — hidden × W_gate_up → shared_gate_up_output.
            // GG-5c: GGUF reads THIS shared expert's own per-projection types
            // (fused when gate==up, split otherwise — same as dense), not routed.
            if (use_gguf) {
                GgufDenseGateUpArgs gu{};
                gu.num_tokens = num_tokens;
                gu.intermediate_local = intermediate_local;
                gu.hidden = hidden;
                gu.a_base = norm_input;
                gu.gate_up_output = scratch.shared_gate_up_output;
                gu.gate_scratch = scratch.shared_activation;   // [B, I_local]
                gu.up_scratch = scratch.gguf_gate_up_split;     // [B, I_dense] ≥ [B, I_local]
                gu.gate_type = to_gguf_compute(se->gate_gguf_type);
                gu.up_type   = to_gguf_compute(se->up_gguf_type);
                gu.up_block_offset = model::gguf::gguf_packed_bytes(
                    intermediate_local, hidden, se->gate_gguf_type);
                gu.strategy = gguf_strategy;
                gu.gate_up_weight = se->gate_up;
                gu.single_b_ptr = scratch.gguf_single_b_ptr;
                gu.expert_offsets =
                    static_cast<const int32_t*>(scratch.shared_expert_offsets);
                gu.dev = dev;
                gu.gpu_dev = gpu_dev;
                gu.gemm_workspace = scratch.gemm_workspace;
                gu.gemm_workspace_bytes = scratch.gemm_workspace_bytes;
                gu.stream = stream;
                launch_gguf_dense_gate_up(gu);
            } else {
                GroupedGemmArgs gargs{use_fp8, 1, 2 * intermediate_local, hidden,
                    scratch.quant_act,
                    se->gate_up, scratch.shared_gate_up_output,
                    scratch.quant_scale, se->gate_up_scales,
                    static_cast<const float*>(scratch.nvfp4_alpha),
                    static_cast<const int32_t*>(scratch.shared_expert_offsets),
                    static_cast<const int32_t*>(scratch.shared_sf_offsets),
                    static_cast<const int32_t*>(scratch.shared_problem_sizes),
                    dev, scratch.gemm_workspace, scratch.gemm_workspace_bytes, stream};
                launch_grouped_gemm(gargs);
            }

            // 7b: SwiGLU — shared_gate_up_output → shared_activation
            // (V4-4b: clamp applies to routed AND shared experts —
            // swiglu_clamp_exp == swiglu_clamp_shexp in the V4 metadata.)
            launch_swiglu(dev, scratch.shared_activation, scratch.shared_gate_up_output,
                          num_tokens, intermediate_local, stream,
                          static_cast<float>(mc.swiglu_limit));

            // KD-3f: re-populate shared problem_sizes for down GEMM.
            const int32_t shared_dn_ps[3] = {num_tokens, hidden, intermediate_local};
            gpu_dev->memcpy_h2d_async(scratch.shared_problem_sizes, shared_dn_ps,
                                      3 * sizeof(int32_t), stream);
            // sf_offsets unchanged ({0, aligned_num_tokens}) — reuse from gate+up.

            // Upload down_proj alpha + input_scale (may differ from gate+up).
            if (!use_fp8 && scratch.nvfp4_alpha) {
                gpu_dev->memcpy_h2d_async(scratch.nvfp4_alpha,
                                           &se->alpha_down, sizeof(float), stream);
                gpu_dev->memcpy_h2d_async(scratch.moe_input_scales,
                                           &se->input_scale_down, sizeof(float), stream);
            }

            // KD-3g: quantize shared_activation [B, I_local] BF16 → FP8/FP4 for shared down.
            // GG-5b: GGUF feeds BF16 shared_activation directly (no pre-quant).
            if (scratch.quant_act && !use_gguf) {
                launch_activation_quant({use_fp8, num_tokens, intermediate_local,
                    scratch.shared_activation, scratch.quant_act, scratch.quant_scale,
                    scratch.quant_scale_bytes, scratch.shared_expert_offsets,
                    scratch.shared_sf_offsets, num_tokens, 1, gpu_dev, stream,
                    use_fp8 ? nullptr : scratch.moe_input_scales});
            }
            // GG-5b: rebind the 1-element B_ptrs array to the shared down weight.
            if (use_gguf) {
                gpu_dev->memcpy_h2d_async(scratch.gguf_single_b_ptr, &se->down,
                                          sizeof(void*), stream);
            }

            // 7c: Down GEMM — shared_activation × W_down → shared_expert_output
            // Row-parallel: each rank computes partial sum, allreduce needed before residual.
            {
                GroupedGemmArgs gargs{use_fp8, 1, hidden, intermediate_local,
                    use_gguf ? scratch.shared_activation : scratch.quant_act,
                    se->down, scratch.shared_expert_output,
                    scratch.quant_scale, se->down_scales,
                    static_cast<const float*>(scratch.nvfp4_alpha),
                    static_cast<const int32_t*>(scratch.shared_expert_offsets),
                    static_cast<const int32_t*>(scratch.shared_sf_offsets),
                    static_cast<const int32_t*>(scratch.shared_problem_sizes),
                    dev, scratch.gemm_workspace, scratch.gemm_workspace_bytes, stream};
                if (use_gguf) {
                    gargs.use_gguf = true;
                    gargs.gguf_type = to_gguf_compute(se->down_gguf_type);  // GG-5c: shared's OWN down type
                    gargs.gguf_strategy = gguf_strategy;
                    gargs.gguf_total_tokens = num_tokens;  // shared: 1 expert, B rows
                    gargs.B_ptrs =
                        static_cast<const void**>(scratch.gguf_single_b_ptr);
                }
                launch_grouped_gemm(gargs);
            }

            // KD-4g: stop here — caller will allreduce shared_expert_output across TP ranks.
            if (mp.phase == MoeDispatchPhase::kPreAllreduce)
                return true;

            // 7d: Add shared expert output to routed MoE output
            // TD-52m: guard on both scratch buffers being non-null.
            if (scratch.moe_output && scratch.shared_expert_output) {
                compute::launch_residual_add(scratch.moe_output, scratch.shared_expert_output,
                                             num_tokens * hidden, stream);
                moe_valid = true;  // TD-52w: moe_output now has valid shared expert data.
            }
            }  // V4-7b raw-BF16 vs quantized shared-expert route
        }
    }

    // KD-4g TD-72f: if kPreAllreduce reached here (no shared expert executed),
    // return early — caller will allreduce and call kPostAllreduce.
    if (mp.phase == MoeDispatchPhase::kPreAllreduce)
        return true;

moe_post_allreduce:
    // KD-4g: post-allreduce — add shared expert to moe_output (if kPostAllreduce),
    // then final residual + commit. For kFull, shared expert add already done above.
    if (mp.phase == MoeDispatchPhase::kPostAllreduce) {
        // shared_expert_output now holds allreduced sum — add to moe_output.
        // TD-52m: guard both buffers non-null.
        if (deps_.cuda_kernels_enabled && scratch.moe_output && scratch.shared_expert_output) {
            compute::launch_residual_add(scratch.moe_output, scratch.shared_expert_output,
                                         num_tokens * hidden, stream);
            moe_valid = true;  // TD-52w: post-allreduce shared expert written to moe_output.
        }
    }

    // Step 8: Residual add — h += moe_output (TD-50c)
    // TD-52w: only add moe_output to hidden state when it contains valid data
    // (set by Step 6 unpermute or Step 7d shared expert add). Prevents adding
    // uninitialized/stale device memory to the hidden state for MTP layers or
    // when all routed+shared expert paths were skipped.
    // ── C-6 Milestone A: TP=1 (non-DCP) forced-CPU-expert fold ──────────────
    // The EP (dispatch_moe_all_ranks) fold runs post-combine; the single-GPU
    // finalize folds here, into moe_output before the residual add. Guarded to
    // the standalone kFull finalize so the per-rank EP sub-dispatches
    // (kPreAllreduce / kPostAllreduce) never fold (all_ranks owns that).
    if (deps_.cuda_kernels_enabled && moe_valid && scratch.moe_output
        && mp.phase == MoeDispatchPhase::kFull
        && !(deps_.dcp_communicator && deps_.dcp_communicator->is_active())
        && cpu_layer_has_forced(mp.layer_idx)) {
        if (!fold_cpu_forced_experts(mp.layer_idx, num_tokens,
                                     {static_cast<int>(gpu)}))
            return false;
    }

    // TD-PREFILL-NONDET diagnostic: MoE-stage seams (env-gated, zero work
    // when LS_SEAM_DUMP is unset) — routed+shared sum ('Mout'), shared-only
    // ('Shex'), and the FFN-stage collapsed hc input ('Mhcx').
    if (deps_.live_config) {
        const int hs = deps_.live_config->model.hidden_size;
        if (scratch.moe_output)
            seam_dump_hidden(0x74754f4du /*'Mout'*/,
                             static_cast<int>(mp.layer_idx),
                             static_cast<int>(gpu), scratch.moe_output,
                             static_cast<int>(num_tokens), hs);
        if (scratch.shared_expert_output)
            seam_dump_hidden(0x78656853u /*'Shex'*/,
                             static_cast<int>(mp.layer_idx),
                             static_cast<int>(gpu),
                             scratch.shared_expert_output,
                             static_cast<int>(num_tokens), hs);
        if (deps_.hc_streams > 1 && scratch.hc_x)
            seam_dump_hidden(0x7863684du /*'Mhcx'*/,
                             static_cast<int>(mp.layer_idx),
                             static_cast<int>(gpu), scratch.hc_x,
                             static_cast<int>(num_tokens), hs);
    }

    // TD-52m: also guard on scratch.moe_output being non-null (alloc failure).
    if (deps_.cuda_kernels_enabled && hidden_input && moe_valid && scratch.moe_output) {
        // V4-5b mHC: Step-8 residual update is hc_post (see dense site above)
        // — ArchDeepseekV4Moe::residual_update; base arch: plain residual add.
        arch.residual_update(gpu, hidden_input, scratch.moe_output,
                             num_tokens, hidden, pair_idx, stream);
    }

    // KD-R2: commit MoE output back to attention buffer for next layer.
    // Must run regardless of moe_valid — pipeline sync requires the commit
    // and event recording even when MoE was a no-op (identity: h unchanged).
    if (pair_idx >= 0) {
        deps_.hidden_state_pairs[pair_idx].commit(
            static_cast<size_t>(num_tokens) * hidden * deps_.hc_streams * 2,
            stream,
            deps_.device_backends[gpu]);

        // TD-73l: Record moe_attn_event so next layer's kAttention can wait.
        if (deps_.hidden_state_pairs[pair_idx].moe_attn_event) {
            deps_.stream_manager->record_event(
                deps_.hidden_state_pairs[pair_idx].moe_attn_event,
                static_cast<int>(gpu), compute::StreamId::kExpertFfn);
        }
    }

    return true;
}

// Fused MoE dispatch. For TP>1, dispatches all TP ranks internally via
// dispatch_moe_all_ranks (matching the attention pattern where DcpExecutor
// handles all ranks). The command must be sent for a single GPU only —
// sending D_B_CMD_RUN_MOE for each TP GPU separately would double-dispatch.
// Aborts if called for a non-primary TP GPU to expose such bugs early.
bool CommandDispatcher::dispatch_fused_moe(const ipc::Command& cmd) {
    // DEPRECATION (production path decision, 2026-07-05): routed-MoE layers
    // execute via E_CMD_FETCH_AND_RUN_MOE in production (fused-gate routing
    // export -> routed list -> progressive fetch+run). RUN_MOE stays for
    // DENSE-FFN layers and test scaffolding. Warn once when a routed layer
    // arrives here so legacy callers surface.
    if (deps_.live_config) {
        const auto& m = deps_.live_config->model;
        const bool routed = m.n_routed_experts > 0
            && static_cast<int>(cmd.run_moe.layer_idx)
                   >= m.first_k_dense_replace;
        static bool warned = false;
        if (routed && !warned) {
            warned = true;
            spdlog::warn("D_B_CMD_RUN_MOE on routed layer {} — deprecated for "
                         "routed MoE; production uses E_CMD_FETCH_AND_RUN_MOE",
                         cmd.run_moe.layer_idx);
        }
    }
    InternalMoeParams mp{};
    mp.layer_idx    = cmd.run_moe.layer_idx;
    mp.num_seqs     = cmd.run_moe.num_seqs;
    mp.gpu_idx      = cmd.gpu_idx;
    mp.topk_override = 0;
    mp.store_gating = (cmd.run_moe.store_gating_output != 0);
    mp.moe_mode     = cmd.run_moe.moe_mode;
    // F-2: optional consume of precomputed top-K already in moe_scratch_.
    mp.use_precomputed_gating = (cmd.run_moe.use_precomputed_gating != 0);
    // F-7: emit_checkpoint on RUN_MOE means "publish the attention↔MoE seam
    // routing (top-K) to the sideband before the expert GEMM".
    mp.emit_seam_checkpoint = (cmd.run_moe.emit_checkpoint != 0);

    // KD-4g: TP>1 — dispatch all ranks with allreduce coordination.
    // TD-72b: abort if called for non-primary TP GPU (double-dispatch bug).
    if (deps_.dcp_communicator && deps_.dcp_communicator->is_active()) {
        if (deps_.dcp_executor && !deps_.dcp_executor->gpus().empty()) {
            const int primary = deps_.dcp_executor->gpus()[0].position;
            if (static_cast<int>(cmd.gpu_idx) != primary) {
                spdlog::critical("dispatch_fused_moe: D_B_CMD_RUN_MOE sent for non-primary "
                                 "TP GPU {} (primary={}). Send for primary GPU only — "
                                 "dispatch_moe_all_ranks handles all ranks internally.",
                                 cmd.gpu_idx, primary);
                std::abort();
            }
        }
        return dispatch_moe_all_ranks(mp);
    }
    return dispatch_moe_internal(mp);
}

}  // namespace layerstorm::daemon
