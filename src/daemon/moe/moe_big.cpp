// TD-PREFILL-MOE-BIG: chunked big-batch MoE execution + the
// E_CMD_FETCH_AND_RUN_MOE_BIG handler glue.
//
// dispatch_moe_chunked_internal() is the big-batch sibling of
// dispatch_moe.cpp::dispatch_moe_internal(): the SAME step sequence
// (norm → gate → permute → gate/up GEMM → SwiGLU → down GEMM → unpermute →
// shared expert → allreduce phases → residual → commit), but the token batch
// is processed in chunks of <= moe_chunk_capacity_ tokens so the TRANSIENT
// scratch (permuted_input / gate_up / activation / expert_output / quant —
// all sized at the chunk bound, see command_dispatcher.cpp sizing) is REUSED
// per chunk. Only the persistent [B, H] buffers (moe_output, normalized
// hidden, shared_expert_output) and the [B*topk] routing span the batch —
// this is what makes the batch capacity elastic (never OOM) instead of the
// ~563 KB/token transient scaling (spec/bloat/FETCH_AND_RUN_MOE_BIG.md §3).
//
// Invariants (spec/INVARIANTS.md):
//   INV-MOE-BIG-1  Batches <= moe_chunk_capacity_ NEVER enter this path — the
//                  legacy single-shot pipeline stays byte-identical (all
//                  decode, all existing goldens' 64-token prefill chunks).
//   INV-MOE-BIG-2  Chunked WAVE passes accumulate each wave's chunk-unpermuted
//                  partial into moe_output ([B, H] bf16, zeroed once per
//                  command). A permuted row belongs to exactly one expert, so
//                  within one wave the per-token K-slot sum only sees that
//                  wave's experts (absent-expert rows are exact zeros from the
//                  zero-weight GEMM). SINGLE-wave commands are therefore
//                  bit-identical to the single-shot result (0 + x = x);
//                  multi-wave commands round per wave boundary (token-level
//                  identity, not bit-level — the reduction order differs).
//   INV-MOE-BIG-3  The canonical per-slot EP combine (deterministic_ep_combine)
//                  is single-shot only; chunked dispatches use the legacy
//                  mode-0 [B, H] bf16 EP allreduce (per-slot scratch is sized
//                  at the chunk bound).
//   INV-MOE-BIG-4  Chunked DENSE layers write the down-GEMM into moe_output
//                  (not expert_output, which is chunk-sized) — the TP dense
//                  allreduce and residual read moe_output for chunked batches.
//
// Part of CommandDispatcher — see command_dispatcher.h.

#include "daemon/command_dispatcher.h"
#include "daemon/dispatch_detail.h"
#include "daemon/moe/arch_mla_moe.h"
#include "daemon/moe/arch_deepseek_v4_moe.h"
#include "daemon/moe/quant_routes.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include <spdlog/spdlog.h>

#include "compute/stream_manager.h"
#include "core/device_backend.h"
#include "core/expert_device.h"
#include "core/memory/expert_cache.h"
#include "model/quantization/gguf_kquant.h"
#include "parallelism/dcp_communicator.h"
#include "parallelism/dcp_executor.h"
#include "compute/kernels/elementwise/residual_add.h"
#include "compute/kernels/mhc/mhc.h"
#include "compute/kernels/moe/moe_gemm_meta.h"
#include "compute/kernels/moe/router_projection.h"
#include "compute/kernels/norm/rmsnorm.h"
#include "compute/kernels/moe/hash_gating.h"  // V4-4 hash-layer gating
#include "compute/kernels/sm120/gemm/bf16_gemm.h"  // SC raw-BF16 shexp route
#include "sm120/gating/topk_gating.h"

namespace layerstorm::daemon {

bool CommandDispatcher::dispatch_moe_chunked_internal(const InternalMoeParams& mp) {
    const uint32_t gpu = mp.gpu_idx;

    auto* dev = expert_dev(gpu);
    if (!dev) {
        spdlog::warn("dispatch_moe_chunked: no expert device for gpu {}", gpu);
        return false;
    }
    if (gpu >= moe_scratch_.size() || !moe_scratch_[gpu].topk_weights) {
        spdlog::warn("dispatch_moe_chunked: scratch not ready for gpu {}", gpu);
        return false;
    }
    if (!deps_.live_config) {
        spdlog::warn("dispatch_moe_chunked: no live_config");
        return false;
    }

    const auto& mc = deps_.live_config->model;
    const auto& scratch = moe_scratch_[gpu];
    const int num_tokens = static_cast<int>(mp.num_seqs);
    const int topk = (mp.topk_override > 0) ? mp.topk_override
                                            : mc.num_experts_per_tok;
    const int hidden = mc.hidden_size;
    const int intermediate = mc.moe_intermediate_size;
    const int n_experts = mc.n_routed_experts;
    const int expanded_tokens = num_tokens * topk;

    // MoE by-model split (moe/arch_base.h): the chunked sibling reuses the
    // SAME MoeArch hooks as the single-shot driver (INV-MOE-ARCH); selection
    // and lazy construction mirror dispatch_moe_internal.
    const bool is_v4 =
        mc.architecture == config::Architecture::deepseek_v4;
    if (!moe_arch_mla_) {
        moe_arch_mla_ = std::make_unique<ArchMlaMoe>(*this);
        moe_arch_v4_  = std::make_unique<ArchDeepseekV4Moe>(*this);
    }
    MoeArch& arch = is_v4 ? *moe_arch_v4_ : *moe_arch_mla_;

    // Chunk size: per-command override (E_CMD_FETCH_AND_RUN_MOE_BIG
    // chunk_tokens), clamped to the transient scratch bound.
    const int chunk = std::clamp(
        (mp.chunk_tokens > 0) ? mp.chunk_tokens : moe_chunk_capacity_,
        1, moe_chunk_capacity_);

    dev->set_device();
    void* stream = deps_.stream_manager->stream(
        static_cast<int>(gpu), compute::StreamId::kExpertFfn);
    auto* gpu_dev = deps_.device_backends[gpu];

    const int pair_idx = resolve_pair_idx(gpu);
    if (pair_idx >= 0) {
        const auto& pair = deps_.hidden_state_pairs[pair_idx];
        if (pair.attn_moe_event) {
            deps_.stream_manager->wait_event(
                static_cast<int>(gpu), compute::StreamId::kExpertFfn,
                pair.attn_moe_event);
        }
    }

    // Resolve hidden state input.
    void* hidden_input = nullptr;
    if (pair_idx >= 0) {
        hidden_input = deps_.hidden_state_pairs[pair_idx].moe_buf;
    } else if (gpu < deps_.fused_moe_hidden_states.size()) {
        hidden_input = deps_.fused_moe_hidden_states[gpu];
    }
    auto row_ptr = [hidden](void* base, int row) -> void* {
        return static_cast<uint8_t*>(base)
             + static_cast<size_t>(row) * hidden * 2;
    };

    // TD-PREFILL-FETCH-SEAM-SCALING: rolling-wave pass? (Declared up front —
    // the norm skip and the wave-masked permute below both key off it.)
    const bool wave_mode = mp.wave_pass != MoeWavePass::kNone;
    // Wave passes re-enter this function once per wave with an UNCHANGED
    // hidden_input (the residual/commit runs only at kFinal). normalized_hidden
    // is persistent [B, H] and nothing else writes it inside one progressive
    // command, so the full-batch RMSNorm only needs to run on the FIRST pass
    // (moe_wave_accum_used_[gpu] is set by that pass just below) — later waves
    // and the kFinal pass reuse the buffer.
    const bool norm_cached = wave_mode
        && gpu < moe_wave_accum_used_.size() && moe_wave_accum_used_[gpu];

    // Pre-MoE RMSNorm over the FULL batch (normalized_hidden is [B, H]).
    void* norm_input = hidden_input;
    if (hidden_input && scratch.normalized_hidden) {
        const void* norm_w = nullptr;
        const int layer = static_cast<int>(mp.layer_idx);
        if (pair_idx >= 0
            && layer < static_cast<int>(deps_.per_layer_attn_weights.size())) {
            const int r = deps_.hidden_state_pairs[pair_idx].rank;
            if (r >= 0
                && r < static_cast<int>(deps_.per_layer_attn_weights[layer].size())) {
                norm_w =
                    deps_.per_layer_attn_weights[layer][r].post_attention_layernorm;
            }
        } else if (deps_.dcp_executor
                   && layer < static_cast<int>(deps_.per_layer_attn_weights.size())) {
            const auto& tp_gpus = deps_.dcp_executor->gpus();
            for (int r = 0; r < static_cast<int>(tp_gpus.size()); ++r) {
                if (tp_gpus[r].position == static_cast<int>(gpu)
                    && r < static_cast<int>(deps_.per_layer_attn_weights[layer].size())) {
                    norm_w =
                        deps_.per_layer_attn_weights[layer][r].post_attention_layernorm;
                    break;
                }
            }
        }
        if (norm_w) {
            if (!norm_cached) {
                // V4-5b mHC: collapse the hc-stream residual first; the
                // hc_post/hc_comb outputs persist in scratch for kFinal's
                // hc_post (same caching contract as normalized_hidden).
                const void* big_rms_src = hidden_input;
                if (deps_.hc_streams > 1) {
                    const parallelism::AttentionLayerWeights* lw = nullptr;
                    if (pair_idx >= 0) {
                        const int r = deps_.hidden_state_pairs[pair_idx].rank;
                        lw = &deps_.per_layer_attn_weights[layer][r];
                    }
                    if (!lw || !lw->hc_ffn_fn || !scratch.hc_x) {
                        spdlog::error("dispatch_moe_big: mHC active but hc_ffn "
                                      "weights/scratch missing (layer {})", layer);
                        return false;
                    }
                    compute::launch_mhc_pre(
                        scratch.hc_x, scratch.hc_post, scratch.hc_comb,
                        hidden_input, lw->hc_ffn_fn, lw->hc_ffn_scale,
                        lw->hc_ffn_base, mc.rms_norm_eps, mc.hc_eps, 2.0f,
                        mc.hc_sinkhorn_iters, num_tokens, deps_.hc_streams,
                        hidden, stream);
                    big_rms_src = scratch.hc_x;
                }
                compute::launch_rmsnorm(
                    scratch.normalized_hidden, big_rms_src, norm_w,
                    mc.rms_norm_eps, num_tokens, hidden,
                    compute::NormDtype::kBFloat16, stream);
            }
            norm_input = scratch.normalized_hidden;
        }
    }

    // GG-5b/GG-5c/GG-9: weight-quant route derivation — the shared single
    // source (moe/quant_routes.h; this block formerly mirrored
    // dispatch_moe_internal verbatim under a keep-in-sync comment). The
    // chunked path only runs with CUDA kernels enabled, so the builder's
    // cuda-gated validation fires exactly as the old inline copy did.
    const MoeQuantRoutes qr = build_moe_quant_routes(
        deps_, mp.layer_idx, intermediate, hidden, "dispatch_moe_chunked");
    const bool use_fp8 = qr.use_fp8;
    const bool use_gguf = qr.use_gguf;
    const compute::GgufGemmStrategy gguf_strategy = qr.gguf_strategy;
    auto to_gguf_compute = [](model::GgufKQuantType t) -> compute::GgufQuantType {
        return MoeQuantRoutes::to_gguf_compute(t);
    };
    const compute::GgufQuantType gguf_gate_type = qr.gguf_gate_type;
    const compute::GgufQuantType gguf_up_type   = qr.gguf_up_type;
    const compute::GgufQuantType gguf_down_type = qr.gguf_down_type;
    auto gate_off_fn = [&](const memory::CacheEntry* e) { return qr.gate_off(e); };
    auto up_off_fn   = [&](const memory::CacheEntry* e) { return qr.up_off(e); };
    auto down_off_fn = [&](const memory::CacheEntry* e) { return qr.down_off(e); };

    last_moe_miss_count_ = 0;
    last_seam_checkpoint_ = SeamCheckpoint{};

    // ── Dense layer early-out (chunked) ────────────────────────────────────
    // INV-MOE-BIG-4: the dense down-GEMM writes moe_output rows (persistent
    // [B, H]); expert_output is chunk-sized and reused per chunk.
    const int first_k_dense = mc.first_k_dense_replace;
    if (static_cast<int>(mp.layer_idx) < first_k_dense) {
        const Deps::DenseFFNWeights* dw = nullptr;
        if (mp.layer_idx < deps_.dense_ffn_weight_ptrs.size() &&
            gpu < deps_.dense_ffn_weight_ptrs[mp.layer_idx].size()) {
            dw = &deps_.dense_ffn_weight_ptrs[mp.layer_idx][gpu];
        }
        if (!dw || !dw->gate_up || !dw->down) {
            spdlog::warn("dispatch_moe_chunked: dense layer {} missing "
                         "DenseFFNWeights on gpu {}", mp.layer_idx, gpu);
            return false;
        }

        if (mp.phase != MoeDispatchPhase::kPostAllreduce) {
            const int tp_val =
                std::max(1, deps_.live_config->parallelism.tensor_parallelism);
            const int I_dense_local = mc.intermediate_size / tp_val;
            if (!scratch.shared_expert_offsets || !scratch.shared_problem_sizes ||
                !scratch.shared_sf_offsets || !scratch.moe_output) {
                spdlog::warn("dispatch_moe_chunked: dense layer {} missing "
                             "shared metadata buffers on gpu {}", mp.layer_idx, gpu);
                return false;
            }

            for (int off = 0; off < num_tokens; off += chunk) {
                const int len = std::min(chunk, num_tokens - off);
                const void* a_in = row_ptr(norm_input, off);

                const int32_t d_offsets[2] = {0, len};
                gpu_dev->memcpy_h2d_async(scratch.shared_expert_offsets,
                                          d_offsets, 2 * sizeof(int32_t), stream);
                const int32_t d_gu_ps[3] = {len, 2 * I_dense_local, hidden};
                gpu_dev->memcpy_h2d_async(scratch.shared_problem_sizes, d_gu_ps,
                                          3 * sizeof(int32_t), stream);
                const int32_t d_gu_sf[2] = {0, ((len + 127) / 128) * 128};
                gpu_dev->memcpy_h2d_async(scratch.shared_sf_offsets, d_gu_sf,
                                          2 * sizeof(int32_t), stream);

                if (!use_fp8 && scratch.nvfp4_alpha) {
                    gpu_dev->memcpy_h2d_async(scratch.nvfp4_alpha,
                                              &dw->alpha, sizeof(float), stream);
                    gpu_dev->memcpy_h2d_async(scratch.moe_input_scales,
                                              &dw->input_scale, sizeof(float),
                                              stream);
                }
                if (scratch.quant_act && !use_gguf) {
                    launch_activation_quant({use_fp8, len, hidden,
                        a_in, scratch.quant_act, scratch.quant_scale,
                        scratch.quant_scale_bytes, scratch.shared_expert_offsets,
                        scratch.shared_sf_offsets, len, 1, gpu_dev, stream,
                        use_fp8 ? nullptr : scratch.moe_input_scales});
                }
                if (use_gguf) {
                    GgufDenseGateUpArgs gu{};
                    gu.num_tokens = len;
                    gu.intermediate_local = I_dense_local;
                    gu.hidden = hidden;
                    gu.a_base = a_in;
                    gu.gate_up_output = scratch.gate_up_output;
                    gu.gate_scratch = scratch.activation_output;
                    gu.up_scratch = scratch.gguf_gate_up_split;
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
                        dev, scratch.gemm_workspace, scratch.gemm_workspace_bytes,
                        stream};
                    launch_grouped_gemm(gargs);
                }

                launch_swiglu(dev, scratch.activation_output,
                              scratch.gate_up_output, len, I_dense_local, stream,
                              static_cast<float>(mc.swiglu_limit));

                const int32_t d_dn_ps[3] = {len, hidden, I_dense_local};
                gpu_dev->memcpy_h2d_async(scratch.shared_problem_sizes, d_dn_ps,
                                          3 * sizeof(int32_t), stream);
                if (!use_fp8 && scratch.nvfp4_alpha) {
                    gpu_dev->memcpy_h2d_async(scratch.nvfp4_alpha,
                                              &dw->alpha_down, sizeof(float),
                                              stream);
                    gpu_dev->memcpy_h2d_async(scratch.moe_input_scales,
                                              &dw->input_scale_down,
                                              sizeof(float), stream);
                }
                if (scratch.quant_act && !use_gguf) {
                    launch_activation_quant({use_fp8, len, I_dense_local,
                        scratch.activation_output, scratch.quant_act,
                        scratch.quant_scale, scratch.quant_scale_bytes,
                        scratch.shared_expert_offsets, scratch.shared_sf_offsets,
                        len, 1, gpu_dev, stream,
                        use_fp8 ? nullptr : scratch.moe_input_scales});
                }
                if (use_gguf) {
                    gpu_dev->memcpy_h2d_async(scratch.gguf_single_b_ptr,
                                              &dw->down, sizeof(void*), stream);
                }
                {
                    GroupedGemmArgs gargs{use_fp8, 1, hidden, I_dense_local,
                        use_gguf ? scratch.activation_output : scratch.quant_act,
                        dw->down, row_ptr(scratch.moe_output, off),
                        scratch.quant_scale, dw->down_scales,
                        static_cast<const float*>(scratch.nvfp4_alpha),
                        static_cast<const int32_t*>(scratch.shared_expert_offsets),
                        static_cast<const int32_t*>(scratch.shared_sf_offsets),
                        static_cast<const int32_t*>(scratch.shared_problem_sizes),
                        dev, scratch.gemm_workspace, scratch.gemm_workspace_bytes,
                        stream};
                    if (use_gguf) {
                        gargs.use_gguf = true;
                        gargs.gguf_type = to_gguf_compute(dw->down_gguf_type);
                        gargs.gguf_strategy = gguf_strategy;
                        gargs.gguf_total_tokens = len;
                        gargs.B_ptrs =
                            static_cast<const void**>(scratch.gguf_single_b_ptr);
                    }
                    launch_grouped_gemm(gargs);
                }
            }

            if (mp.phase == MoeDispatchPhase::kPreAllreduce)
                return true;
        }  // end pre-allreduce block

        // Post-allreduce (and kFull): residual add + commit.
        // INV-MOE-BIG-4: the (allreduced) dense FFN output is in moe_output.
        // V4-5b mHC: the residual update is hc_post (extras skip) —
        // ArchDeepseekV4Moe::residual_update; base arch: plain residual add.
        arch.residual_update(gpu, hidden_input, scratch.moe_output,
                             num_tokens, hidden, pair_idx, stream);
        if (pair_idx >= 0) {
            deps_.hidden_state_pairs[pair_idx].commit(
                static_cast<size_t>(num_tokens) * hidden * deps_.hc_streams * 2,
                stream,
                deps_.device_backends[gpu]);
            if (deps_.hidden_state_pairs[pair_idx].moe_attn_event) {
                deps_.stream_manager->record_event(
                    deps_.hidden_state_pairs[pair_idx].moe_attn_event,
                    static_cast<int>(gpu), compute::StreamId::kExpertFfn);
            }
        }
        return true;
    }

    // ── Routed MoE ─────────────────────────────────────────────────────────

    int total_resident = 0;
    bool skip_routed = false;
    bool moe_valid = false;

    // Chunked wave accumulation (INV-MOE-BIG-2): kPartial/kFinal passes add
    // each wave's chunk-unpermuted partial into moe_output. The first pass of
    // a command zeroes the accumulator (moe_wave_accum_used_ is reset by
    // handle_fetch_and_run_moe_impl).
    if (wave_mode) {
        if (!scratch.big_unperm_tmp || !scratch.moe_output
            || gpu >= moe_wave_accum_used_.size()) {
            spdlog::error("dispatch_moe_chunked: wave pass without "
                          "big_unperm_tmp/moe_output on gpu {} (layer {})",
                          gpu, mp.layer_idx);
            return false;
        }
        if (mp.phase != MoeDispatchPhase::kPostAllreduce
            && !moe_wave_accum_used_[gpu]) {
            gpu_dev->memset_async(scratch.moe_output, 0,
                                  static_cast<size_t>(num_tokens) * hidden * 2,
                                  stream);
            moe_wave_accum_used_[gpu] = 1;
        }
    }
    // INV-MOE-BIG-3: the canonical per-slot EP combine is single-shot only.
    if (mp.ep_combine_mode != 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            spdlog::warn("dispatch_moe_chunked: deterministic_ep_combine is "
                         "single-shot only — chunked batch falls back to the "
                         "legacy mode-0 EP combine (INV-MOE-BIG-3)");
        }
    }

    if (mp.phase == MoeDispatchPhase::kPostAllreduce)
        goto chunked_post_allreduce;

    // Per-expert resident bitset (mirrors dispatch_moe_internal TD-89m).
    {
        auto& bitset = moe_scratch_[gpu].expert_resident_bitset;
        if (!mp.bitset_precomputed) {
            std::memset(bitset.data(), 0, bitset.size());
            if (deps_.expert_cache) {
                for (int e = 0; e < n_experts; ++e) {
                    memory::ExpertKey key{static_cast<uint32_t>(mp.layer_idx),
                                          static_cast<uint16_t>(e)};
                    const auto* entry =
                        deps_.expert_cache->lookup(key, static_cast<int>(gpu));
                    if (entry
                        && (entry->sub_components_ready & memory::SubComponent::kAll)
                               == memory::SubComponent::kAll) {
                        bitset[e / 8] |= static_cast<uint8_t>(1u << (e % 8));
                        ++total_resident;
                    }
                }
            }
        } else {
            for (int e = 0; e < n_experts; ++e) {
                if ((bitset[e / 8] >> (e % 8)) & 1)
                    ++total_resident;
            }
        }
    }

    // Zero-resident short-circuit. A kFinal wave pass must still run Step 7+
    // over the accumulated moe_output; kPartial with nothing resident adds
    // nothing (accumulator already zeroed above if first).
    if (total_resident == 0 && mp.wave_pass != MoeWavePass::kFinal) {
        if (mp.wave_pass == MoeWavePass::kPartial)
            return true;  // nothing to accumulate this wave
        if (scratch.moe_output) {
            gpu_dev->memset_async(
                scratch.moe_output, 0,
                static_cast<size_t>(num_tokens) * hidden * 2, stream);
        }
        last_moe_miss_count_ = static_cast<uint8_t>(std::min(topk, 255));
        skip_routed = true;
        publish_seam_routing(mp, gpu, expanded_tokens, stream);
    }

    if (!skip_routed) {

    // Gating (batch-level, persistent buffers): precomputed (F-2) or self-gate.
    bool router_valid = false;
    if (mp.use_precomputed_gating) {
        const int num_layers = mc.num_hidden_layers;
        router_valid = static_cast<int>(mp.layer_idx) >= first_k_dense
                    && static_cast<int>(mp.layer_idx) < num_layers;
    } else {
        if (norm_input) {
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
        if (!router_valid) {
            const int num_layers = mc.num_hidden_layers;
            const bool is_real_moe_layer =
                static_cast<int>(mp.layer_idx) >= first_k_dense
                && static_cast<int>(mp.layer_idx) < num_layers;
            if (is_real_moe_layer) {
                spdlog::error("dispatch_moe_chunked: router projection failed "
                              "for MoE layer {} on gpu {}", mp.layer_idx, gpu);
                return false;
            }
        }
        // Expert selection via the shared MoeArch hook (V4-4 hash layer:
        // ids from tid2eid[token_id]; the big batch's hidden rows
        // [0, num_tokens) align with moe_token_ids rows — embedding writes
        // each sub-chunk at its row_offset. Base arch: learned top-K).
        // Dedup deltas vs the former inline twin (ledgered): error-log
        // prefix now "dispatch_moe:", and the env-gated LS_DRIFT_DUMP
        // diagnostic now also covers big-batch gating.
        if (!arch.select_experts(mp, gpu, num_tokens, topk, n_experts,
                                 router_valid, stream))
            return false;
    }

    publish_seam_routing(mp, gpu, expanded_tokens, stream);

    if (router_valid) {

    // Routed-miss telemetry (mirrors TD-89m; one D2H over the full batch).
    // WAVE passes skip it entirely: the per-pass bitset is a wave SUBSET (the
    // count would be meaningless), the FETCH finalize computes its own
    // miss_count from arrived/total, and the D2H event spin-poll here is a
    // per-wave host sync that would serialize the wave pipeline.
    if (mp.wave_pass == MoeWavePass::kNone) {
        auto& bitset = moe_scratch_[gpu].expert_resident_bitset;
        if (total_resident < n_experts) {
            auto& indices_host = moe_scratch_[gpu].topk_indices_host;
            const size_t copy_bytes =
                static_cast<size_t>(num_tokens) * topk * sizeof(int32_t);
            gpu_dev->memcpy_d2h_async(indices_host.data(), scratch.topk_indices,
                                      copy_bytes, stream);
            void* sync_event =
                deps_.stream_manager->create_event(static_cast<int>(gpu));
            deps_.stream_manager->record_event(sync_event, static_cast<int>(gpu),
                                               compute::StreamId::kExpertFfn);
            bool d2h_ok = false;
            constexpr int kMaxD2hPollIters = 100000;  // big batch: allow ~100ms
            for (int poll_i = 0; poll_i < kMaxD2hPollIters; ++poll_i) {
                auto [status, err] = deps_.stream_manager->query_event(
                    sync_event, static_cast<int>(gpu));
                if (status == compute::EventStatus::kReady) { d2h_ok = true; break; }
                if (status == compute::EventStatus::kError) {
                    spdlog::error("dispatch_moe_chunked: D2H sync event error "
                                  "on gpu {}", gpu);
                    break;
                }
            }
            deps_.stream_manager->destroy_event(sync_event, static_cast<int>(gpu));
            if (d2h_ok) {
                uint8_t seen_missing[32] = {};
                uint8_t miss_count = 0;
                for (int i = 0; i < num_tokens * topk; ++i) {
                    const int e = indices_host[i];
                    if (e >= 0 && e < n_experts) {
                        const bool is_resident = (bitset[e / 8] >> (e % 8)) & 1;
                        const bool already_seen =
                            (seen_missing[e / 8] >> (e % 8)) & 1;
                        if (!is_resident && !already_seen) {
                            seen_missing[e / 8] |=
                                static_cast<uint8_t>(1u << (e % 8));
                            if (miss_count < 255) ++miss_count;
                        }
                    }
                }
                last_moe_miss_count_ = miss_count;
            } else {
                spdlog::warn("dispatch_moe_chunked: D2H poll exhausted on gpu {}"
                             " (layer {}), reporting 0 misses", gpu, mp.layer_idx);
            }
        } else {
            last_moe_miss_count_ = 0;
        }
        if (mp.moe_mode == 1)
            last_moe_miss_count_ = 0;
        if (mp.store_gating)
            publish_routing_export(gpu, num_tokens, topk, mp.layer_idx, stream);
    }

    // Per-projection expert B/scale-B pointers — chunk-INVARIANT (they depend
    // only on the residency bitset + cache slots), built ONCE into host
    // vectors; each chunk H2Ds them into the shared routed_b_ptrs device
    // array before its projection GEMM (the array is reused per projection,
    // exactly like the eager single-shot path).
    {
        auto& bitset = moe_scratch_[gpu].expert_resident_bitset;
        std::vector<const void*> b_gate(n_experts), sb_gate(n_experts);
        std::vector<const void*> b_up(n_experts),   sb_up(n_experts);
        std::vector<const void*> b_down(n_experts), sb_down(n_experts);
        auto fill = [&](std::vector<const void*>& b, std::vector<const void*>& sb,
                        auto off_fn, int64_t weight_bytes) {
            for (int e = 0; e < n_experts; ++e) {
                const bool is_resident = (bitset[e / 8] >> (e % 8)) & 1;
                const memory::CacheEntry* entry = nullptr;
                if (is_resident) {
                    memory::ExpertKey key{static_cast<uint32_t>(mp.layer_idx),
                                          static_cast<uint16_t>(e)};
                    entry = deps_.expert_cache
                        ? deps_.expert_cache->lookup(key, static_cast<int>(gpu))
                        : nullptr;
                }
                if (!entry || !entry->vram_address) {
                    b[e]  = scratch.zero_weight_buf;
                    sb[e] = scratch.zero_weight_buf;
                    continue;
                }
                auto* base = static_cast<uint8_t*>(entry->vram_address);
                const int64_t off = off_fn(entry);
                b[e]  = base + off;
                sb[e] = base + off + weight_bytes;
            }
        };
        fill(b_gate, sb_gate, gate_off_fn, deps_.expert_cache->gate_weight_bytes());
        fill(b_up,   sb_up,   up_off_fn,   deps_.expert_cache->up_weight_bytes());
        fill(b_down, sb_down, down_off_fn, deps_.expert_cache->down_weight_bytes());

        // ── TD-MOE-BIG-GEMM-SWEEP: wave-masked permute ──────────────────────
        // A rolling-wave pass covers only the wave's resident experts (bitset),
        // yet the permute used to group ALL routed rows — so every grouped GEMM
        // swept the full batch with zero-weight groups for the absent experts
        // (waves × chunks × 3 GEMMs over all E tile-floored groups = the
        // dominant prefill MoE cost). Masking non-wave experts' top-K entries
        // to the -1 permute sentinel sorts their rows PAST expert_offsets[E]:
        // absent experts get zero-length groups (zero mmq tiles / mmvq
        // early-return / M_e=0 problem sizes), and each wave pays only for its
        // own rows. Bit-identity: the wave's own rows keep their relative
        // order inside each expert group (stable radix sort) so their GEMM
        // values are unchanged, and the masked rows' unpermute contribution is
        // an exact +0 from the expert_output memset below — exactly what the
        // zero-weight GEMM produced. NVFP4 is excluded: its grouped activation
        // quant resolves each row's expert by offset search and dead rows
        // (past offsets[E]) would index out of range (see TECH_DEBT
        // TD-MOE-BIG-WAVE-MASK-NVFP4).
        const bool wave_mask = wave_mode && moe_wave_mask_enabled()
            && scratch.wave_masked_topk && scratch.wave_expert_mask
            && (use_gguf || use_fp8);
        if (wave_mask) {
            auto& mh = moe_scratch_[gpu].wave_expert_mask_host;
            mh.assign(static_cast<size_t>(n_experts), 0);
            for (int e = 0; e < n_experts; ++e)
                if ((bitset[e / 8] >> (e % 8)) & 1) mh[e] = 1;
            gpu_dev->memcpy_h2d_async(scratch.wave_expert_mask, mh.data(),
                                      static_cast<size_t>(n_experts), stream);
        }

        auto upload_bptrs = [&](const std::vector<const void*>& b,
                                const std::vector<const void*>& sb) {
            gpu_dev->memcpy_h2d_async(scratch.routed_b_ptrs, b.data(),
                                      n_experts * sizeof(void*), stream);
            gpu_dev->memcpy_h2d_async(scratch.routed_sb_ptrs, sb.data(),
                                      n_experts * sizeof(void*), stream);
        };
        auto bp  = static_cast<const void**>(scratch.routed_b_ptrs);
        auto sbp = static_cast<const void**>(scratch.routed_sb_ptrs);

        // ── The chunk loop: Steps 2..6 per chunk, transient scratch reused ──
        for (int off = 0; off < num_tokens; off += chunk) {
            const int len = std::min(chunk, num_tokens - off);
            const int exp_len = len * topk;
            const void* chunk_norm = row_ptr(norm_input, off);
            const int32_t* chunk_topk_idx =
                static_cast<const int32_t*>(scratch.topk_indices)
                + static_cast<size_t>(off) * topk;
            const float* chunk_topk_w =
                static_cast<const float*>(scratch.topk_weights)
                + static_cast<size_t>(off) * topk;

            const MoeGemmEmitter routed_gemm_emit{
                dev, scratch.gemm_workspace, scratch.gemm_workspace_bytes,
                stream, n_experts, use_fp8, use_gguf, gguf_strategy,
                /*total_tokens=*/exp_len,
                static_cast<const int32_t*>(scratch.expert_offsets),
                static_cast<const int32_t*>(scratch.sf_offsets),
                static_cast<const int32_t*>(scratch.problem_sizes)};

            // Step 2: Permute the chunk's rows by expert assignment. Wave
            // passes permute the MASKED top-K (non-wave experts → -1 sentinel
            // → rows sort past expert_offsets[E], excluded from every GEMM
            // group). The unpermute below still uses the ORIGINAL weights —
            // masked slots read exact zeros from the memset'd expert_output.
            const int32_t* permute_topk = chunk_topk_idx;
            if (wave_mask) {
                compute::launch_mask_topk_indices(
                    static_cast<int32_t*>(scratch.wave_masked_topk),
                    chunk_topk_idx,
                    static_cast<const uint8_t*>(scratch.wave_expert_mask),
                    exp_len, n_experts, stream);
                permute_topk =
                    static_cast<const int32_t*>(scratch.wave_masked_topk);
            }
            dev->moe_permute(
                scratch.permuted_input,
                static_cast<int32_t*>(scratch.expert_offsets),
                static_cast<int32_t*>(scratch.src_to_dest_map),
                static_cast<int32_t*>(scratch.permuted_idx),
                chunk_norm, permute_topk,
                len, topk, hidden, n_experts,
                /*elem_size_bytes=*/2,
                scratch.permute_workspace, stream);

            // -- Gate --
            upload_bptrs(b_gate, sb_gate);
            compute::launch_populate_gemm_meta(
                static_cast<int32_t*>(scratch.problem_sizes),
                use_fp8 ? nullptr : static_cast<int32_t*>(scratch.sf_offsets),
                static_cast<const int32_t*>(scratch.expert_offsets),
                intermediate, hidden, n_experts, stream);
            if (!use_fp8 && !use_gguf && scratch.nvfp4_alpha) {
                compute::launch_gather_alphas_scaled(
                    static_cast<float*>(scratch.nvfp4_alpha),
                    static_cast<float*>(scratch.moe_input_scales),
                    bp,
                    deps_.expert_cache->gate_bytes() - 8,
                    deps_.expert_cache->gate_bytes() - 4,
                    n_experts, stream);
            }
            if (scratch.quant_act && !use_gguf) {
                launch_activation_quant({use_fp8, exp_len, hidden,
                    scratch.permuted_input, scratch.quant_act,
                    scratch.quant_scale, scratch.quant_scale_bytes,
                    scratch.expert_offsets, scratch.sf_offsets, exp_len,
                    n_experts, gpu_dev, stream,
                    use_fp8 ? nullptr : scratch.moe_input_scales});
            }
            routed_gemm_emit.routed_gemm(intermediate, hidden,
                use_gguf ? scratch.permuted_input : scratch.quant_act,
                scratch.activation_output, bp, sbp,
                scratch.quant_scale,
                use_fp8 ? nullptr
                        : static_cast<const float*>(scratch.nvfp4_alpha),
                gguf_gate_type);

            // -- Up --
            upload_bptrs(b_up, sb_up);
            if (!use_fp8 && !use_gguf && scratch.nvfp4_alpha) {
                compute::launch_gather_alphas_scaled(
                    static_cast<float*>(scratch.nvfp4_alpha), nullptr,
                    bp,
                    deps_.expert_cache->up_bytes() - 8,
                    deps_.expert_cache->up_bytes() - 4,
                    n_experts, stream);
            }
            routed_gemm_emit.routed_gemm(intermediate, hidden,
                use_gguf ? scratch.permuted_input : scratch.quant_act,
                scratch.expert_output, bp, sbp,
                scratch.quant_scale,
                use_fp8 ? nullptr
                        : static_cast<const float*>(scratch.nvfp4_alpha),
                gguf_up_type);

            // -- Down (meta + alphas) --
            upload_bptrs(b_down, sb_down);
            compute::launch_populate_gemm_meta(
                static_cast<int32_t*>(scratch.problem_sizes),
                use_fp8 ? nullptr : static_cast<int32_t*>(scratch.sf_offsets),
                static_cast<const int32_t*>(scratch.expert_offsets),
                hidden, intermediate, n_experts, stream);
            if (!use_fp8 && !use_gguf && scratch.nvfp4_alpha) {
                compute::launch_gather_alphas_scaled(
                    static_cast<float*>(scratch.nvfp4_alpha),
                    static_cast<float*>(scratch.moe_input_scales),
                    bp,
                    deps_.expert_cache->down_bytes() - 8,
                    deps_.expert_cache->down_bytes() - 4,
                    n_experts, stream);
            }

            // Step 4: SwiGLU (+ quant), per quant mode (mirrors single-shot).
            if (scratch.quant_act || use_gguf) {
                if (use_gguf) {
                    if (exp_len > 0) {
                        const size_t I_bytes =
                            static_cast<size_t>(intermediate) * 2;
                        gpu_dev->memcpy_2d_async(
                            scratch.gate_up_output, I_bytes * 2,
                            scratch.activation_output, I_bytes,
                            I_bytes, exp_len, stream);
                        gpu_dev->memcpy_2d_async(
                            static_cast<uint8_t*>(scratch.gate_up_output)
                                + I_bytes, I_bytes * 2,
                            scratch.expert_output, I_bytes,
                            I_bytes, exp_len, stream);
                    }
                    launch_swiglu(dev, scratch.activation_output,
                                  scratch.gate_up_output, exp_len,
                                  intermediate, stream,
                                  static_cast<float>(mc.swiglu_limit));
                } else if (!use_fp8) {
                    // V4-4b: fused kernel has no swiglu_limit — limit>0 +
                    // NVFP4 experts rejected at config validation.
                    launch_fused_swiglu_nvfp4_quant({
                        scratch.activation_output, scratch.expert_output,
                        scratch.quant_act, scratch.quant_scale,
                        scratch.quant_scale_bytes,
                        scratch.expert_offsets, scratch.sf_offsets,
                        exp_len, n_experts, intermediate,
                        scratch.moe_input_scales, gpu_dev, stream});
                } else {
                    if (exp_len > 0) {
                        const size_t I_bytes =
                            static_cast<size_t>(intermediate) * 2;
                        gpu_dev->memcpy_2d_async(
                            scratch.gate_up_output, I_bytes * 2,
                            scratch.activation_output, I_bytes,
                            I_bytes, exp_len, stream);
                        gpu_dev->memcpy_2d_async(
                            static_cast<uint8_t*>(scratch.gate_up_output)
                                + I_bytes, I_bytes * 2,
                            scratch.expert_output, I_bytes,
                            I_bytes, exp_len, stream);
                    }
                    launch_swiglu(dev, scratch.activation_output,
                                  scratch.gate_up_output, exp_len,
                                  intermediate, stream,
                                  static_cast<float>(mc.swiglu_limit));
                    launch_activation_quant({use_fp8, exp_len, intermediate,
                        scratch.activation_output, scratch.quant_act,
                        scratch.quant_scale, scratch.quant_scale_bytes,
                        scratch.expert_offsets, scratch.sf_offsets, exp_len,
                        n_experts, gpu_dev, stream, nullptr});
                }
            }

            // Wave-masked: the down GEMM writes only the wave's rows (masked
            // experts have zero-length groups), but the unpermute reads a dest
            // row for EVERY top-K slot via src_to_dest_map — zero the whole
            // chunk output first so masked rows contribute an exact +0
            // (bit-identical to the zero-weight GEMM this replaces). Runs
            // after Step 4 (which consumed expert_output as the up-GEMM
            // output) and before Step 5 overwrites it with the down output.
            if (wave_mask) {
                gpu_dev->memset_async(
                    scratch.expert_output, 0,
                    static_cast<size_t>(exp_len) * hidden * 2, stream);
            }

            // Step 5: Down grouped GEMM → expert_output.
            routed_gemm_emit.routed_gemm(hidden, intermediate,
                use_gguf ? scratch.activation_output : scratch.quant_act,
                scratch.expert_output, bp, sbp,
                scratch.quant_scale,
                use_fp8 ? nullptr
                        : static_cast<const float*>(scratch.nvfp4_alpha),
                gguf_down_type);

            // Step 6: Unpermute the chunk. Single-pass → straight into the
            // moe_output rows (bit-identical to single-shot per token, fixed
            // slot order). Wave passes → accumulate (INV-MOE-BIG-2).
            if (!wave_mode) {
                dev->moe_unpermute(
                    row_ptr(scratch.moe_output, off),
                    scratch.expert_output, chunk_topk_w,
                    static_cast<const int32_t*>(scratch.src_to_dest_map),
                    len, topk, hidden, /*elem_size_bytes=*/2, stream,
                    compute::MoeCombineMode::kReducedBf16);
            } else {
                dev->moe_unpermute(
                    scratch.big_unperm_tmp,
                    scratch.expert_output, chunk_topk_w,
                    static_cast<const int32_t*>(scratch.src_to_dest_map),
                    len, topk, hidden, /*elem_size_bytes=*/2, stream,
                    compute::MoeCombineMode::kReducedBf16);
                compute::launch_residual_add(
                    row_ptr(scratch.moe_output, off), scratch.big_unperm_tmp,
                    len * hidden, stream);
            }
        }  // chunk loop
    }

    // Wave-partial pass ends here (no shared expert / allreduce / residual /
    // commit — those run exactly once, in the kFinal pass).
    if (mp.wave_pass == MoeWavePass::kPartial)
        return true;

    moe_valid = true;

    } else if (scratch.moe_output) {
        // router invalid on a non-real layer (MTP/out-of-range): zero
        // moe_output so Step 7d doesn't read stale data.
        gpu_dev->memset_async(
            scratch.moe_output, 0,
            static_cast<size_t>(num_tokens) * hidden * 2, stream);
    }
    } // end if (!skip_routed)

    // kFinal with everything already accumulated: moe_output holds the routed
    // result across all waves.
    if (mp.wave_pass == MoeWavePass::kFinal)
        moe_valid = true;

    // ── Step 7: Shared expert FFN (chunked) ────────────────────────────────
    if (mc.n_shared_experts > 0 && hidden_input
        && scratch.shared_gate_up_output && scratch.shared_expert_output) {
        const Deps::SharedExpertWeights* se = nullptr;
        if (mp.layer_idx < deps_.shared_expert_weight_ptrs.size() &&
            gpu < deps_.shared_expert_weight_ptrs[mp.layer_idx].size()) {
            se = &deps_.shared_expert_weight_ptrs[mp.layer_idx][gpu];
        }
        if (se && se->gate_up && se->down) {
            const int tp =
                std::max(1, deps_.live_config->parallelism.tensor_parallelism);
            const int intermediate_local = intermediate / tp;

            // SC (superchunk port): RAW BF16 shared expert inside a GGUF
            // checkpoint (DeepSeek-V4 shexp is BF16-native) — the chunked
            // BF16 GEMM route (the ticket-H single-shot fix, per chunk).
            // Never decode BF16 bytes as k-quant blocks; mixed raw/packed
            // projections stay fail-loud.
            const bool se_raw_bf16 = use_gguf && !se->gate_is_gguf;
            if (use_gguf
                && (se->gate_is_gguf != se->up_is_gguf
                    || se->gate_is_gguf != se->down_is_gguf)) {
                spdlog::critical("dispatch_moe_big: shared expert mixes "
                                 "raw-BF16 and k-quant projections "
                                 "(layer {})", mp.layer_idx);
                std::abort();
            }

            for (int off = 0; off < num_tokens; off += chunk) {
                const int len = std::min(chunk, num_tokens - off);
                const void* a_in = row_ptr(norm_input, off);

                if (se_raw_bf16) {
                    // gate_up: [len, 2*I_local] = a_in [len, H] @ W[2I, H]^T
                    compute::launch_bf16_gemm_nt(
                        scratch.shared_gate_up_output, a_in, se->gate_up,
                        len, 2 * intermediate_local, hidden,
                        compute::GemmInDtype::kBFloat16,
                        compute::GemmAccOutDtype::kBFloat16, stream);
                    launch_swiglu(dev, scratch.shared_activation,
                                  scratch.shared_gate_up_output, len,
                                  intermediate_local, stream,
                                  static_cast<float>(mc.swiglu_limit));
                    // down: [len, H] into the chunk's output rows.
                    compute::launch_bf16_gemm_nt(
                        row_ptr(scratch.shared_expert_output, off),
                        scratch.shared_activation, se->down,
                        len, hidden, intermediate_local,
                        compute::GemmInDtype::kBFloat16,
                        compute::GemmAccOutDtype::kBFloat16, stream);
                    continue;
                }

                const int32_t offsets[2] = {0, len};
                gpu_dev->memcpy_h2d_async(scratch.shared_expert_offsets,
                                          offsets, 2 * sizeof(int32_t), stream);
                const int32_t shared_gu_ps[3] =
                    {len, 2 * intermediate_local, hidden};
                gpu_dev->memcpy_h2d_async(scratch.shared_problem_sizes,
                                          shared_gu_ps, 3 * sizeof(int32_t),
                                          stream);
                const int32_t shared_gu_sf[2] = {0, ((len + 127) / 128) * 128};
                gpu_dev->memcpy_h2d_async(scratch.shared_sf_offsets,
                                          shared_gu_sf, 2 * sizeof(int32_t),
                                          stream);
                if (!use_fp8 && scratch.nvfp4_alpha) {
                    gpu_dev->memcpy_h2d_async(scratch.nvfp4_alpha,
                                              &se->alpha, sizeof(float), stream);
                    gpu_dev->memcpy_h2d_async(scratch.moe_input_scales,
                                              &se->input_scale, sizeof(float),
                                              stream);
                }
                if (scratch.quant_act && !use_gguf) {
                    launch_activation_quant({use_fp8, len, hidden,
                        a_in, scratch.quant_act, scratch.quant_scale,
                        scratch.quant_scale_bytes, scratch.shared_expert_offsets,
                        scratch.shared_sf_offsets, len, 1, gpu_dev, stream,
                        use_fp8 ? nullptr : scratch.moe_input_scales});
                }
                // 7a: gate+up GEMM.
                if (use_gguf) {
                    GgufDenseGateUpArgs gu{};
                    gu.num_tokens = len;
                    gu.intermediate_local = intermediate_local;
                    gu.hidden = hidden;
                    gu.a_base = a_in;
                    gu.gate_up_output = scratch.shared_gate_up_output;
                    gu.gate_scratch = scratch.shared_activation;
                    gu.up_scratch = scratch.gguf_gate_up_split;
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
                    GroupedGemmArgs gargs{use_fp8, 1, 2 * intermediate_local,
                        hidden, scratch.quant_act,
                        se->gate_up, scratch.shared_gate_up_output,
                        scratch.quant_scale, se->gate_up_scales,
                        static_cast<const float*>(scratch.nvfp4_alpha),
                        static_cast<const int32_t*>(scratch.shared_expert_offsets),
                        static_cast<const int32_t*>(scratch.shared_sf_offsets),
                        static_cast<const int32_t*>(scratch.shared_problem_sizes),
                        dev, scratch.gemm_workspace,
                        scratch.gemm_workspace_bytes, stream};
                    launch_grouped_gemm(gargs);
                }
                // 7b: SwiGLU. (V4-4b: clamp applies to shared experts too.)
                launch_swiglu(dev, scratch.shared_activation,
                              scratch.shared_gate_up_output, len,
                              intermediate_local, stream,
                              static_cast<float>(mc.swiglu_limit));
                // 7c: down GEMM → shared_expert_output rows.
                const int32_t shared_dn_ps[3] =
                    {len, hidden, intermediate_local};
                gpu_dev->memcpy_h2d_async(scratch.shared_problem_sizes,
                                          shared_dn_ps, 3 * sizeof(int32_t),
                                          stream);
                if (!use_fp8 && scratch.nvfp4_alpha) {
                    gpu_dev->memcpy_h2d_async(scratch.nvfp4_alpha,
                                              &se->alpha_down, sizeof(float),
                                              stream);
                    gpu_dev->memcpy_h2d_async(scratch.moe_input_scales,
                                              &se->input_scale_down,
                                              sizeof(float), stream);
                }
                if (scratch.quant_act && !use_gguf) {
                    launch_activation_quant({use_fp8, len, intermediate_local,
                        scratch.shared_activation, scratch.quant_act,
                        scratch.quant_scale, scratch.quant_scale_bytes,
                        scratch.shared_expert_offsets, scratch.shared_sf_offsets,
                        len, 1, gpu_dev, stream,
                        use_fp8 ? nullptr : scratch.moe_input_scales});
                }
                if (use_gguf) {
                    gpu_dev->memcpy_h2d_async(scratch.gguf_single_b_ptr,
                                              &se->down, sizeof(void*), stream);
                }
                {
                    GroupedGemmArgs gargs{use_fp8, 1, hidden, intermediate_local,
                        use_gguf ? scratch.shared_activation : scratch.quant_act,
                        se->down, row_ptr(scratch.shared_expert_output, off),
                        scratch.quant_scale, se->down_scales,
                        static_cast<const float*>(scratch.nvfp4_alpha),
                        static_cast<const int32_t*>(scratch.shared_expert_offsets),
                        static_cast<const int32_t*>(scratch.shared_sf_offsets),
                        static_cast<const int32_t*>(scratch.shared_problem_sizes),
                        dev, scratch.gemm_workspace,
                        scratch.gemm_workspace_bytes, stream};
                    if (use_gguf) {
                        gargs.use_gguf = true;
                        gargs.gguf_type = to_gguf_compute(se->down_gguf_type);
                        gargs.gguf_strategy = gguf_strategy;
                        gargs.gguf_total_tokens = len;
                        gargs.B_ptrs =
                            static_cast<const void**>(scratch.gguf_single_b_ptr);
                    }
                    launch_grouped_gemm(gargs);
                }
            }  // shared chunk loop

            if (mp.phase == MoeDispatchPhase::kPreAllreduce)
                return true;

            // 7d: add shared expert output to routed MoE output.
            if (scratch.moe_output && scratch.shared_expert_output) {
                compute::launch_residual_add(
                    scratch.moe_output, scratch.shared_expert_output,
                    num_tokens * hidden, stream);
                moe_valid = true;
            }
        }
    }

    if (mp.phase == MoeDispatchPhase::kPreAllreduce)
        return true;

chunked_post_allreduce:
    if (mp.phase == MoeDispatchPhase::kPostAllreduce) {
        if (scratch.moe_output && scratch.shared_expert_output) {
            compute::launch_residual_add(
                scratch.moe_output, scratch.shared_expert_output,
                num_tokens * hidden, stream);
            moe_valid = true;
        }
    }

    // Step 8: residual add — h += moe_output.
    // V4-5b mHC: the residual update is hc_post (extras skip) —
    // ArchDeepseekV4Moe::residual_update; base arch: plain residual add.
    if (hidden_input && moe_valid && scratch.moe_output) {
        arch.residual_update(gpu, hidden_input, scratch.moe_output,
                             num_tokens, hidden, pair_idx, stream);
    }

    // Commit MoE output back to the attention buffer for the next layer.
    if (pair_idx >= 0) {
        deps_.hidden_state_pairs[pair_idx].commit(
            static_cast<size_t>(num_tokens) * hidden * deps_.hc_streams * 2,
            stream,
            deps_.device_backends[gpu]);
        if (deps_.hidden_state_pairs[pair_idx].moe_attn_event) {
            deps_.stream_manager->record_event(
                deps_.hidden_state_pairs[pair_idx].moe_attn_event,
                static_cast<int>(gpu), compute::StreamId::kExpertFfn);
        }
    }

    return true;
}

// ── E_CMD_FETCH_AND_RUN_MOE_BIG handler ─────────────────────────────────────

void CommandDispatcher::handle_fetch_and_run_moe_big(const ipc::Command& cmd) {
    handle_fetch_and_run_moe_impl(cmd, /*big=*/true);
}

}  // namespace layerstorm::daemon
