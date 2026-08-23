// ArchDeepseekV4Moe TU — the DeepSeek-V4 MoE quirks (hash gating, mHC
// residual mix, raw-BF16 shexp). Hook bodies are verbatim carve-outs from
// the former dispatch_moe.cpp driver (only `d_.` prefixes added); each
// keeps its original data-gated condition and falls back to the base body
// when it does not hold (INV-MOE-ARCH losslessness contract).

#include "daemon/moe/arch_deepseek_v4_moe.h"
#include "daemon/moe/moe_internal.h"
#include "daemon/dispatch_detail.h"

#include <cstdlib>

#include <spdlog/spdlog.h>

#include "compute/kernels/elementwise/residual_add.h"
#include "compute/kernels/mhc/mhc.h"
#include "compute/kernels/moe/hash_gating.h"  // V4-4 hash-layer gating
#include "compute/kernels/sm120/gemm/bf16_gemm.h"  // V4-7b raw-BF16 shexp

namespace layerstorm::daemon {

// V4-5b mHC: hc_pre collapses the hc-stream residual to the module input
// (emitting the post/comb coefficients the Step-8 hc_post consumes); the
// pre-MoE norm then reads the collapsed x.
bool ArchDeepseekV4Moe::collapse_hidden(
    const CommandDispatcher::InternalMoeParams& mp, uint32_t gpu,
    const parallelism::AttentionLayerWeights* lw, const void* norm_w,
    void* hidden_input, int num_tokens, int hidden, void* stream,
    const void*& rms_src) {
    const auto& scratch = d_.moe_scratch_[gpu];
    const int layer = static_cast<int>(mp.layer_idx);
    if (norm_w && d_.deps_.hc_streams > 1) {
        if (!lw->hc_ffn_fn || !lw->hc_ffn_base || !lw->hc_ffn_scale ||
            !scratch.hc_x || !scratch.hc_post || !scratch.hc_comb) {
            spdlog::error("dispatch_moe: mHC active but hc_ffn weights/"
                          "scratch missing (layer {})", layer);
            return false;
        }
        compute::launch_mhc_pre(
            scratch.hc_x, scratch.hc_post, scratch.hc_comb, hidden_input,
            lw->hc_ffn_fn, lw->hc_ffn_scale, lw->hc_ffn_base,
            d_.deps_.live_config->model.rms_norm_eps,
            d_.deps_.live_config->model.hc_eps, 2.0f,
            d_.deps_.live_config->model.hc_sinkhorn_iters,
            num_tokens, d_.deps_.hc_streams, hidden, stream);
        rms_src = scratch.hc_x;
    }
    return true;
}

// Step 1, V4-4 hash layers (layer_idx < num_hash_layers): expert ids come
// from tid2eid[token_id] (RUN_MOE rows are [0, num_tokens) — no row offset),
// weights from the router logits restricted to those experts; the
// exp_probs_b bias never applies. Non-hash layers fall back to the base
// learned top-K body.
bool ArchDeepseekV4Moe::select_experts(
    const CommandDispatcher::InternalMoeParams& mp, uint32_t gpu,
    int num_tokens, int topk, int n_experts, bool router_valid,
    void* stream) {
    const auto& mc = d_.deps_.live_config->model;
    const auto& scratch = d_.moe_scratch_[gpu];
    if (router_valid && d_.deps_.cuda_kernels_enabled
        && static_cast<int>(mp.layer_idx) < d_.deps_.moe_hash_layers) {
        const int32_t* table = nullptr;
        if (mp.layer_idx < d_.deps_.hash_gating_table_ptrs.size()
            && gpu < d_.deps_.hash_gating_table_ptrs[mp.layer_idx].size())
            table = d_.deps_.hash_gating_table_ptrs[mp.layer_idx][gpu];
        if (!table || !scratch.moe_token_ids) {
            spdlog::error("dispatch_moe: hash layer {} on gpu {} missing {}",
                          mp.layer_idx, gpu,
                          table ? "token-id scratch" : "tid2eid table");
            return false;
        }
        compute::HashGatingParams hp{};
        hp.num_tokens            = num_tokens;
        hp.num_experts           = n_experts;
        hp.topk                  = topk;
        hp.vocab_size            = mc.vocab_size;
        hp.routed_scaling_factor = static_cast<float>(mc.routed_scaling_factor);
        hp.renormalize           = mc.norm_topk_prob;
        hp.scoring_func          = to_scoring_func(mc.gating_score_fn);
        compute::launch_hash_gating(
            static_cast<float*>(scratch.topk_weights),
            static_cast<int32_t*>(scratch.topk_indices),
            static_cast<const float*>(scratch.router_logits),
            table, static_cast<const int32_t*>(scratch.moe_token_ids),
            hp, stream);

        drift_dump_routing(d_.deps_.device_backends[gpu], stream,
                           scratch.router_logits, scratch.topk_indices,
                           scratch.topk_weights, num_tokens, n_experts, topk,
                           mp.layer_idx, static_cast<int>(gpu));
        return true;
    }
    return MoeArch::select_experts(mp, gpu, num_tokens, topk, n_experts,
                                   router_valid, stream);
}

// V4-7b (ticket H): RAW BF16 shared expert inside a GGUF checkpoint
// (DeepSeek-V4 shexp is BF16-native). Runs the plain BF16 GEMM route — the
// packed-block decoder would read BF16 bytes as k-quant blocks (1e6-scale
// garbage), and the FP8/NVFP4 route would treat them as quantized+scales.
// Mixed raw/packed projections are unsupported — fail loud.
bool ArchDeepseekV4Moe::try_shexp_raw_bf16(
    const CommandDispatcher::InternalMoeParams& mp, uint32_t gpu,
    const CommandDispatcher::Deps::SharedExpertWeights* se, bool use_gguf,
    const void* norm_input, int num_tokens, int hidden,
    int intermediate_local, bool& moe_valid, bool& return_early,
    void* stream) {
    const auto& mc = d_.deps_.live_config->model;
    const auto& scratch = d_.moe_scratch_[gpu];
    auto* dev = d_.expert_dev(gpu);
    const bool se_raw_bf16 = use_gguf && !se->gate_is_gguf;
    if (use_gguf
        && (se->gate_is_gguf != se->up_is_gguf
            || se->gate_is_gguf != se->down_is_gguf)) {
        spdlog::critical("dispatch_moe: shared expert mixes raw-BF16 "
                         "and k-quant projections (layer {})",
                         mp.layer_idx);
        std::abort();
    }
    if (!se_raw_bf16)
        return false;
    // gate_up: [B, 2*I_local] = norm_input [B, H] @ W[2I, H]^T
    compute::launch_bf16_gemm_nt(
        scratch.shared_gate_up_output, norm_input, se->gate_up,
        num_tokens, 2 * intermediate_local, hidden,
        compute::GemmInDtype::kBFloat16,
        compute::GemmAccOutDtype::kBFloat16, stream);
    launch_swiglu(dev, scratch.shared_activation,
                  scratch.shared_gate_up_output, num_tokens,
                  intermediate_local, stream,
                  static_cast<float>(mc.swiglu_limit));
    // down: [B, H] = shared_activation [B, I_local] @ W[H, I]^T
    compute::launch_bf16_gemm_nt(
        scratch.shared_expert_output, scratch.shared_activation,
        se->down, num_tokens, hidden, intermediate_local,
        compute::GemmInDtype::kBFloat16,
        compute::GemmAccOutDtype::kBFloat16, stream);
    if (mp.phase == CommandDispatcher::MoeDispatchPhase::kPreAllreduce) {
        return_early = true;
        return true;
    }
    if (scratch.moe_output && scratch.shared_expert_output) {
        compute::launch_residual_add(
            scratch.moe_output, scratch.shared_expert_output,
            num_tokens * hidden, stream);
        moe_valid = true;
    }
    return true;
}

// V4-5b mHC: the FFN residual update is hc_post (doubly-stochastic stream
// mix), not an add. EP-xTP extras (pair_idx < 0) skip it — their local
// result is never committed and they hold no residual streams. Non-mHC
// configs fall back to the base plain residual add.
void ArchDeepseekV4Moe::residual_update(uint32_t gpu, void* hidden_input,
                                        void* add_src, int num_tokens,
                                        int hidden, int pair_idx,
                                        void* stream) {
    if (d_.deps_.hc_streams > 1) {
        if (pair_idx >= 0) {
            const auto& scratch = d_.moe_scratch_[gpu];
            compute::launch_mhc_post(
                hidden_input, add_src, hidden_input,
                scratch.hc_post, scratch.hc_comb,
                num_tokens, d_.deps_.hc_streams, hidden, stream);
        }
    } else {
        MoeArch::residual_update(gpu, hidden_input, add_src, num_tokens,
                                 hidden, pair_idx, stream);
    }
}

}  // namespace layerstorm::daemon
