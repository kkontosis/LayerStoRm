// ArchMlaMoe TU — the GLM / DeepSeek-V3.2 (MLA) family's named home.
// The family's MoE behavior is the COMMON path, so the MoeArch base default
// hook bodies live here (verbatim carve-outs from the former
// dispatch_moe.cpp driver; only `d_.` prefixes added — INV-MOE-ARCH).

#include "daemon/moe/arch_mla_moe.h"
#include "daemon/moe/moe_internal.h"
#include "daemon/dispatch_detail.h"

#include <spdlog/spdlog.h>

#include "compute/kernels/elementwise/residual_add.h"
#include "sm120/gating/topk_gating.h"

namespace layerstorm::daemon {

// Step 1: TopK gating — router_logits → topk_weights, topk_indices.
// (The learned-gating branch of the former driver Step 1; the V4-4
// hash-layer branch is the ArchDeepseekV4Moe override, which falls back
// to this body for non-hash layers.)
bool MoeArch::select_experts(const CommandDispatcher::InternalMoeParams& mp,
                             uint32_t gpu, int num_tokens, int topk,
                             int n_experts, bool router_valid, void* stream) {
    const auto& mc = d_.deps_.live_config->model;
    const auto& scratch = d_.moe_scratch_[gpu];
    if (router_valid && d_.deps_.cuda_kernels_enabled) {
        compute::TopkGatingParams gp{};
        gp.num_tokens            = num_tokens;
        gp.num_experts           = n_experts;
        gp.topk                  = topk;
        gp.n_group               = mc.n_group;
        gp.topk_group            = mc.topk_group;
        gp.routed_scaling_factor = static_cast<float>(mc.routed_scaling_factor);
        gp.renormalize           = mc.norm_topk_prob;
        // V4-4a: sigmoid (V3.2/GLM) vs sqrtsoftplus (V4) scoring.
        gp.scoring_func          = to_scoring_func(mc.gating_score_fn);

        const float* bias = nullptr;
        if (mp.layer_idx < d_.deps_.gating_bias_ptrs.size()
            && gpu < d_.deps_.gating_bias_ptrs[mp.layer_idx].size()) {
            bias = d_.deps_.gating_bias_ptrs[mp.layer_idx][gpu];
        }

        compute::launch_topk_gating(
            static_cast<float*>(scratch.topk_weights),
            static_cast<int32_t*>(scratch.topk_indices),
            static_cast<const float*>(scratch.router_logits),
            bias, gp, stream);

        // TD-DRIFT-ROOTCAUSE: gated logit-level routing dump (off by default).
        drift_dump_routing(d_.deps_.device_backends[gpu], stream,
                           scratch.router_logits, scratch.topk_indices,
                           scratch.topk_weights, num_tokens, n_experts, topk,
                           mp.layer_idx, static_cast<int>(gpu));
    }
    return true;
}

// Residual fold (dense early-out + Step 8): plain residual add. The V4-5b
// mHC hc_post stream mix is the ArchDeepseekV4Moe override.
void MoeArch::residual_update(uint32_t gpu, void* hidden_input, void* add_src,
                              int num_tokens, int hidden, int pair_idx,
                              void* stream) {
    (void)gpu; (void)pair_idx;
    compute::launch_residual_add(hidden_input, add_src,
                                 num_tokens * hidden, stream);
}

}  // namespace layerstorm::daemon
