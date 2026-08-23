// ArchDeepseekV4Moe — DeepSeek-V4 MoE arch participant: the V4 quirks
// behind the MoeArch hooks (INV-MOE-ARCH, see arch_base.h):
//   - V4-4  hash-layer gating (tid2eid) for layer < moe_hash_layers,
//     falling back to the base learned top-K otherwise;
//   - V4-5b mHC: hc_pre residual-stream collapse before the pre-MoE norm +
//     hc_post doubly-stochastic stream mix as the residual update;
//   - V4-7b raw-BF16 shared-expert route inside a GGUF checkpoint
//     (DeepSeek-V4 shexp is BF16-native).
// Every override KEEPS the original data-gated condition verbatim and
// delegates to the base body when it does not hold (losslessness contract).
// (The V4-4b swiglu clamp needs no hook — mc.swiglu_limit parametrizes the
// shared SwiGLU sites for both archs.)

#pragma once

#include "daemon/moe/arch_base.h"

namespace layerstorm::daemon {

class ArchDeepseekV4Moe final : public MoeArch {
public:
    explicit ArchDeepseekV4Moe(CommandDispatcher& d) : MoeArch(d) {}

    bool collapse_hidden(const CommandDispatcher::InternalMoeParams& mp,
                         uint32_t gpu,
                         const parallelism::AttentionLayerWeights* lw,
                         const void* norm_w, void* hidden_input,
                         int num_tokens, int hidden, void* stream,
                         const void*& rms_src) override;

    bool select_experts(const CommandDispatcher::InternalMoeParams& mp,
                        uint32_t gpu, int num_tokens, int topk, int n_experts,
                        bool router_valid, void* stream) override;

    bool try_shexp_raw_bf16(const CommandDispatcher::InternalMoeParams& mp,
                            uint32_t gpu,
                            const CommandDispatcher::Deps::SharedExpertWeights* se,
                            bool use_gguf, const void* norm_input,
                            int num_tokens, int hidden, int intermediate_local,
                            bool& moe_valid, bool& return_early,
                            void* stream) override;

    void residual_update(uint32_t gpu, void* hidden_input, void* add_src,
                         int num_tokens, int hidden, int pair_idx,
                         void* stream) override;
};

}  // namespace layerstorm::daemon
