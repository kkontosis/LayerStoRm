// MoE arch base class — THE single base class for per-model MoE paths
// (TD-MOE-BY-MODEL-SPLIT, attention-refactor-V2 template).
//
// The shared driver (CommandDispatcher::dispatch_moe_internal in
// moe_driver.cpp) owns every arch-agnostic phase:
//   guards → attn→moe event wait → norm → CPU-prime → quant routes →
//   dense early-out → resident bitset → gating intake (precomputed consume /
//   router projection) → seam publish → miss probe → routed Step 2..6
//   (MoeGemmEmitter, FFN graph) → shexp quantized route → allreduce phases →
//   residual/commit. The TP/EP orchestration (moe_ranks.cpp), the progressive
//   FETCH_AND_RUN machinery (moe_progressive.cpp) and the chunked big-batch
//   sibling (moe_big.cpp) are arch-agnostic as well.
// A model arch participates ONLY through the hooks below; all model-special
// MoE code lives behind them (INV-MOE-ARCH):
//   - ArchMlaMoe       (arch_mla_moe.{h,cpp})       — GLM / DeepSeek-V3.2
//     family. The family's MoE behavior IS the common base path (learned
//     top-K self-gating, plain residual add, quantized shexp), so the base
//     default hook bodies live in arch_mla_moe.cpp as the family's named home.
//   - ArchDeepseekV4Moe (arch_deepseek_v4_moe.{h,cpp}) — V4 quirks: V4-4
//     hash-layer gating, V4-5b mHC hc_pre/hc_post residual stream mix,
//     V4-7b raw-BF16 shared-expert route.
//
// LOSSLESSNESS CONTRACT (middle-ground rule): hook bodies are the verbatim
// blocks carved out of the former dispatch_moe.cpp driver AND KEEP their
// original data-gated conditions (hc_streams > 1, layer < moe_hash_layers,
// use_gguf && !gate_is_gguf) inside the override — the V4 overrides fall
// back to the base body when the condition does not hold, so behavior is
// byte-identical under ANY config even where arch selection (the attention
// driver's is_v4 condition) and the data gate could hypothetically disagree.
// The arch classes are stateless facades over CommandDispatcher (friend
// access, `d_`).

#pragma once

#include "daemon/command_dispatcher.h"

namespace layerstorm::daemon {

class MoeArch {
public:
    virtual ~MoeArch() = default;

    /// Norm-phase hook: collapse the residual stream(s) to the module input
    /// before the pre-MoE RMSNorm. Base: identity (rms_src stays
    /// hidden_input). V4: V4-5b mHC hc_pre (emits the hc_post/hc_comb
    /// coefficients the residual_update hc_post consumes) when
    /// norm_w && hc_streams > 1. Returns false on missing mHC weights.
    virtual bool collapse_hidden(
        const CommandDispatcher::InternalMoeParams& mp, uint32_t gpu,
        const parallelism::AttentionLayerWeights* lw, const void* norm_w,
        void* hidden_input, int num_tokens, int hidden, void* stream,
        const void*& rms_src) {
        (void)mp; (void)gpu; (void)lw; (void)norm_w; (void)hidden_input;
        (void)num_tokens; (void)hidden; (void)stream; (void)rms_src;
        return true;
    }

    /// Step-1 hook: expert selection — router_logits → topk_weights /
    /// topk_indices in moe_scratch_[gpu]. Base: learned top-K gating
    /// (V4-4a scoring_func already parametrizes sigmoid vs sqrtsoftplus).
    /// V4: V4-4 hash-layer gating (tid2eid) when layer < moe_hash_layers,
    /// else the base body. Returns false on a missing hash table.
    virtual bool select_experts(
        const CommandDispatcher::InternalMoeParams& mp, uint32_t gpu,
        int num_tokens, int topk, int n_experts, bool router_valid,
        void* stream);

    /// Step-7 hook: the V4-7b raw-BF16 shared-expert route. Returns true
    /// when the arch handled the shared expert (caller skips the quantized
    /// route); `return_early` mirrors the kPreAllreduce early-return of the
    /// carved block. Base: not handled (quantized route runs).
    virtual bool try_shexp_raw_bf16(
        const CommandDispatcher::InternalMoeParams& mp, uint32_t gpu,
        const CommandDispatcher::Deps::SharedExpertWeights* se, bool use_gguf,
        const void* norm_input, int num_tokens, int hidden,
        int intermediate_local, bool& moe_valid, bool& return_early,
        void* stream) {
        (void)mp; (void)gpu; (void)se; (void)use_gguf; (void)norm_input;
        (void)num_tokens; (void)hidden; (void)intermediate_local;
        (void)moe_valid; (void)return_early; (void)stream;
        return false;
    }

    /// Residual-phase hook (dense early-out and Step 8): fold `add_src`
    /// into the hidden state. Base: plain residual add. V4: V4-5b mHC
    /// hc_post stream mix when hc_streams > 1 (EP-xTP extras, pair_idx < 0,
    /// skip it — their local result is never committed).
    virtual void residual_update(uint32_t gpu, void* hidden_input,
                                 void* add_src, int num_tokens, int hidden,
                                 int pair_idx, void* stream);

    MoeArch(const MoeArch&) = delete;
    MoeArch& operator=(const MoeArch&) = delete;

protected:
    explicit MoeArch(CommandDispatcher& d) : d_(d) {}
    CommandDispatcher& d_;
};

}  // namespace layerstorm::daemon
