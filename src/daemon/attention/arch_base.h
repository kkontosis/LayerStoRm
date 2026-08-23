// Attention arch base class — THE single base class for per-model attention
// paths (attention refactor V2, campaign mission #1).
//
// The shared driver (CommandDispatcher::dispatch_attention_internal in
// attention_driver.cpp) owns every arch-agnostic phase:
//   guards → moe→attn event waits → seam/dspark taps → weights → KVT rewind
//   re-promotion → KV metadata build → mHC-pre / row-offset hidden setup →
//   AttentionExecParams fill → residual/mHC-post →
//   fused gating emit → routing export → events.
// A model arch participates ONLY through the three phase hooks below; all
// model-special attention code lives behind them (INV-ATTN-ARCH):
//   - ArchMla        (arch_mla.{h,cpp})         — DeepSeek-V3.2 / GLM MLA
//     geometry: DSA indexer coverage/provisioning staging, chunk-descriptor
//     synthesis, DcpExecutor::execute_attention.
//   - ArchDeepseekV4 (arch_deepseek_v4.{h,cpp}) — V4 shape gating, side-tier
//     provisioning, V4Step assembly, DcpExecutor::execute_attention_v4, the
//     V4 KV-tiering demote sweep.
//
// P1 contract: hook bodies are the verbatim blocks carved out of the former
// dispatch_attention.cpp driver — byte-identical behavior; the arch classes
// are stateless facades over CommandDispatcher (friend access, `d_`).

#pragma once

#include "daemon/command_dispatcher.h"

namespace layerstorm::parallelism {
struct AttentionExecParams;
}

namespace layerstorm::daemon {

class AttentionArch {
public:
    virtual ~AttentionArch() = default;

    /// Phase A (pre-weights / pre-kv-meta): arch shape legality gate.
    /// Sets `batch_cap` (the driver's batch_size bound for this shape).
    /// On failure: set d_.last_internal_error_* and return false.
    virtual bool validate_shape(
        const CommandDispatcher::InternalAttentionParams& p,
        int& batch_cap) = 0;

    /// Phase B (post-kv-meta, pre-execute): arch-owned step staging into
    /// `params` — MLA: DSA indexer coverage state machine + paged-indexer
    /// provisioning + chunk-descriptor synthesis + the GLM KV-tiering
    /// begin/repromote gates (kv_tiering_ is provably MLA-only: its
    /// construction is format-gated to kSnapMlaFp8/kTurboQuantMse4);
    /// V4: chunk-descriptor copy.
    /// On failure: set d_.last_internal_error_* and return false.
    virtual bool stage_step(
        const CommandDispatcher::InternalAttentionParams& p,
        parallelism::AttentionExecParams& params,
        int batch_size, int layer, int dcp_size, bool kv_meta_ok) = 0;

    /// Phase C: arch execution — per-step arch state provisioning + the
    /// executor pipeline call (+ any arch-owned post-attention sweep).
    /// On failure: set d_.last_internal_error_* and return false.
    virtual bool execute(
        const CommandDispatcher::InternalAttentionParams& p,
        parallelism::AttentionExecParams& params,
        int batch_size, int layer, int dcp_size, bool kv_meta_ok) = 0;

    AttentionArch(const AttentionArch&) = delete;
    AttentionArch& operator=(const AttentionArch&) = delete;

protected:
    explicit AttentionArch(CommandDispatcher& d) : d_(d) {}
    CommandDispatcher& d_;
};

}  // namespace layerstorm::daemon
