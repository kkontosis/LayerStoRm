// MLA (DeepSeek-V3.2 / GLM) attention arch — dispatcher-side phase hooks.
// See arch_base.h for the phase contract. The executor-side MLA pipeline
// (DcpExecutor::execute_attention + common prefix + DSA producers + oproj)
// lives in arch_mla.cpp alongside these hook bodies.

#pragma once

#include "daemon/attention/arch_base.h"

namespace layerstorm::daemon {

class ArchMla final : public AttentionArch {
public:
    explicit ArchMla(CommandDispatcher& d) : AttentionArch(d) {}

    /// MLA has no arch shape restriction — batch cap is the plain
    /// max_batch_size bound.
    bool validate_shape(const CommandDispatcher::InternalAttentionParams& p,
                        int& batch_cap) override;

    /// DSA indexer coverage state machine + paged indexer-K provisioning
    /// (decode + prefill-append branches), the chunk-descriptor synthesis
    /// for prefill shapes, and the GLM KV-tiering begin_layer/repromote
    /// gates (P2 move — kv_tiering_ is provably MLA-only).
    bool stage_step(const CommandDispatcher::InternalAttentionParams& p,
                    parallelism::AttentionExecParams& params,
                    int batch_size, int layer, int dcp_size,
                    bool kv_meta_ok) override;

    /// DcpExecutor::execute_attention (steps 1-14 of the MLA pipeline) +
    /// the GLM KV-tiering post-attention demote sweep (P2 move).
    bool execute(const CommandDispatcher::InternalAttentionParams& p,
                 parallelism::AttentionExecParams& params,
                 int batch_size, int layer, int dcp_size,
                 bool kv_meta_ok) override;

private:
    // Per-step GLM KV-tiering scratch (set by stage_step, consumed by
    // execute's post-attention sweep; single daemon thread — no reentry).
    uint64_t tier_seq_ = 0;
    uint32_t tier_pos_ = 0;
    bool tier_step_ = false;
    /// TD-KVT-ADMISSION-UPFRONT: rows of the current tier step (1 = decode /
    /// B==1 sparse prefill chunk; >1 = blessed chunk cohort — after_attention
    /// then demotes behind the LAST row's position).
    int tier_rows_ = 1;
};

}  // namespace layerstorm::daemon
