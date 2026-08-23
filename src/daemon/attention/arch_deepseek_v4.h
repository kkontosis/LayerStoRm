// DeepSeek-V4 attention arch — dispatcher-side phase hooks.
// See arch_base.h for the phase contract. The executor-side V4 pipeline
// (DcpExecutor::execute_attention_v4 + row/chunk drivers + grouped o_proj +
// fork/free/spec-guard) lives in arch_deepseek_v4.cpp alongside these hook
// bodies.

#pragma once

#include "daemon/attention/arch_base.h"

namespace layerstorm::daemon {

class ArchDeepseekV4 final : public AttentionArch {
public:
    explicit ArchDeepseekV4(CommandDispatcher& d) : AttentionArch(d) {}

    /// V4 shape gate: B==1 decode-shaped steps, single-sequence chunked
    /// prefill / superchunk sub-launches (chunk_len == num_seqs), dspark
    /// verify chunks; drafts / multi-sequence shapes fail closed. Sets the
    /// chunk-shape batch cap (kMaxBatchDescriptors) vs max_batch_size.
    bool validate_shape(const CommandDispatcher::InternalAttentionParams& p,
                        int& batch_cap) override;

    /// V4 staging: the chunk-descriptor copy only (a B==1 is_prefill step is
    /// decode-shaped for V4 — never synthesize a chunk descriptor; the DSA
    /// indexer machinery is MLA-only, V4 manages its own LID tier).
    bool stage_step(const CommandDispatcher::InternalAttentionParams& p,
                    parallelism::AttentionExecParams& params,
                    int batch_size, int layer, int dcp_size,
                    bool kv_meta_ok) override;

    /// V4 execution: verify-chunk row validation, side-tier provisioning
    /// (ensure_v4_tier_pages), V4Step per-rank metadata assembly, the
    /// V4 KV-tiering hook arm, DcpExecutor::execute_attention_v4, and the
    /// post-attention V4 KV-tiering demote sweep.
    bool execute(const CommandDispatcher::InternalAttentionParams& p,
                 parallelism::AttentionExecParams& params,
                 int batch_size, int layer, int dcp_size,
                 bool kv_meta_ok) override;
};

}  // namespace layerstorm::daemon
