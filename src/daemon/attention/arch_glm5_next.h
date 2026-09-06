// glm5_next (GLM-5.3-Flash) attention arch — dispatcher-side phase hooks.
// See arch_base.h for the phase contract; PLAN.md Phase GF3 (GF3.9) for the
// integration plan; spec/GLM-5.3-FLASH-MODELINFO.md for the verified model
// geometry; spec/ATTENTION_BOOKLET.md §1/§1.2/§10 for how this arch bends
// the attention stack.
//
// GF3.9 — THE JOIN POINT: the architecture is a per-layer HYBRID
// (model.layer_types): 34 KDA linear-attention layers (Kimi Delta
// Attention, arXiv:2510.26692 — per-request recurrent state, NO KV cache,
// NO indexer) interleaved with 11 NoPE sparse-MLA layers (qk_rope_head_dim
// 0, KV latent = kv_lora_rank 512, IndexPool indexer with index_kpool 4).
// The MTP layer 45 is sparse MLA (non-mHC) and carries no recurrent state.
//
// Per-layer dispatch happens INSIDE these hooks (the driver selects ONE
// arch per boot — attention_driver.cpp active_attention_arch):
//   - SPARSE-MLA layers DELEGATE to the composed ArchMla hooks verbatim —
//     the GF3.4 NoPE geometry and GF3.5 IndexPool arms already live behind
//     the MLA machinery, keyed off config (INV-ATTN-ARCH: reuse, never
//     fork).
//   - KDA layers stage a KdaStep (state-slot addressing + the per-LAYER
//     frontier guard below) and execute through
//     DcpExecutor::execute_attention_kda (the parallelism half of
//     arch_glm5_next.cpp), skipping KV metadata consumption, the DSA
//     coverage machine, sparse/dense selection, and every tiering gate.
//     The driver still BUILDS KV metadata on every dispatch (all-layer
//     scratch, once per step behind its dirty guard) — deliberately: page
//     growth for the step's new token then fails at the step's FIRST
//     dispatched layer, BEFORE any KDA state mutation, which is what keeps
//     the orchestrator's pool-evict re-issue seam exact on a recurrent
//     state (INV-KDA-REWIND: re-applying a token to a mutate-in-place
//     state is never idempotent).
//
// THE KDA FRONTIER GUARD (INV-KDA-REWIND enforcement): every KDA launch
// must start EXACTLY at that layer's per-sequence state frontier
// (SequenceState::kda_next_pos[ordinal]); anything else — a retry-seam
// re-feed, a speculative rewind (GF3.11 owns anchor-and-replay), a
// mis-chunked prefill — REFUSES with kComputeValidation, never computes.
// Prefill launches must also cut on the 64-token grid (INV-KDA-CARRY:
// chunk-64 launch boundaries are the bitwise state-carry points).

#pragma once

#include "daemon/attention/arch_base.h"
#include "daemon/attention/arch_mla.h"
#include "parallelism/dcp_executor.h"

#include <memory>
#include <vector>

namespace layerstorm::daemon {

class ArchGlm5Next final : public AttentionArch {
public:
    explicit ArchGlm5Next(CommandDispatcher& d)
        : AttentionArch(d), mla_(d) {}

    /// TRUE — and in a PURER form than V4's rings (the GF3.2 capability
    /// answer, recorded here and in spec/ATTENTION_BOOKLET.md §1.1/§10):
    ///
    /// (a) The KDA recurrent state (per layer: heads x head_dim x head_dim
    ///     delta-rule matrix + the width-(kernel-1) conv ring) is mutated
    ///     IN PLACE every step and has NO position axis at all — there is
    ///     no slot = pos % capacity to even argue about: state@N is simply
    ///     unrecoverable once the sequence advances past N. V4's rings
    ///     destroy entries lazily (window overwrite); KDA destroys the
    ///     whole interior history algebraically at every step.
    /// (b) Consequently a truncating fork (CMD_SEQ_FORK prefix_len > 0)
    ///     must be REJECTED whole-sequence: the 11 sparse-MLA layers alone
    ///     COULD truncate (their KV/indexer stores are append-only per
    ///     position), but the 34 KDA layers cannot, and a fork is all-or-
    ///     nothing (INV-SEQ-FORK-TRUNC). Prefix reuse on this arch can
    ///     exist only where a STATE CHECKPOINT was taken (GF3.12).
    /// (c) The IndexPool TAIL (raw K + gate score of the in-progress pool,
    ///     ring-updated at pos % index_kpool on the sparse layers) is also
    ///     per-sequence mutable frontier state — but at any pool-ALIGNED
    ///     length it is empty, and every engine grid boundary is a
    ///     multiple of index_kpool, so it does not independently force
    ///     this answer; it must simply travel with the frontier across
    ///     freeze/hibernate/restore (INV-PREFIX-CACHE-3/-4).
    ///
    /// Exported as EngineInfo::seq_fork_truncatable = 0 — the orchestrator
    /// never offers mid-edge prefix reuse on this arch, with zero new
    /// gating code (the property IS the gate).
    bool lossy_position_indexed_state() const override { return true; }

    /// Phase A: arch shape legality. TP >= 2 with REPLICATED KV is
    /// executable since GF3.10 (KDA head-sharded per GF3.3, per-rank state
    /// slots per GF3.8, one o_proj allreduce per KDA layer — INV-KDA-TP).
    /// Refuses (kComputeValidation): sequence-SHARDED KV (a whole-sequence
    /// recurrent state has no token shard; also thrown at boot),
    /// decode-graph steps (no KDA graph arms exist; graphs are
    /// engine-dormant anyway), and draft steps on KDA layers (the MTP
    /// draft is sparse MLA — a KDA draft step is a wiring error).
    bool validate_shape(const CommandDispatcher::InternalAttentionParams& p,
                        int& batch_cap) override;

    /// Phase B: per-layer dispatch — sparse-MLA layers delegate to
    /// ArchMla::stage_step (DSA coverage machine + IndexPool provisioning
    /// + chunk synthesis; tiering gates are inert while kv_tiering_ is
    /// unconstructed — the GF3.9 default-OFF decision); KDA layers run the
    /// frontier guard and stage the KdaStep.
    bool stage_step(const CommandDispatcher::InternalAttentionParams& p,
                    parallelism::AttentionExecParams& params,
                    int batch_size, int layer, int dcp_size,
                    bool kv_meta_ok) override;

    /// Phase C: sparse-MLA layers delegate to ArchMla::execute
    /// (DcpExecutor::execute_attention); KDA layers call
    /// DcpExecutor::execute_attention_kda. Executor throws surface as
    /// kComputeValidation errors here (loud, never approximate).
    bool execute(const CommandDispatcher::InternalAttentionParams& p,
                 parallelism::AttentionExecParams& params,
                 int batch_size, int layer, int dcp_size,
                 bool kv_meta_ok) override;

private:
    /// Lazily built per-layer axis (model.layer_types) — the arch object is
    /// constructed before live_config is guaranteed wired, so the first
    /// hook call builds it. is_linear_[l] for hidden layers; the MTP layer
    /// (>= num_hidden_layers) is sparse MLA by ground truth (MODELINFO §5).
    bool ensure_layer_axis();
    bool refuse(const char* msg);

    ArchMla mla_;                       ///< the sparse-MLA delegate
    std::vector<uint8_t> is_linear_;    ///< [num_hidden_layers]
    std::vector<int> linear_ordinal_;   ///< [num_hidden_layers], -1 = MLA
    int num_linear_ = 0;
    bool axis_ready_ = false;

    // Per-step KdaStep staging (single daemon thread — no reentry).
    parallelism::AttentionExecParams::KdaStep kda_step_{};
    std::vector<parallelism::AttentionExecParams::KdaStepRank> kda_ranks_;
    std::vector<std::vector<int>> kda_slots_;   ///< [dcp][batch]
};

}  // namespace layerstorm::daemon
