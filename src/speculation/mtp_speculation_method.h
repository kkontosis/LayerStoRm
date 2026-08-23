// MTP SpeculationMethod (#16 / GLM-25g): self-draft off the frozen trunk via
// the model's Multi-Token-Prediction head (DeepSeek V3 §2.2 / GLM-5.2 blk.78,
// num_nextn_predict_layers modules).
//
// Role split (see spec/SPECULATION_SUBSYSTEM.md):
//   - The METHOD owns the draft POLICY (depth clamp from speculation.mtp.*,
//     dynamic-depth confidence early-exit), the ACCEPTANCE RULE (greedy
//     longest-prefix match + bonus token — lossless by construction), per-seq
//     draft state (on_accept / on_rollback / reset) and acceptance statistics.
//   - The ENGINE owns the MTP forward EXECUTION.  One MTP draft step =
//     MTP projection (enorm(Emb(t)) ‖ hnorm(prev_hidden) → eh_proj) → the MTP
//     transformer layer (attention + routed MoE on the production
//     FETCH_AND_RUN seam) → shared_head norm + LM head → argmax + top-1 prob.
//     That pipeline lives in CommandDispatcher (D_CMD_MTP_PROJECT +
//     D_B_CMD_RUN_ATTENTION + E_CMD_FETCH_AND_RUN_MOE + CMD_OUTPUT_HEAD, or
//     the fused D_CMD_RUN_MTP_STEP) — the method NEVER reimplements it.
//
// The two are bridged by an MtpStepExecutor installed by the DRIVER of the
// draft loop (integration golden today; the Phase-12 orchestrator drives the
// same engine pipeline directly over the command ring — see
// TD-MTP-ENGINE-PROPOSE in spec/TECH_DEBT.md).  propose_draft() with no
// executor installed declines (count = 0, always legal) so an engine with
// speculation.method=mtp but no driver falls back to plain decode.
//
// CUDA-free (INV-GPU-1): pure host logic; the executor hides all device work.

#pragma once

#include "speculation/speculation_method.h"

#include <cstdint>
#include <functional>
#include <unordered_map>

namespace layerstorm::speculation {

/// One MTP draft-step request handed to the executor.  The trunk-hidden /
/// recurrence anchor is implicit engine state (the hidden-state pair buffer
/// left by the last main-model or MTP-layer forward) — see the file header.
struct MtpStepRequest {
    uint64_t seq_id = 0;        ///< sequence the draft KV rides on
    int32_t  input_token = -1;  ///< token to embed (last accepted / prev draft)
    int      step_idx = 0;      ///< 0-based chain index within this draft round
    int      mtp_layer_idx = 0; ///< absolute layer index (num_hidden_layers + k)
    void*    stream = nullptr;  ///< DraftContext.stream passthrough (may be null)
};

/// One MTP draft-step result.
struct MtpStepResult {
    int32_t token_id = -1;      ///< argmax draft token
    float   confidence = 0.0f;  ///< top-1 probability (dynamic-depth gate)
};

/// Runs ONE MTP draft step on the engine.  Returns false on failure — the
/// method then stops the chain and returns the tokens drafted so far.
using MtpStepExecutor =
    std::function<bool(const MtpStepRequest&, MtpStepResult&)>;

/// Cumulative draft/accept counters (telemetry; see acceptance_rate()).
struct MtpAcceptanceStats {
    uint64_t rounds = 0;           ///< verified draft rounds (on_accept calls)
    uint64_t tokens_proposed = 0;  ///< draft tokens proposed in verified rounds
    uint64_t tokens_accepted = 0;  ///< draft tokens accepted by verification
};

class MtpSpeculationMethod final : public SpeculationMethod {
public:
    MtpSpeculationMethod() = default;
    ~MtpSpeculationMethod() override = default;

    // ── Identity / capabilities ─────────────────────────────────────────────
    config::SpeculationMethodType type() const override;  // ::mtp (needs config_parser.h — .cpp)
    const char* name() const override { return "mtp"; }
    int max_draft_len() const override { return max_depth_; }
    bool needs_hidden_state() const override { return true; }

    // ── Lifecycle ───────────────────────────────────────────────────────────
    /// Requires ctx.config with num_nextn_predict_layers >= 1 and
    /// speculation.mtp.enabled.  When ctx.loaded_model is present it must
    /// carry the MTP block (fail closed on a checkpoint without MTP weights);
    /// unit tests may pass loaded_model = nullptr.
    void init(const SpeculationInitContext& ctx) override;
    void reset(uint64_t seq_id) override;

    // ── Per-step seams ──────────────────────────────────────────────────────
    /// Chains up to min(ctx.max_tokens, max_draft_len()) MTP steps through the
    /// installed executor.  Step 0 embeds the last visible token
    /// (ctx.tokens[ctx.num_tokens-1]); step k embeds the step-(k-1) draft.
    /// dynamic_depth: the chain stops early when a step's confidence falls
    /// below speculation.mtp.confidence_threshold (step 0 always runs).
    /// No executor / empty history → declines (out.count = 0).
    void propose_draft(const DraftContext& ctx, DraftResult& out) override;

    /// Greedy acceptance rule: longest draft prefix equal to the target
    /// tokens; bonus = target at the first mismatch (or the extra target
    /// position on full acceptance).  Lossless by construction.
    void verify(const VerifyContext& ctx, VerifyResult& out) override;

    void on_accept(uint64_t seq_id, int accepted_count) override;
    void on_rollback(uint64_t seq_id) override;

    // ── Concrete accessors (non-virtual; house pattern like TQ setters) ────
    void set_step_executor(MtpStepExecutor executor) {
        executor_ = std::move(executor);
    }
    bool has_step_executor() const { return static_cast<bool>(executor_); }
    const MtpAcceptanceStats& acceptance_stats() const { return stats_; }
    /// Accepted / proposed over all verified rounds; 0 when nothing proposed.
    double acceptance_rate() const {
        return stats_.tokens_proposed > 0
            ? static_cast<double>(stats_.tokens_accepted)
                  / static_cast<double>(stats_.tokens_proposed)
            : 0.0;
    }
    int num_mtp_layers() const { return num_mtp_layers_; }
    int mtp_layer_idx(int step_idx) const {
        return num_hidden_layers_ + (num_mtp_layers_ > 0
                                         ? step_idx % num_mtp_layers_ : 0);
    }

private:
    // Config-derived policy (init()).
    int    max_depth_ = 0;
    bool   dynamic_depth_ = true;
    double confidence_threshold_ = 0.4;
    int    num_hidden_layers_ = 0;
    int    num_mtp_layers_ = 0;

    MtpStepExecutor executor_;

    // In-flight draft rounds: seq_id → tokens proposed this round (set by
    // propose_draft, consumed by on_accept / on_rollback / reset).
    std::unordered_map<uint64_t, int> inflight_proposed_;

    MtpAcceptanceStats stats_;
    bool warned_no_executor_ = false;
};

}  // namespace layerstorm::speculation
