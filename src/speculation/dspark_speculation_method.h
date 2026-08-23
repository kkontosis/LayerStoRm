// DSpark SpeculationMethod (DSP-5): the DSpark drafter behind the pluggable
// speculation seam — draft policy + greedy acceptance rule + per-sequence
// anchor/round state for the DFlash-backbone + Markov-head draft pipeline
// (DSP-3/DSP-4).
//
// Role split (mirrors MtpSpeculationMethod, see mtp_speculation_method.h):
//   - The METHOD owns the draft POLICY (γ clamp from speculation.dspark.*),
//     the ACCEPTANCE RULE (greedy longest-prefix match + bonus token —
//     lossless by construction, INV-DSPARK-LOSSLESS), per-seq round state
//     (on_accept / on_rollback / reset) and acceptance statistics.
//   - The ENGINE owns the draft EXECUTION.  One DSpark draft round = ONE
//     D_CMD_RUN_DSPARK_STEP: DFlash backbone forward over the whole γ block
//     off the target's ingested aux-hidden context KV (INV-DSPARK-ANCHOR)
//     + the chained sequential Markov head (INV-DSPARK-MARKOV), leaving
//     draft_tokens [γ] i32 device-resident AND readable host-side from the
//     sideband readback scratch when the completion fires.  The method
//     NEVER reimplements it.
//
// The two are bridged by a DsparkStepExecutor installed by the DRIVER of
// the draft loop (the DsparkLossless integration golden today; the Phase-12
// orchestrator drives D_CMD_RUN_DSPARK_STEP directly over the command ring
// — python/orchestrator/dspark_draft.py + loop/speculation.py).  One
// executor call = one whole γ-block proposal (contrast MTP's per-token
// chained executor: DFlash latency is ~independent of γ, the point of the
// design).  propose_draft() with no executor installed declines (count = 0,
// always legal) so an engine with speculation.method=dspark but no driver
// falls back to plain decode.
//
// Anchor/context-KV state (NO fork, per DSP-3): the draft's context KV is
// keyed by ABSOLUTE target position and appended automatically by the
// aux-hidden export on every non-draft target step; a rejected draft token
// is never fed to the target (sequential early-stop verification), so the
// context never contains junk and rejection needs no device-side rewind —
// re-drafting from the shorter accepted prefix is just the next
// D_CMD_RUN_DSPARK_STEP at the smaller anchor_pos (and any driver that DID
// feed-then-rewind is covered by capture_aux's in-place overwrite at
// start_pos <= ctx_len, INV-DSPARK-AUX).  The method keeps a host-side
// anchor mirror per sequence (advance on accept; hold on rollback) used for
// cross-checking the driver-visible token history — the history is
// authoritative, the mirror is telemetry/validation.
//
// CUDA-free (INV-GPU-1): pure host logic; the executor hides all device work.

#pragma once

#include "speculation/speculation_method.h"

#include <cstdint>
#include <functional>
#include <unordered_map>

namespace layerstorm::speculation {

/// One whole-γ-block DSpark draft request handed to the executor.  The
/// backbone consumes the runtime's ingested context KV for `seq_id` at
/// positions [0, anchor_pos) — implicit engine state kept current by the
/// aux-hidden export on every target step (INV-DSPARK-AUX).
struct DsparkStepRequest {
    uint64_t seq_id = 0;       ///< context identity (runtime-tracked)
    int32_t  anchor_token = -1;///< newest ACCEPTED token (unfed; query slot 0)
    uint32_t anchor_pos = 0;   ///< slot-0 position == accepted (fed) length
    int      num_query = 0;    ///< γ for this round (<= max_draft_len())
    void*    stream = nullptr; ///< DraftContext.stream passthrough (may be null)
};

/// One whole-γ-block DSpark draft result: the Markov-head-sampled ids,
/// plus — when speculation.dspark.confidence_enabled — the DSP-6 trained
/// per-position survival probabilities.
struct DsparkStepResult {
    /// Hard cap on γ (matches the runtime's 16-slot query staging;
    /// speculation.dspark.block_size is validated <= this at init).
    static constexpr int kMaxGamma = 16;
    int32_t token_ids[kMaxGamma] = {};
    int count = 0;             ///< tokens actually drafted (<= num_query)

    /// Raw (pre-STS-calibration) c_k ∈ (0,1), the TRAINED conditional
    /// survival probability of token k given all predecessors accepted
    /// (INV-DSPARK-CONF — cumprod-composable; NOT the output-head
    /// {top1_prob, entropy} heuristic).  Valid entries: confidence_count
    /// (== count when the executor read them back, 0 when the confidence
    /// head is disabled).
    float confidence[kMaxGamma] = {};
    int confidence_count = 0;
};

/// Runs ONE fused DSpark draft step (D_CMD_RUN_DSPARK_STEP) on the engine.
/// Returns false on failure (unarmed/invalid context, command error) — the
/// method then declines this round (out.count = 0; decode is unaffected).
using DsparkStepExecutor =
    std::function<bool(const DsparkStepRequest&, DsparkStepResult&)>;

/// Cumulative draft/accept counters (telemetry; see acceptance_rate()).
struct DsparkAcceptanceStats {
    uint64_t rounds = 0;           ///< verified draft rounds (on_accept calls)
    uint64_t tokens_proposed = 0;  ///< draft tokens proposed in verified rounds
    uint64_t tokens_accepted = 0;  ///< draft tokens accepted by verification
};

class DsparkSpeculationMethod final : public SpeculationMethod {
public:
    DsparkSpeculationMethod() = default;
    ~DsparkSpeculationMethod() override = default;

    // ── Identity / capabilities ─────────────────────────────────────────────
    config::SpeculationMethodType type() const override;  // ::dspark (.cpp)
    const char* name() const override { return "dspark"; }
    int max_draft_len() const override { return gamma_; }
    /// The draft consumes the target's aux hiddens through the AUTOMATIC
    /// export hook (INV-DSPARK-AUX) — never DraftContext.hidden_state.
    bool needs_hidden_state() const override { return false; }

    // ── Lifecycle ───────────────────────────────────────────────────────────
    /// Requires ctx.config with speculation.dspark self-consistent
    /// (validate_dspark already enforces 1 <= speculative_tokens <=
    /// block_size); fails closed on block_size > kMaxGamma.  The draft
    /// checkpoint / draft GPU are NOT validated here — DsparkRuntime::create
    /// (engine step 19e) is the fail-closed surface for those.
    void init(const SpeculationInitContext& ctx) override;
    void reset(uint64_t seq_id) override;

    // ── Per-step seams ──────────────────────────────────────────────────────
    /// One executor call proposing up to min(ctx.max_tokens, γ, capacity)
    /// tokens off the anchor = the LAST visible history token at position
    /// ctx.num_tokens - 1 (the driver convention: history = prompt + every
    /// committed token, newest not yet fed).  out.probs: filled with the
    /// DSP-6 trained raw per-position survival c_k when the executor read
    /// them back (confidence_enabled — INV-DSPARK-CONF); left untouched
    /// otherwise — consumers must treat absent probs as "no scores".
    void propose_draft(const DraftContext& ctx, DraftResult& out) override;

    /// Greedy acceptance rule: longest draft prefix equal to the target
    /// tokens; bonus = target at the first mismatch (or the extra target
    /// position on full acceptance).  Lossless by construction
    /// (INV-DSPARK-LOSSLESS).
    void verify(const VerifyContext& ctx, VerifyResult& out) override;

    /// Commit: accepted_count drafts + the engine-appended bonus are now
    /// part of the sequence — advance the anchor mirror by accepted+1 fed
    /// positions and fold the round into the acceptance statistics.
    void on_accept(uint64_t seq_id, int accepted_count) override;

    /// Aborted round: discard in-flight state, keep the anchor mirror (the
    /// shorter prefix stays committed; the context KV needs no device work
    /// — see file header).
    void on_rollback(uint64_t seq_id) override;

    // ── Concrete accessors (non-virtual; house pattern like MTP/TQ) ────────
    void set_step_executor(DsparkStepExecutor executor) {
        executor_ = std::move(executor);
    }
    bool has_step_executor() const { return static_cast<bool>(executor_); }
    const DsparkAcceptanceStats& acceptance_stats() const { return stats_; }
    /// Accepted / proposed over all verified rounds; 0 when nothing proposed.
    double acceptance_rate() const {
        return stats_.tokens_proposed > 0
            ? static_cast<double>(stats_.tokens_accepted)
                  / static_cast<double>(stats_.tokens_proposed)
            : 0.0;
    }
    int gamma() const { return gamma_; }
    int block_size() const { return block_size_; }
    /// Host-side anchor mirror: the fed-token count the method expects the
    /// sequence to be at (== the next round's anchor_pos), or -1 when the
    /// sequence has no accepted round yet.  Telemetry/validation only.
    int64_t expected_anchor_pos(uint64_t seq_id) const {
        auto it = anchor_.find(seq_id);
        return it != anchor_.end() ? it->second : -1;
    }

private:
    // Config-derived policy (init()).
    int gamma_ = 0;       ///< speculation.dspark.speculative_tokens
    int block_size_ = 0;  ///< speculation.dspark.block_size (γ ceiling)

    DsparkStepExecutor executor_;

    /// In-flight draft rounds: seq_id → {tokens proposed, anchor used}.
    struct InflightRound {
        int proposed = 0;
        uint32_t anchor_pos = 0;
    };
    std::unordered_map<uint64_t, InflightRound> inflight_;
    /// Committed anchor mirror (see expected_anchor_pos()).
    std::unordered_map<uint64_t, int64_t> anchor_;

    DsparkAcceptanceStats stats_;
    bool warned_no_executor_ = false;
    bool warned_anchor_mismatch_ = false;
};

}  // namespace layerstorm::speculation
