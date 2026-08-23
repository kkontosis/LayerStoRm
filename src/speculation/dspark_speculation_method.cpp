// DsparkSpeculationMethod implementation.  See dspark_speculation_method.h
// for the method/engine role split; the draft forward itself is engine
// machinery (D_CMD_RUN_DSPARK_STEP: DFlash backbone + sequential Markov
// head, DSP-3/DSP-4), reached through the installed DsparkStepExecutor.

#include "speculation/dspark_speculation_method.h"

#include "config/config_parser.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace layerstorm::speculation {

config::SpeculationMethodType DsparkSpeculationMethod::type() const {
    return config::SpeculationMethodType::dspark;
}

void DsparkSpeculationMethod::init(const SpeculationInitContext& ctx) {
    if (!ctx.config) {
        throw std::runtime_error(
            "DsparkSpeculationMethod::init: config is required");
    }
    const auto& d = ctx.config->speculation.dspark;

    // validate_dspark (DSP-1) already rejects speculative_tokens outside
    // [1, block_size] at parse time; re-check here so a method constructed
    // from a hand-built Config fails just as closed (SPEC-SCAFFOLD rule).
    if (d.speculative_tokens < 1 || d.speculative_tokens > d.block_size) {
        throw std::runtime_error(
            "speculation.dspark.speculative_tokens must be in [1, "
            "block_size=" + std::to_string(d.block_size) + "], got " +
            std::to_string(d.speculative_tokens));
    }
    // The runtime's query staging (host_ids_) and DsparkStepResult are both
    // 16-slot fixed arrays — a bigger block cannot be drafted in one step.
    if (d.block_size > DsparkStepResult::kMaxGamma) {
        throw std::runtime_error(
            "speculation.dspark.block_size " + std::to_string(d.block_size) +
            " exceeds the per-step draft cap " +
            std::to_string(DsparkStepResult::kMaxGamma));
    }

    gamma_      = d.speculative_tokens;
    block_size_ = d.block_size;

    spdlog::info("DsparkSpeculationMethod: gamma={} block_size={} "
                 "head_type=markov (draft executes via D_CMD_RUN_DSPARK_STEP)",
                 gamma_, block_size_);
}

void DsparkSpeculationMethod::reset(uint64_t seq_id) {
    // Unknown seq_ids tolerated.  A reset crosses ALL method state for the
    // sequence: the in-flight round and the anchor mirror (the runtime's
    // context re-arms on the next position-0 aux capture, INV-DSPARK-AUX).
    inflight_.erase(seq_id);
    anchor_.erase(seq_id);
}

void DsparkSpeculationMethod::propose_draft(const DraftContext& ctx,
                                            DraftResult& out) {
    out.count = 0;
    if (!executor_) {
        // No driver installed — decline (engine falls back to plain decode).
        // The Phase-12 orchestrator drives D_CMD_RUN_DSPARK_STEP directly
        // over the command ring (python/orchestrator/dspark_draft.py).
        if (!warned_no_executor_) {
            spdlog::warn("DsparkSpeculationMethod::propose_draft: no step "
                         "executor installed — declining to draft "
                         "(logged once)");
            warned_no_executor_ = true;
        }
        return;
    }
    if (!ctx.tokens || ctx.num_tokens <= 0 || !out.token_ids
        || out.capacity <= 0) {
        return;  // nothing to anchor on / nowhere to write — decline
    }

    const int budget = std::min({ctx.max_tokens, gamma_, out.capacity});
    if (budget <= 0) return;

    // Anchor convention (matches the MtpLossless/DsparkLossless drivers):
    // history = prompt + every committed token with the NEWEST one not yet
    // fed — so the anchor token is tokens[n-1] and it will be fed at
    // position n-1, which is exactly the fed-token count == the runtime's
    // ingested context length the backbone attends over (run_step validates
    // anchor_pos <= ctx_len fail-closed).
    DsparkStepRequest req;
    req.seq_id       = ctx.seq_id;
    req.anchor_token = ctx.tokens[ctx.num_tokens - 1];
    req.anchor_pos   = static_cast<uint32_t>(ctx.num_tokens - 1);
    req.num_query    = budget;
    req.stream       = ctx.stream;

    // Cross-check the driver-visible history against the committed anchor
    // mirror (telemetry — the history stays authoritative).
    if (!warned_anchor_mismatch_) {
        auto it = anchor_.find(ctx.seq_id);
        if (it != anchor_.end() && it->second != req.anchor_pos) {
            spdlog::warn("DsparkSpeculationMethod: seq {} anchor mirror {} "
                         "!= history-derived anchor_pos {} — driver history "
                         "convention drift? (logged once)",
                         ctx.seq_id, it->second, req.anchor_pos);
            warned_anchor_mismatch_ = true;
        }
    }

    DsparkStepResult res;
    if (!executor_(req, res) || res.count <= 0) {
        // Fail-closed decline: an unarmed/invalid draft context (e.g. the
        // aux export disabled itself on an unsupported step shape) must not
        // block decoding — the engine simply decodes autoregressively.
        return;
    }

    const int produced =
        std::min({res.count, budget, DsparkStepResult::kMaxGamma});
    for (int k = 0; k < produced; ++k)
        out.token_ids[k] = res.token_ids[k];
    // DSP-6: when the executor read back the trained confidence head's
    // c_k, surface the RAW per-position conditional survival probabilities
    // (INV-DSPARK-CONF; consumers cumprod for a_{r,j}, DSP-7 calibrates
    // the magnitudes).  Absent confidences (head disabled) keep the DSP-5
    // "probs untouched = no scores" contract.
    if (out.probs && res.confidence_count >= produced)
        for (int k = 0; k < produced; ++k)
            out.probs[k] = res.confidence[k];

    out.count = produced;
    if (produced > 0)
        inflight_[ctx.seq_id] = InflightRound{produced, req.anchor_pos};
}

void DsparkSpeculationMethod::verify(const VerifyContext& ctx,
                                     VerifyResult& out) {
    // Greedy acceptance rule (lossless, INV-DSPARK-LOSSLESS): accept the
    // longest prefix where the draft equals the target's chosen token;
    // bonus = the target token at the first rejected position, or the extra
    // target position when the whole draft was accepted.  The standard
    // contract — identical to MtpSpeculationMethod::verify.
    int accepted = 0;
    if (ctx.draft_tokens && ctx.target_tokens) {
        while (accepted < ctx.num_draft &&
               ctx.draft_tokens[accepted] == ctx.target_tokens[accepted]) {
            ++accepted;
        }
    }
    out.accepted_count = accepted;
    out.bonus_token =
        (ctx.target_tokens && ctx.num_draft >= 0 && accepted <= ctx.num_draft)
            ? ctx.target_tokens[accepted]
            : -1;
}

void DsparkSpeculationMethod::on_accept(uint64_t seq_id, int accepted_count) {
    auto it = inflight_.find(seq_id);
    if (it == inflight_.end()) return;  // accept without a tracked round
    const InflightRound round = it->second;
    inflight_.erase(it);
    if (round.proposed <= 0) return;

    const int accepted =
        std::min(std::max(accepted_count, 0), round.proposed);
    stats_.rounds          += 1;
    stats_.tokens_proposed += static_cast<uint64_t>(round.proposed);
    stats_.tokens_accepted += static_cast<uint64_t>(accepted);

    // Anchor mirror: the round feeds the anchor + the accepted drafts, then
    // the engine-appended bonus — accepted+1 new fed positions past the
    // round's anchor.  The draft context KV advanced in lockstep (every fed
    // target step auto-captured its aux hiddens, INV-DSPARK-AUX); a
    // rejected suffix was never fed, so nothing needs rewinding.
    anchor_[seq_id] =
        static_cast<int64_t>(round.anchor_pos) + accepted + 1;
}

void DsparkSpeculationMethod::on_rollback(uint64_t seq_id) {
    // Aborted round: discard in-flight state without touching acceptance
    // statistics (the round was never verified).  The anchor mirror stays —
    // the committed prefix is unchanged, and the next round simply re-drafts
    // at the same (shorter) anchor; any context rows a feed-then-rewind
    // driver wrote above it are overwritten in place by the next capture
    // (INV-DSPARK-AUX), never forked.
    inflight_.erase(seq_id);
}

}  // namespace layerstorm::speculation
