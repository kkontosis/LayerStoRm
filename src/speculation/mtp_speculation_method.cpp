// MtpSpeculationMethod implementation.  See mtp_speculation_method.h for the
// method/engine role split; the MTP forward itself is engine machinery
// (CommandDispatcher MTP projection + layer + shared head), reached through
// the installed MtpStepExecutor.

#include "speculation/mtp_speculation_method.h"

#include "config/config_parser.h"
#include "model/weight_loader/weight_loader.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace layerstorm::speculation {

config::SpeculationMethodType MtpSpeculationMethod::type() const {
    return config::SpeculationMethodType::mtp;
}

void MtpSpeculationMethod::init(const SpeculationInitContext& ctx) {
    if (!ctx.config) {
        throw std::runtime_error(
            "MtpSpeculationMethod::init: config is required");
    }
    const auto& cfg = *ctx.config;

    // Fail closed on contradictory / unusable configs (SPEC-SCAFFOLD rule:
    // a method must throw when a REQUIRED context piece is absent).
    if (cfg.model.num_nextn_predict_layers < 1) {
        throw std::runtime_error(
            "speculation.method=mtp requires model.num_nextn_predict_layers "
            ">= 1 (model has no MTP head)");
    }
    if (!cfg.speculation.mtp.enabled) {
        throw std::runtime_error(
            "speculation.method=mtp requires speculation.mtp.enabled=true");
    }
    // When the loaded checkpoint is visible, verify it actually carries the
    // MTP block (a config claiming nextn layers over a checkpoint without
    // them must not silently run non-speculatively).
    if (ctx.loaded_model && !ctx.loaded_model->mtp) {
        throw std::runtime_error(
            "speculation.method=mtp: checkpoint has no MTP weights "
            "(loaded_model->mtp is empty) despite num_nextn_predict_layers="
            + std::to_string(cfg.model.num_nextn_predict_layers));
    }

    max_depth_            = cfg.speculation.mtp.max_depth;
    dynamic_depth_        = cfg.speculation.mtp.dynamic_depth;
    confidence_threshold_ = cfg.speculation.mtp.confidence_threshold;
    num_hidden_layers_    = cfg.model.num_hidden_layers;
    num_mtp_layers_       = cfg.model.num_nextn_predict_layers;

    spdlog::info("MtpSpeculationMethod: max_depth={} dynamic_depth={} "
                 "confidence_threshold={} mtp_layers={} (base layer {})",
                 max_depth_, dynamic_depth_, confidence_threshold_,
                 num_mtp_layers_, num_hidden_layers_);
}

void MtpSpeculationMethod::reset(uint64_t seq_id) {
    inflight_proposed_.erase(seq_id);  // unknown seq_ids tolerated
}

void MtpSpeculationMethod::propose_draft(const DraftContext& ctx,
                                         DraftResult& out) {
    out.count = 0;
    if (!executor_) {
        // No driver installed — decline (engine falls back to plain decode).
        // The Phase-12 orchestrator drives the engine MTP pipeline directly
        // over the command ring (TD-MTP-ENGINE-PROPOSE).
        if (!warned_no_executor_) {
            spdlog::warn("MtpSpeculationMethod::propose_draft: no step "
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

    const int budget =
        std::min({ctx.max_tokens, max_depth_, out.capacity});
    if (budget <= 0) return;

    // Step 0 embeds the last visible (accepted) token; step k embeds the
    // step-(k-1) draft (DeepSeek V3 §2.2 recurrence: the engine keeps the
    // hidden anchor — trunk hidden for step 0, MTP-layer output after).
    int32_t input_token = ctx.tokens[ctx.num_tokens - 1];
    int produced = 0;
    for (int k = 0; k < budget; ++k) {
        MtpStepRequest req;
        req.seq_id        = ctx.seq_id;
        req.input_token   = input_token;
        req.step_idx      = k;
        req.mtp_layer_idx = mtp_layer_idx(k);
        req.stream        = ctx.stream;

        MtpStepResult res;
        if (!executor_(req, res) || res.token_id < 0) {
            spdlog::warn("MtpSpeculationMethod: step {} executor failed for "
                         "seq {} — stopping chain at {} draft(s)",
                         k, ctx.seq_id, produced);
            break;
        }

        out.token_ids[produced] = res.token_id;
        if (out.probs) out.probs[produced] = res.confidence;
        ++produced;
        input_token = res.token_id;

        // Dynamic-depth early exit: a low-confidence step is unlikely to be
        // accepted — deeper chaining just wastes draft compute (mirrors
        // orchestrator MtpDraft.should_continue()).
        if (dynamic_depth_ && produced < budget
            && res.confidence < static_cast<float>(confidence_threshold_)) {
            break;
        }
    }

    out.count = produced;
    if (produced > 0) inflight_proposed_[ctx.seq_id] = produced;
}

void MtpSpeculationMethod::verify(const VerifyContext& ctx, VerifyResult& out) {
    // Greedy acceptance rule (lossless): accept the longest prefix where the
    // draft equals the target's chosen token; bonus = the target token at the
    // first rejected position, or the extra target position when the whole
    // draft was accepted.  Identical to NullSpeculationMethod::verify — the
    // rule is the standard contract, not method-specific.
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

void MtpSpeculationMethod::on_accept(uint64_t seq_id, int accepted_count) {
    auto it = inflight_proposed_.find(seq_id);
    const int proposed = (it != inflight_proposed_.end()) ? it->second : 0;
    if (it != inflight_proposed_.end()) inflight_proposed_.erase(it);
    if (proposed <= 0) return;  // accept without a tracked round — ignore

    stats_.rounds           += 1;
    stats_.tokens_proposed  += static_cast<uint64_t>(proposed);
    stats_.tokens_accepted  += static_cast<uint64_t>(
        std::min(std::max(accepted_count, 0), proposed));
}

void MtpSpeculationMethod::on_rollback(uint64_t seq_id) {
    // Aborted round: discard in-flight state without touching acceptance
    // statistics (the round was never verified).
    inflight_proposed_.erase(seq_id);
}

}  // namespace layerstorm::speculation
