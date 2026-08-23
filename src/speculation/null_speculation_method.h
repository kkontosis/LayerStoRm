// No-op SpeculationMethod stub (`speculation.method = "null"`).
//
// Proves the pluggable-speculation seam end to end (config enum → factory →
// engine member → interface calls) without changing inference behavior:
// max_draft_len() == 0, propose_draft() always declines (count = 0), so the
// engine never enters the speculative path.  verify() still implements the
// standard greedy acceptance rule so the seam contract is exercised by unit
// tests.  Selectable from config for plumbing diagnostics; mirrors
// NullAttentionDevice (core/null_attention_device.h).  Header-only,
// CUDA-free (INV-GPU-1).

#pragma once

#include "speculation/speculation_method.h"

namespace layerstorm::speculation {

class NullSpeculationMethod final : public SpeculationMethod {
public:
    NullSpeculationMethod() = default;
    ~NullSpeculationMethod() override = default;

    // ── Identity / capabilities ─────────────────────────────────────────────
    config::SpeculationMethodType type() const override;  // ::null (in .cpp — needs config_parser.h)
    const char* name() const override { return "null"; }
    int max_draft_len() const override { return 0; }
    bool needs_hidden_state() const override { return false; }

    // ── Lifecycle ───────────────────────────────────────────────────────────
    void init(const SpeculationInitContext&) override {}  // requires nothing
    void reset(uint64_t) override {}

    // ── Per-step seams ──────────────────────────────────────────────────────
    void propose_draft(const DraftContext&, DraftResult& out) override {
        out.count = 0;  // always decline — engine falls back to normal decode
    }

    /// Greedy acceptance rule: accept the longest prefix where the draft
    /// token equals the target's chosen token at that position; the bonus
    /// token is the target token at the first mismatch (or the target's
    /// extra position when the whole draft matched).  With max_draft_len()
    /// == 0 this is never reached in production — kept exact so the seam
    /// contract is testable.
    void verify(const VerifyContext& ctx, VerifyResult& out) override {
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

    void on_accept(uint64_t, int) override {}
    void on_rollback(uint64_t) override {}
};

}  // namespace layerstorm::speculation
