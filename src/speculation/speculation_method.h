// Abstract interface for a pluggable speculative-decoding method.
//
// SpeculationMethod is the speculation analog of AttentionDevice
// (core/attention_device.h): a pure-abstract seam with POD context structs,
// selected by config enum (`speculation.method`) and baked into a concrete
// type at construction via make_speculation_method() (speculation_factory.h).
// Multiple methods coexist in the binary; exactly one (or none) is
// constructed per engine.
//
// The interface is deliberately NOT MTP-shaped.  It is the intersection of
// what four method families need:
//
//   MTP (DeepSeek/GLM nextn head, self-draft off the frozen trunk — #16 /
//        GLM-25g):   needs the trunk hidden state of the last accepted token
//        (needs_hidden_state() = true), device compute (init context carries
//        AttentionDevice/ExpertDevice), and its own weights (located via
//        LoadedModel: mtp_embed_tokens / eh_proj / enorm / hnorm /
//        shared_head.*).  Recurrence state across draft steps → on_accept /
//        on_rollback / reset.
//   Draft model (separate small LM):  needs device compute + its own weights
//        + its own KV state per sequence → same init context + state hooks;
//        does NOT need the trunk hidden state.
//   Prompt lookup (n-gram suffix match):  needs only the host-visible token
//        history (DraftContext.tokens) — no devices, no hidden state.
//   EAGLE (feature-level autoregression):  needs trunk hidden state(s) +
//        device compute + draft-KV rewind on rejection → hidden_state in
//        DraftContext, on_accept/on_rollback for its feature cache.
//
// Verification contract (TD-VERIFY-FETCHSEAM hook):
//   The METHOD never runs the target model.  The ENGINE executes the target
//   forward over the draft batch through the production routed-MoE seam
//   E_CMD_FETCH_AND_RUN_MOE (SPEC_UPDATES "MoE execution path"; the
//   deprecated RUN_MOE seam must NOT be used) and hands the per-position
//   target outputs to verify() via VerifyContext.  verify() implements only
//   the ACCEPTANCE RULE (greedy match, typical-acceptance, rejection
//   sampling, ...).  This is where the verifier rebase onto
//   FETCH_AND_RUN_MOE (TD-VERIFY-FETCHSEAM) lands: the engine-side
//   draft-batch forward is the FETCH seam adapted for multi-token batches;
//   the method-side rule stays engine-agnostic.
//
// CUDA-free (INV-GPU-1): void* streams, void* device pointers, POD params —
// no GPU SDK types.  A concrete method that needs kernels dispatches them
// through the AttentionDevice/ExpertDevice handles in its init context (or
// designates its own CUDA TU).

#pragma once

#include <cstdint>
#include <vector>

// Forward-declare config types — avoids pulling config_parser.h into
// consumers that only need the interface (mirrors attention_device.h).
namespace layerstorm::config {
struct Config;
enum class SpeculationMethodType : int;
}  // namespace layerstorm::config

namespace layerstorm::compute {
class AttentionDevice;
class ExpertDevice;
}  // namespace layerstorm::compute

namespace layerstorm::model { struct LoadedModel; }

namespace layerstorm::speculation {

// ── Contexts (POD, CUDA-free) ───────────────────────────────────────────────

/// Everything a method may need at construction/init time.  All pointers are
/// non-owning and must outlive the method (the Engine declares its
/// SpeculationMethod member after the pointees; reverse destruction order).
/// Unit tests may leave every field empty/null — a method must tolerate the
/// subset it does not use being absent, and fail loudly (throw) in init() if
/// a field it REQUIRES is missing.
struct SpeculationInitContext {
    /// Full parsed config: speculation.* knobs + model dims
    /// (hidden_size, vocab_size, num_nextn_predict_layers, ...).
    const config::Config* config = nullptr;

    /// Per-TP-rank attention devices (GEMM/RMSNorm/graphs) — empty in
    /// CPU-only tests.  Needed by MTP / draft-model / EAGLE.
    std::vector<compute::AttentionDevice*> attention_devices;

    /// Per-GPU expert devices (grouped GEMM/SwiGLU) — empty in CPU-only
    /// tests.  Needed by methods whose draft path runs MoE FFN layers.
    std::vector<compute::ExpertDevice*> expert_devices;

    /// Host-side loaded weights.  A method locates and uploads its OWN
    /// tensors (e.g. MTP: mtp_embed_tokens/eh_proj/enorm/hnorm/shared_head)
    /// — the interface carries no method-specific weight fields.
    const model::LoadedModel* loaded_model = nullptr;
};

/// Per-step draft-proposal input.
struct DraftContext {
    uint64_t seq_id = 0;

    /// Host pointer to the full visible token history (prompt + accepted
    /// generated tokens), length num_tokens.  Prompt-lookup consumes this
    /// directly; model-based methods typically use only the tail.
    const int32_t* tokens = nullptr;
    int num_tokens = 0;

    /// Device pointer to the trunk's output hidden state of the LAST
    /// ACCEPTED token, [hidden_size] BF16 — the self-draft anchor for MTP /
    /// EAGLE.  nullptr when the method reports needs_hidden_state() == false
    /// (the engine then never materializes it).
    const void* hidden_state = nullptr;

    /// Depth clamp for THIS step, min(max_draft_len(), utility-driven depth
    /// from speculation.utility_scorer / mtp.dynamic_depth).  The method
    /// must not propose more than this many tokens.
    int max_tokens = 0;

    /// Compute stream for device-side draft work (void* per INV-GPU-1);
    /// nullptr for host-only methods.
    void* stream = nullptr;
};

/// Draft-proposal output.  Buffers are CALLER-owned (no allocation on the
/// hot path); the method writes at most `capacity` entries and sets `count`.
struct DraftResult {
    int32_t* token_ids = nullptr;  ///< [capacity] host, drafted token ids
    float*   probs     = nullptr;  ///< [capacity] host, per-token draft
                                   ///< probability/confidence (adaptive exit,
                                   ///< rejection sampling); optional — a
                                   ///< method may leave it untouched when
                                   ///< nullptr or when it has no scores.
    int capacity = 0;              ///< caller: buffer sizes
    int count    = 0;              ///< method: tokens actually proposed (0 = no draft this step)
};

/// Verification input.  Target-model outputs are produced by the ENGINE via
/// the FETCH_AND_RUN_MOE seam (see file header) — never by the method.
struct VerifyContext {
    uint64_t seq_id = 0;

    const int32_t* draft_tokens = nullptr;  ///< [num_draft] host, this step's draft
    int num_draft = 0;

    /// [num_draft + 1] host: the target model's chosen next token at each
    /// draft position, plus the bonus position (index num_draft).  Always
    /// present — sufficient for the greedy acceptance rule.
    const int32_t* target_tokens = nullptr;

    /// Optional device pointer [num_draft + 1, vocab_size] to the target
    /// logits, for stochastic/typical acceptance rules.  nullptr when the
    /// engine samples greedily.
    const void* target_logits = nullptr;
    int vocab_size = 0;

    void* stream = nullptr;
};

/// Verification output (standard speculative-decoding contract): the first
/// `accepted_count` draft tokens are kept, then `bonus_token` (the target's
/// token at the first rejected position — or at the bonus position when the
/// whole draft was accepted) is appended.  Lossless by construction for the
/// greedy rule.
struct VerifyResult {
    int accepted_count = 0;   ///< in [0, num_draft]
    int32_t bonus_token = -1; ///< target token appended after the accepted prefix; -1 = none
};

// ── Interface ───────────────────────────────────────────────────────────────

class SpeculationMethod {
public:
    virtual ~SpeculationMethod() = default;

    // ── Identity / capabilities ─────────────────────────────────────────────

    /// The config enum value this method was constructed for.
    virtual config::SpeculationMethodType type() const = 0;

    /// Stable human-readable name (logging/metrics).
    virtual const char* name() const = 0;

    /// Upper bound of tokens this method can propose per step (0 = never
    /// proposes).  The engine sizes draft buffers and the speculation KV
    /// pool from this; DraftContext.max_tokens never exceeds it.
    virtual int max_draft_len() const = 0;

    /// True if propose_draft() requires DraftContext.hidden_state (MTP /
    /// EAGLE).  False lets the engine skip materializing the trunk hidden
    /// state for the speculation path (prompt lookup, draft model).
    virtual bool needs_hidden_state() const = 0;

    // ── Lifecycle ───────────────────────────────────────────────────────────

    /// One-time init: locate/upload weights, allocate device state, build
    /// tables.  Called once by the factory before the method is returned.
    /// Must throw std::runtime_error if a required context field is absent.
    virtual void init(const SpeculationInitContext& ctx) = 0;

    /// Per-sequence state reset (new sequence admitted, or sequence rewind
    /// crossing method-internal state).  Must tolerate unknown seq_ids.
    virtual void reset(uint64_t seq_id) = 0;

    // ── Per-step seams ──────────────────────────────────────────────────────

    /// Propose up to ctx.max_tokens draft tokens for the sequence.  Writes
    /// out.token_ids/probs, sets out.count (0 = decline to draft this step —
    /// always legal; the engine falls back to normal decode).
    virtual void propose_draft(const DraftContext& ctx, DraftResult& out) = 0;

    /// Apply the acceptance rule to (draft, target outputs).  See the file
    /// header for the engine/method split (TD-VERIFY-FETCHSEAM hook).
    virtual void verify(const VerifyContext& ctx, VerifyResult& out) = 0;

    /// Commit method-internal state after verification: the first
    /// accepted_count draft tokens (plus the engine-appended bonus token)
    /// are now part of the sequence.  MTP advances its recurrence anchor;
    /// EAGLE/draft-model trim their draft KV to the accepted prefix.
    virtual void on_accept(uint64_t seq_id, int accepted_count) = 0;

    /// Discard all in-flight draft state for the sequence without accepting
    /// anything (verification aborted, sequence rewound/cancelled).
    virtual void on_rollback(uint64_t seq_id) = 0;

    // Non-copyable (polymorphic base).
    SpeculationMethod(const SpeculationMethod&) = delete;
    SpeculationMethod& operator=(const SpeculationMethod&) = delete;

protected:
    SpeculationMethod() = default;
};

}  // namespace layerstorm::speculation
