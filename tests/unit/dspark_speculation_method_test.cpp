// Unit tests for DsparkSpeculationMethod (DSP-5) — CPU-only.
//
// The draft forward itself is engine machinery (D_CMD_RUN_DSPARK_STEP:
// DFlash backbone + sequential Markov head, integration-tested by the
// GLM-5.2 golden with GLM52_DSPARK=1); here a fake DsparkStepExecutor
// stands in for it so the METHOD seams are tested in isolation: the
// whole-γ-block proposal + clamps + anchor convention, the greedy
// acceptance rule (lossless), and per-seq anchor/round state / acceptance
// statistics.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "config/config_parser.h"
#include "speculation/dspark_speculation_method.h"
#include "speculation/speculation_factory.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace layerstorm;

namespace {

nlohmann::json minimal_json() {
    return {
        {"model", {
            {"architecture",           "deepseek_v3"},
            {"weights_path",           "/data/models/test/"},
            {"weights_format",         "safetensors"},
            {"num_hidden_layers",      61},
            {"hidden_size",            7168},
            {"num_attention_heads",    128},
            {"num_key_value_heads",    128},
            {"intermediate_size",      18432},
            {"n_routed_experts",       256},
            {"num_experts_per_tok",    8},
            {"vocab_size",             129280},
            {"max_position_embeddings", 163840},
        }},
        {"quantization", {
            {"weights",           "nvfp4"},
            {"attention_compute", "fp8_e4m3"},
            {"kv_cache",          "fp8_e4m3"},
            {"gating_compute",    "fp32"},
        }},
        {"hardware", {
            {"gpus", {{
                {"id",       0},
                {"type",     "rtx5090"},
                {"vram_gb",  32},
            }}},
            {"system_ram_gb", 256},
        }},
        {"speculation", {{"method", "dspark"}}},
    };
}

// Build an initialized method with the given dspark.* overrides.
std::unique_ptr<speculation::DsparkSpeculationMethod> make_dspark(
        config::Config& cfg_out, nlohmann::json overrides = {}) {
    auto j = minimal_json();
    for (auto& [k, v] : overrides.items()) j["speculation"]["dspark"][k] = v;
    cfg_out = config::parse_config(j);

    auto method = std::make_unique<speculation::DsparkSpeculationMethod>();
    speculation::SpeculationInitContext ctx;
    ctx.config = &cfg_out;
    method->init(ctx);
    return method;
}

// Fake executor: one call = one whole γ block.  Emits num_query tokens
// 2000 + k (or a scripted list); records requests for assertions.
struct FakeExecutor {
    std::vector<speculation::DsparkStepRequest> requests;
    std::vector<int32_t> scripted;  // when non-empty: exact ids to emit
    std::vector<float> confidences; // when non-empty: DSP-6 c_k to emit
    int emit_count = -1;            // -1 = num_query
    bool fail = false;

    speculation::DsparkStepExecutor fn() {
        return [this](const speculation::DsparkStepRequest& req,
                      speculation::DsparkStepResult& res) {
            requests.push_back(req);
            if (fail) return false;
            const int n = emit_count >= 0 ? emit_count : req.num_query;
            res.count = n;
            for (int k = 0; k < n && k < res.kMaxGamma; ++k) {
                res.token_ids[k] =
                    k < static_cast<int>(scripted.size()) ? scripted[k]
                                                          : 2000 + k;
            }
            for (int k = 0;
                 k < static_cast<int>(confidences.size())
                 && k < res.kMaxGamma; ++k)
                res.confidence[k] = confidences[static_cast<size_t>(k)];
            res.confidence_count =
                static_cast<int>(confidences.size());
            return true;
        };
    }
};

}  // namespace

// ── init() fail-closed contract ──────────────────────────────────────────────

TEST(DsparkSpeculationMethod, InitThrowsWithoutConfig) {
    speculation::DsparkSpeculationMethod method;
    speculation::SpeculationInitContext ctx;  // no config
    EXPECT_THROW(method.init(ctx), std::runtime_error);
}

TEST(DsparkSpeculationMethod, InitFailsClosedOnInconsistentGamma) {
    // Hand-built config bypassing validate_dspark: speculative_tokens >
    // block_size must still fail at method init (SPEC-SCAFFOLD rule).
    auto cfg = config::parse_config(minimal_json());
    cfg.speculation.dspark.speculative_tokens = 9;
    cfg.speculation.dspark.block_size = 8;
    speculation::DsparkSpeculationMethod method;
    speculation::SpeculationInitContext ctx;
    ctx.config = &cfg;
    EXPECT_THROW(method.init(ctx), std::runtime_error);

    // block_size beyond the 16-slot per-step cap fails too.
    cfg.speculation.dspark.block_size = 32;
    cfg.speculation.dspark.speculative_tokens = 20;
    EXPECT_THROW(method.init(ctx), std::runtime_error);
}

TEST(DsparkSpeculationMethod, IdentityFromConfig) {
    config::Config cfg;
    auto method = make_dspark(cfg);
    EXPECT_EQ(method->type(), config::SpeculationMethodType::dspark);
    EXPECT_STREQ(method->name(), "dspark");
    // Shipped-checkpoint defaults: γ = speculative_tokens = 7, block 8.
    EXPECT_EQ(method->max_draft_len(), 7);
    EXPECT_EQ(method->gamma(), 7);
    EXPECT_EQ(method->block_size(), 8);
    EXPECT_FALSE(method->needs_hidden_state());
}

// ── propose_draft ────────────────────────────────────────────────────────────

TEST(DsparkSpeculationMethod, DeclinesWithoutExecutor) {
    config::Config cfg;
    auto method = make_dspark(cfg);
    std::array<int32_t, 2> history{5, 7};
    std::array<int32_t, 16> out_tokens{};
    speculation::DraftContext ctx;
    ctx.tokens = history.data();
    ctx.num_tokens = 2;
    ctx.max_tokens = 7;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 16;
    out.count = -1;
    method->propose_draft(ctx, out);
    EXPECT_EQ(out.count, 0);  // decline — engine falls back to plain decode
}

TEST(DsparkSpeculationMethod, ProposesWholeGammaBlockOffAnchor) {
    config::Config cfg;
    auto method = make_dspark(cfg);
    FakeExecutor exec;
    method->set_step_executor(exec.fn());

    // History convention: prompt(5) + newest committed-but-unfed token 42
    // → anchor token 42 at anchor_pos 5 (== fed count == ingested ctx_len).
    std::array<int32_t, 6> history{10, 20, 30, 40, 50, 42};
    std::array<int32_t, 16> out_tokens{};
    speculation::DraftContext ctx;
    ctx.seq_id = 9;
    ctx.tokens = history.data();
    ctx.num_tokens = 6;
    ctx.max_tokens = 7;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 16;

    method->propose_draft(ctx, out);
    ASSERT_EQ(out.count, 7);  // one call drafted the whole γ block
    for (int k = 0; k < 7; ++k) EXPECT_EQ(out_tokens[k], 2000 + k);

    ASSERT_EQ(exec.requests.size(), 1u);  // ONE executor call per round
    EXPECT_EQ(exec.requests[0].seq_id, 9u);
    EXPECT_EQ(exec.requests[0].anchor_token, 42);
    EXPECT_EQ(exec.requests[0].anchor_pos, 5u);
    EXPECT_EQ(exec.requests[0].num_query, 7);
}

// DSP-6 (INV-DSPARK-CONF): the executor's raw survival c_k surface as
// out.probs; absent confidences leave probs untouched ("no scores").
TEST(DsparkSpeculationMethod, ConfidencesSurfaceAsProbsWhenProvided) {
    config::Config cfg;
    auto method = make_dspark(cfg);
    FakeExecutor exec;
    exec.confidences = {0.9f, 0.7f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
    method->set_step_executor(exec.fn());

    std::array<int32_t, 6> history{10, 20, 30, 40, 50, 42};
    std::array<int32_t, 16> out_tokens{};
    std::array<float, 16> out_probs{};
    out_probs.fill(-1.0f);  // sentinel: untouched means -1
    speculation::DraftContext ctx;
    ctx.seq_id = 9;
    ctx.tokens = history.data();
    ctx.num_tokens = 6;
    ctx.max_tokens = 7;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.probs = out_probs.data();
    out.capacity = 16;

    method->propose_draft(ctx, out);
    ASSERT_EQ(out.count, 7);
    for (int k = 0; k < 7; ++k)
        EXPECT_FLOAT_EQ(out_probs[static_cast<size_t>(k)],
                        exec.confidences[static_cast<size_t>(k)])
            << "probs[" << k << "]";

    // Confidence head off (executor emits none): probs stay untouched.
    exec.confidences.clear();
    out_probs.fill(-1.0f);
    method->propose_draft(ctx, out);
    ASSERT_EQ(out.count, 7);
    for (int k = 0; k < 7; ++k)
        EXPECT_FLOAT_EQ(out_probs[static_cast<size_t>(k)], -1.0f)
            << "probs[" << k << "] must stay untouched (no scores)";
}

TEST(DsparkSpeculationMethod, GammaClampedByContextAndCapacity) {
    config::Config cfg;
    auto method = make_dspark(cfg);
    FakeExecutor exec;
    method->set_step_executor(exec.fn());

    std::array<int32_t, 1> history{11};
    std::array<int32_t, 3> out_tokens{};
    speculation::DraftContext ctx;
    ctx.tokens = history.data();
    ctx.num_tokens = 1;
    ctx.max_tokens = 5;  // utility clamp below γ=7
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 3;    // buffer clamp below both

    method->propose_draft(ctx, out);
    EXPECT_EQ(out.count, 3);  // min(max_tokens=5, gamma=7, capacity=3)
    ASSERT_EQ(exec.requests.size(), 1u);
    EXPECT_EQ(exec.requests[0].num_query, 3);

    // An executor emitting MORE than requested is clamped to the budget.
    exec.emit_count = 7;
    method->propose_draft(ctx, out);
    EXPECT_EQ(out.count, 3);
}

TEST(DsparkSpeculationMethod, ExecutorFailureDeclinesRound) {
    config::Config cfg;
    auto method = make_dspark(cfg);
    FakeExecutor exec;
    exec.fail = true;  // unarmed/invalid draft context → fail-closed decline
    method->set_step_executor(exec.fn());

    std::array<int32_t, 1> history{11};
    std::array<int32_t, 16> out_tokens{};
    speculation::DraftContext ctx;
    ctx.seq_id = 4;
    ctx.tokens = history.data();
    ctx.num_tokens = 1;
    ctx.max_tokens = 7;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 16;
    out.count = -1;

    method->propose_draft(ctx, out);
    EXPECT_EQ(out.count, 0);
    // No round was opened: a later on_accept is a tolerated no-op.
    method->on_accept(4, 3);
    EXPECT_EQ(method->acceptance_stats().rounds, 0u);
}

// ── verify (greedy rule — lossless by construction) ─────────────────────────

TEST(DsparkSpeculationMethod, GreedyVerifyPartialFullAndReject) {
    config::Config cfg;
    auto method = make_dspark(cfg);

    std::array<int32_t, 3> draft{10, 11, 99};
    std::array<int32_t, 4> target{10, 11, 12, 13};
    speculation::VerifyContext ctx;
    ctx.draft_tokens = draft.data();
    ctx.num_draft = 3;
    ctx.target_tokens = target.data();

    speculation::VerifyResult res;
    method->verify(ctx, res);
    EXPECT_EQ(res.accepted_count, 2);
    EXPECT_EQ(res.bonus_token, 12);  // target at the first mismatch

    std::array<int32_t, 3> draft_ok{10, 11, 12};
    ctx.draft_tokens = draft_ok.data();
    method->verify(ctx, res);
    EXPECT_EQ(res.accepted_count, 3);
    EXPECT_EQ(res.bonus_token, 13);  // full acceptance → extra position

    std::array<int32_t, 3> draft_bad{99, 11, 12};
    ctx.draft_tokens = draft_bad.data();
    method->verify(ctx, res);
    EXPECT_EQ(res.accepted_count, 0);
    EXPECT_EQ(res.bonus_token, 10);
}

// ── state hooks: anchor mirror + acceptance statistics ───────────────────────

TEST(DsparkSpeculationMethod, AcceptAdvancesAnchorAndStats) {
    config::Config cfg;
    auto method = make_dspark(cfg);
    FakeExecutor exec;
    method->set_step_executor(exec.fn());

    std::array<int32_t, 6> history{10, 20, 30, 40, 50, 42};
    std::array<int32_t, 16> out_tokens{};
    speculation::DraftContext ctx;
    ctx.seq_id = 5;
    ctx.tokens = history.data();
    ctx.num_tokens = 6;
    ctx.max_tokens = 7;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 16;

    EXPECT_EQ(method->expected_anchor_pos(5), -1);  // no round yet

    method->propose_draft(ctx, out);
    ASSERT_EQ(out.count, 7);
    method->on_accept(5, 2);  // 2 drafts + bonus fed → anchor 5 + 3 = 8

    EXPECT_EQ(method->acceptance_stats().rounds, 1u);
    EXPECT_EQ(method->acceptance_stats().tokens_proposed, 7u);
    EXPECT_EQ(method->acceptance_stats().tokens_accepted, 2u);
    EXPECT_NEAR(method->acceptance_rate(), 2.0 / 7.0, 1e-9);
    EXPECT_EQ(method->expected_anchor_pos(5), 8);

    // Second round at the advanced anchor (history grew by accepted+1=3):
    // full acceptance advances by γ+1.
    std::array<int32_t, 9> history2{10, 20, 30, 40, 50, 42, 61, 62, 63};
    ctx.tokens = history2.data();
    ctx.num_tokens = 9;
    method->propose_draft(ctx, out);
    ASSERT_EQ(exec.requests.size(), 2u);
    EXPECT_EQ(exec.requests[1].anchor_pos, 8u);  // matches the mirror
    method->on_accept(5, 7);
    EXPECT_EQ(method->expected_anchor_pos(5), 16);
    EXPECT_NEAR(method->acceptance_rate(), 9.0 / 14.0, 1e-9);

    // on_accept without an in-flight round is a tolerated no-op.
    method->on_accept(5, 1);
    EXPECT_EQ(method->acceptance_stats().rounds, 2u);
}

TEST(DsparkSpeculationMethod, RejectionRoundKeepsShorterAnchor) {
    // Full rejection: 0 accepted + bonus fed → anchor advances by exactly 1;
    // the next round re-drafts at the shorter prefix (context-KV rewind is
    // an in-place overwrite by construction — nothing to do here).
    config::Config cfg;
    auto method = make_dspark(cfg);
    FakeExecutor exec;
    method->set_step_executor(exec.fn());

    std::array<int32_t, 6> history{10, 20, 30, 40, 50, 42};
    std::array<int32_t, 16> out_tokens{};
    speculation::DraftContext ctx;
    ctx.seq_id = 6;
    ctx.tokens = history.data();
    ctx.num_tokens = 6;
    ctx.max_tokens = 7;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 16;

    method->propose_draft(ctx, out);
    method->on_accept(6, 0);  // everything rejected; bonus still fed
    EXPECT_EQ(method->expected_anchor_pos(6), 6);  // 5 + 0 + 1
    EXPECT_EQ(method->acceptance_stats().tokens_accepted, 0u);
    EXPECT_EQ(method->acceptance_stats().tokens_proposed, 7u);
}

TEST(DsparkSpeculationMethod, RollbackAndResetDiscardInflightRound) {
    config::Config cfg;
    auto method = make_dspark(cfg);
    FakeExecutor exec;
    method->set_step_executor(exec.fn());

    std::array<int32_t, 2> history{11, 12};
    std::array<int32_t, 16> out_tokens{};
    speculation::DraftContext ctx;
    ctx.seq_id = 5;
    ctx.tokens = history.data();
    ctx.num_tokens = 2;
    ctx.max_tokens = 7;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 16;

    method->propose_draft(ctx, out);
    method->on_rollback(5);        // aborted round — no stats
    method->on_accept(5, 2);       // nothing in flight — no stats
    EXPECT_EQ(method->acceptance_stats().rounds, 0u);
    EXPECT_EQ(method->acceptance_stats().tokens_proposed, 0u);
    // Rollback keeps the committed anchor mirror (none committed yet).
    EXPECT_EQ(method->expected_anchor_pos(5), -1);

    // Commit one round, then rollback the next: the anchor mirror holds.
    method->propose_draft(ctx, out);
    method->on_accept(5, 1);
    EXPECT_EQ(method->expected_anchor_pos(5), 3);  // 1 + 1 + 1
    method->propose_draft(ctx, out);
    method->on_rollback(5);
    EXPECT_EQ(method->expected_anchor_pos(5), 3);  // unchanged

    method->reset(5);              // sequence reset — anchor mirror cleared
    EXPECT_EQ(method->expected_anchor_pos(5), -1);
    method->propose_draft(ctx, out);
    method->reset(5);
    method->on_accept(5, 2);       // round discarded by reset — no stats
    EXPECT_EQ(method->acceptance_stats().rounds, 1u);

    method->reset(9999);           // unknown seq tolerated
    method->on_rollback(9999);
    SUCCEED();
}
