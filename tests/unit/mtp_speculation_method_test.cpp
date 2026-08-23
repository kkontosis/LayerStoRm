// Unit tests for MtpSpeculationMethod (#16 / GLM-25g) — CPU-only.
//
// The MTP forward itself is engine machinery (dispatch_mtp_projection +
// MTP layer + shared head, integration-tested by the GLM-5.2 golden with
// GLM52_MTP=1); here a fake MtpStepExecutor stands in for it so the METHOD
// seams are tested in isolation: draft chaining + depth clamps + dynamic-
// depth confidence exit, the greedy acceptance rule, and per-seq state /
// acceptance statistics.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "config/config_parser.h"
#include "speculation/mtp_speculation_method.h"
#include "speculation/speculation_factory.h"

#include <algorithm>
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
            {"num_nextn_predict_layers", 1},
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
        {"speculation", {{"method", "mtp"}}},
    };
}

// Build an initialized method with the given mtp.* overrides.
std::unique_ptr<speculation::MtpSpeculationMethod> make_mtp(
        config::Config& cfg_out, nlohmann::json overrides = {}) {
    auto j = minimal_json();
    for (auto& [k, v] : overrides.items()) j["speculation"]["mtp"][k] = v;
    cfg_out = config::parse_config(j);

    auto method = std::make_unique<speculation::MtpSpeculationMethod>();
    speculation::SpeculationInitContext ctx;
    ctx.config = &cfg_out;
    method->init(ctx);
    return method;
}

// Fake executor: emits token = 1000 + step_idx with the scripted confidence
// per step; records requests for assertions.
struct FakeExecutor {
    std::vector<speculation::MtpStepRequest> requests;
    std::vector<float> confidences;  // per step; last value repeats
    bool fail_at_step = false;
    int fail_step = -1;

    speculation::MtpStepExecutor fn() {
        return [this](const speculation::MtpStepRequest& req,
                      speculation::MtpStepResult& res) {
            requests.push_back(req);
            if (fail_at_step && req.step_idx == fail_step) return false;
            res.token_id = 1000 + req.step_idx;
            const size_t i = std::min(
                static_cast<size_t>(req.step_idx),
                confidences.empty() ? 0 : confidences.size() - 1);
            res.confidence = confidences.empty() ? 1.0f : confidences[i];
            return true;
        };
    }
};

}  // namespace

// ── init() fail-closed contract ──────────────────────────────────────────────

TEST(MtpSpeculationMethod, InitThrowsWithoutConfig) {
    speculation::MtpSpeculationMethod method;
    speculation::SpeculationInitContext ctx;  // no config
    EXPECT_THROW(method.init(ctx), std::runtime_error);
}

TEST(MtpSpeculationMethod, InitThrowsWhenMtpDisabled) {
    auto j = minimal_json();
    j["speculation"]["mtp"]["enabled"] = false;
    auto cfg = config::parse_config(j);
    speculation::MtpSpeculationMethod method;
    speculation::SpeculationInitContext ctx;
    ctx.config = &cfg;
    EXPECT_THROW(method.init(ctx), std::runtime_error);
}

TEST(MtpSpeculationMethod, IdentityFromConfig) {
    config::Config cfg;
    auto method = make_mtp(cfg, {{"max_depth", 4}});
    EXPECT_EQ(method->type(), config::SpeculationMethodType::mtp);
    EXPECT_STREQ(method->name(), "mtp");
    EXPECT_EQ(method->max_draft_len(), 4);
    EXPECT_TRUE(method->needs_hidden_state());
    EXPECT_EQ(method->num_mtp_layers(), 1);
    // Cyclic MTP layer indexing: num_hidden_layers + (step % num_mtp).
    EXPECT_EQ(method->mtp_layer_idx(0), 61);
    EXPECT_EQ(method->mtp_layer_idx(2), 61);
}

// ── propose_draft ────────────────────────────────────────────────────────────

TEST(MtpSpeculationMethod, DeclinesWithoutExecutor) {
    config::Config cfg;
    auto method = make_mtp(cfg);
    std::array<int32_t, 2> history{5, 7};
    std::array<int32_t, 8> out_tokens{};
    speculation::DraftContext ctx;
    ctx.tokens = history.data();
    ctx.num_tokens = 2;
    ctx.max_tokens = 3;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 8;
    out.count = -1;
    method->propose_draft(ctx, out);
    EXPECT_EQ(out.count, 0);  // decline — engine falls back to plain decode
}

TEST(MtpSpeculationMethod, ChainsStepsAndFeedsBackDraftTokens) {
    config::Config cfg;
    auto method = make_mtp(cfg, {{"max_depth", 3}, {"dynamic_depth", false}});
    FakeExecutor exec;
    method->set_step_executor(exec.fn());

    std::array<int32_t, 3> history{5, 7, 42};
    std::array<int32_t, 8> out_tokens{};
    std::array<float, 8> out_probs{};
    speculation::DraftContext ctx;
    ctx.seq_id = 9;
    ctx.tokens = history.data();
    ctx.num_tokens = 3;
    ctx.max_tokens = 3;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.probs = out_probs.data();
    out.capacity = 8;

    method->propose_draft(ctx, out);
    ASSERT_EQ(out.count, 3);
    EXPECT_EQ(out_tokens[0], 1000);
    EXPECT_EQ(out_tokens[1], 1001);
    EXPECT_EQ(out_tokens[2], 1002);

    // Step 0 embeds the LAST history token; step k embeds draft k-1.
    ASSERT_EQ(exec.requests.size(), 3u);
    EXPECT_EQ(exec.requests[0].input_token, 42);
    EXPECT_EQ(exec.requests[0].seq_id, 9u);
    EXPECT_EQ(exec.requests[0].mtp_layer_idx, 61);
    EXPECT_EQ(exec.requests[1].input_token, 1000);
    EXPECT_EQ(exec.requests[2].input_token, 1001);
}

TEST(MtpSpeculationMethod, DepthClampedByContextAndCapacity) {
    config::Config cfg;
    auto method = make_mtp(cfg, {{"max_depth", 5}, {"dynamic_depth", false}});
    FakeExecutor exec;
    method->set_step_executor(exec.fn());

    std::array<int32_t, 1> history{11};
    std::array<int32_t, 2> out_tokens{};
    speculation::DraftContext ctx;
    ctx.tokens = history.data();
    ctx.num_tokens = 1;
    ctx.max_tokens = 4;  // utility clamp below max_depth
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 2;    // buffer clamp below both

    method->propose_draft(ctx, out);
    EXPECT_EQ(out.count, 2);  // min(max_tokens=4, max_depth=5, capacity=2)
}

TEST(MtpSpeculationMethod, DynamicDepthConfidenceExit) {
    config::Config cfg;
    auto method = make_mtp(cfg, {{"max_depth", 5},
                                 {"dynamic_depth", true},
                                 {"confidence_threshold", 0.4}});
    FakeExecutor exec;
    exec.confidences = {0.9f, 0.2f, 0.9f};  // step 1 below threshold
    method->set_step_executor(exec.fn());

    std::array<int32_t, 1> history{11};
    std::array<int32_t, 8> out_tokens{};
    speculation::DraftContext ctx;
    ctx.tokens = history.data();
    ctx.num_tokens = 1;
    ctx.max_tokens = 5;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 8;

    method->propose_draft(ctx, out);
    // Step 0 (0.9) continues; step 1 (0.2 < 0.4) is KEPT but stops the chain.
    EXPECT_EQ(out.count, 2);
    EXPECT_EQ(exec.requests.size(), 2u);
}

TEST(MtpSpeculationMethod, ExecutorFailureStopsChain) {
    config::Config cfg;
    auto method = make_mtp(cfg, {{"max_depth", 3}, {"dynamic_depth", false}});
    FakeExecutor exec;
    exec.fail_at_step = true;
    exec.fail_step = 1;
    method->set_step_executor(exec.fn());

    std::array<int32_t, 1> history{11};
    std::array<int32_t, 8> out_tokens{};
    speculation::DraftContext ctx;
    ctx.tokens = history.data();
    ctx.num_tokens = 1;
    ctx.max_tokens = 3;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 8;

    method->propose_draft(ctx, out);
    EXPECT_EQ(out.count, 1);  // step 0 kept; failed step 1 stops the chain
}

// ── verify (greedy rule) ─────────────────────────────────────────────────────

TEST(MtpSpeculationMethod, GreedyVerifyPartialFullAndReject) {
    config::Config cfg;
    auto method = make_mtp(cfg);

    std::array<int32_t, 3> draft{10, 11, 99};
    std::array<int32_t, 4> target{10, 11, 12, 13};
    speculation::VerifyContext ctx;
    ctx.draft_tokens = draft.data();
    ctx.num_draft = 3;
    ctx.target_tokens = target.data();

    speculation::VerifyResult res;
    method->verify(ctx, res);
    EXPECT_EQ(res.accepted_count, 2);
    EXPECT_EQ(res.bonus_token, 12);

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

// ── state hooks + acceptance statistics ──────────────────────────────────────

TEST(MtpSpeculationMethod, AcceptanceStatsAccumulateOnAccept) {
    config::Config cfg;
    auto method = make_mtp(cfg, {{"max_depth", 3}, {"dynamic_depth", false}});
    FakeExecutor exec;
    method->set_step_executor(exec.fn());

    std::array<int32_t, 1> history{11};
    std::array<int32_t, 8> out_tokens{};
    speculation::DraftContext ctx;
    ctx.seq_id = 5;
    ctx.tokens = history.data();
    ctx.num_tokens = 1;
    ctx.max_tokens = 3;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 8;

    method->propose_draft(ctx, out);
    ASSERT_EQ(out.count, 3);
    method->on_accept(5, 2);

    EXPECT_EQ(method->acceptance_stats().rounds, 1u);
    EXPECT_EQ(method->acceptance_stats().tokens_proposed, 3u);
    EXPECT_EQ(method->acceptance_stats().tokens_accepted, 2u);
    EXPECT_NEAR(method->acceptance_rate(), 2.0 / 3.0, 1e-9);

    // Second round, full acceptance.
    method->propose_draft(ctx, out);
    method->on_accept(5, 3);
    EXPECT_EQ(method->acceptance_stats().rounds, 2u);
    EXPECT_NEAR(method->acceptance_rate(), 5.0 / 6.0, 1e-9);

    // on_accept without an in-flight round is a tolerated no-op.
    method->on_accept(5, 1);
    EXPECT_EQ(method->acceptance_stats().rounds, 2u);
}

TEST(MtpSpeculationMethod, RollbackAndResetDiscardInflightRound) {
    config::Config cfg;
    auto method = make_mtp(cfg, {{"max_depth", 2}, {"dynamic_depth", false}});
    FakeExecutor exec;
    method->set_step_executor(exec.fn());

    std::array<int32_t, 1> history{11};
    std::array<int32_t, 8> out_tokens{};
    speculation::DraftContext ctx;
    ctx.seq_id = 5;
    ctx.tokens = history.data();
    ctx.num_tokens = 1;
    ctx.max_tokens = 2;
    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.capacity = 8;

    method->propose_draft(ctx, out);
    method->on_rollback(5);        // aborted round — no stats
    method->on_accept(5, 2);       // nothing in flight — no stats
    EXPECT_EQ(method->acceptance_stats().rounds, 0u);
    EXPECT_EQ(method->acceptance_stats().tokens_proposed, 0u);

    method->propose_draft(ctx, out);
    method->reset(5);              // sequence reset — same discard semantics
    method->on_accept(5, 2);
    EXPECT_EQ(method->acceptance_stats().rounds, 0u);

    method->reset(9999);           // unknown seq tolerated
    method->on_rollback(9999);
    SUCCEED();
}
