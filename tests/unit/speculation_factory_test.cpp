// Unit tests for the pluggable speculation scaffolding (SPEC-SCAFFOLD):
// config selector (speculation.method) → factory → SpeculationMethod seams.
// CPU-only — no CUDA, no model, no engine.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "config/config_parser.h"
#include "speculation/speculation_factory.h"
#include "speculation/speculation_method.h"

#include <array>
#include <stdexcept>

using namespace layerstorm;

namespace {

// Minimal valid config JSON (mirrors config_parser_test.cpp).
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
    };
}

}  // namespace

// ── Config selector ──────────────────────────────────────────────────────────

TEST(SpeculationFactory, DefaultMethodIsNone) {
    auto cfg = config::parse_config(minimal_json());
    EXPECT_EQ(cfg.speculation.method, config::SpeculationMethodType::none);
}

TEST(SpeculationFactory, MethodParsesFromConfig) {
    auto j = minimal_json();
    j["speculation"] = {{"method", "null"}};
    auto cfg = config::parse_config(j);
    EXPECT_EQ(cfg.speculation.method, config::SpeculationMethodType::null);

    j["speculation"]["method"] = "mtp";
    EXPECT_EQ(config::parse_config(j).speculation.method,
              config::SpeculationMethodType::mtp);

    j["speculation"]["method"] = "prompt_lookup";
    EXPECT_EQ(config::parse_config(j).speculation.method,
              config::SpeculationMethodType::prompt_lookup);

    j["speculation"]["method"] = "dspark";
    EXPECT_EQ(config::parse_config(j).speculation.method,
              config::SpeculationMethodType::dspark);
}

TEST(SpeculationFactory, InvalidMethodRejectedAtParse) {
    auto j = minimal_json();
    j["speculation"] = {{"method", "warp_drive"}};
    EXPECT_THROW(config::parse_config(j), std::exception);
}

// ── Factory selection ────────────────────────────────────────────────────────

TEST(SpeculationFactory, NoneReturnsNullptr) {
    auto cfg = config::parse_config(minimal_json());
    speculation::SpeculationInitContext ctx;
    ctx.config = &cfg;
    auto method = speculation::make_speculation_method(
        cfg.speculation.method, ctx);
    EXPECT_EQ(method, nullptr);  // subsystem off — nothing constructed
}

TEST(SpeculationFactory, SelectsNullStubByConfig) {
    auto j = minimal_json();
    j["speculation"] = {{"method", "null"}};
    auto cfg = config::parse_config(j);

    speculation::SpeculationInitContext ctx;
    ctx.config = &cfg;  // devices/loaded_model absent — the stub needs none
    auto method = speculation::make_speculation_method(
        cfg.speculation.method, ctx);
    ASSERT_NE(method, nullptr);
    EXPECT_EQ(method->type(), config::SpeculationMethodType::null);
    EXPECT_STREQ(method->name(), "null");
    EXPECT_EQ(method->max_draft_len(), 0);
    EXPECT_FALSE(method->needs_hidden_state());
}

TEST(SpeculationFactory, ReservedMethodsFailClosed) {
    speculation::SpeculationInitContext ctx;
    // mtp is LIVE (#16 / GLM-25g) but still fails closed without a config.
    EXPECT_THROW(speculation::make_speculation_method(
                     config::SpeculationMethodType::mtp, ctx),
                 std::runtime_error);
    EXPECT_THROW(speculation::make_speculation_method(
                     config::SpeculationMethodType::prompt_lookup, ctx),
                 std::runtime_error);
    // dspark is LIVE (DSP-5) but still fails closed without a config.
    EXPECT_THROW(speculation::make_speculation_method(
                     config::SpeculationMethodType::dspark, ctx),
                 std::runtime_error);
}

TEST(SpeculationFactory, SelectsDsparkByConfig) {
    // DSP-5: the dspark case constructs DsparkSpeculationMethod (draft
    // policy + greedy acceptance rule).  The DsparkRuntime (engine step 19e)
    // stays the fail-closed surface for the checkpoint / draft GPU; detailed
    // seam tests live in dspark_speculation_method_test.cpp.
    auto j = minimal_json();
    j["speculation"] = {{"method", "dspark"}};
    auto cfg = config::parse_config(j);
    EXPECT_TRUE(speculation::has_dspark(cfg));

    speculation::SpeculationInitContext ctx;
    ctx.config = &cfg;  // devices/loaded_model absent — tolerated (unit test)
    auto method = speculation::make_speculation_method(
        cfg.speculation.method, ctx);
    ASSERT_NE(method, nullptr);
    EXPECT_EQ(method->type(), config::SpeculationMethodType::dspark);
    EXPECT_STREQ(method->name(), "dspark");
    EXPECT_EQ(method->max_draft_len(),
              cfg.speculation.dspark.speculative_tokens);
    // The draft consumes aux hiddens via the automatic export hook
    // (INV-DSPARK-AUX) — never DraftContext.hidden_state.
    EXPECT_FALSE(method->needs_hidden_state());
}

TEST(SpeculationFactory, SelectsMtpByConfig) {
    // #16 / GLM-25g: the mtp case constructs MtpSpeculationMethod when the
    // model carries an MTP head.  Detailed seam tests live in
    // mtp_speculation_method_test.cpp.
    auto j = minimal_json();
    j["model"]["num_nextn_predict_layers"] = 1;
    j["speculation"] = {{"method", "mtp"}};
    auto cfg = config::parse_config(j);

    speculation::SpeculationInitContext ctx;
    ctx.config = &cfg;  // loaded_model absent — tolerated (unit test)
    auto method = speculation::make_speculation_method(
        cfg.speculation.method, ctx);
    ASSERT_NE(method, nullptr);
    EXPECT_EQ(method->type(), config::SpeculationMethodType::mtp);
    EXPECT_STREQ(method->name(), "mtp");
    EXPECT_EQ(method->max_draft_len(), cfg.speculation.mtp.max_depth);
    EXPECT_TRUE(method->needs_hidden_state());
}

TEST(SpeculationFactory, MtpFailsClosedWithoutMtpHead) {
    // A model without nextn layers must not silently run non-speculatively.
    auto j = minimal_json();
    j["model"]["num_nextn_predict_layers"] = 0;
    j["speculation"] = {{"method", "mtp"}};
    auto cfg = config::parse_config(j);

    speculation::SpeculationInitContext ctx;
    ctx.config = &cfg;
    EXPECT_THROW(speculation::make_speculation_method(
                     cfg.speculation.method, ctx),
                 std::runtime_error);
}

TEST(SpeculationFactory, MethodNames) {
    using config::SpeculationMethodType;
    EXPECT_STREQ(speculation::speculation_method_name(
                     SpeculationMethodType::none), "none");
    EXPECT_STREQ(speculation::speculation_method_name(
                     SpeculationMethodType::null), "null");
    EXPECT_STREQ(speculation::speculation_method_name(
                     SpeculationMethodType::mtp), "mtp");
    EXPECT_STREQ(speculation::speculation_method_name(
                     SpeculationMethodType::prompt_lookup), "prompt_lookup");
    EXPECT_STREQ(speculation::speculation_method_name(
                     SpeculationMethodType::dspark), "dspark");
}

// ── speculation.dspark config sub-section (DSP-1) ────────────────────────────

TEST(SpeculationFactory, DsparkConfigDefaults) {
    auto cfg = config::parse_config(minimal_json());
    const auto& d = cfg.speculation.dspark;
    // DSP-2: defaults match the shipped RedHatAI GLM-5.2 speculator
    // checkpoint (speculators v0.5) — γ=8, not the paper's DSpark-5.
    EXPECT_EQ(d.block_size, 8);       // γ — checkpoint block_size
    EXPECT_EQ(d.speculative_tokens, 7);  // γ−1 (anchor + 7 speculative)
    EXPECT_EQ(d.markov_rank, 256);    // r — low-rank Markov factorization
    EXPECT_EQ(d.head_type, config::DsparkHeadType::markov);
    EXPECT_FALSE(d.confidence_enabled);
    EXPECT_EQ(d.scheduler_mode, config::DsparkSchedulerMode::off);
    EXPECT_DOUBLE_EQ(d.confidence_threshold, 0.5);
    EXPECT_TRUE(d.sts_temperatures.empty());  // identity (no STS calibration)
    EXPECT_EQ(d.aux_hidden_state_layer_ids, (std::vector<int>{8, 23, 39, 55, 70}));
    EXPECT_EQ(d.mask_token_id, 154856);
    EXPECT_EQ(d.max_anchors, 1024);
    EXPECT_EQ(d.draft_vocab_size, 154880);
    EXPECT_EQ(d.checkpoint_path, "test-data/GLM-5.2-speculator.dspark");
    EXPECT_TRUE(d.draft_gpus.empty());  // auto: first non-TP GPU

    // Internal knobs (_internal-dspark).
    EXPECT_DOUBLE_EQ(cfg._internal_dspark.acceptance_ema_alpha, 0.3);

    // Default method stays none — dspark knobs alone select nothing.
    EXPECT_EQ(cfg.speculation.method, config::SpeculationMethodType::none);
    EXPECT_FALSE(speculation::has_dspark(cfg));
}

TEST(SpeculationFactory, DsparkConfigParsesKnobs) {
    auto j = minimal_json();
    j["speculation"] = {
        {"method", "dspark"},
        {"dspark", {
            {"block_size",           8},
            {"markov_rank",          512},
            {"head_type",            "rnn"},        // reserved value, parses
            {"confidence_enabled",   true},
            {"scheduler_mode",       "throughput"},
            {"confidence_threshold", 0.7},
            {"sts_temperatures",     {1.0, 1.1, 0.9, 1.2, 1.05, 1.0, 1.0, 1.0}},
        }},
    };
    auto cfg = config::parse_config(j);
    const auto& d = cfg.speculation.dspark;
    EXPECT_EQ(d.block_size, 8);
    EXPECT_EQ(d.markov_rank, 512);
    EXPECT_EQ(d.head_type, config::DsparkHeadType::rnn);
    EXPECT_TRUE(d.confidence_enabled);
    EXPECT_EQ(d.scheduler_mode, config::DsparkSchedulerMode::throughput);
    EXPECT_DOUBLE_EQ(d.confidence_threshold, 0.7);
    ASSERT_EQ(d.sts_temperatures.size(), 8u);
    EXPECT_DOUBLE_EQ(d.sts_temperatures[1], 1.1);
    EXPECT_TRUE(speculation::has_dspark(cfg));

    // static_threshold is the third scheduler mode.
    j["speculation"]["dspark"]["scheduler_mode"] = "static_threshold";
    EXPECT_EQ(config::parse_config(j).speculation.dspark.scheduler_mode,
              config::DsparkSchedulerMode::static_threshold);
}

TEST(SpeculationFactory, DsparkConfigRejectsInvalidEnums) {
    auto j = minimal_json();
    j["speculation"] = {{"dspark", {{"head_type", "transformer"}}}};
    EXPECT_THROW(config::parse_config(j), std::exception);

    j["speculation"] = {{"dspark", {{"scheduler_mode", "always"}}}};
    EXPECT_THROW(config::parse_config(j), std::exception);
}

// ── Interface seams on the stub ──────────────────────────────────────────────

TEST(SpeculationMethodSeams, NullStubDeclinesToDraft) {
    auto method = speculation::make_null_speculation_method();
    ASSERT_NE(method, nullptr);

    std::array<int32_t, 4> tokens{1, 2, 3, 4};
    std::array<int32_t, 8> out_tokens{};
    std::array<float, 8> out_probs{};

    speculation::DraftContext ctx;
    ctx.seq_id = 7;
    ctx.tokens = tokens.data();
    ctx.num_tokens = static_cast<int>(tokens.size());
    ctx.max_tokens = 4;

    speculation::DraftResult out;
    out.token_ids = out_tokens.data();
    out.probs = out_probs.data();
    out.capacity = static_cast<int>(out_tokens.size());
    out.count = -1;  // poison — must be overwritten

    method->propose_draft(ctx, out);
    EXPECT_EQ(out.count, 0);  // always declines
}

TEST(SpeculationMethodSeams, NullStubGreedyVerify) {
    auto method = speculation::make_null_speculation_method();

    // draft [10, 11, 99]; target chose [10, 11, 12, 13] → accept 2, bonus 12.
    std::array<int32_t, 3> draft{10, 11, 99};
    std::array<int32_t, 4> target{10, 11, 12, 13};

    speculation::VerifyContext ctx;
    ctx.seq_id = 7;
    ctx.draft_tokens = draft.data();
    ctx.num_draft = static_cast<int>(draft.size());
    ctx.target_tokens = target.data();

    speculation::VerifyResult res;
    method->verify(ctx, res);
    EXPECT_EQ(res.accepted_count, 2);
    EXPECT_EQ(res.bonus_token, 12);

    // Full acceptance → bonus is the target's extra position.
    std::array<int32_t, 3> draft_ok{10, 11, 12};
    ctx.draft_tokens = draft_ok.data();
    method->verify(ctx, res);
    EXPECT_EQ(res.accepted_count, 3);
    EXPECT_EQ(res.bonus_token, 13);

    // Immediate rejection → bonus is the target's first token.
    std::array<int32_t, 3> draft_bad{99, 11, 12};
    ctx.draft_tokens = draft_bad.data();
    method->verify(ctx, res);
    EXPECT_EQ(res.accepted_count, 0);
    EXPECT_EQ(res.bonus_token, 10);
}

TEST(SpeculationMethodSeams, NullStubStateHooksCallable) {
    auto method = speculation::make_null_speculation_method();
    // Lifecycle hooks must be callable with arbitrary seq ids (no-ops).
    method->reset(42);
    method->on_accept(42, 3);
    method->on_rollback(42);
    method->on_rollback(9999);  // unknown seq_id tolerated
    SUCCEED();
}
