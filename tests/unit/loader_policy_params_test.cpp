// Unit tests for the I8 policy-weights artifact loader (P-26).
// CPU-only — pure JSON parsing, no GPU, no engine.
//
// Parse contract under test (loader_policy.h):
//   * absent optional fields -> built-in default (partial artifacts OK);
//   * unparsable JSON / non-object root / wrong-typed / out-of-range values
//     -> throw std::runtime_error (fail loud, never silent fallback);
//   * env precedence: explicit LS_LOADER_* env vars > artifact > defaults,
//     with the historical env clamps (negative w -> 0, bad decay/tau -> their
//     defaults).
// Ticket: spec/tickets/P-26_I8_DEPLOYMENT_FIT.md.
#include "core/gpu_loader/loader_policy.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace gl = layerstorm::gpu_loader;

namespace {

// Save/clear/restore the four policy env vars around each test so ambient
// LS_LOADER_* settings (or other tests) can't leak into the assertions.
class LoaderPolicyParams : public ::testing::Test {
 protected:
  static constexpr const char* kVars[4] = {
      "LS_LOADER_FREQ_W", "LS_LOADER_FREQ_DECAY",
      "LS_LOADER_REUSE_W", "LS_LOADER_REUSE_TAU"};
  void SetUp() override {
    for (int i = 0; i < 4; ++i) {
      const char* v = std::getenv(kVars[i]);
      saved_[i] = v ? std::string(v) : std::string();
      had_[i]   = (v != nullptr);
      ::unsetenv(kVars[i]);
    }
  }
  void TearDown() override {
    for (int i = 0; i < 4; ++i) {
      if (had_[i]) ::setenv(kVars[i], saved_[i].c_str(), 1);
      else         ::unsetenv(kVars[i]);
    }
  }
 private:
  std::string saved_[4];
  bool had_[4] = {};
};

TEST_F(LoaderPolicyParams, RoundTrip) {
  gl::PolicyParams p;
  p.freq_w     = 45.0;
  p.freq_decay = 0.05;
  p.reuse_w    = 4000.0;
  p.reuse_tau  = 150.0;
  p.epoch      = "unit-test-epoch";
  const gl::PolicyParams q =
      gl::policy_params_from_json_string(gl::to_json_string(p));
  EXPECT_EQ(p, q);
}

TEST_F(LoaderPolicyParams, EmptyObjectYieldsBuiltinDefaults) {
  const gl::PolicyParams p = gl::policy_params_from_json_string("{}");
  EXPECT_EQ(p, gl::PolicyParams{});  // 60 / 0.1 / 2000 / 300, epoch ""
  EXPECT_DOUBLE_EQ(p.freq_w, 60.0);
  EXPECT_DOUBLE_EQ(p.freq_decay, 0.1);
  EXPECT_DOUBLE_EQ(p.reuse_w, 2000.0);
  EXPECT_DOUBLE_EQ(p.reuse_tau, 300.0);
}

TEST_F(LoaderPolicyParams, PartialArtifactDefaultsMissingFields) {
  const gl::PolicyParams p =
      gl::policy_params_from_json_string(R"({"freq_w": 45, "reuse_tau": 600})");
  EXPECT_DOUBLE_EQ(p.freq_w, 45.0);      // integers coerce to double
  EXPECT_DOUBLE_EQ(p.freq_decay, 0.1);   // absent -> default
  EXPECT_DOUBLE_EQ(p.reuse_w, 2000.0);   // absent -> default
  EXPECT_DOUBLE_EQ(p.reuse_tau, 600.0);
}

TEST_F(LoaderPolicyParams, ExtraKeysIgnored) {
  // The deploy_fit artifact carries _comment + a provenance block; the loader
  // must accept (and ignore) them.
  const gl::PolicyParams p = gl::policy_params_from_json_string(R"({
    "_comment": "x", "epoch": "e2", "freq_w": 60.0, "freq_decay": 0.1,
    "reuse_w": 2000.0, "reuse_tau": 300.0,
    "provenance": {"sim_hit": 0.73, "keeper_fingerprint": null}
  })");
  EXPECT_EQ(p.epoch, "e2");
  EXPECT_DOUBLE_EQ(p.freq_w, 60.0);
}

TEST_F(LoaderPolicyParams, FrozenEpoch2FixtureParses) {
  // Decision-neutrality anchor: the frozen epoch-2 fixture carries the built-in
  // defaults — loading it must reproduce them exactly.
  const gl::PolicyParams p = gl::load_policy_params(
      LAYERSTORM_SOURCE_DIR
      "/tests/integration/loader_offline_sim/fixtures/2026-07-18_epoch2/"
      "policy_params_reef_4gpu.json");
  EXPECT_DOUBLE_EQ(p.freq_w, 60.0);
  EXPECT_DOUBLE_EQ(p.freq_decay, 0.1);
  EXPECT_DOUBLE_EQ(p.reuse_w, 2000.0);
  EXPECT_DOUBLE_EQ(p.reuse_tau, 300.0);
  EXPECT_FALSE(p.epoch.empty());
}

TEST_F(LoaderPolicyParams, MalformedFailsLoud) {
  // Unparsable / wrong root type.
  EXPECT_THROW(gl::policy_params_from_json_string("not json"),
               std::runtime_error);
  EXPECT_THROW(gl::policy_params_from_json_string("[1,2]"),
               std::runtime_error);
  // Wrong-typed fields.
  EXPECT_THROW(gl::policy_params_from_json_string(R"({"freq_w": "sixty"})"),
               std::runtime_error);
  EXPECT_THROW(gl::policy_params_from_json_string(R"({"reuse_w": null})"),
               std::runtime_error);
  EXPECT_THROW(gl::policy_params_from_json_string(R"({"epoch": 3})"),
               std::runtime_error);
  // Out-of-range values (artifacts fail loud; only env overrides clamp).
  EXPECT_THROW(gl::policy_params_from_json_string(R"({"freq_w": -5})"),
               std::runtime_error);
  EXPECT_THROW(gl::policy_params_from_json_string(R"({"freq_decay": 0})"),
               std::runtime_error);
  EXPECT_THROW(gl::policy_params_from_json_string(R"({"reuse_w": -1})"),
               std::runtime_error);
  EXPECT_THROW(gl::policy_params_from_json_string(R"({"reuse_tau": 0})"),
               std::runtime_error);
}

TEST_F(LoaderPolicyParams, MissingFileFailsLoud) {
  EXPECT_THROW(gl::load_policy_params("/nonexistent/policy_params.json"),
               std::runtime_error);
}

TEST_F(LoaderPolicyParams, EnvOverridesTakePrecedenceOverArtifactValues) {
  gl::PolicyParams p = gl::policy_params_from_json_string(
      R"({"freq_w": 45, "freq_decay": 0.05, "reuse_w": 4000, "reuse_tau": 150})");
  // No env set -> no override, artifact values stand.
  EXPECT_FALSE(gl::apply_policy_env_overrides(p));
  EXPECT_DOUBLE_EQ(p.freq_w, 45.0);
  // Partial env: only the set fields override.
  ::setenv("LS_LOADER_FREQ_W", "90", 1);
  ::setenv("LS_LOADER_REUSE_TAU", "600", 1);
  EXPECT_TRUE(gl::apply_policy_env_overrides(p));
  EXPECT_DOUBLE_EQ(p.freq_w, 90.0);        // env wins
  EXPECT_DOUBLE_EQ(p.freq_decay, 0.05);    // artifact value kept
  EXPECT_DOUBLE_EQ(p.reuse_w, 4000.0);     // artifact value kept
  EXPECT_DOUBLE_EQ(p.reuse_tau, 600.0);    // env wins
}

TEST_F(LoaderPolicyParams, EnvOverrideClampsMatchHistoricalSemantics) {
  gl::PolicyParams p;
  ::setenv("LS_LOADER_FREQ_W", "-5", 1);      // clamp -> 0 (disables)
  ::setenv("LS_LOADER_FREQ_DECAY", "0", 1);   // bad -> default 0.1
  ::setenv("LS_LOADER_REUSE_W", "-1", 1);     // clamp -> 0 (disables)
  ::setenv("LS_LOADER_REUSE_TAU", "-3", 1);   // bad -> default 300
  EXPECT_TRUE(gl::apply_policy_env_overrides(p));
  EXPECT_DOUBLE_EQ(p.freq_w, 0.0);
  EXPECT_DOUBLE_EQ(p.freq_decay, 0.1);
  EXPECT_DOUBLE_EQ(p.reuse_w, 0.0);
  EXPECT_DOUBLE_EQ(p.reuse_tau, 300.0);
}

TEST_F(LoaderPolicyParams, EmptyStringEnvIsUnset) {
  gl::PolicyParams p;
  ::setenv("LS_LOADER_FREQ_W", "", 1);
  EXPECT_FALSE(gl::apply_policy_env_overrides(p));
  EXPECT_DOUBLE_EQ(p.freq_w, 60.0);
}

}  // namespace
