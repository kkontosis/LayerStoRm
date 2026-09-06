// Determinism superflags — registry + apply semantics (CPU-only).
//
// Two flags, strict hierarchy: LS_DETERMINISTIC (run-to-run determinism) <
// LS_REFERENCE_TRAJECTORY_IDENTITY (reference-trajectory identity, implies
// the former).  The companion pytest (tests/unit/test_determinism_superflag
// .py) carries the tree scan (no LS_*/LAYERSTORM_* knob may exist
// unregistered) and the C++<->Python registry parity check; this file locks
// the C++ semantics: registry well-formedness, per-mode/axis plan() logic,
// each ConflictKind, the hierarchy (incl. its suppression refusal), and
// apply_or_throw's loud refusal.

#include "core/determinism.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <string>

namespace det = layerstorm::core::determinism;

namespace {

// Injectable env for plan(): a static map behind a getenv-shaped function.
std::map<std::string, std::string>& fake_env() {
    static std::map<std::string, std::string> m;
    return m;
}
const char* fake_get(const char* name) {
    auto it = fake_env().find(name);
    return it == fake_env().end() ? nullptr : it->second.c_str();
}

// RAII save/restore for REAL process env vars touched by apply tests.
class EnvGuard {
  public:
    explicit EnvGuard(std::initializer_list<const char*> names) {
        for (const char* n : names) {
            const char* v = std::getenv(n);
            saved_[n] = v ? std::optional<std::string>(v) : std::nullopt;
        }
    }
    ~EnvGuard() {
        for (auto& [n, v] : saved_) {
            if (v) ::setenv(n.c_str(), v->c_str(), 1);
            else   ::unsetenv(n.c_str());
        }
    }
  private:
    std::map<std::string, std::optional<std::string>> saved_;
};

bool has_conflict_mentioning(const det::Plan& p, const std::string& env_name) {
    for (const auto& c : p.conflicts)
        if (c.find(env_name) != std::string::npos) return true;
    return false;
}

bool forces(const det::Plan& p, const std::string& env_name) {
    for (const auto& a : p.forced)
        if (a.env == env_name) return true;
    return false;
}

const det::Knob* find(const char* env_name) {
    for (const auto& k : det::registry())
        if (std::string(k.env) == env_name) return &k;
    return nullptr;
}

}  // namespace

// ── Registry well-formedness ────────────────────────────────────────────────

TEST(DeterminismRegistry, EntriesAreUniqueAndEvidenced) {
    std::set<std::string> seen;
    for (const auto& k : det::registry()) {
        EXPECT_TRUE(seen.insert(k.env).second) << "duplicate entry: " << k.env;
        ASSERT_NE(k.evidence, nullptr) << k.env;
        EXPECT_GT(std::string(k.evidence).size(), 10u)
            << k.env << " needs a real evidence citation";
    }
    EXPECT_GT(seen.size(), 150u) << "registry unexpectedly small";
}

TEST(DeterminismRegistry, ForcedValueIffForcingClass) {
    for (const auto& k : det::registry()) {
        const bool forcing = k.cls == det::KnobClass::kForceOn ||
                             k.cls == det::KnobClass::kForceOff;
        EXPECT_EQ(forcing, k.forced != nullptr) << k.env;
        if (k.conflict == det::ConflictKind::kIfExact)
            EXPECT_NE(k.conflict_exact, nullptr) << k.env;
        // Every forcing or pinned knob must be able to detect a contrary pin.
        if (forcing || k.cls == det::KnobClass::kPinDefault)
            EXPECT_NE(k.conflict, det::ConflictKind::kNone) << k.env;
    }
}

// The critical knobs the P-29 campaign has already measured MUST be present with the
// right axis+class — a hardcoded floor under the registry (belt to the
// pytest scan's braces).
TEST(DeterminismRegistry, CriticalKnobsRegisteredWithRightAxisAndClass) {
    struct Want {
        const char* env;
        det::Axis axis;
        det::KnobClass cls;
        const char* forced;
    };
    const Want wants[] = {
        {"LAYERSTORM_DETERMINISTIC_REDUCE", det::Axis::kRunToRun,
         det::KnobClass::kForceOn, "1"},
        {"LAYERSTORM_DETERMINISTIC_EP_COMBINE", det::Axis::kRunToRun,
         det::KnobClass::kForceOn, "1"},
        {"LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION", det::Axis::kReference,
         det::KnobClass::kForceOn, "bf16"},
        // THE exemplar: deterministic run-to-run but reference-forking.
        {"LS_TQ_SPLITKV", det::Axis::kReference,
         det::KnobClass::kForceOff, "0"},
        {"LS_SNAPMLA_FP8_DECODE", det::Axis::kReference,
         det::KnobClass::kForceOff, "0"},
        {"LS_ORCH_SUBGRID_MIDEDGE", det::Axis::kReference,
         det::KnobClass::kForceOff, "0"},
        {"LS_SPEC_GOVERNOR", det::Axis::kRunToRun,
         det::KnobClass::kForceOff, "0"},
        {"LS_PINNED_EXACT_WIDTHS", det::Axis::kReference,
         det::KnobClass::kPinDefault, nullptr},
        {"LS_MOE_RESIDENT_OVERLAP", det::Axis::kReference,
         det::KnobClass::kPinDefault, nullptr},
        {"LS_TQ_SPARSE_SLIM", det::Axis::kReference,
         det::KnobClass::kKeepOn, nullptr},
        {"LS_DECODE_CHAIN_GRAPH", det::Axis::kReference,
         det::KnobClass::kKeepOn, nullptr},
        {"LS_FFN_GRAPH_KCOPY", det::Axis::kReference,
         det::KnobClass::kKeepOn, nullptr},
        {"LS_MOE_META_CACHE", det::Axis::kReference,
         det::KnobClass::kKeepOn, nullptr},
    };
    for (const auto& w : wants) {
        const det::Knob* k = find(w.env);
        ASSERT_NE(k, nullptr) << w.env << " missing from registry";
        EXPECT_EQ(static_cast<int>(k->axis), static_cast<int>(w.axis)) << w.env;
        EXPECT_EQ(static_cast<int>(k->cls), static_cast<int>(w.cls)) << w.env;
        if (w.forced) {
            ASSERT_NE(k->forced, nullptr) << w.env;
            EXPECT_STREQ(k->forced, w.forced) << w.env;
        }
    }
}

// ── plan(): the hierarchy is the semantics ──────────────────────────────────

TEST(DeterminismPlan, OffModePlansNothing) {
    fake_env().clear();
    fake_env()["LS_TQ_SPLITKV"] = "1";
    det::Plan p = det::plan(det::Mode::kOff, &fake_get);
    EXPECT_TRUE(p.forced.empty());
    EXPECT_TRUE(p.conflicts.empty());
}

TEST(DeterminismPlan, RunToRunModeForcesOnlyRunToRunRows) {
    fake_env().clear();
    det::Plan p = det::plan(det::Mode::kRunToRun, &fake_get);
    EXPECT_TRUE(p.conflicts.empty());
    std::set<std::string> forced;
    for (const auto& a : p.forced) {
        forced.insert(a.env);
        EXPECT_EQ(static_cast<int>(a.axis),
                  static_cast<int>(det::Axis::kRunToRun)) << a.env;
    }
    const std::set<std::string> expect = {
        "LAYERSTORM_DETERMINISTIC_REDUCE",
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE",
        "LS_SPEC_GOVERNOR",
    };
    EXPECT_EQ(forced, expect);
    // Split-KV SURVIVES flag 1: not forced, and a =1 pin is NOT a conflict.
    fake_env()["LS_TQ_SPLITKV"] = "1";
    det::Plan p2 = det::plan(det::Mode::kRunToRun, &fake_get);
    EXPECT_FALSE(forces(p2, "LS_TQ_SPLITKV"));
    EXPECT_FALSE(has_conflict_mentioning(p2, "LS_TQ_SPLITKV"));
    // Same for the other reference-only rows: fp32 payload pin, FP8 decode.
    fake_env()["LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION"] = "fp32";
    fake_env()["LS_SNAPMLA_FP8_DECODE"] = "1";
    det::Plan p3 = det::plan(det::Mode::kRunToRun, &fake_get);
    EXPECT_TRUE(p3.conflicts.empty());
}

TEST(DeterminismPlan, ReferenceModeForcesAllForcingRowsOfBothAxes) {
    fake_env().clear();
    det::Plan p = det::plan(det::Mode::kReferenceIdentity, &fake_get);
    EXPECT_TRUE(p.conflicts.empty());
    std::set<std::string> forced;
    for (const auto& a : p.forced) forced.insert(a.env);
    std::set<std::string> expect;
    for (const auto& k : det::registry())
        if (k.forced) expect.insert(k.env);
    EXPECT_EQ(forced, expect);  // ALL forcing rows — the hierarchy.
    EXPECT_TRUE(forced.count("LS_TQ_SPLITKV"));
    EXPECT_TRUE(forced.count("LS_SPEC_GOVERNOR"));  // flag 1's rows included
    // Proven-identical optimizations are kept, never forced off.
    EXPECT_FALSE(forced.count("LS_TQ_SPARSE_SLIM"));
    bool slim_kept = false;
    for (const auto& n : p.kept) slim_kept |= (n == "LS_TQ_SPARSE_SLIM");
    EXPECT_TRUE(slim_kept);
    EXPECT_FALSE(p.notes.empty());  // DET-TOPK-TIES note + reference note
}

TEST(DeterminismPlan, ConformingPinIsNotAConflict) {
    fake_env().clear();
    fake_env()["LS_TQ_SPLITKV"] = "0";
    fake_env()["LS_SPEC_GOVERNOR"] = "0";
    fake_env()["LAYERSTORM_DETERMINISTIC_EP_COMBINE"] = "1";
    det::Plan p = det::plan(det::Mode::kReferenceIdentity, &fake_get);
    EXPECT_TRUE(p.conflicts.empty());
    int conforming = 0;
    for (const auto& a : p.forced)
        if (a.already_conforming) ++conforming;
    EXPECT_EQ(conforming, 3);
}

// ── plan(): every ConflictKind fires (and does not overfire) ────────────────

TEST(DeterminismPlan, TruthyPinOnForkOffKnobConflictsInReferenceMode) {
    fake_env().clear();
    fake_env()["LS_TQ_SPLITKV"] = "1";
    det::Plan p = det::plan(det::Mode::kReferenceIdentity, &fake_get);
    EXPECT_TRUE(has_conflict_mentioning(p, "LS_TQ_SPLITKV"));
    EXPECT_FALSE(forces(p, "LS_TQ_SPLITKV"));  // refused, not overridden
}

TEST(DeterminismPlan, FalsyPinOnEnforcerConflictsInBothModes) {
    fake_env().clear();
    fake_env()["LAYERSTORM_DETERMINISTIC_EP_COMBINE"] = "0";
    EXPECT_TRUE(has_conflict_mentioning(
        det::plan(det::Mode::kRunToRun, &fake_get),
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE"));
    EXPECT_TRUE(has_conflict_mentioning(
        det::plan(det::Mode::kReferenceIdentity, &fake_get),
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE"));
}

TEST(DeterminismPlan, ExactPinSemantics) {
    // PRECISION=fp32 conflicts in reference mode; bf16 conforms.
    fake_env().clear();
    fake_env()["LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION"] = "fp32";
    EXPECT_TRUE(has_conflict_mentioning(
        det::plan(det::Mode::kReferenceIdentity, &fake_get),
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION"));
    fake_env()["LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION"] = "bf16";
    EXPECT_FALSE(has_conflict_mentioning(
        det::plan(det::Mode::kReferenceIdentity, &fake_get),
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION"));
    // LS_SNAPMLA_FP8_DECODE: since the P-29 step-14 OQ-7 default flip its parser
    // is default-ON with "anything but exactly '0' = ON", so any truthy pin
    // (="1", ="true") is a contrary ON pin and must be refused; ="0"
    // conforms with the forced canonical value and must not be.
    fake_env().clear();
    fake_env()["LS_SNAPMLA_FP8_DECODE"] = "true";
    EXPECT_TRUE(has_conflict_mentioning(
        det::plan(det::Mode::kReferenceIdentity, &fake_get),
        "LS_SNAPMLA_FP8_DECODE"));
    fake_env()["LS_SNAPMLA_FP8_DECODE"] = "1";
    EXPECT_TRUE(has_conflict_mentioning(
        det::plan(det::Mode::kReferenceIdentity, &fake_get),
        "LS_SNAPMLA_FP8_DECODE"));
    fake_env()["LS_SNAPMLA_FP8_DECODE"] = "0";
    EXPECT_FALSE(has_conflict_mentioning(
        det::plan(det::Mode::kReferenceIdentity, &fake_get),
        "LS_SNAPMLA_FP8_DECODE"));
}

TEST(DeterminismPlan, PinDefaultKnobs) {
    // kIfFalsy pin: exact-widths =0 conflicts (reference mode), =1 conforms.
    fake_env().clear();
    fake_env()["LS_PINNED_EXACT_WIDTHS"] = "0";
    EXPECT_TRUE(has_conflict_mentioning(
        det::plan(det::Mode::kReferenceIdentity, &fake_get),
        "LS_PINNED_EXACT_WIDTHS"));
    // ...and does NOT conflict under flag 1 (it is a reference-axis pin).
    EXPECT_FALSE(has_conflict_mentioning(
        det::plan(det::Mode::kRunToRun, &fake_get), "LS_PINNED_EXACT_WIDTHS"));
    fake_env()["LS_PINNED_EXACT_WIDTHS"] = "1";
    EXPECT_FALSE(has_conflict_mentioning(
        det::plan(det::Mode::kReferenceIdentity, &fake_get),
        "LS_PINNED_EXACT_WIDTHS"));
    // kIfSet pin: any explicit GG strategy pin conflicts in reference mode.
    fake_env().clear();
    fake_env()["LS_GG_FORCE"] = "mmq";
    EXPECT_TRUE(has_conflict_mentioning(
        det::plan(det::Mode::kReferenceIdentity, &fake_get), "LS_GG_FORCE"));
    // The run-to-run-axis pin (CPU expert, run-to-run unproven) fires under
    // BOTH modes — the safe side of an unproven knob.
    fake_env().clear();
    fake_env()["LS_CPU_EXPERT"] = "1";
    EXPECT_TRUE(has_conflict_mentioning(
        det::plan(det::Mode::kRunToRun, &fake_get), "LS_CPU_EXPERT"));
    EXPECT_TRUE(has_conflict_mentioning(
        det::plan(det::Mode::kReferenceIdentity, &fake_get), "LS_CPU_EXPERT"));
}

TEST(DeterminismPlan, NeutralKnobsNeverConflictNorForce) {
    fake_env().clear();
    fake_env()["LS_PERF_TRACE"] = "16000000";
    fake_env()["LS_LOADER_SHADOW"] = "0";
    fake_env()["LS_KDA_XRAY"] = "1";
    det::Plan p = det::plan(det::Mode::kReferenceIdentity, &fake_get);
    EXPECT_TRUE(p.conflicts.empty());
    for (const auto& a : p.forced) {
        EXPECT_NE(a.env, "LS_PERF_TRACE");
        EXPECT_NE(a.env, "LS_LOADER_SHADOW");
        EXPECT_NE(a.env, "LS_KDA_XRAY");
    }
}

TEST(DeterminismPlan, MultipleConflictsAllReported) {
    fake_env().clear();
    fake_env()["LS_TQ_SPLITKV"] = "1";
    fake_env()["LS_SNAPMLA_FP8_DECODE"] = "1";
    fake_env()["LAYERSTORM_DETERMINISTIC_REDUCE"] = "0";
    det::Plan p = det::plan(det::Mode::kReferenceIdentity, &fake_get);
    EXPECT_EQ(p.conflicts.size(), 3u);
}

// ── mode_requested: per-flag env override + the hierarchy ordering ──────────

TEST(DeterminismMode, EnvOverridesConfigEitherWayPerFlag) {
    EnvGuard g({"LS_DETERMINISTIC", "LS_REFERENCE_TRAJECTORY_IDENTITY"});
    ::unsetenv("LS_DETERMINISTIC");
    ::unsetenv("LS_REFERENCE_TRAJECTORY_IDENTITY");
    EXPECT_EQ(det::mode_requested(false, false), det::Mode::kOff);
    EXPECT_EQ(det::mode_requested(true, false), det::Mode::kRunToRun);
    EXPECT_EQ(det::mode_requested(false, true), det::Mode::kReferenceIdentity);
    EXPECT_EQ(det::mode_requested(true, true), det::Mode::kReferenceIdentity);
    ::setenv("LS_DETERMINISTIC", "1", 1);
    EXPECT_EQ(det::mode_requested(false, false), det::Mode::kRunToRun);
    ::setenv("LS_DETERMINISTIC", "0", 1);
    EXPECT_EQ(det::mode_requested(true, false), det::Mode::kOff);
    ::unsetenv("LS_DETERMINISTIC");
    ::setenv("LS_REFERENCE_TRAJECTORY_IDENTITY", "1", 1);
    EXPECT_EQ(det::mode_requested(false, false), det::Mode::kReferenceIdentity);
    ::setenv("LS_REFERENCE_TRAJECTORY_IDENTITY", "0", 1);
    EXPECT_EQ(det::mode_requested(false, true), det::Mode::kOff);
    // Flag 2 off by env does not cancel an independently requested flag 1.
    EXPECT_EQ(det::mode_requested(true, true), det::Mode::kRunToRun);
}

// ── apply_or_throw: real env, hierarchy, loud refusal, forced states ────────

TEST(DeterminismApply, OffModeIsANoop) {
    EnvGuard g({"LS_DETERMINISTIC", "LS_REFERENCE_TRAJECTORY_IDENTITY",
                "LS_TQ_SPLITKV"});
    ::unsetenv("LS_DETERMINISTIC");
    ::unsetenv("LS_REFERENCE_TRAJECTORY_IDENTITY");
    ::setenv("LS_TQ_SPLITKV", "1", 1);  // would conflict in reference mode
    EXPECT_NO_THROW(det::apply_or_throw(false, false));
    EXPECT_STREQ(std::getenv("LS_TQ_SPLITKV"), "1");  // untouched
}

TEST(DeterminismApply, RunToRunModeLeavesSplitKvAlone) {
    EnvGuard g({"LS_DETERMINISTIC", "LS_REFERENCE_TRAJECTORY_IDENTITY",
                "LS_TQ_SPLITKV", "LS_SPEC_GOVERNOR",
                "LAYERSTORM_DETERMINISTIC_REDUCE",
                "LAYERSTORM_DETERMINISTIC_EP_COMBINE"});
    ::unsetenv("LS_REFERENCE_TRAJECTORY_IDENTITY");
    ::unsetenv("LS_SPEC_GOVERNOR");
    ::unsetenv("LAYERSTORM_DETERMINISTIC_REDUCE");
    ::unsetenv("LAYERSTORM_DETERMINISTIC_EP_COMBINE");
    ::setenv("LS_DETERMINISTIC", "1", 1);
    ::setenv("LS_TQ_SPLITKV", "1", 1);  // deterministic-but-forking: allowed
    EXPECT_NO_THROW(det::apply_or_throw(false, false));
    EXPECT_STREQ(std::getenv("LS_TQ_SPLITKV"), "1");           // survives
    EXPECT_STREQ(std::getenv("LS_SPEC_GOVERNOR"), "0");        // forced
    EXPECT_STREQ(std::getenv("LAYERSTORM_DETERMINISTIC_REDUCE"), "1");
    EXPECT_STREQ(std::getenv("LAYERSTORM_DETERMINISTIC_EP_COMBINE"), "1");
}

TEST(DeterminismApply, ReferenceModeForcesEverything) {
    EnvGuard g({"LS_DETERMINISTIC", "LS_REFERENCE_TRAJECTORY_IDENTITY",
                "LS_TQ_SPLITKV", "LS_SNAPMLA_FP8_DECODE", "LS_SPEC_GOVERNOR",
                "LS_ORCH_SUBGRID_MIDEDGE", "LAYERSTORM_DETERMINISTIC_REDUCE",
                "LAYERSTORM_DETERMINISTIC_EP_COMBINE",
                "LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION"});
    for (const char* n : {"LS_DETERMINISTIC", "LS_TQ_SPLITKV",
                          "LS_SNAPMLA_FP8_DECODE", "LS_SPEC_GOVERNOR",
                          "LS_ORCH_SUBGRID_MIDEDGE",
                          "LAYERSTORM_DETERMINISTIC_REDUCE",
                          "LAYERSTORM_DETERMINISTIC_EP_COMBINE",
                          "LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION"})
        ::unsetenv(n);
    ::setenv("LS_REFERENCE_TRAJECTORY_IDENTITY", "1", 1);
    EXPECT_NO_THROW(det::apply_or_throw(false, false));
    EXPECT_STREQ(std::getenv("LAYERSTORM_DETERMINISTIC_REDUCE"), "1");
    EXPECT_STREQ(std::getenv("LAYERSTORM_DETERMINISTIC_EP_COMBINE"), "1");
    EXPECT_STREQ(std::getenv("LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION"),
                 "bf16");
    EXPECT_STREQ(std::getenv("LS_TQ_SPLITKV"), "0");
    EXPECT_STREQ(std::getenv("LS_SNAPMLA_FP8_DECODE"), "0");
    EXPECT_STREQ(std::getenv("LS_SPEC_GOVERNOR"), "0");
    EXPECT_STREQ(std::getenv("LS_ORCH_SUBGRID_MIDEDGE"), "0");
}

TEST(DeterminismApply, RefusesHierarchySuppression) {
    // Requesting flag 2 while explicitly suppressing flag 1 is impossible.
    EnvGuard g({"LS_DETERMINISTIC", "LS_REFERENCE_TRAJECTORY_IDENTITY"});
    ::setenv("LS_REFERENCE_TRAJECTORY_IDENTITY", "1", 1);
    ::setenv("LS_DETERMINISTIC", "0", 1);
    try {
        det::apply_or_throw(false, false);
        FAIL() << "expected throw";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("IMPLIES"), std::string::npos);
        EXPECT_NE(what.find("LS_DETERMINISTIC=0"), std::string::npos);
    }
}

TEST(DeterminismApply, RefusesLoudlyOnConflictingPinWithoutOverridingIt) {
    EnvGuard g({"LS_DETERMINISTIC", "LS_REFERENCE_TRAJECTORY_IDENTITY",
                "LS_TQ_SPLITKV"});
    ::unsetenv("LS_DETERMINISTIC");
    ::setenv("LS_REFERENCE_TRAJECTORY_IDENTITY", "1", 1);
    ::setenv("LS_TQ_SPLITKV", "1", 1);
    EXPECT_THROW(det::apply_or_throw(false, false), std::runtime_error);
    // Refusal, not silent override: the user's pin is preserved.
    EXPECT_STREQ(std::getenv("LS_TQ_SPLITKV"), "1");
    try {
        det::apply_or_throw(false, false);
        FAIL() << "expected throw";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("LS_TQ_SPLITKV"), std::string::npos);
        EXPECT_NE(what.find("REFUSING"), std::string::npos);
    }
}

// Config-driven arming (no env) forces per the requested flag.
TEST(DeterminismApply, ConfigFlagsArmWithoutEnv) {
    EnvGuard g({"LS_DETERMINISTIC", "LS_REFERENCE_TRAJECTORY_IDENTITY",
                "LS_TQ_SPLITKV", "LS_SPEC_GOVERNOR"});
    for (const char* n : {"LS_DETERMINISTIC",
                          "LS_REFERENCE_TRAJECTORY_IDENTITY",
                          "LS_TQ_SPLITKV", "LS_SPEC_GOVERNOR"})
        ::unsetenv(n);
    // compute.deterministic=true: flag 1 only — split-KV untouched.
    EXPECT_NO_THROW(det::apply_or_throw(true, false));
    EXPECT_EQ(std::getenv("LS_TQ_SPLITKV"), nullptr);
    EXPECT_STREQ(std::getenv("LS_SPEC_GOVERNOR"), "0");
    // compute.reference_trajectory_identity=true: flag 2 — forced.
    EXPECT_NO_THROW(det::apply_or_throw(false, true));
    EXPECT_STREQ(std::getenv("LS_TQ_SPLITKV"), "0");
}
