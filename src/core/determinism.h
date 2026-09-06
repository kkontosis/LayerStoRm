// Determinism superflags — TWO flags, one strict hierarchy.
//
// P-29 step 14 (GF3 decode-speed campaign), 2026-09-04.  Two properties get conflated as
// "deterministic", and each gets its own flag:
//
//   FLAG 1 — RUN-TO-RUN DETERMINISM
//     env LS_DETERMINISTIC / config compute.deterministic
//     Same binary, same config, same input => byte-identical output every
//     run.  Forces the determinism ENFORCERS on (DET-REDUCE,
//     DET-EP-COMBINE) and the genuinely nondeterministic knobs off
//     (SpecRoundGovernor).  Trajectory-forking-but-STABLE optimizations
//     survive this flag — LS_TQ_SPLITKV is the worked example: two full
//     runs byte-identical (P-29 step 9), so flag 1 leaves it alone.
//
//   FLAG 2 — REFERENCE-TRAJECTORY IDENTITY
//     env LS_REFERENCE_TRAJECTORY_IDENTITY /
//     config compute.reference_trajectory_identity
//     The stronger property: output matches the CANONICAL trajectory for
//     this configuration — the one goldens, identity shas and A/B
//     baselines are pinned to.  Forces every trajectory-forking knob to
//     its canonical state (LS_TQ_SPLITKV=0, LS_SNAPMLA_FP8_DECODE=0,
//     sub-grid mid-edge off, bf16 EP-combine payload, and refuses contrary
//     pins of the trajectory-relevant knob set).
//
// THE HIERARCHY IS STRICT AND IMPLEMENTED AS ONE: reference identity
// IMPLIES run-to-run determinism (matching a deterministic reference
// exactly means you are deterministic), never the reverse.  Enabling
// flag 2 enables flag 1's forcings too; requesting flag 2 while explicitly
// suppressing flag 1 (LS_DETERMINISTIC=0) is REFUSED at boot.
//
// WHAT "REFERENCE" MEANS (the subtle part — read this before arguing):
// the reference is RELATIVE, not a global golden.  It is what THIS
// configuration produces with no flag-level shortcut forking it — the
// canonical numerics path for the chosen backend, accuracy tier and
// parallelism.  It is not a fixed sha (the reference legitimately moves:
// the OQ-8 truncation fix moved it, a granted default-flip + re-golden
// moves it again — update the registry row in the same commit), and it is
// not "one true backend" (TQ vs snapMLA is a configuration choice).  The
// guarantee is "no FLAG has moved you off your config's canonical
// trajectory", nothing more.
//
// Both flags leave PROVEN-bit-identical optimizations ON (slim kernel,
// decode graphs, KCOPY, MPOKE, ...) — these are make-it-reproducible
// switches, not make-it-slow switches.  Both default OFF.
//
// Composition: neither flag duplicates LAYERSTORM_DETERMINISTIC_EP_COMBINE
// — they FORCE it (env wins over config at every read site) and refuse
// loudly when a user pins any registered knob against the requested mode.
//
// Registry discipline: EVERY env knob that can affect the output trajectory
// or determinism MUST be registered here, classified on BOTH axes (a knob
// can be deterministic AND reference-forking — split-KV — which is why the
// axis field exists).  A unit test scans the tree for LS_*/LAYERSTORM_* env
// reads and fails on any unregistered knob; python/orchestrator/
// determinism.py mirrors the actionable rows and a pytest asserts parity.
//
// Design note: scratchpad/gf3_speed_saga/DETERMINISM_FLAG.md.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace layerstorm {
namespace core {
namespace determinism {

/// Which flag a knob's action belongs to (the two-axis classification).
enum class Axis {
    kRunToRun,   ///< acting under FLAG 1 (and, by the hierarchy, flag 2):
                 ///< the knob threatens run-to-run byte-identity
    kReference,  ///< acting under FLAG 2 only: the knob is run-to-run
                 ///< deterministic but forks the reference trajectory
};

/// Classification of one env knob w.r.t. the requested mode.
enum class KnobClass {
    kForceOn,     ///< enforcer — forced to `forced` (setenv)
    kForceOff,    ///< forking/nondeterministic — forced to `forced`
    kPinDefault,  ///< trajectory-relevant; canonical state IS the code default —
                  ///< nothing forced, but a contrary env pin is REFUSED
    kKeepOn,      ///< optimization PROVEN bit/byte-identical — deliberately untouched
    kNeutral,     ///< audited: diagnostics / placement / transport — cannot move
                  ///< the trajectory (or is dead under the forced states)
};

/// How a user env pin conflicts with the requested mode.
enum class ConflictKind {
    kNone,      ///< never conflicts
    kIfTruthy,  ///< conflicts when set non-empty and first char != '0'
    kIfFalsy,   ///< conflicts when set and first char == '0'
    kIfSet,     ///< conflicts when set non-empty at all
    kIfExact,   ///< conflicts when value == conflict_exact
};

struct Knob {
    const char* env;             ///< env var name
    Axis axis;                   ///< which flag this row acts under
    KnobClass cls;
    const char* forced;          ///< value setenv'd for kForceOn/kForceOff, else nullptr
    ConflictKind conflict;
    const char* conflict_exact;  ///< for kIfExact, else nullptr
    const char* evidence;        ///< one-line proof / measurement / TD citation
};

/// The full registry (single source of truth on the C++ side).
const std::vector<Knob>& registry();

/// The resolved mode.  kReferenceIdentity strictly includes kRunToRun.
enum class Mode {
    kOff,
    kRunToRun,           ///< flag 1 only
    kReferenceIdentity,  ///< flag 2 (implies flag 1)
};

/// Resolve the mode from the two config flags, with env overriding either
/// way per flag (LS_DETERMINISTIC, LS_REFERENCE_TRAJECTORY_IDENTITY: set &
/// first char != '0' -> on, '0' -> off, unset -> config).  Does NOT check
/// the hierarchy-suppression conflict — apply_or_throw does, loudly.
Mode mode_requested(bool config_deterministic, bool config_reference_identity);

/// Pure planning result — what apply would do under `mode`.
struct Plan {
    struct Action {
        std::string env;
        std::string value;      ///< value to force
        Axis axis;              ///< which flag forced it (for the boot log)
        std::string prior;      ///< previous env value ("" = unset)
        bool already_conforming;///< env was already set to a conforming value
    };
    std::vector<Action> forced;             ///< kForceOn/kForceOff actions
    std::vector<std::string> kept;          ///< kKeepOn knobs left enabled (names)
    std::vector<std::string> conflicts;     ///< human-readable refusal lines
    std::vector<std::string> notes;         ///< boot-log notes (build preconditions etc.)
};

/// Compute the plan for `mode` under `get` (a getenv-like callable).
/// Pure; no logging, no mutation.  kRunToRun applies only Axis::kRunToRun
/// rows; kReferenceIdentity applies ALL rows (the hierarchy).
Plan plan(Mode mode, const char* (*get)(const char*));

/// Apply the requested mode (if any): logs every forced knob (tagged with
/// the flag that forced it) and every note via spdlog "[determinism]",
/// setenv's the forced values, and THROWS std::runtime_error listing every
/// conflicting user pin — including the hierarchy-suppression conflict
/// (flag 2 requested while LS_DETERMINISTIC=0 pins flag 1 off) — instead of
/// silently overriding anything.  Call ONCE, as early as possible
/// (Engine::init_modules, right after config parse), before any
/// getenv-latching static runs.  No-op when neither flag is requested.
void apply_or_throw(bool config_deterministic, bool config_reference_identity);

}  // namespace determinism
}  // namespace core
}  // namespace layerstorm
