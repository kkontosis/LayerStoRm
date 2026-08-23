#pragma once
// I8 loader policy weights (P-26 unified deployment fit) — the freq/reuse
// closed-loop-tuned knobs as a loadable, provenance-stamped artifact.
//
// The four weights are BLACK-BOX-optimized (sim grid + keeper confirm), never
// regressed (INV-LOADER-OBJECTIVE-MYOPIC); the artifact is written by
// tools/loader_fit/deploy_fit.py (schema = the frozen epoch fixture
// loader_offline_sim/fixtures/2026-07-18_epoch2/policy_params_reef_4gpu.json).
// Precedence in BOTH consumers (engine dispatcher init_loader_from_env and the
// keeper's test-side ReefOrch): explicit env vars (LS_LOADER_FREQ_W etc.) >
// LS_LOADER_POLICY artifact > built-in defaults (the struct initializers below
// — the P-25 epoch-2 plateau optimum 60/0.1/2000/300).
//
// Parse contract (documented + unit-tested, loader_policy_params_test.cpp):
//   * absent optional fields        -> built-in default (partial artifacts OK);
//   * unparsable JSON, non-object, wrong-typed or out-of-range values
//     (freq_w<0, reuse_w<0, freq_decay<=0, reuse_tau<=0) -> throw
//     std::runtime_error (fail loud — a malformed fitted artifact means a bad
//     fit, never silently fall back).
//
// CUDA-free, pure data + JSON (house style: manual builders, see
// loader_constants.cpp). Lives in layerstorm_core (INV-GPU-1).
// Ticket: spec/tickets/P-26_I8_DEPLOYMENT_FIT.md.
#include <string>

namespace layerstorm::gpu_loader {

// The I8 policy weights. Defaults = the landed engine defaults (dispatcher
// members / ReefOrch initializers) — the no-artifact behavior is unchanged.
struct PolicyParams {
  // Decayed-frequency eviction protection (per routed access f = f*e^{-d}+1,
  // stamped as recency bonus w*f on every resident copy). w<=0 disables.
  double freq_w     = 60.0;
  double freq_decay = 0.1;
  // place_cons cross-token reuse reward w/(1 + age/tau). w<=0 disables.
  double reuse_w    = 2000.0;
  double reuse_tau  = 300.0;   // age half-scale, board layer-visits
  // Provenance passthrough (the artifact's epoch tag; empty = built-in).
  std::string epoch;
  bool operator==(const PolicyParams&) const = default;
};

// JSON serialization (manual builders; matches loader_constants.cpp style).
// Extra keys in the artifact (e.g. the provenance block) are ignored on parse.
std::string  to_json_string(const PolicyParams&, int indent = 2);
PolicyParams policy_params_from_json_string(const std::string&);
// Throws std::runtime_error on missing/unreadable file or malformed content.
PolicyParams load_policy_params(const std::string& path);

// Apply the explicit LS_LOADER_FREQ_W / LS_LOADER_FREQ_DECAY / LS_LOADER_REUSE_W
// / LS_LOADER_REUSE_TAU env overrides on top of `p` (diagnostic arms — the
// historical env semantics, clamps included: negative weights -> 0,
// non-positive decay/tau -> their defaults; empty-string env = unset).
// Returns true iff at least one override was applied.
bool apply_policy_env_overrides(PolicyParams& p);

}  // namespace layerstorm::gpu_loader
