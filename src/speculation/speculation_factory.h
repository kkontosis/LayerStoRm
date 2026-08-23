// SpeculationMethod factory: dispatches config::SpeculationMethodType to
// concrete SpeculationMethod implementations.
//
// Mirrors make_attention_device (core/attention_device.h →
// compute/attention_device_factory.cpp): an explicit switch over the config
// enum, method baked into the concrete type at construction — no runtime
// dispatch or registry lookup after init.  A static self-registration
// registry was deliberately REJECTED: layerstorm_core is a STATIC library,
// so registrar objects in otherwise-unreferenced TUs get dropped by the
// linker (same failure class as the quant-registry clear_registry() pitfall)
// — an explicit switch is the house pattern and fails at compile time
// instead of silently at runtime.
//
// Adding a method (#16 MTP drops in here):
//   1. add the enum value to `speculation.method` in config/schema.json and
//      regen (`node tools/gen_config.mjs`),
//   2. implement SpeculationMethod in src/speculation/<method>.{h,cpp},
//   3. add the case in make_speculation_method (speculation_factory.cpp).

#pragma once

#include "speculation/speculation_method.h"

#include <memory>

namespace layerstorm::speculation {

/// Create + init the SpeculationMethod selected by `type`.
///   none          → nullptr (speculation subsystem entirely off — DEFAULT).
///   null          → NullSpeculationMethod (no-op stub; drafts nothing).
///   mtp           → MtpSpeculationMethod (#16 / GLM-25g) — MTP self-draft
///                   off the frozen trunk; init throws on configs/checkpoints
///                   without an MTP head (fail closed).
///   prompt_lookup → reserved: throws std::runtime_error until the concrete
///                   method lands — engine init fails closed rather than
///                   silently running without the requested speculation.
///   dspark        → reserved (DSP-1 config surface): throws until the
///                   DSparkSpeculationMethod lands in DSP-5 — same
///                   fail-closed contract as prompt_lookup.
/// init(ctx) is called on the method before it is returned; init failures
/// propagate as exceptions.
std::unique_ptr<SpeculationMethod> make_speculation_method(
    config::SpeculationMethodType type, const SpeculationInitContext& ctx);

/// Config dispatch helper (DSP-1): true when the config selects DSpark
/// (speculation.method == dspark).  DSpark draft-checkpoint weight loading
/// and VRAM budget accounting (DSP-2) key off this — the speculation-side
/// analog of ModelConfig::has_mtp() (DSpark is drafter-checkpoint-driven,
/// not model-intrinsic, so the helper lives here, not on ModelConfig).
bool has_dspark(const config::Config& cfg);

/// Create a NullSpeculationMethod directly (unit tests; mirrors
/// make_null_attention_device).  No init context required.
std::unique_ptr<SpeculationMethod> make_null_speculation_method();

/// Stable name for a method enum value (logging; matches the schema strings).
const char* speculation_method_name(config::SpeculationMethodType type);

}  // namespace layerstorm::speculation
