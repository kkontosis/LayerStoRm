// MoE weight-quant route selection — the single source of truth for the
// GGUF/NVFP4/FP8 projection-route derivation shared by the single-shot
// driver (moe_driver.cpp) and the chunked big-batch sibling (moe_big.cpp).
//
// Dedup provenance (TD-MOE-BY-MODEL-SPLIT): both files carried a verbatim
// copy of this block under a "mirrors dispatch_moe_internal exactly; keep
// the two in sync" comment — this struct + builder replace both copies
// (codec-flavored axis; layering precedent: daemon/attention/kv_codec.h).
//
// Contents (GG-5b/GG-5c/GG-9):
//   - use_fp8 / use_gguf weight-quant mode flags + the fail-loud validation
//     (unsupported quant, missing Deps::gguf_quant interface);
//   - the GGUF compute strategy (int vs dequant) from live config;
//   - per-layer routed k-quant types (mixed "XL" GGUF) with uniform-table
//     fallback (GG-9);
//   - per-layer in-slot projection offsets (TD-GG9-EXPERT-PERLAYER-OFFSETS:
//     CacheEntry offsets are the GLOBAL-max layout — a smaller per-layer
//     pack must be read at ITS OWN offsets; gate is always at 0);
//   - the ordinal-preserving model::GgufKQuantType → compute::GgufQuantType
//     cast used by the dense/shared GEMM sites for their OWN types (GG-5c).

#pragma once

#include <cstdint>

#include "daemon/command_dispatcher.h"
#include "core/expert_device.h"
#include "core/memory/expert_cache.h"
#include "model/quantization/gguf_kquant.h"

namespace layerstorm::daemon {

struct MoeQuantRoutes {
    bool use_fp8 = false;
    bool use_gguf = false;
    compute::GgufGemmStrategy gguf_strategy =
        compute::GgufGemmStrategy::int_strategy;
    // GG-9: THIS layer's routed per-projection k-quant types (Q4_K
    // placeholder when !use_gguf — never read by the non-GGUF GEMMs).
    compute::GgufQuantType gguf_gate_type = compute::GgufQuantType::Q4_K;
    compute::GgufQuantType gguf_up_type   = compute::GgufQuantType::Q4_K;
    compute::GgufQuantType gguf_down_type = compute::GgufQuantType::Q4_K;
    // GG-9: per-layer in-slot offsets (0 when !use_gguf).
    int64_t gguf_gate_off = 0;
    int64_t gguf_up_off   = 0;
    int64_t gguf_down_off = 0;

    // GG-5c: ordinal-preserving cast for the dense/shared structs' OWN
    // per-projection model::GgufKQuantType (may differ from the routed
    // types on a mixed-quant GGUF).
    static compute::GgufQuantType to_gguf_compute(model::GgufKQuantType t) {
        return static_cast<compute::GgufQuantType>(static_cast<int>(t));
    }

    // Per-projection in-slot offset: GGUF uses the per-layer offsets above;
    // all other formats keep the CacheEntry's uniform global offsets.
    int64_t gate_off(const memory::CacheEntry* e) const {
        return use_gguf ? gguf_gate_off : e->gate_offset; }
    int64_t up_off(const memory::CacheEntry* e) const {
        return use_gguf ? gguf_up_off : e->up_offset; }
    int64_t down_off(const memory::CacheEntry* e) const {
        return use_gguf ? gguf_down_off : e->down_offset; }
};

/// Derive the quant routes for one MoE dispatch. Aborts (fail loud) on an
/// unsupported weight quant or a GGUF config without the per-projection
/// interface — `ctx` prefixes those messages (the former per-file copies
/// used their function names). `intermediate`/`hidden` mirror the
/// ExpertShape the cache sizes gate_bytes_ with (full moe_intermediate,
/// not TP-sharded, matching engine.cpp's expert_shape).
MoeQuantRoutes build_moe_quant_routes(const CommandDispatcher::Deps& deps,
                                      uint32_t layer_idx, int intermediate,
                                      int hidden, const char* ctx);

}  // namespace layerstorm::daemon
