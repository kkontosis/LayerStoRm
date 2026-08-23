#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "model/quantization/quant_interface.h"

// ── GGUF k-quant expert metadata (host-only) ─────────────────────────────────
//
// Host-side byte-accounting for GGUF-quantized expert weights. This file is
// PURE host C++ — it must NOT include any CUDA/kernel header (INV-GPU-1). The
// per-type block constants below MIRROR the kernel's
// `layerstorm::compute::gguf_block_bytes(GgufType)` /
// `gguf_block_values(GgufType)` in
// deps/LayerStoRmGemmKernels/.../gguf/gguf_dequant_gemm.h, which pulls in
// <cuda_runtime.h> and therefore cannot be included here. See TD-GGUF-BLOCK-
// CONST-MIRROR in spec/TECH_DEBT.md.
//
// GGUF k-quants pack a whole row as `[N, (K/QK) * block_bytes]` uint8, where
// each super-block of QK values carries its scales/mins INTERNALLY. There is no
// separate scale region, so weight_bytes == total_bytes and replicated == 0.

namespace layerstorm::model {

// ── GGUF weight quantization types ──────────────────────────────────────────
// Enum values match layerstorm::compute::GgufType (Q2_K=0 .. Q8_0=5) so the
// two can be cast across the engine/kernel boundary by GG-4/GG-5/GG-6.

enum class GgufKQuantType {
    Q2_K = 0,
    Q3_K = 1,
    Q4_K = 2,
    Q5_K = 3,
    Q6_K = 4,
    Q8_0 = 5,
    MXFP4 = 6,  // OCP MX e2m1 + E8M0 block scale (V4 QAT routed experts)
};

inline constexpr int kNumGgufKQuantTypes = 7;

// ── Per-type block constants ────────────────────────────────────────────────
// block_bytes: packed bytes of one super-block (scales/mins included).
// qk:          weight values per super-block (256 for k-quants, 32 for Q8_0).
// Effective bits/weight = 8 * block_bytes / qk (fractional for k-quants).

struct GgufBlockSpec {
    int block_bytes;
    int qk;
};

namespace gguf {

// Indexed by static_cast<int>(GgufKQuantType). Keep in sync with the enum order.
inline constexpr std::array<GgufBlockSpec, kNumGgufKQuantTypes> kBlockSpecs = {{
    {84, 256},   // Q2_K
    {110, 256},  // Q3_K
    {144, 256},  // Q4_K
    {176, 256},  // Q5_K
    {210, 256},  // Q6_K
    {34, 32},    // Q8_0
    {17, 32},    // MXFP4 (1B E8M0 scale + 16B packed e2m1 = 4.25 bit/w)
}};

/// Packed bytes of one super-block for `type`.
constexpr int block_bytes(GgufKQuantType type) {
    return kBlockSpecs[static_cast<int>(type)].block_bytes;
}

/// Weight values per super-block for `type` (the QK divisor).
constexpr int block_values(GgufKQuantType type) {
    return kBlockSpecs[static_cast<int>(type)].qk;
}

/// Effective (fractional) bytes per stored weight element = block_bytes / qk.
/// e.g. Q4_K = 144/256 = 0.5625, Q6_K = 210/256 = 0.8203125, Q8_0 = 34/32.
constexpr double bytes_per_element(GgufKQuantType type) {
    return static_cast<double>(block_bytes(type))
         / static_cast<double>(block_values(type));
}

/// Human-readable name ("gguf_q4_k", …) for `type`.
std::string_view type_name(GgufKQuantType type);

/// Inverse of type_name(): map "gguf_q4_k" / "q4_k" / "Q4_K" → GgufKQuantType.
/// Throws std::runtime_error on an unrecognized name.
GgufKQuantType type_from_name(std::string_view name);

/// Map a config::WeightQuant uniform GGUF variant to its k-quant type.
/// Throws std::runtime_error for the generic `gguf` enum or any non-GGUF value.
GgufKQuantType type_from_weight_quant(config::WeightQuant wq);

/// True for any GGUF-family weight quant (the generic `gguf` sentinel or one of
/// the six uniform `gguf_qX_k`/`gguf_q8_0` variants). Single home for the
/// "is this a GGUF quant" predicate so call sites don't open-code the enum list.
bool is_gguf_weight_quant(config::WeightQuant wq);

// ── Reusable byte-size primitive (single source of truth) ───────────────────
// The `[out * (in/QK) * block_bytes]` formula lives ONLY here. GG-4 (attention)
// and GG-6 (loader) build on this. Requires in_features % QK == 0.

/// Packed bytes of an `[out_features, in_features]` GGUF k-quant weight.
/// @throws std::runtime_error if in_features is not a multiple of the type's QK.
int64_t gguf_packed_bytes(int64_t out_features, int64_t in_features,
                          GgufKQuantType type);

// ── Host MXFP4 dequant reference (pure host C++) ────────────────────────────
// Faithful port of ggml dequantize_row_mxfp4 + ggml_e8m0_to_fp32_half
// (ref/llama.cpp ggml/src/ggml-quants.c, ggml-impl.h; MIT License,
// Copyright (c) 2023-2026 The ggml authors — see THIRD_PARTY_NOTICES.md).
// Block = [1B E8M0 scale | 16B packed e2m1]; element j in [0,16)
// is the LOW nibble of qs[j], element j+16 the HIGH nibble. Used by the
// dequant-correctness golden and any host-side MXFP4 consumption.

/// E8M0 scale byte → fp32, pre-divided by 2 (un-doubles the e2m1 table).
float e8m0_to_fp32_half(uint8_t e);

/// Dequantize one 17-byte MXFP4 block (32 values) into out[32].
void dequant_mxfp4_block(const uint8_t* block, float* out);

/// Dequantize `n` values (n % 32 == 0) from packed MXFP4 into out.
void dequant_mxfp4_row(const uint8_t* packed, float* out, int64_t n);

}  // namespace gguf

// ── GgufQuantInterface ──────────────────────────────────────────────────────
// Per-projection mixed-quant QuantInterface for expert weights. Real GGUF
// models are per-tensor mixed, so each projection (gate/up/down) carries its
// own k-quant type. The uniform per-type registry singletons (gguf_q4_k, …) are
// just this interface constructed with all three projections the same type.

class GgufQuantInterface final : public QuantInterface {
public:
    /// Mixed construction: each projection sized by its own GGUF k-quant type.
    GgufQuantInterface(GgufKQuantType gate, GgufKQuantType up, GgufKQuantType down);

    /// Uniform construction: all three projections share `type`.
    explicit GgufQuantInterface(GgufKQuantType type);

    std::string_view name() const override;
    config::WeightQuant weight_quant() const override;
    int64_t bytes_per_expert(const ExpertShape& shape) const override;
    int64_t bytes_per_projection(const ExpertShape& shape, Projection proj) const override;
    int64_t weight_bytes_per_projection(const ExpertShape& shape,
                                        Projection proj) const override;
    int64_t replicated_bytes_per_projection() const override;
    double bytes_per_element() const override;
    double dequant_flops_per_element() const override;
    bool tensor_core_eligible() const override;
    MemoryLayout memory_layout() const override;

    GgufKQuantType projection_type(Projection proj) const;

private:
    GgufKQuantType gate_;
    GgufKQuantType up_;
    GgufKQuantType down_;
    bool uniform_;             // all three projections equal
    config::WeightQuant wq_;   // uniform variant enum, or generic `gguf` for mixed
    std::string_view name_;    // "gguf_q4_k" (uniform) or "gguf" (mixed)
};

/// Factory: build a mixed (or uniform) GgufQuantInterface for an expert whose
/// per-projection GGUF k-quant types come from the GGUF file at load time
/// (GG-6). The returned interface is owned by the caller — it is NOT registered
/// in the global registry (the registry holds one static instance per format;
/// mixed types are data-dependent and cannot be a static singleton).
GgufQuantInterface make_gguf_quant(GgufKQuantType gate, GgufKQuantType up,
                                   GgufKQuantType down);

/// (Re)register the GGUF QuantInterface singletons (the six uniform variants +
/// the generic `gguf` sentinel) in the global registry. Idempotent: only adds
/// entries missing from the registry, so it is safe to call after
/// clear_registry() (used by tests for isolation against other suites that
/// clear the registry). The static initializer in gguf_kquant.cpp calls this
/// once at load.
void register_gguf_formats();

}  // namespace layerstorm::model
