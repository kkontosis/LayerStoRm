#include "model/quantization/gguf_kquant.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "model/quantization/registry.h"

namespace layerstorm::model {

namespace gguf {

std::string_view type_name(GgufKQuantType type) {
    switch (type) {
        case GgufKQuantType::Q2_K: return "gguf_q2_k";
        case GgufKQuantType::Q3_K: return "gguf_q3_k";
        case GgufKQuantType::Q4_K: return "gguf_q4_k";
        case GgufKQuantType::Q5_K: return "gguf_q5_k";
        case GgufKQuantType::Q6_K: return "gguf_q6_k";
        case GgufKQuantType::Q8_0: return "gguf_q8_0";
        case GgufKQuantType::MXFP4: return "gguf_mxfp4";
    }
    throw std::runtime_error("gguf::type_name: invalid GgufKQuantType "
                             + std::to_string(static_cast<int>(type)));
}

GgufKQuantType type_from_name(std::string_view name) {
    // Accept the canonical "gguf_qX_k"/"gguf_q8_0" names and the bare "qX_k"
    // forms (manifest stores the canonical name).
    if (name == "gguf_q2_k" || name == "q2_k" || name == "Q2_K") return GgufKQuantType::Q2_K;
    if (name == "gguf_q3_k" || name == "q3_k" || name == "Q3_K") return GgufKQuantType::Q3_K;
    if (name == "gguf_q4_k" || name == "q4_k" || name == "Q4_K") return GgufKQuantType::Q4_K;
    if (name == "gguf_q5_k" || name == "q5_k" || name == "Q5_K") return GgufKQuantType::Q5_K;
    if (name == "gguf_q6_k" || name == "q6_k" || name == "Q6_K") return GgufKQuantType::Q6_K;
    if (name == "gguf_q8_0" || name == "q8_0" || name == "Q8_0") return GgufKQuantType::Q8_0;
    if (name == "gguf_mxfp4" || name == "mxfp4" || name == "MXFP4") return GgufKQuantType::MXFP4;
    throw std::runtime_error("gguf::type_from_name: unrecognized GGUF k-quant name '"
                             + std::string(name) + "'");
}

GgufKQuantType type_from_weight_quant(config::WeightQuant wq) {
    switch (wq) {
        case config::WeightQuant::gguf_q2_k: return GgufKQuantType::Q2_K;
        case config::WeightQuant::gguf_q3_k: return GgufKQuantType::Q3_K;
        case config::WeightQuant::gguf_q4_k: return GgufKQuantType::Q4_K;
        case config::WeightQuant::gguf_q5_k: return GgufKQuantType::Q5_K;
        case config::WeightQuant::gguf_q6_k: return GgufKQuantType::Q6_K;
        case config::WeightQuant::gguf_q8_0: return GgufKQuantType::Q8_0;
        case config::WeightQuant::gguf_mxfp4: return GgufKQuantType::MXFP4;
        case config::WeightQuant::gguf:
            throw std::runtime_error(
                "gguf::type_from_weight_quant: the generic 'gguf' enum is "
                "per-tensor mixed and has no single k-quant type; build a "
                "GgufQuantInterface via make_gguf_quant(gate,up,down) with the "
                "per-projection types parsed from the GGUF file");
        default:
            throw std::runtime_error(
                "gguf::type_from_weight_quant: WeightQuant "
                + std::to_string(static_cast<int>(wq)) + " is not a GGUF variant");
    }
}

bool is_gguf_weight_quant(config::WeightQuant wq) {
    switch (wq) {
        case config::WeightQuant::gguf:
        case config::WeightQuant::gguf_q2_k:
        case config::WeightQuant::gguf_q3_k:
        case config::WeightQuant::gguf_q4_k:
        case config::WeightQuant::gguf_q5_k:
        case config::WeightQuant::gguf_q6_k:
        case config::WeightQuant::gguf_q8_0:
        case config::WeightQuant::gguf_mxfp4:
            return true;
        default:
            return false;
    }
}

int64_t gguf_packed_bytes(int64_t out_features, int64_t in_features,
                          GgufKQuantType type) {
    const int qk = block_values(type);
    if (in_features % qk != 0) {
        throw std::runtime_error(
            "gguf_packed_bytes: in_features (" + std::to_string(in_features)
            + ") not a multiple of QK (" + std::to_string(qk) + ") for "
            + std::string(type_name(type)));
    }
    return out_features * (in_features / qk) * block_bytes(type);
}

// ── Host MXFP4 dequant reference ────────────────────────────────────────────
// Port of ggml_e8m0_to_fp32_half + dequantize_row_mxfp4
// (ref/llama.cpp ggml/src/ggml-impl.h, ggml-quants.c; MIT License,
// Copyright (c) 2023-2026 The ggml authors — see THIRD_PARTY_NOTICES.md).

float e8m0_to_fp32_half(uint8_t e) {
    uint32_t bits;
    if (e < 2) {
        // 0x00200000 = 2^-128, 0x00400000 = 2^-127 (denormal patterns).
        bits = 0x00200000u << e;
    } else {
        // 0.5 * 2^(e-127) = normalized with exponent (e-1).
        bits = static_cast<uint32_t>(e - 1) << 23;
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void dequant_mxfp4_block(const uint8_t* block, float* out) {
    // Doubled e2m1 values; the halved scale un-doubles them (ggml kvalues_fp4).
    static constexpr int8_t kValues[16] = {0, 1, 2,  3,  4,  6,  8,  12,
                                           0, -1, -2, -3, -4, -6, -8, -12};
    const float d = e8m0_to_fp32_half(block[0]);
    const uint8_t* qs = block + 1;
    for (int j = 0; j < 16; ++j) {
        out[j]      = static_cast<float>(kValues[qs[j] & 0xF]) * d;
        out[j + 16] = static_cast<float>(kValues[qs[j] >> 4]) * d;
    }
}

void dequant_mxfp4_row(const uint8_t* packed, float* out, int64_t n) {
    if (n % 32 != 0) {
        throw std::runtime_error("dequant_mxfp4_row: n (" + std::to_string(n)
                                 + ") must be a multiple of 32");
    }
    for (int64_t b = 0; b < n / 32; ++b) {
        dequant_mxfp4_block(packed + b * 17, out + b * 32);
    }
}

}  // namespace gguf

// ── GgufQuantInterface ──────────────────────────────────────────────────────

namespace {

// (N, K) of a projection: gate/up are [intermediate, hidden]; down is
// [hidden, intermediate]. Mirrors the FP8 blockwise-scale shape convention.
void projection_nk(const ExpertShape& shape, Projection proj,
                   int64_t& n, int64_t& k) {
    switch (proj) {
        case Projection::gate: [[fallthrough]];
        case Projection::up:
            n = shape.intermediate_size;
            k = shape.hidden_size;
            return;
        case Projection::down:
            n = shape.hidden_size;
            k = shape.intermediate_size;
            return;
    }
    n = 0;
    k = 0;
}

}  // namespace

GgufQuantInterface::GgufQuantInterface(GgufKQuantType gate, GgufKQuantType up,
                                       GgufKQuantType down)
    : gate_(gate), up_(up), down_(down),
      uniform_(gate == up && up == down) {
    if (uniform_) {
        name_ = gguf::type_name(gate);
        switch (gate) {
            case GgufKQuantType::Q2_K: wq_ = config::WeightQuant::gguf_q2_k; break;
            case GgufKQuantType::Q3_K: wq_ = config::WeightQuant::gguf_q3_k; break;
            case GgufKQuantType::Q4_K: wq_ = config::WeightQuant::gguf_q4_k; break;
            case GgufKQuantType::Q5_K: wq_ = config::WeightQuant::gguf_q5_k; break;
            case GgufKQuantType::Q6_K: wq_ = config::WeightQuant::gguf_q6_k; break;
            case GgufKQuantType::Q8_0: wq_ = config::WeightQuant::gguf_q8_0; break;
            case GgufKQuantType::MXFP4: wq_ = config::WeightQuant::gguf_mxfp4; break;
        }
    } else {
        name_ = "gguf";
        wq_ = config::WeightQuant::gguf;
    }
}

GgufQuantInterface::GgufQuantInterface(GgufKQuantType type)
    : GgufQuantInterface(type, type, type) {}

GgufKQuantType GgufQuantInterface::projection_type(Projection proj) const {
    switch (proj) {
        case Projection::gate: return gate_;
        case Projection::up:   return up_;
        case Projection::down: return down_;
    }
    return gate_;
}

std::string_view GgufQuantInterface::name() const { return name_; }

config::WeightQuant GgufQuantInterface::weight_quant() const { return wq_; }

int64_t GgufQuantInterface::bytes_per_projection(const ExpertShape& shape,
                                                 Projection proj) const {
    int64_t n, k;
    projection_nk(shape, proj, n, k);
    return gguf::gguf_packed_bytes(n, k, projection_type(proj));
}

int64_t GgufQuantInterface::weight_bytes_per_projection(const ExpertShape& shape,
                                                        Projection proj) const {
    // K-quant scales/mins are packed INSIDE each super-block: no separate scale
    // region, so weight bytes == total bytes.
    return bytes_per_projection(shape, proj);
}

int64_t GgufQuantInterface::bytes_per_expert(const ExpertShape& shape) const {
    return bytes_per_projection(shape, Projection::gate)
         + bytes_per_projection(shape, Projection::up)
         + bytes_per_projection(shape, Projection::down);
}

int64_t GgufQuantInterface::replicated_bytes_per_projection() const {
    return 0;  // scales travel inside the blocks; nothing is replicated
}

double GgufQuantInterface::bytes_per_element() const {
    // Average over the three projections (equal for uniform). This is an
    // amortized accounting figure; precise per-projection sizing uses
    // bytes_per_projection().
    return (gguf::bytes_per_element(gate_)
          + gguf::bytes_per_element(up_)
          + gguf::bytes_per_element(down_)) / 3.0;
}

double GgufQuantInterface::dequant_flops_per_element() const {
    // Software dequant per element: scale lookup + (k-quant) min add + multiply.
    // Heavier than NVFP4 (hardware) / FP8. A single representative figure for
    // planning; mirrors the impl-guide note that GGUF GEMM runs ~60-70% of
    // NVFP4 throughput due to software dequant.
    return 3.0;
}

bool GgufQuantInterface::tensor_core_eligible() const {
    return true;  // the int8 tensor-core mmq path exists (GG-5)
}

MemoryLayout GgufQuantInterface::memory_layout() const {
    // bits_per_weight is an int; k-quants are fractional (Q4_K=4.5, Q6_K=6.5,
    // …). Report the rounded effective bits of the gate projection's type for a
    // coarse descriptor; byte-exact sizing always goes through
    // bytes_per_projection(). group_size = the super-block QK (note: k-quants
    // ALSO nest 16/32-value sub-blocks with their own 6-bit scales/mins inside
    // each super-block — not expressible in this flat descriptor).
    const int bits = static_cast<int>(
        std::lround(8.0 * gguf::bytes_per_element(gate_)));
    return MemoryLayout{
        bits,
        gguf::block_values(gate_),  // QK super-block size
        false,                      // k-quant mins are not a zero-point
        gate_ == GgufKQuantType::MXFP4
            ? ScaleDtype::ue8m0     // MXFP4 blocks carry a 1-byte E8M0 scale
            : ScaleDtype::fp16,     // k-quant/Q8_0 super-block scales are fp16
        gguf::block_bytes(gate_),   // align to one super-block of the gate type
    };
}

GgufQuantInterface make_gguf_quant(GgufKQuantType gate, GgufKQuantType up,
                                   GgufKQuantType down) {
    return GgufQuantInterface(gate, up, down);
}

// ── Self-registration ───────────────────────────────────────────────────────
// One static singleton per uniform variant, plus a generic `gguf` sentinel so
// every config::WeightQuant GGUF enum resolves through the registry. Mixed
// experts size per-projection via make_gguf_quant() and are NOT registered.

namespace {

const GgufQuantInterface g_q2_k{GgufKQuantType::Q2_K};
const GgufQuantInterface g_q3_k{GgufKQuantType::Q3_K};
const GgufQuantInterface g_q4_k{GgufKQuantType::Q4_K};
const GgufQuantInterface g_q5_k{GgufKQuantType::Q5_K};
const GgufQuantInterface g_q6_k{GgufKQuantType::Q6_K};
const GgufQuantInterface g_q8_0{GgufKQuantType::Q8_0};
const GgufQuantInterface g_mxfp4{GgufKQuantType::MXFP4};

// Generic `gguf` sentinel (TD-GGUF-GENERIC-DEFAULT-MISSIZE resolved by GG-6):
// the generic enum is per-tensor MIXED and data-dependent, so a single static
// instance CANNOT size it correctly. name()/weight_quant() resolve (so registry
// lookups + config validation work), but every bytes_per_*/sizing method THROWS
// with guidance — real GGUF sizing MUST go through make_gguf_quant(gate,up,down)
// built from the per-projection types read out of the GGUF file (GG-6 routes the
// loader + prepacker through that path). This makes a stray generic-`gguf` sizing
// call fail loudly instead of silently mis-sizing slots/allocations.
[[noreturn]] void generic_gguf_sizing_error() {
    throw std::runtime_error(
        "QuantInterface for the generic 'gguf' weight type cannot size experts: "
        "GGUF is per-tensor mixed and data-dependent. Build a concrete interface "
        "via make_gguf_quant(gate,up,down) using the per-projection k-quant types "
        "parsed from the GGUF file (see gguf_expert_types_from_model / GG-6).");
}

struct GgufGenericSentinel final : QuantInterface {
    // Resolving methods (registry lookup + config validation rely on these).
    std::string_view name() const override { return "gguf"; }
    config::WeightQuant weight_quant() const override {
        return config::WeightQuant::gguf;
    }
    // Sizing methods THROW: the generic instance has no single per-projection type.
    int64_t bytes_per_expert(const ExpertShape&) const override { generic_gguf_sizing_error(); }
    int64_t bytes_per_projection(const ExpertShape&, Projection) const override { generic_gguf_sizing_error(); }
    int64_t weight_bytes_per_projection(const ExpertShape&, Projection) const override { generic_gguf_sizing_error(); }
    int64_t replicated_bytes_per_projection() const override { generic_gguf_sizing_error(); }
    double bytes_per_element() const override { generic_gguf_sizing_error(); }
    double dequant_flops_per_element() const override { generic_gguf_sizing_error(); }
    bool tensor_core_eligible() const override { return true; }  // GGUF is TC-eligible (mmq)
    MemoryLayout memory_layout() const override { generic_gguf_sizing_error(); }
};

const GgufGenericSentinel g_generic_named;

// Idempotent registration of the GGUF singletons (definition inside the anon
// namespace so it can reach the file-local instances; declared in the header
// with external linkage). Re-registers only entries missing from the registry,
// so the static initializer calls it once and tests that clear_registry() can
// call it again without tripping register_format's duplicate guard.
void register_gguf_formats_impl() {
    auto ensure = [](const QuantInterface* impl) {
        if (find_format(impl->name()) == nullptr) register_format(impl);
    };
    ensure(&g_q2_k);
    ensure(&g_q3_k);
    ensure(&g_q4_k);
    ensure(&g_q5_k);
    ensure(&g_q6_k);
    ensure(&g_q8_0);
    ensure(&g_mxfp4);
    ensure(&g_generic_named);
}

[[maybe_unused]] const int g_reg = [] {
    register_gguf_formats_impl();
    return 0;
}();

}  // namespace

void register_gguf_formats() { register_gguf_formats_impl(); }

}  // namespace layerstorm::model
