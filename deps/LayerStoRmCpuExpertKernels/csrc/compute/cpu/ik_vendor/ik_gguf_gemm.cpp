// LayerStoRm <-> ik GGUF GEMM bridge implementation (C-6, CPU expert path).
//
// Includes the vendored ggml headers (CUDA-FREE on the CPU path: ggml-common.h
// only pulls <cuda_fp16.h> under GGML_COMMON_DECL_CUDA, which we do NOT define).
// Calls the vendored verbatim ik activation quantizer + the ls_iqk_mul_mat
// driver. The weight (de)quantizers here are LayerStoRm reference packers
// (block-of-32 / K-quant super-block) used off the hot path (loading + tests);
// the GEMM kernels they feed remain 100% verbatim ik.
//
// CPU-only TU. No CUDA (INV-GPU-1).

#include "compute/cpu/ik_vendor/ik_gguf_gemm.h"

// iqk_common.h is the upstream aggregation header: it pulls ggml-impl.h (fp16
// macros), ggml-quants.h -> ggml-common.h (block_q8_0 / block_q8_2 / block_q4_K /
// ... structs + QK* constants) and the GGML_TYPE_* enum. Guarded by
// IQK_IMPLEMENT (AVX2 / dotprod) — the bridge is only meaningful on such a host.
#include "compute/cpu/ik_vendor/iqk_common.h"

#include <cmath>
#include <cstring>
#include <vector>

// Vendored verbatim ik activation quantizer (ik_q8_quantize.cpp).
//
// Activation-type truth (ik commit d47f484, ggml/src/iqk, x86_64 path):
// iqk_set_kernels_legacy_quants (Q8_0/Q5_0) declares expected_typeB =
// GGML_TYPE_Q8_2_X4 UNCONDITIONALLY, and iqk_set_kernels_kquants declares
// expected_type_B = GGML_TYPE_Q8_2_X4 for Q4_K/Q5_K/Q6_K UNCONDITIONALLY. The
// `#ifdef HAVE_FANCY_SIMD` inside those GEMM kernels only switches the dot-
// product instruction (`_mm256_dpbusd_epi32` vs `_mm256_maddubs_epi16`), NOT the
// activation BLOCK LAYOUT — both branches consume block_q8_2_x4. (The Q8_0_X4 /
// Q8_1_X4 distinction lives in the __aarch64__ kernels, not x86_64.) So on BOTH
// the native VNNI build (HAVE_FANCY_SIMD defined) and the non-VNNI AVX2 build
// (undefined), the correct activation type for every wired weight is Q8_2_X4.
// The selection below is keyed on the SAME HAVE_FANCY_SIMD macro the GEMM uses
// (this TU pulls iqk_config.h via iqk_common.h, compiled with the same package
// arch flags as the GEMM TUs) — verified non-VNNI-correct via a throwaway
// AVX2-only probe (TD-IK-AVX2-ACTQUANT). Multi-ISA RUNTIME dispatch (one binary
// switching paths at run time) remains deferred — TD-IK-MULTIARCH.
void quantize_row_q8_2_x4(const float* x, void* vy, int64_t k);

// LayerStoRm K-quant weight (de)quantizers (ik_qk_quantize.cpp, verbatim ik refs).
#if defined(LS_IK_HAVE_KQUANTS)
void quantize_weight_q4_K_ls(const float* x, void* y, int ne00);
void dequantize_weight_q4_K_ls(const void* x, float* y, int ne00);
void quantize_weight_q5_K_ls(const float* x, void* y, int ne00);
void dequantize_weight_q5_K_ls(const void* x, float* y, int ne00);
void quantize_weight_q6_K_ls(const float* x, void* y, int ne00);
void dequantize_weight_q6_K_ls(const void* x, float* y, int ne00);
#endif

// Minimal dispatcher entry (ik_iqk_dispatch.cpp).
extern "C" bool ls_iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void* A, long strideA,
        int typeB, const void* B, long strideB,
        float* C, long stride_C);

namespace layerstorm::compute::cpu::ik {

namespace {
constexpr int kQ5_0 = 6;      // GGML_TYPE_Q5_0
constexpr int kQ8_0 = 8;      // GGML_TYPE_Q8_0
constexpr int kQ4_K = 12;     // GGML_TYPE_Q4_K
constexpr int kQ5_K = 13;     // GGML_TYPE_Q5_K
constexpr int kQ6_K = 14;     // GGML_TYPE_Q6_K
// (Q8_K is the activation type for the _R4/_R8 repacked GEMMs only, which we do
// NOT expose — all wired plain weights consume Q8_2_X4. See activation_ggml_type.)
constexpr int kQ8_2_X4 = 99;  // GGML_TYPE_Q8_2_X4
}  // namespace

bool gguf_supported(GgufType t) {
#ifndef IQK_IMPLEMENT
    (void)t;
    return false;  // host lacks AVX2 / dotprod: ik kernels not built.
#else
    switch (t) {
        case GgufType::q8_0:
        case GgufType::q5_0: return true;  // legacy path: always built
#if defined(LS_IK_HAVE_KQUANTS)
        case GgufType::q4_k:
        case GgufType::q5_k:
        case GgufType::q6_k: return true;
#else
        case GgufType::q4_k:
        case GgufType::q5_k:
        case GgufType::q6_k: return false;
#endif
    }
    return false;
#endif
}

int weight_ggml_type(GgufType t) {
    switch (t) {
        case GgufType::q8_0: return kQ8_0;
        case GgufType::q5_0: return kQ5_0;
        case GgufType::q4_k: return kQ4_K;
        case GgufType::q5_k: return kQ5_K;
        case GgufType::q6_k: return kQ6_K;
    }
    return -1;
}

int activation_ggml_type(GgufType /*t*/) {
    // Keyed on the SAME macro the GEMM kernels use. On x86_64 both branches are
    // Q8_2_X4 for every wired weight type (legacy Q8_0/Q5_0 + K-quant
    // Q4_K/Q5_K/Q6_K) — see the activation-type truth note above.
#if defined(HAVE_FANCY_SIMD)
    return kQ8_2_X4;  // AVX-512-VNNI (native default)
#else
    return kQ8_2_X4;  // plain AVX2 (non-VNNI) — same activation layout
#endif
}

size_t weight_row_bytes(GgufType t, int ne00) {
    switch (t) {
        case GgufType::q8_0:
            return static_cast<size_t>(ne00 / QK8_0) * sizeof(block_q8_0);
        case GgufType::q5_0:
            return static_cast<size_t>(ne00 / QK5_0) * sizeof(block_q5_0);
        case GgufType::q4_k:
            return static_cast<size_t>(ne00 / QK_K) * sizeof(block_q4_K);
        case GgufType::q5_k:
            return static_cast<size_t>(ne00 / QK_K) * sizeof(block_q5_K);
        case GgufType::q6_k:
            return static_cast<size_t>(ne00 / QK_K) * sizeof(block_q6_K);
    }
    return 0;
}

size_t activation_row_bytes(GgufType /*t*/, int ne00) {
    // Q8_2_X4 for all wired types (VNNI and non-VNNI x86_64 alike): nblocks of
    // q8_2, 34 B each (the _x4 packing is layout only — same per-block footprint
    // as q8_2). Must match activation_ggml_type()/quantize_activations().
    return static_cast<size_t>(ne00 / QK8_2) * sizeof(block_q8_2);
}

void quantize_activations(GgufType t, const float* in, void* out,
                          int num_rows, int ne00) {
    const size_t rb = activation_row_bytes(t, ne00);
    auto* o = static_cast<uint8_t*>(out);
    for (int r = 0; r < num_rows; ++r) {
        const float* x = in + static_cast<size_t>(r) * ne00;
        void* y = o + static_cast<size_t>(r) * rb;
        (void)t;
        // All wired types use Q8_2_X4 activations on both the VNNI and non-VNNI
        // x86_64 build paths (see activation_ggml_type).
        quantize_row_q8_2_x4(x, y, ne00);
    }
}

// ── Reference weight (de)quantizers (off hot path) ──────────────────────────

namespace {
// Q8_0 block-of-32 reference quantizer (ggml's quantize_row_q8_0_ref logic).
void quantize_weight_q8_0(const float* x, block_q8_0* y, int ne00) {
    const int nb = ne00 / QK8_0;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j)
            amax = std::max(amax, std::fabs(x[i * QK8_0 + j]));
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d);
        for (int j = 0; j < QK8_0; ++j)
            y[i].qs[j] = static_cast<int8_t>(std::lround(x[i * QK8_0 + j] * id));
    }
}
void dequantize_weight_q8_0(const block_q8_0* x, float* y, int ne00) {
    const int nb = ne00 / QK8_0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK8_0; ++j)
            y[i * QK8_0 + j] = x[i].qs[j] * d;
    }
}

// ── Q5_0 block-of-32 reference (de)quantizers ────────────────────────────────
// VERBATIM ik_llama.cpp (commit d47f484, MIT) ggml/src/ggml-quants.c bodies of
// quantize_row_q5_0_ref / dequantize_row_q5_0, adapted only to take a plain ne00
// (caller already asserts ne00 % QK5_0 == 0 via weight_row_bytes). Off the hot
// path (weight load + tests); the GEMM kernel they feed stays verbatim ik.
void quantize_weight_q5_0(const float* x, block_q5_0* y, int ne00) {
    static const int qk = QK5_0;
    const int nb = ne00 / qk;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];
            if (amax < std::fabs(v)) { amax = std::fabs(v); max = v; }
        }
        const float d = max / -16;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d);
        uint32_t qh = 0;
        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = x[i * qk + 0 + j] * id;
            const float x1 = x[i * qk + qk / 2 + j] * id;
            const uint8_t xi0 = static_cast<uint8_t>(
                std::min(31, static_cast<int>(static_cast<int8_t>(x0 + 16.5f))));
            const uint8_t xi1 = static_cast<uint8_t>(
                std::min(31, static_cast<int>(static_cast<int8_t>(x1 + 16.5f))));
            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk / 2);
        }
        std::memcpy(&y[i].qh, &qh, sizeof(qh));
    }
}
void dequantize_weight_q5_0(const block_q5_0* x, float* y, int ne00) {
    static const int qk = QK5_0;
    const int nb = ne00 / qk;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        uint32_t qh;
        std::memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < qk / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >> 4) | xh_1) - 16;
            y[i * qk + j + 0] = x0 * d;
            y[i * qk + j + qk / 2] = x1 * d;
        }
    }
}
}  // namespace

void quantize_weight(GgufType t, const float* in, void* out, int ne00) {
    switch (t) {
        case GgufType::q8_0:
            quantize_weight_q8_0(in, static_cast<block_q8_0*>(out), ne00);
            return;
        case GgufType::q5_0:
            quantize_weight_q5_0(in, static_cast<block_q5_0*>(out), ne00);
            return;
#if defined(LS_IK_HAVE_KQUANTS)
        case GgufType::q4_k:
            quantize_weight_q4_K_ls(in, out, ne00);
            return;
        case GgufType::q5_k:
            quantize_weight_q5_K_ls(in, out, ne00);
            return;
        case GgufType::q6_k:
            quantize_weight_q6_K_ls(in, out, ne00);
            return;
#else
        case GgufType::q4_k:
        case GgufType::q5_k:
        case GgufType::q6_k:
            return;
#endif
    }
}

void dequantize_weight(GgufType t, const void* in, float* out, int ne00) {
    switch (t) {
        case GgufType::q8_0:
            dequantize_weight_q8_0(static_cast<const block_q8_0*>(in), out, ne00);
            return;
        case GgufType::q5_0:
            dequantize_weight_q5_0(static_cast<const block_q5_0*>(in), out, ne00);
            return;
#if defined(LS_IK_HAVE_KQUANTS)
        case GgufType::q4_k:
            dequantize_weight_q4_K_ls(in, out, ne00);
            return;
        case GgufType::q5_k:
            dequantize_weight_q5_K_ls(in, out, ne00);
            return;
        case GgufType::q6_k:
            dequantize_weight_q6_K_ls(in, out, ne00);
            return;
#else
        case GgufType::q4_k:
        case GgufType::q5_k:
        case GgufType::q6_k:
            return;
#endif
    }
}

bool gguf_gemm_one(GgufType t, long Nx, long Ny, long ne00,
                   const void* Wq, const void* Aq, float* C, long stride_C) {
    if (!gguf_supported(t)) return false;
    const long strideA = static_cast<long>(weight_row_bytes(t, ne00));
    const long strideB = static_cast<long>(activation_row_bytes(t, ne00));
    return ls_iqk_mul_mat(Nx, Ny, ne00,
                          weight_ggml_type(t), Wq, strideA,
                          activation_ggml_type(t), Aq, strideB,
                          C, stride_C);
}

}  // namespace layerstorm::compute::cpu::ik
