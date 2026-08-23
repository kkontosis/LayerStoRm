#pragma once

// LayerStoRm <-> ik_llama GGUF GEMM bridge (C-6, CPU expert path).
//
// Thin, CUDA-FREE C++ facade over the vendored ik kernels (ik_iqk_dispatch.cpp
// -> ls_iqk_mul_mat -> verbatim iqk_gemm_legacy_quants.cpp / iqk_gemm_kquants.cpp).
// Exposes ONLY what the CPU ExpertDevice's gguf_grouped_gemm needs:
//   * GgufType enum (the GGUF weight quant we support on CPU).
//   * Block / row-size helpers (so callers can size the packed weight + the
//     quantized-activation scratch without including ggml-common.h).
//   * quantize_activations(): BF16/FP32 activation row(s) -> the Q8_x activation
//     format the chosen weight type's ik GEMM consumes. On x86_64 this is
//     Q8_2_X4 for ALL wired families (legacy Q8_0/Q5_0 AND K-quant Q4_K/Q5_K/Q6_K
//     — the K-quant GEMM's expected_type_B is Q8_2_X4, NOT Q8_K; Q8_K is only the
//     _R4/_R8 repacked GEMMs we don't expose). Via the VERBATIM ik quantizer.
//     (aarch64 wants Q8_0_X4/Q8_1_X4, unvendored — TD-IK-AARCH64-ACTQUANT.)
//   * gguf_gemm_one(): single-expert GEMM C[Nx,Ny] = Wq[Nx,ne00] @ Aq[Ny,ne00].
//   * quantize_weight() / dequantize_weight(): reference (de)quant of one weight
//     row, for tests + the fp32-reference fallback. quantize is the LayerStoRm
//     reference packer (not on the hot path); the GEMM kernel stays verbatim ik.
//
// This header pulls in NO ggml headers (keeps the CUDA-free / minimal-include
// contract for core). The .cpp (ik_gguf_gemm.cpp) does the ggml-side work.

#include <cstddef>
#include <cstdint>

namespace layerstorm::compute::cpu::ik {

/// GGUF weight quantizations supported on the CPU expert path. The tag selects
/// the ik kernel family + the activation format. Extend as families are wired.
enum class GgufType {
    q8_0,  ///< legacy 8-bit (block=32, fp16 scale). Activation -> Q8_2_X4.
    q5_0,  ///< legacy 5-bit (block=32, fp16 scale). Activation -> Q8_2_X4.
    q4_k,  ///< K-quant 4-bit (super-block=256). Activation -> Q8_2_X4 (x86).
    q5_k,  ///< K-quant 5-bit (super-block=256). Activation -> Q8_2_X4 (x86).
    q6_k,  ///< K-quant 6-bit (super-block=256). Activation -> Q8_2_X4 (x86).
};

/// True if the build vendored kernels for `t` AND this host has the ISA ik
/// requires (AVX2 / dotprod). When false, gguf_gemm_one() returns false and the
/// caller must fall back to the dequant reference path.
bool gguf_supported(GgufType t);

/// Bytes of one packed weight row of `ne00` elements at quant `t`.
size_t weight_row_bytes(GgufType t, int ne00);

/// Bytes of one quantized-activation row of `ne00` elements for `t`'s GEMM.
size_t activation_row_bytes(GgufType t, int ne00);

/// The ggml type id (int) of the activation format `t`'s GEMM consumes.
int activation_ggml_type(GgufType t);
/// The ggml type id (int) of the weight format `t`.
int weight_ggml_type(GgufType t);

/// Quantize `num_rows` FP32 activation rows (row-major, `ne00` cols, row stride
/// `ne00` floats) into `out` (caller-sized num_rows * activation_row_bytes).
/// Uses the VERBATIM ik activation quantizer for `t`.
void quantize_activations(GgufType t, const float* in, void* out,
                          int num_rows, int ne00);

/// Reference: quantize one FP32 weight row (`ne00` cols) into `out`
/// (weight_row_bytes). LayerStoRm reference packer (matches ik's block layout).
void quantize_weight(GgufType t, const float* in, void* out, int ne00);

/// Reference: dequantize one packed weight row back to FP32 (`ne00` floats).
void dequantize_weight(GgufType t, const void* in, float* out, int ne00);

/// Single-expert GEMM: C[ix, iy] = sum_k Wq[ix, k] * Aq[iy, k], ix in [0,Nx),
/// iy in [0,Ny), k in [0,ne00). C is row-major float with row stride `stride_C`
/// floats. Wq is `Nx` packed weight rows (stride weight_row_bytes); Aq is `Ny`
/// quantized-activation rows (stride activation_row_bytes). Returns false if the
/// type is unsupported on this build/host (caller dequant-fallback).
bool gguf_gemm_one(GgufType t, long Nx, long Ny, long ne00,
                   const void* Wq, const void* Aq, float* C, long stride_C);

}  // namespace layerstorm::compute::cpu::ik
