// Minimal iqk GEMM dispatcher for the LayerStoRm CPU expert path (C-6).
//
// This is the THIN GLUE around ik's verbatim GEMM kernels. The actual quant
// kernels are vendored byte-for-byte in iqk_gemm_legacy_quants.cpp (Q8_0) and
// iqk_gemm_kquants.cpp (Q4_K/Q6_K); the Q8_x activation quantization is vendored
// byte-for-byte in ik_q8_quantize.cpp. Here we provide ONLY:
//   * The MulMat::mul_mat_NxM tiling driver — copied VERBATIM from ik's
//     iqk_mul_mat.cpp (the GEMM-only members; the up-gate / flash-attn / MoE
//     members of the upstream struct are STUBBED OUT — see TECH_DEBT TD-IK-*).
//   * A prepare() that routes ONLY to iqk_set_kernels_legacy_quants and
//     iqk_set_kernels_kquants (the two families we use), instead of the upstream
//     switch over every quant family — so we don't have to link kernels for IQ*,
//     KT, 1-bit, float, etc. (those gemm modules are intentionally NOT vendored).
//   * A simple single-thread driver ls_iqk_mul_mat() that mirrors the
//     non-repack tail of ik's iqk_mul_mat() (num_rows==1, no is_dequant_better
//     repack optimization — that path needs iqk_convert_repack which is stubbed).
//
// Why not vendor iqk_mul_mat.cpp verbatim: its dispatcher #includes and links
// EVERY gemm-quant module (iqk_gemm_{floats,kquants,ktquants,iquants,iqk_quants,
// 1bit,legacy}.cpp ≈ 1.3M lines) plus iqk_quantize.cpp + a repack closure. We use
// two families; vendoring the kernels we use + a thin router is the minimal
// closure. The kernels themselves are 100% verbatim ik.
//
// CPU-only TU. No CUDA (INV-GPU-1). MIT-derived glue; the verbatim kernels carry
// ik's MIT header (see LICENSE.ik).

#include "iqk_common.h"

#ifdef IQK_IMPLEMENT

#include "iqk_gemm_legacy_quants.h"
#if defined(LS_IK_HAVE_KQUANTS)
#include "iqk_gemm_kquants.h"
#endif

#include <array>
#include <cstdio>
#include <algorithm>

namespace {

// VERBATIM (GEMM-only members) from ik_llama.cpp iqk_mul_mat.cpp MulMat struct.
// The up-gate / fa / moe / activation members of the upstream struct are removed
// (not vendored). funcs/func16/mul_mat_NxM are byte-identical to upstream.
struct MulMat {
    std::array<mul_mat_t, IQK_MAX_NY> funcs = {};
    mul_mat_t func16 = nullptr;
    inline void mul_mat_NxM(int n, const void * vx, size_t bx, DataInfo& info, int nrc_x, int nrc_y) {
        constexpr int k_x_step = 64;
        if (func16 && nrc_y >= 16) {
            int n_step = (nrc_y - info.cur_y)/16;
            for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                auto this_info = info;
                this_info.s += ix;
                int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                for (int iy = 0; iy < n_step; ++iy) {
                    func16(n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                    this_info.cur_y += 16;
                }
            }
            info.cur_y += 16 * n_step;
            if (info.cur_y == nrc_y) return;
        }
        int ny = funcs.size();
        while (!funcs[ny-1] && ny > 0) --ny;
        int n_left = nrc_y - info.cur_y;
        int n_step = n_left/ny;
        if (n_step > 0) {
            if (n_step*ny != n_left) {
                ++n_step;
                int ny1 = n_left/n_step;
                int ny2 = ny1 + 1;
                int my1 = n_step*ny2 - n_left;
                int my2 = n_step - my1;
                for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                    auto this_info = info;
                    this_info.s += ix;
                    int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                    for (int iy = 0; iy < my1; ++iy) {
                        funcs[ny1-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny1;
                    }
                    for (int iy = 0; iy < my2; ++iy) {
                        funcs[ny2-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny2;
                    }
                }
                info.cur_y += n_left;
            }
            else {
                for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                    auto this_info = info;
                    this_info.s += ix;
                    int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                    for (int iy = 0; iy < n_step; ++iy) {
                        funcs[ny-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny;
                    }
                }
                info.cur_y += ny * n_step;
            }
        }
        n_left = nrc_y - info.cur_y;
        if (n_left > 0) {
            funcs[n_left-1](n, vx, bx, info, nrc_x);
        }
    }

    // Minimal router: only the two families we vendored. Returns false for any
    // other typeA (upstream routes ~9 families here).
    static bool prepare(int typeA, int typeB, int ne00, MulMat& mm) {
        if (iqk_set_kernels_legacy_quants(ne00, typeA, typeB, mm.funcs, mm.func16)) return true;
#if defined(LS_IK_HAVE_KQUANTS)
        if (iqk_set_kernels_kquants(ne00, typeA, typeB, mm.funcs, mm.func16)) return true;
#endif
        return false;
    }
};

}  // namespace

// Public C entry: single-threaded (ith=0,nth=1) per-expert GEMM. C[Nx, Ny] in
// row-major float, with C row stride = stride_C (floats). A = weights (typeA),
// strideA bytes/weight-row; B = quantized activations (typeB), strideB bytes/row.
// num_rows is assumed 1 (plain Q8_0 / Q4_K weights — no _R8 repack), matching the
// non-repack tail of ik's iqk_mul_mat(). Returns false if the type pair is
// unsupported by the vendored kernels (caller must dequant-fallback).
extern "C" bool ls_iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C) {
    MulMat mm;
    if (!MulMat::prepare(typeA, typeB, ne00, mm)) return false;
    size_t row_size_qy = strideB;
    DataInfo info{C, (const char *)B, (size_t)stride_C, row_size_qy, 0, 1, nullptr, 0};
    mm.mul_mat_NxM((int)ne00, (const char *)A, (size_t)strideA, info, (int)Nx, (int)Ny);
    return true;
}

#else  // !IQK_IMPLEMENT  (no AVX2 / dotprod) — vendored kernels are unavailable.

extern "C" bool ls_iqk_mul_mat(long, long, long, int, const void *, long,
                               int, const void *, long, float *, long) {
    return false;
}

#endif  // IQK_IMPLEMENT
