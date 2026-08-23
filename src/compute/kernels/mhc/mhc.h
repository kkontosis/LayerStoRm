// DeepSeek-V4 mHC (hyper-connection) residual-stream launchers (V4-5b).
//
// The V4 residual is [rows, hc_mult, hidden] BF16 (flattened per token).
// hc_pre collapses it to the module input x [rows, hidden] and emits the
// per-token mix coefficients (post [rows, hc] f32, comb [rows, hc*hc] f32,
// comb layout [src][dst]); hc_post replaces the residual add:
//   R'[dst] = post[dst] * y + Σ_src comb[src][dst] * R[src].
// hc_head is the pre-branch-only collapse before the LM head.
//
// Kernels live in deps/LayerStoRmKernels csrc/smxx/mhc.cu (single-TU include
// via src/compute/kernels/sm120/mhc/mhc.cu). Math reference: ref/vllm
// kernels/mhc/torch.py (Apache-2.0, vLLM project) + ref/llama.cpp
// models/deepseek4.cpp build_hc_* (MIT, llama.cpp authors) — see
// THIRD_PARTY_NOTICES.md.
// CUDA-free header (INV-GPU-1): callable from non-designated TUs.

#pragma once

#include <cstdint>

namespace layerstorm::compute {

// Collapse + mix coefficients. residual rows are hc*hidden wide.
//   x_out:    [rows, hidden] BF16
//   post_out: [rows, hc] F32
//   comb_out: [rows, hc*hc] F32 ([src][dst])
//   fn:       [(2+hc)*hc, hc*hidden] F32 (row-major)
//   scale:    [3] F32; base: [(2+hc)*hc] F32
void launch_mhc_pre(
    void* x_out,
    void* post_out,
    void* comb_out,
    const void* residual,
    const void* fn,
    const void* scale,
    const void* base,
    float rms_eps,
    float hc_eps,
    float post_mult,
    int sinkhorn_iters,
    int rows,
    int hc,
    int hidden,
    void* stream /*cudaStream_t*/);

// Residual re-expand + doubly-stochastic mix. residual_out may alias residual
// (in-place safe) or be a different hc*hidden-wide buffer.
//   y: [rows, hidden] BF16 (module output)
void launch_mhc_post(
    void* residual_out,
    const void* y,
    const void* residual,
    const void* post,
    const void* comb,
    int rows,
    int hc,
    int hidden,
    void* stream /*cudaStream_t*/);

// Head collapse (pre branch only): fn [hc, hc*hidden] F32, scale [1], base [hc].
void launch_mhc_head(
    void* x_out,
    const void* residual,
    const void* fn,
    const void* scale,
    const void* base,
    float rms_eps,
    float hc_eps,
    int rows,
    int hc,
    int hidden,
    void* stream /*cudaStream_t*/);

// Embedding expansion: residual_out[row, s, :] = x_in[row, :] for every
// stream s (llama.cpp deepseek4.cpp:1091-1092 ggml_repeat).
//   x_in: [rows, hidden] BF16; residual_out: [rows, hc*hidden] BF16.
void launch_hc_expand_repeat(
    void* residual_out,
    const void* x_in,
    int rows,
    int hc,
    int hidden,
    void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
