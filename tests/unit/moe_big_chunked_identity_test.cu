// TD-PREFILL-MOE-BIG: chunked-GGEMM identity test (GPU-required).
//
// Runs the routed-MoE compute sequence the production dispatcher emits —
// moe_permute → grouped GEMM (gate) → grouped GEMM (up) → interleave + SwiGLU
// → grouped GEMM (down) → moe_unpermute — over one synthetic token batch TWO
// ways, using the SAME daemon helpers the dispatcher uses (MoeGemmEmitter /
// launch_grouped_gemm from daemon/dispatch_detail.h + the real
// CudaSm120ExpertDevice ops):
//
//   (a) SINGLE-SHOT: one pass over all N tokens (the legacy
//       dispatch_moe_internal Step 2..6 sequence);
//   (b) CHUNKED: 64-token chunks, transient buffers REUSED per chunk, each
//       chunk unpermuted into its own row range of the output (the
//       dispatch_moe_big.cpp chunk loop, INV-MOE-BIG-1/2).
//
// Asserts (the W1 validation bar from spec/bloat/FETCH_AND_RUN_MOE_BIG.md §4):
//   chunked == single-shot  (BIT-identical: per token, both compute the exact
//                            same per-expert GEMM rows and the same fixed
//                            slot-order unpermute sum — the chunk split only
//                            partitions tokens, never a token's K slots), and
//   both ≈ CPU reference    (cosine vs an FP64 reference over the DEQUANTIZED
//                            weights; GGUF dequant strategy = lossless
//                            activations).
//
// Weights are GGUF Q8_0 packed with the ik CPU reference packer (the exact
// block layout the GPU kernel consumes), matching the production GLM-5.2 GGUF
// routed path. Both the dequant and int strategies are exercised.

#include "daemon/dispatch_detail.h"

#include "compute/cuda_sm120_expert_device.h"
#include "compute/cpu/ik_vendor/ik_gguf_gemm.h"
#include "compute/kernels/elementwise/residual_add.h"  // wave accumulate
#include "compute/kernels/moe/moe_gemm_meta.h"         // launch_mask_topk_indices
#include "sm120/gemm/gguf/gguf_grouped_gemm.h"   // workspace bytes
#include "sm120/gemm/gguf/gguf_dequant_gemm.h"   // gguf_block_bytes/values

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace ik = layerstorm::compute::cpu::ik;
namespace lc = layerstorm::compute;
namespace ld = layerstorm::daemon;
namespace lcfg = layerstorm::config;

namespace {

#define CU_CHECK(expr)                                                        \
    do {                                                                      \
        cudaError_t _e = (expr);                                              \
        ASSERT_EQ(_e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_e); \
    } while (0)

__nv_bfloat16 f2bf(float f) { return __float2bfloat16(f); }
float bf2f(__nv_bfloat16 b) { return __bfloat162float(b); }

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += double(a[i]) * b[i];
        na += double(a[i]) * a[i];
        nb += double(b[i]) * b[i];
    }
    if (na == 0 || nb == 0) return 1.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// Test dims: small but structurally faithful (E experts, top-K expansion,
// gate/up [I, H] and down [H, I], Q8_0 needs K % 32 == 0).
constexpr int kH = 256;   // hidden
constexpr int kI = 128;   // moe intermediate
constexpr int kE = 8;     // routed experts
constexpr int kT = 2;     // top-k
constexpr int kN = 160;   // tokens (not a chunk multiple: chunks 64/64/32)
constexpr int kChunk = 64;

struct DeviceBufs {
    // Transient (chunk-capacity in the chunked variant; full in single-shot).
    __nv_bfloat16* permuted = nullptr;   // [rows_cap * T, H]
    int32_t* expert_offsets = nullptr;   // [E + 1]
    int32_t* src_to_dest = nullptr;      // [rows_cap * T]
    int32_t* permuted_idx = nullptr;     // [rows_cap * T]
    int32_t* problem_sizes = nullptr;    // [E * 3] (unused by GGUF; valid mem)
    int32_t* sf_offsets = nullptr;       // [E + 1]  (unused by GGUF; valid mem)
    int32_t* permute_ws = nullptr;       // [rows_cap * T * 8]
    __nv_bfloat16* act_gate = nullptr;   // [rows_cap * T, I]
    __nv_bfloat16* act_up = nullptr;     // [rows_cap * T, I]  (expert_output role)
    __nv_bfloat16* gate_up = nullptr;    // [rows_cap * T, 2I]
    __nv_bfloat16* down_out = nullptr;   // [rows_cap * T, H]
    void* gemm_ws = nullptr;
    size_t gemm_ws_bytes = 0;

    void alloc(int rows_cap) {
        const size_t exp = static_cast<size_t>(rows_cap) * kT;
        CU_CHECK(cudaMalloc(&permuted, exp * kH * 2));
        CU_CHECK(cudaMalloc(&expert_offsets, (kE + 1) * sizeof(int32_t)));
        CU_CHECK(cudaMalloc(&src_to_dest, exp * sizeof(int32_t)));
        CU_CHECK(cudaMalloc(&permuted_idx, exp * sizeof(int32_t)));
        CU_CHECK(cudaMalloc(&problem_sizes, kE * 3 * sizeof(int32_t)));
        CU_CHECK(cudaMalloc(&sf_offsets, (kE + 1) * sizeof(int32_t)));
        CU_CHECK(cudaMalloc(&permute_ws, exp * 8 * sizeof(int32_t)));
        CU_CHECK(cudaMalloc(&act_gate, exp * kI * 2));
        CU_CHECK(cudaMalloc(&act_up, exp * kI * 2));
        CU_CHECK(cudaMalloc(&gate_up, exp * 2 * kI * 2));
        CU_CHECK(cudaMalloc(&down_out, exp * kH * 2));
        gemm_ws_bytes = lc::gguf_grouped_gemm_workspace_bytes(
            static_cast<int>(exp), std::max(kH, kI), kE);
        CU_CHECK(cudaMalloc(&gemm_ws, gemm_ws_bytes));
    }
    void free_all() {
        for (void* p : {static_cast<void*>(permuted),
                        static_cast<void*>(expert_offsets),
                        static_cast<void*>(src_to_dest),
                        static_cast<void*>(permuted_idx),
                        static_cast<void*>(problem_sizes),
                        static_cast<void*>(sf_offsets),
                        static_cast<void*>(permute_ws),
                        static_cast<void*>(act_gate),
                        static_cast<void*>(act_up),
                        static_cast<void*>(gate_up),
                        static_cast<void*>(down_out), gemm_ws})
            cudaFree(p);
    }
};

// One permute→gate→up→SwiGLU→down→unpermute pass over rows
// [row0, row0 + rows) of the batch, writing out rows [row0, ...) of `out`.
// Mirrors emit_routed_ffn (dispatch_moe.cpp) / the dispatch_moe_big.cpp chunk
// body for the GGUF quant mode, using the same daemon helpers.
void run_pass(lc::ExpertDevice* dev, DeviceBufs& b,
              lc::GgufGemmStrategy strategy,
              const __nv_bfloat16* hidden_dev, const int32_t* topk_idx_dev,
              const float* topk_w_dev, const void** b_gate, const void** b_up,
              const void** b_down, __nv_bfloat16* out_dev,
              int row0, int rows) {
    const int exp_rows = rows * kT;
    void* stream = nullptr;  // default stream

    const ld::MoeGemmEmitter emit{
        dev, b.gemm_ws, b.gemm_ws_bytes, stream,
        kE, /*use_fp8=*/false, /*use_gguf=*/true, strategy,
        /*total_tokens=*/exp_rows,
        b.expert_offsets, b.sf_offsets, b.problem_sizes};

    // Step 2: permute this pass's rows.
    dev->moe_permute(
        b.permuted, b.expert_offsets, b.src_to_dest, b.permuted_idx,
        hidden_dev + static_cast<size_t>(row0) * kH,
        topk_idx_dev + static_cast<size_t>(row0) * kT,
        rows, kT, kH, kE, /*elem_size_bytes=*/2, b.permute_ws, stream);

    // Step 3a: gate GEMM (GGUF feeds BF16 permuted rows directly).
    emit.routed_gemm(kI, kH, b.permuted, b.act_gate, b_gate, nullptr,
                     nullptr, nullptr, lc::GgufQuantType::Q8_0);
    // Step 3b: up GEMM.
    emit.routed_gemm(kI, kH, b.permuted, b.act_up, b_up, nullptr,
                     nullptr, nullptr, lc::GgufQuantType::Q8_0);

    // Step 4: interleave gate‖up + SwiGLU (the GGUF branch of Step 4).
    const size_t I_bytes = static_cast<size_t>(kI) * 2;
    CU_CHECK(cudaMemcpy2DAsync(b.gate_up, I_bytes * 2, b.act_gate, I_bytes,
                               I_bytes, exp_rows, cudaMemcpyDeviceToDevice,
                               nullptr));
    CU_CHECK(cudaMemcpy2DAsync(
        reinterpret_cast<uint8_t*>(b.gate_up) + I_bytes, I_bytes * 2,
        b.act_up, I_bytes, I_bytes, exp_rows, cudaMemcpyDeviceToDevice,
        nullptr));
    ld::launch_swiglu(dev, b.act_gate, b.gate_up, exp_rows, kI, stream,
                      /*swiglu_limit=*/0.0f);

    // Step 5: down GEMM.
    emit.routed_gemm(kH, kI, b.act_gate, b.down_out, b_down, nullptr,
                     nullptr, nullptr, lc::GgufQuantType::Q8_0);

    // Step 6: unpermute into the pass's output rows.
    dev->moe_unpermute(
        out_dev + static_cast<size_t>(row0) * kH, b.down_out,
        topk_w_dev + static_cast<size_t>(row0) * kT, b.src_to_dest,
        rows, kT, kH, /*elem_size_bytes=*/2, stream,
        lc::MoeCombineMode::kReducedBf16);
}

void run_identity_case(lc::GgufGemmStrategy strategy, double min_cos_ref) {
    if (!ik::gguf_supported(ik::GgufType::q8_0))
        GTEST_SKIP() << "ik q8_0 unsupported in this build";

    std::mt19937 rng(0x5EED);
    std::uniform_real_distribution<float> wd(-0.5f, 0.5f), ad(-1.0f, 1.0f);

    // Hidden states + routing.
    std::vector<__nv_bfloat16> hidden_bf(static_cast<size_t>(kN) * kH);
    std::vector<float> hidden_f(hidden_bf.size());
    for (size_t i = 0; i < hidden_bf.size(); ++i) {
        hidden_bf[i] = f2bf(ad(rng));
        hidden_f[i] = bf2f(hidden_bf[i]);
    }
    std::vector<int32_t> topk_idx(static_cast<size_t>(kN) * kT);
    std::vector<float> topk_w(topk_idx.size());
    std::uniform_int_distribution<int> ed(0, kE - 1);
    for (int t = 0; t < kN; ++t) {
        int e0 = ed(rng);
        int e1 = (e0 + 1 + ed(rng) % (kE - 1)) % kE;  // distinct second slot
        topk_idx[t * kT + 0] = e0;
        topk_idx[t * kT + 1] = e1;
        topk_w[t * kT + 0] = 0.25f + 0.5f * ad(rng) * ad(rng);
        topk_w[t * kT + 1] = 1.0f - topk_w[t * kT + 0];
    }

    // Q8_0 expert weights (gate/up [I, H], down [H, I]) + FP32 dequant refs.
    const size_t row_h = ik::weight_row_bytes(ik::GgufType::q8_0, kH);
    const size_t row_i = ik::weight_row_bytes(ik::GgufType::q8_0, kI);
    std::vector<std::vector<uint8_t>> pk_gate(kE), pk_up(kE), pk_down(kE);
    std::vector<std::vector<float>> dq_gate(kE), dq_up(kE), dq_down(kE);
    auto make_w = [&](std::vector<uint8_t>& pk, std::vector<float>& dq,
                      int n, int k, size_t row_bytes) {
        pk.resize(static_cast<size_t>(n) * row_bytes);
        dq.resize(static_cast<size_t>(n) * k);
        std::vector<float> wr(k);
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < k; ++c) wr[c] = wd(rng);
            void* dst = pk.data() + static_cast<size_t>(r) * row_bytes;
            ik::quantize_weight(ik::GgufType::q8_0, wr.data(), dst, k);
            ik::dequantize_weight(ik::GgufType::q8_0, dst,
                                  dq.data() + static_cast<size_t>(r) * k, k);
        }
    };
    for (int e = 0; e < kE; ++e) {
        make_w(pk_gate[e], dq_gate[e], kI, kH, row_h);
        make_w(pk_up[e],   dq_up[e],   kI, kH, row_h);
        make_w(pk_down[e], dq_down[e], kH, kI, row_i);
    }

    // ── CPU reference (FP64 over dequantized weights) ─────────────────────
    std::vector<float> ref(static_cast<size_t>(kN) * kH, 0.0f);
    std::vector<double> gate_o(kI), up_o(kI), act(kI), down_o(kH);
    for (int t = 0; t < kN; ++t) {
        std::vector<double> acc(kH, 0.0);
        for (int s = 0; s < kT; ++s) {
            const int e = topk_idx[t * kT + s];
            const float w = topk_w[t * kT + s];
            const float* x = hidden_f.data() + static_cast<size_t>(t) * kH;
            for (int r = 0; r < kI; ++r) {
                double g = 0, u = 0;
                const float* wg = dq_gate[e].data() + static_cast<size_t>(r) * kH;
                const float* wu = dq_up[e].data() + static_cast<size_t>(r) * kH;
                for (int c = 0; c < kH; ++c) { g += double(x[c]) * wg[c];
                                               u += double(x[c]) * wu[c]; }
                gate_o[r] = g; up_o[r] = u;
                act[r] = (g / (1.0 + std::exp(-g))) * u;  // SiLU(g) * u
            }
            for (int r = 0; r < kH; ++r) {
                double d = 0;
                const float* wdn = dq_down[e].data() + static_cast<size_t>(r) * kI;
                for (int c = 0; c < kI; ++c) d += act[c] * wdn[c];
                acc[r] += double(w) * d;
            }
        }
        for (int r = 0; r < kH; ++r)
            ref[static_cast<size_t>(t) * kH + r] = static_cast<float>(acc[r]);
    }

    // ── Device setup ──────────────────────────────────────────────────────
    lcfg::GpuRef gpu{0, 0, lcfg::GpuType::rtx5090};
    auto dev = lc::make_cuda_sm120_expert_device(gpu);
    ASSERT_NE(dev, nullptr);
    dev->set_device();

    __nv_bfloat16* d_hidden = upload(hidden_bf);
    int32_t* d_topk_idx = upload(topk_idx);
    float* d_topk_w = upload(topk_w);

    auto upload_experts = [&](std::vector<std::vector<uint8_t>>& pk,
                              std::vector<void*>& devw) -> const void** {
        devw.resize(kE);
        for (int e = 0; e < kE; ++e) devw[e] = upload(pk[e]);
        void** d_ptrs = nullptr;
        cudaMalloc(&d_ptrs, kE * sizeof(void*));
        cudaMemcpy(d_ptrs, devw.data(), kE * sizeof(void*),
                   cudaMemcpyHostToDevice);
        return const_cast<const void**>(d_ptrs);
    };
    std::vector<void*> devw_gate, devw_up, devw_down;
    const void** d_b_gate = upload_experts(pk_gate, devw_gate);
    const void** d_b_up   = upload_experts(pk_up, devw_up);
    const void** d_b_down = upload_experts(pk_down, devw_down);

    std::vector<__nv_bfloat16> zero_out(static_cast<size_t>(kN) * kH,
                                        f2bf(0.0f));
    __nv_bfloat16* d_out_single = upload(zero_out);
    __nv_bfloat16* d_out_chunked = upload(zero_out);

    // (a) single-shot: one pass over all kN tokens (full-size transients).
    DeviceBufs bufs_single;
    bufs_single.alloc(kN);
    run_pass(dev.get(), bufs_single, strategy, d_hidden, d_topk_idx, d_topk_w,
             d_b_gate, d_b_up, d_b_down, d_out_single, /*row0=*/0, /*rows=*/kN);
    CU_CHECK(cudaDeviceSynchronize());

    // (b) chunked: kChunk-token chunks, transients sized at the CHUNK and
    // reused (the TD-PREFILL-MOE-BIG chunk loop).
    DeviceBufs bufs_chunk;
    bufs_chunk.alloc(kChunk);
    for (int off = 0; off < kN; off += kChunk) {
        const int len = std::min(kChunk, kN - off);
        run_pass(dev.get(), bufs_chunk, strategy, d_hidden, d_topk_idx,
                 d_topk_w, d_b_gate, d_b_up, d_b_down, d_out_chunked,
                 off, len);
    }
    CU_CHECK(cudaDeviceSynchronize());

    // ── Compare ───────────────────────────────────────────────────────────
    std::vector<__nv_bfloat16> h_single(static_cast<size_t>(kN) * kH);
    std::vector<__nv_bfloat16> h_chunked(h_single.size());
    CU_CHECK(cudaMemcpy(h_single.data(), d_out_single,
                        h_single.size() * 2, cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(h_chunked.data(), d_out_chunked,
                        h_chunked.size() * 2, cudaMemcpyDeviceToHost));

    // Chunked == single-shot, BITWISE. The chunk split partitions tokens; a
    // token's K-slot unpermute sum and its per-expert GEMM rows are computed
    // identically in both variants.
    size_t mismatches = 0;
    for (size_t i = 0; i < h_single.size(); ++i) {
        if (std::memcmp(&h_single[i], &h_chunked[i], 2) != 0) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0u)
        << "chunked vs single-shot: " << mismatches << " / "
        << h_single.size() << " bf16 values differ";

    // Both match the FP64 CPU reference (global cosine; the int strategy
    // carries Q8_1 activation-quant noise → looser bound).
    std::vector<float> f_single(h_single.size()), f_chunked(h_single.size());
    for (size_t i = 0; i < h_single.size(); ++i) {
        f_single[i] = bf2f(h_single[i]);
        f_chunked[i] = bf2f(h_chunked[i]);
    }
    const double c_single = cosine(f_single, ref);
    const double c_chunked = cosine(f_chunked, ref);
    EXPECT_GE(c_single, min_cos_ref) << "single-shot vs CPU ref";
    EXPECT_GE(c_chunked, min_cos_ref) << "chunked vs CPU ref";

    // Cleanup.
    bufs_single.free_all();
    bufs_chunk.free_all();
    cudaFree(d_hidden); cudaFree(d_topk_idx); cudaFree(d_topk_w);
    cudaFree(d_out_single); cudaFree(d_out_chunked);
    cudaFree(const_cast<void*>(static_cast<const void*>(d_b_gate)));
    cudaFree(const_cast<void*>(static_cast<const void*>(d_b_up)));
    cudaFree(const_cast<void*>(static_cast<const void*>(d_b_down)));
    for (auto& v : {devw_gate, devw_up, devw_down})
        for (void* p : v) cudaFree(p);
}

// ── TD-MOE-BIG-GEMM-SWEEP: wave-masked permute identity ─────────────────────
// The chunked ROLLING-WAVE path (dispatch_moe_big.cpp) processes the routed
// union in expert waves: per wave, per chunk, permute → gate/up GEMM → SwiGLU
// → down GEMM → unpermute into a chunk tmp → residual-add into the [N, H]
// accumulator. Two variants of that exact sequence:
//   UNMASKED (legacy): full permute; absent (non-wave) experts run the
//     zero-weight GEMM over all their permuted rows (exact-zero outputs);
//   MASKED (production, LS_MOE_WAVE_MASK): non-wave experts' top-K entries →
//     -1 permute sentinel (rows sort past expert_offsets[E], zero-length GEMM
//     groups) + a down-output memset so masked rows contribute exact +0.
// Asserts masked == unmasked BITWISE (the fix's identity claim: same rows per
// live expert in the same intra-expert order + exact zeros either way), and
// the wave result ≈ the CPU reference.
void run_wave_pass(lc::ExpertDevice* dev, DeviceBufs& b,
                   lc::GgufGemmStrategy strategy, bool masked,
                   const uint8_t* d_mask, int32_t* d_masked_topk,
                   const __nv_bfloat16* hidden_dev,
                   const int32_t* topk_idx_dev, const float* topk_w_dev,
                   const void** b_gate, const void** b_up,
                   const void** b_down, __nv_bfloat16* unperm_tmp,
                   __nv_bfloat16* out_dev, int row0, int rows) {
    const int exp_rows = rows * kT;
    void* stream = nullptr;  // default stream

    const ld::MoeGemmEmitter emit{
        dev, b.gemm_ws, b.gemm_ws_bytes, stream,
        kE, /*use_fp8=*/false, /*use_gguf=*/true, strategy,
        /*total_tokens=*/exp_rows,
        b.expert_offsets, b.sf_offsets, b.problem_sizes};

    // Step 2: permute (wave-masked variant drops non-wave experts' entries).
    const int32_t* topk_ptr = topk_idx_dev + static_cast<size_t>(row0) * kT;
    if (masked) {
        lc::launch_mask_topk_indices(d_masked_topk, topk_ptr, d_mask,
                                     exp_rows, kE, stream);
        topk_ptr = d_masked_topk;
    }
    dev->moe_permute(
        b.permuted, b.expert_offsets, b.src_to_dest, b.permuted_idx,
        hidden_dev + static_cast<size_t>(row0) * kH, topk_ptr,
        rows, kT, kH, kE, /*elem_size_bytes=*/2, b.permute_ws, stream);

    // Steps 3a/3b: gate + up GEMM.
    emit.routed_gemm(kI, kH, b.permuted, b.act_gate, b_gate, nullptr,
                     nullptr, nullptr, lc::GgufQuantType::Q8_0);
    emit.routed_gemm(kI, kH, b.permuted, b.act_up, b_up, nullptr,
                     nullptr, nullptr, lc::GgufQuantType::Q8_0);

    // Step 4: interleave + SwiGLU.
    const size_t I_bytes = static_cast<size_t>(kI) * 2;
    cudaMemcpy2DAsync(b.gate_up, I_bytes * 2, b.act_gate, I_bytes,
                      I_bytes, exp_rows, cudaMemcpyDeviceToDevice, nullptr);
    cudaMemcpy2DAsync(reinterpret_cast<uint8_t*>(b.gate_up) + I_bytes,
                      I_bytes * 2, b.act_up, I_bytes, I_bytes, exp_rows,
                      cudaMemcpyDeviceToDevice, nullptr);
    ld::launch_swiglu(dev, b.act_gate, b.gate_up, exp_rows, kI, stream,
                      /*swiglu_limit=*/0.0f);

    // Masked: zero the down output so masked rows read exact +0 in the
    // unpermute (mirrors the dispatch_moe_big.cpp memset before Step 5).
    if (masked)
        cudaMemsetAsync(b.down_out, 0,
                        static_cast<size_t>(exp_rows) * kH * 2, nullptr);

    // Step 5: down GEMM.
    emit.routed_gemm(kH, kI, b.act_gate, b.down_out, b_down, nullptr,
                     nullptr, nullptr, lc::GgufQuantType::Q8_0);

    // Step 6 (wave variant): unpermute into the chunk tmp, accumulate into
    // the [N, H] output rows (INV-MOE-BIG-2 wave accumulation).
    dev->moe_unpermute(
        unperm_tmp, b.down_out,
        topk_w_dev + static_cast<size_t>(row0) * kT, b.src_to_dest,
        rows, kT, kH, /*elem_size_bytes=*/2, stream,
        lc::MoeCombineMode::kReducedBf16);
    lc::launch_residual_add(
        out_dev + static_cast<size_t>(row0) * kH, unperm_tmp,
        rows * kH, stream);
}

void run_wave_mask_identity_case(lc::GgufGemmStrategy strategy,
                                 double min_cos_ref) {
    if (!ik::gguf_supported(ik::GgufType::q8_0))
        GTEST_SKIP() << "ik q8_0 unsupported in this build";

    // Identical data setup to run_identity_case (same seed → same weights,
    // routing, and CPU reference).
    std::mt19937 rng(0x5EED);
    std::uniform_real_distribution<float> wd(-0.5f, 0.5f), ad(-1.0f, 1.0f);

    std::vector<__nv_bfloat16> hidden_bf(static_cast<size_t>(kN) * kH);
    std::vector<float> hidden_f(hidden_bf.size());
    for (size_t i = 0; i < hidden_bf.size(); ++i) {
        hidden_bf[i] = f2bf(ad(rng));
        hidden_f[i] = bf2f(hidden_bf[i]);
    }
    std::vector<int32_t> topk_idx(static_cast<size_t>(kN) * kT);
    std::vector<float> topk_w(topk_idx.size());
    std::uniform_int_distribution<int> ed(0, kE - 1);
    for (int t = 0; t < kN; ++t) {
        int e0 = ed(rng);
        int e1 = (e0 + 1 + ed(rng) % (kE - 1)) % kE;
        topk_idx[t * kT + 0] = e0;
        topk_idx[t * kT + 1] = e1;
        topk_w[t * kT + 0] = 0.25f + 0.5f * ad(rng) * ad(rng);
        topk_w[t * kT + 1] = 1.0f - topk_w[t * kT + 0];
    }

    const size_t row_h = ik::weight_row_bytes(ik::GgufType::q8_0, kH);
    const size_t row_i = ik::weight_row_bytes(ik::GgufType::q8_0, kI);
    std::vector<std::vector<uint8_t>> pk_gate(kE), pk_up(kE), pk_down(kE);
    std::vector<std::vector<float>> dq_gate(kE), dq_up(kE), dq_down(kE);
    auto make_w = [&](std::vector<uint8_t>& pk, std::vector<float>& dq,
                      int n, int k, size_t row_bytes) {
        pk.resize(static_cast<size_t>(n) * row_bytes);
        dq.resize(static_cast<size_t>(n) * k);
        std::vector<float> wr(k);
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < k; ++c) wr[c] = wd(rng);
            void* dst = pk.data() + static_cast<size_t>(r) * row_bytes;
            ik::quantize_weight(ik::GgufType::q8_0, wr.data(), dst, k);
            ik::dequantize_weight(ik::GgufType::q8_0, dst,
                                  dq.data() + static_cast<size_t>(r) * k, k);
        }
    };
    for (int e = 0; e < kE; ++e) {
        make_w(pk_gate[e], dq_gate[e], kI, kH, row_h);
        make_w(pk_up[e],   dq_up[e],   kI, kH, row_h);
        make_w(pk_down[e], dq_down[e], kH, kI, row_i);
    }

    // CPU FP64 reference (identical to run_identity_case).
    std::vector<float> ref(static_cast<size_t>(kN) * kH, 0.0f);
    std::vector<double> gate_o(kI), up_o(kI), act(kI);
    for (int t = 0; t < kN; ++t) {
        std::vector<double> acc(kH, 0.0);
        for (int s = 0; s < kT; ++s) {
            const int e = topk_idx[t * kT + s];
            const float w = topk_w[t * kT + s];
            const float* x = hidden_f.data() + static_cast<size_t>(t) * kH;
            for (int r = 0; r < kI; ++r) {
                double g = 0, u = 0;
                const float* wg = dq_gate[e].data() + static_cast<size_t>(r) * kH;
                const float* wu = dq_up[e].data() + static_cast<size_t>(r) * kH;
                for (int c = 0; c < kH; ++c) { g += double(x[c]) * wg[c];
                                               u += double(x[c]) * wu[c]; }
                gate_o[r] = g; up_o[r] = u;
                act[r] = (g / (1.0 + std::exp(-g))) * u;
            }
            for (int r = 0; r < kH; ++r) {
                double d = 0;
                const float* wdn = dq_down[e].data() + static_cast<size_t>(r) * kI;
                for (int c = 0; c < kI; ++c) d += act[c] * wdn[c];
                acc[r] += double(w) * d;
            }
        }
        for (int r = 0; r < kH; ++r)
            ref[static_cast<size_t>(t) * kH + r] = static_cast<float>(acc[r]);
    }

    // Device setup.
    lcfg::GpuRef gpu{0, 0, lcfg::GpuType::rtx5090};
    auto dev = lc::make_cuda_sm120_expert_device(gpu);
    ASSERT_NE(dev, nullptr);
    dev->set_device();

    __nv_bfloat16* d_hidden = upload(hidden_bf);
    int32_t* d_topk_idx = upload(topk_idx);
    float* d_topk_w = upload(topk_w);

    std::vector<void*> devw_gate(kE), devw_up(kE), devw_down(kE);
    for (int e = 0; e < kE; ++e) {
        devw_gate[e] = upload(pk_gate[e]);
        devw_up[e]   = upload(pk_up[e]);
        devw_down[e] = upload(pk_down[e]);
    }

    // Zero weight buffer for absent (non-wave) experts, sized for the largest
    // projection (mirrors MoeScratch::zero_weight_buf).
    const size_t zero_bytes =
        std::max(static_cast<size_t>(kI) * row_h,
                 static_cast<size_t>(kH) * row_i);
    void* d_zero_w = nullptr;
    CU_CHECK(cudaMalloc(&d_zero_w, zero_bytes));
    CU_CHECK(cudaMemset(d_zero_w, 0, zero_bytes));

    // Shared per-wave device pointer arrays (re-uploaded per wave, stream-
    // ordered exactly like the production shared routed_b_ptrs array).
    void** d_bw_gate = nullptr; void** d_bw_up = nullptr; void** d_bw_down = nullptr;
    CU_CHECK(cudaMalloc(&d_bw_gate, kE * sizeof(void*)));
    CU_CHECK(cudaMalloc(&d_bw_up,   kE * sizeof(void*)));
    CU_CHECK(cudaMalloc(&d_bw_down, kE * sizeof(void*)));

    // Wave-mask scratch.
    uint8_t* d_mask = nullptr;
    int32_t* d_masked_topk = nullptr;
    __nv_bfloat16* d_unperm_tmp = nullptr;
    CU_CHECK(cudaMalloc(&d_mask, kE));
    CU_CHECK(cudaMalloc(&d_masked_topk,
                        static_cast<size_t>(kChunk) * kT * sizeof(int32_t)));
    CU_CHECK(cudaMalloc(&d_unperm_tmp, static_cast<size_t>(kChunk) * kH * 2));

    // Expert waves (union > "free slots" split): 3 rolling waves.
    const std::vector<std::vector<int>> waves = {{0, 1, 2}, {3, 4, 5}, {6, 7}};

    DeviceBufs bufs;
    bufs.alloc(kChunk);

    std::vector<__nv_bfloat16> zero_out(static_cast<size_t>(kN) * kH,
                                        f2bf(0.0f));

    // (void lambda: gtest ASSERT macros require a void-returning scope.)
    auto run_variant = [&](bool masked, __nv_bfloat16* d_out) {
        std::vector<const void*> bw_gate(kE), bw_up(kE), bw_down(kE);
        std::vector<uint8_t> mask_h(kE);
        for (const auto& wave : waves) {
            std::fill(mask_h.begin(), mask_h.end(), 0);
            for (int e = 0; e < kE; ++e) {
                bw_gate[e] = d_zero_w; bw_up[e] = d_zero_w; bw_down[e] = d_zero_w;
            }
            for (int e : wave) {
                mask_h[e] = 1;
                bw_gate[e] = devw_gate[e];
                bw_up[e]   = devw_up[e];
                bw_down[e] = devw_down[e];
            }
            CU_CHECK(cudaMemcpy(d_bw_gate, bw_gate.data(), kE * sizeof(void*),
                                cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_bw_up, bw_up.data(), kE * sizeof(void*),
                                cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_bw_down, bw_down.data(), kE * sizeof(void*),
                                cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_mask, mask_h.data(), kE,
                                cudaMemcpyHostToDevice));
            for (int off = 0; off < kN; off += kChunk) {
                const int len = std::min(kChunk, kN - off);
                run_wave_pass(dev.get(), bufs, strategy, masked, d_mask,
                              d_masked_topk, d_hidden, d_topk_idx, d_topk_w,
                              const_cast<const void**>(d_bw_gate),
                              const_cast<const void**>(d_bw_up),
                              const_cast<const void**>(d_bw_down),
                              d_unperm_tmp, d_out, off, len);
            }
        }
        CU_CHECK(cudaDeviceSynchronize());
    };

    __nv_bfloat16* d_out_unmasked = upload(zero_out);  // [N, H] accumulators
    __nv_bfloat16* d_out_masked   = upload(zero_out);
    run_variant(/*masked=*/false, d_out_unmasked);
    run_variant(/*masked=*/true, d_out_masked);

    std::vector<__nv_bfloat16> h_unmasked(static_cast<size_t>(kN) * kH);
    std::vector<__nv_bfloat16> h_masked(h_unmasked.size());
    CU_CHECK(cudaMemcpy(h_unmasked.data(), d_out_unmasked,
                        h_unmasked.size() * 2, cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(h_masked.data(), d_out_masked,
                        h_masked.size() * 2, cudaMemcpyDeviceToHost));

    // Masked == unmasked, BITWISE: a live expert's rows keep their relative
    // intra-expert order (stable radix sort) so its GEMM rows are computed
    // identically at either offset, and absent rows are exact zeros both ways
    // (zero-weight GEMM vs memset).
    size_t mismatches = 0;
    for (size_t i = 0; i < h_unmasked.size(); ++i) {
        if (std::memcmp(&h_unmasked[i], &h_masked[i], 2) != 0) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0u)
        << "wave-masked vs zero-weight-sweep: " << mismatches << " / "
        << h_unmasked.size() << " bf16 values differ";

    // Both ≈ CPU reference (waves round per wave boundary → token-level
    // identity vs single-shot, INV-MOE-BIG-2; cosine bound as elsewhere).
    std::vector<float> f_unmasked(h_unmasked.size()), f_masked(h_masked.size());
    for (size_t i = 0; i < h_unmasked.size(); ++i) {
        f_unmasked[i] = bf2f(h_unmasked[i]);
        f_masked[i] = bf2f(h_masked[i]);
    }
    EXPECT_GE(cosine(f_unmasked, ref), min_cos_ref) << "unmasked waves vs CPU ref";
    EXPECT_GE(cosine(f_masked, ref), min_cos_ref) << "masked waves vs CPU ref";

    // Cleanup.
    bufs.free_all();
    cudaFree(d_hidden); cudaFree(d_topk_idx); cudaFree(d_topk_w);
    cudaFree(d_out_unmasked); cudaFree(d_out_masked);
    cudaFree(d_zero_w); cudaFree(d_mask); cudaFree(d_masked_topk);
    cudaFree(d_unperm_tmp);
    cudaFree(d_bw_gate); cudaFree(d_bw_up); cudaFree(d_bw_down);
    for (auto& v : {devw_gate, devw_up, devw_down})
        for (void* p : v) cudaFree(p);
}

}  // namespace

// Dequant strategy: lossless activations → tight reference bound.
TEST(MoeBigChunkedIdentity, GgufDequant_ChunkedEqualsSingleShot) {
    REQUIRES_GPU();
    run_identity_case(lc::GgufGemmStrategy::dequant, 0.999);
}

// Int strategy (production GLM-5.2 default): Q8_1 activation-quant noise in
// BOTH variants identically; the bitwise chunked==single assert still holds.
TEST(MoeBigChunkedIdentity, GgufInt_ChunkedEqualsSingleShot) {
    REQUIRES_GPU();
    run_identity_case(lc::GgufGemmStrategy::int_strategy, 0.995);
}

// TD-MOE-BIG-GEMM-SWEEP: wave-masked permute == zero-weight sweep, bitwise
// (the production rolling-wave chunk sequence, 3 waves over 8 experts).
TEST(MoeBigChunkedIdentity, GgufInt_WaveMaskedEqualsZeroWeightSweep) {
    REQUIRES_GPU();
    run_wave_mask_identity_case(lc::GgufGemmStrategy::int_strategy, 0.995);
}

TEST(MoeBigChunkedIdentity, GgufDequant_WaveMaskedEqualsZeroWeightSweep) {
    REQUIRES_GPU();
    run_wave_mask_identity_case(lc::GgufGemmStrategy::dequant, 0.999);
}
