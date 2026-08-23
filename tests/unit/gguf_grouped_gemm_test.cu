// GG-5: GPU GGUF grouped-GEMM numerical test.
//
// Exercises launch_gguf_grouped_gemm (the per-expert dispatch loop) on real
// hardware: synthetic experts with varying per-expert token counts (incl. the
// M_e<=8 mmvq path and the M_e>8 mmq_mma path), for Q8_0 (QK=32) and a k-quant
// (Q4_K, QK=256). Packs weights via the ik CPU reference packer (byte-identical
// to ggml block layout, which the GPU kernel consumes), dequantizes the SAME
// packed bytes for an independent FP32 reference, and checks cosine similarity:
//   int strategy     >= 0.999  (carries Q8_1 activation-quant noise)
//   dequant strategy >= 0.9999 (lossless activations)
//
// GPU-required (REQUIRES_GPU); SKIPPED on headless CI.

#include "sm120/gemm/gguf/gguf_grouped_gemm.h"
#include "sm120/gemm/gguf/gguf_dequant_gemm.h"  // gguf_block_bytes/values

#include "compute/cpu/ik_vendor/ik_gguf_gemm.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace ik = layerstorm::compute::cpu::ik;
using namespace layerstorm::compute;

namespace {

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

// One full grouped-GEMM case. ik_t/krn_t are the matching CPU/GPU type tags.
void run_case(ik::GgufType ik_t, GgufType krn_t, GgufGroupedStrategy strat,
              int N, int K, const std::vector<int>& tokens_per_expert,
              double min_cos, uint32_t seed) {
    if (!ik::gguf_supported(ik_t)) {
        GTEST_SKIP() << "ik type unsupported in this build";
    }
    const int num_experts = static_cast<int>(tokens_per_expert.size());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wd(-0.6f, 0.6f), ad(-1.0f, 1.0f);

    // Offsets + total tokens.
    std::vector<int32_t> offsets(num_experts + 1, 0);
    int total = 0;
    for (int e = 0; e < num_experts; ++e) {
        total += tokens_per_expert[e];
        offsets[e + 1] = total;
    }
    ASSERT_GT(total, 0);

    // Activations A [total, K] (BF16 rounded for both GPU input and reference).
    std::vector<__nv_bfloat16> A_bf(static_cast<size_t>(total) * K);
    std::vector<float> A_f(static_cast<size_t>(total) * K);
    for (size_t i = 0; i < A_f.size(); ++i) {
        float v = ad(rng);
        A_bf[i] = f2bf(v);
        A_f[i] = bf2f(A_bf[i]);
    }

    // Per-expert packed weights [N, K] + parallel FP32 dequant for reference.
    const size_t row_bytes = ik::weight_row_bytes(ik_t, K);
    // Cross-check the GPU kernel's expected packed-row size matches ik's.
    ASSERT_EQ(static_cast<int>(row_bytes),
              (K / gguf_block_values(krn_t)) * gguf_block_bytes(krn_t));

    std::vector<std::vector<uint8_t>> packed(num_experts);
    std::vector<std::vector<float>> wdq(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        packed[e].resize(static_cast<size_t>(N) * row_bytes);
        wdq[e].resize(static_cast<size_t>(N) * K);
        for (int n = 0; n < N; ++n) {
            std::vector<float> wr(K);
            for (int k = 0; k < K; ++k) wr[k] = wd(rng);
            void* dst = packed[e].data() + static_cast<size_t>(n) * row_bytes;
            ik::quantize_weight(ik_t, wr.data(), dst, K);
            ik::dequantize_weight(ik_t, dst, wdq[e].data() + static_cast<size_t>(n) * K, K);
        }
    }

    // Reference D[m,n] = sum_k A_f[m,k] * Wdq_e[n,k], per expert segment.
    std::vector<float> D_ref(static_cast<size_t>(total) * N, 0.0f);
    for (int e = 0; e < num_experts; ++e) {
        for (int m = offsets[e]; m < offsets[e + 1]; ++m)
            for (int n = 0; n < N; ++n) {
                double acc = 0;
                const float* ar = A_f.data() + static_cast<size_t>(m) * K;
                const float* wr = wdq[e].data() + static_cast<size_t>(n) * K;
                for (int k = 0; k < K; ++k) acc += double(ar[k]) * wr[k];
                D_ref[static_cast<size_t>(m) * N + n] = float(acc);
            }
    }

    // ── Upload to device ──────────────────────────────────────────────────────
    __nv_bfloat16* dA = upload(A_bf);
    int32_t* dOff = upload(offsets);
    std::vector<__nv_bfloat16> D_init(static_cast<size_t>(total) * N, f2bf(0.0f));
    __nv_bfloat16* dD = upload(D_init);

    std::vector<void*> dW(num_experts);
    for (int e = 0; e < num_experts; ++e) dW[e] = upload(packed[e]);
    void** dBptrs = nullptr;
    cudaMalloc(&dBptrs, num_experts * sizeof(void*));
    cudaMemcpy(dBptrs, dW.data(), num_experts * sizeof(void*), cudaMemcpyHostToDevice);

    void* ws = nullptr;
    size_t ws_bytes = 0;
    if (strat == GgufGroupedStrategy::Int) {
        ws_bytes = gguf_grouped_gemm_workspace_bytes(total, K, num_experts);
        cudaMalloc(&ws, ws_bytes);
    }

    GgufGroupedGemmKernelParams p{};
    p.type = krn_t;
    p.strategy = strat;
    p.num_experts = num_experts;
    p.N = N;
    p.K = K;
    p.A_base = dA;
    p.D_base = dD;
    p.expert_offsets = dOff;
    p.B_ptrs = const_cast<const void**>(dBptrs);

    launch_gguf_grouped_gemm(p, total, ws, ws_bytes, /*stream=*/0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> D_gpu_bf(static_cast<size_t>(total) * N);
    cudaMemcpy(D_gpu_bf.data(), dD, D_gpu_bf.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    std::vector<float> D_gpu(D_gpu_bf.size());
    for (size_t i = 0; i < D_gpu.size(); ++i) D_gpu[i] = bf2f(D_gpu_bf[i]);

    double c = cosine(D_gpu, D_ref);
    EXPECT_GE(c, min_cos) << "cosine=" << c << " N=" << N << " K=" << K;

    cudaFree(dA); cudaFree(dOff); cudaFree(dD); cudaFree(dBptrs);
    for (void* p2 : dW) cudaFree(p2);
    if (ws) cudaFree(ws);
}

}  // namespace

// Q8_0 (QK=32), int strategy: mix of M_e<=8 (mmvq) and M_e>8 (mmq_mma).
TEST(GgufGroupedGemmGpu, Q8_0_Int_MixedM) {
    REQUIRES_GPU();
    run_case(ik::GgufType::q8_0, GgufType::Q8_0, GgufGroupedStrategy::Int,
             /*N=*/128, /*K=*/256, /*tokens=*/{1, 4, 8, 17, 0, 32}, 0.999, 0xA1);
}

// Q8_0 dequant strategy: lossless activations → tighter bound.
TEST(GgufGroupedGemmGpu, Q8_0_Dequant_MixedM) {
    REQUIRES_GPU();
    run_case(ik::GgufType::q8_0, GgufType::Q8_0, GgufGroupedStrategy::Dequant,
             /*N=*/128, /*K=*/256, /*tokens=*/{2, 8, 16}, 0.9999, 0xB2);
}

// Q4_K (QK=256) k-quant, int strategy.
TEST(GgufGroupedGemmGpu, Q4_K_Int_MixedM) {
    REQUIRES_GPU();
    run_case(ik::GgufType::q4_k, GgufType::Q4_K, GgufGroupedStrategy::Int,
             /*N=*/128, /*K=*/256, /*tokens=*/{1, 8, 24, 0, 5}, 0.999, 0xC3);
}

// Q4_K dequant strategy.
TEST(GgufGroupedGemmGpu, Q4_K_Dequant_MixedM) {
    REQUIRES_GPU();
    run_case(ik::GgufType::q4_k, GgufType::Q4_K, GgufGroupedStrategy::Dequant,
             /*N=*/128, /*K=*/512, /*tokens=*/{3, 12}, 0.9999, 0xD4);
}

// Empty grouped call is a no-op (num_experts==0 / all-empty).
TEST(GgufGroupedGemmGpu, AllEmptyExperts) {
    REQUIRES_GPU();
    run_case(ik::GgufType::q8_0, GgufType::Q8_0, GgufGroupedStrategy::Int,
             /*N=*/64, /*K=*/256, /*tokens=*/{0, 0, 4}, 0.999, 0xE5);
}

// ═══════════════════════════════════════════════════════════════════════════
// MXFP4 (ggml type 39 — V4 QAT routed experts). The ik CPU packer has no
// MXFP4, so the case generates random VALID packed blocks (uniform nibbles +
// a sane fixed E8M0 exponent) and builds the FP32 reference through the
// engine's host port of the llama.cpp type-39 dequant math
// (model::gguf::dequant_mxfp4_row — itself golden-tested CPU-only in
// gguf_kquant_test.cpp). Covers the int strategy across the mmvq (M_e<=8)
// and mmq_mma (M_e>8) crossovers plus the dequant fallback strategy.
// ═══════════════════════════════════════════════════════════════════════════

#include "model/quantization/gguf_kquant.h"

namespace {

void run_mxfp4_case(GgufGroupedStrategy strat, int N, int K,
                    const std::vector<int>& tokens_per_expert,
                    double min_cos, uint32_t seed) {
    namespace mg = layerstorm::model::gguf;
    const int num_experts = static_cast<int>(tokens_per_expert.size());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ad(-1.0f, 1.0f);
    std::uniform_int_distribution<int> nib(0, 255);

    std::vector<int32_t> offsets(num_experts + 1, 0);
    int total = 0;
    for (int e = 0; e < num_experts; ++e) {
        total += tokens_per_expert[e];
        offsets[e + 1] = total;
    }
    ASSERT_GT(total, 0);

    std::vector<__nv_bfloat16> A_bf(static_cast<size_t>(total) * K);
    std::vector<float> A_f(A_bf.size());
    for (size_t i = 0; i < A_f.size(); ++i) {
        float v = ad(rng);
        A_bf[i] = f2bf(v);
        A_f[i] = bf2f(A_bf[i]);
    }

    // Packed rows: (K/32) 17-byte blocks; e = 121 → d = 2^-7 (|w| <= 0.094).
    const size_t row_bytes = static_cast<size_t>(K / 32) * 17;
    ASSERT_EQ(static_cast<int>(row_bytes),
              (K / gguf_block_values(GgufType::MXFP4)) *
                  gguf_block_bytes(GgufType::MXFP4));
    std::vector<std::vector<uint8_t>> packed(num_experts);
    std::vector<std::vector<float>> wdq(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        packed[e].resize(static_cast<size_t>(N) * row_bytes);
        wdq[e].resize(static_cast<size_t>(N) * K);
        for (size_t i = 0; i < packed[e].size(); ++i)
            packed[e][i] = static_cast<uint8_t>(nib(rng));
        for (int n = 0; n < N; ++n) {
            uint8_t* row = packed[e].data() + static_cast<size_t>(n) * row_bytes;
            for (int b = 0; b < K / 32; ++b) row[b * 17] = 121;  // sane E8M0
            mg::dequant_mxfp4_row(row, wdq[e].data() + static_cast<size_t>(n) * K, K);
        }
    }

    std::vector<float> D_ref(static_cast<size_t>(total) * N, 0.0f);
    for (int e = 0; e < num_experts; ++e) {
        for (int m = offsets[e]; m < offsets[e + 1]; ++m)
            for (int n = 0; n < N; ++n) {
                double acc = 0;
                const float* ar = A_f.data() + static_cast<size_t>(m) * K;
                const float* wr = wdq[e].data() + static_cast<size_t>(n) * K;
                for (int k = 0; k < K; ++k) acc += double(ar[k]) * wr[k];
                D_ref[static_cast<size_t>(m) * N + n] = float(acc);
            }
    }

    __nv_bfloat16* dA = upload(A_bf);
    int32_t* dOff = upload(offsets);
    std::vector<__nv_bfloat16> D_init(static_cast<size_t>(total) * N, f2bf(0.0f));
    __nv_bfloat16* dD = upload(D_init);
    std::vector<void*> dW(num_experts);
    for (int e = 0; e < num_experts; ++e) dW[e] = upload(packed[e]);
    void** dBptrs = nullptr;
    cudaMalloc(&dBptrs, num_experts * sizeof(void*));
    cudaMemcpy(dBptrs, dW.data(), num_experts * sizeof(void*), cudaMemcpyHostToDevice);

    void* ws = nullptr;
    size_t ws_bytes = 0;
    if (strat == GgufGroupedStrategy::Int) {
        ws_bytes = gguf_grouped_gemm_workspace_bytes(total, K, num_experts);
        cudaMalloc(&ws, ws_bytes);
    }

    GgufGroupedGemmKernelParams p{};
    p.type = GgufType::MXFP4;
    p.strategy = strat;
    p.num_experts = num_experts;
    p.N = N;
    p.K = K;
    p.A_base = dA;
    p.D_base = dD;
    p.expert_offsets = dOff;
    p.B_ptrs = const_cast<const void**>(dBptrs);

    launch_gguf_grouped_gemm(p, total, ws, ws_bytes, /*stream=*/0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> D_gpu_bf(static_cast<size_t>(total) * N);
    cudaMemcpy(D_gpu_bf.data(), dD, D_gpu_bf.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    std::vector<float> D_gpu(D_gpu_bf.size());
    for (size_t i = 0; i < D_gpu.size(); ++i) D_gpu[i] = bf2f(D_gpu_bf[i]);

    double c = cosine(D_gpu, D_ref);
    EXPECT_GE(c, min_cos) << "cosine=" << c << " N=" << N << " K=" << K;

    cudaFree(dA); cudaFree(dOff); cudaFree(dD); cudaFree(dBptrs);
    for (void* p2 : dW) cudaFree(p2);
    if (ws) cudaFree(ws);
}

}  // namespace

// MXFP4 int strategy across both crossovers (mmvq M_e<=8 + mmq_mma M_e>8).
TEST(GgufGroupedGemmGpu, MXFP4_Int_MixedM) {
    REQUIRES_GPU();
    run_mxfp4_case(GgufGroupedStrategy::Int,
                   /*N=*/128, /*K=*/256, /*tokens=*/{1, 4, 8, 17, 0, 32},
                   0.999, 0xF6);
}

// MXFP4 dequant fallback strategy (lossless activations → tighter bound).
TEST(GgufGroupedGemmGpu, MXFP4_Dequant_MixedM) {
    REQUIRES_GPU();
    run_mxfp4_case(GgufGroupedStrategy::Dequant,
                   /*N=*/128, /*K=*/512, /*tokens=*/{3, 12},
                   0.9999, 0xA7);
}

// MXFP4 int strategy, V4-Flash-shaped projection K (4096) at decode-like M.
TEST(GgufGroupedGemmGpu, MXFP4_Int_V4FlashK) {
    REQUIRES_GPU();
    run_mxfp4_case(GgufGroupedStrategy::Int,
                   /*N=*/64, /*K=*/4096, /*tokens=*/{1, 2, 6},
                   0.999, 0xB8);
}

// ── MXFP4 vs Q8_0 grouped speed probe (V4-Flash gate/up shape) ──────────────
// Prints median wall us for the int strategy at N=2048, K=4096 over 32 active
// experts with decode-like (M_e<=8) and prefill-like (M_e=64) distributions.
// Informational (no assert beyond completion); numbers land in DS4_DOSSIER §B.

namespace {

double time_grouped_case(GgufType t, size_t row_bytes17_or_34, int N, int K,
                         const std::vector<int>& toks, uint32_t seed) {
    const int num_experts = static_cast<int>(toks.size());
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> nib(0, 255);
    std::vector<int32_t> offsets(num_experts + 1, 0);
    int total = 0;
    for (int e = 0; e < num_experts; ++e) { total += toks[e]; offsets[e+1] = total; }

    std::vector<__nv_bfloat16> A_bf(static_cast<size_t>(total) * K, f2bf(0.01f));
    const size_t row_bytes = static_cast<size_t>(K / 32) * row_bytes17_or_34;
    std::vector<uint8_t> one_w(static_cast<size_t>(N) * row_bytes);
    for (auto& b : one_w) b = static_cast<uint8_t>(nib(rng));
    if (t == GgufType::MXFP4)
        for (size_t off = 0; off < one_w.size(); off += 17) one_w[off] = 121;

    __nv_bfloat16* dA = upload(A_bf);
    int32_t* dOff = upload(offsets);
    std::vector<__nv_bfloat16> D_init(static_cast<size_t>(total) * N, f2bf(0.0f));
    __nv_bfloat16* dD = upload(D_init);
    std::vector<void*> dW(num_experts);
    for (int e = 0; e < num_experts; ++e) dW[e] = upload(one_w);
    void** dBptrs = nullptr;
    cudaMalloc(&dBptrs, num_experts * sizeof(void*));
    cudaMemcpy(dBptrs, dW.data(), num_experts * sizeof(void*), cudaMemcpyHostToDevice);
    size_t ws_bytes = gguf_grouped_gemm_workspace_bytes(total, K, num_experts);
    void* ws = nullptr;
    cudaMalloc(&ws, ws_bytes);

    GgufGroupedGemmKernelParams p{};
    p.type = t; p.strategy = GgufGroupedStrategy::Int;
    p.num_experts = num_experts; p.N = N; p.K = K;
    p.A_base = dA; p.D_base = dD; p.expert_offsets = dOff;
    p.B_ptrs = const_cast<const void**>(dBptrs);

    for (int i = 0; i < 5; ++i)
        launch_gguf_grouped_gemm(p, total, ws, ws_bytes, 0);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    constexpr int kIters = 50;
    cudaEventRecord(t0);
    for (int i = 0; i < kIters; ++i)
        launch_gguf_grouped_gemm(p, total, ws, ws_bytes, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    cudaFree(dA); cudaFree(dOff); cudaFree(dD); cudaFree(dBptrs);
    for (void* q : dW) cudaFree(q);
    cudaFree(ws);
    return ms * 1000.0 / kIters;  // us per launch
}

}  // namespace

TEST(GgufGroupedGemmGpu, MXFP4_SpeedProbeVsQ8_0) {
    REQUIRES_GPU();
    const int N = 2048, K = 4096, E = 32;
    std::vector<int> decode(E), prefill(E);
    for (int e = 0; e < E; ++e) { decode[e] = 1 + (e % 6); prefill[e] = 64; }
    const double mx_dec = time_grouped_case(GgufType::MXFP4, 17, N, K, decode, 1);
    const double q8_dec = time_grouped_case(GgufType::Q8_0, 34, N, K, decode, 2);
    const double mx_pre = time_grouped_case(GgufType::MXFP4, 17, N, K, prefill, 3);
    const double q8_pre = time_grouped_case(GgufType::Q8_0, 34, N, K, prefill, 4);
    ::testing::Test::RecordProperty("mxfp4_decode_us", mx_dec);
    std::cerr << "[speed] grouped int N=2048 K=4096 E=32:\n"
              << "  decode-like (M_e 1..6): MXFP4 " << mx_dec << " us, Q8_0 "
              << q8_dec << " us\n"
              << "  prefill-like (M_e 64):  MXFP4 " << mx_pre << " us, Q8_0 "
              << q8_pre << " us\n";
    SUCCEED();
}
