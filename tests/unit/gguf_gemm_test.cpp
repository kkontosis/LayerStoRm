// PART-2/3 GGUF (ik_llama) CPU grouped-GEMM accuracy + speed tests.
//
// Exercises the vendored ik kernels via the shared CPU driver
// (cpu_gguf_grouped_gemm) and the bridge (ik::*). Accuracy is checked vs an
// independent FP32 reference: dequantize the GGUF weights (ik::dequantize_weight)
// and matmul against the (bf16-rounded) FP32 activations. The ik GEMM ALSO
// quantizes activations to Q8_x, so we allow a quant-noise tolerance.
//
// Q8_0 always runs (PART 2). Q4_K / Q6_K run only when LS_IK_HAVE_KQUANTS is
// defined (PART 3 vendored kquants); otherwise those cases are skipped.

#include "compute/cpu/cpu_gguf_gemm.h"
#include "compute/cpu/ik_vendor/ik_gguf_gemm.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace layerstorm::compute::cpu;

namespace {

uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}
float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Build a per-expert packed GGUF weight matrix [N, K] (row-major rows) from a
// dense FP32 source, returning the packed bytes + a parallel FP32 dequant for
// the reference. Returns {packed, dequant_fp32}.
struct PackedW {
    std::vector<uint8_t> packed;     // N rows * weight_row_bytes
    std::vector<float> dq;           // N*K dequantized weights
    size_t row_bytes;
};
PackedW pack_weight(ik::GgufType t, const std::vector<float>& W, int N, int K) {
    PackedW p;
    p.row_bytes = ik::weight_row_bytes(t, K);
    p.packed.resize(static_cast<size_t>(N) * p.row_bytes);
    p.dq.resize(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n) {
        const float* wr = W.data() + static_cast<size_t>(n) * K;
        void* dst = p.packed.data() + static_cast<size_t>(n) * p.row_bytes;
        ik::quantize_weight(t, wr, dst, K);
        ik::dequantize_weight(t, dst, p.dq.data() + static_cast<size_t>(n) * K, K);
    }
    return p;
}

// Reference matmul: D[m,n] = sum_k A_f32[m,k] * Wdq[n,k].
void ref_matmul(std::vector<float>& D, const std::vector<float>& Af,
                const std::vector<float>& Wdq, int M, int N, int K) {
    D.assign(static_cast<size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            for (int k = 0; k < K; ++k)
                acc += double(Af[static_cast<size_t>(m) * K + k]) *
                       double(Wdq[static_cast<size_t>(n) * K + k]);
            D[static_cast<size_t>(m) * N + n] = float(acc);
        }
}

// Run one accuracy case for a GGUF type + dims. rel_tol/abs_tol absorb the
// activation Q8_x quant noise (the ik GEMM quantizes acts; the ref does not).
void accuracy_case(ik::GgufType t, int N, int K, int num_experts,
                   const std::vector<int>& tokens_per_expert,
                   float rel_tol, float abs_tol, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wd(-0.6f, 0.6f), ad(-1.0f, 1.0f);

    int total = 0;
    std::vector<int32_t> offsets(num_experts + 1, 0);
    for (int e = 0; e < num_experts; ++e) {
        total += tokens_per_expert[e];
        offsets[e + 1] = total;
    }

    // Per-expert weights.
    std::vector<PackedW> pw(num_experts);
    std::vector<const void*> B_ptrs(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        std::vector<float> W(static_cast<size_t>(N) * K);
        for (auto& w : W) w = wd(rng);
        pw[e] = pack_weight(t, W, N, K);
        B_ptrs[e] = pw[e].packed.data();
    }

    // Activations (BF16, expert-contiguous).
    std::vector<uint16_t> A(static_cast<size_t>(total) * K);
    std::vector<float> Af(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = f32_to_bf16(ad(rng));
        Af[i] = bf16_to_f32(A[i]);
    }

    std::vector<uint16_t> D(static_cast<size_t>(total) * N, 0);
    cpu_gguf_grouped_gemm(D.data(), A.data(), t, B_ptrs.data(), offsets.data(),
                          num_experts, N, K, /*pool=*/nullptr);

    // Reference per expert.
    int max_bad = 0;
    float worst = 0;
    for (int e = 0; e < num_experts; ++e) {
        const int m0 = offsets[e], m1 = offsets[e + 1];
        const int me = m1 - m0;
        if (me == 0) continue;
        std::vector<float> Ae(static_cast<size_t>(me) * K);
        std::memcpy(Ae.data(), Af.data() + static_cast<size_t>(m0) * K,
                    Ae.size() * sizeof(float));
        std::vector<float> Ref;
        ref_matmul(Ref, Ae, pw[e].dq, me, N, K);
        for (int i = 0; i < me; ++i)
            for (int n = 0; n < N; ++n) {
                float got = bf16_to_f32(D[static_cast<size_t>(m0 + i) * N + n]);
                float ref = Ref[static_cast<size_t>(i) * N + n];
                float tol = std::fabs(ref) * rel_tol + abs_tol;
                worst = std::max(worst, std::fabs(got - ref));
                if (std::fabs(got - ref) > tol) ++max_bad;
            }
    }
    EXPECT_EQ(max_bad, 0) << "worst abs err=" << worst;
}

}  // namespace

// ── Q8_0 accuracy (PART 2) ───────────────────────────────────────────────────
TEST(GgufGemm, Q8_0_SingleExpert) {
    // K=256 (8 q8_0 blocks), N=64. One expert, several tokens.
    accuracy_case(ik::GgufType::q8_0, /*N=*/64, /*K=*/256, /*experts=*/1,
                  {5}, /*rel=*/0.03f, /*abs=*/0.05f, 0xC0FFEE);
}

TEST(GgufGemm, Q8_0_MultiExpert) {
    accuracy_case(ik::GgufType::q8_0, /*N=*/128, /*K=*/512, /*experts=*/3,
                  {1, 4, 2}, /*rel=*/0.03f, /*abs=*/0.05f, 0xD15EA5E);
}

TEST(GgufGemm, Q8_0_Supported) {
    EXPECT_TRUE(ik::gguf_supported(ik::GgufType::q8_0));
}

// ── Q5_0 accuracy (legacy path, always built) ────────────────────────────────
TEST(GgufGemm, Q5_0_MultiExpert) {
    // K=512 (16 q5_0 blocks of 32), N=128. 5-bit -> looser tol than Q8_0.
    accuracy_case(ik::GgufType::q5_0, /*N=*/128, /*K=*/512, /*experts=*/3,
                  {1, 4, 2}, /*rel=*/0.06f, /*abs=*/0.10f, 0x5005EED);
}

TEST(GgufGemm, Q5_0_Supported) {
    EXPECT_TRUE(ik::gguf_supported(ik::GgufType::q5_0));
}

// ── Q4_K / Q5_K / Q6_K accuracy (PART 3) ─────────────────────────────────────
#if defined(LS_IK_HAVE_KQUANTS)
TEST(GgufGemm, Q4_K_MultiExpert) {
    // K must be a multiple of QK_K=256 for K-quants.
    accuracy_case(ik::GgufType::q4_k, /*N=*/128, /*K=*/512, /*experts=*/2,
                  {3, 2}, /*rel=*/0.06f, /*abs=*/0.08f, 0x4BADF00D);
}
TEST(GgufGemm, Q5_K_MultiExpert) {
    // Q5_K accuracy lands between Q4_K and Q6_K.
    accuracy_case(ik::GgufType::q5_k, /*N=*/128, /*K=*/512, /*experts=*/2,
                  {3, 2}, /*rel=*/0.05f, /*abs=*/0.07f, 0x5BADF00D);
}
TEST(GgufGemm, Q6_K_MultiExpert) {
    accuracy_case(ik::GgufType::q6_k, /*N=*/128, /*K=*/512, /*experts=*/2,
                  {3, 2}, /*rel=*/0.04f, /*abs=*/0.06f, 0x6600D);
}
TEST(GgufGemm, Q5_K_Supported) {
    EXPECT_TRUE(ik::gguf_supported(ik::GgufType::q5_k));
}
#endif

// ── Speed bench: us/expert on the V3.2 gate-projection dims ──────────────────
TEST(GgufGemm, SpeedBench) {
    const int H = 7168, I = 2048;
    const int N = I, K = H;  // gate projection
    std::mt19937 rng(0x5EED);
    std::uniform_real_distribution<float> wd(-0.6f, 0.6f), ad(-1.0f, 1.0f);

    auto bench_type = [&](ik::GgufType t, const char* name) {
        if (!ik::gguf_supported(t)) {
            std::printf("  %-6s: (unsupported on this build)\n", name);
            return;
        }
        std::vector<float> W(static_cast<size_t>(N) * K);
        for (auto& w : W) w = wd(rng);
        PackedW pw = pack_weight(t, W, N, K);
        const void* B = pw.packed.data();
        std::vector<int32_t> offsets{0, 1};  // M=1 token (steady-state decode)
        std::vector<uint16_t> A(K);
        for (int k = 0; k < K; ++k) A[k] = f32_to_bf16(ad(rng));
        std::vector<uint16_t> D(static_cast<size_t>(N), 0);

        auto run = [&] {
            cpu_gguf_grouped_gemm(D.data(), A.data(), t, &B, offsets.data(), 1,
                                  N, K, nullptr);
        };
        run();  // warm
        const int iters = 50;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) run();
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        std::printf("  %-6s: %8.1f us/expert (gate N=%d K=%d, M=1)\n", name, us, N, K);
    };

    std::printf("\n=== GGUF CPU GEMM bench ===\n");
    bench_type(ik::GgufType::q8_0, "Q8_0");
    bench_type(ik::GgufType::q5_0, "Q5_0");
#if defined(LS_IK_HAVE_KQUANTS)
    bench_type(ik::GgufType::q4_k, "Q4_K");
    bench_type(ik::GgufType::q5_k, "Q5_K");
    bench_type(ik::GgufType::q6_k, "Q6_K");
#endif
    SUCCEED();
}
