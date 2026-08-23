// PART-4 head-to-head: NVFP4 (CPU AVX-512 / AVX2) vs ik GGUF Q8_0 / Q5_0 / Q4_K / Q5_K / Q6_K
// on the SAME expert projection dims (H=7168, I=2048), measuring BOTH:
//   * ACCURACY: RMS / max relative error of the M=1 GEMV output vs an FP32
//     reference computed from the ORIGINAL (un-quantized) weights — so each
//     scheme's weight-quantization error is included (the real quality metric).
//   * SPEED: us/expert for the M=1 GEMV (steady-state CPU decode case).
//
// This is the head-to-head the C-6 comparison wants: NVFP4-on-CPU vs ik's
// Q4_K/Q8_0/Q6_K-on-CPU. The NVFP4 tiers come from the multi-arch probe
// (nvfp4_arch_*.cpp); the GGUF tiers from the vendored ik kernels via the shared
// cpu_gguf_grouped_gemm driver. Output is a printed table (the report artifact).

#include "nvfp4_arch_probe.h"

#include "compute/cpu/cpu_gguf_gemm.h"
#include "compute/cpu/ik_vendor/ik_gguf_gemm.h"
#include "model/quantization/fp8.h"
#include "model/weight_loader/nvfp4_sfb_reformat.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

using namespace layerstorm::compute::cpu;

namespace {

uint16_t f32_to_bf16(float f) {
    uint32_t bits; std::memcpy(&bits, &f, 4);
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}
float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16; float f;
    std::memcpy(&f, &bits, 4); return f;
}

// ── NVFP4 packing (uniform E4M3 scale = 1.0; weight_scale_2 absorbs range) ──
constexpr float kMag[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
uint8_t quantize_e2m1(float v) {
    float a = std::fabs(v); int best = 0; float bd = std::fabs(a - kMag[0]);
    for (int i = 1; i < 8; ++i) { float d = std::fabs(a - kMag[i]); if (d < bd) { bd = d; best = i; } }
    return static_cast<uint8_t>(((v < 0 ? 1 : 0) << 3) | best);
}
constexpr uint8_t kE4M3_One = 0x38;
int64_t nvfp4_proj_bytes(int N, int K) {
    int64_t params = int64_t(N) * K, wb = (params + 1) / 2;
    int64_t G = K / 16, g_pad = ((G + 3) / 4) * 4, n_pad = ((int64_t(N) + 127) / 128) * 128;
    int64_t raw = wb + n_pad * g_pad + 2 * int64_t(sizeof(float));
    return (raw + 127) & ~int64_t(127);
}
// Pack with a per-row weight_scale_2 folded into a global ws2 (uniform group
// scale) — the CPU NVFP4 path dequant is E2M1(round(W/ws2)) * 1.0 * ws2.
std::vector<uint8_t> pack_nvfp4(const std::vector<float>& W, int N, int K, float ws2) {
    int64_t pb = nvfp4_proj_bytes(N, K);
    std::vector<uint8_t> buf(size_t(pb), 0);
    int64_t params = int64_t(N) * K;
    for (int64_t i = 0; i < params; ++i) {
        uint8_t nib = quantize_e2m1(W[size_t(i)] / ws2);
        int64_t byte = i / 2;
        if (i % 2 == 0) buf[size_t(byte)] |= (nib & 0x0F); else buf[size_t(byte)] |= (nib << 4);
    }
    int G = K / 16;
    std::vector<uint8_t> raw(size_t(N) * G, kE4M3_One);
    layerstorm::model::reformat_nvfp4_sfb(buf.data() + (params + 1) / 2, raw.data(), N, G);
    std::memcpy(buf.data() + pb - 8, &ws2, 4);
    float in_s = 1.0f; std::memcpy(buf.data() + pb - 4, &in_s, 4);
    return buf;
}

struct Stats { double rms_rel; double max_rel; };
Stats error_stats(const std::vector<float>& got, const std::vector<float>& ref) {
    double se = 0, sref = 0, maxr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double e = got[i] - ref[i];
        se += e * e; sref += double(ref[i]) * ref[i];
        double denom = std::max(1e-3, std::fabs(double(ref[i])));
        maxr = std::max(maxr, std::fabs(e) / denom);
    }
    return {std::sqrt(se / std::max(1e-30, sref)), maxr};
}

double bench(const std::function<void()>& fn, int iters) {
    fn();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

}  // namespace

TEST(QuantComparison, Nvfp4VsGgufHeadToHead) {
    const int H = 7168, I = 2048;
    const int N = I, K = H;  // gate projection (largest K)
    std::mt19937 rng(0x1234ABCD);
    std::uniform_real_distribution<float> wd(-0.5f, 0.5f), ad(-1.0f, 1.0f);

    // Shared original FP32 weights + activations (bf16-rounded).
    std::vector<float> W(size_t(N) * K);
    for (auto& w : W) w = wd(rng);
    std::vector<uint16_t> A(K);
    std::vector<float> Af(K);
    for (int k = 0; k < K; ++k) { A[k] = f32_to_bf16(ad(rng)); Af[k] = bf16_to_f32(A[k]); }

    // FP32 reference from the ORIGINAL weights (the ground truth quality target).
    std::vector<float> ref(N, 0.0f);
    for (int n = 0; n < N; ++n) {
        double acc = 0;
        for (int k = 0; k < K; ++k) acc += double(Af[k]) * double(W[size_t(n) * K + k]);
        ref[n] = float(acc);
    }

    struct Row { const char* name; Stats st; double us; };
    std::vector<Row> rows;

    // ── NVFP4 (AVX-512 + AVX2 tiers) ────────────────────────────────────────
    {
        const float ws2 = 6.0f / 0.5f * 0.5f;  // map |W|<=0.5 into E2M1 range via ws2
        // Choose ws2 so max|W| (~0.5) maps near the top E2M1 codepoint (6) -> ws2~0.0833.
        float amax = 0; for (auto w : W) amax = std::max(amax, std::fabs(w));
        const float ws2b = amax / 6.0f;
        (void)ws2;
        auto buf = pack_nvfp4(W, N, K, ws2b);
        for (const nvfp4_arch::ArchKernel* t :
             {&nvfp4_arch::avx512_kernel(), &nvfp4_arch::avx2_kernel()}) {
            std::vector<float> out(N);
            t->gemv_m1_fused(buf.data(), N, K, ws2b, A.data(), out.data());
            Stats st = error_stats(out, ref);
            double us = bench([&] { t->gemv_m1_fused(buf.data(), N, K, ws2b, A.data(), out.data()); }, 50);
            char nm[32]; std::snprintf(nm, sizeof nm, "NVFP4-%s", t->name);
            rows.push_back({strdup(nm), st, us});
        }
    }

    // ── GGUF Q8_0 / Q4_K / Q6_K ─────────────────────────────────────────────
    auto run_gguf = [&](ik::GgufType t, const char* name) {
        if (!ik::gguf_supported(t)) return;
        size_t wrb = ik::weight_row_bytes(t, K);
        std::vector<uint8_t> packed(size_t(N) * wrb);
        for (int n = 0; n < N; ++n)
            ik::quantize_weight(t, W.data() + size_t(n) * K,
                                packed.data() + size_t(n) * wrb, K);
        const void* B = packed.data();
        std::vector<int32_t> off{0, 1};
        std::vector<uint16_t> D(N, 0);
        cpu_gguf_grouped_gemm(D.data(), A.data(), t, &B, off.data(), 1, N, K, nullptr);
        std::vector<float> out(N);
        for (int n = 0; n < N; ++n) out[n] = bf16_to_f32(D[n]);
        Stats st = error_stats(out, ref);
        double us = bench([&] {
            cpu_gguf_grouped_gemm(D.data(), A.data(), t, &B, off.data(), 1, N, K, nullptr);
        }, 50);
        rows.push_back({name, st, us});
    };
    run_gguf(ik::GgufType::q8_0, "GGUF-Q8_0");
    run_gguf(ik::GgufType::q5_0, "GGUF-Q5_0");
    run_gguf(ik::GgufType::q4_k, "GGUF-Q4_K");
    run_gguf(ik::GgufType::q5_k, "GGUF-Q5_K");
    run_gguf(ik::GgufType::q6_k, "GGUF-Q6_K");

    std::printf("\n==== CPU quant head-to-head (gate N=%d K=%d, M=1) ====\n", N, K);
    std::printf("%-12s  %12s  %12s  %14s\n", "scheme", "rms_rel_err", "max_rel_err",
                "us/expert");
    for (const auto& r : rows)
        std::printf("%-12s  %12.5f  %12.4f  %14.1f\n", r.name, r.st.rms_rel,
                    r.st.max_rel, r.us);
    std::printf("(rms_rel/max_rel vs FP32 of ORIGINAL weights; bf16 acts)\n");

    // Sanity: every scheme should be in a reasonable accuracy band.
    for (const auto& r : rows)
        EXPECT_LT(r.st.rms_rel, 0.30) << r.name << " rms_rel too high";
    SUCCEED();
}
