// PART-1 NVFP4 multi-arch (AVX-512 / AVX2 / scalar) accuracy + speed tests.
//
// The production kernel (src/compute/cpu/nvfp4_cpu_kernel.cpp) selects ONE arch
// tier at compile time via #if __AVX512F__ / #elif __AVX2__ / #else. To prove the
// AVX2 tier is correct AND to compare all three on this AVX-512 host, the three
// nvfp4_arch_*.cpp TUs compile the SAME decode/GEMV core (nvfp4_arch_kernel.inc)
// under forced arch flags. This file:
//   * Accuracy: AVX2 == AVX-512 == scalar (and scalar == an independent FP32
//     reference) for decode_full, the M=1 fused GEMV, and the materialize GEMV.
//   * Speed: us/expert for decode + fused GEMV on each tier (H=7168, I=2048).
//
// Scales are NON-UNIFORM E4M3 (varied per group/row) so the Sm1xx de-interleave +
// per-group scale decode are genuinely exercised, not collapsed to 1.0.

#include "nvfp4_arch_probe.h"

#include "model/quantization/fp8.h"
#include "model/weight_loader/nvfp4_sfb_reformat.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

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

constexpr float kMag[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
uint8_t quantize_e2m1(float v) {
    float a = std::fabs(v);
    int best = 0;
    float bd = std::fabs(a - kMag[0]);
    for (int i = 1; i < 8; ++i) {
        float d = std::fabs(a - kMag[i]);
        if (d < bd) { bd = d; best = i; }
    }
    uint8_t sign = (v < 0.0f) ? 1 : 0;
    return static_cast<uint8_t>((sign << 3) | best);
}

// E4M3 encode of a few clean scale magnitudes (decode-exact).
// 0x38=1.0, 0x30=0.5, 0x40=2.0, 0x34=0.75. All decode to exact FP32.
const uint8_t kScaleBytes[4] = {0x38, 0x30, 0x40, 0x34};

int64_t proj_bytes(int N, int K) {
    int64_t params = static_cast<int64_t>(N) * K;
    int64_t wb = (params + 1) / 2;
    // Scale region matches reformat_nvfp4_sfb's PADDED tile layout: rows padded
    // to 128, groups padded to 4 (so unaligned N/G don't overflow the buffer).
    const int64_t G = K / 16;
    const int64_t g_pad = ((G + 3) / 4) * 4;
    const int64_t n_pad = ((static_cast<int64_t>(N) + 127) / 128) * 128;
    int64_t sb = n_pad * g_pad;
    int64_t raw = wb + sb + 2 * static_cast<int64_t>(sizeof(float));
    return (raw + 127) & ~static_cast<int64_t>(127);
}

// Build a packed nvfp4-sm1xx projection from dense FP32 weights [N,K], with a
// caller-supplied per-(row,group) E4M3 scale-byte table (varied scales). The
// effective decoded weight is E2M1(W[n,k]) * decode(scale[n,g]) * ws2.
std::vector<uint8_t> pack_proj(const std::vector<float>& W, int N, int K,
                               float ws2, const std::vector<uint8_t>& scale_ng) {
    const int64_t pb = proj_bytes(N, K);
    std::vector<uint8_t> buf(static_cast<size_t>(pb), 0);
    const int64_t params = static_cast<int64_t>(N) * K;
    for (int64_t i = 0; i < params; ++i) {
        uint8_t nib = quantize_e2m1(W[static_cast<size_t>(i)]);
        int64_t byte = i / 2;
        if (i % 2 == 0) buf[static_cast<size_t>(byte)] |= (nib & 0x0F);
        else            buf[static_cast<size_t>(byte)] |= (nib << 4);
    }
    const int G = K / 16;
    const int64_t wb = (params + 1) / 2;
    // scale_ng is [N, G] row-major; reformat into the Sm1xx interleaved layout.
    layerstorm::model::reformat_nvfp4_sfb(buf.data() + wb, scale_ng.data(), N, G);
    std::memcpy(buf.data() + pb - 8, &ws2, 4);
    float input_scale = 1.0f;
    std::memcpy(buf.data() + pb - 4, &input_scale, 4);
    return buf;
}

// Independent FP32 reference: decode W[n,k] from the dense source + the same
// scale table, accumulate with the BF16-rounded activations.
float ref_dot(const std::vector<float>& W, const std::vector<uint8_t>& scale_ng,
              int N, int K, int n, float ws2, const std::vector<float>& a_f32) {
    (void)N;
    const int G = K / 16;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        uint8_t nib = quantize_e2m1(W[static_cast<size_t>(n) * K + k]);
        float mag = kMag[nib & 0x07];
        float wval = (nib & 0x08) ? -mag : mag;
        float gs = layerstorm::model::fp8_e4m3::decode(
                       scale_ng[static_cast<size_t>(n) * G + (k / 16)]);
        acc += a_f32[k] * (wval * gs * ws2);
    }
    return acc;
}

struct ProjData {
    std::vector<uint8_t> buf;
    std::vector<float> W;
    std::vector<uint8_t> scale_ng;
    int N, K;
    float ws2;
};

ProjData make_proj(int N, int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wd(-5.f, 5.f);
    ProjData p;
    p.N = N; p.K = K; p.ws2 = 0.8125f;
    p.W.resize(static_cast<size_t>(N) * K);
    for (auto& w : p.W) w = wd(rng);
    const int G = K / 16;
    p.scale_ng.resize(static_cast<size_t>(N) * G);
    for (size_t i = 0; i < p.scale_ng.size(); ++i)
        p.scale_ng[i] = kScaleBytes[rng() & 3];
    p.buf = pack_proj(p.W, N, K, p.ws2, p.scale_ng);
    return p;
}

}  // namespace

using nvfp4_arch::ArchKernel;

// ── Accuracy: AVX2 == AVX-512 == scalar (bit-exact within bf16 tol) ──────────
TEST(Nvfp4Arch, DecodeFullAllTiersMatch) {
    const int N = 128, K = 256;  // K=256 => 16 groups; N=128 => one Sm1xx tile
    ProjData p = make_proj(N, K, 0xA11CE);

    const ArchKernel& a512 = nvfp4_arch::avx512_kernel();
    const ArchKernel& a2 = nvfp4_arch::avx2_kernel();
    const ArchKernel& sc = nvfp4_arch::scalar_kernel();

    std::vector<float> d512(static_cast<size_t>(N) * K);
    std::vector<float> d2(d512.size());
    std::vector<float> dsc(d512.size());
    a512.decode_full(p.buf.data(), N, K, p.ws2, d512.data());
    a2.decode_full(p.buf.data(), N, K, p.ws2, d2.data());
    sc.decode_full(p.buf.data(), N, K, p.ws2, dsc.data());

    // All three tiers decode the SAME closed-form value -> BIT-exact equality.
    for (size_t i = 0; i < d512.size(); ++i) {
        ASSERT_EQ(d512[i], dsc[i]) << "avx512 vs scalar @" << i;
        ASSERT_EQ(d2[i], dsc[i]) << "avx2 vs scalar @" << i;
    }
}

TEST(Nvfp4Arch, GemvM1AllTiersMatch) {
    const int N = 132, K = 192;  // N not a mult of 4 (exercises row remainder)
    ProjData p = make_proj(N, K, 0xBEEF1);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ad(-1.5f, 1.5f);
    std::vector<uint16_t> a(K);
    std::vector<float> a_f32(K);
    for (int k = 0; k < K; ++k) {
        a[k] = f32_to_bf16(ad(rng));
        a_f32[k] = bf16_to_f32(a[k]);
    }

    const ArchKernel& a512 = nvfp4_arch::avx512_kernel();
    const ArchKernel& a2 = nvfp4_arch::avx2_kernel();
    const ArchKernel& sc = nvfp4_arch::scalar_kernel();

    std::vector<float> o512(N), o2(N), osc(N), omat(N);
    std::vector<float> scratch(K);
    a512.gemv_m1_fused(p.buf.data(), N, K, p.ws2, a.data(), o512.data());
    a2.gemv_m1_fused(p.buf.data(), N, K, p.ws2, a.data(), o2.data());
    sc.gemv_m1_fused(p.buf.data(), N, K, p.ws2, a.data(), osc.data());
    // Materialize path (AVX-512 tier) — must agree with fused to bf16 tol.
    a512.gemv_m1_materialize(p.buf.data(), N, K, p.ws2, a.data(), omat.data(),
                             scratch.data());

    for (int n = 0; n < N; ++n) {
        float ref = ref_dot(p.W, p.scale_ng, N, K, n, p.ws2, a_f32);
        // Cross-tier: different reduction orders -> allow a tiny FP tol.
        EXPECT_NEAR(o512[n], osc[n], std::fabs(osc[n]) * 1e-4f + 1e-4f)
            << "avx512 vs scalar n=" << n;
        EXPECT_NEAR(o2[n], osc[n], std::fabs(osc[n]) * 1e-4f + 1e-4f)
            << "avx2 vs scalar n=" << n;
        EXPECT_NEAR(o512[n], omat[n], std::fabs(omat[n]) * 1e-4f + 1e-4f)
            << "fused vs materialize n=" << n;
        // vs independent FP32 reference (FP accumulation order differs).
        EXPECT_NEAR(osc[n], ref, std::fabs(ref) * 1e-3f + 1e-3f)
            << "scalar vs ref n=" << n;
    }
}

// ── Speed bench: us/expert for decode + fused GEMV on each tier ──────────────
TEST(Nvfp4Arch, SpeedBench) {
    // DeepSeek V3.2 routed-expert projection dims.
    const int H = 7168, I = 2048;
    // gate/up: N=I, K=H. down: N=H, K=I. Bench the gate projection (largest K).
    const int N = I, K = H;
    ProjData p = make_proj(N, K, 0x5EED);

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> ad(-1.0f, 1.0f);
    std::vector<uint16_t> a(K);
    for (int k = 0; k < K; ++k) a[k] = f32_to_bf16(ad(rng));

    std::vector<float> decoded(static_cast<size_t>(N) * K);
    std::vector<float> out(N);

    struct Tier { const ArchKernel& k; };
    const ArchKernel* tiers[3] = {&nvfp4_arch::avx512_kernel(),
                                  &nvfp4_arch::avx2_kernel(),
                                  &nvfp4_arch::scalar_kernel()};

    auto bench = [&](auto fn, int iters) {
        // warm
        fn();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) fn();
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    };

    std::printf("\n=== NVFP4 CPU arch bench (gate proj N=%d K=%d, us/expert) ===\n",
                N, K);
    std::printf("%-8s  %14s  %16s\n", "tier", "decode_full[us]",
                "gemv_m1_fused[us]");
    for (const ArchKernel* t : tiers) {
        // scalar decode of a full 2048x7168 expert is slow; fewer iters.
        int dec_iters = (std::string(t->name) == "scalar") ? 3 : 10;
        int gemv_iters = (std::string(t->name) == "scalar") ? 30 : 200;
        double dec_us = bench([&] {
            t->decode_full(p.buf.data(), N, K, p.ws2, decoded.data());
        }, dec_iters);
        double gemv_us = bench([&] {
            t->gemv_m1_fused(p.buf.data(), N, K, p.ws2, a.data(), out.data());
        }, gemv_iters);
        std::printf("%-8s  %14.1f  %16.1f\n", t->name, dec_us, gemv_us);
    }
    SUCCEED();
}
