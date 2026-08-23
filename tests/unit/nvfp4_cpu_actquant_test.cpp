// Stage 0: bit-compatible NVFP4 activation-quantization CPU path.
//
// The GPU CUTLASS routed-expert GEMM consumes NVFP4-quantized activations
// (bf16_to_nvfp4_grouped). The default CPU expert path keeps activations in full
// BF16 (more accurate, but DIFFERENT), so a CPU-computed expert would drift the
// target's MoE hidden vs a GPU-computed one — breaking DSpark lossless verify /
// the TQ golden. cpu_nvfp4_grouped_gemm_actquant closes that gap by quantizing
// activations identically. These tests validate it against independent
// references — CPU-only, no GPU (fp8_e4m3::decode comes from layerstorm_core).

#include "compute/cpu/nvfp4_cpu_kernel.h"
#include "model/quantization/fp8.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace lc = layerstorm::compute::cpu;
using layerstorm::model::fp8_e4m3::decode;

namespace {

float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t f2bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1; u += 0x7fff + lsb; return (uint16_t)(u >> 16);
}

// Independent brute-force round-to-nearest-even float→ue4m3 (search all codes;
// tie → even mantissa). Independent of the kernel's bit-twiddle encoder.
uint8_t ref_float_to_ue4m3(float v) {
    if (!(v > 0.0f)) return 0x00;
    if (v >= 448.0f) return 0x7e;
    int best = -1; float bestd = 1e30f;
    for (int b = 0; b <= 0x7e; ++b) {
        float d = decode((uint8_t)b); if (d < 0) continue;
        float e = std::fabs(d - v);
        if (e < bestd - 1e-20f) { bestd = e; best = b; }
        else if (std::fabs(e - bestd) <= 1e-20f && (b & 1) == 0) best = b;
    }
    return (uint8_t)best;
}
int ref_fp4_nibble(float v) {
    uint8_t sign = (v < 0) ? 8 : 0; float a = std::fabs(v);
    uint8_t idx = (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f) +
                  (a >= 2.5f) + (a >= 3.5f) + (a >= 5.0f);
    return idx | sign;
}
void ref_quant_row(const uint16_t* a, int K, float is_in, float* out) {
    float is = (is_in > 0) ? is_in : 1.0f;
    for (int g = 0; g < K; g += 16) {
        float amax = 0; for (int k = 0; k < 16; ++k) amax = std::max(amax, std::fabs(bf2f(a[g + k])));
        float sr = decode(ref_float_to_ue4m3(amax / (6.0f * is)));
        float denom = sr * is;
        for (int k = 0; k < 16; ++k) {
            float x = bf2f(a[g + k]);
            out[g + k] = denom > 0 ? lc::kE2M1Table[ref_fp4_nibble(x / denom)] * sr * is : 0.0f;
        }
    }
}

// Build a valid nvfp4-sm1xx packed projection [N,K] (random nibbles + E4M3 group
// scales + weight_scale_2), matching the kernel's scale_byte_index de-interleave.
struct Packed { std::vector<uint8_t> buf; lc::PackedProjection proj; std::vector<float> wdeq; };
Packed make_projection(int N, int K, std::mt19937& g, float ws2) {
    const int groups = K / 16;
    const int64_t P = (int64_t)N * K, fp4_bytes = (P + 1) / 2, scale_bytes = (P + 15) / 16;
    int64_t proj_bytes = ((fp4_bytes + scale_bytes + 8) + 127) & ~(int64_t)127;
    Packed pk; pk.buf.assign(proj_bytes, 0); pk.wdeq.assign((size_t)N * K, 0.f);
    uint8_t* base = pk.buf.data(); uint8_t* scl = base + fp4_bytes;
    std::uniform_int_distribution<int> nib(0, 15), e4(0x30, 0x40);
    for (int n = 0; n < N; ++n) for (int gg = 0; gg < groups; ++gg)
        scl[lc::scale_byte_index(n, gg, groups)] = (uint8_t)e4(g);
    for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) {
        int nb = nib(g); int64_t flat = (int64_t)n * K + k; uint8_t& byte = base[flat >> 1];
        if (flat & 1) byte = (byte & 0x0F) | ((nb & 0xF) << 4); else byte = (byte & 0xF0) | (nb & 0xF);
        pk.wdeq[(size_t)n * K + k] =
            lc::kE2M1Table[nb] * decode(scl[lc::scale_byte_index(n, k / 16, groups)]) * ws2;
    }
    std::memcpy(base + proj_bytes - 8, &ws2, 4);
    float is = 1.0f; std::memcpy(base + proj_bytes - 4, &is, 4);
    pk.proj = lc::PackedProjection{base, (size_t)proj_bytes, N, K};
    return pk;
}

}  // namespace

// (T1) The kernel's E4M3 encoder + FP4 quantizer bit-match an independent RNE ref.
TEST(Nvfp4CpuActQuant, ActivationQuantMatchesRneReference) {
    std::mt19937 g(1234);
    const int K = 6144;
    std::vector<uint16_t> a(K); std::normal_distribution<float> nd(0, 2.0f);
    for (auto& x : a) x = f2bf(nd(g));
    for (float is : {1.0f, 0.5f, 2.3f}) {
        std::vector<float> got(K), ref(K);
        lc::nvfp4_quantize_activation_row(a.data(), K, is, got.data());
        ref_quant_row(a.data(), K, is, ref.data());
        for (int k = 0; k < K; ++k)
            ASSERT_FLOAT_EQ(got[k], ref[k]) << "is=" << is << " k=" << k;
    }
}

// (T2/T3) Full grouped GEMM bit-compat vs a double-precision reference of the
// quantized arithmetic; and confirm actquant DIFFERS from the lossy BF16 path.
TEST(Nvfp4CpuActQuant, GroupedGemmMatchesQuantizedReferenceAndDiffersFromLossy) {
    std::mt19937 g(9);
    const int N = 128, K = 64, num_experts = 3;
    int tok[3] = {2, 1, 3};
    std::vector<int32_t> off(num_experts + 1, 0);
    for (int e = 0; e < num_experts; ++e) off[e + 1] = off[e] + tok[e];
    const int total = off[num_experts];

    std::vector<Packed> pk; std::vector<lc::PackedProjection> projs;
    std::uniform_real_distribution<float> wd(0.5f, 1.5f);
    for (int e = 0; e < num_experts; ++e) pk.push_back(make_projection(N, K, g, wd(g)));
    for (int e = 0; e < num_experts; ++e) projs.push_back(pk[e].proj);
    lc::CpuNvfp4ExpertWeights W{projs.data(), num_experts};

    std::vector<uint16_t> A((size_t)total * K); std::normal_distribution<float> nd(0, 1.5f);
    for (auto& x : A) x = f2bf(nd(g));

    std::vector<uint16_t> Dact((size_t)total * N, 0), Dlossy((size_t)total * N, 0);
    lc::cpu_nvfp4_grouped_gemm_actquant(Dact.data(), A.data(), W, nullptr, off.data(), N, K, 2, nullptr);
    lc::cpu_nvfp4_grouped_gemm(Dlossy.data(), A.data(), W, off.data(), N, K, 2, nullptr);

    double maxrel = 0, diff_lossy = 0;
    for (int e = 0; e < num_experts; ++e)
        for (int m = off[e]; m < off[e + 1]; ++m) {
            std::vector<float> adq(K); ref_quant_row(A.data() + (size_t)m * K, K, 1.0f, adq.data());
            for (int n = 0; n < N; ++n) {
                double acc = 0;
                for (int k = 0; k < K; ++k) acc += (double)adq[k] * (double)pk[e].wdeq[(size_t)n * K + k];
                float ref = (float)acc, got = bf2f(Dact[(size_t)m * N + n]);
                maxrel = std::max(maxrel, (double)(std::fabs(got - ref) / std::max(1.0f, std::fabs(ref))));
                diff_lossy += std::fabs(bf2f(Dlossy[(size_t)m * N + n]) - ref);
            }
        }
    EXPECT_LT(maxrel, 0.03) << "actquant GEMM should match quantized reference to BF16 round-off";
    EXPECT_GT(diff_lossy, 0.0) << "lossy BF16-activation path must differ (the gap actquant closes)";
}
