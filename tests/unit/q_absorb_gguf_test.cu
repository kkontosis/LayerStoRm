// GG-7: GPU numerical test of the absorbed-MLA q_absorb GGUF W_UK branch.
//
// MLA query absorption folds W_UK (the K-half of kv_b_proj) into the query
// inside the q_absorb kernel (NOT a plain GEMM — GG-4 deliberately excludes
// kv_b). GG-7 wires q_absorb to accept a GGUF-packed kv_b via the kernel's
// in-kernel per-element dequant branch (dequant-only: the packed weight is
// dequanted per element × the BF16 query, with NO activation quant / int path,
// so it is accuracy-equal to a load-time dequant to BF16).
//
// This test drives the REAL CudaSm120DeviceBackend::absorb_q on hardware with a
// GGUF-packed kv_b (Q4_K, Q6_K, Q8_0) and a random BF16 query, then compares
// q_absorbed against the SAME absorption run with the kv_b dequanted to BF16
// (the kernel's BF16 path, gguf_type == -1). Because the GGUF path is
// dequant-only, the two must agree to cosine >= 0.99999 (the only difference is
// the BF16 rounding of each weight element, not any activation-quant error). A
// FP32 CPU reference over the same dequant guards against a trivially-broken
// kernel, and the rope tail (apply_rope off) must be bit-identical between runs.
//
// The L%QK guard (kv_lora_rank % QK == 0) is covered CPU-side in
// dcp_executor_test.cpp (GgufKvBAbsorbGuard*). Here L=512 satisfies QK=256 (k-
// quants) and QK=32 (Q8_0).
//
// GPU-required (REQUIRES_GPU); SKIPPED on headless CI.

#include "compute/cuda_sm120_device_backend.h"
#include "core/attention_device.h"
#include "model/quantization/gguf_kquant.h"

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
namespace lc = layerstorm::compute;
namespace lm = layerstorm::model;
namespace cfg = layerstorm::config;

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

// The two parallel type tags for one GGUF family: the CPU ik packer tag and the
// model GgufKQuantType (its int ordinal is what QAbsorbParams.gguf_type carries
// across to the kernel — 0=Q2_K … 5=Q8_0).
struct TypeTags {
    ik::GgufType ik_t;
    lm::GgufKQuantType mod_t;
};
constexpr TypeTags kQ4_K{ik::GgufType::q4_k, lm::GgufKQuantType::Q4_K};
constexpr TypeTags kQ6_K{ik::GgufType::q6_k, lm::GgufKQuantType::Q6_K};
constexpr TypeTags kQ8_0{ik::GgufType::q8_0, lm::GgufKQuantType::Q8_0};

cfg::GpuRef make_gpu() { return {0, 0, cfg::GpuType::rtx5090}; }

// Pack `N` random weight rows of `K` cols at `t` into a host byte buffer (ik
// reference packer = byte-identical to the ggml block layout the GPU kernel
// consumes) and return the parallel FP32 dequant of the SAME bytes.
struct PackedWeight {
    std::vector<uint8_t> bytes;
    std::vector<float> dq;   // [N, K] FP32
    size_t row_bytes;
};
PackedWeight pack_weight(ik::GgufType t, int N, int K, std::mt19937& rng) {
    std::uniform_real_distribution<float> wd(-0.6f, 0.6f);
    PackedWeight p;
    p.row_bytes = ik::weight_row_bytes(t, K);
    p.bytes.resize(static_cast<size_t>(N) * p.row_bytes);
    p.dq.resize(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n) {
        std::vector<float> wr(K);
        for (int k = 0; k < K; ++k) wr[k] = wd(rng);
        void* dst = p.bytes.data() + static_cast<size_t>(n) * p.row_bytes;
        ik::quantize_weight(t, wr.data(), dst, K);
        ik::dequantize_weight(t, dst, p.dq.data() + static_cast<size_t>(n) * K, K);
    }
    return p;
}

// One absorb_q run with a given (weight ptr, gguf_type). Returns the host BF16
// q_absorbed [s_q, h_q, L+R].
std::vector<__nv_bfloat16> run_absorb(lc::CudaSm120DeviceBackend& dev,
                                      const void* d_w, int gguf_type,
                                      const __nv_bfloat16* d_q,
                                      int s_q, int h_q, int P, int L, int R, int V) {
    const size_t out_n = static_cast<size_t>(s_q) * h_q * (L + R);
    std::vector<__nv_bfloat16> zero(out_n, f2bf(0.0f));
    __nv_bfloat16* d_out = upload(zero);

    lc::QAbsorbParams qa{};
    qa.q_heads         = d_q;
    qa.kv_b_proj       = d_w;
    qa.kv_b_proj_scales = nullptr;   // GGUF scales are in-block; BF16 path has none
    qa.q_absorbed      = d_out;
    qa.s_q             = s_q;
    qa.h_q             = h_q;
    qa.d_nope_in       = P;
    qa.d_c             = L;
    qa.d_rope          = R;
    qa.d_v             = V;
    qa.weight_is_fp8   = false;
    qa.gguf_type       = gguf_type;  // -1 = BF16 path
    qa.apply_rope      = false;

    dev.absorb_q(qa, nullptr);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> out(out_n);
    cudaMemcpy(out.data(), d_out, out_n * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return out;
}

// Extract the absorbed L columns (drop the R rope tail) into a flat FP32 vector.
std::vector<float> absorbed(const std::vector<__nv_bfloat16>& full,
                            int s_q, int h_q, int L, int R) {
    std::vector<float> out(static_cast<size_t>(s_q) * h_q * L);
    const int row = L + R;
    for (int sh = 0; sh < s_q * h_q; ++sh)
        for (int k = 0; k < L; ++k)
            out[static_cast<size_t>(sh) * L + k] =
                bf2f(full[static_cast<size_t>(sh) * row + k]);
    return out;
}

void run_case(TypeTags t, int s_q, int h_q, int P, int L, int R, int V,
              double min_cos_parity, uint32_t seed) {
    if (!ik::gguf_supported(t.ik_t)) GTEST_SKIP() << "ik type unsupported in build";
    std::mt19937 rng(seed);

    const int kv_row = P + V;            // kv_b_proj per-head row count
    const int N = h_q * kv_row;          // kv_b_proj rows = h_q*(P+V)

    // kv_b_proj packed as [N, L] (each row L cols → row-major GGUF blocks).
    PackedWeight w = pack_weight(t.ik_t, N, L, rng);

    // Random BF16 query [s_q, h_q, P+R]; keep the bf-rounded FP32 mirror.
    const int row_in = P + R;
    std::uniform_real_distribution<float> qd(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> q_bf(static_cast<size_t>(s_q) * h_q * row_in);
    std::vector<float> q_f(q_bf.size());
    for (size_t i = 0; i < q_bf.size(); ++i) {
        q_bf[i] = f2bf(qd(rng));
        q_f[i] = bf2f(q_bf[i]);
    }

    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();

    __nv_bfloat16* d_q = upload(q_bf);

    // GGUF weight (raw packed bytes).
    std::byte* d_w_gguf = nullptr;
    cudaMalloc(&d_w_gguf, w.bytes.size());
    cudaMemcpy(d_w_gguf, w.bytes.data(), w.bytes.size(), cudaMemcpyHostToDevice);

    // Same weight dequanted to BF16 [N, L] (the kernel's BF16 path input).
    std::vector<__nv_bfloat16> w_bf(w.dq.size());
    for (size_t i = 0; i < w_bf.size(); ++i) w_bf[i] = f2bf(w.dq[i]);
    __nv_bfloat16* d_w_bf = upload(w_bf);

    auto out_gguf = run_absorb(dev, d_w_gguf, static_cast<int>(t.mod_t),
                               d_q, s_q, h_q, P, L, R, V);
    auto out_bf16 = run_absorb(dev, d_w_bf, /*gguf_type=*/-1,
                               d_q, s_q, h_q, P, L, R, V);

    // ── FP32 CPU reference of the absorbed content (uses the FP32 dequant) ──
    // ref[s,h,k] = sum_d q_nope[s,h,d] * W_UK[h,d,k],
    // W_UK[h,d,k] = kv_b[(h*(P+V)+d)*L + k].
    std::vector<float> ref(static_cast<size_t>(s_q) * h_q * L, 0.0f);
    for (int s = 0; s < s_q; ++s)
        for (int h = 0; h < h_q; ++h) {
            const float* qn = q_f.data() + (static_cast<size_t>(s) * h_q + h) * row_in;
            for (int k = 0; k < L; ++k) {
                double acc = 0;
                for (int d = 0; d < P; ++d)
                    acc += double(qn[d]) *
                           w.dq[(static_cast<size_t>(h) * kv_row + d) * L + k];
                ref[(static_cast<size_t>(s) * h_q + h) * L + k] = float(acc);
            }
        }

    std::vector<float> abs_gguf = absorbed(out_gguf, s_q, h_q, L, R);
    std::vector<float> abs_bf16 = absorbed(out_bf16, s_q, h_q, L, R);

    const double cos_parity = cosine(abs_gguf, abs_bf16);
    const double cos_gguf_ref = cosine(abs_gguf, ref);
    const double cos_bf16_ref = cosine(abs_bf16, ref);

    // Primary GG-7 assertion: dequant-only GGUF path matches the BF16 path.
    EXPECT_GE(cos_parity, min_cos_parity)
        << "GGUF-vs-BF16 q_absorb parity cos=" << cos_parity;
    // Sanity: both paths track the FP32 reference (catches a dead kernel).
    EXPECT_GE(cos_gguf_ref, 0.9999) << "GGUF vs FP32 ref cos=" << cos_gguf_ref;
    EXPECT_GE(cos_bf16_ref, 0.999) << "BF16 vs FP32 ref cos=" << cos_bf16_ref;

    // The rope tail is a plain copy (apply_rope off) of the SAME query in both
    // runs → must be bit-identical.
    const int row = L + R;
    for (int sh = 0; sh < s_q * h_q; ++sh)
        for (int r = 0; r < R; ++r) {
            const size_t idx = static_cast<size_t>(sh) * row + L + r;
            ASSERT_EQ(bf2f(out_gguf[idx]), bf2f(out_bf16[idx]))
                << "rope tail mismatch at sh=" << sh << " r=" << r;
        }

    cudaFree(d_q);
    cudaFree(d_w_gguf);
    cudaFree(d_w_bf);
}

}  // namespace

// DeepSeek-V3.2-like absorbed-MLA shapes: P=128, L=512, R=64, V=128.
// L=512 satisfies QK=256 (k-quants) and QK=32 (Q8_0).
TEST(QAbsorbGgufGpu, Q4K_DequantOnlyMatchesBf16) {
    REQUIRES_GPU();
    run_case(kQ4_K, /*s_q=*/3, /*h_q=*/4, /*P=*/128, /*L=*/512, /*R=*/64, /*V=*/128,
             /*min_cos_parity=*/0.99999, 0x7A01);
}
TEST(QAbsorbGgufGpu, Q6K_DequantOnlyMatchesBf16) {
    REQUIRES_GPU();
    run_case(kQ6_K, /*s_q=*/3, /*h_q=*/4, /*P=*/128, /*L=*/512, /*R=*/64, /*V=*/128,
             /*min_cos_parity=*/0.99999, 0x7A02);
}
TEST(QAbsorbGgufGpu, Q8_0_DequantOnlyMatchesBf16) {
    REQUIRES_GPU();
    // QK=32 family — exercises the blocks_per_row = L/32 = 16 path.
    run_case(kQ8_0, /*s_q=*/2, /*h_q=*/3, /*P=*/128, /*L=*/512, /*R=*/64, /*V=*/128,
             /*min_cos_parity=*/0.99999, 0x7A03);
}
