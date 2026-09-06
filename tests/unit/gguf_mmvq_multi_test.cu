// P-29 step 15 — multi-segment mmvq (LS_KDA_PROJ_FUSED) bitwise oracle.
//
// Contract under test (gguf_mmvq.h launch_gguf_mmvq_multi): for every
// segment s, the fused launch writes EXACTLY the bytes the standalone
// gguf_mmvq call writes for (M, N[s], K, A, B[s], C[s], type) — the fused
// kernel runs the identical per-CTA accumulate/reduce; only the
// blockIdx->(segment, channel) mapping precedes it, and the shared
// activation is quantized once (deterministic quantizer, same bytes as each
// per-call quantize of the same A).
//
// Shapes mirror the production KDA ladder (q/k/v = C, b = Hk, f_a/g_a = 128,
// all K = H) plus deliberately awkward segment sizes (odd N, N=1) to catch
// mapping bugs. Negative control: perturbing one weight byte of one segment
// must change that segment's output and no other's.

#include "compute/cuda_sm120_device_backend.h"
#include "core/attention_device.h"
#include "model/quantization/gguf_kquant.h"
#include "compute/cpu/ik_vendor/ik_gguf_gemm.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace lc = layerstorm::compute;
namespace lm = layerstorm::model;
namespace cfg = layerstorm::config;
namespace ik = layerstorm::compute::cpu::ik;

namespace {

cfg::GpuRef make_gpu() { return {0, 0, cfg::GpuType::rtx5090}; }

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T),
                         cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}

std::vector<uint8_t> pack_weight(ik::GgufType t, int N, int K,
                                 std::mt19937& rng) {
    std::uniform_real_distribution<float> wd(-0.6f, 0.6f);
    const size_t row_bytes = ik::weight_row_bytes(t, K);
    std::vector<uint8_t> bytes(static_cast<size_t>(N) * row_bytes);
    std::vector<float> wr(K);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) wr[k] = wd(rng);
        ik::quantize_weight(t, wr.data(),
                            bytes.data() + static_cast<size_t>(n) * row_bytes,
                            K);
    }
    return bytes;
}

struct TypeTags {
    ik::GgufType ik_t;
    lm::GgufKQuantType mod_t;
};

void run_case(TypeTags t, int M, int K, const std::vector<int>& Ns,
              uint32_t seed, bool negative_control) {
    REQUIRES_GPU();
    if (!ik::gguf_supported(t.ik_t))
        GTEST_SKIP() << "ik packer type unsupported in build";
    std::mt19937 rng(seed);

    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();

    const int nseg = static_cast<int>(Ns.size());
    ASSERT_LE(nseg, lc::GgufGemmMultiParams::kMaxSegs);

    std::uniform_real_distribution<float> ad(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> act(static_cast<size_t>(M) * K);
    for (auto& v : act) v = __float2bfloat16(ad(rng));
    __nv_bfloat16* d_a = upload(act);

    std::vector<uint8_t*> d_w(nseg);
    std::vector<__nv_bfloat16*> d_c_ref(nseg), d_c_fused(nseg);
    std::vector<std::vector<uint8_t>> w_host(nseg);
    for (int s = 0; s < nseg; ++s) {
        w_host[s] = pack_weight(t.ik_t, Ns[s], K, rng);
        if (negative_control && s == 1) {
            // Flip one packed-weight byte in segment 1 (fused arm only —
            // uploaded below AFTER the reference run reads the clean copy).
        }
        d_w[s] = upload(w_host[s]);
        const size_t out_n = static_cast<size_t>(M) * Ns[s];
        std::vector<__nv_bfloat16> zero(out_n, __float2bfloat16(0.0f));
        d_c_ref[s] = upload(zero);
        d_c_fused[s] = upload(zero);
    }

    void* d_ws = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ws, static_cast<size_t>(M) * (K / 32) * 36),
              cudaSuccess);

    // Reference: the serial ladder (one gguf_mmvq per segment).
    for (int s = 0; s < nseg; ++s) {
        lc::GgufGemmParams p{};
        p.M = M; p.N = Ns[s]; p.K = K;
        p.A = d_a; p.B = d_w[s]; p.C = d_c_ref[s];
        p.type = t.mod_t;
        dev.gguf_mmvq(p, d_ws, nullptr);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Negative control: perturb segment 1's weight AFTER the reference ran.
    if (negative_control) {
        std::vector<uint8_t> w1 = w_host[1];
        w1[w1.size() / 2] ^= 0x5A;
        ASSERT_EQ(cudaMemcpy(d_w[1], w1.data(), w1.size(),
                             cudaMemcpyHostToDevice), cudaSuccess);
    }

    // Fused: one quantize + one multi-segment launch.
    lc::GgufGemmMultiParams mp{};
    mp.M = M; mp.K = K;
    mp.A = d_a;
    mp.type = t.mod_t;
    mp.nseg = nseg;
    for (int s = 0; s < nseg; ++s) {
        mp.B[s] = d_w[s];
        mp.C[s] = d_c_fused[s];
        mp.N[s] = Ns[s];
    }
    dev.gguf_mmvq_multi(mp, d_ws, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (int s = 0; s < nseg; ++s) {
        const size_t out_bytes =
            static_cast<size_t>(M) * Ns[s] * sizeof(__nv_bfloat16);
        std::vector<uint8_t> ref(out_bytes), fused(out_bytes);
        ASSERT_EQ(cudaMemcpy(ref.data(), d_c_ref[s], out_bytes,
                             cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(fused.data(), d_c_fused[s], out_bytes,
                             cudaMemcpyDeviceToHost), cudaSuccess);
        if (negative_control && s == 1) {
            EXPECT_NE(0, std::memcmp(ref.data(), fused.data(), out_bytes))
                << "negative control: perturbed segment 1 must diverge";
        } else {
            EXPECT_EQ(0, std::memcmp(ref.data(), fused.data(), out_bytes))
                << "segment " << s << " (N=" << Ns[s]
                << ") not byte-identical to the serial ladder";
        }
    }

    for (int s = 0; s < nseg; ++s) {
        cudaFree(d_w[s]); cudaFree(d_c_ref[s]); cudaFree(d_c_fused[s]);
    }
    cudaFree(d_a); cudaFree(d_ws);
}

constexpr TypeTags kQ8_0{ik::GgufType::q8_0, lm::GgufKQuantType::Q8_0};
constexpr TypeTags kQ4_K{ik::GgufType::q4_k, lm::GgufKQuantType::Q4_K};

}  // namespace

// The production KDA-ladder shape class (scaled): 3 fat + 1 tiny + 2 small,
// M=1 decode fast path, Q8_0 (the glm5_next projection type).
TEST(GgufMmvqMulti, KdaLadderShapeM1Q8_0) {
    run_case(kQ8_0, 1, 512, {96, 96, 96, 32, 128, 128}, 0xA1, false);
}

// Multi-row register-tile path (M>1) + awkward segment sizes incl. N=1.
TEST(GgufMmvqMulti, MultiRowAwkwardSegsQ8_0) {
    run_case(kQ8_0, 3, 256, {33, 1, 7, 128}, 0xB2, false);
}

// A k-quant type (different policy P) through the same fused mapping.
TEST(GgufMmvqMulti, KdaLadderShapeM1Q4_K) {
    run_case(kQ4_K, 1, 512, {64, 64, 32, 128}, 0xC3, false);
}

// Single segment degenerates to exactly the standalone launch.
TEST(GgufMmvqMulti, SingleSegmentM1Q8_0) {
    run_case(kQ8_0, 1, 256, {96}, 0xD4, false);
}

// Negative control: a flipped weight byte in one segment diverges that
// segment and ONLY that segment.
TEST(GgufMmvqMulti, NegativeControlPerturbedWeight) {
    run_case(kQ8_0, 1, 512, {96, 96, 32, 128}, 0xE5, true);
}
