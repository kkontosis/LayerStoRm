// TD-GLM5-TP-COMBINE-PRECISION: fp32-out (f32c) GGUF GEMM variants — the
// bitwise oracle for the TP fp32 partial combine.
//
// Contract under test (gguf_gemm_f32c.h): each f32c variant runs the IDENTICAL
// reduction structure, tile configuration and launch heuristics as its
// bf16-out sibling; only the epilogue store differs (raw fp32 vs
// __float2bfloat16_rn). Therefore for identical inputs:
//     __float2bfloat16_rn(f32c_C[i])  ==  bf16_sibling_C[i]   (BITWISE)
// on every element, for every route (mmvq decode, mmq_mma prefill, dequant
// fallback) and every M-shape crossing the engine's route boundaries.
//
// Types covered: Q4_K / Q5_K / Q6_K / Q8_0 — the ik reference packer's
// coverage (Q8_0 is the glm5_next projection type, the production shape).
// Q2_K/Q3_K/MXFP4 share the same templated epilogue by construction (their
// f32c instance TUs differ from these only in the integer policy P).
//
// Also tests launch_cast_f32_to_bf16 — the single post-allreduce rounding.

#include "compute/cuda_sm120_device_backend.h"
#include "core/attention_device.h"
#include "model/quantization/gguf_kquant.h"
#include "compute/cpu/ik_vendor/ik_gguf_gemm.h"
#include "compute/kernels/sm120/gemm/cast_f32_bf16.h"

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

struct TypeTags {
    ik::GgufType ik_t;
    lm::GgufKQuantType mod_t;
};

// Pack N random weight rows of K cols (ik reference packer = byte-identical
// ggml block layout).
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

enum class Route { kMmvq, kMmq, kDequant };

void run_case(TypeTags t, Route route, int M, int N, int K, uint32_t seed) {
    REQUIRES_GPU();
    if (!ik::gguf_supported(t.ik_t))
        GTEST_SKIP() << "ik packer type unsupported in build";
    std::mt19937 rng(seed);

    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();

    // Random BF16 activation [M, K].
    std::uniform_real_distribution<float> ad(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> act(static_cast<size_t>(M) * K);
    for (auto& v : act) v = __float2bfloat16(ad(rng));
    __nv_bfloat16* d_a = upload(act);

    std::vector<uint8_t> w = pack_weight(t.ik_t, N, K, rng);
    uint8_t* d_w = upload(w);

    const size_t out_n = static_cast<size_t>(M) * N;
    std::vector<__nv_bfloat16> zero_bf(out_n, __float2bfloat16(0.0f));
    std::vector<float> zero_f(out_n, 0.0f);
    __nv_bfloat16* d_c_bf = upload(zero_bf);
    float* d_c_f32 = upload(zero_f);

    // Q8_1 activation workspace (mmvq/mmq int routes): M * (K/32) blocks of
    // 36 B (block_q8_1 = 32 int8 qs + half2 ds).
    void* d_ws = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ws, static_cast<size_t>(M) * (K / 32) * 36),
              cudaSuccess);

    auto run = [&](void* c, bool c_fp32) {
        lc::GgufGemmParams p{};
        p.M = M; p.N = N; p.K = K;
        p.A = d_a; p.B = d_w; p.C = c;
        p.type = t.mod_t;
        p.c_fp32 = c_fp32;
        switch (route) {
            case Route::kMmvq:    dev.gguf_mmvq(p, d_ws, nullptr); break;
            case Route::kMmq:     dev.gguf_mmq(p, d_ws, nullptr); break;
            case Route::kDequant: dev.gguf_dequant_gemm(p, nullptr); break;
        }
    };
    run(d_c_bf, false);
    run(d_c_f32, true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> c_bf(out_n);
    std::vector<float> c_f32(out_n);
    ASSERT_EQ(cudaMemcpy(c_bf.data(), d_c_bf, out_n * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(c_f32.data(), d_c_f32, out_n * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    size_t mismatches = 0;
    for (size_t i = 0; i < out_n; ++i) {
        const __nv_bfloat16 rounded = __float2bfloat16_rn(c_f32[i]);
        uint16_t a, b;
        std::memcpy(&a, &rounded, 2);
        std::memcpy(&b, &c_bf[i], 2);
        if (a != b && ++mismatches <= 4)
            ADD_FAILURE() << "elem " << i << ": bf16-out 0x" << std::hex << b
                          << " != round(f32c) 0x" << a << std::dec
                          << " (f32c=" << c_f32[i] << ")";
    }
    EXPECT_EQ(mismatches, 0u) << mismatches << " of " << out_n
                              << " mismatched";

    cudaFree(d_a); cudaFree(d_w); cudaFree(d_c_bf); cudaFree(d_c_f32);
    cudaFree(d_ws);
}

constexpr TypeTags kQ4_K{ik::GgufType::q4_k, lm::GgufKQuantType::Q4_K};
constexpr TypeTags kQ5_K{ik::GgufType::q5_k, lm::GgufKQuantType::Q5_K};
constexpr TypeTags kQ6_K{ik::GgufType::q6_k, lm::GgufKQuantType::Q6_K};
constexpr TypeTags kQ8_0{ik::GgufType::q8_0, lm::GgufKQuantType::Q8_0};

}  // namespace

// mmvq: the decode route (M=1 fast path + the MT=8 register-tile path).
TEST(GgufGemmF32c, Mmvq_M1_gpu) {
    run_case(kQ8_0, Route::kMmvq, 1, 192, 512, 11);
    run_case(kQ4_K, Route::kMmvq, 1, 192, 512, 12);
    run_case(kQ5_K, Route::kMmvq, 1, 192, 512, 13);
    run_case(kQ6_K, Route::kMmvq, 1, 192, 512, 14);
}
TEST(GgufGemmF32c, Mmvq_M8_gpu) {
    run_case(kQ8_0, Route::kMmvq, 8, 192, 512, 21);
    run_case(kQ4_K, Route::kMmvq, 8, 192, 512, 22);
}
// mmq_mma: the prefill route (both tile-config arms of the small-M heuristic).
TEST(GgufGemmF32c, MmqMma_M64_gpu) {
    run_case(kQ8_0, Route::kMmq, 64, 256, 512, 31);
    run_case(kQ4_K, Route::kMmq, 64, 256, 512, 32);
    run_case(kQ5_K, Route::kMmq, 64, 256, 512, 33);
    run_case(kQ6_K, Route::kMmq, 64, 256, 512, 34);
}
TEST(GgufGemmF32c, MmqMma_M512_gpu) {
    run_case(kQ8_0, Route::kMmq, 512, 256, 512, 41);
}
// dequant: the fallback strategy (GEMV M<=8 + tiled M>8).
TEST(GgufGemmF32c, Dequant_M1_gpu) {
    run_case(kQ8_0, Route::kDequant, 1, 192, 512, 51);
    run_case(kQ4_K, Route::kDequant, 1, 192, 512, 52);
}
TEST(GgufGemmF32c, Dequant_M64_gpu) {
    run_case(kQ8_0, Route::kDequant, 64, 192, 512, 61);
    run_case(kQ6_K, Route::kDequant, 64, 192, 512, 62);
}

// The single post-allreduce rounding: dst[i] == __float2bfloat16_rn(src[i]).
TEST(GgufGemmF32c, CastF32ToBf16_gpu) {
    REQUIRES_GPU();
    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();
    std::mt19937 rng(71);
    std::uniform_real_distribution<float> d(-4.0f, 4.0f);
    const int64_t n = 7168 * 3 + 17;   // odd tail: exercise the guard
    std::vector<float> src(n);
    for (auto& v : src) v = d(rng);
    src[0] = 0.0f; src[1] = -0.0f;
    float* d_src = upload(src);
    std::vector<__nv_bfloat16> zero(n, __float2bfloat16(0.0f));
    __nv_bfloat16* d_dst = upload(zero);

    dev.cast_f32_to_bf16(d_dst, d_src, n, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> dst(n);
    ASSERT_EQ(cudaMemcpy(dst.data(), d_dst, n * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    for (int64_t i = 0; i < n; ++i) {
        const __nv_bfloat16 e = __float2bfloat16_rn(src[i]);
        uint16_t a, b;
        std::memcpy(&a, &e, 2);
        std::memcpy(&b, &dst[i], 2);
        ASSERT_EQ(a, b) << "elem " << i;
    }
    cudaFree(d_src); cudaFree(d_dst);
}
