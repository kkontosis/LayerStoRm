// Unit tests for the MoE grouped-GEMM metadata kernels (FP4-ACT-SCALE).
//
// GPU-required: verifies launch_gather_alphas_scaled reads ws2/input_scale
// from scattered slot pointers, computes alpha = ws2 * is, persists is to the
// quantizer array, and guards is <= 0 → 1.0 (missing experts point at a
// zero buffer — their gathered input_scale reads 0.0).

#include "compute/kernels/moe/moe_gemm_meta.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace lc = layerstorm::compute;

namespace {

template <typename T>
T* to_device(const std::vector<T>& host) {
    T* d = nullptr;
    cudaMalloc(&d, host.size() * sizeof(T));
    cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <typename T>
std::vector<T> to_host(const T* d, size_t count) {
    std::vector<T> h(count);
    cudaMemcpy(h.data(), d, count * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

}  // namespace

class MoeGemmMetaTest : public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        cudaGetDeviceCount(&count);
        if (count == 0) GTEST_SKIP() << "No CUDA device available";
        cudaSetDevice(0);
    }
};

// Host reference for the block-scan kernel: problem_sizes = (M_e, N, K) per
// expert; sf_offsets = exclusive prefix sum of pad128(M_e), with a trailing
// total at index num_experts. Mirrors the former single-thread sequential scan.
static void reference_meta(const std::vector<int32_t>& expert_offsets, int N,
                           int K, int num_experts,
                           std::vector<int32_t>& ps_ref,
                           std::vector<int32_t>& sf_ref) {
    ps_ref.assign(num_experts * 3, 0);
    sf_ref.assign(num_experts + 1, 0);
    int32_t acc = 0;
    for (int e = 0; e < num_experts; ++e) {
        int32_t M_e = expert_offsets[e + 1] - expert_offsets[e];
        ps_ref[e * 3] = M_e;
        ps_ref[e * 3 + 1] = N;
        ps_ref[e * 3 + 2] = K;
        sf_ref[e] = acc;
        acc += ((M_e + 127) / 128) * 128;
    }
    sf_ref[num_experts] = acc;
}

TEST_F(MoeGemmMetaTest, PopulateGemmMetaMatchesReference) {
    // Randomized expert row counts across a range that exercises pad128 rounding
    // (0 rows, exact multiples, and partials). Parallel block-scan output must
    // be bit-identical to the sequential host reference.
    for (int num_experts : {1, 8, 64, 256, 384}) {
        const int N = 2048, K = 6144;
        std::vector<int32_t> offsets(num_experts + 1, 0);
        uint32_t s = 0x1234u + num_experts;
        for (int e = 0; e < num_experts; ++e) {
            s = s * 1664525u + 1013904223u;
            int32_t rows = static_cast<int32_t>(s % 300);  // 0..299 rows
            offsets[e + 1] = offsets[e] + rows;
        }

        std::vector<int32_t> ps_ref, sf_ref;
        reference_meta(offsets, N, K, num_experts, ps_ref, sf_ref);

        auto* d_off = to_device(offsets);
        int32_t* d_ps = nullptr;
        int32_t* d_sf = nullptr;
        cudaMalloc(&d_ps, num_experts * 3 * sizeof(int32_t));
        cudaMalloc(&d_sf, (num_experts + 1) * sizeof(int32_t));

        lc::launch_populate_gemm_meta(d_ps, d_sf, d_off, N, K, num_experts,
                                      nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "ne=" << num_experts;

        auto ps = to_host(d_ps, num_experts * 3);
        auto sf = to_host(d_sf, num_experts + 1);
        EXPECT_EQ(ps, ps_ref) << "problem_sizes mismatch, ne=" << num_experts;
        EXPECT_EQ(sf, sf_ref) << "sf_offsets mismatch, ne=" << num_experts;

        cudaFree(d_ps);
        cudaFree(d_sf);
        cudaFree(d_off);
    }
}

TEST_F(MoeGemmMetaTest, PopulateGemmMetaNullSfOffsets) {
    // FP8 path passes sf_offsets == nullptr; problem_sizes must still be filled.
    const int num_experts = 5, N = 512, K = 512;
    std::vector<int32_t> offsets{0, 3, 3, 130, 131, 400};  // includes empty + pad
    std::vector<int32_t> ps_ref, sf_ref;
    reference_meta(offsets, N, K, num_experts, ps_ref, sf_ref);

    auto* d_off = to_device(offsets);
    int32_t* d_ps = nullptr;
    cudaMalloc(&d_ps, num_experts * 3 * sizeof(int32_t));

    lc::launch_populate_gemm_meta(d_ps, nullptr, d_off, N, K, num_experts,
                                  nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto ps = to_host(d_ps, num_experts * 3);
    EXPECT_EQ(ps, ps_ref);

    cudaFree(d_ps);
    cudaFree(d_off);
}

TEST_F(MoeGemmMetaTest, GatherAlphasScaled) {
    // Three fake 64-byte "projection regions": ws2 at end-8, is at end-4.
    // Expert 2 simulates a missing expert (zero buffer → is reads 0.0).
    constexpr int kE = 3;
    constexpr size_t kProj = 64;
    const float ws2[kE] = {0.25f, 2.0f, 0.0f};
    const float is[kE]  = {0.004f, 0.5f, 0.0f};

    std::vector<uint8_t> h_slots(kE * kProj, 0);
    for (int e = 0; e < kE; ++e) {
        std::memcpy(h_slots.data() + e * kProj + kProj - 8, &ws2[e], 4);
        std::memcpy(h_slots.data() + e * kProj + kProj - 4, &is[e], 4);
    }
    auto* d_slots = to_device(h_slots);

    std::vector<const void*> h_ptrs(kE);
    for (int e = 0; e < kE; ++e) h_ptrs[e] = d_slots + e * kProj;
    auto* d_ptrs = to_device(h_ptrs);

    float* d_alphas = nullptr;
    float* d_is = nullptr;
    cudaMalloc(&d_alphas, kE * sizeof(float));
    cudaMalloc(&d_is, kE * sizeof(float));

    lc::launch_gather_alphas_scaled(
        d_alphas, d_is, d_ptrs,
        static_cast<int64_t>(kProj) - 8, static_cast<int64_t>(kProj) - 4,
        kE, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto alphas = to_host(d_alphas, kE);
    auto is_out = to_host(d_is, kE);

    EXPECT_FLOAT_EQ(alphas[0], 0.25f * 0.004f);
    EXPECT_FLOAT_EQ(is_out[0], 0.004f);
    EXPECT_FLOAT_EQ(alphas[1], 2.0f * 0.5f);
    EXPECT_FLOAT_EQ(is_out[1], 0.5f);
    // Missing expert: is 0.0 → guard 1.0; alpha = ws2(0) * 1.0 = 0 (zero
    // weights null the GEMM); quantizer must never see is = 0.
    EXPECT_FLOAT_EQ(alphas[2], 0.0f);
    EXPECT_FLOAT_EQ(is_out[2], 1.0f);

    cudaFree(d_alphas);
    cudaFree(d_is);
    cudaFree(const_cast<void*>(static_cast<const void*>(d_ptrs)));
    cudaFree(d_slots);
}

TEST_F(MoeGemmMetaTest, GatherAlphasScaledNullIsOut) {
    // input_scales_out == nullptr: alphas still ws2 * is.
    constexpr size_t kProj = 32;
    const float ws2 = 3.0f, is = 0.125f;
    std::vector<uint8_t> h_slot(kProj, 0);
    std::memcpy(h_slot.data() + kProj - 8, &ws2, 4);
    std::memcpy(h_slot.data() + kProj - 4, &is, 4);
    auto* d_slot = to_device(h_slot);
    std::vector<const void*> h_ptrs{d_slot};
    auto* d_ptrs = to_device(h_ptrs);

    float* d_alphas = nullptr;
    cudaMalloc(&d_alphas, sizeof(float));
    lc::launch_gather_alphas_scaled(
        d_alphas, nullptr, d_ptrs,
        static_cast<int64_t>(kProj) - 8, static_cast<int64_t>(kProj) - 4,
        1, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto alphas = to_host(d_alphas, 1);
    EXPECT_FLOAT_EQ(alphas[0], 0.375f);

    cudaFree(d_alphas);
    cudaFree(const_cast<void*>(static_cast<const void*>(d_ptrs)));
    cudaFree(d_slot);
}
