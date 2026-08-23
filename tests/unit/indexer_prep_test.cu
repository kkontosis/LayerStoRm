// GLM-25a: GPU tests for the DSA indexer prep kernels (LayerNorm+bias, FP8
// single-head key quant+append) as built into the engine.

#include "compute/kernels/sm120/indexer/indexer_prep.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace lc = layerstorm::compute;

namespace {
template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}
template <typename T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}
__nv_bfloat16 bf(float f) { return __float2bfloat16(f); }
float f32(__nv_bfloat16 b) { return __bfloat162float(b); }
}  // namespace

TEST(IndexerPrep, LayerNormBiasMatchesCpu) {
    REQUIRES_GPU();
    const int rows = 5, dim = 128;
    std::mt19937 rng(3);
    std::normal_distribution<float> dist(0.3f, 2.0f);  // nonzero mean → exercises centering

    std::vector<__nv_bfloat16> x(rows * dim), w(dim), b(dim);
    for (auto& v : x) v = bf(dist(rng));
    for (auto& v : w) v = bf(0.5f + 0.5f * dist(rng));
    for (auto& v : b) v = bf(0.1f * dist(rng));

    auto* dx = upload(x);
    auto* dw = upload(w);
    auto* db = upload(b);
    lc::IndexerLayerNormParams p{};
    p.x = dx; p.weight = dw; p.bias = db; p.num_rows = rows; p.dim = dim; p.eps = 1e-5f;
    lc::launch_indexer_layernorm_bias(p, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto got = download(dx, x.size());

    for (int r = 0; r < rows; ++r) {
        double mean = 0;
        for (int i = 0; i < dim; ++i) mean += f32(x[r * dim + i]);
        mean /= dim;
        double var = 0;
        for (int i = 0; i < dim; ++i) { double d = f32(x[r * dim + i]) - mean; var += d * d; }
        var /= dim;
        double inv = 1.0 / std::sqrt(var + 1e-5);
        for (int i = 0; i < dim; ++i) {
            float exp = float((f32(x[r * dim + i]) - mean) * inv) * f32(w[i]) + f32(b[i]);
            EXPECT_NEAR(f32(got[r * dim + i]), exp, 3e-2f) << "r" << r << " i" << i;
        }
    }
    cudaFree(dx); cudaFree(dw); cudaFree(db);
}

TEST(IndexerPrep, KQuantAppendRoundTrips) {
    REQUIRES_GPU();
    const int tokens = 4, dim = 128, num_blocks = 16;
    std::mt19937 rng(9);
    std::normal_distribution<float> dist(0.0f, 1.5f);

    std::vector<__nv_bfloat16> k(tokens * dim);
    for (auto& v : k) v = bf(dist(rng));
    // Scatter tokens to non-contiguous slots to exercise slot_mapping.
    std::vector<int> slots = {2, 7, 11, 5};

    auto* dk = upload(k);
    auto* dslots = upload(slots);
    __nv_fp8_e4m3* dcache = nullptr;
    float* dscales = nullptr;
    ASSERT_EQ(cudaMalloc(&dcache, size_t(num_blocks) * dim * sizeof(__nv_fp8_e4m3)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dscales, num_blocks * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dcache, 0, size_t(num_blocks) * dim * sizeof(__nv_fp8_e4m3)), cudaSuccess);

    lc::IndexerKQuantAppendParams p{};
    p.k_in = dk; p.slot_mapping = dslots; p.k_cache = dcache; p.k_scales = dscales;
    p.num_tokens = tokens; p.index_head_dim = dim;
    lc::launch_indexer_k_quant_append(p, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto cache = download(dcache, size_t(num_blocks) * dim);
    auto scales = download(dscales, num_blocks);

    for (int t = 0; t < tokens; ++t) {
        int slot = slots[t];
        float absmax = 0;
        for (int i = 0; i < dim; ++i) absmax = std::max(absmax, std::abs(f32(k[t * dim + i])));
        EXPECT_NEAR(scales[slot], absmax / 448.0f, 1e-4f * (1 + absmax)) << "t" << t;
        // Dequant fp8*scale should approximate the original within FP8 resolution.
        for (int i = 0; i < dim; ++i) {
            float deq = float(cache[size_t(slot) * dim + i]) * scales[slot];
            EXPECT_NEAR(deq, f32(k[t * dim + i]), 0.05f * (1 + absmax)) << "t" << t << " i" << i;
        }
    }
    // slot_bias = −1: pass "seqlens" (slot+1) as the mapping — the decode path
    // feeds seqlens_k directly so slot = seqlen − 1 = position.
    std::vector<int> seqlens = {3, 8, 12, 6};  // slots+1
    auto* dseq = upload(seqlens);
    ASSERT_EQ(cudaMemset(dcache, 0, size_t(num_blocks) * dim * sizeof(__nv_fp8_e4m3)), cudaSuccess);
    p.slot_mapping = dseq;
    p.slot_bias = -1;
    lc::launch_indexer_k_quant_append(p, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto cache2 = download(dcache, size_t(num_blocks) * dim);
    auto scales2 = download(dscales, num_blocks);
    for (int t = 0; t < tokens; ++t) {
        int slot = seqlens[t] - 1;
        float absmax = 0;
        for (int i = 0; i < dim; ++i) absmax = std::max(absmax, std::abs(f32(k[t * dim + i])));
        EXPECT_NEAR(scales2[slot], absmax / 448.0f, 1e-4f * (1 + absmax)) << "bias t" << t;
        float deq = float(cache2[size_t(slot) * dim]) * scales2[slot];
        EXPECT_NEAR(deq, f32(k[t * dim]), 0.05f * (1 + absmax)) << "bias t" << t;
    }
    cudaFree(dseq);

    cudaFree(dk); cudaFree(dslots); cudaFree(dcache); cudaFree(dscales);
}
