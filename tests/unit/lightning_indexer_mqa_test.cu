// GLM-25a: GPU validation of the DSA lightning-indexer kernels as built into
// the engine (src/compute/kernels/sm120/indexer/lightning_indexer.cu driver).
//
//  1. MQA score kernel vs a CPU reference (fp8-dequant dot, ReLU, weighted sum).
//  2. MQA kernel vs the per-head kernel fed the SAME key replicated across all
//     heads — the exact equivalence that justifies the memory-saving MQA layout.
//  3. Top-k kernel vs a CPU reference: causality mask + ascending-sorted output.
//
// GLM-5.2 dims: index_n_heads=32, index_head_dim=128, index_topk<=2048.

#include "compute/kernels/sm120/indexer/lightning_indexer.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace lc = layerstorm::compute;

namespace {

constexpr int kHeads = 32;    // GLM-5.2 index_n_heads
constexpr int kDim = 128;     // GLM-5.2 index_head_dim

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice),
              cudaSuccess);
    return d;
}

template <typename T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return h;
}

// Host-side fp8/bf16 round-trip so the CPU reference sees the SAME quantized
// values the kernel reads.
float fp8_roundtrip(float f) { return float(__nv_fp8_e4m3(f)); }
float bf16_roundtrip(float f) { return __bfloat162float(__float2bfloat16(f)); }

struct ScoreFixture {
    std::vector<__nv_bfloat16> q_bf16;      // [kHeads, kDim]
    std::vector<float> q_vals;              // bf16-rounded floats
    std::vector<__nv_fp8_e4m3> k_fp8;       // [num_blocks, kDim] (MQA layout)
    std::vector<float> k_vals;              // fp8-rounded floats
    std::vector<float> k_scales;            // [num_blocks]
    std::vector<float> score_proj;          // [kHeads]
    int num_blocks;

    explicit ScoreFixture(int blocks, uint32_t seed) : num_blocks(blocks) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        q_bf16.resize(size_t(kHeads) * kDim);
        q_vals.resize(q_bf16.size());
        for (size_t i = 0; i < q_bf16.size(); ++i) {
            float v = dist(rng);
            q_bf16[i] = __float2bfloat16(v);
            q_vals[i] = bf16_roundtrip(v);
        }

        k_fp8.resize(size_t(blocks) * kDim);
        k_vals.resize(k_fp8.size());
        for (size_t i = 0; i < k_fp8.size(); ++i) {
            float v = dist(rng);
            k_fp8[i] = __nv_fp8_e4m3(v);
            k_vals[i] = fp8_roundtrip(v);
        }

        k_scales.resize(blocks);
        for (int n = 0; n < blocks; ++n) k_scales[n] = 0.5f + 0.01f * (n % 7);

        score_proj.resize(kHeads);
        for (int h = 0; h < kHeads; ++h) score_proj[h] = dist(rng);
    }

    // CPU reference on the round-tripped values: identical math to the kernel.
    std::vector<float> cpu_scores() const {
        std::vector<float> scores(num_blocks, 0.0f);
        for (int n = 0; n < num_blocks; ++n) {
            float total = 0.0f;
            for (int h = 0; h < kHeads; ++h) {
                float dot = 0.0f;
                for (int d = 0; d < kDim; ++d) {
                    dot += q_vals[size_t(h) * kDim + d] * k_vals[size_t(n) * kDim + d];
                }
                dot *= k_scales[n];
                total += std::max(dot, 0.0f) * score_proj[h];
            }
            scores[n] = total;
        }
        return scores;
    }
};

}  // namespace

TEST(LightningIndexerMqa, ScoreMatchesCpuReference) {
    REQUIRES_GPU();
    // 259: not a multiple of BLOCKS_PER_CTA=4 — exercises the tail CTA.
    ScoreFixture fx(259, /*seed=*/42);

    auto* d_q = upload(fx.q_bf16);
    auto* d_k = upload(fx.k_fp8);
    auto* d_scales = upload(fx.k_scales);
    auto* d_proj = upload(fx.score_proj);
    float* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, fx.num_blocks * sizeof(float)), cudaSuccess);

    sm120::indexer::LightningScoreMqaParams p{};
    p.q_proj = d_q;
    p.indexer_k_cache = d_k;
    p.k_scales = d_scales;
    p.score_proj = d_proj;
    p.scores_out = d_out;
    p.num_blocks = fx.num_blocks;
    p.index_n_heads = kHeads;
    p.index_head_dim = kDim;

    lc::launch_lightning_score_mqa(p, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto gpu = download(d_out, fx.num_blocks);
    auto cpu = fx.cpu_scores();

    // fp32 accumulation in different orders; values are O(sqrt(128)*32) ~ 1e2.
    for (int n = 0; n < fx.num_blocks; ++n) {
        EXPECT_NEAR(gpu[n], cpu[n], 5e-3f * (1.0f + std::abs(cpu[n])))
            << "block " << n;
    }

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_scales);
    cudaFree(d_proj); cudaFree(d_out);
}

TEST(LightningIndexerMqa, MqaEqualsPerHeadWithReplicatedK) {
    REQUIRES_GPU();
    ScoreFixture fx(128, /*seed=*/7);

    auto* d_q = upload(fx.q_bf16);
    auto* d_scales = upload(fx.k_scales);
    auto* d_proj = upload(fx.score_proj);

    // MQA layout: [blocks, kDim].
    auto* d_k_mqa = upload(fx.k_fp8);
    float* d_out_mqa = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out_mqa, fx.num_blocks * sizeof(float)), cudaSuccess);

    sm120::indexer::LightningScoreMqaParams pm{};
    pm.q_proj = d_q;
    pm.indexer_k_cache = d_k_mqa;
    pm.k_scales = d_scales;
    pm.score_proj = d_proj;
    pm.scores_out = d_out_mqa;
    pm.num_blocks = fx.num_blocks;
    pm.index_n_heads = kHeads;
    pm.index_head_dim = kDim;
    lc::launch_lightning_score_mqa(pm, nullptr);

    // Per-head layout: the same key replicated into every head slot.
    std::vector<__nv_fp8_e4m3> k_rep(size_t(fx.num_blocks) * kHeads * kDim);
    for (int n = 0; n < fx.num_blocks; ++n)
        for (int h = 0; h < kHeads; ++h)
            for (int d = 0; d < kDim; ++d)
                k_rep[(size_t(n) * kHeads + h) * kDim + d] =
                    fx.k_fp8[size_t(n) * kDim + d];
    auto* d_k_rep = upload(k_rep);
    float* d_out_ph = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out_ph, fx.num_blocks * sizeof(float)), cudaSuccess);

    sm120::indexer::LightningScoreParams pp{};
    pp.q_proj = d_q;
    pp.indexer_k_cache = d_k_rep;
    pp.k_scales = d_scales;
    pp.score_proj = d_proj;
    pp.scores_out = d_out_ph;
    pp.num_blocks = fx.num_blocks;
    pp.index_n_heads = kHeads;
    pp.index_head_dim = kDim;
    lc::launch_lightning_score(pp, nullptr);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto mqa = download(d_out_mqa, fx.num_blocks);
    auto ph = download(d_out_ph, fx.num_blocks);
    for (int n = 0; n < fx.num_blocks; ++n) {
        EXPECT_NEAR(mqa[n], ph[n], 1e-3f * (1.0f + std::abs(ph[n]))) << "block " << n;
    }

    cudaFree(d_q); cudaFree(d_scales); cudaFree(d_proj);
    cudaFree(d_k_mqa); cudaFree(d_out_mqa);
    cudaFree(d_k_rep); cudaFree(d_out_ph);
}

TEST(LightningIndexerMqa, TopkCausalitySortedAndPadded) {
    REQUIRES_GPU();
    const int num_blocks = 300;
    const int topk = 64;
    const int query_position = 199;  // blocks 200..299 are future → excluded

    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> scores(num_blocks);
    for (auto& s : scores) s = dist(rng);
    std::vector<int> endpoints(num_blocks);
    for (int n = 0; n < num_blocks; ++n) endpoints[n] = n;  // per-token DSA

    auto* d_scores = upload(scores);
    auto* d_end = upload(endpoints);
    int* d_idx = nullptr;
    float* d_sel = nullptr;
    int* d_effk = nullptr;
    ASSERT_EQ(cudaMalloc(&d_idx, topk * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sel, topk * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_effk, sizeof(int)), cudaSuccess);

    sm120::indexer::LightningTopkParams p{};
    p.scores = d_scores;
    p.block_endpoints = d_end;
    p.output_indices = d_idx;
    p.output_scores = d_sel;
    p.effective_k_out = d_effk;
    p.num_blocks = num_blocks;
    p.topk = topk;
    p.query_position = query_position;

    lc::launch_lightning_topk(p, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto idx = download(d_idx, topk);
    auto effk = download(d_effk, 1);

    // CPU reference: top-64 of the 200 causal blocks, returned sorted ascending.
    std::vector<int> causal(200);
    for (int n = 0; n < 200; ++n) causal[n] = n;
    std::partial_sort(causal.begin(), causal.begin() + topk, causal.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<int> expect(causal.begin(), causal.begin() + topk);
    std::sort(expect.begin(), expect.end());

    ASSERT_EQ(effk[0], topk);
    for (int i = 0; i < topk; ++i) {
        EXPECT_EQ(idx[i], expect[i]) << "position " << i;
        EXPECT_LE(idx[i], query_position) << "causality violated at " << i;
    }

    cudaFree(d_scores); cudaFree(d_end);
    cudaFree(d_idx); cudaFree(d_sel); cudaFree(d_effk);
}
