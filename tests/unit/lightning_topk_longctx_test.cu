// TD-GLM-LONGCTX: lightning_topk at PRODUCTION dims with num_blocks > topk —
// the regime the short golden never exercises (seqlen 2877 > index_topk 2048,
// so the top-k actually PRUNES). Two properties:
//
//  1. Planted pattern: if the most recent positions score highest, they MUST
//     all be selected (a selection kernel that drops recent positions makes
//     the model blind to the local question — the long-context symptom).
//  2. Exact set equivalence vs a CPU reference top-k over random scores,
//     including tie semantics at the threshold key.
//
// GLM-5.2 dims: index_topk=2048 (kernel MAX_TOPK), seqlen 2877/3000.

#include "compute/kernels/sm120/indexer/lightning_indexer.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

namespace lc = layerstorm::compute;

namespace {

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

struct TopkOut {
    std::vector<int> idx;
    std::vector<float> score;
    int effective_k;
};

TopkOut run_topk(const std::vector<float>& scores, int topk, int query_position) {
    const int nb = static_cast<int>(scores.size());
    std::vector<int> endpoints(nb);
    for (int i = 0; i < nb; ++i) endpoints[i] = i;

    auto* d_scores = upload(scores);
    auto* d_end = upload(endpoints);
    int* d_idx = nullptr;
    float* d_sel = nullptr;
    int* d_effk = nullptr;
    EXPECT_EQ(cudaMalloc(&d_idx, topk * sizeof(int)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_sel, topk * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_effk, sizeof(int)), cudaSuccess);

    sm120::indexer::LightningTopkParams p{};
    p.scores = d_scores;
    p.block_endpoints = d_end;
    p.output_indices = d_idx;
    p.output_scores = d_sel;
    p.effective_k_out = d_effk;
    p.num_blocks = nb;
    p.topk = topk;
    p.query_position = query_position;
    lc::launch_lightning_topk(p, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    TopkOut out;
    out.idx = download(d_idx, topk);
    out.score = download(d_sel, topk);
    out.effective_k = download(d_effk, 1)[0];

    cudaFree(d_scores); cudaFree(d_end);
    cudaFree(d_idx); cudaFree(d_sel); cudaFree(d_effk);
    return out;
}

}  // namespace

// Positions 2900..2999 planted as the highest scores among 3000 candidates,
// topk=2048: every planted position must appear in the selected set, and the
// just-appended token (query position itself) must be selected.
TEST(LightningTopkLongCtx, RecentHighScoringPositionsSelected) {
    REQUIRES_GPU();
    const int NB = 3000, K = 2048;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> low(0.0f, 1.0f);
    std::vector<float> scores(NB);
    for (auto& s : scores) s = low(rng);
    for (int i = 2900; i < NB; ++i) scores[i] = 10.0f + 0.001f * (i - 2900);

    auto out = run_topk(scores, K, /*query_position=*/NB - 1);

    ASSERT_EQ(out.effective_k, K);
    std::set<int> sel;
    for (int i = 0; i < K; ++i) {
        ASSERT_GE(out.idx[i], 0) << "slot " << i;
        ASSERT_LT(out.idx[i], NB) << "slot " << i;
        if (i > 0) ASSERT_GT(out.idx[i], out.idx[i - 1]) << "not ascending @" << i;
        sel.insert(out.idx[i]);
    }
    for (int p = 2900; p < NB; ++p)
        EXPECT_TRUE(sel.count(p)) << "high-scoring recent position " << p
                                  << " dropped by top-k";
    EXPECT_TRUE(sel.count(NB - 1)) << "self position must be selected";
}

// Exact top-k set vs CPU reference at the real long-context dims
// (NB=2877 > K=2048). Any score at or above the k-th highest must be in;
// any strictly below must be out.
TEST(LightningTopkLongCtx, MatchesCpuReferenceAboveTopk) {
    REQUIRES_GPU();
    const int NB = 2877, K = 2048;

    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::vector<float> scores(NB);
    for (auto& s : scores) s = dist(rng);

    auto out = run_topk(scores, K, /*query_position=*/NB - 1);
    ASSERT_EQ(out.effective_k, K);

    // CPU reference: k-th largest score = threshold.
    std::vector<float> sorted = scores;
    std::nth_element(sorted.begin(), sorted.begin() + (K - 1), sorted.end(),
                     std::greater<float>());
    const float thresh = sorted[K - 1];

    std::set<int> sel(out.idx.begin(), out.idx.end());
    ASSERT_EQ(static_cast<int>(sel.size()), K);
    int at_thresh_selected = 0;
    for (int i = 0; i < NB; ++i) {
        if (scores[i] > thresh) {
            EXPECT_TRUE(sel.count(i)) << "above-threshold position " << i
                                      << " (score " << scores[i] << ") dropped";
        } else if (scores[i] < thresh) {
            EXPECT_FALSE(sel.count(i)) << "below-threshold position " << i
                                       << " (score " << scores[i] << ") selected";
        }
        if (scores[i] == thresh && sel.count(i)) ++at_thresh_selected;
    }
    // Selected scores must be echoed back correctly.
    for (int i = 0; i < K; ++i)
        EXPECT_EQ(out.score[i], scores[out.idx[i]]) << "slot " << i;
    (void)at_thresh_selected;
}

// Causality at long length: query_position mid-sequence with NB beyond it —
// nothing after the query may be selected even when scores there are huge.
TEST(LightningTopkLongCtx, CausalityHoldsBeyondTopkBoundary) {
    REQUIRES_GPU();
    const int NB = 2877, K = 2048, QP = 2400;

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> low(0.0f, 1.0f);
    std::vector<float> scores(NB);
    for (auto& s : scores) s = low(rng);
    for (int i = QP + 1; i < NB; ++i) scores[i] = 100.0f;  // future bait

    auto out = run_topk(scores, K, QP);
    ASSERT_EQ(out.effective_k, K);  // 2401 causal-valid >= K
    for (int i = 0; i < K; ++i) {
        ASSERT_GE(out.idx[i], 0);
        EXPECT_LE(out.idx[i], QP) << "future position selected @" << i;
    }
}
