// KVS-4: GLOBAL→LOCAL DSA sparse-index translation under sequence-sharded KV.
//
// 1. The translate kernel must agree with the INV-4.9e shard math
//    (kv_shard_math.h == PageAllocator::dcp_rank_for_token): a rank's output
//    is exactly the ascending, compacted, −1-padded list of LOCAL slot
//    indices of the globally-selected positions that rank OWNS, with the
//    per-rank lengths partitioning the input length (disjoint + complete).
// 2. A rank owning NONE of the selection gets length 0 / all −1 — and the
//    downstream sparse consumer (launch_prefill_sparse, the exact kernel the
//    sharded QAG path calls) must then emit zero output + lse=+inf, which
//    dcp_lse_correct maps to combine weight 0 (INV-KVS-EMPTY: the empty
//    selected shard is dense-empty-shard-equivalent).

#include "compute/kernels/sm120/indexer/indexer_shard_translate.h"
#include "compute/kernels/attention/mla_attention.h"
#include "daemon/kv_shard_math.h"

#include "sm120/prefill/sparse/params.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace {

using layerstorm::compute::IndexerShardTranslateParams;
using layerstorm::compute::launch_indexer_shard_translate;
namespace kvshard = layerstorm::daemon::kvshard;

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess)                                           \
            << "CUDA error: " << cudaGetErrorString(_err);                     \
    } while (0)

// CPU reference built ONLY from kv_shard_math (independent of the kernel's
// closed-form local-index formula): local slot of g on its owner rank ==
// owned_len(rank, g) == number of positions in [0, g) that rank owns.
std::vector<int> cpu_translate(const std::vector<int>& global_row, int glen,
                               int topk, int chunk, int dcp, int rank,
                               int* out_len) {
    std::vector<int> out(static_cast<size_t>(topk), -1);
    int n = 0;
    for (int i = 0; i < glen; ++i) {
        const int g = global_row[static_cast<size_t>(i)];
        if (g < 0) continue;
        if (kvshard::owner_rank(static_cast<uint32_t>(g), chunk, dcp) != rank)
            continue;
        out[static_cast<size_t>(n++)] =
            kvshard::owned_len(rank, static_cast<uint32_t>(g), chunk, dcp);
    }
    *out_len = n;
    return out;
}

struct DeviceRun {
    std::vector<int> local;    // [B, topk]
    std::vector<int> lengths;  // [B]
};

DeviceRun run_kernel(const std::vector<int>& global_indices,
                     const std::vector<int>& global_lengths,
                     int B, int topk, int chunk, int dcp, int rank) {
    int *dg = nullptr, *dgl = nullptr, *dl = nullptr, *dll = nullptr;
    EXPECT_EQ(cudaMalloc(&dg, global_indices.size() * sizeof(int)),
              cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dgl, global_lengths.size() * sizeof(int)),
              cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dl, global_indices.size() * sizeof(int)),
              cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dll, global_lengths.size() * sizeof(int)),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dg, global_indices.data(),
                         global_indices.size() * sizeof(int),
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dgl, global_lengths.data(),
                         global_lengths.size() * sizeof(int),
                         cudaMemcpyHostToDevice), cudaSuccess);
    // Poison outputs: the kernel must overwrite every entry.
    EXPECT_EQ(cudaMemset(dl, 0xAB, global_indices.size() * sizeof(int)),
              cudaSuccess);
    EXPECT_EQ(cudaMemset(dll, 0xAB, global_lengths.size() * sizeof(int)),
              cudaSuccess);

    IndexerShardTranslateParams p{};
    p.global_indices = dg;
    p.global_lengths = dgl;
    p.local_indices = dl;
    p.local_lengths = dll;
    p.num_tokens = B;
    p.topk = topk;
    p.chunk_tokens = chunk;
    p.dcp_size = dcp;
    p.rank = rank;
    launch_indexer_shard_translate(p, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    DeviceRun r;
    r.local.resize(global_indices.size());
    r.lengths.resize(global_lengths.size());
    EXPECT_EQ(cudaMemcpy(r.local.data(), dl,
                         r.local.size() * sizeof(int),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.lengths.data(), dll,
                         r.lengths.size() * sizeof(int),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dg); cudaFree(dgl); cudaFree(dl); cudaFree(dll);
    return r;
}

// Random sorted-ascending distinct global top-k rows (lightning-topk output
// shape: glen valid entries then −1 padding).
std::vector<int> random_topk_row(std::mt19937& rng, int glen, int topk,
                                 int seqlen) {
    std::vector<int> pool(static_cast<size_t>(seqlen));
    std::iota(pool.begin(), pool.end(), 0);
    std::shuffle(pool.begin(), pool.end(), rng);
    std::vector<int> row(static_cast<size_t>(topk), -1);
    std::copy(pool.begin(), pool.begin() + glen, row.begin());
    std::sort(row.begin(), row.begin() + glen);
    return row;
}

TEST(IndexerShardTranslate, MatchesShardMathReference) {
    REQUIRES_GPU();
    std::mt19937 rng(20260706);

    // Production-shaped (topk=2048, chunk=64-ish) plus odd shapes.
    struct Case { int B, topk, chunk, dcp, seqlen; };
    const Case cases[] = {
        {1, 2048, 64, 2, 2877},   // GLM-5.2 long-golden shape
        {3, 2048, 64, 2, 4096},   // multi-row
        {2, 512, 128, 3, 3000},   // 3 ranks, different chunk
        {1, 64, 32, 4, 250},      // partial last cycle
    };
    for (const auto& c : cases) {
        std::vector<int> gind(static_cast<size_t>(c.B) * c.topk, -1);
        std::vector<int> glen(static_cast<size_t>(c.B), 0);
        for (int b = 0; b < c.B; ++b) {
            const int len = std::min(
                c.topk, 1 + static_cast<int>(rng() % c.seqlen));
            glen[static_cast<size_t>(b)] = len;
            auto row = random_topk_row(rng, len, c.topk, c.seqlen);
            std::copy(row.begin(), row.end(),
                      gind.begin() + static_cast<size_t>(b) * c.topk);
        }

        std::vector<int> len_sum(static_cast<size_t>(c.B), 0);
        for (int r = 0; r < c.dcp; ++r) {
            const auto got = run_kernel(gind, glen, c.B, c.topk, c.chunk,
                                        c.dcp, r);
            for (int b = 0; b < c.B; ++b) {
                int ref_len = 0;
                const std::vector<int> row(
                    gind.begin() + static_cast<size_t>(b) * c.topk,
                    gind.begin() + static_cast<size_t>(b + 1) * c.topk);
                const auto ref = cpu_translate(row, glen[static_cast<size_t>(b)],
                                               c.topk, c.chunk, c.dcp, r,
                                               &ref_len);
                ASSERT_EQ(got.lengths[static_cast<size_t>(b)], ref_len)
                    << "rank " << r << " row " << b << " topk " << c.topk;
                for (int i = 0; i < c.topk; ++i)
                    ASSERT_EQ(got.local[static_cast<size_t>(b) * c.topk + i],
                              ref[static_cast<size_t>(i)])
                        << "rank " << r << " row " << b << " entry " << i;
                // Ascending within the valid prefix (staging order).
                for (int i = 1; i < ref_len; ++i)
                    ASSERT_LT(got.local[static_cast<size_t>(b) * c.topk + i - 1],
                              got.local[static_cast<size_t>(b) * c.topk + i]);
                len_sum[static_cast<size_t>(b)] +=
                    got.lengths[static_cast<size_t>(b)];
            }
        }
        // Disjoint + complete: per-rank lengths partition the selection.
        for (int b = 0; b < c.B; ++b)
            ASSERT_EQ(len_sum[static_cast<size_t>(b)],
                      glen[static_cast<size_t>(b)])
                << "row " << b;
    }
}

TEST(IndexerShardTranslate, EmptySelectedShardYieldsZeroLength) {
    REQUIRES_GPU();
    // Selection entirely inside rank 0's chunks (positions 0..63 with
    // chunk=64, dcp=2) → rank 1 owns none: length 0, all −1.
    const int topk = 128, chunk = 64, dcp = 2;
    std::vector<int> gind(static_cast<size_t>(topk), -1);
    for (int i = 0; i < 64; ++i) gind[static_cast<size_t>(i)] = i;
    const std::vector<int> glen{64};

    const auto r1 = run_kernel(gind, glen, 1, topk, chunk, dcp, /*rank=*/1);
    EXPECT_EQ(r1.lengths[0], 0);
    for (int i = 0; i < topk; ++i)
        EXPECT_EQ(r1.local[static_cast<size_t>(i)], -1) << "entry " << i;

    const auto r0 = run_kernel(gind, glen, 1, topk, chunk, dcp, /*rank=*/0);
    EXPECT_EQ(r0.lengths[0], 64);
    for (int i = 0; i < 64; ++i)
        EXPECT_EQ(r0.local[static_cast<size_t>(i)], i);  // identity inside chunk 0
}

// INV-KVS-EMPTY for the SPARSE consumer: topk_length == 0 (the rank owns
// none of the global selection) must produce zero output + lse=+inf so the
// QAG combine weights this rank 0 — same contract the dense empty-shard
// tests lock (PrefillDenseCausal.EmptyShard*). Uses launch_prefill_sparse,
// the exact entry point the sharded sparse path calls (d_qk=576 absorbed
// MLA, d_v=512).
TEST(IndexerShardTranslate, SparseConsumerEmptySelectionZeroOutputInfLse) {
    REQUIRES_GPU();
    constexpr int kSQ = 2, kSKV = 64, kHQ = 16, kDQK = 576, kDV = 512;
    constexpr int kTopk = 32;

    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    std::vector<__nv_bfloat16> q(static_cast<size_t>(kSQ) * kHQ * kDQK);
    std::vector<__nv_bfloat16> kv(static_cast<size_t>(kSKV) * kDQK);
    for (auto& v : q) v = __float2bfloat16(dist(rng));
    for (auto& v : kv) v = __float2bfloat16(dist(rng));
    // Indices all −1 (translate output at length 0) + lengths 0.
    std::vector<int> ind(static_cast<size_t>(kSQ) * kTopk, -1);
    std::vector<int> len(static_cast<size_t>(kSQ), 0);

    void *dq = nullptr, *dkv = nullptr, *dout = nullptr;
    int *dind = nullptr, *dlen = nullptr;
    float* dlse = nullptr;
    CUDA_CHECK(cudaMalloc(&dq, q.size() * 2));
    CUDA_CHECK(cudaMalloc(&dkv, kv.size() * 2));
    CUDA_CHECK(cudaMalloc(&dout, static_cast<size_t>(kSQ) * kHQ * kDV * 2));
    CUDA_CHECK(cudaMalloc(&dlse, static_cast<size_t>(kSQ) * kHQ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dind, ind.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dlen, len.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dkv, kv.data(), kv.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dind, ind.data(), ind.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dlen, len.data(), len.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    // Poison the output so "zero" is proven, not inherited.
    CUDA_CHECK(cudaMemset(dout, 0xAB, static_cast<size_t>(kSQ) * kHQ * kDV * 2));

    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    SparseAttnFwdParams p{};
    p.s_q = kSQ; p.s_kv = kSKV; p.h_q = kHQ; p.h_kv = 1;
    p.d_qk = kDQK; p.d_v = kDV; p.topk = kTopk;
    p.sm_scale = sm_scale;
    p.sm_scale_div_log2 = sm_scale / std::log(2.0f);
    p.q = static_cast<cutlass::bfloat16_t*>(dq);
    p.kv = static_cast<cutlass::bfloat16_t*>(dkv);
    p.indices = dind;
    p.topk_length = dlen;
    p.attn_sink = nullptr;
    p.stride_q_s_q = kHQ * kDQK; p.stride_q_h_q = kDQK;
    p.stride_kv_s_kv = kDQK; p.stride_kv_h_kv = kDQK;
    p.stride_indices_s_q = kTopk; p.stride_indices_h_kv = kTopk;
    p.out = static_cast<cutlass::bfloat16_t*>(dout);
    p.max_logits = nullptr;
    p.lse = dlse;
    p.num_sm = 0;
    p.stream = nullptr;
    layerstorm::compute::launch_prefill_sparse(p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out(static_cast<size_t>(kSQ) * kHQ * kDV);
    std::vector<float> lse(static_cast<size_t>(kSQ) * kHQ);
    CUDA_CHECK(cudaMemcpy(out.data(), dout, out.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lse.data(), dlse, lse.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < out.size(); ++i)
        ASSERT_EQ(__bfloat162float(out[i]), 0.0f) << "out[" << i << "]";
    for (size_t i = 0; i < lse.size(); ++i) {
        ASSERT_TRUE(std::isinf(lse[i]) && lse[i] > 0) << "lse[" << i << "]";
    }
    cudaFree(dq); cudaFree(dkv); cudaFree(dout);
    cudaFree(dlse); cudaFree(dind); cudaFree(dlen);
}

}  // namespace
