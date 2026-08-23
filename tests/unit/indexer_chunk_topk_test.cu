// TD-SPARSE-CHUNK-PREFILL: per-chunk-position producer top-k vs a CPU
// indexer reference.
//
// The sparse-prefill producer (DcpExecutor::produce_sparse_indices_prefill)
// scores each chunk row b against its OWN causal prefix [0, len_b) — with
// the whole chunk's keys already appended, so storage holds positions PAST
// row b's — and causal-top-k's at query position len_b − 1. This test drives
// the exact backend call the producer's per-row loop issues
// (CudaSm120DeviceBackend::indexer_score_topk over paged indexer-K) and
// checks, per chunk row:
//
//   1. The selection equals a CPU reference: lightning scores
//      (fp8-dequant dot × per-position scale, ReLU, weighted head sum) over
//      [0, len_b), then top-k with ascending-sorted output and
//      effective_k = min(topk, len_b).
//   2. CAUSALITY: no selected index ≥ len_b, even though the storage
//      physically contains all chunk positions (later rows' keys).
//   3. Bounding equivalence: scoring only [0, len_b) blocks (the producer's
//      nb = len_b) equals scoring ALL stored blocks with the causal cutoff
//      query_position = len_b − 1 doing the exclusion — the two halves of
//      the causal defense agree exactly.
//
// Production dims: index_n_heads=32, index_head_dim=128 (GLM-5.2); topk
// small (6) so the top-k actually PRUNES inside the chunk prefix.

#include "compute/cuda_sm120_device_backend.h"
#include "core/attention_device.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace lc = layerstorm::compute;
namespace cfg = layerstorm::config;

namespace {

constexpr int kNIH = 32;   // GLM-5.2 index_n_heads
constexpr int kIHD = 128;  // GLM-5.2 index_head_dim

cfg::GpuRef make_gpu() { return {0, 0, cfg::GpuType::rtx5090}; }

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T),
                         cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}
template <typename T>
std::vector<T> download(const void* d, size_t n) {
    std::vector<T> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return h;
}

float fp8_roundtrip(float f) { return float(__nv_fp8_e4m3(f)); }
float bf16_roundtrip(float f) { return __bfloat162float(__float2bfloat16(f)); }

}  // namespace

TEST(IndexerChunkTopk, PerChunkRowCausalTopkMatchesCpu) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    // Prefix of N=21 positions in PT=8 pages (partial last page); the chunk
    // rows are positions 10..20 — each row's causal prefix (11..21) exceeds
    // ITK=6 so the top-k prunes, and the storage always holds keys PAST the
    // row (the chunk was appended in full before scoring).
    const int N = 21, PT = 8, ITK = 6;
    const int chunk_lo = 10;
    const int n_pages = (N + PT - 1) / PT;

    std::mt19937 rng(31);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Stored keys (FP8) + per-position scales, and their exact host values.
    std::vector<__nv_fp8_e4m3> k(size_t(N) * kIHD);
    std::vector<float> k_vals(k.size());
    for (size_t i = 0; i < k.size(); ++i) {
        float v = dist(rng);
        k[i] = __nv_fp8_e4m3(v);
        k_vals[i] = fp8_roundtrip(v);
    }
    std::vector<float> scales(N);
    for (int i = 0; i < N; ++i) scales[i] = 0.02f + 0.005f * (i % 5);

    // Paged layout: page = [PT×IHD FP8 | PT F32 scales].
    std::vector<void*> page_ptrs(n_pages);
    const size_t page_bytes = size_t(PT) * kIHD + size_t(PT) * sizeof(float);
    for (int p = 0; p < n_pages; ++p) {
        ASSERT_EQ(cudaMalloc(&page_ptrs[p], page_bytes), cudaSuccess);
        const int rows = std::min(PT, N - p * PT);
        ASSERT_EQ(cudaMemcpy(page_ptrs[p], k.data() + size_t(p) * PT * kIHD,
                             size_t(rows) * kIHD, cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(static_cast<std::byte*>(page_ptrs[p])
                                 + size_t(PT) * kIHD,
                             scales.data() + size_t(p) * PT,
                             size_t(rows) * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    // Per chunk row: its own query + score weights (the producer computes
    // [B, NIH*IHD] / [B, NIH] batched; the per-row loop slices row b).
    const int B = N - chunk_lo;  // 11 chunk rows
    std::vector<__nv_bfloat16> q_all(size_t(B) * kNIH * kIHD);
    std::vector<float> q_vals(q_all.size());
    for (size_t i = 0; i < q_all.size(); ++i) {
        float v = dist(rng);
        q_all[i] = __float2bfloat16(v);
        q_vals[i] = bf16_roundtrip(v);
    }
    std::vector<float> w_all(size_t(B) * kNIH);
    for (auto& v : w_all) v = dist(rng) / 8.0f;

    std::vector<int> endpoints(N);
    for (int i = 0; i < N; ++i) endpoints[i] = i;

    auto* dq = upload(q_all);
    auto* dw = upload(w_all);
    auto* dend = upload(endpoints);
    void *dscores, *dtopk_s, *didx, *dlen, *didx2, *dlen2;
    ASSERT_EQ(cudaMalloc(&dscores, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dtopk_s, ITK * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&didx, size_t(B) * ITK * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dlen, size_t(B) * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&didx2, size_t(B) * ITK * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dlen2, size_t(B) * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(didx, 0xFF, size_t(B) * ITK * sizeof(int)),
              cudaSuccess);
    ASSERT_EQ(cudaMemset(didx2, 0xFF, size_t(B) * ITK * sizeof(int)),
              cudaSuccess);

    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();

    // The producer's per-row loop: row b scores [0, len_b) at qpos len_b−1.
    auto run_row = [&](int b, int nb, int qpos, void* idx_out,
                       void* len_out) {
        lc::IndexerScoreTopkArgs a{};
        a.q_all = static_cast<std::byte*>(static_cast<void*>(dq))
                  + size_t(b) * kNIH * kIHD * 2;
        a.score_proj_all = static_cast<float*>(static_cast<void*>(dw))
                           + size_t(b) * kNIH;
        a.block_endpoints = dend;
        a.scores_scratch = dscores;
        a.topk_scores_scratch = dtopk_s;
        a.sparse_indices_out = static_cast<int*>(idx_out) + size_t(b) * ITK;
        a.topk_lengths_out = static_cast<int*>(len_out) + b;
        a.num_tokens = 1;
        a.num_blocks = nb;
        a.n_heads = kNIH;
        a.head_dim = kIHD;
        a.topk = ITK;
        a.query_position_base = qpos;
        a.k_pages = const_cast<const void* const*>(page_ptrs.data());
        a.num_k_pages = (nb + PT - 1) / PT;
        a.page_tokens = PT;
        dev.indexer_score_topk(a, nullptr);
    };

    std::vector<std::vector<float>> gpu_scores(B);
    for (int b = 0; b < B; ++b) {
        const int len_b = chunk_lo + b + 1;   // row b's causal prefix
        // Producer shape: nb = len_b (bounded scoring, vacuous causality).
        run_row(b, /*nb=*/len_b, /*qpos=*/len_b - 1, didx, dlen);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        // Snapshot row b's scores before the shared scratch is reused: the
        // selection reference below is the top-k of the EXACT floats the
        // topk kernel read (removes fp32 accumulation-order ambiguity from
        // the selection compare; the scores themselves are CPU-checked).
        gpu_scores[b] = download<float>(dscores, len_b);
        // Defense-in-depth shape: score ALL stored blocks (incl. the chunk's
        // future keys), rely on the top-k causal cutoff to exclude them.
        run_row(b, /*nb=*/N, /*qpos=*/len_b - 1, didx2, dlen2);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    auto idx = download<int>(didx, size_t(B) * ITK);
    auto len = download<int>(dlen, B);
    auto idx2 = download<int>(didx2, size_t(B) * ITK);
    auto len2 = download<int>(dlen2, B);

    for (int b = 0; b < B; ++b) {
        const int len_b = chunk_lo + b + 1;

        // 1a. Scores match the CPU lightning reference (fp8-dequant dot ×
        //     per-position scale, ReLU, weighted head sum) over [0, len_b).
        for (int n = 0; n < len_b; ++n) {
            float total = 0.0f;
            for (int h = 0; h < kNIH; ++h) {
                float dot = 0.0f;
                for (int d = 0; d < kIHD; ++d)
                    dot += q_vals[(size_t(b) * kNIH + h) * kIHD + d]
                         * k_vals[size_t(n) * kIHD + d];
                dot *= scales[n];
                total += std::max(dot, 0.0f) * w_all[size_t(b) * kNIH + h];
            }
            EXPECT_NEAR(gpu_scores[b][n], total,
                        5e-3f * (1.0f + std::abs(total)))
                << "row " << b << " score block " << n;
        }

        // 1b. Selection = top-ITK of the scores, ascending-sorted, padded.
        std::vector<int> order(len_b);
        for (int n = 0; n < len_b; ++n) order[n] = n;
        const int k_eff = std::min(ITK, len_b);
        std::partial_sort(order.begin(), order.begin() + k_eff, order.end(),
                          [&](int a, int c) {
                              return gpu_scores[b][a] > gpu_scores[b][c];
                          });
        std::vector<int> expect(order.begin(), order.begin() + k_eff);
        std::sort(expect.begin(), expect.end());

        EXPECT_EQ(len[b], k_eff) << "row " << b;
        for (int i = 0; i < k_eff; ++i) {
            EXPECT_EQ(idx[size_t(b) * ITK + i], expect[i])
                << "row " << b << " slot " << i;
            // 2. Causality: never a position at/past the row's own.
            EXPECT_LT(idx[size_t(b) * ITK + i], len_b)
                << "row " << b << " selected a FUTURE position";
        }
        for (int i = k_eff; i < ITK; ++i)
            EXPECT_EQ(idx[size_t(b) * ITK + i], -1)
                << "row " << b << " pad slot " << i;

        // 3. Bounded-nb and causal-cutoff runs agree exactly.
        EXPECT_EQ(len2[b], len[b]) << "row " << b;
        for (int i = 0; i < ITK; ++i)
            EXPECT_EQ(idx2[size_t(b) * ITK + i], idx[size_t(b) * ITK + i])
                << "row " << b << " slot " << i
                << " (nb-bounded vs qpos-bounded)";
    }

    for (auto* p : page_ptrs) cudaFree(p);
    cudaFree(dq); cudaFree(dw); cudaFree(dend);
    cudaFree(dscores); cudaFree(dtopk_s);
    cudaFree(didx); cudaFree(dlen); cudaFree(didx2); cudaFree(dlen2);
}
