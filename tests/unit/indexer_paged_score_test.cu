// TD-GLM-INDEXER-PAGED: paged-vs-contiguous equivalence of the DSA indexer
// score+topk backend path.
//
// The paged mode stores K in coarse pool pages ([page_tokens × head_dim] FP8
// followed by [page_tokens] F32 scales) and launches one score kernel per
// page; the contiguous mode is the executor-arena layout. Same K content ⇒
// the top-k indices, effective lengths, and the raw score array must match
// EXACTLY (identical kernel, identical inputs, only launch partitioning
// differs — FP32 outputs are bit-equal per block).

#include "compute/cuda_sm120_device_backend.h"
#include "core/attention_device.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstring>
#include <random>
#include <vector>

namespace lc = layerstorm::compute;
namespace cfg = layerstorm::config;

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
template <typename T>
std::vector<T> download(const void* d, size_t n) {
    std::vector<T> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return h;
}
}  // namespace

TEST(IndexerPagedScore, PagedEqualsContiguous) {
    REQUIRES_GPU();
    // Ordering hygiene: a prior multi-GPU test may leave another device
    // current (or a sticky error) — all allocations below must land on the
    // backend's device.
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();
    // Non-multiple sizing exercises the partial last page.
    const int NB = 21, PT = 8, NIH = 8, IHD = 128, ITK = 6;
    const int n_pages = (NB + PT - 1) / PT;  // 3 (8, 8, 5)

    std::mt19937 rng(17);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Random FP8 K + per-block scales + query + score weights.
    std::vector<__nv_fp8_e4m3> k(NB * IHD);
    for (auto& v : k) v = __nv_fp8_e4m3(dist(rng));
    std::vector<float> scales(NB);
    for (auto& s : scales) s = 0.01f + std::abs(dist(rng)) * 0.05f;
    std::vector<__nv_bfloat16> q(NIH * IHD);
    for (auto& v : q) v = __float2bfloat16(dist(rng));
    std::vector<float> w(NIH);
    for (auto& v : w) v = dist(rng) / 8.0f;
    std::vector<int> endpoints(NB);
    for (int i = 0; i < NB; ++i) endpoints[i] = i;

    auto* dk = upload(k);
    auto* dscales = upload(scales);
    auto* dq = upload(q);
    auto* dw = upload(w);
    auto* dend = upload(endpoints);

    // Paged copies: page = [PT×IHD FP8 | PT F32], last page partially filled.
    std::vector<void*> page_ptrs(n_pages);
    const size_t page_bytes = size_t(PT) * IHD + size_t(PT) * sizeof(float);
    for (int p = 0; p < n_pages; ++p) {
        ASSERT_EQ(cudaMalloc(&page_ptrs[p], page_bytes), cudaSuccess);
        const int rows = std::min(PT, NB - p * PT);
        ASSERT_EQ(cudaMemcpy(page_ptrs[p], k.data() + size_t(p) * PT * IHD,
                             size_t(rows) * IHD, cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(static_cast<std::byte*>(page_ptrs[p])
                                 + size_t(PT) * IHD,
                             scales.data() + size_t(p) * PT,
                             size_t(rows) * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    // Shared output/scratch buffers, run twice.
    void *dscores, *dtopk_s, *didx, *dlen;
    ASSERT_EQ(cudaMalloc(&dscores, NB * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dtopk_s, ITK * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&didx, ITK * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dlen, sizeof(int)), cudaSuccess);

    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();

    lc::IndexerScoreTopkArgs a{};
    a.q_all = dq; a.score_proj_all = dw; a.block_endpoints = dend;
    a.scores_scratch = dscores; a.topk_scores_scratch = dtopk_s;
    a.sparse_indices_out = didx; a.topk_lengths_out = dlen;
    a.num_tokens = 1; a.num_blocks = NB; a.n_heads = NIH; a.head_dim = IHD;
    a.topk = ITK;
    a.query_position_base = NB - 1;  // all blocks causal-eligible

    // 1) Contiguous reference.
    a.k_cache = dk; a.k_scales = dscales;
    dev.indexer_score_topk(a, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto ref_scores = download<float>(dscores, NB);
    auto ref_idx = download<int>(didx, ITK);
    auto ref_len = download<int>(dlen, 1);

    // 2) Paged.
    ASSERT_EQ(cudaMemset(dscores, 0, NB * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemset(didx, 0xFF, ITK * sizeof(int)), cudaSuccess);
    a.k_cache = nullptr; a.k_scales = nullptr;
    a.k_pages = const_cast<const void* const*>(page_ptrs.data());
    a.num_k_pages = n_pages; a.page_tokens = PT;
    dev.indexer_score_topk(a, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto got_scores = download<float>(dscores, NB);
    auto got_idx = download<int>(didx, ITK);
    auto got_len = download<int>(dlen, 1);

    // Identical kernel over identical rows ⇒ bit-equal scores + same top-k.
    for (int i = 0; i < NB; ++i)
        EXPECT_EQ(ref_scores[i], got_scores[i]) << "score block " << i;
    EXPECT_EQ(ref_len[0], got_len[0]);
    for (int i = 0; i < ITK; ++i)
        EXPECT_EQ(ref_idx[i], got_idx[i]) << "topk slot " << i;

    for (auto* p : page_ptrs) cudaFree(p);
    cudaFree(dk); cudaFree(dscales); cudaFree(dq); cudaFree(dw); cudaFree(dend);
    cudaFree(dscores); cudaFree(dtopk_s); cudaFree(didx); cudaFree(dlen);
}

// TD-GLM-INDEXER-BATCH: two batch entries = two independent sequences with
// different lengths and their OWN page sets — exactly the per-entry loop the
// producer issues at B>1. Each entry's row of sparse_indices/topk_lengths
// must equal the same sequence scored alone (independence), with the shared
// scores scratch reused sequentially between calls.
TEST(IndexerPagedScore, TwoSequencesIndependent) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();
    const int PT = 8, NIH = 8, IHD = 128, ITK = 4;
    const int len[2] = {13, 6};  // seq 0: 2 pages, seq 1: 1 page

    std::mt19937 rng(29);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();

    // Per-seq pages, queries, weights.
    std::vector<std::vector<void*>> pages(2);
    std::vector<__nv_bfloat16*> dq(2);
    std::vector<float*> dw(2);
    const size_t page_bytes = size_t(PT) * IHD + size_t(PT) * sizeof(float);
    for (int s = 0; s < 2; ++s) {
        const int np = (len[s] + PT - 1) / PT;
        pages[s].resize(np);
        std::vector<__nv_fp8_e4m3> k(len[s] * IHD);
        for (auto& v : k) v = __nv_fp8_e4m3(dist(rng));
        std::vector<float> sc(len[s]);
        for (auto& v : sc) v = 0.01f + std::abs(dist(rng)) * 0.05f;
        for (int p = 0; p < np; ++p) {
            ASSERT_EQ(cudaMalloc(&pages[s][p], page_bytes), cudaSuccess);
            const int rows = std::min(PT, len[s] - p * PT);
            ASSERT_EQ(cudaMemcpy(pages[s][p], k.data() + size_t(p) * PT * IHD,
                                 size_t(rows) * IHD, cudaMemcpyHostToDevice),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpy(static_cast<std::byte*>(pages[s][p])
                                     + size_t(PT) * IHD,
                                 sc.data() + size_t(p) * PT,
                                 size_t(rows) * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
        }
        std::vector<__nv_bfloat16> q(NIH * IHD);
        for (auto& v : q) v = __float2bfloat16(dist(rng));
        std::vector<float> w(NIH);
        for (auto& v : w) v = dist(rng) / 8.0f;
        dq[s] = upload(q);
        dw[s] = upload(w);
    }

    const int max_len = std::max(len[0], len[1]);
    std::vector<int> endpoints(max_len);
    for (int i = 0; i < max_len; ++i) endpoints[i] = i;
    auto* dend = upload(endpoints);
    void *dscores, *dtopk_s, *didx, *dlen;
    ASSERT_EQ(cudaMalloc(&dscores, max_len * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dtopk_s, ITK * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&didx, 2 * ITK * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dlen, 2 * sizeof(int)), cudaSuccess);

    auto run_entry = [&](int s, int out_row) {
        lc::IndexerScoreTopkArgs a{};
        a.q_all = dq[s]; a.score_proj_all = dw[s];
        a.block_endpoints = dend;
        a.scores_scratch = dscores; a.topk_scores_scratch = dtopk_s;
        a.sparse_indices_out = static_cast<int*>(didx) + out_row * ITK;
        a.topk_lengths_out = static_cast<int*>(dlen) + out_row;
        a.num_tokens = 1; a.num_blocks = len[s];
        a.n_heads = NIH; a.head_dim = IHD; a.topk = ITK;
        a.query_position_base = len[s] - 1;
        a.k_pages = const_cast<const void* const*>(pages[s].data());
        a.num_k_pages = static_cast<int>(pages[s].size());
        a.page_tokens = PT;
        dev.indexer_score_topk(a, nullptr);
    };

    // Batch pass: entry 0 then entry 1 (the executor's loop order).
    run_entry(0, 0);
    run_entry(1, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto batch_idx = download<int>(didx, 2 * ITK);
    auto batch_len = download<int>(dlen, 2);

    // Solo reference passes into row 0.
    for (int s = 0; s < 2; ++s) {
        ASSERT_EQ(cudaMemset(didx, 0xFF, 2 * ITK * sizeof(int)), cudaSuccess);
        run_entry(s, 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        auto solo_idx = download<int>(didx, ITK);
        auto solo_len = download<int>(dlen, 1);
        EXPECT_EQ(solo_len[0], batch_len[s]) << "seq " << s;
        for (int i = 0; i < ITK; ++i)
            EXPECT_EQ(solo_idx[i], batch_idx[s * ITK + i])
                << "seq " << s << " slot " << i;
    }

    for (int s = 0; s < 2; ++s) {
        for (auto* p : pages[s]) cudaFree(p);
        cudaFree(dq[s]); cudaFree(dw[s]);
    }
    cudaFree(dend); cudaFree(dscores); cudaFree(dtopk_s);
    cudaFree(didx); cudaFree(dlen);
}
