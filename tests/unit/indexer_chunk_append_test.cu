// TD-GLM-INDEXER-PREFILL: chunk-append vs one-by-one equivalence of the DSA
// indexer-K producer chain.
//
// The prefill chunk appender (DcpExecutor::append_indexer_chunk) runs the
// decode producer's K half — indexer LayerNorm(w+bias) → RoPE (per-row
// position from the seqlens array) → Hadamard → FP8 quant-append — BATCHED
// over all chunk rows, with per-page grouped append launches. The decode
// producer appends the same positions ONE BY ONE (B==1 per step). Every
// kernel in the chain operates row-independently (one CTA per row, block dim
// a function of dim only), so the stored FP8 keys and F32 scales must be
// BIT-EQUAL between the two paths — same kernels, same math, only launch
// batching differs. Covers both storage layouts: paged
// ([page_tokens × head_dim FP8 | page_tokens F32] with slot_bias
// = −1 − page_start) and the contiguous executor arena (slot_bias = −1).

#include "compute/cuda_sm120_device_backend.h"
#include "core/attention_device.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstddef>
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

struct Fixture {
    // Non-multiple sizing exercises a partial last page (21 = 8 + 8 + 5).
    static constexpr int N = 21;        // chunk positions 0..N-1
    static constexpr int PT = 8;        // page tokens
    static constexpr int IHD = 128;     // index head dim (FWHT power of two)
    static constexpr int DR = 16;       // rope width (adjacent-pair pairs)
    static constexpr float kEps = 1e-6f;
    int n_pages = (N + PT - 1) / PT;

    std::vector<__nv_bfloat16> k_rows;  // [N, IHD] post-GEMM producer input
    std::vector<__nv_bfloat16> ln_w, ln_b;
    std::vector<float> rope_table;      // [N][DR] cos|sin halves
    std::vector<int> seqlens;           // seqlens[b] = b + 1 (pos = b)

    __nv_bfloat16* d_k = nullptr;       // scratch, rewritten per path
    __nv_bfloat16* d_ln_w = nullptr;
    __nv_bfloat16* d_ln_b = nullptr;
    float* d_rope = nullptr;
    int* d_seqlens = nullptr;

    void init() {
        std::mt19937 rng(23);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        k_rows.resize(size_t(N) * IHD);
        for (auto& v : k_rows) v = __float2bfloat16(dist(rng));
        ln_w.resize(IHD);
        ln_b.resize(IHD);
        for (auto& v : ln_w) v = __float2bfloat16(1.0f + 0.1f * dist(rng));
        for (auto& v : ln_b) v = __float2bfloat16(0.1f * dist(rng));
        rope_table.resize(size_t(N) * DR);
        for (int p = 0; p < N; ++p) {
            for (int i = 0; i < DR / 2; ++i) {
                const float ang = p / std::pow(10000.0f, 2.0f * i / DR);
                rope_table[size_t(p) * DR + i] = std::cos(ang);
                rope_table[size_t(p) * DR + DR / 2 + i] = std::sin(ang);
            }
        }
        seqlens.resize(N);
        for (int b = 0; b < N; ++b) seqlens[b] = b + 1;

        ASSERT_EQ(cudaMalloc(&d_k, size_t(N) * IHD * sizeof(__nv_bfloat16)),
                  cudaSuccess);
        d_ln_w = upload(ln_w);
        d_ln_b = upload(ln_b);
        d_rope = upload(rope_table);
        d_seqlens = upload(seqlens);
    }

    void destroy() {
        cudaFree(d_k);
        cudaFree(d_ln_w);
        cudaFree(d_ln_b);
        cudaFree(d_rope);
        cudaFree(d_seqlens);
    }

    // Reset the producer-input scratch to the pristine post-GEMM rows.
    void reset_k() {
        ASSERT_EQ(cudaMemcpy(d_k, k_rows.data(),
                             size_t(N) * IHD * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice), cudaSuccess);
    }

    // Decode-producer chain over `rows` rows starting at row `b0`, using the
    // per-row seqlens slice — exactly produce_sparse_indices' K half /
    // append_indexer_chunk's batched half depending on `rows`.
    void run_chain(lc::CudaSm120DeviceBackend& dev, int b0, int rows) {
        auto* x = d_k + size_t(b0) * IHD;
        dev.indexer_layernorm_bias(x, d_ln_w, d_ln_b, rows, IHD, kEps, nullptr);
        lc::RopeRotateParams rr{};
        rr.x = x;
        rr.seqlens_k = d_seqlens + b0;
        rr.cos_sin = d_rope;
        rr.num_tokens = rows;
        rr.rows_per_token = 1;
        rr.row_stride = IHD;
        rr.d_rope = DR;
        rr.max_pos = N;
        dev.rope_rotate(rr, nullptr);
        dev.indexer_hadamard(x, rows, IHD, nullptr);
    }
};

}  // namespace

// Paged storage: one-by-one (decode) vs chunk-batched (prefill) appends into
// two independent page sets — stored FP8 keys + F32 scales bit-equal.
TEST(IndexerChunkAppend, PagedChunkEqualsOneByOne) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    Fixture f;
    f.init();
    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();

    const size_t page_bytes =
        size_t(f.PT) * f.IHD + size_t(f.PT) * sizeof(float);
    std::vector<void*> pages_ref(f.n_pages), pages_chunk(f.n_pages);
    for (int p = 0; p < f.n_pages; ++p) {
        ASSERT_EQ(cudaMalloc(&pages_ref[p], page_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&pages_chunk[p], page_bytes), cudaSuccess);
        ASSERT_EQ(cudaMemset(pages_ref[p], 0, page_bytes), cudaSuccess);
        ASSERT_EQ(cudaMemset(pages_chunk[p], 0, page_bytes), cudaSuccess);
    }

    // Reference: decode-producer path — per-position chain + append
    // (num_tokens=1), position b into page b/PT at slot_bias −1 − pg*PT.
    f.reset_k();
    for (int b = 0; b < f.N; ++b) {
        f.run_chain(dev, b, 1);
        const int pg = b / f.PT;
        auto* base = static_cast<std::byte*>(pages_ref[pg]);
        dev.indexer_k_quant_append(
            f.d_k + size_t(b) * f.IHD, f.d_seqlens + b,
            base, base + size_t(f.PT) * f.IHD,
            /*num_tokens=*/1, f.IHD, /*slot_bias=*/-1 - pg * f.PT, nullptr);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Chunk path: batched chain over all rows + per-page grouped appends
    // (append_indexer_chunk's launch partitioning).
    f.reset_k();
    f.run_chain(dev, 0, f.N);
    for (int b = 0; b < f.N; ) {
        const int pg = b / f.PT;
        int e = b + 1;
        while (e < f.N && e / f.PT == pg) ++e;
        auto* base = static_cast<std::byte*>(pages_chunk[pg]);
        dev.indexer_k_quant_append(
            f.d_k + size_t(b) * f.IHD, f.d_seqlens + b,
            base, base + size_t(f.PT) * f.IHD,
            /*num_tokens=*/e - b, f.IHD, /*slot_bias=*/-1 - pg * f.PT,
            nullptr);
        b = e;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (int p = 0; p < f.n_pages; ++p) {
        const int rows = std::min(f.PT, f.N - p * f.PT);
        auto ref_k = download<uint8_t>(pages_ref[p], size_t(rows) * f.IHD);
        auto chk_k = download<uint8_t>(pages_chunk[p], size_t(rows) * f.IHD);
        EXPECT_EQ(std::memcmp(ref_k.data(), chk_k.data(), ref_k.size()), 0)
            << "FP8 keys differ in page " << p;
        auto ref_s = download<float>(
            static_cast<std::byte*>(pages_ref[p]) + size_t(f.PT) * f.IHD,
            rows);
        auto chk_s = download<float>(
            static_cast<std::byte*>(pages_chunk[p]) + size_t(f.PT) * f.IHD,
            rows);
        for (int i = 0; i < rows; ++i)
            EXPECT_EQ(ref_s[i], chk_s[i])
                << "scale differs at page " << p << " row " << i;
    }

    for (int p = 0; p < f.n_pages; ++p) {
        cudaFree(pages_ref[p]);
        cudaFree(pages_chunk[p]);
    }
    f.destroy();
}

// INV-DSA-REWIND: position-keyed OVERWRITE-REWIND. Re-appending positions
// [p, p+k) with NEW content into already-written pages must leave the store
// bit-equal to a fresh append of the final content — the append slot is a
// pure function of the row's position (slot = seqlens[t] + slot_bias), so a
// speculative-verify re-feed overwrites exactly its own rows and no others.
// This is the storage half of the LS_INDEXER_REWIND coverage-guard blessing
// (dispatcher state machine covered by indexer_coverage_guard_test).
TEST(IndexerChunkAppend, PagedRewindOverwriteEqualsFreshAppend) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    Fixture f;
    f.init();
    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();

    // Rewind window [p0, p0+K): crosses the page-1/page-2 boundary (PT=8).
    constexpr int P0 = 6;
    constexpr int K = 5;

    // Alternative raw content (the re-fed tokens' post-GEMM rows).
    std::vector<__nv_bfloat16> alt_rows(size_t(K) * f.IHD);
    {
        std::mt19937 rng(97);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : alt_rows) v = __float2bfloat16(dist(rng));
    }

    const size_t page_bytes =
        size_t(f.PT) * f.IHD + size_t(f.PT) * sizeof(float);
    std::vector<void*> pages_ref(f.n_pages), pages_rw(f.n_pages);
    for (int p = 0; p < f.n_pages; ++p) {
        ASSERT_EQ(cudaMalloc(&pages_ref[p], page_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&pages_rw[p], page_bytes), cudaSuccess);
        ASSERT_EQ(cudaMemset(pages_ref[p], 0, page_bytes), cudaSuccess);
        ASSERT_EQ(cudaMemset(pages_rw[p], 0, page_bytes), cudaSuccess);
    }

    auto append_rows = [&](std::vector<void*>& pages, int b0, int rows) {
        for (int b = b0; b < b0 + rows; ) {
            const int pg = b / f.PT;
            int e = b + 1;
            while (e < b0 + rows && e / f.PT == pg) ++e;
            auto* base = static_cast<std::byte*>(pages[pg]);
            dev.indexer_k_quant_append(
                f.d_k + size_t(b) * f.IHD, f.d_seqlens + b,
                base, base + size_t(f.PT) * f.IHD,
                /*num_tokens=*/e - b, f.IHD, /*slot_bias=*/-1 - pg * f.PT,
                nullptr);
            b = e;
        }
    };

    // Rewind path: full append of content A, then the re-feed — raw
    // alternative rows for [P0, P0+K) through the same chain + append,
    // overwriting in place.
    f.reset_k();
    f.run_chain(dev, 0, f.N);
    append_rows(pages_rw, 0, f.N);
    ASSERT_EQ(cudaMemcpy(f.d_k + size_t(P0) * f.IHD, alt_rows.data(),
                         alt_rows.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice), cudaSuccess);
    f.run_chain(dev, P0, K);
    append_rows(pages_rw, P0, K);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Reference: ONE fresh append of the final content (A outside the
    // window, the alternative rows inside).
    f.reset_k();
    ASSERT_EQ(cudaMemcpy(f.d_k + size_t(P0) * f.IHD, alt_rows.data(),
                         alt_rows.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice), cudaSuccess);
    f.run_chain(dev, 0, f.N);
    append_rows(pages_ref, 0, f.N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (int p = 0; p < f.n_pages; ++p) {
        const int rows = std::min(f.PT, f.N - p * f.PT);
        auto ref_k = download<uint8_t>(pages_ref[p], size_t(rows) * f.IHD);
        auto rw_k = download<uint8_t>(pages_rw[p], size_t(rows) * f.IHD);
        EXPECT_EQ(std::memcmp(ref_k.data(), rw_k.data(), ref_k.size()), 0)
            << "FP8 keys differ after rewind overwrite in page " << p;
        auto ref_s = download<float>(
            static_cast<std::byte*>(pages_ref[p]) + size_t(f.PT) * f.IHD,
            rows);
        auto rw_s = download<float>(
            static_cast<std::byte*>(pages_rw[p]) + size_t(f.PT) * f.IHD,
            rows);
        for (int i = 0; i < rows; ++i)
            EXPECT_EQ(ref_s[i], rw_s[i])
                << "scale differs after rewind at page " << p << " row " << i;
    }

    for (int p = 0; p < f.n_pages; ++p) {
        cudaFree(pages_ref[p]);
        cudaFree(pages_rw[p]);
    }
    f.destroy();
}

// Arena storage (contiguous slot, slot_bias = −1): one-by-one vs one
// whole-chunk launch — bit-equal.
TEST(IndexerChunkAppend, ArenaChunkEqualsOneByOne) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    Fixture f;
    f.init();
    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();

    void* k_ref = nullptr;
    void* k_chunk = nullptr;
    float* s_ref = nullptr;
    float* s_chunk = nullptr;
    ASSERT_EQ(cudaMalloc(&k_ref, size_t(f.N) * f.IHD), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&k_chunk, size_t(f.N) * f.IHD), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&s_ref, f.N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&s_chunk, f.N * sizeof(float)), cudaSuccess);

    f.reset_k();
    for (int b = 0; b < f.N; ++b) {
        f.run_chain(dev, b, 1);
        dev.indexer_k_quant_append(
            f.d_k + size_t(b) * f.IHD, f.d_seqlens + b, k_ref, s_ref,
            /*num_tokens=*/1, f.IHD, /*slot_bias=*/-1, nullptr);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    f.reset_k();
    f.run_chain(dev, 0, f.N);
    dev.indexer_k_quant_append(
        f.d_k, f.d_seqlens, k_chunk, s_chunk,
        /*num_tokens=*/f.N, f.IHD, /*slot_bias=*/-1, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto ref_k = download<uint8_t>(k_ref, size_t(f.N) * f.IHD);
    auto chk_k = download<uint8_t>(k_chunk, size_t(f.N) * f.IHD);
    EXPECT_EQ(std::memcmp(ref_k.data(), chk_k.data(), ref_k.size()), 0)
        << "arena FP8 keys differ";
    auto ref_s = download<float>(s_ref, f.N);
    auto chk_s = download<float>(s_chunk, f.N);
    for (int i = 0; i < f.N; ++i)
        EXPECT_EQ(ref_s[i], chk_s[i]) << "arena scale differs at row " << i;

    cudaFree(k_ref);
    cudaFree(k_chunk);
    cudaFree(s_ref);
    cudaFree(s_chunk);
    f.destroy();
}
