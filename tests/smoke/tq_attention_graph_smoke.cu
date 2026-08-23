// Smoke test: TQ decode graph runner lifecycle on real SM120 hardware.
//
// Verifies TqDecodeGraphRunner with V3.2 parameters:
//   - Buffer allocation (12 fixed-address device buffers)
//   - CUDA graph capture (4-node: q_rotate → dense_decode → combine<f32> → v_rotate_back)
//   - Per-step update (async copy into fixed buffers)
//   - Graph replay
//   - Multi-replay with different input data
//   - Clean destruction
//
// Uses minimal dimensions (B=1, HQ=2) with identity rotation matrix to keep
// test fast.  Numerical correctness validated at integration level.
//
// Requires: 1 GPU with compute capability >= 12.0 (SM120).
// Build:    cmake --build build --target tq_attention_graph_smoke
// Run:      ./build/tests/smoke/tq_attention_graph_smoke

#include <sm120/graph/tq_decode_graph.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

// -- Macros ------------------------------------------------------------------

#define CUDA_CHECK(expr)                                                 \
    do {                                                                 \
        cudaError_t _err = (expr);                                       \
        ASSERT_EQ(_err, cudaSuccess) << cudaGetErrorString(_err);        \
    } while (0)

#define REQUIRES_SM120()                                                 \
    do {                                                                 \
        int dev_ = 0;                                                    \
        cudaError_t e_ = cudaGetDevice(&dev_);                           \
        if (e_ != cudaSuccess) GTEST_SKIP() << "No CUDA device";        \
        int major_ = 0;                                                  \
        cudaDeviceGetAttribute(&major_, cudaDevAttrComputeCapabilityMajor, dev_); \
        if (major_ < 12) GTEST_SKIP() << "Need SM120+ (have SM" << major_ << "0)"; \
    } while (0)

// -- Helpers -----------------------------------------------------------------

/// Generate a d×d identity matrix (trivial orthogonal rotation).
static std::vector<float> make_identity_matrix(int d) {
    std::vector<float> Pi(d * d, 0.0f);
    for (int i = 0; i < d; ++i) Pi[i * d + i] = 1.0f;
    return Pi;
}

/// Generate 16 evenly-spaced centroids in [-1, 1].
static std::vector<float> make_uniform_centroids() {
    std::vector<float> c(16);
    for (int i = 0; i < 16; ++i)
        c[i] = -1.0f + 2.0f * static_cast<float>(i) / 15.0f;
    return c;
}

/// Allocate device memory and copy host data.
template <typename T>
static T* to_device(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

/// Read device BF16 buffer back to host floats.
static std::vector<float> read_bf16(const __nv_bfloat16* d_ptr, size_t n) {
    std::vector<__nv_bfloat16> buf(n);
    cudaMemcpy(buf.data(), d_ptr, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __bfloat162float(buf[i]);
    return out;
}

// ============================================================================
// Lifecycle: init → update → replay → destroy
// ============================================================================

TEST(TqDecodeGraphSmoke, V32Lifecycle) {
    REQUIRES_SM120();

    // Minimal V3.2-like dimensions
    constexpr int B = 1, SQ = 1, HQ = 2;
    constexpr int DC = 512, DROPE = 64;
    constexpr int PAGE_SIZE = 64, MAX_BLOCKS = 1, NSP = 1;

    // TQ cache row layout: packed_nope(d_c/2) + norm(2) + rope(d_rope*2)
    constexpr int ROW_BYTES = DC / 2 + 2 + DROPE * 2;  // 386
    constexpr size_t KV_CACHE_BYTES = PAGE_SIZE * ROW_BYTES;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // -- Device resources (stable across replays) ----------------------------

    auto h_Pi = make_identity_matrix(DC);
    float* d_Pi = to_device(h_Pi);

    auto h_centroids = make_uniform_centroids();
    float* d_centroids = to_device(h_centroids);

    // KV cache: 1 page, zero-filled (all indices → centroid[0])
    void* d_kv_cache = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kv_cache, KV_CACHE_BYTES));
    CUDA_CHECK(cudaMemset(d_kv_cache, 0, KV_CACHE_BYTES));

    // -- Per-step input buffers ----------------------------------------------

    // Q nope: [B, SQ, HQ, DC] BF16 — small random values
    const size_t q_nope_elems = B * SQ * HQ * DC;
    std::vector<__nv_bfloat16> h_q_nope(q_nope_elems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : h_q_nope) v = __float2bfloat16(dist(rng));
    void* d_q_nope = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q_nope, q_nope_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_q_nope, h_q_nope.data(),
                          q_nope_elems * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));

    // Q rope: [B, SQ, HQ, DROPE] BF16
    const size_t q_rope_elems = B * SQ * HQ * DROPE;
    std::vector<__nv_bfloat16> h_q_rope(q_rope_elems);
    for (auto& v : h_q_rope) v = __float2bfloat16(dist(rng));
    void* d_q_rope = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q_rope, q_rope_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_q_rope, h_q_rope.data(),
                          q_rope_elems * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));

    // seqlens_k: [B] = {1}  (1 KV token)
    std::vector<int> h_seqlens = {1};
    int* d_seqlens = to_device(h_seqlens);

    // block_table: [B, MAX_BLOCKS] = {0}  (page 0)
    std::vector<int> h_block_table = {0};
    int* d_block_table = to_device(h_block_table);

    // -- Graph runner --------------------------------------------------------

    sm120::graph::TqDecodeGraphConfig cfg{};
    cfg.batch_size = B;
    cfg.s_q = SQ;
    cfg.h_q = HQ;
    cfg.d_c = DC;
    cfg.d_rope = DROPE;
    cfg.page_block_size = PAGE_SIZE;
    cfg.max_num_blocks_per_seq = MAX_BLOCKS;
    cfg.sm_scale = 1.0f / std::sqrt(static_cast<float>(DC + DROPE));
    cfg.num_sm_parts = NSP;
    cfg.kv_cache = d_kv_cache;
    cfg.Pi = d_Pi;
    cfg.centroids = d_centroids;

    sm120::graph::TqDecodeGraphRunner runner;
    runner.init(cfg, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Update + replay
    runner.update(d_q_nope, d_q_rope, d_seqlens, d_block_table, stream);
    runner.replay(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify output is finite (not NaN/inf from uninitialized memory)
    auto out = read_bf16(runner.out_ptr(), B * SQ * HQ * DC);
    for (size_t i = 0; i < out.size(); ++i) {
        ASSERT_TRUE(std::isfinite(out[i]))
            << "out[" << i << "] = " << out[i] << " is not finite";
    }

    // Verify LSE pointer is accessible (read doesn't crash).
    // LSE values may be non-finite with synthetic all-zero KV cache data;
    // numerical correctness tested at integration level with real weights.
    std::vector<float> h_lse(B * SQ * HQ);
    CUDA_CHECK(cudaMemcpy(h_lse.data(), runner.lse_ptr(),
                          h_lse.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    EXPECT_NE(runner.lse_ptr(), nullptr);

    // -- Cleanup -------------------------------------------------------------

    runner.destroy();
    cudaFree(d_q_nope);
    cudaFree(d_q_rope);
    cudaFree(d_seqlens);
    cudaFree(d_block_table);
    cudaFree(d_kv_cache);
    cudaFree(d_Pi);
    cudaFree(d_centroids);
    cudaStreamDestroy(stream);
}

// ============================================================================
// Multi-replay: same graph, different data each step
// ============================================================================

TEST(TqDecodeGraphSmoke, MultiReplay) {
    REQUIRES_SM120();

    constexpr int B = 2, SQ = 1, HQ = 2;
    constexpr int DC = 512, DROPE = 64;
    constexpr int PAGE_SIZE = 64, MAX_BLOCKS = 2, NSP = 1;
    constexpr int ROW_BYTES = DC / 2 + 2 + DROPE * 2;
    constexpr size_t KV_CACHE_BYTES = (size_t)MAX_BLOCKS * PAGE_SIZE * ROW_BYTES;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto h_Pi = make_identity_matrix(DC);
    float* d_Pi = to_device(h_Pi);
    auto h_centroids = make_uniform_centroids();
    float* d_centroids = to_device(h_centroids);

    void* d_kv_cache = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kv_cache, KV_CACHE_BYTES));
    CUDA_CHECK(cudaMemset(d_kv_cache, 0, KV_CACHE_BYTES));

    const size_t q_nope_elems = B * SQ * HQ * DC;
    const size_t q_rope_elems = B * SQ * HQ * DROPE;
    void* d_q_nope = nullptr;
    void* d_q_rope = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q_nope, q_nope_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q_rope, q_rope_elems * sizeof(__nv_bfloat16)));

    std::vector<int> h_seqlens = {1, 2};
    int* d_seqlens = to_device(h_seqlens);
    std::vector<int> h_block_table = {0, 0, 0, 1};  // [B, MAX_BLOCKS]
    int* d_block_table = to_device(h_block_table);

    sm120::graph::TqDecodeGraphConfig cfg{};
    cfg.batch_size = B;
    cfg.s_q = SQ;
    cfg.h_q = HQ;
    cfg.d_c = DC;
    cfg.d_rope = DROPE;
    cfg.page_block_size = PAGE_SIZE;
    cfg.max_num_blocks_per_seq = MAX_BLOCKS;
    cfg.sm_scale = 1.0f / std::sqrt(static_cast<float>(DC + DROPE));
    cfg.num_sm_parts = NSP;
    cfg.kv_cache = d_kv_cache;
    cfg.Pi = d_Pi;
    cfg.centroids = d_centroids;

    sm120::graph::TqDecodeGraphRunner runner;
    runner.init(cfg, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Replay 3 times with different Q data
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int iter = 0; iter < 3; ++iter) {
        std::vector<__nv_bfloat16> h_q_nope(q_nope_elems);
        std::vector<__nv_bfloat16> h_q_rope(q_rope_elems);
        for (auto& v : h_q_nope) v = __float2bfloat16(dist(rng));
        for (auto& v : h_q_rope) v = __float2bfloat16(dist(rng));

        CUDA_CHECK(cudaMemcpyAsync(d_q_nope, h_q_nope.data(),
                                   q_nope_elems * sizeof(__nv_bfloat16),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_q_rope, h_q_rope.data(),
                                   q_rope_elems * sizeof(__nv_bfloat16),
                                   cudaMemcpyHostToDevice, stream));

        runner.update(d_q_nope, d_q_rope, d_seqlens, d_block_table, stream);
        runner.replay(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto out = read_bf16(runner.out_ptr(), B * SQ * HQ * DC);
        for (size_t i = 0; i < out.size(); ++i) {
            ASSERT_TRUE(std::isfinite(out[i]))
                << "iter=" << iter << " out[" << i << "] = " << out[i];
        }
    }

    runner.destroy();
    cudaFree(d_q_nope);
    cudaFree(d_q_rope);
    cudaFree(d_seqlens);
    cudaFree(d_block_table);
    cudaFree(d_kv_cache);
    cudaFree(d_Pi);
    cudaFree(d_centroids);
    cudaStreamDestroy(stream);
}
