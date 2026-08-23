// Unit tests for DCP LSE correction kernel.
//
// Tests: single-rank passthrough, multi-rank correction, numerical stability.

#include "compute/kernels/attention/dcp_lse_correct.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;

// ── CUDA error checking ─────────────────────────────────────────────────────

#define CUDA_CHECK(expr)                                                 \
    do {                                                                 \
        cudaError_t _err = (expr);                                       \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_err); \
    } while (0)

// ── Host-side BF16 conversion ────────────────────────────────────────────────

static uint16_t float_to_bf16_bits(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint32_t lsb = (f >> 16) & 1;
    f += 0x7FFF + lsb;
    return static_cast<uint16_t>(f >> 16);
}

static float bf16_bits_to_float(uint16_t bits) {
    uint32_t f = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

// ── CPU reference for LSE correction ─────────────────────────────────────────

static void dcp_lse_correct_ref(
    std::vector<float>& output_f32,  // [B*H*D] — corrected in-place
    const std::vector<float>& lses,  // [N*B*H]
    std::vector<float>& global_lse,  // [B*H] — output
    int B, int H, int D, int N, int rank) {

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            // Compute global LSE = logsumexp over all ranks
            float max_lse = -1e30f;
            for (int n = 0; n < N; ++n) {
                float l = lses[n * B * H + b * H + h];
                max_lse = std::max(max_lse, l);
            }
            float sum_exp = 0.0f;
            for (int n = 0; n < N; ++n) {
                float l = lses[n * B * H + b * H + h];
                sum_exp += std::exp(l - max_lse);
            }
            float glse = max_lse + std::log(sum_exp);
            global_lse[b * H + h] = glse;

            // Correct this rank's output: out *= exp(local_lse - global_lse)
            float local_lse = lses[rank * B * H + b * H + h];
            float scale = std::exp(local_lse - glse);
            for (int d = 0; d < D; ++d) {
                output_f32[b * H * D + h * D + d] *= scale;
            }
        }
    }
}

// ── Helper: create BF16 device buffer from float host data ───────────────────

static void* alloc_bf16_device(const std::vector<float>& host_data) {
    std::vector<uint16_t> bf16_data(host_data.size());
    for (size_t i = 0; i < host_data.size(); ++i)
        bf16_data[i] = float_to_bf16_bits(host_data[i]);
    void* d_ptr = nullptr;
    cudaMalloc(&d_ptr, bf16_data.size() * sizeof(uint16_t));
    cudaMemcpy(d_ptr, bf16_data.data(), bf16_data.size() * sizeof(uint16_t),
               cudaMemcpyHostToDevice);
    return d_ptr;
}

static std::vector<float> read_bf16_device(void* d_ptr, size_t count) {
    std::vector<uint16_t> bf16_data(count);
    cudaMemcpy(bf16_data.data(), d_ptr, count * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i)
        result[i] = bf16_bits_to_float(bf16_data[i]);
    return result;
}

// ── Tests ────────────────────────────────────────────────────────────────────

TEST(DcpLseCorrect, SingleRankPassthrough) {
    REQUIRES_GPU();

    const int B = 4, H = 8, D = 64, N = 1, rank = 0;

    // With N=1, the correction scale = exp(lse - lse) = 1.0
    // Output should be unchanged.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> output_h(B * H * D);
    for (auto& v : output_h) v = dist(rng);

    std::vector<float> lses_h(N * B * H);
    for (auto& v : lses_h) v = dist(rng) * 5.0f;

    // Save original output
    auto original = output_h;

    // GPU buffers
    void* d_output = alloc_bf16_device(output_h);
    float* d_lses = nullptr;
    float* d_global_lse = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lses, lses_h.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global_lse, B * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lses, lses_h.data(), lses_h.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

    DcpLseCorrectParams params{};
    params.output = d_output;
    params.lses = d_lses;
    params.global_lse = d_global_lse;
    params.B = B; params.H = H; params.D = D; params.N = N; params.rank = rank;
    params.stride_o_B = H * D; params.stride_o_H = D; params.stride_o_D = 1;
    params.stride_lse_N = B * H; params.stride_lse_B = H; params.stride_lse_H = 1;

    run_dcp_lse_correct_kernel(params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result = read_bf16_device(d_output, B * H * D);
    for (int i = 0; i < B * H * D; ++i) {
        EXPECT_NEAR(result[i], original[i], 0.05f)
            << "Mismatch at index " << i;
    }

    cudaFree(d_output);
    cudaFree(d_lses);
    cudaFree(d_global_lse);
}

TEST(DcpLseCorrect, MultiRankCorrection) {
    REQUIRES_GPU();

    const int B = 2, H = 4, D = 32, N = 4;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> lse_dist(-2.0f, 2.0f);

    // Create per-rank outputs and LSEs
    std::vector<std::vector<float>> rank_outputs(N);
    std::vector<float> lses_h(N * B * H);

    for (int n = 0; n < N; ++n) {
        rank_outputs[n].resize(B * H * D);
        for (auto& v : rank_outputs[n]) v = dist(rng);
        for (int j = 0; j < B * H; ++j)
            lses_h[n * B * H + j] = lse_dist(rng);
    }

    // CPU reference: sum of corrected outputs across all ranks
    // This simulates what the orchestrator does: run kernel on each rank,
    // then allreduce the corrected outputs.
    std::vector<float> expected(B * H * D, 0.0f);
    for (int n = 0; n < N; ++n) {
        auto corrected = rank_outputs[n];
        std::vector<float> glse(B * H);
        dcp_lse_correct_ref(corrected, lses_h, glse, B, H, D, N, n);
        for (int i = 0; i < B * H * D; ++i)
            expected[i] += corrected[i];
    }

    // GPU: run correction for each rank and sum
    std::vector<float> gpu_sum(B * H * D, 0.0f);
    float* d_lses = nullptr;
    float* d_global_lse = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lses, lses_h.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global_lse, B * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lses, lses_h.data(), lses_h.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

    for (int n = 0; n < N; ++n) {
        void* d_output = alloc_bf16_device(rank_outputs[n]);

        DcpLseCorrectParams params{};
        params.output = d_output;
        params.lses = d_lses;
        params.global_lse = d_global_lse;
        params.B = B; params.H = H; params.D = D; params.N = N; params.rank = n;
        params.stride_o_B = H * D; params.stride_o_H = D; params.stride_o_D = 1;
        params.stride_lse_N = B * H; params.stride_lse_B = H; params.stride_lse_H = 1;

        run_dcp_lse_correct_kernel(params, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto result = read_bf16_device(d_output, B * H * D);
        for (int i = 0; i < B * H * D; ++i)
            gpu_sum[i] += result[i];

        cudaFree(d_output);
    }

    // Compare GPU sum with CPU reference sum
    for (int i = 0; i < B * H * D; ++i) {
        EXPECT_NEAR(gpu_sum[i], expected[i], 0.1f)
            << "Sum mismatch at index " << i;
    }

    cudaFree(d_lses);
    cudaFree(d_global_lse);
}

TEST(DcpLseCorrect, NumericalStability) {
    REQUIRES_GPU();

    // Large LSE differences: one rank has lse=100, another has lse=-100.
    // naive exp(100) overflows, but logsumexp should handle it.
    const int B = 1, H = 1, D = 16, N = 2;

    std::vector<float> output0(D, 1.0f);
    std::vector<float> output1(D, 2.0f);
    std::vector<float> lses_h = {100.0f, -100.0f};  // [N*B*H] = [2]

    // Rank 0 dominates: scale0 = exp(100 - 100) = 1.0, scale1 ≈ 0
    void* d_output0 = alloc_bf16_device(output0);
    float* d_lses = nullptr;
    float* d_global_lse = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lses, lses_h.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global_lse, B * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lses, lses_h.data(), lses_h.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

    DcpLseCorrectParams params{};
    params.output = d_output0;
    params.lses = d_lses;
    params.global_lse = d_global_lse;
    params.B = B; params.H = H; params.D = D; params.N = N; params.rank = 0;
    params.stride_o_B = H * D; params.stride_o_H = D; params.stride_o_D = 1;
    params.stride_lse_N = B * H; params.stride_lse_B = H; params.stride_lse_H = 1;

    run_dcp_lse_correct_kernel(params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result = read_bf16_device(d_output0, D);
    for (int d = 0; d < D; ++d) {
        EXPECT_FALSE(std::isnan(result[d])) << "NaN at d=" << d;
        EXPECT_FALSE(std::isinf(result[d])) << "Inf at d=" << d;
        // Rank 0 dominates, so output should be ≈ original (scale ≈ 1.0)
        EXPECT_NEAR(result[d], 1.0f, 0.01f) << "Unexpected value at d=" << d;
    }

    // Check global LSE ≈ 100.0 (dominated by rank 0)
    float h_global_lse;
    CUDA_CHECK(cudaMemcpy(&h_global_lse, d_global_lse, sizeof(float),
                           cudaMemcpyDeviceToHost));
    EXPECT_NEAR(h_global_lse, 100.0f, 0.01f);

    cudaFree(d_output0);
    cudaFree(d_lses);
    cudaFree(d_global_lse);
}

// KVS-3 / INV-KVS-EMPTY: an EMPTY DCP shard (a rank owning no tokens of a
// sequence — e.g. any sequence shorter than dcp_chunk_size at dcp=2) exports
// lse = +inf (the attention kernels' empty convention: sL == 0). The combine
// must (a) treat that rank's weight as EXACTLY 0 (its output contributes
// nothing to the allreduce), (b) compute global_lse from the remaining ranks
// only, and (c) never produce NaN — including the degenerate all-empty case.
TEST(DcpLseCorrect, EmptyShardInfLseWeightZero) {
    REQUIRES_GPU();

    const int B = 1, H = 2, D = 16, N = 2;

    // Rank 0: real attention (lse finite). Rank 1: empty shard (lse = +inf),
    // output filled with finite garbage — weight-0 must zero it regardless of
    // content (the executor does not clear the empty rank's output buffer).
    std::vector<float> output0(B * H * D, 1.0f);
    std::vector<float> output1(B * H * D, 7.0f);
    const float kInf = std::numeric_limits<float>::infinity();
    std::vector<float> lses_h = {3.0f, 2.5f,     // rank 0 [B*H]
                                 kInf, kInf};    // rank 1 [B*H] — empty shard

    float* d_lses = nullptr;
    float* d_global_lse = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lses, lses_h.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global_lse, B * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lses, lses_h.data(), lses_h.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

    auto run_rank = [&](void* d_out, int rank) {
        DcpLseCorrectParams params{};
        params.output = d_out;
        params.lses = d_lses;
        params.global_lse = d_global_lse;
        params.B = B; params.H = H; params.D = D; params.N = N;
        params.rank = rank;
        params.stride_o_B = H * D; params.stride_o_H = D; params.stride_o_D = 1;
        params.stride_lse_N = B * H; params.stride_lse_B = H;
        params.stride_lse_H = 1;
        run_dcp_lse_correct_kernel(params, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    void* d_output0 = alloc_bf16_device(output0);
    void* d_output1 = alloc_bf16_device(output1);
    run_rank(d_output0, 0);
    run_rank(d_output1, 1);

    // (a) Empty rank's output → exactly 0 (weight exp(−inf − g) = 0, no NaN).
    auto r1 = read_bf16_device(d_output1, B * H * D);
    for (int i = 0; i < B * H * D; ++i) {
        EXPECT_FALSE(std::isnan(r1[i])) << "empty-shard NaN at " << i;
        EXPECT_EQ(r1[i], 0.0f) << "empty-shard weight != 0 at " << i;
    }
    // Rank 0's factor = exp(own − global) = 1 (it IS the whole softmax).
    auto r0 = read_bf16_device(d_output0, B * H * D);
    for (int i = 0; i < B * H * D; ++i) {
        EXPECT_FALSE(std::isnan(r0[i])) << "real-rank NaN at " << i;
        EXPECT_NEAR(r0[i], 1.0f, 0.01f) << "real rank rescaled at " << i;
    }
    // (b) global_lse == rank 0's lse (the empty rank contributes nothing).
    std::vector<float> glse(B * H);
    CUDA_CHECK(cudaMemcpy(glse.data(), d_global_lse, B * H * sizeof(float),
                           cudaMemcpyDeviceToHost));
    EXPECT_NEAR(glse[0], 3.0f, 1e-4f);
    EXPECT_NEAR(glse[1], 2.5f, 1e-4f);

    // (c) Degenerate: EVERY rank empty (cannot happen for a real token — the
    // union always covers it — but must still be NaN-free: outputs go to 0).
    const std::vector<float> all_inf(N * B * H, kInf);
    CUDA_CHECK(cudaMemcpy(d_lses, all_inf.data(), all_inf.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
    void* d_output2 = alloc_bf16_device(output1);
    run_rank(d_output2, 1);
    auto r2 = read_bf16_device(d_output2, B * H * D);
    for (int i = 0; i < B * H * D; ++i) {
        EXPECT_FALSE(std::isnan(r2[i])) << "all-empty NaN at " << i;
        EXPECT_EQ(r2[i], 0.0f) << "all-empty weight != 0 at " << i;
    }

    cudaFree(d_output0);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_lses);
    cudaFree(d_global_lse);
}

// INV-KVS-QAG (TD-KVS-Q-ALLGATHER): the sharded-KV combine premise — per-head
// partials over DISJOINT token shards, corrected by this kernel and summed
// (the allreduce), must equal single-rank FULL-KV attention for every head.
// This is exactly the executor's post-Q-allgather dataflow: each rank attends
// ALL heads over its LOCAL token shard; the combine merges same-head partials.
TEST(DcpLseCorrect, DisjointShardCombineEqualsFullAttention) {
    REQUIRES_GPU();

    const int B = 1, H = 4, D = 8, N = 2, T = 10;
    const int shard_start[N + 1] = {0, 6, T};   // rank 0: [0,6), rank 1: [6,10)

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> sdist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);

    // Per-head attention scores + per-token value rows.
    std::vector<float> scores(H * T);
    std::vector<float> values(H * T * D);
    for (auto& v : scores) v = sdist(rng);
    for (auto& v : values) v = vdist(rng);

    // CPU: full-KV softmax attention per head (the single-rank reference).
    std::vector<float> full(H * D, 0.0f);
    for (int h = 0; h < H; ++h) {
        float mx = -1e30f;
        for (int t = 0; t < T; ++t) mx = std::max(mx, scores[h * T + t]);
        float denom = 0.0f;
        for (int t = 0; t < T; ++t) denom += std::exp(scores[h * T + t] - mx);
        for (int t = 0; t < T; ++t) {
            const float w = std::exp(scores[h * T + t] - mx) / denom;
            for (int d = 0; d < D; ++d)
                full[h * D + d] += w * values[(h * T + t) * D + d];
        }
    }

    // CPU: per-rank SHARD partials — local softmax over the shard's tokens
    // plus the natural-unit LSE (INV-LSE-NAT) the kernel consumes.
    std::vector<std::vector<float>> part(N, std::vector<float>(H * D, 0.0f));
    std::vector<float> lses_h(N * B * H);
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            float mx = -1e30f;
            for (int t = shard_start[n]; t < shard_start[n + 1]; ++t)
                mx = std::max(mx, scores[h * T + t]);
            float denom = 0.0f;
            for (int t = shard_start[n]; t < shard_start[n + 1]; ++t)
                denom += std::exp(scores[h * T + t] - mx);
            for (int t = shard_start[n]; t < shard_start[n + 1]; ++t) {
                const float w = std::exp(scores[h * T + t] - mx) / denom;
                for (int d = 0; d < D; ++d)
                    part[n][h * D + d] += w * values[(h * T + t) * D + d];
            }
            lses_h[n * B * H + h] = mx + std::log(denom);
        }
    }

    // GPU: correct each rank's partial with the kernel, then sum (allreduce).
    float* d_lses = nullptr;
    float* d_global_lse = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lses, lses_h.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global_lse, B * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lses, lses_h.data(), lses_h.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

    std::vector<float> combined(H * D, 0.0f);
    for (int n = 0; n < N; ++n) {
        void* d_out = alloc_bf16_device(part[n]);
        DcpLseCorrectParams params{};
        params.output = d_out;
        params.lses = d_lses;
        params.global_lse = d_global_lse;
        params.B = B; params.H = H; params.D = D; params.N = N;
        params.rank = n;
        params.stride_o_B = H * D; params.stride_o_H = D; params.stride_o_D = 1;
        params.stride_lse_N = B * H; params.stride_lse_B = H;
        params.stride_lse_H = 1;
        run_dcp_lse_correct_kernel(params, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto r = read_bf16_device(d_out, H * D);
        for (int i = 0; i < H * D; ++i) combined[i] += r[i];
        cudaFree(d_out);
    }

    // Combined per-head output == full-KV attention (bf16 tolerance).
    for (int h = 0; h < H; ++h) {
        for (int d = 0; d < D; ++d) {
            EXPECT_NEAR(combined[h * D + d], full[h * D + d], 0.02f)
                << "head " << h << " dim " << d;
        }
    }

    // global_lse == full-attention LSE per head.
    std::vector<float> glse(B * H);
    CUDA_CHECK(cudaMemcpy(glse.data(), d_global_lse, B * H * sizeof(float),
                           cudaMemcpyDeviceToHost));
    for (int h = 0; h < H; ++h) {
        float mx = -1e30f;
        for (int t = 0; t < T; ++t) mx = std::max(mx, scores[h * T + t]);
        float denom = 0.0f;
        for (int t = 0; t < T; ++t) denom += std::exp(scores[h * T + t] - mx);
        EXPECT_NEAR(glse[h], mx + std::log(denom), 1e-3f) << "head " << h;
    }

    cudaFree(d_lses);
    cudaFree(d_global_lse);
}
