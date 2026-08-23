// Unit tests for the DSpark V4 dflash draft kernels (ticket J).
//
// CPU double references implement the draft math directly:
//   kv rope:   [nope | interleaved-pair rotation of the tail] at
//              position base_pos + row (ticket-D cos|sin half-row table)
//   attention: latent MQA (K == V single vector), per-query SWA window over
//              the context rows + non-causal block rows, softmax in the
//              duplicated-rope score form (q_nope . k + q_rope . k_tail),
//              natural-unit logsumexp out
//   stream mean: out[r, d] = mean_s in[r, s*H + d]

#include "compute/kernels/dspark/dspark_v4.h"
#include "compute/kernels/mhc/hc_stream_mean.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: "                         \
                                     << cudaGetErrorString(_err);              \
    } while (0)

namespace {

float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& v) {
    std::vector<__nv_bfloat16> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = __float2bfloat16_rn(v[i]);
    return out;
}

std::vector<float> from_bf16(const std::vector<__nv_bfloat16>& v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = __bfloat162float(v[i]);
    return out;
}

// Ticket-D table layout: row = [cos_0..cos_{r/2-1} | sin_0..sin_{r/2-1}].
std::vector<float> make_cos_sin(int max_pos, int rope_dim, double theta) {
    const int half = rope_dim / 2;
    std::vector<float> t(size_t(max_pos) * rope_dim);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double freq = std::pow(theta, -2.0 * i / rope_dim);
            t[size_t(p) * rope_dim + i] = float(std::cos(p * freq));
            t[size_t(p) * rope_dim + half + i] = float(std::sin(p * freq));
        }
    }
    return t;
}

template <typename T>
T* dev_upload(const std::vector<T>& host) {
    T* d = nullptr;
    if (host.empty()) return d;
    EXPECT_EQ(cudaMalloc(&d, host.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    return d;
}

template <typename T>
std::vector<T> dev_download(const T* d, size_t n) {
    std::vector<T> out(n);
    EXPECT_EQ(cudaMemcpy(out.data(), d, n * sizeof(T),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return out;
}

TEST(DsparkV4Kernel, KvRopeMatchesReference) {
    REQUIRES_GPU();
    const int rows = 6, head_dim = 64, rope_dim = 8, base_pos = 37;
    const int half = rope_dim / 2, nope = head_dim - rope_dim;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> in(size_t(rows) * head_dim);
    for (auto& x : in) x = dist(rng);
    const auto table = make_cos_sin(base_pos + rows + 1, rope_dim, 1e4);

    auto h_in = to_bf16(in);
    auto* d_in = dev_upload(h_in);
    auto* d_tab = dev_upload(table);
    __nv_bfloat16* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, h_in.size() * sizeof(__nv_bfloat16)));

    lc::launch_dspark_v4_kv_rope(d_out, d_in, d_tab, base_pos, rows,
                                 head_dim, rope_dim, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto out = from_bf16(dev_download(d_out, h_in.size()));

    for (int r = 0; r < rows; ++r) {
        const float* cs = table.data() + size_t(base_pos + r) * rope_dim;
        for (int i = 0; i < nope; ++i) {
            EXPECT_EQ(out[size_t(r) * head_dim + i],
                      bf16_round(in[size_t(r) * head_dim + i]))
                << "nope r=" << r << " i=" << i;
        }
        for (int i = 0; i < half; ++i) {
            const float e = bf16_round(in[size_t(r) * head_dim + nope + 2 * i]);
            const float o =
                bf16_round(in[size_t(r) * head_dim + nope + 2 * i + 1]);
            const float c = cs[i], s = cs[half + i];
            EXPECT_EQ(out[size_t(r) * head_dim + nope + 2 * i],
                      bf16_round(e * c - o * s));
            EXPECT_EQ(out[size_t(r) * head_dim + nope + 2 * i + 1],
                      bf16_round(e * s + o * c));
        }
    }
    cudaFree(d_in);
    cudaFree(d_tab);
    cudaFree(d_out);
}

struct AttnFixture {
    int nq = 3, n_ctx = 40, base_pos = 40, window = 16;
    int h_q = 2, head_dim = 64, rope_dim = 8;
    float scale = 0.0f;
    std::vector<float> q_nope, q_rope, ctx, blk;

    void fill(uint32_t seed) {
        scale = 1.0f / std::sqrt(float(head_dim));
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        const int nope = head_dim - rope_dim;
        q_nope.assign(size_t(nq) * h_q * head_dim, 0.0f);
        q_rope.assign(size_t(nq) * h_q * rope_dim, 0.0f);
        for (int t = 0; t < nq; ++t)
            for (int h = 0; h < h_q; ++h) {
                for (int i = 0; i < nope; ++i)
                    q_nope[(size_t(t) * h_q + h) * head_dim + i] = dist(rng);
                // rope tail of q_nope stays ZERO (the q_prep contract).
                for (int i = 0; i < rope_dim; ++i)
                    q_rope[(size_t(t) * h_q + h) * rope_dim + i] = dist(rng);
            }
        ctx.assign(size_t(n_ctx) * head_dim, 0.0f);
        for (auto& x : ctx) x = dist(rng);
        blk.assign(size_t(nq) * head_dim, 0.0f);
        for (auto& x : blk) x = dist(rng);
    }

    // CPU double reference over bf16-rounded inputs.
    void reference(std::vector<float>& out, std::vector<float>& lse) const {
        const int nope_off = head_dim - rope_dim;
        out.assign(size_t(nq) * h_q * head_dim, 0.0f);
        lse.assign(size_t(nq) * h_q, 0.0f);
        for (int t = 0; t < nq; ++t) {
            const int p = base_pos + t;
            const int c0 = std::max(0, p - window + 1);
            const int nc = std::max(0, n_ctx - c0);
            const int total = nc + nq;
            for (int h = 0; h < h_q; ++h) {
                std::vector<double> sc(static_cast<size_t>(total), 0.0);
                auto key = [&](int j) -> const float* {
                    return j < nc ? ctx.data() + size_t(c0 + j) * head_dim
                                  : blk.data() + size_t(j - nc) * head_dim;
                };
                for (int j = 0; j < total; ++j) {
                    const float* k = key(j);
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; ++d)
                        dot += double(bf16_round(
                                   q_nope[(size_t(t) * h_q + h) * head_dim +
                                          d])) *
                               bf16_round(k[d]);
                    for (int d = 0; d < rope_dim; ++d)
                        dot += double(bf16_round(
                                   q_rope[(size_t(t) * h_q + h) * rope_dim +
                                          d])) *
                               bf16_round(k[nope_off + d]);
                    sc[size_t(j)] = dot * scale;
                }
                double m = -1e300;
                for (double s : sc) m = std::max(m, s);
                double l = 0.0;
                for (double& s : sc) {
                    s = std::exp(s - m);
                    l += s;
                }
                for (int d = 0; d < head_dim; ++d) {
                    double acc = 0.0;
                    for (int j = 0; j < total; ++j)
                        acc += sc[size_t(j)] * bf16_round(key(j)[d]);
                    out[(size_t(t) * h_q + h) * head_dim + d] =
                        float(acc / l);
                }
                lse[size_t(t) * h_q + h] = float(m + std::log(l));
            }
        }
    }

    void run(std::vector<float>& out, std::vector<float>& lse) {
        auto h_qn = to_bf16(q_nope);
        auto h_qr = to_bf16(q_rope);
        auto h_ctx = to_bf16(ctx);
        auto h_blk = to_bf16(blk);
        auto* d_qn = dev_upload(h_qn);
        auto* d_qr = dev_upload(h_qr);
        auto* d_ctx = dev_upload(h_ctx);
        auto* d_blk = dev_upload(h_blk);
        __nv_bfloat16* d_out = nullptr;
        float* d_lse = nullptr;
        ASSERT_EQ(cudaMalloc(&d_out, h_qn.size() * sizeof(__nv_bfloat16)),
                  cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_lse,
                             size_t(nq) * h_q * sizeof(float)),
                  cudaSuccess);
        lc::launch_dspark_v4_attention(d_out, d_lse, d_qn, d_qr,
                                       n_ctx > 0 ? d_ctx : nullptr, d_blk,
                                       nq, n_ctx, base_pos, window, h_q,
                                       head_dim, rope_dim, scale, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        out = from_bf16(dev_download(d_out, h_qn.size()));
        lse = dev_download(d_lse, size_t(nq) * h_q);
        cudaFree(d_qn);
        cudaFree(d_qr);
        cudaFree(d_ctx);
        cudaFree(d_blk);
        cudaFree(d_out);
        cudaFree(d_lse);
    }
};

TEST(DsparkV4Kernel, AttentionMatchesReference) {
    REQUIRES_GPU();
    AttnFixture f;
    f.fill(11);
    std::vector<float> ref_out, ref_lse, out, lse;
    f.reference(ref_out, ref_lse);
    f.run(out, lse);
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], ref_out[i], 2e-2f) << "out[" << i << "]";
    for (size_t i = 0; i < lse.size(); ++i)
        EXPECT_NEAR(lse[i], ref_lse[i], 1e-3f) << "lse[" << i << "]";
}

TEST(DsparkV4Kernel, AttentionNoContext) {
    REQUIRES_GPU();
    AttnFixture f;
    f.n_ctx = 0;
    f.base_pos = 0;
    f.fill(12);
    std::vector<float> ref_out, ref_lse, out, lse;
    f.reference(ref_out, ref_lse);
    f.run(out, lse);
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], ref_out[i], 2e-2f);
}

TEST(DsparkV4Kernel, AttentionWindowMasksContext) {
    REQUIRES_GPU();
    // Poison every out-of-window context row: outputs must be IDENTICAL to
    // the clean run (the window mask excludes them entirely).
    AttnFixture f;
    f.fill(13);
    std::vector<float> out_clean, lse_clean;
    f.run(out_clean, lse_clean);
    const int oldest_visible = f.base_pos + 0 - f.window + 1;
    ASSERT_GT(oldest_visible, 0);
    AttnFixture g = f;
    for (int j = 0; j < oldest_visible; ++j)
        for (int d = 0; d < f.head_dim; ++d)
            g.ctx[size_t(j) * f.head_dim + d] = 1e4f;
    std::vector<float> out_poison, lse_poison;
    g.run(out_poison, lse_poison);
    EXPECT_EQ(out_clean, out_poison);
    EXPECT_EQ(lse_clean, lse_poison);
}

TEST(DsparkV4Kernel, StreamMeanMatchesReference) {
    REQUIRES_GPU();
    const int rows = 5, hc = 4, hidden = 96;
    std::mt19937 rng(21);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> in(size_t(rows) * hc * hidden);
    for (auto& x : in) x = dist(rng);
    auto h_in = to_bf16(in);
    auto* d_in = dev_upload(h_in);
    __nv_bfloat16* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out,
                          size_t(rows) * hidden * sizeof(__nv_bfloat16)));
    lc::launch_hc_stream_mean(d_out, d_in, rows, hc, hidden, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto out = from_bf16(dev_download(d_out, size_t(rows) * hidden));
    for (int r = 0; r < rows; ++r)
        for (int d = 0; d < hidden; ++d) {
            float acc = 0.0f;
            for (int s = 0; s < hc; ++s)
                acc += bf16_round(
                    in[(size_t(r) * hc + s) * hidden + d]);
            EXPECT_EQ(out[size_t(r) * hidden + d], bf16_round(acc / hc))
                << "r=" << r << " d=" << d;
        }
    cudaFree(d_in);
    cudaFree(d_out);
}

}  // namespace
