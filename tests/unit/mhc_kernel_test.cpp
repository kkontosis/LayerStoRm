// Unit tests for the DeepSeek-V4 mHC kernels (V4-5b).
//
// Reference semantics (ref/vllm kernels/mhc/torch.py mhc_pre_torch/
// mhc_post_torch + ref/llama.cpp models/deepseek4.cpp build_hc_pre/
// build_hc_post/build_hc_head/build_hc_sinkhorn), computed here in double
// precision:
//   mixes = fn @ (flat / rms(flat));  rms over the full hc*hidden vector
//   pre   = sigmoid(mixes[0:hc]*s0 + b[0:hc]) + eps
//   post  = sigmoid(mixes[hc:2hc]*s1 + b[hc:2hc]) * 2
//   comb  = sinkhorn(mixes[2hc:]*s2 + b[2hc:])  (softmax over dst → +eps →
//           col-norm → 19×{row,col}; eps in every denominator)
//   x     = Σ_s pre[s]·R[s]         R'[d] = post[d]·y + Σ_s comb[s][d]·R[s]

#include "compute/kernels/mhc/mhc.h"

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

constexpr int kHc = 4;
constexpr float kRmsEps = 1e-6f;
constexpr float kHcEps = 1e-6f;
constexpr float kPostMult = 2.0f;
constexpr int kIters = 20;

double sigmoid_ref(double x) { return 1.0 / (1.0 + std::exp(-x)); }

float bf16_round(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// CPU double reference for mhc_pre. residual given as float (bf16-rounded).
// Outputs: post [hc], comb [hc*hc] ([src][dst]), x [hidden].
void ref_mhc_pre(const std::vector<float>& residual_row,  // [hc*hidden]
                 const std::vector<float>& fn,            // [hc_mix, hc*hidden]
                 const std::vector<float>& scale,         // [3]
                 const std::vector<float>& base,          // [hc_mix]
                 int hidden, std::vector<double>& post_out,
                 std::vector<double>& comb_out, std::vector<double>& x_out) {
    const int hc_dim = kHc * hidden;
    const int hc_mix = (2 + kHc) * kHc;
    double sqrsum = 0.0;
    for (int i = 0; i < hc_dim; ++i)
        sqrsum += double(residual_row[i]) * residual_row[i];
    const double inv_rms = 1.0 / std::sqrt(sqrsum / hc_dim + kRmsEps);

    std::vector<double> mixes(hc_mix, 0.0);
    for (int j = 0; j < hc_mix; ++j) {
        double acc = 0.0;
        for (int i = 0; i < hc_dim; ++i)
            acc += double(fn[size_t(j) * hc_dim + i]) * residual_row[i];
        mixes[j] = acc * inv_rms;
    }

    std::vector<double> pre(kHc);
    post_out.assign(kHc, 0.0);
    for (int k = 0; k < kHc; ++k) {
        pre[k] = sigmoid_ref(mixes[k] * scale[0] + base[k]) + kHcEps;
        post_out[k] =
            sigmoid_ref(mixes[kHc + k] * scale[1] + base[kHc + k]) * kPostMult;
    }

    comb_out.assign(kHc * kHc, 0.0);
    for (int e = 0; e < kHc * kHc; ++e)
        comb_out[e] = mixes[2 * kHc + e] * scale[2] + base[2 * kHc + e];
    // Sinkhorn: stable row softmax (over dst) → +eps → col → 19×{row,col}.
    for (int s = 0; s < kHc; ++s) {
        double m = comb_out[s * kHc];
        for (int d = 1; d < kHc; ++d) m = std::max(m, comb_out[s * kHc + d]);
        double sum = 0.0;
        for (int d = 0; d < kHc; ++d) {
            comb_out[s * kHc + d] = std::exp(comb_out[s * kHc + d] - m);
            sum += comb_out[s * kHc + d];
        }
        for (int d = 0; d < kHc; ++d)
            comb_out[s * kHc + d] = comb_out[s * kHc + d] / sum + kHcEps;
    }
    auto norm_cols = [&]() {
        for (int d = 0; d < kHc; ++d) {
            double sum = 0.0;
            for (int s = 0; s < kHc; ++s) sum += comb_out[s * kHc + d];
            for (int s = 0; s < kHc; ++s) comb_out[s * kHc + d] /= (sum + kHcEps);
        }
    };
    auto norm_rows = [&]() {
        for (int s = 0; s < kHc; ++s) {
            double sum = 0.0;
            for (int d = 0; d < kHc; ++d) sum += comb_out[s * kHc + d];
            for (int d = 0; d < kHc; ++d) comb_out[s * kHc + d] /= (sum + kHcEps);
        }
    };
    norm_cols();
    for (int it = 1; it < kIters; ++it) {
        norm_rows();
        norm_cols();
    }

    x_out.assign(hidden, 0.0);
    for (int s = 0; s < kHc; ++s)
        for (int i = 0; i < hidden; ++i)
            x_out[i] += pre[s] * residual_row[size_t(s) * hidden + i];
}

struct DeviceBuf {
    void* p = nullptr;
    explicit DeviceBuf(size_t bytes) { cudaMalloc(&p, bytes); }
    ~DeviceBuf() { cudaFree(p); }
};

std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& v) {
    std::vector<__nv_bfloat16> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = __float2bfloat16_rn(v[i]);
    return out;
}

void run_pre_case(int rows, int hidden, uint32_t seed) {
    const int hc_dim = kHc * hidden;
    const int hc_mix = (2 + kHc) * kHc;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> residual(size_t(rows) * hc_dim);
    for (auto& v : residual) v = bf16_round(dist(rng));
    std::vector<float> fn(size_t(hc_mix) * hc_dim);
    for (auto& v : fn) v = dist(rng) * 0.02f;
    std::vector<float> scale = {1.1f, 0.9f, 1.3f};
    std::vector<float> base(hc_mix);
    for (auto& v : base) v = dist(rng) * 0.5f;

    auto residual_bf = to_bf16(residual);
    DeviceBuf d_res(residual_bf.size() * 2), d_fn(fn.size() * 4),
        d_scale(scale.size() * 4), d_base(base.size() * 4),
        d_post(size_t(rows) * kHc * 4), d_comb(size_t(rows) * kHc * kHc * 4),
        d_x(size_t(rows) * hidden * 2);
    CUDA_CHECK(cudaMemcpy(d_res.p, residual_bf.data(), residual_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fn.p, fn.data(), fn.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale.p, scale.data(), scale.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base.p, base.data(), base.size() * 4,
                          cudaMemcpyHostToDevice));

    lc::launch_mhc_pre(d_x.p, d_post.p, d_comb.p, d_res.p, d_fn.p, d_scale.p,
                       d_base.p, kRmsEps, kHcEps, kPostMult, kIters, rows, kHc,
                       hidden, /*stream=*/nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> post(size_t(rows) * kHc), comb(size_t(rows) * kHc * kHc);
    std::vector<__nv_bfloat16> x(size_t(rows) * hidden);
    CUDA_CHECK(cudaMemcpy(post.data(), d_post.p, post.size() * 4,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb.data(), d_comb.p, comb.size() * 4,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x.data(), d_x.p, x.size() * 2,
                          cudaMemcpyDeviceToHost));

    for (int t = 0; t < rows; ++t) {
        std::vector<float> row(residual.begin() + size_t(t) * hc_dim,
                               residual.begin() + size_t(t + 1) * hc_dim);
        std::vector<double> rpost, rcomb, rx;
        ref_mhc_pre(row, fn, scale, base, hidden, rpost, rcomb, rx);
        for (int k = 0; k < kHc; ++k) {
            EXPECT_NEAR(post[size_t(t) * kHc + k], rpost[k], 2e-4)
                << "post t=" << t << " k=" << k;
        }
        double row_sum_err = 0.0, col_sum_err = 0.0;
        for (int e = 0; e < kHc * kHc; ++e) {
            EXPECT_NEAR(comb[size_t(t) * kHc * kHc + e], rcomb[e], 2e-4)
                << "comb t=" << t << " e=" << e;
        }
        // Doubly-stochastic check on the kernel output.
        for (int s = 0; s < kHc; ++s) {
            double rs = 0.0, cs = 0.0;
            for (int d = 0; d < kHc; ++d) {
                rs += comb[size_t(t) * kHc * kHc + s * kHc + d];
                cs += comb[size_t(t) * kHc * kHc + d * kHc + s];
            }
            row_sum_err = std::max(row_sum_err, std::abs(rs - 1.0));
            col_sum_err = std::max(col_sum_err, std::abs(cs - 1.0));
        }
        // Sinkhorn ends on a COLUMN normalization: col sums are exact (up to
        // the +eps terms); row sums only converge approximately in 20 iters
        // (extreme random logits can leave a few % — matches the vLLM torch
        // reference bit-for-bit, verified by the comb comparison above).
        EXPECT_LT(row_sum_err, 0.1);
        EXPECT_LT(col_sum_err, 1e-2);
        for (int i = 0; i < hidden; ++i) {
            const float got = __bfloat162float(x[size_t(t) * hidden + i]);
            EXPECT_NEAR(got, rx[i], 3e-2 + 2e-2 * std::abs(rx[i]))
                << "x t=" << t << " i=" << i;
        }
    }
}

}  // namespace

TEST(MhcKernel, PreMatchesDoubleReference_1Token) {
    REQUIRES_GPU();
    run_pre_case(1, 4096, 42);
}

TEST(MhcKernel, PreMatchesDoubleReference_8Tokens) {
    REQUIRES_GPU();
    run_pre_case(8, 4096, 43);
}

TEST(MhcKernel, PreMatchesDoubleReference_SmallHidden) {
    REQUIRES_GPU();
    run_pre_case(16, 512, 44);
}

TEST(MhcKernel, PostMatchesDoubleReference) {
    REQUIRES_GPU();
    const int rows = 8, hidden = 4096, hc_dim = kHc * hidden;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> residual(size_t(rows) * hc_dim), y(size_t(rows) * hidden);
    for (auto& v : residual) v = bf16_round(dist(rng));
    for (auto& v : y) v = bf16_round(dist(rng));
    std::vector<float> post(size_t(rows) * kHc), comb(size_t(rows) * kHc * kHc);
    for (auto& v : post) v = std::abs(dist(rng));
    for (auto& v : comb) v = std::abs(dist(rng)) * 0.25f;

    auto res_bf = to_bf16(residual);
    auto y_bf = to_bf16(y);
    DeviceBuf d_res(res_bf.size() * 2), d_y(y_bf.size() * 2),
        d_post(post.size() * 4), d_comb(comb.size() * 4),
        d_out(res_bf.size() * 2);
    CUDA_CHECK(cudaMemcpy(d_res.p, res_bf.data(), res_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y.p, y_bf.data(), y_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_post.p, post.data(), post.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_comb.p, comb.data(), comb.size() * 4,
                          cudaMemcpyHostToDevice));

    lc::launch_mhc_post(d_out.p, d_y.p, d_res.p, d_post.p, d_comb.p, rows, kHc,
                        4096, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out(res_bf.size());
    CUDA_CHECK(cudaMemcpy(out.data(), d_out.p, out.size() * 2,
                          cudaMemcpyDeviceToHost));

    for (int t = 0; t < rows; ++t) {
        for (int d = 0; d < kHc; ++d) {
            for (int i = 0; i < hidden; ++i) {
                double acc = double(post[size_t(t) * kHc + d]) *
                             y[size_t(t) * hidden + i];
                for (int s = 0; s < kHc; ++s)
                    acc += double(comb[size_t(t) * kHc * kHc + s * kHc + d]) *
                           residual[size_t(t) * hc_dim + size_t(s) * hidden + i];
                const float got = __bfloat162float(
                    out[size_t(t) * hc_dim + size_t(d) * hidden + i]);
                EXPECT_NEAR(got, acc, 3e-2 + 2e-2 * std::abs(acc))
                    << "t=" << t << " d=" << d << " i=" << i;
            }
        }
    }
}

TEST(MhcKernel, PostInPlaceMatchesOutOfPlace) {
    REQUIRES_GPU();
    const int rows = 4, hidden = 4096, hc_dim = kHc * hidden;
    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> residual(size_t(rows) * hc_dim), y(size_t(rows) * hidden);
    for (auto& v : residual) v = bf16_round(dist(rng));
    for (auto& v : y) v = bf16_round(dist(rng));
    std::vector<float> post(size_t(rows) * kHc, 1.5f),
        comb(size_t(rows) * kHc * kHc, 0.25f);

    auto res_bf = to_bf16(residual);
    auto y_bf = to_bf16(y);
    DeviceBuf d_res(res_bf.size() * 2), d_res2(res_bf.size() * 2),
        d_y(y_bf.size() * 2), d_post(post.size() * 4), d_comb(comb.size() * 4),
        d_out(res_bf.size() * 2);
    CUDA_CHECK(cudaMemcpy(d_res.p, res_bf.data(), res_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res2.p, res_bf.data(), res_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y.p, y_bf.data(), y_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_post.p, post.data(), post.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_comb.p, comb.data(), comb.size() * 4,
                          cudaMemcpyHostToDevice));

    lc::launch_mhc_post(d_out.p, d_y.p, d_res.p, d_post.p, d_comb.p, rows, kHc,
                        hidden, nullptr);
    // In-place: residual_out == residual.
    lc::launch_mhc_post(d_res2.p, d_y.p, d_res2.p, d_post.p, d_comb.p, rows,
                        kHc, hidden, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> a(res_bf.size()), b(res_bf.size());
    CUDA_CHECK(cudaMemcpy(a.data(), d_out.p, a.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), d_res2.p, b.size() * 2,
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(a, b) << "in-place hc_post must be bit-identical to out-of-place";
}

TEST(MhcKernel, HeadMatchesDoubleReference) {
    REQUIRES_GPU();
    const int rows = 4, hidden = 4096, hc_dim = kHc * hidden;
    std::mt19937 rng(21);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> residual(size_t(rows) * hc_dim);
    for (auto& v : residual) v = bf16_round(dist(rng));
    std::vector<float> fn(size_t(kHc) * hc_dim);
    for (auto& v : fn) v = dist(rng) * 0.02f;
    std::vector<float> scale = {0.8f};
    std::vector<float> base = {0.1f, -0.2f, 0.3f, 0.05f};

    auto res_bf = to_bf16(residual);
    DeviceBuf d_res(res_bf.size() * 2), d_fn(fn.size() * 4), d_scale(4),
        d_base(base.size() * 4), d_x(size_t(rows) * hidden * 2);
    CUDA_CHECK(cudaMemcpy(d_res.p, res_bf.data(), res_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fn.p, fn.data(), fn.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale.p, scale.data(), 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base.p, base.data(), base.size() * 4,
                          cudaMemcpyHostToDevice));

    lc::launch_mhc_head(d_x.p, d_res.p, d_fn.p, d_scale.p, d_base.p, kRmsEps,
                        kHcEps, rows, kHc, hidden, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> x(size_t(rows) * hidden);
    CUDA_CHECK(cudaMemcpy(x.data(), d_x.p, x.size() * 2,
                          cudaMemcpyDeviceToHost));

    for (int t = 0; t < rows; ++t) {
        const float* row = residual.data() + size_t(t) * hc_dim;
        double sqrsum = 0.0;
        for (int i = 0; i < hc_dim; ++i) sqrsum += double(row[i]) * row[i];
        const double inv_rms = 1.0 / std::sqrt(sqrsum / hc_dim + kRmsEps);
        double pre[kHc];
        for (int k = 0; k < kHc; ++k) {
            double acc = 0.0;
            for (int i = 0; i < hc_dim; ++i)
                acc += double(fn[size_t(k) * hc_dim + i]) * row[i];
            pre[k] = sigmoid_ref(acc * inv_rms * scale[0] + base[k]) + kHcEps;
        }
        for (int i = 0; i < hidden; ++i) {
            double acc = 0.0;
            for (int s = 0; s < kHc; ++s)
                acc += pre[s] * row[size_t(s) * hidden + i];
            const float got = __bfloat162float(x[size_t(t) * hidden + i]);
            EXPECT_NEAR(got, acc, 3e-2 + 2e-2 * std::abs(acc))
                << "t=" << t << " i=" << i;
        }
    }
}

TEST(MhcKernel, ExpandRepeat) {
    REQUIRES_GPU();
    const int rows = 5, hidden = 4096;
    std::mt19937 rng(31);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(size_t(rows) * hidden);
    for (auto& v : x) v = bf16_round(dist(rng));
    auto x_bf = to_bf16(x);
    DeviceBuf d_x(x_bf.size() * 2), d_out(size_t(rows) * kHc * hidden * 2);
    CUDA_CHECK(cudaMemcpy(d_x.p, x_bf.data(), x_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    lc::launch_hc_expand_repeat(d_out.p, d_x.p, rows, kHc, hidden, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> out(size_t(rows) * kHc * hidden);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out.p, out.size() * 2,
                          cudaMemcpyDeviceToHost));
    const auto* xs = reinterpret_cast<const uint16_t*>(x_bf.data());
    for (int t = 0; t < rows; ++t)
        for (int s = 0; s < kHc; ++s)
            for (int i = 0; i < hidden; ++i)
                ASSERT_EQ(out[(size_t(t) * kHc + s) * hidden + i],
                          xs[size_t(t) * hidden + i])
                    << "t=" << t << " s=" << s << " i=" << i;
}
