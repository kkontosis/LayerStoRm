// TD-GLM-INDEXER-HADAMARD: orthonormal Walsh-Hadamard rotation of DSA
// lightning-indexer q/k rows.
//
// Validates the indexer_hadamard kernel against the llama.cpp convention
// (ref/llama.cpp/src/llama-kv-cache.cpp:20-58 ggml_gen_hadamard: Sylvester
// H_n seeded with H[0][0] = 1/sqrt(n) → orthonormal, symmetric, H^2 == I):
//   1) kernel output == CPU FWHT/sqrt(n) reference (BF16 tolerance);
//   2) involution: applying H twice returns the input (BF16 rounding);
//   3) score invariance: dot(q, k) == dot(Hq, Hk) in FP32 accumulation.

#include "compute/cuda_sm120_device_backend.h"
#include "core/attention_device.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
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

// CPU reference: in-place natural-order FWHT (== Sylvester Hadamard, no
// permutation) scaled by 1/sqrt(dim), FP32.
void cpu_fwht(std::vector<float>& row) {
    const int n = static_cast<int>(row.size());
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += h << 1) {
            for (int j = i; j < i + h; ++j) {
                const float a = row[j];
                const float b = row[j + h];
                row[j]     = a + b;
                row[j + h] = a - b;
            }
        }
    }
    const float s = 1.0f / std::sqrt(static_cast<float>(n));
    for (auto& v : row) v *= s;
}

std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& f) {
    std::vector<__nv_bfloat16> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = __float2bfloat16(f[i]);
    return h;
}
std::vector<float> to_f32(const std::vector<__nv_bfloat16>& h) {
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); ++i) f[i] = __bfloat162float(h[i]);
    return f;
}
}  // namespace

// (a) Kernel vs CPU FWHT reference on random BF16 rows.
TEST(IndexerHadamard, MatchesCpuReference) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();
    const int ROWS = 37, DIM = 128;  // odd row count exercises grid bounds

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(ROWS) * DIM);
    for (auto& v : x) v = dist(rng);

    // BF16-round the input first so CPU and GPU see identical values.
    auto hx = to_bf16(x);
    x = to_f32(hx);

    auto* dx = upload(hx);
    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();
    dev.indexer_hadamard(dx, ROWS, DIM, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto got = to_f32(download<__nv_bfloat16>(dx, hx.size()));

    for (int r = 0; r < ROWS; ++r) {
        std::vector<float> ref(x.begin() + static_cast<size_t>(r) * DIM,
                               x.begin() + static_cast<size_t>(r + 1) * DIM);
        cpu_fwht(ref);
        for (int i = 0; i < DIM; ++i) {
            const float g = got[static_cast<size_t>(r) * DIM + i];
            const float tol = 1e-2f * std::max(1.0f, std::abs(ref[i]));
            EXPECT_NEAR(g, ref[i], tol) << "row " << r << " elem " << i;
        }
    }
    cudaFree(dx);
}

// Non-128 power-of-two width (generic path).
TEST(IndexerHadamard, Dim64MatchesCpuReference) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();
    const int ROWS = 5, DIM = 64;

    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(ROWS) * DIM);
    for (auto& v : x) v = dist(rng);
    auto hx = to_bf16(x);
    x = to_f32(hx);

    auto* dx = upload(hx);
    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();
    dev.indexer_hadamard(dx, ROWS, DIM, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto got = to_f32(download<__nv_bfloat16>(dx, hx.size()));

    for (int r = 0; r < ROWS; ++r) {
        std::vector<float> ref(x.begin() + static_cast<size_t>(r) * DIM,
                               x.begin() + static_cast<size_t>(r + 1) * DIM);
        cpu_fwht(ref);
        for (int i = 0; i < DIM; ++i)
            EXPECT_NEAR(got[static_cast<size_t>(r) * DIM + i], ref[i],
                        1e-2f * std::max(1.0f, std::abs(ref[i])))
                << "row " << r << " elem " << i;
    }
    cudaFree(dx);
}

// (b) Involution: H is normalized-symmetric (H = H^T, H·H = I per the llama.cpp
// comment "res^2 == I"), so applying the kernel twice returns the input up to
// BF16 rounding of the intermediate.
TEST(IndexerHadamard, InvolutionReturnsInput) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();
    const int ROWS = 33, DIM = 128;

    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(ROWS) * DIM);
    for (auto& v : x) v = dist(rng);
    auto hx = to_bf16(x);
    x = to_f32(hx);

    auto* dx = upload(hx);
    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();
    dev.indexer_hadamard(dx, ROWS, DIM, nullptr);
    dev.indexer_hadamard(dx, ROWS, DIM, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto got = to_f32(download<__nv_bfloat16>(dx, hx.size()));

    for (size_t i = 0; i < x.size(); ++i) {
        // Intermediate BF16 rounding (~2^-8 relative) amplified by the
        // second transform's +/- accumulation over 128 elements.
        const float tol = 5e-2f * std::max(1.0f, std::abs(x[i]));
        EXPECT_NEAR(got[i], x[i], tol) << "elem " << i;
    }
    cudaFree(dx);
}

// (c) Score invariance: dot(q, k) == dot(Hq, Hk) (H orthogonal), FP32
// accumulation on the host over the BF16 outputs.
TEST(IndexerHadamard, ScoreInvariance) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();
    const int NIH = 32, DIM = 128;  // GLM-5.2 indexer shape: 32 q heads, 1 k

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> q(static_cast<size_t>(NIH) * DIM), k(DIM);
    for (auto& v : q) v = dist(rng);
    for (auto& v : k) v = dist(rng);
    auto hq = to_bf16(q);
    auto hk = to_bf16(k);
    q = to_f32(hq);
    k = to_f32(hk);

    auto* dq = upload(hq);
    auto* dk = upload(hk);
    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();
    dev.indexer_hadamard(dq, NIH, DIM, nullptr);  // all 32 q head rows
    dev.indexer_hadamard(dk, 1, DIM, nullptr);    // single k row
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto rq = to_f32(download<__nv_bfloat16>(dq, hq.size()));
    auto rk = to_f32(download<__nv_bfloat16>(dk, hk.size()));

    for (int h = 0; h < NIH; ++h) {
        double ref = 0.0, rot = 0.0;
        for (int i = 0; i < DIM; ++i) {
            ref += static_cast<double>(q[static_cast<size_t>(h) * DIM + i])
                 * static_cast<double>(k[i]);
            rot += static_cast<double>(rq[static_cast<size_t>(h) * DIM + i])
                 * static_cast<double>(rk[i]);
        }
        // BF16 rounding of both rotated operands; dot over 128 elements of
        // O(1) magnitude → absolute tolerance dominated by ~sqrt(128)*2^-8.
        EXPECT_NEAR(rot, ref, 0.35 + 1e-2 * std::abs(ref)) << "head " << h;
    }
    cudaFree(dq);
    cudaFree(dk);
}
