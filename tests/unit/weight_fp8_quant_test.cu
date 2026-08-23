// Unit tests for weight_fp8_quant kernel (KD-4f-c3).
//
// GPU-required: allocates BF16 data on device, quantizes to FP8 row-major
// with tile-level scales, and verifies correctness.

#include "smxx/quant/weight_fp8_quant.h"

#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace lc = layerstorm::compute;

namespace {

// Helper: allocate + copy host→device.
template <typename T>
T* to_device(const std::vector<T>& host) {
    T* d = nullptr;
    cudaMalloc(&d, host.size() * sizeof(T));
    cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

// Helper: copy device→host.
template <typename T>
std::vector<T> to_host(const T* d, size_t count) {
    std::vector<T> h(count);
    cudaMemcpy(h.data(), d, count * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

}  // namespace

// ── WeightFp8QuantTest (GPU required) ──────────────────────────────────────

class WeightFp8QuantTest : public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        cudaGetDeviceCount(&count);
        if (count == 0) GTEST_SKIP() << "No CUDA device available";
        cudaSetDevice(0);
    }
};

// Basic: quantize a small matrix, verify output is finite.
TEST_F(WeightFp8QuantTest, SmallMatrixFinite) {
    constexpr int N = 256;
    constexpr int K = 256;

    // Fill BF16 input with a ramp pattern.
    std::vector<__nv_bfloat16> h_input(N * K);
    for (int i = 0; i < N * K; ++i) {
        float val = (static_cast<float>(i % 1000) - 500.0f) * 0.1f;
        h_input[i] = __float2bfloat16(val);
    }

    auto* d_input = to_device(h_input);

    // Allocate output buffers.
    const int N_blocks = (N + 127) / 128;
    const int K_blocks = (K + 127) / 128;

    __nv_fp8_e4m3* d_output = nullptr;
    cudaMalloc(&d_output, static_cast<size_t>(N) * K);

    float* d_scales = nullptr;
    cudaMalloc(&d_scales, static_cast<size_t>(N_blocks) * K_blocks * sizeof(float));

    lc::WeightFp8QuantParams params{};
    params.N = N;
    params.K = K;
    params.input = d_input;
    params.output = d_output;
    params.scales = d_scales;

    ASSERT_NO_THROW(lc::launch_weight_fp8_quant(params, nullptr));
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // Verify scales are finite and positive.
    auto h_scales = to_host(d_scales, static_cast<size_t>(N_blocks) * K_blocks);
    for (size_t i = 0; i < h_scales.size(); ++i) {
        EXPECT_TRUE(std::isfinite(h_scales[i]))
            << "Scale[" << i << "] = " << h_scales[i];
        EXPECT_GT(h_scales[i], 0.0f)
            << "Scale[" << i << "] should be positive";
    }

    // Verify FP8 output is finite (no NaN/Inf).
    auto h_output = to_host(d_output, static_cast<size_t>(N) * K);
    for (size_t i = 0; i < h_output.size(); ++i) {
        float val = static_cast<float>(h_output[i]);
        EXPECT_TRUE(std::isfinite(val))
            << "FP8[" << i << "] is not finite";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scales);
}

// Row-major layout: verify element at (n, k) is stored at offset n * K + k.
TEST_F(WeightFp8QuantTest, RowMajorLayout) {
    constexpr int N = 128;
    constexpr int K = 128;

    // Fill BF16 row-major: input[n, k] = n * K + k (as float).
    // After quantization: output[n * K + k] should correspond to input[n, k].
    std::vector<__nv_bfloat16> h_input(N * K);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            // Use distinct values per element so we can check layout.
            float val = static_cast<float>(n + 1) * 0.5f;
            h_input[n * K + k] = __float2bfloat16(val);
        }
    }

    auto* d_input = to_device(h_input);

    __nv_fp8_e4m3* d_output = nullptr;
    cudaMalloc(&d_output, static_cast<size_t>(N) * K);

    float* d_scales = nullptr;
    cudaMalloc(&d_scales, sizeof(float));

    lc::WeightFp8QuantParams params{N, K, d_input, d_output, d_scales};
    lc::launch_weight_fp8_quant(params, nullptr);
    cudaDeviceSynchronize();

    auto h_output = to_host(d_output, static_cast<size_t>(N) * K);
    auto h_scales = to_host(d_scales, 1);
    float scale = h_scales[0];

    // Verify row-major layout: all K columns for the same row n should
    // have the same FP8 value, since the input is constant per row.
    // This confirms data is stored at output[n * K + k] (K stride-1).
    for (int n = 0; n < N; ++n) {
        float ref_val = static_cast<float>(h_output[n * K + 0]);
        for (int k = 1; k < K; ++k) {
            float col_val = static_cast<float>(h_output[n * K + k]);
            EXPECT_EQ(col_val, ref_val)
                << "Row-major layout broken: row " << n << " col " << k
                << " differs from col 0";
        }
    }

    // Verify monotonicity: dequantized values should increase with n.
    float prev = 0.0f;
    for (int n = 0; n < N; ++n) {
        float dequant = static_cast<float>(h_output[n * K + 0]) * scale;
        EXPECT_GE(dequant, prev)
            << "Monotonicity broken at n=" << n;
        prev = dequant;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scales);
}

// Non-aligned dimensions: N and K not divisible by 128.
TEST_F(WeightFp8QuantTest, NonAlignedDimensions) {
    constexpr int N = 200;
    constexpr int K = 300;

    std::vector<__nv_bfloat16> h_input(N * K);
    for (int i = 0; i < N * K; ++i) {
        h_input[i] = __float2bfloat16(1.0f);
    }

    auto* d_input = to_device(h_input);
    const int N_blocks = (N + 127) / 128;  // 2
    const int K_blocks = (K + 127) / 128;  // 3

    __nv_fp8_e4m3* d_output = nullptr;
    cudaMalloc(&d_output, static_cast<size_t>(N) * K);

    float* d_scales = nullptr;
    cudaMalloc(&d_scales, static_cast<size_t>(N_blocks) * K_blocks * sizeof(float));

    lc::WeightFp8QuantParams params{N, K, d_input, d_output, d_scales};
    ASSERT_NO_THROW(lc::launch_weight_fp8_quant(params, nullptr));
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // All scales should be finite.
    auto h_scales = to_host(d_scales, static_cast<size_t>(N_blocks) * K_blocks);
    for (size_t i = 0; i < h_scales.size(); ++i) {
        EXPECT_TRUE(std::isfinite(h_scales[i]));
    }

    // All FP8 values should be finite.
    auto h_output = to_host(d_output, static_cast<size_t>(N) * K);
    for (size_t i = 0; i < h_output.size(); ++i) {
        float val = static_cast<float>(h_output[i]);
        EXPECT_TRUE(std::isfinite(val));
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scales);
}

// Scale K-major layout: scales[k_blk + n_blk * K_blocks] — the scale_B
// contract of launch_fp8_gemm ([ceil(K/128), ceil(N/128)] column-major).
// TD-GOLDEN: this test previously asserted N-major — the layout the kernel
// USED to write — validating the quantizer against itself instead of the
// GEMM that consumes it. The mismatch mis-scaled every off-diagonal tile.
TEST_F(WeightFp8QuantTest, ScaleKMajorLayout) {
    constexpr int N = 256;
    constexpr int K = 256;
    const int N_blocks = 2;
    const int K_blocks = 2;

    // Fill input so each 128x128 tile has a distinct max magnitude.
    std::vector<__nv_bfloat16> h_input(N * K);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int n_blk = n / 128;
            int k_blk = k / 128;
            // Tile (0,0)=1.0, (0,1)=2.0, (1,0)=3.0, (1,1)=4.0 (K-major index)
            float tile_max = static_cast<float>((k_blk + n_blk * K_blocks) + 1);
            h_input[n * K + k] = __float2bfloat16(tile_max);
        }
    }

    auto* d_input = to_device(h_input);

    __nv_fp8_e4m3* d_output = nullptr;
    cudaMalloc(&d_output, static_cast<size_t>(N) * K);

    float* d_scales = nullptr;
    cudaMalloc(&d_scales, static_cast<size_t>(N_blocks) * K_blocks * sizeof(float));

    lc::WeightFp8QuantParams params{N, K, d_input, d_output, d_scales};
    lc::launch_weight_fp8_quant(params, nullptr);
    cudaDeviceSynchronize();

    auto h_scales = to_host(d_scales, static_cast<size_t>(N_blocks) * K_blocks);

    // scale = tile_max / 448.0.  Check K-major ordering.
    for (int n_blk = 0; n_blk < N_blocks; ++n_blk) {
        for (int k_blk = 0; k_blk < K_blocks; ++k_blk) {
            int idx = k_blk + n_blk * K_blocks;
            float expected_max = static_cast<float>(idx + 1);
            float expected_scale = expected_max / 448.0f;
            EXPECT_NEAR(h_scales[idx], expected_scale, expected_scale * 0.01f)
                << "Scale mismatch at n_blk=" << n_blk << " k_blk=" << k_blk;
        }
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scales);
}

// Dequantization accuracy: verify round-trip BF16→FP8→BF16 is close.
TEST_F(WeightFp8QuantTest, DequantAccuracy) {
    constexpr int N = 128;
    constexpr int K = 128;

    std::vector<__nv_bfloat16> h_input(N * K);
    for (int i = 0; i < N * K; ++i) {
        float val = (static_cast<float>(i) / (N * K) - 0.5f) * 100.0f;
        h_input[i] = __float2bfloat16(val);
    }

    auto* d_input = to_device(h_input);

    __nv_fp8_e4m3* d_output = nullptr;
    cudaMalloc(&d_output, static_cast<size_t>(N) * K);

    float* d_scales = nullptr;
    cudaMalloc(&d_scales, sizeof(float));

    lc::WeightFp8QuantParams params{N, K, d_input, d_output, d_scales};
    lc::launch_weight_fp8_quant(params, nullptr);
    cudaDeviceSynchronize();

    auto h_output = to_host(d_output, static_cast<size_t>(N) * K);
    auto h_scales = to_host(d_scales, 1);
    float scale = h_scales[0];

    // Dequantize row-major FP8 and compare to original row-major BF16.
    float max_err = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float original = __bfloat162float(h_input[n * K + k]);
            // Row-major: output[n * K + k]
            float dequant = static_cast<float>(h_output[n * K + k]) * scale;
            float err = std::abs(dequant - original);
            max_err = std::max(max_err, err);
        }
    }

    // FP8 E4M3 tile-level quantization: the max error is bounded by the
    // tile scale times the FP8 quantization granularity.  For values spanning
    // [-50, +50], the tile amax is ~50, scale ~0.112.  FP8 E4M3 has 3 mantissa
    // bits, so relative error up to ~12.5% at each exponent level.  The absolute
    // max error should be within scale * 448 * 2^-3 = tile_amax * 2^-3.
    float tile_amax = scale * 448.0f;
    float fp8_max_error = tile_amax * 0.125f;  // 2^-3 for 3-bit mantissa
    EXPECT_LT(max_err, fp8_max_error)
        << "Max dequant error " << max_err << " exceeds FP8 precision bound ("
        << fp8_max_error << ")";
}

// Null pointer rejection.
TEST_F(WeightFp8QuantTest, NullPointerThrows) {
    lc::WeightFp8QuantParams params{128, 128, nullptr, nullptr, nullptr};
    EXPECT_THROW(lc::launch_weight_fp8_quant(params, nullptr), std::runtime_error);
}

// Zero dimensions: should return without error.
TEST_F(WeightFp8QuantTest, ZeroDimensionsNoop) {
    lc::WeightFp8QuantParams params{0, 128, nullptr, nullptr, nullptr};
    EXPECT_NO_THROW(lc::launch_weight_fp8_quant(params, nullptr));
}
