// P-30 step 5 — TD-MOE-BIG-CHUNK-ROWS-KERNEL-CEILING boundary tests.
//
// An 8,192-token MoE big chunk expands to 8192 x topk(8) = 65,536 activation
// rows. Both activation quantizers put the row count in grid.y, whose
// hardware limit is 65,535:
//   - quantize_bf16_to_q8_1 (gguf_mmvq.cu, GGUF int route): the launch failed
//     with cudaErrorInvalidValue and NO local check — the sticky error
//     surfaced at the next checked launch (mhc.cu:540) and killed the daemon.
//   - dynamic_fp8_quant_kernel (dynamic_fp8_quant.cu, FP8 route): same
//     geometry; its launcher checks and throws, which also kills the daemon.
// The fix grid-strides the row dimension (launcher caps grid.y at 65,535);
// for rows <= 65,535 the launch and per-element math are bit-identical to the
// pre-fix kernels — proven here by full-call vs row-slice comparison (each
// slice <= 65,535 rows behaves exactly as the pre-fix kernel did).
//
// These tests FAIL on the pre-fix kernels: the *AtCrashingRowCount tests
// drive the exact 65,536-row boundary shape.

#include "sm120/gemm/gguf/gguf_mmvq.h"
#include "smxx/quant/dynamic_fp8_quant.h"

#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace lc = layerstorm::compute;

namespace {

constexpr int kCrashRows = 65536;    // 8192-token chunk x topk 8 (first bad)
constexpr int kControlRows = 65528;  // 8191-token chunk x topk 8 (last good)

std::vector<__nv_bfloat16> ramp_bf16(size_t n) {
    std::vector<__nv_bfloat16> h(n);
    for (size_t i = 0; i < n; ++i) {
        h[i] = __float2bfloat16(0.125f * static_cast<float>(i % 257) -
                                 16.0f + 0.001f * static_cast<float>(i % 31));
    }
    return h;
}

template <typename T>
T* to_device(const std::vector<T>& host) {
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, host.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}

}  // namespace

class QuantRowsGridCeiling : public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count == 0)
            GTEST_SKIP() << "No CUDA device available";
        cudaSetDevice(0);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);  // clean slate
    }
};

// ── GGUF Q8_1 activation quantizer (the mhc.cu:540 crash) ──────────────────

// The exact crashing shape: 65,536 rows. Pre-fix: grid=(K/32, 65536) is
// rejected with cudaErrorInvalidValue (sticky, unchecked in production).
TEST_F(QuantRowsGridCeiling, GgufQ8_1QuantAtCrashingRowCount) {
    constexpr int M = kCrashRows, K = 64;
    auto h_in = ramp_bf16(static_cast<size_t>(M) * K);
    __nv_bfloat16* d_in = to_device(h_in);
    const size_t ws_bytes = lc::gguf_mmvq_workspace_bytes(M, K);
    void* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, ws_bytes), cudaSuccess);
    // Poison the output so "last row written" is provable.
    ASSERT_EQ(cudaMemset(d_out, 0xFF, ws_bytes), cudaSuccess);

    lc::gguf_quantize_q8_1_cuda(d_in, d_out, M, K, nullptr);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess)
        << "quantize_bf16_to_q8_1 launch failed at " << M
        << " rows (TD-MOE-BIG-CHUNK-ROWS-KERNEL-CEILING)";
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // The final row (index > 65,535) must have been written (not poison).
    const size_t row_bytes = ws_bytes / M;
    std::vector<uint8_t> last(row_bytes);
    ASSERT_EQ(cudaMemcpy(last.data(),
                         static_cast<uint8_t*>(d_out) + (M - 1) * row_bytes,
                         row_bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    bool all_poison = true;
    for (uint8_t b : last) all_poison &= (b == 0xFF);
    EXPECT_FALSE(all_poison) << "row " << (M - 1) << " never written";

    cudaFree(d_in);
    cudaFree(d_out);
}

// Negative control just below the boundary (8191-token chunk): must pass on
// pre-fix and post-fix kernels alike.
TEST_F(QuantRowsGridCeiling, GgufQ8_1QuantBoundaryNegativeControl) {
    constexpr int M = kControlRows, K = 64;
    auto h_in = ramp_bf16(static_cast<size_t>(M) * K);
    __nv_bfloat16* d_in = to_device(h_in);
    void* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, lc::gguf_mmvq_workspace_bytes(M, K)),
              cudaSuccess);

    lc::gguf_quantize_q8_1_cuda(d_in, d_out, M, K, nullptr);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cudaFree(d_in);
    cudaFree(d_out);
}

// Bit-identity: one call at M > 65,535 must equal two row-slice calls, each
// of which (<= 65,535 rows, single grid-stride iteration) is bit-identical
// to the pre-fix kernel. Proves the fix changes nothing where the old kernel
// was legal and is correct beyond the old ceiling.
TEST_F(QuantRowsGridCeiling, GgufQ8_1QuantBigMBitIdenticalToSlices) {
    constexpr int M = 70000, K = 64, M0 = 40000;
    auto h_in = ramp_bf16(static_cast<size_t>(M) * K);
    __nv_bfloat16* d_in = to_device(h_in);
    const size_t ws_bytes = lc::gguf_mmvq_workspace_bytes(M, K);
    const size_t row_bytes = ws_bytes / M;
    void* d_full = nullptr;
    void* d_slice = nullptr;
    ASSERT_EQ(cudaMalloc(&d_full, ws_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_slice, ws_bytes), cudaSuccess);

    lc::gguf_quantize_q8_1_cuda(d_in, d_full, M, K, nullptr);
    lc::gguf_quantize_q8_1_cuda(d_in, d_slice, M0, K, nullptr);
    lc::gguf_quantize_q8_1_cuda(d_in + static_cast<size_t>(M0) * K,
                                static_cast<uint8_t*>(d_slice) + M0 * row_bytes,
                                M - M0, K, nullptr);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint8_t> h_full(ws_bytes), h_slice(ws_bytes);
    ASSERT_EQ(cudaMemcpy(h_full.data(), d_full, ws_bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_slice.data(), d_slice, ws_bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(std::memcmp(h_full.data(), h_slice.data(), ws_bytes), 0)
        << "grid-strided full call diverges from slice-wise (pre-fix) result";

    cudaFree(d_in);
    cudaFree(d_full);
    cudaFree(d_slice);
}

// ── FP8 activation quantizer (same ceiling, throwing launcher) ─────────────

// Pre-fix: launch_dynamic_fp8_quant throws "invalid argument" at 65,536 rows
// (grid.y over the limit) — a config-legal chunk killed the daemon that way.
TEST_F(QuantRowsGridCeiling, Fp8QuantAtCrashingRowCount) {
    constexpr int M = kCrashRows, K = 128;
    auto h_in = ramp_bf16(static_cast<size_t>(M) * K);
    __nv_bfloat16* d_in = to_device(h_in);
    void* d_out = nullptr;
    float* d_scales = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(M) * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_scales, static_cast<size_t>(M) * sizeof(float)),
              cudaSuccess);

    lc::DynamicFp8QuantParams p{};
    p.num_tokens = M;
    p.hidden_size = K;
    p.input = d_in;
    p.output = d_out;
    p.scales = d_scales;
    EXPECT_NO_THROW(lc::launch_dynamic_fp8_quant(p, nullptr))
        << "dynamic_fp8_quant refused " << M
        << " rows (TD-MOE-BIG-CHUNK-ROWS-KERNEL-CEILING)";
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_scales);
}

// Bit-identity of the FP8 quantizer past the old ceiling (row-major scales:
// the slice-wise layout matches; each slice call behaves as pre-fix).
TEST_F(QuantRowsGridCeiling, Fp8QuantBigMBitIdenticalToSlices) {
    constexpr int M = 70000, K = 256, M0 = 40000;
    constexpr int NB = 2;  // ceil(256/128) scale blocks per row
    auto h_in = ramp_bf16(static_cast<size_t>(M) * K);
    __nv_bfloat16* d_in = to_device(h_in);
    void* d_out_full = nullptr;
    void* d_out_slice = nullptr;
    float* d_sc_full = nullptr;
    float* d_sc_slice = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out_full, static_cast<size_t>(M) * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out_slice, static_cast<size_t>(M) * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sc_full, static_cast<size_t>(M) * NB * sizeof(float)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sc_slice, static_cast<size_t>(M) * NB * sizeof(float)),
              cudaSuccess);

    auto quant = [&](const __nv_bfloat16* in, void* out, float* sc, int rows) {
        lc::DynamicFp8QuantParams p{};
        p.num_tokens = rows;
        p.hidden_size = K;
        p.input = in;
        p.output = out;
        p.scales = sc;
        p.m_major_scales = false;
        lc::launch_dynamic_fp8_quant(p, nullptr);
    };
    quant(d_in, d_out_full, d_sc_full, M);
    quant(d_in, d_out_slice, d_sc_slice, M0);
    quant(d_in + static_cast<size_t>(M0) * K,
          static_cast<uint8_t*>(d_out_slice) + static_cast<size_t>(M0) * K,
          d_sc_slice + static_cast<size_t>(M0) * NB, M - M0);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint8_t> f_out(static_cast<size_t>(M) * K),
        s_out(static_cast<size_t>(M) * K);
    std::vector<float> f_sc(static_cast<size_t>(M) * NB),
        s_sc(static_cast<size_t>(M) * NB);
    ASSERT_EQ(cudaMemcpy(f_out.data(), d_out_full, f_out.size(),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(s_out.data(), d_out_slice, s_out.size(),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(f_sc.data(), d_sc_full, f_sc.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(s_sc.data(), d_sc_slice, s_sc.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(std::memcmp(f_out.data(), s_out.data(), f_out.size()), 0)
        << "FP8 payload diverges from slice-wise (pre-fix) result";
    EXPECT_EQ(std::memcmp(f_sc.data(), s_sc.data(),
                          f_sc.size() * sizeof(float)), 0)
        << "FP8 scales diverge from slice-wise (pre-fix) result";

    cudaFree(d_in);
    cudaFree(d_out_full);
    cudaFree(d_out_slice);
    cudaFree(d_sc_full);
    cudaFree(d_sc_slice);
}
