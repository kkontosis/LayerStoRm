// Unit tests for kv_bv_extract_dequant CUDA kernel (GPU-gated).
// Validates FP8 E4M3 → BF16 dequantization against a CPU reference
// for DeepSeek V3.2 dims (P==V==128) and GLM-5 dims (P=192, V=256).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "compute/kernels/smxx/quant/kv_bv_extract_dequant.h"
#include "model/quantization/fp8.h"
#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;
namespace lm = layerstorm::model;

// ── CUDA error checking ─────────────────────────────────────────────────────

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: "                         \
                                     << cudaGetErrorString(_err);              \
    } while (0)

// ── Host BF16 conversion helpers ────────────────────────────────────────────

static float bf16_to_float(__nv_bfloat16 b) {
    uint16_t bits;
    std::memcpy(&bits, &b, sizeof(bits));
    uint32_t f = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

// ── FP8 E4M3 host encode (matches grouped_gemm_test.cpp) ───────────────────

static uint8_t encode_fp8_e4m3(float val) {
    if (std::isnan(val)) return 0x7F;
    uint8_t sign = val < 0 ? 1 : 0;
    float abs_val = std::fabs(val);
    if (abs_val == 0.0f) return 0;
    if (abs_val > 448.0f) abs_val = 448.0f;

    float log2_val = std::log2(abs_val);
    int exp_unbiased = static_cast<int>(std::floor(log2_val));
    exp_unbiased = std::max(exp_unbiased, -6);
    exp_unbiased = std::min(exp_unbiased, 8);

    float mantissa_f = abs_val / std::ldexp(1.0f, exp_unbiased) - 1.0f;
    int mant = static_cast<int>(std::round(mantissa_f * 8.0f));
    mant = std::min(mant, 7);

    int biased_exp = exp_unbiased + 7;
    biased_exp = std::max(biased_exp, 0);
    biased_exp = std::min(biased_exp, 15);

    return static_cast<uint8_t>((sign << 7) | (biased_exp << 3) | mant);
}

static float decode_fp8_e4m3(uint8_t bits) {
    return lm::fp8_e4m3::decode(bits);
}

// ── CPU reference implementation ────────────────────────────────────────────

// Computes the same extraction + dequant as the CUDA kernel:
//   For each (head h, v_row v, col c):
//     src_row = h*(P+V) + P + v_row
//     fp8_val = fp8_src[src_row * D_c + c]
//     n_block = src_row / 128
//     k_block = c / 128
//     N_scale_blocks = ceil(N_total / 128)
//     output[h*V*D_c + v*D_c + c] = decode(fp8_val) * scales[k_block * N_scale_blocks + n_block]

static void kv_bv_extract_dequant_ref(
    std::vector<float>& output,
    const std::vector<uint8_t>& fp8_src,
    const std::vector<float>& scales,
    int HL, int P, int V, int D_c)
{
    const int N_total = HL * (P + V);
    const int N_scale_blocks = (N_total + 127) / 128;

    output.resize(HL * V * D_c);

    for (int h = 0; h < HL; ++h) {
        for (int v = 0; v < V; ++v) {
            const int src_row = h * (P + V) + P + v;
            const int n_block = src_row / 128;
            const int dst_idx_base = h * V * D_c + v * D_c;

            for (int c = 0; c < D_c; ++c) {
                float val = decode_fp8_e4m3(fp8_src[src_row * D_c + c]);
                int k_block = c / 128;
                float scale = scales[k_block * N_scale_blocks + n_block];
                output[dst_idx_base + c] = val * scale;
            }
        }
    }
}

// ── Helper: run kernel and compare ──────────────────────────────────────────

static void run_and_compare(int HL, int P, int V, int D_c,
                            const std::vector<uint8_t>& h_fp8,
                            const std::vector<float>& h_scales,
                            float tolerance) {
    const int N_total = HL * (P + V);
    const int N_scale_blocks = (N_total + 127) / 128;
    const int K_scale_blocks = (D_c + 127) / 128;
    const size_t output_elems = HL * V * D_c;

    ASSERT_EQ(h_fp8.size(), static_cast<size_t>(N_total) * D_c);
    ASSERT_EQ(h_scales.size(), static_cast<size_t>(K_scale_blocks) * N_scale_blocks);

    // CPU reference.
    std::vector<float> ref;
    kv_bv_extract_dequant_ref(ref, h_fp8, h_scales, HL, P, V, D_c);

    // Device memory.
    void* d_fp8 = nullptr;
    void* d_scales = nullptr;
    void* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fp8, h_fp8.size()));
    CUDA_CHECK(cudaMalloc(&d_scales, h_scales.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_elems * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemcpy(d_fp8, h_fp8.data(), h_fp8.size(),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(),
                           h_scales.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Launch kernel.
    lc::KvBvExtractDequantParams params{};
    params.num_heads_local = HL;
    params.qk_nope_head_dim = P;
    params.v_head_dim = V;
    params.kv_lora_rank = D_c;
    params.kv_b_proj = d_fp8;
    params.scales = d_scales;
    params.output = d_output;

    lc::launch_kv_bv_extract_dequant(params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back.
    std::vector<__nv_bfloat16> h_output(output_elems);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                           output_elems * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost));

    // Compare.
    int mismatches = 0;
    for (size_t i = 0; i < output_elems; ++i) {
        float gpu_val = bf16_to_float(h_output[i]);
        float ref_val = ref[i];
        float diff = std::fabs(gpu_val - ref_val);
        // Use relative tolerance for non-zero values, absolute for zero.
        float denom = std::max(std::fabs(ref_val), 1e-6f);
        if (diff / denom > tolerance && diff > tolerance) {
            if (mismatches < 5) {
                ADD_FAILURE() << "Mismatch at index " << i
                              << ": gpu=" << gpu_val << " ref=" << ref_val
                              << " diff=" << diff;
            }
            ++mismatches;
        }
    }
    if (mismatches > 5) {
        ADD_FAILURE() << "Total mismatches: " << mismatches
                      << " / " << output_elems;
    }

    cudaFree(d_fp8);
    cudaFree(d_scales);
    cudaFree(d_output);
}

// ── Tests ───────────────────────────────────────────────────────────────────

TEST(KvBvExtractDequantTest, V32Dims) {
    REQUIRES_GPU();

    const int HL = 2, P = 128, V = 128, D_c = 512;
    const int N_total = HL * (P + V);
    const int N_scale_blocks = (N_total + 127) / 128;
    const int K_scale_blocks = (D_c + 127) / 128;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scale_dist(0.5f, 2.0f);

    // Generate random FP8 input.
    std::vector<uint8_t> h_fp8(N_total * D_c);
    for (auto& v : h_fp8)
        v = encode_fp8_e4m3(val_dist(rng));

    // Generate random scales.
    std::vector<float> h_scales(K_scale_blocks * N_scale_blocks);
    for (auto& s : h_scales)
        s = scale_dist(rng);

    run_and_compare(HL, P, V, D_c, h_fp8, h_scales, 1e-2f);
}

TEST(KvBvExtractDequantTest, GLM5Dims) {
    REQUIRES_GPU();

    // GLM-5: P=192, V=256 — V rows span multiple scale blocks per head.
    const int HL = 2, P = 192, V = 256, D_c = 512;
    const int N_total = HL * (P + V);
    const int N_scale_blocks = (N_total + 127) / 128;
    const int K_scale_blocks = (D_c + 127) / 128;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> val_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> scale_dist(0.5f, 2.0f);

    std::vector<uint8_t> h_fp8(N_total * D_c);
    for (auto& v : h_fp8)
        v = encode_fp8_e4m3(val_dist(rng));

    std::vector<float> h_scales(K_scale_blocks * N_scale_blocks);
    for (auto& s : h_scales)
        s = scale_dist(rng);

    run_and_compare(HL, P, V, D_c, h_fp8, h_scales, 1e-2f);
}

TEST(KvBvExtractDequantTest, SingleHead) {
    REQUIRES_GPU();

    const int HL = 1, P = 128, V = 128, D_c = 128;
    const int N_total = HL * (P + V);
    const int N_scale_blocks = (N_total + 127) / 128;
    const int K_scale_blocks = (D_c + 127) / 128;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scale_dist(0.8f, 1.2f);

    std::vector<uint8_t> h_fp8(N_total * D_c);
    for (auto& v : h_fp8)
        v = encode_fp8_e4m3(val_dist(rng));

    std::vector<float> h_scales(K_scale_blocks * N_scale_blocks);
    for (auto& s : h_scales)
        s = scale_dist(rng);

    run_and_compare(HL, P, V, D_c, h_fp8, h_scales, 1e-2f);
}

TEST(KvBvExtractDequantTest, AllZeroInput) {
    REQUIRES_GPU();

    const int HL = 2, P = 128, V = 128, D_c = 256;
    const int N_total = HL * (P + V);
    const int N_scale_blocks = (N_total + 127) / 128;
    const int K_scale_blocks = (D_c + 127) / 128;

    // All FP8 bytes = 0 (encodes 0.0).
    std::vector<uint8_t> h_fp8(N_total * D_c, 0);

    // Arbitrary non-zero scales.
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> scale_dist(1.0f, 3.0f);
    std::vector<float> h_scales(K_scale_blocks * N_scale_blocks);
    for (auto& s : h_scales)
        s = scale_dist(rng);

    // CPU reference — all zeros regardless of scales.
    std::vector<float> ref;
    kv_bv_extract_dequant_ref(ref, h_fp8, h_scales, HL, P, V, D_c);
    for (float v : ref)
        ASSERT_EQ(v, 0.0f);

    run_and_compare(HL, P, V, D_c, h_fp8, h_scales, 0.0f);
}

TEST(KvBvExtractDequantTest, DcNotAligned) {
    REQUIRES_GPU();

    // D_c=100 is not divisible by 128 (scale block size).
    // Tests partial scale block usage: K_scale_blocks=1 but only 100 of 128 cols used.
    const int HL = 2, P = 128, V = 128, D_c = 100;
    const int N_total = HL * (P + V);
    const int N_scale_blocks = (N_total + 127) / 128;
    const int K_scale_blocks = (D_c + 127) / 128;

    std::mt19937 rng(77);
    std::uniform_real_distribution<float> val_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> scale_dist(0.5f, 2.0f);

    std::vector<uint8_t> h_fp8(N_total * D_c);
    for (auto& v : h_fp8)
        v = encode_fp8_e4m3(val_dist(rng));

    std::vector<float> h_scales(K_scale_blocks * N_scale_blocks);
    for (auto& s : h_scales)
        s = scale_dist(rng);

    run_and_compare(HL, P, V, D_c, h_fp8, h_scales, 1e-2f);
}

TEST(KvBvExtractDequantTest, UnityScale) {
    REQUIRES_GPU();

    const int HL = 2, P = 128, V = 128, D_c = 256;
    const int N_total = HL * (P + V);
    const int N_scale_blocks = (N_total + 127) / 128;
    const int K_scale_blocks = (D_c + 127) / 128;

    std::mt19937 rng(55);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);

    std::vector<uint8_t> h_fp8(N_total * D_c);
    for (auto& v : h_fp8)
        v = encode_fp8_e4m3(val_dist(rng));

    // All scales = 1.0.
    std::vector<float> h_scales(K_scale_blocks * N_scale_blocks, 1.0f);

    run_and_compare(HL, P, V, D_c, h_fp8, h_scales, 1e-3f);
}
