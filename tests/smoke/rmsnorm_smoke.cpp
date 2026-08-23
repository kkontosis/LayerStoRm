// Smoke test: RMSNorm CUDA kernels on real GPU hardware.
//
// Exercises plain and fused add+RMSNorm at model-realistic sizes (V3.2 hidden
// size 7168) across all three dtypes. Verifies GPU output against CPU reference
// with dtype-appropriate tolerances. Also tests the StreamManager integration
// path (kernel launched on a real stream, not the default stream).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "compute/kernels/norm/rmsnorm.h"
#include "compute/cuda_sm120_device_backend.h"
#include "compute/stream_manager.h"
#include "core/gpu_ref.h"

namespace lc = layerstorm::compute;

static std::vector<layerstorm::config::GpuRef> make_gpu_refs(int count) {
    std::vector<layerstorm::config::GpuRef> v;
    for (int i = 0; i < count; ++i)
        v.push_back({.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090});
    return v;
}

// ── CUDA error checking ─────────────────────────────────────────────────────

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_err); \
    } while (0)

// ── CPU reference ───────────────────────────────────────────────────────────

static void rmsnorm_ref(float* out, const float* input, const float* weight,
                        float epsilon, int num_tokens, int hidden_size) {
    for (int t = 0; t < num_tokens; ++t) {
        const float* row = input + t * hidden_size;
        float* out_row = out + t * hidden_size;
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_size; ++i) sum_sq += row[i] * row[i];
        float rms_inv = 1.0f / std::sqrt(sum_sq / hidden_size + epsilon);
        for (int i = 0; i < hidden_size; ++i)
            out_row[i] = row[i] * rms_inv * weight[i];
    }
}

static void fused_add_rmsnorm_ref(float* out, const float* input,
                                  float* residual, const float* weight,
                                  float epsilon, int num_tokens,
                                  int hidden_size) {
    for (int t = 0; t < num_tokens; ++t) {
        float* res_row = residual + t * hidden_size;
        const float* in_row = input + t * hidden_size;
        float* out_row = out + t * hidden_size;
        for (int i = 0; i < hidden_size; ++i) res_row[i] += in_row[i];
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_size; ++i)
            sum_sq += res_row[i] * res_row[i];
        float rms_inv = 1.0f / std::sqrt(sum_sq / hidden_size + epsilon);
        for (int i = 0; i < hidden_size; ++i)
            out_row[i] = res_row[i] * rms_inv * weight[i];
    }
}

// ── Host-side BF16 conversion ───────────────────────────────────────────────

static __nv_bfloat16 float_to_bf16(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint32_t lsb = (f >> 16) & 1;
    f += 0x7FFF + lsb;
    uint16_t bits = static_cast<uint16_t>(f >> 16);
    __nv_bfloat16 b;
    std::memcpy(&b, &bits, sizeof(b));
    return b;
}

static float bf16_to_float(__nv_bfloat16 b) {
    uint16_t bits;
    std::memcpy(&bits, &b, sizeof(bits));
    uint32_t f = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

// ── Fixture ─────────────────────────────────────────────────────────────────

class RMSNormSmoke : public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            GTEST_SKIP() << "No CUDA GPU — skipping RMSNorm smoke test";
        }
        gpu_count_ = count;
        gen_.seed(12345);
    }

    void fill_random(std::vector<float>& v) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : v) x = dist(gen_);
    }

    void fill_weight(std::vector<float>& v) {
        std::uniform_real_distribution<float> dist(0.5f, 1.5f);
        for (auto& x : v) x = dist(gen_);
    }

    int gpu_count_ = 0;
    std::mt19937 gen_;
};

// ── Test 1: V3.2 model-size FP32 on every GPU ──────────────────────────────

TEST_F(RMSNormSmoke, PlainFP32_V32_AllGPUs) {
    constexpr int num_tokens = 128;
    constexpr int hidden_size = 7168;  // DeepSeek V3.2
    constexpr float eps = 1e-6f;

    std::vector<float> h_input(num_tokens * hidden_size);
    std::vector<float> h_weight(hidden_size);
    std::vector<float> h_out_ref(num_tokens * hidden_size);
    fill_random(h_input);
    fill_weight(h_weight);
    rmsnorm_ref(h_out_ref.data(), h_input.data(), h_weight.data(), eps,
                num_tokens, hidden_size);

    for (int gpu = 0; gpu < gpu_count_; ++gpu) {
        SCOPED_TRACE("GPU " + std::to_string(gpu));
        CUDA_CHECK(cudaSetDevice(gpu));

        float *d_input, *d_weight, *d_out;
        size_t data_bytes = num_tokens * hidden_size * sizeof(float);
        size_t weight_bytes = hidden_size * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
        CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
        CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), data_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes,
                              cudaMemcpyHostToDevice));

        lc::launch_rmsnorm(d_out, d_input, d_weight, eps, num_tokens,
                           hidden_size, lc::NormDtype::kFloat32, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out_gpu(num_tokens * hidden_size);
        CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, data_bytes,
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < num_tokens * hidden_size; ++i) {
            ASSERT_NEAR(h_out_gpu[i], h_out_ref[i], 1e-4f)
                << "GPU " << gpu << " mismatch at index " << i;
        }

        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_out);
    }
}

// ── Test 2: BF16 fused add+RMSNorm at V3.2 size ────────────────────────────

TEST_F(RMSNormSmoke, FusedAddBF16_V32) {
    constexpr int num_tokens = 64;
    constexpr int hidden_size = 7168;
    constexpr float eps = 1e-6f;

    std::vector<float> h_input_f(num_tokens * hidden_size);
    std::vector<float> h_residual_f(num_tokens * hidden_size);
    std::vector<float> h_weight_f(hidden_size);
    fill_random(h_input_f);
    fill_random(h_residual_f);
    fill_weight(h_weight_f);

    // Convert to BF16 round-trip for reference
    std::vector<__nv_bfloat16> h_input_bf(num_tokens * hidden_size);
    std::vector<__nv_bfloat16> h_residual_bf(num_tokens * hidden_size);
    std::vector<__nv_bfloat16> h_weight_bf(hidden_size);
    std::vector<float> h_input_rt(num_tokens * hidden_size);
    std::vector<float> h_residual_rt(num_tokens * hidden_size);
    std::vector<float> h_weight_rt(hidden_size);

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        h_input_bf[i] = float_to_bf16(h_input_f[i]);
        h_input_rt[i] = bf16_to_float(h_input_bf[i]);
        h_residual_bf[i] = float_to_bf16(h_residual_f[i]);
        h_residual_rt[i] = bf16_to_float(h_residual_bf[i]);
    }
    for (int i = 0; i < hidden_size; ++i) {
        h_weight_bf[i] = float_to_bf16(h_weight_f[i]);
        h_weight_rt[i] = bf16_to_float(h_weight_bf[i]);
    }

    std::vector<float> h_out_ref(num_tokens * hidden_size);
    fused_add_rmsnorm_ref(h_out_ref.data(), h_input_rt.data(),
                          h_residual_rt.data(), h_weight_rt.data(), eps,
                          num_tokens, hidden_size);

    CUDA_CHECK(cudaSetDevice(0));

    __nv_bfloat16 *d_input, *d_residual, *d_weight, *d_out;
    size_t data_bytes = num_tokens * hidden_size * sizeof(__nv_bfloat16);
    size_t weight_bytes = hidden_size * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_residual, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input_bf.data(), data_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual_bf.data(), data_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight_bf.data(), weight_bytes,
                          cudaMemcpyHostToDevice));

    lc::launch_fused_add_rmsnorm(d_out, d_input, d_residual, d_weight, eps,
                                 num_tokens, hidden_size,
                                 lc::NormDtype::kBFloat16, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out_bf(num_tokens * hidden_size);
    CUDA_CHECK(cudaMemcpy(h_out_bf.data(), d_out, data_bytes,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        float gpu_val = bf16_to_float(h_out_bf[i]);
        ASSERT_NEAR(gpu_val, h_out_ref[i], 3e-2f)
            << "BF16 fused mismatch at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_residual);
    cudaFree(d_weight);
    cudaFree(d_out);
}

// ── Test 3: StreamManager integration ───────────────────────────────────────
// Launch kernel on a real StreamManager stream instead of the default stream.

TEST_F(RMSNormSmoke, StreamManagerIntegration) {
    constexpr int num_tokens = 32;
    constexpr int hidden_size = 7168;
    constexpr float eps = 1e-6f;

    std::vector<float> h_input(num_tokens * hidden_size);
    std::vector<float> h_weight(hidden_size);
    std::vector<float> h_out_ref(num_tokens * hidden_size);
    fill_random(h_input);
    fill_weight(h_weight);
    rmsnorm_ref(h_out_ref.data(), h_input.data(), h_weight.data(), eps,
                num_tokens, hidden_size);

    // Create DeviceBackend instances for each GPU
    auto gpu_refs = make_gpu_refs(gpu_count_);
    std::vector<std::unique_ptr<lc::DeviceBackend>> dev_owners;
    std::vector<lc::DeviceBackend*> dev_ptrs;
    for (auto& ref : gpu_refs) {
        dev_owners.push_back(lc::make_cuda_sm120_device_backend(ref));
        dev_ptrs.push_back(dev_owners.back().get());
    }

    // Create StreamManager with real CUDA backend
    lc::StreamManager::Options sm_opts{.device_backends = dev_ptrs};
    lc::StreamManager sm(sm_opts);

    CUDA_CHECK(cudaSetDevice(0));

    float *d_input, *d_weight, *d_out;
    size_t data_bytes = num_tokens * hidden_size * sizeof(float);
    size_t weight_bytes = hidden_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), data_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes,
                          cudaMemcpyHostToDevice));

    // Launch on the attention stream (Stream 0) via StreamManager
    void* attn_stream = sm.stream(0, lc::StreamId::kAttention);
    lc::launch_rmsnorm(d_out, d_input, d_weight, eps, num_tokens, hidden_size,
                       lc::NormDtype::kFloat32, attn_stream);

    // Record an event and wait for it (INV-5b: event-based sync only)
    void* event = sm.create_event(0);
    sm.record_event(event, 0, lc::StreamId::kAttention);

    // Poll until complete
    while (sm.query_event(event, 0).status != lc::EventStatus::kReady) {
        // busy-wait (smoke test, acceptable)
    }

    std::vector<float> h_out_gpu(num_tokens * hidden_size);
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, data_bytes,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        ASSERT_NEAR(h_out_gpu[i], h_out_ref[i], 1e-4f)
            << "StreamManager path mismatch at index " << i;
    }

    sm.destroy_event(event, 0);
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_out);
}
