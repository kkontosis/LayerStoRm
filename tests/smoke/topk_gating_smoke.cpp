// Smoke test: top-K gating CUDA kernels on real GPU hardware.
//
// Tests all three target model configurations at model-realistic scale:
//   V3.2  — grouped routing (n_group=8, topk_group=4, 256 experts)
//   GLM-5 — simple routing  (n_group=1, 256 experts)
//   K2.5  — simple routing  (n_group=1, 384 experts)
//
// Also tests StreamManager integration path.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "sm120/gating/topk_gating.h"
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

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: "                         \
                                     << cudaGetErrorString(_err);              \
    } while (0)

// ── CPU reference ───────────────────────────────────────────────────────────

static float sigmoid_ref(float x) {
    return 0.5f * std::tanh(0.5f * x) + 0.5f;
}

static void topk_gating_ref(
    float* weights, int32_t* indices, const float* logits, const float* bias,
    int num_tokens, int num_experts, int n_group, int topk_group, int topk,
    float routed_scaling_factor, bool renormalize) {

    const int experts_per_group = num_experts / n_group;

    for (int t = 0; t < num_tokens; ++t) {
        const float* tok_logits = logits + t * num_experts;
        std::vector<float> scores(num_experts), sel(num_experts);
        for (int e = 0; e < num_experts; ++e) {
            scores[e] = sigmoid_ref(tok_logits[e]);
            sel[e] = bias ? scores[e] + bias[e] : scores[e];
        }

        if (n_group > 1) {
            std::vector<float> gs(n_group);
            for (int g = 0; g < n_group; ++g) {
                int s = g * experts_per_group;
                float t1 = -FLT_MAX, t2 = -FLT_MAX;
                for (int i = 0; i < experts_per_group; ++i) {
                    float v = sel[s + i];
                    if (v > t1) { t2 = t1; t1 = v; }
                    else if (v > t2) { t2 = v; }
                }
                gs[g] = t1 + t2;
            }
            std::vector<bool> gsel(n_group, false);
            for (int g = 0; g < topk_group; ++g) {
                float b = -FLT_MAX; int bg = 0;
                for (int i = 0; i < n_group; ++i)
                    if (!gsel[i] && gs[i] > b) { b = gs[i]; bg = i; }
                gsel[bg] = true;
            }
            for (int e = 0; e < num_experts; ++e)
                if (!gsel[e / experts_per_group]) sel[e] = -FLT_MAX;
        }

        float* ow = weights + t * topk;
        int32_t* oi = indices + t * topk;
        for (int k = 0; k < topk; ++k) {
            float bv = -FLT_MAX; int be = 0;
            for (int e = 0; e < num_experts; ++e)
                if (sel[e] > bv || (sel[e] == bv && e < be)) { bv = sel[e]; be = e; }
            oi[k] = be; ow[k] = scores[be]; sel[be] = -FLT_MAX;
        }

        if (renormalize) {
            float sum = 0; for (int k = 0; k < topk; ++k) sum += ow[k];
            if (sum > 0) { float s = routed_scaling_factor / sum;
                for (int k = 0; k < topk; ++k) ow[k] *= s; }
        }
    }
}

// ── Fixture ─────────────────────────────────────────────────────────────────

class TopkGatingSmoke : public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            GTEST_SKIP() << "No CUDA GPU — skipping gating smoke test";
        gpu_count_ = count;
        gen_.seed(54321);
    }

    void fill_separated(std::vector<float>& v, int T, int E,
                        float range = 4.0f) {
        v.resize(static_cast<size_t>(T) * E);
        const float step = range / E;
        const float noise_amp = step * 0.01f;
        std::uniform_real_distribution<float> noise(-noise_amp, noise_amp);
        for (int t = 0; t < T; ++t) {
            std::vector<int> perm(E);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), gen_);
            for (int e = 0; e < E; ++e)
                v[t * E + e] =
                    (static_cast<float>(perm[e]) - E * 0.5f) * step +
                    noise(gen_);
        }
    }

    void fill_bias(std::vector<float>& v, int E) {
        v.resize(E);
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        for (auto& x : v) x = dist(gen_);
    }

    int gpu_count_ = 0;
    std::mt19937 gen_;
};

// ── Helper ──────────────────────────────────────────────────────────────────

static void verify_against_ref(
    const std::vector<float>& gpu_w, const std::vector<int32_t>& gpu_i,
    const std::vector<float>& ref_w, const std::vector<int32_t>& ref_i,
    int T, int K, float tol = 1e-4f) {
    for (int t = 0; t < T; ++t) {
        std::set<int> rs, gs;
        for (int k = 0; k < K; ++k) {
            rs.insert(ref_i[t * K + k]);
            gs.insert(gpu_i[t * K + k]);
        }
        ASSERT_EQ(rs, gs) << "Token " << t << ": expert sets differ";

        for (int k = 0; k < K; ++k) {
            int eid = gpu_i[t * K + k];
            float rw = 0;
            for (int j = 0; j < K; ++j)
                if (ref_i[t * K + j] == eid) { rw = ref_w[t * K + j]; break; }
            ASSERT_NEAR(gpu_w[t * K + k], rw, tol)
                << "Token " << t << " expert " << eid;
        }
    }
}

// ── Test 1: V3.2 grouped routing on every GPU ──────────────────────────────

TEST_F(TopkGatingSmoke, V32_Grouped_AllGPUs) {
    constexpr int T = 128, E = 256, K = 8;
    constexpr int G = 8, TG = 4;
    constexpr float scaling = 2.5f;

    std::vector<float> logits, bias;
    fill_separated(logits, T, E);
    fill_bias(bias, E);

    std::vector<float> ref_w(T * K);
    std::vector<int32_t> ref_i(T * K);
    topk_gating_ref(ref_w.data(), ref_i.data(), logits.data(), bias.data(),
                    T, E, G, TG, K, scaling, true);

    for (int gpu = 0; gpu < gpu_count_; ++gpu) {
        SCOPED_TRACE("GPU " + std::to_string(gpu));
        CUDA_CHECK(cudaSetDevice(gpu));

        float *d_logits, *d_bias, *d_weights;
        int32_t* d_indices;
        CUDA_CHECK(cudaMalloc(&d_logits, T * E * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias, E * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights, T * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_indices, T * K * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), T * E * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), E * sizeof(float),
                              cudaMemcpyHostToDevice));

        lc::TopkGatingParams params{T, E, K, G, TG, scaling, true};
        lc::launch_topk_gating(d_weights, d_indices, d_logits, d_bias, params,
                               nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> gpu_w(T * K);
        std::vector<int32_t> gpu_i(T * K);
        CUDA_CHECK(cudaMemcpy(gpu_w.data(), d_weights, T * K * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_i.data(), d_indices, T * K * sizeof(int32_t),
                              cudaMemcpyDeviceToHost));

        verify_against_ref(gpu_w, gpu_i, ref_w, ref_i, T, K);

        cudaFree(d_logits);
        cudaFree(d_bias);
        cudaFree(d_weights);
        cudaFree(d_indices);
    }
}

// ── Test 2: GLM-5 simple routing (256 experts, n_group=1) ──────────────────

TEST_F(TopkGatingSmoke, GLM5_Simple_256) {
    constexpr int T = 128, E = 256, K = 8;
    constexpr float scaling = 2.5f;

    std::vector<float> logits, bias;
    fill_separated(logits, T, E);
    fill_bias(bias, E);

    std::vector<float> ref_w(T * K);
    std::vector<int32_t> ref_i(T * K);
    topk_gating_ref(ref_w.data(), ref_i.data(), logits.data(), bias.data(),
                    T, E, 1, 1, K, scaling, true);

    CUDA_CHECK(cudaSetDevice(0));

    float *d_logits, *d_bias, *d_weights;
    int32_t* d_indices;
    CUDA_CHECK(cudaMalloc(&d_logits, T * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, T * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, T * K * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), T * E * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), E * sizeof(float),
                          cudaMemcpyHostToDevice));

    lc::TopkGatingParams params{T, E, K, 1, 1, scaling, true};
    lc::launch_topk_gating(d_weights, d_indices, d_logits, d_bias, params,
                           nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_w(T * K);
    std::vector<int32_t> gpu_i(T * K);
    CUDA_CHECK(cudaMemcpy(gpu_w.data(), d_weights, T * K * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_i.data(), d_indices, T * K * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    verify_against_ref(gpu_w, gpu_i, ref_w, ref_i, T, K);

    cudaFree(d_logits);
    cudaFree(d_bias);
    cudaFree(d_weights);
    cudaFree(d_indices);
}

// ── Test 3: K2.5 simple routing (384 experts, scaling=2.827) ────────────────

TEST_F(TopkGatingSmoke, K25_Simple_384) {
    constexpr int T = 64, E = 384, K = 8;
    constexpr float scaling = 2.827f;

    std::vector<float> logits, bias;
    fill_separated(logits, T, E);
    fill_bias(bias, E);

    std::vector<float> ref_w(T * K);
    std::vector<int32_t> ref_i(T * K);
    topk_gating_ref(ref_w.data(), ref_i.data(), logits.data(), bias.data(),
                    T, E, 1, 1, K, scaling, true);

    CUDA_CHECK(cudaSetDevice(0));

    float *d_logits, *d_bias, *d_weights;
    int32_t* d_indices;
    CUDA_CHECK(cudaMalloc(&d_logits, T * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, T * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, T * K * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), T * E * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), E * sizeof(float),
                          cudaMemcpyHostToDevice));

    lc::TopkGatingParams params{T, E, K, 1, 1, scaling, true};
    lc::launch_topk_gating(d_weights, d_indices, d_logits, d_bias, params,
                           nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_w(T * K);
    std::vector<int32_t> gpu_i(T * K);
    CUDA_CHECK(cudaMemcpy(gpu_w.data(), d_weights, T * K * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_i.data(), d_indices, T * K * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    verify_against_ref(gpu_w, gpu_i, ref_w, ref_i, T, K);

    cudaFree(d_logits);
    cudaFree(d_bias);
    cudaFree(d_weights);
    cudaFree(d_indices);
}

// ── Test 4: StreamManager integration ───────────────────────────────────────

TEST_F(TopkGatingSmoke, StreamManagerIntegration) {
    constexpr int T = 32, E = 256, K = 8;
    constexpr float scaling = 2.5f;

    std::vector<float> logits;
    fill_separated(logits, T, E);

    std::vector<float> ref_w(T * K);
    std::vector<int32_t> ref_i(T * K);
    topk_gating_ref(ref_w.data(), ref_i.data(), logits.data(), nullptr,
                    T, E, 1, 1, K, scaling, true);

    // Create DeviceBackend instances for each GPU.
    auto gpu_refs = make_gpu_refs(gpu_count_);
    std::vector<std::unique_ptr<lc::DeviceBackend>> dev_owners;
    std::vector<lc::DeviceBackend*> dev_ptrs;
    for (auto& ref : gpu_refs) {
        dev_owners.push_back(lc::make_cuda_sm120_device_backend(ref));
        dev_ptrs.push_back(dev_owners.back().get());
    }

    lc::StreamManager::Options sm_opts{.device_backends = dev_ptrs};
    lc::StreamManager sm(sm_opts);

    CUDA_CHECK(cudaSetDevice(0));

    float *d_logits, *d_weights;
    int32_t* d_indices;
    CUDA_CHECK(cudaMalloc(&d_logits, T * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, T * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, T * K * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), T * E * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch on gating stream (Stream 2) via StreamManager.
    void* gating_stream = sm.stream(0, lc::StreamId::kGating);
    lc::TopkGatingParams params{T, E, K, 1, 1, scaling, true};
    lc::launch_topk_gating(d_weights, d_indices, d_logits, nullptr, params,
                           gating_stream);

    // Event-based sync (INV-5b).
    void* event = sm.create_event(0);
    sm.record_event(event, 0, lc::StreamId::kGating);
    while (sm.query_event(event, 0).status != lc::EventStatus::kReady) {}

    std::vector<float> gpu_w(T * K);
    std::vector<int32_t> gpu_i(T * K);
    CUDA_CHECK(cudaMemcpy(gpu_w.data(), d_weights, T * K * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_i.data(), d_indices, T * K * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    verify_against_ref(gpu_w, gpu_i, ref_w, ref_i, T, K);

    sm.destroy_event(event, 0);
    cudaFree(d_logits);
    cudaFree(d_weights);
    cudaFree(d_indices);
}
