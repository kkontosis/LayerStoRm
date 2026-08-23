// Smoke test: Embedding lookup and output head CUDA kernels on real GPU hardware.
//
// Exercises embedding lookup at V3.2 model sizes across all GPUs, output head
// GEMM at full vocabulary dimensions, and StreamManager integration.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "compute/kernels/embedding/embedding.h"
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

// ── Host BF16 conversion ────────────────────────────────────────────────────

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

// ── CPU reference ───────────────────────────────────────────────────────────

static void embedding_lookup_ref(float* out, const float* table,
                                 const int32_t* token_ids, int num_tokens,
                                 int vocab_size, int hidden_size) {
    for (int t = 0; t < num_tokens; ++t) {
        int tok = token_ids[t];
        float* dst = out + static_cast<int64_t>(t) * hidden_size;
        if (tok < 0 || tok >= vocab_size) {
            for (int h = 0; h < hidden_size; ++h) dst[h] = 0.0f;
        } else {
            const float* src =
                table + static_cast<int64_t>(tok) * hidden_size;
            for (int h = 0; h < hidden_size; ++h) dst[h] = src[h];
        }
    }
}

// ── Fixture ─────────────────────────────────────────────────────────────────

class EmbeddingSmoke : public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            GTEST_SKIP() << "No CUDA GPU — skipping embedding smoke test";
        gpu_count_ = count;
        gen_.seed(12345);
    }

    void fill_random(std::vector<float>& v, float lo = -1.0f,
                     float hi = 1.0f) {
        std::uniform_real_distribution<float> dist(lo, hi);
        for (auto& x : v) x = dist(gen_);
    }

    void fill_token_ids(std::vector<int32_t>& ids, int vocab_size) {
        std::uniform_int_distribution<int32_t> dist(0, vocab_size - 1);
        for (auto& x : ids) x = dist(gen_);
    }

    int gpu_count_ = 0;
    std::mt19937 gen_;
};

// ── Test 1: V3.2 embedding lookup on every GPU (BF16) ──────────────────────

TEST_F(EmbeddingSmoke, EmbeddingLookup_V32_AllGPUs) {
    constexpr int num_tokens = 128;
    constexpr int vocab_size = 129280;  // DeepSeek V3.2
    constexpr int hidden_size = 7168;

    // Build BF16 table and token IDs on host.
    const int64_t table_elems =
        static_cast<int64_t>(vocab_size) * hidden_size;
    std::vector<__nv_bfloat16> h_table_bf(table_elems);
    {
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        for (int64_t i = 0; i < table_elems; ++i)
            h_table_bf[i] = float_to_bf16(dist(gen_));
    }

    std::vector<int32_t> h_ids(num_tokens);
    fill_token_ids(h_ids, vocab_size);

    // CPU reference in float (round-trip through BF16 for selected rows only).
    auto ref_for_token = [&](int token_idx) -> std::vector<float> {
        int tok = h_ids[token_idx];
        std::vector<float> row(hidden_size);
        for (int h = 0; h < hidden_size; ++h)
            row[h] = bf16_to_float(
                h_table_bf[static_cast<int64_t>(tok) * hidden_size + h]);
        return row;
    };

    for (int gpu = 0; gpu < gpu_count_; ++gpu) {
        SCOPED_TRACE("GPU " + std::to_string(gpu));
        CUDA_CHECK(cudaSetDevice(gpu));

        __nv_bfloat16* d_table = nullptr;
        __nv_bfloat16* d_out = nullptr;
        int32_t* d_ids = nullptr;
        size_t table_bytes = table_elems * sizeof(__nv_bfloat16);
        size_t out_bytes =
            static_cast<size_t>(num_tokens) * hidden_size * sizeof(__nv_bfloat16);
        size_t ids_bytes = num_tokens * sizeof(int32_t);

        CUDA_CHECK(cudaMalloc(&d_table, table_bytes));
        CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
        CUDA_CHECK(cudaMalloc(&d_ids, ids_bytes));
        CUDA_CHECK(cudaMemcpy(d_table, h_table_bf.data(), table_bytes,
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), ids_bytes,
                               cudaMemcpyHostToDevice));

        lc::launch_embedding_lookup(d_out, d_table, d_ids, num_tokens,
                                    vocab_size, hidden_size,
                                    lc::EmbeddingDtype::kBFloat16, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Spot-check first 4 tokens fully.
        std::vector<__nv_bfloat16> h_out_bf(
            static_cast<size_t>(num_tokens) * hidden_size);
        CUDA_CHECK(cudaMemcpy(h_out_bf.data(), d_out, out_bytes,
                               cudaMemcpyDeviceToHost));

        for (int t = 0; t < std::min(num_tokens, 4); ++t) {
            auto ref_row = ref_for_token(t);
            for (int h = 0; h < hidden_size; ++h) {
                float gpu_val = bf16_to_float(
                    h_out_bf[static_cast<int64_t>(t) * hidden_size + h]);
                ASSERT_NEAR(gpu_val, ref_row[h], 1e-6f)
                    << "GPU " << gpu << " token=" << t << " h=" << h;
            }
        }

        cudaFree(d_table);
        cudaFree(d_out);
        cudaFree(d_ids);
    }
}

// ── Test 2: V3.2 output head BF16 GEMM ─────────────────────────────────────

TEST_F(EmbeddingSmoke, OutputHead_V32_BF16) {
    constexpr int num_tokens = 4;
    constexpr int vocab_size = 129280;
    constexpr int hidden_size = 7168;

    const int64_t weight_elems =
        static_cast<int64_t>(vocab_size) * hidden_size;
    const int64_t hidden_elems =
        static_cast<int64_t>(num_tokens) * hidden_size;
    const int64_t logit_elems =
        static_cast<int64_t>(num_tokens) * vocab_size;

    std::vector<__nv_bfloat16> h_weight_bf(weight_elems);
    std::vector<__nv_bfloat16> h_hidden_bf(hidden_elems);
    {
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        for (int64_t i = 0; i < weight_elems; ++i)
            h_weight_bf[i] = float_to_bf16(dist(gen_));
        for (int64_t i = 0; i < hidden_elems; ++i)
            h_hidden_bf[i] = float_to_bf16(dist(gen_));
    }

    CUDA_CHECK(cudaSetDevice(0));

    __nv_bfloat16 *d_hidden, *d_weight;
    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_hidden, hidden_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_logits, logit_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden_bf.data(),
                           hidden_elems * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight_bf.data(),
                           weight_elems * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));

    lc::launch_output_head(d_logits, d_hidden, d_weight, nullptr, num_tokens,
                           vocab_size, hidden_size,
                           lc::EmbeddingDtype::kBFloat16, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Spot-check first 32 logits per token against CPU reference.
    constexpr int check_count = 32;
    std::vector<float> h_logits(logit_elems);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits,
                           logit_elems * sizeof(float),
                           cudaMemcpyDeviceToHost));

    for (int t = 0; t < num_tokens; ++t) {
        for (int v = 0; v < check_count; ++v) {
            double sum = 0.0;
            for (int h = 0; h < hidden_size; ++h) {
                float hv = bf16_to_float(
                    h_hidden_bf[static_cast<int64_t>(t) * hidden_size + h]);
                float wv = bf16_to_float(
                    h_weight_bf[static_cast<int64_t>(v) * hidden_size + h]);
                sum += static_cast<double>(hv) * static_cast<double>(wv);
            }
            float gpu_val =
                h_logits[static_cast<int64_t>(t) * vocab_size + v];
            ASSERT_NEAR(gpu_val, static_cast<float>(sum), 5e-1f)
                << "token=" << t << " vocab=" << v;
        }
    }

    cudaFree(d_hidden);
    cudaFree(d_weight);
    cudaFree(d_logits);
}

// ── Test 3: StreamManager integration ───────────────────────────────────────
// Launch embedding lookup on a real StreamManager stream, sync via event.

TEST_F(EmbeddingSmoke, StreamManagerIntegration) {
    constexpr int num_tokens = 32;
    constexpr int vocab_size = 256;
    constexpr int hidden_size = 7168;

    std::vector<float> h_table(
        static_cast<int64_t>(vocab_size) * hidden_size);
    std::vector<int32_t> h_ids(num_tokens);
    fill_random(h_table);
    fill_token_ids(h_ids, vocab_size);

    std::vector<float> h_ref(
        static_cast<int64_t>(num_tokens) * hidden_size);
    embedding_lookup_ref(h_ref.data(), h_table.data(), h_ids.data(),
                         num_tokens, vocab_size, hidden_size);

    // Create DeviceBackend instances for each GPU.
    auto gpu_refs = make_gpu_refs(gpu_count_);
    std::vector<std::unique_ptr<lc::DeviceBackend>> dev_owners;
    std::vector<lc::DeviceBackend*> dev_ptrs;
    for (auto& ref : gpu_refs) {
        dev_owners.push_back(lc::make_cuda_sm120_device_backend(ref));
        dev_ptrs.push_back(dev_owners.back().get());
    }

    // Create StreamManager with real CUDA backend.
    lc::StreamManager::Options sm_opts{.device_backends = dev_ptrs};
    lc::StreamManager sm(sm_opts);

    CUDA_CHECK(cudaSetDevice(0));

    float *d_table, *d_out;
    int32_t* d_ids;
    size_t table_bytes =
        static_cast<size_t>(vocab_size) * hidden_size * sizeof(float);
    size_t out_bytes =
        static_cast<size_t>(num_tokens) * hidden_size * sizeof(float);
    size_t ids_bytes = num_tokens * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_table, table_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_ids, ids_bytes));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), table_bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_ids, h_ids.data(), ids_bytes, cudaMemcpyHostToDevice));

    // Launch on the attention stream (Stream 0) via StreamManager.
    void* attn_stream = sm.stream(0, lc::StreamId::kAttention);
    lc::launch_embedding_lookup(d_out, d_table, d_ids, num_tokens, vocab_size,
                                hidden_size, lc::EmbeddingDtype::kFloat32,
                                attn_stream);

    // Record an event and poll until complete (INV-5b: event-based sync).
    void* event = sm.create_event(0);
    sm.record_event(event, 0, lc::StreamId::kAttention);
    while (sm.query_event(event, 0).status != lc::EventStatus::kReady) {
        // busy-wait (acceptable in smoke test)
    }

    std::vector<float> h_out(
        static_cast<int64_t>(num_tokens) * hidden_size);
    CUDA_CHECK(
        cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    for (int64_t i = 0;
         i < static_cast<int64_t>(num_tokens) * hidden_size; ++i)
        ASSERT_NEAR(h_out[i], h_ref[i], 1e-4f)
            << "StreamManager path mismatch at " << i;

    sm.destroy_event(event, 0);
    cudaFree(d_table);
    cudaFree(d_out);
    cudaFree(d_ids);
}
