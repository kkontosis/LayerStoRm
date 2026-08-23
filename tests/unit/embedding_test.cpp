// Unit tests for embedding lookup and output head (LM head) CUDA kernels.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "compute/kernels/embedding/embedding.h"
#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;

// ── CUDA error checking ─────────────────────────────────────────────────────

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: "                         \
                                     << cudaGetErrorString(_err);              \
    } while (0)

// ── Host BF16/FP16 conversion helpers ───────────────────────────────────────

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

static __half float_to_fp16(float v) {
    __half h;
    uint16_t bits;
    // Use the standard conversion: round to nearest even.
    // Simplified: truncate via bit manipulation.
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127;
    uint32_t mantissa = f & 0x7FFFFF;
    if (exp > 15) {
        bits = static_cast<uint16_t>(sign | 0x7C00);  // inf
    } else if (exp < -14) {
        bits = static_cast<uint16_t>(sign);  // zero/denorm
    } else {
        bits = static_cast<uint16_t>(sign | ((exp + 15) << 10) |
                                     (mantissa >> 13));
    }
    std::memcpy(&h, &bits, sizeof(h));
    return h;
}

static float fp16_to_float(__half h) {
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000) << 16;
    uint32_t exp = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x03FF;
    uint32_t f;
    if (exp == 0) {
        f = sign;  // zero/denorm → zero for simplicity
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000 | (mantissa << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

// ── CPU reference implementations ───────────────────────────────────────────

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

static void output_head_ref(float* logits, const float* hidden,
                            const float* weight, const float* bias,
                            int num_tokens, int vocab_size, int hidden_size) {
    for (int t = 0; t < num_tokens; ++t) {
        for (int v = 0; v < vocab_size; ++v) {
            double sum = 0.0;
            for (int h = 0; h < hidden_size; ++h)
                sum += static_cast<double>(hidden[t * hidden_size + h]) *
                       static_cast<double>(
                           weight[static_cast<int64_t>(v) * hidden_size + h]);
            if (bias) sum += bias[v];
            logits[static_cast<int64_t>(t) * vocab_size + v] =
                static_cast<float>(sum);
        }
    }
}

// ── Fixture ─────────────────────────────────────────────────────────────────

class EmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override { gen_.seed(42); }

    void fill_random(std::vector<float>& v, float lo = -1.0f,
                     float hi = 1.0f) {
        std::uniform_real_distribution<float> dist(lo, hi);
        for (auto& x : v) x = dist(gen_);
    }

    void fill_token_ids(std::vector<int32_t>& ids, int vocab_size) {
        std::uniform_int_distribution<int32_t> dist(0, vocab_size - 1);
        for (auto& x : ids) x = dist(gen_);
    }

    std::mt19937 gen_;
};

// ═════════════════════════════════════════════════════════════════════════════
// Embedding lookup tests
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(EmbeddingTest, EmbeddingLookup_SmallFP32) {
    REQUIRES_GPU();

    constexpr int num_tokens = 4;
    constexpr int vocab_size = 16;
    constexpr int hidden_size = 8;

    std::vector<float> h_table(vocab_size * hidden_size);
    std::vector<int32_t> h_ids(num_tokens);
    fill_random(h_table);
    fill_token_ids(h_ids, vocab_size);

    std::vector<float> h_ref(num_tokens * hidden_size);
    embedding_lookup_ref(h_ref.data(), h_table.data(), h_ids.data(),
                         num_tokens, vocab_size, hidden_size);

    float* d_table = nullptr;
    float* d_out = nullptr;
    int32_t* d_ids = nullptr;
    size_t table_bytes = vocab_size * hidden_size * sizeof(float);
    size_t out_bytes = num_tokens * hidden_size * sizeof(float);
    size_t ids_bytes = num_tokens * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_table, table_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_ids, ids_bytes));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), table_bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_ids, h_ids.data(), ids_bytes, cudaMemcpyHostToDevice));

    lc::launch_embedding_lookup(d_out, d_table, d_ids, num_tokens, vocab_size,
                                hidden_size, lc::EmbeddingDtype::kFloat32,
                                nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(num_tokens * hidden_size);
    CUDA_CHECK(
        cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i)
        ASSERT_NEAR(h_out[i], h_ref[i], 1e-6f) << "mismatch at index " << i;

    cudaFree(d_table);
    cudaFree(d_out);
    cudaFree(d_ids);
}

TEST_F(EmbeddingTest, EmbeddingLookup_BF16) {
    REQUIRES_GPU();

    constexpr int num_tokens = 16;
    constexpr int vocab_size = 64;
    constexpr int hidden_size = 128;

    std::vector<float> h_table_f(vocab_size * hidden_size);
    std::vector<int32_t> h_ids(num_tokens);
    fill_random(h_table_f);
    fill_token_ids(h_ids, vocab_size);

    // Convert to BF16 round-trip for reference.
    std::vector<__nv_bfloat16> h_table_bf(vocab_size * hidden_size);
    std::vector<float> h_table_rt(vocab_size * hidden_size);
    for (int i = 0; i < vocab_size * hidden_size; ++i) {
        h_table_bf[i] = float_to_bf16(h_table_f[i]);
        h_table_rt[i] = bf16_to_float(h_table_bf[i]);
    }

    std::vector<float> h_ref(num_tokens * hidden_size);
    embedding_lookup_ref(h_ref.data(), h_table_rt.data(), h_ids.data(),
                         num_tokens, vocab_size, hidden_size);

    __nv_bfloat16* d_table = nullptr;
    __nv_bfloat16* d_out = nullptr;
    int32_t* d_ids = nullptr;
    size_t table_bytes = vocab_size * hidden_size * sizeof(__nv_bfloat16);
    size_t out_bytes = num_tokens * hidden_size * sizeof(__nv_bfloat16);
    size_t ids_bytes = num_tokens * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_table, table_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_ids, ids_bytes));
    CUDA_CHECK(cudaMemcpy(d_table, h_table_bf.data(), table_bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_ids, h_ids.data(), ids_bytes, cudaMemcpyHostToDevice));

    lc::launch_embedding_lookup(d_out, d_table, d_ids, num_tokens, vocab_size,
                                hidden_size, lc::EmbeddingDtype::kBFloat16,
                                nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out_bf(num_tokens * hidden_size);
    CUDA_CHECK(cudaMemcpy(h_out_bf.data(), d_out, out_bytes,
                           cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        float gpu_val = bf16_to_float(h_out_bf[i]);
        ASSERT_NEAR(gpu_val, h_ref[i], 2e-2f) << "BF16 mismatch at " << i;
    }

    cudaFree(d_table);
    cudaFree(d_out);
    cudaFree(d_ids);
}

TEST_F(EmbeddingTest, EmbeddingLookup_FP16) {
    REQUIRES_GPU();

    constexpr int num_tokens = 16;
    constexpr int vocab_size = 64;
    constexpr int hidden_size = 128;

    std::vector<float> h_table_f(vocab_size * hidden_size);
    std::vector<int32_t> h_ids(num_tokens);
    fill_random(h_table_f);
    fill_token_ids(h_ids, vocab_size);

    // Convert to FP16 round-trip for reference.
    std::vector<__half> h_table_fp(vocab_size * hidden_size);
    std::vector<float> h_table_rt(vocab_size * hidden_size);
    for (int i = 0; i < vocab_size * hidden_size; ++i) {
        h_table_fp[i] = float_to_fp16(h_table_f[i]);
        h_table_rt[i] = fp16_to_float(h_table_fp[i]);
    }

    std::vector<float> h_ref(num_tokens * hidden_size);
    embedding_lookup_ref(h_ref.data(), h_table_rt.data(), h_ids.data(),
                         num_tokens, vocab_size, hidden_size);

    __half* d_table = nullptr;
    __half* d_out = nullptr;
    int32_t* d_ids = nullptr;
    size_t table_bytes = vocab_size * hidden_size * sizeof(__half);
    size_t out_bytes = num_tokens * hidden_size * sizeof(__half);
    size_t ids_bytes = num_tokens * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_table, table_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_ids, ids_bytes));
    CUDA_CHECK(cudaMemcpy(d_table, h_table_fp.data(), table_bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_ids, h_ids.data(), ids_bytes, cudaMemcpyHostToDevice));

    lc::launch_embedding_lookup(d_out, d_table, d_ids, num_tokens, vocab_size,
                                hidden_size, lc::EmbeddingDtype::kFloat16,
                                nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> h_out_fp(num_tokens * hidden_size);
    CUDA_CHECK(cudaMemcpy(h_out_fp.data(), d_out, out_bytes,
                           cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        float gpu_val = fp16_to_float(h_out_fp[i]);
        ASSERT_NEAR(gpu_val, h_ref[i], 1e-2f) << "FP16 mismatch at " << i;
    }

    cudaFree(d_table);
    cudaFree(d_out);
    cudaFree(d_ids);
}

TEST_F(EmbeddingTest, EmbeddingLookup_LargeBatch) {
    REQUIRES_GPU();

    constexpr int num_tokens = 1024;
    constexpr int vocab_size = 1024;
    constexpr int hidden_size = 7168;  // V3.2

    std::vector<float> h_table(static_cast<int64_t>(vocab_size) * hidden_size);
    std::vector<int32_t> h_ids(num_tokens);
    fill_random(h_table);
    fill_token_ids(h_ids, vocab_size);

    std::vector<float> h_ref(static_cast<int64_t>(num_tokens) * hidden_size);
    embedding_lookup_ref(h_ref.data(), h_table.data(), h_ids.data(),
                         num_tokens, vocab_size, hidden_size);

    float* d_table = nullptr;
    float* d_out = nullptr;
    int32_t* d_ids = nullptr;
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

    lc::launch_embedding_lookup(d_out, d_table, d_ids, num_tokens, vocab_size,
                                hidden_size, lc::EmbeddingDtype::kFloat32,
                                nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(static_cast<int64_t>(num_tokens) * hidden_size);
    CUDA_CHECK(
        cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    for (int64_t i = 0; i < static_cast<int64_t>(num_tokens) * hidden_size;
         ++i)
        ASSERT_NEAR(h_out[i], h_ref[i], 1e-5f) << "mismatch at index " << i;

    cudaFree(d_table);
    cudaFree(d_out);
    cudaFree(d_ids);
}

TEST_F(EmbeddingTest, EmbeddingLookup_SingleToken) {
    REQUIRES_GPU();

    constexpr int num_tokens = 1;
    constexpr int vocab_size = 128;
    constexpr int hidden_size = 7168;

    std::vector<float> h_table(static_cast<int64_t>(vocab_size) * hidden_size);
    fill_random(h_table);
    std::vector<int32_t> h_ids = {42};

    std::vector<float> h_ref(hidden_size);
    embedding_lookup_ref(h_ref.data(), h_table.data(), h_ids.data(),
                         num_tokens, vocab_size, hidden_size);

    float* d_table = nullptr;
    float* d_out = nullptr;
    int32_t* d_ids = nullptr;
    size_t table_bytes =
        static_cast<size_t>(vocab_size) * hidden_size * sizeof(float);
    size_t out_bytes = hidden_size * sizeof(float);
    size_t ids_bytes = sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_table, table_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_ids, ids_bytes));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), table_bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_ids, h_ids.data(), ids_bytes, cudaMemcpyHostToDevice));

    lc::launch_embedding_lookup(d_out, d_table, d_ids, num_tokens, vocab_size,
                                hidden_size, lc::EmbeddingDtype::kFloat32,
                                nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(hidden_size);
    CUDA_CHECK(
        cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < hidden_size; ++i)
        ASSERT_NEAR(h_out[i], h_ref[i], 1e-6f) << "mismatch at " << i;

    cudaFree(d_table);
    cudaFree(d_out);
    cudaFree(d_ids);
}

TEST_F(EmbeddingTest, EmbeddingLookup_BoundaryIds) {
    REQUIRES_GPU();

    constexpr int vocab_size = 64;
    constexpr int hidden_size = 32;
    constexpr int num_tokens = 2;

    std::vector<float> h_table(vocab_size * hidden_size);
    fill_random(h_table);
    std::vector<int32_t> h_ids = {0, vocab_size - 1};

    std::vector<float> h_ref(num_tokens * hidden_size);
    embedding_lookup_ref(h_ref.data(), h_table.data(), h_ids.data(),
                         num_tokens, vocab_size, hidden_size);

    float* d_table = nullptr;
    float* d_out = nullptr;
    int32_t* d_ids = nullptr;
    size_t table_bytes = vocab_size * hidden_size * sizeof(float);
    size_t out_bytes = num_tokens * hidden_size * sizeof(float);
    size_t ids_bytes = num_tokens * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_table, table_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_ids, ids_bytes));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), table_bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_ids, h_ids.data(), ids_bytes, cudaMemcpyHostToDevice));

    lc::launch_embedding_lookup(d_out, d_table, d_ids, num_tokens, vocab_size,
                                hidden_size, lc::EmbeddingDtype::kFloat32,
                                nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(num_tokens * hidden_size);
    CUDA_CHECK(
        cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * hidden_size; ++i)
        ASSERT_NEAR(h_out[i], h_ref[i], 1e-6f) << "mismatch at " << i;

    cudaFree(d_table);
    cudaFree(d_out);
    cudaFree(d_ids);
}

// ═════════════════════════════════════════════════════════════════════════════
// Output head tests
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(EmbeddingTest, OutputHead_SmallFP32) {
    REQUIRES_GPU();

    constexpr int num_tokens = 4;
    constexpr int vocab_size = 16;
    constexpr int hidden_size = 8;

    std::vector<float> h_hidden(num_tokens * hidden_size);
    std::vector<float> h_weight(vocab_size * hidden_size);
    fill_random(h_hidden);
    fill_random(h_weight);

    std::vector<float> h_ref(num_tokens * vocab_size);
    output_head_ref(h_ref.data(), h_hidden.data(), h_weight.data(), nullptr,
                    num_tokens, vocab_size, hidden_size);

    float *d_hidden, *d_weight, *d_logits;
    size_t hidden_bytes = num_tokens * hidden_size * sizeof(float);
    size_t weight_bytes = vocab_size * hidden_size * sizeof(float);
    size_t logit_bytes = num_tokens * vocab_size * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_hidden, hidden_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_logits, logit_bytes));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(), hidden_bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes,
                           cudaMemcpyHostToDevice));

    lc::launch_output_head(d_logits, d_hidden, d_weight, nullptr, num_tokens,
                           vocab_size, hidden_size,
                           lc::EmbeddingDtype::kFloat32, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_logits(num_tokens * vocab_size);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits, logit_bytes,
                           cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * vocab_size; ++i)
        ASSERT_NEAR(h_logits[i], h_ref[i], 1e-4f) << "mismatch at " << i;

    cudaFree(d_hidden);
    cudaFree(d_weight);
    cudaFree(d_logits);
}

TEST_F(EmbeddingTest, OutputHead_WithBias) {
    REQUIRES_GPU();

    constexpr int num_tokens = 4;
    constexpr int vocab_size = 16;
    constexpr int hidden_size = 8;

    std::vector<float> h_hidden(num_tokens * hidden_size);
    std::vector<float> h_weight(vocab_size * hidden_size);
    std::vector<float> h_bias(vocab_size);
    fill_random(h_hidden);
    fill_random(h_weight);
    fill_random(h_bias, -0.5f, 0.5f);

    std::vector<float> h_ref(num_tokens * vocab_size);
    output_head_ref(h_ref.data(), h_hidden.data(), h_weight.data(),
                    h_bias.data(), num_tokens, vocab_size, hidden_size);

    float *d_hidden, *d_weight, *d_logits, *d_bias;
    CUDA_CHECK(
        cudaMalloc(&d_hidden, num_tokens * hidden_size * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_weight, vocab_size * hidden_size * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_logits, num_tokens * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(),
                           num_tokens * hidden_size * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(),
                           vocab_size * hidden_size * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), vocab_size * sizeof(float),
                           cudaMemcpyHostToDevice));

    lc::launch_output_head(d_logits, d_hidden, d_weight, d_bias, num_tokens,
                           vocab_size, hidden_size,
                           lc::EmbeddingDtype::kFloat32, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_logits(num_tokens * vocab_size);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits,
                           num_tokens * vocab_size * sizeof(float),
                           cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * vocab_size; ++i)
        ASSERT_NEAR(h_logits[i], h_ref[i], 1e-4f) << "mismatch at " << i;

    cudaFree(d_hidden);
    cudaFree(d_weight);
    cudaFree(d_logits);
    cudaFree(d_bias);
}

TEST_F(EmbeddingTest, OutputHead_NoBias) {
    REQUIRES_GPU();

    constexpr int num_tokens = 4;
    constexpr int vocab_size = 16;
    constexpr int hidden_size = 8;

    std::vector<float> h_hidden(num_tokens * hidden_size);
    std::vector<float> h_weight(vocab_size * hidden_size);
    fill_random(h_hidden);
    fill_random(h_weight);

    std::vector<float> h_ref(num_tokens * vocab_size);
    output_head_ref(h_ref.data(), h_hidden.data(), h_weight.data(), nullptr,
                    num_tokens, vocab_size, hidden_size);

    float *d_hidden, *d_weight, *d_logits;
    CUDA_CHECK(
        cudaMalloc(&d_hidden, num_tokens * hidden_size * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_weight, vocab_size * hidden_size * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_logits, num_tokens * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(),
                           num_tokens * hidden_size * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(),
                           vocab_size * hidden_size * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Explicitly pass nullptr for bias — should not crash.
    lc::launch_output_head(d_logits, d_hidden, d_weight, nullptr, num_tokens,
                           vocab_size, hidden_size,
                           lc::EmbeddingDtype::kFloat32, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_logits(num_tokens * vocab_size);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits,
                           num_tokens * vocab_size * sizeof(float),
                           cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_tokens * vocab_size; ++i)
        ASSERT_NEAR(h_logits[i], h_ref[i], 1e-4f) << "mismatch at " << i;

    cudaFree(d_hidden);
    cudaFree(d_weight);
    cudaFree(d_logits);
}

TEST_F(EmbeddingTest, OutputHead_BF16) {
    REQUIRES_GPU();

    constexpr int num_tokens = 8;
    constexpr int vocab_size = 64;
    constexpr int hidden_size = 128;

    std::vector<float> h_hidden_f(num_tokens * hidden_size);
    std::vector<float> h_weight_f(vocab_size * hidden_size);
    fill_random(h_hidden_f);
    fill_random(h_weight_f);

    // Convert to BF16 for GPU and round-trip for CPU reference.
    std::vector<__nv_bfloat16> h_hidden_bf(num_tokens * hidden_size);
    std::vector<__nv_bfloat16> h_weight_bf(vocab_size * hidden_size);
    std::vector<float> h_hidden_rt(num_tokens * hidden_size);
    std::vector<float> h_weight_rt(vocab_size * hidden_size);
    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        h_hidden_bf[i] = float_to_bf16(h_hidden_f[i]);
        h_hidden_rt[i] = bf16_to_float(h_hidden_bf[i]);
    }
    for (int i = 0; i < vocab_size * hidden_size; ++i) {
        h_weight_bf[i] = float_to_bf16(h_weight_f[i]);
        h_weight_rt[i] = bf16_to_float(h_weight_bf[i]);
    }

    std::vector<float> h_ref(num_tokens * vocab_size);
    output_head_ref(h_ref.data(), h_hidden_rt.data(), h_weight_rt.data(),
                    nullptr, num_tokens, vocab_size, hidden_size);

    __nv_bfloat16 *d_hidden, *d_weight;
    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_hidden,
                           num_tokens * hidden_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_weight,
                           vocab_size * hidden_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(
        cudaMalloc(&d_logits, num_tokens * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden_bf.data(),
                           num_tokens * hidden_size * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight_bf.data(),
                           vocab_size * hidden_size * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));

    lc::launch_output_head(d_logits, d_hidden, d_weight, nullptr, num_tokens,
                           vocab_size, hidden_size,
                           lc::EmbeddingDtype::kBFloat16, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_logits(num_tokens * vocab_size);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits,
                           num_tokens * vocab_size * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // BF16 inputs with FP32 accumulation: expect tighter tolerance than pure
    // BF16 ops, but still allow for rounding in the tensor-core path.
    for (int i = 0; i < num_tokens * vocab_size; ++i)
        ASSERT_NEAR(h_logits[i], h_ref[i], 5e-2f)
            << "BF16 output head mismatch at " << i;

    cudaFree(d_hidden);
    cudaFree(d_weight);
    cudaFree(d_logits);
}

TEST_F(EmbeddingTest, OutputHead_LargeVocab) {
    REQUIRES_GPU();

    // V3.2 real dimensions. Only 4 tokens to keep VRAM reasonable.
    constexpr int num_tokens = 4;
    constexpr int vocab_size = 129280;
    constexpr int hidden_size = 7168;

    // Allocate weight on host in BF16 directly to save host memory.
    const int64_t weight_elems =
        static_cast<int64_t>(vocab_size) * hidden_size;
    const int64_t hidden_elems =
        static_cast<int64_t>(num_tokens) * hidden_size;
    const int64_t logit_elems =
        static_cast<int64_t>(num_tokens) * vocab_size;

    std::vector<__nv_bfloat16> h_weight_bf(weight_elems);
    std::vector<__nv_bfloat16> h_hidden_bf(hidden_elems);
    {
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (int64_t i = 0; i < weight_elems; ++i)
            h_weight_bf[i] = float_to_bf16(dist(gen_));
        for (int64_t i = 0; i < hidden_elems; ++i)
            h_hidden_bf[i] = float_to_bf16(dist(gen_));
    }

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

    // Spot-check a subset: first 64 logits of each token.
    constexpr int check_count = 64;
    std::vector<float> h_logits(logit_elems);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits,
                           logit_elems * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // Compute CPU reference for the checked subset.
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
                << "LargeVocab mismatch at token=" << t << " vocab=" << v;
        }
    }

    cudaFree(d_hidden);
    cudaFree(d_weight);
    cudaFree(d_logits);
}

TEST_F(EmbeddingTest, EmbeddingLookup_OutOfBounds) {
    REQUIRES_GPU();

    constexpr int num_tokens = 2;
    constexpr int vocab_size = 16;
    constexpr int hidden_size = 8;

    std::vector<float> h_table(vocab_size * hidden_size);
    fill_random(h_table);
    // One valid, one out-of-bounds.
    std::vector<int32_t> h_ids = {5, vocab_size};

    float* d_table = nullptr;
    float* d_out = nullptr;
    int32_t* d_ids = nullptr;
    CUDA_CHECK(
        cudaMalloc(&d_table, vocab_size * hidden_size * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_out, num_tokens * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ids, num_tokens * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(),
                           vocab_size * hidden_size * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), num_tokens * sizeof(int32_t),
                           cudaMemcpyHostToDevice));

    lc::launch_embedding_lookup(d_out, d_table, d_ids, num_tokens, vocab_size,
                                hidden_size, lc::EmbeddingDtype::kFloat32,
                                nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(num_tokens * hidden_size);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                           num_tokens * hidden_size * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // Token 0 (id=5): should match table row 5.
    for (int i = 0; i < hidden_size; ++i)
        ASSERT_NEAR(h_out[i], h_table[5 * hidden_size + i], 1e-6f)
            << "valid token mismatch at " << i;

    // Token 1 (id=vocab_size, out-of-bounds): should be zeros.
    for (int i = 0; i < hidden_size; ++i)
        ASSERT_FLOAT_EQ(h_out[hidden_size + i], 0.0f)
            << "OOB token should be zero at " << i;

    cudaFree(d_table);
    cudaFree(d_out);
    cudaFree(d_ids);
}

TEST_F(EmbeddingTest, EmbeddingLookup_VocabShardedMasked) {
    // TD-GOLDEN-EMB-OOB: the sum of two masked half-shard lookups must equal
    // the full-table lookup (the dispatcher allreduce-sums per-rank results),
    // and an out-of-shard token must yield a zero row on that shard.
    REQUIRES_GPU();

    constexpr int num_tokens = 6;
    constexpr int vocab_size = 32;
    constexpr int hidden_size = 8;
    constexpr int local_vocab = vocab_size / 2;

    std::vector<float> h_table(vocab_size * hidden_size);
    fill_random(h_table);
    // Span both shards, including the boundary rows 15/16 and the extremes.
    std::vector<int32_t> h_ids = {0, 7, 15, 16, 25, 31};

    std::vector<float> h_ref(num_tokens * hidden_size);
    embedding_lookup_ref(h_ref.data(), h_table.data(), h_ids.data(),
                         num_tokens, vocab_size, hidden_size);

    float* d_table = nullptr;  // full table; shard r starts at row r*16
    float* d_out0 = nullptr;
    float* d_out1 = nullptr;
    int32_t* d_ids = nullptr;
    const size_t out_bytes = num_tokens * hidden_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out0, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_out1, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_ids, num_tokens * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(),
                           vocab_size * hidden_size * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), num_tokens * sizeof(int32_t),
                           cudaMemcpyHostToDevice));

    // Rank 0 shard: rows [0, 16); rank 1 shard: rows [16, 32).
    lc::launch_embedding_lookup(d_out0, d_table, d_ids, num_tokens,
                                vocab_size, hidden_size,
                                lc::EmbeddingDtype::kFloat32, nullptr,
                                /*vocab_offset=*/0, local_vocab);
    lc::launch_embedding_lookup(d_out1, d_table + local_vocab * hidden_size,
                                d_ids, num_tokens, vocab_size, hidden_size,
                                lc::EmbeddingDtype::kFloat32, nullptr,
                                /*vocab_offset=*/local_vocab, local_vocab);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out0(num_tokens * hidden_size);
    std::vector<float> h_out1(num_tokens * hidden_size);
    CUDA_CHECK(cudaMemcpy(h_out0.data(), d_out0, out_bytes,
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out1.data(), d_out1, out_bytes,
                           cudaMemcpyDeviceToHost));

    for (int t = 0; t < num_tokens; ++t) {
        const bool in_shard0 = h_ids[t] < local_vocab;
        for (int i = 0; i < hidden_size; ++i) {
            const int idx = t * hidden_size + i;
            // Out-of-shard rows are exactly zero on that shard.
            ASSERT_FLOAT_EQ(in_shard0 ? h_out1[idx] : h_out0[idx], 0.0f)
                << "token " << h_ids[t] << " elem " << i;
            // Sum of shards == full lookup.
            ASSERT_NEAR(h_out0[idx] + h_out1[idx], h_ref[idx], 1e-6f)
                << "token " << h_ids[t] << " elem " << i;
        }
    }

    cudaFree(d_table);
    cudaFree(d_out0);
    cudaFree(d_out1);
    cudaFree(d_ids);
}
