// Unit tests for TQ MLA attention prep kernel wrappers and decode dispatch.
//
// Tests: TQ k_append + dequant round-trip, q_rotate + v_rotate_back round-trip,
// dense decode smoke, sparse==dense equivalence.

#include "compute/kernels/attention/tq_mla_attention.h"
#include "compute/kernels/attention/mla_attention.h"  // launch_mla_combine (BF16)
#include "compute/kernels/sm120/attention/tq_prep_params.h"
#include "compute/tq_init.h"

// Shared metadata kernel + param structs
#include "smxx/get_mla_metadata.h"
#include "smxx/params.h"

// Decode param structs
#include <sm120/decode/tq_dense/params.h>
#include <sm120/decode/tq_sparse/params.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;

#define CUDA_CHECK(expr)                                                 \
    do {                                                                 \
        cudaError_t _err = (expr);                                       \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_err); \
    } while (0)

// ── Host BF16 helpers ──────────────────────────────────────────────────────

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

// ── Codebook path helper ───────────────────────────────────────────────────

#ifdef LAYERSTORM_SOURCE_DIR
static std::string codebook_dir() {
    return std::string(LAYERSTORM_SOURCE_DIR) + "/config/tq_codebooks";
}
#endif

// ── Cosine similarity helper ───────────────────────────────────────────────

static float cosine_similarity(const float* a, const float* b, int n) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        norm_a += static_cast<double>(a[i]) * a[i];
        norm_b += static_cast<double>(b[i]) * b[i];
    }
    if (norm_a < 1e-30 || norm_b < 1e-30) return 0.0f;
    return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}

// ── Tests: TQ k_append + dequant round-trip ────────────────────────────────

#ifdef LAYERSTORM_SOURCE_DIR
TEST(TqMlaAttentionPrep, KAppendDequantRoundTrip) {
    REQUIRES_GPU();

    const int num_tokens = 8;
    const int d_c = 512;     // V3.2 kv_lora_rank
    const int d_rope = 64;
    const int page_size = 64;
    // TQ row: d_c/2 packed + 2 (fp16 norm) + d_rope*2 (BF16 rope) = 386 bytes
    const int row_bytes = d_c / 2 + 2 + d_rope * 2;
    const int page_bytes = page_size * row_bytes;

    // Load codebook
    auto cb = lc::load_codebook(codebook_dir() + "/codebook_d512_b4.json");
    ASSERT_EQ(cb.d, 512);
    ASSERT_EQ(cb.n_clusters, 16);

    // Generate rotation matrix for layer 0
    const int layer_seed = 42 + 0 * 7;
    std::vector<float> Pi(d_c * d_c);
    lc::generate_rotation_matrix_cpu(Pi.data(), d_c, layer_seed);
    ASSERT_LT(lc::verify_orthogonality(Pi.data(), d_c), 1e-4f);

    // Generate random BF16 input data
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> c_kv_float(num_tokens * d_c);
    std::vector<float> k_rope_float(num_tokens * d_rope);
    std::vector<__nv_bfloat16> c_kv_bf16(num_tokens * d_c);
    std::vector<__nv_bfloat16> k_rope_bf16(num_tokens * d_rope);

    for (int i = 0; i < num_tokens * d_c; ++i) {
        c_kv_float[i] = dist(rng);
        c_kv_bf16[i] = float_to_bf16(c_kv_float[i]);
    }
    for (int i = 0; i < num_tokens * d_rope; ++i) {
        k_rope_float[i] = dist(rng);
        k_rope_bf16[i] = float_to_bf16(k_rope_float[i]);
    }

    // Slot mapping: identity (tokens 0..7 in page 0)
    std::vector<int> slot_mapping(num_tokens);
    for (int i = 0; i < num_tokens; ++i) slot_mapping[i] = i;

    // Device allocations
    __nv_bfloat16* d_c_kv = nullptr;
    __nv_bfloat16* d_k_rope = nullptr;
    uint8_t* d_kv_cache = nullptr;
    int* d_slot_mapping = nullptr;
    float* d_Pi = nullptr;
    float* d_centroids = nullptr;
    float* d_boundaries = nullptr;
    __nv_bfloat16* d_output = nullptr;
    int* d_indices = nullptr;

    CUDA_CHECK(cudaMalloc(&d_c_kv, num_tokens * d_c * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k_rope, num_tokens * d_rope * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_kv_cache, page_bytes));
    CUDA_CHECK(cudaMalloc(&d_slot_mapping, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Pi, d_c * d_c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, cb.n_clusters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_boundaries, cb.interior_boundaries.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * (d_c + d_rope) * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_indices, num_tokens * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_kv_cache, 0, page_bytes));

    CUDA_CHECK(cudaMemcpy(d_c_kv, c_kv_bf16.data(),
        num_tokens * d_c * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_rope, k_rope_bf16.data(),
        num_tokens * d_rope * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slot_mapping, slot_mapping.data(),
        num_tokens * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Pi, Pi.data(),
        d_c * d_c * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, cb.centroids.data(),
        cb.n_clusters * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_boundaries, cb.interior_boundaries.data(),
        cb.interior_boundaries.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Dequant indices = identity (read back same tokens)
    CUDA_CHECK(cudaMemcpy(d_indices, slot_mapping.data(),
        num_tokens * sizeof(int), cudaMemcpyHostToDevice));

    // Step 1: TQ k_append
    sm120::prep::TqFusedKAppendParams ka{};
    ka.c_kv = d_c_kv;
    ka.k_rope = d_k_rope;
    ka.kv_cache = d_kv_cache;
    ka.cache_stride_block = page_bytes;
    ka.cache_stride_row = row_bytes;
    ka.slot_mapping = d_slot_mapping;
    ka.Pi = d_Pi;
    ka.centroids = d_centroids;
    ka.decision_boundaries = d_boundaries;
    ka.num_tokens = num_tokens;
    ka.d_c = d_c;
    ka.d_rope = d_rope;
    ka.page_size = page_size;
    ka.num_centroids = cb.n_clusters;

    lc::launch_tq_k_append(ka, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: TQ dequant
    sm120::prep::TqDequantCKVIndexedParams dq{};
    dq.kv_cache = d_kv_cache;
    dq.cache_stride_block = page_bytes;
    dq.cache_stride_row = row_bytes;
    dq.page_size = page_size;
    dq.indices = d_indices;
    dq.num_fetch = num_tokens;
    dq.k_out = d_output;
    dq.Pi = d_Pi;
    dq.centroids = d_centroids;
    dq.d_c = d_c;
    dq.d_rope = d_rope;

    lc::launch_tq_dequant_ckv_indexed(dq, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back and verify round-trip
    std::vector<__nv_bfloat16> output(num_tokens * (d_c + d_rope));
    CUDA_CHECK(cudaMemcpy(output.data(), d_output,
        output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Per-token cosine similarity for NOPE part
    for (int t = 0; t < num_tokens; ++t) {
        std::vector<float> orig(d_c), recon(d_c);
        for (int d = 0; d < d_c; ++d) {
            orig[d] = c_kv_float[t * d_c + d];
            recon[d] = bf16_to_float(output[t * (d_c + d_rope) + d]);
        }
        float cos = cosine_similarity(orig.data(), recon.data(), d_c);
        EXPECT_GT(cos, 0.90f) << "Token " << t << " NOPE cosine too low: " << cos;
    }

    // ROPE part should be exact (BF16 passthrough, no quantization)
    for (int t = 0; t < num_tokens; ++t) {
        for (int d = 0; d < d_rope; ++d) {
            float orig = k_rope_float[t * d_rope + d];
            float recon = bf16_to_float(output[t * (d_c + d_rope) + d_c + d]);
            float err = std::abs(orig - recon);
            // BF16 round-trip: allow tiny error
            EXPECT_LT(err, 0.02f) << "Token " << t << " ROPE element " << d
                << " error " << err;
        }
    }

    cudaFree(d_c_kv);
    cudaFree(d_k_rope);
    cudaFree(d_kv_cache);
    cudaFree(d_slot_mapping);
    cudaFree(d_Pi);
    cudaFree(d_centroids);
    cudaFree(d_boundaries);
    cudaFree(d_output);
    cudaFree(d_indices);
}
#endif  // LAYERSTORM_SOURCE_DIR

// ── Tests: q_rotate + v_rotate_back round-trip ─────────────────────────────

// ── TD-PREFILL-CHUNK-ATTN: strided (interleaved) TQ k_append A/B ────────────
//
// Same lock-in as MlaAttentionPrep.KAppendStridedInterleavedBatchBitEqualsPerToken
// for the TQ cache path: an interleaved [num_tokens, d_c + d_rope] source
// appended in ONE batched call with explicit strides must produce BIT-IDENTICAL
// cache rows to per-token tight appends.
TEST(TqMlaAttentionPrep, KAppendStridedInterleavedBatchBitEqualsPerToken) {
    REQUIRES_GPU();

    const int num_tokens = 4;
    const int d_c = 512;
    const int d_rope = 64;
    const int kv_dim = d_c + d_rope;
    const int page_size = 64;
    const int row_bytes = d_c / 2 + 2 + d_rope * 2;
    const int page_bytes = page_size * row_bytes;

    auto cb = lc::load_codebook(codebook_dir() + "/codebook_d512_b4.json");
    ASSERT_EQ(cb.d, 512);
    std::vector<float> Pi(d_c * d_c);
    lc::generate_rotation_matrix_cpu(Pi.data(), d_c, 42);

    std::mt19937 rng(654);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<__nv_bfloat16> inter_host(num_tokens * kv_dim);
    for (auto& v : inter_host) v = float_to_bf16(dist(rng));

    std::vector<int> slots(num_tokens);
    for (int i = 0; i < num_tokens; ++i) slots[i] = i;

    __nv_bfloat16* d_inter = nullptr;
    __nv_bfloat16* d_tight = nullptr;
    uint8_t* d_cache_a = nullptr;
    uint8_t* d_cache_b = nullptr;
    int* d_slots = nullptr;
    float* d_Pi = nullptr;
    float* d_centroids = nullptr;
    float* d_boundaries = nullptr;

    CUDA_CHECK(cudaMalloc(&d_inter, inter_host.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_tight, kv_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cache_a, page_bytes));
    CUDA_CHECK(cudaMalloc(&d_cache_b, page_bytes));
    CUDA_CHECK(cudaMalloc(&d_slots, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Pi, d_c * d_c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, cb.n_clusters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_boundaries,
                          cb.interior_boundaries.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_cache_a, 0, page_bytes));
    CUDA_CHECK(cudaMemset(d_cache_b, 0, page_bytes));
    CUDA_CHECK(cudaMemcpy(d_inter, inter_host.data(),
                          inter_host.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, slots.data(), num_tokens * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Pi, Pi.data(), d_c * d_c * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, cb.centroids.data(),
                          cb.n_clusters * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_boundaries, cb.interior_boundaries.data(),
                          cb.interior_boundaries.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    auto base_params = [&](uint8_t* cache) {
        sm120::prep::TqFusedKAppendParams p{};
        p.kv_cache = cache;
        p.cache_stride_block = page_bytes;
        p.cache_stride_row = row_bytes;
        p.Pi = d_Pi;
        p.centroids = d_centroids;
        p.decision_boundaries = d_boundaries;
        p.d_c = d_c;
        p.d_rope = d_rope;
        p.page_size = page_size;
        p.num_centroids = cb.n_clusters;
        return p;
    };

    // A: one batched call over the interleaved rows with explicit strides.
    {
        auto pa = base_params(d_cache_a);
        pa.c_kv = d_inter;
        pa.k_rope = d_inter + d_c;
        pa.src_stride_ckv = kv_dim;
        pa.src_stride_rope = kv_dim;
        pa.slot_mapping = d_slots;
        pa.num_tokens = num_tokens;
        lc::launch_tq_k_append(pa, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // B: per-token tight appends (zero-stride default = legacy tight layout).
    for (int t = 0; t < num_tokens; ++t) {
        CUDA_CHECK(cudaMemcpy(d_tight, inter_host.data() + t * kv_dim,
                              kv_dim * sizeof(__nv_bfloat16),
                              cudaMemcpyHostToDevice));
        auto pb = base_params(d_cache_b);
        pb.c_kv = d_tight;
        pb.k_rope = d_tight + d_c;
        pb.slot_mapping = d_slots + t;
        pb.num_tokens = 1;
        lc::launch_tq_k_append(pb, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<uint8_t> cache_a(page_bytes), cache_b(page_bytes);
    CUDA_CHECK(cudaMemcpy(cache_a.data(), d_cache_a, page_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cache_b.data(), d_cache_b, page_bytes,
                          cudaMemcpyDeviceToHost));
    for (int t = 0; t < num_tokens; ++t) {
        ASSERT_EQ(std::memcmp(cache_a.data() + (size_t)t * row_bytes,
                              cache_b.data() + (size_t)t * row_bytes,
                              row_bytes), 0)
            << "TQ cache row " << t << ": strided batch != per-token tight";
    }

    cudaFree(d_inter);
    cudaFree(d_tight);
    cudaFree(d_cache_a);
    cudaFree(d_cache_b);
    cudaFree(d_slots);
    cudaFree(d_Pi);
    cudaFree(d_centroids);
    cudaFree(d_boundaries);
}

TEST(TqMlaAttentionPrep, QRotateVRotateBackRoundTrip) {
    REQUIRES_GPU();

    const int batch_heads = 16;  // b=2, s_q=1, h_q=8
    const int d_c = 512;

    // Generate rotation matrix
    std::vector<float> Pi(d_c * d_c);
    lc::generate_rotation_matrix_cpu(Pi.data(), d_c, 42);

    // Generate random BF16 q_nope
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q_orig(batch_heads * d_c);
    std::vector<__nv_bfloat16> q_bf16(batch_heads * d_c);
    for (int i = 0; i < batch_heads * d_c; ++i) {
        q_orig[i] = dist(rng);
        q_bf16[i] = float_to_bf16(q_orig[i]);
    }

    // Device allocations
    __nv_bfloat16* d_q_nope = nullptr;
    float* d_q_rot = nullptr;
    __nv_bfloat16* d_q_back = nullptr;
    float* d_Pi = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q_nope, batch_heads * d_c * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q_rot, batch_heads * d_c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_back, batch_heads * d_c * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_Pi, d_c * d_c * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q_nope, q_bf16.data(),
        batch_heads * d_c * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Pi, Pi.data(),
        d_c * d_c * sizeof(float), cudaMemcpyHostToDevice));

    // Step 1: q_rotate (BF16 → FP32 rotated)
    sm120::prep::TqQRotateParams qr{};
    qr.q_nope = d_q_nope;
    qr.Pi = d_Pi;
    qr.q_rot = d_q_rot;
    qr.batch_heads = batch_heads;
    qr.d_c = d_c;

    lc::launch_tq_q_rotate(qr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: v_rotate_back (FP32 → BF16, applies Π inverse)
    sm120::prep::TqVRotateBackParams vr{};
    vr.out_rotated = d_q_rot;
    vr.Pi = d_Pi;
    vr.out_final = d_q_back;
    vr.batch_heads = batch_heads;
    vr.d_c = d_c;

    lc::launch_tq_v_rotate_back(vr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back result
    std::vector<__nv_bfloat16> q_result(batch_heads * d_c);
    CUDA_CHECK(cudaMemcpy(q_result.data(), d_q_back,
        batch_heads * d_c * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Verify: v_rotate_back(q_rotate(x)) ≈ x
    // Expected precision loss: BF16 → FP32 → matmul → matmul → BF16
    for (int h = 0; h < batch_heads; ++h) {
        std::vector<float> orig(d_c), recon(d_c);
        for (int d = 0; d < d_c; ++d) {
            orig[d] = q_orig[h * d_c + d];
            recon[d] = bf16_to_float(q_result[h * d_c + d]);
        }
        float cos = cosine_similarity(orig.data(), recon.data(), d_c);
        EXPECT_GT(cos, 0.999f) << "Head " << h << " rotate round-trip cosine: " << cos;
    }

    cudaFree(d_q_nope);
    cudaFree(d_q_rot);
    cudaFree(d_q_back);
    cudaFree(d_Pi);
}

// ── Tests: Dense decode smoke ──────────────────────────────────────────────

#ifdef LAYERSTORM_SOURCE_DIR
TEST(TqMlaAttentionDecode, DenseDecodeSmoke) {
    REQUIRES_GPU();

    const int b = 1, s_q = 1, h_q = 8;
    const int d_c = 512, d_rope = 64;
    const int seqlen_k = 128;
    const int page_size = 64;
    const int num_pages = (seqlen_k + page_size - 1) / page_size;
    const int row_bytes = d_c / 2 + 2 + d_rope * 2;
    const int page_bytes = page_size * row_bytes;

    // Load codebook + generate Pi
    auto cb = lc::load_codebook(codebook_dir() + "/codebook_d512_b4.json");
    std::vector<float> Pi(d_c * d_c);
    lc::generate_rotation_matrix_cpu(Pi.data(), d_c, 42);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Generate random KV tokens and populate TQ cache
    std::vector<__nv_bfloat16> c_kv(seqlen_k * d_c);
    std::vector<__nv_bfloat16> k_rope(seqlen_k * d_rope);
    for (auto& v : c_kv) v = float_to_bf16(dist(rng));
    for (auto& v : k_rope) v = float_to_bf16(dist(rng));

    std::vector<int> slots(seqlen_k);
    for (int i = 0; i < seqlen_k; ++i) slots[i] = i;

    // Generate random Q
    std::vector<__nv_bfloat16> q_nope_bf16(b * s_q * h_q * d_c);
    std::vector<__nv_bfloat16> q_rope_bf16(b * s_q * h_q * d_rope);
    for (auto& v : q_nope_bf16) v = float_to_bf16(dist(rng));
    for (auto& v : q_rope_bf16) v = float_to_bf16(dist(rng));

    // Block table: [page_0, page_1]
    std::vector<int> block_table(num_pages);
    for (int i = 0; i < num_pages; ++i) block_table[i] = i;

    // Device allocations
    __nv_bfloat16* d_c_kv = nullptr;
    __nv_bfloat16* d_k_rope = nullptr;
    uint8_t* d_cache = nullptr;
    int* d_slots = nullptr;
    float* d_Pi = nullptr;
    float* d_centroids = nullptr;
    float* d_boundaries = nullptr;
    __nv_bfloat16* d_q_nope = nullptr;
    __nv_bfloat16* d_q_rope = nullptr;
    float* d_q_rot = nullptr;
    float* d_out = nullptr;
    float* d_lse = nullptr;
    int* d_seqlens = nullptr;
    int* d_block_table = nullptr;

    CUDA_CHECK(cudaMalloc(&d_c_kv, seqlen_k * d_c * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k_rope, seqlen_k * d_rope * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cache, num_pages * page_bytes));
    CUDA_CHECK(cudaMalloc(&d_slots, seqlen_k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Pi, d_c * d_c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, cb.n_clusters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_boundaries, cb.interior_boundaries.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_nope, b * s_q * h_q * d_c * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q_rope, b * s_q * h_q * d_rope * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q_rot, b * s_q * h_q * d_c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, b * s_q * h_q * d_c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lse, b * s_q * h_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_seqlens, b * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_table, num_pages * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_cache, 0, num_pages * page_bytes));

    CUDA_CHECK(cudaMemcpy(d_c_kv, c_kv.data(), c_kv.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_rope, k_rope.data(), k_rope.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, slots.data(), slots.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Pi, Pi.data(), d_c * d_c * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, cb.centroids.data(), cb.n_clusters * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_boundaries, cb.interior_boundaries.data(), cb.interior_boundaries.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_nope, q_nope_bf16.data(), q_nope_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_rope, q_rope_bf16.data(), q_rope_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    int seqlen = seqlen_k;
    CUDA_CHECK(cudaMemcpy(d_seqlens, &seqlen, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_table, block_table.data(), num_pages * sizeof(int), cudaMemcpyHostToDevice));

    // Step 1: Populate TQ cache
    sm120::prep::TqFusedKAppendParams ka{};
    ka.c_kv = d_c_kv;
    ka.k_rope = d_k_rope;
    ka.kv_cache = d_cache;
    ka.cache_stride_block = page_bytes;
    ka.cache_stride_row = row_bytes;
    ka.slot_mapping = d_slots;
    ka.Pi = d_Pi;
    ka.centroids = d_centroids;
    ka.decision_boundaries = d_boundaries;
    ka.num_tokens = seqlen_k;
    ka.d_c = d_c;
    ka.d_rope = d_rope;
    ka.page_size = page_size;
    ka.num_centroids = cb.n_clusters;

    lc::launch_tq_k_append(ka, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Pre-rotate Q
    sm120::prep::TqQRotateParams qr{};
    qr.q_nope = d_q_nope;
    qr.Pi = d_Pi;
    qr.q_rot = d_q_rot;
    qr.batch_heads = b * s_q * h_q;
    qr.d_c = d_c;

    lc::launch_tq_q_rotate(qr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: Dense decode (no split-KV for smoke test)
    sm120::decode::tq_dense::TqDenseDecodeParams dp{};
    std::memset(&dp, 0, sizeof(dp));
    dp.b = b; dp.s_q = s_q; dp.h_q = h_q; dp.h_kv = 1;
    dp.d_c = d_c; dp.d_rope = d_rope;
    dp.sm_scale = 1.0f / std::sqrt(static_cast<float>(d_c + d_rope));
    dp.q_rot = d_q_rot;
    dp.q_rope = d_q_rope;
    dp.kv_cache = d_cache;
    dp.cache_stride_block = page_bytes;
    dp.cache_stride_row = row_bytes;
    dp.block_table = d_block_table;
    dp.block_table_batch_stride = num_pages;
    dp.page_block_size = page_size;
    dp.seqlens_k = d_seqlens;
    dp.centroids = d_centroids;
    dp.out = d_out;
    dp.lse = d_lse;
    dp.stride_o_b = s_q * h_q * d_c;
    dp.stride_o_s_q = h_q * d_c;
    dp.stride_o_h_q = d_c;
    dp.stride_lse_b = s_q * h_q;
    dp.stride_lse_s_q = h_q;
    dp.num_sm_parts = 1;
    dp.stream = nullptr;

    lc::launch_decode_dense_tq(dp);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back and verify: output is finite and non-zero
    std::vector<float> out_host(b * s_q * h_q * d_c);
    std::vector<float> lse_host(b * s_q * h_q);
    CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lse_host.data(), d_lse, lse_host.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool any_nonzero = false;
    for (size_t i = 0; i < out_host.size(); ++i) {
        EXPECT_FALSE(std::isnan(out_host[i])) << "Output NaN at " << i;
        EXPECT_FALSE(std::isinf(out_host[i])) << "Output Inf at " << i;
        if (std::abs(out_host[i]) > 1e-8f) any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero) << "All output elements are zero";

    for (int i = 0; i < b * s_q * h_q; ++i) {
        EXPECT_FALSE(std::isnan(lse_host[i])) << "LSE NaN at " << i;
    }

    cudaFree(d_c_kv); cudaFree(d_k_rope); cudaFree(d_cache);
    cudaFree(d_slots); cudaFree(d_Pi); cudaFree(d_centroids);
    cudaFree(d_boundaries); cudaFree(d_q_nope); cudaFree(d_q_rope);
    cudaFree(d_q_rot); cudaFree(d_out); cudaFree(d_lse);
    cudaFree(d_seqlens); cudaFree(d_block_table);
}
#endif  // LAYERSTORM_SOURCE_DIR

// ── CPU reference: absorbed MLA attention ─────────────────────────────────

/// Standard absorbed MLA attention on original (unquantized) BF16 data.
/// Uses double precision internally for numerically stable softmax.
/// q_nope: [batch_heads, d_c], q_rope: [batch_heads, d_rope]
/// c_kv: [seq_len, d_c], k_rope: [seq_len, d_rope]
/// out: [batch_heads, d_c]  (V = K in MLA: output uses c_kv as values)
static void cpu_mla_attention_reference(
    const float* q_nope, const float* q_rope,
    const float* c_kv, const float* k_rope,
    float* out,
    int batch_heads, int seq_len, int d_c, int d_rope, float sm_scale)
{
    std::vector<double> scores(seq_len);
    std::vector<double> weights(seq_len);

    for (int h = 0; h < batch_heads; ++h) {
        const float* qn = q_nope + h * d_c;
        const float* qr = q_rope + h * d_rope;

        // Compute attention scores
        for (int t = 0; t < seq_len; ++t) {
            double nope_score = 0.0;
            for (int d = 0; d < d_c; ++d)
                nope_score += static_cast<double>(qn[d]) * c_kv[t * d_c + d];
            double rope_score = 0.0;
            for (int d = 0; d < d_rope; ++d)
                rope_score += static_cast<double>(qr[d]) * k_rope[t * d_rope + d];
            scores[t] = (nope_score + rope_score) * sm_scale;
        }

        // Stable softmax
        double max_s = *std::max_element(scores.begin(),
                                         scores.begin() + seq_len);
        double sum_exp = 0.0;
        for (int t = 0; t < seq_len; ++t) {
            weights[t] = std::exp(scores[t] - max_s);
            sum_exp += weights[t];
        }
        for (int t = 0; t < seq_len; ++t) weights[t] /= sum_exp;

        // PV: output[h] = weights @ c_kv  (V = K in MLA)
        float* oh = out + h * d_c;
        for (int d = 0; d < d_c; ++d) {
            double acc = 0.0;
            for (int t = 0; t < seq_len; ++t)
                acc += weights[t] * c_kv[t * d_c + d];
            oh[d] = static_cast<float>(acc);
        }
    }
}

// ── Tests: End-to-end TQ decode vs BF16 reference ────────────────────────

#ifdef LAYERSTORM_SOURCE_DIR
TEST(TqMlaAttentionDecode, EndToEndDecodeVsReference) {
    REQUIRES_GPU();

    const int b = 1, s_q = 1, h_q = 8;
    const int d_c = 512, d_rope = 64;
    const int seqlen_k = 128;
    const int page_size = 64;
    const int num_pages = (seqlen_k + page_size - 1) / page_size;
    const int row_bytes = d_c / 2 + 2 + d_rope * 2;  // TQ: 386 B
    const int page_bytes = page_size * row_bytes;
    const int batch_heads = b * s_q * h_q;
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(d_c + d_rope));

    // Load codebook + generate rotation matrix
    auto cb = lc::load_codebook(codebook_dir() + "/codebook_d512_b4.json");
    ASSERT_EQ(cb.n_clusters, 16);
    std::vector<float> Pi(d_c * d_c);
    lc::generate_rotation_matrix_cpu(Pi.data(), d_c, 42);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Generate random BF16 data
    std::vector<__nv_bfloat16> c_kv_bf16(seqlen_k * d_c);
    std::vector<__nv_bfloat16> k_rope_bf16(seqlen_k * d_rope);
    std::vector<__nv_bfloat16> q_nope_bf16(batch_heads * d_c);
    std::vector<__nv_bfloat16> q_rope_bf16(batch_heads * d_rope);
    for (auto& v : c_kv_bf16) v = float_to_bf16(dist(rng));
    for (auto& v : k_rope_bf16) v = float_to_bf16(dist(rng));
    for (auto& v : q_nope_bf16) v = float_to_bf16(dist(rng));
    for (auto& v : q_rope_bf16) v = float_to_bf16(dist(rng));

    // Keep float copies for CPU reference (BF16 round-tripped)
    std::vector<float> c_kv_f(seqlen_k * d_c);
    std::vector<float> k_rope_f(seqlen_k * d_rope);
    std::vector<float> q_nope_f(batch_heads * d_c);
    std::vector<float> q_rope_f(batch_heads * d_rope);
    for (int i = 0; i < seqlen_k * d_c; ++i)
        c_kv_f[i] = bf16_to_float(c_kv_bf16[i]);
    for (int i = 0; i < seqlen_k * d_rope; ++i)
        k_rope_f[i] = bf16_to_float(k_rope_bf16[i]);
    for (int i = 0; i < batch_heads * d_c; ++i)
        q_nope_f[i] = bf16_to_float(q_nope_bf16[i]);
    for (int i = 0; i < batch_heads * d_rope; ++i)
        q_rope_f[i] = bf16_to_float(q_rope_bf16[i]);

    // Slot mapping + block table
    std::vector<int> slots(seqlen_k);
    for (int i = 0; i < seqlen_k; ++i) slots[i] = i;
    std::vector<int> block_table(num_pages);
    for (int i = 0; i < num_pages; ++i) block_table[i] = i;

    // Device allocations
    __nv_bfloat16* d_c_kv = nullptr;
    __nv_bfloat16* d_k_rope = nullptr;
    uint8_t* d_cache = nullptr;
    int* d_slots = nullptr;
    float* d_Pi = nullptr;
    float* d_centroids = nullptr;
    float* d_boundaries = nullptr;
    __nv_bfloat16* d_q_nope = nullptr;
    __nv_bfloat16* d_q_rope = nullptr;
    float* d_q_rot = nullptr;
    float* d_out_rot = nullptr;    // FP32 decode output (rotated space)
    float* d_lse = nullptr;
    __nv_bfloat16* d_out_bf16 = nullptr;  // BF16 final output
    int* d_seqlens = nullptr;
    int* d_block_table = nullptr;

    CUDA_CHECK(cudaMalloc(&d_c_kv, seqlen_k * d_c * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k_rope, seqlen_k * d_rope * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cache, num_pages * page_bytes));
    CUDA_CHECK(cudaMalloc(&d_slots, seqlen_k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Pi, d_c * d_c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, cb.n_clusters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_boundaries,
        cb.interior_boundaries.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_nope, batch_heads * d_c * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q_rope, batch_heads * d_rope * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q_rot, batch_heads * d_c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_rot, batch_heads * d_c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lse, batch_heads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_bf16, batch_heads * d_c * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_seqlens, b * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_table, num_pages * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_cache, 0, num_pages * page_bytes));

    CUDA_CHECK(cudaMemcpy(d_c_kv, c_kv_bf16.data(),
        c_kv_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_rope, k_rope_bf16.data(),
        k_rope_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, slots.data(),
        slots.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Pi, Pi.data(),
        d_c * d_c * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, cb.centroids.data(),
        cb.n_clusters * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_boundaries, cb.interior_boundaries.data(),
        cb.interior_boundaries.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_nope, q_nope_bf16.data(),
        q_nope_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_rope, q_rope_bf16.data(),
        q_rope_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    int seqlen = seqlen_k;
    CUDA_CHECK(cudaMemcpy(d_seqlens, &seqlen, sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_table, block_table.data(),
        num_pages * sizeof(int), cudaMemcpyHostToDevice));

    // ── GPU Pipeline ────────────────────────────────────────────────────

    // Step 1: Populate TQ cache
    sm120::prep::TqFusedKAppendParams ka{};
    ka.c_kv = d_c_kv;
    ka.k_rope = d_k_rope;
    ka.kv_cache = d_cache;
    ka.cache_stride_block = page_bytes;
    ka.cache_stride_row = row_bytes;
    ka.slot_mapping = d_slots;
    ka.Pi = d_Pi;
    ka.centroids = d_centroids;
    ka.decision_boundaries = d_boundaries;
    ka.num_tokens = seqlen_k;
    ka.d_c = d_c;
    ka.d_rope = d_rope;
    ka.page_size = page_size;
    ka.num_centroids = cb.n_clusters;
    lc::launch_tq_k_append(ka, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Pre-rotate Q nope
    sm120::prep::TqQRotateParams qr{};
    qr.q_nope = d_q_nope;
    qr.Pi = d_Pi;
    qr.q_rot = d_q_rot;
    qr.batch_heads = batch_heads;
    qr.d_c = d_c;
    lc::launch_tq_q_rotate(qr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: Dense decode (num_sm_parts=1, no split-KV)
    sm120::decode::tq_dense::TqDenseDecodeParams dp{};
    std::memset(&dp, 0, sizeof(dp));
    dp.b = b; dp.s_q = s_q; dp.h_q = h_q; dp.h_kv = 1;
    dp.d_c = d_c; dp.d_rope = d_rope;
    dp.sm_scale = sm_scale;
    dp.q_rot = d_q_rot;
    dp.q_rope = d_q_rope;
    dp.kv_cache = d_cache;
    dp.cache_stride_block = page_bytes;
    dp.cache_stride_row = row_bytes;
    dp.block_table = d_block_table;
    dp.block_table_batch_stride = num_pages;
    dp.page_block_size = page_size;
    dp.seqlens_k = d_seqlens;
    dp.centroids = d_centroids;
    dp.out = d_out_rot;
    dp.lse = d_lse;
    dp.stride_o_b = s_q * h_q * d_c;
    dp.stride_o_s_q = h_q * d_c;
    dp.stride_o_h_q = d_c;
    dp.stride_lse_b = s_q * h_q;
    dp.stride_lse_s_q = h_q;
    dp.num_sm_parts = 1;
    dp.stream = nullptr;
    lc::launch_decode_dense_tq(dp);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: Inverse rotate  →  BF16 output in original space
    sm120::prep::TqVRotateBackParams vr{};
    vr.out_rotated = d_out_rot;
    vr.Pi = d_Pi;
    vr.out_final = d_out_bf16;
    vr.batch_heads = batch_heads;
    vr.d_c = d_c;
    lc::launch_tq_v_rotate_back(vr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Read back GPU result ────────────────────────────────────────────

    std::vector<__nv_bfloat16> out_bf16_host(batch_heads * d_c);
    CUDA_CHECK(cudaMemcpy(out_bf16_host.data(), d_out_bf16,
        batch_heads * d_c * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    std::vector<float> gpu_out(batch_heads * d_c);
    for (int i = 0; i < batch_heads * d_c; ++i)
        gpu_out[i] = bf16_to_float(out_bf16_host[i]);

    // ── CPU reference ───────────────────────────────────────────────────

    std::vector<float> ref_out(batch_heads * d_c);
    cpu_mla_attention_reference(
        q_nope_f.data(), q_rope_f.data(),
        c_kv_f.data(), k_rope_f.data(),
        ref_out.data(),
        batch_heads, seqlen_k, d_c, d_rope, sm_scale);

    // ── Validate ────────────────────────────────────────────────────────

    // Sanity: all outputs finite and non-zero
    bool any_nonzero = false;
    for (int i = 0; i < batch_heads * d_c; ++i) {
        EXPECT_FALSE(std::isnan(gpu_out[i])) << "Output NaN at " << i;
        EXPECT_FALSE(std::isinf(gpu_out[i])) << "Output Inf at " << i;
        if (std::abs(gpu_out[i]) > 1e-8f) any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero) << "All output elements are zero";

    // Per-head cosine similarity: TQ 4-bit output vs BF16 reference
    for (int h = 0; h < batch_heads; ++h) {
        float cos = cosine_similarity(
            &gpu_out[h * d_c], &ref_out[h * d_c], d_c);
        EXPECT_GE(cos, 0.98f)
            << "Head " << h << " end-to-end cosine too low: " << cos;
    }

    // Cleanup
    cudaFree(d_c_kv); cudaFree(d_k_rope); cudaFree(d_cache);
    cudaFree(d_slots); cudaFree(d_Pi); cudaFree(d_centroids);
    cudaFree(d_boundaries); cudaFree(d_q_nope); cudaFree(d_q_rope);
    cudaFree(d_q_rot); cudaFree(d_out_rot); cudaFree(d_lse);
    cudaFree(d_out_bf16); cudaFree(d_seqlens); cudaFree(d_block_table);
}
#endif  // LAYERSTORM_SOURCE_DIR

// ── CPU reference for mla_combine (log-sum-exp merge) ─────────────────────

static void cpu_mla_combine(
    const float* lseaccum,   // [num_splits, num_q_seqs]
    const float* oaccum,     // [num_splits, num_q_seqs, d_v]
    float* out,              // [num_q_seqs, d_v]
    float* out_lse,          // [num_q_seqs]
    int num_splits, int num_q_seqs, int d_v)
{
    for (int q = 0; q < num_q_seqs; ++q) {
        // Find max LSE across splits (base-2 log space)
        float max_lse = -std::numeric_limits<float>::infinity();
        for (int s = 0; s < num_splits; ++s) {
            max_lse = std::max(max_lse, lseaccum[s * num_q_seqs + q]);
        }
        if (std::isinf(max_lse) && max_lse < 0.0f) max_lse = 0.0f;

        // Global LSE via log-sum-exp in base 2
        float sum_exp = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            sum_exp += std::exp2f(lseaccum[s * num_q_seqs + q] - max_lse);
        }
        float global_lse = (sum_exp == 0.0f || std::isnan(sum_exp))
            ? std::numeric_limits<float>::infinity()
            : std::log2f(sum_exp) + max_lse;

        // Kernel writes LSE as ln (global_lse / M_LOG2E = global_lse * ln(2))
        out_lse[q] = global_lse / static_cast<float>(M_LOG2E);

        // Accumulate rescaled partial outputs
        for (int d = 0; d < d_v; ++d) {
            float acc = 0.0f;
            for (int s = 0; s < num_splits; ++s) {
                float scale = std::exp2f(lseaccum[s * num_q_seqs + q] - global_lse);
                acc += scale * oaccum[(s * num_q_seqs + q) * d_v + d];
            }
            out[q * d_v + d] = acc;
        }
    }
}

// ── Tests: mla_combine<float> round-trip ──────────────────────────────────

TEST(TqMlaAttentionCombine, CombineF32RoundTrip) {
    REQUIRES_GPU();

    const int b = 1, h_q = 8, h_k = 1, s_q = 1;
    const int d_v = 512;
    const int num_splits = 3;
    const int num_q_seqs = h_q * s_q;  // 8

    // Generate synthetic split-KV data
    std::mt19937 rng(314);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> lse_dist(-1.0f, 3.0f);

    std::vector<float> lseaccum(num_splits * num_q_seqs);
    std::vector<float> oaccum(num_splits * num_q_seqs * d_v);
    for (auto& v : lseaccum) v = lse_dist(rng);
    for (auto& v : oaccum) v = dist(rng);

    std::vector<int> num_splits_arr = {0, num_splits};

    // CPU reference
    std::vector<float> ref_out(num_q_seqs * d_v);
    std::vector<float> ref_lse(num_q_seqs);
    cpu_mla_combine(lseaccum.data(), oaccum.data(),
                    ref_out.data(), ref_lse.data(),
                    num_splits, num_q_seqs, d_v);

    // Device allocations
    float* d_o = nullptr;
    float* d_lse = nullptr;
    float* d_lseaccum = nullptr;
    float* d_oaccum = nullptr;
    int* d_num_splits = nullptr;

    CUDA_CHECK(cudaMalloc(&d_o, num_q_seqs * d_v * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lse, num_q_seqs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lseaccum, lseaccum.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oaccum, oaccum.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_num_splits, num_splits_arr.size() * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_o, 0, num_q_seqs * d_v * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_lse, 0, num_q_seqs * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lseaccum, lseaccum.data(),
        lseaccum.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oaccum, oaccum.data(),
        oaccum.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_splits, num_splits_arr.data(),
        num_splits_arr.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Fill combine params
    MlaCombineParams params{};
    params.b = b;
    params.h_q = h_q;
    params.h_k = h_k;
    params.q_seq_per_hk = h_q * s_q;
    params.d_v = d_v;
    params.o_ptr = d_o;
    params.softmax_lse_ptr = d_lse;
    params.o_batch_stride = s_q * h_q * d_v;
    params.o_head_stride = s_q * d_v;
    params.o_row_stride = d_v;
    params.num_splits_ptr = d_num_splits;
    params.num_sm_parts = num_splits;
    params.softmax_lseaccum_ptr = d_lseaccum;
    params.oaccum_ptr = d_oaccum;

    lc::launch_mla_combine_f32(params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back
    std::vector<float> out_host(num_q_seqs * d_v);
    std::vector<float> lse_host(num_q_seqs);
    CUDA_CHECK(cudaMemcpy(out_host.data(), d_o,
        out_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lse_host.data(), d_lse,
        lse_host.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify LSE
    for (int q = 0; q < num_q_seqs; ++q) {
        EXPECT_NEAR(lse_host[q], ref_lse[q], 1e-4f)
            << "LSE mismatch at q=" << q;
    }

    // Verify output: per-head cosine + element-wise relative error
    for (int q = 0; q < num_q_seqs; ++q) {
        float cos = cosine_similarity(
            &out_host[q * d_v], &ref_out[q * d_v], d_v);
        EXPECT_GT(cos, 0.99999f)
            << "Head " << q << " cosine too low: " << cos;

        for (int d = 0; d < d_v; ++d) {
            float ref = ref_out[q * d_v + d];
            float got = out_host[q * d_v + d];
            float rel_err = std::abs(got - ref) / (std::abs(ref) + 1e-8f);
            EXPECT_LT(rel_err, 1e-3f)
                << "Elem (" << q << "," << d << ") got=" << got
                << " ref=" << ref << " rel_err=" << rel_err;
        }
    }

    cudaFree(d_o); cudaFree(d_lse); cudaFree(d_lseaccum);
    cudaFree(d_oaccum); cudaFree(d_num_splits);
}

// ── Tests: combine<float> matches combine<bf16> on BF16-range inputs ──────

TEST(TqMlaAttentionCombine, F32MatchesBF16OnBF16RangeInputs) {
    REQUIRES_GPU();

    const int b = 1, h_q = 8, h_k = 1, s_q = 1;
    const int d_v = 512;
    const int num_splits = 4;
    const int num_q_seqs = h_q * s_q;

    // Generate data — oaccum round-tripped through BF16 for exact representability
    std::mt19937 rng(271);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> lse_dist(0.0f, 5.0f);

    std::vector<float> lseaccum(num_splits * num_q_seqs);
    std::vector<float> oaccum(num_splits * num_q_seqs * d_v);
    for (auto& v : lseaccum) v = lse_dist(rng);
    for (auto& v : oaccum) {
        v = bf16_to_float(float_to_bf16(dist(rng)));
    }

    std::vector<int> num_splits_arr = {0, num_splits};

    // Device allocations — two output sets
    float* d_o_f32 = nullptr;
    __nv_bfloat16* d_o_bf16 = nullptr;
    float* d_lse_f32 = nullptr;
    float* d_lse_bf16 = nullptr;
    float* d_lseaccum = nullptr;
    float* d_oaccum = nullptr;
    int* d_num_splits = nullptr;

    CUDA_CHECK(cudaMalloc(&d_o_f32, num_q_seqs * d_v * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o_bf16, num_q_seqs * d_v * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_lse_f32, num_q_seqs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lse_bf16, num_q_seqs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lseaccum, lseaccum.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oaccum, oaccum.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_num_splits, num_splits_arr.size() * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_o_f32, 0, num_q_seqs * d_v * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_o_bf16, 0, num_q_seqs * d_v * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_lse_f32, 0, num_q_seqs * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_lse_bf16, 0, num_q_seqs * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lseaccum, lseaccum.data(),
        lseaccum.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oaccum, oaccum.data(),
        oaccum.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_splits, num_splits_arr.data(),
        num_splits_arr.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Run FP32 combine
    MlaCombineParams params_f32{};
    params_f32.b = b;
    params_f32.h_q = h_q;
    params_f32.h_k = h_k;
    params_f32.q_seq_per_hk = h_q * s_q;
    params_f32.d_v = d_v;
    params_f32.o_ptr = d_o_f32;
    params_f32.softmax_lse_ptr = d_lse_f32;
    params_f32.o_batch_stride = s_q * h_q * d_v;
    params_f32.o_head_stride = s_q * d_v;
    params_f32.o_row_stride = d_v;
    params_f32.num_splits_ptr = d_num_splits;
    params_f32.num_sm_parts = num_splits;
    params_f32.softmax_lseaccum_ptr = d_lseaccum;
    params_f32.oaccum_ptr = d_oaccum;

    lc::launch_mla_combine_f32(params_f32, nullptr);

    // Run BF16 combine (same input data, different output buffer)
    MlaCombineParams params_bf16{};
    params_bf16.b = b;
    params_bf16.h_q = h_q;
    params_bf16.h_k = h_k;
    params_bf16.q_seq_per_hk = h_q * s_q;
    params_bf16.d_v = d_v;
    params_bf16.o_ptr = d_o_bf16;
    params_bf16.softmax_lse_ptr = d_lse_bf16;
    params_bf16.o_batch_stride = s_q * h_q * d_v;
    params_bf16.o_head_stride = s_q * d_v;
    params_bf16.o_row_stride = d_v;
    params_bf16.num_splits_ptr = d_num_splits;
    params_bf16.num_sm_parts = num_splits;
    params_bf16.softmax_lseaccum_ptr = d_lseaccum;
    params_bf16.oaccum_ptr = d_oaccum;

    lc::launch_mla_combine(params_bf16, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back both
    std::vector<float> out_f32(num_q_seqs * d_v);
    std::vector<__nv_bfloat16> out_bf16(num_q_seqs * d_v);
    std::vector<float> lse_f32(num_q_seqs), lse_bf16(num_q_seqs);
    CUDA_CHECK(cudaMemcpy(out_f32.data(), d_o_f32,
        out_f32.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_bf16.data(), d_o_bf16,
        out_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lse_f32.data(), d_lse_f32,
        lse_f32.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lse_bf16.data(), d_lse_bf16,
        lse_bf16.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // LSE should match (both write float32)
    for (int q = 0; q < num_q_seqs; ++q) {
        EXPECT_NEAR(lse_f32[q], lse_bf16[q], 1e-5f)
            << "LSE mismatch at q=" << q;
    }

    // Per-head cosine: FP32 output vs BF16 output (converted to float)
    for (int q = 0; q < num_q_seqs; ++q) {
        std::vector<float> f32_head(d_v), bf16_head(d_v);
        for (int d = 0; d < d_v; ++d) {
            f32_head[d] = out_f32[q * d_v + d];
            bf16_head[d] = bf16_to_float(out_bf16[q * d_v + d]);
        }
        float cos = cosine_similarity(f32_head.data(), bf16_head.data(), d_v);
        EXPECT_GT(cos, 0.9999f)
            << "Head " << q << " f32-vs-bf16 cosine: " << cos;
    }

    // Element-wise: FP32 cast to BF16 should match BF16 kernel output
    for (int q = 0; q < num_q_seqs; ++q) {
        for (int d = 0; d < d_v; ++d) {
            float f32_as_bf16 = bf16_to_float(float_to_bf16(out_f32[q * d_v + d]));
            float bf16_val = bf16_to_float(out_bf16[q * d_v + d]);
            float err = std::abs(f32_as_bf16 - bf16_val);
            EXPECT_LT(err, 0.02f)
                << "Mismatch at (" << q << "," << d << "): "
                << "f32->bf16=" << f32_as_bf16 << " bf16=" << bf16_val;
        }
    }

    cudaFree(d_o_f32); cudaFree(d_o_bf16);
    cudaFree(d_lse_f32); cudaFree(d_lse_bf16);
    cudaFree(d_lseaccum); cudaFree(d_oaccum); cudaFree(d_num_splits);
}
