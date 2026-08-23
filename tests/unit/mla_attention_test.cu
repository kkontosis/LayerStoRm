// Unit tests for MLA attention prep kernels and dispatch wrappers.
//
// Tests: FusedQQuant round-trip, FusedKAppend+DequantCKV round-trip,
// decode dense/sparse dispatch, SplitKV consistency.

#include "compute/kernels/attention/mla_attention.h"
#include "compute/kernels/attention/prep_params.h"

// SMXX
#include "smxx/get_mla_metadata.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;

#define CUDA_CHECK(expr)                                                 \
    do {                                                                 \
        cudaError_t _err = (expr);                                       \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_err); \
    } while (0)

// ── Host BF16 helpers ────────────────────────────────────────────────────────

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

// ── Tests: FusedQQuant ──────────────────────────────────────────────────────

TEST(MlaAttentionPrep, FusedQQuantRoundTrip) {
    REQUIRES_GPU();

    // V3.2 geometry: d_qk=576, d_nope=512, d_rope=64
    const int s_q = 4, h_q = 8, d_qk = 576, d_nope = 512;
    const int d_rope = d_qk - d_nope;  // 64

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Generate random Q data in BF16
    std::vector<__nv_bfloat16> q_host(s_q * h_q * d_qk);
    std::vector<float> q_float(s_q * h_q * d_qk);
    for (int i = 0; i < s_q * h_q * d_qk; ++i) {
        float v = dist(rng);
        q_float[i] = v;
        q_host[i] = float_to_bf16(v);
    }

    // Device buffers
    __nv_bfloat16* d_q = nullptr;
    __nv_fp8_e4m3* d_q_nope = nullptr;
    __nv_bfloat16* d_q_rope = nullptr;
    float* d_q_scales = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, q_host.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q_nope, s_q * h_q * d_nope * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_q_rope, s_q * h_q * d_rope * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q_scales, s_q * h_q * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, q_host.data(), q_host.size() * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));

    sm120::prep::FusedQQuantParams params{};
    params.q_bf16 = d_q;
    params.q_nope_fp8 = d_q_nope;
    params.q_rope_bf16 = d_q_rope;
    params.q_scales = d_q_scales;
    params.s_q = s_q;
    params.h_q = h_q;
    params.d_qk = d_qk;
    params.d_nope = d_nope;

    lc::launch_fused_q_quant(params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back scales
    std::vector<float> scales(s_q * h_q);
    CUDA_CHECK(cudaMemcpy(scales.data(), d_q_scales, scales.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // Verify: all scales > 0 (they are amax / 448.0)
    for (int i = 0; i < s_q * h_q; ++i) {
        EXPECT_GT(scales[i], 0.0f) << "Scale should be positive at " << i;
        EXPECT_FALSE(std::isnan(scales[i])) << "Scale is NaN at " << i;
    }

    // Read back ROPE output and verify it's pre-scaled (not NaN)
    std::vector<__nv_bfloat16> rope_out(s_q * h_q * d_rope);
    CUDA_CHECK(cudaMemcpy(rope_out.data(), d_q_rope,
                           rope_out.size() * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < rope_out.size(); ++i) {
        float v = bf16_to_float(rope_out[i]);
        EXPECT_FALSE(std::isnan(v)) << "ROPE output NaN at " << i;
    }

    // Read back FP8 NOPE and verify rough round-trip:
    // dequant(fp8) * scale ≈ original_nope (within FP8 tolerance)
    std::vector<__nv_fp8_e4m3> nope_out(s_q * h_q * d_nope);
    CUDA_CHECK(cudaMemcpy(nope_out.data(), d_q_nope,
                           nope_out.size() * sizeof(__nv_fp8_e4m3),
                           cudaMemcpyDeviceToHost));

    int mismatch_count = 0;
    for (int sq = 0; sq < s_q; ++sq) {
        for (int hq = 0; hq < h_q; ++hq) {
            float scale = scales[sq * h_q + hq];
            for (int d = 0; d < d_nope; ++d) {
                int idx = (sq * h_q + hq) * d_nope + d;
                float fp8_val = static_cast<float>(nope_out[idx]);
                float reconstructed = fp8_val * scale;
                float original = q_float[(sq * h_q + hq) * d_qk + d];
                // FP8 e4m3 has ±448 range with ~7.8% relative error
                if (std::abs(original) > 0.01f) {
                    float rel_err = std::abs(reconstructed - original) / std::abs(original);
                    if (rel_err > 0.15f) mismatch_count++;
                }
            }
        }
    }
    // Allow up to 5% mismatches (FP8 quantization has outlier errors)
    float mismatch_frac = static_cast<float>(mismatch_count) / (s_q * h_q * d_nope);
    EXPECT_LT(mismatch_frac, 0.05f) << mismatch_count << " of "
        << (s_q * h_q * d_nope) << " elements have >15% relative error";

    cudaFree(d_q);
    cudaFree(d_q_nope);
    cudaFree(d_q_rope);
    cudaFree(d_q_scales);
}

// ── Tests: FusedKAppend + DequantCKV round-trip ─────────────────────────────

TEST(MlaAttentionPrep, KAppendDequantRoundTrip) {
    REQUIRES_GPU();

    const int num_tokens = 8;
    const int d_c = 512;   // V3.2 kv_lora_rank
    const int d_rope = 64;
    const int d_qk = d_c + d_rope;
    const int page_size = 64;
    const int num_pages = 1;
    // Row stride: d_c + 4 (scale) + d_rope*2 (BF16) = 644 bytes
    const int row_bytes = d_c + 4 + d_rope * 2;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Create c_kv and k_rope in BF16
    std::vector<__nv_bfloat16> ckv_host(num_tokens * d_c);
    std::vector<__nv_bfloat16> rope_host(num_tokens * d_rope);
    std::vector<float> ckv_float(num_tokens * d_c);
    std::vector<float> rope_float(num_tokens * d_rope);

    for (int i = 0; i < num_tokens * d_c; ++i) {
        float v = dist(rng);
        ckv_float[i] = v;
        ckv_host[i] = float_to_bf16(v);
    }
    for (int i = 0; i < num_tokens * d_rope; ++i) {
        float v = dist(rng);
        rope_float[i] = v;
        rope_host[i] = float_to_bf16(v);
    }

    // Slot mapping: token i → slot i (simple sequential)
    std::vector<int> slot_mapping(num_tokens);
    for (int i = 0; i < num_tokens; ++i) slot_mapping[i] = i;

    // Device buffers
    __nv_bfloat16* d_ckv = nullptr;
    __nv_bfloat16* d_rope_buf = nullptr;
    __nv_fp8_e4m3* d_cache = nullptr;
    int* d_slots = nullptr;
    __nv_bfloat16* d_k_out = nullptr;
    int* d_indices = nullptr;

    int64_t cache_bytes = (int64_t)num_pages * page_size * row_bytes;

    CUDA_CHECK(cudaMalloc(&d_ckv, num_tokens * d_c * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_rope_buf, num_tokens * d_rope * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cache, cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_slots, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_k_out, num_tokens * d_qk * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_indices, num_tokens * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_cache, 0, cache_bytes));
    CUDA_CHECK(cudaMemcpy(d_ckv, ckv_host.data(),
                           num_tokens * d_c * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rope_buf, rope_host.data(),
                           num_tokens * d_rope * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, slot_mapping.data(),
                           num_tokens * sizeof(int), cudaMemcpyHostToDevice));
    // Indices for dequant: same as slot mapping
    CUDA_CHECK(cudaMemcpy(d_indices, slot_mapping.data(),
                           num_tokens * sizeof(int), cudaMemcpyHostToDevice));

    // Step 1: FusedKAppend — write to cache
    sm120::prep::FusedKAppendParams k_params{};
    k_params.c_kv = d_ckv;
    k_params.k_rope = d_rope_buf;
    k_params.kv_cache = d_cache;
    k_params.cache_stride_block = (int64_t)page_size * row_bytes;
    k_params.cache_stride_row = row_bytes;
    k_params.slot_mapping = d_slots;
    k_params.num_tokens = num_tokens;
    k_params.d_c = d_c;
    k_params.d_rope = d_rope;
    k_params.page_size = page_size;

    lc::launch_fused_k_append(k_params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: DequantCKV — read from cache
    sm120::prep::DequantCKVIndexedParams dq_params{};
    dq_params.kv_cache = d_cache;
    dq_params.cache_stride_block = (int64_t)page_size * row_bytes;
    dq_params.cache_stride_row = row_bytes;
    dq_params.page_size = page_size;
    dq_params.indices = d_indices;
    dq_params.num_fetch = num_tokens;
    dq_params.k_out = d_k_out;
    dq_params.d_c = d_c;
    dq_params.d_rope = d_rope;

    lc::launch_dequant_ckv_indexed(dq_params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back dequantized output
    std::vector<__nv_bfloat16> k_out_host(num_tokens * d_qk);
    CUDA_CHECK(cudaMemcpy(k_out_host.data(), d_k_out,
                           num_tokens * d_qk * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost));

    // Verify round-trip: dequant(quant(ckv)) ≈ ckv, dequant(rope) ≈ rope
    int nope_errors = 0;
    int rope_errors = 0;
    for (int t = 0; t < num_tokens; ++t) {
        // NOPE component (d_c dims): FP8 round-trip has quantization error
        for (int d = 0; d < d_c; ++d) {
            float orig = ckv_float[t * d_c + d];
            float result = bf16_to_float(k_out_host[t * d_qk + d]);
            if (std::abs(orig) > 0.01f) {
                float rel = std::abs(result - orig) / std::abs(orig);
                if (rel > 0.15f) nope_errors++;
            }
        }
        // ROPE component (d_rope dims): BF16→pre-scaled→un-scaled round-trip
        for (int d = 0; d < d_rope; ++d) {
            float orig = rope_float[t * d_rope + d];
            float result = bf16_to_float(k_out_host[t * d_qk + d_c + d]);
            if (std::abs(orig) > 0.01f) {
                float rel = std::abs(result - orig) / std::abs(orig);
                // ROPE has two BF16 conversions + scale mul/div, larger error
                if (rel > 0.20f) rope_errors++;
            }
        }
    }

    float nope_frac = static_cast<float>(nope_errors) / (num_tokens * d_c);
    float rope_frac = static_cast<float>(rope_errors) / (num_tokens * d_rope);
    EXPECT_LT(nope_frac, 0.05f) << "NOPE round-trip: " << nope_errors << " errors";
    EXPECT_LT(rope_frac, 0.05f) << "ROPE round-trip: " << rope_errors << " errors";

    cudaFree(d_ckv);
    cudaFree(d_rope_buf);
    cudaFree(d_cache);
    cudaFree(d_slots);
    cudaFree(d_k_out);
    cudaFree(d_indices);
}

// ── Tests: GetMlaMetadata smoke ─────────────────────────────────────────────

TEST(MlaAttention, GetMlaMetadataSmoke) {
    REQUIRES_GPU();

    const int batch_size = 4;
    const int block_size_n = 64;
    const int num_sm_parts = 8;

    // Simple seqlens: each batch has 128 tokens
    std::vector<int> seqlens_k(batch_size, 128);

    int* d_seqlens = nullptr;
    int* d_metadata = nullptr;
    int* d_num_splits = nullptr;

    CUDA_CHECK(cudaMalloc(&d_seqlens, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_metadata, num_sm_parts * 8 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_splits, (batch_size + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_seqlens, seqlens_k.data(),
                           batch_size * sizeof(int), cudaMemcpyHostToDevice));

    GetMlaMetadataParams params{};
    params.seqlens_k_ptr = d_seqlens;
    params.tile_scheduler_metadata_ptr = d_metadata;
    params.num_splits_ptr = d_num_splits;
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = 1;
    params.num_sm_parts = num_sm_parts;
    params.topk = -1;  // dense mode

    lc::launch_get_mla_metadata(params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back num_splits — should be non-zero
    std::vector<int> num_splits(batch_size + 1);
    CUDA_CHECK(cudaMemcpy(num_splits.data(), d_num_splits,
                           (batch_size + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost));

    // num_splits[0] should be 0 (cumulative), last entry > 0
    EXPECT_EQ(num_splits[0], 0);
    EXPECT_GT(num_splits[batch_size], 0);

    cudaFree(d_seqlens);
    cudaFree(d_metadata);
    cudaFree(d_num_splits);
}

// ── TD-PREFILL-CHUNK-ATTN: strided (interleaved) k_append A/B ───────────────
//
// The engine hands fused_k_append pointers into ONE interleaved kv_a output
// row [c_kv | k_pe] (both source strides = d_c + d_rope). The kernel's former
// tight-row addressing was exact only at num_tokens == 1 — every later token
// of a multi-token prefill chunk appended garbage KV. Lock-in: appending a
// 4-token interleaved batch with explicit strides must produce BIT-IDENTICAL
// cache rows to appending each token separately from tight copies.
TEST(MlaAttentionPrep, KAppendStridedInterleavedBatchBitEqualsPerToken) {
    REQUIRES_GPU();

    const int num_tokens = 4;
    const int d_c = 512;
    const int d_rope = 64;
    const int kv_dim = d_c + d_rope;   // interleaved source row width
    const int page_size = 64;
    const int row_bytes = d_c + 4 + d_rope * 2;

    std::mt19937 rng(321);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Interleaved source rows [c_kv | k_pe], stride kv_dim.
    std::vector<__nv_bfloat16> inter_host(num_tokens * kv_dim);
    for (auto& v : inter_host) v = float_to_bf16(dist(rng));

    std::vector<int> slots(num_tokens);
    for (int i = 0; i < num_tokens; ++i) slots[i] = i;

    __nv_bfloat16* d_inter = nullptr;
    __nv_bfloat16* d_tight = nullptr;   // one token's tight [c_kv][rope]
    __nv_fp8_e4m3* d_cache_a = nullptr;
    __nv_fp8_e4m3* d_cache_b = nullptr;
    int* d_slots = nullptr;
    const int64_t cache_bytes = (int64_t)page_size * row_bytes;

    CUDA_CHECK(cudaMalloc(&d_inter, inter_host.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_tight, kv_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cache_a, cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_cache_b, cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_slots, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cache_a, 0, cache_bytes));
    CUDA_CHECK(cudaMemset(d_cache_b, 0, cache_bytes));
    CUDA_CHECK(cudaMemcpy(d_inter, inter_host.data(),
                          inter_host.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, slots.data(), num_tokens * sizeof(int),
                          cudaMemcpyHostToDevice));

    // A: ONE batched call over the interleaved rows with explicit strides
    //    (exactly the engine call shape: k_rope points d_c elements in).
    sm120::prep::FusedKAppendParams pa{};
    pa.c_kv = d_inter;
    pa.k_rope = d_inter + d_c;
    pa.src_stride_ckv = kv_dim;
    pa.src_stride_rope = kv_dim;
    pa.kv_cache = d_cache_a;
    pa.cache_stride_block = cache_bytes;
    pa.cache_stride_row = row_bytes;
    pa.slot_mapping = d_slots;
    pa.num_tokens = num_tokens;
    pa.d_c = d_c;
    pa.d_rope = d_rope;
    pa.page_size = page_size;
    lc::launch_fused_k_append(pa, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // B: per-token tight append (zero-stride default = legacy tight layout).
    for (int t = 0; t < num_tokens; ++t) {
        CUDA_CHECK(cudaMemcpy(d_tight, inter_host.data() + t * kv_dim,
                              kv_dim * sizeof(__nv_bfloat16),
                              cudaMemcpyHostToDevice));
        sm120::prep::FusedKAppendParams pb{};
        pb.c_kv = d_tight;
        pb.k_rope = d_tight + d_c;
        pb.kv_cache = d_cache_b;
        pb.cache_stride_block = cache_bytes;
        pb.cache_stride_row = row_bytes;
        pb.slot_mapping = d_slots + t;
        pb.num_tokens = 1;
        pb.d_c = d_c;
        pb.d_rope = d_rope;
        pb.page_size = page_size;
        lc::launch_fused_k_append(pb, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<uint8_t> cache_a(cache_bytes), cache_b(cache_bytes);
    CUDA_CHECK(cudaMemcpy(cache_a.data(), d_cache_a, cache_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cache_b.data(), d_cache_b, cache_bytes,
                          cudaMemcpyDeviceToHost));

    for (int t = 0; t < num_tokens; ++t) {
        const uint8_t* ra = cache_a.data() + (size_t)t * row_bytes;
        const uint8_t* rb = cache_b.data() + (size_t)t * row_bytes;
        ASSERT_EQ(std::memcmp(ra, rb, row_bytes), 0)
            << "cache row " << t << ": strided batch != per-token tight";
    }

    cudaFree(d_inter);
    cudaFree(d_tight);
    cudaFree(d_cache_a);
    cudaFree(d_cache_b);
    cudaFree(d_slots);
}
