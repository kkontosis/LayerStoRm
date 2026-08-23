// KVS-1 / TD-LSE-UNITS-DECODE: LSE unit contract for the SnapMLA FP8 split-KV
// DECODE kernels (sparse_fp8 + dense_fp8) — the values the DCP cross-rank LSE
// combine consumes in graph decode mode (decode_graph_lse_ptr → partial_lse).
//
// The SM120 port scores in NATURAL log units (rS *= sm_scale, exp via
// exp2f(x*LOG2E) == e^x) and accumulates the FULL-PRECISION exp row-sum sL
// BEFORE the FP8 P quantization, but its epilogues kept FlashMLA's log2-unit
// quantized-sum idiom:
//   no-split: lse       = logf(L*P_scale) + M/LOG2E   (spurious ×P_scale, ×ln2 on M)
//   split:    lse_accum = log2f(L*P_scale) + M        (natural M in a log2 expr)
// The split form mis-weighted mla_combine (weights ∝ 2^lse_accum), corrupting
// the multi-split combined OUTPUT whenever splits differ in max or P_scale.
// Fixed (KVS-1): lse = M + log(L); lse_accum = log2(L) + M*LOG2E.
//
// This test locks the contract with a CPU reference computed from the EXACT
// stored values (read-back FP8 cache bytes + quantized Q), covering:
//   - no-split (num_sm_parts=1): external lse NATURAL + output vs reference
//   - multi-split (num_sm_parts=8) + mla_combine: combined output vs
//     reference (the lse_accum weighting discriminator — a planted high-logit
//     token makes one split's max dominate) + combined lse NATURAL.

#include "compute/kernels/attention/mla_attention.h"
#include "compute/kernels/attention/prep_params.h"

#include "smxx/get_mla_metadata.h"
#include "smxx/mla_combine.h"
#include "sm120/decode/sparse_fp8/params.h"
#include "sm120/decode/dense_fp8/params.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace lc = layerstorm::compute;

namespace {

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess)                                           \
            << "CUDA error: " << cudaGetErrorString(_err);                     \
    } while (0)

constexpr int kDC = 512;      // kv_lora_rank / d_nope (V3.2/GLM)
constexpr int kDR = 64;       // qk_rope_head_dim
constexpr int kDQK = kDC + kDR;
constexpr int kHQ = 64;       // heads (full BLOCK_SIZE_M tile)
constexpr int kPage = 64;
constexpr int kRowBytes = kDC + 4 + kDR * 2;  // [512 fp8 | f32 scale | 64 bf16]

float bf16r(float f) { return __bfloat162float(__float2bfloat16(f)); }

// Host FP8 e4m3 decode (bit-exact).
float fp8_e4m3_to_float(uint8_t b) {
    const int sign = (b >> 7) & 1;
    const int exp = (b >> 3) & 0xF;
    const int man = b & 0x7;
    float v;
    if (exp == 0) {
        v = std::ldexp(static_cast<float>(man) / 8.0f, -6);  // subnormal
    } else if (exp == 0xF && man == 0x7) {
        v = std::nanf("");
    } else {
        v = std::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
    }
    return sign ? -v : v;
}

// Effective per-token K/V values as the kernel sees them, parsed from the
// read-back cache bytes: k_nope_eff[d] = fp8dec(byte)*k_scale (the kernel
// folds k_scale into the score dequant); rope stored PRE-SCALED (rope/scale,
// bf16) and re-scaled by the same dequant.
struct TokenRow {
    std::vector<float> nope_fp8;   // [512] raw fp8-decoded (NOT scaled)
    float k_scale;
    std::vector<float> rope_pre;   // [64] pre-scaled bf16 (NOT re-scaled)
};

struct DecodeRig {
    // Device buffers
    void* d_cache = nullptr;
    __nv_fp8_e4m3* d_q_nope = nullptr;
    __nv_bfloat16* d_q_rope = nullptr;
    float* d_q_scales = nullptr;
    int* d_indices = nullptr;      // identity [0..N)
    int* d_block_table = nullptr;  // identity pages
    int* d_seqlens = nullptr;      // [1] = N
    __nv_bfloat16* d_out = nullptr;
    float* d_lse = nullptr;
    float* d_lse_accum = nullptr;
    float* d_o_accum = nullptr;
    int* d_sched_meta = nullptr;
    int* d_num_splits = nullptr;

    int num_tokens = 0;
    int num_pages = 0;
    int max_sm_parts = 0;

    // Host-side effective values for the CPU reference
    std::vector<TokenRow> rows;                // [N]
    std::vector<float> q_nope_dec;             // [h, 512] fp8-decoded
    std::vector<float> q_rope_pre;             // [h, 64]  pre-scaled bf16
    std::vector<float> q_scales;               // [h]

    ~DecodeRig() {
        for (void* p : {d_cache, (void*)d_q_nope, (void*)d_q_rope,
                        (void*)d_q_scales, (void*)d_indices,
                        (void*)d_block_table, (void*)d_seqlens, (void*)d_out,
                        (void*)d_lse, (void*)d_lse_accum, (void*)d_o_accum,
                        (void*)d_sched_meta, (void*)d_num_splits})
            if (p) cudaFree(p);
    }
};

// Build the SnapMLA FP8 cache via the production k_append kernel, quantize Q
// via the production fused_q_quant, and read back every stored value for the
// CPU reference. plant_token >= 0 aligns that token's key with the queries
// (high logit deep in the sequence — makes one split's max dominate).
void build_rig(DecodeRig& rig, int num_tokens, int max_sm_parts,
               int plant_token, unsigned seed) {
    rig.num_tokens = num_tokens;
    rig.num_pages = (num_tokens + kPage - 1) / kPage;
    rig.max_sm_parts = max_sm_parts;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> qf(static_cast<size_t>(kHQ) * kDQK);
    std::vector<float> ckv(static_cast<size_t>(num_tokens) * kDC);
    std::vector<float> rope(static_cast<size_t>(num_tokens) * kDR);
    for (auto& v : qf) v = bf16r(dist(rng));
    for (auto& v : ckv) v = bf16r(dist(rng));
    for (auto& v : rope) v = bf16r(dist(rng));

    if (plant_token >= 0) {
        // Align the planted token's key with the mean query direction, scaled
        // so its logit sits ~10 NATURAL units above the noise floor: its
        // split's max then dominates the combine, and the OLD mixed-unit
        // lse_accum (weights ∝ 2^m·P_scale instead of e^m) under-weights that
        // split by ~(e/2)^10 ≈ e^3.2 — an output error far outside tolerance.
        constexpr float kPlantGain = 110.0f;
        for (int d = 0; d < kDC; ++d) {
            float acc = 0.f;
            for (int h = 0; h < kHQ; ++h) acc += qf[static_cast<size_t>(h) * kDQK + d];
            ckv[static_cast<size_t>(plant_token) * kDC + d] =
                bf16r(acc / kHQ * kPlantGain);
        }
        for (int d = 0; d < kDR; ++d) {
            float acc = 0.f;
            for (int h = 0; h < kHQ; ++h)
                acc += qf[static_cast<size_t>(h) * kDQK + kDC + d];
            rope[static_cast<size_t>(plant_token) * kDR + d] =
                bf16r(acc / kHQ * kPlantGain);
        }
    }

    // ── K append (production prep kernel) ──
    std::vector<__nv_bfloat16> ckv_b(ckv.size()), rope_b(rope.size());
    for (size_t i = 0; i < ckv.size(); ++i) ckv_b[i] = __float2bfloat16(ckv[i]);
    for (size_t i = 0; i < rope.size(); ++i) rope_b[i] = __float2bfloat16(rope[i]);
    std::vector<int> slots(num_tokens);
    for (int i = 0; i < num_tokens; ++i) slots[i] = i;

    __nv_bfloat16 *d_ckv = nullptr, *d_rope = nullptr;
    int* d_slots = nullptr;
    const int64_t cache_bytes = (int64_t)rig.num_pages * kPage * kRowBytes;
    CUDA_CHECK(cudaMalloc(&rig.d_cache, cache_bytes));
    CUDA_CHECK(cudaMemset(rig.d_cache, 0, cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_ckv, ckv_b.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_rope, rope_b.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_slots, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ckv, ckv_b.data(), ckv_b.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rope, rope_b.data(), rope_b.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, slots.data(), num_tokens * sizeof(int),
                          cudaMemcpyHostToDevice));

    sm120::prep::FusedKAppendParams kp{};
    kp.c_kv = d_ckv;
    kp.k_rope = d_rope;
    kp.kv_cache = static_cast<__nv_fp8_e4m3*>(rig.d_cache);
    kp.cache_stride_block = (int64_t)kPage * kRowBytes;
    kp.cache_stride_row = kRowBytes;
    kp.slot_mapping = d_slots;
    kp.num_tokens = num_tokens;
    kp.d_c = kDC;
    kp.d_rope = kDR;
    kp.page_size = kPage;
    lc::launch_fused_k_append(kp, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_ckv); cudaFree(d_rope); cudaFree(d_slots);

    // ── Q quantization (production prep kernel) ──
    std::vector<__nv_bfloat16> qb(qf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    __nv_bfloat16* d_q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, qb.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_q, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&rig.d_q_nope,
                          static_cast<size_t>(kHQ) * kDC));
    CUDA_CHECK(cudaMalloc(&rig.d_q_rope, static_cast<size_t>(kHQ) * kDR * 2));
    CUDA_CHECK(cudaMalloc(&rig.d_q_scales, kHQ * sizeof(float)));

    sm120::prep::FusedQQuantParams qp{};
    qp.q_bf16 = d_q;
    qp.q_nope_fp8 = rig.d_q_nope;
    qp.q_rope_bf16 = rig.d_q_rope;
    qp.q_scales = rig.d_q_scales;
    qp.s_q = 1;
    qp.h_q = kHQ;
    qp.d_qk = kDQK;
    qp.d_nope = kDC;
    lc::launch_fused_q_quant(qp, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_q);

    // ── Read back EXACT stored values for the CPU reference ──
    std::vector<uint8_t> cache_h(cache_bytes);
    CUDA_CHECK(cudaMemcpy(cache_h.data(), rig.d_cache, cache_bytes,
                          cudaMemcpyDeviceToHost));
    rig.rows.resize(num_tokens);
    for (int t = 0; t < num_tokens; ++t) {
        const uint8_t* row = cache_h.data() +
            (int64_t)(t / kPage) * kPage * kRowBytes +
            (int64_t)(t % kPage) * kRowBytes;
        auto& tr = rig.rows[t];
        tr.nope_fp8.resize(kDC);
        for (int d = 0; d < kDC; ++d) tr.nope_fp8[d] = fp8_e4m3_to_float(row[d]);
        std::memcpy(&tr.k_scale, row + kDC, 4);
        tr.rope_pre.resize(kDR);
        const uint16_t* rp = reinterpret_cast<const uint16_t*>(row + kDC + 4);
        for (int d = 0; d < kDR; ++d) {
            __nv_bfloat16 b;
            std::memcpy(&b, &rp[d], 2);
            tr.rope_pre[d] = __bfloat162float(b);
        }
    }

    std::vector<uint8_t> qn(static_cast<size_t>(kHQ) * kDC);
    std::vector<uint16_t> qr(static_cast<size_t>(kHQ) * kDR);
    rig.q_scales.resize(kHQ);
    CUDA_CHECK(cudaMemcpy(qn.data(), rig.d_q_nope, qn.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qr.data(), rig.d_q_rope, qr.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rig.q_scales.data(), rig.d_q_scales,
                          kHQ * sizeof(float), cudaMemcpyDeviceToHost));
    rig.q_nope_dec.resize(qn.size());
    for (size_t i = 0; i < qn.size(); ++i)
        rig.q_nope_dec[i] = fp8_e4m3_to_float(qn[i]);
    rig.q_rope_pre.resize(qr.size());
    for (size_t i = 0; i < qr.size(); ++i) {
        __nv_bfloat16 b;
        std::memcpy(&b, &qr[i], 2);
        rig.q_rope_pre[i] = __bfloat162float(b);
    }

    // ── Common decode I/O buffers ──
    std::vector<int> indices(num_tokens);
    for (int i = 0; i < num_tokens; ++i) indices[i] = i;
    CUDA_CHECK(cudaMalloc(&rig.d_indices, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(rig.d_indices, indices.data(),
                          num_tokens * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> btable(rig.num_pages);
    for (int i = 0; i < rig.num_pages; ++i) btable[i] = i;
    CUDA_CHECK(cudaMalloc(&rig.d_block_table, rig.num_pages * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(rig.d_block_table, btable.data(),
                          rig.num_pages * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&rig.d_seqlens, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(rig.d_seqlens, &num_tokens, sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&rig.d_out, static_cast<size_t>(kHQ) * kDC * 2));
    CUDA_CHECK(cudaMalloc(&rig.d_lse, kHQ * sizeof(float)));
    const int max_splits = max_sm_parts + 2;
    CUDA_CHECK(cudaMalloc(&rig.d_lse_accum,
                          static_cast<size_t>(max_splits) * kHQ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rig.d_o_accum, static_cast<size_t>(max_splits) *
                          kHQ * kDC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rig.d_sched_meta, max_sm_parts * 8 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&rig.d_num_splits, 2 * sizeof(int)));
}

// CPU reference from the exact stored values. score_t(h) =
//   [ sum_d qn(h,d)*kn(t,d) + sum_d qr(h,d)*rope_pre(t,d) ]
//     * q_scale(h) * k_scale(t) * sm_scale
// out(h,d) = softmax-weighted sum of v_eff(t,d) = kn(t,d)*k_scale(t).
void cpu_reference(const DecodeRig& rig, float sm_scale,
                   std::vector<float>& out, std::vector<float>& lse) {
    const int N = rig.num_tokens;
    out.assign(static_cast<size_t>(kHQ) * kDC, 0.0f);
    lse.assign(kHQ, 0.0f);
    std::vector<double> scores(N);
    for (int h = 0; h < kHQ; ++h) {
        const float* qn = rig.q_nope_dec.data() + static_cast<size_t>(h) * kDC;
        const float* qr = rig.q_rope_pre.data() + static_cast<size_t>(h) * kDR;
        double m = -1e300;
        for (int t = 0; t < N; ++t) {
            const auto& tr = rig.rows[t];
            double dot = 0.0;
            for (int d = 0; d < kDC; ++d)
                dot += static_cast<double>(qn[d]) * tr.nope_fp8[d];
            for (int d = 0; d < kDR; ++d)
                dot += static_cast<double>(qr[d]) * tr.rope_pre[d];
            scores[t] = dot * rig.q_scales[h] * tr.k_scale * sm_scale;
            m = std::max(m, scores[t]);
        }
        double l = 0.0;
        for (int t = 0; t < N; ++t) {
            scores[t] = std::exp(scores[t] - m);
            l += scores[t];
        }
        lse[h] = static_cast<float>(m + std::log(l));
        float* orow = out.data() + static_cast<size_t>(h) * kDC;
        for (int t = 0; t < N; ++t) {
            const auto& tr = rig.rows[t];
            const double p = scores[t] / l;
            for (int d = 0; d < kDC; ++d)
                orow[d] += static_cast<float>(p * tr.nope_fp8[d] * tr.k_scale);
        }
    }
}

void fill_metadata(DecodeRig& rig, int num_sm_parts, bool sparse_topk) {
    GetMlaMetadataParams mp{};
    mp.seqlens_k_ptr = rig.d_seqlens;
    mp.tile_scheduler_metadata_ptr = rig.d_sched_meta;
    mp.num_splits_ptr = rig.d_num_splits;
    mp.batch_size = 1;
    mp.block_size_n = 64;
    mp.fixed_overhead_num_blocks = 1;
    mp.num_sm_parts = num_sm_parts;
    mp.topk = sparse_topk ? rig.num_tokens : -1;
    lc::launch_get_mla_metadata(mp, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

int read_num_splits(DecodeRig& rig) {
    int ns[2] = {0, 0};
    cudaMemcpy(ns, rig.d_num_splits, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(ns[0], 0);
    return ns[1];
}

void fill_sparse_params(
    sm120::decode::sparse_fp8::SparseAttnDecodeParams& p,
    DecodeRig& rig, float sm_scale, int num_sm_parts) {
    std::memset(&p, 0, sizeof(p));
    p.b = 1; p.s_q = 1; p.h_q = kHQ; p.h_kv = 1;
    p.d_qk = kDQK; p.d_v = kDC;
    p.sm_scale = sm_scale;
    p.sm_scale_div_log2 = sm_scale / std::log(2.0f);
    p.num_blocks = 0;
    p.page_block_size = kPage;
    p.topk = rig.num_tokens;
    p.model_type = sm120::sparse::ModelType::V32;
    p.q = reinterpret_cast<cutlass::bfloat16_t*>(rig.d_q_nope);
    p.q_rope = reinterpret_cast<cutlass::bfloat16_t*>(rig.d_q_rope);
    p.q_scales = rig.d_q_scales;
    p.kv = static_cast<cutlass::bfloat16_t*>(rig.d_cache);
    p.indices = rig.d_indices;
    p.topk_length = nullptr;
    p.attn_sink = nullptr;
    p.lse = rig.d_lse;
    p.out = reinterpret_cast<cutlass::bfloat16_t*>(rig.d_out);
    p.stride_q_b = kHQ * kDC;
    p.stride_q_s_q = kHQ * kDC;
    p.stride_q_h_q = kDC;
    p.stride_kv_block = kPage * kRowBytes;
    p.stride_kv_row = kRowBytes;
    p.stride_indices_b = rig.num_tokens;
    p.stride_indices_s_q = rig.num_tokens;
    p.stride_lse_b = kHQ;
    p.stride_lse_s_q = kHQ;
    p.stride_o_b = kHQ * kDC;
    p.stride_o_s_q = kHQ * kDC;
    p.stride_o_h_q = kDC;
    p.lse_accum = rig.d_lse_accum;
    p.o_accum = rig.d_o_accum;
    p.stride_lse_accum_split = kHQ;
    p.stride_lse_accum_s_q = kHQ;
    p.stride_o_accum_split = kHQ * kDC;
    p.stride_o_accum_s_q = kHQ * kDC;
    p.stride_o_accum_h_q = kDC;
    p.tile_scheduler_metadata_ptr =
        reinterpret_cast<sm120::decode::sparse_fp8::DecodingSchedMeta*>(
            rig.d_sched_meta);
    p.num_splits_ptr = rig.d_num_splits;
    p.num_sm_parts = num_sm_parts;
    p.stream = nullptr;
}

void fill_dense_params(
    sm120::decode::dense_fp8::DenseAttnDecodeParams& p,
    DecodeRig& rig, float sm_scale, int num_sm_parts) {
    std::memset(&p, 0, sizeof(p));
    p.b = 1; p.s_q = 1; p.h_q = kHQ; p.h_kv = 1;
    p.d_qk = kDQK; p.d_v = kDC; p.d_nope = kDC;
    p.sm_scale = sm_scale;
    p.q_nope = reinterpret_cast<cutlass::bfloat16_t*>(rig.d_q_nope);
    p.q_rope = reinterpret_cast<cutlass::bfloat16_t*>(rig.d_q_rope);
    p.q_scales = rig.d_q_scales;
    p.kv_cache = static_cast<cutlass::bfloat16_t*>(rig.d_cache);
    p.stride_kv_block = kPage * kRowBytes;
    p.stride_kv_row = kRowBytes;
    p.block_table = rig.d_block_table;
    p.block_table_batch_stride = rig.num_pages;
    p.page_block_size = kPage;
    p.seqlens_k = rig.d_seqlens;
    p.out = reinterpret_cast<cutlass::bfloat16_t*>(rig.d_out);
    p.lse = rig.d_lse;
    p.stride_o_b = kHQ * kDC;
    p.stride_o_s_q = kHQ * kDC;
    p.stride_o_h_q = kDC;
    p.stride_lse_b = kHQ;
    p.stride_lse_s_q = kHQ;
    p.lse_accum = rig.d_lse_accum;
    p.o_accum = rig.d_o_accum;
    p.stride_lse_accum_split = kHQ;
    p.stride_lse_accum_s_q = kHQ;
    p.stride_o_accum_split = kHQ * kDC;
    p.stride_o_accum_s_q = kHQ * kDC;
    p.stride_o_accum_h_q = kDC;
    p.tile_scheduler_metadata_ptr =
        reinterpret_cast<sm120::decode::sparse_fp8::DecodingSchedMeta*>(
            rig.d_sched_meta);
    p.num_splits_ptr = rig.d_num_splits;
    p.num_sm_parts = num_sm_parts;
    p.deterministic_reduce = false;
    p.stream = nullptr;
}

void run_combine(DecodeRig& rig, int num_sm_parts) {
    MlaCombineParams c{};
    std::memset(&c, 0, sizeof(c));
    c.b = 1;
    c.h_q = kHQ;
    c.h_k = 1;
    c.q_seq_per_hk = kHQ;
    c.d_v = kDC;
    c.o_ptr = rig.d_out;
    c.softmax_lse_ptr = rig.d_lse;
    c.o_batch_stride = kHQ * kDC;
    c.o_head_stride = kDC;
    c.o_row_stride = kDC;
    c.num_splits_ptr = rig.d_num_splits;
    c.num_sm_parts = num_sm_parts;
    c.softmax_lseaccum_ptr = rig.d_lse_accum;
    c.oaccum_ptr = rig.d_o_accum;
    lc::launch_mla_combine(c, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

void check_result(DecodeRig& rig, const std::vector<float>& ref_out,
                  const std::vector<float>& ref_lse, const char* tag) {
    std::vector<__nv_bfloat16> outb(static_cast<size_t>(kHQ) * kDC);
    std::vector<float> lse(kHQ);
    ASSERT_EQ(cudaMemcpy(outb.data(), rig.d_out, outb.size() * 2,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(lse.data(), rig.d_lse, kHQ * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    // LSE: NATURAL units. The denominator sL is full-precision fp32 in-kernel;
    // scores match the CPU reference bit-for-bit modulo MMA fp32 accumulation
    // order → tight. A ×ln2 / +log(P_scale) regression is orders larger.
    for (int h = 0; h < kHQ; ++h)
        EXPECT_NEAR(lse[h], ref_lse[h], 5e-2f) << tag << " lse[" << h << "]";

    // Output: FP8 P quantization + per-tile V dequant/requant noise → loose
    // per-element tolerance. Still far tighter than the old lse_accum split
    // mis-weighting (order-of-magnitude with the planted token).
    for (size_t i = 0; i < outb.size(); ++i) {
        const double got = __bfloat162float(outb[i]);
        const double want = ref_out[i];
        ASSERT_NEAR(got, want, 0.10 + 0.10 * std::abs(want))
            << tag << " out[" << i << "]";
    }
}

}  // namespace

// ── sparse_fp8 decode ────────────────────────────────────────────────────────

TEST(DecodeLseUnits, SparseNoSplitLseNatural) {
    REQUIRES_GPU();
    DecodeRig rig;
    build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
              /*plant_token=*/-1, /*seed=*/23);
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    std::vector<float> ref_out, ref_lse;
    cpu_reference(rig, sm_scale, ref_out, ref_lse);

    fill_metadata(rig, /*num_sm_parts=*/1, /*sparse_topk=*/true);
    ASSERT_EQ(read_num_splits(rig), 1) << "expected the no-split path";

    sm120::decode::sparse_fp8::SparseAttnDecodeParams p{};
    fill_sparse_params(p, rig, sm_scale, /*num_sm_parts=*/1);
    lc::launch_decode_sparse_fp8(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    check_result(rig, ref_out, ref_lse, "sparse-nosplit");
}

TEST(DecodeLseUnits, SparseMultiSplitCombineMatchesReference) {
    REQUIRES_GPU();
    DecodeRig rig;
    // Planted high-logit token deep in the sequence: one split's max
    // dominates — the discriminator for the old mixed-unit lse_accum
    // (mla_combine weights ∝ 2^lse_accum).
    build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
              /*plant_token=*/2040, /*seed=*/29);
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    std::vector<float> ref_out, ref_lse;
    cpu_reference(rig, sm_scale, ref_out, ref_lse);

    fill_metadata(rig, /*num_sm_parts=*/8, /*sparse_topk=*/true);
    const int nsplits = read_num_splits(rig);
    ASSERT_GT(nsplits, 1) << "test requires a genuine multi-split schedule";

    sm120::decode::sparse_fp8::SparseAttnDecodeParams p{};
    fill_sparse_params(p, rig, sm_scale, /*num_sm_parts=*/8);
    lc::launch_decode_sparse_fp8(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    run_combine(rig, /*num_sm_parts=*/8);
    check_result(rig, ref_out, ref_lse, "sparse-split");
}

// ── dense_fp8 decode ─────────────────────────────────────────────────────────

TEST(DecodeLseUnits, DenseNoSplitLseNatural) {
    REQUIRES_GPU();
    DecodeRig rig;
    build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
              /*plant_token=*/-1, /*seed=*/31);
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    std::vector<float> ref_out, ref_lse;
    cpu_reference(rig, sm_scale, ref_out, ref_lse);

    fill_metadata(rig, /*num_sm_parts=*/1, /*sparse_topk=*/false);
    ASSERT_EQ(read_num_splits(rig), 1) << "expected the no-split path";

    sm120::decode::dense_fp8::DenseAttnDecodeParams p{};
    fill_dense_params(p, rig, sm_scale, /*num_sm_parts=*/1);
    lc::launch_decode_dense_fp8(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    check_result(rig, ref_out, ref_lse, "dense-nosplit");
}

TEST(DecodeLseUnits, DenseMultiSplitCombineMatchesReference) {
    REQUIRES_GPU();
    DecodeRig rig;
    build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
              /*plant_token=*/2040, /*seed=*/37);
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    std::vector<float> ref_out, ref_lse;
    cpu_reference(rig, sm_scale, ref_out, ref_lse);

    fill_metadata(rig, /*num_sm_parts=*/8, /*sparse_topk=*/false);
    const int nsplits = read_num_splits(rig);
    ASSERT_GT(nsplits, 1) << "test requires a genuine multi-split schedule";

    sm120::decode::dense_fp8::DenseAttnDecodeParams p{};
    fill_dense_params(p, rig, sm_scale, /*num_sm_parts=*/8);
    lc::launch_decode_dense_fp8(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    run_combine(rig, /*num_sm_parts=*/8);
    check_result(rig, ref_out, ref_lse, "dense-split");
}
