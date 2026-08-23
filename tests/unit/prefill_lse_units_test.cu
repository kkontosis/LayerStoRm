// KVS-1 / TD-LSE-UNITS-DECODE: LSE unit contract for the two prefill kernels
// that were never covered by an LSE-vs-CPU-reference test:
//
//   1. sparse fwd_for_small_topk (head64, D_QK=512) prefill mode — its
//      epilogue used FlashMLA's log2-unit idiom (lse = fmaf(m, ln2, log(l)),
//      max_logits = m*ln2) on NATURAL-unit online-softmax accumulators
//      (rS *= sm_scale, exp via exp2f(x*LOG2E) == e^x).
//   2. csa_fp8 prefill (D_QK=576) — same log2-unit idiom on natural units.
//
// Contract (INV-LSE-NAT): every kernel-exported external LSE is NATURAL log
// units: lse = m + log(sum exp(score - m)) with sm_scale applied to scores;
// max_logits (where exported) is the natural-unit stabilization max m.
// The DCP cross-rank LSE combine (dcp_lse_correct) consumes these values —
// mixed units reintroduce the INV-DCP-KVREP class of long-context corruption.
//
// Modeled on prefill_dense_longctx_test.cu (multi-k-block coverage at
// production dims, CPU double-precision reference).

#include "compute/kernels/attention/mla_attention.h"

#include "sm120/prefill/sparse/fwd_for_small_topk/head64/phase1.h"
#include "sm120/prefill/csa_fp8/phase1.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace {

float bf16r(float f) { return __bfloat162float(__float2bfloat16(f)); }

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess)                                           \
            << "CUDA error: " << cudaGetErrorString(_err);                     \
    } while (0)

// ── sparse fwd_for_small_topk prefill (D_QK=512, absorbed: V = KV[0:512]) ───

// CPU reference over gathered topk indices (same score definition as the
// kernel: dot(q, kv[idx]) * sm_scale; invalid idx (<0 or >= s_kv) masked).
struct Ref {
    std::vector<float> out;   // [s_q, h_q, d_v]
    std::vector<float> lse;   // [s_q, h_q] NATURAL units
    std::vector<float> maxl;  // [s_q, h_q] NATURAL units (true global max)
};

Ref small_topk_cpu_reference(const std::vector<float>& q,
                             const std::vector<float>& kv,
                             const std::vector<int>& indices,
                             int s_q, int s_kv, int h_q, int topk,
                             int d_qk, int d_v, float sm_scale) {
    Ref r;
    r.out.assign(static_cast<size_t>(s_q) * h_q * d_v, 0.0f);
    r.lse.assign(static_cast<size_t>(s_q) * h_q, 0.0f);
    r.maxl.assign(static_cast<size_t>(s_q) * h_q, 0.0f);
    std::vector<double> scores(topk);
    for (int sq = 0; sq < s_q; ++sq) {
        for (int h = 0; h < h_q; ++h) {
            const float* qrow =
                q.data() + (static_cast<size_t>(sq) * h_q + h) * d_qk;
            double m = -1e300;
            for (int t = 0; t < topk; ++t) {
                const int idx = indices[static_cast<size_t>(sq) * topk + t];
                if (idx < 0 || idx >= s_kv) { scores[t] = -1e300; continue; }
                const float* krow = kv.data() + static_cast<size_t>(idx) * d_qk;
                double dot = 0.0;
                for (int d = 0; d < d_qk; ++d)
                    dot += static_cast<double>(qrow[d]) * krow[d];
                scores[t] = dot * sm_scale;
                m = std::max(m, scores[t]);
            }
            double l = 0.0;
            for (int t = 0; t < topk; ++t) {
                if (scores[t] <= -1e299) { scores[t] = 0.0; continue; }
                scores[t] = std::exp(scores[t] - m);
                l += scores[t];
            }
            float* orow =
                r.out.data() + (static_cast<size_t>(sq) * h_q + h) * d_v;
            for (int t = 0; t < topk; ++t) {
                if (scores[t] == 0.0) continue;
                const int idx = indices[static_cast<size_t>(sq) * topk + t];
                const float* vrow = kv.data() + static_cast<size_t>(idx) * d_qk;
                const double p = scores[t] / l;
                for (int d = 0; d < d_v; ++d)
                    orow[d] += static_cast<float>(p * vrow[d]);
            }
            r.lse[static_cast<size_t>(sq) * h_q + h] =
                static_cast<float>(m + std::log(l));
            r.maxl[static_cast<size_t>(sq) * h_q + h] = static_cast<float>(m);
        }
    }
    return r;
}

void run_small_topk_case(int s_q, int s_kv, int topk, unsigned seed) {
    constexpr int kDQK = 512;   // this kernel's only instantiation
    constexpr int kDV = 512;
    constexpr int kHQ = 64;     // full B_H tile

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> qf(static_cast<size_t>(s_q) * kHQ * kDQK);
    std::vector<float> kvf(static_cast<size_t>(s_kv) * kDQK);
    for (auto& v : qf) v = bf16r(dist(rng));
    for (auto& v : kvf) v = bf16r(dist(rng));

    // Distinct random indices per query row; sprinkle invalid entries (-1)
    // to exercise the sValid masking (must not contribute to lse/max).
    std::vector<int> indices(static_cast<size_t>(s_q) * topk);
    std::vector<int> pool(s_kv);
    std::iota(pool.begin(), pool.end(), 0);
    for (int sq = 0; sq < s_q; ++sq) {
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int t = 0; t < topk; ++t)
            indices[static_cast<size_t>(sq) * topk + t] = pool[t];
        indices[static_cast<size_t>(sq) * topk + (topk / 3)] = -1;
        indices[static_cast<size_t>(sq) * topk + (2 * topk / 3)] = -1;
    }

    std::vector<__nv_bfloat16> qb(qf.size()), kvb(kvf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    for (size_t i = 0; i < kvf.size(); ++i) kvb[i] = __float2bfloat16(kvf[i]);

    void *dq = nullptr, *dkv = nullptr, *dout = nullptr;
    float *dlse = nullptr, *dmaxl = nullptr;
    int* dind = nullptr;
    CUDA_CHECK(cudaMalloc(&dq, qb.size() * 2));
    CUDA_CHECK(cudaMalloc(&dkv, kvb.size() * 2));
    CUDA_CHECK(cudaMalloc(&dout, static_cast<size_t>(s_q) * kHQ * kDV * 2));
    CUDA_CHECK(cudaMalloc(&dlse, static_cast<size_t>(s_q) * kHQ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dmaxl, static_cast<size_t>(s_q) * kHQ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dind, indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dq, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dkv, kvb.data(), kvb.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dind, indices.data(), indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));

    SparseAttnFwdParams p{};
    p.s_q = s_q; p.s_kv = s_kv; p.h_q = kHQ; p.h_kv = 1;
    p.d_qk = kDQK; p.d_v = kDV; p.topk = topk;
    p.sm_scale = sm_scale;
    p.sm_scale_div_log2 = sm_scale / std::log(2.0f);
    p.q = static_cast<cutlass::bfloat16_t*>(dq);
    p.kv = static_cast<cutlass::bfloat16_t*>(dkv);
    p.indices = dind;
    p.attn_sink = nullptr;
    p.topk_length = nullptr;
    p.stride_q_s_q = kHQ * kDQK;
    p.stride_q_h_q = kDQK;
    p.stride_kv_s_kv = kDQK;
    p.stride_kv_h_kv = kDQK;
    p.stride_indices_s_q = topk;
    p.stride_indices_h_kv = topk;
    p.out = static_cast<cutlass::bfloat16_t*>(dout);
    p.max_logits = dmaxl;
    p.lse = dlse;
    p.num_sm = 0;
    p.stream = nullptr;

    sm120::prefill::sparse::small_topk::head64::run_fwd_for_small_topk_phase1_kernel<
        SparseAttnFwdMode::Prefill, 512>(p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> outb(static_cast<size_t>(s_q) * kHQ * kDV);
    std::vector<float> lse(static_cast<size_t>(s_q) * kHQ);
    std::vector<float> maxl(static_cast<size_t>(s_q) * kHQ);
    CUDA_CHECK(cudaMemcpy(outb.data(), dout, outb.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lse.data(), dlse, lse.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(maxl.data(), dmaxl, maxl.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const Ref ref = small_topk_cpu_reference(qf, kvf, indices, s_q, s_kv, kHQ,
                                             topk, kDQK, kDV, sm_scale);

    // LSE: NATURAL units, fp32 in-kernel — tight. A ×ln2 regression shifts
    // this by (1-ln2)*m, far outside tolerance.
    for (size_t i = 0; i < lse.size(); ++i)
        EXPECT_NEAR(lse[i], ref.lse[i], 5e-2f)
            << "lse[" << i << "] topk=" << topk;

    // max_logits: NATURAL units, but under multi-block this kernel's
    // current-block-max scheme leaves sM at the LAST block's max — a valid
    // stabilization max (the (m, l) pair stays consistent; output and LSE
    // are shift-invariant, both asserted here), not the global max
    // (TD-PREFILL-MAXLOGITS-STABMAX; no consumer reads max_logits for
    // correctness — the DCP combine consumes lse only). Assert equality
    // single-block; natural-unit upper-bound sanity otherwise (a ×ln2
    // regression is caught by the LSE check regardless).
    const bool single_block = (topk <= 64);
    for (size_t i = 0; i < maxl.size(); ++i) {
        if (single_block) {
            EXPECT_NEAR(maxl[i], ref.maxl[i], 5e-2f)
                << "max_logits[" << i << "] topk=" << topk;
        } else {
            EXPECT_TRUE(std::isfinite(maxl[i]))
                << "max_logits[" << i << "] topk=" << topk;
            EXPECT_LE(maxl[i], ref.maxl[i] + 5e-2f)
                << "max_logits[" << i << "] topk=" << topk;
        }
    }

    // Output: bf16 P matrix + bf16 store.
    for (size_t i = 0; i < outb.size(); ++i) {
        const double got = __bfloat162float(outb[i]);
        const double want = ref.out[i];
        ASSERT_NEAR(got, want, 0.05 + 0.05 * std::abs(want))
            << "out[" << i << "] topk=" << topk;
    }

    cudaFree(dq); cudaFree(dkv); cudaFree(dout);
    cudaFree(dlse); cudaFree(dmaxl); cudaFree(dind);
}

// ── csa_fp8 prefill (D_QK=576, separate K/V, per-query causal) ──────────────

void run_csa_case(int s_q, int s_kv, int h_q,
                  const std::vector<int>& causal_seqlens, unsigned seed) {
    constexpr int kDQK = 576;
    constexpr int kDV = 512;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> qf(static_cast<size_t>(s_q) * h_q * kDQK);
    std::vector<float> kf(static_cast<size_t>(s_kv) * kDQK);
    std::vector<float> vf(static_cast<size_t>(s_kv) * kDV);
    for (auto& v : qf) v = bf16r(dist(rng));
    for (auto& v : kf) v = bf16r(dist(rng));
    for (auto& v : vf) v = bf16r(dist(rng));

    std::vector<__nv_bfloat16> qb(qf.size()), kb(kf.size()), vb(vf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    for (size_t i = 0; i < kf.size(); ++i) kb[i] = __float2bfloat16(kf[i]);
    for (size_t i = 0; i < vf.size(); ++i) vb[i] = __float2bfloat16(vf[i]);

    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    float* dlse = nullptr;
    int* dcs = nullptr;
    CUDA_CHECK(cudaMalloc(&dq, qb.size() * 2));
    CUDA_CHECK(cudaMalloc(&dk, kb.size() * 2));
    CUDA_CHECK(cudaMalloc(&dv, vb.size() * 2));
    CUDA_CHECK(cudaMalloc(&dout, static_cast<size_t>(s_q) * h_q * kDV * 2));
    CUDA_CHECK(cudaMalloc(&dlse, static_cast<size_t>(s_q) * h_q * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dq, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, kb.data(), kb.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, vb.data(), vb.size() * 2, cudaMemcpyHostToDevice));
    if (!causal_seqlens.empty()) {
        CUDA_CHECK(cudaMalloc(&dcs, s_q * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(dcs, causal_seqlens.data(), s_q * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));

    sm120::prefill::csa_fp8::CsaFp8PrefillParams p{};
    p.s_q = s_q; p.s_kv = s_kv; p.h_q = h_q; p.d_qk = kDQK; p.d_v = kDV;
    p.sm_scale = sm_scale;
    p.sm_scale_div_log2 = sm_scale / std::log(2.0f);
    p.q = static_cast<cutlass::bfloat16_t*>(dq);
    p.k = static_cast<cutlass::bfloat16_t*>(dk);
    p.v = static_cast<cutlass::bfloat16_t*>(dv);
    p.causal_seqlens = dcs;
    p.stride_q_s_q = h_q * kDQK;
    p.stride_q_h_q = kDQK;
    p.stride_k_s_kv = kDQK;
    p.stride_v_s_kv = kDV;
    p.out = static_cast<cutlass::bfloat16_t*>(dout);
    p.lse = dlse;
    p.num_sm = 0;
    p.stream = nullptr;

    sm120::prefill::csa_fp8::run_csa_fp8_prefill_kernel<576>(p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> outb(static_cast<size_t>(s_q) * h_q * kDV);
    std::vector<float> lse(static_cast<size_t>(s_q) * h_q);
    CUDA_CHECK(cudaMemcpy(outb.data(), dout, outb.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lse.data(), dlse, lse.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // CPU reference with the kernel's per-query causal bound.
    std::vector<double> scores(s_kv);
    for (int sq = 0; sq < s_q; ++sq) {
        const int bound = causal_seqlens.empty()
                              ? s_kv
                              : std::min(s_kv, causal_seqlens[sq]);
        for (int h = 0; h < h_q; ++h) {
            const float* qrow =
                qf.data() + (static_cast<size_t>(sq) * h_q + h) * kDQK;
            double m = -1e300;
            for (int t = 0; t < bound; ++t) {
                double dot = 0.0;
                for (int d = 0; d < kDQK; ++d)
                    dot += static_cast<double>(qrow[d]) *
                           kf[static_cast<size_t>(t) * kDQK + d];
                scores[t] = dot * sm_scale;
                m = std::max(m, scores[t]);
            }
            double l = 0.0;
            for (int t = 0; t < bound; ++t) {
                scores[t] = std::exp(scores[t] - m);
                l += scores[t];
            }
            // LSE: NATURAL units (m + log(l)).
            const float want_lse = static_cast<float>(m + std::log(l));
            const float got_lse = lse[static_cast<size_t>(sq) * h_q + h];
            EXPECT_NEAR(got_lse, want_lse, 5e-2f)
                << "lse[" << sq << "," << h << "] bound=" << bound;

            for (int d = 0; d < kDV; ++d) {
                double want = 0.0;
                for (int t = 0; t < bound; ++t)
                    want += scores[t] / l *
                            vf[static_cast<size_t>(t) * kDV + d];
                const double got = __bfloat162float(
                    outb[(static_cast<size_t>(sq) * h_q + h) * kDV + d]);
                ASSERT_NEAR(got, want, 0.05 + 0.05 * std::abs(want))
                    << "out[" << sq << "," << h << "," << d << "]";
            }
        }
    }

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dlse);
    if (dcs) cudaFree(dcs);
}

}  // namespace

TEST(PrefillLseUnits, SmallTopkSingleBlock) {
    REQUIRES_GPU();
    // One k-block (topk <= B_TOPK=64): epilogue exercised without rescale.
    run_small_topk_case(/*s_q=*/2, /*s_kv=*/128, /*topk=*/48, /*seed=*/7);
}

TEST(PrefillLseUnits, SmallTopkMultiBlock) {
    REQUIRES_GPU();
    // Multi-k-block online-softmax rescale path (160 = 2.5 * B_TOPK).
    run_small_topk_case(/*s_q=*/3, /*s_kv=*/500, /*topk=*/160, /*seed=*/11);
}

TEST(PrefillLseUnits, CsaFp8CausalMultiBlock) {
    REQUIRES_GPU();
    // h_q=96 exercises the head-group split (2 groups of B_H=64/32); causal
    // bounds cross k-block boundaries (B_TOPK=64 tile).
    run_csa_case(/*s_q=*/4, /*s_kv=*/300, /*h_q=*/96,
                 /*causal_seqlens=*/{1, 100, 129, 300}, /*seed=*/13);
}

TEST(PrefillLseUnits, CsaFp8NonCausal) {
    REQUIRES_GPU();
    run_csa_case(/*s_q=*/2, /*s_kv=*/200, /*h_q=*/64,
                 /*causal_seqlens=*/{}, /*seed=*/17);
}
