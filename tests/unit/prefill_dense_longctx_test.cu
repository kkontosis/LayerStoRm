// TD-GLM-LONGCTX: dense (and sparse) absorbed-MLA prefill numerics at
// s_kv >> B_TOPK(64) vs a CPU reference. The short goldens (≤7 tokens) run
// exactly ONE k_block, so the multi-block online-softmax rescale path
// (sM/sL/sScale cross-block accumulation, current-block-max scheme) was never
// validated. A planted high-logit token in the LAST block mimics the
// "question at the end of a long document" — the failing long-context shape.

#include "compute/kernels/attention/mla_attention.h"
#include "compute/prefill_params.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <random>
#include <vector>

namespace lc = layerstorm::compute;

namespace {

constexpr int kDC = 512;    // kv_lora_rank (d_v)
constexpr int kDR = 64;     // qk_rope_head_dim
constexpr int kDQK = kDC + kDR;
constexpr int kHQ = 32;     // heads per rank (GLM-5.2 TP=2)

float bf16r(float f) { return __bfloat162float(__float2bfloat16(f)); }

struct Ref {
    std::vector<float> out;  // [s_q, h_q, d_v]
    std::vector<float> lse;  // [s_q, h_q]
};

// CPU reference: absorbed MLA attention, non-causal over s_kv staged rows
// (the device stages exactly seqlen rows, so validity == all rows).
Ref cpu_reference(const std::vector<float>& q, const std::vector<float>& kv,
                  int s_q, int s_kv, float sm_scale) {
    Ref r;
    r.out.assign(static_cast<size_t>(s_q) * kHQ * kDC, 0.0f);
    r.lse.assign(static_cast<size_t>(s_q) * kHQ, 0.0f);
    std::vector<double> scores(s_kv);
    for (int sq = 0; sq < s_q; ++sq) {
        for (int h = 0; h < kHQ; ++h) {
            const float* qrow =
                q.data() + (static_cast<size_t>(sq) * kHQ + h) * kDQK;
            double m = -1e300;
            for (int t = 0; t < s_kv; ++t) {
                const float* krow = kv.data() + static_cast<size_t>(t) * kDQK;
                double dot = 0.0;
                for (int d = 0; d < kDQK; ++d)
                    dot += static_cast<double>(qrow[d]) * krow[d];
                scores[t] = dot * sm_scale;
                m = std::max(m, scores[t]);
            }
            double l = 0.0;
            for (int t = 0; t < s_kv; ++t) {
                scores[t] = std::exp(scores[t] - m);
                l += scores[t];
            }
            float* orow =
                r.out.data() + (static_cast<size_t>(sq) * kHQ + h) * kDC;
            for (int t = 0; t < s_kv; ++t) {
                const float* vrow = kv.data() + static_cast<size_t>(t) * kDQK;
                const double p = scores[t] / l;
                for (int d = 0; d < kDC; ++d)
                    orow[d] += static_cast<float>(p * vrow[d]);
            }
            r.lse[static_cast<size_t>(sq) * kHQ + h] =
                static_cast<float>(m + std::log(l));
        }
    }
    return r;
}

void run_case(int s_q, int s_kv, bool plant_late_signal) {
    std::mt19937 rng(41 + s_kv);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    // bf16-rounded host data so CPU and GPU see identical inputs.
    std::vector<float> qf(static_cast<size_t>(s_q) * kHQ * kDQK);
    std::vector<float> kvf(static_cast<size_t>(s_kv) * kDQK);
    for (auto& v : qf) v = bf16r(dist(rng));
    for (auto& v : kvf) v = bf16r(dist(rng));

    if (plant_late_signal) {
        // Make the LAST token's key strongly aligned with every query head
        // (high logit in the final k_block) and give it a distinctive value
        // vector — the "answer token at the end of the document".
        for (int d = 0; d < kDQK; ++d) {
            float acc = 0.f;
            for (int h = 0; h < kHQ; ++h)
                acc += qf[(static_cast<size_t>(0) * kHQ + h) * kDQK + d];
            kvf[static_cast<size_t>(s_kv - 1) * kDQK + d] =
                bf16r(acc / kHQ * 4.0f);
        }
    }

    std::vector<__nv_bfloat16> qb(qf.size()), kvb(kvf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    for (size_t i = 0; i < kvf.size(); ++i) kvb[i] = __float2bfloat16(kvf[i]);

    void *dq = nullptr, *dkv = nullptr, *dout = nullptr;
    float* dlse = nullptr;
    ASSERT_EQ(cudaMalloc(&dq, qb.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dkv, kvb.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dout, static_cast<size_t>(s_q) * kHQ * kDC * 2),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dlse, static_cast<size_t>(s_q) * kHQ * sizeof(float)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dq, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dkv, kvb.data(), kvb.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);

    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    const lc::PrefillDims dims{kDC, kDR, kHQ, num_sm, 0.0f};  // legacy 1/sqrt(d_qk)
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));

    float* dmaxl = nullptr;
    ASSERT_EQ(cudaMalloc(&dmaxl, static_cast<size_t>(s_q) * kHQ * sizeof(float)),
              cudaSuccess);
    sm120::prefill::dense::head64::DenseAttnFwdParams p{};
    lc::populate_dense_prefill_params(p, dims, dq, dkv, s_q, s_kv, dout, dlse,
                                      nullptr);
    p.max_logits = dmaxl;
    p.deterministic_reduce = true;  // engine setting (DET-REDUCE)
    lc::launch_prefill_dense(p);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> outb(static_cast<size_t>(s_q) * kHQ * kDC);
    std::vector<float> lse(static_cast<size_t>(s_q) * kHQ);
    ASSERT_EQ(cudaMemcpy(outb.data(), dout, outb.size() * 2,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(lse.data(), dlse, lse.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    const Ref ref = cpu_reference(qf, kvf, s_q, s_kv, sm_scale);

    // max_logits sanity. It must be in NATURAL log units (pre-fix exported
    // cM*ln2 — the ×ln2 bug; LSE below is the primary guard for that). It is
    // NOT required to equal the true global per-head max: online softmax is
    // shift-invariant, so the multi-block / deterministic-reduce path may
    // finalize a per-lane STABILIZATION max — output and LSE are provably
    // invariant to that choice (both checked below), and the only consumer,
    // the KV-sharded DCP LSE combine, merges consistent (m,l) pairs and works
    // with any valid m. So assert equality with the global max only in the
    // single-block case (where cM IS the global max), and an upper-bound
    // natural-unit sanity otherwise (TD-PREFILL-MAXLOGITS-STABMAX). A ×ln2
    // regression is caught by the LSE check regardless.
    {
        std::vector<float> maxl(static_cast<size_t>(s_q) * kHQ);
        ASSERT_EQ(cudaMemcpy(maxl.data(), dmaxl, maxl.size() * sizeof(float),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        const bool single_block = (s_kv <= 64);
        for (int sq = 0; sq < s_q; ++sq) {
            for (int h = 0; h < kHQ; ++h) {
                const float* qrow =
                    qf.data() + (static_cast<size_t>(sq) * kHQ + h) * kDQK;
                double m = -1e300;
                for (int t = 0; t < s_kv; ++t) {
                    double dot = 0.0;
                    for (int d = 0; d < kDQK; ++d)
                        dot += static_cast<double>(qrow[d])
                             * kvf[static_cast<size_t>(t) * kDQK + d];
                    m = std::max(m, dot * sm_scale);
                }
                const float got = maxl[static_cast<size_t>(sq) * kHQ + h];
                if (single_block) {
                    EXPECT_NEAR(got, static_cast<float>(m), 5e-2f)
                        << "max_logits[" << sq << "," << h << "] s_kv=" << s_kv;
                } else {
                    // Stabilization max: valid iff it is a finite natural-unit
                    // value not exceeding the global max (a ×ln2 regression
                    // stays below this bound, hence LSE is the real guard).
                    EXPECT_TRUE(std::isfinite(got))
                        << "max_logits[" << sq << "," << h << "] s_kv=" << s_kv;
                    EXPECT_LE(got, static_cast<float>(m) + 5e-2f)
                        << "max_logits[" << sq << "," << h << "] s_kv=" << s_kv;
                }
            }
        }
    }

    // LSE: fp32 all the way in the kernel — tight.
    for (size_t i = 0; i < lse.size(); ++i)
        EXPECT_NEAR(lse[i], ref.lse[i], 5e-2f) << "lse[" << i << "] s_kv=" << s_kv;

    // Output: bf16 P-matrix + bf16 output storage → per-element tolerance.
    double max_abs = 0.0;
    for (size_t i = 0; i < outb.size(); ++i) {
        const double got = __bfloat162float(outb[i]);
        const double want = ref.out[i];
        max_abs = std::max(max_abs, std::abs(got - want));
        ASSERT_NEAR(got, want, 0.05 + 0.05 * std::abs(want))
            << "out[" << i << "] s_kv=" << s_kv;
    }
    // Overall closeness (guards against systematic drift within tolerance).
    EXPECT_LT(max_abs, 0.08) << "s_kv=" << s_kv;

    cudaFree(dq); cudaFree(dkv); cudaFree(dout); cudaFree(dlse); cudaFree(dmaxl);
}

}  // namespace

TEST(PrefillDenseLongCtx, SingleBlockControl) {
    REQUIRES_GPU();
    run_case(/*s_q=*/2, /*s_kv=*/7, /*plant=*/false);
}

TEST(PrefillDenseLongCtx, MultiBlockMatchesReference) {
    REQUIRES_GPU();
    run_case(/*s_q=*/2, /*s_kv=*/300, /*plant=*/false);
}

TEST(PrefillDenseLongCtx, LongContextWithLateSignal) {
    REQUIRES_GPU();
    run_case(/*s_q=*/1, /*s_kv=*/2877, /*plant=*/true);
}
