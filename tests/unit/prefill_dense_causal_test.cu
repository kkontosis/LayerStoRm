// TD-PREFILL-CHUNK-ATTN (perf follow-up): batched CAUSAL dense prefill via
// DenseAttnFwdParams::s_kv_per_row. Locks the causal-mask contract:
//
//   1. BIT-EQUALITY: one batched causal call over B query rows with per-row
//      bounds {len_0..len_{B-1}} over a shared union KV prefix [0, s_kv)
//      produces byte-identical out/lse/max_logits to B per-row batch-of-1
//      calls (the retired per-row-loop oracle) with s_kv = len_b each. This
//      is the exact replacement contract for the dcp_executor per-row loop.
//   2. NUMERICS: batched causal matches a CPU causal reference (row b
//      softmaxes over rows [0, len_b) only) within bf16 tolerance — proves
//      the mask actually restricts attention, including partial-B_TOPK-block
//      boundaries and multi-block online-softmax rescale.
//   3. nullptr s_kv_per_row keeps the legacy flat behavior (row attends all
//      s_kv rows) — guarded by comparing against a flat batched call.

#include "compute/kernels/attention/mla_attention.h"
#include "compute/prefill_params.h"

#include <sm120/prep/dequant_ckv_indexed.h>

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
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

// CPU reference: absorbed MLA attention where query row sq attends staged
// rows [0, bounds[sq]) only (causal-chunk semantics).
Ref cpu_causal_reference(const std::vector<float>& q,
                         const std::vector<float>& kv,
                         const std::vector<int>& bounds,
                         int s_q, float sm_scale) {
    Ref r;
    r.out.assign(static_cast<size_t>(s_q) * kHQ * kDC, 0.0f);
    r.lse.assign(static_cast<size_t>(s_q) * kHQ, 0.0f);
    for (int sq = 0; sq < s_q; ++sq) {
        const int len = bounds[sq];
        std::vector<double> scores(len);
        for (int h = 0; h < kHQ; ++h) {
            const float* qrow =
                q.data() + (static_cast<size_t>(sq) * kHQ + h) * kDQK;
            double m = -1e300;
            for (int t = 0; t < len; ++t) {
                const float* krow = kv.data() + static_cast<size_t>(t) * kDQK;
                double dot = 0.0;
                for (int d = 0; d < kDQK; ++d)
                    dot += static_cast<double>(qrow[d]) * krow[d];
                scores[t] = dot * sm_scale;
                m = std::max(m, scores[t]);
            }
            double l = 0.0;
            for (int t = 0; t < len; ++t) {
                scores[t] = std::exp(scores[t] - m);
                l += scores[t];
            }
            float* orow =
                r.out.data() + (static_cast<size_t>(sq) * kHQ + h) * kDC;
            for (int t = 0; t < len; ++t) {
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

struct DeviceRun {
    std::vector<uint16_t> out;   // raw bf16 bits [s_q, h_q, d_v]
    std::vector<float> lse;      // [s_q, h_q]
    std::vector<float> maxl;     // [s_q, h_q]
};

// Run the dense prefill kernel. bounds == nullptr → flat (non-causal).
// per_row == true → the retired per-row-loop oracle: B batch-of-1 calls,
// row b with s_kv = bounds[b], q/out/lse/maxl offset by row.
DeviceRun run_device(const std::vector<__nv_bfloat16>& qb,
                     const std::vector<__nv_bfloat16>& kvb,
                     const std::vector<int>& bounds,
                     int s_q, int s_kv, bool causal, bool per_row,
                     bool deterministic) {
    void *dq = nullptr, *dkv = nullptr, *dout = nullptr;
    float *dlse = nullptr, *dmaxl = nullptr;
    int* dbounds = nullptr;
    const size_t out_elems = static_cast<size_t>(s_q) * kHQ * kDC;
    const size_t sh = static_cast<size_t>(s_q) * kHQ;
    EXPECT_EQ(cudaMalloc(&dq, qb.size() * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dkv, kvb.size() * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dout, out_elems * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dlse, sh * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dmaxl, sh * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dbounds, s_q * sizeof(int)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dq, qb.data(), qb.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dkv, kvb.data(), kvb.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dbounds, bounds.data(), s_q * sizeof(int),
                         cudaMemcpyHostToDevice), cudaSuccess);
    // Poison outputs so "kernel wrote nothing" cannot pass bit-equality.
    EXPECT_EQ(cudaMemset(dout, 0xA5, out_elems * 2), cudaSuccess);
    EXPECT_EQ(cudaMemset(dlse, 0xA5, sh * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMemset(dmaxl, 0xA5, sh * sizeof(float)), cudaSuccess);

    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    const lc::PrefillDims dims{kDC, kDR, kHQ, num_sm, 0.0f};

    if (per_row) {
        for (int b = 0; b < s_q; ++b) {
            sm120::prefill::dense::head64::DenseAttnFwdParams p{};
            lc::populate_dense_prefill_params(
                p, dims,
                static_cast<const __nv_bfloat16*>(dq)
                    + static_cast<size_t>(b) * kHQ * kDQK,
                dkv, /*batch_size=*/1, /*seq_len_kv=*/bounds[b],
                static_cast<__nv_bfloat16*>(dout)
                    + static_cast<size_t>(b) * kHQ * kDC,
                dlse + static_cast<size_t>(b) * kHQ, nullptr);
            p.max_logits = dmaxl + static_cast<size_t>(b) * kHQ;
            p.deterministic_reduce = deterministic;
            lc::launch_prefill_dense(p);
        }
    } else {
        sm120::prefill::dense::head64::DenseAttnFwdParams p{};
        lc::populate_dense_prefill_params(p, dims, dq, dkv, s_q, s_kv,
                                          dout, dlse, nullptr);
        p.max_logits = dmaxl;
        p.deterministic_reduce = deterministic;
        if (causal) p.s_kv_per_row = dbounds;
        lc::launch_prefill_dense(p);
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    DeviceRun r;
    r.out.resize(out_elems);
    r.lse.resize(sh);
    r.maxl.resize(sh);
    EXPECT_EQ(cudaMemcpy(r.out.data(), dout, out_elems * 2,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.lse.data(), dlse, sh * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.maxl.data(), dmaxl, sh * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dq); cudaFree(dkv); cudaFree(dout);
    cudaFree(dlse); cudaFree(dmaxl); cudaFree(dbounds);
    return r;
}

// bit_equal: assert byte-identity between the batched-causal call and the
// per-row oracle. Requires deterministic_reduce — the legacy cross-warp
// atomicAdd denominator is not bit-reproducible across launches (DET-REDUCE),
// so the non-deterministic path is held to CPU-reference tolerance only.
void run_case(int s_q, int s_kv, const std::vector<int>& bounds,
              bool deterministic, bool bit_equal = true) {
    ASSERT_EQ(static_cast<int>(bounds.size()), s_q);
    std::mt19937 rng(97 + s_kv + s_q);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> qf(static_cast<size_t>(s_q) * kHQ * kDQK);
    std::vector<float> kvf(static_cast<size_t>(s_kv) * kDQK);
    for (auto& v : qf) v = bf16r(dist(rng));
    for (auto& v : kvf) v = bf16r(dist(rng));

    std::vector<__nv_bfloat16> qb(qf.size()), kvb(kvf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    for (size_t i = 0; i < kvf.size(); ++i) kvb[i] = __float2bfloat16(kvf[i]);

    const auto batched = run_device(qb, kvb, bounds, s_q, s_kv,
                                    /*causal=*/true, /*per_row=*/false,
                                    deterministic);

    // 1. Bit-equality vs the per-row-loop oracle (deterministic reduce only).
    if (bit_equal) {
        const auto oracle = run_device(qb, kvb, bounds, s_q, s_kv,
                                       /*causal=*/true, /*per_row=*/true,
                                       deterministic);
        ASSERT_EQ(batched.out.size(), oracle.out.size());
        EXPECT_EQ(std::memcmp(batched.out.data(), oracle.out.data(),
                              batched.out.size() * 2), 0)
            << "batched-causal out != per-row out (bitwise), s_kv=" << s_kv;
        EXPECT_EQ(std::memcmp(batched.lse.data(), oracle.lse.data(),
                              batched.lse.size() * sizeof(float)), 0)
            << "batched-causal lse != per-row lse (bitwise), s_kv=" << s_kv;
        EXPECT_EQ(std::memcmp(batched.maxl.data(), oracle.maxl.data(),
                              batched.maxl.size() * sizeof(float)), 0)
            << "batched-causal max_logits != per-row (bitwise), s_kv=" << s_kv;
    }

    // 2. CPU causal reference (bf16 tolerances as prefill_dense_longctx_test).
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    const Ref ref = cpu_causal_reference(qf, kvf, bounds, s_q, sm_scale);
    for (size_t i = 0; i < batched.lse.size(); ++i)
        EXPECT_NEAR(batched.lse[i], ref.lse[i], 5e-2f)
            << "lse[" << i << "] s_kv=" << s_kv;
    for (size_t i = 0; i < batched.out.size(); ++i) {
        __nv_bfloat16 b;
        std::memcpy(&b, &batched.out[i], 2);
        const double got = __bfloat162float(b);
        const double want = ref.out[i];
        ASSERT_NEAR(got, want, 0.05 + 0.05 * std::abs(want))
            << "out[" << i << "] s_kv=" << s_kv;
    }

    // 3. nullptr s_kv_per_row == legacy flat mask: when every bound == s_kv,
    //    causal and flat batched calls must be bit-identical.
    if (bit_equal) {
        const std::vector<int> full(s_q, s_kv);
        const auto causal_full = run_device(qb, kvb, full, s_q, s_kv,
                                            /*causal=*/true, /*per_row=*/false,
                                            deterministic);
        const auto flat = run_device(qb, kvb, full, s_q, s_kv,
                                     /*causal=*/false, /*per_row=*/false,
                                     deterministic);
        EXPECT_EQ(std::memcmp(causal_full.out.data(), flat.out.data(),
                              flat.out.size() * 2), 0)
            << "full-bound causal != flat (bitwise), s_kv=" << s_kv;
        EXPECT_EQ(std::memcmp(causal_full.lse.data(), flat.lse.data(),
                              flat.lse.size() * sizeof(float)), 0)
            << "full-bound causal lse != flat (bitwise), s_kv=" << s_kv;
    }
}

}  // namespace

TEST(PrefillDenseCausal, ChunkSingleBlock) {
    REQUIRES_GPU();
    // The golden GLM52_PREFILL_CHUNK=4 shape: 4 consecutive positions of a
    // fresh prompt, bounds {1,2,3,4} (row b attends [0, b+1)).
    run_case(/*s_q=*/4, /*s_kv=*/4, {1, 2, 3, 4}, /*deterministic=*/true);
}

TEST(PrefillDenseCausal, BlockBoundaryPartialBlocks) {
    REQUIRES_GPU();
    // Bounds straddling the B_TOPK=64 block edge: exercises the partial-last-
    // block mask on both sides of the boundary within one batch.
    run_case(/*s_q=*/4, /*s_kv=*/65, {63, 64, 65, 1}, /*deterministic=*/true);
}

TEST(PrefillDenseCausal, MultiBlockLongPrefixes) {
    REQUIRES_GPU();
    // Multi-k_block online-softmax rescale under per-row bounds; a mid-chunk
    // 512-token-style shape (chunk rows deep into a longer context).
    run_case(/*s_q=*/4, /*s_kv=*/300, {150, 233, 299, 300},
             /*deterministic=*/true);
}

// KVS-3 / INV-KVS-EMPTY: an empty DCP shard reaches this kernel as a row
// bound of 0 (chunk path) or a whole call with s_kv == 0 (decode/B==1 path).
// The empty convention is FINITE zero output + lse = +inf (dcp_lse_correct
// maps +inf/NaN to combine weight 0) — NaN output would poison the DCP
// output-allreduce on the other rank. The kernel is safe by construction
// because NEGATIVE_INFINITY / MAX_INIT_VAL are the FINITE sentinel -1e30
// (an all-masked block keeps rescale = 1 and sL = 0, never 0 × exp2f(inf));
// these tests LOCK that convention against sentinel/rescale refactors.
namespace {
void check_empty_rows(const DeviceRun& r, const std::vector<int>& bounds,
                      int s_q, const char* tag) {
    for (int sq = 0; sq < s_q; ++sq) {
        if (bounds[sq] != 0) continue;
        for (int h = 0; h < kHQ; ++h) {
            const float lse = r.lse[static_cast<size_t>(sq) * kHQ + h];
            EXPECT_TRUE(std::isinf(lse) && lse > 0.0f)
                << tag << ": empty row " << sq << " head " << h
                << " lse=" << lse << " (want +inf)";
            for (int d = 0; d < kDC; ++d) {
                __nv_bfloat16 b;
                std::memcpy(&b,
                            &r.out[(static_cast<size_t>(sq) * kHQ + h) * kDC
                                   + d], 2);
                const float v = __bfloat162float(b);
                EXPECT_EQ(v, 0.0f)
                    << tag << ": empty row " << sq << " head " << h
                    << " out[" << d << "]=" << v << " (want 0, NaN would "
                    << "poison the DCP output-allreduce)";
                if (std::isnan(v)) return;  // one NaN says it all
            }
        }
    }
}
}  // namespace
TEST(PrefillDenseCausal, EmptyShardRowsZeroOutputInfLse) {
    REQUIRES_GPU();
    // Mixed batch: rows 0/2 empty (bound 0), row 1 real — the sharded chunk
    // shape where this rank owns none of some rows' tokens.
    const int s_q = 3, s_kv = 5;
    const std::vector<int> bounds = {0, 3, 0};
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    std::vector<float> qf(static_cast<size_t>(s_q) * kHQ * kDQK);
    std::vector<float> kvf(static_cast<size_t>(s_kv) * kDQK);
    for (auto& v : qf) v = bf16r(dist(rng));
    for (auto& v : kvf) v = bf16r(dist(rng));
    std::vector<__nv_bfloat16> qb(qf.size()), kvb(kvf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    for (size_t i = 0; i < kvf.size(); ++i) kvb[i] = __float2bfloat16(kvf[i]);

    for (bool det : {true, false}) {
        const auto r = run_device(qb, kvb, bounds, s_q, s_kv,
                                  /*causal=*/true, /*per_row=*/false, det);
        check_empty_rows(r, bounds, s_q, det ? "det" : "nondet");
        // The non-empty row must still match the CPU causal reference (the
        // empty-block guard must not perturb real rows).
        const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
        const Ref ref = cpu_causal_reference(qf, kvf, bounds, s_q, sm_scale);
        for (int h = 0; h < kHQ; ++h) {
            EXPECT_NEAR(r.lse[static_cast<size_t>(1) * kHQ + h],
                        ref.lse[static_cast<size_t>(1) * kHQ + h], 5e-2f);
            for (int d = 0; d < kDC; ++d) {
                __nv_bfloat16 b;
                std::memcpy(&b,
                            &r.out[(static_cast<size_t>(1) * kHQ + h) * kDC
                                   + d], 2);
                const double got = __bfloat162float(b);
                const double want =
                    ref.out[(static_cast<size_t>(1) * kHQ + h) * kDC + d];
                ASSERT_NEAR(got, want, 0.05 + 0.05 * std::abs(want));
            }
        }
    }
}

TEST(PrefillDenseCausal, EmptyShardWholeCallSkvZero) {
    REQUIRES_GPU();
    // Whole-call s_kv == 0 (nongraph decode / B==1 prefill on a rank whose
    // local shard is empty; the executor passes the rank-local max = 0 and
    // the device skips linearize/dequant). Flat mask (no s_kv_per_row).
    const int s_q = 2, s_kv = 0;
    const std::vector<int> bounds = {0, 0};
    std::mt19937 rng(77);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    std::vector<float> qf(static_cast<size_t>(s_q) * kHQ * kDQK);
    for (auto& v : qf) v = bf16r(dist(rng));
    std::vector<__nv_bfloat16> qb(qf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    // One dummy KV row so cudaMalloc gets a nonzero size; never read (all
    // mask bits false at s_kv=0).
    std::vector<__nv_bfloat16> kvb(kDQK, __float2bfloat16(0.0f));

    for (bool det : {true, false}) {
        const auto r = run_device(qb, kvb, bounds, s_q, s_kv,
                                  /*causal=*/false, /*per_row=*/false, det);
        check_empty_rows(r, bounds, s_q, det ? "det-skv0" : "nondet-skv0");
    }
}

TEST(PrefillDenseCausal, NonDeterministicReducePath) {
    REQUIRES_GPU();
    // Same mask contract on the legacy atomicAdd denominator path — CPU
    // tolerance only (that path is not bit-reproducible across launches).
    run_case(/*s_q=*/3, /*s_kv=*/130, {7, 64, 130}, /*deterministic=*/false,
             /*bit_equal=*/false);
}

// Perf comparison (not a correctness gate — run explicitly with
// --gtest_also_run_disabled_tests): the retired per-row loop
// (B × [dequant prefix + attend]) vs ONE batched causal call
// ([dequant union once] + attend). Staging traffic drops from
// O(B·s_kv) to O(s_kv); prints measured wall times.
TEST(PrefillDenseCausal, DISABLED_PerRowVsBatchedTiming) {
    REQUIRES_GPU();
    constexpr int B = 128;         // chunk rows
    constexpr int S_KV = 8192;     // context length (mid-context chunk)
    constexpr int PAGE = 64;
    constexpr int ROW_BYTES = kDC + 4 + kDR * 2;  // FP8 nope + scale + BF16 rope
    const int num_pages = S_KV / PAGE;

    std::vector<int> bounds(B);
    for (int b = 0; b < B; ++b) bounds[b] = S_KV - B + b + 1;

    // Device buffers (contents irrelevant for timing; zeroed cache).
    void *dcache = nullptr, *dstage = nullptr, *dq = nullptr, *dout = nullptr;
    float* dlse = nullptr;
    int *didx = nullptr, *dbounds = nullptr;
    ASSERT_EQ(cudaMalloc(&dcache,
                         static_cast<size_t>(num_pages) * PAGE * ROW_BYTES),
              cudaSuccess);
    ASSERT_EQ(cudaMemset(dcache, 0,
                         static_cast<size_t>(num_pages) * PAGE * ROW_BYTES),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dstage, static_cast<size_t>(S_KV) * kDQK * 2),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dq, static_cast<size_t>(B) * kHQ * kDQK * 2),
              cudaSuccess);
    ASSERT_EQ(cudaMemset(dq, 0, static_cast<size_t>(B) * kHQ * kDQK * 2),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dout, static_cast<size_t>(B) * kHQ * kDC * 2),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dlse,
                         static_cast<size_t>(B) * kHQ * sizeof(float)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&didx, S_KV * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dbounds, B * sizeof(int)), cudaSuccess);
    std::vector<int> idx(S_KV);
    std::iota(idx.begin(), idx.end(), 0);
    ASSERT_EQ(cudaMemcpy(didx, idx.data(), S_KV * sizeof(int),
                         cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dbounds, bounds.data(), B * sizeof(int),
                         cudaMemcpyHostToDevice), cudaSuccess);

    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    const lc::PrefillDims dims{kDC, kDR, kHQ, num_sm, 0.0f};

    auto make_dq = [&](int num_fetch) {
        sm120::prep::DequantCKVIndexedParams d{};
        d.kv_cache = static_cast<const __nv_fp8_e4m3*>(dcache);
        d.cache_stride_block = static_cast<int64_t>(PAGE) * ROW_BYTES;
        d.cache_stride_row = ROW_BYTES;
        d.page_size = PAGE;
        d.indices = didx;
        d.num_fetch = num_fetch;
        d.k_out = static_cast<__nv_bfloat16*>(dstage);
        d.d_c = kDC;
        d.d_rope = kDR;
        return d;
    };

    auto per_row = [&] {
        for (int b = 0; b < B; ++b) {
            auto d = make_dq(bounds[b]);
            lc::launch_dequant_ckv_indexed(d, nullptr);
            sm120::prefill::dense::head64::DenseAttnFwdParams p{};
            lc::populate_dense_prefill_params(
                p, dims,
                static_cast<const __nv_bfloat16*>(dq)
                    + static_cast<size_t>(b) * kHQ * kDQK,
                dstage, 1, bounds[b],
                static_cast<__nv_bfloat16*>(dout)
                    + static_cast<size_t>(b) * kHQ * kDC,
                dlse + static_cast<size_t>(b) * kHQ, nullptr);
            p.deterministic_reduce = true;
            lc::launch_prefill_dense(p);
        }
    };
    auto batched = [&] {
        auto d = make_dq(S_KV);
        lc::launch_dequant_ckv_indexed(d, nullptr);
        sm120::prefill::dense::head64::DenseAttnFwdParams p{};
        lc::populate_dense_prefill_params(p, dims, dq, dstage, B, S_KV,
                                          dout, dlse, nullptr);
        p.deterministic_reduce = true;
        p.s_kv_per_row = dbounds;
        lc::launch_prefill_dense(p);
    };

    auto time_ms = [&](auto&& fn) {
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        for (int i = 0; i < 3; ++i) fn();          // warmup
        cudaDeviceSynchronize();
        constexpr int ITERS = 10;
        cudaEventRecord(t0);
        for (int i = 0; i < ITERS; ++i) fn();
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        return ms / ITERS;
    };

    const float ms_per_row = time_ms(per_row);
    const float ms_batched = time_ms(batched);
    size_t per_row_rows = 0;
    for (int b = 0; b < B; ++b) per_row_rows += bounds[b];
    fprintf(stderr,
            "[causal-perf] B=%d s_kv=%d: per-row loop %.3f ms "
            "(staged %zu rows = %.1f MB), batched causal %.3f ms "
            "(staged %d rows = %.1f MB), speedup %.1fx\n",
            B, S_KV, ms_per_row, per_row_rows,
            per_row_rows * kDQK * 2 / 1048576.0, ms_batched, S_KV,
            static_cast<size_t>(S_KV) * kDQK * 2 / 1048576.0,
            ms_per_row / ms_batched);
    EXPECT_LT(ms_batched, ms_per_row);

    cudaFree(dcache); cudaFree(dstage); cudaFree(dq); cudaFree(dout);
    cudaFree(dlse); cudaFree(didx); cudaFree(dbounds);
}
