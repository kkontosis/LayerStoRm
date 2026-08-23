// TD-SPARSE-CHUNK-PREFILL: chunk-causal SPARSE prefill via
// SparseAttnFwdParams::s_kv_per_row (INV-SPARSE-CHUNK-CAUSAL). Locks the
// combined top-k × causal-bound mask contract:
//
//   1. PER-ROW EQUIVALENCE: one batched chunk-causal sparse call over B
//      query rows with per-row bounds {len_0..len_{B-1}} over a shared
//      staged KV [0, s_kv) equals B per-row batch-of-1 sparse calls with
//      s_kv = len_b each (the legacy in-bound masking `index < s_kv` IS the
//      per-row causal mask at batch-of-1) — the exact replacement contract
//      that lets a prefill chunk consume per-row top-k. Identical mask and
//      block sequence. Under deterministic_reduce=true the equivalence is
//      BIT-EXACT (memcmp); the legacy atomicAdd denominator path
//      (deterministic_reduce=false) is held to ~1-ULP tolerances
//      (warp-scheduling launch-order jitter).
//   2. NUMERICS: the batched chunk-causal sparse call matches a CPU sparse
//      reference — row b softmaxes over EXACTLY {indices[b][j] : j <
//      topk_len[b], 0 <= idx < len_b} — within bf16 tolerance. Selected
//      indices AT or PAST the row's own bound are deliberately included in
//      the index rows and must be masked, never attended (the causal half of
//      the combined mask); -1 padding and topk_length truncation must hold
//      (the top-k half).
//   3. nullptr s_kv_per_row keeps the legacy flat behavior: bounds == s_kv
//      everywhere must equal a flat batched call (same tolerance as #1).
//   4. Fully masked rows (topk_length 0, or every selected index >= the
//      row's bound) keep the empty convention: zero output + lse = +inf
//      (finite sentinels — NaN would poison the DCP output-allreduce).
//   5. DET-REDUCE (TD-SPARSE-PREFILL-DETREDUCE, INV-DRIFT-DETREDUCE): with
//      SparseAttnFwdParams::deterministic_reduce=true, two runs on the SAME
//      input are BIT-IDENTICAL (memcmp on out + lse — no tolerance), and the
//      batched-causal call is BIT-EQUAL to the per-row oracle and to the flat
//      call at full bounds — matching the dense twin's DET-REDUCE bar. The
//      deterministic order is mathematically the same reduction (fixed-order
//      cross-warp combine of the softmax denominator), so the CPU-reference
//      numerics checks hold identically in both modes.
//
// Production dims: d_qk=576 (kv_lora 512 + rope 64), h_q=32 (GLM-5.2 TP=2).

#include "compute/kernels/attention/mla_attention.h"
#include "compute/prefill_params.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
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
    std::vector<float> lse;  // [s_q, h_q]  (+inf for empty rows)
};

// CPU reference: absorbed MLA sparse attention where query row sq attends
// exactly {indices[sq][j] : j < topk_len[sq], 0 <= indices[sq][j] <
// bounds[sq]} (chunk-causal sparse semantics — the top-k mask combined with
// the row's causal bound).
Ref cpu_sparse_causal_reference(const std::vector<float>& q,
                                const std::vector<float>& kv,
                                const std::vector<int>& indices,
                                const std::vector<int>& topk_len,
                                const std::vector<int>& bounds,
                                int s_q, int topk, float sm_scale) {
    Ref r;
    r.out.assign(static_cast<size_t>(s_q) * kHQ * kDC, 0.0f);
    r.lse.assign(static_cast<size_t>(s_q) * kHQ, 0.0f);
    for (int sq = 0; sq < s_q; ++sq) {
        std::vector<int> sel;
        for (int j = 0; j < topk_len[sq]; ++j) {
            const int t = indices[static_cast<size_t>(sq) * topk + j];
            if (t >= 0 && t < bounds[sq]) sel.push_back(t);
        }
        for (int h = 0; h < kHQ; ++h) {
            const float* qrow =
                q.data() + (static_cast<size_t>(sq) * kHQ + h) * kDQK;
            if (sel.empty()) {
                r.lse[static_cast<size_t>(sq) * kHQ + h] = INFINITY;
                continue;  // out stays 0
            }
            std::vector<double> scores(sel.size());
            double m = -1e300;
            for (size_t i = 0; i < sel.size(); ++i) {
                const float* krow =
                    kv.data() + static_cast<size_t>(sel[i]) * kDQK;
                double dot = 0.0;
                for (int d = 0; d < kDQK; ++d)
                    dot += static_cast<double>(qrow[d]) * krow[d];
                scores[i] = dot * sm_scale;
                m = std::max(m, scores[i]);
            }
            double l = 0.0;
            for (auto& s : scores) { s = std::exp(s - m); l += s; }
            float* orow =
                r.out.data() + (static_cast<size_t>(sq) * kHQ + h) * kDC;
            for (size_t i = 0; i < sel.size(); ++i) {
                const float* vrow =
                    kv.data() + static_cast<size_t>(sel[i]) * kDQK;
                const double p = scores[i] / l;
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
};

// Run the sparse prefill kernel. mode:
//   kBatchedCausal — one call, s_kv_per_row = bounds (chunk-causal sparse).
//   kBatchedFlat   — one call, s_kv_per_row = nullptr (legacy flat).
//   kPerRow        — B batch-of-1 calls, row b with s_kv = bounds[b] (the
//                    legacy in-bound mask IS the causal mask at batch-of-1).
enum class Mode { kBatchedCausal, kBatchedFlat, kPerRow };

DeviceRun run_device(const std::vector<__nv_bfloat16>& qb,
                     const std::vector<__nv_bfloat16>& kvb,
                     const std::vector<int>& indices,
                     const std::vector<int>& topk_len,
                     const std::vector<int>& bounds,
                     int s_q, int s_kv, int topk, Mode mode,
                     bool deterministic = false) {
    void *dq = nullptr, *dkv = nullptr, *dout = nullptr;
    float* dlse = nullptr;
    int *dind = nullptr, *dtkl = nullptr, *dbounds = nullptr;
    const size_t out_elems = static_cast<size_t>(s_q) * kHQ * kDC;
    const size_t sh = static_cast<size_t>(s_q) * kHQ;
    EXPECT_EQ(cudaMalloc(&dq, qb.size() * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dkv, kvb.size() * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dout, out_elems * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dlse, sh * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dind, indices.size() * sizeof(int)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dtkl, s_q * sizeof(int)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dbounds, s_q * sizeof(int)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dq, qb.data(), qb.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dkv, kvb.data(), kvb.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dind, indices.data(), indices.size() * sizeof(int),
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dtkl, topk_len.data(), s_q * sizeof(int),
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dbounds, bounds.data(), s_q * sizeof(int),
                         cudaMemcpyHostToDevice), cudaSuccess);
    // Poison outputs so "kernel wrote nothing" cannot pass bit-equality.
    EXPECT_EQ(cudaMemset(dout, 0xA5, out_elems * 2), cudaSuccess);
    EXPECT_EQ(cudaMemset(dlse, 0xA5, sh * sizeof(float)), cudaSuccess);

    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    const lc::PrefillDims dims{kDC, kDR, kHQ, num_sm, 0.0f};

    if (mode == Mode::kPerRow) {
        for (int b = 0; b < s_q; ++b) {
            SparseAttnFwdParams p{};
            lc::populate_sparse_prefill_params(
                p, dims,
                static_cast<const __nv_bfloat16*>(dq)
                    + static_cast<size_t>(b) * kHQ * kDQK,
                dkv,
                static_cast<const int*>(dind) + static_cast<size_t>(b) * topk,
                static_cast<const int*>(dtkl) + b, topk,
                /*batch_size=*/1, /*seq_len_kv=*/bounds[b],
                static_cast<__nv_bfloat16*>(dout)
                    + static_cast<size_t>(b) * kHQ * kDC,
                dlse + static_cast<size_t>(b) * kHQ, nullptr);
            p.deterministic_reduce = deterministic;
            lc::launch_prefill_sparse(p);
        }
    } else {
        SparseAttnFwdParams p{};
        lc::populate_sparse_prefill_params(p, dims, dq, dkv, dind, dtkl, topk,
                                           s_q, s_kv, dout, dlse, nullptr);
        p.deterministic_reduce = deterministic;
        if (mode == Mode::kBatchedCausal) p.s_kv_per_row = dbounds;
        lc::launch_prefill_sparse(p);
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    DeviceRun r;
    r.out.resize(out_elems);
    r.lse.resize(sh);
    EXPECT_EQ(cudaMemcpy(r.out.data(), dout, out_elems * 2,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.lse.data(), dlse, sh * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dq); cudaFree(dkv); cudaFree(dout);
    cudaFree(dlse); cudaFree(dind); cudaFree(dtkl); cudaFree(dbounds);
    return r;
}

// Build a row's index list: `n_causal` unique ascending indices < bound plus
// `n_future` unique ascending indices in [bound, s_kv) (which the kernel
// must mask), then -1 padding. topk_len counts the filled slots (causal +
// future) — exactly the shape a producer WITHOUT causal top-k would emit, so
// passing means the kernel's own bound does the masking.
void fill_row(std::vector<int>& indices, int row, int topk, int bound,
              int s_kv, int n_causal, int n_future, std::mt19937& rng,
              int* topk_len_out) {
    std::vector<int> pool_c(bound), pool_f(s_kv - bound);
    for (int i = 0; i < bound; ++i) pool_c[i] = i;
    for (int i = 0; i < s_kv - bound; ++i) pool_f[i] = bound + i;
    std::shuffle(pool_c.begin(), pool_c.end(), rng);
    std::shuffle(pool_f.begin(), pool_f.end(), rng);
    std::vector<int> sel;
    for (int i = 0; i < n_causal && i < static_cast<int>(pool_c.size()); ++i)
        sel.push_back(pool_c[i]);
    for (int i = 0; i < n_future && i < static_cast<int>(pool_f.size()); ++i)
        sel.push_back(pool_f[i]);
    std::sort(sel.begin(), sel.end());   // real top-k emits ascending
    *topk_len_out = static_cast<int>(sel.size());
    for (int j = 0; j < topk; ++j)
        indices[static_cast<size_t>(row) * topk + j] =
            j < static_cast<int>(sel.size()) ? sel[j] : -1;
}

// Bit-exact run compare (deterministic_reduce path): out bf16 bits and lse
// f32 bits must be BYTE-IDENTICAL — no tolerance (DET-REDUCE bar, matching
// the dense twin's prefill_dense_causal_test).
void compare_runs_bitexact(const DeviceRun& a, const DeviceRun& b,
                           const char* tag) {
    ASSERT_EQ(a.out.size(), b.out.size());
    ASSERT_EQ(a.lse.size(), b.lse.size());
    EXPECT_EQ(std::memcmp(a.out.data(), b.out.data(), a.out.size() * 2), 0)
        << tag << " out differs (bitwise)";
    EXPECT_EQ(std::memcmp(a.lse.data(), b.lse.data(),
                          a.lse.size() * sizeof(float)), 0)
        << tag << " lse differs (bitwise)";
}

// Near-identical run compare (legacy atomicAdd path): same mask + block
// sequence ⇒ only cross-warp atomicAdd denominator jitter may differ
// (~1 ULP in fp32; ≤1 step in bf16).
void compare_runs(const DeviceRun& a, const DeviceRun& b, const char* tag) {
    for (size_t i = 0; i < a.lse.size(); ++i) {
        if (std::isinf(a.lse[i]) || std::isinf(b.lse[i])) {
            EXPECT_EQ(a.lse[i], b.lse[i]) << tag << " lse[" << i << "]";
        } else {
            EXPECT_NEAR(a.lse[i], b.lse[i],
                        1e-5f * (1.0f + std::abs(b.lse[i])))
                << tag << " lse[" << i << "]";
        }
    }
    for (size_t i = 0; i < a.out.size(); ++i) {
        __nv_bfloat16 ba, bb;
        std::memcpy(&ba, &a.out[i], 2);
        std::memcpy(&bb, &b.out[i], 2);
        const float fa = __bfloat162float(ba), fb = __bfloat162float(bb);
        ASSERT_FALSE(std::isnan(fa) || std::isnan(fb))
            << tag << " NaN out[" << i << "]";
        ASSERT_NEAR(fa, fb, 1e-2f * (1.0f + std::abs(fb)))
            << tag << " out[" << i << "]";
    }
}

// deterministic == false: legacy atomicAdd denominator — cross-run
// comparisons use ~1-ULP tolerances (compare_runs).
// deterministic == true (DET-REDUCE): every cross-run comparison is BIT-EXACT
// (memcmp), PLUS a run-to-run repeat of the identical batched call must be
// byte-identical (the TD-SPARSE-PREFILL-DETREDUCE acceptance check).
void run_case(int s_q, int s_kv, int topk,
              const std::vector<int>& bounds,
              const std::vector<int>& n_causal,
              const std::vector<int>& n_future,
              uint32_t seed, bool deterministic = false) {
    ASSERT_EQ(static_cast<int>(bounds.size()), s_q);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> qf(static_cast<size_t>(s_q) * kHQ * kDQK);
    std::vector<float> kvf(static_cast<size_t>(s_kv) * kDQK);
    for (auto& v : qf) v = bf16r(dist(rng));
    for (auto& v : kvf) v = bf16r(dist(rng));
    std::vector<__nv_bfloat16> qb(qf.size()), kvb(kvf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    for (size_t i = 0; i < kvf.size(); ++i) kvb[i] = __float2bfloat16(kvf[i]);

    std::vector<int> indices(static_cast<size_t>(s_q) * topk, -1);
    std::vector<int> topk_len(s_q, 0);
    for (int b = 0; b < s_q; ++b)
        fill_row(indices, b, topk, bounds[b], s_kv, n_causal[b], n_future[b],
                 rng, &topk_len[b]);

    const auto batched = run_device(qb, kvb, indices, topk_len, bounds,
                                    s_q, s_kv, topk, Mode::kBatchedCausal,
                                    deterministic);

    // 0. DET-REDUCE run-to-run bit-reproducibility: the IDENTICAL batched
    //    call repeated must be byte-identical (out bf16 bits + lse f32 bits,
    //    memcmp — no tolerance). This is the exact property the legacy
    //    cross-warp atomicAdd denominator lacks.
    if (deterministic) {
        const auto rerun = run_device(qb, kvb, indices, topk_len, bounds,
                                      s_q, s_kv, topk, Mode::kBatchedCausal,
                                      /*deterministic=*/true);
        compare_runs_bitexact(batched, rerun, "run-to-run (deterministic)");
    }

    // 1. Equivalence vs the per-row batch-of-1 oracle: identical mask and
    //    block sequence. deterministic → BIT-EXACT (fixed-order denominator
    //    combine); legacy atomicAdd → launch-order jitter only, held to
    //    ~1-ULP tolerances.
    const auto oracle = run_device(qb, kvb, indices, topk_len, bounds,
                                   s_q, s_kv, topk, Mode::kPerRow,
                                   deterministic);
    ASSERT_EQ(batched.out.size(), oracle.out.size());
    if (deterministic)
        compare_runs_bitexact(batched, oracle, "chunk-causal vs per-row oracle");
    else
        compare_runs(batched, oracle, "chunk-causal vs per-row oracle");

    // 2. CPU sparse-causal reference (bf16 tolerances as the dense twin).
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    const Ref ref = cpu_sparse_causal_reference(
        qf, kvf, indices, topk_len, bounds, s_q, topk, sm_scale);
    for (size_t i = 0; i < batched.lse.size(); ++i) {
        if (std::isinf(ref.lse[i])) {
            EXPECT_TRUE(std::isinf(batched.lse[i]) && batched.lse[i] > 0.0f)
                << "empty row lse[" << i << "]=" << batched.lse[i]
                << " (want +inf)";
        } else {
            EXPECT_NEAR(batched.lse[i], ref.lse[i], 5e-2f)
                << "lse[" << i << "] s_kv=" << s_kv;
        }
    }
    for (size_t i = 0; i < batched.out.size(); ++i) {
        __nv_bfloat16 b;
        std::memcpy(&b, &batched.out[i], 2);
        const double got = __bfloat162float(b);
        const double want = ref.out[i];
        ASSERT_FALSE(std::isnan(got)) << "NaN out[" << i << "]";
        ASSERT_NEAR(got, want, 0.05 + 0.05 * std::abs(want))
            << "out[" << i << "] s_kv=" << s_kv;
    }

    // 3. nullptr s_kv_per_row == legacy flat mask: when every bound == s_kv,
    //    causal and flat batched calls must be equivalent (bit-exact when
    //    deterministic; same jitter-only tolerance as #1 otherwise).
    {
        const std::vector<int> full(s_q, s_kv);
        const auto causal_full = run_device(qb, kvb, indices, topk_len, full,
                                            s_q, s_kv, topk,
                                            Mode::kBatchedCausal,
                                            deterministic);
        const auto flat = run_device(qb, kvb, indices, topk_len, full,
                                     s_q, s_kv, topk, Mode::kBatchedFlat,
                                     deterministic);
        if (deterministic)
            compare_runs_bitexact(causal_full, flat, "full-bound causal vs flat");
        else
            compare_runs(causal_full, flat, "full-bound causal vs flat");
    }
}

}  // namespace

TEST(PrefillSparseCausal, ChunkShapeMasksFutureSelections) {
    REQUIRES_GPU();
    // 4 consecutive chunk rows (bounds ascending) with index rows that
    // deliberately include FUTURE positions (>= bound) — the kernel's causal
    // half must mask them; the CPU reference attends only the causal subset.
    run_case(/*s_q=*/4, /*s_kv=*/48, /*topk=*/16,
             /*bounds=*/{1, 2, 24, 48},
             /*n_causal=*/{1, 2, 10, 16},
             /*n_future=*/{4, 4, 6, 0}, /*seed=*/101);
}

TEST(PrefillSparseCausal, BlockBoundaryPartialTopkBlocks) {
    REQUIRES_GPU();
    // topk = 96 straddles the B_TOPK=64 gather-block edge; topk_lengths on
    // both sides of the boundary exercise the partial-last-block valid mask
    // together with the per-row bound.
    run_case(/*s_q=*/4, /*s_kv=*/300, /*topk=*/96,
             /*bounds=*/{63, 64, 150, 300},
             /*n_causal=*/{40, 64, 90, 96},
             /*n_future=*/{30, 20, 6, 0}, /*seed=*/202);
}

TEST(PrefillSparseCausal, EmptyRowsZeroOutputInfLse) {
    REQUIRES_GPU();
    // Row 0: topk_length 0 (no selection). Row 1: every selected index is at
    // or past its bound (all masked by the causal half). Row 2: real. The
    // empty convention (zero out + lse=+inf, finite sentinels — never NaN)
    // must hold for both empty flavors while the real row stays exact.
    run_case(/*s_q=*/3, /*s_kv=*/80, /*topk=*/16,
             /*bounds=*/{5, 8, 80},
             /*n_causal=*/{0, 0, 16},
             /*n_future=*/{0, 6, 0}, /*seed=*/303);
}

TEST(PrefillSparseCausal, MidContextChunkPruningShape) {
    REQUIRES_GPU();
    // The production shape sparse chunk prefill runs at: a chunk of rows deep
    // into a longer context where the top-k PRUNES (topk < bound) — every
    // selection is causal (real producer output), the bound still clips the
    // staging tail the row must not see.
    run_case(/*s_q=*/6, /*s_kv=*/512, /*topk=*/64,
             /*bounds=*/{507, 508, 509, 510, 511, 512},
             /*n_causal=*/{64, 64, 64, 64, 64, 64},
             /*n_future=*/{0, 0, 0, 0, 0, 0}, /*seed=*/404);
}

// ── DET-REDUCE (TD-SPARSE-PREFILL-DETREDUCE, INV-DRIFT-DETREDUCE) ──────────
// deterministic_reduce=true: same shapes as above, but EVERY cross-run
// comparison is memcmp bit-exact — run-to-run repeat, batched-causal vs
// per-row oracle, and full-bound causal vs flat — while the CPU-reference
// numerics checks inside run_case prove the fixed-order combine did not
// change the math. The legacy-path (tolerance) tests above stay as the
// deterministic=false coverage.

TEST(PrefillSparseDetReduce, ChunkShapeBitReproducible) {
    REQUIRES_GPU();
    // Scattered selections incl. FUTURE positions the causal half masks.
    run_case(/*s_q=*/4, /*s_kv=*/48, /*topk=*/16,
             /*bounds=*/{1, 2, 24, 48},
             /*n_causal=*/{1, 2, 10, 16},
             /*n_future=*/{4, 4, 6, 0}, /*seed=*/101, /*deterministic=*/true);
}

TEST(PrefillSparseDetReduce, BlockBoundaryPartialBlocksBitReproducible) {
    REQUIRES_GPU();
    // Partial B_TOPK=64 blocks on both sides of the gather-block edge —
    // partially masked blocks exercise the per-warp partial slots where some
    // consumer warps own no valid column of a row.
    run_case(/*s_q=*/4, /*s_kv=*/300, /*topk=*/96,
             /*bounds=*/{63, 64, 150, 300},
             /*n_causal=*/{40, 64, 90, 96},
             /*n_future=*/{30, 20, 6, 0}, /*seed=*/202, /*deterministic=*/true);
}


TEST(PrefillSparseDetReduce, EmptyRowsBitReproducible) {
    REQUIRES_GPU();
    // Empty rows (topk_length 0 / all-masked) under the deterministic
    // combine: the all-masked-block rescale=1 carry must stay intact and the
    // zero-out + lse=+inf convention must be bit-stable too.
    run_case(/*s_q=*/3, /*s_kv=*/80, /*topk=*/16,
             /*bounds=*/{5, 8, 80},
             /*n_causal=*/{0, 0, 16},
             /*n_future=*/{0, 6, 0}, /*seed=*/303, /*deterministic=*/true);
}

TEST(PrefillSparseDetReduce, MidContextChunkPruningBitReproducible) {
    REQUIRES_GPU();
    // The production sparse-chunk-prefill shape (multi-k_block online-softmax
    // rescale, topk pruning), bit-reproducible end to end.
    run_case(/*s_q=*/6, /*s_kv=*/512, /*topk=*/64,
             /*bounds=*/{507, 508, 509, 510, 511, 512},
             /*n_causal=*/{64, 64, 64, 64, 64, 64},
             /*n_future=*/{0, 0, 0, 0, 0, 0}, /*seed=*/404,
             /*deterministic=*/true);
}
