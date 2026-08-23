// TD-TQ-PREFILL-DETREDUCE-WIRING: the TQ attention DEVICE must thread the
// DET-REDUCE gate (config compute.deterministic_reduce, set at init via
// tq_device_set_deterministic_reduce) into the prefill kernel params —
// DenseAttnFwdParams / SparseAttnFwdParams .deterministic_reduce — so that
// under attention_backend=tq the prefill softmax denominator uses the
// fixed-order deterministic combine (INV-DRIFT-DETREDUCE), exactly like the
// SnapMLA twin. Kernel-side bit-identity is already locked by
// prefill_sparse_causal_test.cu (PrefillSparseDetReduce.*) on the SHARED
// kernels; the NEW coverage here is the TQ device + TQ composite launcher
// threading:
//
//   1. RUN-TO-RUN: with the setter ON, repeated identical
//      TqSm120AttentionDevice::prefill_attention calls (full production
//      pipeline: TQ k_append'ed paged cache → linearize → indexed dequant →
//      shared BF16 prefill) are BYTE-IDENTICAL (memcmp out bf16 + lse f32,
//      no tolerance) — the property the legacy cross-warp atomicAdd
//      denominator lacks.
//   2. THREADING: every device run is BIT-EQUAL to an independent direct
//      launch_prefill_{dense,sparse}_tq composite call with
//      params.deterministic_reduce=true set EXPLICITLY on identical inputs.
//      The deterministic kernel is a pure bit-stable function of its inputs,
//      so equality holds iff the device selected the deterministic
//      instantiation — an unthreaded device (flag stuck false) runs the
//      atomic kernel, whose bits diverge from the fixed-order combine
//      (VERIFIED negative control: with the setter forced false both the
//      sparse and dense cases fail the reference memcmp on the first run
//      at topk=256; see the kTopk comment for the shape requirement).
//
// Shapes: both branches use the production chunk-causal prefill shape
// (mid-context chunk rows, ascending per-row bounds); sparse additionally
// prunes via topk (multi-B_TOPK-block, the atomic-jitter-exposing shape).
// Production dims: d_c=512, d_rope=64 (d_qk=576), h_q=32 (GLM-5.2 TP=2).

#include "compute/tq_sm120_attention_device.h"
#include "compute/tq_init.h"
#include "compute/kernels/attention/tq_mla_attention.h"
#include "compute/kernels/sm120/attention/tq_prep_params.h"
#include "compute/prefill_params.h"
#include "core/attention_device.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace lc = layerstorm::compute;
namespace cfg = layerstorm::config;

#ifdef LAYERSTORM_SOURCE_DIR

namespace {

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        ASSERT_EQ(_err, cudaSuccess)                                          \
            << "CUDA error: " << cudaGetErrorString(_err);                    \
    } while (0)

constexpr int kDC   = 512;               // kv_lora_rank (codebook d)
constexpr int kDR   = 64;                // qk_rope_head_dim
constexpr int kDQK  = kDC + kDR;
constexpr int kHQ   = 32;                // heads per rank (GLM-5.2 TP=2)
constexpr int kPage = 64;
constexpr int kSKV  = 512;               // union prefix length (8 full pages)
constexpr int kSQ   = 6;                 // chunk rows
// topk MUST straddle multiple B_TOPK=64 gather blocks: verified negative
// control (setter forced false) — at topk=64 (single block) the atomic
// denominator happened to be order-stable AND bit-equal to the fixed-order
// combine (an unwired device passed); at topk=256 both the sparse and dense
// unwired devices FAIL the reference memcmp on the first run.
constexpr int kTopk = 256;
// TQ cache row: d_c/2 packed nibbles + 2 B fp16 norm + d_rope BF16 rope.
constexpr int kRowBytes  = kDC / 2 + 2 + kDR * 2;
constexpr int kPageBytes = kPage * kRowBytes;
constexpr int kNumPages  = kSKV / kPage;
// Repeated device runs: an unthreaded (atomic) device matching the
// deterministic reference bit pattern this many consecutive times is
// vanishingly unlikely (the verified negative control fails on run 1).
constexpr int kReruns = 4;

std::string codebook_dir() {
    return std::string(LAYERSTORM_SOURCE_DIR) + "/config/tq_codebooks";
}

struct RunOut {
    std::vector<uint16_t> out;  // raw bf16 bits [s_q, h_q, d_c]
    std::vector<float> lse;     // [s_q, h_q]
};

void read_back(const void* d_out, const float* d_lse, RunOut& r) {
    r.out.resize(static_cast<size_t>(kSQ) * kHQ * kDC);
    r.lse.resize(static_cast<size_t>(kSQ) * kHQ);
    CUDA_CHECK(cudaMemcpy(r.out.data(), d_out, r.out.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.lse.data(), d_lse, r.lse.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void expect_bitexact(const RunOut& a, const RunOut& b, const char* tag) {
    ASSERT_EQ(a.out.size(), b.out.size());
    ASSERT_EQ(a.lse.size(), b.lse.size());
    EXPECT_EQ(std::memcmp(a.out.data(), b.out.data(), a.out.size() * 2), 0)
        << tag << " out differs (bitwise)";
    EXPECT_EQ(std::memcmp(a.lse.data(), b.lse.data(),
                          a.lse.size() * sizeof(float)), 0)
        << tag << " lse differs (bitwise)";
}

// Full TQ-device DET-REDUCE case (sparse or dense). Builds a real TQ paged
// cache through the device's k_append, runs prefill_attention kReruns+1
// times with deterministic_reduce set via the R0H-1d free-function setter,
// and checks (1) run-to-run bit-identity and (2) bit-equality with a direct
// deterministic launch_prefill_{dense,sparse}_tq composite reference.
void run_case(bool is_sparse) {
    auto dev = lc::make_tq_sm120_attention_device(
        cfg::GpuRef{0, 0, cfg::GpuType::rtx5090});
    ASSERT_NE(dev, nullptr);
    dev->set_device();

    // TQ resources (codebook + per-layer Pi) through the real init path.
    lc::TqInitOptions opts;
    opts.d_c = kDC;
    opts.bits = 4;
    opts.num_layers = 1;
    opts.codebook_dir = codebook_dir();
    opts.attention_device = dev.get();
    auto res = lc::init_tq_resources(opts);
    ASSERT_NE(res, nullptr);

    lc::tq_device_set_resources(dev.get(), res.get());
    lc::tq_device_set_model_dims(dev.get(), /*batch_size=*/8, kDC, kDR, kHQ,
                                 /*s_q=*/1, /*sm_scale=*/0.0f);
    lc::tq_device_set_prefill_scratch(dev.get(), kSKV);
    // THE WIRING UNDER TEST (TD-TQ-PREFILL-DETREDUCE-WIRING): engine.cpp
    // calls this from compute.deterministic_reduce for the TQ backend.
    lc::tq_device_set_deterministic_reduce(dev.get(), true);

    std::mt19937 rng(20260708u + (is_sparse ? 1 : 0));
    std::normal_distribution<float> dist(0.0f, 0.5f);

    // ── Populate the TQ paged cache via the device's own k_append ──────────
    std::vector<__nv_bfloat16> c_kv(static_cast<size_t>(kSKV) * kDC);
    std::vector<__nv_bfloat16> k_rope(static_cast<size_t>(kSKV) * kDR);
    for (auto& v : c_kv) v = __float2bfloat16(dist(rng));
    for (auto& v : k_rope) v = __float2bfloat16(dist(rng));
    std::vector<int> slots(kSKV);
    std::iota(slots.begin(), slots.end(), 0);

    void *d_ckv = nullptr, *d_rope = nullptr, *d_cache = nullptr;
    int* d_slots = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ckv, c_kv.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_rope, k_rope.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_cache,
                          static_cast<size_t>(kNumPages) * kPageBytes));
    CUDA_CHECK(cudaMalloc(&d_slots, kSKV * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ckv, c_kv.data(), c_kv.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rope, k_rope.data(), k_rope.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, slots.data(), kSKV * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_cache, 0,
                          static_cast<size_t>(kNumPages) * kPageBytes));

    dev->k_append(d_ckv, d_rope, d_cache,
                  /*cache_stride_block=*/kPageBytes,
                  /*cache_stride_row=*/kRowBytes,
                  d_slots, kSKV, kDC, kDR,
                  /*c_kv_row_stride=*/kDC, /*k_rope_row_stride=*/kDR,
                  kPage, /*layer_idx=*/0, /*stream=*/nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Query, block tables, bounds, sparse selection ───────────────────────
    std::vector<__nv_bfloat16> q(static_cast<size_t>(kSQ) * kHQ * kDQK);
    for (auto& v : q) v = __float2bfloat16(dist(rng));

    // Identity paging: every row's block table is pages {0..7}.
    std::vector<int> bt(static_cast<size_t>(kSQ) * kNumPages);
    for (int b = 0; b < kSQ; ++b)
        for (int p = 0; p < kNumPages; ++p) bt[b * kNumPages + p] = p;

    // Both branches use the production chunk_causal shape (a chunk of rows
    // mid-context with ascending per-row bounds; the LAST row's prefix is the
    // staged union — the flat batch>1 shape is not a production call: the
    // device linearize scratch is sized max_kv, sum(seqlens_k) would
    // overflow it). Sparse additionally prunes via per-row ascending unique
    // selections < bound (topk < bound).
    std::vector<int> bounds(kSQ);
    for (int b = 0; b < kSQ; ++b)
        bounds[b] = kSKV - (kSQ - 1) + b;
    std::vector<int> indices(static_cast<size_t>(kSQ) * kTopk, -1);
    std::vector<int> topk_len(kSQ, kTopk);
    for (int b = 0; b < kSQ; ++b) {
        std::vector<int> pool(bounds[b]);
        std::iota(pool.begin(), pool.end(), 0);
        std::shuffle(pool.begin(), pool.end(), rng);
        std::sort(pool.begin(), pool.begin() + kTopk);
        std::copy(pool.begin(), pool.begin() + kTopk,
                  indices.begin() + static_cast<size_t>(b) * kTopk);
    }

    void* d_q = nullptr;
    int *d_bt = nullptr, *d_bounds = nullptr, *d_ind = nullptr,
        *d_tkl = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, q.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_bt, bt.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bounds, kSQ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ind, indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tkl, kSQ * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_q, q.data(), q.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt, bt.data(), bt.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bounds, bounds.data(), kSQ * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ind, indices.data(), indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tkl, topk_len.data(), kSQ * sizeof(int),
                          cudaMemcpyHostToDevice));

    const size_t out_elems = static_cast<size_t>(kSQ) * kHQ * kDC;
    const size_t lse_elems = static_cast<size_t>(kSQ) * kHQ;
    void* d_out = nullptr;
    float* d_lse = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, out_elems * 2));
    CUDA_CHECK(cudaMalloc(&d_lse, lse_elems * sizeof(float)));

    // ── Deterministic reference: direct TQ composite launcher with the flag
    //    set EXPLICITLY (own indexed dequant into own staging — bit-identical
    //    inputs to the device's internal pipeline). ─────────────────────────
    RunOut ref;
    {
        int* d_lin = nullptr;               // identity slot indices [0, kSKV)
        void* d_staging = nullptr;          // BF16 [kSKV, kDQK]
        CUDA_CHECK(cudaMalloc(&d_lin, kSKV * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_staging,
                              static_cast<size_t>(kSKV) * kDQK * 2));
        CUDA_CHECK(cudaMemcpy(d_lin, slots.data(), kSKV * sizeof(int),
                              cudaMemcpyHostToDevice));

        sm120::prep::TqDequantCKVIndexedParams dq{};
        dq.kv_cache = static_cast<const uint8_t*>(d_cache);
        dq.cache_stride_block = kPageBytes;
        dq.cache_stride_row = kRowBytes;
        dq.page_size = kPage;
        dq.Pi = res->device_Pi(0);
        dq.centroids = res->device_centroids();
        dq.d_c = kDC;
        dq.d_rope = kDR;
        dq.indices = d_lin;
        dq.num_fetch = kSKV;
        dq.k_out = static_cast<__nv_bfloat16*>(d_staging);

        int num_sm = 0;
        cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
        const lc::PrefillDims dims{kDC, kDR, kHQ, num_sm, 0.0f};

        CUDA_CHECK(cudaMemset(d_out, 0xA5, out_elems * 2));
        CUDA_CHECK(cudaMemset(d_lse, 0xA5, lse_elems * sizeof(float)));
        if (is_sparse) {
            SparseAttnFwdParams p{};
            lc::populate_sparse_prefill_params(p, dims, d_q, d_staging,
                d_ind, d_tkl, kTopk, kSQ, kSKV, d_out, d_lse, nullptr);
            p.deterministic_reduce = true;   // the explicit reference gate
            p.s_kv_per_row = d_bounds;       // chunk_causal
            lc::launch_prefill_sparse_tq(dq, p, nullptr);
        } else {
            sm120::prefill::dense::head64::DenseAttnFwdParams p{};
            lc::populate_dense_prefill_params(p, dims, d_q, d_staging,
                kSQ, kSKV, d_out, d_lse, nullptr);
            p.deterministic_reduce = true;   // the explicit reference gate
            p.s_kv_per_row = d_bounds;       // chunk_causal
            lc::launch_prefill_dense_tq(dq, p, nullptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        read_back(d_out, d_lse, ref);
        CUDA_CHECK(cudaFree(d_lin));
        CUDA_CHECK(cudaFree(d_staging));
    }

    // ── Device runs: kReruns + 1 identical prefill_attention calls ─────────
    RunOut first;
    for (int r = 0; r <= kReruns; ++r) {
        // Poison outputs so "kernel wrote nothing" cannot pass bit-equality.
        CUDA_CHECK(cudaMemset(d_out, 0xA5, out_elems * 2));
        CUDA_CHECK(cudaMemset(d_lse, 0xA5, lse_elems * sizeof(float)));
        dev->prefill_attention(
            d_q, kSQ, kSKV,
            /*seqlens_k=*/d_bounds, d_bt, /*max_blocks_per_seq=*/kNumPages,
            d_cache, kPageBytes, kRowBytes, kPage,
            is_sparse, /*chunk_causal=*/true,
            is_sparse ? d_ind : nullptr, is_sparse ? d_tkl : nullptr,
            is_sparse ? kTopk : 0,
            d_out, d_lse, /*layer_idx=*/0, /*stream=*/nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        RunOut got;
        read_back(d_out, d_lse, got);
        // 2. THREADING: bit-equal to the explicit deterministic reference.
        expect_bitexact(got, ref, is_sparse
            ? "TQ device sparse vs explicit deterministic reference"
            : "TQ device dense vs explicit deterministic reference");
        // 1. RUN-TO-RUN bit-reproducibility through the device.
        if (r == 0) first = std::move(got);
        else expect_bitexact(got, first, "TQ device run-to-run");
    }

    // No poisoned bytes may survive (every lse must have been WRITTEN —
    // otherwise ref==got would hold trivially, both being 0xA5-poisoned).
    for (float v : first.lse) {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        EXPECT_NE(bits, 0xA5A5A5A5u) << "unwritten (poisoned) lse";
    }

    CUDA_CHECK(cudaFree(d_ckv));
    CUDA_CHECK(cudaFree(d_rope));
    CUDA_CHECK(cudaFree(d_cache));
    CUDA_CHECK(cudaFree(d_slots));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_bt));
    CUDA_CHECK(cudaFree(d_bounds));
    CUDA_CHECK(cudaFree(d_ind));
    CUDA_CHECK(cudaFree(d_tkl));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_lse));

    lc::destroy_tq_resources(*res, *dev);
}

}  // namespace

TEST(TqPrefillDetReduce, SparseDeviceThreadsFlagBitReproducible) {
    REQUIRES_GPU();
    run_case(/*is_sparse=*/true);
}

TEST(TqPrefillDetReduce, DenseDeviceThreadsFlagBitReproducible) {
    REQUIRES_GPU();
    run_case(/*is_sparse=*/false);
}

#endif  // LAYERSTORM_SOURCE_DIR
