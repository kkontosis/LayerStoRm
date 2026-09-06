// Split-KV tq_sparse_decode (P-29 step 9, OQ-2) — the NEXT3 §1
// unit-level bar:
//
//   (a) split-vs-unsplit numeric bound: max abs + max ULP delta of out (and
//       lse) against the unsplit slim walk, reported, with a negative
//       control (a flipped attended KV byte MUST change the split output —
//       the comparison is not vacuous);
//   (b) determinism: the split path run twice on the same inputs (buffers
//       poisoned differently) is BIT-IDENTICAL with itself — fixed
//       partition count/chunking (pure function of host topk via
//       tq_sparse_split_parts), sequential fixed-order combine, no atomics;
//   (c) graph-capture safety: the split pair (partial kernel + PDL-launched
//       mla_combine) captured into a CUDA graph and replayed is
//       bit-identical to its own eager run (the engine's LS_TQ_DECODE_GRAPH
//       captures this chain);
//   (d) extent corners: empty / sub-block / block-boundary / interspersed
//       -1 / null topk_length, at both live shapes (glm5_next d_rope=0,
//       GLM-5.2 d_rope=64), plus an explicit-parts sweep.
//
// The split path is NOT bit-identical to the unsplit walk (the combine
// reorders FP32 sums) — that is why the bound is measured and asserted
// small rather than memcmp'd, and why the engine-side bar adds
// token-identity + TF-NLL (P-29 step 9 log).

#include <sm120/decode/tq_sparse/params.h>

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

namespace {

namespace tqs = sm120::decode::tq_sparse;

#define ASSERT_CUDA(expr)                                                     \
    do {                                                                      \
        cudaError_t _e = (expr);                                              \
        ASSERT_EQ(_e, cudaSuccess) << cudaGetErrorString(_e);                 \
    } while (0)

struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(size_t n) { EXPECT_EQ(cudaMalloc(&p, n), cudaSuccess); }
    ~DevBuf() { if (p) cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

// ULP distance between two finite floats (monotonic integer mapping).
int64_t ulp_dist(float a, float b) {
    if (a == b) return 0;  // covers +0 vs -0
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    const auto mono = [](int32_t i) -> int64_t {
        return i >= 0 ? (int64_t)i : (int64_t)0x80000000LL - i;
    };
    return std::llabs(mono(ia) - mono(ib));
}

struct SplitCase {
    int d_rope;          // 0 (glm5_next) or 64 (GLM-5.2)
    int page_size;       // 16 live
    int topk;            // host constant (2051 live)
    int limit;           // populated extent (topk_length value)
    int parts;           // requested split parts (<=1 unsplit)
    bool interspersed;   // sprinkle -1 within the populated prefix
    bool null_tkl;       // pass topk_length = nullptr
    bool zero_norms;     // adversarial zero-norm / zero-row corners
};

class TqSparseDecodeSplitKv : public ::testing::Test {
  protected:
    void SetUp() override {
        REQUIRES_GPU();
        ASSERT_CUDA(cudaSetDevice(0));
    }

    // Builds inputs for `c`, runs the UNSPLIT slim arm and the SPLIT arm
    // (c.parts partitions) into separately poisoned buffers. Asserts the
    // split-vs-unsplit numeric bound, split-vs-split bit identity, and —
    // when `negative_control` — that a flipped attended KV byte changes the
    // split output. When `graph_mode`, the split arm additionally runs
    // captured+replayed and must byte-match its own eager run.
    void run_case(const SplitCase& c, bool negative_control,
                  bool graph_mode = false) {
        const int h_q = 32, d_c = 512;
        const int packed = d_c / 2;
        const int row_bytes = packed + 2 + c.d_rope * 2;
        const int64_t blk_stride = (int64_t)row_bytes * c.page_size;
        const int pool_tokens = 4096;
        const int num_pages = (pool_tokens + c.page_size - 1) / c.page_size;

        std::mt19937 rng(0x5B171000u ^ (c.d_rope << 8) ^ c.limit ^
                         (c.parts << 16) ^ (c.zero_norms ? 0x10000 : 0));

        std::vector<uint8_t> h_kv((size_t)num_pages * blk_stride);
        for (auto& b : h_kv) b = (uint8_t)rng();
        for (int t = 0; t < pool_tokens; ++t) {
            uint8_t* row = h_kv.data() +
                (size_t)(t / c.page_size) * blk_stride +
                (size_t)(t % c.page_size) * row_bytes;
            float nv = (float)(rng() % 4096) / 2048.0f;
            if (c.zero_norms && (rng() % 5 == 0)) nv = 0.0f;
            __half nh = __float2half(nv);
            std::memcpy(row + packed, &nh, 2);
            for (int d = 0; d < c.d_rope; ++d) {
                float rv = ((float)(rng() % 2048) - 1024.0f) / 512.0f;
                __nv_bfloat16 bv = __float2bfloat16(rv);
                std::memcpy(row + packed + 2 + d * 2, &bv, 2);
            }
        }
        if (c.zero_norms)
            for (int t = 0; t < pool_tokens; t += 7)
                std::memset(h_kv.data() +
                    (size_t)(t / c.page_size) * blk_stride +
                    (size_t)(t % c.page_size) * row_bytes, 0, packed);

        std::vector<int> h_idx(c.topk, -1);
        std::vector<int> perm(pool_tokens);
        for (int i = 0; i < pool_tokens; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i < c.limit && i < c.topk; ++i) {
            if (c.interspersed && (rng() % 6 == 0)) continue;  // stays -1
            h_idx[i] = perm[i % pool_tokens];
        }
        const int h_tkl = c.limit;

        std::vector<float> h_qrot((size_t)h_q * d_c);
        for (auto& v : h_qrot)
            v = ((float)(rng() % 8192) - 4096.0f) / 1024.0f;
        const int d_qk = d_c + c.d_rope;
        std::vector<__nv_bfloat16> h_qrope(
            (size_t)h_q * (c.d_rope > 0 ? d_qk : 1));
        for (auto& v : h_qrope)
            v = __float2bfloat16(((float)(rng() % 2048) - 1024.0f) / 512.0f);
        std::vector<float> h_cent(16);
        for (auto& v : h_cent)
            v = ((float)(rng() % 4096) - 2048.0f) / 512.0f;

        const int parts_eff = tqs::tq_sparse_split_parts(c.topk, c.parts);
        ASSERT_GT(parts_eff, 1) << "case must exercise the split path";
        const size_t out_sz = (size_t)h_q * d_c * sizeof(float);
        const size_t lse_sz = (size_t)h_q * sizeof(float);
        const size_t oacc_sz = (size_t)parts_eff * h_q * d_c * sizeof(float);
        const size_t lacc_sz = (size_t)parts_eff * h_q * sizeof(float);

        DevBuf d_kv(h_kv.size()), d_idx(c.topk * sizeof(int)),
            d_tkl(sizeof(int)), d_qrot(h_qrot.size() * sizeof(float)),
            d_qrope(h_qrope.size() * sizeof(__nv_bfloat16)),
            d_cent(16 * sizeof(float)),
            d_out_ref(out_sz), d_out_s1(out_sz), d_out_s2(out_sz),
            d_lse_ref(lse_sz), d_lse_s1(lse_sz), d_lse_s2(lse_sz),
            d_oacc(oacc_sz), d_lacc(lacc_sz), d_nsp(2 * sizeof(int));
        ASSERT_CUDA(cudaMemcpy(d_kv.p, h_kv.data(), h_kv.size(),
                               cudaMemcpyHostToDevice));
        ASSERT_CUDA(cudaMemcpy(d_idx.p, h_idx.data(), c.topk * sizeof(int),
                               cudaMemcpyHostToDevice));
        ASSERT_CUDA(cudaMemcpy(d_tkl.p, &h_tkl, sizeof(int),
                               cudaMemcpyHostToDevice));
        ASSERT_CUDA(cudaMemcpy(d_qrot.p, h_qrot.data(),
                               h_qrot.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
        ASSERT_CUDA(cudaMemcpy(d_qrope.p, h_qrope.data(),
                               h_qrope.size() * sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice));
        ASSERT_CUDA(cudaMemcpy(d_cent.p, h_cent.data(), 16 * sizeof(float),
                               cudaMemcpyHostToDevice));
        const int ns[2] = {0, parts_eff};
        ASSERT_CUDA(cudaMemcpy(d_nsp.p, ns, sizeof(ns),
                               cudaMemcpyHostToDevice));

        tqs::TqSparseDecodeParams p{};
        p.b = 1; p.s_q = 1; p.h_q = h_q; p.h_kv = 1;
        p.d_c = d_c; p.d_rope = c.d_rope;
        p.sm_scale = 1.0f / std::sqrt((float)d_qk);
        p.q_rot = static_cast<const float*>(d_qrot.p);
        p.q_rope = static_cast<const __nv_bfloat16*>(d_qrope.p);
        p.q_rope_row_stride = c.d_rope > 0 ? d_qk : 0;
        p.kv_cache = static_cast<const uint8_t*>(d_kv.p);
        p.cache_stride_block = blk_stride;
        p.cache_stride_row = row_bytes;
        p.page_block_size = c.page_size;
        p.indices = static_cast<const int*>(d_idx.p);
        p.topk = c.topk;
        p.stride_indices_b = 0; p.stride_indices_s_q = 0;
        p.topk_length =
            c.null_tkl ? nullptr : static_cast<const int*>(d_tkl.p);
        p.use_slim = 1;
        p.centroids = static_cast<const float*>(d_cent.p);
        p.stride_o_b = h_q * d_c; p.stride_o_s_q = h_q * d_c;
        p.stride_o_h_q = d_c;
        p.stride_lse_b = h_q; p.stride_lse_s_q = h_q;
        p.stream = nullptr;

        // ── Arm A: unsplit slim (num_sm_parts left 0) ──
        ASSERT_CUDA(cudaMemset(d_out_ref.p, 0xAA, out_sz));
        ASSERT_CUDA(cudaMemset(d_lse_ref.p, 0xAA, lse_sz));
        p.out = static_cast<float*>(d_out_ref.p);
        p.lse = static_cast<float*>(d_lse_ref.p);
        tqs::run_tq_sparse_decode(p);
        ASSERT_CUDA(cudaGetLastError());

        // ── Arm B: split (poison out, lse AND accumulators) ──
        auto arm_split = [&](void* out, void* lse, uint8_t poison) {
            ASSERT_CUDA(cudaMemset(out, poison, out_sz));
            ASSERT_CUDA(cudaMemset(lse, poison, lse_sz));
            ASSERT_CUDA(cudaMemset(d_oacc.p, poison ^ 0xFF, oacc_sz));
            ASSERT_CUDA(cudaMemset(d_lacc.p, poison ^ 0xFF, lacc_sz));
            p.num_sm_parts = c.parts;
            p.o_accum = static_cast<float*>(d_oacc.p);
            p.lse_accum = static_cast<float*>(d_lacc.p);
            p.num_splits_ptr = static_cast<int*>(d_nsp.p);
            p.out = static_cast<float*>(out);
            p.lse = static_cast<float*>(lse);
            tqs::run_tq_sparse_decode(p);
            ASSERT_EQ(cudaGetLastError(), cudaSuccess);
            p.num_sm_parts = 0;
            p.o_accum = nullptr; p.lse_accum = nullptr;
            p.num_splits_ptr = nullptr;
        };
        arm_split(d_out_s1.p, d_lse_s1.p, 0x55);
        if (HasFatalFailure()) return;
        arm_split(d_out_s2.p, d_lse_s2.p, 0x33);
        if (HasFatalFailure()) return;
        ASSERT_CUDA(cudaDeviceSynchronize());

        std::vector<float> ref(h_q * d_c), s1(h_q * d_c), s2(h_q * d_c);
        std::vector<float> lref(h_q), l1(h_q), l2(h_q);
        ASSERT_CUDA(cudaMemcpy(ref.data(), d_out_ref.p, out_sz,
                               cudaMemcpyDeviceToHost));
        ASSERT_CUDA(cudaMemcpy(s1.data(), d_out_s1.p, out_sz,
                               cudaMemcpyDeviceToHost));
        ASSERT_CUDA(cudaMemcpy(s2.data(), d_out_s2.p, out_sz,
                               cudaMemcpyDeviceToHost));
        ASSERT_CUDA(cudaMemcpy(lref.data(), d_lse_ref.p, lse_sz,
                               cudaMemcpyDeviceToHost));
        ASSERT_CUDA(cudaMemcpy(l1.data(), d_lse_s1.p, lse_sz,
                               cudaMemcpyDeviceToHost));
        ASSERT_CUDA(cudaMemcpy(l2.data(), d_lse_s2.p, lse_sz,
                               cudaMemcpyDeviceToHost));

        // (b) determinism: split run twice → bit-identical with itself.
        ASSERT_EQ(std::memcmp(s1.data(), s2.data(), out_sz), 0)
            << "split path is run-to-run NONDETERMINISTIC (out)";
        ASSERT_EQ(std::memcmp(l1.data(), l2.data(), lse_sz), 0)
            << "split path is run-to-run NONDETERMINISTIC (lse)";

        // (a) split-vs-unsplit bound.
        if (c.limit == 0 && !c.null_tkl) {
            // Empty extent: both arms must produce exact zeros; lse is the
            // -1e30 empty-class sentinel in both (combine may round it).
            for (int i = 0; i < h_q * d_c; ++i) {
                ASSERT_EQ(ref[i], 0.0f) << "unsplit empty out not zero @" << i;
                ASSERT_EQ(s1[i], 0.0f) << "split empty out not zero @" << i;
            }
            for (int i = 0; i < h_q; ++i) {
                ASSERT_LE(lref[i], -1e29f);
                ASSERT_LE(l1[i], -1e29f);
            }
        } else {
            // Mixed bound: an element passes on EITHER a tight absolute
            // delta (near-zero outputs, where ULP distance is meaningless —
            // a 1e-5 shift at 1e-7 magnitude is thousands of ULP) OR a
            // tight ULP distance (large outputs, where abs alone is loose).
            // Wide misses on BOTH = a combine wiring bug, not reassociation.
            constexpr double kAbsTol = 5e-5;
            constexpr int64_t kUlpTol = 256;
            double max_abs = 0.0; int64_t max_ulp = 0; int argmax_i = -1;
            int violations = 0; int first_viol = -1;
            for (int i = 0; i < h_q * d_c; ++i) {
                ASSERT_TRUE(std::isfinite(s1[i])) << "split out NaN/Inf @" << i;
                const double d = std::fabs((double)ref[i] - (double)s1[i]);
                const int64_t u = ulp_dist(ref[i], s1[i]);
                if (d > max_abs) { max_abs = d; argmax_i = i; }
                max_ulp = std::max(max_ulp, u);
                if (d > kAbsTol && u > kUlpTol) {
                    if (first_viol < 0) first_viol = i;
                    ++violations;
                }
            }
            double lse_max_abs = 0.0;
            for (int i = 0; i < h_q; ++i)
                lse_max_abs = std::max(lse_max_abs,
                    std::fabs((double)lref[i] - (double)l1[i]));
            RecordProperty("max_abs", std::to_string(max_abs));
            RecordProperty("max_ulp", std::to_string(max_ulp));
            std::printf("[splitkv] d_rope=%d limit=%d parts=%d(eff %d): "
                        "out max_abs=%.3e (elem %d) max_ulp=%lld "
                        "mixed-bound violations=%d; lse max_abs=%.3e\n",
                        c.d_rope, c.limit, c.parts, parts_eff, max_abs,
                        argmax_i, (long long)max_ulp, violations,
                        lse_max_abs);
            EXPECT_EQ(violations, 0)
                << "split-vs-unsplit delta too wide (first elem "
                << first_viol << ": unsplit=" << ref[std::max(first_viol, 0)]
                << " split=" << s1[std::max(first_viol, 0)] << ")";
            EXPECT_LE(max_abs, 1e-3) << "split-vs-unsplit abs delta too wide";
            EXPECT_LE(lse_max_abs, 1e-4) << "lse delta too wide";
        }

        // (c) graph capture + replay bit-identity with the eager split run.
        if (graph_mode) {
            cudaStream_t cs = nullptr;
            ASSERT_CUDA(cudaStreamCreate(&cs));
            ASSERT_CUDA(cudaMemset(d_out_s2.p, 0x77, out_sz));
            ASSERT_CUDA(cudaMemset(d_lse_s2.p, 0x77, lse_sz));
            p.num_sm_parts = c.parts;
            p.o_accum = static_cast<float*>(d_oacc.p);
            p.lse_accum = static_cast<float*>(d_lacc.p);
            p.num_splits_ptr = static_cast<int*>(d_nsp.p);
            p.out = static_cast<float*>(d_out_s2.p);
            p.lse = static_cast<float*>(d_lse_s2.p);
            p.stream = cs;
            ASSERT_CUDA(cudaStreamBeginCapture(
                cs, cudaStreamCaptureModeThreadLocal));
            tqs::run_tq_sparse_decode(p);
            cudaGraph_t graph = nullptr;
            ASSERT_CUDA(cudaStreamEndCapture(cs, &graph));
            ASSERT_NE(graph, nullptr);
            cudaGraphExec_t exec = nullptr;
            ASSERT_CUDA(cudaGraphInstantiateWithFlags(&exec, graph, 0));
            ASSERT_CUDA(cudaGraphLaunch(exec, cs));
            ASSERT_CUDA(cudaGraphLaunch(exec, cs));  // replay twice
            ASSERT_CUDA(cudaStreamSynchronize(cs));
            cudaGraphDestroy(graph);
            cudaGraphExecDestroy(exec);
            cudaStreamDestroy(cs);
            p.stream = nullptr;
            p.num_sm_parts = 0;
            p.o_accum = nullptr; p.lse_accum = nullptr;
            p.num_splits_ptr = nullptr;

            std::vector<float> sg(h_q * d_c), lg(h_q);
            ASSERT_CUDA(cudaMemcpy(sg.data(), d_out_s2.p, out_sz,
                                   cudaMemcpyDeviceToHost));
            ASSERT_CUDA(cudaMemcpy(lg.data(), d_lse_s2.p, lse_sz,
                                   cudaMemcpyDeviceToHost));
            ASSERT_EQ(std::memcmp(s1.data(), sg.data(), out_sz), 0)
                << "captured+replayed split differs from its eager run (out)";
            ASSERT_EQ(std::memcmp(l1.data(), lg.data(), lse_sz), 0)
                << "captured+replayed split differs from its eager run (lse)";
        }

        // Negative control: a flipped attended KV byte MUST change the
        // split output — proves the comparison is sensitive to the data.
        if (negative_control) {
            ASSERT_GE(c.limit, 8) << "control needs valid tokens";
            const int tok = h_idx[5];
            ASSERT_GE(tok, 0) << "control token must be valid";
            const size_t off =
                (size_t)(tok / c.page_size) * blk_stride +
                (size_t)(tok % c.page_size) * row_bytes + 17;
            const uint8_t flipped = h_kv[off] ^ 0xFF;
            ASSERT_CUDA(cudaMemcpy(static_cast<uint8_t*>(d_kv.p) + off,
                                   &flipped, 1, cudaMemcpyHostToDevice));
            arm_split(d_out_s2.p, d_lse_s2.p, 0x11);
            if (HasFatalFailure()) return;
            ASSERT_CUDA(cudaDeviceSynchronize());
            std::vector<float> sflip(h_q * d_c);
            ASSERT_CUDA(cudaMemcpy(sflip.data(), d_out_s2.p, out_sz,
                                   cudaMemcpyDeviceToHost));
            ASSERT_NE(std::memcmp(s1.data(), sflip.data(), out_sz), 0)
                << "negative control: a flipped KV byte in an attended token "
                   "left the split output identical — the comparison is "
                   "vacuous";
        }
    }
};

// Pure-function contract of the chunking helper (host-side, no GPU).
TEST(TqSparseSplitParts, PureFunctionOfShape) {
    // topk 2051 → 33 blocks → auto (request 33) = 33 parts, chunk 1.
    EXPECT_EQ(tqs::tq_sparse_split_parts(2051, 33), 33);
    // Requests are recomputed so no partition has an empty HOST range.
    EXPECT_EQ(tqs::tq_sparse_split_parts(2051, 16), 11);  // chunk 3 → 11
    EXPECT_EQ(tqs::tq_sparse_split_parts(2051, 8), 7);    // chunk 5 → ceil(33/5)
    EXPECT_EQ(tqs::tq_sparse_split_parts(2051, 64), 33);  // capped at blocks
    EXPECT_EQ(tqs::tq_sparse_split_parts(2051, 1), 1);
    EXPECT_EQ(tqs::tq_sparse_split_parts(2051, 0), 1);
    EXPECT_EQ(tqs::tq_sparse_split_parts(64, 33), 1);     // 1 block → unsplit
    EXPECT_EQ(tqs::tq_sparse_split_parts(128, 33), 2);
}

// glm5_next live shape (d_rope 0, page 16, topk 2051) at the saturated 8k
// budget, auto parts — bound + determinism + negative control.
TEST_F(TqSparseDecodeSplitKv, Glm5NextFullBudgetAutoParts) {
    run_case({0, 16, 2051, 2051, 33, false, false, false}, true);
}

// Graph capture + replay of the split pair (partial + PDL combine) must be
// bit-identical to the eager split run — the engine captures this chain.
TEST_F(TqSparseDecodeSplitKv, GraphCaptureReplayBitIdentical) {
    run_case({0, 16, 2051, 2051, 33, false, false, false}, false, true);
}

// Partial populated extents: empty, sub-block, block boundary, short ctx.
TEST_F(TqSparseDecodeSplitKv, Glm5NextPartialExtents) {
    for (int limit : {0, 1, 63, 64, 65, 400}) {
        run_case({0, 16, 2051, limit, 33, false, false, false}, false);
        if (HasFatalFailure()) return;
    }
}

// -1 interspersed WITHIN the populated prefix (bound-rejected DSA picks).
TEST_F(TqSparseDecodeSplitKv, Glm5NextInterspersedInvalid) {
    run_case({0, 16, 2051, 1500, 33, true, false, false}, false);
}

// Null topk_length → split walks the full host topk, chunked.
TEST_F(TqSparseDecodeSplitKv, NullTopkLengthFullWalk) {
    run_case({0, 16, 2051, 2051, 33, false, true, false}, false);
}

// Zero norms + zero rows: the ±0 softmax corners under partitioning.
TEST_F(TqSparseDecodeSplitKv, ZeroNormAndZeroRowCorners) {
    run_case({0, 16, 2051, 2051, 33, true, false, true}, false);
    if (HasFatalFailure()) return;
    run_case({0, 16, 2051, 128, 33, false, false, true}, false);
}

// Explicit part-count sweep (the LS_TQ_SPLITKV=N experiment surface).
TEST_F(TqSparseDecodeSplitKv, ExplicitPartsSweep) {
    for (int parts : {2, 8, 16, 33}) {
        run_case({0, 16, 2051, 2051, parts, false, false, false}, false);
        if (HasFatalFailure()) return;
    }
}

// GLM-5.2 live shape (d_rope 64) — the TQ golden's kernel path.
TEST_F(TqSparseDecodeSplitKv, Glm52RopeShape) {
    run_case({64, 16, 2051, 2051, 33, false, false, false}, false);
    if (HasFatalFailure()) return;
    run_case({64, 16, 2051, 777, 33, true, false, false}, false);
}

}  // namespace
