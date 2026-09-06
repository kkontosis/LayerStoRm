// Bit-identity proof for the slim tq_sparse_decode kernel (P-29 step 6 —
// slim tq_sparse_decode): the slim variant (params.use_slim = 1) must produce bytes
// IDENTICAL to the reference kernel on the same inputs — out AND lse — for
// every live shape (glm5_next d_rope=0 and GLM-5.2 d_rope=64, page 16), for
// partial/empty/interspersed populated extents (the translate-kernel -1
// padding contract), for a null topk_length, and for adversarial zero-norm /
// zero-row inputs that drive the ±0 softmax corners.
//
// Anti-vacuity (house pattern, graph_node_ops_test.cu): outputs are poisoned
// with different fills before each arm, and a negative control perturbs one
// gathered KV byte and asserts the comparison DOES fail — proving the memcmp
// is sensitive to the attended data.

#include <sm120/decode/tq_sparse/params.h>

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
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

struct SlimCase {
    int d_rope;          // 0 (glm5_next) or 64 (GLM-5.2)
    int page_size;       // 16 live
    int topk;            // host constant (2051 live)
    int limit;           // populated extent (topk_length value)
    bool interspersed;   // sprinkle -1 within the populated prefix
    bool null_tkl;       // pass topk_length = nullptr
    bool zero_norms;     // zero norms + zero packed rows (±0 corners)
};

class TqSparseDecodeSlim : public ::testing::Test {
  protected:
    void SetUp() override {
        REQUIRES_GPU();
        ASSERT_CUDA(cudaSetDevice(0));
    }

    // Builds inputs for `c`, runs reference and slim arms into separately
    // poisoned buffers, and asserts byte equality of out + lse. When
    // `negative_control`, additionally perturbs one attended KV byte and
    // asserts the reference re-run DIFFERS from the clean reference.
    void run_case(const SlimCase& c, bool negative_control) {
        const int h_q = 32, d_c = 512;
        const int packed = d_c / 2;
        const int row_bytes = packed + 2 + c.d_rope * 2;
        const int64_t blk_stride = (int64_t)row_bytes * c.page_size;
        const int pool_tokens = 4096;
        const int num_pages = (pool_tokens + c.page_size - 1) / c.page_size;

        // Shape-seeded RNG: position-dependent content (house pattern).
        std::mt19937 rng(0xBEEF0000u ^ (c.d_rope << 8) ^ c.limit ^
                         (c.zero_norms ? 0x10000 : 0));

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

        // Translate-kernel contract: indices[i] == -1 for i >= limit.
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

        const size_t out_sz = (size_t)h_q * d_c * sizeof(float);
        const size_t lse_sz = (size_t)h_q * sizeof(float);
        DevBuf d_kv(h_kv.size()), d_idx(c.topk * sizeof(int)),
            d_tkl(sizeof(int)), d_qrot(h_qrot.size() * sizeof(float)),
            d_qrope(h_qrope.size() * sizeof(__nv_bfloat16)),
            d_cent(16 * sizeof(float)), d_out_a(out_sz), d_out_b(out_sz),
            d_lse_a(lse_sz), d_lse_b(lse_sz);
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
        p.centroids = static_cast<const float*>(d_cent.p);
        p.stride_o_b = h_q * d_c; p.stride_o_s_q = h_q * d_c;
        p.stride_o_h_q = d_c;
        p.stride_lse_b = h_q; p.stride_lse_s_q = h_q;
        p.stream = nullptr;

        // Different poison per arm — a non-writing kernel cannot pass.
        ASSERT_CUDA(cudaMemset(d_out_a.p, 0xAA, out_sz));
        ASSERT_CUDA(cudaMemset(d_out_b.p, 0x55, out_sz));
        ASSERT_CUDA(cudaMemset(d_lse_a.p, 0xAA, lse_sz));
        ASSERT_CUDA(cudaMemset(d_lse_b.p, 0x55, lse_sz));

        p.use_slim = 0;
        p.out = static_cast<float*>(d_out_a.p);
        p.lse = static_cast<float*>(d_lse_a.p);
        tqs::run_tq_sparse_decode(p);
        ASSERT_CUDA(cudaGetLastError());
        p.use_slim = 1;
        p.out = static_cast<float*>(d_out_b.p);
        p.lse = static_cast<float*>(d_lse_b.p);
        tqs::run_tq_sparse_decode(p);
        ASSERT_CUDA(cudaGetLastError());
        ASSERT_CUDA(cudaDeviceSynchronize());

        std::vector<uint8_t> a(out_sz), b(out_sz), la(lse_sz), lb(lse_sz);
        ASSERT_CUDA(cudaMemcpy(a.data(), d_out_a.p, out_sz,
                               cudaMemcpyDeviceToHost));
        ASSERT_CUDA(cudaMemcpy(b.data(), d_out_b.p, out_sz,
                               cudaMemcpyDeviceToHost));
        ASSERT_CUDA(cudaMemcpy(la.data(), d_lse_a.p, lse_sz,
                               cudaMemcpyDeviceToHost));
        ASSERT_CUDA(cudaMemcpy(lb.data(), d_lse_b.p, lse_sz,
                               cudaMemcpyDeviceToHost));
        size_t diff = 0, first = out_sz;
        for (size_t i = 0; i < out_sz; ++i)
            if (a[i] != b[i]) { ++diff; if (first == out_sz) first = i; }
        ASSERT_EQ(diff, 0u) << "out: " << diff
            << " differing byte(s), first at " << first;
        ASSERT_EQ(std::memcmp(la.data(), lb.data(), lse_sz), 0)
            << "lse bytes differ";

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
            p.use_slim = 0;
            p.out = static_cast<float*>(d_out_b.p);
            p.lse = static_cast<float*>(d_lse_b.p);
            tqs::run_tq_sparse_decode(p);
            ASSERT_CUDA(cudaDeviceSynchronize());
            ASSERT_CUDA(cudaMemcpy(b.data(), d_out_b.p, out_sz,
                                   cudaMemcpyDeviceToHost));
            ASSERT_NE(std::memcmp(a.data(), b.data(), out_sz), 0)
                << "negative control: a flipped KV byte in an attended token "
                   "left the output bytes identical — the comparison is "
                   "vacuous";
        }
    }
};

// glm5_next live shape (d_rope 0, page 16, topk 2051) at the 8k plateau
// (full budget populated) — with the negative control.
TEST_F(TqSparseDecodeSlim, Glm5NextFullBudgetBitIdentical) {
    run_case({0, 16, 2051, 2051, false, false, false}, true);
}

// Partial populated extents: sub-block, exactly one block, short context.
TEST_F(TqSparseDecodeSlim, Glm5NextPartialExtents) {
    for (int limit : {0, 1, 63, 64, 65, 400}) {
        run_case({0, 16, 2051, limit, false, false, false}, false);
        if (HasFatalFailure()) return;
    }
}

// -1 interspersed WITHIN the populated prefix (bound-rejected DSA picks).
TEST_F(TqSparseDecodeSlim, Glm5NextInterspersedInvalid) {
    run_case({0, 16, 2051, 1500, true, false, false}, false);
}

// Null topk_length → slim must walk the full topk like the reference.
TEST_F(TqSparseDecodeSlim, NullTopkLengthFallsBackToFullWalk) {
    run_case({0, 16, 2051, 2051, false, true, false}, false);
}

// Zero norms + all-zero packed rows: exercises the ±0 score corners the
// d_rope==0 rope-degeneration argument relies on.
TEST_F(TqSparseDecodeSlim, ZeroNormAndZeroRowCorners) {
    run_case({0, 16, 2051, 2051, true, false, true}, false);
    if (HasFatalFailure()) return;
    run_case({0, 16, 2051, 128, false, false, true}, false);
}

// GLM-5.2 live shape (d_rope 64): full budget with negative control, then a
// short interspersed extent.
TEST_F(TqSparseDecodeSlim, Glm52RopeBitIdentical) {
    run_case({64, 16, 2051, 2051, false, false, false}, true);
    if (HasFatalFailure()) return;
    run_case({64, 16, 2051, 777, true, false, false}, false);
}

}  // namespace
