// TD-LSE-UNITS-DECODE-TQ: LSE unit contract for the TurboQuant (TQ) decode
// kernels — KVS-1 repeated for the TQ backend (tq_dense, tq_sparse, csa_tq).
//
// All TQ decode kernels score in NATURAL log units (scores *= sm_scale,
// expf(score - M)) and accumulate the full-precision exp row-sum L. The
// contract (INV-LSE-NAT + mla_combine):
//   external lse       = M + log(L)            (NATURAL)
//   split   lse_accum  = log2(L) + M*LOG2E     (TRUE log2 — mla_combine
//                        reweights splits by exp2f(lse_accum - global))
//   split   o_accum    = NORMALIZED partial (acc / L) — mla_combine forms a
//                        convex combination of per-split outputs.
// Audit result:
//   tq_dense:  LSE units correct on all three counts (confirmed here).
//   tq_sparse: single-split only (no lse_accum path); external lse natural.
//   csa_tq:    external lse natural BUT the split path wrote NATURAL-unit
//              lse_accum AND UNNORMALIZED o_accum → the multi-split combined
//              output was silently corrupted (KVS-1-class bug, fixed; the
//              natural-unit lse_accum alone shifts the combined lse by ×ln2).
//   BONUS (found by the planted-token discriminator, NOT an LSE-unit bug):
//              all three TQ kernels loaded q_rope registers with a
//              `tid*2 < d_rope` guard while the ROPE score loop reduces a
//              full-d_rope dot product inside EACH warp — warps 1..7 held
//              qrope=0, silently zeroing the rope score for tokens 8..63 of
//              every 64-token KV block (fixed: load by lane). The plant at
//              token 2040 sits at in-block offset 56 (warp 7), so its +15
//              rope logit vanished and both multi-split tests read a
//              combined lse of ~ln(2048)≈7.63 instead of ~15.
//
// The caches are built HOST-SIDE (exact packed codes / fp16 norms / bf16
// rope), so the CPU reference is exact modulo fp32 accumulation order —
// tolerances are tight. The multi-split tests plant a +15-natural-unit token
// deep in the sequence (bf16-exact rope alignment with a constant q_rope) so
// one split's max dominates: the discriminator that catches both the
// natural-vs-log2 lse_accum mis-weighting and the missing normalization.

#include "compute/kernels/attention/tq_mla_attention.h"

#include "smxx/get_mla_metadata.h"
#include "smxx/mla_combine.h"
#include <sm120/decode/tq_dense/params.h>
#include <sm120/decode/tq_sparse/params.h>
#include <sm120/decode/csa_tq/params.h>

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstring>
#include <limits>
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

constexpr int kDC = 512;          // kv_lora_rank (d_c / head_dim)
constexpr int kDR = 64;           // qk_rope_head_dim
constexpr int kHQ = 64;           // heads (num_q_seqs = 64, s_q = 1, b = 1)
constexpr int kPage = 64;
constexpr int kPacked = kDC / 2;  // 256 packed bytes per token
constexpr int kNumCentroids = 16;

// V3.2 TQ row: [256B packed | 2B fp16 norm | 128B bf16 rope] = 386 B
constexpr int kTqRowBytes = kPacked + 2 + kDR * 2;
// V4 csa_tq entry: K [256B packed | 2B norm | 128B rope] + V [256B packed | 2B norm]
constexpr int kV4KNormOff = 256;
constexpr int kV4KRopeOff = 258;
constexpr int kV4VPackedOff = 386;
constexpr int kV4VNormOff = 642;
constexpr int kV4EntryBytes = 644;

float bf16r(float f) { return __bfloat162float(__float2bfloat16(f)); }
float fp16r(float f) { return __half2float(__float2half(f)); }

struct Tok {
    std::vector<uint8_t> k_packed;  // [256] two 4-bit codes per byte
    float k_norm;                   // fp16-rounded
    std::vector<float> rope;        // [64] bf16-rounded
    std::vector<uint8_t> v_packed;  // [256] (csa_tq V phase)
    float v_norm;                   // fp16-rounded
};

struct TqRig {
    std::vector<Tok> toks;
    std::vector<float> centroids;   // [16]
    std::vector<float> q_rot;       // [kHQ * kDC] fp32 (as the kernel consumes)
    std::vector<float> q_rope;      // [kHQ * kDR] bf16-rounded

    // Device buffers
    uint8_t* d_cache_v32 = nullptr;   // paged 386B rows (tq_dense / tq_sparse)
    uint8_t* d_cache_v4 = nullptr;    // flat 644B entries (csa_tq)
    float* d_centroids = nullptr;
    float* d_q_rot = nullptr;
    __nv_bfloat16* d_q_rope = nullptr;
    int* d_indices = nullptr;         // identity [0..N)
    int* d_block_table = nullptr;     // identity pages
    int* d_seqlens = nullptr;         // [1] = N
    float* d_out = nullptr;           // [kHQ, kDC] fp32 (rotated space)
    float* d_lse = nullptr;           // [kHQ]
    float* d_lse_accum = nullptr;
    float* d_o_accum = nullptr;
    int* d_sched_meta = nullptr;
    int* d_num_splits = nullptr;

    int num_tokens = 0;
    int num_pages = 0;
    int max_splits = 0;

    ~TqRig() {
        for (void* p : {(void*)d_cache_v32, (void*)d_cache_v4,
                        (void*)d_centroids, (void*)d_q_rot, (void*)d_q_rope,
                        (void*)d_indices, (void*)d_block_table,
                        (void*)d_seqlens, (void*)d_out, (void*)d_lse,
                        (void*)d_lse_accum, (void*)d_o_accum,
                        (void*)d_sched_meta, (void*)d_num_splits})
            if (p) cudaFree(p);
    }
};

// Build tokens + Q host-side, upload both cache formats. plant_token >= 0
// gives that token a bf16-exact rope aligned with the constant q_rope:
// logit = 64 * 0.5 * 0.46875 * sm_scale(=1) = +15 natural units above the
// noise floor — its split's max dominates the combine.
void build_rig(TqRig& rig, int num_tokens, int max_sm_parts, int plant_token,
               unsigned seed) {
    rig.num_tokens = num_tokens;
    rig.num_pages = (num_tokens + kPage - 1) / kPage;
    rig.max_splits = max_sm_parts + 2;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> qdist(-0.02f, 0.02f);
    std::uniform_real_distribution<float> rdist(-0.02f, 0.02f);
    std::uniform_real_distribution<float> ndist(0.5f, 1.5f);
    std::uniform_int_distribution<int> bdist(0, 255);

    rig.centroids.resize(kNumCentroids);
    for (int i = 0; i < kNumCentroids; ++i)
        rig.centroids[i] = (static_cast<float>(i) - 7.5f) / 7.5f;

    rig.q_rot.resize(static_cast<size_t>(kHQ) * kDC);
    for (auto& v : rig.q_rot) v = qdist(rng);
    rig.q_rope.assign(static_cast<size_t>(kHQ) * kDR, bf16r(0.5f));

    rig.toks.resize(num_tokens);
    for (int t = 0; t < num_tokens; ++t) {
        auto& tk = rig.toks[t];
        tk.k_packed.resize(kPacked);
        tk.v_packed.resize(kPacked);
        for (auto& b : tk.k_packed) b = static_cast<uint8_t>(bdist(rng));
        for (auto& b : tk.v_packed) b = static_cast<uint8_t>(bdist(rng));
        tk.k_norm = fp16r(ndist(rng));
        tk.v_norm = fp16r(ndist(rng));
        tk.rope.resize(kDR);
        for (auto& v : tk.rope) v = bf16r(rdist(rng));
    }
    if (plant_token >= 0) {
        // 0.46875 and 0.5 are bf16-exact: planted logit = 64*0.5*0.46875 = 15.
        for (auto& v : rig.toks[plant_token].rope) v = 0.46875f;
    }

    // ── V3.2 paged cache (tq_dense / tq_sparse) ──
    const int64_t v32_bytes = (int64_t)rig.num_pages * kPage * kTqRowBytes;
    std::vector<uint8_t> v32(v32_bytes, 0);
    for (int t = 0; t < num_tokens; ++t) {
        uint8_t* row = v32.data() + (int64_t)(t / kPage) * kPage * kTqRowBytes +
                       (int64_t)(t % kPage) * kTqRowBytes;
        const auto& tk = rig.toks[t];
        std::memcpy(row, tk.k_packed.data(), kPacked);
        __half hn = __float2half(tk.k_norm);
        std::memcpy(row + kPacked, &hn, 2);
        for (int d = 0; d < kDR; ++d) {
            __nv_bfloat16 rb = __float2bfloat16(tk.rope[d]);
            std::memcpy(row + kPacked + 2 + d * 2, &rb, 2);
        }
    }
    CUDA_CHECK(cudaMalloc(&rig.d_cache_v32, v32_bytes));
    CUDA_CHECK(cudaMemcpy(rig.d_cache_v32, v32.data(), v32_bytes,
                          cudaMemcpyHostToDevice));

    // ── V4 flat cache (csa_tq) ──
    const int64_t v4_bytes = (int64_t)num_tokens * kV4EntryBytes;
    std::vector<uint8_t> v4(v4_bytes, 0);
    for (int t = 0; t < num_tokens; ++t) {
        uint8_t* e = v4.data() + (int64_t)t * kV4EntryBytes;
        const auto& tk = rig.toks[t];
        std::memcpy(e, tk.k_packed.data(), kPacked);
        __half kn = __float2half(tk.k_norm);
        std::memcpy(e + kV4KNormOff, &kn, 2);
        for (int d = 0; d < kDR; ++d) {
            __nv_bfloat16 rb = __float2bfloat16(tk.rope[d]);
            std::memcpy(e + kV4KRopeOff + d * 2, &rb, 2);
        }
        std::memcpy(e + kV4VPackedOff, tk.v_packed.data(), kPacked);
        __half vn = __float2half(tk.v_norm);
        std::memcpy(e + kV4VNormOff, &vn, 2);
    }
    CUDA_CHECK(cudaMalloc(&rig.d_cache_v4, v4_bytes));
    CUDA_CHECK(cudaMemcpy(rig.d_cache_v4, v4.data(), v4_bytes,
                          cudaMemcpyHostToDevice));

    // ── Q + codebook ──
    CUDA_CHECK(cudaMalloc(&rig.d_centroids, kNumCentroids * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(rig.d_centroids, rig.centroids.data(),
                          kNumCentroids * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&rig.d_q_rot, rig.q_rot.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(rig.d_q_rot, rig.q_rot.data(),
                          rig.q_rot.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    std::vector<__nv_bfloat16> qr_b(rig.q_rope.size());
    for (size_t i = 0; i < rig.q_rope.size(); ++i)
        qr_b[i] = __float2bfloat16(rig.q_rope[i]);
    CUDA_CHECK(cudaMalloc(&rig.d_q_rope, qr_b.size() * 2));
    CUDA_CHECK(cudaMemcpy(rig.d_q_rope, qr_b.data(), qr_b.size() * 2,
                          cudaMemcpyHostToDevice));

    // ── Index / paging / split scratch ──
    std::vector<int> idx(num_tokens);
    for (int i = 0; i < num_tokens; ++i) idx[i] = i;
    CUDA_CHECK(cudaMalloc(&rig.d_indices, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(rig.d_indices, idx.data(), num_tokens * sizeof(int),
                          cudaMemcpyHostToDevice));
    std::vector<int> bt(rig.num_pages);
    for (int i = 0; i < rig.num_pages; ++i) bt[i] = i;
    CUDA_CHECK(cudaMalloc(&rig.d_block_table, rig.num_pages * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(rig.d_block_table, bt.data(),
                          rig.num_pages * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&rig.d_seqlens, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(rig.d_seqlens, &num_tokens, sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&rig.d_out,
                          static_cast<size_t>(kHQ) * kDC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rig.d_lse, kHQ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rig.d_lse_accum, static_cast<size_t>(rig.max_splits)
                          * kHQ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rig.d_o_accum, static_cast<size_t>(rig.max_splits)
                          * kHQ * kDC * sizeof(float)));
    std::vector<float> neg_inf(static_cast<size_t>(rig.max_splits) * kHQ,
                               -std::numeric_limits<float>::infinity());
    CUDA_CHECK(cudaMemcpy(rig.d_lse_accum, neg_inf.data(),
                          neg_inf.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(rig.d_o_accum, 0, static_cast<size_t>(rig.max_splits)
                          * kHQ * kDC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rig.d_sched_meta, max_sm_parts * 8 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&rig.d_num_splits, 2 * sizeof(int)));
}

// CPU reference from the EXACT stored values (double softmax).
//   score_t(h) = [ (sum_d q_rot(h,d)*cent(k_code(t,d))) * k_norm(t)
//                  + sum_d q_rope(h,d)*rope(t,d) ] * sm_scale
//   out(h,d)   = sum_t p_t * pv_norm(t) * cent(pv_code(t,d))   (rotated space)
// use_v selects the csa_tq V-phase codes/norm; otherwise V = K (V3.2 TQ).
void cpu_reference(const TqRig& rig, float sm_scale, bool use_v,
                   std::vector<float>& out, std::vector<float>& lse) {
    const int N = rig.num_tokens;
    out.assign(static_cast<size_t>(kHQ) * kDC, 0.0f);
    lse.assign(kHQ, 0.0f);
    std::vector<double> scores(N);
    for (int h = 0; h < kHQ; ++h) {
        const float* qn = rig.q_rot.data() + static_cast<size_t>(h) * kDC;
        const float* qr = rig.q_rope.data() + static_cast<size_t>(h) * kDR;
        double m = -1e300;
        for (int t = 0; t < N; ++t) {
            const auto& tk = rig.toks[t];
            double nope = 0.0;
            for (int i = 0; i < kPacked; ++i) {
                const uint8_t b = tk.k_packed[i];
                nope += static_cast<double>(qn[i * 2]) *
                            rig.centroids[b & 0x0F] +
                        static_cast<double>(qn[i * 2 + 1]) *
                            rig.centroids[(b >> 4) & 0x0F];
            }
            double rope = 0.0;
            for (int d = 0; d < kDR; ++d)
                rope += static_cast<double>(qr[d]) * tk.rope[d];
            scores[t] = (nope * tk.k_norm + rope) * sm_scale;
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
            const auto& tk = rig.toks[t];
            const auto& codes = use_v ? tk.v_packed : tk.k_packed;
            const double w = scores[t] / l *
                             (use_v ? tk.v_norm : tk.k_norm);
            for (int i = 0; i < kPacked; ++i) {
                const uint8_t b = codes[i];
                orow[i * 2] += static_cast<float>(w * rig.centroids[b & 0x0F]);
                orow[i * 2 + 1] +=
                    static_cast<float>(w * rig.centroids[(b >> 4) & 0x0F]);
            }
        }
    }
}

void fill_metadata(TqRig& rig, int num_sm_parts, bool topk_mode) {
    GetMlaMetadataParams mp{};
    mp.seqlens_k_ptr = rig.d_seqlens;
    mp.tile_scheduler_metadata_ptr = rig.d_sched_meta;
    mp.num_splits_ptr = rig.d_num_splits;
    mp.batch_size = 1;
    mp.block_size_n = 64;
    mp.fixed_overhead_num_blocks = 1;
    mp.num_sm_parts = num_sm_parts;
    mp.topk = topk_mode ? rig.num_tokens : -1;
    lc::launch_get_mla_metadata_tq(mp, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

int read_num_splits(TqRig& rig) {
    int ns[2] = {0, 0};
    cudaMemcpy(ns, rig.d_num_splits, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(ns[0], 0);
    return ns[1];
}

void fill_dense_params(sm120::decode::tq_dense::TqDenseDecodeParams& p,
                       TqRig& rig, float sm_scale, int num_sm_parts) {
    std::memset(&p, 0, sizeof(p));
    p.b = 1; p.s_q = 1; p.h_q = kHQ; p.h_kv = 1;
    p.d_c = kDC; p.d_rope = kDR;
    p.sm_scale = sm_scale;
    p.q_rot = rig.d_q_rot;
    p.q_rope = rig.d_q_rope;
    p.kv_cache = rig.d_cache_v32;
    p.cache_stride_block = (int64_t)kPage * kTqRowBytes;
    p.cache_stride_row = kTqRowBytes;
    p.block_table = rig.d_block_table;
    p.block_table_batch_stride = rig.num_pages;
    p.page_block_size = kPage;
    p.seqlens_k = rig.d_seqlens;
    p.centroids = rig.d_centroids;
    p.out = rig.d_out;
    p.lse = rig.d_lse;
    p.stride_o_b = kHQ * kDC;
    p.stride_o_s_q = kHQ * kDC;
    p.stride_o_h_q = kDC;
    p.stride_lse_b = kHQ;
    p.stride_lse_s_q = kHQ;
    p.o_accum = rig.d_o_accum;
    p.lse_accum = rig.d_lse_accum;
    p.stride_o_accum_split = kHQ * kDC;
    p.stride_o_accum_s_q = kHQ * kDC;
    p.stride_o_accum_h_q = kDC;
    p.stride_lse_accum_split = kHQ;
    p.stride_lse_accum_s_q = kHQ;
    p.tile_scheduler_metadata_ptr =
        reinterpret_cast<sm120::decode::tq_dense::TqDecodingSchedMeta*>(
            rig.d_sched_meta);
    p.num_splits_ptr = rig.d_num_splits;
    p.num_sm_parts = num_sm_parts;
    p.stream = nullptr;
}

void fill_sparse_params(sm120::decode::tq_sparse::TqSparseDecodeParams& p,
                        TqRig& rig, float sm_scale) {
    std::memset(&p, 0, sizeof(p));
    p.b = 1; p.s_q = 1; p.h_q = kHQ; p.h_kv = 1;
    p.d_c = kDC; p.d_rope = kDR;
    p.sm_scale = sm_scale;
    p.q_rot = rig.d_q_rot;
    p.q_rope = rig.d_q_rope;
    p.kv_cache = rig.d_cache_v32;
    p.cache_stride_block = (int64_t)kPage * kTqRowBytes;
    p.cache_stride_row = kTqRowBytes;
    p.page_block_size = kPage;
    p.indices = rig.d_indices;
    p.topk = rig.num_tokens;
    p.stride_indices_b = rig.num_tokens;
    p.stride_indices_s_q = rig.num_tokens;
    p.centroids = rig.d_centroids;
    p.out = rig.d_out;
    p.lse = rig.d_lse;
    p.stride_o_b = kHQ * kDC;
    p.stride_o_s_q = kHQ * kDC;
    p.stride_o_h_q = kDC;
    p.stride_lse_b = kHQ;
    p.stride_lse_s_q = kHQ;
    p.stream = nullptr;
}

void fill_csa_tq_params(sm120::decode::csa_tq::CsaTqDecodeParams& p,
                        TqRig& rig, float sm_scale, int num_sm_parts) {
    std::memset(&p, 0, sizeof(p));
    p.b = 1; p.s_q = 1; p.h_q = kHQ;
    p.head_dim = kDC;
    p.qk_rope_head_dim = kDR;
    p.sm_scale = sm_scale;
    p.q_rot = rig.d_q_rot;
    p.q_rope = rig.d_q_rope;
    p.kv_cache = rig.d_cache_v4;
    p.indices = rig.d_indices;
    p.topk = rig.num_tokens;
    p.stride_indices_b = rig.num_tokens;
    p.stride_indices_s_q = rig.num_tokens;
    p.centroids = rig.d_centroids;
    p.out = rig.d_out;
    p.lse = rig.d_lse;
    p.stride_o_b = kHQ * kDC;
    p.stride_o_s_q = kHQ * kDC;
    p.stride_o_h_q = kDC;
    p.stride_lse_b = kHQ;
    p.stride_lse_s_q = kHQ;
    p.num_sm_parts = num_sm_parts;
    p.o_accum = rig.d_o_accum;
    p.lse_accum = rig.d_lse_accum;
    p.stride_o_accum_split = kHQ * kDC;
    p.stride_o_accum_s_q = kHQ * kDC;
    p.stride_o_accum_h_q = kDC;
    p.stride_lse_accum_split = kHQ;
    p.stride_lse_accum_s_q = kHQ;
    p.num_splits_ptr = rig.d_num_splits;
    p.tile_scheduler_metadata_ptr = rig.d_sched_meta;
    p.stream = nullptr;
}

void run_combine(TqRig& rig, int num_sm_parts) {
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
    lc::launch_mla_combine_f32(c, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

// Exact host-built cache values → LSE is fp32-exact (tight 2e-3, and it is the
// unit-under-test: natural-vs-log2 lse_accum shifts the combined lse by ×ln2
// ≈ -4.4 at lse≈15). The OUTPUT carries the kernel's bf16 P@V accumulation
// rounding (probabilities/V rounded to bf16 before the fp32 accumulate), ~2e-3
// absolute on 4-bit-dequant V — so the output uses a looser tolerance. Both are
// still orders of magnitude below the bugs under test: a missing split
// normalization scales the combined output by ~L (hundreds), a units error
// scales combine weights by exp2 vs exp — the discriminators trip regardless.
void check_result(TqRig& rig, const std::vector<float>& ref_out,
                  const std::vector<float>& ref_lse, const char* tag) {
    std::vector<float> out(static_cast<size_t>(kHQ) * kDC);
    std::vector<float> lse(kHQ);
    ASSERT_EQ(cudaMemcpy(out.data(), rig.d_out, out.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(lse.data(), rig.d_lse, kHQ * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    for (int h = 0; h < kHQ; ++h)
        EXPECT_NEAR(lse[h], ref_lse[h], 2e-3f) << tag << " lse[" << h << "]";
    for (size_t i = 0; i < out.size(); ++i) {
        const double got = out[i];
        const double want = ref_out[i];
        // bf16 P@V accumulation rounding on 4-bit-dequant V (see header note);
        // LSE above stays tight — this floor is still ~2 orders below the
        // ×ln2 / ×L bug magnitudes the multi-split discriminators target.
        EXPECT_NEAR(got, want, 1e-2 + 5e-3 * std::abs(want))
            << tag << " out[" << i << "]";
    }
}

}  // namespace

// ── tq_dense decode ──────────────────────────────────────────────────────────

TEST(TqLseUnits, DenseNoSplitLseNatural) {
    REQUIRES_GPU();
    TqRig rig;
    build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
              /*plant_token=*/-1, /*seed=*/101);
    std::vector<float> ref_out, ref_lse;
    cpu_reference(rig, /*sm_scale=*/1.0f, /*use_v=*/false, ref_out, ref_lse);

    sm120::decode::tq_dense::TqDenseDecodeParams p{};
    fill_dense_params(p, rig, /*sm_scale=*/1.0f, /*num_sm_parts=*/1);
    lc::launch_decode_dense_tq(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    check_result(rig, ref_out, ref_lse, "tq-dense-nosplit");
}

TEST(TqLseUnits, DenseMultiSplitCombineMatchesReference) {
    REQUIRES_GPU();
    TqRig rig;
    build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
              /*plant_token=*/2040, /*seed=*/103);
    std::vector<float> ref_out, ref_lse;
    cpu_reference(rig, /*sm_scale=*/1.0f, /*use_v=*/false, ref_out, ref_lse);

    fill_metadata(rig, /*num_sm_parts=*/8, /*topk_mode=*/false);
    ASSERT_GT(read_num_splits(rig), 1)
        << "test requires a genuine multi-split schedule";

    sm120::decode::tq_dense::TqDenseDecodeParams p{};
    fill_dense_params(p, rig, /*sm_scale=*/1.0f, /*num_sm_parts=*/8);
    lc::launch_decode_dense_tq(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    run_combine(rig, /*num_sm_parts=*/8);
    check_result(rig, ref_out, ref_lse, "tq-dense-split");
}

// ── tq_sparse decode (single-split only — no lse_accum path exists) ─────────

TEST(TqLseUnits, SparseNoSplitLseNatural) {
    REQUIRES_GPU();
    TqRig rig;
    build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
              /*plant_token=*/-1, /*seed=*/107);
    std::vector<float> ref_out, ref_lse;
    cpu_reference(rig, /*sm_scale=*/1.0f, /*use_v=*/false, ref_out, ref_lse);

    sm120::decode::tq_sparse::TqSparseDecodeParams p{};
    fill_sparse_params(p, rig, /*sm_scale=*/1.0f);
    lc::launch_decode_sparse_tq(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    check_result(rig, ref_out, ref_lse, "tq-sparse");
}

// ── csa_tq decode (V4, separate K/V TQ entries) ──────────────────────────────

TEST(TqLseUnits, CsaTqNoSplitLseNatural) {
    REQUIRES_GPU();
    TqRig rig;
    build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
              /*plant_token=*/-1, /*seed=*/109);
    std::vector<float> ref_out, ref_lse;
    cpu_reference(rig, /*sm_scale=*/1.0f, /*use_v=*/true, ref_out, ref_lse);

    sm120::decode::csa_tq::CsaTqDecodeParams p{};
    fill_csa_tq_params(p, rig, /*sm_scale=*/1.0f, /*num_sm_parts=*/1);
    sm120::decode::csa_tq::run_csa_tq_decode(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    check_result(rig, ref_out, ref_lse, "csa-tq-nosplit");
}

// THE discriminator (KVS-1 class): before the fix, csa_tq's split path wrote
// NATURAL-unit lse_accum (mla_combine weights are exp2f-based) and
// UNNORMALIZED o_accum — this test fails by orders of magnitude on both the
// combined output and the combined lse. After the fix it passes.
TEST(TqLseUnits, CsaTqMultiSplitCombineMatchesReference) {
    REQUIRES_GPU();
    TqRig rig;
    build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
              /*plant_token=*/2040, /*seed=*/113);
    std::vector<float> ref_out, ref_lse;
    cpu_reference(rig, /*sm_scale=*/1.0f, /*use_v=*/true, ref_out, ref_lse);

    fill_metadata(rig, /*num_sm_parts=*/8, /*topk_mode=*/true);
    ASSERT_GT(read_num_splits(rig), 1)
        << "test requires a genuine multi-split schedule";

    sm120::decode::csa_tq::CsaTqDecodeParams p{};
    fill_csa_tq_params(p, rig, /*sm_scale=*/1.0f, /*num_sm_parts=*/8);
    sm120::decode::csa_tq::run_csa_tq_decode(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    run_combine(rig, /*num_sm_parts=*/8);
    check_result(rig, ref_out, ref_lse, "csa-tq-split");
}
