// TD-DSP52-BATCHED-VERIFY-EQUIV (c): CPU-reference numeric validation of the
// TQ chunk-prefill attention at B>1.
//
// dsp52's batched verify measured max|dLogit| ≈ 2.0 end-to-end between the
// TQ chunk pipeline and B=1 TQ decode. The TQ chunk_causal B>1 path
// (tq_sm120_attention_device.cpp: linearize → indexed dequant → shared BF16
// phase1 prefill) had NEVER been validated against an independent numeric
// reference — goldens covered TQ only at B=1 decode, and chunk prefill only
// on the snapmla backend. This test establishes whether the chunk path is
// numerically healthy (divergence = legitimate FP/BF16 reordering between
// two different kernel pipelines) or defective (index/scale error).
//
// Method — three computations of the SAME attention over the same TQ paged
// cache (built once through the device's own k_append):
//   (a) B=1 decode: the direct-TQ sparse-decode chain (translate → q_rotate →
//       codebook-space attend → v_rotate_back), token-by-token over N tokens
//       (token t attends context [0, t]).  LS_TQ_DECODE_GRAPH=0 (ungraphed
//       host-bound route; graph replay is trajectory-bit-identical per §12n
//       and covered elsewhere).
//   (b) chunk_causal B=N: ONE dense prefill call over the staged union with
//       ascending per-row bounds seqlens_k[t] = t+1 — the dsp52 batched-verify
//       shape.
//   (b2) chunk_causal B=N sparse: same, through the sparse phase1 kernel with
//       full-coverage per-row index lists (the DSA-shaped chunk route).
//   (c) CPU float reference with TQ quant/dequant modeled EXACTLY: the
//       quantized cache bytes are read back from the GPU and dequantized on
//       the host (nibble → centroid, fp16 norm, Π^T inverse rotation in
//       double) — the reconstruction basis is bit-identical to what both GPU
//       paths consume; attention runs in double (stable softmax).
//
// Gates: (a) vs (c) and (b)/(b2) vs (c) element-wise within the per-kernel
// band |x - ref| <= 0.06 + 0.05*|ref| (the established BF16 reduction-order
// band, same as dsp52's logits-equivalence gate); (a) vs (b) within the
// doubled band (implied by both single-band passes). LSE (natural log,
// INV-LSE-NAT) is gated with the same band. Max/mean deviations are printed
// so the a-vs-b divergence magnitude is characterized in the log.
//
// Shapes: SmallDims h_q=8, N=33 (partial phase1 tile, single page, partial
// KV block); Glm52HeadGeometry h_q=32 (GLM-5.2 TP=2, the dsp52 production
// rank shape), N=96 (crosses the page-64 boundary and the 64-token KV-block
// boundary in both decode gather and prefill main loop).

#include "compute/tq_sm120_attention_device.h"
#include "compute/tq_init.h"
#include "core/attention_device.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

constexpr int kDC   = 512;                    // kv_lora_rank (codebook d)
constexpr int kDR   = 64;                     // qk_rope_head_dim
constexpr int kDQK  = kDC + kDR;
constexpr int kPage = 64;
// TQ cache row: d_c/2 packed nibbles + 2 B fp16 norm + d_rope BF16 rope.
constexpr int kRowBytes  = kDC / 2 + 2 + kDR * 2;
constexpr int kPageBytes = kPage * kRowBytes;

std::string codebook_dir() {
    return std::string(LAYERSTORM_SOURCE_DIR) + "/config/tq_codebooks";
}

float bf16_bits_to_float(uint16_t bits) {
    uint32_t f = static_cast<uint32_t>(bits) << 16;
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
}

// The per-kernel BF16 reduction-order band (dsp52 equivalence-gate band).
float band(float ref) { return 0.06f + 0.05f * std::fabs(ref); }

struct CmpStats {
    double max_abs = 0.0;    // max |got - ref|
    double max_norm = 0.0;   // max |got - ref| / band(ref)
    double sum_abs = 0.0;
    long   count = 0;
    long   violations = 0;
    long   worst_i = -1;
    float  worst_got = 0.0f, worst_ref = 0.0f;

    void add(long i, float got, float ref, float band_scale = 1.0f) {
        const double d = std::fabs(static_cast<double>(got) - ref);
        const double nb = d / (band_scale * band(ref));
        sum_abs += d;
        ++count;
        if (nb > max_norm) {
            max_norm = nb;
            worst_i = i;
            worst_got = got;
            worst_ref = ref;
        }
        max_abs = std::max(max_abs, d);
        if (nb > 1.0) ++violations;
    }
};

void report(const char* tag, const CmpStats& s, int h_q) {
    const long hd = static_cast<long>(h_q) * kDC;
    printf("[tq-chunk-equiv] %-28s max|d|=%.6f  max|d|/band=%.4f  "
           "mean|d|=%.7f  viol=%ld/%ld  worst@(t=%ld,h=%ld,d=%ld) got=%.6f "
           "ref=%.6f\n",
           tag, s.max_abs, s.max_norm, s.count ? s.sum_abs / s.count : 0.0,
           s.violations, s.count,
           s.worst_i >= 0 ? s.worst_i / hd : -1,
           s.worst_i >= 0 ? (s.worst_i % hd) / kDC : -1,
           s.worst_i >= 0 ? s.worst_i % kDC : -1, s.worst_got, s.worst_ref);
    EXPECT_EQ(s.violations, 0)
        << tag << ": " << s.violations << " elements outside the band "
        << "(max |d|/band = " << s.max_norm << ", max |d| = " << s.max_abs
        << ", worst flat index " << s.worst_i << ": got " << s.worst_got
        << " vs ref " << s.worst_ref << ")";
}

// (a) decode / (b) chunk / (b2) sparse-chunk vs the CPU float reference, plus
// (a) vs (b) directly. One TQ paged cache, one device, three GPU routes.
void run_case(int h_q, int n, unsigned seed) {
    // Ungraphed decode route (see file header). Must be set BEFORE device
    // construction — the flag is latched in the constructor. Restored right
    // after so it cannot leak into other tests in the same process.
    const char* old_graph_env = std::getenv("LS_TQ_DECODE_GRAPH");
    const std::string old_graph_val = old_graph_env ? old_graph_env : "";
    setenv("LS_TQ_DECODE_GRAPH", "0", 1);
    auto dev = lc::make_tq_sm120_attention_device(
        cfg::GpuRef{0, 0, cfg::GpuType::rtx5090});
    if (old_graph_env)
        setenv("LS_TQ_DECODE_GRAPH", old_graph_val.c_str(), 1);
    else
        unsetenv("LS_TQ_DECODE_GRAPH");
    ASSERT_NE(dev, nullptr);
    dev->set_device();

    lc::TqInitOptions opts;
    opts.d_c = kDC;
    opts.bits = 4;
    opts.num_layers = 1;
    opts.codebook_dir = codebook_dir();
    opts.attention_device = dev.get();
    auto res = lc::init_tq_resources(opts);
    ASSERT_NE(res, nullptr);

    lc::tq_device_set_resources(dev.get(), res.get());
    lc::tq_device_set_model_dims(dev.get(), /*batch_size=*/n, kDC, kDR, h_q,
                                 /*s_q=*/1, /*sm_scale=*/0.0f);
    lc::tq_device_set_prefill_scratch(dev.get(), n);

    const int num_pages = (n + kPage - 1) / kPage;
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    // ── Inputs ─────────────────────────────────────────────────────────────
    std::vector<__nv_bfloat16> c_kv(static_cast<size_t>(n) * kDC);
    std::vector<__nv_bfloat16> k_rope(static_cast<size_t>(n) * kDR);
    // Interleaved absorbed query [n, h_q, d_c + d_rope] — the layout BOTH
    // paths consume (chunk row t == decode token t's query).
    std::vector<__nv_bfloat16> q(static_cast<size_t>(n) * h_q * kDQK);
    for (auto& v : c_kv) v = __float2bfloat16(dist(rng));
    for (auto& v : k_rope) v = __float2bfloat16(dist(rng));
    for (auto& v : q) v = __float2bfloat16(dist(rng));
    std::vector<int> slots(n);
    std::iota(slots.begin(), slots.end(), 0);

    // ── Build the TQ paged cache through the device's own k_append ─────────
    void *d_ckv = nullptr, *d_rope = nullptr, *d_cache = nullptr;
    int* d_slots = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ckv, c_kv.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_rope, k_rope.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_cache,
                          static_cast<size_t>(num_pages) * kPageBytes));
    CUDA_CHECK(cudaMalloc(&d_slots, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ckv, c_kv.data(), c_kv.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rope, k_rope.data(), k_rope.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, slots.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_cache, 0,
                          static_cast<size_t>(num_pages) * kPageBytes));
    dev->k_append(d_ckv, d_rope, d_cache, kPageBytes, kRowBytes, d_slots, n,
                  kDC, kDR, /*c_kv_row_stride=*/kDC, /*k_rope_row_stride=*/kDR,
                  kPage, /*layer_idx=*/0, /*stream=*/nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── CPU exact TQ dequant from the quantized cache bytes ────────────────
    // Reads the GPU-quantized rows back so the reconstruction basis (codes +
    // fp16 norm + BF16 rope) is bit-identical to what both GPU paths consume;
    // the Π^T inverse rotation runs in double (no BF16 staging round — this
    // is the float reference, the band absorbs the GPU BF16 staging).
    std::vector<uint8_t> cache_h(static_cast<size_t>(num_pages) * kPageBytes);
    CUDA_CHECK(cudaMemcpy(cache_h.data(), d_cache, cache_h.size(),
                          cudaMemcpyDeviceToHost));
    const std::vector<float>& Pi = res->rotations[0].Pi;      // host copy
    const std::vector<float>& cent = res->codebook.centroids; // host copy
    ASSERT_EQ(static_cast<int>(Pi.size()), kDC * kDC);
    ASSERT_EQ(static_cast<int>(cent.size()), 16);
    std::vector<double> khat(static_cast<size_t>(n) * kDC);   // Π^T ŷ · norm
    std::vector<double> krope_f(static_cast<size_t>(n) * kDR);
    std::vector<double> yhat(kDC);
    for (int t = 0; t < n; ++t) {
        const uint8_t* row = cache_h.data()
            + static_cast<size_t>(t / kPage) * kPageBytes
            + static_cast<size_t>(t % kPage) * kRowBytes;
        for (int b = 0; b < kDC / 2; ++b) {
            yhat[2 * b]     = cent[row[b] & 0x0F];
            yhat[2 * b + 1] = cent[(row[b] >> 4) & 0x0F];
        }
        __half norm_h;
        std::memcpy(&norm_h, row + kDC / 2, sizeof(norm_h));
        const double norm = __half2float(norm_h);
        for (int j = 0; j < kDC; ++j) {
            double acc = 0.0;
            for (int i = 0; i < kDC; ++i)
                acc += yhat[i] * Pi[static_cast<size_t>(i) * kDC + j];
            khat[static_cast<size_t>(t) * kDC + j] = acc * norm;
        }
        for (int d = 0; d < kDR; ++d) {
            uint16_t bits;
            std::memcpy(&bits, row + kDC / 2 + 2 + 2 * d, sizeof(bits));
            krope_f[static_cast<size_t>(t) * kDR + d] = bf16_bits_to_float(bits);
        }
    }

    // ── (c) CPU float reference attention (double softmax) ─────────────────
    std::vector<float> qf(q.size());
    for (size_t i = 0; i < q.size(); ++i) qf[i] = __bfloat162float(q[i]);
    std::vector<float> ref_out(static_cast<size_t>(n) * h_q * kDC);
    std::vector<float> ref_lse(static_cast<size_t>(n) * h_q);
    std::vector<double> scores(n), weights(n);
    for (int t = 0; t < n; ++t) {
        const int ctx = t + 1;                       // causal: [0, t]
        for (int h = 0; h < h_q; ++h) {
            const float* qn = qf.data()
                + (static_cast<size_t>(t) * h_q + h) * kDQK;
            const float* qr = qn + kDC;
            for (int j = 0; j < ctx; ++j) {
                double s = 0.0;
                const double* kj = khat.data() + static_cast<size_t>(j) * kDC;
                for (int d = 0; d < kDC; ++d) s += qn[d] * kj[d];
                const double* rj = krope_f.data()
                    + static_cast<size_t>(j) * kDR;
                for (int d = 0; d < kDR; ++d) s += qr[d] * rj[d];
                scores[j] = s * sm_scale;
            }
            const double m = *std::max_element(scores.begin(),
                                               scores.begin() + ctx);
            double l = 0.0;
            for (int j = 0; j < ctx; ++j) {
                weights[j] = std::exp(scores[j] - m);
                l += weights[j];
            }
            ref_lse[static_cast<size_t>(t) * h_q + h] =
                static_cast<float>(m + std::log(l));   // natural (INV-LSE-NAT)
            float* oh = ref_out.data()
                + (static_cast<size_t>(t) * h_q + h) * kDC;
            for (int d = 0; d < kDC; ++d) {
                double acc = 0.0;
                for (int j = 0; j < ctx; ++j)
                    acc += weights[j] * khat[static_cast<size_t>(j) * kDC + d];
                oh[d] = static_cast<float>(acc / l);
            }
        }
    }

    // ── Shared GPU buffers ─────────────────────────────────────────────────
    void* d_q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, q.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_q, q.data(), q.size() * 2,
                          cudaMemcpyHostToDevice));

    const size_t out_elems = static_cast<size_t>(n) * h_q * kDC;
    const size_t lse_elems = static_cast<size_t>(n) * h_q;
    auto alloc_out = [&](void** o, float** l) {
        CUDA_CHECK(cudaMalloc(o, out_elems * 2));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(l),
                              lse_elems * sizeof(float)));
        CUDA_CHECK(cudaMemset(*o, 0xA5, out_elems * 2));
        CUDA_CHECK(cudaMemset(*l, 0xA5, lse_elems * sizeof(float)));
    };
    struct HostOut {
        std::vector<float> out, lse;
    };
    auto read_out = [&](const void* d_out, const float* d_lse, HostOut& r) {
        std::vector<__nv_bfloat16> raw(out_elems);
        CUDA_CHECK(cudaMemcpy(raw.data(), d_out, out_elems * 2,
                              cudaMemcpyDeviceToHost));
        r.out.resize(out_elems);
        for (size_t i = 0; i < out_elems; ++i)
            r.out[i] = __bfloat162float(raw[i]);
        r.lse.resize(lse_elems);
        CUDA_CHECK(cudaMemcpy(r.lse.data(), d_lse,
                              lse_elems * sizeof(float),
                              cudaMemcpyDeviceToHost));
        // Poison check: every lse must have been written.
        for (float v : r.lse) {
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            ASSERT_NE(bits, 0xA5A5A5A5u) << "unwritten (poisoned) lse";
        }
    };

    // ── (a) B=1 decode, token-by-token ─────────────────────────────────────
    // Block table [num_pages] (single row), full position list [0, n) with
    // per-token topk_length = t+1 and seq_len_kv = t+1: the translate kernel
    // masks positions ≥ bound to -1, so token t attends exactly [0, t].
    void* d_out_a = nullptr;
    float* d_lse_a = nullptr;
    alloc_out(&d_out_a, &d_lse_a);
    {
        std::vector<int> bt_dec(num_pages);
        std::iota(bt_dec.begin(), bt_dec.end(), 0);
        std::vector<int> pos(n);
        std::iota(pos.begin(), pos.end(), 0);
        int *d_bt = nullptr, *d_seq = nullptr, *d_idx = nullptr,
            *d_tkl = nullptr;
        CUDA_CHECK(cudaMalloc(&d_bt, num_pages * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_seq, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_tkl, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_bt, bt_dec.data(), num_pages * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_idx, pos.data(), n * sizeof(int),
                              cudaMemcpyHostToDevice));
        for (int t = 0; t < n; ++t) {
            const int bound = t + 1;
            CUDA_CHECK(cudaMemcpy(d_seq, &bound, sizeof(int),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tkl, &bound, sizeof(int),
                                  cudaMemcpyHostToDevice));
            dev->prefill_attention(
                static_cast<const __nv_bfloat16*>(d_q)
                    + static_cast<size_t>(t) * h_q * kDQK,
                /*batch_size=*/1, /*seq_len_kv=*/bound,
                d_seq, d_bt, /*max_blocks_per_seq=*/num_pages,
                d_cache, kPageBytes, kRowBytes, kPage,
                /*is_sparse=*/true, /*chunk_causal=*/false,
                d_idx, d_tkl, /*topk=*/n,
                static_cast<__nv_bfloat16*>(d_out_a)
                    + static_cast<size_t>(t) * h_q * kDC,
                d_lse_a + static_cast<size_t>(t) * h_q,
                /*layer_idx=*/0, /*stream=*/nullptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_bt));
        CUDA_CHECK(cudaFree(d_seq));
        CUDA_CHECK(cudaFree(d_idx));
        CUDA_CHECK(cudaFree(d_tkl));
    }

    // ── (b) chunk_causal B=n dense + (b2) sparse — one call each ───────────
    // Per-row block tables (identical rows), ascending bounds [1..n]; the
    // device stages the union prefix once via the LAST row (chunk contract).
    std::vector<int> bt(static_cast<size_t>(n) * num_pages);
    for (int b = 0; b < n; ++b)
        for (int p = 0; p < num_pages; ++p) bt[b * num_pages + p] = p;
    std::vector<int> bounds(n);
    std::iota(bounds.begin(), bounds.end(), 1);
    int *d_bt = nullptr, *d_bounds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bt, bt.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bounds, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_bt, bt.data(), bt.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bounds, bounds.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));

    void* d_out_b = nullptr;
    float* d_lse_b = nullptr;
    alloc_out(&d_out_b, &d_lse_b);
    dev->prefill_attention(
        d_q, /*batch_size=*/n, /*seq_len_kv=*/n,
        d_bounds, d_bt, /*max_blocks_per_seq=*/num_pages,
        d_cache, kPageBytes, kRowBytes, kPage,
        /*is_sparse=*/false, /*chunk_causal=*/true,
        nullptr, nullptr, /*topk=*/0,
        d_out_b, d_lse_b, /*layer_idx=*/0, /*stream=*/nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    void* d_out_b2 = nullptr;
    float* d_lse_b2 = nullptr;
    alloc_out(&d_out_b2, &d_lse_b2);
    {
        // Full-coverage per-row selections: row b selects [0, b] (== dense).
        std::vector<int> idx2(static_cast<size_t>(n) * n, -1);
        std::vector<int> tkl2(n);
        for (int b = 0; b < n; ++b) {
            tkl2[b] = b + 1;
            for (int j = 0; j <= b; ++j) idx2[static_cast<size_t>(b) * n + j] = j;
        }
        int *d_idx2 = nullptr, *d_tkl2 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_idx2, idx2.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_tkl2, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_idx2, idx2.data(), idx2.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tkl2, tkl2.data(), n * sizeof(int),
                              cudaMemcpyHostToDevice));
        dev->prefill_attention(
            d_q, /*batch_size=*/n, /*seq_len_kv=*/n,
            d_bounds, d_bt, /*max_blocks_per_seq=*/num_pages,
            d_cache, kPageBytes, kRowBytes, kPage,
            /*is_sparse=*/true, /*chunk_causal=*/true,
            d_idx2, d_tkl2, /*topk=*/n,
            d_out_b2, d_lse_b2, /*layer_idx=*/0, /*stream=*/nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_idx2));
        CUDA_CHECK(cudaFree(d_tkl2));
    }

    // ── Read back + compare ────────────────────────────────────────────────
    HostOut a, b, b2;
    read_out(d_out_a, d_lse_a, a);
    read_out(d_out_b, d_lse_b, b);
    read_out(d_out_b2, d_lse_b2, b2);

    auto vs_ref = [&](const char* tag, const HostOut& g) {
        CmpStats so, sl;
        for (size_t i = 0; i < out_elems; ++i)
            so.add(static_cast<long>(i), g.out[i], ref_out[i]);
        for (size_t i = 0; i < lse_elems; ++i)
            sl.add(static_cast<long>(i) * kDC, g.lse[i], ref_lse[i]);
        report((std::string(tag) + " out vs cpu-ref").c_str(), so, h_q);
        report((std::string(tag) + " lse vs cpu-ref").c_str(), sl, h_q);
    };
    vs_ref("(a) decode", a);
    vs_ref("(b) chunk dense", b);
    vs_ref("(b2) chunk sparse", b2);

    // (a) vs (b) directly: both are in-band vs the reference, so their
    // mutual distance is bounded by the DOUBLED band; report the actual
    // magnitude (this is the number that contextualizes dsp52's dLogit).
    {
        CmpStats so, sl;
        for (size_t i = 0; i < out_elems; ++i)
            so.add(static_cast<long>(i), a.out[i], b.out[i], 2.0f);
        for (size_t i = 0; i < lse_elems; ++i)
            sl.add(static_cast<long>(i) * kDC, a.lse[i], b.lse[i], 2.0f);
        report("(a) vs (b) out [2x band]", so, h_q);
        report("(a) vs (b) lse [2x band]", sl, h_q);
    }

    CUDA_CHECK(cudaFree(d_ckv));
    CUDA_CHECK(cudaFree(d_rope));
    CUDA_CHECK(cudaFree(d_cache));
    CUDA_CHECK(cudaFree(d_slots));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_bt));
    CUDA_CHECK(cudaFree(d_bounds));
    CUDA_CHECK(cudaFree(d_out_a));
    CUDA_CHECK(cudaFree(d_lse_a));
    CUDA_CHECK(cudaFree(d_out_b));
    CUDA_CHECK(cudaFree(d_lse_b));
    CUDA_CHECK(cudaFree(d_out_b2));
    CUDA_CHECK(cudaFree(d_lse_b2));

    lc::destroy_tq_resources(*res, *dev);
}

}  // namespace

TEST(TqChunkPrefillEquiv, SmallDimsDecodeVsChunkVsCpuRef) {
    REQUIRES_GPU();
    run_case(/*h_q=*/8, /*n=*/33, /*seed=*/20260801u);
}

TEST(TqChunkPrefillEquiv, Glm52HeadGeometryDecodeVsChunkVsCpuRef) {
    REQUIRES_GPU();
    run_case(/*h_q=*/32, /*n=*/96, /*seed=*/20260802u);
}

#endif  // LAYERSTORM_SOURCE_DIR
