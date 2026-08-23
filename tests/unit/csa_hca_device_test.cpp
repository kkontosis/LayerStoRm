// Device-level composition tests for CsaHcaSm120AttentionDevice (V4-5a/5d/5e).
//
// The individual kernels are golden-tested in v4_prep_kernel_test /
// mhc_kernel_test / the deps suites; these tests validate the DEVICE glue:
// entry/staging layouts, slot math, split-KV metadata + combine, the
// prefill-as-decode per-row-window contract, the graph runner arm, and the
// sinks + inverse-rope epilogue composition — against a CPU double reference
// that parses the actual cache bytes (so FP8 storage error cancels; the
// remaining tolerance is the kernel's FP8 P/V quantization of the softmax).

#include "compute/csa_hca_sm120_attention_device.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "compute/kernels/attention/v4_prep.h"
#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: "                         \
                                     << cudaGetErrorString(_err);              \
    } while (0)

namespace {

constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kEntryBytes = kHeadDim + 4 + kRopeDim * 2 + kHeadDim + 4;
constexpr int kHq = 64;
constexpr int kWindow = 128;
constexpr float kSmScale = 0.044194173824159216f;  // 1/sqrt(512)

float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& v) {
    std::vector<__nv_bfloat16> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = __float2bfloat16_rn(v[i]);
    return out;
}

std::vector<float> make_cos_sin(int max_pos, double theta) {
    const int half = kRopeDim / 2;
    std::vector<float> t(size_t(max_pos) * kRopeDim);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double f = std::pow(theta, -2.0 * i / kRopeDim);
            t[size_t(p) * kRopeDim + i] = float(std::cos(p * f));
            t[size_t(p) * kRopeDim + half + i] = float(std::sin(p * f));
        }
    }
    return t;
}

struct DeviceBuf {
    void* p = nullptr;
    explicit DeviceBuf(size_t bytes) { cudaMalloc(&p, bytes); }
    ~DeviceBuf() { cudaFree(p); }
};

// Dequantized entry view (score K = k_nope 512 + k_rope 64, V = v 512).
struct Entry {
    std::vector<float> k = std::vector<float>(kHeadDim);
    std::vector<float> kr = std::vector<float>(kRopeDim);
    std::vector<float> v = std::vector<float>(kHeadDim);
};
Entry parse_entry(const uint8_t* e) {
    Entry out;
    const float ks = *reinterpret_cast<const float*>(e + kHeadDim);
    const float vs = *reinterpret_cast<const float*>(e + kEntryBytes - 4);
    const auto* k8 = reinterpret_cast<const __nv_fp8_e4m3*>(e);
    const auto* v8 = reinterpret_cast<const __nv_fp8_e4m3*>(
        e + kHeadDim + 4 + kRopeDim * 2);
    const auto* kr = reinterpret_cast<const __nv_bfloat16*>(e + kHeadDim + 4);
    for (int i = 0; i < kHeadDim; ++i) {
        out.k[i] = float(k8[i]) * ks;
        out.v[i] = float(v8[i]) * vs;
    }
    for (int i = 0; i < kRopeDim; ++i) out.kr[i] = __bfloat162float(kr[i]);
    return out;
}

// CPU double-reference attention over a set of dequantized entries, then
// sinks + inverse rope (interleaved) at `pos`.
void ref_attention(const std::vector<float>& q_nope,  // [h, 512]
                   const std::vector<float>& q_rope,  // [h, 64]
                   const std::vector<Entry>& entries,
                   const std::vector<float>& sinks,   // [h] (empty → skip)
                   const std::vector<float>& tab, int pos,
                   std::vector<double>& out,          // [h, 512]
                   std::vector<double>& lse) {        // [h]
    out.assign(size_t(kHq) * kHeadDim, 0.0);
    lse.assign(kHq, 0.0);
    for (int h = 0; h < kHq; ++h) {
        std::vector<double> scores(entries.size());
        double m = -1e300;
        for (size_t e = 0; e < entries.size(); ++e) {
            double s = 0;
            for (int i = 0; i < kHeadDim; ++i)
                s += double(q_nope[size_t(h) * kHeadDim + i]) * entries[e].k[i];
            for (int i = 0; i < kRopeDim; ++i)
                s += double(q_rope[size_t(h) * kRopeDim + i]) * entries[e].kr[i];
            scores[e] = s * kSmScale;
            m = std::max(m, scores[e]);
        }
        double sum = 0;
        for (size_t e = 0; e < entries.size(); ++e) {
            scores[e] = std::exp(scores[e] - m);
            sum += scores[e];
        }
        for (size_t e = 0; e < entries.size(); ++e) {
            const double p = scores[e] / sum;
            for (int i = 0; i < kHeadDim; ++i)
                out[size_t(h) * kHeadDim + i] += p * entries[e].v[i];
        }
        lse[h] = m + std::log(sum);
        if (!sinks.empty()) {
            const double s = sinks[h];
            const double factor = 1.0 / (1.0 + std::exp(s - lse[h]));
            for (int i = 0; i < kHeadDim; ++i)
                out[size_t(h) * kHeadDim + i] *= factor;
            const double mx = std::max(lse[h], (double)s);
            lse[h] = mx + std::log1p(std::exp(std::min(lse[h], (double)s) - mx));
        }
        // Inverse rope on out[448:512].
        const int nope = kHeadDim - kRopeDim;
        const int half = kRopeDim / 2;
        for (int i = 0; i < half; ++i) {
            const double c = tab[size_t(pos) * kRopeDim + i];
            const double s = tab[size_t(pos) * kRopeDim + half + i];
            const double e0 = out[size_t(h) * kHeadDim + nope + 2 * i];
            const double o0 = out[size_t(h) * kHeadDim + nope + 2 * i + 1];
            out[size_t(h) * kHeadDim + nope + 2 * i] = e0 * c + o0 * s;
            out[size_t(h) * kHeadDim + nope + 2 * i + 1] = -e0 * s + o0 * c;
        }
    }
}

struct Fixture {
    std::unique_ptr<lc::AttentionDevice> dev;

    // Host-mirrored caches.
    std::vector<uint8_t> csa_cache_host;
    std::vector<uint8_t> swa_cache_host;
    std::unique_ptr<DeviceBuf> d_csa, d_swa, d_tab;
    std::vector<float> tab;
    int num_blocks = 0;
    int T = 0;
    bool det_reduce = false;  ///< set BEFORE build() (ticket J DET-REDUCE)
    int table_pos = 0;        ///< set BEFORE build(): min rope-table rows
                              ///< (compress ropes block j at pos 4j+3 —
                              ///< tests reaching past seq_len+8 need more)

    void build(uint32_t seed, int seq_len, int blocks) {
        T = seq_len;
        num_blocks = blocks;
        dev = lc::make_csa_hca_sm120_attention_device(
            layerstorm::config::GpuRef{.position = 0, .id = 0,
                                       .type = layerstorm::config::GpuType::rtx5090});
        dev->set_device();

        lc::V4DeviceOptions o;
        o.max_batch = 4;
        o.max_attn_rows = 64;  // SC: covers the batched-prefill device test
        o.h_q = kHq;
        o.sm_scale = kSmScale;
        o.topk = 64;
        o.sliding_window = kWindow;
        o.num_sm_parts = 4;
        o.csa_entries_per_page = 64;
        o.hca_entries_per_page = 2;
        o.swa_page_tokens = kWindow;
        o.idx_entries_per_page = 4;
        o.idx_page_bytes = 4 * (128 + 4);
        o.index_n_heads = 4;
        o.index_head_dim = 128;
        o.max_index_blocks = 64;
        o.deterministic_reduce = det_reduce;  // ticket J: DET-REDUCE gate
        lc::csa_hca_device_configure(dev.get(), o);
        lc::csa_hca_device_set_scratch(dev.get(), 1024);

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        tab = make_cos_sin(std::max(seq_len + 8, table_pos), 160000.0);
        d_tab = std::make_unique<DeviceBuf>(tab.size() * 4);
        CUDA_CHECK(cudaMemcpy(d_tab->p, tab.data(), tab.size() * 4,
                              cudaMemcpyHostToDevice));

        // Raw SWA ring page: append T tokens at slot pos % 128; the ring
        // holds exactly the last 128 (order-invariant keys).
        std::vector<float> kv(size_t(T) * kHeadDim);
        for (auto& v : kv) v = bf16_round(dist(rng) * 0.5f);
        auto kv_bf = to_bf16(kv);
        std::vector<int> pos(T), slots(T);
        for (int t = 0; t < T; ++t) {
            pos[t] = t;
            slots[t] = t % kWindow;
        }
        d_swa = std::make_unique<DeviceBuf>(size_t(kWindow) * kEntryBytes);
        DeviceBuf d_kv(kv_bf.size() * 2), d_pos(T * 4), d_slots(T * 4);
        CUDA_CHECK(cudaMemcpy(d_kv.p, kv_bf.data(), kv_bf.size() * 2,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos.p, pos.data(), T * 4,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_slots.p, slots.data(), T * 4,
                              cudaMemcpyHostToDevice));
        lc::launch_v4_raw_kv_append(d_kv.p, static_cast<const int*>(d_pos.p),
                                    static_cast<const int*>(d_slots.p),
                                    d_swa->p, d_tab->p, T, kHeadDim, kRopeDim,
                                    nullptr);

        // Compressed CSA entries via the model-faithful compress kernel.
        const int stride = 4;
        const int state_dim = 2 * kHeadDim;
        const int st_T = num_blocks * stride;
        std::vector<float> kv_state(size_t(st_T) * state_dim),
            score_state(size_t(st_T) * state_dim),
            ape(size_t(stride) * state_dim), norm_w(kHeadDim, 1.0f);
        for (auto& v : kv_state) v = bf16_round(dist(rng) * 0.5f);
        for (auto& v : score_state) v = bf16_round(dist(rng) * 0.5f);
        for (auto& v : ape) v = dist(rng) * 0.25f;
        auto kvs_bf = to_bf16(kv_state);
        auto scs_bf = to_bf16(score_state);
        std::vector<int> bslots(num_blocks);
        for (int b = 0; b < num_blocks; ++b) bslots[b] = b;
        d_csa = std::make_unique<DeviceBuf>(
            size_t(num_blocks) * kEntryBytes);
        DeviceBuf d_kvs(kvs_bf.size() * 2), d_scs(scs_bf.size() * 2),
            d_ape(ape.size() * 4), d_norm(norm_w.size() * 4),
            d_bslots(size_t(num_blocks) * 4);
        CUDA_CHECK(cudaMemcpy(d_kvs.p, kvs_bf.data(), kvs_bf.size() * 2,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scs.p, scs_bf.data(), scs_bf.size() * 2,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ape.p, ape.data(), ape.size() * 4,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_norm.p, norm_w.data(), norm_w.size() * 4,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bslots.p, bslots.data(),
                              size_t(num_blocks) * 4,
                              cudaMemcpyHostToDevice));
        lc::V4CompressArgs ca;
        ca.kv_state = d_kvs.p;
        ca.score_state = d_scs.p;
        ca.ring_capacity = st_T;
        ca.state_dim = state_dim;
        ca.overlap = true;
        ca.stride = stride;
        ca.ape = d_ape.p;
        ca.norm_w = d_norm.p;
        ca.cos_sin = d_tab->p;
        ca.rms_eps = 1e-6f;
        ca.D = kHeadDim;
        ca.rope_dim = kRopeDim;
        ca.first_block = 0;
        ca.num_blocks = num_blocks;
        ca.slots = static_cast<const int*>(d_bslots.p);
        ca.out_mode = lc::V4CompressArgs::Out::kFp8Entry;
        ca.kv_cache = d_csa->p;
        lc::launch_v4_compress_insert(ca, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        csa_cache_host.resize(size_t(num_blocks) * kEntryBytes);
        swa_cache_host.resize(size_t(kWindow) * kEntryBytes);
        CUDA_CHECK(cudaMemcpy(csa_cache_host.data(), d_csa->p,
                              csa_cache_host.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(swa_cache_host.data(), d_swa->p,
                              swa_cache_host.size(), cudaMemcpyDeviceToHost));
    }

    // Entries a query at `qpos` sees, given comp indices (into csa cache
    // slots) and its raw window rows.
    std::vector<Entry> visible_entries(const std::vector<int>& comp_slots,
                                       int swa_len,
                                       const std::vector<int>& swa_rows) {
        std::vector<Entry> es;
        for (int s : comp_slots) {
            if (s < 0) continue;
            es.push_back(
                parse_entry(csa_cache_host.data() + size_t(s) * kEntryBytes));
        }
        for (int i = 0; i < swa_len; ++i) {
            es.push_back(parse_entry(swa_cache_host.data() +
                                     size_t(swa_rows[i]) * kEntryBytes));
        }
        return es;
    }
};

void expect_close(const std::vector<__nv_bfloat16>& got_out,
                  const std::vector<float>& got_lse,
                  const std::vector<double>& ref_out,
                  const std::vector<double>& ref_lse, int rows,
                  const char* tag) {
    // The kernel quantizes Q per head to FP8 before the QK MMA — scores
    // carry ~1-2% noise vs the exact-BF16 reference, so LSE tolerance is
    // 8e-2 (still pins scores/masking/indices/layout: an entry wrongly
    // included/excluded shifts LSE by O(exp weight), far above this).
    for (int h = 0; h < rows * kHq; ++h) {
        EXPECT_NEAR(got_lse[h], ref_lse[h], 8e-2) << tag << " lse h=" << h;
    }
    // FP8 P/V quantization inside the kernel: compare with cosine + loose
    // elementwise bounds.
    double dot = 0, na = 0, nb = 0, max_abs = 0;
    for (size_t i = 0; i < got_out.size(); ++i) {
        const double a = __bfloat162float(got_out[i]);
        const double b = ref_out[i];
        dot += a * b;
        na += a * a;
        nb += b * b;
        max_abs = std::max(max_abs, std::abs(a - b));
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    // FP8 P/V quantization noise floor: the deps V4K-10a validation gate
    // itself measured attn cos ~0.994 vs BF16 reference. LSE (pre-PV-quant)
    // is held tight above — it pins scores/softmax/indices/layout exactly.
    EXPECT_GT(cos, 0.99) << tag << " cosine";
    EXPECT_LT(max_abs, 0.12) << tag << " max abs err";
}

}  // namespace

TEST(CsaHcaDevice, DecodeCsaPlusSwaMatchesReference) {
    REQUIRES_GPU();
    Fixture f;
    f.build(1234, /*seq_len=*/200, /*blocks=*/50);

    std::mt19937 rng(55);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> qn(size_t(kHq) * kHeadDim, 0.0f),
        qr(size_t(kHq) * kRopeDim);
    for (int h = 0; h < kHq; ++h)
        for (int i = 0; i < kHeadDim - kRopeDim; ++i)
            qn[size_t(h) * kHeadDim + i] = bf16_round(dist(rng) * 0.3f);
    for (auto& v : qr) v = bf16_round(dist(rng) * 0.3f);
    std::vector<float> sinks(kHq);
    for (auto& v : sinks) v = dist(rng);

    // topk=64: 50 real blocks + 14 × -1 padding.
    std::vector<int> indices(64, -1);
    for (int b = 0; b < 50; ++b) indices[b] = b;
    const int qpos = f.T - 1;
    std::vector<int> swa_bt = {0};
    std::vector<int> swa_sl = {kWindow};
    std::vector<int> positions = {qpos};

    auto qn_bf = to_bf16(qn);
    auto qr_bf = to_bf16(qr);
    DeviceBuf d_qn(qn_bf.size() * 2), d_qr(qr_bf.size() * 2),
        d_idx(64 * 4), d_bt(4), d_sl(4), d_pos(4), d_sinks(kHq * 4),
        d_out(size_t(kHq) * kHeadDim * 2), d_lse(kHq * 4);
    CUDA_CHECK(cudaMemcpy(d_qn.p, qn_bf.data(), qn_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qr.p, qr_bf.data(), qr_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx.p, indices.data(), 64 * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt.p, swa_bt.data(), 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sl.p, swa_sl.data(), 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.p, positions.data(), 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sinks.p, sinks.data(), kHq * 4,
                          cudaMemcpyHostToDevice));

    lc::V4AttentionArgs a;
    a.rows = 1;
    a.q_nope = d_qn.p;
    a.q_rope = d_qr.p;
    a.comp_cache = f.d_csa->p;
    a.sparse_indices = static_cast<const int*>(d_idx.p);
    a.topk = 64;
    a.swa_cache = f.d_swa->p;
    a.swa_block_table = static_cast<const int*>(d_bt.p);
    a.swa_block_table_stride = 1;
    a.swa_seqlens = static_cast<const int*>(d_sl.p);
    a.sinks = d_sinks.p;
    a.positions = static_cast<const int*>(d_pos.p);
    a.rope_table = f.d_tab->p;
    a.out = d_out.p;
    a.lse = static_cast<float*>(d_lse.p);
    lc::csa_hca_device_attention(f.dev.get(), a);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out(size_t(kHq) * kHeadDim);
    std::vector<float> lse(kHq);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out.p, out.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lse.data(), d_lse.p, kHq * 4,
                          cudaMemcpyDeviceToHost));

    std::vector<int> swa_rows(kWindow);
    for (int i = 0; i < kWindow; ++i) swa_rows[i] = i;
    auto entries = f.visible_entries(indices, kWindow, swa_rows);
    std::vector<double> ref_out, ref_lse;
    ref_attention(qn, qr, entries, sinks, f.tab, qpos, ref_out, ref_lse);
    expect_close(out, lse, ref_out, ref_lse, 1, "decode");
}

// Ticket J (V4 decode nondeterminism): with deterministic_reduce the CSA
// decode's softmax denominator uses the fixed-order cross-warp combine
// (dense_fp8 DET-REDUCE port) instead of arrival-order atomicAdd on sL.
// Guards (a) the flag plumbing V4DeviceOptions -> CsaFp8DecodeParams ->
// kernel template, (b) numerical agreement with the reference, and (c)
// run-to-run bit-stability of out AND lse through the deterministic path.
TEST(CsaHcaDevice, DecodeDetReduceBitStableAndMatchesReference) {
    REQUIRES_GPU();
    Fixture f;
    f.det_reduce = true;
    f.build(1234, /*seq_len=*/200, /*blocks=*/50);

    std::mt19937 rng(55);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> qn(size_t(kHq) * kHeadDim, 0.0f),
        qr(size_t(kHq) * kRopeDim);
    for (int h = 0; h < kHq; ++h)
        for (int i = 0; i < kHeadDim - kRopeDim; ++i)
            qn[size_t(h) * kHeadDim + i] = bf16_round(dist(rng) * 0.3f);
    for (auto& v : qr) v = bf16_round(dist(rng) * 0.3f);
    std::vector<float> sinks(kHq);
    for (auto& v : sinks) v = dist(rng);

    std::vector<int> indices(64, -1);
    for (int b = 0; b < 50; ++b) indices[b] = b;
    const int qpos = f.T - 1;
    std::vector<int> swa_bt = {0};
    std::vector<int> swa_sl = {kWindow};
    std::vector<int> positions = {qpos};

    auto qn_bf = to_bf16(qn);
    auto qr_bf = to_bf16(qr);
    DeviceBuf d_qn(qn_bf.size() * 2), d_qr(qr_bf.size() * 2),
        d_idx(64 * 4), d_bt(4), d_sl(4), d_pos(4), d_sinks(kHq * 4),
        d_out(size_t(kHq) * kHeadDim * 2), d_lse(kHq * 4);
    CUDA_CHECK(cudaMemcpy(d_qn.p, qn_bf.data(), qn_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qr.p, qr_bf.data(), qr_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx.p, indices.data(), 64 * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt.p, swa_bt.data(), 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sl.p, swa_sl.data(), 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.p, positions.data(), 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sinks.p, sinks.data(), kHq * 4,
                          cudaMemcpyHostToDevice));

    lc::V4AttentionArgs a;
    a.rows = 1;
    a.q_nope = d_qn.p;
    a.q_rope = d_qr.p;
    a.comp_cache = f.d_csa->p;
    a.sparse_indices = static_cast<const int*>(d_idx.p);
    a.topk = 64;
    a.swa_cache = f.d_swa->p;
    a.swa_block_table = static_cast<const int*>(d_bt.p);
    a.swa_block_table_stride = 1;
    a.swa_seqlens = static_cast<const int*>(d_sl.p);
    a.sinks = d_sinks.p;
    a.positions = static_cast<const int*>(d_pos.p);
    a.rope_table = f.d_tab->p;
    a.out = d_out.p;
    a.lse = static_cast<float*>(d_lse.p);

    std::vector<__nv_bfloat16> out0(size_t(kHq) * kHeadDim);
    std::vector<float> lse0(kHq);
    for (int rep = 0; rep < 3; ++rep) {
        CUDA_CHECK(cudaMemset(d_out.p, 0xA5, out0.size() * 2));
        CUDA_CHECK(cudaMemset(d_lse.p, 0xA5, kHq * 4));
        lc::csa_hca_device_attention(f.dev.get(), a);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> out(size_t(kHq) * kHeadDim);
        std::vector<float> lse(kHq);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out.p, out.size() * 2,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lse.data(), d_lse.p, kHq * 4,
                              cudaMemcpyDeviceToHost));
        if (rep == 0) {
            out0 = out;
            lse0 = lse;
            std::vector<int> swa_rows(kWindow);
            for (int i = 0; i < kWindow; ++i) swa_rows[i] = i;
            auto entries = f.visible_entries(indices, kWindow, swa_rows);
            std::vector<double> ref_out, ref_lse;
            ref_attention(qn, qr, entries, sinks, f.tab, qpos, ref_out,
                          ref_lse);
            expect_close(out, lse, ref_out, ref_lse, 1, "det decode");
        } else {
            ASSERT_EQ(std::memcmp(out.data(), out0.data(), out.size() * 2),
                      0) << "det-reduce out not bit-stable (rep " << rep
                         << ")";
            ASSERT_EQ(std::memcmp(lse.data(), lse0.data(), kHq * 4), 0)
                << "det-reduce lse not bit-stable (rep " << rep << ")";
        }
    }
}

TEST(CsaHcaDevice, SwaOnlyDecodeMatchesReference) {
    REQUIRES_GPU();
    Fixture f;
    f.build(77, 150, 8);

    std::mt19937 rng(66);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> qn(size_t(kHq) * kHeadDim, 0.0f),
        qr(size_t(kHq) * kRopeDim);
    for (int h = 0; h < kHq; ++h)
        for (int i = 0; i < kHeadDim - kRopeDim; ++i)
            qn[size_t(h) * kHeadDim + i] = bf16_round(dist(rng) * 0.3f);
    for (auto& v : qr) v = bf16_round(dist(rng) * 0.3f);

    std::vector<int> swa_bt = {0};
    std::vector<int> swa_sl = {kWindow};
    std::vector<int> positions = {f.T - 1};

    auto qn_bf = to_bf16(qn);
    auto qr_bf = to_bf16(qr);
    DeviceBuf d_qn(qn_bf.size() * 2), d_qr(qr_bf.size() * 2), d_bt(4), d_sl(4),
        d_pos(4), d_out(size_t(kHq) * kHeadDim * 2), d_lse(kHq * 4);
    CUDA_CHECK(cudaMemcpy(d_qn.p, qn_bf.data(), qn_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qr.p, qr_bf.data(), qr_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt.p, swa_bt.data(), 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sl.p, swa_sl.data(), 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.p, positions.data(), 4,
                          cudaMemcpyHostToDevice));

    lc::V4AttentionArgs a;
    a.rows = 1;
    a.q_nope = d_qn.p;
    a.q_rope = d_qr.p;
    a.topk = 0;  // SWA-only arm
    a.swa_cache = f.d_swa->p;
    a.swa_block_table = static_cast<const int*>(d_bt.p);
    a.swa_block_table_stride = 1;
    a.swa_seqlens = static_cast<const int*>(d_sl.p);
    a.positions = static_cast<const int*>(d_pos.p);
    a.rope_table = f.d_tab->p;
    a.out = d_out.p;
    a.lse = static_cast<float*>(d_lse.p);
    lc::csa_hca_device_attention(f.dev.get(), a);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out(size_t(kHq) * kHeadDim);
    std::vector<float> lse(kHq);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out.p, out.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lse.data(), d_lse.p, kHq * 4,
                          cudaMemcpyDeviceToHost));

    std::vector<int> swa_rows(kWindow);
    for (int i = 0; i < kWindow; ++i) swa_rows[i] = i;
    auto entries = f.visible_entries({}, kWindow, swa_rows);
    std::vector<double> ref_out, ref_lse;
    ref_attention(qn, qr, entries, {}, f.tab, f.T - 1, ref_out, ref_lse);
    expect_close(out, lse, ref_out, ref_lse, 1, "swa-only");
}

TEST(CsaHcaDevice, PrefillAsDecodePerRowWindows) {
    REQUIRES_GPU();
    Fixture f;
    f.build(4321, 200, 50);

    // 3 query rows at positions {150, 180, 199}: each sees its own raw
    // window + its visible compressed prefix — expressed as per-row indices
    // over a gathered staging of ALL entries (comp rows [0,50) then raw ring
    // rows [50, 178)).
    const int rows = 3;
    const std::vector<int> qpos = {150, 180, 199};

    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> qn(size_t(rows) * kHq * kHeadDim, 0.0f),
        qr(size_t(rows) * kHq * kRopeDim);
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < kHq; ++h)
            for (int i = 0; i < kHeadDim - kRopeDim; ++i)
                qn[(size_t(r) * kHq + h) * kHeadDim + i] =
                    bf16_round(dist(rng) * 0.3f);
    for (auto& v : qr) v = bf16_round(dist(rng) * 0.3f);

    // Gather: comp slots 0..49 → staging rows 0..49; raw ring rows 0..127 →
    // staging rows 50..177 (ring row i holds the token with pos%128 == i).
    std::vector<int> gather1(50), gather2(kWindow);
    for (int i = 0; i < 50; ++i) gather1[i] = i;
    for (int i = 0; i < kWindow; ++i) gather2[i] = i;
    DeviceBuf d_g1(50 * 4), d_g2(kWindow * 4);
    CUDA_CHECK(cudaMemcpy(d_g1.p, gather1.data(), 50 * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g2.p, gather2.data(), kWindow * 4,
                          cudaMemcpyHostToDevice));
    void* staging = nullptr;
    lc::V4EntryGatherArgs g1;
    g1.src_cache = f.d_csa->p;
    g1.slots = static_cast<const int*>(d_g1.p);
    g1.count = 50;
    g1.dst_row_offset = 0;
    lc::csa_hca_device_gather_entries(f.dev.get(), g1, &staging, nullptr);
    lc::V4EntryGatherArgs g2;
    g2.src_cache = f.d_swa->p;
    g2.slots = static_cast<const int*>(d_g2.p);
    g2.count = kWindow;
    g2.dst_row_offset = 50;
    lc::csa_hca_device_gather_entries(f.dev.get(), g2, &staging, nullptr);

    // Per-row indices: visible comp blocks (endpoint 4j+3 <= pos) + raw
    // window rows (staging 50 + (t % 128) for t in [pos-127, pos]).
    const int topk = 256;  // 50 comp + 128 raw = 178 → pad to 256
                            // (whole-tile -1 padding exercises the deps
                            //  empty-tile no-op fix)
    const bool kDebugCompOnly = false;
    const bool kDebugRealCache = false;
    std::vector<int> indices(size_t(rows) * topk, -1);
    std::vector<std::vector<int>> ref_slots(rows);
    for (int r = 0; r < rows; ++r) {
        int n = 0;
        const int nvis = (qpos[r] + 1) / 4;
        for (int j = 0; j < std::min(nvis, 50); ++j) {
            indices[size_t(r) * topk + n++] = j;
        }
        if (!kDebugCompOnly) {
        for (int t = qpos[r] - kWindow + 1; t <= qpos[r]; ++t) {
            if (t < 0) continue;
            indices[size_t(r) * topk + n++] = 50 + (t % kWindow);
        }
        }
        ref_slots[r].assign(indices.begin() + size_t(r) * topk,
                            indices.begin() + size_t(r) * topk + n);
    }

    auto qn_bf = to_bf16(qn);
    auto qr_bf = to_bf16(qr);
    DeviceBuf d_qn(qn_bf.size() * 2), d_qr(qr_bf.size() * 2),
        d_idx(indices.size() * 4), d_sl(rows * 4), d_pos(rows * 4),
        d_out(size_t(rows) * kHq * kHeadDim * 2), d_lse(rows * kHq * 4);
    std::vector<int> zero_sl(rows, 0);
    CUDA_CHECK(cudaMemcpy(d_qn.p, qn_bf.data(), qn_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qr.p, qr_bf.data(), qr_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx.p, indices.data(), indices.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sl.p, zero_sl.data(), rows * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.p, qpos.data(), rows * 4,
                          cudaMemcpyHostToDevice));

    lc::V4AttentionArgs a;
    a.rows = rows;
    a.q_nope = d_qn.p;
    a.q_rope = d_qr.p;
    a.comp_cache = kDebugRealCache ? f.d_csa->p : staging;
    a.sparse_indices = static_cast<const int*>(d_idx.p);
    a.topk = topk;
    a.swa_cache = staging;          // valid pointer; swa_seqlens = 0
    a.swa_block_table = nullptr;
    a.swa_block_table_stride = 0;
    a.swa_seqlens = static_cast<const int*>(d_sl.p);
    a.positions = static_cast<const int*>(d_pos.p);
    a.rope_table = f.d_tab->p;
    a.out = d_out.p;
    a.lse = static_cast<float*>(d_lse.p);
    lc::csa_hca_device_attention(f.dev.get(), a);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out(size_t(rows) * kHq * kHeadDim);
    std::vector<float> lse(size_t(rows) * kHq);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out.p, out.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lse.data(), d_lse.p, lse.size() * 4,
                          cudaMemcpyDeviceToHost));

    // Staging host mirror for the reference.
    std::vector<uint8_t> staging_host(size_t(50 + kWindow) * kEntryBytes);
    CUDA_CHECK(cudaMemcpy(staging_host.data(),
                          kDebugRealCache ? f.d_csa->p : staging,
                          kDebugRealCache ? size_t(50) * kEntryBytes
                                          : staging_host.size(),
                          cudaMemcpyDeviceToHost));
    for (int r = 0; r < rows; ++r) {
        std::vector<Entry> entries;
        for (int s : ref_slots[r]) {
            entries.push_back(parse_entry(staging_host.data() +
                                          size_t(s) * kEntryBytes));
        }
        std::vector<float> qn_r(qn.begin() + size_t(r) * kHq * kHeadDim,
                                qn.begin() + size_t(r + 1) * kHq * kHeadDim);
        std::vector<float> qr_r(qr.begin() + size_t(r) * kHq * kRopeDim,
                                qr.begin() + size_t(r + 1) * kHq * kRopeDim);
        std::vector<double> ref_out, ref_lse;
        ref_attention(qn_r, qr_r, entries, {}, f.tab, qpos[r], ref_out,
                      ref_lse);
        std::vector<__nv_bfloat16> out_r(
            out.begin() + size_t(r) * kHq * kHeadDim,
            out.begin() + size_t(r + 1) * kHq * kHeadDim);
        std::vector<float> lse_r(lse.begin() + size_t(r) * kHq,
                                 lse.begin() + size_t(r + 1) * kHq);
        expect_close(out_r, lse_r, ref_out, ref_lse, 1,
                     ("prefill row " + std::to_string(r)).c_str());
    }
}

// SC (superchunk port): the batch-shaped prefill call shape — per-row
// compressed indices from launch_v4_prefill_indices (+ elementwise slot
// translate) AND the raw window via the SWA arm at swa_page_block_size = 1
// over a chunk staging (ring prefix gather + chunk appends), one batched
// device call for all rows — against the CPU double reference per row.
TEST(CsaHcaDevice, BatchPrefillSwaPageSize1MatchesReference) {
    REQUIRES_GPU();
    Fixture f;
    // Ring holds positions [0, 128) exactly (the chunk starts at 128) —
    // but compress/append/inverse rope reach position 199 (block 49 ropes
    // at 4*49+3), so the table must cover the full range.
    f.table_pos = 208;
    f.build(7777, kWindow, 50);

    const int p0 = kWindow;      // 128
    const int rows = 64;         // chunk positions 128..191
    const int W_pref = kWindow;  // min(p0, W)

    std::mt19937 rng(555);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Chunk kv rows (positions p0..p0+rows) → staging rows [W_pref, ...).
    std::vector<float> ckv(size_t(rows) * kHeadDim);
    for (auto& v : ckv) v = bf16_round(dist(rng) * 0.5f);
    auto ckv_bf = to_bf16(ckv);
    std::vector<int> cpos(rows), cslots(rows), prefix(kWindow);
    for (int i = 0; i < rows; ++i) {
        cpos[i] = p0 + i;
        cslots[i] = W_pref + i;
    }
    for (int i = 0; i < kWindow; ++i) prefix[i] = i;  // ring slot of pos i
    DeviceBuf d_ckv(ckv_bf.size() * 2), d_cpos(rows * 4), d_cslots(rows * 4),
        d_pfx(kWindow * 4);
    CUDA_CHECK(cudaMemcpy(d_ckv.p, ckv_bf.data(), ckv_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cpos.p, cpos.data(), rows * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cslots.p, cslots.data(), rows * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pfx.p, prefix.data(), kWindow * 4,
                          cudaMemcpyHostToDevice));

    // Staging: prefix gather then chunk append (the executor's order).
    void* staging = nullptr;
    lc::V4EntryGatherArgs ga;
    ga.src_cache = f.d_swa->p;
    ga.slots = static_cast<const int*>(d_pfx.p);
    ga.count = kWindow;
    ga.dst_row_offset = 0;
    lc::csa_hca_device_gather_entries(f.dev.get(), ga, &staging, nullptr);
    lc::launch_v4_raw_kv_append(d_ckv.p, static_cast<const int*>(d_cpos.p),
                                static_cast<const int*>(d_cslots.p), staging,
                                f.d_tab->p, rows, kHeadDim, kRopeDim,
                                nullptr);

    // Per-row metadata: swa_len = min(pos+1, W) (= W here), row_nb =
    // (pos+1)/4 visible compressed blocks.
    std::vector<int> swa_len(rows), row_nb(rows);
    for (int i = 0; i < rows; ++i) {
        swa_len[i] = std::min(cpos[i] + 1, kWindow);
        row_nb[i] = (cpos[i] + 1) / 4;
    }
    DeviceBuf d_swalen(rows * 4), d_rownb(rows * 4);
    CUDA_CHECK(cudaMemcpy(d_swalen.p, swa_len.data(), rows * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rownb.p, row_nb.data(), rows * 4,
                          cudaMemcpyHostToDevice));

    // Per-row SWA block table (index list at page size 1) via the kernel.
    DeviceBuf d_bt(size_t(rows) * kWindow * 4);
    lc::launch_v4_prefill_swa_bt(static_cast<int*>(d_bt.p),
                                 static_cast<const int*>(d_swalen.p), rows,
                                 kWindow, W_pref, nullptr);

    // Per-row compressed indices: IOTA build → identity slot translate
    // (page table [0], entries_per_page 64 — cache rows ARE the slots).
    const int nvis_max = (cpos[rows - 1] + 1) / 4;  // 48 <= 50 built
    const int topk = 64;                            // pad64(48)
    DeviceBuf d_logical(size_t(rows) * topk * 4),
        d_phys(size_t(rows) * topk * 4), d_pt(4);
    lc::launch_v4_prefill_indices(static_cast<int*>(d_logical.p), nullptr,
                                  static_cast<const int*>(d_rownb.p), rows,
                                  topk, nullptr);
    const int pt0 = 0;
    CUDA_CHECK(cudaMemcpy(d_pt.p, &pt0, 4, cudaMemcpyHostToDevice));
    lc::launch_v4_slot_translate(static_cast<int*>(d_phys.p),
                                 static_cast<const int*>(d_logical.p),
                                 static_cast<const int*>(d_pt.p), 64,
                                 nvis_max, rows * topk, nullptr);

    // Queries.
    std::vector<float> qn(size_t(rows) * kHq * kHeadDim, 0.0f),
        qr(size_t(rows) * kHq * kRopeDim);
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < kHq; ++h)
            for (int i = 0; i < kHeadDim - kRopeDim; ++i)
                qn[(size_t(r) * kHq + h) * kHeadDim + i] =
                    bf16_round(dist(rng) * 0.3f);
    for (auto& v : qr) v = bf16_round(dist(rng) * 0.3f);
    auto qn_bf = to_bf16(qn);
    auto qr_bf = to_bf16(qr);
    DeviceBuf d_qn(qn_bf.size() * 2), d_qr(qr_bf.size() * 2),
        d_out(size_t(rows) * kHq * kHeadDim * 2), d_lse(rows * kHq * 4);
    CUDA_CHECK(cudaMemcpy(d_qn.p, qn_bf.data(), qn_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qr.p, qr_bf.data(), qr_bf.size() * 2,
                          cudaMemcpyHostToDevice));

    lc::V4AttentionArgs a;
    a.rows = rows;
    a.q_nope = d_qn.p;
    a.q_rope = d_qr.p;
    a.comp_cache = f.d_csa->p;
    a.sparse_indices = static_cast<const int*>(d_phys.p);
    a.topk = topk;
    a.swa_cache = staging;
    a.swa_block_table = static_cast<const int*>(d_bt.p);
    a.swa_block_table_stride = kWindow;
    a.swa_seqlens = static_cast<const int*>(d_swalen.p);
    a.swa_page_block_size = 1;      // SC per-token index list
    a.positions = static_cast<const int*>(d_cpos.p);
    a.rope_table = f.d_tab->p;
    a.out = d_out.p;
    a.lse = static_cast<float*>(d_lse.p);
    lc::csa_hca_device_attention(f.dev.get(), a);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out(size_t(rows) * kHq * kHeadDim);
    std::vector<float> lse(size_t(rows) * kHq);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out.p, out.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lse.data(), d_lse.p, lse.size() * 4,
                          cudaMemcpyDeviceToHost));
    std::vector<uint8_t> staging_host(size_t(W_pref + rows) * kEntryBytes);
    CUDA_CHECK(cudaMemcpy(staging_host.data(), staging, staging_host.size(),
                          cudaMemcpyDeviceToHost));

    for (int r = 0; r < rows; ++r) {
        std::vector<Entry> entries;
        for (int j = 0; j < row_nb[r]; ++j)
            entries.push_back(parse_entry(f.csa_cache_host.data() +
                                          size_t(j) * kEntryBytes));
        // Window (ascending chronological): staging row of position
        // cpos[r]-swa_len+1+j is (pos - p0 + W_pref).
        for (int j = 0; j < swa_len[r]; ++j) {
            const int pos = cpos[r] - swa_len[r] + 1 + j;
            entries.push_back(parse_entry(
                staging_host.data() +
                size_t(pos - p0 + W_pref) * kEntryBytes));
        }
        std::vector<float> qn_r(qn.begin() + size_t(r) * kHq * kHeadDim,
                                qn.begin() + size_t(r + 1) * kHq * kHeadDim);
        std::vector<float> qr_r(qr.begin() + size_t(r) * kHq * kRopeDim,
                                qr.begin() + size_t(r + 1) * kHq * kRopeDim);
        std::vector<double> ref_out, ref_lse;
        ref_attention(qn_r, qr_r, entries, {}, f.tab, cpos[r], ref_out,
                      ref_lse);
        std::vector<__nv_bfloat16> out_r(
            out.begin() + size_t(r) * kHq * kHeadDim,
            out.begin() + size_t(r + 1) * kHq * kHeadDim);
        std::vector<float> lse_r(lse.begin() + size_t(r) * kHq,
                                 lse.begin() + size_t(r + 1) * kHq);
        expect_close(out_r, lse_r, ref_out, ref_lse, 1,
                     ("batch prefill row " + std::to_string(r)).c_str());
    }
}

TEST(CsaHcaDevice, GraphReplayMatchesDirectCall) {
    REQUIRES_GPU();
    Fixture f;
    f.build(2026, 200, 50);

    std::mt19937 rng(31);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> qn(size_t(kHq) * kHeadDim, 0.0f),
        qr(size_t(kHq) * kRopeDim);
    for (int h = 0; h < kHq; ++h)
        for (int i = 0; i < kHeadDim - kRopeDim; ++i)
            qn[size_t(h) * kHeadDim + i] = bf16_round(dist(rng) * 0.3f);
    for (auto& v : qr) v = bf16_round(dist(rng) * 0.3f);

    std::vector<int> indices(64, -1);
    for (int b = 0; b < 50; ++b) indices[b] = b;
    std::vector<int> swa_bt = {0};
    std::vector<int> swa_sl = {kWindow};
    std::vector<int> positions = {f.T - 1};
    std::vector<float> sinks(kHq, 0.5f);

    auto qn_bf = to_bf16(qn);
    auto qr_bf = to_bf16(qr);
    DeviceBuf d_qn(qn_bf.size() * 2), d_qr(qr_bf.size() * 2), d_idx(64 * 4),
        d_bt(4), d_sl(4), d_pos(4), d_sinks(kHq * 4),
        d_out(size_t(kHq) * kHeadDim * 2), d_lse(kHq * 4);
    CUDA_CHECK(cudaMemcpy(d_qn.p, qn_bf.data(), qn_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qr.p, qr_bf.data(), qr_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx.p, indices.data(), 64 * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt.p, swa_bt.data(), 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sl.p, swa_sl.data(), 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.p, positions.data(), 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sinks.p, sinks.data(), kHq * 4,
                          cudaMemcpyHostToDevice));

    // Direct call with nsp=1 (the graph arm captures nsp=1).
    lc::V4AttentionArgs a;
    a.rows = 1;
    a.q_nope = d_qn.p;
    a.q_rope = d_qr.p;
    a.comp_cache = f.d_csa->p;
    a.sparse_indices = static_cast<const int*>(d_idx.p);
    a.topk = 64;
    a.swa_cache = f.d_swa->p;
    a.swa_block_table = static_cast<const int*>(d_bt.p);
    a.swa_block_table_stride = 1;
    a.swa_seqlens = static_cast<const int*>(d_sl.p);
    a.num_sm_parts = 1;
    a.sinks = d_sinks.p;
    a.positions = static_cast<const int*>(d_pos.p);
    a.rope_table = f.d_tab->p;
    a.out = d_out.p;
    a.lse = static_cast<float*>(d_lse.p);
    lc::csa_hca_device_attention(f.dev.get(), a);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> direct_out(size_t(kHq) * kHeadDim);
    std::vector<float> direct_lse(kHq);
    CUDA_CHECK(cudaMemcpy(direct_out.data(), d_out.p, direct_out.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(direct_lse.data(), d_lse.p, kHq * 4,
                          cudaMemcpyDeviceToHost));

    // Graph arm.
    lc::V4GraphInitArgs ga;
    ga.batch = 1;
    ga.topk = 64;
    ga.comp_cache = f.d_csa->p;
    ga.swa_cache = f.d_swa->p;
    ga.max_swa_blocks = 1;
    ga.num_sm_parts = 1;
    void* runner = lc::csa_hca_device_graph_init(f.dev.get(), ga);
    ASSERT_NE(runner, nullptr);
    lc::csa_hca_device_graph_update(
        f.dev.get(), runner, d_qn.p, d_qr.p,
        static_cast<const int*>(d_idx.p), static_cast<const int*>(d_bt.p),
        static_cast<const int*>(d_sl.p), nullptr);
    lc::csa_hca_device_graph_replay(f.dev.get(), runner, nullptr);
    lc::csa_hca_device_graph_epilogue(f.dev.get(), runner, 1, d_sinks.p, 0,
                                      static_cast<const int*>(d_pos.p),
                                      f.d_tab->p, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    void* g_out = nullptr;
    float* g_lse = nullptr;
    lc::csa_hca_device_graph_outputs(f.dev.get(), runner, &g_out, &g_lse);
    std::vector<uint16_t> graph_out(size_t(kHq) * kHeadDim);
    std::vector<float> graph_lse(kHq);
    CUDA_CHECK(cudaMemcpy(graph_out.data(), g_out, graph_out.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(graph_lse.data(), g_lse, kHq * 4,
                          cudaMemcpyDeviceToHost));

    EXPECT_EQ(direct_out, graph_out)
        << "graph replay must be bit-identical to the direct nsp=1 call";
    for (int h = 0; h < kHq; ++h) {
        EXPECT_FLOAT_EQ(direct_lse[h], graph_lse[h]) << "lse h=" << h;
    }
}

TEST(CsaHcaDevice, LightningSelectMatchesReference) {
    REQUIRES_GPU();
    Fixture f;
    f.build(9, 64, 8);
    const int D = 128, NH = 4, rows = 2, nblocks = 8, topk = 4;

    std::mt19937 rng(15);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Paged indexer tier: 4 entries/page, page = 4*(128+4) bytes.
    const int page_entries = 4;
    const int64_t page_bytes = page_entries * (D + 4);
    const int npages = nblocks / page_entries;
    std::vector<uint8_t> pages(size_t(npages) * page_bytes);
    std::vector<std::vector<float>> k_deq(nblocks, std::vector<float>(D));
    for (int b = 0; b < nblocks; ++b) {
        const int pg = b / page_entries, rw = b % page_entries;
        auto* base = pages.data() + size_t(pg) * page_bytes;
        auto* k8 = reinterpret_cast<__nv_fp8_e4m3*>(base + size_t(rw) * D);
        auto* sc = reinterpret_cast<float*>(base + size_t(page_entries) * D +
                                            size_t(rw) * 4);
        const float scale = 0.05f + 0.01f * b;
        *sc = scale;
        for (int i = 0; i < D; ++i) {
            k8[i] = __nv_fp8_e4m3(dist(rng));
            k_deq[b][i] = float(k8[i]) * scale;
        }
    }
    DeviceBuf d_pages(pages.size());
    CUDA_CHECK(cudaMemcpy(d_pages.p, pages.data(), pages.size(),
                          cudaMemcpyHostToDevice));

    // Per-row page tables (both rows share the pages here).
    std::vector<const void*> ptab(size_t(rows) * npages);
    for (int r = 0; r < rows; ++r)
        for (int p = 0; p < npages; ++p)
            ptab[size_t(r) * npages + p] =
                static_cast<const uint8_t*>(d_pages.p) + size_t(p) * page_bytes;
    DeviceBuf d_ptab(ptab.size() * sizeof(void*));
    CUDA_CHECK(cudaMemcpy(d_ptab.p, ptab.data(), ptab.size() * sizeof(void*),
                          cudaMemcpyHostToDevice));

    std::vector<float> q(size_t(rows) * NH * D), w(size_t(rows) * NH);
    for (auto& v : q) v = bf16_round(dist(rng) * 0.3f);
    for (auto& v : w) v = std::abs(dist(rng));
    auto q_bf = to_bf16(q);
    std::vector<int> nb = {nblocks, nblocks};
    std::vector<int> endpoints(nblocks), qpos = {31, 63};
    for (int b = 0; b < nblocks; ++b) endpoints[b] = 4 * b + 3;

    DeviceBuf d_q(q_bf.size() * 2), d_w(w.size() * 4), d_nb(rows * 4),
        d_ep(nblocks * 4), d_qpos(rows * 4), d_idx(size_t(rows) * topk * 4);
    CUDA_CHECK(cudaMemcpy(d_q.p, q_bf.data(), q_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w.p, w.data(), w.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nb.p, nb.data(), rows * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ep.p, endpoints.data(), nblocks * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qpos.p, qpos.data(), rows * 4,
                          cudaMemcpyHostToDevice));

    lc::V4LightningArgs la;
    la.rows = rows;
    la.q_proj = d_q.p;
    la.score_w = d_w.p;
    la.row_num_blocks = static_cast<const int*>(d_nb.p);
    la.k_page_table = static_cast<const void* const*>(d_ptab.p);
    la.page_table_stride = npages;
    la.block_endpoints = static_cast<const int*>(d_ep.p);
    la.query_positions = static_cast<const int*>(d_qpos.p);
    la.topk = topk;
    la.indices_out = static_cast<int*>(d_idx.p);
    lc::csa_hca_device_lightning_select(f.dev.get(), la, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> got(size_t(rows) * topk);
    CUDA_CHECK(cudaMemcpy(got.data(), d_idx.p, got.size() * 4,
                          cudaMemcpyDeviceToHost));

    for (int r = 0; r < rows; ++r) {
        // Reference scores: sum_h w[h] * relu(q_h · k_deq_b).
        std::vector<std::pair<float, int>> sc;
        for (int b = 0; b < nblocks; ++b) {
            if (endpoints[b] > qpos[r]) continue;  // causality cutoff
            double s = 0;
            for (int h = 0; h < NH; ++h) {
                double d = 0;
                for (int i = 0; i < D; ++i)
                    d += double(q[(size_t(r) * NH + h) * D + i]) * k_deq[b][i];
                s += w[size_t(r) * NH + h] * std::max(0.0, d);
            }
            sc.push_back({float(s), b});
        }
        std::sort(sc.begin(), sc.end(),
                  [](auto& x, auto& y) { return x.first > y.first; });
        std::vector<int> want;
        for (int k = 0; k < topk && k < (int)sc.size(); ++k)
            want.push_back(sc[k].second);
        std::sort(want.begin(), want.end());
        std::vector<int> got_r(got.begin() + size_t(r) * topk,
                               got.begin() + size_t(r) * topk + want.size());
        EXPECT_EQ(got_r, want) << "row " << r;
    }
}

// Ticket-H regression probe: the golden-boot call shape (seqlen 1, pos 0,
// num_sm_parts 32, sinks) lost the q_rope·k_rope score term in situ. This
// reproduces the exact decode call the executor makes at pos 0.
TEST(CsaHcaDevice, SwaSingleTokenNsp32KeepsRopeTerm) {
    REQUIRES_GPU();
    Fixture f;
    f.build(4242, /*seq_len=*/1, /*blocks=*/2);

    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> qn(size_t(kHq) * kHeadDim, 0.0f),
        qr(size_t(kHq) * kRopeDim);
    for (int h = 0; h < kHq; ++h)
        for (int i = 0; i < kHeadDim - kRopeDim; ++i)
            qn[size_t(h) * kHeadDim + i] = bf16_round(dist(rng) * 0.3f);
    // V4-realistic magnitude asymmetry: the roped pe dwarfs the nope amax
    // (engine L0: |q_rope| rms ~2, values to ~30 vs nope ~0.9). This makes
    // the rope score term LARGE, so a kernel that loses it (the ticket-H
    // dequant-ordering bug: FP8 qk descale applied AFTER the BF16 rope MMA)
    // fails the 8e-2 LSE tolerance instead of hiding inside it.
    for (auto& v : qr) v = bf16_round(dist(rng) * 6.0f);
    std::vector<float> sinks(kHq);
    for (auto& v : sinks) v = dist(rng);

    std::vector<int> swa_bt = {0};
    std::vector<int> swa_sl = {1};
    std::vector<int> positions = {0};

    auto qn_bf = to_bf16(qn);
    auto qr_bf = to_bf16(qr);
    DeviceBuf d_qn(qn_bf.size() * 2), d_qr(qr_bf.size() * 2), d_bt(4),
        d_sl(4), d_pos(4), d_sinks(kHq * 4),
        d_out(size_t(kHq) * kHeadDim * 2), d_lse(kHq * 4);
    CUDA_CHECK(cudaMemcpy(d_qn.p, qn_bf.data(), qn_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qr.p, qr_bf.data(), qr_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt.p, swa_bt.data(), 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sl.p, swa_sl.data(), 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.p, positions.data(), 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sinks.p, sinks.data(), kHq * 4,
                          cudaMemcpyHostToDevice));

    lc::V4AttentionArgs a;
    a.rows = 1;
    a.q_nope = d_qn.p;
    a.q_rope = d_qr.p;
    a.topk = 0;  // SWA-only arm
    a.swa_cache = f.d_swa->p;
    a.swa_block_table = static_cast<const int*>(d_bt.p);
    a.swa_block_table_stride = 1;
    a.swa_seqlens = static_cast<const int*>(d_sl.p);
    a.num_sm_parts = 32;  // engine default (V4DeviceOptions)
    a.sinks = d_sinks.p;
    a.positions = static_cast<const int*>(d_pos.p);
    a.rope_table = f.d_tab->p;
    a.out = d_out.p;
    a.lse = static_cast<float*>(d_lse.p);
    lc::csa_hca_device_attention(f.dev.get(), a);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out(size_t(kHq) * kHeadDim);
    std::vector<float> lse(kHq);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out.p, out.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lse.data(), d_lse.p, kHq * 4,
                          cudaMemcpyDeviceToHost));

    std::vector<int> swa_rows = {0};
    auto entries = f.visible_entries({}, 1, swa_rows);
    std::vector<double> ref_out, ref_lse;
    ref_attention(qn, qr, entries, sinks, f.tab, 0, ref_out, ref_lse);
    expect_close(out, lse, ref_out, ref_lse, 1, "swa-1tok-nsp32");
}

// Ticket-H forensic replay: feed the EXACT engine-dumped bytes (from the V4
// golden boot, V4_L0_DUMP) through the device and print per-head LSE next to
// the values the engine recorded. Gated on V4L0 dir env; not a CI test.
TEST(CsaHcaDevice, ReplayEngineL0Dump) {
    REQUIRES_GPU();
    const char* dir = std::getenv("V4L0");
    if (!dir || !*dir) GTEST_SKIP() << "set V4L0=<dump dir>";
    auto slurp = [&](const char* n) {
        std::string p = std::string(dir) + "/" + n;
        FILE* f = fopen(p.c_str(), "rb");
        EXPECT_NE(f, nullptr) << p;
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> v(sz);
        (void)!fread(v.data(), 1, sz, f); fclose(f);
        return v;
    };
    auto qn = slurp("dcp.v4_q_nope.rank0.bin");
    auto qr = slurp("dcp.v4_q_rope.rank0.bin");
    auto swa = slurp("vram.kv_swa.gpu0.bin");
    auto lse_engine = slurp("dcp.v4_attn_lse.rank0.bin");

    Fixture f;  // configures device with the TEST options
    f.build(1, 1, 2);
    // Reconfigure with the ENGINE's options (max_batch 64, rows 512, nsp 32,
    // topk 512, idx page 2048) to match the boot context.
    lc::V4DeviceOptions o;
    o.max_batch = 64;
    o.max_attn_rows = 512;
    o.h_q = kHq;
    o.sm_scale = kSmScale;
    o.topk = 512;
    o.sliding_window = kWindow;
    o.num_sm_parts = 32;
    o.csa_entries_per_page = 64;
    o.hca_entries_per_page = 2;
    o.swa_page_tokens = kWindow;
    o.idx_entries_per_page = 2048;
    o.idx_page_bytes = 2048 * 132;
    o.index_n_heads = 64;
    o.index_head_dim = 128;
    o.max_index_blocks = 8192;
    lc::csa_hca_device_configure(f.dev.get(), o);
    lc::csa_hca_device_set_scratch(f.dev.get(), 9000);

    DeviceBuf d_qn(size_t(kHq) * kHeadDim * 2), d_qr(size_t(kHq) * kRopeDim * 2),
        d_swa(size_t(kWindow) * kEntryBytes), d_bt(4), d_sl(4), d_pos(4),
        d_out(size_t(kHq) * kHeadDim * 2), d_lse(kHq * 4);
    CUDA_CHECK(cudaMemcpy(d_qn.p, qn.data(), size_t(kHq) * kHeadDim * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qr.p, qr.data(), size_t(kHq) * kRopeDim * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_swa.p, swa.data(), size_t(kWindow) * kEntryBytes,
                          cudaMemcpyHostToDevice));
    int one = 1, zero = 0;
    CUDA_CHECK(cudaMemcpy(d_bt.p, &zero, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sl.p, &one, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.p, &zero, 4, cudaMemcpyHostToDevice));

    lc::V4AttentionArgs a;
    a.rows = 1;
    a.q_nope = d_qn.p;
    a.q_rope = d_qr.p;
    a.topk = 0;
    a.swa_cache = d_swa.p;
    a.swa_block_table = static_cast<const int*>(d_bt.p);
    a.swa_block_table_stride = 1;
    a.swa_seqlens = static_cast<const int*>(d_sl.p);
    a.positions = static_cast<const int*>(d_pos.p);
    // NO sinks (compare raw LSE), rope table = identity at pos 0 anyway.
    a.rope_table = f.d_tab->p;
    a.out = d_out.p;
    a.lse = static_cast<float*>(d_lse.p);
    lc::csa_hca_device_attention(f.dev.get(), a);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> lse(kHq);
    CUDA_CHECK(cudaMemcpy(lse.data(), d_lse.p, kHq * 4,
                          cudaMemcpyDeviceToHost));
    const float* le = reinterpret_cast<const float*>(lse_engine.data());
    for (int h = 0; h < 8; ++h) {
        fprintf(stderr, "head %d: replay_lse=%.4f engine_lse(post-sink)=%.4f\n",
                h, lse[h], le[h]);
    }
}
