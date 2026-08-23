// Unit tests for the DeepSeek-V4 attention prep/compress kernels (V4-5a).
//
// CPU double references implement the llama.cpp deepseek4 math directly:
//   q prep:      rms_norm(q[512]) pre-rope, split 448|64, interleaved rope
//   raw append:  duplicated-rope 1160-B FP8 entry (scale = amax/448)
//   compress:    per-CHANNEL softmax over the overlap window (prev|cur
//                halves + APE[in-stride pos]; t<0 excluded), weighted RMS
//                norm, rope at block endpoint (deepseek4.cpp:382-520)
//   sinks:       out·sigmoid(lse−s), lse→logaddexp(lse, s)
//   inv rope:    transpose rotation (rope→inverse == identity)

#include "compute/kernels/attention/v4_prep.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

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
constexpr float kRmsEps = 1e-6f;
constexpr float kFp8Max = 448.0f;
constexpr int kEntryBytes = kHeadDim + 4 + kRopeDim * 2 + kHeadDim + 4;

float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& v) {
    std::vector<__nv_bfloat16> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = __float2bfloat16_rn(v[i]);
    return out;
}

// Ticket-D table layout: row = [cos_0..cos_{r/2-1} | sin_0..sin_{r/2-1}].
std::vector<float> make_cos_sin(int max_pos, int rope_dim, double theta) {
    const int half = rope_dim / 2;
    std::vector<float> t(size_t(max_pos) * rope_dim);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double freq = std::pow(theta, -2.0 * i / rope_dim);
            t[size_t(p) * rope_dim + i] = float(std::cos(p * freq));
            t[size_t(p) * rope_dim + half + i] = float(std::sin(p * freq));
        }
    }
    return t;
}

// Interleaved forward rope on x[rope_dim] (double, in place).
void ref_rope(std::vector<double>& x, const std::vector<float>& tab, int pos,
              int rope_dim) {
    const int half = rope_dim / 2;
    for (int i = 0; i < half; ++i) {
        const double c = tab[size_t(pos) * rope_dim + i];
        const double s = tab[size_t(pos) * rope_dim + half + i];
        const double e = x[2 * i], o = x[2 * i + 1];
        x[2 * i] = e * c - o * s;
        x[2 * i + 1] = e * s + o * c;
    }
}

float fp8_round(float x) {
    return float(__nv_fp8_e4m3(x));
}

struct DeviceBuf {
    void* p = nullptr;
    explicit DeviceBuf(size_t bytes) { cudaMalloc(&p, bytes); }
    ~DeviceBuf() { cudaFree(p); }
};

// Reference duplicated-rope entry from a roped 512 vector (double).
struct RefEntry {
    std::vector<float> k_nope{std::vector<float>(kHeadDim)};
    std::vector<float> k_rope{std::vector<float>(kRopeDim)};
    std::vector<float> v_nope{std::vector<float>(kHeadDim)};
    float k_scale = 0, v_scale = 0;
};
RefEntry ref_entry_from(const std::vector<double>& roped512) {
    RefEntry e;
    // The kernel quantizes BF16-rounded pair values; amax over bf16 inputs?
    // The kernel computes amax over the FLOAT values it assembled (not
    // bf16-rounded) — match that.
    double amax = 0;
    for (int i = 0; i < kHeadDim; ++i)
        amax = std::max(amax, std::abs(roped512[i]));
    e.k_scale = float(amax) / kFp8Max;
    e.v_scale = e.k_scale;
    const float inv = e.k_scale > 0 ? 1.0f / e.k_scale : 0.0f;
    for (int i = 0; i < kHeadDim; ++i) {
        const float q = fp8_round(std::max(
            -kFp8Max, std::min(kFp8Max, float(roped512[i]) * inv)));
        e.k_nope[i] = q * e.k_scale;
        e.v_nope[i] = q * e.k_scale;
    }
    for (int i = 0; i < kRopeDim; ++i)
        e.k_rope[i] = bf16_round(float(roped512[kHeadDim - kRopeDim + i]));
    return e;
}

// Parse a device 1160-B entry into dequantized floats.
RefEntry parse_entry(const uint8_t* entry) {
    RefEntry e;
    e.k_scale = *reinterpret_cast<const float*>(entry + kHeadDim);
    e.v_scale = *reinterpret_cast<const float*>(entry + kEntryBytes - 4);
    const auto* k8 = reinterpret_cast<const __nv_fp8_e4m3*>(entry);
    const auto* v8 = reinterpret_cast<const __nv_fp8_e4m3*>(
        entry + kHeadDim + 4 + kRopeDim * 2);
    const auto* kr = reinterpret_cast<const __nv_bfloat16*>(
        entry + kHeadDim + 4);
    for (int i = 0; i < kHeadDim; ++i) {
        e.k_nope[i] = float(k8[i]) * e.k_scale;
        e.v_nope[i] = float(v8[i]) * e.v_scale;
    }
    for (int i = 0; i < kRopeDim; ++i) e.k_rope[i] = __bfloat162float(kr[i]);
    return e;
}

}  // namespace

TEST(V4Prep, QPrepMatchesReference) {
    REQUIRES_GPU();
    const int B = 3, H = 4;
    const int max_pos = 64;
    std::mt19937 rng(5);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> q(size_t(B) * H * kHeadDim);
    for (auto& v : q) v = bf16_round(dist(rng));
    auto tab = make_cos_sin(max_pos, kRopeDim, 10000.0);
    std::vector<int> pos = {3, 17, 41};

    auto q_bf = to_bf16(q);
    DeviceBuf d_q(q_bf.size() * 2), d_tab(tab.size() * 4), d_pos(B * 4),
        d_qn(q_bf.size() * 2), d_qr(size_t(B) * H * kRopeDim * 2);
    CUDA_CHECK(cudaMemcpy(d_q.p, q_bf.data(), q_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tab.p, tab.data(), tab.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.p, pos.data(), B * 4, cudaMemcpyHostToDevice));

    lc::launch_v4_q_prep(d_qn.p, d_qr.p, d_q.p,
                         static_cast<const int*>(d_pos.p), d_tab.p, kRmsEps, B,
                         H, kHeadDim, kRopeDim, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> qn(q_bf.size()), qr(size_t(B) * H * kRopeDim);
    CUDA_CHECK(cudaMemcpy(qn.data(), d_qn.p, qn.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qr.data(), d_qr.p, qr.size() * 2,
                          cudaMemcpyDeviceToHost));

    const int nope = kHeadDim - kRopeDim;
    for (int t = 0; t < B; ++t) {
        for (int h = 0; h < H; ++h) {
            const size_t r = size_t(t) * H + h;
            const float* src = q.data() + r * kHeadDim;
            double sq = 0;
            for (int i = 0; i < kHeadDim; ++i) sq += double(src[i]) * src[i];
            const double inv = 1.0 / std::sqrt(sq / kHeadDim + kRmsEps);
            std::vector<double> pe(kRopeDim);
            for (int i = 0; i < kRopeDim; ++i) pe[i] = src[nope + i] * inv;
            ref_rope(pe, tab, pos[t], kRopeDim);
            for (int i = 0; i < nope; ++i) {
                EXPECT_NEAR(__bfloat162float(qn[r * kHeadDim + i]),
                            src[i] * inv, 1e-2)
                    << "qn t=" << t << " h=" << h << " i=" << i;
            }
            for (int i = nope; i < kHeadDim; ++i) {
                EXPECT_EQ(__bfloat162float(qn[r * kHeadDim + i]), 0.0f)
                    << "pad t=" << t << " h=" << h << " i=" << i;
            }
            for (int i = 0; i < kRopeDim; ++i) {
                EXPECT_NEAR(__bfloat162float(qr[r * kRopeDim + i]), pe[i],
                            2e-2)
                    << "qr t=" << t << " h=" << h << " i=" << i;
            }
        }
    }
}

TEST(V4Prep, RawKvAppendWritesDuplicatedRopeEntry) {
    REQUIRES_GPU();
    const int B = 4, max_pos = 32, num_slots = 16;
    std::mt19937 rng(9);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> kv(size_t(B) * kHeadDim);
    for (auto& v : kv) v = bf16_round(dist(rng));
    auto tab = make_cos_sin(max_pos, kRopeDim, 160000.0);
    std::vector<int> pos = {0, 7, 8, 21};
    std::vector<int> slots = {2, 5, 9, 15};

    auto kv_bf = to_bf16(kv);
    DeviceBuf d_kv(kv_bf.size() * 2), d_tab(tab.size() * 4), d_pos(B * 4),
        d_slots(B * 4), d_cache(size_t(num_slots) * kEntryBytes);
    CUDA_CHECK(cudaMemset(d_cache.p, 0, size_t(num_slots) * kEntryBytes));
    CUDA_CHECK(cudaMemcpy(d_kv.p, kv_bf.data(), kv_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tab.p, tab.data(), tab.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.p, pos.data(), B * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots.p, slots.data(), B * 4,
                          cudaMemcpyHostToDevice));

    lc::launch_v4_raw_kv_append(d_kv.p, static_cast<const int*>(d_pos.p),
                                static_cast<const int*>(d_slots.p), d_cache.p,
                                d_tab.p, B, kHeadDim, kRopeDim, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> cache(size_t(num_slots) * kEntryBytes);
    CUDA_CHECK(cudaMemcpy(cache.data(), d_cache.p, cache.size(),
                          cudaMemcpyDeviceToHost));

    const int nope = kHeadDim - kRopeDim;
    for (int t = 0; t < B; ++t) {
        std::vector<double> roped(kHeadDim);
        for (int i = 0; i < nope; ++i) roped[i] = kv[size_t(t) * kHeadDim + i];
        std::vector<double> pe(kRopeDim);
        for (int i = 0; i < kRopeDim; ++i)
            pe[i] = kv[size_t(t) * kHeadDim + nope + i];
        ref_rope(pe, tab, pos[t], kRopeDim);
        for (int i = 0; i < kRopeDim; ++i) roped[nope + i] = pe[i];
        const RefEntry want = ref_entry_from(roped);
        const RefEntry got =
            parse_entry(cache.data() + size_t(slots[t]) * kEntryBytes);
        EXPECT_NEAR(got.k_scale, want.k_scale, 1e-6) << "t=" << t;
        for (int i = 0; i < kHeadDim; ++i) {
            EXPECT_NEAR(got.k_nope[i], want.k_nope[i],
                        std::abs(want.k_nope[i]) * 0.10 + 0.05)
                << "k t=" << t << " i=" << i;
            EXPECT_EQ(got.v_nope[i], got.k_nope[i])
                << "v==k t=" << t << " i=" << i;
        }
        for (int i = 0; i < kRopeDim; ++i) {
            EXPECT_NEAR(got.k_rope[i], want.k_rope[i], 2e-2)
                << "kr t=" << t << " i=" << i;
        }
    }
}

namespace {

// CPU reference of the overlap/HCA compression producing the roped D vector.
// States are token-indexed [T, state_dim] (float), APE [stride, state_dim].
std::vector<double> ref_compress(const std::vector<float>& kv_state,
                                 const std::vector<float>& score_state,
                                 const std::vector<float>& ape,
                                 const std::vector<float>& norm_w,
                                 const std::vector<float>& tab, bool overlap,
                                 int stride, int D, int rope_dim, int j,
                                 int state_dim) {
    const int W = overlap ? 2 * stride : stride;
    std::vector<double> pooled(D, 0.0);
    for (int c = 0; c < D; ++c) {
        double m = -1e300;
        for (int w = 0; w < W; ++w) {
            int t, half;
            if (overlap) {
                if (w < stride) { t = (j - 1) * stride + w; half = 0; }
                else { t = j * stride + (w - stride); half = 1; }
            } else { t = j * stride + w; half = 0; }
            if (t < 0) continue;
            const int ch = half * D + c;
            const double sc = double(score_state[size_t(t) * state_dim + ch]) +
                              ape[size_t(t % stride) * state_dim + ch];
            m = std::max(m, sc);
        }
        double sum = 0, acc = 0;
        for (int w = 0; w < W; ++w) {
            int t, half;
            if (overlap) {
                if (w < stride) { t = (j - 1) * stride + w; half = 0; }
                else { t = j * stride + (w - stride); half = 1; }
            } else { t = j * stride + w; half = 0; }
            if (t < 0) continue;
            const int ch = half * D + c;
            const double sc = double(score_state[size_t(t) * state_dim + ch]) +
                              ape[size_t(t % stride) * state_dim + ch];
            const double e = std::exp(sc - m);
            sum += e;
            acc += e * kv_state[size_t(t) * state_dim + ch];
        }
        pooled[c] = sum > 0 ? acc / sum : 0.0;
    }
    double sq = 0;
    for (int c = 0; c < D; ++c) sq += pooled[c] * pooled[c];
    const double inv = 1.0 / std::sqrt(sq / D + kRmsEps);
    for (int c = 0; c < D; ++c) pooled[c] = pooled[c] * inv * norm_w[c];
    std::vector<double> pe(rope_dim);
    for (int i = 0; i < rope_dim; ++i) pe[i] = pooled[D - rope_dim + i];
    ref_rope(pe, tab, (j + 1) * stride - 1, rope_dim);
    for (int i = 0; i < rope_dim; ++i) pooled[D - rope_dim + i] = pe[i];
    return pooled;
}

void run_compress_case(bool overlap, int stride, int D, int first_block,
                       int num_blocks, bool indexer_paged, uint32_t seed) {
    const int state_dim = overlap ? 2 * D : D;
    const int last_token = (first_block + num_blocks) * stride - 1;
    const int T = last_token + 1;
    const int ring_capacity = T;  // contiguous (prefill-style) ring
    const int max_pos = T + stride;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> kv_state(size_t(T) * state_dim),
        score_state(size_t(T) * state_dim), ape(size_t(stride) * state_dim),
        norm_w(D);
    for (auto& v : kv_state) v = bf16_round(dist(rng));
    for (auto& v : score_state) v = bf16_round(dist(rng));
    for (auto& v : ape) v = dist(rng) * 0.5f;
    for (auto& v : norm_w) v = 1.0f + 0.1f * dist(rng);
    auto tab = make_cos_sin(max_pos, kRopeDim, 160000.0);

    std::vector<int> slots(num_blocks);
    for (int b = 0; b < num_blocks; ++b) slots[b] = first_block + b;

    auto kv_bf = to_bf16(kv_state);
    auto sc_bf = to_bf16(score_state);
    DeviceBuf d_kv(kv_bf.size() * 2), d_sc(sc_bf.size() * 2),
        d_ape(ape.size() * 4), d_norm(norm_w.size() * 4),
        d_tab(tab.size() * 4), d_slots(size_t(num_blocks) * 4);
    CUDA_CHECK(cudaMemcpy(d_kv.p, kv_bf.data(), kv_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sc.p, sc_bf.data(), sc_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ape.p, ape.data(), ape.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm.p, norm_w.data(), norm_w.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tab.p, tab.data(), tab.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots.p, slots.data(), size_t(num_blocks) * 4,
                          cudaMemcpyHostToDevice));

    lc::V4CompressArgs args;
    args.kv_state = d_kv.p;
    args.score_state = d_sc.p;
    args.ring_capacity = ring_capacity;
    args.state_dim = state_dim;
    args.overlap = overlap;
    args.stride = stride;
    args.ape = d_ape.p;
    args.norm_w = d_norm.p;
    args.cos_sin = d_tab.p;
    args.rms_eps = kRmsEps;
    args.D = D;
    args.rope_dim = kRopeDim;
    args.first_block = first_block;
    args.num_blocks = num_blocks;
    args.slots = static_cast<const int*>(d_slots.p);

    const int total_slots = first_block + num_blocks;
    const int entry_bytes = D + 4 + kRopeDim * 2 + D + 4;
    const int page_tokens = 4;
    const int64_t page_bytes = int64_t(page_tokens) * (D + 4);
    const int num_pages = (total_slots + page_tokens - 1) / page_tokens;

    DeviceBuf d_cache(size_t(total_slots) * entry_bytes),
        d_pages(size_t(num_pages) * page_bytes);
    CUDA_CHECK(cudaMemset(d_cache.p, 0, size_t(total_slots) * entry_bytes));
    CUDA_CHECK(cudaMemset(d_pages.p, 0, size_t(num_pages) * page_bytes));
    if (indexer_paged) {
        args.out_mode = lc::V4CompressArgs::Out::kIndexerPaged;
        args.idx_pages = d_pages.p;
        args.idx_page_tokens = page_tokens;
        args.idx_page_bytes = page_bytes;
    } else {
        args.out_mode = lc::V4CompressArgs::Out::kFp8Entry;
        args.kv_cache = d_cache.p;
    }

    lc::launch_v4_compress_insert(args, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int b = 0; b < num_blocks; ++b) {
        const int j = first_block + b;
        auto ref = ref_compress(kv_state, score_state, ape, norm_w, tab,
                                overlap, stride, D, kRopeDim, j, state_dim);
        if (!indexer_paged) {
            std::vector<uint8_t> entry(entry_bytes);
            CUDA_CHECK(cudaMemcpy(
                entry.data(),
                static_cast<uint8_t*>(d_cache.p) +
                    size_t(slots[b]) * entry_bytes,
                entry_bytes, cudaMemcpyDeviceToHost));
            // Dequant + compare (FP8 tolerance).
            const float k_scale = *reinterpret_cast<float*>(entry.data() + D);
            const auto* k8 =
                reinterpret_cast<const __nv_fp8_e4m3*>(entry.data());
            const auto* kr = reinterpret_cast<const __nv_bfloat16*>(
                entry.data() + D + 4);
            for (int c = 0; c < D; ++c) {
                EXPECT_NEAR(float(k8[c]) * k_scale, ref[c],
                            std::abs(ref[c]) * 0.12 + 0.06)
                    << "blk " << j << " c=" << c;
            }
            for (int i = 0; i < kRopeDim; ++i) {
                EXPECT_NEAR(__bfloat162float(kr[i]), ref[D - kRopeDim + i],
                            std::abs(ref[D - kRopeDim + i]) * 0.03 + 0.03)
                    << "blk " << j << " rope i=" << i;
            }
        } else {
            const int page = slots[b] / page_tokens;
            const int row = slots[b] % page_tokens;
            std::vector<uint8_t> pg(page_bytes);
            CUDA_CHECK(cudaMemcpy(
                pg.data(),
                static_cast<uint8_t*>(d_pages.p) + size_t(page) * page_bytes,
                page_bytes, cudaMemcpyDeviceToHost));
            const auto* k8 = reinterpret_cast<const __nv_fp8_e4m3*>(
                pg.data() + size_t(row) * D);
            const float scale = *reinterpret_cast<const float*>(
                pg.data() + size_t(page_tokens) * D + size_t(row) * 4);
            for (int c = 0; c < D; ++c) {
                EXPECT_NEAR(float(k8[c]) * scale, ref[c],
                            std::abs(ref[c]) * 0.12 + 0.06)
                    << "idx blk " << j << " c=" << c;
            }
        }
    }
}

}  // namespace

TEST(V4Prep, CompressOverlapCsaMain) {
    REQUIRES_GPU();
    // Includes block 0 (prev stride excluded — the -inf path).
    run_compress_case(/*overlap=*/true, /*stride=*/4, /*D=*/512,
                      /*first_block=*/0, /*num_blocks=*/5,
                      /*indexer_paged=*/false, 101);
}

TEST(V4Prep, CompressHcaMain) {
    REQUIRES_GPU();
    run_compress_case(false, 128, 512, 0, 2, false, 102);
}

TEST(V4Prep, CompressOverlapIndexerPaged) {
    REQUIRES_GPU();
    run_compress_case(true, 4, 128, 1, 6, true, 103);
}

TEST(V4Prep, AttnSinksExactFold) {
    REQUIRES_GPU();
    const int B = 2, H = 8, D = 512;
    std::mt19937 rng(77);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> out(size_t(B) * H * D), lse(size_t(B) * H),
        sinks(H + 3);
    for (auto& v : out) v = bf16_round(dist(rng));
    for (auto& v : lse) v = dist(rng) + 4.0f;  // plausible log-sum
    for (auto& v : sinks) v = dist(rng);

    auto out_bf = to_bf16(out);
    DeviceBuf d_out(out_bf.size() * 2), d_lse(lse.size() * 4),
        d_sinks(sinks.size() * 4);
    CUDA_CHECK(cudaMemcpy(d_out.p, out_bf.data(), out_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lse.p, lse.data(), lse.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sinks.p, sinks.data(), sinks.size() * 4,
                          cudaMemcpyHostToDevice));

    const int head_offset = 3;
    lc::launch_v4_attn_sinks(d_out.p, d_lse.p, d_sinks.p, head_offset, B, H, D,
                             nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> got(out_bf.size());
    std::vector<float> got_lse(lse.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_out.p, got.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_lse.data(), d_lse.p, got_lse.size() * 4,
                          cudaMemcpyDeviceToHost));

    for (int t = 0; t < B; ++t) {
        for (int h = 0; h < H; ++h) {
            const size_t r = size_t(t) * H + h;
            const double s = sinks[head_offset + h];
            const double factor = 1.0 / (1.0 + std::exp(s - lse[r]));
            for (int i = 0; i < D; ++i) {
                EXPECT_NEAR(__bfloat162float(got[r * D + i]),
                            out[r * D + i] * factor, 2e-2)
                    << "t=" << t << " h=" << h << " i=" << i;
            }
            const double want_lse =
                std::max<double>(lse[r], s) +
                std::log1p(std::exp(std::min<double>(lse[r], s) -
                                    std::max<double>(lse[r], s)));
            EXPECT_NEAR(got_lse[r], want_lse, 1e-5);
        }
    }
}

TEST(V4Prep, InverseRopeUndoesForwardRope) {
    REQUIRES_GPU();
    const int B = 2, H = 4, max_pos = 64;
    std::mt19937 rng(88);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> out(size_t(B) * H * kHeadDim);
    for (auto& v : out) v = bf16_round(dist(rng));
    auto tab = make_cos_sin(max_pos, kRopeDim, 10000.0);
    std::vector<int> pos = {11, 37};

    // Forward-rope the pe of every head on the CPU (double).
    std::vector<float> roped = out;
    const int nope = kHeadDim - kRopeDim;
    for (int t = 0; t < B; ++t) {
        for (int h = 0; h < H; ++h) {
            std::vector<double> pe(kRopeDim);
            const size_t r = (size_t(t) * H + h) * kHeadDim;
            for (int i = 0; i < kRopeDim; ++i) pe[i] = out[r + nope + i];
            ref_rope(pe, tab, pos[t], kRopeDim);
            for (int i = 0; i < kRopeDim; ++i)
                roped[r + nope + i] = bf16_round(float(pe[i]));
        }
    }

    auto roped_bf = to_bf16(roped);
    DeviceBuf d_out(roped_bf.size() * 2), d_tab(tab.size() * 4), d_pos(B * 4);
    CUDA_CHECK(cudaMemcpy(d_out.p, roped_bf.data(), roped_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tab.p, tab.data(), tab.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.p, pos.data(), B * 4, cudaMemcpyHostToDevice));

    lc::launch_v4_out_inverse_rope(d_out.p, static_cast<const int*>(d_pos.p),
                                   d_tab.p, B, H, kHeadDim, kRopeDim, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> got(roped_bf.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_out.p, got.size() * 2,
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(__bfloat162float(got[i]), out[i], 3e-2) << "i=" << i;
    }
}

TEST(V4Prep, SlotTranslateLogicalAndIota) {
    REQUIRES_GPU();
    // Page table: logical page p → physical page id (pool-relative).
    const std::vector<int> pt = {7, 3, 12, 0};
    const int epp = 64;
    const int n_valid = 4 * epp - 10;  // last page partially filled

    DeviceBuf d_pt(pt.size() * 4);
    CUDA_CHECK(cudaMemcpy(d_pt.p, pt.data(), pt.size() * 4,
                          cudaMemcpyHostToDevice));

    // Logical mode: lightning-style -1-padded ascending ids.
    const int topk = 128;
    std::vector<int> logical(topk, -1);
    logical[0] = 0;                 // page 0 entry 0
    logical[1] = 63;                // page 0 entry 63
    logical[2] = 64;                // page 1 entry 0
    logical[3] = 200;               // page 3 entry 8
    logical[4] = n_valid;           // out of causal bound → -1
    DeviceBuf d_log(topk * 4), d_out(topk * 4);
    CUDA_CHECK(cudaMemcpy(d_log.p, logical.data(), topk * 4,
                          cudaMemcpyHostToDevice));
    lc::launch_v4_slot_translate(static_cast<int*>(d_out.p),
                                 static_cast<const int*>(d_log.p),
                                 static_cast<const int*>(d_pt.p), epp,
                                 n_valid, topk, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> got(topk);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out.p, topk * 4,
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(got[0], 7 * epp);
    EXPECT_EQ(got[1], 7 * epp + 63);
    EXPECT_EQ(got[2], 3 * epp);
    EXPECT_EQ(got[3], 0 * epp + 8);
    EXPECT_EQ(got[4], -1);
    for (int i = 5; i < topk; ++i) EXPECT_EQ(got[i], -1) << i;

    // Iota mode (HCA dense visibility): epp 2, 5 valid entries padded to 64.
    const std::vector<int> pt2 = {9, 4, 5};
    DeviceBuf d_pt2(pt2.size() * 4), d_out2(64 * 4);
    CUDA_CHECK(cudaMemcpy(d_pt2.p, pt2.data(), pt2.size() * 4,
                          cudaMemcpyHostToDevice));
    lc::launch_v4_slot_translate(static_cast<int*>(d_out2.p), nullptr,
                                 static_cast<const int*>(d_pt2.p), 2, 5, 64,
                                 nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> got2(64);
    CUDA_CHECK(cudaMemcpy(got2.data(), d_out2.p, 64 * 4,
                          cudaMemcpyDeviceToHost));
    const std::vector<int> want2 = {18, 19, 8, 9, 10};
    for (int i = 0; i < 5; ++i) EXPECT_EQ(got2[i], want2[i]) << i;
    for (int i = 5; i < 64; ++i) EXPECT_EQ(got2[i], -1) << i;
}

// SC (superchunk port): per-row IOTA/-1 index build + lightning merge.
TEST(V4Prep, PrefillIndicesIotaAndLightningMerge) {
    REQUIRES_GPU();
    const int rows = 5, topk = 64;
    // Rows 0..3 within the top-k budget (IOTA); row 4 beyond it (lightning
    // passthrough, -1 padding preserved).
    const std::vector<int> row_nb = {0, 1, 37, 64, 65};
    std::vector<int> lightning(size_t(rows) * topk, -1);
    for (int j = 0; j < topk; ++j)
        lightning[size_t(4) * topk + j] = 1000 + j;   // row 4's selection
    lightning[size_t(4) * topk + 63] = -1;            // padded tail passes
    // Poison the iota rows' lightning input — must be ignored.
    for (int r = 0; r < 4; ++r)
        for (int j = 0; j < topk; ++j) lightning[size_t(r) * topk + j] = -7;

    DeviceBuf d_nb(rows * 4), d_in(lightning.size() * 4),
        d_out(lightning.size() * 4);
    CUDA_CHECK(cudaMemcpy(d_nb.p, row_nb.data(), rows * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in.p, lightning.data(), lightning.size() * 4,
                          cudaMemcpyHostToDevice));
    lc::launch_v4_prefill_indices(static_cast<int*>(d_out.p),
                                  static_cast<const int*>(d_in.p),
                                  static_cast<const int*>(d_nb.p), rows,
                                  topk, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> got(lightning.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_out.p, got.size() * 4,
                          cudaMemcpyDeviceToHost));
    for (int r = 0; r < 4; ++r)
        for (int j = 0; j < topk; ++j)
            EXPECT_EQ(got[size_t(r) * topk + j], j < row_nb[r] ? j : -1)
                << "r=" << r << " j=" << j;
    for (int j = 0; j < 63; ++j)
        EXPECT_EQ(got[size_t(4) * topk + j], 1000 + j) << j;
    EXPECT_EQ(got[size_t(4) * topk + 63], -1);

    // Null lightning input → IOTA for every row (incl. over-budget rows).
    lc::launch_v4_prefill_indices(static_cast<int*>(d_out.p), nullptr,
                                  static_cast<const int*>(d_nb.p), rows,
                                  topk, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got.data(), d_out.p, got.size() * 4,
                          cudaMemcpyDeviceToHost));
    for (int j = 0; j < topk; ++j)
        EXPECT_EQ(got[size_t(4) * topk + j], j) << j;  // 65 > 64 → all iota
}

// SC (superchunk port): per-row SWA index list over the chunk staging.
TEST(V4Prep, PrefillSwaBlockTable) {
    REQUIRES_GPU();
    const int rows = 4, W = 8, W_pref = 8;
    // swa_len per row: full windows and a partial one.
    const std::vector<int> swa_len = {8, 8, 3, 1};
    DeviceBuf d_len(rows * 4), d_bt(size_t(rows) * W * 4);
    CUDA_CHECK(cudaMemcpy(d_len.p, swa_len.data(), rows * 4,
                          cudaMemcpyHostToDevice));
    lc::launch_v4_prefill_swa_bt(static_cast<int*>(d_bt.p),
                                 static_cast<const int*>(d_len.p), rows, W,
                                 W_pref, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> got(size_t(rows) * W);
    CUDA_CHECK(cudaMemcpy(got.data(), d_bt.p, got.size() * 4,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < W; ++j) {
            const int want = j < swa_len[i]
                ? i - swa_len[i] + 1 + j + W_pref : -1;
            EXPECT_EQ(got[size_t(i) * W + j], want)
                << "i=" << i << " j=" << j;
        }
}

// ── V4-5T (TD-V4-TQ-DEVICE): TQ codec seams ──────────────────────────────

// kBf16Rows: the compress out-mode feeding the TQ pack must emit the SAME
// pooled+normed+roped vector the FP8 entry quantizes — compared against the
// CPU reference at BF16-rounding tolerance (no FP8 quant in this path).
TEST(V4Prep, CompressBf16RowsMatchesReference) {
    REQUIRES_GPU();
    const bool overlap = true;
    const int stride = 4, D = 512, first_block = 0, num_blocks = 5;
    const int state_dim = 2 * D;
    const int last_token = (first_block + num_blocks) * stride - 1;
    const int T = last_token + 1;
    const int max_pos = T + stride;
    std::mt19937 rng(202);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> kv_state(size_t(T) * state_dim),
        score_state(size_t(T) * state_dim), ape(size_t(stride) * state_dim),
        norm_w(D);
    for (auto& v : kv_state) v = bf16_round(dist(rng));
    for (auto& v : score_state) v = bf16_round(dist(rng));
    for (auto& v : ape) v = dist(rng) * 0.5f;
    for (auto& v : norm_w) v = 1.0f + 0.1f * dist(rng);
    auto tab = make_cos_sin(max_pos, kRopeDim, 160000.0);
    std::vector<int> slots(num_blocks);
    for (int b = 0; b < num_blocks; ++b) slots[b] = first_block + b;

    auto kv_bf = to_bf16(kv_state);
    auto sc_bf = to_bf16(score_state);
    DeviceBuf d_kv(kv_bf.size() * 2), d_sc(sc_bf.size() * 2),
        d_ape(ape.size() * 4), d_norm(norm_w.size() * 4),
        d_tab(tab.size() * 4), d_slots(size_t(num_blocks) * 4),
        d_rows(size_t(num_blocks) * D * 2),
        d_rope_rows(size_t(num_blocks) * kRopeDim * 2);
    CUDA_CHECK(cudaMemcpy(d_kv.p, kv_bf.data(), kv_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sc.p, sc_bf.data(), sc_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ape.p, ape.data(), ape.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm.p, norm_w.data(), norm_w.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tab.p, tab.data(), tab.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots.p, slots.data(), size_t(num_blocks) * 4,
                          cudaMemcpyHostToDevice));

    lc::V4CompressArgs args;
    args.kv_state = d_kv.p;
    args.score_state = d_sc.p;
    args.ring_capacity = T;
    args.state_dim = state_dim;
    args.overlap = overlap;
    args.stride = stride;
    args.ape = d_ape.p;
    args.norm_w = d_norm.p;
    args.cos_sin = d_tab.p;
    args.rms_eps = kRmsEps;
    args.D = D;
    args.rope_dim = kRopeDim;
    args.first_block = first_block;
    args.num_blocks = num_blocks;
    args.slots = static_cast<const int*>(d_slots.p);
    args.out_mode = lc::V4CompressArgs::Out::kBf16Rows;
    args.bf16_rows = d_rows.p;
    args.bf16_rope_rows = d_rope_rows.p;
    lc::launch_v4_compress_insert(args, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> rows(size_t(num_blocks) * D),
        rope_rows(size_t(num_blocks) * kRopeDim);
    CUDA_CHECK(cudaMemcpy(rows.data(), d_rows.p, rows.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rope_rows.data(), d_rope_rows.p,
                          rope_rows.size() * 2, cudaMemcpyDeviceToHost));
    for (int b = 0; b < num_blocks; ++b) {
        const int j = first_block + b;
        auto ref = ref_compress(kv_state, score_state, ape, norm_w, tab,
                                overlap, stride, D, kRopeDim, j, state_dim);
        for (int c = 0; c < D; ++c) {
            EXPECT_NEAR(__bfloat162float(rows[size_t(b) * D + c]), ref[c],
                        std::abs(ref[c]) * 0.02 + 0.02)
                << "blk " << j << " c=" << c;
        }
        for (int i = 0; i < kRopeDim; ++i) {
            EXPECT_NEAR(
                __bfloat162float(rope_rows[size_t(b) * kRopeDim + i]),
                ref[D - kRopeDim + i],
                std::abs(ref[D - kRopeDim + i]) * 0.02 + 0.02)
                << "blk " << j << " rope i=" << i;
        }
    }
}

// launch_v4_lse_merge2: exact LSE-weighted fold of two softmax partials
// over disjoint key sets (natural-log units), incl. the empty-side
// (-inf-LSE) weight-0 case — CPU double reference.
TEST(V4Prep, LseMerge2MatchesReference) {
    REQUIRES_GPU();
    const int rows = 3, heads = 4, dv = 512;
    const int rh = rows * heads;
    std::mt19937 rng(203);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> oa(size_t(rh) * dv), ob(size_t(rh) * dv);
    std::vector<float> la(rh), lb(rh);
    for (auto& v : oa) v = bf16_round(dist(rng));
    for (auto& v : ob) v = bf16_round(dist(rng));
    for (int i = 0; i < rh; ++i) {
        la[i] = dist(rng) * 3.0f;
        lb[i] = dist(rng) * 3.0f;
    }
    lb[1] = -INFINITY;   // empty compressed side for one (row, head)
    la[2] = -INFINITY;   // empty SWA side (never happens live; math holds)

    auto oa_bf = to_bf16(oa);
    auto ob_bf = to_bf16(ob);
    DeviceBuf d_oa(oa_bf.size() * 2), d_ob(ob_bf.size() * 2),
        d_la(size_t(rh) * 4), d_lb(size_t(rh) * 4);
    CUDA_CHECK(cudaMemcpy(d_oa.p, oa_bf.data(), oa_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ob.p, ob_bf.data(), ob_bf.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_la.p, la.data(), size_t(rh) * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lb.p, lb.data(), size_t(rh) * 4,
                          cudaMemcpyHostToDevice));

    lc::V4LseMerge2Args ma;
    ma.out_a = d_oa.p;
    ma.lse_a = static_cast<float*>(d_la.p);
    ma.out_b = d_ob.p;
    ma.lse_b = static_cast<const float*>(d_lb.p);
    ma.rows = rows;
    ma.heads = heads;
    ma.head_dim = dv;
    lc::launch_v4_lse_merge2(ma, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> got(size_t(rh) * dv);
    std::vector<float> got_lse(rh);
    CUDA_CHECK(cudaMemcpy(got.data(), d_oa.p, got.size() * 2,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_lse.data(), d_la.p, size_t(rh) * 4,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < rh; ++i) {
        const double m = std::max(la[i], lb[i]);
        const double wa = std::isinf(la[i]) ? 0.0 : std::exp(la[i] - m);
        const double wb = std::isinf(lb[i]) ? 0.0 : std::exp(lb[i] - m);
        const double inv = 1.0 / (wa + wb);
        EXPECT_NEAR(got_lse[i], m + std::log(wa + wb), 1e-4) << "lse " << i;
        for (int c = 0; c < dv; ++c) {
            const double want =
                (wa * oa[size_t(i) * dv + c] + wb * ob[size_t(i) * dv + c]) *
                inv;
            EXPECT_NEAR(__bfloat162float(got[size_t(i) * dv + c]), want,
                        std::abs(want) * 0.02 + 0.02)
                << "i=" << i << " c=" << c;
        }
    }
}
