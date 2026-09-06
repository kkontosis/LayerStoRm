// kda_reference_test.cpp — GF3.6: KDA reference scan + goldens (glm5_next).
//
// What is being pinned here (PLAN.md GF3.6):
//   1. chunked ≡ recurrent — the agreement gate every later KDA item (GF3.7
//      SM120 kernels, GF3.8 state pool, GF3.9 hybrid execution) is judged
//      against, across chunk-aligned and arbitrary shapes.
//   2. Cross-implementation goldens: tools/gen_kda_golden.py is an
//      INDEPENDENT NumPy transcription of the vLLM recurrent kernel; its
//      outputs are committed under test-data/kda-goldens/ and this test must
//      reproduce them. Two golden cases:
//        synth_*  — fully synthetic inputs (splitmix64 streams, bf16-rounded),
//        real_*   — REAL GLM-5.3-Flash layer-0 KDA tensors (A_log, dt_bias,
//                   q/k/v conv kernels, o_norm.weight, f_b/g_b low-rank gate
//                   mats fetched from HF rev 04c4e9e9; see
//                   test-data/GLM-5.3-Flash/kda-layer0/manifest.json) driving
//                   the real gate dynamics, with synthetic activations.
//   3. State semantics GF3.8/GF3.11/GF3.12 lean on:
//        - checkpoint/replay at token boundaries is EXACT (bitwise on the
//          same path; chunk boundaries are bitwise even for the chunked path);
//        - the state has NO position axis: backward reconstruction is
//          exponentially ill-conditioned even with every token input in hand
//          (measured here), so rewind is anchor-and-replay, never
//          position-addressed (the KDA analogue of INV-DSA-REWIND).
//
// Everything is CPU fp32 (double in the rewind demonstration). No GPU.

#include "kda_reference.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using namespace layerstorm::kda_ref;

// ── fixture I/O ─────────────────────────────────────────────────────────────

std::string resolve_test_data(const std::string& rel) {
    namespace fs = std::filesystem;
    const std::vector<std::string> candidates = {
        rel,
        "../" + rel,
        "../../" + rel,
#ifdef LAYERSTORM_SOURCE_DIR
        std::string(LAYERSTORM_SOURCE_DIR) + "/" + rel,
#endif
    };
    for (const auto& c : candidates) {
        if (fs::exists(c)) return c;
    }
    return rel;  // let the open fail with the raw name in the message
}

std::vector<float> load_f32(const std::string& rel, size_t expect_count) {
    const std::string path = resolve_test_data(rel);
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    EXPECT_TRUE(f.good()) << "missing fixture: " << path;
    if (!f.good()) return {};
    const size_t bytes = static_cast<size_t>(f.tellg());
    EXPECT_EQ(bytes, expect_count * 4) << path;
    std::vector<float> v(expect_count);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
    return v;
}

std::vector<float> load_bf16_as_f32(const std::string& rel, size_t expect_count) {
    const std::string path = resolve_test_data(rel);
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    EXPECT_TRUE(f.good()) << "missing fixture: " << path;
    if (!f.good()) return {};
    const size_t bytes = static_cast<size_t>(f.tellg());
    EXPECT_EQ(bytes, expect_count * 2) << path;
    std::vector<uint16_t> raw(expect_count);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
    std::vector<float> v(expect_count);
    for (size_t i = 0; i < expect_count; ++i) v[i] = bf16_to_f32(raw[i]);
    return v;
}

// ── deterministic inputs (must match tools/gen_kda_golden.py exactly) ───────

std::vector<float> synth(uint64_t seed, size_t n, float lo, float hi, bool round_bf16) {
    std::vector<float> v(n);
    uint64_t s = seed;
    for (auto& x : v) {
        x = splitmix_uniform(s, lo, hi);
        if (round_bf16) x = bf16_round(x);
    }
    return v;
}

// ── comparison ──────────────────────────────────────────────────────────────

struct Closeness {
    double max_abs = 0, max_rel_excess = 0;
    size_t worst = 0;
};

// Scale-aware bound: |a_i - b_i| <= frac_scale * max|b| + frac_elem * |b_i|.
// The golden tensors span 4+ orders of magnitude across cases (the real-tensor
// case has tiny core outputs), so a flat atol would either mask errors or
// false-fail; anchoring the absolute term to the tensor's own scale keeps the
// pin equally tight everywhere. All bounds below were set from a measured
// probe (2026-08-29) at ~10x the observed worst-case deviation, so they
// absorb cross-host libm/summation-order variation without masking real
// errors.
Closeness closeness(const std::vector<float>& a, const std::vector<float>& b,
                    double frac_scale, double frac_elem) {
    Closeness c;
    EXPECT_EQ(a.size(), b.size());
    double scale = 0;
    for (float x : b) scale = std::max(scale, std::abs(double(x)));
    if (scale == 0) scale = 1e-30;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        const double d = std::abs(double(a[i]) - double(b[i]));
        const double bound = frac_scale * scale + frac_elem * std::abs(double(b[i]));
        if (d > c.max_abs) c.max_abs = d;
        const double excess = d / bound;
        if (excess > c.max_rel_excess) {
            c.max_rel_excess = excess;
            c.worst = i;
        }
    }
    return c;
}

void expect_close(const std::vector<float>& a, const std::vector<float>& b, double frac_scale,
                  double frac_elem, const char* label) {
    const Closeness c = closeness(a, b, frac_scale, frac_elem);
    EXPECT_LE(c.max_rel_excess, 1.0)
        << label << ": max_abs_diff=" << c.max_abs << " at flat index " << c.worst
        << " (a=" << (c.worst < a.size() ? a[c.worst] : 0.f)
        << " b=" << (c.worst < b.size() ? b[c.worst] : 0.f)
        << ", frac_scale=" << frac_scale << ", frac_elem=" << frac_elem << ")";
}

// Common synthetic-scan input bundle for the self-consistency tests.
struct ScanInputs {
    std::vector<float> q, k, v, raw_g, g2, beta, a_log, dt_bias;
    std::vector<float> conv_wq, conv_wk, conv_wv, onorm_w, s0, cs_q, cs_k, cs_v;
};

ScanInputs make_inputs(const KdaParams& p, int t_len, uint64_t seed_base) {
    ScanInputs in;
    const size_t n = static_cast<size_t>(t_len) * p.channels();
    const size_t c = p.channels();
    in.q = synth(seed_base + 0, n, -1.0f, 1.0f, true);
    in.k = synth(seed_base + 1, n, -1.0f, 1.0f, true);
    in.v = synth(seed_base + 2, n, -1.0f, 1.0f, true);
    in.raw_g = synth(seed_base + 3, n, -3.0f, 3.0f, true);
    in.g2 = synth(seed_base + 4, n, -2.0f, 2.0f, true);
    in.beta = synth(seed_base + 5, static_cast<size_t>(t_len) * p.num_heads, -2.0f, 2.0f, true);
    in.a_log = synth(seed_base + 6, p.num_heads, -1.0f, 1.5f, false);
    in.dt_bias = synth(seed_base + 7, c, -1.0f, 1.0f, false);
    in.conv_wq = synth(seed_base + 8, c * p.conv_width, -0.5f, 0.5f, true);
    in.conv_wk = synth(seed_base + 9, c * p.conv_width, -0.5f, 0.5f, true);
    in.conv_wv = synth(seed_base + 10, c * p.conv_width, -0.5f, 0.5f, true);
    in.onorm_w = synth(seed_base + 11, p.head_dim, 0.5f, 1.5f, true);
    in.s0 = synth(seed_base + 12, static_cast<size_t>(p.num_heads) * p.head_dim * p.head_dim,
                  -1.0f, 1.0f, false);
    in.cs_q = synth(seed_base + 13, c * (p.conv_width - 1), -1.0f, 1.0f, true);
    in.cs_k = synth(seed_base + 14, c * (p.conv_width - 1), -1.0f, 1.0f, true);
    in.cs_v = synth(seed_base + 15, c * (p.conv_width - 1), -1.0f, 1.0f, true);
    return in;
}

struct ScanResult {
    std::vector<float> core, onorm;
    KdaState st;
    ConvState cq, ck, cv;
};

ScanResult run_scan(const KdaParams& p, ScanPath path, const ScanInputs& in, int t_len) {
    ScanResult r;
    const size_t n = static_cast<size_t>(t_len) * p.channels();
    r.core.assign(n, 0.0f);
    r.onorm.assign(n, 0.0f);
    r.st = KdaState::zeros(p);
    r.st.s = in.s0;
    r.cq = ConvState::zeros(p);
    r.cq.ring = in.cs_q;
    r.ck = ConvState::zeros(p);
    r.ck.ring = in.cs_k;
    r.cv = ConvState::zeros(p);
    r.cv.ring = in.cs_v;
    kda_scan(p, path, in.q.data(), in.k.data(), in.v.data(), in.raw_g.data(), in.g2.data(),
             in.beta.data(), in.a_log.data(), in.dt_bias.data(), in.conv_wq.data(),
             in.conv_wk.data(), in.conv_wv.data(), in.onorm_w.data(), t_len, r.st, r.cq,
             r.ck, r.cv, r.core.data(), r.onorm.data());
    return r;
}

// ── tests ───────────────────────────────────────────────────────────────────

// The C++ splitmix64/bf16 input machinery must be bit-identical to the Python
// golden generator's, or every golden comparison below is vacuously wrong.
TEST(KdaReference, PrngMatchesGoldenGenerator) {
    uint64_t s = 101;
    EXPECT_EQ(splitmix64(s), 15060681878671775511ull);
    EXPECT_EQ(splitmix64(s), 317125338075985571ull);
    EXPECT_EQ(splitmix64(s), 4955761374856264282ull);
    EXPECT_EQ(splitmix64(s), 10230203001953340975ull);

    s = 101;
    const float expect[4] = {0.6328125f, -0.96484375f, -0.462890625f, 0.109375f};
    for (float e : expect) {
        EXPECT_EQ(bf16_round(splitmix_uniform(s, -1.0f, 1.0f)), e);
    }
}

// Gate direction: very negative raw gate => decay ~1 (remember); very positive
// => decay ~e^-5 (max forgetting). All log-decays strictly inside (LB, 0).
TEST(KdaReference, GateBoundsAndDirection) {
    const int d = 8;
    std::vector<float> raw(d), bias(d, 0.0f), g(d);
    for (int c = 0; c < d; ++c) raw[c] = -12.0f + 3.0f * c;  // -12 .. +9
    safe_gate_log_decay<float>(raw.data(), bias.data(), 0.3f, -5.0f, d, g.data());
    for (int c = 0; c < d; ++c) {
        EXPECT_GT(g[c], -5.0f);
        EXPECT_LT(g[c], 0.0f);
        if (c > 0) EXPECT_LT(g[c], g[c - 1]);  // monotone: larger raw => more forgetting
    }
    EXPECT_GT(std::exp(g[0]), 0.99f);          // remember end
    EXPECT_LT(std::exp(g[d - 1]), 0.0070f);    // ~e^-5 = 0.0067 forgetting end
}

// The conv is a pure per-channel ring: splitting a prefill anywhere and
// carrying the state is bitwise-identical to the one-shot conv.
TEST(KdaReference, ConvSplitEquivalence) {
    KdaParams p;
    p.num_heads = 2;
    p.head_dim = 16;
    const int t_len = 29, split = 17;
    const size_t n = static_cast<size_t>(t_len) * p.channels();
    auto x = synth(7001, n, -1.0f, 1.0f, true);
    auto w = synth(7002, static_cast<size_t>(p.channels()) * p.conv_width, -0.5f, 0.5f, true);
    auto s0 = synth(7003, static_cast<size_t>(p.channels()) * (p.conv_width - 1), -1.0f, 1.0f,
                    true);

    ConvState one = ConvState::zeros(p);
    one.ring = s0;
    std::vector<float> out_one(n);
    conv_prefill(p, x.data(), t_len, w.data(), one, out_one.data());

    ConvState two = ConvState::zeros(p);
    two.ring = s0;
    std::vector<float> out_two(n);
    conv_prefill(p, x.data(), split, w.data(), two, out_two.data());
    conv_prefill(p, x.data() + static_cast<size_t>(split) * p.channels(), t_len - split,
                 w.data(), two, out_two.data() + static_cast<size_t>(split) * p.channels());

    for (size_t i = 0; i < n; ++i) ASSERT_EQ(out_one[i], out_two[i]) << i;
    for (size_t i = 0; i < one.ring.size(); ++i) ASSERT_EQ(one.ring[i], two.ring[i]) << i;
}

// ── chunked ≡ recurrent (the agreement gate) ────────────────────────────────

void chunked_equals_recurrent(int heads, int dim, int t_len, uint64_t seed) {
    KdaParams p;
    p.num_heads = heads;
    p.head_dim = dim;
    ScanInputs in = make_inputs(p, t_len, seed);
    ScanResult rec = run_scan(p, ScanPath::kRecurrent, in, t_len);
    ScanResult chk = run_scan(p, ScanPath::kChunked, in, t_len);
    char label[128];
    std::snprintf(label, sizeof(label), "core H=%d D=%d T=%d", heads, dim, t_len);
    expect_close(chk.core, rec.core, 2e-5, 2e-4, label);
    std::snprintf(label, sizeof(label), "onorm H=%d D=%d T=%d", heads, dim, t_len);
    expect_close(chk.onorm, rec.onorm, 2e-5, 2e-4, label);
    std::snprintf(label, sizeof(label), "state H=%d D=%d T=%d", heads, dim, t_len);
    expect_close(chk.st.s, rec.st.s, 2e-5, 2e-4, label);
    // Conv states are computed identically on both paths.
    for (size_t i = 0; i < rec.cq.ring.size(); ++i) {
        ASSERT_EQ(chk.cq.ring[i], rec.cq.ring[i]);
        ASSERT_EQ(chk.ck.ring[i], rec.ck.ring[i]);
        ASSERT_EQ(chk.cv.ring[i], rec.cv.ring[i]);
    }
}

TEST(KdaReference, ChunkedEqualsRecurrentSingleChunk) {
    chunked_equals_recurrent(2, 32, 64, 11000);
}

TEST(KdaReference, ChunkedEqualsRecurrentPartialChunks) {
    chunked_equals_recurrent(2, 64, 150, 12000);  // 64 + 64 + 22
    chunked_equals_recurrent(2, 64, 1, 13000);
    chunked_equals_recurrent(2, 64, 2, 14000);
    chunked_equals_recurrent(2, 64, 63, 15000);
}

TEST(KdaReference, ChunkedEqualsRecurrentModelGeometry) {
    // The shipped geometry per head: D=128 (H reduced — heads are independent).
    chunked_equals_recurrent(4, 128, 144, 16000);
}

// Chunk boundaries are STATE-CARRY points: splitting the chunked scan at a
// chunk boundary and carrying (recurrent state, conv states) is bitwise
// identical to the one-shot chunked run. At an arbitrary (non-aligned)
// boundary the chunk grid changes, so equality is up to fp reassociation
// only — both are asserted, because GF3.9's prefill will cut at superchunk
// boundaries (multiples of 64) and GF3.12's checkpoints must be exact there.
TEST(KdaReference, ScanSplitBoundaries) {
    KdaParams p;
    p.num_heads = 2;
    p.head_dim = 64;
    const int t_len = 150;
    ScanInputs in = make_inputs(p, t_len, 17000);
    ScanResult one = run_scan(p, ScanPath::kChunked, in, t_len);

    auto run_split = [&](int split) {
        ScanResult r;
        const size_t n = static_cast<size_t>(t_len) * p.channels();
        r.core.assign(n, 0.0f);
        r.onorm.assign(n, 0.0f);
        r.st = KdaState::zeros(p);
        r.st.s = in.s0;
        r.cq.ring = in.cs_q;
        r.ck.ring = in.cs_k;
        r.cv.ring = in.cs_v;
        const size_t tok = p.channels();
        const size_t off = static_cast<size_t>(split) * tok;
        const size_t boff = static_cast<size_t>(split) * p.num_heads;
        kda_scan(p, ScanPath::kChunked, in.q.data(), in.k.data(), in.v.data(),
                 in.raw_g.data(), in.g2.data(), in.beta.data(), in.a_log.data(),
                 in.dt_bias.data(), in.conv_wq.data(), in.conv_wk.data(), in.conv_wv.data(),
                 in.onorm_w.data(), split, r.st, r.cq, r.ck, r.cv, r.core.data(),
                 r.onorm.data());
        kda_scan(p, ScanPath::kChunked, in.q.data() + off, in.k.data() + off,
                 in.v.data() + off, in.raw_g.data() + off, in.g2.data() + off,
                 in.beta.data() + boff, in.a_log.data(), in.dt_bias.data(),
                 in.conv_wq.data(), in.conv_wk.data(), in.conv_wv.data(), in.onorm_w.data(),
                 t_len - split, r.st, r.cq, r.ck, r.cv, r.core.data() + off,
                 r.onorm.data() + off);
        return r;
    };

    {  // chunk-aligned: bitwise
        ScanResult sp = run_split(128);
        for (size_t i = 0; i < one.core.size(); ++i) {
            ASSERT_EQ(sp.core[i], one.core[i]) << "core (aligned split) @" << i;
        }
        for (size_t i = 0; i < one.st.s.size(); ++i) {
            ASSERT_EQ(sp.st.s[i], one.st.s[i]) << "state (aligned split) @" << i;
        }
    }
    {  // arbitrary: equal up to fp reassociation (grid shifts)
        ScanResult sp = run_split(37);
        expect_close(sp.core, one.core, 2e-5, 2e-4, "core (split @37)");
        expect_close(sp.onorm, one.onorm, 2e-5, 2e-4, "onorm (split @37)");
        expect_close(sp.st.s, one.st.s, 2e-5, 2e-4, "state (split @37)");
    }
}

// Chunked prefill hands its state to the recurrent decode path: prefill(100)
// + 10 decode steps must equal the pure recurrent run of 110 tokens.
TEST(KdaReference, PrefillThenDecodeContinuity) {
    KdaParams p;
    p.num_heads = 2;
    p.head_dim = 64;
    const int t_pre = 100, t_dec = 10, t_len = t_pre + t_dec;
    ScanInputs in = make_inputs(p, t_len, 18000);
    ScanResult ref = run_scan(p, ScanPath::kRecurrent, in, t_len);

    // Chunked prefill.
    ScanResult r;
    const size_t n = static_cast<size_t>(t_len) * p.channels();
    r.core.assign(n, 0.0f);
    r.onorm.assign(n, 0.0f);
    r.st = KdaState::zeros(p);
    r.st.s = in.s0;
    r.cq.ring = in.cs_q;
    r.ck.ring = in.cs_k;
    r.cv.ring = in.cs_v;
    kda_scan(p, ScanPath::kChunked, in.q.data(), in.k.data(), in.v.data(), in.raw_g.data(),
             in.g2.data(), in.beta.data(), in.a_log.data(), in.dt_bias.data(),
             in.conv_wq.data(), in.conv_wk.data(), in.conv_wv.data(), in.onorm_w.data(),
             t_pre, r.st, r.cq, r.ck, r.cv, r.core.data(), r.onorm.data());
    // Decode steps through the step API (exactly what the decode kernel does).
    const size_t tok = p.channels();
    std::vector<float> qc(tok), kc(tok), vc(tok);
    for (int t = t_pre; t < t_len; ++t) {
        const size_t off = static_cast<size_t>(t) * tok;
        conv_step(p, in.q.data() + off, in.conv_wq.data(), r.cq, qc.data());
        conv_step(p, in.k.data() + off, in.conv_wk.data(), r.ck, kc.data());
        conv_step(p, in.v.data() + off, in.conv_wv.data(), r.cv, vc.data());
        recurrent_step(p, qc.data(), kc.data(), vc.data(), in.raw_g.data() + off,
                       in.beta.data() + static_cast<size_t>(t) * p.num_heads,
                       in.a_log.data(), in.dt_bias.data(), r.st, r.core.data() + off);
        gated_rmsnorm(p, r.core.data() + off, in.g2.data() + off, in.onorm_w.data(), 1,
                      r.onorm.data() + off);
    }
    expect_close(r.core, ref.core, 2e-5, 2e-4, "prefill+decode core");
    expect_close(r.onorm, ref.onorm, 2e-5, 2e-4, "prefill+decode onorm");
    expect_close(r.st.s, ref.st.s, 2e-5, 2e-4, "prefill+decode state");
}

// Forward checkpoint/replay is EXACT: capture (S, conv rings) at any token
// boundary on the recurrent path and resume — bitwise identical continuation.
// This is the property GF3.12's state checkpoints and GF3.11's
// anchor-and-replay rewind rest on.
TEST(KdaReference, CheckpointReplayIsExact) {
    KdaParams p;
    p.num_heads = 2;
    p.head_dim = 64;
    const int t_len = 150, anchor = 83;  // deliberately not chunk-aligned
    ScanInputs in = make_inputs(p, t_len, 19000);
    ScanResult one = run_scan(p, ScanPath::kRecurrent, in, t_len);

    // First leg to the anchor, checkpoint, then replay the tail.
    ScanResult r;
    const size_t n = static_cast<size_t>(t_len) * p.channels();
    r.core.assign(n, 0.0f);
    r.onorm.assign(n, 0.0f);
    r.st = KdaState::zeros(p);
    r.st.s = in.s0;
    r.cq.ring = in.cs_q;
    r.ck.ring = in.cs_k;
    r.cv.ring = in.cs_v;
    kda_scan(p, ScanPath::kRecurrent, in.q.data(), in.k.data(), in.v.data(), in.raw_g.data(),
             in.g2.data(), in.beta.data(), in.a_log.data(), in.dt_bias.data(),
             in.conv_wq.data(), in.conv_wk.data(), in.conv_wv.data(), in.onorm_w.data(),
             anchor, r.st, r.cq, r.ck, r.cv, r.core.data(), r.onorm.data());
    const KdaState snap_s = r.st;      // the checkpoint: plain copies
    const ConvState snap_q = r.cq, snap_k = r.ck, snap_v = r.cv;

    auto run_tail = [&](ScanResult& dst) {
        const size_t off = static_cast<size_t>(anchor) * p.channels();
        const size_t boff = static_cast<size_t>(anchor) * p.num_heads;
        kda_scan(p, ScanPath::kRecurrent, in.q.data() + off, in.k.data() + off,
                 in.v.data() + off, in.raw_g.data() + off, in.g2.data() + off,
                 in.beta.data() + boff, in.a_log.data(), in.dt_bias.data(),
                 in.conv_wq.data(), in.conv_wk.data(), in.conv_wv.data(), in.onorm_w.data(),
                 t_len - anchor, dst.st, dst.cq, dst.ck, dst.cv, dst.core.data() + off,
                 dst.onorm.data() + off);
    };
    run_tail(r);  // continue in place

    ScanResult replay;
    replay.core.assign(n, 0.0f);
    replay.onorm.assign(n, 0.0f);
    replay.st = snap_s;
    replay.cq = snap_q;
    replay.ck = snap_k;
    replay.cv = snap_v;
    run_tail(replay);

    for (size_t i = static_cast<size_t>(anchor) * p.channels(); i < n; ++i) {
        ASSERT_EQ(replay.core[i], r.core[i]) << "replay core @" << i;
        ASSERT_EQ(replay.onorm[i], r.onorm[i]) << "replay onorm @" << i;
    }
    for (size_t i = 0; i < r.st.s.size(); ++i) ASSERT_EQ(replay.st.s[i], r.st.s[i]);
    // And the continued run equals the uninterrupted one bitwise (same path).
    for (size_t i = 0; i < n; ++i) ASSERT_EQ(r.core[i], one.core[i]);
    for (size_t i = 0; i < r.st.s.size(); ++i) ASSERT_EQ(r.st.s[i], one.st.s[i]);
}

// The no-position-axis demonstration. One backward step is algebraically
// invertible when every token input is in hand, so we invert — in DOUBLE —
// and watch the reconstruction blow up: each channel amplifies error by
// e^{-g_log} (up to e^5 ≈ 148) and the delta inversion by 1/(1-βk·k).
// Contrast: forward replay from an anchor is bitwise exact. This is why
// glm5_next reports lossy_position_indexed_state() == true, why truncating
// forks are rejected, and why speculative rewind must be anchor-and-replay
// (GF3.11) — there is no cheap "drop n rows" analogue.
TEST(KdaReference, RewindIsNotPositionAddressable) {
    const int d = 64, t_len = 96;
    const double lb = -5.0, l2eps = 1e-6, scale = 1.0 / std::sqrt(double(d));
    const double a_log = 0.25;
    std::vector<double> dt_bias(d);
    {
        auto tmp = synth(21001, d, -1.0f, 1.0f, false);
        for (int c = 0; c < d; ++c) dt_bias[c] = tmp[c];
    }
    auto f32 = [&](uint64_t seed, size_t n, float lo, float hi) {
        auto tmp = synth(seed, n, lo, hi, true);
        return std::vector<double>(tmp.begin(), tmp.end());
    };
    auto q = f32(21002, static_cast<size_t>(t_len) * d, -1, 1);
    auto k = f32(21003, static_cast<size_t>(t_len) * d, -1, 1);
    auto v = f32(21004, static_cast<size_t>(t_len) * d, -1, 1);
    auto rg = f32(21005, static_cast<size_t>(t_len) * d, -3, 3);
    auto beta = f32(21006, t_len, -2, 2);

    // Forward in double, storing the state after every token.
    std::vector<std::vector<double>> states;
    states.reserve(t_len);
    std::vector<double> s(static_cast<size_t>(d) * d, 0.0), core(d);
    for (int t = 0; t < t_len; ++t) {
        recurrent_step_head<double>(d, lb, l2eps, scale, &q[static_cast<size_t>(t) * d],
                                    &k[static_cast<size_t>(t) * d],
                                    &v[static_cast<size_t>(t) * d],
                                    &rg[static_cast<size_t>(t) * d], beta[t], a_log,
                                    dt_bias.data(), s.data(), core.data());
        states.push_back(s);
    }
    double state_mag = 0;
    for (double x : states.back()) state_mag = std::max(state_mag, std::abs(x));

    // Backward reconstruction from the final state, with FULL token knowledge.
    std::vector<double> inv = states.back();
    auto max_err_vs = [&](const std::vector<double>& truth) {
        double e = 0;
        for (size_t i = 0; i < truth.size(); ++i) e = std::max(e, std::abs(inv[i] - truth[i]));
        return e;
    };
    double err1 = -1, err32 = -1;
    for (int back = 1; back <= 32; ++back) {
        const int t = t_len - back;  // token being inverted
        inverse_step_head<double>(d, lb, l2eps, &k[static_cast<size_t>(t) * d],
                                  &v[static_cast<size_t>(t) * d],
                                  &rg[static_cast<size_t>(t) * d], beta[t], a_log,
                                  dt_bias.data(), inv.data());
        const double e = max_err_vs(states[static_cast<size_t>(t) - 1]);
        if (back == 1) err1 = e;
        if (back == 32) err32 = e;
    }
    // One step back: essentially recovered (double precision, well-posed).
    EXPECT_LT(err1, 1e-8) << "single-step algebraic inversion should be near-exact";
    // 32 steps back: hopeless — error dwarfs the state itself by orders of
    // magnitude. THIS is the lossy-state property in one number.
    EXPECT_GT(err32, 1e3 * state_mag);
    EXPECT_GT(err32, 1e6 * std::max(err1, 1e-300));

    // Contrast: forward replay of the same 32 tokens from the t=63 anchor is
    // bitwise identical to the original forward pass.
    std::vector<double> replay = states[t_len - 33];
    for (int t = t_len - 32; t < t_len; ++t) {
        recurrent_step_head<double>(d, lb, l2eps, scale, &q[static_cast<size_t>(t) * d],
                                    &k[static_cast<size_t>(t) * d],
                                    &v[static_cast<size_t>(t) * d],
                                    &rg[static_cast<size_t>(t) * d], beta[t], a_log,
                                    dt_bias.data(), replay.data(), core.data());
    }
    for (size_t i = 0; i < replay.size(); ++i) ASSERT_EQ(replay[i], states.back()[i]);
}

// The golden cases' core outputs are small enough that o_norm's rms is
// eps-dominated (see test-data/kda-goldens/manifest.json caveats), so the
// normalization DIVISION is only weakly exercised there. Pin it here at O(1)
// magnitudes against a double-precision recomputation.
TEST(KdaReference, GatedRmsNormFormulaAtUnitScale) {
    const int d = 128;
    auto core = synth(31001, d, -2.0f, 2.0f, false);
    auto g2 = synth(31002, d, -2.0f, 2.0f, false);
    auto w = synth(31003, d, 0.5f, 1.5f, false);
    std::vector<float> out(d);
    gated_rmsnorm_head(d, 1e-5f, core.data(), g2.data(), w.data(), out.data());
    double ss = 0;
    for (int v = 0; v < d; ++v) ss += double(core[v]) * core[v];
    const double r = 1.0 / std::sqrt(ss / d + 1e-5);
    EXPECT_GT(ss / d, 0.5);  // genuinely NOT eps-dominated at this scale
    for (int v = 0; v < d; ++v) {
        const double expect = double(core[v]) * r * w[v] / (1.0 + std::exp(-double(g2[v])));
        EXPECT_NEAR(out[v], expect, 1e-5 * std::abs(expect) + 1e-6) << v;
    }
}

// ── goldens (cross-implementation, committed fixtures) ──────────────────────

struct GoldenSet {
    std::vector<float> prefill_core, prefill_onorm, state_after_prefill;
    std::vector<float> conv_q, conv_k, conv_v;
    std::vector<float> decode_core, decode_onorm, state_after_decode;
};

constexpr int kGoldT = 152, kGoldPre = 144, kGoldDec = 8, kGoldH = 4, kGoldD = 128;
constexpr size_t kGoldTok = static_cast<size_t>(kGoldH) * kGoldD;  // 512

GoldenSet load_goldens(const std::string& prefix) {
    const std::string base = "test-data/kda-goldens/" + prefix;
    GoldenSet g;
    g.prefill_core = load_f32(base + "prefill_core.bin", kGoldPre * kGoldTok);
    g.prefill_onorm = load_f32(base + "prefill_onorm.bin", kGoldPre * kGoldTok);
    g.state_after_prefill =
        load_f32(base + "state_after_prefill.bin", kGoldH * static_cast<size_t>(kGoldD) * kGoldD);
    g.conv_q = load_f32(base + "conv_state_q.bin", kGoldTok * 3);
    g.conv_k = load_f32(base + "conv_state_k.bin", kGoldTok * 3);
    g.conv_v = load_f32(base + "conv_state_v.bin", kGoldTok * 3);
    g.decode_core = load_f32(base + "decode_core.bin", kGoldDec * kGoldTok);
    g.decode_onorm = load_f32(base + "decode_onorm.bin", kGoldDec * kGoldTok);
    g.state_after_decode =
        load_f32(base + "state_after_decode.bin", kGoldH * static_cast<size_t>(kGoldD) * kGoldD);
    return g;
}

// Runs the 152-token golden protocol (144 prefill via `path`, 8 decode steps
// via the step API) and checks every output family against the fixture.
void run_golden_case(const ScanInputs& in, const GoldenSet& g, ScanPath prefill_path,
                     double atol, double rtol, const char* label) {
    KdaParams p;
    p.num_heads = kGoldH;
    p.head_dim = kGoldD;

    ScanResult r;
    const size_t n = static_cast<size_t>(kGoldT) * p.channels();
    r.core.assign(n, 0.0f);
    r.onorm.assign(n, 0.0f);
    r.st = KdaState::zeros(p);
    if (!in.s0.empty()) r.st.s = in.s0;
    r.cq = ConvState::zeros(p);
    r.ck = ConvState::zeros(p);
    r.cv = ConvState::zeros(p);
    if (!in.cs_q.empty()) r.cq.ring = in.cs_q;
    if (!in.cs_k.empty()) r.ck.ring = in.cs_k;
    if (!in.cs_v.empty()) r.cv.ring = in.cs_v;

    kda_scan(p, prefill_path, in.q.data(), in.k.data(), in.v.data(), in.raw_g.data(),
             in.g2.data(), in.beta.data(), in.a_log.data(), in.dt_bias.data(),
             in.conv_wq.data(), in.conv_wk.data(), in.conv_wv.data(), in.onorm_w.data(),
             kGoldPre, r.st, r.cq, r.ck, r.cv, r.core.data(), r.onorm.data());

    auto slice = [](const std::vector<float>& v, size_t a, size_t b) {
        return std::vector<float>(v.begin() + a, v.begin() + b);
    };
    std::string l;
    l = std::string(label) + " prefill_core";
    expect_close(slice(r.core, 0, kGoldPre * kGoldTok), g.prefill_core, atol, rtol, l.c_str());
    l = std::string(label) + " prefill_onorm";
    expect_close(slice(r.onorm, 0, kGoldPre * kGoldTok), g.prefill_onorm, atol, rtol,
                 l.c_str());
    l = std::string(label) + " state_after_prefill";
    expect_close(r.st.s, g.state_after_prefill, atol, rtol, l.c_str());
    l = std::string(label) + " conv_state";
    expect_close(r.cq.ring, g.conv_q, atol, rtol, l.c_str());
    expect_close(r.ck.ring, g.conv_k, atol, rtol, l.c_str());
    expect_close(r.cv.ring, g.conv_v, atol, rtol, l.c_str());

    // Decode leg through the step API.
    const size_t tok = p.channels();
    std::vector<float> qc(tok), kc(tok), vc(tok);
    for (int t = kGoldPre; t < kGoldT; ++t) {
        const size_t off = static_cast<size_t>(t) * tok;
        conv_step(p, in.q.data() + off, in.conv_wq.data(), r.cq, qc.data());
        conv_step(p, in.k.data() + off, in.conv_wk.data(), r.ck, kc.data());
        conv_step(p, in.v.data() + off, in.conv_wv.data(), r.cv, vc.data());
        recurrent_step(p, qc.data(), kc.data(), vc.data(), in.raw_g.data() + off,
                       in.beta.data() + static_cast<size_t>(t) * p.num_heads,
                       in.a_log.data(), in.dt_bias.data(), r.st, r.core.data() + off);
        gated_rmsnorm(p, r.core.data() + off, in.g2.data() + off, in.onorm_w.data(), 1,
                      r.onorm.data() + off);
    }
    l = std::string(label) + " decode_core";
    expect_close(slice(r.core, kGoldPre * kGoldTok, kGoldT * kGoldTok), g.decode_core, atol,
                 rtol, l.c_str());
    l = std::string(label) + " decode_onorm";
    expect_close(slice(r.onorm, kGoldPre * kGoldTok, kGoldT * kGoldTok), g.decode_onorm, atol,
                 rtol, l.c_str());
    l = std::string(label) + " state_after_decode";
    expect_close(r.st.s, g.state_after_decode, atol, rtol, l.c_str());
}

ScanInputs make_golden_synth_inputs() {
    KdaParams p;
    p.num_heads = kGoldH;
    p.head_dim = kGoldD;
    const size_t n = static_cast<size_t>(kGoldT) * p.channels();
    const size_t c = p.channels();
    ScanInputs in;
    in.q = synth(101, n, -1.0f, 1.0f, true);
    in.k = synth(102, n, -1.0f, 1.0f, true);
    in.v = synth(103, n, -1.0f, 1.0f, true);
    in.raw_g = synth(104, n, -3.0f, 3.0f, true);
    in.g2 = synth(105, n, -2.0f, 2.0f, true);
    in.beta = synth(106, static_cast<size_t>(kGoldT) * kGoldH, -2.0f, 2.0f, true);
    in.a_log = synth(107, kGoldH, -1.0f, 1.5f, false);
    in.dt_bias = synth(108, c, -1.0f, 1.0f, false);
    in.conv_wq = synth(109, c * 4, -0.5f, 0.5f, true);
    in.conv_wk = synth(110, c * 4, -0.5f, 0.5f, true);
    in.conv_wv = synth(111, c * 4, -0.5f, 0.5f, true);
    in.onorm_w = synth(112, kGoldD, 0.5f, 1.5f, true);
    in.s0 = synth(113, static_cast<size_t>(kGoldH) * kGoldD * kGoldD, -1.0f, 1.0f, false);
    in.cs_q = synth(114, c * 3, -1.0f, 1.0f, true);
    in.cs_k = synth(115, c * 3, -1.0f, 1.0f, true);
    in.cs_v = synth(116, c * 3, -1.0f, 1.0f, true);
    return in;
}

// Case A: fully synthetic inputs; golden generated by the independent NumPy
// transcription (tools/gen_kda_golden.py). Recurrent path = the tight pin;
// chunked path must hit the same goldens within scan-reassociation tolerance.
TEST(KdaReference, GoldenSynthCase) {
    GoldenSet g = load_goldens("synth_");
    if (g.prefill_core.empty()) GTEST_FAIL() << "goldens missing — run tools/gen_kda_golden.py";
    ScanInputs in = make_golden_synth_inputs();
    run_golden_case(in, g, ScanPath::kRecurrent, 1e-5, 1e-4, "synth/recurrent");
    run_golden_case(in, g, ScanPath::kChunked, 3e-5, 3e-4, "synth/chunked");
}

// Case B: REAL GLM-5.3-Flash layer-0 KDA tensors (heads {0,17,42,63}) drive
// the gate/conv/o_norm; activations are synthetic; raw_g/g2 go through the
// REAL f_b/g_b low-rank mats. Pins the reference against the actual parameter
// distributions of the shipped checkpoint (the gate regime in particular:
// real exp(A_log) reaches 8.2 and dt_bias mean -0.8 — long-memory bias).
// Tolerance is looser than the synth case: this test recomputes the low-rank
// GEMMs with sequential fp32 dots while the generator used numpy pairwise
// sums; after bf16 rounding a gate input can land 1 bf16 ulp apart, and with
// the real steep gates (exp(A_log) up to 8.2) one flipped ulp moves a
// channel's decay measurably. MEASURED worst-case deviation at these bounds
// is ~10x below the limit (probe run 2026-08-29: max excess 0.018 at bounds
// 10x looser); any structural transcription error fails by orders of
// magnitude.
TEST(KdaReference, GoldenRealLayer0Case) {
    GoldenSet g = load_goldens("real_");
    if (g.prefill_core.empty()) GTEST_FAIL() << "goldens missing — run tools/gen_kda_golden.py";

    const std::string fx = "test-data/GLM-5.3-Flash/kda-layer0/";
    const int kRealHeads[kGoldH] = {0, 17, 42, 63};
    auto a_log_all = load_f32(fx + "A_log.bin", 64);
    auto dt_bias_all = load_f32(fx + "dt_bias.bin", 8192);
    auto conv_q_all = load_bf16_as_f32(fx + "q_conv1d_weight.bin", 8192 * 4);
    auto conv_k_all = load_bf16_as_f32(fx + "k_conv1d_weight.bin", 8192 * 4);
    auto conv_v_all = load_bf16_as_f32(fx + "v_conv1d_weight.bin", 8192 * 4);
    auto onorm_w = load_bf16_as_f32(fx + "o_norm_weight.bin", 128);
    auto f_b = load_bf16_as_f32(fx + "f_b_proj_weight.bin", 8192 * 128);
    auto g_b = load_bf16_as_f32(fx + "g_b_proj_weight.bin", 8192 * 128);
    if (a_log_all.empty() || f_b.empty()) {
        GTEST_FAIL() << "real layer-0 fixture missing — run tools/fetch_glm53_kda_layer0.py";
    }

    ScanInputs in;
    const size_t n = static_cast<size_t>(kGoldT) * kGoldTok;
    in.q = synth(303, n, -1.0f, 1.0f, true);
    in.k = synth(304, n, -1.0f, 1.0f, true);
    in.v = synth(305, n, -1.0f, 1.0f, true);
    in.beta = synth(306, static_cast<size_t>(kGoldT) * kGoldH, -2.0f, 2.0f, true);
    in.g2.assign(n, 0.0f);
    in.raw_g.assign(n, 0.0f);
    // raw_g / g2 through the REAL low-rank mats from synthetic rank-128
    // activations, bf16-rounded like the engine's GEMM output.
    auto fa = synth(301, static_cast<size_t>(kGoldT) * 128, -1.0f, 1.0f, true);
    auto ga = synth(302, static_cast<size_t>(kGoldT) * 128, -1.0f, 1.0f, true);
    for (int t = 0; t < kGoldT; ++t) {
        for (int h = 0; h < kGoldH; ++h) {
            const int rh = kRealHeads[h];
            for (int dd = 0; dd < kGoldD; ++dd) {
                const size_t row = static_cast<size_t>(rh) * kGoldD + dd;
                float acc_f = 0, acc_g = 0;
                for (int j = 0; j < 128; ++j) {
                    acc_f += f_b[row * 128 + j] * fa[static_cast<size_t>(t) * 128 + j];
                    acc_g += g_b[row * 128 + j] * ga[static_cast<size_t>(t) * 128 + j];
                }
                const size_t o = static_cast<size_t>(t) * kGoldTok +
                                 static_cast<size_t>(h) * kGoldD + dd;
                in.raw_g[o] = bf16_round(acc_f);
                in.g2[o] = bf16_round(acc_g);
            }
        }
    }
    in.a_log.resize(kGoldH);
    in.dt_bias.resize(kGoldTok);
    in.conv_wq.resize(kGoldTok * 4);
    in.conv_wk.resize(kGoldTok * 4);
    in.conv_wv.resize(kGoldTok * 4);
    for (int h = 0; h < kGoldH; ++h) {
        const int rh = kRealHeads[h];
        in.a_log[h] = a_log_all[rh];
        for (int dd = 0; dd < kGoldD; ++dd) {
            in.dt_bias[static_cast<size_t>(h) * kGoldD + dd] =
                dt_bias_all[static_cast<size_t>(rh) * kGoldD + dd];
            for (int j = 0; j < 4; ++j) {
                const size_t src = (static_cast<size_t>(rh) * kGoldD + dd) * 4 + j;
                const size_t dst = (static_cast<size_t>(h) * kGoldD + dd) * 4 + j;
                in.conv_wq[dst] = conv_q_all[src];
                in.conv_wk[dst] = conv_k_all[src];
                in.conv_wv[dst] = conv_v_all[src];
            }
        }
    }
    in.onorm_w = onorm_w;
    // Zero initial recurrent + conv state (run_golden_case defaults).
    run_golden_case(in, g, ScanPath::kRecurrent, 4e-4, 4e-3, "real/recurrent");
    run_golden_case(in, g, ScanPath::kChunked, 4e-4, 4e-3, "real/chunked");
}

}  // namespace
