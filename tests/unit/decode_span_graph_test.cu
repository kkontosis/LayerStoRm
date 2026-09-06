// GPU tests for the INV-0.6(b) decode span-graph runner (P-29 step 7).
//
// The runner's whole contract is: the span's kernels produce IDENTICAL bytes
// whether the span was run eagerly, captured, or replayed — only the
// submission mechanism changes — and any failure falls back EAGER (the token
// is never lost, numerics never change). Cases below drive a real multi-kernel
// span (zero_fill + copy via the graph-node movers + residual_add) through the
// warmup -> capture -> replay state machine and memcmp against a pure-eager
// reference each time, including per-replay CONTENT changes (the device-read
// discipline), fingerprint-drift re-capture, the kill switch, and the
// exception fail-safe (negative control: a throwing emit must abort the
// capture cleanly and the stream must remain usable).
//
// Footprint is tiny (a few hundred KB on device 0) so this suite can run next
// to a live engine holding the rest of VRAM.

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "compute/graphs/decode_span_graph.h"
#include "compute/kernels/elementwise/graph_node_ops.h"
#include "compute/kernels/elementwise/residual_add.h"

namespace lcomp = layerstorm::compute;

namespace {

#define ASSERT_CUDA(expr)                                                  \
    do {                                                                   \
        cudaError_t _e = (expr);                                           \
        ASSERT_EQ(cudaSuccess, _e) << #expr << " -> "                      \
                                   << cudaGetErrorString(_e);              \
    } while (0)

struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(size_t n) { EXPECT_EQ(cudaSuccess, cudaMalloc(&p, n)); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { if (p) cudaFree(p); }
};

// One span = zero_fill(out) ; interleave-free copy in ; residual_add(out += in2)
// — three dependent kernel launches on one stream, mirroring the production
// span shape (norm -> projections -> state update).
struct SpanFixture {
    static constexpr int kN = 4096;  // BF16 elements
    static constexpr size_t kBytes = kN * 2;
    DevBuf in{kBytes}, in2{kBytes}, out{kBytes};
    cudaStream_t stream = nullptr;
    std::vector<uint16_t> h_in, h_in2;

    SpanFixture() {
        EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
        std::mt19937 rng(1234);
        h_in.resize(kN);
        h_in2.resize(kN);
        for (int i = 0; i < kN; ++i) {
            h_in[i] = static_cast<uint16_t>(rng());
            h_in2[i] = static_cast<uint16_t>(rng());
        }
        upload();
    }
    ~SpanFixture() { if (stream) cudaStreamDestroy(stream); }

    void upload() {
        cudaMemcpy(in.p, h_in.data(), kBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(in2.p, h_in2.data(), kBytes, cudaMemcpyHostToDevice);
    }

    void emit() {
        // Three dependent kernels on one stream — the production span shape.
        lcomp::launch_zero_fill(out.p, kBytes, stream);
        lcomp::launch_residual_add(out.p, in.p, kN, stream);
        lcomp::launch_residual_add(out.p, in2.p, kN, stream);
    }

    std::vector<uint16_t> read_out() {
        std::vector<uint16_t> h(kN);
        cudaStreamSynchronize(stream);
        cudaMemcpy(h.data(), out.p, kBytes, cudaMemcpyDeviceToHost);
        return h;
    }

    void poison_out() {
        cudaMemset(out.p, 0xAB, kBytes);
        cudaStreamSynchronize(stream);
    }

    lcomp::DecodeSpanGraphs::Fp fp() const {
        lcomp::DecodeSpanGraphs::Fp f;
        f.add(in.p);
        f.add(in2.p);
        f.add(out.p);
        f.add_s(kN);
        return f;
    }
};

}  // namespace

TEST(DecodeSpanGraphTest, WarmupCaptureReplayBitIdentical) {
    REQUIRES_GPU();
    ASSERT_CUDA(cudaSetDevice(0));
    SpanFixture fx;
    lcomp::DecodeSpanGraphs graphs;
    graphs.arm(true);  // tests run on a real GPU (REQUIRES_GPU above)
    if (!graphs.enabled())
        GTEST_SKIP() << "LS_DECODE_CHAIN_GRAPH=0 in the environment";
    const uint32_t key = lcomp::DecodeSpanGraphs::make_key(
        lcomp::DecodeSpanGraphs::kKdaLayer, 7, 0);

    // Pure-eager reference.
    fx.poison_out();
    fx.emit();
    const auto ref = fx.read_out();

    // Run 1: warmup (eager).
    fx.poison_out();
    graphs.run(fx.stream, key, fx.fp(), [&] { fx.emit(); });
    EXPECT_EQ(fx.read_out(), ref) << "warmup run must be byte-identical";
    EXPECT_EQ(graphs.stats().eager, 1u);
    EXPECT_EQ(graphs.stats().captured, 0u);

    // Run 2: capture + first graph launch.
    fx.poison_out();
    graphs.run(fx.stream, key, fx.fp(), [&] { fx.emit(); });
    EXPECT_EQ(fx.read_out(), ref) << "captured run must be byte-identical";
    EXPECT_EQ(graphs.stats().captured, 1u);

    // Run 3: replay.
    fx.poison_out();
    graphs.run(fx.stream, key, fx.fp(), [&] { fx.emit(); });
    EXPECT_EQ(fx.read_out(), ref) << "replayed run must be byte-identical";
    EXPECT_GE(graphs.stats().replayed, 1u);
    EXPECT_EQ(graphs.stats().failed, 0u);
}

TEST(DecodeSpanGraphTest, ReplayReadsFreshDeviceContent) {
    REQUIRES_GPU();
    ASSERT_CUDA(cudaSetDevice(0));
    SpanFixture fx;
    lcomp::DecodeSpanGraphs graphs;
    graphs.arm(true);  // tests run on a real GPU (REQUIRES_GPU above)
    if (!graphs.enabled())
        GTEST_SKIP() << "LS_DECODE_CHAIN_GRAPH=0 in the environment";
    const uint32_t key = lcomp::DecodeSpanGraphs::make_key(
        lcomp::DecodeSpanGraphs::kMlaPrefix, 3, 1);

    // Warm + capture on content A.
    graphs.run(fx.stream, key, fx.fp(), [&] { fx.emit(); });
    graphs.run(fx.stream, key, fx.fp(), [&] { fx.emit(); });
    ASSERT_EQ(graphs.stats().captured, 1u);

    // Change the CONTENT of the input buffers (addresses unchanged) — the
    // per-token pattern (activations/slot ids restaged into fixed buffers).
    for (auto& v : fx.h_in) v ^= 0x5A5A;
    for (auto& v : fx.h_in2) v ^= 0x00FF;
    fx.upload();
    // Eager reference on the new content.
    fx.poison_out();
    fx.emit();
    const auto ref_new = fx.read_out();

    // Replay must consume the fresh content, not the capture-time bytes.
    fx.poison_out();
    graphs.run(fx.stream, key, fx.fp(), [&] { fx.emit(); });
    EXPECT_GE(graphs.stats().replayed, 1u);
    EXPECT_EQ(fx.read_out(), ref_new)
        << "replay must read device content at execution time";
}

TEST(DecodeSpanGraphTest, FingerprintDriftRecaptures) {
    REQUIRES_GPU();
    ASSERT_CUDA(cudaSetDevice(0));
    SpanFixture fx;
    DevBuf alt_in2{SpanFixture::kBytes};
    ASSERT_CUDA(cudaMemcpy(alt_in2.p, fx.h_in2.data(), SpanFixture::kBytes,
                           cudaMemcpyHostToDevice));
    lcomp::DecodeSpanGraphs graphs;
    graphs.arm(true);  // tests run on a real GPU (REQUIRES_GPU above)
    if (!graphs.enabled())
        GTEST_SKIP() << "LS_DECODE_CHAIN_GRAPH=0 in the environment";
    const uint32_t key = lcomp::DecodeSpanGraphs::make_key(
        lcomp::DecodeSpanGraphs::kGate, 11, 0);

    graphs.run(fx.stream, key, fx.fp(), [&] { fx.emit(); });
    graphs.run(fx.stream, key, fx.fp(), [&] { fx.emit(); });
    ASSERT_EQ(graphs.stats().captured, 1u);

    // Same key, DIFFERENT pointer in the fingerprint: must not replay the
    // stale graph (which would read the old buffer) — a second variant is
    // captured after its own warmup accounting.
    auto fp2 = fx.fp();
    fp2.p[1] = alt_in2.p;  // in2 slot
    auto emit2 = [&] {
        lcomp::launch_zero_fill(fx.out.p, SpanFixture::kBytes, fx.stream);
        lcomp::launch_residual_add(fx.out.p, fx.in.p, SpanFixture::kN,
                                   fx.stream);
        lcomp::launch_residual_add(fx.out.p, alt_in2.p, SpanFixture::kN,
                                   fx.stream);
    };
    fx.poison_out();
    graphs.run(fx.stream, key, fp2, emit2);   // new variant captured
    ASSERT_EQ(graphs.stats().captured, 2u)
        << "fingerprint drift must capture a new variant, never replay stale";
    fx.poison_out();
    fx.emit();
    const auto ref = fx.read_out();  // same bytes either way here
    fx.poison_out();
    graphs.run(fx.stream, key, fp2, emit2);   // replays variant 2
    EXPECT_EQ(fx.read_out(), ref);
    EXPECT_EQ(graphs.stats().failed, 0u);
}

TEST(DecodeSpanGraphTest, KillSwitchRunsEagerOnly) {
    REQUIRES_GPU();
    ASSERT_CUDA(cudaSetDevice(0));
    // The switch is read at construction — build a disabled runner.
    ASSERT_EQ(0, setenv("LS_DECODE_CHAIN_GRAPH", "0", 1));
    lcomp::DecodeSpanGraphs graphs;
    ASSERT_EQ(0, unsetenv("LS_DECODE_CHAIN_GRAPH"));
    graphs.arm(true);
    EXPECT_FALSE(graphs.enabled()) << "kill switch must win over arming";

    SpanFixture fx;
    fx.poison_out();
    fx.emit();
    const auto ref = fx.read_out();
    const uint32_t key = lcomp::DecodeSpanGraphs::make_key(
        lcomp::DecodeSpanGraphs::kKdaLayer, 1, 1);
    for (int i = 0; i < 4; ++i) {
        fx.poison_out();
        graphs.run(fx.stream, key, fx.fp(), [&] { fx.emit(); });
        EXPECT_EQ(fx.read_out(), ref);
    }
    EXPECT_EQ(graphs.stats().captured, 0u);
    EXPECT_EQ(graphs.stats().replayed, 0u);
    EXPECT_EQ(graphs.num_keys(), 0u) << "disabled runner must keep no state";
}

TEST(DecodeSpanGraphTest, ThrowingEmitAbortsCaptureAndStreamStaysUsable) {
    REQUIRES_GPU();
    ASSERT_CUDA(cudaSetDevice(0));
    SpanFixture fx;
    lcomp::DecodeSpanGraphs graphs;
    graphs.arm(true);  // tests run on a real GPU (REQUIRES_GPU above)
    if (!graphs.enabled())
        GTEST_SKIP() << "LS_DECODE_CHAIN_GRAPH=0 in the environment";
    const uint32_t key = lcomp::DecodeSpanGraphs::make_key(
        lcomp::DecodeSpanGraphs::kIndexer, 5, 0);

    // Warmup (eager, throw suppressed by the span's own guard here: we let
    // the first run succeed so run 2 attempts capture).
    graphs.run(fx.stream, key, fx.fp(), [&] { fx.emit(); });

    // Negative control: an emit that throws mid-capture. The runner must
    // abort the capture, mark the key dead, and rethrow.
    EXPECT_THROW(
        graphs.run(fx.stream, key, fx.fp(), [&]() -> void {
            lcomp::launch_zero_fill(fx.out.p, SpanFixture::kBytes, fx.stream);
            throw std::runtime_error("span blew up mid-capture");
        }),
        std::runtime_error);
    EXPECT_EQ(graphs.stats().failed, 1u);

    // The stream must be fully usable afterwards (capture aborted, not
    // wedged) and the dead key must run eagerly and correctly forever.
    fx.poison_out();
    fx.emit();
    const auto ref = fx.read_out();
    fx.poison_out();
    graphs.run(fx.stream, key, fx.fp(), [&] { fx.emit(); });
    EXPECT_EQ(fx.read_out(), ref)
        << "dead key must fall back to correct eager execution";
    EXPECT_EQ(graphs.stats().captured, 0u);
    EXPECT_EQ(graphs.stats().replayed, 0u);
}

// P-29 step 21 (TD-SPAN-INDEXER-VARIANT-RATCHET): the variant cap must not be
// a one-way ratchet. A 5th fingerprint shape used to run eager FOREVER
// (variants were never evicted) — a long-lived server whose kIndexer
// need_pages bucket outgrew the cache dropped to permanent eager on every
// affected key. Now: a new shape at cap runs eager once (hysteresis), then
// evicts the least-recently-replayed variant and recaptures; shapes that
// merely ALTERNATE never see two consecutive misses and cause no churn.
// The replay assertions below FAIL on the pre-step-21 code (permanent eager
// for shape E). Output bytes are memcmp'd against the eager reference at
// every stage — replay-vs-eager is submission mechanism only.
TEST(DecodeSpanGraphTest, VariantCapEvictsLruInsteadOfPermanentEager) {
    REQUIRES_GPU();
    ASSERT_CUDA(cudaSetDevice(0));
    SpanFixture fx;
    lcomp::DecodeSpanGraphs graphs;
    graphs.arm(true);
    if (!graphs.enabled())
        GTEST_SKIP() << "LS_DECODE_CHAIN_GRAPH=0 in the environment";
    const uint32_t key = lcomp::DecodeSpanGraphs::make_key(
        lcomp::DecodeSpanGraphs::kIndexer, 7, 0);

    fx.emit();
    const auto ref = fx.read_out();  // eager reference bytes

    // Shape tag rides the scalar fingerprint (same work, distinct variants —
    // exactly the need_pages grid-bound pattern).
    auto shape = [&](int tag) {
        auto f = fx.fp();
        f.add_s(tag);
        return f;
    };
    auto run = [&](int tag) {
        fx.poison_out();
        graphs.run(fx.stream, key, shape(tag), [&] { fx.emit(); });
        EXPECT_EQ(fx.read_out(), ref) << "shape tag " << tag;
    };

    // Warmup (runs=1 eager), then fill the cache to kVariantCap=4 variants.
    run(0);                       // eager warmup
    for (int tag : {0, 1, 2, 3}) run(tag);   // 4 captures
    ASSERT_EQ(graphs.stats().captured, 4u);
    ASSERT_EQ(graphs.stats().evicted, 0u);

    // Replay shape 1..3 so shape 0 becomes the LRU variant.
    for (int tag : {1, 2, 3}) run(tag);
    const uint64_t replays_before = graphs.stats().replayed;
    ASSERT_EQ(replays_before, 3u);

    // NEW shape 4 at cap: first sighting = eager (hysteresis, no eviction)…
    run(4);
    EXPECT_EQ(graphs.stats().captured, 4u);
    EXPECT_EQ(graphs.stats().evicted, 0u);
    // …second consecutive sighting: evict LRU (shape 0), capture shape 4…
    run(4);
    EXPECT_EQ(graphs.stats().evicted, 1u);
    EXPECT_EQ(graphs.stats().captured, 5u);
    // …third sighting REPLAYS (this is permanent-eager on the old code).
    run(4);
    EXPECT_EQ(graphs.stats().replayed, replays_before + 1)
        << "5th shape ratcheted to permanent eager instead of recovering";

    // Hot shapes 1..3 still replay from their untouched variants.
    for (int tag : {1, 2, 3}) run(tag);
    EXPECT_EQ(graphs.stats().replayed, replays_before + 4);

    // ALTERNATING shapes at cap (5,0,5,0…): never two consecutive misses ⇒
    // eager, zero further evictions, zero capture churn.
    const uint64_t captured_before = graphs.stats().captured;
    for (int i = 0; i < 3; ++i) { run(5); run(0); }
    EXPECT_EQ(graphs.stats().evicted, 1u)
        << "alternating shapes must not thrash the variant cache";
    EXPECT_EQ(graphs.stats().captured, captured_before);
    EXPECT_EQ(graphs.stats().failed, 0u);
}
