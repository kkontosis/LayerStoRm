// GF3.5: IndexPool ("kpool") indexer kernels — GPU tests against the CPU
// reference in tests/unit/kpool_reference.h.
//
// Under test (deps/LayerStoRmKernels/csrc/sm120/indexer/kpool_compress.cu via
// the src/ launch wrappers):
//   * launch_kpool_append_decode  — tail stash + conditional pool completion
//   * launch_kpool_chunk_append   — prefill pool completion + tail seed
//   * launch_kpool_expand         — pool ids -> token rows + always-select tail
//   * launch_kpool_cand_scatter   — dcp local-mode candidate scatter (pool domain)
// plus the UNCHANGED lightning kernels reused in the pooled ENTRY domain
// (launch_lightning_score_mqa over pooled entries, launch_lightning_topk with
// pooled endpoints i*P + P-1 and the POOL budget).
//
// A pooled indexer-K page is
//   [E x head_dim FP8 e4m3 | E x 4B f32 scales |
//    P x head_dim bf16 raw-K tail | P x head_dim bf16 raw-gate tail]
// with E = page_entries pooled entries covering E*P token positions.
//
// Numerics contract: scratchpad/GF35_KPOOL_REFERENCE.md §1 (K-side chain),
// §2 (tail state), §4 (selection/expansion), §7 (precision table),
// §8 (gotchas). Section refs below point there.

#include "compute/kernels/sm120/indexer/indexer_kpool.h"
#include "compute/kernels/sm120/indexer/lightning_indexer.h"
#include "compute/kernels/sm120/indexer/indexer_decode_bounds.h"

#include "kpool_reference.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

namespace lc = layerstorm::compute;
namespace si = sm120::indexer;

namespace {

// Production indexer head dim. The kernel only takes the literal
// 0.08838834764831845f Hadamard constant at head_dim == 128; every other value
// goes through fast-math rsqrtf, which the reference cannot reproduce bitwise
// (kpool_reference.h header comment). All tests therefore use D = 128.
constexpr int D = 128;

// ── device helpers (same conventions as indexer_paged_score_test.cu) ────────

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T),
                         cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}

template <typename T>
std::vector<T> download(const void* d, size_t n) {
    std::vector<T> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return h;
}

void* zeroed(size_t bytes) {
    void* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, bytes), cudaSuccess);
    EXPECT_EQ(cudaMemset(d, 0, bytes), cudaSuccess);
    return d;
}

/// Pooled indexer-K page geometry (see the file header for the byte layout).
struct PageGeom {
    int E;   ///< pooled entries per page
    int P;   ///< kpool

    size_t scales_off() const { return static_cast<size_t>(E) * D; }
    size_t tail_k_off() const { return static_cast<size_t>(E) * D + static_cast<size_t>(E) * 4; }
    size_t tail_g_off() const { return tail_k_off() + static_cast<size_t>(P) * D * 2; }
    size_t bytes() const { return tail_g_off() + static_cast<size_t>(P) * D * 2; }
};

/// Deterministic bf16 rows kept in both the device dtype and the exact fp32
/// values the reference must consume (bf16 -> fp32 is lossless).
struct Rows {
    std::vector<__nv_bfloat16> bf;
    std::vector<float> f;
};

Rows make_rows(std::mt19937& rng, int n, float sigma) {
    Rows r;
    r.bf.resize(static_cast<size_t>(n) * D);
    r.f.resize(static_cast<size_t>(n) * D);
    std::normal_distribution<float> dist(0.0f, sigma);
    for (size_t i = 0; i < r.bf.size(); ++i) {
        const __nv_bfloat16 v = __float2bfloat16_rn(dist(rng));
        r.bf[i] = v;
        r.f[i] = __bfloat162float(v);
    }
    return r;
}

std::vector<float> make_ape(std::mt19937& rng, int P) {
    std::vector<float> ape(static_cast<size_t>(P) * D);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    for (auto& v : ape) v = dist(rng);
    return ape;
}

/// Feed rows [row_begin, row_end) through the decode append path, one launch
/// per token, at in-page positions pos_begin + (row - row_begin).
void decode_append_range(void* dpage, const PageGeom& g, const __nv_bfloat16* dk,
                         const __nv_bfloat16* dg, const float* dape,
                         int row_begin, int row_end, int pos_begin) {
    for (int t = row_begin; t < row_end; ++t) {
        si::KpoolAppendDecodeParams p{};
        p.k_row = dk + static_cast<size_t>(t) * D;
        p.gate_row = dg + static_cast<size_t>(t) * D;
        p.ape = dape;
        p.page_base = dpage;
        p.page_entries = g.E;
        p.pos_in_page = pos_begin + (t - row_begin);
        p.kpool = g.P;
        p.head_dim = D;
        lc::launch_kpool_append_decode(p, nullptr);
    }
}

void chunk_append(void* dpage, const PageGeom& g, const __nv_bfloat16* dk,
                  const __nv_bfloat16* dg, const float* dape, int num_rows,
                  int pos0_in_page, bool seed_tail) {
    si::KpoolChunkAppendParams p{};
    p.k_rows = dk;
    p.gate_rows = dg;
    p.ape = dape;
    p.page_base = dpage;
    p.page_entries = g.E;
    p.pos0_in_page = pos0_in_page;
    p.num_rows = num_rows;
    p.kpool = g.P;
    p.head_dim = D;
    p.seed_tail = seed_tail;
    lc::launch_kpool_chunk_append(p, nullptr);
}

float page_scale(const std::vector<uint8_t>& page, const PageGeom& g, int entry) {
    float s = 0.0f;
    std::memcpy(&s, page.data() + g.scales_off() + static_cast<size_t>(entry) * 4,
                sizeof(float));
    return s;
}

/// Assert one pooled entry against the reference chain.
///
/// PRIMARY (always): the DEQUANTIZED entry, banded to 2 fp8 ULP. This is the
/// invariant that holds unconditionally — see the boundary note below for why
/// it, and not the raw bytes, is the scale-independent statement.
///
/// STRICT (whenever the reference absmax is safely off a ue8m0 boundary):
/// scale EXACTLY equal and FP8 bytes EXACTLY equal. Fast-math expf perturbs
/// the pooling weights by a few fp32 ULP, which the mandatory bf16 round-trip
/// and the 3-mantissa-bit e4m3 grid both absorb, so the bytes agree exactly.
///
/// BOUNDARY case: scale = exp2(ceil(log2(absmax/448))) is a step function of
/// absmax. When absmax lands ON a power-of-two multiple of 448, a SINGLE bf16
/// ULP of fast-math drift in the absmax dimension flips the ceil and the whole
/// entry's scale moves by one power of two. That is not a precision loss:
/// e4m3 is a floating-point grid, so halving/doubling the scale shifts every
/// quantized exponent by one and leaves the dequantized value IDENTICAL
/// (verified: byte 0xE4 at 2^-7 and byte 0xEC at 2^-8 both dequantize to
/// -0.375). Only the byte pattern and the scale differ, so those two
/// assertions are dropped exactly there and the dequantized band carries the
/// check. IndexerKpool.ChunkThenDecodeContinuity's third pool is a real
/// instance (reference absmax 1.75 = 448 * 2^-8 exactly).
void expect_entry_matches(const std::vector<uint8_t>& page, const PageGeom& g,
                          int entry, const kpool_ref::PooledEntry& ref,
                          const char* what) {
    const float got_scale = page_scale(page, g, entry);
    ASSERT_GT(got_scale, 0.0f) << what << ": entry " << entry << " was never written";

    // Distance (in log2) from the ue8m0 step. One bf16 ULP is 2^-8 relative,
    // i.e. ~0.0056 in log2; 0.01 covers just under two of them.
    const float l = std::log2(std::fmax(ref.absmax, 1e-4f) / 448.0f);
    const bool on_boundary = std::fabs(l - std::nearbyint(l)) < 0.01f;

    // Dequantized band: 2 fp8 ULP measured in units of the entry scale.
    const float band = 2.0f * std::fmax(got_scale, ref.scale);
    for (int d = 0; d < D; ++d) {
        const float got = kpool_ref::e4m3_to_f32(page[static_cast<size_t>(entry) * D + d])
                          * got_scale;
        const float want = kpool_ref::e4m3_to_f32(ref.bytes[static_cast<size_t>(d)])
                           * ref.scale;
        ASSERT_NEAR(got, want, band)
            << what << ": entry " << entry << " dequantized dim " << d;
    }

    if (on_boundary) {
        // Only the one-step scale ambiguity is tolerated, nothing wider.
        EXPECT_TRUE(got_scale == ref.scale || got_scale == 2.0f * ref.scale
                    || got_scale == 0.5f * ref.scale)
            << what << ": entry " << entry << " scale " << got_scale
            << " is more than one ue8m0 step from the reference " << ref.scale;
        return;
    }
    EXPECT_EQ(got_scale, ref.scale)
        << what << ": entry " << entry << " scale (absmax " << ref.absmax << ")";
    for (int d = 0; d < D; ++d) {
        ASSERT_EQ(page[static_cast<size_t>(entry) * D + d], ref.bytes[static_cast<size_t>(d)])
            << what << ": entry " << entry << " byte " << d;
    }
}

/// Bitwise compare one tail slot (raw bf16 K and gate) against a source row.
void expect_tail_slot(const std::vector<uint8_t>& page, const PageGeom& g, int slot,
                      const Rows& k, const Rows& gate, int row, const char* what) {
    const size_t nbytes = static_cast<size_t>(D) * sizeof(__nv_bfloat16);
    EXPECT_EQ(std::memcmp(page.data() + g.tail_k_off() + static_cast<size_t>(slot) * nbytes,
                          k.bf.data() + static_cast<size_t>(row) * D, nbytes), 0)
        << what << ": tail K slot " << slot;
    EXPECT_EQ(std::memcmp(page.data() + g.tail_g_off() + static_cast<size_t>(slot) * nbytes,
                          gate.bf.data() + static_cast<size_t>(row) * D, nbytes), 0)
        << what << ": tail gate slot " << slot;
}

/// launch_lightning_topk over pooled entries: pooled endpoints i*P + P-1 and
/// the POOL budget (select_k = index_topk / kpool, §4).
void pooled_topk(const float* dscores, const int* dendpoints, int num_pools,
                 int budget, int qpos, int* didx, float* dscore_out, int* deff) {
    si::LightningTopkParams tp{};
    tp.scores = dscores;
    tp.block_endpoints = dendpoints;
    tp.output_indices = didx;
    tp.output_scores = dscore_out;
    tp.effective_k_out = deff;
    tp.num_blocks = num_pools;
    tp.topk = budget;
    tp.query_position = qpos;
    lc::launch_lightning_topk(tp, nullptr);
}

void kpool_expand(const int* dids, const int* deff, const int* dseq, int* dout,
                  int* dlen, int num_rows, int P, int pool_stride, int out_stride,
                  int out_cols) {
    si::KpoolExpandParams ep{};
    ep.pool_ids = dids;
    ep.eff_pools = deff;
    ep.row_seq_len = dseq;
    ep.indices_out = dout;
    ep.lengths_out = dlen;
    ep.num_rows = num_rows;
    ep.kpool = P;
    ep.pool_stride = pool_stride;
    ep.out_stride = out_stride;
    ep.out_cols = out_cols;
    lc::launch_kpool_expand(ep, nullptr);
}

std::vector<int> pooled_endpoints(int num_pools, int P) {
    std::vector<int> e(static_cast<size_t>(num_pools));
    for (int i = 0; i < num_pools; ++i) e[static_cast<size_t>(i)] = i * P + P - 1;
    return e;
}

}  // namespace

// ── 1. Decode append: composition matches the CPU reference ────────────────
//
// Eight tokens fed one at a time (§2 decode ring update) build two pooled
// entries; the ninth opens the next pool and must land in tail slot 0 BITWISE
// (raw post-LayerNorm K and raw gate, unrotated and unquantized).
TEST(IndexerKpool, DecodeAppendComposeMatchesReference) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    const PageGeom g{16, 4};
    const int n_tokens = 9;
    std::mt19937 rng(9001);
    const Rows k = make_rows(rng, n_tokens, 1.0f);
    const Rows gate = make_rows(rng, n_tokens, 1.0f);
    const std::vector<float> ape = make_ape(rng, g.P);

    auto* dk = upload(k.bf);
    auto* dg = upload(gate.bf);
    auto* dape = upload(ape);
    void* dpage = zeroed(g.bytes());

    decode_append_range(dpage, g, dk, dg, dape, 0, n_tokens, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto page = download<uint8_t>(dpage, g.bytes());

    for (int pool = 0; pool < 2; ++pool) {
        const int row0 = pool * g.P;
        const auto ref = kpool_ref::ref_kpool_compose(
            k.f.data() + static_cast<size_t>(row0) * D,
            gate.f.data() + static_cast<size_t>(row0) * D, ape.data(), D, g.P);
        expect_entry_matches(page, g, pool, ref, "decode");
    }
    // Token 8 opened pool 2 (slot 0): raw stash, no entry yet.
    expect_tail_slot(page, g, 0, k, gate, 8, "decode");
    // The frontier pool has no entry — entry 2 is still zeroed.
    EXPECT_EQ(page_scale(page, g, 2), 0.0f);

    cudaFree(dk); cudaFree(dg); cudaFree(dape); cudaFree(dpage);
}

// kpool = 2 geometry: same chain, half the pool width (kMaxKpool covers 2..8).
TEST(IndexerKpool, DecodeAppendKpool2MatchesReference) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    const PageGeom g{8, 2};
    const int n_tokens = 7;   // 3 full pools + 1 frontier token
    std::mt19937 rng(4242);
    const Rows k = make_rows(rng, n_tokens, 1.0f);
    const Rows gate = make_rows(rng, n_tokens, 1.0f);
    const std::vector<float> ape = make_ape(rng, g.P);

    auto* dk = upload(k.bf);
    auto* dg = upload(gate.bf);
    auto* dape = upload(ape);
    void* dpage = zeroed(g.bytes());
    void* dpage_chunk = zeroed(g.bytes());

    decode_append_range(dpage, g, dk, dg, dape, 0, n_tokens, 0);
    chunk_append(dpage_chunk, g, dk, dg, dape, n_tokens, 0, /*seed_tail=*/true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto page = download<uint8_t>(dpage, g.bytes());
    const auto page_chunk = download<uint8_t>(dpage_chunk, g.bytes());

    for (int pool = 0; pool < 3; ++pool) {
        const int row0 = pool * g.P;
        const auto ref = kpool_ref::ref_kpool_compose(
            k.f.data() + static_cast<size_t>(row0) * D,
            gate.f.data() + static_cast<size_t>(row0) * D, ape.data(), D, g.P);
        expect_entry_matches(page, g, pool, ref, "decode-P2");
        expect_entry_matches(page_chunk, g, pool, ref, "chunk-P2");
    }
    expect_tail_slot(page, g, 0, k, gate, 6, "decode-P2");
    expect_tail_slot(page_chunk, g, 0, k, gate, 6, "chunk-P2");

    cudaFree(dk); cudaFree(dg); cudaFree(dape);
    cudaFree(dpage); cudaFree(dpage_chunk);
}

// ── 2. Chunk append == decode append, bitwise ──────────────────────────────
//
// The same 11 tokens (2 pools + 3 tail rows) through both entry points. Both
// go through the identical compose_and_write_entry body, so entries, scales
// and the LIVE tail slots must agree bit-for-bit.
TEST(IndexerKpool, ChunkAppendBitwiseMatchesDecodeAppend) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    const PageGeom g{16, 4};
    const int n_tokens = 11;   // 2 pools + rem 3
    const int rem = n_tokens % g.P;
    std::mt19937 rng(1234);
    const Rows k = make_rows(rng, n_tokens, 1.0f);
    const Rows gate = make_rows(rng, n_tokens, 1.0f);
    const std::vector<float> ape = make_ape(rng, g.P);

    auto* dk = upload(k.bf);
    auto* dg = upload(gate.bf);
    auto* dape = upload(ape);
    void* dpage_dec = zeroed(g.bytes());
    void* dpage_chunk = zeroed(g.bytes());

    decode_append_range(dpage_dec, g, dk, dg, dape, 0, n_tokens, 0);
    chunk_append(dpage_chunk, g, dk, dg, dape, n_tokens, 0, /*seed_tail=*/true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto dec = download<uint8_t>(dpage_dec, g.bytes());
    const auto chk = download<uint8_t>(dpage_chunk, g.bytes());

    // Entries + scales: bitwise.
    EXPECT_EQ(std::memcmp(dec.data(), chk.data(),
                          g.scales_off() + static_cast<size_t>(g.E) * 4), 0)
        << "pooled entries / scales differ between chunk and decode append";

    // LIVE tail slots 0..rem-1: bitwise. Slot P-1 is deliberately NOT compared:
    // the decode path leaves it holding the last COMPLETED pool's final token
    // (its stash runs after the completion read, §2), while the chunk path only
    // seeds slots 0..rem-1. That residue is never read back — pool completion
    // only ever reads slots 0..P-2 — so the two pages are equivalent states.
    const size_t row_bytes = static_cast<size_t>(D) * sizeof(__nv_bfloat16);
    for (int slot = 0; slot < rem; ++slot) {
        EXPECT_EQ(std::memcmp(dec.data() + g.tail_k_off() + slot * row_bytes,
                              chk.data() + g.tail_k_off() + slot * row_bytes,
                              row_bytes), 0) << "tail K slot " << slot;
        EXPECT_EQ(std::memcmp(dec.data() + g.tail_g_off() + slot * row_bytes,
                              chk.data() + g.tail_g_off() + slot * row_bytes,
                              row_bytes), 0) << "tail gate slot " << slot;
        expect_tail_slot(chk, g, slot, k, gate, 2 * g.P + slot, "chunk");
    }

    cudaFree(dk); cudaFree(dg); cudaFree(dape);
    cudaFree(dpage_dec); cudaFree(dpage_chunk);
}

// ── 3. Chunk-then-decode continuity ────────────────────────────────────────
//
// A prompt ending mid-pool is the ONLY recomputation path (§5): the pool is
// completed later, from the tail, by decode appends. Pool 2 built that way must
// equal pool 2 built by feeding all 12 tokens through decode append.
TEST(IndexerKpool, ChunkThenDecodeContinuity) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    const PageGeom g{16, 4};
    const int prompt = 10;     // 2 pools + 2 tail rows
    const int total = 12;      // decode tokens at positions 10 and 11
    std::mt19937 rng(777);
    const Rows k = make_rows(rng, total, 1.0f);
    const Rows gate = make_rows(rng, total, 1.0f);
    const std::vector<float> ape = make_ape(rng, g.P);

    auto* dk = upload(k.bf);
    auto* dg = upload(gate.bf);
    auto* dape = upload(ape);
    void* dpage_mixed = zeroed(g.bytes());
    void* dpage_dec = zeroed(g.bytes());

    chunk_append(dpage_mixed, g, dk, dg, dape, prompt, 0, /*seed_tail=*/true);
    decode_append_range(dpage_mixed, g, dk, dg, dape, prompt, total, prompt);
    decode_append_range(dpage_dec, g, dk, dg, dape, 0, total, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto mixed = download<uint8_t>(dpage_mixed, g.bytes());
    const auto dec = download<uint8_t>(dpage_dec, g.bytes());

    // All three pooled entries (2 from the chunk, 1 completed across the seam).
    EXPECT_EQ(std::memcmp(mixed.data(), dec.data(), static_cast<size_t>(3) * D), 0)
        << "entries differ across the chunk/decode seam";
    for (int e = 0; e < 3; ++e) {
        EXPECT_EQ(page_scale(mixed, g, e), page_scale(dec, g, e)) << "scale " << e;
    }
    // The seam-completed pool is the interesting one: check it against the
    // reference too, not just against the other GPU path.
    const auto ref = kpool_ref::ref_kpool_compose(
        k.f.data() + static_cast<size_t>(2 * g.P) * D,
        gate.f.data() + static_cast<size_t>(2 * g.P) * D, ape.data(), D, g.P);
    expect_entry_matches(mixed, g, 2, ref, "seam");

    cudaFree(dk); cudaFree(dg); cudaFree(dape);
    cudaFree(dpage_mixed); cudaFree(dpage_dec);
}

// ── 4. Pooled paged vs contiguous scoring, bit-equal ───────────────────────
//
// The pooled analogue of IndexerPagedScore.PagedEqualsContiguous: entries live
// in per-page [E x D FP8 | E f32] regions (scales at base + page_entries*D,
// exactly as the backend's per-page score loop addresses them,
// cuda_sm120_device_backend.cu indexer_score_topk), so scoring page-by-page
// must be bit-identical to scoring one contiguous entry array.
TEST(IndexerKpool, PooledPagedVsContiguousScoreBitEqual) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    const PageGeom g{8, 4};
    const int n_pages = 3;
    const int entries_per_page[3] = {8, 8, 5};   // partial last page
    const int NB = 21;
    const int NIH = 8;

    std::mt19937 rng(31337);
    const std::vector<float> ape = make_ape(rng, g.P);
    auto* dape = upload(ape);

    std::vector<void*> pages(n_pages);
    std::vector<Rows> ks, gs;
    for (int p = 0; p < n_pages; ++p) {
        const int rows = entries_per_page[p] * g.P;
        ks.push_back(make_rows(rng, rows, 1.0f));
        gs.push_back(make_rows(rng, rows, 1.0f));
        pages[static_cast<size_t>(p)] = zeroed(g.bytes());
        auto* dk = upload(ks.back().bf);
        auto* dg = upload(gs.back().bf);
        chunk_append(pages[static_cast<size_t>(p)], g, dk, dg, dape, rows, 0, false);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaFree(dk); cudaFree(dg);
    }

    // Contiguous mirror: [NB x D FP8] then [NB f32 scales].
    auto* dcontig_k = static_cast<uint8_t*>(zeroed(static_cast<size_t>(NB) * D));
    auto* dcontig_s = static_cast<float*>(zeroed(static_cast<size_t>(NB) * sizeof(float)));
    int base = 0;
    for (int p = 0; p < n_pages; ++p) {
        const int n = entries_per_page[p];
        ASSERT_EQ(cudaMemcpy(dcontig_k + static_cast<size_t>(base) * D,
                             pages[static_cast<size_t>(p)], static_cast<size_t>(n) * D,
                             cudaMemcpyDeviceToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(dcontig_s + base,
                             static_cast<uint8_t*>(pages[static_cast<size_t>(p)]) + g.scales_off(),
                             static_cast<size_t>(n) * sizeof(float),
                             cudaMemcpyDeviceToDevice), cudaSuccess);
        base += n;
    }

    std::vector<__nv_bfloat16> q(static_cast<size_t>(NIH) * D);
    std::vector<float> qf(q.size());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = __float2bfloat16_rn(dist(rng));
        qf[i] = __bfloat162float(q[i]);
    }
    std::vector<float> w(static_cast<size_t>(NIH));
    for (auto& v : w) v = dist(rng) / 8.0f;
    auto* dq = upload(q);
    auto* dw = upload(w);
    auto* dscores_paged = static_cast<float*>(zeroed(static_cast<size_t>(NB) * sizeof(float)));
    auto* dscores_contig = static_cast<float*>(zeroed(static_cast<size_t>(NB) * sizeof(float)));

    for (int p = 0; p < n_pages; ++p) {
        const auto* pbase = static_cast<const __nv_fp8_e4m3*>(pages[static_cast<size_t>(p)]);
        si::LightningScoreMqaParams sp{};
        sp.q_proj = dq;
        sp.indexer_k_cache = pbase;
        sp.k_scales = reinterpret_cast<const float*>(
            reinterpret_cast<const uint8_t*>(pbase) + g.scales_off());
        sp.score_proj = dw;
        sp.scores_out = dscores_paged + static_cast<size_t>(p) * g.E;
        sp.num_blocks = entries_per_page[p];
        sp.index_n_heads = NIH;
        sp.index_head_dim = D;
        lc::launch_lightning_score_mqa(sp, nullptr);
    }
    {
        si::LightningScoreMqaParams sp{};
        sp.q_proj = dq;
        sp.indexer_k_cache = reinterpret_cast<const __nv_fp8_e4m3*>(dcontig_k);
        sp.k_scales = dcontig_s;
        sp.score_proj = dw;
        sp.scores_out = dscores_contig;
        sp.num_blocks = NB;
        sp.index_n_heads = NIH;
        sp.index_head_dim = D;
        lc::launch_lightning_score_mqa(sp, nullptr);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto paged = download<float>(dscores_paged, NB);
    const auto contig = download<float>(dscores_contig, NB);
    for (int i = 0; i < NB; ++i) {
        EXPECT_EQ(paged[static_cast<size_t>(i)], contig[static_cast<size_t>(i)])
            << "pooled entry " << i;
    }

    // The CPU reference reproduces the same logits to fp32 reduction noise —
    // this is what gives ref_kpool_score() its coverage.
    const auto entries = download<uint8_t>(dcontig_k, static_cast<size_t>(NB) * D);
    const auto scales = download<float>(dcontig_s, NB);
    const auto ref = kpool_ref::ref_kpool_score(qf.data(), w.data(), entries.data(),
                                                scales.data(), NB, NIH, D);
    float mag = 1e-6f;
    for (int i = 0; i < NB; ++i) mag = std::fmax(mag, std::fabs(ref[static_cast<size_t>(i)]));
    for (int i = 0; i < NB; ++i) {
        EXPECT_NEAR(contig[static_cast<size_t>(i)], ref[static_cast<size_t>(i)], 2e-4f * mag)
            << "reference score mismatch at pooled entry " << i;
    }

    for (auto* p : pages) cudaFree(p);
    cudaFree(dape); cudaFree(dcontig_k); cudaFree(dcontig_s);
    cudaFree(dq); cudaFree(dw); cudaFree(dscores_paged); cudaFree(dscores_contig);
}

// ── 5. Selection + expansion against the reference ─────────────────────────
//
// Pooled endpoints i*P + P-1 into the UNCHANGED lightning top-k, then
// launch_kpool_expand. Covers: mid-pool query position, pool-aligned length
// (tail EMPTY — the GF3.2 booklet note (c)), short context (eff < budget),
// seq_len < P (eff == 0, tail only), and the -1 padding tail of the row.
// lengths_out ALIASES eff_pools here, exactly as the engine wires it (the
// top-k kernel's pool count is overwritten in place by the token count).
TEST(IndexerKpool, SelectionExpansionMatchesReference) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    const int NB = 64, P = 4, budget = 8;
    const int out_cols = budget * P + P - 1;   // index_topk_rows capacity (§4)

    std::mt19937 rng(20260830);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> scores(static_cast<size_t>(NB));
    for (auto& v : scores) v = dist(rng);
    // Deliberate exact tie: exercises DET-TOPK-TIES (lower index wins) on both
    // sides. Pools 11 and 40 carry the same score.
    scores[40] = scores[11];

    auto* dscores = upload(scores);
    const auto endpoints = pooled_endpoints(NB, P);
    auto* dend = upload(endpoints);
    auto* didx = static_cast<int*>(zeroed(static_cast<size_t>(budget) * sizeof(int)));
    auto* dtopk_s = static_cast<float*>(zeroed(static_cast<size_t>(budget) * sizeof(float)));
    auto* deff = static_cast<int*>(zeroed(sizeof(int)));
    auto* dout = static_cast<int*>(zeroed(static_cast<size_t>(out_cols) * sizeof(int)));
    auto* dseq = static_cast<int*>(zeroed(sizeof(int)));

    // qpos values: mid-pool, pool-aligned (tail empty), short context, and a
    // sequence shorter than one pool (no entry exists at all).
    const int qpos_cases[] = {45, 47, 10, 2};
    for (int qpos : qpos_cases) {
        const int seq_len = qpos + 1;
        ASSERT_EQ(cudaMemcpy(dseq, &seq_len, sizeof(int), cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemset(dout, 0, static_cast<size_t>(out_cols) * sizeof(int)),
                  cudaSuccess);
        pooled_topk(dscores, dend, NB, budget, qpos, didx, dtopk_s, deff);
        // lengths_out == eff_pools: in-place pool-count -> token-count.
        kpool_expand(didx, deff, dseq, dout, deff, 1, P, budget, out_cols, out_cols);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto got_ids = download<int>(didx, budget);
        const auto got_len = download<int>(deff, 1);
        const auto got_cols = download<int>(dout, out_cols);

        const auto ref_sel = kpool_ref::ref_topk_pools(scores.data(), NB, budget, qpos, P);
        for (int i = 0; i < budget; ++i) {
            EXPECT_EQ(got_ids[static_cast<size_t>(i)], ref_sel.ids[static_cast<size_t>(i)])
                << "qpos " << qpos << " pool slot " << i;
        }
        const auto ref_exp = kpool_ref::ref_kpool_expand(ref_sel.ids.data(), ref_sel.eff,
                                                         seq_len, P, out_cols);
        EXPECT_EQ(got_len[0], ref_exp.count) << "qpos " << qpos << " token count";
        for (int c = 0; c < out_cols; ++c) {
            EXPECT_EQ(got_cols[static_cast<size_t>(c)], ref_exp.cols[static_cast<size_t>(c)])
                << "qpos " << qpos << " col " << c;
        }

        // Structural expectations, stated independently of the reference.
        const int tail = seq_len - (seq_len / P) * P;
        if (qpos == 47) {
            EXPECT_EQ(tail, 0) << "qpos 47 must be pool-aligned";
            EXPECT_EQ(got_len[0], ref_sel.eff * P);
            // Pool-aligned length: NO tail rows appended.
            for (int c = ref_sel.eff * P; c < out_cols; ++c) {
                EXPECT_EQ(got_cols[static_cast<size_t>(c)], -1)
                    << "pool-aligned tail must be empty, col " << c;
            }
        }
        if (qpos == 10) EXPECT_LT(ref_sel.eff, budget) << "short context: eff < budget";
        if (qpos == 2) {
            EXPECT_EQ(ref_sel.eff, 0) << "seq_len < kpool: no settled pool";
            EXPECT_EQ(got_len[0], seq_len);
            for (int c = 0; c < seq_len; ++c) {
                EXPECT_EQ(got_cols[static_cast<size_t>(c)], c) << "tail-only col " << c;
            }
        }
        // Padding sentinel: everything past the token count is -1.
        for (int c = got_len[0]; c < out_cols; ++c) {
            EXPECT_EQ(got_cols[static_cast<size_t>(c)], -1)
                << "qpos " << qpos << " padding col " << c;
        }
    }

    cudaFree(dscores); cudaFree(dend); cudaFree(didx); cudaFree(dtopk_s);
    cudaFree(deff); cudaFree(dout); cudaFree(dseq);
}

// ── 6. Local (dcp) merge == replicated selection ───────────────────────────
//
// dcp_size = 2 simulated on one GPU. Pages are owned round-robin in the ENTRY
// domain (page_entries pooled entries per page), so rank r owns global pages
// r, r+dcp, r+2*dcp, ... Each rank runs the pooled top-k over its OWN compacted
// entries, the candidate segments are scattered back to global entry slots
// (launch_kpool_cand_scatter), and the global top-k + expand must reproduce the
// replicated path exactly. Per-rank candidate cap > final budget makes the
// merge exact by construction.
TEST(IndexerKpool, LocalMergeMatchesReplicated) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    const int NB = 64, P = 4, E = 8, dcp = 2;
    const int cand_count = 16;    // per-rank candidate cap
    const int cand_stride = 35;   // candidate ROW stride > cand_count
    const int budget = 8;         // final pool budget
    const int batch = 1, token = 0;
    const int seg_words = 2 * batch * cand_stride;
    const int qpos = 250;         // seq_len 251 -> 62 settled pools, tail 3
    const int seq_len = qpos + 1;
    const int out_cols = budget * P + P - 1;

    std::mt19937 rng(555);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> scores(static_cast<size_t>(NB));
    for (auto& v : scores) v = dist(rng);

    const auto endpoints = pooled_endpoints(NB, P);
    auto* dend = upload(endpoints);
    auto* dscores_full = upload(scores);

    // Per-rank local views: local = lp*E + off, global page = lp*dcp + rank.
    auto* dgathered = static_cast<int*>(zeroed(static_cast<size_t>(dcp) * seg_words * sizeof(int)));
    ASSERT_EQ(cudaMemset(dgathered, 0xFF,
                         static_cast<size_t>(dcp) * seg_words * sizeof(int)), cudaSuccess);
    auto* deff_local = static_cast<int*>(zeroed(static_cast<size_t>(dcp) * sizeof(int)));
    std::vector<float*> dlocal_scores(static_cast<size_t>(dcp));
    std::vector<int*> dlocal_end(static_cast<size_t>(dcp));
    for (int r = 0; r < dcp; ++r) {
        std::vector<float> ls;
        std::vector<int> le;
        for (int gp = r; gp * E < NB; gp += dcp) {
            for (int off = 0; off < E && gp * E + off < NB; ++off) {
                const int global = gp * E + off;
                ls.push_back(scores[static_cast<size_t>(global)]);
                le.push_back(endpoints[static_cast<size_t>(global)]);
            }
        }
        dlocal_scores[static_cast<size_t>(r)] = upload(ls);
        dlocal_end[static_cast<size_t>(r)] = upload(le);
        int* idx_row = dgathered + static_cast<size_t>(r) * seg_words
                       + static_cast<size_t>(token) * cand_stride;
        float* sc_row = reinterpret_cast<float*>(
            dgathered + static_cast<size_t>(r) * seg_words
            + static_cast<size_t>(batch) * cand_stride
            + static_cast<size_t>(token) * cand_stride);
        pooled_topk(dlocal_scores[static_cast<size_t>(r)], dlocal_end[static_cast<size_t>(r)],
                    static_cast<int>(ls.size()), cand_count, qpos, idx_row, sc_row,
                    deff_local + r);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto* dscratch = static_cast<float*>(zeroed(static_cast<size_t>(NB) * sizeof(float)));
    si::KpoolCandScatterParams sc{};
    sc.gathered = dgathered;
    sc.seg_words = seg_words;
    sc.batch = batch;
    sc.token = token;
    sc.cand_stride = cand_stride;
    sc.cand_count = cand_count;
    sc.scores_scratch = dscratch;
    sc.num_entries = NB;
    sc.dcp_size = dcp;
    sc.page_entries = E;
    lc::launch_kpool_cand_scatter(sc, nullptr);

    auto* didx_m = static_cast<int*>(zeroed(static_cast<size_t>(budget) * sizeof(int)));
    auto* dsc_m = static_cast<float*>(zeroed(static_cast<size_t>(budget) * sizeof(float)));
    auto* deff_m = static_cast<int*>(zeroed(sizeof(int)));
    auto* dout_m = static_cast<int*>(zeroed(static_cast<size_t>(out_cols) * sizeof(int)));
    auto* dseq = upload(std::vector<int>{seq_len});
    auto* dlen_m = static_cast<int*>(zeroed(sizeof(int)));
    pooled_topk(dscratch, dend, NB, budget, qpos, didx_m, dsc_m, deff_m);
    kpool_expand(didx_m, deff_m, dseq, dout_m, dlen_m, 1, P, budget, out_cols, out_cols);

    auto* didx_r = static_cast<int*>(zeroed(static_cast<size_t>(budget) * sizeof(int)));
    auto* dsc_r = static_cast<float*>(zeroed(static_cast<size_t>(budget) * sizeof(float)));
    auto* deff_r = static_cast<int*>(zeroed(sizeof(int)));
    auto* dout_r = static_cast<int*>(zeroed(static_cast<size_t>(out_cols) * sizeof(int)));
    auto* dlen_r = static_cast<int*>(zeroed(sizeof(int)));
    pooled_topk(dscores_full, dend, NB, budget, qpos, didx_r, dsc_r, deff_r);
    kpool_expand(didx_r, deff_r, dseq, dout_r, dlen_r, 1, P, budget, out_cols, out_cols);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto merged = download<int>(dout_m, out_cols);
    const auto replicated = download<int>(dout_r, out_cols);
    EXPECT_EQ(download<int>(dlen_m, 1)[0], download<int>(dlen_r, 1)[0]);
    for (int c = 0; c < out_cols; ++c) {
        EXPECT_EQ(merged[static_cast<size_t>(c)], replicated[static_cast<size_t>(c)])
            << "local-merge vs replicated token col " << c;
    }
    // And both equal the CPU reference over the full score array.
    const auto ref_sel = kpool_ref::ref_topk_pools(scores.data(), NB, budget, qpos, P);
    const auto ref_exp = kpool_ref::ref_kpool_expand(ref_sel.ids.data(), ref_sel.eff,
                                                     seq_len, P, out_cols);
    for (int c = 0; c < out_cols; ++c) {
        EXPECT_EQ(merged[static_cast<size_t>(c)], ref_exp.cols[static_cast<size_t>(c)])
            << "local-merge vs reference token col " << c;
    }

    for (int r = 0; r < dcp; ++r) {
        cudaFree(dlocal_scores[static_cast<size_t>(r)]);
        cudaFree(dlocal_end[static_cast<size_t>(r)]);
    }
    cudaFree(dend); cudaFree(dscores_full); cudaFree(dgathered); cudaFree(deff_local);
    cudaFree(dscratch); cudaFree(didx_m); cudaFree(dsc_m); cudaFree(deff_m);
    cudaFree(dout_m); cudaFree(dseq); cudaFree(dlen_m);
    cudaFree(didx_r); cudaFree(dsc_r); cudaFree(deff_r); cudaFree(dout_r); cudaFree(dlen_r);
}

// ── 7. Needle retrieval under selection pressure ───────────────────────────
//
// 4098 tokens => 1024 settled pools across 8 full pages plus a tail-only page.
// One mid-context pool is made the needle (its raw keys are amplified) and the
// query is set to that pool's own dequantized entry, so its logit dominates the
// 1023 random pools. With a budget of 64 pools out of 1024, the needle's four
// token positions must survive expansion — and the always-selected tail must
// appear regardless of scoring.
TEST(IndexerKpool, NeedleUnderPressureRetrieval) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    const PageGeom g{128, 4};            // E = 128 entries, PT = 512 positions
    const int seq_len = 4098;            // 1024 settled pools + 2 tail tokens
    const int n_pools = seq_len / g.P;
    const int full_pages = seq_len / (g.E * g.P);        // 8
    const int tail_rows = seq_len - full_pages * g.E * g.P;  // 2
    const int needle_pool = 700;
    const int budget = 64;
    const int NIH = 8;
    const int out_cols = budget * g.P + g.P - 1;
    const int qpos = seq_len - 1;

    std::mt19937 rng(606060);
    Rows k = make_rows(rng, seq_len, 0.25f);
    const Rows gate = make_rows(rng, seq_len, 0.25f);
    const std::vector<float> ape = make_ape(rng, g.P);

    // Amplify the needle pool's keys so its pooled entry stands out in norm.
    for (int j = 0; j < g.P; ++j) {
        const size_t row = static_cast<size_t>(needle_pool * g.P + j);
        for (int d = 0; d < D; ++d) {
            const float v = k.f[row * D + d] * 4.0f;
            k.bf[row * D + d] = __float2bfloat16_rn(v);
            k.f[row * D + d] = __bfloat162float(k.bf[row * D + d]);
        }
    }

    auto* dk = upload(k.bf);
    auto* dg = upload(gate.bf);
    auto* dape = upload(ape);

    std::vector<void*> pages(static_cast<size_t>(full_pages + 1));
    for (size_t p = 0; p < pages.size(); ++p) pages[p] = zeroed(g.bytes());
    for (int p = 0; p < full_pages; ++p) {
        const size_t row0 = static_cast<size_t>(p) * g.E * g.P;
        chunk_append(pages[static_cast<size_t>(p)], g, dk + row0 * D, dg + row0 * D, dape,
                     g.E * g.P, 0, /*seed_tail=*/false);
    }
    {
        const size_t row0 = static_cast<size_t>(full_pages) * g.E * g.P;
        chunk_append(pages[static_cast<size_t>(full_pages)], g, dk + row0 * D,
                     dg + row0 * D, dape, tail_rows, 0, /*seed_tail=*/true);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Query = the needle entry itself (the "aligned keys" of the needle).
    const int needle_page = needle_pool / g.E;
    const int needle_slot = needle_pool % g.E;
    const auto npage = download<uint8_t>(pages[static_cast<size_t>(needle_page)], g.bytes());
    const float nscale = page_scale(npage, g, needle_slot);
    ASSERT_GT(nscale, 0.0f) << "needle entry was never composed";
    std::vector<__nv_bfloat16> q(static_cast<size_t>(NIH) * D);
    for (int h = 0; h < NIH; ++h) {
        for (int d = 0; d < D; ++d) {
            const float v = kpool_ref::e4m3_to_f32(
                                npage[static_cast<size_t>(needle_slot) * D + d]) * nscale;
            q[static_cast<size_t>(h) * D + d] = __float2bfloat16_rn(v);
        }
    }
    std::vector<float> w(static_cast<size_t>(NIH), 1.0f / NIH);
    auto* dq = upload(q);
    auto* dw = upload(w);

    auto* dscores = static_cast<float*>(zeroed(static_cast<size_t>(n_pools) * sizeof(float)));
    for (int p = 0; p * g.E < n_pools; ++p) {
        const auto* pbase = static_cast<const __nv_fp8_e4m3*>(pages[static_cast<size_t>(p)]);
        si::LightningScoreMqaParams sp{};
        sp.q_proj = dq;
        sp.indexer_k_cache = pbase;
        sp.k_scales = reinterpret_cast<const float*>(
            reinterpret_cast<const uint8_t*>(pbase) + g.scales_off());
        sp.score_proj = dw;
        sp.scores_out = dscores + static_cast<size_t>(p) * g.E;
        sp.num_blocks = std::min(g.E, n_pools - p * g.E);
        sp.index_n_heads = NIH;
        sp.index_head_dim = D;
        lc::launch_lightning_score_mqa(sp, nullptr);
    }

    const auto endpoints = pooled_endpoints(n_pools, g.P);
    auto* dend = upload(endpoints);
    auto* didx = static_cast<int*>(zeroed(static_cast<size_t>(budget) * sizeof(int)));
    auto* dsc = static_cast<float*>(zeroed(static_cast<size_t>(budget) * sizeof(float)));
    auto* deff = static_cast<int*>(zeroed(sizeof(int)));
    auto* dout = static_cast<int*>(zeroed(static_cast<size_t>(out_cols) * sizeof(int)));
    auto* dseq = upload(std::vector<int>{seq_len});
    auto* dlen = static_cast<int*>(zeroed(sizeof(int)));
    pooled_topk(dscores, dend, n_pools, budget, qpos, didx, dsc, deff);
    kpool_expand(didx, deff, dseq, dout, dlen, 1, g.P, budget, out_cols, out_cols);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto sel = download<int>(dout, out_cols);
    const auto len = download<int>(dlen, 1);
    EXPECT_EQ(len[0], budget * g.P + tail_rows);
    for (int j = 0; j < g.P; ++j) {
        const int tok = needle_pool * g.P + j;
        EXPECT_NE(std::find(sel.begin(), sel.end(), tok), sel.end())
            << "needle token " << tok << " missing from the expanded selection";
    }
    // The tail is appended unconditionally (index_kpool_always_select_tail).
    for (int t = full_pages * g.E * g.P; t < seq_len; ++t) {
        EXPECT_NE(std::find(sel.begin(), sel.end(), t), sel.end())
            << "tail token " << t << " missing";
    }

    for (auto* p : pages) cudaFree(p);
    cudaFree(dk); cudaFree(dg); cudaFree(dape); cudaFree(dq); cudaFree(dw);
    cudaFree(dscores); cudaFree(dend); cudaFree(didx); cudaFree(dsc);
    cudaFree(deff); cudaFree(dout); cudaFree(dseq); cudaFree(dlen);
}

// ── 8. Pool-aligned rewind replay is bitwise idempotent ────────────────────
//
// The decode stash is POSITION-keyed (slot = pos % P) and happens AFTER the
// completion read (§2/§8.2), so replaying a blessed pool-aligned rewind with
// identical inputs rewrites exactly its own slots and recomposes the identical
// entry — the whole page must come back bit-for-bit.
TEST(IndexerKpool, RewindPoolAlignedReplayBitwise) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    const PageGeom g{16, 4};
    const int total = 12;
    const int rewind_to = 8;   // pool-aligned
    std::mt19937 rng(24680);
    const Rows k = make_rows(rng, total, 1.0f);
    const Rows gate = make_rows(rng, total, 1.0f);
    const std::vector<float> ape = make_ape(rng, g.P);

    auto* dk = upload(k.bf);
    auto* dg = upload(gate.bf);
    auto* dape = upload(ape);
    void* dpage = zeroed(g.bytes());

    decode_append_range(dpage, g, dk, dg, dape, 0, total, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto snapshot = download<uint8_t>(dpage, g.bytes());

    decode_append_range(dpage, g, dk, dg, dape, rewind_to, total, rewind_to);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto replayed = download<uint8_t>(dpage, g.bytes());

    EXPECT_EQ(std::memcmp(snapshot.data(), replayed.data(), g.bytes()), 0)
        << "pool-aligned rewind replay changed the page";

    cudaFree(dk); cudaFree(dg); cudaFree(dape); cudaFree(dpage);
}

// ── P-29 step 7: device-indexed decode append == host-args append, bitwise ──
//
// The graph-capturable arm resolves page + in-page offset from device state
// (page_table[(seqlen-1)/page_tokens], pos-in-page remainder) at execution
// time; the host arm bakes them into the launch. Same math, same writes —
// the two pages must agree byte-for-byte after the identical token stream,
// including a page-crossing step. Negative control: an off-by-one device
// seqlen must produce a DIFFERENT page (the equality above is not vacuous).
TEST(IndexerKpool, DeviceIndexedDecodeAppendBitwiseMatchesHostArgs) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    const PageGeom g{4, 4};          // small page: PT = 16 → token 16 crosses
    const int PT = g.E * g.P;        // page_tokens
    const int n_tokens = PT + 5;     // crosses into page 1
    std::mt19937 rng(70707);
    const Rows k = make_rows(rng, n_tokens, 1.0f);
    const Rows gate = make_rows(rng, n_tokens, 1.0f);
    const std::vector<float> ape = make_ape(rng, g.P);

    auto* dk = upload(k.bf);
    auto* dg = upload(gate.bf);
    auto* dape = upload(ape);

    // Host-args arm: two pages, explicit page_base / pos_in_page.
    void* ref_p0 = zeroed(g.bytes());
    void* ref_p1 = zeroed(g.bytes());
    decode_append_range(ref_p0, g, dk, dg, dape, 0, PT, 0);
    decode_append_range(ref_p1, g, dk, dg, dape, PT, n_tokens, 0);

    // Device-indexed arm: page table + advancing device seqlen.
    void* dev_p0 = zeroed(g.bytes());
    void* dev_p1 = zeroed(g.bytes());
    const std::vector<const void*> ptab_h{dev_p0, dev_p1};
    auto* dptab = upload(ptab_h);
    int* dseq = static_cast<int*>(zeroed(sizeof(int)));
    for (int t = 0; t < n_tokens; ++t) {
        const int len = t + 1;  // decode step: seqlens_k = pos + 1
        ASSERT_EQ(cudaMemcpy(dseq, &len, sizeof(int), cudaMemcpyHostToDevice),
                  cudaSuccess);
        si::KpoolAppendDecodeParams p{};
        p.k_row = dk + static_cast<size_t>(t) * D;
        p.gate_row = dg + static_cast<size_t>(t) * D;
        p.ape = dape;
        p.page_base = nullptr;   // ignored on the device-indexed arm
        p.page_entries = g.E;
        p.pos_in_page = -12345;  // poison: must never be read
        p.kpool = g.P;
        p.head_dim = D;
        p.page_table = dptab;
        p.seqlen = dseq;
        p.page_tokens = PT;
        lc::launch_kpool_append_decode(p, nullptr);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto r0 = download<uint8_t>(ref_p0, g.bytes());
    const auto r1 = download<uint8_t>(ref_p1, g.bytes());
    const auto d0 = download<uint8_t>(dev_p0, g.bytes());
    const auto d1 = download<uint8_t>(dev_p1, g.bytes());
    EXPECT_EQ(std::memcmp(r0.data(), d0.data(), g.bytes()), 0)
        << "device-indexed append diverged on page 0";
    EXPECT_EQ(std::memcmp(r1.data(), d1.data(), g.bytes()), 0)
        << "device-indexed append diverged on page 1 (page crossing)";

    // Negative control: replay token 3's append with seqlen off by one —
    // the write lands in a different slot, so the page must change.
    {
        void* neg = zeroed(g.bytes());
        const std::vector<const void*> ptab_n{neg, neg};
        auto* dptab_n = upload(ptab_n);
        for (int t = 0; t < 4; ++t) {
            const int len = t + 1 + (t == 3 ? 1 : 0);  // last step off by one
            ASSERT_EQ(cudaMemcpy(dseq, &len, sizeof(int),
                                 cudaMemcpyHostToDevice), cudaSuccess);
            si::KpoolAppendDecodeParams p{};
            p.k_row = dk + static_cast<size_t>(t) * D;
            p.gate_row = dg + static_cast<size_t>(t) * D;
            p.ape = dape;
            p.page_entries = g.E;
            p.pos_in_page = 0;
            p.kpool = g.P;
            p.head_dim = D;
            p.page_table = dptab_n;
            p.seqlen = dseq;
            p.page_tokens = PT;
            lc::launch_kpool_append_decode(p, nullptr);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto pn = download<uint8_t>(neg, g.bytes());
        // Reference for tokens 0..3 straight.
        void* pos_ok = zeroed(g.bytes());
        decode_append_range(pos_ok, g, dk, dg, dape, 0, 4, 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto pr = download<uint8_t>(pos_ok, g.bytes());
        EXPECT_NE(std::memcmp(pr.data(), pn.data(), g.bytes()), 0)
            << "off-by-one seqlen produced an identical page — the bitwise "
               "equality above would be vacuous";
        cudaFree(neg); cudaFree(dptab_n); cudaFree(pos_ok);
    }

    cudaFree(dk); cudaFree(dg); cudaFree(dape);
    cudaFree(ref_p0); cudaFree(ref_p1); cudaFree(dev_p0); cudaFree(dev_p1);
    cudaFree(dptab); cudaFree(dseq);
}

// ── P-29 step 7: B=1 batched score+topk (device bounds) == per-row loop ─────
//
// The kIndexer span replaces the decode producer's per-page score loop +
// single-row top-k (host-scalar bound/cutoff) with ONE batched score + ONE
// batched top-k whose bound/cutoff are DEVICE integers written by
// launch_indexer_decode_bounds from the live seqlen. INV-DSA-BATCH says the
// batched kernels run the exact single-query device bodies — this test pins
// that for the B=1 decode shape: identical scores over [0, nb), identical
// selection (indices AND order AND effective length — DET-TOPK-TIES), and
// bound/cutoff integers identical to the host arithmetic.
TEST(IndexerKpool, DecodeBatchedScoreTopkBitEqualsLoopB1) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    const PageGeom g{8, 4};
    const int n_pages = 3;
    const int entries_per_page[3] = {8, 8, 5};   // partial last page
    const int NB = 21;                           // settled pools
    const int NIH = 8;
    const int SELK = 6;                          // pool budget
    const int len = NB * g.P + 2;                // frontier mid-pool
    const int qpos = len - 1;

    std::mt19937 rng(90210);
    const std::vector<float> ape = make_ape(rng, g.P);
    auto* dape = upload(ape);

    std::vector<void*> pages(n_pages);
    for (int p = 0; p < n_pages; ++p) {
        const int rows = entries_per_page[p] * g.P;
        const Rows kk = make_rows(rng, rows, 1.0f);
        const Rows gg = make_rows(rng, rows, 1.0f);
        pages[static_cast<size_t>(p)] = zeroed(g.bytes());
        auto* dk = upload(kk.bf);
        auto* dg = upload(gg.bf);
        chunk_append(pages[static_cast<size_t>(p)], g, dk, dg, dape, rows, 0,
                     false);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaFree(dk); cudaFree(dg);
    }

    std::vector<__nv_bfloat16> q(static_cast<size_t>(NIH) * D);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : q) v = __float2bfloat16_rn(dist(rng));
    std::vector<float> w(static_cast<size_t>(NIH));
    for (auto& v : w) v = dist(rng) / 8.0f;
    auto* dq = upload(q);
    auto* dw = upload(w);
    const auto endpoints = pooled_endpoints(NB, g.P);
    auto* dend = upload(endpoints);

    // Loop arm (the legacy decode producer): per-page score, single top-k
    // with host-scalar bound/cutoff.
    auto* ls = static_cast<float*>(zeroed(static_cast<size_t>(NB) * 4));
    auto* lidx = static_cast<int*>(zeroed(static_cast<size_t>(SELK) * 4));
    auto* lsc = static_cast<float*>(zeroed(static_cast<size_t>(SELK) * 4));
    auto* leff = static_cast<int*>(zeroed(4));
    for (int p = 0; p * g.E < NB && p < n_pages; ++p) {
        const auto* base =
            static_cast<const __nv_fp8_e4m3*>(pages[static_cast<size_t>(p)]);
        si::LightningScoreMqaParams sp{};
        sp.q_proj = dq;
        sp.indexer_k_cache = base;
        sp.k_scales = reinterpret_cast<const float*>(
            reinterpret_cast<const uint8_t*>(base) + g.scales_off());
        sp.score_proj = dw;
        sp.scores_out = ls + static_cast<size_t>(p) * g.E;
        sp.num_blocks = std::min(g.E, NB - p * g.E);
        sp.index_n_heads = NIH;
        sp.index_head_dim = D;
        lc::launch_lightning_score_mqa(sp, nullptr);
    }
    {
        si::LightningTopkParams tp{};
        tp.scores = ls;
        tp.block_endpoints = dend;
        tp.output_indices = lidx;
        tp.output_scores = lsc;
        tp.effective_k_out = leff;
        tp.num_blocks = NB;
        tp.topk = SELK;
        tp.query_position = qpos;
        lc::launch_lightning_topk(tp, nullptr);
    }

    // Batched arm (the kIndexer span): device bounds from the live seqlen,
    // one batched score + one batched top-k over the page table.
    auto* bs = static_cast<float*>(zeroed(static_cast<size_t>(NB) * 4));
    auto* bidx = static_cast<int*>(zeroed(static_cast<size_t>(SELK) * 4));
    auto* bsc = static_cast<float*>(zeroed(static_cast<size_t>(SELK) * 4));
    auto* beff = static_cast<int*>(zeroed(4));
    int* dbounds = static_cast<int*>(zeroed(2 * sizeof(int)));
    int* dseq = static_cast<int*>(zeroed(sizeof(int)));
    ASSERT_EQ(cudaMemcpy(dseq, &len, sizeof(int), cudaMemcpyHostToDevice),
              cudaSuccess);
    lc::launch_indexer_decode_bounds(dseq, g.P, dbounds, nullptr);
    const std::vector<const void*> ptab_h{pages[0], pages[1], pages[2]};
    auto* dptab = upload(ptab_h);
    {
        si::LightningScoreMqaBatchedParams sp{};
        sp.q_all = dq;
        sp.score_proj_all = dw;
        sp.row_num_blocks = dbounds;
        sp.k_page_table = dptab;
        sp.page_table_stride = n_pages;
        sp.page_tokens = g.E;
        sp.scores_out = bs;
        sp.scores_stride = NB;
        sp.num_rows = 1;
        sp.max_num_blocks = n_pages * g.E;   // over-provisioned grid
        sp.index_n_heads = NIH;
        sp.index_head_dim = D;
        si::run_lightning_score_mqa_batched(sp, nullptr);
    }
    {
        si::LightningTopkBatchedParams tp{};
        tp.scores = bs;
        tp.block_endpoints = dend;
        tp.row_num_blocks = dbounds;
        tp.row_query_position = dbounds + 1;
        tp.output_indices = bidx;
        tp.output_scores = bsc;
        tp.effective_k_out = beff;
        tp.scores_stride = NB;
        tp.num_rows = 1;
        tp.topk = SELK;
        si::run_lightning_topk_batched(tp, nullptr);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Bound/cutoff integers must equal the host arithmetic.
    const auto bounds = download<int>(dbounds, 2);
    EXPECT_EQ(bounds[0], NB) << "device bound != len/kpool";
    EXPECT_EQ(bounds[1], qpos) << "device cutoff != len-1";

    const auto s_loop = download<float>(ls, NB);
    const auto s_bat = download<float>(bs, NB);
    for (int i = 0; i < NB; ++i)
        ASSERT_EQ(s_loop[static_cast<size_t>(i)], s_bat[static_cast<size_t>(i)])
            << "score diverged at pooled entry " << i;
    const auto i_loop = download<int>(lidx, SELK);
    const auto i_bat = download<int>(bidx, SELK);
    for (int i = 0; i < SELK; ++i)
        ASSERT_EQ(i_loop[static_cast<size_t>(i)], i_bat[static_cast<size_t>(i)])
            << "top-k selection diverged at slot " << i << " (DET-TOPK-TIES)";
    const auto sc_loop = download<float>(lsc, SELK);
    const auto sc_bat = download<float>(bsc, SELK);
    for (int i = 0; i < SELK; ++i)
        ASSERT_EQ(sc_loop[static_cast<size_t>(i)],
                  sc_bat[static_cast<size_t>(i)])
            << "top-k score diverged at slot " << i;
    EXPECT_EQ(download<int>(leff, 1)[0], download<int>(beff, 1)[0])
        << "effective k diverged";

    for (auto* p : pages) cudaFree(p);
    cudaFree(dape); cudaFree(dq); cudaFree(dw); cudaFree(dend);
    cudaFree(ls); cudaFree(lidx); cudaFree(lsc); cudaFree(leff);
    cudaFree(bs); cudaFree(bidx); cudaFree(bsc); cudaFree(beff);
    cudaFree(dbounds); cudaFree(dseq); cudaFree(dptab);
}
