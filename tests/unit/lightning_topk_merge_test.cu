// TD-GLM-INDEXER-LOCAL-MERGE: exact cross-rank top-k merge for
// dcp_indexer_mode=local.
//
// Contract under test: per-rank shard-local lightning_topk over each rank's
// OWNED positions (round-robin by indexer page, owner(pos) = (pos/PT) % dcp,
// LOCAL-compacted score arrays) followed by the cross-rank merge
// (indexer_topk_merge: scatter candidates back to global positions over a
// -inf-filled scratch + re-run lightning_topk) must equal
//   (a) a single full-history lightning_topk over the complete score array
//       (the replicated-indexer producer's selection), and
//   (b) an independent CPU top-k reference (score desc, distinct scores),
// including: shards larger than topk (candidate truncation), a rank owning
// ZERO positions (empty candidate list — locks the num_blocks==0 topk launch
// writing the valid empty row), all-causal short rows (len <= topk), 3 ranks,
// and B>1 rows with different lengths.

#include "compute/cuda_sm120_device_backend.h"
#include "compute/kernels/sm120/indexer/lightning_indexer.h"
#include "core/attention_device.h"
#include "daemon/kv_shard_math.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace lc = layerstorm::compute;
namespace cfg = layerstorm::config;

namespace {

cfg::GpuRef make_gpu() { return {0, 0, cfg::GpuType::rtx5090}; }

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

int owner_of(int pos, int PT, int dcp) { return (pos / PT) % dcp; }

// CPU reference: indices of the top-min(len, K) scores (all distinct),
// sorted ascending — the lightning_topk output order contract.
std::vector<int> cpu_topk(const std::vector<float>& scores, int len, int K) {
    std::vector<int> idx(static_cast<size_t>(len));
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
    });
    const int k = std::min(len, K);
    idx.resize(static_cast<size_t>(k));
    std::sort(idx.begin(), idx.end());
    return idx;
}

// Distinct pseudo-random scores (positive and negative, no ties → the CPU
// reference is the unique exact answer).
std::vector<float> distinct_scores(std::mt19937& rng, int n) {
    std::vector<float> s(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        s[static_cast<size_t>(i)] = (i - n / 2) * 0.125f;
    std::shuffle(s.begin(), s.end(), rng);
    return s;
}

struct Case {
    int max_len;   // row 0's length; row b's length = max_len - 33*b
    int K, PT, dcp, B;
};

}  // namespace

TEST(LightningTopkMerge, MergeEqualsFullHistoryTopk) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    lc::CudaSm120DeviceBackend dev(make_gpu());
    std::mt19937 rng(20260706);

    const Case cases[] = {
        {2877, 2048, 1024, 2, 1},  // golden shape: both shards < K, prune
        {300, 64, 32, 2, 1},       // both shards > K: candidate truncation
        {100, 64, 128, 2, 1},      // rank 1 owns ZERO positions (len < PT)
        {50, 64, 16, 3, 1},        // len <= K (all causal selected), 3 ranks
        {130, 32, 16, 2, 2},       // B=2 rows, different lengths (130, 97)
    };

    for (const auto& c : cases) {
        SCOPED_TRACE("max_len=" + std::to_string(c.max_len)
                     + " K=" + std::to_string(c.K)
                     + " PT=" + std::to_string(c.PT)
                     + " dcp=" + std::to_string(c.dcp)
                     + " B=" + std::to_string(c.B));
        const int K = c.K;
        std::vector<int> lens(static_cast<size_t>(c.B));
        for (int b = 0; b < c.B; ++b)
            lens[static_cast<size_t>(b)] = c.max_len - 33 * b;

        // Per-row full score arrays (host).
        std::vector<std::vector<float>> scores(static_cast<size_t>(c.B));
        for (int b = 0; b < c.B; ++b)
            scores[static_cast<size_t>(b)] =
                distinct_scores(rng, lens[static_cast<size_t>(b)]);

        // Static iota endpoints — serves the full run, the shard runs
        // (local slot i < owned <= len ⇒ endpoint <= qpos, causality
        // vacuous exactly as in the producer), and the merge.
        std::vector<int> iota(static_cast<size_t>(c.max_len));
        std::iota(iota.begin(), iota.end(), 0);
        int* d_endpoints = upload(iota);

        // Candidate buffer, NCCL-allgather layout: dcp segments of
        // seg_words = 2*B*K words, each [B*K int32 indices][B*K f32 scores].
        const int seg_words = 2 * c.B * K;
        std::vector<int> cand_host(
            static_cast<size_t>(c.dcp) * seg_words, -7777);
        int* d_cand = upload(cand_host);
        float* d_out_scores = nullptr;
        ASSERT_EQ(cudaMalloc(&d_out_scores, sizeof(float) * K), cudaSuccess);
        int* d_len1 = nullptr;
        ASSERT_EQ(cudaMalloc(&d_len1, sizeof(int)), cudaSuccess);

        // 1) Shard-local top-k per (rank, row) → candidate rows.
        for (int b = 0; b < c.B; ++b) {
            const int len = lens[static_cast<size_t>(b)];
            const auto& s = scores[static_cast<size_t>(b)];
            for (int n = 0; n < c.dcp; ++n) {
                // LOCAL-compacted owned scores (page-aligned compaction).
                std::vector<float> local;
                for (int pos = 0; pos < len; ++pos)
                    if (owner_of(pos, c.PT, c.dcp) == n)
                        local.push_back(s[static_cast<size_t>(pos)]);
                float* d_local = nullptr;
                if (!local.empty()) d_local = upload(local);

                int* seg = d_cand + static_cast<size_t>(n) * seg_words;
                sm120::indexer::LightningTopkParams tp{};
                tp.scores = d_local;
                tp.block_endpoints = d_endpoints;
                tp.output_indices = seg + static_cast<size_t>(b) * K;
                tp.output_scores = reinterpret_cast<float*>(
                    seg + static_cast<size_t>(c.B) * K)
                    + static_cast<size_t>(b) * K;
                tp.effective_k_out = d_len1;  // scratch (merge rewrites)
                tp.num_blocks = static_cast<int>(local.size());
                tp.topk = K;
                tp.query_position = len - 1;
                lc::launch_lightning_topk(tp, nullptr);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

                // Empty shard: the num_blocks==0 launch must still write the
                // valid empty candidate row (all −1) — the merge consumes it.
                if (local.empty()) {
                    auto row = download<int>(seg + static_cast<size_t>(b) * K,
                                             static_cast<size_t>(K));
                    for (int i = 0; i < K; ++i)
                        ASSERT_EQ(row[static_cast<size_t>(i)], -1)
                            << "empty-shard candidate entry " << i;
                }
                if (d_local) cudaFree(d_local);
            }
        }

        // 2) Cross-rank merge per row via the backend seam.
        float* d_scratch = nullptr;
        ASSERT_EQ(cudaMalloc(&d_scratch, sizeof(float) * c.max_len),
                  cudaSuccess);
        int* d_merged = nullptr;
        ASSERT_EQ(cudaMalloc(&d_merged, sizeof(int) * c.B * K), cudaSuccess);
        int* d_merged_len = nullptr;
        ASSERT_EQ(cudaMalloc(&d_merged_len, sizeof(int) * c.B), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_merged, 0xAB, sizeof(int) * c.B * K),
                  cudaSuccess);

        for (int b = 0; b < c.B; ++b) {
            const int len = lens[static_cast<size_t>(b)];
            lc::IndexerTopkMergeArgs ma{};
            ma.gathered = d_cand;
            ma.seg_words = seg_words;
            ma.batch = c.B;
            ma.token = b;
            ma.scores_scratch = d_scratch;
            ma.block_endpoints = d_endpoints;
            ma.topk_scores_scratch = d_out_scores;
            ma.indices_out = d_merged + static_cast<size_t>(b) * K;
            ma.length_out = d_merged_len + b;
            ma.num_blocks = len;
            ma.topk = K;
            ma.query_position = len - 1;
            ma.dcp_size = c.dcp;
            ma.page_tokens = c.PT;
            dev.indexer_topk_merge(ma, nullptr);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        // 3a) Full-history single top-k (replicated-mode selection).
        int* d_full = nullptr;
        ASSERT_EQ(cudaMalloc(&d_full, sizeof(int) * K), cudaSuccess);
        int* d_full_len = nullptr;
        ASSERT_EQ(cudaMalloc(&d_full_len, sizeof(int)), cudaSuccess);

        for (int b = 0; b < c.B; ++b) {
            const int len = lens[static_cast<size_t>(b)];
            float* d_scores = upload(scores[static_cast<size_t>(b)]);
            sm120::indexer::LightningTopkParams tp{};
            tp.scores = d_scores;
            tp.block_endpoints = d_endpoints;
            tp.output_indices = d_full;
            tp.output_scores = d_out_scores;
            tp.effective_k_out = d_full_len;
            tp.num_blocks = len;
            tp.topk = K;
            tp.query_position = len - 1;
            lc::launch_lightning_topk(tp, nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            cudaFree(d_scores);

            const auto full = download<int>(d_full, static_cast<size_t>(K));
            const auto full_len = download<int>(d_full_len, 1);
            const auto merged = download<int>(
                d_merged + static_cast<size_t>(b) * K, static_cast<size_t>(K));
            const auto merged_len = download<int>(d_merged_len + b, 1);

            // 3b) Independent CPU reference (distinct scores → unique).
            const auto ref = cpu_topk(scores[static_cast<size_t>(b)], len, K);

            ASSERT_EQ(merged_len[0], static_cast<int>(ref.size()))
                << "row " << b;
            ASSERT_EQ(full_len[0], static_cast<int>(ref.size()))
                << "row " << b << " (full-run harness sanity)";
            for (int i = 0; i < K; ++i) {
                const int want = i < static_cast<int>(ref.size())
                    ? ref[static_cast<size_t>(i)] : -1;
                ASSERT_EQ(merged[static_cast<size_t>(i)], want)
                    << "row " << b << " entry " << i << " (vs CPU ref)";
                ASSERT_EQ(merged[static_cast<size_t>(i)],
                          full[static_cast<size_t>(i)])
                    << "row " << b << " entry " << i << " (vs full run)";
            }
        }

        cudaFree(d_full);
        cudaFree(d_full_len);
        cudaFree(d_merged);
        cudaFree(d_merged_len);
        cudaFree(d_scratch);
        cudaFree(d_cand);
        cudaFree(d_out_scores);
        cudaFree(d_len1);
        cudaFree(d_endpoints);
    }
}

// TD-SPARSE-PREFILL-LOCAL-INDEXER: the SPARSE CHUNK PREFILL local-indexer
// regime. Chunk row b (one sequence, consecutive positions, len_b =
// chunk_start + b + 1) must get the SAME per-row global top-k the replicated
// producer emits, via: per-rank shard-local lightning_topk over the rank's
// LOCAL-compacted stored keys — which, unlike decode, already contain the
// WHOLE chunk (append_indexer_chunk runs first), so the shard scoring is
// bounded PER ROW at nb = owned_len(rank, len_b) — followed by the per-row
// cross-rank merge (indexer_topk_merge with row b's causal bound/cutoff).
//
// The per-row shard bound is an EXACTNESS requirement, not just causality:
// scoring the shard's later-chunk keys can crowd true causal candidates out
// of the ≤K candidate list BEFORE the merge's causal cutoff. Locked by a
// VERIFIED NEGATIVE CONTROL (tail-heavy scores + unbounded shard scoring ⇒
// the merge provably diverges from the replicated selection).
//
// Covers: scattered top-k (shuffled distinct scores), partial pages
// (len_b % PT != 0), empty shard rows (a rank owning ZERO causal positions
// for early chunk rows), chunk_start == 0 (fresh prefill, row 0 len 1),
// chunk_start > 0 (chunked continuation), 3 ranks, and the owned_len bound
// == the structural count of owned causal positions (kv_shard_math tie-in).
TEST(LightningTopkMerge, PrefillChunkRowsShardBoundMergeEqualsReplicatedTopk) {
    REQUIRES_GPU();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    (void)cudaGetLastError();

    lc::CudaSm120DeviceBackend dev(make_gpu());
    std::mt19937 rng(20260708);

    struct PrefillCase {
        int len_total;   // stored keys after the chunk append
        int chunk_len;   // B: chunk rows; chunk_start = len_total - chunk_len
        int K, PT, dcp;
        bool tail_heavy; // scores ascend with position (crowding adversary)
    };
    const PrefillCase cases[] = {
        {300, 64, 48, 32, 2, false},   // partial pages, scattered top-k
        {200, 200, 64, 128, 2, false}, // chunk_start=0; rank1 empty for
                                       // early rows (len_b <= 128); row0 len1
        {150, 30, 16, 8, 3, false},    // 3 ranks, small pages
        {256, 96, 32, 16, 2, true},    // crowding adversary + neg. control
    };

    for (const auto& c : cases) {
        SCOPED_TRACE("len_total=" + std::to_string(c.len_total)
                     + " chunk_len=" + std::to_string(c.chunk_len)
                     + " K=" + std::to_string(c.K)
                     + " PT=" + std::to_string(c.PT)
                     + " dcp=" + std::to_string(c.dcp)
                     + " tail_heavy=" + std::to_string(c.tail_heavy));
        const int K = c.K;
        const int B = c.chunk_len;
        const int chunk_start = c.len_total - c.chunk_len;

        // Per-row full score arrays over ALL stored keys [0, len_total) —
        // each chunk row queries with its own q, so scores are row-specific;
        // non-causal positions (>= len_b) have scores too (they are stored).
        // tail_heavy: score strictly ascends with position (distinct), so
        // every later-chunk key beats every causal one.
        std::vector<std::vector<float>> scores(static_cast<size_t>(B));
        for (int b = 0; b < B; ++b) {
            if (c.tail_heavy) {
                std::vector<float> s(static_cast<size_t>(c.len_total));
                for (int i = 0; i < c.len_total; ++i)
                    s[static_cast<size_t>(i)] =
                        (i - c.len_total / 2) * 0.25f + b * 0.001f;
                scores[static_cast<size_t>(b)] = std::move(s);
            } else {
                scores[static_cast<size_t>(b)] =
                    distinct_scores(rng, c.len_total);
            }
        }

        std::vector<int> iota(static_cast<size_t>(c.len_total));
        std::iota(iota.begin(), iota.end(), 0);
        int* d_endpoints = upload(iota);

        const int seg_words = 2 * B * K;
        std::vector<int> cand_host(
            static_cast<size_t>(c.dcp) * seg_words, -7777);
        int* d_cand = upload(cand_host);
        int* d_cand_unbounded = upload(cand_host);  // negative control
        float* d_out_scores = nullptr;
        ASSERT_EQ(cudaMalloc(&d_out_scores, sizeof(float) * K), cudaSuccess);
        int* d_len1 = nullptr;
        ASSERT_EQ(cudaMalloc(&d_len1, sizeof(int)), cudaSuccess);

        // 1) Per (rank, chunk row): shard-local top-k over the rank's
        //    LOCAL-compacted stored keys, bounded at owned_len(r, len_b)
        //    (the producer's nb). The local array holds the rank's owned
        //    positions over the FULL stored range — exactly the executor's
        //    storage shape after append_indexer_chunk.
        for (int b = 0; b < B; ++b) {
            const int len_b = chunk_start + b + 1;
            for (int n = 0; n < c.dcp; ++n) {
                std::vector<float> local;  // over [0, len_total): stored keys
                int causal_owned = 0;      // structural count < len_b
                for (int pos = 0; pos < c.len_total; ++pos)
                    if (owner_of(pos, c.PT, c.dcp) == n) {
                        local.push_back(
                            scores[static_cast<size_t>(b)]
                                  [static_cast<size_t>(pos)]);
                        if (pos < len_b) ++causal_owned;
                    }
                // The executor's per-row bound == structural causal count.
                const int nb = layerstorm::daemon::kvshard::owned_len(
                    n, static_cast<uint32_t>(len_b), c.PT, c.dcp);
                ASSERT_EQ(nb, causal_owned)
                    << "row " << b << " rank " << n;

                float* d_local = nullptr;
                if (!local.empty()) d_local = upload(local);

                auto shard_topk = [&](int* cand_base, int bound) {
                    int* seg = cand_base + static_cast<size_t>(n) * seg_words;
                    sm120::indexer::LightningTopkParams tp{};
                    tp.scores = d_local;
                    tp.block_endpoints = d_endpoints;
                    tp.output_indices = seg + static_cast<size_t>(b) * K;
                    tp.output_scores = reinterpret_cast<float*>(
                        seg + static_cast<size_t>(B) * K)
                        + static_cast<size_t>(b) * K;
                    tp.effective_k_out = d_len1;  // scratch (merge rewrites)
                    tp.num_blocks = bound;
                    tp.topk = K;
                    tp.query_position = len_b - 1;
                    lc::launch_lightning_topk(tp, nullptr);
                };
                shard_topk(d_cand, nb);  // the producer's bounded scoring
                // Negative control: score the WHOLE stored shard (the bug
                // the bound prevents — later-chunk keys enter the list).
                shard_topk(d_cand_unbounded, static_cast<int>(local.size()));
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                if (d_local) cudaFree(d_local);
            }
        }

        // 2) Per-row cross-rank merge (both candidate sets).
        float* d_scratch = nullptr;
        ASSERT_EQ(cudaMalloc(&d_scratch, sizeof(float) * c.len_total),
                  cudaSuccess);
        int* d_merged = nullptr;
        ASSERT_EQ(cudaMalloc(&d_merged, sizeof(int) * 2 * B * K),
                  cudaSuccess);
        int* d_merged_len = nullptr;
        ASSERT_EQ(cudaMalloc(&d_merged_len, sizeof(int) * 2 * B),
                  cudaSuccess);
        ASSERT_EQ(cudaMemset(d_merged, 0xAB, sizeof(int) * 2 * B * K),
                  cudaSuccess);
        int* d_merged_unb = d_merged + static_cast<size_t>(B) * K;
        int* d_merged_len_unb = d_merged_len + B;

        for (int b = 0; b < B; ++b) {
            const int len_b = chunk_start + b + 1;
            auto merge = [&](const int* gathered, int* out, int* len_out) {
                lc::IndexerTopkMergeArgs ma{};
                ma.gathered = gathered;
                ma.seg_words = seg_words;
                ma.batch = B;
                ma.token = b;
                ma.scores_scratch = d_scratch;
                ma.block_endpoints = d_endpoints;
                ma.topk_scores_scratch = d_out_scores;
                ma.indices_out = out + static_cast<size_t>(b) * K;
                ma.length_out = len_out + b;
                ma.num_blocks = len_b;
                ma.topk = K;
                ma.query_position = len_b - 1;
                ma.dcp_size = c.dcp;
                ma.page_tokens = c.PT;
                dev.indexer_topk_merge(ma, nullptr);
            };
            merge(d_cand, d_merged, d_merged_len);
            merge(d_cand_unbounded, d_merged_unb, d_merged_len_unb);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        // 3) Per row: bounded merge == replicated full-history top-k ==
        //    CPU reference; the unbounded negative control must DIVERGE on
        //    the tail-heavy adversary (proves the bound is load-bearing and
        //    this test catches its absence).
        int* d_full = nullptr;
        ASSERT_EQ(cudaMalloc(&d_full, sizeof(int) * K), cudaSuccess);
        int* d_full_len = nullptr;
        ASSERT_EQ(cudaMalloc(&d_full_len, sizeof(int)), cudaSuccess);
        int diverged_rows = 0;

        for (int b = 0; b < B; ++b) {
            const int len_b = chunk_start + b + 1;
            float* d_scores = upload(scores[static_cast<size_t>(b)]);
            sm120::indexer::LightningTopkParams tp{};
            tp.scores = d_scores;
            tp.block_endpoints = d_endpoints;
            tp.output_indices = d_full;
            tp.output_scores = d_out_scores;
            tp.effective_k_out = d_full_len;
            tp.num_blocks = len_b;  // replicated bound: the causal prefix
            tp.topk = K;
            tp.query_position = len_b - 1;
            lc::launch_lightning_topk(tp, nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            cudaFree(d_scores);

            const auto full = download<int>(d_full, static_cast<size_t>(K));
            const auto full_len = download<int>(d_full_len, 1);
            const auto merged = download<int>(
                d_merged + static_cast<size_t>(b) * K, static_cast<size_t>(K));
            const auto merged_len = download<int>(d_merged_len + b, 1);
            const auto ref =
                cpu_topk(scores[static_cast<size_t>(b)], len_b, K);

            ASSERT_EQ(merged_len[0], static_cast<int>(ref.size()))
                << "row " << b;
            ASSERT_EQ(full_len[0], static_cast<int>(ref.size()))
                << "row " << b << " (full-run harness sanity)";
            for (int i = 0; i < K; ++i) {
                const int want = i < static_cast<int>(ref.size())
                    ? ref[static_cast<size_t>(i)] : -1;
                ASSERT_EQ(merged[static_cast<size_t>(i)], want)
                    << "row " << b << " entry " << i << " (vs CPU ref)";
                ASSERT_EQ(merged[static_cast<size_t>(i)],
                          full[static_cast<size_t>(i)])
                    << "row " << b << " entry " << i << " (vs full run)";
            }

            if (c.tail_heavy) {
                const auto unb = download<int>(
                    d_merged_unb + static_cast<size_t>(b) * K,
                    static_cast<size_t>(K));
                const auto unb_len = download<int>(d_merged_len_unb + b, 1);
                bool same = unb_len[0] == merged_len[0];
                for (int i = 0; same && i < K; ++i)
                    same = unb[static_cast<size_t>(i)]
                        == merged[static_cast<size_t>(i)];
                if (!same) ++diverged_rows;
            }
        }
        if (c.tail_heavy) {
            // Rows whose causal prefix extends >= K*dcp beyond nothing —
            // with strictly ascending scores every shard's unbounded
            // candidate list is the shard's LAST K stored keys; for rows
            // len_b <= len_total - K*dcp they are all non-causal → the
            // cutoff empties the merge. At least those rows must diverge.
            EXPECT_GT(diverged_rows, 0)
                << "negative control failed: unbounded shard scoring was "
                   "not caught (the per-row owned_len bound looks "
                   "non-load-bearing)";
        }

        cudaFree(d_full);
        cudaFree(d_full_len);
        cudaFree(d_merged);
        cudaFree(d_merged_len);
        cudaFree(d_scratch);
        cudaFree(d_cand);
        cudaFree(d_cand_unbounded);
        cudaFree(d_out_scores);
        cudaFree(d_len1);
        cudaFree(d_endpoints);
    }
}
