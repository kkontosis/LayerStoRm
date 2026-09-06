// GLM-25k: DSA-guided KV tiering — unit tests.
//
// Covers (spec/TESTING.md Tier 2A, REQUIRES_GPU-gated):
//   1. kv_row_copy gather/scatter kernels (byte-exact round trips).
//   2. KvTieringManager materialization: hot D2D + cold staged H2D into the
//      dense scratch equals the true selected rows byte-for-byte — INCLUDING
//      after the demoted pool pages are clobbered (proves the cold copy is
//      the source, not the freed VRAM page).
//   3. Hot row-cache behavior: second identical selection is all cache hits;
//      LRU eviction under a tiny hot buffer stays byte-correct.
//   4. Demotion bookkeeping: free_page called once per demoted page with the
//      right (layer, logical); retention window respected; sticky-dense
//      layers never demote; on_dense_layer throws on cold layers (INV-KVT-2).
//   5. NUMA home-node placement of the cold pool / staging arena (P-22 rule).
//   6. TD-KVT-SYNC/-PREFETCH: prepare() overlapped readback (host wait ~0
//      even under a busy attention stream), IndexShare host-copy reuse (the
//      device selection buffer is CLOBBERED after the full layer — bytes
//      must come from the host copy), IndexShare lookahead prefetch into the
//      successor's row cache (successor materializes with ZERO cold misses,
//      byte-identical to the sync-fetch path), stale-pending drain across
//      steps.
//   7. TD-KVT-SPEC: cold_page_host_ptr serves the snapshot path (demoted
//      page bytes byte-exact from the pinned cold pool after the VRAM pages
//      are clobbered); position rewind allowed at/above the demoted
//      frontier, throws below it.
//   8. TD-KVT-DCP-SHARDED: two-rank sharded fixture (round-robin page
//      ownership) — owner-only demotion (one cold copy, in the owner's
//      pool), per-rank materialization of translated LOCAL selections
//      byte-exact vs global ground truth, owner-pool cold_page_host_ptr.
//   9. TD-KVT-PREFILL: the B==1 sparse chunked-prefill step flow (one
//      begin_layer/after_attention per prompt position) demotes behind the
//      chunk frontier DURING prefill; a multi-turn append into the demoted
//      sequence stays tierable (no re-promotion) and materializes its
//      selection byte-exact with the demoted VRAM pages clobbered.

#include "daemon/kv_tiering_manager.h"

#include "compute/cuda_sm120_device_backend.h"
#include "compute/kernels/kv/kv_row_copy.h"
#include "core/device_backend.h"
#include "core/memory/numa_manager.h"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <dirent.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "../gpu_test_utils.h"

namespace lc = layerstorm::compute;
namespace ld = layerstorm::daemon;
namespace lm = layerstorm::memory;
namespace cfg = layerstorm::config;

namespace {

constexpr int kPageSize = 4;          // tokens per page (small for tests)
constexpr int kStrideRow = 656;       // bytes per row (SnapMLA-like: 16-mult)
constexpr int64_t kStrideBlock = static_cast<int64_t>(kPageSize) * kStrideRow;
constexpr int kTopk = 16;
constexpr int kLayers = 2;

cfg::GpuRef ref0() { return {.position = 0, .id = 0,
                             .type = cfg::GpuType::rtx5090}; }

/// Deterministic per-(layer, position) row pattern.
void fill_row(char* dst, int layer, int pos) {
    for (int b = 0; b < kStrideRow; ++b)
        dst[b] = static_cast<char>((layer * 131 + pos * 31 + b) & 0xFF);
}

struct PoolFixture {
    std::unique_ptr<lc::CudaSm120DeviceBackend> be;
    void* pool = nullptr;             // device: kPoolPages pages
    void* stream = nullptr;
    int pool_pages = 0;
    std::vector<char> host_pool;      // mirror of initial pool contents

    explicit PoolFixture(int num_pages, int layer_for_pattern) {
        be = std::make_unique<lc::CudaSm120DeviceBackend>(ref0());
        be->set_device();
        pool_pages = num_pages;
        const size_t bytes = static_cast<size_t>(num_pages) * kStrideBlock;
        pool = be->device_alloc(bytes);
        stream = be->create_stream();
        host_pool.resize(bytes);
        // Physical page p holds positions p*kPageSize .. +kPageSize-1
        // (identity logical→physical mapping in these tests).
        for (int p = 0; p < num_pages; ++p)
            for (int r = 0; r < kPageSize; ++r)
                fill_row(host_pool.data() + p * kStrideBlock + r * kStrideRow,
                         layer_for_pattern, p * kPageSize + r);
        be->memcpy_h2d(pool, host_pool.data(), bytes);
    }
    ~PoolFixture() {
        be->set_device();
        if (stream) be->destroy_stream(stream);
        be->device_free(pool);
    }
};

}  // namespace

// ── 1. kv_row_copy kernels ──────────────────────────────────────────────────

TEST(KvRowCopyKernel, GatherFromScatteredSources) {
    REQUIRES_GPU();
    PoolFixture fx(/*num_pages=*/8, /*layer=*/0);
    auto* be = fx.be.get();

    // Select scattered positions; source ptr per row.
    std::vector<int> sel = {31, 0, 17, 5, 5, 30, 12};
    const int n = static_cast<int>(sel.size());
    std::vector<const void*> h_src(n);
    for (int i = 0; i < n; ++i)
        h_src[i] = static_cast<char*>(fx.pool)
            + (sel[i] / kPageSize) * kStrideBlock
            + (sel[i] % kPageSize) * kStrideRow;

    void* dev_src = be->device_alloc(n * sizeof(void*));
    be->memcpy_h2d(dev_src, h_src.data(), n * sizeof(void*));

    const int fake_pages = (n + kPageSize - 1) / kPageSize;
    void* scratch = be->device_alloc(fake_pages * kStrideBlock);

    lc::launch_kv_row_gather(scratch, kStrideBlock, kStrideRow, kPageSize,
                             reinterpret_cast<const void* const*>(dev_src),
                             n, kStrideRow, fx.stream);
    be->synchronize_device();

    std::vector<char> out(fake_pages * kStrideBlock);
    std::vector<char> want(kStrideRow);
    // D2H compare
    std::vector<char> full(out.size());
    be->memcpy_d2h_async(full.data(), scratch, out.size(), fx.stream);
    be->synchronize_device();
    for (int i = 0; i < n; ++i) {
        const char* got = full.data() + (i / kPageSize) * kStrideBlock
                        + (i % kPageSize) * kStrideRow;
        fill_row(want.data(), 0, sel[i]);
        ASSERT_EQ(std::memcmp(got, want.data(), kStrideRow), 0)
            << "gather row " << i << " (pos " << sel[i] << ") mismatch";
    }
    be->device_free(scratch);
    be->device_free(dev_src);
}

TEST(KvRowCopyKernel, ScatterToArbitraryDestinations) {
    REQUIRES_GPU();
    auto be = std::make_unique<lc::CudaSm120DeviceBackend>(ref0());
    be->set_device();
    void* stream = be->create_stream();

    const int n = 5;
    std::vector<char> packed(n * kStrideRow);
    for (int i = 0; i < n; ++i)
        fill_row(packed.data() + i * kStrideRow, 1, 100 + i);
    void* dev_packed = be->device_alloc(packed.size());
    be->memcpy_h2d(dev_packed, packed.data(), packed.size());

    // Destination slots scattered inside a cache-like buffer (reversed order,
    // via an index table selecting packed rows 4,3,2,1,0).
    const int slots = 9;
    void* cache = be->device_alloc(static_cast<size_t>(slots) * kStrideRow);
    std::vector<void*> h_dst(n);
    std::vector<int> h_idx(n);
    for (int i = 0; i < n; ++i) {
        h_dst[i] = static_cast<char*>(cache) + (2 * i) * kStrideRow;
        h_idx[i] = n - 1 - i;
    }
    void* dev_dst = be->device_alloc(n * sizeof(void*));
    void* dev_idx = be->device_alloc(n * sizeof(int));
    be->memcpy_h2d(dev_dst, h_dst.data(), n * sizeof(void*));
    be->memcpy_h2d(dev_idx, h_idx.data(), n * sizeof(int));

    lc::launch_kv_row_scatter(reinterpret_cast<void* const*>(dev_dst),
                              dev_packed, kStrideRow,
                              static_cast<const int*>(dev_idx), n,
                              kStrideRow, stream);
    be->synchronize_device();

    std::vector<char> out(static_cast<size_t>(slots) * kStrideRow);
    be->memcpy_d2h_async(out.data(), cache, out.size(), stream);
    be->synchronize_device();
    std::vector<char> want(kStrideRow);
    for (int i = 0; i < n; ++i) {
        fill_row(want.data(), 1, 100 + (n - 1 - i));
        ASSERT_EQ(std::memcmp(out.data() + (2 * i) * kStrideRow, want.data(),
                              kStrideRow), 0)
            << "scatter slot " << i << " mismatch";
    }
    be->device_free(dev_idx);
    be->device_free(dev_dst);
    be->device_free(cache);
    be->device_free(dev_packed);
    be->destroy_stream(stream);
}

TEST(KvRowCopyKernel, GatherUnalignedTqGeometryRows) {
    REQUIRES_GPU();
    // TD-KVT-TQ: TurboQuant MSE4 rows are 386 B (256 B packed 4-bit nope +
    // 2 B fp16 norm + 128 B BF16 rope) — NOT 16 B-aligned, so every
    // source/destination hits copy_row's byte fallback.  Full gather →
    // scatter → gather round trip at TQ geometry must be byte-exact.
    auto be = std::make_unique<lc::CudaSm120DeviceBackend>(ref0());
    be->set_device();
    void* stream = be->create_stream();

    constexpr int kTqRow = 386;   // d_c/2 + 2 + d_rope*2 (V3.2 geometry)
    constexpr int kTqPage = 16;
    constexpr int64_t kTqBlock = static_cast<int64_t>(kTqPage) * kTqRow;
    const int pool_pages = 4;

    std::vector<char> host_pool(static_cast<size_t>(pool_pages) * kTqBlock);
    for (size_t b = 0; b < host_pool.size(); ++b)
        host_pool[b] = static_cast<char>((b * 2654435761u >> 16) & 0xFF);
    void* pool = be->device_alloc(host_pool.size());
    be->memcpy_h2d(pool, host_pool.data(), host_pool.size());

    std::vector<int> sel = {63, 0, 17, 41, 5, 40, 62, 33};
    const int n = static_cast<int>(sel.size());
    std::vector<const void*> h_src(n);
    for (int i = 0; i < n; ++i)
        h_src[i] = static_cast<char*>(pool)
            + (sel[i] / kTqPage) * kTqBlock + (sel[i] % kTqPage) * kTqRow;
    void* dev_src = be->device_alloc(n * sizeof(void*));
    be->memcpy_h2d(dev_src, h_src.data(), n * sizeof(void*));

    const int fake_pages = (n + kTqPage - 1) / kTqPage;
    void* scratch = be->device_alloc(fake_pages * kTqBlock);
    lc::launch_kv_row_gather(scratch, kTqBlock, kTqRow, kTqPage,
                             reinterpret_cast<const void* const*>(dev_src),
                             n, kTqRow, stream);
    be->synchronize_device();

    std::vector<char> out(static_cast<size_t>(fake_pages) * kTqBlock);
    be->memcpy_d2h_async(out.data(), scratch, out.size(), stream);
    be->synchronize_device();
    for (int i = 0; i < n; ++i) {
        const char* got = out.data() + (i / kTqPage) * kTqBlock
                        + (i % kTqPage) * kTqRow;
        const char* want = host_pool.data()
            + (sel[i] / kTqPage) * kTqBlock + (sel[i] % kTqPage) * kTqRow;
        ASSERT_EQ(std::memcmp(got, want, kTqRow), 0)
            << "TQ-geometry gather row " << i << " (pos " << sel[i]
            << ") mismatch";
    }

    // Scatter the packed rows to unaligned cache-slot destinations and read
    // them back (row-cache insert path at TQ geometry).
    void* cache = be->device_alloc(static_cast<size_t>(n) * kTqRow);
    std::vector<void*> h_dst(n);
    std::vector<int> h_idx(n);
    for (int i = 0; i < n; ++i) {
        h_dst[i] = static_cast<char*>(cache)
            + static_cast<size_t>(n - 1 - i) * kTqRow;
        h_idx[i] = i;
    }
    void* dev_dst = be->device_alloc(n * sizeof(void*));
    void* dev_idx = be->device_alloc(n * sizeof(int));
    be->memcpy_h2d(dev_dst, h_dst.data(), n * sizeof(void*));
    be->memcpy_h2d(dev_idx, h_idx.data(), n * sizeof(int));
    // Source: densely packed rows (stride == row) — reuse the scratch's
    // page-0 rows only when n <= page; here n=8 <= 16 so page 0 is packed.
    lc::launch_kv_row_scatter(reinterpret_cast<void* const*>(dev_dst),
                              scratch, kTqRow,
                              static_cast<const int*>(dev_idx), n,
                              kTqRow, stream);
    be->synchronize_device();
    std::vector<char> cout(static_cast<size_t>(n) * kTqRow);
    be->memcpy_d2h_async(cout.data(), cache, cout.size(), stream);
    be->synchronize_device();
    for (int i = 0; i < n; ++i) {
        const char* got = cout.data() + static_cast<size_t>(n - 1 - i) * kTqRow;
        const char* want = host_pool.data()
            + (sel[i] / kTqPage) * kTqBlock + (sel[i] % kTqPage) * kTqRow;
        ASSERT_EQ(std::memcmp(got, want, kTqRow), 0)
            << "TQ-geometry scatter row " << i << " mismatch";
    }
    be->device_free(dev_idx);
    be->device_free(dev_dst);
    be->device_free(cache);
    be->device_free(scratch);
    be->device_free(dev_src);
    be->device_free(pool);
    be->destroy_stream(stream);
}

// ── 2-4. KvTieringManager ───────────────────────────────────────────────────

namespace {

struct ManagerFixture {
    PoolFixture fx;
    std::unique_ptr<ld::KvTieringManager> mgr;
    std::vector<lm::PageHandle> handles;  // (logical j, layer l) at [j*L + l]
    std::vector<int> host_bt;             // identity logical→physical
    std::vector<std::pair<int, int>> freed;  // (layer, logical) free order
    void* dev_indices = nullptr;
    void* dev_len = nullptr;
    // TD-KVT-SPEC-FORK re-promotion seam: spare pool pages back the fixture
    // alloc_page callback (fresh VRAM page per re-promoted page; the fixture
    // handle table + block table are updated like the dispatcher's
    // seq_pages_ un-neutralize + dirty-guard poison).
    int next_free_page = 0;
    std::vector<std::array<int, 3>> alloced;  // (layer, logical, phys)

    explicit ManagerFixture(int hot_slots, int num_logical = 8,
                            std::vector<uint8_t> full_mask = {},
                            double host_to_device_ratio = 8.0,
                            int spare_pages = 0,
                            int cohort_rows = 0,
                            int union_rows = 0,
                            std::vector<uint8_t> bearing_mask = {})
        : fx(num_logical + spare_pages, /*layer=*/0),
          next_free_page(num_logical) {
        auto* be = fx.be.get();
        // Layer-major handles: layer l of logical j → physical page j (both
        // layers alias the same pool page in this fixture; materialization
        // is exercised on layer 0 only, layer 1 exists for demote scoping).
        for (int j = 0; j < num_logical; ++j)
            for (int l = 0; l < kLayers; ++l)
                handles.push_back(lm::PageHandle{
                    .gpu_idx = 0, .page_idx = j,
                    .gpu_ptr = static_cast<char*>(fx.pool)
                             + static_cast<int64_t>(j) * kStrideBlock,
                    .pool = lm::Pool::kMain});
        for (int j = 0; j < num_logical; ++j) host_bt.push_back(j);

        ld::KvTieringManager::Options o;
        o.dcp_size = 1;
        o.gpus = {ref0()};
        o.device_backends = {be};
        o.stream_manager = nullptr;   // tests synchronize explicitly
        o.numa_manager = nullptr;     // plain pinned arena
        o.free_page = [this](uint64_t, int layer, int logical,
                             const lm::PageHandle&) {
            freed.emplace_back(layer, logical);
        };
        // Re-promotion alloc seam (mirrors the dispatcher's): hand out spare
        // pool pages, update the handle table + block table (only layer 0 is
        // demoted/materialized in these tests — the shared bt tracks it).
        o.alloc_page = [this](uint64_t, int layer, int logical)
                -> std::optional<lm::PageHandle> {
            if (next_free_page >= fx.pool_pages) return std::nullopt;
            const int p = next_free_page++;
            lm::PageHandle h{
                .gpu_idx = 0, .page_idx = p,
                .gpu_ptr = static_cast<char*>(fx.pool)
                         + static_cast<int64_t>(p) * kStrideBlock,
                .pool = lm::Pool::kMain};
            handles[static_cast<size_t>(logical) * kLayers + layer] = h;
            if (layer == 0) host_bt[static_cast<size_t>(logical)] = p;
            alloced.push_back({layer, logical, p});
            return h;
        };
        o.kv_main_bases = {fx.pool};
        o.stride_block = kStrideBlock;
        o.stride_row = kStrideRow;
        o.page_size = kPageSize;
        o.kv_layers = kLayers;
        o.index_topk = kTopk;
        o.hot_buffer_slots = hot_slots;
        o.host_to_device_ratio = host_to_device_ratio;
        o.indexer_full_layers = std::move(full_mask);
        o.kv_bearing_layers = std::move(bearing_mask);  // GF3.9
        o.cohort_rows_max = cohort_rows;  // TD-KVT-ADMISSION-UPFRONT
        o.union_rows_max = union_rows;    // TD-KVT-COHORT-BATCHED-MATERIALIZE
        mgr = std::make_unique<ld::KvTieringManager>(std::move(o));

        const int cr = std::max(1, cohort_rows);
        dev_indices = be->device_alloc(
            static_cast<size_t>(cr) * kTopk * sizeof(int));
        dev_len = be->device_alloc(static_cast<size_t>(cr) * sizeof(int));
    }
    ~ManagerFixture() {
        mgr.reset();  // drains before the pool dies
        fx.be->set_device();
        fx.be->device_free(dev_indices);
        fx.be->device_free(dev_len);
    }

    const int* bt() const { return host_bt.data(); }

    /// D2H read of physical pool page `phys` (settled).
    std::vector<char> read_page(int phys) {
        std::vector<char> out(kStrideBlock);
        fx.be->set_device();
        fx.be->synchronize_device();
        fx.be->memcpy_d2h_async(out.data(),
                                static_cast<char*>(fx.pool)
                                    + static_cast<int64_t>(phys) * kStrideBlock,
                                out.size(), fx.stream);
        fx.be->synchronize_device();
        return out;
    }

    void upload_selection(const std::vector<int>& sel) {
        std::vector<int> padded(kTopk, -1);
        for (size_t i = 0; i < sel.size(); ++i) padded[i] = sel[i];
        const int n = static_cast<int>(sel.size());
        fx.be->memcpy_h2d(dev_indices, padded.data(), kTopk * sizeof(int));
        fx.be->memcpy_h2d(dev_len, &n, sizeof(int));
    }

    /// TD-KVT-ADMISSION-UPFRONT: upload a chunk cohort's per-row selections
    /// (row b at dev_indices + b*kTopk, length at dev_len + b).
    void upload_cohort(const std::vector<std::vector<int>>& rows_sel) {
        const int R = static_cast<int>(rows_sel.size());
        std::vector<int> padded(static_cast<size_t>(R) * kTopk, -1);
        std::vector<int> lens(static_cast<size_t>(R), 0);
        for (int b = 0; b < R; ++b) {
            for (size_t i = 0; i < rows_sel[b].size(); ++i)
                padded[static_cast<size_t>(b) * kTopk + i] = rows_sel[b][i];
            lens[static_cast<size_t>(b)] =
                static_cast<int>(rows_sel[b].size());
        }
        fx.be->memcpy_h2d(dev_indices, padded.data(),
                          padded.size() * sizeof(int));
        fx.be->memcpy_h2d(dev_len, lens.data(), lens.size() * sizeof(int));
    }

    /// Demote every eligible page of `layer` at position `pos` and wait.
    void demote(int layer, uint32_t pos, uint64_t seq = 1) {
        fx.be->synchronize_device();  // no stream_manager ordering — quiesce
        const int* bts[1] = {bt()};
        ASSERT_TRUE(mgr->begin_layer(layer, seq, pos, bts));
        mgr->after_attention(layer, seq, pos, handles.data() + layer,
                             static_cast<int>(handles.size()) / kLayers,
                             kLayers);
        // Drain: the D2H copies run on the manager's owned stream.
        fx.be->synchronize_device();
        mgr->poll_demotions();
    }

    /// Byte-compare the materialized scratch rows against fill_row ground
    /// truth (selection order).
    void check_scratch(const layerstorm::parallelism::TieredKvView& tv,
                       const std::vector<int>& sel) {
        ASSERT_EQ(tv.seq_len_kv, static_cast<int>(sel.size()));
        ASSERT_NE(tv.kv_cache, nullptr);
        ASSERT_NE(tv.sparse_indices, nullptr);
        ASSERT_NE(tv.block_tables, nullptr);
        fx.be->synchronize_device();

        const size_t bytes = static_cast<size_t>(tv.max_blocks_per_seq)
                           * kStrideBlock;
        std::vector<char> out(bytes);
        fx.be->memcpy_d2h_async(out.data(), tv.kv_cache, bytes, fx.stream);
        fx.be->synchronize_device();
        std::vector<char> want(kStrideRow);
        for (size_t i = 0; i < sel.size(); ++i) {
            fill_row(want.data(), 0, sel[i]);
            ASSERT_EQ(std::memcmp(
                out.data() + (i / kPageSize) * kStrideBlock
                    + (i % kPageSize) * kStrideRow,
                want.data(), kStrideRow), 0)
                << "materialized row " << i << " (pos " << sel[i]
                << ") != ground truth";
        }
    }

    /// Run materialize on layer `layer` and byte-compare the scratch rows
    /// against fill_row ground truth.
    void materialize_and_check(int layer, const std::vector<int>& sel,
                               uint64_t seq = 1) {
        upload_selection(sel);
        const int* bts[1] = {bt()};
        ASSERT_TRUE(mgr->begin_layer(layer, seq,
                                     /*pos=*/31, bts));
        layerstorm::parallelism::TieredKvView tv{};
        const bool tiered = mgr->materialize(
            0, layer, static_cast<const int*>(dev_indices),
            static_cast<const int*>(dev_len),
            /*batch=*/1, fx.stream, &tv);
        ASSERT_TRUE(tiered) << "expected tiered path (layer has cold pages)";
        check_scratch(tv, sel);
    }
};

}  // namespace

TEST(KvTieringManager, MaterializeHotAndColdMatchesGroundTruth) {
    REQUIRES_GPU();
    // 8 logical pages × 4 tokens = positions 0..31. Retention =
    // hot_buffer_slots = 8 tokens → at pos 31, pages 0..5 (tokens 0..23)
    // are behind the window; frontier page 7 and page 6 stay hot.
    ManagerFixture m(/*hot_slots=*/8);
    m.demote(/*layer=*/0, /*pos=*/31);
    ASSERT_TRUE(m.mgr->has_demotions());
    ASSERT_TRUE(m.mgr->layer_has_cold(1, 0));
    // Pages 0..5 of layer 0 demoted, in order.
    ASSERT_EQ(m.freed.size(), 6u);
    for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(m.freed[j].first, 0);
        EXPECT_EQ(m.freed[j].second, j);
    }
    EXPECT_FALSE(m.mgr->layer_has_cold(1, 1));  // other layer untouched

    // Clobber the demoted pool pages: the ONLY correct source is now the
    // manager's pinned cold copy.
    std::vector<char> junk(6 * kStrideBlock, '\xAA');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());

    // Mixed selection: cold (0, 5, 9, 23), hot pool (24, 28, 31).
    m.materialize_and_check(0, {23, 0, 31, 9, 5, 24, 28});
    const auto& s1 = m.mgr->stats();
    EXPECT_EQ(s1.cold_misses, 4u);
    EXPECT_EQ(s1.pool_hits, 3u);
    EXPECT_EQ(s1.cache_hits, 0u);
    EXPECT_EQ(s1.h2d_bursts, 1u);  // ONE burst H2D for all cold rows

    // Same selection again: every cold row must now be a row-cache hit.
    m.materialize_and_check(0, {23, 0, 31, 9, 5, 24, 28});
    const auto& s2 = m.mgr->stats();
    EXPECT_EQ(s2.cold_misses, 4u) << "second pass must not refetch";
    EXPECT_EQ(s2.cache_hits, 4u);
    EXPECT_EQ(s2.pool_hits, 6u);
    EXPECT_EQ(m.mgr->cache_entries(0, 0), 4);
}

TEST(KvTieringManager, NonKvBearingLayersAreGatedOut) {
    REQUIRES_GPU();
    // GF3.9 (glm5_next hybrid): the attention-type mask gates tiering to
    // KV-bearing layers only. A KDA linear layer (bearing_mask[l] == 0)
    // must be refused by begin_layer, ignored by after_attention, and
    // never counted as a shared successor (its page list is
    // empty/sentinel on the real model). Layer 1 (bearing) behaves
    // exactly like the unmasked manager.
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/10,
                     /*full_mask=*/{}, /*host_to_device_ratio=*/8.0,
                     /*spare_pages=*/0, /*cohort_rows=*/0,
                     /*union_rows=*/0,
                     /*bearing_mask=*/{0, 1});
    const int* bts[1] = {m.bt()};
    const int NL = static_cast<int>(m.handles.size()) / kLayers;
    m.fx.be->synchronize_device();

    // Layer 0 = "KDA": refused outright; after_attention is a no-op.
    EXPECT_FALSE(m.mgr->begin_layer(0, 1, /*pos=*/0, bts));
    m.mgr->after_attention(0, 1, /*pos=*/31, m.handles.data() + 0, NL,
                           kLayers);
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    EXPECT_TRUE(m.freed.empty())
        << "a non-bearing layer must never produce demotions";

    // Layer 1 = sparse MLA: the normal tier flow engages.
    for (uint32_t pos = 0; pos <= 31; ++pos) {
        ASSERT_TRUE(m.mgr->begin_layer(1, 1, pos, bts));
        m.mgr->after_attention(1, 1, pos, m.handles.data() + 1, NL,
                               kLayers);
    }
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    EXPECT_FALSE(m.freed.empty())
        << "the bearing layer must tier normally under the mask";
    for (const auto& [layer, logical] : m.freed)
        EXPECT_EQ(layer, 1) << "only the bearing layer may demote";
}

TEST(KvTieringManager, PrefillChunkFlowDemotesBehindFrontierByteExact) {
    REQUIRES_GPU();
    // TD-KVT-PREFILL: simulate B==1 SPARSE chunked prefill — one
    // begin_layer/after_attention pair per prompt position, exactly the
    // dispatcher's tier-step flow for a blessed sparse prefill chunk (the
    // manager is step-shape agnostic; this pins that contract).  Pages must
    // demote DURING prefill once they fall fully behind the retention
    // window, a later chunk must keep tiering INTO the now-demoted sequence
    // (multi-turn append, no re-promotion), and its selection-only
    // materialization must be byte-exact with the demoted VRAM pages
    // clobbered (the pinned cold copy is the source — never the full
    // prefix through the real block tables).
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/10);
    const int* bts[1] = {m.bt()};
    const int NL = static_cast<int>(m.handles.size()) / kLayers;
    m.fx.be->synchronize_device();

    // "Prefill" positions 0..31 (pages 0..7); retention = 8 tokens.
    for (uint32_t pos = 0; pos <= 31; ++pos) {
        ASSERT_TRUE(m.mgr->begin_layer(0, 1, pos, bts));
        m.mgr->after_attention(0, 1, pos, m.handles.data() + 0, NL, kLayers);
        if (pos == 15) {
            // Mid-prefill: demote_end = 16 − 8 = 8 → pages 0 (tokens 0..3)
            // and 1 (4..7) must have demoted DURING prefill.
            m.fx.be->synchronize_device();
            m.mgr->poll_demotions();
            ASSERT_EQ(m.freed.size(), 2u)
                << "behind-frontier demotion must fire mid-prefill";
            EXPECT_TRUE(m.mgr->seq_has_demotions(1));
        }
    }
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    // At pos 31: demote_end = 24 → pages 0..5 cold, in order.
    ASSERT_EQ(m.freed.size(), 6u);
    for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(m.freed[j].first, 0);
        EXPECT_EQ(m.freed[j].second, j);
    }
    ASSERT_TRUE(m.mgr->layer_has_cold(1, 0));

    // Clobber the demoted pool pages 0..5: the ONLY correct source for
    // their rows is now the pinned cold pool.
    std::vector<char> junk(6 * kStrideBlock, '\xCC');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());

    // Multi-turn APPEND into the demoted sequence: chunk positions 32..35
    // (page 8) keep tiering — begin_layer must accept them and demotion
    // must continue (page 6 falls behind at pos 35: demote_end = 28).
    for (uint32_t pos = 32; pos <= 35; ++pos) {
        ASSERT_TRUE(m.mgr->begin_layer(0, 1, pos, bts))
            << "prefill append into a demoted sequence must stay tierable";
        m.mgr->after_attention(0, 1, pos, m.handles.data() + 0, NL, kLayers);
    }
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    ASSERT_EQ(m.freed.size(), 7u);
    EXPECT_EQ(m.freed[6].second, 6);

    // "Chunk row 36's" causal top-k: cold rows (0, 9, 23, 26) + hot pool
    // rows (28, 33, 35) — selection-only materialization, byte-exact.
    const std::vector<int> sel{0, 28, 9, 33, 23, 26, 35};
    m.upload_selection(sel);
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 36, bts));
    layerstorm::parallelism::TieredKvView tv{};
    ASSERT_TRUE(m.mgr->materialize(
        0, 0, static_cast<const int*>(m.dev_indices),
        static_cast<const int*>(m.dev_len), /*batch=*/1, m.fx.stream, &tv));
    m.check_scratch(tv, sel);
    const auto& s = m.mgr->stats();
    EXPECT_EQ(s.cold_misses, 4u);
    EXPECT_EQ(s.pool_hits, 3u);
    EXPECT_EQ(s.h2d_bursts, 1u);
    EXPECT_EQ(s.demoted_pages, 7u);
}

TEST(KvTieringManager, HibernateLayerDemotesAllButFrontierByteExact) {
    REQUIRES_GPU();
    // R3 holder hibernation (TD-PREFIX-POOL-PRESSURE-EVICTS-THE-PRIZE): a
    // FROZEN prefix holder never steps, so window demotion never reaches
    // it — measured on the GLM champion, each deep holder pinned ~5,100
    // pages/rank (its hot retention window at fork time) for its whole
    // life, and pool pressure evicted the cache as fast as it filled.
    // hibernate_layer must demote EVERY hot page EXCEPT the
    // append-frontier logical page (INV-KVT-4's frontier rule), regardless
    // of the retention window, with byte-exact cold copies (INV-KVT-1) —
    // and a second call must be a no-op (pages already cold).
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/10);
    const int NL = static_cast<int>(m.handles.size()) / kLayers;
    m.fx.be->synchronize_device();

    // No step ever ran on this sequence (a holder forked from an
    // undemoted parent has no tiering state) — hibernate creates it.
    const int n = m.mgr->hibernate_layer(0, /*seq=*/1,
                                         m.handles.data() + 0, NL, kLayers);
    EXPECT_EQ(n, NL - 1) << "all but the frontier page must demote";
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    ASSERT_EQ(m.freed.size(), static_cast<size_t>(NL - 1));
    for (int j = 0; j < NL - 1; ++j) {
        EXPECT_EQ(m.freed[j].first, 0);
        EXPECT_EQ(m.freed[j].second, j) << "demotion order is positional";
    }
    EXPECT_TRUE(m.mgr->seq_has_demotions(1));

    // Frontier page stays HOT (no cold copy); every demoted page's cold
    // copy is byte-exact with the initial VRAM contents.
    EXPECT_EQ(m.mgr->cold_page_host_ptr(1, 0, NL - 1), nullptr);
    for (int j = 0; j < NL - 1; ++j) {
        const void* c = m.mgr->cold_page_host_ptr(1, 0, j);
        ASSERT_NE(c, nullptr) << "page " << j << " must be COLD";
        EXPECT_EQ(std::memcmp(c,
                              m.fx.host_pool.data()
                                  + static_cast<int64_t>(j) * kStrideBlock,
                              kStrideBlock),
                  0)
            << "cold copy of page " << j << " must be byte-exact";
    }

    // Idempotent: nothing left to demote.
    EXPECT_EQ(m.mgr->hibernate_layer(0, 1, m.handles.data() + 0, NL,
                                     kLayers),
              0);

    // Explicit frontier (the dispatcher passes kv_len/page_size): a
    // holder whose parent over-allocated must keep pages AT/AFTER its
    // coverage-end page hot — hibernate at frontier 7 on a fresh seq
    // demotes exactly pages 0..6.
    EXPECT_EQ(m.mgr->hibernate_layer(0, /*seq=*/9, m.handles.data() + 0,
                                     NL, kLayers, /*frontier_logical=*/7),
              7);
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    EXPECT_EQ(m.mgr->cold_page_host_ptr(9, 0, 6) != nullptr, true);
    EXPECT_EQ(m.mgr->cold_page_host_ptr(9, 0, 7), nullptr);
    EXPECT_EQ(m.mgr->cold_page_host_ptr(9, 0, 8), nullptr);
    m.mgr->on_seq_free(9);

    // A fork FROM the hibernated holder shares the cold slots (INV-KVT-15)
    // and the child may step from the holder frontier: demoted_frontier is
    // (NL-1)*page_size, so begin_layer at that position is legal (the
    // fork-child delta prefill's first write lands on the hot frontier).
    m.mgr->on_seq_fork(1, 2);
    const int* bts[1] = {m.bt()};
    EXPECT_TRUE(m.mgr->begin_layer(0, 2, (NL - 1) * kPageSize, bts));
    m.mgr->on_seq_free(2);
    m.mgr->on_seq_free(1);
}

TEST(KvTieringManager, CohortPerRowMaterializeByteExactChunkBoundaryDemote) {
    REQUIRES_GPU();
    // TD-KVT-ADMISSION-UPFRONT: a blessed sparse prefill CHUNK cohort —
    // ONE batched selection readback per (rank, layer), per-row fake views
    // byte-exact against ground truth with the demoted VRAM pages
    // clobbered, chunk-boundary demotion behind the LAST row's position,
    // and per-LAYER cohort write legality (superchunk layer-wise sweep:
    // replaying an earlier sub-chunk at a later layer is legal while
    // writing over THIS layer's demoted territory throws, INV-KVT-2).
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/10, {}, 8.0,
                     /*spare_pages=*/0, /*cohort_rows=*/4);
    // Decode-style history to pos 31: pages 0..5 of layer 0 demote.
    m.demote(/*layer=*/0, /*pos=*/31);
    ASSERT_EQ(m.freed.size(), 6u);
    std::vector<char> junk(6 * kStrideBlock, '\xAA');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());

    // Chunk cohort [32, 36): per-layer legality passes (no non-hot page of
    // layer 0 at/after logical 8).
    const int* bts[1] = {m.bt()};
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, /*pos=*/32, bts, /*rows=*/4));
    const std::vector<std::vector<int>> sel = {
        {23, 0, 31},      // cold 23, 0 + hot 31
        {9, 33},          // cold 9 + hot 33
        {},               // empty selection → untiered row
        {5, 24, 0, 35},   // cold 5 + hot 24, 35 + repeat 0 (cache hit)
    };
    m.upload_cohort(sel);
    for (int b = 0; b < 4; ++b) {
        layerstorm::parallelism::TieredKvView tv{};
        const bool t = m.mgr->materialize_row(
            0, /*layer=*/0, b, /*rows=*/4,
            static_cast<const int*>(m.dev_indices),
            static_cast<const int*>(m.dev_len),
            /*selection_fresh=*/true, m.fx.stream, &tv);
        if (sel[static_cast<size_t>(b)].empty()) {
            EXPECT_FALSE(t) << "empty row " << b;
            continue;
        }
        ASSERT_TRUE(t) << "row " << b;
        // Per-row check BEFORE the next row's gather reuses the scratch.
        m.check_scratch(tv, sel[static_cast<size_t>(b)]);
    }
    const auto& s = m.mgr->stats();
    EXPECT_EQ(s.cohort_readbacks, 1u)
        << "ONE batched selection readback per (rank, layer)";
    EXPECT_EQ(s.cohort_rows_tiered, 3u);
    EXPECT_GE(s.cache_hits, 1u) << "repeated cold pos must hit the row cache";

    // Chunk-boundary demotion: after_attention at the LAST row's position.
    const int NL = static_cast<int>(m.handles.size()) / kLayers;
    m.fx.be->synchronize_device();
    m.mgr->after_attention(0, 1, /*token_pos=*/35, m.handles.data() + 0,
                           NL, kLayers);
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    // demote_end = 36 − 8 = 28 → page 6 (tokens 24..27) newly demoted.
    ASSERT_EQ(m.freed.size(), 7u);
    EXPECT_EQ(m.freed.back().second, 6);

    // Cohort write over layer 0's demoted territory: fail-loud.
    EXPECT_THROW(m.mgr->begin_layer(0, 1, /*pos=*/16, bts, /*rows=*/4),
                 std::runtime_error);
    // Same replay on layer 1 (no demotions): legal — the superchunk
    // layer-wise sweep revisits earlier sub-chunks at later layers.
    EXPECT_TRUE(m.mgr->begin_layer(1, 1, /*pos=*/16, bts, /*rows=*/4));
    // Rows above the cohort staging cap are rejected (not a tier step).
    EXPECT_FALSE(m.mgr->begin_layer(1, 1, /*pos=*/16, bts, /*rows=*/5));
}

TEST(KvTieringManager, CohortBatchedUnionMaterializeByteExactRewrite) {
    REQUIRES_GPU();
    // TD-KVT-COHORT-BATCHED-MATERIALIZE: the batched union consumer — ONE
    // gather of the ascending unique union of the cohort's selections
    // (byte-exact against ground truth with the demoted VRAM pages
    // clobbered), per-row indices rewritten to union slots ORDER-PRESERVING,
    // per-row seqlens all = the union extent, and the per-row arm still
    // byte-exact afterwards (fallback coexistence).
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/10, {}, 8.0,
                     /*spare_pages=*/0, /*cohort_rows=*/4, /*union_rows=*/64);
    m.demote(/*layer=*/0, /*pos=*/31);  // pages 0..5 of layer 0 demote
    ASSERT_EQ(m.freed.size(), 6u);
    std::vector<char> junk(6 * kStrideBlock, '\xBB');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());

    const int* bts[1] = {m.bt()};
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, /*pos=*/32, bts, /*rows=*/4));
    const std::vector<std::vector<int>> sel = {
        {23, 0, 31},      // cold 23, 0 + hot 31
        {9, 33},          // cold 9 + hot 33
        {},               // empty selection (rewritten length 0)
        {5, 24, 0, 35},   // cold 5 + hot 24, 35 + repeat 0 (union dedup)
    };
    m.upload_cohort(sel);
    layerstorm::parallelism::TieredKvView tv{};
    ASSERT_TRUE(m.mgr->materialize_cohort(
        0, /*layer=*/0, /*rows=*/4,
        static_cast<const int*>(m.dev_indices),
        static_cast<const int*>(m.dev_len),
        /*selection_fresh=*/true, m.fx.stream, &tv));
    // Union = ascending unique positions across all rows.
    const std::vector<int> uni{0, 5, 9, 23, 24, 31, 33, 35};
    m.check_scratch(tv, uni);

    // Rewritten per-row indices (union slots, order-preserving) + seqlens.
    m.fx.be->synchronize_device();
    std::vector<int> got_idx(4 * kTopk);
    std::vector<int> got_len(4);
    m.fx.be->memcpy_d2h_async(got_idx.data(), tv.sparse_indices,
                              got_idx.size() * sizeof(int), m.fx.stream);
    m.fx.be->memcpy_d2h_async(got_len.data(), tv.seqlens_k,
                              got_len.size() * sizeof(int), m.fx.stream);
    m.fx.be->synchronize_device();
    const std::vector<std::vector<int>> want_rw = {
        {3, 0, 5}, {2, 6}, {}, {1, 4, 0, 7}};
    for (int b = 0; b < 4; ++b) {
        EXPECT_EQ(got_len[static_cast<size_t>(b)],
                  static_cast<int>(uni.size()))
            << "row " << b << " seqlen must be the union extent";
        for (size_t i = 0; i < want_rw[static_cast<size_t>(b)].size(); ++i) {
            EXPECT_EQ(got_idx[static_cast<size_t>(b) * kTopk + i],
                      want_rw[static_cast<size_t>(b)][i])
                << "row " << b << " rewritten index " << i;
        }
    }
    const auto& s = m.mgr->stats();
    EXPECT_EQ(s.cohort_unions, 1u);
    EXPECT_EQ(s.cohort_union_rows, uni.size());
    EXPECT_EQ(s.cohort_union_rewrites, 1u);
    EXPECT_EQ(s.cohort_readbacks, 1u)
        << "ONE batched selection readback per (rank, layer)";
    EXPECT_EQ(s.cache_hits, 0u)
        << "union gathers must not cache-insert (nothing to hit yet)";

    // Same step, per-row arm still byte-exact after the union (fallback
    // coexistence: the union and per-row staging sets are independent).
    layerstorm::parallelism::TieredKvView rtv{};
    ASSERT_TRUE(m.mgr->materialize_row(
        0, /*layer=*/0, /*row=*/0, /*rows=*/4,
        static_cast<const int*>(m.dev_indices),
        static_cast<const int*>(m.dev_len),
        /*selection_fresh=*/false, m.fx.stream, &rtv));
    m.check_scratch(rtv, sel[0]);
}

TEST(KvTieringManager, CohortUnionCapacityAndRowwiseFallback) {
    REQUIRES_GPU();
    // Union larger than union_rows_max → materialize_cohort returns false
    // (fail-safe per-row fallback), never throws.
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/10, {}, 8.0,
                     /*spare_pages=*/0, /*cohort_rows=*/4, /*union_rows=*/4);
    m.demote(/*layer=*/0, /*pos=*/31);
    const int* bts[1] = {m.bt()};
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, /*pos=*/32, bts, /*rows=*/4));
    const std::vector<std::vector<int>> sel = {
        {23, 0, 31}, {9, 33}, {}, {5, 24, 0, 35}};  // union 8 > cap 4
    m.upload_cohort(sel);
    layerstorm::parallelism::TieredKvView tv{};
    EXPECT_FALSE(m.mgr->materialize_cohort(
        0, 0, 4, static_cast<const int*>(m.dev_indices),
        static_cast<const int*>(m.dev_len), true, m.fx.stream, &tv));
    // Per-row arm consumes the same host selection (identity reuse).
    layerstorm::parallelism::TieredKvView rtv{};
    ASSERT_TRUE(m.mgr->materialize_row(
        0, 0, 0, 4, static_cast<const int*>(m.dev_indices),
        static_cast<const int*>(m.dev_len), false, m.fx.stream, &rtv));
    m.check_scratch(rtv, sel[0]);
    EXPECT_EQ(m.mgr->stats().cohort_unions, 0u);

    // LS_KVT_COHORT_ROWWISE=1 kill-switch: the arm is disabled at init.
    setenv("LS_KVT_COHORT_ROWWISE", "1", 1);
    {
        ManagerFixture m2(8, 10, {}, 8.0, 0, /*cohort_rows=*/4,
                          /*union_rows=*/64);
        m2.demote(0, 31);
        const int* bts2[1] = {m2.bt()};
        ASSERT_TRUE(m2.mgr->begin_layer(0, 1, 32, bts2, /*rows=*/4));
        m2.upload_cohort(sel);
        layerstorm::parallelism::TieredKvView tv2{};
        EXPECT_FALSE(m2.mgr->materialize_cohort(
            0, 0, 4, static_cast<const int*>(m2.dev_indices),
            static_cast<const int*>(m2.dev_len), true, m2.fx.stream, &tv2));
    }
    unsetenv("LS_KVT_COHORT_ROWWISE");
}

TEST(KvTieringManager, LruEvictionUnderTinyHotBufferStaysCorrect) {
    REQUIRES_GPU();
    // hot_slots = 4: retention 4 tokens → at pos 31 pages 0..6 demote
    // (frontier page 7 stays); row cache holds only 4 entries per layer.
    ManagerFixture m(/*hot_slots=*/4);
    m.demote(0, 31);
    ASSERT_EQ(m.freed.size(), 7u);

    std::vector<char> junk(7 * kStrideBlock, '\x55');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());

    // 6 cold rows > 4 slots: correctness must hold; evictions occur on the
    // second, disjoint selection.
    m.materialize_and_check(0, {0, 4, 8, 12, 16, 20});
    m.materialize_and_check(0, {1, 5, 9, 13, 17, 21});
    const auto& s = m.mgr->stats();
    EXPECT_GT(s.cache_evictions, 0u);
    EXPECT_EQ(s.cold_misses, 12u);  // disjoint sets: all cold both times

    // Third pass repeats the second: the 4 cached entries hit, 2 refetch.
    m.materialize_and_check(0, {1, 5, 9, 13, 17, 21});
    const auto& s3 = m.mgr->stats();
    EXPECT_EQ(s3.cache_hits, 4u);
    EXPECT_EQ(s3.cold_misses, 14u);
}

TEST(KvTieringManager, DenseLayerThrowsOnColdAndSticksOtherwise) {
    REQUIRES_GPU();
    ManagerFixture m(/*hot_slots=*/8);
    const int* bts[1] = {m.bt()};
    ASSERT_TRUE(m.mgr->begin_layer(1, 1, 31, bts));
    // Layer 1 has no cold pages: on_dense_layer is a no-op that marks it
    // sticky-dense — subsequent after_attention must NOT demote layer 1.
    EXPECT_NO_THROW(m.mgr->on_dense_layer(1));
    m.fx.be->synchronize_device();
    m.mgr->after_attention(1, 1, 31, m.handles.data() + 1,
                           static_cast<int>(m.handles.size()) / kLayers,
                           kLayers);
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    EXPECT_TRUE(m.freed.empty()) << "sticky-dense layer must never demote";

    // Layer 0 demotes; a later dense fallback on it must throw (INV-KVT-2).
    m.demote(0, 31);
    ASSERT_TRUE(m.mgr->layer_has_cold(1, 0));
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 31, bts));
    EXPECT_THROW(m.mgr->on_dense_layer(0), std::runtime_error);
}

TEST(KvTieringManager, SecondSequenceEngagesWithIndependentState) {
    REQUIRES_GPU();
    // TD-KVT-BATCH: tiering is per-sequence — a second sequence engages
    // (begin_layer true) with its OWN state: no demotions until it demotes,
    // and its low positions never trip seq 1's demoted frontier.
    ManagerFixture m(/*hot_slots=*/8);
    m.demote(0, 31);  // seq 1 demotes pages 0..5 (frontier 24)
    const int* bts[1] = {m.bt()};
    EXPECT_TRUE(m.mgr->begin_layer(0, /*seq=*/2, /*pos=*/10, bts))
        << "second sequence must engage tiering (TD-KVT-BATCH)";
    EXPECT_FALSE(m.mgr->seq_has_demotions(2));
    EXPECT_TRUE(m.mgr->seq_has_demotions(1));
    EXPECT_FALSE(m.mgr->layer_has_cold(2, 0));
    // Position 10 < seq 1's frontier 24 — but it is seq 2's OWN (fresh)
    // position history: no rewind, no throw.
    EXPECT_TRUE(m.mgr->begin_layer(0, 2, 11, bts));
}

TEST(KvTieringManager, PerSeqHotBudgetFairShareEviction) {
    REQUIRES_GPU();
    // TD-KVT-BATCH hot fair-share: the shared row-cache slab is budgeted
    // per sequence — a second sequence's inserts evict the OVER-share
    // sequence's rows (hot_slots / n_seqs each), never starve, and every
    // materialization stays byte-exact.
    ManagerFixture m(/*hot_slots=*/4);  // retention 4 → pages 0..6 demote
    m.demote(0, 31, /*seq=*/1);
    m.demote(0, 31, /*seq=*/2);
    ASSERT_EQ(m.freed.size(), 14u);  // 7 pages per sequence, freed once each

    std::vector<char> junk(7 * kStrideBlock, '\x44');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());

    // Seq 1 fills the whole 4-slot layer-0 cache.
    m.materialize_and_check(0, {0, 4, 8, 12}, /*seq=*/1);
    EXPECT_EQ(m.mgr->cache_entries_seq(0, 0, 1), 4);

    // Seq 2 inserts 2 cold rows: fair share = 4/2 = 2 → seq 1 (over share)
    // is evicted down to its share; seq 2 gets its 2 slots.  Byte-exact.
    m.materialize_and_check(0, {1, 5}, /*seq=*/2);
    EXPECT_EQ(m.mgr->cache_entries_seq(0, 0, 1), 2);
    EXPECT_EQ(m.mgr->cache_entries_seq(0, 0, 2), 2);
    EXPECT_EQ(m.mgr->cache_entries(0, 0), 4);
    EXPECT_GT(m.mgr->stats().cache_evictions, 0u);

    // Both sequences' surviving entries still serve byte-exact hits.
    m.materialize_and_check(0, {8, 12}, /*seq=*/1);
    m.materialize_and_check(0, {1, 5}, /*seq=*/2);
    EXPECT_EQ(m.mgr->stats().cache_hits, 4u);
}

TEST(KvTieringManager, SeqFreeReturnsColdSlotsNoCrossSeqLeakage) {
    REQUIRES_GPU();
    // TD-KVT-BATCH cold-slot isolation: freeing one sequence returns
    // exactly ITS cold slots; the surviving sequence's cold copies stay
    // intact and byte-exact (no cross-seq slot reuse/leakage).
    ManagerFixture m(/*hot_slots=*/8);  // cold pool 32 pages
    ASSERT_EQ(m.mgr->cold_pool_capacity_pages(), 32);
    m.demote(0, 31, /*seq=*/1);  // pages 0..5
    m.demote(0, 31, /*seq=*/2);  // pages 0..5 (per-seq view of same layer)
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 12);
    EXPECT_EQ(m.mgr->seq_cold_used_pages(1, 0), 6);
    EXPECT_EQ(m.mgr->seq_cold_used_pages(2, 0), 6);

    m.mgr->on_seq_free(1);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 6)
        << "seq 1's cold slots must return in full";
    EXPECT_FALSE(m.mgr->seq_has_demotions(1));
    EXPECT_TRUE(m.mgr->seq_has_demotions(2));
    EXPECT_TRUE(m.mgr->has_demotions());
    EXPECT_TRUE(m.mgr->layer_has_cold(2, 0));

    // Clobber the freed VRAM pages: seq 2's ONLY correct source is its own
    // (untouched) cold slots.
    std::vector<char> junk(6 * kStrideBlock, '\x99');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());
    m.materialize_and_check(0, {0, 5, 9, 23, 24, 28}, /*seq=*/2);
    EXPECT_EQ(m.mgr->stats().cold_misses, 4u);

    m.mgr->on_seq_free(2);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);
    EXPECT_FALSE(m.mgr->has_demotions());
}

TEST(KvTieringManager, ColdPoolFairShareCapAcrossSequences) {
    REQUIRES_GPU();
    // TD-KVT-BATCH cold fair-share: with N demoting sequences a sequence's
    // per-rank cold usage is capped at capacity / N — over-cap demotions
    // are SKIPPED (pages stay hot, fail-safe), never fail-corrupt.
    // ratio 4.0 → per-layer 8 pages × 2 layers = 16-slot pool.
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/8, /*full_mask=*/{},
                     /*host_to_device_ratio=*/4.0);
    ASSERT_EQ(m.mgr->cold_pool_capacity_pages(), 16);
    m.demote(0, 23, /*seq=*/1);  // pages 0..3 (retention 8)
    m.demote(0, 23, /*seq=*/2);  // pages 0..3
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 8);
    EXPECT_EQ(m.mgr->stats().budget_skips, 0u);

    // Seq 3 at pos 31 has 6 demotion candidates but cap = 16/3 = 5: it
    // demotes pages 0..4 and SKIPS page 5 (stays hot) even though 8 slots
    // are still free.
    m.demote(0, 31, /*seq=*/3);
    EXPECT_EQ(m.mgr->seq_cold_used_pages(3, 0), 5);
    EXPECT_EQ(m.mgr->stats().budget_skips, 1u);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 13);
    ASSERT_EQ(m.freed.size(), 13u);  // 4 + 4 + 5, each page freed once

    // Mixed cold/hot materialization for seq 3 stays byte-exact: pages
    // 0..4 come from its cold slots (clobbered in VRAM), the budget-skipped
    // page 5 and the frontier stay hot in the pool.
    std::vector<char> junk(5 * kStrideBlock, '\x66');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());
    m.materialize_and_check(0, {0, 8, 20, 24}, /*seq=*/3);
    EXPECT_EQ(m.mgr->stats().cold_misses, 2u);   // positions 0, 8
    EXPECT_EQ(m.mgr->stats().pool_hits, 2u);     // positions 20 (page 5), 24
}

TEST(KvTieringManager, HibernateBypassesColdFairShareCap) {
    REQUIRES_GPU();
    // R3: the per-seq cold fair-share cap protects the pool from one LIVE
    // sequence, but it counts fork-family holders as independent sequences
    // — a chained holder's inherited slots already exceed capacity/nseq,
    // so the cap would veto exactly the deep holders hibernation exists
    // for.  hibernate_layer must therefore ignore the cap (bounded by pool
    // capacity + holder eviction instead).
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/8, /*full_mask=*/{},
                     /*host_to_device_ratio=*/4.0);
    ASSERT_EQ(m.mgr->cold_pool_capacity_pages(), 16);
    m.demote(0, 23, /*seq=*/1);   // live seq: 4 cold pages
    m.demote(0, 23, /*seq=*/2);   // live seq: 4 cold pages
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 8);

    // Hibernating holder seq 3: 7 eligible pages, fair-share cap would be
    // 16/3 = 5 — hibernation must take all 7 anyway (8 slots free).
    const int NL = static_cast<int>(m.handles.size()) / kLayers;
    const int n = m.mgr->hibernate_layer(0, /*seq=*/3,
                                         m.handles.data() + 0, NL, kLayers);
    EXPECT_EQ(n, NL - 1);
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    EXPECT_EQ(m.mgr->seq_cold_used_pages(3, 0), NL - 1);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 8 + NL - 1);
    // A hibernated holder is EXCLUDED from the fair-share census.  Free
    // seq 1 (4 slots back → 5 free); live demoters are then {2, 4}: seq
    // 4's cap is 16/2-with-itself = 8, so its 6th candidate is stopped by
    // POOL exhaustion (5 free), never by a holder-inflated budget skip
    // (a census counting hibernated seq 3 would cap at 16/3 = 5 and skip
    // the 6th candidate as over-budget).
    m.mgr->on_seq_free(1);
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 11);
    const auto skips_before = m.mgr->stats().budget_skips;
    m.demote(0, 31, /*seq=*/4);              // 6 candidates, 5 slots free
    EXPECT_EQ(m.mgr->stats().budget_skips, skips_before)
        << "hibernated holders must not inflate the live census";
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 16);   // pool exhausted
    m.mgr->on_seq_free(4);
    m.mgr->on_seq_free(3);
}

// ── TD-KVT-COLD-FULL-HOT-WEDGE: cold-full skip + out-of-step recovery ──────

TEST(KvTieringManager, ColdPoolFullWedgeRecoversViaPressureDemote) {
    REQUIRES_GPU();
    // The wedge class end-to-end at manager level: a FULL cold pool skips
    // demotions in-step (fail-safe for DATA — a cold slot is the sole copy
    // of demoted KV, the pool never evicts one), the live sequence
    // accumulates a hot behind-window backlog, holder EVICTION returns the
    // slots (INV-KVT-17), and the out-of-step pressure sweep then drains
    // the backlog WITHOUT a step — the freed pages are what un-wedges
    // kMain.  ratio 1.0 → per-layer 2 pages × 2 layers = 4-slot pool.
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/8, /*full_mask=*/{},
                     /*host_to_device_ratio=*/1.0);
    ASSERT_EQ(m.mgr->cold_pool_capacity_pages(), 4);
    const int NL = static_cast<int>(m.handles.size()) / kLayers;

    // Holder seq 9 hibernates layer 0: 7 eligible pages, 4 slots — the
    // pool fills and the tail is skipped in-step.  A sequence that itself
    // holds EVERY slot skips via the cap check (budget_skips: cold_used ==
    // capacity), before the acquire path can see the empty free list.
    m.fx.be->synchronize_device();
    const int hib = m.mgr->hibernate_layer(0, /*seq=*/9,
                                           m.handles.data() + 0, NL, kLayers);
    EXPECT_EQ(hib, 4);
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 4);
    EXPECT_EQ(m.mgr->stats().budget_skips, 3u);
    EXPECT_EQ(m.mgr->stats().cold_full_skips, 0u);
    ASSERT_EQ(m.freed.size(), 4u);

    // Live seq 1 steps at pos 23: 4 window-demotion candidates (pages
    // 0..3), ALL skipped in-step — a DIFFERENT sequence hitting the full
    // pool takes the acquire-fail path (cold_full_skips) — and no VRAM
    // page returns: the wedge class.
    m.demote(0, 23, /*seq=*/1);
    EXPECT_EQ(m.mgr->stats().cold_full_skips, 1u);
    EXPECT_EQ(m.freed.size(), 4u);
    EXPECT_FALSE(m.mgr->layer_has_cold(1, 0));

    // Pressure sweep while the pool is STILL full: reclaims nothing — the
    // dispatcher then surfaces the retryable kKvPoolExhausted error and the
    // orchestrator's eviction seam takes over.
    EXPECT_EQ(m.mgr->pressure_demote(1, m.handles.data(), NL), 0);
    EXPECT_EQ(m.mgr->stats().pressure_demoted, 0u);

    // Holder eviction (the orchestrator seam): every slot returns at
    // refcount 0 (INV-KVT-17)...
    m.mgr->on_seq_free(9);
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 0);

    // ...and the sweep now drains the live backlog out-of-step.  S3: the
    // sweep flushes ALL layers as one batch ordered by physical page, so
    // the 4 scarce slots go to the LOWEST token positions across BOTH
    // layers (position-first draining is what completes slabs) — in this
    // aliased fixture (page_idx == j on both layers) that is logical 0..1
    // of each layer, not layer 0's 0..3 as the per-layer legacy sweep
    // gave.  Fail-safe (remaining pages stay hot) is unchanged.
    m.fx.be->synchronize_device();
    const int enq = m.mgr->pressure_demote(1, m.handles.data(), NL);
    EXPECT_EQ(enq, 4);
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    EXPECT_EQ(m.freed.size(), 8u);  // 4 holder + 4 backlog pages reclaimed
    EXPECT_EQ(m.mgr->stats().pressure_demoted, 4u);

    // Data integrity: the pressure-demoted bytes are byte-exact in the cold
    // pool even with the freed VRAM pages clobbered (INV-KVT-1), and the
    // retention window + append frontier stayed hot (INV-KVT-4).
    std::vector<char> junk(4 * kStrideBlock, '\x55');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());
    std::vector<char> want(kStrideBlock);
    int cold_pages = 0;
    for (int j = 0; j < 4; ++j) {
        for (int l = 0; l < kLayers; ++l) {
            const void* cold = m.mgr->cold_page_host_ptr(1, l, j);
            if (!cold) continue;
            ++cold_pages;
            // Both layers alias physical page j (fixture) — the ground
            // truth is the layer-0 fill pattern of position j.
            for (int r = 0; r < kPageSize; ++r)
                fill_row(want.data() + r * kStrideRow, 0,
                         j * kPageSize + r);
            EXPECT_EQ(std::memcmp(cold, want.data(),
                                  static_cast<size_t>(kStrideBlock)), 0)
                << "pressure-demoted page " << j << " (layer " << l
                << ") bytes differ";
        }
    }
    EXPECT_EQ(cold_pages, 4) << "exactly the 4 freed slots were reused";
    for (int j = 0; j < 2; ++j)
        for (int l = 0; l < kLayers; ++l)
            EXPECT_NE(m.mgr->cold_page_host_ptr(1, l, j), nullptr)
                << "lowest positions must win the scarce slots (position-"
                   "first draining completes slabs)";
    for (int j = 4; j < NL; ++j)
        EXPECT_EQ(m.mgr->cold_page_host_ptr(1, 0, j), nullptr)
            << "page " << j << " must stay hot (window/frontier)";
    m.mgr->on_seq_free(1);
}

TEST(KvTieringManager, PressureDemoteSkipsHibernatedUnknownAndFresh) {
    REQUIRES_GPU();
    // The pressure sweep only drains LIVE stepped sequences' backlogs:
    // unknown / never-stepped sequences are no-ops, hibernated holders have
    // nothing demote-eligible left (their frontier page must stay hot,
    // INV-KVT-4), a sequence whose whole window is hot enqueues nothing,
    // and the sweep covers EVERY layer exactly once (idempotent after).
    ManagerFixture m(/*hot_slots=*/8);
    const int NL = static_cast<int>(m.handles.size()) / kLayers;

    // Unknown sequence: no state — no-op.
    EXPECT_EQ(m.mgr->pressure_demote(7, m.handles.data(), NL), 0);

    // Hibernated holder: no-op even with hot pages remaining above its
    // hibernation frontier.
    m.fx.be->synchronize_device();
    ASSERT_GT(m.mgr->hibernate_layer(0, /*seq=*/9, m.handles.data(), NL,
                                     kLayers, /*frontier_logical=*/2), 0);
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    EXPECT_EQ(m.mgr->pressure_demote(9, m.handles.data(), NL), 0);

    // Live sequence with its whole retention window hot: nothing behind the
    // window at pos 7 — enqueues 0, everything stays hot.
    m.fx.be->synchronize_device();
    const int* bts[1] = {m.bt()};
    ASSERT_TRUE(m.mgr->begin_layer(0, /*seq=*/1, /*pos=*/7, bts));
    EXPECT_EQ(m.mgr->pressure_demote(1, m.handles.data(), NL), 0);
    EXPECT_FALSE(m.mgr->layer_has_cold(1, 0));

    // After an in-step layer-0 sweep at pos 23 (slots free — nothing
    // skipped), the pressure sweep still finds layer 1's backlog (the
    // fixture's step only swept layer 0)...
    m.demote(0, 23, /*seq=*/1);
    m.fx.be->synchronize_device();
    EXPECT_EQ(m.mgr->pressure_demote(1, m.handles.data(), NL), 4);
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    // ...and a second sweep is a no-op: no residual backlog.
    EXPECT_EQ(m.mgr->pressure_demote(1, m.handles.data(), NL), 0);
    m.mgr->on_seq_free(1);
    m.mgr->on_seq_free(9);
}

// ── 7. TD-KVT-SPEC: snapshot cold reads + rewind narrowing ─────────────────

TEST(KvTieringManager, ColdPageHostPtrCapturesBothTiersForSnapshot) {
    REQUIRES_GPU();
    // CMD_SEQ_SNAPSHOT of a tiered sequence reads demoted pages from the
    // pinned cold pool (cold_page_host_ptr) — the bytes must equal the
    // original VRAM page EXACTLY even after the freed pool pages are
    // clobbered (proving the cold copy, not VRAM, is the source).
    ManagerFixture m(/*hot_slots=*/8);
    m.demote(/*layer=*/0, /*pos=*/31);  // pages 0..5 demote
    m.mgr->drain_demotions();           // snapshot precondition
    ASSERT_EQ(m.freed.size(), 6u);

    std::vector<char> junk(6 * kStrideBlock, '\x77');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());

    std::vector<char> want(kStrideBlock);
    for (int j = 0; j < 6; ++j) {
        const void* cold = m.mgr->cold_page_host_ptr(1, 0, j);
        ASSERT_NE(cold, nullptr) << "demoted page " << j << " must be COLD";
        for (int r = 0; r < kPageSize; ++r)
            fill_row(want.data() + r * kStrideRow, 0, j * kPageSize + r);
        EXPECT_EQ(std::memcmp(cold, want.data(),
                              static_cast<size_t>(kStrideBlock)), 0)
            << "cold page " << j << " bytes != pre-demotion VRAM bytes";
    }
    // Hot pages / untouched layers / out-of-range logicals: no cold copy.
    EXPECT_EQ(m.mgr->cold_page_host_ptr(1, 0, 6), nullptr);
    EXPECT_EQ(m.mgr->cold_page_host_ptr(1, 0, 7), nullptr);
    EXPECT_EQ(m.mgr->cold_page_host_ptr(1, 1, 0), nullptr);
    EXPECT_EQ(m.mgr->cold_page_host_ptr(1, 0, 999), nullptr);
    EXPECT_EQ(m.mgr->cold_page_host_ptr(1, -1, 0), nullptr);
    EXPECT_EQ(m.mgr->cold_page_host_ptr(/*unknown seq*/ 42, 0, 0), nullptr);
}

TEST(KvTieringManager, RewindAllowedAboveDemotedFrontierThrowsBelow) {
    REQUIRES_GPU();
    // TD-KVT-SPEC narrowing (boundary tightened by TD-KVT-SPEC-FORK): the
    // step's k_append REWRITES position token_pos, so a rewind is legal
    // without re-promotion only while demoted_frontier <= token_pos (no
    // cold page holds a position that gets rewritten).  Reaching
    // begin_layer below that without the dispatcher's repromote hook
    // throws (INV-KVT-2 fail-loud backstop).
    ManagerFixture m(/*hot_slots=*/8);
    m.demote(/*layer=*/0, /*pos=*/31);  // pages 0..5 → frontier = 24
    const int* bts[1] = {m.bt()};
    // max_pos_seen = 32; rewind to pos 27: rewrites 27 (page 6, hot) — legal.
    EXPECT_TRUE(m.mgr->begin_layer(0, 1, /*pos=*/27, bts));
    // pos 24: rewrites 24 (page 6, hot); [0, 24) ⊇ demoted [0, 24) — legal
    // boundary.
    EXPECT_TRUE(m.mgr->begin_layer(0, 1, /*pos=*/24, bts));
    // pos 23: would rewrite position 23 INSIDE cold page 5 — fail-loud
    // without prior re-promotion (the old frontier <= pos+1 boundary let a
    // k_append target a demoted page's last row through a neutralized
    // handle).
    EXPECT_THROW(m.mgr->begin_layer(0, 1, /*pos=*/23, bts),
                 std::runtime_error);
    // pos 20: strictly inside demoted territory — fail-loud.
    EXPECT_THROW(m.mgr->begin_layer(0, 1, /*pos=*/20, bts),
                 std::runtime_error);
}

// ── TD-KVT-SPEC-FORK: fork sharing, re-promotion, restore re-init ──────────

TEST(KvTieringManager, ForkSharesRefcountedColdSlotsDoubleFreeSafe) {
    REQUIRES_GPU();
    // seq_fork of a demoted parent: the child REFCOUNT-shares the parent's
    // cold slots (no cold-byte copy, no extra pool usage); either teardown
    // order returns each slot exactly once (double-free-safe), and both
    // family members materialize byte-exact from the shared copies.
    ManagerFixture m(/*hot_slots=*/8);
    m.demote(/*layer=*/0, /*pos=*/31, /*seq=*/1);  // pages 0..5 cold
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 6);

    m.mgr->on_seq_fork(1, 2);
    EXPECT_TRUE(m.mgr->seq_has_demotions(2));
    EXPECT_TRUE(m.mgr->layer_has_cold(2, 0));
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 6)
        << "shared slots — usage must NOT double";
    EXPECT_EQ(m.mgr->seq_cold_used_pages(1, 0), 6);
    EXPECT_EQ(m.mgr->seq_cold_used_pages(2, 0), 6);

    // The child inherits the demoted frontier: an un-repromoted rewind into
    // the shared territory fail-louds for the child too (INV-KVT-2).
    const int* bts[1] = {m.bt()};
    EXPECT_THROW(m.mgr->begin_layer(0, 2, /*pos=*/10, bts),
                 std::runtime_error);

    // Clobber the freed VRAM pages: both sequences' ONLY correct source is
    // the single shared cold copy per page.
    std::vector<char> junk(6 * kStrideBlock, '\x21');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());
    m.materialize_and_check(0, {0, 5, 9, 23, 28}, /*seq=*/1);
    m.materialize_and_check(0, {0, 9, 23, 28}, /*seq=*/2);

    // Parent teardown releases only ITS holds: the child keeps every slot.
    m.mgr->on_seq_free(1);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 6)
        << "child's refcounts must keep the shared slots alive";
    EXPECT_FALSE(m.mgr->seq_has_demotions(1));
    EXPECT_TRUE(m.mgr->seq_has_demotions(2));
    m.materialize_and_check(0, {0, 5, 9, 23}, /*seq=*/2);  // still byte-exact

    // Child teardown returns the slots exactly once.
    m.mgr->on_seq_free(2);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);
    EXPECT_FALSE(m.mgr->has_demotions());
}

TEST(KvTieringManager, RewindIntoDemotedRepromotesThenSucceeds) {
    REQUIRES_GPU();
    // Rewind INTO demoted territory: the dispatcher pre-step hook
    // (repromote_for_rewind) re-promotes every cold page holding a position
    // >= token_pos back to fresh VRAM pages with the EXACT demoted bytes
    // (INV-KVT-1), releases their slots and lowers the frontier — the
    // rewound step then engages normally.  VRAM exhaustion keeps the step
    // fail-closed (capacity, not correctness).
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/8, /*full_mask=*/{},
                     /*host_to_device_ratio=*/8.0, /*spare_pages=*/8);
    m.demote(/*layer=*/0, /*pos=*/31);  // pages 0..5 cold → frontier 24
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 6);

    // Clobber the freed VRAM pages: re-promotion must restore from the
    // pinned cold pool, never from the (reused) VRAM.
    std::vector<char> junk(6 * kStrideBlock, '\x42');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());

    // Rewind to pos 10: positions >= 10 must go hot → pages 2..5 re-promote
    // (page 2 holds 8..11 ∋ 10); pages 0..1 (positions 0..7) stay cold.
    ASSERT_TRUE(m.mgr->repromote_for_rewind(1, /*token_pos=*/10));
    EXPECT_EQ(m.mgr->stats().repromoted_pages, 4u);
    ASSERT_EQ(m.alloced.size(), 4u);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 2);
    EXPECT_TRUE(m.mgr->seq_has_demotions(1));  // pages 0..1 remain cold

    // Re-promoted bytes == the exact demoted bytes (INV-KVT-1).
    std::vector<char> want(kStrideBlock);
    for (const auto& [layer, logical, phys] : m.alloced) {
        ASSERT_EQ(layer, 0);
        const auto got = m.read_page(phys);
        for (int r = 0; r < kPageSize; ++r)
            fill_row(want.data() + r * kStrideRow, 0,
                     logical * kPageSize + r);
        EXPECT_EQ(std::memcmp(got.data(), want.data(),
                              static_cast<size_t>(kStrideBlock)), 0)
            << "re-promoted page " << logical << " bytes != demoted bytes";
    }

    // The rewound step now engages (frontier 8 <= pos 10).
    const int* bts[1] = {m.bt()};
    EXPECT_TRUE(m.mgr->begin_layer(0, 1, /*pos=*/10, bts));
    // Mixed materialization: rows 0/5 still cold (pinned pool), row 9 now a
    // pool hit through the re-promoted page's updated block-table entry.
    m.materialize_and_check(0, {0, 5, 9}, /*seq=*/1);
    const auto& s = m.mgr->stats();
    EXPECT_EQ(s.cold_misses, 2u);
    EXPECT_EQ(s.pool_hits, 1u);

    // Deeper rewind with the VRAM pool exhausted: re-promotion reports
    // failure and the un-repromoted step fail-louds (INV-KVT-2).
    m.next_free_page = m.fx.pool_pages;  // exhaust the alloc seam
    EXPECT_FALSE(m.mgr->repromote_for_rewind(1, /*token_pos=*/3));
    EXPECT_THROW(m.mgr->begin_layer(0, 1, /*pos=*/3, bts),
                 std::runtime_error);
}

TEST(KvTieringManager, RepromoteAllReinitsForRestoreOntoDemoted) {
    REQUIRES_GPU();
    // seq_restore ONTO a demoted sequence: repromote_seq(seq, 0) re-promotes
    // EVERYTHING — every handle valid again (restore then writes through
    // them), no cold slots held, frontier 0 — and tiering can re-demote
    // from scratch on subsequent steps.
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/8, /*full_mask=*/{},
                     /*host_to_device_ratio=*/8.0, /*spare_pages=*/8);
    m.demote(/*layer=*/0, /*pos=*/31);  // pages 0..5 cold
    std::vector<char> junk(6 * kStrideBlock, '\x7E');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());

    ASSERT_TRUE(m.mgr->repromote_seq(1, /*keep_frontier=*/0));
    EXPECT_FALSE(m.mgr->has_demotions());
    EXPECT_FALSE(m.mgr->seq_has_demotions(1));
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);
    EXPECT_EQ(m.mgr->stats().repromoted_pages, 6u);
    ASSERT_EQ(m.alloced.size(), 6u);

    // Exact bytes restored for every page (INV-KVT-1).
    std::vector<char> want(kStrideBlock);
    for (const auto& [layer, logical, phys] : m.alloced) {
        const auto got = m.read_page(phys);
        for (int r = 0; r < kPageSize; ++r)
            fill_row(want.data() + r * kStrideRow, 0,
                     logical * kPageSize + r);
        EXPECT_EQ(std::memcmp(got.data(), want.data(),
                              static_cast<size_t>(kStrideBlock)), 0)
            << "re-promoted page " << logical << " bytes != demoted bytes";
    }

    // A rewind to ANY position is now legal (frontier 0) — the restore's
    // subsequent decode from an earlier position engages cleanly...
    const int* bts[1] = {m.bt()};
    EXPECT_TRUE(m.mgr->begin_layer(0, 1, /*pos=*/0, bts));
    // ...and demotion restarts from scratch with fresh slots.
    m.freed.clear();
    m.demote(/*layer=*/0, /*pos=*/31);
    EXPECT_EQ(m.freed.size(), 6u);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 6);
    EXPECT_TRUE(m.mgr->seq_has_demotions(1));
}

TEST(KvTieringManager, ForkChildRewindSplitsColdCopyOnWriteParentIntact) {
    REQUIRES_GPU();
    // CoW-on-rewind across a fork family: the child rewinding into SHARED
    // demoted territory re-promotes its OWN fresh VRAM copies and releases
    // only its refcounts — the parent's cold copies survive untouched and
    // keep serving byte-exact reads.
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/8, /*full_mask=*/{},
                     /*host_to_device_ratio=*/8.0, /*spare_pages=*/8);
    m.demote(/*layer=*/0, /*pos=*/31, /*seq=*/1);  // pages 0..5 cold
    m.mgr->on_seq_fork(1, 2);
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 6);

    std::vector<char> junk(6 * kStrideBlock, '\x9C');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());

    // Child rewinds to pos 10: its pages 2..5 split copy-on-write.
    ASSERT_TRUE(m.mgr->repromote_for_rewind(2, /*token_pos=*/10));
    EXPECT_EQ(m.mgr->stats().repromoted_pages, 4u);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 6)
        << "parent's refcounts must keep every shared slot alive";
    EXPECT_EQ(m.mgr->seq_cold_used_pages(1, 0), 6);
    EXPECT_EQ(m.mgr->seq_cold_used_pages(2, 0), 2);

    const int* bts[1] = {m.bt()};
    EXPECT_TRUE(m.mgr->begin_layer(0, 2, /*pos=*/10, bts));

    // Parent: every demoted row still reads byte-exact from ITS (shared)
    // cold copies; hot rows through the block table.
    m.materialize_and_check(0, {0, 9, 23, 24, 28}, /*seq=*/1);
    // Child: row 0 still cold (shared slot), row 9 re-promoted (pool hit
    // via the child's fresh VRAM page).
    m.materialize_and_check(0, {0, 9}, /*seq=*/2);

    // Parent teardown: slots 2..5 (child released its refs) free now; the
    // child's still-shared pages 0..1 survive.
    m.mgr->on_seq_free(1);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 2);
    m.materialize_and_check(0, {0, 5}, /*seq=*/2);  // byte-exact to the end
    m.mgr->on_seq_free(2);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);
    EXPECT_FALSE(m.mgr->has_demotions());
}

TEST(KvTieringManager, TruncatedForkSharesOnlyPrefixColdSlots) {
    REQUIRES_GPU();
    // R4a truncating fork from a demoted (e.g. HIBERNATED holder) parent:
    // on_seq_fork(src, dst, prefix_len) truncates the child's inherited
    // tiering state to logical pages [0, ceil(prefix_len / page_size)) —
    // only the KEPT cold slots gain a refcount (a full-copy share would
    // leak the parent's tail slots at the child's release, since
    // release_seq walks the child's own page vectors) — and the child's
    // demoted frontier / cold accounting are recomputed from the kept
    // prefix.  The COLD straddling page then re-promotes copy-on-write
    // exactly like a rewind (the dispatcher calls
    // repromote_for_rewind(dst, prefix_len) for a live truncated child).
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/8, /*full_mask=*/{},
                     /*host_to_device_ratio=*/8.0, /*spare_pages=*/8);
    m.demote(/*layer=*/0, /*pos=*/31, /*seq=*/1);  // pages 0..5 cold
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 6);

    // prefix_len 10 → keep ceil(10/4) = 3 logical pages (0..2).
    m.mgr->on_seq_fork(1, 2, /*prefix_len=*/10);
    EXPECT_TRUE(m.mgr->seq_has_demotions(2));
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 6) << "shared, never doubled";
    EXPECT_EQ(m.mgr->seq_cold_used_pages(1, 0), 6);  // parent untouched
    EXPECT_EQ(m.mgr->seq_cold_used_pages(2, 0), 3)
        << "child holds ONLY the prefix slots";

    // Child frontier = 12 (page 2 straddles the boundary at 10): a step
    // writing at 10 fail-louds until the straddle re-promotes (INV-KVT-2).
    const int* bts[1] = {m.bt()};
    EXPECT_THROW(m.mgr->begin_layer(0, 2, /*pos=*/10, bts),
                 std::runtime_error);
    ASSERT_TRUE(m.mgr->repromote_for_rewind(2, /*token_pos=*/10));
    EXPECT_EQ(m.mgr->stats().repromoted_pages, 1u) << "straddle page only";
    EXPECT_EQ(m.mgr->seq_cold_used_pages(2, 0), 2);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 6)
        << "parent refcounts keep every slot alive";
    EXPECT_TRUE(m.mgr->begin_layer(0, 2, /*pos=*/10, bts));

    // Parent teardown: the never-shared tail (pages 3..5) AND page 2 (the
    // child released its ref at re-promotion) free; the shared prefix
    // (pages 0..1) survives on the child's refs, byte-exact.
    m.mgr->on_seq_free(1);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 2);
    m.materialize_and_check(0, {0, 5}, /*seq=*/2);
    m.mgr->on_seq_free(2);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);
    EXPECT_FALSE(m.mgr->has_demotions());
}

TEST(KvTieringManager, TruncatedForkHotStraddleNeedsNoRepromotion) {
    REQUIRES_GPU();
    // Truncation boundary in HOT territory of a partially demoted parent:
    // the kept prefix still includes the parent's cold pages (window
    // demotion colds the OLDEST positions, so cold pages are always a
    // position-prefix), but the STRADDLING page is hot — the child's
    // recomputed frontier is already <= prefix_len, so a step at the
    // boundary engages with NO re-promotion and the cold prefix keeps
    // serving reads from the shared slots.
    ManagerFixture m(/*hot_slots=*/24);
    m.demote(/*layer=*/0, /*pos=*/31, /*seq=*/1);  // demote_end 8 → pages 0..1
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 2);

    // prefix_len 10 → keep pages 0..2; pages 0..1 cold, straddle page 2 HOT.
    m.mgr->on_seq_fork(1, 2, /*prefix_len=*/10);
    EXPECT_EQ(m.mgr->seq_cold_used_pages(2, 0), 2);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 2);

    // Child frontier 8 <= 10: the boundary step runs without re-promotion.
    const int* bts[1] = {m.bt()};
    EXPECT_TRUE(m.mgr->begin_layer(0, 2, /*pos=*/10, bts));
    EXPECT_EQ(m.mgr->stats().repromoted_pages, 0u);
    m.materialize_and_check(0, {0, 5, 9}, /*seq=*/2);  // cold+hot byte-exact

    m.mgr->on_seq_free(1);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 2) << "child refs pin both";
    m.mgr->on_seq_free(2);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);
}

// ── 6. TD-KVT-SYNC / TD-KVT-PREFETCH ────────────────────────────────────────

TEST(KvTieringManager, IndexShareReuseAndLookaheadPrefetch) {
    REQUIRES_GPU();
    // Layer 0 FULL, layer 1 SHARED (reuses layer 0's selection).  Demote
    // both layers' pages 0..5 (positions 0..23 cold), then clobber the pool:
    // every correct byte must come from the cold copies.
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/8,
                     /*full_mask=*/{1, 0});
    m.demote(/*layer=*/0, /*pos=*/31);
    m.demote(/*layer=*/1, /*pos=*/31);
    ASSERT_TRUE(m.mgr->layer_has_cold(1, 0));
    ASSERT_TRUE(m.mgr->layer_has_cold(1, 1));
    std::vector<char> junk(6 * kStrideBlock, '\xAA');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());

    const std::vector<int> sel = {23, 0, 31, 9, 5, 24, 28};  // cold: 23,0,9,5
    const int* bts[1] = {m.bt()};

    // ── FULL layer 0: prepare(fresh) → materialize.  The overlapped
    // readback is consumed (no sync fallback) and the lookahead prefetch
    // stages layer 1's 4 cold misses into its row cache.
    m.upload_selection(sel);
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 31, bts));
    m.mgr->prepare(0, 0, static_cast<const int*>(m.dev_indices),
                   static_cast<const int*>(m.dev_len), 1, m.fx.stream,
                   /*selection_fresh=*/true);
    {
        layerstorm::parallelism::TieredKvView tv{};
        ASSERT_TRUE(m.mgr->materialize(
            0, 0, static_cast<const int*>(m.dev_indices),
            static_cast<const int*>(m.dev_len), 1, m.fx.stream, &tv));
        m.check_scratch(tv, sel);
    }
    const auto& s1 = m.mgr->stats();
    EXPECT_EQ(s1.prepares, 1u);
    EXPECT_EQ(s1.sync_overlapped, 1u);
    EXPECT_EQ(s1.sync_fallbacks, 0u);
    EXPECT_EQ(s1.cold_misses, 4u);
    EXPECT_EQ(s1.prefetch_bursts, 1u);
    EXPECT_EQ(s1.prefetch_rows, 4u) << "layer 1's cold misses prefetched";
    EXPECT_EQ(m.mgr->cache_entries(0, 1), 4);

    // ── SHARED layer 1: prepare(fresh=false) blesses host-copy reuse.
    // CLOBBER the device selection buffer: correct materialization proves
    // the manager did NOT re-read it (no D2H at all).
    ASSERT_TRUE(m.mgr->begin_layer(1, 1, 31, bts));
    m.mgr->prepare(0, 1, static_cast<const int*>(m.dev_indices),
                   static_cast<const int*>(m.dev_len), 1, m.fx.stream,
                   /*selection_fresh=*/false);
    {
        std::vector<int> garbage(kTopk, 7);  // valid-looking but WRONG
        m.fx.be->memcpy_h2d(m.dev_indices, garbage.data(),
                            kTopk * sizeof(int));
        layerstorm::parallelism::TieredKvView tv{};
        ASSERT_TRUE(m.mgr->materialize(
            0, 1, static_cast<const int*>(m.dev_indices),
            static_cast<const int*>(m.dev_len), 1, m.fx.stream, &tv));
        m.check_scratch(tv, sel);  // == ORIGINAL selection, not the garbage
    }
    const auto& s2 = m.mgr->stats();
    EXPECT_EQ(s2.prepares, 1u) << "shared layer must not issue a readback";
    EXPECT_EQ(s2.sync_reuses, 1u);
    EXPECT_EQ(s2.sync_fallbacks, 0u);
    EXPECT_EQ(s2.cold_misses, 4u) << "prefetch must eliminate layer 1's "
                                     "cold misses";
    EXPECT_EQ(s2.h2d_bursts, 1u) << "no synchronous burst for layer 1";
    EXPECT_EQ(s2.prefetch_hits, 4u);
    EXPECT_EQ(s2.cache_hits, 4u);
}

TEST(KvTieringManager, PrefetchMatchesSyncFetchBytes) {
    REQUIRES_GPU();
    // Prefetch correctness = same materialized bytes as the sync-fetch path:
    // run the SAME selection through a prefetched successor (fixture A) and
    // through the plain sync path (fixture B, no prepare) — both must equal
    // ground truth (transitively each other), with A taking zero cold
    // misses on the successor and B taking them all.
    const std::vector<int> sel = {0, 4, 8, 12, 16, 20, 30};  // 6 cold + 1 hot
    auto run = [&](bool use_prepare) -> std::vector<uint64_t> {
        ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/8,
                         /*full_mask=*/{1, 0});
        m.demote(0, 31);
        m.demote(1, 31);
        std::vector<char> junk(6 * kStrideBlock, '\x5A');
        m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());
        const int* bts[1] = {m.bt()};
        m.upload_selection(sel);
        EXPECT_TRUE(m.mgr->begin_layer(0, 1, 31, bts)) << "begin_layer(0)";
        if (use_prepare) {
            m.mgr->prepare(0, 0, static_cast<const int*>(m.dev_indices),
                           static_cast<const int*>(m.dev_len), 1,
                           m.fx.stream, /*selection_fresh=*/true);
        }
        {
            layerstorm::parallelism::TieredKvView tv{};
            EXPECT_TRUE(m.mgr->materialize(
                0, 0, static_cast<const int*>(m.dev_indices),
                static_cast<const int*>(m.dev_len), 1, m.fx.stream, &tv));
            m.check_scratch(tv, sel);
        }
        EXPECT_TRUE(m.mgr->begin_layer(1, 1, 31, bts)) << "begin_layer(1)";
        if (use_prepare) {
            m.mgr->prepare(0, 1, static_cast<const int*>(m.dev_indices),
                           static_cast<const int*>(m.dev_len), 1,
                           m.fx.stream, /*selection_fresh=*/false);
        }
        {
            layerstorm::parallelism::TieredKvView tv{};
            EXPECT_TRUE(m.mgr->materialize(
                0, 1, static_cast<const int*>(m.dev_indices),
                static_cast<const int*>(m.dev_len), 1, m.fx.stream, &tv));
            m.check_scratch(tv, sel);  // byte-identical to ground truth
        }
        const auto& s = m.mgr->stats();
        return {s.cold_misses, s.prefetch_rows, s.prefetch_hits};
    };
    const auto with_pf = run(/*use_prepare=*/true);
    const auto without_pf = run(/*use_prepare=*/false);
    EXPECT_EQ(with_pf[0], 6u) << "layer 0 misses only (layer 1 prefetched)";
    EXPECT_EQ(with_pf[1], 6u);
    EXPECT_EQ(with_pf[2], 6u);
    EXPECT_EQ(without_pf[0], 12u) << "sync path pays both layers' misses";
    EXPECT_EQ(without_pf[1], 0u);
}

TEST(KvTieringManager, PrepareOverlapEliminatesHostSyncWait) {
    REQUIRES_GPU();
    // TD-KVT-SYNC measurement: under a BUSY attention stream, the legacy
    // path host-waits for the whole enqueued stretch (the D2H is ordered
    // behind it), while prepare() issues the D2H BEFORE the stretch — the
    // materialize-time wait collapses to ~0.
    ManagerFixture m(/*hot_slots=*/8);
    m.demote(0, 31);
    std::vector<char> junk(6 * kStrideBlock, '\xCC');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());
    auto* be = m.fx.be.get();

    const size_t dummy_bytes = 64ull << 20;
    void* da = be->device_alloc(dummy_bytes);
    void* db = be->device_alloc(dummy_bytes);
    ASSERT_TRUE(da && db);
    auto busy_stream = [&](int reps) {
        for (int i = 0; i < reps; ++i)
            be->memcpy_d2d_async(da, db, dummy_bytes, m.fx.stream);
    };
    const std::vector<int> sel = {23, 0, 31, 9, 5, 24, 28};
    const int* bts[1] = {m.bt()};

    // Legacy: selection upload → busy stretch → materialize (sync fallback).
    m.upload_selection(sel);
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 31, bts));
    busy_stream(200);
    uint64_t wait0 = m.mgr->stats().sync_wait_us;
    {
        layerstorm::parallelism::TieredKvView tv{};
        ASSERT_TRUE(m.mgr->materialize(
            0, 0, static_cast<const int*>(m.dev_indices),
            static_cast<const int*>(m.dev_len), 1, m.fx.stream, &tv));
        m.check_scratch(tv, sel);
    }
    const uint64_t legacy_wait = m.mgr->stats().sync_wait_us - wait0;
    EXPECT_EQ(m.mgr->stats().sync_fallbacks, 1u);
    be->synchronize_device();

    // Overlapped: selection upload → prepare → busy stretch → materialize.
    m.upload_selection(sel);
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 31, bts));
    m.mgr->prepare(0, 0, static_cast<const int*>(m.dev_indices),
                   static_cast<const int*>(m.dev_len), 1, m.fx.stream,
                   /*selection_fresh=*/true);
    busy_stream(200);
    wait0 = m.mgr->stats().sync_wait_us;
    {
        layerstorm::parallelism::TieredKvView tv{};
        ASSERT_TRUE(m.mgr->materialize(
            0, 0, static_cast<const int*>(m.dev_indices),
            static_cast<const int*>(m.dev_len), 1, m.fx.stream, &tv));
        m.check_scratch(tv, sel);
    }
    const uint64_t overlapped_wait = m.mgr->stats().sync_wait_us - wait0;
    EXPECT_EQ(m.mgr->stats().sync_overlapped, 1u);
    be->synchronize_device();
    be->device_free(da);
    be->device_free(db);

    // 200 × 64 MiB D2D ≈ several ms of stream backlog; the overlapped wait
    // must be a small fraction of the legacy wait (generous margin for CI
    // noise; typical: legacy = ms-scale, overlapped = µs-scale).
    spdlog::info("KvTiering SYNC measurement: legacy_wait={} us, "
                 "overlapped_wait={} us", legacy_wait, overlapped_wait);
    EXPECT_GT(legacy_wait, 1000u) << "busy stretch too short to measure";
    EXPECT_LT(overlapped_wait, legacy_wait / 4)
        << "prepare() failed to overlap the selection readback";
}

TEST(KvTieringManager, StalePendingReadbackDrainsAcrossSteps) {
    REQUIRES_GPU();
    // A prepare()-issued readback that is never consumed (its layer went
    // untiered) must be drained by the next step's fallback — and the NEW
    // selection must win, byte-exact.
    ManagerFixture m(/*hot_slots=*/8, /*num_logical=*/8,
                     /*full_mask=*/{1, 0});
    m.demote(0, 31);
    std::vector<char> junk(6 * kStrideBlock, '\x33');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());
    const int* bts[1] = {m.bt()};

    const std::vector<int> old_sel = {0, 4, 8};
    m.upload_selection(old_sel);
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 31, bts));
    m.mgr->prepare(0, 0, static_cast<const int*>(m.dev_indices),
                   static_cast<const int*>(m.dev_len), 1, m.fx.stream,
                   /*selection_fresh=*/true);
    // Never materialized (step ends).  Next step (new position): the
    // pending host copy is STALE — materialize must reject reuse, drain the
    // pending D2H, and synchronously re-read the NEW selection.
    const std::vector<int> new_sel = {23, 9, 5, 31};
    m.upload_selection(new_sel);
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, /*pos=*/32, bts));
    {
        layerstorm::parallelism::TieredKvView tv{};
        ASSERT_TRUE(m.mgr->materialize(
            0, 0, static_cast<const int*>(m.dev_indices),
            static_cast<const int*>(m.dev_len), 1, m.fx.stream, &tv));
        m.check_scratch(tv, new_sel);
    }
    EXPECT_EQ(m.mgr->stats().sync_fallbacks, 1u);
    EXPECT_EQ(m.mgr->stats().sync_overlapped, 0u);
}

TEST(KvTieringManager, ColdPoolOnGpuHomeNumaNode) {
    REQUIRES_GPU();
    // P-22 home-node rule: the cold pool + staging arena must be bound to the
    // rank GPU's NUMA node when a NumaManager is provided.
    cfg::HardwareConfig hw{};
    cfg::GpuConfig g{};
    g.id = 0;
    g.type = cfg::GpuType::rtx5090;
    g.numa_node = 0;
    g.ref = ref0();
    hw.gpus.push_back(g);
    lm::NumaManager nm(hw);
    if (!nm.numa_available()) GTEST_SKIP() << "libnuma unavailable";

    PoolFixture fx(4, 0);
    ld::KvTieringManager::Options o;
    o.dcp_size = 1;
    o.gpus = {ref0()};
    o.device_backends = {fx.be.get()};
    o.numa_manager = &nm;
    o.free_page = [](uint64_t, int, int, const lm::PageHandle&) {};
    o.kv_main_bases = {fx.pool};
    o.stride_block = kStrideBlock;
    o.stride_row = kStrideRow;
    o.page_size = kPageSize;
    o.kv_layers = kLayers;
    o.index_topk = kTopk;
    o.hot_buffer_slots = 8;
    ld::KvTieringManager mgr(std::move(o));
    EXPECT_EQ(mgr.cold_pool_numa_node(0), nm.gpu_numa_node(0))
        << "cold pool must live on the rank GPU's home NUMA node";
}

// ── 8. TD-KVT-DCP-SHARDED: per-rank shard tiering ───────────────────────────

namespace {

/// Two-rank sequence-sharded fixture on ONE physical GPU (both "ranks" share
/// the backend; positions 0/1 index device_backends).  dcp_chunk_tokens ==
/// kPageSize (one page per ownership chunk): global page j is owned by rank
/// j % 2 and is that rank's LOCAL page j / 2 (kv_shard_math round-robin).
/// Rank r's pool holds ONLY its own shard, in local-page order — exactly the
/// sharded allocator layout (INV-4.9e).
struct ShardedFixture {
    static constexpr int kLogical = 8;      // global logical pages
    static constexpr int kLocalPages = 4;   // per rank

    std::unique_ptr<lc::CudaSm120DeviceBackend> be;
    void* pools[2] = {nullptr, nullptr};
    void* stream = nullptr;
    std::unique_ptr<ld::KvTieringManager> mgr;
    std::vector<lm::PageHandle> handles;  // (global j, layer l) at [j*L + l]
    std::vector<int> host_bt;             // identity LOCAL table (both ranks)
    std::vector<std::pair<int, int>> freed;
    void* dev_indices = nullptr;
    void* dev_len = nullptr;

    explicit ShardedFixture(int hot_slots) {
        be = std::make_unique<lc::CudaSm120DeviceBackend>(ref0());
        be->set_device();
        stream = be->create_stream();
        for (int r = 0; r < 2; ++r) {
            const size_t bytes =
                static_cast<size_t>(kLocalPages) * kStrideBlock;
            pools[r] = be->device_alloc(bytes);
            std::vector<char> host(bytes);
            for (int jl = 0; jl < kLocalPages; ++jl) {
                const int g = 2 * jl + r;  // global page this local one backs
                for (int rr = 0; rr < kPageSize; ++rr)
                    fill_row(host.data() + jl * kStrideBlock + rr * kStrideRow,
                             0, g * kPageSize + rr);
            }
            be->memcpy_h2d(pools[r], host.data(), bytes);
        }
        for (int j = 0; j < kLogical; ++j) {
            const int owner = j % 2;
            for (int l = 0; l < kLayers; ++l)
                handles.push_back(lm::PageHandle{
                    .gpu_idx = owner, .page_idx = j / 2,
                    .gpu_ptr = static_cast<char*>(pools[owner])
                             + static_cast<int64_t>(j / 2) * kStrideBlock,
                    .pool = lm::Pool::kMain});
        }
        for (int jl = 0; jl < kLocalPages; ++jl) host_bt.push_back(jl);

        ld::KvTieringManager::Options o;
        o.dcp_size = 2;
        o.gpus = {cfg::GpuRef{.position = 0, .id = 0,
                              .type = cfg::GpuType::rtx5090},
                  cfg::GpuRef{.position = 1, .id = 0,
                              .type = cfg::GpuType::rtx5090}};
        o.device_backends = {be.get(), be.get()};
        o.stream_manager = nullptr;
        o.numa_manager = nullptr;
        o.free_page = [this](uint64_t, int layer, int logical,
                             const lm::PageHandle&) {
            freed.emplace_back(layer, logical);
        };
        o.kv_main_bases = {pools[0], pools[1]};
        o.stride_block = kStrideBlock;
        o.stride_row = kStrideRow;
        o.page_size = kPageSize;
        o.kv_layers = kLayers;
        o.index_topk = kTopk;
        o.hot_buffer_slots = hot_slots;
        o.host_to_device_ratio = 8.0;
        o.kv_sharded = true;
        o.dcp_chunk_tokens = kPageSize;
        mgr = std::make_unique<ld::KvTieringManager>(std::move(o));

        dev_indices = be->device_alloc(kTopk * sizeof(int));
        dev_len = be->device_alloc(sizeof(int));
    }
    ~ShardedFixture() {
        mgr.reset();
        be->set_device();
        be->device_free(dev_indices);
        be->device_free(dev_len);
        be->destroy_stream(stream);
        be->device_free(pools[0]);
        be->device_free(pools[1]);
    }

    /// Rank-LOCAL slot index of global token t (KVS-4 translation output;
    /// chunk == kPageSize, dcp == 2).
    static int local_of(int t) {
        const int c = t / kPageSize;
        return (c / 2) * kPageSize + t % kPageSize;
    }

    void upload_selection(const std::vector<int>& sel_local) {
        std::vector<int> padded(kTopk, -1);
        for (size_t i = 0; i < sel_local.size(); ++i) padded[i] = sel_local[i];
        const int n = static_cast<int>(sel_local.size());
        be->memcpy_h2d(dev_indices, padded.data(), kTopk * sizeof(int));
        be->memcpy_h2d(dev_len, &n, sizeof(int));
    }

    /// Materialize `globals` (all owned by `rank`) via their translated
    /// LOCAL indices and byte-compare the scratch against the GLOBAL ground
    /// truth rows.
    void materialize_and_check(int rank, const std::vector<int>& globals) {
        std::vector<int> sel_local;
        for (int t : globals) sel_local.push_back(local_of(t));
        upload_selection(sel_local);
        layerstorm::parallelism::TieredKvView tv{};
        const bool tiered = mgr->materialize(
            rank, /*layer=*/0, static_cast<const int*>(dev_indices),
            static_cast<const int*>(dev_len), /*batch=*/1, stream, &tv);
        ASSERT_TRUE(tiered) << "expected tiered path on rank " << rank;
        ASSERT_EQ(tv.seq_len_kv, static_cast<int>(globals.size()));
        be->synchronize_device();
        const size_t bytes = static_cast<size_t>(tv.max_blocks_per_seq)
                           * kStrideBlock;
        std::vector<char> out(bytes);
        be->memcpy_d2h_async(out.data(), tv.kv_cache, bytes, stream);
        be->synchronize_device();
        std::vector<char> want(kStrideRow);
        for (size_t i = 0; i < globals.size(); ++i) {
            fill_row(want.data(), 0, globals[i]);
            ASSERT_EQ(std::memcmp(
                out.data() + (i / kPageSize) * kStrideBlock
                    + (i % kPageSize) * kStrideRow,
                want.data(), kStrideRow), 0)
                << "rank " << rank << " materialized row " << i << " (global "
                << globals[i] << ", local " << sel_local[i]
                << ") != ground truth";
        }
    }
};

}  // namespace

TEST(KvTieringManagerSharded, OwnerOnlyDemotionAndLocalMaterializeByteExact) {
    REQUIRES_GPU();
    // 8 global pages × 4 tokens = positions 0..31; retention 8 tokens → at
    // pos 31 pages 0..5 demote.  Owners: rank 0 holds pages 0/2/4, rank 1
    // holds 1/3/5 — each demotes ITS shard only (single cold copy, in the
    // owner's pool).
    ShardedFixture m(/*hot_slots=*/8);
    const int* bts[2] = {m.host_bt.data(), m.host_bt.data()};
    m.be->synchronize_device();
    ASSERT_TRUE(m.mgr->begin_layer(0, /*seq=*/1, /*pos=*/31, bts));
    m.mgr->after_attention(0, 1, 31, m.handles.data(),
                           ShardedFixture::kLogical, kLayers);
    m.be->synchronize_device();
    m.mgr->drain_demotions();

    ASSERT_TRUE(m.mgr->has_demotions());
    ASSERT_TRUE(m.mgr->layer_has_cold(1, 0));
    ASSERT_EQ(m.freed.size(), 6u) << "each demoted page freed exactly ONCE";
    // S3: the demote flush orders by (storing owner rank, physical page
    // index) so per-rank D2H runs coalesce — the SET of freed pages is the
    // invariant, not the order.
    {
        std::vector<int> got;
        for (const auto& [l, j] : m.freed) {
            EXPECT_EQ(l, 0);
            got.push_back(j);
        }
        std::sort(got.begin(), got.end());
        EXPECT_EQ(got, (std::vector<int>{0, 1, 2, 3, 4, 5}));
    }
    EXPECT_EQ(m.mgr->stats().demoted_pages, 6u);

    // The cold copy must live in the OWNER's pool (owner-only D2H): the
    // snapshot seam returns byte-exact page content for every demoted page.
    std::vector<char> want(kStrideBlock);
    for (int j = 0; j < 6; ++j) {
        const void* cold = m.mgr->cold_page_host_ptr(1, 0, j);
        ASSERT_NE(cold, nullptr);
        for (int r = 0; r < kPageSize; ++r)
            fill_row(want.data() + r * kStrideRow, 0, j * kPageSize + r);
        EXPECT_EQ(std::memcmp(cold, want.data(),
                              static_cast<size_t>(kStrideBlock)), 0)
            << "cold copy of global page " << j << " mismatch";
    }
    EXPECT_EQ(m.mgr->cold_page_host_ptr(1, 0, 6), nullptr);  // hot frontier

    // Clobber the demoted local pages in BOTH pools: every correct byte must
    // now come from the owner's cold copy.
    std::vector<char> junk(3 * kStrideBlock, '\xEE');
    m.be->memcpy_h2d(m.pools[0], junk.data(), junk.size());
    m.be->memcpy_h2d(m.pools[1], junk.data(), junk.size());

    // Rank 0's translated local selection: globals {0, 9, 17, 24, 26}
    // (chunks 0/2/4/6 → rank 0) — cold 0/9/17, pool-hot 24/26.
    m.materialize_and_check(0, {0, 9, 17, 24, 26});
    const auto& s1 = m.mgr->stats();
    EXPECT_EQ(s1.cold_misses, 3u);
    EXPECT_EQ(s1.pool_hits, 2u);
    EXPECT_EQ(s1.h2d_bursts, 1u);

    // Rank 1: globals {4, 13, 21, 28} (chunks 1/3/5/7) — cold 4/13/21,
    // pool-hot 28.
    m.materialize_and_check(1, {4, 13, 21, 28});
    const auto& s2 = m.mgr->stats();
    EXPECT_EQ(s2.cold_misses, 6u);
    EXPECT_EQ(s2.pool_hits, 3u);

    // Repeat rank 0's selection: cold rows are row-cache hits now.
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 31, bts));
    m.materialize_and_check(0, {0, 9, 17, 24, 26});
    const auto& s3 = m.mgr->stats();
    EXPECT_EQ(s3.cold_misses, 6u) << "repeat must not refetch";
    EXPECT_EQ(s3.cache_hits, 3u);
}

// ── 9. TD-KVT-REPLICA-COLD-DEDUP: replicated KV, single cold copy ───────────

namespace {

/// Two-rank REPLICATED fixture on ONE physical GPU (both "ranks" share the
/// backend; positions 0/1 index device_backends).  Both pools hold the FULL
/// byte-identical replica (INV-KV-REP lockstep: same page_idx everywhere);
/// the block table is the same GLOBAL identity table on both ranks.
struct ReplicatedFixture {
    static constexpr int kLogical = 8;

    std::unique_ptr<lc::CudaSm120DeviceBackend> be;
    void* pools[2] = {nullptr, nullptr};
    void* stream = nullptr;
    std::unique_ptr<ld::KvTieringManager> mgr;
    std::vector<lm::PageHandle> handles;  // (logical j, layer l) at [j*L + l]
    std::vector<int> host_bt;             // identity GLOBAL table (both ranks)
    std::vector<std::pair<int, int>> freed;
    void* dev_indices = nullptr;
    void* dev_len = nullptr;
    // TD-KVT-SPEC-FORK re-promotion seam (INV-KV-REP lockstep: the same
    // fresh page_idx on BOTH replicas).
    int next_free_page = kLogical;
    int spare_ = 0;
    std::vector<std::pair<int, int>> alloced;  // (logical, phys)

    explicit ReplicatedFixture(int hot_slots, bool dedup,
                               int spare_pages = 0) {
        be = std::make_unique<lc::CudaSm120DeviceBackend>(ref0());
        be->set_device();
        stream = be->create_stream();
        const size_t bytes =
            static_cast<size_t>(kLogical + spare_pages) * kStrideBlock;
        std::vector<char> host(bytes, '\0');
        for (int p = 0; p < kLogical; ++p)
            for (int r = 0; r < kPageSize; ++r)
                fill_row(host.data() + p * kStrideBlock + r * kStrideRow,
                         0, p * kPageSize + r);
        for (int r = 0; r < 2; ++r) {
            pools[r] = be->device_alloc(bytes);
            be->memcpy_h2d(pools[r], host.data(), bytes);  // byte-identical
        }
        for (int j = 0; j < kLogical; ++j)
            for (int l = 0; l < kLayers; ++l)
                handles.push_back(lm::PageHandle{
                    .gpu_idx = 0, .page_idx = j,
                    .gpu_ptr = static_cast<char*>(pools[0])
                             + static_cast<int64_t>(j) * kStrideBlock,
                    .pool = lm::Pool::kMain});
        for (int j = 0; j < kLogical; ++j) host_bt.push_back(j);

        ld::KvTieringManager::Options o;
        o.dcp_size = 2;
        o.gpus = {cfg::GpuRef{.position = 0, .id = 0,
                              .type = cfg::GpuType::rtx5090},
                  cfg::GpuRef{.position = 1, .id = 0,
                              .type = cfg::GpuType::rtx5090}};
        o.device_backends = {be.get(), be.get()};
        o.stream_manager = nullptr;
        o.numa_manager = nullptr;
        o.free_page = [this](uint64_t, int layer, int logical,
                             const lm::PageHandle&) {
            freed.emplace_back(layer, logical);
        };
        // Re-promotion alloc seam: lockstep — the fresh page_idx is valid
        // against BOTH ranks' pool bases (INV-KV-REP), exactly what
        // PageAllocator::allocate_replicated guarantees the dispatcher.
        o.alloc_page = [this](uint64_t, int layer, int logical)
                -> std::optional<lm::PageHandle> {
            if (next_free_page >= kLogical + spare_) return std::nullopt;
            const int p = next_free_page++;
            lm::PageHandle h{
                .gpu_idx = 0, .page_idx = p,
                .gpu_ptr = static_cast<char*>(pools[0])
                         + static_cast<int64_t>(p) * kStrideBlock,
                .pool = lm::Pool::kMain};
            handles[static_cast<size_t>(logical) * kLayers + layer] = h;
            if (layer == 0) host_bt[static_cast<size_t>(logical)] = p;
            alloced.emplace_back(logical, p);
            return h;
        };
        spare_ = spare_pages;
        o.kv_main_bases = {pools[0], pools[1]};
        o.stride_block = kStrideBlock;
        o.stride_row = kStrideRow;
        o.page_size = kPageSize;
        o.kv_layers = kLayers;
        o.index_topk = kTopk;
        o.hot_buffer_slots = hot_slots;
        o.host_to_device_ratio = 8.0;
        o.kv_sharded = false;  // replicated
        o.replica_cold_dedup = dedup;
        mgr = std::make_unique<ld::KvTieringManager>(std::move(o));

        dev_indices = be->device_alloc(kTopk * sizeof(int));
        dev_len = be->device_alloc(sizeof(int));
    }
    ~ReplicatedFixture() {
        mgr.reset();
        be->set_device();
        be->device_free(dev_indices);
        be->device_free(dev_len);
        be->destroy_stream(stream);
        be->device_free(pools[0]);
        be->device_free(pools[1]);
    }

    /// Demote every eligible layer-0 page at `pos` and settle.
    void demote(uint32_t pos) {
        be->synchronize_device();
        const int* bts[2] = {host_bt.data(), host_bt.data()};
        ASSERT_TRUE(mgr->begin_layer(0, /*seq=*/1, pos, bts));
        mgr->after_attention(0, 1, pos, handles.data(), kLogical, kLayers);
        be->synchronize_device();
        mgr->drain_demotions();
    }

    /// Materialize `sel` (global positions) on `rank` and byte-compare the
    /// scratch against ground truth.
    void materialize_and_check(int rank, const std::vector<int>& sel) {
        std::vector<int> padded(kTopk, -1);
        for (size_t i = 0; i < sel.size(); ++i) padded[i] = sel[i];
        const int n = static_cast<int>(sel.size());
        be->memcpy_h2d(dev_indices, padded.data(), kTopk * sizeof(int));
        be->memcpy_h2d(dev_len, &n, sizeof(int));
        layerstorm::parallelism::TieredKvView tv{};
        const bool tiered = mgr->materialize(
            rank, /*layer=*/0, static_cast<const int*>(dev_indices),
            static_cast<const int*>(dev_len), /*batch=*/1, stream, &tv);
        ASSERT_TRUE(tiered) << "expected tiered path on rank " << rank;
        ASSERT_EQ(tv.seq_len_kv, n);
        be->synchronize_device();
        const size_t bytes = static_cast<size_t>(tv.max_blocks_per_seq)
                           * kStrideBlock;
        std::vector<char> out(bytes);
        be->memcpy_d2h_async(out.data(), tv.kv_cache, bytes, stream);
        be->synchronize_device();
        std::vector<char> want(kStrideRow);
        for (size_t i = 0; i < sel.size(); ++i) {
            fill_row(want.data(), 0, sel[i]);
            ASSERT_EQ(std::memcmp(
                out.data() + (i / kPageSize) * kStrideBlock
                    + (i % kPageSize) * kStrideRow,
                want.data(), kStrideRow), 0)
                << "rank " << rank << " materialized row " << i << " (pos "
                << sel[i] << ") != ground truth";
        }
    }
};

}  // namespace

TEST(KvTieringManagerReplicated, ColdDedupSingleOwnerCopyByteExact) {
    REQUIRES_GPU();
    // TD-KVT-REPLICA-COLD-DEDUP (INV-KVT-11): under replicated KV at dcp=2
    // each demoted page gets ONE cold copy, in the round-robin cold owner's
    // (j % 2) NUMA-local pool — the per-rank pool is HALVED (total pinned
    // host RAM unchanged vs the single-rank layout, divided by dcp vs the
    // per-rank-copy layout).  Every rank still materializes byte-exact,
    // including pages whose single cold copy lives in the OTHER rank's pool
    // (cross-pool host memcpy; the burst H2D stays rank-local).
    ReplicatedFixture m(/*hot_slots=*/8, /*dedup=*/true);
    ASSERT_EQ(m.mgr->cold_pool_capacity_pages(), 16)
        << "dedup pool per rank = ceil(32 / dcp)";
    m.demote(/*pos=*/31);  // pages 0..5 demote

    ASSERT_EQ(m.freed.size(), 6u) << "each page freed exactly once";
    EXPECT_EQ(m.mgr->stats().demoted_pages, 6u);
    // Round-robin owners: rank 0 stores pages 0/2/4, rank 1 stores 1/3/5.
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 3);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(1), 3);

    // Snapshot seam: byte-exact from the single owner copy.
    std::vector<char> want(kStrideBlock);
    for (int j = 0; j < 6; ++j) {
        const void* cold = m.mgr->cold_page_host_ptr(1, 0, j);
        ASSERT_NE(cold, nullptr);
        for (int r = 0; r < kPageSize; ++r)
            fill_row(want.data() + r * kStrideRow, 0, j * kPageSize + r);
        EXPECT_EQ(std::memcmp(cold, want.data(),
                              static_cast<size_t>(kStrideBlock)), 0)
            << "single cold copy of page " << j << " mismatch";
    }

    // Clobber the demoted pages in BOTH replicas: every correct byte must
    // come from the owners' cold copies.
    std::vector<char> junk(6 * kStrideBlock, '\xBB');
    m.be->memcpy_h2d(m.pools[0], junk.data(), junk.size());
    m.be->memcpy_h2d(m.pools[1], junk.data(), junk.size());

    // Rank 0: cold 0 (own pool), 5/23 (rank 1's pool), 9 (own); hot 24, 28.
    m.materialize_and_check(0, {0, 5, 9, 23, 24, 28});
    const auto& s1 = m.mgr->stats();
    EXPECT_EQ(s1.cold_misses, 4u);
    EXPECT_EQ(s1.pool_hits, 2u);
    EXPECT_EQ(s1.h2d_bursts, 1u);

    // Rank 1: cold 2 (rank 0's pool), 13 (own), 17 (rank 0's); hot 26.
    m.materialize_and_check(1, {2, 13, 17, 26});
    const auto& s2 = m.mgr->stats();
    EXPECT_EQ(s2.cold_misses, 7u);
    EXPECT_EQ(s2.pool_hits, 3u);
}

TEST(KvTieringManagerReplicated, ColdDedupRepromoteRestoresAllReplicasByteExact) {
    REQUIRES_GPU();
    // TD-KVT-SPEC-FORK re-promotion under replicated cold-dedup: every rank's
    // replica must be restored in lockstep at the same fresh page_idx
    // (INV-KV-REP).  For each page ONE rank holds the single cold copy
    // (round-robin owner, j % 2) and takes the direct node-local H2D; the
    // OTHER rank exercises the staged path (owner-pool host memcpy into its
    // own pinned staging, then a node-local H2D — INV-KVT-11).  Both
    // replicas must be byte-exact with the demoted VRAM pages clobbered.
    ReplicatedFixture m(/*hot_slots=*/8, /*dedup=*/true, /*spare_pages=*/8);
    m.demote(/*pos=*/31);  // pages 0..5 demote, single copy each
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 3);
    ASSERT_EQ(m.mgr->cold_pool_used_pages(1), 3);

    std::vector<char> junk(6 * kStrideBlock, '\xB7');
    m.be->memcpy_h2d(m.pools[0], junk.data(), junk.size());
    m.be->memcpy_h2d(m.pools[1], junk.data(), junk.size());

    ASSERT_TRUE(m.mgr->repromote_seq(1, /*keep_frontier=*/0));
    EXPECT_FALSE(m.mgr->has_demotions());
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(1), 0);
    EXPECT_EQ(m.mgr->stats().repromoted_pages, 6u);
    ASSERT_EQ(m.alloced.size(), 6u);

    m.be->synchronize_device();
    std::vector<char> want(kStrideBlock);
    std::vector<char> got(kStrideBlock);
    for (const auto& [logical, phys] : m.alloced) {
        for (int r = 0; r < kPageSize; ++r)
            fill_row(want.data() + r * kStrideRow, 0,
                     logical * kPageSize + r);
        for (int rank = 0; rank < 2; ++rank) {
            m.be->memcpy_d2h_async(
                got.data(),
                static_cast<char*>(m.pools[rank])
                    + static_cast<int64_t>(phys) * kStrideBlock,
                got.size(), m.stream);
            m.be->synchronize_device();
            EXPECT_EQ(std::memcmp(got.data(), want.data(),
                                  static_cast<size_t>(kStrideBlock)), 0)
                << "rank " << rank << " replica of re-promoted page "
                << logical << " mismatch (owner "
                << (logical % 2) << (rank == logical % 2
                        ? ", direct path)" : ", staged path)");
        }
    }
}

TEST(KvTieringManagerReplicated, NoDedupKeepsPerRankCopies) {
    REQUIRES_GPU();
    // replica_cold_dedup=false restores the Phase-4 behavior: every rank
    // demotes its OWN replica to its OWN pool (2× slots used, all-local
    // host memcpy on fetch), still byte-exact.
    ReplicatedFixture m(/*hot_slots=*/8, /*dedup=*/false);
    ASSERT_EQ(m.mgr->cold_pool_capacity_pages(), 32)
        << "non-dedup pool per rank = full per-layer × layers sizing";
    m.demote(/*pos=*/31);  // pages 0..5 demote

    ASSERT_EQ(m.freed.size(), 6u) << "each page freed exactly once";
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 6);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(1), 6);

    std::vector<char> junk(6 * kStrideBlock, '\xDD');
    m.be->memcpy_h2d(m.pools[0], junk.data(), junk.size());
    m.be->memcpy_h2d(m.pools[1], junk.data(), junk.size());
    m.materialize_and_check(0, {0, 5, 9, 23, 24, 28});
    m.materialize_and_check(1, {2, 13, 17, 26});
    const auto& s = m.mgr->stats();
    EXPECT_EQ(s.cold_misses, 7u);
    EXPECT_EQ(s.pool_hits, 3u);
}

// ── 10. S3 — tiering by slab (RADIX_SLAB_DESIGN §5 S3) ─────────────────────
//
// Position-major slabbed fixture: physical page for (logical j, layer l) is
// j * L + l (L = fixture layer count) — the S2 packing order (INV-SLAB-2: one slab holds ALL
// layers' pages for a contiguous token range).  pages_per_slab = 3, so a
// slab is 1.5 position-cohorts (cohorts straddle slab boundaries, as on the
// GLM champion where 79 layers meet a 103-page slab).  Covers:
//   * the in-step window sweep DEFERS per-layer candidates and flushes at
//     the last layer (or the next step's begin_layer) — demotion selection
//     stays page-precise, the D2H leaves as contiguous slab runs;
//   * whole-slab accounting (slabs_demoted_whole) and run coalescing;
//   * byte-exact cold copies for every layer (INV-KVT-1 unchanged);
//   * pending-batch interaction seams: seq_free discards, fork flushes
//     (slot sharing preserved), hibernate_seq batches all layers;
//   * promotion-side run coalescing (repromote_seq) byte-exact.

namespace {

struct SlabFixture {
    int L = 2;  // kv layers (runtime — flush triggers depend on it)
    PoolFixture fx;
    std::unique_ptr<ld::KvTieringManager> mgr;
    std::vector<lm::PageHandle> handles;  // (logical j, layer l) at [j*L + l]
    std::vector<int> host_bt;             // layer-0 logical→physical
    std::vector<std::pair<int, int>> freed;
    int num_logical = 0;
    int next_free_page = 0;

    static constexpr int kPps = 3;  // pages per slab

    explicit SlabFixture(int hot_slots, int num_logical_, int spare_pages = 0,
                         int layers = 2, const std::string& spill_dir = "",
                         int64_t spill_cap = 0,
                         double host_to_device_ratio = 8.0)
        : L(layers),
          fx(num_logical_ * layers + spare_pages, /*layer=*/0),
          num_logical(num_logical_),
          next_free_page(num_logical_ * layers) {
        (void)spill_dir; (void)spill_cap; (void)host_to_device_ratio;
        // Position-major ground truth: physical page j*L+l holds layer l's
        // rows for positions j*kPageSize.. (PoolFixture filled a layer-0
        // pattern; rewrite with the per-layer pattern).
        for (int j = 0; j < num_logical; ++j)
            for (int l = 0; l < L; ++l) {
                const int phys = j * L + l;
                for (int r = 0; r < kPageSize; ++r)
                    fill_row(fx.host_pool.data()
                                 + static_cast<int64_t>(phys) * kStrideBlock
                                 + static_cast<int64_t>(r) * kStrideRow,
                             l, j * kPageSize + r);
            }
        fx.be->memcpy_h2d(fx.pool, fx.host_pool.data(),
                          static_cast<size_t>(num_logical) * L
                              * kStrideBlock);
        for (int j = 0; j < num_logical; ++j)
            for (int l = 0; l < L; ++l) {
                const int phys = j * L + l;
                handles.push_back(lm::PageHandle{
                    .gpu_idx = 0, .page_idx = phys,
                    .gpu_ptr = static_cast<char*>(fx.pool)
                             + static_cast<int64_t>(phys) * kStrideBlock,
                    .pool = lm::Pool::kMain});
            }
        for (int j = 0; j < num_logical; ++j)
            host_bt.push_back(j * L);  // layer 0 view

        ld::KvTieringManager::Options o;
        o.dcp_size = 1;
        o.gpus = {ref0()};
        o.device_backends = {fx.be.get()};
        o.stream_manager = nullptr;
        o.numa_manager = nullptr;
        o.free_page = [this](uint64_t, int layer, int logical,
                             const lm::PageHandle&) {
            freed.emplace_back(layer, logical);
        };
        o.alloc_page = [this](uint64_t, int layer, int logical)
                -> std::optional<lm::PageHandle> {
            if (next_free_page >= fx.pool_pages) return std::nullopt;
            const int p = next_free_page++;
            lm::PageHandle h{
                .gpu_idx = 0, .page_idx = p,
                .gpu_ptr = static_cast<char*>(fx.pool)
                         + static_cast<int64_t>(p) * kStrideBlock,
                .pool = lm::Pool::kMain};
            handles[static_cast<size_t>(logical) * L + layer] = h;
            return h;
        };
        o.kv_main_bases = {fx.pool};
        o.stride_block = kStrideBlock;
        o.stride_row = kStrideRow;
        o.page_size = kPageSize;
        o.kv_layers = L;
        o.index_topk = kTopk;
        o.hot_buffer_slots = hot_slots;
        o.host_to_device_ratio = host_to_device_ratio;
        o.pages_per_slab = kPps;                        // S3 geometry
        o.slab_span_pages = {num_logical * L};    // spare pages = loose
        o.spill_dir = spill_dir;                  // TD-PREFIX-TIDY-COLD-SPILL
        o.spill_max_bytes = spill_cap;
        mgr = std::make_unique<ld::KvTieringManager>(std::move(o));
    }

    const int* bt() const { return host_bt.data(); }

    /// One full step at `pos`: begin_layer + after_attention for EVERY
    /// layer (the dispatcher shape — the last layer triggers the S3 flush).
    void step(uint32_t pos, uint64_t seq = 1) {
        const int* bts[1] = {bt()};
        for (int l = 0; l < L; ++l) {
            ASSERT_TRUE(mgr->begin_layer(l, seq, pos, bts));
            mgr->after_attention(l, seq, pos, handles.data() + l,
                                 num_logical, L);
        }
    }

    std::vector<char> read_page(int phys) {
        std::vector<char> out(kStrideBlock);
        fx.be->set_device();
        fx.be->synchronize_device();
        fx.be->memcpy_d2h_async(out.data(),
                                static_cast<char*>(fx.pool)
                                    + static_cast<int64_t>(phys)
                                          * kStrideBlock,
                                out.size(), fx.stream);
        fx.be->synchronize_device();
        return out;
    }

    /// Ground-truth bytes of (logical j, layer l).
    const char* truth(int j, int l) const {
        return fx.host_pool.data()
            + static_cast<int64_t>(j * L + l) * kStrideBlock;
    }
};

}  // namespace

TEST(KvTieringSlab, WindowSweepDefersToStepEndThenOneRunWholeSlabs) {
    REQUIRES_GPU();
    // 10 logical pages × 4 tokens; retention 8 tokens.  A step at pos 39:
    // demote_end = 32 → logical 0..7 eligible on BOTH layers = physical
    // pages 0..15 — one position-major contiguous run = 5 whole slabs
    // (pps 3) plus one page of slab 5.
    SlabFixture m(/*hot_slots=*/8, /*num_logical=*/10);
    m.fx.be->synchronize_device();
    const int* bts[1] = {m.bt()};

    // Layer 0's sweep must DEFER: no slots taken, nothing freed, pages
    // still hot (page-precise selection, transfer deferred to step end).
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 39, bts));
    m.mgr->after_attention(0, 1, 39, m.handles.data() + 0, m.num_logical,
                           m.L);
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    EXPECT_TRUE(m.freed.empty()) << "layer sweep must defer, not demote";
    EXPECT_FALSE(m.mgr->has_demotions());
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);

    // The last layer's sweep still only COLLECTS (its later superchunk
    // sub-chunks could be outstanding) — the batch flushes at the NEXT
    // step's layer-0 begin_layer.
    ASSERT_TRUE(m.mgr->begin_layer(1, 1, 39, bts));
    m.mgr->after_attention(1, 1, 39, m.handles.data() + 1, m.num_logical,
                           m.L);
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    EXPECT_TRUE(m.freed.empty()) << "no in-step flush";
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 40, bts));  // next step boundary
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    ASSERT_EQ(m.freed.size(), 16u) << "both layers' behind-window pages";
    const auto& s = m.mgr->stats();
    EXPECT_EQ(s.slab_flushes, 1u);
    EXPECT_EQ(s.slab_runs, 1u) << "position-major cohort = ONE contiguous "
                                  "D2H run";
    EXPECT_EQ(s.slab_pages, 16u);
    EXPECT_EQ(s.slabs_demoted_whole, 5u) << "physical 0..15 covers slabs "
                                            "0..4 whole (pps 3)";

    // Byte-exact cold copies for EVERY layer (INV-KVT-1).
    for (int j = 0; j < 8; ++j)
        for (int l = 0; l < m.L; ++l) {
            const void* c = m.mgr->cold_page_host_ptr(1, l, j);
            ASSERT_NE(c, nullptr) << "layer " << l << " logical " << j;
            EXPECT_EQ(std::memcmp(c, m.truth(j, l), kStrideBlock), 0)
                << "cold copy layer " << l << " logical " << j;
        }
    EXPECT_EQ(m.mgr->cold_page_host_ptr(1, 0, 8), nullptr) << "in-window";
    m.mgr->on_seq_free(1);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0) << "INV-KVT-17";
}

TEST(KvTieringSlab, SuperchunkLayer0StreamKeepsAccumulating) {
    REQUIRES_GPU();
    // The superchunk executor walks LAYER-OUTER over sub-chunks, so
    // consecutive layer-0 dispatches of one sequence are the SAME pass:
    // begin_layer(0) must NOT flush a batch whose newest candidates are
    // still layer 0's — the batch keeps accumulating and goes out with
    // the step's last layer as ONE run set.
    SlabFixture m(/*hot_slots=*/8, /*num_logical=*/10);
    m.fx.be->synchronize_device();
    const int* bts[1] = {m.bt()};
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 39, bts));
    m.mgr->after_attention(0, 1, 39, m.handles.data() + 0, m.num_logical,
                           m.L);
    // Next layer-0 sub-chunk: same pass — still no flush.
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 40, bts));
    EXPECT_TRUE(m.freed.empty()) << "layer-0 stream must keep accumulating";
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);
    m.mgr->after_attention(0, 1, 40, m.handles.data() + 0, m.num_logical,
                           m.L);
    // The pass reaches the last layer (collect only), then the NEXT
    // step's layer-0 begin flushes everything as ONE batch, ONE run.
    ASSERT_TRUE(m.mgr->begin_layer(1, 1, 40, bts));
    m.mgr->after_attention(1, 1, 40, m.handles.data() + 1, m.num_logical,
                           m.L);
    EXPECT_TRUE(m.freed.empty()) << "no in-step flush at the last layer";
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 41, bts));  // next step boundary
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    ASSERT_EQ(m.freed.size(), 16u) << "both layers' logical 0..7";
    EXPECT_EQ(m.mgr->stats().slab_flushes, 1u) << "ONE batch, not per "
                                                  "sub-chunk";
    EXPECT_EQ(m.mgr->stats().slab_runs, 1u);
    m.mgr->on_seq_free(1);
}

TEST(KvTieringSlab, MissedLastLayerFlushesAtNextStepBegin) {
    REQUIRES_GPU();
    // A step whose LAST kv layer never reports (skipped MTP shape): the
    // batch holds candidates from layers past 0, so the NEXT step's
    // layer-0 begin_layer is a step boundary and must flush it.
    SlabFixture m(/*hot_slots=*/8, /*num_logical=*/10, /*spare=*/0,
                  /*layers=*/3);
    m.fx.be->synchronize_device();
    const int* bts[1] = {m.bt()};
    for (int l = 0; l < 2; ++l) {  // layers 0 and 1; layer 2 (last) skipped
        ASSERT_TRUE(m.mgr->begin_layer(l, 1, 39, bts));
        m.mgr->after_attention(l, 1, 39, m.handles.data() + l,
                               m.num_logical, m.L);
    }
    EXPECT_TRUE(m.freed.empty());
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 40, bts));  // next step
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    ASSERT_EQ(m.freed.size(), 16u) << "layers 0+1, logical 0..7 each";
    EXPECT_TRUE(m.mgr->seq_has_demotions(1));
    m.mgr->on_seq_free(1);
}

TEST(KvTieringSlab, SeqFreeDiscardsPendingWithoutDemotion) {
    REQUIRES_GPU();
    SlabFixture m(/*hot_slots=*/8, /*num_logical=*/10);
    m.fx.be->synchronize_device();
    const int* bts[1] = {m.bt()};
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 39, bts));
    m.mgr->after_attention(0, 1, 39, m.handles.data() + 0, m.num_logical,
                           m.L);
    m.mgr->on_seq_free(1);  // dying sequence: pending batch must be dropped
    m.fx.be->synchronize_device();
    m.mgr->poll_demotions();
    EXPECT_TRUE(m.freed.empty()) << "no demotion may start for a dying seq";
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);
    EXPECT_FALSE(m.mgr->has_demotions());
    // The seq id can come back fresh.
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 3, bts));
}

TEST(KvTieringSlab, ForkFlushesPendingThenSharesRefcountedSlots) {
    REQUIRES_GPU();
    SlabFixture m(/*hot_slots=*/8, /*num_logical=*/10);
    m.fx.be->synchronize_device();
    const int* bts[1] = {m.bt()};
    ASSERT_TRUE(m.mgr->begin_layer(0, 1, 39, bts));
    m.mgr->after_attention(0, 1, 39, m.handles.data() + 0, m.num_logical,
                           m.L);
    EXPECT_TRUE(m.freed.empty());
    // Fork must flush the pending batch first (settled refcounted slots),
    // then share them with the child exactly as pre-S3.
    m.mgr->on_seq_fork(1, 2);
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    ASSERT_EQ(m.freed.size(), 8u);
    EXPECT_EQ(m.mgr->seq_cold_used_pages(1, 0), 8);
    EXPECT_EQ(m.mgr->seq_cold_used_pages(2, 0), 8) << "child shares slots";
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 8) << "refcount-shared, not "
                                                    "duplicated";
    m.mgr->on_seq_free(2);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 8) << "parent still holds";
    m.mgr->on_seq_free(1);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0) << "INV-KVT-17";
}

TEST(KvTieringSlab, HibernateSeqBatchesAllLayersIntoSlabRuns) {
    REQUIRES_GPU();
    // R3 holder hibernation through the S3 whole-sequence sweep: all
    // layers' pages below the frontier collect into ONE flush → one
    // contiguous run of 18 physical pages = 6 whole slabs; frontier page
    // stays hot on every layer (INV-KVT-4).
    SlabFixture m(/*hot_slots=*/8, /*num_logical=*/10);
    m.fx.be->synchronize_device();
    const int n = m.mgr->hibernate_seq(/*seq=*/5, m.handles.data(),
                                       m.num_logical);
    EXPECT_EQ(n, 18) << "all but the frontier page, BOTH layers";
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    ASSERT_EQ(m.freed.size(), 18u);
    const auto& s = m.mgr->stats();
    EXPECT_EQ(s.slab_flushes, 1u);
    EXPECT_EQ(s.slab_runs, 1u) << "one contiguous run for the whole holder";
    EXPECT_EQ(s.slabs_demoted_whole, 6u);
    for (int j = 0; j < m.num_logical - 1; ++j)
        for (int l = 0; l < m.L; ++l) {
            const void* c = m.mgr->cold_page_host_ptr(5, l, j);
            ASSERT_NE(c, nullptr);
            EXPECT_EQ(std::memcmp(c, m.truth(j, l), kStrideBlock), 0)
                << "cold copy layer " << l << " logical " << j;
        }
    EXPECT_EQ(m.mgr->cold_page_host_ptr(5, 0, m.num_logical - 1), nullptr);
    EXPECT_EQ(m.mgr->cold_page_host_ptr(5, 1, m.num_logical - 1), nullptr);
    // Idempotent (everything already cold).
    EXPECT_EQ(m.mgr->hibernate_seq(5, m.handles.data(), m.num_logical), 0);
    m.mgr->on_seq_free(5);
}

TEST(KvTieringSlab, RepromoteCoalescesRunsByteExact) {
    REQUIRES_GPU();
    // Promotion side: a hibernated holder's cold range re-promotes through
    // the alloc seam; the H2D must land byte-exact and coalesce into runs
    // (ascending dst pages + the slots the demote flush assigned in the
    // same order).
    SlabFixture m(/*hot_slots=*/8, /*num_logical=*/10, /*spare_pages=*/18);
    m.fx.be->synchronize_device();
    ASSERT_EQ(m.mgr->hibernate_seq(6, m.handles.data(), m.num_logical), 18);
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    // Clobber the demoted VRAM pages — the pinned cold copies are now the
    // only correct source.
    std::vector<char> junk(18 * kStrideBlock, '\x5A');
    m.fx.be->memcpy_h2d(m.fx.pool, junk.data(), junk.size());

    ASSERT_TRUE(m.mgr->repromote_seq(6, /*keep_frontier=*/0));
    const auto& s = m.mgr->stats();
    EXPECT_EQ(s.repromoted_pages, 18u);
    EXPECT_GE(s.promote_runs, 1u);
    EXPECT_LE(s.promote_runs, 3u) << "H2D must coalesce, not go per page";
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0) << "slots released";
    EXPECT_FALSE(m.mgr->seq_has_demotions(6));
    // Byte-exact restore at the freshly allocated physical pages.
    for (int j = 0; j < m.num_logical - 1; ++j)
        for (int l = 0; l < m.L; ++l) {
            const auto& h = m.handles[static_cast<size_t>(j) * m.L + l];
            const auto got = m.read_page(h.page_idx);
            EXPECT_EQ(std::memcmp(got.data(), m.truth(j, l), kStrideBlock),
                      0)
                << "repromoted bytes layer " << l << " logical " << j;
        }
    m.mgr->on_seq_free(6);
}

// ── 11. TD-PREFIX-TIDY-COLD-SPILL — the 2nd tiering hop ───────────────────

namespace {
std::string make_spill_dir() {
    char tmpl[] = "/tmp/ls_spill_test_XXXXXX";
    const char* d = mkdtemp(tmpl);
    EXPECT_NE(d, nullptr);
    return d ? d : "/tmp";
}
int spill_files_in(const std::string& dir) {
    int n = 0;
    if (DIR* dp = opendir(dir.c_str())) {
        while (dirent* e = readdir(dp)) {
            const std::string name = e->d_name;
            if (name.rfind("ls-spill-", 0) == 0) ++n;
        }
        closedir(dp);
    }
    return n;
}
}  // namespace

TEST(KvTieringSpill, SpillReleasesSlotsUnspillRestoresByteExact) {
    REQUIRES_GPU();
    const std::string dir = make_spill_dir();
    SlabFixture m(/*hot_slots=*/8, /*num_logical=*/10, /*spare=*/0,
                  /*layers=*/2, dir, /*cap=*/int64_t{1} << 30);
    m.fx.be->synchronize_device();
    ASSERT_EQ(m.mgr->hibernate_seq(7, m.handles.data(), m.num_logical), 18);
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 18);

    // Spill: every pinned slot returns, the bytes live in ONE file, the
    // pages stay "demoted" (fail-closed gates keep holding).
    EXPECT_EQ(m.mgr->spill_seq(7), 18);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0) << "slots must return";
    EXPECT_TRUE(m.mgr->seq_has_demotions(7));
    EXPECT_TRUE(m.mgr->seq_spilled(7));
    EXPECT_EQ(m.mgr->spilled_bytes_total(),
              static_cast<int64_t>(18) * kStrideBlock);
    EXPECT_EQ(spill_files_in(dir), 1);
    EXPECT_EQ(m.mgr->cold_page_host_ptr(7, 0, 0), nullptr)
        << "a spilled page is not COLD (host ptr must refuse)";
    // Idempotent.
    EXPECT_EQ(m.mgr->spill_seq(7), 0);

    // Unspill: fresh slots, byte-exact reload (the ONLY source is the
    // file), file deleted, cap accounting returns to zero.
    ASSERT_TRUE(m.mgr->unspill_seq(7));
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 18);
    EXPECT_FALSE(m.mgr->seq_spilled(7));
    EXPECT_EQ(m.mgr->spilled_bytes_total(), 0);
    EXPECT_EQ(spill_files_in(dir), 0) << "file must be unlinked";
    for (int j = 0; j < m.num_logical - 1; ++j)
        for (int l = 0; l < m.L; ++l) {
            const void* c = m.mgr->cold_page_host_ptr(7, l, j);
            ASSERT_NE(c, nullptr);
            EXPECT_EQ(std::memcmp(c, m.truth(j, l), kStrideBlock), 0)
                << "unspilled bytes layer " << l << " logical " << j;
        }
    m.mgr->on_seq_free(7);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);
    rmdir(dir.c_str());
}

TEST(KvTieringSpill, ByteCapRefusesBeforeWriting) {
    REQUIRES_GPU();
    const std::string dir = make_spill_dir();
    // Cap below the holder's 18-page size: the spill REFUSES with -1,
    // nothing is written, every slot stays.
    SlabFixture m(/*hot_slots=*/8, /*num_logical=*/10, /*spare=*/0,
                  /*layers=*/2, dir,
                  /*cap=*/static_cast<int64_t>(5) * kStrideBlock);
    m.fx.be->synchronize_device();
    ASSERT_EQ(m.mgr->hibernate_seq(8, m.handles.data(), m.num_logical), 18);
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    EXPECT_EQ(m.mgr->spill_seq(8), -1) << "cap must refuse BEFORE writing";
    EXPECT_EQ(spill_files_in(dir), 0);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 18) << "slots untouched";
    EXPECT_FALSE(m.mgr->seq_spilled(8));
    m.mgr->on_seq_free(8);
    rmdir(dir.c_str());
}

TEST(KvTieringSpill, ForkRequiresUnspillThenSharesSlots) {
    REQUIRES_GPU();
    const std::string dir = make_spill_dir();
    SlabFixture m(/*hot_slots=*/8, /*num_logical=*/10, /*spare=*/0,
                  /*layers=*/2, dir, /*cap=*/int64_t{1} << 30);
    m.fx.be->synchronize_device();
    ASSERT_EQ(m.mgr->hibernate_seq(9, m.handles.data(), m.num_logical), 18);
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    ASSERT_EQ(m.mgr->spill_seq(9), 18);
    // The dispatcher must unspill before forking — forking a spilled
    // holder directly is a contract violation and fails LOUD.
    EXPECT_THROW(m.mgr->on_seq_fork(9, 10), std::runtime_error);
    ASSERT_TRUE(m.mgr->unspill_seq(9));
    m.mgr->on_seq_fork(9, 10);
    EXPECT_EQ(m.mgr->seq_cold_used_pages(10, 0), 18) << "child shares";
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 18) << "refcounted";
    m.mgr->on_seq_free(10);
    m.mgr->on_seq_free(9);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);
    rmdir(dir.c_str());
}

TEST(KvTieringSpill, HolderEvictionDeletesTheSpillFile) {
    REQUIRES_GPU();
    const std::string dir = make_spill_dir();
    SlabFixture m(/*hot_slots=*/8, /*num_logical=*/10, /*spare=*/0,
                  /*layers=*/2, dir, /*cap=*/int64_t{1} << 30);
    m.fx.be->synchronize_device();
    ASSERT_EQ(m.mgr->hibernate_seq(11, m.handles.data(), m.num_logical),
              18);
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    ASSERT_EQ(m.mgr->spill_seq(11), 18);
    ASSERT_EQ(spill_files_in(dir), 1);
    // Holder eviction IS the spill-directory eviction (INV-KVT-17
    // extended to the disk hop).
    m.mgr->on_seq_free(11);
    EXPECT_EQ(spill_files_in(dir), 0);
    EXPECT_EQ(m.mgr->spilled_bytes_total(), 0);
    rmdir(dir.c_str());
}

TEST(KvTieringSpill, UnspillSlotExhaustionIsRetryable) {
    REQUIRES_GPU();
    const std::string dir = make_spill_dir();
    // ratio 1.0 → tiny cold pool: 2 pages/layer × 2 layers = 4 slots.
    // Holder A (4 eligible of 18 fit) spills; holder B then takes the
    // freed slots; A's unspill must fail RETRYABLE and succeed after B
    // is evicted.
    SlabFixture m(/*hot_slots=*/8, /*num_logical=*/10, /*spare=*/0,
                  /*layers=*/2, dir, /*cap=*/int64_t{1} << 30,
                  /*ratio=*/1.0);
    ASSERT_EQ(m.mgr->cold_pool_capacity_pages(), 4);
    m.fx.be->synchronize_device();
    const int a = m.mgr->hibernate_seq(20, m.handles.data(), m.num_logical);
    ASSERT_EQ(a, 4) << "pool-capacity fail-safe bounds the demotions";
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    ASSERT_EQ(m.mgr->spill_seq(20), 4);
    EXPECT_EQ(m.mgr->cold_pool_used_pages(0), 0);
    // Holder B takes the pool.
    ASSERT_EQ(m.mgr->hibernate_seq(21, m.handles.data(), m.num_logical), 4);
    m.fx.be->synchronize_device();
    m.mgr->drain_demotions();
    ASSERT_EQ(m.mgr->cold_pool_used_pages(0), 4);
    // A cannot reload — retryable false, file intact, partial state
    // consistent.
    EXPECT_FALSE(m.mgr->unspill_seq(20));
    EXPECT_TRUE(m.mgr->seq_spilled(20));
    EXPECT_EQ(spill_files_in(dir), 1);
    // The eviction seam frees B; the retry completes byte-exact.
    m.mgr->on_seq_free(21);
    ASSERT_TRUE(m.mgr->unspill_seq(20));
    EXPECT_EQ(spill_files_in(dir), 0);
    for (int l = 0; l < m.L; ++l)
        for (int j = 0; j < 2; ++j) {
            const void* c = m.mgr->cold_page_host_ptr(20, l, j);
            if (!c) continue;  // only 4 of 18 pages ever demoted
            EXPECT_EQ(std::memcmp(c, m.truth(j, l), kStrideBlock), 0);
        }
    m.mgr->on_seq_free(20);
    rmdir(dir.c_str());
}
