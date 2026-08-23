// GLM-25k: DSA-guided KV tiering manager ("HiSparse-style").
//
// Runs 1M-class contexts on a single GPU by keeping the indexer-K (the
// discoverability oracle) plus a HOT main-KV working set VRAM-resident and
// demoting COLD MLA-KV pages (cache_stride_row B/token) to NUMA-local pinned
// host RAM.  Before each tiered layer's sparse attention the manager fetches
// EXACTLY the indexer's ≤index_topk selected rows into a dense per-layer
// scratch (hot rows via device gather, cold rows via CPU-gather into pinned
// staging → ONE burst H2D), then hands the executor a fake paged view with
// identity indices — kernel-change-free, bit-identical to the non-tiered
// path (INV-KVT-1: tiering changes PLACEMENT, never SELECTION or values).
//
// Tiers per rank (replicated KV: each rank tiers its own replica; sharded
// KV: each rank tiers ITS token shard — see below):
//   1. Paged pool (VRAM)  — write tier: k_append always writes hot; pages
//      fully behind the retention window (hot_buffer_slots tokens) demote.
//   2. Hot row cache (VRAM) — per-layer LRU cache of hot_buffer_slots
//      previously-fetched cold rows (HiSparse device_buffer analogue);
//      consecutive-step top-k overlap makes steady-state fetches small deltas.
//   3. Cold pool (pinned host RAM, NUMA home node of the rank's GPU —
//      P-22 home-node rule; capacity = host_to_device_ratio × hot buffer).
//
// Phase-1 scope (GLM-25k): B==1 steps, replicated KV (dcp==1 or symmetric
// replicated dcp>=2), SnapMLA FP8 cache format, non-graph decode.
//
// TD-KVT-BATCH (partial, multi-sequence): tiering state is PER SEQUENCE —
// each sequence carries its own page-state table, demotion frontier and
// cold-slot accounting, so any number of sequences may be tiered
// concurrently as long as each STEP is a B==1 decode (interleaved
// single-sequence steps).  The per-(rank, layer) row cache is shared
// device memory keyed by (seq, position) with fair-share budgets: once a
// sequence holds >= hot_slots / n_seqs entries, its inserts evict its OWN
// LRU rows (or an over-share sequence's) before touching an under-share
// sequence's rows — placement/fairness only, never selection (INV-KVT-1).
// The COLD pool is likewise shared with a fair-share cap: with N demoting
// sequences, a sequence's per-rank cold slots are capped at capacity / N
// and further demotions are skipped (fail-safe, pages stay hot) — one
// sequence can never monopolize the pool.  Slots return in full at seq
// teardown (release_seq); other sequences' slots are never touched.
// A true B>1 cohort in ONE step remains fail-closed (TD-KVT-BATCH-COHORT:
// needs per-row fake views + batched materialization).
//
// TD-KVT-REPLICA-COLD-DEDUP (Options.replica_cold_dedup, replicated KV at
// dcp>=2): replicated ranks hold byte-identical KV pages (INV-KV-REP
// lockstep allocation + deterministic replicated k_append — the same
// equality the non-tiered snapshot path relies on when it reads rank 0's
// replica), so demoting one cold copy PER RANK duplicates host RAM
// tp_degree x.  Dedup keys each page's single cold copy on a round-robin
// COLD OWNER rank (global page % dcp): only the owner D2Hs (its own
// replica -> its own NUMA-local pool, fully local DMA), every rank still
// fences the free behind its attention-stream work (its replica page is
// freed too), and a non-owner rank's cold fetch CPU-gathers from the
// owner's pool into its OWN node-local pinned staging — the H2D DMA stays
// node-local; only the host memcpy may cross nodes (INV-KVT-11).
//
// TD-KVT-DCP-SHARDED (Options.kv_sharded, dcp>=2): sequence-sharded KV
// (INV-4.9e round-robin chunks).  Each global logical page is owned by
// exactly ONE rank (dcp_chunk_tokens is a multiple of page_size); demotion
// D2Hs the page from the OWNER's pool to the OWNER's NUMA-local cold pool
// only, and the selection the executor hands materialize() is the KVS-4
// translated rank-LOCAL index list — the manager maps each local index back
// to its global logical page (kv_shard_math) for the hot/cold classification
// while addressing the hot path through the rank's LOCAL block table.  The
// fake tiered view feeds the same QAG combine as the non-tiered sharded
// path (placement-only under it too, INV-KVT-9).
//
// TD-KVT-SPEC (resolved): CMD_SEQ_SNAPSHOT of a tiered sequence captures
// BOTH tiers — the dispatcher drains in-flight demotions
// (drain_demotions()) then reads each demoted page's bytes from the cold
// pool (cold_page_host_ptr(); rank 0's replica, or the shard owner's copy
// under sharded KV) — byte-identical to a non-tiered snapshot.
//
// TD-KVT-SPEC-FORK (narrowed; residual = Phase-12 drafts/graph-replay +
// TD-KVT-BATCH-COHORT): cold slots are REFCOUNTED per (rank, slot), so
// seq_fork of a demoted parent shares the parent's cold pages with the
// child (on_seq_fork: the child inherits the page states / demoted
// frontier and add-refs every cold slot — no cold-byte copy; a slot
// returns to the pool only when its LAST holder releases it).  Cold-page
// RE-PROMOTION (repromote_seq) lifts rewind-into-demoted-territory and
// restore-ONTO-a-demoted-sequence: each affected cold page gets a fresh
// VRAM page via the dispatcher's standard growth path (alloc_page seam —
// INV-KV-REP lockstep under replicated KV, owner routing under sharded),
// ONE batched H2D per holding rank restores the EXACT demoted bytes
// (node-local DMA, INV-KVT-3/-11; a dedup non-owner stages the owner's
// copy through its own node-local pinned staging), the dispatcher handle
// is un-neutralized and the kv-meta dirty guard poisoned so block tables
// re-upload, and the slots are released respecting refcounts — a forked
// child re-promoting shared territory splits copy-on-write, leaving the
// other holders' cold copies untouched.  A rewind is legal without
// re-promotion only while NO cold page holds a position >= the step's
// write position token_pos (the step's k_append rewrites token_pos, so
// demoted_frontier <= token_pos); the dispatcher's repromote_for_rewind
// hook runs BEFORE the kv-meta build so the step's block tables / slot
// mappings carry the fresh handles.  VRAM-full keeps re-promotion
// fail-closed (capacity, not correctness).  Drafts / graph replay on a
// demoted sequence remain fail-closed (Phase-12 speculation).
//
// Phase-2 (TD-KVT-SYNC / TD-KVT-PREFETCH / TD-KVT-H2D-CONTENTION):
//   * Overlapped selection readback — the executor calls prepare() right
//     after the top-k is enqueued; the D2H lands while the executor enqueues
//     the projection/staging work, so materialize()'s host wait is ~0.
//   * IndexShare reuse — a SHARED layer's selection is byte-identical to the
//     preceding FULL layer's (selection_fresh == false); the manager reuses
//     its host copy and skips the D2H + wait entirely (~3/4 of layers).
//   * IndexShare lookahead prefetch — during a FULL layer's materialize the
//     selection is already known for its shared successors; their cold
//     misses are CPU-gathered + burst-H2D + scattered into their row caches
//     on the tiering H2D stream, overlapping this layer's attention/MoE.
//     Prefetch is placement-only: rows land in the row cache keyed by
//     position; the successor's own classification decides what is read
//     (INV-KVT-1 holds regardless of prediction quality).
//   * Dedicated per-rank tiering H2D stream — cold bursts never queue behind
//     ~18 MB expert-weight copies on kH2dTransfer.
//   Temporal warm-start (ticket lever 3b) is realized by the row cache
//   itself: rows fetched for step t are cache hits at step t+1.
// TD-KVT-PREFILL (resolved): a blessed B==1 SPARSE prefill chunk
// (compute.dsa_sparse_prefill, replicated KV) is a tierable step — its
// per-row causal top-k consumption is decode-shaped, so begin_layer/
// materialize/after_attention cover it unchanged: long/chunked prefill
// demotes behind the chunk frontier and prefill INTO a sequence with cold
// pages works (cold rows come from the pinned pool via materialize — no
// re-promotion).  TD-KVT-PREFILL-REPROMOTE (resolved): a NON-TIERABLE
// prefill shape on a demoted sequence (dense prefill chunk — sparse gate
// off / coverage-dead / sharded-KV dense fallback — or a B>1 chunk
// cohort) no longer fails closed: the dispatcher lifts it by FULL
// re-promotion (repromote_seq(seq, 0)) at the post-tierability gate and
// rebuilds the kv metadata in place from the fresh handles; allocation
// failure keeps the step fail-closed (capacity, not correctness).
// Deferred: B>1 TIERED cohorts in one step (TD-KVT-BATCH-COHORT: per-row
// fake views), drafts / graph replay on a demoted sequence
// (TD-KVT-SPEC-DRAFT Phase-12 residual: fail-closed guard in the
// dispatcher).
//
// Threading: daemon thread only (INV-3.4.2).  CUDA-free TU (INV-GPU-1):
// device work goes through DeviceBackend + the kv_row_copy launch wrappers.

#pragma once

#include "core/gpu_ref.h"
#include "core/memory/numa_manager.h"
#include "core/memory/page_allocator.h"
#include "parallelism/kv_tiering_hook.h"

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace layerstorm::compute {
class DeviceBackend;
class StreamManager;
}  // namespace layerstorm::compute

namespace layerstorm::daemon {

class KvTieringManager final : public parallelism::KvTieringHook {
public:
    struct Options {
        int dcp_size = 1;
        /// TP GPUs in rank order (INV-4.18 positions).
        std::vector<config::GpuRef> gpus;
        /// Per GPU POSITION (engine layout); indexed by gpus[r].position.
        std::vector<compute::DeviceBackend*> device_backends;
        compute::StreamManager* stream_manager = nullptr;  ///< nullable (tests)
        memory::NumaManager* numa_manager = nullptr;       ///< nullable → plain pinned
        /// Returns a demoted page to the allocator (replicated handles mirror
        /// the free across all TP GPUs) AND neutralizes the owner's stored
        /// handle (seq_pages_ entry) so later bulk frees skip it.  Called
        /// once per demoted page after every rank's D2H completed.
        std::function<void(uint64_t seq_id, int layer, int logical,
                           const memory::PageHandle&)> free_page;
        /// TD-KVT-SPEC-FORK (re-promotion): allocate a fresh kMain VRAM page
        /// for (seq, layer, logical) via the dispatcher's standard growth
        /// path (allocate_replicated lockstep under replicated KV — the
        /// returned page_idx is valid against EVERY rank's kv_main base,
        /// INV-KV-REP; owner routing under sharded KV), write the handle
        /// back into the dispatcher's seq_pages_ (un-neutralize page_idx)
        /// and poison the kv-meta dirty guard so block tables re-upload.
        /// nullopt on pool exhaustion — re-promotion then fails CLOSED
        /// (capacity, not correctness).  Nullable: without the seam every
        /// re-promotion path stays fail-closed.
        std::function<std::optional<memory::PageHandle>(
            uint64_t seq_id, int layer, int logical)> alloc_page;
        /// kv_main pool base per RANK.
        std::vector<void*> kv_main_bases;

        int64_t stride_block = 0;  ///< bytes per physical page
        int stride_row = 0;        ///< bytes per token row (self-contained)
        int page_size = 16;        ///< tokens per page
        int kv_layers = 1;         ///< layers incl. MTP
        int index_topk = 2048;

        int hot_buffer_slots = 0;        ///< 0 = auto (2 × index_topk)
        double host_to_device_ratio = 8.0;

        /// TD-KVT-DCP-SHARDED: sequence-sharded KV (hardware.dcp_kv_mode =
        /// sharded, effective at dcp_size >= 2).  Each rank owns round-robin
        /// token chunks (INV-4.9e); demotion targets the OWNER rank only and
        /// materialize() consumes the KVS-4 translated rank-LOCAL selection.
        bool kv_sharded = false;
        /// Tokens per ownership chunk (memory.kv_cache.dcp_chunk_size).
        /// Required under kv_sharded; must be a multiple of page_size.
        int dcp_chunk_tokens = 0;

        /// TD-KVT-REPLICA-COLD-DEDUP: under REPLICATED KV at dcp >= 2 keep a
        /// single cold copy per page (round-robin cold-owner rank) instead
        /// of one per rank — halves (1/dcp) the pinned cold RAM; the
        /// per-rank pool shrinks by the same factor so total capacity is
        /// unchanged.  Placement-only (INV-KVT-11).  Ignored under sharded
        /// KV (already single-copy, INV-KVT-9) and at dcp == 1.
        bool replica_cold_dedup = true;

        /// IndexShare full-layer mask, EXECUTOR semantics (DcpExecutor::
        /// Options::indexer_full_layers): layer l is FULL iff the mask is
        /// empty OR (l < size && mask[l]); layers beyond the mask (MTP) are
        /// SHARED.  Drives the IndexShare reuse skip + lookahead prefetch
        /// (TD-KVT-SYNC / TD-KVT-PREFETCH).  Empty = no sharing = both off.
        std::vector<uint8_t> indexer_full_layers;
    };

    explicit KvTieringManager(Options opts);
    ~KvTieringManager() override;

    KvTieringManager(const KvTieringManager&) = delete;
    KvTieringManager& operator=(const KvTieringManager&) = delete;

    // ── Dispatcher API (per attention dispatch) ────────────────────────────

    /// Stage the per-layer step context before execute_attention.  B==1
    /// steps only: decode steps and (TD-KVT-PREFILL) blessed B==1 sparse
    /// prefill chunks — both are single-row selections to the manager.
    /// host_block_tables: per-RANK host pointer to this layer's block-table
    /// row (batch row 0).  Also polls in-flight demotions.  Per-sequence
    /// state is created on first sight (TD-KVT-BATCH: any number of
    /// sequences may be tiered concurrently; each step is still
    /// single-sequence).  Returns false when tiering must not engage this
    /// step (bad layer) — the caller then must NOT pass the hook to the
    /// executor.
    bool begin_layer(int layer, uint64_t seq_id, uint32_t token_pos,
                     const int* const* host_block_tables);

    /// Demote this layer's pages that fell fully behind the retention window
    /// (background D2H per rank on the D2H stream; page freed to the
    /// allocator once every rank's copy completed).  `pages` points at the
    /// sequence's handle for (logical 0, layer); consecutive logical pages
    /// are `handle_stride` handles apart (dispatcher layout: pages[j*L + l]).
    void after_attention(int layer, uint64_t seq_id, uint32_t token_pos,
                         const memory::PageHandle* pages, int num_logical,
                         int handle_stride);

    /// Poll in-flight demotion D2H copies; completed pages flip to COLD and
    /// their device pages return to the allocator.
    void poll_demotions();

    /// Block until every in-flight demotion completed (spin + yield).
    /// Throws on device error or timeout.  Used by sequence teardown and by
    /// CMD_SEQ_SNAPSHOT of a tiered sequence (TD-KVT-SPEC) so every demoted
    /// page is either HOT (valid handle) or COLD (host copy complete).
    void drain_demotions();

    /// TD-KVT-SPEC: host pointer to a COLD (fully demoted) page's bytes in
    /// the pinned cold pool — the round-robin cold owner's single copy
    /// under replicated dedup (rank 0's replica without dedup), the shard
    /// OWNER's copy under sharded KV (content identical to the freed VRAM
    /// page, INV-KVT-1).  nullptr when the page is not COLD.  Call
    /// drain_demotions() first to settle kD2hInflight pages.
    const void* cold_page_host_ptr(uint64_t seq_id, int layer,
                                   int logical) const;

    /// True once any page has been demoted (or is in flight) for ANY
    /// sequence — gates fork/snapshot/draft/prefill fail-closed paths.
    bool has_demotions() const { return total_demoted_or_inflight_ > 0; }

    /// has_demotions() scoped to one sequence (TD-KVT-BATCH per-seq state).
    bool seq_has_demotions(uint64_t seq_id) const;

    /// Sequence teardown: drain that sequence's in-flight demotions, return
    /// its cold slots and row-cache entries, drop its state.  Other
    /// sequences' tiering state is untouched (TD-KVT-BATCH isolation).
    /// Cold slots shared with a fork family (refcounted) return to the pool
    /// only when the LAST holder releases them (TD-KVT-SPEC-FORK).
    void on_seq_free(uint64_t seq_id);

    /// TD-KVT-SPEC-FORK: seq_fork interop.  Drains the parent's in-flight
    /// demotions, then gives the child a copy of the parent's tiering state
    /// (page states, demoted frontier, per-rank cold accounting) with every
    /// COLD slot REFCOUNT-shared — no cold-byte copy at fork time; a shared
    /// slot returns to the pool only when its last holder releases it
    /// (release_seq / repromote_seq).  No-op when the parent has no
    /// demotions (the child starts fresh on first sight).
    void on_seq_fork(uint64_t src_seq_id, uint64_t dst_seq_id);

    /// TD-KVT-SPEC-FORK / TD-KVT-PREFILL-REPROMOTE machinery: re-promote
    /// every COLD page of `seq_id` holding any position >= keep_frontier
    /// back to VRAM: drain in-flight demotions; per page allocate a fresh
    /// VRAM page through the alloc_page seam (INV-KV-REP lockstep under
    /// replicated KV / owner routing under sharded — the dispatcher handle
    /// is un-neutralized and the kv-meta dirty guard poisoned); batched H2D
    /// per holding rank restores the EXACT demoted bytes (INV-KVT-1;
    /// node-local DMA per INV-KVT-3/-11 — a dedup non-owner stages the
    /// owner's copy through its own node-local pinned staging); release the
    /// cold slots respecting fork-family refcounts (a shared slot survives
    /// for the other holders — copy-on-write split) and lower the demoted
    /// frontier.  keep_frontier == 0 re-promotes everything (seq_restore
    /// re-init).  Returns false when the alloc_page seam is missing or VRAM
    /// is exhausted (partial progress is kept consistent; the caller must
    /// fail the operation CLOSED — capacity, not correctness, INV-KVT-2).
    bool repromote_seq(uint64_t seq_id, uint32_t keep_frontier);

    /// Dispatcher pre-step hook (BEFORE build_kv_metadata): a step writing
    /// at `token_pos` needs every position >= token_pos hot (its k_append
    /// rewrites token_pos through the slot mapping).  No-op unless a cold
    /// page reaches token_pos (demoted_frontier > token_pos); otherwise
    /// delegates to repromote_seq(seq_id, token_pos).  Returns false on
    /// re-promotion failure (the step must fail closed).
    bool repromote_for_rewind(uint64_t seq_id, uint32_t token_pos);

    // ── KvTieringHook (executor seam) ─────────────────────────────────────

    /// TD-KVT-SYNC: overlapped selection readback.  On a FULL layer (fresh
    /// selection) with cold pages in this layer or its IndexShare
    /// successors, enqueue the async D2H of the selection now (ordered after
    /// the just-enqueued top-k on `stream`); on a SHARED layer whose
    /// selection this manager already holds for this step, bless host-copy
    /// reuse instead (no D2H at all).  Pure hint (no-op on any mismatch);
    /// materialize() falls back to the synchronous readback when unprepared.
    void prepare(int rank, int layer_idx,
                 const int* sparse_indices_dev,
                 const int* topk_lengths_dev,
                 int batch_size, void* stream,
                 bool selection_fresh) override;

    bool materialize(int rank, int layer_idx,
                     const int* sparse_indices_dev,
                     const int* topk_lengths_dev,
                     int batch_size, void* stream,
                     parallelism::TieredKvView* out) override;

    void on_dense_layer(int layer_idx) override;

    // ── Introspection (tests, logging) ────────────────────────────────────

    struct Stats {
        uint64_t materializations = 0;  ///< tiered (rank, layer) gathers
        uint64_t rows_gathered = 0;     ///< total rows materialized
        uint64_t pool_hits = 0;         ///< rows sourced from the paged pool
        uint64_t cache_hits = 0;        ///< rows sourced from the hot row cache
        uint64_t cold_misses = 0;       ///< rows fetched from the cold host pool
        uint64_t cold_fetch_bytes = 0;  ///< H2D bytes for cold fetches
        uint64_t h2d_bursts = 0;        ///< cold burst H2D count
        uint64_t demoted_pages = 0;     ///< pages demoted (completed)
        uint64_t cache_evictions = 0;   ///< hot row-cache LRU evictions
        uint64_t budget_skips = 0;      ///< demotions skipped by the per-seq
                                        ///< cold fair-share cap (TD-KVT-BATCH)
        // TD-KVT-SYNC / TD-KVT-PREFETCH
        uint64_t prepares = 0;          ///< prepare() readbacks issued
        uint64_t sync_reuses = 0;       ///< materializations with NO D2H
                                        ///< (IndexShare host-copy reuse)
        uint64_t sync_overlapped = 0;   ///< materializations consuming a
                                        ///< prepare()-issued readback
        uint64_t sync_fallbacks = 0;    ///< legacy synchronous readbacks
        uint64_t sync_wait_us = 0;      ///< host µs blocked on the selection
                                        ///< readback (overlapped + fallback)
        uint64_t guard_wait_us = 0;     ///< host µs in staging-reuse guards
        uint64_t prefetch_rows = 0;     ///< rows prefetched into successor
                                        ///< layers' row caches
        uint64_t prefetch_bursts = 0;   ///< lookahead H2D bursts
        uint64_t prefetch_bytes = 0;    ///< lookahead H2D bytes
        uint64_t prefetch_hits = 0;     ///< cache hits served by a prefetch
        // TD-KVT-SPEC-FORK (re-promotion)
        uint64_t repromoted_pages = 0;  ///< cold pages re-promoted to VRAM
        uint64_t repromote_bytes = 0;   ///< H2D bytes across holding ranks
    };
    const Stats& stats() const { return stats_; }

    /// NUMA node the rank's cold pool / staging were bound to (-1 = unbound
    /// fallback).  Test seam for the P-22 home-node rule.
    int cold_pool_numa_node(int rank) const;

    /// Row-cache entries currently mapped for (rank, layer), all sequences.
    /// Test seam.
    int cache_entries(int rank, int layer) const;

    /// Row-cache entries for one sequence (TD-KVT-BATCH budget seam).
    int cache_entries_seq(int rank, int layer, uint64_t seq_id) const;

    /// True if any page of `layer` is COLD (completed demotion) for `seq`.
    bool layer_has_cold(uint64_t seq_id, int layer) const;

    /// Cold pool slots currently in use / capacity per rank.  Test seams
    /// (TD-KVT-REPLICA-COLD-DEDUP sizing + leak checks).
    int cold_pool_used_pages(int rank) const;
    int cold_pool_capacity_pages() const { return cold_pool_pages_; }

    /// Cold pool slots `seq_id` holds in `rank`'s pool (TD-KVT-BATCH
    /// fair-share budget seam).
    int seq_cold_used_pages(uint64_t seq_id, int rank) const;

    int hot_buffer_slots() const { return hot_slots_; }

    /// One-line stats log (rate-limited by the caller).
    void log_stats() const;

private:
    enum class PageState : uint8_t { kHot = 0, kD2hInflight = 1, kCold = 2 };

    struct PageInfo {
        PageState state = PageState::kHot;
        std::vector<int> cold_slot;  ///< per-rank cold pool slot (-1 = none;
                                     ///< exactly one >= 0 under dedup/
                                     ///< sharded).  Slots are REFCOUNTED
                                     ///< (RankBufs::cold_ref) — a fork
                                     ///< family shares a demoted parent's
                                     ///< slots (TD-KVT-SPEC-FORK).
    };

    /// TD-KVT-BATCH: all per-sequence tiering state.
    struct SeqState {
        std::vector<std::vector<PageInfo>> pages;  ///< [layer][logical]
        uint32_t max_pos_seen = 0;
        /// First position AFTER the highest demoted (cold or in-flight)
        /// page: max over demoted pages of (logical+1)*page_size.  A step
        /// at token_pos (its k_append rewrites token_pos) is legal while
        /// demoted_frontier <= token_pos (cold content stays immutable
        /// prefix); a rewind below that needs cold-page re-promotion first
        /// (repromote_for_rewind, TD-KVT-SPEC-FORK) — reaching begin_layer
        /// with frontier > token_pos throws (INV-KVT-2 fail-loud).
        uint32_t demoted_frontier = 0;
        int demoted_or_inflight = 0;  ///< cold + in-flight page count
        int inflight = 0;             ///< in-flight demotion page count
        /// Cold pool slots held per rank — fair-share budget input: with
        /// N demoting sequences a sequence's per-rank cold usage is capped
        /// at capacity / N; further demotions are skipped (fail-safe, pages
        /// stay hot) so one sequence cannot monopolize the shared pool.
        /// A single demoting sequence keeps the full pool.
        std::vector<int> cold_used;
    };

    /// Shared per-(rank, layer) device row-cache slab, entries keyed by
    /// (seq, position) with per-seq fair-share budgets (TD-KVT-BATCH).
    struct LayerCache {
        /// seq → (position → slot).  Outer map size = sequences holding
        /// entries; inner size = the sequence's slot count (budget input).
        std::unordered_map<uint64_t, std::unordered_map<int, int>> map;
        std::vector<int> slot_pos;          ///< slot → position (-1 = free)
        std::vector<uint64_t> slot_seq;     ///< slot → owning sequence
        std::vector<uint64_t> slot_use;     ///< slot → LRU tick
        std::vector<uint8_t> slot_prefetched;  ///< inserted by prefetch,
                                               ///< not yet hit (stats only)
        int filled = 0;
        std::vector<int> free_slots;  ///< slots released by seq teardown
        // Lazy per-tick LRU eviction order (acquire_cache_slot).
        std::vector<std::pair<uint64_t, int>> evict_scratch;
        size_t evict_idx = 0;
        uint64_t evict_tick = 0;
        /// Under-share other-seq slots skipped by the fair-share walk this
        /// tick (oldest first) — the last-resort eviction source.
        std::vector<int> evict_deferred;
        size_t deferred_idx = 0;
    };

    /// One after_attention call's batch of demoting pages: all D2H copies
    /// were issued together, so one completion event per rank covers them.
    struct InflightDemotion {
        uint64_t seq = 0;
        std::vector<memory::PageHandle> handles;
        std::vector<std::pair<int, int>> pages_ll;  ///< (layer, logical)
        std::vector<void*> events;  ///< per-rank completion events (D2H on
                                    ///< storing ranks; attention-order fence
                                    ///< on non-storing ranks under dedup)
    };

    struct RankBufs {
        // Device
        void* scratch = nullptr;        ///< fake pages: n_fake_pages × stride_block
        void* row_cache = nullptr;      ///< kv_layers × hot_slots × stride_row
        void* cold_incoming = nullptr;  ///< index_topk × stride_row (packed misses)
        void* dev_src_ptrs = nullptr;   ///< index_topk × void*
        void* dev_scatter_ptrs = nullptr;   ///< index_topk × void*
        void* dev_scatter_idx = nullptr;    ///< index_topk × int
        void* dev_ident_indices = nullptr;  ///< index_topk × int iota
        void* dev_fake_bt = nullptr;        ///< n_fake_pages × int iota
        // Device — IndexShare lookahead prefetch (TD-KVT-PREFETCH); separate
        // from the materialize buffers: a prefetch burst may be in flight
        // while the next materialize reuses cold_incoming/h_stage.
        void* pf_incoming = nullptr;        ///< index_topk × stride_row
        void* dev_pf_scatter_ptrs = nullptr;  ///< index_topk × void*
        void* dev_pf_scatter_idx = nullptr;   ///< index_topk × int
        // Pinned host (NUMA home node)
        memory::NumaBuffer host_arena{};    ///< staging + tables + cold pool
        bool host_registered = false;
        int numa_node = -1;
        char* h_stage = nullptr;            ///< index_topk × stride_row
        const void** h_src_ptrs = nullptr;  ///< index_topk
        const void** h_scatter_ptrs = nullptr;
        int* h_scatter_idx = nullptr;
        int* h_indices = nullptr;           ///< index_topk
        int* h_topk_len = nullptr;          ///< 1
        char* h_pf_stage = nullptr;         ///< index_topk × stride_row
        const void** h_pf_scatter_ptrs = nullptr;  ///< index_topk
        int* h_pf_scatter_idx = nullptr;    ///< index_topk
        char* cold_base = nullptr;          ///< cold_pool_pages × stride_block
        // Streams / events.  h2d_stream is ALWAYS owned (TD-KVT-H2D-
        // CONTENTION: a dedicated per-rank tiering H2D stream so cold bursts
        // never queue behind ~18 MB expert-weight copies on kH2dTransfer);
        // d2h_stream is the shared kD2hTransfer when a StreamManager exists.
        void* h2d_stream = nullptr;
        void* d2h_stream = nullptr;
        bool owns_d2h_stream = false;
        void* ev_sync = nullptr;     ///< host-wait event (indices D2H)
        void* ev_h2d = nullptr;      ///< cold burst completion
        void* ev_attn_order = nullptr;  ///< attention→D2H ordering
        void* ev_pf = nullptr;          ///< prefetch burst+scatter completion
        void* ev_pf_order = nullptr;    ///< attention→prefetch-write ordering
        void* ev_mat_done = nullptr;    ///< recorded after the step-5 gather
                                        ///< (host staging-reuse guard)
        bool pf_inflight = false;       ///< ev_pf recorded, h_pf_* in use
        bool mat_inflight = false;      ///< ev_mat_done recorded, staging in use
        std::vector<int> cold_free;  ///< free cold pool slots
        /// Per-slot holder count (TD-KVT-SPEC-FORK): 0 = free, 1 = single
        /// owner, >1 = shared across a fork family.  A slot returns to
        /// cold_free only when the count reaches 0 (release_cold_slot).
        std::vector<uint32_t> cold_ref;
    };

    /// TD-KVT-SYNC: per-rank host-selection state across prepare/materialize.
    struct SelCtx {
        bool pending = false;    ///< prepare() D2H issued, not yet consumed
        bool valid = false;      ///< h_indices/h_topk_len hold (seq, pos)'s
                                 ///< selection (n rows)
        bool reuse_ok = false;   ///< prepare() blessed host-copy reuse for
                                 ///< prepared_layer (IndexShare shared layer)
        bool fresh = false;      ///< last prepare's selection_fresh (gates
                                 ///< the lookahead prefetch)
        int prepared_layer = -1;
        uint64_t seq = 0;
        uint32_t pos = 0;
        int n = 0;
    };

    compute::DeviceBackend* backend(int rank) const;
    void* attn_stream(int rank) const;
    void wait_event(compute::DeviceBackend* be, void* ev) const;
    /// wait_event + host-blocked time in µs.
    uint64_t wait_event_us(compute::DeviceBackend* be, void* ev) const;
    char* cache_slot_addr(int rank, int layer, int slot) const;
    /// Pick a row-cache slot for `seq`'s new entry: released/free slots
    /// first, else fair-share LRU eviction among slots not touched at
    /// `tick` — the requesting sequence's own rows and over-share
    /// sequences' rows evict before an under-share sequence's (per-seq
    /// budget = hot_slots / sequences holding entries).  -1 = none
    /// available.  The lazy eviction order is cached per tick.
    int acquire_cache_slot(LayerCache& lc, uint64_t tick, uint64_t seq);
    /// Evict slot `s` of `lc` (erase its (seq, position) mapping).
    void evict_slot(LayerCache& lc, int s);
    /// Consume/produce the host copy of the selection for (rank, layer):
    /// prepared readback (wait), IndexShare host-copy reuse (no D2H), or the
    /// legacy synchronous fallback.  Returns n (0 = degenerate selection).
    int ensure_selection(int rank, int layer_idx,
                         const int* sparse_indices_dev,
                         const int* topk_lengths_dev, void* stream);
    /// GLOBAL logical page a selection index maps to: idx/page_size under
    /// replicated KV (indices are global positions); under sharded KV the
    /// index is a rank-LOCAL slot — invert the round-robin chunk layout
    /// (kv_shard_math::global_page_of_local).  Within-page row offsets are
    /// identical in global and local numbering (chunk % page_size == 0).
    int global_page_of_index(int rank, int sel_idx) const;
    /// Rank holding global page j's single cold copy: the shard owner under
    /// sharded KV, the round-robin cold owner (j % dcp) under replicated
    /// dedup (INV-KVT-11), -1 = every rank holds one (no dedup).
    int cold_owner_rank(int global_page) const;
    /// Rank whose pinned pool `rank` must read page j's cold bytes from
    /// (== rank without dedup; the cold owner under dedup — a cross-node
    /// host memcpy when their home nodes differ; sharded selections only
    /// ever map to the rank's own pages).
    int cold_src_rank(int rank, int global_page) const {
        const int o = cold_owner_rank(global_page);
        return (o >= 0 && !sharded_) ? o : rank;
    }
    SeqState* find_seq(uint64_t seq_id);
    const SeqState* find_seq(uint64_t seq_id) const;
    /// IndexShare lookahead (TD-KVT-PREFETCH): CPU-gather the shared
    /// successors' cold misses for the CURRENT host-resident selection into
    /// pinned staging, ONE burst H2D + scatter into their row caches on the
    /// tiering H2D stream (overlaps this layer's attention/MoE).
    void prefetch_successors(int rank, int layer_idx, int n, void* stream);
    void ensure_layer_pages(SeqState& ss, int layer, int num_logical);
    /// True if any page of `layer` is COLD in `ss`.
    static bool state_layer_has_cold(const SeqState& ss, int layer);
    /// Return `seq_id`'s cold slots + row-cache entries, drop its state
    /// (in-flight demotions must be drained first).  Shared (fork-family)
    /// cold slots survive for their remaining holders.
    void release_seq(uint64_t seq_id);
    /// Drop one holder of (rank, slot); the slot returns to the free list
    /// at refcount 0 (TD-KVT-SPEC-FORK).  Throws on double release.
    void release_cold_slot(int rank, int slot);

    Options opts_;
    int hot_slots_ = 0;
    int retention_tokens_ = 0;
    int n_fake_pages_ = 0;
    int cold_pool_pages_ = 0;  ///< per rank, shared across layers and seqs
    bool sharded_ = false;     ///< kv_sharded && dcp_size >= 2
    bool dedup_ = false;       ///< replica_cold_dedup && replicated && dcp>=2
    int ppc_ = 1;              ///< pages per ownership chunk (sharded)

    std::vector<RankBufs> ranks_;
    std::vector<std::vector<LayerCache>> cache_;   ///< [rank][layer]
    std::unordered_map<uint64_t, SeqState> seqs_;  ///< TD-KVT-BATCH
    std::vector<InflightDemotion> inflight_;

    // TD-KVT-SYNC / TD-KVT-PREFETCH
    std::vector<SelCtx> sel_;                      ///< [rank]
    std::vector<std::vector<uint8_t>> pf_pending_; ///< [rank][layer]: a
        ///< prefetch targeted this layer's cache; its next materialize must
        ///< stream-wait ev_pf before reading/overwriting cache slots
    std::vector<int> share_succ_;  ///< [layer]: # of following IndexShare
        ///< SHARED layers reusing this FULL layer's selection (0 on shared)

    // Step context (begin_layer)
    uint64_t ctx_seq_ = 0;        ///< sequence of the current step (0 = none)
    int ctx_layer_ = -1;
    uint32_t ctx_pos_ = 0;        ///< token_pos of the current begin_layer
    std::vector<const int*> ctx_host_bt_;  ///< per rank, this layer's row

    uint64_t tick_ = 0;
    /// Sticky per-layer dense flag: a layer that ever fell back to dense on a
    /// tierable step (layer-invariant causes, e.g. missing indexer weights)
    /// is never demoted — its dense staging needs full residency forever.
    std::vector<uint8_t> layer_dense_;
    int total_demoted_or_inflight_ = 0;  ///< across all sequences
    bool cold_pool_full_warned_ = false;
    Stats stats_;
    uint64_t last_logged_materializations_ = 0;
};

}  // namespace layerstorm::daemon
