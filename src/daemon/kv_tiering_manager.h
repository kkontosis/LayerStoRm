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
// A FULL pool skips the same way IN-STEP ONLY (a cold slot is the sole
// copy of demoted KV — the pool never evicts one, TD-KVT-COLD-FULL-HOT-
// WEDGE / INV-KVT-18); the skipped hot backlog is recovered OUT of step:
// kMain-exhausted growth drives the dispatcher's pressure_demote sweep,
// and cold CAPACITY comes back only as whole-holder retirement through
// the orchestrator's evict-retry seam (INV-KVT-17).
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
        /// GF3.5 (IndexPool): model.index_kpool (1 = legacy). Selection is
        /// index_topk/index_kpool POOLS; the expanded per-row selection is
        /// up to index_topk_rows() token rows — every per-row buffer, cap
        /// and stride in this manager is ROWS-denominated. hot_buffer_slots
        /// auto-sizing (2 x index_topk) stays TOKEN-denominated (a
        /// retention window, not a row capacity — MODELINFO §3d).
        int index_kpool = 1;
        int index_topk_rows() const {
            return index_topk + (index_kpool > 1 ? index_kpool - 1 : 0);
        }

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

        /// TD-KVT-ADMISSION-UPFRONT (cohort seam): maximum chunk rows a
        /// blessed sparse prefill chunk may carry through materialize_row.
        /// Sizes the pinned cohort selection staging (cohort_rows_max x
        /// index_topk_rows ints + cohort_rows_max lengths).  0 = cohort seam
        /// disabled (materialize_row fails loud; legacy B==1 staging only).
        int cohort_rows_max = 0;

        /// TD-KVT-COHORT-BATCHED-MATERIALIZE: capacity (rows) of the cohort
        /// UNION staging — the batched consumer materializes the union of a
        /// chunk's per-row selections once instead of per-row fake views.
        /// Caller sizes it min(cohort_rows_max × index_topk_rows, the rank-local
        /// max prefix rows) — a union can never exceed either bound.  0 =
        /// batched cohort arm disabled (materialize_cohort returns false;
        /// per-row consumption only).  Ignored when cohort_rows_max <= 1.
        int union_rows_max = 0;

        /// IndexShare full-layer mask, EXECUTOR semantics (DcpExecutor::
        /// Options::indexer_full_layers): layer l is FULL iff the mask is
        /// empty OR (l < size && mask[l]); layers beyond the mask (MTP) are
        /// SHARED.  Drives the IndexShare reuse skip + lookahead prefetch
        /// (TD-KVT-SYNC / TD-KVT-PREFETCH).  Empty = no sharing = both off.
        std::vector<uint8_t> indexer_full_layers;

        /// GF3.9 (glm5_next hybrid): per-layer ATTENTION-TYPE mask —
        /// kv_bearing_layers[l] != 0 iff layer l appends per-token KV
        /// (sparse MLA).  KDA linear layers carry a per-request recurrent
        /// state instead (Pool::kKdaState — never demoted, never tiered,
        /// INV-KDA-STATE (d)), so tiering must SKIP them entirely:
        /// begin_layer / after_attention refuse, the share-successor walk
        /// never counts them (lookahead prefetch would otherwise walk
        /// their empty/sentinel page lists), and the S3 step-boundary
        /// flush keys on the FIRST bearing layer instead of literal 0.
        /// EMPTY = every layer bears KV (GLM-5.2/V3.2 — byte-identical
        /// legacy behavior).  Model-layer indexed, size kv_layers.
        std::vector<uint8_t> kv_bearing_layers;

        /// S3 (tiering by slab, RADIX_SLAB_DESIGN §5): kMain pages per slab
        /// (S1/S2 geometry, PageAllocator::pages_per_slab()).  > 0 enables
        /// SLAB-COHORT demotion: the in-step window sweep DEFERS each
        /// layer's demotion candidates and flushes them ONCE per step (all
        /// layers of the behind-window token range together — under
        /// position-major packing, INV-SLAB-2, that set is a run of
        /// complete slabs), so the flush issues ONE contiguous D2H per
        /// (rank, physically-contiguous run) instead of one copy per page,
        /// and a fully-demoted slab returns WHOLE to the free-slab list
        /// through the ordinary refcounted free path.  Demotion SELECTION
        /// stays PAGE-PRECISE in both modes: INV-KVT-4 retention semantics
        /// are unchanged, and a slab STRADDLING the retention window
        /// demotes its behind-window pages individually, completing on
        /// later flushes — it is never demoted whole while any of its
        /// pages sits inside the window or on the append frontier.
        /// 0 = unslabbed: legacy per-call (per-layer) demotion — fixtures
        /// and non-slabbed models.  Coalescing itself is unconditional
        /// (physically contiguous candidate pages batch into one D2H in
        /// both modes); only the cross-layer DEFERRAL is gated here.
        /// Env kill switch: LS_KVT_SLAB_DEMOTE=0 forces per-call demotion.
        int pages_per_slab = 0;
        /// Per-RANK kMain slab-span page count
        /// (PageAllocator::total_pages(gpu, kMain)): a page_idx >= span is
        /// a LOOSE page (INV-4.9b promoted spec-range index — physically
        /// outside the slab region), so a D2H run must never coalesce
        /// across the span boundary.  Empty = no span limit (tests).
        std::vector<int> slab_span_pages;

        /// TD-PREFIX-TIDY-COLD-SPILL: directory for holder cold-page spill
        /// files ("" = spilling disabled).  Created on demand (mkdir -p,
        /// "~" already expanded by the config layer or here via $HOME);
        /// any I/O failure DISABLES spilling for the boot (loudly) and
        /// never fails a step — the cold pool simply stays the last hop.
        /// The directory is a CACHE, never a store: files carry a
        /// boot-unique nonce and the orchestrator reclaims stale ones at
        /// startup; nothing in it is load-bearing across boots.
        std::string spill_dir;
        /// MANDATORY byte cap over this manager's LIVE spill files,
        /// enforced BEFORE each write (a single 25k GLM holder is
        /// ~600 MiB/rank of cold KV — a handful can fill a home
        /// partition).  A spill that would exceed the cap is REFUSED so
        /// the caller can evict spilled holders (deleting their files)
        /// and retry.  <= 0 disables spilling.
        int64_t spill_max_bytes = 0;
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
    /// `rows` > 1 (TD-KVT-ADMISSION-UPFRONT): the step is a blessed sparse
    /// prefill CHUNK of `rows` consecutive positions [token_pos,
    /// token_pos + rows) of ONE sequence, consumed per row through
    /// materialize_row — token_pos is the FIRST row's write position (the
    /// demote-legality frontier bound), the high-water mark advances by the
    /// whole chunk.
    bool begin_layer(int layer, uint64_t seq_id, uint32_t token_pos,
                     const int* const* host_block_tables, int rows = 1);

    /// Demote this layer's pages that fell fully behind the retention window
    /// (background D2H per rank on the D2H stream; page freed to the
    /// allocator once every rank's copy completed).  `pages` points at the
    /// sequence's handle for (logical 0, layer); consecutive logical pages
    /// are `handle_stride` handles apart (dispatcher layout: pages[j*L + l]).
    void after_attention(int layer, uint64_t seq_id, uint32_t token_pos,
                         const memory::PageHandle* pages, int num_logical,
                         int handle_stride);

    /// R3 holder hibernation (CMD_SEQ_HIBERNATE): demote ALL of one layer's
    /// demote-eligible hot pages of a FROZEN sequence — everything except
    /// the append-frontier logical page (INV-KVT-4's frontier rule kept) —
    /// through the same machinery as window demotion.  Returns the number
    /// of pages enqueued for D2H (0 = nothing eligible / dense layer).
    int hibernate_layer(int layer, uint64_t seq_id,
                        const memory::PageHandle* pages, int num_logical,
                        int handle_stride, int frontier_logical = -1);

    // ── TD-PREFIX-TIDY-COLD-SPILL (S3 rider): 2nd tiering hop ──────────

    /// Spill a HIBERNATED holder's COLD pages to ONE file under spill_dir
    /// (the second tiering hop: VRAM → pinned cold pool → disk).  Writes
    /// one copy per page (replicas are byte-identical, INV-KV-REP; the
    /// dedup/shard owner's slot is the source), releases every cold slot
    /// (respecting fork-family refcounts — a shared slot survives for its
    /// other holders and frees no RAM), flips pages kCold → kSpilled and
    /// returns pages spilled.  Returns -1 when the byte cap would be
    /// exceeded (nothing written — the caller evicts spilled holders and
    /// may retry); 0 when disabled / nothing cold / not hibernated.  An
    /// I/O failure unlinks the partial file, keeps every slot (pages stay
    /// kCold) and disables spilling for the boot — capacity, never
    /// correctness.
    int spill_seq(uint64_t seq_id);

    /// True when the sequence has kSpilled pages — fork/snapshot/
    /// repromote must unspill_seq() first (the dispatcher owns those call
    /// sites; reaching a spilled page on a read path throws, INV-KVT-2
    /// fail-loud).
    bool seq_spilled(uint64_t seq_id) const;

    /// Reload every kSpilled page into fresh cold slots (byte-exact,
    /// INV-KVT-1) and delete the spill file.  Slot acquisition follows
    /// the demotion storing-rank rule (all ranks under replicated
    /// non-dedup — the one file copy fans out; the owner under dedup/
    /// sharded) and bypasses the fair-share cap (holder reload, like
    /// hibernation).  False on cold-slot exhaustion or I/O failure —
    /// partial progress stays consistent (reloaded pages are kCold; the
    /// rest stay kSpilled with the file intact) and the caller fails
    /// RETRYABLE (kKvPoolExhausted: holder eviction frees slots).
    bool unspill_seq(uint64_t seq_id);

    /// Live bytes across this manager's spill files (cap accounting).
    int64_t spilled_bytes_total() const { return spilled_total_; }

    /// S3 whole-sequence hibernation: hibernate_layer over EVERY layer of
    /// the dispatcher's layer-major handle table (handle (logical j, layer
    /// l) at pages[j * kv_layers + l]) collected into ONE slab-grouped
    /// flush — all layers of the holder's cold token range demote together,
    /// so complete slabs leave in single contiguous D2H runs (one
    /// attention-order fence for the whole sweep).  Same demote set and
    /// frontier rule as the per-layer loop it replaces (INV-KVT-4;
    /// fair-share-cap-exempt like hibernate_layer).  Returns total pages
    /// enqueued for D2H.
    int hibernate_seq(uint64_t seq_id, const memory::PageHandle* pages,
                      int num_logical, int frontier_logical = -1);

    /// TD-KVT-COLD-FULL-HOT-WEDGE: out-of-step pressure sweep.  Re-runs the
    /// after_attention window demotion for EVERY layer of one LIVE sequence
    /// at its recorded high-water position — the behind-window hot BACKLOG
    /// that accumulated while demotions were skipped (cold pool full /
    /// fair-share budget) becomes demotable the moment slots free up (the
    /// orchestrator's holder-eviction seam, INV-KVT-17), and this call
    /// drains it WITHOUT the very step that kMain exhaustion is blocking.
    /// Same demote_layer_range body as a step's sweep: retention window and
    /// append-frontier rules hold unchanged (INV-KVT-4), fair-share
    /// applies, hibernated holders / unknown / never-stepped sequences are
    /// no-ops.  `pages` is the dispatcher's layer-major handle table
    /// (handle (logical j, layer l) at pages[j * kv_layers + l]).  Returns
    /// pages enqueued for D2H; the caller must drain_demotions() before
    /// retrying an allocation.
    int pressure_demote(uint64_t seq_id, const memory::PageHandle* pages,
                        int num_logical);

    /// Poll in-flight demotion D2H copies; completed pages flip to COLD and
    /// their device pages return to the allocator.
    void poll_demotions();

private:
    /// Shared demotion body (window demotion + hibernation): demote this
    /// layer's HOT pages with (j+1)*page_size <= demote_end_tok and
    /// j < frontier.  Returns pages enqueued.  S3: implemented as
    /// collect_layer_range + an immediate flush_pending.
    int demote_layer_range(int layer, uint64_t seq_id,
                           const memory::PageHandle* pages, int num_logical,
                           int handle_stride, int64_t demote_end_tok,
                           int frontier, bool fair_share = true);

    /// S3 phase 1: mark this layer's window-eligible HOT pages
    /// pending-demote and append them (handle copies) to the step batch.
    /// Same eligibility predicate as the pre-S3 demotion loop (retention
    /// window, append frontier, sticky-dense layer, valid handle).  A
    /// foreign-sequence pending tail is flushed first (one batch, one
    /// sequence).  Returns candidates collected.
    int collect_layer_range(int layer, uint64_t seq_id,
                            const memory::PageHandle* pages, int num_logical,
                            int handle_stride, int64_t demote_end_tok,
                            int frontier);

    /// S3 phase 2: demote the pending batch — per-page cold-slot
    /// acquisition (fair-share budget + pool-capacity fail-safe exactly as
    /// before), then the D2H issued as physically-contiguous runs (sorted
    /// by (storing rank, page_idx); a run needs source pages AND every
    /// storing rank's cold slots consecutive, and never crosses the
    /// slab-span boundary).  One InflightDemotion group + one
    /// attention-order fence per participating rank per flush.  Returns
    /// pages enqueued for D2H; skipped candidates stay HOT (fail-safe).
    int flush_pending(bool fair_share);

    /// Drop a dying sequence's pending batch without demoting it (its
    /// pages are freed by ordinary sequence teardown).
    void discard_pending(uint64_t seq_id);

public:

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
    /// R4a TRUNCATING fork: prefix_len > 0 = the child took only the
    /// parent's first prefix_len TOKENS, i.e. logical pages
    /// [0, ceil(prefix_len / page_size)).  The child's copied state is
    /// truncated to that logical prefix per layer, only the KEPT cold
    /// slots are refcount-shared (a full-copy share would leak the
    /// parent's tail slots at the child's release: release_seq walks the
    /// child's own page vectors), and demoted_frontier /
    /// demoted_or_inflight / cold_used / max_pos_seen are RECOMPUTED from
    /// the kept pages.  prefix_len == 0 keeps the full-fork semantics
    /// above byte-identically.
    void on_seq_fork(uint64_t src_seq_id, uint64_t dst_seq_id,
                     uint32_t prefix_len = 0);

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

    /// TD-KVT-ADMISSION-UPFRONT: per-row cohort materialization for a
    /// blessed sparse prefill chunk (begin_layer with rows > 1).  One
    /// batched selection readback per (rank, layer) — IndexShare shared
    /// layers reuse the host copy (selection_fresh == false, INV-KVT-6
    /// identity: same seq/pos/rows) — then row slices classify/gather
    /// through the same helper as the B==1 path (INV-KVT-1 placement-only).
    /// Cohort rows never trigger the IndexShare lookahead prefetch.
    bool materialize_row(int rank, int layer_idx, int row, int rows,
                         const int* sparse_indices_dev,
                         const int* topk_lengths_dev,
                         bool selection_fresh, void* stream,
                         parallelism::TieredKvView* out) override;

    /// TD-KVT-COHORT-BATCHED-MATERIALIZE: batched union cohort consumption.
    /// ONE gather of the union of the chunk's per-row selections into the
    /// union staging set + host-side order-preserving index rewrite to union
    /// slots ([rows x index_topk_rows] upload; IndexShare shared layers reuse the
    /// union + rewrite under the (seq, pos, rows) identity — the gather still
    /// runs per layer, per-layer KV bytes/cold sets).  Same classify/burst/
    /// gather placement body as the per-row path (INV-KVT-1); union cold
    /// misses skip row-cache INSERTS (the union dedups intra-step; mass
    /// inserts would thrash the LRU) while cache hits still serve as sources.
    /// False when the layer has no cold pages, union_rows_max is 0/exceeded,
    /// or LS_KVT_COHORT_ROWWISE=1 — caller falls back to materialize_row.
    bool materialize_cohort(int rank, int layer_idx, int rows,
                            const int* sparse_indices_dev,
                            const int* topk_lengths_dev,
                            bool selection_fresh, void* stream,
                            parallelism::TieredKvView* out) override;

    /// Hybrid cohort gate: true iff the current begin_layer cohort's layer
    /// has cold pages (per-row consumption required) — all-hot layers keep
    /// the batched sparse chunk kernel.  Env LS_KVT_COHORT_ALWAYS=1 forces
    /// true for every tier-step chunk (strict same-arm identity runs: the
    /// per-row shape then never depends on placement state).
    bool cohort_layer_tiered(int layer_idx) override;

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
        uint64_t cold_full_skips = 0;   ///< demote batches cut short by a
                                        ///< FULL cold pool (in-step skip,
                                        ///< TD-KVT-COLD-FULL-HOT-WEDGE)
        uint64_t pressure_demoted = 0;  ///< backlog pages demoted by
                                        ///< out-of-step pressure sweeps
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
        // TD-KVT-ADMISSION-UPFRONT (cohort seam)
        uint64_t cohort_readbacks = 0;      ///< batched cohort selection D2Hs
        uint64_t cohort_rows_tiered = 0;    ///< chunk rows consumed via a
                                            ///< materialize_row fake view
        // TD-KVT-COHORT-BATCHED-MATERIALIZE (union arm)
        uint64_t cohort_unions = 0;         ///< batched union materializations
                                            ///< (one per (rank, layer))
        uint64_t cohort_union_rows = 0;     ///< union rows gathered
        uint64_t cohort_union_rewrites = 0; ///< union builds + index rewrites
                                            ///< (IndexShare reuse skips these)
        // TD-PREFIX-TIDY-COLD-SPILL (2nd hop)
        uint64_t spill_files = 0;       ///< holder spill files written
        uint64_t spill_pages = 0;       ///< cold pages written to disk
        uint64_t spill_bytes = 0;       ///< bytes written to spill files
        uint64_t spill_cap_refusals = 0;///< spills refused by the byte cap
        uint64_t unspill_pages = 0;     ///< pages reloaded from disk
        uint64_t unspill_bytes = 0;     ///< bytes reloaded from disk
        // S3 (tiering by slab) — slab-cohort demotion/promotion telemetry
        uint64_t slab_flushes = 0;      ///< pending-batch demote flushes
        uint64_t slab_runs = 0;         ///< contiguous D2H runs issued
        uint64_t slab_pages = 0;        ///< pages demoted through flushes
        uint64_t slabs_demoted_whole = 0;  ///< whole slabs covered by a
                                           ///< single contiguous run
        uint64_t promote_runs = 0;      ///< coalesced re-promotion H2D runs
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
    enum class PageState : uint8_t { kHot = 0, kD2hInflight = 1, kCold = 2,
                                     kSpilled = 3 };

    struct PageInfo {
        PageState state = PageState::kHot;
        /// S3: collected into the pending slab-cohort demote batch (still
        /// HOT — the flag only prevents double collection before the
        /// step's flush).  Cleared by flush_pending / discard_pending.
        bool pending_demote = false;
        std::vector<int> cold_slot;  ///< per-rank cold pool slot (-1 = none;
                                     ///< exactly one >= 0 under dedup/
                                     ///< sharded).  Slots are REFCOUNTED
                                     ///< (RankBufs::cold_ref) — a fork
                                     ///< family shares a demoted parent's
                                     ///< slots (TD-KVT-SPEC-FORK).
        /// TD-PREFIX-TIDY-COLD-SPILL: byte offset of this page's single
        /// copy inside the sequence's spill file (kSpilled only; -1
        /// otherwise).
        int64_t spill_off = -1;
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
        /// R3: this sequence was HIBERNATED (a frozen prefix holder).  It
        /// never steps, so it will never demote again — the cold
        /// fair-share census (nseq_cold) excludes it: its retained slots
        /// are the prefix cache's budget (bounded by holder eviction),
        /// not a competing live demoter's share.  Without the exclusion a
        /// live long prefill beside k hibernated holders is throttled to
        /// capacity/(k+1) and its retention window stops draining
        /// (measured: budget_skips=7063 during one 8k prefill beside two
        /// hibernated warm-up holders).
        bool hibernated = false;
        /// Cold pool slots held per rank — fair-share budget input: with
        /// N demoting sequences a sequence's per-rank cold usage is capped
        /// at capacity / N; further demotions are skipped (fail-safe, pages
        /// stay hot) so one sequence cannot monopolize the shared pool.
        /// A single demoting sequence keeps the full pool.
        std::vector<int> cold_used;
        /// TD-PREFIX-TIDY-COLD-SPILL: this holder's spill file (empty =
        /// none) and its live byte count (cap accounting; deleted +
        /// released at unspill / release_seq).
        std::string spill_path;
        int64_t spill_bytes = 0;
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

    /// One materialize staging set.  TWO exist per rank (ping-pong,
    /// TD-KVT-ADMISSION-UPFRONT): per-row cohort consumption materializes
    /// once per chunk ROW, and a single set host-serializes every
    /// materialize behind the PREVIOUS row's device-side gather (the
    /// mat_inflight guard measured ~37% of the tiered prefill wall).
    /// Alternating sets lets the host run ahead; device-side ordering is
    /// unchanged (all cache reads/writes and the gather stay on the
    /// attention stream).
    struct MatSet {
        // Device
        void* scratch = nullptr;        ///< fake pages: n_fake_pages × stride_block
        void* cold_incoming = nullptr;  ///< index_topk_rows × stride_row (packed misses)
        void* dev_src_ptrs = nullptr;   ///< index_topk_rows × void*
        void* dev_scatter_ptrs = nullptr;   ///< index_topk_rows × void*
        void* dev_scatter_idx = nullptr;    ///< index_topk_rows × int
        // Pinned host (inside the rank arena)
        char* h_stage = nullptr;            ///< index_topk_rows × stride_row
        const void** h_src_ptrs = nullptr;  ///< index_topk_rows
        const void** h_scatter_ptrs = nullptr;
        int* h_scatter_idx = nullptr;
        // Events / state
        void* ev_h2d = nullptr;      ///< cold burst completion
        void* ev_mat_done = nullptr; ///< recorded after the step-5 gather
                                     ///< (host staging-reuse guard)
        bool mat_inflight = false;   ///< ev_mat_done recorded, staging in use
    };

    struct RankBufs {
        // Device
        void* row_cache = nullptr;      ///< kv_layers × hot_slots × stride_row
        void* dev_ident_indices = nullptr;  ///< index_topk_rows × int iota
        void* dev_fake_bt = nullptr;        ///< n_fake_pages × int iota
        MatSet mat[2];                  ///< ping-pong materialize staging
        int mat_parity = 0;             ///< next set to use
        // TD-KVT-COHORT-BATCHED-MATERIALIZE: union staging (single set —
        // one union materialize per (rank, layer), the inter-layer attention
        // + MoE work drains the previous gather).  Allocated only when the
        // union arm is enabled (union_rows_max > 0, cohort seam on, not
        // LS_KVT_COHORT_ROWWISE).  umat capacity = union_rows_max rows;
        // umat.h_scatter_* stay null (union gathers never cache-insert).
        MatSet umat;
        void* dev_uidx = nullptr;       ///< cohort_rows_max × index_topk_rows int
                                        ///< (per-row indices rewritten to
                                        ///< union slots, order-preserving)
        void* dev_useq = nullptr;       ///< cohort_rows_max × int (= union_n)
        void* dev_union_bt = nullptr;   ///< cohort_rows_max × u_pages_cap int
                                        ///< (replicated iota rows)
        int* h_uidx = nullptr;          ///< pinned mirror of dev_uidx
        int* h_useq = nullptr;          ///< pinned mirror of dev_useq
        void* ev_uidx = nullptr;        ///< h_uidx/h_useq upload fence
                                        ///< (INV-KVT-7 single-writer)
        bool uidx_inflight = false;
        // Union identity (IndexShare reuse of the union build + rewrite;
        // the GATHER always reruns per layer — per-layer bytes/cold sets).
        bool u_valid = false;
        uint64_t u_seq = 0;
        uint32_t u_pos = 0;
        int u_rows = 0;
        int u_n = 0;                    ///< union rows (host)
        std::vector<int> u_pos_list;    ///< ascending unique union positions
        std::vector<int> u_slot_of;     ///< position → union slot (-1 = none)
        // Device — IndexShare lookahead prefetch (TD-KVT-PREFETCH); separate
        // from the materialize buffers: a prefetch burst may be in flight
        // while the next materialize reuses cold_incoming/h_stage.
        void* pf_incoming = nullptr;        ///< index_topk_rows × stride_row
        void* dev_pf_scatter_ptrs = nullptr;  ///< index_topk_rows × void*
        void* dev_pf_scatter_idx = nullptr;   ///< index_topk_rows × int
        // Pinned host (NUMA home node)
        memory::NumaBuffer host_arena{};    ///< staging + tables + cold pool
        bool host_registered = false;
        int numa_node = -1;
        int* h_indices = nullptr;           ///< cohort_rows_max × index_topk_rows
        int* h_topk_len = nullptr;          ///< cohort_rows_max (min 1)
        char* h_pf_stage = nullptr;         ///< index_topk_rows × stride_row
        const void** h_pf_scatter_ptrs = nullptr;  ///< index_topk_rows
        int* h_pf_scatter_idx = nullptr;    ///< index_topk_rows
        char* cold_base = nullptr;          ///< cold_pool_pages × stride_block
        // Streams / events.  h2d_stream is ALWAYS owned (TD-KVT-H2D-
        // CONTENTION: a dedicated per-rank tiering H2D stream so cold bursts
        // never queue behind ~18 MB expert-weight copies on kH2dTransfer);
        // d2h_stream is the shared kD2hTransfer when a StreamManager exists.
        void* h2d_stream = nullptr;
        void* d2h_stream = nullptr;
        bool owns_d2h_stream = false;
        void* ev_sync = nullptr;     ///< host-wait event (indices D2H)
        void* ev_attn_order = nullptr;  ///< attention→D2H ordering
        void* ev_pf = nullptr;          ///< prefetch burst+scatter completion
        void* ev_pf_order = nullptr;    ///< attention→prefetch-write ordering
        bool pf_inflight = false;       ///< ev_pf recorded, h_pf_* in use
        std::vector<int> cold_free;  ///< free cold pool slots (kept sorted
                                     ///< DESCENDING so pops hand out
                                     ///< ascending, contiguous slot runs —
                                     ///< S3 D2H coalescing)
        bool cold_free_dirty = false;  ///< out-of-order releases since the
                                       ///< last flush-time re-sort
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
        int rows = 0;            ///< cohort rows held (0/1 = single-row;
                                 ///< TD-KVT-ADMISSION-UPFRONT)
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
    /// Cohort variant (TD-KVT-ADMISSION-UPFRONT): ONE batched D2H of the
    /// whole chunk's selection (rows x index_topk_rows indices + rows lengths)
    /// into the cohort staging; IndexShare shared layers reuse the host copy
    /// (selection_fresh == false + same (seq, pos, rows) — INV-KVT-6).
    void ensure_cohort_selection(int rank, int layer_idx,
                                 const int* sparse_indices_dev,
                                 const int* topk_lengths_dev,
                                 bool selection_fresh, void* stream);
    /// Shared classify/burst/gather/view body over one row's host-resident
    /// selection (h_idx, n rows): steps 1c-6 of the original materialize —
    /// used by BOTH the B==1 path and materialize_row (INV-KVT-1: identical
    /// placement machinery).  `seqlens_dev` is the device int the fake view
    /// exposes as its per-row length (topk_lengths_dev [+ row]).
    void materialize_selection(int rank, int layer_idx, const int* h_idx,
                               int n, const int* seqlens_dev, void* stream,
                               parallelism::TieredKvView* out);
    /// Steps 1c-5 (classify / cold burst / scatter / gather / fence) over a
    /// host-resident selection into staging set `ms` (capacity `cap` rows) —
    /// the placement body shared by materialize_selection (ping-pong sets,
    /// cache_insert=true) and the union arm (umat, cache_insert=false: the
    /// union dedups intra-step; row-cache hits still serve as sources).
    void gather_selection(int rank, int layer_idx, MatSet& ms, int cap,
                          const int* h_idx, int n, bool cache_insert,
                          void* stream);
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
    /// GF3.9: layer bears per-token KV (attention-type gating; empty mask
    /// = every layer — legacy byte-identical).
    bool layer_bears_kv(int l) const {
        return opts_.kv_bearing_layers.empty()
            || (l >= 0
                && l < static_cast<int>(opts_.kv_bearing_layers.size())
                && opts_.kv_bearing_layers[static_cast<size_t>(l)] != 0);
    }
    int first_bearing_layer_ = 0;  ///< GF3.9: S3 step-boundary flush key
    std::vector<int> share_succ_;  ///< [layer]: # of following IndexShare
        ///< SHARED layers reusing this FULL layer's selection (0 on shared)

    // S3 slab-cohort demotion: the deferred per-step candidate batch.
    struct PendingDemote {
        int layer;
        int logical;
        memory::PageHandle handle;  ///< copied at collect time (valid until
                                    ///< the flush: every state-mutating
                                    ///< entry point flushes or discards
                                    ///< first, and appends never touch
                                    ///< behind-window pages)
    };
    std::vector<PendingDemote> pend_;
    uint64_t pend_seq_ = 0;       ///< sequence of the pending batch
    int pend_last_layer_ = -1;    ///< layer of the newest collected
                                  ///< candidates — layer 0 marks a
                                  ///< still-running layer-outer superchunk
                                  ///< pass (begin_layer(0) must not flush
                                  ///< it; see the trigger comment)
    bool slab_demote_ = false;    ///< pages_per_slab > 0 && env not "0":
                                  ///< after_attention defers to a per-step
                                  ///< flush (layer == kv_layers-1 / next
                                  ///< step's begin_layer)

    // Step context (begin_layer)
    uint64_t ctx_seq_ = 0;        ///< sequence of the current step (0 = none)
    int ctx_layer_ = -1;
    uint32_t ctx_pos_ = 0;        ///< token_pos of the current begin_layer
    int ctx_rows_ = 1;            ///< chunk rows (1 = decode/B==1 step;
                                  ///< TD-KVT-ADMISSION-UPFRONT cohorts > 1)
    bool ctx_cold_valid_ = false; ///< cohort per-layer any-cold cache below
    bool ctx_cold_ = false;       ///< layer ctx_layer_ has cold pages
    std::vector<const int*> ctx_host_bt_;  ///< per rank, this layer's row

    uint64_t tick_ = 0;
    /// Sticky per-layer dense flag: a layer that ever fell back to dense on a
    /// tierable step (layer-invariant causes, e.g. missing indexer weights)
    /// is never demoted — its dense staging needs full residency forever.
    std::vector<uint8_t> layer_dense_;
    int total_demoted_or_inflight_ = 0;  ///< across all sequences
    bool cohort_always_ = false;  ///< LS_KVT_COHORT_ALWAYS=1 (strict per-row)
    bool cohort_rowwise_ = false; ///< LS_KVT_COHORT_ROWWISE=1: kill-switch —
                                  ///< force per-row cohort consumption (the
                                  ///< batched union arm returns false)
    int union_cap_ = 0;           ///< union staging rows (0 = arm disabled)
    int u_pages_cap_ = 0;         ///< union fake pages (ceil(cap / page))
    // TD-PREFIX-TIDY-COLD-SPILL
    int64_t spilled_total_ = 0;   ///< live bytes across spill files
    bool spill_disabled_ = false; ///< sticky: first I/O failure disables
    uint64_t spill_nonce_ = 0;    ///< boot-unique file-name component
    bool verify_ = false;         ///< LS_KVT_VERIFY=1 byte-oracle (debug)
    bool cold_pool_full_warned_ = false;
    Stats stats_;
    uint64_t last_logged_materializations_ = 0;
};

}  // namespace layerstorm::daemon
