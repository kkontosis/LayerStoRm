#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "core/memory/vram_allocator.h"

namespace layerstorm::compute { class DeviceBackend; }

namespace layerstorm::memory {

// ── Pool type ────────────────────────────────────────────────────────────────

// kMain/kSpeculation share one page size (MLA main pool; V4: the CSA bucket).
// kIndexerK, kHca, kSwa are independent side pools with their own regions and
// page sizes (kHca/kSwa exist only for V4 models, V4-3c; kIndexerK serves DSA
// or the V4 Lightning-Indexer tier). kKdaState (GF3.8, glm5_next only) is the
// per-REQUEST KDA recurrent+conv state pool: a "page" is one whole-request
// SLOT (VramLayout::kda.slot_bytes; no position axis, no growth) claimed at
// seq_create, D2D-copied on fork, freed at seq_free. Values are APPENDED,
// never reordered (the SASS byte-identity design rule, dossier section 18).
enum class Pool { kMain, kSpeculation, kIndexerK, kHca, kSwa, kKdaState };

// ── Page handle (lightweight value, returned from allocation) ────────────────

struct PageHandle {
    int gpu_idx = -1;    ///< GPU index (into VramAllocator regions)
    int page_idx = -1;   ///< Index into per-GPU page array (pool-relative for indexer K)
    void* gpu_ptr = nullptr;  ///< Direct pointer to GPU memory for this page
    Pool pool = Pool::kMain;  ///< Which pool this page belongs to
};

// ── Page metadata (CPU-side only, not on GPU) ────────────────────────────────

struct PageMeta {
    uint32_t layer_index = 0;
    uint64_t sequence_id = 0;
    uint32_t token_start = 0;   ///< Inclusive
    uint32_t token_end = 0;     ///< Exclusive
    uint32_t refcount = 0;      ///< 0 = free
    Pool pool = Pool::kMain;
    /// INV-KV-REP: page_idx claimed in lockstep on EVERY TP GPU (replicated
    /// KV).  free/add_ref/cow_copy mirror across all TP GPUs; the canonical
    /// (rank-0) GPU's meta is authoritative for sequence fields.
    bool replicated = false;
};

// ── DCP token-routing configuration ──────────────────────────────────────────

/// DCP token-routing config.  Constructed by orchestrator from Config.
/// Controls round-robin chunk assignment of tokens to DCP ranks.
struct DcpConfig {
    int dcp_size = 1;                ///< tensor_parallelism (1 = no DCP)
    int dcp_chunk_size = 16;         ///< Tokens per chunk for round-robin
    int page_size_tokens = 16;       ///< Tokens per KV page (from config)
    std::vector<int> tp_gpu_indices; ///< rank → gpu_idx mapping
    bool indexer_k_sharded = false;  ///< true when dcp_indexer_mode == local
    /// TD-GLM-INDEXER-LOCAL-MERGE: indexer-K local-mode ownership unit
    /// (memory.kv_cache.indexer_k_page_size_tokens). Indexer-K pages are
    /// rank-routed round-robin by INDEXER PAGE — owner(pos) =
    /// (pos / indexer_k_page_size_tokens) % dcp_size — NOT by the KV chunk
    /// above: an indexer page (8192 tokens default) spans many KV chunks
    /// (16 default), so chunk routing could not keep pages rank-atomic (and
    /// would degenerately map every page-aligned start to rank 0). Must
    /// match the producer/dispatcher page size. 0 falls back to chunk
    /// routing (pre-local-merge behavior; not used by the engine).
    int indexer_k_page_size_tokens = 0;
    /// true when hardware.dcp_kv_mode == sharded.  Sharded: kMain KV pages
    /// are owner-routed by token position (INV-4.9e).  Replicated (default,
    /// INV-KV-REP): every TP GPU holds the FULL KV — each logical page claims
    /// the SAME physical index on every TP GPU in lockstep, so the replicated
    /// block tables / slot mappings are valid on every rank
    /// (TD-KV-REPLICATED-PAGE-ALIAS fix).
    bool kv_sharded = false;

    bool enabled() const { return dcp_size >= 2; }
};

// ── PageAllocator ────────────────────────────────────────────────────────────

/// Paged KV cache allocator with two logical pools (main + speculation) plus
/// independent side pools: indexer K (DSA / V4 lightning tier) and — for V4
/// models — the HCA and SWA page buckets (V4-3c; page sizes from
/// VramLayout::v4).
///
/// Operates on GPU memory already allocated by VramAllocator. Main pages use
/// the kv_main region; speculation pages use the kv_speculation region (may
/// be non-contiguous). Pool membership is tracked via CPU-side metadata,
/// enabling zero-copy promotion (INV-4.9b).
///
/// Not thread-safe. Designed for single-threaded orchestrator (INV-3.4.2).
class PageAllocator {
public:
    /// Construct from a VramAllocator. The VramAllocator must outlive this object.
    /// @param copy_backend  DeviceBackend used for D2D copies (CoW). Must outlive this.
    explicit PageAllocator(const VramAllocator& vram,
                           compute::DeviceBackend* copy_backend);

    // ── Allocation ───────────────────────────────────────────────────────

    /// Allocate a page from the specified pool on the given GPU.
    /// Returns std::nullopt if the pool is exhausted.
    std::optional<PageHandle> allocate(int gpu_idx, Pool pool);

    /// Free a page. Decrements refcount; if it reaches 0, returns the page
    /// to its current pool's free list and clears metadata.
    void free(PageHandle handle);

    // ── Promotion (speculation → main) ───────────────────────────────────

    /// Promote a speculation page to the main pool (metadata-only, no copy).
    /// The page must be allocated and in the speculation pool.
    /// INV-KV-REP: replicated pages flip the pool on EVERY TP GPU's mirror
    /// (a canonical-only flip would desync return_to_free_list routing).
    void promote(PageHandle handle);

    // ── Copy-on-write ────────────────────────────────────────────────────

    /// Increment refcount for copy-on-write sharing.
    void add_ref(PageHandle handle);

    /// Copy-on-write split. If refcount == 1, returns the same handle (no-op).
    /// If refcount > 1, allocates a new page in the same pool, copies data,
    /// decrements the old page's refcount, and returns the new handle.
    /// Throws std::runtime_error if the pool is exhausted and a copy is needed.
    ///
    /// S2 (slabbed kMain): the copy PREFERS free space in the SOURCE page's
    /// slab — the fork-frontier split then lands next to the page it split
    /// from (same fork family, the blessed cross-sequence tenancy) instead
    /// of claiming a fresh slab per holder, which is what keeps GLM prefix
    /// holders refcount-cheap (INV-PREFIX-CACHE-3) under rule 3. Fallback
    /// order: the destination sequence's own run (position-routed by the
    /// source page's token_start — this is how a DIVERGING fork child
    /// claims its own frontier slabs), then loose.
    /// @param dst_seq_id  the sequence the copy will belong to (fork child);
    ///   0 keeps the source page's sequence (metadata is copied either way —
    ///   the caller still relabels meta.sequence_id as before; this param
    ///   only routes the S2 run placement).
    PageHandle cow_copy(PageHandle handle, uint64_t dst_seq_id = 0);

    // ── Metadata access ──────────────────────────────────────────────────

    PageMeta& meta(PageHandle handle);
    const PageMeta& meta(PageHandle handle) const;

    // ── Queries ──────────────────────────────────────────────────────────

    /// Number of free pages available in the specified pool. O(1).
    int free_pages(int gpu_idx, Pool pool) const;

    /// Initial configured capacity of the specified pool.
    int total_pages(int gpu_idx, Pool pool) const;

    /// Number of pages currently in use in the specified pool. O(n) scan.
    int used_pages(int gpu_idx, Pool pool) const;

    /// Number of GPUs managed.
    int gpu_count() const;

    /// KV cache format (informational — does not affect allocation logic).
    KvCacheFormat kv_cache_format() const { return format_; }

    /// GF3.8: KDA state slot bytes (uniform across GPUs; 0 when the model
    /// carries no KDA state pool). The whole-request slot stride — also the
    /// CHECKPOINT byte size (GF3.6 unit: fp32 recurrent + 3 conv rings for
    /// every linear layer travel as one contiguous slot copy).
    int64_t kda_state_slot_bytes() const {
        // Mode-independent LOGICAL size: mapped mode repurposes the side
        // pool's bytes_per_page for the per-layer unit, but the slot — the
        // checkpoint/spill/fork unit — is always the whole-request layout
        // size (TD-KDA-STATE-MAPPED-SLABS keeps the file format and every
        // host-side blob byte-identical across modes).
        return kda_layout_.enabled ? kda_layout_.slot_bytes : 0;
    }

    // ── TD-KDA-STATE-MAPPED-SLABS: mapped-state mode ────────────────────
    /// True when the KDA state is MAPPED over the shared slab region
    /// (per-(request, linear layer) contiguous whole-slab runs) instead of
    /// the GF3.8 dedicated carve. Decided at sizing (VramLayout::kda.mapped).
    bool kda_state_mapped() const { return kda_mapped_; }
    /// Handles a sequence owns per engaged GPU: 1 (carve: the whole slot)
    /// or kda_layout().num_layers (mapped: one per-layer unit each).
    int kda_units_per_rank() const {
        return !kda_layout_.enabled ? 0
             : (kda_mapped_ ? kda_layout_.num_layers : 1);
    }
    /// LOGICAL bytes one handle covers (slot_bytes carve; per_layer_bytes
    /// mapped). units_per_rank * unit_bytes == slot_bytes in both modes.
    int64_t kda_unit_bytes() const {
        return !kda_layout_.enabled ? 0
             : (kda_mapped_ ? kda_layout_.per_layer_bytes
                            : kda_layout_.slot_bytes);
    }
    /// PHYSICAL stride the kernels multiply PageHandle::page_idx by
    /// (slot_bytes carve; slab_bytes mapped — page_idx is then the unit's
    /// run-start slab id over the shared region).
    int64_t kda_state_stride_bytes() const {
        return kda_mapped_ ? slab_bytes_ : kda_state_slot_bytes();
    }
    /// Claim one sequence's KDA state on ONE GPU: all units, all-or-nothing
    /// (empty vector = retryable exhaustion; nothing claimed). Carve mode
    /// returns 1 slot handle; mapped mode returns num_layers handles, each
    /// a contiguous run of kda_unit_slabs() whole slabs from the shared
    /// free-slab list (honors the INV-4.9f headroom floor). meta().
    /// sequence_id is stamped with seq_id. Free through the ordinary
    /// free(handle) / free_sequence paths (runs return whole).
    std::vector<PageHandle> allocate_kda_state(int gpu_idx, uint64_t seq_id);
    /// TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA: the boot-computed admissible
    /// single-request context ceiling (tokens; min over attention-host
    /// GPUs; 0 = not computed — V4, unpaged shapes, hand-built layouts).
    /// Quoted by the seq_create fast-fail when a request's own whole-life
    /// demand exceeds the entire kMain pool.
    int admissible_ctx_tokens() const { return admissible_ctx_tokens_; }
    /// Mapped-mode allocation unit: whole slabs per per-layer run (0 when
    /// not mapped).
    int kda_unit_slabs() const { return kda_unit_slabs_; }

    /// TD-KDA-MAPPED-FRAG observability: per-GPU mapped-claim outcome
    /// counters with the byte-vs-contiguity DISCRIMINATOR. A refusal is
    /// CONTIGUITY-class when the free-slab count could cover the whole
    /// demand (free_slabs >= num_layers * unit_slabs) but the scan found
    /// fewer than num_layers runs of unit_slabs consecutive free slabs —
    /// i.e. fragmentation, the failure eviction cannot fix
    /// (TD-KDA-MAPPED-RETRY-EFFECTIVE). Anything else (including a
    /// headroom-floor refusal, last_found_runs == -1) is CAPACITY-class.
    /// The numbers are free: the failed claim scan computes them anyway.
    struct KdaMappedStats {
        uint64_t claims = 0;               ///< successful whole-GPU claims
        uint64_t refusals_capacity = 0;    ///< byte shortage (evict helps)
        uint64_t refusals_contiguity = 0;  ///< fragmentation (evict may not)
        // Snapshot of the LAST refusal (claim_kda_state's message source).
        int last_found_runs = -1;          ///< -1 = refused before the scan
        int last_needed_runs = 0;
        int last_free_slabs = 0;
        bool last_was_contiguity = false;
    };
    const KdaMappedStats& kda_mapped_stats(int gpu_idx) const {
        return gpus_[gpu_idx].kda_stats;
    }

    // ── 44z: expert-zone grants (the FOURTH slab tenant class) ──────────
    /// A contiguous run of whole kMain slabs LENT to the expert cache.
    /// The shared region's tenants are now: (1) per-sequence KV bump runs
    /// (S2), (2) elastic single-slab indexer-K claims (S4), (3) mapped KDA
    /// state runs (TD-KDA-STATE-MAPPED-SLABS), and (4) these grants.
    ///
    /// THE ASYMMETRY that makes the lending safe in one direction only: an
    /// expert slot is a CACHE LINE over host RAM — reclaiming it costs a
    /// re-fetch (~0.46 ms) and destroys nothing; a KV page is STATE — no
    /// host copy exists to re-read, so dropping one destroys work already
    /// paid for. Pressure therefore flows expert -> KV: a grant is handed
    /// back when KV needs the bytes, never the reverse. That is why a
    /// grant is *capacity*, not an allocation, in every accounting sense
    /// below (see claim_expert_zone on total_pages()).
    struct ExpertZoneRun {
        int gpu_idx = -1;
        int start_slab = -1;
        int num_slabs = 0;
        void* base = nullptr;  ///< kv_main_base + start_slab * slab_bytes
    };
    /// Per-GPU grant counters (inert zeros on unslabbed models).
    struct ExpertZoneStats {
        int64_t grants = 0;       ///< successful claims (both entry points)
        int64_t releases = 0;
        int64_t refusals = 0;     ///< headroom floor, or no contiguous run
        int granted_slabs = 0;    ///< currently outstanding
    };

    /// Grant the expert cache a contiguous run of `num_slabs` whole slabs,
    /// TOP-DOWN first-fit (the topmost such run). `extra_reserve_pages` is
    /// an ADDITIONAL headroom term on top of the INV-4.9f floor — the
    /// rebalancer's "leave this much room for KV growth" margin. Returns
    /// nullopt (and counts a refusal) when the pool is unslabbed, the
    /// request is empty, the headroom would be breached, or no run of that
    /// length exists.
    ///
    /// Granted slabs REMAIN counted in total_pages(gpu, kMain): a grant is
    /// reclaimable cache, not a capacity loss, so boot capacity and the
    /// admissible-ceiling semantics must not shrink underneath it — the
    /// non-retryable seq_create refusal class (a request whose whole-life
    /// demand exceeds the entire pool) must mean exactly what it meant
    /// before any grant existed.
    std::optional<ExpertZoneRun> claim_expert_zone(int gpu_idx, int num_slabs,
                                                   int64_t extra_reserve_pages);
    /// Targeted mirror claim: take exactly [start_slab, start_slab +
    /// num_slabs) with the same checks and stamping. Used by the
    /// rebalancer to mirror rank 0's chosen run onto the other TP ranks —
    /// glm5_next replicated KV claims slabs in LOCKSTEP BY INDEX, so
    /// per-rank-divergent free sets would collapse replicated-KV capacity
    /// to the INTERSECTION of the ranks' free slabs (the S4 /
    /// INV-KVT-14b lesson). Returns false (counting a refusal) unless
    /// EVERY slab in the span is unclaimed and the headroom holds; on
    /// false nothing is claimed.
    bool claim_expert_zone_at(int gpu_idx, int start_slab, int num_slabs,
                              int64_t extra_reserve_pages);
    /// Hand a grant back to the shared region. The run must match a
    /// recorded grant exactly (same start, same length).
    void release_expert_zone(int gpu_idx, int start_slab, int num_slabs);
    /// Longest run of consecutive UNCLAIMED slabs (0 when unslabbed or
    /// none free) — what the rebalancer sizes its next grant against.
    int largest_free_run(int gpu_idx) const;
    const ExpertZoneStats& expert_zone_stats(int gpu_idx) const;

    /// TEST-ONLY: claim one specific FREE slab of the shared region as an
    /// elastic indexer-K tenant (free through the ordinary free(handle)).
    /// Exists so fragmentation shapes can be CRAFTED deterministically —
    /// production indexer claims draw from the free-list LIFO and cannot
    /// target a slab. Returns nullopt when unslabbed / not elastic /
    /// slab not free.
    std::optional<PageHandle> claim_one_slab_for_test(int gpu_idx,
                                                      int slab_id);

    /// GF3.9: base of this GPU's KDA state region (nullptr when absent).
    /// The uniform-stride span the GF3.7 kernels' base + slot * stride
    /// indirection addresses: slot base = kda_state_base(g) +
    /// PageHandle::page_idx * kda_state_slot_bytes().
    void* kda_state_base(int gpu_idx) const;

    /// GF3.9: the KDA state slot layout (per-linear-layer recurrent/ring
    /// offsets), copied from VramLayout at construction — `enabled == false`
    /// on non-glm5_next models. The `l` its offset methods take is the
    /// DENSE linear-layer ordinal
    /// (ModelConfig::linear_attention_layer_ordinal).
    const KdaStateLayout& kda_layout() const { return kda_layout_; }

    /// V4 tier layout snapshot (entry bytes / per-tier formats / page
    /// geometry), copied from VramLayout at construction. `enabled == false`
    /// for non-V4 models. Authority for per-tier entry accounting in the
    /// attention stack (attention refactor V2 P2; see
    /// daemon/attention/kv_codec.h).
    const V4KvLayout& v4_layout() const { return v4_layout_; }

    // ── S1 shared-region slab geometry (TD-INDEXER-POOL-ELASTIC) ────────
    /// Slab byte size (0 when the model carries no indexer-K pool). A slab
    /// is one indexer page's claim unit AND an aligned run of
    /// pages_per_slab() flat kMain page indices (RADIX_SLAB_DESIGN §1).
    int64_t slab_bytes() const { return slab_bytes_; }
    /// kMain pages per slab (0 when unslabbed). On slabbed models the boot
    /// carve guarantees total_pages(gpu, kMain) % pages_per_slab() == 0.
    int pages_per_slab() const { return pages_per_slab_; }

    // ── S2 position-major per-sequence bump allocation
    //    (TD-INDEXER-POOL-ELASTIC + TD-SLAB-S2-GLM-MIN-FOOTPRINT,
    //    RADIX_SLAB_DESIGN §5 S2) ─────────────────────────────────────────
    /// Run key for allocations that carry no sequence identity (the KVS-2
    /// trash page, unit-test allocate() calls). Anonymous claims share one
    /// run per GPU — production KV append always carries (seq, layer).
    static constexpr uint64_t kAnonymousRunSeq = ~0ULL;

    /// S2 run-aware KV allocation. On slabbed models (pages_per_slab() > 0)
    /// a kMain claim routes through the sequence's POSITION-MAJOR BUMP RUN:
    /// one slab holds ALL LAYERS' pages for a contiguous token range of ONE
    /// sequence (the KV append call order — position-outer, layer-inner —
    /// makes the bump packing position-major by construction, so a token
    /// range that falls cold demotes as a set of COMPLETE slabs, and the
    /// minimum per-sequence footprint is ceil(kv_layers / pages_per_slab)
    /// slabs, ~1 — not kv_layers slabs as the reverted layer-major shape
    /// required, TD-SLAB-S2-GLM-MIN-FOOTPRINT). Placement order for a page
    /// at token_start:
    ///   1. position-matched hole — a freed slot in a run slab whose token
    ///      watermark range contains token_start (rewind refill and KVT
    ///      re-promotion return to their own cohort), newest slab first;
    ///   2. the frontier (newest) slab — holes, then the bump offset;
    ///   3. a fresh slab claim from the free-slab list (rule 3: a slab is
    ///      claimed by exactly one SEQUENCE);
    ///   4. a LOOSE page (a spec-range index recycled into kMain by
    ///      INV-4.9b promotion — physically outside the slab span; pins no
    ///      slab, so it beats cohort pollution);
    ///   5. pressure fallback — any free slot in any OLDER slab of the SAME
    ///      run, newest first (cohort pollution accepted over failing while
    ///      the sequence owns free space; never another sequence's slab).
    /// kSpeculation and unslabbed models behave exactly like allocate()/
    /// allocate_unreserved(). Sets sequence/layer/token-range metadata.
    /// Returns nullopt on exhaustion (or, when !unreserved, when free pages
    /// would drop to the INV-4.9f headroom reservation).
    std::optional<PageHandle> allocate_for_sequence(
        int gpu_idx, Pool pool, uint64_t seq_id, uint32_t layer_index,
        uint32_t token_start, uint32_t token_end, bool unreserved = false);

    /// S2 fragmentation telemetry for the slabbed kMain pool (zeros when
    /// unslabbed). fragmented_free_pages counts free pages INSIDE claimed
    /// slabs — allocatable only by the owning run (+ CoW colocation), i.e.
    /// the tail-fragmentation cost the design says to measure.
    struct KvFragmentation {
        int total_slabs = 0;
        int free_slabs = 0;         ///< unclaimed whole slabs
        int live_slabs = 0;         ///< claimed (>=1 live page)
        int used_pages = 0;         ///< live pages inside claimed slabs
        int fragmented_free_pages = 0;  ///< free pages inside claimed slabs
        int loose_free_pages = 0;   ///< recycled spec-range kMain pages
        int granted_slabs = 0;      ///< 44z: slabs lent to the expert cache
                                    ///  (a SUBSET of live_slabs — reclaimable
                                    ///  cache, not KV state)
    };
    KvFragmentation kv_fragmentation(int gpu_idx) const;

    /// Base pointer to the kv_main pool region on a given GPU.
    /// Returns nullptr if gpu_idx is out of range.
    void* kv_main_base(int gpu_idx) const;

    // ── Headroom reservation ────────────────────────────────────────────

    /// Domain-level inputs for computing per-pool page reservations.
    /// Each field maps to a named headroom term.  `configure_headroom()`
    /// computes the total and applies it per GPU.  Generalizable: future
    /// subsystems add fields here.
    struct HeadroomConfig {
        int max_concurrent_forks      = 0;  ///< → 1 CoW page per fork
        int max_concurrent_sequences  = 0;  ///< multiplier for page_growth
        int page_growth_chunk_pages   = 0;  ///< pages per growth event (KD-4e1)
        // Future: draft_kv_pages_per_seq, prefill_spill_pages, ...
    };

    /// Compute and apply headroom reservations from domain-level config.
    /// Applies uniformly to all GPUs for the kMain pool.
    void configure_headroom(const HeadroomConfig& config);

    /// Total reserved pages for a pool on a GPU.
    int reserved_pages(int gpu_idx, Pool pool) const;

    /// Free pages above the reservation (available for normal allocation).
    int available_pages(int gpu_idx, Pool pool) const;

    // ── Bulk operations ──────────────────────────────────────────────────

    /// Free all pages matching the given sequence_id on a GPU.
    /// Handles CoW correctly: only decrements refcount, freeing when it hits 0.
    void free_sequence(int gpu_idx, uint64_t sequence_id);

    // ── DCP token routing (INV-4.9e) ─────────────────────────────────────

    /// Set DCP routing config.  Must be called before DCP-aware methods.
    /// Asserts dcp_chunk_size is a multiple of page_size_tokens.
    void set_dcp_config(DcpConfig config);

    /// Current DCP configuration.
    const DcpConfig& dcp_config() const { return dcp_; }

    /// Which DCP rank owns a token position.  Pure function.
    /// Returns (token_pos / dcp_chunk_size) % dcp_size.
    /// Returns 0 when DCP is disabled (dcp_size <= 1).
    static int dcp_rank_for_token(uint32_t token_pos, const DcpConfig& dcp);

    /// GPU index that owns a token position under DCP.
    /// Maps: token → rank → tp_gpu_indices[rank].
    int dcp_gpu_for_token(uint32_t token_pos) const;

    /// Set metadata on pre-allocated pages to mark a DCP token range.
    /// Distributes [start_token, end_token) across handles, page_size_tokens
    /// per page.  Does NOT allocate or free — purely metadata assignment.
    void assign_dcp_range(int gpu_idx, uint64_t seq_id,
                          std::span<PageHandle> handles,
                          uint32_t start_token, uint32_t end_token);

    /// Allocate a kMain page for a token position, honoring the KV placement
    /// mode.  Sharded (dcp_.kv_sharded): owner-routed via dcp_gpu_for_token.
    /// Replicated (default at dcp>=2): delegates to allocate_main_replicated
    /// (INV-KV-REP).  Sets metadata.  When DCP is disabled, allocates on
    /// default_gpu.
    std::optional<PageHandle> allocate_for_dcp_append(
        uint64_t seq_id, uint32_t token_pos, uint32_t layer_index,
        int default_gpu = 0);

    /// INV-KV-REP (TD-KV-REPLICATED-PAGE-ALIAS fix): replicated-KV kMain
    /// allocation — claims the SAME physical page index on EVERY TP GPU in
    /// lockstep (canonical index popped from the rank-0 GPU's free list,
    /// erased from every mirror's free list; metadata + refcount mirrored).
    /// The returned handle carries the canonical (rank-0) GPU; the index is
    /// valid against every rank's kv_main base, which is exactly what the
    /// replicated block tables / slot mappings / seq_restore assume.
    /// free()/add_ref()/cow_copy() on the handle mirror across all TP GPUs.
    /// @param unreserved  bypass the headroom reservation (page growth, CoW).
    /// Returns std::nullopt when any TP GPU cannot supply the index.
    /// Delegates to allocate_replicated(Pool::kMain).
    std::optional<PageHandle> allocate_main_replicated(
        uint64_t seq_id, uint32_t token_start, uint32_t layer_index,
        bool unreserved = false);

    /// INV-KV-REP extended to Pool::kSpeculation (TD-KV-REPLICATED-SPEC fix):
    /// pool-general replicated-KV allocation — same lockstep same-index-on-
    /// every-TP-GPU discipline as allocate_main_replicated, drawing from the
    /// given pool's free list (kMain or kSpeculation; indexer K has its own
    /// replicated path in allocate_indexer_k_for_dcp).  build_kv_metadata's
    /// replicated branch replicates a draft sequence's page_idx to every
    /// rank's block table, so a speculation page allocated on ONE GPU only
    /// would alias physical pages exactly like the kMain bug.
    std::optional<PageHandle> allocate_replicated(
        uint64_t seq_id, uint32_t token_start, uint32_t layer_index,
        Pool pool, bool unreserved = false);

    /// Allocate indexer K page(s) with DCP mode awareness.
    /// Replicated mode: allocates on ALL TP GPUs (returns dcp_size
    /// handles), all-or-nothing. On the S4 ELASTIC pool each rank claims a
    /// whole slab from ITS OWN shared free-slab list — per-rank INDEPENDENT
    /// indices (the S1 lockstep-same-index rule is deliberately dropped:
    /// per-rank free-slab sets diverge under sharded KV and a mutual-index
    /// requirement collapses capacity to the set intersection — S4 gate
    /// finding; nothing consumes cross-rank index equality for indexer
    /// pages). Legacy fixed spans keep the S1 lockstep discipline (their
    /// free lists hold only symmetric indexer claims).
    /// Local mode: allocates only on the owning rank's GPU (1 handle).
    /// Non-DCP: 1 handle on the first tp_gpu or gpu 0.
    std::vector<PageHandle> allocate_indexer_k_for_dcp(
        uint64_t seq_id, uint32_t token_pos, uint32_t layer_index);

    /// Allocate bypassing headroom reservation (for cow_copy, page growth).
    /// Page growth is exactly what headroom protects (INV-4.9f).
    std::optional<PageHandle> allocate_unreserved(int gpu_idx, Pool pool);

private:
    /// Side pool: independent region + page size + free list (indexer K,
    /// V4 HCA bucket, V4 SWA bucket).  No promotion, no replication mirrors,
    /// no headroom reservation — those are main/spec-pool concepts.
    ///
    /// S1 (TD-INDEXER-POOL-ELASTIC): the indexer-K side pool is SLABBED —
    /// its pages are whole slabs of the shared [indexer | kMain] region:
    /// page_idx == slab index within the indexer span, physical stride
    /// `stride_bytes` = VramLayout::slab_bytes (>= bytes_per_page; the
    /// per-slab tail sliver is never addressed). allocate == slab claim,
    /// free-to-zero-refcount == whole-slab release; this free list IS the
    /// S1 free-slab list (KV slabs are boot-claimed until S2 migrates KV
    /// onto it). kHca/kSwa keep stride == bytes_per_page.
    struct SidePool {
        void* base = nullptr;
        int64_t bytes_per_page = 0;   ///< content bytes (CoW copy size)
        int64_t stride_bytes = 0;     ///< physical stride (== bytes_per_page
                                      ///  unless slabbed)
        int total = 0;
        std::vector<int> free;        ///< legacy fixed span only (EMPTY and
                                      ///  unused when elastic)
        std::vector<PageMeta> pages;
        /// S4 (TD-INDEXER-POOL-ELASTIC): elastic indexer-K pool on slabbed
        /// models — NO dedicated span exists. page_idx == kMain SLAB id
        /// over the shared region (base = kv_main base, stride =
        /// slab_bytes); allocate claims a whole free slab from the SHARED
        /// free-slab list (kv_free_slabs, honoring the INV-4.9f headroom
        /// floor), free at refcount 0 returns it whole. `total` = the
        /// region's slab count (capacity ceiling, shared with KV).
        bool elastic = false;
        /// TD-KDA-STATE-MAPPED-SLABS: elastic claim unit in CONTIGUOUS
        /// slabs (1 for the indexer; kda_unit_slabs_ for the mapped KDA
        /// state pool — page_idx is the run START, side_release returns
        /// the whole run). Claims with run_slabs > 1 go through
        /// allocate_kda_state only (generic allocate_side refuses: a LIFO
        /// pop cannot honor contiguity).
        int run_slabs = 1;
    };

    // ── S2 (TD-INDEXER-POOL-ELASTIC): position-major per-seq KV bump runs ─
    /// One slab of the kMain span: pages_per_slab_ contiguous flat page
    /// indices [id * pps, (id+1) * pps). Claimed by exactly ONE SEQUENCE
    /// (rule 3, position-major form): it holds ALL LAYERS' pages for a
    /// contiguous token range of that sequence. Cross-SEQUENCE tenancy is
    /// allowed only WITHIN a fork family (must-hold 3 of
    /// TD-SLAB-S2-GLM-MIN-FOOTPRINT): (a) a CoW split colocated with its
    /// source page (cow_copy doc), and (b) refcount-shared pages that
    /// outlive the claiming sequence — a fork child's refs keep the page
    /// (and so the slab) alive after the parent frees; the slab returns
    /// whole to the free-slab list when its LAST live page frees, so no
    /// unrelated tenant can ever pin it. tok_lo/tok_hi are insert-time
    /// position watermarks (never shrunk on free — a routing heuristic,
    /// not an occupancy fact) used for position-matched hole reuse.
    struct KvSlab {
        uint64_t owner_seq = 0;
        bool claimed = false;
        int used = 0;                ///< live pages
        int bump = 0;                ///< next never-used offset [0, pps]
        std::vector<int> holes;      ///< freed offsets below bump (LIFO)
        uint32_t tok_lo = UINT32_MAX;  ///< min token_start inserted
        uint32_t tok_hi = 0;           ///< max token_start inserted
    };
    /// A SEQUENCE's run: the slabs it claimed, in claim order. The LAST
    /// slab is the bump frontier (earlier slabs are bump-full by
    /// construction outside the pressure fallback; holes may exist
    /// anywhere). A dead sequence's run entry persists while fork-family
    /// descendants keep shared pages alive in its slabs; it receives no new
    /// allocations (a freed seq_id never allocates again).
    struct KvRun {
        std::vector<int> slab_ids;
    };
    using KvRunKey = uint64_t;  // sequence id (kAnonymousRunSeq for none)

    struct GpuPool {
        void* base = nullptr;            ///< kv_main pointer (main pool region)
        void* spec_base = nullptr;       ///< kv_speculation pointer (spec pool region)
        int64_t bytes_per_page = 0;
        int initial_main_pages = 0;
        int initial_spec_pages = 0;
        int total = 0;                   ///< initial_main + initial_spec
        std::vector<int> main_free;      ///< Free page indices (main pool stack;
                                         ///  UNUSED when kv_slabbed — S2)
        std::vector<int> spec_free;      ///< Free page indices (speculation pool stack)
        std::vector<PageMeta> pages;     ///< Metadata for all pages
        int main_reserved = 0;           ///< Headroom: allocate() refuses below this count
        int spec_reserved = 0;

        // S2 slabbed-kMain state (empty when the model is unslabbed).
        std::vector<KvSlab> kv_slabs;    ///< initial_main_pages / pps slabs
        std::vector<int> kv_free_slabs;  ///< LIFO of unclaimed slab ids
        std::vector<int> kv_loose_free;  ///< kMain-pool pages OUTSIDE the slab
                                         ///  span (INV-4.9b promoted spec-range
                                         ///  indices recycled into kMain)
        int kv_free_pages = 0;           ///< O(1): free-slab + in-slab + loose
        std::map<KvRunKey, KvRun> kv_runs;

        SidePool indexer_k;              ///< DSA / V4 lightning indexer tier
        SidePool hca;                    ///< V4 HCA main tier (empty otherwise)
        SidePool swa;                    ///< V4 SWA/raw tier (empty otherwise)
        /// GF3.8: KDA per-request state slots (glm5_next; empty otherwise).
        /// One page == one whole-request slot of VramLayout::kda.slot_bytes
        /// in a dedicated uniform-stride region (kernels address it as
        /// base + slot * stride — the GF3.7 plug-in surface). Never
        /// elastic: contiguity is what the kernel contract buys.
        SidePool kda_state;
        /// TD-KDA-MAPPED-FRAG: mapped-claim outcome counters (inert
        /// zeros in carve mode and on non-KDA models).
        KdaMappedStats kda_stats;
        /// 44z: outstanding expert-zone grants, start_slab -> num_slabs.
        /// std::map (not unordered) so iteration order is DETERMINISTIC —
        /// the rebalancer's reclaim order must be reproducible across
        /// ranks and across runs.
        std::map<int, int> expert_zone_runs;
        ExpertZoneStats ez_stats;
    };

    std::vector<int>& free_list(GpuPool& gpu, Pool pool);
    const std::vector<int>& free_list(const GpuPool& gpu, Pool pool) const;
    void return_to_free_list(GpuPool& gpu, int page_idx);

    // ── S2 slabbed-kMain internals ───────────────────────────────────────
    /// True when this GPU's kMain span is slab-managed (S2 active).
    bool kv_slabbed(const GpuPool& gpu) const {
        return pages_per_slab_ > 0 && !gpu.kv_slabs.empty();
    }
    /// Take a specific free page (hole or bump frontier) from its slab and
    /// update the slab's position watermarks with token_start. The slab
    /// must already be claimed. Asserts consistency (used by the INV-KV-REP
    /// mirror path — replicated ranks follow the canonical rank's choice
    /// with the same token_start, so their slab states evolve in lockstep).
    void kv_take_specific(GpuPool& gpu, int page_idx, uint32_t token_start);
    /// Claim a specific free slab for a sequence run (mirror path).
    void kv_claim_slab(GpuPool& gpu, int slab_id, uint64_t seq_id);
    /// S4 elastic side claims: take a whole free slab OUT of the shared
    /// free-slab list for a side tenant (indexer-K). The slab is marked
    /// fully used (used = bump = pages_per_slab) so no KV placement ever
    /// lands in it; kv_free_pages drops by a whole slab. Returns the slab
    /// id, or -1 when no slab is free or (honor_reserved) the claim would
    /// dip kv_free_pages below the INV-4.9f headroom floor.
    int kv_claim_side_slab(GpuPool& gpu, bool honor_reserved);
    /// Mirror-path variant: claim slab_id specifically (replicated
    /// lockstep-by-index). Caller verified it is free.
    void kv_claim_side_slab_at(GpuPool& gpu, int slab_id);
    /// Return a side-claimed slab whole to the shared free-slab list.
    void kv_release_side_slab(GpuPool& gpu, int slab_id);
    /// Reset a side page's meta at refcount 0 and return its storage
    /// (legacy: push to sp.free; elastic: release the slab).
    void side_release(GpuPool& gpu, SidePool& sp, int idx);
    /// Return one page to its slab; releases the slab whole when its last
    /// live page frees (rule 3 makes freeing — per-sequence or last
    /// fork-family ref — return whole slabs). Loose (spec-range) kMain
    /// pages go to kv_loose_free.
    void kv_return_page(GpuPool& gpu, int page_idx);
    /// Free space available to a sequence run for a page at token_start
    /// (or, when prefer_slab >= 0, inside that specific slab — CoW
    /// colocation): position-matched holes newest-slab-first, then the
    /// frontier slab (holes, bump). Returns the page index without
    /// claiming new slabs; -1 if none (steps 1-2 of the placement order).
    int kv_find_in_run(GpuPool& gpu, uint64_t seq_id, uint32_t token_start,
                       int prefer_slab) const;
    /// Older-slab pressure fallback (step 5): any free slot in an older
    /// slab of the run, newest first; -1 if none.
    int kv_find_in_run_pressure(GpuPool& gpu, uint64_t seq_id) const;
    /// Non-replicated S2 allocation: run free space → new slab → loose →
    /// older-slab pressure fallback (the allocate_for_sequence placement
    /// order). Honors the INV-4.9f reservation unless unreserved.
    std::optional<int> kv_alloc_page(GpuPool& gpu, uint64_t seq_id,
                                     uint32_t token_start, bool unreserved,
                                     int prefer_slab = -1);
    /// INV-KV-REP replicated S2 allocation: the canonical rank runs the
    /// kv_alloc_page policy with mirror-awareness at slab claim (candidate
    /// slab must be free on EVERY TP GPU; capacity = smallest rank) and
    /// every mirror follows the chosen index in lockstep. Returns the
    /// claimed index or nullopt (exhaustion / reservation).
    std::optional<int> kv_alloc_page_replicated(uint64_t seq_id,
                                                uint32_t token_start,
                                                bool unreserved,
                                                int prefer_slab = -1);

    /// Side-pool selector: kIndexerK/kHca/kSwa → pool; kMain/kSpeculation →
    /// nullptr.
    static SidePool* side_pool(GpuPool& gpu, Pool pool);
    static const SidePool* side_pool(const GpuPool& gpu, Pool pool);
    std::optional<PageHandle> allocate_side(int gpu_idx, Pool pool);

    /// Shared allocation from a main/spec free list with min_free threshold.
    static std::optional<PageHandle> allocate_from_pool(
        GpuPool& gpu, int gpu_idx, Pool pool,
        std::vector<int>& fl, int min_free);

    /// Physical pointer for a main/spec page, derived from the INDEX RANGE
    /// (idx < initial_main_pages → kv_main region, else kv_speculation),
    /// not from the pool tag: zero-copy promotion (INV-4.9b) migrates a
    /// spec-range index into the main free list, and its physical bytes
    /// stay in the speculation region.
    static void* page_ptr(const GpuPool& gpu, int page_idx);

    std::vector<GpuPool> gpus_;
    compute::DeviceBackend* copy_backend_ = nullptr;
    DcpConfig dcp_;
    KvCacheFormat format_ = KvCacheFormat::kSnapMlaFp8;
    V4KvLayout v4_layout_{};  // copied at construction (see v4_layout())
    KdaStateLayout kda_layout_{};  // GF3.9: copied at construction
    bool kda_mapped_ = false;      // TD-KDA-STATE-MAPPED-SLABS
    int kda_unit_slabs_ = 0;
    int admissible_ctx_tokens_ = 0;  // TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA       // slabs per per-layer unit when mapped
    int64_t slab_bytes_ = 0;  // S1 slab geometry (see slab_bytes())
    int pages_per_slab_ = 0;
};

}  // namespace layerstorm::memory
