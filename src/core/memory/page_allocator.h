#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "core/memory/vram_allocator.h"

namespace layerstorm::compute { class DeviceBackend; }

namespace layerstorm::memory {

// ── Pool type ────────────────────────────────────────────────────────────────

// kMain/kSpeculation share one page size (MLA main pool; V4: the CSA bucket).
// kIndexerK, kHca, kSwa are independent side pools with their own regions and
// page sizes (kHca/kSwa exist only for V4 models, V4-3c; kIndexerK serves DSA
// or the V4 Lightning-Indexer tier).
enum class Pool { kMain, kSpeculation, kIndexerK, kHca, kSwa };

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
    PageHandle cow_copy(PageHandle handle);

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

    /// V4 tier layout snapshot (entry bytes / per-tier formats / page
    /// geometry), copied from VramLayout at construction. `enabled == false`
    /// for non-V4 models. Authority for per-tier entry accounting in the
    /// attention stack (attention refactor V2 P2; see
    /// daemon/attention/kv_codec.h).
    const V4KvLayout& v4_layout() const { return v4_layout_; }

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
    /// Replicated mode: allocates on ALL TP GPUs (returns dcp_size handles).
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
    struct SidePool {
        void* base = nullptr;
        int64_t bytes_per_page = 0;
        int total = 0;
        std::vector<int> free;
        std::vector<PageMeta> pages;
    };

    struct GpuPool {
        void* base = nullptr;            ///< kv_main pointer (main pool region)
        void* spec_base = nullptr;       ///< kv_speculation pointer (spec pool region)
        int64_t bytes_per_page = 0;
        int initial_main_pages = 0;
        int initial_spec_pages = 0;
        int total = 0;                   ///< initial_main + initial_spec
        std::vector<int> main_free;      ///< Free page indices (main pool stack)
        std::vector<int> spec_free;      ///< Free page indices (speculation pool stack)
        std::vector<PageMeta> pages;     ///< Metadata for all pages
        int main_reserved = 0;           ///< Headroom: allocate() refuses below this count
        int spec_reserved = 0;

        SidePool indexer_k;              ///< DSA / V4 lightning indexer tier
        SidePool hca;                    ///< V4 HCA main tier (empty otherwise)
        SidePool swa;                    ///< V4 SWA/raw tier (empty otherwise)
    };

    std::vector<int>& free_list(GpuPool& gpu, Pool pool);
    const std::vector<int>& free_list(const GpuPool& gpu, Pool pool) const;
    void return_to_free_list(GpuPool& gpu, int page_idx);

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
};

}  // namespace layerstorm::memory
