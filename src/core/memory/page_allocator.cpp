#include "core/memory/page_allocator.h"

#include <cstring>
#include <stdexcept>

#include "core/device_backend.h"

namespace layerstorm::memory {

// ── PageAllocator ────────────────────────────────────────────────────────────

PageAllocator::PageAllocator(const VramAllocator& vram,
                             compute::DeviceBackend* copy_backend)
    : copy_backend_(copy_backend) {
    if (!copy_backend_) {
        throw std::invalid_argument(
            "PageAllocator: copy_backend must be non-null");
    }
    const auto& layout = vram.layout();
    format_ = layout.kv_cache_format;
    v4_layout_ = layout.v4;
    gpus_.resize(vram.gpu_count());

    for (int i = 0; i < vram.gpu_count(); ++i) {
        auto& gpu = gpus_[i];
        const auto& gpu_layout = layout.gpus[i];
        const auto& region = vram.region(i);

        gpu.base = region.kv_main;
        gpu.spec_base = region.kv_speculation;
        gpu.bytes_per_page = layout.kv_bytes_per_page;
        gpu.initial_main_pages = gpu_layout.kv_main_pages;
        gpu.initial_spec_pages = gpu_layout.kv_speculation_pages;
        gpu.total = gpu.initial_main_pages + gpu.initial_spec_pages;

        // Metadata for all pages, initialized to free (refcount=0)
        gpu.pages.resize(gpu.total);

        // Populate main free list: indices [0, initial_main_pages)
        gpu.main_free.reserve(gpu.initial_main_pages);
        for (int j = gpu.initial_main_pages - 1; j >= 0; --j) {
            gpu.main_free.push_back(j);
        }

        // Populate speculation free list: indices [initial_main_pages, total)
        gpu.spec_free.reserve(gpu.initial_spec_pages);
        for (int j = gpu.total - 1; j >= gpu.initial_main_pages; --j) {
            gpu.spec_free.push_back(j);
        }

        // Side pools (separate regions and page sizes).
        auto init_side = [](SidePool& sp, void* base, int64_t bpp, int total) {
            sp.base = base;
            sp.bytes_per_page = bpp;
            sp.total = total;
            sp.pages.resize(total);
            sp.free.reserve(total);
            for (int j = total - 1; j >= 0; --j) sp.free.push_back(j);
        };
        init_side(gpu.indexer_k, region.indexer_k,
                  layout.indexer_k_bytes_per_page, gpu_layout.indexer_k_pages);
        // V4 tier buckets (zero-sized / collapsed for non-V4 models, V4-3c).
        init_side(gpu.hca, region.kv_hca, layout.v4.hca_bytes_per_page,
                  gpu_layout.kv_hca_pages);
        init_side(gpu.swa, region.kv_swa, layout.v4.swa_bytes_per_page,
                  gpu_layout.kv_swa_pages);
    }
}

// ── Side-pool selector ───────────────────────────────────────────────────────

PageAllocator::SidePool* PageAllocator::side_pool(GpuPool& gpu, Pool pool) {
    switch (pool) {
        case Pool::kIndexerK: return &gpu.indexer_k;
        case Pool::kHca:      return &gpu.hca;
        case Pool::kSwa:      return &gpu.swa;
        case Pool::kMain:
        case Pool::kSpeculation: break;
    }
    return nullptr;
}

const PageAllocator::SidePool* PageAllocator::side_pool(const GpuPool& gpu,
                                                        Pool pool) {
    return side_pool(const_cast<GpuPool&>(gpu), pool);
}

// ── Allocation ───────────────────────────────────────────────────────────────

/// Shared allocation logic for main/spec pools.
/// \param min_free  Minimum free list size to allow allocation (0 = no limit).
std::optional<PageHandle> PageAllocator::allocate_from_pool(
        GpuPool& gpu, int gpu_idx, Pool pool,
        std::vector<int>& fl, int min_free) {
    if (static_cast<int>(fl.size()) <= min_free)
        return std::nullopt;

    int idx = fl.back();
    fl.pop_back();

    auto& meta = gpu.pages[idx];
    meta.refcount = 1;
    meta.pool = pool;
    meta.layer_index = 0;
    meta.sequence_id = 0;
    meta.token_start = 0;
    meta.token_end = 0;

    // Pointer derives from the INDEX RANGE, not the pool tag: a promoted
    // (INV-4.9b zero-copy) spec-range index recycled through the main free
    // list keeps its physical bytes in the speculation region.
    return PageHandle{gpu_idx, idx, page_ptr(gpu, idx), pool};
}

std::optional<PageHandle> PageAllocator::allocate_side(int gpu_idx, Pool pool) {
    auto& gpu = gpus_[gpu_idx];
    SidePool* sp = side_pool(gpu, pool);
    assert(sp);
    if (sp->free.empty()) return std::nullopt;

    int idx = sp->free.back();
    sp->free.pop_back();

    auto& meta = sp->pages[idx];
    meta.refcount = 1;
    meta.pool = pool;
    meta.layer_index = 0;
    meta.sequence_id = 0;
    meta.token_start = 0;
    meta.token_end = 0;

    void* ptr = static_cast<char*>(sp->base) +
                static_cast<int64_t>(idx) * sp->bytes_per_page;
    return PageHandle{gpu_idx, idx, ptr, pool};
}

std::optional<PageHandle> PageAllocator::allocate(int gpu_idx, Pool pool) {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    auto& gpu = gpus_[gpu_idx];

    if (side_pool(gpu, pool)) return allocate_side(gpu_idx, pool);

    const int reserved = (pool == Pool::kMain)  ? gpu.main_reserved
                       : (pool == Pool::kSpeculation) ? gpu.spec_reserved
                       : 0;
    return allocate_from_pool(gpu, gpu_idx, pool, free_list(gpu, pool), reserved);
}

std::optional<PageHandle> PageAllocator::allocate_unreserved(int gpu_idx, Pool pool) {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    // Side pools carry no headroom reservation — plain allocation.
    if (side_pool(gpus_[gpu_idx], pool)) return allocate_side(gpu_idx, pool);
    return allocate_from_pool(gpus_[gpu_idx], gpu_idx, pool,
                              free_list(gpus_[gpu_idx], pool), 0);
}

// ── Free ─────────────────────────────────────────────────────────────────────

void PageAllocator::free(PageHandle handle) {
    assert(handle.gpu_idx >= 0 &&
           handle.gpu_idx < static_cast<int>(gpus_.size()));
    auto& gpu = gpus_[handle.gpu_idx];

    if (SidePool* sp = side_pool(gpu, handle.pool)) {
        assert(handle.page_idx >= 0 && handle.page_idx < sp->total);
        auto& meta = sp->pages[handle.page_idx];
        assert(meta.refcount > 0);
        --meta.refcount;
        if (meta.refcount == 0) {
            meta.layer_index = 0;
            meta.sequence_id = 0;
            meta.token_start = 0;
            meta.token_end = 0;
            sp->free.push_back(handle.page_idx);
        }
        return;
    }

    assert(handle.page_idx >= 0 && handle.page_idx < gpu.total);
    auto& meta = gpu.pages[handle.page_idx];
    assert(meta.refcount > 0);

    // INV-KV-REP: replicated pages were claimed in lockstep on every TP GPU —
    // the free must mirror, or the mirrors leak and the lists desync.
    if (meta.replicated && dcp_.enabled()) {
        for (int gi : dcp_.tp_gpu_indices) {
            auto& m = gpus_[gi].pages[handle.page_idx];
            assert(m.refcount > 0);
            --m.refcount;
            if (m.refcount == 0) {
                return_to_free_list(gpus_[gi], handle.page_idx);
            }
        }
        return;
    }

    --meta.refcount;
    if (meta.refcount == 0) {
        return_to_free_list(gpu, handle.page_idx);
    }
}

// ── Promotion ────────────────────────────────────────────────────────────────

void PageAllocator::promote(PageHandle handle) {
    assert(handle.gpu_idx >= 0 &&
           handle.gpu_idx < static_cast<int>(gpus_.size()));
    auto& gpu = gpus_[handle.gpu_idx];
    assert(!side_pool(gpu, handle.pool));  // Side pools don't promote
    assert(handle.page_idx >= 0 && handle.page_idx < gpu.total);

    auto& meta = gpu.pages[handle.page_idx];
    assert(meta.refcount > 0);
    assert(meta.pool == Pool::kSpeculation);

    // INV-KV-REP (TD-KV-REPLICATED-SPEC): replicated speculation pages were
    // claimed in lockstep on every TP GPU — the promotion must flip the pool
    // on every mirror, or a later free would route the canonical page to the
    // main free list and the mirrors back to the spec free list (desync).
    if (meta.replicated && dcp_.enabled()) {
        for (int gi : dcp_.tp_gpu_indices) {
            auto& m = gpus_[gi].pages[handle.page_idx];
            assert(m.refcount > 0);
            assert(m.pool == Pool::kSpeculation);
            m.pool = Pool::kMain;
        }
        return;
    }

    meta.pool = Pool::kMain;
}

// ── Copy-on-write ────────────────────────────────────────────────────────────

void PageAllocator::add_ref(PageHandle handle) {
    assert(handle.gpu_idx >= 0 &&
           handle.gpu_idx < static_cast<int>(gpus_.size()));
    auto& gpu = gpus_[handle.gpu_idx];

    if (SidePool* sp = side_pool(gpu, handle.pool)) {
        assert(handle.page_idx >= 0 && handle.page_idx < sp->total);
        assert(sp->pages[handle.page_idx].refcount > 0);
        ++sp->pages[handle.page_idx].refcount;
        return;
    }

    assert(handle.page_idx >= 0 && handle.page_idx < gpu.total);
    assert(gpu.pages[handle.page_idx].refcount > 0);

    // INV-KV-REP: mirror the refcount on every TP GPU's replica.
    if (gpu.pages[handle.page_idx].replicated && dcp_.enabled()) {
        for (int gi : dcp_.tp_gpu_indices)
            ++gpus_[gi].pages[handle.page_idx].refcount;
        return;
    }

    ++gpu.pages[handle.page_idx].refcount;
}

PageHandle PageAllocator::cow_copy(PageHandle handle) {
    assert(handle.gpu_idx >= 0 &&
           handle.gpu_idx < static_cast<int>(gpus_.size()));
    auto& gpu = gpus_[handle.gpu_idx];

    if (SidePool* sp = side_pool(gpu, handle.pool)) {
        assert(handle.page_idx >= 0 && handle.page_idx < sp->total);
        auto& old_meta = sp->pages[handle.page_idx];
        assert(old_meta.refcount > 0);
        if (old_meta.refcount == 1) return handle;

        auto new_handle = allocate(handle.gpu_idx, handle.pool);
        if (!new_handle) {
            throw std::runtime_error(
                "PageAllocator::cow_copy: side pool exhausted");
        }
        copy_backend_->memcpy_d2d(new_handle->gpu_ptr, handle.gpu_ptr,
                 sp->bytes_per_page);

        auto& new_meta = sp->pages[new_handle->page_idx];
        new_meta.layer_index = old_meta.layer_index;
        new_meta.sequence_id = old_meta.sequence_id;
        new_meta.token_start = old_meta.token_start;
        new_meta.token_end = old_meta.token_end;

        --old_meta.refcount;
        if (old_meta.refcount == 0) {
            old_meta.layer_index = 0;
            old_meta.sequence_id = 0;
            old_meta.token_start = 0;
            old_meta.token_end = 0;
            sp->free.push_back(handle.page_idx);
        }
        return *new_handle;
    }

    assert(handle.page_idx >= 0 && handle.page_idx < gpu.total);

    auto& old_meta = gpu.pages[handle.page_idx];
    assert(old_meta.refcount > 0);

    if (old_meta.refcount == 1) {
        return handle;
    }

    // INV-KV-REP: replicated page split — allocate a replicated destination
    // from the SAME pool (kMain or kSpeculation, TD-KV-REPLICATED-SPEC) and
    // copy every TP GPU's replica (index-derived offsets against each rank's
    // region; plain D2D within each GPU).
    if (old_meta.replicated && dcp_.enabled()) {
        auto nh = allocate_replicated(
            old_meta.sequence_id, old_meta.token_start, old_meta.layer_index,
            old_meta.pool, /*unreserved=*/true);
        if (!nh) {
            throw std::runtime_error(
                "PageAllocator::cow_copy: replicated pool exhausted, "
                "cannot allocate new page");
        }
        for (int gi : dcp_.tp_gpu_indices) {
            auto& g = gpus_[gi];
            copy_backend_->memcpy_d2d(page_ptr(g, nh->page_idx),
                                      page_ptr(g, handle.page_idx),
                                      g.bytes_per_page);
            // allocate_replicated set token_end from the DCP page size;
            // preserve the source's (possibly partial) range instead.
            g.pages[nh->page_idx].token_end = old_meta.token_end;
        }
        for (int gi : dcp_.tp_gpu_indices) {
            auto& m = gpus_[gi].pages[handle.page_idx];
            assert(m.refcount > 0);
            --m.refcount;
            if (m.refcount == 0) {
                return_to_free_list(gpus_[gi], handle.page_idx);
            }
        }
        return *nh;
    }

    // Allocate new page from the same pool (bypass headroom — CoW is
    // exactly the kind of operation the headroom is protecting).
    auto new_handle = allocate_unreserved(handle.gpu_idx, old_meta.pool);
    if (!new_handle) {
        throw std::runtime_error(
            "PageAllocator::cow_copy: pool exhausted, cannot allocate new page");
    }

    // Copy GPU data
    copy_backend_->memcpy_d2d(new_handle->gpu_ptr, handle.gpu_ptr, gpu.bytes_per_page);

    // Copy metadata (allocate already set refcount=1 and pool)
    auto& new_meta = gpu.pages[new_handle->page_idx];
    new_meta.layer_index = old_meta.layer_index;
    new_meta.sequence_id = old_meta.sequence_id;
    new_meta.token_start = old_meta.token_start;
    new_meta.token_end = old_meta.token_end;

    // Decrement old refcount
    --old_meta.refcount;
    if (old_meta.refcount == 0) {
        return_to_free_list(gpu, handle.page_idx);
    }

    return *new_handle;
}

// ── Metadata access ──────────────────────────────────────────────────────────

PageMeta& PageAllocator::meta(PageHandle handle) {
    assert(handle.gpu_idx >= 0 &&
           handle.gpu_idx < static_cast<int>(gpus_.size()));
    auto& gpu = gpus_[handle.gpu_idx];
    if (SidePool* sp = side_pool(gpu, handle.pool)) {
        assert(handle.page_idx >= 0 && handle.page_idx < sp->total);
        return sp->pages[handle.page_idx];
    }
    assert(handle.page_idx >= 0 && handle.page_idx < gpu.total);
    return gpu.pages[handle.page_idx];
}

const PageMeta& PageAllocator::meta(PageHandle handle) const {
    assert(handle.gpu_idx >= 0 &&
           handle.gpu_idx < static_cast<int>(gpus_.size()));
    const auto& gpu = gpus_[handle.gpu_idx];
    if (const SidePool* sp = side_pool(gpu, handle.pool)) {
        assert(handle.page_idx >= 0 && handle.page_idx < sp->total);
        return sp->pages[handle.page_idx];
    }
    assert(handle.page_idx >= 0 && handle.page_idx < gpu.total);
    return gpu.pages[handle.page_idx];
}

// ── Queries ──────────────────────────────────────────────────────────────────

int PageAllocator::free_pages(int gpu_idx, Pool pool) const {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    if (const SidePool* sp = side_pool(gpus_[gpu_idx], pool)) {
        return static_cast<int>(sp->free.size());
    }
    return static_cast<int>(free_list(gpus_[gpu_idx], pool).size());
}

int PageAllocator::total_pages(int gpu_idx, Pool pool) const {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    const auto& gpu = gpus_[gpu_idx];
    if (const SidePool* sp = side_pool(gpu, pool)) return sp->total;
    return (pool == Pool::kMain) ? gpu.initial_main_pages
                                 : gpu.initial_spec_pages;
}

int PageAllocator::used_pages(int gpu_idx, Pool pool) const {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    const auto& gpu = gpus_[gpu_idx];
    if (const SidePool* sp = side_pool(gpu, pool)) {
        int count = 0;
        for (const auto& p : sp->pages) {
            if (p.refcount > 0) ++count;
        }
        return count;
    }
    int count = 0;
    for (const auto& p : gpu.pages) {
        if (p.pool == pool && p.refcount > 0) {
            ++count;
        }
    }
    return count;
}

int PageAllocator::gpu_count() const {
    return static_cast<int>(gpus_.size());
}

void* PageAllocator::kv_main_base(int gpu_idx) const {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size()))
        return nullptr;
    return gpus_[gpu_idx].base;
}

// ── Headroom reservation ────────────────────────────────────────────────────

void PageAllocator::configure_headroom(const HeadroomConfig& config) {
    const int cow_pages    = config.max_concurrent_forks;  // 1 page per fork
    const int growth_pages = config.max_concurrent_sequences
                           * config.page_growth_chunk_pages;
    const int total = cow_pages + growth_pages;

    for (auto& gpu : gpus_) {
        gpu.main_reserved = total;
        // Speculation pool gets the CoW term only (page growth targets kMain).
        gpu.spec_reserved = cow_pages;
    }
}

int PageAllocator::reserved_pages(int gpu_idx, Pool pool) const {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size())) return 0;
    const auto& gpu = gpus_[gpu_idx];
    if (pool == Pool::kMain) return gpu.main_reserved;
    if (pool == Pool::kSpeculation) return gpu.spec_reserved;
    return 0;
}

int PageAllocator::available_pages(int gpu_idx, Pool pool) const {
    const int free = free_pages(gpu_idx, pool);
    const int reserved = reserved_pages(gpu_idx, pool);
    return std::max(0, free - reserved);
}

// ── Bulk operations ──────────────────────────────────────────────────────────

void PageAllocator::free_sequence(int gpu_idx, uint64_t sequence_id) {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    auto& gpu = gpus_[gpu_idx];

    // Scan main/spec pages
    for (int i = 0; i < gpu.total; ++i) {
        auto& meta = gpu.pages[i];
        if (meta.sequence_id == sequence_id && meta.refcount > 0) {
            // INV-KV-REP: replicated pages are canonical-owned — free them
            // across all TP GPUs when scanning the canonical GPU, and never
            // from a mirror's scan (the canonical scan handles the mirrors).
            if (meta.replicated && dcp_.enabled()) {
                if (gpu_idx != dcp_.tp_gpu_indices[0]) continue;
                for (int gi : dcp_.tp_gpu_indices) {
                    auto& m = gpus_[gi].pages[i];
                    assert(m.refcount > 0);
                    --m.refcount;
                    if (m.refcount == 0) {
                        return_to_free_list(gpus_[gi], i);
                    }
                }
                continue;
            }
            --meta.refcount;
            if (meta.refcount == 0) {
                return_to_free_list(gpu, i);
            }
        }
    }

    // Scan side pools (indexer K + V4 HCA/SWA buckets)
    for (Pool p : {Pool::kIndexerK, Pool::kHca, Pool::kSwa}) {
        SidePool* sp = side_pool(gpu, p);
        for (int i = 0; i < sp->total; ++i) {
            auto& meta = sp->pages[i];
            if (meta.sequence_id == sequence_id && meta.refcount > 0) {
                --meta.refcount;
                if (meta.refcount == 0) {
                    meta.layer_index = 0;
                    meta.sequence_id = 0;
                    meta.token_start = 0;
                    meta.token_end = 0;
                    sp->free.push_back(i);
                }
            }
        }
    }
}

// ── DCP token routing ────────────────────────────────────────────────────

void PageAllocator::set_dcp_config(DcpConfig config) {
    if (config.enabled()) {
        assert(config.dcp_chunk_size >= 1);
        assert(config.page_size_tokens >= 1);
        assert(config.dcp_chunk_size % config.page_size_tokens == 0 &&
               "dcp_chunk_size must be a multiple of page_size_tokens");
        assert(static_cast<int>(config.tp_gpu_indices.size()) == config.dcp_size);
    }
    dcp_ = std::move(config);
}

int PageAllocator::dcp_rank_for_token(uint32_t token_pos, const DcpConfig& dcp) {
    if (dcp.dcp_size <= 1) return 0;
    return static_cast<int>((token_pos / dcp.dcp_chunk_size) % dcp.dcp_size);
}

int PageAllocator::dcp_gpu_for_token(uint32_t token_pos) const {
    if (!dcp_.enabled()) {
        return dcp_.tp_gpu_indices.empty() ? 0 : dcp_.tp_gpu_indices[0];
    }
    int rank = dcp_rank_for_token(token_pos, dcp_);
    return dcp_.tp_gpu_indices[rank];
}

void PageAllocator::assign_dcp_range(int gpu_idx, uint64_t seq_id,
                                      std::span<PageHandle> handles,
                                      uint32_t start_token, uint32_t end_token) {
    assert(start_token < end_token);
    int pst = dcp_.page_size_tokens;
    uint32_t pos = start_token;
    for (auto& h : handles) {
        assert(h.gpu_idx == gpu_idx);
        auto& m = meta(h);
        m.sequence_id = seq_id;
        m.token_start = pos;
        m.token_end = std::min(pos + static_cast<uint32_t>(pst), end_token);
        pos = m.token_end;
    }
}

std::optional<PageHandle> PageAllocator::allocate_for_dcp_append(
    uint64_t seq_id, uint32_t token_pos, uint32_t layer_index,
    int default_gpu) {
    // INV-KV-REP (TD-KV-REPLICATED-PAGE-ALIAS): replicated KV — every rank
    // holds the full KV, so the page must exist at the SAME index on every
    // TP GPU.  Owner routing here would pop identical indices from the TP
    // GPUs' independent free lists and alias physical pages in the
    // replicated block tables past dcp_chunk_size tokens.
    if (dcp_.enabled() && !dcp_.kv_sharded) {
        return allocate_main_replicated(seq_id, token_pos, layer_index,
                                        /*unreserved=*/false);
    }
    int gpu = dcp_.enabled() ? dcp_gpu_for_token(token_pos) : default_gpu;
    auto h = allocate(gpu, Pool::kMain);
    if (!h) return std::nullopt;
    auto& m = meta(*h);
    m.sequence_id = seq_id;
    m.token_start = token_pos;
    m.token_end = token_pos + static_cast<uint32_t>(dcp_.page_size_tokens);
    m.layer_index = layer_index;
    return h;
}

std::optional<PageHandle> PageAllocator::allocate_main_replicated(
    uint64_t seq_id, uint32_t token_start, uint32_t layer_index,
    bool unreserved) {
    return allocate_replicated(seq_id, token_start, layer_index, Pool::kMain,
                               unreserved);
}

std::optional<PageHandle> PageAllocator::allocate_replicated(
    uint64_t seq_id, uint32_t token_start, uint32_t layer_index,
    Pool pool, bool unreserved) {
    assert(dcp_.enabled());
    assert(!dcp_.tp_gpu_indices.empty());
    assert(pool == Pool::kMain || pool == Pool::kSpeculation);
    const int canonical = dcp_.tp_gpu_indices[0];
    assert(canonical >= 0 && canonical < static_cast<int>(gpus_.size()));
    auto& cgpu = gpus_[canonical];
    auto& cfl = free_list(cgpu, pool);
    const int reserved = (pool == Pool::kMain) ? cgpu.main_reserved
                                               : cgpu.spec_reserved;
    const int min_free = unreserved ? 0 : reserved;
    if (static_cast<int>(cfl.size()) <= min_free) return std::nullopt;

    // A candidate index is claimable only if it lies inside the POOL's index
    // range on the mirror (kMain: [0, initial_main); kSpeculation:
    // [initial_main, total)) and is free there.
    auto mirror_claimable = [pool](const GpuPool& mg, int idx) {
        const bool in_range = (pool == Pool::kMain)
            ? idx < mg.initial_main_pages
            : (idx >= mg.initial_main_pages && idx < mg.total);
        return in_range && mg.pages[idx].refcount == 0;
    };

    // Lockstep discipline keeps the canonical stack top free on every mirror;
    // the scan below is robustness against any desync (e.g. a smaller mirror
    // pool whose high indices never existed on it).
    int pos = -1;
    for (int i = static_cast<int>(cfl.size()) - 1; i >= 0; --i) {
        const int idx = cfl[i];
        bool ok = true;
        for (size_t g = 1; g < dcp_.tp_gpu_indices.size(); ++g) {
            if (!mirror_claimable(gpus_[dcp_.tp_gpu_indices[g]], idx)) {
                ok = false;
                break;
            }
        }
        if (ok) { pos = i; break; }
    }
    if (pos < 0) return std::nullopt;
    const int idx = cfl[pos];
    cfl.erase(cfl.begin() + pos);

    // Claim the same index on every mirror GPU's free list.
    for (size_t g = 1; g < dcp_.tp_gpu_indices.size(); ++g) {
        auto& mfl = free_list(gpus_[dcp_.tp_gpu_indices[g]], pool);
        auto it = std::find(mfl.begin(), mfl.end(), idx);
        assert(it != mfl.end() && "replicated KV free lists desynced");
        if (it != mfl.end()) mfl.erase(it);
    }

    for (int gi : dcp_.tp_gpu_indices) {
        auto& m = gpus_[gi].pages[idx];
        m.refcount = 1;
        m.pool = pool;
        m.replicated = true;
        m.sequence_id = seq_id;
        m.token_start = token_start;
        m.token_end =
            token_start + static_cast<uint32_t>(dcp_.page_size_tokens);
        m.layer_index = layer_index;
    }

    return PageHandle{canonical, idx, page_ptr(cgpu, idx), pool};
}

std::vector<PageHandle> PageAllocator::allocate_indexer_k_for_dcp(
    uint64_t seq_id, uint32_t token_pos, uint32_t layer_index) {
    std::vector<PageHandle> results;

    auto set_meta = [&](PageHandle& h) {
        auto& m = meta(h);
        m.sequence_id = seq_id;
        m.token_start = token_pos;
        m.layer_index = layer_index;
    };

    if (!dcp_.enabled()) {
        int gpu = dcp_.tp_gpu_indices.empty() ? 0 : dcp_.tp_gpu_indices[0];
        auto h = allocate(gpu, Pool::kIndexerK);
        if (h) {
            set_meta(*h);
            results.push_back(*h);
        }
        return results;
    }

    if (dcp_.indexer_k_sharded) {
        // Local mode: allocate only on the owning rank's GPU. Ownership is
        // round-robin by INDEXER PAGE (owner = (pos / indexer_page_tokens) %
        // dcp — page-atomic by construction; see DcpConfig doc), falling
        // back to KV-chunk routing only when the page size is unset.
        int gpu;
        if (dcp_.indexer_k_page_size_tokens > 0) {
            const int rank = static_cast<int>(
                (token_pos
                 / static_cast<uint32_t>(dcp_.indexer_k_page_size_tokens))
                % static_cast<uint32_t>(dcp_.dcp_size));
            gpu = dcp_.tp_gpu_indices[rank];
        } else {
            gpu = dcp_gpu_for_token(token_pos);
        }
        auto h = allocate(gpu, Pool::kIndexerK);
        if (h) {
            set_meta(*h);
            results.push_back(*h);
        }
    } else {
        // Replicated mode: allocate on ALL TP GPUs
        for (int gpu : dcp_.tp_gpu_indices) {
            auto h = allocate(gpu, Pool::kIndexerK);
            if (h) {
                set_meta(*h);
                results.push_back(*h);
            }
        }
    }
    return results;
}

// ── Private helpers ──────────────────────────────────────────────────────────

std::vector<int>& PageAllocator::free_list(GpuPool& gpu, Pool pool) {
    return (pool == Pool::kMain) ? gpu.main_free : gpu.spec_free;
}

const std::vector<int>& PageAllocator::free_list(const GpuPool& gpu,
                                                  Pool pool) const {
    return (pool == Pool::kMain) ? gpu.main_free : gpu.spec_free;
}

void* PageAllocator::page_ptr(const GpuPool& gpu, int page_idx) {
    if (page_idx < gpu.initial_main_pages) {
        return static_cast<char*>(gpu.base) +
               static_cast<int64_t>(page_idx) * gpu.bytes_per_page;
    }
    return static_cast<char*>(gpu.spec_base) +
           static_cast<int64_t>(page_idx - gpu.initial_main_pages) *
               gpu.bytes_per_page;
}

void PageAllocator::return_to_free_list(GpuPool& gpu, int page_idx) {
    auto& meta = gpu.pages[page_idx];
    meta.layer_index = 0;
    meta.sequence_id = 0;
    meta.token_start = 0;
    meta.token_end = 0;
    meta.replicated = false;  // INV-KV-REP marker must not leak to reuse

    free_list(gpu, meta.pool).push_back(page_idx);
}

}  // namespace layerstorm::memory
