#include "core/memory/page_allocator.h"

#include <spdlog/spdlog.h>

#include <algorithm>
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
    kda_layout_ = layout.kda;  // GF3.9: KDA state slot layout snapshot
    slab_bytes_ = layout.slab_bytes;
    pages_per_slab_ = layout.pages_per_slab;

    // ── S1 geometry gates (RADIX_SLAB_DESIGN §1 rule 2, fail-loud) ──
    if (slab_bytes_ > 0) {
        if (layout.kv_bytes_per_page <= 0 ||
            slab_bytes_ % layout.kv_bytes_per_page != 0 ||
            slab_bytes_ < layout.indexer_k_bytes_per_page) {
            throw std::logic_error(
                "PageAllocator: S1 slab geometry violates rule 2 — slab " +
                std::to_string(slab_bytes_) + " B must be a whole multiple "
                "of the kMain page (" +
                std::to_string(layout.kv_bytes_per_page) + " B) and hold "
                "one indexer page (" +
                std::to_string(layout.indexer_k_bytes_per_page) + " B)");
        }
    }

    gpus_.resize(vram.gpu_count());

    for (int i = 0; i < vram.gpu_count(); ++i) {
        auto& gpu = gpus_[i];
        const auto& gpu_layout = layout.gpus[i];
        const auto& region = vram.region(i);

        gpu.base = region.kv_main;
        gpu.spec_base = region.kv_speculation;
        // TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA: carry the boot-computed
        // single-request admissible-context ceiling (min over attention
        // hosts) so the admission fast-fail can quote it.
        if (gpu_layout.attention_host && gpu_layout.admissible_ctx_tokens > 0) {
            admissible_ctx_tokens_ = admissible_ctx_tokens_ > 0
                ? std::min(admissible_ctx_tokens_,
                           gpu_layout.admissible_ctx_tokens)
                : gpu_layout.admissible_ctx_tokens;
        }
        gpu.bytes_per_page = layout.kv_bytes_per_page;
        gpu.initial_main_pages = gpu_layout.kv_main_pages;
        gpu.initial_spec_pages = gpu_layout.kv_speculation_pages;
        gpu.total = gpu.initial_main_pages + gpu.initial_spec_pages;

        // Metadata for all pages, initialized to free (refcount=0)
        gpu.pages.resize(gpu.total);

        // S2 (TD-INDEXER-POOL-ELASTIC): on slabbed models the kMain span is
        // slab-managed — position-major per-SEQUENCE bump runs over whole
        // slabs (one slab = all layers of a contiguous token range) replace
        // the page-granular main free list (which stays EMPTY; the slab
        // structures below are the authority). Unslabbed models keep the
        // legacy page stack.
        const bool kv_slab_managed =
            pages_per_slab_ > 0 && gpu.initial_main_pages > 0;
        if (kv_slab_managed) {
            if (gpu.initial_main_pages % pages_per_slab_ != 0) {
                throw std::logic_error(
                    "PageAllocator GPU " + std::to_string(i) + ": kMain " +
                    std::to_string(gpu.initial_main_pages) +
                    " pages is not a whole number of slabs (" +
                    std::to_string(pages_per_slab_) +
                    " pages/slab) — S2 bump allocation needs the S1 "
                    "whole-slab quantization on every GPU");
            }
            const int n_slabs = gpu.initial_main_pages / pages_per_slab_;
            gpu.kv_slabs.resize(n_slabs);
            gpu.kv_free_slabs.reserve(n_slabs);
            for (int j = n_slabs - 1; j >= 0; --j)
                gpu.kv_free_slabs.push_back(j);
            gpu.kv_free_pages = gpu.initial_main_pages;
        } else {
            // Populate main free list: indices [0, initial_main_pages)
            gpu.main_free.reserve(gpu.initial_main_pages);
            for (int j = gpu.initial_main_pages - 1; j >= 0; --j) {
                gpu.main_free.push_back(j);
            }
        }

        // Populate speculation free list: indices [initial_main_pages, total)
        gpu.spec_free.reserve(gpu.initial_spec_pages);
        for (int j = gpu.total - 1; j >= gpu.initial_main_pages; --j) {
            gpu.spec_free.push_back(j);
        }

        // Side pools (separate regions and page sizes).
        auto init_side = [](SidePool& sp, void* base, int64_t bpp,
                            int64_t stride, int total) {
            sp.base = base;
            sp.bytes_per_page = bpp;
            sp.stride_bytes = stride > 0 ? stride : bpp;
            sp.total = total;
            sp.pages.resize(total);
            sp.free.reserve(total);
            for (int j = total - 1; j >= 0; --j) sp.free.push_back(j);
        };
        // S4 (TD-INDEXER-POOL-ELASTIC): on slabbed models the indexer-K
        // pool is ELASTIC — no dedicated span exists. An indexer page is a
        // whole slab of the SHARED region claimed from kv_free_slabs on
        // demand (admission reservation, INV-DSA-RESERVE) and returned
        // whole at refcount 0; page_idx == the kMain slab id, pointer =
        // kv_main base + id * slab_bytes. The legacy fixed span survives
        // only on unslabbed layouts.
        if (kv_slab_managed && layout.indexer_k_bytes_per_page > 0) {
            if (gpu_layout.indexer_k_pages > 0) {
                throw std::logic_error(
                    "PageAllocator GPU " + std::to_string(i) +
                    ": fixed indexer-K carve (" +
                    std::to_string(gpu_layout.indexer_k_pages) +
                    " pages) on a slabbed model — S4 deleted the boot "
                    "carve; the sizing must fold the indexer share into "
                    "kv_main and leave indexer_k_pages == 0");
            }
            auto& sp = gpu.indexer_k;
            sp.base = region.kv_main;
            sp.bytes_per_page = layout.indexer_k_bytes_per_page;
            sp.stride_bytes = slab_bytes_;
            sp.total = gpu.initial_main_pages / pages_per_slab_;
            sp.pages.resize(static_cast<size_t>(sp.total));
            sp.elastic = true;   // free list stays empty — claims come
                                 // from the shared kv_free_slabs
        } else {
            // S1 legacy fixed span: page stride = slab_bytes over the
            // shared-region head; page_idx is the span-relative slab index.
            init_side(gpu.indexer_k, region.indexer_k,
                      layout.indexer_k_bytes_per_page, slab_bytes_,
                      gpu_layout.indexer_k_pages);
        }
        // V4 tier buckets (zero-sized / collapsed for non-V4 models, V4-3c).
        init_side(gpu.hca, region.kv_hca, layout.v4.hca_bytes_per_page, 0,
                  gpu_layout.kv_hca_pages);
        init_side(gpu.swa, region.kv_swa, layout.v4.swa_bytes_per_page, 0,
                  gpu_layout.kv_swa_pages);
        // GF3.8: KDA per-request state pool (zero-sized / collapsed on
        // non-glm5_next models). One page == one whole-request slot;
        // stride == bytes_per_page == VramLayout::kda.slot_bytes, so the
        // pool is the uniform-stride span the GF3.7 kernels' base +
        // slot * stride indirection requires. Never a SCATTERED elastic
        // tenant — but TD-KDA-STATE-MAPPED-SLABS adds the MAPPED mode: the
        // pool becomes an elastic RUN tenant of the shared slab region
        // (one contiguous run of kda_unit_slabs_ whole slabs per
        // (request, linear layer); page_idx = run-start slab id; physical
        // stride slab_bytes — still one base + uniform stride per launch,
        // so the kernel contract is intact and the carve disappears).
        // TD-KDA-MAPPED-NONTP-GPUS: the mapped pool exists only on GPUs
        // that CARRY the state — `GpuVramLayout::attention_host`, the same
        // published predicate the sizing gated the mapped share on
        // (vram_allocator.cpp). An expert-only (non-TP) GPU hosts no
        // attention, no KV and no KDA state: its kMain is legitimately
        // EMPTY, so it takes the collapsed zero-sized branch below instead
        // of the unslabbed-model guard (which used to fire on EVERY GPU and
        // made every tp=1 + expert-host EP shape unbootable on the mapped
        // default).
        if (layout.kda.enabled && layout.kda.mapped
            && gpu_layout.attention_host) {
            if (!kv_slab_managed) {
                throw std::logic_error(
                    "PageAllocator GPU " + std::to_string(i) +
                    ": mapped KDA state requires slabbed kMain "
                    "(TD-KDA-STATE-MAPPED-SLABS; sizing must refuse "
                    "unslabbed models)");
            }
            if (gpu_layout.kda_state_slots > 0) {
                throw std::logic_error(
                    "PageAllocator GPU " + std::to_string(i) +
                    ": mapped KDA state with a dedicated carve (" +
                    std::to_string(gpu_layout.kda_state_slots) +
                    " slots) — the sizing must leave kda_state_slots == 0 "
                    "when VramLayout::kda.mapped is set");
            }
            kda_mapped_ = true;
            kda_unit_slabs_ = static_cast<int>(
                (layout.kda.per_layer_bytes + slab_bytes_ - 1)
                / slab_bytes_);
            auto& sp = gpu.kda_state;
            sp.base = region.kv_main;
            sp.bytes_per_page = layout.kda.per_layer_bytes;
            sp.stride_bytes = slab_bytes_;
            sp.total = gpu.initial_main_pages / pages_per_slab_;
            sp.pages.resize(static_cast<size_t>(sp.total));
            sp.elastic = true;   // free list stays empty — claims come
                                 // from the shared kv_free_slabs
            sp.run_slabs = kda_unit_slabs_;
            // Fail-loud parity with the carve's "not even ONE slot fits"
            // boot check: a shared region that cannot hold a single
            // request's mapped state (num_layers contiguous runs) would
            // refuse EVERY admission forever — say so at boot, not at the
            // first seq_create (mapped is the DEFAULT since 2026-08-31).
            const int64_t need_slabs =
                static_cast<int64_t>(layout.kda.num_layers)
                * kda_unit_slabs_;
            if (sp.total < need_slabs) {
                throw std::runtime_error(
                    "PageAllocator GPU " + std::to_string(i) +
                    ": shared slab region (" + std::to_string(sp.total) +
                    " slabs) cannot hold even ONE request's mapped KDA "
                    "state (" + std::to_string(need_slabs) +
                    " slabs = " + std::to_string(layout.kda.num_layers) +
                    " layers x " + std::to_string(kda_unit_slabs_) +
                    " slabs/unit) — a glm5_next engine cannot serve any "
                    "sequence without its recurrent state; free VRAM or "
                    "reduce pinned layers (TD-KDA-STATE-MAPPED-SLABS).");
            }
        } else {
            // Carve mode, non-KDA models, and — under mapped — every
            // expert-only GPU: kda_state_slots is 0 there, so this is the
            // zero-sized collapsed pool (TD-KDA-MAPPED-NONTP-GPUS).
            if (layout.kda.enabled && layout.kda.mapped
                && gpu_layout.kda_state_slots > 0) {
                throw std::logic_error(
                    "PageAllocator GPU " + std::to_string(i) +
                    ": mapped KDA state carved " +
                    std::to_string(gpu_layout.kda_state_slots) +
                    " slots on a GPU that hosts no attention — the sizing "
                    "must leave expert-only GPUs with no KDA state at all");
            }
            init_side(gpu.kda_state, region.kda_state,
                      layout.kda.slot_bytes, 0,
                      gpu_layout.kda_state_slots);
        }

        // S1 shared-region gates: the indexer span must abut kv_main
        // slab-exactly and the kMain span must be whole slabs — the
        // properties that make (slab, offset) an ordinary flat page index.
        if (slab_bytes_ > 0 && gpu_layout.indexer_k_pages > 0) {
            const auto span = static_cast<int64_t>(gpu_layout.indexer_k_pages)
                              * slab_bytes_;
            if (static_cast<char*>(region.kv_main) -
                    static_cast<char*>(region.indexer_k) != span) {
                throw std::logic_error(
                    "PageAllocator GPU " + std::to_string(i) +
                    ": indexer span (" + std::to_string(span) +
                    " B) does not abut kv_main — S1 shared region broken");
            }
            if (pages_per_slab_ > 0 &&
                gpu.initial_main_pages % pages_per_slab_ != 0) {
                throw std::logic_error(
                    "PageAllocator GPU " + std::to_string(i) + ": kMain " +
                    std::to_string(gpu.initial_main_pages) +
                    " pages is not a whole number of slabs (" +
                    std::to_string(pages_per_slab_) + " pages/slab)");
            }
        }
    }
}

// ── Side-pool selector ───────────────────────────────────────────────────────

PageAllocator::SidePool* PageAllocator::side_pool(GpuPool& gpu, Pool pool) {
    switch (pool) {
        case Pool::kIndexerK: return &gpu.indexer_k;
        case Pool::kHca:      return &gpu.hca;
        case Pool::kSwa:      return &gpu.swa;
        case Pool::kKdaState: return &gpu.kda_state;
        case Pool::kMain:
        case Pool::kSpeculation: break;
    }
    return nullptr;
}

const PageAllocator::SidePool* PageAllocator::side_pool(const GpuPool& gpu,
                                                        Pool pool) {
    return side_pool(const_cast<GpuPool&>(gpu), pool);
}

// ── S2 slabbed-kMain internals (TD-INDEXER-POOL-ELASTIC, RADIX_SLAB_DESIGN
//    §5 S2): position-major per-SEQUENCE bump runs over whole slabs ────────

void PageAllocator::kv_claim_slab(GpuPool& gpu, int slab_id,
                                  uint64_t seq_id) {
    auto& slab = gpu.kv_slabs[slab_id];
    assert(!slab.claimed && slab.used == 0 && slab.bump == 0 &&
           slab.holes.empty());
    slab.claimed = true;
    slab.owner_seq = seq_id;
    slab.tok_lo = UINT32_MAX;
    slab.tok_hi = 0;
    auto it = std::find(gpu.kv_free_slabs.begin(), gpu.kv_free_slabs.end(),
                        slab_id);
    assert(it != gpu.kv_free_slabs.end() && "S2 free-slab list desync");
    if (it != gpu.kv_free_slabs.end()) gpu.kv_free_slabs.erase(it);
    gpu.kv_runs[seq_id].slab_ids.push_back(slab_id);
}

// ── S4 elastic side-slab claims (TD-INDEXER-POOL-ELASTIC) ────────────────
// A side tenant (indexer-K) claims a WHOLE slab out of the shared free-slab
// list: marked fully used (used = bump = pps) so no KV placement — hole
// scan, frontier bump, fresh claim or pressure fallback — can ever land in
// it (it also joins no kv_run). kv_free_pages drops/rises by whole slabs,
// keeping the INV-4.9f O(1) count exact.

int PageAllocator::kv_claim_side_slab(GpuPool& gpu, bool honor_reserved) {
    if (gpu.kv_free_slabs.empty()) return -1;
    if (honor_reserved &&
        gpu.kv_free_pages - pages_per_slab_ < gpu.main_reserved)
        return -1;
    const int slab_id = gpu.kv_free_slabs.back();
    gpu.kv_free_slabs.pop_back();
    auto& slab = gpu.kv_slabs[slab_id];
    assert(!slab.claimed && slab.used == 0 && slab.bump == 0 &&
           slab.holes.empty());
    slab.claimed = true;
    slab.owner_seq = 0;               // side tenant — never a kv_run owner
    slab.used = pages_per_slab_;
    slab.bump = pages_per_slab_;
    slab.tok_lo = UINT32_MAX;
    slab.tok_hi = 0;
    gpu.kv_free_pages -= pages_per_slab_;
    return slab_id;
}

void PageAllocator::kv_claim_side_slab_at(GpuPool& gpu, int slab_id) {
    auto it = std::find(gpu.kv_free_slabs.begin(), gpu.kv_free_slabs.end(),
                        slab_id);
    assert(it != gpu.kv_free_slabs.end() && "S4 side-slab mirror desync");
    if (it == gpu.kv_free_slabs.end()) return;
    gpu.kv_free_slabs.erase(it);
    auto& slab = gpu.kv_slabs[slab_id];
    assert(!slab.claimed && slab.used == 0);
    slab.claimed = true;
    slab.owner_seq = 0;
    slab.used = pages_per_slab_;
    slab.bump = pages_per_slab_;
    slab.tok_lo = UINT32_MAX;
    slab.tok_hi = 0;
    gpu.kv_free_pages -= pages_per_slab_;
}

void PageAllocator::kv_release_side_slab(GpuPool& gpu, int slab_id) {
    auto& slab = gpu.kv_slabs[slab_id];
    assert(slab.claimed && slab.used == pages_per_slab_);
    slab.claimed = false;
    slab.owner_seq = 0;
    slab.used = 0;
    slab.bump = 0;
    slab.holes.clear();
    slab.tok_lo = UINT32_MAX;
    slab.tok_hi = 0;
    gpu.kv_free_slabs.push_back(slab_id);
    gpu.kv_free_pages += pages_per_slab_;
}

void PageAllocator::side_release(GpuPool& gpu, SidePool& sp, int idx) {
    auto& meta = sp.pages[idx];
    meta.layer_index = 0;
    meta.sequence_id = 0;
    meta.token_start = 0;
    meta.token_end = 0;
    if (sp.elastic) {
        if (&sp == &gpu.kda_state) {
            // TD-KDA-MAPPED-FRAG (b2): a freed STATE run's slabs go to the
            // free-list BOTTOM, not the LIFO top — otherwise the very next
            // KV/indexer single-slab claim (pop_back) would colonize the
            // just-freed run and split it (a hibernate frees ~num_layers
            // runs in one act; measured accelerator, see the ticket). The
            // slabs stay perfectly claimable by KV under pressure — they
            // are merely LAST in line, so state neighborhoods re-coalesce
            // instead of being re-fragmented. One O(free + n) front
            // insert per released unit.
            for (int j = 0; j < sp.run_slabs; ++j) {
                auto& slab = gpu.kv_slabs[idx + j];
                assert(slab.claimed && slab.used == pages_per_slab_);
                slab.claimed = false;
                slab.owner_seq = 0;
                slab.used = 0;
                slab.bump = 0;
                slab.holes.clear();
                slab.tok_lo = UINT32_MAX;
                slab.tok_hi = 0;
                gpu.kv_free_pages += pages_per_slab_;
            }
            std::vector<int> ids(static_cast<size_t>(sp.run_slabs));
            for (int j = 0; j < sp.run_slabs; ++j) ids[j] = idx + j;
            gpu.kv_free_slabs.insert(gpu.kv_free_slabs.begin(),
                                     ids.begin(), ids.end());
        } else {
            // Indexer: the S4 single-slab release, LIFO top (unchanged).
            for (int j = 0; j < sp.run_slabs; ++j)
                kv_release_side_slab(gpu, idx + j);
        }
    } else {
        sp.free.push_back(idx);
    }
}

void PageAllocator::kv_take_specific(GpuPool& gpu, int page_idx,
                                     uint32_t token_start) {
    assert(page_idx >= 0 && page_idx < gpu.initial_main_pages);
    auto& slab = gpu.kv_slabs[page_idx / pages_per_slab_];
    const int off = page_idx % pages_per_slab_;
    assert(slab.claimed);
    auto hit = std::find(slab.holes.begin(), slab.holes.end(), off);
    if (hit != slab.holes.end()) {
        slab.holes.erase(hit);
    } else {
        assert(off == slab.bump && "S2 slab bump desync");
        ++slab.bump;
    }
    ++slab.used;
    slab.tok_lo = std::min(slab.tok_lo, token_start);
    slab.tok_hi = std::max(slab.tok_hi, token_start);
    --gpu.kv_free_pages;
}

void PageAllocator::kv_return_page(GpuPool& gpu, int page_idx) {
    if (page_idx >= gpu.initial_main_pages) {
        // INV-4.9b: a promoted spec-range index recycled into kMain lives
        // physically outside the slab span — it stays a LOOSE page.
        gpu.kv_loose_free.push_back(page_idx);
        ++gpu.kv_free_pages;
        return;
    }
    const int slab_id = page_idx / pages_per_slab_;
    auto& slab = gpu.kv_slabs[slab_id];
    assert(slab.claimed && slab.used > 0);
    --slab.used;
    ++gpu.kv_free_pages;
    if (slab.used > 0) {
        slab.holes.push_back(page_idx % pages_per_slab_);
        return;
    }
    // Last live page gone — the slab returns WHOLE to the free list (rule
    // 3: freeing — the owning sequence or the last fork-family ref — never
    // strands a slab on dead neighbours; an unrelated tenant cannot pin it
    // because no unrelated tenant can ever be placed in it).
    slab.holes.clear();
    slab.bump = 0;
    slab.claimed = false;
    slab.tok_lo = UINT32_MAX;
    slab.tok_hi = 0;
    auto it = gpu.kv_runs.find(slab.owner_seq);
    if (it != gpu.kv_runs.end()) {
        auto& ids = it->second.slab_ids;
        ids.erase(std::remove(ids.begin(), ids.end(), slab_id), ids.end());
        if (ids.empty()) gpu.kv_runs.erase(it);
    }
    slab.owner_seq = 0;
    gpu.kv_free_slabs.push_back(slab_id);
}

int PageAllocator::kv_find_in_run(GpuPool& gpu, uint64_t seq_id,
                                  uint32_t token_start,
                                  int prefer_slab) const {
    const int pps = pages_per_slab_;
    auto from_slab = [&](int sid) -> int {
        const auto& slab = gpu.kv_slabs[sid];
        if (!slab.claimed) return -1;
        if (!slab.holes.empty()) return sid * pps + slab.holes.back();
        if (slab.bump < pps) return sid * pps + slab.bump;
        return -1;
    };
    // CoW colocation: free space in the source page's slab first (fork-
    // family cross-sequence tenancy — see cow_copy).
    if (prefer_slab >= 0 &&
        prefer_slab < static_cast<int>(gpu.kv_slabs.size())) {
        const int idx = from_slab(prefer_slab);
        if (idx >= 0) return idx;
    }
    auto it = gpu.kv_runs.find(seq_id);
    if (it == gpu.kv_runs.end()) return -1;
    const auto& ids = it->second.slab_ids;
    // 1. Position-matched hole, newest slab first: a rewind refill or KVT
    // re-promotion goes back into its own position cohort instead of
    // polluting the frontier slab with an out-of-range position.
    for (auto rit = ids.rbegin(); rit != ids.rend(); ++rit) {
        const auto& slab = gpu.kv_slabs[*rit];
        if (slab.claimed && !slab.holes.empty() &&
            slab.tok_lo <= token_start && token_start <= slab.tok_hi)
            return *rit * pps + slab.holes.back();
    }
    // 2. The frontier (newest) slab: holes, then the bump offset — the
    // append path. Older slabs are deliberately NOT touched here: filling
    // an old cohort's holes with frontier tokens would keep that slab
    // alive after its own range demotes (the layer-major failure mode,
    // transposed). They remain reachable via kv_find_in_run_pressure.
    if (!ids.empty()) {
        const int idx = from_slab(ids.back());
        if (idx >= 0) return idx;
    }
    return -1;
}

int PageAllocator::kv_find_in_run_pressure(GpuPool& gpu,
                                           uint64_t seq_id) const {
    const int pps = pages_per_slab_;
    auto it = gpu.kv_runs.find(seq_id);
    if (it == gpu.kv_runs.end()) return -1;
    const auto& ids = it->second.slab_ids;
    // Step 5 (pressure fallback): any free slot in any OLDER slab of the
    // SAME run, newest first — cohort pollution accepted over failing
    // while the sequence owns free space; never another sequence's slab.
    for (auto rit = ids.rbegin(); rit != ids.rend(); ++rit) {
        const auto& slab = gpu.kv_slabs[*rit];
        if (!slab.claimed) continue;
        if (!slab.holes.empty()) return *rit * pps + slab.holes.back();
        if (slab.bump < pps) return *rit * pps + slab.bump;
    }
    return -1;
}

std::optional<int> PageAllocator::kv_alloc_page(GpuPool& gpu, uint64_t seq_id,
                                                uint32_t token_start,
                                                bool unreserved,
                                                int prefer_slab) {
    // INV-4.9f headroom: same count semantics as the legacy free-list check.
    if (!unreserved && gpu.kv_free_pages <= gpu.main_reserved)
        return std::nullopt;
    // 1-2. Position-matched hole / frontier slab (kv_find_in_run doc).
    int idx = kv_find_in_run(gpu, seq_id, token_start, prefer_slab);
    // 3. Fresh slab claim (rule 3: one sequence per slab).
    if (idx < 0 && !gpu.kv_free_slabs.empty()) {
        const int slab_id = gpu.kv_free_slabs.back();
        kv_claim_slab(gpu, slab_id, seq_id);
        idx = slab_id * pages_per_slab_;  // bump == 0
    }
    if (idx >= 0) {
        kv_take_specific(gpu, idx, token_start);
        return idx;
    }
    // 4. Loose fallback: recycled spec-range kMain pages (outside the
    // slabs). Preferred over polluting an older slab's cohort — a loose
    // page pins no slab at all.
    if (!gpu.kv_loose_free.empty()) {
        const int loose = gpu.kv_loose_free.back();
        gpu.kv_loose_free.pop_back();
        --gpu.kv_free_pages;
        return loose;
    }
    // 5. Pressure fallback: older-slab free space of the SAME run.
    idx = kv_find_in_run_pressure(gpu, seq_id);
    if (idx >= 0) {
        kv_take_specific(gpu, idx, token_start);
        return idx;
    }
    return std::nullopt;
}

std::optional<int> PageAllocator::kv_alloc_page_replicated(
        uint64_t seq_id, uint32_t token_start, bool unreserved,
        int prefer_slab) {
    assert(dcp_.enabled() && !dcp_.tp_gpu_indices.empty());
    const int canonical = dcp_.tp_gpu_indices[0];
    auto& cgpu = gpus_[canonical];
    if (!unreserved && cgpu.kv_free_pages <= cgpu.main_reserved)
        return std::nullopt;
    // 1-2/3. Run free space, else claim a slab free on EVERY rank in
    // LOCKSTEP (INV-KV-REP applied to slabs; capacity = smallest rank).
    // Replicated ops all mirror with the same token_start, so the per-rank
    // slab states (including position watermarks) evolve identically and
    // the canonical rank's in-run choice is valid on every mirror.
    int idx = kv_find_in_run(cgpu, seq_id, token_start, prefer_slab);
    if (idx < 0) {
        int chosen = -1;
        for (int i = static_cast<int>(cgpu.kv_free_slabs.size()) - 1; i >= 0;
             --i) {
            const int sid = cgpu.kv_free_slabs[i];
            bool ok = true;
            for (size_t g = 1; g < dcp_.tp_gpu_indices.size(); ++g) {
                const auto& mg = gpus_[dcp_.tp_gpu_indices[g]];
                if (!kv_slabbed(mg) ||
                    sid >= static_cast<int>(mg.kv_slabs.size()) ||
                    mg.kv_slabs[sid].claimed) {
                    ok = false;
                    break;
                }
            }
            if (ok) { chosen = sid; break; }
        }
        if (chosen >= 0) {
            for (int gi : dcp_.tp_gpu_indices)
                kv_claim_slab(gpus_[gi], chosen, seq_id);
            idx = chosen * pages_per_slab_;
        }
    }
    if (idx >= 0) {
        for (int gi : dcp_.tp_gpu_indices)
            kv_take_specific(gpus_[gi], idx, token_start);
        return idx;
    }
    // 4. Loose fallback — the index must be loose on every mirror (promote/
    // free mirror across ranks, so the lists stay symmetric).
    for (int i = static_cast<int>(cgpu.kv_loose_free.size()) - 1; i >= 0;
         --i) {
        const int loose = cgpu.kv_loose_free[i];
        bool ok = true;
        for (size_t g = 1; g < dcp_.tp_gpu_indices.size(); ++g) {
            auto& mg = gpus_[dcp_.tp_gpu_indices[g]];
            if (std::find(mg.kv_loose_free.begin(), mg.kv_loose_free.end(),
                          loose) == mg.kv_loose_free.end()) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;
        for (int gi : dcp_.tp_gpu_indices) {
            auto& mg = gpus_[gi];
            auto it = std::find(mg.kv_loose_free.begin(),
                                mg.kv_loose_free.end(), loose);
            assert(it != mg.kv_loose_free.end());
            if (it != mg.kv_loose_free.end()) mg.kv_loose_free.erase(it);
            --mg.kv_free_pages;
        }
        return loose;
    }
    // 5. Pressure fallback: older-slab free space of the SAME run. The
    // canonical choice is valid on every mirror (lockstep slab states).
    idx = kv_find_in_run_pressure(cgpu, seq_id);
    if (idx >= 0) {
        for (int gi : dcp_.tp_gpu_indices)
            kv_take_specific(gpus_[gi], idx, token_start);
        return idx;
    }
    return std::nullopt;
}

PageAllocator::KvFragmentation PageAllocator::kv_fragmentation(
        int gpu_idx) const {
    KvFragmentation f{};
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    const auto& gpu = gpus_[gpu_idx];
    if (!kv_slabbed(gpu)) return f;
    f.total_slabs = static_cast<int>(gpu.kv_slabs.size());
    f.free_slabs = static_cast<int>(gpu.kv_free_slabs.size());
    for (const auto& slab : gpu.kv_slabs) {
        if (!slab.claimed) continue;
        ++f.live_slabs;
        f.used_pages += slab.used;
        f.fragmented_free_pages += pages_per_slab_ - slab.used;
    }
    f.loose_free_pages = static_cast<int>(gpu.kv_loose_free.size());
    // 44z: a SUBSET of live_slabs — slabs lent to the expert cache read as
    // live and fully used (side-tenant stamp), but they are reclaimable
    // cache rather than KV state, so the rebalancer needs them named apart.
    f.granted_slabs = gpu.ez_stats.granted_slabs;
    return f;
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
    int idx;
    if (sp->elastic) {
        // TD-KDA-STATE-MAPPED-SLABS: run claims (contiguous multi-slab
        // units) cannot come off the LIFO free list — they go through
        // allocate_kda_state exclusively.
        assert(sp->run_slabs == 1 &&
               "run-unit side pool: use allocate_kda_state");
        if (sp->run_slabs != 1) return std::nullopt;
        // S4: claim a whole slab from the shared free-slab list.
        idx = kv_claim_side_slab(gpu, /*honor_reserved=*/true);
        if (idx < 0) return std::nullopt;
    } else {
        if (sp->free.empty()) return std::nullopt;
        idx = sp->free.back();
        sp->free.pop_back();
    }

    auto& meta = sp->pages[idx];
    meta.refcount = 1;
    meta.pool = pool;
    meta.layer_index = 0;
    meta.sequence_id = 0;
    meta.token_start = 0;
    meta.token_end = 0;

    // Physical stride: slab_bytes for the slabbed indexer pool (S1),
    // bytes_per_page otherwise.
    void* ptr = static_cast<char*>(sp->base) +
                static_cast<int64_t>(idx) * sp->stride_bytes;
    return PageHandle{gpu_idx, idx, ptr, pool};
}

std::optional<PageHandle> PageAllocator::allocate(int gpu_idx, Pool pool) {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    auto& gpu = gpus_[gpu_idx];

    if (side_pool(gpu, pool)) return allocate_side(gpu_idx, pool);

    // S2: slabbed kMain claims with no sequence identity (trash page, unit
    // tests) share the per-GPU ANONYMOUS run. Production KV append carries
    // (seq, layer, token range) via allocate_for_sequence / the DCP entry
    // points; the run is keyed by SEQUENCE and packed position-major.
    if (pool == Pool::kMain && kv_slabbed(gpu)) {
        auto idx = kv_alloc_page(gpu, kAnonymousRunSeq, 0,
                                 /*unreserved=*/false);
        if (!idx) return std::nullopt;
        auto& meta = gpu.pages[*idx];
        meta.refcount = 1;
        meta.pool = pool;
        meta.layer_index = 0;
        meta.sequence_id = 0;
        meta.token_start = 0;
        meta.token_end = 0;
        return PageHandle{gpu_idx, *idx, page_ptr(gpu, *idx), pool};
    }

    const int reserved = (pool == Pool::kMain)  ? gpu.main_reserved
                       : (pool == Pool::kSpeculation) ? gpu.spec_reserved
                       : 0;
    return allocate_from_pool(gpu, gpu_idx, pool, free_list(gpu, pool), reserved);
}

std::optional<PageHandle> PageAllocator::allocate_unreserved(int gpu_idx, Pool pool) {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    // Side pools carry no headroom reservation — plain allocation.
    if (side_pool(gpus_[gpu_idx], pool)) return allocate_side(gpu_idx, pool);
    auto& gpu = gpus_[gpu_idx];
    if (pool == Pool::kMain && kv_slabbed(gpu)) {
        auto idx = kv_alloc_page(gpu, kAnonymousRunSeq, 0,
                                 /*unreserved=*/true);
        if (!idx) return std::nullopt;
        auto& meta = gpu.pages[*idx];
        meta.refcount = 1;
        meta.pool = pool;
        meta.layer_index = 0;
        meta.sequence_id = 0;
        meta.token_start = 0;
        meta.token_end = 0;
        return PageHandle{gpu_idx, *idx, page_ptr(gpu, *idx), pool};
    }
    return allocate_from_pool(gpu, gpu_idx, pool, free_list(gpu, pool), 0);
}

std::optional<PageHandle> PageAllocator::allocate_for_sequence(
        int gpu_idx, Pool pool, uint64_t seq_id, uint32_t layer_index,
        uint32_t token_start, uint32_t token_end, bool unreserved) {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    auto& gpu = gpus_[gpu_idx];
    assert(!side_pool(gpu, pool) &&
           "allocate_for_sequence serves the kMain/kSpeculation pools");

    std::optional<PageHandle> h;
    if (pool == Pool::kMain && kv_slabbed(gpu)) {
        auto idx = kv_alloc_page(gpu, seq_id, token_start, unreserved);
        if (!idx) return std::nullopt;
        auto& meta = gpu.pages[*idx];
        meta.refcount = 1;
        meta.pool = pool;
        h = PageHandle{gpu_idx, *idx, page_ptr(gpu, *idx), pool};
    } else {
        h = unreserved ? allocate_unreserved(gpu_idx, pool)
                       : allocate(gpu_idx, pool);
        if (!h) return std::nullopt;
    }
    auto& meta = gpu.pages[h->page_idx];
    meta.sequence_id = seq_id;
    meta.layer_index = layer_index;
    meta.token_start = token_start;
    meta.token_end = token_end;
    return h;
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
        if (meta.refcount == 0)
            side_release(gpu, *sp, handle.page_idx);
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

PageHandle PageAllocator::cow_copy(PageHandle handle, uint64_t dst_seq_id) {
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
        if (old_meta.refcount == 0)
            side_release(gpu, *sp, handle.page_idx);
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
        std::optional<PageHandle> nh;
        if (old_meta.pool == Pool::kMain && kv_slabbed(gpu)) {
            // S2 CoW placement (cow_copy doc): free space in the SOURCE
            // page's slab first, else the destination sequence's run.
            const uint64_t run_seq =
                dst_seq_id ? dst_seq_id : old_meta.sequence_id;
            const int prefer = handle.page_idx < gpu.initial_main_pages
                ? handle.page_idx / pages_per_slab_ : -1;
            auto idx = kv_alloc_page_replicated(
                run_seq, old_meta.token_start, /*unreserved=*/true, prefer);
            if (idx) {
                const int canonical = dcp_.tp_gpu_indices[0];
                for (int gi : dcp_.tp_gpu_indices) {
                    auto& m = gpus_[gi].pages[*idx];
                    m.refcount = 1;
                    m.pool = old_meta.pool;
                    m.replicated = true;
                    m.sequence_id = old_meta.sequence_id;
                    m.token_start = old_meta.token_start;
                    m.token_end = old_meta.token_end;
                    m.layer_index = old_meta.layer_index;
                }
                nh = PageHandle{canonical, *idx,
                                page_ptr(gpus_[canonical], *idx),
                                old_meta.pool};
            }
        } else {
            nh = allocate_replicated(old_meta.sequence_id,
                                     old_meta.token_start,
                                     old_meta.layer_index, old_meta.pool,
                                     /*unreserved=*/true);
        }
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
    // S2 (slabbed kMain): prefer free space in the SOURCE page's slab, else
    // the destination sequence's own run — see the cow_copy doc.
    std::optional<PageHandle> new_handle;
    if (old_meta.pool == Pool::kMain && kv_slabbed(gpu)) {
        const uint64_t run_seq =
            dst_seq_id ? dst_seq_id : old_meta.sequence_id;
        const int prefer = handle.page_idx < gpu.initial_main_pages
            ? handle.page_idx / pages_per_slab_ : -1;
        auto idx = kv_alloc_page(gpu, run_seq, old_meta.token_start,
                                 /*unreserved=*/true, prefer);
        if (idx) {
            auto& m = gpu.pages[*idx];
            m.refcount = 1;
            m.pool = old_meta.pool;
            new_handle = PageHandle{handle.gpu_idx, *idx,
                                    page_ptr(gpu, *idx), old_meta.pool};
        }
    } else {
        new_handle = allocate_unreserved(handle.gpu_idx, old_meta.pool);
    }
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
        // S4 elastic: claimable capacity = the shared pool's free slabs
        // (each side page is one whole slab). Run units (mapped KDA state)
        // report free-slab count / run size — an upper bound that ignores
        // contiguity (message/telemetry use only; the claim itself is the
        // truth).
        if (sp->elastic)
            return static_cast<int>(gpus_[gpu_idx].kv_free_slabs.size())
                 / std::max(sp->run_slabs, 1);
        return static_cast<int>(sp->free.size());
    }
    // S2: slabbed kMain — free-slab pages + in-slab free space + loose
    // pages, maintained O(1). NOTE: in-slab free space is allocatable only
    // by the owning sequence's run (+ CoW colocation); a NEW sequence can
    // draw only whole free slabs + loose pages (kv_fragmentation() splits
    // the count when the distinction matters).
    if (pool == Pool::kMain && kv_slabbed(gpus_[gpu_idx]))
        return gpus_[gpu_idx].kv_free_pages;
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

void* PageAllocator::kda_state_base(int gpu_idx) const {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size()))
        return nullptr;
    return gpus_[gpu_idx].kda_state.base;
}

// TEST-ONLY (TD-KDA-MAPPED-FRAG): claim one specific free slab as an
// elastic indexer tenant so unit tests can CRAFT fragmentation shapes
// deterministically. Same bookkeeping as allocate_side's elastic arm with
// a chosen id; free through the ordinary free(handle).
std::optional<PageHandle> PageAllocator::claim_one_slab_for_test(
        int gpu_idx, int slab_id) {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    auto& gpu = gpus_[gpu_idx];
    auto& sp = gpu.indexer_k;
    if (!sp.elastic || slab_id < 0
        || slab_id >= static_cast<int>(gpu.kv_slabs.size())
        || gpu.kv_slabs[slab_id].claimed)
        return std::nullopt;
    kv_claim_side_slab_at(gpu, slab_id);
    auto& m = sp.pages[slab_id];
    m.refcount = 1;
    m.pool = Pool::kIndexerK;
    m.sequence_id = 0;
    m.layer_index = 0;
    m.token_start = 0;
    m.token_end = 0;
    void* ptr = static_cast<char*>(sp.base)
              + static_cast<int64_t>(slab_id) * sp.stride_bytes;
    return PageHandle{gpu_idx, slab_id, ptr, Pool::kIndexerK};
}

// ── TD-KDA-STATE-MAPPED-SLABS: whole-sequence KDA state claim ──────────────
// One call = one GPU's complete claim for one sequence, all-or-nothing.
// Carve mode: the single whole-request slot (the GF3.8 path, unchanged).
// Mapped mode: num_layers per-layer units, each a CONTIGUOUS run of
// kda_unit_slabs_ whole slabs from the shared free-slab list — first-fit
// over the slab array, claimed exactly like S4 side slabs (used = bump =
// pps so no KV placement can land inside; joins no kv_run), with ONE
// deferred sweep of kv_free_slabs so the claim is O(slabs + free-list)
// rather than per-slab list surgery. Honors the INV-4.9f headroom floor
// up front (the whole demand, not per unit). Failure claims NOTHING and
// returns empty — the caller's retryable-exhaustion seam is unchanged.
std::vector<PageHandle> PageAllocator::allocate_kda_state(int gpu_idx,
                                                          uint64_t seq_id) {
    std::vector<PageHandle> out;
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    auto& gpu = gpus_[gpu_idx];
    if (!kda_mapped_) {
        auto h = allocate(gpu_idx, Pool::kKdaState);
        if (!h || !h->gpu_ptr) return out;
        meta(*h).sequence_id = seq_id;
        out.push_back(*h);
        return out;
    }
    const int units = kda_layout_.num_layers;
    const int n = kda_unit_slabs_;
    assert(units > 0 && n > 0);
    auto& stats = gpu.kda_stats;
    // Headroom: the whole demand must clear the reserve floor. A headroom
    // refusal is CAPACITY-class by definition (no scan happened).
    const int need_pages = units * n * pages_per_slab_;
    if (gpu.kv_free_pages - need_pages < gpu.main_reserved) {
        ++stats.refusals_capacity;
        stats.last_found_runs = -1;
        stats.last_needed_runs = units;
        stats.last_free_slabs = static_cast<int>(gpu.kv_free_slabs.size());
        stats.last_was_contiguity = false;
        return out;
    }
    // TD-KDA-MAPPED-FRAG (b2) tenant segregation: state runs are found
    // TOP-DOWN (descending slab ids) while KV/indexer single-slab claims
    // pop the boot free-slab LIFO from slab 0 upward — the two tenants
    // grow toward each other from opposite ends of the region instead of
    // interleaving from request one. First-fit from the top: scan for
    // `units` runs of `n` consecutive unclaimed slabs.
    std::vector<int> starts;
    starts.reserve(static_cast<size_t>(units));
    const int total_slabs = static_cast<int>(gpu.kv_slabs.size());
    int run = 0;
    for (int sid = total_slabs - 1;
         sid >= 0 && static_cast<int>(starts.size()) < units;
         --sid) {
        if (!gpu.kv_slabs[sid].claimed) {
            if (++run == n) {
                starts.push_back(sid);  // run occupies [sid, sid + n)
                run = 0;
            }
        } else {
            run = 0;
        }
    }
    if (static_cast<int>(starts.size()) < units) {
        // Refused. Discriminate: enough free SLABS but not enough RUNS is
        // fragmentation — the failure holder eviction cannot fix
        // (TD-KDA-MAPPED-RETRY-EFFECTIVE); the scan terminated naturally,
        // so found_runs is complete and the classification is free.
        const int free_slabs = static_cast<int>(gpu.kv_free_slabs.size());
        const bool contiguity = free_slabs >= units * n;
        stats.last_found_runs = static_cast<int>(starts.size());
        stats.last_needed_runs = units;
        stats.last_free_slabs = free_slabs;
        stats.last_was_contiguity = contiguity;
        if (contiguity) {
            ++stats.refusals_contiguity;
            spdlog::warn(
                "KDA mapped state: CONTIGUITY refusal on GPU {} — {} free "
                "slabs cover the {}-slab demand but only {}/{} unit runs "
                "of {} consecutive slabs exist (fragmentation; holder "
                "eviction returns bytes, not adjacency — "
                "TD-KDA-MAPPED-FRAG / TD-KDA-MAPPED-RETRY-EFFECTIVE; "
                "lifetime: {} contiguity vs {} capacity refusals, {} "
                "claims).",
                gpu_idx, free_slabs, units * n, starts.size(), units, n,
                stats.refusals_contiguity + 1, stats.refusals_capacity,
                stats.claims);
        } else {
            ++stats.refusals_capacity;
        }
        return out;
    }
    ++stats.claims;
    // Claim: mark every run slab (side-tenant shape: fully used, no owner
    // run), then ONE sweep restores the "free list holds only unclaimed
    // slabs" invariant.
    for (int st : starts) {
        for (int j = 0; j < n; ++j) {
            auto& slab = gpu.kv_slabs[st + j];
            assert(!slab.claimed && slab.used == 0 && slab.bump == 0 &&
                   slab.holes.empty());
            slab.claimed = true;
            slab.owner_seq = 0;  // side tenant — never a kv_run owner
            slab.used = pages_per_slab_;
            slab.bump = pages_per_slab_;
            slab.tok_lo = UINT32_MAX;
            slab.tok_hi = 0;
            gpu.kv_free_pages -= pages_per_slab_;
        }
    }
    gpu.kv_free_slabs.erase(
        std::remove_if(gpu.kv_free_slabs.begin(), gpu.kv_free_slabs.end(),
                       [&](int id) { return gpu.kv_slabs[id].claimed; }),
        gpu.kv_free_slabs.end());
    auto& sp = gpu.kda_state;
    out.reserve(static_cast<size_t>(units));
    for (int st : starts) {
        auto& m = sp.pages[st];
        assert(m.refcount == 0);
        m.refcount = 1;
        m.pool = Pool::kKdaState;
        m.sequence_id = seq_id;
        m.layer_index = 0;
        m.token_start = 0;
        m.token_end = 0;
        void* ptr = static_cast<char*>(sp.base)
                  + static_cast<int64_t>(st) * sp.stride_bytes;
        out.push_back(PageHandle{gpu_idx, st, ptr, Pool::kKdaState});
    }
    return out;
}

// ── 44z: expert-zone grants (fourth slab tenant class) ─────────────────────
// A grant is a contiguous run of whole slabs lent to the expert cache. It is
// claimed with the SAME side-tenant stamp the indexer and the mapped KDA
// state use (claimed, owner_seq 0, used = bump = pps, watermarks reset) — so
// no KV placement can ever land inside a granted slab, and every existing
// accounting path sees it as one live, fully-used slab.
//
// WHY TOP-DOWN: grants are the LONGEST-LIVED contiguous tenant of the region
// (a rebalance epoch, not a request), so they pack at the TOP — physically
// adjacent to the expert_streaming zone in the VRAM layout — with the
// churning KDA state runs beneath them and bottom-up KV/indexer claims below
// that. Same tenant-segregation family as TD-KDA-MAPPED-FRAG (b2): tenants
// with different lifetimes grow toward each other from opposite ends instead
// of interleaving from request one. The KDA claim's found_runs discriminator
// is the regression watch — if grants start fragmenting the state band it
// shows up there as CONTIGUITY-class refusals.
std::optional<PageAllocator::ExpertZoneRun> PageAllocator::claim_expert_zone(
        int gpu_idx, int num_slabs, int64_t extra_reserve_pages) {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size()))
        return std::nullopt;
    auto& gpu = gpus_[gpu_idx];
    if (!kv_slabbed(gpu) || num_slabs <= 0) return std::nullopt;
    auto& stats = gpu.ez_stats;

    // Headroom: the whole demand must clear the INV-4.9f floor PLUS the
    // caller's extra margin (the rebalancer's KV growth room).
    const int64_t need_pages =
        static_cast<int64_t>(num_slabs) * pages_per_slab_;
    if (static_cast<int64_t>(gpu.kv_free_pages) - need_pages
            < static_cast<int64_t>(gpu.main_reserved) + extra_reserve_pages) {
        ++stats.refusals;
        return std::nullopt;
    }

    // First-fit from the TOP: the topmost run of num_slabs consecutive
    // unclaimed slabs (reset-on-claimed, the allocate_kda_state scan shape).
    const int total_slabs = static_cast<int>(gpu.kv_slabs.size());
    int start = -1;
    int run = 0;
    for (int sid = total_slabs - 1; sid >= 0; --sid) {
        if (!gpu.kv_slabs[sid].claimed) {
            if (++run == num_slabs) {
                start = sid;  // run occupies [sid, sid + num_slabs)
                break;
            }
        } else {
            run = 0;
        }
    }
    if (start < 0) {
        ++stats.refusals;
        return std::nullopt;
    }

    // Stamp the side-tenant shape, then ONE deferred sweep restores the
    // "free list holds only unclaimed slabs" invariant.
    for (int j = 0; j < num_slabs; ++j) {
        auto& slab = gpu.kv_slabs[start + j];
        assert(!slab.claimed && slab.used == 0 && slab.bump == 0 &&
               slab.holes.empty());
        slab.claimed = true;
        slab.owner_seq = 0;  // side tenant — never a kv_run owner
        slab.used = pages_per_slab_;
        slab.bump = pages_per_slab_;
        slab.tok_lo = UINT32_MAX;
        slab.tok_hi = 0;
        gpu.kv_free_pages -= pages_per_slab_;
    }
    gpu.kv_free_slabs.erase(
        std::remove_if(gpu.kv_free_slabs.begin(), gpu.kv_free_slabs.end(),
                       [&](int id) { return gpu.kv_slabs[id].claimed; }),
        gpu.kv_free_slabs.end());

    gpu.expert_zone_runs[start] = num_slabs;
    ++stats.grants;
    stats.granted_slabs += num_slabs;
    void* base = static_cast<char*>(gpu.base)
               + static_cast<int64_t>(start) * slab_bytes_;
    return ExpertZoneRun{gpu_idx, start, num_slabs, base};
}

bool PageAllocator::claim_expert_zone_at(int gpu_idx, int start_slab,
                                         int num_slabs,
                                         int64_t extra_reserve_pages) {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size()))
        return false;
    auto& gpu = gpus_[gpu_idx];
    if (!kv_slabbed(gpu) || num_slabs <= 0) return false;
    auto& stats = gpu.ez_stats;
    const int total_slabs = static_cast<int>(gpu.kv_slabs.size());
    if (start_slab < 0 || start_slab + num_slabs > total_slabs) {
        ++stats.refusals;
        return false;
    }
    const int64_t need_pages =
        static_cast<int64_t>(num_slabs) * pages_per_slab_;
    if (static_cast<int64_t>(gpu.kv_free_pages) - need_pages
            < static_cast<int64_t>(gpu.main_reserved) + extra_reserve_pages) {
        ++stats.refusals;
        return false;
    }
    // All-or-nothing: verify the WHOLE span before touching anything, so a
    // refused mirror claim leaves the rank byte-identical to before.
    for (int j = 0; j < num_slabs; ++j) {
        if (gpu.kv_slabs[start_slab + j].claimed) {
            ++stats.refusals;
            return false;
        }
    }
    for (int j = 0; j < num_slabs; ++j) {
        auto& slab = gpu.kv_slabs[start_slab + j];
        assert(!slab.claimed && slab.used == 0 && slab.bump == 0 &&
               slab.holes.empty());
        slab.claimed = true;
        slab.owner_seq = 0;
        slab.used = pages_per_slab_;
        slab.bump = pages_per_slab_;
        slab.tok_lo = UINT32_MAX;
        slab.tok_hi = 0;
        gpu.kv_free_pages -= pages_per_slab_;
    }
    gpu.kv_free_slabs.erase(
        std::remove_if(gpu.kv_free_slabs.begin(), gpu.kv_free_slabs.end(),
                       [&](int id) { return gpu.kv_slabs[id].claimed; }),
        gpu.kv_free_slabs.end());
    gpu.expert_zone_runs[start_slab] = num_slabs;
    ++stats.grants;
    stats.granted_slabs += num_slabs;
    return true;
}

void PageAllocator::release_expert_zone(int gpu_idx, int start_slab,
                                        int num_slabs) {
    assert(gpu_idx >= 0 && gpu_idx < static_cast<int>(gpus_.size()));
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size())) return;
    auto& gpu = gpus_[gpu_idx];
    auto it = gpu.expert_zone_runs.find(start_slab);
    // Never corrupt the region on a bad hand-back: a mismatched release
    // would reset slabs some OTHER tenant owns. Loud and inert instead.
    assert(it != gpu.expert_zone_runs.end() &&
           "44z: release of an unknown expert-zone run");
    if (it == gpu.expert_zone_runs.end()) {
        spdlog::error(
            "expert zone: release of an unknown grant on GPU {} — start "
            "slab {}, {} slabs (no grant recorded at that start); ignoring.",
            gpu_idx, start_slab, num_slabs);
        return;
    }
    assert(it->second == num_slabs &&
           "44z: expert-zone release length mismatch");
    if (it->second != num_slabs) {
        spdlog::error(
            "expert zone: release length mismatch on GPU {} — grant at slab "
            "{} is {} slabs, caller returned {}; ignoring.",
            gpu_idx, start_slab, it->second, num_slabs);
        return;
    }
    // Reset exactly like side_release's KDA arm, then FRONT-INSERT the ids:
    // TD-KDA-MAPPED-FRAG (b2) — a freed run must be LAST in line so it
    // re-coalesces, instead of being colonized and split by the very next
    // single-slab KV/indexer claim (which pops the LIFO back).
    for (int j = 0; j < num_slabs; ++j) {
        auto& slab = gpu.kv_slabs[start_slab + j];
        assert(slab.claimed && slab.used == pages_per_slab_);
        slab.claimed = false;
        slab.owner_seq = 0;
        slab.used = 0;
        slab.bump = 0;
        slab.holes.clear();
        slab.tok_lo = UINT32_MAX;
        slab.tok_hi = 0;
        gpu.kv_free_pages += pages_per_slab_;
    }
    std::vector<int> ids(static_cast<size_t>(num_slabs));
    for (int j = 0; j < num_slabs; ++j) ids[j] = start_slab + j;
    gpu.kv_free_slabs.insert(gpu.kv_free_slabs.begin(), ids.begin(), ids.end());

    gpu.expert_zone_runs.erase(it);
    ++gpu.ez_stats.releases;
    gpu.ez_stats.granted_slabs -= num_slabs;
    assert(gpu.ez_stats.granted_slabs >= 0);
}

int PageAllocator::largest_free_run(int gpu_idx) const {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size())) return 0;
    const auto& gpu = gpus_[gpu_idx];
    if (!kv_slabbed(gpu)) return 0;
    int best = 0;
    int run = 0;
    for (int sid = static_cast<int>(gpu.kv_slabs.size()) - 1; sid >= 0; --sid) {
        if (!gpu.kv_slabs[sid].claimed) {
            if (++run > best) best = run;
        } else {
            run = 0;
        }
    }
    return best;
}

const PageAllocator::ExpertZoneStats& PageAllocator::expert_zone_stats(
        int gpu_idx) const {
    static const ExpertZoneStats kNone{};
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size())) return kNone;
    return gpus_[gpu_idx].ez_stats;
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
    for (Pool p : {Pool::kIndexerK, Pool::kHca, Pool::kSwa,
                   Pool::kKdaState}) {
        SidePool* sp = side_pool(gpu, p);
        for (int i = 0; i < sp->total; ++i) {
            auto& meta = sp->pages[i];
            if (meta.sequence_id == sequence_id && meta.refcount > 0) {
                --meta.refcount;
                if (meta.refcount == 0) side_release(gpu, *sp, i);
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
    // Sharded / non-DCP: owner-routed single-GPU claim. S2 routes it
    // through the sequence's position-major bump run of the owning GPU.
    int gpu = dcp_.enabled() ? dcp_gpu_for_token(token_pos) : default_gpu;
    return allocate_for_sequence(
        gpu, Pool::kMain, seq_id, layer_index, token_pos,
        token_pos + static_cast<uint32_t>(dcp_.page_size_tokens));
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

    // S2: slabbed kMain — the canonical rank runs the per-seq bump-run
    // policy, slabs claim in LOCKSTEP by slab index on every rank
    // (INV-KV-REP applied to slabs), and every mirror follows the chosen
    // page index. kSpeculation and unslabbed models keep the legacy scan.
    if (pool == Pool::kMain && kv_slabbed(cgpu)) {
        auto idx = kv_alloc_page_replicated(seq_id, token_start, unreserved);
        if (!idx) return std::nullopt;
        for (int gi : dcp_.tp_gpu_indices) {
            auto& m = gpus_[gi].pages[*idx];
            m.refcount = 1;
            m.pool = pool;
            m.replicated = true;
            m.sequence_id = seq_id;
            m.token_start = token_start;
            m.token_end =
                token_start + static_cast<uint32_t>(dcp_.page_size_tokens);
            m.layer_index = layer_index;
        }
        return PageHandle{canonical, *idx, page_ptr(cgpu, *idx), pool};
    }
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
        // Replicated mode: allocate on ALL TP GPUs, claiming the SAME
        // page/slab index on every rank in LOCKSTEP (S1, RADIX_SLAB_DESIGN
        // §5 — the INV-KV-REP argument applied to indexer slabs; capacity =
        // smallest rank). Scan the canonical rank's free list top-down for
        // an index free on every mirror — the lockstep discipline keeps the
        // stack tops aligned, the scan is robustness against desync.
        const int canonical = dcp_.tp_gpu_indices[0];
        SidePool* csp = side_pool(gpus_[canonical], Pool::kIndexerK);
        assert(csp);
        if (csp->elastic) {
            // S4 elastic: claims are PER-RANK INDEPENDENT — each rank takes
            // a whole slab from ITS OWN shared free-slab list (all-or-
            // nothing across the group; rollback on any rank's refusal).
            // The S1 lockstep-same-index discipline is deliberately DROPPED
            // here: in the shared pool the per-rank free-slab SETS diverge
            // (sharded-KV claims differ per rank), so requiring one mutual
            // index collapses capacity to the set INTERSECTION — measured
            // live on the S4 gate: admission refusals with 82/106 free
            // slabs per rank. Nothing consumes cross-rank index equality
            // for indexer pages: every executor consumer is pointer-table
            // indirected (indexer_page_row), metadata and frees are
            // per-handle per-GPU, and side pages never set the INV-KV-REP
            // `replicated` mirror flag. Capacity = each rank's own free
            // slabs (worst rank refuses first — the honest bound).
            for (int gpu : dcp_.tp_gpu_indices) {
                const int sid = kv_claim_side_slab(gpus_[gpu],
                                                   /*honor_reserved=*/true);
                if (sid < 0) {
                    for (auto& h : results) free(h);
                    results.clear();
                    return results;  // exhausted (caller evict-retries)
                }
                SidePool* sp = side_pool(gpus_[gpu], Pool::kIndexerK);
                auto& m = sp->pages[sid];
                m.refcount = 1;
                m.pool = Pool::kIndexerK;
                m.layer_index = 0;
                m.sequence_id = 0;
                m.token_start = 0;
                m.token_end = 0;
                void* ptr = static_cast<char*>(sp->base) +
                            static_cast<int64_t>(sid) * sp->stride_bytes;
                PageHandle h{gpu, sid, ptr, Pool::kIndexerK};
                set_meta(h);
                results.push_back(h);
            }
            return results;
        }
        {
            // Legacy fixed-span pool: the free lists contain only lockstep
            // indexer claims, so a mutual index always exists — keep the
            // S1 same-index discipline here.
            int pos = -1;
            for (int i = static_cast<int>(csp->free.size()) - 1; i >= 0;
                 --i) {
                const int cand = csp->free[i];
                bool ok = true;
                for (size_t g = 1; g < dcp_.tp_gpu_indices.size(); ++g) {
                    const SidePool* msp = side_pool(
                        gpus_[dcp_.tp_gpu_indices[g]], Pool::kIndexerK);
                    if (cand >= msp->total ||
                        msp->pages[cand].refcount != 0) {
                        ok = false;
                        break;
                    }
                }
                if (ok) { pos = i; break; }
            }
            if (pos < 0) return results;  // exhausted (caller evict-retries)
            const int idx = csp->free[pos];
            for (int gpu : dcp_.tp_gpu_indices) {
                SidePool* sp = side_pool(gpus_[gpu], Pool::kIndexerK);
                auto it = std::find(sp->free.begin(), sp->free.end(), idx);
                assert(it != sp->free.end() &&
                       "replicated indexer-K free lists desynced");
                if (it == sp->free.end()) {
                    // Roll back this group's claims — the caller must see
                    // all-or-nothing (ensure_indexer_pages frees partials).
                    for (auto& h : results) free(h);
                    results.clear();
                    return results;
                }
                sp->free.erase(it);
                auto& m = sp->pages[idx];
                m.refcount = 1;
                m.pool = Pool::kIndexerK;
                m.layer_index = 0;
                m.sequence_id = 0;
                m.token_start = 0;
                m.token_end = 0;
                void* ptr = static_cast<char*>(sp->base) +
                            static_cast<int64_t>(idx) * sp->stride_bytes;
                PageHandle h{gpu, idx, ptr, Pool::kIndexerK};
                set_meta(h);
                results.push_back(h);
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

    // S2: slabbed kMain pages return to their slab (whole-slab release at
    // zero live pages); promoted spec-range indices stay loose.
    if (meta.pool == Pool::kMain && kv_slabbed(gpu)) {
        kv_return_page(gpu, page_idx);
        return;
    }
    free_list(gpu, meta.pool).push_back(page_idx);
}

}  // namespace layerstorm::memory
