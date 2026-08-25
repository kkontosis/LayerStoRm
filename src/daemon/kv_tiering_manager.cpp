// GLM-25k: DSA-guided KV tiering manager — see kv_tiering_manager.h.
//
// CUDA-free TU (INV-GPU-1): all device work goes through DeviceBackend /
// StreamManager plus the kv_row_copy launch wrappers (CUDA-free headers).

#include "daemon/kv_tiering_manager.h"

#include "daemon/kv_shard_math.h"

#include "compute/kernels/kv/kv_row_copy.h"
#include "compute/stream_manager.h"
#include "core/cuda_hardware_query.h"
#include "core/device_backend.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace layerstorm::daemon {

namespace {
constexpr size_t kAlign = 4096;
size_t align_up(size_t v) { return (v + kAlign - 1) & ~(kAlign - 1); }
}  // namespace

// ── Construction ────────────────────────────────────────────────────────────

KvTieringManager::KvTieringManager(Options opts) : opts_(std::move(opts)) {
    const int R = std::max(opts_.dcp_size, 1);
    if (static_cast<int>(opts_.gpus.size()) != R
        || static_cast<int>(opts_.kv_main_bases.size()) != R) {
        throw std::invalid_argument(
            "KvTieringManager: gpus/kv_main_bases must have dcp_size entries");
    }
    if (opts_.stride_row <= 0 || opts_.stride_block <= 0
        || opts_.page_size <= 0 || opts_.index_topk <= 0
        || opts_.kv_layers <= 0) {
        throw std::invalid_argument("KvTieringManager: bad geometry");
    }
    if (!opts_.free_page) {
        throw std::invalid_argument("KvTieringManager: free_page required");
    }
    // TD-KVT-DCP-SHARDED: round-robin chunk ownership (INV-4.9e).  Every KV
    // page must be owned entirely by ONE rank, so the chunk must be a
    // page-size multiple (mirrors PageAllocator::set_dcp_config).
    sharded_ = opts_.kv_sharded && R >= 2;
    if (sharded_) {
        if (opts_.dcp_chunk_tokens <= 0
            || opts_.dcp_chunk_tokens % opts_.page_size != 0) {
            throw std::invalid_argument(
                "KvTieringManager: sharded KV requires dcp_chunk_tokens > 0 "
                "and a multiple of page_size");
        }
        ppc_ = opts_.dcp_chunk_tokens / opts_.page_size;
    }

    // TD-KVT-REPLICA-COLD-DEDUP: one cold copy per page under replicated
    // KV at dcp >= 2 (round-robin cold owner) instead of one per rank.
    dedup_ = opts_.replica_cold_dedup && !sharded_ && R >= 2;

    hot_slots_ = opts_.hot_buffer_slots > 0 ? opts_.hot_buffer_slots
                                            : 2 * opts_.index_topk;
    retention_tokens_ = hot_slots_;
    const int ITK = opts_.index_topk;
    const int PS = opts_.page_size;
    n_fake_pages_ = (ITK + PS - 1) / PS;
    // Cold pool per rank: host_to_device_ratio × per-layer hot buffer,
    // per layer, page-granular, shared free list across layers (and, at
    // TD-KVT-BATCH, across sequences — exhaustion skips demotion,
    // fail-safe).  Under dedup each rank stores only ~1/dcp of the cold
    // pages (round-robin owners), so the per-rank pool shrinks by dcp —
    // total capacity unchanged, pinned host RAM divided by dcp (the TD's
    // entire point).
    const auto per_layer_pages = static_cast<int>(
        (opts_.host_to_device_ratio * hot_slots_ + PS - 1) / PS);
    cold_pool_pages_ = per_layer_pages * opts_.kv_layers;
    if (dedup_) cold_pool_pages_ = (cold_pool_pages_ + R - 1) / R;

    const size_t row = static_cast<size_t>(opts_.stride_row);
    const size_t blk = static_cast<size_t>(opts_.stride_block);

    // TD-KVT-COHORT-BATCHED-MATERIALIZE: union-arm gates, read BEFORE the
    // per-rank allocation (the kill-switch skips the union staging).
    if (const char* rw = std::getenv("LS_KVT_COHORT_ROWWISE"))
        cohort_rowwise_ = (*rw == '1');
    union_cap_ = (!cohort_rowwise_ && opts_.cohort_rows_max > 1
                  && opts_.union_rows_max > 0)
        ? opts_.union_rows_max : 0;
    u_pages_cap_ = union_cap_ > 0 ? (union_cap_ + PS - 1) / PS : 0;

    ranks_.resize(R);
    cache_.assign(R, {});
    ctx_host_bt_.assign(R, nullptr);
    sel_.assign(static_cast<size_t>(R), {});
    pf_pending_.assign(static_cast<size_t>(R),
                       std::vector<uint8_t>(
                           static_cast<size_t>(opts_.kv_layers), 0));

    // IndexShare share groups (executor full/shared rule): layer l is FULL
    // iff the mask is empty OR (l < size && mask[l]); share_succ_[l] counts
    // the consecutive SHARED layers reusing full layer l's selection.
    // Empty mask ⇒ every layer full ⇒ no reuse skip, no lookahead prefetch.
    share_succ_.assign(static_cast<size_t>(opts_.kv_layers), 0);
    const auto layer_is_full = [&](int l) {
        return opts_.indexer_full_layers.empty()
            || (l < static_cast<int>(opts_.indexer_full_layers.size())
                && opts_.indexer_full_layers[static_cast<size_t>(l)]);
    };
    for (int l = 0; l < opts_.kv_layers; ++l) {
        if (!layer_is_full(l)) continue;
        int c = 0;
        for (int m = l + 1; m < opts_.kv_layers && !layer_is_full(m); ++m) ++c;
        share_succ_[static_cast<size_t>(l)] = c;
    }

    std::vector<int> iota_itk(ITK);
    for (int i = 0; i < ITK; ++i) iota_itk[i] = i;
    std::vector<int> iota_pages(n_fake_pages_);
    for (int i = 0; i < n_fake_pages_; ++i) iota_pages[i] = i;

    for (int r = 0; r < R; ++r) {
        auto* be = backend(r);
        if (!be) throw std::invalid_argument(
            "KvTieringManager: missing device backend for rank "
            + std::to_string(r));
        be->set_device();
        auto& rb = ranks_[r];

        // ── Device buffers ──
        rb.row_cache     = be->device_alloc(static_cast<size_t>(opts_.kv_layers)
                                            * hot_slots_ * row);
        for (auto& ms : rb.mat) {
            ms.scratch       = be->device_alloc(static_cast<size_t>(n_fake_pages_) * blk);
            ms.cold_incoming = be->device_alloc(static_cast<size_t>(ITK) * row);
            ms.dev_src_ptrs  = be->device_alloc(static_cast<size_t>(ITK) * sizeof(void*));
            ms.dev_scatter_ptrs = be->device_alloc(static_cast<size_t>(ITK) * sizeof(void*));
            ms.dev_scatter_idx  = be->device_alloc(static_cast<size_t>(ITK) * sizeof(int));
            if (!ms.scratch || !ms.cold_incoming || !ms.dev_src_ptrs
                || !ms.dev_scatter_ptrs || !ms.dev_scatter_idx) {
                throw std::runtime_error(
                    "KvTieringManager: device_alloc failed (rank "
                    + std::to_string(r) + ")");
            }
        }
        rb.dev_ident_indices = be->device_alloc(static_cast<size_t>(ITK) * sizeof(int));
        rb.dev_fake_bt = be->device_alloc(
            static_cast<size_t>(n_fake_pages_) * sizeof(int));
        rb.pf_incoming = be->device_alloc(static_cast<size_t>(ITK) * row);
        rb.dev_pf_scatter_ptrs =
            be->device_alloc(static_cast<size_t>(ITK) * sizeof(void*));
        rb.dev_pf_scatter_idx =
            be->device_alloc(static_cast<size_t>(ITK) * sizeof(int));
        if (!rb.row_cache
            || !rb.dev_ident_indices || !rb.dev_fake_bt || !rb.pf_incoming
            || !rb.dev_pf_scatter_ptrs || !rb.dev_pf_scatter_idx) {
            throw std::runtime_error(
                "KvTieringManager: device_alloc failed (rank "
                + std::to_string(r) + ")");
        }
        be->memcpy_h2d(rb.dev_ident_indices, iota_itk.data(),
                       static_cast<size_t>(ITK) * sizeof(int));
        be->memcpy_h2d(rb.dev_fake_bt, iota_pages.data(),
                       static_cast<size_t>(n_fake_pages_) * sizeof(int));

        // TD-KVT-COHORT-BATCHED-MATERIALIZE: union staging (single set) +
        // rewritten-index / seqlens / replicated-iota-block-table buffers.
        if (union_cap_ > 0) {
            const size_t CRMu =
                static_cast<size_t>(std::max(1, opts_.cohort_rows_max));
            auto& us = rb.umat;
            us.scratch = be->device_alloc(
                static_cast<size_t>(u_pages_cap_) * blk);
            us.cold_incoming = be->device_alloc(
                static_cast<size_t>(union_cap_) * row);
            us.dev_src_ptrs = be->device_alloc(
                static_cast<size_t>(union_cap_) * sizeof(void*));
            rb.dev_uidx = be->device_alloc(
                CRMu * static_cast<size_t>(ITK) * sizeof(int));
            rb.dev_useq = be->device_alloc(CRMu * sizeof(int));
            rb.dev_union_bt = be->device_alloc(
                CRMu * static_cast<size_t>(u_pages_cap_) * sizeof(int));
            if (!us.scratch || !us.cold_incoming || !us.dev_src_ptrs
                || !rb.dev_uidx || !rb.dev_useq || !rb.dev_union_bt) {
                throw std::runtime_error(
                    "KvTieringManager: union staging device_alloc failed "
                    "(rank " + std::to_string(r) + ")");
            }
            // Every cohort row shares the same identity fake pages: the
            // block table is CRMu replicated iota rows (row stride =
            // u_pages_cap_ = the view's max_blocks_per_seq).
            std::vector<int> ubt(CRMu * static_cast<size_t>(u_pages_cap_));
            for (size_t b = 0; b < CRMu; ++b)
                for (int p = 0; p < u_pages_cap_; ++p)
                    ubt[b * static_cast<size_t>(u_pages_cap_)
                        + static_cast<size_t>(p)] = p;
            be->memcpy_h2d(rb.dev_union_bt, ubt.data(),
                           ubt.size() * sizeof(int));
        }

        // ── Pinned host arena on the rank GPU's NUMA home node (P-22) ──
        // The materialize staging (stage + ptr/idx tables) exists TWICE —
        // ping-pong MatSets (see the header note): set s occupies
        // [s * set_span, (s+1) * set_span).
        const size_t set_stage_b = align_up(static_cast<size_t>(ITK) * row);
        const size_t set_ptrs_b  = align_up(static_cast<size_t>(ITK) * sizeof(void*));
        const size_t set_idx_b   = align_up(static_cast<size_t>(ITK) * sizeof(int));
        const size_t set_span    = set_stage_b + 2 * set_ptrs_b + set_idx_b;
        // TD-KVT-ADMISSION-UPFRONT: the selection staging holds a whole
        // chunk cohort's selection (cohort_rows_max x ITK indices +
        // cohort_rows_max lengths) when the cohort seam is enabled; the
        // legacy single-row layout is the CRM == 1 special case.
        const size_t CRM = static_cast<size_t>(
            std::max(1, opts_.cohort_rows_max));
        const size_t off_idx     = align_up(2 * set_span);
        const size_t off_len     = align_up(off_idx + CRM * static_cast<size_t>(ITK) * sizeof(int));
        const size_t off_pf_stage = align_up(off_len
                                             + std::max<size_t>(64, CRM * sizeof(int)));
        const size_t off_pf_scatp = align_up(off_pf_stage + static_cast<size_t>(ITK) * row);
        const size_t off_pf_scati = align_up(off_pf_scatp + static_cast<size_t>(ITK) * sizeof(void*));
        const size_t off_cold    = align_up(off_pf_scati + static_cast<size_t>(ITK) * sizeof(int));
        size_t total             = off_cold
                                 + static_cast<size_t>(cold_pool_pages_) * blk;
        // TD-KVT-COHORT-BATCHED-MATERIALIZE: union staging tail (stage +
        // src-ptr table + rewritten indices + per-row seqlens).  Zero-cost
        // when the arm is disabled (legacy layout unchanged).
        size_t off_u_stage = 0, off_u_ptrs = 0, off_u_idx = 0, off_u_len = 0;
        if (union_cap_ > 0) {
            const size_t u_cap = static_cast<size_t>(union_cap_);
            off_u_stage = align_up(total);
            off_u_ptrs  = align_up(off_u_stage + u_cap * row);
            off_u_idx   = align_up(off_u_ptrs + u_cap * sizeof(void*));
            off_u_len   = align_up(off_u_idx
                                   + CRM * static_cast<size_t>(ITK)
                                         * sizeof(int));
            total       = off_u_len + std::max<size_t>(64, CRM * sizeof(int));
        }

        char* base = nullptr;
        const int gpu_pos = opts_.gpus[r].position;
        if (opts_.numa_manager) {
            rb.host_arena = opts_.numa_manager->allocate_for_gpu(total, gpu_pos);
            base = static_cast<char*>(rb.host_arena.data);
            rb.numa_node = rb.host_arena.numa_node;
            const int rc = core::host_register_pinned_portable(base, total);
            rb.host_registered = (rc == 0);
            if (!rb.host_registered) {
                spdlog::warn("KvTiering: host_register failed (rank {}, err {})"
                             " — cold pool unpinned (slow but correct)", r, rc);
            }
        } else {
            base = static_cast<char*>(be->host_alloc_pinned(total));
            if (!base) throw std::runtime_error(
                "KvTieringManager: host_alloc_pinned failed");
            rb.host_arena.data = base;
            rb.host_arena.size = total;
            rb.host_arena.numa_node = -1;
            rb.host_registered = false;  // cudaHostAlloc'd — freed via backend
            rb.numa_node = -1;
        }
        for (int s = 0; s < 2; ++s) {
            char* sb = base + static_cast<size_t>(s) * set_span;
            auto& ms = rb.mat[s];
            ms.h_stage        = sb;
            ms.h_src_ptrs     = reinterpret_cast<const void**>(sb + set_stage_b);
            ms.h_scatter_ptrs = reinterpret_cast<const void**>(
                sb + set_stage_b + set_ptrs_b);
            ms.h_scatter_idx  = reinterpret_cast<int*>(
                sb + set_stage_b + 2 * set_ptrs_b);
        }
        rb.h_indices      = reinterpret_cast<int*>(base + off_idx);
        rb.h_topk_len     = reinterpret_cast<int*>(base + off_len);
        rb.h_pf_stage     = base + off_pf_stage;
        rb.h_pf_scatter_ptrs = reinterpret_cast<const void**>(base + off_pf_scatp);
        rb.h_pf_scatter_idx  = reinterpret_cast<int*>(base + off_pf_scati);
        rb.cold_base      = base + off_cold;
        if (union_cap_ > 0) {
            rb.umat.h_stage    = base + off_u_stage;
            rb.umat.h_src_ptrs =
                reinterpret_cast<const void**>(base + off_u_ptrs);
            // umat.h_scatter_* stay null: union gathers never cache-insert.
            rb.h_uidx = reinterpret_cast<int*>(base + off_u_idx);
            rb.h_useq = reinterpret_cast<int*>(base + off_u_len);
        }

        rb.cold_free.reserve(cold_pool_pages_);
        for (int s = cold_pool_pages_ - 1; s >= 0; --s) rb.cold_free.push_back(s);
        rb.cold_ref.assign(static_cast<size_t>(cold_pool_pages_), 0);

        // ── Streams / events ──
        // TD-KVT-H2D-CONTENTION: the tiering H2D stream is ALWAYS a
        // dedicated owned stream — on the shared kH2dTransfer a ~1 MB cold
        // burst (which the attention stream waits on) could queue behind a
        // multi-copy backlog of ~18 MB expert weights (ms-scale).  A
        // dedicated stream bounds the stall to at most the one expert copy
        // already in flight on the link.  Demotion D2H stays on the shared
        // kD2hTransfer (background; completion-polled, never stream-waited
        // by compute).
        rb.h2d_stream = be->create_stream();
        if (opts_.stream_manager) {
            rb.d2h_stream = opts_.stream_manager->stream(
                gpu_pos, compute::StreamId::kD2hTransfer);
            rb.owns_d2h_stream = false;
        } else {
            rb.d2h_stream = be->create_stream();
            rb.owns_d2h_stream = true;
        }
        rb.ev_sync       = be->create_event();
        rb.ev_attn_order = be->create_event();
        rb.ev_pf         = be->create_event();
        rb.ev_pf_order   = be->create_event();
        for (auto& ms : rb.mat) {
            ms.ev_h2d      = be->create_event();
            ms.ev_mat_done = be->create_event();
        }
        if (union_cap_ > 0) {
            rb.umat.ev_h2d      = be->create_event();
            rb.umat.ev_mat_done = be->create_event();
            rb.ev_uidx          = be->create_event();
        }

        // ── Per-layer row caches ──
        cache_[r].resize(opts_.kv_layers);
        for (auto& lc : cache_[r]) {
            lc.slot_pos.assign(hot_slots_, -1);
            lc.slot_seq.assign(hot_slots_, 0);
            lc.slot_use.assign(hot_slots_, 0);
            lc.slot_prefetched.assign(hot_slots_, 0);
        }
    }
    layer_dense_.assign(static_cast<size_t>(opts_.kv_layers), 0);
    // TD-KVT-ADMISSION-UPFRONT hybrid gate override (strict identity runs).
    if (const char* ca = std::getenv("LS_KVT_COHORT_ALWAYS"))
        cohort_always_ = (*ca == '1');
    // Debug oracle: after every materialize gather, byte-verify the scratch
    // against its sources and every cold-classified row against the pinned
    // cold-pool authority.  Massive sync overhead — diagnostics only.
    if (const char* vf = std::getenv("LS_KVT_VERIFY"))
        verify_ = (*vf == '1');

    spdlog::info("KvTiering: enabled — hot_buffer_slots={} (retention {} tok), "
                 "cold pool {} pages/rank ({:.1f} MiB pinned/rank, node[s] {}),"
                 " fake pages {}, row {} B, kv={}",
                 hot_slots_, retention_tokens_, cold_pool_pages_,
                 static_cast<double>(cold_pool_pages_) * blk / (1024.0 * 1024.0),
                 ranks_.empty() ? -1 : ranks_[0].numa_node, n_fake_pages_,
                 opts_.stride_row,
                 sharded_ ? "sharded (per-rank shard tiering)"
                 : dedup_ ? "replicated (cold-dedup: one copy/page)"
                          : "replicated");
    if (opts_.cohort_rows_max > 1) {
        spdlog::info("KvTiering: cohort consumer = {} (union staging {} rows"
                     " / {} fake pages per rank{})",
                     union_cap_ > 0 ? "BATCHED-UNION (per-row fallback)"
                                    : "PER-ROW",
                     union_cap_, u_pages_cap_,
                     cohort_rowwise_ ? "; LS_KVT_COHORT_ROWWISE=1" : "");
    }
}

KvTieringManager::~KvTieringManager() {
    // Drain in-flight demotions (events must complete before freeing).
    try {
        drain_demotions();
    } catch (const std::exception& e) {
        spdlog::error("KvTiering: demotion drain failed in destructor: {}",
                      e.what());
    }
    log_stats();
    for (int r = 0; r < static_cast<int>(ranks_.size()); ++r) {
        auto* be = backend(r);
        if (!be) continue;
        be->set_device();
        auto& rb = ranks_[r];
        // Quiesce async work still referencing the buffers below: an
        // unconsumed prepare() readback, an in-flight lookahead prefetch,
        // and the last materialize's gather.
        if (r < static_cast<int>(sel_.size()) && sel_[r].pending)
            wait_event(be, rb.ev_sync);
        if (rb.pf_inflight) wait_event(be, rb.ev_pf);
        for (auto& ms : rb.mat) {
            if (ms.mat_inflight) wait_event(be, ms.ev_mat_done);
            if (ms.ev_h2d) wait_event(be, ms.ev_h2d);  // cold burst (h_stage)
        }
        if (rb.umat.mat_inflight) wait_event(be, rb.umat.ev_mat_done);
        if (rb.umat.ev_h2d) wait_event(be, rb.umat.ev_h2d);
        if (rb.uidx_inflight) wait_event(be, rb.ev_uidx);
        be->device_free(rb.row_cache);
        for (auto& ms : rb.mat) {
            be->device_free(ms.scratch);
            be->device_free(ms.cold_incoming);
            be->device_free(ms.dev_src_ptrs);
            be->device_free(ms.dev_scatter_ptrs);
            be->device_free(ms.dev_scatter_idx);
            if (ms.ev_h2d) be->destroy_event(ms.ev_h2d);
            if (ms.ev_mat_done) be->destroy_event(ms.ev_mat_done);
        }
        be->device_free(rb.umat.scratch);
        be->device_free(rb.umat.cold_incoming);
        be->device_free(rb.umat.dev_src_ptrs);
        be->device_free(rb.dev_uidx);
        be->device_free(rb.dev_useq);
        be->device_free(rb.dev_union_bt);
        if (rb.umat.ev_h2d) be->destroy_event(rb.umat.ev_h2d);
        if (rb.umat.ev_mat_done) be->destroy_event(rb.umat.ev_mat_done);
        if (rb.ev_uidx) be->destroy_event(rb.ev_uidx);
        be->device_free(rb.dev_ident_indices);
        be->device_free(rb.dev_fake_bt);
        be->device_free(rb.pf_incoming);
        be->device_free(rb.dev_pf_scatter_ptrs);
        be->device_free(rb.dev_pf_scatter_idx);
        if (rb.ev_sync) be->destroy_event(rb.ev_sync);
        if (rb.ev_attn_order) be->destroy_event(rb.ev_attn_order);
        if (rb.ev_pf) be->destroy_event(rb.ev_pf);
        if (rb.ev_pf_order) be->destroy_event(rb.ev_pf_order);
        if (rb.h2d_stream) be->destroy_stream(rb.h2d_stream);
        if (rb.owns_d2h_stream && rb.d2h_stream)
            be->destroy_stream(rb.d2h_stream);
        if (opts_.numa_manager && rb.host_arena.data) {
            if (rb.host_registered)
                core::host_unregister_pinned(rb.host_arena.data);
            opts_.numa_manager->free(rb.host_arena);
        } else if (rb.host_arena.data) {
            be->host_free_pinned(rb.host_arena.data);
        }
    }
}

// ── Helpers ────────────────────────────────────────��───────────────────────

compute::DeviceBackend* KvTieringManager::backend(int rank) const {
    const int pos = opts_.gpus[static_cast<size_t>(rank)].position;
    if (pos < 0 || pos >= static_cast<int>(opts_.device_backends.size()))
        return nullptr;
    return opts_.device_backends[static_cast<size_t>(pos)];
}

void KvTieringManager::wait_event(compute::DeviceBackend* be, void* ev) const {
    // TD-BRIDGE-CPP-GAP residual (bridge-gap campaign 2026-08-23): the
    // selection-sync host wait uses the SAME hot query spin as the FAR
    // handler's attention completion-event wait (dispatch_reef.cpp). Under
    // the fused FAR pipeline the handler reaches ensure_selection() while
    // the previous layer's FFN still occupies the GPU stream, so this wait
    // covers the FFN tail + top-k + D2H (measured 91-96 us/mat vs 19-20 us
    // on the client-paced split path); a yield-based spin additionally pays
    // scheduler wake latency on every materialization, delaying the
    // post-selection enqueue. Hot-spin for a bounded window (covers any
    // in-pipeline wait), then fall back to yielding (teardown / idle-class
    // guard waits stay polite). LS_KVT_HOT_WAIT=0 restores the legacy
    // yield loop (diagnostic off-switch).
    static const bool hot_wait = [] {
        const char* e = std::getenv("LS_KVT_HOT_WAIT");
        return !(e && *e == '0');
    }();
    constexpr auto kHotWindow = std::chrono::milliseconds(5);
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        auto q = be->query_event(ev);
        if (q.status == compute::EventStatus::kReady) return;
        if (q.status == compute::EventStatus::kError) {
            throw std::runtime_error(
                "KvTiering: event query failed (device error "
                + std::to_string(q.error_code) + ")");
        }
        if (!hot_wait || std::chrono::steady_clock::now() - t0 > kHotWindow)
            std::this_thread::yield();
    }
}

uint64_t KvTieringManager::wait_event_us(compute::DeviceBackend* be,
                                         void* ev) const {
    const auto t0 = std::chrono::steady_clock::now();
    wait_event(be, ev);
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count());
}

char* KvTieringManager::cache_slot_addr(int rank, int layer, int slot) const {
    return static_cast<char*>(ranks_[static_cast<size_t>(rank)].row_cache)
        + (static_cast<size_t>(layer) * hot_slots_ + slot)
              * static_cast<size_t>(opts_.stride_row);
}

void KvTieringManager::evict_slot(LayerCache& lc, int s) {
    const int pos = lc.slot_pos[static_cast<size_t>(s)];
    if (pos >= 0) {
        auto sit = lc.map.find(lc.slot_seq[static_cast<size_t>(s)]);
        // NOTE: never erase the (possibly now empty) OUTER map entry here —
        // materialize/prefetch hold a reference to the requesting seq's
        // inner map across their insert loops; erasing the referred element
        // would dangle it.  Empty entries are purged by the callers/release.
        if (sit != lc.map.end()) sit->second.erase(pos);
        ++stats_.cache_evictions;
    }
    lc.slot_prefetched[static_cast<size_t>(s)] = 0;
}

int KvTieringManager::acquire_cache_slot(LayerCache& lc, uint64_t tick,
                                         uint64_t seq) {
    if (!lc.free_slots.empty()) {  // released by a sequence teardown
        const int s = lc.free_slots.back();
        lc.free_slots.pop_back();
        return s;
    }
    if (lc.filled < hot_slots_) return lc.filled++;
    if (lc.evict_tick != tick) {
        // Lazy per-tick LRU order: slots not touched at `tick`, oldest first.
        lc.evict_tick = tick;
        lc.evict_scratch.clear();
        lc.evict_scratch.reserve(static_cast<size_t>(hot_slots_));
        for (int s = 0; s < hot_slots_; ++s)
            if (lc.slot_use[static_cast<size_t>(s)] != tick)
                lc.evict_scratch.emplace_back(
                    lc.slot_use[static_cast<size_t>(s)], s);
        std::sort(lc.evict_scratch.begin(), lc.evict_scratch.end());
        lc.evict_idx = 0;
        lc.evict_deferred.clear();
        lc.deferred_idx = 0;
    }
    // TD-KVT-BATCH fair-share: a sequence's inserts evict its OWN rows or
    // an OVER-share sequence's rows first; under-share other-seq rows are
    // deferred (last resort).  Single-seq: every slot is own → identical to
    // the plain LRU walk.  Placement/fairness only — never selection.
    const auto seq_count = [&](uint64_t q) -> int {
        const auto it = lc.map.find(q);
        return it == lc.map.end() ? 0 : static_cast<int>(it->second.size());
    };
    size_t nseqs = seq_count(seq) > 0 ? 0u : 1u;  // requester counts once
    for (const auto& [q, m] : lc.map)
        if (!m.empty()) ++nseqs;
    const int share = hot_slots_ / static_cast<int>(std::max<size_t>(nseqs, 1));
    while (lc.evict_idx < lc.evict_scratch.size()) {
        const int s = lc.evict_scratch[lc.evict_idx++].second;
        if (lc.slot_use[static_cast<size_t>(s)] == tick) continue;  // touched
        const uint64_t owner = lc.slot_seq[static_cast<size_t>(s)];
        if (lc.slot_pos[static_cast<size_t>(s)] >= 0 && owner != seq
            && seq_count(owner) <= share) {
            lc.evict_deferred.push_back(s);  // under-share: protect for now
            continue;
        }
        evict_slot(lc, s);
        return s;
    }
    while (lc.deferred_idx < lc.evict_deferred.size()) {
        const int s = lc.evict_deferred[lc.deferred_idx++];
        if (lc.slot_use[static_cast<size_t>(s)] == tick) continue;  // touched
        evict_slot(lc, s);
        return s;
    }
    return -1;  // saturated by this pass — row stays uncached
}

void KvTieringManager::ensure_layer_pages(SeqState& ss, int layer,
                                          int num_logical) {
    auto& v = ss.pages[static_cast<size_t>(layer)];
    if (static_cast<int>(v.size()) < num_logical)
        v.resize(static_cast<size_t>(num_logical));
}

KvTieringManager::SeqState* KvTieringManager::find_seq(uint64_t seq_id) {
    auto it = seqs_.find(seq_id);
    return it == seqs_.end() ? nullptr : &it->second;
}

const KvTieringManager::SeqState*
KvTieringManager::find_seq(uint64_t seq_id) const {
    auto it = seqs_.find(seq_id);
    return it == seqs_.end() ? nullptr : &it->second;
}

bool KvTieringManager::state_layer_has_cold(const SeqState& ss, int layer) {
    if (layer < 0 || layer >= static_cast<int>(ss.pages.size())) return false;
    for (const auto& p : ss.pages[static_cast<size_t>(layer)])
        if (p.state == PageState::kCold) return true;
    return false;
}

bool KvTieringManager::layer_has_cold(uint64_t seq_id, int layer) const {
    const SeqState* ss = find_seq(seq_id);
    return ss && state_layer_has_cold(*ss, layer);
}

bool KvTieringManager::seq_has_demotions(uint64_t seq_id) const {
    const SeqState* ss = find_seq(seq_id);
    return ss && ss->demoted_or_inflight > 0;
}

int KvTieringManager::cold_pool_numa_node(int rank) const {
    if (rank < 0 || rank >= static_cast<int>(ranks_.size())) return -1;
    return ranks_[static_cast<size_t>(rank)].numa_node;
}

int KvTieringManager::cold_pool_used_pages(int rank) const {
    if (rank < 0 || rank >= static_cast<int>(ranks_.size())) return 0;
    return cold_pool_pages_
        - static_cast<int>(ranks_[static_cast<size_t>(rank)].cold_free.size());
}

int KvTieringManager::seq_cold_used_pages(uint64_t seq_id, int rank) const {
    const SeqState* ss = find_seq(seq_id);
    if (!ss || rank < 0 || rank >= static_cast<int>(ss->cold_used.size()))
        return 0;
    return ss->cold_used[static_cast<size_t>(rank)];
}

int KvTieringManager::cache_entries(int rank, int layer) const {
    if (rank < 0 || rank >= static_cast<int>(cache_.size())) return 0;
    if (layer < 0 || layer >= static_cast<int>(cache_[rank].size())) return 0;
    int n = 0;
    for (const auto& [seq, m] : cache_[rank][layer].map)
        n += static_cast<int>(m.size());
    return n;
}

int KvTieringManager::cache_entries_seq(int rank, int layer,
                                        uint64_t seq_id) const {
    if (rank < 0 || rank >= static_cast<int>(cache_.size())) return 0;
    if (layer < 0 || layer >= static_cast<int>(cache_[rank].size())) return 0;
    const auto it = cache_[rank][layer].map.find(seq_id);
    return it == cache_[rank][layer].map.end()
        ? 0 : static_cast<int>(it->second.size());
}

int KvTieringManager::global_page_of_index(int rank, int sel_idx) const {
    const int jl = sel_idx / opts_.page_size;
    if (!sharded_) return jl;  // replicated: index IS the global position
    return kvshard::global_page_of_local(rank, jl, ppc_,
                                         static_cast<int>(ranks_.size()));
}

int KvTieringManager::cold_owner_rank(int global_page) const {
    const int R = static_cast<int>(ranks_.size());
    if (sharded_) return kvshard::page_owner_rank(global_page, ppc_, R);
    if (dedup_) return global_page % R;  // round-robin cold owner
    return -1;  // replicated non-dedup: every rank stores its own copy
}

const void* KvTieringManager::cold_page_host_ptr(uint64_t seq_id, int layer,
                                                 int logical) const {
    const SeqState* ss = find_seq(seq_id);
    if (!ss) return nullptr;
    if (layer < 0 || layer >= static_cast<int>(ss->pages.size()))
        return nullptr;
    const auto& lv = ss->pages[static_cast<size_t>(layer)];
    if (logical < 0 || logical >= static_cast<int>(lv.size())) return nullptr;
    const auto& p = lv[static_cast<size_t>(logical)];
    if (p.state != PageState::kCold) return nullptr;
    // Replicated non-dedup: every rank's copy is identical — rank 0
    // suffices.  Dedup: the single copy lives in the round-robin cold
    // owner's pool.  Sharded: in the shard owner's pool.
    const int owner = cold_owner_rank(logical);
    const int r = owner >= 0 ? owner : 0;
    if (r >= static_cast<int>(p.cold_slot.size()) || p.cold_slot[r] < 0)
        return nullptr;
    return ranks_[static_cast<size_t>(r)].cold_base
        + static_cast<int64_t>(p.cold_slot[static_cast<size_t>(r)])
              * opts_.stride_block;
}

void KvTieringManager::drain_demotions() {
    for (int spin = 0; !inflight_.empty() && spin < 1000000; ++spin) {
        poll_demotions();
        if (!inflight_.empty()) std::this_thread::yield();
    }
    if (!inflight_.empty()) {
        throw std::runtime_error(
            "KvTiering: demotion drain timed out with "
            + std::to_string(inflight_.size()) + " groups in flight");
    }
}

// ── Dispatcher API ──────────────────────────────────────────────────────────

bool KvTieringManager::begin_layer(int layer, uint64_t seq_id,
                                   uint32_t token_pos,
                                   const int* const* host_block_tables,
                                   int rows) {
    poll_demotions();
    if (layer < 0 || layer >= opts_.kv_layers) return false;
    // TD-KVT-ADMISSION-UPFRONT: a chunk cohort (rows > 1) writes
    // [token_pos, token_pos + rows) — legality below is checked against the
    // FIRST write position; the cohort staging must fit the rows.
    if (rows < 1 || (rows > 1 && rows > opts_.cohort_rows_max)) return false;
    // TD-KVT-BATCH: per-sequence state, created on first sight — any number
    // of sequences may be tiered concurrently (each step is still B==1).
    auto [it, inserted] = seqs_.try_emplace(seq_id);
    SeqState& ss = it->second;
    if (inserted)
        ss.pages.assign(static_cast<size_t>(opts_.kv_layers), {});
    if (rows > 1) {
        // TD-KVT-ADMISSION-UPFRONT cohort legality is PER-LAYER: the chunk
        // k_appends [token_pos, token_pos + rows) into THIS layer's pages,
        // and a superchunk's layer-wise sweep legitimately REPLAYS earlier
        // sub-chunks at later layers (max_pos_seen already covers them —
        // NOT a rewind; the global demoted frontier is too coarse for that
        // shape).  Fail loud if THIS layer holds a non-hot page at/after
        // the first write position — its bytes would be rewritten through a
        // neutralized handle or mid-D2H (INV-KVT-2).  Per-layer demotion
        // trails each layer's own ascending pass (after_attention), so a
        // legal sweep never trips this.
        const auto& lsv = ss.pages[static_cast<size_t>(layer)];
        const int j0 = static_cast<int>(token_pos) / opts_.page_size;
        for (int j = j0; j < static_cast<int>(lsv.size()); ++j) {
            if (lsv[static_cast<size_t>(j)].state != PageState::kHot) {
                throw std::runtime_error(
                    "KvTiering: cohort chunk write at pos "
                    + std::to_string(token_pos) + " (rows "
                    + std::to_string(rows) + ") over a demoted page of layer "
                    + std::to_string(layer) + " (logical " + std::to_string(j)
                    + ") — INV-KVT-2");
            }
        }
        ss.max_pos_seen = std::max(
            ss.max_pos_seen, token_pos + static_cast<uint32_t>(rows));
        ctx_seq_ = seq_id;
        ctx_layer_ = layer;
        ctx_pos_ = token_pos;
        ctx_rows_ = rows;
        ctx_cold_valid_ = false;
        for (int r = 0; r < static_cast<int>(ranks_.size()); ++r)
            ctx_host_bt_[static_cast<size_t>(r)] = host_block_tables
                ? host_block_tables[r] : nullptr;
        return true;
    }
    if (token_pos + 1 < ss.max_pos_seen) {
        // Rollback/rewind (speculation truncation).  TD-KVT-SPEC: cold rows
        // are immutable PREFIX content.  This step's k_append REWRITES
        // position token_pos (slot mapping targets its page), so the rewind
        // is legal only while every position at/after token_pos is HOT
        // (demoted_frontier <= token_pos); [0, token_pos) is the kept
        // prefix.  Rewinding INTO demoted territory needs cold-page
        // re-promotion FIRST — the dispatcher runs repromote_for_rewind()
        // BEFORE the kv-meta build so this step's block tables / slot
        // mappings already carry the fresh VRAM handles.  Reaching this
        // guard with a frontier above token_pos means that hook did not run
        // or failed — fail loudly rather than score stale KV or write
        // through a neutralized handle (INV-KVT-2, TD-KVT-SPEC-FORK).
        if (ss.demoted_frontier > token_pos) {
            throw std::runtime_error(
                "KvTiering: sequence rewind (pos " + std::to_string(token_pos)
                + " < seen " + std::to_string(ss.max_pos_seen)
                + ") reaches demoted territory (frontier "
                + std::to_string(ss.demoted_frontier)
                + ") without cold-page re-promotion "
                  "(TD-KVT-SPEC-FORK / INV-KVT-2)");
        }
        ss.max_pos_seen = token_pos + 1;  // demoted prefix intact — safe
    }
    ss.max_pos_seen = std::max(
        ss.max_pos_seen, token_pos + static_cast<uint32_t>(rows));
    ctx_seq_ = seq_id;
    ctx_layer_ = layer;
    ctx_pos_ = token_pos;
    ctx_rows_ = rows;
    ctx_cold_valid_ = false;  // per-layer any-cold cache (cohort fast path)
    for (int r = 0; r < static_cast<int>(ranks_.size()); ++r)
        ctx_host_bt_[static_cast<size_t>(r)] = host_block_tables
            ? host_block_tables[r] : nullptr;
    return true;
}

void KvTieringManager::after_attention(int layer, uint64_t seq_id,
                                       uint32_t token_pos,
                                       const memory::PageHandle* pages,
                                       int num_logical, int handle_stride) {
    if (layer < 0 || layer >= opts_.kv_layers || !pages) return;
    if (layer_dense_[static_cast<size_t>(layer)]) return;  // sticky-dense
    auto [sit, inserted] = seqs_.try_emplace(seq_id);
    SeqState& ss = sit->second;
    if (inserted)
        ss.pages.assign(static_cast<size_t>(opts_.kv_layers), {});

    ensure_layer_pages(ss, layer, num_logical);
    auto& ls = ss.pages[static_cast<size_t>(layer)];

    const int PS = opts_.page_size;
    const int64_t demote_end_tok =
        static_cast<int64_t>(token_pos) + 1 - retention_tokens_;
    const int frontier = static_cast<int>(token_pos) / PS;
    const int R = static_cast<int>(ranks_.size());

    // TD-KVT-BATCH cold fair-share: with N demoting sequences, cap this
    // sequence's per-rank cold slots at capacity / N — one sequence can
    // never monopolize the shared pool.  Over-cap demotions are skipped
    // (fail-safe: pages stay hot), placement-only.  A single demoting
    // sequence keeps the full pool (Phase-1 behavior).  Existing over-cap
    // holders keep their slots (no reclaim without re-promotion); the cap
    // only gates NEW demotions.
    int nseq_cold = 0;
    for (const auto& [q, s2] : seqs_)
        if (s2.demoted_or_inflight > 0 || q == seq_id) ++nseq_cold;
    const int cold_cap = nseq_cold > 1
        ? std::max(cold_pool_pages_ / nseq_cold, 1)
        : cold_pool_pages_;
    if (static_cast<int>(ss.cold_used.size()) < R)
        ss.cold_used.resize(static_cast<size_t>(R), 0);

    InflightDemotion group;
    group.seq = seq_id;
    std::vector<uint8_t> order_recorded(static_cast<size_t>(R), 0);
    std::vector<uint8_t> participated(static_cast<size_t>(R), 0);
    for (int j = 0; j < num_logical && j < static_cast<int>(ls.size()); ++j) {
        if (static_cast<int64_t>(j + 1) * PS > demote_end_tok) break;
        if (j >= frontier) break;  // never demote the append frontier page
        auto& pst = ls[static_cast<size_t>(j)];
        if (pst.state != PageState::kHot) continue;
        const memory::PageHandle& h =
            pages[static_cast<size_t>(j) * handle_stride];
        if (h.page_idx < 0 || !h.gpu_ptr) continue;

        // STORING ranks (cold slot + D2H copy): every rank under replicated
        // non-dedup KV; ONLY the round-robin cold owner under replicated
        // dedup (TD-KVT-REPLICA-COLD-DEDUP, INV-KVT-11 — replicas are
        // byte-identical, one copy suffices); ONLY the chunk owner under
        // sharded KV (INV-4.9e — the page physically exists on that rank's
        // GPU alone).  FENCING ranks (completion event before the free):
        // every rank whose VRAM replica the free releases — under dedup the
        // non-storing ranks still fence behind their attention streams so
        // in-flight reads of their replica never outlive the free.
        const int owner = cold_owner_rank(j);
        int s_begin = 0;
        int s_end = R;
        if (owner >= 0) {
            if (sharded_
                && h.gpu_idx
                       != opts_.gpus[static_cast<size_t>(owner)].position) {
                // Ownership sanity: the handle's GPU must be the owner
                // rank's (allocation routed by the same rule) — never D2H
                // from a pool the page does not live in.
                throw std::runtime_error(
                    "KvTiering: sharded page ownership mismatch (layer "
                    + std::to_string(layer) + ", logical " + std::to_string(j)
                    + ": handle gpu " + std::to_string(h.gpu_idx)
                    + " != owner rank " + std::to_string(owner) + " gpu "
                    + std::to_string(opts_.gpus[owner].position) + ")");
            }
            s_begin = owner;
            s_end = owner + 1;
        }
        const int f_begin = sharded_ ? s_begin : 0;
        const int f_end = sharded_ ? s_end : R;

        // Per-seq cold fair-share cap (storing ranks only).  `continue`, not
        // `break`: under dedup/sharded the next page's storing rank differs
        // (round-robin) and may still have budget headroom.
        bool over_budget = false;
        for (int r = s_begin; r < s_end; ++r)
            if (ss.cold_used[static_cast<size_t>(r)] >= cold_cap)
                over_budget = true;
        if (over_budget) {
            ++stats_.budget_skips;
            continue;
        }

        // Acquire one cold slot per STORING rank (all or nothing).
        std::vector<int> slots(static_cast<size_t>(R), -1);
        bool ok = true;
        for (int r = s_begin; r < s_end; ++r) {
            if (ranks_[r].cold_free.empty()) { ok = false; break; }
            slots[r] = ranks_[r].cold_free.back();
            ranks_[r].cold_free.pop_back();
            // Single holder until a fork shares it (TD-KVT-SPEC-FORK).
            ranks_[r].cold_ref[static_cast<size_t>(slots[r])] = 1;
        }
        if (!ok) {
            for (int r = s_begin; r < s_end; ++r)
                if (slots[r] >= 0) {
                    ranks_[r].cold_ref[static_cast<size_t>(slots[r])] = 0;
                    ranks_[r].cold_free.push_back(slots[r]);
                }
            if (!cold_pool_full_warned_) {
                cold_pool_full_warned_ = true;
                spdlog::warn("KvTiering: cold pool full — further demotions "
                             "skipped (pages stay hot; raise "
                             "host_to_device_ratio)");
            }
            break;
        }

        for (int r = f_begin; r < f_end; ++r) {
            auto* be = backend(r);
            be->set_device();
            auto& rb = ranks_[r];
            if (!order_recorded[static_cast<size_t>(r)]
                && opts_.stream_manager) {
                // All prior attention-stream work (k_append writes, this
                // layer's reads) must precede the D2H reads on storing
                // ranks AND the page free on fence-only ranks (dedup).
                const int gpu_pos = opts_.gpus[r].position;
                opts_.stream_manager->record_event(
                    rb.ev_attn_order, gpu_pos, compute::StreamId::kAttention);
                be->stream_wait_event(rb.d2h_stream, rb.ev_attn_order);
            }
            order_recorded[static_cast<size_t>(r)] = 1;
            participated[static_cast<size_t>(r)] = 1;
            if (slots[r] < 0) continue;  // fence-only rank (dedup non-owner)
            const char* src = static_cast<const char*>(opts_.kv_main_bases[r])
                + static_cast<int64_t>(h.page_idx) * opts_.stride_block;
            char* dst = rb.cold_base
                + static_cast<int64_t>(slots[r]) * opts_.stride_block;
            be->memcpy_d2h_async(dst, src,
                                 static_cast<size_t>(opts_.stride_block),
                                 rb.d2h_stream);
        }

        for (int r = s_begin; r < s_end; ++r)
            ++ss.cold_used[static_cast<size_t>(r)];
        pst.state = PageState::kD2hInflight;
        pst.cold_slot = std::move(slots);
        ss.demoted_frontier = std::max(
            ss.demoted_frontier, static_cast<uint32_t>(j + 1) * PS);
        group.handles.push_back(h);
        group.pages_ll.emplace_back(layer, j);
        ++ss.demoted_or_inflight;
        ++ss.inflight;
        ++total_demoted_or_inflight_;
    }

    if (!group.handles.empty()) {
        // One completion event per PARTICIPATING rank covers the whole
        // batch (non-participants keep nullptr — poll skips them).
        group.events.resize(static_cast<size_t>(R), nullptr);
        for (int r = 0; r < R; ++r) {
            if (!participated[static_cast<size_t>(r)]) continue;
            auto* be = backend(r);
            be->set_device();
            group.events[r] = be->create_event();
            be->record_event(group.events[r], ranks_[r].d2h_stream);
        }
        inflight_.push_back(std::move(group));
    }
}

void KvTieringManager::poll_demotions() {
    for (auto it = inflight_.begin(); it != inflight_.end();) {
        bool done = true;
        for (int r = 0; r < static_cast<int>(it->events.size()); ++r) {
            if (!it->events[r]) continue;  // rank did not participate (sharded)
            auto q = backend(r)->query_event(it->events[r]);
            if (q.status == compute::EventStatus::kError) {
                throw std::runtime_error(
                    "KvTiering: demotion D2H failed (device error "
                    + std::to_string(q.error_code) + ")");
            }
            if (q.status != compute::EventStatus::kReady) { done = false; break; }
        }
        if (!done) { ++it; continue; }
        SeqState* ss = find_seq(it->seq);
        for (size_t k = 0; k < it->handles.size(); ++k) {
            const auto [layer, j] = it->pages_ll[k];
            if (ss) {
                ss->pages[static_cast<size_t>(layer)][static_cast<size_t>(j)]
                    .state = PageState::kCold;
                --ss->inflight;
            }
            // Returns the VRAM page (the capacity win) + neutralizes the
            // owner's stored handle so bulk frees skip it.
            opts_.free_page(it->seq, layer, j, it->handles[k]);
            ++stats_.demoted_pages;
        }
        for (int r = 0; r < static_cast<int>(it->events.size()); ++r) {
            if (!it->events[r]) continue;
            backend(r)->set_device();
            backend(r)->destroy_event(it->events[r]);
        }
        it = inflight_.erase(it);
    }
}

void KvTieringManager::release_seq(uint64_t seq_id) {
    auto it = seqs_.find(seq_id);
    if (it == seqs_.end()) return;
    SeqState& ss = it->second;
    // Release the sequence's cold-slot holds (its in-flight demotions were
    // drained by the caller — every cold_slot is settled).  A slot shared
    // with a fork family returns to the pool only when the LAST holder
    // releases it (TD-KVT-SPEC-FORK refcounts — never a double free).
    for (auto& lv : ss.pages) {
        for (auto& p : lv) {
            if (p.state == PageState::kHot) continue;
            for (int r = 0; r < static_cast<int>(ranks_.size()); ++r) {
                if (r < static_cast<int>(p.cold_slot.size())
                    && p.cold_slot[r] >= 0)
                    release_cold_slot(r, p.cold_slot[r]);
            }
        }
    }
    total_demoted_or_inflight_ -= ss.demoted_or_inflight;
    // Quiesce in-flight lookahead prefetches BEFORE releasing this
    // sequence's row-cache slots: a released slot could be re-acquired
    // while a stale prefetch scatter is still writing it (write-write
    // race).  Sequence teardown path — a host wait is fine here.
    for (int r = 0; r < static_cast<int>(ranks_.size()); ++r) {
        if (ranks_[static_cast<size_t>(r)].pf_inflight) {
            stats_.guard_wait_us += wait_event_us(
                backend(r), ranks_[static_cast<size_t>(r)].ev_pf);
            ranks_[static_cast<size_t>(r)].pf_inflight = false;
        }
    }
    // Release the sequence's row-cache entries on every (rank, layer)
    // (TD-KVT-BATCH: other sequences' entries are untouched).
    for (auto& rc : cache_) {
        for (auto& lc : rc) {
            auto mit = lc.map.find(seq_id);
            if (mit == lc.map.end()) continue;
            for (const auto& [pos, s] : mit->second) {
                lc.slot_pos[static_cast<size_t>(s)] = -1;
                lc.slot_seq[static_cast<size_t>(s)] = 0;
                lc.slot_use[static_cast<size_t>(s)] = 0;
                lc.slot_prefetched[static_cast<size_t>(s)] = 0;
                lc.free_slots.push_back(s);
            }
            lc.map.erase(mit);
            // The per-tick eviction order may reference released slots —
            // rebuild it on the next acquisition.
            lc.evict_tick = 0;
            lc.evict_scratch.clear();
            lc.evict_idx = 0;
            lc.evict_deferred.clear();
            lc.deferred_idx = 0;
        }
    }
    // Selection step state: invalidate host copies bound to this sequence
    // (mat_inflight — the last gather on the persistent per-rank staging —
    // intentionally survives: it guards the next writer, whatever sequence
    // it serves; a pending readback also survives so the next consumer /
    // destructor waits it — step-identity mismatch already rejects reuse).
    for (auto& sc : sel_) {
        if (sc.seq != seq_id) continue;
        sc.valid = false;
        sc.reuse_ok = false;
        sc.fresh = false;
        sc.prepared_layer = -1;
    }
    if (ctx_seq_ == seq_id) {
        ctx_seq_ = 0;
        ctx_layer_ = -1;
        ctx_pos_ = 0;
    }
    seqs_.erase(it);
    if (seqs_.empty()) cold_pool_full_warned_ = false;
}

void KvTieringManager::on_seq_free(uint64_t seq_id) {
    auto it = seqs_.find(seq_id);
    if (it == seqs_.end()) return;
    // Drain in-flight demotions so their device pages are freed exactly once
    // (before free_sequence scans the allocator metadata).  Only needed when
    // THIS sequence has copies in flight (drain settles all sequences).
    if (it->second.inflight > 0) drain_demotions();
    if (it->second.demoted_or_inflight > 0) log_stats();
    release_seq(seq_id);
}

void KvTieringManager::release_cold_slot(int rank, int slot) {
    auto& rb = ranks_[static_cast<size_t>(rank)];
    uint32_t& ref = rb.cold_ref[static_cast<size_t>(slot)];
    if (ref == 0) {
        // Fail loudly: a zero-ref release means the slot was already
        // returned — continuing would hand the same slot to two demotions
        // (silent cold-copy aliasing, INV-KVT-2).
        throw std::runtime_error(
            "KvTiering: cold slot double release (rank " + std::to_string(rank)
            + ", slot " + std::to_string(slot) + ")");
    }
    if (--ref == 0) rb.cold_free.push_back(slot);
}

// ── TD-KVT-SPEC-FORK: fork interop + cold-page re-promotion ────────────────

void KvTieringManager::on_seq_fork(uint64_t src_seq_id, uint64_t dst_seq_id) {
    if (src_seq_id == dst_seq_id) return;  // dispatcher rejects; belt+braces
    auto it = seqs_.find(src_seq_id);
    if (it == seqs_.end()) return;  // parent never tiered — child starts fresh
    // Settle the parent's in-flight demotions first: the copied state must
    // hold only HOT/COLD pages (an inherited kD2hInflight entry would let
    // the child read a cold slot whose D2H has not landed).
    if (it->second.inflight > 0) drain_demotions();
    if (it->second.demoted_or_inflight == 0) return;  // nothing to share
    if (seqs_.count(dst_seq_id)) on_seq_free(dst_seq_id);  // stale id reuse
                                                           // (drains first)
    // Deep copy: page states, demoted frontier, per-rank cold accounting
    // (the child holds the same slots).  The child's row cache starts empty
    // (entries are (seq, position)-keyed); its cold reads warm it lazily.
    SeqState child = it->second;
    int shared_slots = 0;
    for (auto& lv : child.pages) {
        for (auto& p : lv) {
            if (p.state != PageState::kCold) continue;
            for (size_t r = 0; r < p.cold_slot.size(); ++r) {
                if (p.cold_slot[r] < 0) continue;
                // REFCOUNT share — no cold-byte copy; the slot frees only
                // when the last fork-family holder releases it.
                ++ranks_[r].cold_ref[static_cast<size_t>(p.cold_slot[r])];
                ++shared_slots;
            }
        }
    }
    total_demoted_or_inflight_ += child.demoted_or_inflight;
    const int demoted = child.demoted_or_inflight;
    seqs_[dst_seq_id] = std::move(child);
    spdlog::info("KvTiering: fork seq {} -> {} shares {} cold pages "
                 "({} refcounted slots)", src_seq_id, dst_seq_id, demoted,
                 shared_slots);
}

bool KvTieringManager::repromote_for_rewind(uint64_t seq_id,
                                            uint32_t token_pos) {
    const SeqState* ss = find_seq(seq_id);
    // The step's k_append rewrites token_pos — legal without re-promotion
    // only while no cold page holds a position >= token_pos.
    if (!ss || ss->demoted_frontier <= token_pos) return true;
    return repromote_seq(seq_id, token_pos);
}

bool KvTieringManager::repromote_seq(uint64_t seq_id, uint32_t keep_frontier) {
    SeqState* ss = find_seq(seq_id);
    if (!ss || ss->demoted_or_inflight == 0) return true;
    // Settle in-flight demotions: every demoted page becomes COLD (host copy
    // complete) so the copies below read settled bytes, and the demotion
    // groups' frees land exactly once (INV-KVT-4).
    drain_demotions();

    const int PS = opts_.page_size;
    const int R = static_cast<int>(ranks_.size());
    struct Job {
        int layer;
        int logical;
        memory::PageHandle handle{};
    };
    std::vector<Job> jobs;
    for (int l = 0; l < static_cast<int>(ss->pages.size()); ++l) {
        const auto& lv = ss->pages[static_cast<size_t>(l)];
        for (int j = 0; j < static_cast<int>(lv.size()); ++j) {
            if (lv[static_cast<size_t>(j)].state != PageState::kCold) continue;
            if (static_cast<uint32_t>(j + 1) * static_cast<uint32_t>(PS)
                <= keep_frontier)
                continue;  // fully inside the kept prefix — stays cold
            jobs.push_back(Job{l, j, {}});
        }
    }
    if (jobs.empty()) return true;

    bool ok = static_cast<bool>(opts_.alloc_page);
    if (!ok) {
        spdlog::error("KvTiering: repromote_seq(seq {}) without an alloc_page "
                      "seam — fail-closed", seq_id);
        jobs.clear();
    }

    // Phase A — allocate a fresh VRAM page per job through the dispatcher's
    // standard growth path (INV-KV-REP lockstep under replicated KV, owner
    // routing under sharded; the seam un-neutralizes the dispatcher handle
    // and poisons the kv-meta dirty guard so block tables re-upload).  On
    // exhaustion keep the already-allocated jobs — their bytes still get
    // copied below so every valid handle carries valid content — and report
    // failure (capacity, not correctness: the caller fails CLOSED).
    size_t n_alloc = 0;
    for (auto& job : jobs) {
        auto h = opts_.alloc_page(seq_id, job.layer, job.logical);
        if (!h || h->page_idx < 0 || !h->gpu_ptr) {
            spdlog::warn("KvTiering: repromote_seq(seq {}) VRAM allocation "
                         "failed at layer {} logical {} ({}/{} pages done) — "
                         "fail-closed", seq_id, job.layer, job.logical,
                         n_alloc, jobs.size());
            ok = false;
            break;
        }
        if (sharded_) {
            // Ownership sanity (INV-KVT-9): the fresh page must live on the
            // chunk owner's GPU — the same routing the original allocation
            // used; never H2D into a pool the page will not be read from.
            const int owner = cold_owner_rank(job.logical);
            if (owner < 0
                || h->gpu_idx
                       != opts_.gpus[static_cast<size_t>(owner)].position) {
                throw std::runtime_error(
                    "KvTiering: repromoted page ownership mismatch (layer "
                    + std::to_string(job.layer) + ", logical "
                    + std::to_string(job.logical) + ": handle gpu "
                    + std::to_string(h->gpu_idx) + " != owner rank "
                    + std::to_string(owner) + ")");
            }
        }
        job.handle = *h;
        ++n_alloc;
    }
    jobs.resize(n_alloc);

    // Quiesce every async user of the pinned staging / row caches on ALL
    // ranks before the copies and the cache purge below (INV-KVT-7: h_stage
    // is single-writer; a stale prefetch scatter must not race a released
    // cache slot).
    for (int r = 0; r < R; ++r) {
        auto* be = backend(r);
        auto& rb = ranks_[static_cast<size_t>(r)];
        for (auto& ms : rb.mat) {
            if (ms.mat_inflight) {
                stats_.guard_wait_us += wait_event_us(be, ms.ev_mat_done);
                ms.mat_inflight = false;
            }
        }
        if (rb.pf_inflight) {
            stats_.guard_wait_us += wait_event_us(be, rb.ev_pf);
            rb.pf_inflight = false;
        }
    }

    // Phase B — copy the EXACT demoted bytes back (INV-KVT-1 placement-only).
    // Receiving set: EVERY rank under replicated KV (each replica restored
    // in lockstep at the same page_idx, INV-KV-REP); the chunk OWNER alone
    // under sharded KV (INV-KVT-9).  A rank whose own pool holds the cold
    // copy enqueues direct batched async H2Ds from its node-local pinned
    // slot; a dedup non-owner stages the owner's copy through ITS node-local
    // pinned staging in capacity-bounded pieces so the H2D DMA stays
    // node-local — only the host memcpy may cross NUMA nodes (INV-KVT-3/-11).
    // Completion is host-waited per rank BEFORE any slot is released
    // (INV-KVT-4 inverse: a cold slot frees only after every holder's H2D
    // completed).
    const size_t blk = static_cast<size_t>(opts_.stride_block);
    const size_t stage_cap = static_cast<size_t>(opts_.index_topk)
                           * static_cast<size_t>(opts_.stride_row);
    uint64_t h2d_bytes = 0;
    for (int r = 0; r < R && !jobs.empty(); ++r) {
        auto* be = backend(r);
        be->set_device();
        auto& rb = ranks_[static_cast<size_t>(r)];
        bool pending = false;
        for (const auto& job : jobs) {
            const int owner = cold_owner_rank(job.logical);
            if (sharded_ && owner != r) continue;  // not a holder (INV-KVT-9)
            const auto& pst =
                ss->pages[static_cast<size_t>(job.layer)]
                         [static_cast<size_t>(job.logical)];
            char* dst =
                static_cast<char*>(opts_.kv_main_bases[static_cast<size_t>(r)])
                + static_cast<int64_t>(job.handle.page_idx)
                      * opts_.stride_block;
            const int own_slot = r < static_cast<int>(pst.cold_slot.size())
                ? pst.cold_slot[static_cast<size_t>(r)] : -1;
            if (own_slot >= 0) {
                // Node-local pinned source — direct batched async H2D.
                be->memcpy_h2d_async(
                    dst,
                    rb.cold_base
                        + static_cast<int64_t>(own_slot) * opts_.stride_block,
                    blk, rb.h2d_stream);
                pending = true;
                h2d_bytes += blk;
                continue;
            }
            // Replicated dedup non-owner: the single cold copy lives in the
            // round-robin owner's pool (possibly another NUMA node) — stage
            // through THIS rank's node-local pinned staging (INV-KVT-11).
            const int cslot =
                owner >= 0 && owner < static_cast<int>(pst.cold_slot.size())
                    ? pst.cold_slot[static_cast<size_t>(owner)] : -1;
            if (cslot < 0) {
                throw std::runtime_error(
                    "KvTiering: cold page without an owner cold slot "
                    "(repromote, layer " + std::to_string(job.layer)
                    + ", logical " + std::to_string(job.logical) + ")");
            }
            const char* csrc = ranks_[static_cast<size_t>(owner)].cold_base
                + static_cast<int64_t>(cslot) * opts_.stride_block;
            size_t off = 0;
            while (off < blk) {
                const size_t piece = std::min(blk - off, stage_cap);
                std::memcpy(rb.mat[0].h_stage, csrc + off, piece);
                be->memcpy_h2d_async(dst + off, rb.mat[0].h_stage, piece,
                                     rb.h2d_stream);
                // Staging-reuse fence (INV-KVT-7): h_stage is rewritten by
                // the next piece — host-wait the burst (also drains any
                // direct copies enqueued above; stream order).
                be->record_event(rb.mat[0].ev_h2d, rb.h2d_stream);
                wait_event(be, rb.mat[0].ev_h2d);
                pending = false;
                off += piece;
                h2d_bytes += piece;
            }
        }
        if (pending) {
            be->record_event(rb.mat[0].ev_h2d, rb.h2d_stream);
            wait_event(be, rb.mat[0].ev_h2d);
        }
    }

    // Phase C — bookkeeping: release the slots (fork-family refcounts — a
    // shared slot survives for its remaining holders: copy-on-write split),
    // flip pages HOT, lower the frontier, purge the sequence's row-cache
    // entries in rewritten territory (positions >= keep_frontier get new
    // content; a stale entry would serve old bytes if the page re-demotes).
    for (const auto& job : jobs) {
        auto& pst = ss->pages[static_cast<size_t>(job.layer)]
                             [static_cast<size_t>(job.logical)];
        for (int r = 0; r < static_cast<int>(pst.cold_slot.size()); ++r) {
            const int s = pst.cold_slot[static_cast<size_t>(r)];
            if (s < 0) continue;
            release_cold_slot(r, s);
            if (r < static_cast<int>(ss->cold_used.size())
                && ss->cold_used[static_cast<size_t>(r)] > 0)
                --ss->cold_used[static_cast<size_t>(r)];
        }
        pst.cold_slot.clear();
        pst.state = PageState::kHot;
        --ss->demoted_or_inflight;
        --total_demoted_or_inflight_;
        ++stats_.repromoted_pages;
    }
    stats_.repromote_bytes += h2d_bytes;

    uint32_t fr = 0;
    for (const auto& lv : ss->pages)
        for (size_t j = 0; j < lv.size(); ++j)
            if (lv[j].state != PageState::kHot)
                fr = std::max(fr, (static_cast<uint32_t>(j) + 1)
                                      * static_cast<uint32_t>(PS));
    ss->demoted_frontier = fr;

    for (int r = 0; r < R; ++r) {
        for (auto& lc : cache_[static_cast<size_t>(r)]) {
            auto mit = lc.map.find(seq_id);
            if (mit == lc.map.end()) continue;
            bool touched = false;
            for (auto pit = mit->second.begin(); pit != mit->second.end();) {
                const int pos = pit->first;
                // Cache keys are global positions (replicated) or rank-LOCAL
                // slots (sharded) — compare in GLOBAL position space.
                const uint32_t gpos = static_cast<uint32_t>(
                    global_page_of_index(r, pos) * PS + pos % PS);
                if (gpos < keep_frontier) { ++pit; continue; }
                const int s = pit->second;
                lc.slot_pos[static_cast<size_t>(s)] = -1;
                lc.slot_seq[static_cast<size_t>(s)] = 0;
                lc.slot_use[static_cast<size_t>(s)] = 0;
                lc.slot_prefetched[static_cast<size_t>(s)] = 0;
                lc.free_slots.push_back(s);
                pit = mit->second.erase(pit);
                touched = true;
            }
            if (mit->second.empty()) lc.map.erase(mit);
            if (touched) {
                // The per-tick eviction order may reference released slots.
                lc.evict_tick = 0;
                lc.evict_scratch.clear();
                lc.evict_idx = 0;
                lc.evict_deferred.clear();
                lc.deferred_idx = 0;
            }
        }
    }

    spdlog::info("KvTiering: re-promoted {} cold pages of seq {} "
                 "({:.2f} MiB H2D, keep_frontier {}, new frontier {}){}",
                 jobs.size(), seq_id,
                 static_cast<double>(h2d_bytes) / (1024.0 * 1024.0),
                 keep_frontier, ss->demoted_frontier,
                 ok ? "" : " — INCOMPLETE (VRAM exhausted, fail-closed)");
    return ok;
}

// ── Hook: prepare (TD-KVT-SYNC overlapped readback) ─────────────────────────

void KvTieringManager::prepare(int rank, int layer_idx,
                               const int* sparse_indices_dev,
                               const int* topk_lengths_dev,
                               int batch_size, void* stream,
                               bool selection_fresh) {
    // Pure hint: silently no-op on any mismatch — materialize() has its own
    // synchronous fallback and full validation.
    if (batch_size != 1) return;
    if (rank < 0 || rank >= static_cast<int>(ranks_.size())) return;
    if (layer_idx < 0 || layer_idx >= opts_.kv_layers) return;
    if (layer_idx != ctx_layer_ || ctx_seq_ == 0) return;  // no step ctx
    if (!sparse_indices_dev || !topk_lengths_dev || !stream) return;
    const SeqState* ss = find_seq(ctx_seq_);
    if (!ss) return;

    // Readback needed only when this layer — or, on a FRESH (full) layer,
    // one of its IndexShare successors (lookahead prefetch source) — has
    // cold pages for the step's sequence.  All-hot groups keep the
    // zero-overhead fast path.
    bool need = state_layer_has_cold(*ss, layer_idx);
    if (!need && selection_fresh) {
        const int succ = share_succ_[static_cast<size_t>(layer_idx)];
        for (int l = layer_idx + 1;
             l <= layer_idx + succ && l < opts_.kv_layers; ++l) {
            if (!layer_dense_[static_cast<size_t>(l)]
                && state_layer_has_cold(*ss, l)) {
                need = true;
                break;
            }
        }
    }
    if (!need) return;

    auto& sc = sel_[static_cast<size_t>(rank)];
    if (!selection_fresh && sc.seq == ctx_seq_ && sc.pos == ctx_pos_
        && (sc.valid || sc.pending)) {
        // IndexShare SHARED layer: the producer reused the full layer's
        // buffers (byte-identical content) and this manager already holds /
        // is receiving this step's selection — no D2H at all.
        sc.reuse_ok = true;
        sc.fresh = false;
        sc.prepared_layer = layer_idx;
        return;
    }

    // Overlapped readback: enqueue the D2H now (ordered after the
    // just-enqueued top-k on the attention stream); the host wait happens in
    // materialize(), by which point the executor has enqueued the whole
    // projection/staging stretch — the copy is long done.
    auto* be = backend(rank);
    be->set_device();
    auto& rb = ranks_[static_cast<size_t>(rank)];
    be->memcpy_d2h_async(rb.h_indices, sparse_indices_dev,
                         static_cast<size_t>(opts_.index_topk) * sizeof(int),
                         stream);
    be->memcpy_d2h_async(rb.h_topk_len, topk_lengths_dev, sizeof(int), stream);
    be->record_event(rb.ev_sync, stream);
    sc.pending = true;
    sc.valid = false;
    sc.reuse_ok = false;
    sc.fresh = selection_fresh;
    sc.prepared_layer = layer_idx;
    sc.seq = ctx_seq_;
    sc.pos = ctx_pos_;
    ++stats_.prepares;
}

// ── Selection consumption ───────────────────────────────────────────────────

int KvTieringManager::ensure_selection(int rank, int layer_idx,
                                       const int* sparse_indices_dev,
                                       const int* topk_lengths_dev,
                                       void* stream) {
    auto* be = backend(rank);
    auto& rb = ranks_[static_cast<size_t>(rank)];
    auto& sc = sel_[static_cast<size_t>(rank)];
    const int ITK = opts_.index_topk;

    const bool step_match = sc.seq == ctx_seq_ && sc.pos == ctx_pos_
        && sc.prepared_layer == layer_idx;
    if (step_match && sc.pending) {
        // prepare()-issued readback: the wait overlapped the executor's own
        // enqueueing — normally ~0 by now.
        stats_.sync_wait_us += wait_event_us(be, rb.ev_sync);
        sc.pending = false;
        sc.valid = true;
        sc.n = *rb.h_topk_len;
        ++stats_.sync_overlapped;
    } else if (step_match && sc.reuse_ok && sc.valid) {
        // IndexShare host-copy reuse: h_indices already holds this step's
        // selection — no D2H, no wait.
        ++stats_.sync_reuses;
    } else {
        // Legacy synchronous fallback (prepare never ran / mismatch).
        if (sc.pending) {
            // A stale unconsumed readback still targets h_indices — drain it
            // before overwriting (same buffer, same event).
            stats_.guard_wait_us += wait_event_us(be, rb.ev_sync);
            sc.pending = false;
        }
        be->memcpy_d2h_async(rb.h_indices, sparse_indices_dev,
                             static_cast<size_t>(ITK) * sizeof(int), stream);
        be->memcpy_d2h_async(rb.h_topk_len, topk_lengths_dev, sizeof(int),
                             stream);
        be->record_event(rb.ev_sync, stream);
        stats_.sync_wait_us += wait_event_us(be, rb.ev_sync);
        sc.valid = true;
        sc.reuse_ok = false;  // reuse is only ever blessed by prepare()
        sc.fresh = false;
        sc.prepared_layer = layer_idx;
        sc.seq = ctx_seq_;
        sc.pos = ctx_pos_;
        sc.n = *rb.h_topk_len;
        ++stats_.sync_fallbacks;
    }
    if (sc.n > ITK) {
        throw std::runtime_error("KvTiering: topk_length "
                                 + std::to_string(sc.n) + " > index_topk "
                                 + std::to_string(ITK));
    }
    return sc.n;
}

// ── IndexShare lookahead prefetch (TD-KVT-PREFETCH) ─────────────────────────

void KvTieringManager::prefetch_successors(int rank, int layer_idx, int n,
                                           void* stream) {
    auto* be = backend(rank);
    auto& rb = ranks_[static_cast<size_t>(rank)];
    const int PS = opts_.page_size;
    const int ITK = opts_.index_topk;
    const size_t row = static_cast<size_t>(opts_.stride_row);
    SeqState* ssp = find_seq(ctx_seq_);
    if (!ssp) return;

    // Host guard: the previous prefetch burst may still be DMA-reading
    // h_pf_stage / the scatter tables (normally long done — a full share
    // group of attention+MoE has passed).
    if (rb.pf_inflight) {
        stats_.guard_wait_us += wait_event_us(be, rb.ev_pf);
        rb.pf_inflight = false;
    }

    int m = 0;  // staged rows across ALL successors (≤ ITK)
    const int succ = share_succ_[static_cast<size_t>(layer_idx)];
    std::vector<int> touched;
    for (int l = layer_idx + 1;
         l <= layer_idx + succ && l < opts_.kv_layers && m < ITK; ++l) {
        if (layer_dense_[static_cast<size_t>(l)]) continue;
        auto& lsl = ssp->pages[static_cast<size_t>(l)];
        auto& lc = cache_[static_cast<size_t>(rank)][static_cast<size_t>(l)];
        auto& seq_map = lc.map[ctx_seq_];
        ++tick_;
        const int m_before = m;
        for (int i = 0; i < n && m < ITK; ++i) {
            const int pos = rb.h_indices[i];
            if (pos < 0) {
                throw std::runtime_error(
                    "KvTiering: negative index inside topk_length (prefetch)");
            }
            // pos is a global position (replicated) or a rank-LOCAL slot
            // (sharded, KVS-4 translated) — page state is keyed by the
            // GLOBAL logical page; within-page offsets are mode-invariant.
            const int j = global_page_of_index(rank, pos);
            if (j >= static_cast<int>(lsl.size())
                || lsl[static_cast<size_t>(j)].state != PageState::kCold)
                continue;  // pool-resident in layer l — no fetch needed
            if (auto it = seq_map.find(pos); it != seq_map.end()) {
                // Already cached: refresh recency so this prefetch pass does
                // not evict a row the successor is about to hit.
                lc.slot_use[static_cast<size_t>(it->second)] = tick_;
                continue;
            }
            const int s = acquire_cache_slot(lc, tick_, ctx_seq_);
            if (s < 0) continue;  // cache saturated — successor will fetch
            // Cold source pool: the rank's own copy, or the round-robin
            // cold owner's single copy under replicated dedup (the host
            // memcpy may cross NUMA nodes; the burst H2D below still reads
            // this rank's node-local staging — INV-KVT-11).
            const int cr = cold_src_rank(rank, j);
            const int cslot =
                lsl[static_cast<size_t>(j)].cold_slot[static_cast<size_t>(cr)];
            if (cslot < 0) {
                // Under sharded KV a translated local index can only map to a
                // page THIS rank owns; under dedup the cold owner must hold
                // the copy — a missing slot is a bookkeeping bug.
                throw std::runtime_error(
                    "KvTiering: cold page without a rank-" + std::to_string(cr)
                    + " cold slot (prefetch, layer " + std::to_string(l)
                    + ", logical " + std::to_string(j) + ")");
            }
            const char* csrc = ranks_[static_cast<size_t>(cr)].cold_base
                + static_cast<int64_t>(cslot) * opts_.stride_block
                + static_cast<int64_t>(pos % PS) * opts_.stride_row;
            std::memcpy(rb.h_pf_stage + static_cast<size_t>(m) * row, csrc,
                        row);
            rb.h_pf_scatter_ptrs[m] = cache_slot_addr(rank, l, s);
            rb.h_pf_scatter_idx[m] = m;
            seq_map[pos] = s;
            lc.slot_pos[static_cast<size_t>(s)] = pos;
            lc.slot_seq[static_cast<size_t>(s)] = ctx_seq_;
            lc.slot_use[static_cast<size_t>(s)] = tick_;
            lc.slot_prefetched[static_cast<size_t>(s)] = 1;
            ++m;
        }
        if (seq_map.empty()) lc.map.erase(ctx_seq_);  // no phantom seq entry
        if (m > m_before) touched.push_back(l);
    }
    if (m == 0) return;

    // Order the prefetch cache writes after all previously-enqueued
    // attention-stream work (earlier gathers may still read slots being
    // reused), then ONE burst + scatter on the dedicated tiering H2D stream —
    // overlapping this layer's attention/MoE, never the successor's stream
    // wait (TD-KVT-H2D-CONTENTION).
    be->record_event(rb.ev_pf_order, stream);
    be->stream_wait_event(rb.h2d_stream, rb.ev_pf_order);
    be->memcpy_h2d_async(rb.pf_incoming, rb.h_pf_stage,
                         static_cast<size_t>(m) * row, rb.h2d_stream);
    be->memcpy_h2d_async(rb.dev_pf_scatter_ptrs, rb.h_pf_scatter_ptrs,
                         static_cast<size_t>(m) * sizeof(void*),
                         rb.h2d_stream);
    be->memcpy_h2d_async(rb.dev_pf_scatter_idx, rb.h_pf_scatter_idx,
                         static_cast<size_t>(m) * sizeof(int), rb.h2d_stream);
    compute::launch_kv_row_scatter(
        reinterpret_cast<void* const*>(rb.dev_pf_scatter_ptrs),
        rb.pf_incoming, opts_.stride_row,
        static_cast<const int*>(rb.dev_pf_scatter_idx), m,
        opts_.stride_row, rb.h2d_stream);
    be->record_event(rb.ev_pf, rb.h2d_stream);
    rb.pf_inflight = true;
    for (const int l : touched)
        pf_pending_[static_cast<size_t>(rank)][static_cast<size_t>(l)] = 1;
    stats_.prefetch_rows += static_cast<uint64_t>(m);
    stats_.prefetch_bytes += static_cast<uint64_t>(m) * row;
    ++stats_.prefetch_bursts;
}

// ── Hook: materialize ───────────────────────────────────────────────────────

bool KvTieringManager::materialize(int rank, int layer_idx,
                                   const int* sparse_indices_dev,
                                   const int* topk_lengths_dev,
                                   int batch_size, void* stream,
                                   parallelism::TieredKvView* out) {
    if (batch_size != 1) {
        throw std::runtime_error(
            "KvTiering: materialize requires B==1 per step "
            "(TD-KVT-BATCH-COHORT)");
    }
    if (rank < 0 || rank >= static_cast<int>(ranks_.size())
        || layer_idx < 0 || layer_idx >= opts_.kv_layers
        || !sparse_indices_dev || !topk_lengths_dev || !out) {
        throw std::runtime_error("KvTiering: bad materialize arguments");
    }
    SeqState* ssp = find_seq(ctx_seq_);
    if (ctx_seq_ == 0 || !ssp) {
        throw std::runtime_error(
            "KvTiering: materialize without a step sequence (layer "
            + std::to_string(layer_idx) + ")");
    }
    auto& ls = ssp->pages[static_cast<size_t>(layer_idx)];
    // Fast path: no COLD page in this layer → the original full-residency
    // sparse path is still valid (byte-identical, zero tiering overhead) —
    // UNLESS a fresh (full-layer) selection was prepared and an IndexShare
    // successor has cold pages: then this call still runs the lookahead
    // prefetch on their behalf before returning false.
    bool any_cold = false;
    for (const auto& p : ls)
        if (p.state == PageState::kCold) { any_cold = true; break; }

    auto& sc = sel_[static_cast<size_t>(rank)];
    bool want_pf = sc.fresh && (sc.pending || sc.valid)
        && sc.prepared_layer == layer_idx
        && sc.seq == ctx_seq_ && sc.pos == ctx_pos_
        && share_succ_[static_cast<size_t>(layer_idx)] > 0;
    if (want_pf) {
        bool succ_cold = false;
        const int succ = share_succ_[static_cast<size_t>(layer_idx)];
        for (int l = layer_idx + 1;
             l <= layer_idx + succ && l < opts_.kv_layers; ++l) {
            if (!layer_dense_[static_cast<size_t>(l)]
                && state_layer_has_cold(*ssp, l)) {
                succ_cold = true;
                break;
            }
        }
        want_pf = succ_cold;
    }
    if (!any_cold && !want_pf) return false;

    const int* host_bt = ctx_host_bt_[static_cast<size_t>(rank)];
    if (layer_idx != ctx_layer_ || !host_bt) {
        throw std::runtime_error(
            "KvTiering: materialize without begin_layer context (layer "
            + std::to_string(layer_idx) + ")");
    }

    auto* be = backend(rank);
    be->set_device();
    auto& rb = ranks_[static_cast<size_t>(rank)];

    // 1) Selection → host: consume the prepare()-issued overlapped readback,
    //    reuse the IndexShare host copy, or fall back to the synchronous
    //    D2H + wait (TD-KVT-SYNC).
    const int n = ensure_selection(rank, layer_idx, sparse_indices_dev,
                                   topk_lengths_dev, stream);

    // 1b) IndexShare lookahead (TD-KVT-PREFETCH): the successors' selection
    //     IS this selection — stage their cold misses now so their bursts
    //     overlap this layer's attention/MoE.
    if (want_pf && n > 0) prefetch_successors(rank, layer_idx, n, stream);
    if (!any_cold) return false;  // successor-only call: this layer untiered
    if (n <= 0) return false;  // degenerate: sparse kernel reads no rows

    // Steps 1c-6 shared with the cohort path (TD-KVT-ADMISSION-UPFRONT).
    materialize_selection(rank, layer_idx, rb.h_indices, n,
                          topk_lengths_dev, stream, out);

    ++stats_.materializations;
    stats_.rows_gathered += static_cast<uint64_t>(n);
    if (stats_.materializations - last_logged_materializations_ >= 4096) {
        last_logged_materializations_ = stats_.materializations;
        log_stats();
    }
    return true;
}

// TD-KVT-ADMISSION-UPFRONT: steps 1c-6 of the original materialize over one
// row's host-resident selection — the shared placement body for the B==1
// path and materialize_row (INV-KVT-1: identical machinery either way).
void KvTieringManager::materialize_selection(
        int rank, int layer_idx, const int* h_idx, int n,
        const int* seqlens_dev, void* stream,
        parallelism::TieredKvView* out) {
    auto& rb = ranks_[static_cast<size_t>(rank)];
    // Ping-pong staging set (see MatSet): alternating sets keep the host
    // from serializing behind the PREVIOUS materialize's device-side
    // gather — critical for per-row cohort consumption (one materialize
    // per chunk row; the single-set guard measured ~37% of the tiered
    // prefill wall at rung-0).
    auto& ms = rb.mat[static_cast<size_t>(rb.mat_parity)];
    rb.mat_parity ^= 1;
    gather_selection(rank, layer_idx, ms, opts_.index_topk, h_idx, n,
                     /*cache_insert=*/true, stream);

    // 6) Fake paged view: identity indices over the dense scratch.
    out->kv_cache = ms.scratch;
    out->block_tables = static_cast<const int*>(rb.dev_fake_bt);
    out->seqlens_k = seqlens_dev;  // device int holding this row's n
    out->max_blocks_per_seq = n_fake_pages_;
    out->seq_len_kv = n;
    out->sparse_indices = static_cast<const int*>(rb.dev_ident_indices);
}

// Steps 1c-5 (classify / cold burst / cache scatter / gather / fence) over a
// host-resident selection into staging set `ms` (capacity `cap` rows).  The
// ONE placement body shared by the B==1 path, materialize_row, AND the
// batched union arm (TD-KVT-COHORT-BATCHED-MATERIALIZE) — INV-KVT-1:
// identical machinery regardless of the consuming kernel shape.
// `cache_insert` = false (union gathers): cold misses skip row-cache INSERTS
// (the union dedups intra-step; mass inserts would thrash the LRU) while
// cache hits still serve as sources.
void KvTieringManager::gather_selection(int rank, int layer_idx, MatSet& ms,
                                        int cap, const int* h_idx, int n,
                                        bool cache_insert, void* stream) {
    auto* be = backend(rank);
    auto& rb = ranks_[static_cast<size_t>(rank)];
    const int PS = opts_.page_size;
    const size_t row = static_cast<size_t>(opts_.stride_row);
    SeqState* ssp = find_seq(ctx_seq_);
    if (!ssp) {
        throw std::runtime_error(
            "KvTiering: materialize_selection without a step sequence");
    }
    if (n > cap) {
        throw std::runtime_error(
            "KvTiering: selection (" + std::to_string(n)
            + " rows) exceeds the staging capacity ("
            + std::to_string(cap) + ")");
    }
    auto& ls = ssp->pages[static_cast<size_t>(layer_idx)];
    const int* host_bt = ctx_host_bt_[static_cast<size_t>(rank)];

    // 1c) Host staging-reuse guard: THIS set's previous uploads/gather may
    //     still be DMA-reading h_src_ptrs/h_stage/scatter tables and
    //     cold_incoming (two materializes back — normally long done).
    if (ms.mat_inflight) {
        stats_.guard_wait_us += wait_event_us(be, ms.ev_mat_done);
        ms.mat_inflight = false;
    }

    // 2) Classify each selected row: paged pool (hot) / row cache / cold miss.
    ++tick_;
    auto& lc = cache_[static_cast<size_t>(rank)][static_cast<size_t>(layer_idx)];
    auto& seq_map = lc.map[ctx_seq_];  // (seq, position)-keyed entries
    const char* kv_base =
        static_cast<const char*>(opts_.kv_main_bases[static_cast<size_t>(rank)]);
    const char* incoming =
        static_cast<const char*>(ms.cold_incoming);

    int m = 0;                       // cold misses (packed staging rows)
    int k = 0;                       // cache-insert scatter entries
    // LS_KVT_VERIFY: authoritative pinned cold-pool pointer per cold-
    // classified row (nullptr = hot/pool row, no independent authority).
    std::vector<const char*> vf_auth;
    if (verify_) vf_auth.assign(static_cast<size_t>(n), nullptr);

    for (int i = 0; i < n; ++i) {
        const int pos = h_idx[i];
        if (pos < 0) {
            throw std::runtime_error(
                "KvTiering: negative index inside topk_length");
        }
        // pos is a global position (replicated KV) or a rank-LOCAL slot
        // (sharded KV, KVS-4 translated).  The rank's block table is keyed
        // by the SAME numbering as pos (jl); the manager's page state is
        // keyed by the GLOBAL logical page (j).  Row offsets within a page
        // are mode-invariant (chunk % page_size == 0).
        const int jl = pos / PS;
        const int j = global_page_of_index(rank, pos);
        const int rw = pos % PS;
        const bool cold = j < static_cast<int>(ls.size())
            && ls[static_cast<size_t>(j)].state == PageState::kCold;
        if (!cold) {
            // Paged pool (hot or D2H-inflight — content resident + immutable).
            ms.h_src_ptrs[i] = kv_base
                + static_cast<int64_t>(host_bt[jl]) * opts_.stride_block
                + static_cast<int64_t>(rw) * opts_.stride_row;
            ++stats_.pool_hits;
            continue;
        }
        if (auto it = seq_map.find(pos); it != seq_map.end()) {
            const int s = it->second;
            ms.h_src_ptrs[i] = cache_slot_addr(rank, layer_idx, s);
            lc.slot_use[s] = tick_;
            if (verify_) {
                const int vcr = cold_src_rank(rank, j);
                const int vslot =
                    ls[static_cast<size_t>(j)].cold_slot[
                        static_cast<size_t>(vcr)];
                if (vslot >= 0)
                    vf_auth[static_cast<size_t>(i)] =
                        ranks_[static_cast<size_t>(vcr)].cold_base
                        + static_cast<int64_t>(vslot) * opts_.stride_block
                        + static_cast<int64_t>(rw) * opts_.stride_row;
            }
            ++stats_.cache_hits;
            if (lc.slot_prefetched[static_cast<size_t>(s)]) {
                lc.slot_prefetched[static_cast<size_t>(s)] = 0;
                ++stats_.prefetch_hits;  // hit served by a lookahead prefetch
            }
            continue;
        }
        // Cold miss: CPU-gather the row into pinned staging (packed), source
        // the gather from the device cold_incoming buffer post-burst.  The
        // source pool is the rank's own copy, or the round-robin cold
        // owner's single copy under replicated dedup (host memcpy may cross
        // NUMA nodes; the burst H2D reads this rank's node-local staging —
        // INV-KVT-11).
        const auto& pst = ls[static_cast<size_t>(j)];
        const int cr = cold_src_rank(rank, j);
        const int cslot = pst.cold_slot[static_cast<size_t>(cr)];
        if (cslot < 0) {
            // Sharded KV: a translated local index only maps to pages this
            // rank owns, so its cold copy must be in THIS rank's pool;
            // under dedup the cold owner must hold the copy.
            throw std::runtime_error(
                "KvTiering: cold page without a rank-" + std::to_string(cr)
                + " cold slot (materialize, layer " + std::to_string(layer_idx)
                + ", logical " + std::to_string(j) + ")");
        }
        const char* csrc = ranks_[static_cast<size_t>(cr)].cold_base
            + static_cast<int64_t>(cslot) * opts_.stride_block
            + static_cast<int64_t>(rw) * opts_.stride_row;
        std::memcpy(ms.h_stage + static_cast<size_t>(m) * row, csrc, row);
        ms.h_src_ptrs[i] = incoming + static_cast<size_t>(m) * row;
        if (verify_) vf_auth[static_cast<size_t>(i)] = csrc;
        // Insert into the row cache for future steps (LRU, fair-share).
        // Union gathers skip inserts (cache_insert=false): the union dedups
        // repeats intra-step and a >hot_slots union would churn the LRU.
        if (cache_insert) {
            if (const int s = acquire_cache_slot(lc, tick_, ctx_seq_);
                s >= 0) {
                seq_map[pos] = s;
                lc.slot_pos[s] = pos;
                lc.slot_seq[static_cast<size_t>(s)] = ctx_seq_;
                lc.slot_use[s] = tick_;
                ms.h_scatter_ptrs[k] = cache_slot_addr(rank, layer_idx, s);
                ms.h_scatter_idx[k] = m;
                ++k;
            }
        }
        ++m;
        ++stats_.cold_misses;
    }
    if (seq_map.empty()) lc.map.erase(ctx_seq_);  // no phantom seq entry

    // 2b) A lookahead prefetch targeted THIS layer's row cache: order every
    //     attention-stream consumer/overwriter of the cache slots after the
    //     prefetch scatter (device-side wait — no host block).
    if (pf_pending_[static_cast<size_t>(rank)][static_cast<size_t>(layer_idx)]) {
        be->stream_wait_event(stream, rb.ev_pf);
        pf_pending_[static_cast<size_t>(rank)][static_cast<size_t>(layer_idx)]
            = 0;
    }

    // 3) Cold burst: ONE packed H2D on the dedicated tiering H2D stream
    //    (TD-KVT-H2D-CONTENTION); attention stream waits.
    if (m > 0) {
        be->memcpy_h2d_async(ms.cold_incoming, ms.h_stage,
                             static_cast<size_t>(m) * row, rb.h2d_stream);
        be->record_event(ms.ev_h2d, rb.h2d_stream);
        be->stream_wait_event(stream, ms.ev_h2d);
        stats_.cold_fetch_bytes += static_cast<uint64_t>(m) * row;
        ++stats_.h2d_bursts;
    }

    // 4) Upload the per-row source table + optional cache-insert scatter.
    be->memcpy_h2d_async(ms.dev_src_ptrs, ms.h_src_ptrs,
                         static_cast<size_t>(n) * sizeof(void*), stream);
    if (k > 0) {
        be->memcpy_h2d_async(ms.dev_scatter_ptrs, ms.h_scatter_ptrs,
                             static_cast<size_t>(k) * sizeof(void*), stream);
        be->memcpy_h2d_async(ms.dev_scatter_idx, ms.h_scatter_idx,
                             static_cast<size_t>(k) * sizeof(int), stream);
        compute::launch_kv_row_scatter(
            reinterpret_cast<void* const*>(ms.dev_scatter_ptrs),
            ms.cold_incoming, opts_.stride_row,
            static_cast<const int*>(ms.dev_scatter_idx), k,
            opts_.stride_row, stream);
    }

    // 5) Gather all n rows into the dense fake-paged scratch.
    compute::launch_kv_row_gather(
        ms.scratch, opts_.stride_block, opts_.stride_row, PS,
        reinterpret_cast<const void* const*>(ms.dev_src_ptrs), n,
        opts_.stride_row, stream);
    // Staging-reuse fence: the next materialize host-waits this before
    // touching h_stage/h_src_ptrs/scatter tables or re-bursting into
    // cold_incoming (gather done ⇒ its ev_h2d wait passed ⇒ burst done).
    be->record_event(ms.ev_mat_done, stream);
    ms.mat_inflight = true;

    // LS_KVT_VERIFY byte oracle (debug): scratch row i must equal its
    // source row, and every cold-classified row must equal the pinned
    // cold-pool authority (catches gather races, stale cache slots, wrong
    // staging).  Fail loud on the first mismatch.
    if (verify_) {
        // One D2H of the used scratch extent + per-row source D2H for every
        // COLD-classified row (cold-pool authority) and a 1/16 sample of
        // hot rows (gather integrity).
        be->synchronize_device();
        const int used_pages = (n + PS - 1) / PS;
        std::vector<char> sc_all(static_cast<size_t>(used_pages)
                                 * opts_.stride_block);
        be->memcpy_d2h_async(sc_all.data(), ms.scratch, sc_all.size(),
                             stream);
        be->synchronize_device();
        // Sampled source rows (cold rows always; 1/16 of hot rows): batch
        // the D2Hs, one sync, then host-compare.
        std::vector<int> vf_pick;
        for (int i = 0; i < n; ++i)
            if (vf_auth[static_cast<size_t>(i)] || (i & 15) == 0)
                vf_pick.push_back(i);
        std::vector<char> src_all(vf_pick.size() * row);
        for (size_t q = 0; q < vf_pick.size(); ++q)
            be->memcpy_d2h_async(src_all.data() + q * row,
                                 ms.h_src_ptrs[vf_pick[q]], row, stream);
        be->synchronize_device();
        for (size_t q = 0; q < vf_pick.size(); ++q) {
            const int i = vf_pick[q];
            const char* got = sc_all.data()
                + static_cast<int64_t>(i / PS) * opts_.stride_block
                + static_cast<int64_t>(i % PS) * opts_.stride_row;
            const char* src = src_all.data() + q * row;
            if (std::memcmp(got, src, row) != 0) {
                throw std::runtime_error(
                    "KvTiering VERIFY: gather mismatch (layer "
                    + std::to_string(layer_idx) + ", sel " + std::to_string(i)
                    + ", pos " + std::to_string(h_idx[i]) + ", rank "
                    + std::to_string(rank) + ")");
            }
            if (vf_auth[static_cast<size_t>(i)]
                && std::memcmp(src, vf_auth[static_cast<size_t>(i)],
                               row) != 0) {
                throw std::runtime_error(
                    "KvTiering VERIFY: cold source != cold-pool authority "
                    "(layer " + std::to_string(layer_idx) + ", sel "
                    + std::to_string(i) + ", pos " + std::to_string(h_idx[i])
                    + ", rank " + std::to_string(rank) + ")");
            }
        }
    }
}

// ── Hook: materialize_row (TD-KVT-ADMISSION-UPFRONT cohort seam) ────────────

void KvTieringManager::ensure_cohort_selection(int rank, int layer_idx,
                                               const int* sparse_indices_dev,
                                               const int* topk_lengths_dev,
                                               bool selection_fresh,
                                               void* stream) {
    auto* be = backend(rank);
    auto& rb = ranks_[static_cast<size_t>(rank)];
    auto& sc = sel_[static_cast<size_t>(rank)];
    const int ITK = opts_.index_topk;

    const bool identity = sc.valid && sc.seq == ctx_seq_ && sc.pos == ctx_pos_
        && sc.rows == ctx_rows_;
    if (identity && sc.prepared_layer == layer_idx)
        return;  // later row of the same (rank, layer) visit — host-resident
    if (identity && !selection_fresh) {
        // IndexShare SHARED layer: the producer reused the preceding full
        // layer's buffers (byte-identical content) and this manager holds
        // that step's cohort selection — INV-KVT-6 (certified identity +
        // matching step identity), no D2H.
        sc.prepared_layer = layer_idx;
        ++stats_.sync_reuses;
        return;
    }

    // Drain a stale pending single-row prepare() readback targeting
    // h_indices before overwriting (same buffer, same event).
    if (sc.pending) {
        stats_.guard_wait_us += wait_event_us(be, rb.ev_sync);
        sc.pending = false;
    }
    be->memcpy_d2h_async(rb.h_indices, sparse_indices_dev,
                         static_cast<size_t>(ctx_rows_) * ITK * sizeof(int),
                         stream);
    be->memcpy_d2h_async(rb.h_topk_len, topk_lengths_dev,
                         static_cast<size_t>(ctx_rows_) * sizeof(int),
                         stream);
    be->record_event(rb.ev_sync, stream);
    stats_.sync_wait_us += wait_event_us(be, rb.ev_sync);
    sc.pending = false;
    sc.valid = true;
    sc.reuse_ok = false;
    sc.fresh = false;  // cohorts never feed the lookahead prefetch
    sc.prepared_layer = layer_idx;
    sc.seq = ctx_seq_;
    sc.pos = ctx_pos_;
    sc.rows = ctx_rows_;
    sc.n = 0;
    ++stats_.cohort_readbacks;
}

bool KvTieringManager::cohort_layer_tiered(int layer_idx) {
    if (ctx_seq_ == 0 || ctx_rows_ <= 1 || layer_idx != ctx_layer_)
        return false;
    if (layer_idx < 0 || layer_idx >= opts_.kv_layers) return false;
    if (layer_dense_[static_cast<size_t>(layer_idx)]) return false;
    if (cohort_always_) return true;  // strict per-row arm (identity runs)
    if (!ctx_cold_valid_) {
        const SeqState* ssp = find_seq(ctx_seq_);
        if (!ssp) return false;
        ctx_cold_ = state_layer_has_cold(*ssp, layer_idx);
        ctx_cold_valid_ = true;
    }
    return ctx_cold_;
}

bool KvTieringManager::materialize_row(int rank, int layer_idx, int row,
                                       int rows,
                                       const int* sparse_indices_dev,
                                       const int* topk_lengths_dev,
                                       bool selection_fresh, void* stream,
                                       parallelism::TieredKvView* out) {
    if (opts_.cohort_rows_max <= 0 || rows < 1
        || rows > opts_.cohort_rows_max || row < 0 || row >= rows) {
        throw std::runtime_error(
            "KvTiering: bad materialize_row cohort shape (row "
            + std::to_string(row) + " of " + std::to_string(rows)
            + ", staging cap " + std::to_string(opts_.cohort_rows_max) + ")");
    }
    if (rank < 0 || rank >= static_cast<int>(ranks_.size())
        || layer_idx < 0 || layer_idx >= opts_.kv_layers
        || !sparse_indices_dev || !topk_lengths_dev || !out) {
        throw std::runtime_error("KvTiering: bad materialize_row arguments");
    }
    if (ctx_seq_ == 0 || layer_idx != ctx_layer_ || rows != ctx_rows_) {
        throw std::runtime_error(
            "KvTiering: materialize_row without a matching begin_layer "
            "cohort (layer " + std::to_string(layer_idx) + ")");
    }
    // Per-layer any-cold answer, computed once per begin_layer — page
    // states are per (seq, layer), shared by every rank and row.
    if (!ctx_cold_valid_) {
        const SeqState* ssp = find_seq(ctx_seq_);
        if (!ssp) {
            throw std::runtime_error(
                "KvTiering: materialize_row without a step sequence");
        }
        ctx_cold_ = state_layer_has_cold(*ssp, layer_idx);
        ctx_cold_valid_ = true;
    }
    if (!ctx_cold_) return false;  // full-residency per-row path valid
    if (!ctx_host_bt_[static_cast<size_t>(rank)]) {
        throw std::runtime_error(
            "KvTiering: materialize_row without begin_layer block tables "
            "(rank " + std::to_string(rank) + ")");
    }

    auto* be = backend(rank);
    be->set_device();
    ensure_cohort_selection(rank, layer_idx, sparse_indices_dev,
                            topk_lengths_dev, selection_fresh, stream);
    auto& rb = ranks_[static_cast<size_t>(rank)];
    const int n = rb.h_topk_len[row];
    if (n <= 0) return false;  // empty local selection (INV-KVS-EMPTY class)
    if (n > opts_.index_topk) {
        throw std::runtime_error("KvTiering: cohort topk_length "
                                 + std::to_string(n) + " > index_topk "
                                 + std::to_string(opts_.index_topk));
    }
    materialize_selection(
        rank, layer_idx,
        rb.h_indices + static_cast<size_t>(row) * opts_.index_topk, n,
        topk_lengths_dev + row, stream, out);

    ++stats_.materializations;
    stats_.rows_gathered += static_cast<uint64_t>(n);
    ++stats_.cohort_rows_tiered;
    if (stats_.materializations - last_logged_materializations_ >= 4096) {
        last_logged_materializations_ = stats_.materializations;
        log_stats();
    }
    return true;
}

// ── Hook: materialize_cohort (TD-KVT-COHORT-BATCHED-MATERIALIZE) ────────────
//
// The batched cold-layer consumer: ONE union gather + ONE batched sparse
// chunk kernel per (rank, layer) instead of `rows` per-row B==1
// sub-dispatches (measured ~745 us/(row·layer) — the RIPTIDE cohort tax).
// The union of the cohort's per-row selections is materialized through the
// SAME classify/burst/gather placement body as the per-row arm
// (gather_selection — INV-KVT-1), each row's indices are rewritten to union
// slots host-side in an ORDER-PRESERVING map (ascending unique positions ⇒
// deterministic slots; per-row accumulation order unchanged), per-row
// topk_lengths stay the producer's device buffer, and every row's seqlen is
// the union extent (the producer's selection is already causal —
// INV-SPARSE-CHUNK-CAUSAL — so the batched kernel's per-row causal bound
// never bites).  IndexShare shared layers reuse the union build + rewrite
// under the (seq, pos, rows) identity; the GATHER always reruns per layer
// (per-layer KV bytes and cold sets).  False (caller falls back to the
// per-row loop, always correct) when: the arm is disabled
// (LS_KVT_COHORT_ROWWISE=1 / union_rows_max 0), the layer has no cold
// pages, every row's selection is empty, or the union exceeds the staging
// capacity (fail-safe).
bool KvTieringManager::materialize_cohort(int rank, int layer_idx, int rows,
                                          const int* sparse_indices_dev,
                                          const int* topk_lengths_dev,
                                          bool selection_fresh, void* stream,
                                          parallelism::TieredKvView* out) {
    if (union_cap_ <= 0) return false;  // arm disabled (or LS_KVT_COHORT_ROWWISE)
    if (opts_.cohort_rows_max <= 0 || rows < 2
        || rows > opts_.cohort_rows_max) {
        throw std::runtime_error(
            "KvTiering: bad materialize_cohort shape (rows "
            + std::to_string(rows) + ", staging cap "
            + std::to_string(opts_.cohort_rows_max) + ")");
    }
    if (rank < 0 || rank >= static_cast<int>(ranks_.size())
        || layer_idx < 0 || layer_idx >= opts_.kv_layers
        || !sparse_indices_dev || !topk_lengths_dev || !out) {
        throw std::runtime_error("KvTiering: bad materialize_cohort arguments");
    }
    if (ctx_seq_ == 0 || layer_idx != ctx_layer_ || rows != ctx_rows_) {
        throw std::runtime_error(
            "KvTiering: materialize_cohort without a matching begin_layer "
            "cohort (layer " + std::to_string(layer_idx) + ")");
    }
    if (!ctx_cold_valid_) {
        const SeqState* ssp = find_seq(ctx_seq_);
        if (!ssp) {
            throw std::runtime_error(
                "KvTiering: materialize_cohort without a step sequence");
        }
        ctx_cold_ = state_layer_has_cold(*ssp, layer_idx);
        ctx_cold_valid_ = true;
    }
    if (!ctx_cold_) return false;  // batched real-block-table path valid
    if (!ctx_host_bt_[static_cast<size_t>(rank)]) {
        throw std::runtime_error(
            "KvTiering: materialize_cohort without begin_layer block tables "
            "(rank " + std::to_string(rank) + ")");
    }

    auto* be = backend(rank);
    be->set_device();
    ensure_cohort_selection(rank, layer_idx, sparse_indices_dev,
                            topk_lengths_dev, selection_fresh, stream);
    auto& rb = ranks_[static_cast<size_t>(rank)];
    const int ITK = opts_.index_topk;

    // Union build + order-preserving rewrite (selection-only — reused across
    // IndexShare shared layers under the step identity, INV-KVT-6 extended).
    if (!(rb.u_valid && rb.u_seq == ctx_seq_ && rb.u_pos == ctx_pos_
          && rb.u_rows == rows)) {
        rb.u_valid = false;
        int max_pos = -1;
        for (int b = 0; b < rows; ++b) {
            const int len = rb.h_topk_len[b];
            if (len > ITK) {
                throw std::runtime_error(
                    "KvTiering: cohort topk_length " + std::to_string(len)
                    + " > index_topk " + std::to_string(ITK));
            }
            const int* src = rb.h_indices + static_cast<size_t>(b) * ITK;
            for (int i = 0; i < len; ++i) {
                const int p = src[i];
                if (p < 0) {
                    throw std::runtime_error(
                        "KvTiering: negative index inside topk_length");
                }
                if (p > max_pos) max_pos = p;
            }
        }
        if (max_pos < 0) return false;  // all rows empty (INV-KVS-EMPTY class)
        auto& slot_of = rb.u_slot_of;
        if (static_cast<int>(slot_of.size()) < max_pos + 1)
            slot_of.resize(static_cast<size_t>(max_pos) + 1);
        std::fill(slot_of.begin(),
                  slot_of.begin() + static_cast<size_t>(max_pos) + 1, -1);
        for (int b = 0; b < rows; ++b) {
            const int len = rb.h_topk_len[b];
            const int* src = rb.h_indices + static_cast<size_t>(b) * ITK;
            for (int i = 0; i < len; ++i)
                slot_of[static_cast<size_t>(src[i])] = 0;
        }
        auto& up = rb.u_pos_list;
        up.clear();
        for (int p = 0; p <= max_pos; ++p) {
            if (slot_of[static_cast<size_t>(p)] == 0) {
                slot_of[static_cast<size_t>(p)] =
                    static_cast<int>(up.size());
                up.push_back(p);
            }
        }
        const int u_n = static_cast<int>(up.size());
        if (u_n > union_cap_) return false;  // fail-safe: per-row fallback
        // Single-writer fence for the pinned rewrite buffers (INV-KVT-7):
        // the previous step's async uploads may still read h_uidx/h_useq.
        if (rb.uidx_inflight) {
            stats_.guard_wait_us += wait_event_us(be, rb.ev_uidx);
            rb.uidx_inflight = false;
        }
        for (int b = 0; b < rows; ++b) {
            const int len = rb.h_topk_len[b];
            const int* src = rb.h_indices + static_cast<size_t>(b) * ITK;
            int* dst = rb.h_uidx + static_cast<size_t>(b) * ITK;
            for (int i = 0; i < len; ++i)
                dst[i] = slot_of[static_cast<size_t>(src[i])];
            // Deterministic padding: the kernel reads only [0, len) per row,
            // but the whole [rows x ITK] buffer uploads — uninitialized
            // pinned padding would be geometry/boot-dependent garbage (the
            // per-row arm's analogue, the identity iota, is CONSTANT).
            // Zero-fill so masked lanes always see slot 0 (valid scratch).
            std::fill(dst + len, dst + ITK, 0);
            rb.h_useq[b] = u_n;
        }
        be->memcpy_h2d_async(rb.dev_uidx, rb.h_uidx,
                             static_cast<size_t>(rows) * ITK * sizeof(int),
                             stream);
        be->memcpy_h2d_async(rb.dev_useq, rb.h_useq,
                             static_cast<size_t>(rows) * sizeof(int), stream);
        be->record_event(rb.ev_uidx, stream);
        rb.uidx_inflight = true;
        rb.u_n = u_n;
        rb.u_seq = ctx_seq_;
        rb.u_pos = ctx_pos_;
        rb.u_rows = rows;
        rb.u_valid = true;
        ++stats_.cohort_union_rewrites;
    }
    if (rb.u_n <= 0) return false;

    // Per-layer union gather (classification against THIS layer's page
    // states; hot rows device-gathered, cold rows staged from the pinned
    // pool — placement-only, INV-KVT-1).
    gather_selection(rank, layer_idx, rb.umat, union_cap_,
                     rb.u_pos_list.data(), rb.u_n, /*cache_insert=*/false,
                     stream);

    // Union fake view: every row shares the identity fake pages; per-row
    // indices are the rewritten union slots; per-row seqlens all = u_n.
    out->kv_cache = rb.umat.scratch;
    out->block_tables = static_cast<const int*>(rb.dev_union_bt);
    out->seqlens_k = static_cast<const int*>(rb.dev_useq);
    out->max_blocks_per_seq = u_pages_cap_;
    out->seq_len_kv = rb.u_n;
    out->sparse_indices = static_cast<const int*>(rb.dev_uidx);

    ++stats_.materializations;
    ++stats_.cohort_unions;
    stats_.cohort_union_rows += static_cast<uint64_t>(rb.u_n);
    stats_.rows_gathered += static_cast<uint64_t>(rb.u_n);
    stats_.cohort_rows_tiered += static_cast<uint64_t>(rows);
    if (stats_.materializations - last_logged_materializations_ >= 4096) {
        last_logged_materializations_ = stats_.materializations;
        log_stats();
    }
    return true;
}

void KvTieringManager::on_dense_layer(int layer_idx) {
    if (ctx_seq_ == 0) return;
    // Dense attention reads the step sequence's FULL prefix through its
    // real block tables — legal only while none of ITS pages are cold.
    if (layer_has_cold(ctx_seq_, layer_idx)) {
        throw std::runtime_error(
            "KvTiering: DENSE attention on layer " + std::to_string(layer_idx)
            + " with cold (demoted) KV pages — dense staging would read "
              "stale/reused pages (INV-KVT-2 fail-loud)");
    }
    // Layer-invariant dense causes (e.g. missing indexer weights) fire from
    // step 1 — mark sticky BEFORE any demotion so this layer never demotes.
    if (layer_idx >= 0 && layer_idx < static_cast<int>(layer_dense_.size()))
        layer_dense_[static_cast<size_t>(layer_idx)] = 1;
}

void KvTieringManager::log_stats() const {
    const auto& s = stats_;
    const uint64_t total = s.pool_hits + s.cache_hits + s.cold_misses;
    const double hit_rate = total
        ? 100.0 * static_cast<double>(s.pool_hits + s.cache_hits)
              / static_cast<double>(total)
        : 0.0;
    const double avg_sync_us = s.materializations
        ? static_cast<double>(s.sync_wait_us)
              / static_cast<double>(s.materializations)
        : 0.0;
    spdlog::info(
        "KvTiering stats: materializations={} rows={} pool_hits={} "
        "cache_hits={} cold_misses={} (hot hit-rate {:.2f}%) "
        "cold_fetch={:.2f} MiB in {} bursts, demoted_pages={}, "
        "repromoted_pages={} ({:.2f} MiB), "
        "budget_skips={}, "
        "cache_evictions={} | sync: overlapped={} reuses={} fallbacks={} "
        "wait={} us ({:.2f} us/mat) guard={} us | prefetch: rows={} "
        "bursts={} {:.2f} MiB hits={} | cohort: readbacks={} rows_tiered={} "
        "unions={} union_rows={} rewrites={}",
        s.materializations, s.rows_gathered, s.pool_hits, s.cache_hits,
        s.cold_misses, hit_rate,
        static_cast<double>(s.cold_fetch_bytes) / (1024.0 * 1024.0),
        s.h2d_bursts, s.demoted_pages, s.repromoted_pages,
        static_cast<double>(s.repromote_bytes) / (1024.0 * 1024.0),
        s.budget_skips, s.cache_evictions,
        s.sync_overlapped, s.sync_reuses, s.sync_fallbacks, s.sync_wait_us,
        avg_sync_us, s.guard_wait_us, s.prefetch_rows, s.prefetch_bursts,
        static_cast<double>(s.prefetch_bytes) / (1024.0 * 1024.0),
        s.prefetch_hits, s.cohort_readbacks, s.cohort_rows_tiered,
        s.cohort_unions, s.cohort_union_rows, s.cohort_union_rewrites);
}

}  // namespace layerstorm::daemon
